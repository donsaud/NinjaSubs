"""Redact API keys and authorization tokens from log records.

Outbound HTTP logging (``httpx``) and provider loggers can otherwise leak
``api_key`` query parameters and ``Authorization``/``X-API-Key`` headers into
container logs. Installing :func:`install_log_redaction` wraps the global
``LogRecord`` factory so every emitted message is scrubbed before it is handled.
"""

from __future__ import annotations

import logging
import re

REDACTED = "[REDACTED]"

_QUERY_KEY_REGEX = re.compile(
    r"(?i)\b("
    r"api[_-]?key|apikey|"
    r"subdl_api_key|subsource_api_key|opensubtitles_api_key|"
    # Account passwords. Included because uvicorn's access log echoes the raw
    # request target, so any credential a client puts in a query string lands in
    # the container log verbatim. The configure page posts the password in a body
    # for exactly this reason, but a hand-crafted or third-party URL would still
    # reach here, and this filter is the backstop for that.
    r"password|passwd|pwd|"
    r"opensubtitles_password"
    r")=([^&\s'\"]+)"
)
_HEADER_REGEX = re.compile(
    r"(?i)((?:x-api-key|api-key|authorization)['\"]?\s*[:=]\s*['\"]?)(bearer\s+)?([^,;'\"}\s]+)"
)
_BEARER_REGEX = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]+")


def redact_secrets(text: str) -> str:
    """Return ``text`` with credentials replaced by ``[REDACTED]``.

    Covers API keys, account passwords, and authorization tokens.
    """
    if not text:
        return text
    text = _QUERY_KEY_REGEX.sub(lambda m: f"{m.group(1)}={REDACTED}", text)
    text = _HEADER_REGEX.sub(
        lambda m: f"{m.group(1)}{m.group(2) or ''}{REDACTED}",
        text,
    )
    text = _BEARER_REGEX.sub(f"Bearer {REDACTED}", text)
    return text


_installed = False


def install_log_redaction() -> None:
    """Install a global ``LogRecord`` factory that scrubs secrets from logs."""
    global _installed
    if _installed:
        return
    _installed = True

    previous_factory = logging.getLogRecordFactory()

    def _redacting_factory(*args: object, **kwargs: object) -> logging.LogRecord:
        record = previous_factory(*args, **kwargs)
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - defensive
            return record
        redacted = redact_secrets(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return record

    logging.setLogRecordFactory(_redacting_factory)
