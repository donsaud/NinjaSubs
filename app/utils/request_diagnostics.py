"""Request diagnostics and access-log redaction for configured add-on URLs.

A configured NinjaSubs URL carries the user's provider credentials in its first
path segment, base64-encoded. That is reversible, so it must never reach a log.

Two pieces here:

``ConfigRedactingFilter``
    A ``logging.Filter`` on ``uvicorn.access`` that rewrites the request target
    before the access line is formatted, replacing the configuration segment
    with ``{cfg}``. Without this, ``docker logs`` shows every configured request
    with the API key and password sitting in plain sight.

``RequestDiagnosticsMiddleware``
    Emits one sanitized line per incoming request saying what arrived and what
    the server made of it -- route shape, whether a configuration was present
    and decodable, and whether each credential field was populated. It reports
    booleans and lengths only. No value from the configuration is ever
    interpolated, logged, or returned, so this is safe at DEBUG level.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("uvicorn.error")

#: Fields whose presence is worth knowing, never whose value.
_CREDENTIAL_FIELDS = (
    "subdl_key",
    "subsource_key",
    "opensubtitles_key",
    "opensubtitles_username",
    "opensubtitles_password",
)


#: Route markers that always follow the configuration segment. The token is
#: standard base64 and can therefore contain "/", so it cannot be matched as a
#: single path segment; the route boundary is what delimits it.
_ROUTE_MARKERS = (
    "/manifest.json",
    "/manifest",
    "/subtitles/",
    "/sub/",
    "/configure",
    "/api/",
    "/health",
    "/static/",
    "/catalog",
    "/stream",
    "/meta",
)


def redact_path(path: str) -> str:
    """Replace a leading configuration segment with ``{cfg}``.

    The configuration is standard base64 and may itself contain "/", so the
    segment is delimited by the first known route marker rather than by a
    character class. Everything before that marker is replaced wholesale, so
    nothing of the token can survive even if the token is malformed.

    The query string is dropped too, since an add-on configured with query
    parameters carries the same values there.
    """
    if not path:
        return path
    head, sep, tail = path.partition("?")
    if not head.startswith("/"):
        return head + (sep + "<query redacted>" if sep and tail else "")
    positions = [head.find(m) for m in _ROUTE_MARKERS]
    positions = [p for p in positions if p > 0]
    boundary = min(positions) if positions else -1
    if boundary > 0:
        head = "/{cfg}" + head[boundary:]
    return head + (sep + "<query redacted>" if sep and tail else "")


def looks_like_config_segment(segment: str) -> bool:
    """True when the segment decodes to a JSON object, i.e. a configuration."""
    if len(segment) < 40:
        return False
    padded = segment + "=" * ((4 - len(segment) % 4) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded)
    except (binascii.Error, ValueError):
        return False
    return raw.lstrip().startswith(b"{")


def sanitized_request_id(request: Request) -> str:
    """A short, stable id for correlating log lines.

    Derived from a digest rather than the path itself, so it cannot be reversed
    into the configuration it came from.
    """
    material = f"{request.method}|{request.url.path}|{id(request)}".encode()
    return hashlib.sha256(material).hexdigest()[:12]


class ConfigRedactingFilter(logging.Filter):
    """Strip credential-bearing configuration from uvicorn access lines.

    uvicorn's access record carries ``(client_addr, method, full_path,
    http_version, status_code)``; the path is replaced in place, before the
    formatter runs, so nothing unredacted is ever written.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple):
            patched = list(args)
            for i, value in enumerate(patched):
                if isinstance(value, str) and value.startswith("/"):
                    patched[i] = redact_path(value)
            record.args = tuple(patched)
        # Some formatters read these attributes instead of args.
        for attr in ("request_line", "path", "full_path", "uri"):
            value = getattr(record, attr, None)
            if isinstance(value, str) and value.startswith("/"):
                setattr(record, attr, redact_path(value))
        return True


def install_access_log_redaction() -> ConfigRedactingFilter:
    """Attach the filter to the access logger. Safe to call more than once."""
    log_filter = ConfigRedactingFilter()
    access_logger = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, ConfigRedactingFilter) for f in access_logger.filters):
        access_logger.addFilter(log_filter)
    return log_filter


def describe_configuration(token: str) -> dict:
    """Booleans and lengths for a configuration token. Never the values."""
    summary: dict[str, object] = {"config_present": bool(token)}
    if not token or not looks_like_config_segment(token):
        summary["config_decoded"] = False
        return summary
    padded = token + "=" * ((4 - len(token) % 4) % 4)
    try:
        import json

        data = json.loads(base64.urlsafe_b64decode(padded).decode())
    except Exception:
        summary["config_decoded"] = False
        return summary
    if not isinstance(data, dict):
        summary["config_decoded"] = False
        return summary
    summary["config_decoded"] = True
    for field in _CREDENTIAL_FIELDS:
        value = data.get(field)
        summary[f"{field}_present"] = bool(value)
        summary[f"{field}_len"] = len(str(value)) if value else 0
    for flag in (
        "enable_subdl",
        "enable_subsource",
        "enable_opensubtitles",
        "enable_yifysubtitles",
        "enable_subtitlecat",
    ):
        if flag in data:
            summary[flag] = bool(data.get(flag))
    if "languages" in data:
        summary["languages_count"] = len(data.get("languages") or [])
    return summary


class RequestDiagnosticsMiddleware(BaseHTTPMiddleware):
    """One sanitized line per request describing configuration handling."""

    async def dispatch(self, request: Request, call_next):
        raw_path = request.url.path
        token = ""
        head = raw_path.lstrip("/").split("/", 1)[0]
        if looks_like_config_segment(head):
            token = head

        route = "configured" if token else "unconfigured"
        request_id = sanitized_request_id(request)
        # Attach so other layers can correlate without re-deriving it.
        request.state.diagnostics_id = request_id

        if token:
            summary = describe_configuration(token)
            logger.info(
                "[diag] req=%s route=%s status=arriving config_present=1 "
                "config_decoded=%s os_key_present=%s os_user_present=%s "
                "os_pass_present=%s enable_os=%s",
                request_id,
                route,
                summary.get("config_decoded"),
                summary.get("opensubtitles_key_present"),
                summary.get("opensubtitles_username_present"),
                summary.get("opensubtitles_password_present"),
                summary.get("enable_opensubtitles", "(default)"),
            )
        else:
            logger.info(
                "[diag] req=%s route=%s status=arriving config_present=0",
                request_id,
                route,
            )

        response: Response = await call_next(request)

        logger.info(
            "[diag] req=%s route=%s status=%s served",
            request_id,
            route,
            response.status_code,
        )
        return response
