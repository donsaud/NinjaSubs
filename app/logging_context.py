"""Request correlation for sync logging.

Sync work fans out to three providers concurrently, and several HTTP requests
for the same episode overlap. Without a correlation id, log lines from different
requests interleave and a line can be attributed to the wrong request. That is
not a cosmetic problem: during a production incident it produced an apparently
contradictory pair of diagnostics that took a full forensic pass to explain.

The id is attached by a logging filter rather than at every call site, so no
call site has to remember to include it and no decision path changes.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar, Token

_request_id: ContextVar[str | None] = ContextVar("sync_request_id", default=None)

#: Only sync-namespace loggers are tagged. Tagging everything would change
#: access logs and provider traces that other tooling already parses.
SYNC_LOGGER_PREFIXES = ("app.services.sync", "app.services.sync_service")


def stable_request_id(meta: dict, target_id: str | None, payload: bytes) -> str:
    """A short id that is stable for one request and distinct between requests.

    Derived from the stream identity, the subtitle identity and the payload
    digest, so two concurrent requests for the same episode are still told
    apart, and a retried identical request keeps the same id.
    """
    import hashlib

    seed = "|".join(
        str(part)
        for part in (
            meta.get("imdb_id"),
            meta.get("season"),
            meta.get("episode"),
            meta.get("filename"),
            meta.get("videosize"),
            target_id,
            hashlib.sha256(payload).hexdigest()[:12] if payload else "",
        )
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8]


def set_request_id(value: str | None) -> Token:
    return _request_id.set(value)


def reset_request_id(token: Token) -> None:
    _request_id.reset(token)


def current_request_id() -> str | None:
    return _request_id.get()


class RequestIdFilter(logging.Filter):
    """Append ``[req=xxxxxxxx]`` to sync records emitted during a request.

    Attached to the sync loggers, so the id appears without any call site
    changing. A record emitted outside a request is left untouched.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        request_id = _request_id.get()
        if not request_id:
            return True
        existing = getattr(record, "sync_request_id", None)
        if existing:
            return True
        record.sync_request_id = request_id  # type: ignore[attr-defined]
        return True


def format_with_request_id(fmt: str) -> str:
    """A formatter string that shows the request id when there is one.

    Appended to the existing format so no existing field changes and callers
    that do not opt in are unaffected.
    """
    return f"{fmt} [req=%(sync_request_id)s]"


def install_request_id_filtering() -> None:
    """Attach the filter to the sync logger tree. Idempotent."""
    log_filter = RequestIdFilter()
    for name in SYNC_LOGGER_PREFIXES:
        logger = logging.getLogger(name)
        if not any(isinstance(existing, RequestIdFilter) for existing in logger.filters):
            logger.addFilter(log_filter)
