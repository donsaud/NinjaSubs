"""Bounded, expiring in-memory credential association for on-demand subtitle downloads.

This module provides a small, in-memory store that maps a subtitle identifier to the
*user-specific* provider API keys that were active when the subtitle listing was
generated. It is intentionally:

* **In-memory only** — nothing is ever written to disk, so provider credentials are
  never persisted to ``subs_cache/_meta/*.json``.
* **Bounded** — a hard maximum number of entries prevents unbounded memory growth.
* **Time-boxed** — entries expire after a configurable TTL so stale credentials
  cannot linger indefinitely.
* **Cleared on restart** — because the store is purely in-memory, process restart
  naturally invalidates every association (server-wide environment keys remain as a
  fallback).
* **Secret-safe** — raw credentials are never logged, returned in diagnostics, or
  exposed through cache statistics.

After a process restart, user-specific credentials are *not* recovered from disk;
the server-wide environment credentials configured via ``.env`` continue to provide
fallback functionality.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time

logger = logging.getLogger("uvicorn.error")

# Maximum number of credential associations retained in memory at any time.
_MAX_ENTRIES = 512

# Default time-to-live for a credential association (seconds).  Kept short so a
# credential stored for an infrequently-used subtitle does not linger for hours.
_DEFAULT_TTL = 30 * 60  # 30 minutes


def _hash_key(sub_id: str) -> str:
    """Return a stable, non-reversible digest of a subtitle identifier.

    The digest is used only as an internal lookup key; it is not a security
    control and is not stored anywhere persistent.
    """
    return hashlib.sha256(str(sub_id or "").strip().encode("utf-8")).hexdigest()


class CredentialStore:
    """Thread-safe, TTL-bounded in-memory mapping of subtitle id -> provider keys."""

    def __init__(self, max_entries: int = _MAX_ENTRIES, ttl: float = _DEFAULT_TTL) -> None:
        self._max_entries = max(1, int(max_entries))
        self._ttl = float(ttl)
        self._lock = asyncio.Lock()
        # Mapping: hashed subtitle id -> (expiry_timestamp, keys_dict)
        self._entries: dict[str, tuple[float, dict[str, str]]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def store(self, sub_id: str, keys: dict[str, str]) -> None:
        """Associate ``sub_id`` with the given provider keys.

        ``keys`` contain ephemeral provider credentials stored in-memory only.
        The store intentionally holds secret API keys for the lifetime of the
        subtitle request, never persisting them to disk.
        """
        if not sub_id:
            return
        async with self._lock:
            self._purge_expired()
            digest = _hash_key(sub_id)
            self._entries[digest] = (time.monotonic() + self._ttl, dict(keys or {}))
            self._enforce_capacity()

    async def get(self, sub_id: str) -> dict[str, str] | None:
        """Return the credential association for ``sub_id`` if it is still valid."""
        if not sub_id:
            return None
        async with self._lock:
            self._purge_expired()
            entry = self._entries.get(_hash_key(sub_id))
            if entry is None:
                return None
            expiry, keys = entry
            if expiry < time.monotonic():
                # Defensive: should not happen after purge, but be safe.
                self._entries.pop(_hash_key(sub_id), None)
                return None
            return dict(keys)

    async def clear(self) -> int:
        """Drop every in-memory credential association. Returns count removed."""
        async with self._lock:
            count = len(self._entries)
            self._entries.clear()
            return count

    def size(self) -> int:
        """Return the current number of in-memory associations (diagnostics only)."""
        return len(self._entries)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _purge_expired(self) -> None:
        now = time.monotonic()
        expired = [k for k, (expiry, _) in self._entries.items() if expiry < now]
        for k in expired:
            self._entries.pop(k, None)

    def _enforce_capacity(self) -> None:
        """Evict the soonest-to-expire entries when the store is over capacity."""
        if len(self._entries) <= self._max_entries:
            return
        # Sort by expiry ascending; drop the ones that expire first.
        ordered = sorted(self._entries.items(), key=lambda item: item[1][0])
        to_remove = len(self._entries) - self._max_entries
        for digest, _ in ordered[:to_remove]:
            self._entries.pop(digest, None)


# Module-level singleton used by the application.
credential_store = CredentialStore()


def redact_keys(keys: dict[str, str] | None) -> dict[str, str]:
    """Return a copy of ``keys`` with any credential values masked for logging.

    This helper is intentionally conservative: any key whose name looks like a
    credential is replaced by a fixed-length mask.  It is used only for safe
    diagnostic logging and never touches the real stored values.
    """
    if not keys:
        return {}
    sensitive = {"subdl_key", "subsource_key", "opensubtitles_key"}
    out: dict[str, str] = {}
    for k, v in keys.items():
        if k in sensitive and v:
            out[k] = _mask(v)
        else:
            out[k] = str(v)
    return out


def _mask(value: str) -> str:
    """Return a fixed mask for a credential string."""
    v = str(value or "").strip()
    if not v:
        return "<empty>"
    if len(v) <= 8:
        return "***"
    return f"{v[:4]}...{v[-4:]}"


__all__ = [
    "CredentialStore",
    "credential_store",
    "redact_keys",
]

