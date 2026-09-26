"""Cache layer for synchronized subtitle results.

Uses Redis when ``REDIS_URL`` is configured and the ``redis`` package is
available, otherwise transparently falls back to an in-process TTL cache so the
service keeps working (and tests run) without Redis.

Payloads are stored under ``final_sub:*`` keys; a parallel metadata companion
(``set_meta``/``get_meta``) records decision provenance (status, applied
shift, reference fingerprint, decision kind, timestamp) under ``{key}:meta``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from cachetools import TTLCache

from app.config import settings

logger = logging.getLogger(__name__)

try:  # redis is optional at runtime
    import redis.asyncio as aioredis  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    aioredis = None  # type: ignore[assignment]

_DEFAULT_TTL = 86_400  # 24 hours
# Short TTL for "no reference could be resolved" verdicts, so client retries
# stop re-running the whole provider fan-out but a transient outage self-heals.
_NEGATIVE_TTL = 300  # 5 minutes


class SyncCache:
    """24h TTL cache for synced/Arabic-normalized subtitle payloads."""

    def __init__(self, ttl: int = _DEFAULT_TTL, maxsize: int = 1024) -> None:
        self.ttl = ttl
        self._local: TTLCache[str, bytes] = TTLCache(maxsize=maxsize, ttl=ttl)
        self._local_meta: TTLCache[str, str] = TTLCache(maxsize=maxsize, ttl=ttl)
        self._local_fail: TTLCache[str, str] = TTLCache(maxsize=maxsize, ttl=_NEGATIVE_TTL)
        self._redis = None

    @staticmethod
    def _meta_key(key: str) -> str:
        return f"{key}:meta"

    @staticmethod
    def build_key(
        imdb_id: str,
        season_ep: str,
        fingerprint: str,
        arabic_sub_id: str,
        decision: str | None = None,
        content_hash: str | None = None,
    ) -> str:
        """Cache key: ``final_sub:{imdb}:{season_ep}:{fp}:{sub_id}[:{decision}][:{hash}]``.

        The fingerprint (video hash when known, else the normalized playback
        filename) plus the decision kind and content hash bind the key to the
        exact media, edition verdict, and payload, so one edition's sync can
        never serve another.
        """
        key = f"final_sub:{imdb_id}:{season_ep}:{fingerprint}:{arabic_sub_id}"
        if decision:
            key = f"{key}:{decision}"
        if content_hash:
            key = f"{key}:{content_hash}"
        return key

    async def connect(self) -> None:
        url = (getattr(settings, "REDIS_URL", None) or "").strip()
        if not url or aioredis is None:
            return
        try:
            client = aioredis.from_url(url)
            await client.ping()
            self._redis = client
            logger.info("[sync-cache] connected to Redis")
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("[sync-cache] Redis unavailable (%s); using in-process TTL", exc)
            self._redis = None

    async def get(self, key: str) -> bytes | None:
        if self._redis is not None:
            try:
                value = await self._redis.get(key)
                if value:
                    return value if isinstance(value, bytes) else str(value).encode()
            except Exception as exc:  # pragma: no cover - environment dependent
                logger.debug("[sync-cache] redis get failed: %s", exc)
        return self._local.get(key)

    async def set(self, key: str, value: bytes) -> None:
        if self._redis is not None:
            try:
                await self._redis.set(key, value, ex=self.ttl)
                return
            except Exception as exc:  # pragma: no cover - environment dependent
                logger.debug("[sync-cache] redis set failed: %s", exc)
        self._local[key] = value

    async def get_meta(self, key: str) -> dict[str, Any] | None:
        """Return the provenance companion for a cached payload, if recorded."""
        meta_key = self._meta_key(key)
        raw: str | None = None
        if self._redis is not None:
            try:
                value = await self._redis.get(meta_key)
                if value:
                    raw = value if isinstance(value, str) else bytes(value).decode()
            except Exception as exc:  # pragma: no cover - environment dependent
                logger.debug("[sync-cache] redis meta get failed: %s", exc)
        if raw is None:
            raw = self._local_meta.get(meta_key)
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    async def set_meta(self, key: str, meta: dict[str, Any]) -> None:
        """Record a provenance companion (JSON) for a cached payload."""
        try:
            raw = json.dumps(meta)
        except (TypeError, ValueError) as exc:
            logger.debug("[sync-cache] meta not serializable, skipping: %s", exc)
            return
        meta_key = self._meta_key(key)
        if self._redis is not None:
            try:
                await self._redis.set(meta_key, raw, ex=self.ttl)
                return
            except Exception as exc:  # pragma: no cover - environment dependent
                logger.debug("[sync-cache] redis meta set failed: %s", exc)
        self._local_meta[meta_key] = raw

    @staticmethod
    def _fail_key(key: str) -> str:
        return f"{key}:fail"

    async def mark_failed(self, key: str, ttl: int = _NEGATIVE_TTL) -> None:
        """Negatively cache a failed/mismatched resolution to damp retry storms."""
        fail_key = self._fail_key(key)
        if self._redis is not None:
            try:
                await self._redis.set(fail_key, "1", ex=ttl)
                return
            except Exception as exc:  # pragma: no cover - environment dependent
                logger.debug("[sync-cache] redis mark_failed failed: %s", exc)
        self._local_fail[fail_key] = "1"

    async def is_failed(self, key: str) -> bool:
        """True when this request recently resolved to no usable reference."""
        fail_key = self._fail_key(key)
        if self._redis is not None:
            try:
                if await self._redis.get(fail_key):
                    return True
            except Exception as exc:  # pragma: no cover - environment dependent
                logger.debug("[sync-cache] redis is_failed failed: %s", exc)
        return fail_key in self._local_fail

    async def clear_failed(self, key: str) -> None:
        """Drop a negative verdict (e.g. after a later successful sync)."""
        fail_key = self._fail_key(key)
        if self._redis is not None:
            try:
                await self._redis.delete(fail_key)
            except Exception as exc:  # pragma: no cover - environment dependent
                logger.debug("[sync-cache] redis clear_failed failed: %s", exc)
        self._local_fail.pop(fail_key, None)

    async def close(self) -> None:
        if self._redis is not None:
            try:
                await self._redis.aclose()
            except Exception:  # pragma: no cover
                pass
            self._redis = None
