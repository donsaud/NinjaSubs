"""Cache layer for synchronized subtitle results.

Uses Redis when ``REDIS_URL`` is configured and the ``redis`` package is
available, otherwise transparently falls back to an in-process TTL cache so the
service keeps working (and tests run) without Redis.

Payloads are stored under ``final_sub:*`` keys; a parallel metadata companion
(``set_meta``/``get_meta``) records decision provenance (status, applied
shift, reference fingerprint, decision kind, timestamp) under ``{key}:meta``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
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
# Bump when the verification/ordering logic changes so stale verdicts written by
# an older engine are never reused. Mirrors ReferenceDiskCache's engine_version.
SYNC_VERDICT_ENGINE_VERSION = 1
# Verdict TTL. Shorter than the payload TTL: a positive synchronization claim
# should be re-derived reasonably often rather than trusted for a full day.
_VERDICT_TTL = 7 * 86_400  # 7 days


class SyncCache:
    """24h TTL cache for synced/Arabic-normalized subtitle payloads."""

    def __init__(self, ttl: int = _DEFAULT_TTL, maxsize: int = 1024) -> None:
        self.ttl = ttl
        self._local: TTLCache[str, bytes] = TTLCache(maxsize=maxsize, ttl=ttl)
        self._local_meta: TTLCache[str, str] = TTLCache(maxsize=maxsize, ttl=ttl)
        self._local_fail: TTLCache[str, str] = TTLCache(maxsize=maxsize, ttl=_NEGATIVE_TTL)
        # Cached *measured* synchronization verdicts, keyed by video fingerprint
        # + subtitle hash + language + engine version. Distinct from _local_meta,
        # which records provenance for an already-cached payload.
        self._local_verdict: TTLCache[str, str] = TTLCache(maxsize=maxsize, ttl=_VERDICT_TTL)
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

    async def find_synced_for(self, imdb_id: str, sub_id: str) -> bytes | None:
        """Return a synced artifact for this subtitle, if one was cached.

        Matches ``final_sub:{imdb}:...:{sub_id}:...`` keys so the listing path
        can surface previously-synced subtitles. Best-effort: scans the
        in-process cache and, when connected, Redis.
        """
        imdb = (imdb_id or "").strip()
        sid = (sub_id or "").strip()
        if not imdb or not sid:
            return None
        prefix = f"final_sub:{imdb}:"
        needle = f":{sid}:"
        for key in list(self._local.keys()):
            if key.startswith(prefix) and needle in key:
                value = self._local.get(key)
                if value:
                    return value
        if self._redis is not None:
            try:
                cursor: int = 0
                while True:
                    cursor, keys = await self._redis.scan(cursor, match=f"{prefix}*", count=200)
                    for key in keys:
                        text = key.decode() if isinstance(key, bytes) else str(key)
                        if needle in text:
                            value = await self._redis.get(text)
                            if value:
                                return value if isinstance(value, bytes) else str(value).encode()
                    if not cursor:
                        break
            except Exception as exc:  # pragma: no cover - environment dependent
                logger.debug("[sync-cache] find_synced_for scan failed: %s", exc)
        return None

    # ------------------------------------------------------------------ #
    # Measured synchronization verdicts
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_verdict_key(
        video_fingerprint: str,
        subtitle_hash: str,
        language: str,
        engine_version: int = SYNC_VERDICT_ENGINE_VERSION,
    ) -> str:
        """Key a measured sync verdict: ``verdict:{engine}:{video}:{sub}:{lang}``.

        Binding the video fingerprint is what makes reuse safe. Without it, a
        verdict measured for one video would be handed to a different video that
        happens to carry the same subtitle, which is precisely the false-positive
        class this cache must not create.
        """
        return (
            f"verdict:{engine_version}:{video_fingerprint}:{subtitle_hash}:"
            f"{(language or 'und').lower()}"
        )

    @staticmethod
    def video_fingerprint_from_meta(meta: dict[str, Any]) -> str | None:
        """Derive a stable video identity from request metadata, or ``None``.

        Uses the strongest available signals in order. A metadata object with
        only title/year/season (a catalogue request with no resolved stream)
        yields ``None`` so no verdict is ever keyed to a guessed video.
        """
        from app.services.sync.matching import has_video_fingerprint

        if not has_video_fingerprint(meta):
            return None
        parts = [
            str(meta.get("imdb_id") or "").strip().lower(),
            str(meta.get("season") if meta.get("season") is not None else "movie"),
            str(meta.get("episode") if meta.get("episode") is not None else "x"),
            # Normalized so a case or spacing difference does not split the key.
            re.sub(r"\s+", " ", str(meta.get("target_filename") or "")).strip().lower(),
            str(meta.get("video_hash") or "").strip().lower(),
            str(meta.get("video_size") or ""),
        ]
        joined = "|".join(parts)
        # An empty identity is indistinguishable from "unknown"; refuse it.
        if not joined.strip("|x "):
            return None
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:24]

    async def get_verdict(self, key: str) -> dict[str, Any] | None:
        """Return a previously measured verdict, or ``None`` on a miss.

        Only verdicts that were actually measured are stored, so a hit is
        reported with ``verification`` forced to ``cached`` on the way out: a
        recalled result is never presented as a fresh measurement.
        """
        raw: str | None = None
        if self._redis is not None:
            try:
                value = await self._redis.get(key)
                if value:
                    raw = value if isinstance(value, str) else bytes(value).decode()
            except Exception as exc:  # pragma: no cover - environment dependent
                logger.debug("[sync-cache] redis get_verdict failed: %s", exc)
        if raw is None:
            raw = self._local_verdict.get(key)
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None
        if int(data.get("engine_version") or 0) != SYNC_VERDICT_ENGINE_VERSION:
            return None
        data["verification"] = "cached"
        return data

    @staticmethod
    def build_verdict_alias_key(
        video_fingerprint: str,
        subtitle_ref: str,
        engine_version: int = SYNC_VERDICT_ENGINE_VERSION,
    ) -> str:
        """Secondary index: look a verdict up by a stable candidate id.

        The primary key is content-addressed, but at search time the subtitle
        bytes have not been fetched, so the content hash is unknown. This index
        lets the listing path find a previously measured verdict by candidate id
        while still binding it to the video fingerprint, so a verdict is never
        surfaced for a different video.
        """
        return f"verdict-alias:{engine_version}:{video_fingerprint}:{subtitle_ref}"

    async def get_verdict_by_ref(
        self, video_fingerprint: str, subtitle_ref: str
    ) -> dict[str, Any] | None:
        """Search-time lookup by (video fingerprint, candidate id)."""
        key = self.build_verdict_alias_key(video_fingerprint, subtitle_ref)
        return await self.get_verdict(key)

    async def set_verdict(
        self, key: str, verdict: dict[str, Any], *, alias_key: str | None = None
    ) -> None:
        """Persist a measured verdict for later reuse.

        A verdict is only accepted when the analyzer actually measured
        something. Process success is not a verdict: ``alass_successful`` alone
        with no measured state is refused, so a successful run that produced an
        untrustworthy alignment can never be recalled as a positive result.
        """
        from app.services.sync.alignment import SyncState, VerificationAvailability

        state = verdict.get("sync_state")
        measured = verdict.get("verification")
        if measured == VerificationAvailability.UNKNOWN.value:
            logger.debug("[sync-cache] refusing to cache a verdict with no measurement")
            return
        if state in (SyncState.VERIFIED_SYNCED.value, SyncState.VERIFIED_RESYNCED.value):
            if measured not in (
                VerificationAvailability.VERIFIED.value,
                VerificationAvailability.CACHED.value,
            ):
                logger.debug(
                    "[sync-cache] refusing to cache %s backed by unmeasured %s", state, measured
                )
                return
        try:
            payload = dict(verdict)
            payload["engine_version"] = SYNC_VERDICT_ENGINE_VERSION
            raw = json.dumps(payload)
        except (TypeError, ValueError) as exc:
            logger.debug("[sync-cache] verdict not serializable, skipping: %s", exc)
            return
        if self._redis is not None:
            try:
                await self._redis.set(key, raw, ex=_VERDICT_TTL)
                if alias_key:
                    await self._redis.set(alias_key, raw, ex=_VERDICT_TTL)
                return
            except Exception as exc:  # pragma: no cover - environment dependent
                logger.debug("[sync-cache] redis set_verdict failed: %s", exc)
        self._local_verdict[key] = raw
        if alias_key:
            self._local_verdict[alias_key] = raw

    async def close(self) -> None:
        if self._redis is not None:
            try:
                await self._redis.aclose()
            except Exception:  # pragma: no cover
                pass
            self._redis = None
