"""Primary sync strategy: ground-truth reference via video-hash match.

When the player supplies a video hash, OpenSubtitles can return subtitles
uploaded for those exact bytes (``moviehash_match``). Same bytes means same
timing — no edition guessing, so a hash-confirmed English subtitle is used
verbatim as the alignment reference. Anything less than a confirmed hash
match aborts this strategy.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from app.services.sync.cache import ReferenceDiskCache
from app.services.sync.decode import decode_payload
from app.services.sync.query import ResolvedReference

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.services.sync.query import ReferenceQuery

logger = logging.getLogger(__name__)

_MIN_REFERENCE_BYTES = 5120

# A MovieHash match is a byte-exact 1:1 timeline match regardless of the
# subtitle's language, so accept any major release language as a timing
# reference (alass only needs the reference's cue timings).
_HASH_REFERENCE_LANGUAGES = ("en", "es", "fr", "de", "it")


class HashExactStrategy:
    """Hash-exact reference acquisition via OpenSubtitles moviehash lookup."""

    name = "opensubtitles-hash"

    def __init__(
        self,
        opensubtitles_provider=None,
        *,
        timeout: float = 6.0,
        stream_hash_timeout: float = 1.5,
        client=None,
        min_bytes: int = _MIN_REFERENCE_BYTES,
        cache: ReferenceDiskCache | None = None,
    ) -> None:
        self._provider = opensubtitles_provider
        self._client = client
        self.timeout = timeout
        self.stream_hash_timeout = stream_hash_timeout
        self.min_bytes = min_bytes
        self.cache = cache if cache is not None else ReferenceDiskCache(min_bytes=min_bytes)

    async def resolve(self, query: ReferenceQuery) -> str | None:
        """Return a hash-confirmed English reference or ``None``."""
        resolved = await self.resolve_with_provenance(query)
        return resolved.text

    async def resolve_with_provenance(
        self, query: ReferenceQuery
    ) -> ResolvedReference:
        """Resolve a reference, reporting the ``hash`` decision kind with it."""
        cached = self.cache.get(query)
        if cached is not None:
            return cached
        if self._provider is None:
            return ResolvedReference(None)

        video_hash = (query.video_hash or "").strip()
        video_size = query.video_size
        if not video_hash and query.stream_url and self._client is not None:
            # No client-supplied hash: compute the OpenSubtitles MovieHash from
            # the stream itself (two 64 KiB range requests, bounded timeout).
            from app.config import settings
            from app.services.sync.stream_hash import fetch_stream_moviehash

            computed = await fetch_stream_moviehash(
                query.stream_url,
                self._client,
                timeout=self.stream_hash_timeout,
                allow_private=bool(getattr(settings, "ALLOW_PRIVATE_STREAM_URLS", True)),
            )
            if computed is not None:
                video_hash, video_size = computed
        if not video_hash:
            logger.info("[reference] hash strategy skipped: no video hash (client or computed)")
            return ResolvedReference(None)
        try:
            text = await asyncio.wait_for(
                self._run(query, video_hash, video_size), self.timeout
            )
        except TimeoutError:
            logger.warning("[reference] hash strategy timed out after %.1fs", self.timeout)
            return ResolvedReference(None)
        if text is None:
            return ResolvedReference(None)
        return ResolvedReference(text, kind="hash")

    async def _run(self, query: ReferenceQuery, video_hash: str, video_size) -> str | None:
        api_key = query.api_keys.get("opensubtitles")
        try:
            releases = await self._provider.search_subtitles(
                imdb_id=query.imdb_id,
                is_series=query.is_series,
                season=query.season,
                episode=query.episode,
                title=query.title,
                year=query.year,
                api_key=api_key,
                languages=list(_HASH_REFERENCE_LANGUAGES),
                video_hash=video_hash,
                video_size=video_size,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[reference] hash search failed: %s", exc)
            return None

        # Any ``moviehash_match`` is a byte-exact timeline match; language is
        # irrelevant for a timing reference. Rank: non-HI first, English next,
        # then OpenSubtitles' own order.
        matches = [rel for rel in (releases or []) if bool(getattr(rel, "is_hash_match", False))]
        if not matches:
            logger.info("[reference] hash strategy: no hash-confirmed subtitle")
            return None
        matches.sort(
            key=lambda rel: (
                bool(getattr(rel, "hearing_impaired", False)),
                0 if str(getattr(rel, "lang", "") or "").lower().startswith("en") else 1,
            )
        )
        best = matches[0]
        logger.info(
            "[reference] hash strategy: confirmed %r (lang=%s) via opensubtitles",
            getattr(best, "release_name", "?"),
            getattr(best, "lang", "?"),
        )
        try:
            raw = await self._provider.download_archive(best.download_url, api_key=api_key)
        except Exception as exc:
            logger.warning("[reference] hash download failed: %s", exc)
            return None
        if not raw:
            logger.warning("[reference] hash strategy: empty download payload")
            return None

        decoded = decode_payload(
            raw,
            getattr(best, "release_name", "?"),
            query.season,
            query.episode,
            self.min_bytes,
        )
        if not decoded:
            logger.warning("[reference] hash strategy: payload did not decode")
            return None
        text = decoded.decode("utf-8", "replace")
        if len(text.encode("utf-8")) <= self.min_bytes:
            logger.warning("[reference] hash strategy: reference below size floor")
            return None

        from app.services.sync_service import sanitize_subtitle

        sanitized = sanitize_subtitle(text)
        if sanitized:
            self.cache.set(query, "opensubtitles", sanitized, kind="hash")
            return sanitized
        return text
