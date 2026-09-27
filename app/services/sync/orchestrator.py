"""Sync orchestrator: routes each request through candidate strategies.

Flow per request: gates (server flag, user preference, Arabic only) →
content-bound sync-cache lookup → hash-exact strategy → external exact-match
strategy → alass alignment → cache the result. Any step may fail; on any
failure the original, unmodified subtitle bytes are returned so the player
always receives a usable subtitle (strict fail-safe fallback).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

from app.config import settings
from app.services.sync.matching import (
    _edition_tags,
    _release_group,
    _source_kind,
    _sources_compatible,
    is_informative_release_name,
)
from app.services.sync.query import ReferenceQuery, ResolvedReference
from app.services.sync_cache import SyncCache

logger = logging.getLogger(__name__)


def _normalize_fingerprint(value: Any) -> str:
    """Lowercase, whitespace-collapsed fingerprint for cache-key scoping."""
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def build_synced_cache_key(
    meta: dict,
    target_id: str,
    content_hash: str | None = None,
    decision: str | None = None,
) -> str:
    """Build the 24h sync cache key, scoped strictly to media + decision.

    ``final_sub:{imdb}:{season_ep}:{videohash_or_normalized_file}:{sub_id}:{decision}[:{hash}]``.
    The decision segment keeps team/hash/embedded syncs (same edition, same
    timing) isolated from generic edition syncs of the same bytes.
    """
    imdb_id = str(meta.get("imdb_id") or "").strip() or "unknown"
    season = meta.get("season")
    episode = meta.get("episode")
    if season is not None and episode is not None:
        season_ep = f"s{season}e{episode}"
    else:
        season_ep = "movie"
    fingerprint = _normalize_fingerprint(meta.get("video_hash"))
    if meta.get("stream_url"):
        # URLs are case-sensitive and can expose credentials: use a digest.
        stream_digest = hashlib.sha256(str(meta["stream_url"]).encode()).hexdigest()[:16]
        fingerprint = f"{fingerprint}:stream-{stream_digest}"
    if not fingerprint:
        fingerprint = _normalize_fingerprint(meta.get("target_filename")) or _normalize_fingerprint(
            target_id
        )
    return SyncCache.build_key(imdb_id, season_ep, fingerprint, target_id, decision, content_hash)


def _consume_task_exception(task: asyncio.Future) -> None:
    """Retrieve a detached task's result so its exception is not "never retrieved"."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.debug("[sync] background task finished with error: %s", exc)


def _parse_year(value: Any) -> int | None:
    try:
        return int(str(value).strip()) if str(value or "").strip() else None
    except (TypeError, ValueError):
        return None


class SyncOrchestrator:
    """Run the sync pipeline: strategies first, alass last, original on failure."""

    def __init__(
        self,
        *,
        hash_strategy: Any | None = None,
        embedded_strategy: Any | None = None,
        external_strategy: Any | None = None,
        aiostreams: Any | None = None,
        sync_service: Any | None = None,
        sync_cache: Any | None = None,
    ) -> None:
        self._hash_strategy = hash_strategy
        self._embedded_strategy = embedded_strategy
        self._external_strategy = external_strategy
        self._aiostreams = aiostreams
        self._sync_service = sync_service
        self._sync_cache = sync_cache
        self._inflight: dict[str, asyncio.Future] = {}
        self._inflight_lock = asyncio.Lock()
        # In-flight background embedded-track warm-ups (dedup by request key).
        self._warmups: set[str] = set()

    def _strategies(self) -> list[tuple[str, Any]]:
        # The embedded track is the ground-truth reference (the video's own
        # bytes), so it is tried first — the range-based extractor reads only a
        # couple of small windows. Fall back to the external exact-match tree,
        # then the stream MovieHash.
        return [
            ("embedded", self._embedded_strategy),
            ("external exact-match", self._external_strategy),
            ("hash-exact", self._hash_strategy),
        ]

    async def _maybe_resolve_stream_url(self, query: ReferenceQuery) -> None:
        """Ask the stream resolver for a probeable direct stream URL.

        Uses the user's configured stream-addon URL when supplied, else the
        ``AIOSTREAMS_URL`` fallback baked into the resolver. Skipped when the
        client already supplied a stream URL. Any failure (unreachable, timeout,
        no match) is swallowed so the pipeline falls through to the external
        tier.
        """
        if self._aiostreams is None or (query.stream_url or "").strip():
            return
        kwargs: dict[str, str] = {}
        user_base = (query.stream_addon_url or "").strip()
        if user_base:
            kwargs["base_url"] = user_base
        try:
            url = await self._aiostreams.resolve_stream_url(
                query.imdb_id,
                query.media_type,
                query.target_filename,
                season=query.season,
                episode=query.episode,
                **kwargs,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.info("[sync] stream addon resolution failed: %s", exc)
            return
        if url:
            query.stream_url = url
            logger.info("[sync] stream addon resolved a direct stream URL for probing")

    def _embedded_reference_from_disk(self, query: ReferenceQuery) -> ResolvedReference | None:
        """Return the on-disk internal-track reference if the warm-up produced one.

        Only an ``embedded`` entry counts here: ``team``/``edition``/``hash``
        entries are the normal strategy loop's job, not a signal to override the
        negative cache.
        """
        cache = getattr(self._embedded_strategy, "cache", None)
        if cache is None:
            return None
        try:
            cached = cache.get(query)
        except Exception as exc:  # pragma: no cover - defensive
            logger.info("[sync] embedded reference cache lookup failed: %s", exc)
            return None
        if cached is not None and cached.kind == "embedded" and cached.text:
            return cached
        return None

    def _schedule_embedded_warmup(
        self,
        query: ReferenceQuery,
        meta: dict,
        target_id: str,
        sub_bytes: bytes,
        resolved: ResolvedReference,
    ) -> None:
        """Detach a background internal-track extraction + cache, deduped."""
        warm = getattr(self._embedded_strategy, "warm_reference", None)
        if warm is None or not (query.stream_url or "").strip():
            return
        if resolved.text and resolved.kind == "embedded":
            return  # already using the internal track inline
        key = self._flight_key(meta, target_id, sub_bytes)
        if key in self._warmups:
            return
        self._warmups.add(key)
        task = asyncio.ensure_future(self._run_embedded_warmup(warm, query, key))
        task.add_done_callback(_consume_task_exception)

    async def _run_embedded_warmup(self, warm, query: ReferenceQuery, key: str) -> None:
        try:
            result = await warm(query)
            if result and result.text:
                logger.info("[sync] background internal-track reference is ready for %s", key)
        except Exception as exc:  # pragma: no cover - background best effort
            logger.info("[sync] background embedded warm-up failed: %s", exc)
        finally:
            self._warmups.discard(key)

    def _build_query(self, meta: dict) -> ReferenceQuery:
        return ReferenceQuery(
            imdb_id=str(meta.get("imdb_id") or ""),
            target_filename=meta.get("target_filename") or meta.get("release_name"),
            media_type=str(meta.get("media_type") or "movie"),
            title=meta.get("title"),
            year=_parse_year(meta.get("year")),
            video_hash=meta.get("video_hash"),
            video_size=meta.get("video_size"),
            stream_url=meta.get("stream_url"),
            stream_addon_url=meta.get("stream_addon_url"),
            season=meta.get("season"),
            episode=meta.get("episode"),
            api_keys={
                "subdl": meta.get("subdl_key") or "",
                "subsource": meta.get("subsource_key") or "",
                "opensubtitles": meta.get("opensubtitles_key") or "",
            },
            languages=("eng",),
        )

    async def evaluate_and_sync(
        self, sub_bytes: bytes, meta: dict, target_id: str, auto_sync: bool = False
    ) -> bytes:
        """Evaluate sync need and synchronize an Arabic subtitle, or return the original bytes."""
        server_enabled = bool(getattr(settings, "ENABLE_SUBTITLE_SYNC", False))
        lang = str(meta.get("lang") or "").lower()
        logger.info(
            "[sync] evaluate sub=%s: server_enabled=%s user_auto_sync=%s lang=%s imdb=%s S%sE%s file=%r",
            target_id,
            server_enabled,
            auto_sync,
            lang or "?",
            meta.get("imdb_id"),
            meta.get("season"),
            meta.get("episode"),
            meta.get("target_filename"),
        )
        if not server_enabled:
            logger.info("[sync] skipped: ENABLE_SUBTITLE_SYNC=false (server gate off)")
            return sub_bytes
        if not auto_sync:
            logger.info("[sync] skipped: user preference 'Auto-Sync Subtitles' is disabled")
            return sub_bytes
        if not lang.startswith("ar"):
            logger.info("[sync] skipped: subtitle language %r is not Arabic", lang)
            return sub_bytes

        # Pre-check: a subtitle already cut for the stream's release group
        # needs no alignment. The raw filenames must differ (at listing time
        # target_filename falls back to the subtitle's own release name, which
        # would otherwise trivially "match" and disable sync entirely).
        stream_file = meta.get("target_filename")
        subtitle_file = meta.get("release_name")
        if stream_file and subtitle_file and stream_file != subtitle_file:
            stream_group = _release_group(stream_file)
            subtitle_group = _release_group(subtitle_file)
            if (
                stream_group
                and subtitle_group
                and stream_group.lower() == subtitle_group.lower()
                and _edition_tags(stream_file) == _edition_tags(subtitle_file)
                and _sources_compatible(_source_kind(stream_file), _source_kind(subtitle_file))
            ):
                logger.info(
                    "[sync] target subtitle already matches release group (%s) "
                    "-> skipping alass, serving direct original",
                    stream_group,
                )
                return sub_bytes

        # Single-flight: an identical in-progress request (same media +
        # payload fingerprint) is joined instead of re-run, so concurrent
        # players never duplicate provider downloads or alass processes.
        flight_key = self._flight_key(meta, target_id, sub_bytes)
        async with self._inflight_lock:
            future = self._inflight.get(flight_key)
            if future is None:
                future = asyncio.ensure_future(
                    self._execute(sub_bytes, meta, target_id, auto_sync)
                )
                self._inflight[flight_key] = future
                future.add_done_callback(lambda done: self._finish_flight(flight_key, done))
            else:
                logger.info("[sync] coalescing onto in-flight request %s", flight_key)
        return await asyncio.shield(future)

    def _finish_flight(self, key: str, future: asyncio.Future) -> None:
        """The job owns its lifetime; cancelling a waiter must not unregister it."""
        if self._inflight.get(key) is future:
            del self._inflight[key]
        if not future.cancelled():
            # Retrieve detached failures even when all players disconnected.
            future.exception()

    async def close(self) -> None:
        """Cancel and drain outstanding provider work before the HTTP client closes."""
        tasks = list(self._inflight.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def _flight_key(meta: dict, target_id: str, sub_bytes: bytes) -> str:
        """Identity for request coalescing (finer than the sync-cache key)."""
        content_hash = hashlib.sha256(sub_bytes).hexdigest()[:16]
        key = build_synced_cache_key(meta, target_id, content_hash=content_hash)
        credentials = tuple(str(meta.get(name) or "") for name in (
            "subdl_key", "subsource_key", "opensubtitles_key"
        ))
        auth_digest = hashlib.sha256(repr(credentials).encode()).hexdigest()[:16]
        context_digest = hashlib.sha256(json.dumps([
            meta.get("media_type"), meta.get("target_filename"), meta.get("video_hash"),
            str(meta.get("video_size") or ""), meta.get("lang"),
        ], ensure_ascii=False).encode()).hexdigest()[:16]
        strict = bool(getattr(settings, "SYNC_REQUIRE_EXACT_MATCH", True))
        return f"{key}:v2:{strict}:{context_digest}:{auth_digest}"

    async def _execute(
        self, sub_bytes: bytes, meta: dict, target_id: str, auto_sync: bool
    ) -> bytes:
        """The expensive pipeline body: strategies, cache, alass (see evaluate_and_sync)."""
        content_hash = hashlib.sha256(sub_bytes).hexdigest()[:16]
        # Positive cache: serve an already-synced subtitle without any work.
        resolution_key = self._flight_key(meta, target_id, sub_bytes)
        if self._sync_cache is not None:
            cached = await self._sync_cache.get(resolution_key)
            if cached:
                logger.info(
                    "[sync] cache HIT for sub=%s -> serving pre-synced subtitle immediately",
                    target_id,
                )
                return cached

        query = self._build_query(meta)
        ready_embedded: ResolvedReference | None = None
        if self._sync_cache is not None and await self._sync_cache.is_failed(resolution_key):
            # The previous attempt aborted before the background embedded warm-up
            # finished. If the internal-track reference has since landed on disk,
            # the negative entry is stale: bust it and sync against the ready
            # reference instead of aborting immediately.
            ready_embedded = self._embedded_reference_from_disk(query)
            if ready_embedded is None:
                logger.info(
                    "[sync] negative cache hit for %s -> skipping provider fan-out",
                    resolution_key,
                )
                return sub_bytes
            logger.info(
                "[sync] negative cache overridden by ready embedded reference for %s",
                resolution_key,
            )
            await self._sync_cache.clear_failed(resolution_key)

        if ready_embedded is not None:
            resolved = ready_embedded
        else:
            await self._maybe_resolve_stream_url(query)
            resolved = ResolvedReference(None)
            for strategy_name, strategy in self._strategies():
                if strategy is None:
                    continue
                try:
                    resolved = await strategy.resolve_with_provenance(query)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("[sync] %s strategy failed: %s -> next", strategy_name, exc)
                    resolved = ResolvedReference(None)
                if resolved.text:
                    logger.info(
                        "[sync] %s strategy provided a reference (decision=%s)",
                        strategy_name,
                        resolved.kind,
                    )
                    break
            # Warm the internal-track reference in the background when the inline
            # tiers did not use it, so a later request can sync against the
            # video's own subtitle track without any player-visible latency.
            self._schedule_embedded_warmup(query, meta, target_id, sub_bytes, resolved)

        if not resolved.text:
            logger.info("[sync] no deterministic reference available -> aborting sync")
            if self._sync_cache is not None:
                await self._sync_cache.mark_failed(resolution_key)
            return sub_bytes
        reference = resolved.text
        decision_kind = resolved.kind
        logger.info("[sync] reference selected (%d bytes) -> invoking alass", len(reference.encode()))

        # Keep decision provenance alongside the result. Request-bound cache
        # lookup already happened before any provider calls above.
        key = build_synced_cache_key(
            meta, target_id, content_hash=content_hash, decision=decision_kind
        )
        # Decoded with the same Arabic-aware priority order as the download path;
        # decode_subtitle_bytes never raises, so no try/except is needed here.
        from app.extractor import decode_subtitle_bytes

        target_text = decode_subtitle_bytes(sub_bytes, lang=meta.get("lang"))

        if self._sync_service is None:  # pragma: no cover - defensive
            logger.warning("[sync] no sync service configured -> serving original subtitle")
            return sub_bytes
        # A best-effort edition sync of a target with no release info at all is
        # a relaxed fallback: tighten the pre-alass timeline gate because no
        # edition attribute was available to confirm the reference.
        relaxed = decision_kind == "edition" and not is_informative_release_name(
            meta.get("target_filename") or meta.get("release_name")
        )
        # Embedded references are sampled prefixes (first ~15 min), so their
        # end runtime is intentionally short: exempt them from duration gates.
        reference_partial = decision_kind == "embedded" or bool(
            getattr(resolved, "partial", False)
        )
        synced = await self._sync_service.sync_async(
            target_text,
            reference,
            decision_kind=decision_kind,
            is_series=query.is_series,
            source_confirmed=resolved.bluray_match,
            relaxed=relaxed,
            reference_partial=reference_partial,
        )
        if not synced:
            logger.warning("[sync] alass returned no output -> serving original subtitle")
            if self._sync_cache is not None:
                await self._sync_cache.mark_failed(resolution_key)
            return sub_bytes

        if self._sync_cache is not None:
            await self._sync_cache.clear_failed(resolution_key)
            await self._sync_cache.set(key, synced.encode("utf-8"))
            await self._sync_cache.set(resolution_key, synced.encode("utf-8"))
            await self._sync_cache.set_meta(
                key,
                {
                    "status": "synced",
                    "applied_shift": self._applied_shifts(target_text, synced),
                    "reference_sha": hashlib.sha256(reference.encode("utf-8")).hexdigest()[:16],
                    "decision": decision_kind,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
        logger.info("[sync] synchronized subtitle %s (%d bytes) and cached", target_id, len(synced))
        return synced.encode("utf-8")

    @staticmethod
    def _applied_shifts(target_text: str, synced: str) -> list[float]:
        """First-cue shifts applied, mirroring the service's own accounting."""
        from app.services.sync_service import _cue_starts_ms

        target_starts = _cue_starts_ms(target_text)
        synced_starts = _cue_starts_ms(synced)
        return [
            round((after - before) / 1000.0, 2)
            for before, after in zip(target_starts, synced_starts, strict=False)
        ][:3]
