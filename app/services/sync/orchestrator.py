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
from app.services.subtitle_matcher import (
    ALIGNED_OFFSET_THRESHOLD_S,
    FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
    median_cue_offset,
    validate_cue_sanity,
)
from app.services.sync.matching import is_informative_release_name
from app.services.sync.query import ReferenceQuery, ResolvedReference
from app.services.sync_cache import SyncCache
from app.utils.cleaners import strip_intro_credits

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
    The decision segment keeps team/hash verdicts (same edition, same timing)
    isolated from generic edition syncs of the same bytes.
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
        external_strategy: Any | None = None,
        sync_service: Any | None = None,
        sync_cache: Any | None = None,
    ) -> None:
        self._hash_strategy = hash_strategy
        self._external_strategy = external_strategy
        self._sync_service = sync_service
        self._sync_cache = sync_cache
        self._inflight: dict[str, asyncio.Future] = {}
        self._inflight_lock = asyncio.Lock()

    def _strategies(self) -> list[tuple[str, Any]]:
        strats = []
        if self._external_strategy is not None:
            strats.append(("external exact-match", self._external_strategy))
        if self._hash_strategy is not None:
            strats.append(("hash-exact", self._hash_strategy))
        return strats

    def _build_query(self, meta: dict, target_id: str | None = None) -> ReferenceQuery:
        return ReferenceQuery(
            imdb_id=str(meta.get("imdb_id") or ""),
            target_filename=meta.get("target_filename") or meta.get("release_name"),
            target_sub_release_name=meta.get("release_name"),
            target_sub_id=target_id or meta.get("sub_id"),
            target_download_url=meta.get("download_url"),
            media_type=str(meta.get("media_type") or "movie"),
            title=meta.get("title"),
            year=_parse_year(meta.get("year")),
            video_hash=meta.get("video_hash"),
            video_size=meta.get("video_size"),
            stream_url=meta.get("stream_url"),
            season=meta.get("season"),
            episode=meta.get("episode"),
            api_keys={
                "subdl": meta.get("subdl_key") or "",
                "subsource": meta.get("subsource_key") or "",
                "opensubtitles": meta.get("opensubtitles_key") or "",
            },
            languages=("eng", "ara"),
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

        # No filename-only shortcut: a matching release group does not prove the
        # timings align (mislabeled uploads, unadjusted translator timings). Every
        # candidate is verified against the reference's actual cue content below.
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
        return f"{key}:v4:{strict}:{context_digest}:{auth_digest}"

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

        query = self._build_query(meta, target_id=target_id)
        if self._sync_cache is not None and await self._sync_cache.is_failed(resolution_key):
            logger.info(
                "[sync] negative cache hit for %s -> skipping provider fan-out",
                resolution_key,
            )
            return sub_bytes

        from app.extractor import decode_subtitle_bytes

        target_text = decode_subtitle_bytes(sub_bytes, lang=meta.get("lang"))

        if self._sync_service is None:  # pragma: no cover - defensive
            logger.warning("[sync] no sync service configured -> serving original subtitle")
            return sub_bytes

        # Reference strategies in priority order (external exact-match reference,
        # then OpenSubtitles MovieHash). The playing stream is never probed.
        attempts: list[tuple[str, Any]] = list(self._strategies())

        last_resolved = ResolvedReference(None)
        for strategy_name, strategy_obj in attempts:
            try:
                resolved = await strategy_obj.resolve_with_provenance(query)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("[sync] %s strategy failed: %s -> next", strategy_name, exc)
                resolved = ResolvedReference(None)

            if not resolved.text:
                continue

            last_resolved = resolved
            reference = resolved.text
            decision_kind = resolved.kind
            logger.info(
                "[sync] %s strategy provided a reference (decision=%s, %d bytes) -> invoking alass",
                strategy_name,
                decision_kind,
                len(reference.encode("utf-8")),
            )

            # Ground-truth content gate: filename metadata can be mislabeled, so
            # verify the candidate's first substantive dialogue against the
            # reference before trusting it, regardless of the release name.
            # Execution uses the wide ±20s window so alass can still fix
            # realistic uniform intro/bumper shifts between Web and BluRay
            # masters; only a larger delta (different cut/episode) skips.
            # (Ranking keeps the strict 1.5s/3.0s penalty via the default
            # thresholds in validate_cue_sanity.)
            sanity = validate_cue_sanity(
                target_text,
                reference,
                threshold_ms=FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
            )
            if not sanity["ok"]:
                logger.warning(
                    "[sync] %s reference failed cue-sanity (%s, penalty %s) -> next strategy",
                    strategy_name,
                    sanity["reason"],
                    sanity["penalty"],
                )
                continue

            # Strip pre-speech intro branding/cards so alass never anchors a
            # translator credit at 00:00:02 to an audio speech cue at 00:00:49.
            if sanity["reference_first_ms"] is not None:
                stripped = strip_intro_credits(target_text, sanity["reference_first_ms"])
                if stripped != target_text:
                    logger.info("[sync] stripped intro non-speech cue(s) before alass")
                    target_text = stripped

            # Deterministic serving gate: an already-aligned subtitle (median
            # offset below threshold) is served as-is and cached; otherwise
            # alass must run. A raw passthrough with an observable offset is
            # never served when a valid reference exists.
            offset = median_cue_offset(target_text, reference)
            if offset is not None and abs(offset) < ALIGNED_OFFSET_THRESHOLD_S:
                logger.info(
                    "[sync] target already aligned (median offset %+.2fs) -> serving original",
                    offset,
                )
                if self._sync_cache is not None:
                    await self._sync_cache.clear_failed(resolution_key)
                    await self._sync_cache.set(resolution_key, sub_bytes)
                return sub_bytes

            relaxed = decision_kind == "edition" and not is_informative_release_name(
                meta.get("target_filename") or meta.get("release_name")
            )
            reference_partial = bool(getattr(resolved, "partial", False))

            synced = await self._sync_service.sync_async(
                target_text,
                reference,
                decision_kind=decision_kind,
                is_series=query.is_series,
                source_confirmed=resolved.bluray_match,
                relaxed=relaxed,
                reference_partial=reference_partial,
            )

            if synced:
                key = build_synced_cache_key(
                    meta, target_id, content_hash=content_hash, decision=decision_kind
                )
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
                logger.info(
                    "[sync] synchronized subtitle %s (%d bytes, strategy=%s) and cached",
                    target_id,
                    len(synced),
                    strategy_name,
                )
                return synced.encode("utf-8")

            logger.warning(
                "[sync] %s strategy reference failed sync/validation -> falling back to next strategy",
                strategy_name,
            )

        if not last_resolved.text:
            logger.info("[sync] no deterministic reference available -> aborting sync")
        else:
            logger.warning("[sync] all sync strategies failed -> serving original subtitle")
        if self._sync_cache is not None:
            await self._sync_cache.mark_failed(resolution_key)
        return sub_bytes

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
