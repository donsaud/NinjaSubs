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
import uuid
from datetime import UTC, datetime
from typing import Any

from app.config import settings
from app.logging_context import reset_request_id, set_request_id, stable_request_id
from app.services.subtitle_matcher import (
    ALIGNED_OFFSET_THRESHOLD_S,
    FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
    median_cue_offset,
    parse_srt_cues,
    validate_cue_sanity,
)
from app.services.sync.alignment import (
    AlignmentAnalyzer,
    SubtitleEvaluation,
    SyncState,
    VerificationAvailability,
    is_reusable_verified,
    may_serve_synchronized,
)
from app.services.sync.matching import is_informative_release_name
from app.services.sync.query import ReferenceQuery, ResolvedReference, fingerprint_target_cues
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


#: Identifies this OS process, for cache diagnostics only.
#:
#: The payload store is process-local when Redis is absent, so a cache miss
#: after a restart is expected architecture rather than a defect. A production
#: trace cannot tell "same process, key drifted" from "new process, store
#: empty" unless the log says which. This makes that distinction readable
#: without exposing anything sensitive: it is a random value generated at
#: import, not derived from configuration.
_PROCESS_INSTANCE_ID = uuid.uuid4().hex[:12]


def _fingerprint_source(meta: dict, target_id: str) -> str:
    """Which input actually identified the TARGET VIDEO for cache scoping.

    Recorded because the strength of the binding differs sharply between these
    cases, and a log line that cannot distinguish them hides a real risk. The
    order below is the full fallback chain, strongest first:

        video_hash       a real content hash of the video: true identity
        stream_context   sha256 of the stream URL: identifies the stream, not
                         the bytes, and a signed URL may rotate
        target_filename  the release name a user reported: identifies a release
                         at best, and a re-mux of the same release is identical
        target_id        the SUBTITLE id. This is not video identity at all and
                         is named explicitly so it is never mistaken for a
                         cryptographic fingerprint.

    The weaker sources are retained deliberately: refusing every request that
    arrives without a video hash would reject most real playback. They are
    labelled, not hidden.
    """
    if str(meta.get("video_hash") or "").strip():
        return "video_hash"
    if meta.get("stream_url"):
        return "stream_context"
    if str(meta.get("target_filename") or "").strip():
        return "target_filename"
    return "target_id"


def credential_source(meta: dict) -> str:
    """Where the request's EFFECTIVE provider credentials came from.

    The precedence rule already exists and is applied in
    ``config_parser.parse_user_config``::

        effective = (user_config_value or settings.ENV_VALUE or "").strip()

    The manifest/config value wins, but an empty one falls through to the
    environment, and both may legitimately be empty for a deployment that
    receives keys per-request through the Stremio config URL.

    Provenance is READ, not re-derived. Once the resolved values land in
    ``meta`` the origin is gone, so a helper that tried to infer it from the
    values would report "manifest" for an environment-supplied key whenever a
    user also supplied one. ``UserPreferences.credential_source`` captures the
    real answer at the moment the choice is made.

    This function only labels. It never changes precedence and never feeds the
    credential digest, so it cannot alter cache identity.
    """
    recorded = meta.get("credential_source")
    if isinstance(recorded, str) and recorded:
        return recorded
    # No recorded provenance (older cached metadata, or a caller that did not
    # supply preferences). Fall back to presence only, and say so rather than
    # guessing an origin that is genuinely unknowable at this point.
    return "present_unlabelled" if any(
        str(meta.get(field) or "").strip()
        for field in ("subdl_key", "subsource_key", "opensubtitles_key")
    ) else "default"


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
        # Verification layer: measures alass output instead of trusting exit 0.
        self._analyzer = AlignmentAnalyzer()
        # Most expensive runs allowed per request, so one subtitle request can
        # never fan out into dozens of alass subprocesses.
        try:
            self._alass_candidate_limit = max(
                1, int(getattr(settings, "ALASS_CANDIDATE_LIMIT", 3) or 3)
            )
        except (TypeError, ValueError):
            self._alass_candidate_limit = 3
        # Most recent alignment verdict, for diagnostics/debug endpoints.
        self._last_evaluation: SubtitleEvaluation | None = None
        # Observability counters. Reset per request by the caller if a rate is
        # wanted; exposed via .metrics for debug logging.
        self._metrics: dict[str, int] = {
            "verification_cache_hits": 0,
            "verification_cache_misses": 0,
            "alass_runs": 0,
            "candidates_considered": 0,
            "candidates_verified": 0,
        }

    def _strategies(self) -> list[tuple[str, Any]]:
        strats = []
        if self._external_strategy is not None:
            strats.append(("external exact-match", self._external_strategy))
        if self._hash_strategy is not None:
            strats.append(("hash-exact", self._hash_strategy))
        return strats

    def _build_query(
        self, meta: dict, target_id: str | None = None, target_text: str | None = None
    ) -> ReferenceQuery:
        # target_filename is the target VIDEO filename only. The candidate's own
        # release_name must never stand in for it: doing so would hand reference
        # tiering a fabricated edition (e.g. "Dexter.2006.S08E05.srt" yields no
        # source/resolution/group, demoting every reference to TIER_FALLBACK)
        # when the request in fact carried no video fingerprint at all.
        return ReferenceQuery(
            imdb_id=str(meta.get("imdb_id") or ""),
            target_filename=meta.get("target_filename"),
            target_sub_id=target_id or meta.get("sub_id"),
            target_sub_release_name=meta.get("release_name"),
            target_download_url=meta.get("download_url"),
            target_cue_digest=fingerprint_target_cues(target_text),
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
        # Tag every sync log line emitted from here with a correlation id.
        # Concurrent requests for the same episode otherwise interleave, and
        # attributing their lines to the wrong request is how an incident
        # appeared to show contradictory diagnostics that were not.
        request_token = set_request_id(stable_request_id(meta, target_id, sub_bytes))
        try:
            return await self._evaluate_and_sync(
                sub_bytes, meta, target_id, auto_sync
            )
        finally:
            reset_request_id(request_token)

    async def _evaluate_and_sync(
        self, sub_bytes: bytes, meta: dict, target_id: str, auto_sync: bool = False
    ) -> bytes:
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

    @staticmethod
    def _verdict_key(meta: dict, subtitle_hash: str) -> str | None:
        """Cache key for a measured verdict, or ``None`` when unknowable.

        Without a video fingerprint there is nothing safe to key on, so no
        verdict is stored or reused. That is the whole point: a catalogue
        request must not inherit a verdict measured against some other video.
        """
        from app.services.sync.matching import has_video_fingerprint
        from app.services.sync_cache import SyncCache

        if not has_video_fingerprint(meta):
            return None
        fingerprint = SyncCache.video_fingerprint_from_meta(meta)
        if not fingerprint:
            return None
        return SyncCache.build_verdict_key(
            fingerprint, subtitle_hash, str(meta.get("lang") or "und")
        )

    @staticmethod
    def _evaluation_from_verdict(verdict: dict) -> SubtitleEvaluation:
        """Rebuild an evaluation from a stored verdict.

        ``verification`` is forced to ``cached`` by the cache on read, so a
        recalled result is never presented as a fresh measurement.
        """
        evaluation = SubtitleEvaluation(
            sync_state=verdict.get("sync_state", SyncState.UNVERIFIED.value),
            verification=verdict.get("verification", VerificationAvailability.CACHED.value),
            reasons=list(verdict.get("reasons") or []),
            median_offset_ms=verdict.get("median_offset_ms"),
            p95_offset_ms=verdict.get("p95_offset_ms"),
            mad_offset_ms=verdict.get("mad_offset_ms"),
            drift_ms_per_minute=verdict.get("drift_ms_per_minute"),
            coverage_score=verdict.get("coverage_score"),
            structural_similarity=verdict.get("structural_similarity"),
            cut_verdict=verdict.get("cut_verdict"),
        )
        return evaluation

    def _audit_serve(
        self,
        evaluation: SubtitleEvaluation,
        meta: dict,
        target_id: str,
        *,
        from_cache: bool,
        resolved: Any = None,
    ) -> None:
        """Shadow-record a serve-time verification. Never affects the result.

        A no-op unless the audit is explicitly enabled, so the default install
        pays nothing beyond a boolean check.
        """
        from app.services.sync.audit import (
            AUDIT_LOG,
            record_for_evaluation,
            stable_id,
        )
        from app.services.sync.matching import has_video_fingerprint
        from app.services.sync_cache import SyncCache

        if not AUDIT_LOG.enabled:
            return
        try:
            fingerprint = SyncCache.video_fingerprint_from_meta(meta)
            AUDIT_LOG.record_serve(
                record_for_evaluation(
                    evaluation,
                    phase="serve",
                    reference=resolved,
                    video_id=fingerprint,
                    subtitle_id=stable_id(target_id),
                    language=str(meta.get("lang") or "und"),
                    has_video_fingerprint=has_video_fingerprint(meta),
                    request_context=(
                        "resolved_stream" if has_video_fingerprint(meta) else "catalogue_request"
                    ),
                    from_cache=from_cache,
                )
            )
        except Exception as exc:  # pragma: no cover - telemetry must not break serving
            logger.debug("[audit] serve record dropped: %s", exc)

    @property
    def metrics(self) -> dict[str, int]:
        """Counters for observability (cache hit rate, alass runs per request)."""
        return dict(self._metrics)

    async def _store_verdict(
        self,
        verdict_key: str | None,
        evaluation: SubtitleEvaluation,
        content_hash: str,
        meta: dict | None = None,
        target_id: str | None = None,
        artifact_key: str | None = None,
    ) -> None:
        """Persist a measured verdict for reuse, if it is safe to key.

        Two keys are written: the content-addressed primary key used at serve
        time, and a fingerprint-bound alias keyed by candidate id so the listing
        path can find it before the subtitle bytes exist. Both are bound to the
        video fingerprint, so neither can leak across videos.

        ``artifact_key`` records where the synchronized bytes were stored. A
        resync verdict claims the subtitle was re-timed, so reusing that claim
        requires the transformed artifact; without the key the reuse path would
        have to guess the decision segment of the cache key, and a wrong guess
        would serve the unsynchronized original under a verified label.
        """
        if not verdict_key or self._sync_cache is None:
            return
        alias_key = None
        if meta and target_id:
            from app.services.sync_cache import SyncCache

            fingerprint = SyncCache.video_fingerprint_from_meta(meta)
            if fingerprint:
                alias_key = SyncCache.build_verdict_alias_key(fingerprint, target_id)
        payload = evaluation.model_dump(mode="json")
        if artifact_key:
            payload["artifact_key"] = artifact_key
        await self._sync_cache.set_verdict(
            verdict_key, payload, alias_key=alias_key
        )

    async def _execute(
        self, sub_bytes: bytes, meta: dict, target_id: str, auto_sync: bool
    ) -> bytes:
        """The expensive pipeline body: strategies, cache, alass (see evaluate_and_sync)."""
        content_hash = hashlib.sha256(sub_bytes).hexdigest()[:16]
        # Positive cache: serve an already-synced subtitle without any work.
        resolution_key = self._flight_key(meta, target_id, sub_bytes)
        if self._sync_cache is not None:
            cached = await self._sync_cache.get(resolution_key)
            # Structured lookup trace. A production trace showed a cache HIT at
            # 14:08 and a MISS three minutes later for the same sub_id and the
            # same target, with nothing in the log able to say why. The key is
            # a composite of several independently-varying inputs, so which one
            # moved is only knowable if the inputs are recorded.
            #
            # `result` distinguishes the states that were previously
            # indistinguishable: hit, miss, negative_hit. Nothing sensitive is
            # recorded here: sub_id is already logged on this path, and the
            # fingerprint is reported as a presence flag plus a digest.
            fingerprint_source = _fingerprint_source(meta, target_id)
            digest = hashlib.sha256(resolution_key.encode()).hexdigest()[:12]
            if cached:
                # A payload hit is NOT a verification. This fast path returns
                # the stored bytes with no analyzer call, so an entry written
                # from a non-positive evaluation would otherwise be re-served as
                # a finished synchronization -- a real defect observed on a
                # Whiplash 2160p REMUX request, where an UNVERIFIED/UNKNOWN
                # Alass output was stored here and later re-presented as a
                # pre-synced subtitle without re-verification.
                #
                # Fail closed: only an entry whose recorded outcome is
                # verified-and-measured may be served from here. Anything else
                # is treated as a miss, which falls through to the normal path
                # and recomputes from the original subtitle. The original bytes
                # are always available, so a miss is safe; serving an
                # unverified artifact is not.
                meta_record = None
                try:
                    meta_record = await self._sync_cache.get_meta(resolution_key)
                except Exception as exc:  # pragma: no cover - cache must not break sync
                    logger.debug("[sync-cache] meta read failed: %s", exc)
                stored_state = (meta_record or {}).get("sync_state")
                stored_verification = (meta_record or {}).get("verification")
                if is_reusable_verified(stored_state, stored_verification):
                    logger.info(
                        "[sync-cache] layer=payload result=hit sub_id=%s process=%s "
                        "fingerprint_source=%s credential_source=%s content_hash=%s "
                        "key_digest=%s sync_state=%s verification=%s",
                        target_id,
                        _PROCESS_INSTANCE_ID,
                        fingerprint_source,
                        credential_source(meta),
                        content_hash,
                        digest,
                        stored_state,
                        stored_verification,
                    )
                    logger.info(
                        "[sync] cache HIT for sub=%s -> serving pre-synced subtitle immediately",
                        target_id,
                    )
                    self._metrics["verification_cache_hits"] += 1
                    return cached
                logger.info(
                    "[sync-cache] layer=payload result=miss sub_id=%s process=%s "
                    "fingerprint_source=%s credential_source=%s content_hash=%s "
                    "key_digest=%s store_layer=%s "
                    "miss_reason=stored_result_not_reusable sync_state=%s "
                    "verification=%s",
                    target_id,
                    _PROCESS_INSTANCE_ID,
                    fingerprint_source,
                    credential_source(meta),
                    content_hash,
                    digest,
                    "in_memory" if self._sync_cache.is_ephemeral else "redis",
                    stored_state or "absent",
                    stored_verification or "absent",
                )
            # Only a genuine absence is reported as such. An entry that was
            # present but gated above already reported
            # miss_reason=stored_result_not_reusable; logging absent_in_store as
            # well would make diagnostics contradict themselves.
            if not cached:
                logger.info(
                    "[sync-cache] layer=payload result=miss sub_id=%s process=%s "
                    "fingerprint_source=%s credential_source=%s content_hash=%s "
                    "key_digest=%s miss_reason=absent_in_store store_layer=%s",
                    target_id,
                    _PROCESS_INSTANCE_ID,
                    fingerprint_source,
                    credential_source(meta),
                    content_hash,
                    digest,
                    "in_memory" if self._sync_cache.is_ephemeral else "redis",
                )

        if self._sync_cache is not None and await self._sync_cache.is_failed(resolution_key):
            logger.info(
                "[sync] negative cache hit for %s -> skipping provider fan-out",
                resolution_key,
            )
            return sub_bytes

        from app.extractor import decode_subtitle_bytes

        target_text = decode_subtitle_bytes(sub_bytes, lang=meta.get("lang"))
        query = self._build_query(meta, target_id=target_id, target_text=target_text)

        # Measured-verdict cache: reuse a previous *verified* result for this
        # exact video + subtitle + language + engine. Keyed on the video
        # fingerprint, so a verdict measured for one video is never handed to
        # another. A miss here is normal, not a failure.
        verdict_key = self._verdict_key(meta, content_hash)
        if verdict_key:
            remembered = await self._sync_cache.get_verdict(verdict_key) if self._sync_cache else None
            remembered_state = str((remembered or {}).get("sync_state") or "")
            # VERIFIED_RESYNCED means alass DID re-time this subtitle, so the
            # correct bytes are the transformed artifact, not the original. This
            # branch serves `sub_bytes` (the original), so it may only claim a
            # resync when that artifact is actually present; otherwise it would
            # serve unsynchronized bytes under a "verified resync" label.
            resync_artifact = None
            if remembered_state == SyncState.VERIFIED_RESYNCED.value and self._sync_cache is not None:
                artifact_key = (remembered or {}).get("artifact_key")
                if isinstance(artifact_key, str) and artifact_key:
                    resync_artifact = await self._sync_cache.get(artifact_key)
            resync_artifact_available = resync_artifact is not None
            if (
                remembered
                and remembered_state in (
                    SyncState.VERIFIED_SYNCED.value,
                    SyncState.VERIFIED_RESYNCED.value,
                )
                and (remembered_state != SyncState.VERIFIED_RESYNCED.value or resync_artifact_available)
            ):
                self._metrics["verification_cache_hits"] += 1
                self._last_evaluation = self._evaluation_from_verdict(remembered)
                # A verdict cache hit means no reference was fetched this
                # request, so there is no reference evidence to record.
                self._audit_serve(
                    self._last_evaluation, meta, target_id, from_cache=True, resolved=None
                )
                logger.info(
                    "[sync] reusing measured verdict for sub=%s: %s (no alass run)",
                    target_id,
                    self._last_evaluation.explain(),
                )
                if self._sync_cache is not None:
                    # A resync verdict serves the transformed artifact; a no-op
                    # verdict has nothing to re-time, so the original is
                    # already the synchronized subtitle.
                    reused_bytes = resync_artifact if resync_artifact is not None else sub_bytes
                    await self._sync_cache.set(resolution_key, reused_bytes)
                    # The verdict re-checked above is a verified one, so this
                    # entry is recorded as reusable.
                    await self._sync_cache.set_meta(
                        resolution_key,
                        {
                            "sync_state": str(remembered.get("sync_state")),
                            "verification": str(remembered.get("verification") or "cached"),
                            "sync_confidence": remembered.get("sync_confidence"),
                        },
                    )
                return reused_bytes

        if self._sync_service is None:  # pragma: no cover - defensive
            logger.warning("[sync] no sync service configured -> serving original subtitle")
            return sub_bytes

        # Reference strategies in priority order (external exact-match reference,
        # then OpenSubtitles MovieHash). The playing stream is never probed.
        attempts: list[tuple[str, Any]] = list(self._strategies())

        def _passes_cue_sanity(reference_text: str) -> bool:
            """Execution-window cue check, used to reject a wrong-cut reference.

            A mislabeled or season-pack reference can be downloaded and only
            then revealed as unusable; validating inside the strategy lets it
            walk to the next-ranked candidate instead of aborting the sync.
            """
            verdict = validate_cue_sanity(
                target_text,
                reference_text,
                threshold_ms=FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
            )
            if not verdict["ok"]:
                logger.warning(
                    "[sync] candidate reference rejected by cue-sanity (%s); trying next candidate",
                    verdict["reason"],
                )
            return bool(verdict["ok"])

        last_resolved = ResolvedReference(None)
        # Request-scoped alass budget.
        #
        # This counter used to live in ``self._metrics``, which is created once
        # in ``__init__`` on an orchestrator that ``main.py`` deliberately keeps
        # as a process-wide singleton. Nothing reset it per request, so the
        # allowance was consumed by whichever requests arrived first and never
        # replenished: in a long-lived process every later request deferred its
        # candidates immediately and served the original subtitle unsynchronized.
        # The code's own comment stated the intended policy -- "per request ...
        # left for a later request" -- which a process-lifetime counter cannot
        # deliver. A production trace showed exactly that: the alass limit
        # reported exhausted with no alass execution in that request at all.
        #
        # Kept local, so concurrent requests cannot share or reset each other's
        # allowance. ``self._metrics["alass_runs"]`` is still incremented, but
        # purely for process-wide observability; it no longer gates anything.
        alass_attempted = 0
        alass_started = 0
        alass_completed = 0
        alass_deferred = 0
        candidates_rejected = 0

        for strategy_name, strategy_obj in attempts:
            try:
                if getattr(strategy_obj, "validates_target", False):
                    resolved = await strategy_obj.resolve_with_provenance(
                        query, update_validator=_passes_cue_sanity
                    )
                else:
                    resolved = await strategy_obj.resolve_with_provenance(query)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("[sync] %s strategy failed: %s -> next", strategy_name, exc)
                resolved = ResolvedReference(None)

            if not resolved.text:
                continue

            last_resolved = resolved
            reference = resolved.text
            decision_kind = resolved.kind
            # This only means a reference is now IN HAND. Alass may still not
            # run: the already-aligned shortcut below returns the original
            # subtitle before any subprocess is spawned. Claiming "invoking
            # alass" here made live traces read as though Alass had executed
            # when it had not. The exec site logs it truthfully instead.
            logger.info(
                "[sync] %s strategy provided a reference (decision=%s, %d bytes); "
                "evaluating whether alass is needed",
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
                candidates_rejected += 1
                logger.info(
                    "[sync-budget] limit=%d attempted=%d alass_started=%d "
                    "alass_completed=%d rejected_before_alass=%d deferred=%d "
                    "candidate=%s strategy=%s reason=cue_sanity_rejected "
                    "counted_toward_limit=false",
                    self._alass_candidate_limit,
                    alass_attempted,
                    alass_started,
                    alass_completed,
                    candidates_rejected,
                    alass_deferred,
                    hashlib.sha256(str(strategy_name).encode()).hexdigest()[:8],
                    strategy_name,
                )
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
                # No re-timing needed. Classify explicitly so "already aligned"
                # is a measured claim, not an assumption.
                # Same handoff as the post-alass path: parse the text alass
                # would have been given, then hand the collection over.
                target_cues = parse_srt_cues(target_text)
                logger.info(
                    "[sync] verifier input: target_raw_chars=%d "
                    "target_parsed_cue_count=%d verifier_target_cue_count=%d",
                    len(target_text),
                    len(target_cues),
                    len(target_cues),
                )
                evaluation = self._analyzer.analyze(
                    target_cues,
                    None,
                    reference,
                    alass_applied=False,
                    reference_trust=getattr(resolved, "reference_trust", None),
                    reference_reasons=list(getattr(resolved, "reference_reasons", None) or []),
                    reference_consensus=getattr(resolved, "reference_consensus", None),
                    reference_independent_sources=getattr(
                        resolved, "reference_independent_sources", None
                    ) or None,
                    reference_failure=getattr(resolved, "reference_failure", None),
                )
                self._last_evaluation = evaluation
                self._metrics["candidates_verified"] += 1
                await self._store_verdict(
                    verdict_key, evaluation, content_hash, meta, target_id
                )
                self._audit_serve(
                    evaluation, meta, target_id, from_cache=False, resolved=resolved
                )
                logger.info(
                    "[sync] target already aligned (median offset %+.2fs) -> serving original "
                    "[%s]",
                    offset,
                    evaluation.sync_state.value,
                )
                if self._sync_cache is not None:
                    await self._sync_cache.clear_failed(resolution_key)
                    await self._sync_cache.set(resolution_key, sub_bytes)
                    await self._sync_cache.set_meta(
                        resolution_key,
                        {
                            "sync_state": evaluation.sync_state.value,
                            "verification": evaluation.verification.value,
                            "sync_confidence": evaluation.sync_confidence,
                        },
                    )
                return sub_bytes

            # "Relaxed" means the target name carries no edition signal, so the
            # timeline gate must be tighter. An absent target_filename is the
            # strongest form of that: it is never borrowed from the subtitle.
            relaxed = decision_kind == "edition" and not is_informative_release_name(
                meta.get("target_filename")
            )
            reference_partial = bool(getattr(resolved, "partial", False))

            # Cost control: at most ALASS_CANDIDATE_LIMIT expensive alass
            # executions PER REQUEST. Candidates past that are left for a later
            # request rather than spawning unbounded subprocesses.
            #
            # The check stays after the already-aligned shortcut on purpose: that
            # path costs no subprocess and still benefits from having a
            # reference to measure against, so it must not be starved by a
            # budget that only exists to bound subprocesses.
            if alass_attempted >= self._alass_candidate_limit:
                alass_deferred += 1
                logger.info(
                    "[sync-budget] limit=%d attempted=%d alass_started=%d "
                    "alass_completed=%d rejected_before_alass=%d deferred=%d "
                    "candidate=%s strategy=%s reason=budget_exhausted "
                    "counted_toward_limit=false",
                    self._alass_candidate_limit,
                    alass_attempted,
                    alass_started,
                    alass_completed,
                    candidates_rejected,
                    alass_deferred,
                    hashlib.sha256(str(strategy_name).encode()).hexdigest()[:8],
                    strategy_name,
                )
                logger.info(
                    "[sync] alass candidate limit reached (%d) -> deferring remaining candidates",
                    self._alass_candidate_limit,
                )
                continue

            alass_attempted += 1
            alass_started += 1
            self._metrics["alass_runs"] += 1
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
                alass_completed += 1

            if synced:
                # Independent verification: alass exiting 0 is not proof of
                # synchronization. Measure what it actually produced and record
                # an explainable verdict.
                #
                # This is deliberately OBSERVATIONAL in this phase. The existing
                # gates (_validate_synced_output, validate_cue_sanity, the
                # timeline band) remain authoritative for accept/reject; the
                # analyzer classifies and explains rather than overruling them.
                # Otherwise a stricter new metric would silently change which
                # subtitles are served, which is exactly the coupling to avoid.
                #
                # The target cues are parsed HERE, from the exact text alass was
                # given, and handed over as a collection. The analyzer accepts
                # either raw text or cues; it used to be given the text and
                # reparse it with its own ``parse_srt_cues``. Two parsers with
                # two accepted timestamp grammars, and a real trace where the
                # sync service counted 584 target cues while the verifier was
                # handed 0. Parsing at the boundary makes
                # ``pre-alass count == verifier count`` an invariant.
                target_cues = parse_srt_cues(target_text)
                logger.info(
                    "[sync] verifier input: target_raw_chars=%d "
                    "target_parsed_cue_count=%d verifier_target_cue_count=%d",
                    len(target_text),
                    len(target_cues),
                    len(target_cues),
                )
                evaluation = self._analyzer.analyze(
                    target_cues,
                    synced,
                    reference,
                    alass_applied=True,
                    alass_successful=True,
                    reference_trust=getattr(resolved, "reference_trust", None),
                    reference_reasons=list(getattr(resolved, "reference_reasons", None) or []),
                    reference_consensus=getattr(resolved, "reference_consensus", None),
                    reference_independent_sources=getattr(
                        resolved, "reference_independent_sources", None
                    ) or None,
                    reference_failure=getattr(resolved, "reference_failure", None),
                )
                self._last_evaluation = evaluation
                self._metrics["candidates_verified"] += 1
                # Computed before the verdict is stored: a resync verdict records
                # where its transformed artifact lives, so a later reuse can
                # serve the synchronized bytes instead of the original.
                key = build_synced_cache_key(
                    meta, target_id, content_hash=content_hash, decision=decision_kind
                )
                await self._store_verdict(
                    verdict_key, evaluation, content_hash, meta, target_id, artifact_key=key
                )
                self._audit_serve(
                    evaluation, meta, target_id, from_cache=False, resolved=resolved
                )
                log = logger.warning if evaluation.sync_state is SyncState.REJECTED else logger.info
                log("[sync] alignment %s: %s", evaluation.sync_state.value, evaluation.explain())

                if self._sync_cache is not None:
                    await self._sync_cache.clear_failed(resolution_key)
                    await self._sync_cache.set(key, synced.encode("utf-8"))
                    await self._sync_cache.set(resolution_key, synced.encode("utf-8"))
                    # Record the measured outcome next to BOTH artifacts. The
                    # read path refuses to serve a payload whose recorded state
                    # is not verified-and-measured, so the artifact may exist
                    # without ever being reusable as a verified result.
                    outcome_meta = {
                        "sync_state": evaluation.sync_state.value,
                        "verification": evaluation.verification.value,
                        "sync_confidence": evaluation.sync_confidence,
                    }
                    await self._sync_cache.set_meta(key, {**outcome_meta, **{
                        "status": "synced",
                        "applied_shift": self._applied_shifts(target_text, synced),
                        "reference_sha": hashlib.sha256(reference.encode("utf-8")).hexdigest()[:16],
                        "decision": decision_kind,
                        "timestamp": datetime.now(UTC).isoformat(),
                    }})
                    await self._sync_cache.set_meta(resolution_key, outcome_meta)
                # Serving contract: the analyzer is the authority on whether
                # these bytes may replace the original. An attempt it does not
                # trust is kept for diagnostics/cache but never returned to the
                # user, who gets the original provider subtitle instead -- the
                # unsynchronized original is far more usable than a mangled one.
                serve_synchronized = may_serve_synchronized(
                    evaluation.sync_state.value, evaluation.verification.value
                )
                if serve_synchronized:
                    served = synced.encode("utf-8")
                else:
                    logger.warning(
                        "[sync] not serving the synchronized output for %s: verifier did not "
                        "trust it (state=%s, verification=%s) -> serving the original subtitle",
                        target_id,
                        evaluation.sync_state.value,
                        evaluation.verification.value,
                    )
                    served = sub_bytes
                logger.info(
                    "[sync] %s synchronization result for %s (%d bytes, "
                    "strategy=%s, state=%s, verification=%s)",
                    "stored" if serve_synchronized else "rejected",
                    target_id,
                    len(synced),
                    strategy_name,
                    evaluation.sync_state.value,
                    evaluation.verification.value,
                )
                return served

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
