"""Sync orchestrator: routes each request through candidate strategies.

Flow per request: gates (server flag, user preference, known language) →
content-bound sync-cache lookup → hash-exact strategy → external-release-reference
strategy → alass alignment → cache the result. Any step may fail; on any
failure the original, unmodified subtitle bytes are returned so the player
always receives a usable subtitle (strict fail-safe fallback).
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import partial
from typing import Any, cast

from app.config import settings
from app.logging_context import reset_request_id, set_request_id, stable_request_id
from app.services.subtitle_matcher import (
    FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
    measure_alignment_consistency,
    parse_srt_cues,
    validate_cue_sanity,
)
from app.services.sync.alignment import (
    MAX_PLAUSIBLE_OFFSET_MS,
    AlignmentAnalyzer,
    SubtitleEvaluation,
    SyncState,
    VerificationAvailability,
    is_reusable_verified,
    may_serve_synchronized,
)
from app.services.sync.anchor_preshift import (
    decide_anchor_preshift,
    preshift_text,
)
from app.services.sync.boundary_verifier import BoundaryShadow
from app.services.sync.candidate_reconstruction import (
    reconstruct_with_original_dialogue,
)
from app.services.sync.case_capture import (
    capture_affine,
    capture_output,
    capture_provider,
    capture_reference,
    capture_selected,
    capture_transformed,
    capture_verifier,
)
from app.services.sync.constant_shift_exemption import (
    ConstantShiftExemptionState,
    apply_exemption_to_evaluation,
    decide_constant_shift_exemption,
    measure_median_abs_residual,
)
from app.services.sync.delivery import DeliveryAuthorization, DeliveryPayload
from app.services.sync.hybrid_router import unified_authority_enabled, unified_review_async
from app.services.sync.large_offset import (
    LARGE_OFFSET_MAX_MS,
    REASON_MAX_OFFSET_EXCEEDED,
    LargeOffsetAssessment,
    assess_large_offset,
    classify_large_offset_candidate,
)
from app.services.sync.large_offset_investigation import (
    LargeOffsetInvestigation,
    LargeOffsetServingState,
    decide_large_offset_serving,
    investigate_large_offset,
    validate_alass_output,
)
from app.services.sync.matching import (
    is_bare_disc_track_name,
    is_informative_release_name,
    parse_release_identity,
)
from app.services.sync.metadata_gate import MetadataGateAction, decide_metadata_gate
from app.services.sync.outcomes import (
    LARGE_OFFSET_SERVED,
    REJECTED,
    SYNC_OUTCOMES,
    VERIFIED,
    classify_outcome,
)
from app.services.sync.quality_fallback import assess_quality_fallback
from app.services.sync.query import ReferenceQuery, ResolvedReference, fingerprint_target_cues
from app.services.sync.reference import is_dialogue_cue
from app.services.sync.release_knowledge import compatible_source
from app.services.sync.target_binding import (
    BindingAssessment,
    TargetBinding,
    classify_target_binding,
)
from app.services.sync.telemetry import SyncTiming, current_sync_timing
from app.services.sync_cache import SYNC_PAYLOAD_VERSION, SyncCache
from app.utils.cleaners import strip_intro_credits

logger = logging.getLogger(__name__)

# Reference ``kind`` values that assert exact current-video identity (i.e. a
# verified MovieHash match) rather than filename/title similarity. Only a
# hash-derived reference may align a target that carries no identity of its
# own; everything else is a heuristic guess.
_HASH_DERIVED_REFERENCE_KINDS = frozenset({"hash"})

#: Primary linear synchronization: two verified dialogue anchors fitted to
#: ``t_new = scale * t + offset`` on the pristine original, judged only by
#: the Unified timing measurement. ALASS remains the fallback for anything
#: this cannot solve. Versioned into cache decision segments so future
#: algorithm changes cannot masquerade as this one.
TWO_POINT_STRATEGY_ID = "two_point"
TWO_POINT_MODEL_VERSION = "two-point-v1"

#: Wall-clock bound for one Two-Point anchor-detection run. Real releases
#: measure 0.06-0.9s; adversarially repetitive texts can explode the RANSAC
#: pair space instead of converging. Exceeding the bound abstains to the
#: ALASS fallback (or to ORIGINAL) rather than stalling the request, and
#: never changes a verdict: detection either completes identically or its
#: result is discarded unused.
TWO_POINT_DETECTION_TIMEOUT_SECONDS = 8.0

#: Version of the guarded Large-Offset delivery authorization that may be
#: reused from the result cache. Stored beside every corrected artifact.
#: Bumping it invalidates old entries after any change to the guard logic;
#: an unmarked entry simply re-runs the normal pipeline once.
LARGE_OFFSET_CACHE_AUTHORIZATION_VERSION = "los-serve-v1"

#: Decision segment for the content-addressed guarded artifact. It is constant
#: across subtitle IDs, so a byte-identical original for the same video and
#: language can find the already-authorized correction without discovery.
CONTENT_ARTIFACT_DECISION = "large_offset_content_v1"


def _add_known_source(query: Any, digest: str) -> tuple[str, ...]:
    """Append a source-download digest to the query's exclusions.

    The sticky, season-pack and video-cache tiers each contribute one; an
    assignment from a later tier used to drop the earlier exclusion, letting
    planning re-download (and spend provider quota on) an archive already used.
    """
    existing = tuple(getattr(query, "known_source_digests", ()) or ())
    return existing if digest in existing else existing + (digest,)


class _PreResolvedReference:
    """Replays one already-resolved reference through the existing loop.

    Holds a candidate produced by `ExternalExactStrategy.resolve_all_with_provenance`
    so the orchestrator's normal per-candidate path -- cue-sanity gate, disc-track
    skip, ALASS budget, and the verifier -- applies to it unchanged.
    """

    validates_target = False

    def __init__(self, resolved: Any) -> None:
        self._resolved = resolved

    async def resolve_with_provenance(self, query: Any, **kwargs: Any) -> Any:
        return self._resolved


@dataclass(frozen=True)
class TimingWitnessDecision:
    """What a local timing witness proved about a metadata-compatible pairing.

    ``reason`` is one of the gate module's bounded codes:

    * ``timing_witness_aligned`` -- measured, no re-timing needed (fast original);
    * ``timing_witness_insufficient`` -- unmeasurable or too little
      correspondence; missing evidence is not a contradiction (fast original);
    * ``timing_witness_unavailable`` -- no local reference for this video at all
      (fast original, and no provider request is made to find one);
    * ``timing_witness_conflict`` -- a measured displacement (investigate).
    """

    investigate: bool
    reason: str
    witness: str | None = None
    median_offset_ms: float | None = None


class _LazyCandidate:
    """Downloads ONE planned reference the first time the loop reaches it.

    Holds a ``(release, provider)`` pair from
    `ExternalExactStrategy.plan_candidates` plus the request-scoped ``shared``
    state (``{"downloaded": int, "seen": set[str]}``). The download -- with the
    strategy's breaker, quota, pacing, cue-sanity validation and content-dedup
    guards -- happens inside `resolve_with_provenance`, i.e. only when the
    verification loop actually reaches this candidate. When an earlier candidate
    verifies, the loop returns and the remaining adapters never download, so a
    request that succeeds on the first candidate performs exactly one download
    instead of filling the whole pool upfront.

    Every guarantee of the eager path applies unchanged: the same per-candidate
    guards run (at download time, when breaker/quota state is freshest), and
    the loop's own cue-sanity gate, disc-track skip, ALASS budget and verifier
    still judge every downloaded candidate independently.
    """

    validates_target = False

    def __init__(
        self,
        strategy: Any,
        release: Any,
        provider: Any,
        query: Any,
        shared: dict[str, Any],
        validate: Any | None,
        plan_index: int | None = None,
    ) -> None:
        self._strategy = strategy
        self._release = release
        self._provider = provider
        self._query = query
        self._shared = shared
        self._validate = validate
        self._plan_index = plan_index

    async def resolve_with_provenance(self, query: Any, **kwargs: Any) -> Any:
        validate = kwargs.get("update_validator", self._validate)
        download = self._strategy.download_planned_candidate
        try:
            parameters = inspect.signature(download).parameters.values()
            accepts_plan_index = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                or parameter.name == "plan_index"
                for parameter in parameters
            )
        except (TypeError, ValueError):  # pragma: no cover - introspection is best effort
            accepts_plan_index = True
        extra = {"plan_index": self._plan_index} if accepts_plan_index else {}
        try:
            return await download(
                self._release,
                self._provider,
                self._query,
                validate,
                self._shared,
                **extra,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[sync] lazy candidate download failed: %s -> next", type(exc).__name__)
            return ResolvedReference(None)


class _PlannedPrefetcher:
    """Bounded, demand-driven preparation of planned reference downloads.

    The verification loop used to download candidate *N+1* only after candidate
    *N* had been fully evaluated. That serialised network latency behind
    ALASS/affine/verification: a real request spent ~2.4s waiting for downloads
    that could have been prepared while the previous candidate was being
    judged.

    This helper starts a bounded number of the planner's already-ranked
    candidates while the loop evaluates the current one. It never changes the
    plan: the same releases, provider guards, cue-sanity validator and pacing
    apply, and a prepared-but-unused download is cancelled. When a prefetched
    candidate turns out to be deterministically unusable (duplicate content,
    download failure, cue-sanity rejection -- the same cases the loop skips),
    the next candidate is prepared so that skip is instant too.
    """

    def __init__(
        self,
        *,
        strategy: Any,
        planned: list[tuple[Any, Any]],
        query: Any,
        shared: dict[str, Any],
        validate: Any | None,
        concurrency: int,
        lookahead: int,
    ) -> None:
        self._strategy = strategy
        self._planned = planned
        self._query = query
        self._shared = shared
        self._validate = validate
        self._lookahead = max(0, int(lookahead))
        self._tasks: dict[int, asyncio.Task] = {}
        self._consumed = -1
        self._closed = False
        # Speculative lookahead stays OFF until the orchestrator observes a
        # candidate that is both a large-offset candidate and has a
        # same-release-family sibling in the plan. A normal candidate that
        # verifies immediately must still cost exactly one download.
        self._armed = False
        self._ahead: list[int] = []
        self._evidence: dict[int, Any] = {}
        self._concurrency = max(1, int(concurrency))
        self._semaphore = asyncio.Semaphore(self._concurrency)
        self._families = [self._family_of(release) for release, _ in planned]
        self._identities = [
            self._identity_of(release) for release, _ in planned
        ]
        self._accepts_plan_index = True
        download = getattr(strategy, "download_planned_candidate", None)
        if callable(download):
            try:
                parameters = inspect.signature(download).parameters.values()
                self._accepts_plan_index = any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    or parameter.name == "plan_index"
                    for parameter in parameters
                )
            except (TypeError, ValueError):  # pragma: no cover - best effort
                self._accepts_plan_index = True

    @staticmethod
    def _family_of(release: Any) -> str:
        try:
            from app.services.sync.reference_v2 import release_family_key

            return release_family_key(str(getattr(release, "release_name", "") or ""))
        except Exception:
            return ""

    @staticmethod
    def _identity_of(release: Any) -> tuple[str, str]:
        """Provider-metadata identity: normalized release name + language.

        Used only to avoid *speculative* downloads of what two providers
        advertise as the same release. It never removes a candidate from the
        plan: if the preferred candidates fail, the deferred sibling is still
        downloaded and evaluated on demand.
        """
        name = str(getattr(release, "release_name", "") or "").strip().lower()
        for suffix in (".srt", ".ass", ".ssa", ".vtt", ".zip"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        name = " ".join(name.replace("_", " ").replace(".", " ").split())
        lang = str(getattr(release, "lang", "") or "").strip().lower()
        return name, lang

    def family(self, index: int) -> str:
        if 0 <= index < len(self._families):
            return self._families[index]
        return ""

    def identity(self, index: int) -> tuple[str, str]:
        if 0 <= index < len(self._identities):
            return self._identities[index]
        return ("", "")

    @property
    def lookahead(self) -> int:
        """Maximum number of unconsumed candidates this helper may prepare."""
        return self._lookahead

    @property
    def armed(self) -> bool:
        return self._armed

    def arm(self) -> None:
        """Allow one-ahead preparation after a large-offset preflight trigger."""
        if not self._armed:
            self._armed = True
            logger.info(
                "[sync] progressive reference preparation armed: "
                "lookahead=%d concurrency=%d",
                self._lookahead,
                self._concurrency,
            )

    def arm_ahead(self, index: int, ahead: list[int] | None = None) -> None:
        """Arm and immediately start the bounded window behind ``index``.

        ``ahead`` is the pre-computed set of speculative indices. It excludes
        siblings whose provider metadata is identical to the current release,
        so the same file mirrored by another provider is not downloaded twice
        speculatively. Those siblings stay in the plan and are fetched on
        demand if the preferred candidates fail.
        """
        self.arm()
        if ahead is not None:
            self._ahead = sorted({int(item) for item in ahead})
        self._schedule(index)
        for item in self._ahead:
            if item > index:
                self._schedule(item)

    def _next_ahead(self, index: int) -> int | None:
        for item in self._ahead:
            if item > index:
                return item
        return None

    def set_evidence(self, index: int, evidence: Any) -> None:
        self._evidence[index] = evidence

    def get_evidence(self, index: int) -> Any:
        return self._evidence.get(index)

    def has_task(self, index: int) -> bool:
        return index in self._tasks

    def peek(self, index: int) -> tuple[bool, Any]:
        """Return ``(finished, result)`` without awaiting or cancelling."""
        task = self._tasks.get(index)
        if task is None:
            return False, None
        if not task.done():
            return False, None
        if task.cancelled():
            return True, None
        try:
            return True, task.result()
        except Exception:
            return True, None

    def _schedule(self, index: int) -> None:
        if self._closed or index < 0 or index >= len(self._planned):
            return
        if index in self._tasks:
            return
        # Never prepare more than ``lookahead`` candidates beyond the one the
        # loop has reached. This is what keeps the pool from being downloaded
        # indiscriminately while one candidate is evaluated.
        if index > self._consumed + self._lookahead:
            return
        task = asyncio.create_task(self._download(index))
        self._tasks[index] = task
        task.add_done_callback(partial(self._after_done, index))
        timing = current_sync_timing.get()
        if timing is not None:
            timing.prefetch_started += 1

    async def _download(self, index: int) -> Any:
        async with self._semaphore:
            release, provider = self._planned[index]
            extra = {"plan_index": index + 1} if self._accepts_plan_index else {}
            try:
                return await self._strategy.download_planned_candidate(
                    release,
                    provider,
                    self._query,
                    self._validate,
                    self._shared,
                    **extra,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "[sync] prefetched candidate failed: %s -> next",
                    type(exc).__name__,
                )
                return ResolvedReference(None)

    def _after_done(self, index: int, task: asyncio.Task) -> None:
        """Keep the pipe warm after a deterministically skipped candidate.

        A result with text is NOT advanced past: only the verifier may decide
        that it is unusable. An empty result is exactly what the loop's
        ``if not resolved.text: continue`` will skip, so preparing the next
        candidate changes no decision and merely removes the next wait.
        Only armed (large-offset) requests advance speculatively.
        """
        if self._closed or not self._armed or task.cancelled():
            return
        try:
            result = task.result()
        except Exception:
            return
        if not getattr(result, "text", None):
            next_index = self._next_ahead(index)
            if next_index is not None:
                self._schedule(next_index)

    def _schedule_window(self, index: int) -> None:
        self._schedule(index)
        if not self._armed:
            return
        for item in self._ahead:
            if item > index:
                self._schedule(item)

    async def prepare(self, index: int) -> Any:
        """Await one planned candidate, scheduling the bounded window first."""
        self._schedule_window(index)
        task = self._tasks.get(index)
        if task is None:
            return ResolvedReference(None)
        timing = current_sync_timing.get()
        if task.done() and not task.cancelled() and timing is not None:
            # Already decoded before the loop reached it.
            timing.prefetch_reused += 1
        try:
            result = await task
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[sync] prefetched candidate failed: %s -> next", type(exc).__name__)
            result = ResolvedReference(None)
        self._consumed = max(self._consumed, index)
        if self._armed:
            next_index = self._next_ahead(index)
            if next_index is not None:
                self._schedule(next_index)
        return result

    async def try_prepare(self, index: int, timeout: float) -> Any:
        """Prepare a candidate within ``timeout``; never cancels the download."""
        self._schedule_window(index)
        task = self._tasks.get(index)
        if task is None:
            return None
        if not task.done():
            done, _ = await asyncio.wait({task}, timeout=max(0.0, timeout))
            if not done:
                return None
        try:
            return task.result()
        except Exception:
            return None

    async def close(self) -> None:
        """Cancel speculative work and account for it truthfully."""
        if self._closed:
            return
        self._closed = True
        pending = [(index, task) for index, task in self._tasks.items() if not task.done()]
        for _, task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*(task for _, task in pending), return_exceptions=True)
        timing = current_sync_timing.get()
        if timing is None:
            return
        for index, task in self._tasks.items():
            if index <= self._consumed:
                continue
            if task.cancelled():
                timing.prefetch_cancelled += 1
                continue
            try:
                result = task.result()
            except Exception:
                continue
            if getattr(result, "text", None):
                timing.prefetch_wasted += 1


def _lookup_video_reference(strategy_obj: Any, query: Any, timing: Any) -> Any:
    """Return a reusable video-scoped reference artifact, or ``None``.

    Revalidation is deliberately limited to what makes an artifact usable
    evidence AT ALL -- identity binding (already enforced by the cache key),
    payload integrity (enforced by the cache), and independence from the target.
    It is NOT a timing check. A REMUX reference is *expected* to disagree with
    an HDTV subtitle; that disagreement is precisely what Auto-Sync exists to
    correct, so requiring prior agreement here would reject exactly the
    reference the user asked for. Synchronization quality remains the sole
    authority of ALASS / affine / target binding / the final verifier.

    A rejected artifact is skipped for this request only and is never deleted:
    failing one subtitle says nothing about whether it is valid evidence for
    another.
    """
    cache = getattr(strategy_obj, "cache", None)
    getter = getattr(cache, "get_video_scoped", None)
    if getter is None:
        return None
    try:
        found = getter(query)
    except Exception:  # pragma: no cover - a cache must never break sync
        logger.debug("[sync] video-scoped lookup failed", exc_info=True)
        return None
    if not found:
        timing.note_reference_failure("cache_miss_video_reference")
        return None
    resolved = found[0]
    if not resolved.text:
        timing.note_reference_failure("cached_reference_validation_failure")
        return None
    from app.services.sync.reference_content_identity import identical_content

    reason = identical_content(getattr(query, "target_text", None), resolved.text)
    if reason:
        logger.warning(
            "[sync] video-scoped artifact rejected: reason=%s "
            "reason_code=reference_content_identical_to_target candidate=%s",
            reason, resolved.candidate or "?",
        )
        timing.note_reference_failure("reference_content_identical_to_target")
        return None
    timing.note_reference_failure("cache_hit_video_reference")
    logger.info(
        "[sync] reusing video-scoped reference artifact: provider=%s candidate=%s "
        "bytes=%d; provider discovery skipped for this video",
        resolved.provider or "?", resolved.candidate or "?",
        len(resolved.text.encode("utf-8")),
    )
    return resolved


def decision_kind_is_name_based(kind: str | None) -> bool:
    """True when a reference's provenance is similarity, not exact identity.

    ``hash_reference.py`` emits ``kind="hash"`` and that kind is only reached
    after OpenSubtitles returned ``attributes.moviehash_match is True`` for the
    hash the request supplied. Every other kind ("edition", title-based,
    provider default) rests on filename or title similarity.
    """
    return (kind or "").strip().lower() not in _HASH_DERIVED_REFERENCE_KINDS


def _content_digest(data: bytes) -> str:
    """Short SHA-256 of a payload, for delivery logging.

    Digests only. Subtitle text and credentials must never reach a log line, and
    a digest is enough to answer the question a delivery log exists for: was the
    artifact served the provider's own bytes, or a transformed one?
    """
    return hashlib.sha256(data or b"").hexdigest()[:16]


def _normalize_fingerprint(value: Any) -> str:
    """Lowercase, whitespace-collapsed fingerprint for cache-key scoping."""
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _stable_stream_identity(url: str) -> str:
    """Scheme, host and path of a stream URL; query and fragment dropped."""
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url
    if not parts.netloc:
        return url.split("?", 1)[0].split("#", 1)[0]
    return f"{parts.scheme}://{parts.netloc}{parts.path}"


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
        # Only the stable part identifies the video: debrid/AIOStreams sign each
        # request with a fresh query token, which would otherwise give every
        # request its own key and defeat both the payload cache and in-flight
        # coalescing (see the same exclusion in ``query.py``).
        stream_digest = hashlib.sha256(
            _stable_stream_identity(str(meta["stream_url"])).encode()
        ).hexdigest()[:16]
        fingerprint = f"{fingerprint}:stream-{stream_digest}"
    if not fingerprint:
        fingerprint = _normalize_fingerprint(meta.get("target_filename")) or _normalize_fingerprint(
            target_id
        )
    key = SyncCache.build_key(imdb_id, season_ep, fingerprint, target_id, decision, content_hash)
    return f"{key}:timing-v3" if unified_authority_enabled(meta) else key


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
        hash_reference_strategy: Any | None = None,
        external_strategy: Any | None = None,
        sync_service: Any | None = None,
        sync_cache: Any | None = None,
    ) -> None:
        self._hash_strategy = hash_strategy
        self._hash_reference_strategy = hash_reference_strategy
        self._external_strategy = external_strategy
        self._sync_service = sync_service
        self._sync_cache = sync_cache
        self._inflight: dict[str, asyncio.Future] = {}
        self._inflight_lock = asyncio.Lock()
        # Verification layer: measures alass output instead of trusting exit 0.
        self._analyzer = AlignmentAnalyzer()
        self._boundary_shadow = BoundaryShadow()
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
            "large_offset_cache_hits": 0,
            "large_offset_content_cache_hits": 0,
        }

    def _strategies(self) -> list[tuple[str, Any]]:
        strats = []
        # A byte-exact MovieHash reference outranks everything the tiered search
        # can offer, so it is consulted first. It is independent of the normal
        # OpenSubtitles provider toggle: that governs Stremio listings, while
        # this governs whether auto-sync may borrow a proven-exact reference.
        # Returning no text simply falls through to the next strategy.
        if self._hash_reference_strategy is not None:
            strats.append(("opensubtitles moviehash", self._hash_reference_strategy))
        if self._external_strategy is not None:
            # Deliberately NOT called "exact-match". This strategy's certainty is
            # entirely a function of what it resolved: kind="hash" means proven
            # exact identity, while kind="edition" is only a release-name
            # guess. A single strategy label implied the stronger of the two
            # for every candidate, which made a `decision=edition` reference
            # read as an exact match in the live trace. The per-candidate
            # `decision=%s` in the log line below carries the real provenance.
            strats.append(("external-release-reference", self._external_strategy))
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
            target_text=target_text,
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
            target_lang=(str(meta.get("lang") or "").strip().lower() or None),
        )

    async def evaluate_and_sync(
        self, sub_bytes: bytes, meta: dict, target_id: str, auto_sync: bool = False
    ) -> bytes:
        """Evaluate sync need and synchronize a subtitle, or return the original bytes.

        Language-agnostic: the target may be any language and the reference may
        be a different one. The reference supplies timing only.
        """
        # Tag every sync log line emitted from here with a correlation id.
        # Concurrent requests for the same episode otherwise interleave, and
        # attributing their lines to the wrong request is how an incident
        # appeared to show contradictory diagnostics that were not.
        request_token = set_request_id(stable_request_id(meta, target_id, sub_bytes))
        try:
            result = await self._evaluate_and_sync(
                sub_bytes, meta, target_id, auto_sync
            )
            if unified_authority_enabled(meta) and auto_sync:
                from dataclasses import replace

                if isinstance(result, DeliveryPayload):
                    return DeliveryPayload(result, replace(result.authorization, original=sub_bytes,
                        request_id=stable_request_id(meta, target_id, sub_bytes)))
                return DeliveryPayload(result, DeliveryAuthorization(
                    stable_request_id(meta, target_id, sub_bytes), None, None,
                    "unknown", "UNVERIFIED", False, sub_bytes))
            return result
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
        # No language gate here. This used to reject anything not starting with
        # "ar", which was a product-scope decision for an Arabic-only add-on
        # rather than a technical requirement: alass aligns any two subtitle
        # tracks, and the reference may be in a different language from the
        # target. Language-specific transforms downstream stay conditional on the
        # subtitle's actual language.
        if not lang:
            logger.info("[sync] skipped: subtitle language is unknown; nothing to align")
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
                    self._execute_job(sub_bytes, meta, target_id, auto_sync)
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
        await self._boundary_shadow.close()

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
        # An explicit sync request bypasses the metadata gate; joining an
        # identical non-explicit job in flight would hand it that job's
        # gate-skipped original instead.
        explicit = bool(meta.get("force_sync") or meta.get("explicit_sync_request"))
        flight = f"{key}:{SYNC_PAYLOAD_VERSION}:{strict}:{context_digest}:{auth_digest}"
        return f"{flight}:explicit" if explicit else flight

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
            fingerprint, subtitle_hash, str(meta.get("lang") or "und"),
            methodology="timing-v3" if unified_authority_enabled(meta) else None,
        )

    def _reference_cache(self) -> Any | None:
        """Shared on-disk reference store backing sticky promotion/lookup, if any.

        Prefers the external strategy's store; any strategy cache exposing
        the sticky tier works, since all of them share the reference root.
        """
        for _, strategy in self._strategies():
            cache = getattr(strategy, "cache", None)
            if (cache is not None
                    and callable(getattr(cache, "get_sticky_reference", None))
                    and callable(getattr(cache, "set_sticky_reference", None))):
                return cache
        return None

    def _local_timing_witness(
        self, query: Any, meta: dict
    ) -> tuple[Any | None, str | None]:
        """Best *local* reference for this exact video, or ``(None, None)``.

        Cache-only on purpose: the Metadata Gate must not turn a metadata-based
        decision into a provider request just to look for evidence. Tiers follow
        the Phase 5 evidence order (sticky, then the subtitle/video artifact
        tiers, then a retained season pack), and every lookup fails closed to
        "unavailable".
        """
        cache = self._reference_cache()
        if cache is not None:
            getter = getattr(cache, "get_sticky_reference", None)
            if callable(getter):
                try:
                    methodology = (
                        ("unified", "timing-v3") if unified_authority_enabled(meta)
                        else ("legacy", "large-offset-v1")
                    )
                    sticky = getter(query, methodology=methodology)
                except Exception:  # pragma: no cover - cache must never break sync
                    sticky = None
                if sticky is not None and getattr(sticky, "text", None):
                    return sticky, "sticky"
            getter = getattr(cache, "get", None)
            if callable(getter):
                try:
                    cached = getter(query)
                except Exception:  # pragma: no cover - cache must never break sync
                    cached = None
                if cached is not None and getattr(cached, "text", None):
                    return cached, "subtitle_cache"
            getter = getattr(cache, "get_video_scoped", None)
            if callable(getter):
                try:
                    found = getter(query)
                except Exception:  # pragma: no cover - cache must never break sync
                    found = None
                if found:
                    artifact = found[0]
                    if getattr(artifact, "text", None):
                        return artifact, "video_cache"
        try:
            from app.services.sync.season_pack import season_pack_reference

            packed = season_pack_reference(query)
        except Exception:  # pragma: no cover - cache must never break sync
            packed = None
        if packed is not None and getattr(packed, "text", None):
            return packed, "season_pack"
        return None, None

    def _timing_witness_decision(
        self, target_text: str | None, query: Any, meta: dict, target_id: str
    ) -> TimingWitnessDecision:
        """Measure a local reference to tell metadata compatibility from timing.

        Bounded by construction: the witness is a reference already on disk and
        the measurement is one pass over two local documents -- no provider
        request, no download, no ALASS. The elapsed cost is reported as
        ``metadata_gate_timing_ms``.

        Only a *sufficient* measurement may contradict the metadata verdict:
        the witness must pair at least half the target's dialogue across the
        timeline before a disagreement is treated as evidence. Absence of
        correspondence, too few matched cues, or an unreadable artifact keep the
        original (item 6: unavailable evidence is never a contradiction).
        """
        if not target_text:
            return TimingWitnessDecision(False, "timing_witness_insufficient")
        witness, source = self._local_timing_witness(query, meta)
        if witness is None:
            return TimingWitnessDecision(False, "timing_witness_unavailable")
        witness_text = getattr(witness, "text", None)
        if not isinstance(witness_text, str) or not witness_text:
            return TimingWitnessDecision(False, "timing_witness_insufficient", source)
        try:
            consistency = measure_alignment_consistency(
                target_text, witness_text, is_dialogue=is_dialogue_cue)
        except Exception as exc:  # pragma: no cover - measurement must not break sync
            logger.info(
                "[sync] timing witness measurement failed for sub=%s: %s",
                target_id, type(exc).__name__,
            )
            return TimingWitnessDecision(False, "timing_witness_insufficient", source)
        if consistency is None:
            return TimingWitnessDecision(False, "timing_witness_insufficient", source)
        # Evidence quality floor, reused rather than re-invented: the same
        # coverage and section requirements the already-aligned claim uses.
        from app.services.subtitle_matcher import (
            ALIGNED_MIN_SECTIONS,
            ALIGNED_SECTION_MIN_COVERAGE,
        )

        sufficient = (
            consistency.coverage >= ALIGNED_SECTION_MIN_COVERAGE
            and consistency.sections_measured >= ALIGNED_MIN_SECTIONS
        )
        if not sufficient:
            return TimingWitnessDecision(
                False, "timing_witness_insufficient", source,
                (consistency.median_offset_s or 0.0) * 1000.0)
        if consistency.aligned:
            return TimingWitnessDecision(
                False, "timing_witness_aligned", source,
                (consistency.median_offset_s or 0.0) * 1000.0)
        return TimingWitnessDecision(
            True, "timing_witness_conflict", source,
            (consistency.median_offset_s or 0.0) * 1000.0)

    def _promote_sticky_reference(
        self, *, meta: dict, target_id: str | None, pristine_text: str | None,
        resolved: Any | None, evaluation: Any | None,
        serving_state: str | None = None,
    ) -> None:
        """Record a verified, deliverable reference as this video's sticky first choice.

        Called only on TRANSFORMED serve paths, after identity, distinctness,
        Timing-First verification, reconstruction, binding, may_serve, and
        delivery-eligibility all passed. Last verified serve wins
        (replacement); failures never reach this method, so one bad request
        can neither erase nor oscillate the preference. Every later reuse
        still re-verifies, re-checks binding, and re-checks HTTP integrity.

        Only independently verified, strongly/exactly bound results can
        promote preference. Guarded unverified delivery remains non-promotable.

        Never raises and never affects the serve in progress.
        """
        try:
            if evaluation is None or resolved is None:
                return
            state = str(getattr(getattr(evaluation, "sync_state", ""), "value", "") or "")
            if (not unified_authority_enabled(meta)
                    or state not in {"verified_synced", "verified_resynced"}
                    or getattr(evaluation, "target_binding", None) not in {"exact", "strong"}
                    or getattr(evaluation, "verification", None) is not VerificationAvailability.VERIFIED
                    or getattr(resolved, "partial", False)):
                return
            methodology = ("unified", "timing-v3")
            text = getattr(resolved, "text", None)
            if not text:
                return
            from app.services.sync.reference_content_identity import identical_content

            if identical_content(pristine_text, text):
                return
            query = self._build_query(meta, target_id, pristine_text)
            from app.services.sync.season_pack import promote_season_pack

            promote_season_pack(query, resolved)
            cache = self._reference_cache()
            if cache is None:
                return
            from app.services.sync_service import sanitize_subtitle

            sanitized = sanitize_subtitle(text)
            if not sanitized:
                return
            evidence = {
                "sync_state": state,
                "target_binding": evaluation.target_binding,
                "serving_state": serving_state,
                "verification": str(
                    getattr(getattr(evaluation, "verification", ""), "value", "")
                    or ""
                ),
                "median_offset_ms": getattr(evaluation, "median_offset_ms", None),
                "p95_offset_ms": getattr(evaluation, "p95_offset_ms", None),
            }
            cache.set_sticky_reference(
                query,
                provider=getattr(resolved, "provider", None),
                candidate=getattr(resolved, "candidate", "") or "",
                kind=getattr(resolved, "kind", "edition") or "edition",
                text=sanitized,
                member_season=getattr(resolved, "member_season", None),
                member_episode=getattr(resolved, "member_episode", None),
                score=getattr(resolved, "weighted_score", 0.0),
                score_without_hash=getattr(resolved, "weighted_score_without_hash", 0.0),
                source_download_digest=getattr(resolved, "source_download_digest", None),
                bluray_match=bool(getattr(resolved, "bluray_match", False)),
                partial=bool(getattr(resolved, "partial", False)),
                evidence=evidence,
                methodology=methodology,
            )
        except Exception as exc:
            logger.debug("[reference-sticky] promotion skipped: %s", type(exc).__name__)

    @staticmethod
    def _guarded_large_offset_meta_authorized(
        meta_record: dict[str, Any] | None, cached: bytes
    ) -> bool:
        """Validate the authorization/provenance fields of a corrected artifact."""
        record = meta_record or {}
        if record.get("serving_state") != (
            LargeOffsetServingState.ALASS_CORRECTED_LARGE_OFFSET.value
        ):
            return False
        if record.get("large_offset_authorization") != (
            LARGE_OFFSET_CACHE_AUTHORIZATION_VERSION
        ):
            return False
        if record.get("sync_state") != SyncState.UNVERIFIED.value:
            return False
        if record.get("verification") not in {
            VerificationAvailability.UNKNOWN.value,
            VerificationAvailability.CACHED.value,
        }:
            return False
        if record.get("target_binding") not in {"exact", "strong"}:
            return False
        if not re.fullmatch(r"[a-f0-9]{64}", str(record.get("reference_sha") or "")):
            return False
        if record.get("artifact_sha256") != hashlib.sha256(cached).hexdigest():
            return False
        return True

    @staticmethod
    def _guarded_large_offset_cached_serving(
        meta_record: dict[str, Any] | None, cached: bytes, *, meta: dict
    ) -> bool:
        """Whether stored bytes may be replayed under the Large-Offset guard.

        This is deliberately separate from ``is_reusable_verified``. The
        corrected artifact stays ``UNVERIFIED``: a cache hit must never turn it
        into a verified claim. It may only be served again because the *same
        guarded delivery decision* was already made for this exact request
        identity, and every input that decision depended on is still recorded:

        * the recorded serving state is ``alass_corrected_large_offset``
          (the only state the guard authorises);
        * the stored verdict is still unverified/unknown, so replay cannot
          upgrade it;
        * target binding is exact/strong and a full reference digest exists;
        * the cached bytes are exactly the approved artifact;
        * a guard-generation marker matches this code.

        Unified authority is excluded because the Large-Offset override is
        fenced off there; its verified-result reuse path is unchanged.
        """
        if unified_authority_enabled(meta):
            return False
        return SyncOrchestrator._guarded_large_offset_meta_authorized(
            meta_record, cached
        )

    @staticmethod
    def _content_artifact_key(meta: dict, content_hash: str) -> str:
        """Content-addressed key for a guarded Large-Offset artifact.

        The subtitle ID is replaced by a language-scoped constant, so two
        subtitle IDs with byte-identical original content share one entry. The
        video fingerprint and content hash still isolate video, release,
        language and payload.
        """
        credentials = tuple(str(meta.get(name) or "") for name in (
            "subdl_key", "subsource_key", "opensubtitles_key"
        ))
        auth_digest = hashlib.sha256(repr(credentials).encode()).hexdigest()[:12]
        strict = "strict" if getattr(settings, "SYNC_REQUIRE_EXACT_MATCH", True) else "loose"
        size = re.sub(r"[^0-9]", "", str(meta.get("video_size") or "")) or "x"
        return build_synced_cache_key(
            meta,
            (
                f"content-{str(meta.get('lang') or 'und')}-{size}-"
                f"{auth_digest}-{strict}"
            ),
            content_hash=content_hash,
            decision=CONTENT_ARTIFACT_DECISION,
        )

    async def _load_guarded_content_artifact(
        self, meta: dict, content_hash: str
    ) -> tuple[bytes, dict[str, Any], str] | None:
        """Load a guarded corrected artifact for this video + original content.

        Every authorization field is re-checked exactly as for the same-subtitle
        payload path. The result remains UNVERIFIED/UNKNOWN; this never promotes
        trust or bypasses the verifier (the artifact was already authorized by
        the Large-Offset guard for this exact video and byte-identical content).
        """
        if self._sync_cache is None or unified_authority_enabled(meta):
            return None
        key = self._content_artifact_key(meta, content_hash)
        try:
            artifact = await self._sync_cache.get(key)
            if not artifact:
                return None
            record = await self._sync_cache.get_meta(key)
        except Exception:  # pragma: no cover - cache must not break sync
            return None
        if not self._guarded_large_offset_meta_authorized(record, artifact):
            return None
        if (record or {}).get("original_content_hash") not in (None, content_hash):
            return None
        return artifact, record or {}, key

    @staticmethod
    def _guarded_replay_meta(verdict: dict[str, Any]) -> dict[str, Any]:
        """Publication-safe metadata for a reused corrected artifact."""
        return {
            "sync_state": verdict.get("sync_state"),
            "verification": verdict.get("verification"),
            "sync_confidence": verdict.get("sync_confidence"),
            "serving_state": verdict.get("serving_state"),
            "large_offset_authorization": verdict.get("large_offset_authorization"),
            "target_binding": verdict.get("target_binding"),
            "target_binding_reasons": list(verdict.get("target_binding_reasons") or []),
            "artifact_sha256": verdict.get("artifact_sha256"),
            "reference_sha": verdict.get("reference_sha"),
            "original_content_hash": verdict.get("original_content_hash"),
            "verification_method": verdict.get("verification_method"),
            "verification_version": verdict.get("verification_version"),
        }

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
            target_binding=verdict.get("target_binding"),
            target_binding_reasons=list(verdict.get("target_binding_reasons") or []),
            median_offset_ms=verdict.get("median_offset_ms"),
            p95_offset_ms=verdict.get("p95_offset_ms"),
            mad_offset_ms=verdict.get("mad_offset_ms"),
            drift_ms_per_minute=verdict.get("drift_ms_per_minute"),
            coverage_score=verdict.get("coverage_score"),
            structural_similarity=verdict.get("structural_similarity"),
            cut_verdict=verdict.get("cut_verdict"),
        )
        return evaluation

    @staticmethod
    def _delivery_payload(payload: bytes, original: bytes, evaluation: SubtitleEvaluation,
                          meta: dict, target_id: str, reference_sha: str | None) -> bytes:
        if not unified_authority_enabled(meta):
            return payload
        return DeliveryPayload(payload, DeliveryAuthorization(
            stable_request_id(meta, target_id, original), hashlib.sha256(payload).hexdigest(),
            reference_sha, str(evaluation.target_binding or "unknown"), "VERIFIED",
            may_serve_synchronized(evaluation.sync_state, evaluation.verification), original))

    @staticmethod
    def _reusable_rejection(verdict: dict, *, historical: bool = False) -> bool:
        if verdict.get("sync_state") == SyncState.UNVERIFIED.value:
            return True  # Existing exact-request measured-negative behavior.
        if verdict.get("sync_state") != SyncState.REJECTED.value:
            return False
        # An explicit identity/safety rejection applies to this content-bound
        # request. Candidate output damage or timing-model refusal does not
        # forbid a genuinely new independent candidate/methodology.
        scope = verdict.get("rejection_scope")
        if scope in {"identity", "safety", "request"}:
            return True
        if scope in {"candidate", "methodology"}:
            return not historical and scope == "methodology"
        return verdict.get("rejection_reason") not in {
            "cue_loss", "implausible_offset", "low_confidence", "structure_mismatch",
            "insufficient_evidence", "alass_failed",
        }

    @staticmethod
    def _stale_unified_integrity(verdict: dict) -> bool:
        """True for pre-reconstruction Unified dialogue-integrity negatives.

        Those verdicts measured sanitized-based ALASS bytes against the raw
        target, so reusing them would suppress the corrected (reconstructed)
        artifact forever. Only unmarked verdicts are stale: the fixed pipeline
        tags every Unified store with ``unified_pipeline: v2``, after which
        normal negative reuse resumes (at most one extra evaluation per stale
        entry, then the fresh verdict governs).
        """
        if verdict.get("verification_method") != "unified":
            return False
        if verdict.get("unified_pipeline") == "v2":
            return False
        reasons = verdict.get("reasons") or []
        return any("dialogue_text_modified" in str(reason) for reason in reasons)

    @staticmethod
    def _stale_unified_generation(verdict: dict) -> bool:
        """True for Unified negatives measured before verification leniency.

        Covers every negative shape (parse aborts, timing refusals, wasted
        runs): all predate the current verification-input boundary, so each
        stale key costs at most one extra evaluation, after which its fresh
        marked verdict governs. Positives are never stale (verified artifacts
        keep their bytes and digest checks).
        """
        if verdict.get("verification_method") != "unified":
            return False
        if verdict.get("verification_input") == "lenient-v1":
            return False
        return verdict.get("sync_state") in (
            SyncState.UNVERIFIED.value, SyncState.REJECTED.value,
        )

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

    @staticmethod
    def _record_outcome(meta: dict | None, outcome: str) -> None:
        """Record one terminal request outcome for per-title telemetry.

        Exactly one call per request, at the point the final bytes are decided.
        Telemetry must never break serving, so every failure here is swallowed.
        """
        try:
            SYNC_OUTCOMES.record(
                (meta or {}).get("imdb_id"),
                (meta or {}).get("season"),
                (meta or {}).get("episode"),
                outcome,
            )
        except Exception:  # pragma: no cover - telemetry must never break sync
            pass

    async def _store_verdict(
        self,
        verdict_key: str | None,
        evaluation: SubtitleEvaluation,
        content_hash: str,
        meta: dict | None = None,
        target_id: str | None = None,
        artifact_key: str | None = None,
        artifact_bytes: bytes | None = None,
        reference_sha: str | None = None,
        serving_state: str | None = None,
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
                alias_key = SyncCache.build_verdict_alias_key(
                    fingerprint, target_id,
                    methodology="timing-v3" if unified_authority_enabled(meta) else None,
                )
        payload = evaluation.model_dump(mode="json")
        # Content/provenance fields, not Unified-only claims. They are required
        # to validate a guarded Large-Offset artifact reused by a different
        # subtitle ID carrying byte-identical original content.
        if reference_sha:
            payload["reference_sha"] = reference_sha
        if artifact_bytes is not None:
            payload["artifact_sha256"] = hashlib.sha256(artifact_bytes).hexdigest()
        if serving_state:
            payload["serving_state"] = serving_state
            if serving_state == LargeOffsetServingState.ALASS_CORRECTED_LARGE_OFFSET.value:
                payload["large_offset_authorization"] = (
                    LARGE_OFFSET_CACHE_AUTHORIZATION_VERSION
                )
        if meta and unified_authority_enabled(meta):
            payload.update(verification_method="unified", verification_version="timing-v3")
            # Candidate-provenance generation and verification-input boundary:
            # both markers let stale Unified negatives re-evaluate exactly once
            # instead of suppressing the corrected artifact forever. Positives
            # are unaffected.
            payload["unified_pipeline"] = "v2"
            payload["verification_input"] = "lenient-v1"
        if artifact_key:
            payload["artifact_key"] = artifact_key
        await self._sync_cache.set_verdict(
            verdict_key, payload, alias_key=alias_key
        )

    async def _select_evaluation(
        self, legacy: SubtitleEvaluation, *, target_text: str, reference_text: str,
        candidate_text: str, meta: dict, algorithm: str,
        resolved: ResolvedReference | None = None,
    ) -> SubtitleEvaluation:
        """Select correctness before authorization or caching; never rewrite Legacy."""
        if not unified_authority_enabled(meta):
            return legacy
        started = time.monotonic()
        evaluation, telemetry = await unified_review_async(
            target_text=target_text, reference_text=reference_text,
            synced_text=candidate_text, candidate_algorithm=algorithm,
        )
        _timing = current_sync_timing.get()
        if _timing is not None:
            _timing.verification_total_ms += (time.monotonic() - started) * 1000
        logger.info("[verification] algorithm=%s evidence=%s", algorithm, telemetry)
        # Keep the existing withhold-only ReferencePolicy ceiling.
        trust = resolved.reference_trust if resolved else legacy.reference_trust
        evaluation.reference_trust = trust
        reference_evidence = resolved or legacy
        if resolved:
            evaluation.reasons.extend(f"reference: {reason}" for reason in resolved.reference_reasons or [])
        evaluation.reference_consensus = reference_evidence.reference_consensus
        evaluation.reference_independent_sources = reference_evidence.reference_independent_sources
        evaluation.reference_failure = reference_evidence.reference_failure
        # Trust/ranking labels remain provenance. The independently matched
        # dialogue decides timing; identity is enforced at the delivery boundary.
        self._last_evaluation = evaluation
        return evaluation

    def _legacy_diagnostic(self, meta: dict, *args: Any, **kwargs: Any) -> SubtitleEvaluation:
        if unified_authority_enabled(meta):
            return SubtitleEvaluation()
        try:
            started = time.monotonic()
            evaluation = self._analyzer.analyze(*args, **kwargs)
            timing = current_sync_timing.get()
            if timing is not None:
                timing.verification_total_ms += (time.monotonic() - started) * 1000
            return evaluation
        except Exception as exc:
            logger.info("[unified] legacy_diagnostic_error=%s", type(exc).__name__)
            return SubtitleEvaluation(reasons=[f"legacy_diagnostic_error:{type(exc).__name__}"])

    def _maybe_arm_prefetch(
        self,
        *,
        current_index: int,
        current_text: str,
        pending: list[int],
        prefetcher: _PlannedPrefetcher,
        query: ReferenceQuery,
        timing: SyncTiming,
    ) -> None:
        """Start lookahead only when it has evidence and a useful target.

        A normal candidate that verifies immediately must still cost exactly one
        download. Speculation is armed only when all of the following hold:

        * the current reference is a large-offset candidate under the existing
          Large Offset Evidence Gate (measured on the sanitized text ALASS
          actually receives);
        * the plan contains a same-release-family sibling within the bounded
          lookahead.

        The gate is cheap (~50-70 ms) relative to the ALASS/affine evaluation it
        precedes, and it is skipped entirely when no sibling exists.
        """
        if prefetcher.armed or not pending:
            return
        family = prefetcher.family(current_index)
        if not family:
            return
        window = pending[: max(1, prefetcher.lookahead)]
        if not any(prefetcher.family(index) == family for index in window):
            return
        # A sibling advertised under the same normalized release name and
        # language is not worth downloading speculatively: it is very likely
        # the same mirrored file. Keep it in the plan for fallback, but spend
        # the speculative slot on a genuinely distinct release.
        current_identity = prefetcher.identity(current_index)
        ahead = [
            index
            for index in window
            if prefetcher.family(index) == family
            and prefetcher.identity(index) != current_identity
        ]
        if not ahead:
            return
        started = time.monotonic()
        try:
            from app.services.sync.reference_preflight import measure_reference_timing

            evidence = measure_reference_timing(
                query.target_text, current_text, family=family
            )
            timing.preflight_runs += 1
            prefetcher.set_evidence(current_index, evidence)
            if evidence is not None and evidence.large_offset and evidence.accepted:
                prefetcher.arm_ahead(current_index, ahead)
                logger.info(
                    "[sync] large-offset preflight armed bounded reference "
                    "preparation for candidate %d (offset %.2fs, dispersion %.0fms)",
                    current_index + 1,
                    (evidence.offset_ms or 0) / 1000.0,
                    evidence.dispersion_ms or 0.0,
                )
        except Exception as exc:  # pragma: no cover - preflight is best effort
            logger.debug("[sync-preflight] trigger unavailable: %s", type(exc).__name__)
        finally:
            timing.preflight_ms += (time.monotonic() - started) * 1000

    async def _preferred_sibling(
        self,
        *,
        current_index: int,
        current_text: str,
        pending: list[int],
        prefetcher: _PlannedPrefetcher,
        query: ReferenceQuery,
        timing: SyncTiming,
    ) -> int | None:
        """Return a same-family sibling that should be tried first, or ``None``.

        This is a prioritisation only. It requires the existing Large Offset
        Evidence Gate to accept BOTH references on the sanitized text ALASS
        receives, their independently estimated offsets to agree, and the
        sibling's anchor dispersion to be materially tighter. It never accepts,
        excludes, or transforms anything; the sibling is moved in front of the
        current candidate, which remains immediately behind it.
        """
        family = prefetcher.family(current_index)
        if not prefetcher.armed or not family or not pending:
            return None
        window = pending[: max(1, prefetcher.lookahead)]
        current_identity = prefetcher.identity(current_index)
        siblings = [
            index
            for index in window
            if index > current_index
            and prefetcher.family(index) == family
            and prefetcher.identity(index) != current_identity
        ]
        if not siblings:
            return None
        try:
            from app.services.sync.reference_preflight import (
                measure_reference_timing,
                sibling_is_materially_tighter,
            )
        except Exception:  # pragma: no cover - preflight must not break sync
            return None

        started = time.monotonic()
        current = prefetcher.get_evidence(current_index)
        if current is None:
            current = measure_reference_timing(
                query.target_text, current_text, family=family
            )
            prefetcher.set_evidence(current_index, current)
            timing.preflight_runs += 1
        try:
            if current is None or not current.accepted:
                return None
            agreement = int(
                getattr(settings, "REFERENCE_PREFLIGHT_OFFSET_AGREEMENT_MS", 750) or 750
            )
            wait_seconds = float(
                getattr(settings, "REFERENCE_PREFLIGHT_WAIT_SECONDS", 2.5) or 2.5
            )
            deadline = started + max(0.0, wait_seconds)
            best: tuple[int, float, Any] | None = None
            for index in siblings:
                budget = deadline - time.monotonic()
                if budget <= 0:
                    break
                resolved = await prefetcher.try_prepare(index, budget)
                if resolved is None or not getattr(resolved, "text", None):
                    continue
                # Never jump over an intervening candidate that is itself a
                # valid, different-family reference. That would be a genuine
                # ordering change beyond same-release prioritisation.
                blocked = False
                for between in range(current_index + 1, index):
                    if prefetcher.identity(between) == current_identity:
                        # A mirrored copy of the current release. It stays in
                        # the plan for fallback, but it is not independent
                        # evidence and must not block a distinct sibling.
                        continue
                    _done, _result = prefetcher.peek(between)
                    if not _done:
                        blocked = True
                        break
                    if getattr(_result, "text", None):
                        blocked = True
                        break
                if blocked:
                    continue
                sibling = measure_reference_timing(
                    query.target_text, resolved.text, family=prefetcher.family(index)
                )
                timing.preflight_runs += 1
                if sibling is None or not sibling_is_materially_tighter(
                    current, sibling, agreement_ms=agreement
                ):
                    continue
                dispersion = float(sibling.dispersion_ms or float("inf"))
                if best is None or dispersion < best[1]:
                    best = (index, dispersion, sibling)
            if best is None:
                return None
            logger.info(
                "[sync-preflight] preferring same-family candidate %d over %d "
                "(offset %.2fs/%.2fs agree within %dms; dispersion %.0fms -> %.0fms); "
                "deferred candidate stays next",
                best[0] + 1,
                current_index + 1,
                (current.offset_ms or 0) / 1000.0,
                (best[2].offset_ms or 0) / 1000.0,
                agreement,
                current.dispersion_ms or 0.0,
                best[1],
            )
            return best[0]
        except Exception as exc:  # pragma: no cover - preflight is best effort
            logger.debug("[sync-preflight] unavailable: %s", type(exc).__name__)
            return None
        finally:
            timing.preflight_ms += (time.monotonic() - started) * 1000

    async def _reference_attempts(
        self, attempts: list[tuple[str, Any]], query: ReferenceQuery,
        _passes_cue_sanity: Any, timing: SyncTiming,
        meta: dict | None = None,
    ) -> AsyncIterator[tuple[str, Any]]:
        """Expand each strategy only when execution reaches it; cache precedes planning."""
        # Baseline read these cache entries before executing MovieHash. Keep that
        # snapshot: hash resolution may write the shared disk cache even if sync fails.
        cache_hits: dict[int, Any] = {}
        for _, strategy in attempts:
            if id(strategy) in cache_hits:
                continue
            if not (callable(getattr(strategy, "plan_candidates", None))
                    or callable(getattr(strategy, "resolve_all_with_provenance", None))):
                continue
            cache_hits[id(strategy)] = None
            cache = getattr(strategy, "cache", None)
            if cache is not None:
                try:
                    cache_hits[id(strategy)] = cache.get(query)
                except Exception:  # pragma: no cover - cache must not break sync
                    pass
            hit = cache_hits[id(strategy)]
            timing.cached_reference_present |= bool(hit is not None and getattr(hit, "text", None))

        expanded: set[int] = set()
        # Season-pack member lookup is request-scoped (not per strategy):
        # the retained pack is shared evidence, so it is offered at most once.
        _pack_digest: list[str | None] = [None]
        # Sticky verified reference: one lookup per request, attempted before
        # every other tier. Identical bytes met again later in this request
        # are skipped via _sticky_digest (re-verifying them would only repeat
        # a verdict this request already measured).
        _sticky_digest: list[str | None] = [None]
        _sticky_cache = self._reference_cache()
        _sticky_hit = None
        if _sticky_cache is not None:
            _sticky_methodology = (
                ("unified", "timing-v3")
                if meta is None or unified_authority_enabled(meta)
                else ("legacy", "large-offset-v1")
            )
            try:
                _sticky_hit = _sticky_cache.get_sticky_reference(
                    query, methodology=_sticky_methodology
                )
            except Exception:  # lookup must never break sync
                _sticky_hit = None
        if _sticky_hit is not None and getattr(_sticky_hit, "text", None):
            _sticky_digest[0] = hashlib.sha256(
                str(_sticky_hit.text).encode("utf-8", "replace")
            ).hexdigest()
            timing.sticky_reference_attempted = True
            yield "retained-reference#sticky", _PreResolvedReference(_sticky_hit)
            sticky_source = getattr(_sticky_hit, "source_download_digest", None)
            if sticky_source:
                query.known_source_digests = _add_known_source(query, sticky_source)
        # Season-pack tier (once per request): a member retained from
        # an earlier same-lineage episode serves locally with no
        # provider search and no download. Episode-specific bytes
        # only; verification downstream is unchanged.
        try:
            from app.services.sync.season_pack import season_pack_reference

            _pack_hit = season_pack_reference(query)
        except Exception:  # lookup must never break sync
            _pack_hit = None
        if _pack_hit is not None and getattr(_pack_hit, "text", None):
            _pack_digest[0] = hashlib.sha256(
                str(_pack_hit.text).encode("utf-8", "replace")).hexdigest()
            timing.season_pack_reference_attempted = True
            yield "retained-reference#season_pack", _PreResolvedReference(_pack_hit)
            pack_digest = getattr(_pack_hit, "source_download_digest", None)
            if pack_digest:
                query.known_source_digests = _add_known_source(query, pack_digest)
        for _name, _obj in attempts:
            _plan = getattr(_obj, "plan_candidates", None)
            _multi = getattr(_obj, "resolve_all_with_provenance", None)
            if not callable(_plan) and not callable(_multi):
                yield _name, _obj
                continue
            if id(_obj) in expanded:
                continue
            expanded.add(id(_obj))
            # Preserve the strategy cache-hit shortcut BEFORE any fan-out. A
            # previously proven reference must still be reused without asking a
            # provider again, exactly as the single-reference path did.
            _hit = cache_hits.get(id(_obj))
            # A cache hit is NOT an exit. It used to be, which silently
            # disabled multi-candidate expansion for every repeat request --
            # i.e. exactly the Whiplash case, where a cached `subdl_edition`
            # reference meant one ALASS run and no alternative was ever tried.
            # The proven reference stays FIRST (it is still the best-informed
            # candidate); the planned releases then supply genuine alternatives,
            # each downloaded only if the loop actually reaches it.
            _shared: dict[str, Any] = {"downloaded": 0, "seen": set()}

            def _seed_reference_bytes(shared: dict[str, Any], text: Any) -> str | None:
                """Record an artifact already offered so planning cannot repeat it.

                The digest must be registered in *prepared* form as well: a
                provider copy of the same subtitle can differ only in packaging
                (BOM, line endings, trailing spaces), and the planned stream
                compares prepared identities.
                """
                if not text:
                    return None
                from app.services.sync.reference import prepared_reference_digest

                raw_digest = hashlib.sha256(
                    str(text).encode("utf-8", "replace")).hexdigest()
                shared["seen"].add(raw_digest)
                shared["seen"].add(prepared_reference_digest(str(text)))
                return raw_digest

            if _pack_digest[0]:
                _seed_reference_bytes(_shared, getattr(_pack_hit, "text", None))
            if _sticky_digest[0] is not None:
                # Bytes already attempted first this request: planned
                # releases must not re-offer them under another name.
                _seed_reference_bytes(_shared, getattr(_sticky_hit, "text", None))
            _planned: list[tuple[Any, Any]] = []
            has_plan = callable(_plan)
            cached_available = bool(_hit is not None and getattr(_hit, "text", None))
            if has_plan and cached_available:
                if _hit is not None and getattr(_hit, "text", None):
                    # Keep the cached reference first. Seed the shared digest set
                    # with its content so the planned releases cannot re-offer
                    # identical bytes under a different release name.
                    _hit_digest = hashlib.sha256(
                        str(_hit.text).encode("utf-8", "replace")).hexdigest()
                    _seed_reference_bytes(_shared, _hit.text)
                    if _hit_digest in {_sticky_digest[0], _pack_digest[0]}:
                        # Identical bytes were already attempted first via the
                        # sticky tier this request; re-verifying them here can
                        # only repeat that verdict, never improve it.
                        logger.info("[sync] %s cached reference skipped: identical "
                                    "bytes already attempted via sticky tier", _name)
                    else:
                        timing.cached_reference_attempted = True
                        yield f"{_name}#cache", _PreResolvedReference(_hit)
            _video_hit = None
            if has_plan:
                # Video-scoped artifact tier. The subtitle-scoped lookup above
                # only says "a reference was proven for THIS subtitle's cue
                # layout"; it can be a sampled/partial artifact. Which file to
                # use is a property of the video, so this tier is consulted even
                # when the subtitle tier hit: it may hold the complete artifact
                # for the same video. Identical bytes are skipped by digest
                # below, so the common case costs one local lookup.
                _video_hit = _lookup_video_reference(_obj, query, timing)
            cached_available = cached_available or bool(
                _video_hit is not None and getattr(_video_hit, "text", None)
            )
            if has_plan and _video_hit is not None:
                # Reuse the artifact, but do not make it a dead end. Provider
                # discovery is deferred until the consumer asks for a second
                # candidate, meaning a successful artifact pays no discovery or
                # download cost, while a failed one still permits fresh planning.
                # It is not an acceptance: the loop below still runs
                # ALASS/affine/binding and the final verifier on it exactly as
                # for a fresh download.
                from app.services.sync.reference import prepared_reference_digest

                if prepared_reference_digest(str(_video_hit.text)) in _shared["seen"]:
                    # Same bytes an earlier tier already offered this request
                    # (sticky, season pack or the subtitle-scoped cache); a
                    # second attempt can only repeat that verdict. The source
                    # exclusion below still applies.
                    logger.info("[sync] %s video-cache reference skipped: identical "
                                "bytes already offered", _name)
                else:
                    _seed_reference_bytes(_shared, _video_hit.text)
                    timing.cached_reference_attempted = True
                    yield f"{_name}#video_cache", _PreResolvedReference(_video_hit)
                source_digest = getattr(_video_hit, "source_download_digest", None)
                if source_digest:
                    # Do not spend a bounded plan slot fetching the same archive
                    # again if this artifact cannot serve the request. The raw
                    # address is never stored or logged; only its digest is used.
                    query.known_source_digests = _add_known_source(query, source_digest)
            if has_plan:
                timing.external_plan_started = True
                started = time.monotonic()
                try:
                    _planned = await cast(Any, _plan)(query)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "[sync] %s candidate planning failed: %s -> single reference",
                        _name, type(exc).__name__,
                    )
                    _planned = []
                finally:
                    timing.external_plan_ms = (time.monotonic() - started) * 1000
                timing.candidate_count_planned += len(_planned)
            if callable(_multi) and not has_plan:
                try:
                    _cands = await _multi(query, update_validator=_passes_cue_sanity)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "[sync] %s multi-candidate expansion failed: %s -> single reference",
                        _name, type(exc).__name__,
                    )
                    _cands = []
                for _i, _c in enumerate(_cands):
                    yield f"{_name}#candidate{_i + 1}", _PreResolvedReference(_c)
                if _cands:
                    logger.info(
                        "[sync] %s offered %d distinct candidate(s); each will be verified independently",
                        _name, len(_cands),
                    )
                else:
                    # The fan-out already ran and found nothing usable. Calling the
                    # strategy again here would repeat the identical provider search
                    # and reach the identical empty result, so the request simply
                    # proceeds with no reference and serves the original.
                    logger.info(
                        "[sync] %s produced no usable reference candidate; serving original",
                        _name,
                    )
                continue
            if not _planned and not cached_available:
                # No plan and no cache hit: nothing to try for this strategy.
                # (When a plan exists, emptiness after download failures is
                # discovered per candidate by the loop itself, each skip
                # carrying its own reason.)
                logger.info(
                    "[sync] %s produced no usable reference candidate; serving original",
                    _name,
                )
                continue
            logger.info(
                "[sync] %s planned %d candidate(s); each downloads only if reached "
                "and is verified independently",
                _name, len(_planned),
            )
            _prefetch_enabled = bool(
                getattr(settings, "REFERENCE_PREFETCH_ENABLED", True)
            )
            _preflight_enabled = bool(
                getattr(settings, "REFERENCE_PREFLIGHT_ENABLED", True)
            )
            _prefetcher: _PlannedPrefetcher | None = None
            if _prefetch_enabled and len(_planned) > 1:
                try:
                    _prefetcher = _PlannedPrefetcher(
                        strategy=_obj,
                        planned=_planned,
                        query=query,
                        shared=_shared,
                        validate=_passes_cue_sanity,
                        concurrency=max(
                            1,
                            int(getattr(settings, "REFERENCE_PREFETCH_CONCURRENCY", 2) or 2),
                        ),
                        lookahead=max(
                            0,
                            int(getattr(settings, "REFERENCE_PREFETCH_LOOKAHEAD", 2) or 2),
                        ),
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("[sync] prefetch unavailable: %s", type(exc).__name__)
                    _prefetcher = None
            if _prefetcher is None:
                for _i, (_rel, _prov) in enumerate(_planned):
                    yield (
                        f"{_name}#candidate{_i + 1}",
                        _LazyCandidate(
                            _obj, _rel, _prov, query, _shared, _passes_cue_sanity,
                            plan_index=_i + 1,
                        ),
                    )
                continue
            # Prepared candidates are yielded in plan order unless the bounded
            # preflight prefers an already-downloaded same-family sibling. The
            # deferred candidate is re-inserted immediately behind the
            # preferred one, so fallback is preserved exactly.
            try:
                _pending = list(range(len(_planned)))
                while _pending:
                    _i = _pending.pop(0)
                    _resolved = await _prefetcher.prepare(_i)
                    if getattr(_resolved, "text", None) and _preflight_enabled:
                        self._maybe_arm_prefetch(
                            current_index=_i,
                            current_text=_resolved.text,
                            pending=_pending,
                            prefetcher=_prefetcher,
                            query=query,
                            timing=timing,
                        )
                        _sibling = await self._preferred_sibling(
                            current_index=_i,
                            current_text=_resolved.text,
                            pending=_pending,
                            prefetcher=_prefetcher,
                            query=query,
                            timing=timing,
                        )
                        if _sibling is not None and _sibling in _pending:
                            _pending.remove(_sibling)
                            _pending.insert(0, _i)
                            _pending.insert(0, _sibling)
                            timing.preflight_reorders += 1
                            continue
                    # Empty results are yielded too: the orchestrator records
                    # the failure and advances exactly as it did with lazy
                    # adapters, while the prefetcher has already kept the next
                    # candidate warm.
                    yield f"{_name}#candidate{_i + 1}", _PreResolvedReference(_resolved)
            finally:
                # Cancel speculative downloads as soon as the request no
                # longer needs them (success, failure, or client disconnect).
                await _prefetcher.close()

    async def _execute_job(
        self, sub_bytes: bytes, meta: dict, target_id: str, auto_sync: bool
    ) -> bytes:
        """Account once per owned single-flight job, including early cache returns."""
        timing = SyncTiming(
            external_plan_available=callable(getattr(self._external_strategy, "plan_candidates", None))
        )
        token = current_sync_timing.set(timing)
        started = time.monotonic()
        result = "error"
        try:
            output = await self._execute(sub_bytes, meta, target_id, auto_sync)
            result = "cached" if timing.cache_hit else (
                "transformed" if output != sub_bytes else "original"
            )
            if result == "transformed":
                if timing.winning_sync_strategy != "two_point_linear_pal":
                    timing.winning_candidate_index = timing.candidates_considered
                if timing.winning_sync_strategy is None:
                    timing.winning_sync_strategy = "alass"
            capture_output(output, timing.winning_sync_strategy, timing.cache_hit)
            return output
        except asyncio.CancelledError:
            result = "cancelled"
            raise
        finally:
            attempt_iterator = getattr(timing, "active_attempt_iterator", None)
            if attempt_iterator is not None:
                try:
                    await attempt_iterator.aclose()
                except Exception:  # pragma: no cover - cleanup must not mask result
                    pass
                timing.active_attempt_iterator = None  # type: ignore[attr-defined]
            timing.log(result, (time.monotonic() - started) * 1000)
            current_sync_timing.reset(token)

    async def _execute(
        self, sub_bytes: bytes, meta: dict, target_id: str, auto_sync: bool
    ) -> bytes:
        """The expensive pipeline body: strategies, cache, alass (see evaluate_and_sync)."""
        timing = current_sync_timing.get()
        if timing is None:
            timing = SyncTiming()
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
                # Guarded Large-Offset replay. The artifact deliberately stays
                # UNVERIFIED, so it cannot use the verified fast path above --
                # and the unverified verdict below would otherwise serve the
                # *original* on a retry, losing a correction the user already
                # confirmed. Replay is allowed only for the exact guarded
                # delivery decision recorded for this request identity.
                if self._guarded_large_offset_cached_serving(
                    meta_record, cached, meta=meta
                ):
                    evaluation = self._evaluation_from_verdict(meta_record or {})
                    self._last_evaluation = evaluation
                    self._metrics["verification_cache_hits"] += 1
                    self._metrics["large_offset_cache_hits"] += 1
                    timing.cache_hit = True
                    timing.winning_sync_strategy = "large_offset_cache"
                    logger.info(
                        "[sync-cache] layer=payload result=hit_guarded_large_offset "
                        "sub_id=%s process=%s content_hash=%s key_digest=%s "
                        "sync_state=%s verification=%s serving_state=%s "
                        "large_offset_authorization=%s",
                        target_id,
                        _PROCESS_INSTANCE_ID,
                        content_hash,
                        digest,
                        stored_state,
                        stored_verification,
                        (meta_record or {}).get("serving_state"),
                        LARGE_OFFSET_CACHE_AUTHORIZATION_VERSION,
                    )
                    self._audit_serve(
                        evaluation, meta, target_id, from_cache=True, resolved=None
                    )
                    # Publish the same content-addressed twin from a same-subtitle
                    # replay too, so a later different subtitle ID with identical
                    # bytes can reuse it even if this process never executed the
                    # original ALASS run.
                    if self._sync_cache is not None:
                        content_key = self._content_artifact_key(meta, content_hash)
                        replay_meta = {
                            **self._guarded_replay_meta(meta_record or {}),
                            "original_content_hash": content_hash,
                        }
                        await self._sync_cache.set(content_key, cached)
                        await self._sync_cache.set_meta(content_key, replay_meta)
                    self._record_outcome(meta, LARGE_OFFSET_SERVED)
                    return self._delivery_payload(
                        cached, sub_bytes, evaluation, meta, target_id,
                        (meta_record or {}).get("reference_sha"),
                    )
                if (is_reusable_verified(stored_state, stored_verification)
                        and (meta_record or {}).get("target_binding") in {"exact", "strong"}
                        and (not unified_authority_enabled(meta) or (
                            (meta_record or {}).get("verification_method") == "unified"
                            and (meta_record or {}).get("verification_version") == "timing-v3"
                            and re.fullmatch(r"[a-f0-9]{64}", str((meta_record or {}).get("reference_sha") or ""))
                            and (meta_record or {}).get("artifact_sha256") == hashlib.sha256(cached).hexdigest()))):
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
                    timing.cache_hit = True
                    return self._delivery_payload(cached, sub_bytes,
                        self._evaluation_from_verdict(meta_record or {}), meta, target_id,
                        (meta_record or {}).get("reference_sha"))
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
        pristine_shadow_target = target_text
        query = self._build_query(meta, target_id=target_id, target_text=target_text)

        # Cross-subtitle reuse of a guarded Large-Offset correction. The
        # content-addressed entry excludes the subtitle ID and binds the video
        # fingerprint, byte-identical original payload, language and decision,
        # so a newly selected subtitle cannot inherit a correction measured for
        # different original content. This is a delivered-artifact cache, not a
        # verified verdict: the replay stays UNVERIFIED/UNKNOWN.
        content_hit = await self._load_guarded_content_artifact(meta, content_hash)
        if content_hit is not None:
            # A cache hit must not bypass the Metadata Gate. The artifact was
            # authorized for this video and byte-identical content, but the
            # current request still has to be one the gate would investigate.
            # FAST_ORIGINAL and IDENTITY_CONFLICT retain their normal behavior
            # (serve the original without synchronization).
            pre_gate = decide_metadata_gate(
                meta.get("target_filename"),
                meta.get("release_name"),
                explicit_sync_request=bool(
                    meta.get("force_sync") or meta.get("explicit_sync_request")
                ),
            )
            timing.metadata_gate_decision = pre_gate.action.value
            timing.metadata_gate_reason = pre_gate.reasons[0] if pre_gate.reasons else None
            if pre_gate.action is not MetadataGateAction.INVESTIGATE:
                logger.info(
                    "[sync-cache] layer=content result=suppressed_by_metadata_gate "
                    "sub_id=%s decision=%s reason=%s",
                    target_id,
                    pre_gate.action.value,
                    pre_gate.reasons[0] if pre_gate.reasons else "none",
                )
                content_hit = None
        if content_hit is not None:
            artifact, content_meta, content_key = content_hit
            evaluation = self._evaluation_from_verdict(content_meta)
            self._last_evaluation = evaluation
            self._metrics["verification_cache_hits"] += 1
            self._metrics["large_offset_content_cache_hits"] += 1
            timing.cache_hit = True
            timing.winning_sync_strategy = "large_offset_content_cache"
            logger.info(
                "[sync-cache] layer=content result=hit_guarded_large_offset "
                "sub_id=%s content_hash=%s artifact_sha=%s key_digest=%s "
                "sync_state=%s verification=%s serving_state=%s; reuse across "
                "subtitle identity (no discovery, no alass)",
                target_id,
                content_hash,
                str(content_meta.get("artifact_sha256") or "")[:12],
                hashlib.sha256(content_key.encode()).hexdigest()[:12],
                content_meta.get("sync_state"),
                content_meta.get("verification"),
                content_meta.get("serving_state"),
            )
            if self._sync_cache is not None:
                await self._sync_cache.set(resolution_key, artifact)
                await self._sync_cache.set_meta(
                    resolution_key, self._guarded_replay_meta(content_meta)
                )
            self._audit_serve(
                evaluation, meta, target_id, from_cache=True, resolved=None
            )
            self._record_outcome(meta, LARGE_OFFSET_SERVED)
            return self._delivery_payload(
                artifact, sub_bytes, evaluation, meta, target_id,
                content_meta.get("reference_sha"),
            )

        # Measured-verdict cache: reuse a previous *verified* result for this
        # exact video + subtitle + language + engine. Keyed on the video
        # fingerprint, so a verdict measured for one video is never handed to
        # another. A miss here is normal, not a failure.
        verdict_key = self._verdict_key(meta, content_hash)
        if verdict_key:
            remembered = await self._sync_cache.get_verdict(verdict_key) if self._sync_cache else None
            if remembered is None and self._sync_cache and unified_authority_enabled(meta):
                # Preserve old measured rejection blocking, never old acceptance.
                old_key = verdict_key.replace("verdict:8:timing-v3:", "verdict:7:", 1)
                old = await self._sync_cache.get_verdict(old_key)
                if old and self._reusable_rejection(old, historical=True):
                    remembered = old
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
            if (remembered and unified_authority_enabled(meta)
                    and remembered_state in {"verified_synced", "verified_resynced"}):
                artifact = resync_artifact if resync_artifact is not None else sub_bytes
                if (remembered.get("artifact_sha256") != hashlib.sha256(artifact).hexdigest()
                        or not re.fullmatch(r"[a-f0-9]{64}", str(remembered.get("reference_sha") or ""))):
                    remembered = None
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
                            "target_binding": remembered.get("target_binding"),
                            "target_binding_reasons": remembered.get("target_binding_reasons", []),
                            **self._methodology_metadata(meta),
                            "artifact_sha256": hashlib.sha256(reused_bytes).hexdigest(),
                            "reference_sha": remembered.get("reference_sha"),
                        },
                    )
                self._record_outcome(meta, VERIFIED)
                timing.cache_hit = True
                return self._delivery_payload(reused_bytes, sub_bytes, self._last_evaluation,
                    meta, target_id, remembered.get("reference_sha"))

            # Negative cache for a MEASURED rejection. The verified fast path
            # above only short-circuits successes, so a stored UNVERIFIED
            # verdict used to fall through and re-run alass on every request.
            # Production evidence: two Whiplash requests 33 minutes apart
            # produced byte-identical rejections (p95 2416ms, mad 9356ms,
            # residual unmatched 33 -> 130) and each burned a subprocess.
            #
            # The verdict is content-bound: `_verdict_key` is derived from the
            # video fingerprint + subtitle hash + language, so this can only
            # ever be recalled for the exact same pairing that failed. A
            # different video, a different subtitle, or a different language
            # gets a different key and still runs normally.
            #
            # This serves `sub_bytes`, the ORIGINAL, because an unverified
            # rejection has no trustworthy transformed artifact. It never
            # upgrades a rejection into a verified claim -- `_evaluation_from_verdict`
            # reports it as `cached`, and `sync_state` stays UNVERIFIED.
            if (
                remembered
                and unified_authority_enabled(meta)
                and self._stale_unified_integrity(remembered)
            ):
                # Pre-reconstruction integrity negative: re-evaluate once under
                # the corrected boundary instead of suppressing it. The fresh
                # verdict carries the pipeline marker, so this costs at most
                # one extra evaluation per stale entry.
                logger.info(
                    "[sync] cached Unified integrity rejection predates the "
                    "reconstruction boundary for sub=%s -> re-evaluating",
                    target_id,
                )
                remembered = None
            if (
                remembered
                and unified_authority_enabled(meta)
                and self._stale_unified_generation(remembered)
            ):
                # Pre-leniency Unified negative: measured under strict
                # whole-document parsing, which aborted on harmless formatting
                # artifacts before any evidence existed. Re-evaluate once
                # under the current verification-input boundary; the fresh
                # verdict carries the generation marker, so normal reuse
                # resumes immediately after. Positives are never skipped here.
                logger.info(
                    "[sync] cached Unified negative predates the verification-"
                    "input boundary for sub=%s -> re-evaluating",
                    target_id,
                )
                remembered = None
            if remembered and self._reusable_rejection(remembered):
                self._metrics["verification_cache_hits"] += 1
                self._last_evaluation = self._evaluation_from_verdict(remembered)
                self._audit_serve(
                    self._last_evaluation, meta, target_id, from_cache=True, resolved=None
                )
                logger.info(
                    "[sync] reusing measured rejection for sub=%s: %s (no alass run; "
                    "previously failed verification for this exact video/subtitle pair)",
                    target_id,
                    self._last_evaluation.explain(),
                )
                if self._sync_cache is not None:
                    await self._sync_cache.set(resolution_key, sub_bytes)
                self._record_outcome(meta, REJECTED)
                return sub_bytes

        if self._sync_service is None:  # pragma: no cover - defensive
            logger.warning("[sync] no sync service configured -> serving original subtitle")
            return sub_bytes

        # Stage 2 -- metadata gate. No reusable verdict exists at this point,
        # so decide from already-local release evidence whether the expensive
        # investigation (reference discovery, verification, ALASS) is
        # warranted at all. The check is pure local parsing: no provider
        # download, no subprocess, no timing measurement.
        gate_started = time.monotonic()
        gate = decide_metadata_gate(
            meta.get("target_filename"), meta.get("release_name"),
            explicit_sync_request=bool(
                meta.get("force_sync") or meta.get("explicit_sync_request")),
        )
        timing.metadata_gate_decision = gate.action.value
        timing.metadata_gate_reason = gate.reasons[0] if gate.reasons else None
        timing.metadata_gate_ms = round((time.monotonic() - gate_started) * 1000, 3)
        logger.info(
            "[sync] metadata-gate decision=%s reason=%s video=%r subtitle=%r",
            gate.action.value, ",".join(gate.reasons) or "none",
            meta.get("target_filename"), meta.get("release_name"),
        )
        if gate.action is not MetadataGateAction.INVESTIGATE:
            # Metadata-based original delivery, NOT verified synchronization:
            # no verified cache entry is written, no Sticky Reference is
            # promoted, and no timing-v3 evidence is manufactured. The same
            # subtitle stays eligible for a later explicit sync request,
            # which bypasses this gate via force_sync/explicit_sync_request.
            fast_original = gate.action is MetadataGateAction.FAST_ORIGINAL
            if fast_original:
                # Release compatibility is not timing compatibility. When a
                # reference for this exact video is already on disk, measuring
                # it costs no provider request and turns the fast path into a
                # *measured* claim. A witness that measurably contradicts the
                # target means the metadata cannot be trusted, so the normal
                # pipeline runs; anything less than a measurement keeps the
                # original (missing evidence is not a contradiction).
                witness_started = time.monotonic()
                witness = self._timing_witness_decision(
                    target_text, query, meta, target_id)
                timing.metadata_gate_timing_ms = round(
                    (time.monotonic() - witness_started) * 1000, 3)
                timing.metadata_gate_timing_witness = witness.witness
                timing.metadata_gate_timing_reason = witness.reason
                if witness.investigate:
                    logger.info(
                        "[sync] metadata gate deferred for sub=%s: %s witness "
                        "%s measurably contradicts the target (median offset %+.2fs); "
                        "investigating",
                        target_id,
                        gate.reasons[0] if gate.reasons else "no_mismatch_evidence",
                        witness.witness,
                        (witness.median_offset_ms or 0.0) / 1000.0,
                    )
                    fast_original = False
                else:
                    logger.info(
                        "[sync] metadata gate fast-original confirmed for sub=%s: "
                        "reason=%s witness=%s",
                        target_id, witness.reason, witness.witness or "none",
                    )
            if fast_original:
                self._record_outcome(meta, classify_outcome(False, None, False))
                return sub_bytes
            if gate.action is MetadataGateAction.IDENTITY_CONFLICT:
                self._record_outcome(meta, classify_outcome(False, None, False))
                logger.warning(
                    "[sync] metadata-gate identity conflict for sub=%s: %s -> "
                    "serving original without investigation",
                    target_id, ",".join(gate.reasons) or "none",
                )
                return sub_bytes
            # A timing witness disproved fast delivery: fall through to the
            # unchanged investigation pipeline. Self-references and weak
            # evidence are still rejected there exactly as before.

        # Reference strategies in priority order (external-release-reference reference,
        # then OpenSubtitles MovieHash). The playing stream is never probed.
        attempts: list[tuple[str, Any]] = list(self._strategies())

        def _first_dialogue_ms(text: str) -> int | None:
            """First real dialogue cue, for diagnostic context only."""
            from app.services.subtitle_matcher import (
                first_dialogue_cluster,
                strip_intro_nonspeech,
            )

            try:
                return first_dialogue_cluster(strip_intro_nonspeech(text))
            except Exception:  # pragma: no cover - diagnostics must never break sync
                return None

        def _passes_cue_sanity(reference_text: str) -> bool:
            """Execution-window cue check, used to reject a wrong-cut reference.

            A mislabeled or season-pack reference can be downloaded and only
            then revealed as unusable; validating inside the strategy lets it
            walk to the next-ranked candidate instead of aborting the sync.
            """
            if unified_authority_enabled(meta):
                return bool(reference_text)
            verdict = validate_cue_sanity(
                target_text,
                reference_text,
                threshold_ms=FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
            )
            if verdict["ok"]:
                return True
            from app.services.sync.timeline_affine import plausible_affine_reference

            if plausible_affine_reference(target_text, reference_text):
                # Admission to independent verification, never identity proof.
                return True
            # Beyond the normal window the reference is not rejected
            # outright any more: it becomes a Large Offset Candidate and
            # has to earn an alass attempt from the evidence gate. Reusing
            # the same helper as the main loop keeps the two entry points
            # from disagreeing about what counts as alignable.
            assessment = _large_offset_assessment(reference_text)
            if assessment is not None and assessment.accepted:
                logger.info(
                    "[sync] candidate reference is a large-offset candidate; "
                    "evidence accepted (%s)",
                    assessment.summary(),
                )
                return True
            # Diagnostic detail. The rejection REASON alone cannot distinguish
            # "wrong cut" from "this release simply opens with a credit block",
            # which is why the 5-of-6 candidate skip in the Whiplash pool could
            # not be explained from the logs. These are the exact numbers the
            # gate measured, so the next decision is made on evidence rather
            # than on a hypothesis.
            logger.warning(
                "[sync] candidate reference rejected by cue-sanity: reason=%s "
                "delta_ms=%s candidate_first_ms=%s reference_first_ms=%s "
                "window_ms=%s penalty=%s target_first_ms=%s; trying next candidate",
                verdict["reason"],
                verdict["delta_ms"],
                verdict["candidate_first_ms"],
                verdict["reference_first_ms"],
                verdict["threshold_ms"],
                verdict["penalty"],
                _first_dialogue_ms(target_text),
            )
            return False

        def _large_offset_assessment(
            reference_text: str,
        ) -> LargeOffsetAssessment | None:
            """Run the Large Offset Evidence Gate for one reference.

            ``None`` means "not a large-offset candidate at all" --
            either the offset already fits the normal window, or there was
            no measurable opening dialogue to reason from. Identity counts
            as supported because the caller only ever offers references the
            selector already bound to this exact target; the gate treats
            that as a precondition, never as proof.
            """
            seed = classify_large_offset_candidate(
                target_text,
                reference_text,
                threshold_ms=FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
            )
            if seed is None:
                return None
            logger.info(
                "[sync] large-offset candidate detected (seed=%+.2fs, "
                "normal window %.0fs)",
                seed / 1000.0,
                FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS / 1000.0,
            )
            return assess_large_offset(
                target_text,
                reference_text,
                seed_offset_ms=seed,
                identity_supported=True,
            )

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
        # Quality-aware fallback state. The first weak-binding guarded result
        # is provisional; later independent candidates may still win. The
        # extra attempt/time budget is separate from, and normally smaller
        # than, the existing per-request ALASS candidate limit.
        quality_fallback_seen = False
        quality_fallback_started = 0.0
        quality_fallback_alass_base = 0
        # Shadow-evaluation wall clock consumed so far this request. The shadow
        # path must never starve the response: once the budget is spent, later
        # rejected candidates skip shadow measurement (serving is unaffected;
        # the original is served either way).
        # True once a reference was resolved, aligned, and REJECTED. The tail
        # must not then mark_failed: that flag means "no usable reference",
        # and overloading it with "the reference did not verify" would
        # suppress the very retry this multi-candidate loop exists to perform.
        rejected_candidates_seen = False
        # Likewise, a candidate left untried because the per-request alass
        # budget ran out is not evidence that no usable reference exists; a
        # later request may well verify it, so it must not be suppressed.
        candidates_deferred = False

        # Pristine inputs for the two-point PAL fallback at the tail. The loop
        # below mutates ``target_text`` (intro stripping, anchor pre-shift),
        # but the PAL path was validated against unmodified texts, so it must
        # measure exactly what the request carried. Likewise the first resolved
        # reference is the highest-priority strategy's pick.
        pal_target_snapshot = target_text
        pal_first_reference: str | None = None
        pal_resolved: ResolvedReference | None = None
        # Immutable self-reference identity, precomputed once: the working
        # copy mutates across iterations, so a per-iteration comparison alone
        # misses byte-identical references evaluated after any preparation.
        from app.services.sync.reference_content_identity import (
            PristineTargetIdentity,
        )

        _pristine_identity = PristineTargetIdentity.precompute(pal_target_snapshot)

        _attempt_iterator = self._reference_attempts(
            attempts, query, _passes_cue_sanity, timing, meta
        )
        # The job runner closes this on every exit path, including an early
        # verified return. That is what cancels speculative prefetches instead
        # of leaving them downloading after the response is already served.
        timing.active_attempt_iterator = _attempt_iterator  # type: ignore[attr-defined]
        async for strategy_name, strategy_obj in _attempt_iterator:
            # Wall-clock bound for the quality fallback: stop before paying for
            # another candidate download/evaluation once the window is spent.
            if quality_fallback_seen:
                try:
                    _quality_max_seconds = float(
                        getattr(settings, "QUALITY_FALLBACK_MAX_SECONDS", 8.0) or 0.0
                    )
                except (TypeError, ValueError):
                    _quality_max_seconds = 8.0
                _quality_elapsed = time.monotonic() - quality_fallback_started
                if _quality_elapsed > _quality_max_seconds:
                    timing.quality_fallback_exhausted = True
                    logger.warning(
                        "[sync-quality] fallback time budget exceeded (%.1fs > %.1fs); "
                        "not trying another candidate",
                        _quality_elapsed,
                        _quality_max_seconds,
                    )
                    break
            # Every candidate starts from the subtitle as requested. Intro
            # stripping and the anchor pre-shift below are prepared against
            # *one* reference; carried into the next iteration, a pre-shift
            # rejected for candidate 1 made candidate 2 measure the shifted
            # text as "already aligned" and then serve the unshifted original
            # as verified. The cue-sanity validator reads this same name, so
            # it too now judges each download against the pristine target.
            target_text = pristine_shadow_target
            timing.candidates_considered += 1
            try:
                if getattr(strategy_obj, "validates_target", False):
                    resolved = await strategy_obj.resolve_with_provenance(
                        query, update_validator=_passes_cue_sanity
                    )
                else:
                    resolved = await strategy_obj.resolve_with_provenance(query)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("[sync] %s strategy failed: %s -> next", strategy_name, type(exc).__name__)
                resolved = ResolvedReference(None)

            if not resolved.text:
                if resolved.reference_failure:
                    timing.note_reference_failure(resolved.reference_failure)
                continue
            # Independent evidence only. A reference that is the target subtitle
            # -- byte for byte, or the same dialogue under a different cue grid,
            # e.g. the same member re-extracted from the same season-pack archive
            # -- aligns trivially and would otherwise be reported as a verified
            # synchronization. Skip it and let the loop try the next candidate;
            # the original subtitle is still served if none survives.
            from app.services.sync.reference_content_identity import identical_content

            # Pristine identity first: the working copy may already carry
            # intro-stripping or pre-shift edits from an earlier iteration,
            # which would let a byte-identical download slip through.
            _self_reference_reason = _pristine_identity.matches(resolved.text)
            if not _self_reference_reason:
                _self_reference_reason = identical_content(target_text, resolved.text)
            if _self_reference_reason:
                logger.warning(
                    "[sync] reference rejected: reason=%s reason_code="
                    "reference_content_identical_to_target strategy=%s candidate=%s; "
                    "a reference identical to the target is not independent evidence",
                    _self_reference_reason, strategy_name, resolved.candidate or "?",
                )
                timing.note_reference_failure("reference_content_identical_to_target")
                continue

            # Futile-alignment skip for a bare disc track with no identity
            # evidence. A container filename like ``00001.m2ts`` names no
            # release, so the only reference reachable here is a name-based
            # guess about a *different* cut. Running alass against it cannot
            # succeed: the observed Whiplash run burned a subprocess and
            # ~1.7s only for the verifier to reject it (p95 2416ms > 2000ms,
            # movement mad 9356ms, residual matching degraded 33 -> 130).
            #
            # This is a *skip*, never an acceptance. It runs only when BOTH
            # hold:
            #   1. no video_hash was supplied, so there is no exact identity;
            #   2. the reference is not hash-derived (``decision_kind`` is not
            #      "hash"), so it rests on filename/title similarity alone.
            #
            # With a real MovieHash the hash strategy supplies exact identity
            # evidence and the reference is legitimately trustworthy, so a disc
            # track name must NOT block it. Gating on both conditions is what
            # keeps this from weakening any trust path: a genuine hash match
            # still aligns normally.
            if (
                decision_kind_is_name_based(resolved.kind)
                and is_bare_disc_track_name(meta.get("target_filename"))
                and not str(meta.get("video_hash") or "").strip()
            ):
                logger.info(
                    "[sync] skipping futile alass: target=%r is a bare disc track with no "
                    "video_hash, and the best reference is a name-based %s match; serving "
                    "the original subtitle unsynchronized",
                    meta.get("target_filename"),
                    resolved.kind,
                )
                continue

            timing.reference_kind = resolved.kind
            last_resolved = resolved
            reference = resolved.text
            if resolved.kind == "hash":
                capture_provider(reference, "opensubtitles")
            capture_reference(reference, resolved.candidate, resolved.kind)
            decision_kind = resolved.kind
            binding = self._target_binding(resolved, meta)
            if pal_first_reference is None and reference:
                pal_first_reference = reference
                pal_resolved = resolved
                timing.pal_reference_kind = resolved.kind
                timing.pal_reference_index = timing.candidates_considered
            # Cheap global candidate on pristine input, independently verified
            # per reference. Never promotes edition provenance to MovieHash.
            affine_served = await self._try_timeline_affine(
                pal_target_snapshot, resolved, meta, target_id, content_hash,
                resolution_key, verdict_key,
            )
            affine_reference_corroborated = False
            if isinstance(affine_served, BindingAssessment):
                # This return type means affine independently accepted, but
                # metadata alone could not authorize serving its artifact.
                affine_reference_corroborated = self._affine_corroborates_weak_binding(affine_served)
                if not affine_reference_corroborated:
                    rejected_candidates_seen = True
                    continue
                logger.info("[sync] affine corroborated same-episode source-family reference; "
                            "binding remains weak; continuing to alass verification")
            elif affine_served is not None:
                return affine_served
            source_family_reference_eligible = self._affine_corroborates_weak_binding(binding)
            if binding.state is TargetBinding.WEAK and not source_family_reference_eligible:
                rejected_candidates_seen = True
                continue
            if source_family_reference_eligible and not affine_reference_corroborated:
                logger.info("[sync] same-episode source-family reference admitted to alass; "
                            "affine did not solve timing; final verification required")
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
            # A reference only earns a larger offset *magnitude* ceiling
            # after the gate accepted it. Reset per candidate so one
            # vetted reference cannot widen the window for the next one.
            offset_ceiling_ms = MAX_PLAUSIBLE_OFFSET_MS
            # Set only when this reference arrived through the large-offset
            # path; None on the normal path.
            large_offset: LargeOffsetAssessment | None = None
            # The reference-first investigation that accompanies an accepted
            # large-offset candidate. Built only from the exact target text
            # alass is about to receive, and consulted only after alass has
            # produced something to judge. None on the normal path, which keeps
            # the ordinary serving decision untouched.
            large_offset_inv: LargeOffsetInvestigation | None = None
            if not sanity["ok"] and not unified_authority_enabled(meta):
                large_offset = _large_offset_assessment(reference)
                if large_offset is not None and large_offset.accepted:
                    offset_ceiling_ms = LARGE_OFFSET_MAX_MS
                    logger.info(
                        "[sync] large-offset evidence accepted for %s "
                        "(ceiling %.0fs); %s",
                        strategy_name,
                        LARGE_OFFSET_MAX_MS / 1000.0,
                        large_offset.summary(),
                    )
                else:
                    candidates_rejected += 1
                    # ``None`` means the offset never looked like a
                    # large-offset candidate at all, so the plain
                    # cue-sanity path owns this rejection.
                    if large_offset is None:
                        # Name the candidate AND the measured numbers. Without
                        # them a drop here is indistinguishable from a candidate
                        # that was never considered, which is exactly the
                        # confusion that hid the inert multi-candidate loop.
                        logger.info(
                            "[sync] normal cue-sanity rejection: candidate=%s "
                            "reason=%s delta_ms=%s candidate_first_ms=%s "
                            "reference_first_ms=%s window_ms=%s target_first_ms=%s "
                            "(offset within the %.0fs window, or no measurable "
                            "dialogue to compare)",
                            strategy_name,
                            sanity["reason"],
                            sanity["delta_ms"],
                            sanity["candidate_first_ms"],
                            sanity["reference_first_ms"],
                            sanity["threshold_ms"],
                            _first_dialogue_ms(target_text),
                            FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS / 1000.0,
                        )
                    elif REASON_MAX_OFFSET_EXCEEDED in large_offset.reason_codes:
                        logger.info(
                            "[sync] large-offset rejected because max offset "
                            "exceeded (%.2fs > %.0fs); no alass attempt",
                            (large_offset.offset_seed_ms or 0) / 1000.0,
                            LARGE_OFFSET_MAX_MS / 1000.0,
                        )
                    else:
                        logger.info(
                            "[sync] large-offset evidence rejected: %s",
                            large_offset.summary(),
                        )
                    logger.info(
                        "[sync-budget] limit=%d attempted=%d alass_started=%d "
                        "alass_completed=%d rejected_before_alass=%d deferred=%d "
                        "candidate=%s strategy=%s reason=%s "
                        "counted_toward_limit=false",
                        self._alass_candidate_limit,
                        alass_attempted,
                        alass_started,
                        alass_completed,
                        candidates_rejected,
                        alass_deferred,
                        hashlib.sha256(str(strategy_name).encode()).hexdigest()[:8],
                        strategy_name,
                        (
                            "large_offset_evidence_rejected"
                            if large_offset is not None
                            else "cue_sanity_rejected"
                        ),
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

            # Deterministic serving gate: a subtitle that genuinely needs no
            # re-timing is served as-is and cached; otherwise alass must run. A
            # raw passthrough with an observable offset is never served when a
            # valid reference exists.
            #
            # This must be a *measured* claim. A nearest-neighbour median cannot
            # support it: in a densely cued subtitle every cue has some
            # reference cue within a second or two, so the median collapses
            # towards zero however far apart the timelines are. Production hit
            # exactly that -- three real releases measured +0.06s / +0.14s /
            # +0.06s and were served as "already aligned" while their residual
            # p95 was 4401 / 4588 / 4401ms and no section of the film agreed.
            # So the claim additionally requires real cue correspondence, a
            # residual spread inside the verifier's own tolerance, and agreement
            # across several sections of the timeline. Any failure hands the
            # subtitle to alass instead of serving it as-is.
            consistency = measure_alignment_consistency(
                target_text, reference, is_dialogue=is_dialogue_cue
            )
            # The evidence gate's regional consensus is a better measurement than
            # anything computed here, so when it has established a real
            # displacement the shortcut is simply unavailable.
            if large_offset is not None and large_offset.accepted:
                consistency = None
            if (consistency is not None and consistency.aligned
                    and not affine_reference_corroborated and not source_family_reference_eligible):
                # No re-timing needed. Classify explicitly so "already aligned"
                # is a measured claim, not an assumption.
                # Same handoff as the post-alass path: parse the text alass
                # would have been given, then hand the collection over.
                # Unified authority replaces the legacy diagnostic with a
                # stub, so its document-wide parse is skipped when ON.
                _unified_only = unified_authority_enabled(meta)
                target_cues = [] if _unified_only else parse_srt_cues(target_text)
                if not _unified_only:
                    logger.info(
                        "[sync] verifier input: target_raw_chars=%d "
                        "target_parsed_cue_count=%d verifier_target_cue_count=%d",
                        len(target_text),
                        len(target_cues),
                        len(target_cues),
                    )
                evaluation = self._legacy_diagnostic(meta,
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
                    max_plausible_offset_ms=offset_ceiling_ms,
                )
                evaluation = await self._select_evaluation(
                    evaluation, target_text=pristine_shadow_target, reference_text=reference,
                    candidate_text=pristine_shadow_target, meta=meta, algorithm="already_aligned", resolved=resolved,
                )
                if (unified_authority_enabled(meta) and not may_serve_synchronized(
                        evaluation.sync_state, evaluation.verification)):
                    rejected_candidates_seen = True
                    continue
                if not self._gate_target_binding(evaluation, binding):
                    rejected_candidates_seen = True
                    continue
                self._last_evaluation = evaluation
                self._metrics["candidates_verified"] += 1
                await self._store_verdict(
                    verdict_key, evaluation, content_hash, meta, target_id, artifact_bytes=sub_bytes,
                    reference_sha=hashlib.sha256(reference.encode("utf-8")).hexdigest(),
                )
                self._audit_serve(
                    evaluation, meta, target_id, from_cache=False, resolved=resolved
                )
                logger.info(
                    "[sync] target already aligned (median offset %+.2fs, "
                    "residual p95 %.0fms, %d/%d sections agree, "
                    "coverage %.0f%%) -> serving original [%s]",
                    consistency.median_offset_s or 0.0,
                    consistency.p95_ms or 0.0,
                    consistency.sections_agreeing,
                    consistency.sections_measured,
                    consistency.coverage * 100,
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
                            **self._binding_metadata(binding),
                            "artifact_sha256": hashlib.sha256(sub_bytes).hexdigest(),
                            "reference_sha": hashlib.sha256(reference.encode("utf-8")).hexdigest(),
                            **self._methodology_metadata(meta),
                        },
                    )
                self._record_outcome(
                    meta,
                    VERIFIED
                    if str(evaluation.sync_state.value).startswith("verified")
                    else REJECTED,
                )
                timing.winning_sync_strategy = "already_aligned"
                timing.winning_candidate_index = timing.candidates_considered
                self._promote_sticky_reference(
                    meta=meta, target_id=target_id, pristine_text=pristine_shadow_target,
                    resolved=resolved, evaluation=evaluation,
                )
                return self._delivery_payload(sub_bytes, sub_bytes, evaluation, meta, target_id,
                    hashlib.sha256(reference.encode("utf-8")).hexdigest())

            # "Relaxed" means the target name carries no edition signal. It is
            # passed through to ``sync_async`` for call compatibility only:
            # ``SyncService.sync`` no longer applies a timeline gate (its
            # ``_timeline_rejection`` helper is unused), so acceptance is
            # decided by the independent verifier below. An absent
            # target_filename is never borrowed from the subtitle.
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
                candidates_deferred = True
                continue

            # Two-Point primary: ordinary constant-offset and linear-drift
            # cases are solved here on the pristine original, so ALASS never
            # runs for a candidate Two-Point already verified. Returns None
            # (anchor failure, unified refusal, binding refusal) exactly when
            # the ALASS fallback below must still be tried.
            two_point_served = await self._try_two_point_primary(
                pristine_shadow_target, resolved, meta, target_id, content_hash,
                resolution_key, verdict_key,
                affine_reference_corroborated=affine_reference_corroborated,
                source_family_reference_verified=source_family_reference_eligible,
            )
            if two_point_served is not None:
                return two_point_served

            # Reference-first identity check for the large-offset path. Runs
            # here, on the exact ``target_text`` alass receives (post-stripping),
            # so the evidence the serving decision is built on describes the
            # same two documents the correction is built from.
            #
            # ``identity_supported`` mirrors the gate above (orchestrator passes
            # True at its exec site): the investigation re-evaluates the same
            # gate with the same inputs, so a disagreement here would mean the
            # two disagree about the same evidence rather than about different
            # evidence. Season/episode metadata is not available on
            # ``ResolvedReference``, so no episode claim is invented -- the
            # temporal signals are what establish identity.
            if large_offset is not None and large_offset.accepted:
                large_offset_inv = investigate_large_offset(
                    target_text,
                    reference,
                    identity_supported=True,
                    reference_trust=getattr(resolved, "reference_trust", None),
                    reference_failure=getattr(resolved, "reference_failure", None),
                    reference_reasons=list(
                        getattr(resolved, "reference_reasons", None) or []
                    ),
                    independent_reference_count=max(
                        1,
                        int(getattr(resolved, "reference_independent_sources", 0) or 0),
                    ),
                )
                logger.info(
                    "[sync] large_offset.investigation entered=%s decision=%s "
                    "same_episode=%s eligible=%s seed=%+.2fs reasons=%s",
                    large_offset_inv.entered,
                    large_offset_inv.decision.value,
                    large_offset_inv.same_episode.value,
                    large_offset_inv.eligible_for_alass,
                    (large_offset_inv.seed_offset_ms or 0) / 1000.0,
                    ",".join(large_offset_inv.reason_codes) or "none",
                )

            # Anchor-correspondence pre-shift. Runs on the normal path, where
            # cue-sanity has already accepted the reference, so nothing has
            # proposed an offset and alass is handed an ~11s disagreement
            # between the two opening dialogues with no hint that it exists.
            #
            # Deliberately after the strip above and before alass: the shift is
            # computed from the exact text alass receives, and applied to that
            # same text, so the evidence describes the documents that actually
            # get aligned. When alass cannot bridge a large constant offset it
            # leaves the timeline alone (production: mad=0ms while peak shift
            # was +11.08s) and the verifier then refuses on residual p95.
            #
            # A refusal here is routine, not an error. The proposal has to be
            # corroborated across regions *and* have to measurably improve
            # correspondence; a wrong-but-self-consistent hypothesis is rejected
            # on the second test. Nothing downstream changes: alass still runs,
            # and the verifier still decides whether anything is served.
            preshift_started = time.monotonic()
            preshift_decision = decide_anchor_preshift(target_text, reference)
            timing.anchor_preshift_ms += (time.monotonic() - preshift_started) * 1000
            if preshift_decision.accepted and preshift_decision.preshift is not None:
                preshifted_text = preshift_text(
                    target_text, preshift_decision.preshift.shift_ms
                )
                if preshifted_text is not None:
                    logger.info(
                        "[sync] anchor pre-shift applied before alass: %s",
                        preshift_decision.preshift.summary(),
                    )
                    target_text = preshifted_text
                else:
                    logger.info(
                        "[sync] anchor pre-shift discarded: the shift left no cues"
                    )
            else:
                logger.info(
                    "[sync] anchor pre-shift not applied: %s",
                    preshift_decision.reason,
                )

            if quality_fallback_seen:
                try:
                    max_extra = max(
                        0, int(getattr(settings, "QUALITY_FALLBACK_MAX_ATTEMPTS", 2) or 0)
                    )
                    max_seconds = max(
                        0.0,
                        float(getattr(settings, "QUALITY_FALLBACK_MAX_SECONDS", 8.0) or 0.0),
                    )
                except (TypeError, ValueError):
                    max_extra, max_seconds = 2, 8.0
                elapsed = time.monotonic() - quality_fallback_started
                if (
                    alass_attempted - quality_fallback_alass_base >= max_extra
                    or elapsed > max_seconds
                ):
                    timing.quality_fallback_exhausted = True
                    logger.warning(
                        "[sync-quality] fallback budget exhausted after %d extra attempt(s) "
                        "and %.1fs; not serving the provisional weak-binding correction",
                        max(0, alass_attempted - quality_fallback_alass_base),
                        elapsed,
                    )
                    break
                timing.quality_fallback_attempts += 1

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
            if synced and unified_authority_enabled(meta):
                # Timing-only reconstruction across the sanitization boundary:
                # ALASS aligns a sanitized working copy (ads removed), while
                # Unified judges against the pristine original. Transplant the
                # verified ALASS timings onto the original dialogue when the
                # 1:1 mapping is unambiguous; otherwise keep fail-closed
                # behavior (verify the raw bytes, which then reject as before).
                # Legacy path untouched: no reconstruction when gate is off.
                reconstructed = reconstruct_with_original_dialogue(
                    pristine_shadow_target, synced
                )
                if reconstructed is not None:
                    if reconstructed != synced:
                        logger.info(
                            "[sync] reconstructed candidate with original dialogue "
                            "for sub=%s (alass timings preserved, %d chars)",
                            target_id,
                            len(reconstructed),
                        )
                    synced = reconstructed

            if synced:
                capture_transformed(reference, synced, "alass", alignment_input=target_text)
                # Independent verification: alass exiting 0 is not proof of
                # synchronization. Measure what it actually produced and record
                # an explainable verdict.
                #
                # NOTE: this comment once named ``_validate_synced_output`` and
                # a "timeline band" as the authoritative accept/reject gates.
                # Neither is called on this path any more; cue sanity (before
                # alass) and the verifier evaluation below decide what is
                # served.
                #
                # The target cues are parsed HERE, from the exact text alass was
                # given, and handed over as a collection. The analyzer accepts
                # either raw text or cues; it used to be given the text and
                # reparse it with its own ``parse_srt_cues``. Two parsers with
                # two accepted timestamp grammars, and a real trace where the
                # sync service counted 584 target cues while the verifier was
                # handed 0. Parsing at the boundary makes
                # ``pre-alass count == verifier count`` an invariant.
                # Unified authority replaces the legacy diagnostic with a
                # stub, so its document-wide parse is skipped when ON.
                _unified_only = unified_authority_enabled(meta)
                target_cues = [] if _unified_only else parse_srt_cues(target_text)
                if not _unified_only:
                    logger.info(
                        "[sync] verifier input: target_raw_chars=%d "
                        "target_parsed_cue_count=%d verifier_target_cue_count=%d",
                        len(target_text),
                        len(target_cues),
                        len(target_cues),
                    )
                evaluation = self._legacy_diagnostic(meta,
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
                    max_plausible_offset_ms=offset_ceiling_ms,
                )
                self._last_evaluation = evaluation
                # Large-offset path only: decide whether the corrected bytes may
                # replace the original in this response. The analyzer's own
                # verdict above stays authoritative for ``sync_state`` and for
                # cache reuse -- this decision only gates delivery of the bytes.
                # Under Unified authority the large-offset/constant-shift serving
                # exceptions below are fenced off, so a ``serving_denied`` here
                # is purely diagnostic: it never overrides a Unified VERIFIED
                # verdict, which alone (with binding and may_serve) controls
                # serving. A denial followed later by TRANSFORMED delivery is
                # therefore not a contradiction -- it is a diagnostic veto that
                # did not apply, followed by an independent verified serve.
                serving_state = LargeOffsetServingState.ORIGINAL
                self._boundary_shadow.schedule(pristine_shadow_target, reference, synced, evaluation)
                if large_offset_inv is not None:
                    alass_validation = validate_alass_output(
                        target_text, synced, reference
                    )
                    serving_state = decide_large_offset_serving(
                        large_offset_inv, alass_validation, evaluation
                    )
                    logger.info(
                        "[sync] large_offset.decision serving_state=%s "
                        "reasons=%s eligible=%s alass_valid=%s mad_ms=%s "
                        "structural_similarity=%s sync_state=%s verification=%s",
                        serving_state.value,
                        ",".join(large_offset_inv.serving_reason_codes) or "none",
                        large_offset_inv.eligible_for_alass,
                        alass_validation.ok,
                        evaluation.mad_offset_ms,
                        evaluation.structural_similarity,
                        evaluation.sync_state.value,
                        evaluation.verification.value,
                    )
                # Constant-shift exemption, evaluated BEFORE the verdict is stored
                # so the cached verdict reflects the exemption. Applies to the
                # normal path, where an anchor pre-shift was accepted above.
                #
                # Scope: residual p95 only, and only for a correction evidenced as
                # a single constant shift by corroborated multi-region anchors,
                # a tight median residual, a stable movement MAD and structural
                # agreement. See app.services.sync.constant_shift_exemption for
                # why p95 is the wrong measurement when two releases disagree on
                # cue segmentation -- and why it must not be exempted on its own.
                constant_shift_state = ConstantShiftExemptionState.ORIGINAL
                if preshift_decision is not None and preshift_decision.accepted:
                    exemption = decide_constant_shift_exemption(
                        preshift_decision.preshift,
                        evaluation,
                        validate_alass_output(target_text, synced, reference),
                        measure_median_abs_residual(synced, reference),
                    )
                    constant_shift_state = exemption.state
                    logger.info(
                        "[sync] constant_shift.decision serving_state=%s reasons=%s %s",
                        exemption.state.value,
                        ",".join(exemption.reason_codes) or "none",
                        exemption.summary(),
                    )
                    if exemption.exempted and not unified_authority_enabled(meta):
                        # Legacy-only rewrite: under Unified authority the
                        # stub diagnostic below carries no measurement to
                        # exempt, and Unified supersedes this verdict at
                        # _select_evaluation. Serving exceptions are fenced
                        # off downstream for the same reason.
                        apply_exemption_to_evaluation(evaluation)
                        self._last_evaluation = evaluation

                evaluation = await self._select_evaluation(
                    evaluation, target_text=pristine_shadow_target, reference_text=reference,
                    candidate_text=synced, meta=meta, algorithm="alass", resolved=resolved,
                )
                binding_allowed = self._gate_target_binding(
                    evaluation, binding, affine_reference_corroborated=affine_reference_corroborated,
                    source_family_reference_verified=source_family_reference_eligible,
                )
                if not binding_allowed:
                    constant_shift_state = ConstantShiftExemptionState.ORIGINAL
                capture_verifier(reference, synced, evaluation)
                self._metrics["candidates_verified"] += 1
                # Decide whether this guarded correction may be served
                # immediately or must remain provisional while the loop tries
                # a better independent reference. Strong/exact binding keeps
                # the confirmed Large-Offset behavior; weak/unknown/conflict
                # binding does not get an immediate override.
                serve_without_override = binding_allowed and may_serve_synchronized(
                    evaluation.sync_state.value, evaluation.verification.value
                )
                rejection = getattr(evaluation, "rejection_reason", None)
                quality = assess_quality_fallback(
                    serving_state=serving_state.value,
                    binding_state=binding.state.value,
                    would_serve_without_override=serve_without_override,
                    sync_state=str(evaluation.sync_state.value),
                    verification=str(evaluation.verification.value),
                    rejection_reason=(
                        str(getattr(rejection, "value", rejection))
                        if rejection is not None else None
                    ),
                    p95_ms=getattr(evaluation, "p95_offset_ms", None),
                    mad_ms=getattr(evaluation, "mad_offset_ms", None),
                    structural_similarity=getattr(
                        evaluation, "structural_similarity", None
                    ),
                )
                quality_defer = bool(
                    getattr(settings, "QUALITY_FALLBACK_ENABLED", True)
                    and quality.defer
                )
                if quality_defer:
                    if not quality_fallback_seen:
                        quality_fallback_seen = True
                        quality_fallback_started = time.monotonic()
                        quality_fallback_alass_base = alass_attempted
                        timing.quality_fallback_triggered = True
                        timing.quality_fallback_first_p95_ms = getattr(
                            evaluation, "p95_offset_ms", None
                        )
                        timing.quality_fallback_first_binding = binding.state.value
                    rejected_candidates_seen = True
                    logger.warning(
                        "[sync-quality] provisional weak-binding guarded result "
                        "deferred for sub=%s candidate=%s: %s; trying the next "
                        "independent reference within the fallback budget",
                        target_id,
                        strategy_name,
                        quality.summary,
                    )
                # Computed before the verdict is stored: a resync verdict records
                # where its transformed artifact lives, so a later reuse can
                # serve the synchronized bytes instead of the original.
                key = build_synced_cache_key(
                    meta, target_id, content_hash=content_hash, decision=decision_kind
                )
                # Content-addressed twin for cross-subtitle reuse. It deliberately
                # excludes the subtitle ID and includes the video fingerprint,
                # original content hash and language, so only a byte-identical
                # original for the same video/language can read it.
                content_key = self._content_artifact_key(meta, content_hash)
                corrected_serving = (
                    serving_state
                    is LargeOffsetServingState.ALASS_CORRECTED_LARGE_OFFSET
                )
                await self._store_verdict(
                    verdict_key, evaluation, content_hash, meta, target_id,
                    artifact_key=content_key if corrected_serving else key,
                    artifact_bytes=synced.encode("utf-8"),
                    reference_sha=hashlib.sha256(reference.encode("utf-8")).hexdigest(),
                    serving_state=serving_state.value,
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
                        "reference_sha": hashlib.sha256(reference.encode("utf-8")).hexdigest(),
                        "artifact_sha256": hashlib.sha256(synced.encode("utf-8")).hexdigest(),
                        "sync_state": evaluation.sync_state.value,
                        "verification": evaluation.verification.value,
                        "sync_confidence": evaluation.sync_confidence,
                        "original_content_hash": content_hash,
                        # Distinct from ``sync_state`` on purpose: this records
                        # which Large Offset decision delivered the bytes and
                        # never asserts a verification the analyzer refused.
                        "serving_state": serving_state.value,
                        # Generation marker for the guarded replay path. Only a
                        # corrected artifact written under the current guard may
                        # be replayed without re-running the pipeline.
                        "large_offset_authorization": (
                            LARGE_OFFSET_CACHE_AUTHORIZATION_VERSION
                            if serving_state
                            is LargeOffsetServingState.ALASS_CORRECTED_LARGE_OFFSET
                            else None
                        ),
                        **self._binding_metadata(binding),
                        **self._methodology_metadata(meta),
                    }
                    await self._sync_cache.set_meta(key, {**outcome_meta, **{
                        "status": "synced",
                        "applied_shift": self._applied_shifts(target_text, synced),
                        "reference_sha": (
                            hashlib.sha256(reference.encode("utf-8")).hexdigest()
                            if unified_authority_enabled(meta)
                            else hashlib.sha256(reference.encode("utf-8")).hexdigest()[:16]
                        ),
                        "decision": decision_kind,
                        "timestamp": datetime.now(UTC).isoformat(),
                    }})
                    await self._sync_cache.set_meta(resolution_key, outcome_meta)
                    if corrected_serving and not quality_defer:
                        # Cross-subtitle twin for the guarded Large-Offset
                        # artifact. The verdict record points at this key; a
                        # different subtitle ID with byte-identical original
                        # content can validate and serve it without discovery or
                        # another ALASS run. The verdict itself remains
                        # UNVERIFIED/UNKNOWN.
                        await self._sync_cache.set(content_key, synced.encode("utf-8"))
                        await self._sync_cache.set_meta(content_key, {
                            **outcome_meta,
                            "status": "synced",
                            "decision": decision_kind,
                            "timestamp": datetime.now(UTC).isoformat(),
                        })
                # Serving contract: the analyzer is the authority on whether
                # these bytes may replace the original. An attempt it does not
                # trust is kept for diagnostics/cache but never returned to the
                # user, who gets the original provider subtitle instead -- the
                # unsynchronized original is far more usable than a mangled one.
                serve_synchronized = serve_without_override
                # Scoped exception to "an attempt the verifier does not trust
                # must never replace the original".
                #
                # The general verifier refuses these cases on residual p95 alone
                # (segmentation differs between releases); that is the one
                # measurement the Large Offset investigation is allowed to
                # exempt, and it is exempt ONLY after same-episode identity was
                # established reference-first, alass output was validated, the
                # measured movement was one constant shift within
                # MAX_MAD_MS_FOR_STABLE, and structural agreement cleared its
                # floor. Every other refusal is untouched: a content or
                # structural rejection, an unmeasured movement, a failed
                # investigation or an invalid output all leave ``serving_state``
                # at ORIGINAL and this branch behaves exactly as before.
                #
                # The analyzer's verdict is never rewritten to make this pass:
                # ``sync_state`` and ``verification`` below are the values the
                # analyzer recorded, so the artifact stays non-reusable
                # (is_reusable_verified) and this state never becomes a
                # VERIFIED_* claim.
                if (
                    serving_state
                    is LargeOffsetServingState.ALASS_CORRECTED_LARGE_OFFSET
                    and not unified_authority_enabled(meta)
                    and not quality_defer
                ):
                    if not serve_synchronized:
                        logger.info(
                            "[sync] large_offset corrected serving allowed for %s "
                            "(state=%s, verification=%s) -> serving the corrected "
                            "subtitle",
                            target_id,
                            evaluation.sync_state.value,
                            evaluation.verification.value,
                        )
                    serve_synchronized = True
                # Constant-shift exemption. Same scoped exception as the Large
                # Offset branch above -- residual p95 set aside for a correction
                # that four independent measurements agree was one constant
                # shift -- but unlike that path this one rewrites the verdict (see
                # ``apply_exemption_to_evaluation``), so it is only reached when
                # the exemption already granted it.
                if (
                    constant_shift_state
                    is ConstantShiftExemptionState.EXEMPT_CONSTANT_SHIFT
                    and not unified_authority_enabled(meta)
                ):
                    if not serve_synchronized:
                        logger.info(
                            "[sync] constant-shift serving allowed for %s "
                            "(state=%s, verification=%s) -> serving the corrected "
                            "subtitle",
                            target_id,
                            evaluation.sync_state.value,
                            evaluation.verification.value,
                        )
                    serve_synchronized = True
                # No pre-delivery V2 review here. A hybrid gate used to be
                # called at this point for exemption-driven serves, but only
                # when Unified authority was OFF -- and the gate itself returns
                # "proceed" whenever Unified is off -- so it could never block
                # and gave legacy mode a safeguard it did not have. Under
                # Unified authority the exemptions are judged by the Unified
                # verifier itself.
                if serve_synchronized:
                    served = synced.encode("utf-8")
                else:
                    logger.warning(
                        "[sync] not serving the synchronized output for %s: verifier did not "
                        "trust it (state=%s, verification=%s, serving_state=%s) -> serving "
                        "the original subtitle",
                        target_id,
                        evaluation.sync_state.value,
                        evaluation.verification.value,
                        serving_state.value,
                    )
                    served = sub_bytes
                # One line that answers "which bytes went out?" without ever
                # logging subtitle text. Digests only, so this is safe to keep in
                # logs at INFO while remaining specific enough to prove which
                # artifact a request received.
                logger.info(
                    "[sync] delivery sub=%s decision=%s state=%s verification=%s "
                    "serving_state=%s large_offset_decision=%s "
                    "original_sha256=%s transformed_sha256=%s served_sha256=%s "
                    "served_is_transformed=%s bytes=%d",
                    target_id,
                    "TRANSFORMED" if serve_synchronized else "ORIGINAL",
                    evaluation.sync_state.value,
                    evaluation.verification.value,
                    serving_state.value,
                    large_offset_inv.decision.value if large_offset_inv else "n/a",
                    _content_digest(sub_bytes),
                    _content_digest(synced.encode("utf-8")),
                    _content_digest(served),
                    serve_synchronized,
                    len(served),
                )
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
                if serve_synchronized:
                    if quality_fallback_seen:
                        timing.quality_fallback_served = True
                        logger.info(
                            "[sync-quality] selected a stronger independent candidate "
                            "after deferring a weak provisional: sub=%s candidate=%s "
                            "binding=%s p95=%s mad=%s struct=%s verification=%s",
                            target_id,
                            strategy_name,
                            binding.state.value,
                            evaluation.p95_offset_ms,
                            evaluation.mad_offset_ms,
                            evaluation.structural_similarity,
                            evaluation.verification.value,
                        )
                    capture_selected(reference, synced, "alass")
                    self._record_outcome(
                        meta,
                        classify_outcome(True, serving_state.value, False),
                    )
                    self._promote_sticky_reference(
                        meta=meta, target_id=target_id,
                        pristine_text=pristine_shadow_target,
                        resolved=resolved, evaluation=evaluation,
                        serving_state=serving_state.value)
                    return self._delivery_payload(served, sub_bytes, evaluation, meta, target_id,
                        hashlib.sha256(reference.encode("utf-8")).hexdigest())
                # A REJECTED candidate must NOT end the request.
                #
                # This used to return unconditionally, which made the
                # multi-candidate expansion structurally inert: candidate1 always
                # terminated the loop, so 6 discovered candidates produced exactly
                # 1 alass run. Observed live -- only
                # external-release-reference#candidate1 ever appeared, with no
                # SKIPPED/cue-sanity lines, proving the other five were never
                # rejected on their merits, just never reached.
                #
                # Continuing is safe and does not weaken verification: the
                # rejection has already been measured, audited and stored as an
                # UNVERIFIED verdict, and NOTHING is served here. If no later
                # candidate verifies either, the loop falls through to the
                # existing tail, which logs "all sync strategies failed" and
                # returns the original subtitle unchanged.
                logger.info(
                    "[sync] candidate %s was rejected by verification; "
                    "trying the next reference candidate (original will be served "
                    "if none verifies)",
                    strategy_name,
                )
                rejected_candidates_seen = True
                continue

            logger.warning(
                "[sync] %s strategy reference failed sync/validation -> falling back to next strategy",
                strategy_name,
            )

        if not last_resolved.text:
            timing.note_reference_failure("no_independent_reference_available")
            logger.info("[sync] no deterministic reference available -> aborting sync")
        else:
            logger.warning("[sync] all sync strategies failed -> serving original subtitle")
            # Hybrid Two-Point + ALASS routing. Default-off test gate: when
            # disabled this is a no-op and the PAL fallback below runs exactly
            # as before. Uses pristine snapshots and the highest-priority
            # resolved reference; never re-runs provider discovery.
            hybrid_served = await self._try_hybrid_sync(
                pal_target_snapshot,
                pal_first_reference,
                meta,
                target_id,
                sub_bytes,
                content_hash,
                resolution_key,
                verdict_key,
                resolved=pal_resolved,
            )
            if hybrid_served is not None:
                if quality_fallback_seen:
                    timing.quality_fallback_served = True
                return hybrid_served
            # Two-point PAL-speedup fallback. Runs only here -- after Alass
            # and every verified path failed -- using the pristine request
            # texts and the highest-priority resolved reference. A single
            # transform, a single verifier evaluation, then either a narrowly
            # accepted PAL result or the original subtitle, unchanged.
            pal_served = await self._try_pal_two_point_fallback(
                pal_target_snapshot,
                pal_first_reference,
                meta,
                target_id,
                sub_bytes,
                content_hash,
                resolution_key,
                verdict_key,
                resolved=pal_resolved,
            )
            if pal_served is not None:
                if quality_fallback_seen:
                    timing.quality_fallback_served = True
                return pal_served
        if quality_fallback_seen:
            timing.quality_fallback_exhausted = True
            logger.warning(
                "[sync-quality] no candidate met the serving policy after deferring "
                "a weak-binding guarded correction for sub=%s -> serving the original",
                target_id,
            )
        if (
            self._sync_cache is not None
            and not rejected_candidates_seen
            and not candidates_deferred
        ):
            await self._sync_cache.mark_failed(resolution_key)
        self._record_outcome(
            meta, classify_outcome(False, None, rejected_candidates_seen)
        )
        return sub_bytes

    async def _try_timeline_affine(
        self,
        target_text: str | None,
        resolved: ResolvedReference,
        meta: dict,
        target_id: str,
        content_hash: str,
        resolution_key: str,
        verdict_key: str | None,
    ) -> bytes | BindingAssessment | None:
        """Serve/cache only an independently verified affine artifact."""
        from app.services.sync.timeline_affine import (
            MODEL_VERSION,
            STRATEGY_ID,
            attempt_timeline_affine,
        )

        if not target_text or not resolved.text or resolved.partial:
            return None
        try:
            affine_started = time.monotonic()
            result = await asyncio.to_thread(attempt_timeline_affine, target_text, resolved.text)
            _timing = current_sync_timing.get()
            if _timing is not None:
                _timing.timeline_affine_ms += (time.monotonic() - affine_started) * 1000
        except Exception as exc:
            logger.info("[timeline-affine] accepted=false reason=exception:%s", type(exc).__name__)
            return None
        capture_affine(resolved.text, result)
        evaluation = result.evaluation
        if not result.transformed_text or evaluation is None:
            return None
        evaluation = await self._select_evaluation(
            evaluation, target_text=target_text, reference_text=resolved.text,
            candidate_text=result.transformed_text, meta=meta, algorithm=STRATEGY_ID, resolved=resolved,
        )
        if ((not result.accepted and not unified_authority_enabled(meta))
                or not may_serve_synchronized(evaluation.sync_state, evaluation.verification)):
            return None
        binding = self._target_binding(resolved, meta)
        if self._affine_corroborates_weak_binding(binding):
            # A handoff, not authority to serve the affine artifact. Preserve
            # its measured verdict and require fresh evidence for this attempt.
            return binding if evaluation.verification is VerificationAvailability.VERIFIED else None
        if not self._gate_target_binding(evaluation, binding):
            return binding
        evaluation.reference_trust = resolved.reference_trust
        transformed = result.transformed_text.encode("utf-8")
        capture_transformed(resolved.text, result.transformed_text, STRATEGY_ID, evaluation)
        key = build_synced_cache_key(
            meta, target_id, content_hash=content_hash, decision=f"{STRATEGY_ID}-{MODEL_VERSION}"
        )
        self._last_evaluation = evaluation
        await self._store_verdict(
            verdict_key, evaluation, content_hash, meta, target_id, artifact_key=key,
            artifact_bytes=transformed,
            reference_sha=hashlib.sha256(resolved.text.encode("utf-8")).hexdigest(),
        )
        self._audit_serve(evaluation, meta, target_id, from_cache=False, resolved=resolved)
        self._metrics["candidates_verified"] += 1
        if self._sync_cache is not None:
            await self._sync_cache.clear_failed(resolution_key)
            outcome_meta = {
                "artifact_sha256": hashlib.sha256(transformed).hexdigest(),
                "sync_state": evaluation.sync_state.value,
                "verification": evaluation.verification.value,
                "sync_confidence": evaluation.sync_confidence,
                "decision": STRATEGY_ID,
                "model_version": MODEL_VERSION,
                "reference_kind": resolved.kind,
                "reference_sha": hashlib.sha256(resolved.text.encode()).hexdigest(),
                "slope": result.slope,
                "intercept_ms": result.intercept_ms,
                **self._binding_metadata(binding),
                **self._methodology_metadata(meta),
            }
            await self._sync_cache.set(key, transformed)
            await self._sync_cache.set_meta(key, outcome_meta)
            await self._sync_cache.set(resolution_key, transformed)
            await self._sync_cache.set_meta(resolution_key, outcome_meta)
        self._record_outcome(meta, classify_outcome(True, STRATEGY_ID, False))
        logger.info(
            "[sync] delivery sub=%s strategy=%s reference_sha=%s reference_provenance=%s "
            "verification=%s serving_state=synchronized served_is_transformed=True "
            "slope=%s intercept_ms=%s",
            target_id, STRATEGY_ID, hashlib.sha256(resolved.text.encode()).hexdigest()[:16],
            resolved.kind, evaluation.verification.value, result.slope, result.intercept_ms,
        )
        timing = current_sync_timing.get()
        if timing is not None:
            timing.winning_sync_strategy = STRATEGY_ID
        capture_selected(resolved.text, result.transformed_text, STRATEGY_ID)
        self._promote_sticky_reference(
            meta=meta, target_id=target_id, pristine_text=target_text,
            resolved=resolved, evaluation=evaluation)
        return self._delivery_payload(transformed, target_text.encode("utf-8"), evaluation,
            meta, target_id, hashlib.sha256(resolved.text.encode("utf-8")).hexdigest())

    async def _try_two_point_primary(
        self,
        pristine_text: str | None,
        resolved: ResolvedReference,
        meta: dict,
        target_id: str,
        content_hash: str,
        resolution_key: str,
        verdict_key: str | None,
        *,
        affine_reference_corroborated: bool = False,
        source_family_reference_verified: bool = False,
    ) -> bytes | None:
        """Two-Point primary: fit two dialogue anchors, transform the pristine
        original, serve only a Unified-verified artifact.

        Runs per reference after the already-aligned shortcut and before any
        ALASS subprocess, so ordinary constant-offset and linear-drift cases
        never pay for alignment they do not need. The anchors calculate the
        correction; only the independent Timing-First measurement across all
        thirds authorizes serving. Timestamps alone change: dialogue,
        credits, ordering, formatting, and unplaceable lines survive through
        the block-level transfer. Returns served bytes or None (the ALASS
        fallback proceeds unchanged).
        """
        from app.services.sync.candidate_reconstruction import apply_linear_timings
        from two_point_sync.anchors import detect_anchors
        from two_point_sync.parser import parse_srt_for_verification

        if not unified_authority_enabled(meta):
            return None
        if not pristine_text or not resolved.text or resolved.partial:
            logger.debug("[two-point] sub=%s evaluated=false reason=missing_input", target_id)
            return None
        try:
            target_cues, _ = parse_srt_for_verification(
                pristine_text, exclude_ambiguous=True)
            reference_cues, _ = parse_srt_for_verification(
                resolved.text, exclude_ambiguous=True)
            if target_cues is None or reference_cues is None:
                logger.info("[two-point] sub=%s evaluated=false reason=unparseable", target_id)
                return None
            # Anchor detection is bounded: repetitive texts can explode the
            # RANSAC pair space instead of converging (real releases measure
            # well under a second). Exceeding the bound abstains to the ALASS
            # fallback; a result is either complete-identical or discarded.
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    detect_anchors, list(target_cues), list(reference_cues)),
                TWO_POINT_DETECTION_TIMEOUT_SECONDS)
        except Exception as exc:  # fail closed, always (includes TimeoutError)
            logger.info("[two-point] sub=%s evaluated=false error=%s",
                        target_id, type(exc).__name__)
            return None
        scale: float | None = None
        offset_ms: float | None = None
        regime_note: str | None = None
        if (result.accepted and result.first_anchor is not None
                and result.last_anchor is not None and result.scale is not None
                and result.offset_ms is not None):
            # The reported scale/offset derive strictly from the two accepted
            # anchors (never a RANSAC-pair artefact).
            scale, offset_ms = result.scale, result.offset_ms
        elif "NONLINEAR_DISCONTINUITY" in result.reasons and result.regimes:
            # A contradictory timing region (e.g. a differently-cut recap)
            # vetoes the GLOBAL two-anchor claim -- but the majority primary
            # regime may still be proposed for verification. The Unified
            # verifier measures the bulk evidence independently (order-filtered
            # correspondences across all thirds) and refuses unless it genuinely
            # agrees, so authority moves from proposer-veto to evidence.
            # Anything weaker (insufficient anchors, no model, thin or narrow
            # primary support) still refuses, and ALASS owns those cases.
            primary = result.regimes[0]
            contra_support = sum(regime.support for regime in result.regimes[1:])
            if primary.scale is not None and primary.scale > 0 and primary.offset_ms is not None:
                scale, offset_ms = primary.scale, primary.offset_ms
                regime_note = (
                    "primary_regime_only:"
                    f"{len(result.regimes) - 1}_contradictory_region_unmeasured")
                logger.info(
                    "[two-point] sub=%s discontinuity: proposing primary regime "
                    "scale=%.6f offset_ms=%.0f support=%d span=%d-%dms "
                    "contradictory_support=%d (verifier authorizes)",
                    target_id, primary.scale, primary.offset_ms, primary.support,
                    primary.span_start_ms, primary.span_end_ms, contra_support)
        if scale is None or offset_ms is None:
            logger.info("[two-point] sub=%s evaluated=true accepted=false reasons=%s",
                        target_id, ",".join(result.reasons) or "none")
            return None
        candidate = apply_linear_timings(pristine_text, scale, offset_ms)
        if candidate is None:
            logger.info("[two-point] sub=%s evaluated=true accepted=false "
                        "reason=unrepresentable_transform", target_id)
            return None
        evaluation = await self._select_evaluation(
            SubtitleEvaluation(), target_text=pristine_text, reference_text=resolved.text,
            candidate_text=candidate, meta=meta, algorithm=TWO_POINT_STRATEGY_ID,
            resolved=resolved,
        )
        if not may_serve_synchronized(evaluation.sync_state, evaluation.verification):
            logger.info("[two-point] sub=%s evaluated=true accepted=false "
                        "reason=unified_refused state=%s",
                        target_id, evaluation.sync_state.value)
            return None
        binding = self._target_binding(resolved, meta)
        if not self._gate_target_binding(
                evaluation, binding,
                affine_reference_corroborated=affine_reference_corroborated,
                source_family_reference_verified=source_family_reference_verified):
            logger.info("[two-point] sub=%s evaluated=true accepted=false reason=binding refused "
                        "binding=%s", target_id, binding)
            return None
        if regime_note is not None:
            # Honest labelling: the served claim rests on bulk evidence; the
            # contradictory region was preserved but never measured.
            evaluation.reasons.append(regime_note)
        transformed = candidate.encode("utf-8")
        capture_transformed(resolved.text, candidate, TWO_POINT_STRATEGY_ID, evaluation)
        key = build_synced_cache_key(
            meta, target_id, content_hash=content_hash,
            decision=f"{TWO_POINT_STRATEGY_ID}-{TWO_POINT_MODEL_VERSION}"
        )
        self._last_evaluation = evaluation
        await self._store_verdict(
            verdict_key, evaluation, content_hash, meta, target_id, artifact_key=key,
            artifact_bytes=transformed,
            reference_sha=hashlib.sha256(resolved.text.encode("utf-8")).hexdigest(),
        )
        self._audit_serve(evaluation, meta, target_id, from_cache=False, resolved=resolved)
        self._metrics["candidates_verified"] += 1
        if self._sync_cache is not None:
            await self._sync_cache.clear_failed(resolution_key)
            outcome_meta = {
                "artifact_sha256": hashlib.sha256(transformed).hexdigest(),
                "sync_state": evaluation.sync_state.value,
                "verification": evaluation.verification.value,
                "sync_confidence": evaluation.sync_confidence,
                "decision": TWO_POINT_STRATEGY_ID,
                "model_version": TWO_POINT_MODEL_VERSION,
                "reference_kind": resolved.kind,
                "reference_sha": hashlib.sha256(resolved.text.encode()).hexdigest(),
                "slope": result.scale,
                "intercept_ms": result.offset_ms,
                **self._binding_metadata(binding),
                **self._methodology_metadata(meta),
            }
            await self._sync_cache.set(key, transformed)
            await self._sync_cache.set_meta(key, outcome_meta)
            await self._sync_cache.set(resolution_key, transformed)
            await self._sync_cache.set_meta(resolution_key, outcome_meta)
        self._record_outcome(meta, classify_outcome(True, TWO_POINT_STRATEGY_ID, False))
        logger.info(
            "[sync] delivery sub=%s strategy=%s reference_sha=%s reference_provenance=%s "
            "verification=%s serving_state=synchronized served_is_transformed=True "
            "slope=%s intercept_ms=%s groups=%s",
            target_id, TWO_POINT_STRATEGY_ID,
            hashlib.sha256(resolved.text.encode()).hexdigest()[:16],
            resolved.kind, evaluation.verification.value, result.scale, result.offset_ms,
            evaluation.coverage_score,
        )
        timing = current_sync_timing.get()
        if timing is not None:
            timing.winning_sync_strategy = TWO_POINT_STRATEGY_ID
        capture_selected(resolved.text, candidate, TWO_POINT_STRATEGY_ID)
        self._promote_sticky_reference(
            meta=meta, target_id=target_id, pristine_text=pristine_text,
            resolved=resolved, evaluation=evaluation)
        return self._delivery_payload(transformed, pristine_text.encode("utf-8"), evaluation,
            meta, target_id, hashlib.sha256(resolved.text.encode("utf-8")).hexdigest())

    async def _try_pal_two_point_fallback(
        self,
        target_text: str | None,
        reference_text: str | None,
        meta: dict,
        target_id: str,
        sub_bytes: bytes,
        content_hash: str,
        resolution_key: str,
        verdict_key: str | None,
        resolved: ResolvedReference | None = None,
    ) -> bytes | None:
        """Two-point PAL-speedup fallback. Returns served bytes or None.

        Runs only after Alass and every verified path failed. A single linear
        transform from first/last dialogue anchors, one verifier evaluation,
        then a narrow PAL-class policy acceptance. Returns None (serve the
        original) on any failure at any stage: missing anchors, non-PAL slope,
        weak residuals/structure/coverage, verification refusal, or any
        exception. Never overrides an already-verified result -- this method is
        only reachable when none exists.
        """
        from app.services.sync.pal_two_point import (
            MODEL_VERSION as PAL_MODEL_VERSION,
        )
        from app.services.sync.pal_two_point import (
            STRATEGY_ID as PAL_STRATEGY_ID,
        )
        from app.services.sync.pal_two_point import (
            attempt_pal_two_point,
        )

        if not bool(getattr(settings, "PAL_TWO_POINT_ENABLED", True)):
            logger.debug("[pal-two-point] sub=%s evaluated=false reason=disabled", target_id)
            return None
        if not target_text or not reference_text:
            logger.debug("[pal-two-point] sub=%s evaluated=false reason=missing_input", target_id)
            return None
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    attempt_pal_two_point,
                    target_text,
                    reference_text,
                    self._analyzer,
                ),
                TWO_POINT_DETECTION_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # fail closed, always (includes TimeoutError)
            logger.info("[pal-two-point] sub=%s evaluated=false error=%s", target_id, type(exc).__name__)
            return None
        # One structured outcome line per tail-reached request. Without this a
        # rejection is indistinguishable from the path never running, which is
        # exactly what made a live Whiplash fallback undiagnosable: the logs
        # showed ALASS rejections and then silence.
        first_ms = result.first.target_ms if result.first else None
        last_ms = result.last.target_ms if result.last else None
        # first_ms/last_ms above are the SELECTED anchors' target-side
        # timestamps (not subtitle bounds); r1_ms/r2_ms are their reference
        # counterparts, and pairs_evaluated counts pairs that reached the
        # verifier, so a future trace shows exactly what was tried and won.
        r1_ms = result.first.reference_ms if result.first else None
        r2_ms = result.last.reference_ms if result.last else None
        evaluation = result.evaluation
        logger.info(
            "[pal-two-point] sub=%s evaluated=true accepted=%s slope=%s "
            "intercept_ms=%s first_ms=%s last_ms=%s r1_ms=%s r2_ms=%s "
            "pairs_evaluated=%d p95=%s struct=%s "
            "coverage=%s reasons=%s",
            target_id,
            result.accepted,
            f"{result.slope:.6f}" if result.slope is not None else "n/a",
            f"{result.intercept_ms:.0f}" if result.intercept_ms is not None else "n/a",
            first_ms if first_ms is not None else "n/a",
            last_ms if last_ms is not None else "n/a",
            r1_ms if r1_ms is not None else "n/a",
            r2_ms if r2_ms is not None else "n/a",
            result.pairs_evaluated,
            evaluation.p95_offset_ms if evaluation is not None else "n/a",
            evaluation.structural_similarity if evaluation is not None else "n/a",
            evaluation.coverage_score if evaluation is not None else "n/a",
            ",".join(result.reason_codes) or "none",
        )
        if (not result.accepted and not unified_authority_enabled(meta)) or not result.transformed_text:
            return None
        evaluation = result.evaluation
        if evaluation is None:
            return None
        # Narrow policy acceptance, mirroring the constant-shift exemption
        # precedent: rewrite the verdict through the only supported setter,
        # with explicit provenance recording the true producer. The analyzer's
        # measured metrics (p95, structure, coverage) stay exactly as measured;
        # only the serving claim changes, and only because the PAL slope match
        # independently corroborates the correction.
        if not unified_authority_enabled(meta):
            evaluation.set_verdict(
                SyncState.VERIFIED_RESYNCED, VerificationAvailability.VERIFIED
            )
        evaluation = await self._select_evaluation(
            evaluation, target_text=target_text, reference_text=reference_text,
            candidate_text=result.transformed_text, meta=meta, algorithm=PAL_STRATEGY_ID, resolved=resolved,
        )
        if not may_serve_synchronized(evaluation.sync_state, evaluation.verification):
            return None
        binding = self._target_binding(resolved or ResolvedReference(reference_text), meta)
        if not self._gate_target_binding(evaluation, binding):
            return None
        evaluation.reasons.append(
            f"{PAL_STRATEGY_ID}: narrow PAL-speedup policy acceptance "
            f"(slope={result.slope:.6f}, model={PAL_MODEL_VERSION}); "
            "movement MAD set aside for proven frame-rate drift only"
        )
        if evaluation.sync_state is not SyncState.VERIFIED_RESYNCED:
            return None
        transformed = result.transformed_text.encode("utf-8")
        capture_transformed(reference_text, result.transformed_text, PAL_STRATEGY_ID, evaluation)
        key = build_synced_cache_key(
            meta, target_id, content_hash=content_hash, decision=PAL_STRATEGY_ID
        )
        await self._store_verdict(
            verdict_key, evaluation, content_hash, meta, target_id, artifact_key=key,
            artifact_bytes=transformed,
            reference_sha=hashlib.sha256(reference_text.encode("utf-8")).hexdigest(),
        )
        self._audit_serve(
            evaluation, meta, target_id, from_cache=False, resolved=None
        )
        self._metrics["candidates_verified"] += 1
        if self._sync_cache is not None:
            await self._sync_cache.clear_failed(resolution_key)
            await self._sync_cache.set(key, transformed)
            await self._sync_cache.set(resolution_key, transformed)
            await self._sync_cache.set_meta(key, {
                "status": "synced",
                "applied_shift": self._applied_shifts(target_text, result.transformed_text),
                "reference_sha": (
                    hashlib.sha256(reference_text.encode("utf-8")).hexdigest()
                    if unified_authority_enabled(meta)
                    else hashlib.sha256(reference_text.encode("utf-8")).hexdigest()[:16]
                ),
                "decision": PAL_STRATEGY_ID,
                "timestamp": datetime.now(UTC).isoformat(),
                "sync_state": evaluation.sync_state.value,
                "verification": evaluation.verification.value,
                "sync_confidence": evaluation.sync_confidence,
                **self._binding_metadata(binding),
                **self._methodology_metadata(meta),
                "artifact_sha256": hashlib.sha256(transformed).hexdigest(),
            })
            await self._sync_cache.set_meta(resolution_key, {
                "reference_sha": hashlib.sha256(reference_text.encode("utf-8")).hexdigest(),
                "sync_state": evaluation.sync_state.value,
                "verification": evaluation.verification.value,
                "sync_confidence": evaluation.sync_confidence,
                **self._binding_metadata(binding),
                **self._methodology_metadata(meta),
                "artifact_sha256": hashlib.sha256(transformed).hexdigest(),
            })
        self._record_outcome(
            meta, classify_outcome(True, PAL_STRATEGY_ID, False)
        )
        logger.info(
            "[sync] delivery sub=%s decision=TRANSFORMED state=%s verification=%s "
            "strategy=%s slope=%.6f p95=%s struct=%s original_sha256=%s "
            "served_sha256=%s served_is_transformed=True bytes=%d",
            target_id,
            evaluation.sync_state.value,
            evaluation.verification.value,
            PAL_STRATEGY_ID,
            result.slope,
            evaluation.p95_offset_ms,
            evaluation.structural_similarity,
            _content_digest(sub_bytes),
            _content_digest(transformed),
            len(transformed),
        )
        timing = current_sync_timing.get()
        if timing is not None:
            timing.winning_sync_strategy = PAL_STRATEGY_ID
            timing.reference_kind = timing.pal_reference_kind
            timing.winning_candidate_index = timing.pal_reference_index
        capture_selected(reference_text, result.transformed_text, PAL_STRATEGY_ID)
        self._promote_sticky_reference(
            meta=meta, target_id=target_id, pristine_text=target_text,
            resolved=resolved, evaluation=evaluation)
        return self._delivery_payload(transformed, sub_bytes, evaluation, meta, target_id,
            hashlib.sha256(reference_text.encode("utf-8")).hexdigest())

    async def _try_hybrid_sync(
        self,
        target_text: str | None,
        reference_text: str | None,
        meta: dict,
        target_id: str,
        sub_bytes: bytes,
        content_hash: str,
        resolution_key: str,
        verdict_key: str | None,
        resolved: ResolvedReference | None = None,
    ) -> bytes | None:
        """Hybrid Two-Point + ALASS routing. Returns served bytes or None.

        Default-off test gate (``HYBRID_TWO_POINT_ALASS_ENABLED``): when
        disabled this returns None immediately and the caller falls through
        to the existing PAL fallback unchanged. When enabled, the isolated
        nucleus classifies the pair (LINEAR / NONLINEAR / INSUFFICIENT);
        LINEAR transforms via Two-Point and NONLINEAR attempts ALASS on the
        unchanged original target, each subject to the existing production
        analyzer, Target Binding, and serve gates with no verdict rewritten.
        Any failure at any stage returns None (serve the original).
        """
        from app.services.sync.hybrid_router import (
            GATE_SETTING as HYBRID_GATE_LABEL,
        )
        from app.services.sync.hybrid_router import (
            MODEL_VERSION as HYBRID_MODEL_VERSION,
        )
        from app.services.sync.hybrid_router import (
            STRATEGY_ID_LINEAR as HYBRID_LINEAR,
        )
        from app.services.sync.hybrid_router import (
            STRATEGY_ID_NONLINEAR as HYBRID_NONLINEAR,
        )
        from app.services.sync.hybrid_router import (
            attempt_hybrid,
        )
        from app.services.sync.hybrid_router import (
            is_enabled as hybrid_is_enabled,
        )

        if not hybrid_is_enabled():
            logger.info(
                "[hybrid] sub=%s evaluated=false reason=disabled gate=%s",
                target_id, HYBRID_GATE_LABEL,
            )
            return None
        from app.services.sync.hybrid_router import scope_allows as hybrid_scope_allows

        if not hybrid_scope_allows(meta):
            logger.info(
                "[hybrid] sub=%s evaluated=false reason=scoped_out gate=%s",
                target_id, HYBRID_GATE_LABEL,
            )
            return None
        if not target_text or not reference_text:
            logger.debug("[hybrid] sub=%s evaluated=false reason=missing_input", target_id)
            return None

        async def _runner(original_target: str, eligible_reference: str) -> str | None:
            if self._sync_service is None:
                return None
            return await self._sync_service.sync_async(
                original_target,
                eligible_reference,
                decision_kind=resolved.kind if resolved is not None else "edition",
                is_series=bool(meta.get("season") or meta.get("episode")),
                source_confirmed=False,
                relaxed=False,
                reference_partial=False,
            )

        try:
            result = await attempt_hybrid(
                target_text,
                reference_text,
                analyzer=self._analyzer,
                alass_runner=_runner,
                reference_partial=bool(resolved.partial) if resolved is not None else False,
                unified_authority=True,
            )
        except Exception as exc:  # fail closed, always
            logger.info("[hybrid] sub=%s evaluated=false error=%s", target_id, type(exc).__name__)
            return None
        strategy = HYBRID_LINEAR if result.route == "linear" else (
            HYBRID_NONLINEAR if result.route == "nonlinear" else "hybrid_unrouted"
        )
        correspondence = result.correspondence or {}
        logger.info("[unified] hybrid_diagnostics=%s", correspondence)
        logger.info(
            "[hybrid] sub=%s evaluated=true route=%s accepted=%s "
            "scale=%s offset_ms=%s timings_ms=%s reasons=%s",
            target_id,
            result.route,
            result.accepted,
            f"{result.scale:.6f}" if result.scale is not None else "n/a",
            f"{result.offset_ms:.0f}" if result.offset_ms is not None else "n/a",
            ",".join(f"{key}={value:.0f}" for key, value in result.timings_ms.items()) or "n/a",
            ",".join(result.reason_codes) or "none",
        )
        logger.info(
            "[hybrid] sub=%s verification method=%s candidate=%s groups=%s/%s "
            "coverage=%s median=%sms p90=%sms p95=%sms max=%sms movement=%sms "
            "discontinuities=%s outliers=%s clamped=%s failure=%s",
            target_id,
            correspondence.get("verification_method", "n/a"),
            correspondence.get("candidate_algorithm", "n/a"),
            correspondence.get("independent_groups_used", "n/a"),
            correspondence.get("correspondence_groups_found", "n/a"),
            correspondence.get("regional_coverage", "n/a"),
            correspondence.get("median_residual_ms", "n/a"),
            correspondence.get("p90_residual_ms", "n/a"),
            correspondence.get("p95_residual_ms", "n/a"),
            correspondence.get("max_residual_ms", "n/a"),
            correspondence.get("movement_magnitude_ms", "n/a"),
            correspondence.get("regional_discontinuities", "n/a"),
            correspondence.get("outlier_count", "n/a"),
            correspondence.get("clamped", "n/a"),
            correspondence.get("failure_reasons", "n/a"),
        )
        self._last_evaluation = result.evaluation
        if not result.accepted or not result.transformed_text:
            return None
        evaluation = result.evaluation
        if evaluation is None:
            return None
        trust = resolved.reference_trust if resolved else None
        evaluation.reference_trust = trust
        if trust is not None and trust not in {"verified", "strong"}:
            evaluation.set_verdict(SyncState.PROBABLE_SYNC, evaluation.verification)
            evaluation.reasons.append(f"reference trust '{trust}' caps the verified claim")
        binding = self._target_binding(
            resolved or ResolvedReference(reference_text), meta
        )
        if not self._gate_target_binding(evaluation, binding):
            return None
        if not may_serve_synchronized(evaluation.sync_state, evaluation.verification):
            return None
        transformed = result.transformed_text.encode("utf-8")
        capture_transformed(reference_text, result.transformed_text, strategy, evaluation)
        key = build_synced_cache_key(
            meta, target_id, content_hash=content_hash,
            decision=f"{strategy}-{HYBRID_MODEL_VERSION}",
        )
        self._last_evaluation = evaluation
        await self._store_verdict(
            verdict_key, evaluation, content_hash, meta, target_id, artifact_key=key,
            artifact_bytes=transformed,
            reference_sha=hashlib.sha256(reference_text.encode("utf-8")).hexdigest(),
        )
        self._audit_serve(evaluation, meta, target_id, from_cache=False, resolved=resolved)
        self._metrics["candidates_verified"] += 1
        if self._sync_cache is not None:
            await self._sync_cache.clear_failed(resolution_key)
            await self._sync_cache.set(key, transformed)
            await self._sync_cache.set(resolution_key, transformed)
            await self._sync_cache.set_meta(key, {
                "status": "synced",
                "applied_shift": self._applied_shifts(target_text, result.transformed_text),
                "reference_sha": hashlib.sha256(reference_text.encode("utf-8")).hexdigest(),
                "decision": strategy,
                "verification_method": (
                    correspondence.get("verification_method", "unified")
                ),
                "timestamp": datetime.now(UTC).isoformat(),
                "sync_state": evaluation.sync_state.value,
                "verification": evaluation.verification.value,
                "sync_confidence": evaluation.sync_confidence,
                **self._binding_metadata(binding),
                **self._methodology_metadata(meta),
                "artifact_sha256": hashlib.sha256(transformed).hexdigest(),
            })
            await self._sync_cache.set_meta(resolution_key, {
                "reference_sha": hashlib.sha256(reference_text.encode("utf-8")).hexdigest(),
                "sync_state": evaluation.sync_state.value,
                "verification": evaluation.verification.value,
                "sync_confidence": evaluation.sync_confidence,
                **self._binding_metadata(binding),
                **self._methodology_metadata(meta),
                "artifact_sha256": hashlib.sha256(transformed).hexdigest(),
            })
        self._record_outcome(
            meta, classify_outcome(True, strategy, False)
        )
        logger.info(
            "[sync] delivery sub=%s decision=TRANSFORMED state=%s verification=%s "
            "strategy=%s verification_method=%s original_sha256=%s "
            "served_sha256=%s served_is_transformed=True bytes=%d",
            target_id,
            evaluation.sync_state.value,
            evaluation.verification.value,
            strategy,
            correspondence.get("verification_method", "unified"),
            _content_digest(sub_bytes),
            _content_digest(transformed),
            len(transformed),
        )
        timing = current_sync_timing.get()
        if timing is not None:
            timing.winning_sync_strategy = strategy
        capture_selected(reference_text, result.transformed_text, strategy)
        self._promote_sticky_reference(
            meta=meta, target_id=target_id, pristine_text=target_text,
            resolved=resolved, evaluation=evaluation)
        return self._delivery_payload(transformed, sub_bytes, evaluation, meta, target_id,
            hashlib.sha256(reference_text.encode("utf-8")).hexdigest())

    async def _hybrid_predelivery_gate(
        self,
        *,
        target_id: str,
        target_text: str,
        reference_text: str,
        synced_text: str,
        meta: dict,
        exemption: str,
        legacy_unverified: bool = True,
    ) -> bool:
        """Correspondence V2 review before an exemption-driven delivery.

        Not currently called from the serving loop (see the note there): the
        only call site sat on a path where this review is a guaranteed no-op.
        Kept, with its tests, for wiring into a path where it can decide.

        Returns True to proceed or False to downgrade this candidate to
        fallback (loop continues; nothing is served or cached as verified).
        Emits one request-correlated telemetry line; never logs subtitle
        text. Verification runs bounded off-loop; a timeout fails closed
        exactly like an internal error.
        """
        from app.services.sync.hybrid_router import pre_delivery_review_async

        proceed, telemetry = await pre_delivery_review_async(
            target_text=target_text,
            reference_text=reference_text,
            synced_text=synced_text,
            meta=meta,
            exemption=exemption,
            legacy_unverified=legacy_unverified,
        )
        logger.info(
            "[hybrid] pre_delivery sub=%s evaluated=%s method=%s "
            "verdict=%s proceed=%s candidate_sha=%s served_sha=%s "
            "reasons=%s",
            target_id,
            telemetry.get("evaluated", False),
            telemetry.get("verification_method", "n/a"),
            telemetry.get("verdict", "n/a"),
            proceed,
            telemetry.get("candidate_sha", "n/a"),
            telemetry.get("served_sha", "n/a"),
            ",".join(str(reason) for reason in telemetry.get("reasons", ())) or "none",
        )
        return proceed

    @staticmethod
    def _target_binding(resolved: ResolvedReference, meta: dict) -> BindingAssessment:
        reference = parse_release_identity(resolved.candidate)
        # Member identity is proven by existing extraction/re-scoring (or retained
        # cache proof), never copied from the request. Season-pack names alone
        # cannot provide an episode. Contradictory evidence remains ineligible.
        for field, proven in (("season", resolved.member_season),
                              ("episode", resolved.member_episode)):
            if proven is None or not re.fullmatch(r"[0-9]+", str(proven)):
                continue
            number = int(proven)
            if field == "episode" and number == 0:
                continue
            recorded = getattr(reference, field)
            if recorded is None:
                reference = (replace(reference, season=number) if field == "season"
                             else replace(reference, episode=number))
            elif recorded != number:
                reference = replace(reference, ambiguous_fields=tuple(sorted(
                    set(reference.ambiguous_fields) | {field})))
        binding = classify_target_binding(
            meta.get("target_filename"), resolved.candidate,
            reference_metadata=reference,
            exact_moviehash=(resolved.kind == "hash" and bool(str(meta.get("video_hash") or "").strip())),
            embedded=resolved.kind == "embedded",
        )
        if unified_authority_enabled(meta):
            from app.services.sync.release_metadata import words

            target = binding.target
            same_episode = (target.season is not None and target.season == reference.season
                            and target.episode is not None and target.episode == reference.episode)
            same_movie = (target.season is None and reference.season is None
                          and target.year is not None and target.year == reference.year)
            if (not target.ambiguous_fields and not reference.ambiguous_fields
                    and target.title and reference.title and words(target.title) == words(reference.title)
                    and (not target.year or not reference.year or target.year == reference.year)
                    and (same_episode or same_movie)):
                # Binding establishes media identity, not release compatibility.
                # Only independently measured dialogue can authorize timing.
                binding = replace(binding, state=TargetBinding.STRONG,
                                  reasons=("media_identity_confirmed",))
        if resolved.from_season_pack and resolved.proven_video_lineage:
            from app.services.sync.season_pack import lineages_match, video_lineage

            # Association was proved by a winning member, not by the pack's
            # filename. Exact member identity and the current lineage still
            # have to match; a title-only member name needs no new ranking.
            if (lineages_match(resolved.proven_video_lineage, video_lineage(meta.get("target_filename")))
                    and resolved.member_season == meta.get("season")
                    and resolved.member_episode == meta.get("episode")
                    and not reference.ambiguous_fields):
                binding = replace(binding, state=TargetBinding.STRONG,
                                  reasons=("season_pack_proven_video_lineage",))
        # Log canonical technical metadata only; never raw names or request config.
        logger.info(
            "[target-binding] provider=%s candidate_sha=%s state=%s reasons=%s "
            "target_edition=%s reference_edition=%s "
            "target_cut=%s reference_cut=%s target_group=%s reference_group=%s "
            "target_source=%s reference_source=%s",
            resolved.provider if resolved.provider in {"subdl", "subsource", "opensubtitles"}
            else ("opensubtitles" if resolved.kind == "hash" else "unknown"),
            hashlib.sha256(resolved.candidate.encode()).hexdigest()[:12],
            binding.state.value, ",".join(binding.reasons),
            binding.target.edition_tags, binding.reference.edition_tags,
            binding.target.cut_tags, binding.reference.cut_tags,
            binding.target.release_group, binding.reference.release_group,
            binding.target.source_family, binding.reference.source_family,
        )
        return binding

    @staticmethod
    def _methodology_metadata(meta: dict) -> dict:
        return ({"verification_method": "unified", "verification_version": "timing-v3"}
                if unified_authority_enabled(meta) else {})

    @staticmethod
    def _binding_metadata(binding: BindingAssessment) -> dict:
        return {"target_binding": binding.state.value,
                "target_binding_reasons": list(binding.reasons)}

    @staticmethod
    def _affine_corroborates_weak_binding(binding: BindingAssessment) -> bool:
        """Strict identity eligibility for request-local timing verification.

        Shared by accepted-affine and ALASS handoffs. Never infer episode identity
        for season-pack labels or movies, upgrade binding authority, or let timing
        evidence override cut identity.
        """
        target, reference = binding.target, binding.reference
        return (
            binding.state is TargetBinding.WEAK
            and binding.reasons == ("source_family_only",)
            and not target.ambiguous_fields and not reference.ambiguous_fields
            and target.season is not None and target.season == reference.season
            and target.episode is not None and target.episode == reference.episode
            and compatible_source(target.source_family, reference.source_family)
            and target.cut_tags == reference.cut_tags
            and target.edition_tags == reference.edition_tags
        )

    @staticmethod
    def _gate_target_binding(
        evaluation: SubtitleEvaluation, binding: BindingAssessment, *,
        affine_reference_corroborated: bool = False,
        source_family_reference_verified: bool = False,
    ) -> bool:
        evaluation.target_binding = binding.state.value
        evaluation.target_binding_reasons = list(binding.reasons)
        if binding.supports_verified:
            return True
        if ((affine_reference_corroborated or source_family_reference_verified)
                and SyncOrchestrator._affine_corroborates_weak_binding(binding)
                and evaluation.verification is VerificationAvailability.VERIFIED
                and may_serve_synchronized(evaluation.sync_state, evaluation.verification)):
            evaluation.reasons.append("affine_reference_corroborated" if affine_reference_corroborated
                                      else "source_family_reference_verified")
            return True
        if is_reusable_verified(evaluation.sync_state, evaluation.verification):
            logger.info("[sync] alignment accepted but target binding insufficient; "
                        "continuing to next external candidate state=%s reasons=%s",
                        binding.state.value, ",".join(binding.reasons))
            evaluation.set_verdict(SyncState.UNVERIFIED, VerificationAvailability.UNKNOWN)
            evaluation.reasons.append("target_binding_insufficient")
        return False

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
