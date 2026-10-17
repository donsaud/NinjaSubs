"""Bounded numeric telemetry owned by one synchronization job."""
from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from uuid import uuid4

logger = logging.getLogger(__name__)

#: Diagnostic outcomes a reference attempt can end in. Deliberately a closed
#: allowlist so a provider-controlled string can never reach the logs.
REFERENCE_OUTCOMES = (
    "cache_hit_video_reference",
    "cache_miss_video_reference",
    "cached_reference_validation_failure",
    "candidate_rejected",
    "provider_circuit_breaker_open",
    "provider_rate_limited",
    "provider_timeout",
    "download_failure",
    "reference_content_identical_to_target",
    "no_independent_reference_available",
)


@dataclass
class SyncTiming:
    job_id: str = field(default_factory=lambda: uuid4().hex[:12])
    external_plan_available: bool = False
    external_plan_started: bool = False
    external_plan_ms: float | None = None
    cached_reference_present: bool = False
    cached_reference_attempted: bool = False
    #: A sticky verified reference supplied this request's first attempt,
    #: so provider planning was eligible to be skipped.
    sticky_reference_attempted: bool = False
    #: A retained same-lineage season-pack member supplied this request's
    #: reference, so provider planning was eligible to be skipped.
    season_pack_reference_attempted: bool = False
    #: Candidates the external planner RETURNED, before any download.
    candidate_count_planned: int = 0
    #: Every reference attempt the loop actually reached, across all strategies,
    #: including strategies that have no planner and any pre-resolved cache
    #: entry. It is therefore >= candidate_count_planned, and the two are not
    #: meant to be equal: the difference is the non-planned attempts.
    candidates_considered: int = 0
    references_downloaded: int = 0
    references_deduplicated: int = 0
    alass_processes_launched: int = 0
    #: Wall-clock cost of actual reference payload transfer (excluding pacing
    #: and archive extraction) and of decoding/extracting the payload.
    reference_network_ms: float = 0.0
    reference_decode_ms: float = 0.0
    #: Bounded progressive-preparation counters. ``prefetch_started`` counts
    #: speculative downloads begun, ``prefetch_reused`` those already ready
    #: when the loop reached them, ``prefetch_wasted`` those completed but
    #: never consumed because an earlier candidate verified, and
    #: ``prefetch_cancelled`` those cancelled at the same point.
    prefetch_started: int = 0
    prefetch_reused: int = 0
    prefetch_wasted: int = 0
    prefetch_cancelled: int = 0
    #: Lightweight preflight (existing cue/anchor evidence). It may reorder
    #: same-release-family candidates but never accepts or excludes one.
    preflight_runs: int = 0
    preflight_reorders: int = 0
    preflight_ms: float = 0.0
    #: Per-algorithm wall clock, for latencies that used to be invisible in
    #: the aggregate: affine search, anchor pre-shift, ALASS processes, and
    #: the verifier itself.
    timeline_affine_ms: float = 0.0
    anchor_preshift_ms: float = 0.0
    alass_total_ms: float = 0.0
    verification_total_ms: float = 0.0
    #: Bounded quality-aware fallback counters. A weak-binding guarded result
    #: is held provisionally and the loop may try extra independent references.
    quality_fallback_triggered: bool = False
    quality_fallback_attempts: int = 0
    quality_fallback_served: bool = False
    quality_fallback_exhausted: bool = False
    quality_fallback_first_p95_ms: float | None = None
    quality_fallback_first_binding: str | None = None
    winning_candidate_index: int | None = None
    pal_reference_kind: str | None = None
    pal_reference_index: int | None = None
    reference_kind: str | None = None
    winning_sync_strategy: str | None = None
    cache_hit: bool = False
    #: Metadata-gate outcome for this request (fast_original, investigate,
    #: identity_conflict, or None when the gate abstained before judging).
    #: Closed allowlist enforced by the gate module; never a raw filename.
    metadata_gate_decision: str | None = None
    #: Primary gate reason code (same allowlist discipline as the decision).
    metadata_gate_reason: str | None = None
    #: Wall-clock cost of the local metadata comparison (no I/O inside).
    metadata_gate_ms: float | None = None
    #: Bounded counts of why reference attempts ended, keyed by REFERENCE_OUTCOMES.
    reference_outcomes: dict = field(default_factory=dict)
    #: Fixed telemetry definitions. A planned candidate is an attempt slot
    #: returned before any download; a considered candidate is an attempt the
    #: loop actually reached, including cache entries and strategies without a
    #: planner. The latter is therefore normally greater than or equal to the
    #: former, and their difference is not itself a defect.
    planned_definition: str = "attempt_slots_returned_before_download"
    considered_definition: str = "attempts_actually_reached_across_strategies"

    def note_reference_failure(self, reason: str | None) -> None:
        """Record why an attempt ended. Unknown reasons are ignored, never logged raw."""
        if reason not in REFERENCE_OUTCOMES:
            return
        counts = self.reference_outcomes
        counts[reason] = counts.get(reason, 0) + 1

    def log(self, result: str, total_ms: float) -> None:
        # Avoided means successful completion with a configured planner that was not run.
        # Errors, cancellations, and ordinary failure fallbacks are not latency savings.
        # A metadata-fast delivery avoided planning by policy decision, not by luck.
        gate_fast = self.metadata_gate_decision == "fast_original"
        avoided = (
            result in {"original", "transformed", "cached"}
            and self.external_plan_available
            and not self.external_plan_started
            and (self.cache_hit or self.winning_sync_strategy is not None or gate_fast)
        )
        # Provenance is an allowlisted category, never a provider-controlled string.
        kind = self.reference_kind if self.reference_kind in {
            "hash", "edition", "group", "embedded", "bluray_candidate", "team"
        } else "other"
        outcomes = " ".join(
            f"{name}={self.reference_outcomes.get(name, 0)}" for name in REFERENCE_OUTCOMES
        )
        logger.info(
            "[sync-latency] job=%s external_plan_started=%s external_plan_avoided=%s "
            "external_plan_ms=%s cached_reference_present=%s cached_reference_attempted=%s "
            "sticky_reference_attempted=%s "
            "season_pack_reference_attempted=%s "
            "metadata_gate_decision=%s metadata_gate_reason=%s metadata_gate_ms=%s "
            "candidate_count_planned=%d candidates_considered=%d references_downloaded=%d "
            "references_deduplicated=%d ALASS_processes_launched=%d "
            "reference_network_ms=%.1f reference_decode_ms=%.1f "
            "prefetch_started=%d prefetch_reused=%d prefetch_wasted=%d "
            "prefetch_cancelled=%d preflight_runs=%d preflight_reorders=%d "
            "preflight_ms=%.1f timeline_affine_ms=%.1f anchor_preshift_ms=%.1f "
            "alass_total_ms=%.1f verification_total_ms=%.1f "
            "quality_fallback_triggered=%s quality_fallback_attempts=%d "
            "quality_fallback_served=%s quality_fallback_exhausted=%s "
            "quality_fallback_first_p95_ms=%s quality_fallback_first_binding=%s "
            "planned_definition=%s considered_definition=%s "
            "winning_candidate_index=%s winning_reference_kind=%s winning_sync_strategy=%s "
            "total_ms=%.3f result=%s | reference_outcomes: %s",
            self.job_id, self.external_plan_started, avoided,
            self.external_plan_ms, self.cached_reference_present, self.cached_reference_attempted,
            self.sticky_reference_attempted,
            self.season_pack_reference_attempted,
            self.metadata_gate_decision, self.metadata_gate_reason, self.metadata_gate_ms,
            self.candidate_count_planned, self.candidates_considered, self.references_downloaded,
            self.references_deduplicated, self.alass_processes_launched,
            self.reference_network_ms, self.reference_decode_ms,
            self.prefetch_started, self.prefetch_reused, self.prefetch_wasted,
            self.prefetch_cancelled, self.preflight_runs, self.preflight_reorders,
            self.preflight_ms, self.timeline_affine_ms, self.anchor_preshift_ms,
            self.alass_total_ms, self.verification_total_ms,
            self.quality_fallback_triggered, self.quality_fallback_attempts,
            self.quality_fallback_served, self.quality_fallback_exhausted,
            self.quality_fallback_first_p95_ms, self.quality_fallback_first_binding,
            self.planned_definition, self.considered_definition,
            self.winning_candidate_index, kind if self.winning_sync_strategy is not None else None,
            self.winning_sync_strategy, total_ms, result, outcomes,
        )


current_sync_timing: ContextVar[SyncTiming | None] = ContextVar("sync_timing", default=None)
