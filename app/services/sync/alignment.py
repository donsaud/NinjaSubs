"""Alignment analysis: measure what ``alass`` actually did, and how much to trust it.

``alass`` is an *alignment engine*, not a truth detector. It exits 0 and writes a
file for inputs it cannot align, so exit status is not evidence. This module
compares the pre-sync target cues against the post-sync cues and reports the
statistics that do constitute evidence: a stable global shift, a progressive
drift, or a piecewise discontinuity.

Design rules, in priority order:

1. **Missing evidence stays missing.** Every metric is ``Optional``. A metric
   that cannot be computed is never replaced by a favourable default, so an
   under-determined alignment cannot masquerade as a verified one.
2. **Content match is not synchronization.** These numbers say nothing about
   whether the subtitle belongs to the release; that remains
   ``calculate_compatibility``'s job and the existing ``MatchTier`` hierarchy.
3. **Fail closed on the sync claim.** :func:`evaluate_sync_state` only returns
   ``VERIFIED_*`` when enough independent measurements line up. Anything
   ambiguous degrades to ``UNVERIFIED``.

All cue parsing, the median offset, cue-sanity validation, and FPS-relation
logic are reused from the existing matcher rather than reimplemented.
"""

from __future__ import annotations

import logging
import math
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.services.subtitle_matcher import (
    ALIGNED_OFFSET_THRESHOLD_S,
    FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
    determine_fps_relation,
    parse_srt_cues,
    validate_cue_sanity,
)

logger = logging.getLogger(__name__)

Cue = tuple[int, int, str]

# --------------------------------------------------------------------------- #
# Thresholds. Every one is deliberately conservative and lives here rather than
# being scattered through the decision logic.
# --------------------------------------------------------------------------- #

# Minimum paired cues before drift/segment statistics mean anything at all.
MIN_CUES_FOR_ANALYSIS = 8
# Minimum pairs before a percentile is considered measurable.
MIN_CUES_FOR_PERCENTILES = 5
# Offsets beyond this (ms) are a different cut, not an alignment result.
MAX_PLAUSIBLE_OFFSET_MS = FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS
# A "stable" alignment keeps the 95th-percentile |delta| under this.
MAX_P95_MS_FOR_STABLE = 2000.0
# Median absolute deviation ceiling for a stable alignment.
MAX_MAD_MS_FOR_STABLE = 800.0
# |slope| (ms per minute) above which the shift is progressive, not global.
MAX_DRIFT_MS_PER_MINUTE = 120.0
# A change point is a step in the running median beyond this magnitude.
CHANGE_POINT_MS = 5000.0
# Minimum number of aligned cues required before claiming any sync at all.
MIN_CUES_FOR_VERIFICATION = 5
# Cue-count retention: fewer than this fraction of input cues surviving means
# alass dropped real content, whatever its exit code.
MIN_CUE_RETENTION = 0.80
# Movement pairing is deliberately loose: a legitimate re-timing can move a cue
# a long way (PAL speed-up, a recap, an inserted scene), and those large moves
# are exactly the change points we want to measure. Residual pairing against the
# reference stays tight, because a good alignment leaves almost nothing behind.
MOVEMENT_TOLERANCE_MS = 30_000
RESIDUAL_TOLERANCE_MS = 5_000


class SyncState(str, Enum):
    """How much is actually known about a subtitle's timing."""

    # Already aligned with the target video.
    VERIFIED_SYNCED = "verified_synced"
    # Was misaligned; alass aligned it and the result measures as sound.
    VERIFIED_RESYNCED = "verified_resynced"
    # Suggestive but unverified.
    PROBABLE_SYNC = "probable_sync"
    # Not enough evidence to make any synchronization claim.
    UNVERIFIED = "unverified"
    # Evidence says this should not be served as synchronized.
    REJECTED = "rejected"

    @property
    def rank(self) -> int:
        """Presentation priority; lower sorts first."""
        return _STATE_RANK[self]


_STATE_RANK: dict[SyncState, int] = {
    SyncState.VERIFIED_SYNCED: 0,
    SyncState.VERIFIED_RESYNCED: 1,
    SyncState.PROBABLE_SYNC: 2,
    SyncState.UNVERIFIED: 3,
    SyncState.REJECTED: 4,
}


class RejectionReason(str, Enum):
    """Why a synchronization claim was refused. Distinct from content rejection."""

    WRONG_CONTENT = "wrong_content"
    INVALID_SUBTITLE = "invalid_subtitle"
    INVALID_TIMINGS = "invalid_timings"
    LOW_CONFIDENCE = "low_confidence"
    SYNC_VERIFICATION_FAILED = "sync_verification_failed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    IMPLAUSIBLE_OFFSET = "implausible_offset"
    ALASS_FAILED = "alass_failed"
    CUE_LOSS = "cue_loss"
    OTHER = "other"


class SubtitleEvaluation(BaseModel):
    """Internal, explainable evaluation of one subtitle candidate.

    Content compatibility and synchronization confidence are deliberately
    separate fields. A high ``content_match_score`` never implies a
    synchronization state: they answer different questions.
    """

    content_match_score: float | None = None
    sync_confidence: float | None = None
    verification_confidence: float | None = None

    # Alignment measurements; ``None`` means "not measurable", never "good".
    median_offset_ms: float | None = None
    mad_offset_ms: float | None = None
    p95_offset_ms: float | None = None
    drift_ms_per_minute: float | None = None
    coverage_score: float | None = None
    consensus_score: float | None = None

    alass_applied: bool = False
    alass_successful: bool = False

    sync_state: SyncState = SyncState.UNVERIFIED
    reasons: list[str] = Field(default_factory=list)
    rejection_reason: RejectionReason | None = None

    # Change points as (position_ms, step_ms) pairs describing piecewise shifts.
    change_points: list[tuple[int, float]] = Field(default_factory=list)

    def explain(self) -> str:
        """Single-line, human-readable rationale for logs and debug output."""
        parts = [f"state={self.sync_state.value}"]
        if self.content_match_score is not None:
            parts.append(f"content={self.content_match_score:.0f}")
        if self.sync_confidence is not None:
            parts.append(f"sync_conf={self.sync_confidence:.0f}")
        if self.median_offset_ms is not None:
            parts.append(f"median={self.median_offset_ms:+.0f}ms")
        if self.mad_offset_ms is not None:
            parts.append(f"mad={self.mad_offset_ms:.0f}ms")
        if self.p95_offset_ms is not None:
            parts.append(f"p95={self.p95_offset_ms:.0f}ms")
        if self.drift_ms_per_minute is not None:
            parts.append(f"drift={self.drift_ms_per_minute:+.1f}ms/min")
        if self.rejection_reason is not None:
            parts.append(f"reject={self.rejection_reason.value}")
        return " ".join(parts) + (f" | {'; '.join(self.reasons)}" if self.reasons else "")


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of an already-sorted-agnostic list."""
    if not values:
        raise ValueError("percentile of empty sequence")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def pair_cue_starts(
    before: list[Cue],
    after: list[Cue],
    *,
    tolerance_ms: int = 5_000,
) -> list[tuple[int, float]]:
    """Pair pre/post cue starts, returning ``(position_ms, delta_ms)`` pairs.

    Cues are matched by nearest start within ``tolerance_ms`` so a uniform shift
    pairs one-to-one instead of collapsing onto a single neighbour. An unmatched
    cue in either side is dropped rather than guessed at, which keeps the
    statistics honest when alass merges or splits cues.
    """
    if not before or not after:
        return []
    after_starts = [start for start, _, _ in after]
    pairs: list[tuple[int, float]] = []
    used: set[int] = set()
    for start, _, _ in sorted(before, key=lambda c: c[0]):
        best_index = -1
        best_distance = tolerance_ms + 1
        for offset in range(len(after_starts)):
            distance = abs(after_starts[offset] - start)
            if distance < best_distance and offset not in used:
                best_distance = distance
                best_index = offset
        if best_index >= 0 and best_distance <= tolerance_ms:
            used.add(best_index)
            pairs.append((start, float(after_starts[best_index] - start)))
    return pairs


def analyze_drift(pairs: list[tuple[int, float]]) -> float | None:
    """Least-squares slope of delta over time, in milliseconds per minute.

    Distinguishes a *stable global offset* from *progressive drift*:
    a stable alignment has a slope near zero, whereas a subtitle being
    progressively re-timed does not.
    """
    if len(pairs) < MIN_CUES_FOR_ANALYSIS:
        return None
    xs = [position / 60_000.0 for position, _ in pairs]
    ys = [delta for _, delta in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    variance = sum((x - mean_x) ** 2 for x in xs)
    if variance <= 0.0:
        return None
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=False))
    return covariance / variance


def detect_change_points(
    pairs: list[tuple[int, float]],
    *,
    threshold_ms: float = CHANGE_POINT_MS,
    window: int = 4,
) -> list[tuple[int, float]]:
    """Locate piecewise timing steps via a step change in the running median.

    A change point is a *structural* difference (scene insertion, recap,
    different cut), not automatically an error, so this is reported as evidence
    for the decision engine rather than used as a rejection rule on its own.
    """
    if len(pairs) < MIN_CUES_FOR_ANALYSIS * 2:
        return []
    deltas = [delta for _, delta in pairs]
    found: list[tuple[int, float]] = []
    for index in range(window, len(pairs) - window):
        before = deltas[index - window : index]
        after = deltas[index : index + window]
        step = _median(after) - _median(before)
        if abs(step) >= threshold_ms:
            # Collapse runs of adjacent detections into the first point.
            if not found or pairs[index][0] - found[-1][0] > window * 5_000:
                found.append((pairs[index][0], step))
    return found


class AlignmentAnalyzer:
    """Measure an alass result and classify the synchronization evidence.

    ``analyze`` is a pure function of its inputs: no I/O, no globals, no
    mutation of the subtitles it is given. That keeps the whole decision
    testable without spawning the alass binary.
    """

    def __init__(self, *, min_cues: int = MIN_CUES_FOR_VERIFICATION) -> None:
        self.min_cues = min_cues

    @staticmethod
    def _as_cues(value: str | list[Cue] | None) -> list[Cue]:
        if value is None:
            return []
        if isinstance(value, str):
            return parse_srt_cues(value)
        return sorted(value, key=lambda cue: cue[0])

    def analyze(
        self,
        target: str | list[Cue],
        synced: str | list[Cue] | None,
        reference: str | list[Cue] | None = None,
        *,
        alass_applied: bool = False,
        alass_successful: bool = False,
        content_match_score: float | None = None,
        target_fps: float | None = None,
        reference_fps: float | None = None,
    ) -> SubtitleEvaluation:
        """Compare pre/post cues and classify the result conservatively."""
        target_cues = self._as_cues(target)
        synced_cues = self._as_cues(synced)
        evaluation = SubtitleEvaluation(
            alass_applied=alass_applied,
            alass_successful=alass_successful,
            content_match_score=content_match_score,
        )

        if len(target_cues) < self.min_cues:
            evaluation.sync_state = SyncState.UNVERIFIED
            evaluation.rejection_reason = RejectionReason.INSUFFICIENT_EVIDENCE
            evaluation.reasons.append(
                f"target has {len(target_cues)} cues, below the {self.min_cues} needed to verify"
            )
            return evaluation

        if not alass_applied:
            # No alignment ran. The only honest claim available is whether the
            # subtitle was *already* close to the reference.
            return self._classify_without_alignment(
                evaluation, target_cues, reference, target_fps, reference_fps
            )

        if not alass_successful or not synced_cues:
            evaluation.sync_state = SyncState.UNVERIFIED
            evaluation.rejection_reason = RejectionReason.ALASS_FAILED
            evaluation.reasons.append("alass did not produce a usable aligned subtitle")
            return evaluation

        # Coverage: how much of the target survived the sync. Cues lost by alass
        # mean real content disappeared, whatever the exit code was.
        coverage = min(1.0, len(synced_cues) / len(target_cues))
        evaluation.coverage_score = round(coverage, 4)
        if coverage < MIN_CUE_RETENTION:
            evaluation.sync_state = SyncState.REJECTED
            evaluation.rejection_reason = RejectionReason.CUE_LOSS
            evaluation.reasons.append(
                f"alass retained only {coverage:.0%} of {len(target_cues)} cues"
            )
            return evaluation

        return self._classify_alignment(evaluation, target_cues, synced_cues, reference)

    def _measure(
        self, evaluation: SubtitleEvaluation, pairs: list[tuple[int, float]]
    ) -> None:
        """Populate offset statistics from paired cue deltas."""
        if not pairs:
            return
        deltas = [delta for _, delta in pairs]
        median = _median(deltas)
        evaluation.median_offset_ms = round(median, 1)
        evaluation.mad_offset_ms = round(
            _median([abs(delta - median) for delta in deltas]), 1
        )
        if len(deltas) >= MIN_CUES_FOR_PERCENTILES:
            evaluation.p95_offset_ms = round(
                _percentile([abs(delta) for delta in deltas], 0.95), 1
            )
        evaluation.drift_ms_per_minute = drift = analyze_drift(pairs)
        if drift is not None:
            evaluation.drift_ms_per_minute = round(drift, 2)
        evaluation.change_points = detect_change_points(pairs)

    def _classify_alignment(
        self,
        evaluation: SubtitleEvaluation,
        target_cues: list[Cue],
        synced_cues: list[Cue],
        reference: str | list[Cue] | None,
    ) -> SubtitleEvaluation:
        """Classify an alignment that alass actually produced.

        Two different quantities matter and must not be conflated:

        * **movement** - how far alass moved each cue. A large, *consistent*
          movement is exactly what a correct re-timing looks like, so its
          magnitude is not by itself a quality signal.
        * **residual** - how far the aligned result still sits from the
          reference. This is the actual quality signal: after a good alignment
          the residual collapses to ~0 regardless of how much alass moved.

        When a reference is available the verdict is driven by the residual,
        with the movement's stability (MAD, drift, change points) as the
        supporting evidence.
        """
        movement_pairs = pair_cue_starts(
            target_cues, synced_cues, tolerance_ms=MOVEMENT_TOLERANCE_MS
        )
        if movement_pairs:
            deltas = [delta for _, delta in movement_pairs]
            evaluation.median_offset_ms = round(_median(deltas), 1)
            evaluation.mad_offset_ms = round(
                _median([abs(delta - _median(deltas)) for delta in deltas]), 1
            )
            drift = analyze_drift(movement_pairs)
            if drift is not None:
                evaluation.drift_ms_per_minute = round(drift, 2)
            evaluation.change_points = detect_change_points(movement_pairs)

        if len(movement_pairs) < MIN_CUES_FOR_PERCENTILES:
            evaluation.sync_state = SyncState.UNVERIFIED
            evaluation.rejection_reason = RejectionReason.INSUFFICIENT_EVIDENCE
            evaluation.reasons.append(
                f"only {len(movement_pairs)} cues could be paired; alignment not measurable"
            )
            return evaluation

        reference_cues = self._as_cues(reference)
        residual_p95: float | None
        if reference_cues:
            # Residual against the reference is the authoritative signal.
            residual_pairs = pair_cue_starts(
                synced_cues, reference_cues, tolerance_ms=RESIDUAL_TOLERANCE_MS
            )
            if len(residual_pairs) < MIN_CUES_FOR_PERCENTILES:
                evaluation.sync_state = SyncState.UNVERIFIED
                evaluation.rejection_reason = RejectionReason.INSUFFICIENT_EVIDENCE
                evaluation.reasons.append(
                    "too few cues could be compared against the reference to verify"
                )
                return evaluation
            residuals = [abs(delta) for _, delta in residual_pairs]
            residual_p95 = _percentile(residuals, 0.95)
            # Report the residual as the headline offset: after a successful
            # alignment this is the meaningful "how far off are we" number.
            evaluation.median_offset_ms = round(
                _median([delta for _, delta in residual_pairs]), 1
            )
            evaluation.p95_offset_ms = round(residual_p95, 1)
        else:
            # No reference: fall back to movement stability. Anything weaker
            # than a clean, low-variance shift stays UNVERIFIED.
            if evaluation.p95_offset_ms is None:
                residuals = [abs(delta) for _, delta in movement_pairs]
                if len(residuals) >= MIN_CUES_FOR_PERCENTILES:
                    evaluation.p95_offset_ms = round(_percentile(residuals, 0.95), 1)
            residual_p95 = evaluation.p95_offset_ms

        mad = evaluation.mad_offset_ms or 0.0
        drift = evaluation.drift_ms_per_minute

        # Confidence accumulates from measured quality only; an unavailable
        # metric contributes nothing instead of defaulting to "fine".
        components = 0.0
        if residual_p95 is not None:
            components += max(0.0, 1.0 - residual_p95 / MAX_P95_MS_FOR_STABLE)
        components += max(0.0, 1.0 - mad / MAX_MAD_MS_FOR_STABLE)
        if drift is not None:
            components += max(0.0, 1.0 - abs(drift) / MAX_DRIFT_MS_PER_MINUTE)
        confidence = min(100.0, 70.0 + 30.0 * (components / 3.0))

        residual_ok = residual_p95 is not None and residual_p95 <= MAX_P95_MS_FOR_STABLE
        mad_ok = mad <= MAX_MAD_MS_FOR_STABLE
        drift_ok = drift is None or abs(drift) <= MAX_DRIFT_MS_PER_MINUTE

        if not residual_ok or not mad_ok:
            # The result is still far from the reference, or the movement was
            # too scattered to be a deliberate re-timing. This is checked first
            # so genuinely untrustworthy output can never be softened into
            # PROBABLE by the drift branch below.
            evaluation.sync_state = SyncState.UNVERIFIED
            evaluation.rejection_reason = RejectionReason.LOW_CONFIDENCE
            detail = (
                f"residual p95={residual_p95:.0f}ms" if residual_p95 is not None else "residual n/a"
            )
            evaluation.reasons.append(
                f"alass output not trustworthy: {detail} (max "
                f"{MAX_P95_MS_FOR_STABLE:.0f}ms), movement mad={mad:.0f}ms"
            )
        elif not drift_ok:
            # The alignment lands correctly but shifts progressively, so it is
            # not a clean global offset. That can be legitimate (PAL
            # speed-up, different cut), so it lowers confidence rather than
            # triggering rejection.
            evaluation.sync_state = SyncState.PROBABLE_SYNC
            evaluation.sync_confidence = round(min(confidence, 70.0), 1)
            evaluation.verification_confidence = 50.0
            evaluation.reasons.append(
                f"progressive drift {drift:+.1f}ms/min exceeds "
                f"{MAX_DRIFT_MS_PER_MINUTE:.0f}ms/min; alignment is not a stable global shift"
            )
        else:
            evaluation.sync_state = SyncState.VERIFIED_RESYNCED
            evaluation.sync_confidence = round(confidence, 1)
            evaluation.verification_confidence = evaluation.sync_confidence
            if evaluation.change_points:
                steps = ", ".join(
                    f"{pos / 1000:.0f}s{step:+.0f}ms" for pos, step in evaluation.change_points
                )
                evaluation.reasons.append(
                    f"residual p95={residual_p95:.0f}ms after a stable shift; "
                    f"structural change points noted: {steps}"
                )
            else:
                evaluation.reasons.append(
                    f"residual p95={residual_p95:.0f}ms, movement mad={mad:.0f}ms, "
                    f"drift={0.0 if drift is None else drift:+.1f}ms/min"
                )

        if evaluation.change_points and evaluation.sync_state is SyncState.VERIFIED_RESYNCED:
            evaluation.reasons.append(
                "change points: "
                + ", ".join(f"{pos / 1000:.0f}s{step:+.0f}ms" for pos, step in evaluation.change_points)
            )

        return evaluation

    def _classify_without_alignment(
        self,
        evaluation: SubtitleEvaluation,
        target_cues: list[Cue],
        reference: str | list[Cue] | None,
        target_fps: float | None,
        reference_fps: float | None,
    ) -> SubtitleEvaluation:
        """Classify a subtitle that was never re-timed.

        ``VERIFIED_SYNCED`` requires positive proof: a measurable offset against
        a real reference, and that offset already being negligible.
        """
        reference_cues = self._as_cues(reference)
        if not reference_cues:
            evaluation.sync_state = SyncState.UNVERIFIED
            evaluation.rejection_reason = RejectionReason.INSUFFICIENT_EVIDENCE
            evaluation.reasons.append("no reference available; sync state cannot be established")
            return evaluation

        pairs = pair_cue_starts(target_cues, reference_cues)
        self._measure(evaluation, pairs)
        if len(pairs) < MIN_CUES_FOR_PERCENTILES:
            # Too few cues paired to measure. That is usually because the two
            # files share almost no timing, which the existing first-dialogue
            # check can characterise: a proven mismatch is a wrong cut, not an
            # unmeasurable one. Reused rather than reimplemented.
            sanity = validate_cue_sanity(
                target_cues,
                reference_cues,
                threshold_ms=FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
            )
            if not sanity["ok"]:
                evaluation.sync_state = SyncState.REJECTED
                evaluation.rejection_reason = RejectionReason.IMPLAUSIBLE_OFFSET
                evaluation.reasons.append(f"unalignable and {sanity['reason']}")
            else:
                evaluation.sync_state = SyncState.UNVERIFIED
                evaluation.rejection_reason = RejectionReason.INSUFFICIENT_EVIDENCE
                evaluation.reasons.append("insufficient paired cues against the reference")
            return evaluation

        median_s = (evaluation.median_offset_ms or 0.0) / 1000.0
        aligned_threshold = ALIGNED_OFFSET_THRESHOLD_S
        if abs(median_s) < aligned_threshold:
            evaluation.sync_state = SyncState.VERIFIED_SYNCED
            evaluation.sync_confidence = 100.0
            evaluation.verification_confidence = 100.0
            evaluation.reasons.append(
                f"already aligned: median offset {median_s:+.2f}s "
                f"(threshold {aligned_threshold:.2f}s); no re-timing required"
            )
            return evaluation

        if abs(evaluation.median_offset_ms or 0.0) > MAX_PLAUSIBLE_OFFSET_MS:
            evaluation.sync_state = SyncState.REJECTED
            evaluation.rejection_reason = RejectionReason.IMPLAUSIBLE_OFFSET
            evaluation.reasons.append(
                f"median offset {median_s:+.2f}s is a different cut, not a sync offset"
            )
            return evaluation

        # Measurably offset but never re-timed: a candidate, not a fact.
        evaluation.sync_state = SyncState.PROBABLE_SYNC
        evaluation.sync_confidence = 40.0
        evaluation.verification_confidence = 40.0
        evaluation.reasons.append(
            f"offset {median_s:+.2f}s from the reference but never re-timed; "
            "a stable offset is suggestive, not verified"
        )
        if target_fps and reference_fps:
            relation = determine_fps_relation(target_fps, reference_fps)
            evaluation.reasons.append(f"fps relation: {relation}")
        return evaluation


def evaluate_sync_state(
    target: str | list[Cue],
    synced: str | list[Cue] | None = None,
    reference: str | list[Cue] | None = None,
    **kwargs: Any,
) -> SubtitleEvaluation:
    """Convenience wrapper around :class:`AlignmentAnalyzer`."""
    return AlignmentAnalyzer().analyze(target, synced, reference, **kwargs)
