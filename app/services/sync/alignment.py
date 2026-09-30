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

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.services.subtitle_matcher import (
    ALIGNED_OFFSET_THRESHOLD_S,
    FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
    determine_fps_relation,
    parse_srt_cues,
    validate_cue_sanity,
)
from app.services.sync.structural import (
    CutVerdict,
    StructuralSimilarity,
    classify_cut,
    compare_structures,
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
# Minimum cue population before a *verified* claim. Measurement is possible
# from 5 cues, but 5 cues cannot distinguish a real global re-timing from a
# coincidence, so below this the ceiling is PROBABLE_SYNC. The false-positive
# benchmark ("sparse_dialogue") is what sets this bar: 6 cues previously
# produced a VERIFIED_SYNCED claim.
MIN_CUES_FOR_VERIFIED = 12
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
    STRUCTURE_MISMATCH = "structure_mismatch"
    OTHER = "other"


class VerificationAvailability(str, Enum):
    """Whether synchronization knowledge is measured, recalled, or only inferred.

    This is orthogonal to :class:`SyncState`. ``SyncState`` answers "what is the
    timing verdict?"; this answers "how do we actually know it?" The pair
    together prevents the two failure modes that matter:

    * inferring a positive claim from metadata alone (``PREDICTED`` must never
      upgrade to ``VERIFIED_SYNCED``);
    * presenting a recalled verdict as if it had just been measured.
    """

    # Measured against the target video in this request and passed.
    VERIFIED = "verified"
    # A previously measured verdict for the identical video + subtitle + engine.
    CACHED = "cached"
    # Strong metadata/release evidence, but never checked against the video.
    PREDICTED = "predicted"
    # Not enough evidence for any synchronization claim.
    UNKNOWN = "unknown"

    @property
    def rank(self) -> int:
        """Strength of evidence; lower is stronger."""
        return _AVAILABILITY_RANK[self]


_AVAILABILITY_RANK: dict[VerificationAvailability, int] = {
    VerificationAvailability.VERIFIED: 0,
    VerificationAvailability.CACHED: 1,
    VerificationAvailability.PREDICTED: 2,
    VerificationAvailability.UNKNOWN: 3,
}

# Only these may ever accompany a VERIFIED_* synchronization state.
_VERIFIED_AVAILABILITIES = frozenset(
    {VerificationAvailability.VERIFIED, VerificationAvailability.CACHED}
)
# Reference trust levels that can support a verified claim. Plain strings so
# this module does not import the reference layer and create a cycle.
_REFERENCE_TRUST_SUPPORTING_VERIFIED = frozenset({"verified", "strong"})


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
    # How the state was established. Defaults to UNKNOWN, never to a positive
    # claim, so a bare SubtitleEvaluation() cannot look verified.
    verification: VerificationAvailability = VerificationAvailability.UNKNOWN
    reasons: list[str] = Field(default_factory=list)
    rejection_reason: RejectionReason | None = None

    # Change points as (position_ms, step_ms) pairs describing piecewise shifts.
    change_points: list[tuple[int, float]] = Field(default_factory=list)

    # Structural comparison against the reference (see structural.py).
    structural_similarity: float | None = None
    cut_verdict: str | None = None
    # What the reference we aligned against was worth. Measurement context only.
    reference_trust: str | None = None
    reference_consensus: float | None = None
    reference_independent_sources: int | None = None
    reference_failure: str | None = None

    #: validate_assignment is what closes the attribute-assignment bypass: the
    #: field validator below then runs on ordinary writes, not just on
    #: construction and not just through set_verdict.
    model_config = ConfigDict(validate_assignment=True)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # Route through the guard even when only a state was supplied, so
        # constructing with a bare string literal cannot produce a verified
        # verdict. The values are coerced first: pydantic coerces the fields
        # itself, but ``data`` still holds whatever the caller passed.
        state = data.get("sync_state")
        if state is None:
            state = SyncState.UNVERIFIED
        elif not isinstance(state, SyncState):
            state = SyncState(state)
        verification = data.get("verification")
        if verification is None:
            verification = VerificationAvailability.UNKNOWN
        elif not isinstance(verification, VerificationAvailability):
            verification = VerificationAvailability(verification)
        self.set_verdict(state, verification)

    @field_validator("sync_state")
    @classmethod
    def _enforce_verified_requires_evidence(cls, value: SyncState, info) -> SyncState:
        """The structural guard, enforced on every assignment.

        ``validate_assignment`` means this runs whether the state arrives by
        construction, by :meth:`set_verdict`, or by a plain attribute
        assignment several modules away. That last case is the one that used to
        work: ``evaluation.sync_state = SyncState.VERIFIED_SYNCED`` set a
        verified claim with no evidence behind it, because nothing inspected an
        ordinary attribute write.

        The check reads ``verification``, so :meth:`set_verdict` sets that
        first. That ordering is what makes a legitimate verdict possible while
        still making an unbacked one impossible.
        """
        if value not in (SyncState.VERIFIED_SYNCED, SyncState.VERIFIED_RESYNCED):
            return value
        verification = info.data.get("verification")
        if verification in _VERIFIED_AVAILABILITIES:
            return value
        return (
            SyncState.PROBABLE_SYNC
            if verification is VerificationAvailability.PREDICTED
            else SyncState.UNVERIFIED
        )

    def set_verdict(
        self, state: SyncState, verification: VerificationAvailability
    ) -> SubtitleEvaluation:
        """Set the sync state paired with how the evidence was obtained.

        This is the only supported way to establish a verdict, and
        ``_enforce_verified_requires_evidence`` backs it structurally, so a
        verified state is backed only by evidence that was actually measured,
        now or recalled from an identical video + subtitle + engine.
        """
        if state in (SyncState.VERIFIED_SYNCED, SyncState.VERIFIED_RESYNCED):
            if verification not in _VERIFIED_AVAILABILITIES:
                original = state.value
                state = (
                    SyncState.PROBABLE_SYNC
                    if verification is VerificationAvailability.PREDICTED
                    else SyncState.UNVERIFIED
                )
                if not self.rejection_reason:
                    self.rejection_reason = RejectionReason.INSUFFICIENT_EVIDENCE
                self.reasons.append(
                    f"downgraded from {original}: verification={verification.value} "
                    "cannot support a verified state"
                )
        # Verification first: the field validator reads it to permit a verified
        # state, so the order here is load-bearing.
        self.verification = verification
        self.sync_state = state
        return self

    def measured(self) -> bool:
        """True when this verdict rests on a real measurement."""
        return self.verification is not VerificationAvailability.UNKNOWN

    def explain(self) -> str:
        """Single-line, human-readable rationale for logs and debug output."""
        parts = [f"state={self.sync_state.value}", f"verification={self.verification.value}"]
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
        if self.structural_similarity is not None:
            parts.append(f"struct={self.structural_similarity:.2f}")
        if self.reference_trust is not None:
            parts.append(f"ref={self.reference_trust}")
        if self.cut_verdict:
            parts.append(f"cut={self.cut_verdict}")
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


def is_reusable_verified(state: str | None, verification: str | None) -> bool:
    """True only for a verified, measured, reusable synchronization result.

    The single definition of "this artifact may be reused as a verified
    synchronization". It is deliberately stricter than checking ``state`` alone,
    because the payload store has no store-side guarantee the way the verdict
    store does: ``SyncCache.set_verdict`` refuses to persist a ``VERIFIED_*``
    state backed by an unmeasured result, so a readable verdict implies
    measurement. A payload entry carries no such implication, so both halves
    must be checked here.

    Anything not positively verified and measured is not reusable. That
    includes ``UNVERIFIED``, ``REJECTED``, ``PROBABLE_SYNC``, ``PREDICTED`` and
    a missing state, which is the fail-closed direction.
    """
    return state in (
        SyncState.VERIFIED_SYNCED.value,
        SyncState.VERIFIED_RESYNCED.value,
    ) and verification in (
        VerificationAvailability.VERIFIED.value,
        VerificationAvailability.CACHED.value,
    )


def may_serve_synchronized(state: str | None, verification: str | None) -> bool:
    """True only when the verifier actually measured and accepted the result.

    This is the *delivery* half of the trust decision, and it is deliberately
    separate from :func:`is_reusable_verified`, which governs *caching*. The two
    answer different questions:

    * caching asks "may this artifact be recalled later as a finished sync?"
      That needs a ``VERIFIED_*`` state, so a merely probable result is not
      reusable.
    * delivery asks "may these bytes replace the original subtitle in this
      response?" That only needs the analyzer to have measured the alignment
      and accepted it, which is what ``VerificationAvailability.VERIFIED``
      records. ``PROBABLE_SYNC`` with a verified measurement is an accepted
      result whose *confidence* was capped, so it keeps its current serving
      semantics.

    The invariant this protects: a synchronization attempt the verifier does not
    trust must never replace the original subtitle. An ``UNVERIFIED``/``UNKNOWN``
    outcome means the analyzer had no trustworthy measurement -- a real Whiplash
    2160p REMUX request produced an alass output the analyzer explicitly called
    "not trustworthy" (residual p95 4200 ms against a 2000 ms bound) whose
    timestamps were displaced by up to +293 s, and it was served anyway. A
    ``REJECTED`` outcome is likewise never served.
    """
    if state == SyncState.REJECTED.value:
        return False
    return verification in (
        VerificationAvailability.VERIFIED.value,
        VerificationAvailability.CACHED.value,
    )


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
        reference_trust: str | None = None,
        reference_reasons: list[str] | None = None,
        reference_consensus: float | None = None,
        reference_independent_sources: int | None = None,
        reference_failure: str | None = None,
    ) -> SubtitleEvaluation:
        """Compare pre/post cues and classify the result conservatively."""
        target_cues = self._as_cues(target)
        synced_cues = self._as_cues(synced)
        evaluation = SubtitleEvaluation(
            alass_applied=alass_applied,
            alass_successful=alass_successful,
            content_match_score=content_match_score,
        )
        # What the reference was worth, so a withheld claim can explain itself.
        evaluation.reference_trust = reference_trust
        evaluation.reference_consensus = reference_consensus
        evaluation.reference_independent_sources = reference_independent_sources
        evaluation.reference_failure = reference_failure
        if reference_reasons:
            evaluation.reasons.extend(reference_reasons[:4])

        if len(target_cues) < self.min_cues:
            evaluation.set_verdict(SyncState.UNVERIFIED, VerificationAvailability.UNKNOWN)
            evaluation.rejection_reason = RejectionReason.INSUFFICIENT_EVIDENCE
            evaluation.reasons.append(
                f"target has {len(target_cues)} cues, below the {self.min_cues} needed to verify"
            )
            return evaluation

        # Structural comparison is measured whenever both sides have cues and
        # is independent of whether alass ran. It is supporting evidence: it
        # narrows which cut verdicts stay possible, never a rejection on its
        # own, because providers legitimately split and merge cues differently.
        structural = compare_structures(target_cues, reference)
        if structural.score is not None:
            evaluation.structural_similarity = structural.score
            evaluation.reasons.append(structural.explain())
            evaluation.reasons.extend(structural.reasons)

        if not alass_applied:
            # No alignment ran. The only honest claim available is whether the
            # subtitle was *already* close to the reference.
            return self._cap_by_evidence(
                self._classify_without_alignment(
                    evaluation, target_cues, reference, target_fps, reference_fps, structural
                ),
                len(target_cues),
                reference_trust,
            )
        if not alass_successful or not synced_cues:
            evaluation.set_verdict(SyncState.UNVERIFIED, VerificationAvailability.UNKNOWN)
            evaluation.rejection_reason = RejectionReason.ALASS_FAILED
            evaluation.reasons.append("alass did not produce a usable aligned subtitle")
            return evaluation

        # Coverage: how much of the target survived the sync. Cues lost by alass
        # mean real content disappeared, whatever the exit code was.
        coverage = min(1.0, len(synced_cues) / len(target_cues))
        evaluation.coverage_score = round(coverage, 4)
        if coverage < MIN_CUE_RETENTION:
            evaluation.set_verdict(SyncState.REJECTED, VerificationAvailability.VERIFIED)
            evaluation.rejection_reason = RejectionReason.CUE_LOSS
            evaluation.reasons.append(
                f"alass retained only {coverage:.0%} of {len(target_cues)} cues"
            )
            return evaluation

        return self._cap_by_evidence(
            self._classify_alignment(
                evaluation, target_cues, synced_cues, reference, structural
            ),
            len(target_cues),
            reference_trust,
        )

    def _cap_by_evidence(
        self,
        evaluation: SubtitleEvaluation,
        cue_count: int,
        reference_trust: str | None = None,
    ) -> SubtitleEvaluation:
        """Ceiling a verified claim by how much evidence actually existed.

        Timings can measure cleanly from a handful of cues and still be a
        coincidence rather than a real global re-timing. Below
        ``MIN_CUES_FOR_VERIFIED`` the strongest defensible claim is
        ``PROBABLE_SYNC``, however good the numbers look.
        """
        if (
            cue_count < MIN_CUES_FOR_VERIFIED
            and evaluation.sync_state
            in (SyncState.VERIFIED_SYNCED, SyncState.VERIFIED_RESYNCED)
        ):
            evaluation.set_verdict(SyncState.PROBABLE_SYNC, evaluation.verification)
            evaluation.reasons.append(
                f"only {cue_count} cues, below the {MIN_CUES_FOR_VERIFIED} required for a "
                "verified claim; capped at probable"
            )

        # Withhold-only reference gate. Verification is only as trustworthy as
        # the reference it verifies against, so an unproven reference cannot
        # support a verified claim. This only ever REMOVES a claim: it never
        # creates one, and it does not change which reference was selected.
        if (
            reference_trust is not None
            and reference_trust not in _REFERENCE_TRUST_SUPPORTING_VERIFIED
            and evaluation.sync_state
            in (SyncState.VERIFIED_SYNCED, SyncState.VERIFIED_RESYNCED)
        ):
            evaluation.set_verdict(SyncState.PROBABLE_SYNC, evaluation.verification)
            evaluation.rejection_reason = RejectionReason.SYNC_VERIFICATION_FAILED
            evaluation.reasons.append(
                f"reference trust '{reference_trust}' is not strong enough to support a "
                "verified synchronization claim"
            )
        return evaluation

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
        structural: StructuralSimilarity | None = None,
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
            evaluation.set_verdict(SyncState.UNVERIFIED, VerificationAvailability.UNKNOWN)
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
                evaluation.set_verdict(SyncState.UNVERIFIED, VerificationAvailability.UNKNOWN)
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

        # Separate "re-timable offset" from "different cut" using the measured
        # structure alongside the timing, so a large offset is not automatically
        # an error and a small one is not automatically proof.
        cut = classify_cut(
            median_offset_ms=evaluation.median_offset_ms,
            p95_offset_ms=evaluation.p95_offset_ms,
            mad_offset_ms=evaluation.mad_offset_ms,
            drift_ms_per_minute=drift,
            change_points=evaluation.change_points,
            structural=structural,
            max_plausible_offset_ms=MAX_PLAUSIBLE_OFFSET_MS,
            max_p95_ms=MAX_P95_MS_FOR_STABLE,
            max_drift_ms_per_minute=MAX_DRIFT_MS_PER_MINUTE,
        )
        evaluation.cut_verdict = cut.value
        evaluation.reasons.append(f"cut classification: {cut.value}")

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

        if cut is CutVerdict.DIFFERENT_CUT:
            evaluation.set_verdict(SyncState.UNVERIFIED, VerificationAvailability.VERIFIED)
            evaluation.rejection_reason = RejectionReason.STRUCTURE_MISMATCH
            evaluation.reasons.append(
                "cue structure does not correspond to the reference; not the same cut"
            )
        elif not residual_ok or not mad_ok:
            # The result is still far from the reference, or the movement was
            # too scattered to be a deliberate re-timing. This is checked first
            # so genuinely untrustworthy output can never be softened into
            # PROBABLE by the drift branch below.
            evaluation.set_verdict(SyncState.UNVERIFIED, VerificationAvailability.UNKNOWN)
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
            evaluation.set_verdict(SyncState.PROBABLE_SYNC, VerificationAvailability.VERIFIED)
            evaluation.sync_confidence = round(min(confidence, 70.0), 1)
            evaluation.verification_confidence = 50.0
            evaluation.reasons.append(
                f"progressive drift {drift:+.1f}ms/min exceeds "
                f"{MAX_DRIFT_MS_PER_MINUTE:.0f}ms/min; alignment is not a stable global shift"
            )
        else:
            evaluation.set_verdict(SyncState.VERIFIED_RESYNCED, VerificationAvailability.VERIFIED)
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
        structural: StructuralSimilarity | None = None,
    ) -> SubtitleEvaluation:
        """Classify a subtitle that was never re-timed.

        ``VERIFIED_SYNCED`` requires positive proof: a measurable offset against
        a real reference, and that offset already being negligible.
        """
        reference_cues = self._as_cues(reference)
        if reference_cues:
            cut = classify_cut(
                median_offset_ms=evaluation.median_offset_ms,
                p95_offset_ms=evaluation.p95_offset_ms,
                mad_offset_ms=evaluation.mad_offset_ms,
                drift_ms_per_minute=evaluation.drift_ms_per_minute,
                change_points=evaluation.change_points,
                structural=structural,
                max_plausible_offset_ms=MAX_PLAUSIBLE_OFFSET_MS,
                max_p95_ms=MAX_P95_MS_FOR_STABLE,
            max_drift_ms_per_minute=MAX_DRIFT_MS_PER_MINUTE,
            )
            evaluation.cut_verdict = cut.value
            evaluation.reasons.append(f"cut classification: {cut.value}")
        if not reference_cues:
            evaluation.set_verdict(SyncState.UNVERIFIED, VerificationAvailability.UNKNOWN)
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
                evaluation.set_verdict(SyncState.REJECTED, VerificationAvailability.VERIFIED)
                evaluation.rejection_reason = RejectionReason.IMPLAUSIBLE_OFFSET
                evaluation.reasons.append(f"unalignable and {sanity['reason']}")
            else:
                evaluation.set_verdict(SyncState.UNVERIFIED, VerificationAvailability.UNKNOWN)
                evaluation.rejection_reason = RejectionReason.INSUFFICIENT_EVIDENCE
                evaluation.reasons.append("insufficient paired cues against the reference")
            return evaluation

        median_s = (evaluation.median_offset_ms or 0.0) / 1000.0
        aligned_threshold = ALIGNED_OFFSET_THRESHOLD_S
        if abs(median_s) < aligned_threshold:
            evaluation.set_verdict(SyncState.VERIFIED_SYNCED, VerificationAvailability.VERIFIED)
            evaluation.sync_confidence = 100.0
            evaluation.verification_confidence = 100.0
            evaluation.reasons.append(
                f"already aligned: median offset {median_s:+.2f}s "
                f"(threshold {aligned_threshold:.2f}s); no re-timing required"
            )
            return evaluation

        if abs(evaluation.median_offset_ms or 0.0) > MAX_PLAUSIBLE_OFFSET_MS:
            # Beyond the +/-20s window this is a different cut, not a sync
            # offset. The gate is deliberately NOT softened by structure: a
            # matching shape cannot prove a 97s displacement is re-timable, and
            # loosening it would trade false positives for false negatives.
            evaluation.set_verdict(SyncState.REJECTED, VerificationAvailability.VERIFIED)
            evaluation.rejection_reason = RejectionReason.IMPLAUSIBLE_OFFSET
            evaluation.reasons.append(
                f"median offset {median_s:+.2f}s is a different cut, not a sync offset"
            )
            return evaluation

        # Measurably offset but never re-timed: a candidate, not a fact.
        evaluation.set_verdict(SyncState.PROBABLE_SYNC, VerificationAvailability.VERIFIED)
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
