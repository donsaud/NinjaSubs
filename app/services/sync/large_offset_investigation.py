"""Large Offset Investigation: reference-first same-episode evidence.

This is the *deep* stage that a candidate enters once the normal first-dialogue
execution window (:data:`FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS`, 20s) has been
missed. The 20s boundary is a routing boundary, not a rejection boundary: inside
it, nothing here runs and the normal verifier keeps its exact existing
semantics. Outside it, the candidate has to earn an alass attempt from evidence
about *whether it is the same episode as the REFERENCE*.

It deliberately does three separate things, each independently testable:

1. :func:`investigate_large_offset` -- reference-first identity and temporal
   structure comparison, producing a :class:`LargeOffsetInvestigation` with
   named, explainable evidence fields. It answers "is this the same episode,
   with a large systematic displacement?" and never "is this synchronized?".
2. :func:`validate_alass_output` -- lightweight validity checking of what alass
   actually produced. It answers "is this a plausible correction?" and is
   deliberately *not* the general verifier's p95/drift/structural battery.
3. :func:`decide_large_offset_serving` -- may these corrected bytes replace the
   original in this response? The result is a **serving state**,
   :class:`LargeOffsetServingState.ALASS_CORRECTED_LARGE_OFFSET`, which is
   emphatically not :class:`~.alignment.SyncState.VERIFIED_*`. Nothing in this
   module can create a verification claim, and the general verifier keeps
   recording its own verdict untouched. The override it grants is deliberately
   narrow and enumerated: it exempts *only* the residual-p95 measurement that
   cross-release cue segmentation makes too strict, and only when the analyzer
   independently measured the correction as one constant shift.

Evidence provenance is explicit: a single trusted reference is acceptable when
the existing reference health/trust system says it is usable, but the
investigation records *that* it is one reference and what its health was, so a
reader can never mistake one reference for independent corroboration. Duplicate
copies of the same subtitle collapse in
:func:`~.reference.detect_duplicate_groups` upstream and never arrive here as
"more references".

All computation is bounded (:data:`LARGE_OFFSET_MAX_SAMPLED_CUES`, fixed bin
counts) and deterministic: no network, no media decode, no ffmpeg, no
unbounded dynamic program. Alass remains the only alignment engine; this module
only decides eligibility to reach it.
"""

from __future__ import annotations

import bisect
import logging
import math
from collections.abc import Sequence
from enum import Enum

from pydantic import BaseModel, Field

from ..subtitle_matcher import FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS, parse_srt_cues
from .alignment import (
    MAX_MAD_MS_FOR_STABLE,
    Cue,
    RejectionReason,
    SubtitleEvaluation,
    _median,
)
from .large_offset import (
    LARGE_OFFSET_ANCHOR_TOLERANCE_MS,
    LARGE_OFFSET_MAX_MS,
    LARGE_OFFSET_MIN_ANCHORS,
    REASON_DISPERSION_TOO_HIGH,
    REASON_DRIFT_TOO_HIGH,
    REASON_IDENTITY_UNSUPPORTED,
    REASON_INSUFFICIENT_ANCHORS,
    REASON_MAX_OFFSET_EXCEEDED,
    REASON_NO_DIALOGUE_SEED,
    REASON_STRUCTURAL_MISMATCH,
    LargeOffsetAssessment,
    _bounded,
    _shifted_profile_similarity,
    assess_large_offset,
    classify_large_offset_candidate,
)
from .reference import MIN_REFERENCE_CUES, analyze_reference_health

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Tunables for the signals this module adds on top of the evidence gate.
# Internal, deterministic, and each one is named in the evidence it produces so
# a failing case can be explained without reading the implementation.
# --------------------------------------------------------------------------- #

# Gap-sequence similarity compares the *distribution* of consecutive cue-start
# gaps, not the gaps one by one. Splitting one cue into two moves mass between
# bins but leaves the overall shape recognisable, which is exactly the property
# a one-to-one gap comparison lacks.
LARGE_OFFSET_GAP_BIN_EDGES_MS: tuple[int, ...] = (
    0,
    400,
    1_200,
    2_500,
    5_000,
    10_000,
    20_000,
    45_000,
    10**9,
)

# A pause at least this long marks a dialogue block boundary. Chosen well below
# a scene change and well above a normal inter-cue breath.
LARGE_OFFSET_SILENCE_GAP_MS = 4_000
# How far apart two silence landmarks may sit and still count as the same one.
# Generous on purpose: segmentation moves a boundary by a cue, not by a scene.
LARGE_OFFSET_SILENCE_TOLERANCE_MS = 30_000
# Below this many landmarks on either side the signal is not measurable, and
# "not measurable" is never silently upgraded to "agrees".
LARGE_OFFSET_MIN_SILENCE_LANDMARKS = 2

# The episode is cut into this many equal bins to check that structural
# agreement is *distributed*, rather than one accidental local region.
LARGE_OFFSET_COVERAGE_BINS = 10

# Floors for the added signals. Each is a named constant so a test can make one
# load-bearing by tightening it, the way the existing gate tests do.
LARGE_OFFSET_MIN_GAP_SIMILARITY = 0.60
LARGE_OFFSET_MIN_LANDMARK_AGREEMENT = 0.50
LARGE_OFFSET_MIN_TEMPORAL_COVERAGE = 0.55
LARGE_OFFSET_MIN_OFFSET_CONSISTENCY = 0.50

# Post-alass validation bounds. These answer "is this a plausible correction?",
# not "is this verified" -- that distinction is the whole point of the module.
#: The corrected output must retain at least this fraction of the input cues.
LARGE_OFFSET_MIN_OUTPUT_RETENTION = 0.60
#: ... and no more than this, or alass invented content.
LARGE_OFFSET_MAX_OUTPUT_RETENTION = 1.60
#: The corrected output's span must overlap the reference span by at least this
#: fraction of the reference span.
LARGE_OFFSET_MIN_OUTPUT_COVERAGE = 0.50
#: After correction the output must sit inside the normal window of the
#: reference. A correction that is still tens of seconds out corrected nothing.
LARGE_OFFSET_MAX_OUTPUT_RESIDUAL_MS = FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS

# --------------------------------------------------------------------------- #
# Serving-override bounds: when may the corrected bytes replace the original
# even though the general verifier declined them?
#
# The general verifier refuses these cases on the *residual p95* alone -- cue
# segmentation differs between releases, so per-cue residuals land above
# MAX_P95_MS_FOR_STABLE however correct the alignment is. Measured on real
# alass runs (Docker, alass-cli 2.0.0, --split-penalty 7.0):
#
#   case              movement MAD   p95      struct   outcome
#   REAL EVOLV              0 ms     3310     0.855    must be served
#   REAL ASAP               0 ms     4750     0.729    must be served
#   BAD wrong_episode   11039 ms     3691     0.626    must be refused
#   BAD drifting         3601 ms     3336     0.856    must be refused
#   BAD different_cut       0 ms     3210     0.762    KNOWN FALSE ACCEPT
#
# Movement MAD is what separates them: a genuine constant displacement is
# applied as one constant shift (MAD 0), whereas aligning an unrelated or
# drifting timeline needs a varying correction. p95 deliberately is *not* a
# condition here -- it is the very measurement that is too strict for
# cross-release segmentation, and requiring it would refuse both real cases.
#
# ``different_cut`` is documented as a known false accept: it too is a genuine
# constant relationship, so MAD cannot see it, and no measured signal separates
# it from the required-positive ASAP (struct 0.762 vs 0.729; content
# correspondence 0.245 vs 0.176 -- it scores *higher* on both). Closing it
# would require refusing ASAP, which the feature exists to serve.
# --------------------------------------------------------------------------- #

#: The correction must be a constant shift. Reuses the analyzer's own stability
#: bound so the two cannot drift apart.
LARGE_OFFSET_MAX_MOVEMENT_MAD_MS = MAX_MAD_MS_FOR_STABLE
#: Independent backstop on structural agreement. Set below the required
#: positives (ASAP 0.729, EVOLV 0.855) and above the structurally broken
#: negative (wrong_episode 0.626), so it never carries the decision alone --
#: movement MAD already refuses that case decisively -- but a structurally
#: inconsistent result can never be served on MAD alone.
LARGE_OFFSET_MIN_STRUCTURAL_SIMILARITY = 0.65


# --------------------------------------------------------------------------- #
# The models. Everything the decision rests on is a named field.
# --------------------------------------------------------------------------- #


class LargeOffsetSameEpisode(str, Enum):
    """Whether the candidate belongs to the same episode as the REFERENCE.

    Established *before* alass runs. ``SAME_EPISODE`` is never granted because
    alass happened to produce an output -- that would make the check circular.
    """

    # Offset is within the normal window; this stage never ran.
    NOT_APPLICABLE = "not_applicable"
    # Reference or timing data too weak or sparse to decide.
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    # Available evidence contradicts same-episode identity.
    MISMATCH = "mismatch"
    # Sufficient temporal/structural agreement to reach Large Offset alignment.
    SAME_EPISODE = "same_episode"


class LargeOffsetDecision(str, Enum):
    """The precise outcome of the investigation, most decisive first."""

    # Offset inside the normal window: normal path owns the candidate.
    NORMAL_PATH = "normal_path"
    REFERENCE_UNUSABLE = "reference_unusable"
    OFFSET_ABOVE_CEILING = "offset_above_ceiling"
    IDENTITY_CONTRADICTION = "identity_contradiction"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    STRUCTURE_MISMATCH = "structure_mismatch"
    ELIGIBLE_FOR_ALASS = "eligible_for_alass"


class LargeOffsetServingState(str, Enum):
    """What may be delivered for a large-offset candidate.

    Deliberately separate from :class:`~.alignment.SyncState`. A
    ``VERIFIED_*`` state claims the general verifier measured and accepted the
    alignment; this claims only that the dedicated investigation passed and
    alass produced a valid correction. Conflating them would be exactly the
    quiet promotion this design exists to prevent.
    """

    ALASS_CORRECTED_LARGE_OFFSET = "alass_corrected_large_offset"
    ORIGINAL = "original"


class LargeOffsetEvidence(BaseModel):
    """Named temporal evidence. No opaque aggregate score.

    Each field is reported even when it did not decide the outcome, so a log or
    a test can show *why* a case passed or failed rather than only that it did.
    """

    # A. Offset consistency: agreement of the regional offset estimates.
    # 1 - dispersion relative to the anchor tolerance, clamped to [0, 1].
    offset_consistency_score: float | None = None
    estimated_offset_ms: float | None = None
    offset_dispersion_ms: float | None = None
    drift_ms_per_minute: float | None = None
    anchor_count: int = 0
    regions_sampled: int = 0

    # B. Gap-sequence similarity: cosine similarity of the binned distribution
    # of consecutive cue-start gaps, after the global offset is removed.
    gap_distribution_similarity: float | None = None

    # C. Silence/transition landmarks: agreement of long-pause locations once
    # the candidate is mapped onto the reference timeline.
    silence_landmark_agreement: float | None = None
    silence_landmark_count_target: int = 0
    silence_landmark_count_reference: int = 0

    # D. Temporal coverage: fraction of reference-populated episode bins that
    # the candidate also populates. Distributes, rather than accumulates,
    # the agreement.
    temporal_coverage: float | None = None

    # E. Cue density profile: cosine similarity of onset density histograms
    # after applying the offset. Language-agnostic: reads where events fall.
    density_profile_similarity: float | None = None


class LargeOffsetInvestigation(BaseModel):
    """Full, explainable result of the Large Offset Investigation.

    Deterministic and testable without HTTP, without alass, and without the
    analyzer. Carries the provenance of every input so the decision can be
    re-read later: *why* it entered, what reference it used, what that
    reference was worth, and which signal disagreed.
    """

    # --- entry ------------------------------------------------------------ #
    entered: bool = False
    threshold_ms: int = FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS
    seed_offset_ms: int | None = None

    # --- reference provenance --------------------------------------------- #
    reference_available: bool = False
    reference_cue_count: int = 0
    reference_healthy: bool = False
    reference_trust: str | None = None
    reference_failure: str | None = None
    reference_reasons: list[str] = Field(default_factory=list)
    #: How many *independent* references backed this decision. 1 means one
    #: reference, never corroboration; recorded so the distinction survives.
    independent_reference_count: int = 0

    # --- existing compatibility / identity evidence ----------------------- #
    identity_supported: bool = False
    season_match: bool | None = None
    episode_match: bool | None = None
    metadata_contradiction: bool = False

    # --- evidence and verdict --------------------------------------------- #
    evidence: LargeOffsetEvidence = Field(default_factory=LargeOffsetEvidence)
    assessment: LargeOffsetAssessment | None = None
    same_episode: LargeOffsetSameEpisode = LargeOffsetSameEpisode.NOT_APPLICABLE
    decision: LargeOffsetDecision = LargeOffsetDecision.NORMAL_PATH
    eligible_for_alass: bool = False
    reason_codes: list[str] = Field(default_factory=list)

    # --- serving outcome, filled by decide_large_offset_serving ----------- #
    #: Defaults to ORIGINAL so an investigation nobody made a serving decision
    #: about can never read as though bytes were replaced.
    serving_state: LargeOffsetServingState = LargeOffsetServingState.ORIGINAL
    serving_reason_codes: list[str] = Field(default_factory=list)

    def summary(self) -> str:
        """One line, credential-free, safe for logs."""
        ev = self.evidence

        def fmt(value: float | None, digits: int = 3) -> str:
            return "n/a" if value is None else f"{value:.{digits}f}"

        return (
            f"entered={self.entered} seed={self.seed_offset_ms} "
            f"same_episode={self.same_episode.value} decision={self.decision.value} "
            f"eligible={self.eligible_for_alass} "
            f"refs={self.independent_reference_count} "
            f"cues={self.reference_cue_count} healthy={self.reference_healthy} "
            f"anchors={ev.anchor_count}/{ev.regions_sampled} "
            f"consistency={fmt(ev.offset_consistency_score)} "
            f"gaps={fmt(ev.gap_distribution_similarity)} "
            f"landmarks={fmt(ev.silence_landmark_agreement)} "
            f"coverage={fmt(ev.temporal_coverage)} "
            f"density={fmt(ev.density_profile_similarity)} "
            f"reasons={','.join(self.reason_codes) or 'none'}"
        )


class AlassOutputValidation(BaseModel):
    """Lightweight validity verdict on what alass actually produced.

    Deliberately *not* the general verifier. It answers "did alass produce a
    valid, plausibly aligned subtitle?" -- non-empty, parseable, monotonic,
    plausible in size, still covering the reference timeline, and now sitting
    near the reference. It never answers "is this verified".
    """

    ok: bool = False
    cue_count: int = 0
    input_cue_count: int = 0
    retention_ratio: float | None = None
    coverage_ratio: float | None = None
    residual_offset_ms: float | None = None
    monotonic: bool = False
    timestamps_valid: bool = False
    reason_codes: list[str] = Field(default_factory=list)

    def summary(self) -> str:
        return (
            f"ok={self.ok} cues={self.cue_count}/{self.input_cue_count} "
            f"retention={self.retention_ratio} coverage={self.coverage_ratio} "
            f"residual_ms={self.residual_offset_ms} "
            f"reasons={','.join(self.reason_codes) or 'none'}"
        )


# Stable reason codes for the investigation itself.
INV_REASON_ENTERED = "large_offset_investigation_entered"
INV_REASON_NO_SEED = "no_measured_opening_offset"
INV_REASON_REFERENCE_MISSING = "reference_unavailable"
INV_REASON_REFERENCE_UNHEALTHY = "reference_health_failed"
INV_REASON_LOW_TRUST = "reference_trust_below_usable"
INV_REASON_METADATA_CONTRADICTION = "episode_metadata_contradiction"
INV_REASON_TOO_SPARSE = "insufficient_cues_to_compare"
INV_REASON_CONSISTENCY = "offset_consistency_below_floor"
INV_REASON_GAPS = "gap_distribution_below_floor"
INV_REASON_LANDMARKS = "silence_landmarks_disagree"
INV_REASON_LANDMARKS_UNMEASURABLE = "silence_landmarks_unmeasurable"
INV_REASON_COVERAGE = "temporal_coverage_below_floor"
INV_REASON_DENSITY = "density_profile_below_floor"
INV_REASON_ELIGIBLE = "same_episode_established"

# Post-alass reason codes.
ALASS_REASON_EMPTY = "alass_output_empty"
ALASS_REASON_UNPARSEABLE = "alass_output_unparseable"
ALASS_REASON_INVALID_TIMESTAMPS = "alass_output_invalid_timestamps"
ALASS_REASON_NOT_MONOTONIC = "alass_output_not_monotonic"
ALASS_REASON_RETENTION_LOW = "alass_output_retention_too_low"
ALASS_REASON_RETENTION_HIGH = "alass_output_retention_too_high"
ALASS_REASON_COVERAGE_LOW = "alass_output_coverage_too_low"
ALASS_REASON_RESIDUAL_HIGH = "alass_output_residual_too_high"

# Serving reason codes, echoed by the caller for observability.
SERVE_REASON_CORRECTED = "large_offset_investigation_passed"
SERVE_REASON_NOT_ELIGIBLE = "large_offset_investigation_not_eligible"
SERVE_REASON_ALASS_INVALID = "alass_output_not_valid"
# Without the analyzer's own evaluation there is no way to tell whether its only
# objection was the segmentation-driven p95, so serving fails closed.
SERVE_REASON_NO_EVALUATION = "analyzer_evaluation_missing"
# The correction was not one constant shift, so "large systematic offset" is
# not what was corrected.
SERVE_REASON_MOVEMENT_NOT_CONSTANT = "movement_not_a_constant_shift"
# The analyzer refused on content/structure/cue evidence rather than on
# confidence, which the p95 exemption has no business overriding.
SERVE_REASON_STRUCTURAL_REJECTED = "structural_rejection_not_overridden"
SERVE_REASON_STRUCTURE_TOO_LOW = "structural_similarity_below_floor"


# --------------------------------------------------------------------------- #
# Signal helpers. Pure, bounded, deterministic.
# --------------------------------------------------------------------------- #


def _as_cues(value: str | Sequence[Cue]) -> list[Cue]:
    if isinstance(value, str):
        parsed = parse_srt_cues(value)
    else:
        parsed = [tuple(cue) for cue in value]  # type: ignore[misc]
    return sorted(parsed, key=lambda cue: cue[0])


def _cosine(left: Sequence[float], right: Sequence[float]) -> float | None:
    if not left or not right or len(left) != len(right):
        return None
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left <= 0.0 or norm_right <= 0.0:
        return None
    return dot / (norm_left * norm_right)


def _gap_histogram(cues: Sequence[Cue]) -> list[float]:
    """Histogram of consecutive cue-start gaps, one count per bin.

    Binned rather than compared element-wise: a split cue changes two adjacent
    gaps and nothing else, so an element-wise comparison would collapse on any
    segmentation difference while the *distribution* stays recognisable.
    """
    counts = [0.0] * (len(LARGE_OFFSET_GAP_BIN_EDGES_MS) - 1)
    for index in range(1, len(cues)):
        gap = cues[index][0] - cues[index - 1][0]
        if gap < 0:
            continue
        for edge_index in range(len(counts)):
            if LARGE_OFFSET_GAP_BIN_EDGES_MS[edge_index] <= gap < (
                LARGE_OFFSET_GAP_BIN_EDGES_MS[edge_index + 1]
            ):
                counts[edge_index] += 1.0
                break
    return counts


def gap_distribution_similarity(
    target: str | Sequence[Cue], reference: str | Sequence[Cue]
) -> float | None:
    """Cosine similarity of the two gap distributions.

    Offset-free by construction: a gap between consecutive starts is unchanged
    by any global shift, so the signal cannot be manufactured by the offset it
    is helping to confirm.
    """
    return _cosine(_gap_histogram(_as_cues(target)), _gap_histogram(_as_cues(reference)))


def silence_landmarks(cues: Sequence[Cue]) -> list[int]:
    """Start positions of the cues that follow a meaningful pause.

    Uses cue-end to next-cue-start, so a long single cue does not masquerade as
    a scene boundary and a cluster of short cues does not hide one.
    """
    landmarks: list[int] = []
    for index in range(1, len(cues)):
        pause = cues[index][0] - cues[index - 1][1]
        if pause >= LARGE_OFFSET_SILENCE_GAP_MS:
            landmarks.append(cues[index][0])
    return landmarks


def silence_landmark_agreement(
    target: str | Sequence[Cue],
    reference: str | Sequence[Cue],
    offset_ms: float,
) -> tuple[float | None, int, int]:
    """Agreement of long-pause locations once the reference is shifted onto the
    candidate timeline.

    Returns ``(score, target_count, reference_count)``. ``score`` is ``None``
    when either side has fewer than
    :data:`LARGE_OFFSET_MIN_SILENCE_LANDMARKS` landmarks -- "too few to judge"
    is never folded into "agrees", because a subtitle with no pauses at all
    would otherwise pass trivially.

    The score is the minimum of the two directional recall rates, so a candidate
    that carries only half the reference's scene boundaries cannot score well by
    matching the few it kept.
    """
    target_cues = _as_cues(target)
    reference_cues = _as_cues(reference)
    target_lm = silence_landmarks(target_cues)
    # Reference landmarks expressed on the candidate's timeline.
    reference_lm = [int(round(landmark + offset_ms)) for landmark in silence_landmarks(reference_cues)]
    if (
        len(target_lm) < LARGE_OFFSET_MIN_SILENCE_LANDMARKS
        or len(reference_lm) < LARGE_OFFSET_MIN_SILENCE_LANDMARKS
    ):
        return None, len(target_lm), len(reference_lm)

    def recall(needles: list[int], haystack: list[int]) -> float:
        if not needles:
            return 0.0
        found = 0
        for needle in needles:
            if any(
                abs(needle - other) <= LARGE_OFFSET_SILENCE_TOLERANCE_MS
                for other in haystack
            ):
                found += 1
        return found / len(needles)

    return (
        min(recall(target_lm, reference_lm), recall(reference_lm, target_lm)),
        len(target_lm),
        len(reference_lm),
    )


def temporal_coverage(
    target: str | Sequence[Cue],
    reference: str | Sequence[Cue],
    offset_ms: float,
) -> float | None:
    """Fraction of reference-populated episode bins the candidate also fills.

    This is what stops a single well-matching region from carrying the decision:
    the reference's own timeline is cut into
    :data:`LARGE_OFFSET_COVERAGE_BINS` equal bins, and agreement has to be
    present in most of them. Bins where *neither* side has dialogue are excluded
    from the denominator, so an episode's silent cold open cannot be held against
    a candidate that matches everywhere else.
    """
    target_cues = _as_cues(target)
    reference_cues = _as_cues(reference)
    if not target_cues or not reference_cues:
        return None

    shifted_reference = [(start + offset_ms, end + offset_ms, text) for start, end, text in reference_cues]
    span_start = min(target_cues[0][0], shifted_reference[0][0])
    span_end = max(
        target_cues[-1][0], shifted_reference[-1][0], span_start + 1
    )
    width = (span_end - span_start) / LARGE_OFFSET_COVERAGE_BINS

    target_bins = set()
    for start, _, _ in target_cues:
        index = int((start - span_start) // width)
        if 0 <= index < LARGE_OFFSET_COVERAGE_BINS:
            target_bins.add(index)
    reference_bins = set()
    for ref_start, _, _ in shifted_reference:
        index = int((ref_start - span_start) // width)
        if 0 <= index < LARGE_OFFSET_COVERAGE_BINS:
            reference_bins.add(index)

    relevant = reference_bins | target_bins
    if not relevant:
        return None
    agreed = target_bins & reference_bins
    return len(agreed) / len(relevant)


def offset_consistency_score(dispersion_ms: float | None) -> float | None:
    """Map the regional dispersion onto ``[0, 1]``; higher is more consistent."""
    if dispersion_ms is None:
        return None
    ratio = max(0.0, min(1.0, dispersion_ms / LARGE_OFFSET_ANCHOR_TOLERANCE_MS))
    return round(1.0 - ratio, 4)


def _reference_is_usable(
    reference_text: str | Sequence[Cue] | None,
) -> tuple[bool, int, list[str]]:
    """Apply the existing reference *structural* health rules, unmodified.

    Reuses :func:`reference.analyze_reference_health` so the investigation and
    the reference selector cannot disagree about what "structurally sound"
    means. A reference that is empty, unparseable, too sparse, structurally
    broken, or carries no usable dialogue fails here rather than being quietly
    assumed good.

    The runtime-coverage floor is deliberately not applied (see
    ``require_dialogue_coverage``). It is a selection preference measured
    against a 2s mean cue, and it rejects every reference this investigation is
    required to accept -- including the production reference, at 39%. Identity
    is judged instead by :func:`temporal_coverage`, which asks the two-sided
    question the floor approximates one-sidedly: once shifted onto the
    candidate, do the reference's populated regions line up with the
    candidate's? That signal is required to clear its own floor below, so
    dropping the one-sided proxy removes no evidence.

    The coverage figure is still returned as a note so it stays visible in the
    investigation record rather than disappearing.
    """
    if reference_text is None or (isinstance(reference_text, str) and not reference_text.strip()):
        return False, 0, ["reference produced no parseable cues"]
    if isinstance(reference_text, str):
        health = analyze_reference_health(
            reference_text, require_dialogue_coverage=False
        )
        notes = list(health.reasons)
        if health.dialogue_coverage is not None:
            notes.append(
                "dialogue covers "
                f"{health.dialogue_coverage:.0%} of the runtime "
                "(coverage floor not applied to this path)"
            )
        if not health.healthy:
            return False, health.cue_count, notes
        return True, health.cue_count, notes
    # Already-parsed cues: health analysis expects text, so fall back to a
    # structural check on the cue population itself.
    cues = _as_cues(reference_text)
    if len(cues) < MIN_REFERENCE_CUES:
        return False, len(cues), [f"only {len(cues)} cues (need {MIN_REFERENCE_CUES})"]
    return True, len(cues), []


def _signal_floor_verdicts(evidence: LargeOffsetEvidence, offset_ms: float) -> tuple[list[str], list[str]]:
    """Evaluate the added signals.

    Returns ``(mismatch_reasons, unmeasurable_reasons)``. A signal that could
    not be measured contributes to ``unmeasurable`` and is never counted as
    agreeing; only a measured signal below its floor counts as a mismatch.
    """
    mismatch: list[str] = []
    unmeasurable: list[str] = []

    if evidence.offset_consistency_score is None:
        unmeasurable.append(INV_REASON_CONSISTENCY)
    elif evidence.offset_consistency_score < LARGE_OFFSET_MIN_OFFSET_CONSISTENCY:
        mismatch.append(INV_REASON_CONSISTENCY)

    if evidence.gap_distribution_similarity is None:
        unmeasurable.append(INV_REASON_GAPS)
    elif evidence.gap_distribution_similarity < LARGE_OFFSET_MIN_GAP_SIMILARITY:
        mismatch.append(INV_REASON_GAPS)

    if evidence.silence_landmark_agreement is None:
        unmeasurable.append(INV_REASON_LANDMARKS_UNMEASURABLE)
    elif evidence.silence_landmark_agreement < LARGE_OFFSET_MIN_LANDMARK_AGREEMENT:
        mismatch.append(INV_REASON_LANDMARKS)

    if evidence.temporal_coverage is None:
        unmeasurable.append(INV_REASON_COVERAGE)
    elif evidence.temporal_coverage < LARGE_OFFSET_MIN_TEMPORAL_COVERAGE:
        mismatch.append(INV_REASON_COVERAGE)

    if evidence.density_profile_similarity is None:
        unmeasurable.append(INV_REASON_DENSITY)
    elif evidence.density_profile_similarity < 0.55:
        mismatch.append(INV_REASON_DENSITY)

    return mismatch, unmeasurable


# --------------------------------------------------------------------------- #
# Stage 1: the investigation itself
# --------------------------------------------------------------------------- #


def investigate_large_offset(
    target: str | Sequence[Cue],
    reference: str | Sequence[Cue] | None,
    *,
    threshold_ms: int = FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
    identity_supported: bool = False,
    reference_trust: str | None = None,
    reference_failure: str | None = None,
    reference_reasons: Sequence[str] | None = None,
    independent_reference_count: int = 1,
    season_match: bool | None = None,
    episode_match: bool | None = None,
    max_offset_ms: int | None = None,
) -> LargeOffsetInvestigation:
    """Investigate whether a large-offset candidate is the same episode.

    Reference-first by construction: the question is never "can this be forced
    to fit" but "is this structurally consistent with the episode the REFERENCE
    describes". The reference is evidence, never a guarantee -- if it is
    unavailable, unhealthy, or structurally unusable the investigation says so
    instead of proceeding.

    The candidate never selects the reference here; the caller passes the one
    the existing selector already bound to this target, so there is no
    circular evidence.
    """
    seed = classify_large_offset_candidate(
        target,
        reference if reference is not None else target,
        threshold_ms=threshold_ms,
    )
    investigation = LargeOffsetInvestigation(
        threshold_ms=threshold_ms,
        seed_offset_ms=seed,
        reference_available=reference is not None
        and bool(reference if isinstance(reference, str) else reference),
        identity_supported=identity_supported,
        reference_trust=reference_trust,
        reference_failure=reference_failure,
        reference_reasons=list(reference_reasons or []),
        independent_reference_count=independent_reference_count,
        season_match=season_match,
        episode_match=episode_match,
        metadata_contradiction=season_match is False or episode_match is False,
    )

    if seed is None:
        # Inside the normal window (or no measurable opening): this stage does
        # not apply and the normal path keeps full authority.
        investigation.decision = LargeOffsetDecision.NORMAL_PATH
        investigation.same_episode = LargeOffsetSameEpisode.NOT_APPLICABLE
        investigation.reason_codes.append(INV_REASON_NO_SEED)
        return investigation

    investigation.entered = True
    investigation.reason_codes.append(INV_REASON_ENTERED)
    logger.info(
        "large_offset.investigation_started seed_ms=%d threshold_ms=%d",
        seed,
        threshold_ms,
    )

    if not investigation.reference_available:
        investigation.decision = LargeOffsetDecision.REFERENCE_UNUSABLE
        investigation.same_episode = LargeOffsetSameEpisode.INSUFFICIENT_EVIDENCE
        investigation.reason_codes.append(INV_REASON_REFERENCE_MISSING)
        return investigation

    usable, reference_cue_count, health_notes = _reference_is_usable(reference)
    investigation.reference_cue_count = reference_cue_count
    investigation.reference_healthy = usable
    investigation.reference_reasons.extend(health_notes)
    if not usable:
        investigation.reason_codes.append(INV_REASON_REFERENCE_UNHEALTHY)
        investigation.decision = LargeOffsetDecision.REFERENCE_UNUSABLE
        investigation.same_episode = LargeOffsetSameEpisode.INSUFFICIENT_EVIDENCE
        return investigation

    if investigation.metadata_contradiction:
        # Timing structure must never override an explicit season/episode
        # contradiction established by the existing compatibility evidence.
        investigation.decision = LargeOffsetDecision.IDENTITY_CONTRADICTION
        investigation.same_episode = LargeOffsetSameEpisode.MISMATCH
        investigation.reason_codes.append(INV_REASON_METADATA_CONTRADICTION)
        return investigation

    if reference_trust in {"rejected", "unknown"} and reference_failure is None:
        investigation.decision = LargeOffsetDecision.INSUFFICIENT_EVIDENCE
        investigation.same_episode = LargeOffsetSameEpisode.INSUFFICIENT_EVIDENCE
        investigation.reason_codes.append(INV_REASON_LOW_TRUST)
        return investigation

    target_cues = _bounded(_as_cues(target))
    reference_cues = _bounded(_as_cues(reference))  # type: ignore[arg-type]
    if len(target_cues) < LARGE_OFFSET_MIN_ANCHORS or len(reference_cues) < (
        LARGE_OFFSET_MIN_ANCHORS
    ):
        investigation.decision = LargeOffsetDecision.INSUFFICIENT_EVIDENCE
        investigation.same_episode = LargeOffsetSameEpisode.INSUFFICIENT_EVIDENCE
        investigation.reason_codes.append(INV_REASON_TOO_SPARSE)
        return investigation

    # --- the existing evidence gate supplies anchors, dispersion, drift ---- #
    assessment = assess_large_offset(
        target,
        reference,  # type: ignore[arg-type]
        seed_offset_ms=seed,
        identity_supported=identity_supported,
        max_offset_ms=(
            LARGE_OFFSET_MAX_MS if max_offset_ms is None else max_offset_ms
        ),
    )
    investigation.assessment = assessment

    evidence = investigation.evidence
    evidence.anchor_count = assessment.anchor_count
    evidence.regions_sampled = assessment.regions_sampled
    evidence.estimated_offset_ms = assessment.estimated_offset_ms
    evidence.offset_dispersion_ms = assessment.offset_dispersion_ms
    evidence.drift_ms_per_minute = assessment.drift_ms_per_minute
    evidence.offset_consistency_score = offset_consistency_score(
        assessment.offset_dispersion_ms
    )
    evidence.density_profile_similarity = assessment.structural_similarity

    offset_ms = assessment.estimated_offset_ms
    if offset_ms is None:
        # The gate could not produce a usable estimate, so every offset-relative
        # signal would be measuring against a guess.
        investigation.decision = LargeOffsetDecision.INSUFFICIENT_EVIDENCE
        investigation.same_episode = _gate_verdict(assessment)
        investigation.reason_codes.extend(assessment.reason_codes)
        return investigation

    evidence.gap_distribution_similarity = gap_distribution_similarity(
        target_cues, reference_cues
    )
    (
        evidence.silence_landmark_agreement,
        evidence.silence_landmark_count_target,
        evidence.silence_landmark_count_reference,
    ) = silence_landmark_agreement(target_cues, reference_cues, offset_ms)
    evidence.temporal_coverage = temporal_coverage(target_cues, reference_cues, offset_ms)
    if evidence.density_profile_similarity is None:
        evidence.density_profile_similarity = _shifted_profile_similarity(
            target_cues, reference_cues, offset_ms
        )

    # --- decision ---------------------------------------------------------- #
    if REASON_MAX_OFFSET_EXCEEDED in assessment.reason_codes:
        investigation.decision = LargeOffsetDecision.OFFSET_ABOVE_CEILING
        investigation.same_episode = LargeOffsetSameEpisode.INSUFFICIENT_EVIDENCE
        investigation.reason_codes.append(REASON_MAX_OFFSET_EXCEEDED)
        return investigation

    gate_verdict = _gate_verdict(assessment)
    mismatch, unmeasurable = _signal_floor_verdicts(evidence, offset_ms)

    if gate_verdict is LargeOffsetSameEpisode.MISMATCH or mismatch:
        investigation.decision = LargeOffsetDecision.STRUCTURE_MISMATCH
        investigation.same_episode = LargeOffsetSameEpisode.MISMATCH
        investigation.reason_codes.extend(assessment.reason_codes)
        investigation.reason_codes.extend(mismatch)
        return investigation

    if gate_verdict is LargeOffsetSameEpisode.INSUFFICIENT_EVIDENCE:
        investigation.decision = LargeOffsetDecision.INSUFFICIENT_EVIDENCE
        investigation.same_episode = LargeOffsetSameEpisode.INSUFFICIENT_EVIDENCE
        investigation.reason_codes.extend(assessment.reason_codes)
        investigation.reason_codes.extend(unmeasurable)
        return investigation

    # Every measurable signal cleared its floor, but the offset-relative ones
    # only mean something if *some* of them could be measured at all.
    required_measurable = (
        evidence.gap_distribution_similarity is not None
        or evidence.temporal_coverage is not None
        or evidence.density_profile_similarity is not None
    )
    if not required_measurable:
        investigation.decision = LargeOffsetDecision.INSUFFICIENT_EVIDENCE
        investigation.same_episode = LargeOffsetSameEpisode.INSUFFICIENT_EVIDENCE
        investigation.reason_codes.extend(unmeasurable)
        return investigation

    investigation.decision = LargeOffsetDecision.ELIGIBLE_FOR_ALASS
    investigation.same_episode = LargeOffsetSameEpisode.SAME_EPISODE
    investigation.eligible_for_alass = True
    investigation.reason_codes.extend(assessment.reason_codes)
    investigation.reason_codes.append(INV_REASON_ELIGIBLE)
    logger.info(
        "large_offset.episode_identity_passed offset_ms=%d anchors=%d/%d "
        "coverage=%s gaps=%s landmarks=%s density=%s",
        offset_ms,
        evidence.anchor_count,
        evidence.regions_sampled,
        evidence.temporal_coverage,
        evidence.gap_distribution_similarity,
        evidence.silence_landmark_agreement,
        evidence.density_profile_similarity,
    )
    return investigation


def _gate_verdict(assessment: LargeOffsetAssessment) -> LargeOffsetSameEpisode:
    """Map the existing evidence gate's reason codes onto the identity verdict.

    Refusals that mean "the regions disagree" are a mismatch: dispersion, drift
    and structural similarity are positive claims about the timeline that the
    evidence contradicts. Refusals that mean "too little to judge" are
    insufficient evidence: a missing seed, too few anchors, an unsupported
    identity precondition, or a ceiling we refuse to search past.
    """
    if assessment.accepted:
        return LargeOffsetSameEpisode.SAME_EPISODE
    if any(
        reason in assessment.reason_codes
        for reason in (REASON_DISPERSION_TOO_HIGH, REASON_DRIFT_TOO_HIGH, REASON_STRUCTURAL_MISMATCH)
    ):
        return LargeOffsetSameEpisode.MISMATCH
    if any(
        reason in assessment.reason_codes
        for reason in (
            REASON_NO_DIALOGUE_SEED,
            REASON_IDENTITY_UNSUPPORTED,
            REASON_INSUFFICIENT_ANCHORS,
            REASON_MAX_OFFSET_EXCEEDED,
        )
    ):
        return LargeOffsetSameEpisode.INSUFFICIENT_EVIDENCE
    return LargeOffsetSameEpisode.INSUFFICIENT_EVIDENCE


# --------------------------------------------------------------------------- #
# Stage 2: post-alass output validation
# --------------------------------------------------------------------------- #


def validate_alass_output(
    original_target: str | Sequence[Cue],
    alass_output: str | Sequence[Cue] | None,
    reference: str | Sequence[Cue] | None,
) -> AlassOutputValidation:
    """Decide whether alass produced a valid, plausibly aligned subtitle.

    Checks, in order: non-empty, parses, timestamps valid, monotonic, plausible
    cue population, meaningful coverage of the reference timeline, and a
    residual small enough that the correction actually happened.

    This is deliberately *not* the general verifier. The known-good Dexter
    S08E04 corrections fail that verifier's p95 purely because of segmentation
    differences; requiring p95 here would reproduce the bug this feature exists
    to fix. Conversely nothing here can confer verification -- see
    :func:`decide_large_offset_serving`.
    """
    validation = AlassOutputValidation()
    input_cues = _as_cues(original_target)
    validation.input_cue_count = len(input_cues)

    if alass_output is None or (isinstance(alass_output, str) and not alass_output.strip()):
        validation.reason_codes.append(ALASS_REASON_EMPTY)
        return validation

    output_cues = _as_cues(alass_output)
    validation.cue_count = len(output_cues)
    if not output_cues:
        validation.reason_codes.append(ALASS_REASON_UNPARSEABLE)
        return validation

    timestamps_valid = all(
        start >= 0 and end >= start and end - start <= 10 * 60 * 1000
        for start, end, _ in output_cues
    )
    validation.timestamps_valid = timestamps_valid
    if not timestamps_valid:
        validation.reason_codes.append(ALASS_REASON_INVALID_TIMESTAMPS)

    monotonic = all(
        output_cues[index][0] >= output_cues[index - 1][0]
        for index in range(1, len(output_cues))
    )
    validation.monotonic = monotonic
    if not monotonic:
        validation.reason_codes.append(ALASS_REASON_NOT_MONOTONIC)

    if input_cues:
        retention = len(output_cues) / len(input_cues)
        validation.retention_ratio = round(retention, 4)
        if retention < LARGE_OFFSET_MIN_OUTPUT_RETENTION:
            validation.reason_codes.append(ALASS_REASON_RETENTION_LOW)
        elif retention > LARGE_OFFSET_MAX_OUTPUT_RETENTION:
            validation.reason_codes.append(ALASS_REASON_RETENTION_HIGH)

    if reference:
        reference_cues = _as_cues(reference)
        if reference_cues:
            ref_start = reference_cues[0][0]
            ref_end = reference_cues[-1][0]
            ref_span = max(1, ref_end - ref_start)
            out_start = output_cues[0][0]
            out_end = output_cues[-1][0]
            overlap_start = max(ref_start, out_start)
            overlap_end = min(ref_end, out_end)
            overlap = max(0, overlap_end - overlap_start)
            coverage = overlap / ref_span
            validation.coverage_ratio = round(coverage, 4)
            if coverage < LARGE_OFFSET_MIN_OUTPUT_COVERAGE:
                validation.reason_codes.append(ALASS_REASON_COVERAGE_LOW)

            residual = median_offset_between(output_cues, reference_cues)
            validation.residual_offset_ms = residual
            if residual is None or abs(residual) > LARGE_OFFSET_MAX_OUTPUT_RESIDUAL_MS:
                validation.reason_codes.append(ALASS_REASON_RESIDUAL_HIGH)

    validation.ok = not validation.reason_codes
    return validation


def median_offset_between(
    left: Sequence[Cue], right: Sequence[Cue]
) -> float | None:
    """Median nearest-counterpart offset of ``left`` on ``right``.

    Reuses the project's existing nearest-cue convention rather than inventing
    a pairing. ``None`` when either side is empty or nothing lands within the
    generous pairing radius, which is reported as "not measurable", never as
    "aligned".
    """
    if not left or not right:
        return None

    starts = [cue[0] for cue in right]
    radius = LARGE_OFFSET_MAX_OUTPUT_RESIDUAL_MS * 3
    offsets: list[float] = []
    for start, _, _ in left:
        index = bisect.bisect_left(starts, start)
        best: int | None = None
        for candidate in (index - 1, index, index + 1):
            if 0 <= candidate < len(starts):
                if best is None or abs(starts[candidate] - start) < abs(starts[best] - start):
                    best = candidate
        if best is None or abs(starts[best] - start) > radius:
            continue
        offsets.append(float(start - starts[best]))
    if not offsets:
        return None
    return round(_median(offsets), 1)


# --------------------------------------------------------------------------- #
# Stage 3: the serving decision
# --------------------------------------------------------------------------- #


def decide_large_offset_serving(
    investigation: LargeOffsetInvestigation,
    validation: AlassOutputValidation,
    evaluation: SubtitleEvaluation | None = None,
) -> LargeOffsetServingState:
    """May the corrected bytes replace the original in this response?

    Four conditions, all of which must hold:

    1. the investigation established same-episode identity and granted alass
       eligibility;
    2. alass produced a structurally valid output (:func:`validate_alass_output`);
    3. the analyzer measured the correction as a **constant** shift -- movement
       MAD within :data:`LARGE_OFFSET_MAX_MOVEMENT_MAD_MS` -- and refused on
       confidence alone rather than on content, structure or cue evidence;
    4. structural agreement clears :data:`LARGE_OFFSET_MIN_STRUCTURAL_SIMILARITY`.

    Condition 3 is the scoped exception that makes this feature work. The
    general verifier refuses these cases on residual p95 alone, because cue
    segmentation differs between releases; that is precisely the measurement
    this path exempts, and it is the *only* one it exempts. Everything else the
    verifier says still binds: a ``REJECTED``-for-content or structural
    outcome, a missing measurement (``None`` MAD), or a correction that was not
    one constant shift all return the original.

    This never weakens :func:`~.alignment.may_serve_synchronized` or
    :func:`~.alignment.is_reusable_verified`. The analyzer keeps recording its
    own verdict unchanged, ``ALASS_CORRECTED_LARGE_OFFSET`` is not a
    ``SyncState`` and never maps to one, and no cache entry becomes reusable
    because of this function.

    ``evaluation=None`` fails closed: without the analyzer's own measurement
    there is no way to tell a segmentation-driven refusal from a structural
    one, so the original is served.
    """
    investigation.serving_reason_codes = []

    if not investigation.eligible_for_alass:
        investigation.serving_reason_codes.append(SERVE_REASON_NOT_ELIGIBLE)
        logger.info(
            "large_offset.episode_identity_failed decision=%s same_episode=%s "
            "reasons=%s",
            investigation.decision.value,
            investigation.same_episode.value,
            ",".join(investigation.reason_codes) or "none",
        )
        return _record_serving(investigation, LargeOffsetServingState.ORIGINAL)

    if not validation.ok:
        investigation.serving_reason_codes.append(SERVE_REASON_ALASS_INVALID)
        logger.warning("large_offset.alass_output_invalid %s", validation.summary())
        return _record_serving(investigation, LargeOffsetServingState.ORIGINAL)

    if evaluation is None:
        investigation.serving_reason_codes.append(SERVE_REASON_NO_EVALUATION)
        logger.info("large_offset.serving_denied %s", SERVE_REASON_NO_EVALUATION)
        return _record_serving(investigation, LargeOffsetServingState.ORIGINAL)

    mad = evaluation.mad_offset_ms
    if mad is None or mad > LARGE_OFFSET_MAX_MOVEMENT_MAD_MS:
        # Not measurable, or not one constant shift: "large systematic offset"
        # is not what was corrected, so nothing may be served in its place.
        investigation.serving_reason_codes.append(
            SERVE_REASON_MOVEMENT_NOT_CONSTANT
        )
        logger.info(
            "large_offset.serving_denied %s movement_mad_ms=%s bound_ms=%.0f",
            SERVE_REASON_MOVEMENT_NOT_CONSTANT,
            mad,
            LARGE_OFFSET_MAX_MOVEMENT_MAD_MS,
        )
        return _record_serving(investigation, LargeOffsetServingState.ORIGINAL)

    rejection = evaluation.rejection_reason
    if rejection is not None and rejection is not RejectionReason.LOW_CONFIDENCE:
        # The verifier refused on content, structure, cue loss or invalid
        # timing. Only a confidence objection is in scope for this path.
        investigation.serving_reason_codes.append(SERVE_REASON_STRUCTURAL_REJECTED)
        logger.info(
            "large_offset.serving_denied %s rejection=%s",
            SERVE_REASON_STRUCTURAL_REJECTED,
            rejection.value,
        )
        return _record_serving(investigation, LargeOffsetServingState.ORIGINAL)

    structure = evaluation.structural_similarity
    if structure is None or structure < LARGE_OFFSET_MIN_STRUCTURAL_SIMILARITY:
        investigation.serving_reason_codes.append(SERVE_REASON_STRUCTURE_TOO_LOW)
        logger.info(
            "large_offset.serving_denied %s structural_similarity=%s floor=%.2f",
            SERVE_REASON_STRUCTURE_TOO_LOW,
            structure,
            LARGE_OFFSET_MIN_STRUCTURAL_SIMILARITY,
        )
        return _record_serving(investigation, LargeOffsetServingState.ORIGINAL)

    investigation.serving_reason_codes.append(SERVE_REASON_CORRECTED)
    logger.info(
        "large_offset.serving_allowed movement_mad_ms=%.0f p95_ms=%s "
        "structural_similarity=%.3f rejection=%s serving_state=%s",
        mad,
        evaluation.p95_offset_ms,
        structure,
        rejection.value if rejection else "none",
        LargeOffsetServingState.ALASS_CORRECTED_LARGE_OFFSET.value,
    )
    return _record_serving(
        investigation, LargeOffsetServingState.ALASS_CORRECTED_LARGE_OFFSET
    )


def _record_serving(
    investigation: LargeOffsetInvestigation, state: LargeOffsetServingState
) -> LargeOffsetServingState:
    """Stamp the outcome on the investigation so the caller can log it."""
    investigation.serving_state = state
    return state
