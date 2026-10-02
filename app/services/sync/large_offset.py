"""Large Offset Evidence Gate.

The normal first-dialogue execution window
(:data:`subtitle_matcher.FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS`) is 20s, and
anything beyond it is rejected as a different cut. That is the right default,
but it is wrong for a real and recurring shape: a subtitle that is the *same
episode* re-timed onto a *different master* whose opening runs long, so the
whole file sits ~90-100s later. The confirmed case is
``Dexter.s8e04.Scar.tissue.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv`` (first dialogue
107.5s) against compatible BluRay references (first dialogue ~10.6s): a
~96.6s displacement that is a constant shift, not a mismatch.

This module decides whether such a displacement is a *constant offset between
the same timeline* or a *different cut*. It answers one narrow question --
"may alass be attempted?" -- and deliberately grants no verification at all.
The post-alass analyzer in :mod:`.alignment` remains the only authority on
whether a result is a sync, and nothing here can mark a subtitle verified.

What counts as evidence, and why each signal is required:

* **Identity** is a precondition, not proof. The existing reference selector
  already matched this reference to this exact IMDb/season/episode and ruled
  out release-family incompatibility; the gate reuses that verdict instead of
  re-implementing release matching. A same-episode label with a structurally
  unrelated timeline still fails on the signals below.
* **Several distant anchors** must independently recover the *same* offset.
  One cue is not evidence -- a single first-cue comparison is exactly the check
  that produced the false rejection, so it is never sufficient on its own.
* **Low dispersion and no drift** separate a constant shift from a re-cut or a
  progressively re-timed file. Dispersion and slope are measured with the
  project's existing :func:`~.alignment.analyze_drift` and median helpers.
* **Timing-structure similarity** compares where subtitle *events* fall, never
  what they say, so an Arabic target matches an English reference and vice
  versa. Text is never inspected and no language model or media decode is used.
* **Ceiling**: an offset beyond :data:`LARGE_OFFSET_MAX_SECONDS` is rejected
  before any of the above, so the gate cannot become an unbounded search.

All statistics are bounded (:data:`LARGE_OFFSET_MAX_SAMPLED_CUES`,
:data:`LARGE_OFFSET_CUES_PER_REGION`, :data:`LARGE_OFFSET_REGION_COUNT`) and
deterministic: the same inputs always produce the same anchors in the same
order, with no randomness, no network, and no ffmpeg.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence

from pydantic import BaseModel, Field

from ..subtitle_matcher import (
    first_dialogue_cluster,
    parse_srt_cues,
    strip_intro_nonspeech,
)
from .alignment import (
    MAX_DRIFT_MS_PER_MINUTE,
    Cue,
    _median,
    analyze_drift,
)

# --------------------------------------------------------------------------- #
# Tunables. Internal, deterministic, and deliberately conservative.
# --------------------------------------------------------------------------- #

# Bounded ceiling for the large-offset path. A TV episode whose entire body
# sits more than three minutes away from a compatible reference is a different
# cut, a different episode, or a broken file -- not a re-timing alass can fix.
# This is an implementation starting point rather than a proven universal
# value; `tools/run_sync_benchmarks.py` reports the offsets it observes so the
# ceiling can be retuned against evidence instead of opinion.
LARGE_OFFSET_MAX_SECONDS = 180
LARGE_OFFSET_MAX_MS = LARGE_OFFSET_MAX_SECONDS * 1000

# Timeline is split into this many equal-duration regions and one local offset
# is estimated per region, so evidence is spread across the episode instead of
# clustering in the opening. Five regions, four required: enough to distinguish
# a constant shift from a cut that happens to start aligned, while still
# tolerating a region whose cues are missing or unrepresentative.
LARGE_OFFSET_REGION_COUNT = 5
LARGE_OFFSET_MIN_ANCHORS = 4

# Cues sampled per region. Each is paired against the reference under the
# current offset hypothesis; the region reports their median, which makes a
# single stray cue harmless.
LARGE_OFFSET_CUES_PER_REGION = 6

# A sampled cue counts as an anchor only if a reference cue sits within this
# distance of where the hypothesis predicts it. Deliberately *wider* than
# ``LARGE_OFFSET_MAX_DISPERSION_MS``: the tolerance answers "is this region
# plausibly the same constant offset", while dispersion answers "do the regions
# actually agree". Keeping the tolerance loose enough to admit mildly scattered
# regions is what lets the dispersion ceiling reject them on their merits
# instead of every bad case collapsing into "not enough anchors".
LARGE_OFFSET_ANCHOR_TOLERANCE_MS = 2500

# Hard ceiling on sampling, so a 5000-cue file costs the same as a 500-cue one.
LARGE_OFFSET_MAX_SAMPLED_CUES = 600

# Median absolute deviation across regional medians. Above this the regions do
# not describe one displacement and the reference is a different cut.
LARGE_OFFSET_MAX_DISPERSION_MS = 1200.0

# Slope ceiling, in ms per minute, reused from the alignment analyzer so the
# large-offset path and post-alass verification judge drift identically.
LARGE_OFFSET_MAX_DRIFT_MS_PER_MINUTE = MAX_DRIFT_MS_PER_MINUTE

# Minimum cosine similarity between the target's cue-event profile and the
# reference's profile shifted by the estimated offset. Language-agnostic: it
# reads only *where* events fall, never their text.
LARGE_OFFSET_MIN_STRUCTURAL_SIMILARITY = 0.55

# Width of one structural histogram bin.
LARGE_OFFSET_DENSITY_BIN_MS = 30_000

# Two references count as agreeing when their independently estimated offsets
# fall within this distance of each other.
LARGE_OFFSET_CONSENSUS_TOLERANCE_MS = 2000.0

# Reason codes. Stable strings so tests and logs can assert on them.
REASON_MAX_OFFSET_EXCEEDED = "max_offset_exceeded"
REASON_IDENTITY_UNSUPPORTED = "identity_unsupported"
REASON_NO_DIALOGUE_SEED = "no_dialogue_seed"
REASON_INSUFFICIENT_ANCHORS = "insufficient_anchors"
REASON_DISPERSION_TOO_HIGH = "dispersion_too_high"
REASON_DRIFT_TOO_HIGH = "drift_too_high"
REASON_STRUCTURAL_MISMATCH = "structural_mismatch"
REASON_CONSENSUS_AGREES = "consensus_agrees"
REASON_CONSENSUS_CONFLICTS = "consensus_conflicts"
REASON_ANCHORS_CONSISTENT = "anchors_consistent"


class LargeOffsetAssessment(BaseModel):
    """Outcome of the Large Offset Evidence Gate.

    ``accepted`` means only "alass may be attempted against this reference". It
    is never a sync claim: ``sync_state``/``verification`` stay with the
    post-alass analyzer, and a rejected assessment never reaches a transform.
    """

    # Whether alass may be attempted. Not a verification result.
    accepted: bool = False

    # Robust consensus of the regional medians, in ms (target - reference,
    # matching validate_cue_sanity's sign convention).
    estimated_offset_ms: float | None = None
    offset_seed_ms: int | None = None

    # Median absolute deviation across regional medians, in ms.
    offset_dispersion_ms: float | None = None
    # Least-squares slope of local offset over time, in ms per minute.
    drift_ms_per_minute: float | None = None

    # How many of the sampled regions produced an anchor near the hypothesis.
    anchor_count: int = 0
    regions_sampled: int = 0

    # Cosine similarity of the cue-event profiles, target vs shifted reference.
    structural_similarity: float | None = None

    # How many *other, independent* references independently agreed.
    consensus_count: int = 0

    # Supporting identity facts already established by the reference selector.
    identity_supported: bool = False
    same_release_family: bool | None = None

    # Stable machine-readable codes, most decisive first.
    reason_codes: list[str] = Field(default_factory=list)

    def summary(self) -> str:
        """One-line, credential-free description for logs."""
        offset = (
            "n/a"
            if self.estimated_offset_ms is None
            else f"{self.estimated_offset_ms / 1000.0:+.2f}s"
        )
        similarity = (
            "n/a"
            if self.structural_similarity is None
            else f"{self.structural_similarity:.3f}"
        )
        return (
            f"accepted={self.accepted} offset={offset} "
            f"anchors={self.anchor_count}/{self.regions_sampled} "
            f"dispersion_ms={self.offset_dispersion_ms} "
            f"drift_ms_per_min={self.drift_ms_per_minute} "
            f"structural={similarity} consensus={self.consensus_count} "
            f"reasons={','.join(self.reason_codes) or 'none'}"
        )


def _as_cues(value: str | Sequence[Cue]) -> list[Cue]:
    """Parse once, then sort, so every downstream stage sees a stable order."""
    if isinstance(value, str):
        parsed = parse_srt_cues(value)
    else:
        parsed = [tuple(cue) for cue in value]  # type: ignore[misc]
    return sorted(parsed, key=lambda cue: cue[0])


def _bounded(cues: list[Cue], limit: int = LARGE_OFFSET_MAX_SAMPLED_CUES) -> list[Cue]:
    """Deterministically thin a cue list to at most ``limit`` entries.

    An even stride keeps the surviving cues spread across the whole timeline
    rather than clustered at the head, which matters because the regions are
    derived from the span these cues cover.
    """
    if len(cues) <= limit:
        return cues
    stride = math.ceil(len(cues) / limit)
    return cues[::stride][:limit]


def _nearest_reference_offset(
    target_start: int,
    reference_starts: list[int],
    hypothesis: float,
    tolerance_ms: int,
) -> float | None:
    """Local offset of the reference cue nearest ``target_start - hypothesis``.

    Returns ``target_start - reference_start`` so the sign matches
    :func:`validate_cue_sanity`, or ``None`` when no reference cue lands within
    ``tolerance_ms`` of the prediction. A tolerance keeps a real-but-imperfect
    hypothesis from failing outright; the region median then absorbs the slack.
    """
    if not reference_starts:
        return None
    predicted = target_start - hypothesis
    index = bisect.bisect_left(reference_starts, predicted)
    best: int | None = None
    for candidate in (index - 1, index, index + 1):
        if 0 <= candidate < len(reference_starts):
            if best is None or abs(reference_starts[candidate] - predicted) < abs(
                reference_starts[best] - predicted
            ):
                best = candidate
    if best is None or abs(reference_starts[best] - predicted) > tolerance_ms:
        return None
    return float(target_start - reference_starts[best])


def _region_medians(
    target_cues: list[Cue],
    reference_starts: list[int],
    hypothesis: float,
) -> tuple[list[tuple[int, float]], int, list[tuple[int, float]]]:
    """Per-region median local offsets, the region count, and every sample.

    The timeline is split into equal-duration regions so anchors are spread
    early / early-middle / middle / late-middle / late rather than clustering
    at the start. Each region samples a bounded, evenly strided set of target
    cues and reports the median of those that pair up, which keeps a region
    usable when some of its cues are missing from the reference.

    The third return value is the full list of cue-level ``(position, offset)``
    pairs, not the region medians. :func:`~.alignment.analyze_drift` needs at
    least ``MIN_CUES_FOR_ANALYSIS`` points to fit a slope, and five region
    medians can never reach that -- so drift has to be measured across the
    individual anchors or the progressive-drift guard would never fire.
    """
    if not target_cues:
        return [], 0, []
    span_start = target_cues[0][0]
    span = max(1, target_cues[-1][0] - span_start)
    region_span = span / LARGE_OFFSET_REGION_COUNT
    collected: list[tuple[int, float]] = []
    samples: list[tuple[int, float]] = []
    for region in range(LARGE_OFFSET_REGION_COUNT):
        lower = span_start + region * region_span
        upper = span_start + (region + 1) * region_span
        bucket = [
            cue
            for cue in target_cues
            if lower <= cue[0] < upper
            or (region == LARGE_OFFSET_REGION_COUNT - 1 and cue[0] >= upper)
        ]
        if not bucket:
            continue
        stride = max(1, len(bucket) // LARGE_OFFSET_CUES_PER_REGION)
        offsets: list[float] = []
        position = bucket[0][0]
        for start, _, _ in bucket[::stride][:LARGE_OFFSET_CUES_PER_REGION]:
            local = _nearest_reference_offset(
                start, reference_starts, hypothesis, LARGE_OFFSET_ANCHOR_TOLERANCE_MS
            )
            if local is None:
                continue
            offsets.append(local)
            position = start
            samples.append((start, local))
        if not offsets:
            continue
        collected.append((position, _median(offsets)))
    return collected, LARGE_OFFSET_REGION_COUNT, samples


def _shifted_profile_similarity(
    target_cues: list[Cue],
    reference_cues: list[Cue],
    estimated_offset_ms: float,
) -> float | None:
    """Cosine similarity of cue-event profiles after shifting the reference.

    Builds a histogram of where subtitle events *start* for the target, and a
    second histogram for the reference displaced by ``estimated_offset_ms``.
    If the two describe the same episode under one constant shift, the profiles
    line up; a different cut, a different episode, or a drifting timeline does
    not. Text is never read, so this is language-agnostic and equally valid for
    an Arabic target against an English reference. Binned to
    :data:`LARGE_OFFSET_DENSITY_BIN_MS`, so cost is O(cues + bins).
    """
    if not target_cues or not reference_cues:
        return None
    shifted_starts: list[float] = [
        start + estimated_offset_ms for start, _, _ in reference_cues
    ]
    origin = min(target_cues[0][0], shifted_starts[0])
    last = max(target_cues[-1][0], shifted_starts[-1])
    bins = max(1, int((last - origin) // LARGE_OFFSET_DENSITY_BIN_MS) + 1)
    target_counts = [0.0] * bins
    reference_counts = [0.0] * bins
    for start, _, _ in target_cues:
        index = int((start - origin) // LARGE_OFFSET_DENSITY_BIN_MS)
        if 0 <= index < bins:
            target_counts[index] += 1.0
    for moved in shifted_starts:
        index = int((moved - origin) // LARGE_OFFSET_DENSITY_BIN_MS)
        if 0 <= index < bins:
            reference_counts[index] += 1.0
    dot = sum(a * b for a, b in zip(target_counts, reference_counts, strict=True))
    target_norm = math.sqrt(sum(a * a for a in target_counts))
    reference_norm = math.sqrt(sum(b * b for b in reference_counts))
    if target_norm <= 0.0 or reference_norm <= 0.0:
        return None
    return dot / (target_norm * reference_norm)


def _consensus_count(
    estimated_offset_ms: float,
    other_offsets_ms: Sequence[float] | None,
) -> int:
    """How many other references independently agree on the same offset.

    The caller is responsible for passing only *independent* references; the
    gate cannot tell two identical downloads apart, so de-duplication belongs
    to the caller that already holds the reference-family and download state.
    """
    if not other_offsets_ms:
        return 0
    return sum(
        1
        for other in other_offsets_ms
        if abs(other - estimated_offset_ms) <= LARGE_OFFSET_CONSENSUS_TOLERANCE_MS
    )


def assess_large_offset(
    target: str | Sequence[Cue],
    reference: str | Sequence[Cue],
    *,
    seed_offset_ms: int | None,
    identity_supported: bool,
    same_release_family: bool | None = None,
    consensus_offsets_ms: Sequence[float] | None = None,
    max_offset_ms: int = LARGE_OFFSET_MAX_MS,
) -> LargeOffsetAssessment:
    """Decide whether a large first-dialogue offset is a constant shift.

    Called only after the normal 20s cue-sanity window has already failed, so
    ``seed_offset_ms`` is that already-measured opening disagreement rather
    than a fresh guess -- and ``None`` when the opening could not be measured
    at all, which abstains instead of guessing. Two deterministic passes refine
    it: the first seeds a hypothesis from the opening dialogue, the second
    re-measures the anchors against the median those anchors produced, so one
    badly-placed opening cue cannot decide the outcome.

    Returns a :class:`LargeOffsetAssessment`. ``accepted=True`` authorizes an
    alass attempt and nothing more; every post-alass check, and the decision to
    serve anything at all, stays with the existing analyzer.
    """
    assessment = LargeOffsetAssessment(
        offset_seed_ms=seed_offset_ms,
        identity_supported=identity_supported,
        same_release_family=same_release_family,
    )

    if seed_offset_ms is None:
        # No measured opening disagreement means no evidence base at all.
        # Abstaining is the fail-safe answer: without a seed there is nothing
        # to corroborate, and "large offset" is not something to assume.
        assessment.reason_codes.append(REASON_NO_DIALOGUE_SEED)
        return assessment

    # Ceiling first, so an absurd offset never pays for the analysis below.
    if abs(seed_offset_ms) > max_offset_ms:
        assessment.reason_codes.append(REASON_MAX_OFFSET_EXCEEDED)
        return assessment

    if not identity_supported:
        assessment.reason_codes.append(REASON_IDENTITY_UNSUPPORTED)
        return assessment

    target_cues = _bounded(_as_cues(target))
    reference_cues = _bounded(_as_cues(reference))
    if not target_cues or not reference_cues:
        assessment.reason_codes.append(REASON_NO_DIALOGUE_SEED)
        return assessment

    reference_starts = [start for start, _, _ in reference_cues]
    hypothesis = float(seed_offset_ms)
    regions: list[tuple[int, float]] = []
    samples: list[tuple[int, float]] = []
    regions_sampled = LARGE_OFFSET_REGION_COUNT
    for _ in range(2):  # seed pass, then a pass anchored on their median
        regions, regions_sampled, samples = _region_medians(
            target_cues, reference_starts, hypothesis
        )
        offsets = [value for _, value in regions]
        if len(offsets) >= LARGE_OFFSET_MIN_ANCHORS:
            hypothesis = _median(offsets)

    offsets = [value for _, value in regions]
    assessment.regions_sampled = regions_sampled
    assessment.anchor_count = len(offsets)
    if not offsets:
        assessment.reason_codes.append(REASON_INSUFFICIENT_ANCHORS)
        return assessment

    estimated = _median(offsets)
    assessment.estimated_offset_ms = round(estimated, 1)
    assessment.offset_dispersion_ms = round(
        _median([abs(value - estimated) for value in offsets]), 1
    )
    # Measured across the individual anchors, not the region medians, so the
    # slope fit has enough points to see a progressive shift.
    assessment.drift_ms_per_minute = analyze_drift(samples)
    if assessment.drift_ms_per_minute is not None:
        assessment.drift_ms_per_minute = round(assessment.drift_ms_per_minute, 2)
    assessment.structural_similarity = _shifted_profile_similarity(
        target_cues, reference_cues, estimated
    )
    assessment.consensus_count = _consensus_count(estimated, consensus_offsets_ms)

    if len(offsets) < LARGE_OFFSET_MIN_ANCHORS:
        assessment.reason_codes.append(REASON_INSUFFICIENT_ANCHORS)
        return assessment
    if abs(estimated) > max_offset_ms:
        assessment.reason_codes.append(REASON_MAX_OFFSET_EXCEEDED)
        return assessment
    if (assessment.offset_dispersion_ms or 0.0) > LARGE_OFFSET_MAX_DISPERSION_MS:
        assessment.reason_codes.append(REASON_DISPERSION_TOO_HIGH)
        return assessment
    if assessment.drift_ms_per_minute is not None and (
        abs(assessment.drift_ms_per_minute) > LARGE_OFFSET_MAX_DRIFT_MS_PER_MINUTE
    ):
        assessment.reason_codes.append(REASON_DRIFT_TOO_HIGH)
        return assessment
    if assessment.structural_similarity is None or (
        assessment.structural_similarity < LARGE_OFFSET_MIN_STRUCTURAL_SIMILARITY
    ):
        assessment.reason_codes.append(REASON_STRUCTURAL_MISMATCH)
        return assessment

    assessment.reason_codes.append(REASON_ANCHORS_CONSISTENT)
    if assessment.consensus_count:
        assessment.reason_codes.append(REASON_CONSENSUS_AGREES)
    elif consensus_offsets_ms:
        assessment.reason_codes.append(REASON_CONSENSUS_CONFLICTS)
    assessment.accepted = True
    return assessment


def classify_large_offset_candidate(
    target: str | Sequence[Cue],
    reference: str | Sequence[Cue],
    *,
    threshold_ms: int,
) -> int | None:
    """Seed offset when a reference misses the normal window, else ``None``.

    The single place that decides "this is not the normal fast path, it is a
    large-offset candidate". Reuses the existing
    :func:`~.subtitle_matcher.validate_cue_sanity` and dialogue parsing so the
    two paths cannot drift apart, and deliberately returns only the *measured*
    opening disagreement -- never a decision.
    """
    target_first = first_dialogue_cluster(
        strip_intro_nonspeech(target if isinstance(target, str) else list(target))
    )
    reference_first = first_dialogue_cluster(
        strip_intro_nonspeech(reference if isinstance(reference, str) else list(reference))
    )
    if target_first is None or reference_first is None:
        return None
    seed = target_first - reference_first
    if abs(seed) <= threshold_ms:
        return None
    return seed
