"""Interval correspondence: anchor-bounded, monotonic, split/merge aware.

DIAGNOSTIC ONLY. Imported by no production module (asserted by test). Writes no
verdict, artifact, alias or cache entry. Never logs subtitle text.

This replaces nearest-start correspondence. It exists because the previous
experiment measured the actual failure:

* A one-to-one nearest-start model has no way to say "this target cue is part of a
  reference cue that was split into three lines". It instead force-pairs the
  surplus, and the surplus shows up as a fat high-residual tail.
* On the ASAP fixture the target carries ~28% more cues than the reference at
  almost identical per-cue duration (780x2300ms vs 562x2250ms). Per-quarter p50
  was 77-307ms -- the timing is fine -- while p95 was 2067-2357ms in *every*
  quarter, because the surplus cues had nowhere to go.

So the model must be able to represent a legitimate segmentation difference and
must be willing to declare surplus. Both are first-class here.

Design, kept deliberately small:

* Monotonic dynamic programming over (target index, reference index). Each move
  pairs a contiguous run of target cues with a contiguous run of reference cues,
  which covers 1:1, split (1 ref -> k target) and merge (k ref -> 1 target)
  without a bespoke rule for each.
* Search is banded around the anchor mapping, so once anchors are established the
  model cannot wander across the whole reference.
* Skipping is *cheaper* than a poor match. This is the primary safety property:
  the objective is built so that force-matching a bad candidate costs more than
  declaring it unmatchable.
* Reference coverage and target coverage are accounted separately, so "the
  reference is only 60% explained" is observable rather than inferred.
* Text is never used. The real reference is a different release and language, and
  measured text overlap with both targets is 4-6%.
"""

from __future__ import annotations

import bisect
import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum

from app.services.sync.anchor_correspondence import (
    AnchorMapping,
    AnchorRegion,
    MappingSegment,
    build_mapping,
    collect_anchor_regions,
)

#: Half-width of the search band around the anchor-mapped prediction. The band
#: IS the anchor bound: outside it, a reference cue is not considered at all.
#: Widening it is how a bounded model becomes an unbounded one, so a mutation
#: that widens it must be caught.
SEARCH_WINDOW_MS = 4000

#: Largest number of cues on either side of a single correspondence group.
#: Unlimited grouping would let any target cue be "explained" by any number of
#: reference cues and would absorb duplication as if it were segmentation.
MAX_GROUP = 3

#: Largest temporal span a single group may cover. Bounds grouping in time even
#: when the count limit permits it.
GROUP_SPAN_MS = 12000

#: A group is only legitimate if, after alignment, the two spans genuinely
#: overlap. Prevents grouping cues that merely sit near each other.
MIN_GROUP_OVERLAP_RATIO = 0.4

#: Hard ceiling on how far a group's centre may sit from the mapped prediction.
#: A group that needs more than this is not a correspondence.
MAX_GROUP_CENTER_MS = 2500

#: Objective weights. A skip must be cheaper than any match beyond
#: ``MATCH_TOLERANT_CENTER_MS``; that inequality is what makes surplus visible.
MATCH_BASE_COST = 0.0
MATCH_COST_PER_MS = 1.0
MATCH_TOLERANT_CENTER_MS = 400.0
SKIP_TARGET_COST = 900.0
SKIP_REFERENCE_COST = 60.0
GROUP_EXTRA_COST = 250.0
IMPASSABLE = float("inf")

#: Diagnostic only: what fraction of the reference must the alignment explain
#: before the result may be described as aligned. Not a production threshold and
#: not derived from any verifier limit. "Aligned" is meant to mean "most of the
#: reference is accounted for"; below this the model says what is missing
#: instead. It is deliberately not tuned per case -- one number, used once.
ALIGNED_MIN_REFERENCE_COVERAGE = 0.8

#: Diagnostic only: surplus beyond these fractions is reported as the dominant
#: explanation rather than hidden inside "aligned".
SURPLUS_MIN_FRACTION = 0.15

#: Diagnostic only: how far total target speech time may differ from the
#: reference's before it is reported as a content volume difference. NOT an
#: acceptance threshold and not part of the timing outcome.
CONTENT_VOLUME_TOLERANCE = 0.05


class ContentObservation(str, Enum):
    """Observed content volume. Explicitly NOT a timing claim.

    This is where duplication and deletion become visible. Timing correspondence
    cannot see duplication: an adjacent duplicate of a cue is a perfectly legal
    SPLIT group, so a duplicated file presents as fully aligned with zero
    residual. Total speech time separates the two, because *splitting* a cue
    preserves its total time while *duplicating* one adds to it.
    """

    CONTENT_COMPARABLE = "CONTENT_COMPARABLE"
    TARGET_CONTENT_EXCESS = "TARGET_CONTENT_EXCESS"
    TARGET_CONTENT_LOSS = "TARGET_CONTENT_LOSS"


class GroupKind(str, Enum):
    ONE_TO_ONE = "ONE_TO_ONE"
    SPLIT = "SPLIT"          # one reference cue -> several target cues
    MERGE = "MERGE"          # several reference cues -> one target cue
    SPLIT_MERGE = "SPLIT_MERGE"


@dataclass(frozen=True)
class Group:
    """One correspondence: contiguous runs of target and reference cues."""

    target_indices: tuple[int, ...]
    reference_indices: tuple[int, ...]
    #: Distance from the group's reference centre to the mapped prediction.
    center_delta_ms: float
    #: For each target cue in the group, how far its mapped prediction sits
    #: outside the reference span. Zero when the prediction lands inside.
    interval_residuals_ms: tuple[float, ...]
    kind: GroupKind

    @property
    def size(self) -> int:
        return len(self.target_indices)


class AlignmentOutcome(str, Enum):
    """What the alignment found. Not a verdict -- nothing here accepts a sync."""

    ALIGNED = "INTERVAL_ALIGNED"
    #: Alignment exists but the reference is only partly explained.
    LOW_REFERENCE_COVERAGE = "INTERVAL_LOW_REFERENCE_COVERAGE"
    #: Target cues with no reference partner: the target carries more content.
    SURPLUS_TARGET = "INTERVAL_SURPLUS_TARGET"
    #: Reference cues with no target partner: the target carries less content.
    SURPLUS_REFERENCE = "INTERVAL_SURPLUS_REFERENCE"
    #: Grouping or monotonicity limits stopped an otherwise plausible match.
    AMBIGUOUS = "INTERVAL_AMBIGUOUS"
    #: No alignment at all.
    NO_CORRESPONDENCE = "INTERVAL_NO_CORRESPONDENCE"
    #: Anchors did not agree; the band would have been untrustworthy.
    ANCHORS_INCONSISTENT = "INTERVAL_ANCHORS_INCONSISTENT"


@dataclass
class IntervalReport:
    outcome: AlignmentOutcome
    groups: list[Group]
    mapping: AnchorMapping
    #: Number of anchor regions that disagreed beyond the agreement margin.
    inconsistent_regions: int

    target_cues: int = 0
    reference_cues: int = 0
    matched_target: int = 0
    matched_reference: int = 0
    ambiguous_target: list[int] = field(default_factory=list)
    surplus_target: list[int] = field(default_factory=list)
    surplus_reference: list[int] = field(default_factory=list)
    total_cost: float = 0.0
    completeness: dict[str, float] = field(default_factory=dict)
    content: dict[str, float] = field(default_factory=dict)
    content_observation: ContentObservation = ContentObservation.CONTENT_COMPARABLE
    #: Populated only by the graded-anchor variant.
    anchor_grading: AnchorGrading | None = None
    notes: list[str] = field(default_factory=list)

    # ---- coverage, reported separately on purpose -------------------------- #
    @property
    def target_coverage(self) -> float:
        return self.matched_target / self.target_cues if self.target_cues else 0.0

    @property
    def reference_coverage(self) -> float:
        return (
            self.matched_reference / self.reference_cues if self.reference_cues else 0.0
        )

    @property
    def group_count(self) -> int:
        return len(self.groups)

    def residuals(self) -> list[float]:
        return [g.center_delta_ms for g in self.groups]

    def pct(self, q: float) -> float:
        vals = self.residuals()
        if not vals:
            return float("inf")
        ordered = sorted(vals)
        k = max(0, min(len(ordered) - 1, int(round(q * (len(ordered) - 1)))))
        return float(ordered[k])

    def p50(self) -> float:
        return self.pct(0.5)

    def p95(self) -> float:
        return self.pct(0.95)

    def p99(self) -> float:
        return self.pct(0.99)

    def worst(self) -> float:
        return max((abs(g.center_delta_ms) for g in self.groups), default=0.0)

    def split_count(self) -> int:
        return sum(1 for g in self.groups if g.kind is GroupKind.SPLIT)

    def merge_count(self) -> int:
        return sum(1 for g in self.groups if g.kind is GroupKind.MERGE)

    def as_row(self) -> dict[str, object]:
        return {
            "outcome": self.outcome.value,
            "groups": self.group_count,
            "target_cues": self.target_cues,
            "reference_cues": self.reference_cues,
            "matched_target": self.matched_target,
            "matched_reference": self.matched_reference,
            "target_coverage": round(self.target_coverage, 4),
            "reference_coverage": round(self.reference_coverage, 4),
            "splits": self.split_count(),
            "merges": self.merge_count(),
            "ambiguous": len(self.ambiguous_target),
            "surplus_target": len(self.surplus_target),
            "surplus_reference": len(self.surplus_reference),
            "p50": round(self.pct(0.5), 1),
            "p95": round(self.pct(0.95), 1),
            "p99": round(self.pct(0.99), 1),
            "worst": round(max((abs(g.center_delta_ms) for g in self.groups),
                               default=0.0), 1),
            "mapping_pieces": self.mapping.piece_count,
            "offsets": [s.offset_ms for s in self.mapping.segments],
            "breakpoints": list(self.mapping.breakpoints_ms),
            "content_observation": self.content_observation.value,
            "completeness": dict(self.completeness),
            "content": dict(self.content),
        }


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #


def _centre(start: int, end: int) -> float:
    return (start + end) / 2.0


def _span(start: float, end: float) -> float:
    return float(max(0, end - start))


def offset_at(mapping: AnchorMapping, when_ms: float) -> float:
    """Local offset (``target - reference``) in force at ``when_ms``."""
    segments = mapping.segments
    if not segments:
        return 0.0
    for seg in segments:
        if seg.start_ms <= when_ms < seg.end_ms:
            return seg.offset_ms
    return segments[-1].offset_ms if when_ms >= segments[-1].start_ms else (
        segments[0].offset_ms
    )


def anchors_consistent(
    mapping: AnchorMapping,
    agreement_ms: float,
) -> tuple[bool, int]:
    """Do the anchor regions agree well enough to trust the band?

    A disagreement means the model would be searching around a different offset
    in different parts of the file. Rather than propagate that silently, the
    caller abstains.
    """
    bad = 0
    segs = mapping.segments
    for i in range(len(segs) - 1):
        if abs(segs[i + 1].offset_ms - segs[i].offset_ms) > agreement_ms:
            bad += 1
    return bad == 0, bad


# --------------------------------------------------------------------------- #
# Anchor grading (PART 2)
# --------------------------------------------------------------------------- #


class AnchorGrade(str, Enum):
    """How much an individual anchor may be trusted. Diagnostic only."""

    #: Agrees tightly with the estimated offset.
    HARD = "ANCHOR_HARD"
    #: Within the dispersion the gate already accepts, but not tightly.
    SOFT = "ANCHOR_SOFT"
    #: Beyond what the gate accepts. Excluded from the band, never fatal.
    REJECTED = "ANCHOR_REJECTED"


@dataclass(frozen=True)
class GradedAnchor:
    """One anchor region with its position, measured offset and grade."""

    position_ms: int
    offset_ms: float
    grade: AnchorGrade
    deviation_ms: float
    implied_reference_ms: int


@dataclass
class AnchorGrading:
    graded: list[GradedAnchor]
    estimated_offset_ms: float
    dispersion_ms: float
    slope_ms_per_ms: float
    origin_ms: int
    hard: list[GradedAnchor]
    soft: list[GradedAnchor]
    rejected: list[GradedAnchor]
    #: Anchors that contradict a single monotone trend.
    contradictory: bool
    #: Spread of trusted anchors about the fitted trend, in ms.
    residual_spread: float = 0.0
    notes: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {
            "hard": len(self.hard),
            "soft": len(self.soft),
            "rejected": len(self.rejected),
        }


def _median_of(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def grade_anchors(
    regions: list[tuple[int, float]],
    dispersion_limit_ms: float,
    hard_fraction: float,
    drift_ms_per_minute: float | None = None,
    window_ms: int = SEARCH_WINDOW_MS,
) -> AnchorGrading:
    """Grade anchors using the same statistic the production gate uses.

    The gate asks "is this one shift?" and answers with the median absolute
    deviation about the median of the region medians. An earlier version of this
    experiment instead asked "is the offset identical between adjacent regions?"
    -- a strictly local question that rejects real drift: EVOLV's anchors drift
    at -51.76 ms/min, so adjacent regions differ by up to 1990ms while the gate's
    dispersion is 950ms and correctly accepted.

    So the *global* statistic is used to decide whether a region is an outlier at
    all (``dispersion_limit_ms``, the gate's own limit), and ``hard_fraction`` of
    it decides which surviving regions are tight enough to pin the band. No new
    threshold is invented and the margin is not loosened: the gate's limit is
    used unchanged.

    ``drift_ms_per_minute`` should be the gate's own measurement, taken across the
    individual anchor samples rather than the five region medians. Re-deriving a
    slope from five medians is strictly worse: on EVOLV the two tightest anchors
    sit next to each other in the middle, so fitting them yields +15.7 ms/min
    where the gate measures -51.76 ms/min -- the wrong sign, from the same data.
    """
    if not regions:
        return AnchorGrading(
            graded=[],
            estimated_offset_ms=0.0,
            dispersion_ms=float("nan"),
            slope_ms_per_ms=0.0,
            origin_ms=0,
            hard=[],
            soft=[],
            rejected=[],
            contradictory=False,
            notes=["no anchor regions"],
        )
    offsets = [v for _, v in regions]
    estimated = _median_of(offsets)
    dispersion = _median_of([abs(v - estimated) for v in offsets])

    hard_band = hard_fraction * dispersion_limit_ms
    graded: list[GradedAnchor] = []
    for position, offset in regions:
        deviation = abs(offset - estimated)
        if deviation > dispersion_limit_ms:
            grade = AnchorGrade.REJECTED
        elif deviation <= hard_band:
            grade = AnchorGrade.HARD
        else:
            grade = AnchorGrade.SOFT
        graded.append(
            GradedAnchor(
                position_ms=int(position),
                offset_ms=float(offset),
                grade=grade,
                deviation_ms=round(deviation, 1),
                implied_reference_ms=int(position - offset),
            )
        )

    hard = [g for g in graded if g.grade is AnchorGrade.HARD]
    soft = [g for g in graded if g.grade is AnchorGrade.SOFT]
    rejected = [g for g in graded if g.grade is AnchorGrade.REJECTED]

    # Prefer the gate's drift measurement. Falling back to a slope through the
    # region medians is only done when the caller has none to offer.
    if drift_ms_per_minute is not None:
        slope = drift_ms_per_minute / 60000.0
    elif len(regions) >= 2:
        xs = [float(p) for p, _ in regions]
        ys = [v for _, v in regions]
        xbar = sum(xs) / len(xs)
        ybar = sum(ys) / len(ys)
        denom = sum((x - xbar) ** 2 for x in xs)
        slope = (
            sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys, strict=False)) / denom
            if denom > 0 else 0.0
        )
    else:
        slope = 0.0

    # Do the trusted anchors agree with that trend? A single bad region is
    # dropped; mutually contradictory ones are the case to abstain on.
    #
    # The test is deliberately NOT the gate's dispersion limit. That limit answers
    # "is this one shift?", which is a question about the *centre*. What matters
    # here is whether the band can still reach the target: the band is
    # SEARCH_WINDOW_MS wide, so anchor scatter beyond that width makes the search
    # unable to cover the timeline no matter how good the centre is. On EVOLV the
    # trusted anchors scatter 2303ms about the drift line -- inside a 4000ms
    # window, and therefore usable, while exceeding the 1200ms dispersion limit
    # purely because that limit answers a different question.
    contradictory = False
    residual_spread = 0.0
    trusted = [g for g in graded if g.grade is not AnchorGrade.REJECTED]
    if len(trusted) >= 3:
        origin = float(trusted[0].position_ms)
        residuals = [
            g.offset_ms - (estimated + slope * (g.position_ms - origin))
            for g in trusted
        ]
        residual_spread = max(residuals) - min(residuals)
        contradictory = residual_spread > float(window_ms)

    return AnchorGrading(
        graded=graded,
        estimated_offset_ms=round(estimated, 1),
        dispersion_ms=round(dispersion, 1),
        slope_ms_per_ms=slope,
        origin_ms=int(graded[0].position_ms),
        hard=hard,
        soft=soft,
        rejected=rejected,
        residual_spread=round(residual_spread, 1),
        contradictory=contradictory,
    )


def graded_offset_fn(grading: AnchorGrading) -> Callable[[float], float]:
    """Local offset from the estimated shift plus its measured drift.

    This is what fixes EVOLV: the anchors show a smooth ramp, and a
    piecewise-constant mapping turned that ramp into three invented breakpoints.
    A single linear term represents it without inventing anything.
    """
    base = grading.estimated_offset_ms
    slope = grading.slope_ms_per_ms
    origin = float(grading.origin_ms)

    def predict(when_ms: float) -> float:
        return base + slope * (when_ms - origin)

    return predict


def align_with_graded_anchors(
    target: Iterable[tuple[int, int, str]],
    reference: Iterable[tuple[int, int, str]],
    hypothesis_ms: float,
    *,
    window_ms: int = SEARCH_WINDOW_MS,
    dispersion_limit_ms: float | None = None,
    hard_fraction: float = 0.5,
) -> IntervalReport:
    """PART 2 variant: the DP constrained by graded anchors.

    Follows consistent anchors, drops isolated bad ones instead of abstaining on
    them, and abstains only when the trusted anchors genuinely contradict one
    another. Uses the production gate's own dispersion limit, not a new one.
    """
    from app.services.sync.alignment import analyze_drift
    from app.services.sync.large_offset import (
        LARGE_OFFSET_MAX_DISPERSION_MS,
        _as_cues,
        _bounded,
        _region_medians,
    )

    t = list(target)
    r = list(reference)
    limit = dispersion_limit_ms if dispersion_limit_ms is not None else (
        LARGE_OFFSET_MAX_DISPERSION_MS
    )

    if not t or not r:
        return IntervalReport(
            outcome=AlignmentOutcome.NO_CORRESPONDENCE,
            groups=[],
            mapping=AnchorMapping([], [], 0, True),
            inconsistent_regions=0,
            target_cues=len(t),
            reference_cues=len(r),
            notes=["empty input"],
        )

    ref_starts = [c[0] for c in r]
    # Mirror the gate's two deterministic passes exactly, over the gate's *bounded*
    # cue list. The gate seeds from the opening disagreement, then re-measures the
    # anchors against the median those anchors produced, so one badly-placed cue
    # cannot decide the outcome -- and the drift it reports comes from the second
    # pass over at most ``LARGE_OFFSET_MAX_SAMPLED_CUES`` cues. Three concrete
    # differences were found between this and a naive single pass over every cue,
    # and each one changes the answer on real fixtures:
    #   * statistic:  gate = median absolute deviation about the median
    #   * drift:      gate = analyze_drift over ~30 samples, not 5 region medians
    #   * bounding:   gate samples at most 600 cues (ASAP has 781)
    # Sampling once from the raw seed gives ASAP -29.4 ms/min where the gate says
    # -10.8, because the bounding changes which cues are sampled.
    hypothesis = float(hypothesis_ms)
    regions: list[tuple[int, float]] = []
    samples: list[tuple[int, float]] = []
    bounded = _bounded(_as_cues(t))
    for _ in range(2):
        regions, _, samples = _region_medians(
            bounded, ref_starts, hypothesis
        )
        if regions:
            hypothesis = _median_of([v for _, v in regions])
    if not regions:
        return IntervalReport(
            outcome=AlignmentOutcome.NO_CORRESPONDENCE,
            groups=[],
            mapping=AnchorMapping([], [], 0, True),
            inconsistent_regions=0,
            target_cues=len(t),
            reference_cues=len(r),
            notes=["no anchor regions: refusing to search without bounds"],
        )

    grading = grade_anchors(
        regions,
        limit,
        hard_fraction,
        drift_ms_per_minute=analyze_drift(samples),
        window_ms=window_ms,
    )
    if grading.contradictory:
        return IntervalReport(
            outcome=AlignmentOutcome.ANCHORS_INCONSISTENT,
            groups=[],
            mapping=AnchorMapping([], [], len(regions), True),
            inconsistent_regions=len(grading.hard),
            target_cues=len(t),
            reference_cues=len(r),
            notes=["trusted anchors contradict a single trend"],
        )

    # The band model is the estimated shift plus its measured drift. A single
    # segment, so no breakpoint can be invented from per-region noise.
    mapping = AnchorMapping(
        segments=[
            MappingSegment(0, 10_000_000_000, grading.estimated_offset_ms)
        ],
        regions=[],
        anchor_samples=len(regions),
        time_map_increasing=True,
    )
    report = _align(
        t,
        r,
        mapping,
        window_ms,
        # A single trusted segment cannot disagree with itself, so the local
        # piecewise-consistency check is satisfied by construction. The gate that
        # matters here is the grading above.
        1500.0,
        offset_fn=graded_offset_fn(grading),
    )
    report.anchor_grading = grading
    report.notes.append(
        f"anchors hard={len(grading.hard)} soft={len(grading.soft)} "
        f"rejected={len(grading.rejected)} "
        f"dispersion={grading.dispersion_ms:.0f}ms "
        f"slope={grading.slope_ms_per_ms * 60000:.2f}ms/min"
    )
    return report


# --------------------------------------------------------------------------- #
# Group feasibility
# --------------------------------------------------------------------------- #


def _group_cost(
    t_lo: int,
    t_hi: int,
    r_lo: int,
    r_hi: int,
    target: list[tuple[int, int, str]],
    reference: list[tuple[int, int, str]],
    offset_fn: Callable[[float], float],
    window_ms: int,
) -> tuple[float, float, tuple[float, ...]] | None:
    """Cost of pairing a target run with a reference run, or ``None`` if invalid.

    Returns ``(cost, centre_delta, per_cue_interval_residuals)``.
    """
    kt = t_hi - t_lo + 1
    kr = r_hi - r_lo + 1
    if kt < 1 or kr < 1:
        return None
    # Both group limits are hard. Unlimited grouping is the failure mode that
    # lets duplication masquerade as segmentation.
    if kt > MAX_GROUP or kr > MAX_GROUP:
        return None

    t_start = target[t_lo][0]
    t_end = target[t_hi][1]
    r_start = reference[r_lo][0]
    r_end = reference[r_hi][1]
    if _span(t_start, t_end) > GROUP_SPAN_MS or _span(r_start, r_end) > GROUP_SPAN_MS:
        return None

    offset = offset_fn(_centre(t_start, t_end))
    predicted = _centre(t_start, t_end) - offset
    centre_delta = _centre(r_start, r_end) - predicted

    # Hard anchor bound: the group's centre must sit inside the band.
    if abs(centre_delta) > window_ms:
        return None
    # Hard ceiling: a group that needs more than this is not a correspondence.
    if abs(centre_delta) > MAX_GROUP_CENTER_MS:
        return None

    # Interval compatibility: after alignment the spans must genuinely overlap,
    # otherwise these are merely two unrelated stretches that happen to be near.
    aligned_start = t_start - offset
    aligned_end = t_end - offset
    overlap = min(aligned_end, r_end) - max(aligned_start, r_start)
    smaller = min(_span(aligned_start, aligned_end), _span(r_start, r_end))
    if smaller > 0 and overlap / smaller < MIN_GROUP_OVERLAP_RATIO:
        return None

    interval_residuals: list[float] = []
    for i in range(t_lo, t_hi + 1):
        s, e, _ = target[i]
        p = _centre(s, e) - offset
        if r_start <= p <= r_end:
            interval_residuals.append(0.0)
        else:
            interval_residuals.append(min(abs(p - r_start), abs(p - r_end)))

    extra = (kt - 1) + (kr - 1)
    cost = (
        MATCH_BASE_COST
        + MATCH_COST_PER_MS * abs(centre_delta)
        + GROUP_EXTRA_COST * extra
    )
    return cost, centre_delta, tuple(interval_residuals)


# --------------------------------------------------------------------------- #
# The alignment
# --------------------------------------------------------------------------- #


def align_intervals(
    target: Iterable[tuple[int, int, str]],
    reference: Iterable[tuple[int, int, str]],
    hypothesis_ms: float,
    *,
    region_count: int | None = None,
    window_ms: int = SEARCH_WINDOW_MS,
    agreement_ms: float | None = None,
) -> IntervalReport:
    """Anchor-banded monotonic alignment over cue intervals.

    ``hypothesis_ms`` is the accepted large-offset estimate. Anchor regions are
    collected with the existing production helper -- no second anchor detector.
    """
    t = list(target)
    r = list(reference)
    if not t or not r:
        return IntervalReport(
            outcome=AlignmentOutcome.NO_CORRESPONDENCE,
            groups=[],
            mapping=AnchorMapping([], [], 0, True),
            inconsistent_regions=0,
            target_cues=len(t),
            reference_cues=len(r),
        )

    from app.services.sync.anchor_correspondence import ANCHOR_REGION_AGREEMENT_MS
    from app.services.sync.large_offset import (
        LARGE_OFFSET_ANCHOR_TOLERANCE_MS,
        LARGE_OFFSET_CUES_PER_REGION,
        LARGE_OFFSET_REGION_COUNT,
    )

    regions: list[AnchorRegion] = collect_anchor_regions(
        t,
        [c[0] for c in r],
        hypothesis_ms,
        region_count=region_count or LARGE_OFFSET_REGION_COUNT,
        cues_per_region=LARGE_OFFSET_CUES_PER_REGION,
        tolerance_ms=LARGE_OFFSET_ANCHOR_TOLERANCE_MS,
    )
    mapping = build_mapping(regions)
    agree = agreement_ms if agreement_ms is not None else ANCHOR_REGION_AGREEMENT_MS

    if not regions:
        # No anchor evidence at all. Refuse rather than search freely.
        return IntervalReport(
            outcome=AlignmentOutcome.NO_CORRESPONDENCE,
            groups=[],
            mapping=mapping,
            inconsistent_regions=0,
            target_cues=len(t),
            reference_cues=len(r),
            notes=["no anchor regions: refusing to search without bounds"],
        )

    consistent, bad = anchors_consistent(mapping, agree)
    if not consistent:
        return IntervalReport(
            outcome=AlignmentOutcome.ANCHORS_INCONSISTENT,
            groups=[],
            mapping=mapping,
            inconsistent_regions=bad,
            target_cues=len(t),
            reference_cues=len(r),
            notes=[f"{bad} anchor region boundary/boundaries disagree"],
        )

    return _align(t, r, mapping, window_ms, agree)


def _align(
    t: list[tuple[int, int, str]],
    r: list[tuple[int, int, str]],
    mapping: AnchorMapping,
    window_ms: int,
    agree: float,
    offset_fn: Callable[[float], float] | None = None,
) -> IntervalReport:
    """The alignment itself.

    The anchor-consistency check lives here rather than in the caller so that
    *every* path into the DP is gated by it. A search band built from anchors the
    model does not trust would silently propagate the disagreement.

    ``offset_fn`` lets a caller supply a different local-offset model -- a fitted
    drift ramp, for instance -- without teaching the DP anything new. It defaults
    to the piecewise-constant mapping.
    """
    consistent, bad = anchors_consistent(mapping, agree)
    if not consistent:
        return IntervalReport(
            outcome=AlignmentOutcome.ANCHORS_INCONSISTENT,
            groups=[],
            mapping=mapping,
            inconsistent_regions=bad,
            target_cues=len(t),
            reference_cues=len(r),
            notes=[f"{bad} anchor region boundary/boundaries disagree"],
        )

    nt, nr = len(t), len(r)
    predict: Callable[[float], float] = (
        offset_fn if offset_fn is not None
        else (lambda when: offset_at(mapping, when))
    )

    # Reference centres, for band lookup.
    ref_centres = [_centre(c[0], c[1]) for c in r]

    # Precompute the admissible reference index window per target index. This is
    # the anchor bound made explicit: a reference cue outside it is never scored.
    band: list[tuple[int, int]] = []
    for i in range(nt):
        c = _centre(t[i][0], t[i][1])
        p = c - predict(c)
        lo = bisect.bisect_left(ref_centres, p - window_ms)
        hi = bisect.bisect_right(ref_centres, p + window_ms)
        band.append((lo, hi))

    # Forward DP. cost[i][j] = best cost having consumed target[:i], reference[:j].
    cost = [[IMPASSABLE] * (nr + 1) for _ in range(nt + 1)]
    back: list[list[tuple | None]] = [[None] * (nr + 1) for _ in range(nt + 1)]
    cost[0][0] = 0.0

    for i in range(nt + 1):
        row = cost[i]
        for j in range(nr + 1):
            here = row[j]
            if here == IMPASSABLE:
                continue
            lo, hi = (band[i] if i < nt else (0, nr))
            # --- skip a reference cue: it has no target partner yet --------- #
            if j < nr and j + 1 <= hi:
                nxt = here + SKIP_REFERENCE_COST
                if nxt < cost[i][j + 1]:
                    cost[i][j + 1] = nxt
                    back[i][j + 1] = ("skip_ref", i, j)
            # --- skip a target cue: it has no reference partner ------------- #
            if i < nt and lo <= j <= hi:
                nxt = here + SKIP_TARGET_COST
                if nxt < cost[i + 1][j]:
                    cost[i + 1][j] = nxt
                    back[i + 1][j] = ("skip_target", i, j)
            # --- pair a run of target cues with a run of reference cues ------ #
            if i >= nt or j >= nr:
                continue
            if not (lo <= j <= hi):
                continue
            for kt in range(1, MAX_GROUP + 1):
                t_hi = i + kt - 1
                if t_hi >= nt:
                    break
                for kr in range(1, MAX_GROUP + 1):
                    r_hi = j + kr - 1
                    if r_hi >= nr:
                        break
                    # every reference cue in the run must be inside the band
                    if r_hi > hi:
                        break
                    scored = _group_cost(i, t_hi, j, r_hi, t, r, predict, window_ms)
                    if scored is None:
                        continue
                    gcost = scored[0]
                    nxt = here + gcost
                    if nxt < cost[i + kt][r_hi + 1]:
                        cost[i + kt][r_hi + 1] = nxt
                        back[i + kt][r_hi + 1] = ("group", i, j, kt, kr)

    end = cost[nt][nr]
    if end == IMPASSABLE:
        # Allow trailing skips so a partial alignment is still reportable.
        best_j = min(range(nr + 1), key=lambda j: cost[nt][j] + (nr - j) * SKIP_REFERENCE_COST)
        best_i = min(range(nt + 1), key=lambda i: cost[i][best_j] + (nt - i) * SKIP_TARGET_COST)
        if cost[best_i][best_j] == IMPASSABLE:
            return IntervalReport(
                outcome=AlignmentOutcome.NO_CORRESPONDENCE,
                groups=[],
                mapping=mapping,
                inconsistent_regions=0,
                target_cues=nt,
                reference_cues=nr,
                notes=["no admissible monotone alignment"],
            )
        end = cost[best_i][best_j]
        back[best_i][best_j] = None  # type: ignore[assignment]

    # ---- backtrack --------------------------------------------------------- #
    groups: list[Group] = []
    skipped_target: list[int] = []
    skipped_reference: list[int] = []
    i, j = nt, nr
    while (i, j) != (0, 0) and back[i][j] is not None:
        step = back[i][j]
        assert step is not None
        kind = step[0]
        if kind == "skip_ref":
            skipped_reference.append(step[2] - 1)
            i, j = step[1], step[2]
        elif kind == "skip_target":
            skipped_target.append(step[1] - 1)
            i, j = step[1], step[2]
        else:
            _, pi, pj, kt, kr = step
            scored = _group_cost(
                pi, pi + kt - 1, pj, pj + kr - 1, t, r, predict, window_ms
            )
            if scored is None:
                raise AssertionError("backtracked group is no longer feasible")
            _, centre_delta, interval_residuals = scored
            if kt == 1 and kr == 1:
                gk = GroupKind.ONE_TO_ONE
            elif kt > 1 and kr == 1:
                gk = GroupKind.SPLIT
            elif kt == 1 and kr > 1:
                gk = GroupKind.MERGE
            else:
                gk = GroupKind.SPLIT_MERGE
            groups.append(
                Group(
                    target_indices=tuple(range(pi, pi + kt)),
                    reference_indices=tuple(range(pj, pj + kr)),
                    center_delta_ms=centre_delta,
                    interval_residuals_ms=interval_residuals,
                    kind=gk,
                )
            )
            i, j = pi, pj

    groups.reverse()

    matched_target: set[int] = set()
    matched_reference: set[int] = set()
    for g in groups:
        matched_target.update(g.target_indices)
        matched_reference.update(g.reference_indices)

    surplus_target = sorted(set(skipped_target) - matched_target)
    surplus_reference = sorted(set(skipped_reference) - matched_reference)

    # Classify each unmatched target cue: did the grouping limit stop a plausible
    # match, or is there genuinely nothing there? Both are worth knowing and they
    # mean different things.
    ambiguous: list[int] = []
    truly_surplus: list[int] = []
    for idx in surplus_target:
        c = _centre(t[idx][0], t[idx][1])
        p = c - predict(c)
        inside = any(
            r[g.reference_indices[0]][0] <= p <= r[g.reference_indices[-1]][1]
            for g in groups
            if g.reference_indices and g.reference_indices[0] <= idx + 1
            and g.reference_indices[-1] >= idx - 1
        )
        (ambiguous if inside else truly_surplus).append(idx)

    report = IntervalReport(
        outcome=AlignmentOutcome.ALIGNED,
        groups=groups,
        mapping=mapping,
        inconsistent_regions=0,
        target_cues=nt,
        reference_cues=nr,
        matched_target=len(matched_target),
        matched_reference=len(matched_reference),
        ambiguous_target=sorted(ambiguous),
        surplus_target=sorted(truly_surplus),
        surplus_reference=surplus_reference,
        total_cost=end,
    )

    # ---- what did the alignment actually find? ----------------------------- #
    # Prefer ABSTAIN over claiming a clean alignment. The dominant unexplained
    # side is named, so a damaged or mismatched target cannot quietly present as
    # "aligned" just because the cues that survived do line up.
    report.outcome = _classify(report)

    # ---- completeness: observed, separate, never part of the timing outcome - #
    t_active = sum(max(0, e - s) for s, e, _ in t)
    r_active = sum(max(0, e - s) for s, e, _ in r)
    matched_ref_span = sum(
        max(0, r[g.reference_indices[-1]][1] - r[g.reference_indices[0]][0])
        for g in groups
    )
    report.completeness = {
        "cue_count_ratio": round(nt / nr, 4) if nr else 0.0,
        "active_duration_ratio": round(t_active / r_active, 4) if r_active else 0.0,
        "reference_coverage": round(report.reference_coverage, 4),
        "target_coverage": round(report.target_coverage, 4),
        "matched_reference_span_ratio": (
            round(matched_ref_span / r_active, 4) if r_active else 0.0
        ),
        "first_cue_delta_ms": (
            float(t[0][0] - r[0][0]) if nt and nr else float("inf")
        ),
        "last_cue_delta_ms": (
            float(t[-1][1] - r[-1][1]) if nt and nr else float("inf")
        ),
    }
    # Explicitly not timing: these are the signals that expose content excess or
    # loss. Timing correspondence cannot see duplication, so it is reported here
    # rather than folded into the outcome.
    report.content = {
        "target_active_ms": t_active,
        "reference_active_ms": r_active,
        "surplus_target_cues": len(truly_surplus),
        "surplus_reference_cues": len(surplus_reference),
        "ambiguous_target_cues": len(ambiguous),
        "split_groups": sum(1 for g in groups if g.kind is GroupKind.SPLIT),
        "merge_groups": sum(1 for g in groups if g.kind is GroupKind.MERGE),
    }
    report.content_observation = _observe_content(report)

    report.notes.append(
        f"median target duration {statistics.median(e - s for s, e, _ in t):.0f}ms "
        f"vs reference {statistics.median(e - s for s, e, _ in r):.0f}ms"
    )
    return report


def _observe_content(report: IntervalReport) -> ContentObservation:
    """Observed content volume, reported separately from the timing outcome.

    Total speech time alone is not enough. Coarser segmentation makes displays
    longer, so speech time *rises* while the cue count *falls* -- which is a
    legitimate segmentation difference, not duplication. Duplication raises
    both. So the two ratios are read together:

    * both up   -> the target carries more content (duplication, or a genuinely
      different release with more dialogue; timing cannot tell these apart)
    * both down -> the target carries less content
    * mixed     -> a segmentation difference, reported as comparable
    """
    act = report.completeness.get("active_duration_ratio", 0.0)
    cue = report.completeness.get("cue_count_ratio", 0.0)
    tol = CONTENT_VOLUME_TOLERANCE
    if act > 1.0 + tol and cue > 1.0 + tol:
        return ContentObservation.TARGET_CONTENT_EXCESS
    if act < 1.0 - tol and cue < 1.0 - tol:
        return ContentObservation.TARGET_CONTENT_LOSS
    return ContentObservation.CONTENT_COMPARABLE


def _classify(report: IntervalReport) -> AlignmentOutcome:
    """Name the dominant explanation. Prefers abstention over a clean claim."""
    if not report.groups or report.matched_reference == 0:
        return AlignmentOutcome.NO_CORRESPONDENCE

    surplus_t = len(report.surplus_target) / max(1, report.target_cues)
    surplus_r = len(report.surplus_reference) / max(1, report.reference_cues)
    ambiguous = len(report.ambiguous_target) / max(1, report.target_cues)

    if ambiguous >= SURPLUS_MIN_FRACTION:
        return AlignmentOutcome.AMBIGUOUS
    if surplus_t >= SURPLUS_MIN_FRACTION and surplus_t > surplus_r:
        return AlignmentOutcome.SURPLUS_TARGET
    if surplus_r >= SURPLUS_MIN_FRACTION and surplus_r > surplus_t:
        return AlignmentOutcome.SURPLUS_REFERENCE
    if report.reference_coverage < ALIGNED_MIN_REFERENCE_COVERAGE:
        return AlignmentOutcome.LOW_REFERENCE_COVERAGE
    return AlignmentOutcome.ALIGNED
