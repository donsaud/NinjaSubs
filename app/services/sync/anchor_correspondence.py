"""ANCHOR-GUIDED CORRESPONDENCE -- shadow experiment, diagnostic only.

Purpose
-------
Answer one question:

    "Can the existing large-offset anchor evidence establish reliable
     correspondence neighbourhoods before residual measurement?"

The two real Dexter S08E04 cases carry a large, non-uniform temporal
discontinuity. The previous experiment showed a piecewise offset field *can*
represent that discontinuity when given true correspondences, but that
nearest-start matching cannot discover them: once the offset is large, shifted
target cues collide over the same reference cues and the survivors pair to
distant partners.

The large-offset gate does not have that problem, because it searches near a
*hypothesis* rather than globally. It already runs in production, it already
finds anchors on the real cases, and this module reuses it rather than inventing
a second detector.

What this is not
----------------
Not a verifier change, not an acceptance rule, not a threshold, and not a
production input. It reads cues, returns a report, and writes nothing: no
verdict, no artifact, no alias, no cache entry. No production module imports it.
It never modifies a subtitle or a served artifact.

The mapping is never applied to subtitle bytes. It is applied to *time
coordinates* to decide which reference cue a target cue should be compared
against, which is a correspondence question, not a delivery one.
"""

from __future__ import annotations

import bisect
import statistics
from dataclasses import dataclass, field
from enum import Enum

from app.services.sync.large_offset import (
    LARGE_OFFSET_ANCHOR_TOLERANCE_MS,
    LARGE_OFFSET_CUES_PER_REGION,
    LARGE_OFFSET_REGION_COUNT,
    _nearest_reference_offset,
)

# Model shape. Bounds for the experiment; none is a production threshold and
# none is tuned to a real case.

#: Tolerated disagreement between adjacent region medians before the mapping is
#: allowed to bend. Below this the simplest model -- one constant -- wins.
ANCHOR_REGION_AGREEMENT_MS = 1500.0

#: Radius, around the mapped prediction, in which a reference cue may be found.
#: Deliberately modest: widening it is how a correspondence model starts
#: absorbing corrupted content, so a mutation that widens it must be caught.
NEIGHBOURHOOD_MS = 2500

#: If two admissible reference cues differ by this much in local offset, the cue
#: is ambiguous and is left unmatched rather than guessed.
AMBIGUITY_MS = 2000

#: The local offset model may not bend more than this from the global anchor
#: median. Keeps a noisy region from inventing a region of its own.
REGION_DEVIATION_MS = 8000

#: Below this many accepted correspondences no label is offered.
MIN_CORRESPONDENCES = 25

#: Residual bound for the SHADOW label only. Not a production threshold and not
#: derived from MAX_P95_MS_FOR_STABLE. It exists so that having correspondences
#: cannot be reported as GOOD when the mapping explains them badly.
SHADOW_RESIDUAL_BOUND_MS = 2000.0


class CorrespondenceDecision(str, Enum):
    GOOD = "ANCHOR_CORRESPONDENCE_GOOD"
    BAD = "ANCHOR_CORRESPONDENCE_BAD"
    ABSTAIN = "ANCHOR_CORRESPONDENCE_ABSTAIN"


@dataclass(frozen=True)
class AnchorRegion:
    """One anchor region's evidence, reused from the existing machinery."""

    start_ms: int
    end_ms: int
    offset_ms: float
    samples: int


@dataclass
class MappingSegment:
    start_ms: int
    end_ms: int
    offset_ms: float

    def as_row(self) -> dict:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "offset_ms": round(self.offset_ms, 1),
        }


@dataclass
class AnchorMapping:
    """A monotonic piecewise-constant offset field derived from anchors.

    A single segment means the anchors agreed everywhere, which is the expected
    outcome for a uniform offset and is the *simplest* model that fits. A
    breakpoint is kept only where adjacent regions genuinely disagree, so there
    is no incentive to farm pieces.
    """

    segments: list[MappingSegment]
    regions: list[AnchorRegion]
    anchor_samples: int
    time_map_increasing: bool

    @property
    def piece_count(self) -> int:
        return len(self.segments)

    def offset_at(self, t_ms: int) -> float:
        for seg in self.segments:
            if seg.start_ms <= t_ms < seg.end_ms:
                return seg.offset_ms
        return self.segments[-1].offset_ms if self.segments else 0.0

    @property
    def breakpoints_ms(self) -> list[int]:
        return [s.start_ms for s in self.segments[1:]]

    def as_row(self) -> dict:
        return {
            "piece_count": self.piece_count,
            "breakpoints_ms": self.breakpoints_ms,
            "segments": [s.as_row() for s in self.segments],
            "regions": [
                {"start_ms": r.start_ms, "offset_ms": round(r.offset_ms, 1),
                 "samples": r.samples}
                for r in self.regions
            ],
            "anchor_samples": self.anchor_samples,
            "time_map_increasing": self.time_map_increasing,
        }


@dataclass
class Correspondence:
    target_index: int
    reference_index: int
    target_start_ms: int
    reference_start_ms: int
    residual_ms: float


@dataclass
class CorrespondenceReport:
    decision: CorrespondenceDecision
    mapping: AnchorMapping
    correspondences: list[Correspondence]
    unmatched_target: list[int]
    unmatched_reference: list[int]
    ambiguous_target: list[int]
    target_total: int
    reference_total: int
    coverage: float
    median_gap_ms: float
    reasons: list[str] = field(default_factory=list)
    completeness: dict = field(default_factory=dict)

    def residuals(self) -> list[float]:
        return [abs(c.residual_ms) for c in self.correspondences]

    def _pct(self, q: float) -> float:
        vals = sorted(self.residuals())
        if not vals:
            return float("inf")
        k = max(0, min(len(vals) - 1, int(round(q * (len(vals) - 1)))))
        return float(vals[k])

    def residual_median(self) -> float:
        vals = self.residuals()
        return float(statistics.median(vals)) if vals else float("inf")

    def residual_p95(self) -> float:
        return self._pct(0.95)

    def residual_p99(self) -> float:
        return self._pct(0.99)

    def residual_worst(self) -> float:
        vals = self.residuals()
        return float(max(vals)) if vals else float("inf")

    def as_row(self) -> dict:
        return {
            "decision": self.decision.value,
            "target_cues": self.target_total,
            "reference_cues": self.reference_total,
            "correspondences": len(self.correspondences),
            "unmatched_target": len(self.unmatched_target),
            "unmatched_reference": len(self.unmatched_reference),
            "ambiguous_target": len(self.ambiguous_target),
            "coverage": round(self.coverage, 4),
            "residual_median": round(self.residual_median(), 1),
            "residual_p95": round(self.residual_p95(), 1),
            "residual_p99": round(self.residual_p99(), 1),
            "residual_worst": round(self.residual_worst(), 1),
            "segmentation_median_gap_ms": round(self.median_gap_ms, 1),
            "mapping": self.mapping.as_row(),
            "completeness": self.completeness,
            "reasons": list(self.reasons),
        }


# --------------------------------------------------------------------------- #
# Anchor evidence -- reused, not reinvented
# --------------------------------------------------------------------------- #


def collect_anchor_regions(
    target: list[tuple[int, int, str]],
    reference_starts: list[int],
    hypothesis_ms: float,
    *,
    region_count: int = LARGE_OFFSET_REGION_COUNT,
    cues_per_region: int = LARGE_OFFSET_CUES_PER_REGION,
    tolerance_ms: int = LARGE_OFFSET_ANCHOR_TOLERANCE_MS,
) -> list[AnchorRegion]:
    """Per-region anchor evidence, using the existing hypothesis-guided search.

    Calls the production large-offset helper rather than reimplementing it, so
    the experiment consumes the same evidence the gate already produces. The
    search predicts ``target_start - hypothesis`` and looks near that point,
    which is exactly why it survives a large offset where global nearest-start
    does not.
    """
    if not target or not reference_starts:
        return []
    start = target[0][0]
    span = max(1, target[-1][0] - start)
    region_span = span / region_count
    regions: list[AnchorRegion] = []
    for r in range(region_count):
        lower = start + r * region_span
        upper = start + (r + 1) * region_span
        bucket = [
            c for c in target
            if lower <= c[0] < upper or (r == region_count - 1 and c[0] >= upper)
        ]
        if not bucket:
            continue
        stride = max(1, len(bucket) // cues_per_region)
        offsets: list[float] = []
        for cue in bucket[::stride][:cues_per_region]:
            local = _nearest_reference_offset(
                cue[0], reference_starts, hypothesis_ms, tolerance_ms
            )
            if local is not None:
                offsets.append(local)
        if not offsets:
            continue
        regions.append(
            AnchorRegion(
                start_ms=int(lower),
                end_ms=int(upper),
                offset_ms=float(statistics.median(offsets)),
                samples=len(offsets),
            )
        )
    return regions


def _body_of_evidence(regions: list[AnchorRegion]) -> tuple[float, int]:
    """Largest cluster of mutually agreeing regions, as ``(offset, size)``.

    Returns ``(median, 1)`` when nothing agrees, so the caller can tell "no
    consensus" from "consensus of one region".
    """
    best_offset = statistics.median(r.offset_ms for r in regions)
    best_size = 0
    for candidate in regions:
        size = sum(
            1
            for r in regions
            if abs(r.offset_ms - candidate.offset_ms) <= ANCHOR_REGION_AGREEMENT_MS
        )
        if size > best_size:
            best_offset, best_size = candidate.offset_ms, size
    return best_offset, best_size


def build_mapping(regions: list[AnchorRegion]) -> AnchorMapping:
    """Collapse region evidence into the *simplest* monotonic offset field.

    Adjacent regions whose medians agree within ``ANCHOR_REGION_AGREEMENT_MS``
    are merged, so a uniform offset reduces to one segment and only a genuine
    disagreement produces a breakpoint. A region that deviates from the global
    anchor median by more than ``REGION_DEVIATION_MS`` is treated as noise and
    folded in, rather than being allowed to invent a segment of its own.
    """
    if not regions:
        return AnchorMapping([], [], 0, True)
    # A region may only be folded into a consensus, never into a lone reading.
    # The body of evidence is the largest cluster of agreeing regions; if no
    # cluster has more than one member there is no consensus to appeal to, and
    # the offsets are kept as measured. Without this a genuine two-sided
    # disagreement (two regions 96s apart) was collapsed onto whichever region
    # sorted first, erasing exactly the breakpoint this mapping exists to find.
    body_offset, body_size = _body_of_evidence(regions)
    merged: list[MappingSegment] = []
    for region in regions:
        offset = region.offset_ms
        if body_size > 1 and abs(offset - body_offset) > REGION_DEVIATION_MS:
            # Too far from the agreed body to be a real region of its own.
            offset = body_offset
        if merged and abs(offset - merged[-1].offset_ms) <= ANCHOR_REGION_AGREEMENT_MS:
            prev = merged[-1]
            merged[-1] = MappingSegment(
                start_ms=prev.start_ms,
                end_ms=region.end_ms,
                offset_ms=round((prev.offset_ms + offset) / 2, 1),
            )
        else:
            merged.append(
                MappingSegment(
                    start_ms=region.start_ms, end_ms=region.end_ms, offset_ms=offset
                )
            )
    # The implied time map must advance. Offset here is ``target - reference``,
    # so a mapped time is ``t - offset(t)`` and the comparison subtracts. Adding
    # it -- the sign used where offset means ``reference - target`` -- inverts
    # the test and falsely rejects a legitimate large correction, so the
    # convention is stated explicitly.
    increasing = all(
        (merged[i + 1].start_ms - merged[i + 1].offset_ms)
        > (merged[i].end_ms - 1 - merged[i].offset_ms)
        for i in range(len(merged) - 1)
    )
    return AnchorMapping(
        segments=merged,
        regions=regions,
        anchor_samples=sum(r.samples for r in regions),
        time_map_increasing=increasing,
    )


# --------------------------------------------------------------------------- #
# Correspondence
# --------------------------------------------------------------------------- #


def _pct_of(vals: list[float], q: float) -> float:
    if not vals:
        return float("inf")
    ordered = sorted(vals)
    k = max(0, min(len(ordered) - 1, int(round(q * (len(ordered) - 1)))))
    return float(ordered[k])


def _admissible(
    predicted: float,
    reference_starts: list[int],
    first_allowed: int,
    neighbourhood_ms: int,
) -> list[int]:
    """Reference indices within the neighbourhood, at or after ``first_allowed``.

    Monotonicity is enforced by refusing to look before the previous match, so
    the correspondence can never run backwards even if the mapping is noisy.
    """
    lo = bisect.bisect_left(reference_starts, predicted - neighbourhood_ms)
    hi = bisect.bisect_right(reference_starts, predicted + neighbourhood_ms)
    return [i for i in range(lo, hi) if i >= first_allowed]


def anchor_guided_correspondence(
    target: list[tuple[int, int, str]],
    reference: list[tuple[int, int, str]],
    hypothesis_ms: float,
    *,
    neighbourhood_ms: int = NEIGHBOURHOOD_MS,
    ambiguity_ms: int = AMBIGUITY_MS,
) -> CorrespondenceReport:
    """Build correspondences from anchor evidence and measure their residual.

    For each target cue the mapping predicts a reference time; reference cues in
    that neighbourhood are candidates. A cue whose admissible candidates disagree
    by more than ``ambiguity_ms`` is left *unmatched* rather than guessed, so
    ambiguity produces abstention and never a forced match. Residuals are the
    departures from the local anchor offset, which is what a useful
    correspondence should be able to explain.
    """
    reference_starts = sorted(c[0] for c in reference)
    regions = collect_anchor_regions(target, reference_starts, hypothesis_ms)
    mapping = build_mapping(regions)

    reasons: list[str] = []
    correspondences: list[Correspondence] = []
    unmatched_target: list[int] = []
    ambiguous: list[int] = []
    used_reference: set[int] = set()

    if not mapping.segments:
        reasons.append("no anchor regions could be established at the hypothesis")
        return _finish(
            CorrespondenceDecision.ABSTAIN, mapping, correspondences,
            list(range(len(target))), list(range(len(reference_starts))),
            ambiguous, target, reference, reasons,
        )

    first_allowed = 0
    for idx in sorted(range(len(target)), key=lambda k: target[k][0]):
        t_start = target[idx][0]
        local_offset = mapping.offset_at(t_start)
        predicted = t_start - local_offset
        candidates = [
            c
            for c in _admissible(
                predicted, reference_starts, first_allowed, neighbourhood_ms
            )
            if c not in used_reference
        ]
        if not candidates:
            unmatched_target.append(idx)
            continue
        local_offsets = [t_start - reference_starts[c] for c in candidates]
        if (
            len(candidates) > 1
            and max(local_offsets) - min(local_offsets) > ambiguity_ms
        ):
            ambiguous.append(idx)
            continue
        best = min(candidates, key=lambda c: abs(reference_starts[c] - predicted))
        used_reference.add(best)
        first_allowed = best + 1
        correspondences.append(
            Correspondence(
                target_index=idx,
                reference_index=best,
                target_start_ms=t_start,
                reference_start_ms=reference_starts[best],
                residual_ms=reference_starts[best] - t_start + local_offset,
            )
        )

    unmatched_reference = [
        i for i in range(len(reference_starts)) if i not in used_reference
    ]

    decision, decision_reasons = _decide(mapping, correspondences)
    reasons.extend(decision_reasons)

    return _finish(
        decision, mapping, correspondences, unmatched_target, unmatched_reference,
        ambiguous, target, reference, reasons,
    )


def _decide(
    mapping: AnchorMapping,
    correspondences: list[Correspondence],
) -> tuple[CorrespondenceDecision, list[str]]:
    """Label the evidence. Split out so each branch is reachable by a test.

    A single hypothesis confines cross-region offset spread to the search
    neighbourhood, so a *folding* time map cannot be produced through the public
    entry point. The branch is still correct and worth keeping, but inlined it
    was untestable -- the mutation that removed it was reported UNPROTECTED.
    """
    if len(correspondences) < MIN_CORRESPONDENCES:
        return (
            CorrespondenceDecision.ABSTAIN,
            [f"only {len(correspondences)} correspondences; refusing to label"],
        )
    if not mapping.time_map_increasing:
        return (
            CorrespondenceDecision.ABSTAIN,
            ["implied time map does not advance"],
        )
    # Having correspondences is not the same as having *good* ones. A wrong
    # release, a corrupted timeline and a re-segmented cut all produce plenty of
    # monotone correspondences, so the label must also look at how well the
    # mapping explains them. Without this the model called every damaged fixture
    # GOOD, which is the failure this experiment exists to detect.
    p95 = _pct_of([abs(c.residual_ms) for c in correspondences], 0.95)
    if p95 <= SHADOW_RESIDUAL_BOUND_MS:
        return (
            CorrespondenceDecision.GOOD,
            [
                f"{len(correspondences)} anchor-guided correspondences over "
                f"{mapping.piece_count} mapping piece(s), residual p95 {p95:.0f}ms",
                "TIMING ONLY -- completeness is reported separately and is not "
                "part of this label",
            ],
        )
    return (
        CorrespondenceDecision.BAD,
        [
            f"correspondences exist but residual p95 {p95:.0f}ms exceeds the "
            f"shadow bound ({SHADOW_RESIDUAL_BOUND_MS:.0f}ms)"
        ],
    )


def _finish(
    decision,
    mapping,
    correspondences,
    unmatched_target,
    unmatched_reference,
    ambiguous,
    target,
    reference,
    reasons,
) -> CorrespondenceReport:
    gaps = [
        correspondences[i + 1].target_start_ms - correspondences[i].target_start_ms
        for i in range(len(correspondences) - 1)
    ]
    coverage = len(correspondences) / len(target) if target else 0.0
    return CorrespondenceReport(
        decision=decision,
        mapping=mapping,
        correspondences=correspondences,
        unmatched_target=unmatched_target,
        unmatched_reference=unmatched_reference,
        ambiguous_target=ambiguous,
        target_total=len(target),
        reference_total=len(reference),
        coverage=coverage,
        median_gap_ms=float(statistics.median(gaps)) if gaps else 0.0,
        reasons=reasons,
        completeness=completeness_signals(target, reference),
    )


def completeness_signals(
    target: list[tuple[int, int, str]],
    reference: list[tuple[int, int, str]] | None = None,
    *,
    bins: int = 120,
) -> dict:
    """Content signals, kept apart from any timing judgement.

    Reported, never thresholded. Surviving cues stay aligned when content is
    deleted, so a clean correspondence is not evidence of a complete subtitle,
    and that is why these are reported separately rather than folded into the
    decision above.
    """
    out: dict[str, float] = {}
    if not target:
        return out
    starts = [c[0] for c in target]
    ends = [c[1] for c in target]
    span = max(ends) - min(starts)
    active = sum(max(0, e - s) for s, e in zip(starts, ends, strict=False))
    if reference:
        r_starts = [c[0] for c in reference]
        r_ends = [c[1] for c in reference]
        ref_span = max(r_ends) - min(r_starts)
        ref_active = sum(
            max(0, e - s) for s, e in zip(r_starts, r_ends, strict=False)
        )
        out["cue_count_ratio"] = round(len(target) / len(reference), 4)
        out["active_duration_ratio"] = (
            round(active / ref_active, 4) if ref_active else 0.0
        )
        occupied = 0
        step = max(1, ref_span // max(1, bins)) if ref_span else 1
        ref_lo = min(r_starts)
        for k in range(bins):
            lo = ref_lo + k * step
            hi = lo + step
            if any(s < hi and e > lo for s, e in zip(starts, ends, strict=False)):
                occupied += 1
        out["temporal_density"] = round(occupied / max(1, bins), 4)
    if span > 0:
        out["active_span_ratio"] = round(active / span, 4)
    return out

