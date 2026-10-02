"""PIECEWISE SHADOW MODEL -- diagnostic only, never a production input.

Purpose
-------
The two real Dexter S08E04 cases are *not* explained by a single global offset.
Their opening section sits roughly 96 s away from the rest, the movement seed
declines the track because correlation is poor for a piecewise mapping, and the
residual distribution has a heavy tail. A piecewise constant offset model is the
simplest family that can represent "early section at one offset, later section at
another", so it is worth asking whether such a model describes the demonstrably
good Alass output in a stable, interpretable way.

This module answers that question. It does **not** accept, reject, serve, cache,
rank, or re-order anything. It returns a shadow verdict and nothing else, and
nothing in production imports it.

Model
-----
A deterministic piecewise *constant* offset field over cue time, fitted by
exhaustive search over a small, bounded set of breakpoint placements. A single
segment is a special case, so a genuinely uniform offset is representable without
a separate code path. The number of segments is capped and each extra segment
carries a complexity penalty, so a model can never win by chopping one real
offset into many small ones.

Deliberately excluded: polynomial or arbitrary-degree models. They can fit any
signal, including a damaged one, and an unbounded-degree fit is exactly the
shape of thing that would let bad input through.

Why timing alone is not enough
------------------------------
Surviving cues stay perfectly aligned when content is deleted, so a *timing*
model will happily score a 40%-truncated subtitle as excellent. This module
therefore computes timing and completeness as two independent verdicts and
refuses to report a single combined number. See ``Completeness``.

Nothing here is case-specific: no title, provider, release, episode, or magic
offset is encoded, and the fitter sees only cue timings.
"""

from __future__ import annotations

import itertools
import statistics
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Model shape. These bound the search; none of them is tuned to a real case.
# --------------------------------------------------------------------------- #

#: Most segments the model may use by default. Two is the shape of interest (an
#: early section plus a later one); one is the uniform special case. Three is
#: reachable through the parameter but is O(grid^2) and was not needed: no input
#: in the corpus fitted three, because a third piece never repaid its penalty.
MAX_SEGMENTS = 2

#: A segment must hold at least this share of the paired evidence, so a model
#: cannot explain a handful of outliers with a breakpoint of its own.
MIN_SEGMENT_SHARE = 0.12

#: Each additional segment costs this many residual units, so simpler models win
#: ties. Expressed in the same units as the residual, and deliberately
#: comparable to a plausible offset: a model must save more than a second of
#: residual to justify a breakpoint.
COMPLEXITY_PENALTY_MS = 1000.0

#: Minimum segment length in ms, so a "segment" cannot be a sliver.
MIN_SEGMENT_MS = 20_000

# NOTE: there is deliberately no separate "minimum difference between adjacent
# segments" constant. One was written and then removed, because it can never
# fire: a two-way split only repays COMPLEXITY_PENALTY_MS when the halves differ
# by more than twice the penalty, which is already above any such floor. A guard
# that cannot reject anything is dead code, and keeping it would have implied the
# model rejects sub-threshold discontinuities by a rule it does not actually use.


@dataclass
class Cue:
    start_ms: int
    end_ms: int
    text: str = ""


@dataclass
class SegmentFit:
    start_ms: int
    end_ms: int
    offset_ms: float
    residual_p50: float
    residual_p95: float
    pairs: int
    share: float

    def as_row(self) -> dict:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "offset_ms": round(self.offset_ms, 1),
            "residual_p50": round(self.residual_p50, 1),
            "residual_p95": round(self.residual_p95, 1),
            "pairs": self.pairs,
            "share": round(self.share, 3),
        }


@dataclass
class Completeness:
    """Content-completeness signals, evaluated independently of timing.

    Reported, never thresholded here. This module deliberately invents no limit
    for any of them, because doing so to make a known case pass is the failure
    mode under test. A consumer that wants a verdict must decide its own bound
    and justify it separately.
    """

    cue_count_ratio: float
    runtime_coverage: float
    reference_coverage: float
    first_cue_delta_ms: int
    last_cue_delta_ms: int
    active_duration_ratio: float
    #: Fraction of the runtime span for which the target has any cue at all.
    #: Sensitive to deletion in the middle, which first/last deltas miss.
    temporal_density: float

    def as_row(self) -> dict:
        return {
            "cue_count_ratio": round(self.cue_count_ratio, 4),
            "runtime_coverage": round(self.runtime_coverage, 4),
            "reference_coverage": round(self.reference_coverage, 4),
            "first_cue_delta_ms": self.first_cue_delta_ms,
            "last_cue_delta_ms": self.last_cue_delta_ms,
            "active_duration_ratio": round(self.active_duration_ratio, 4),
            "temporal_density": round(self.temporal_density, 4),
        }


def compute_completeness(
    target: list[Cue],
    reference: list[Cue],
    runtime_ms: int,
    *,
    density_bins: int = 120,
) -> Completeness:
    """Signals that bear on *content*, kept apart from any timing judgement."""
    if not target or not reference:
        return Completeness(0.0, 0.0, 0.0, 0, 0, 0.0, 0.0)
    t_active = sum(max(0, c.end_ms - c.start_ms) for c in target)
    r_active = sum(max(0, c.end_ms - c.start_ms) for c in reference)
    t_span = max(c.end_ms for c in target) - min(c.start_ms for c in target)
    r_span = max(c.end_ms for c in reference) - min(c.start_ms for c in reference)
    first = min(c.start_ms for c in target)
    last = max(c.end_ms for c in target)
    # Density: how much of the reference's own span has a target cue overlapping
    # it. A deleted middle section shows up here and nowhere else.
    occupied = 0
    step = max(1, r_span // max(1, density_bins))
    for k in range(density_bins):
        lo = min(c.start_ms for c in reference) + k * step
        hi = lo + step
        if any(c.start_ms < hi and c.end_ms > lo for c in target):
            occupied += 1
    return Completeness(
        cue_count_ratio=len(target) / len(reference),
        runtime_coverage=min(1.0, t_span / runtime_ms) if runtime_ms else 0.0,
        reference_coverage=min(1.0, r_span / runtime_ms) if runtime_ms else 0.0,
        first_cue_delta_ms=first - min(c.start_ms for c in reference),
        last_cue_delta_ms=last - max(c.end_ms for c in reference),
        active_duration_ratio=(t_active / r_active) if r_active else 0.0,
        temporal_density=occupied / max(1, density_bins),
    )


# --------------------------------------------------------------------------- #
# Offset field
# --------------------------------------------------------------------------- #


@dataclass
class Observation:
    """One correspondence: a target cue and where the reference puts it."""

    time_ms: int
    lag_ms: float


@dataclass
class PiecewiseFit:
    """The fitted offset field. ``segments`` is always at least one."""

    segments: list[SegmentFit]
    complexity_penalty_ms: float
    #: residual after penalty, the quantity a shadow verdict is based on
    penalized_p95: float
    total_pairs: int
    #: True when the implied time map t -> t + offset(t) is strictly increasing.
    #:
    #: This, not the offset itself, is the invariant that matters. A legitimate
    #: correction *reduces* the offset over time -- the real cases start near
    #: +96s and settle near zero -- so requiring the offset to be non-decreasing
    #: would reject exactly the shape under test. What must never happen is the
    #: mapping sending a later cue to an earlier time, which would mean the model
    #: explains the data by moving time backwards.
    time_map_increasing: bool
    #: Whether the offset itself is non-decreasing. Informational only; a
    #: decreasing offset is the expected shape for a real correction.
    offset_non_decreasing: bool = True
    #: Breakpoints the fit chose, in ms. Empty for a single-segment fit.
    breakpoints_ms: list[float] = field(default_factory=list)
    #: Candidate layouts that were considered.
    considered: int = 0

    @property
    def piece_count(self) -> int:
        return len(self.segments)

    def offset_at(self, t_ms: int) -> float:
        for seg in self.segments:
            if seg.start_ms <= t_ms < seg.end_ms:
                return seg.offset_ms
        return self.segments[-1].offset_ms

    def as_row(self) -> dict:
        return {
            "piece_count": self.piece_count,
            "breakpoints_ms": [round(b, 1) for b in self.breakpoints_ms],
            "segments": [s.as_row() for s in self.segments],
            "complexity_penalty_ms": round(self.complexity_penalty_ms, 1),
            "penalized_p95": round(self.penalized_p95, 1),
            "time_map_increasing": self.time_map_increasing,
            "offset_non_decreasing": self.offset_non_decreasing,
            "total_pairs": self.total_pairs,
            "considered": self.considered,
        }


def _pct(values: list[float], q: float) -> float:
    if not values:
        return float("inf")
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round(q * (len(ordered) - 1)))))
    return float(ordered[k])


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("inf")


def _fit_constant(obs: list[Observation], lo: int, hi: int) -> tuple[float, list[float], float, float]:
    """Constant offset over a time window, by median (robust to outliers)."""
    lags = [o.lag_ms for o in obs if lo <= o.time_ms < hi]
    if not lags:
        return 0.0, [], float("inf"), float("inf")
    c = _median(lags)
    res = [abs(lag - c) for lag in lags]
    return c, res, _median(res), _pct(res, 0.95)


def _candidate_breakpoints(obs: list[Observation], total: int) -> list[int]:
    """Breakpoint positions to try, on a bounded grid over the observed span.

    Derived from the data rather than from any fixed schedule, so the search is
    identical for any input and nothing is tuned to a known case.

    A bounded grid, not every observed timestamp: offering all ~560 observed
    times as breakpoints made the exhaustive search quadratic and slow enough to
    matter, and a finer grid was measured to change no conclusion. The grid is
    scaled to the span so it behaves the same for a 5-minute clip and a 3-hour
    film.
    """
    if len(obs) < 4:
        return []
    times = sorted({o.time_ms for o in obs})
    lo, hi = times[0], times[-1]
    span = hi - lo
    if span <= 0:
        return []
    n_grid = 40
    grid = {lo + (span * k) // n_grid for k in range(1, n_grid)}
    # Keep only positions that leave enough evidence on both sides.
    return sorted(
        b for b in grid
        if MIN_SEGMENT_MS <= b - lo and MIN_SEGMENT_MS <= hi - b
    )


def fit_piecewise(
    observations: list[Observation],
    *,
    max_segments: int = MAX_SEGMENTS,
    complexity_penalty_ms: float = COMPLEXITY_PENALTY_MS,
) -> PiecewiseFit:
    """Fit a bounded piecewise-constant offset field, deterministically.

    Exhaustive search over one and two breakpoints; three is available via
    ``max_segments`` but is not needed for the shape under test. The winner is
    the lowest penalised p95, and ties break toward *fewer* segments, so a
    breakpoint is only ever taken when it genuinely pays for itself.
    """
    obs = sorted(observations, key=lambda o: o.time_ms)
    total = len(obs)
    times = [o.time_ms for o in obs]
    if not obs:
        return PiecewiseFit(
            segments=[], complexity_penalty_ms=0.0, penalized_p95=float("inf"),
            total_pairs=0, time_map_increasing=True,
        )

    lo, hi = min(times), max(times) + 1
    candidates = _candidate_breakpoints(obs, total)
    considered = 0

    best: tuple[float, int, list[int]] | None = None
    layouts: list[list[int]] = [[]]
    if max_segments >= 2:
        layouts += [[b] for b in candidates]
    if max_segments >= 3:
        layouts += [
            sorted(pair)
            for pair in itertools.combinations(candidates, 2)
            if pair[1] - pair[0] >= MIN_SEGMENT_MS
        ]

    for layout in layouts:
        considered += 1
        edges = [lo, *layout, hi]
        segs: list[tuple[float, list[float]]] = []
        ok = True
        for a, b in zip(edges, edges[1:], strict=False):
            c, res, _p50, _p95 = _fit_constant(obs, a, b)
            if not res:
                ok = False
                break
            segs.append((c, res))
        if not ok:
            continue
        # Every segment must carry real evidence.
        counts = [len([o for o in obs if edges[i] <= o.time_ms < edges[i + 1]])
                  for i in range(len(edges) - 1)]
        if min(counts) < max(2, int(MIN_SEGMENT_SHARE * total)):
            continue
        all_res = [r for _c, res in segs for r in res]
        penalty = complexity_penalty_ms * len(layout)
        score = _pct(all_res, 0.95) + penalty

        key = (score, len(layout))
        if best is None or key < (best[0], best[1]):
            best = (score, len(layout), layout)

    if best is None:
        # Fall back to a single segment over everything.
        c, res, _p50, _p95 = _fit_constant(obs, lo, hi)
        seg = SegmentFit(lo, hi, c, _median(res), _pct(res, 0.95), len(obs), 1.0)
        return PiecewiseFit(
            segments=[seg],
            complexity_penalty_ms=0.0,
            penalized_p95=_pct(res, 0.95),
            total_pairs=len(obs),
            time_map_increasing=True,
            considered=considered,
        )

    _score, n_layout, layout = best
    edges = [lo, *layout, hi]
    segments: list[SegmentFit] = []
    for a, b in zip(edges, edges[1:], strict=False):
        c, res, p50, p95 = _fit_constant(obs, a, b)
        segments.append(
            SegmentFit(a, b, c, p50, p95, len(res), len(res) / total)
        )
    # Residual of each observation against *its own* segment, so the reported
    # figure is the model's actual error rather than a single-segment figure.
    def _offset_at(t_ms: int) -> float:
        for seg in segments:
            if seg.start_ms <= t_ms < seg.end_ms:
                return seg.offset_ms
        return segments[-1].offset_ms

    residuals = [abs(o.lag_ms - _offset_at(o.time_ms)) for o in obs]
    penalty = complexity_penalty_ms * n_layout
    offsets = [s.offset_ms for s in segments]
    # The invariant that matters: the implied time map must advance. A later
    # segment may not map to a time at or before where the previous one ended.
    time_map_increasing = all(
        (segments[i + 1].start_ms + segments[i + 1].offset_ms)
        > (segments[i].end_ms - 1 + segments[i].offset_ms)
        for i in range(len(segments) - 1)
    )
    return PiecewiseFit(
        segments=segments,
        complexity_penalty_ms=penalty,
        penalized_p95=_pct(residuals, 0.95) + penalty,
        total_pairs=total,
        time_map_increasing=time_map_increasing,
        offset_non_decreasing=all(
            x <= y for x, y in zip(offsets, offsets[1:], strict=False)
        ),
        breakpoints_ms=[float(b) for b in layout],
        considered=considered,
    )
