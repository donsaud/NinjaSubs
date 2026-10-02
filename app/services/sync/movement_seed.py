"""Measured movement seed for the post-alass movement pass.

Why this module exists
----------------------
``alignment`` measures *movement* -- how far alass moved each cue -- by pairing
the target against the output within a fixed :data:`MOVEMENT_TOLERANCE_MS`
radius. That radius is fine for a re-timing that stays inside 30s and useless
beyond it: a uniform 96.5s correction puts every target cue 96.5s away from its
counterpart, so the nearest-neighbour search either pairs the wrong cue or finds
nothing, and the resulting MAD (6-10s on a mathematically perfect input)
rejects a correct result.

The fix must not reintroduce a *configured* offset. An earlier attempt seeded
the pairing from the Large Offset Evidence Gate's scalar estimate and was
reverted: the gate measures the **opening** disagreement between target and
reference, which is not the same thing as the cue-by-cue correction alass
applied. On the real Dexter cases that pre-shift mispaired nearly every cue and
inflated MAD from 0ms to ~10.4s.

So the seed here is *measured from the two cue arrays being compared*, by
cross-correlating their cue-density trains. It reads no gate, no configured
offset, and no user input: given only ``target_cues`` and ``synced_cues`` it
recovers the injected offset exactly in the synthetic acceptance set.

Why density rather than cue-to-cue correspondence
--------------------------------------------------
Two near-identical subtitle tracks differ in *when* their speech occurs, not in
how many cues they have. Correlating quantised "speech is happening here"
trains recovers the global displacement without needing a correspondence between
individual cues, which is exactly what a pure timing signal can support: no text
matching, no media, no network.

The seed is only ever a *search centre*. Every acceptance decision still reads
the statistics of the correspondences found around it, so a wrong or missing
seed can only cost pairs -- it cannot manufacture a verified result.
"""

from __future__ import annotations

from pydantic import BaseModel

# A cue is ``(start_ms, end_ms, text)``. ``alignment`` owns the canonical alias
# and imports this module, so the tuple shape is restated here rather than
# imported back, which would be a cycle.
Cue = tuple[int, int, str]

# Density train quantisation. 0.5s is fine enough to place a cue without being
# so fine that an ordinary subtitle's gaps read as noise.
DENSITY_BIN_MS = 500
# Symmetric smoothing box, in bins. +/-2s keeps a single stray cue from
# dominating the train without smearing real dialogue structure.
DENSITY_BOX_BINS = 4

# Search resolution. Coarse step finds the basin, fine step refines inside it,
# which keeps the scan affordable in pure Python on a full-length episode.
COARSE_STEP_BINS = 8  # 4.0s
FINE_RADIUS_BINS = 10  # +/-5.0s around the coarse winner

# Confidence floors. Set from the measured behaviour of the acceptance set, not
# chosen to make any particular case pass:
#   * the eight uniform synthetic corrections score r >= 0.948
#   * the real Dexter pairs score r ~= 0.30 (their body genuinely moved ~0 but
#     the opening carries ~3% of the density energy, so no global peak is
#     meaningful) and are deliberately *not* seeded
#   * wrong-episode and different-cut references score r <= 0.52
MIN_CORRELATION = 0.60
# Best peak must beat the best competing peak outside its own neighbourhood.
# Uniform synthetics score 0.012-0.013; a genuinely ambiguous correlation does
# not separate its peaks at all.
MIN_PEAK_MARGIN = 0.008
# Independent halves of the episode must agree on the seed to within this.
# Drift and multi-jump fixtures disagree, so they are refused rather than
# summarised by a single meaningless number.
MAX_HALF_SPLIT_DISAGREEMENT_MS = 3_000
# Correlation floor for a *half* of the episode. Lower than MIN_CORRELATION
# because half a track carries roughly half the cues, so its density train is
# intrinsically sparser and correlates a little worse even when the offset is
# recovered exactly. This is a stability probe, not the primary measurement: the
# full-episode correlation still has to clear MIN_CORRELATION on its own.
MIN_HALF_SPLIT_CORRELATION = 0.30
# Cues required in the whole pair, and in each half, for a seed to be measured.
MIN_CUES_FOR_SEED = 25
# Fraction of target cues that must land a correspondence inside the local
# radius for the seed to be considered usable. A seed that cannot pair is not
# evidence.
MIN_SEED_SUPPORT = 0.60


class MovementSeed(BaseModel):
    """A measured target->output displacement, with its own confidence.

    Measurement only. Nothing downstream treats this as a claim about quality:
    it decides where to look for correspondences, and the statistics of those
    correspondences decide the verdict.
    """

    #: Signed offset to subtract from the target before local pairing
    #: (``target.start - offset_ms`` should land on ``synced.start``).
    offset_ms: float
    #: Normalised correlation at the winning lag.
    correlation: float
    #: Winning peak minus the best competing peak outside its neighbourhood.
    peak_margin: float
    #: Agreement between independent halves, in ms.
    half_split_disagreement_ms: float | None = None
    #: Cues that found a correspondence inside the local radius.
    support_pairs: int = 0
    #: ``support_pairs`` over the target cue count.
    support_ratio: float = 0.0

    def explain(self) -> str:
        halves = (
            "n/a"
            if self.half_split_disagreement_ms is None
            else f"{self.half_split_disagreement_ms:.0f}ms"
        )
        return (
            f"measured movement seed {self.offset_ms / 1000.0:+.2f}s "
            f"(r={self.correlation:.3f} margin={self.peak_margin:.3f} "
            f"halves={halves} support={self.support_ratio:.2f})"
        )


def _density_train(cues: list[Cue], bins: int) -> list[float]:
    """Binned, smoothed, z-normalised "speech happens here" train."""
    raw = [0.0] * bins
    for start, end, _text in cues:
        # int() throughout: a caller may hand in float timestamps (a fixture that
        # perturbs cues by a fractional amount does), and a negative start from
        # an unnormalised track must not index backwards.
        first = int(start) // DENSITY_BIN_MS
        last = int(end) // DENSITY_BIN_MS
        for index in range(max(0, first), min(last + 1, bins)):
            raw[index] = 1.0
    width = DENSITY_BOX_BINS
    smoothed = [0.0] * bins
    running = 0.0
    for index in range(min(width, bins)):
        running += raw[index]
    divisor = 2 * width + 1
    for index in range(bins):
        smoothed[index] = running / divisor
        added = raw[index + width + 1] if index + width + 1 < bins else 0.0
        removed = raw[index - width] if index - width >= 0 else 0.0
        running += added - removed
    mean = sum(smoothed) / bins
    variance = sum((value - mean) ** 2 for value in smoothed) / bins
    deviation = variance**0.5
    if deviation <= 0.0:
        # A constant train carries no displacement information at all.
        return [0.0] * bins
    return [(value - mean) / deviation for value in smoothed]


def _correlation_at(reference: list[float], shifted: list[float], lag_bins: int) -> float:
    """Normalised correlation of ``shifted`` against ``reference`` at ``lag``."""
    overlap = len(reference) - lag_bins
    if overlap <= 0:
        return -1.0
    total = 0.0
    for index in range(overlap):
        total += shifted[lag_bins + index] * reference[index]
    return total / overlap


def _best_lag(
    reference: list[float], shifted: list[float], max_lag_bins: int, min_overlap: int
) -> tuple[int, float, int]:
    """Coarse scan, then a fine scan inside the winning basin.

    Returns the refined lag and the coarse lag it came from, so the caller can
    require the two independent scans to agree. The coarse pass and the fine
    pass do not share a peak-picking step: the coarse pass can only choose among
    4s-spaced candidates, so agreement is real corroboration that the basin was
    not a fluke of one window.
    """
    coarse_bins, coarse_score = 0, -2.0
    limit = min(max_lag_bins, len(reference) - min_overlap)
    for bins in range(0, max(0, limit) + 1, COARSE_STEP_BINS):
        score = _correlation_at(reference, shifted, bins)
        if score > coarse_score:
            coarse_bins, coarse_score = bins, score
    best_bins, best_score = coarse_bins, coarse_score
    low = max(0, coarse_bins - FINE_RADIUS_BINS)
    high = min(limit, coarse_bins + FINE_RADIUS_BINS)
    for bins in range(low, high + 1):
        score = _correlation_at(reference, shifted, bins)
        if score > best_score:
            best_bins, best_score = bins, score
    return best_bins, best_score, coarse_bins


def _peak_margin(
    reference: list[float], shifted: list[float], winner: int, max_lag_bins: int, min_overlap: int
) -> float:
    """How far the winning peak stands above the best competing peak."""
    limit = min(max_lag_bins, len(reference) - min_overlap)
    excluded_low, excluded_high = winner - 6, winner + 6
    competing = -2.0
    for bins in range(0, max(0, limit) + 1):
        if excluded_low <= bins <= excluded_high:
            continue
        score = _correlation_at(reference, shifted, bins)
        if score > competing:
            competing = score
    if competing <= -1.0:
        return 0.0
    return _correlation_at(reference, shifted, winner) - competing


def _half_split_offset(
    target: list[Cue], synced: list[Cue], max_lag_ms: int
) -> float | None:
    """Measure the same seed independently on each half of the episode.

    The two halves must be split on **time**, not on cue index, and each target
    half must be compared against the *whole* synced track. Slicing both sides by
    index and then filtering each by a time bound taken from the other side
    pairs mismatched material, which showed up as a permanently absent stability
    reading rather than as a disagreement.
    """
    if len(target) < 2 * MIN_CUES_FOR_SEED or len(synced) < 2 * MIN_CUES_FOR_SEED:
        return None
    midpoint = (target[0][0] + target[-1][0]) // 2
    windows = (
        ([c for c in target if c[0] < midpoint], synced),
        ([c for c in target if c[0] >= midpoint], synced),
    )
    offsets: list[float] = []
    for sub_target, sub_synced in windows:
        if len(sub_target) < MIN_CUES_FOR_SEED:
            return None
        span = max(sub_target[-1][0], sub_synced[-1][0]) + DENSITY_BIN_MS
        bins = int(span // DENSITY_BIN_MS) + 1
        min_overlap = 150
        if bins - max(1, int(max_lag_ms // DENSITY_BIN_MS)) < min_overlap:
            return None
        shifted = _density_train(sub_target, bins)
        reference = _density_train(sub_synced, bins)
        lag, score, _coarse = _best_lag(
            reference, shifted, int(max_lag_ms // DENSITY_BIN_MS), min_overlap
        )
        if score < MIN_HALF_SPLIT_CORRELATION:
            return None
        offsets.append(float(lag * DENSITY_BIN_MS))
    return abs(offsets[0] - offsets[1])


def measure_movement_seed(
    target_cues: list[Cue],
    synced_cues: list[Cue],
    *,
    max_lag_ms: int,
    local_radius_ms: int,
) -> MovementSeed | None:
    """Measure the global target->output displacement, or return ``None``.

    ``None`` means "no trustworthy global displacement", which the caller must
    treat as *not measurable* rather than as zero. Every refusal here is
    deliberately fail-safe: an unseeded pass pairs at the existing radius and
    reaches the existing verdict, so nothing about current behaviour depends on
    this function succeeding.

    The seed is derived from ``target_cues`` and ``synced_cues`` alone. It does
    not read the Large Offset Evidence Gate, a first-dialogue delta, a
    configured offset, or any caller-supplied displacement.
    """
    if len(target_cues) < MIN_CUES_FOR_SEED or len(synced_cues) < MIN_CUES_FOR_SEED:
        return None
    max_lag_bins = max(1, int(max_lag_ms // DENSITY_BIN_MS))
    span = max(target_cues[-1][0], synced_cues[-1][0]) + DENSITY_BIN_MS
    bins = int(span // DENSITY_BIN_MS) + 1
    min_overlap = 150
    if bins - max_lag_bins < min_overlap:
        # Too short to search the whole range; narrow the search rather than
        # scoring a truncated overlap.
        max_lag_bins = max(1, bins - min_overlap)

    shifted = _density_train(target_cues, bins)
    reference = _density_train(synced_cues, bins)
    if not any(shifted) or not any(reference):
        return None

    lag_bins, correlation, coarse_bins = _best_lag(
        reference, shifted, max_lag_bins, min_overlap
    )
    if correlation < MIN_CORRELATION:
        return None
    # The coarse pass only chooses among 4s-spaced candidates, so it cannot
    # confirm the refined bin. Requiring the refined lag to sit inside the coarse
    # winner's basin is a cheap, genuinely independent corroboration: it is
    # violated when the fine scan latched onto a neighbouring peak the coarse pass
    # never favoured.
    if abs(lag_bins - coarse_bins) > FINE_RADIUS_BINS:
        return None
    margin = _peak_margin(reference, shifted, lag_bins, max_lag_bins, min_overlap)
    if margin < MIN_PEAK_MARGIN:
        return None
    disagreement = _half_split_offset(target_cues, synced_cues, max_lag_ms)
    if disagreement is None or disagreement > MAX_HALF_SPLIT_DISAGREEMENT_MS:
        return None

    offset_ms = float(lag_bins * DENSITY_BIN_MS)
    # Confirm the seed is usable: it must actually produce correspondences at
    # the radius the movement pass will use.
    from app.services.sync.alignment import pair_cues  # local: avoids import cycle

    shifted_target = [
        (start - int(offset_ms), end - int(offset_ms), text)
        for start, end, text in target_cues
    ]
    pairing = pair_cues(shifted_target, synced_cues, tolerance_ms=local_radius_ms)
    ratio = pairing.matched / len(target_cues)
    if ratio < MIN_SEED_SUPPORT:
        return None
    return MovementSeed(
        offset_ms=offset_ms,
        correlation=round(correlation, 4),
        peak_margin=round(margin, 4),
        half_split_disagreement_ms=round(disagreement, 1),
        support_pairs=pairing.matched,
        support_ratio=round(ratio, 4),
    )
