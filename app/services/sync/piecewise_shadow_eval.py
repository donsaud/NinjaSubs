"""Shadow evaluation for the piecewise offset model.

Produces a label and an explanation. It changes nothing: no artifact is served,
no verdict is written, no cache key is touched, no state is transitioned. The
name is chosen so that a log line is unambiguous about being a shadow.

    PIECEWISE_SHADOW_ACCEPT
    PIECEWISE_SHADOW_REJECT
    PIECEWISE_SHADOW_ABSTAIN

ACCEPT and REJECT are statements about the *timing model only*. Completeness is
computed and reported separately and is never folded in, because surviving cues
stay aligned when content is deleted and a combined number would hide exactly
the damage the model cannot see. A shadow ACCEPT therefore always carries its
completeness row, and callers must read both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from app.services.sync.piecewise_shadow import (
    Completeness,
    Cue,
    Observation,
    PiecewiseFit,
    compute_completeness,
    fit_piecewise,
)


class ShadowDecision(str, Enum):
    ACCEPT = "PIECEWISE_SHADOW_ACCEPT"
    REJECT = "PIECEWISE_SHADOW_REJECT"
    ABSTAIN = "PIECEWISE_SHADOW_ABSTAIN"


@dataclass
class ShadowReport:
    decision: ShadowDecision
    fit: PiecewiseFit
    completeness: Completeness
    #: Why the decision was reached, in plain terms. Never contains subtitle text.
    reasons: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "decision": self.decision.value,
            "fit": self.fit.as_row(),
            "completeness": self.completeness.as_row(),
            "reasons": list(self.reasons),
        }


# Bounds for the shadow verdict itself. These are diagnostic bounds: they decide
# only which SHADOW label is printed, and are not production thresholds. Kept in
# one place so it is obvious they belong to the shadow and to nothing else.
SHADOW_MAX_PENALIZED_P95_MS = 2000.0
SHADOW_MIN_PAIRS = 40
SHADOW_MIN_SEGMENT_PAIRS = 12


def build_observations(
    target: list[Cue],
    reference: list[Cue],
    *,
    max_lag_ms: int = 400_000,
) -> list[Observation]:
    """Correspond target cues to reference cues, and record the local lag.

    Nearest-start matching over a wide window, so a large offset -- the whole
    point of the model -- is still found. A pair contributes one observation: the
    local lag at the target's own time. Nothing here is case-specific.
    """
    if not target or not reference:
        return []
    ref_sorted = sorted(reference, key=lambda c: c.start_ms)
    ref_starts = [c.start_ms for c in ref_sorted]
    observations: list[Observation] = []
    used: set[int] = set()
    for t in sorted(target, key=lambda c: c.start_ms):
        # Binary search for the nearest reference start.
        lo, hi = 0, len(ref_starts)
        while lo < hi:
            mid = (lo + hi) // 2
            if ref_starts[mid] < t.start_ms:
                lo = mid + 1
            else:
                hi = mid
        best_j, best_d = -1, None
        for j in (lo - 1, lo):
            if 0 <= j < len(ref_starts) and j not in used:
                d = abs(ref_starts[j] - t.start_ms)
                if best_d is None or d < best_d:
                    best_j, best_d = j, d
        if best_j >= 0 and best_d is not None and best_d <= max_lag_ms:
            used.add(best_j)
            observations.append(
                Observation(t.start_ms, float(ref_starts[best_j] - t.start_ms))
            )
    return observations


def evaluate_shadow(
    target: list[Cue],
    reference: list[Cue],
    runtime_ms: int,
    *,
    max_segments: int = 2,
) -> ShadowReport:
    """Fit the offset field and label it. Never mutates anything."""
    observations = build_observations(target, reference)
    fit = fit_piecewise(observations, max_segments=max_segments)
    completeness = compute_completeness(target, reference, runtime_ms)
    reasons: list[str] = []

    if len(observations) < SHADOW_MIN_PAIRS:
        reasons.append(
            f"too few correspondences ({len(observations)}) to fit an offset field"
        )
        return ShadowReport(ShadowDecision.ABSTAIN, fit, completeness, reasons)

    thin = [
        s for s in fit.segments if s.pairs < SHADOW_MIN_SEGMENT_PAIRS
    ]
    if thin:
        reasons.append(
            f"{len(thin)} segment(s) below the minimum evidence "
            f"({SHADOW_MIN_SEGMENT_PAIRS} pairs)"
        )
        return ShadowReport(ShadowDecision.ABSTAIN, fit, completeness, reasons)

    if fit.penalized_p95 <= SHADOW_MAX_PENALIZED_P95_MS:
        reasons.append(
            f"penalised p95 {fit.penalized_p95:.0f}ms within the shadow bound "
            f"({SHADOW_MAX_PENALIZED_P95_MS:.0f}ms) across {fit.piece_count} piece(s)"
        )
        if not fit.time_map_increasing:
            # Reported, not enforced. A correction that *reduces* the offset over
            # time is legitimate -- that is the shape the real cases have -- and
            # its image necessarily overlaps, so a strictly increasing time map
            # cannot be required. Enforcing it would reject the very case the
            # model exists to describe.
            reasons.append(
                "offset decreases over time, so the mapped ranges overlap; "
                "reported, not penalised"
            )
        reasons.append(
            "TIMING ONLY -- completeness is reported separately and is not part "
            "of this label"
        )
        return ShadowReport(ShadowDecision.ACCEPT, fit, completeness, reasons)

    reasons.append(
        f"penalised p95 {fit.penalized_p95:.0f}ms above the shadow bound "
        f"({SHADOW_MAX_PENALIZED_P95_MS:.0f}ms)"
    )
    return ShadowReport(ShadowDecision.REJECT, fit, completeness, reasons)
