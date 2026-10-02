"""PART 7-9: segmentation-aware correspondence experiment.

DIAGNOSTIC ONLY. Nothing here is wired to any verification state, threshold, or
production decision. It exists to answer whether the residual p95 that rejects
the known-good Alass artifacts is an artefact of greedy one-to-one cue pairing
rather than a real timing error.

Current production pairing is greedy: for each cue in one sequence, take the
nearest unused cue in the other. When a subtitle is re-segmented -- one reference
line rendered as two output cues, or two merged into one -- a strictly one-to-one
match cannot represent that correspondence, so the closest partner is chosen and
the residual inherits the segmentation error instead of the true offset.

This module implements aggregation across adjacent cues and measures whether it
reduces the false residual without letting damaged input look clean.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

TIMING = re.compile(
    r"(\d{1,2}:\d{2}:\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2})[,.](\d{1,3})"
)


def to_ms(hms: str, frac: str) -> int:
    h, m, s = (int(x) for x in hms.split(":"))
    return ((h * 60 + m) * 60 + s) * 1000 + int(frac.ljust(3, "0"))


def parse_srt(data: bytes) -> list[tuple[int, int, str]]:
    """(start, end, text) per cue."""
    out = []
    for block in re.split(r"\n\s*\n", data.decode("utf-8", "replace").strip()):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        m = TIMING.search(block)
        if not m:
            continue
        a, b, c, d = m.groups()
        start, end = to_ms(a, b), to_ms(c, d)
        if end < start:
            start, end = end, start
        out.append((start, end, "\n".join(lines[2:])))
    return out


@dataclass
class PairingResult:
    pairs: list[tuple[int, int, int]] = field(default_factory=list)
    unmatched_before: list[int] = field(default_factory=list)
    unmatched_after: list[int] = field(default_factory=list)

    @property
    def residuals(self) -> list[float]:
        return [abs(d) for _, _, d in self.pairs]

    @property
    def monotonic(self) -> bool:
        """Both index sequences must advance together."""
        bi = [p[0] for p in self.pairs]
        ai = [p[1] for p in self.pairs]
        return all(x < y for x, y in zip(bi, bi[1:], strict=False)) and all(
            x < y for x, y in zip(ai, ai[1:], strict=False)
        )

    def p95(self) -> float:
        r = sorted(self.residuals)
        if not r:
            return float("inf")
        k = max(0, min(len(r) - 1, int(round(0.95 * (len(r) - 1)))))
        return float(r[k])

    def median(self) -> float:
        return statistics.median(self.residuals) if self.residuals else float("inf")


# --------------------------------------------------------------------------- #
# Candidate 1: greedy one-to-one (mirrors production)
# --------------------------------------------------------------------------- #


def greedy_pair(
    before: list[tuple[int, int, str]],
    after: list[tuple[int, int, str]],
    tolerance_ms: int = 5000,
) -> PairingResult:
    """Nearest unused partner, in order. The behaviour to beat."""
    res = PairingResult()
    used: set[int] = set()
    for i, (bs, be, _bt) in enumerate(before):
        best_j, best_d = -1, None
        for j, (as_, ae, _at) in enumerate(after):
            if j in used:
                continue
            # Distance between interval centres.
            d = abs(((as_ + ae) // 2) - ((bs + be) // 2))
            if best_d is None or d < best_d:
                best_j, best_d = j, d
        if best_j >= 0 and best_d is not None and best_d <= tolerance_ms:
            used.add(best_j)
            as_, ae, _ = after[best_j]
            delta = as_ - bs
            res.pairs.append((i, best_j, delta))
        else:
            res.unmatched_before.append(i)
    res.unmatched_after = [j for j in range(len(after)) if j not in used]
    return res


# --------------------------------------------------------------------------- #
# Candidate 2: segmentation-aware aggregation
# --------------------------------------------------------------------------- #


def _mergeable(span_ms: int, max_span_ms: int, gap_ms: int) -> bool:
    return span_ms <= max_span_ms and gap_ms <= 2000


def aggregated_pair(
    before: list[tuple[int, int, str]],
    after: list[tuple[int, int, str]],
    tolerance_ms: int = 5000,
    max_group: int = 3,
    max_group_span_ms: int = 6000,
) -> PairingResult:
    """Monotonic correspondence allowing one group to span several cues.

    Builds a monotonic alignment over cue *starts*, permitting an anchor to
    consume up to ``max_group`` consecutive cues on the other side when the
    combined span and the internal gap look like one event rendered as several
    cues. Greedy nearest-unused is used only to choose among admissible
    candidates, and the monotonicity constraint is what keeps it from
    double-counting.

    Rationale: a re-segmented line shifts the *boundaries*, not the event. The
    residual of a group should be taken against the group as a whole, so the
    segmentation error is not charged to the timing error.
    """
    res = PairingResult()
    if not before or not after:
        res.unmatched_before = list(range(len(before)))
        res.unmatched_after = list(range(len(after)))
        return res

    i = j = 0
    used_before: set[int] = set()
    used_after: set[int] = set()
    n_b, n_a = len(before), len(after)

    while i < n_b and j < n_a:
        bs, be, _ = before[i]
        as_, ae, _ = after[j]

        if abs(as_ - bs) <= tolerance_ms:
            # Try to extend the group in both directions, bounded and adjacent.
            g_end_i, g_end_j = i, j
            g_start_b, g_end_b = bs, be
            g_start_a, g_end_a = as_, ae
            for _ in range(max_group - 1):
                can_b = g_end_i + 1 < n_b
                can_a = g_end_j + 1 < n_a
                if not (can_b or can_a):
                    break
                next_b_start = before[g_end_i + 1][0] if can_b else None
                next_a_start = after[g_end_j + 1][0] if can_a else None
                # Which side is currently behind? Extend that one.
                if next_b_start is not None and (
                    next_a_start is None or next_b_start < next_a_start
                ):
                    cand_start, cand_end = before[g_end_i + 1][0], before[g_end_i + 1][1]
                    gap = cand_start - g_end_b
                    span = max(g_end_b, cand_end) - g_start_b
                    if not _mergeable(span, max_group_span_ms, gap):
                        break
                    g_end_i += 1
                    g_end_b = max(g_end_b, cand_end)
                elif next_a_start is not None:
                    cand_start, cand_end = after[g_end_j + 1][0], after[g_end_j + 1][1]
                    gap = cand_start - g_end_a
                    span = max(g_end_a, cand_end) - g_start_a
                    if not _mergeable(span, max_group_span_ms, gap):
                        break
                    g_end_j += 1
                    g_end_a = max(g_end_a, cand_end)
                else:
                    break
            # Residual of the group, taken on the group as a whole.
            delta = g_start_a - g_start_b
            res.pairs.append((i, j, delta))
            for x in range(i, g_end_i + 1):
                used_before.add(x)
            for y in range(j, g_end_j + 1):
                used_after.add(y)
            i, j = g_end_i + 1, g_end_j + 1
            continue

        # Outside tolerance: advance the side that is further ahead.
        if bs < as_:
            used_before.add(i)
            i += 1
        else:
            used_after.add(j)
            j += 1

    res.unmatched_before = [x for x in range(n_b) if x not in used_before]
    res.unmatched_after = [y for y in range(n_a) if y not in used_after]
    return res


# --------------------------------------------------------------------------- #
# Completeness signals -- deliberately kept separate from timing (Part 8)
# --------------------------------------------------------------------------- #


@dataclass
class Completeness:
    """Signals that bear on CONTENT, kept apart from any timing judgement.

    No threshold is applied and none is implied. These are reported so a future
    review can see what is and is not being measured -- the existing study
    established that timing metrics cannot detect content deletion, so a clean
    timing result must never be read as a complete one.
    """

    cue_count_ratio: float
    runtime_coverage: float
    reference_coverage: float
    first_cue_delta_ms: int
    last_cue_delta_ms: int
    active_duration_ratio: float

    def as_row(self) -> dict:
        return {
            "cue_count_ratio": round(self.cue_count_ratio, 4),
            "runtime_coverage": round(self.runtime_coverage, 4),
            "reference_coverage": round(self.reference_coverage, 4),
            "first_cue_delta_ms": self.first_cue_delta_ms,
            "last_cue_delta_ms": self.last_cue_delta_ms,
            "active_duration_ratio": round(self.active_duration_ratio, 4),
        }


def completeness(
    before: list[tuple[int, int, str]],
    after: list[tuple[int, int, str]],
    runtime_ms: int,
) -> Completeness:
    if not before or not after:
        # An absent side is itself the finding. Report neutral signals rather
        # than raising, so a diagnostic report can never be lost to a crash on
        # exactly the damaged input it exists to characterise.
        return Completeness(
            cue_count_ratio=0.0 if not after else float("inf"),
            runtime_coverage=0.0,
            reference_coverage=0.0,
            first_cue_delta_ms=0,
            last_cue_delta_ms=0,
            active_duration_ratio=0.0,
        )
    b_active = sum(e - s for s, e, _ in before)
    a_active = sum(e - s for s, e, _ in after)
    b_span = (max(e for _, e, _ in before) - min(s for s, _, _ in before)) or 1
    a_span = (max(e for _, e, _ in after) - min(s for s, _, _ in after)) or 1
    return Completeness(
        cue_count_ratio=len(after) / len(before),
        runtime_coverage=min(1.0, a_span / runtime_ms) if runtime_ms else 0.0,
        reference_coverage=min(1.0, b_span / runtime_ms) if runtime_ms else 0.0,
        first_cue_delta_ms=after[0][0] - before[0][0],
        last_cue_delta_ms=after[-1][1] - before[-1][1],
        active_duration_ratio=a_active / b_active if b_active else 0.0,
    )
