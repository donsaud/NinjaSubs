"""Bounded, text-free affine search with independent timeline verification.

Union occupancy makes duplicate, overlapping, split and merged cues equivalent.
Gap boundaries supply hypotheses, never identity evidence. Distributed occupancy
and local residual checks arbitrate every hypothesis independently of cue counts.
No legacy verifier thresholds or MovieHash provenance are changed here.
"""

from __future__ import annotations

import logging
import math
import re
import statistics
from dataclasses import dataclass, field

from app.services.subtitle_matcher import parse_srt_cues
from app.services.sync.alignment import (
    SubtitleEvaluation,
    SyncState,
    VerificationAvailability,
)
from app.services.sync.ninja_points import MAX_SLOPE, MIN_SLOPE
from app.services.sync.pal_two_point import _is_usable_dialogue, find_endpoint_anchors
from app.services.sync_service import _parse_timestamp_ms

logger = logging.getLogger(__name__)
STRATEGY_ID = "timeline_affine"
MODEL_VERSION = "timeline-affine-v2"
MAX_LANDMARKS = 40
MAX_HYPOTHESES = 384
WINDOWS = 8
MIN_SPAN_MS = 120_000
JOIN_GAP_MS = 250
TOLERANCE_MS = 350
MIN_SCORE = 0.80
MIN_WINDOW_SCORE = 0.70
MIN_IMPROVEMENT = 0.06
Interval = tuple[float, float]
Cue = tuple[int, int, str]


@dataclass(frozen=True)
class Landmark:
    time_ms: float
    kind: str
    gap_ms: float = 0


def union(intervals: list[Interval], join_ms: float = 0) -> list[Interval]:
    """Sorted occupancy union; duplicates add no weight."""
    out: list[Interval] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if out and start <= out[-1][1] + join_ms:
            out[-1] = (out[-1][0], max(end, out[-1][1]))
        else:
            out.append((start, end))
    return out


@dataclass(frozen=True)
class Timeline:
    occupancy: list[Interval]
    landmarks: list[Landmark]
    cue_count: int

    @classmethod
    def from_cues(cls, cues: list[Cue]) -> Timeline:
        dialogue = [c for c in cues if c[1] > c[0] and _is_usable_dialogue(c[2])]
        occupied = union([(s, e) for s, e, _ in dialogue], JOIN_GAP_MS)
        if not occupied:
            return cls([], [], len(cues))
        # Endpoints require the project's existing reliable dialogue-run policy.
        endpoints = find_endpoint_anchors(dialogue, dialogue)
        marks: list[Landmark] = []
        if endpoints:
            marks = [Landmark(endpoints[0].target_ms, "first"),
                     Landmark(endpoints[1].target_ms, "last")]
        # Isolated closing dialogue is a candidate too. Run corroboration may
        # skip a genuine final line after a long silent scene. These filtered
        # boundaries still need the same distributed verification as any gap.
        marks.extend((Landmark(dialogue[0][0], "boundary_first"),
                      Landmark(dialogue[-1][0], "boundary_last")))
        gaps = [(b[0] - a[1], a[1], b[0])
                for a, b in zip(occupied, occupied[1:], strict=False)
                if b[0] - a[1] >= 2000]
        # Reserve landmarks in every temporal window before taking large gaps.
        span = occupied[-1][1] - occupied[0][0]
        selected: set[tuple[float, float, float]] = set()
        for window in range(WINDOWS):
            local = [g for g in gaps if min(WINDOWS - 1,
                     int((g[2] - occupied[0][0]) * WINDOWS / span)) == window]
            selected.update(sorted(local, reverse=True)[:2])
        for gap in sorted(gaps, reverse=True):
            if len(selected) >= (MAX_LANDMARKS - len(marks)) // 2:
                break
            selected.add(gap)
        for size, end, start in sorted(selected):
            marks.extend((Landmark(end, "gap_end", size),
                          Landmark(start, "gap_start", size)))
        return cls(occupied, sorted(marks, key=lambda m: m.time_ms), len(cues))

    @property
    def span(self) -> float:
        return self.occupancy[-1][1] - self.occupancy[0][0] if self.occupancy else 0


def plausible_affine_reference(target_text: str, reference_text: str) -> bool:
    """Discovery hint only: admit a full-length candidate to verification.

    An intro displacement is not proof of a cut. This cheap span check prevents
    discovery from discarding plausible drift before timeline verification.
    """
    target = Timeline.from_cues(parse_srt_cues(target_text))
    reference = Timeline.from_cues(parse_srt_cues(reference_text))
    return (min(target.span, reference.span) >= MIN_SPAN_MS
            and min(len(target.occupancy), len(reference.occupancy)) >= 16
            and MIN_SLOPE <= reference.span / target.span <= MAX_SLOPE)


def intersection(a: list[Interval], b: list[Interval]) -> float:
    total = 0.0
    i = j = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def duration(intervals: list[Interval]) -> float:
    return sum(e - s for s, e in intervals)


def occupancy_score(a: list[Interval], b: list[Interval]) -> float:
    """Symmetric duration coverage with a small fixed boundary tolerance.

    Each side's true duration is the denominator, so padding cannot turn a
    sparse or truncated reference into perfect agreement.
    """
    da, db = duration(a), duration(b)
    if not da or not db:
        return 0
    expanded_a = union([(s - TOLERANCE_MS, e + TOLERANCE_MS) for s, e in a])
    expanded_b = union([(s - TOLERANCE_MS, e + TOLERANCE_MS) for s, e in b])
    ca = intersection(a, expanded_b) / da
    cb = intersection(b, expanded_a) / db
    return 2 * ca * cb / (ca + cb) if ca + cb else 0


def mapped(timeline: Timeline, slope: float, intercept: float) -> list[Interval]:
    return [(s * slope + intercept, e * slope + intercept) for s, e in timeline.occupancy]


def clip(intervals: list[Interval], start: float, end: float) -> list[Interval]:
    return [(max(s, start), min(e, end)) for s, e in intervals if e > start and s < end]


def candidate_pairs(target: Timeline, reference: Timeline) -> list[tuple[Landmark, Landmark]]:
    pairs: list[tuple[Landmark, Landmark]] = []
    for t in target.landmarks:
        tp = (t.time_ms - target.occupancy[0][0]) / target.span
        options = []
        for r in reference.landmarks:
            rp = (r.time_ms - reference.occupancy[0][0]) / reference.span
            if t.kind != r.kind or abs(tp - rp) > 0.12:
                continue
            if t.gap_ms and abs(t.gap_ms - r.gap_ms) > max(1500, t.gap_ms * 0.25):
                continue
            options.append((abs(tp - rp), r))
        pairs.extend((t, r) for _, r in sorted(options, key=lambda x: x[0])[:3])
    return pairs


def hypotheses(target: Timeline, reference: Timeline) -> list[tuple[float, float]]:
    pairs = candidate_pairs(target, reference)
    models: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()

    def add(slope: float, intercept: float) -> None:
        key = (round(slope * 1_000_000), round(intercept / 25))
        if MIN_SLOPE <= slope <= MAX_SLOPE and abs(intercept) <= 300_000 and key not in seen:
            seen.add(key)
            models.append((slope, intercept))

    # Endpoint model is only one hypothesis; all middle windows must verify it.
    opening = [(t, r) for t, r in pairs if t.kind in {"first", "boundary_first"}]
    closing = [(t, r) for t, r in pairs if t.kind in {"last", "boundary_last"}]
    ordered = [(p, q) for p in opening for q in closing]
    ordered.extend((p, q) for i, p in enumerate(pairs) for q in pairs[i + 1:]
                   if q[0].time_ms - p[0].time_ms >= target.span * 0.35)
    for (t1, r1), (t2, r2) in ordered:
        if len(models) >= MAX_HYPOTHESES:
            break
        span = t2.time_ms - t1.time_ms
        if span <= 0:
            continue
        slope = (r2.time_ms - r1.time_ms) / span
        add(slope, r1.time_ms - slope * t1.time_ms)
    for t, r in pairs:
        if len(models) >= MAX_HYPOTHESES:
            break
        add(1.0, r.time_ms - t.time_ms)
    return models


def hypothesis_score(target: Timeline, reference: Timeline, slope: float,
                     intercept: float) -> float:
    """Occupancy plus robust onset precision; display duration cannot select drift.

    A high coverage plateau can hide a small systematic timing error. Gap
    onsets supply independent precision. Cap the correction at 0.1 so poor
    occupancy cannot win just because two isolated endpoints fit.
    """
    residuals = []
    boundary_residuals = []
    for t in target.landmarks:
        if t.kind not in {"gap_start", "boundary_first", "boundary_last"}:
            continue
        options = [r for r in reference.landmarks if r.kind == t.kind]
        if options:
            residual = min(abs(t.time_ms * slope + intercept - r.time_ms) for r in options)
            if t.kind.startswith("boundary_"):
                boundary_residuals.append(min(1000, residual))
            if residual <= 1000:
                residuals.append(residual)
    penalty = statistics.median(residuals) if residuals else 1000
    # Small, bounded corroboration from outer onsets resolves a precision tie;
    # at most 0.01 score weight, never enough to rescue poor middle coverage.
    boundary_penalty = (sum(boundary_residuals) / len(boundary_residuals)
                        if boundary_residuals else 1000)
    return (occupancy_score(mapped(target, slope, intercept), reference.occupancy)
            - penalty / 10000 - boundary_penalty / 100000)


def refine_model(target: Timeline, reference: Timeline, slope: float,
                 intercept: float) -> tuple[float, float]:
    """Remove millisecond quantization using only precise consensus landmarks.

    Gap ends can differ with display duration, so this refinement uses onsets.
    No noisy or merely near timing is averaged into a precision fit.
    """
    pairs = []
    for t in target.landmarks:
        if t.kind not in {"boundary_first", "boundary_last", "gap_start"}:
            continue
        options = [r for r in reference.landmarks if r.kind == t.kind]
        if not options:
            continue
        r = min(options, key=lambda r: abs(t.time_ms * slope + intercept - r.time_ms))
        if abs(t.time_ms * slope + intercept - r.time_ms) <= 10:
            pairs.append((t.time_ms, r.time_ms))
    if len(pairs) < 6 or max(t for t, _ in pairs) - min(t for t, _ in pairs) < target.span * 0.7:
        return slope, intercept
    mt, mr = sum(t for t, _ in pairs) / len(pairs), sum(r for _, r in pairs) / len(pairs)
    a = sum((t - mt) * (r - mr) for t, r in pairs) / sum((t - mt) ** 2 for t, _ in pairs)
    return a, mr - a * mt


@dataclass
class TimelineVerification:
    accepted: bool = False
    before: float = 0
    after: float = 0
    windows: list[float] = field(default_factory=list)
    local_offsets_ms: list[int] = field(default_factory=list)
    anchor_pairs: list[tuple[float, float]] = field(default_factory=list)
    anchor_outliers: int = 0
    classification: str = "INSUFFICIENT_EVIDENCE"
    reasons: list[str] = field(default_factory=list)


def verify_timeline(target: Timeline, reference: Timeline,
                    slope: float, intercept: float) -> TimelineVerification:
    """Independent global verification. Local fits diagnose, never repair cuts."""
    report = TimelineVerification()
    if (not math.isfinite(slope) or not math.isfinite(intercept)
            or not MIN_SLOPE <= slope <= MAX_SLOPE
            or min(target.span, reference.span) < MIN_SPAN_MS
            or min(len(target.occupancy), len(reference.occupancy)) < 16):
        report.reasons.append("insufficient_evidence_or_invalid_model")
        return report
    aligned = mapped(target, slope, intercept)
    report.before = occupancy_score(target.occupancy, reference.occupancy)
    report.after = occupancy_score(aligned, reference.occupancy)
    start, end = min(aligned[0][0], reference.occupancy[0][0]), max(
        aligned[-1][1], reference.occupancy[-1][1])
    for window in range(WINDOWS):
        lo, hi = start + (end - start) * window / WINDOWS, start + (end - start) * (window + 1) / WINDOWS
        a, b = clip(aligned, lo, hi), clip(reference.occupancy, lo, hi)
        score = occupancy_score(a, b)
        report.windows.append(score)
        # Hold the reference window fixed; shifting is diagnostic only.
        local = [(occupancy_score(clip([(s + d, e + d) for s, e in aligned], lo, hi), b), d)
                 for d in range(-8000, 8001, 500)]
        best_score, offset = max(local, key=lambda x: (x[0], -abs(x[1])))
        report.local_offsets_ms.append(offset if best_score - score >= 0.04 else 0)
    for t in target.landmarks:
        options = [r for r in reference.landmarks if r.kind == t.kind]
        if not options:
            report.anchor_outliers += 1
            continue
        r = min(options, key=lambda r: abs(t.time_ms * slope + intercept - r.time_ms))
        if abs(t.time_ms * slope + intercept - r.time_ms) <= 1000:
            report.anchor_pairs.append((t.time_ms, r.time_ms))
        else:
            report.anchor_outliers += 1
    regions = {min(WINDOWS - 1, int((t - target.occupancy[0][0]) * WINDOWS / target.span))
               for t, _ in report.anchor_pairs}
    if len(report.anchor_pairs) < 6 or len(regions) < 5:
        report.reasons.append("insufficient_distributed_landmarks")
    if report.after < MIN_SCORE:
        report.reasons.append("poor_timeline_agreement")
    if report.after - report.before < MIN_IMPROVEMENT:
        report.reasons.append("no_meaningful_improvement")
    if min(report.windows) < MIN_WINDOW_SCORE:
        report.reasons.append("inconsistent_windows")
    if any(abs(d) > 1500 for d in report.local_offsets_ms):
        report.reasons.append("local_discontinuity")
    report.accepted = not report.reasons
    if report.accepted:
        report.classification = ("CONSTANT_OFFSET" if abs(slope - 1) < 0.0001 else
                                 "PAL_LIKE_AFFINE" if min(abs(slope - 25 / 23.976),
                                 abs(slope - 23.976 / 25)) < 0.001 else "GLOBAL_AFFINE")
    elif ("local_discontinuity" in report.reasons or "inconsistent_windows" in report.reasons
          or "poor_timeline_agreement" in report.reasons):
        report.classification = "DISCONTINUITY_OR_DIFFERENT_CUT"
    return report


# Only timestamp lines are replaced. Cue bodies, order, IDs, whitespace and
# trailing settings survive byte-for-byte in the decoded target string.
_TIMING = re.compile(r"(?m)^(?P<prefix>[ \t]*)(?P<start>(?:\d{1,2}:)?\d{1,2}:\d{2}[,.]\d{1,3})"
                     r"(?P<arrow>[ \t]*-->[ \t]*)(?P<end>(?:\d{1,2}:)?\d{1,2}:\d{2}[,.]\d{1,3})")


def transform_text(text: str, slope: float, intercept: float) -> str:
    def timestamp(value: str) -> str:
        ms = round(_parse_timestamp_ms(value) * slope + intercept)
        if ms < 0:
            raise ValueError("negative transformed timestamp")
        h, rest = divmod(ms, 3_600_000)
        m, rest = divmod(rest, 60_000)
        s, millis = divmod(rest, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"

    def replace(match: re.Match[str]) -> str:
        return (match["prefix"] + timestamp(match["start"]) + match["arrow"]
                + timestamp(match["end"]))

    return _TIMING.sub(replace, text)


@dataclass
class AffineResult:
    accepted: bool = False
    slope: float | None = None
    intercept_ms: float | None = None
    transformed_text: str | None = None
    evaluation: SubtitleEvaluation | None = None
    diagnostics: TimelineVerification = field(default_factory=TimelineVerification)
    hypotheses_tested: int = 0


def attempt_timeline_affine(target_text: str, reference_text: str) -> AffineResult:
    target_cues, reference_cues = parse_srt_cues(target_text), parse_srt_cues(reference_text)
    target, reference = Timeline.from_cues(target_cues), Timeline.from_cues(reference_cues)
    result = AffineResult()
    if min(target.span, reference.span) < MIN_SPAN_MS:
        result.diagnostics.reasons.append("insufficient_timeline_span")
        logger.info("[timeline-affine] target_cues=%d reference_cues=%d accepted=false "
                    "hypotheses_tested=0 reason=insufficient_timeline_span",
                    len(target_cues), len(reference_cues))
        return result
    models = hypotheses(target, reference)
    if not models:
        result.diagnostics.reasons.append("no_plausible_hypotheses")
    # Cheap occupancy ranking precedes expensive independent verification.
    ranked = sorted(models, key=lambda m: hypothesis_score(target, reference, *m),
                    reverse=True)
    for slope, intercept in ranked[:8]:
        slope, intercept = refine_model(target, reference, slope, intercept)
        report = verify_timeline(target, reference, slope, intercept)
        result.hypotheses_tested += 1
        if result.slope is None:
            result.slope, result.intercept_ms, result.diagnostics = slope, intercept, report
        if not report.accepted:
            continue
        try:
            transformed = transform_text(target_text, slope, intercept)
        except ValueError:
            report.accepted = False
            report.reasons.append("invalid_transformed_timestamps")
            continue
        generated = parse_srt_cues(transformed)
        if (len(generated) != len(target_cues)
                or [t for _, _, t in generated] != [t for _, _, t in target_cues]
                or _TIMING.sub("", transformed) != _TIMING.sub("", target_text)
                or any((s, e) != (round(os * slope + intercept), round(oe * slope + intercept))
                       for (s, e, _), (os, oe, _) in zip(generated, target_cues, strict=True))
                or any(e <= s for s, e, _ in generated)
                or len(_TIMING.findall(target_text)) != len(target_cues)):
            report.accepted = False
            report.reasons.append("artifact_integrity_failed")
            continue
        # Re-measure the actual artifact rather than trusting the transform code.
        artifact = Timeline.from_cues(generated)
        if occupancy_score(artifact.occupancy, reference.occupancy) < MIN_SCORE:
            report.accepted = False
            report.reasons.append("artifact_timeline_failed")
            continue
        evaluation = SubtitleEvaluation(
            coverage_score=report.after, structural_similarity=min(report.windows),
            sync_confidence=report.after * 100, cut_verdict="drift",
            reasons=[f"{STRATEGY_ID}/{MODEL_VERSION}: {report.classification}; "
                     f"before={report.before:.4f} after={report.after:.4f}"],
        )
        evaluation.set_verdict(SyncState.VERIFIED_RESYNCED, VerificationAvailability.VERIFIED)
        result.accepted, result.slope, result.intercept_ms = True, slope, intercept
        result.transformed_text, result.evaluation, result.diagnostics = transformed, evaluation, report
        break
    logger.info(
        "[timeline-affine] target_cues=%d reference_cues=%d target_runs=%d reference_runs=%d "
        "landmarks_target=%d landmarks_reference=%d hypotheses_scored=%d hypotheses_tested=%d "
        "winner_slope=%s winner_intercept_ms=%s anchor_inliers=%d anchor_outliers=%d "
        "timeline_score_before=%.4f timeline_score_after=%.4f coverage=%.4f "
        "window_consistency=%s local_offsets_ms=%s classification=%s accepted=%s reason=%s",
        len(target_cues), len(reference_cues), len(target.occupancy), len(reference.occupancy),
        len(target.landmarks), len(reference.landmarks), len(models), result.hypotheses_tested,
        result.slope, result.intercept_ms, len(result.diagnostics.anchor_pairs),
        result.diagnostics.anchor_outliers, result.diagnostics.before, result.diagnostics.after,
        result.diagnostics.after, result.diagnostics.windows, result.diagnostics.local_offsets_ms,
        result.diagnostics.classification, result.accepted, result.diagnostics.reasons,
    )
    return result
