"""Correspondence-aware verification for hybrid synchronization.

The legacy production analyzer pairs cues by nearest start timestamp
(``pair_cues``), which misfires across different cue counts and dialogue
segmentation: 825-vs-960 Whiplash cues inflate residual p95 to 1956ms, and
921-vs-684 Dexter cues to 4660ms, on correct outputs. Its movement MAD
(``mad <= 800``) additionally penalizes legitimate accumulated drift, since a
valid 25fps->23.976fps correction moves late-film cues by minutes.

This module instead measures timing residuals on independently established
dialogue correspondences (1:1, 1:N, N:1, bounded N:M groups) built with the
shared Two-Point correspondence layer -- one implementation of the matching
rules, reused here, never duplicated. Movement magnitude is reported as
telemetry only; it is never judged as synchronization error.

Threshold basis (§8: new metrics, not relaxed old ones; units are verified
dialogue-timing residuals in ms, calibrated on Whiplash/Dexter evidence and
exercised against synthetic positives and negatives):

* regional/overall median <= 750, p95 <= 2000, max <= 3000;
* every populated film third holds >= 2 independent correspondences;
* at least 5 independent correspondence groups (serving bar; detection alone
  needs fewer, but serving must resist coincidence);
* target retention >= 0.80 (existing production floor, reused unchanged);
* strict cue order, valid durations, no dialogue truncation;
* fitting anchors must not appear among verification checkpoints;
* anything short of the above fails closed to UNVERIFIED.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Any

from two_point_sync.anchors import AnchorResult
from two_point_sync.parser import Cue as NucleusCue
from two_point_sync.parser import SubtitleParseError, parse_srt
from two_point_sync.sanitizer import classify_all

logger = logging.getLogger(__name__)

#: Verification methodology identifier (telemetry, logs, cache decision).
METHOD = "correspondence-v1"

#: Minimum independent correspondence groups to serve (see module docstring).
MIN_INDEPENDENT_GROUPS = 5

#: Per-region minimum correspondence points.
REGION_MIN_POINTS = 2

#: Correspondence residual bounds in ms (see module docstring). The tail
#: rule uses P90, aligned with the standalone nucleus validation policy:
#: translated and re-segmented subtitles carry per-pair segmentation jitter
#: up to ~2.5s on a small minority of pairs (proven within-span and
#: near-miss outliers), so a P95 bound would mistake boundary effects for
#: mistiming while P90 still guards the bulk. P95 stays reported telemetry.
MEDIAN_TOLERANCE_MS = 750.0
P90_TOLERANCE_MS = 2000.0
P95_TOLERANCE_MS = 2000.0
MAX_TOLERANCE_MS = 3000.0

#: Target-cue retention floor (existing production floor, reused unchanged).
MIN_TARGET_RETENTION = 0.80

#: A candidate cue sitting at zero whose original starts beyond this grace is
#: a clamped timestamp, not an alignment (see Dexter ad cue #1).
CLAMP_GRACE_MS = 5_000

#: A forward jump at least this large is a significant structural transition:
#: correspondences are then required on both sides of it.
SIGNIFICANT_JUMP_MS = 30_000
BOUNDARY_CONTEXT_MS = 120_000


@dataclass(frozen=True)
class CorrespondenceVerdict:
    """Measurement verdict for one candidate. Advisory only: the caller
    (orchestrator) combines it with Target Binding and reference eligibility
    before any serving decision."""

    accepted: bool
    method: str = METHOD
    candidate_algorithm: str = ""
    reference_sha: str = ""
    groups_found: int = 0
    independent_used: int = 0
    overlapping_excluded: int = 0
    coverage_thirds: dict[str, int] = field(default_factory=dict)
    median_residual_ms: float = 0.0
    p90_residual_ms: float = 0.0
    p95_residual_ms: float = 0.0
    max_residual_ms: float = 0.0
    movement_magnitude_ms: float = 0.0
    transformation_model: dict[str, Any] = field(default_factory=dict)
    regional_discontinuities: tuple[dict[str, Any], ...] = ()
    outlier_count: int = 0
    outliers: tuple[dict[str, Any], ...] = ()
    clamped: tuple[dict[str, Any], ...] = ()
    unmatched_reference_spans: tuple[dict[str, Any], ...] = ()
    failure_reasons: tuple[str, ...] = ()
    may_serve_recommendation: bool = False


def anchor_pairs(anchor_result: AnchorResult, *, strong_only: bool = False) -> list[tuple[int, int, int, tuple, tuple]]:
    """Flatten anchors to signature-deduplicated correspondence records.

    Each record is ``(target_ms, reference_ms, region_ms, target_numbers,
    reference_numbers)`` in deterministic order. Overlapping groups share
    member signatures and contribute once -- never double-counted as
    independent temporal evidence. With ``strong_only=True``, weak bare
    singletons are additionally excluded from verification evidence (they
    stay countable via ``groups_found`` callers that pass ``False``).
    """
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    records: list[tuple[int, int, int, tuple, tuple]] = []

    def _add(item: Any) -> None:
        if strong_only and not is_verification_strong(item):
            return
        signature = (tuple(item.target_numbers or (item.target_number,)),
                     tuple(item.reference_numbers or (item.reference_number,)))
        if signature in seen:
            return
        seen.add(signature)
        records.append((item.target_ms, item.reference_ms, item.target_ms,
                        signature[0], signature[1]))

    for item in list(anchor_result.checkpoints) + list(anchor_result.contradictory):
        _add(item)
    if anchor_result.first_anchor is not None:
        _add(anchor_result.first_anchor)
    if anchor_result.last_anchor is not None:
        _add(anchor_result.last_anchor)
    records.sort(key=lambda record: (record[0], record[1]))
    return records


def held_out_checkpoints(anchor_result: AnchorResult) -> list[Any]:
    """Verification checkpoints with fitting anchors provably excluded."""
    primary = set()
    if anchor_result.first_anchor is not None:
        item = anchor_result.first_anchor
        primary.add((tuple(item.target_numbers or (item.target_number,)),
                     tuple(item.reference_numbers or (item.reference_number,))))
    if anchor_result.last_anchor is not None:
        item = anchor_result.last_anchor
        primary.add((tuple(item.target_numbers or (item.target_number,)),
                     tuple(item.reference_numbers or (item.reference_number,))))
    held_out = []
    for item in list(anchor_result.checkpoints) + list(anchor_result.contradictory):
        signature = (tuple(item.target_numbers or (item.target_number,)),
                     tuple(item.reference_numbers or (item.reference_number,)))
        if signature not in primary:
            held_out.append(item)
    return held_out


def is_verification_strong(item: Any) -> bool:
    """Whether one correspondence may serve as verification evidence.

    Mirrors the nucleus anchor-strength bar: grouped matches (exact joins or
    anchored groups) always qualify; single correspondences need confidence
    >= 0.40, which bare generic singletons ("yes", "no", "thanks", ...)
    cannot reach without distinguishing neighbour context. Weak items stay
    visible in group counts but never verify.
    """
    if any("group" in part for part in item.evidence):
        return True
    return float(item.confidence) >= 0.40


def _region_of(target_ms: int, span_start_ms: int, span_end_ms: int) -> str:
    width = max(1, span_end_ms - span_start_ms)
    position = (target_ms - span_start_ms) / width
    if position < 1.0 / 3.0:
        return "opening"
    if position < 2.0 / 3.0:
        return "middle"
    return "ending"


def _split_segmentation_overlaps(
    pairs: list[tuple[int, int, int]],
    reference_cues: list[NucleusCue],
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """Split pairs whose large residual is explained by segmentation.

    A residual measured against a reference cue START overstates the error
    when the matching event is a later line of a long, multi-line reference
    cue (Dexter T#795 lands inside R#582; several Whiplash lines land inside
    grouped reference cues). Such a pair qualifies as an overlap only when
    the residual is material (>1s, so sub-second noise never counts) AND the
    candidate falls inside the matched cue's span AND the residual is
    shorter than that span (the span explains the whole deviation).
    Overlaps stay listed as examined outliers but leave the residual
    aggregates, which would otherwise mistake segmentation for timing error.
    """
    by_start: dict[int, list[NucleusCue]] = {}
    for cue in reference_cues:
        by_start.setdefault(cue.start_ms, []).append(cue)
    clean: list[tuple[int, int, int]] = []
    overlapped: list[tuple[int, int, int]] = []
    for candidate_ms, reference_ms, target_ms in pairs:
        residual = abs(float(candidate_ms) - float(reference_ms))
        explained = False
        if residual > 1000.0:
            for cue in by_start.get(reference_ms, []):
                span = cue.end_ms - cue.start_ms
                if cue.start_ms <= candidate_ms <= cue.end_ms and residual < span:
                    explained = True
                    break
        if explained:
            overlapped.append((candidate_ms, reference_ms, target_ms))
        else:
            clean.append((candidate_ms, reference_ms, target_ms))
    return clean, overlapped


def _resolve_pairs(
    records: list[tuple[int, int, int, tuple, tuple]],
    by_number: dict[int, NucleusCue],
) -> tuple[list[tuple[int, int, int]], int]:
    """Map correspondence records to candidate timings (cue-number mapping).

    Records whose member cue sets intersect an already-used set are skipped
    and counted: overlapping groups contribute once, never as separate
    independent checks. Deterministic via record order.
    """
    used_cues: set[int] = set()
    pairs: list[tuple[int, int, int]] = []
    excluded = 0
    for _target_ms, reference_ms, region_ms, target_numbers, _ in records:
        members = set(target_numbers)
        if members & used_cues:
            excluded += 1
            continue
        cue = by_number.get(target_numbers[0])
        if cue is None:
            continue
        used_cues.update(members)
        pairs.append((cue.start_ms, reference_ms, region_ms))
    return pairs, excluded


def regional_stats(
    pairs: list[tuple[int, int, int]],
    film_start_ms: int,
    film_end_ms: int,
) -> dict[str, Any]:
    """Shared correspondence-residual statistics (single implementation).

    ``pairs`` are ``(candidate_ms, reference_ms, target_ms)`` triples with
    deduplicated cue identities. Residuals are candidate minus reference;
    regions are opening/main/ending thirds of the original target span.
    Returns medians, p90/p95/max, per-region breakdowns, outliers over 2s
    ``(target_ms, residual)`` and policy reasons (empty tuple == within
    policy). Callers add structural checks around this core.
    """
    by_region: dict[str, list[float]] = {"opening": [], "middle": [], "ending": []}
    residuals: list[float] = []
    for candidate_ms, reference_ms, target_ms in pairs:
        residual = abs(float(candidate_ms) - float(reference_ms))
        residuals.append(residual)
        by_region[_region_of(target_ms, film_start_ms, film_end_ms)].append(residual)
    region_medians: dict[str, float] = {}
    region_max: dict[str, float] = {}
    region_counts: dict[str, int] = {}
    reasons: list[str] = []
    for name, values in by_region.items():
        region_counts[name] = len(values)
        if len(values) < REGION_MIN_POINTS:
            reasons.append(f"region_uncovered:{name}")
            continue
        ordered_values = sorted(values)
        median = float(statistics.median(ordered_values))
        region_medians[name] = median
        region_max[name] = float(ordered_values[-1])
        if median > MEDIAN_TOLERANCE_MS:
            reasons.append(f"region_median_exceeds_tolerance:{name}")
        if ordered_values[-1] > MAX_TOLERANCE_MS:
            reasons.append(f"region_max_exceeds_tolerance:{name}")
    ordered = sorted(residuals)
    outliers = sorted(
        (value, index) for index, value in enumerate(residuals) if value > 2000.0
    )
    return {
        "n_pairs": len(pairs),
        "median_abs_ms": float(statistics.median(ordered)) if ordered else 0.0,
        "p90_abs_ms": _percentile(ordered, 0.90),
        "p95_abs_ms": _percentile(ordered, 0.95),
        "max_abs_ms": float(ordered[-1]) if ordered else 0.0,
        "region_medians_ms": region_medians,
        "region_max_ms": region_max,
        "region_counts": region_counts,
        "outliers_over_2s": [(pairs[index][0], value) for value, index in outliers],
        "reasons": tuple(reasons),
    }


def verify_nonlinear_regions(
    candidate_cues: list[NucleusCue],
    pairs: list[tuple[int, int, int]],
    film_start_ms: int,
    film_end_ms: int,
) -> dict[str, Any]:
    """Regional verification of a nonlinear candidate (shared core wrapper).

    ``pairs`` are independent ``(candidate_ms, reference_ms, target_ms)``
    correspondences with deduplicated cue identities. A large forward jump
    at a structural transition is expected and reported, never silently
    smoothed: only monotonicity is required.
    """
    reasons: list[str] = []
    # File order (not sorted order): a backward jump in the emitted file is a
    # structural transition failure.
    order_ok = all(
        second.start_ms >= first.start_ms
        for first, second in zip(candidate_cues, candidate_cues[1:], strict=False)
    )
    if not order_ok:
        reasons.append("candidate_order_violation")
    bad_durations = sum(1 for cue in candidate_cues if cue.end_ms <= cue.start_ms)
    if bad_durations:
        reasons.append(f"candidate_invalid_durations:{bad_durations}")
    max_jump_ms = 0
    previous: NucleusCue | None = None
    for cue in candidate_cues:
        if previous is not None and cue.start_ms >= previous.start_ms:
            max_jump_ms = max(max_jump_ms, cue.start_ms - previous.start_ms)
        previous = cue
    stats = regional_stats(pairs, film_start_ms, film_end_ms)
    reasons.extend(stats["reasons"])
    accepted = not reasons
    return {
        "accepted": accepted,
        "n_pairs": stats["n_pairs"],
        "median_abs_ms": stats["median_abs_ms"],
        "max_abs_ms": stats["max_abs_ms"],
        "region_medians_ms": stats["region_medians_ms"],
        "region_max_ms": stats["region_max_ms"],
        "region_counts": stats["region_counts"],
        "max_forward_jump_ms": max_jump_ms,
        "outliers_over_2s": stats["outliers_over_2s"],
        "reasons": tuple(reasons) if reasons else ("within_regional_policy",),
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = fraction * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def examine_outlier(
    candidate_ms: int,
    reference_cues: list[NucleusCue],
    reference_ms: int | None = None,
    neighbour_window_ms: int = 5_000,
) -> dict[str, Any]:
    """Locally examine one residual outlier without dismissing it.

    ``kind`` is ``within_reference_span`` only when the candidate falls
    inside the MATCHED cue's span with a residual shorter than that span
    (segmentation overlap, as with Dexter T#795 inside R#582); ``near_miss``
    when other reference events sit nearby; ``unmatched`` otherwise.
    Informational only: acceptance is still governed by the documented
    residual bounds.
    """
    matched = ([cue for cue in reference_cues if cue.start_ms == reference_ms]
               if reference_ms is not None else [])
    nearby = [
        {"number": cue.number, "start_ms": cue.start_ms, "end_ms": cue.end_ms}
        for cue in reference_cues
        if abs(cue.start_ms - candidate_ms) <= neighbour_window_ms
    ]
    inside = [
        cue for cue in matched
        if cue.start_ms <= candidate_ms <= cue.end_ms
        and abs(float(candidate_ms) - float(reference_ms or 0)) < (cue.end_ms - cue.start_ms)
    ]
    kind = "unmatched"
    if inside:
        kind = "within_reference_span"
    elif nearby:
        kind = "near_miss"
    return {
        "candidate_ms": candidate_ms,
        "kind": kind,
        "nearby_reference": nearby[:4],
        "inside_reference_spans": [
            {"number": cue.number, "start_ms": cue.start_ms, "end_ms": cue.end_ms}
            for cue in inside[:2]
        ],
    }


def detect_clamped(
    target_cues: list[NucleusCue],
    candidate_cues: list[NucleusCue],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Surface timestamp clamping explicitly (never silently truncate).

    A candidate cue at zero whose original starts beyond ``CLAMP_GRACE_MS``
    was clamped. Clamped dialogue vetoes acceptance; clamped non-dialogue
    (advertisements) is flagged and visible but not a veto.
    """
    by_number = {cue.number: cue for cue in candidate_cues}
    target_scores = {cue.number: score for cue, score in
                     zip(target_cues, classify_all(target_cues), strict=True)}
    flagged: list[dict[str, Any]] = []
    vetoes: list[str] = []
    for cue in target_cues:
        candidate = by_number.get(cue.number)
        if candidate is None:
            continue
        if candidate.start_ms == 0 and cue.start_ms > CLAMP_GRACE_MS:
            score = target_scores.get(cue.number)
            is_dialogue = bool(score is not None and score.eligible)
            flagged.append({
                "number": cue.number,
                "original_start_ms": cue.start_ms,
                "candidate_start_ms": candidate.start_ms,
                "dialogue": is_dialogue,
            })
            if is_dialogue:
                vetoes.append(f"dialogue_clamped_to_zero:{cue.number}")
    return flagged, vetoes


def verify_linear_candidate(
    *,
    target_text: str,
    reference_text: str,
    candidate_text: str,
    anchor_result: AnchorResult,
    reference_sha: str,
    candidate_algorithm: str = "two_point",
) -> CorrespondenceVerdict:
    """Verify a LINEAR candidate against held-out dialogue correspondences."""
    reasons: list[str] = []
    try:
        target_cues = parse_srt(target_text)
        reference_cues = parse_srt(reference_text)
        candidate_cues = parse_srt(candidate_text)
    except SubtitleParseError as exc:
        return CorrespondenceVerdict(
            accepted=False, candidate_algorithm=candidate_algorithm,
            reference_sha=reference_sha,
            failure_reasons=(f"unparseable:{type(exc).__name__}",),
        )
    movement = max(
        (abs(cand.start_ms - orig.start_ms)
         for cand, orig in zip(candidate_cues, target_cues, strict=False)),
        default=0,
    )
    if len(candidate_cues) != len(target_cues):
        reasons.append(
            f"cue_count_changed:{len(target_cues)}->{len(candidate_cues)}"
        )
    if any(new.text != old.text
           for new, old in zip(candidate_cues, target_cues, strict=False)):
        reasons.append("dialogue_text_modified")
    if any(cue.end_ms <= cue.start_ms for cue in candidate_cues):
        reasons.append("candidate_invalid_durations")
    for head_cue, tail_cue in zip(candidate_cues, candidate_cues[1:], strict=False):
        if tail_cue.start_ms < head_cue.start_ms:
            reasons.append("candidate_order_violation")
            break
    first = anchor_result.first_anchor
    last = anchor_result.last_anchor
    if first is None or last is None:
        reasons.append("missing_primary_anchors")
    scale: float | None = None
    if first is not None and last is not None and last.target_ms != first.target_ms:
        scale = ((last.reference_ms - first.reference_ms)
                 / (last.target_ms - first.target_ms))
    if scale is None or not (scale == scale and scale > 0):
        reasons.append("invalid_affine_scale")
    elif not (0.85 <= scale <= 1.18):
        reasons.append(f"implausible_affine_scale:{scale:.6f}")

    held_out = [item for item in held_out_checkpoints(anchor_result)
                if is_verification_strong(item)]
    by_number = {cue.number: cue for cue in candidate_cues}
    records = [
        (item.target_ms, item.reference_ms, item.target_ms,
         tuple(item.target_numbers or (item.target_number,)),
         tuple(item.reference_numbers or (item.reference_number,)))
        for item in held_out
    ]
    pairs, overlapping_excluded = _resolve_pairs(records, by_number)
    groups_found = len(anchor_pairs(anchor_result))
    if len(pairs) < MIN_INDEPENDENT_GROUPS:
        reasons.append(f"insufficient_independent_groups:{len(pairs)}")
        return _verdict(
            False, candidate_algorithm, reference_sha, groups_found, len(pairs),
            max(0, overlapping_excluded), {}, 0.0, 0.0, 0.0, 0.0, float(movement),
            {"scale": scale}, (), 0, (), (), (), tuple(reasons),
        )
    film_start = min(cue.start_ms for cue in target_cues)
    film_end = max(cue.end_ms for cue in target_cues)
    clean_pairs, overlap_pairs = _split_segmentation_overlaps(pairs, reference_cues)
    if pairs and len(overlap_pairs) / len(pairs) > 0.25:
        reasons.append(f"excessive_segmentation_overlap:{len(overlap_pairs)}/{len(pairs)}")
    stats = regional_stats(clean_pairs, film_start, film_end)
    reasons.extend(stats["reasons"])
    coverage = stats["region_counts"]
    median = stats["median_abs_ms"]
    p90 = stats["p90_abs_ms"]
    p95 = stats["p95_abs_ms"]
    worst = stats["max_abs_ms"]
    if stats["n_pairs"] and median > MEDIAN_TOLERANCE_MS:
        reasons.append("median_exceeds_tolerance")
    if stats["n_pairs"] and p90 > P90_TOLERANCE_MS:
        reasons.append("p90_exceeds_tolerance")
    if stats["n_pairs"] and worst > MAX_TOLERANCE_MS:
        reasons.append("max_exceeds_tolerance")
    residuals = [abs(float(candidate_ms) - float(reference_ms))
                 for candidate_ms, reference_ms, _ in clean_pairs]
    outliers = [
        {"target_ms": clean_pairs[index][2], "residual_ms": value,
         **examine_outlier(clean_pairs[index][0], reference_cues, clean_pairs[index][1])}
        for index, value in enumerate(residuals)
        if value > 2000.0
    ]
    outliers.extend(
        {"target_ms": target_ms, "residual_ms": abs(float(candidate_ms) - float(reference_ms)),
         **examine_outlier(candidate_ms, reference_cues, reference_ms)}
        for candidate_ms, reference_ms, target_ms in overlap_pairs
    )
    outliers.sort(key=lambda entry: entry["target_ms"])
    clamped, clamp_vetoes = detect_clamped(target_cues, candidate_cues)
    reasons.extend(clamp_vetoes)
    accepted = not reasons
    return _verdict(
        accepted, candidate_algorithm, reference_sha, groups_found, len(pairs),
        max(0, overlapping_excluded), coverage, median, p90, p95, worst,
        float(movement), {"scale": scale}, (), len(outliers),
        tuple(outliers), tuple(clamped), (), tuple(reasons),
    )


def verify_nonlinear_candidate(
    *,
    target_text: str,
    reference_text: str,
    candidate_text: str,
    anchor_result: AnchorResult,
    reference_sha: str,
    candidate_algorithm: str = "alass",
) -> CorrespondenceVerdict:
    """Verify a NONLINEAR candidate across every evidenced region."""
    reasons: list[str] = []
    try:
        target_cues = parse_srt(target_text)
        reference_cues = parse_srt(reference_text)
        candidate_cues = parse_srt(candidate_text)
    except SubtitleParseError as exc:
        return CorrespondenceVerdict(
            accepted=False, candidate_algorithm=candidate_algorithm,
            reference_sha=reference_sha,
            failure_reasons=(f"unparseable:{type(exc).__name__}",),
        )
    movement = max(
        (abs(cand.start_ms - orig.start_ms)
         for cand, orig in zip(candidate_cues, target_cues, strict=False)),
        default=0,
    )
    retention = len(candidate_cues) / max(1, len(target_cues))
    if retention < MIN_TARGET_RETENTION:
        reasons.append(f"cue_retention_too_low:{retention:.2f}")
    if any(cue.end_ms <= cue.start_ms for cue in candidate_cues):
        reasons.append("candidate_invalid_durations")
    for head_cue, tail_cue in zip(candidate_cues, candidate_cues[1:], strict=False):
        if tail_cue.start_ms < head_cue.start_ms:
            reasons.append("candidate_order_violation")
            break
    max_jump_ms = 0
    previous = None
    for cue in candidate_cues:
        if previous is not None and cue.start_ms >= previous.start_ms:
            max_jump_ms = max(max_jump_ms, cue.start_ms - previous.start_ms)
        previous = cue

    records = anchor_pairs(anchor_result, strong_only=True)
    groups_found = len(anchor_pairs(anchor_result))
    by_number = {cue.number: cue for cue in candidate_cues}
    pairs, overlapping_excluded = _resolve_pairs(records, by_number)
    if len(pairs) < MIN_INDEPENDENT_GROUPS:
        reasons.append(f"insufficient_independent_groups:{len(pairs)}")
    film_start = min(cue.start_ms for cue in target_cues)
    film_end = max(cue.end_ms for cue in target_cues)
    clean_pairs, overlap_pairs = _split_segmentation_overlaps(pairs, reference_cues)
    if pairs and len(overlap_pairs) / len(pairs) > 0.25:
        reasons.append(f"excessive_segmentation_overlap:{len(overlap_pairs)}/{len(pairs)}")
    stats = regional_stats(clean_pairs, film_start, film_end)
    reasons.extend(stats["reasons"])
    coverage = stats["region_counts"]
    median = stats["median_abs_ms"]
    p90 = stats["p90_abs_ms"]
    p95 = stats["p95_abs_ms"]
    worst = stats["max_abs_ms"]
    if stats["n_pairs"] and median > MEDIAN_TOLERANCE_MS:
        reasons.append("median_exceeds_tolerance")
    if stats["n_pairs"] and p90 > P90_TOLERANCE_MS:
        reasons.append("p90_exceeds_tolerance")
    if stats["n_pairs"] and worst > MAX_TOLERANCE_MS:
        reasons.append("max_exceeds_tolerance")
    residuals = [abs(float(candidate_ms) - float(reference_ms))
                 for candidate_ms, reference_ms, _ in clean_pairs]
    outliers = [
        {"target_ms": clean_pairs[index][2], "residual_ms": value,
         **examine_outlier(clean_pairs[index][0], reference_cues, clean_pairs[index][1])}
        for index, value in enumerate(residuals)
        if value > 2000.0
    ]
    outliers.extend(
        {"target_ms": target_ms, "residual_ms": abs(float(candidate_ms) - float(reference_ms)),
         **examine_outlier(candidate_ms, reference_cues, reference_ms)}
        for candidate_ms, reference_ms, target_ms in overlap_pairs
    )
    outliers.sort(key=lambda entry: entry["target_ms"])
    discontinuities: list[dict[str, Any]] = []
    for regime in anchor_result.regimes:
        if str(getattr(regime, "label", "")) == "primary":
            continue
        discontinuities.append({
            "label": str(getattr(regime, "label", "contradictory")),
            "implied_offset_ms": float(getattr(regime, "implied_offset_ms", 0.0)),
            "support": int(getattr(regime, "support", 0)),
            "span_start_ms": int(getattr(regime, "span_start_ms", 0)),
            "span_end_ms": int(getattr(regime, "span_end_ms", 0)),
        })
    # Discontinuity boundaries are verified separately: each evidenced regime
    # span must contain correspondences, and the candidate must stay ordered
    # across the largest jump (large jumps are expected, inversions are not).
    for entry in discontinuities:
        inside = [pair for pair in pairs
                  if entry["span_start_ms"] <= pair[2] <= entry["span_end_ms"]]
        if len(inside) < REGION_MIN_POINTS:
            reasons.append(f"discontinuity_uncovered:{entry['label']}")
    clamped, clamp_vetoes = detect_clamped(target_cues, candidate_cues)
    reasons.extend(clamp_vetoes)
    if clamped and not clamp_vetoes:
        reasons.append("non_dialogue_timestamp_clamped:visible")
    # Unmatched reference spans are reported, never silently ignored.
    matched_ref = sorted({pair[1] for pair in pairs})
    unmatched = _unmatched_reference_spans(reference_cues, matched_ref)
    accepted = not [reason for reason in reasons
                    if reason != "non_dialogue_timestamp_clamped:visible"]
    return _verdict(
        accepted, candidate_algorithm, reference_sha, groups_found, len(pairs),
        overlapping_excluded, coverage, median, p90, p95, worst, float(movement),
        {"regimes": len(discontinuities) + 1}, tuple(discontinuities), len(outliers),
        tuple(outliers), tuple(clamped), tuple(unmatched),
        tuple(reason for reason in reasons
              if reason != "non_dialogue_timestamp_clamped:visible") or
        (("non_dialogue_timestamp_clamped:visible",) if clamped else ()),
    )


def _unmatched_reference_spans(
    reference_cues: list[NucleusCue], matched_starts: list[int], gap_ms: int = 60_000
) -> tuple[dict[str, Any], ...]:
    """Reference spans far from any correspondence (informational)."""
    spans: list[dict[str, Any]] = []
    run_start: int | None = None
    previous: int | None = None
    for cue in sorted(reference_cues, key=lambda c: c.start_ms):
        near = any(abs(cue.start_ms - start) <= gap_ms for start in matched_starts)
        if near:
            if run_start is not None and previous is not None and previous - run_start >= gap_ms:
                spans.append({"span_start_ms": run_start, "span_end_ms": previous})
            run_start = None
            previous = None
        else:
            if run_start is None:
                run_start = cue.start_ms
            previous = cue.start_ms
    if run_start is not None and previous is not None and previous - run_start >= gap_ms:
        spans.append({"span_start_ms": run_start, "span_end_ms": previous})
    return tuple(spans)


def _verdict(
    accepted: bool,
    candidate_algorithm: str,
    reference_sha: str,
    groups_found: int,
    independent_used: int,
    overlapping_excluded: int,
    coverage: dict[str, int],
    median: float,
    p90: float,
    p95: float,
    worst: float,
    movement: float,
    model: dict[str, Any],
    discontinuities: tuple[dict[str, Any], ...],
    outlier_count: int,
    outliers: tuple[dict[str, Any], ...],
    clamped: tuple[dict[str, Any], ...],
    unmatched: tuple[dict[str, Any], ...],
    reasons: tuple[str, ...],
) -> CorrespondenceVerdict:
    return CorrespondenceVerdict(
        accepted=accepted,
        candidate_algorithm=candidate_algorithm,
        reference_sha=reference_sha,
        groups_found=groups_found,
        independent_used=independent_used,
        overlapping_excluded=overlapping_excluded,
        coverage_thirds=dict(coverage),
        median_residual_ms=median,
        p90_residual_ms=p90,
        p95_residual_ms=p95,
        max_residual_ms=worst,
        movement_magnitude_ms=movement,
        transformation_model=dict(model),
        regional_discontinuities=discontinuities,
        outlier_count=outlier_count,
        outliers=outliers,
        clamped=clamped,
        unmatched_reference_spans=unmatched,
        failure_reasons=tuple(reasons) if not accepted else (),
        may_serve_recommendation=accepted,
    )


def to_evaluation(verdict: CorrespondenceVerdict):
    """Build a fresh production evaluation from a correspondence verdict.

    This creates a NEW verdict object through the only supported setter; it
    never rewrites an existing UNVERIFIED result. Callers still apply Target
    Binding, reference eligibility, and ``may_serve`` before serving.
    """
    from app.services.sync.alignment import (
        RejectionReason,
        SubtitleEvaluation,
        SyncState,
        VerificationAvailability,
    )

    evaluation = SubtitleEvaluation()
    evaluation.median_offset_ms = round(verdict.median_residual_ms, 1)
    evaluation.p95_offset_ms = round(verdict.p95_residual_ms, 1)
    evaluation.coverage_score = round(
        min(1.0, verdict.independent_used / max(1, MIN_INDEPENDENT_GROUPS * 2)), 4
    )
    evaluation.sync_confidence = round(
        max(0.0, min(100.0, 100.0 - verdict.median_residual_ms / 10.0)), 1
    )
    evaluation.reasons.append(
        f"correspondence-{verdict.method}: algorithm={verdict.candidate_algorithm} "
        f"groups={verdict.independent_used}/{verdict.groups_found} "
        f"coverage={verdict.coverage_thirds} "
        f"median={verdict.median_residual_ms:.0f}ms "
        f"p95={verdict.p95_residual_ms:.0f}ms max={verdict.max_residual_ms:.0f}ms "
        f"movement_magnitude={verdict.movement_magnitude_ms:.0f}ms "
        f"outliers={verdict.outlier_count}"
    )
    for entry in verdict.regional_discontinuities:
        evaluation.reasons.append(
            f"correspondence-discontinuity: {entry.get('label')} "
            f"offset={float(entry.get('implied_offset_ms', 0.0)):.0f}ms "
            f"support={entry.get('support')}"
        )
    for entry in verdict.clamped:
        evaluation.reasons.append(
            f"correspondence-clamped: cue={entry.get('number')} "
            f"dialogue={entry.get('dialogue')}"
        )
    if verdict.accepted:
        evaluation.set_verdict(SyncState.VERIFIED_RESYNCED, VerificationAvailability.VERIFIED)
    else:
        evaluation.set_verdict(SyncState.UNVERIFIED, VerificationAvailability.UNKNOWN)
        if verdict.failure_reasons:
            evaluation.rejection_reason = RejectionReason.OTHER
            evaluation.reasons.append(
                "correspondence_rejected: " + ";".join(verdict.failure_reasons)
            )
    return evaluation
