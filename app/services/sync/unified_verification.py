"""Dialogue timing measurement for identity-bound, independent references.

The orchestrator owns authoritative release identity and ReferencePolicy.
This module measures corresponding dialogue once, regardless of the algorithm
that proposed the timestamps. Structure and fitted scale are not verdicts.
Existing real-media tolerances: median 750ms, P90 2000ms, maximum 3000ms,
at least five independent groups. Episode thirds are diagnostic only.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from app.services.sync.correspondence_verification import (
    MAX_TOLERANCE_MS,
    MEDIAN_TOLERANCE_MS,
    MIN_INDEPENDENT_GROUPS,
    MIN_TARGET_RETENTION,
    P90_TOLERANCE_MS,
    detect_clamped,
    regional_stats,
)
from app.services.sync.reference_content_identity import identical_content
from two_point_sync.anchors import dialogue_correspondences
from two_point_sync.parser import (
    Cue as NucleusCue,
)
from two_point_sync.parser import (
    SubtitleParseError,
    _split_blocks,
    parse_srt_for_verification,
)

METHOD = "unified"
VERSION = "timing-v3"
CONSECUTIVE_ERROR_RUN = 3
OUTLIER_THRESHOLD_MS = 2000.0


@dataclass(frozen=True)
class UnifiedVerdict:
    """One authoritative verdict with its supporting evidence (§8)."""

    verification_method: str = METHOD
    verification_version: str = VERSION
    candidate_algorithm: str = ""
    candidate_sha256: str = ""
    target_identity: dict[str, Any] = field(default_factory=dict)
    reference_identity: dict[str, Any] = field(default_factory=dict)
    integrity_result: dict[str, Any] = field(default_factory=dict)
    correspondence_result: dict[str, Any] = field(default_factory=dict)
    temporal_result: dict[str, Any] = field(default_factory=dict)
    structural_result: dict[str, Any] = field(default_factory=dict)
    regional_result: dict[str, Any] = field(default_factory=dict)
    risk_findings: tuple[dict[str, Any], ...] = ()
    independent_evidence_count: int = 0
    coverage: dict[str, int] = field(default_factory=dict)
    residual_statistics: dict[str, float] = field(default_factory=dict)
    failure_reasons: tuple[str, ...] = ()
    proposed_verdict: str = "UNVERIFIED"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _identity(cues: list[NucleusCue], text: str) -> dict[str, Any]:
    return {
        "sha256": _sha(text),
        "cues": len(cues),
        "span_start_ms": min(c.start_ms for c in cues) if cues else 0,
        "span_end_ms": max(c.end_ms for c in cues) if cues else 0,
    }


def _script_profile(text: str) -> dict[str, float]:
    """Share of Arabic- vs Latin-script letters in dialogue text."""
    arabic = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")
    latin = sum(1 for ch in text if "A" <= ch <= "Z" or "a" <= ch <= "z")
    total = max(1, arabic + latin)
    return {"arabic": arabic / total, "latin": latin / total}


def _scripts_differ(target_text: str, reference_text: str) -> bool:
    """True when the two sides' dominant dialogue scripts differ."""
    target = _script_profile(target_text)
    reference = _script_profile(reference_text)
    dominant = lambda profile: "arabic" if profile["arabic"] >= profile["latin"] else "latin"  # noqa: E731
    return dominant(target) != dominant(reference)




def _map_to_candidate(
    records: list[tuple[int, int, int, tuple, tuple]],
    candidate_cues: list[NucleusCue],
) -> tuple[list[tuple[int, int, int]], int]:
    """Map records to candidate timings with overlap deduplication.

    Returns ``(pairs, excluded)`` where pairs are
    ``(candidate_ms, reference_ms, region_ms)``.
    """
    by_number = {cue.number: cue for cue in candidate_cues}
    used: set[int] = set()
    pairs: list[tuple[int, int, int]] = []
    excluded = 0
    for _, reference_ms, region_ms, target_numbers, _ in records:
        members = set(target_numbers)
        if members & used:
            excluded += 1
            continue
        cue = by_number.get(target_numbers[0])
        if cue is None:
            continue
        used.update(members)
        pairs.append((cue.start_ms, reference_ms, region_ms))
    return pairs, excluded


def _consecutive_error_run(pairs: list[tuple[int, int, int]],
                           residuals: list[float]) -> int:
    """Longest target-ordered run of residuals over the outlier threshold."""
    ordered = sorted(zip([p[2] for p in pairs], residuals, strict=True),
                     key=lambda entry: entry[0])
    longest = current = 0
    for _, residual in ordered:
        if residual > OUTLIER_THRESHOLD_MS:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _check_integrity(target_cues: list[NucleusCue],
                     candidate_cues: list[NucleusCue]) -> dict[str, Any]:
    """Integrity Verification: retention, order, durations, text identity.

    Original cue text must survive exactly. ALASS working-copy formatting is
    restored by timing reconstruction before reaching this boundary.
    """
    reasons: list[str] = []
    retention = len(candidate_cues) / max(1, len(target_cues))
    if retention < MIN_TARGET_RETENTION:
        reasons.append(f"cue_retention_too_low:{retention:.2f}")
    order_ok = all(
        second.start_ms >= first.start_ms
        for first, second in zip(candidate_cues, candidate_cues[1:], strict=False))
    inherited_disorder = 0
    if not order_ok:
        # Provenance check: disorder the candidate inherited verbatim from
        # the target is a source formatting artifact (real broadcast rips
        # ship misordered adjacent cues), not pipeline scrambling. Only
        # newly introduced disorder rejects; inherited disorder is recorded
        # as telemetry. Positional comparison is sound here because cue
        # order is the sequence ALASS must preserve.
        inherited_disorder = sum(
            1
            for (old_first, old_second), (new_first, new_second) in zip(
                zip(target_cues, target_cues[1:], strict=False),
                zip(candidate_cues, candidate_cues[1:], strict=False),
                strict=False,
            )
            if new_second.start_ms < new_first.start_ms
            and old_second.start_ms < old_first.start_ms
        )
        total_disorder = sum(
            1
            for first, second in zip(candidate_cues, candidate_cues[1:], strict=False)
            if second.start_ms < first.start_ms
        )
        if total_disorder > inherited_disorder:
            reasons.append("candidate_order_violation")
    bad_durations = sum(1 for cue in candidate_cues if cue.end_ms <= cue.start_ms)
    if bad_durations:
        reasons.append(f"candidate_invalid_durations:{bad_durations}")
    modified = sum(
        1 for new, old in zip(candidate_cues, target_cues, strict=False)
        if new.text != old.text)
    if len(candidate_cues) != len(target_cues):
        reasons.append("dialogue_count_changed")
    if modified:
        reasons.append(f"dialogue_text_modified:{modified}")
    return {
        "accepted": not reasons,
        "retention": retention,
        "order_ok": order_ok,
        "inherited_order_violations": inherited_disorder,
        "bad_durations": bad_durations,
        "modified_texts": modified,
        "reasons": tuple(reasons),
    }


def to_evaluation(verdict: UnifiedVerdict):
    """Build a fresh production evaluation through the only supported setter.

    Never rewrites an existing verdict object.
    """
    from app.services.sync.alignment import (
        RejectionReason,
        SubtitleEvaluation,
        SyncState,
        VerificationAvailability,
    )

    evaluation = SubtitleEvaluation()
    evaluation.structural_similarity = verdict.structural_result.get("score")
    evaluation.drift_ms_per_minute = verdict.temporal_result.get("drift_ms_per_minute")
    evaluation.cut_verdict = verdict.temporal_result.get("cut", {}).get("cut")
    evaluation.median_offset_ms = round(verdict.residual_statistics.get("median", 0.0), 1)
    evaluation.p95_offset_ms = round(verdict.residual_statistics.get("p95", 0.0), 1)
    evaluation.coverage_score = round(
        min(1.0, verdict.independent_evidence_count / max(1, MIN_INDEPENDENT_GROUPS * 2)), 4)
    evaluation.sync_confidence = round(
        max(0.0, min(100.0, 100.0 - verdict.residual_statistics.get("median", 0.0) / 10.0)), 1)
    evaluation.reasons.append(
        f"unified-{VERSION}: groups={verdict.independent_evidence_count} "
        f"coverage={verdict.coverage} "
        f"median={verdict.residual_statistics.get('median', 0.0):.0f}ms "
        f"p95={verdict.residual_statistics.get('p95', 0.0):.0f}ms "
        f"max={verdict.residual_statistics.get('max', 0.0):.0f}ms "
        "movement_magnitude="
        f"{verdict.temporal_result.get('movement_magnitude_ms', 0.0):.0f}ms "
        f"outliers={verdict.temporal_result.get('outlier_count', 0)}"
    )
    if verdict.regional_result.get("scope"):
        evaluation.reasons.append(
            f"dialogue_scope: {verdict.regional_result}; unmeasured or conflicting "
            "dialogue outside the supported span is not verified")
    if verdict.proposed_verdict == "VERIFIED":
        evaluation.set_verdict(SyncState.VERIFIED_RESYNCED, VerificationAvailability.VERIFIED)
    elif verdict.proposed_verdict == "REJECTED":
        evaluation.set_verdict(SyncState.UNVERIFIED, VerificationAvailability.UNKNOWN)
        evaluation.rejection_reason = RejectionReason.OTHER
        if verdict.failure_reasons:
            evaluation.reasons.append("unified_rejected: " + ";".join(verdict.failure_reasons))
    else:
        evaluation.set_verdict(SyncState.UNVERIFIED, VerificationAvailability.UNKNOWN)
        if verdict.failure_reasons:
            evaluation.rejection_reason = RejectionReason.OTHER
            evaluation.reasons.append(
                "unified_unverified: " + ";".join(verdict.failure_reasons))
    return evaluation


def cache_identity(*, target_text: str, reference_text: str, candidate_text: str,
                   video_binding: dict[str, Any] | None = None) -> dict[str, Any]:
    """Methodology-segmented cache identity (design per audit §11/§13).

    Pure function, no cache access: binds methodology + version + document
    identities so old Legacy/correspondence-v1 decisions can never be
    reinterpreted as unified decisions. Video-scoped references are keyed
    independently of verdicts (callers preserve them separately).
    """
    return {
        "verification_method": METHOD,
        "verification_version": VERSION,
        "target_sha256": _sha(target_text),
        "reference_sha256": _sha(reference_text),
        "candidate_sha256": _sha(candidate_text),
        "video_binding": dict(video_binding or {}),
    }


@lru_cache(maxsize=12)
def _prepared(text: str):
    """Bounded immutable parse reuse; never caches a video authorization."""
    cues, report = parse_srt_for_verification(text, exclude_ambiguous=True)
    assert cues is not None
    blocks = _split_blocks(text.lstrip("\ufeff"))
    unmeasurable = tuple(blocks[index - 1] for index in report.ambiguous)
    unmeasurable += tuple(blocks[index - 1] for index, kind in report.excluded
                          if kind == "non_positive_duration")
    return tuple(cues), report, unmeasurable


@lru_cache(maxsize=8)
def _correspondence(target: tuple[NucleusCue, ...], reference: tuple[NucleusCue, ...]):
    """Reuse text correspondence across ORIGINAL and candidate measurements."""
    return dialogue_correspondences(list(target), list(reference))


def verify_candidate(target_text: str, reference_text: str, candidate_text: str,
                     *, candidate_algorithm: str = "") -> UnifiedVerdict:
    """Measure timing; release identity and trust remain mandatory at delivery.

    Unplaceable source blocks contribute no evidence. They must survive verbatim
    in the output; guessing their cue ownership is never allowed.
    """
    common: dict[str, Any] = {
        "candidate_algorithm": candidate_algorithm,
        "candidate_sha256": _sha(candidate_text),
    }
    if identical_content(target_text, reference_text):
        return UnifiedVerdict(**common,
                              failure_reasons=("reference_content_identical_to_target",))
    try:
        target, target_report, target_unknown = _prepared(target_text)
        reference, _, _ = _prepared(reference_text)
        candidate, candidate_report, candidate_unknown = _prepared(candidate_text)
    except SubtitleParseError:
        return UnifiedVerdict(**common, failure_reasons=("unparseable:SubtitleParseError",))
    integrity = _check_integrity(list(target), list(candidate))
    reasons = list(integrity["reasons"])
    if target_unknown != candidate_unknown:
        reasons.append("unmeasurable_dialogue_modified")
    integrity.update(accepted=not reasons, reasons=tuple(reasons),
                     target_excluded_textless=len(target_report.excluded),
                     candidate_excluded_textless=len(candidate_report.excluded),
                     unmeasurable_blocks=len(target_unknown))
    common.update(target_identity=_identity(list(target), target_text),
                  reference_identity=_identity(list(reference), reference_text),
                  integrity_result=integrity)
    if reasons:
        return UnifiedVerdict(**common, proposed_verdict="REJECTED",
                              failure_reasons=tuple(reasons))
    try:
        all_detection = _correspondence(target, reference)
        # Conservatively withhold endpoint events used by Two-Point fitting.
        detection = all_detection[1:-1]
    except Exception as exc:
        return UnifiedVerdict(**common,
                              failure_reasons=(f"correspondence_error:{type(exc).__name__}",))
    pairs, excluded = _map_to_candidate(list(detection), list(candidate))
    clean = pairs
    overlaps: list[tuple[int, int, int]] = []
    stats = regional_stats(clean, min(c.start_ms for c in target), max(c.end_ms for c in target))
    common.update(
        correspondence_result={"route": "dialogue_timing", "groups": len(pairs),
                               "independent": len(clean), "overlapping_excluded": excluded,
                               "segmentation_unmeasurable": len(overlaps)},
        independent_evidence_count=len(clean), coverage=stats["region_counts"],
        residual_statistics={name: stats[f"{name}_abs_ms"] for name in ("median", "p90", "p95", "max")},
        regional_result={"regions": stats["region_counts"], "medians": stats["region_medians_ms"]},
        temporal_result={"route": "dialogue_timing", "movement_magnitude_ms": max(
            abs(a.start_ms - b.start_ms) for a, b in zip(target, candidate, strict=True))},
    )
    # Missing dialogue is absence of evidence, never evidence of mistiming.
    # Report the measured span explicitly; never claim unmeasured cues align.
    insufficient = []
    if len(clean) < MIN_INDEPENDENT_GROUPS:
        insufficient.append(f"insufficient_independent_groups:{len(clean)}")
    if insufficient:
        if _scripts_differ(target_text, reference_text):
            insufficient.append("cross_language_insufficient_evidence")
        return UnifiedVerdict(**common, failure_reasons=tuple(insufficient))
    ordered = sorted(clean, key=lambda pair: pair[2])
    # A different-cut head/tail may contradict an otherwise aligned episode.
    # Only trim consecutive *edge* errors, never interior contradictions, and
    # only when the retained evidence dominates and spans the main episode.
    first, last = 0, len(ordered)
    while first < last and abs(ordered[first][0] - ordered[first][1]) > MAX_TOLERANCE_MS:
        first += 1
    while last > first and abs(ordered[last - 1][0] - ordered[last - 1][1]) > MAX_TOLERANCE_MS:
        last -= 1
    supported = ordered[first:last]
    span = max(c.end_ms for c in target) - min(c.start_ms for c in target)
    evidence_span = supported[-1][2] - supported[0][2] if supported else 0
    common["regional_result"].update(
        scope="corresponding_dialogue_only",
        supported_start_ms=supported[0][2] if supported else None,
        supported_end_ms=supported[-1][2] if supported else None,
        conflicting_edge_groups=first + len(ordered) - last,
        unmeasured_cues=len(target) - len({n for record in detection for n in record[3]}),
        global_alignment_claim=False,
    )
    # Require sustained evidence, rather than five nearby utterances. A minute
    # of measured dialogue (half the span for very short subtitles) is enough
    # to verify that measured span, never to claim the entire episode aligns.
    if len(supported) < len(clean) * 0.8:
        return UnifiedVerdict(**common, proposed_verdict="REJECTED",
                              failure_reasons=("conflicting_dialogue_timing",))
    if len(supported) < MIN_INDEPENDENT_GROUPS or evidence_span < min(60_000, span * 0.5):
        return UnifiedVerdict(**common, failure_reasons=("insufficient_main_dialogue_support",))
    supported_stats = regional_stats(supported, 0, max(1, span))
    common["temporal_result"]["supported_residual_statistics"] = {
        name: supported_stats[f"{name}_abs_ms"] for name in ("median", "p90", "p95", "max")}
    reasons = []
    for metric, limit in (("median", MEDIAN_TOLERANCE_MS), ("p90", P90_TOLERANCE_MS),
                          ("max", MAX_TOLERANCE_MS)):
        if supported_stats[f"{metric}_abs_ms"] > limit:
            reasons.append(f"{metric}_exceeds_tolerance")
    residuals = [abs(float(c) - float(r)) for c, r, _ in supported]
    if _consecutive_error_run(supported, residuals) >= CONSECUTIVE_ERROR_RUN:
        reasons.append("concentrated_error_cluster")
    _, clamp_reasons = detect_clamped(list(target), list(candidate))
    # Zero is a valid timestamp. Movement to zero is not itself proof of
    # clamping: allow it when genuine matching reference dialogue starts there.
    # Endpoint events establish this integrity fact without counting as held-out
    # timing evidence; an unexplained or actually mistimed collapse still fails.
    supported_zero = {record[3][0] for record in all_detection
                      if abs(record[1]) <= MEDIAN_TOLERANCE_MS}
    reasons.extend(reason for reason in clamp_reasons
                   if int(reason.rsplit(":", 1)[-1]) not in supported_zero)
    return UnifiedVerdict(**common, failure_reasons=tuple(reasons),
                          proposed_verdict="REJECTED" if reasons else "VERIFIED")
