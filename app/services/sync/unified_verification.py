"""Unified Production Verifier — Shadow mode (Phase 2).

Combines the strongest proven capabilities of the Legacy analyzer and
Correspondence Verifier V2 into one authoritative measurement, without
transferring live serving authority (that migration is Phase 3 and is NOT
part of this module: nothing here serves, caches, discovers references, or
runs ALASS).

Architecture (audit §9):

    Original Target + Synced Candidate + Eligible Reference
          │  (single immutable parse per document; production-tuple views
          │   adapted from it — computational reuse, never evidentiary reuse)
          v
    Integrity Verification ..... retention≥0.80, order, durations, text
    │                             identity, clamp detector
    ├── Dialogue Correspondence . shared nucleus layer (units/groups/markers/
    │                             regimes) + held-out/fit separation + dedupe
    │                             + strong-evidence bar
    ├── Temporal Verification
    │       ├── Linear / FPS Drift  correspondence residuals + drift
    │       │                        least-squares + scale plausibility +
    │       │                        landmark corroboration (non-rejecting)
    │       └── Nonlinear Regions .. thirds + regime spans + boundary checks
    │                                + change-point evidence (non-rejecting)
    ├── Structural Verification  similarity score + cut verdict
    │                             (corroboration, non-rejecting — Legacy role)
    └── Risk & Evidence Assessment  outliers, consecutive-error runs,
                                     unmatched spans, movement magnitude,
                                     evidence sufficiency
          v
    ONE AUTHORITATIVE VERDICT (VERIFIED / REJECTED / UNVERIFIED + method tag)

Corrected measurements (audit §5):
- Residuals are measured on justified dialogue correspondences, never on
  greedy nearest-start pairs across differing segmentation.
- Movement magnitude is telemetry only; it is never judged as error, and a
  near-zero MAD never certifies constancy (regime scan governs instead).
- Tail rule is P90 (nucleus-policy alignment, §8 record in
  correspondence_verification); P95 stays reported telemetry.

Threshold basis (§8: documented, adversarially tested, fail-closed):
- Correspondence residual median ≤750ms, P90 ≤2000ms, max ≤3000ms
  (reused V2 bounds; units are verified-dialogue timing residuals).
- Every populated film third holds ≥2 independent correspondences;
  at least 5 independent groups (serving bar above detection floor).
- Consecutive run of >2000ms residuals ≥3 rejects (concentrated-cluster
  guard: a thin minority of adjacent mistimed events must not hide behind
  a clean majority).
- Linear drift: |least-squares slope| ≤120 ms/min on correspondence
  residuals (reused Legacy bound, new domain); ≥5000ms change-point step
  on a linear fit rejects as hidden discontinuity.
- Structural/landmark/change-point evidence corroborates only.
- Anything short fails closed: UNVERIFIED (insufficient evidence) or
  REJECTED (positive evidence of wrongness).
"""

from __future__ import annotations

import hashlib
import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from app.services.sync.alignment import (
    MAX_DRIFT_MS_PER_MINUTE,
    MAX_P95_MS_FOR_STABLE,
    MAX_PLAUSIBLE_OFFSET_MS,
    analyze_drift,
    detect_change_points,
)
from app.services.sync.correspondence_verification import (
    MAX_TOLERANCE_MS,
    MEDIAN_TOLERANCE_MS,
    MIN_INDEPENDENT_GROUPS,
    MIN_TARGET_RETENTION,
    P90_TOLERANCE_MS,
    REGION_MIN_POINTS,
    _split_segmentation_overlaps,
    _unmatched_reference_spans,
    anchor_pairs,
    detect_clamped,
    examine_outlier,
    held_out_checkpoints,
    is_verification_strong,
    regional_stats,
)
from app.services.sync.reference_content_identity import identical_content
from app.services.sync.structural import classify_cut, compare_structures
from app.services.sync.timeline_affine import Timeline, verify_timeline
from two_point_sync.anchors import (
    MAX_SCALE,
    MIN_SCALE,
    AnchorResult,
    detect_anchors,
    normalize_text,
)
from two_point_sync.parser import Cue as NucleusCue
from two_point_sync.parser import SubtitleParseError, parse_srt
from two_point_sync.sanitizer import classify_all
from two_point_sync.transform import fit_from_anchors

logger = logging.getLogger(__name__)

#: Methodology identity (§9/§11): family tag + pinned version. Cache keys
#: must segment on these (see ``cache_identity``); old methodologies must
#: never be reinterpreted as unified decisions.
METHOD = "unified"
VERSION = "unified-v1"

#: Consecutive correspondences (target-time order) with residuals over
#: 2000ms that reject: a concentrated cluster of mistimed events.
CONSECUTIVE_ERROR_RUN = 3

#: Residual magnitude that marks an outlier entry (listed, examined).
OUTLIER_THRESHOLD_MS = 2000.0

#: Overlap fraction above which segmentation ambiguity fails closed.
OVERLAP_FRACTION_LIMIT = 0.25


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


def _as_production_tuples(cues: list[NucleusCue]) -> list[tuple[int, int, str]]:
    """Adapt parsed cues to the (start, end, text) view Legacy functions take.

    Computational reuse of one immutable parse; evidentiary roles (fit vs
    verify) are tracked separately and never shared.
    """
    return [(c.start_ms, c.end_ms, c.text) for c in cues]


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


def _strong_records(anchor_result: AnchorResult) -> list[tuple[int, int, int, tuple, tuple]]:
    """Held-out, signature-deduplicated, strong-only correspondence records.

    Fitting anchors are excluded by signature (held-out rule); overlapping
    groups collapse to one record; weak bare singletons never verify.
    """
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    records: list[tuple[int, int, int, tuple, tuple]] = []
    for item in held_out_checkpoints(anchor_result):
        signature = (tuple(item.target_numbers or (item.target_number,)),
                     tuple(item.reference_numbers or (item.reference_number,)))
        if signature in seen:
            continue
        if not is_verification_strong(item):
            continue
        seen.add(signature)
        records.append((item.target_ms, item.reference_ms, item.target_ms,
                        signature[0], signature[1]))
    records.sort(key=lambda record: (record[0], record[1]))
    return records


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

    Dialogue text is compared normalized: cosmetic whitespace changes (ALASS
    strips a leading space on some cues) are not tampering, while any real
    wording change is.
    """
    reasons: list[str] = []
    retention = len(candidate_cues) / max(1, len(target_cues))
    if retention < MIN_TARGET_RETENTION:
        reasons.append(f"cue_retention_too_low:{retention:.2f}")
    order_ok = all(
        second.start_ms >= first.start_ms
        for first, second in zip(candidate_cues, candidate_cues[1:], strict=False))
    if not order_ok:
        reasons.append("candidate_order_violation")
    bad_durations = sum(1 for cue in candidate_cues if cue.end_ms <= cue.start_ms)
    if bad_durations:
        reasons.append(f"candidate_invalid_durations:{bad_durations}")
    modified = sum(
        1 for new, old in zip(candidate_cues, target_cues, strict=False)
        if normalize_text(new.text) != normalize_text(old.text))
    if len(candidate_cues) != len(target_cues) or modified:
        # Count drift alone is not proof of tampering (ALASS may legitimately
        # re-split); text modification always is.
        if modified:
            reasons.append(f"dialogue_text_modified:{modified}")
    return {
        "accepted": not reasons,
        "retention": retention,
        "order_ok": order_ok,
        "bad_durations": bad_durations,
        "modified_texts": modified,
        "reasons": tuple(reasons),
    }


def _check_structural(candidate_cues: list[NucleusCue],
                      reference_cues: list[NucleusCue]) -> dict[str, Any]:
    """Structural Verification: similarity + cut verdict (corroboration only).

    Never rejects on its own: providers legitimately split cues differently,
    so a structural mismatch is evidence to report, not a verdict to impose.
    """
    candidate_t = _as_production_tuples(candidate_cues)
    reference_t = _as_production_tuples(reference_cues)
    try:
        similarity = compare_structures(candidate_t, reference_t)
    except Exception as exc:  # fail soft: corroboration must not veto
        return {"available": False, "reasons": (f"structural_unavailable:{type(exc).__name__}",)}
    return {
        "available": True,
        "score": similarity.score,
        "same_structure": similarity.same_structure,
        "cue_count_ratio": similarity.cue_count_ratio,
        "duration_ratio": similarity.duration_ratio,
        "density_ratio": similarity.density_ratio,
        "reasons": tuple(similarity.reasons),
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


def verify_candidate(
    target_text: str,
    reference_text: str,
    candidate_text: str,
    *,
    candidate_algorithm: str = "",
) -> UnifiedVerdict:
    """One authoritative verification of a synced candidate (shadow-safe).

    Pure measurement: never serves, caches, discovers references, or runs
    ALASS. Fail-closed: anything short of independently supported
    correctness is UNVERIFIED (insufficient evidence) or REJECTED (positive
    evidence of wrongness).
    """
    started = time.monotonic()
    try:
        target_cues = parse_srt(target_text)
        reference_cues = parse_srt(reference_text)
        candidate_cues = parse_srt(candidate_text)
    except SubtitleParseError as exc:
        return UnifiedVerdict(
            candidate_algorithm=candidate_algorithm,
            candidate_sha256=_sha(candidate_text),
            target_identity={"parse": "failed"},
            reference_identity={"parse": "failed"},
            failure_reasons=(f"unparseable:{type(exc).__name__}",),
            proposed_verdict="UNVERIFIED",
        )
    target_identity = _identity(target_cues, target_text)
    reference_identity = _identity(reference_cues, reference_text)

    if identical_content(target_text, reference_text):
        return UnifiedVerdict(
            candidate_algorithm=candidate_algorithm,
            candidate_sha256=_sha(candidate_text),
            target_identity=target_identity,
            reference_identity=reference_identity,
            failure_reasons=("reference_content_identical_to_target",),
            proposed_verdict="UNVERIFIED",
        )

    integrity = _check_integrity(target_cues, candidate_cues)
    structural = _check_structural(candidate_cues, reference_cues)
    movement_magnitude = max(
        (abs(cand.start_ms - orig.start_ms)
         for cand, orig in zip(candidate_cues, target_cues, strict=False)),
        default=0,
    )

    try:
        detection = detect_anchors(
            target_cues, reference_cues,
            target_scores=classify_all(target_cues),
            reference_scores=classify_all(reference_cues),
        )
    except Exception as exc:
        return UnifiedVerdict(
            candidate_algorithm=candidate_algorithm,
            candidate_sha256=_sha(candidate_text),
            target_identity=target_identity,
            reference_identity=reference_identity,
            integrity_result=integrity,
            structural_result=structural,
            failure_reasons=(f"correspondence_error:{type(exc).__name__}",),
            proposed_verdict="UNVERIFIED",
        )

    if "NONLINEAR_DISCONTINUITY" in detection.reasons:
        return _verify_nonlinear(
            target_cues, reference_cues, candidate_cues, detection,
            target_identity, reference_identity, integrity, structural,
            movement_magnitude, candidate_algorithm, candidate_text, started)
    if detection.accepted and detection.first_anchor is not None \
            and detection.last_anchor is not None:
        return _verify_linear(
            target_cues, reference_cues, candidate_cues, detection,
            target_identity, reference_identity, integrity, structural,
            movement_magnitude, candidate_algorithm, candidate_text, started)
    if _scripts_differ(target_text, reference_text):
        return UnifiedVerdict(
            candidate_algorithm=candidate_algorithm,
            candidate_sha256=_sha(candidate_text),
            target_identity=target_identity,
            reference_identity=reference_identity,
            integrity_result=integrity,
            correspondence_result={"route": "cross_language_abstain"},
            structural_result=structural,
            failure_reasons=("cross_language_insufficient_evidence",),
            proposed_verdict="UNVERIFIED",
        )
    return UnifiedVerdict(
        candidate_algorithm=candidate_algorithm,
        candidate_sha256=_sha(candidate_text),
        target_identity=target_identity,
        reference_identity=reference_identity,
        integrity_result=integrity,
        correspondence_result={"route": "insufficient"},
        structural_result=structural,
        failure_reasons=("insufficient_correspondence_evidence",),
        proposed_verdict="UNVERIFIED",
    )


def _common_pairs(anchor_result: AnchorResult,
                  candidate_cues: list[NucleusCue]
                  ) -> tuple[list[tuple[int, int, int]], int, int]:
    """Shared pair construction: strong held-out records mapped to candidate.

    Returns ``(pairs, groups_found, overlapping_excluded)``.
    """
    records = _strong_records(anchor_result)
    pairs, excluded = _map_to_candidate(records, candidate_cues)
    return pairs, len(anchor_pairs(anchor_result)), excluded


def _risk_outliers(pairs: list[tuple[int, int, int]],
                   clean: list[tuple[int, int, int]],
                   overlapped: list[tuple[int, int, int]],
                   reference_cues: list[NucleusCue]) -> list[dict[str, Any]]:
    """All >2000ms deviations listed with local examination (never hidden)."""
    _ = pairs
    entries = []
    for candidate_ms, reference_ms, target_ms in clean:
        residual = abs(float(candidate_ms) - float(reference_ms))
        if residual > OUTLIER_THRESHOLD_MS:
            entries.append({"target_ms": target_ms, "residual_ms": residual,
                            **examine_outlier(candidate_ms, reference_cues, reference_ms)})
    for candidate_ms, reference_ms, target_ms in overlapped:
        entries.append({"target_ms": target_ms,
                        "residual_ms": abs(float(candidate_ms) - float(reference_ms)),
                        **examine_outlier(candidate_ms, reference_cues, reference_ms)})
    entries.sort(key=lambda entry: entry["target_ms"])
    return entries


def _verify_linear(
    target_cues: list[NucleusCue],
    reference_cues: list[NucleusCue],
    candidate_cues: list[NucleusCue],
    detection: AnchorResult,
    target_identity: dict[str, Any],
    reference_identity: dict[str, Any],
    integrity: dict[str, Any],
    structural: dict[str, Any],
    movement_magnitude: int,
    candidate_algorithm: str,
    candidate_text: str,
    started: float,
) -> UnifiedVerdict:
    """Linear / FPS-drift branch."""
    reasons: list[str] = []
    reasons.extend(integrity.get("reasons", ()))
    first = detection.first_anchor
    last = detection.last_anchor
    scale: float | None = None
    if first is not None and last is not None and last.target_ms != first.target_ms:
        try:
            fitted = fit_from_anchors(first.target_ms, first.reference_ms,
                                      last.target_ms, last.reference_ms)
            scale = fitted.scale
        except Exception:
            scale = None
    if scale is None or not (scale == scale and scale > 0):
        reasons.append("invalid_affine_scale")
        return _unified_reject("linear", scale, candidate_algorithm, candidate_text,
                               target_identity, reference_identity, integrity, structural,
                               movement_magnitude, reasons, started, {}, [], 0, 0)
    if not (MIN_SCALE <= scale <= MAX_SCALE):
        reasons.append(f"implausible_affine_scale:{scale:.6f}")
        return _unified_reject("linear", scale, candidate_algorithm, candidate_text,
                               target_identity, reference_identity, integrity, structural,
                               movement_magnitude, reasons, started, {}, [], 0, 0)
    pairs, groups_found, excluded = _common_pairs(detection, candidate_cues)
    if len(pairs) < MIN_INDEPENDENT_GROUPS:
        reasons.append(f"insufficient_independent_groups:{len(pairs)}")
        return _unified_reject("linear", scale, candidate_algorithm, candidate_text,
                               target_identity, reference_identity, integrity, structural,
                               movement_magnitude, reasons, started,
                               {"groups": groups_found}, [], len(pairs), excluded)
    film_start = min(c.start_ms for c in target_cues)
    film_end = max(c.end_ms for c in target_cues)
    clean, overlapped = _split_segmentation_overlaps(pairs, reference_cues)
    if pairs and len(overlapped) / len(pairs) > OVERLAP_FRACTION_LIMIT:
        reasons.append(f"excessive_segmentation_overlap:{len(overlapped)}/{len(pairs)}")
    stats = regional_stats(clean, film_start, film_end)
    reasons.extend(stats["reasons"])
    if stats["n_pairs"] and stats["median_abs_ms"] > MEDIAN_TOLERANCE_MS:
        reasons.append("median_exceeds_tolerance")
    if stats["n_pairs"] and stats["p90_abs_ms"] > P90_TOLERANCE_MS:
        reasons.append("p90_exceeds_tolerance")
    if stats["n_pairs"] and stats["max_abs_ms"] > MAX_TOLERANCE_MS:
        reasons.append("max_exceeds_tolerance")
    signed = [(t, float(c) - float(r)) for c, r, t in clean]
    drift = analyze_drift([(t, d) for t, d in signed])
    drift_ok: bool | None = None
    if drift is not None:
        drift_ok = abs(drift) <= MAX_DRIFT_MS_PER_MINUTE
        if not drift_ok:
            reasons.append(f"residual_drift_exceeds_bound:{drift:+.1f}ms/min")
    change_points = detect_change_points([(t, d) for t, d in signed])
    landmark: dict[str, Any] = {"available": False}
    try:
        candidate_tl = Timeline.from_cues(_as_production_tuples(candidate_cues))
        reference_tl = Timeline.from_cues(_as_production_tuples(reference_cues))
        offset = first.reference_ms - scale * first.target_ms if first else 0.0
        timeline_report = verify_timeline(candidate_tl, reference_tl, scale, offset)
        landmark = {"available": True,
                    "before": timeline_report.before,
                    "after": timeline_report.after,
                    "classification": timeline_report.classification,
                    "reasons": list(timeline_report.reasons)}
    except Exception as exc:
        landmark = {"available": False, "reasons": (f"landmark_unavailable:{type(exc).__name__}",)}
    if change_points:
        reasons.append("hidden_discontinuity_on_linear_fit:"
                       + ",".join(f"{pos / 1000:.0f}s{step:+.0f}ms" for pos, step in change_points))
    residuals = [abs(d) for _, d in signed]
    if _consecutive_error_run(pairs, residuals) >= CONSECUTIVE_ERROR_RUN:
        # Recompute on clean pairs only: overlaps already have an explanation.
        clean_residuals = [abs(float(c) - float(r)) for c, r, _ in clean]
        if _consecutive_error_run(clean, clean_residuals) >= CONSECUTIVE_ERROR_RUN:
            reasons.append("concentrated_error_cluster:"
                           f"run={_consecutive_error_run(clean, clean_residuals)}")
    outliers = _risk_outliers(pairs, clean, overlapped, reference_cues)
    clamped, clamp_vetoes = detect_clamped(target_cues, candidate_cues)
    reasons.extend(clamp_vetoes)
    temporal = {"route": "linear", "scale": scale,
                "drift_ms_per_minute": drift, "drift_ok": drift_ok,
                "change_points": change_points, "landmark": landmark,
                "cut": _cut_evidence(candidate_cues, reference_cues,
                                     stats["median_abs_ms"], stats["p95_abs_ms"],
                                     _mad([abs(float(c) - float(r)) for c, r, _ in clean]),
                                     drift, change_points),
                "movement_magnitude_ms": float(movement_magnitude),
                "outlier_count": len(outliers)}
    regional = {"regions": stats["region_counts"], "medians": stats["region_medians_ms"]}
    if not reasons:
        return _unified_accept(
            candidate_algorithm, candidate_text, target_identity, reference_identity,
            integrity, structural, movement_magnitude, started, temporal, regional,
            stats, outliers, clamped, (), len(pairs), len(overlapped), excluded)
    return _unified_reject(
        "linear", scale, candidate_algorithm, candidate_text,
        target_identity, reference_identity, integrity, structural,
        movement_magnitude, reasons, started, temporal, outliers,
        len(pairs), excluded)


def _cut_evidence(candidate_cues: list[NucleusCue],
                  reference_cues: list[NucleusCue],
                  median: float, p95: float, mad: float,
                  drift: float | None,
                  change_points: list[tuple[int, float]]) -> dict[str, Any]:
    """Legacy cut classification on correspondence statistics (corroboration).

    Never rejects: like its Legacy role, the cut verdict contextualizes the
    residual evidence (STABLE_OFFSET vs DRIFT vs PIECEWISE vs DIFFERENT_CUT)
    but only correspondence residuals gate acceptance.
    """
    try:
        similarity = compare_structures(
            _as_production_tuples(candidate_cues),
            _as_production_tuples(reference_cues))
        cut = classify_cut(
            median_offset_ms=median, p95_offset_ms=p95, mad_offset_ms=mad,
            drift_ms_per_minute=drift, change_points=change_points,
            structural=similarity, max_plausible_offset_ms=MAX_PLAUSIBLE_OFFSET_MS,
            max_p95_ms=MAX_P95_MS_FOR_STABLE,
            max_drift_ms_per_minute=MAX_DRIFT_MS_PER_MINUTE)
        return {"available": True,
                "cut": cut.value if hasattr(cut, "value") else str(cut),
                "structural_score": similarity.score}
    except Exception as exc:
        return {"available": False, "reasons": (f"cut_unavailable:{type(exc).__name__}",)}


def _mad(values: list[float]) -> float:
    if not values:
        return 0.0
    median = float(statistics.median(sorted(values)))
    return float(statistics.median(sorted(abs(value - median) for value in values)))


def _unified_accept(
    candidate_algorithm: str, candidate_text: str,
    target_identity: dict[str, Any], reference_identity: dict[str, Any],
    integrity: dict[str, Any], structural: dict[str, Any],
    movement_magnitude: int, started: float,
    temporal: dict[str, Any], regional: dict[str, Any],
    stats: dict[str, Any], outliers: list[dict[str, Any]],
    clamped: list[dict[str, Any]], unmatched: tuple[dict[str, Any], ...],
    n_pairs: int, n_overlapped: int, excluded: int,
) -> UnifiedVerdict:
    _ = started
    return UnifiedVerdict(
        candidate_algorithm=candidate_algorithm,
        candidate_sha256=_sha(candidate_text),
        target_identity=target_identity,
        reference_identity=reference_identity,
        integrity_result=integrity,
        correspondence_result={"groups": n_pairs + n_overlapped,
                               "independent": n_pairs,
                               "overlapping_excluded": excluded},
        temporal_result=temporal,
        structural_result=structural,
        regional_result={**regional,
                         "median": stats.get("median_abs_ms", 0.0),
                         "p90": stats.get("p90_abs_ms", 0.0),
                         "p95": stats.get("p95_abs_ms", 0.0),
                         "max": stats.get("max_abs_ms", 0.0)},
        risk_findings=tuple(
            {"kind": "outlier", **entry} for entry in outliers),
        independent_evidence_count=n_pairs,
        coverage=dict(stats.get("region_counts", {})),
        residual_statistics={
            "median": stats.get("median_abs_ms", 0.0),
            "p90": stats.get("p90_abs_ms", 0.0),
            "p95": stats.get("p95_abs_ms", 0.0),
            "max": stats.get("max_abs_ms", 0.0)},
        failure_reasons=(),
        proposed_verdict="VERIFIED",
    )


def _unified_reject(
    route: str, scale: float | None, candidate_algorithm: str, candidate_text: str,
    target_identity: dict[str, Any], reference_identity: dict[str, Any],
    integrity: dict[str, Any], structural: dict[str, Any],
    movement_magnitude: int, reasons: list[str], started: float,
    temporal: dict[str, Any], outliers: list[dict[str, Any]],
    n_pairs: int, excluded: int,
) -> UnifiedVerdict:
    _ = (started, movement_magnitude)
    positive = any(
        reason.split(":")[0] in {
            "median_exceeds_tolerance", "p90_exceeds_tolerance",
            "max_exceeds_tolerance", "regional_median_exceeds_tolerance",
            "region_max_exceeds_tolerance", "concentrated_error_cluster",
            "hidden_discontinuity_on_linear_fit", "residual_drift_exceeds_bound",
            "implausible_affine_scale", "invalid_affine_scale",
            "dialogue_text_modified", "candidate_order_violation",
            "candidate_invalid_durations", "cue_retention_too_low",
        } or reason.startswith("dialogue_clamped")
        for reason in reasons)
    return UnifiedVerdict(
        candidate_algorithm=candidate_algorithm,
        candidate_sha256=_sha(candidate_text),
        target_identity=target_identity,
        reference_identity=reference_identity,
        integrity_result=integrity,
        correspondence_result={"route": route, "scale": scale},
        temporal_result=temporal,
        structural_result=structural,
        regional_result={},
        risk_findings=tuple(
            {"kind": "outlier", **entry} for entry in outliers),
        independent_evidence_count=n_pairs,
        coverage={},
        residual_statistics={},
        failure_reasons=tuple(reasons),
        proposed_verdict="REJECTED" if positive else "UNVERIFIED",
    )


def _verify_nonlinear(
    target_cues: list[NucleusCue],
    reference_cues: list[NucleusCue],
    candidate_cues: list[NucleusCue],
    detection: AnchorResult,
    target_identity: dict[str, Any],
    reference_identity: dict[str, Any],
    integrity: dict[str, Any],
    structural: dict[str, Any],
    movement_magnitude: int,
    candidate_algorithm: str,
    candidate_text: str,
    started: float,
) -> UnifiedVerdict:
    """Nonlinear branch: per-regime correspondence verification."""
    reasons: list[str] = []
    reasons.extend(integrity.get("reasons", ()))
    regimes = [
        {"label": str(getattr(regime, "label", "contradictory")),
         "implied_offset_ms": float(getattr(regime, "implied_offset_ms", 0.0)),
         "support": int(getattr(regime, "support", 0)),
         "span_start_ms": int(getattr(regime, "span_start_ms", 0)),
         "span_end_ms": int(getattr(regime, "span_end_ms", 0))}
        for regime in detection.regimes
        if str(getattr(regime, "label", "")) != "primary"
    ]
    if not regimes:
        reasons.append("nonlinear_without_evidenced_regimes")
        return _unified_reject(
            "nonlinear", None, candidate_algorithm, candidate_text,
            target_identity, reference_identity, integrity, structural,
            movement_magnitude, reasons, started,
            {"route": "nonlinear", "regimes": []}, [], 0, 0)
    records = _strong_records(detection)
    pairs, excluded = _map_to_candidate(records, candidate_cues)
    if len(pairs) < MIN_INDEPENDENT_GROUPS:
        reasons.append(f"insufficient_independent_groups:{len(pairs)}")
        return _unified_reject(
            "nonlinear", None, candidate_algorithm, candidate_text,
            target_identity, reference_identity, integrity, structural,
            movement_magnitude, reasons, started,
            {"route": "nonlinear",
             "regimes": regimes}, [], len(pairs), excluded)
    film_start = min(c.start_ms for c in target_cues)
    film_end = max(c.end_ms for c in target_cues)
    clean, overlapped = _split_segmentation_overlaps(pairs, reference_cues)
    if pairs and len(overlapped) / len(pairs) > OVERLAP_FRACTION_LIMIT:
        reasons.append(f"excessive_segmentation_overlap:{len(overlapped)}/{len(pairs)}")
    stats = regional_stats(clean, film_start, film_end)
    reasons.extend(stats["reasons"])
    if stats["n_pairs"] and stats["median_abs_ms"] > MEDIAN_TOLERANCE_MS:
        reasons.append("median_exceeds_tolerance")
    if stats["n_pairs"] and stats["p90_abs_ms"] > P90_TOLERANCE_MS:
        reasons.append("p90_exceeds_tolerance")
    if stats["n_pairs"] and stats["max_abs_ms"] > MAX_TOLERANCE_MS:
        reasons.append("max_exceeds_tolerance")
    signed = [(t, float(c) - float(r)) for c, r, t in clean]
    change_points = detect_change_points([(t, d) for t, d in signed])
    for entry in regimes:
        inside = [pair for pair in pairs
                  if int(str(entry["span_start_ms"])) <= pair[2]
                  <= int(str(entry["span_end_ms"]))]
        if len(inside) < REGION_MIN_POINTS:
            reasons.append(f"discontinuity_uncovered:{entry['label']}")
    residuals = [abs(d) for _, d in signed]
    if _consecutive_error_run(clean, residuals) >= CONSECUTIVE_ERROR_RUN:
        reasons.append("concentrated_error_cluster:"
                       f"run={_consecutive_error_run(clean, residuals)}")
    outliers = _risk_outliers(pairs, clean, overlapped, reference_cues)
    clamped, clamp_vetoes = detect_clamped(target_cues, candidate_cues)
    reasons.extend(clamp_vetoes)
    matched_ref = sorted({pair[1] for pair in pairs})
    unmatched = _unmatched_reference_spans(reference_cues, matched_ref)
    temporal = {"route": "nonlinear", "regimes": regimes,
                "change_points": change_points,
                "cut": _cut_evidence(candidate_cues, reference_cues,
                                     stats["median_abs_ms"], stats["p95_abs_ms"],
                                     _mad([abs(float(c) - float(r)) for c, r, _ in clean]),
                                     None, change_points),
                "movement_magnitude_ms": float(movement_magnitude),
                "outlier_count": len(outliers)}
    regional = {"regions": stats["region_counts"],
                "medians": stats["region_medians_ms"]}
    if not reasons:
        return _unified_accept(
            candidate_algorithm, candidate_text, target_identity, reference_identity,
            integrity, structural, movement_magnitude, started, temporal, regional,
            stats, outliers, clamped, unmatched, len(pairs), len(overlapped), excluded)
    return _unified_reject(
        "nonlinear", None, candidate_algorithm, candidate_text,
        target_identity, reference_identity, integrity, structural,
        movement_magnitude, reasons, started, temporal, outliers,
        len(pairs), excluded)


def shadow_compare(target_text: str, reference_text: str, candidate_text: str,
                   *, candidate_algorithm: str = "") -> dict[str, Any]:
    """Shadow-mode three-way comparison: Legacy vs V2 vs Unified.

    Pure measurement for tests and diagnostics. Never serves, caches,
    discovers references, or runs ALASS. Every branch is fail-closed and
    every disagreement carries concrete evidence — Unified is never assumed
    correct for being newer.
    """
    from app.services.sync.alignment import AlignmentAnalyzer
    from app.services.sync.correspondence_verification import (
        verify_linear_candidate,
        verify_nonlinear_candidate,
    )

    started = time.monotonic()
    legacy: dict[str, Any] = {"evaluated": False}
    try:
        evaluation = AlignmentAnalyzer().analyze(
            target_text, candidate_text, reference_text,
            alass_applied=True, alass_successful=True)
        legacy = {
            "evaluated": True,
            "sync_state": evaluation.sync_state.value,
            "verification": evaluation.verification.value,
            "median_offset_ms": evaluation.median_offset_ms,
            "p95_offset_ms": evaluation.p95_offset_ms,
            "mad_offset_ms": evaluation.mad_offset_ms,
            "drift_ms_per_minute": evaluation.drift_ms_per_minute,
            "coverage_score": evaluation.coverage_score,
            "structural_similarity": evaluation.structural_similarity,
            "cut_verdict": evaluation.cut_verdict,
            "rejection_reason": (evaluation.rejection_reason.value
                                 if evaluation.rejection_reason is not None else None),
            "reasons": list(evaluation.reasons),
        }
    except Exception as exc:
        legacy = {"evaluated": False, "error": f"{type(exc).__name__}"}
    v2: dict[str, Any] = {"evaluated": False}
    unified = verify_candidate(target_text, reference_text, candidate_text,
                               candidate_algorithm=candidate_algorithm)
    try:
        from two_point_sync.anchors import detect_anchors as _detect
        from two_point_sync.parser import parse_srt as _parse
        from two_point_sync.sanitizer import classify_all as _classify

        target_cues = _parse(target_text)
        reference_cues = _parse(reference_text)
        detection = _detect(
            target_cues, reference_cues,
            target_scores=_classify(target_cues),
            reference_scores=_classify(reference_cues))
        reference_sha = hashlib.sha256(reference_text.encode("utf-8")).hexdigest()
        if "NONLINEAR_DISCONTINUITY" in detection.reasons:
            verdict = verify_nonlinear_candidate(
                target_text=target_text, reference_text=reference_text,
                candidate_text=candidate_text, anchor_result=detection,
                reference_sha=reference_sha, candidate_algorithm=candidate_algorithm)
        elif detection.accepted:
            verdict = verify_linear_candidate(
                target_text=target_text, reference_text=reference_text,
                candidate_text=candidate_text, anchor_result=detection,
                reference_sha=reference_sha, candidate_algorithm=candidate_algorithm)
        else:
            verdict = None
        if verdict is None:
            v2 = {"evaluated": False, "reason": "insufficient_correspondence"}
        else:
            v2 = {"evaluated": True, "accepted": verdict.accepted,
                  "median": verdict.median_residual_ms,
                  "p90": verdict.p90_residual_ms,
                  "p95": verdict.p95_residual_ms,
                  "max": verdict.max_residual_ms,
                  "independent": verdict.independent_used,
                  "failure_reasons": list(verdict.failure_reasons)}
    except Exception as exc:
        v2 = {"evaluated": False, "error": f"{type(exc).__name__}"}
    agreement = (
        legacy.get("evaluated") is True and v2.get("evaluated") is True
        and ((legacy.get("verification") in ("verified", "cached")
              and legacy.get("sync_state", "").startswith("verified"))
             == bool(v2.get("accepted")))
        and (unified.proposed_verdict == "VERIFIED") == bool(v2.get("accepted")))
    return {
        "legacy": legacy,
        "v2": v2,
        "unified": {
            "proposed_verdict": unified.proposed_verdict,
            "failure_reasons": list(unified.failure_reasons),
            "independent_evidence_count": unified.independent_evidence_count,
            "coverage": dict(unified.coverage),
            "residual_statistics": dict(unified.residual_statistics),
            "risk_findings": [dict(entry) for entry in unified.risk_findings],
        },
        "agreement": agreement,
        "elapsed_ms": (time.monotonic() - started) * 1000.0,
    }
