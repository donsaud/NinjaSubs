"""Offline evaluation of the synchronization pipeline against golden ground truth.

    python tools/evaluate_golden_sync.py tests/fixtures/golden_sync/manifest.json

The evaluation hierarchy this implements is deliberately one-directional:

    ground truth  ->  system output  ->  error classification  ->  metrics

Ground truth is read from the manifest and is never produced by the system
under test. Nothing here writes production state, populates production caches,
emits audit records, or changes a threshold. The tool measures.

It does not tell you what to change. It ends with INVESTIGATION CANDIDATES.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.sync.alignment import AlignmentAnalyzer, SyncState
from app.services.sync.golden import (
    GOLDEN_ENGINE_VERSION,
    AlignmentExpectation,
    CandidateTruth,
    GoldenCase,
    GoldenCaseError,
    GoldenManifest,
    GroundTruthContent,
    GroundTruthFinal,
    load_manifest,
    resolve_fixture,
)
from app.services.sync.video_timeline import (
    is_withholding_evidence,
    validate_timeline_against_reference,
)

#: Below this many labelled observations, a rate is not printed as a number.
MIN_SAMPLE = 10
#: Wilson intervals are only meaningful past this too.
MIN_INTERVAL_SAMPLE = 5


# --- statistics (mirrors calibrate_sync discipline) ------------------------ #


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    if total <= 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5)
    return (max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom))


def rate(successes: int, total: int) -> dict[str, Any]:
    if total < MIN_SAMPLE:
        return {
            "n": total,
            "count": successes,
            "rate": None,
            "status": "INSUFFICIENT_SAMPLE",
        }
    interval = wilson(successes, total)
    return {
        "n": total,
        "count": successes,
        "rate": round(successes / total, 4),
        "wilson_low": round(interval[0], 4) if interval else None,
        "wilson_high": round(interval[1], 4) if interval else None,
        "status": "OK",
    }


# --- per-case system output ----------------------------------------------- #


def _parse_cues(path: Path) -> list[tuple[int, int, str]]:
    from app.services.subtitle_matcher import parse_srt_cues

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    try:
        return parse_srt_cues(text)
    except Exception:
        return []


def build_alignment(
    target_cues: list[tuple[int, int, str]],
    candidate_cues: list[tuple[int, int, str]],
    expectation: AlignmentExpectation,
) -> list[tuple[int, int, str]] | None:
    """Materialise the DECLARED alignment outcome, from ground truth only.

    This is the harness supplying the verifier with an alignment result, not
    the harness measuring an aligner. Every parameter comes from the manifest,
    so a reader can see exactly what the verifier was shown.

    ``AS_TARGET`` places the candidate's lines on the target's timing, which is
    what a correct aligner produces. ``PARTIAL_MISALIGN`` matches the opening
    segment and then displaces the rest, which is what an aligner does when
    handed a different cut: it exits 0 with small residuals and a wrong answer.
    """
    from app.services.sync.golden import AlignmentSimulation

    if expectation.mode is AlignmentSimulation.NO_ALIGNMENT or not target_cues:
        return None

    source = candidate_cues or target_cues
    if expectation.mode is AlignmentSimulation.PARTIAL_MISALIGN:
        pivot = expectation.displaced_from_index
        if pivot is None or pivot >= len(source):
            return None
        out = []
        for index, (start, end, text) in enumerate(source):
            shift = expectation.displaced_by_ms if index >= pivot else 0
            out.append((start + shift, end + shift, text))
        return out

    # AS_TARGET / AS_TARGET_WITH_DRIFT: the candidate's own lines land on the
    # target's timing, with the declared residual and drift applied.
    take = min(len(source), len(target_cues))
    if take == 0:
        return None
    out = []
    for index in range(take):
        target_start, target_end, _ = target_cues[index]
        text = source[index][2]
        minutes = (target_start - target_cues[0][0]) / 60_000
        shift = int(expectation.drift_per_minute_ms * minutes)
        residual = expectation.residual_ms
        jitter = residual if index % 2 == 0 else -residual
        out.append((target_start + shift + jitter, target_end + shift + jitter, text))
    if expectation.coverage < 1.0:
        keep = max(1, int(len(out) * expectation.coverage))
        out = out[:keep]
    return out


def _profile_from_truth(truth: VideoTimelineTruth | None):
    """Turn a declared video timeline into a profile the validator can read.

    The landmarks are ground truth, not a measurement, so this is a hand-off
    and not an inference.
    """
    if truth is None or not truth.is_available:
        return None
    from app.services.sync.video_timeline import VideoTimelineProfile

    return VideoTimelineProfile(
        duration_ms=truth.duration_ms,
        audio_stream_count=truth.audio_stream_count,
        video_stream_count=truth.video_stream_count,
        audio_landmarks=list(truth.audio_landmarks),
        video_landmarks=list(truth.video_landmarks),
    )


def classify_outcome(
    verified: bool, rejected: bool, final: GroundTruthFinal
) -> str:
    """Name the disagreement between the system's verdict and ground truth.

    A single function, used by the evaluator and asserted directly by the
    tests, so the scoring rule cannot drift away from what is tested.
    """
    if final is GroundTruthFinal.UNKNOWN:
        return "UNLABELLED"
    if verified and final is GroundTruthFinal.INCORRECT:
        return "FALSE_VERIFIED"
    if verified and final is GroundTruthFinal.CORRECT:
        return "TRUE_VERIFIED"
    if rejected and final is GroundTruthFinal.CORRECT:
        return "MISSED_RESYNCABLE"
    if rejected and final is GroundTruthFinal.INCORRECT:
        return "CORRECTLY_REFUSED"
    if not verified:
        return "UNVERIFIED_CORRECT" if final is GroundTruthFinal.CORRECT else "UNVERIFIED_INCORRECT"
    return "OTHER"


def _predictor_would_verify(state: SyncState) -> bool:
    """Whether the pipeline would treat this end state as synchronized.

    Derived from the observed state alone, so it never invents evidence.
    """
    return state in (SyncState.VERIFIED_SYNCED, SyncState.VERIFIED_RESYNCED)


@dataclass
class CaseResult:
    case_id: str
    failure_mode: str
    release_class: str
    annotation_source: str
    provider: str | None
    release_name: str | None
    truth: CandidateTruth
    system_state: str
    system_verified: bool
    system_rejected: bool
    system_predicted_verified: bool
    reference_trust: str | None
    error_class: str
    annotated: str
    median_offset_ms: float | None = None
    p95_offset_ms: float | None = None
    coverage: float | None = None
    # Independent video-derived evidence. Absent for every case with no
    # declared target video, which is every production request.
    video_available: bool = False
    video_verdict: str = "unavailable"
    video_reason: str = "VIDEO_PROFILE_UNAVAILABLE"
    video_audio_similarity: float | None = None
    video_scene_similarity: float | None = None
    video_duration_similarity: float | None = None
    video_would_withhold: bool = False
    notes: list[str] = field(default_factory=list)


def evaluate_case(
    case: GoldenCase, manifest_path: Path, analyzer: AlignmentAnalyzer
) -> list[CaseResult]:
    """Run one case through the verification layer and score it.

    The analyzer's contract is to judge a subtitle that has already been
    re-timed, so the harness supplies the alignment outcome the annotation
    declares. ``target`` and ``reference`` are both the video's true timing:
    the reference is the known-good material, the target is the subtitle that
    has to be matched. No field is fabricated - an absent filename stays absent.
    """
    target_path = resolve_fixture(manifest_path, case.target.path)
    target_cues = _parse_cues(target_path)
    target_name = case.target.filename  # None stays None: never invented

    results: list[CaseResult] = []
    for candidate in case.candidates:
        truth = case.truth_for(candidate.provider, candidate.release_name)
        if truth is None:  # pragma: no cover - loader rejects this
            raise GoldenCaseError(
                f"case {case.case_id!r} candidate {candidate.release_name!r} lacks ground truth"
            )
        candidate_cues = _parse_cues(resolve_fixture(manifest_path, candidate.path))

        if not candidate_cues:
            state = SyncState.REJECTED
            evaluation = None
        else:
            aligned = build_alignment(target_cues, candidate_cues, truth.alignment)
            if aligned is None:
                # The aligner produced nothing: the honest outcome is that
                # nothing can be claimed, not a rejection of the content.
                evaluation = analyzer.analyze(target_cues, None, reference=target_cues)
            else:
                evaluation = analyzer.analyze(
                    target_cues,
                    aligned,
                    reference=target_cues,
                    alass_applied=True,
                    alass_successful=True,
                )
            state = evaluation.sync_state

        verified = state in (SyncState.VERIFIED_SYNCED, SyncState.VERIFIED_RESYNCED)
        rejected = state is SyncState.REJECTED
        predicted = _predictor_would_verify(state)
        final = truth.final_alignment
        annotated = final.value
        known = final is not GroundTruthFinal.UNKNOWN

        error_class = classify_outcome(verified, rejected, final)

        # --- independent video-derived evidence (shadow only) ---------------
        # The video side comes from declared ground truth, never from the
        # candidate. This never alters `state`; it only records what an
        # independent witness would have said.
        profile = _profile_from_truth(case.target.video)
        aligned_cues = build_alignment(target_cues, candidate_cues, truth.alignment)
        evidence = validate_timeline_against_reference(
            profile, [cue[0] for cue in (aligned_cues or candidate_cues)]
        )

        notes = [f"state={state.value}", f"target_name={target_name!r}"]
        if evaluation is not None and evaluation.rejection_reason is not None:
            notes.append(f"rejection={evaluation.rejection_reason.value}")

        results.append(
            CaseResult(
                case_id=case.case_id,
                failure_mode=case.failure_mode,
                release_class=case.release_class,
                annotation_source=case.annotation_source.value,
                provider=candidate.provider,
                release_name=candidate.release_name,
                truth=truth,
                system_state=state.value,
                system_verified=verified,
                system_rejected=rejected,
                system_predicted_verified=predicted,
                reference_trust=evaluation.reference_trust if evaluation else None,
                error_class=error_class,
                annotated=annotated,
                median_offset_ms=evaluation.median_offset_ms if evaluation else None,
                p95_offset_ms=evaluation.p95_offset_ms if evaluation else None,
                coverage=evaluation.coverage_score if evaluation else None,
                video_available=evidence.available,
                video_verdict=evidence.verdict.value,
                video_reason=evidence.reason.value,
                video_audio_similarity=evidence.audio_activity_similarity,
                video_scene_similarity=evidence.scene_boundary_similarity,
                video_duration_similarity=evidence.duration_similarity,
                video_would_withhold=is_withholding_evidence(evidence),
                notes=notes,
            )
        )
    return results


# --- per-layer evaluation (section 13) ------------------------------------ #


def layer_metrics(results: list[CaseResult]) -> dict[str, Any]:
    """Attribute outcomes to a layer, so a final error has a plausible origin.

    This attributes by observed signals, not by proof. It says where to look,
    not what happened.
    """
    layers: dict[str, Counter] = defaultdict(Counter)
    for r in results:
        if r.error_class in ("UNLABELLED",):
            continue
        if r.error_class == "FALSE_VERIFIED":
            if r.truth.content is GroundTruthContent.DIFFERENT_CONTENT:
                layers["A_match_tier"]["content_mismatch_not_gated"] += 1
            if r.truth.cut.value == "different_cut":
                layers["D_alignment"]["aligned_to_the_wrong_cut"] += 1
            else:
                layers["E_verification"]["verified_on_insufficient_evidence"] += 1
        elif r.error_class == "MISSED_RESYNCABLE":
            if r.system_state == SyncState.REJECTED.value:
                layers["D_alignment"]["rejected_usable_candidate"] += 1
            else:
                layers["E_verification"]["under_verified"] += 1
        elif r.error_class in ("TRUE_VERIFIED", "CORRECTLY_REFUSED"):
            layers["E_verification"]["ok"] += 1
    return {k: dict(v) for k, v in layers.items()}


# --- aggregate ------------------------------------------------------------- #


def build_report(
    manifest: GoldenManifest, results: list[CaseResult], provenance: dict[str, Any]
) -> dict[str, Any]:
    labelled = [r for r in results if r.error_class != "UNLABELLED"]
    error_counts = Counter(r.error_class for r in labelled)

    false_verified = error_counts.get("FALSE_VERIFIED", 0)
    verified_total = sum(1 for r in labelled if r.system_verified)
    coverage = rate(verified_total, len(labelled))

    by_mode: dict[str, Counter] = defaultdict(Counter)
    by_class: dict[str, Counter] = defaultdict(Counter)
    for r in labelled:
        by_mode[r.failure_mode][r.error_class] += 1
        by_class[r.release_class][r.error_class] += 1

    # Missing-fingerprint impact: measured, not assumed.
    incomplete = [r for r in labelled if r.case_id == "missing_fingerprint"]

    # --- video-derived validation: coverage and counterfactual benefit ------ #
    video_metrics = _video_validation_metrics(labelled)

    sweep = threshold_sweep(labelled)

    return {
        "provenance": provenance,
        "dataset": {
            "version": manifest.dataset_version,
            "engine_version": manifest.engine_version,
            "cases": len(manifest.cases),
            "candidates": len(results),
            "labelled": len(labelled),
            "unlabelled": len(results) - len(labelled),
            "objective": sum(1 for r in results if r.annotation_source == "objective"),
            "human_reviewed": sum(
                1 for r in results if r.annotation_source == "human_reviewed"
            ),
            "valid_reference_set_cases": sum(
                1 for c in manifest.cases if c.is_valid_reference_set
            ),
        },
        "final_verification": {
            "ground_truth_correct": sum(1 for r in labelled if r.annotated == "correct"),
            "ground_truth_incorrect": sum(
                1 for r in labelled if r.annotated == "incorrect"
            ),
            "ground_truth_unknown": sum(1 for r in results if r.annotated == "unknown"),
            "system_verified": verified_total,
            "system_unverified": sum(1 for r in labelled if not r.system_verified),
            "system_rejected": sum(1 for r in labelled if r.system_rejected),
        },
        "error_classes": dict(error_counts.most_common()),
        "false_positives": {
            "false_verified": false_verified,
            "false_predicted": sum(
                1
                for r in labelled
                if r.system_predicted_verified and r.annotated == "incorrect"
            ),
            "false_resynced": sum(
                1
                for r in labelled
                if r.system_state == SyncState.VERIFIED_RESYNCED.value
                and r.annotated == "incorrect"
            ),
        },
        "false_negatives": {
            "missed_resyncable": error_counts.get("MISSED_RESYNCABLE", 0),
            "missed_synced": error_counts.get("UNVERIFIED_CORRECT", 0),
        },
        "rates": {
            "verification_coverage": coverage,
            "precision_on_verified": rate(
                sum(1 for r in labelled if r.system_verified and r.annotated == "correct"),
                verified_total,
            ),
            "correct_resync_rate": rate(
                error_counts.get("TRUE_VERIFIED", 0), verified_total
            ),
        },
        "prediction": {
            "predicted": sum(1 for r in labelled if r.system_predicted_verified),
            "false_predicted": sum(
                1
                for r in labelled
                if r.system_predicted_verified and r.annotated == "incorrect"
            ),
        },
        "reference_selection": {
            "cases_with_multiple_candidates": sum(
                1 for c in manifest.cases if len(c.candidates) > 1
            ),
            "valid_reference_set_cases": sum(
                1 for c in manifest.cases if c.is_valid_reference_set
            ),
            "note": (
                "VALID_REFERENCE_SET cases do not force a single correct reference; "
                "both anchors are correct by construction."
            ),
        },
        "layers": layer_metrics(labelled),
        "by_failure_mode": {k: dict(v) for k, v in by_mode.items()},
        "by_release_class": {k: dict(v) for k, v in by_class.items()},
        "missing_fingerprint": {
            "n": len(incomplete),
            "verified": sum(1 for r in incomplete if r.system_verified),
            "truth_correct": sum(1 for r in incomplete if r.annotated == "correct"),
            "note": (
                "Ground truth is known; the request carries no fingerprint. The gap "
                "between truth_correct and verified is capability lost to incomplete "
                "Stremio metadata, not a defect in the analyzer."
            ),
        },
        "circular_error_case": {
            "case_id": "circular_shared_wrong_timing",
            "error_class": next(
                (
                    r.error_class
                    for r in labelled
                    if r.case_id == "circular_shared_wrong_timing"
                ),
                "not_present",
            ),
            "note": (
                "Providers agree with each other and are still wrong. Self-consistency "
                "is not evidence; only independent ground truth catches this."
            ),
        },
        "threshold_sweep": sweep,
        "video_validation": video_metrics,
    }


def _video_validation_metrics(results: list[CaseResult]) -> dict[str, Any]:
    """How often the independent video witness was available, and what it said.

    The counterfactual is the point: what the existing verifier alone said,
    against what it would have said with video evidence. Nothing here changes
    any verdict; it counts what would have changed.
    """
    with_video = [r for r in results if r.video_available]
    without_video = [r for r in results if not r.video_available]
    withholding = [r for r in with_video if r.video_would_withhold]

    correct_blocks = [r for r in withholding if r.annotated == "incorrect"]
    wrong_blocks = [r for r in withholding if r.annotated == "correct"]

    existing_false_verified = sum(1 for r in results if r.error_class == "FALSE_VERIFIED")
    remaining_false_verified = sum(
        1
        for r in results
        if r.error_class == "FALSE_VERIFIED" and not r.video_would_withhold
    )
    newly_blocked = [
        r for r in results if r.error_class == "FALSE_VERIFIED" and r.video_would_withhold
    ]
    newly_held_true = [
        r for r in results if r.error_class == "TRUE_VERIFIED" and r.video_would_withhold
    ]
    return {
        "available": len(with_video),
        "unavailable": len(without_video),
        "availability_rate": rate(len(with_video), len(results)),
        "verdicts": dict(Counter(r.video_verdict for r in with_video).most_common()),
        "reasons": dict(Counter(r.video_reason for r in with_video).most_common()),
        "would_withhold": len(withholding),
        "would_withhold_and_correct": len(correct_blocks),
        "would_withhold_but_ground_truth_correct": len(wrong_blocks),
        "existing_false_verified": existing_false_verified,
        "after_video_validation_false_verified": remaining_false_verified,
        "newly_blocked_false_verified": len(newly_blocked),
        "newly_blocked_cases": [r.case_id for r in newly_blocked],
        "newly_withheld_true_verified": len(newly_held_true),
        "note": (
            "Observational. The verifier's verdict is unchanged; this counts what an "
            "independent video witness would have said about it. A case with no "
            "declared target video is abstained, never assumed to pass."
        ),
    }


def threshold_sweep(results: list[CaseResult]) -> dict[str, Any]:
    """Offline Pareto view over hypothetical residual thresholds.

    Recorded measurements only. No configuration is changed and no production
    code is re-run: this replays already-measured residuals against
    hypothetical cut-offs and shows the trade-off. It deliberately does not
    select a threshold.
    """
    samples = [(r.median_offset_ms, r) for r in results if r.median_offset_ms is not None]
    if len(samples) < MIN_SAMPLE:
        return {
            "status": "INSUFFICIENT_SAMPLE",
            "n": len(samples),
            "note": "Not enough measured residuals to sweep a threshold.",
        }
    points = []
    for cut in sorted({s[0] for s in samples}):
        selected = [r for value, r in samples if abs(value) <= cut]
        points.append(
            {
                "residual_cut_ms": cut,
                "verified": len(selected),
                "false_verified": sum(1 for r in selected if r.annotated == "incorrect"),
                "coverage": round(len(selected) / len(results), 4),
            }
        )
    return {
        "status": "OK",
        "n": len(samples),
        "points": points,
        "note": (
            "Pareto view over measured residuals. Presents the trade-off; it does not "
            "choose a threshold, and no threshold was changed."
        ),
    }


# --- rendering ------------------------------------------------------------- #


def _fmt_rate(value: dict[str, Any]) -> str:
    if value.get("rate") is None:
        return f"{value.get('status')} (n={value.get('n', 0)})"
    return (
        f"{value['rate']:.1%} (n={value['n']}, "
        f"95% CI {value['wilson_low']:.1%}-{value['wilson_high']:.1%})"
    )


def render(report: dict[str, Any]) -> str:
    out: list[str] = []
    add = out.append
    add("=" * 74)
    add("GOLDEN SYNCHRONIZATION BENCHMARK")
    add("=" * 74)
    prov = report["provenance"]
    ds = report["dataset"]
    add("")
    add("Provenance")
    add("----------")
    add(f"  Dataset version : {ds['version']}")
    add(f"  Dataset engine   : {ds['engine_version']}")
    add(f"  Harness engine   : {prov['harness_engine_version']}")
    add(f"  Git commit      : {prov['git_commit']}")
    add(f"  Tool            : {prov['tool']}")
    add("")
    add("Dataset")
    add("-------")
    add(f"  Cases: {ds['cases']}")
    add(f"  Candidates evaluated: {ds['candidates']}")
    add(f"  With independent ground truth: {ds['labelled']}")
    add(f"  Unknown-label candidates: {ds['unlabelled']}")
    add(f"  Objective labels: {ds['objective']}")
    add(f"  Human-reviewed labels: {ds['human_reviewed']}")
    add(f"  VALID_REFERENCE_SET cases: {ds['valid_reference_set_cases']}")

    fv = report["final_verification"]
    add("")
    add("Final Verification")
    add("------------------")
    add(f"  Ground truth correct  : {fv['ground_truth_correct']}")
    add(f"  Ground truth incorrect: {fv['ground_truth_incorrect']}")
    add(f"  Ground truth unknown  : {fv['ground_truth_unknown']}")
    add(f"  System verified       : {fv['system_verified']}")
    add(f"  System unverified     : {fv['system_unverified']}")
    add(f"  System rejected       : {fv['system_rejected']}")

    add("")
    add("Error classes")
    add("-------------")
    for name, count in report["error_classes"].items():
        add(f"  {count:>4}  {name}")

    fps = report["false_positives"]
    add("")
    add("False Positives  (the failure the architecture exists to prevent)")
    add("------------------------------------------------------")
    add(f"  false_verified : {fps['false_verified']}")
    add(f"  false_resynced : {fps['false_resynced']}")
    add(f"  false_predicted: {fps['false_predicted']}")

    fns = report["false_negatives"]
    add("")
    add("False Negatives  (cost of being conservative)")
    add("-----------------------------------------")
    add(f"  missed_resyncable: {fns['missed_resyncable']}")
    add(f"  missed_synced    : {fns['missed_synced']}")

    rates = report["rates"]
    add("")
    add("Rates")
    add("-----")
    add(f"  verification_coverage : {_fmt_rate(rates['verification_coverage'])}")
    add(f"  precision_on_verified : {_fmt_rate(rates['precision_on_verified'])}")
    add(f"  correct_resync_rate   : {_fmt_rate(rates['correct_resync_rate'])}")

    add("")
    add("By release class")
    add("----------------")
    for name, counts in sorted(report["by_release_class"].items()):
        add(f"  {name}: {counts}")

    add("")
    add("By failure mode")
    add("---------------")
    for name, counts in sorted(report["by_failure_mode"].items()):
        add(f"  {name}: {counts}")

    mf = report["missing_fingerprint"]
    add("")
    add("Missing-fingerprint impact")
    add("-------------------------")
    add(f"  candidates: {mf['n']}, verified by system: {mf['verified']}, "
        f"ground truth correct: {mf['truth_correct']}")
    add(f"  {mf['note']}")

    circ = report["circular_error_case"]
    add("")
    add("Circular-error scenario")
    add("-----------------------")
    add(f"  {circ['case_id']} -> {circ['error_class']}")
    add(f"  {circ['note']}")

    vv = report.get("video_validation") or {}
    add("")
    add("Video-derived timeline validation (observational, shadow only)")
    add("--------------------------------------------------")
    add(f"  video evidence available: {vv.get('available', 0)} of "
        f"{vv.get('available', 0) + vv.get('unavailable', 0)} candidates")
    add(f"  abstained (no video)   : {vv.get('unavailable', 0)}")
    if vv.get("verdicts"):
        for verdict, count in vv["verdicts"].items():
            add(f"    {count:>4}  {verdict}")
    add("")
    add("  Counterfactual: what the verifier said vs what it would have said")
    add(f"    false_verified, existing verifier      : {vv.get('existing_false_verified', 0)}")
    add(f"    false_verified, with video validation  : {vv.get('after_video_validation_false_verified', 0)}")
    add(f"    newly blocked false-verified cases     : {vv.get('newly_blocked_false_verified', 0)}")
    if vv.get("newly_blocked_cases"):
        add(f"      {', '.join(vv['newly_blocked_cases'])}")
    add(f"    would-withhold where truth is correct  : {vv.get('would_withhold_but_ground_truth_correct', 0)}")
    add(f"    true verifications it would have held  : {vv.get('newly_withheld_true_verified', 0)}")
    add(f"  {vv.get('note', '')}")

    sweep = report["threshold_sweep"]
    add("")
    add("Threshold sweep (offline, informational)")
    add("---------------------------------------")
    add(f"  {sweep['status']} (n={sweep['n']})")
    if sweep["status"] == "OK":
        for point in sweep["points"][:8]:
            add(
                f"    cut<={point['residual_cut_ms']:>8.0f}ms  "
                f"verified={point['verified']:>3}  "
                f"false_verified={point['false_verified']:>3}  "
                f"coverage={point['coverage']:.2f}"
            )
    add(f"  {sweep['note']}")

    add("")
    add("=" * 74)
    add("INVESTIGATION CANDIDATES")
    add("=" * 74)
    for line in investigation_candidates(report):
        add(f"  - {line}")
    add("")
    add("This benchmark does not recommend threshold changes, does not promote")
    add("Reference Selection v2 or shadow pool fetching, and does not modify any")
    add("configuration. It measures.")
    return "\n".join(out)


def investigation_candidates(report: dict[str, Any]) -> list[str]:
    """Where to look next. Never what to change."""
    out: list[str] = []
    fps = report["false_positives"]
    if fps["false_verified"]:
        out.append(
            f"{fps['false_verified']} false-verified candidates. Group by failure_mode "
            "below before forming any theory."
        )
    else:
        out.append("No false-verified candidate in this dataset.")
    for mode, counts in sorted(report["by_failure_mode"].items()):
        if counts.get("FALSE_VERIFIED"):
            out.append(f"false_verified concentrated in failure_mode={mode}")
    for name, counts in sorted(report["by_release_class"].items()):
        if counts.get("FALSE_VERIFIED"):
            out.append(f"false_verified concentrated in release_class={name}")
    circ = report["circular_error_case"]
    out.append(
        f"circular case classified {circ['error_class']}: confirms whether internal "
        "consistency is currently sufficient to detect a shared wrong timing model"
    )
    if report["rates"]["verification_coverage"].get("rate") is None:
        out.append(
            "verification_coverage is below the reporting minimum; the dataset is too "
            "small to support a coverage statement"
        )
    out.append(
        "No threshold, weight or policy should move until a false-positive class is "
        "identified here AND the change is re-measured against this same dataset."
    )
    return out


# --- provenance ------------------------------------------------------------ #


def provenance(manifest: GoldenManifest) -> dict[str, Any]:
    try:
        commit = (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            or "unknown"
        )
    except Exception:  # pragma: no cover - git may be absent
        commit = "unknown"
    return {
        "dataset_version": manifest.dataset_version,
        "dataset_engine_version": manifest.engine_version,
        "harness_engine_version": GOLDEN_ENGINE_VERSION,
        "git_commit": commit,
        "tool": "evaluate_golden_sync",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate synchronization against independent golden ground truth"
    )
    parser.add_argument("manifest", type=Path, help="path to manifest.json")
    parser.add_argument("--json-out", type=Path, help="write the structured report here")
    parser.add_argument(
        "--case", action="append", help="evaluate only these case_ids (repeatable)"
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="also print the offline threshold trade-off (already included)",
    )
    args = parser.parse_args(argv)

    if not args.manifest.is_file():
        print(f"error: manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    try:
        manifest = load_manifest(args.manifest)
    except GoldenCaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if manifest.engine_version != GOLDEN_ENGINE_VERSION:
        print(
            f"warning: dataset was authored for engine {manifest.engine_version}, "
            f"harness targets {GOLDEN_ENGINE_VERSION}. Results are labelled, not merged.",
            file=sys.stderr,
        )

    analyzer = AlignmentAnalyzer()
    results: list[CaseResult] = []
    for case in manifest.cases:
        if args.case and case.case_id not in args.case:
            continue
        results.extend(evaluate_case(case, args.manifest, analyzer))

    report = build_report(manifest, results, provenance(manifest))
    print(render(report))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        print(f"\nstructured output written to {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
