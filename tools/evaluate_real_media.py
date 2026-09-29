"""Evaluate the video-derived timeline validator on real media.

    python tools/evaluate_real_media.py /path/to/manifest.json

Real media stays outside the repository. This harness is offline by
construction: it takes an explicit local path, writes its cache to a dedicated
directory, and never touches production state.

What it reports, and deliberately does not:

* every raw signal the profile produced, uncollapsed, so it is possible to see
  which signal actually carries the information on real programmes;
* whether wrong cuts are detected and global offsets accepted, measured
  separately;
* extraction and per-candidate cost, and cache behaviour;
* a classification of every observed false positive into the four categories
  from the phase specification, because only one of them is evidence that a
  semantic or audio layer is warranted.

It does not recommend VAD, and it does not change a threshold.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.sync.alignment import AlignmentAnalyzer, SyncState
from app.services.sync.golden import (
    GOLDEN_ENGINE_VERSION,
    GoldenCaseError,
    GroundTruthFinal,
    RealMediaManifest,
    load_real_media_manifest,
    sha256_file,
)
from app.services.sync.video_timeline import (
    VIDEO_PROFILE_VERSION,
    VideoProfileCache,
    VideoVerdict,
    extract_video_profile,
    is_withholding_evidence,
    validate_timeline_against_reference,
)

MIN_SAMPLE = 10
#: Only category C is evidence that a semantic/audio layer is warranted.
VAD_CATEGORIES = {
    "A_timeline_solvable": (
        "A",
        "Solvable with the current timeline evidence: the landmarks were there "
        "and the comparison was right to object.",
    ),
    "B_better_landmarks": (
        "B",
        "Solvable with better landmark extraction: the signal was present but "
        "too coarse or too sensitive to use.",
    ),
    "C_needs_semantic_audio": (
        "C",
        "Requires semantic or audio evidence: the timeline signal cannot see "
        "this failure at all.",
    ),
    "D_insufficient_data": (
        "D",
        "Insufficient data: too few observations to classify.",
    ),
}


def _cue_starts(path: Path) -> list[int]:
    from app.services.subtitle_matcher import parse_srt_cues

    try:
        return [cue[0] for cue in parse_srt_cues(path.read_text(encoding="utf-8", errors="replace"))]
    except Exception:
        return []


def _verify_hashes(case, *, strict: bool) -> list[str]:
    problems: list[str] = []
    for label, spec in (("video", case.video), ("reference", case.reference), ("candidate", case.candidate)):
        path = Path(spec.path)
        if not path.is_file():
            problems.append(f"{label}: file missing ({spec.path})")
            continue
        if not spec.sha256:
            continue
        actual = sha256_file(path)
        if actual != spec.sha256:
            problems.append(f"{label}: sha256 mismatch (expected {spec.sha256[:12]}, got {actual[:12]})")
    return problems


def evaluate_case(
    case,
    cache: VideoProfileCache,
    analyzer: AlignmentAnalyzer,
    *,
    timeout: float,
    use_cache: bool,
) -> dict[str, Any]:
    """One real-media case: profile, validate, verify, compare to truth."""
    identity = case.video.sha256 or f"path:{case.video.path}"
    row: dict[str, Any] = {
        "case_id": case.case_id,
        "category": case.category,
        "release_class": case.release_class,
        "cut": case.ground_truth.cut.value,
        "final": case.ground_truth.final.value,
        "established_by": case.ground_truth.established_by.value,
        "review_status": case.ground_truth.review_status,
        "reviewer_count": case.ground_truth.reviewer_count,
    }

    profile = cache.get(identity) if use_cache else None
    if profile is not None:
        row["profile_source"] = "cache"
        cache_ms = 0.0
    else:
        row["profile_source"] = "extracted"
        started = time.monotonic()
        profile = extract_video_profile(case.video.path, timeout=timeout)
        cache_ms = (time.monotonic() - started) * 1000
        if profile is not None and use_cache:
            cache.set(identity, profile)
    row["profile_extraction_ms"] = round(cache_ms, 3)
    row["profile_present"] = profile is not None

    if profile is None:
        row.update(
            {
                "video_verdict": VideoVerdict.UNAVAILABLE.value,
                "video_reason": "VIDEO_PROFILE_UNAVAILABLE",
                "would_withhold": False,
                "supported": False,
            }
        )
        return row

    row.update(
        {
            "profile_digest": profile.digest,
            "profile_version": profile.profile_version,
            "container_duration_ms": profile.duration_ms,
            "audio_start_time_ms": profile.audio_start_time_ms,
            "audio_stream_count": profile.audio_stream_count,
            "selected_audio_stream": profile.selected_audio_stream,
            "audio_stream_selection": profile.audio_stream_selection,
            "audio_stream_ambiguous": profile.audio_stream_ambiguous,
            "audio_landmark_count": len(profile.audio_landmarks),
            "scene_landmark_count": len(profile.video_landmarks),
            # Raw evidence, uncollapsed, per the phase specification.
            "audio_landmarks": list(profile.audio_landmarks),
            "scene_landmarks": list(profile.video_landmarks),
            "extraction_ms_reported": profile.extraction_ms,
        }
    )

    reference_cues = _cue_starts(Path(case.reference.path))
    candidate_cues = _cue_starts(Path(case.candidate.path))
    row["reference_cue_count"] = len(reference_cues)
    row["candidate_cue_count"] = len(candidate_cues)

    started = time.monotonic()
    evidence = validate_timeline_against_reference(profile, reference_cues)
    row["candidate_validation_ms"] = round((time.monotonic() - started) * 1000, 4)

    row.update(
        {
            "video_verdict": evidence.verdict.value,
            "video_reason": evidence.reason.value,
            "duration_similarity": evidence.duration_similarity,
            "audio_activity_similarity": evidence.audio_activity_similarity,
            "scene_boundary_similarity": evidence.scene_boundary_similarity,
            "best_offset_ms": evidence.best_offset_ms,
            "first_half_similarity": evidence.first_half_similarity,
            "second_half_similarity": evidence.second_half_similarity,
            "would_withhold": is_withholding_evidence(evidence),
            "detail": evidence.detail[:6],
        }
    )

    # The existing subtitle/reference verification, for the incremental view.
    if candidate_cues:
        evaluation = analyzer.analyze(reference_cues, candidate_cues)
        state = evaluation.sync_state
        row["sync_state"] = state.value
        row["existing_verified"] = state in (
            SyncState.VERIFIED_SYNCED,
            SyncState.VERIFIED_RESYNCED,
        )
        row["median_offset_ms"] = evaluation.median_offset_ms
    else:
        row["sync_state"] = None
        row["existing_verified"] = False
        row["median_offset_ms"] = None

    row["truth_incorrect"] = case.ground_truth.final is GroundTruthFinal.INCORRECT
    row["truth_correct"] = case.ground_truth.final is GroundTruthFinal.CORRECT
    row["truth_unknown"] = not (row["truth_incorrect"] or row["truth_correct"])
    row["supported"] = True

    # Video-level confusion, kept separate from the verifier's confusion.
    if row["truth_unknown"] or not evidence.available:
        row["video_outcome"] = "unknown"
    elif evidence.verdict is VideoVerdict.MISMATCH:
        row["video_outcome"] = "correct_mismatch" if row["truth_incorrect"] else "false_mismatch"
    elif evidence.verdict is VideoVerdict.MATCH:
        row["video_outcome"] = "correct_match" if row["truth_correct"] else "false_match"
    else:
        row["video_outcome"] = "unknown"
    return row


def _classify_false_positive(row: dict[str, Any]) -> tuple[str, str]:
    """Category A-D for a false video match, per §20.

    Only category C is evidence that a semantic or audio layer is warranted,
    and this returns C only when the timeline signal genuinely had nothing to
    work with. A missing or ambiguous signal is D, not C.
    """
    if row.get("video_outcome") != "false_match":
        return ("", "")
    landmarks = row.get("audio_landmark_count") or 0
    if not row.get("profile_present"):
        return ("D_insufficient_data", "no profile could be extracted")
    if row.get("audio_stream_ambiguous"):
        return ("D_insufficient_data", "programme audio track not identifiable")
    if landmarks < 4:
        return (
            "B_better_landmarks",
            f"only {landmarks} audio landmarks; a denser extraction might separate this cut",
        )
    return (
        "C_needs_semantic_audio",
        "the timeline signal was present and dense but could not see this difference",
    )


def build_report(
    manifest: RealMediaManifest,
    rows: list[dict[str, Any]],
    provenance: dict[str, Any],
    *,
    manifest_path: Path,
) -> dict[str, Any]:
    supported = [r for r in rows if r.get("supported")]
    unsupported = [r for r in rows if not r.get("supported")]
    # "Media absent" and "extraction failed" are different facts and must not
    # be reported as one. A missing file is a local-setup fact; a failed
    # extraction is a finding about the file.
    media_absent = [r for r in rows if r.get("profile_source") == "missing_media"]
    extraction_failed = [
        r
        for r in rows
        if r.get("profile_source") == "extracted" and not r.get("profile_present")
    ]
    outcomes = Counter(r.get("video_outcome", "unknown") for r in supported)

    false_matches = [r for r in supported if r.get("video_outcome") == "false_match"]
    fp_categories: list[dict[str, Any]] = []
    for row in false_matches:
        code, note = _classify_false_positive(row)
        fp_categories.append({"case_id": row["case_id"], "category": code, "note": note})

    existing_false_verified = [r for r in supported if r.get("truth_incorrect") and r.get("existing_verified")]
    remaining = [
        r
        for r in existing_false_verified
        if not r.get("would_withhold")
    ]
    newly_blocked_correct = [
        r for r in supported if r.get("would_withhold") and r.get("truth_correct")
    ]
    newly_blocked_incorrect = [
        r for r in supported if r.get("would_withhold") and r.get("truth_incorrect")
    ]

    extractions = [r["profile_extraction_ms"] for r in rows if r["profile_source"] == "extracted"]
    validations = [r["candidate_validation_ms"] for r in supported]

    def _pct(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
        return round(ordered[index], 3)

    return {
        "provenance": provenance,
        "dataset": {
            "manifest": str(manifest_path),
            "version": manifest.dataset_version,
            "cases": len(manifest.cases),
            "media_present": len(supported) + len(extraction_failed),
            "categories": dict(Counter(c.case_id and c.category for c in manifest.cases).most_common()),
            "release_classes": dict(Counter(c.release_class for c in manifest.cases).most_common()),
            "established_by": dict(Counter(c.ground_truth.established_by.value for c in manifest.cases).most_common()),
        },
        "media_coverage": {
            "profiles_available": len(supported),
            "media_absent": len(media_absent),
            "extraction_failures": len(extraction_failed),
            "unsupported": len(unsupported),
            "audio_stream_ambiguous": sum(1 for r in rows if r.get("audio_stream_ambiguous")),
        },
        "video_validation": {
            "correct_match": outcomes.get("correct_match", 0),
            "correct_mismatch": outcomes.get("correct_mismatch", 0),
            "false_match": outcomes.get("false_match", 0),
            "false_mismatch": outcomes.get("false_mismatch", 0),
            "unknown": outcomes.get("unknown", 0),
            "verdicts": dict(Counter(r.get("video_verdict") for r in supported).most_common()),
            "reasons": dict(Counter(r.get("video_reason") for r in supported).most_common()),
        },
        "synchronization_impact": {
            "existing_false_verified": len(existing_false_verified),
            "with_video_validation": len(remaining),
            "newly_blocked_incorrect": len(newly_blocked_incorrect),
            "newly_blocked_correct": len(newly_blocked_correct),
            "newly_blocked_correct_cases": [r["case_id"] for r in newly_blocked_correct],
        },
        "performance": {
            "extractions": len(extractions),
            "median_extraction_ms": round(statistics.median(extractions), 3) if extractions else None,
            "p95_extraction_ms": _pct(extractions, 0.95),
            "median_candidate_validation_ms": (
                round(statistics.median(validations), 4) if validations else None
            ),
            "p95_candidate_validation_ms": _pct(validations, 0.95),
            "cache_hits": sum(1 for r in rows if r["profile_source"] == "cache"),
            "cache_misses": sum(1 for r in rows if r["profile_source"] == "extracted"),
        },
        "false_positive_classification": fp_categories,
        "vad_gate": {
            "categories": {code: note for code, (_letter, note) in VAD_CATEGORIES.items()},
            "counts": dict(Counter(item["category"] for item in fp_categories).most_common()),
            "evidence_for_vad": sum(
                1 for item in fp_categories if item["category"] == "C_needs_semantic_audio"
            ),
            "note": (
                "Only category C counts as evidence that a semantic or audio layer is "
                "warranted. 'VAD might help' is not the same claim as 'VAD is required', "
                "and nothing here decides it."
            ),
        },
        "cases": rows,
    }


def provenance(manifest: RealMediaManifest) -> dict[str, Any]:
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
    except Exception:  # pragma: no cover
        commit = "unknown"
    return {
        "git_commit": commit,
        "engine_version": GOLDEN_ENGINE_VERSION,
        "video_profile_version": VIDEO_PROFILE_VERSION,
        "dataset_version": manifest.dataset_version,
        "tool": "evaluate_real_media",
    }


def render(report: dict[str, Any]) -> str:
    out: list[str] = []
    add = out.append
    add("=" * 74)
    add("REAL MEDIA VIDEO VALIDATION BENCHMARK")
    add("=" * 74)
    prov = report["provenance"]
    add("")
    add("Provenance")
    add("----------")
    add(f"  Git commit          : {prov['git_commit']}")
    add(f"  Engine version      : {prov['engine_version']}")
    add(f"  Video profile ver.  : {prov['video_profile_version']}")
    add(f"  Dataset version     : {prov['dataset_version']}")
    add(f"  Tool                : {prov['tool']}")

    ds = report["dataset"]
    mc = report["media_coverage"]
    add("")
    add("Dataset")
    add("-------")
    add(f"  Manifest: {ds['manifest']}")
    add(f"  Cases declared: {ds['cases']}")
    add(f"  Media present: {ds['media_present']}")
    add(f"  Media absent (local setup, not a finding): {mc['media_absent']}")
    if ds["categories"]:
        add(f"  Categories: {ds['categories']}")
    if ds["release_classes"]:
        add(f"  Release classes: {ds['release_classes']}")
    add(f"  Truth established by: {ds['established_by']}")

    add("")
    add("Media coverage")
    add("--------------")
    add(f"  Video profiles available: {mc['profiles_available']}")
    add(f"  Extraction failures: {mc['extraction_failures']}")
    add(f"  Unsupported cases: {mc['unsupported']}")
    add(f"  Ambiguous programme audio: {mc['audio_stream_ambiguous']}")

    vv = report["video_validation"]
    add("")
    add("Video validation")
    add("----------------")
    add(f"  Correct matches:   {vv['correct_match']}")
    add(f"  Correct mismatches:{vv['correct_mismatch']}")
    add(f"  FALSE matches:     {vv['false_match']}")
    add(f"  FALSE mismatches:  {vv['false_mismatch']}")
    add(f"  Unknown:           {vv['unknown']}")
    if vv["reasons"]:
        add("")
        add("  Reasons")
        for reason, count in vv["reasons"].items():
            add(f"    {count:>4}  {reason}")

    si = report["synchronization_impact"]
    add("")
    add("Synchronization impact")
    add("---------------------")
    add(f"  Existing false_verified: {si['existing_false_verified']}")
    add(f"  With video validation:   {si['with_video_validation']}")
    add(f"  Newly blocked, incorrect: {si['newly_blocked_incorrect']}")
    add(f"  Newly blocked, CORRECT:   {si['newly_blocked_correct']}")
    if si["newly_blocked_correct_cases"]:
        add(f"    {', '.join(si['newly_blocked_correct_cases'])}")

    perf = report["performance"]
    add("")
    add("Performance")
    add("-----------")
    add(f"  Extractions performed: {perf['extractions']}")
    add(f"  Median extraction time: {perf['median_extraction_ms']} ms")
    add(f"  P95 extraction time:    {perf['p95_extraction_ms']} ms")
    add(f"  Median candidate validation: {perf['median_candidate_validation_ms']} ms")
    add(f"  P95 candidate validation:    {perf['p95_candidate_validation_ms']} ms")
    add(f"  Cache hits: {perf['cache_hits']}  misses: {perf['cache_misses']}")

    add("")
    add("Per-case raw evidence")
    add("--------------------")
    for row in report["cases"]:
        add(f"  {row['case_id']} [{row['category']}] cut={row['cut']} final={row['final']}")
        if not row.get("supported"):
            add(f"    UNSUPPORTED: {row['video_reason']}")
            continue
        add(
            f"    audio={row.get('audio_activity_similarity')} "
            f"scene={row.get('scene_boundary_similarity')} "
            f"duration={row.get('duration_similarity')} "
            f"offset={row.get('best_offset_ms')}ms "
            f"halves=({row.get('first_half_similarity')},{row.get('second_half_similarity')})"
        )
        add(
            f"    landmarks: audio={row.get('audio_landmark_count')} "
            f"scene={row.get('scene_landmark_count')} "
            f"stream={row.get('selected_audio_stream')} "
            f"({row.get('audio_stream_selection')}) "
            f"start={row.get('audio_start_time_ms')}ms"
        )
        add(
            f"    verdict={row['video_verdict']} outcome={row['video_outcome']} "
            f"withhold={row['would_withhold']} state={row.get('sync_state')}"
        )

    gate = report["vad_gate"]
    add("")
    add("=" * 74)
    add("FALSE POSITIVE CLASSIFICATION")
    add("=" * 74)
    if not report["false_positive_classification"]:
        add("  No false matches observed, so nothing to classify.")
    for item in report["false_positive_classification"]:
        add(f"  {item['case_id']}: {item['category']}")
        add(f"    {item['note']}")
    add("")
    for code, (_letter, note) in VAD_CATEGORIES.items():
        add(f"  {code}: {note}")
    add("")
    add(f"  Evidence for a semantic/audio layer: {gate['evidence_for_vad']} case(s).")
    add(f"  {gate['note']}")

    add("")
    add("=" * 74)
    add("INVESTIGATION CANDIDATES")
    add("=" * 74)
    for line in investigation_candidates(report):
        add(f"  - {line}")
    add("")
    add("This benchmark does not recommend VAD, does not change a threshold, and")
    add("does not modify production behaviour. It measures.")
    return "\n".join(out)


def investigation_candidates(report: dict[str, Any]) -> list[str]:
    out: list[str] = []
    mc = report["media_coverage"]
    ds = report["dataset"]
    if mc["profiles_available"] == 0:
        out.append(
            "No media was present. Nothing about real-programme audio has been "
            "measured; the generated-media result does not extend to real content."
        )
    else:
        out.append(
            f"{mc['profiles_available']} of {ds['cases']} cases had media. "
            "Check the ratio before reading any rate."
        )
    if ds["established_by"].get("human_reviewed", 0) == 0 and mc["profiles_available"]:
        out.append(
            "No case is human-reviewed, so the labels are not yet independent of "
            "the material; conclusions drawn from them are provisional."
        )
    for item in report["false_positive_classification"]:
        out.append(f"{item['case_id']} classified {item['category']}: {item['note']}")
    gate = report["vad_gate"]
    if gate["evidence_for_vad"]:
        out.append(
            f"{gate['evidence_for_vad']} false match(es) fall in category C. That is "
            "the only category that counts as evidence for a semantic/audio layer, "
            "and it is a reason to evaluate one, not to add one."
        )
    else:
        out.append(
            "No false match landed in category C, so the current timeline signal "
            "has no measured need for a semantic/audio layer on this data."
        )
    if report["video_validation"]["false_mismatch"]:
        out.append(
            f"{report['video_validation']['false_mismatch']} correct mismatches: the "
            "signal flagged something it should not have. This is the cost side and "
            "must be weighed against the false matches."
        )
    out.append(
        "Two runs over the same media should reproduce the same profile digest and "
        "verdict; if they do not, the extractor is not deterministic."
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the video timeline validator against real media"
    )
    parser.add_argument("manifest", type=Path, help="path to a real-media manifest")
    parser.add_argument("--json-out", type=Path, help="write the structured report here")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="dedicated profile cache directory; a temp dir is used when omitted",
    )
    parser.add_argument("--no-cache", action="store_true", help="always re-extract profiles")
    parser.add_argument("--timeout", type=float, default=180.0, help="per-file extraction timeout")
    parser.add_argument(
        "--strict-hashes",
        action="store_true",
        help="treat a sha256 mismatch as a fatal error rather than a report line",
    )
    parser.add_argument("--case", action="append", help="evaluate only these case_ids")
    args = parser.parse_args(argv)

    if not args.manifest.is_file():
        print(f"error: manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    try:
        manifest = load_real_media_manifest(args.manifest)
    except GoldenCaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    import tempfile

    temp_dir: tempfile.TemporaryDirectory | None = None
    if args.cache_dir is not None:
        cache_dir = args.cache_dir
    else:
        # A dedicated cache, never the production one.
        temp_dir = tempfile.TemporaryDirectory(prefix="real_media_profiles_")
        cache_dir = Path(temp_dir.name)
    cache = VideoProfileCache(cache_dir)

    analyzer = AlignmentAnalyzer()
    rows: list[dict[str, Any]] = []
    try:
        for case in manifest.cases:
            if args.case and case.case_id not in args.case:
                continue
            problems = _verify_hashes(case, strict=args.strict_hashes)
            if args.strict_hashes and problems:
                print(f"error: {case.case_id}: " + "; ".join(problems), file=sys.stderr)
                return 2
            if not case.has_media:
                rows.append(
                    {
                        "case_id": case.case_id,
                        "category": case.category,
                        "release_class": case.release_class,
                        "cut": case.ground_truth.cut.value,
                        "final": case.ground_truth.final.value,
                        "established_by": case.ground_truth.established_by.value,
                        "review_status": case.ground_truth.review_status,
                        "reviewer_count": case.ground_truth.reviewer_count,
                        "profile_source": "missing_media",
                        "profile_present": False,
                        "profile_extraction_ms": 0.0,
                        "video_verdict": VideoVerdict.UNAVAILABLE.value,
                        "video_reason": "VIDEO_PROFILE_UNAVAILABLE",
                        "would_withhold": False,
                        "supported": False,
                        "problems": problems,
                    }
                )
                continue
            rows.append(
                evaluate_case(
                    case,
                    cache,
                    analyzer,
                    timeout=args.timeout,
                    use_cache=not args.no_cache,
                )
            )

        report = build_report(manifest, rows, provenance(manifest), manifest_path=args.manifest)
        print(render(report))
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(
                json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8"
            )
            print(f"\nstructured output written to {args.json_out}", file=sys.stderr)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
