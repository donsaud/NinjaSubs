"""Offline fixed-artifact replay. No providers, ALASS execution or serving imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.services.subtitle_matcher import parse_srt_cues
from app.services.sync.alignment import AlignmentAnalyzer
from app.services.sync.boundary_verifier import verify_boundary
from app.services.sync.ninja_points import render_srt
from app.services.sync.timeline_affine import attempt_timeline_affine

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests/replay/corpus.json"


def fixture_path(value: str, root: Path) -> Path:
    """Refuse absolute paths and traversal, including symlink escapes."""
    path = (root / value).resolve()
    if Path(value).is_absolute() or ":" in value or "\\" in value:
        raise ValueError("fixture paths must be portable relative paths")
    if not path.is_relative_to(root.resolve()):
        raise ValueError("fixture outside corpus root")
    return path


def load_text(spec: dict[str, Any], root: Path) -> str:
    path = fixture_path(spec["path"], root)
    if "role" not in spec:
        return path.read_text(encoding="utf-8-sig")
    data = json.loads(path.read_text(encoding="utf-8"))
    return render_srt([(s, e, "Dialogue event." if usable else "Translated by: uploader")
                       for s, e, usable in data["files"][spec["role"]]["cues"]])


def derive(text: str, operations: list[dict[str, Any]]) -> str:
    """Declarative timing mutations; indices refer to complete, zero-based cue order."""
    if not operations:
        return text  # Preserve pristine real artifact formatting/order.
    cues = parse_srt_cues(text)
    for op in operations:
        start, stop = op.get("start", 0), op.get("stop", len(cues))
        kind = op["kind"]
        if kind == "slice":
            cues = cues[start:stop]
        elif kind == "drop":
            cues = cues[:start] + cues[stop:]
        elif kind == "shift":
            cues[start:stop] = [(s + op["ms"], e + op["ms"], t)
                                for s, e, t in cues[start:stop]]
        elif kind == "scale":
            cues = [(round(s * op["slope"] + op.get("intercept", 0)),
                     round(e * op["slope"] + op.get("intercept", 0)), t) for s, e, t in cues]
        elif kind == "end":
            s, _, t = cues[start]
            cues[start] = (s, op["ms"], t)
        elif kind == "swap":
            cues[start], cues[stop] = cues[stop], cues[start]
        elif kind == "split":
            cues = [part for s, e, t in cues for part in
                    ((s, (s + e) // 2, t), ((s + e) // 2, e, t))]
        else:
            raise ValueError("unknown corpus operation")
    return render_srt(cues)


def replay_case(case: dict[str, Any], root: Path = ROOT) -> dict[str, Any]:
    started = time.perf_counter()
    texts = {role: derive(load_text(spec, root), spec.get("operations", []))
             for role, spec in case["fixtures"].items()}
    target, reference, output = (texts[k] for k in ("target", "reference", "transformed"))
    # Explicit recorded measurement context, not an authorization or policy override.
    context = case.get("production_context", {})
    production = AlignmentAnalyzer().analyze(target, output, reference, **context)
    pristine_verdict = production.model_dump(mode="json")
    # Pure existing global model, used only to identify unnecessary complexity.
    affine = attempt_timeline_affine(target, reference)
    # A fit to the reference alone cannot exonerate a different fixed artifact.
    fitted = parse_srt_cues(affine.transformed_text or "")
    fixed = parse_srt_cues(output)
    affine_supports_output = affine.accepted and len(fitted) == len(fixed) and all(
        max(abs(s - fs), abs(e - fe)) <= 350
        for (s, e, _), (fs, fe, _) in zip(fitted, fixed, strict=True)
    )
    simpler = affine_supports_output or production.sync_state.value.startswith("verified")
    shadow = verify_boundary(target, reference, output, global_verified=simpler)
    evidence = asdict(shadow)
    duration = evidence.pop("duration_ms")
    ungated = asdict(verify_boundary(target, reference, output)) if simpler else dict(evidence)
    ungated.pop("duration_ms", None)
    assert production.model_dump(mode="json") == pristine_verdict
    segments = evidence["segments"]
    result = {
        "case_id": case["case_id"], "production": pristine_verdict,
        "simpler_model": {"accepted": affine.accepted,
                          "supports_fixed_output": affine_supports_output,
                          "classification": affine.diagnostics.classification,
                          "slope": affine.slope, "intercept_ms": affine.intercept_ms},
        "boundary_should_run": not simpler, "shadow": evidence,
        "ungated_shadow": ungated,
        "input_sha256": {role: hashlib.sha256(text.encode()).hexdigest()
                         for role, text in texts.items()},
        "segment_count": len(segments),
    }
    if "served" in texts:
        served = asdict(verify_boundary(target, reference, texts["served"],
                                       global_verified=simpler, artifact="final_served"))
        served.pop("duration_ms")
        result["final_served_shadow"] = served
    return {"evidence": result, "observations": {"shadow_duration_ms": duration,
                                                 "replay_duration_ms": (time.perf_counter()-started)*1000}}


def replay(manifest: Path = MANIFEST, root: Path = ROOT) -> dict[str, Any]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    cases = data["cases"]
    if not cases:
        raise ValueError("empty corpus")
    ids = [case["case_id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case_id")
    results = [replay_case(case, root) for case in cases]
    counts = dict.fromkeys(("expected_positives", "true_positives", "false_positives",
                           "expected_negatives", "true_negatives", "false_negatives",
                           "unnecessary_escalations"), 0)
    mismatches = []
    for case, result in zip(cases, results, strict=True):
        evidence = result["evidence"]
        expected, accepted = case["expected_shadow_outcome"], evidence["shadow"]["accepted"]
        counts["expected_positives" if expected else "expected_negatives"] += 1
        counts[("true_positives" if accepted else "false_negatives") if expected else
               ("false_positives" if accepted else "true_negatives")] += 1
        if accepted and not case["boundary_should_run"]:
            counts["unnecessary_escalations"] += 1
        reasons = evidence["shadow"]["reasons"]
        model = case["expected_model"]
        model_matches = (model in {"UNSUPPORTED", "SAME_EDITION"}
                         or (model == "EDGE_DISCONTINUITY" and accepted)
                         or (evidence["simpler_model"]["supports_fixed_output"]
                             and evidence["simpler_model"]["classification"] == model))
        if (not model_matches or accepted != expected or ungated_acceptance_mismatch(case, evidence)
                or evidence["boundary_should_run"] != case["boundary_should_run"]
                or (case["expected_reason"] and case["expected_reason"] not in reasons)):
            mismatches.append(case["case_id"])
    return {"schema_version": 1, "cases": results, "aggregate": counts,
            "expectation_mismatches": mismatches}


def ungated_acceptance_mismatch(case: dict[str, Any], evidence: dict[str, Any]) -> bool:
    """Even simpler-model controls must resist false positives without the guard."""
    return bool(evidence["ungated_shadow"]["accepted"]) != case["expected_shadow_outcome"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-only", action="store_true",
                        help="omit variable wall-clock observations for reproducible JSON")
    args = parser.parse_args()
    report = replay(args.manifest)
    if args.evidence_only:
        for result in report["cases"]:
            result.pop("observations")
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], sort_keys=True))
    return int(bool(report["expectation_mismatches"] or report["aggregate"]["false_positives"]))


if __name__ == "__main__":
    raise SystemExit(main())
