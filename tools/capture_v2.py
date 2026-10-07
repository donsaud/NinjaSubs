"""Offline, explicit-provenance import/replay of forward-only capture V2."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.extractor import decode_subtitle_bytes  # noqa: E402
from app.services.sync.boundary_verifier import verify_boundary  # noqa: E402
from app.services.sync.case_capture import safe_payload  # noqa: E402
from app.services.sync.parser_projection import (  # noqa: E402
    canonical_bytes,
    implementation_sha,
    project,
    render_projection,
    sha,
)
from tools.import_sync_case import safe_id  # noqa: E402


def relative_file(root: Path, name: str) -> Path:
    if (not isinstance(name, str) or ":" in name or "\\" in name
            or Path(name).is_absolute() or ".." in Path(name).parts):
        raise ValueError("unsafe artifact path")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("artifact outside case")
    return path


def evidence(result: Any) -> dict[str, Any]:
    result = asdict(result)
    result.pop("duration_ms")
    return json.loads(canonical_bytes(result))


def check_capture(root: Path, attempt: str) -> tuple[dict[str, bytes], dict[str, Any]]:
    if not re.fullmatch(r"attempt-\d{2}", attempt):
        raise ValueError("explicit attempt required")
    metadata = json.loads((root / "metadata.json").read_bytes())
    if metadata.get("schema_version") != 2:
        raise ValueError("complete forward V2 capture required")
    rows = [a for a in metadata["attempts"] if a["directory"] == attempt]
    if len(rows) != 1 or rows[0].get("producer_observation") != "alass_output_file":
        raise ValueError("exact ALASS producer observation required")
    selected = rows[0]
    media = [metadata.get(k) for k in ("imdb_id", "season", "episode")]
    if not isinstance(media[0], str) or not re.fullmatch(r"tt\d{5,12}", media[0]):
        raise ValueError("bounded media identity required")
    if any(v is not None and (type(v) is not int or not 1 <= v <= 999) for v in media[1:]):
        raise ValueError("invalid episode identity")
    if selected.get("boundary_shadow_status") != "completed":
        raise ValueError("completed live shadow observation required")
    artifacts = {}
    manifest = metadata.get("artifact_sha256", {})
    if not isinstance(manifest, dict) or len(manifest) > 160:
        raise ValueError("bounded hash manifest required")
    for name, expected in manifest.items():
        path = relative_file(root, name)
        if path.stat().st_size > 2_000_000:
            raise ValueError("artifact limit")
        raw = path.read_bytes()
        if sha(raw) != expected:
            raise ValueError("artifact SHA mismatch")
        if path.suffix == ".srt" and not safe_payload(raw):
            raise ValueError("artifact privacy refusal")
        artifacts[name] = raw
    required = ["pristine_target.srt", f"{attempt}/selected_reference.srt",
                f"{attempt}/raw_transformed.srt", f"{attempt}/boundary_parser_projection.json",
                *(f"{attempt}/boundary_{role}_input.srt"
                  for role in ("target", "reference", "transformed"))]
    final = "final_served." + metadata.get("final_format", "")
    if final not in {"final_served.srt", "final_served.ass", "final_served.ssa", "final_served.vtt"}:
        raise ValueError("exact final response required")
    required.append(final)
    if any(name not in artifacts for name in required):
        raise ValueError("incomplete V2 provenance")
    identities = {"pristine_target.srt": metadata["target_sha256"],
                  f"{attempt}/selected_reference.srt": selected["reference_sha256"],
                  f"{attempt}/raw_transformed.srt": selected["raw_transformed_sha256"],
                  final: metadata["final_served_sha256"],
                  f"{attempt}/boundary_parser_projection.json": selected.get("parser_projection_sha256")}
    if any(sha(artifacts[name]) != expected for name, expected in identities.items()):
        raise ValueError("identity SHA mismatch")
    projection = json.loads(artifacts[f"{attempt}/boundary_parser_projection.json"])
    if (projection.get("version") != 2
            or projection.get("parser_implementation_sha256") != implementation_sha()
            or projection.get("original_target_sha256") != metadata["target_sha256"]):
        raise ValueError("parser drift or original mismatch")
    language = metadata.get("parser_language")
    if language is not None and not re.fullmatch(r"[a-z]{2,3}", language):
        raise ValueError("unsafe parser language")
    original = artifacts["pristine_target.srt"]
    texts = {role: artifacts[f"{attempt}/boundary_{role}_input.srt"].decode("utf-8")
             for role in ("target", "reference", "transformed")}
    if texts["target"] != decode_subtitle_bytes(original, lang=language):
        raise ValueError("original-to-parser relation mismatch")
    if texts["reference"] != artifacts[f"{attempt}/selected_reference.srt"].decode("utf-8"):
        raise ValueError("reference representation mismatch")
    normalized_raw = artifacts[f"{attempt}/raw_transformed.srt"].decode("utf-8").replace(
        "\r\n", "\n").replace("\r", "\n")
    if texts["transformed"] != normalized_raw or sha(normalized_raw.encode()) != selected.get(
            "verifier_transformed_sha256"):
        raise ValueError("raw-to-verifier relation mismatch")
    if any(projection["input_sha256"].get(role) != sha(text.encode())
           or projection["projections"].get(role) != project(text)
           for role, text in texts.items()):
        raise ValueError("tampered parser projection")
    direct = evidence(verify_boundary(*(texts[r] for r in ("target", "reference", "transformed"))))
    projected = evidence(verify_boundary(*(render_projection(projection["projections"][r])
                                        for r in ("target", "reference", "transformed"))))
    recorded = dict(selected["boundary_shadow"])
    recorded.pop("duration_ms", None)
    if direct != projected or direct != json.loads(canonical_bytes(recorded)):
        raise ValueError("live/replay semantic mismatch")
    legacy = selected.get("verifier", {})
    if legacy.get("verification") not in {"unknown", "predicted"}:
        raise ValueError("globally verified or missing legacy evidence")
    if legacy.get("alass_applied") is not True or legacy.get("alass_successful") is not True:
        raise ValueError("successful live ALASS evidence required")
    legacy = {"verification": legacy["verification"], "alass_applied": True,
              "alass_successful": True}
    safe_meta = {"schema_version": 2, "parser_language": language,
                 **dict(zip(("imdb_id", "season", "episode"), media, strict=True)),
                 "target_sha256": metadata["target_sha256"],
                 "final_format": metadata["final_format"],
                 "final_served_sha256": metadata["final_served_sha256"],
                 "artifact_sha256": {n: manifest[n] for n in required},
                 "attempts": [{"directory": attempt,
                               "producer_observation": "alass_output_file",
                               "boundary_shadow_status": "completed",
                               "reference_sha256": selected["reference_sha256"],
                               "raw_transformed_sha256": selected["raw_transformed_sha256"],
                               "verifier_transformed_sha256": selected["verifier_transformed_sha256"],
                               "parser_projection_sha256": selected["parser_projection_sha256"],
                               "boundary_shadow": recorded, "verifier": legacy}]}
    return {name: artifacts[name] for name in required}, {"metadata": safe_meta,
              "shadow": direct, "identity": [metadata["target_sha256"], selected["reference_sha256"]],
              "media_identity": media}


def replay_v2(case: dict[str, Any], root: Path) -> dict[str, Any]:
    if case.get("live_origin_confirmed") is not True or case.get("independent_outcome_confirmed") is not True:
        raise ValueError("missing independent live origin confirmation")
    folder = relative_file(root, case["capture_v2"]["path"])
    if sha((folder / "metadata.json").read_bytes()) != case["capture_v2"].get("metadata_sha256"):
        raise ValueError("case metadata SHA mismatch")
    _, result = check_capture(folder, case["capture_v2"]["attempt"])
    if (result["identity"] != case["independence_identity"]
            or result["media_identity"] != case.get("media_identity")):
        raise ValueError("case identity mismatch")
    shadow = result["shadow"]
    ev = {"case_id": case["case_id"], "production": {"source": "recorded_live_only"},
          "simpler_model": {"accepted": False, "supports_fixed_output": False,
                            "classification": "not_recomputed_from_neutral_projection"},
          "boundary_should_run": True, "shadow": shadow, "ungated_shadow": shadow,
          "input_sha256": dict(zip(("target", "reference"), result["identity"], strict=True)),
          "segment_count": len(shadow["segments"])}
    return {"evidence": ev, "observations": {}}


def _import_capture(root: Path, source: Path, attempt: str, case_id: str, *,
                   live_confirmed: bool, expected_boundary_accepted: bool, outcome_confirmed: bool,
                   meaningful_negative: bool = False) -> dict[str, Any]:
    safe_id(case_id)
    if not live_confirmed:
        raise ValueError("independent REAL live provenance confirmation required")
    if type(expected_boundary_accepted) is not bool or not outcome_confirmed:
        raise ValueError("independent expected outcome confirmation required")
    artifacts, result = check_capture(source, attempt)
    if result["shadow"]["accepted"] != expected_boundary_accepted:
        raise ValueError("independent expectation differs from replay")
    manifest = root / "tests/replay/corpus.json"
    corpus = json.loads(manifest.read_bytes())
    if any(c["case_id"] == case_id or c.get("independence_identity") == result["identity"]
           for c in corpus["cases"]):
        raise ValueError("duplicate case or target/reference pair")
    # Existing legacy seeds also participate in pair identity deduplication.
    from tools.sync_replay import load_text
    for c in corpus["cases"]:
        if "capture_v2" not in c:
            identity = [sha(load_text(c["fixtures"][r], root).encode()) for r in ("target", "reference")]
            if identity == result["identity"]:
                raise ValueError("duplicate existing pair")
    accepted = result["shadow"]["accepted"]
    destination = root / "tests/replay/cases" / case_id
    if destination.exists():
        raise ValueError("case directory exists")
    entry = {"case_id": case_id, "provenance": "real", "source_case_id": None,
             "live_origin_confirmed": True, "independent_outcome_confirmed": True,
             "independence_identity": result["identity"], "media_identity": result["media_identity"],
             "meaningful_negative": meaningful_negative and not accepted,
             "case_type": "EDGE_DISCONTINUITY" if accepted else "INSUFFICIENT_EVIDENCE_REJECT",
             "expected_model": "EDGE_DISCONTINUITY" if accepted else "UNSUPPORTED",
             "expected_shadow_outcome": accepted, "boundary_should_run": True,
             "expected_reason": result["shadow"]["reasons"][0] if result["shadow"]["reasons"] else "",
             "capture_v2": {"path": destination.relative_to(root).as_posix(), "attempt": attempt,
                            "metadata_sha256": sha(canonical_bytes(result["metadata"]))}}
    # Artifacts published first; manifest atomically last. No existing files overwritten.
    with tempfile.TemporaryDirectory(prefix="capture-v2-", dir=manifest.parent) as temporary:
        stage = Path(temporary) / "case"
        stage.mkdir()
        for name, raw in artifacts.items():
            path = relative_file(stage, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        (stage / "metadata.json").write_bytes(canonical_bytes(result["metadata"]))
        check_capture(stage, attempt)
        # Fail before publication if source changed while being checked/copied.
        again, _ = check_capture(source, attempt)
        if again != artifacts:
            raise ValueError("source changed during import")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(stage), str(destination))
        try:
            corpus["cases"].append(entry)
            updated = Path(temporary) / "corpus.json"
            updated.write_bytes(canonical_bytes(corpus))
            updated.replace(manifest)
        except Exception:
            shutil.rmtree(destination)
            raise
    return entry


def import_capture(root: Path, source: Path, attempt: str, case_id: str, *,
                   live_confirmed: bool, expected_boundary_accepted: bool, outcome_confirmed: bool,
                   meaningful_negative: bool = False) -> dict[str, Any]:
    lock = root / "tests/replay/.case-import.lock"
    try:
        lock.touch(exist_ok=False)
    except FileExistsError:
        raise ValueError("another import is active") from None
    try:
        return _import_capture(root, source, attempt, case_id,
                               live_confirmed=live_confirmed,
                               expected_boundary_accepted=expected_boundary_accepted,
                               outcome_confirmed=outcome_confirmed,
                               meaningful_negative=meaningful_negative)
    finally:
        lock.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--real-live-confirmed", action="store_true")
    parser.add_argument("--expected-boundary-accepted", required=True, choices=("true", "false"))
    parser.add_argument("--independent-outcome-confirmed", action="store_true")
    parser.add_argument("--meaningful-negative-confirmed", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    entry = import_capture(root, args.source, args.attempt, args.case_id,
                           live_confirmed=args.real_live_confirmed,
                           expected_boundary_accepted=args.expected_boundary_accepted == "true",
                           outcome_confirmed=args.independent_outcome_confirmed,
                           meaningful_negative=args.meaningful_negative_confirmed)
    print(json.dumps({"case_id": entry["case_id"], "provenance": "real",
                      "accepted": entry["expected_shadow_outcome"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
