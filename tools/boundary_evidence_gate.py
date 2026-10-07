"""Offline gate accounting: support/derived evidence cannot satisfy REAL minima."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.sync.parser_projection import sha  # noqa: E402
from tools.capture_v2 import relative_file  # noqa: E402
from tools.sync_replay import ROOT, load_text, replay  # noqa: E402


def inventory(root: Path, captures: Path | None = None, *, safety_regressions: int | None = None,
              unknown_triplet: Path | None = None) -> dict[str, Any]:
    manifest = json.loads((root / "tests/replay/corpus.json").read_bytes())
    policy = json.loads((root / "tests/replay/evidence_gate.json").read_bytes())
    report = replay(root / "tests/replay/corpus.json", root)
    rows = []
    for case, result in zip(manifest["cases"], report["cases"], strict=True):
        ev = result["evidence"]
        identity = case.get("independence_identity")
        if identity is None:
            identity = [sha(load_text(case["fixtures"][r], root).encode()) for r in ("target", "reference")]
        kind = case["provenance"].upper()
        if kind == "REAL":
            verified_origin = "capture_v2" in case
            seed = policy["historical_positive"]
            if case["case_id"] == seed["case_id"]:
                path = relative_file(root, case["fixtures"]["target"]["path"])
                verified_origin = sha(path.read_bytes()) == seed["fixture_sha256"]
                identity = seed["identity"]
            global_seed = policy["historical_global_control"]
            if case["case_id"] == global_seed["case_id"]:
                verified_origin = all(sha(relative_file(root, case["fixtures"][role]["path"]).read_bytes())
                                      == expected for role, expected in global_seed["fixture_sha256"].items())
            if ev["shadow"]["accepted"] and ev["shadow"]["classification"] == "EDGE_DISCONTINUITY":
                kind = "REPLAYABLE_REAL_POSITIVE" if verified_origin else "SUPPORTING_REAL_LIVE_POSITIVE"
            else:
                kind = "REPLAYABLE_REAL_NEGATIVE" if verified_origin else "SUPPORTING_REAL_LIVE_NEGATIVE"
        rows.append({"case_id": case["case_id"], "identity": identity, "category": kind,
                     "media_identity": (["tt0773262", 8, 4] if case["case_id"] == "dexter_real"
                                        else case.get("media_identity")),
                     "meaningful_negative": bool(case.get("meaningful_negative", False))
                     and case.get("case_type") not in {"GLOBAL_AFFINE", "CONSTANT_OFFSET"},
                     "accepted": ev["shadow"]["accepted"]})
    if captures is not None:
        known = {tuple(c["identity"]): c for c in policy["supporting_live"]}
        for p in sorted(captures.glob('*/metadata.json')):
            value = json.loads(p.read_bytes())
            original = p.parent / "pristine_target.srt"
            target_sha = sha(original.read_bytes()) if original.exists() else "missing"
            attempts = value.get("attempts", [])
            row = {"case_id": p.parent.name, "identity": [target_sha, "unknown"],
                   "category": "UNKNOWN", "meaningful_negative": False, "accepted": None,
                   "media_identity": [value.get(k) for k in ("imdb_id", "season", "episode")]}
            for attempt in attempts:
                identity = [target_sha, attempt.get("reference_sha256")]
                anchor = known.get(tuple(identity))
                if anchor is None or attempt.get("raw_transformed_sha256") != anchor["raw_sha256"]:
                    continue
                expected = value.get("artifact_sha256", {})
                if not expected or any(sha(relative_file(p.parent, name).read_bytes()) != digest
                                       for name, digest in expected.items()):
                    continue
                if (sha(relative_file(p.parent, attempt["directory"] + '/selected_reference.srt').read_bytes())
                        != identity[1] or sha(relative_file(p.parent, attempt["directory"] + '/raw_transformed.srt').read_bytes())
                        != anchor["raw_sha256"]):
                    continue
                row.update(identity=identity, category="SUPPORTING_REAL_LIVE_NEGATIVE", accepted=False)
            rows.append(row)
    if unknown_triplet is not None:
        targets = sorted(unknown_triplet.glob("target.srt*"))
        references = sorted(unknown_triplet.glob("ref.srt*"))
        if len(targets) == len(references) == 1:
            rows.append({"case_id": "unproven-local-triplet",
                         "identity": [sha(targets[0].read_bytes()), sha(references[0].read_bytes())],
                         "category": "UNKNOWN", "meaningful_negative": False, "accepted": None,
                   "media_identity": [value.get(k) for k in ("imdb_id", "season", "episode")]})
    report["safety_regressions"] = safety_regressions
    counts = gate_counts(rows, report)
    return {"counts": counts, "evidence": rows, "replay": report["aggregate_by_provenance"],
            "promotion": "PASS" if counts["gate_passed"] else "BLOCKED"}


def gate_counts(rows: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
    names = ("REPLAYABLE_REAL_POSITIVE", "REPLAYABLE_REAL_NEGATIVE", "SUPPORTING_REAL_LIVE_POSITIVE",
             "SUPPORTING_REAL_LIVE_NEGATIVE", "DERIVED", "SYNTHETIC", "UNKNOWN")
    counts: dict[str, Any] = {name: 0 for name in names}
    seen = set()
    meaningful = 0
    # Replayable takes priority over the same pair's repeated supporting capture.
    ordered = sorted(rows, key=lambda r: not r["category"].startswith("REPLAYABLE"))
    for row in ordered:
        # Derived controls remain separate controls, never independent REAL cases.
        identity = tuple(row["identity"])
        key = ("control", row["case_id"], row["category"]) if row["category"] in {
            "DERIVED", "SYNTHETIC"} else ("pair", *identity)
        keys = [key]
        if row["category"] not in {"DERIVED", "SYNTHETIC", "UNKNOWN"} and row.get("media_identity"):
            keys.append(("media", *row["media_identity"]))
        if any(key in seen for key in keys):
            continue
        seen.update(keys)
        counts[row["category"]] += 1
        if row["category"] == "REPLAYABLE_REAL_NEGATIVE" and row.get("meaningful_negative"):
            meaningful += 1
    counts["replayable_real_meaningful_negatives"] = meaningful
    counts["false_positives"] = report["aggregate"]["false_positives"]
    counts["unnecessary_escalations"] = report["aggregate"]["unnecessary_escalations"]
    counts["expectation_mismatches"] = len(report["expectation_mismatches"])
    counts["safety_regressions"] = report.get("safety_regressions")
    counts["gate_passed"] = (counts["safety_regressions"] == 0 and counts["REPLAYABLE_REAL_POSITIVE"] >= 2 and meaningful >= 2
                              and counts["false_positives"] == counts["unnecessary_escalations"] == 0
                              and not report["expectation_mismatches"])
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path)
    parser.add_argument("--unknown-triplet", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--safety-regressions", type=int, choices=range(0, 1000))
    args = parser.parse_args()
    result = inventory(ROOT, args.captures, safety_regressions=args.safety_regressions,
                       unknown_triplet=args.unknown_triplet)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"counts": result["counts"], "promotion": result["promotion"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
