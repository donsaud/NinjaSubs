"""Offline independent expected results; never queries providers or synchronizes."""
import argparse
import json
import sys
from itertools import permutations
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.release_scoring import CONTRACT_VERSION
from app.services.release_scoring.engine import evaluate
from app.services.release_scoring.policies import DisplayPolicy, ReferencePolicy, rank
from app.services.release_scoring.profiles import candidate_profiles, target_profile


def run_case(case: dict) -> dict:
    results = tuple(evaluate(target_profile(case["target"]), c) for c in candidate_profiles(case["candidates"]))
    policies: dict[str, DisplayPolicy | ReferencePolicy] = {"discovery_order": ReferencePolicy(discovery=True), "reference_order": ReferencePolicy(),
                "display_order": DisplayPolicy(tuple(case.get("display_languages", ["ara"])))}
    actual: dict[str, Any] = {name: [r.candidate.identity for r in rank(results, policy)] for name, policy in policies.items()}
    actual["scores"] = {r.candidate.identity: r.score for r in results}
    actual["rejections"] = {r.candidate.identity: list(r.rejection_reasons) for r in results if r.rejection_reasons}
    differences = []
    for name, expected in case["expected"].items():
        if isinstance(expected, dict):
            for identity, value in expected.items():
                if actual[name].get(identity) != value:
                    differences.append({"field": name, "identity": identity, "expected": value, "actual": actual[name].get(identity)})
        elif actual[name] != expected:
            differences.append({"field": name, "expected": expected, "actual": actual[name]})
    return {"case_id": case["case_id"], "provenance": case["provenance"], "actual": actual,
            "differences": differences, "contract_version": CONTRACT_VERSION}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=Path("tests/fixtures/weighted_release_matching"))
    parser.add_argument("--bazarr-observations", type=Path, help="Independently exported oracle JSON; never generated from this matcher")
    args = parser.parse_args()
    reports = []
    permutation_count = 0
    for path in sorted(args.fixtures.glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        report = run_case(case)
        if len(case["candidates"]) > 7:
            raise ValueError("Permutation validation is bounded to seven records")
        for order in permutations(case["candidates"]):
            permutation_count += 1
            if run_case({**case, "candidates": list(order)}) != report:
                report["differences"].append({"field": "determinism"})
                break
        reports.append(report)
    # External observed orders are kept separate: do not overwrite expectations
    # or fabricate an oracle from our output. Differences fail this command.
    if args.bazarr_observations:
        observations = json.loads(args.bazarr_observations.read_text(encoding="utf-8"))
        indexed = {r["case_id"]: r for r in reports}
        for item in observations["cases"]:
            report = indexed[item["case_id"]]
            for name, expected in item["expected"].items():
                if report["actual"][name] != expected:
                    report["differences"].append({"field": "external_bazarr_oracle." + name,
                                                   "expected": expected, "actual": report["actual"][name]})
    failed = [r for r in reports if r["differences"]]
    print(json.dumps({"contract_version": CONTRACT_VERSION, "passed": len(reports)-len(failed),
                      "failed": len(failed), "permutations_checked": permutation_count,
                      "failures": failed}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
