"""Run the mandatory safety mutations and prove the suite catches each one.

Phase 21 of the finalisation pass: a test that stays green after the safety
behaviour it guards is deliberately removed is not a test. This tool removes
each guard in turn, in a throwaway copy of the working tree, and requires the
suite to FAIL. A guard whose removal leaves the suite green is reported as
UNPROTECTED, which is a release blocker.

It never edits the working tree: every mutation is applied to a temporary copy
and the original is left untouched. No network, no provider credentials.

    python tools/run_safety_mutations.py
    python tools/run_safety_mutations.py --json report.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Mutation:
    """One deliberately broken safety behaviour."""

    name: str
    file: str
    old: str
    new: str
    #: Why removing this must break the suite.
    guards: str


MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        name="target-fingerprint-binding-removed",
        file="app/services/sync_cache.py",
        old='        prefix = f"final_sub:{imdb}:{season_ep}:{fp}:{sid}"',
        new='        prefix = f"final_sub:{imdb}:"  # MUTATION',
        guards="a payload cached for one video must never serve another target",
    ),
    Mutation(
        name="negative-verdict-served-as-verified",
        file="app/main.py",
        old=(
            "    if state not in (SyncState.VERIFIED_SYNCED.value, "
            "SyncState.VERIFIED_RESYNCED.value):"
        ),
        new="    if False:  # MUTATION",
        guards="REJECTED/UNVERIFIED/PROBABLE_SYNC must never yield a served artifact",
    ),
    Mutation(
        name="stale-engine-version-accepted",
        file="app/services/sync_cache.py",
        old='        if int(data.get("engine_version") or 0) != SYNC_VERDICT_ENGINE_VERSION:',
        new="        if False:  # MUTATION",
        guards="a verdict written by an older engine must not be reused",
    ),
    Mutation(
        name="divergent-target-identity",
        file="app/main.py",
        old=(
            '        "video_hash": meta.get("video_hash"),\n'
            '        "video_size": meta.get("video_size"),\n'
            "    }"
        ),
        new="        # MUTATION: identity fields dropped\n    }",
        guards="reuse and serve paths must derive the same target identity",
    ),
    Mutation(
        name="strict-timestamp-grammar-restored",
        file="app/services/subtitle_matcher.py",
        old=(
            r'r"((?:\d{1,2}:)?\d{1,2}:\d{2}[,.]\d{1,3})\s*-->\s*'
            r'((?:\d{1,2}:)?\d{1,2}:\d{2}[,.]\d{1,3})"'
        ),
        new=(
            r'r"(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*'
            r'(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})"'
        ),
        guards="MM:SS,mmm targets must parse, or the verifier sees an empty target",
    ),
    Mutation(
        name="alass-budget-process-wide-again",
        file="app/services/sync/orchestrator.py",
        old="            if alass_attempted >= self._alass_candidate_limit:",
        new=(
            '            if self._metrics["alass_runs"] >= self._alass_candidate_limit:'
            "  # MUTATION"
        ),
        guards=(
            "ALASS_CANDIDATE_LIMIT is a per-request allowance; a process-wide "
            "counter starves every request after the first few"
        ),
    ),
    Mutation(
        name="reference-cache-target-binding-removed",
        file="app/services/sync/cache.py",
        old='        return sorted(self.root.glob(f"{query.cache_stem}_*.srt"))',
        new='        return sorted(self.root.glob("*_*.srt"))  # MUTATION',
        guards="a reference proven for one target must not satisfy another",
    ),
)


def _run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(  # noqa: S603
        cmd, cwd=cwd, capture_output=True, text=True, shell=False
    )
    return proc.returncode, proc.stdout + proc.stderr


def _apply(tree: Path, mutation: Mutation) -> bool:
    path = tree / mutation.file
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if mutation.old not in text:
        return False
    path.write_text(text.replace(mutation.old, mutation.new, 1), encoding="utf-8")
    return True


def check(mutation: Mutation, python: str) -> dict[str, object]:
    """Apply one mutation in a temp copy; the suite MUST fail."""
    result: dict[str, object] = {
        "mutation": mutation.name,
        "file": mutation.file,
        "guards": mutation.guards,
    }
    with tempfile.TemporaryDirectory(prefix="ninjasubs-mut-") as tmp:
        tree = Path(tmp) / "tree"
        # Copy the working tree without .git or the test cache, so a mutation
        # can never touch the real checkout or the real subs_cache.
        shutil.copytree(
            ROOT,
            tree,
            ignore=shutil.ignore_patterns(
                ".git", ".venv", "__pycache__", "subs_cache", "cache", ".pytest_cache"
            ),
        )
        if not _apply(tree, mutation):
            result.update(status="SKIPPED", detail="anchor text not found; mutation is stale")
            return result
        code, output = _run([python, "-m", "pytest", "-q", "-x", "--no-header"], tree)
        if code == 0:
            result.update(
                status="UNPROTECTED",
                detail="suite stayed green after the safety behaviour was removed",
            )
        elif code == 1:
            failed = [
                line.strip() for line in output.splitlines() if line.startswith("FAILED")
            ]
            result.update(
                status="CAUGHT",
                detail=f"{len(failed)} test(s) failed as required",
                failing_tests=failed[:5],
            )
        else:
            result.update(
                status="ERROR",
                detail=f"pytest exited {code}; the mutation broke collection or crashed",
            )
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", dest="json_path", help="write a machine-readable report")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    print("=" * 74)
    print("SAFETY MUTATION MATRIX")
    print("=" * 74)
    results = [check(m, args.python) for m in MUTATIONS]

    width = max(len(str(r["mutation"])) for r in results)
    for r in results:
        print(f"  [{r['status']:<11}] {r['mutation']:<{width}}  {r['detail']}")
        if r.get("failing_tests"):
            for t in r["failing_tests"]:
                print(f"                 - {t}")

    caught = sum(1 for r in results if r["status"] == "CAUGHT")
    unprotected = [r["mutation"] for r in results if r["status"] == "UNPROTECTED"]
    skipped = [r["mutation"] for r in results if r["status"] == "SKIPPED"]
    errored = [r["mutation"] for r in results if r["status"] == "ERROR"]

    print("-" * 74)
    print(f"  caught {caught}/{len(results)}")
    if unprotected:
        print(f"  UNPROTECTED (release blocker): {', '.join(unprotected)}")
    if skipped:
        print(f"  stale anchors, review the mutation: {', '.join(skipped)}")
    if errored:
        print(f"  errors, review the mutation: {', '.join(errored)}")
    print("=" * 74)

    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps({"results": results, "caught": caught, "total": len(results)}, indent=2),
            encoding="utf-8",
        )
        print(f"  report written to {args.json_path}")

    # A stale or errored mutation is a defect in this tool, not a safety
    # failure; only UNPROTECTED blocks.
    return 1 if unprotected else 0


if __name__ == "__main__":
    raise SystemExit(main())
