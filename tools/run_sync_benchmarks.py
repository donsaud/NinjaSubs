"""Run every offline synchronization suite in one command.

    python tools/run_sync_benchmarks.py

Or orchestrates nothing and touches no production state. It runs the suites
that exist, reports the ones that cannot run, and says plainly when real media
is unavailable rather than treating its absence as a pass.

The exit code is a summary of what ran, not a pass/fail judgement: these
benchmarks measure behaviour, and a suite that reports a known failure is doing
its job. A non-zero exit means a suite could not run at all.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
GOLDEN_MANIFEST = REPO / "tests" / "fixtures" / "golden_sync" / "manifest.json"
REAL_MEDIA_SAMPLE = REPO / "tests" / "fixtures" / "golden_sync" / "real_media_sample.redacted.json"


def _has(binary: str) -> bool:
    import shutil

    return shutil.which(binary) is not None


def _run(label: str, argv: list[str], *, requires: tuple[str, ...] = ()) -> dict[str, Any]:
    """Run one suite, capturing its output and timing it."""
    missing = [name for name in requires if not _has(name)]
    if missing:
        return {
            "label": label,
            "status": "NOT AVAILABLE",
            "reason": f"requires {', '.join(missing)}",
            "seconds": 0.0,
        }
    started = time.monotonic()
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            cwd=REPO,
            timeout=1800,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "label": label,
            "status": "ERROR",
            "reason": str(exc),
            "seconds": round(time.monotonic() - started, 2),
        }
    elapsed = round(time.monotonic() - started, 2)
    if result.returncode != 0:
        return {
            "label": label,
            "status": "ERROR",
            "reason": (result.stderr or result.stdout or "").strip()[-400:],
            "seconds": elapsed,
            "returncode": result.returncode,
        }
    return {
        "label": label,
        "status": "OK",
        "seconds": elapsed,
        "output": result.stdout,
    }


def _summarise_golden(output: str) -> dict[str, Any]:
    """Pull the headline numbers out of the golden report, if present."""
    facts: dict[str, Any] = {}
    for line in output.splitlines():
        stripped = line.strip()
        # Exact, positional matches only. A substring search also matched the
        # `after_video_validation_false_verified` line, which reported a
        # different number entirely.
        for key, prefix in (
            ("false_verified", "false_verified"),
            ("false_resynced", "false_resynced"),
            ("missed_resyncable", "missed_resyncable"),
        ):
            if stripped.startswith(prefix + " :"):
                facts[key] = stripped.split(":", 1)[1].strip()
        if stripped.startswith("Unknown-label candidates"):
            facts["unknown_labels"] = stripped.split(":", 1)[1].strip()
        if stripped.startswith("Ground truth incorrect"):
            facts["ground_truth_incorrect"] = stripped.split(":", 1)[1].strip()
    return facts


def _real_media_status(manifest: Path | None) -> tuple[str, str]:
    """Whether real media is actually present, stated without spin."""
    if manifest is None:
        return "NOT AVAILABLE", "no manifest supplied"
    if not manifest.is_file():
        return "NOT AVAILABLE", f"manifest not found: {manifest}"
    try:
        sys.path.insert(0, str(REPO))
        from app.services.sync.golden import load_real_media_manifest

        parsed = load_real_media_manifest(manifest)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return "NOT AVAILABLE", f"manifest unreadable: {exc}"
    present = parsed.available_cases()
    if not present:
        return (
            "NOT AVAILABLE",
            f"{len(parsed.cases)} cases declared, 0 media files present locally",
        )
    return "AVAILABLE", f"{len(present)} of {len(parsed.cases)} cases have media"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run every offline sync suite")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--real-media", type=Path, default=None)
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help="skip the suites that generate and decode audio",
    )
    args = parser.parse_args(argv)

    python = sys.executable
    suites: list[dict[str, Any]] = []

    suites.append(
        _run(
            "golden synchronization benchmark",
            [python, "tools/evaluate_golden_sync.py", str(GOLDEN_MANIFEST)],
        )
    )

    if not args.skip_slow:
        suites.append(
            _run(
                "audio landmark stress (fixed vs adaptive)",
                [
                    python,
                    "tools/stress_audio_landmarks.py",
                    "--workdir",
                    "subs_cache/stress_audio",
                    "--compare",
                ],
                requires=("ffmpeg",),
            )
        )
        suites.append(
            _run(
                "task-level cut detection",
                [
                    python,
                    "tools/benchmark_cut_detection.py",
                    "--workdir",
                    "subs_cache/cutbench",
                ],
                requires=("ffmpeg",),
            )
        )

    status, reason = _real_media_status(args.real_media)
    real_media: dict[str, Any] = {"status": status, "reason": reason, "report": None}
    if status == "AVAILABLE" and args.real_media is not None:
        suite = _run(
            "real media validation",
            [python, "tools/evaluate_real_media.py", str(args.real_media)],
        )
        suites.append(suite)
        real_media["report"] = "OK" if suite["status"] == "OK" else suite["status"]

    payload = {
        "suites": [
            {k: v for k, v in suite.items() if k != "output"} for suite in suites
        ],
        "real_media": {k: v for k, v in real_media.items() if k != "report"},
    }

    print("=" * 74)
    print("OFFLINE SYNCHRONIZATION SUITES")
    print("=" * 74)
    for suite in suites:
        marker = {"OK": "ok ", "NOT AVAILABLE": "-- ", "ERROR": "ERR"}.get(
            suite["status"], "?? "
        )
        print(f"  [{marker}] {suite['label']:<44} {suite['seconds']:>7.2f}s")
        if suite["status"] == "ERROR":
            print(f"          {suite.get('reason', '')[:200]}")
        elif suite["status"] == "NOT AVAILABLE":
            print(f"          {suite.get('reason', '')}")

    print()
    print("  REAL MEDIA: " + real_media["status"])
    print(f"    {real_media['reason']}")
    if real_media["status"] != "AVAILABLE":
        print("    Absence is not a pass. Every number above is synthetic evidence")
        print("    of behaviour, not of real-world accuracy.")

    golden = next((s for s in suites if "golden" in s["label"]), None)
    if golden and golden["status"] == "OK":
        facts = _summarise_golden(golden["output"])
        if facts:
            print()
            print("  Golden benchmark headline")
            for key, value in facts.items():
                print(f"    {key}: {value}")

    print()
    errors = [s for s in suites if s["status"] == "ERROR"]
    unavailable = [s for s in suites if s["status"] == "NOT AVAILABLE"]
    print(
        f"  {len(suites) - len(errors) - len(unavailable)} run, "
        f"{len(unavailable)} unavailable, {len(errors)} errored"
    )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"\nstructured output written to {args.json_out}", file=sys.stderr)

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
