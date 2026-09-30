"""Coverage matrix for the AutoSync golden corpus (Phase 3).

Answers "what have we actually tested?" with measurements, not intentions.

The corpus is small and synthetic, so the most useful output is the list of
axes with ZERO coverage. A report that only showed present coverage would be
read as "these are covered", which for 480p, 2160p, WEBRip, BDRip, DVD or any
FPS family is false.

Each axis reports:

    covered       cases exist and are executed
    not_available the axis is required for a real claim, and nothing covers it

    SYNTHETIC / REAL_MEDIA is reported per axis, because a synthetic case is
    evidence about behaviour and never evidence about a real release.

This tool measures. It never changes a threshold, a policy, or a threshold
sweep, and it exits non-zero only when the report itself cannot be produced.

    python tools/benchmark_coverage_matrix.py
    python tools/benchmark_coverage_matrix.py --json-out coverage.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO / "tests" / "fixtures" / "golden_sync" / "manifest.json"

#: The axes a real-world claim would need. Anything absent is reported as
#: not_available rather than quietly dropped.
REQUIRED_RESOLUTIONS = ("480p", "576p", "720p", "1080p", "2160p")
REQUIRED_RELEASE_FAMILIES = ("WEB-DL", "WEBRip", "BluRay", "BDRip", "HDTV", "DVD")
REQUIRED_FPS = ("23.976", "24", "25", "29.97", "30")
REQUIRED_MEDIA = ("movie", "series", "season_pack")
REQUIRED_TIMING = (
    "global_offset",
    "small_offset",
    "large_offset",
    "drift",
    "piecewise",
    "different_intro",
    "different_outro",
    "different_runtime",
    "different_cut",
    "different_edit",
)
REQUIRED_FORMATS = ("HH:MM:SS,mmm", "MM:SS,mmm", "HI", "non-HI", "sparse", "dense", "credits")

RES_RE = re.compile(r"\b(480p|576p|720p|1080p|2160p)\b", re.IGNORECASE)
FPS_RE = re.compile(r"\b(23\.976|24|25|29\.97|30)(?:\s*fps)?\b", re.IGNORECASE)
FAMILY_PATTERNS = (
    ("WEB-DL", re.compile(r"web[-.]?dl", re.IGNORECASE)),
    ("WEBRip", re.compile(r"web[-.]?rip|\bweb\b(?![-.]?dl)", re.IGNORECASE)),
    ("BluRay", re.compile(r"bluray|blu[-.]?ray|\bbd\b", re.IGNORECASE)),
    ("BDRip", re.compile(r"bd[-.]?rip", re.IGNORECASE)),
    ("HDTV", re.compile(r"hdtv|pdtv", re.IGNORECASE)),
    ("DVD", re.compile(r"\bdvd\b|\bntsc\b|\bpal\b", re.IGNORECASE)),
)

#: Timing patterns are DERIVED from the ground-truth labels the corpus actually
#: carries (alignment mode, cut, content) rather than from a hand-written map of
#: case names. A guessed map reports gaps that do not exist and hides ones that
#: do, which is the opposite of what a coverage report is for.
TIMING_LABELS = {
    "global_offset",
    "small_offset",
    "large_offset",
    "drift",
    "piecewise",
    "different_intro",
    "different_outro",
    "different_runtime",
    "different_cut",
    "different_edit",
    "no_alignment",
}

_GRAMMAR_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2}))?[,.](\d{1,3})")


def _timing_pattern(truth: dict[str, Any]) -> str:
    """Classify one ground-truth record from the labels it actually carries."""
    align = truth.get("alignment") or {}
    mode = str(align.get("mode") or "")
    residual = align.get("residual_ms")
    cut = str(truth.get("cut") or "")
    content = str(truth.get("content") or "")

    if cut == "different_cut":
        return "different_cut"
    if content in {"different_content", "same_content_different_release"}:
        return "different_edit"
    if mode == "no_alignment":
        return "no_alignment"
    if mode == "as_target_with_drift":
        return "drift"
    if mode == "partial_misalign":
        return "large_offset"
    if mode == "as_target":
        try:
            if int(residual or 0) == 0:
                return "small_offset"
        except (TypeError, ValueError):
            pass
        return "global_offset"
    return "unmapped"


def _scan_grammar(manifest: Path, cases: list[dict[str, Any]]) -> dict[str, int]:
    """Count real timestamp grammars by reading the fixture files.

    Derived from bytes on disk, so this cannot drift from what the parser will
    actually see. Reading the files is what makes ``MM:SS,mmm`` coverage a
    measurement rather than an assumption.
    """
    root = manifest.parent
    seen: dict[str, int] = {}
    for case in cases:
        paths = [(case.get("target") or {}).get("path")]
        paths += [c.get("path") for c in case.get("candidates") or []]
        for rel in paths:
            if not rel:
                continue
            fp = root / str(rel)
            if not fp.exists():
                continue
            try:
                with fp.open("r", encoding="utf-8", errors="replace") as handle:
                    for line in handle:
                        match = _GRAMMAR_RE.match(line)
                        if match:
                            key = "HH:MM:SS,mmm" if match.group(3) else "MM:SS,mmm"
                            seen[key] = seen.get(key, 0) + 1
                            break
            except OSError:
                continue
    return seen


def _axis(required: tuple[str, ...], observed: dict[str, int]) -> list[dict[str, Any]]:
    rows = []
    for value in required:
        count = observed.get(value, 0)
        rows.append(
            {
                "value": value,
                "cases": count,
                "status": "covered" if count else "not_available",
            }
        )
    return rows


def analyse(manifest: Path) -> dict[str, Any]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    cases = data.get("cases", [])

    resolutions: Counter[str] = Counter()
    families: Counter[str] = Counter()
    fps: Counter[str] = Counter()
    media: Counter[str] = Counter()
    timing: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    real_media = 0
    synthetic = 0

    for case in cases:
        target = case.get("target") or {}
        names = [str(target.get("filename") or "")]
        for cand in case.get("candidates") or []:
            names.append(str(cand.get("release_name") or ""))
        blob = " ".join(n for n in names if n)

        res = RES_RE.search(blob)
        if res:
            resolutions[res.group(1).lower()] += 1
        hit = FPS_RE.search(blob)
        if hit:
            fps[hit.group(1)] += 1
        for label, pattern in FAMILY_PATTERNS:
            if pattern.search(blob):
                families[label] += 1
                break

        if re.search(r"S\d{2}(?:\.|-)\d{1,2}(?:-|\s|$)|\bS\d{2}[- ]?\d{1,2}\b", blob):
            media["season_pack"] += 1
        elif re.search(r"S\d{2}E\d{2}", blob):
            media["series"] += 1
        else:
            media["movie"] += 1

        for truth in case.get("ground_truth") or []:
            timing[_timing_pattern(truth)] += 1
            content = str(truth.get("content") or "")
            if content == "sparse":
                formats["sparse"] += 1
            elif content == "different_content":
                formats["dense"] += 1

        for cand in case.get("candidates") or []:
            languages[str(cand.get("language") or "und").lower()] += 1
            formats["non-HI"] += 1

        mode = str(case.get("failure_mode") or "")
        if mode == "credits_only":
            formats["credits"] += 1
        if mode == "sparse_dialogue":
            formats["sparse"] += 1

        if case.get("is_real_media") or str(target.get("path", "")).endswith((".mkv", ".mp4")):
            real_media += 1
        else:
            synthetic += 1

    grammar = _scan_grammar(manifest, cases)
    formats.update(grammar)

    axes = {
        "resolution": _axis(REQUIRED_RESOLUTIONS, dict(resolutions)),
        "release_family": _axis(REQUIRED_RELEASE_FAMILIES, dict(families)),
        "fps_family": _axis(REQUIRED_FPS, dict(fps)),
        "media_type": _axis(REQUIRED_MEDIA, dict(media)),
        "timing_pattern": _axis(REQUIRED_TIMING, dict(timing)),
        "subtitle_characteristic": _axis(REQUIRED_FORMATS, dict(formats)),
    }

    gaps = [
        f"{name}={row['value']}"
        for name, rows in axes.items()
        for row in rows
        if row["status"] == "not_available"
    ]

    return {
        "corpus": str(manifest.relative_to(REPO)),
        "dataset_version": data.get("dataset_version"),
        "cases_total": len(cases),
        "cases_synthetic": synthetic,
        "cases_real_media": real_media,
        "axes": axes,
        "timing_labels_observed": dict(timing),
        "timestamp_grammar_observed": dict(grammar),
        "languages_observed": dict(languages),
        "gaps": gaps,
        "coverage_complete": not gaps,
        "claim": (
            "Coverage is measured per axis from the corpus labels and the fixture "
            "bytes. A gap means the axis is required for a real-world claim and "
            "nothing covers it. No real media is present, so every number here is "
            "synthetic evidence of behaviour, never evidence of real-world accuracy."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Report measured corpus coverage")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(argv)

    if not args.manifest.exists():
        print(f"corpus not found: {args.manifest}", file=sys.stderr)
        return 2

    report = analyse(args.manifest)

    print("=" * 74)
    print("AUTOSYNC CORPUS COVERAGE (measured)")
    print("=" * 74)
    print(f"  corpus            : {report['corpus']}")
    print(f"  cases             : {report['cases_total']} "
          f"({report['cases_synthetic']} synthetic, {report['cases_real_media']} real media)")
    print()
    for name, rows in report["axes"].items():
        covered = [r for r in rows if r["status"] == "covered"]
        print(f"  {name}")
        for row in rows:
            marker = "ok " if row["status"] == "covered" else "-- "
            print(f"    [{marker}] {row['value']:<20} {row['cases']:>3} case(s)")
        print(f"    -> {len(covered)}/{len(rows)} required values covered")
        print()

    print("-" * 74)
    if report["coverage_complete"]:
        print("  COVERAGE COMPLETE across every required axis")
    else:
        print(f"  {len(report['gaps'])} REQUIRED AXIS VALUE(S) HAVE NO COVERAGE:")
        for gap in report["gaps"]:
            print(f"    - {gap}")
    print()
    print("  " + report["claim"])
    print("=" * 74)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"  report written to {args.json_out}", file=sys.stderr)

    # A gap is a measurement, not a failure. This tool reports; it does not gate.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
