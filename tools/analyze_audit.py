"""Offline analysis of exported sync decision records.

Reads the JSONL produced by ``app.services.sync.audit`` and reports the numbers
needed to decide whether a threshold, predictor rule, or cut-detection change is
warranted. Standard library only, no service calls, no network.

Usage:

    python tools/analyze_audit.py path/to/audit.jsonl
    python tools/analyze_audit.py audit.jsonl --min-observations 50

Note on interpretation: prediction outcomes here are *counts*, not calibrated
probabilities. Nothing here is called a probability unless the observation
count is large enough to be meaningful, which is what ``--min-observations``
controls. The headline number to protect is ``false_verified``: a candidate the
decision layer called strongly synchronized that later failed verification.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

VERIFIED_STATES = {"verified_synced", "verified_resynced"}


def load(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"warn: skipping malformed line {line_no}: {exc}", file=sys.stderr)
    return records


def _rate(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "n/a"
    return f"{numerator / denominator:.1%} ({numerator}/{denominator})"


def analyze(records: list[dict[str, Any]], min_observations: int = 20) -> dict[str, Any]:
    serve = [r for r in records if r.get("phase") == "serve"]
    search = [r for r in records if r.get("phase") == "search"]

    state_counts = Counter(r.get("sync_state") or "unknown" for r in records)
    serve_state_counts = Counter(r.get("sync_state") or "unknown" for r in serve)

    # Prediction -> verification, only for serve records that resolved a prior
    # search-time prediction.
    confusion: dict[str, Counter] = defaultdict(Counter)
    by_rule: dict[str, Counter] = defaultdict(Counter)
    for record in serve:
        predicted = record.get("prediction_state")
        if not predicted:
            continue
        actual = record.get("sync_state") or "unknown"
        confusion[predicted][actual] += 1
        by_rule[record.get("prediction_rule_at_search") or "unknown"][actual] += 1

    # Prediction outcomes. A PREDICTED hint that verification then confirms is
    # a SUCCESS, not a false positive, so these are counted as distinct
    # outcomes rather than collapsed into one misleading number.
    predicted_total = sum(sum(counter.values()) for counter in confusion.values())
    predicted_confirmed = sum(
        sum(counter.get(state, 0) for state in VERIFIED_STATES)
        for counter in confusion.values()
    )
    predicted_unconfirmed = sum(
        counter.get("unverified", 0) + counter.get("rejected", 0)
        for counter in confusion.values()
    )

    # The metric that actually protects users: a VERIFIED_* state recorded
    # without measured backing. A prediction can never produce this (it has no
    # code path to a verified state), so any non-zero value is a real bug.
    false_verified = sum(
        1
        for record in records
        if record.get("sync_state") in VERIFIED_STATES
        and record.get("verification") not in ("verified", "cached")
    )

    # A false rejection: a candidate the decision layer called rejected that
    # verification later accepted. Tracked separately and never used to justify
    # loosening a gate.
    false_rejected = confusion["rejected"].get("verified_synced", 0) + confusion[
        "rejected"
    ].get("verified_resynced", 0)

    cache_hits = sum(1 for r in records if r.get("from_cache"))
    alass_runs = sum(1 for r in records if r.get("alass_applied"))
    no_fingerprint = sum(1 for r in records if not r.get("has_video_fingerprint"))
    contexts = Counter(r.get("request_context") or "unknown" for r in records)

    # Performance by release/source class, using only fields already present.
    by_source: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        by_source[record.get("release_source") or "unknown"][record.get("sync_state") or "unknown"] += 1

    return {
        "records": len(records),
        "search_records": len(search),
        "serve_records": len(serve),
        "state_counts": state_counts,
        "serve_state_counts": serve_state_counts,
        "confusion": confusion,
        "by_rule": by_rule,
        "predicted_total": predicted_total,
        "predicted_confirmed": predicted_confirmed,
        "predicted_unconfirmed": predicted_unconfirmed,
        "false_verified": false_verified,
        "false_rejected": false_rejected,
        "cache_hits": cache_hits,
        "alass_runs": alass_runs,
        "no_fingerprint": no_fingerprint,
        "contexts": contexts,
        "by_source": by_source,
        "min_observations": min_observations,
    }


def report(stats: dict[str, Any]) -> str:
    out: list[str] = []
    add = out.append

    add("=" * 72)
    add("SYNC DECISION AUDIT")
    add("=" * 72)
    add(f"records: {stats['records']}  (search={stats['search_records']} serve={stats['serve_records']})")
    add("")

    add("Decision states (all records)")
    for state, count in stats["state_counts"].most_common():
        add(f"  {state:20s} {count:6d}  {_rate(count, stats['records'])}")
    add("")

    add("Measured states (serve only)")
    for state, count in stats["serve_state_counts"].most_common():
        add(f"  {state:20s} {count:6d}  {_rate(count, stats['serve_records'])}")
    add("")

    add("Prediction -> verification")
    if stats["predicted_total"] == 0:
        add("  no paired observations yet")
    else:
        for predicted, counter in stats["confusion"].items():
            total = sum(counter.values())
            reliability = _rate(counter.get("verified_synced", 0), total)
            add(f"  {predicted} -> {total} observations, confirmed verified {reliability}")
            for actual, count in counter.most_common():
                add(f"      {actual:20s} {count:6d}  {_rate(count, total)}")
    add("")

    add("HEADLINE METRICS")
    add(
        f"  false_verified (unsupported VERIFIED_* claim): {stats['false_verified']}"
    )
    add(f"  false_rejected                             : {stats['false_rejected']}")
    add(f"  predictions confirmed                      : {_rate(stats['predicted_confirmed'], stats['predicted_total'])}")
    add(f"  predictions unconfirmed                    : {_rate(stats['predicted_unconfirmed'], stats['predicted_total'])}")
    add(
        "  note: a PREDICTED hint that verification confirms is a success, not a"
        " false positive;"
    )
    add("        only a VERIFIED_* state without measured backing counts as false_verified.")
    if stats["predicted_total"] < stats["min_observations"]:
        add(
            f"  NOTE: {stats['predicted_total']} observations is below "
            f"--min-observations={stats['min_observations']}; do not tune thresholds yet."
        )
    add("")

    add("Predictor rules (vs actual verification)")
    for rule, counter in stats["by_rule"].items():
        total = sum(counter.values())
        confirmed = counter.get("verified_synced", 0) + counter.get("verified_resynced", 0)
        add(f"  {rule:24s} n={total:5d}  confirmed={_rate(confirmed, total)}")
    add("")

    add("Operational")
    add(f"  cache-sourced records       : {_rate(stats['cache_hits'], stats['records'])}")
    add(f"  alass executions (records)  : {stats['alass_runs']}")
    add(f"  missing fingerprint         : {_rate(stats['no_fingerprint'], stats['records'])}")
    add("  request context:")
    for context, count in stats["contexts"].most_common():
        add(f"      {context:24s} {count:6d}")
    add("")

    add("By release source class")
    for source, counter in sorted(
        stats["by_source"].items(), key=lambda kv: -sum(kv[1].values())
    ):
        total = sum(counter.values())
        confirmed = counter.get("verified_synced", 0) + counter.get("verified_resynced", 0)
        rejected = counter.get("rejected", 0)
        add(f"  {source:20s} n={total:5d} confirmed={_rate(confirmed, total)} rejected={rejected}")
    add("")
    add("=" * 72)
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze exported sync decision records.")
    parser.add_argument("path", type=Path, help="JSONL file written by the audit log")
    parser.add_argument(
        "--min-observations",
        type=int,
        default=20,
        help="Below this many paired predictions, warn against tuning (default 20)",
    )
    args = parser.parse_args(argv)

    if not args.path.is_file():
        print(f"error: {args.path} not found", file=sys.stderr)
        return 2

    records = load(args.path)
    if not records:
        print("error: no records found", file=sys.stderr)
        return 1

    print(report(analyze(records, args.min_observations)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
