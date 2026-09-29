"""Offline calibration and rule analysis for the sync predictor.

Turns exported audit records into evidence for the *next* engineering decision.
This module only reads a JSONL file and prints; it never touches the running
service, never writes code, and never proposes a threshold value. The output
ends in INVESTIGATION CANDIDATES, not recommended edits.

Statistical care taken here:

* **Minimum samples.** Any group below the configured minimum is reported as
  ``INSUFFICIENT_SAMPLE`` rather than as a rate, so a 2-observation "100%"
  cannot read as reliable.
* **Wilson intervals.** Confirmation rates carry a 95% interval computed with
  Wilson score, not a normal approximation that misbehaves near 0 and 1. The
  interval describes the observed sample; it is not a future probability.
* **Outcome separation.** ``verified_synced`` and ``verified_resynced`` stay
  distinct, because a rule that mostly yields resyncs may still be useful.
* **Concentration warnings.** A rule that looks poor for BluRay might reflect
  one provider or one title. Concentration is reported next to the rate so the
  rate is not read as a property of the class.
* **PREDICTED != VERIFIED.** Only a serve-phase record with a measured
  verification populates the verified outcome buckets. A prediction, a cache
  hit, a successful alass exit, or MatchTier=HASH never counts as verified.

Standard library only. No network, no service, no configuration writes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Outcome taxonomy. Kept separate on purpose: collapsing resync into sync would
# hide rules that are useful but only ever re-time.
OUTCOME_SYNCED = "verified_synced"
OUTCOME_RESYNCED = "verified_resynced"
OUTCOME_UNVERIFIED = "unverified"
OUTCOME_REJECTED = "rejected"
OUTCOME_UNKNOWN = "unknown"
OUTCOMES = (
    OUTCOME_SYNCED,
    OUTCOME_RESYNCED,
    OUTCOME_UNVERIFIED,
    OUTCOME_REJECTED,
    OUTCOME_UNKNOWN,
)
VERIFIED_OUTCOMES = (OUTCOME_SYNCED, OUTCOME_RESYNCED)
NEGATIVE_OUTCOMES = (OUTCOME_REJECTED,)

# States that may legitimately carry a verified outcome. Anything else paired
# with a verified state is a data-quality problem, not a result.
VERIFICATION_FOR_VERIFIED = {"verified", "cached"}

# Source classes the system already normalizes. No new parser is introduced.
SOURCE_CLASSES = ("WEB-DL", "WEBRip", "BluRay", "HDTV", "unknown")

DEFAULT_MIN_RULE_OBSERVATIONS = 30
DEFAULT_MIN_GROUP_OBSERVATIONS = 20
DEFAULT_Z = 1.96  # 95% two-sided


# --------------------------------------------------------------------------- #
# Loading and data-quality validation
# --------------------------------------------------------------------------- #


@dataclass
class DataQuality:
    """Problems found before any metric is computed. Never silently dropped."""

    total_lines: int = 0
    malformed: list[int] = field(default_factory=list)
    duplicates: int = 0
    missing_engine_version: int = 0
    missing_rule: int = 0
    predictions_without_outcome: int = 0
    impossible_state_verification: int = 0
    reused_pair_key: int = 0
    unknown_schema_version: int = 0
    records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.malformed
            or self.duplicates
            or self.impossible_state_verification
            or self.unknown_schema_version
        )

    def summary(self) -> list[str]:
        out = []
        if self.malformed:
            out.append(f"malformed JSONL lines: {len(self.malformed)} (kept, not analysed)")
        if self.duplicates:
            out.append(f"duplicate decision records: {self.duplicates} (deduplicated)")
        if self.missing_engine_version:
            out.append(f"records missing engine_version: {self.missing_engine_version}")
        if self.unknown_schema_version:
            out.append(f"records with unknown schema_version: {self.unknown_schema_version}")
        if self.missing_rule:
            out.append(f"paired predictions missing a rule name: {self.missing_rule}")
        if self.predictions_without_outcome:
            out.append(
                f"predictions with no later outcome: {self.predictions_without_outcome} "
                "(not counted as failures)"
            )
        if self.impossible_state_verification:
            out.append(
                f"verified state without measured verification: {self.impossible_state_verification} "
                "- this is a BUG indicator, not a result"
            )
        if self.reused_pair_key:
            out.append(f"consumed pair keys seen more than once: {self.reused_pair_key}")
        return out or ["no data-quality problems detected"]


def _fingerprint(record: dict[str, Any]) -> str:
    """Identity of a record for duplicate detection (content, not identity)."""
    return json.dumps(record, sort_keys=True, ensure_ascii=False)


def load_records(path: Path) -> DataQuality:
    quality = DataQuality()
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            quality.total_lines += 1
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                quality.malformed.append(line_no)
                continue
            if not isinstance(record, dict):
                quality.malformed.append(line_no)
                continue
            digest = _fingerprint(record)
            if digest in seen:
                quality.duplicates += 1
                continue
            seen.add(digest)
            if not record.get("engine_version"):
                quality.missing_engine_version += 1
            if record.get("schema_version") != 1:
                quality.unknown_schema_version += 1
            state = record.get("sync_state")
            if state in (OUTCOME_SYNCED, OUTCOME_RESYNCED) and record.get(
                "verification"
            ) not in VERIFICATION_FOR_VERIFIED:
                quality.impossible_state_verification += 1
            quality.records.append(record)
    return quality


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def wilson_interval(successes: int, total: int, z: float = DEFAULT_Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation, which produces nonsensical bounds
    near 0 and 1 - exactly the regime small audit samples live in.
    """
    if total <= 0:
        return (0.0, 1.0)
    phat = successes / total
    denominator = 1.0 + (z * z) / total
    centre = (phat + (z * z) / (2 * total)) / denominator
    margin = (z / denominator) * math.sqrt(
        (phat * (1 - phat) / total) + (z * z) / (4 * total * total)
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def normalize_source(value: Any) -> str:
    """Map an already-normalized source string onto a known class."""
    text = str(value or "").strip().lower()
    if not text:
        return "unknown"
    for candidate in ("web-dl", "webdl", "web dl"):
        if text == candidate:
            return "WEB-DL"
    for candidate in ("webrip", "web-rip", "web rip"):
        if text == candidate:
            return "WEBRip"
    for candidate in ("bluray", "blu-ray", "remux", "bdrip", "brrip"):
        if text == candidate:
            return "BluRay"
    for candidate in ("hdtv", "pdtv", "dvdrip", "dvd", "cam", "telesync"):
        if text == candidate:
            return "HDTV"
    return "unknown"


# --------------------------------------------------------------------------- #
# Observations: only measured serve outcomes count
# --------------------------------------------------------------------------- #


@dataclass
class Observation:
    """One prediction resolved by one later measurement."""

    rule: str
    confidence: float | None
    predicted_state: str
    outcome: str
    source: str
    provider: str | None
    match_tier: str | None
    resolution: str | None
    edition: str | None
    fps_relation: str | None
    media_type: str | None
    video_id: str | None
    from_cache: bool
    timestamp: str | None

    @property
    def verified_any(self) -> bool:
        return self.outcome in VERIFIED_OUTCOMES


def build_observations(records: list[dict[str, Any]]) -> list[Observation]:
    """Extract prediction -> measured-outcome pairs.

    Only serve-phase records carrying a ``prediction_state`` qualify, so a
    prediction, a cache entry, or a successful alass exit can never populate a
    verified bucket on its own.
    """
    observations: list[Observation] = []
    for record in records:
        if record.get("phase") != "serve":
            continue
        predicted = record.get("prediction_state")
        if not predicted:
            continue
        outcome = record.get("sync_state") or OUTCOME_UNKNOWN
        if outcome not in OUTCOMES:
            outcome = OUTCOME_UNKNOWN
        confidence = record.get("prediction_confidence")
        observations.append(
            Observation(
                rule=record.get("prediction_rule_at_search") or "missing_rule",
                confidence=(
                    float(confidence) if isinstance(confidence, int | float) else None
                ),
                predicted_state=str(predicted),
                outcome=outcome,
                source=normalize_source(record.get("release_source")),
                provider=record.get("provider"),
                match_tier=record.get("match_tier"),
                resolution=record.get("release_resolution"),
                edition=record.get("release_edition"),
                fps_relation=record.get("fps_relation"),
                media_type=record.get("media_type"),
                video_id=record.get("video_id"),
                from_cache=bool(record.get("from_cache")),
                timestamp=record.get("timestamp"),
            )
        )
    return observations


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


@dataclass
class GroupStats:
    """Outcome distribution for one group, with an explicit sufficiency label."""

    label: str
    total: int = 0
    counts: Counter = field(default_factory=Counter)
    concentration: float | None = None

    def add(self, observation: Observation) -> None:
        self.total += 1
        self.counts[observation.outcome] += 1

    def count(self, outcome: str) -> int:
        return self.counts.get(outcome, 0)

    @property
    def verified_any(self) -> int:
        return sum(self.count(o) for o in VERIFIED_OUTCOMES)

    @property
    def confirmed_rate(self) -> float | None:
        return self.verified_any / self.total if self.total else None

    def interval(self) -> tuple[float, float] | None:
        if not self.total:
            return None
        return wilson_interval(self.verified_any, self.total)

    def sufficient(self, minimum: int) -> bool:
        return self.total >= minimum

    def label_with(self, minimum: int) -> str:
        return "ok" if self.sufficient(minimum) else "INSUFFICIENT_SAMPLE"

    def to_dict(self, minimum: int) -> dict[str, Any]:
        return {
            "label": self.label,
            "total": self.total,
            "sufficient": self.sufficient(minimum),
            "verified_synced": self.count(OUTCOME_SYNCED),
            "verified_resynced": self.count(OUTCOME_RESYNCED),
            "verified_any": self.verified_any,
            "unverified": self.count(OUTCOME_UNVERIFIED),
            "rejected": self.count(OUTCOME_REJECTED),
            "unknown": self.count(OUTCOME_UNKNOWN),
            "confirmed_rate": self.confirmed_rate,
            "wilson_95": self.interval(),
        }


def _group(observations: list[Observation], key) -> dict[str, GroupStats]:
    groups: dict[str, GroupStats] = {}
    for observation in observations:
        raw = key(observation)
        label = str(raw) if raw not in (None, "") else "unknown"
        groups.setdefault(label, GroupStats(label=label)).add(observation)
    return groups


def _concentration(observations: list[Observation], key) -> float | None:
    """Fraction of observations in the most common bucket of ``key``."""
    if not observations:
        return None
    counts = Counter(str(key(o)) for o in observations)
    return max(counts.values()) / len(observations)


# --------------------------------------------------------------------------- #
# Rule-split detection
# --------------------------------------------------------------------------- #


def _chi_square_2xk(groups: list[GroupStats]) -> float:
    """Pearson chi-square for a 2xK contingency table (verified vs not)."""
    if len(groups) < 2:
        return 0.0
    table = [[g.verified_any, g.total - g.verified_any] for g in groups]
    row_totals = [sum(row) for row in table]
    col_totals = [sum(table[i][j] for i in range(len(table))) for j in (0, 1)]
    grand = sum(row_totals)
    if grand <= 0 or any(t == 0 for t in col_totals):
        return 0.0
    expected = [
        [row_totals[i] * col_totals[j] / grand for j in (0, 1)] for i in range(len(table))
    ]
    chi = 0.0
    for i, row in enumerate(table):
        for j, cell in enumerate(row):
            if expected[i][j] > 0:
                chi += (cell - expected[i][j]) ** 2 / expected[i][j]
    return chi


def find_split_candidates(
    observations: list[Observation],
    *,
    min_group: int,
    min_total: int,
) -> list[dict[str, Any]]:
    """Rules whose outcome distribution differs materially across source classes.

    A suggestion for investigation only. Nothing here changes the predictor.
    """
    by_rule: dict[str, list[Observation]] = defaultdict(list)
    for observation in observations:
        by_rule[observation.rule].append(observation)

    candidates: list[dict[str, Any]] = []
    for rule, rule_observations in by_rule.items():
        if len(rule_observations) < min_total:
            continue
        groups = sorted(
            _group(rule_observations, lambda o: o.source).values(),
            key=lambda g: g.total,
            reverse=True,
        )
        usable = [g for g in groups if g.sufficient(min_group)]
        if len(usable) < 2:
            continue
        rates = [g.confirmed_rate or 0.0 for g in usable]
        spread = max(rates) - min(rates)
        if spread < 0.25:
            continue
        chi = _chi_square_2xk(usable)
        concentration = max(
            (g.count(OUTCOME_SYNCED) + g.count(OUTCOME_RESYNCED) + g.count(OUTCOME_REJECTED))
            for g in usable
        )
        candidates.append(
            {
                "rule": rule,
                "chi_square": round(chi, 2),
                "rate_spread": round(spread, 3),
                "concentration_warning": concentration >= 0.5,
                "classes": {
                    g.label: {
                        "n": g.total,
                        "confirmed_rate": g.confirmed_rate,
                        "verified_any": g.verified_any,
                        "rejected": g.count(OUTCOME_REJECTED),
                    }
                    for g in usable
                },
            }
        )
    candidates.sort(key=lambda c: c["rate_spread"], reverse=True)
    return candidates


# --------------------------------------------------------------------------- #
# Full analysis
# --------------------------------------------------------------------------- #


def analyze(quality: DataQuality, *, min_rule: int, min_group: int) -> dict[str, Any]:
    records = quality.records
    observations = build_observations(records)
    serve = [r for r in records if r.get("phase") == "serve"]
    search = [r for r in records if r.get("phase") == "search"]

    rule_stats = _group(observations, lambda o: o.rule)
    by_source = _group(observations, lambda o: o.source)
    by_provider = _group(observations, lambda o: o.provider or "unknown")
    by_tier = _group(observations, lambda o: o.match_tier or "unknown")
    by_confidence = _group(observations, lambda o: o.confidence)
    by_media = _group(observations, lambda o: o.media_type or "unknown")
    by_fps = _group(observations, lambda o: o.fps_relation or "unknown")
    by_resolution = _group(observations, lambda o: o.resolution or "unknown")
    by_edition = _group(observations, lambda o: o.edition or "unknown")

    # Rule x MatchTier cross-tab, to expose redundancy between the matcher and
    # the predictor.
    cross: dict[str, Counter] = defaultdict(Counter)
    for observation in observations:
        cross[f"{observation.match_tier or 'unknown'} + {observation.rule}"][
            observation.outcome
        ] += 1

    # Confidence ordering: does 95 really beat 90 really beat 75?
    confidence_order = []
    for label, stats in sorted(
        by_confidence.items(), key=lambda kv: -float(kv[0] or 0)
    ):
        confidence_order.append({"confidence": float(label or 0), **stats.to_dict(min_rule)})

    # Fingerprint impact, using only the explicit recorded context.
    with_fp = [r for r in records if r.get("has_video_fingerprint")]
    without_fp = [r for r in records if not r.get("has_video_fingerprint")]
    contexts = Counter(str(r.get("request_context") or "unknown") for r in records)

    # Cache warming and repeated-video reuse.
    videos_with_multiple = sum(1 for _, n in Counter(o.video_id for o in observations if o.video_id).items() if n > 1)
    cache_sourced = sum(1 for o in observations if o.from_cache)
    alass_runs = sum(1 for r in records if r.get("alass_applied"))

    # Ranking impact is recorded by the service; the tool only reports it.
    ranking_changed = sum(
        1 for r in records if r.get("phase") == "search" and r.get("ranking_reordered")
    )

    # Concentration across providers, titles/videos, and languages.
    concentration = {
        "by_source": {
            label: round(
                _concentration(
                    [o for o in observations if o.source == label],
                    lambda o: o.provider or "unknown",
                )
                or 0.0,
                3,
            )
            for label in by_source
        },
        "top_provider": _concentration(observations, lambda o: o.provider or "unknown"),
        "top_video": _concentration(observations, lambda o: o.video_id or "unknown"),
        "top_language": _concentration(records, lambda r: r.get("language") or "unknown"),
        "videos_with_repeated_observations": videos_with_multiple,
    }

    # Predictions that never received a measured outcome. Counted by pair key so
    # it cannot go negative, and reported separately from algorithm metrics:
    # an unresolved prediction is a coverage gap, never a failure.
    consumed = {
        (r.get("video_id"), r.get("subtitle_id"))
        for r in records
        if r.get("phase") == "serve" and r.get("prediction_state")
    }
    predicted_search = [
        r
        for r in search
        if r.get("verification") == "predicted"
    ]
    unresolved = sum(
        1 for r in predicted_search if (r.get("video_id"), r.get("subtitle_id")) not in consumed
    )

    return {
        "records_total": len(records),
        "records_search": len(search),
        "records_serve": len(serve),
        "observations": len(observations),
        "predictions_total": len(predicted_search),
        "predictions_without_outcome": unresolved,
        "min_rule": min_rule,
        "min_group": min_group,
        "data_quality": quality.summary(),
        "data_quality_ok": quality.ok,
        "rules": {k: v.to_dict(min_rule) for k, v in sorted(rule_stats.items())},
        "by_source": {k: v.to_dict(min_group) for k, v in sorted(by_source.items())},
        "by_provider": {k: v.to_dict(min_group) for k, v in sorted(by_provider.items())},
        "by_match_tier": {k: v.to_dict(min_group) for k, v in sorted(by_tier.items())},
        "by_media_type": {k: v.to_dict(min_group) for k, v in sorted(by_media.items())},
        "by_fps_relation": {k: v.to_dict(min_group) for k, v in sorted(by_fps.items())},
        "by_resolution": {k: v.to_dict(min_group) for k, v in sorted(by_resolution.items())},
        "by_edition": {k: v.to_dict(min_group) for k, v in sorted(by_edition.items())},
        "confidence_calibration": confidence_order,
        "rule_x_tier": {k: dict(v) for k, v in sorted(cross.items())},
        "split_candidates": find_split_candidates(
            observations, min_group=min_group, min_total=min_rule
        ),
        "fingerprint": {
            "with_fingerprint": len(with_fp),
            "without_fingerprint": len(without_fp),
            "prediction_coverage_with": round(
                sum(1 for o in observations if o.video_id) / len(with_fp), 4
            )
            if with_fp
            else None,
            "prediction_coverage_without": 0.0,
            "contexts": dict(contexts),
        },
        "cache": {
            "cache_sourced_observations": cache_sourced,
            "alass_runs": alass_runs,
            "videos_with_repeated_observations": videos_with_multiple,
            "cache_reuse_rate": round(cache_sourced / len(observations), 4)
            if observations
            else 0.0,
        },
        "ranking": {"search_records_with_reorder_flag": ranking_changed},
        "concentration": concentration,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _ci(stats: dict[str, Any]) -> str:
    interval = stats.get("wilson_95")
    if not interval or not stats.get("total"):
        return "n/a"
    return f"[{interval[0]:.2f}, {interval[1]:.2f}]"


def _outcome_row(label: str, stats: dict[str, Any], minimum: int) -> list[str]:
    flag = "" if stats.get("sufficient") else "  INSUFFICIENT_SAMPLE"
    rate = _pct(stats.get("confirmed_rate"))
    ci = _ci(stats) if stats.get("sufficient") else "n/a"
    return [
        f"  {label:16s} n={stats['total']:<5d} "
        f"synced={stats['verified_synced']:<4d} resynced={stats['verified_resynced']:<4d} "
        f"unver={stats['unverified']:<4d} rej={stats['rejected']:<4d} "
        f"unk={stats['unknown']:<4d} confirmed={rate:>6s} 95%CI={ci}{flag}"
    ]


def render_report(a: dict[str, Any]) -> str:
    out: list[str] = []
    add = out.append
    rule = "=" * 74
    sub = "-" * 74

    add(rule)
    add("SYNC PREDICTOR CALIBRATION REPORT")
    add(rule)
    add("Dataset")
    add(sub)
    add(f"  records                : {a['records_total']} "
        f"(search={a['records_search']} serve={a['records_serve']})")
    add(f"  paired observations    : {a['observations']}")
    add(f"  predictions recorded   : {a['predictions_total']}")
    add(f"  predictions w/o outcome: {a['predictions_without_outcome']} "
        "(coverage gap, not a failure)")
    add(f"  min rule observations  : {a['min_rule']}")
    add(f"  min group observations : {a['min_group']}")
    add("")

    add("Data quality")
    add(sub)
    for line in a["data_quality"]:
        add(f"  {line}")
    if not a["data_quality_ok"]:
        add("  ** metrics below may be affected; treat with caution **")
    add("")

    add("Predictor rules")
    add(sub)
    if not a["rules"]:
        add("  no paired observations yet")
    for stats in a["rules"].values():
        add(_outcome_row(stats["label"], stats, a["min_rule"])[0])
    add("")

    add("Confidence calibration  (is 95 > 90 > 75 actually supported?)")
    add(sub)
    ordered = [c for c in a["confidence_calibration"] if c["total"]]
    if not ordered:
        add("  no observations")
    for entry in ordered:
        add(_outcome_row(f"conf={entry['confidence']:.0f}", entry, a["min_rule"])[0])
    monotone = _confidence_is_monotone(ordered)
    if monotone is True:
        add("  observed: confidence ordering is supported by these samples")
    elif monotone is False:
        add("  observed: confidence ordering is NOT supported (a lower confidence")
        add("           bucket confirmed at least as often as a higher one)")
    else:
        add("  observed: insufficient sample to assess ordering")
    add("")

    add("Release class analysis")
    add(sub)
    for section in ("by_source", "by_resolution", "by_edition", "by_fps_relation", "by_media_type"):
        title = section.replace("by_", "").replace("_", " ").title()
        add(f"  [{title}]")
        for stats in a[section].values():
            add(_outcome_row(stats["label"], stats, a["min_group"])[0])
    add("")

    add("Provider analysis  (observational only; not a provider ranking)")
    add(sub)
    for stats in a["by_provider"].values():
        add(_outcome_row(stats["label"], stats, a["min_group"])[0])
    add("")

    add("Predictor x MatchTier")
    add(sub)
    for key, counts in a["rule_x_tier"].items():
        total = sum(counts.values())
        add(f"  {key:44s} n={total}")
    add("")

    add("Candidate rule splits  (investigation only)")
    add(sub)
    if not a["split_candidates"]:
        add("  none detected")
    for candidate in a["split_candidates"]:
        add(f"  rule: {candidate['rule']}  spread={candidate['rate_spread']:.2f} "
            f"chi2={candidate['chi_square']}")
        for label, stats in sorted(
            candidate["classes"].items(), key=lambda kv: -(kv[1]["confirmed_rate"] or 0)
        ):
            add(f"      {label:14s} n={stats['n']:<5d} confirmed={_pct(stats['confirmed_rate']):>6s} "
                f"rejected={stats['rejected']}")
        if candidate["concentration_warning"]:
            add("      CONCENTRATION WARNING: an outcome dominates this rule's classes")
    add("")

    add("Concentration warnings")
    add(sub)
    concentration = a["concentration"]
    add(f"  top provider share : {_pct(concentration['top_provider'])}")
    add(f"  top video share    : {_pct(concentration['top_video'])}")
    add(f"  top language share : {_pct(concentration['top_language'])}")
    add(f"  videos with repeat observations: "
        f"{concentration['videos_with_repeated_observations']}")
    for source, share in sorted(concentration["by_source"].items()):
        if share >= 0.5:
            add(f"  {source}: {share:.0%} of its observations come from one provider")
    add("")

    add("Missing-fingerprint impact")
    add(sub)
    fingerprint = a["fingerprint"]
    add(f"  with fingerprint    : {fingerprint['with_fingerprint']}")
    add(f"  without fingerprint : {fingerprint['without_fingerprint']}")
    add(f"  request contexts    : {fingerprint['contexts']}")
    add("")

    add("Cache effectiveness")
    add(sub)
    cache = a["cache"]
    add(f"  cache-sourced observations : {cache['cache_sourced_observations']}")
    add(f"  cache reuse rate           : {_pct(cache['cache_reuse_rate'])}")
    add(f"  alass runs recorded        : {cache['alass_runs']}")
    add(f"  videos with repeats        : {cache['videos_with_repeated_observations']}")
    add("")

    add("Ranking impact  (observational; ordering is not adjusted from this)")
    add(sub)
    add(f"  search records flagged as reordered: "
        f"{a['ranking']['search_records_with_reorder_flag']}")
    add("")

    add("INVESTIGATION CANDIDATES")
    add(sub)
    for item in investigation_candidates(a):
        add(f"  {item}")
    add("")
    add("  No option is selected automatically. Each needs human review and, where")
    add("  a threshold changes, a documented before/after metric.")
    add(rule)
    return "\n".join(out)


def _confidence_is_monotone(ordered: list[dict[str, Any]]) -> bool | None:
    """Do higher confidence buckets confirm at least as often, given enough data?"""
    usable = [c for c in ordered if c.get("sufficient") and c.get("confirmed_rate") is not None]
    if len(usable) < 2:
        return None
    rates = [c["confirmed_rate"] for c in usable]
    for higher, lower in zip(rates, rates[1:], strict=False):
        if lower > higher + 1e-9:
            return False
    return True


def investigation_candidates(a: dict[str, Any]) -> list[str]:
    """Evidence for each possible next step. Never a decision.

    Each letter from the agreed option list is emitted at most once, so a
    reader is never left with two contradictory "E"s.
    """
    out: list[str] = []
    if not a["observations"]:
        return ["A. no paired observations yet; keep predictor unchanged and keep collecting"]

    used: set[str] = set()

    def emit(letter: str, text: str) -> None:
        if letter not in used:
            used.add(letter)
            out.append(f"{letter}. {text}")

    splits = a["split_candidates"]
    if splits:
        top = splits[0]
        emit(
            "B",
            f"split '{top['rule']}' by source class "
            f"(observed spread {top['rate_spread']:.2f}, chi2={top['chi_square']})",
        )
        if len(splits) > 1:
            emit(
                "C",
                "other rules also vary by class: "
                + ", ".join(c["rule"] for c in splits[1:]),
            )
    else:
        emit("A", "keep predictor unchanged: no rule shows a material class split")

    calibration = [c for c in a["confidence_calibration"] if c["total"]]
    if _confidence_is_monotone(calibration) is False:
        emit(
            "E",
            "revisit predictor confidence values: a lower confidence bucket confirmed "
            "at least as often as a higher one",
        )

    weak = [
        stats["label"]
        for stats in a["by_source"].values()
        if stats.get("sufficient") and (stats.get("confirmed_rate") or 0) < 0.5
    ]
    if weak:
        emit(
            "D",
            f"improve structural/cut verification: weak classes {', '.join(sorted(weak))}",
        )

    concentrated = [
        source
        for source, share in a["concentration"]["by_source"].items()
        if share >= 0.7 and (a["by_source"].get(source, {}).get("total") or 0) >= a["min_group"]
    ]
    if concentrated:
        emit(
            "F",
            f"investigate provider normalization: {', '.join(sorted(concentrated))} "
            "dominated by a single provider",
        )

    fingerprint = a["fingerprint"]
    if fingerprint["with_fingerprint"] == 0:
        emit("G", "investigate missing fingerprint: no observation had a video fingerprint")
    elif fingerprint["without_fingerprint"] > fingerprint["with_fingerprint"]:
        emit(
            "G",
            f"investigate missing fingerprint frequency: {fingerprint['without_fingerprint']} "
            f"records without vs {fingerprint['with_fingerprint']} with",
        )

    cache = a["cache"]
    if cache["cache_reuse_rate"] == 0.0 and a["observations"] > a["min_rule"]:
        emit("C", "cache reuse is zero despite enough observations; check alias writes")

    emit("H", "no change warranted if none of the above is supported by evidence")
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline calibration and rule analysis of sync audit records."
    )
    parser.add_argument("path", type=Path, help="JSONL written by the audit log")
    parser.add_argument("--min-observations", type=int, default=DEFAULT_MIN_RULE_OBSERVATIONS)
    parser.add_argument("--min-group", type=int, default=DEFAULT_MIN_GROUP_OBSERVATIONS)
    parser.add_argument("--json-out", type=Path, default=None, help="structured aggregate output")
    parser.add_argument(
        "--reference-report",
        action="store_true",
        help="report reference-trust integrity instead of predictor calibration",
    )
    parser.add_argument(
        "--shadow-report",
        action="store_true",
        help="compare the shadow (v2) reference selector against the current one",
    )
    parser.add_argument(
        "--shadow-replay",
        metavar="CASE_ID",
        help=(
            "offline reference replay for one recorded case; reports "
            "SIMULATION_UNAVAILABLE when the local artifacts are absent"
        ),
    )
    args = parser.parse_args(argv)

    if not args.path.is_file():
        print(f"error: {args.path} not found", file=sys.stderr)
        return 2

    quality = load_records(args.path)
    if not quality.records:
        print("error: no usable records", file=sys.stderr)
        for line in quality.summary():
            print(f"  {line}", file=sys.stderr)
        return 1

    def _emit(analysis: dict[str, Any], rendered: str) -> int:
        print(rendered)
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(
                json.dumps(analysis, indent=2, sort_keys=True, default=str), encoding="utf-8"
            )
            print(f"\nstructured output written to {args.json_out}", file=sys.stderr)
        return 0

    if args.shadow_replay:
        return _emit(
            *shadow_replay_analysis(quality.records, args.shadow_replay, args.path)
        )
    if args.shadow_report:
        analysis = shadow_comparison_analysis(quality.records)
        return _emit(analysis, render_shadow_report(analysis))
    if args.reference_report:
        analysis = reference_analysis(quality.records, min_observations=args.min_observations)
        return _emit(analysis, render_reference_report(analysis))

    analysis = analyze(quality, min_rule=args.min_observations, min_group=args.min_group)
    print(render_report(analysis))

    if args.json_out:
        # Aggregates only: no subtitle text, URLs, or credentials reach this file.
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(analysis, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        print(f"\nstructured output written to {args.json_out}", file=sys.stderr)
    return 0




# --------------------------------------------------------------------------- #
# Reference-integrity report
# --------------------------------------------------------------------------- #


def reference_analysis(records: list[dict[str, Any]], min_observations: int) -> dict[str, Any]:
    """Aggregate the reference evidence behind each serve-time decision.

    Reference trust is a *separate* axis from the decision outcome: a verified
    decision resting on an unproven reference is exactly the correlated-error
    case worth surfacing.
    """
    serve = [r for r in records if r.get("phase") == "serve"]

    by_trust: dict[str, Counter] = defaultdict(Counter)
    trust_totals: Counter = Counter()
    conflicts = 0
    for record in serve:
        trust = record.get("reference_trust") or "unassessed"
        outcome = record.get("sync_state") or OUTCOME_UNKNOWN
        by_trust[trust][outcome] += 1
        trust_totals[trust] += 1
        if record.get("reference_failure"):
            conflicts += 1

    verified_by_trust = {
        trust: sum(
            by_trust[trust].get(state, 0) for state in VERIFIED_OUTCOMES
        )
        for trust in by_trust
    }

    clusters = Counter(
        record.get("reference_independent_sources")
        for record in serve
        if record.get("reference_independent_sources")
    )
    cached_references = sum(1 for r in serve if r.get("reference_from_cache"))
    consensus_values = [
        r["reference_consensus"]
        for r in serve
        if isinstance(r.get("reference_consensus"), int | float)
    ]

    # The correlated-error headline: verified outcomes built on a reference that
    # was never itself trusted.
    verified_on_unproven = sum(
        verified_by_trust.get(trust, 0)
        for trust in ("acceptable", "unknown", "unassessed")
    )
    return {
        "serve_records": len(serve),
        "min_observations": min_observations,
        "trust_distribution": dict(trust_totals),
        "outcomes_by_trust": {k: dict(v) for k, v in sorted(by_trust.items())},
        "verified_by_trust": verified_by_trust,
        "verified_on_unproven_reference": verified_on_unproven,
        "reference_failures": conflicts,
        "consensus_clusters": {str(k): v for k, v in sorted(clusters.items())},
        "consensus_mean": round(sum(consensus_values) / len(consensus_values), 3)
        if consensus_values
        else None,
        "cached_references": cached_references,
    }


def render_reference_report(a: dict[str, Any]) -> str:
    out: list[str] = []
    add = out.append
    rule = "=" * 74
    add(rule)
    add("REFERENCE INTEGRITY REPORT")
    add(rule)
    add(f"  serve records analysed : {a['serve_records']}")
    add(f"  min observations      : {a['min_observations']}")
    add("")

    add("Reference trust distribution")
    add("-" * 74)
    if not a["trust_distribution"]:
        add("  no reference evidence recorded yet")
    for trust, total in sorted(a["trust_distribution"].items()):
        outcomes = a["outcomes_by_trust"].get(trust, {})
        verified = a["verified_by_trust"].get(trust, 0)
        flag = "" if total >= a["min_observations"] else "  INSUFFICIENT_SAMPLE"
        add(f"  {trust:14s} n={total:<5d} verified_outcomes={verified:<5d} "
            f"unverified={outcomes.get('unverified', 0):<4d} "
            f"rejected={outcomes.get('rejected', 0):<4d}{flag}")
    add("")

    add("CORRELATED-ERROR WATCH")
    add("-" * 74)
    add(f"  verified outcomes resting on an unproven reference : "
        f"{a['verified_on_unproven_reference']}")
    add(f"  explicit reference failures recorded              : {a['reference_failures']}")
    add(f"  references taken from a verified cache            : {a['cached_references']}")
    if a["verified_on_unproven_reference"]:
        add("  -> investigate whether reference policy should weight trust, or whether")
        add("     the downstream gates are already sufficient to catch these cases.")
    else:
        add("  -> no verified outcome currently rests on an unproven reference.")
    add("")

    add("Consensus clusters")
    add("-" * 74)
    if a["consensus_clusters"]:
        for size, count in a["consensus_clusters"].items():
            add(f"  {size} independent reference group(s): {count} decision(s)")
    else:
        add("  no multi-reference consensus recorded yet")
    if a["consensus_mean"] is not None:
        add(f"  mean consensus score: {a['consensus_mean']}")
    add("")

    add("INVESTIGATION CANDIDATES (reference)")
    add("-" * 74)
    for item in reference_investigation_candidates(a):
        add(f"  {item}")
    add("")
    add("  Reference policy is NOT changed by this report. Selection order is")
    add("  unchanged; only a verified claim can be withheld by an unproven reference.")
    add(rule)
    return "\n".join(out)


def reference_investigation_candidates(a: dict[str, Any]) -> list[str]:
    out: list[str] = []
    if a["serve_records"] < a["min_observations"]:
        out.append(
            f"I. collect more data: {a['serve_records']} serve records is below the "
            f"minimum {a['min_observations']}"
        )
        return out
    if a["verified_on_unproven_reference"]:
        out.append(
            "J. consider weighting reference trust when promoting a decision to verified"
        )
    else:
        out.append("I. keep reference policy unchanged: no verified decision rests on an unproven reference")
    if a["reference_failures"]:
        out.append("K. investigate why references are failing health checks")
    if not a["consensus_clusters"]:
        out.append("L. multi-reference consensus is not observable yet; only one reference is fetched per request")
    return out


# --------------------------------------------------------------------------- #
# Shadow reference-selection comparison
# --------------------------------------------------------------------------- #

# Reference trust strengths, for comparing legacy vs shadow picks.
_TRUST_STRENGTH = {"verified": 3, "strong": 2, "acceptable": 1, "unknown": 0}


def shadow_replay_analysis(
    records: list[dict[str, Any]], case_id: str, audit_path: Path
) -> tuple[dict[str, Any], str]:
    """Offline reference replay for a single recorded case.

    A real replay needs the actual reference payloads. The audit deliberately
    stores no subtitle text, so this reports SIMULATION_UNAVAILABLE rather than
    approximating an alignment from metadata. It never re-runs alass over
    production data.
    """
    matches = [r for r in records if str(r.get("request_id") or "") == case_id]
    unavailable = {
        "available": False,
        "case_id": case_id,
        "case_found": bool(matches),
        "records_scanned": len(records),
        "audit_path": str(audit_path),
        "reason": (
            "SIMULATION_UNAVAILABLE: the audit stores no subtitle text or reference "
            "payload, so the reference for this case cannot be re-aligned offline. "
            "Re-running alass across production audit data is not performed by default."
        ),
        "recovered_artifacts": [],
    }
    if not matches:
        unavailable["reason"] = (
            f"SIMULATION_UNAVAILABLE: no record with request_id={case_id!r} in "
            f"{audit_path}; checked {len(records)} records"
        )

    out: list[str] = []
    add = out.append
    add("OFFLINE SHADOW REPLAY")
    add("---------------------")
    add(f"  Case: {case_id}")
    add(f"  Case found in audit: {'yes' if matches else 'no'}")
    add("")
    add(f"  {unavailable['reason']}")
    add("")
    add("  No approximate result is produced. Collecting the reference payloads")
    add("  needed for a true replay is a separate, opt-in change.")
    return unavailable, "\n".join(out)


def shadow_comparison_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare the legacy reference pick against the shadow (v2) pick.

    Purely observational. Where the two disagree, this records *that* they
    disagree and on which axis; it never declares one policy better.
    """
    serve = [r for r in records if r.get("phase") == "serve"]

    legacy = [r for r in serve if r.get("reference_trust")]
    compared = [r for r in legacy if r.get("shadow_reference_id")]
    changed = [r for r in compared if r.get("shadow_reference_changed")]

    by_reason: Counter = Counter()
    for record in changed:
        for reason in record.get("shadow_reasons") or []:
            by_reason[reason.split(":")[0].strip()[:60]] += 1

    def _axis(records_subset: list[dict[str, Any]], key: str) -> dict[str, int]:
        return dict(Counter(str(r.get(key) or "unknown") for r in records_subset))

    potential = [r for r in serve if r.get("potential_reference_selection_issue")]
    with_verified_cache = [
        r
        for r in changed
        if r.get("shadow_reference_trust") in ("verified", "strong")
    ]
    multi_group = [r for r in changed if (r.get("shadow_independent_groups") or 0) > 1]

    # --- pool coverage ---------------------------------------------------- #
    # The legacy resolver stops at the first candidate that passes cue sanity,
    # so a one-candidate pool is the common case and records "agreement"
    # between policies that were never actually compared. Coverage is reported
    # first, and disagreement over meaningful comparisons only.
    with_pool = [r for r in legacy if r.get("shadow_pool_materialized")]
    two_plus = [r for r in with_pool if (r.get("shadow_pool_size") or 0) >= 2]
    two_plus_groups = [
        r for r in with_pool if (r.get("shadow_pool_independent_groups") or 0) >= 2
    ]
    one_candidate = [r for r in with_pool if (r.get("shadow_pool_size") or 0) < 2]
    truncated = [r for r in with_pool if r.get("shadow_pool_truncated")]
    limited = [r for r in with_pool if r.get("shadow_pool_limited_by_payloads")]
    extra_fetches = sum(int(r.get("shadow_additional_fetches") or 0) for r in with_pool)

    # Mirrored from the v2 reference selector (app.services.sync.reference_v2).
    # This tool stays stdlib-only by design so analysis can run without the app
    # installed; test_tool_does_not_import_application_modules enforces that,
    # and test_shadow_comparison_class_literals_match_the_selector guards drift.
    COMPARISON_NO = "NO_COMPARISON"
    COMPARISON_MEANINGFUL = "MEANINGFUL_COMPARISON"

    def _classify(record: dict[str, Any]) -> str:
        explicit = record.get("shadow_comparison_class")
        if explicit in (COMPARISON_NO, COMPARISON_MEANINGFUL):
            return str(explicit)
        # Records predating pool telemetry: infer, never assume agreement.
        size = int(record.get("shadow_pool_size") or 0)
        groups = int(record.get("shadow_pool_independent_groups") or 0)
        if size < 2 or groups < 2:
            return COMPARISON_NO
        return COMPARISON_MEANINGFUL
    meaningful = [r for r in compared if _classify(r) == COMPARISON_MEANINGFUL]
    non_comparable = [r for r in compared if _classify(r) != COMPARISON_MEANINGFUL]
    meaningful_changed = [r for r in meaningful if r.get("shadow_reference_changed")]

    coverage = {
        "serve_records": len(serve),
        "with_pool": len(with_pool),
        "two_or_more_candidates": len(two_plus),
        "two_or_more_independent_groups": len(two_plus_groups),
        "one_candidate_only": len(one_candidate),
        "truncated_by_limit": len(truncated),
        "limited_by_available_payloads": len(limited),
        "additional_reference_fetches": extra_fetches,
        "pool_limit_values": dict(
            Counter(str(r.get("shadow_pool_limit") or "?") for r in with_pool)
        ),
        "mean_pool_size": round(
            sum(int(r.get("shadow_pool_size") or 0) for r in with_pool) / len(with_pool), 3
        )
        if with_pool
        else None,
    }

    # Switch simulation is only possible when the artifacts needed to re-score
    # exist locally. The audit deliberately does not store subtitle text, so a
    # real re-alignment is impossible from the record alone.
    simulation = {
        "available": False,
        "reason": (
            "SIMULATION_UNAVAILABLE: the audit stores no subtitle text or reference "
            "payload, so an offline re-alignment cannot be derived from the record"
        ),
    }

    return {
        "serve_records": len(serve),
        "legacy_assessed": len(legacy),
        "compared": len(compared),
        "agreed": len(compared) - len(changed),
        "changed": len(changed),
        "agreement_rate": round((len(compared) - len(changed)) / len(compared), 4)
        if compared
        else None,
        "meaningful_comparisons": len(meaningful),
        "meaningful_agreed": len(meaningful) - len(meaningful_changed),
        "meaningful_changed": len(meaningful_changed),
        "meaningful_agreement_rate": round(
            (len(meaningful) - len(meaningful_changed)) / len(meaningful), 4
        )
        if meaningful
        else None,
        "non_comparable": len(non_comparable),
        "pool_coverage": coverage,
        "comparison_class_counts": dict(Counter(_classify(r) for r in compared).most_common()),
        "change_reasons": dict(by_reason.most_common()),
        "meaningful_change_reasons": dict(
            Counter(
                reason.split(":")[0].strip()[:60]
                for r in meaningful_changed
                for reason in (r.get("shadow_reasons") or [])
            ).most_common()
        ),
        "switch_labels": dict(
            Counter(
                label
                for r in meaningful_changed
                for label in (r.get("shadow_switch_labels") or [])
            ).most_common()
        ),
        "changed_by_match_tier": _axis(changed, "match_tier"),
        "changed_by_provider": _axis(changed, "provider"),
        "changed_by_source": _axis(changed, "release_source"),
        "changed_by_legacy_trust": _axis(changed, "reference_trust"),
        "changed_by_legacy_health": _axis(changed, "reference_health"),
        "changed_rate": round(len(changed) / len(compared), 4) if compared else None,
        "changed_when_shadow_verified": len(with_verified_cache),
        "changed_with_multiple_independent_groups": len(multi_group),
        "potential_reference_selection_issue": len(potential),
        "legacy_verified_outcomes": sum(
            1
            for r in serve
            if r.get("sync_state") in VERIFIED_OUTCOMES
        ),
        "switch_simulation": simulation,
    }


def render_shadow_report(a: dict[str, Any]) -> str:
    out: list[str] = []
    add = out.append
    rule = "=" * 74
    add(rule)
    add("SHADOW REFERENCE SELECTION COMPARISON (v2 vs current)")
    cov = a.get("pool_coverage") or {}
    add("")
    add("Shadow Coverage")
    add("---------------")
    add(f"  Requests with shadow pool: {cov.get('with_pool', 0)}")
    add(f"  Requests with >=2 candidates: {cov.get('two_or_more_candidates', 0)}")
    add(
        "  Requests with >=2 independent timing groups: "
        f"{cov.get('two_or_more_independent_groups', 0)}"
    )
    add(f"  Requests with only one candidate: {cov.get('one_candidate_only', 0)}")
    add(f"  Requests truncated by pool limit: {cov.get('truncated_by_limit', 0)}")
    add(
        "  Requests limited by available payloads: "
        f"{cov.get('limited_by_available_payloads', 0)}"
    )
    add(f"  Extra reference fetches caused by the pool: {cov.get('additional_reference_fetches', 0)}")
    if cov.get("mean_pool_size") is not None:
        add(f"  Mean pool size: {cov['mean_pool_size']}")

    add("")
    add("Comparison classification")
    add("--------------------------")
    for cls, count in (a.get("comparison_class_counts") or {}).items():
        add(f"  {cls}: {count}")
    add(f"  Meaningful comparisons: {a.get('meaningful_comparisons', 0)}")
    add(f"  Non-comparable cases: {a.get('non_comparable', 0)}")
    if not a.get("meaningful_comparisons"):
        add("")
        add("  NO MEANINGFUL COMPARISONS YET. The legacy resolver stops at the first")
        add("  candidate that passes cue sanity, so a low disagreement rate here would")
        add("  reflect the early break, not policy agreement. Read meaningful_agreement")
        add("  only, and only once this count is non-zero.")

    add("")
    add("Agreement (raw vs meaningful)")
    add("-----------------------------")
    add(f"  Raw agreement: {_pct(a.get('agreement_rate'))} over {a.get('compared', 0)} requests")
    add(
        f"  Meaningful agreement: {_pct(a.get('meaningful_agreement_rate'))} over "
        f"{a.get('meaningful_comparisons', 0)} requests"
    )
    add(
        f"  Meaningful disagreements: {a.get('meaningful_changed', 0)} "
        f"(agreed: {a.get('meaningful_agreed', 0)})"
    )
    if (a.get("compared") or 0) > (a.get("meaningful_comparisons") or 0):
        add("  The raw figure is inflated by requests where the shadow had no real")
        add("  alternative. Do not quote it as policy agreement.")

    if a.get("meaningful_change_reasons"):
        add("")
        add("Meaningful disagreement axes (legacy -> shadow)")
        for reason, count in a["meaningful_change_reasons"].items():
            add(f"  {count:>4}  {reason}")
    if a.get("switch_labels"):
        add("")
        add("Counterfactual quality labels (not a quality claim)")
        for label, count in a["switch_labels"].items():
            add(f"  {count:>4}  {label}")
    add(rule)
    add(f"  serve records            : {a['serve_records']}")
    add(f"  legacy references graded : {a['legacy_assessed']}")
    add(f"  raw comparable pairs    : {a['compared']}  (NOT policy agreement)")
    add(f"  meaningful comparisons  : {a.get('meaningful_comparisons', 0)}")
    add(
        f"  meaningful agreement     : {_pct(a.get('meaningful_agreement_rate'))}"
    )
    add("")
    add("  The raw comparable-pair count includes requests where the legacy")
    add("  resolver's early break left the shadow with a single candidate and")
    add("  therefore no choice. Read the meaningful figures only.")

    add("")
    add("All disagreement axes below (legacy -> shadow), for context")
    add("-" * 74)
    for title, key in (
        ("match tier", "changed_by_match_tier"),
        ("provider", "changed_by_provider"),
        ("release source", "changed_by_source"),
        ("legacy trust", "changed_by_legacy_trust"),
        ("legacy health", "changed_by_legacy_health"),
    ):
        values = a[key]
        add(f"  {title:16s} {values if values else 'n/a'}")
    if a["change_reasons"]:
        add("  recorded reasons:")
        for reason, count in a["change_reasons"].items():
            add(f"      {count:5d}  {reason}")
    add("")

    add("POTENTIAL_REFERENCE_SELECTION_ISSUE")
    add("-" * 74)
    add(f"  cases: {a['potential_reference_selection_issue']}")
    add("  A weaker legacy pick, a stronger shadow pick, and a decision that did not")
    add("  verify. This is a CANDIDATE for investigation, not evidence that the")
    add("  reference caused the failure.")
    add("")

    add("Reference switch simulation")
    add("-" * 74)
    add(f"  {a['switch_simulation']['reason']}")
    add("  Re-running alass across the dataset is deliberately not done: it would")
    add("  add runtime cost to production and duplicate a real sync pass.")
    add("")

    add("INVESTIGATION CANDIDATES (reference selection)")
    add("-" * 74)
    for item in shadow_investigation_candidates(a):
        add(f"  {item}")
    add("")
    add("  Neither policy is declared better. Promotion requires measured")
    add("  verification success, false-positive rate, and unverified/rejected")
    add("  rates over a real sample - not this comparison alone.")
    add(rule)
    return "\n".join(out)


def shadow_investigation_candidates(a: dict[str, Any]) -> list[str]:
    out: list[str] = []
    if a["compared"] == 0:
        out.append("M. no comparable shadow pairs yet; keep collecting observations")
        return out
    if a["changed"] == 0:
        out.append("N. legacy and shadow agree on every observation so far")
    else:
        out.append(
            f"O. {a['changed']} of {a['compared']} observations disagree "
            f"({_pct(a['changed_rate'])}); inspect the recorded reasons"
        )
    if a["potential_reference_selection_issue"]:
        out.append(
            "P. review POTENTIAL_REFERENCE_SELECTION_ISSUE cases before considering any change"
        )
    if a["changed_when_shadow_verified"]:
        out.append(
            "Q. shadow prefers a verified-cache reference in some cases; check whether "
            "exact cache evidence should outrank release identity"
        )
    return out

if __name__ == "__main__":
    raise SystemExit(main())
