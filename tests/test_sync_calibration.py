"""Tests for the offline calibration tool.

The tool is read-only analysis, so the tests focus on the properties that make
its output trustworthy: correct outcome separation, honest sample-size gating,
sound interval math, working data-quality detection, and - most importantly -
that it cannot be mistaken for something that tunes the system.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_TOOL = Path("tools/calibrate_sync.py")


def _load_tool():
    """Import the tool by path; it lives outside the importable package."""
    spec = importlib.util.spec_from_file_location("calibrate_sync", _TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["calibrate_sync"] = module
    spec.loader.exec_module(module)
    return module


cal = _load_tool()


def _record(
    *,
    phase="serve",
    rule="exact_release",
    outcome="verified_resynced",
    prediction=True,
    source="WEB-DL",
    confidence=90.0,
    provider="subdl",
    verification="verified",
    video_id="v1",
    subtitle_id="s1",
    **extra,
):
    record = {
        "schema_version": 1,
        "engine_version": "sync-eval-1",
        "timestamp": "2026-09-29T07:00:00+00:00",
        "phase": phase,
        "video_id": video_id,
        "subtitle_id": subtitle_id,
        "language": "ara",
        "has_video_fingerprint": True,
        "request_context": "resolved_stream",
        "match_tier": "EXACT",
        "provider": provider,
        "release_source": source,
        "verification": verification,
        "sync_state": outcome,
        "from_cache": False,
        "reasons": [],
    }
    if prediction:
        record["prediction_state"] = "probable_sync"
        record["prediction_rule_at_search"] = rule
        record["prediction_confidence"] = confidence
    record.update(extra)
    return record


def _write(tmp_path: Path, records, *, raw_extra: str = "") -> Path:
    path = tmp_path / "audit.jsonl"
    lines = [json.dumps(r) for r in records]
    path.write_text("\n".join(lines) + ("\n" + raw_extra if raw_extra else ""), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Wilson intervals
# --------------------------------------------------------------------------- #


def test_wilson_interval_brackets_the_observation():
    low, high = cal.wilson_interval(78, 100)
    assert low < 0.78 < high
    # Wilson is conservative near 1.0, unlike a normal approximation.
    low, high = cal.wilson_interval(100, 100)
    assert high == pytest.approx(1.0)
    # Still strictly below 1.0, which a naive point estimate of 100% would not be.
    assert 0.9 < low < 1.0


def test_wilson_interval_handles_empty_samples():
    assert cal.wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_narrows_as_the_sample_grows():
    narrow = cal.wilson_interval(900, 1000)
    wide = cal.wilson_interval(9, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


# --------------------------------------------------------------------------- #
# Outcome separation
# --------------------------------------------------------------------------- #


def test_verified_synced_and_resynced_stay_separate(tmp_path):
    records = [
        _record(outcome="verified_synced", subtitle_id="a"),
        _record(outcome="verified_resynced", subtitle_id="b"),
        _record(outcome="unverified", subtitle_id="c"),
        _record(outcome="rejected", subtitle_id="d", verification="verified"),
    ]
    analysis = cal.analyze(cal.load_records(_write(tmp_path, records)), min_rule=1, min_group=1)
    stats = analysis["rules"]["exact_release"]
    assert stats["verified_synced"] == 1
    assert stats["verified_resynced"] == 1
    assert stats["unverified"] == 1
    assert stats["rejected"] == 1
    assert stats["verified_any"] == 2


def test_a_rule_that_only_resyncs_is_not_reported_as_fully_verified(tmp_path):
    records = [_record(outcome="verified_resynced", subtitle_id=f"s{i}") for i in range(10)]
    analysis = cal.analyze(cal.load_records(_write(tmp_path, records)), min_rule=1, min_group=1)
    stats = analysis["rules"]["exact_release"]
    assert stats["verified_synced"] == 0
    assert stats["verified_resynced"] == 10


# --------------------------------------------------------------------------- #
# PREDICTED != VERIFIED
# --------------------------------------------------------------------------- #


def test_only_serve_phase_records_become_observations(tmp_path):
    records = [_record(phase="search", prediction=False, outcome="probable_sync")]
    analysis = cal.analyze(cal.load_records(_write(tmp_path, records)), min_rule=1, min_group=1)
    assert analysis["observations"] == 0


def test_successful_alass_exit_alone_is_not_a_verified_outcome(tmp_path):
    """A serve record with alass_applied but no measured state stays unknown."""
    records = [
        _record(
            outcome="probable_sync",
            alass_applied=True,
            alass_successful=True,
            subtitle_id=f"s{i}",
        )
        for i in range(5)
    ]
    analysis = cal.analyze(cal.load_records(_write(tmp_path, records)), min_rule=1, min_group=1)
    stats = analysis["rules"]["exact_release"]
    assert stats["verified_any"] == 0
    assert stats["unknown"] == 5


def test_a_prediction_record_never_populates_verified_buckets(tmp_path):
    records = [_record(phase="search", prediction=False, outcome="verified_resynced")]
    analysis = cal.analyze(cal.load_records(_write(tmp_path, records)), min_rule=1, min_group=1)
    assert analysis["observations"] == 0
    assert analysis["rules"] == {}


# --------------------------------------------------------------------------- #
# Minimum sample gating
# --------------------------------------------------------------------------- #


def test_small_groups_are_labelled_insufficient_sample(tmp_path):
    records = [
        _record(outcome="verified_synced", subtitle_id=f"s{i}") for i in range(3)
    ]
    analysis = cal.analyze(cal.load_records(_write(tmp_path, records)), min_rule=30, min_group=20)
    stats = analysis["rules"]["exact_release"]
    assert stats["total"] == 3
    assert stats["sufficient"] is False
    assert stats["confirmed_rate"] == 1.0
    report = cal.render_report(analysis)
    assert "INSUFFICIENT_SAMPLE" in report


def test_sufficient_group_is_not_labelled_insufficient(tmp_path):
    records = [
        _record(outcome="verified_synced", subtitle_id=f"s{i}") for i in range(40)
    ]
    analysis = cal.analyze(cal.load_records(_write(tmp_path, records)), min_rule=30, min_group=20)
    assert analysis["rules"]["exact_release"]["sufficient"] is True
    assert "INSUFFICIENT_SAMPLE" not in cal.render_report(
        cal.analyze(cal.load_records(_write(tmp_path, records)), min_rule=30, min_group=20)
    )


# --------------------------------------------------------------------------- #
# Data quality
# --------------------------------------------------------------------------- #


def test_malformed_lines_are_reported_not_silently_dropped(tmp_path):
    records = [_record(subtitle_id=f"s{i}") for i in range(5)]
    quality = cal.load_records(_write(tmp_path, records, raw_extra="{not json"))
    assert len(quality.malformed) == 1
    assert len(quality.records) == 5
    assert "malformed" in " ".join(quality.summary())


def test_duplicate_records_are_detected(tmp_path):
    record = _record()
    quality = cal.load_records(_write(tmp_path, [record, dict(record)]))
    assert quality.duplicates == 1
    assert len(quality.records) == 1


def test_verified_state_without_measurement_is_flagged_as_a_bug(tmp_path):
    records = [_record(outcome="verified_resynced", verification="predicted")]
    quality = cal.load_records(_write(tmp_path, records))
    assert quality.impossible_state_verification == 1
    assert quality.ok is False
    assert "BUG indicator" in " ".join(quality.summary())


def test_missing_engine_version_is_reported(tmp_path):
    record = _record()
    record["engine_version"] = ""
    quality = cal.load_records(_write(tmp_path, [record]))
    assert quality.missing_engine_version == 1


def test_unknown_schema_version_is_reported(tmp_path):
    record = _record()
    record["schema_version"] = 99
    quality = cal.load_records(_write(tmp_path, [record]))
    assert quality.unknown_schema_version == 1


def test_prediction_without_outcome_is_a_coverage_gap_not_a_failure(tmp_path):
    records = [
        _record(phase="search", outcome="probable_sync", verification="predicted",
                subtitle_id=f"p{i}")
        for i in range(4)
    ]
    analysis = cal.analyze(cal.load_records(_write(tmp_path, records)), min_rule=1, min_group=1)
    assert analysis["predictions_total"] == 4
    assert analysis["predictions_without_outcome"] == 4
    assert analysis["observations"] == 0


# --------------------------------------------------------------------------- #
# Stratification and calibration
# --------------------------------------------------------------------------- #


def test_source_class_normalization_reuses_known_labels():
    assert cal.normalize_source("web-dl") == "WEB-DL"
    assert cal.normalize_source("WEBDL") == "WEB-DL"
    assert cal.normalize_source("bluray") == "BluRay"
    assert cal.normalize_source("remux") == "BluRay"
    assert cal.normalize_source("hdtv") == "HDTV"
    assert cal.normalize_source("webrip") == "WEBRip"
    assert cal.normalize_source("") == "unknown"
    assert cal.normalize_source("something-else") == "unknown"


def test_rule_outcomes_are_stratified_by_source(tmp_path):
    records = [
        _record(outcome="verified_synced", source="WEB-DL", subtitle_id=f"w{i}")
        for i in range(15)
    ] + [
        _record(outcome="rejected", source="HDTV", verification="verified", subtitle_id=f"h{i}")
        for i in range(15)
    ]
    analysis = cal.analyze(cal.load_records(_write(tmp_path, records)), min_rule=10, min_group=10)
    assert analysis["by_source"]["WEB-DL"]["confirmed_rate"] == 1.0
    assert analysis["by_source"]["HDTV"]["confirmed_rate"] == 0.0


def test_confidence_calibration_is_reported_per_bucket(tmp_path):
    records = [
        _record(confidence=95.0, outcome="verified_synced", subtitle_id=f"a{i}")
        for i in range(10)
    ] + [
        _record(confidence=75.0, outcome="rejected", verification="verified", subtitle_id=f"b{i}")
        for i in range(10)
    ]
    analysis = cal.analyze(cal.load_records(_write(tmp_path, records)), min_rule=5, min_group=5)
    buckets = {c["confidence"]: c for c in analysis["confidence_calibration"]}
    assert buckets[95.0]["verified_synced"] == 10
    assert buckets[75.0]["rejected"] == 10


def test_inverted_confidence_ordering_is_detected(tmp_path):
    records = [
        _record(confidence=95.0, outcome="rejected", verification="verified", subtitle_id=f"a{i}")
        for i in range(10)
    ] + [
        _record(confidence=75.0, outcome="verified_synced", subtitle_id=f"b{i}")
        for i in range(10)
    ]
    analysis = cal.analyze(cal.load_records(_write(tmp_path, records)), min_rule=5, min_group=5)
    assert "NOT supported" in cal.render_report(analysis)


def test_rule_split_candidate_is_detected(tmp_path):
    records = [
        _record(outcome="verified_synced", source="WEB-DL", subtitle_id=f"w{i}")
        for i in range(20)
    ] + [
        _record(outcome="rejected", source="HDTV", verification="verified", subtitle_id=f"h{i}")
        for i in range(20)
    ]
    analysis = cal.analyze(cal.load_records(_write(tmp_path, records)), min_rule=10, min_group=10)
    assert analysis["split_candidates"]
    assert analysis["split_candidates"][0]["rule"] == "exact_release"
    assert "Candidate rule splits" in cal.render_report(analysis)


def test_concentration_is_reported(tmp_path):
    records = [
        _record(provider="subdl", subtitle_id=f"s{i}") for i in range(20)
    ]
    analysis = cal.analyze(cal.load_records(_write(tmp_path, records)), min_rule=5, min_group=5)
    assert analysis["concentration"]["top_provider"] == 1.0
    assert "Concentration warnings" in cal.render_report(analysis)


def test_match_tier_cross_tab_is_present(tmp_path):
    records = [_record(match_tier="HASH", subtitle_id=f"s{i}") for i in range(5)]
    analysis = cal.analyze(cal.load_records(_write(tmp_path, records)), min_rule=1, min_group=1)
    assert "HASH + exact_release" in analysis["rule_x_tier"]


# --------------------------------------------------------------------------- #
# The tool cannot tune anything
# --------------------------------------------------------------------------- #


def test_tool_does_not_import_application_modules():
    source = _TOOL.read_text(encoding="utf-8")
    assert "import app" not in source
    assert "from app" not in source


def test_shadow_comparison_class_literals_match_the_selector():
    """The tool mirrors the classification strings; they must not drift.

    The calibration tool is stdlib-only by design, so it cannot import the
    selector module that defines them.
    """
    from app.services.sync import reference_v2

    source = _TOOL.read_text(encoding="utf-8")
    assert f'COMPARISON_NO = "{reference_v2.COMPARISON_NO}"' in source
    assert f'COMPARISON_MEANINGFUL = "{reference_v2.COMPARISON_MEANINGFUL}"' in source


def test_shadow_expansion_outcome_literals_match_the_module():
    """The tool mirrors the expansion outcome vocabulary; it must not drift.

    The calibration tool is stdlib-only by design, so it cannot import the
    experimental module that defines these.
    """
    from app.services.sync import shadow_expansion

    source = _TOOL.read_text(encoding="utf-8")
    assert f'"{shadow_expansion.EXPANSION_NOT_ATTEMPTED}"' in source
    assert f'"{shadow_expansion.EXPANSION_NO_VALUE}"' in source
    assert f'"{shadow_expansion.EXPANSION_DUPLICATE_GROUP}"' in source
    assert f'"{shadow_expansion.EXPANSION_NEW_TIMING_GROUP}"' in source
    assert f'"{shadow_expansion.EXPANSION_ENABLED_COMPARISON}"' in source
    assert f'"{shadow_expansion.EXPANSION_FETCH_FAILURE}"' in source


def test_tool_writes_nothing_to_the_repo_by_default(tmp_path):
    records = [_record(subtitle_id=f"s{i}") for i in range(5)]
    path = _write(tmp_path, records)
    before = {p.name for p in tmp_path.iterdir()}
    assert cal.main([str(path), "--min-observations", "1"]) == 0
    after = {p.name for p in tmp_path.iterdir()}
    assert before == after, "the tool must not write files without --json-out"


def test_json_out_contains_aggregates_only(tmp_path):
    records = [_record(subtitle_id=f"s{i}") for i in range(5)]
    path = _write(tmp_path, records)
    out = tmp_path / "calibration.json"
    assert cal.main([str(path), "--min-observations", "1", "--json-out", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "rules" in payload and "by_source" in payload
    blob = json.dumps(payload)
    for forbidden in ("sub_text", "download_url", "api_key", "-->"):
        assert forbidden not in blob


def test_investigation_candidates_never_choose_an_option(tmp_path):
    records = [
        _record(outcome="verified_synced", source="WEB-DL", subtitle_id=f"w{i}")
        for i in range(20)
    ] + [
        _record(outcome="rejected", source="HDTV", verification="verified", subtitle_id=f"h{i}")
        for i in range(20)
    ]
    analysis = cal.analyze(cal.load_records(_write(tmp_path, records)), min_rule=10, min_group=10)
    items = cal.investigation_candidates(analysis)
    assert items
    letters = [item.split(".")[0] for item in items]
    assert len(letters) == len(set(letters)), "an option letter was emitted twice"
    report = cal.render_report(analysis)
    assert "INVESTIGATION CANDIDATES" in report
    assert "No option is selected automatically" in report


def test_thresholds_are_untouched_by_the_tool():
    """The tool cannot have changed any production constant."""
    from app.services.sync.alignment import MIN_CUES_FOR_VERIFIED
    from app.services.sync.predictor import (
        CONFIDENCE_EXACT_IDENTITY,
        CONFIDENCE_HASH,
        CONFIDENCE_PREDICTION_FLOOR,
        CONFIDENCE_SOURCE_EDITION,
    )

    assert MIN_CUES_FOR_VERIFIED == 12
    assert CONFIDENCE_HASH == 95.0
    assert CONFIDENCE_EXACT_IDENTITY == 90.0
    assert CONFIDENCE_SOURCE_EDITION == 75.0
    assert CONFIDENCE_PREDICTION_FLOOR == 70.0
