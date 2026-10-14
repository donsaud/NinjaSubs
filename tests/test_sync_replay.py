"""Real seed replay, derived negative controls and portable offline extension."""
from __future__ import annotations

import copy
import json

import pytest

from tools.sync_replay import MANIFEST, ROOT, derive, fixture_path, replay, replay_case


@pytest.fixture(scope="module")
def corpus():
    from tests.private_fixtures import require_many

    cases = json.loads(MANIFEST.read_text())["cases"]
    require_many([ROOT / spec["path"] for case in cases
                  for spec in case.get("fixtures", {}).values()
                  if "path" in spec and spec["path"].endswith((".srt", ".vtt", ".ass", ".ssa"))]
                 + [ROOT / case["capture_v2"]["path"] / "pristine_target.srt"
                    for case in cases if "capture_v2" in case])
    first, second = replay(), replay()
    # Durations are observations, deliberately excluded from deterministic evidence.
    assert [r["evidence"] for r in first["cases"]] == [r["evidence"] for r in second["cases"]]
    assert first["aggregate"] == second["aggregate"]
    assert first["expectation_mismatches"] == second["expectation_mismatches"] == []
    return first


def test_confusion_matrix(corpus):
    assert corpus["aggregate"] == {
        "expected_positives": 1, "true_positives": 1, "false_positives": 0,
        "expected_negatives": 14, "true_negatives": 14, "false_negatives": 0,
        "unnecessary_escalations": 0,
    }


def test_dexter_raw_and_served_and_legacy(corpus):
    result = corpus["cases"][0]["evidence"]
    assert result["production"]["p95_offset_ms"] == 4489
    assert result["production"]["sync_state"] == "unverified"
    assert result["production"]["verification"] == "unknown"
    for s in (result["shadow"], result["final_served_shadow"]):
        assert s["accepted"] and s["classification"] == "EDGE_DISCONTINUITY"
        assert len(s["segments"]) == 2
        assert s["boundary_interval_ms"] == (187250, 193890)
        assert (s["offset_a_ms"], s["offset_b_ms"], s["jump_ms"]) == (-96961, 9661, 106622)
        assert s["gap_disagreement_ms"] == 58 and s["holdout_pass"]
        assert [e["residual_p95_ms"] for e in s["segments"]] == [0, 0]


def test_whiplash_stays_affine(corpus):
    result = corpus["cases"][1]["evidence"]
    model = result["simpler_model"]
    assert model["accepted"] and model["supports_fixed_output"]
    assert model["classification"] == "PAL_LIKE_AFFINE"
    assert model["slope"] == pytest.approx(1.042755149812734)
    assert model["intercept_ms"] == pytest.approx(-8621.594569288398)
    assert not result["shadow"]["accepted"]
    assert result["shadow"]["reasons"] == ("simpler_verified_model",)


def test_all_declared_negative_controls(corpus):
    manifest = json.loads(MANIFEST.read_text())
    for case, result in zip(manifest["cases"], corpus["cases"], strict=True):
        if not case["expected_shadow_outcome"]:
            assert not result["evidence"]["shadow"]["accepted"], case["case_id"]
            assert not result["evidence"]["ungated_shadow"]["accepted"], case["case_id"]
            assert case["expected_reason"] in result["evidence"]["shadow"]["reasons"]
        if "capture_v2" in case:
            # V2 replay keeps deterministic projection evidence without timing observations.
            assert result["observations"] == {}
        else:
            assert result["observations"]["shadow_duration_ms"] >= 0


def test_global_reference_fit_cannot_hide_wrong_fixed_artifact():
    case = copy.deepcopy(json.loads(MANIFEST.read_text())["cases"][10])
    case["fixtures"]["transformed"]["operations"] = [{"kind": "shift", "ms": 17000}]
    result = replay_case(case)["evidence"]
    assert result["simpler_model"]["accepted"]
    assert not result["simpler_model"]["supports_fixed_output"]
    assert result["boundary_should_run"]
    assert result["shadow"]["reasons"] == ("not_exactly_one_boundary",)


@pytest.mark.parametrize("value", ["../escape.srt", "C:/private.srt", "https://private.invalid", "/tmp/sub.srt", "..\\escape.srt"])
def test_fixture_paths_are_portable_and_contained(value):
    with pytest.raises(ValueError):
        fixture_path(value, ROOT)


def test_unknown_derivation_is_refused():
    with pytest.raises(ValueError, match="unknown corpus operation"):
        derive("", [{"kind": "unknown"}])


def test_duplicate_manifest_ids_are_refused(tmp_path):
    case = json.loads(MANIFEST.read_text())["cases"][0]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": [case, case]}))
    with pytest.raises(ValueError, match="duplicate case_id"):
        replay(manifest)


def test_symlink_escape_is_refused(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.srt"
    outside.write_text("outside")
    try:
        (root / "link.srt").symlink_to(outside)
    except OSError:
        pytest.skip("platform lacks symlink privileges")
    with pytest.raises(ValueError, match="outside corpus root"):
        fixture_path("link.srt", root)


def test_new_srt_case_needs_only_manifest_entry_and_files(tmp_path):
    text = "1\n00:00:10,000 --> 00:00:11,000\nDialogue.\n"
    (tmp_path / "target.srt").write_text(text)
    case = copy.deepcopy(json.loads(MANIFEST.read_text())["cases"][3])
    case["case_id"] = "new_portable_case"
    case["fixtures"] = {role: {"path": "target.srt"}
                        for role in ("target", "reference", "transformed")}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "cases": [case]}))
    result = replay(manifest, tmp_path)
    assert result["expectation_mismatches"] == []
    assert result["aggregate"]["true_negatives"] == 1


def test_empty_corpus_is_not_success(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"cases": []}')
    with pytest.raises(ValueError, match="empty corpus"):
        replay(manifest)


@pytest.mark.parametrize("mismatch", [False, True])
def test_cli_writes_evidence_only_and_returns_expectation_status(corpus, tmp_path, monkeypatch, mismatch):
    from tools import sync_replay
    report = copy.deepcopy(corpus)
    if mismatch:
        report["expectation_mismatches"] = ["injected_test_mismatch"]
    output = tmp_path / "report.json"
    monkeypatch.setattr(sync_replay, "replay", lambda *args: report)
    monkeypatch.setattr("sys.argv", ["sync_replay", "--output", str(output), "--evidence-only"])
    assert sync_replay.main() == int(mismatch)
    saved = json.loads(output.read_text())
    assert all("observations" not in case for case in saved["cases"])
    assert "Dialogue event." not in output.read_text()
