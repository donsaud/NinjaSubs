"""Simulated captures in temporary roots; never new REAL corpus evidence."""
from __future__ import annotations

import asyncio
import json

import pytest

from app.services.sync import case_capture as cc
from app.services.sync.alignment import SubtitleEvaluation
from app.services.sync.boundary_verifier import verify_boundary
from app.services.sync.ninja_points import render_srt
from app.services.sync.parser_projection import canonical_bytes, project, render_projection, sha
from tools import capture_v2 as v2
from tools.import_sync_case import validate_srt
from tools.sync_replay import ROOT, load_text, replay


@pytest.fixture(autouse=True)
def writer_cleanup():
    yield
    assert cc.flush_captures()
    asyncio.run(cc.close_captures())
    cc.current_capture.set(None)


def make_capture(tmp_path, *, positive=False):
    if positive:
        spec = {"path": "tests/fixtures/boundary/dexter_s08e04_timings.json"}
        target, reference, output = [load_text({**spec, "role": r}, ROOT)
                                     for r in ("target", "reference", "transformed")]
    else:
        target = render_srt([(10000+i*5000, 11000+i*5000, "Dialogue event.") for i in range(50)])
        # Empty and untimed blocks are exactly the production omission semantics.
        target += "\n51\n00:05:00,000 --> 00:05:01,000\n\nUntimed annotation.\n"
        reference = render_srt([(10000+i*5000, 11000+i*5000, "Reference dialogue.")
                                for i in range(50)])
        output = render_srt([(10000+i*5000, 11000+i*5000, "Dialogue event.")
                             for i in range(49)])
    raw = output.replace("\n", "\r\n").encode()
    folder = tmp_path / "capture"
    owner = cc.CaseCapture(folder, {"imdb_id": "tt1234567", "season": 8, "episode": 7},
                           target.encode(), "simulated test capture")
    owner.alass_bytes(raw, reference, reference, target)
    owner.verifier(reference, output, SubtitleEvaluation(alass_applied=True, alass_successful=True))
    owner.boundary(reference, output, verify_boundary(target, reference, output), target)
    owner.response_input(output.encode())
    owner.final(output.encode(), "srt")
    assert cc.flush_captures()
    return folder, target, reference, output


def sandbox(tmp_path):
    root = tmp_path / "repo"
    (root / "tests/replay").mkdir(parents=True)
    (root / "tests/replay/corpus.json").write_text('{"schema_version":1,"cases":[]}')
    return root


def test_v2_preserves_original_raw_and_projection(tmp_path):
    folder, target, reference, output = make_capture(tmp_path)
    artifacts, result = v2.check_capture(folder, "attempt-01")
    assert artifacts["pristine_target.srt"] == target.encode()
    assert artifacts["attempt-01/raw_transformed.srt"] == output.replace("\n", "\r\n").encode()
    manifest = json.loads(artifacts["attempt-01/boundary_parser_projection.json"])
    assert "Dialogue" not in artifacts["attempt-01/boundary_parser_projection.json"].decode()
    assert manifest["parser_fingerprint_schema"] == 2
    assert manifest["projections"]["target"] == project(target)
    assert project(target) == project(target)
    assert project(render_projection(project(target))) == project(target)
    assert len(project(target)["cues"]) == 50
    assert not result["shadow"]["accepted"]
    assert result["shadow"]["reasons"] == ["cue_population_changed"]
    assert sha(artifacts["pristine_target.srt"]) == manifest["original_target_sha256"]
    assert project(reference) == manifest["projections"]["reference"]


@pytest.mark.parametrize("text", [
    "1\n00:00:10,000 --> 00:00:09,000\nDialogue.",
    "1\ninvalid --> invalid\nDialogue.",
    "1\n00:00:10,000 --> 00:00:11,000\nDialogue.\n00:00:12,000 --> 00:00:13,000\nOther.",
    render_srt([(20000,22000,"Later."),(10000,11000,"Earlier.")]),
])
def test_ambiguous_invalid_or_reordered_projection_rejected(text):
    with pytest.raises(ValueError):
        project(text)
    if text.count("-->") != 2 or "Later." in text:
        with pytest.raises(ValueError):
            validate_srt(text.encode())


@pytest.mark.parametrize("name", ["pristine_target.srt", "attempt-01/selected_reference.srt",
    "attempt-01/raw_transformed.srt", "final_served.srt",
    "attempt-01/boundary_parser_projection.json"])
def test_all_artifact_sha_mismatches_rejected(tmp_path, name):
    folder, *_ = make_capture(tmp_path)
    p = folder / name
    p.write_bytes(p.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA mismatch"):
        v2.check_capture(folder, "attempt-01")


def test_tampered_projection_with_updated_hash_still_rejected(tmp_path):
    folder, *_ = make_capture(tmp_path)
    name = "attempt-01/boundary_parser_projection.json"
    value = json.loads((folder/name).read_bytes())
    value["projections"]["target"]["cues"][0][0] += 1
    raw = canonical_bytes(value)
    (folder/name).write_bytes(raw)
    metadata = json.loads((folder/"metadata.json").read_bytes())
    metadata["artifact_sha256"][name] = sha(raw)
    metadata["attempts"][0]["parser_projection_sha256"] = sha(raw)
    (folder/"metadata.json").write_bytes(canonical_bytes(metadata))
    with pytest.raises(ValueError, match="tampered"):
        v2.check_capture(folder, "attempt-01")


def test_parser_drift_is_refused(tmp_path, monkeypatch):
    folder, *_ = make_capture(tmp_path)
    monkeypatch.setattr(v2, "implementation_sha", lambda: "different-parser")
    with pytest.raises(ValueError, match="parser drift"):
        v2.check_capture(folder, "attempt-01")


def test_v1_not_retroactively_upgraded(tmp_path):
    folder, *_ = make_capture(tmp_path)
    p = folder / "metadata.json"
    value = json.loads(p.read_bytes())
    value["schema_version"] = 1
    p.write_bytes(canonical_bytes(value))
    with pytest.raises(ValueError, match="forward V2"):
        v2.check_capture(folder, "attempt-01")


def test_import_replay_duplicate_and_source_immutability(tmp_path, monkeypatch):
    import socket
    import subprocess
    def deny(*a, **k):
        raise AssertionError("offline only")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(subprocess, "run", deny)
    folder, *_ = make_capture(tmp_path)
    before = {p: p.read_bytes() for p in folder.rglob('*') if p.is_file()}
    root = sandbox(tmp_path)
    with pytest.raises(ValueError, match="provenance"):
        v2.import_capture(root, folder, "attempt-01", "simulated-unit-case", live_confirmed=False, expected_boundary_accepted=False, outcome_confirmed=True)
    entry = v2.import_capture(root, folder, "attempt-01", "simulated-unit-case",
                              live_confirmed=True, expected_boundary_accepted=False, outcome_confirmed=True,
                              meaningful_negative=True)
    # REAL confirmation is simulated only in this isolated unit-test repository.
    assert entry["provenance"] == "real"
    assert before == {p: p.read_bytes() for p in before}
    first, second = replay(root/"tests/replay/corpus.json", root), replay(root/"tests/replay/corpus.json", root)
    assert first == second
    assert not first["expectation_mismatches"]
    with pytest.raises(ValueError, match="duplicate"):
        v2.import_capture(root, folder, "attempt-01", "another-case", live_confirmed=True,
                          expected_boundary_accepted=False, outcome_confirmed=True)
    monkeypatch.undo()  # Restore sockets before Windows asyncio teardown creates its self-pipe.


def test_derived_timing_seed_matches_verifier_positive_without_new_real_evidence(tmp_path):
    folder, *_ = make_capture(tmp_path, positive=True)
    _, evidence = v2.check_capture(folder, "attempt-01")
    assert evidence["shadow"]["accepted"]
    assert evidence["shadow"]["classification"] == "EDGE_DISCONTINUITY"
    assert len(evidence["shadow"]["segments"]) == 2


def test_projection_failure_is_export_only(tmp_path, monkeypatch):
    from app.services.sync import parser_projection as pp
    def fail(*a):
        raise RuntimeError("private failure")
    monkeypatch.setattr(pp, "project", fail)
    folder, _, _, output = make_capture(tmp_path)
    # Delivery construction is independent; diagnostic manifest can be incomplete.
    from app import main
    response = main._build_subtitle_response(output.encode(), "safe", enable_rtl_fix=False)
    assert response.body
    assert not (folder/"attempt-01/boundary_parser_projection.json").exists()


async def test_shutdown_drains_v2_projection(tmp_path):
    folder, *_ = make_capture(tmp_path)
    await cc.close_captures()
    assert cc._writer is None
    v2.check_capture(folder, "attempt-01")


def test_evidence_gate_supporting_duplicates_and_global_controls():
    from tools.boundary_evidence_gate import gate_counts
    rows = [
        {"case_id": "edge", "identity": ["target", "reference"],
         "category": "REPLAYABLE_REAL_POSITIVE", "media_identity": ["tt1234567", 8, 4]},
        {"case_id": "repeat", "identity": ["target", "reference"],
         "category": "SUPPORTING_REAL_LIVE_POSITIVE", "media_identity": ["tt1234567", 8, 7]},
        {"case_id": "repeat-with-new-bytes", "identity": ["another-target", "another-reference"],
         "category": "SUPPORTING_REAL_LIVE_POSITIVE", "media_identity": ["tt1234567", 8, 4]},
        {"case_id": "live-negative", "identity": ["negative", "reference"],
         "category": "SUPPORTING_REAL_LIVE_NEGATIVE", "meaningful_negative": True},
        {"case_id": "normal-affine", "identity": ["normal", "reference"],
         "category": "REPLAYABLE_REAL_NEGATIVE", "meaningful_negative": False},
        {"case_id": "derived", "identity": ["derived", "reference"], "category": "DERIVED"},
        {"case_id": "synthetic", "identity": ["synthetic", "reference"], "category": "SYNTHETIC"},
        {"case_id": "unknown", "identity": ["unknown", "unknown"], "category": "UNKNOWN"},
    ]
    report = {"aggregate": {"false_positives": 0, "unnecessary_escalations": 0},
              "expectation_mismatches": [], "safety_regressions": 0}
    counts = gate_counts(rows, report)
    assert counts["REPLAYABLE_REAL_POSITIVE"] == 1
    assert counts["SUPPORTING_REAL_LIVE_POSITIVE"] == 0
    assert counts["SUPPORTING_REAL_LIVE_NEGATIVE"] == 1
    assert counts["replayable_real_meaningful_negatives"] == 0
    assert counts["DERIVED"] == counts["SYNTHETIC"] == counts["UNKNOWN"] == 1
    assert not counts["gate_passed"]


def test_no_safety_attestation_cannot_pass_gate():
    from tools.boundary_evidence_gate import gate_counts
    rows = [{"case_id": str(i), "identity": [str(i), "reference"],
             "category": "REPLAYABLE_REAL_POSITIVE" if i < 2 else "REPLAYABLE_REAL_NEGATIVE",
             "meaningful_negative": True} for i in range(4)]
    report = {"aggregate": {"false_positives": 0, "unnecessary_escalations": 0},
              "expectation_mismatches": []}
    assert not gate_counts(rows, report)["gate_passed"]
    report["safety_regressions"] = 0
    assert gate_counts(rows, report)["gate_passed"]
    report["aggregate"]["false_positives"] = 1
    assert not gate_counts(rows, report)["gate_passed"]
