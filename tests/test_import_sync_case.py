"""Offline import admission, artifact identity, privacy and transaction failures."""
from __future__ import annotations

import hashlib
import json

import pytest

from app.services.sync.ninja_points import render_srt
from tools import import_sync_case as im
from tools.sync_replay import ROOT, case_mismatches, corpus_summary, replay


@pytest.fixture
def sandbox(tmp_path):
    (tmp_path / "tests/replay").mkdir(parents=True)
    (tmp_path / "tests/replay/corpus.json").write_text('{"schema_version": 1, "cases": []}')
    source = tmp_path / "incoming"
    source.mkdir()
    target = render_srt([(10000 + i * 5000, 11000 + i * 5000, f"Dialogue event {i}.")
                         for i in range(4)])
    reference = render_srt([(15000 + i * 5000, 16000 + i * 5000, "Reference dialogue.")
                            for i in range(4)])
    inputs = {}
    for role, text in (("target", target), ("reference", reference), ("transformed", target)):
        inputs[role] = source / (role + ".srt")
        inputs[role].write_text(text, encoding="utf-8")
    return tmp_path, inputs


def perform(sandbox, **kwargs):
    root, inputs = sandbox
    return im.import_case(root=root, case_id="local-sparse-case", inputs=inputs,
        case_type="INSUFFICIENT_EVIDENCE_REJECT", expected_boundary_accepted=False, **kwargs)


def assert_not_admitted(root):
    assert json.loads((root / "tests/replay/corpus.json").read_text())["cases"] == []
    assert not list((root / "tests/replay").glob(".sync-import-*"))
    assert not list((root / "tests/replay").glob("*.tmp"))
    assert not (root / "tests/replay/.case-import.lock").exists()
    assert not list((root / "tests/replay/cases").glob("*"))


def test_valid_real_import_reuses_whiplash_exact_fixtures(sandbox):
    root, _ = sandbox
    from tests.private_fixtures import require_many

    fixtures = root / "tests/fixtures/whiplash"
    fixtures.mkdir(parents=True)
    inputs = {}
    names = [("target", "target"), ("reference", "reference"), ("transformed", "expected")]
    require_many([ROOT / "tests/fixtures/whiplash" / (name + ".srt") for _, name in names])
    for role, name in names:
        source = ROOT / "tests/fixtures/whiplash" / (name + ".srt")
        path = fixtures / source.name
        path.write_bytes(source.read_bytes())
        inputs[role] = path
    report = im.import_case(root=root, case_id="whiplash-real-import", inputs=inputs,
        case_type="GLOBAL_AFFINE", expected_boundary_accepted=False,
        expected_parent_model="PAL_LIKE_AFFINE", metadata={"media_type": "movie"})
    assert report["passed"]
    assert report["replay"]["evidence"]["simpler_model"]["classification"] == "PAL_LIKE_AFFINE"
    folder = root / "tests/replay/cases/whiplash-real-import"
    assert sorted(p.name for p in folder.iterdir()) == ["case.json", "diagnostic.json"]
    entry = json.loads((folder / "case.json").read_text())
    assert entry["provenance"] == "real" and entry["source_case_id"] is None
    assert all(a["reused"] for a in entry["artifact_metadata"].values())
    assert entry["artifact_metadata"]["target"]["cue_count"] == 825
    assert entry["artifact_metadata"]["target"]["sha256"] == hashlib.sha256(inputs["target"].read_bytes()).hexdigest()
    assert "incoming" not in (folder / "diagnostic.json").read_text()
    assert report["replay"]["evidence"]["shadow"]["accepted"] is False


def test_duplicate_sha_within_and_between_cases(sandbox):
    root, inputs = sandbox
    inputs["served"] = inputs["target"]
    perform(sandbox)
    first = root / "tests/replay/cases/local-sparse-case"
    assert sorted(p.name for p in first.glob("*.srt")) == ["reference.srt", "target.srt"]
    entry = json.loads((first / "case.json").read_text())
    assert entry["fixtures"]["served"] == entry["fixtures"]["transformed"] == entry["fixtures"]["target"]
    report = im.import_case(root=root, case_id="second-sparse-case", inputs=inputs,
        case_type="INSUFFICIENT_EVIDENCE_REJECT", expected_boundary_accepted=False)
    assert all(a["reused"] for a in report["artifacts"].values())
    assert not list((root / "tests/replay/cases/second-sparse-case").glob("*.srt"))


@pytest.mark.parametrize("bad", [b"", b"not SRT", b"1\n-00:00:10,000 --> 00:00:11,000\nDialogue.",
    b"1\n00:00:60,000 --> 00:01:01,000\nDialogue.", b"1\n00:00:10,000 --> 00:00:09,000\nDialogue.",
    b"1\n00:00:10,000 --> 00:00:11,000\nDialogue.\n\nmalformed trailing block"])
def test_invalid_srt_rejected_without_admission(sandbox, bad):
    root, inputs = sandbox
    inputs["target"].write_bytes(bad)
    with pytest.raises(ValueError):
        perform(sandbox)
    assert_not_admitted(root)


def test_nonmonotonic_cues_are_not_sorted_into_validity(sandbox):
    root, inputs = sandbox
    inputs["target"].write_text(render_srt([(20000, 22000, "Later."), (10000, 11000, "Earlier.")]))
    with pytest.raises(ValueError, match="ordering"):
        perform(sandbox)
    assert_not_admitted(root)


def test_unrelated_transformed_dialogue_is_refused(sandbox):
    root, inputs = sandbox
    inputs["transformed"].write_text(inputs["transformed"].read_text().replace("event 1", "unrelated content"))
    with pytest.raises(ValueError, match="dialogue population"):
        perform(sandbox)
    assert_not_admitted(root)


def test_missing_expected_outcome_is_refused(sandbox):
    root, inputs = sandbox
    with pytest.raises(ValueError, match="outcome required"):
        im.import_case(root=root, case_id="missing-expectation", inputs=inputs,
                       case_type="GLOBAL_AFFINE", expected_boundary_accepted=None)
    assert_not_admitted(root)


@pytest.mark.parametrize("metadata", [
    {"target_release_family": "C:/private/target.srt"},
    {"target_release_family": "/home/user/target.srt"},
    {"target_release_family": "https://provider.invalid?token=PRIVATE"},
    {"target_release_family": "api_key=PRIVATE"},
    {"target_release_family": "eyJhbGciOiJIUzI1NiJ9.payload.signature"},
    {"target_release_family": "a" * 64}, {"api_key": "PRIVATE"},
    {"media_type": "request-config"}, {"season": True},
])
def test_secret_or_path_metadata_is_refused(sandbox, metadata):
    root, _ = sandbox
    with pytest.raises(ValueError):
        perform(sandbox, metadata=metadata)
    assert_not_admitted(root)


def test_sensitive_subtitle_payload_is_refused(sandbox):
    root, inputs = sandbox
    inputs["target"].write_text(inputs["target"].read_text().replace("Dialogue event 0.", "api\u200d_key=PRIVATE"), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe"):
        perform(sandbox)
    assert_not_admitted(root)


@pytest.mark.parametrize("case_id", ["../escape", "Unsafe Case", "con", "case.", "C:/private", "sk-" + "a" * 20])
def test_unsafe_case_ids_cannot_choose_directories(sandbox, case_id):
    root, inputs = sandbox
    with pytest.raises(ValueError, match="case_id"):
        im.import_case(root=root, case_id=case_id, inputs=inputs,
                       case_type="INSUFFICIENT_EVIDENCE_REJECT", expected_boundary_accepted=False)
    assert_not_admitted(root)


def test_identical_reference_requires_explicit_opt_in(sandbox):
    root, inputs = sandbox
    inputs["reference"] = inputs["target"]
    with pytest.raises(ValueError, match="identical target/reference"):
        perform(sandbox)
    assert_not_admitted(root)
    assert perform(sandbox, allow_identical_target_reference=True)["passed"]


def test_repeat_import_does_not_overwrite_case_or_manifest(sandbox):
    root, _ = sandbox
    perform(sandbox)
    manifest = (root / "tests/replay/corpus.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        perform(sandbox)
    assert (root / "tests/replay/corpus.json").read_bytes() == manifest


def test_replay_mismatch_refuses_admission(sandbox):
    root, inputs = sandbox
    with pytest.raises(ValueError, match="replay expectation mismatch"):
        im.import_case(root=root, case_id="fake-edge-success", inputs=inputs,
                       case_type="EDGE_DISCONTINUITY", expected_boundary_accepted=True)
    assert_not_admitted(root)


def test_atomic_manifest_failure_rolls_back_only_new_case(sandbox, monkeypatch):
    root, _ = sandbox
    def fail(*args):
        raise OSError("test replacement failure")
    monkeypatch.setattr(im.os, "replace", fail)
    with pytest.raises(OSError):
        perform(sandbox)
    assert_not_admitted(root)


def test_active_import_lock_is_preserved(sandbox):
    root, _ = sandbox
    lock = root / "tests/replay/.case-import.lock"
    lock.write_text("another owner")
    with pytest.raises(ValueError, match="owns the corpus lock"):
        perform(sandbox)
    assert lock.read_text() == "another owner"


def test_deterministic_paths_and_case_json_across_roots(tmp_path):
    # Case metadata never includes temporary absolute roots or source filenames.
    results = []
    for name in ("first", "second"):
        root, inputs = sandbox.__wrapped__(tmp_path / name)
        if name == "second":
            inputs = dict(reversed(list(inputs.items())))
        perform((root, inputs), provenance="synthetic")
        results.append((root / "tests/replay/cases/local-sparse-case/case.json").read_bytes())
    assert results[0] == results[1]


def test_provenance_breakdown_and_summary(sandbox):
    root, inputs = sandbox
    perform(sandbox)
    for case_id, provenance, source in (("derived-case", "derived", "local-sparse-case"),
                                         ("synthetic-case", "synthetic", None)):
        im.import_case(root=root, case_id=case_id, inputs=inputs,
            case_type="INSUFFICIENT_EVIDENCE_REJECT", expected_boundary_accepted=False,
            provenance=provenance, source_case_id=source)
    report = replay(root / "tests/replay/corpus.json", root)
    for group in report["aggregate_by_provenance"].values():
        assert group["expected_negatives"] == group["true_negatives"] == 1
        assert group["false_positives"] == 0
    summary = corpus_summary(report)
    assert "real_positive_cases=0 real_negative_cases=1 derived_cases=1 synthetic_cases=1" in summary
    assert summary.count("PASS") == 3
    assert all(kind in summary for kind in ("REAL", "DERIVED", "SYNTHETIC"))
    # Make the expected boolean wrong; summary must display the contract, not actual.
    case = json.loads((root / "tests/replay/corpus.json").read_text())["cases"][0]
    case["expectations"]["expected_parent_model_failure"] = False
    assert "expected_parent_model_failure" in case_mismatches(case, report["cases"][0]["evidence"])


def test_cli_errors_do_not_echo_private_file_paths(sandbox, monkeypatch, capsys):
    _, inputs = sandbox
    monkeypatch.setattr("sys.argv", ["import_sync_case", "--case-id", "safe-case", "--target", str(inputs["target"]),
        "--reference", str(inputs["reference"]), "--transformed", str(inputs["transformed"]),
        "--expected-model", "INSUFFICIENT_EVIDENCE_REJECT", "--expected-boundary-accepted", "false"])
    monkeypatch.setattr(im, "import_case", lambda **kwargs: (_ for _ in ()).throw(OSError("PRIVATE_PATH")))
    assert im.main() == 1
    assert "PRIVATE_PATH" not in capsys.readouterr().out


def test_manifest_edit_during_replay_is_preserved_and_case_refused(sandbox, monkeypatch):
    root, _ = sandbox
    original = im.replay_case
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        (root / "tests/replay/corpus.json").write_text('{"schema_version": 2, "cases": []}')
        return result
    monkeypatch.setattr(im, "replay_case", changed)
    with pytest.raises(ValueError, match="manifest changed"):
        perform(sandbox)
    assert json.loads((root / "tests/replay/corpus.json").read_text())["schema_version"] == 2
    assert_not_admitted(root)


def test_raw_bytes_and_safe_output_names_are_preserved(sandbox):
    root, inputs = sandbox
    original = inputs["target"].read_bytes()
    private_filename = inputs["target"].with_name("PRIVATE-source-name.srt")
    private_filename.write_bytes(original)
    inputs["target"] = private_filename
    report = perform(sandbox)
    folder = root / "tests/replay/cases/local-sparse-case"
    assert (folder / "target.srt").read_bytes() == original
    assert report["artifacts"]["target"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert "PRIVATE-source-name" not in (folder / "case.json").read_text()
    assert "PRIVATE-source-name" not in (folder / "diagnostic.json").read_text()


@pytest.mark.parametrize("kwargs", [
    {"provenance": "unknown"}, {"provenance": "derived"},
    {"provenance": "derived", "source_case_id": "unknown-source"},
    {"source_case_id": "real-cannot-be-derived"},
    {"expected_parent_model": "unsupported-model"},
    {"expected_boundary_run": "false"},
])
def test_invalid_provenance_or_expectations_are_refused(sandbox, kwargs):
    root, _ = sandbox
    with pytest.raises(ValueError):
        perform(sandbox, **kwargs)
    assert_not_admitted(root)


def test_summary_cli_needs_no_output_file(sandbox, monkeypatch, capsys):
    from tools import sync_replay
    root, _ = sandbox
    perform(sandbox)
    report = replay(root / "tests/replay/corpus.json", root)
    monkeypatch.setattr(sync_replay, "replay", lambda *args: report)
    monkeypatch.setattr("sys.argv", ["sync_replay", "--corpus-summary"])
    assert sync_replay.main() == 0
    output = capsys.readouterr().out
    assert "local-sparse-case" in output and "REAL" in output and "PASS" in output
    report["expectation_mismatches"] = ["local-sparse-case"]
    report["cases"][0]["evidence"]["expected_boundary_accepted"] = True
    assert sync_replay.main() == 1
    output = capsys.readouterr().out
    assert "True | False" in output and "FAIL" in output


def test_reused_artifact_mutation_cannot_misreport_identity(sandbox, monkeypatch):
    root, inputs = sandbox
    folder = root / "tests/fixtures"
    folder.mkdir()
    reused = folder / "existing-target.srt"
    reused.write_bytes(inputs["target"].read_bytes())
    original = im.replay_case
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        reused.write_text("changed externally")
        return result
    monkeypatch.setattr(im, "replay_case", mutate)
    with pytest.raises(ValueError, match="artifact changed"):
        perform(sandbox)
    assert_not_admitted(root)
    assert reused.read_text() == "changed externally"
