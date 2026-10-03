"""Property tests for the golden ground-truth evaluation harness.

The harness is only worth anything if its answer key is genuinely independent
of the system it grades. Most of these tests exist to defend that property,
and to prove the harness fails loudly rather than quietly producing numbers.
"""

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.services.sync.golden import (
    GOLDEN_ENGINE_VERSION,
    AlignmentSimulation,
    GoldenCaseError,
    GroundTruthContent,
    GroundTruthCut,
    GroundTruthFinal,
    GroundTruthSource,
    GroundTruthSync,
    load_manifest,
)

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "tests" / "fixtures" / "golden_sync" / "manifest.json"
GOLDEN_MODULE = REPO / "app" / "services" / "sync" / "golden.py"
EVALUATOR = REPO / "tools" / "evaluate_golden_sync.py"



def load_evaluator():
    """Import the evaluator by path: it is a script, not an installed module."""
    spec = importlib.util.spec_from_file_location("golden_evaluator", EVALUATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cache_file_count(root: Path) -> int:
    return len([p for p in root.rglob("*") if p.is_file()]) if root.exists() else 0


#: The golden corpus subtitle payloads are generated artifacts and stay gitignored
#: (see ``.gitignore``). Every case in the manifest points at one, so a fresh clone
#: has no corpus at all and these tests have nothing to measure. They must skip
#: rather than fail: a red suite on every clean checkout reads as a real
#: regression and trains people to ignore it.
requires_golden_corpus = pytest.mark.skipif(
    _cache_file_count(REPO / "tests" / "fixtures" / "golden_sync" / "subtitles") == 0,
    reason="golden-corpus subtitles are absent (generated, gitignored)",
)


def run_evaluator(*args: str) -> subprocess.CompletedProcess:
    """Run the evaluator in a subprocess with a real environment."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    return subprocess.run(
        [sys.executable, str(EVALUATOR), *args],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=env,
    )


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(MANIFEST)


# --- 1. the answer key is independent of the system under test ------------ #

FORBIDDEN_IN_GOLDEN = (
    "alignment",
    "predictor",
    "reference_v2",
    "reference",
    "subtitle_matcher",
    "external_strategy",
    "cache",
    "ordering",
)


def test_ground_truth_module_imports_nothing_from_the_system_under_test():
    """The schema must not be able to consult the thing it grades."""
    tree = ast.parse(GOLDEN_MODULE.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    app_imports = [name for name in imported if name.startswith("app.")]
    for name in app_imports:
        assert name == "app.services.sync.golden", f"golden.py must not import {name}"
    for forbidden in FORBIDDEN_IN_GOLDEN:
        assert not any(forbidden in name for name in app_imports), (
            f"golden.py imports {forbidden}, which is under evaluation"
        )


def test_ground_truth_labels_are_authored_verbatim_not_derived():
    """Every label in the manifest is present in the file as a literal.

    If the loader computed any field, this comparison would diverge.
    """
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest = load_manifest(MANIFEST)
    for raw_case, case in zip(raw["cases"], manifest.cases, strict=True):
        assert case.case_id == raw_case["case_id"]
        for raw_truth, truth in zip(raw_case["ground_truth"], case.ground_truth, strict=True):
            assert truth.content.value == raw_truth["content"]
            assert truth.cut.value == raw_truth["cut"]
            assert truth.original_sync.value == raw_truth["original_sync"]
            assert truth.final_alignment.value == raw_truth["final_alignment"]
            assert truth.resyncable == raw_truth["resyncable"]
            # The declared alignment outcome is authored, not computed.
            assert truth.alignment.mode.value == raw_truth["alignment"]["mode"]


def test_dataset_contains_no_label_the_system_produced():
    """Guard against a future 'just let the tool fill it in' shortcut."""
    source = (REPO / "tools" / "generate_golden_fixtures.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"AlignmentAnalyzer", "SyncPredictor", "ReferenceTrust", "analyze", "sync_state"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in forbidden:
            raise AssertionError(
                f"the generator consults {node.attr}, which would make labels circular"
            )
        if isinstance(node, ast.Name) and node.id in forbidden:
            raise AssertionError(f"the generator references {node.id}")


# --- 2. unknown ground truth is excluded from precision -------------------- #


def test_unknown_ground_truth_is_excluded_from_precision():
    evaluator = load_evaluator()

    from app.services.sync.alignment import SyncState
    from app.services.sync.golden import CandidateTruth

    def _result(state: SyncState, final: GroundTruthFinal) -> object:
        return evaluator.CaseResult(
            case_id="c",
            failure_mode="m",
            release_class="r",
            annotation_source="objective",
            provider="p",
            release_name="n",
            truth=CandidateTruth(final_alignment=final),
            system_state=state.value,
            system_verified=state
            in (SyncState.VERIFIED_SYNCED, SyncState.VERIFIED_RESYNCED),
            system_rejected=state is SyncState.REJECTED,
            system_predicted_verified=state
            in (SyncState.VERIFIED_SYNCED, SyncState.VERIFIED_RESYNCED),
            reference_trust=None,
            error_class=evaluator.classify_outcome(
                state in (SyncState.VERIFIED_SYNCED, SyncState.VERIFIED_RESYNCED),
                state is SyncState.REJECTED,
                final,
            ),
            annotated=final.value,
        )

    mixed = [
        _result(SyncState.VERIFIED_RESYNCED, GroundTruthFinal.CORRECT),
        _result(SyncState.VERIFIED_RESYNCED, GroundTruthFinal.UNKNOWN),
    ]
    counted = [r for r in mixed if r.error_class != "UNLABELLED"]
    # The unknown one is not evidence either way.
    assert len(counted) == 1
    assert counted[0].annotated == "correct"

    # And an unknown label is never counted as a false positive.
    wrong = [_result(SyncState.VERIFIED_RESYNCED, GroundTruthFinal.UNKNOWN)]
    report = evaluator.build_report(
        type(manifest_stub())(cases=[]), wrong, {"git_commit": "x", "tool": "t"}
    )
    assert report["false_positives"]["false_verified"] == 0


def manifest_stub():
    from app.services.sync.golden import GoldenManifest

    return GoldenManifest()


# --- 3. sample-size gates -------------------------------------------------- #


def test_rates_are_withheld_below_the_minimum_sample():
    evaluator = load_evaluator()

    thin = evaluator.rate(3, 3)
    assert thin["rate"] is None
    assert thin["status"] == "INSUFFICIENT_SAMPLE"
    solid = evaluator.rate(9, 10)
    assert solid["rate"] == 0.9
    assert solid["wilson_low"] is not None and solid["wilson_high"] is not None
    assert solid["wilson_low"] < 0.9 < solid["wilson_high"]


def test_three_successful_cases_do_not_report_100_percent_accuracy():
    """A perfect 3/3 run must not be presented as evidence."""
    evaluator = load_evaluator()
    value = evaluator.rate(3, 3)
    assert value["rate"] is None
    assert value["status"] == "INSUFFICIENT_SAMPLE"
    # 3/3 is genuine but unpublishable as a rate.
    assert evaluator.rate(3, 3)["count"] == 3


def test_wilson_interval_is_bounded_and_widens_on_small_samples():
    evaluator = load_evaluator()
    small = evaluator.wilson(8, 10)
    large = evaluator.wilson(800, 1000)
    assert small is not None and large is not None
    assert 0.0 <= small[0] <= small[1] <= 1.0
    width_small = small[1] - small[0]
    width_large = large[1] - large[0]
    assert width_small > width_large


# --- 4/5. versions are recorded and not silently combined ------------------ #


def test_dataset_and_engine_versions_are_recorded(manifest):
    assert manifest.dataset_version == "v1"
    assert manifest.engine_version == GOLDEN_ENGINE_VERSION
    assert manifest.schema_version == 1


def test_provenance_records_commit_and_versions(manifest):
    evaluator = load_evaluator()
    prov = evaluator.provenance(manifest)
    assert prov["dataset_version"] == "v1"
    assert prov["harness_engine_version"] == GOLDEN_ENGINE_VERSION
    assert prov["git_commit"]


def test_engine_version_mismatch_is_reported_not_hidden(tmp_path, manifest):
    """A dataset authored for another engine must be labelled, not merged."""
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    raw["engine_version"] = GOLDEN_ENGINE_VERSION + 7
    other = tmp_path / "other.json"
    other.write_text(json.dumps(raw), encoding="utf-8")

    result = run_evaluator(str(other))
    assert result.returncode == 0
    assert "authored for engine" in result.stderr
    assert "not merged" in result.stderr
    # The report still states which engine it targeted.
    assert f"Dataset engine   : {GOLDEN_ENGINE_VERSION + 7}" in result.stdout
    assert f"Harness engine   : {GOLDEN_ENGINE_VERSION}" in result.stdout


# --- 6. missing fingerprints are not fabricated ---------------------------- #


def test_missing_fingerprint_case_keeps_its_missing_filename(manifest):
    case = manifest.case("missing_fingerprint")
    assert case.target.filename is None
    assert case.target.fingerprint_complete is False


def test_evaluator_does_not_invent_a_target_filename(manifest):
    evaluator = load_evaluator()
    from app.services.sync.alignment import AlignmentAnalyzer

    results = evaluator.evaluate_case(
        manifest.case("missing_fingerprint"), MANIFEST, AlignmentAnalyzer()
    )
    assert results
    for result in results:
        # The note records the filename exactly as authored, i.e. None.
        assert "target_name=None" in result.notes
        assert "target_name=" in " ".join(result.notes)


# --- 7. evaluation is side-effect free ------------------------------------- #


def test_evaluation_does_not_touch_production_caches():
    """Running the benchmark must not populate subs_cache or the audit log."""
    from app.services.sync.audit import AUDIT_LOG

    before_audit = len(AUDIT_LOG.records())
    subs_cache = REPO / "subs_cache"
    before_cache = _cache_file_count(subs_cache)

    run_evaluator(str(MANIFEST)).check_returncode()

    after_audit = len(AUDIT_LOG.records())
    after_cache = _cache_file_count(subs_cache)
    assert after_audit == before_audit
    assert after_cache == before_cache


def test_benchmark_does_not_change_any_threshold():
    """The evaluator must be read-only with respect to production constants."""
    from app.services.sync import alignment

    before = {
        name: getattr(alignment, name)
        for name in dir(alignment)
        if name.isupper() and isinstance(getattr(alignment, name), (int, float))
    }
    run_evaluator(str(MANIFEST)).check_returncode()
    after = {
        name: getattr(alignment, name)
        for name in dir(alignment)
        if name.isupper() and isinstance(getattr(alignment, name), (int, float))
    }
    assert before == after
    assert alignment.MIN_CUES_FOR_VERIFIED == 12


# --- 8. malformed manifests fail loudly ------------------------------------ #


def test_missing_manifest_fails_loudly(tmp_path):
    with pytest.raises(GoldenCaseError, match="manifest not found"):
        load_manifest(tmp_path / "nope.json")


def test_invalid_json_fails_loudly(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(GoldenCaseError, match="not valid JSON"):
        load_manifest(bad)


def test_empty_dataset_fails_loudly(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"cases": []}), encoding="utf-8")
    with pytest.raises(GoldenCaseError, match="no cases"):
        load_manifest(empty)


def test_candidate_without_ground_truth_fails_loudly(tmp_path):
    broken = {
        "cases": [
            {
                "case_id": "c1",
                "target": {"path": "subtitles/x.srt"},
                "candidates": [
                    {"provider": "subdl", "release_name": "a.srt", "path": "subtitles/a.srt"}
                ],
                "ground_truth": [
                    {
                        "provider": "subdl",
                        "release_name": "other.srt",
                        "content": "same_release",
                        "cut": "same_cut",
                        "original_sync": "already_synced",
                        "final_alignment": "correct",
                    }
                ],
            }
        ]
    }
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(GoldenCaseError, match="has no ground truth"):
        load_manifest(path)


def test_duplicate_case_ids_fail_loudly(tmp_path, manifest):
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    raw["cases"] = [raw["cases"][0], raw["cases"][0]]
    path = tmp_path / "dup.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(GoldenCaseError, match="duplicate case_id"):
        load_manifest(path)


def test_evaluator_cli_rejects_a_missing_manifest():
    result = run_evaluator("does_not_exist.json")
    assert result.returncode == 2
    assert "not found" in result.stderr


# --- 9. the dataset itself is fit for purpose ----------------------------- #


def test_dataset_covers_the_required_failure_modes(manifest):
    modes = {case.failure_mode for case in manifest.cases}
    required = {
        "already_synced",
        "offset",
        "drift",
        "offset_different_release",
        "different_cut",
        "wrong_episode",
        "credits_only",
        "cue_collapse",
        "sparse_dialogue",
        "malformed",
        "duplicate_grid",
        "circular",
        "vote_sub",
        "resplit",
        "missing_fingerprint",
    }
    assert required <= modes, f"missing failure modes: {required - modes}"


def test_dataset_covers_the_required_release_classes(manifest):
    classes = {case.release_class for case in manifest.cases}
    assert {"bluray", "webdl", "webrip", "hdtv"} <= classes


def test_dataset_contains_both_positive_and_negative_cases(manifest):
    finals = {t.final_alignment for case in manifest.cases for t in case.ground_truth}
    assert GroundTruthFinal.CORRECT in finals
    assert GroundTruthFinal.INCORRECT in finals
    contents = {t.content for case in manifest.cases for t in case.ground_truth}
    assert GroundTruthContent.SAME_RELEASE in contents
    assert GroundTruthContent.DIFFERENT_CONTENT in contents
    cuts = {t.cut for case in manifest.cases for t in case.ground_truth}
    assert GroundTruthCut.DIFFERENT_CUT in cuts
    syncs = {t.original_sync for case in manifest.cases for t in case.ground_truth}
    assert GroundTruthSync.ALREADY_SYNCED in syncs
    assert GroundTruthSync.RESYNCABLE in syncs


def test_hard_negatives_look_plausible_by_metadata(manifest):
    """The negatives must not be rejectable on filename alone."""
    cut = manifest.case("same_release_different_cut")
    truth = cut.ground_truth[0]
    assert truth.content is GroundTruthContent.SAME_RELEASE
    assert truth.final_alignment is GroundTruthFinal.INCORRECT
    # Same group as the target video, so only timing can distinguish them.
    assert "PiR8" in cut.target.filename
    assert "PiR8" in truth.release_name


def test_circular_case_is_self_consistent_and_wrong(manifest):
    case = manifest.case("circular_shared_wrong_timing")
    assert len(case.candidates) == 2
    assert len({c.provider for c in case.candidates}) == 2
    for truth in case.ground_truth:
        assert truth.final_alignment is GroundTruthFinal.INCORRECT
        assert truth.cut is GroundTruthCut.DIFFERENT_CUT
        # Perfectly aligned: the error is not in the residuals.
        assert truth.alignment.mode is AlignmentSimulation.AS_TARGET
        assert truth.alignment.residual_ms == 0


def test_valid_reference_set_case_does_not_force_one_reference(manifest):
    case = manifest.case("duplicate_timing_grid")
    assert case.is_valid_reference_set
    assert len(case.valid_reference_keys) >= 2


def test_annotation_source_is_recorded(manifest):
    sources = {case.annotation_source for case in manifest.cases}
    assert sources <= {GroundTruthSource.OBJECTIVE, GroundTruthSource.HUMAN_REVIEWED}
    assert GroundTruthSource.OBJECTIVE in sources


# --- 10. the headline result is reproducible ------------------------------- #


@requires_golden_corpus
def test_benchmark_detects_the_circular_error_case():
    """Self-consistency must not be mistaken for correctness."""
    evaluator = load_evaluator()
    from app.services.sync.alignment import AlignmentAnalyzer

    manifest = load_manifest(MANIFEST)
    results = evaluator.evaluate_case(
        manifest.case("circular_shared_wrong_timing"), MANIFEST, AlignmentAnalyzer()
    )
    assert results
    for result in results:
        # The system is internally confident and perfectly aligned...
        assert result.median_offset_ms == 0.0
        assert result.system_verified is True
        # ...and ground truth still says the answer is wrong.
        assert result.annotated == "incorrect"
        assert result.error_class == "FALSE_VERIFIED"


def test_benchmark_reports_no_recommendation_to_change_thresholds():
    result = run_evaluator(str(MANIFEST))
    result.check_returncode()
    assert "INVESTIGATION CANDIDATES" in result.stdout
    assert "It measures." in result.stdout
    for forbidden in ("recommended threshold", "should change", "optimal threshold"):
        assert forbidden not in result.stdout.lower()
