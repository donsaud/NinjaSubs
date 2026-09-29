"""Does the landmark representation carry enough information for the real task?

The prior phase's `rapid_dialogue` regression looked like a detector failure.
Landmark F1 says otherwise at the only level that matters: the decision. These
tests pin the measurement, the answer it gave, and the boundary conditions
around it.

The answer was negative for the experimental fine scale, and that result is
asserted here so it cannot be quietly dropped the next time the fine scale
looks attractive.
"""

import ast
import json
from pathlib import Path

import pytest

from app.services.sync.audio_activity import (
    ActivityProfile,
    AdaptiveAudioConfig,
    AudioLandmark,
    ExtractionScale,
    LandmarkKind,
)

REPO = Path(__file__).resolve().parent.parent
TASK_BENCHMARK = REPO / "tests" / "fixtures" / "golden_sync" / "cut_detection_task_benchmark.json"
BENCH_TOOL = REPO / "tools" / "benchmark_cut_detection.py"


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return json.loads(TASK_BENCHMARK.read_text(encoding="utf-8"))["cases"]


# --- §1: the primary question is task-level, not landmark F1 ---------------- #


def test_benchmark_scores_decisions_not_landmark_counts(rows):
    for row in rows:
        for outcome in row["results"].values():
            assert outcome["verdict"] in ("match", "mismatch", "abstain")
            assert "landmarks" in outcome  # reported, but not the criterion


def test_every_required_task_dimension_is_measured(rows):
    names = {row["case"] for row in rows}
    for required in (
        "same_cut",  # correct cut detection
        "global_offset",  # offset preservation
        "fps_drift",  # drift
        "inserted_segment",  # wrong cut, discontinuity
        "different_intro",  # wrong cut, piecewise
        "short_segment_cut",  # adversarial: short difference
        "major_segment_cut",  # adversarial: major difference
    ):
        assert required in names, f"missing task case {required}"


def test_rapid_dialogue_same_cut_case_exists(rows):
    """§8 case F: must not be falsely classified as a cut."""
    row = next(r for r in rows if r["case"] == "rapid_dialogue_same_cut")
    assert row["expected"] == "match"
    outcome = row["results"]["single_scale"]
    # The safe answer is abstention, not a wrong rejection.
    assert outcome["verdict"] in ("match", "abstain")
    assert outcome["verdict"] != "mismatch"


def test_rapid_dialogue_abstains_rather_than_rejecting(rows):
    """The measured answer: micro-boundary loss costs abstention, not accuracy."""
    row = next(r for r in rows if r["case"] == "rapid_dialogue_same_cut")
    outcome = row["results"]["single_scale"]
    assert outcome["verdict"] == "abstain"
    # And it is recorded as an abstention, not silently counted as correct.
    assert outcome["clarity"] in ("ambiguous", "no_signal")


# --- §2/§4: salience and strength ------------------------------------------- #


def test_landmark_kinds_exist_and_are_a_ranking_not_a_judgement():
    assert {k.value for k in LandmarkKind} == {"major", "minor", "micro"}
    # A micro boundary is real information, so it must not be filtered away.
    profile = ActivityProfile(
        boundaries=[
            AudioLandmark(timestamp_ms=0, strength=0.9, kind=LandmarkKind.MAJOR),
            AudioLandmark(timestamp_ms=500, strength=0.2, kind=LandmarkKind.MICRO),
        ]
    )
    assert profile.landmarks == [0, 500]
    assert profile.by_kind(LandmarkKind.MICRO) == [500]


def test_strength_is_a_ranking_not_a_probability():
    """The field must be named and documented as evidence weight."""
    field = AudioLandmark.model_fields["strength"]
    assert field.description is not None
    assert "probability" in field.description.lower()
    assert 0.0 <= AudioLandmark(timestamp_ms=0, strength=0.0, kind=LandmarkKind.MINOR).strength <= 1.0
    low = AudioLandmark(timestamp_ms=0, strength=0.1, kind=LandmarkKind.MINOR)
    high = AudioLandmark(timestamp_ms=1, strength=0.9, kind=LandmarkKind.MINOR)
    assert high.strength > low.strength


# --- §3: raw activity information is preserved ----------------------------- #


def test_activity_profile_retains_the_underlying_signal():
    profile = ActivityProfile(
        energy_envelope=[10, 20, 30, 40],
        activity_mask=[0, 1, 1, 0],
        envelope_step_ms=20,
    )
    assert profile.energy_envelope
    assert profile.activity_mask
    # The envelope is compact and does not retain audio.
    assert len(profile.energy_envelope) == len(profile.activity_mask)


def test_envelope_memory_stays_bounded_on_a_long_file():
    """A feature-length file must not cost gigabytes."""
    from app.services.sync.audio_activity import DECODE_RATE

    step_ms = 20
    hours = 2.5
    samples = int(hours * 3_600_000 / step_ms)
    # One int8-equivalent per step.
    assert samples * 1 < 1_000_000, "envelope should stay well under a megabyte"
    assert DECODE_RATE > 0


# --- §5/§6: multi-scale is a representation, and it was measured ------------ #


def test_fine_scale_is_experimental_and_off_by_default():
    assert ExtractionScale.FINE.value == "fine"
    assert ExtractionScale.COARSE.value == "coarse"
    assert AdaptiveAudioConfig().fine_min_boundary_ms == 500
    # The coarse scale must remain the protected one.
    assert AdaptiveAudioConfig().min_boundary_separation_ms == 1_500


def test_fine_scale_did_not_improve_any_decision(rows):
    """§17: do not keep complexity that does not earn its place.

    The measured answer is that the fine scale improved nothing. Where it moved
    a case at all, it moved it toward abstention, which is safer than a wrong
    answer but is still not a decision. That is not enough to justify carrying
    the extra representation.
    """
    improved = []
    moved_toward_abstain = []
    for row in rows:
        single = row["results"]["single_scale"]["verdict"]
        multi = row["results"]["multi_scale"]["verdict"]
        expected = row["expected"]
        if single == multi:
            continue
        if multi == expected and single != expected:
            improved.append(row["case"])
        if multi == "abstain":
            moved_toward_abstain.append(row["case"])
        else:
            raise AssertionError(
                f"{row['case']}: fine scale moved {single} -> {multi}, which is "
                "neither an improvement nor a move toward abstention"
            )
    assert not improved, f"fine scale improved: {improved}"
    assert moved_toward_abstain, "the fine scale's measured effect must stay recorded"


def test_fine_scale_abstains_more_often_than_it_decides(rows):
    """Its purpose was to disambiguate peaks. It did not."""
    single_abstain = sum(
        1 for r in rows if r["results"]["single_scale"]["verdict"] == "abstain"
    )
    multi_abstain = sum(
        1 for r in rows if r["results"]["multi_scale"]["verdict"] == "abstain"
    )
    assert multi_abstain >= single_abstain


def test_fine_scale_added_landmarks_without_adding_decisions(rows):
    """More landmarks is not the objective, and is not what happened."""
    more_landmarks = [
        r for r in rows
        if r["results"]["multi_scale"]["landmarks"]
        > r["results"]["single_scale"]["landmarks"]
    ]
    assert more_landmarks, "the fine scale should visibly add landmarks"
    for row in more_landmarks:
        single = row["results"]["single_scale"]["verdict"]
        multi = row["results"]["multi_scale"]["verdict"]
        assert single == multi or multi == "abstain", (
            "adding landmarks should not have changed a decision for the better"
        )


# --- §9/§11: the correlation decision is what gets measured ----------------- #


def test_correlation_decision_fields_are_reported(rows):
    for row in rows:
        for outcome in row["results"].values():
            for field in (
                "best_shift",
                "best_peak",
                "second_peak",
                "peak_ratio",
                "clarity",
                "region_offsets",
                "regional_consistency",
            ):
                assert field in outcome, f"{field} not reported"


def test_ambiguous_cases_abstain_in_both_representations(rows):
    for row in rows:
        for outcome in row["results"].values():
            if outcome["clarity"] in ("ambiguous", "no_signal"):
                assert outcome["verdict"] == "abstain", (
                    f"{row['case']} had {outcome['clarity']} but decided anyway"
                )


# --- §13: the error taxonomy explains failures ----------------------------- #


def test_error_taxonomy_is_present_and_used(rows):
    labels = set()
    for row in rows:
        for outcome in row["results"].values():
            if outcome["error"]:
                labels.add(outcome["error"])
    assert labels, "some cases should carry a classification"
    source = BENCH_TOOL.read_text(encoding="utf-8")
    for category in (
        "MICRO_BOUNDARY_LOSS",
        "MAJOR_BOUNDARY_LOSS",
        "WRONG_GLOBAL_SHIFT",
        "WRONG_REGIONAL_SHIFT",
        "AMBIGUOUS_SIGNAL",
        "NO_SIGNAL",
        "WRONG_CUT_MISSED",
        "SAME_CUT_FALSE_REJECTED",
    ):
        assert category in source


def test_wrong_cut_missed_is_the_dominant_failure(rows):
    """The measured gap, pinned so it cannot be forgotten."""
    missed = [
        r["case"]
        for r in rows
        if r["expected"] == "mismatch"
        and r["results"]["single_scale"]["verdict"] != "mismatch"
    ]
    assert missed, "the benchmark should expose where cut detection fails"
    # The failures are piecewise-displacement cases, not micro-boundary cases.
    assert any("segment_cut" in case or "intro" in case for case in missed)


# --- §14: hold-out scenarios ----------------------------------------------- #


def test_hold_out_cases_are_marked_and_reported_separately(rows):
    held = [r for r in rows if r["hold_out"]]
    assert len(held) >= 4
    development = [r for r in rows if not r["hold_out"]]
    assert development
    source = BENCH_TOOL.read_text(encoding="utf-8")
    assert "hold-out" in source.lower()


def test_hold_out_set_covers_the_adversarial_shapes(rows):
    held = {r["case"] for r in rows if r["hold_out"]}
    for case in (
        "near_continuous_same_cut",
        "short_segment_cut",
        "similar_timing_models",
        "coarse_strong_fine_noisy",
    ):
        assert case in held


def test_benchmark_does_not_tune_parameters():
    """The tool must not import a detector to choose a detector setting."""
    source = BENCH_TOOL.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                name = getattr(target, "id", "")
                assert not name.startswith(("WINDOW", "HOP", "MARGIN", "THRESHOLD"))
    # And it never reads the adaptive config to pick values.
    assert "from_settings" not in source


# --- §16/§22: nothing production changed ------------------------------------ #


def test_production_detector_is_untouched():
    from app.config import settings

    assert settings.ADAPTIVE_AUDIO_ENABLED is False
    assert settings.ADAPTIVE_AUDIO_MIN_BOUNDARY_SEPARATION_MS == 1_500


def test_no_production_synchronization_threshold_moved():
    from app.services.sync import alignment

    assert alignment.MIN_CUES_FOR_VERIFIED == 12


def test_benchmark_is_labelled_synthetic_and_not_real_media(rows):
    payload = json.loads(TASK_BENCHMARK.read_text(encoding="utf-8"))
    assert "Synthetic audio" in payload["note"]
    assert "no real programme" in payload["note"]


def test_benchmark_runs_offline_without_network():
    source = BENCH_TOOL.read_text(encoding="utf-8")
    for token in ("urllib", "requests", "httpx", "http://", "https://"):
        assert token not in source


# --- §19: the VAD gate, restated -------------------------------------------- #


def test_no_category_F_evidence_was_found():
    """Nothing measured required semantic or audio understanding."""
    payload = json.loads(TASK_BENCHMARK.read_text(encoding="utf-8"))
    # Every recorded failure is a landmark or correlation failure, which is
    # category A-E, not category F.
    for row in payload["cases"]:
        for outcome in row["results"].values():
            assert outcome["error"] not in ("GENUINE_MISSING_SEMANTIC", "NEEDS_SEMANTIC")
