"""Real-media benchmark harness, and the assumptions the stress run exposed.

The stress run against hostile audio found something the generated-media tests
could not: the current amplitude floor recovers boundaries only from audio
containing true digital silence. Room tone, crowd noise and a music bed all
yield zero landmarks. The failure is safe - the validator abstains - but it
means the signal's real-world reach is far narrower than the generated-media
result suggested.

These tests pin the safe behaviour, the audio-track selection rules, and the
harness itself. They do not tune anything.
"""

import json
import subprocess
import sys
from pathlib import Path

from app.services.sync.golden import GroundTruthFinal, load_real_media_manifest
from app.services.sync.video_timeline import (
    VideoTimelineProfile,
    VideoVerdict,
    is_withholding_evidence,
    select_audio_stream,
    validate_timeline_against_reference,
)

REPO = Path(__file__).resolve().parent.parent
SAMPLE = REPO / "tests" / "fixtures" / "golden_sync" / "real_media_sample.redacted.json"
STRESS = REPO / "tests" / "fixtures" / "golden_sync" / "audio_stress_proxy.json"
EVALUATOR = REPO / "tools" / "evaluate_real_media.py"
STRESS_TOOL = REPO / "tools" / "stress_audio_landmarks.py"


def _run(script: Path, *args: str) -> subprocess.CompletedProcess:
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    return subprocess.run(
        [sys.executable, str(script), *args], capture_output=True, text=True, cwd=REPO, env=env
    )


# --- the measurement artifact ------------------------------------------------ #


def test_stress_artifact_is_labelled_as_proxy_data():
    payload = json.loads(STRESS.read_text(encoding="utf-8"))
    assert "PROXY DATA" in payload["note"]
    assert "not real programme audio" in payload["note"]


def test_stress_artifact_records_the_measured_limitation():
    payload = json.loads(STRESS.read_text(encoding="utf-8"))
    by_name = {row["scenario"]: row for row in payload["scenarios"]}
    # The scenarios that reproduce real conditions.
    for hostile in ("ambience", "music_bed", "crowd"):
        assert by_name[hostile]["landmark_count"] == 0, (
            f"{hostile} no longer yields zero landmarks; re-measure before "
            "relying on this artifact"
        )
        assert "UNDER_SENSITIVE" in by_name[hostile]["verdict"]


def test_stress_artifact_shows_the_structure_is_recoverable():
    """The limitation is the threshold, not missing information."""
    payload = json.loads(STRESS.read_text(encoding="utf-8"))
    by_name = {row["scenario"]: row for row in payload["scenarios"]}
    ambience = by_name["ambience"]["floor_sensitivity"]
    # Twelve silences in each clip; the curve counts silence starts.
    assert ambience["-45"] == 0
    assert ambience["-35"] == 0
    # A higher floor recovers every silence, so the structure is there.
    assert ambience["-30"] == 12
    # And a high floor has its own cost: quiet material stops being found.
    uneven = by_name["uneven_gain"]["floor_sensitivity"]
    assert uneven["-20"] < uneven["-45"]
    # A music bed needs a markedly higher floor still, which is why no single
    # floor serves all content.
    assert by_name["music_bed"]["floor_sensitivity"]["-25"] == 0
    assert by_name["music_bed"]["floor_sensitivity"]["-20"] == 12


# --- the safe failure mode --------------------------------------------------- #


def test_zero_landmarks_abstain_rather_than_mismatch():
    """Hostile audio must produce no evidence, not wrong evidence."""
    profile = VideoTimelineProfile(
        duration_ms=288_000,
        audio_stream_count=1,
        selected_audio_stream=0,
        audio_stream_ambiguous=False,
        audio_landmarks=[],
    )
    evidence = validate_timeline_against_reference(profile, list(range(0, 200_000, 20_000)))
    assert evidence.verdict is VideoVerdict.INSUFFICIENT_EVIDENCE
    assert evidence.reason.value == "VIDEO_EVIDENCE_INSUFFICIENT"
    assert is_withholding_evidence(evidence) is False


def test_ambiguous_programme_audio_abstains():
    profile = VideoTimelineProfile(
        duration_ms=288_000,
        audio_stream_count=3,
        selected_audio_stream=None,
        audio_stream_ambiguous=True,
        audio_landmarks=[],
    )
    evidence = validate_timeline_against_reference(profile, list(range(0, 200_000, 20_000)))
    assert evidence.verdict is VideoVerdict.INSUFFICIENT_EVIDENCE
    assert evidence.reason.value == "AUDIO_STREAM_AMBIGUOUS"
    assert is_withholding_evidence(evidence) is False


def test_evidence_records_which_stream_was_used():
    profile = VideoTimelineProfile(
        duration_ms=288_000,
        audio_stream_count=2,
        selected_audio_stream=1,
        audio_stream_selection="default_disposition",
        audio_stream_ambiguous=False,
        audio_landmarks=[20_000, 24_000, 44_000, 48_000, 68_000, 72_000],
    )
    evidence = validate_timeline_against_reference(profile, profile.audio_landmarks)
    # Raw extraction context survives into the evidence, uncollapsed.
    assert evidence.audio_stream_count == 2
    assert evidence.selected_audio_stream == 1
    assert evidence.audio_stream_selection == "default_disposition"
    assert evidence.audio_landmarks == profile.audio_landmarks


# --- §10/§11: audio stream selection ---------------------------------------- #


def _audio(codec_type="audio", default=0, tags=None):
    return {"codec_type": codec_type, "disposition": {"default": default}, "tags": tags or {}}


def test_single_audio_stream_is_selected():
    choice = select_audio_stream([_audio()])
    assert choice.index == 0
    assert choice.ambiguous is False


def test_no_audio_stream_is_reported_not_guessed():
    choice = select_audio_stream([{"codec_type": "video"}])
    assert choice.index is None
    assert choice.ambiguous is True


def test_default_disposition_wins_over_ordering():
    """Track order must not decide; the container's own default must."""
    streams = [_audio(default=0), _audio(default=1)]
    choice = select_audio_stream(streams)
    assert choice.index == 1
    assert choice.reason == "default_disposition"


def test_commentary_track_is_excluded():
    streams = [
        _audio(default=0, tags={"commentary": "1"}),
        _audio(default=0, tags={"title": "English"}),
    ]
    choice = select_audio_stream(streams)
    assert choice.index == 1
    assert choice.reason == "only_untagged_programme_stream"
    assert any("excluded" in signal for signal in choice.signals)


def test_several_untagged_tracks_refuse_to_guess():
    """Two plausible programme tracks and no default: abstain."""
    streams = [_audio(default=0, tags={"title": "English"}), _audio(default=0, tags={"title": "French"})]
    choice = select_audio_stream(streams)
    assert choice.index is None
    assert choice.ambiguous is True
    assert any("no default disposition" in signal for signal in choice.signals)


# --- harness behaviour ------------------------------------------------------- #


def test_harness_reports_absent_media_as_a_setup_fact():
    result = _run(EVALUATOR, str(SAMPLE))
    result.check_returncode()
    assert "REAL MEDIA VIDEO VALIDATION BENCHMARK" in result.stdout
    # Absent media is not an extraction failure. Conflating them would read as
    # a finding about the files.
    assert "Media absent (local setup, not a finding): 3" in result.stdout
    assert "Extraction failures: 0" in result.stdout
    assert "Video profiles available: 0" in result.stdout


def test_harness_states_that_real_programme_audio_is_unmeasured():
    result = _run(EVALUATOR, str(SAMPLE))
    assert "No media was present" in result.stdout
    assert "does not extend to real content" in result.stdout


def test_harness_never_recommends_vad():
    result = _run(EVALUATOR, str(SAMPLE))
    lowered = result.stdout.lower()
    assert "does not recommend vad" in lowered
    # The harness quotes the stronger claim only to rule it out, so the test
    # checks for actual recommendations rather than the phrase appearing.
    for phrase in (
        "we recommend adding vad",
        "vad should be added",
        "add vad to",
        "implement vad",
    ):
        assert phrase not in lowered
    # And it separates the weak claim from the strong one.
    assert "not the same claim" in result.stdout


def test_harness_publishes_the_vad_decision_gate():
    result = _run(EVALUATOR, str(SAMPLE))
    assert "FALSE POSITIVE CLASSIFICATION" in result.stdout
    for category in (
        "A_timeline_solvable",
        "B_better_landmarks",
        "C_needs_semantic_audio",
        "D_insufficient_data",
    ):
        assert category in result.stdout
    # Only C counts as evidence, and that is stated rather than implied.
    assert "Only category C counts" in result.stdout
    assert "not the same claim" in result.stdout


def test_harness_uses_a_dedicated_cache_not_the_production_one():
    """§15: do not populate production caches."""
    from app.config import settings

    before = getattr(settings, "SUBS_CACHE_DIR", None)
    result = _run(EVALUATOR, str(SAMPLE))
    result.check_returncode()
    after = getattr(settings, "SUBS_CACHE_DIR", None)
    assert before == after


def test_harness_rejects_a_missing_manifest():
    result = _run(EVALUATOR, "no_such_manifest.json")
    assert result.returncode == 2
    assert "not found" in result.stderr


def test_harness_rejects_a_malformed_manifest(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{oops", encoding="utf-8")
    result = _run(EVALUATOR, str(bad))
    assert result.returncode == 2
    assert "failed validation" in result.stderr


def test_sample_ground_truth_stays_independent():
    manifest = load_real_media_manifest(SAMPLE)
    for case in manifest.cases:
        # Every label is authored, and provenance is recorded per case.
        assert case.ground_truth.established_by.value in ("objective", "human_reviewed")
    reviewed = [c for c in manifest.cases if c.ground_truth.reviewer_count > 0]
    assert reviewed, "a sample with no reviewer metadata cannot support review claims"


def test_sample_includes_a_positive_case_to_guard_false_positives():
    manifest = load_real_media_manifest(SAMPLE)
    assert any(c.ground_truth.final is GroundTruthFinal.CORRECT for c in manifest.cases)
    assert any(c.ground_truth.final is GroundTruthFinal.INCORRECT for c in manifest.cases)


# --- §18: no media in git --------------------------------------------------- #


def test_no_audio_or_video_artifacts_are_committed():
    """Generated media is not a licence to commit real media."""
    fixtures = REPO / "tests" / "fixtures" / "golden_sync"
    forbidden = {".wav", ".mkv", ".mp4", ".avi", ".m4v", ".webm", ".mov", ".flac", ".aac"}
    for path in fixtures.rglob("*"):
        if path.is_file():
            assert path.suffix.lower() not in forbidden, f"media committed: {path.name}"


def test_stress_tool_is_runnable_and_offline():
    """The generator writes audio to a workdir and needs no network."""
    source = STRESS_TOOL.read_text(encoding="utf-8")
    assert "urllib" not in source
    assert "requests" not in source
    assert "httpx" not in source
    assert "PROXY DATA" in source
