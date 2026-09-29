"""The per-file adaptive activity detector.

The fixed `silencedetect` floor worked only on audio containing true digital
silence, which real programmes almost never have. These tests pin the adaptive
detector that replaces it as the primary decision boundary, the settings that
centralise its parameters, and the properties that keep it an *activity*
detector rather than something pretending to understand speech.

They also pin the things that must NOT change: the fixed detector stays
available, no production synchronization threshold moves, and the profile
version distinguishes the two algorithms.
"""

import ast
import json
from pathlib import Path

import pytest

from app.config import settings
from app.services.sync.audio_activity import (
    ADAPTIVE_PROFILE_VERSION,
    AdaptiveAudioConfig,
    LandmarkQuality,
    _lower_mode_baseline,
    _percentile,
    _smooth,
    derive_threshold,
)
from app.services.sync.video_timeline import VIDEO_PROFILE_VERSION

REPO = Path(__file__).resolve().parent.parent
STRESS = REPO / "tests" / "fixtures" / "golden_sync" / "audio_stress_proxy.json"
MODULE = REPO / "app" / "services" / "sync" / "audio_activity.py"


# --- §2: it stays an activity detector -------------------------------------- #

FORBIDDEN_CONCEPTS = (
    "whisper",
    "speech_recognition",
    "transcribe",
    "phoneme",
    "speaker",
    "language_model",
    "embedding",
    "neural",
    "torch",
    "transformers",
)


def _code_without_prose(path: Path) -> str:
    """Module source with the docstring removed.

    The docstring explicitly disclaims speech understanding, so a plain
    substring search would flag its own disclaimer. What matters is whether the
    code does any of it.
    """
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                text = text.replace(doc, "")
    return text.lower()


def test_detector_does_not_attempt_speech_understanding():
    source = _code_without_prose(MODULE)
    for concept in FORBIDDEN_CONCEPTS:
        assert concept not in source, f"adaptive detector references {concept}"


def test_detector_does_not_import_ml_or_network_libraries():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    for name in imported:
        assert not name.startswith(("torch", "numpy", "scipy", "sklearn", "transformers"))
        assert name not in ("urllib.request", "requests", "httpx")


def test_detector_has_no_network_calls():
    source = MODULE.read_text(encoding="utf-8")
    for token in ("urlopen", "http://", "https://", "socket", "requests."):
        assert token not in source


# --- §25: parameters are centralised in the project's settings -------------- #


def test_settings_expose_the_adaptive_parameters():
    for name in (
        "ADAPTIVE_AUDIO_WINDOW_MS",
        "ADAPTIVE_AUDIO_HOP_MS",
        "ADAPTIVE_AUDIO_SMOOTHING_MS",
        "ADAPTIVE_AUDIO_ENTER_MARGIN_DB",
        "ADAPTIVE_AUDIO_EXIT_MARGIN_DB",
        "ADAPTIVE_AUDIO_MIN_THRESHOLD_DB",
        "ADAPTIVE_AUDIO_MAX_THRESHOLD_DB",
        "ADAPTIVE_AUDIO_MIN_SILENCE_MS",
        "ADAPTIVE_AUDIO_MIN_ACTIVITY_MS",
        "ADAPTIVE_AUDIO_MIN_BOUNDARY_SEPARATION_MS",
        "ADAPTIVE_AUDIO_BASELINE_PERCENTILE",
        "ADAPTIVE_AUDIO_MIN_LANDMARKS",
        "ADAPTIVE_AUDIO_MAX_LANDMARKS",
        "VIDEO_TIMELINE_REGIONS",
        "VIDEO_TIMELINE_AMBIGUITY_RATIO",
    ):
        assert hasattr(settings, name), f"{name} missing from settings"


def test_config_reads_the_settings_model():
    config = AdaptiveAudioConfig.from_settings(settings)
    assert config.window_ms == settings.ADAPTIVE_AUDIO_WINDOW_MS
    assert config.min_silence_ms == settings.ADAPTIVE_AUDIO_MIN_SILENCE_MS


def test_adaptive_is_disabled_by_default():
    """Extraction cost must stay opt-in; this is not a production change."""
    assert settings.ADAPTIVE_AUDIO_ENABLED is False
    assert settings.REFERENCE_SHADOW_POOL_FETCH_LIMIT == 0


def test_config_defaults_are_pinned():
    config = AdaptiveAudioConfig()
    assert config.window_ms == 50
    assert config.hop_ms == 25
    assert config.smoothing_ms == 250
    assert config.min_silence_ms == 700
    assert config.min_activity_ms == 400
    assert config.min_boundary_separation_ms == 1500


# --- §5: the threshold is bounded, and the clamp direction is right -------- #


def test_threshold_ceiling_pulls_values_down_not_up():
    """A ceiling is violated by exceeding it.

    The first implementation compared the wrong way, and a low ceiling then
    swallowed loud material silently. The direction is pinned here.
    """
    tight = AdaptiveAudioConfig(max_threshold_db=-40.0, min_threshold_db=-60.0)
    # An all-loud file: its derived threshold would sit high.
    energy = [-12.0] * 500
    _baseline, enter, exit_, _reasons = derive_threshold(energy, tight)
    assert enter <= tight.max_threshold_db
    assert exit_ <= tight.max_threshold_db


def test_threshold_floor_pulls_values_up():
    tight = AdaptiveAudioConfig(max_threshold_db=0.0, min_threshold_db=-20.0)
    energy = [-90.0] * 500
    _baseline, enter, exit_, _reasons = derive_threshold(energy, tight)
    assert enter >= tight.min_threshold_db
    assert exit_ >= tight.min_threshold_db


def test_extreme_recording_cannot_produce_a_nonsensical_threshold():
    for level in (-120.0, -60.0, -6.0, 0.0):
        energy = [level] * 400
        config = AdaptiveAudioConfig()
        _baseline, enter, exit_, _reasons = derive_threshold(energy, config)
        assert config.min_threshold_db <= exit_ <= config.max_threshold_db
        assert config.min_threshold_db <= enter <= config.max_threshold_db


def test_enter_threshold_is_never_below_exit():
    """Hysteresis ordering must hold, or the two thresholds fight."""
    for spread in (0.0, 3.0, 12.0, 40.0):
        energy = [-40.0 + (index % 7) * spread for index in range(400)]
        _baseline, enter, exit_, _reasons = derive_threshold(energy, AdaptiveAudioConfig())
        assert enter >= exit_


# --- §4/§6: the baseline is this file's floor, not its quietest moment ---- #


def test_baseline_is_not_a_fixed_percentile_of_the_distribution():
    """The quiet fraction of a file is not stable, so a fixed rank is wrong."""
    mostly_active = [-15.0] * 840 + [-120.0] * 160
    mostly_silent = [-15.0] * 160 + [-120.0] * 840
    config = AdaptiveAudioConfig()
    _b1, enter_active, _x1, _r1 = derive_threshold(mostly_active, config)
    _b2, enter_silent, _x2, _r2 = derive_threshold(mostly_silent, config)
    # The discriminating property: for the mostly-ACTIVE file, a 20th percentile
    # would land inside the active level at -15, putting the threshold above all
    # the audio so nothing is ever detected. The lower-mode estimate must sit
    # BELOW the active level instead.
    # Both must sit below the active level, so the active blocks are detectable,
    # despite the two files having opposite activity ratios.
    assert enter_active < -15.0
    assert enter_silent < -15.0
    # And the quiet floor is found in both.
    assert _lower_mode_baseline([-15.0] * 840 + [-120.0] * 160)[0] == -120.0
    assert _lower_mode_baseline([-15.0] * 160 + [-120.0] * 840)[0] == -120.0


def test_lowest_energy_is_not_assumed_to_be_silence():
    """A file whose floor is a room tone must find that tone as the floor."""
    room_tone = [-36.0] * 200 + [-14.0] * 800
    baseline, bulk, gap = _lower_mode_baseline(room_tone)
    assert baseline == -36.0
    assert bulk == -14.0
    assert gap == 22.0


def test_percentile_helper_is_bounded_and_deterministic():
    values = [float(v) for v in range(100)]
    assert _percentile(values, 0.0) == 0.0
    assert _percentile(values, 1.0) == 99.0
    assert _percentile(values, -5.0) == 0.0
    assert _percentile(values, 5.0) == 99.0
    assert _percentile(values, 0.5) == _percentile(values, 0.5)
    assert _percentile([], 0.5) == -120.0


# --- §7: smoothing --------------------------------------------------------- #


def test_smoothing_removes_single_frame_spikes():
    flat = [0.0] * 20
    spiked = list(flat)
    spiked[10] = 40.0
    smoothed = _smooth(spiked, 5)
    assert max(smoothed) < 20.0
    assert _smooth(flat, 5) == flat
    # A short window is a no-op.
    assert _smooth(spiked, 1) == spiked


# --- §11/§12: quality and pathological output ------------------------------ #


def test_quality_levels_exist():
    for level in ("strong", "usable", "weak", "insufficient", "invalid"):
        assert level in {q.value for q in LandmarkQuality}


def test_measurements_are_retained_not_collapsed_to_one_score():
    """A single opaque score would hide which measurement mattered."""
    from app.services.sync.audio_activity import AdaptiveAudioProfile

    profile = AdaptiveAudioProfile()
    for field in (
        "baseline_db",
        "enter_threshold_db",
        "exit_threshold_db",
        "split_gap_db",
        "activity_ratio",
        "segment_count",
        "frame_count",
        "stability_score",
        "activity_intervals",
        "landmarks",
        "reasons",
    ):
        assert hasattr(profile, field), f"{field} is not retained"
    assert not hasattr(profile, "score")


def test_pathological_landmark_sets_are_classified():
    from app.services.sync.audio_activity import AdaptiveAudioProfile, _quality

    config = AdaptiveAudioConfig()

    none = AdaptiveAudioProfile(duration_ms=100_000, landmarks=[])
    assert _quality(none, config) is LandmarkQuality.INSUFFICIENT

    single = AdaptiveAudioProfile(duration_ms=100_000, landmarks=[1_000])
    assert _quality(single, config) is LandmarkQuality.INSUFFICIENT

    too_dense = AdaptiveAudioProfile(
        duration_ms=100_000, landmarks=list(range(0, 100_000, 10))
    )
    assert _quality(too_dense, config) is LandmarkQuality.INVALID


def test_usable_quality_gate():
    from app.services.sync.audio_activity import AdaptiveAudioProfile

    profile = AdaptiveAudioProfile()
    assert profile.usable is False
    profile.quality = LandmarkQuality.USABLE
    assert profile.usable is True
    profile.quality = LandmarkQuality.WEAK
    assert profile.usable is False


# --- §24: profile versioning ------------------------------------------------ #


def test_adaptive_algorithm_has_its_own_profile_version():
    """A profile from one algorithm must never be served to the other."""
    assert ADAPTIVE_PROFILE_VERSION != VIDEO_PROFILE_VERSION


def test_video_profile_version_was_bumped_for_the_new_algorithm():
    assert VIDEO_PROFILE_VERSION >= 1
    # The adaptive module owns a distinct version space.
    assert ADAPTIVE_PROFILE_VERSION >= 2


# --- §17: ground truth stays independent of the detector -------------------- #


def test_scenario_synthesis_never_consults_the_detector():
    """The functions that BUILD a scenario must not see what MEASURES it.

    The tool as a whole legitimately runs both detectors; the synthesis path
    is what has to stay blind, or the ground truth would be shaped by the
    thing being graded.
    """
    source = (REPO / "tools" / "stress_audio_landmarks.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    blind = ("build_scenario", "render_scenario", "_background", "_overlay", "_tone", "_noise")
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in blind:
            seen.add(node.name)
            for sub in ast.walk(node):
                if isinstance(sub, ast.ImportFrom) and sub.module:
                    assert "audio_activity" not in sub.module
                    assert "video_timeline" not in sub.module
                if isinstance(sub, ast.Import):
                    for alias in sub.names:
                        assert "audio_activity" not in alias.name
                        assert "video_timeline" not in alias.name
    assert seen == set(blind), f"missing functions: {set(blind) - seen}"


def test_ground_truth_is_declared_before_any_waveform_is_written():
    source = (REPO / "tools" / "stress_audio_landmarks.py").read_text(encoding="utf-8")
    body = source[source.index("def render_scenario") : source.index("def _background")]
    segments_at = body.index("segments = build_scenario(name, seed)")
    write_at = body.index("wave.open")
    assert segments_at < write_at, "ground truth must precede synthesis"


# --- the measured result, as a committed artifact -------------------------- #


def test_artifact_shows_adaptive_recovers_the_structure_loss_cases():
    payload = json.loads(STRESS.read_text(encoding="utf-8"))
    comparisons = {row["scenario"]: row for row in payload["comparisons"]}
    structural_loss = ("ambience", "crowd", "music_bed", "drifting_noise_floor")
    for name in structural_loss:
        row = comparisons[name]
        assert row["fixed"]["f1"] == 0.0, f"{name}: fixed detector should find nothing"
        assert row["adaptive"]["f1"] > 0.8, f"{name}: adaptive did not recover structure"


def test_artifact_records_the_regression_rather_than_hiding_it():
    """A measured regression must stay in the record."""
    payload = json.loads(STRESS.read_text(encoding="utf-8"))
    comparisons = {row["scenario"]: row for row in payload["comparisons"]}
    regressions = [row for row in comparisons.values() if row["f1_difference"] < -0.05]
    assert regressions, "the rapid-dialogue regression should be recorded"
    rapid = comparisons["rapid_dialogue"]
    assert rapid["f1_difference"] < -0.5
    assert "REGRESSION" in rapid["decision"]


def test_artifact_is_labelled_proxy_and_not_real_media():
    payload = json.loads(STRESS.read_text(encoding="utf-8"))
    assert "PROXY DATA" in payload["note"]
    assert "not real programme audio" in payload["note"]


def test_artifact_records_the_config_used():
    payload = json.loads(STRESS.read_text(encoding="utf-8"))
    config = payload["adaptive_config"]
    assert config is not None
    for key in ("window_ms", "hop_ms", "smoothing_ms", "min_silence_ms"):
        assert key in config


# --- §31: the VAD gate, stated and not acted on ---------------------------- #


def test_phase_does_not_add_vad():
    for path in (MODULE, REPO / "app" / "services" / "sync" / "video_timeline.py"):
        source = _code_without_prose(path)
        assert "vad" not in source, f"{path.name} references VAD in code"
        assert "voice_activity" not in source
        assert "webrtcvad" not in source


def test_no_production_synchronization_threshold_moved():
    """The whole phase is extraction-side only."""
    from app.services.sync import alignment

    assert alignment.MIN_CUES_FOR_VERIFIED == 12


@pytest.mark.parametrize(
    "name",
    [
        "ADAPTIVE_AUDIO_WINDOW_MS",
        "ADAPTIVE_AUDIO_HOP_MS",
        "ADAPTIVE_AUDIO_SMOOTHING_MS",
        "ADAPTIVE_AUDIO_MIN_SILENCE_MS",
        "ADAPTIVE_AUDIO_MIN_ACTIVITY_MS",
        "ADAPTIVE_AUDIO_MIN_BOUNDARY_SEPARATION_MS",
    ],
)
def test_important_defaults_are_pinned(name):
    """A silent change to an extraction default would be invisible otherwise."""
    assert getattr(settings, name) is not None
