"""Video-derived timeline validation.

The failure this layer exists to catch is not a bad residual. It is correctly
proving the wrong thing: a candidate and a reference sharing one incorrect
timing model produce a perfect residual, total provider consensus and a clean
verification. These tests pin the independent witness that can see it, and pin
the abstention behaviour everywhere else.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.services.sync.video_timeline import (
    MIN_SILENCE_MS,
    VIDEO_PROFILE_VERSION,
    VideoProfileCache,
    VideoTimelineProfile,
    VideoVerdict,
    is_withholding_evidence,
    validate_timeline_against_reference,
)

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "tests" / "fixtures" / "golden_sync" / "manifest.json"
EVALUATOR = REPO / "tools" / "evaluate_golden_sync.py"

#: The golden corpus subtitle payloads are generated artifacts and stay gitignored
#: (see ``.gitignore``). The cases below measure against them, so a fresh clone
#: has nothing to measure and they must skip rather than fail -- a permanently red
#: suite on every clean checkout reads as a regression and gets ignored.
requires_golden_corpus = pytest.mark.skipif(
    not any((REPO / "tests" / "fixtures" / "golden_sync" / "subtitles").glob("*.srt")),
    reason="golden-corpus subtitles are absent (generated, gitignored)",
)


def _blocks(
    count: int,
    *,
    block: int = 24_000,
    gap: int = 4_000,
    start: int = 2_000,
    vary: bool = True,
) -> list[int]:
    """Dialogue-block onsets separated by silences.

    ``vary`` makes the block lengths non-uniform, which is what real programme
    structure looks like. Strictly periodic input is a different case with a
    different correct answer, and it has its own test.
    """
    out: list[int] = []
    position = start
    for index in range(count):
        out.append(position)
        position += block + gap
        if vary:
            position += (index * 7_000) % 11_000
    return out


def _profile(landmarks: list[int], duration: int | None = None) -> VideoTimelineProfile:
    return VideoTimelineProfile(
        duration_ms=duration if duration is not None else landmarks[-1] + 60_000,
        audio_stream_count=1,
        video_stream_count=1,
        audio_landmarks=landmarks,
        video_landmarks=landmarks,
    )


# --- the wrong-cut signature ------------------------------------------------ #


def test_same_cut_matches_the_video():
    video = _blocks(12)
    evidence = validate_timeline_against_reference(_profile(video), video)
    assert evidence.verdict is VideoVerdict.MATCH
    assert is_withholding_evidence(evidence) is False


def test_spliced_wrong_cut_is_detected():
    """One shift explains the first half and not the rest: that is an edit."""
    video = _blocks(12)
    spliced = video[:6] + [value + 120_000 for value in video[6:]]
    evidence = validate_timeline_against_reference(_profile(video), spliced)
    assert evidence.verdict is VideoVerdict.MISMATCH
    assert is_withholding_evidence(evidence) is True
    # The diagnosis is preserved, not just the boolean.
    assert evidence.first_half_similarity is not None
    assert evidence.second_half_similarity is not None
    assert abs(evidence.first_half_similarity - evidence.second_half_similarity) >= 0.4
    assert any("splice signature" in detail for detail in evidence.detail)


def test_spliced_front_half_is_also_detected():
    """The test must not depend on which half diverges."""
    video = _blocks(12)
    spliced = [value - 60_000 for value in video[:6]] + video[6:]
    evidence = validate_timeline_against_reference(_profile(video), spliced)
    assert evidence.verdict is VideoVerdict.MISMATCH
    assert is_withholding_evidence(evidence) is True


def test_recap_edit_is_detected_by_a_different_signature():
    """§21: a second wrong cut, so the validator is not fitted to one shape."""
    video = _blocks(12)
    # A longer recap: the opening diverges and the timeline re-converges.
    recap = _blocks(3, block=34_000, gap=6_000, start=2_000) + _blocks(9, start=146_000)
    evidence = validate_timeline_against_reference(_profile(video), recap)
    assert evidence.verdict is VideoVerdict.MISMATCH
    assert is_withholding_evidence(evidence) is True


# --- §8: an offset is not a cut ------------------------------------------- #


@pytest.mark.parametrize("shift", [95_000, -12_000, 7_500])
def test_large_stable_offset_is_not_mistaken_for_a_wrong_cut(shift):
    video = _blocks(12)
    evidence = validate_timeline_against_reference(_profile(video), [v + shift for v in video])
    assert evidence.verdict is VideoVerdict.MATCH
    assert is_withholding_evidence(evidence) is False


def test_similar_duration_alone_does_not_prove_same_cut():
    """Duration is recorded but carries no weight on its own.

    Two subtitles with the same span are compared against the same video: one
    matches, one does not. If duration were doing the deciding, both verdicts
    would agree.
    """
    video = _blocks(12)
    profile = _profile(video, duration=video[-1] + 60_000)

    correct = validate_timeline_against_reference(profile, video)
    # Same opening, same final timestamp, different internal pacing in the back
    # half. Identical span, so the runtime cannot be what separates them.
    # A run in the middle is displaced. The first and last landmarks are
    # untouched, so the span and therefore the runtime are identical.
    wrong = validate_timeline_against_reference(
        profile, video[:3] + [v + 60_000 for v in video[3:9]] + video[9:]
    )
    assert wrong.reference_span_ms == correct.reference_span_ms

    assert correct.verdict is VideoVerdict.MATCH
    assert wrong.verdict is VideoVerdict.MISMATCH
    # Identical duration evidence, opposite verdicts.
    assert correct.video_duration_ms == wrong.video_duration_ms


# --- §19: absence of evidence is never negative evidence ------------------ #


def test_no_profile_abstains_and_cannot_withhold():
    evidence = validate_timeline_against_reference(None, _blocks(12))
    assert evidence.available is False
    assert evidence.verdict is VideoVerdict.UNAVAILABLE
    assert evidence.reason.value == "VIDEO_PROFILE_UNAVAILABLE"
    assert is_withholding_evidence(evidence) is False


def test_video_without_audio_abstains():
    profile = VideoTimelineProfile(
        duration_ms=100_000, audio_stream_count=0, video_stream_count=1, audio_landmarks=[]
    )
    evidence = validate_timeline_against_reference(profile, _blocks(12))
    assert evidence.verdict is VideoVerdict.INSUFFICIENT_EVIDENCE
    assert evidence.reason.value == "VIDEO_AUDIO_STREAM_MISSING"
    assert is_withholding_evidence(evidence) is False


def test_too_few_landmarks_abstains():
    profile = _profile([1_000, 2_000, 3_000])
    evidence = validate_timeline_against_reference(profile, _blocks(12))
    assert evidence.verdict is VideoVerdict.INSUFFICIENT_EVIDENCE
    assert evidence.reason.value == "VIDEO_EVIDENCE_INSUFFICIENT"
    assert is_withholding_evidence(evidence) is False


def test_sparse_subtitle_abstains_rather_than_guessing():
    profile = _profile(_blocks(12))
    evidence = validate_timeline_against_reference(profile, [1_000, 2_000])
    assert evidence.verdict is VideoVerdict.INSUFFICIENT_EVIDENCE
    assert is_withholding_evidence(evidence) is False


# --- §6: signals stay separate -------------------------------------------- #


def test_signals_are_reported_separately_and_never_collapsed():
    video = _blocks(12)
    evidence = validate_timeline_against_reference(_profile(video), video)
    # Each signal is addressable on its own. There is no combined score.
    assert evidence.audio_activity_similarity is not None
    assert evidence.scene_boundary_similarity is not None
    assert evidence.duration_similarity is not None
    assert not hasattr(evidence, "video_score")
    dumped = evidence.model_dump()
    assert "video_score" not in dumped


def test_silence_threshold_is_the_declared_one():
    assert MIN_SILENCE_MS == 700


# --- §15: caching by exact identity and version ---------------------------- #


def test_cache_round_trips_a_profile(tmp_path):
    cache = VideoProfileCache(tmp_path)
    profile = _profile(_blocks(12))
    cache.set("sha256:abc", profile)
    loaded = cache.get("sha256:abc")
    assert loaded is not None
    assert loaded.audio_landmarks == profile.audio_landmarks
    assert loaded.digest == profile.digest


def test_cache_does_not_reuse_across_identity(tmp_path):
    cache = VideoProfileCache(tmp_path)
    cache.set("sha256:abc", _profile(_blocks(12)))
    assert cache.get("sha256:different") is None


def test_cache_rejects_a_profile_from_another_version(tmp_path):
    cache = VideoProfileCache(tmp_path)
    profile = _profile(_blocks(12))
    cache.set("sha256:abc", profile)
    path = cache._path("sha256:abc")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["profile_version"] = VIDEO_PROFILE_VERSION + 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert cache.get("sha256:abc") is None


def test_cache_does_not_serve_a_malformed_partial_as_valid(tmp_path):
    cache = VideoProfileCache(tmp_path)
    cache.set("sha256:abc", _profile(_blocks(12)))
    path = cache._path("sha256:abc")
    path.write_text("{ truncated", encoding="utf-8")
    assert cache.get("sha256:abc") is None


def test_failure_is_cached_rather_than_retried_forever(tmp_path):
    cache = VideoProfileCache(tmp_path)
    cache.record_failure("sha256:broken", "VIDEO_PROFILE_EXTRACTION_FAILED")
    # A failure is not a profile, so it is never returned as one.
    assert cache.get("sha256:broken") is None


# --- determinism ----------------------------------------------------------- #


def test_profile_digest_is_deterministic():
    first = _profile(_blocks(12)).digest
    second = _profile(_blocks(12)).digest
    assert first == second
    assert _profile(_blocks(11)).digest != first


def test_extraction_absent_returns_none_rather_than_raising(tmp_path):
    from app.services.sync.video_timeline import extract_video_profile

    missing = tmp_path / "nope.mkv"
    assert extract_video_profile(missing) is None
    garbage = tmp_path / "garbage.mkv"
    garbage.write_bytes(b"not a video")
    # Either None (no ffprobe) or None (probe failed). Never an exception.
    assert extract_video_profile(garbage) is None


# --- §24: observational in the golden benchmark ---------------------------- #


def test_golden_benchmark_reports_video_validation_observational():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    result = subprocess.run(
        [sys.executable, str(EVALUATOR), str(MANIFEST)],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=env,
    )
    result.check_returncode()
    assert "Video-derived timeline validation" in result.stdout
    assert "Counterfactual" in result.stdout


@requires_golden_corpus
def test_video_validation_closes_the_measured_wrong_cut_gap():
    from app.services.sync.alignment import AlignmentAnalyzer
    from app.services.sync.golden import load_manifest

    sys.path.insert(0, str(REPO))
    import importlib.util

    spec = importlib.util.spec_from_file_location("golden_eval_vt", EVALUATOR)
    assert spec and spec.loader
    evaluator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = evaluator
    spec.loader.exec_module(evaluator)

    manifest = load_manifest(MANIFEST)
    analyzer = AlignmentAnalyzer()
    for case_id in (
        "same_release_different_cut",
        "circular_shared_wrong_timing",
        "wrong_cut_tv_edit_recap",
    ):
        case = manifest.case(case_id)
        assert case.target.video is not None, f"{case_id} needs a declared video timeline"
        results = evaluator.evaluate_case(case, MANIFEST, analyzer)
        assert results
        for result in results:
            assert result.annotated == "incorrect", f"{case_id} ground truth drifted"
            # The subtitle-side evidence is perfectly happy...
            assert result.system_verified is True
            # Clean, not necessarily exactly zero: the point is that the
            # subtitle-side evidence is entirely happy.
            assert abs(result.median_offset_ms or 0.0) <= 1_000
            # ...and the independent witness is not.
            assert result.video_would_withhold is True, f"{case_id} went undetected"


def test_video_validation_does_not_change_any_verdict():
    """§24: the layer observes. It must not alter a single SyncState."""
    from app.services.sync.alignment import AlignmentAnalyzer
    from app.services.sync.golden import load_manifest

    sys.path.insert(0, str(REPO))
    import importlib.util

    spec = importlib.util.spec_from_file_location("golden_eval_vt2", EVALUATOR)
    assert spec and spec.loader
    evaluator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = evaluator
    spec.loader.exec_module(evaluator)

    manifest = load_manifest(MANIFEST)
    analyzer = AlignmentAnalyzer()
    total = 0
    for case in manifest.cases:
        for result in evaluator.evaluate_case(case, MANIFEST, analyzer):
            total += 1
            assert result.error_class != "VIDEO_BLOCKED"
            assert result.system_state in {
                "verified_synced",
                "verified_resynced",
                "probable_sync",
                "unverified",
                "rejected",
            }
    assert total == sum(len(case.candidates) for case in manifest.cases)


def test_cases_without_a_video_abstain():
    from app.services.sync.alignment import AlignmentAnalyzer
    from app.services.sync.golden import load_manifest

    sys.path.insert(0, str(REPO))
    import importlib.util

    spec = importlib.util.spec_from_file_location("golden_eval_vt3", EVALUATOR)
    assert spec and spec.loader
    evaluator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = evaluator
    spec.loader.exec_module(evaluator)

    manifest = load_manifest(MANIFEST)
    case = manifest.case("same_release_constant_offset")
    assert case.target.video is None
    for result in evaluator.evaluate_case(case, MANIFEST, AlignmentAnalyzer()):
        assert result.video_available is False
        assert result.video_verdict == "unavailable"
        assert result.video_would_withhold is False


# --- §12: real-media manifest schema --------------------------------------- #


def test_real_media_manifest_reports_absent_media_rather_than_failing(tmp_path):
    from app.services.sync.golden import load_real_media_manifest

    manifest_path = tmp_path / "real.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_version": "real-v1",
                "cases": [
                    {
                        "case_id": "real_001",
                        "video": {"path": str(tmp_path / "movie.mkv")},
                        "reference": {"path": str(tmp_path / "ref.srt")},
                        "candidate": {"path": str(tmp_path / "cand.srt")},
                        "ground_truth": {
                            "cut": "different_cut",
                            "sync": "resyncable",
                            "final": "incorrect",
                            "review_status": "reviewed",
                            "reviewer_count": 2,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = load_real_media_manifest(manifest_path)
    assert len(manifest.cases) == 1
    # The media is absent, which is the expected local state.
    assert manifest.cases[0].has_media is False
    assert manifest.available_cases() == []


def test_real_media_manifest_loads_when_media_is_present(tmp_path):
    from app.services.sync.golden import load_real_media_manifest

    for name in ("movie.mkv", "ref.srt", "cand.srt"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    manifest_path = tmp_path / "real.json"
    manifest_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "real_001",
                        "video": {"path": str(tmp_path / "movie.mkv")},
                        "reference": {"path": str(tmp_path / "ref.srt")},
                        "candidate": {"path": str(tmp_path / "cand.srt")},
                        "ground_truth": {"final": "correct", "review_status": "reviewed"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = load_real_media_manifest(manifest_path)
    assert manifest.available_cases()[0].case_id == "real_001"


def test_no_media_is_committed_to_the_repository():
    """The schema and docs live in git; the media must not."""
    fixtures = REPO / "tests" / "fixtures" / "golden_sync"
    for path in fixtures.rglob("*"):
        if path.is_file():
            assert path.suffix in (".srt", ".vtt", ".json"), f"unexpected fixture: {path.name}"
            if path.suffix == ".json":
                assert path.stat().st_size < 200_000


def test_redacted_sample_manifest_is_valid_and_has_no_media():
    """The committed sample must parse, and must not point at real files."""
    from app.services.sync.golden import load_real_media_manifest

    sample = REPO / "tests" / "fixtures" / "golden_sync" / "real_media_sample.redacted.json"
    manifest = load_real_media_manifest(sample)
    assert manifest.cases, "the sample must demonstrate the format"
    assert manifest.available_cases() == []
    # Every path is a placeholder, so nothing resolves on any machine.
    for case in manifest.cases:
        assert "/REDACTED/" in case.video.path
        assert case.ground_truth.review_status
    categories = {case.category for case in manifest.cases}
    assert {"different_cut", "tv_edit_recap", "constant_offset_positive"} <= categories
    # A positive offset case is required, or the sample has no false-positive
    # guard and would only ever encourage a more aggressive validator.
    assert any(case.ground_truth.final.value == "correct" for case in manifest.cases)
    assert any(case.ground_truth.final.value == "incorrect" for case in manifest.cases)


# --- §27: the four golden video-validation cases --------------------------- #


def test_case_a_global_offset_is_not_withheld():
    """Same cut, shifted in time. A shift is not a cut."""
    video = _blocks(14)
    evidence = validate_timeline_against_reference(
        _profile(video), [value + 95_000 for value in video]
    )
    assert evidence.verdict is VideoVerdict.MATCH
    assert is_withholding_evidence(evidence) is False
    assert abs(evidence.best_offset_ms) > 1_000, "the offset was actually found"


def test_case_b_structural_cut_is_detected():
    """A different cut, with a discontinuity no single shift explains."""
    video = _blocks(14)
    spliced = video[:7] + [value + 90_000 for value in video[7:]]
    evidence = validate_timeline_against_reference(_profile(video), spliced)
    assert evidence.verdict is VideoVerdict.MISMATCH
    assert is_withholding_evidence(evidence) is True
    assert evidence.regional_consistency in ("mixed", "scattered")


def test_case_c_no_structure_abstains_rather_than_mismatching():
    """Continuous audio with nothing to compare. Absence, not disagreement."""
    profile = _profile(_blocks(14))
    empty = VideoTimelineProfile(
        duration_ms=profile.duration_ms,
        audio_stream_count=1,
        selected_audio_stream=0,
        audio_stream_ambiguous=False,
        audio_landmarks=[],
    )
    evidence = validate_timeline_against_reference(empty, _blocks(14))
    assert evidence.verdict is VideoVerdict.INSUFFICIENT_EVIDENCE
    assert evidence.reason.value == "VIDEO_EVIDENCE_INSUFFICIENT"
    assert is_withholding_evidence(evidence) is False


def test_case_d_ambiguous_dual_peaks_abstain():
    """Perfectly periodic structure correlates at several shifts equally.

    Nothing has been aligned, and a guess must not become a verification.
    """
    periodic = list(range(2_000, 2_000 + 12 * 28_000, 28_000))
    evidence = validate_timeline_against_reference(_profile(periodic), periodic)
    assert evidence.correlation_clarity == "ambiguous"
    assert evidence.verdict is VideoVerdict.INSUFFICIENT_EVIDENCE
    assert is_withholding_evidence(evidence) is False
    assert evidence.correlation_second_peak is not None
    assert evidence.correlation_peak_ratio is not None


def test_ambiguity_is_recorded_even_when_the_peak_is_weak():
    """A weak best peak is disagreement, not ambiguity.

    Otherwise a badly wrong cut could pass as merely unclear.
    """
    video = _blocks(14)
    unrelated = [value * 3 + 913 for value in video]
    evidence = validate_timeline_against_reference(_profile(video), unrelated)
    assert evidence.verdict is VideoVerdict.MISMATCH


def test_regional_analysis_is_separate_from_the_global_peak():
    video = _blocks(14)
    good = validate_timeline_against_reference(_profile(video), video)
    assert good.regional_consistency == "consistent"
    assert len(good.region_offsets) >= 2
    # One shift everywhere.
    assert max(good.region_offsets) - min(good.region_offsets) <= 1_500

    spliced = validate_timeline_against_reference(
        _profile(video), video[:7] + [value + 90_000 for value in video[7:]]
    )
    assert spliced.regional_consistency in ("mixed", "scattered")
    # Regional offsets are recorded, not averaged away.
    assert len(spliced.region_offsets) >= 2
    assert max(spliced.region_offsets) - min(spliced.region_offsets) > 1_500
