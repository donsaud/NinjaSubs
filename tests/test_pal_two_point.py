"""Two-point PAL-speedup synchronization: focused tests.

Covers the safety contract, not the success rate: exact PAL acceptance,
narrow rejection of everything else, metadata-free anchors, immutability,
determinism, provenance, and fail-closed behavior. The Whiplash regression
test at the bottom runs against real local files only when explicitly
pointed at them; it never commits media.
"""

from __future__ import annotations

import os

import pytest

from app.services.sync.pal_two_point import (
    MIN_ANCHOR_SPAN_MS,
    MODEL_VERSION,
    PAL_SLOPE,
    PAL_SLOPE_TOLERANCE,
    STRATEGY_ID,
    EndpointAnchor,
    attempt_pal_two_point,
    find_endpoint_anchors,
    is_pal_slope,
)


def _ts(ms: int) -> str:
    ms = max(0, int(round(ms)))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _srt(cues) -> str:
    return "\n\n".join(
        f"{i}\n{_ts(s)} --> {_ts(e)}\n{t}" for i, (s, e, t) in enumerate(cues, start=1)
    ) + "\n"


def _film(n: int = 1100, tag: str = "dialogue", start_ms: int = 1000, step_ms: int = 3000):
    return [
        (start_ms + i * step_ms, start_ms + i * step_ms + 2000, f"{tag} line {i} here")
        for i in range(n)
    ]


def _drift(cues, slope: float, intercept_ms: float):
    return [(slope * s + intercept_ms, slope * e + intercept_ms, t) for s, e, t in cues]


def _texts(target_cues, ref_cues):
    return _srt(target_cues), _srt(ref_cues)


class _StubEvaluation:
    def __init__(self, p95=100.0, struct=0.9, coverage=1.0):
        self.p95_offset_ms = p95
        self.structural_similarity = struct
        self.coverage_score = coverage
        self.reasons: list = []


class _StubAnalyzer:
    def __init__(self, p95=100.0, struct=0.9, coverage=1.0):
        self.seen: list = []
        self._p95, self._struct, self._cov = p95, struct, coverage

    def analyze(self, target_cues, transformed, reference_text, **kwargs):
        self.seen.append((target_cues, transformed, reference_text, kwargs))
        return _StubEvaluation(self._p95, self._struct, self._cov)


def _pal_pair(noise_ms: int = 0):
    import random

    rng = random.Random(7)
    target = _film()
    ref = [
        (
            PAL_SLOPE * s + 500.0 + (rng.randint(-noise_ms, noise_ms) if noise_ms else 0),
            PAL_SLOPE * e + 500.0,
            t,
        )
        for s, e, t in target
    ]
    return _srt(target), _srt(ref)


def test_pal_slope_exact_accepts():
    """1. Exact PAL slope with clean evidence is accepted."""
    target, reference = _pal_pair()
    result = attempt_pal_two_point(target, reference)
    assert result.accepted is True
    assert result.slope == pytest.approx(PAL_SLOPE, abs=1e-6)
    assert result.reason_codes == ["pal_accepted"]
    assert result.transformed_text is not None
    assert result.evaluation is not None


def test_pal_slope_with_small_timing_noise():
    """2. Real subtitle timing noise does not break PAL-class membership."""
    target, reference = _pal_pair(noise_ms=200)
    result = attempt_pal_two_point(target, reference)
    assert result.accepted is True


def test_non_pal_slope_rejected():
    """3. A constant shift (slope 1.0) is not PAL, however plausible."""
    target = _film(1100)
    reference = _film(1100)
    ref_text = _srt([(s + 2000, e + 2000, t) for s, e, t in reference])
    result = attempt_pal_two_point(_srt(target), ref_text)
    assert result.accepted is False
    assert "not_pal_class" in result.reason_codes
    assert result.transformed_text is None


def test_missing_first_anchor():
    """4. No usable opening dialogue means no attempt."""
    result = attempt_pal_two_point("", _srt(_film(1100)))
    assert result.accepted is False
    assert "no_endpoints" in result.reason_codes


def test_missing_last_anchor():
    """5. A single cue cannot supply both endpoints."""
    one = _srt([(1000, 3000, "only line here")])
    result = attempt_pal_two_point(one, _srt(_film(1100)))
    assert result.accepted is False
    assert "no_endpoints" in result.reason_codes


def test_first_missing_implies_last_missing():
    """5b. Endpoint runs are defined identically forward and backward, so a
    missing first run always means a missing last run: either guard alone
    aborts the attempt."""
    assert find_endpoint_anchors([], []) is None


def test_metadata_only_target_aborts_without_fabrication():
    """5c. A file with cues but no dialogue runs (credits only) must abort
    with no_endpoints, not proceed with zero-filled correspondences."""
    credits = [(1000 + i * 5000, 3000 + i * 5000, "Translated by XYZ") for i in range(5)]
    result = attempt_pal_two_point(_srt(credits), _srt(_film(1100)))
    assert result.accepted is False
    assert "no_endpoints" in result.reason_codes


def test_reversed_timestamps_rejected():
    """6. End before start is not a timeline."""
    target = _film(1100)
    fwd, rev = _srt(target), _srt(list(reversed(target)))
    assert attempt_pal_two_point(fwd, rev).accepted is False
    assert attempt_pal_two_point(rev, fwd).accepted is False


def test_metadata_never_becomes_an_endpoint():
    """7. Credits/URLs/handles at either end are skipped, not anchored."""
    from app.services.sync.ninja_points import _is_metadata_cue

    assert _is_metadata_cue("Translated by XYZ") is True
    credit = "Translated by XYZ www.example.com"
    assert _is_metadata_cue(credit) is True
    target = [(1000, 3000, credit)] + _film(1100, start_ms=5000)
    t_text, r_text = _srt(target), _srt(_drift(_film(1100), PAL_SLOPE, 500.0))
    result = attempt_pal_two_point(t_text, r_text)
    # The credit at 1000 ms must not be the first anchor; the run starts later.
    assert result.first is not None
    assert result.first.target_ms == 5000


def test_low_p95_rejected_with_stub():
    """8. High residual p95 fails the narrow policy even with PAL slope."""
    target, reference = _pal_pair()
    analyzer = _StubAnalyzer(p95=2500.0, struct=0.9, coverage=1.0)
    result = attempt_pal_two_point(target, reference, analyzer)
    assert result.accepted is False
    assert "p95_too_high" in result.reason_codes


def test_weak_structure_rejected_with_stub():
    """9. Weak structural agreement fails the narrow policy."""
    target, reference = _pal_pair()
    analyzer = _StubAnalyzer(p95=100.0, struct=0.4, coverage=1.0)
    result = attempt_pal_two_point(target, reference, analyzer)
    assert result.accepted is False
    assert "structure_too_low" in result.reason_codes


def test_low_coverage_rejected_with_stub():
    """10. Thin coverage fails the narrow policy."""
    target, reference = _pal_pair()
    analyzer = _StubAnalyzer(p95=100.0, struct=0.9, coverage=0.5)
    result = attempt_pal_two_point(target, reference, analyzer)
    assert result.accepted is False
    assert "coverage_too_low" in result.reason_codes


def test_successful_pal_transformation_properties():
    """11. Accepted result carries slope, transform, and evaluation."""
    target, reference = _pal_pair()
    result = attempt_pal_two_point(target, reference)
    assert result.accepted is True
    assert result.first is not None and result.last is not None
    assert result.last.target_ms - result.first.target_ms >= MIN_ANCHOR_SPAN_MS
    assert result.transformed_text is not None
    assert result.transformed_text != target
    assert result.evaluation is not None


def test_input_subtitle_immutability():
    """12. Original texts are never mutated."""
    target, reference = _pal_pair()
    frozen_target, frozen_reference = target, reference
    attempt_pal_two_point(target, reference)
    assert target == frozen_target
    assert reference == frozen_reference


def test_deterministic_output():
    """13. Same inputs produce byte-identical slope and decision."""
    target, reference = _pal_pair()
    first = attempt_pal_two_point(target, reference)
    second = attempt_pal_two_point(target, reference)
    assert (first.accepted, first.slope) == (second.accepted, second.slope)
    assert first.reason_codes == second.reason_codes


def test_cache_provenance_preserved():
    """14. Strategy and model identifiers are stable and distinct."""
    assert STRATEGY_ID == "two_point_linear_pal"
    assert MODEL_VERSION == "pal-two-point-v1"
    assert PAL_SLOPE == pytest.approx(25.0 / 23.976, rel=0)
    assert PAL_SLOPE_TOLERANCE == 0.001
    # Non-PAL ratios stay outside the tolerance band.
    assert not is_pal_slope(1.0)
    assert not is_pal_slope(25.0 / 24.0)
    assert is_pal_slope(PAL_SLOPE)


def test_fail_closed_on_analyzer_error():
    """15. An analyzer failure is a rejection, never an acceptance."""
    target, reference = _pal_pair()

    class _Boom:
        def analyze(self, *args, **kwargs):
            raise RuntimeError("verifier exploded")

    result = attempt_pal_two_point(target, reference, _Boom())
    assert result.accepted is False
    assert result.transformed_text is None


def test_endpoint_anchors_are_real_dialogue():
    """Endpoints skip credits and require dialogue runs on both sides."""
    from app.services.subtitle_matcher import parse_srt_cues

    target = (
        "1\n00:00:01,000 --> 00:00:03,000\nTranslated by XYZ\n\n"
        + _srt(_film(1100, start_ms=5000))
    )
    cues = parse_srt_cues(target)
    assert cues[0][2] == "Translated by XYZ"
    result = find_endpoint_anchors(cues, cues)
    assert result is not None
    first, last = result
    assert first.target_ms == 5000
    assert isinstance(first, EndpointAnchor) and isinstance(last, EndpointAnchor)
    assert last.target_ms > first.target_ms


SUBTEST_DIR = os.environ.get("NINJASUBS_SUBTEST_DIR", r"C:\Users\Saud\Desktop\SubTEST")


def _subtest_paths():
    import pathlib

    root = pathlib.Path(SUBTEST_DIR)
    target = root / "Whiplash (2014) WEBRip.iTunes.ar.srt"
    reference = root / "Whiplash.2014.1080p.REMUX.BluRay.AVC.DTS.HD.MA.5.1-HDA.srt"
    if not target.is_file() or not reference.is_file():
        pytest.skip("SubTEST fixtures absent (NINJASUBS_SUBTEST_DIR unset or missing files)")
    return target, reference


def test_whiplash_pal_two_point_regression():
    """Real-world Whiplash PAL-speedup regression (local files only)."""
    target_path, reference_path = _subtest_paths()
    target = target_path.read_text(encoding="utf-8", errors="replace")
    reference = reference_path.read_text(encoding="utf-8", errors="replace")
    result = attempt_pal_two_point(target, reference)
    assert result.accepted is True
    assert result.slope == pytest.approx(1.042748, abs=0.001)
    assert result.first is not None and result.first.target_ms == 142640
    assert result.transformed_text is not None
    assert "two_point_linear_pal" in STRATEGY_ID

async def test_pal_rejection_emits_structured_outcome_log(caplog):
    """Regression: a PAL rejection must be diagnosable from logs.

    A live Whiplash fallback once produced ALASS rejections followed by
    silence -- rejection emitted no log line, making "ran and rejected"
    indistinguishable from "never ran".
    """
    import logging

    from app.services.sync.orchestrator import SyncOrchestrator

    target = _srt(_film(200))
    # Constant shift: valid anchors, non-PAL slope -> policy rejection.
    shifted = [(s + 2000, e + 2000, t) for s, e, t in _film(200)]
    reference = _srt(shifted)
    orch = SyncOrchestrator(sync_cache=None, external_strategy=None, sync_service=None)
    with caplog.at_level(logging.INFO, logger="app.services.sync.orchestrator"):
        out = await orch._try_pal_two_point_fallback(
            target, reference, {}, "test-sub", b"x", "ch", "rk", None
        )
    assert out is None
    hits = [r for r in caplog.records if "pal-two-point" in r.message]
    assert hits, "rejection must leave a structured outcome line"
    line = hits[0].message
    assert "evaluated=true" in line and "accepted=False" in line
    assert "slope=" in line and "reasons=" in line
