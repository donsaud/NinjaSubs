"""Hybrid Two-Point + ALASS routing tests (default-off gate).

Exercises the real integration adapter (``app.services.sync.hybrid_router``)
and the orchestrator tail method against the original Whiplash and Dexter
fixtures. The production analyzer, Target Binding, and serve gates stay
intact: fallbacks to the original subtitle are asserted wherever safety
requirements refuse serving. No live behavior changes (gate defaults off).
"""

from __future__ import annotations

import hashlib
import logging
import os
import pathlib
import shutil
import subprocess
import tempfile

import pytest

from app.config import settings
from app.services.sync.alignment import AlignmentAnalyzer, may_serve_synchronized
from app.services.sync.correspondence_verification import verify_nonlinear_regions
from app.services.sync.hybrid_router import (
    GATE_SETTING,
    attempt_hybrid,
    classify_pair,
)
from app.services.sync.orchestrator import SyncOrchestrator
from app.services.sync.target_binding import classify_target_binding
from tests.two_point_sync._helpers import build_dialogue
from two_point_sync.anchors import detect_anchors
from two_point_sync.parser import format_srt, parse_srt
from two_point_sync.sanitizer import classify_all
from two_point_sync.transform import apply_transform, fit_from_anchors

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"
WHIP_TARGET = FIXTURES / "whiplash" / "Whiplash (2014) WEBRip.iTunes.ar(1).srt"
WHIP_REFERENCE = (
    FIXTURES / "whiplash" / "Whiplash.2014.1080p.REMUX.BluRay.AVC.DTS.HD.MA.5.1-HDA(1).srt"
)
DEX_DIR = FIXTURES / "two_point_sync" / "dexter"
DEX_TARGET = DEX_DIR / "Dexter.S08E01.480p.HDTV.x264-mSD.srt"
DEX_REFERENCE = DEX_DIR / "Dexter.S08E01.A.Beautiful.Day.BluRay.1080p.TrueHD.5.1.AVC.REMUX-FraMeSToR.ar.srt"


def _read(path: pathlib.Path) -> str:
    if not path.exists():
        pytest.skip(f"fixture unavailable: {path.name}")
    return path.read_bytes().decode("utf-8-sig")


def _whiplash() -> tuple[str, str]:
    return _read(WHIP_TARGET), _read(WHIP_REFERENCE)


def _dexter() -> tuple[str, str]:
    return _read(DEX_TARGET), _read(DEX_REFERENCE)


class _RecordingRunner:
    """ALASS-runner double: records inputs, replays a canned outcome."""

    def __init__(self, outcome: str | BaseException | None) -> None:
        self.outcome = outcome
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, target_text: str, reference_text: str) -> str | None:
        self.calls.append((target_text, reference_text))
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def _block_shifted_dexter(target_text: str) -> str:
    """Replay the measured ALASS block shifts as fixture-generated output."""
    cues = parse_srt(target_text)
    shifted = []
    for cue in cues:
        delta = -96_485 if cue.number <= 81 else 10_322
        shifted.append(f"{cue.number}\n"
                       f"{_fmt(cue.start_ms + delta)} --> {_fmt(cue.end_ms + delta)}\n"
                       f"{cue.text}")
    return "\n\n".join(shifted) + "\n"


def _fmt(total_ms: int) -> str:
    total_ms = max(0, total_ms)
    millis = total_ms % 1000
    seconds = (total_ms // 1000) % 60
    minutes = (total_ms // 60_000) % 60
    hours = total_ms // 3_600_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _sentences(n: int, start_ms: int = 60_000, step_ms: int = 60_000) -> tuple[str, str]:
    target_cues = build_dialogue(n, start_ms=start_ms, step_ms=step_ms)
    target = format_srt(target_cues)
    shifted = [
        f"{c.number}\n{_fmt(c.start_ms + 2_000)} --> {_fmt(c.end_ms + 2_000)}\n{c.text}"
        for c in target_cues
    ]
    return target, "\n\n".join(shifted) + "\n"


def _paraphrased(n: int, start_ms: int = 60_000, step_ms: int = 60_000) -> tuple[str, str]:
    """Same timing shift as :func:`_sentences` but with reworded lines, so the
    reference carries genuinely different content (independent timing proof)
    while keeping exact correspondences for the nucleus."""
    target_cues = build_dialogue(n, start_ms=start_ms, step_ms=step_ms)
    target = format_srt(target_cues)
    alternates = [
        "Completely different sentence one",
        "Completely different sentence two",
        "Completely different sentence three",
    ]
    shifted = []
    for c in target_cues:
        text = c.text
        if c.number in (4, 8, 12):
            text = alternates[(c.number // 4) - 1]
        shifted.append(
            f"{c.number}\n{_fmt(c.start_ms + 2_000)} --> {_fmt(c.end_ms + 2_000)}\n{text}"
        )
    return target, "\n\n".join(shifted) + "\n"


# ---------------------------------------------------------------- gate ---


def test_gate_defaults_off() -> None:
    assert getattr(settings, GATE_SETTING, False) is False


async def test_gate_disabled_returns_none_immediately(caplog) -> None:
    orch = SyncOrchestrator(sync_cache=None, external_strategy=None, sync_service=None)
    with caplog.at_level(logging.DEBUG, logger="app.services.sync.orchestrator"):
        out = await orch._try_hybrid_sync(
            "t", "r", {}, "test-sub", b"orig", "ch", "rk", None,
        )
    assert out is None
    assert any("reason=disabled" in r.message for r in caplog.records)


# ------------------------------------------------------------- whiplash ---


async def test_whiplash_routes_linear_and_preserves_transform() -> None:
    target, reference = _whiplash()
    result = await attempt_hybrid(target, reference, analyzer=AlignmentAnalyzer())
    assert result.evaluated and result.route == "linear"
    assert result.scale is not None and abs(result.scale - 1.042755) <= 0.0005
    assert result.offset_ms is not None
    assert abs(result.offset_ms / 1000.0 - (-8.622)) <= 0.75
    # Transform preserves every cue and the Arabic text.
    assert result.accepted is True
    # Single-authority timing verification accepts the drift-corrected
    # output; the verdict, not a legacy positional refusal, is recorded.
    assert result.correspondence.get("verdict") == "ACCEPTED"
    assert result.transformed_text is not None
    assert result.correspondence.get("verification_method") == "unified"
    assert result.correspondence.get("median_residual_ms", 1e12) <= 750.0
    assert result.correspondence.get("independent_groups_used", 0) >= 5
    # The single authority agrees with itself: evaluation authorizes serving.
    assert result.evaluation is not None
    assert may_serve_synchronized(
        result.evaluation.sync_state, result.evaluation.verification
    )
    assert result.reference_sha == hashlib.sha256(reference.encode("utf-8")).hexdigest()


async def test_whiplash_linear_transform_content_preserved() -> None:
    target, reference = _whiplash()
    runner = _RecordingRunner(None)
    result = await attempt_hybrid(target, reference, analyzer=AlignmentAnalyzer(),
                                  alass_runner=runner)
    assert result.route == "linear"
    assert runner.calls == [], "linear path must never invoke ALASS"
    assert result.scale is not None and result.offset_ms is not None
    cues = parse_srt(target)
    detection = detect_anchors(cues, parse_srt(reference),
                               target_scores=classify_all(cues),
                               reference_scores=classify_all(parse_srt(reference)))
    assert detection.accepted
    model = fit_from_anchors(
        detection.first_anchor.target_ms, detection.first_anchor.reference_ms,
        detection.last_anchor.target_ms, detection.last_anchor.reference_ms,
    )
    assert abs(model.scale - result.scale) < 1e-9
    converted = apply_transform(cues, model)
    assert len(converted) == 825
    assert all(new.text == old.text for new, old in zip(converted, cues, strict=True))


# --------------------------------------------------------------- dexter ---


async def test_dexter_nonlinear_routes_to_alass_with_original() -> None:
    target, reference = _dexter()
    runner = _RecordingRunner(None)
    result = await attempt_hybrid(target, reference, analyzer=AlignmentAnalyzer(),
                                  alass_runner=runner)
    assert result.evaluated and result.route == "nonlinear"
    # ALASS receives the original Target, exactly once.
    assert len(runner.calls) == 1
    assert runner.calls[0][0] == target
    assert runner.calls[0][1] == reference
    # No rejected global transform exists to serve.
    assert result.transformed_text is None
    assert result.scale is None and result.offset_ms is None
    assert result.accepted is False
    assert result.reason_codes, "failure must carry explicit telemetry reasons"


async def test_dexter_regional_verification_with_block_shifts() -> None:
    target, reference = _dexter()
    runner = _RecordingRunner(_block_shifted_dexter(target))
    result = await attempt_hybrid(target, reference, analyzer=AlignmentAnalyzer(),
                                  alass_runner=runner)
    assert result.route == "nonlinear"
    telemetry = result.correspondence
    assert telemetry.get("verdict") == "ACCEPTED"
    coverage = telemetry.get("temporal_coverage_thirds", {})
    assert set(coverage) == {"opening", "middle", "ending"}
    for name in ("opening", "middle", "ending"):
        assert coverage[name] >= 2
    assert telemetry.get("median_residual_ms", 1e12) <= 1000.0
    # Single-authority verification accepts the ALASS output; the unified
    # evaluation authorizes serving (no shadow legacy refusal is recorded).
    assert result.accepted is True
    assert result.transformed_text is not None
    assert result.correspondence.get("verification_method") == "unified"
    assert result.evaluation is not None
    assert may_serve_synchronized(
        result.evaluation.sync_state, result.evaluation.verification
    )


# --------------------------------------------------------------- safety ---


async def test_alass_failure_modes_fall_back_to_original() -> None:
    target, reference = _dexter()
    raising = _RecordingRunner(TimeoutError("slow"))
    result = await attempt_hybrid(target, reference, analyzer=AlignmentAnalyzer(),
                                  alass_runner=raising)
    assert result.accepted is False and result.transformed_text is None
    assert any("alass_failed" in code for code in result.reason_codes)

    garbage = _RecordingRunner("this is not a subtitle file at all\n")
    result = await attempt_hybrid(target, reference, analyzer=AlignmentAnalyzer(),
                                  alass_runner=garbage)
    assert result.accepted is False and result.transformed_text is None
    assert any("unparseable" in code for code in result.reason_codes)


async def test_generic_only_pair_is_insufficient() -> None:
    async def _raising_runner(t: str, r: str):  # pragma: no cover - must not run
        raise AssertionError("no algorithm may run without evidence")

    def _generic(order: str) -> str:
        lines = []
        words = ["Yes", "No"] if order == "yn" else ["No", "Yes"]
        for position in range(8):
            word = words[position % 2]
            start = 60_000 + position * 60_000
            lines.append(f"{position + 1}\n{_fmt(start)} --> {_fmt(start + 2_000)}\n{word}")
        return "\n\n".join(lines) + "\n"

    result = await attempt_hybrid(_generic("yn"), _generic("ny"),
                                  analyzer=AlignmentAnalyzer(),
                                  alass_runner=_raising_runner)
    assert result.route == "insufficient"
    assert result.accepted is False and result.transformed_text is None
    assert any("nucleus_rejected" in code for code in result.reason_codes)


async def test_identical_content_cannot_prove_timing() -> None:
    target, _reference = _dexter()

    async def _raising_runner(t: str, r: str):  # pragma: no cover - must not run
        raise AssertionError("identical content proves nothing")

    result = await attempt_hybrid(target, target, analyzer=AlignmentAnalyzer(),
                                  alass_runner=_raising_runner)
    assert result.route == "insufficient"
    assert any("identical" in code for code in result.reason_codes)
    assert result.accepted is False


async def test_weak_binding_cannot_be_bypassed(monkeypatch) -> None:
    target, reference = _paraphrased(14)
    orch = SyncOrchestrator(sync_cache=None, external_strategy=None, sync_service=None)
    weak = classify_target_binding(
        "Dexter.S08E01.480p.HDTV.x264", "Dexter.S08E01.720p.HDTV.x264"
    )
    assert weak.state.value == "weak"
    monkeypatch.setattr(
        SyncOrchestrator, "_target_binding", staticmethod(lambda resolved, meta: weak)
    )
    monkeypatch.setattr("app.services.sync.hybrid_router._enabled", True)
    out = await orch._try_hybrid_sync(target, reference, {}, "test-sub", target.encode(),
                                      "ch", "rk", None)
    assert out is None, "weak binding must veto serving"


async def test_strong_binding_serves_verified_linear(monkeypatch) -> None:
    target, reference = _paraphrased(14)
    orch = SyncOrchestrator(sync_cache=None, external_strategy=None, sync_service=None)
    strong = classify_target_binding(
        "Some.Show.S08E01.480p.HDTV.x264-GRP", "Some.Show.S08E01.480p.HDTV.x264-GRP"
    )
    assert strong.supports_verified
    monkeypatch.setattr(
        SyncOrchestrator, "_target_binding", staticmethod(lambda resolved, meta: strong)
    )
    stored: dict = {}

    async def _store(self, *args, **kwargs) -> None:
        stored["called"] = True

    monkeypatch.setattr(SyncOrchestrator, "_store_verdict", _store)
    monkeypatch.setattr(SyncOrchestrator, "_audit_serve", lambda self, *a, **k: None)
    monkeypatch.setattr(
        SyncOrchestrator, "_record_outcome", lambda self, *a, **k: stored.setdefault("o", True)
    )
    monkeypatch.setattr("app.services.sync.hybrid_router._enabled", True)
    out = await orch._try_hybrid_sync(target, reference, {}, "test-sub", target.encode(),
                                      "ch", "rk", None)
    assert out is not None and out != target.encode(), "verified linear must serve"
    assert stored.get("called") is True
    served = parse_srt(out.decode("utf-8"))
    original = parse_srt(target)
    assert [c.start_ms for c in served] == [c.start_ms + 2_000 for c in original]


async def test_orchestrator_hybrid_off_preserves_existing_path(caplog) -> None:
    target, reference = _sentences(6)
    orch = SyncOrchestrator(sync_cache=None, external_strategy=None, sync_service=None)
    with caplog.at_level(logging.DEBUG, logger="app.services.sync.orchestrator"):
        assert await orch._try_hybrid_sync(target, reference, {}, "t", b"x", "ch", "rk",
                                           None) is None
    assert any("reason=disabled" in r.message for r in caplog.records)


def test_scope_allows_intended_video_only(monkeypatch) -> None:
    import app.services.sync.hybrid_router as router

    assert router.scope_allows({}) is True  # no scope configured: permissive core
    monkeypatch.setenv("HYBRID_TEST_SCOPE", "tt0773262:8:1")
    assert router.scope_allows(
        {"imdb_id": "tt0773262", "season": 8, "episode": 1}) is True
    assert router.scope_allows(
        {"imdb_id": "tt0773262", "season": 7, "episode": 1}) is False
    assert router.scope_allows({"imdb_id": "tt0903747"}) is False
    assert router.scope_allows(
        {"target_filename": "Dexter.s8e01.A.Beautiful.Day.1080p.BluRay-PiR8.mkv"}) is False
    assert router.scope_allows(
        {"target_filename": "Breaking.Bad.S05E07.720p.HDTV.mkv"}) is False


def test_env_opt_in_without_module_flag(monkeypatch) -> None:
    import app.services.sync.hybrid_router as router

    assert router.is_enabled() is False
    # An enabled flag WITHOUT a parseable scope must not globally activate.
    monkeypatch.setenv("HYBRID_TEST_ENABLED", "1")
    assert router.is_enabled() is False
    monkeypatch.setenv("HYBRID_TEST_SCOPE", "tt0773262:8:1")
    assert router.is_enabled() is True


@pytest.mark.parametrize("scope", ["", " ", "bad scope!!", "tt0773262:8"])
def test_env_enable_requires_parseable_scope(monkeypatch, scope: str) -> None:
    import app.services.sync.hybrid_router as router

    monkeypatch.setenv("HYBRID_TEST_ENABLED", "1")
    monkeypatch.setenv("HYBRID_TEST_SCOPE", scope)
    assert router.is_enabled() is False
    assert router.unified_authority_enabled({"imdb_id": "tt0773262"}) is False


def test_emergency_off_overrides_everything(monkeypatch) -> None:
    import app.services.sync.hybrid_router as router

    monkeypatch.setattr(router, "_enabled", True)
    monkeypatch.setenv("HYBRID_TEST_ENABLED", "1")
    monkeypatch.setenv("HYBRID_TEST_SCOPE", "tt0773262:8:1")
    assert router.is_enabled() is True
    monkeypatch.setenv(router.KILL_ENV, "1")
    assert router.is_enabled() is False
    assert router.unified_authority_enabled(
        {"imdb_id": "tt0773262", "season": 8, "episode": 1}) is False


async def test_scoped_out_requests_fall_through(caplog, monkeypatch) -> None:
    target, reference = _sentences(6)
    orch = SyncOrchestrator(sync_cache=None, external_strategy=None, sync_service=None)
    monkeypatch.setenv("HYBRID_TEST_ENABLED", "1")
    monkeypatch.setenv("HYBRID_TEST_SCOPE", "tt0773262:8:1")
    meta = {"imdb_id": "tt0903747", "target_filename": "Breaking.Bad.S05E07.mkv"}
    with caplog.at_level(logging.INFO, logger="app.services.sync.orchestrator"):
        assert await orch._try_hybrid_sync(target, reference, meta, "t", b"x", "ch",
                                           "rk", None) is None
    assert any("reason=scoped_out" in r.message for r in caplog.records)


class _StubSyncService:
    """Minimal sync-service double exposing only ``sync_async``."""

    def __init__(self, outcome: str | None) -> None:
        self.outcome = outcome
        self.calls: list[tuple[str, str]] = []

    async def sync_async(self, target_srt: str, reference_srt: str, *args, **kwargs):
        self.calls.append((target_srt, reference_srt))
        return self.outcome


def _serve_harness(monkeypatch, binding):
    orch_stubs: dict = {}
    monkeypatch.setattr(
        SyncOrchestrator, "_target_binding", staticmethod(lambda resolved, meta: binding)
    )

    async def _store(self, *args, **kwargs) -> None:
        orch_stubs["verdict_kwargs"] = kwargs

    monkeypatch.setattr(SyncOrchestrator, "_store_verdict", _store)
    monkeypatch.setattr(SyncOrchestrator, "_audit_serve", lambda self, *a, **k: None)
    monkeypatch.setattr(SyncOrchestrator, "_record_outcome", lambda self, *a, **k: None)
    monkeypatch.setattr("app.services.sync.hybrid_router._enabled", True)
    return orch_stubs


async def test_orchestrator_serves_whiplash_via_correspondence(monkeypatch, caplog) -> None:
    target, reference = _whiplash()
    strong = classify_target_binding(
        "Whiplash.2014.WEBRip.iTunes-GRP", "Whiplash.2014.WEBRip.iTunes-GRP"
    )
    assert strong.supports_verified
    orch_stubs = _serve_harness(monkeypatch, strong)
    orch = SyncOrchestrator(sync_cache=None, external_strategy=None, sync_service=None)
    with caplog.at_level(logging.INFO, logger="app.services.sync.orchestrator"):
        out = await orch._try_hybrid_sync(target, reference, {}, "test-sub", target.encode(),
                                          "ch", "rk", None)
    assert out is not None and out != target.encode()
    served = parse_srt(out.decode("utf-8"))
    assert len(served) == 825
    assert all(new.text == old.text for new, old in zip(served, parse_srt(target), strict=True))
    # Cache methodology is distinguished: hybrid-v2, never a legacy key.
    assert "hybrid-v2" in str(orch_stubs["verdict_kwargs"].get("artifact_key", ""))
    assert any("unified" in r.message for r in caplog.records)


async def test_two_point_real_whiplash_serves_under_unified(monkeypatch, caplog) -> None:
    """Genuine Two-Point (nucleus linear) case under Unified, end to end.

    Real Whiplash WEBRip target × 1080p REMUX reference takes the hybrid
    linear route (no ALASS involved); Unified VERIFIES, strong binding +
    may_serve authorize, and the served bytes survive final delivery
    integrity. Authority source is asserted, not just HTTP 200.
    """
    from app import main
    from app.services.sync.alignment import may_serve_synchronized

    target, reference = _whiplash()
    strong = classify_target_binding(
        "Whiplash.2014.WEBRip.iTunes-GRP", "Whiplash.2014.WEBRip.iTunes-GRP"
    )
    assert strong.supports_verified
    _serve_harness(monkeypatch, strong)

    async def _raising_runner(t: str, r: str):  # linear must never invoke ALASS
        raise AssertionError("two-point linear needs no ALASS")

    orch = SyncOrchestrator(sync_cache=None, external_strategy=None, sync_service=None)
    result = await attempt_hybrid(target, reference, analyzer=AlignmentAnalyzer(),
                                  alass_runner=_raising_runner)
    assert result.evaluated and result.route == "linear" and result.accepted is True
    out = await orch._try_hybrid_sync(target, reference, {}, "test-sub", target.encode(),
                                      "ch", "rk", None)
    assert out is not None and out != target.encode()
    served = parse_srt(out.decode("utf-8"))
    assert len(served) == 825
    assert all(new.text == old.text for new, old in zip(served, parse_srt(target), strict=True))
    evaluation = orch._last_evaluation
    assert evaluation is not None
    assert any(reason.startswith("unified-timing-v3:") for reason in evaluation.reasons)
    assert may_serve_synchronized(evaluation.sync_state, evaluation.verification)
    assert evaluation.target_binding in {"exact", "strong"}
    response = main._build_subtitle_response(out, "two-point", "srt", auto_sync=True)
    assert response.status_code == 200
    with caplog.at_level(logging.INFO, logger="app.services.sync.delivery"):
        main._build_subtitle_response(out, "two-point", "srt", auto_sync=True)
    assert any("final_integrity=True" in r.message for r in caplog.records)


async def test_two_point_real_whiplash_live_reference_dialogue_aligns() -> None:
    """Regional medians are diagnostic; actual residuals meet timing tolerances."""
    live_ref = FIXTURES / "whiplash" / "whiplash-live-framestor-926.srt"
    if not live_ref.exists():
        pytest.skip("live reference fixture unavailable")
    target, _ = _whiplash()
    reference = live_ref.read_bytes().decode("utf-8-sig")

    async def _raising_runner(t: str, r: str):
        raise AssertionError("linear must never invoke ALASS")

    result = await attempt_hybrid(target, reference, analyzer=AlignmentAnalyzer(),
                                  alass_runner=_raising_runner, unified_authority=True)
    assert result.evaluated and result.route == "linear"
    assert result.accepted is True
    assert result.correspondence["median_residual_ms"] == 670.0
    assert result.correspondence["dialogue_scope"]["global_alignment_claim"] is False


async def test_orchestrator_serves_dexter_via_correspondence(monkeypatch) -> None:
    target, reference = _dexter()
    strong = classify_target_binding(
        "Dexter.S08E01.480p.HDTV.x264-GRP", "Dexter.S08E01.480p.HDTV.x264-GRP"
    )
    assert strong.supports_verified
    service = _StubSyncService(_block_shifted_dexter(target))
    orch_stubs = _serve_harness(monkeypatch, strong)
    orch = SyncOrchestrator(sync_cache=None, external_strategy=None, sync_service=service)
    out = await orch._try_hybrid_sync(target, reference, {}, "test-sub", target.encode(),
                                      "ch", "rk", None)
    # ALASS ran once against the untouched original target ...
    assert len(service.calls) == 1
    assert service.calls[0][0] == target
    # ... and the regionally verified candidate is served (no global affine
    # transform exists or is served for this pair).
    assert out is not None and out != target.encode()
    served = parse_srt(out.decode("utf-8"))
    assert len(served) == 921
    assert "hybrid-v2" in str(orch_stubs["verdict_kwargs"].get("artifact_key", ""))


async def test_classify_pair_reports_explicit_reasons() -> None:
    assert classify_pair("", "")["route"] == "insufficient"
    target, reference = _dexter()
    assert classify_pair(target, reference)["route"] == "nonlinear"
    target, reference = _whiplash()
    assert classify_pair(target, reference)["route"] == "linear"


async def test_verify_nonlinear_regions_rejects_broken_transition() -> None:
    target, _ = _sentences(12)
    cues = parse_srt(target)
    # Candidate with a backward jump at the boundary: order violation.
    # (Shifted -300s so all timestamps stay positive and parseable.)
    bad = [f"{c.number}\n{_fmt(c.start_ms if c.number <= 6 else c.start_ms - 300_000)}"
           f" --> {_fmt(c.end_ms if c.number <= 6 else c.end_ms - 300_000)}\n{c.text}"
           for c in cues]
    bad_text = "\n\n".join(bad) + "\n"
    pairs = [(c.start_ms, c.start_ms, c.start_ms) for c in cues]
    report = verify_nonlinear_regions(parse_srt(bad_text), pairs, 60_000, 720_000)
    assert report["accepted"] is False
    assert any("order" in reason for reason in report["reasons"])


async def test_verify_nonlinear_regions_flags_large_outlier() -> None:
    target, _ = _sentences(12)
    cues = parse_srt(target)
    pairs = [(c.start_ms, c.start_ms + (5_000 if c.number == 12 else 0), c.start_ms)
             for c in cues]
    report = verify_nonlinear_regions(cues, pairs, 60_000, 720_000)
    assert report["accepted"] is False
    assert any("max_exceeds" in reason for reason in report["reasons"])


async def test_reference_provenance_preserved() -> None:
    for loader in (_whiplash, _dexter):
        target, reference = loader()
        result = await attempt_hybrid(target, reference,
                                      analyzer=AlignmentAnalyzer(),
                                      alass_runner=_RecordingRunner(None))
        assert result.reference_sha == hashlib.sha256(reference.encode()).hexdigest()


async def test_real_alass_dexter_end_to_end() -> None:
    binary = os.getenv("NINJASUBS_TEST_ALASS") or shutil.which("alass")
    if binary is None:
        pytest.skip("alass not available in this environment")

    async def _real_runner(t: str, r: str) -> str | None:
        with tempfile.TemporaryDirectory() as tmp:
            ref = pathlib.Path(tmp) / "ref.srt"
            tgt = pathlib.Path(tmp) / "tgt.srt"
            out = pathlib.Path(tmp) / "out.srt"
            # Verbatim newlines: text-mode translation would corrupt CRLF and
            # the reference ALASS binary rejects the result.
            ref.write_text(r, encoding="utf-8", newline="")
            tgt.write_text(t, encoding="utf-8", newline="")
            completed = subprocess.run(
                [binary, str(ref), str(tgt), str(out), "--split-penalty", "7.0"],
                capture_output=True, timeout=300,
            )
            assert completed.returncode == 0, "real ALASS execution failed"
            return out.read_text(encoding="utf-8")

    target, reference = _dexter()
    result = await attempt_hybrid(target, reference, analyzer=AlignmentAnalyzer(),
                                  alass_runner=_real_runner)
    assert result.route == "nonlinear"
    assert result.correspondence.get("verdict") == "ACCEPTED"
    assert result.evaluation is not None  # single authority authorizes serving
    assert may_serve_synchronized(
        result.evaluation.sync_state, result.evaluation.verification
    )
    # Correspondence-aware verification accepts; serving additionally needs
    # Target Binding, enforced by the orchestrator (tested below).
    assert result.accepted is True
    assert result.transformed_text is not None
    assert len(parse_srt(result.transformed_text)) == 921
