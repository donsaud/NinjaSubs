"""Whiplash live golden regression (Timeline Affine, tt2582802).

Live evidence (user-confirmed 6/6): SubDL `Whiplash (2014) WEBRip.iTunes.ar`
(825 cues) against a naturally selected SubDL FraMeSToR reference (926 cues)
via `timeline_affine` — PAL_LIKE_AFFINE, slope 1.0427472775355806, intercept
2688.544928596355 ms, 0 ALASS launches.

Committed fixture `whiplash-live-framestor-926.srt` is a byte copy of the
live-downloaded reference (SHA verified below); do NOT edit it. It must not
be confused with the offline 960-cue REMUX fixture used by Two-Point tests.
The target fixture is the existing WEBRip file (live bytes differ only by the
download-stripped BOM; decoded text identical). The approved candidate is
deterministically regenerated in-test with the real affine algorithm — never
fabricated — and pinned against the live approved SHA.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import pathlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app import main
from app.services.sync import hybrid_router as router
from app.services.sync import orchestrator as om
from app.services.sync.delivery import DeliveryAuthorization, DeliveryPayload, sha256
from app.services.sync.query import ResolvedReference
from app.services.sync.timeline_affine import attempt_timeline_affine
from app.services.sync.unified_verification import verify_candidate
from app.services.sync_cache import SyncCache
from tests.test_unified_authority import META
from two_point_sync.parser import Cue, format_srt, parse_srt

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"
WHIP_DIR = FIXTURES / "whiplash"
TARGET_PATH = WHIP_DIR / "Whiplash (2014) WEBRip.iTunes.ar(1).srt"
LIVE_REFERENCE_PATH = WHIP_DIR / "whiplash-live-framestor-926.srt"

LIVE_REFERENCE_SHA = "c6c1038c4cfa5984b82b34294d9f500d92658ee5d562d155b3c910ff2b2772b3"
LIVE_APPROVED_SHA = "a34e5b8616fdd782e6ecb1b8e17e80acc501a9f974257a888c4e891f7750a0bd"
LIVE_SLOPE = 1.0427472775355806
LIVE_INTERCEPT_MS = 2688.544928596355


def _read(path: pathlib.Path) -> str:
    return path.read_bytes().decode("utf-8-sig")


@pytest.fixture
def whiplash() -> tuple[str, str]:
    return _read(TARGET_PATH), _read(LIVE_REFERENCE_PATH)


def test_live_reference_bytes_are_exact(whiplash) -> None:
    _target, reference = whiplash
    assert hashlib.sha256(reference.encode("utf-8")).hexdigest() == LIVE_REFERENCE_SHA
    assert len(parse_srt(reference)) == 926


def test_affine_produces_live_candidate_from_real_bytes(whiplash) -> None:
    """Requirement 1: the real algorithm yields the live transformation."""
    target, reference = whiplash
    result = asyncio.run(asyncio.to_thread(attempt_timeline_affine, target, reference))
    assert result.accepted
    assert result.slope == LIVE_SLOPE
    assert result.intercept_ms == LIVE_INTERCEPT_MS
    assert hashlib.sha256(result.transformed_text.encode("utf-8")).hexdigest() == LIVE_APPROVED_SHA
    assert len(parse_srt(result.transformed_text)) == 825


def test_unified_verifies_live_candidate(whiplash) -> None:
    """Requirement 2: independent verification with live evidence numbers."""
    target, reference = whiplash
    result = asyncio.run(asyncio.to_thread(attempt_timeline_affine, target, reference))
    verdict = verify_candidate(target, reference, result.transformed_text,
                               candidate_algorithm="timeline_affine")
    assert verdict.proposed_verdict == "VERIFIED", verdict.failure_reasons
    assert verdict.independent_evidence_count == 51
    assert verdict.coverage == {"opening": 23, "middle": 17, "ending": 11}
    assert verdict.residual_statistics["median"] == 86.0
    assert verdict.residual_statistics["max"] == 1854.0


def test_live_candidate_preserves_dialogue_and_order(whiplash) -> None:
    """Requirement 3: cue count, order, and dialogue intact."""
    target, reference = whiplash
    result = asyncio.run(asyncio.to_thread(attempt_timeline_affine, target, reference))
    original = parse_srt(target)
    produced = parse_srt(result.transformed_text)
    assert len(produced) == len(original) == 825
    assert [cue.text for cue in produced] == [cue.text for cue in original]
    assert [cue.number for cue in produced] == [cue.number for cue in original]
    assert all(b.start_ms >= a.start_ms for a, b in zip(produced, produced[1:], strict=False))
    assert any(b.start_ms != a.start_ms for a, b in zip(original, produced, strict=False))


def _orchestrator(monkeypatch, reference: str):
    from app.services.sync.target_binding import TargetBinding

    strong = SimpleNamespace(
        state=TargetBinding.STRONG, reasons=("test_strong",), supports_verified=True,
        target=SimpleNamespace(), reference=SimpleNamespace(),
    )
    monkeypatch.setattr(
        om.SyncOrchestrator, "_target_binding", staticmethod(lambda resolved, meta: strong)
    )
    monkeypatch.setattr(om.SyncOrchestrator, "_audit_serve", lambda self, *a, **k: None)
    monkeypatch.setattr(om.SyncOrchestrator, "_record_outcome", lambda self, *a, **k: None)
    resolved = ResolvedReference(
        reference, kind="edition", candidate="Whiplash.2014.1080p.REMUX-FraMeSToR",
        reference_trust="strong",
    )
    strategy = SimpleNamespace(resolve_with_provenance=AsyncMock(return_value=resolved))
    service = SimpleNamespace(sync_async=AsyncMock(return_value=None))
    cache = SyncCache()
    return (om.SyncOrchestrator(external_strategy=strategy, sync_service=service,
                                sync_cache=cache), cache, service, strategy)


async def test_affine_path_serves_live_bytes_with_authorization(monkeypatch, whiplash) -> None:
    """Requirements 4+6: authorized only with binding/may_serve; no Legacy credit."""
    target, reference = whiplash
    monkeypatch.setattr(router, "_enabled", True)
    monkeypatch.delenv(router.TEST_SCOPE_ENV, raising=False)
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True)
    orch, cache, service, strategy = _orchestrator(monkeypatch, reference)
    out = await orch.evaluate_and_sync(target.encode("utf-8"), META, "sub", auto_sync=True)
    # Mirrors live telemetry: the affine route won with zero ALASS launches.
    assert service.sync_async.await_count == 0
    assert strategy.resolve_with_provenance.await_count == 1
    assert isinstance(out, DeliveryPayload)
    assert out.authorization.may_serve is True
    assert out.authorization.authoritative_verdict == "VERIFIED"
    assert hashlib.sha256(bytes(out)).hexdigest() == LIVE_APPROVED_SHA
    assert orch._last_evaluation is not None
    assert str(orch._last_evaluation.sync_state.value).startswith("verified")


async def test_affine_weak_binding_withholds_transformed(monkeypatch, whiplash) -> None:
    target, reference = whiplash
    monkeypatch.setattr(router, "_enabled", True)
    monkeypatch.delenv(router.TEST_SCOPE_ENV, raising=False)
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True)
    # Real weak assessment (mechanism test: any weak binding must veto). A
    # SimpleNamespace stand-in would bypass the orchestrator's BindingAssessment
    # type gate, so the genuine classifier is used here.
    from app.services.sync.target_binding import classify_target_binding

    weak = classify_target_binding(
        "Dexter.S08E01.480p.HDTV.x264", "Dexter.S08E01.720p.HDTV.x264"
    )
    assert weak.state.value == "weak"
    monkeypatch.setattr(
        om.SyncOrchestrator, "_target_binding", staticmethod(lambda resolved, meta: weak)
    )
    monkeypatch.setattr(om.SyncOrchestrator, "_audit_serve", lambda self, *a, **k: None)
    monkeypatch.setattr(om.SyncOrchestrator, "_record_outcome", lambda self, *a, **k: None)
    resolved = ResolvedReference(reference, kind="edition", reference_trust="strong")
    orch = om.SyncOrchestrator(
        external_strategy=SimpleNamespace(
            resolve_with_provenance=AsyncMock(return_value=resolved)),
        sync_service=SimpleNamespace(sync_async=AsyncMock(return_value=None)),
        sync_cache=SyncCache(),
    )
    out = await orch.evaluate_and_sync(target.encode("utf-8"), META, "sub", auto_sync=True)
    assert out == target.encode("utf-8")


async def test_legacy_positive_cannot_authorize_whiplash(monkeypatch, whiplash) -> None:
    """Requirement 6: a stored Legacy positive never authorizes engine-8."""
    target, reference = whiplash
    monkeypatch.setattr(router, "_enabled", True)
    monkeypatch.delenv(router.TEST_SCOPE_ENV, raising=False)
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True)
    orch, cache, service, strategy = _orchestrator(monkeypatch, reference)
    from app.services.sync_cache import SyncCache as _SC

    fingerprint = _SC.video_fingerprint_from_meta(META)
    old_key = _SC.build_verdict_key(
        fingerprint, hashlib.sha256(target.encode("utf-8")).hexdigest()[:16], "ara"
    )
    assert "unified" not in old_key
    await cache.set_verdict(old_key, {
        "sync_state": "verified_resynced", "verification": "verified",
        "reasons": ["legacy accepted"], "target_binding": "exact",
    })
    out = await orch.evaluate_and_sync(target.encode("utf-8"), META, "sub", auto_sync=True)
    assert strategy.resolve_with_provenance.await_count == 1
    assert isinstance(out, DeliveryPayload)
    assert hashlib.sha256(bytes(out)).hexdigest() == LIVE_APPROVED_SHA


def test_mutated_live_candidate_rejected(whiplash) -> None:
    """Requirement 7: one-word mutation of the approved bytes fails closed."""
    target, reference = whiplash
    result = asyncio.run(asyncio.to_thread(attempt_timeline_affine, target, reference))
    cues = parse_srt(result.transformed_text)
    mutated = format_srt([
        Cue(number=c.number, order=c.order, start_ms=c.start_ms, end_ms=c.end_ms,
            text="completely corrupted dialogue line" if c.number == 100 else c.text)
        for c in cues
    ])
    verdict = verify_candidate(target, reference, mutated,
                               candidate_algorithm="timeline_affine")
    assert verdict.proposed_verdict == "REJECTED"
    assert any("dialogue_text_modified" in reason for reason in verdict.failure_reasons)


def test_reordered_live_candidate_rejected(whiplash) -> None:
    target, reference = whiplash
    result = asyncio.run(asyncio.to_thread(attempt_timeline_affine, target, reference))
    cues = parse_srt(result.transformed_text)
    cues[200], cues[201] = cues[201], cues[200]
    reordered = format_srt([
        Cue(number=position + 1, order=position, start_ms=c.start_ms,
            end_ms=c.end_ms, text=c.text)
        for position, c in enumerate(cues)
    ])
    verdict = verify_candidate(target, reference, reordered,
                               candidate_algorithm="timeline_affine")
    assert verdict.proposed_verdict in ("REJECTED", "UNVERIFIED")


def test_live_bytes_survive_final_rendering(whiplash, caplog) -> None:
    """Requirement 5: SRT/VTT semantic integrity (body hash is prefs-dependent).

    No byte-for-byte HTTP fixture exists for the live prefs, so semantic
    preservation is asserted instead; the missing exact body hash is an
    explicit, accepted limitation.
    """
    caplog.set_level(logging.INFO)
    target, reference = whiplash
    result = asyncio.run(asyncio.to_thread(attempt_timeline_affine, target, reference))
    candidate = result.transformed_text.encode("utf-8")
    payload = DeliveryPayload(candidate, DeliveryAuthorization(
        "req-whip", sha256(candidate), sha256(reference.encode("utf-8")),
        "strong", "VERIFIED", True, target.encode("utf-8")))
    for fmt in ("srt", "vtt"):
        response = main._build_subtitle_response(payload, "whiplash", fmt, auto_sync=True)
        assert response.status_code == 200
        records = [r.message for r in caplog.records if "[unified-delivery]" in r.message]
        assert records and "decision=TRANSFORMED" in records[-1]
        assert "final_integrity=True" in records[-1]
        caplog.clear()


async def test_unified_inert_when_gate_off(monkeypatch, whiplash) -> None:
    """Requirement 8: default-OFF path never consults Unified."""
    target, reference = whiplash
    monkeypatch.setattr(router, "_enabled", False)
    monkeypatch.delenv(router.TEST_ENABLED_ENV, raising=False)
    review = AsyncMock(side_effect=AssertionError("unified must not run"))
    monkeypatch.setattr(om, "unified_review_async", review)
    monkeypatch.setattr(router, "unified_review_async", review)
    resolved = ResolvedReference(reference, kind="edition", reference_trust="strong")
    orch = om.SyncOrchestrator(
        external_strategy=SimpleNamespace(
            resolve_with_provenance=AsyncMock(return_value=resolved)),
        sync_service=SimpleNamespace(sync_async=AsyncMock(return_value=None)),
        sync_cache=SyncCache(),
    )
    out = await orch.evaluate_and_sync(target.encode("utf-8"), META, "sub", auto_sync=True)
    assert isinstance(out, bytes)
    review.assert_not_called()
