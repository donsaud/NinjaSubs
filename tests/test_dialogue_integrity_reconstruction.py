"""Dexter PiR8 live regression: sanitization-boundary dialogue integrity.

Live evidence: raw mSD target (56,748 chars, 921 cues) was sanitized to 52,973
chars for ALASS (cue 921 lost its ``facebook.com`` promo line; the translator
credit was kept). Real ALASS v2.0.0 (``--split-penalty 7.0``) preserved the
working copy byte-for-byte (blocks -96.5s/+10.2s) and exited 0, producing
``2adc320d...``. Unified compared those bytes against the RAW target and
rejected with ``dialogue_text_modified:1`` while Correspondence V2 (nonlinear
timing only, no text check) had accepted the same timings.

Committed fixtures under ``tests/fixtures/hybrid_rovers/`` are the exact live
bytes: the sanitized ALASS input (52,973 chars) and the raw ALASS output whose
SHA256 equals the live rejected candidate. Tests read bytes and decode only;
no fixture may be edited (hashes are asserted).
"""

from __future__ import annotations

import hashlib
import logging
import pathlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app import main
from app.services.sync import hybrid_router as router
from app.services.sync import orchestrator as om
from app.services.sync.candidate_reconstruction import (
    reconstruct_with_original_dialogue,
)
from app.services.sync.delivery import DeliveryPayload
from app.services.sync.query import ResolvedReference
from app.services.sync.unified_verification import verify_candidate
from app.services.sync_cache import SyncCache
from tests.test_unified_authority import META
from two_point_sync.anchors import normalize_text
from two_point_sync.parser import Cue, format_srt, parse_srt

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"
TARGET_PATH = FIXTURES / "two_point_sync" / "dexter" / "Dexter.S08E01.480p.HDTV.x264-mSD.srt"
REFERENCE_PATH = FIXTURES / "hybrid_rovers" / "dexter-s08e01-rovers-634.srt"
LIVE_INPUT_PATH = FIXTURES / "hybrid_rovers" / "dexter-msd-sanitized-alass-input.srt"
LIVE_OUTPUT_PATH = (
    FIXTURES / "hybrid_rovers" / "dexter-msd-rovers-alass-live-output.srt"
)

LIVE_CANDIDATE_SHA = "2adc320dbf2331a7ffdf082bc8444de2c8cc80b54c65c71c5210a00425718c24"
LIVE_REFERENCE_SHA = "4ca0a36b98d296581f56f923d37ad185b669503f4c359c1bce0eafb173207c42"
# Controlled-test approved bytes: original dialogue restored over verified ALASS
# timings (offline reconstruction reproduces these bytes exactly). The old
# sanitized-only candidate SHA above must NOT be re-required.
LIVE_APPROVED_SHA = "f3ff792bf014e9e59a5e17a17dfc342dffc0905aed242a0b4eee3b51a1abe0db"


def _read(path: pathlib.Path) -> str:
    return path.read_bytes().decode("utf-8-sig")


@pytest.fixture
def dexter() -> tuple[str, str, str, str]:
    return (
        _read(TARGET_PATH),
        _read(REFERENCE_PATH),
        _read(LIVE_INPUT_PATH),
        _read(LIVE_OUTPUT_PATH),
    )


def test_live_fixtures_are_exact(dexter) -> None:
    target, reference, live_input, live_output = dexter
    assert len(target) == 56748
    assert len(live_input) == 52973
    assert hashlib.sha256(live_output.encode("utf-8")).hexdigest() == LIVE_CANDIDATE_SHA
    assert hashlib.sha256(reference.encode("utf-8")).hexdigest() == LIVE_REFERENCE_SHA
    assert len(parse_srt(target)) == len(parse_srt(live_output)) == 921


def test_live_failure_reproduced_at_verifier(dexter) -> None:
    """The exact live bytes reject with dialogue_text_modified:1 (bug record)."""
    target, reference, _live_input, live_output = dexter
    verdict = verify_candidate(target, reference, live_output, candidate_algorithm="alass")
    assert verdict.proposed_verdict == "REJECTED"
    assert verdict.failure_reasons == ("dialogue_text_modified:1",)


def test_offending_cue_is_sanitization_provenance(dexter) -> None:
    """Exactly cue 921 differs raw-vs-live; sanitized-vs-live is identical."""
    target, _reference, live_input, live_output = dexter
    raw_cues, san_cues, out_cues = (
        parse_srt(target),
        parse_srt(live_input),
        parse_srt(live_output),
    )
    assert [c.number for c in out_cues] == [c.number for c in raw_cues]
    differing = [
        cue.number
        for cue, produced in zip(raw_cues, out_cues, strict=False)
        if normalize_text(cue.text) != normalize_text(produced.text)
    ]
    assert differing == [921]
    assert all(
        normalize_text(a.text) == normalize_text(b.text)
        for a, b in zip(san_cues, out_cues, strict=False)
    )
    # The removed line is promotional, the kept line is the translator credit.
    assert "facebook.com" in raw_cues[920].text
    assert "facebook.com" not in out_cues[920].text
    assert "Mahmoud Ibrahim" in out_cues[920].text


def test_reconstruction_preserves_dialogue_and_timings(dexter) -> None:
    target, _reference, _live_input, live_output = dexter
    reconstructed = reconstruct_with_original_dialogue(target, live_output)
    assert reconstructed is not None
    original_cues = parse_srt(target)
    produced_cues = parse_srt(live_output)
    rebuilt_cues = parse_srt(reconstructed)
    assert len(rebuilt_cues) == 921
    # Original user-facing dialogue verbatim (including the promo line).
    assert [cue.text for cue in rebuilt_cues] == [cue.text for cue in original_cues]
    assert "facebook.com" in rebuilt_cues[920].text
    # ALASS timings only.
    assert [(cue.start_ms, cue.end_ms) for cue in rebuilt_cues] == [
        (cue.start_ms, cue.end_ms) for cue in produced_cues
    ]
    assert rebuilt_cues[0].start_ms != original_cues[0].start_ms


def test_reconstructed_candidate_genuinely_verified(dexter) -> None:
    """VERIFIED on real timing evidence, not by construction."""
    target, reference, _live_input, live_output = dexter
    reconstructed = reconstruct_with_original_dialogue(target, live_output)
    assert reconstructed is not None
    verdict = verify_candidate(target, reference, reconstructed, candidate_algorithm="alass")
    assert verdict.proposed_verdict == "VERIFIED", verdict.failure_reasons
    assert verdict.independent_evidence_count >= 5
    assert verdict.residual_statistics["median"] <= 750.0
    assert verdict.residual_statistics["max"] <= 3000.0


def _tamper_text(cues: list[Cue], number: int, text: str) -> str:
    return format_srt([
        Cue(number=c.number, order=c.order, start_ms=c.start_ms, end_ms=c.end_ms,
            text=text if c.number == number else c.text)
        for c in cues
    ])


def test_genuine_mutation_has_no_reconstruction(dexter) -> None:
    """A real one-word change is not sanitization: helper refuses, Unified rejects."""
    target, reference, _live_input, live_output = dexter
    produced = parse_srt(live_output)
    mutated = _tamper_text(produced, 10, "completely corrupted dialogue line")
    assert reconstruct_with_original_dialogue(target, mutated) is None
    verdict = verify_candidate(target, reference, mutated, candidate_algorithm="alass")
    assert verdict.proposed_verdict == "REJECTED"
    assert any("dialogue_text_modified" in reason for reason in verdict.failure_reasons)


@pytest.mark.parametrize("mode", ["delete", "duplicate", "reorder", "drop_run"])
def test_structural_damage_has_no_reconstruction(dexter, mode: str) -> None:
    target, _reference, _live_input, live_output = dexter
    cues = parse_srt(live_output)
    if mode == "delete":
        damaged = [c for c in cues if c.number != 6]
    elif mode == "duplicate":
        clone = Cue(number=957, order=957, start_ms=cues[9].start_ms,
                    end_ms=cues[9].end_ms, text=cues[9].text)
        damaged = cues + [clone]
    elif mode == "reorder":
        damaged = list(cues)
        damaged[5], damaged[6] = damaged[6], damaged[5]
        damaged = [
            Cue(number=position + 1, order=position, start_ms=c.start_ms,
                end_ms=c.end_ms, text=c.text)
            for position, c in enumerate(damaged)
        ]
    else:
        # Drop a middle run: retention stays high, but every later cue
        # misaligns, so text identity must fail (no silent cue loss).
        damaged = [c for c in cues if not 100 <= c.number <= 120]
    assert reconstruct_with_original_dialogue(target, format_srt(damaged)) is None
    verdict = verify_candidate(target, _reference, format_srt(damaged),
                               candidate_algorithm="alass")
    assert verdict.proposed_verdict in ("REJECTED", "UNVERIFIED")


def test_helper_fail_closed_on_shape_problems(dexter) -> None:
    target, _reference, _live_input, _live_output = dexter
    assert reconstruct_with_original_dialogue("", "") is None
    assert reconstruct_with_original_dialogue("not a subtitle", "neither") is None
    assert reconstruct_with_original_dialogue(target, target[:1000]) is None


def test_helper_tolerates_lf_original(dexter) -> None:
    # The committed target uses CRLF; an LF-normalized original is the same
    # dialogue and must reconstruct identically (sanitize normalizes newlines).
    target, _reference, _live_input, live_output = dexter
    assert "\r\n" in target
    lf = target.replace("\r\n", "\n")
    reconstructed = reconstruct_with_original_dialogue(lf, live_output)
    assert reconstructed is not None
    assert len(parse_srt(reconstructed)) == 921


async def test_hybrid_nonlinear_reconstructs_original_dialogue(monkeypatch, dexter) -> None:
    """Router-level wiring: sanitized ALASS bytes regain original dialogue."""
    target, reference, live_input, live_output = dexter
    monkeypatch.setattr(router, "_enabled", True)
    monkeypatch.delenv(router.TEST_SCOPE_ENV, raising=False)

    async def _sanitized_runner(_t: str, _r: str) -> str | None:
        return live_output

    result = await router.attempt_hybrid(
        target, reference, alass_runner=_sanitized_runner, unified_authority=True
    )
    assert result.evaluated and result.accepted
    assert result.transformed_text is not None
    served = parse_srt(result.transformed_text)
    assert len(served) == 921
    assert "facebook.com" in served[920].text
    assert served[0].start_ms == parse_srt(live_output)[0].start_ms


def _orchestrator_with_live_bytes(monkeypatch, live_output: str, reference: str):
    from app.services.sync.target_binding import TargetBinding

    strong = SimpleNamespace(
        state=TargetBinding.STRONG, reasons=("test_strong",), supports_verified=True,
        target=SimpleNamespace(), reference=SimpleNamespace(),
    )
    monkeypatch.setattr(
        om.SyncOrchestrator, "_target_binding", staticmethod(lambda resolved, meta: strong)
    )
    monkeypatch.setattr(om.SyncOrchestrator, "_audit_serve", lambda self, *a, **k: None)
    monkeypatch.setattr(
        om.SyncOrchestrator, "_record_outcome", lambda self, *a, **k: None
    )
    resolved = ResolvedReference(
        reference, kind="hash", candidate=META["target_filename"],
        reference_trust="strong",
    )
    strategy = SimpleNamespace(
        resolve_with_provenance=AsyncMock(return_value=resolved)
    )
    service = SimpleNamespace(sync_async=AsyncMock(return_value=live_output))
    cache = SyncCache()
    orch = om.SyncOrchestrator(
        external_strategy=strategy, sync_service=service, sync_cache=cache
    )
    return orch, cache, service, strategy


async def test_orchestrator_serves_reconstructed_original(monkeypatch, dexter, caplog) -> None:
    """End to end: live bytes in, original dialogue with ALASS timings out."""
    caplog.set_level(logging.INFO)
    target, reference, _live_input, live_output = dexter
    monkeypatch.setattr(router, "_enabled", True)
    monkeypatch.delenv(router.TEST_SCOPE_ENV, raising=False)
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True)
    orch, cache, service, _strategy = _orchestrator_with_live_bytes(
        monkeypatch, live_output, reference
    )
    out = await orch.evaluate_and_sync(target.encode("utf-8"), META, "sub", auto_sync=True)
    assert service.sync_async.await_count == 1
    assert isinstance(out, DeliveryPayload)
    assert out.authorization.may_serve is True
    assert out.authorization.authoritative_verdict == "VERIFIED"
    served = parse_srt(out.decode("utf-8"))
    assert len(served) == 921
    assert served[920].text == parse_srt(target)[920].text
    assert "facebook.com" in served[920].text
    assert (served[0].start_ms, served[0].end_ms) == (
        parse_srt(live_output)[0].start_ms, parse_srt(live_output)[0].end_ms,
    )
    # Fresh verdict carries the pipeline marker for future reuse.
    verdict_key = om.SyncOrchestrator._verdict_key(
        META, hashlib.sha256(target.encode("utf-8")).hexdigest()[:16]
    )
    stored = await cache.get_verdict(verdict_key)
    assert stored is not None and stored.get("unified_pipeline") == "v2"


async def test_orchestrator_weak_binding_still_serves_original(monkeypatch, dexter) -> None:
    """Reconstruction does not bypass Target Binding."""
    target, reference, _live_input, live_output = dexter
    monkeypatch.setattr(router, "_enabled", True)
    monkeypatch.delenv(router.TEST_SCOPE_ENV, raising=False)
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True)
    from app.services.sync.target_binding import TargetBinding

    weak = SimpleNamespace(
        state=TargetBinding.WEAK, reasons=("test_weak",), supports_verified=False,
        target=SimpleNamespace(), reference=SimpleNamespace(),
    )
    monkeypatch.setattr(
        om.SyncOrchestrator, "_target_binding", staticmethod(lambda resolved, meta: weak)
    )
    monkeypatch.setattr(om.SyncOrchestrator, "_audit_serve", lambda self, *a, **k: None)
    monkeypatch.setattr(
        om.SyncOrchestrator, "_record_outcome", lambda self, *a, **k: None
    )
    resolved = ResolvedReference(reference, kind="hash", reference_trust="strong")
    orch = om.SyncOrchestrator(
        external_strategy=SimpleNamespace(
            resolve_with_provenance=AsyncMock(return_value=resolved)
        ),
        sync_service=SimpleNamespace(sync_async=AsyncMock(return_value=live_output)),
        sync_cache=SyncCache(),
    )
    out = await orch.evaluate_and_sync(target.encode("utf-8"), META, "sub", auto_sync=True)
    assert out == target.encode("utf-8")


async def test_stale_unmarked_integrity_negative_reevaluates(monkeypatch, dexter) -> None:
    """A pre-fix cached REJECTED must not suppress the corrected artifact."""
    target, reference, _live_input, live_output = dexter
    monkeypatch.setattr(router, "_enabled", True)
    monkeypatch.delenv(router.TEST_SCOPE_ENV, raising=False)
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True)
    orch, cache, service, strategy = _orchestrator_with_live_bytes(
        monkeypatch, live_output, reference
    )
    verdict_key = om.SyncOrchestrator._verdict_key(
        META, hashlib.sha256(target.encode("utf-8")).hexdigest()[:16]
    )
    # Pre-fix shape: Unified integrity rejection without the pipeline marker.
    await cache.set_verdict(verdict_key, {
        "sync_state": "unverified", "verification": "verified",
        "reasons": ["unified_rejected: dialogue_text_modified:1"],
        "target_binding": "exact",
        "verification_method": "unified", "verification_version": "unified-v1",
    })
    out = await orch.evaluate_and_sync(target.encode("utf-8"), META, "sub", auto_sync=True)
    assert strategy.resolve_with_provenance.await_count == 1
    assert service.sync_async.await_count == 1
    assert out != target.encode("utf-8")
    assert "facebook.com" in parse_srt(out.decode("utf-8"))[920].text


async def test_marked_integrity_negative_reuses(monkeypatch, dexter) -> None:
    """Post-fix negatives keep normal exact-request reuse (no ALASS loop)."""
    target, reference, _live_input, _live_output = dexter
    monkeypatch.setattr(router, "_enabled", True)
    monkeypatch.delenv(router.TEST_SCOPE_ENV, raising=False)
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True)
    orch, cache, service, strategy = _orchestrator_with_live_bytes(
        monkeypatch, "mutated", reference
    )
    verdict_key = om.SyncOrchestrator._verdict_key(
        META, hashlib.sha256(target.encode("utf-8")).hexdigest()[:16]
    )
    await cache.set_verdict(verdict_key, {
        "sync_state": "unverified", "verification": "verified",
        "reasons": ["unified_rejected: dialogue_text_modified:1"],
        "target_binding": "exact",
        "verification_method": "unified", "verification_version": "unified-v1",
        "unified_pipeline": "v2",
    })
    out = await orch.evaluate_and_sync(target.encode("utf-8"), META, "sub", auto_sync=True)
    assert out == target.encode("utf-8")
    strategy.resolve_with_provenance.assert_not_called()
    service.sync_async.assert_not_called()


def test_reconstructed_delivery_survives_rendering(dexter, caplog) -> None:
    """Reconstructed bytes pass final HTTP integrity in SRT and VTT."""
    caplog.set_level(logging.INFO)
    target, reference, _live_input, live_output = dexter
    reconstructed = reconstruct_with_original_dialogue(target, live_output)
    assert reconstructed is not None

    from app.services.sync.delivery import DeliveryAuthorization, sha256

    candidate = reconstructed.encode("utf-8")
    payload = DeliveryPayload(candidate, DeliveryAuthorization(
        "req-recon", sha256(candidate), sha256(reference.encode("utf-8")),
        "exact", "VERIFIED", True, target.encode("utf-8")))
    for fmt in ("srt", "vtt"):
        response = main._build_subtitle_response(payload, "recon", fmt, auto_sync=True)
        assert response.status_code == 200
        records = [r.message for r in caplog.records if "[unified-delivery]" in r.message]
        assert records and "may_serve=True" in records[-1]
        assert "final_integrity=True" in records[-1]
        caplog.clear()


def test_golden_live_anchors(dexter) -> None:
    """Permanent golden: live evidence anchors reproduced offline, byte-exact.

    Pins the controlled-test outcome (921 cues, sanitized working input,
    reconstructed approved bytes, reference identity, independent Unified
    evidence). The reconstructed SHA must equal the live approved SHA; it
    must NOT equal the old sanitized-only candidate SHA.
    """
    target, reference, live_input, live_output = dexter
    assert len(parse_srt(target)) == 921
    assert len(live_input) == 52973
    reconstructed = reconstruct_with_original_dialogue(target, live_output)
    assert reconstructed is not None
    assert hashlib.sha256(reconstructed.encode("utf-8")).hexdigest() == LIVE_APPROVED_SHA
    assert hashlib.sha256(reconstructed.encode("utf-8")).hexdigest() != LIVE_CANDIDATE_SHA
    assert hashlib.sha256(reference.encode("utf-8")).hexdigest() == LIVE_REFERENCE_SHA
    rebuilt = parse_srt(reconstructed)
    original = parse_srt(target)
    produced = parse_srt(live_output)
    assert [cue.text for cue in rebuilt] == [cue.text for cue in original]
    assert [(cue.start_ms, cue.end_ms) for cue in rebuilt] == [
        (cue.start_ms, cue.end_ms) for cue in produced
    ]
    verdict = verify_candidate(target, reference, reconstructed,
                               candidate_algorithm="alass")
    assert verdict.proposed_verdict == "VERIFIED"
    assert verdict.independent_evidence_count == 47
    assert verdict.coverage == {"opening": 12, "middle": 18, "ending": 9}
    assert verdict.residual_statistics["median"] == 351.0


async def test_golden_legacy_positive_cannot_authorize(monkeypatch, dexter) -> None:
    """A stored Legacy positive for the Dexter target never authorizes Unified."""
    target, reference, _live_input, live_output = dexter
    monkeypatch.setattr(router, "_enabled", True)
    monkeypatch.delenv(router.TEST_SCOPE_ENV, raising=False)
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True)
    orch, cache, service, strategy = _orchestrator_with_live_bytes(
        monkeypatch, live_output, reference
    )
    # Legacy-era positive under the pre-Unified key: no methodology segment,
    # no Unified fields. Under the gate the lookup key differs, and even a
    # direct payload hit would fail the Unified method/SHA requirements.
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
    # Fresh Unified evaluation ran (no authorization from the Legacy record).
    assert strategy.resolve_with_provenance.await_count == 1
    assert service.sync_async.await_count == 1
    assert isinstance(out, DeliveryPayload)
    assert out.authorization.may_serve is True
    assert hashlib.sha256(bytes(out)).hexdigest() == LIVE_APPROVED_SHA


FRAMESTOR_REFERENCE = (
    FIXTURES / "two_point_sync" / "dexter"
    / "Dexter.S08E01.A.Beautiful.Day.BluRay.1080p.TrueHD.5.1.AVC.REMUX-FraMeSToR.ar.srt"
)
FRAMESTOR_CANDIDATE = pathlib.Path("reports/two_point_sync/dexter.alass.verified.srt")


async def test_framestor_second_reference_end_to_end(monkeypatch) -> None:
    # Second real reference: FraMeSToR nonlinear path serves Unified-authorized
    # bytes. The committed FraMeSToR ALASS candidate was produced from the raw
    # target (dialogue identical), so reconstruction is a passthrough and
    # authority rests entirely on the fresh Unified measurement, never Legacy.
    if not FRAMESTOR_CANDIDATE.is_file():
        pytest.skip("fixture unavailable: dexter.alass.verified.srt")
    target = _read(TARGET_PATH)
    reference = FRAMESTOR_REFERENCE.read_bytes().decode("utf-8-sig")
    candidate = FRAMESTOR_CANDIDATE.read_bytes().decode("utf-8-sig")
    assert len(parse_srt(target)) == len(parse_srt(candidate)) == 921
    monkeypatch.setattr(router, "_enabled", True)
    monkeypatch.delenv(router.TEST_SCOPE_ENV, raising=False)
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True)
    orch, _cache, service, strategy = _orchestrator_with_live_bytes(
        monkeypatch, candidate, reference
    )
    out = await orch.evaluate_and_sync(target.encode("utf-8"), META, "sub", auto_sync=True)
    assert service.sync_async.await_count == 1
    assert strategy.resolve_with_provenance.await_count == 1
    assert isinstance(out, DeliveryPayload)
    assert out.authorization.may_serve is True
    assert out.authorization.authoritative_verdict == "VERIFIED"
    assert bytes(out) == candidate.encode("utf-8")
    assert orch._last_evaluation is not None
    assert str(orch._last_evaluation.sync_state.value).startswith("verified")
