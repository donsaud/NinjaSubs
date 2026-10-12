"""Hybrid reachability regressions: the live 921-cue Dexter request.

Reproduces the observed production chain on real bytes with real functions
(no mocks on the decision path): large-offset investigation grants
eligibility, ALASS output validates, the legacy analyzer reports UNVERIFIED
(p95~=4735), the large-offset exemption grants serving -- and, with the test
gate on, Correspondence V2 now evaluates the candidate before delivery.

Real-data tests skip when fixtures are absent. The committed fixtures under
tests/fixtures/hybrid_rovers/ make the chain deterministic without the
ALASS binary; one test regenerates the candidate with the real binary when
``NINJASUBS_TEST_ALASS`` provides it.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pathlib
import shutil
import subprocess

import pytest

from app.services.subtitle_matcher import parse_srt_cues
from app.services.sync.alignment import AlignmentAnalyzer
from app.services.sync.hybrid_router import pre_delivery_review
from app.services.sync.large_offset_investigation import (
    decide_large_offset_serving,
    investigate_large_offset,
    validate_alass_output,
)
from app.services.sync.orchestrator import SyncOrchestrator
from two_point_sync.parser import Cue, format_srt, parse_srt

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"
MSD_TARGET = FIXTURES / "two_point_sync" / "dexter" / "Dexter.S08E01.480p.HDTV.x264-mSD.srt"
ROVERS_DIR = FIXTURES / "hybrid_rovers"
ROVERS_REFERENCE = ROVERS_DIR / "dexter-s08e01-rovers-634.srt"
ROVERS_CANDIDATE = ROVERS_DIR / "dexter-msd-rovers-alass-synced.srt"

EXPECTED_CANDIDATE_SHA256 = "ef56eefb5c075ad0b09a494fd7fb0a2795859989ac4232379a1c591ba6c37a8c"
EXPECTED_REFERENCE_SHA256 = "4ca0a36b98d296581f56f923d37ad185b669503f4c359c1bce0eafb173207c42"

META = {"imdb_id": "tt0773262", "season": 8, "episode": 1}


def _read(path: pathlib.Path) -> str:
    if not path.exists():
        pytest.skip(f"fixture unavailable: {path.name}")
    return path.read_bytes().decode("utf-8-sig")


def _target() -> str:
    return _read(MSD_TARGET)


def _reference() -> str:
    return _read(ROVERS_REFERENCE)


def _candidate() -> str:
    return _read(ROVERS_CANDIDATE)


def _enable_hybrid(monkeypatch) -> None:
    monkeypatch.setattr("app.services.sync.hybrid_router._enabled", True)
    monkeypatch.setenv("HYBRID_TEST_SCOPE", "tt0773262:8:1")


# ------------------------------------------------- legacy chain pinned ---


def test_legacy_chain_unverified_then_large_offset_serves() -> None:
    """Pins the BEFORE behavior: investigation eligible, analyzer
    UNVERIFIED with p95~=4735, large-offset exemption grants serving."""
    target, reference, candidate = _target(), _reference(), _candidate()
    assert len(parse_srt_cues(target)) == 921
    assert len(parse_srt_cues(reference)) == 634
    assert len(parse_srt_cues(candidate)) == 921

    investigation = investigate_large_offset(target, reference, identity_supported=True)
    assert investigation.eligible_for_alass
    assert abs((investigation.seed_offset_ms or 0) / 1000.0 - 96.53) < 2.0

    evaluation = AlignmentAnalyzer().analyze(
        parse_srt_cues(target), candidate, reference,
        alass_applied=True, alass_successful=True, max_plausible_offset_ms=20_000)
    assert evaluation.sync_state.value == "unverified"
    assert evaluation.p95_offset_ms is not None and evaluation.p95_offset_ms > 2000.0
    assert abs(evaluation.p95_offset_ms - 4735.0) < 500.0
    assert evaluation.rejection_reason is not None
    assert evaluation.rejection_reason.value == "low_confidence"

    validation = validate_alass_output(target, candidate, reference)
    assert validation.ok
    serving = decide_large_offset_serving(investigation, validation, evaluation)
    assert serving.value == "alass_corrected_large_offset"


def test_committed_fixture_identities() -> None:
    assert hashlib.sha256(ROVERS_CANDIDATE.read_bytes()).hexdigest() == EXPECTED_CANDIDATE_SHA256
    assert hashlib.sha256(ROVERS_DIR.joinpath(
        "dexter-s08e01-rovers-634.srt").read_bytes()).hexdigest() == EXPECTED_REFERENCE_SHA256


def test_alass_regeneration_is_deterministic() -> None:
    binary = os.getenv("NINJASUBS_TEST_ALASS") or shutil.which("alass")
    if binary is None:
        pytest.skip("alass not available in this environment")
    import tempfile

    target, reference = _target(), _reference()
    with tempfile.TemporaryDirectory() as tmp:
        ref = pathlib.Path(tmp) / "ref.srt"
        tgt = pathlib.Path(tmp) / "tgt.srt"
        out = pathlib.Path(tmp) / "out.srt"
        ref.write_text(reference, encoding="utf-8", newline="")
        tgt.write_text(target, encoding="utf-8", newline="")
        completed = subprocess.run(
            [binary, str(ref), str(tgt), str(out), "--split-penalty", "7.0"],
            capture_output=True, timeout=300)
        assert completed.returncode == 0
        assert hashlib.sha256(out.read_bytes()).hexdigest() == EXPECTED_CANDIDATE_SHA256


# ------------------------------------------------- V2 before delivery ---


def test_predelivery_accepts_verified_candidate(monkeypatch) -> None:
    _enable_hybrid(monkeypatch)
    target, reference, candidate = _target(), _reference(), _candidate()
    proceed, telemetry = pre_delivery_review(
        target_text=target, reference_text=reference, synced_text=candidate,
        meta=dict(META), exemption="large_offset")
    assert proceed is True
    assert telemetry["evaluated"] is True
    assert telemetry["route"] == "nonlinear"
    assert telemetry["verification_method"] == "unified"
    assert telemetry["verdict"] == "ACCEPTED"
    assert telemetry["transformed_sha"] == hashlib.sha256(
        candidate.encode("utf-8")).hexdigest()[:16]
    assert telemetry["reference_sha"] == hashlib.sha256(
        reference.encode("utf-8")).hexdigest()
    assert telemetry["timings_ms"]["unified_ms"] >= 0


def test_predelivery_rejects_tampered_candidate(monkeypatch) -> None:
    _enable_hybrid(monkeypatch)
    target, reference, candidate = _target(), _reference(), _candidate()
    cues = parse_srt(candidate)
    tampered = [
        Cue(number=c.number, order=c.order,
            start_ms=c.start_ms + (60_000 if c.number >= 82 else 0),
            end_ms=c.end_ms + (60_000 if c.number >= 82 else 0), text=c.text)
        for c in cues
    ]
    proceed, telemetry = pre_delivery_review(
        target_text=target, reference_text=reference,
        synced_text=format_srt(tampered), meta=dict(META), exemption="large_offset")
    assert proceed is False
    assert telemetry["verdict"] == "REJECTED"
    assert telemetry["reasons"], "rejection must carry explicit reasons"


def test_predelivery_insufficient_blocks_when_legacy_unverified(monkeypatch) -> None:
    _enable_hybrid(monkeypatch)
    target, _unused_reference, candidate = _target(), _reference(), _candidate()
    proceed, telemetry = pre_delivery_review(
        target_text=target, reference_text=target, synced_text=candidate,
        meta=dict(META), exemption="large_offset", legacy_unverified=True)
    assert proceed is False
    assert telemetry["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert telemetry["served_sha"] == "not_served"


def test_predelivery_insufficient_blocks_even_when_legacy_verified(monkeypatch) -> None:
    _enable_hybrid(monkeypatch)
    target, _unused_reference, candidate = _target(), _reference(), _candidate()
    proceed, telemetry = pre_delivery_review(
        target_text=target, reference_text=target, synced_text=candidate,
        meta=dict(META), exemption="large_offset", legacy_unverified=False)
    assert proceed is False
    assert telemetry["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_predelivery_error_blocks_when_legacy_unverified(monkeypatch) -> None:
    _enable_hybrid(monkeypatch)
    target, reference, candidate = _target(), _reference(), _candidate()
    monkeypatch.setattr(
        "app.services.sync.hybrid_router.unified_review",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    proceed, telemetry = pre_delivery_review(
        target_text=target, reference_text=reference, synced_text=candidate,
        meta=dict(META), exemption="large_offset", legacy_unverified=True)
    assert proceed is False
    assert telemetry["verdict"] == "ERROR"
    assert any("unified_error" in reason for reason in telemetry["reasons"])


def test_predelivery_error_blocks_even_when_legacy_verified(monkeypatch) -> None:
    _enable_hybrid(monkeypatch)
    target, reference, candidate = _target(), _reference(), _candidate()
    monkeypatch.setattr(
        "app.services.sync.hybrid_router.unified_review",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    proceed, telemetry = pre_delivery_review(
        target_text=target, reference_text=reference, synced_text=candidate,
        meta=dict(META), exemption="large_offset", legacy_unverified=False)
    assert proceed is False
    assert telemetry["verdict"] == "ERROR"


async def test_predelivery_timeout_blocks_when_legacy_unverified(monkeypatch) -> None:
    _enable_hybrid(monkeypatch)
    target, reference, candidate = _target(), _reference(), _candidate()
    import app.services.sync.hybrid_router as router

    def _slow(*args, **kwargs):
        import time as _time

        _time.sleep(0.5)
        raise RuntimeError("slow verifier")

    monkeypatch.setattr(router, "unified_review", _slow)
    from app.services.sync.hybrid_router import pre_delivery_review_async

    proceed, telemetry = await pre_delivery_review_async(
        target_text=target, reference_text=reference, synced_text=candidate,
        meta=dict(META), exemption="large_offset", legacy_unverified=True,
        timeout_s=0.01)
    assert proceed is False
    assert telemetry["verdict"] == "ERROR"
    assert any("unified_timeout" in reason for reason in telemetry["reasons"])


async def test_predelivery_disabled_is_noop(monkeypatch, caplog) -> None:
    target, reference, candidate = _target(), _reference(), _candidate()
    orch = SyncOrchestrator(sync_cache=None, external_strategy=None, sync_service=None)
    with caplog.at_level(logging.INFO, logger="app.services.sync.orchestrator"):
        assert await orch._hybrid_predelivery_gate(
            target_id="t", target_text=target, reference_text=reference,
            synced_text=candidate, meta=dict(META), exemption="large_offset") is True
    # Gate off: no evaluation telemetry, legacy behavior byte-identical.
    assert not [r for r in caplog.records if "unified" in r.message]


async def test_predelivery_out_of_scope_is_noop(monkeypatch) -> None:
    monkeypatch.setattr("app.services.sync.hybrid_router._enabled", True)
    monkeypatch.setenv("HYBRID_TEST_SCOPE", "tt0773262:8:1")
    target, reference, candidate = _target(), _reference(), _candidate()
    orch = SyncOrchestrator(sync_cache=None, external_strategy=None, sync_service=None)
    assert await orch._hybrid_predelivery_gate(
        target_id="t", target_text=target, reference_text=reference,
        synced_text=candidate,
        meta={"imdb_id": "tt0903747", "target_filename": "Breaking.Bad.S05E07.mkv"},
        exemption="large_offset") is True


async def test_orchestrator_gate_logs_request_correlated_telemetry(monkeypatch, caplog) -> None:
    _enable_hybrid(monkeypatch)
    target, reference, candidate = _target(), _reference(), _candidate()
    orch = SyncOrchestrator(sync_cache=None, external_strategy=None, sync_service=None)
    with caplog.at_level(logging.INFO, logger="app.services.sync.orchestrator"):
        assert await orch._hybrid_predelivery_gate(
            target_id="test-sub-921", target_text=target, reference_text=reference,
            synced_text=candidate, meta=dict(META), exemption="large_offset") is True
    lines = [r.message for r in caplog.records if "pre_delivery" in r.message]
    assert lines, "telemetry line required"
    line = lines[0]
    for fragment in ("test-sub-921", "evaluated=True", "unified", "ACCEPTED",
                     "proceed=True",
                     hashlib.sha256(candidate.encode()).hexdigest()):
        assert fragment in line, f"telemetry missing {fragment}: {line}"


async def test_orchestrator_gate_denies_tampered_candidate(monkeypatch) -> None:
    _enable_hybrid(monkeypatch)
    target, reference, candidate = _target(), _reference(), _candidate()
    cues = parse_srt(candidate)
    tampered = [
        Cue(number=c.number, order=c.order,
            start_ms=c.start_ms + (60_000 if c.number >= 82 else 0),
            end_ms=c.end_ms + (60_000 if c.number >= 82 else 0), text=c.text)
        for c in cues
    ]
    orch = SyncOrchestrator(sync_cache=None, external_strategy=None, sync_service=None)
    assert await orch._hybrid_predelivery_gate(
        target_id="t", target_text=target, reference_text=reference,
        synced_text=format_srt(tampered), meta=dict(META),
        exemption="large_offset") is False
