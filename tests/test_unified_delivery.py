"""Phase 4 release blockers: rendered artifacts, scope, negatives and telemetry."""
import hashlib
import logging
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app import main
from app.services.sync import hybrid_router as router
from app.services.sync import orchestrator as om
from app.services.sync.delivery import DeliveryAuthorization, DeliveryPayload, sha256
from app.services.sync.query import ResolvedReference
from app.services.sync.unified_verification import to_evaluation, verify_candidate
from app.services.sync_cache import SyncCache
from tests.test_unified_authority import META
from tests.test_unified_verification import (
    DEX_FRAMESTOR,
    DEX_TARGET,
    DEX_VERIFIED_CANDIDATE,
    ROVERS_CANDIDATE,
    ROVERS_REFERENCE,
    WHIP_REFERENCE,
    WHIP_TARGET,
    _linear_pair,
    _true_candidate,
)


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(router, "_enabled", True)
    monkeypatch.delenv(router.TEST_SCOPE_ENV, raising=False)
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True)


def authorized(candidate: bytes, original: bytes, request="request-one"):
    return DeliveryPayload(candidate, DeliveryAuthorization(
        request, sha256(candidate), "b" * 64, "exact", "VERIFIED", True, original))


def final_record(caplog):
    records = [r.message for r in caplog.records if "[unified-delivery]" in r.message]
    assert len(records) == 1
    return records[0]


@pytest.mark.parametrize("case", ["whiplash", "framestor", "rovers"])
@pytest.mark.parametrize("response_format", ["srt", "vtt"])
def test_real_verified_candidate_survives_final_rendering(case, response_format, caplog):
    caplog.set_level(logging.INFO)
    from tests.private_fixtures import require_subtitle

    if case == "whiplash":
        target = require_subtitle(WHIP_TARGET).read_bytes().decode("utf-8-sig")
        reference = require_subtitle(WHIP_REFERENCE).read_bytes().decode("utf-8-sig")
        classified = router.classify_pair(target, reference)
        candidate, _ = router.build_linear_output(classified["target_cues"], classified["anchor_result"])
    else:
        target = require_subtitle(DEX_TARGET).read_bytes().decode("utf-8-sig")
        reference = require_subtitle(
            DEX_FRAMESTOR if case == "framestor" else ROVERS_REFERENCE
        ).read_bytes().decode("utf-8-sig")
        candidate = require_subtitle(
            DEX_VERIFIED_CANDIDATE if case == "framestor" else ROVERS_CANDIDATE
        ).read_bytes().decode("utf-8-sig")
    from app.services.sync.candidate_reconstruction import reconstruct_with_original_dialogue
    candidate = reconstruct_with_original_dialogue(target, candidate)
    assert candidate is not None
    verdict = verify_candidate(target, reference, candidate, candidate_algorithm=case)
    assert verdict.proposed_verdict == "VERIFIED", verdict.failure_reasons
    payload = authorized(candidate.encode(), target.encode(), case)
    payload.authorization = replace(payload.authorization, reference_sha256=sha256(reference.encode()))
    response = main._build_subtitle_response(payload, case, response_format, auto_sync=True)
    record = final_record(caplog)
    assert "may_serve=True" in record and "final_integrity=True" in record
    assert "decision=TRANSFORMED" in record
    assert f"approved_candidate_sha256={verdict.candidate_sha256}" in record
    assert f"http_body_sha256={sha256(response.body)}" in record
    assert response.status_code == 200
    if response_format == "vtt":
        assert response.body.startswith(b"WEBVTT")
        assert sha256(response.body) != verdict.candidate_sha256


@pytest.mark.parametrize("fmt", ["srt", "vtt"])
def test_preferences_preserved_without_authorizing_arbitrary_dialogue(fmt, caplog):
    caplog.set_level(logging.INFO)
    candidate = ("1\n00:00:01,000 --> 00:00:03,000\n<i>مَرْحَبًا ,  123</i>\n\n"
                 "2\n00:02:00,000 --> 00:02:02,000\n[DOOR SLAMS] Hello  there!\n").encode()
    payload = authorized(candidate, candidate.replace(b"00:02:00", b"00:01:55"))
    response = main._build_subtitle_response(payload, "prefs", fmt, strip_hi=True,
        eastern_arabic_numerals=True, strip_diacritics=True, enable_rtl_fix=True)
    assert "may_serve=True" in final_record(caplog)
    assert response.body != candidate
    assert b"DOOR SLAMS" not in response.body
    assert "١٢٣".encode() in response.body


@pytest.mark.parametrize("mutation", ["text", "time", "drop", "reorder"])
def test_unexpected_post_verification_mutation_falls_back(mutation, caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    target, _ = _linear_pair()
    candidate = _true_candidate(target, 1, 5000).encode()
    render = main._run_subtitle_optimization_pipeline

    def tamper(body, **kwargs):
        body = render(body, **kwargs)
        if mutation == "text":
            return body.replace(b"\n", b"\nunauthorized dialogue\n", 1)
        if mutation == "time":
            return body.replace(b"00:01:05", b"00:01:45", 1)
        cues = body.split(b"\n\n")
        if mutation == "drop":
            del cues[2]
        else:
            cues[2], cues[3] = cues[3], cues[2]
        return b"\n\n".join(cues)

    monkeypatch.setattr(main, "_run_subtitle_optimization_pipeline", tamper)
    response = main._build_subtitle_response(authorized(candidate, target.encode()), "bad")
    assert response.body == render(target.encode())
    record = final_record(caplog)
    assert "may_serve=False" in record and "decision=ORIGINAL" in record
    assert "final_integrity=False" in record
    assert "unauthorized dialogue" not in record
    assert int(response.headers["content-length"]) == len(response.body)


def test_tampered_vtt_converter_fails_closed(caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    target, _ = _linear_pair()
    candidate = _true_candidate(target, 1, 5000).encode()
    convert = main.srt_to_vtt
    monkeypatch.setattr(main, "srt_to_vtt", lambda b: convert(b).replace(b"00:01:05", b"00:01:45", 1))
    response = main._build_subtitle_response(authorized(candidate, target.encode()), "bad", "vtt")
    assert "may_serve=False" in final_record(caplog)
    assert response.body.startswith(b"WEBVTT")
    assert b"00:01:45" not in response.body


@pytest.mark.parametrize("failure", ["unverified", "binding", "digest", "reference"])
def test_final_authorization_mandatory(failure, caplog):
    caplog.set_level(logging.INFO)
    target, _ = _linear_pair()
    payload = authorized(_true_candidate(target, 1, 5000).encode(), target.encode())
    changes = {"unverified":{"may_serve":False, "authoritative_verdict":"UNVERIFIED"},
               "binding":{"binding":"unknown"}, "digest":{"approved_sha256":"c"*64},
               "reference":{"reference_sha256":None}}
    payload.authorization = replace(payload.authorization, **changes[failure])
    response = main._build_subtitle_response(payload, "bad")
    assert "may_serve=False" in final_record(caplog)
    assert b"00:01:05" not in response.body


def test_cache_only_artifact_with_unsafe_render_is_withheld(caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    target, _ = _linear_pair()
    payload = authorized(target.encode(), target.encode())
    payload.authorization = replace(payload.authorization, original=None)
    monkeypatch.setattr(main, "_run_subtitle_optimization_pipeline", lambda *a, **k: b"tampered")
    response = main._build_subtitle_response(payload, "bad")
    assert response.status_code == 503 and response.body == b""
    assert "decision=WITHHELD" in final_record(caplog)


async def test_gate_off_and_unproven_transform(enabled, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    target, _ = _linear_pair()
    monkeypatch.setattr(main, "_maybe_sync_subtitle", AsyncMock(return_value=b"unproven transformed"))
    monkeypatch.setattr(main, "forced_transformed_bytes", lambda *a: b"forced")
    payload = await main._sync_subtitle_for_response(target.encode(), META, {}, "sub", None, True, True)
    assert payload == target.encode() and isinstance(payload, DeliveryPayload)
    main._build_subtitle_response(payload, "test")
    assert "may_serve=False" in final_record(caplog)
    monkeypatch.setattr(router, "_enabled", False)
    monkeypatch.delenv(router.TEST_ENABLED_ENV, raising=False)
    assert await main._sync_subtitle_for_response(target.encode(), META, {}, "sub", None, True, True) == b"forced"


@pytest.mark.parametrize("scope", ["bad:bad:scope", "tt0773262:8", "tt0773262::1", "tt0773262:8:",
                                    "tt0773262:8:0", "tt0773262:8:1:extra", "::", " "])
def test_invalid_scope_never_activates(monkeypatch, scope):
    monkeypatch.setenv(router.TEST_SCOPE_ENV, scope)
    if scope.strip():
        assert not router.scope_allows({"imdb_id":"tt0773262", "season":8, "episode":1})


@pytest.mark.parametrize("meta", [{}, {"imdb_id":"tt0773262"}, {"imdb_id":"tt0773262","season":8},
    {"season":8,"episode":1}, {"target_filename":"Dexter.s8e01.PiR8.mkv"},
    {"imdb_id":"tt0773262","season":8,"episode":2},
    {"imdb_id":"tt07732620","season":8,"episode":1},
    {"imdb_id":"tt0773262","season":8,"episode":True}])
def test_partial_or_unrelated_identity_denied(monkeypatch, meta):
    monkeypatch.setenv(router.TEST_SCOPE_ENV, "tt0773262:8:1")
    assert not router.scope_allows(meta)


def test_complete_scope_identity_and_default_off(monkeypatch):
    monkeypatch.setenv(router.TEST_SCOPE_ENV, "tt0773262:8:1")
    assert router.scope_allows({"imdb_id":"tt0773262", "season":"8", "episode":"1"})
    monkeypatch.setattr(router, "_enabled", False)
    monkeypatch.delenv(router.TEST_ENABLED_ENV, raising=False)
    assert not router.unified_authority_enabled({"imdb_id":"tt0773262", "season":8, "episode":1})


@pytest.mark.parametrize("meta,allowed", [
    ({"imdb_id":"tt2582802", "media_type":"movie"}, True),
    ({"imdb_id":"tt2582802"}, True),
    ({"imdb_id":"tt0773262", "season":8, "episode":1}, False),
    ({"imdb_id":"tt0111161", "media_type":"movie"}, False),
    ({"imdb_id":"tt2582802", "season":1}, False),
    ({"imdb_id":"tt2582802", "season":1, "episode":2}, False),
    ({"imdb_id":"tt2582802", "episode":1}, False),
    ({}, False),
])
def test_movie_scope_matrix(monkeypatch, meta, allowed):
    monkeypatch.setenv(router.TEST_SCOPE_ENV, "tt2582802")
    assert router.scope_allows(meta) is allowed


def test_activation_lives_only_in_the_temporary_override():
    # Tripwire against global activation: the permanent Compose file must not
    # enable the gate, while the temporary local-test override must carry the
    # complete controlled scope plus the forced-override kill switch.
    import pathlib

    repo = pathlib.Path(__file__).resolve().parent.parent
    plain = (repo / "docker-compose.yml").read_text(encoding="utf-8")
    override = (repo / "tests" / "compose" / "hybrid-test.yml").read_text(encoding="utf-8")
    assert "HYBRID_TEST_ENABLED=1" not in plain
    assert "HYBRID_TEST_SCOPE" not in plain
    assert "HYBRID_TEST_ENABLED=1" in override
    assert "HYBRID_TEST_SCOPE=tt0773262:8:1" in override
    assert "ENABLE_AUTOSYNC_TEST_OVERRIDE=false" in override
    assert router._enabled is False


@pytest.mark.parametrize("scope,parses", [
    ("ALL", True), (" ALL ", True),
    ("all", False), ("All", False), ("ALL:1", False), ("ALL:8:1", False),
])
def test_all_scope_exactness(monkeypatch, scope, parses):
    monkeypatch.setenv(router.TEST_SCOPE_ENV, scope)
    parsed = router.test_scope()
    assert (parsed == ("ALL", None, None)) is parses


@pytest.mark.parametrize("meta", [
    {"imdb_id": "tt2582802", "media_type": "movie"},
    {"imdb_id": "tt0773262", "season": 8, "episode": 1},
    {"imdb_id": "tt0111161"},
    {},
])
def test_all_mode_authorizes_evaluation_scope(monkeypatch, meta):
    monkeypatch.setenv(router.TEST_ENABLED_ENV, "1")
    monkeypatch.setenv(router.TEST_SCOPE_ENV, "ALL")
    assert router.is_enabled() is True
    assert router.activation_mode() == "ALL"
    assert router.scope_allows(meta) is True
    assert router.unified_authority_enabled(meta) is True


def test_all_mode_needs_explicit_flag(monkeypatch):
    monkeypatch.delenv(router.TEST_ENABLED_ENV, raising=False)
    monkeypatch.setenv(router.TEST_SCOPE_ENV, "ALL")
    monkeypatch.setattr(router, "_enabled", False)
    assert router.activation_mode() == "OFF"
    assert router.unified_authority_enabled({"imdb_id": "tt2582802"}) is False


def test_all_mode_kill_switch_wins(monkeypatch):
    monkeypatch.setenv(router.TEST_ENABLED_ENV, "1")
    monkeypatch.setenv(router.TEST_SCOPE_ENV, "ALL")
    monkeypatch.setenv(router.KILL_ENV, "1")
    assert router.activation_mode() == "OFF"
    assert router.unified_authority_enabled({"imdb_id": "tt2582802"}) is False


def test_activation_modes_reported(monkeypatch):
    monkeypatch.delenv(router.KILL_ENV, raising=False)
    monkeypatch.delenv(router.TEST_ENABLED_ENV, raising=False)
    monkeypatch.delenv(router.TEST_SCOPE_ENV, raising=False)
    monkeypatch.setattr(router, "_enabled", False)
    assert router.activation_mode() == "OFF"
    monkeypatch.setenv(router.TEST_ENABLED_ENV, "1")
    monkeypatch.setenv(router.TEST_SCOPE_ENV, "tt2582802")
    assert router.activation_mode() == "SCOPED"


def test_all_override_file_contents():
    import pathlib

    repo = pathlib.Path(__file__).resolve().parent.parent
    override = (repo / "tests" / "compose" / "unified-all.yml").read_text(encoding="utf-8")
    assert "HYBRID_TEST_ENABLED=1" in override
    assert "HYBRID_TEST_SCOPE=ALL" in override
    assert "ENABLE_AUTOSYNC_TEST_OVERRIDE=false" in override


@pytest.mark.parametrize("old,state,scope,changed,blocked", [
    (True,"rejected","identity",False,True), (True,"rejected",None,False,True),
    (True,"unverified",None,False,True), (True,"rejected","safety",True,False),
    (True,"rejected","candidate",False,False), (True,"rejected","methodology",False,False),
    (False,"rejected","methodology",False,True), (False,"rejected","candidate",False,False),
])
async def test_rejection_scope_and_exact_request_isolation(enabled, old, state, scope, changed, blocked):
    target, _ = _linear_pair()
    original = target.encode()
    cache = SyncCache()
    key = om.SyncOrchestrator._verdict_key(META, hashlib.sha256(original).hexdigest()[:16])
    if old:
        key = key.replace("verdict:8:timing-v3:", "verdict:7:")
    verdict = {"sync_state":state, "verification":"verified", "target_binding":"exact",
               "verification_method":"unified", "verification_version":"timing-v3",
               "unified_pipeline":"v2", "verification_input":"lenient-v1"}
    if scope:
        verdict["rejection_scope"] = scope
    await cache.set_verdict(key, verdict)
    strategy = SimpleNamespace(resolve_with_provenance=AsyncMock(return_value=ResolvedReference("")))
    orch = om.SyncOrchestrator(sync_cache=cache, external_strategy=strategy,
        sync_service=SimpleNamespace(sync_async=AsyncMock(return_value=None)))
    request = original + b"\n" if changed else original
    assert await orch.evaluate_and_sync(request, META, "sub", auto_sync=True) == request
    assert strategy.resolve_with_provenance.called is not blocked


async def test_cached_receipt_survives_reuse(enabled, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    target, reference = _linear_pair()
    candidate = _true_candidate(target, 1, 5000)
    evaluation = to_evaluation(verify_candidate(target, reference, candidate))
    evaluation.target_binding = "exact"
    orch = om.SyncOrchestrator(sync_cache=SyncCache())
    first = orch._delivery_payload(candidate.encode(), target.encode(), evaluation, META, "sub", sha256(reference.encode()))
    resolution = orch._flight_key(META, "sub", target.encode())
    await orch._sync_cache.set(resolution, bytes(first))
    await orch._sync_cache.set_meta(resolution, {"sync_state":"verified_resynced", "verification":"verified",
        "target_binding":"exact", "artifact_sha256":sha256(first), "reference_sha":sha256(reference.encode()),
        "verification_method":"unified", "verification_version":"timing-v3"})
    result = await orch.evaluate_and_sync(target.encode(), META, "sub", auto_sync=True)
    assert isinstance(result, DeliveryPayload)
    assert result.authorization.reference_sha256 == sha256(reference.encode())
    main._build_subtitle_response(result, "cached")
    assert "may_serve=True" in final_record(caplog)


def test_final_record_has_only_correlated_authorized_evidence(caplog):
    caplog.set_level(logging.INFO)
    target, _ = _linear_pair()
    candidate = _true_candidate(target, 1, 5000).encode()
    response = main._build_subtitle_response(authorized(candidate, target.encode(), "req-123"), "private")
    record = final_record(caplog)
    for field in ["req=req-123", "engine=8", "method=timing-v3", "authoritative_verdict=VERIFIED",
                  "reference_sha256="+"b"*64, "binding=exact", "may_serve=True",
                  "approved_candidate_sha256="+sha256(candidate), "http_body_sha256="+sha256(response.body),
                  "response_format=srt", "decision=TRANSFORMED"]:
        assert field in record
    assert "private" not in record
    assert target.splitlines()[2] not in record
