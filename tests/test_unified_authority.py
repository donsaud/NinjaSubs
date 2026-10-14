"""Phase 3: no losing verifier or cache shortcut may authorize delivery."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.sync import hybrid_router as router
from app.services.sync import orchestrator as om
from app.services.sync.alignment import (
    SubtitleEvaluation,
    SyncState,
    VerificationAvailability,
    may_serve_synchronized,
)
from app.services.sync.query import ResolvedReference
from app.services.sync.unified_verification import UnifiedVerdict, to_evaluation
from app.services.sync_cache import SyncCache
from tests.test_unified_verification import _linear_pair, _true_candidate

META = {"imdb_id": "tt1", "lang": "ara", "video_hash": "a" * 16,
        "target_filename": "Film.2020.1080p.BluRay.x264-GRP"}


def _legacy(accepted=True):
    evaluation = SubtitleEvaluation(median_offset_ms=10, p95_offset_ms=20,
                                    coverage_score=1, structural_similarity=1)
    if accepted:
        evaluation.set_verdict(SyncState.VERIFIED_RESYNCED, VerificationAvailability.VERIFIED)
    return evaluation


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(router, "_enabled", True)
    monkeypatch.delenv(router.TEST_SCOPE_ENV, raising=False)
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True)


@pytest.mark.parametrize("verdict", ["VERIFIED", "UNVERIFIED", "REJECTED"])
async def test_selection_uses_only_unified(enabled, monkeypatch, verdict):
    legacy = _legacy(verdict != "VERIFIED")
    before = legacy.model_dump()
    selected = to_evaluation(UnifiedVerdict(proposed_verdict=verdict,
                                          independent_evidence_count=14))
    monkeypatch.setattr(om, "unified_review_async", AsyncMock(return_value=(selected, {})))
    orch = om.SyncOrchestrator()
    result = await orch._select_evaluation(legacy, target_text="t", reference_text="r",
                                          candidate_text="c", meta=META, algorithm="alass")
    assert result is selected and result is not legacy
    assert may_serve_synchronized(result.sync_state, result.verification) == (verdict == "VERIFIED")
    assert legacy.model_dump() == before


@pytest.mark.parametrize("trust", ["strong", "acceptable", "unknown", "rejected"])
async def test_reference_label_does_not_override_measured_dialogue(enabled, trust):
    target, reference = _linear_pair()
    result = await om.SyncOrchestrator()._select_evaluation(
        _legacy(False), target_text=target, reference_text=reference,
        candidate_text=_true_candidate(target, 1, 5000), meta=META, algorithm="alass",
        resolved=ResolvedReference(reference, reference_trust=trust),
    )
    assert result.reference_trust == trust
    assert result.sync_state == SyncState.VERIFIED_RESYNCED


async def test_disabled_and_scoped_out_never_run_unified(monkeypatch):
    review = AsyncMock(side_effect=AssertionError("must not verify"))
    monkeypatch.setattr(om, "unified_review_async", review)
    legacy = _legacy()
    orch = om.SyncOrchestrator()
    monkeypatch.setattr(router, "_enabled", False)
    monkeypatch.delenv(router.TEST_ENABLED_ENV, raising=False)
    assert await orch._select_evaluation(legacy, target_text="", reference_text="",
                                        candidate_text="", meta=META, algorithm="alass") is legacy
    monkeypatch.setattr(router, "_enabled", True)
    monkeypatch.setenv(router.TEST_SCOPE_ENV, "tt2:8:1")
    assert await orch._select_evaluation(legacy, target_text="", reference_text="",
                                        candidate_text="", meta=META, algorithm="alass") is legacy
    review.assert_not_called()


@pytest.mark.parametrize("path", ["alass", "large_offset", "constant_shift", "already_aligned", "timeline_affine", "pal", "hybrid"])
@pytest.mark.parametrize("accepted", [False, True])
async def test_every_serving_path_selects_unified(enabled, monkeypatch, path, accepted):
    offset = 0 if path == "already_aligned" else 96_000 if path == "large_offset" else 5000
    target, reference = _linear_pair(count=120 if path == "large_offset" else 36, offset_ms=offset)
    candidate = target if path == "already_aligned" else _true_candidate(target, 1, offset)
    if path == "large_offset":
        from tests.test_hybrid_predelivery import _candidate, _reference, _target

        target, reference, candidate = _target(), _reference(), _candidate()
    legacy = _legacy(not accepted)
    selected = to_evaluation(UnifiedVerdict(proposed_verdict="VERIFIED" if accepted else "REJECTED",
                                          independent_evidence_count=14))
    if accepted and path == "already_aligned":
        selected.set_verdict(SyncState.VERIFIED_SYNCED, VerificationAvailability.VERIFIED)
    review = AsyncMock(return_value=(selected, {"proceed": accepted, "verification_method": "unified"}))
    monkeypatch.setattr(om, "unified_review_async", review)
    monkeypatch.setattr(router, "unified_review_async", review)
    resolved = ResolvedReference(reference, kind="hash", candidate=META["target_filename"],
                                 reference_trust="strong")
    strategy = SimpleNamespace(resolve_with_provenance=AsyncMock(return_value=resolved))
    service = SimpleNamespace(sync_async=AsyncMock(return_value=candidate))
    cache = SyncCache()
    orch = om.SyncOrchestrator(external_strategy=strategy, sync_service=service, sync_cache=cache)
    orch._analyzer = SimpleNamespace(analyze=lambda *a, **k: legacy)
    if path in {"alass", "large_offset", "constant_shift", "already_aligned"}:
        monkeypatch.setattr(orch, "_try_timeline_affine", AsyncMock(return_value=None))
        monkeypatch.setattr(orch, "_try_hybrid_sync", AsyncMock(return_value=None))
        monkeypatch.setattr(orch, "_try_pal_two_point_fallback", AsyncMock(return_value=None))
        # Two-Point primary is intentionally bypassed here: with the authority
        # mocked to always VERIFIED it would preempt ALASS on anchorable pairs
        # before the path under test executes. Priority is covered by the
        # dedicated Two-Point/large-offset suites; here each legacy path must
        # still select Unified.
        monkeypatch.setattr(orch, "_try_two_point_primary", AsyncMock(return_value=None))
        monkeypatch.setattr(om, "decide_anchor_preshift", lambda *a: SimpleNamespace(accepted=False, reason="test_disabled"))
        if path in {"alass", "large_offset", "constant_shift"}:
            monkeypatch.setattr(om, "measure_alignment_consistency", lambda *a, **k: None)
        decisions = []
        if path == "large_offset":
            def grant(*args):
                decisions.append(True)
                return om.LargeOffsetServingState.ALASS_CORRECTED_LARGE_OFFSET
            monkeypatch.setattr(om, "decide_large_offset_serving", grant)
        if path == "constant_shift":
            preshift = SimpleNamespace(shift_ms=5000, summary=lambda: "test preshift")
            decision = SimpleNamespace(accepted=True, preshift=preshift, summary=lambda: "test preshift")
            exemption = SimpleNamespace(exempted=True,
                state=om.ConstantShiftExemptionState.EXEMPT_CONSTANT_SHIFT,
                reason_codes=(), summary=lambda: "test exemption")
            monkeypatch.setattr(om, "decide_anchor_preshift", lambda *a: decision)
            monkeypatch.setattr(om, "preshift_text", lambda *a: target)
            monkeypatch.setattr(om, "decide_constant_shift_exemption", lambda *a: exemption)
        output = await orch.evaluate_and_sync(target.encode(), META, "sub", auto_sync=True)
        assert output == (candidate.encode() if accepted else target.encode())
        if path == "large_offset":
            assert not decisions, "obsolete large-offset policy must not run under Unified"
        if accepted:
            # Both positive early-cache returns must preserve the selected authority.
            calls = review.await_count
            assert await orch.evaluate_and_sync(target.encode(), META, "sub", auto_sync=True) == output
            assert review.await_count == calls
            # A corrupted payload must miss; the independently bound verdict
            # artifact can still take the measured-verdict early return.
            resolution = orch._flight_key(META, "sub", target.encode())
            await cache.set(resolution, b"corrupted payload")
            assert await orch.evaluate_and_sync(target.encode(), META, "sub", auto_sync=True) == output
            assert review.await_count == calls
        if path == "already_aligned":
            service.sync_async.assert_not_called()
    else:
        result = SimpleNamespace(accepted=True, transformed_text=candidate, evaluation=legacy,
                                 first=None, last=None, slope=1.0427, intercept_ms=5000, pairs_evaluated=1,
                                 reason_codes=())
        if path == "timeline_affine":
            monkeypatch.setattr("app.services.sync.timeline_affine.attempt_timeline_affine", lambda *a: result)
            output = await orch._try_timeline_affine(target, resolved, META, "sub", "hash", "rk", None)
        elif path == "pal":
            monkeypatch.setattr("app.services.sync.pal_two_point.attempt_pal_two_point", lambda *a: result)
            output = await orch._try_pal_two_point_fallback(target, reference, META, "sub", target.encode(),
                                                          "hash", "rk", None, resolved=resolved)
        else:
            output = await orch._try_hybrid_sync(target, reference, META, "sub", target.encode(),
                                               "hash", "rk", None, resolved=resolved)
        assert output == (candidate.encode() if accepted else None)
    assert review.await_count >= 1, f"{path} bypassed the authority"
    assert orch._last_evaluation.sync_state is selected.sync_state


async def test_old_caches_cannot_authorize_under_gate(enabled, monkeypatch):
    cache = SyncCache()
    target = b"original"
    monkeypatch.setattr(router, "_enabled", False)
    old_payload = om.SyncOrchestrator._flight_key(META, "sub", target)
    old_verdict = om.SyncOrchestrator._verdict_key(META, "hash")
    await cache.set(old_payload, b"legacy transformed")
    monkeypatch.setattr(router, "_enabled", True)
    assert om.SyncOrchestrator._flight_key(META, "sub", target) != old_payload
    new_verdict = om.SyncOrchestrator._verdict_key(META, "hash")
    assert new_verdict != old_verdict and new_verdict.startswith("verdict:8:timing-v3:")
    await cache.set_verdict(new_verdict, {**_legacy().model_dump(mode="json"), "target_binding": "exact"})
    assert await cache.get_verdict(new_verdict) is None


async def test_unified_primary_and_alias_require_methodology(enabled):
    cache = SyncCache()
    key = SyncCache.build_verdict_key("video", "target", "ara", methodology="timing-v3")
    alias = SyncCache.build_verdict_alias_key("video", "sub", methodology="timing-v3")
    payload = {**_legacy().model_dump(mode="json"), "target_binding": "exact",
               "verification_method": "unified", "verification_version": "timing-v3"}
    await cache.set_verdict(key, payload, alias_key=alias)
    assert (await cache.get_verdict(key))["engine_version"] == 8
    assert (await cache.get_verdict_by_ref("video", "sub", methodology="timing-v3"))["verification"] == "cached"
    assert await cache.get_verdict_by_ref("video", "sub") is None


async def test_legacy_error_is_diagnostic_only(enabled, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("legacy unavailable")

    orch = om.SyncOrchestrator()
    orch._analyzer = SimpleNamespace(analyze=broken)
    # Gate ON: legacy never runs, so no legacy error can break serving.
    assert orch._legacy_diagnostic(META).reasons == []
    # Gate OFF: a legacy failure is captured as a diagnostic reason.
    monkeypatch.setattr(router, "_enabled", False)
    assert "legacy_diagnostic_error:RuntimeError" in orch._legacy_diagnostic(META).reasons


async def test_unified_timeout_blocks_legacy_acceptance(enabled, monkeypatch):
    import time

    def slow(**kwargs):
        time.sleep(0.03)
        return _legacy(), {"proceed": True}

    monkeypatch.setattr(router, "unified_review", slow)
    proceed, telemetry = await router.pre_delivery_review_async(
        target_text="target", reference_text="reference", synced_text="candidate",
        meta=META, exemption="ordinary_alass", legacy_unverified=False, timeout_s=0.001,
    )
    assert proceed is False and telemetry["verdict"] == "ERROR"
    assert telemetry["served_sha"] == "not_served"


async def test_old_unverified_verdict_still_blocks(enabled, monkeypatch):
    import hashlib

    cache = SyncCache()
    target = b"original subtitle"
    content_hash = hashlib.sha256(target).hexdigest()[:16]
    new_key = om.SyncOrchestrator._verdict_key(META, content_hash)
    old_key = new_key.replace("verdict:8:timing-v3:", "verdict:7:")
    payload = _legacy().model_dump(mode="json")
    payload.update(sync_state="unverified", target_binding="exact")
    await cache.set_verdict(old_key, payload)
    strategy = SimpleNamespace(resolve_with_provenance=AsyncMock(side_effect=AssertionError("no discovery")))
    orch = om.SyncOrchestrator(sync_cache=cache, external_strategy=strategy,
                              sync_service=SimpleNamespace(sync_async=AsyncMock()))
    assert await orch.evaluate_and_sync(target, META, "sub", auto_sync=True) == target
    strategy.resolve_with_provenance.assert_not_called()


async def test_compatible_verdict_serves_original_without_work(enabled):
    """Fast path headline: proven-compatible serves ORIGINAL, zero sync work."""
    import hashlib

    from tests.test_unified_verification import _linear_pair

    target, _ = _linear_pair()
    original = target.encode()
    cache = SyncCache()
    key = om.SyncOrchestrator._verdict_key(META, hashlib.sha256(original).hexdigest()[:16])
    assert key is not None and "timing-v3" in key
    await cache.set_verdict(key, {
        "sync_state": "verified_synced", "verification": "verified",
        "target_binding": "exact", "verification_method": "unified",
        "verification_version": "timing-v3",
        "artifact_sha256": hashlib.sha256(original).hexdigest(),
        "reference_sha": "c" * 64,
    })
    strategy = SimpleNamespace(resolve_with_provenance=AsyncMock(side_effect=AssertionError("no discovery")))
    service = SimpleNamespace(sync_async=AsyncMock(side_effect=AssertionError("no alass")))
    orch = om.SyncOrchestrator(sync_cache=cache, external_strategy=strategy,
                              sync_service=service)
    assert await orch.evaluate_and_sync(original, META, "sub", auto_sync=True) == original
    strategy.resolve_with_provenance.assert_not_called()
    service.sync_async.assert_not_called()


async def test_cached_transformed_reused_without_recompute(enabled):
    """Verified transformed artifact HIT: reuse bytes, no discovery, no ALASS."""
    import hashlib

    from tests.test_unified_verification import _linear_pair, _true_candidate

    target, _ = _linear_pair()
    transformed = _true_candidate(target, 1.0, 5_000).encode()
    cache = SyncCache()
    orch = om.SyncOrchestrator(sync_cache=SyncCache())
    resolution = orch._flight_key(META, "sub", target.encode())
    reference_sha = hashlib.sha256(b"reference").hexdigest()
    await cache.set(resolution, transformed)
    await cache.set_meta(resolution, {
        "sync_state": "verified_resynced", "verification": "verified",
        "target_binding": "exact",
        "artifact_sha256": hashlib.sha256(transformed).hexdigest(),
        "reference_sha": reference_sha,
        "verification_method": "unified", "verification_version": "timing-v3"})
    strategy = SimpleNamespace(resolve_with_provenance=AsyncMock(side_effect=AssertionError("no discovery")))
    service = SimpleNamespace(sync_async=AsyncMock(side_effect=AssertionError("no alass")))
    orch = om.SyncOrchestrator(sync_cache=cache, external_strategy=strategy,
                              sync_service=service)
    out = await orch.evaluate_and_sync(target.encode(), META, "sub", auto_sync=True)
    assert bytes(out) == transformed
    strategy.resolve_with_provenance.assert_not_called()
    service.sync_async.assert_not_called()


@pytest.mark.parametrize("corrupt", [False, True])
async def test_download_fallback_uses_exact_unified_artifact(enabled, monkeypatch, corrupt):
    import hashlib

    from app import main

    cache = SyncCache()
    monkeypatch.setattr(main, "_sync_cache", cache)
    release = SimpleNamespace(provider="subsource", release_name="Film.srt", download_url="url")
    ref = main._fallback_candidate_ref(release, "subsource", None, None)
    fingerprint = cache.video_fingerprint_from_meta(META)
    key = cache.build_verdict_key(fingerprint, "target", "ara", methodology="timing-v3")
    alias = cache.build_verdict_alias_key(fingerprint, ref, methodology="timing-v3")
    artifact_key = "artifact:timing-v3"
    payload = {**_legacy().model_dump(mode="json"), "target_binding": "exact",
               "verification_method": "unified", "verification_version": "timing-v3",
               "reference_sha": "b" * 64,
               "artifact_key": artifact_key, "artifact_sha256": hashlib.sha256(b"correct").hexdigest()}
    await cache.set_verdict(key, payload, alias_key=alias)
    await cache.set(artifact_key, b"tampered" if corrupt else b"correct")
    monkeypatch.setattr(cache, "find_synced_for_target", AsyncMock(side_effect=AssertionError("no broad search")))
    output = await main._reusable_verified_fallback_sync(
        release=release, imdb_id=META["imdb_id"], season=None, episode=None, lang="ara",
        target_filename=META["target_filename"], video_hash=META["video_hash"],
    )
    assert output == (None if corrupt else b"correct")


async def test_listing_uses_methodology_alias(enabled):
    from app.main import _lookup_search_verdict

    cache = SimpleNamespace(get_verdict_by_ref=AsyncMock(return_value=None))
    release = SimpleNamespace(provider="subdl", download_url="url")
    assert await _lookup_search_verdict(cache, "video", "sub", release,
                                       methodology="timing-v3") is None
    cache.get_verdict_by_ref.assert_awaited_once_with("video", "sub", methodology="timing-v3")


async def test_legacy_broad_scans_cannot_find_unified_artifacts():
    cache = SyncCache()
    legacy_key = SyncCache.build_key("tt1", "movie", "video", "sub", "edition", "target")
    await cache.set(f"{legacy_key}:timing-v3", b"unified")
    await cache.set(f"{legacy_key}:timing-v3:v3:True:context:auth", b"unified resolution")
    assert await cache.find_synced_for("tt1", "sub") is None
    assert await cache.find_synced_for_target("tt1", "movie", "video", "sub") is None
    await cache.set(legacy_key, b"legacy")
    assert await cache.find_synced_for("tt1", "sub") == b"legacy"
    assert await cache.find_synced_for_target("tt1", "movie", "video", "sub") == b"legacy"
