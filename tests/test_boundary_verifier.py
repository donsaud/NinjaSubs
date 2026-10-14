"""Timing-only regression, adversarial controls and observational isolation."""
from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.sync import boundary_verifier as bv
from app.services.sync.alignment import AlignmentAnalyzer, SubtitleEvaluation, is_reusable_verified
from app.services.sync.ninja_points import render_srt
from app.services.sync.orchestrator import SyncOrchestrator
from app.services.sync.query import ResolvedReference
from app.services.sync.timeline_affine import attempt_timeline_affine
from app.services.sync_cache import SyncCache

FIXTURE = Path(__file__).parent / "fixtures" / "boundary" / "dexter_s08e04_timings.json"


@pytest.fixture(scope="module")
def dexter():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return {key: render_srt([(s, e, "Dialogue event." if dialogue else "Translated by: uploader")
                            for s, e, dialogue in value["cues"]])
            for key, value in data["files"].items()}


def case():
    target = []
    cursor = 200000
    for i in range(120):
        end = cursor+1200+(i % 3)*400
        target.append((cursor, end, "Target dialogue."))
        cursor = end + (100 if i % 2 == 0 else 4000+(i % 7)*500)
    output = [(s+(-150000 if i<60 else 5000), e+(-150000 if i<60 else 5000), text)
              for i, (s, e, text) in enumerate(target)]
    reference = [(s, e, "Reference in another language.") for s, e, _ in output]
    return target, reference, output


def verify(cues):
    return bv.verify_boundary(*(render_srt(c) for c in cues))


@pytest.mark.parametrize("artifact", ["raw_transformed", "final_served"])
def test_dexter_fixed_artifact_independently_supported(dexter, artifact):
    output = dexter["transformed" if artifact == "raw_transformed" else "served"]
    result = bv.verify_boundary(dexter["target"], dexter["reference"], output, artifact=artifact)
    assert result.accepted, asdict(result)
    assert result.classification == "EDGE_DISCONTINUITY"
    assert result.artifact == artifact
    assert result.boundary_ms == 193890
    assert result.boundary_interval_ms == (187250, 193890)
    assert result.offset_a_ms == -96961
    assert result.offset_b_ms == 9661
    assert result.jump_ms == 106622
    assert result.gap_disagreement_ms == 58
    assert [s.support for s in result.segments] == [38, 739]
    assert [s.runs for s in result.segments] == [4, 124]
    assert all(s.residual_p95_ms == s.residual_mad_ms == 0 for s in result.segments)
    assert min(min(s.windows) for s in result.segments) > .95
    assert result.improvement == 1
    assert result.holdout_pass
    assert result.holdout_support == (15, 352)
    assert result.holdout_p95_ms == (0, 0)


def test_dexter_legacy_p95_unchanged(dexter):
    result = AlignmentAnalyzer().analyze(dexter["target"], dexter["transformed"],
        dexter["reference"], alass_applied=True, alass_successful=True,
        reference_trust="high", max_plausible_offset_ms=180000)
    assert result.p95_offset_ms == 4489
    assert result.structural_similarity == .72
    assert result.sync_state.value == "unverified"
    assert result.verification.value == "unknown"
    assert result.cut_verdict == "piecewise"


@pytest.mark.parametrize("segmentation", ["one_to_one", "split", "merge"])
def test_reference_segmentation_and_language_do_not_control_acceptance(segmentation):
    target, reference, output = case()
    if segmentation == "split":
        reference = [cue for s, e, text in reference for cue in
                     [(s, (s+e)//2, text), ((s+e)//2, e, text)]]
    elif segmentation == "merge":
        reference = [(reference[i][0], reference[i+1][1], "Other language.")
                     for i in range(0, len(reference), 2)]
    result = verify((target, reference, output))
    assert result.accepted, result


@pytest.mark.parametrize("control", ["wrong_reference", "tiny_segment", "unsupported_boundary",
    "three_regimes", "ordering", "downward", "sparse", "unsupported_interval", "crossing_cue",
    "short_span"])
def test_negative_controls(control):
    target, reference, output = case()
    if control == "wrong_reference":
        reference = [(s+17000, e+17000, text) for s, e, text in reference]
    elif control == "tiny_segment":
        output = [(s+(-150000 if i<4 else 5000), e+(-150000 if i<4 else 5000), text)
                  for i, (s, e, text) in enumerate(target)]
        reference = output.copy()
    elif control == "unsupported_boundary":
        # Fixed output retains its jump, while the reference has no supporting gap.
        reference = [(s, e+100000, text) if i == 59 else (s, e, text)
                     for i, (s, e, text) in enumerate(reference)]
    elif control == "three_regimes":
        output = [(s+(10000 if i>=90 else 0), e+(10000 if i>=90 else 0), text)
                  for i, (s, e, text) in enumerate(output)]
        reference = output.copy()
    elif control == "ordering":
        output[20], output[21] = output[21], output[20]
    elif control == "downward":
        # A large target gap prevents sorting alone from exposing a downward jump.
        target = [(s+(200000 if i>=60 else 0), e+(200000 if i>=60 else 0), text)
                  for i, (s, e, text) in enumerate(target)]
    elif control == "sparse":
        target, reference, output = target[:8], reference[:8], output[:8]
    elif control == "unsupported_interval":
        reference = reference[:22]+reference[28:]
    elif control == "crossing_cue":
        s, _, text = target[59]
        target[59] = (s, target[60][0]+1000, text)
    elif control == "short_span":
        target[:60] = [(200000+i*500, 200300+i*500, text)
                       for i, (_, _, text) in enumerate(target[:60])]
        output = [(s+(-150000 if i<60 else 5000), e+(-150000 if i<60 else 5000), text)
                  for i, (s, e, text) in enumerate(target)]
        reference = output.copy()
    result = verify((target, reference, output))
    assert not result.accepted, (control, result)
    assert result.reasons


def test_small_third_regime_cannot_hide_below_p95():
    target, reference, output = case()
    output[10:12] = [(s+1000, e+1000, text) for s, e, text in output[10:12]]
    result = verify((target, reference, output))
    assert not result.accepted
    assert result.reasons == ("unexplained_regime_or_movement",)


def test_long_unsupported_dialogue_cannot_hide_inside_good_windows():
    target, _, _ = case()
    s, _, text = target[31]
    target[31] = (s, s+7000, text)
    output = [(s+(-150000 if i<60 else 5000), e+(-150000 if i<60 else 5000), text)
              for i, (s, e, text) in enumerate(target)]
    reference = output[:31]+output[32:]
    result = verify((target, reference, output))
    assert not result.accepted
    assert result.reasons == ("unsupported_dialogue_interval",)


def test_wrong_reference_with_correct_boundary_is_still_rejected():
    target, reference, output = case()
    reference = [(s+(3000 if i % 60 in range(2, 58) else 0),
                  e+(3000 if i % 60 in range(2, 58) else 0), text)
                 for i, (s, e, text) in enumerate(reference)]
    result = verify((target, reference, output))
    assert result.gap_disagreement_ms == 0
    assert not result.accepted
    assert result.reasons == ("poor_reference_agreement",)


def test_holdout_gate_is_required_even_when_full_evidence_passes(monkeypatch):
    target, _, _ = case()
    opening = [(200000+i*1100, 201000+i*1100, "Dialogue.") for i in range(14)]
    opening += [(230000, 245000, "Dialogue."), (245100, 250100, "Dialogue.")]
    opening += [(270000+i*1100, 271000+i*1100, "Dialogue.") for i in range(14)]
    target = opening + target[60:]
    output = [(s+(-150000 if i<30 else 5000), e+(-150000 if i<30 else 5000), text)
              for i, (s, e, text) in enumerate(target)]
    result = verify((target, output, output))
    assert result.reasons == ("holdout_sparse",)
    # A targeted test-only mutation proves the negative reaches the holdout gate.
    require = bv._require
    monkeypatch.setattr(bv, "_require", lambda ok, reason: None if reason == "holdout_sparse"
                        else require(ok, reason))
    assert verify((target, output, output)).accepted


def test_two_segments_without_strong_simpler_model_improvement_are_refused():
    target, _, _ = case()
    output = [(s+(-150000 if i<20 else -144000)+(300 if i % 2 == 0 else -300),
               e+(-150000 if i<20 else -144000)+(300 if i % 2 == 0 else -300), text)
              for i, (s, e, text) in enumerate(target)]
    result = verify((target, output, output))
    assert result.reasons == ("insufficient_improvement",)


def test_budget_and_input_limits_fail_closed(monkeypatch):
    monkeypatch.setattr(bv, "BUDGET_SECONDS", -1)
    assert verify(case()).reasons == ("budget_exceeded",)
    assert bv.verify_boundary("x"*(bv.MAX_CHARS+1), "", "").reasons == ("input_limit",)


def test_whiplash_no_unnecessary_escalation():
    from tests.private_fixtures import read_subtitle_text

    folder = Path(__file__).parent / "fixtures" / "whiplash"
    target, reference = (read_subtitle_text(folder / name)
                         for name in ("target.srt", "reference.srt"))
    affine = attempt_timeline_affine(target, reference)
    assert affine.accepted
    assert affine.diagnostics.classification == "PAL_LIKE_AFFINE"
    assert bv.verify_boundary(target, reference, affine.transformed_text,
                              global_verified=True).reasons == ("simpler_verified_model",)
    assert not bv.verify_boundary(target, reference, affine.transformed_text).accepted


def evaluation():
    return SubtitleEvaluation(alass_applied=True, alass_successful=True,
        cut_verdict="piecewise", rejection_reason="low_confidence")


async def test_shadow_is_nonblocking_bounded_and_cannot_mutate_verdict(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []
    def blocked(*args, **kwargs):
        calls.append(args)
        entered.set()
        release.wait(timeout=5)
        return bv.BoundaryResult(True, "EDGE_DISCONTINUITY")
    monkeypatch.setattr(bv, "verify_boundary", blocked)
    shadow, verdict = bv.BoundaryShadow(), evaluation()
    # Production binding annotation is independent of the shadow measurement.
    verdict.target_binding = "exact"
    verdict.target_binding_reasons = ["moviehash_exact"]
    before = verdict.model_dump()
    try:
        assert shadow.schedule("pristine", "reference", "fixed", verdict) is None
        shadow.schedule("second", "reference", "fixed", verdict)
        assert len(shadow.tasks) == 1
        await asyncio.to_thread(entered.wait, 2)
        assert entered.is_set()
        assert len(calls) == 1
        assert verdict.model_dump() == before
    finally:
        release.set()
        await asyncio.gather(*shadow.tasks)
        await shadow.close()
    assert not shadow.tasks
    assert verdict.model_dump() == before


async def test_shutdown_drains_running_worker_and_blocks_rescheduling(monkeypatch):
    entered, cancel_seen, release, finished = (threading.Event() for _ in range(4))
    worker_threads = []
    def blocked(*args, cancelled):
        worker_threads.append(threading.current_thread())
        entered.set()
        try:
            assert cancelled.wait(timeout=3)
            cancel_seen.set()
            release.wait(timeout=3)
            raise RuntimeError("PRIVATE_WORKER_ERROR")
        finally:
            finished.set()
    monkeypatch.setattr(bv, "verify_boundary", blocked)
    shadow = bv.BoundaryShadow()
    shadow.schedule("target", "reference", "output", evaluation())
    assert await asyncio.to_thread(entered.wait, 2)
    job = next(iter(shadow.tasks))
    close = asyncio.create_task(shadow.close())
    try:
        assert await asyncio.to_thread(cancel_seen.wait, 2)
        assert not close.done() and not job.done()
        job.cancel()  # Repeated cancellation must not orphan the worker.
        await asyncio.sleep(0)
        assert not job.done()
        assert len(shadow.tasks) == 1
        shadow.schedule("other", "reference", "output", evaluation())
        assert len(shadow.tasks) == 1
    finally:
        release.set()
        await close
    assert finished.is_set() and job.cancelled()
    assert not shadow.tasks
    assert all(not worker.is_alive() for worker in worker_threads)
    await shadow.close()  # Idempotent and permanently closed.
    shadow.schedule("target", "reference", "output", evaluation())
    assert not shadow.tasks


async def test_private_executor_does_not_inherit_request_telemetry(monkeypatch):
    from app.services.sync.telemetry import SyncTiming, current_sync_timing
    timing = SyncTiming()
    seen = []
    def worker(*args, **kwargs):
        seen.append((threading.current_thread().name, current_sync_timing.get()))
        return bv.BoundaryResult()
    monkeypatch.setattr(bv, "verify_boundary", worker)
    token = current_sync_timing.set(timing)
    shadow = bv.BoundaryShadow()
    try:
        shadow.schedule("target", "reference", "output", evaluation())
        await asyncio.gather(*shadow.tasks)
        assert current_sync_timing.get() is timing
        assert seen and seen[0][0].startswith("boundary-shadow")
        assert seen[0][1] is None
    finally:
        current_sync_timing.reset(token)
        await shadow.close()


def test_scheduling_outside_running_loop_has_no_unawaited_coroutine(recwarn):
    shadow = bv.BoundaryShadow()
    shadow.schedule("target", "reference", "output", evaluation())
    assert not shadow.tasks
    assert not recwarn.list


async def test_owner_retrieves_unexpected_task_failures(monkeypatch):
    async def failed(*args):
        raise RuntimeError("PRIVATE_TASK_ERROR")
    loop = asyncio.get_running_loop()
    errors = []
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda *args: errors.append(args))
    shadow = bv.BoundaryShadow()
    monkeypatch.setattr(shadow, "_run", failed)
    try:
        shadow.schedule("target", "reference", "output", evaluation())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not shadow.tasks
        assert not errors
    finally:
        await shadow.close()
        loop.set_exception_handler(previous)


def test_cooperative_cancellation_refuses_work_before_parsing():
    cancelled = threading.Event()
    cancelled.set()
    result = bv.verify_boundary("", "", "", cancelled=cancelled)
    assert result.reasons == ("cancelled",)


async def test_large_offset_delivers_fixed_dexter_while_shadow_worker_is_blocked(
        dexter, monkeypatch):
    from app.services.sync import orchestrator as om
    entered, release, finished = (threading.Event() for _ in range(3))
    def blocked(*args, **kwargs):
        entered.set()
        try:
            release.wait(timeout=5)
            return bv.BoundaryResult(True, "EDGE_DISCONTINUITY")
        finally:
            finished.set()
    monkeypatch.setattr(bv, "verify_boundary", blocked)
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True)
    async def resolve(*args, **kwargs):
        return ResolvedReference(dexter["reference"], kind="hash",
                                 reference_trust="high", bluray_match=True)
    async def sync(*args, **kwargs):
        return dexter["transformed"]
    orch = SyncOrchestrator(hash_reference_strategy=SimpleNamespace(resolve_with_provenance=resolve),
        sync_service=SimpleNamespace(sync_async=sync), sync_cache=SyncCache())
    meta = {"imdb_id": "tt0773262", "lang": "ara", "media_type": "series",
            "season": 8, "episode": 4, "target_filename": "Dexter.S08E04.BluRay.REMUX.mkv",
            "release_name": "Dexter.S08E04.720p.HDTV.x264-EVOLVE.srt",
            "video_hash": "1234567890abcdef", "video_size": 123456}
    response = asyncio.create_task(orch.evaluate_and_sync(
        dexter["target"].encode(), meta, "dexter-fixture", auto_sync=True))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        served = await asyncio.wait_for(asyncio.shield(response), 2)
        assert served == dexter["transformed"].encode()
        assert not finished.is_set()
        assert orch._last_evaluation.p95_offset_ms == 4489
        assert orch._last_evaluation.sync_state.value == "unverified"
        assert orch._last_evaluation.verification.value == "unknown"
    finally:
        release.set()
        await response
        await orch.close()
    assert finished.is_set() and not orch._boundary_shadow.tasks


async def test_shadow_exception_redaction_and_eligibility(monkeypatch, caplog):
    def failed(*args, **kwargs):
        raise RuntimeError("PRIVATE_DIALOGUE https://private.invalid/?key=SECRET")
    monkeypatch.setattr(bv, "verify_boundary", failed)
    caplog.set_level("INFO", logger=bv.__name__)
    shadow = bv.BoundaryShadow()
    shadow.schedule("PRIVATE_DIALOGUE", "PRIVATE_URL", "PRIVATE_CONFIG", evaluation())
    await asyncio.gather(*shadow.tasks)
    assert "reason=evaluation_failed" in caplog.text
    assert "PRIVATE" not in caplog.text and "SECRET" not in caplog.text
    verdict = evaluation()
    verdict.set_verdict("verified_resynced", "verified")
    shadow.schedule("", "", "", verdict)
    assert not shadow.tasks
    shadow.schedule("", "", "", SubtitleEvaluation())
    assert not shadow.tasks


async def test_dexter_safe_shadow_log_has_artifact_label(dexter, caplog):
    caplog.set_level("INFO", logger=bv.__name__)
    shadow = bv.BoundaryShadow()
    shadow.schedule(dexter["target"], dexter["reference"], dexter["transformed"], evaluation())
    await asyncio.gather(*shadow.tasks)
    assert "accepted=true classification=EDGE_DISCONTINUITY artifact=raw_transformed" in caplog.text
    assert "offset_a_ms=-96961" in caplog.text
    assert "holdout_pass=true" in caplog.text
    assert "Dialogue" not in caplog.text and "uploader" not in caplog.text


@pytest.mark.parametrize("accepted", [True, False])
async def test_orchestrator_shadow_does_not_change_fallback_cache_or_verdict(monkeypatch, accepted):
    from app.services.sync import orchestrator as om
    target, reference, output = case()
    output = [(s+(5000 if i<60 else 12000), e+(5000 if i<60 else 12000), text)
              for i, (s, e, text) in enumerate(target)]
    reference = [(s, e, "Independent reference " + text) for s, e, text in output]
    original, transformed = render_srt(target).encode(), render_srt(output)
    async def resolve(query, **kwargs):
        return ResolvedReference(render_srt(reference), kind="hash", reference_trust="high")
    async def sync(*args, **kwargs):
        return transformed
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True)
    monkeypatch.setattr(om, "measure_alignment_consistency", lambda *a, **kw: None)
    cache = SyncCache()
    cache_write = AsyncMock(wraps=cache.set)
    monkeypatch.setattr(cache, "set", cache_write)
    orch = SyncOrchestrator(hash_reference_strategy=SimpleNamespace(resolve_with_provenance=resolve),
        sync_service=SimpleNamespace(sync_async=sync), sync_cache=cache)
    async def skip(*args, **kwargs):
        return None
    monkeypatch.setattr(orch, "_try_timeline_affine", skip)
    monkeypatch.setattr(orch, "_try_pal_two_point_fallback", skip)
    verdict = evaluation()
    verdict.mad_offset_ms = 1000  # Refuse Large Offset exception as well.
    monkeypatch.setattr(orch._analyzer, "analyze", lambda *a, **kw: verdict)
    seen = []
    def check(pristine, ref, fixed, **kwargs):
        seen.append((pristine, ref, fixed))
        return bv.BoundaryResult(accepted, "EDGE_DISCONTINUITY" if accepted else "UNSUPPORTED")
    monkeypatch.setattr(bv, "verify_boundary", check)
    # Production binding annotation is independent of the shadow measurement.
    verdict.target_binding = "exact"
    verdict.target_binding_reasons = ["moviehash_exact"]
    before = verdict.model_dump()
    meta = {"imdb_id": "tt1234567", "lang": "ara", "target_filename": "Series.BluRay.mkv",
            "media_type": "series", "season": 8, "episode": 4,
            "video_hash": "1234567890abcdef", "video_size": 123456}
    served = await orch.evaluate_and_sync(original, meta, "target", auto_sync=True)
    await asyncio.gather(*orch._boundary_shadow.tasks)
    assert served == original
    assert verdict.model_dump() == before
    assert seen and seen[0][0] == original.decode()
    assert seen[0][2] == transformed
    assert orch.metrics["alass_runs"] == 1
    # The existing path stores rejected artifacts with non-reusable metadata.
    assert cache_write.await_count == 2
    for call in cache_write.await_args_list:
        metadata = await cache.get_meta(call.args[0])
        assert metadata["sync_state"] == "unverified"
        assert metadata["verification"] == "unknown"
        assert metadata["serving_state"] == "original"
        assert not is_reusable_verified(metadata["sync_state"], metadata["verification"])
    await orch.close()
