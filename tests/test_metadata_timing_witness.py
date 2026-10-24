"""Focused local-witness compatibility for the integrated metadata gate."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.sync import hybrid_router as router
from app.services.sync import orchestrator as om
from app.services.sync import timeline_affine
from app.services.sync.cache import ReferenceDiskCache
from app.services.sync.orchestrator import SyncOrchestrator
from app.services.sync.query import ResolvedReference
from app.services.sync.timeline_affine import AffineResult
from app.services.sync_cache import SyncCache
from two_point_sync.parser import Cue, format_srt

_DIALOGUE = "The lighthouse keeper arrives at dawn and waits for the tide"
_VIDEO = "Film.2020.1080p.BluRay.x264-GRP.mkv"
_SUBTITLE = "Film.2020.1080p.BluRay.x264-GRP.srt"


def _srt(count: int = 40, step_ms: int = 4_000, *, start: int = 0,
         text: str = _DIALOGUE) -> str:
    cues = [
        Cue(number=index + 1, order=index + 1,
            start_ms=start + index * step_ms, end_ms=start + index * step_ms + 2_000,
            text=f"{text} number {index + 1}")
        for index in range(count)
    ]
    return format_srt(cues)



def _shifted(text: str, offset_ms: int, *, count: int | None = None) -> str:
    from two_point_sync.parser import parse_srt

    cues = parse_srt(text)
    if count is not None:
        cues = cues[:count]
    return format_srt([
        type(cue)(number=cue.number, order=cue.order,
                  start_ms=cue.start_ms + offset_ms, end_ms=cue.end_ms + offset_ms,
                  text=cue.text)
        for cue in cues
    ])



def _meta(**extra):
    base = {
        "imdb_id": "tt1",
        "lang": "ara",
        "media_type": "movie",
        "target_filename": _VIDEO,
        "release_name": _SUBTITLE,
        "video_hash": "a" * 16,
    }
    base.update(extra)
    return base



@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(router, "_enabled", True)
    monkeypatch.delenv(router.TEST_SCOPE_ENV, raising=False)
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True)



def _gate_orchestrator(tmp_path, monkeypatch, *, witness: str | None,
                       candidate: str | None = None):
    """FAST_ORIGINAL metadata pair, optional local witness, recording pipeline."""
    monkeypatch.setattr(timeline_affine, "attempt_timeline_affine",
                        lambda *a, **k: AffineResult())
    resolve = AsyncMock(return_value=ResolvedReference(None))
    sync = SimpleNamespace(sync_async=AsyncMock(return_value=candidate))
    reference_cache = ReferenceDiskCache(tmp_path / "refs", min_bytes=1_000)
    strategy = SimpleNamespace(resolve_with_provenance=resolve, cache=reference_cache)
    orch = SyncOrchestrator(sync_cache=SyncCache(), external_strategy=strategy,
                            sync_service=sync)
    if witness is not None:
        target = _srt(40, 4_000)
        query = orch._build_query(_meta(), "sub", target)
        reference_cache.set(query, "subdl", witness, kind="edition")
    return orch, resolve, sync



def _assert_fast_gate():
    from app.services.sync.metadata_gate import decide_metadata_gate

    decision = decide_metadata_gate(_VIDEO, _SUBTITLE)
    assert decision.fast, decision



async def test_fast_original_with_mistimed_local_witness_investigates(tmp_path, monkeypatch, enabled, caplog):
    """Compatible release names do not prove the timings align.

    The local witness is free (no provider request); when it measurably
    contradicts the target, the request must be investigated rather than
    delivered untouched on metadata alone.
    """
    import logging
    caplog.set_level(logging.INFO)
    _assert_fast_gate()
    target = _srt(40, 4_000)
    witness = _shifted(target, 8_000)
    orch, resolve, sync = _gate_orchestrator(
        tmp_path, monkeypatch, witness=witness, candidate=witness)

    await orch.evaluate_and_sync(target.encode(), _meta(), "sub", auto_sync=True)

    assert "metadata gate deferred" in caplog.text
    assert "timing_witness_conflict" in caplog.text
    assert "metadata_gate_timing_witness=subtitle_cache" in caplog.text



def test_timing_witness_decision_measures_the_displacement(tmp_path, monkeypatch):
    """The decision itself: a measured displacement is a contradiction."""
    _assert_fast_gate()
    target = _srt(40, 4_000)
    orch, _resolve, _sync = _gate_orchestrator(
        tmp_path, monkeypatch, witness=_shifted(target, 8_000))
    query = orch._build_query(_meta(), "sub", target)

    decision = orch._timing_witness_decision(target, query, _meta(), "sub")

    assert decision.investigate is True
    assert decision.reason == "timing_witness_conflict"
    assert decision.witness == "subtitle_cache"
    assert abs(decision.median_offset_ms or 0.0) >= 1_000



async def test_fast_original_with_aligned_witness_stays_fast(tmp_path, monkeypatch, enabled):
    """Control: measured agreement keeps the fast path and contacts nobody."""
    _assert_fast_gate()
    target = _srt(40, 4_000)
    orch, resolve, sync = _gate_orchestrator(
        tmp_path, monkeypatch, witness=_shifted(target, 0), candidate=None)

    result = await orch.evaluate_and_sync(target.encode(), _meta(), "sub", auto_sync=True)

    assert bytes(result) == target.encode()
    assert resolve.await_count == 0
    assert sync.sync_async.await_count == 0



async def test_fast_original_without_witness_contacts_no_provider(tmp_path, monkeypatch, enabled):
    """Control: no local evidence means no provider work (item 4)."""
    _assert_fast_gate()
    target = _srt(40, 4_000)
    orch, resolve, sync = _gate_orchestrator(tmp_path, monkeypatch, witness=None)

    result = await orch.evaluate_and_sync(target.encode(), _meta(), "sub", auto_sync=True)

    assert bytes(result) == target.encode()
    assert resolve.await_count == 0
    assert sync.sync_async.await_count == 0



async def test_fast_original_with_insufficient_witness_serves_original(tmp_path, monkeypatch, enabled):
    """Control: missing evidence is not a contradiction (item 6).

    A three-cue witness cannot support any alignment claim in either direction.
    """
    _assert_fast_gate()
    target = _srt(40, 4_000)
    orch, resolve, sync = _gate_orchestrator(
        tmp_path, monkeypatch, witness=_shifted(target, 9_000, count=3))

    result = await orch.evaluate_and_sync(target.encode(), _meta(), "sub", auto_sync=True)

    assert bytes(result) == target.encode()
    assert resolve.await_count == 0
