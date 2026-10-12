"""Stremio response waits for sync completion instead of returning stale bytes."""
import asyncio
from types import SimpleNamespace

import pytest

from app import main
from app.config import settings


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [b"verified transformed", b"original"])
async def test_initial_deadline_keeps_same_response_pending(monkeypatch, result):
    ready = asyncio.Event()
    async def sync(*args, **kwargs):
        await ready.wait()
        return result
    monkeypatch.setattr(main, "_get_sync_orchestrator", lambda: SimpleNamespace(evaluate_and_sync=sync))
    monkeypatch.setattr(settings, "SYNC_TOTAL_REQUEST_BUDGET", 0.001)
    monkeypatch.setattr(settings, "SYNC_COMPLETION_REQUEST_BUDGET", 1.0)
    request = asyncio.create_task(main._maybe_sync_subtitle(b"original", {}, "test", True))
    try:
        await asyncio.sleep(0.03)
        assert not request.done()
        ready.set()
        assert await asyncio.wait_for(request, 1.0) == result
    finally:
        request.cancel()
        await asyncio.gather(request, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancellation_during_completion_wait_cancels_sync(monkeypatch):
    cancelled = asyncio.Event()
    async def sync(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    monkeypatch.setattr(main, "_get_sync_orchestrator", lambda: SimpleNamespace(evaluate_and_sync=sync))
    monkeypatch.setattr(settings, "SYNC_TOTAL_REQUEST_BUDGET", 0.001)
    monkeypatch.setattr(settings, "SYNC_COMPLETION_REQUEST_BUDGET", 1.0)
    request = asyncio.create_task(main._maybe_sync_subtitle(b"original", {}, "test", True))
    await asyncio.sleep(0.03)
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    await asyncio.wait_for(cancelled.wait(), 1.0)


@pytest.mark.asyncio
async def test_hard_deadline_retains_bounded_fallback_and_consumes_background_error(monkeypatch):
    ready = asyncio.Event()
    finished = asyncio.Event()
    async def sync(*args, **kwargs):
        try:
            await ready.wait()
            raise RuntimeError("diagnostic failure")
        finally:
            finished.set()
    monkeypatch.setattr(main, "_get_sync_orchestrator", lambda: SimpleNamespace(evaluate_and_sync=sync))
    monkeypatch.setattr(settings, "SYNC_TOTAL_REQUEST_BUDGET", 0.001)
    monkeypatch.setattr(settings, "SYNC_COMPLETION_REQUEST_BUDGET", 0.01)
    assert await main._maybe_sync_subtitle(b"original", {}, "test", True) == b"original"
    ready.set()
    await asyncio.wait_for(finished.wait(), 1.0)
    await asyncio.sleep(0)
