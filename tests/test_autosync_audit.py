"""Regressions found by the autosync audit, exercised at cache and serving seams."""

import asyncio
import os
import shutil
import subprocess
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from starlette.requests import Request

from app import main
from app.config import settings
from app.services.sync.cache import ReferenceDiskCache
from app.services.sync.external_strategy import ExternalExactStrategy
from app.services.sync.orchestrator import SyncOrchestrator
from app.services.sync.query import ReferenceQuery, ResolvedReference
from app.services.sync_service import SubtitleSyncService
from app.utils.config_parser import encode_user_config


def _srt(shifts=None):
    shifts = shifts or [0] * 10
    blocks = []
    for index, shift in enumerate(shifts):
        seconds = 10 + index * 3 + shift
        blocks.append(
            f"{index + 1}\n00:{seconds // 60:02d}:{seconds % 60:02d},000 --> "
            f"00:{(seconds + 1) // 60:02d}:{(seconds + 1) % 60:02d},000\nمرحبا"
        )
    return "\n\n".join(blocks) + "\n"


@pytest.fixture
def enabled_sync(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_SUBTITLE_SYNC", True)


@pytest.mark.parametrize("change", [
    {"target_filename": "Movie.2020.1080p.WEB-DL-GRP.mkv"},
    {"target_filename": "Movie.2020.Extended.1080p.BluRay-GRP.mkv"},
    {"video_hash": "bbbbbbbbbbbbbbbb"},
    # NB: stream_url is intentionally NOT part of the cache identity — debrid
    # tokens are freshly signed per request.
])
def test_reference_cache_does_not_share_group_across_media(tmp_path, change):
    cache = ReferenceDiskCache(tmp_path, min_bytes=10)
    query = ReferenceQuery(
        imdb_id="tt1", target_filename="Movie.2020.1080p.BluRay-GRP.mkv",
        video_hash="aaaaaaaaaaaaaaaa", stream_url="https://cdn.example/first.mkv",
    )
    cache.set(query, "subdl", _srt(), kind="team")
    assert cache.get(query) is not None
    assert cache.get(replace(query, **change)) is None


def test_edition_cache_write_preserves_exact_reference_provenance(tmp_path):
    cache = ReferenceDiskCache(tmp_path, min_bytes=10)
    query = ReferenceQuery(imdb_id="tt1", target_filename="Movie.2020.BluRay-GRP.mkv")
    cache.set(query, "subsource", _srt(), kind="team", bluray_match=True, candidate="exact")
    cache.set(query, "subdl", _srt(), kind="edition")
    assert cache.get(query) == ResolvedReference(_srt(), "team", True, "exact")


def _meta(**overrides):
    return {"imdb_id": "tt1", "lang": "ara", **overrides}


def _orch(strategy, service=None):
    return SyncOrchestrator(
        external_strategy=strategy,
        sync_service=service or SimpleNamespace(sync_async=AsyncMock(return_value=_srt())),
        sync_cache=main.SyncCache(),
    )


@pytest.mark.asyncio
async def test_cancelling_waiter_keeps_running_flight_registered(enabled_sync):
    entered, release = asyncio.Event(), asyncio.Event()

    async def resolve(query):
        entered.set()
        await release.wait()
        return ResolvedReference(_srt(), "team")

    strategy = SimpleNamespace(resolve_with_provenance=AsyncMock(side_effect=resolve))
    orch = _orch(strategy)
    first = asyncio.create_task(orch.evaluate_and_sync(_srt().encode(), _meta(), "t", True))
    await entered.wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    still_registered = len(orch._inflight)
    second = asyncio.create_task(orch.evaluate_and_sync(_srt().encode(), _meta(), "t", True))
    await asyncio.sleep(0)
    release.set()
    await second
    await asyncio.sleep(0)
    assert still_registered == 1
    assert strategy.resolve_with_provenance.await_count == 1
    assert orch._inflight == {}


@pytest.mark.asyncio
async def test_warm_sync_cache_survives_reference_provider_outage(enabled_sync):
    strategy = SimpleNamespace(resolve_with_provenance=AsyncMock(
        return_value=ResolvedReference(_srt([1] * 10), "team")
    ))
    service = SimpleNamespace(sync_async=AsyncMock(return_value=_srt([1] * 10)))
    orch = _orch(strategy, service)
    original = _srt().encode()
    synced = await orch.evaluate_and_sync(original, _meta(), "t", True)
    strategy.resolve_with_provenance.return_value = ResolvedReference(None)
    assert await orch.evaluate_and_sync(original, _meta(), "t", True) == synced
    assert synced != original
    assert strategy.resolve_with_provenance.await_count == 1


@pytest.mark.asyncio
async def test_same_group_different_cut_does_not_bypass_sync(enabled_sync):
    strategy = SimpleNamespace(resolve_with_provenance=AsyncMock(
        return_value=ResolvedReference(_srt(), "team")
    ))
    orch = _orch(strategy)
    await orch.evaluate_and_sync(_srt().encode(), _meta(
        target_filename="Movie.2020.Extended.1080p.BluRay-GRP.mkv",
        release_name="Movie.2020.Theatrical.1080p.BluRay-GRP.srt",
    ), "t", True)
    assert strategy.resolve_with_provenance.await_count == 1


def test_request_playback_context_overrides_shared_search_metadata():
    meta = _meta(target_filename="Movie.2020.WEB-DL-OTHER.mkv", video_hash="old")
    merged = main._merge_sync_meta(meta, {
        "target_filename": "Movie.2020.BluRay-GRP.mkv", "video_hash": "current",
    })
    assert merged["target_filename"] == "Movie.2020.BluRay-GRP.mkv"
    assert merged["video_hash"] == "current"
    assert meta["video_hash"] == "old"


@pytest.mark.asyncio
async def test_opensubtitles_cached_route_runs_autosync(monkeypatch, enabled_sync):
    config = encode_user_config(auto_sync=True, subdl_key="current-key")
    request = Request({
        "type": "http", "method": "GET", "path": f"/{config}/sub/opensubtitles/42.srt",
        "query_string": b"imdb=tt1&filename=Movie.2020.BluRay-GRP.mkv", "headers": [],
    })
    monkeypatch.setattr(main.cache_manager, "get_subtitle", AsyncMock(return_value=_srt().encode()))
    monkeypatch.setattr(main.cache_manager, "get_metadata", lambda _: _meta())
    sync = AsyncMock(return_value=_srt([1] * 10).encode())
    monkeypatch.setattr(main, "_maybe_sync_subtitle", sync)
    response = await main.proxy_opensubtitles_stream(42, request, config)
    assert sync.await_count == 1
    args = sync.call_args.args
    assert args[1]["target_filename"] == "Movie.2020.BluRay-GRP.mkv"
    assert args[1]["subdl_key"] == "current-key"
    assert args[3] is True
    assert b"00:00:11,000" in response.body


def test_sync_serves_alass_output_regardless_of_drift(monkeypatch):
    def run(command, **kwargs):
        with open(command[3], "w", encoding="utf-8") as output:
            output.write(_srt([0] * 3 + [20] * 7))
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(subprocess, "run", run)
    # The post-alass shift guardrail is gone: a zero-exit output is trusted.
    assert SubtitleSyncService().sync(_srt(), _srt(), decision_kind="edition") is not None


@pytest.mark.asyncio
async def test_external_cancellation_stops_provider_tasks(tmp_path):
    started, stopped, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def search(**kwargs):
        started.set()
        try:
            await release.wait()
        finally:
            stopped.set()
        return []

    strategy = ExternalExactStrategy(
        subdl_provider=SimpleNamespace(search_subtitles=search),
        cache=ReferenceDiskCache(tmp_path),
    )
    task = asyncio.create_task(strategy.resolve(ReferenceQuery(imdb_id="tt1")))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    was_stopped = stopped.is_set()
    release.set()
    await asyncio.sleep(0)
    assert was_stopped


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["download", "fallback"])
async def test_opensubtitles_fresh_routes_run_autosync(monkeypatch, enabled_sync, source):
    config = encode_user_config(auto_sync=True, subdl_key="current-key")
    request = Request({
        "type": "http", "method": "GET", "path": "/sub/opensubtitles/42.srt",
        "query_string": b"imdb=tt1&filename=Movie.2020.BluRay-GRP.mkv", "headers": [],
    })
    monkeypatch.setattr(main.cache_manager, "get_subtitle", AsyncMock(return_value=None))
    monkeypatch.setattr(main.cache_manager, "save_subtitle", AsyncMock())
    monkeypatch.setattr(main.cache_manager, "store_metadata", lambda *args: None)
    monkeypatch.setattr(main.cache_manager, "get_metadata", lambda _: _meta())
    monkeypatch.setattr(main, "_http_client", SimpleNamespace(get=AsyncMock(return_value=
        SimpleNamespace(status_code=200, content=_srt().encode())
    )))
    provider = SimpleNamespace(get_download_url=AsyncMock(
        return_value="https://example/sub.srt" if source == "download" else None
    ))
    monkeypatch.setattr(main, "OpenSubtitlesProvider", lambda _: provider)
    monkeypatch.setattr(main, "_fallback_download_subsource", AsyncMock(return_value=_srt().encode()))
    sync = AsyncMock(return_value=_srt([1] * 10).encode())
    monkeypatch.setattr(main, "_maybe_sync_subtitle", sync)
    response = await main.proxy_opensubtitles_stream(42, request, config)
    assert sync.await_count == 1
    assert sync.call_args.args[3] is True
    assert b"00:00:11,000" in response.body


@pytest.mark.asyncio
async def test_failed_alignment_is_not_repeated_on_every_retry(enabled_sync):
    # Reference offset +1s from the target: close enough to pass cue-sanity
    # (first-dialogue delta 1s) yet misaligned (median 1s), so alass runs once,
    # fails, and the negative cache suppresses the retry.
    strategy = SimpleNamespace(resolve_with_provenance=AsyncMock(
        return_value=ResolvedReference(_srt([1] * 10), "edition")
    ))
    service = SimpleNamespace(sync_async=AsyncMock(return_value=None))
    orch = _orch(strategy, service)
    for _ in range(2):
        assert await orch.evaluate_and_sync(_srt().encode(), _meta(), "t", True) == _srt().encode()
    assert service.sync_async.await_count == 1


@pytest.mark.asyncio
async def test_strict_policy_does_not_reuse_relaxed_sync(monkeypatch, enabled_sync):
    strategy = SimpleNamespace(resolve_with_provenance=AsyncMock(
        return_value=ResolvedReference(_srt(), "edition")
    ))
    service = SimpleNamespace(sync_async=AsyncMock(return_value=_srt([1] * 10)))
    orch = _orch(strategy, service)
    monkeypatch.setattr(settings, "SYNC_REQUIRE_EXACT_MATCH", False)
    await orch.evaluate_and_sync(_srt().encode(), _meta(), "t", True)
    monkeypatch.setattr(settings, "SYNC_REQUIRE_EXACT_MATCH", True)
    strategy.resolve_with_provenance.return_value = ResolvedReference(None)
    assert await orch.evaluate_and_sync(_srt().encode(), _meta(), "t", True) == _srt().encode()
    assert strategy.resolve_with_provenance.await_count == 2


@pytest.mark.asyncio
async def test_cached_raw_subtitle_uses_current_users_sync_credentials(monkeypatch):
    config = encode_user_config(auto_sync=True, subdl_key="current-key")
    monkeypatch.setattr(main, "_http_client", object())
    monkeypatch.setattr(main.cache_manager, "get_subtitle", AsyncMock(return_value=_srt().encode()))
    monkeypatch.setattr(main.cache_manager, "get_metadata", lambda _: _meta(subdl_key="other-user"))
    sync = AsyncMock(return_value=_srt().encode())
    monkeypatch.setattr(main, "_maybe_sync_subtitle", sync)
    await main._serve_subtitle_handler("t", config_str=config)
    assert sync.call_args.args[1]["subdl_key"] == "current-key"


@pytest.mark.asyncio
async def test_autosync_response_can_be_refetched_after_background_alignment(monkeypatch, enabled_sync):
    config = encode_user_config(auto_sync=True)
    monkeypatch.setattr(main, "_http_client", object())
    monkeypatch.setattr(main.cache_manager, "get_subtitle", AsyncMock(return_value=_srt().encode()))
    monkeypatch.setattr(main.cache_manager, "get_metadata", lambda _: _meta())
    monkeypatch.setattr(main, "_maybe_sync_subtitle", AsyncMock(return_value=_srt().encode()))
    response = await main._serve_subtitle_handler("t", config_str=config)
    assert response.headers["cache-control"] == "no-store, no-cache, must-revalidate"


@pytest.mark.asyncio
async def test_native_ass_preference_preserves_format_during_autosync(monkeypatch):
    ass = b"[Script Info]\n[Events]\nDialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,Hello\n"
    config = encode_user_config(auto_sync=True, convert_ass_to_srt=False)
    monkeypatch.setattr(main, "_http_client", object())
    monkeypatch.setattr(main.cache_manager, "get_subtitle", AsyncMock(return_value=ass))
    monkeypatch.setattr(main.cache_manager, "get_metadata", lambda _: _meta())
    sync = AsyncMock(return_value=_srt().encode())
    monkeypatch.setattr(main, "_maybe_sync_subtitle", sync)
    response = await main._serve_subtitle_handler("t", config_str=config, req_format="ass")
    assert response.body == ass
    assert "text/x-ssa" in response.media_type
    assert sync.await_count == 0


@pytest.mark.asyncio
async def test_alignment_concurrency_is_bounded(monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def worker(*args):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return _srt()

    monkeypatch.setattr(asyncio, "to_thread", worker)
    service = SubtitleSyncService()
    first = asyncio.create_task(service.sync_async(_srt(), _srt()))
    await entered.wait()
    second = asyncio.create_task(service.sync_async(_srt(), _srt()))
    await asyncio.sleep(0)
    before_release = calls
    release.set()
    await asyncio.gather(first, second)
    assert before_release == 1
    assert calls == 2


def test_reference_cache_rejects_mixed_payload_and_provenance(tmp_path):
    cache = ReferenceDiskCache(tmp_path, min_bytes=10)
    query = ReferenceQuery(imdb_id="tt1")
    cache.set(query, "subdl", _srt(), kind="hash")
    path = next(tmp_path.glob("*.srt"))
    path.write_text(_srt([5] * 10), encoding="utf-8")
    assert cache.get(query) is None


def test_real_alass_corrects_offset_and_preserves_arabic():
    binary = os.getenv("NINJASUBS_TEST_ALASS") or shutil.which(settings.ALASS_PATH)
    if not binary:
        pytest.skip("Set NINJASUBS_TEST_ALASS to run the real alignment check")

    starts = [10000, 15870, 24620, 33410, 42090, 58080, 64120, 74930, 84510, 98870]

    def stamp(milliseconds):
        seconds, ms = divmod(milliseconds, 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"

    def subtitles(offset, text):
        return "\n\n".join(
            f"{i + 1}\n{stamp(start + offset)} --> "
            f"{stamp(start + offset + 1300 + i * 137)}\n{text} {i + 1}"
            for i, start in enumerate(starts)
        ) + "\n"

    service = SubtitleSyncService(alass_path=binary)
    synced = service.sync(subtitles(3000, "حوار عربي"), subtitles(0, "Reference"), decision_kind="hash")
    assert synced is not None
    from app.services.sync_service import _cue_starts_ms

    actual = _cue_starts_ms(synced, limit=len(starts))
    assert len(actual) == len(starts)
    # alass's speed/FPS heuristics can introduce a few milliseconds of jitter.
    assert max(abs(after - before) for after, before in zip(actual, starts, strict=True)) <= 20
    assert synced.count("حوار عربي") == len(starts)
