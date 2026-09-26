"""Tests for the AIOStreams bridge, stream MovieHash tier, and PGS skipping."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from app.services.sync.aiostreams import AIOStreamsClient
from app.services.sync.embedded_strategy import EmbeddedStrategy
from app.services.sync.hash_strategy import HashExactStrategy
from app.services.sync.query import ReferenceQuery
from app.services.sync.stream_hash import _opensubtitles_hash, fetch_stream_moviehash


# --------------------------------------------------------------------------- #
# MovieHash via HTTP Range
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, status_code=200, content=b"", headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}


class _RangeClient:
    def __init__(self, size, first, last):
        self._size = size
        self._first = first
        self._last = last

    async def head(self, url, follow_redirects=True):
        return _FakeResponse(200, b"", {"content-length": str(self._size)})

    async def get(self, url, headers=None, follow_redirects=True):
        rng = (headers or {}).get("Range", "")
        if rng == "bytes=0-0":
            return _FakeResponse(206, b"\x00", {"content-range": f"bytes 0-0/{self._size}"})
        if rng.startswith("bytes=0-"):
            return _FakeResponse(206, self._first)
        if rng.startswith(f"bytes={self._size - 65536}-"):
            return _FakeResponse(206, self._last)
        return _FakeResponse(206, b"")


@pytest.mark.asyncio
async def test_fetch_stream_moviehash_computes_hash(monkeypatch):
    monkeypatch.setattr(
        "app.services.sync.stream_hash.is_safe_public_url", lambda url: (True, "ok")
    )
    size = 200_000
    first = b"A" * 65536
    last = b"B" * 65536
    client = _RangeClient(size, first, last)
    result = await fetch_stream_moviehash("https://cdn.example/movie.mkv", client)
    assert result == (_opensubtitles_hash(first, last, size), size)


@pytest.mark.asyncio
async def test_fetch_stream_moviehash_blocks_unsafe_url(monkeypatch):
    monkeypatch.setattr(
        "app.services.sync.stream_hash.is_safe_public_url",
        lambda url: (False, "blocked address 127.0.0.1"),
    )
    client = _RangeClient(200_000, b"A" * 65536, b"B" * 65536)
    assert await fetch_stream_moviehash("http://127.0.0.1/x.mkv", client) is None


@pytest.mark.asyncio
async def test_fetch_stream_moviehash_rejects_ignored_range(monkeypatch):
    monkeypatch.setattr(
        "app.services.sync.stream_hash.is_safe_public_url", lambda url: (True, "ok")
    )

    class _IgnoreRangeClient(_RangeClient):
        async def get(self, url, headers=None, follow_redirects=True):
            return _FakeResponse(200, b"Z" * (1024 * 1024))  # whole file, huge

    client = _IgnoreRangeClient(200_000, b"A" * 65536, b"B" * 65536)
    assert await fetch_stream_moviehash("https://cdn.example/x.mkv", client) is None


# --------------------------------------------------------------------------- #
# AIOStreams bridge
# --------------------------------------------------------------------------- #
class _JsonClient:
    def __init__(self, payload=None, status_code=200, exc=None):
        self._payload = payload
        self._status = status_code
        self._exc = exc

    async def get(self, url, timeout=None, follow_redirects=True):
        if self._exc is not None:
            raise self._exc
        return SimpleNamespace(status_code=self._status, json=lambda: self._payload, text="")


@pytest.mark.asyncio
async def test_aiostreams_matches_stream_by_filename():
    payload = {
        "streams": [
            {"title": "Some.Other.Movie.2020.1080p", "url": "https://cdn/other.mkv"},
            {
                "title": "Into.the.Wild.2007.1080p.BluRay.x264-FSiHD.mkv",
                "url": "https://cdn/itw.mkv",
            },
        ]
    }
    client = AIOStreamsClient(_JsonClient(payload), "http://aiostreams:3000")
    url = await client.resolve_stream_url(
        "tt0758758", "movie", "Into.the.Wild.2007.1080p.BluRay.x264-FSiHD.mkv"
    )
    assert url == "https://cdn/itw.mkv"


@pytest.mark.asyncio
async def test_aiostreams_no_match_returns_none():
    payload = {"streams": [{"title": "Unrelated.Release", "url": "https://cdn/x.mkv"}]}
    client = AIOStreamsClient(_JsonClient(payload), "http://aiostreams:3000")
    assert (
        await client.resolve_stream_url("tt1", "movie", "Into.the.Wild.2007.1080p.BluRay.mkv")
        is None
    )


@pytest.mark.asyncio
async def test_aiostreams_errors_degrade_to_none():
    client = AIOStreamsClient(_JsonClient(exc=RuntimeError("timeout")), "http://aiostreams:3000")
    assert await client.resolve_stream_url("tt1", "movie", "Movie.mkv") is None
    client = AIOStreamsClient(_JsonClient(payload=None, status_code=502), "http://aiostreams:3000")
    assert await client.resolve_stream_url("tt1", "movie", "Movie.mkv") is None


@pytest.mark.asyncio
async def test_aiostreams_series_appends_season_episode():
    captured = {}

    class _Capturing(_JsonClient):
        async def get(self, url, timeout=None, follow_redirects=True):
            captured["url"] = url
            return SimpleNamespace(status_code=200, json=lambda: {"streams": []}, text="")

    client = AIOStreamsClient(_Capturing(), "http://aiostreams:3000/token")
    await client.resolve_stream_url("tt1632701", "series", "Suits.S01E01.mkv", season=1, episode=1)
    assert captured["url"] == "http://aiostreams:3000/token/stream/series/tt1632701:1:1.json"


# --------------------------------------------------------------------------- #
# Hash strategy uses the stream hash when no client hash is supplied
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_hash_strategy_computes_stream_hash(monkeypatch):
    async def fake_fetch(url, client, timeout=1.5):
        return ("deadbeefdeadbeef", 123456)

    monkeypatch.setattr("app.services.sync.stream_hash.fetch_stream_moviehash", fake_fetch)

    class _Prov:
        def __init__(self):
            self.calls = []

        async def search_subtitles(self, **kwargs):
            self.calls.append(kwargs)
            return []

        async def download_archive(self, *args, **kwargs):  # pragma: no cover
            return None

    prov = _Prov()
    strategy = HashExactStrategy(prov, client=object())
    query = ReferenceQuery(
        imdb_id="tt1",
        media_type="movie",
        stream_url="https://cdn.example/movie.mkv",
    )
    resolved = await strategy.resolve_with_provenance(query)
    assert resolved.text is None
    assert prov.calls[0]["video_hash"] == "deadbeefdeadbeef"
    assert prov.calls[0]["video_size"] == 123456


# --------------------------------------------------------------------------- #
# Embedded tier skips image-based (PGS) subtitle tracks
# --------------------------------------------------------------------------- #
def _pgs_probe(*langs):
    streams = [
        {"index": i, "codec_name": "hdmv_pgs_subtitle", "tags": {"language": lang}}
        for i, lang in enumerate(langs)
    ]
    return json.dumps({"streams": streams}).encode()


@pytest.mark.asyncio
async def test_embedded_skips_pgs_only_track(monkeypatch):
    calls = []

    def fake_run(cmd, capture_output=True, timeout=None):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout=_pgs_probe("eng"), stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        "app.services.sync.embedded_strategy.is_safe_public_url", lambda url: (True, "ok")
    )
    strategy = EmbeddedStrategy(
        ffprobe_path="/fake/ffprobe", ffmpeg_path="/fake/ffmpeg", timeout=2.0
    )
    query = ReferenceQuery(
        imdb_id="tt1", media_type="movie", stream_url="https://cdn.example/movie.mkv"
    )
    assert await strategy.resolve(query) is None
    # ffmpeg must never be invoked for a PGS-only stream.
    assert all("ffmpeg" not in cmd[0] for cmd in calls)


BIG_REF = ("1\n00:00:01,000 --> 00:00:02,000\n" + ("reference line\n" * 600) + "\n").encode()


def _arabic_bytes():
    return (
        b"1\n00:00:01,000 --> 00:00:02,000\nx\n\n"
        b"2\n00:00:03,000 --> 00:00:04,000\ny\n\n"
        b"3\n00:00:05,000 --> 00:00:06,000\nz\n\n"
        b"4\n00:00:07,000 --> 00:00:08,000\nw\n\n"
        b"5\n00:00:09,000 --> 00:00:10,000\nv\n"
    )


@pytest.mark.asyncio
async def test_orchestrator_resolves_stream_url_via_aiostreams(monkeypatch):
    from app.config import settings as app_settings
    from app.services.sync.orchestrator import SyncOrchestrator
    from app.services.sync.query import ResolvedReference
    from app.services.sync_cache import SyncCache

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)

    class _AIO:
        def __init__(self):
            self.calls = []

        async def resolve_stream_url(self, imdb_id, media_type, filename, season=None, episode=None):
            self.calls.append((imdb_id, media_type, filename))
            return "https://cdn/stream.mkv"

    class _CaptureEmbedded:
        def __init__(self):
            self.urls = []

        async def resolve_with_provenance(self, query):
            self.urls.append(query.stream_url)
            return ResolvedReference(BIG_REF.decode(), kind="embedded")

    class _Sync:
        async def sync_async(self, *args, **kwargs):
            return "1\n00:00:03,000 --> 00:00:04,000\nsynced\n"

    aio = _AIO()
    embedded = _CaptureEmbedded()
    orch = SyncOrchestrator(
        embedded_strategy=embedded,
        aiostreams=aio,
        sync_service=_Sync(),
        sync_cache=SyncCache(),
    )
    meta = {
        "imdb_id": "tt1",
        "media_type": "movie",
        "lang": "ara",
        "target_filename": "Movie.2024.1080p.BluRay.x264-GRP.mkv",
    }
    out = await orch.evaluate_and_sync(_arabic_bytes(), meta, "t", True)
    assert aio.calls and aio.calls[0][0] == "tt1"
    assert embedded.urls == ["https://cdn/stream.mkv"]
    assert b"synced" in out
