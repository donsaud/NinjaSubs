"""Tests for the AIOStreams bridge, stream MovieHash tier, and PGS skipping."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from app.services.sync.aiostreams import AIOStreamsClient
from app.services.sync.cache import ReferenceDiskCache
from app.services.sync.embedded_strategy import EmbeddedStrategy
from app.services.sync.hash_strategy import HashExactStrategy
from app.services.sync.query import ReferenceQuery
from app.services.sync.stream_hash import _opensubtitles_hash, fetch_stream_moviehash


@pytest.fixture(autouse=True)
def _isolated_reference_cache(tmp_path, monkeypatch):
    """Keep strategy tests hermetic: redirect the default reference cache to tmp."""
    original = ReferenceDiskCache.__init__

    def __init__(self, root=None, **kwargs):
        if root is None:
            root = tmp_path / "refs"
        original(self, root, **kwargs)

    monkeypatch.setattr(ReferenceDiskCache, "__init__", __init__)


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
        "app.services.sync.stream_hash.is_safe_public_url", lambda url, **kwargs: (True, "ok")
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
        lambda url, **kwargs: (False, "blocked address 127.0.0.1"),
    )
    client = _RangeClient(200_000, b"A" * 65536, b"B" * 65536)
    assert await fetch_stream_moviehash("http://127.0.0.1/x.mkv", client) is None


@pytest.mark.asyncio
async def test_fetch_stream_moviehash_rejects_ignored_range(monkeypatch):
    monkeypatch.setattr(
        "app.services.sync.stream_hash.is_safe_public_url", lambda url, **kwargs: (True, "ok")
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
    async def fake_fetch(url, client, timeout=1.5, **kwargs):
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
        "app.services.sync.embedded_strategy.is_safe_public_url", lambda url, **kwargs: (True, "ok")
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


def _text_probe(*langs):
    streams = [
        {"index": i, "codec_name": "subrip", "tags": {"language": lang}}
        for i, lang in enumerate(langs)
    ]
    return json.dumps({"streams": streams}).encode()


@pytest.mark.asyncio
async def test_fetch_stream_moviehash_allows_private_literal_with_flag():
    from app.services.sync.stream_hash import _opensubtitles_hash

    size = 200_000
    first = b"A" * 65536
    last = b"B" * 65536
    client = _RangeClient(size, first, last)

    # Strict default blocks the LAN host...
    assert (
        await fetch_stream_moviehash("http://192.168.8.115:4000/movie.mkv", client) is None
    )
    # ...allow_private permits it and the hash still computes.
    result = await fetch_stream_moviehash(
        "http://192.168.8.115:4000/movie.mkv", client, allow_private=True
    )
    assert result == (_opensubtitles_hash(first, last, size), size)


@pytest.mark.asyncio
async def test_embedded_allows_private_stream_url_when_enabled(monkeypatch):
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ALLOW_PRIVATE_STREAM_URLS", True)

    def fake_run(cmd, capture_output=True, timeout=None):
        if cmd[0].endswith("ffprobe"):
            return SimpleNamespace(returncode=0, stdout=_text_probe("eng"), stderr=b"")
        srt = b"1\n00:00:01,000 --> 00:00:02,000\n" + (b"hello\n" * 600) + b"\n"
        return SimpleNamespace(returncode=0, stdout=srt, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    strategy = EmbeddedStrategy(
        ffprobe_path="/fake/ffprobe", ffmpeg_path="/fake/ffmpeg", timeout=2.0, min_bytes=100
    )
    query = ReferenceQuery(
        imdb_id="tt1", media_type="movie", stream_url="http://192.168.8.115:4000/movie.mkv"
    )
    assert await strategy.resolve(query) is not None


@pytest.mark.asyncio
async def test_embedded_blocks_private_stream_url_when_disabled(monkeypatch, caplog):
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ALLOW_PRIVATE_STREAM_URLS", False)

    def fake_run(cmd, capture_output=True, timeout=None):  # pragma: no cover
        raise AssertionError("must not spawn ffprobe for a blocked URL")

    monkeypatch.setattr(subprocess, "run", fake_run)
    strategy = EmbeddedStrategy(
        ffprobe_path="/fake/ffprobe", ffmpeg_path="/fake/ffmpeg", timeout=2.0
    )
    query = ReferenceQuery(
        imdb_id="tt1", media_type="movie", stream_url="http://192.168.8.115:4000/movie.mkv"
    )
    with caplog.at_level("WARNING"):
        assert await strategy.resolve(query) is None
    assert "blocked unsafe stream URL" in caplog.text


# --------------------------------------------------------------------------- #
# stream_hash: HEAD 405 fallback to a zero-byte range GET
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_content_length_falls_back_on_head_405():
    from app.services.sync.stream_hash import _content_length

    class _Head405Client:
        async def head(self, url, follow_redirects=True):
            return _FakeResponse(405, b"", {})

        async def get(self, url, headers=None, follow_redirects=True):
            return _FakeResponse(206, b"\x00", {"content-range": "bytes 0-0/987654"})

    assert await _content_length(_Head405Client(), "http://192.168.8.115:4000/x.mkv") == 987654


@pytest.mark.asyncio
async def test_fetch_stream_moviehash_survives_head_405(monkeypatch):
    monkeypatch.setattr(
        "app.services.sync.stream_hash.is_safe_public_url", lambda url, **k: (True, "ok")
    )
    size = 200_000
    first = b"A" * 65536
    last = b"B" * 65536

    class _Head405RangeClient(_RangeClient):
        async def head(self, url, follow_redirects=True):
            return _FakeResponse(405, b"", {})

    client = _Head405RangeClient(size, first, last)
    result = await fetch_stream_moviehash("https://cdn.example/movie.mkv", client)
    assert result == (_opensubtitles_hash(first, last, size), size)


# --------------------------------------------------------------------------- #
# embedded extraction command flags
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_embedded_extract_command_is_remote_stream_optimized(monkeypatch):
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ALLOW_PRIVATE_STREAM_URLS", True)
    calls = []

    def fake_run(cmd, capture_output=True, timeout=None):
        calls.append(cmd)
        if cmd[0].endswith("ffprobe"):
            return SimpleNamespace(returncode=0, stdout=_text_probe("eng"), stderr=b"")
        srt = b"1\n00:00:01,000 --> 00:00:02,000\n" + (b"hi\n" * 600) + b"\n"
        return SimpleNamespace(returncode=0, stdout=srt, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    strategy = EmbeddedStrategy(
        ffprobe_path="/fake/ffprobe", ffmpeg_path="/fake/ffmpeg", timeout=2.0, min_bytes=100
    )
    query = ReferenceQuery(
        imdb_id="tt1", media_type="movie", stream_url="http://192.168.8.115:4000/movie.mkv"
    )
    assert await strategy.resolve(query) is not None

    cmd = next(c for c in calls if "ffmpeg" in c[0])
    for flag, value in (
        ("-nostdin", None),
        ("-threads", "1"),
        ("-fflags", "+nobuffer+fastseek"),
        ("-analyzeduration", "10000000"),
        ("-probesize", "10000000"),
        ("-copyts", None),
    ):
        assert flag in cmd, flag
        if value is not None:
            assert cmd[cmd.index(flag) + 1] == value
    assert cmd[cmd.index("-map") + 1] == "0:0"
    assert cmd[cmd.index("-c:s") + 1] == "srt"
    assert cmd[-1] == "-"


@pytest.mark.asyncio
async def test_content_length_prefers_content_range_on_partial_response():
    """A 206 includes Content-Length of the chunk; Content-Range holds the total."""
    from app.services.sync.stream_hash import _content_length

    class _PartialClient:
        async def head(self, url, follow_redirects=True):
            return _FakeResponse(405, b"", {})

        async def get(self, url, headers=None, follow_redirects=True):
            return _FakeResponse(
                206, b"\x00", {"content-length": "1", "content-range": "bytes 0-0/200000"}
            )

    assert await _content_length(_PartialClient(), "http://192.168.8.115:4000/x.mkv") == 200000


# --------------------------------------------------------------------------- #
# Hash strategy: accept major-language hash matches
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_hash_strategy_accepts_multi_language_hash_match():
    from app.models import SubtitleRelease

    payload = ("1\n00:00:01,000 --> 00:00:02,000\n" + ("ref line\n" * 800) + "\n").encode()

    class _Prov:
        def __init__(self):
            self.calls = []

        async def search_subtitles(self, **kwargs):
            self.calls.append(kwargs)
            return [
                SubtitleRelease(
                    release_name="Movie.2024.Spanish.srt",
                    download_url="http://os/movie",
                    provider="opensubtitles",
                    lang="spa",
                    is_hash_match=True,
                )
            ]

        async def download_archive(self, url, api_key=None):
            return payload

    prov = _Prov()
    strategy = HashExactStrategy(prov)
    query = ReferenceQuery(
        imdb_id="tt1", media_type="movie", video_hash="51b3392cbbf534e0", video_size=123
    )
    resolved = await strategy.resolve_with_provenance(query)
    assert resolved.kind == "hash" and resolved.text
    langs = prov.calls[0]["languages"]
    assert "en" in langs and "es" in langs and "fr" in langs


# --------------------------------------------------------------------------- #
# Embedded extraction: -t cap, 8.0s timeout, partial fallback
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_embedded_extraction_uses_t_cap_and_timeout(monkeypatch):
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ALLOW_PRIVATE_STREAM_URLS", True)
    captured = {}

    def fake_run(cmd, capture_output=True, timeout=None):
        if cmd[0].endswith("ffprobe"):
            return SimpleNamespace(returncode=0, stdout=_text_probe("eng"), stderr=b"")
        captured["cmd"] = cmd
        captured["timeout"] = timeout
        srt = b"1\n00:00:01,000 --> 00:00:02,000\n" + (b"hi\n" * 600) + b"\n"
        return SimpleNamespace(returncode=0, stdout=srt, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    # extract_timeout defaults to 8.0 for slow remote/USENET streams.
    strategy = EmbeddedStrategy(
        ffprobe_path="/fake/ffprobe",
        ffmpeg_path="/fake/ffmpeg",
        timeout=2.0,
        min_bytes=100,
    )
    assert strategy.extract_timeout == 8.0
    query = ReferenceQuery(
        imdb_id="tt1", media_type="movie", stream_url="http://192.168.8.115:4000/movie.mkv"
    )
    assert await strategy.resolve(query) is not None
    cmd = captured["cmd"]
    # ``-t 900`` is an *output* option: it must come after ``-i`` on HTTP MKV.
    assert cmd.index("-t") > cmd.index("-i")
    assert cmd[cmd.index("-t") + 1] == "900"
    assert captured["timeout"] == 8.0
    # Low-latency demux flags requested for HTTP/WebDAV streams.
    assert "+nobuffer+fastseek" in cmd
    assert cmd[cmd.index("-fflags") + 1] == "+nobuffer+fastseek"


@pytest.mark.asyncio
async def test_embedded_extraction_uses_partial_output_on_timeout(monkeypatch):
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ALLOW_PRIVATE_STREAM_URLS", True)

    def fake_run(cmd, capture_output=True, timeout=None):
        if cmd[0].endswith("ffprobe"):
            return SimpleNamespace(returncode=0, stdout=_text_probe("eng"), stderr=b"")
        partial = b"1\n00:00:01,000 --> 00:00:02,000\n" + (b"hi\n" * 400) + b"\n"
        raise subprocess.TimeoutExpired(cmd, timeout, output=partial)

    monkeypatch.setattr(subprocess, "run", fake_run)
    strategy = EmbeddedStrategy(
        ffprobe_path="/fake/ffprobe",
        ffmpeg_path="/fake/ffmpeg",
        timeout=2.0,
        extract_timeout=8.0,
        min_bytes=100,
    )
    query = ReferenceQuery(
        imdb_id="tt1", media_type="movie", stream_url="http://192.168.8.115:4000/movie.mkv"
    )
    resolved = await strategy.resolve_with_provenance(query)
    assert resolved.text is not None
    assert resolved.kind == "embedded" and resolved.partial is True


@pytest.mark.asyncio
async def test_orchestrator_marks_embedded_reference_partial(monkeypatch):
    from app.config import settings as app_settings
    from app.services.sync.orchestrator import SyncOrchestrator
    from app.services.sync.query import ResolvedReference
    from app.services.sync_cache import SyncCache

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)

    class _Emb:
        async def resolve_with_provenance(self, query):
            return ResolvedReference(BIG_REF.decode(), kind="embedded", partial=True)

    class _Sync:
        def __init__(self):
            self.partial = []

        async def sync_async(self, *args, **kwargs):
            self.partial.append(kwargs.get("reference_partial"))
            return "1\n00:00:03,000 --> 00:00:04,000\nsynced\n"

    sync = _Sync()
    orch = SyncOrchestrator(embedded_strategy=_Emb(), sync_service=sync, sync_cache=SyncCache())
    meta = {"imdb_id": "tt1", "media_type": "movie", "lang": "ara", "target_filename": "Movie.mkv"}
    await orch.evaluate_and_sync(_arabic_bytes(), meta, "t", True)
    assert sync.partial == [True]


# --------------------------------------------------------------------------- #
# Background embedded warm-up
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_embedded_warm_reference_caches_internal_track(monkeypatch, tmp_path):
    from app.config import settings as app_settings
    from app.services.sync import mkv_range

    monkeypatch.setattr(app_settings, "ALLOW_PRIVATE_STREAM_URLS", True)
    monkeypatch.setattr(
        "app.services.sync.embedded_strategy.is_safe_public_url", lambda url, **k: (True, "ok")
    )
    srt = ("1\n00:00:01,000 --> 00:00:02,000\n" + ("hi\n" * 600) + "\n").encode()

    async def fake_extract(url, client, **kwargs):
        return srt

    monkeypatch.setattr(mkv_range, "extract_embedded_srt", fake_extract)

    cache = ReferenceDiskCache(root=tmp_path / "refs", ttl=3600.0, min_bytes=100)
    strategy = EmbeddedStrategy(client=object(), cache=cache)
    query = ReferenceQuery(
        imdb_id="tt1", media_type="movie", stream_url="http://192.168.8.115:4000/movie.mkv"
    )
    resolved = await strategy.warm_reference(query)
    assert resolved is not None and resolved.partial is True
    assert cache.get(query) is not None  # persisted for the next request


@pytest.mark.asyncio
async def test_orchestrator_schedules_embedded_warmup(monkeypatch):
    import asyncio

    from app.config import settings as app_settings
    from app.services.sync.orchestrator import SyncOrchestrator
    from app.services.sync.query import ResolvedReference
    from app.services.sync_cache import SyncCache

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)

    class _AIO:
        async def resolve_stream_url(self, imdb_id, media_type, filename, season=None, episode=None):
            return "http://192.168.8.115:4000/movie.mkv"

    warm_calls = {"n": 0}

    class _Embedded:
        async def resolve_with_provenance(self, query):
            return ResolvedReference(None)

        async def warm_reference(self, query):
            warm_calls["n"] += 1
            return None

    class _External:
        async def resolve_with_provenance(self, query):
            return ResolvedReference(BIG_REF.decode(), kind="edition")

    class _Sync:
        async def sync_async(self, *a, **k):
            return "1\n00:00:03,000 --> 00:00:04,000\nsynced\n"

    orch = SyncOrchestrator(
        embedded_strategy=_Embedded(),
        external_strategy=_External(),
        aiostreams=_AIO(),
        sync_service=_Sync(),
        sync_cache=SyncCache(),
    )
    meta = {"imdb_id": "tt1", "media_type": "movie", "lang": "ara", "target_filename": "Movie.mkv"}
    await orch.evaluate_and_sync(_arabic_bytes(), meta, "t", True)
    for _ in range(20):
        await asyncio.sleep(0)
        if warm_calls["n"]:
            break
    assert warm_calls["n"] == 1


# --------------------------------------------------------------------------- #
# Negative-cache invalidation once a warm embedded reference is ready
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_negative_cache_overridden_by_ready_embedded_reference(monkeypatch, tmp_path):
    from app.config import settings as app_settings
    from app.services.sync.cache import ReferenceDiskCache
    from app.services.sync.orchestrator import SyncOrchestrator
    from app.services.sync.query import ResolvedReference
    from app.services.sync_cache import SyncCache

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)

    cache = ReferenceDiskCache(root=tmp_path / "refs", ttl=3600.0, min_bytes=100)

    class _Embedded:
        def __init__(self):
            self.cache = cache

        async def resolve_with_provenance(self, query):
            return ResolvedReference(None)  # inline embedded tier is cache-only

        async def warm_reference(self, query):
            return None

    class _Sync:
        def __init__(self):
            self.calls = 0
            self.partial: list = []

        async def sync_async(self, *args, **kwargs):
            self.calls += 1
            self.partial.append(kwargs.get("reference_partial"))
            return "1\n00:00:03,000 --> 00:00:04,000\nsynced\n"

    sync = _Sync()
    sync_cache = SyncCache()
    orch = SyncOrchestrator(
        embedded_strategy=_Embedded(), sync_service=sync, sync_cache=sync_cache
    )
    sub = _arabic_bytes()
    meta = {
        "imdb_id": "tt1",
        "media_type": "movie",
        "lang": "ara",
        "target_filename": "Movie.2024.1080p.BluRay.x264-GRP.mkv",
    }
    resolution_key = orch._flight_key(meta, "t", sub)
    await sync_cache.mark_failed(resolution_key)

    # 1) No reference on disk yet -> the negative cache short-circuits the sync.
    assert await orch.evaluate_and_sync(sub, meta, "t", True) == sub
    assert sync.calls == 0

    # 2) The background warm-up finishes and writes the internal-track reference.
    cache.set(orch._build_query(meta), "embedded", BIG_REF.decode(), kind="embedded")

    # 3) The next request ignores the stale negative entry and syncs against it.
    out = await orch.evaluate_and_sync(sub, meta, "t", True)
    assert b"synced" in out
    assert sync.calls == 1
    assert sync.partial == [True]  # embedded references are partial by nature
    assert not await sync_cache.is_failed(resolution_key)
    assert await sync_cache.get(resolution_key) is not None
