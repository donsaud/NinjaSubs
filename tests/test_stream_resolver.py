"""Tests for the generic Stremio stream resolver (Torrentio/Comet/AIOStreams)."""

from types import SimpleNamespace

import pytest

from app.services.sync.stream_resolver import StreamResolver, clean_stream_addon_url


class _JsonClient:
    """Minimal httpx.AsyncClient stand-in that records request URLs."""

    def __init__(self, payload=None, status_code=200, exc=None):
        self._payload = payload
        self._status = status_code
        self._exc = exc
        self.calls: list[str] = []

    async def get(self, url, timeout=None, follow_redirects=True):
        self.calls.append(url)
        if self._exc is not None:
            raise self._exc
        return SimpleNamespace(status_code=self._status, json=lambda: self._payload, text="")


# --------------------------------------------------------------------------- #
# URL cleaning
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://torrentio.strem.fun/", "https://torrentio.strem.fun"),
        ("https://torrentio.strem.fun", "https://torrentio.strem.fun"),
        ("https://torrentio.strem.fun/manifest.json", "https://torrentio.strem.fun"),
        ("https://comet.example.com/config/manifest.json/", "https://comet.example.com/config"),
        ("  https://mediafusion.example.com/  ", "https://mediafusion.example.com"),
        ("", ""),
        (None, ""),
    ],
)
def test_clean_stream_addon_url(raw, expected):
    assert clean_stream_addon_url(raw) == expected


# --------------------------------------------------------------------------- #
# Torrentio-style response (rich title + behaviorHints.filename)
# --------------------------------------------------------------------------- #
TORRENTIO_PAYLOAD = {
    "streams": [
        {
            "name": "Torrentio\n1080p",
            "title": "Into the Wild 2007 1080p BluRay x264-YTS\nPeer: 3 Seed: 1",
            "behaviorHints": {"filename": "Into.the.Wild.2007.1080p.BluRay.x264-YTS.mkv"},
            "url": "https://torrentio.strem.fun/resolve/yts.mkv",
        },
        {
            "name": "Torrentio\n1080p",
            "title": "Into the Wild 2007 1080p BluRay x264-FSiHD\nPeer: 12 Seed: 8",
            "behaviorHints": {"filename": "Into.the.Wild.2007.1080p.BluRay.x264-FSiHD.mkv"},
            "url": "https://torrentio.strem.fun/resolve/fsihd.mkv",
        },
    ]
}


@pytest.mark.asyncio
async def test_torrentio_picks_release_group_match():
    client = _JsonClient(TORRENTIO_PAYLOAD)
    resolver = StreamResolver(client, "https://torrentio.strem.fun/manifest.json")
    url = await resolver.resolve_stream_url(
        "tt0758758", "movie", "Into.the.Wild.2007.1080p.BluRay.x264-FSiHD.mkv"
    )
    assert url == "https://torrentio.strem.fun/resolve/fsihd.mkv"
    assert client.calls == [
        "https://torrentio.strem.fun/stream/movie/tt0758758.json"
    ]


@pytest.mark.asyncio
async def test_resolution_tiebreak_prefers_matching_resolution():
    payload = {
        "streams": [
            {
                "title": "Movie 2020 1080p WEB-DL x264-GRP",
                "behaviorHints": {"filename": "Movie.2020.1080p.WEB-DL.x264-GRP.mkv"},
                "url": "https://cdn/1080.mkv",
            },
            {
                "title": "Movie 2020 720p WEB-DL x264-GRP",
                "behaviorHints": {"filename": "Movie.2020.720p.WEB-DL.x264-GRP.mkv"},
                "url": "https://cdn/720.mkv",
            },
        ]
    }
    resolver = StreamResolver(_JsonClient(payload), "https://addon.example")
    url = await resolver.resolve_stream_url("tt1", "movie", "Movie.2020.1080p.WEB-DL.x264-GRP.mkv")
    assert url == "https://cdn/1080.mkv"


# --------------------------------------------------------------------------- #
# Comet-style response (description carries the release name)
# --------------------------------------------------------------------------- #
COMET_PAYLOAD = {
    "streams": [
        {
            "name": "Comet 1080p",
            "description": "Suits S01E01 1080p WEB-DL DDP5.1 H.264-FLUX",
            "url": "https://comet.example.com/resolve/s01e01.mkv",
        },
        {
            "name": "Comet 1080p",
            "description": "Suits S01E02 1080p WEB-DL DDP5.1 H.264-FLUX",
            "url": "https://comet.example.com/resolve/s01e02.mkv",
        },
    ]
}


@pytest.mark.asyncio
async def test_comet_matches_episode_by_description():
    client = _JsonClient(COMET_PAYLOAD)
    resolver = StreamResolver(client, "https://comet.example.com")
    url = await resolver.resolve_stream_url(
        "tt1632701",
        "series",
        "Suits.S01E02.1080p.WEB-DL.DDP5.1.H.264-FLUX.mkv",
        season=1,
        episode=2,
    )
    assert url == "https://comet.example.com/resolve/s01e02.mkv"
    assert client.calls == [
        "https://comet.example.com/stream/series/tt1632701:1:2.json"
    ]


# --------------------------------------------------------------------------- #
# User URL overrides the env fallback; env fallback used when none supplied
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_user_base_url_overrides_env_fallback():
    client = _JsonClient({"streams": []})
    resolver = StreamResolver(client, "http://aiostreams:3000")
    await resolver.resolve_stream_url(
        "tt1",
        "movie",
        "Movie.mkv",
        base_url="https://torrentio.strem.fun/manifest.json/",
    )
    assert client.calls == ["https://torrentio.strem.fun/stream/movie/tt1.json"]


@pytest.mark.asyncio
async def test_env_fallback_used_when_no_user_url():
    client = _JsonClient({"streams": []})
    resolver = StreamResolver(client, "http://aiostreams:3000/")
    await resolver.resolve_stream_url("tt1", "movie", "Movie.mkv")
    assert client.calls == ["http://aiostreams:3000/stream/movie/tt1.json"]


@pytest.mark.asyncio
async def test_no_base_url_returns_none_without_requesting():
    client = _JsonClient({"streams": []})
    resolver = StreamResolver(client, "")
    assert await resolver.resolve_stream_url("tt1", "movie", "Movie.mkv") is None
    assert client.calls == []


# --------------------------------------------------------------------------- #
# Errors / no match degrade to None
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_no_match_returns_none():
    payload = {"streams": [{"title": "Unrelated.Release", "url": "https://cdn/x.mkv"}]}
    resolver = StreamResolver(_JsonClient(payload), "https://addon.example")
    assert (
        await resolver.resolve_stream_url("tt1", "movie", "Into.the.Wild.2007.1080p.BluRay.mkv")
        is None
    )


@pytest.mark.asyncio
async def test_errors_degrade_to_none():
    resolver = StreamResolver(_JsonClient(exc=RuntimeError("timeout")), "https://addon.example")
    assert await resolver.resolve_stream_url("tt1", "movie", "Movie.mkv") is None
    resolver = StreamResolver(
        _JsonClient(payload=None, status_code=502), "https://addon.example"
    )
    assert await resolver.resolve_stream_url("tt1", "movie", "Movie.mkv") is None


# --------------------------------------------------------------------------- #
# Orchestrator forwards the user's addon URL to the resolver
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_orchestrator_forwards_user_stream_addon_url(monkeypatch):
    from app.config import settings as app_settings
    from app.services.sync.orchestrator import SyncOrchestrator
    from app.services.sync.query import ResolvedReference
    from app.services.sync_cache import SyncCache

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)

    class _Resolver:
        def __init__(self):
            self.bases: list[str | None] = []

        async def resolve_stream_url(
            self, imdb_id, media_type, filename, season=None, episode=None, base_url=None
        ):
            self.bases.append(base_url)
            return None

    class _Embedded:
        async def resolve_with_provenance(self, query):
            return ResolvedReference(None)

    resolver = _Resolver()
    orch = SyncOrchestrator(
        embedded_strategy=_Embedded(),
        aiostreams=resolver,
        sync_service=SimpleNamespace(sync_async=lambda *a, **k: None),
        sync_cache=SyncCache(),
    )

    class _Sync:
        async def sync_async(self, *a, **k):
            return None

    orch._sync_service = _Sync()
    meta = {
        "imdb_id": "tt1",
        "media_type": "movie",
        "lang": "ara",
        "target_filename": "Movie.mkv",
        "stream_addon_url": "https://torrentio.strem.fun/manifest.json",
    }
    await orch.evaluate_and_sync(b"1\n00:00:01,000 --> 00:00:02,000\nx\n", meta, "t", True)
    assert resolver.bases == ["https://torrentio.strem.fun/manifest.json"]
