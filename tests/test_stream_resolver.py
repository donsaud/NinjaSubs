"""Unit tests for Stremio StreamResolver."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.sync.stream_resolver import (
    StreamResolver,
    _match_score,
    clean_stream_addon_url,
)


def test_clean_stream_addon_url():
    assert clean_stream_addon_url(None) == ""
    assert clean_stream_addon_url("") == ""
    assert clean_stream_addon_url("   ") == ""
    assert clean_stream_addon_url("https://torrentio.strem.fun/manifest.json") == "https://torrentio.strem.fun"
    assert clean_stream_addon_url("https://torrentio.strem.fun/manifest.json/") == "https://torrentio.strem.fun"
    assert clean_stream_addon_url("https://torrentio.strem.fun/sort=quality/manifest.json") == "https://torrentio.strem.fun/sort=quality"
    assert clean_stream_addon_url("https://aiostreams.example.com/") == "https://aiostreams.example.com"
    # stremio:// protocol conversion
    assert clean_stream_addon_url("stremio://aiostreams.elfhosted.com/my-uuid/manifest.json") == "https://aiostreams.elfhosted.com/my-uuid"
    assert clean_stream_addon_url("stremio://torrentio.strem.fun/manifest.json") == "https://torrentio.strem.fun"
    # Auto-prepend https:// when missing scheme
    assert clean_stream_addon_url("comet.elfhosted.com/stremio/manifest.json") == "https://comet.elfhosted.com/stremio"
    # Stripping surrounding quotes
    assert clean_stream_addon_url('"https://mediafusion.elfhosted.com/manifest.json"') == "https://mediafusion.elfhosted.com"
    assert clean_stream_addon_url("'https://torrentio.strem.fun/manifest.json'") == "https://torrentio.strem.fun"
    # Local container port rewrite
    assert clean_stream_addon_url("http://aiostreams:4000/manifest.json") == "http://aiostreams:3000"


def test_match_score():
    target = "Whiplash.2014.1080p.BluRay.x264.DTS-WiHD.mkv"
    exact_candidate = "Whiplash.2014.1080p.BluRay.x264.DTS-WiHD"
    different_group = "Whiplash.2014.1080p.BluRay.x264-GECKOS"
    different_res = "Whiplash.2014.720p.BluRay.x264-WiHD"

    score_exact = _match_score(target, exact_candidate)
    score_diff_group = _match_score(target, different_group)
    score_diff_res = _match_score(target, different_res)

    assert score_exact > 0.8
    assert score_exact > score_diff_group
    assert score_exact > score_diff_res


@pytest.mark.asyncio
async def test_resolve_stream_url_success():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "streams": [
            {
                "name": "Torrentio 1080p",
                "title": "Whiplash.2014.1080p.BluRay.x264.DTS-WiHD\n💾 8.7 GB",
                "url": "https://debrid.example.com/stream/whiplash-wihd.mkv",
                "behaviorHints": {"filename": "Whiplash.2014.1080p.BluRay.x264.DTS-WiHD.mkv"},
            },
            {
                "name": "Torrentio 720p",
                "title": "Whiplash.2014.720p.BluRay.x264-GECKOS\n💾 4.4 GB",
                "url": "https://debrid.example.com/stream/whiplash-geckos.mkv",
                "behaviorHints": {"filename": "Whiplash.2014.720p.BluRay.x264-GECKOS.mkv"},
            },
        ]
    }
    client = AsyncMock()
    client.get = AsyncMock(return_value=mock_resp)

    resolver = StreamResolver(client)
    resolved_url = await resolver.resolve_stream_url(
        imdb_id="tt2582802",
        media_type="movie",
        filename="Whiplash.2014.1080p.BluRay.x264.DTS-WiHD.mkv",
        base_url="https://torrentio.strem.fun/manifest.json",
    )

    assert resolved_url == "https://debrid.example.com/stream/whiplash-wihd.mkv"
    client.get.assert_awaited_once()
    call_args, call_kwargs = client.get.call_args
    assert call_args[0] == "https://torrentio.strem.fun/stream/movie/tt2582802.json"
    assert "Mozilla" in call_kwargs["headers"]["User-Agent"]
    assert call_kwargs["timeout"] == resolver.timeout
    assert call_kwargs["follow_redirects"] is True

    # Test cache: calling again shouldn't hit HTTP client
    cached_url = await resolver.resolve_stream_url(
        imdb_id="tt2582802",
        media_type="movie",
        filename="Whiplash.2014.1080p.BluRay.x264.DTS-WiHD.mkv",
        base_url="https://torrentio.strem.fun/manifest.json",
    )
    assert cached_url == "https://debrid.example.com/stream/whiplash-wihd.mkv"
    assert client.get.await_count == 1


@pytest.mark.asyncio
async def test_resolve_stream_url_series_formatting():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "streams": [
            {
                "title": "Breaking.Bad.S01E01.1080p.BluRay.x264-ROVERS",
                "url": "https://stream.example.com/bb-s01e01.mkv",
            }
        ]
    }
    client = AsyncMock()
    client.get = AsyncMock(return_value=mock_resp)

    resolver = StreamResolver(client, default_base_url="https://torrentio.strem.fun")
    resolved_url = await resolver.resolve_stream_url(
        imdb_id="tt0903747",
        media_type="series",
        filename="Breaking.Bad.S01E01.1080p.BluRay.x264-ROVERS.mkv",
        season=1,
        episode=1,
    )

    assert resolved_url == "https://stream.example.com/bb-s01e01.mkv"
    client.get.assert_awaited_once()
    call_args, call_kwargs = client.get.call_args
    assert call_args[0] == "https://torrentio.strem.fun/stream/series/tt0903747:1:1.json"
    assert "Mozilla" in call_kwargs["headers"]["User-Agent"]


@pytest.mark.asyncio
async def test_resolve_stream_url_failures_return_none():
    client = AsyncMock()
    # 1. HTTP 500 error
    mock_500 = MagicMock()
    mock_500.status_code = 500
    client.get = AsyncMock(return_value=mock_500)
    resolver = StreamResolver(client, default_base_url="https://torrentio.strem.fun")
    assert await resolver.resolve_stream_url("tt123", "movie", "File.mkv") is None

    # 2. Connection exception
    client.get = AsyncMock(side_effect=Exception("Connection timed out"))
    assert await resolver.resolve_stream_url("tt123", "movie", "File.mkv") is None

    # 3. Empty base URL
    assert await resolver.resolve_stream_url("tt123", "movie", "File.mkv", base_url="") is None


@pytest.mark.asyncio
async def test_resolve_stream_url_elfhosted_mediafusion_comet():
    """Verify various public and debrid formats (MediaFusion, Comet, Elfhosted)."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "streams": [
            {
                "name": "[RD+] MediaFusion 4K",
                "description": "📁 Inception.2010.2160p.UHD.BluRay.x265-TERMiNAL.mkv\n💾 24.5 GB",
                "url": "https://mediafusion.elfhosted.com/streaming/playback/12345/video.mkv",
            },
            {
                "name": "[RD+] Comet 1080p",
                "title": "Inception.2010.1080p.BluRay.x264-SPARKS.mkv",
                "url": "https://comet.elfhosted.com/stream/54321/video.mkv",
            }
        ]
    }
    client = AsyncMock()
    client.get = AsyncMock(return_value=mock_resp)

    resolver = StreamResolver(client)
    # Test stremio:// URL from Elfhosted
    url = await resolver.resolve_stream_url(
        imdb_id="tt1375666",
        media_type="movie",
        filename="Inception.2010.2160p.UHD.BluRay.x265-TERMiNAL.mkv",
        base_url="stremio://mediafusion.elfhosted.com/my-config/manifest.json",
    )
    assert url == "https://mediafusion.elfhosted.com/streaming/playback/12345/video.mkv"

