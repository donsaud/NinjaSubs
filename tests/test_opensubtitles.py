"""Unit tests for OpenSubtitles.com v1 REST API provider implementation."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app.cache import cache_manager
from app.main import app
from app.models import SubtitleRelease
from app.providers.opensubtitles import OpenSubtitlesProvider
from app.utils.config_parser import encode_user_config


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.asyncio
async def test_opensubtitles_missing_api_key_returns_empty(monkeypatch):
    """When no API key is provided and none in env settings, search immediately returns empty list."""
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "OPENSUBTITLES_API_KEY", "")
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    provider = OpenSubtitlesProvider(mock_client)

    results = await provider.search_subtitles(imdb_id="tt0111161", api_key="")
    assert results == []
    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_opensubtitles_search_movie_v1():
    """Verify OpenSubtitles movie search queries /subtitles with correct headers and numeric IMDb ID."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    captured_requests = []

    def mock_get(url, params=None, headers=None, timeout=None, **kwargs):
        captured_requests.append(
            {"url": url, "params": params, "headers": headers, "kwargs": kwargs}
        )
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.return_value = {
            "total_pages": 1,
            "total_count": 2,
            "data": [
                {
                    "id": "item_1",
                    "type": "subtitle",
                    "attributes": {
                        "subtitle_id": "111111",
                        "language": "ar",
                        "release": "The.Shawshank.Redemption.1994.1080p.BluRay.x264-ARABIC",
                        "hearing_impaired": False,
                        "files": [
                            {"file_id": 98765, "file_name": "The.Shawshank.Redemption.1994.srt"}
                        ],
                    },
                },
                {
                    "id": "item_2",
                    "type": "subtitle",
                    "attributes": {
                        "subtitle_id": "222222",
                        "language": "ar",
                        "release": "The.Shawshank.Redemption.1994.720p.HDTV",
                        "hearing_impaired": True,
                        "files": [
                            {
                                "file_id": 98766,
                                "file_name": "The.Shawshank.Redemption.1994.720p.srt",
                            }
                        ],
                    },
                },
            ],
        }
        return resp

    mock_client.get.side_effect = mock_get

    provider = OpenSubtitlesProvider(mock_client)
    results = await provider.search_subtitles(
        imdb_id="tt0111161",
        is_series=False,
        api_key="test_opensubtitles_key_123",
        languages=["ara"],
        exclude_hi=False,
    )

    assert len(results) == 2
    assert len(captured_requests) == 1

    req = captured_requests[0]
    assert req["url"] == "https://api.opensubtitles.com/api/v1/subtitles"
    assert (
        req["params"]["imdb_id"] == "111161"
    )  # tt0111161 -> numeric string without leading tt/zeroes
    assert req["params"]["languages"] == "ar"
    assert req["params"]["type"] == "movie"
    assert "hearing_impaired" not in req["params"]

    assert req["headers"]["Api-Key"] == "test_opensubtitles_key_123"
    assert req["headers"]["User-Agent"] == "StremioArabicSubs v1.0.0"
    assert req["headers"]["Accept"] == "application/json"
    assert req["kwargs"].get("follow_redirects") is True

    item1 = results[0]
    assert item1.provider == "opensubtitles"
    assert item1.release_name == "The.Shawshank.Redemption.1994.1080p.BluRay.x264-ARABIC"
    assert item1.download_url == "/sub/opensubtitles/98765.srt"
    assert item1.hearing_impaired is False
    assert item1.lang == "ara"

    item2 = results[1]
    assert item2.hearing_impaired is True


@pytest.mark.asyncio
async def test_opensubtitles_search_extracts_uploader():
    """Verify the uploader username is extracted from attributes.uploader.name."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    def mock_get(url, params=None, headers=None, timeout=None, **kwargs):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.return_value = {
            "data": [
                {
                    "id": "item_1",
                    "attributes": {
                        "language": "ar",
                        "release": "Movie.2024.1080p.BluRay",
                        "uploader": {"name": "subsmaster", "uploader_id": 42},
                        "files": [{"file_id": 111, "file_name": "Movie.2024.srt"}],
                    },
                },
                {
                    "id": "item_2",
                    "attributes": {
                        "language": "ar",
                        "release": "Movie.2024.720p.HDTV",
                        "files": [{"file_id": 222, "file_name": "Movie.2024.720p.srt"}],
                    },
                },
            ]
        }
        return resp

    mock_client.get.side_effect = mock_get

    provider = OpenSubtitlesProvider(mock_client)
    results = await provider.search_subtitles(
        imdb_id="tt0111161",
        api_key="os_key",
        languages=["ara"],
    )

    assert len(results) == 2
    assert results[0].uploader == "subsmaster"
    assert results[1].uploader == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("video_hash", [None, "", "   ", "8e245d9679d31e12"])
@pytest.mark.parametrize(
    "confirmation",
    [
        {},
        {"moviehash_match": None},
        {"moviehash_match": False},
        {"moviehash_match": "false"},
        {"moviehash_match": "true"},
        {"moviehash_match": 1},
        {"moviehash_match": True},
    ],
)
async def test_hash_priority_requires_requested_hash_and_boolean_confirmation(
    video_hash, confirmation
):
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get.return_value = httpx.Response(
        200,
        json={
            "data": [
                {
                    "attributes": {
                        "release": "Show.S01E03.WEB-DL",
                        "language": "ar",
                        "files": [{"file_id": 12345}],
                        **confirmation,
                    }
                }
            ],
        },
    )
    provider = OpenSubtitlesProvider(mock_client)

    results = await provider.search_subtitles(
        imdb_id="tt0903747",
        api_key="test_key",
        video_hash=video_hash,
    )

    expected = bool((video_hash or "").strip()) and confirmation.get("moviehash_match") is True
    assert len(results) == 1
    assert results[0].is_hash_match is expected

    from app.services.subtitle_matcher import rank_subtitles

    ranked = rank_subtitles("Show.S01E02.WEB-DL", results)
    assert bool(ranked) is expected


@pytest.mark.asyncio
async def test_opensubtitles_search_series_v1():
    """Verify series search adds season_number, episode_number, and type=episode."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    captured_requests = []

    def mock_get(url, params=None, headers=None, timeout=None, **kwargs):
        captured_requests.append({"url": url, "params": params, "headers": headers})
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.return_value = {
            "data": [
                {
                    "id": "sub_ep_1",
                    "attributes": {
                        "language": "ar",
                        "release": "Breaking.Bad.S05E16.1080p.BluRay",
                        "hearing_impaired": False,
                        "files": [{"file_id": 55555, "file_name": "Breaking.Bad.S05E16.srt"}],
                    },
                }
            ]
        }
        return resp

    mock_client.get.side_effect = mock_get

    provider = OpenSubtitlesProvider(mock_client)
    results = await provider.search_subtitles(
        imdb_id="tt0903747:5:16",
        is_series=True,
        season=5,
        episode=16,
        api_key="valid_key",
    )

    assert len(results) == 1
    req = captured_requests[0]
    assert req["params"]["imdb_id"] == "903747"
    assert req["params"]["type"] == "episode"
    assert req["params"]["season_number"] == 5
    assert req["params"]["episode_number"] == 16


@pytest.mark.asyncio
async def test_opensubtitles_exclude_hi():
    """Verify exclude_hi sends hearing_impaired=exclude and filters out any HI items from results."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    captured_requests = []

    def mock_get(url, params=None, headers=None, timeout=None, **kwargs):
        captured_requests.append({"url": url, "params": params, "headers": headers})
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.return_value = {
            "data": [
                {
                    "attributes": {
                        "language": "ar",
                        "release": "Non.HI.Release",
                        "hearing_impaired": False,
                        "files": [{"file_id": 111}],
                    }
                },
                {
                    "attributes": {
                        "language": "ar",
                        "release": "HI.Release",
                        "hearing_impaired": True,
                        "files": [{"file_id": 222}],
                    }
                },
            ]
        }
        return resp

    mock_client.get.side_effect = mock_get

    provider = OpenSubtitlesProvider(mock_client)
    results = await provider.search_subtitles(
        imdb_id="tt0111161",
        api_key="valid_key",
        exclude_hi=True,
    )

    assert captured_requests[0]["params"]["hearing_impaired"] == "exclude"
    assert len(results) == 1
    assert results[0].release_name == "Non.HI.Release"


@pytest.mark.asyncio
async def test_opensubtitles_multi_language_mapping():
    """Verify ISO-639-2 codes are converted to OpenSubtitles ISO-639-1 format (ara, eng -> ar, en)."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    captured_requests = []

    def mock_get(url, params=None, headers=None, timeout=None, **kwargs):
        captured_requests.append({"params": params})
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.return_value = {"data": []}
        return resp

    mock_client.get.side_effect = mock_get

    provider = OpenSubtitlesProvider(mock_client)
    await provider.search_subtitles(
        imdb_id="tt0111161",
        api_key="valid_key",
        languages=["ara", "eng"],
    )

    assert captured_requests[0]["params"]["languages"] == "ar,en"


@pytest.mark.asyncio
async def test_opensubtitles_download_archive_success():
    """Verify download_archive resolves download link via POST /download and fetches file content."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    def mock_post(url, json=None, headers=None, timeout=None, **kwargs):
        assert url == "https://api.opensubtitles.com/api/v1/download"
        assert json == {"file_id": 12345}
        assert headers["Api-Key"] == "valid_key"
        assert kwargs.get("follow_redirects") is True
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.return_value = {
            "link": "https://download.opensubtitles.com/temp/12345.srt",
            "file_name": "subtitle.srt",
            "requests": 1,
            "remaining": 99,
        }
        return resp

    def mock_get(url, timeout=None, follow_redirects=True):
        assert url == "https://download.opensubtitles.com/temp/12345.srt"
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.content = b"1\n00:00:01,000 --> 00:00:04,000\nHello world\n"
        return resp

    mock_client.post.side_effect = mock_post
    mock_client.get.side_effect = mock_get

    provider = OpenSubtitlesProvider(mock_client)
    content = await provider.download_archive(
        "/sub/opensubtitles/12345.srt",
        api_key="valid_key",
    )

    assert content is not None
    assert b"Hello world" in content


@pytest.mark.asyncio
async def test_opensubtitles_download_archive_direct_http_fallback():
    """When download_ref is already a direct HTTP link and no key or post needed."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    def mock_get(url, timeout=None, follow_redirects=True):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.content = b"1\n00:00:01,000 --> 00:00:02,000\nDirect link test\n"
        return resp

    mock_client.get.side_effect = mock_get

    provider = OpenSubtitlesProvider(mock_client)
    content = await provider.download_archive(
        "https://cdn.opensubtitles.org/subtitles/direct.srt",
        api_key="",
    )

    assert content is not None
    assert b"Direct link test" in content


@pytest.mark.asyncio
async def test_opensubtitles_resilient_error_handling():
    """Verify provider handles 401, 429, 500, and network errors gracefully without crashing."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    provider = OpenSubtitlesProvider(mock_client)

    # 401 Unauthorized
    mock_client.get.return_value = MagicMock(status_code=401, text="Unauthorized")
    assert await provider.search_subtitles("tt0111161", api_key="bad_key") == []

    # 429 Rate limited
    mock_client.get.return_value = MagicMock(status_code=429, text="Too Many Requests")
    assert await provider.search_subtitles("tt0111161", api_key="valid_key") == []

    # 500 Internal error
    mock_client.get.return_value = MagicMock(status_code=500, text="Internal Server Error")
    assert await provider.search_subtitles("tt0111161", api_key="valid_key") == []

    # Timeout
    mock_client.get.side_effect = httpx.TimeoutException("Timeout")
    assert await provider.search_subtitles("tt0111161", api_key="valid_key") == []


@pytest.mark.asyncio
async def test_opensubtitles_search_follows_redirects():
    """Verify search_subtitles passes follow_redirects=True to handle HTTP 301 gracefully."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    captured = {}

    def mock_get(url, **kwargs):
        captured["kwargs"] = kwargs
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.return_value = {"data": []}
        return resp

    mock_client.get.side_effect = mock_get
    provider = OpenSubtitlesProvider(mock_client)
    await provider.search_subtitles("tt0111161", api_key="my_key")

    assert captured["kwargs"].get("follow_redirects") is True


@pytest.mark.asyncio
async def test_opensubtitles_stremio_endpoint_integration(client):
    """Test full integration: passing opensubtitles_key queries OpenSubtitles and formats label correctly."""
    mock_os_release = SubtitleRelease(
        release_name="Gladiator.2000.1080p.BluRay.x264-AMIABLE",
        download_url="/sub/opensubtitles/99999.srt",
        provider="opensubtitles",
        lang="ara",
    )

    user_cfg = encode_user_config(
        subdl_key="subdl_key",
        subsource_key="subsource_key",
        opensubtitles_key="os_key_123",
        enable_opensubtitles=True,
        languages=["ara"],
    )

    with (
        patch("app.providers.subdl.SubdlProvider.search_subtitles", new=AsyncMock(return_value=[])),
        patch(
            "app.providers.subsource.SubsourceProvider.search_subtitles",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "app.providers.opensubtitles.OpenSubtitlesProvider.search_subtitles",
            new=AsyncMock(return_value=[mock_os_release]),
        ),
    ):
        resp = client.get(f"/{user_cfg}/subtitles/movie/tt0172495.json")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["subtitles"]) >= 1

        sub = data["subtitles"][0]
        # Label format: [{score}%] [OpenSubtitles] {cleaned_release_name}
        assert "[OpenSubtitles]" in sub["title"]
        assert sub["lang"] == "ara"
        # URL may carry media-context query params for auto-sync.
        assert "/sub/opensubtitles/99999.srt" in sub["url"]


@pytest.mark.asyncio
async def test_opensubtitles_serve_subtitle_endpoint(client):
    """Verify _serve_subtitle_handler properly calls OpenSubtitlesProvider.download_archive and caches result."""
    import uuid

    sub_id = f"os_{uuid.uuid4().hex[:12]}"
    cache_manager.store_metadata(
        sub_id,
        {
            "sub_id": sub_id,
            "provider": "opensubtitles",
            "download_url": "/sub/opensubtitles/77777.srt",
            "release_name": "Gladiator.2000.1080p.BluRay",
            "opensubtitles_key": "my_os_key",
            "lang": "ara",
        },
    )

    fake_srt_bytes = b"1\n00:00:01,000 --> 00:00:03,000\nSubtitle delivered successfully\n"

    with (
        patch(
            "app.providers.opensubtitles.OpenSubtitlesProvider.download_archive",
            new=AsyncMock(return_value=fake_srt_bytes),
        ) as mock_dl,
        patch(
            "app.cache.cache_manager.get_subtitle",
            new=AsyncMock(return_value=None),
        ),
    ):
        resp = client.get(f"/sub/{sub_id}.srt")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/x-subrip")
        assert b"Subtitle delivered successfully" in resp.content
        mock_dl.assert_called_once_with(
            "/sub/opensubtitles/77777.srt",
            api_key="my_os_key",
        )


@pytest.mark.asyncio
async def test_opensubtitles_proxy_stream_direct_delivery(client):
    """Verify /sub/opensubtitles/{file_id}.srt directly downloads, transcodes, and normalizes Arabic."""
    file_id = 987654321
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b"1\n00:00:01,000 --> 00:00:04,000\n\xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7 \xd8\xa8\xd9\x83.\n"

    with (
        patch(
            "app.providers.opensubtitles.OpenSubtitlesProvider.get_download_url",
            new=AsyncMock(return_value=f"https://download.opensubtitles.com/temp/{file_id}.srt"),
        ),
        patch("app.cache.cache_manager.get_subtitle", new=AsyncMock(return_value=None)),
        patch("app.main._http_client.get", new=AsyncMock(return_value=mock_resp)),
    ):
        resp = client.get(f"/sub/opensubtitles/{file_id}.srt?api_key=my_key")
        assert resp.status_code == 200
        # Arabic line ends with a period -> the UBA fix appends RLM (U+200F).
        assert resp.content == mock_resp.content.replace(b".\n", b".\xe2\x80\x8f\n")
        assert "application/x-subrip" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_opensubtitles_proxy_stream_configured_direct_delivery(client):
    """Verify /{config}/sub/opensubtitles/{file_id}.srt extracts key and delivers normalized subtitle."""
    file_id = 543210987
    user_cfg = encode_user_config(opensubtitles_key="cfg_os_key")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b"1\n00:00:01,000 --> 00:00:04,000\n\xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7 \xd8\xa8\xd9\x83.\n"

    with (
        patch(
            "app.providers.opensubtitles.OpenSubtitlesProvider.get_download_url",
            new=AsyncMock(return_value=f"https://download.opensubtitles.com/temp/{file_id}.srt"),
        ) as mock_get_dl,
        patch("app.cache.cache_manager.get_subtitle", new=AsyncMock(return_value=None)),
        patch("app.main._http_client.get", new=AsyncMock(return_value=mock_resp)),
    ):
        resp = client.get(f"/{user_cfg}/sub/opensubtitles/{file_id}.srt")
        assert resp.status_code == 200
        # Arabic line ends with a period -> the UBA fix appends RLM (U+200F).
        assert resp.content == mock_resp.content.replace(b".\n", b".\xe2\x80\x8f\n")
        mock_get_dl.assert_called_once_with(file_id, "cfg_os_key")


@pytest.mark.asyncio
async def test_opensubtitles_proxy_stream_not_found(client):
    """Verify proxy route raises 404 when download URL cannot be resolved (expired or limit)."""
    with (
        patch(
            "app.providers.opensubtitles.OpenSubtitlesProvider.get_download_url",
            new=AsyncMock(return_value=None),
        ),
        patch("app.cache.cache_manager.get_subtitle", new=AsyncMock(return_value=None)),
    ):
        resp = client.get("/sub/opensubtitles/999999999.srt?api_key=my_key", follow_redirects=False)
        assert resp.status_code == 404
        assert "expired or limit reached" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_opensubtitles_proxy_stream_fallback_to_subsource(client):
    """Verify that when OpenSubtitles hits quota limit (get_download_url returns None), it falls back to Subsource."""
    file_id = 88888888
    cache_manager.store_metadata(
        str(file_id),
        {
            "sub_id": "abc123subid",
            "imdb_id": "tt9288030",
            "media_type": "series",
            "provider": "opensubtitles",
            "download_url": f"/sub/opensubtitles/{file_id}.srt",
            "release_name": "Reacher.S01E01.1080p.WEB-DL",
            "season": 1,
            "episode": 1,
            "subsource_key": "test_subsource_key",
            "lang": "ara",
        },
    )

    fake_arabic_srt = b"1\n00:00:01,000 --> 00:00:04,000\n\xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7 \xd8\xa8\xd9\x83\n"

    with (
        patch(
            "app.providers.opensubtitles.OpenSubtitlesProvider.get_download_url",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "app.main._fallback_download_subsource",
            new=AsyncMock(return_value=fake_arabic_srt),
        ) as mock_fallback,
        patch(
            "app.cache.cache_manager.get_subtitle",
            new=AsyncMock(return_value=None),
        ),
    ):
        resp = client.get(f"/sub/opensubtitles/{file_id}.srt")
        assert resp.status_code == 200
        assert resp.content == fake_arabic_srt
        assert "application/x-subrip" in resp.headers["content-type"]
        assert resp.headers["access-control-allow-origin"] == "*"
        assert "HEAD" in resp.headers["access-control-allow-methods"]
        mock_fallback.assert_called_once()


@pytest.mark.asyncio
async def test_subdl_stream_fallback_to_subsource(client):
    """Verify that when Subdl hits daily limit (HTTP 429), _serve_subtitle_handler falls back to Subsource."""
    sub_id = "subdl_fail_1234"
    cache_manager.store_metadata(
        sub_id,
        {
            "sub_id": sub_id,
            "imdb_id": "tt9288030",
            "media_type": "series",
            "provider": "subdl",
            "download_url": "https://subdl.com/sub/fake.zip",
            "release_name": "Reacher.S01E01.1080p.WEB-DL",
            "season": 1,
            "episode": 1,
            "subdl_key": "test_subdl_key",
            "subsource_key": "test_subsource_key",
            "lang": "ara",
        },
    )

    fake_arabic_srt = b"1\n00:00:01,000 --> 00:00:04,000\n\xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7 \xd8\xa8\xd9\x83\n"

    with (
        patch(
            "app.providers.subdl.SubdlProvider.download_archive",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "app.main._fallback_download_subsource",
            new=AsyncMock(return_value=fake_arabic_srt),
        ) as mock_fallback,
        patch(
            "app.cache.cache_manager.get_subtitle",
            new=AsyncMock(return_value=None),
        ),
    ):
        resp = client.get(f"/sub/{sub_id}.srt")
        assert resp.status_code == 200
        assert resp.content == fake_arabic_srt
        assert "application/x-subrip" in resp.headers["content-type"]
        mock_fallback.assert_called_once()


@pytest.mark.asyncio
async def test_head_and_options_requests(client):
    """Verify HEAD and OPTIONS requests succeed on manifest and subtitle routes with CORS headers."""
    # 1. Manifest HEAD
    resp = client.head("/manifest.json")
    assert resp.status_code == 200
    assert "HEAD" in resp.headers["access-control-allow-methods"]

    # 2. Subtitles query HEAD
    with patch("app.main._fetch_subtitles_handler", new=AsyncMock()) as mock_fetch:
        from app.models import SubtitlesResponse

        mock_fetch.return_value = SubtitlesResponse(subtitles=[])
        resp_sub = client.head("/subtitles/series/tt9288030:1:1.json")
        assert resp_sub.status_code == 200
        assert "HEAD" in resp_sub.headers["access-control-allow-methods"]


def test_exact_fallback_release_matching():
    from types import SimpleNamespace

    from app.main import _is_exact_fallback_release

    requested = "Sopranos.S01E03.1080p.WEB-DL.DD5.1.H.264-BS.srt"
    assert _is_exact_fallback_release(
        requested,
        " Ahmed ",
        False,
        SimpleNamespace(
            release_name="Sopranos S01E03 1080p WEB-DL DD5.1 H.264-BS",
            uploader="ahmed",
            hearing_impaired=False,
        ),
    ) is True
    assert _is_exact_fallback_release(
        requested,
        "Ahmed",
        False,
        SimpleNamespace(
            release_name="Sopranos.S01E11.1080p.WEB-DL.srt",
            uploader="Ahmed",
            hearing_impaired=False,
        ),
    ) is False
    assert _is_exact_fallback_release(
        requested,
        "Ahmed",
        False,
        SimpleNamespace(
            release_name=requested,
            uploader="SomeoneElse",
            hearing_impaired=False,
        ),
    ) is False
    assert _is_exact_fallback_release("", "Ahmed", False, SimpleNamespace(release_name=requested)) is False


@pytest.mark.asyncio
async def test_subsource_fallback_uses_weak_compatible_candidate_when_not_exact():
    """No exact release is a reason to keep looking, not a reason to fail.

    Previously a SubSource season pack was refused outright whenever it was not
    the exact requested release name, so a candidate that the target's
    compatibility engine accepts was never even downloaded, let alone tried.
    Eligibility now stops at ``hard_compatibility_filter``; deciding whether it
    is actually synchronizable is the pipeline's job, not the provider's.
    """
    from app.main import _fallback_download_subsource

    requested_name = "Sopranos.S01E03.1080p.WEB-DL.DD5.1.H.264-BS.srt"
    pack_srt = b"1\n00:00:01,000 --> 00:00:04,000\nPack candidate\n"
    download_calls: list[str] = []

    class _GenericSubsource:
        async def search_subtitles(self, **kwargs):
            return [
                SubtitleRelease(
                    release_name="Sopranos 1-11.srt",
                    download_url="https://subsource.test/generic.zip",
                    provider="subsource",
                    lang="ara",
                    uploader="generic-uploader",
                )
            ]

        async def download_archive(self, download_ref, api_key=None):
            download_calls.append(download_ref)
            return pack_srt

    meta: dict = {"release_name": requested_name, "uploader": "Ahmed"}
    outcome: dict = {}
    with patch("app.main.SubsourceProvider", new=lambda client: _GenericSubsource()):
        result = await _fallback_download_subsource(
            imdb_id="tt0141842",
            media_type="series",
            season=1,
            episode=3,
            subsource_key="test_subsource_key",
            target_filename=requested_name,
            lang="ara",
            client=object(),
            requested_release_name=requested_name,
            requested_uploader="Ahmed",
            requested_hearing_impaired=False,
            meta=meta,
            outcome=outcome,
        )
    assert result == pack_srt
    assert download_calls == ["https://subsource.test/generic.zip"]
    # The two facts are reported separately: no exact release, but one
    # target-compatible candidate.
    assert outcome["exact_candidates"] == 0
    assert outcome["compatible_candidates"] == 1
    assert outcome["category"] == "NO_EXACT_RELEASE"


@pytest.mark.asyncio
async def test_subsource_fallback_refuses_wrong_episode_release():
    """Eligibility still ends at the hard filter: a wrong episode is refused."""
    from app.main import _fallback_download_subsource

    requested_name = "Sopranos.S01E03.1080p.WEB-DL.DD5.1.H.264-BS.srt"
    download_calls: list[str] = []

    class _WrongEpisode:
        async def search_subtitles(self, **kwargs):
            return [
                SubtitleRelease(
                    release_name="Sopranos.S01E09.1080p.WEB-DL.DD5.1.H.264-BS.srt",
                    download_url="https://subsource.test/wrong.zip",
                    provider="subsource",
                    lang="ara",
                    uploader="Ahmed",
                )
            ]

        async def download_archive(self, download_ref, api_key=None):
            download_calls.append(download_ref)
            raise AssertionError("a wrong-episode candidate must never be downloaded")

    meta: dict = {"release_name": requested_name, "uploader": "Ahmed"}
    outcome: dict = {}
    with patch("app.main.SubsourceProvider", new=lambda client: _WrongEpisode()):
        result = await _fallback_download_subsource(
            imdb_id="tt0141842",
            media_type="series",
            season=1,
            episode=3,
            subsource_key="test_subsource_key",
            target_filename=requested_name,
            lang="ara",
            client=object(),
            requested_release_name=requested_name,
            requested_uploader="Ahmed",
            requested_hearing_impaired=False,
            meta=meta,
            outcome=outcome,
        )
    assert result is None
    assert download_calls == []
    assert outcome["category"] == "NO_COMPATIBLE_CANDIDATE"
    assert outcome["exact_candidates"] == 0
    assert outcome["compatible_candidates"] == 0


@pytest.mark.asyncio
async def test_subsource_fallback_uses_exact_equivalent_release():
    from app.main import _fallback_download_subsource

    requested_name = "Sopranos.S01E03.1080p.WEB-DL.DD5.1.H.264-BS.srt"
    exact_srt = b"1\n00:00:01,000 --> 00:00:04,000\nExact equivalent\n"
    download_calls: list[str] = []

    class _ExactSubsource:
        async def search_subtitles(self, **kwargs):
            return [
                SubtitleRelease(
                    release_name="Sopranos 1-11.srt",
                    download_url="https://subsource.test/generic.zip",
                    provider="subsource",
                    lang="ara",
                    uploader="generic-uploader",
                ),
                SubtitleRelease(
                    release_name="Sopranos.S01E03.1080p.WEB-DL.DD5.1.H.264-BS",
                    download_url="https://subsource.test/exact.srt",
                    provider="subsource",
                    lang="ara",
                    uploader="Ahmed",
                ),
            ]

        async def download_archive(self, download_ref, api_key=None):
            download_calls.append(download_ref)
            assert download_ref == "https://subsource.test/exact.srt"
            return exact_srt

    meta: dict = {"release_name": requested_name, "uploader": "Ahmed"}
    with patch("app.main.SubsourceProvider", new=lambda client: _ExactSubsource()):
        result = await _fallback_download_subsource(
            imdb_id="tt0141842",
            media_type="series",
            season=1,
            episode=3,
            subsource_key="test_subsource_key",
            target_filename=requested_name,
            lang="ara",
            client=object(),
            requested_release_name=requested_name,
            requested_uploader="Ahmed",
            requested_hearing_impaired=False,
            meta=meta,
        )
    assert result == exact_srt
    assert download_calls == ["https://subsource.test/exact.srt"]
    assert meta["provider"] == "subsource"
    assert meta["download_url"] == "https://subsource.test/exact.srt"
    assert meta["fallback_download_url"] == "https://subsource.test/exact.srt"
    assert meta["uploader"] == "Ahmed"


@pytest.mark.asyncio
async def test_subdl_stream_without_exact_subsource_match_returns_502(client):
    """A failed SubDL ID must fail alone rather than receive a generic SubSource file."""
    sub_id = "subdl_no_exact_match"
    requested_name = "Sopranos.S01E03.1080p.WEB-DL.DD5.1.H.264-BS.srt"
    cache_manager.store_metadata(
        sub_id,
        {
            "sub_id": sub_id,
            "imdb_id": "tt0141842",
            "media_type": "series",
            "provider": "subdl",
            "download_url": "https://subdl.com/sub/fake.zip",
            "release_name": requested_name,
            "season": 1,
            "episode": 3,
            "subdl_key": "test_subdl_key",
            "subsource_key": "test_subsource_key",
            "lang": "ara",
            "uploader": "Ahmed",
        },
    )

    class _GenericSubsource:
        async def search_subtitles(self, **kwargs):
            return [
                SubtitleRelease(
                    release_name="Sopranos 1-11.srt",
                    download_url="https://subsource.test/generic.zip",
                    provider="subsource",
                    lang="ara",
                    uploader="generic-uploader",
                )
            ]

        async def download_archive(self, download_ref, api_key=None):
            raise AssertionError("generic fallback must not be downloaded")

    with (
        patch(
            "app.providers.subdl.SubdlProvider.download_archive",
            new=AsyncMock(return_value=None),
        ),
        patch("app.main.SubsourceProvider", new=lambda client: _GenericSubsource()),
        patch(
            "app.cache.cache_manager.get_subtitle",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "app.cache.cache_manager.save_subtitle",
            new=AsyncMock(),
        ) as mock_save,
    ):
        resp = client.get(f"/sub/{sub_id}.srt")
        assert resp.status_code == 502
        assert mock_save.await_count == 0


def _zip_of(files: dict) -> bytes:
    """Build an in-memory ZIP from {name: str|bytes}."""
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content.encode("utf-8") if isinstance(content, str) else content)
    return buf.getvalue()


def _store_subdl_meta(sub_id: str) -> None:
    cache_manager.store_metadata(
        sub_id,
        {
            "sub_id": sub_id,
            "imdb_id": "tt9288030",
            "media_type": "series",
            "provider": "subdl",
            "download_url": "https://subdl.com/sub/bad.zip",
            "release_name": "Reacher.S01E01.1080p.WEB-DL",
            "season": 1,
            "episode": 1,
            "subdl_key": "test_subdl_key",
            "subsource_key": "test_subsource_key",
            "lang": "ara",
        },
    )


@pytest.mark.asyncio
async def test_subdl_invalid_zip_returns_502_not_500(client):
    """A ZIP with no recognized subtitle falls back, then returns 502 (never 500)."""
    sub_id = "subdl_invalid_zip_0001"
    _store_subdl_meta(sub_id)
    bad_zip = _zip_of({"info.nfo": "no subs here", "cover.jpg": b"\xff\xd8\xff"})

    with (
        patch(
            "app.providers.subdl.SubdlProvider.download_archive",
            new=AsyncMock(return_value=bad_zip),
        ),
        patch(
            "app.main._fallback_download_subsource",
            new=AsyncMock(return_value=None),
        ) as mock_fallback,
        patch("app.cache.cache_manager.get_subtitle", new=AsyncMock(return_value=None)),
        patch(
            "app.cache.cache_manager.save_subtitle", new=AsyncMock(return_value=True)
        ) as mock_save,
    ):
        resp = client.get(f"/sub/{sub_id}.srt")

    assert resp.status_code == 502
    assert resp.status_code != 500
    assert "no usable subtitle" in resp.json()["detail"].lower()
    mock_fallback.assert_called_once()
    # The invalid archive must never be persisted.
    mock_save.assert_not_called()


@pytest.mark.asyncio
async def test_subdl_invalid_zip_falls_back_to_subsource(client):
    """A valid-but-empty ZIP triggers the SubSource fallback and serves it."""
    sub_id = "subdl_invalid_zip_0002"
    _store_subdl_meta(sub_id)
    bad_zip = _zip_of({"info.nfo": "nothing useful"})
    fake_arabic = b"1\n00:00:01,000 --> 00:00:03,000\n\xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7\n"

    with (
        patch(
            "app.providers.subdl.SubdlProvider.download_archive",
            new=AsyncMock(return_value=bad_zip),
        ),
        patch(
            "app.main._fallback_download_subsource",
            new=AsyncMock(return_value=fake_arabic),
        ) as mock_fallback,
        patch("app.cache.cache_manager.get_subtitle", new=AsyncMock(return_value=None)),
        patch(
            "app.cache.cache_manager.save_subtitle", new=AsyncMock(return_value=True)
        ) as mock_save,
    ):
        resp = client.get(f"/sub/{sub_id}.srt")

    assert resp.status_code == 200
    assert resp.content == fake_arabic
    mock_fallback.assert_called_once()
    # Only the resolved fallback payload is cached.
    mock_save.assert_called_once()


@pytest.mark.asyncio
async def test_cache_refuses_empty_subtitle(tmp_path):
    """Empty/whitespace payloads are never persisted (invalid archive guard)."""
    from app.cache import LRUCacheManager

    manager = LRUCacheManager(cache_dir=str(tmp_path), max_bytes=10000, max_files=5)
    assert await manager.save_subtitle("empty", b"") is False
    assert await manager.save_subtitle("blank", b"   \n\t") is False
    assert await manager.get_subtitle("empty") is None


@pytest.mark.asyncio
async def test_failed_sub_id_short_circuits_retries(client):
    """A broken upstream archive is hit once; retries are answered from the failure cache."""
    sub_id = "subdl_broken_retry_0001"
    _store_subdl_meta(sub_id)
    bad_zip = _zip_of({"Suits.S01.720p.Web-DL.ReEnc-DeeJayAhmed.txt": "notice"})
    download = AsyncMock(return_value=bad_zip)

    with (
        patch("app.providers.subdl.SubdlProvider.download_archive", new=download),
        patch(
            "app.main._fallback_download_subsource", new=AsyncMock(return_value=None)
        ),
        patch("app.cache.cache_manager.get_subtitle", new=AsyncMock(return_value=None)),
    ):
        first = client.get(f"/sub/{sub_id}.srt")
        assert first.status_code == 502
        assert cache_manager.is_failed(sub_id) is True

        second = client.get(f"/sub/{sub_id}.srt")
        assert second.status_code == 502
        assert "recent extraction failure" in second.json()["detail"]

    # Two retries, but the upstream archive was only fetched once.
    assert download.await_count == 1


@pytest.mark.asyncio
async def test_extract_error_reports_archive_members():
    """The extraction error must name the archive members for diagnosis."""
    from app.extractor import SubtitleExtractionError, extract_srt_from_zip

    zip_bytes = _zip_of({"Suits.S01.720p.Web-DL.ReEnc-DeeJayAhmed.txt": "notice"})
    with pytest.raises(SubtitleExtractionError, match="No valid subtitle files") as excinfo:
        extract_srt_from_zip(zip_bytes)
    assert "Suits.S01.720p.Web-DL.ReEnc-DeeJayAhmed.txt" in str(excinfo.value)


@pytest.mark.asyncio
async def test_serve_rejects_non_subtitle_payload(client):
    """A raw (non-archive) HTML/notice body must never be cached or served."""
    sub_id = "subdl_non_subtitle_0001"
    _store_subdl_meta(sub_id)
    html = b"<html><body>upstream error</body></html>"
    with (
        patch(
            "app.providers.subdl.SubdlProvider.download_archive",
            new=AsyncMock(return_value=html),
        ),
        patch("app.main._fallback_download_subsource", new=AsyncMock(return_value=None)),
        patch("app.cache.cache_manager.get_subtitle", new=AsyncMock(return_value=None)),
        patch(
            "app.cache.cache_manager.save_subtitle", new=AsyncMock(return_value=True)
        ) as mock_save,
    ):
        resp = client.get(f"/sub/{sub_id}.srt")
    assert resp.status_code == 502
    mock_save.assert_not_called()
    assert cache_manager.is_failed(sub_id) is True


@pytest.mark.asyncio
async def test_serve_extracts_rar_archive(client, monkeypatch):
    """A RAR archive (DVDRip packs) is extracted, not served as binary."""
    sub_id = "subdl_rar_0001"
    _store_subdl_meta(sub_id)
    srt = "1\n00:00:01,000 --> 00:00:02,000\nhello rar\n"
    monkeypatch.setattr(
        "app.extractor._extract_nonzip_members",
        lambda raw, kind: [("fl-itw-xvid-cd2.srt", srt.encode("utf-8"))],
    )
    with (
        patch(
            "app.providers.subdl.SubdlProvider.download_archive",
            new=AsyncMock(return_value=b"Rar!\x1a\x07\x00fake"),
        ),
        patch("app.cache.cache_manager.get_subtitle", new=AsyncMock(return_value=None)),
        patch(
            "app.cache.cache_manager.save_subtitle", new=AsyncMock(return_value=True)
        ) as mock_save,
    ):
        resp = client.get(f"/sub/{sub_id}.srt")
    assert resp.status_code == 200
    assert b"hello rar" in resp.content
    mock_save.assert_called_once()
