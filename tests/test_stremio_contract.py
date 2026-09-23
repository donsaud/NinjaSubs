"""Contract tests for Stremio Addon Protocol v3 endpoints and JSON schemas."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app.cache import cache_manager
from app.main import app
from app.models import SubtitleRelease
from app.providers.subdl import SubdlProvider


@pytest.fixture
def client():
    """FastAPI TestClient fixture with initialized mock HTTP client."""
    with TestClient(app) as test_client:
        yield test_client


def test_manifest_schema_and_cors(client):
    """
    Assert /manifest.json matches exact official Stremio Addon SDK specification
    and contains required CORS headers.
    """
    response = client.get("/manifest.json")
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "*"
    data = response.json()
    assert data["id"] == "org.ninjasubs.addon"
    assert data["name"] == "NinjaSubs"
    assert data["description"] == (
        "Smart, high-accuracy subtitle aggregator for Stremio featuring advanced Arabic subtitle optimization."
    )
    assert data["version"] == "1.0.0"
    assert data["resources"] == ["subtitles"]
    assert "movie" in data["types"]
    assert "series" in data["types"]
    assert "anime" in data["types"]
    assert "tt" in data["idPrefixes"]
    assert "kitsu" in data["idPrefixes"]
    assert "logo" in data and data["logo"].endswith("/static/icon.png")
    assert "icon" in data and data["icon"].endswith("/static/icon.png")

    # Test HEAD request (Nuvio ping compatibility)
    head_resp = client.head("/manifest.json")
    assert head_resp.status_code == 200
    assert head_resp.headers.get("access-control-allow-origin") == "*"
    assert "application/json" in head_resp.headers.get("content-type", "")

    # Test OPTIONS request (pre-flight / probing compatibility)
    options_resp = client.options("/manifest.json")
    assert options_resp.status_code == 200
    assert options_resp.headers.get("access-control-allow-origin") == "*"


@pytest.mark.asyncio
async def test_subtitles_endpoint_schema_anime(client):
    """
    Assert /subtitles/anime/{id}.json returns valid Stremio subtitles schema.
    """
    mock_subdl_results = [
        SubtitleRelease(
            release_name="Attack.on.Titan.S01E05.1080p.BluRay.x264.srt",
            download_url="https://dl.subdl.com/subtitle/99999.zip",
            provider="subdl",
            format="srt",
            hearing_impaired=False,
            lang="ara",
        )
    ]
    with patch("app.main._http_client", new_callable=AsyncMock):
        with (
            patch(
                "app.providers.subdl.SubdlProvider.search_subtitles",
                new=AsyncMock(return_value=mock_subdl_results),
            ),
            patch(
                "app.providers.subsource.SubsourceProvider.search_subtitles",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "app.providers.opensubtitles.OpenSubtitlesProvider.search_subtitles",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "app.providers.cinemeta.CinemetaClient.get_metadata",
                new=AsyncMock(return_value={"title": "Attack on Titan", "year": 2013}),
            ),
        ):
            resp = client.get("/subtitles/anime/tt0409591:1:5.json")
            assert resp.status_code == 200
            assert resp.headers.get("access-control-allow-origin") == "*"

            data = resp.json()
            assert "subtitles" in data
            assert len(data["subtitles"]) >= 1
            assert data["subtitles"][0]["format"] == "srt"
            assert data["subtitles"][0]["lang"] == "ara"


@pytest.mark.asyncio
async def test_subtitles_endpoint_schema_movie(client):
    """
    Assert /subtitles/movie/{id}.json returns valid Stremio subtitles schema
    with strict release-name display in 'lang' and unmodified 'title'.
    """
    mock_subdl_results = [
        SubtitleRelease(
            release_name="The.Shawshank.Redemption.1994.1080p.BluRay.x264-ARABIC.srt",
            download_url="https://dl.subdl.com/subtitle/12345.zip",
            provider="subdl",
            format="srt",
            hearing_impaired=False,
            lang="ara",
        )
    ]
    mock_subsource_results = [
        SubtitleRelease(
            release_name="The.Shawshank.Redemption.1994.720p.WEB-DL-ARA.srt",
            download_url="https://api.subsource.net/api/v1/subtitles/67890/download",
            provider="subsource",
            format="srt",
            hearing_impaired=False,
            lang="ara",
        )
    ]

    # Initialize lifespan mock client if needed
    with patch("app.main._http_client", new_callable=AsyncMock):
        with (
            patch(
                "app.providers.subdl.SubdlProvider.search_subtitles",
                new=AsyncMock(return_value=mock_subdl_results),
            ),
            patch(
                "app.providers.subsource.SubsourceProvider.search_subtitles",
                new=AsyncMock(return_value=mock_subsource_results),
            ),
            patch(
                "app.providers.cinemeta.CinemetaClient.get_metadata",
                new=AsyncMock(return_value={"title": "The Shawshank Redemption", "year": 1994}),
            ),
        ):
            resp = client.get("/subtitles/movie/tt0111161.json")
            assert resp.status_code == 200
            assert resp.headers.get("access-control-allow-origin") == "*"

            data = resp.json()
            assert "subtitles" in data
            subs = data["subtitles"]
            assert len(subs) == 2
            # Validate each item strictly matches required Stremio/Nuvio contract
            for item in subs:
                assert "id" in item and len(item["id"]) > 0
                assert "url" in item and item["url"].endswith(".srt")
                assert item["lang"] == "ara"
                assert "[" in item["title"] and "%]" in item["title"]
                assert "[SubDL]" in item["title"] or "[SubSource]" in item["title"]
                assert item["format"] == "srt"


@pytest.mark.asyncio
async def test_subtitles_deduplication(client):
    """
    Assert duplicate release names across providers are deduplicated.
    """
    identical_release = "House.of.the.Dragon.S02E01.1080p.WEB-DL.srt"
    mock_subdl = [
        SubtitleRelease(
            release_name=identical_release,
            download_url="https://dl.subdl.com/sub1.zip",
            provider="subdl",
        )
    ]
    mock_subsource = [
        SubtitleRelease(
            release_name=identical_release,
            download_url="https://api.subsource.net/sub2.zip",
            provider="subsource",
        )
    ]

    with patch("app.main._http_client", new_callable=AsyncMock):
        with (
            patch(
                "app.providers.subdl.SubdlProvider.search_subtitles",
                new=AsyncMock(return_value=mock_subdl),
            ),
            patch(
                "app.providers.subsource.SubsourceProvider.search_subtitles",
                new=AsyncMock(return_value=mock_subsource),
            ),
            patch(
                "app.providers.cinemeta.CinemetaClient.get_metadata",
                new=AsyncMock(return_value=None),
            ),
        ):
            resp = client.get("/subtitles/series/tt15264370:2:1.json")
            assert resp.status_code == 200
            subs = resp.json()["subtitles"]
            assert "House.of.the.Dragon.S02E01.1080p.WEB-DL" in subs[0]["title"]
            assert "[SubDL]" in subs[0]["title"]


@pytest.mark.asyncio
async def test_subtitles_fallback_when_one_provider_fails(client):
    """
    Assert that if one provider raises an error or times out, the other provider's
    results are safely returned without failing the HTTP request.
    """
    mock_subsource_only = [
        SubtitleRelease(
            release_name="Breaking.Bad.S05E16.720p.HDTV.srt",
            download_url="https://api.subsource.net/sub_bb.zip",
            provider="subsource",
        )
    ]

    with patch("app.main._http_client", new_callable=AsyncMock):
        # Subdl raises a network timeout exception
        with (
            patch(
                "app.providers.subdl.SubdlProvider.search_subtitles",
                new=AsyncMock(side_effect=httpx.TimeoutException("Upstream timeout")),
            ),
            patch(
                "app.providers.subsource.SubsourceProvider.search_subtitles",
                new=AsyncMock(return_value=mock_subsource_only),
            ),
            patch(
                "app.providers.cinemeta.CinemetaClient.get_metadata",
                new=AsyncMock(return_value=None),
            ),
        ):
            resp = client.get("/subtitles/series/tt0903747:5:16.json")
            assert resp.status_code == 200
            subs = resp.json()["subtitles"]
            assert len(subs) == 1
            assert subs[0]["title"] == "[100%] [SubSource] Breaking.Bad.S05E16.720p.HDTV"


@pytest.mark.asyncio
async def test_serve_subtitle_endpoint(client):
    """
    Assert /sub/{sub_id}.srt serves content with UTF-8 encoding and caching headers.
    """
    sub_id = "test_sub_123"
    srt_text = "1\n00:00:01,000 --> 00:00:02,000\nاختبار الترجمة\n"
    await cache_manager.save_subtitle(sub_id, srt_text.encode("utf-8"))

    resp = client.get(f"/sub/{sub_id}.srt")
    assert resp.status_code == 200
    assert "application/x-subrip" in resp.headers.get("content-type", "")
    assert "public, max-age=86400" in resp.headers.get("cache-control", "")
    assert resp.text == srt_text

    # WebVTT support
    resp_vtt = client.get(f"/sub/{sub_id}.vtt")
    assert resp_vtt.status_code == 200
    assert "text/vtt" in resp_vtt.headers.get("content-type", "")
    assert "WEBVTT" in resp_vtt.text


@pytest.mark.asyncio
async def test_serve_ass_subtitle_endpoint(client):
    """
    With the default ``convert_ass_to_srt`` preference, .ass/.ssa files are
    converted to color-preserved SRT; disabling the toggle serves raw ASS.
    """
    from app.utils.config_parser import encode_user_config

    sub_id = "test_anime_sub_456"
    ass_text = (
        "[Script Info]\n"
        "Title: Arabic Anime Subtitle\n"
        "ScriptType: v4.00+\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour\n"
        "Style: Default,Arial,24,&H00FFFFFF\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:01.00,0:00:03.50,Default,,0,0,0,,{\\pos(192,200)\\c&H0000FF&}ترجمة أنمي احترافية\n"
    )
    await cache_manager.save_subtitle(sub_id, ass_text.encode("utf-8"))

    # 1. Default preference: converted to color-preserved SRT.
    resp = client.get(f"/sub/{sub_id}.ass")
    assert resp.status_code == 200
    assert "application/x-subrip" in resp.headers.get("content-type", "")
    assert "public, max-age=86400" in resp.headers.get("cache-control", "")
    assert 'filename="test_anime_sub_456.srt"' in resp.headers.get("content-disposition", "")
    assert "00:00:01,000 --> 00:00:03,500" in resp.text
    assert '<font color="#FF0000">ترجمة أنمي احترافية</font>' in resp.text
    assert "\\pos" not in resp.text and "Dialogue:" not in resp.text

    # 2. Explicitly disabled: serve the original ASS untouched.
    raw_cfg = encode_user_config(convert_ass_to_srt=False)
    resp_cfg = client.get(f"/{raw_cfg}/sub/{sub_id}.ass")
    assert resp_cfg.status_code == 200
    assert "text/x-ssa" in resp_cfg.headers.get("content-type", "")
    assert resp_cfg.text == ass_text

    # 3. Requesting an ASS sub via the .srt route also converts to SRT.
    resp_srt = client.get(f"/sub/{sub_id}.srt")
    assert resp_srt.status_code == 200
    assert "application/x-subrip" in resp_srt.headers.get("content-type", "")
    assert "[Script Info]" not in resp_srt.text


@pytest.mark.asyncio
async def test_serve_native_vtt_subtitle_endpoint(client):
    """
    Assert /sub/{sub_id}.vtt serves native WebVTT content with text/vtt; charset=utf-8
    and preserves it directly as pass-through without conversion.
    """
    sub_id = "test_native_vtt_789"
    vtt_text = "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\nترجمة أصلية بصيغة ويب في تي تي\n"
    await cache_manager.save_subtitle(sub_id, vtt_text.encode("utf-8"))

    # 1. Requesting via .vtt route
    resp = client.get(f"/sub/{sub_id}.vtt")
    assert resp.status_code == 200
    assert "text/vtt" in resp.headers.get("content-type", "")
    assert "charset=utf-8" in resp.headers.get("content-type", "").lower()
    assert 'filename="test_native_vtt_789.vtt"' in resp.headers.get("content-disposition", "")
    assert resp.text == vtt_text

    # 2. If a native VTT sub is requested with .srt extension, server detects WebVTT format and preserves it as text/vtt
    resp_srt = client.get(f"/sub/{sub_id}.srt")
    assert resp_srt.status_code == 200
    assert "text/vtt" in resp_srt.headers.get("content-type", "")
    assert resp_srt.text == vtt_text


@pytest.mark.asyncio
async def test_subtitles_endpoint_accurately_reports_format(client):
    """
    Assert /subtitles/{type}/{id}.json accurately reports format ('srt', 'ass', 'vtt') in stream items.
    """
    mock_subs = [
        SubtitleRelease(
            release_name="Show.S01E01.720p.srt",
            download_url="https://dl.subdl.com/sub1.zip",
            provider="subdl",
            format="srt",
        ),
        SubtitleRelease(
            release_name="Show.S01E01.1080p.ass",
            download_url="https://dl.subdl.com/sub2.zip",
            provider="subdl",
            format="ass",
        ),
        SubtitleRelease(
            release_name="Show.S01E01.WEB.vtt",
            download_url="https://dl.subdl.com/sub3.zip",
            provider="subdl",
            format="vtt",
        ),
    ]

    with patch("app.main._http_client", new_callable=AsyncMock):
        with (
            patch(
                "app.providers.subdl.SubdlProvider.search_subtitles",
                new=AsyncMock(return_value=mock_subs),
            ),
            patch(
                "app.providers.subsource.SubsourceProvider.search_subtitles",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "app.providers.cinemeta.CinemetaClient.get_metadata",
                new=AsyncMock(return_value=None),
            ),
        ):
            resp = client.get("/subtitles/series/tt0944947:1:1.json")
            assert resp.status_code == 200
            data = resp.json()
            subs = data["subtitles"]
            assert len(subs) == 3

            # Check that each format and URL extension matches the source format
            formats_found = {s["format"]: s["url"] for s in subs}
            assert "srt" in formats_found and formats_found["srt"].endswith(".srt")
            assert "ass" in formats_found and formats_found["ass"].endswith(".ass")
            assert "vtt" in formats_found and formats_found["vtt"].endswith(".vtt")


@pytest.mark.asyncio
async def test_subtitles_endpoint_with_extra_path(client):
    """
    Assert /subtitles/{type}/{id}/{extra}.json handles extra path routing properly.
    """
    mock_sub = [
        SubtitleRelease(
            release_name="Series.S01E01.720p.srt",
            download_url="https://dl.subdl.com/sub.zip",
            provider="subdl",
        )
    ]

    with patch("app.main._http_client", new_callable=AsyncMock):
        with (
            patch(
                "app.providers.subdl.SubdlProvider.search_subtitles",
                new=AsyncMock(return_value=mock_sub),
            ),
            patch(
                "app.providers.subsource.SubsourceProvider.search_subtitles",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "app.providers.cinemeta.CinemetaClient.get_metadata",
                new=AsyncMock(return_value=None),
            ),
        ):
            resp = client.get("/subtitles/series/tt0944947:1:1/extraParam=foo.json")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["subtitles"]) == 1
            assert data["subtitles"][0]["title"] == "[100%] [SubDL] Series.S01E01.720p"


def test_health_endpoint(client):
    """Verify health endpoint returns status and cache statistics."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "cache" in data
    assert "total_size_bytes" in data["cache"]
    assert "file_count" in data["cache"]


@pytest.mark.asyncio
async def test_lru_cache_manager_enforce_limits(temp_cache_dir):
    """Verify LRUCacheManager evicts oldest accessed files when exceeding max_files."""
    import time

    from app.cache import LRUCacheManager

    mgr = LRUCacheManager(cache_dir=str(temp_cache_dir), max_bytes=10000, max_files=3)

    # Save 4 files (exceeding max_files=3)
    for i in range(1, 5):
        await mgr.save_subtitle(f"sub_{i}", f"content {i}".encode())
        time.sleep(0.01)

    # Oldest (sub_1) should have been evicted
    assert await mgr.get_subtitle("sub_1") is None
    assert await mgr.get_subtitle("sub_2") is not None
    assert await mgr.get_subtitle("sub_3") is not None
    assert await mgr.get_subtitle("sub_4") is not None


def test_configured_manifest_endpoint(client):
    """Verify /{config}/manifest.json returns valid Manifest with strictly 'NinjaSubs' name."""
    from app.utils.config_parser import encode_user_config

    config = encode_user_config("subdl_test_key", "subsource_test_key")
    resp = client.get(f"/{config}/manifest.json")
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "*"
    data = resp.json()
    assert data["id"] == "org.ninjasubs.addon"
    assert data["name"] == "NinjaSubs"
    assert data["description"] == (
        "Smart, high-accuracy subtitle aggregator for Stremio featuring advanced Arabic subtitle optimization."
    )
    assert "logo" in data and data["logo"].endswith("/static/icon.png")
    assert "icon" in data and data["icon"].endswith("/static/icon.png")


def test_parameterized_dummy_token_manifest_endpoint(client):
    """Verify parameterized URLs like /{dummy_token}/manifest.json return HTTP 200 with strictly 'NinjaSubs'."""
    resp = client.get("/dummy_token_nuvio_compat/manifest.json")
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "*"
    data = resp.json()
    assert data["id"] == "org.ninjasubs.addon"
    assert data["name"] == "NinjaSubs"
    assert data["description"] == (
        "Smart, high-accuracy subtitle aggregator for Stremio featuring advanced Arabic subtitle optimization."
    )
    assert "logo" in data and data["logo"].endswith("/static/icon.png")
    assert "icon" in data and data["icon"].endswith("/static/icon.png")

    # Test HEAD request on parameterized route
    head_resp = client.head("/dummy_token_nuvio_compat/manifest.json")
    assert head_resp.status_code == 200
    assert head_resp.headers.get("access-control-allow-origin") == "*"

    # Test OPTIONS request on parameterized route
    options_resp = client.options("/dummy_token_nuvio_compat/manifest.json")
    assert options_resp.status_code == 200
    assert options_resp.headers.get("access-control-allow-origin") == "*"


@pytest.mark.asyncio
async def test_configure_page_rendering(client):
    """Verify /configure and /{config}/configure return responsive HTML page."""
    from app.utils.config_parser import encode_user_config

    # Direct configure page
    resp = client.get("/configure")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Subtitle Providers" in resp.text
    assert "SubDL" in resp.text
    assert "SubSource" in resp.text
    assert "Exclude HI (Hearing Impaired)" in resp.text
    assert "Install to Stremio" in resp.text
    assert "Ninja" in resp.text and "Subs" in resp.text
    assert "https://subdl.com/panel/api" in resp.text
    assert "https://subsource.net/dashboard/profile" in resp.text
    # Searchable Multi-Select Dropdown components
    assert "tom-select.css" in resp.text
    assert "tom-select.complete.min.js" in resp.text
    assert 'id="languageSelect"' in resp.text
    assert 'placeholder="Select languages you like"' in resp.text
    # Fresh page: no pre-selected language — the user chooses (empty by default).
    assert '<option value="ara" >Arabic (ara)</option>' in resp.text
    assert (
        '<option value="eng" >English (eng)</option>' in resp.text
        or '<option value="eng">English (eng)</option>' in resp.text
    )
    assert 'value="fas"' in resp.text  # Persian/Farsi
    assert 'value="ind"' in resp.text  # Indonesian

    assert "Subtitle Type" not in resp.text
    assert 'name="sub_type"' not in resp.text

    # Prefilled configure page
    cfg = encode_user_config("my_subdl", "my_subsource", languages=["eng", "fas"], exclude_hi=True)
    resp_prefill = client.get(f"/{cfg}/configure")
    assert resp_prefill.status_code == 200
    assert "my_subdl" in resp_prefill.text
    assert "my_subsource" in resp_prefill.text
    assert "checked" in resp_prefill.text
    assert '<option value="eng" selected>English (eng)</option>' in resp_prefill.text
    assert '<option value="fas" selected>Persian (Farsi) (fas)</option>' in resp_prefill.text
    assert (
        '<option value="ara" ' in resp_prefill.text
        and "selected" not in resp_prefill.text.split('<option value="ara" ')[1].split(">")[0]
    )


async def test_configured_subtitles_endpoint(client):
    """Verify /{config}/subtitles/... parses user keys and injects them into providers."""
    from app.utils.config_parser import encode_user_config

    user_subdl_key = "user_subdl_abc123"
    user_subsource_key = "user_subsource_def456"
    config = encode_user_config(user_subdl_key, user_subsource_key)

    mock_subdl_call = AsyncMock(
        return_value=[
            SubtitleRelease(
                release_name="Configured.Movie.1080p.srt",
                download_url="https://dl.subdl.com/sub.zip",
                provider="subdl",
            )
        ]
    )
    mock_subsource_call = AsyncMock(return_value=[])

    with patch("app.main._http_client", new_callable=AsyncMock):
        with (
            patch("app.providers.subdl.SubdlProvider.search_subtitles", new=mock_subdl_call),
            patch(
                "app.providers.subsource.SubsourceProvider.search_subtitles",
                new=mock_subsource_call,
            ),
            patch(
                "app.providers.cinemeta.CinemetaClient.get_metadata",
                new=AsyncMock(return_value=None),
            ),
        ):
            resp = client.get(f"/{config}/subtitles/movie/tt0111161.json")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["subtitles"]) == 1
            # Check subtitle URL maintains config prefix
            assert f"/{config}/sub/" in data["subtitles"][0]["url"]

            # Assert Subdl was called with user's personal key
            mock_subdl_call.assert_called_once()
            call_kwargs = mock_subdl_call.call_args.kwargs
            assert call_kwargs.get("api_key") == user_subdl_key

            # Assert Subsource was called with user's personal key
            mock_subsource_call.assert_called_once()
            subsource_kwargs = mock_subsource_call.call_args.kwargs
            assert subsource_kwargs.get("api_key") == user_subsource_key


@pytest.mark.asyncio
async def test_missing_keys_graceful_fallback(client):
    """Verify when no API keys are provided (neither env nor config), providers log clean warning and return empty."""
    from app.config import settings

    with patch.object(settings, "SUBDL_API_KEY", ""):
        with patch.object(settings, "SUBSOURCE_API_KEY", ""):
            with patch("app.main._http_client", new_callable=AsyncMock):
                with patch(
                    "app.providers.cinemeta.CinemetaClient.get_metadata",
                    new=AsyncMock(return_value=None),
                ):
                    resp = client.get("/subtitles/movie/tt0111161.json")
                    assert resp.status_code == 200
                    data = resp.json()
                    # Handled gracefully without crash
                    assert data["subtitles"] == []


def test_extensionless_manifest_routes(client):
    """Verify /manifest and /{config}/manifest work identically to .json routes."""
    from app.utils.config_parser import encode_user_config

    resp1 = client.get("/manifest")
    assert resp1.status_code == 200
    assert resp1.json()["id"] == "org.ninjasubs.addon"

    cfg = encode_user_config("subdl_123", "subsource_456")
    resp2 = client.get(f"/{cfg}/manifest")
    assert resp2.status_code == 200
    assert resp2.json()["id"] == "org.ninjasubs.addon"


@pytest.mark.asyncio
async def test_extensionless_subtitles_routes(client):
    """Verify /subtitles/{type}/{id} without .json works for Nuvio."""
    mock_subs = [
        SubtitleRelease(
            release_name="Extensionless.Test.1080p.srt",
            download_url="https://dl.subdl.com/sub.zip",
            provider="subdl",
        )
    ]
    with patch("app.main._http_client", new_callable=AsyncMock):
        with (
            patch(
                "app.providers.subdl.SubdlProvider.search_subtitles",
                new=AsyncMock(return_value=mock_subs),
            ),
            patch(
                "app.providers.subsource.SubsourceProvider.search_subtitles",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "app.providers.cinemeta.CinemetaClient.get_metadata",
                new=AsyncMock(return_value=None),
            ),
        ):
            # Direct without .json
            resp = client.get("/subtitles/movie/tt0111161")
            assert resp.status_code == 200
            assert len(resp.json()["subtitles"]) == 1

            # Configured without .json
            from app.utils.config_parser import encode_user_config

            cfg = encode_user_config("dl", "source")
            resp_cfg = client.get(f"/{cfg}/subtitles/movie/tt0111161")
            assert resp_cfg.status_code == 200
            assert len(resp_cfg.json()["subtitles"]) == 1


@pytest.mark.asyncio
async def test_standardized_subtitle_response(client):
    """Verify modern standardized subtitle response: clean ISO lang and formatted clean title."""
    from app.utils.config_parser import encode_user_config

    mock_subs = [
        SubtitleRelease(
            release_name="Gladiator.2000.1080p.BluRay.srt",
            download_url="https://dl.subdl.com/sub.zip",
            provider="subdl",
        )
    ]

    with patch("app.main._http_client", new_callable=AsyncMock):
        with (
            patch(
                "app.providers.subdl.SubdlProvider.search_subtitles",
                new=AsyncMock(return_value=mock_subs),
            ),
            patch(
                "app.providers.subsource.SubsourceProvider.search_subtitles",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "app.providers.cinemeta.CinemetaClient.get_metadata",
                new=AsyncMock(return_value=None),
            ),
        ):
            cfg = encode_user_config("dl", "src")
            resp = client.get(f"/{cfg}/subtitles/movie/tt0172495.json")
            sub = resp.json()["subtitles"][0]
            assert sub["lang"] == "ara"
            assert "id" in sub and len(sub["id"]) > 0
            assert sub["title"] == "[55%] [SubDL] Gladiator.2000.1080p.BluRay"

            # Direct server-wide route
            resp_direct = client.get("/subtitles/movie/tt0172495.json")
            sub_direct = resp_direct.json()["subtitles"][0]
            assert sub_direct["lang"] == "ara"
            assert sub_direct["title"] == "[55%] [SubDL] Gladiator.2000.1080p.BluRay"


@pytest.mark.asyncio
async def test_release_matching_scoring_and_ranking_with_extra(client):
    """
    Verify intelligent release-matching scores candidates against target stream filename:
    - High-scoring subtitle (score 100%) is at index 0.
    - Format: '[{score}%] [{source}] {cleaned_release_name}' in title and lang.
    - Results are sorted descending by score.
    """
    mock_subs = [
        SubtitleRelease(
            release_name="Dune.Part.Two.2024.720p.HDTV.x264-AVS.srt",
            download_url="https://dl.subdl.com/hdtv.zip",
            provider="subdl",
        ),
        SubtitleRelease(
            release_name="Dune.Part.Two.2024.2160p.UHD.BluRay.x265-FLUX.srt_86f1f22e8f1fd5bd",
            download_url="https://dl.subdl.com/flux.zip",
            provider="subdl",
        ),
        SubtitleRelease(
            release_name="Dune.Part.Two.2024.1080p.WEB-DL.x264-NTb.srt",
            download_url="https://dl.subdl.com/webdl.zip",
            provider="subsource",
        ),
    ]

    with patch("app.main._http_client", new_callable=AsyncMock):
        with (
            patch(
                "app.providers.subdl.SubdlProvider.search_subtitles",
                new=AsyncMock(return_value=mock_subs[:2]),
            ),
            patch(
                "app.providers.subsource.SubsourceProvider.search_subtitles",
                new=AsyncMock(return_value=mock_subs[2:]),
            ),
            patch(
                "app.providers.cinemeta.CinemetaClient.get_metadata",
                new=AsyncMock(return_value=None),
            ),
        ):
            # Stream filename matches FLUX 2160p BluRay
            extra = "filename=Dune.Part.Two.2024.2160p.UHD.BluRay.x265-FLUX.mkv"
            resp = client.get(f"/subtitles/movie/tt15239678/{extra}.json")
            assert resp.status_code == 200

            subs = resp.json()["subtitles"]
            assert len(subs) == 3

            # FLUX 2160p BluRay should be ranked first at index 0 with score = 100%
            best_match = subs[0]
            assert best_match["title"] == "[100%] [SubDL] Dune.Part.Two.2024.2160p.UHD.BluRay.x265-FLUX"
            assert best_match["lang"] == "ara"

            # Subsource WEB-DL item
            assert subs[1]["title"] == "[0%] [SubSource] Dune.Part.Two.2024.1080p.WEB-DL.x264-NTb"

            # HDTV with penalty should be ranked last
            assert subs[2]["title"] == "[0%] [SubDL] Dune.Part.Two.2024.720p.HDTV.x264-AVS"


@pytest.mark.asyncio
async def test_multi_language_sorting_and_deduplication(client):
    """
    Verify multi-language handling:
    1. Does not deduplicate identical release names across DIFFERENT languages.
    2. Primary language (e.g. Arabic) top percentage match is at index 0.
    3. Secondary language (e.g. English) top percentage match heads the secondary category.
    4. Stremio hybrid tags use proper prefixes ('AR | ...', 'EN | ...').
    5. Nuvio mode uses clean ISO codes ('ara', 'eng').
    """
    from app.utils.config_parser import encode_user_config

    mock_subs = [
        # Arabic releases
        SubtitleRelease(
            release_name="Oppenheimer.2023.720p.HDTV.srt",
            download_url="https://dl.subdl.com/opp_ar_low.zip",
            provider="subdl",
            lang="ara",
        ),
        SubtitleRelease(
            release_name="Oppenheimer.2023.1080p.BluRay.x264-FLUX.srt",
            download_url="https://dl.subdl.com/opp_ar_high.zip",
            provider="subdl",
            lang="ara",
        ),
        # English release with identical name as the high-scoring Arabic release
        SubtitleRelease(
            release_name="Oppenheimer.2023.1080p.BluRay.x264-FLUX.srt",
            download_url="https://api.subsource.net/opp_en_high.zip",
            provider="subsource",
            lang="eng",
        ),
        SubtitleRelease(
            release_name="Oppenheimer.2023.720p.HDTV.srt",
            download_url="https://api.subsource.net/opp_en_low.zip",
            provider="subsource",
            lang="eng",
        ),
    ]

    cfg = encode_user_config(languages=["ara", "eng"])

    with patch("app.main._http_client", new_callable=AsyncMock):
        with (
            patch(
                "app.providers.subdl.SubdlProvider.search_subtitles",
                new=AsyncMock(return_value=mock_subs[:2]),
            ),
            patch(
                "app.providers.subsource.SubsourceProvider.search_subtitles",
                new=AsyncMock(return_value=mock_subs[2:]),
            ),
            patch(
                "app.providers.cinemeta.CinemetaClient.get_metadata",
                new=AsyncMock(return_value=None),
            ),
        ):
            extra = "filename=Oppenheimer.2023.1080p.BluRay.x264-FLUX.mkv"
            resp = client.get(f"/{cfg}/subtitles/movie/tt15398776/{extra}.json")
            assert resp.status_code == 200
            subs = resp.json()["subtitles"]

            # Both Arabic and English 1080p BluRay FLUX must exist (not deduplicated across languages)
            assert len(subs) == 4

            # Index 0: Top match for Arabic (primary language)
            assert subs[0]["lang"] == "ara"
            assert subs[0]["title"] == "[100%] [SubDL] Oppenheimer.2023.1080p.BluRay.x264-FLUX"

            # Index 1: Lower match for Arabic
            assert subs[1]["lang"] == "ara"

            # Index 2: Top match for English (secondary category head)
            assert subs[2]["lang"] == "eng"
            assert subs[2]["title"] == "[100%] [SubSource] Oppenheimer.2023.1080p.BluRay.x264-FLUX"

            # Index 3: Lower match for English
            assert subs[3]["lang"] == "eng"


def test_get_local_lan_ip_and_base_url():
    """Verify get_local_lan_ip returns a valid IP and get_base_url dynamically matches request host."""
    from unittest.mock import MagicMock, patch

    from starlette.datastructures import URL

    from app.main import get_base_url, get_local_lan_ip

    with patch("app.main.settings.BASE_URL", None):
        # 1. get_local_lan_ip returns non-empty IPv4
        lan_ip = get_local_lan_ip()
        assert isinstance(lan_ip, str)
        assert len(lan_ip.split(".")) == 4

        # 2. Localhost and 127.0.0.1 requests preserve client's exact host dynamically
        req_localhost = MagicMock()
        req_localhost.url = URL("http://localhost:7000/manifest.json")
        req_localhost.base_url = URL("http://localhost:7000/")
        assert get_base_url(req_localhost) == "http://localhost:7000"

        req_127 = MagicMock()
        req_127.url = URL("http://127.0.0.1:7000/manifest.json")
        req_127.base_url = URL("http://127.0.0.1:7000/")
        assert get_base_url(req_127) == "http://127.0.0.1:7000"

        # 3. Explicit LAN request preserves the client's host
        req_lan = MagicMock()
        req_lan.url = URL("http://192.168.1.99:7000/manifest.json")
        req_lan.base_url = URL("http://192.168.1.99:7000/")
        assert get_base_url(req_lan) == "http://192.168.1.99:7000"

        # 4. Fallback without request resolves to LAN IP or 127.0.0.1
        resolved_none = get_base_url(None)
        assert lan_ip in resolved_none or "127.0.0.1" in resolved_none

        # 5. HOST_IP and ADDON_BASE_URL overrides
        with patch.dict("os.environ", {"HOST_IP": "10.0.0.88"}):
            assert get_local_lan_ip() == "10.0.0.88"
        with patch.dict(
            "os.environ", {"HOST_IP": "", "LAN_IP": "", "ADDON_BASE_URL": "http://10.0.0.99:7000"}
        ):
            assert get_local_lan_ip() == "10.0.0.99"


def test_is_local_or_container_host():
    """Loopback and Docker bridge hosts must be flagged; real LAN/domain hosts must not."""
    from app.utils.network import is_local_or_container_host

    for host in ("localhost", "127.0.0.1", "127.0.1.1", "::1", "0.0.0.0", "", "172.17.0.1", "172.18.5.4", "172.31.255.255"):
        assert is_local_or_container_host(host) is True, host
    for host in ("192.168.8.114", "10.0.0.88", "172.15.0.1", "172.32.0.1", "example.com", "subs.mydomain.net"):
        assert is_local_or_container_host(host) is False, host


def test_configure_page_embeds_lan_resolution_logic(client):
    """The configure page must swap loopback/Docker hosts for the detected LAN IP + port."""
    body = client.get("/configure").text
    assert "isLocalOrContainerHost" in body
    assert "buildLanBaseUrl" in body
    # Docker 172.16-31 range detection present in the client script.
    assert "172" in body and "serverLanIp" in body
    assert "serverPort" in body


@pytest.mark.asyncio
async def test_subdl_provider_requests_max_subs_per_page():
    """Verify SubdlProvider explicitly requests subs_per_page=30 to retrieve maximum available subtitles."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {
        "status": True,
        "subtitles": [
            {
                "release_name": "Test.Movie.2024.1080p.srt",
                "url": "/subtitle/test.zip",
                "lang": "Arabic",
            }
        ],
    }
    mock_client.get.return_value = resp

    provider = SubdlProvider(mock_client)
    subs = await provider.search_subtitles(
        imdb_id="tt1234567",
        api_key="test_subdl_key",
    )

    assert len(subs) == 1
    assert mock_client.get.call_count == 1
    call_args, call_kwargs = mock_client.get.call_args
    assert call_kwargs["params"]["subs_per_page"] == 30
    assert call_kwargs["params"]["imdb_id"] == "tt1234567"


@pytest.mark.asyncio
async def test_subtitles_endpoint_returns_all_uncapped_results_sorted(client):
    """Verify subtitle route does not slice or cap results (e.g. at 10) and maintains descending score sorting."""
    # Generate 40 distinct releases with different simulated compatibility scores
    mock_releases = [
        SubtitleRelease(
            release_name=f"Movie.2024.Release.{i:02d}.1080p.BluRay.srt",
            download_url=f"https://dl.subdl.com/sub_{i}.zip",
            provider="subdl" if i % 2 == 0 else "subsource",
            lang="ara",
        )
        for i in range(40)
    ]

    with patch("app.main._http_client", new_callable=AsyncMock):
        with (
            patch(
                "app.providers.subdl.SubdlProvider.search_subtitles",
                new=AsyncMock(return_value=mock_releases[:20]),
            ),
            patch(
                "app.providers.subsource.SubsourceProvider.search_subtitles",
                new=AsyncMock(return_value=mock_releases[20:]),
            ),
            patch(
                "app.providers.cinemeta.CinemetaClient.get_metadata",
                new=AsyncMock(return_value=None),
            ),
        ):
            resp = client.get("/subtitles/movie/tt9999999.json")
            assert resp.status_code == 200
            subs = resp.json()["subtitles"]

            # All 40 items must be returned without any artificial slice or truncation
            assert len(subs) == 40

            # Extract percentage numbers from titles (e.g. "[85%] [Subdl] ...")
            import re

            scores = [int(re.search(r"\[(\d+)%\]", s["title"]).group(1)) for s in subs]
            # Ensure sorted descending
            assert scores == sorted(scores, reverse=True)
            assert scores[0] >= scores[-1]


@pytest.mark.asyncio
async def test_subdl_multi_page_pagination():
    """Verify SubdlProvider loops across multiple pages when totalPages > 1."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    def side_effect(url, params=None, headers=None, timeout=None):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        page = params.get("page", 1)
        if page == 1:
            resp.json.return_value = {
                "status": True,
                "totalPages": 2,
                "subtitles": [
                    {
                        "release_name": f"Movie.P1.Sub{i}.srt",
                        "url": f"/sub1_{i}.zip",
                        "lang": "Arabic",
                    }
                    for i in range(30)
                ],
            }
        elif page == 2:
            resp.json.return_value = {
                "status": True,
                "totalPages": 2,
                "subtitles": [
                    {
                        "release_name": f"Movie.P2.Sub{i}.srt",
                        "url": f"/sub2_{i}.zip",
                        "lang": "Arabic",
                    }
                    for i in range(15)
                ],
            }
        else:
            resp.json.return_value = {"status": True, "subtitles": []}
        return resp

    mock_client.get.side_effect = side_effect
    provider = SubdlProvider(mock_client)
    subs = await provider.search_subtitles(imdb_id="tt0111161", api_key="test_key")

    assert len(subs) == 45
    assert mock_client.get.call_count == 2


@pytest.mark.asyncio
async def test_subdl_filtering_loop_exclude_hi(caplog):
    """Verify SubdlProvider filters items by exclude_hi with log verification."""
    import logging

    caplog.set_level(logging.INFO)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {
        "status": True,
        "subtitles": [
            {
                "release_name": "Movie.1080p.Retail.BluRay.x264.srt",
                "url": "/sub_retail.zip",
                "lang": "Arabic",
                "hi": 0,
            },
            {
                "release_name": "Movie.1080p.ChatGPT.srt",
                "url": "/sub_machine.zip",
                "lang": "Arabic",
                "hi": 0,
            },
            {
                "release_name": "Movie.1080p.Forced.srt",
                "url": "/sub_forced.zip",
                "lang": "Arabic",
                "forced": True,
                "hi": 0,
            },
            {
                "release_name": "Movie.HDTV.FanSub.HI.srt",
                "url": "/sub_trans_hi.zip",
                "lang": "Arabic",
                "hi": 1,
            },
            {
                "release_name": "Movie.HDTV.FanSub.srt",
                "url": "/sub_trans_clean.zip",
                "lang": "Arabic",
                "hi": 0,
            },
        ],
    }
    mock_client.get.return_value = resp

    provider = SubdlProvider(mock_client)

    # User sets exclude_hi=True
    subs = await provider.search_subtitles(
        imdb_id="tt0111161",
        api_key="key123",
        languages=["ara"],
        exclude_hi=True,
    )

    assert len(subs) == 4
    names = [s.release_name for s in subs]
    assert "Movie.HDTV.FanSub.HI.srt" not in names
    assert "Movie.1080p.Retail.BluRay.x264.srt" in names
    assert "Movie.HDTV.FanSub.srt" in names

    # Verify log output for live reporting of dropped counts
    log_records = [r.message for r in caplog.records if "[SubDL Filter Verification]" in r.message]
    assert len(log_records) == 1
    log_msg = log_records[0]
    assert "Total: 5" in log_msg
    assert "Dropped (HI): 1" in log_msg
    assert "Final Passed: 4" in log_msg
