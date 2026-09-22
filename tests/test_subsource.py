"""Unit tests for SubSource official REST API v1 provider implementation."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.providers.subsource import SubsourceProvider, SubSourceService


@pytest.mark.asyncio
async def test_subsource_service_direct():
    """Verify SubSourceService resolves movieId via /movies/search and queries /subtitles?movieId=..."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    def side_effect(url, params=None, headers=None, timeout=None):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        if "movies/search" in url:
            resp.json.return_value = {
                "status": True,
                "data": [
                    {
                        "movieId": 9999,
                        "title": "Breaking Bad",
                    }
                ],
            }
        else:
            resp.json.return_value = {
                "status": True,
                "data": [
                    {
                        "id": "sub_ar_1",
                        "release_name": "Breaking.Bad.S05E16.1080p.BluRay",
                        "season": 5,
                        "episode": 16,
                        "language": "Arabic",
                    },
                    {
                        "id": "sub_ar_wrong_ep",
                        "release_name": "Breaking.Bad.S05E15.1080p.BluRay",
                        "season": 5,
                        "episode": 15,
                        "language": "Arabic",
                    },
                    {
                        "id": "sub_en_1",
                        "release_name": "Breaking.Bad.S05E16.1080p.BluRay",
                        "season": 5,
                        "episode": 16,
                        "language": "English",
                    },
                ],
            }
        return resp

    mock_client.get.side_effect = side_effect

    service = SubSourceService("test_key_123")
    subs = await service.get_subtitles(
        client=mock_client,
        media_type="series",
        imdb_id="tt0903747",
        season=5,
        episode=16,
        language="Arabic",
    )

    assert len(subs) == 1
    assert subs[0]["id"] == "sub_ar_1"
    assert subs[0]["raw_name"] == "Breaking.Bad.S05E16.1080p.BluRay"
    assert subs[0]["lang"] == "ara"

    # Verify calls: 1 to movies/search, 2 to subtitles (page 1 + page 2 duplicate check)
    assert mock_client.get.call_count == 3
    call1_args, call1_kwargs = mock_client.get.call_args_list[0]
    assert call1_args[0].endswith("/movies/search")
    assert "tt0903747" in str(call1_kwargs["params"])

    call2_args, call2_kwargs = mock_client.get.call_args_list[1]
    assert call2_args[0].endswith("/subtitles")
    assert call2_kwargs["params"]["movieId"] == 9999
    assert call2_kwargs["params"]["season"] == 5
    assert call2_kwargs["params"]["episode"] == 16
    assert call2_kwargs["params"]["limit"] == 100
    assert call2_kwargs["params"]["perPage"] == 100
    assert call2_kwargs["params"]["take"] == 100
    assert call2_kwargs["params"]["page"] == 1


@pytest.mark.asyncio
async def test_subsource_search_movie_v1():
    """Verify SubSource provider queries /movies/search then /subtitles with movieId."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    def side_effect(url, params=None, headers=None, timeout=None):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        if "movies/search" in url:
            resp.json.return_value = {
                "status": True,
                "data": [{"movieId": 12345, "title": "The Shawshank Redemption"}],
            }
        else:
            resp.json.return_value = {
                "status": True,
                "data": [
                    {
                        "subtitleId": "sub_101",
                        "releaseInfo": ["The.Shawshank.Redemption.1994.1080p.BluRay.x264.srt"],
                        "hearingImpaired": False,
                        "language": "Arabic",
                    }
                ],
            }
        return resp

    mock_client.get.side_effect = side_effect

    provider = SubsourceProvider(mock_client)
    subs = await provider.search_subtitles(
        imdb_id="tt0111161",
        api_key="test_api_key_xyz",
        languages=["ara"],
    )

    assert len(subs) == 1
    assert subs[0].release_name == "The.Shawshank.Redemption.1994.1080p.BluRay.x264.srt"
    assert subs[0].provider == "subsource"
    assert subs[0].lang == "ara"
    assert subs[0].download_url == "https://api.subsource.net/api/v1/subtitles/sub_101/download"

    assert mock_client.get.call_count == 2
    call1_args, call1_kwargs = mock_client.get.call_args_list[0]
    assert call1_args[0].endswith("/movies/search")

    call2_args, call2_kwargs = mock_client.get.call_args_list[1]
    assert call2_args[0].endswith("/subtitles")
    assert call2_kwargs["params"]["movieId"] == 12345
    assert call2_kwargs["params"]["limit"] == 100
    assert call2_kwargs["params"]["page"] == 1
    assert call2_kwargs["headers"]["X-API-Key"] == "test_api_key_xyz"


@pytest.mark.asyncio
async def test_subsource_uploader_from_contributors_displayname():
    """SubSource exposes the uploader via contributors[].displayname (not uploaderId)."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    def side_effect(url, params=None, headers=None, timeout=None):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        if "movies/search" in url:
            resp.json.return_value = {
                "status": True,
                "data": [{"movieId": 777, "title": "Movie"}],
            }
        else:
            resp.json.return_value = {
                "status": True,
                "data": [
                    {
                        "subtitleId": "sub_with_uploader",
                        "releaseInfo": ["Movie.2024.1080p.BluRay.srt"],
                        "language": "Arabic",
                        "hearingImpaired": False,
                        "uploaderId": 4242,
                        "contributors": [{"id": 4242, "displayname": "subsmaster"}],
                    },
                    {
                        "subtitleId": "sub_id_only",
                        "releaseInfo": ["Movie.2024.720p.WEB-DL.srt"],
                        "language": "Arabic",
                        "hearingImpaired": False,
                        "uploaderId": 9999,
                    },
                ],
            }
        return resp

    mock_client.get.side_effect = side_effect

    provider = SubsourceProvider(mock_client)
    subs = await provider.search_subtitles(
        imdb_id="tt0111161",
        api_key="test_api_key_xyz",
        languages=["ara"],
    )

    assert len(subs) == 2
    with_uploader = next(s for s in subs if "sub_with_uploader" in s.download_url)
    id_only = next(s for s in subs if "sub_id_only" in s.download_url)

    assert with_uploader.uploader == "subsmaster"
    # Numeric uploaderId alone is not a username.
    assert id_only.uploader == ""


@pytest.mark.asyncio
async def test_subsource_search_series_v1_filters_episodes():
    """Verify series query fetches subtitles via movieId and filters episode locally."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    def side_effect(url, params=None, headers=None, timeout=None):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        if "movies/search" in url:
            resp.json.return_value = {"movieId": 8888, "title": "Breaking Bad"}
        else:
            resp.json.return_value = {
                "status": True,
                "subtitles": [
                    {
                        "subtitleId": "sub_s05e16",
                        "releaseInfo": ["Breaking.Bad.S05E16.720p.HDTV.x264"],
                        "season": 5,
                        "episode": 16,
                        "hearingImpaired": False,
                        "language": "Arabic",
                    },
                    {
                        "subtitleId": "sub_s05e15",
                        "releaseInfo": ["Breaking.Bad.S05E15.720p.HDTV.x264"],
                        "season": 5,
                        "episode": 15,
                        "hearingImpaired": False,
                        "language": "Arabic",
                    },
                ],
            }
        return resp

    mock_client.get.side_effect = side_effect

    provider = SubsourceProvider(mock_client)
    subs = await provider.search_subtitles(
        imdb_id="tt0903747",
        is_series=True,
        season=5,
        episode=16,
        api_key="key_bb",
        languages=["ara"],
    )

    # Episode 15 should have been filtered out locally
    assert len(subs) == 1
    assert "S05E16" in subs[0].release_name
    assert subs[0].download_url == "https://api.subsource.net/api/v1/subtitles/sub_s05e16/download"

    assert mock_client.get.call_count == 2
    call2_args, call2_kwargs = mock_client.get.call_args_list[1]
    assert call2_args[0].endswith("/subtitles")
    assert call2_kwargs["params"]["movieId"] == 8888
    assert call2_kwargs["params"]["season"] == 5
    assert call2_kwargs["params"]["episode"] == 16
    assert call2_kwargs["params"]["limit"] == 100
    assert call2_kwargs["params"]["page"] == 1


@pytest.mark.asyncio
async def test_subsource_release_info_fallback():
    """Verify fallback to releaseInfo when movieId cannot be resolved."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    def side_effect(url, params=None, headers=None, timeout=None):
        resp = MagicMock(spec=httpx.Response)
        if "movies/search" in url:
            resp.status_code = 404
            resp.json.return_value = {"error": "Not Found"}
        else:
            resp.status_code = 200
            resp.json.return_value = {
                "data": [
                    {
                        "id": "sub_rel_1",
                        "release_name": "Dune.Part.Two.2024.2160p.UHD.BluRay.srt",
                        "language": "Arabic",
                    }
                ]
            }
        return resp

    mock_client.get.side_effect = side_effect

    provider = SubsourceProvider(mock_client)
    subs = await provider.search_subtitles(
        imdb_id="tt15239678",
        api_key="test_key",
        languages=["ara"],
        target_filename="Dune.Part.Two.2024.2160p.UHD.BluRay.x265-FLUX.mkv",
    )

    assert len(subs) == 1
    assert "Dune.Part.Two" in subs[0].release_name
    last_call_args, last_call_kwargs = mock_client.get.call_args_list[-1]
    assert last_call_args[0].endswith("/subtitles")
    assert (
        last_call_kwargs["params"]["releaseInfo"]
        == "Dune.Part.Two.2024.2160p.UHD.BluRay.x265-FLUX.mkv"
    )
    assert last_call_kwargs["params"]["limit"] == 100
    assert last_call_kwargs["params"]["page"] == 1


@pytest.mark.asyncio
async def test_subsource_series_id_parsing():
    """Verify compound series ID like tt0903747:2:4 is parsed into components."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    def side_effect(url, params=None, headers=None, timeout=None):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        if "movies/search" in url:
            resp.json.return_value = {"movieId": 4444}
        else:
            resp.json.return_value = [
                {
                    "id": "sub_44",
                    "release_name": "Show.S02E04.720p.srt",
                    "season": 2,
                    "episode": 4,
                    "language": "arabic",
                }
            ]
        return resp

    mock_client.get.side_effect = side_effect

    provider = SubsourceProvider(mock_client)
    subs = await provider.search_subtitles(
        imdb_id="tt0903747:2:4",
        api_key="key_bb",
        languages=["ara"],
    )

    assert len(subs) == 1
    call2_args, call2_kwargs = mock_client.get.call_args_list[1]
    assert call2_kwargs["params"]["movieId"] == 4444
    assert call2_kwargs["params"]["season"] == 2
    assert call2_kwargs["params"]["episode"] == 4
    assert call2_kwargs["params"]["limit"] == 100
    assert call2_kwargs["params"]["page"] == 1


@pytest.mark.asyncio
async def test_subsource_multi_language_search():
    """Verify multiple languages query SubSource and filter locally by language."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    def side_effect(url, params=None, headers=None, timeout=None):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        if "movies/search" in url:
            resp.json.return_value = {"movieId": 7777}
        else:
            resp.json.return_value = {
                "data": [
                    {
                        "subtitleId": "sub_ar",
                        "releaseInfo": ["Gladiator.2000.1080p.BluRay.x264.srt"],
                        "language": "Arabic",
                    },
                    {
                        "subtitleId": "sub_en",
                        "releaseInfo": ["Gladiator.2000.1080p.BluRay.x264.srt"],
                        "language": "English",
                    },
                ]
            }
        return resp

    mock_client.get.side_effect = side_effect

    provider = SubsourceProvider(mock_client)
    subs = await provider.search_subtitles(
        imdb_id="tt0172495",
        api_key="test_key",
        languages=["ara", "eng"],
    )

    assert len(subs) == 2
    langs = {s.lang for s in subs}
    assert langs == {"ara", "eng"}


@pytest.mark.asyncio
async def test_subsource_missing_api_key_returns_empty():
    """Verify when no API key is set, returns empty list without making HTTP calls."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    provider = SubsourceProvider(mock_client)

    with patch("app.config.settings.SUBSOURCE_API_KEY", ""):
        subs = await provider.search_subtitles(
            imdb_id="tt0111161",
            api_key=None,
        )
        assert subs == []
        mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_subsource_resilient_error_handling():
    """Verify HTTP 401, 403, 429, and timeouts fail gracefully returning empty list."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    provider = SubsourceProvider(mock_client)

    # 1. HTTP 401 Unauthorized
    resp_401 = MagicMock(spec=httpx.Response)
    resp_401.status_code = 401
    mock_client.get.return_value = resp_401
    subs = await provider.search_subtitles(imdb_id="tt123", api_key="bad_key")
    assert subs == []

    # 2. HTTP 429 Rate Limit
    resp_429 = MagicMock(spec=httpx.Response)
    resp_429.status_code = 429
    mock_client.get.return_value = resp_429
    subs = await provider.search_subtitles(imdb_id="tt123", api_key="key")
    assert subs == []

    # 3. Timeout Exception
    mock_client.get.side_effect = httpx.TimeoutException("Timeout")
    subs = await provider.search_subtitles(imdb_id="tt123", api_key="key")
    assert subs == []


@pytest.mark.asyncio
async def test_subsource_download_archive():
    """Verify download_archive uses X-API-Key and handles responses safely."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    provider = SubsourceProvider(mock_client)

    # Successful download
    resp_200 = MagicMock(spec=httpx.Response)
    resp_200.status_code = 200
    resp_200.content = b"1\n00:00:01,000 --> 00:00:02,000\nHello\n"
    mock_client.get.side_effect = None
    mock_client.get.return_value = resp_200

    content = await provider.download_archive("12345", api_key="my_secret_key")
    assert content == b"1\n00:00:01,000 --> 00:00:02,000\nHello\n"

    args, kwargs = mock_client.get.call_args
    assert args[0] == "https://api.subsource.net/api/v1/subtitles/12345/download"
    assert kwargs["headers"]["X-API-Key"] == "my_secret_key"

    # Failed download 403
    resp_403 = MagicMock(spec=httpx.Response)
    resp_403.status_code = 403
    mock_client.get.return_value = resp_403
    assert await provider.download_archive("12345", api_key="bad_key") is None


@pytest.mark.asyncio
async def test_subsource_multi_page_pagination():
    """Verify SubSource multi-page loop retrieves all pages up to max_pages or batch completion."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    def side_effect(url, params=None, headers=None, timeout=None):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        if "movies/search" in url:
            resp.json.return_value = {"movieId": 1111}
        else:
            page = (params or {}).get("page", 1)
            if page == 1:
                resp.json.return_value = {
                    "data": [
                        {
                            "id": "sub_1",
                            "release_name": "Movie.Release.1.srt",
                            "language": "Arabic",
                        },
                        {
                            "id": "sub_2",
                            "release_name": "Movie.Release.2.srt",
                            "language": "Arabic",
                        },
                        {
                            "id": "sub_3",
                            "release_name": "Movie.Release.3.srt",
                            "language": "Arabic",
                        },
                    ]
                }
            elif page == 2:
                resp.json.return_value = {
                    "data": [
                        {
                            "id": "sub_4",
                            "release_name": "Movie.Release.4.srt",
                            "language": "Arabic",
                        },
                        {
                            "id": "sub_5",
                            "release_name": "Movie.Release.5.srt",
                            "language": "Arabic",
                        },
                        {
                            "id": "sub_6",
                            "release_name": "Movie.Release.6.srt",
                            "language": "Arabic",
                        },
                    ]
                }
            elif page == 3:
                resp.json.return_value = {
                    "data": [
                        {
                            "id": "sub_7",
                            "release_name": "Movie.Release.7.srt",
                            "language": "Arabic",
                        },
                    ]
                }
            else:
                resp.json.return_value = {"data": []}
        return resp

    mock_client.get.side_effect = side_effect

    provider = SubsourceProvider(mock_client)
    subs = await provider.search_subtitles(
        imdb_id="tt0111161",
        api_key="test_key",
        languages=["ara"],
    )

    # 3 from page 1 + 3 from page 2 + 1 from page 3 = 7 total subtitles
    assert len(subs) == 7
    # 1 search call + 3 pagination page calls = 4 total calls
    assert mock_client.get.call_count == 4
    for i, s in enumerate(subs, 1):
        assert f"Movie.Release.{i}.srt" in s.release_name
        assert s.provider == "subsource"
        assert s.lang == "ara"


@pytest.mark.asyncio
async def test_subsource_all_subtitles_returned_without_type_filtering():
    """Verify SubSource returns all subtitles across all types without filtering."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    def side_effect(url, params=None, headers=None, timeout=None):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        if "movies/search" in url:
            resp.json.return_value = {"movieId": 5555}
        else:
            resp.json.return_value = {
                "data": [
                    {
                        "id": "s_trans",
                        "release_name": "Movie.Translated.srt",
                        "language": "Arabic",
                        "type": "translated",
                    },
                    {
                        "id": "s_retail",
                        "release_name": "Movie.Retail.srt",
                        "language": "Arabic",
                        "type": "retail",
                    },
                    {
                        "id": "s_ai",
                        "release_name": "Movie.AI.srt",
                        "language": "Arabic",
                        "type": "machine",
                    },
                    {
                        "id": "s_forced",
                        "release_name": "Movie.Forced.srt",
                        "language": "Arabic",
                        "type": "forced",
                    },
                ]
            }
        return resp

    mock_client.get.side_effect = side_effect

    provider = SubsourceProvider(mock_client)
    subs = await provider.search_subtitles(
        imdb_id="tt0111161",
        api_key="test_key",
        languages=["ara"],
    )
    assert len(subs) == 4
    names = [s.release_name for s in subs]
    assert "Movie.Retail.srt" in names
    assert "Movie.Translated.srt" in names
    assert "Movie.AI.srt" in names
    assert "Movie.Forced.srt" in names


@pytest.mark.asyncio
async def test_subsource_resolve_movie_id_direct_imdb():
    """Verify resolve_movie_id performs single direct lookup with searchType=imdb."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {"movieId": 97027, "name": "The Shawshank Redemption"}
    resp.text = '{"movieId": 97027}'
    mock_client.get.return_value = resp

    service = SubSourceService("key_123")
    movie_id = await service.resolve_movie_id(
        client=mock_client,
        imdb_id="tt0111161",
        title="The Shawshank Redemption",
        year=1994,
    )

    assert movie_id == 97027
    # Exactly 1 call should be made directly to IMDb searchType
    assert mock_client.get.call_count == 1
    args, kwargs = mock_client.get.call_args
    assert kwargs["params"] == {"searchType": "imdb", "imdb": "tt0111161"}


@pytest.mark.asyncio
async def test_subsource_resolve_movie_id_text_fallback():
    """Verify resolve_movie_id falls back to searchType=text only if IMDb query yields no result."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    def side_effect(url, params=None, headers=None, timeout=None):
        resp = MagicMock(spec=httpx.Response)
        if params.get("searchType") == "imdb":
            resp.status_code = 200
            resp.json.return_value = {"data": []}
            resp.text = '{"data": []}'
        elif params.get("searchType") == "text":
            resp.status_code = 200
            resp.json.return_value = {"movieId": 97027, "name": "The Shawshank Redemption"}
            resp.text = '{"movieId": 97027}'
        return resp

    mock_client.get.side_effect = side_effect

    service = SubSourceService("key_123")
    movie_id = await service.resolve_movie_id(
        client=mock_client,
        imdb_id="tt0111161",
        title="The Shawshank Redemption",
        year=1994,
    )

    assert movie_id == 97027
    assert mock_client.get.call_count == 2

    # First call: searchType=imdb
    call1_args, call1_kwargs = mock_client.get.call_args_list[0]
    assert call1_kwargs["params"] == {"searchType": "imdb", "imdb": "tt0111161"}

    # Second call: searchType=text fallback with q and year
    call2_args, call2_kwargs = mock_client.get.call_args_list[1]
    assert call2_kwargs["params"] == {
        "searchType": "text",
        "q": "The Shawshank Redemption",
        "year": 1994,
    }


@pytest.mark.asyncio
async def test_subsource_filtering_loop_strictly_bound_hi(caplog):
    """Verify filtering loop strictly binds hearingImpaired, drops only HI when exclude_hi=True, and logs drop counts."""
    import logging

    caplog.set_level(logging.INFO)

    mock_client = AsyncMock(spec=httpx.AsyncClient)

    def side_effect(url, params=None, headers=None, timeout=None):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        if "movies/search" in url:
            resp.json.return_value = {"movieId": 12345}
        else:
            resp.json.return_value = {
                "data": [
                    {
                        "subtitleId": "sub_retail",
                        "releaseInfo": ["Movie.1994.Retail.1080p"],
                        "language": "Arabic",
                        "productionType": "retail",
                        "hearingImpaired": False,
                    },
                    {
                        "subtitleId": "sub_machine",
                        "releaseInfo": ["Movie.1994.AI.1080p"],
                        "language": "Arabic",
                        "commentary": "Generated via chatgpt",
                        "hearingImpaired": False,
                    },
                    {
                        "subtitleId": "sub_forced",
                        "releaseInfo": ["Movie.1994.Forced.1080p"],
                        "language": "Arabic",
                        "foreignParts": True,
                        "hearingImpaired": False,
                    },
                    {
                        "subtitleId": "sub_trans_hi",
                        "releaseInfo": ["Movie.1994.FanSub.HI.1080p"],
                        "language": "Arabic",
                        "productionType": None,
                        "foreignParts": False,
                        "hearingImpaired": True,
                    },
                    {
                        "subtitleId": "sub_trans_normal",
                        "releaseInfo": ["Movie.1994.FanSub.1080p"],
                        "language": "Arabic",
                        "productionType": None,
                        "foreignParts": False,
                        "hearingImpaired": False,
                    },
                    {
                        "subtitleId": "sub_trans_snake_hi",
                        "releaseInfo": ["Movie.1994.FanSub2.1080p"],
                        "language": "Arabic",
                        "productionType": None,
                        "foreignParts": False,
                        # Not camelCase hearingImpaired: True -> should NOT be dropped by strictly bound check
                        "hearing_impaired": True,
                    },
                ]
            }
        return resp

    mock_client.get.side_effect = side_effect

    service = SubSourceService("key_test")
    # All types should pass without type filtering, exclude_hi=True
    subs = await service.get_subtitles(
        client=mock_client,
        media_type="movie",
        imdb_id="tt0111161",
        language="Arabic",
        exclude_hi=True,
    )

    # Expected surviving:
    # 1. sub_retail (retail, HI False)
    # 2. sub_machine (machine, HI False)
    # 3. sub_forced (forced, HI False)
    # 4. sub_trans_normal (translated, HI False)
    # 5. sub_trans_snake_hi (translated, hearingImpaired is None, not True)
    assert len(subs) == 5
    sub_ids = [s["id"] for s in subs]
    assert "sub_retail" in sub_ids
    assert "sub_machine" in sub_ids
    assert "sub_forced" in sub_ids
    assert "sub_trans_normal" in sub_ids
    assert "sub_trans_snake_hi" in sub_ids
    assert "sub_trans_hi" not in sub_ids

    # Verify log output for live reporting of dropped counts by category
    log_records = [r.message for r in caplog.records if "[Filter Verification]" in r.message]
    assert len(log_records) == 1
    log_msg = log_records[0]

    assert "Total: 6" in log_msg
    assert "Dropped (HI): 1" in log_msg
    assert "Final Passed: 5" in log_msg


def test_matches_series_episode():
    """Verify series episode filtering in SubSource provider."""
    from app.providers.subsource import _matches_series_episode

    target_s, target_e = 1, 2

    # Episode 91 dash number should be rejected for S01E02
    assert not _matches_series_episode(
        {"releaseInfo": ["Hunter X Hunter-91 [1080p].srt"]}, target_s, target_e, ""
    )

    # Clean Episode 2 should be accepted
    assert _matches_series_episode(
        {"releaseInfo": ["Hunter X Hunter.2011.E2.srt"]}, target_s, target_e, ""
    )

    # Bracket episode [02] should be accepted
    assert _matches_series_episode(
        {"releaseInfo": ["[MST-Luckysubs] Hunter X Hunter [02] [BD]"]}, target_s, target_e, ""
    )

    # Episode range covering episode 2 (001-025) should be accepted
    assert _matches_series_episode(
        {"releaseInfo": ["[MST-Luckysubs] Hunter X Hunter 001-025 [BD]"]}, target_s, target_e, ""
    )

    # Episode range NOT covering episode 2 (026-050) should be rejected
    assert not _matches_series_episode(
        {"releaseInfo": ["Hunter X Hunter 026-050 [BD]"]}, target_s, target_e, ""
    )

    # Standard SxxExx
    assert _matches_series_episode(
        {"release_name": "Show.S01E02.1080p"}, target_s, target_e, "Show.S01E02.1080p"
    )
    assert not _matches_series_episode(
        {"release_name": "Show.S01E03.1080p"}, target_s, target_e, "Show.S01E03.1080p"
    )
    assert not _matches_series_episode(
        {"release_name": "Show.S02E02.1080p"}, target_s, target_e, "Show.S02E02.1080p"
    )

    # Explicit season and episode fields
    assert _matches_series_episode({"season": 1, "episode": 2}, target_s, target_e, "")
    assert not _matches_series_episode({"season": 1, "episode": 5}, target_s, target_e, "")
    assert not _matches_series_episode({"season": 2, "episode": 2}, target_s, target_e, "")
