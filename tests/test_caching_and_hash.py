"""Comprehensive tests for OpenSubtitles MovieHash matching and TTLCache aggregation caching."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import MatchTier, SubtitleRelease
from app.providers.opensubtitles import OpenSubtitlesProvider
from app.providers.subdl import SubdlProvider
from app.providers.subsource import SubsourceProvider
from app.services.aggregator import aggregate_subtitles
from app.services.cache import (
    SUBTITLE_CACHE,
    build_cache_key,
    clear_subtitle_cache,
    get_cached_subtitles,
    set_cached_subtitles,
)
from app.services.ranking import extract_stream_params
from app.services.subtitle_matcher import (
    calculate_compatibility,
)
from app.utils.config_parser import encode_user_config


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


# ==========================================
# 1. MOVIEHASH SCORING & RANKING TESTS
# ==========================================








# ==========================================
# 2. OPENSUBTITLES MOVIEHASH QUERY TESTS
# ==========================================


@pytest.mark.asyncio
async def test_opensubtitles_provider_sends_moviehash_and_sets_flag():
    """
    Verify OpenSubtitlesProvider properly sets moviehash and moviebytesize params,
    and correctly flags returned SubtitleRelease with is_hash_match=True.
    """
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    provider = OpenSubtitlesProvider(mock_client)

    mock_resp_data = {
        "total_count": 1,
        "data": [
            {
                "id": "os_sub_123",
                "attributes": {
                    "release": "Movie.2024.1080p.WEB-DL-GROUP",
                    "language": "ar",
                    "moviehash_match": True,
                    "hearing_impaired": False,
                    "files": [{"file_id": 98765, "file_name": "Movie.2024.1080p.srt"}],
                },
            }
        ],
    }

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_resp_data
    mock_client.get.return_value = mock_resp

    results = await provider.search_subtitles(
        imdb_id="tt1234567",
        api_key="valid_api_key",
        video_hash="8e245d9679d31e12",
        video_size=1048576000,
    )

    assert len(results) == 1
    assert results[0].is_hash_match is True
    assert results[0].provider == "opensubtitles"

    # Verify query params passed to httpx client
    call_args = mock_client.get.call_args
    assert call_args is not None
    params = call_args.kwargs.get("params", {})
    assert params.get("moviehash") == "8e245d9679d31e12"
    assert params.get("moviebytesize") == "1048576000"


# ==========================================
# 3. HIGH-PERFORMANCE IN-MEMORY CACHE TESTS
# ==========================================


@pytest.mark.asyncio
async def test_aggregator_caching_returns_instantly_without_reinvoking_providers():
    """
    Verify calling aggregate_subtitles twice with identical parameters returns cached
    results without re-invoking upstream providers (0ms external API overhead).
    """
    clear_subtitle_cache()

    mock_subdl = AsyncMock()
    mock_subdl.search_subtitles.return_value = [
        SubtitleRelease(
            release_name="Gladiator.II.2024.1080p.BluRay.x264.srt",
            download_url="https://subdl.com/sub.zip",
            provider="subdl",
            lang="ara",
        )
    ]
    mock_subsource = AsyncMock()
    mock_subsource.search_subtitles.return_value = []
    mock_opensubtitles = AsyncMock()
    mock_opensubtitles.search_subtitles.return_value = []

    # First call: cache miss, invokes providers
    first_res = await aggregate_subtitles(
        imdb_id="tt2104996",
        media_type="movie",
        filename="Gladiator.II.2024.1080p.BluRay.x264.mkv",
        subdl_key="test_subdl_key",
        subdl_provider=mock_subdl,
        subsource_provider=mock_subsource,
        opensubtitles_provider=mock_opensubtitles,
        use_cache=True,
    )

    assert len(first_res) == 1
    assert mock_subdl.search_subtitles.call_count == 1
    assert mock_subsource.search_subtitles.call_count == 1

    # Second call with exact identical parameters: cache hit, zero provider calls
    second_res = await aggregate_subtitles(
        imdb_id="tt2104996",
        media_type="movie",
        filename="Gladiator.II.2024.1080p.BluRay.x264.mkv",
        subdl_key="test_subdl_key",
        subdl_provider=mock_subdl,
        subsource_provider=mock_subsource,
        opensubtitles_provider=mock_opensubtitles,
        use_cache=True,
    )

    assert len(second_res) == 1
    assert second_res[0].release_name == first_res[0].release_name
    # Provider call counts must remain 1
    assert mock_subdl.search_subtitles.call_count == 1
    assert mock_subsource.search_subtitles.call_count == 1


@pytest.mark.asyncio
async def test_aggregator_cache_handles_missing_optional_parameters_gracefully():
    """
    Verify aggregator handles missing optional parameters (season, episode, filename,
    hash, size, preferences) cleanly and caches successfully without error.
    """
    clear_subtitle_cache()

    mock_subdl = AsyncMock()
    mock_subdl.search_subtitles.return_value = [
        SubtitleRelease(
            release_name="Inception.2010.1080p.BluRay.srt",
            download_url="https://subdl.com/sub.zip",
            provider="subdl",
            lang="ara",
        )
    ]
    mock_subsource = AsyncMock()
    mock_subsource.search_subtitles.return_value = []
    mock_os = AsyncMock()
    mock_os.search_subtitles.return_value = []

    # Call with bare minimum parameters
    res1 = await aggregate_subtitles(
        imdb_id="tt1375666",
        subdl_provider=mock_subdl,
        subsource_provider=mock_subsource,
        opensubtitles_provider=mock_os,
    )

    assert len(res1) == 1
    assert mock_subdl.search_subtitles.call_count == 1

    # Second call also with bare minimum parameters
    res2 = await aggregate_subtitles(
        imdb_id="tt1375666",
        subdl_provider=mock_subdl,
        subsource_provider=mock_subsource,
        opensubtitles_provider=mock_os,
    )

    assert len(res2) == 1
    assert mock_subdl.search_subtitles.call_count == 1


def test_cache_key_deterministic_and_order_independent():
    """
    Verify cache key is deterministic, sensitive to differentiating factors,
    and insensitive to language list ordering.
    """
    key1 = build_cache_key(
        media_type="movie",
        imdb_id="tt0111161",
        languages=["ara", "eng"],
        filename="Shawshank.mkv",
        video_hash="hash123",
        video_size=1024,
    )

    # Same parameters but different language order
    key2 = build_cache_key(
        media_type="movie",
        imdb_id="tt0111161",
        languages=["eng", "ara"],
        filename="Shawshank.mkv",
        video_hash="hash123",
        video_size=1024,
    )

    assert key1 == key2, "Cache key should be identical regardless of language list order"

    # Different filename
    key_diff_file = build_cache_key(
        media_type="movie",
        imdb_id="tt0111161",
        languages=["ara", "eng"],
        filename="Shawshank.Remastered.mkv",
        video_hash="hash123",
        video_size=1024,
    )
    assert key1 != key_diff_file

    # Different hash
    key_diff_hash = build_cache_key(
        media_type="movie",
        imdb_id="tt0111161",
        languages=["ara", "eng"],
        filename="Shawshank.mkv",
        video_hash="diff_hash",
        video_size=1024,
    )
    assert key1 != key_diff_hash

    # Different season / episode
    key_series = build_cache_key(
        media_type="series",
        imdb_id="tt0903747",
        season=1,
        episode=1,
    )
    key_series_ep2 = build_cache_key(
        media_type="series",
        imdb_id="tt0903747",
        season=1,
        episode=2,
    )
    assert key_series != key_series_ep2


def test_ttl_cache_capacity_and_clear():
    """
    Verify TTLCache configuration (maxsize=1000, ttl=21600) and clear_subtitle_cache.
    """
    assert SUBTITLE_CACHE.maxsize == 1000
    assert SUBTITLE_CACHE.ttl == 21600

    set_cached_subtitles(
        "sample_key",
        [
            SubtitleRelease(
                release_name="Test.srt",
                download_url="http://test.srt",
                provider="subdl",
            )
        ],
    )

    cached = get_cached_subtitles("sample_key")
    assert cached is not None
    assert len(cached) == 1

    clear_subtitle_cache()
    assert get_cached_subtitles("sample_key") is None


# ==========================================
# 4. STREAM PARAMS EXTRACTION & STREMIO CONTRACT
# ==========================================


def test_extract_stream_params_parsing():
    """
    Verify extract_stream_params parses filename, videoHash, and videoSize from {extra} and query params.
    """
    extra_combined = (
        "videoHash=a1b2c3d4e5f67890&videoSize=524288000&filename=Interstellar.2014.2160p.mkv.json"
    )
    params = extract_stream_params(extra_combined)
    assert params["filename"] == "Interstellar.2014.2160p.mkv"
    assert params["video_hash"] == "a1b2c3d4e5f67890"
    assert params["video_size"] == 524288000

    # From query params. The hash must be a valid 16-hex MovieHash: the
    # ingestion boundary now rejects anything else as unavailable rather than
    # forwarding a malformed value into a metered OpenSubtitles lookup.
    params_q = extract_stream_params(
        None,
        query_params={
            "videoHash": "a1b2c3d4e5f67890",
            "videoSize": "2048",
            "filename": "Movie.mkv",
        },
    )
    assert params_q["filename"] == "Movie.mkv"
    assert params_q["video_hash"] == "a1b2c3d4e5f67890"
    assert params_q["video_size"] == 2048


def test_stremio_endpoint_caching_and_hash_integration(client):
    """
    Verify end-to-end that:
    1. A Stremio subtitle request with videoHash/videoSize/filename calls providers once.
    2. A repeated request with the same parameters returns instantly from cache without re-invoking providers.
    """
    clear_subtitle_cache()

    mock_release = SubtitleRelease(
        release_name="Gladiator.II.2024.1080p.BluRay.x264.srt",
        download_url="/sub/mock.srt",
        provider="subdl",
        format="srt",
        lang="ara",
    )

    mock_subdl = AsyncMock(return_value=[mock_release])
    mock_subsource = AsyncMock(return_value=[])
    mock_os = AsyncMock(return_value=[])

    with (
        patch("app.main._http_client", new_callable=AsyncMock),
        patch("app.providers.subdl.SubdlProvider.search_subtitles", new=mock_subdl),
        patch("app.providers.subsource.SubsourceProvider.search_subtitles", new=mock_subsource),
        patch("app.providers.opensubtitles.OpenSubtitlesProvider.search_subtitles", new=mock_os),
        patch(
            "app.providers.cinemeta.CinemetaClient.get_metadata",
            new=AsyncMock(return_value={"title": "Gladiator II", "year": 2024}),
        ),
    ):
        url = "/subtitles/movie/tt2104996/videoHash=1234567890abcdef&videoSize=1048576&filename=Gladiator.II.2024.1080p.BluRay.x264.mkv.json"

        # Request 1
        resp1 = client.get(url)
        assert resp1.status_code == 200
        data1 = resp1.json()
        assert len(data1["subtitles"]) >= 1
        assert mock_subdl.call_count == 1

        # Request 2 (Identical: seeks, pauses, reconnections)
        resp2 = client.get(url)
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert len(data2["subtitles"]) == len(data1["subtitles"])

        # Call count must still be 1 (0ms upstream network overhead!)
        assert mock_subdl.call_count == 1


def test_cache_clear_endpoint(client):
    """Verify /cache/clear clears the in-memory subtitle TTLCache."""
    from app.services.cache import get_cached_subtitles, set_cached_subtitles

    set_cached_subtitles(
        "test_key",
        [SubtitleRelease(release_name="Test.srt", download_url="http://test", provider="subdl")],
    )
    assert get_cached_subtitles("test_key") is not None

    resp = client.get("/cache/clear")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert get_cached_subtitles("test_key") is None


def test_cache_bypass_via_query_param(client):
    """Verify bypass_cache=1 or nocache=1 bypasses in-memory TTLCache."""
    from app.services.cache import clear_subtitle_cache

    clear_subtitle_cache()

    mock_release = SubtitleRelease(
        release_name="Gladiator.II.2024.1080p.BluRay.x264.srt",
        download_url="/sub/mock.srt",
        provider="subdl",
        format="srt",
        lang="ara",
    )

    mock_subdl = AsyncMock(return_value=[mock_release])

    with (
        patch("app.main._http_client", new_callable=AsyncMock),
        patch("app.providers.subdl.SubdlProvider.search_subtitles", new=mock_subdl),
        patch(
            "app.providers.subsource.SubsourceProvider.search_subtitles",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "app.providers.opensubtitles.OpenSubtitlesProvider.search_subtitles",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "app.providers.cinemeta.CinemetaClient.get_metadata", new=AsyncMock(return_value=None)
        ),
    ):
        url = "/subtitles/movie/tt2104996/filename=Gladiator.II.2024.1080p.BluRay.x264.mkv.json"

        resp1 = client.get(url)
        assert resp1.status_code == 200
        assert mock_subdl.call_count == 1

        # Request with bypass_cache=1 should query upstream provider again
        resp2 = client.get(f"{url}?bypass_cache=1")
        assert resp2.status_code == 200
        assert mock_subdl.call_count == 2


# ============================================================================
# 5. BINARY HASH MATCH HIERARCHY & TIERS 1-4 FALLBACK TESTS
# ============================================================================


def test_match_tier_enum_values_and_ordering():
    """Verify MatchTier values strictly represent the hierarchy: HASH (0) to FALLBACK (4)."""
    assert MatchTier.HASH == 0
    assert MatchTier.EXACT == 1
    assert MatchTier.SOURCE_FAMILY == 2
    assert MatchTier.CLOSE == 3
    assert MatchTier.FALLBACK == 4

    assert MatchTier.HASH < MatchTier.EXACT < MatchTier.SOURCE_FAMILY < MatchTier.CLOSE < MatchTier.FALLBACK


def test_match_tier_string_compatibility():
    """Verify MatchTier seamlessly compares with case-insensitive strings and str()."""
    assert MatchTier.HASH == "hash"
    assert MatchTier.HASH == "HASH"
    assert MatchTier.EXACT == "exact"
    assert MatchTier.SOURCE_FAMILY == "source_family"
    assert MatchTier.CLOSE == "close"
    assert MatchTier.FALLBACK == "fallback"
    assert str(MatchTier.HASH) == "hash"








def test_tier_2_source_family_match():
    """Tier 2 (SOURCE_FAMILY): Same episode & source family (WEB-DL), but different release group."""
    video = "House.of.the.Dragon.S02E01.1080p.HMAX.WEB-DL.DDP5.1.Atmos.H.264-FLUX.mkv"
    sub = "House.of.the.Dragon.S02E01.1080p.WEB-DL.DDP5.1.Atmos-NTb.srt"

    compat = calculate_compatibility(video, sub, is_hash_match=False)
    assert compat.accepted is True
    assert compat.match_tier == MatchTier.SOURCE_FAMILY
    assert compat.match_tier == 2


def test_tier_3_close_match():
    """Tier 3 (CLOSE): Same episode, but cross-source (HDTV vs WEB-DL) or lower tier."""
    video = "House.of.the.Dragon.S02E01.1080p.HMAX.WEB-DL.DDP5.1.Atmos.H.264-FLUX.mkv"
    sub = "House.of.the.Dragon.S02E01.720p.HDTV.x264-SYNCOPY.srt"

    compat = calculate_compatibility(video, sub, is_hash_match=False)
    assert compat.accepted is True
    assert compat.match_tier in (MatchTier.CLOSE, MatchTier.SOURCE_FAMILY)


def test_tier_4_fallback_and_hard_rejection():
    """Tier 4 (FALLBACK): Hard rejected candidates get score -1000 and MatchTier.FALLBACK."""
    video = "House.of.the.Dragon.S02E01.1080p.mkv"
    sub_wrong_ep = "House.of.the.Dragon.S02E02.1080p.mkv"

    compat = calculate_compatibility(video, sub_wrong_ep, is_hash_match=False)
    assert compat.accepted is False
    assert compat.score == -1000
    assert compat.match_tier == MatchTier.FALLBACK
    assert compat.percentage == 0






@pytest.mark.asyncio
async def test_subdl_and_subsource_gracefully_ignore_hash_kwargs():
    """Ensure SubDL and SubSource handle video_hash and video_size kwargs without error."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get.return_value = httpx.Response(200, json={"subtitles": []})

    subdl = SubdlProvider(mock_client)
    res_subdl = await subdl.search_subtitles(
        imdb_id="tt0172495",
        video_hash="8e245d9679d31e12",
        video_size=104857600,
        moviehash="8e245d9679d31e12",
        moviebytesize=104857600,
    )
    assert isinstance(res_subdl, list)

    subsource = SubsourceProvider(mock_client)
    res_subsource = await subsource.search_subtitles(
        imdb_id="tt0172495",
        video_hash="8e245d9679d31e12",
        video_size=104857600,
        moviehash="8e245d9679d31e12",
        moviebytesize=104857600,
    )
    assert isinstance(res_subsource, list)




def test_series_sub_id_scoped_to_season_episode(client):
    """Verify series subtitles from a season pack receive episode-scoped sub_ids."""
    sub_pack = SubtitleRelease(
        release_name="Mad.Men.S01.1080p.BluRay.DD5.1.x264-SA89.srt",
        download_url="https://subdl.com/pack.zip",
        provider="subdl",
        lang="ara",
    )

    user_cfg = encode_user_config(subdl_key="test_key")
    with (
        patch("app.providers.subdl.SubdlProvider.search_subtitles", new=AsyncMock(return_value=[sub_pack])),
        patch("app.providers.subsource.SubsourceProvider.search_subtitles", new=AsyncMock(return_value=[])),
        patch("app.providers.opensubtitles.OpenSubtitlesProvider.search_subtitles", new=AsyncMock(return_value=[])),
        patch("app.providers.cinemeta.CinemetaClient.get_metadata", new=AsyncMock(return_value=None)),
    ):
        resp_ep2 = client.get(f"/{user_cfg}/subtitles/series/tt0804503%3A1%3A2.json")
        assert resp_ep2.status_code == 200
        subs_ep2 = resp_ep2.json()["subtitles"]
        assert len(subs_ep2) == 1
        url_ep2 = subs_ep2[0]["url"]

        resp_ep3 = client.get(f"/{user_cfg}/subtitles/series/tt0804503%3A1%3A3.json")
        assert resp_ep3.status_code == 200
        subs_ep3 = resp_ep3.json()["subtitles"]
        assert len(subs_ep3) == 1
        url_ep3 = subs_ep3[0]["url"]

        # Different episodes of the same season pack MUST have distinct sub_ids
        assert url_ep2 != url_ep3


@pytest.mark.asyncio
async def test_serve_subtitle_cross_episode_collision_guard(client):
    """Verify that a stale cached subtitle for a different episode is discarded."""
    from app.main import cache_manager

    stale_sub_id = "stale_s1e2_sub"
    # Store metadata indicating season 1 episode 2
    cache_manager.store_metadata(
        stale_sub_id,
        {
            "sub_id": stale_sub_id,
            "imdb_id": "tt0804503",
            "media_type": "series",
            "provider": "subdl",
            "download_url": "https://subdl.com/pack.zip",
            "release_name": "Mad.Men.S01.srt",
            "season": 1,
            "episode": 2,
            "lang": "ara",
        },
    )
    # Save dummy subtitle content for ep 2
    await cache_manager.save_subtitle(stale_sub_id, b"1\n00:00:01,000 --> 00:00:05,000\nEpisode 2 Content\n")

    # Now simulate a request asking for Episode 3 with this stale sub_id
    valid_sub_bytes = b"1\n00:00:01,000 --> 00:00:05,000\nEpisode 3 Fresh Content\n"
    with patch("app.providers.subdl.SubdlProvider.download_archive", new=AsyncMock(return_value=valid_sub_bytes)):
        resp = client.get(f"/sub/{stale_sub_id}.srt?imdb=tt0804503&type=series&season=1&episode=3")
        assert resp.status_code == 200
        # Must NOT serve the stale Episode 2 content
        assert "Episode 2 Content" not in resp.text
        assert "Episode 3 Fresh Content" in resp.text


def test_reference_disk_cache_partial_roundtrip(tmp_path):
    """Verify ReferenceDiskCache preserves the partial flag across saves and gets."""
    from app.services.sync.cache import ReferenceDiskCache
    from app.services.sync.query import ReferenceQuery

    cache = ReferenceDiskCache(root=tmp_path / "refs", ttl=3600.0, min_bytes=10)
    query = ReferenceQuery(imdb_id="tt0804503", media_type="series", season=1, episode=3)
    text = "1\n00:00:01,000 --> 00:00:02,000\nPartial dialogue line\n"

    # Save with partial=True
    cache.set(query, "embedded", text, kind="embedded", partial=True)
    res = cache.get(query)
    assert res is not None
    assert res.kind == "embedded"
    assert res.partial is True

    # Save with partial=False (e.g. full local extraction)
    cache.set(query, "embedded", text, kind="embedded", partial=False)
    res2 = cache.get(query)
    assert res2 is not None
    assert res2.partial is False

