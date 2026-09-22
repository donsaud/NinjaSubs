"""Comprehensive tests for OpenSubtitles MovieHash matching and TTLCache aggregation caching."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import MatchTier, SubtitleRelease, UserPreferences
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
    WEIGHT_EXACT_HASH,
    calculate_compatibility,
    calculate_match_score,
    rank_subtitles,
)
from app.utils.config_parser import encode_user_config


@pytest.fixture
def client():
    return TestClient(app)


# ==========================================
# 1. MOVIEHASH SCORING & RANKING TESTS
# ==========================================


def test_hash_match_receives_absolute_priority_and_ranks_number_one():
    """
    Verify that an OpenSubtitles candidate matched via MovieHash:
    1. Receives absolute priority score (baseline >= 500).
    2. Receives a match percentage of 100%.
    3. Ranks #1 even if its filename is obscure or completely different from the video filename.
    """
    video_file = "House.of.the.Dragon.S02E01.1080p.HMAX.WEB-DL.DDP5.1.Atmos.H.264-FLUX.mkv"

    # Perfect filename-only match (Score is typically ~140-210, 100%)
    sub_filename_perfect = SubtitleRelease(
        release_name="House.of.the.Dragon.S02E01.1080p.HMAX.WEB-DL.DDP5.1.Atmos.H.264-FLUX.srt",
        download_url="https://subdl.com/sub1.srt",
        provider="subdl",
        format="srt",
        lang="ara",
        is_hash_match=False,
    )

    # Hash match with generic / obscure filename
    sub_hash_match = SubtitleRelease(
        release_name="HotD_201_arabic_subs.srt",
        download_url="https://api.opensubtitles.com/sub2.srt",
        provider="opensubtitles",
        format="srt",
        lang="ara",
        is_hash_match=True,
    )

    # Calculate match score directly
    score_res = calculate_match_score(video_file, sub_hash_match.release_name, is_hash_match=True)
    assert score_res.score >= 500, f"Expected score >= 500, got {score_res.score}"
    assert score_res.percentage == 100

    # Passing SubtitleRelease object directly to calculate_match_score
    score_obj_res = calculate_match_score(video_file, sub_hash_match)
    assert score_obj_res.score >= 500
    assert score_obj_res.percentage == 100

    # Passing candidate dict with is_hash_match
    score_dict_res = calculate_match_score(
        video_file, {"release_name": "HotD.srt", "is_hash_match": True}
    )
    assert score_dict_res.score >= 500
    assert score_dict_res.percentage == 100

    # Rank both candidates
    ranked = rank_subtitles(
        video_filename=video_file,
        subtitles=[sub_filename_perfect, sub_hash_match],
    )

    assert len(ranked) == 2
    assert ranked[0].is_hash_match is True
    assert ranked[0].provider == "opensubtitles"
    assert ranked[0].score > ranked[1].score
    assert ranked[0].score >= 500
    assert ranked[0].match_percentage == 100


def test_hash_match_never_rejected_by_episode_mismatch():
    """
    Ensure that a candidate matched by binary MovieHash is never rejected by
    strict episode mismatch logic even if the subtitle release filename suggests a mismatch.
    """
    video_file = "Series.Name.S01E01.1080p.WEB-DL.mkv"
    # Subtitle filename labeled differently (e.g. S01E02) but matched by exact stream hash
    sub_candidate = SubtitleRelease(
        release_name="Series.Name.S01E02.1080p.WEB-DL.srt",
        download_url="https://api.opensubtitles.com/sub.srt",
        provider="opensubtitles",
        format="srt",
        lang="ara",
        is_hash_match=True,
    )

    score_res = calculate_match_score(video_file, sub_candidate.release_name, is_hash_match=True)
    assert score_res.score >= 500
    assert score_res.percentage == 100


def test_hash_match_without_video_filename():
    """
    Ensure that when no video filename is available, hash-matched subtitles
    receive +500 baseline bonus and 100% match percentage.
    """
    sub1 = SubtitleRelease(
        release_name="Release.A.srt",
        download_url="url1",
        provider="subdl",
        is_hash_match=False,
    )
    sub2 = SubtitleRelease(
        release_name="Release.B.srt",
        download_url="url2",
        provider="opensubtitles",
        is_hash_match=True,
    )

    ranked = rank_subtitles(video_filename=None, subtitles=[sub1, sub2])
    assert ranked[0].is_hash_match is True
    assert ranked[0].score >= 500
    assert ranked[0].match_percentage == 100


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

    # From query params
    params_q = extract_stream_params(
        None, query_params={"videoHash": "hash999", "videoSize": "2048", "filename": "Movie.mkv"}
    )
    assert params_q["filename"] == "Movie.mkv"
    assert params_q["video_hash"] == "hash999"
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


def test_tier_0_deterministic_hash_match_short_circuits_mismatched_filename():
    """
    Tier 0 Short-Circuit:
    Even when the subtitle filename indicates a completely different show, season, or episode,
    a verified binary hash match short-circuits Stage 1 & Stage 2, granting 500 score, 100%, and MatchTier.HASH.
    """
    video_target = "House.of.the.Dragon.S02E01.1080p.HMAX.WEB-DL.mkv"
    mismatched_sub = "SpongeBob.SquarePants.S01E05.480p.srt"

    compat = calculate_compatibility(video_target, mismatched_sub, is_hash_match=True)

    assert compat.accepted is True
    assert compat.score == WEIGHT_EXACT_HASH  # 500
    assert compat.percentage == 100
    assert compat.match_tier == MatchTier.HASH
    assert compat.match_tier == 0
    assert compat.match_tier == "hash"
    assert compat.confidence == "deterministic"
    assert compat.match_method == "hash"
    assert compat.is_hash_match is True
    assert compat.reasons == ["Exact binary MovieHash match (100% sync guaranteed)"]


def test_tier_0_short_circuit_with_subtitle_release_object():
    """Verify passing SubtitleRelease with is_hash_match=True short-circuits in calculate_compatibility & calculate_match_score."""
    video_target = "Dune.Part.Two.2024.2160p.UHD.Remux.DV.Atmos-FLUX.mkv"
    sub_obj = SubtitleRelease(
        release_name="Arbitrary.Subtitle.Name.srt",
        download_url="https://api.opensubtitles.com/sub.srt",
        provider="opensubtitles",
        is_hash_match=True,
    )

    compat = calculate_compatibility(video_target, sub_obj)
    assert compat.match_tier == MatchTier.HASH
    assert compat.score == 500
    assert compat.percentage == 100
    assert compat.confidence == "deterministic"

    score_result = calculate_match_score(video_target, sub_obj)
    assert score_result.score == 500
    assert score_result.percentage == 100
    assert score_result.compatibility.match_tier == MatchTier.HASH


def test_tier_1_exact_release_group_and_source():
    """Tier 1 (EXACT): Perfect metadata match with identical release group and source."""
    video = "Breaking.Bad.S05E16.1080p.BluRay.x264-DEMAND.mkv"
    sub = "Breaking.Bad.S05E16.1080p.BluRay.x264-DEMAND.srt"

    compat = calculate_compatibility(video, sub, is_hash_match=False)
    assert compat.accepted is True
    assert compat.match_tier == MatchTier.EXACT
    assert compat.match_tier == 1
    assert compat.percentage == 100
    assert compat.score >= 110


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


def test_ranking_hierarchy_orders_hash_above_exact_text_match():
    """Verify that Tier 0 (HASH) candidate ranks #1 above Tier 1 (EXACT) text matches."""
    video_file = "Gladiator.2000.1080p.BluRay.x264-FLUX.mkv"

    # Perfect filename match (Tier 1 EXACT, score ~150-180)
    sub_exact = SubtitleRelease(
        release_name="Gladiator.2000.1080p.BluRay.x264-FLUX.srt",
        download_url="https://subdl.com/sub1.srt",
        provider="subdl",
        lang="ara",
        is_hash_match=False,
    )

    # Hash match candidate with generic/obscure name
    sub_hash = SubtitleRelease(
        release_name="arabic_subtitles_gladiator.srt",
        download_url="https://api.opensubtitles.com/sub2.srt",
        provider="opensubtitles",
        lang="ara",
        is_hash_match=True,
    )

    # Cross-source candidate (Tier 2 SOURCE_FAMILY)
    sub_source_family = SubtitleRelease(
        release_name="Gladiator.2000.1080p.WEB-DL.x264-NTb.srt",
        download_url="https://subdl.com/sub3.srt",
        provider="subdl",
        lang="ara",
        is_hash_match=False,
    )

    ranked = rank_subtitles(
        video_filename=video_file,
        subtitles=[sub_source_family, sub_exact, sub_hash],
        preferred_languages=["ara"],
    )

    assert len(ranked) == 3
    # Rank 1: Binary Hash Match
    assert ranked[0].is_hash_match is True
    assert ranked[0].match_tier == MatchTier.HASH
    assert ranked[0].score >= 500
    assert ranked[0].match_percentage == 100

    # Rank 2: Exact Filename Match
    assert ranked[1].is_hash_match is False
    assert ranked[1].match_tier == MatchTier.EXACT
    assert ranked[1].release_name == sub_exact.release_name

    # Rank 3: Source Family Match
    assert ranked[2].is_hash_match is False
    assert ranked[2].release_name == sub_source_family.release_name


@pytest.mark.asyncio
async def test_aggregator_deduplication_preserves_hash_match_over_text_duplicate():
    """
    If SubDL and OpenSubtitles return the same release name, but OpenSubtitles verified a binary hash match,
    the aggregator deduplication must preserve the hash-matched candidate.
    """
    clear_subtitle_cache()

    shared_release_name = "Gladiator.2000.1080p.BluRay.x264-FLUX.srt"

    subdl_item = SubtitleRelease(
        release_name=shared_release_name,
        download_url="https://subdl.com/sub.srt",
        provider="subdl",
        lang="ara",
        is_hash_match=False,
    )

    os_item = SubtitleRelease(
        release_name=shared_release_name,
        download_url="https://opensubtitles.com/sub.srt",
        provider="opensubtitles",
        lang="ara",
        is_hash_match=True,
    )

    mock_subdl = AsyncMock(return_value=[subdl_item])
    mock_subsource = AsyncMock(return_value=[])
    mock_os = AsyncMock(return_value=[os_item])

    with (
        patch("app.main._http_client", new_callable=AsyncMock),
        patch("app.providers.subdl.SubdlProvider.search_subtitles", new=mock_subdl),
        patch("app.providers.subsource.SubsourceProvider.search_subtitles", new=mock_subsource),
        patch("app.providers.opensubtitles.OpenSubtitlesProvider.search_subtitles", new=mock_os),
    ):
        results = await aggregate_subtitles(
            imdb_id="tt0172495",
            filename="Gladiator.2000.1080p.BluRay.x264-FLUX.mkv",
            video_hash="8e245d9679d31e12",
            video_size=104857600,
            use_cache=False,
            subdl_key="key1",
            opensubtitles_key="key2",
            user_preferences=UserPreferences(enable_opensubtitles=True),
            http_client=AsyncMock(spec=httpx.AsyncClient),
        )

        assert len(results) == 1
        assert results[0].is_hash_match is True
        assert results[0].provider == "opensubtitles"
        assert results[0].match_percentage == 100
        assert results[0].match_tier == MatchTier.HASH


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


def test_stremio_endpoint_hash_short_circuit_e2e(client):
    """
    End-to-End Stremio endpoint test:
    Verify Stremio request passing videoHash and videoSize returns the hash match with [100%] [OpenSubtitles] as #1.
    """
    clear_subtitle_cache()

    target_video = "Gladiator.2000.1080p.BluRay.x264-FLUX.mkv"

    sub_exact_text = SubtitleRelease(
        release_name="Gladiator.2000.1080p.BluRay.x264-FLUX.srt",
        download_url="https://subdl.com/gladiator.srt",
        provider="subdl",
        lang="ara",
        is_hash_match=False,
    )

    sub_hash_match = SubtitleRelease(
        release_name="Gladiator.Arabic.Subtitles.srt",
        download_url="https://api.opensubtitles.com/gladiator_hash.srt",
        provider="opensubtitles",
        lang="ara",
        is_hash_match=True,
    )

    with (
        patch("app.main._http_client", new_callable=AsyncMock),
        patch(
            "app.providers.subdl.SubdlProvider.search_subtitles",
            new=AsyncMock(return_value=[sub_exact_text]),
        ),
        patch(
            "app.providers.subsource.SubsourceProvider.search_subtitles",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "app.providers.opensubtitles.OpenSubtitlesProvider.search_subtitles",
            new=AsyncMock(return_value=[sub_hash_match]),
        ),
        patch("app.providers.cinemeta.CinemetaClient.get_metadata", new=AsyncMock(return_value=None)),
    ):
        user_cfg = encode_user_config(enable_opensubtitles=True)
        url = (
            f"/{user_cfg}/subtitles/movie/tt0172495/"
            f"videoHash=8e245d9679d31e12&videoSize=2147483648&filename={target_video}.json"
        )
        response = client.get(url)
        assert response.status_code == 200
        data = response.json()
        assert "subtitles" in data
        items = data["subtitles"]
        assert len(items) == 2

        # Candidate #1: The verified hash match
        assert "[100%] [OpenSubtitles]" in items[0]["title"]
        assert "Gladiator.Arabic.Subtitles" in items[0]["title"]

        # Candidate #2: The text-matched release
        assert "[OpenSubtitles]" not in items[1]["title"]
        assert "[SubDL]" in items[1]["title"]
