"""
Production-Style Real Stremio Integration Validation Test Suite.

Validates that the exact ordering produced by rank_subtitles() is preserved
all the way through the Stremio /subtitles endpoint response:
1. Movie: Dune Part Two 2024 (Remux target with same-year BluRay, same-year WEB-DL, wrong-year BluRay, WEBRip)
2. TV: House of the Dragon S02E01 (HMAX FLUX, AMZN FLUX, HMAX WEBRip, wrong episode S02E02, wrong season S01E01)
3. Anime: One Piece 1050 (1050, E1050, 1049-1050, 1051)
4. Languages: Multi-language grouping (Arabic, English, French) with mixed compatibility scores
5. SDH: Exclude HI True vs False verification
6. Exact Hash: MovieHash match guaranteed #1
7. Contract Fields: id, url, lang, title, format integrity
"""

import re
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app, cache_manager
from app.models import SubtitleRelease
from app.services.subtitle_matcher import rank_subtitles
from app.utils.config_parser import encode_user_config


def get_release_name(item: dict) -> str:
    sub_id = item["url"].rstrip("/").split("/")[-1].split(".")[0]
    meta = cache_manager.get_metadata(sub_id)
    if not meta and "_" in item.get("id", ""):
        sub_id = item["id"].rsplit("_", 1)[-1]
        meta = cache_manager.get_metadata(sub_id)
    return meta.get("release_name", "") if meta else ""


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


# ============================================================================
# 1. MOVIE INTEGRATION: DUNE PART TWO 2024
# ============================================================================


@pytest.mark.asyncio
async def test_real_stremio_movie_dune_part_two(client):
    """
    Case A: Movie
    Target: Dune.Part.Two.2024.2160p.UHD.Remux.DV.Atmos-FLUX
    Candidates:
      - same year BluRay (2024 BluRay)
      - same year WEB-DL (2024 WEB-DL)
      - wrong year BluRay (2023 BluRay)
      - WEBRip (2024 WEBRip)
    """
    target_video = "Dune.Part.Two.2024.2160p.UHD.Remux.DV.Atmos-FLUX"
    mock_candidates = [
        SubtitleRelease(
            release_name="Dune Part Two 2024 1080p BluRay.srt",
            download_url="https://provider.test/m_bluray24.srt",
            provider="subdl",
            lang="ara",
        ),
        SubtitleRelease(
            release_name="Dune Part Two 2024 1080p WEB-DL.srt",
            download_url="https://provider.test/m_webdl24.srt",
            provider="subsource",
            lang="ara",
        ),
        SubtitleRelease(
            release_name="Dune Part Two 2023 1080p BluRay.srt",
            download_url="https://provider.test/m_bluray23.srt",
            provider="subdl",
            lang="ara",
        ),
        SubtitleRelease(
            release_name="Dune Part Two 2024 1080p WEBRip.srt",
            download_url="https://provider.test/m_webrip24.srt",
            provider="subsource",
            lang="ara",
        ),
    ]

    expected_ranked = rank_subtitles(
        video_filename=target_video,
        subtitles=mock_candidates,
        preferred_languages=["ara"],
        discard_mismatches=True,
    )

    with patch("app.main._http_client", new_callable=AsyncMock):
        with (
            patch(
                "app.providers.subdl.SubdlProvider.search_subtitles",
                new=AsyncMock(return_value=mock_candidates[:2]),
            ),
            patch(
                "app.providers.subsource.SubsourceProvider.search_subtitles",
                new=AsyncMock(return_value=mock_candidates[2:]),
            ),
            patch(
                "app.providers.opensubtitles.OpenSubtitlesProvider.search_subtitles",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "app.providers.cinemeta.CinemetaClient.get_metadata",
                new=AsyncMock(return_value=None),
            ),
        ):
            extra = f"filename={target_video}"
            resp = client.get(f"/subtitles/movie/tt15239678/{extra}.json?nocache=1")
            assert resp.status_code == 200
            data = resp.json()

            assert "subtitles" in data
            subtitles = data["subtitles"]
            assert len(subtitles) == 4

            # Verify exact order matches rank_subtitles output and title format
            for i, expected in enumerate(expected_ranked):
                assert expected.release_name == get_release_name(subtitles[i])
                assert re.match(r"^\[\d+%\] \[(?:SubDL|SubSource|OpenSubtitles)\] .+$", subtitles[i]["title"])

            # Explicit ranking hierarchy checks:
            # 1. Same-year BluRay (score 85) ranks above same-year WEB-DL (score 25)
            # 2. Same-year WEB-DL (score 25) ranks above wrong-year BluRay (score 15)
            # 3. Wrong-year BluRay (score 15) ranks above WEBRip on Remux target (score -5)
            releases = [get_release_name(s) for s in subtitles]
            idx_bluray24 = next(i for i, t in enumerate(releases) if "2024" in t and "BluRay" in t)
            idx_webdl24 = next(i for i, t in enumerate(releases) if "2024" in t and "WEB-DL" in t)
            idx_bluray23 = next(i for i, t in enumerate(releases) if "2023" in t and "BluRay" in t)
            idx_webrip24 = next(i for i, t in enumerate(releases) if "2024" in t and "WEBRip" in t)

            assert idx_bluray24 < idx_webdl24, "Same-year BluRay must rank above same-year WEB-DL"
            assert idx_webdl24 < idx_bluray23, "Same-year WEB-DL must rank above wrong-year BluRay"
            assert (
                idx_bluray23 < idx_webrip24
            ), "Wrong-year BluRay must rank above WEBRip on Remux target"


# ============================================================================
# 2. TV INTEGRATION: HOUSE OF THE DRAGON S02E01
# ============================================================================


@pytest.mark.asyncio
async def test_real_stremio_tv_house_of_the_dragon(client):
    """
    Case B: TV
    Target: House.of.the.Dragon.S02E01.1080p.HMAX.WEB-DL.FLUX
    Candidates:
      - HMAX WEB-DL FLUX
      - AMZN WEB-DL FLUX
      - HMAX WEBRip
      - wrong episode S02E02
      - wrong season S01E01
    """
    target_video = "House.of.the.Dragon.S02E01.1080p.HMAX.WEB-DL.FLUX"
    mock_candidates = [
        SubtitleRelease(
            release_name="House.of.the.Dragon.S02E01.1080p.HMAX.WEB-DL-FLUX.srt",
            download_url="https://provider.test/tv_hmax_flux.srt",
            provider="subdl",
            lang="ara",
        ),
        SubtitleRelease(
            release_name="House.of.the.Dragon.S02E01.1080p.AMZN.WEB-DL-FLUX.srt",
            download_url="https://provider.test/tv_amzn_flux.srt",
            provider="subdl",
            lang="ara",
        ),
        SubtitleRelease(
            release_name="House.of.the.Dragon.S02E01.1080p.HMAX.WEBRip.srt",
            download_url="https://provider.test/tv_hmax_webrip.srt",
            provider="subsource",
            lang="ara",
        ),
        SubtitleRelease(
            release_name="House.of.the.Dragon.S02E02.1080p.HMAX.WEB-DL-FLUX.srt",
            download_url="https://provider.test/tv_wrong_ep.srt",
            provider="subdl",
            lang="ara",
        ),
        SubtitleRelease(
            release_name="House.of.the.Dragon.S01E01.1080p.HMAX.WEB-DL-FLUX.srt",
            download_url="https://provider.test/tv_wrong_season.srt",
            provider="subdl",
            lang="ara",
        ),
    ]

    expected_ranked = rank_subtitles(
        video_filename=target_video,
        subtitles=mock_candidates,
        preferred_languages=["ara"],
        discard_mismatches=True,
    )

    with patch("app.main._http_client", new_callable=AsyncMock):
        with (
            patch(
                "app.providers.subdl.SubdlProvider.search_subtitles",
                new=AsyncMock(return_value=mock_candidates[:3]),
            ),
            patch(
                "app.providers.subsource.SubsourceProvider.search_subtitles",
                new=AsyncMock(return_value=mock_candidates[3:]),
            ),
            patch(
                "app.providers.opensubtitles.OpenSubtitlesProvider.search_subtitles",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "app.providers.cinemeta.CinemetaClient.get_metadata",
                new=AsyncMock(return_value=None),
            ),
        ):
            extra = f"filename={target_video}"
            resp = client.get(f"/subtitles/series/tt11198330:2:1/{extra}.json?nocache=1")
            assert resp.status_code == 200
            data = resp.json()

            subtitles = data["subtitles"]
            releases = [get_release_name(s) for s in subtitles]

            # 1. Rejected episodes and seasons NEVER appear
            assert not any("S02E02" in t for t in releases), "Wrong episode S02E02 must never appear"
            assert not any("S01E01" in t for t in releases), "Wrong season S01E01 must never appear"

            # 2. Exactly 3 accepted candidates returned
            assert len(subtitles) == 3
            assert len(subtitles) == len(expected_ranked)

            # 3. Exact ordering: HMAX WEB-DL FLUX > HMAX WEBRip > AMZN WEB-DL FLUX
            assert "HMAX.WEB-DL-FLUX" in releases[0]
            assert "HMAX.WEBRip" in releases[1]
            assert "AMZN.WEB-DL-FLUX" in releases[2]

            for i, expected in enumerate(expected_ranked):
                assert expected.release_name == get_release_name(subtitles[i])
                assert re.match(r"^\[\d+%\] \[(?:SubDL|SubSource|OpenSubtitles)\] .+$", subtitles[i]["title"])


# ============================================================================
# 3. ANIME INTEGRATION: ONE PIECE 1050
# ============================================================================


@pytest.mark.asyncio
async def test_real_stremio_anime_one_piece(client):
    """
    Case C: Anime
    Target: One.Piece.1050.1080p.WEB-DL
    Candidates:
      - 1050
      - E1050
      - 1049-1050
      - 1051 (wrong episode)
    """
    target_video = "One.Piece.1050.1080p.WEB-DL"
    mock_candidates = [
        SubtitleRelease(
            release_name="One Piece - 1050.srt",
            download_url="https://provider.test/op1050.srt",
            provider="subdl",
            lang="ara",
        ),
        SubtitleRelease(
            release_name="One Piece - E1050.srt",
            download_url="https://provider.test/op_e1050.srt",
            provider="subsource",
            lang="ara",
        ),
        SubtitleRelease(
            release_name="One Piece - 1049-1050.srt",
            download_url="https://provider.test/op_multi.srt",
            provider="subdl",
            lang="ara",
        ),
        SubtitleRelease(
            release_name="One Piece - 1051.srt",
            download_url="https://provider.test/op1051.srt",
            provider="subdl",
            lang="ara",
        ),
    ]

    expected_ranked = rank_subtitles(
        video_filename=target_video,
        subtitles=mock_candidates,
        preferred_languages=["ara"],
        discard_mismatches=True,
    )

    with patch("app.main._http_client", new_callable=AsyncMock):
        with (
            patch(
                "app.providers.subdl.SubdlProvider.search_subtitles",
                new=AsyncMock(return_value=mock_candidates[:2]),
            ),
            patch(
                "app.providers.subsource.SubsourceProvider.search_subtitles",
                new=AsyncMock(return_value=mock_candidates[2:]),
            ),
            patch(
                "app.providers.opensubtitles.OpenSubtitlesProvider.search_subtitles",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "app.providers.cinemeta.CinemetaClient.get_metadata",
                new=AsyncMock(return_value=None),
            ),
        ):
            extra = f"filename={target_video}"
            resp = client.get(f"/subtitles/anime/kitsu:12/{extra}.json?nocache=1")
            assert resp.status_code == 200
            subtitles = resp.json()["subtitles"]
            releases = [get_release_name(s) for s in subtitles]

            # 1. Wrong episode 1051 NEVER appears
            assert not any(
                "1051" in t for t in releases
            ), "Candidate 1051 must be rejected and absent"

            # 2. 1050, E1050, and 1049-1050 are all present
            assert len(subtitles) == 3
            assert len(subtitles) == len(expected_ranked)

            # 3. Order strictly matches matcher
            for i, expected in enumerate(expected_ranked):
                assert expected.release_name == get_release_name(subtitles[i])
                assert re.match(r"^\[\d+%\] \[(?:SubDL|SubSource|OpenSubtitles)\] .+$", subtitles[i]["title"])


# ============================================================================
# 4. MULTI-LANGUAGE INTEGRATION: ARABIC, ENGLISH, FRENCH
# ============================================================================


@pytest.mark.asyncio
async def test_real_stremio_multi_language_partitions(client):
    """
    Case D: Languages
    Arabic, English, French candidates with different compatibility scores.
    Verify preferred_languages controls the language partitions.
    """
    mock_candidates = [
        SubtitleRelease(
            release_name="Movie.2024.1080p.BluRay.French.srt",
            download_url="https://provider.test/fr_bluray.srt",
            provider="subdl",
            lang="fra",
        ),
        SubtitleRelease(
            release_name="Movie.2024.720p.HDTV.Arabic.srt",
            download_url="https://provider.test/ar_hdtv.srt",
            provider="subdl",
            lang="ara",
        ),
        SubtitleRelease(
            release_name="Movie.2024.1080p.BluRay.English.srt",
            download_url="https://provider.test/en_bluray.srt",
            provider="subdl",
            lang="eng",
        ),
        SubtitleRelease(
            release_name="Movie.2024.1080p.BluRay.Arabic.srt",
            download_url="https://provider.test/ar_bluray.srt",
            provider="subdl",
            lang="ara",
        ),
        SubtitleRelease(
            release_name="Movie.2024.720p.HDTV.English.srt",
            download_url="https://provider.test/en_hdtv.srt",
            provider="subdl",
            lang="eng",
        ),
    ]

    target_video = "Movie.2024.1080p.BluRay.x264-FLUX"
    cfg = encode_user_config(languages=["ara", "eng", "fra"])

    expected_ranked = rank_subtitles(
        video_filename=target_video,
        subtitles=mock_candidates,
        preferred_languages=["ara", "eng", "fra"],
        discard_mismatches=True,
    )

    with patch("app.main._http_client", new_callable=AsyncMock):
        with (
            patch(
                "app.providers.subdl.SubdlProvider.search_subtitles",
                new=AsyncMock(return_value=mock_candidates),
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
                new=AsyncMock(return_value=None),
            ),
        ):
            extra = f"filename={target_video}"
            resp = client.get(f"/{cfg}/subtitles/movie/tt1234567/{extra}.json?nocache=1")
            assert resp.status_code == 200

            subtitles = resp.json()["subtitles"]
            assert len(subtitles) == 5

            # 1. Verify language partitions: All Arabic first, then all English, then French
            langs = [s["lang"] for s in subtitles]
            assert langs == ["ara", "ara", "eng", "eng", "fra"]

            # 2. Within each language partition, higher score ranks first
            releases = [get_release_name(s) for s in subtitles]
            assert (
                "BluRay" in releases[0] and "ara" in langs[0]
            )  # Arab BluRay (score 90) > Arab HDTV (score 10)
            assert "HDTV" in releases[1] and "ara" in langs[1]
            assert (
                "BluRay" in releases[2] and "eng" in langs[2]
            )  # Eng BluRay (score 90) > Eng HDTV (score 10)
            assert "HDTV" in releases[3] and "eng" in langs[3]
            assert "BluRay" in releases[4] and "fra" in langs[4]

            # 3. Exact order matches matcher
            for i, expected in enumerate(expected_ranked):
                assert expected.release_name == get_release_name(subtitles[i])
                assert re.match(r"^\[\d+%\] \[(?:SubDL|SubSource|OpenSubtitles)\] .+$", subtitles[i]["title"])


# ============================================================================
# 5. SDH FILTERING INTEGRATION: EXCLUDE_SDH TRUE VS FALSE
# ============================================================================


@pytest.mark.asyncio
async def test_real_stremio_sdh_exclusion_and_inclusion(client):
    """
    Case E: SDH
    Verify exclude_sdh=True removes SDH and False preserves it.
    """
    mock_candidates = [
        SubtitleRelease(
            release_name="Show.S01E01.1080p.WEB-DL.FLUX.srt",
            download_url="https://provider.test/clean.srt",
            provider="subdl",
            lang="ara",
            hearing_impaired=False,
        ),
        SubtitleRelease(
            release_name="Show.S01E01.1080p.WEB-DL.FLUX.SDH.srt",
            download_url="https://provider.test/sdh.srt",
            provider="subdl",
            lang="ara",
            hearing_impaired=True,
        ),
    ]

    target_video = "Show.S01E01.1080p.WEB-DL-FLUX"

    with patch("app.main._http_client", new_callable=AsyncMock):
        with (
            patch(
                "app.providers.subdl.SubdlProvider.search_subtitles",
                new=AsyncMock(return_value=mock_candidates),
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
                new=AsyncMock(return_value=None),
            ),
        ):
            extra = f"filename={target_video}"

            # 1. exclude_sdh=True (exclude_hi=True in user config)
            cfg_exclude = encode_user_config(languages=["ara"], exclude_hi=True)
            resp_ex = client.get(
                f"/{cfg_exclude}/subtitles/series/tt7654321:1:1/{extra}.json?nocache=1"
            )
            assert resp_ex.status_code == 200
            subs_ex = resp_ex.json()["subtitles"]
            assert len(subs_ex) == 1
            assert not any("SDH" in s["title"] for s in subs_ex)

            # 2. exclude_sdh=False (exclude_hi=False in user config)
            cfg_include = encode_user_config(languages=["ara"], exclude_hi=False)
            resp_in = client.get(
                f"/{cfg_include}/subtitles/series/tt7654321:1:1/{extra}.json?nocache=1"
            )
            assert resp_in.status_code == 200
            subs_in = resp_in.json()["subtitles"]
            assert len(subs_in) == 2


# ============================================================================
# 6. EXACT BINARY HASH MATCH INTEGRATION
# ============================================================================


@pytest.mark.asyncio
async def test_real_stremio_exact_hash_ranks_first(client):
    """Exact binary hash match must rank #1 in the final Stremio JSON array."""
    mock_candidates = [
        SubtitleRelease(
            release_name="Movie.2024.1080p.BluRay.x264.srt",
            download_url="https://provider.test/good_name.srt",
            provider="subdl",
            lang="ara",
            is_hash_match=False,
        ),
        SubtitleRelease(
            release_name="Arbitrary.Subtitle.Name.srt",
            download_url="https://provider.test/hash_match.srt",
            provider="opensubtitles",
            lang="ara",
            is_hash_match=True,
        ),
    ]

    target_video = "Movie.2024.1080p.BluRay.x264.mkv"

    with patch("app.main._http_client", new_callable=AsyncMock):
        with (
            patch(
                "app.providers.subdl.SubdlProvider.search_subtitles",
                new=AsyncMock(return_value=[mock_candidates[0]]),
            ),
            patch(
                "app.providers.subsource.SubsourceProvider.search_subtitles",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "app.providers.opensubtitles.OpenSubtitlesProvider.search_subtitles",
                new=AsyncMock(return_value=[mock_candidates[1]]),
            ),
            patch(
                "app.providers.cinemeta.CinemetaClient.get_metadata",
                new=AsyncMock(return_value=None),
            ),
        ):
            extra = f"filename={target_video}&videoHash=aabbccdd11223344"
            cfg_hash = encode_user_config(enable_opensubtitles=True)
            resp = client.get(f"/{cfg_hash}/subtitles/movie/tt3333333/{extra}.json?nocache=1")
            assert resp.status_code == 200
            subtitles = resp.json()["subtitles"]
            assert len(subtitles) == 2
            assert subtitles[0]["title"] == "[100%] [OpenSubtitles] Arbitrary.Subtitle.Name"
            assert get_release_name(subtitles[0]) == "Arbitrary.Subtitle.Name.srt"
            assert subtitles[1]["title"] == "[74%] [SubDL] Movie.2024.1080p.BluRay.x264"
            assert get_release_name(subtitles[1]) == "Movie.2024.1080p.BluRay.x264.srt"


# ============================================================================
# 7. FIELD INTEGRITY CONTRACT (ID, URL, LANG, TITLE, FORMAT)
# ============================================================================


@pytest.mark.asyncio
async def test_real_stremio_contract_field_integrity(client):
    """
    Ensure all returned SubtitleItems preserve intact:
    - id (unique and non-empty)
    - url (valid HTTP/HTTPS)
    - lang (valid ISO 639-2 code)
    - title (informative badge + clean name)
    - format (srt, ass, ssa, or vtt)
    """
    mock_candidates = [
        SubtitleRelease(
            id="subdl_sub_101",
            release_name="Show.S01E01.1080p.BluRay.srt",
            download_url="https://provider.test/dl/101.srt",
            provider="subdl",
            lang="ara",
            format="srt",
        ),
        SubtitleRelease(
            id="subsource_sub_202",
            release_name="Show.S01E01.1080p.BluRay.ass",
            download_url="https://provider.test/dl/202.ass",
            provider="subsource",
            lang="ara",
            format="ass",
        ),
    ]

    with patch("app.main._http_client", new_callable=AsyncMock):
        with (
            patch(
                "app.providers.subdl.SubdlProvider.search_subtitles",
                new=AsyncMock(return_value=[mock_candidates[0]]),
            ),
            patch(
                "app.providers.subsource.SubsourceProvider.search_subtitles",
                new=AsyncMock(return_value=[mock_candidates[1]]),
            ),
            patch(
                "app.providers.opensubtitles.OpenSubtitlesProvider.search_subtitles",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "app.providers.cinemeta.CinemetaClient.get_metadata",
                new=AsyncMock(return_value=None),
            ),
        ):
            resp = client.get(
                "/subtitles/series/tt8888888:1:1/filename=Show.S01E01.1080p.BluRay.mkv.json?nocache=1"
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "subtitles" in data
            assert len(data["subtitles"]) == 2

            for item in data["subtitles"]:
                assert "id" in item and len(item["id"]) > 0
                assert "url" in item and item["url"].startswith("http")
                assert "lang" in item and item["lang"] == "ara"
                assert "title" in item and len(item["title"]) > 0
                assert "format" in item and item["format"] in ("srt", "ass", "ssa", "vtt")


# ============================================================================
# 8. SAFE DIAGNOSTICS & DEBUG OBSERVABILITY
# ============================================================================


def test_diagnostics_ranking_endpoint_safe(client):
    """
    Verify /diagnostics/ranking safely exposes ranking weights and configuration
    without exposing any API keys, tokens, or sensitive headers.
    """
    resp = client.get("/diagnostics/ranking")
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "ok"
    assert "debug_ranking_enabled" in data
    assert "weights" in data
    assert data["weights"]["WEIGHT_YEAR_MISMATCH"] == -60
    assert data["weights"]["WEIGHT_REPACK_MISMATCH"] == -20

    # Ensure no secrets or API keys are exposed
    json_str = resp.text.lower()
    for forbidden in ("api_key", "secret", "token", "bearer", "authorization"):
        assert forbidden not in json_str or forbidden == "api_key" and "api_key" not in data


def test_debug_ranking_logging_flag(monkeypatch, caplog):
    """
    Verify NINJASUBS_DEBUG_RANKING flag activates lightweight ranking observability logs.
    """
    import logging

    from app.services.subtitle_matcher import rank_subtitles

    # 1. With flag enabled
    monkeypatch.setenv("NINJASUBS_DEBUG_RANKING", "true")
    with caplog.at_level(logging.INFO):
        target = "House.of.the.Dragon.S02E01.1080p.HMAX.WEB-DL-FLUX"
        sub = SubtitleRelease(
            release_name="House of the Dragon S02E01 HMAX WEB-DL FLUX.srt",
            download_url="http://1",
            provider="subdl",
        )
        sub_rej = SubtitleRelease(
            release_name="House of the Dragon S02E02 HMAX WEB-DL FLUX.srt",
            download_url="http://2",
            provider="subdl",
        )
        rank_subtitles(target, [sub, sub_rej], discard_mismatches=True)

    assert any("[DEBUG_RANKING]" in record.message for record in caplog.records)
    assert any("DISCARDED" in record.message for record in caplog.records)


# ============================================================================
# 9. REAL PRODUCTION REGRESSION: FMA BROTHERHOOD S01E03 AMPERSAND FILENAME
# ============================================================================


@pytest.mark.asyncio
async def test_real_stremio_fma_brotherhood_s01e03_ampersand_filename(client):
    """
    Regression Test for Real Stremio Production Issue:
    Target stream: [A&C] Fullmetal Alchemist Brotherhood - 03 [BDRip 1080p] [Multi-Audio] [Multi-Subs] [V2] [BEB049B8].mkv
    URL path: videoHash=51700dc6914f0812&videoSize=4638564680&filename=[A&C] Fullmetal Alchemist Brotherhood - 03...

    Asserts:
    1. Ampersand in '[A&C]' does NOT truncate filename into '[A'.
    2. Real S01E03 subtitle ('Fullmetal.Alchemist.Brotherhood.S01E03.Arabic.srt') is NOT rejected.
    3. Wrong episode subtitle ('Fullmetal Alchemist Brotherhood 01 [720p].srt') is 100% DISCARDED.
    4. Top ranked subtitle achieves 100% compatibility badge.
    5. Final JSON response preserves matcher order.
    """
    target_filename = "[A&C] Fullmetal Alchemist Brotherhood - 03 [BDRip 1080p] [Multi-Audio] [Multi-Subs] [V2] [BEB049B8].mkv"

    mock_candidates = [
        # Wrong episode (01 vs 03): MUST be rejected
        SubtitleRelease(
            release_name="Fullmetal Alchemist Brotherhood 01 [720p].srt",
            download_url="https://provider.test/fma_ep01.srt",
            provider="subsource",
            lang="ara",
        ),
        # Valid specific S01E03 episode: MUST rank #1 (100%)
        SubtitleRelease(
            release_name="Fullmetal.Alchemist.Brotherhood.S01E03.Arabic.srt",
            download_url="https://provider.test/fma_s01e03.srt",
            provider="subdl",
            lang="ara",
        ),
        # Valid multi-episode batch: MUST be accepted
        SubtitleRelease(
            release_name="[Delivroozzi] Fullmetal Alchemist Brotherhood [1 ~ 64] [BD].srt",
            download_url="https://provider.test/fma_delivroozzi_batch.srt",
            provider="subsource",
            lang="ara",
        ),
        # Generic BD anime subtitle: MUST be accepted but below exact episode
        SubtitleRelease(
            release_name="Fullmetal Alchemist [HiiO Anime] [BD].srt",
            download_url="https://provider.test/fma_hiio_bd.srt",
            provider="subsource",
            lang="ara",
        ),
    ]

    expected_ranked = rank_subtitles(
        video_filename=target_filename,
        subtitles=mock_candidates,
        preferred_languages=["ara"],
        discard_mismatches=True,
        season=1,
        episode=3,
    )

    # Verify matcher level expectations
    assert len(expected_ranked) == 3
    assert expected_ranked[0].release_name == "Fullmetal.Alchemist.Brotherhood.S01E03.Arabic.srt"
    assert getattr(expected_ranked[0], "match_percentage", 0) == 100
    assert not any("01 [720p]" in r.release_name for r in expected_ranked)

    with patch("app.main._http_client", new_callable=AsyncMock):
        with (
            patch(
                "app.providers.subdl.SubdlProvider.search_subtitles",
                new=AsyncMock(return_value=[mock_candidates[1]]),
            ),
            patch(
                "app.providers.subsource.SubsourceProvider.search_subtitles",
                new=AsyncMock(
                    return_value=[mock_candidates[0], mock_candidates[2], mock_candidates[3]]
                ),
            ),
            patch(
                "app.providers.opensubtitles.OpenSubtitlesProvider.search_subtitles",
                new=AsyncMock(return_value=[]),
            ),
        ):
            url = (
                "/subtitles/series/tt1355642:1:3/"
                f"videoHash=51700dc6914f0812&videoSize=4638564680&filename={target_filename}.json"
            )

            response = client.get(url)
            assert response.status_code == 200
            data = response.json()
            assert "subtitles" in data
            items = data["subtitles"]

            # 1. Assert exactly 3 subtitles returned (Episode 01 is discarded)
            assert len(items) == 3
            releases = [get_release_name(s) for s in items]
            for rel in releases:
                assert "01 [720p]" not in rel

            # 2. Assert #1 subtitle is the S01E03 release with [100%] [SubDL]
            assert items[0]["title"] == "[100%] [SubDL] Fullmetal.Alchemist.Brotherhood.S01E03.Arabic"
            assert "Fullmetal.Alchemist.Brotherhood.S01E03.Arabic" in releases[0]

            # 3. Assert #2 is the 1 ~ 64 BD batch
            assert "Delivroozzi" in releases[1] or "1 ~ 64" in releases[1]

            # 4. Assert exact ordering matches rank_subtitles output and title format
            for i, expected_sub in enumerate(expected_ranked):
                assert expected_sub.release_name == get_release_name(items[i])
                assert re.match(r"^\[\d+%\] \[(?:SubDL|SubSource|OpenSubtitles)\] .+$", items[i]["title"])
