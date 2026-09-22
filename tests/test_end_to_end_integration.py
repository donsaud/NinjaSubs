"""
End-to-End Integration Verification Test Suite for NinjaSubs Stremio Addon.

Verifies the complete request path from the FastAPI HTTP subtitle endpoints
to the final JSON response delivered to Stremio:
1. TV series matching & ranking order preservation (House of the Dragon S02E01).
2. Movie matching, edition conflicts, and source compatibility (Dune Part Two).
3. Anime absolute episode matching, multi-episode range, and false positive protection (One Piece).
4. Exact binary hash priority.
5. Strict preferred language grouping without conflating language with compatibility score.
6. Hearing Impaired (SDH) filtering on/off behavior.
7. Strict Stremio Addon v3 response contract (id, url, lang, title, format).
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
# 1. TV INTEGRATION: HOUSE OF THE DRAGON S02E01
# ============================================================================


@pytest.mark.asyncio
async def test_e2e_tv_house_of_the_dragon_ranking_and_rejection(client):
    """
    Target: House.of.the.Dragon.S02E01.1080p.HMAX.WEB-DL.DDP5.1.Atmos.H.264-FLUX
    Candidates:
      A: House of the Dragon S02E01 HMAX WEB-DL FLUX Arabic
      B: House of the Dragon S02E01 AMZN WEB-DL NTb Arabic
      C: House of the Dragon S02E02 HMAX WEB-DL Arabic (Wrong episode)
      D: House of the Dragon S02E01 HMAX WEBRip Arabic (Family source)
      E: House of the Dragon S02E01 HMAX WEB-DL FLUX English (Secondary language)

    Requirements:
      - C (Episode 2) must NOT appear in the response.
      - A must rank above B (exact scene group + platform vs different).
      - D must remain compatible but rank below A (WEB-DL vs WEBRip).
      - Arabic candidates (A, B, D) must remain ahead of English (E).
      - The final response array order MUST equal the matcher order.
    """
    mock_candidates = [
        SubtitleRelease(
            release_name="House of the Dragon S02E01 HMAX WEB-DL FLUX Arabic.srt",
            download_url="https://provider.test/a.srt",
            provider="subdl",
            lang="ara",
        ),
        SubtitleRelease(
            release_name="House of the Dragon S02E01 AMZN WEB-DL NTb Arabic.srt",
            download_url="https://provider.test/b.srt",
            provider="subsource",
            lang="ara",
        ),
        SubtitleRelease(
            release_name="House of the Dragon S02E02 HMAX WEB-DL Arabic.srt",
            download_url="https://provider.test/c.srt",
            provider="subdl",
            lang="ara",
        ),
        SubtitleRelease(
            release_name="House of the Dragon S02E01 HMAX WEBRip Arabic.srt",
            download_url="https://provider.test/d.srt",
            provider="subdl",
            lang="ara",
        ),
        SubtitleRelease(
            release_name="House of the Dragon S02E01 HMAX WEB-DL FLUX English.srt",
            download_url="https://provider.test/e.srt",
            provider="subdl",
            lang="eng",
        ),
    ]

    target_video = "House.of.the.Dragon.S02E01.1080p.HMAX.WEB-DL.DDP5.1.Atmos.H.264-FLUX"
    cfg = encode_user_config(languages=["ara", "eng"])

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
            # Directly compute expected order from the matcher
            expected_ranked = rank_subtitles(
                video_filename=target_video,
                subtitles=mock_candidates,
                preferred_languages=["ara", "eng"],
                discard_mismatches=True,
            )

            # Query the live FastAPI endpoint with nocache=1 to bypass caching
            extra = f"filename={target_video}"
            resp = client.get(f"/{cfg}/subtitles/series/tt11198330:2:1/{extra}.json?nocache=1")
            assert resp.status_code == 200

            data = resp.json()
            subtitles = data["subtitles"]

            # 1. Episode mismatch C must NOT appear
            returned_releases = [get_release_name(s) for s in subtitles]
            assert not any(
                "S02E02" in t for t in returned_releases
            ), "Candidate C (S02E02) must be hard-rejected and absent"

            # 2. Number of accepted candidates: A, B, D, E = 4
            assert len(subtitles) == len(expected_ranked)
            assert len(subtitles) == 4

            # 3. Final response array order MUST exactly match the matcher order and format
            for i, expected in enumerate(expected_ranked):
                assert expected.release_name == get_release_name(subtitles[i])
                assert re.match(r"^\[\d+%\] \[(?:SubDL|SubSource|OpenSubtitles)\] .+$", subtitles[i]["title"])

            # 4. A (FLUX HMAX) ranks above B (NTb AMZN)
            idx_a = next(
                i for i, s in enumerate(subtitles) if "FLUX" in get_release_name(s) and s["lang"] == "ara"
            )
            idx_b = next(
                i for i, s in enumerate(subtitles) if "NTb" in get_release_name(s) and s["lang"] == "ara"
            )
            assert idx_a < idx_b, "FLUX HMAX must rank ahead of NTb AMZN"

            # 5. D (WEBRip) ranks below A (WEB-DL)
            idx_d = next(i for i, s in enumerate(subtitles) if "WEBRip" in get_release_name(s))
            assert idx_a < idx_d, "Exact WEB-DL source must rank ahead of WEBRip"

            # 6. Arabic candidates remain ahead of English candidates
            ara_indices = [i for i, s in enumerate(subtitles) if s["lang"] == "ara"]
            eng_indices = [i for i, s in enumerate(subtitles) if s["lang"] == "eng"]
            assert max(ara_indices) < min(
                eng_indices
            ), "All Arabic candidates must precede English candidates"


# ============================================================================
# 2. MOVIE INTEGRATION: DUNE PART TWO
# ============================================================================


@pytest.mark.asyncio
async def test_e2e_movie_dune_part_two_ranking_and_edition_rejection(client):
    """
    Target: Dune.Part.Two.2024.2160p.UHD.Remux.DV.Atmos-FLUX
    Candidates:
      M1: Dune Part Two 2024 UHD Remux Arabic
      M2: Dune Part Two 2024 WEB-DL Arabic
      M3: Dune Part Two 2023 BluRay Arabic (Wrong year)
      M4: Dune Part Two Extended Arabic (Conflicting edition)

    Requirements:
      - Same movie/year remains compatible (M1, M2 accepted).
      - WEB-DL (M2) remains compatible with UHD Remux.
      - Wrong year (M3) has lower score than same year.
      - Explicit conflicting edition (M4 Extended) is rejected.
    """
    mock_subs = [
        SubtitleRelease(
            release_name="Dune Part Two 2024 UHD Remux Arabic.srt",
            download_url="https://provider.test/m1.srt",
            provider="subdl",
            lang="ara",
        ),
        SubtitleRelease(
            release_name="Dune Part Two 2024 WEB-DL Arabic.srt",
            download_url="https://provider.test/m2.srt",
            provider="subsource",
            lang="ara",
        ),
        SubtitleRelease(
            release_name="Dune Part Two 2023 BluRay Arabic.srt",
            download_url="https://provider.test/m3.srt",
            provider="subdl",
            lang="ara",
        ),
        SubtitleRelease(
            release_name="Dune Part Two Extended Arabic.srt",
            download_url="https://provider.test/m4.srt",
            provider="subdl",
            lang="ara",
        ),
    ]

    target_video = "Dune.Part.Two.2024.2160p.UHD.Remux.DV.Atmos-FLUX"

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
                "app.providers.opensubtitles.OpenSubtitlesProvider.search_subtitles",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "app.providers.cinemeta.CinemetaClient.get_metadata",
                new=AsyncMock(return_value=None),
            ),
        ):
            expected_ranked = rank_subtitles(
                video_filename=target_video,
                subtitles=mock_subs,
                preferred_languages=["ara"],
                discard_mismatches=True,
            )

            extra = f"filename={target_video}"
            resp = client.get(f"/subtitles/movie/tt15239678/{extra}.json?nocache=1")
            assert resp.status_code == 200

            subtitles = resp.json()["subtitles"]

            # Both M1 (Remux) and M2 (WEB-DL) are present and compatible
            returned_releases = [get_release_name(s) for s in subtitles]
            assert any("Remux" in t for t in returned_releases)
            assert any("WEB-DL" in t for t in returned_releases)

            # M1 (Remux) ranks above M2 (WEB-DL)
            idx_m1 = next(i for i, s in enumerate(subtitles) if "Remux" in get_release_name(s))
            idx_m2 = next(i for i, s in enumerate(subtitles) if "WEB-DL" in get_release_name(s))
            assert idx_m1 < idx_m2, "UHD Remux match must rank above WEB-DL on Remux target"

            # Wrong year M3 (2023) ranks below same-year M1 (2024)
            if any("2023" in t for t in returned_releases):
                idx_m3 = next(i for i, s in enumerate(subtitles) if "2023" in get_release_name(s))
                assert idx_m1 < idx_m3, "Wrong year must rank lower than same-year match"

            # Extended M4 ranks at the very bottom due to heavy edition penalty
            if any("Extended" in t for t in returned_releases):
                idx_m4 = next(i for i, s in enumerate(subtitles) if "Extended" in get_release_name(s))
                assert (
                    idx_m4 == len(subtitles) - 1
                ), "Extended cut subtitle must rank at the bottom on unmarked target"

            # Exact match with matcher output order and title formatting
            assert len(subtitles) == len(expected_ranked)
            for i, expected in enumerate(expected_ranked):
                assert expected.release_name == get_release_name(subtitles[i])
                assert re.match(r"^\[\d+%\] \[(?:SubDL|SubSource|OpenSubtitles)\] .+$", subtitles[i]["title"])

            # Verify explicit conflicting edition rejection when video explicitly specifies Theatrical
            theatrical_video = "Dune.Part.Two.2024.Theatrical.2160p.UHD.Remux.DV.Atmos-FLUX"
            extra_th = f"filename={theatrical_video}"
            resp_th = client.get(f"/subtitles/movie/tt15239678/{extra_th}.json?nocache=1")
            assert resp_th.status_code == 200
            th_releases = [get_release_name(s) for s in resp_th.json()["subtitles"]]
            assert not any(
                "Extended" in t for t in th_releases
            ), "Extended cut must be rejected when video is explicitly Theatrical"


# ============================================================================
# 3. ANIME INTEGRATION: ONE PIECE
# ============================================================================


@pytest.mark.asyncio
async def test_e2e_anime_one_piece_ranking_and_rejection(client):
    """
    Target: One.Piece.1050.1080p.WEB-DL
    Candidates:
      A1: One Piece 1050 Arabic
      A2: One Piece E1050 Arabic
      A3: One Piece 1051 Arabic (Wrong episode)
      A4: One Piece 1049-1050 Arabic (Multi-episode range)

    Requirements:
      - 1050 candidates (A1, A2) are accepted.
      - 1051 (A3) is rejected.
      - 1049-1050 (A4) is accepted.
    """
    mock_subs = [
        SubtitleRelease(
            release_name="One Piece 1050 Arabic.srt",
            download_url="https://provider.test/a1.srt",
            provider="subdl",
            lang="ara",
        ),
        SubtitleRelease(
            release_name="One Piece E1050 Arabic.srt",
            download_url="https://provider.test/a2.srt",
            provider="subsource",
            lang="ara",
        ),
        SubtitleRelease(
            release_name="One Piece 1051 Arabic.srt",
            download_url="https://provider.test/a3.srt",
            provider="subdl",
            lang="ara",
        ),
        SubtitleRelease(
            release_name="One Piece 1049-1050 Arabic.srt",
            download_url="https://provider.test/a4.srt",
            provider="subdl",
            lang="ara",
        ),
    ]

    target_video = "One.Piece.1050.1080p.WEB-DL"

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
                "app.providers.opensubtitles.OpenSubtitlesProvider.search_subtitles",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "app.providers.cinemeta.CinemetaClient.get_metadata",
                new=AsyncMock(return_value=None),
            ),
        ):
            expected_ranked = rank_subtitles(
                video_filename=target_video,
                subtitles=mock_subs,
                preferred_languages=["ara"],
                discard_mismatches=True,
            )

            extra = f"filename={target_video}"
            resp = client.get(f"/subtitles/anime/kitsu:12/{extra}.json?nocache=1")
            assert resp.status_code == 200

            subtitles = resp.json()["subtitles"]
            returned_releases = [get_release_name(s) for s in subtitles]

            # 1051 is rejected
            assert not any(
                "1051" in t for t in returned_releases
            ), "Candidate 1051 must be rejected on episode 1050"

            # 1050 and 1049-1050 are present
            assert any("1050" in t for t in returned_releases)
            assert any("1049-1050" in t for t in returned_releases)

            # Order matches matcher and title is formatted
            assert len(subtitles) == len(expected_ranked)
            for i, expected in enumerate(expected_ranked):
                assert expected.release_name == get_release_name(subtitles[i])
                assert re.match(r"^\[\d+%\] \[(?:SubDL|SubSource|OpenSubtitles)\] .+$", subtitles[i]["title"])


# ============================================================================
# 4. BINARY HASH PRIORITY INTEGRATION
# ============================================================================


@pytest.mark.asyncio
async def test_e2e_exact_hash_priority_over_filename_match(client):
    """Exact binary hash match must rank #1 ahead of even an exact filename match without hash."""
    mock_subs = [
        SubtitleRelease(
            release_name="Movie.2024.1080p.BluRay.x264-FLUX.srt",
            download_url="https://provider.test/file_match.srt",
            provider="subdl",
            lang="ara",
            is_hash_match=False,
        ),
        SubtitleRelease(
            release_name="Movie.Different.Filename.srt",
            download_url="https://provider.test/hash_match.srt",
            provider="opensubtitles",
            lang="ara",
            is_hash_match=True,  # Exact OpenSubtitles / MovieHash match
        ),
    ]

    target_video = "Movie.2024.1080p.BluRay.x264-FLUX.mkv"

    with patch("app.main._http_client", new_callable=AsyncMock):
        with (
            patch(
                "app.providers.subdl.SubdlProvider.search_subtitles",
                new=AsyncMock(return_value=[mock_subs[0]]),
            ),
            patch(
                "app.providers.subsource.SubsourceProvider.search_subtitles",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "app.providers.opensubtitles.OpenSubtitlesProvider.search_subtitles",
                new=AsyncMock(return_value=[mock_subs[1]]),
            ),
            patch(
                "app.providers.cinemeta.CinemetaClient.get_metadata",
                new=AsyncMock(return_value=None),
            ),
        ):
            extra = f"filename={target_video}&videoHash=0123456789abcdef"
            cfg_hash = encode_user_config(enable_opensubtitles=True)
            resp = client.get(f"/{cfg_hash}/subtitles/movie/tt9999999/{extra}.json?nocache=1")
            assert resp.status_code == 200

            subtitles = resp.json()["subtitles"]
            assert len(subtitles) == 2

            # Hash match must be rank 1 (index 0)
            assert subtitles[0]["title"] == "[100%] [OpenSubtitles] Movie.Different.Filename"
            assert get_release_name(subtitles[0]) == "Movie.Different.Filename.srt"
            assert subtitles[1]["title"] == "[100%] [SubDL] Movie.2024.1080p.BluRay.x264-FLUX"
            assert get_release_name(subtitles[1]) == "Movie.2024.1080p.BluRay.x264-FLUX.srt"


# ============================================================================
# 5. LANGUAGE ORDERING WITHOUT CONFLATING WITH COMPATIBILITY SCORE
# ============================================================================


@pytest.mark.asyncio
async def test_e2e_language_preference_grouping(client):
    """
    Preferred languages: Arabic, English.
    Candidates:
      - Mediocre Arabic match (HDTV, 720p)
      - Perfect English match (2160p UHD Remux FLUX)
    Arabic must remain ahead of English without artificial score distortion.
    """
    mock_subs = [
        SubtitleRelease(
            release_name="Show.S01E01.720p.HDTV.Arabic.srt",
            download_url="https://provider.test/ar.srt",
            provider="subdl",
            lang="ara",
        ),
        SubtitleRelease(
            release_name="Show.S01E01.2160p.UHD.Remux.FLUX.English.srt",
            download_url="https://provider.test/en.srt",
            provider="subdl",
            lang="eng",
        ),
    ]

    target_video = "Show.S01E01.2160p.UHD.Remux.FLUX"
    cfg = encode_user_config(languages=["ara", "eng"])

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
                "app.providers.opensubtitles.OpenSubtitlesProvider.search_subtitles",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "app.providers.cinemeta.CinemetaClient.get_metadata",
                new=AsyncMock(return_value=None),
            ),
        ):
            extra = f"filename={target_video}"
            resp = client.get(f"/{cfg}/subtitles/series/tt1234567:1:1/{extra}.json?nocache=1")
            assert resp.status_code == 200

            subtitles = resp.json()["subtitles"]
            assert len(subtitles) == 2

            # Index 0 is Arabic, Index 1 is English
            assert subtitles[0]["lang"] == "ara"
            assert subtitles[1]["lang"] == "eng"
            assert "HDTV" in get_release_name(subtitles[0])
            assert "Remux" in get_release_name(subtitles[1])
            assert re.match(r"^\[\d+%\] \[(?:SubDL|SubSource|OpenSubtitles)\] .+$", subtitles[0]["title"])
            assert re.match(r"^\[\d+%\] \[(?:SubDL|SubSource|OpenSubtitles)\] .+$", subtitles[1]["title"])


# ============================================================================
# 6. HEARING IMPAIRED (SDH) FILTERING
# ============================================================================


@pytest.mark.asyncio
async def test_e2e_sdh_filtering_enabled_vs_disabled(client):
    """
    When Exclude HI is True: SDH candidate is filtered out.
    When Exclude HI is False: SDH candidate is retained.
    """
    mock_subs = [
        SubtitleRelease(
            release_name="Movie.2024.1080p.BluRay.srt",
            download_url="https://provider.test/regular.srt",
            provider="subdl",
            lang="ara",
            hearing_impaired=False,
        ),
        SubtitleRelease(
            release_name="Movie.2024.1080p.BluRay.SDH.srt",
            download_url="https://provider.test/sdh.srt",
            provider="subdl",
            lang="ara",
            hearing_impaired=True,
        ),
    ]

    target_video = "Movie.2024.1080p.BluRay"

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
                "app.providers.opensubtitles.OpenSubtitlesProvider.search_subtitles",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "app.providers.cinemeta.CinemetaClient.get_metadata",
                new=AsyncMock(return_value=None),
            ),
        ):
            extra = f"filename={target_video}"

            # 1. With Exclude HI = True
            cfg_exclude_hi = encode_user_config(languages=["ara"], exclude_hi=True)
            resp_hi = client.get(
                f"/{cfg_exclude_hi}/subtitles/movie/tt1234567/{extra}.json?nocache=1"
            )
            assert resp_hi.status_code == 200
            subs_hi = resp_hi.json()["subtitles"]
            assert len(subs_hi) == 1
            assert not any("SDH" in s["title"] for s in subs_hi)

            # 2. With Exclude HI = False
            cfg_include_hi = encode_user_config(languages=["ara"], exclude_hi=False)
            resp_no_hi = client.get(
                f"/{cfg_include_hi}/subtitles/movie/tt1234567/{extra}.json?nocache=1"
            )
            assert resp_no_hi.status_code == 200
            subs_no_hi = resp_no_hi.json()["subtitles"]
            assert len(subs_no_hi) == 2


# ============================================================================
# 7. STREMIO CONTRACT & FORMAT VERIFICATION
# ============================================================================


@pytest.mark.asyncio
async def test_e2e_stremio_response_contract_fields(client):
    """
    Verify the final HTTP response strictly conforms to Stremio Addon v3:
    Every entry has 'id', 'url', 'lang', 'title', and 'format'.
    """
    mock_subs = [
        SubtitleRelease(
            release_name="Gladiator.2000.1080p.BluRay.x264-CMRG.srt",
            download_url="https://provider.test/gladiator.srt",
            provider="subdl",
            lang="ara",
        ),
        SubtitleRelease(
            release_name="Gladiator.2000.1080p.BluRay.ass",
            download_url="https://provider.test/gladiator.ass",
            provider="subsource",
            lang="ara",
            format="ass",
        ),
    ]

    with patch("app.main._http_client", new_callable=AsyncMock):
        with (
            patch(
                "app.providers.subdl.SubdlProvider.search_subtitles",
                new=AsyncMock(return_value=[mock_subs[0]]),
            ),
            patch(
                "app.providers.subsource.SubsourceProvider.search_subtitles",
                new=AsyncMock(return_value=[mock_subs[1]]),
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
                "/subtitles/movie/tt0172495/filename=Gladiator.2000.1080p.BluRay.x264-CMRG.json?nocache=1"
            )
            assert resp.status_code == 200
            data = resp.json()

            assert "subtitles" in data
            assert len(data["subtitles"]) == 2

            for sub in data["subtitles"]:
                assert "id" in sub and len(sub["id"]) > 0
                assert "url" in sub and sub["url"].startswith("http")
                assert "lang" in sub and sub["lang"] == "ara"
                assert "title" in sub and len(sub["title"]) > 0
                assert "format" in sub and sub["format"] in ("srt", "ass", "ssa", "vtt")
