"""Unit tests for the comprehensive Subtitle Matcher Engine."""

import pytest

from app.models import SubtitleRelease
from app.services.subtitle_matcher import (
    MatchResult,
    calculate_match_score,
    extract_metadata,
    parse_video_metadata,
    rank_subtitles,
)


def test_match_result_compatibility():
    """Verify MatchResult behaves as integer, tuple, and structured object."""
    res = MatchResult(135, 100)
    assert res > 130
    assert res == 135
    assert res.score == 135
    assert res.percentage == 100

    # Tuple unpacking
    score, pct = res
    assert score == 135
    assert pct == 100

    # Indexing
    assert res[0] == 135
    assert res[1] == 100




def test_bluray_vs_webrip_movie_ranking():
    """
    Test Case 2:
    Video: 'Movie.Title.2024.2160p.UHD.BluRay.x265-CMRG'
    Subtitle A: 'Movie Title 2024 BluRay'
    Subtitle B: 'Movie Title 2024 WEBRip'
    -> Subtitle A must rank higher than Subtitle B.
    """
    video = "Movie.Title.2024.2160p.UHD.BluRay.x265-CMRG"
    sub_a = SubtitleRelease(
        release_name="Movie Title 2024 BluRay",
        download_url="http://example.com/sub_a.srt",
        provider="subdl",
    )
    sub_b = SubtitleRelease(
        release_name="Movie Title 2024 WEBRip",
        download_url="http://example.com/sub_b.srt",
        provider="subsource",
    )

    score_a = calculate_match_score(video, sub_a.release_name)
    score_b = calculate_match_score(video, sub_b.release_name)

    assert score_a.score > score_b.score

    ranked = rank_subtitles(video, [sub_b, sub_a])
    assert len(ranked) == 2
    assert ranked[0].release_name == sub_a.release_name
    assert ranked[1].release_name == sub_b.release_name


def test_episode_mismatch_rejection():
    """
    Test Case 3:
    Episode mismatch rejection:
    Video S01E02 vs Sub S01E03 -> Must be discarded (Score < -500).
    """
    video = "Breaking.Bad.S01E02.1080p.WEB-DL-FLUX"
    sub_match = SubtitleRelease(
        release_name="Breaking Bad S01E02 1080p WEB-DL-FLUX",
        download_url="http://example.com/e02.srt",
        provider="subdl",
    )
    sub_mismatch = SubtitleRelease(
        release_name="Breaking Bad S01E03 1080p WEB-DL-FLUX",
        download_url="http://example.com/e03.srt",
        provider="subdl",
    )

    score_mismatch = calculate_match_score(video, sub_mismatch.release_name)
    assert score_mismatch.score < -500, f"Expected score < -500, got {score_mismatch.score}"

    ranked = rank_subtitles(video, [sub_mismatch, sub_match])
    assert len(ranked) == 1
    assert ranked[0].release_name == sub_match.release_name
    assert sub_mismatch not in ranked


def test_extended_cut_vs_theatrical_ranking():
    """
    Verify: 'Movie.Extended.Cut.1080p.BluRay-FLUX' ranks 'Extended' subtitle strictly above 'Theatrical' subtitle.
    An extended cut subtitle desyncs completely on a theatrical cut.
    """
    video = "Movie.Extended.Cut.1080p.BluRay-FLUX"
    sub_extended = SubtitleRelease(
        release_name="Movie.Extended.Cut.1080p.BluRay",
        download_url="http://example.com/ext.srt",
        provider="subdl",
    )
    sub_theatrical = SubtitleRelease(
        release_name="Movie.Theatrical.Cut.1080p.BluRay",
        download_url="http://example.com/th.srt",
        provider="subdl",
    )

    meta_v = extract_metadata(video)
    meta_ext = extract_metadata(sub_extended.release_name)
    meta_th = extract_metadata(sub_theatrical.release_name)

    assert meta_v["edition"] == "EXTENDED"
    assert meta_ext["edition"] == "EXTENDED"
    assert meta_th["edition"] == "THEATRICAL"

    score_ext = calculate_match_score(video, sub_extended.release_name)
    score_th = calculate_match_score(video, sub_theatrical.release_name)

    # Extended matches -> positive bonus; Theatrical conflicts with Extended -> severe penalty (-400)
    assert score_ext.score > 0
    assert score_th.score < -200

    ranked = rank_subtitles(video, [sub_theatrical, sub_extended])
    assert len(ranked) >= 1
    assert ranked[0].release_name == sub_extended.release_name






def test_remux_and_sdh_parsing():
    """Verify Remux and SDH flags extraction."""
    meta_remux = extract_metadata("Oppenheimer.2023.2160p.UHD.Remux.HEVC.TrueHD.Atmos.SDH.mkv")
    assert meta_remux["is_remux"] is True
    assert meta_remux["source"] == "Remux"
    assert meta_remux["is_sdh"] is True

    meta_no_remux = extract_metadata("Oppenheimer.2023.1080p.WEB-DL.x264.mkv")
    assert meta_no_remux["is_remux"] is False
    assert meta_no_remux["is_sdh"] is False


@pytest.mark.parametrize(
    "filename,expected_service",
    [
        ("Show.S01E01.1080p.BINGE.WEB-DL.DDP5.1-FLUX", "BINGE"),
        ("Series.S02E03.1080p.AMCP.WEB-DL.AAC2.0-NTb", "AMCP"),
        ("Anime.S01E01.1080p.HIDIVE.WEB-DL.AAC2.0-Erai", "HIDIVE"),
        ("Show.S01E01.1080p.AMC.WEB-DL", "AMC"),
        ("Show.S01E01.1080p.FOXT.WEB-DL", "FOXT"),
        ("Show.S01E01.1080p.NOW.WEB-DL", "NOW"),
        ("Show.S01E01.1080p.SKY.WEB-DL", "SKY"),
        ("Show.S01E01.1080p.CPLUS.WEB-DL", "CPLUS"),
        ("Show.S01E01.1080p.VIAPLAY.WEB-DL", "VIAPLAY"),
        ("Show.S01E01.1080p.TVING.WEB-DL", "TVING"),
        ("Show.S01E01.1080p.WAVVE.WEB-DL", "WAVVE"),
        ("Show.S01E01.1080p.IQIYI.WEB-DL", "IQIYI"),
        ("Show.S01E01.1080p.WETV.WEB-DL", "WETV"),
        ("Show.S01E01.1080p.BILIBILI.WEB-DL", "BILIBILI"),
        ("Show.S01E01.1080p.VIKI.WEB-DL", "VIKI"),
        ("Show.S01E01.1080p.TUBI.WEB-DL", "TUBI"),
        ("Show.S01E01.1080p.PLUTO.WEB-DL", "PLUTO"),
        ("Show.S01E01.1080p.FREEVEE.WEB-DL", "FREEVEE"),
        ("Show.S01E01.1080p.DISC.WEB-DL", "DISC"),
        ("Show.S01E01.1080p.HIST.WEB-DL", "HIST"),
        ("Show.S01E01.1080p.SYFY.WEB-DL", "SYFY"),
        ("Show.S01E01.1080p.CW.WEB-DL", "CW"),
        ("Show.S01E01.1080p.FX.WEB-DL", "FX"),
        ("Show.S01E01.1080p.PBS.WEB-DL", "PBS"),
    ],
)
def test_expanded_streaming_services(filename, expected_service):
    """Verify newly added global, regional, Asian/Anime, and FAST platforms."""
    meta = parse_video_metadata(filename)
    assert meta["service"] == expected_service
    assert meta["source"] == "WEB-DL"


def test_20_plus_streaming_services_detected():
    """Verify recognition of classic 20+ distinct streaming platforms."""
    expected_services = [
        ("Show.S01E01.HMAX.WEB-DL", "HMAX"),
        ("Show.S01E01.AMZN.WEB-DL", "AMZN"),
        ("Show.S01E01.DSNP.WEB-DL", "DSNP"),
        ("Show.S01E01.NF.WEB-DL", "NF"),
        ("Show.S01E01.ATVP.WEB-DL", "ATVP"),
        ("Show.S01E01.HULU.WEB-DL", "HULU"),
        ("Show.S01E01.PCOK.WEB-DL", "PCOK"),
        ("Show.S01E01.PMTP.WEB-DL", "PMTP"),
        ("Show.S01E01.CRAV.WEB-DL", "CRAV"),
        ("Show.S01E01.STAN.WEB-DL", "STAN"),
        ("Show.S01E01.CR.WEB-DL", "CR"),
        ("Show.S01E01.IT.WEB-DL", "IT"),
        ("Show.S01E01.SHO.WEB-DL", "SHO"),
        ("Show.S01E01.STARZ.WEB-DL", "STARZ"),
        ("Show.S01E01.BBC.WEB-DL", "BBC"),
        ("Show.S01E01.CBC.WEB-DL", "CBC"),
        ("Show.S01E01.ALL4.WEB-DL", "ALL4"),
        ("Show.S01E01.ROKU.WEB-DL", "ROKU"),
        ("Show.S01E01.WOW.WEB-DL", "WOW"),
        ("Show.S01E01.BCORE.WEB-DL", "BCORE"),
        ("Show.S01E01.MGM+.WEB-DL", "MGM+"),
        ("Show.S01E01.DCU.WEB-DL", "DCU"),
    ]

    for filename, expected_key in expected_services:
        meta = parse_video_metadata(filename)
        assert (
            meta["service"] == expected_key
        ), f"Failed for {filename}: expected {expected_key}, got {meta['service']}"
        assert meta["source"] == "WEB-DL"


def test_codecs_and_audio_metadata_parsing():
    """Verify resolution, codec, and audio attribute extraction."""
    meta = parse_video_metadata(
        "Dune.Part.Two.2024.2160p.UHD.BluRay.Remux.HEVC.DV.TrueHD.Atmos.7.1-FraMeSToR"
    )
    assert meta["resolution"] == "2160p"
    assert meta["source"] == "Remux"
    assert meta["codec"] in ("x265", "DV")
    assert meta["audio"] in ("TrueHD", "Atmos")
    assert meta["group"] == "FraMeSToR"
    assert meta["year"] == 2024


def test_multi_language_ranking():
    """Verify multi-language grouping sorts primary language first, then secondary."""
    video = "Movie.2024.1080p.BluRay-FLUX"
    sub_ar_best = SubtitleRelease(
        release_name="Movie.2024.1080p.BluRay-FLUX",
        download_url="http://a",
        provider="subdl",
        lang="ara",
    )
    sub_ar_low = SubtitleRelease(
        release_name="Movie.2024.CAM", download_url="http://b", provider="subdl", lang="ara"
    )
    sub_en_best = SubtitleRelease(
        release_name="Movie.2024.1080p.BluRay-FLUX",
        download_url="http://c",
        provider="subsource",
        lang="eng",
    )
    sub_en_low = SubtitleRelease(
        release_name="Movie.2024.CAM", download_url="http://d", provider="subsource", lang="eng"
    )

    ranked = rank_subtitles(
        video,
        [sub_en_low, sub_ar_low, sub_en_best, sub_ar_best],
        preferred_languages=["ara", "eng"],
    )
    assert len(ranked) == 4
    assert ranked[0].lang == "ara" and ranked[0].release_name == sub_ar_best.release_name
    assert ranked[1] is sub_en_best
    assert ranked[2] is sub_ar_low
    assert ranked[3].lang == "eng" and ranked[3].release_name == sub_en_low.release_name




def test_se_ep_pattern_extraction():
    """Verify SeXX / EpXX patterns accurately extract season and episode numbers."""
    meta = extract_metadata("Show.Title.Se05.Ep02.720p.HDTV.x264.srt")
    assert meta["season"] == 5
    assert meta["episode"] == 2

    meta2 = extract_metadata("Dexter.Se08.Ep12.1080p.mkv")
    assert meta2["season"] == 8
    assert meta2["episode"] == 12


def test_season_mismatch_s01e02_vs_se05ep02():
    """
    Severe season mismatch:
    Video S01E02 vs Subtitle Se05.Ep02 must strictly return score <= -500 (0%) and be discarded.
    """
    video = "Show.Title.S01E02.1080p.WEB-DL-FLUX"
    sub_se05 = SubtitleRelease(
        release_name="Show.Title.Se05.Ep02.720p.HDTV.x264.srt",
        download_url="http://example.com/se05.srt",
        provider="subdl",
    )
    sub_s01 = SubtitleRelease(
        release_name="Show.Title.S01E02.1080p.WEB-DL.srt",
        download_url="http://example.com/s01.srt",
        provider="subdl",
    )

    res_mismatch = calculate_match_score(video, sub_se05.release_name)
    assert res_mismatch.score <= -500, f"Expected score <= -500, got {res_mismatch.score}"
    assert res_mismatch.percentage == 0

    ranked = rank_subtitles(video, [sub_se05, sub_s01])
    assert len(ranked) == 1
    assert ranked[0].release_name == sub_s01.release_name
    assert sub_se05 not in ranked
