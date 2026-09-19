"""
Unit and integration tests for:
1. Anime Absolute Episode Numbering (2-4 digit detection & matching).
2. Informative Subtitle Display Badges ([NinjaSubs] specs and verified hash).
3. Fallback Graceful Ranking when video filename is missing.
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.cache import cache_manager
from app.models import SubtitleRelease
from app.services.aggregator import format_informative_badge
from app.services.subtitle_matcher import (
    calculate_match_score,
    extract_leading_bracket_group,
    extract_metadata,
    is_anime_content,
    rank_subtitles,
)

# =======================================================
# 1. ANIME ABSOLUTE NUMBERING EXTRACTION & MATCHING TESTS
# =======================================================


@pytest.mark.parametrize(
    "filename,expected_abs_ep",
    [
        ("One Piece 1050 1080p.mkv", 1050),
        ("One.Piece.E1050.1080p.CR.WEB-DL", 1050),
        ("Bleach - 24 [1080p][HEVC].mkv", 24),
        ("Show - E15.mkv", 15),
        ("Jujutsu Kaisen #24 (1080p).mkv", 24),
        ("[Erai-raws] Jujutsu Kaisen 2nd Season - 23 [1080p][HEVC].mkv", 23),
        ("Naruto Shippuden 500 [720p].mp4", 500),
        ("Fullmetal Alchemist Brotherhood - 01", 1),
        ("Brotherhood 01", 1),
        ("Brotherhood - 01", 1),
        ("Brotherhood - 1", 1),
        ("Fullmetal.Alchemist.Brotherhood.01", 1),
    ],
)
def test_anime_absolute_episode_extraction(filename, expected_abs_ep):
    """Verify standalone 2-4 digit episode numbers are accurately extracted."""
    meta = extract_metadata(filename)
    assert meta["absolute_episode"] == expected_abs_ep


def test_anime_absolute_numbering_ignores_year_and_resolution():
    """Ensure release year and standard video resolutions are not confused for anime episodes."""
    meta = extract_metadata("One Piece 1999 - 1050 1080p WEB-DL.mkv")
    assert meta["year"] == 1999
    assert meta["resolution"] == "1080p"
    assert meta["absolute_episode"] == 1050


def test_anime_episode_matching_and_mismatch_rejection():
    """
    Candidate with matching absolute episode earns bonus (+15).
    Candidate with differing absolute episode is strictly rejected (-1000).
    """
    video = "One.Piece.1050.1080p.CR.WEB-DL-Subs.mkv"
    sub_exact = SubtitleRelease(
        release_name="One Piece - 1050.srt",
        download_url="http://sub1",
        provider="subdl",
    )
    sub_mismatch = SubtitleRelease(
        release_name="One Piece - 1049.srt",
        download_url="http://sub2",
        provider="subdl",
    )

    res_match = calculate_match_score(video, sub_exact.release_name)
    res_mismatch = calculate_match_score(video, sub_mismatch.release_name)

    assert res_match.score > 0
    assert res_mismatch.score <= -500

    ranked = rank_subtitles(video, [sub_mismatch, sub_exact])
    # The mismatched episode should be discarded or ranked below
    assert len(ranked) == 1
    assert ranked[0].release_name == sub_exact.release_name


def test_anime_cross_format_episode_matching():
    """Test matching where video uses E24 and subtitle uses #24 or absolute 24."""
    video = "Anime.Title.E24.1080p.mkv"
    sub = "Anime Title 24 Arabic.srt"
    res = calculate_match_score(video, sub)
    assert res.score > 0


def test_anime_dominant_episode_bonus():
    """Matching anime episode awards dominant bonus, reaching score >= 135 and 100%."""
    video = "Fullmetal Alchemist Brotherhood - 01"
    sub = "Fullmetal Alchemist Brotherhood 01.srt"
    res = calculate_match_score(video, sub)
    assert res.score >= 135
    assert res.percentage == 100


def test_anime_exact_episode_match_never_below_95():
    """Ensure an exact anime episode match never yields below 95% even with noisy title differences."""
    video = "One Piece 1050 1080p.mkv"
    sub = "Totally Different Title 1050.srt"
    res = calculate_match_score(video, sub)
    assert res.percentage >= 95


def test_anime_single_episode_rejects_ova_and_complete():
    """Video is single episode, but subtitle is OVA or Complete -> strict penalty (-500) and discarded."""
    video = "Fullmetal Alchemist Brotherhood - 01"
    sub_exact = SubtitleRelease(
        release_name="Fullmetal Alchemist Brotherhood 01.srt",
        download_url="http://1",
        provider="subdl",
    )
    sub_ova = SubtitleRelease(
        release_name="Fullmetal Alchemist Brotherhood OVA 01.srt",
        download_url="http://2",
        provider="subdl",
    )
    sub_comp = SubtitleRelease(
        release_name="Fullmetal Alchemist Brotherhood Complete Batch.srt",
        download_url="http://3",
        provider="subdl",
    )

    score_ova = calculate_match_score(video, sub_ova.release_name)
    score_comp = calculate_match_score(video, sub_comp.release_name)

    assert score_ova.score <= -500
    assert score_comp.score <= -500

    ranked = rank_subtitles(video, [sub_ova, sub_comp, sub_exact])
    assert len(ranked) == 1
    assert ranked[0].release_name == sub_exact.release_name


def test_anime_exact_match_boost_to_100_percent():
    """Matching title + matching anime episode + matching source yields score >= 130 and 100%."""
    video = "[Kaylith] Fullmetal Alchemist Brotherhood - 01 [BDRip 1080p]"
    sub = "Fullmetal Alchemist Brotherhood - 01 [BDRip 1080p].srt"
    res = calculate_match_score(video, sub)
    assert res.score >= 130
    assert res.percentage == 100


# =======================================================
# 2. INFORMATIVE SUBTITLE DISPLAY BADGES TESTS
# =======================================================


def test_badge_verified_hash_match():
    """Verify cryptographic hash match formats as '[100%] [OpenSubtitles] {Filename}' without prefixes."""
    sub = SubtitleRelease(
        release_name="Arbitrary.Release.Name.mkv",
        download_url="http://hash.srt",
        provider="opensubtitles",
        is_hash_match=True,
    )
    badge_ar = format_informative_badge(
        sub, display_score=100, lang_name="Arabic", source_tag="OpenSubtitles"
    )
    assert badge_ar == "[100%] [OpenSubtitles] Arbitrary.Release.Name"

    badge_en = format_informative_badge(
        sub, display_score=100, lang_name="English", source_tag="OpenSubtitles"
    )
    assert badge_en == "[100%] [OpenSubtitles] Arbitrary.Release.Name"


def test_badge_high_accuracy_group_and_source():
    """Verify high match formats as '[100%] [SubDL] Gladiator.2000.1080p.BluRay.x264-FLUX'."""
    sub = SubtitleRelease(
        release_name="Gladiator.2000.1080p.BluRay.x264-FLUX.srt",
        download_url="http://sub.srt",
        provider="subdl",
        score=150,
        match_percentage=100,
    )
    badge = format_informative_badge(sub, display_score=100, lang_name="Arabic", source_tag="Subdl")
    assert badge == "[100%] [SubDL] Gladiator.2000.1080p.BluRay.x264-FLUX"


def test_badge_high_accuracy_group_only_or_source_only():
    """Verify full natural release name is preserved without shortening or prefixes."""
    # Group only
    sub_group = SubtitleRelease(
        release_name="Movie.Title.2024-FLUX.srt",
        download_url="http://sub1.srt",
        provider="subdl",
        score=140,
        match_percentage=100,
    )
    badge1 = format_informative_badge(
        sub_group, display_score=100, lang_name="Arabic", source_tag="Subdl"
    )
    assert badge1 == "[100%] [SubDL] Movie.Title.2024-FLUX"

    # Service/Source only
    sub_src = SubtitleRelease(
        release_name="Movie.Title.2024.1080p.HMAX.WEB-DL.srt",
        download_url="http://sub2.srt",
        provider="subdl",
        score=140,
        match_percentage=100,
    )
    badge2 = format_informative_badge(
        sub_src, display_score=100, lang_name="Arabic", source_tag="Subdl"
    )
    assert badge2 == "[100%] [SubDL] Movie.Title.2024.1080p.HMAX.WEB-DL"


def test_badge_fallback_when_low_score():
    """Verify score < 130 and pct < 95 formats as '[{score}%] [{source_tag}] {clean_name}'."""
    sub = SubtitleRelease(
        release_name="Movie.Title.2024.720p.HDTV.srt",
        download_url="http://sub3.srt",
        provider="subsource",
        score=45,
        match_percentage=45,
    )
    badge = format_informative_badge(
        sub, display_score=45, lang_name="Arabic", source_tag="SubSource"
    )
    assert badge == "[45%] [SubSource] Movie.Title.2024.720p.HDTV"


def test_clean_subtitle_display_name_artifacts():
    """Verify stripping of provider hashes and orphan brackets while preserving release filename."""
    from app.services.aggregator import clean_subtitle_display_name

    assert (
        clean_subtitle_display_name("The.Sopranos.S01E01.1080p.WEB-DL-FLUX_6e4b079e36dd0457.srt")
        == "The.Sopranos.S01E01.1080p.WEB-DL-FLUX"
    )
    assert (
        clean_subtitle_display_name("NRN] Breaking.Bad.S01E01.720p.HDTV.srt")
        == "Breaking.Bad.S01E01.720p.HDTV"
    )
    assert (
        clean_subtitle_display_name("Movie.Title.2024.1080p.BluRay]")
        == "Movie.Title.2024.1080p.BluRay"
    )
    assert (
        clean_subtitle_display_name("[Kaylith] Fullmetal Alchemist Brotherhood - 01 [1080p].srt")
        == "Fullmetal Alchemist Brotherhood - 01 [1080p]"
    )
    assert (
        clean_subtitle_display_name("[MSRT Fansub] Bleach - 01.srt")
        == "Bleach - 01"
    )
    assert (
        clean_subtitle_display_name("[Shiniori-Raws] Monster 01-74 (BD 720p x264 ALAC).srt")
        == "Monster 01-74 (BD 720p x264 ALAC)"
    )
    assert (
        clean_subtitle_display_name("Gladiator.2000.1080p.BluRay.x264-FLUX")
        == "Gladiator.2000.1080p.BluRay.x264-FLUX"
    )


def test_badge_anime_release_group_removal():
    """Verify anime release group tags like [MSRT Fansub] are stripped while keeping [pct%] [Provider] {filename}."""
    sub = SubtitleRelease(
        release_name="[MSRT Fansub] Bleach - 01.srt",
        download_url="http://sub.srt",
        provider="subdl",
        score=100,
        match_percentage=100,
    )
    badge = format_informative_badge(
        sub, display_score=100, lang_name="Arabic", source_tag="SubDL"
    )
    assert badge == "[100%] [SubDL] Bleach - 01"

    sub_shiniori = SubtitleRelease(
        release_name="[Shiniori-Raws] Monster 01-74.srt",
        download_url="http://sub2.srt",
        provider="subdl",
        score=95,
        match_percentage=95,
    )
    badge2 = format_informative_badge(
        sub_shiniori, display_score=95, lang_name="Arabic", source_tag="SubDL"
    )
    assert badge2 == "[95%] [SubDL] Monster 01-74"


# =======================================================
# 3. FALLBACK GRACEFUL RANKING TESTS
# =======================================================


def test_fallback_ranking_when_video_filename_none():
    """
    When video_filename is None or missing, all valid subtitles must be retained
    and ordered by source tier popularity score without dropping any valid subtitles.
    """
    subs = [
        SubtitleRelease(
            release_name="Movie.2024.720p.HDTV.srt", download_url="http://hdtv", provider="subdl"
        ),
        SubtitleRelease(
            release_name="Movie.2024.1080p.BluRay.srt",
            download_url="http://bluray",
            provider="subdl",
        ),
        SubtitleRelease(
            release_name="Movie.2024.1080p.WEB-DL.srt",
            download_url="http://webdl",
            provider="subsource",
        ),
        SubtitleRelease(
            release_name="Movie.2024.CAM.srt", download_url="http://cam", provider="subdl"
        ),
    ]

    ranked = rank_subtitles(None, subs)
    # All 4 items must be retained
    assert len(ranked) == 4
    # BluRay (tier 55) > WEB-DL (tier 50) > HDTV (tier 35) > CAM (tier 10)
    assert "BluRay" in ranked[0].release_name
    assert "WEB-DL" in ranked[1].release_name
    assert "HDTV" in ranked[2].release_name
    assert "CAM" in ranked[3].release_name


# =======================================================
# 4. END-TO-END STREMIO SUBTITLE ENDPOINT TEST
# =======================================================


@pytest.mark.asyncio
async def test_endpoint_verified_hash_and_informative_badge(client):
    """
    Verify end-to-end that:
    1. A MovieHash match yields '[NinjaSubs] Arabic • 100% (Verified Hash)' in title.
    2. A high-sync release yields '[NinjaSubs] Arabic • 100% ({specs})'.
    """
    hash_sub = SubtitleRelease(
        release_name="Shawshank.1994.mkv",
        download_url="/sub/hash.srt",
        provider="opensubtitles",
        lang="ara",
        is_hash_match=True,
    )
    stream_sub = SubtitleRelease(
        release_name="Shawshank.1994.1080p.BluRay.x264-FLUX.srt",
        download_url="/sub/flux.zip",
        provider="subdl",
        lang="ara",
    )

    with (
        patch("app.main._http_client", new_callable=AsyncMock),
        patch(
            "app.providers.subdl.SubdlProvider.search_subtitles",
            new=AsyncMock(return_value=[stream_sub]),
        ),
        patch(
            "app.providers.subsource.SubsourceProvider.search_subtitles",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "app.providers.opensubtitles.OpenSubtitlesProvider.search_subtitles",
            new=AsyncMock(return_value=[hash_sub]),
        ),
        patch(
            "app.providers.cinemeta.CinemetaClient.get_metadata", new=AsyncMock(return_value=None)
        ),
    ):
        extra = "videoHash=abcdef1234567890&videoSize=2048576&filename=Shawshank.1994.1080p.BluRay.x264-FLUX.mkv"
        resp = client.get(f"/subtitles/movie/tt0111161/{extra}.json")
        assert resp.status_code == 200

        data = resp.json()
        subs = data["subtitles"]
        assert len(subs) == 2

        # Hash match has absolute priority (+500 points) and ranks #1
        assert subs[0]["title"] == "[100%] [OpenSubtitles] Shawshank.1994"
        # Release match ranks #2 with formatted label
        assert subs[1]["title"] == "[100%] [SubDL] Shawshank.1994.1080p.BluRay.x264-FLUX"


def test_clean_final_label_hex_hashes():
    """Verify clean_final_label strictly removes trailing hex IDs/hashes and extensions."""
    from app.services.aggregator import clean_final_label

    assert (
        clean_final_label("[100%] [Subdl] Dexter.2006.S08.1080p.BluRay.x265-ImE_b5f1bc3a5ec49533")
        == "[100%] [Subdl] Dexter.2006.S08.1080p.BluRay.x265-ImE"
    )
    assert (
        clean_final_label(
            "[100%] [Subdl] Dexter.2006.S08.1080p.BluRay.x265-ImE_01703060d61e07a7.srt"
        )
        == "[100%] [Subdl] Dexter.2006.S08.1080p.BluRay.x265-ImE"
    )
    assert (
        clean_final_label("[95%] [SubSource] Show.S01E01.1080p.WEB-DL-NTb_fd3cebb32c50021a")
        == "[95%] [SubSource] Show.S01E01.1080p.WEB-DL-NTb"
    )
    assert (
        clean_final_label("[100%] [OpenSubtitles] Movie.Title.2024.1080p.BluRay.x264-FLUX")
        == "[100%] [OpenSubtitles] Movie.Title.2024.1080p.BluRay.x264-FLUX"
    )
    assert (
        clean_final_label("Dexter.2006.S08.1080p.BluRay.x265-ImE_b5f1bc3a5ec49533.srt")
        == "Dexter.2006.S08.1080p.BluRay.x265-ImE"
    )
    assert clean_final_label("") == ""
    assert clean_final_label(None) == ""


# =======================================================
# 5. ANIME ISOLATED LOGIC & GUARD TESTS
# =======================================================


def test_is_anime_content_guard_accuracy():
    """Verify is_anime_content accurately distinguishes anime from regular movies and series."""
    movies_and_series = [
        "Dune.Part.Two.2024.2160p.UHD.Remux.DV.Atmos-FLUX.mkv",
        "Gladiator.2000.Extended.2160p.BluRay.x265-CMRG.mkv",
        "The.Sopranos.S01E01.1080p.WEB-DL.H.264-NTb.mkv",
        "Breaking.Bad.S01E02.1080p.WEB-DL-FLUX",
        "House.of.the.Dragon.S02E01.1080p.HMAX.WEB-DL.DDP5.1.Atmos.H.264-FLUX",
        "Movie.Title.2024.2160p.UHD.BluRay.x265-CMRG",
        "Iron.Man.3.2013.1080p.BluRay.x264-SPARKS",
        "Toy.Story.2.1999.1080p.BluRay",
        "Blade.Runner.2049.2017.2160p.UHD",
        "1917.2019.1080p.BluRay",
    ]
    for non_anime in movies_and_series:
        assert is_anime_content(non_anime, non_anime) is False, f"Expected False for {non_anime}"

    anime_samples = [
        "One Piece 1050 1080p.mkv",
        "One.Piece.E1050.1080p.CR.WEB-DL",
        "Bleach - 24 [1080p][HEVC].mkv",
        "Show - E15.mkv",
        "Jujutsu Kaisen #24 (1080p).mkv",
        "[Erai-raws] Jujutsu Kaisen 2nd Season - 23 [1080p][HEVC].mkv",
        "Naruto Shippuden 500 [720p].mp4",
        "Fullmetal Alchemist Brotherhood - 01",
        "Brotherhood 01",
        "Brotherhood - 01",
        "Brotherhood - 1",
        "Fullmetal.Alchemist.Brotherhood.01",
        "[Kaylith] Fullmetal Alchemist Brotherhood - 01 [BDRip 1080p]",
        "Anime.Title.E24.1080p.mkv",
        "Fullmetal Alchemist Brotherhood OVA 01.srt",
    ]
    for anime in anime_samples:
        assert is_anime_content(anime, anime) is True, f"Expected True for {anime}"


def test_anime_implicit_s01_equivalence():
    """Rule a: Standalone episode numbers (e.g. - 01) implicitly map to Season 1 when matched against S01Exx."""
    video = "[SubsPlease] Frieren - Beyond Journeys End - 01 (1080p).mkv"
    sub_s01 = "Frieren.Beyond.Journeys.End.S01E01.1080p.CR.WEB-DL.srt"
    sub_s02 = "Frieren.Beyond.Journeys.End.S02E01.1080p.CR.WEB-DL.srt"

    res_match = calculate_match_score(video, sub_s01)
    res_mismatch = calculate_match_score(video, sub_s02)

    assert res_match.percentage == 100
    assert res_match.score >= 135
    assert res_mismatch.score <= -500


def test_anime_leading_bracket_group_matching():
    """Rule b: Compare release groups from leading [...] tags, awarding release group bonus."""
    video = "[SubsPlease] Jujutsu Kaisen - 23 (1080p).mkv"
    sub_matching_group = "[SubsPlease] Jujutsu Kaisen - 23.srt"
    sub_diff_group = "[Erai-raws] Jujutsu Kaisen - 23.srt"

    res_match = calculate_match_score(video, sub_matching_group)
    res_diff = calculate_match_score(video, sub_diff_group)

    assert extract_leading_bracket_group(video) == "SubsPlease"
    assert extract_leading_bracket_group(sub_matching_group) == "SubsPlease"
    assert extract_leading_bracket_group(sub_diff_group) == "Erai-raws"

    assert res_match.score > res_diff.score
    assert res_match.percentage == 100
    assert res_diff.percentage < 100


def test_clean_anime_episode_prioritized_over_fansub_tag():
    """
    Ensure clean/generic episode subtitles matching the exact episode number
    are prioritized as 100% when source matches (BluRay/BDRip), and outrank
    subtitles carrying unshared fansub group tags.
    """
    s_clean_bd = "Fullmetal Alchemist Brotherhood - 02 [BDRip 1080p].srt"
    s_clean = "Fullmetal Alchemist Brotherhood - 02.srt"
    s_darkdream = "[DarkDream] Fullmetal Alchemist Brotherhood - 02 - The First Day.srt"

    # Scenario 1: Video has unshared fansub group [BlackRabbit]
    v_blackrabbit = "[BlackRabbit] Fullmetal Alchemist Brotherhood - 02 [BDRip 1080p]"
    res_br_clean_bd = calculate_match_score(v_blackrabbit, s_clean_bd)
    res_br_clean = calculate_match_score(v_blackrabbit, s_clean)
    res_br_darkdream = calculate_match_score(v_blackrabbit, s_darkdream)

    assert res_br_clean_bd.percentage == 100
    assert res_br_clean.percentage == 100
    assert res_br_darkdream.percentage < 100
    assert res_br_clean_bd.score > res_br_darkdream.score
    assert res_br_clean.score > res_br_darkdream.score

    # Scenario 2: Video is generic BluRay without fansub group
    v_generic = "Fullmetal Alchemist Brotherhood - 02 [BDRip 1080p]"
    res_gen_clean_bd = calculate_match_score(v_generic, s_clean_bd)
    res_gen_clean = calculate_match_score(v_generic, s_clean)
    res_gen_darkdream = calculate_match_score(v_generic, s_darkdream)

    assert res_gen_clean_bd.percentage == 100
    assert res_gen_clean.percentage == 100
    assert res_gen_darkdream.percentage < 100
    assert res_gen_clean_bd.score > res_gen_darkdream.score
    assert res_gen_clean.score > res_gen_darkdream.score


def test_secondary_episode_title_capping():
    """
    Ensure secondary/episode titles (e.g. 'The First Day') have their weight capped
    so they do not push a potentially desynced fansub group above a clean episode match.
    """
    video = "Fullmetal Alchemist Brotherhood - 02 [BDRip 1080p]"
    sub_clean = "Fullmetal Alchemist Brotherhood - 02.srt"
    sub_with_ep_title = "[DarkDream] Fullmetal Alchemist Brotherhood - 02 - The First Day.srt"

    res_clean = calculate_match_score(video, sub_clean)
    res_tagged = calculate_match_score(video, sub_with_ep_title)

    assert res_clean.score > res_tagged.score
    assert res_clean.percentage == 100
    assert res_tagged.percentage < 100


def test_anime_strict_batch_and_ova_exclusion():
    """Rule c: Strict -500 penalty for Batch/Complete/OVA when video is a single regular episode."""
    video = "[Kaylith] Fullmetal Alchemist Brotherhood - 01"
    sub_batch = "Fullmetal Alchemist Brotherhood Complete Batch.srt"
    sub_ova = "Fullmetal Alchemist Brotherhood OVA 01.srt"
    sub_regular = "Fullmetal Alchemist Brotherhood - 01.srt"

    res_batch = calculate_match_score(video, sub_batch)
    res_ova = calculate_match_score(video, sub_ova)
    res_regular = calculate_match_score(video, sub_regular)

    assert res_batch.score <= -500
    assert res_ova.score <= -500
    assert res_regular.score >= 135
    assert res_regular.percentage == 100


def test_anime_episode_mismatch_penalty():
    """Rule d: Strictly penalize (-1000) if anime episode numbers differ."""
    video = "[SubsPlease] Frieren - Beyond Journeys End - 01"
    sub_diff_ep = "[SubsPlease] Frieren - Beyond Journeys End - 02.srt"

    res = calculate_match_score(video, sub_diff_ep)
    assert res.score == -1000
    assert res.percentage == 0


def test_non_anime_isolation_preservation():
    """Confirm regular movies and TV series use standard matching with zero regression or anime logic interference."""
    # Movie Dune
    score_dune = calculate_match_score(
        "Dune.Part.Two.2024.2160p.UHD.Remux.DV.Atmos-FLUX.mkv",
        "Dune.Part.Two.2024.2160p.UHD.Remux.DV.Atmos-FLUX.srt",
    )
    assert score_dune.score >= 135
    assert score_dune.percentage == 100

    # TV Series Breaking Bad episode mismatch
    score_bb = calculate_match_score(
        "Breaking.Bad.S01E02.1080p.WEB-DL-FLUX", "Breaking Bad S01E03 1080p WEB-DL-FLUX"
    )
    assert score_bb.score < -500
    assert score_bb.percentage == 0


def test_implicit_season_guard_with_season_words():
    """
    Implicit Season 1 MUST ONLY apply if the file is truly a single absolute anime episode
    (e.g. Show - 02 or Show 02) and contains NO mention of any other season words.
    """
    video_s01 = "Show.S01E02.1080p.mkv"
    sub_abs = "Show - 02.srt"
    sub_se05 = "Show.Se05.Ep02.srt"
    sub_s02 = "Show.2nd.Season.02.srt"

    # Absolute single episode without season words maps to Season 1
    res_abs = calculate_match_score(video_s01, sub_abs)
    assert res_abs.percentage == 100
    assert res_abs.score >= 135

    # Files with explicit season words are NOT assumed Season 1 and mismatch
    res_se05 = calculate_match_score(video_s01, sub_se05)
    assert res_se05.score <= -500
    assert res_se05.percentage == 0

    res_s02 = calculate_match_score(video_s01, sub_s02)
    assert res_s02.score <= -500
    assert res_s02.percentage == 0


def test_clean_final_label_mixed_case_hex_hashes():
    """Ensure regex stripping hex IDs catches cases with mixed characters (uppercase, lowercase, digits)."""
    from app.services.aggregator import clean_final_label

    assert clean_final_label("Show.S01E02_6E4B079E36DD0457.srt") == "Show.S01E02"
    assert (
        clean_final_label("[100%] [Subdl] Dexter.2006.S08.1080p_b5F1Bc3A5ec49533")
        == "[100%] [Subdl] Dexter.2006.S08.1080p"
    )
    assert (
        clean_final_label("[95%] [SubSource] Show.Se05.Ep02_01703060D61E07A7")
        == "[95%] [SubSource] Show.Se05.Ep02"
    )


@pytest.mark.asyncio
async def test_endpoint_strips_trailing_hex_hashes_dynamically(client):
    """
    Ensure the endpoint dynamically strips any trailing hex hash from sub['title']
    even if the provider or cache supplied a filename with a trailing hash.
    """
    sub_with_hash = SubtitleRelease(
        release_name="Dexter.2006.S08E01.1080p.BluRay.x265-ImE_b5f1bc3a5ec49533.srt",
        download_url="/sub/test.srt",
        provider="subdl",
        lang="ara",
        score=140,
        match_percentage=100,
    )

    with (
        patch("app.main._http_client", new_callable=AsyncMock),
        patch(
            "app.providers.subdl.SubdlProvider.search_subtitles",
            new=AsyncMock(return_value=[sub_with_hash]),
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
            "app.providers.cinemeta.CinemetaClient.get_metadata", new=AsyncMock(return_value=None)
        ),
    ):
        extra = "filename=Dexter.2006.S08E01.1080p.BluRay.x265-ImE.mkv"
        resp = client.get(f"/subtitles/series/tt0773262:8:1/{extra}.json")
        assert resp.status_code == 200
        data = resp.json()
        subs = data["subtitles"]
        assert len(subs) == 1
        # Trailing hash must NOT appear in title
        assert "b5f1bc3a5ec49533" not in subs[0]["title"]
        assert subs[0]["title"] == "[100%] [SubDL] Dexter.2006.S08E01.1080p.BluRay.x265-ImE"

        # Ensure underlying release_name was NOT mutated
        assert (
            sub_with_hash.release_name
            == "Dexter.2006.S08E01.1080p.BluRay.x265-ImE_b5f1bc3a5ec49533.srt"
        )

        # Ensure item.id and item.url are valid and intact
        sub_url = subs[0]["url"]
        assert "/sub/" in sub_url
        assert sub_url.endswith(".srt")
        sub_id = sub_url.split("/sub/")[-1].replace(".srt", "")
        meta = cache_manager.get_metadata(sub_id)
        assert meta is not None
        assert (
            meta["release_name"] == "Dexter.2006.S08E01.1080p.BluRay.x265-ImE_b5f1bc3a5ec49533.srt"
        )


def test_hunter_x_hunter_episode_matching():
    """Verify Hunter X Hunter S01E02 matching favors Episode 2 and rejects Episode 91."""
    v_raw = "Hunter.x.Hunter.2011.S01E02.German.DL.DTS.1080p.BluRay.x265-ABJ"
    s_wrong = "Hunter X Hunter-91 [1080p].srt"
    s_correct1 = "Hunter X Hunter.2011.E2.srt"
    s_correct2 = "[MST-Luckysubs] Hunter X Hunter [02] [BD].srt"

    res_wrong = calculate_match_score(v_raw, s_wrong)
    res_correct1 = calculate_match_score(v_raw, s_correct1)
    res_correct2 = calculate_match_score(v_raw, s_correct2)

    # Episode 91 must be heavily penalized and given 0%
    assert res_wrong.score == -1000
    assert res_wrong.percentage == 0

    # Clean Episode 2 must receive 100%
    assert res_correct1.score >= 135
    assert res_correct1.percentage == 100

    # Fansub Episode 2 must receive at least 95%
    assert res_correct2.percentage >= 95


def test_anime_absolute_numbering_formats_extraction():
    """Verify wide spectrum of absolute anime numbering formats cleanly extract episode and title."""
    samples = {
        "01.srt": (1, ""),
        "02.srt": (2, ""),
        "1.srt": (1, ""),
        "2.srt": (2, ""),
        "[Animeiat] 02.srt": (2, ""),
        "[Animeiat] 2.srt": (2, ""),
        "02 - Arabic.srt": (2, ""),
        "2 - Arabic.srt": (2, ""),
        "Show [02].mkv": (2, "show"),
        "Show_02.srt": (2, "show"),
        "Show - 2.mkv": (2, "show"),
        "Frieren - 02.srt": (2, "frieren"),
        "Frieren 02.srt": (2, "frieren"),
        "Frieren_02.srt": (2, "frieren"),
    }
    for filename, (expected_ep, expected_title) in samples.items():
        meta = extract_metadata(filename)
        assert meta["episode"] == expected_ep, f"Failed episode for {filename}: got {meta['episode']}"
        assert meta["absolute_episode"] == expected_ep, f"Failed abs_ep for {filename}: got {meta['absolute_episode']}"
        if expected_title:
            assert meta["title"] == expected_title, f"Failed title for {filename}: got {meta['title']}"


def test_anime_01_02_rank_first_with_100_percent():
    """Verify compatible anime subtitles for 01, 02 conventions rank #1 with 100% score."""
    video = "[SubsPlease] Sousou no Frieren - 02 [1080p].mkv"
    subs = [
        SubtitleRelease(id="1", release_name="02.srt", lang="ara", provider="subdl", download_url="http://1"),
        SubtitleRelease(id="2", release_name="[Animeiat] 02.srt", lang="ara", provider="subdl", download_url="http://2"),
        SubtitleRelease(id="3", release_name="Frieren - 02.srt", lang="ara", provider="subdl", download_url="http://3"),
        SubtitleRelease(id="4", release_name="02 - Arabic.srt", lang="ara", provider="subdl", download_url="http://4"),
        SubtitleRelease(id="5", release_name="2.srt", lang="ara", provider="subdl", download_url="http://5"),
        SubtitleRelease(id="6", release_name="[SubDL] Sousou no Frieren [1-28].srt", lang="ara", provider="subdl", download_url="http://6"),
        SubtitleRelease(id="7", release_name="01.srt", lang="ara", provider="subdl", download_url="http://7"),
        SubtitleRelease(id="8", release_name="03.srt", lang="ara", provider="subdl", download_url="http://8"),
        SubtitleRelease(id="9", release_name="Sousou no Frieren - 01.srt", lang="ara", provider="subdl", download_url="http://9"),
    ]

    ranked = rank_subtitles(video, subs, season=1, episode=2)

    # Wrong episodes must be discarded
    ranked_names = [r.release_name for r in ranked]
    assert "01.srt" not in ranked_names
    assert "03.srt" not in ranked_names
    assert "Sousou no Frieren - 01.srt" not in ranked_names

    # All matching single-episode subtitles must have 100% and score >= 135
    for r in ranked:
        if r.release_name != "[SubDL] Sousou no Frieren [1-28].srt":
            assert r.match_percentage == 100, f"Expected 100% for {r.release_name}, got {r.match_percentage}%"
            assert r.score >= 135, f"Expected score >= 135 for {r.release_name}, got {r.score}"

    # Batch pack must rank after single-episode candidates
    batch_idx = ranked_names.index("[SubDL] Sousou no Frieren [1-28].srt")
    assert batch_idx == len(ranked) - 1


def test_anime_tv_convention_video_with_absolute_subs():
    """Verify video with S01E02 or title matches absolute episode subtitles 02.srt, [Animeiat] 02.srt at 100%."""
    video = "Sousou no Frieren S01E02.1080p.mkv"
    subs = [
        SubtitleRelease(id="1", release_name="02.srt", lang="ara", provider="subdl", download_url="http://1"),
        SubtitleRelease(id="2", release_name="[Animeiat] 02.srt", lang="ara", provider="subdl", download_url="http://2"),
        SubtitleRelease(id="3", release_name="Sousou no Frieren 02.srt", lang="ara", provider="subdl", download_url="http://3"),
        SubtitleRelease(id="4", release_name="01.srt", lang="ara", provider="subdl", download_url="http://4"),
    ]

    ranked = rank_subtitles(video, subs, season=1, episode=2)
    assert len(ranked) == 3
    assert ranked[0].match_percentage == 100
    assert ranked[0].score >= 135
    for r in ranked:
        assert r.match_percentage == 100


def test_extract_srt_from_zip_anime_episode_matching():
    """Verify in-memory zip extractor extracts the exact episode (e.g. 02.srt, 2.srt) instead of 01.srt."""
    import io
    import zipfile

    from app.extractor import extract_srt_from_zip

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("01.srt", "Content of Episode 1")
        zf.writestr("02.srt", "Content of Episode 2")
        zf.writestr("03.srt", "Content of Episode 3")
    zip_bytes = buf.getvalue()

    ext_ep2 = extract_srt_from_zip(zip_bytes, episode=2)
    assert b"Episode 2" in ext_ep2

    ext_ep3 = extract_srt_from_zip(zip_bytes, episode=3)
    assert b"Episode 3" in ext_ep3

