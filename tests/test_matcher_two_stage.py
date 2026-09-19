"""
Comprehensive test suite for the Two-Stage Subtitle Matcher Architecture:
Stage 1: Hard Compatibility Filter (fatal desync/mismatch rejections).
Stage 2: Soft Compatibility Ranking (media compatibility, codecs, fps, groups, sources).
"""

from app.models import SubtitleRelease
from app.services.subtitle_matcher import (
    calculate_compatibility,
    calculate_match_score,
    deduplicate_subtitles,
    determine_fps_relation,
    extract_metadata,
    hard_compatibility_filter,
    is_anime_content,
    rank_subtitles,
)

# ============================================================================
# 1. NORMAL TV SHOW TESTS
# ============================================================================


def test_tv_exact_season_and_episode_match():
    """Exact S01E01 vs S01E01 must be accepted with high score."""
    video = "Breaking.Bad.S01E01.1080p.WEB-DL.x264-FLUX"
    sub = "Breaking.Bad.S01E01.WEB-DL.x264.srt"

    v_meta = extract_metadata(video)
    s_meta = extract_metadata(sub)

    accepted, reason, method = hard_compatibility_filter(v_meta, s_meta)
    assert accepted is True
    assert reason is None

    compat = calculate_compatibility(v_meta, s_meta)
    assert compat.accepted is True
    assert compat.season_match is True
    assert compat.episode_match is True
    assert compat.score > 80


def test_tv_episode_mismatch_hard_rejection():
    """S01E01 vs S01E02 must be strictly hard rejected (not accepted)."""
    video = "Breaking.Bad.S01E01.1080p.WEB-DL.x264-FLUX"
    sub = "Breaking.Bad.S01E02.1080p.WEB-DL.x264-FLUX.srt"

    v_meta = extract_metadata(video)
    s_meta = extract_metadata(sub)

    accepted, reason, method = hard_compatibility_filter(v_meta, s_meta)
    assert accepted is False
    assert "episode mismatch" in reason.lower()

    compat = calculate_compatibility(v_meta, s_meta)
    assert compat.accepted is False
    assert compat.score == -1000

    res = calculate_match_score(video, sub)
    assert res.score == -1000
    assert res.percentage == 0


def test_tv_season_mismatch_hard_rejection():
    """S01E01 vs S02E01 must be strictly hard rejected."""
    video = "Breaking.Bad.S01E01.1080p.WEB-DL.x264-FLUX"
    sub = "Breaking.Bad.S02E01.1080p.WEB-DL.x264-FLUX.srt"

    v_meta = extract_metadata(video)
    s_meta = extract_metadata(sub)

    accepted, reason, method = hard_compatibility_filter(v_meta, s_meta)
    assert accepted is False
    assert "season mismatch" in reason.lower()


def test_tv_multi_episode_pack_accepted():
    """S01E01 vs S01E01-E03 multi-episode pack must be accepted."""
    video = "Show.S01E01.1080p.WEB-DL"
    sub = "Show.S01E01-E03.1080p.WEB-DL.srt"

    v_meta = extract_metadata(video)
    s_meta = extract_metadata(sub)

    accepted, reason, method = hard_compatibility_filter(v_meta, s_meta)
    assert accepted is True

    compat = calculate_compatibility(v_meta, s_meta)
    assert compat.accepted is True
    assert compat.episode_match is True


# ============================================================================
# 2. MOVIE TESTS
# ============================================================================


def test_movie_exact_year_match():
    """Movie.2024 vs Movie.2024 must be accepted with high compatibility."""
    video = "Dune.Part.Two.2024.2160p.UHD.Remux.DV.Atmos-FLUX.mkv"
    sub = "Dune Part Two 2024 Remux.srt"

    v_meta = extract_metadata(video)
    s_meta = extract_metadata(sub)

    accepted, reason, _ = hard_compatibility_filter(v_meta, s_meta)
    assert accepted is True

    compat = calculate_compatibility(v_meta, s_meta)
    assert compat.accepted is True
    assert compat.score > 70


def test_movie_year_mismatch_soft_penalty():
    """Movie.2024 vs Movie.2023 is a soft penalty, not a hard rejection."""
    video = "Movie.Title.2024.1080p.BluRay"
    sub = "Movie.Title.2023.1080p.BluRay.srt"

    v_meta = extract_metadata(video)
    s_meta = extract_metadata(sub)

    # Years within 1-2 years are soft penalized, not hard rejected
    accepted, reason, _ = hard_compatibility_filter(v_meta, s_meta)
    assert accepted is True

    compat = calculate_compatibility(v_meta, s_meta)
    assert compat.accepted is True
    # Has mismatch penalty
    assert any("year mismatch" in r.lower() for r in compat.reasons)


def test_movie_edition_theatrical_vs_extended_rejection():
    """Theatrical video vs Extended cut subtitle must be hard rejected."""
    video = "Movie.Theatrical.Cut.1080p.BluRay"
    sub = "Movie.Extended.Cut.1080p.BluRay.srt"

    v_meta = extract_metadata(video)
    s_meta = extract_metadata(sub)

    accepted, reason, _ = hard_compatibility_filter(v_meta, s_meta)
    assert accepted is False
    assert "edition conflict" in reason.lower()


# ============================================================================
# 3. ANIME TESTS
# ============================================================================


def test_anime_exact_absolute_episode():
    """One Piece 1050 vs One Piece 1050 must be accepted with top score."""
    video = "One Piece 1050 1080p.mkv"
    sub = "One Piece 1050.srt"

    compat = calculate_compatibility(extract_metadata(video), extract_metadata(sub))
    assert compat.accepted is True
    assert compat.episode_match is True
    assert compat.percentage == 100


def test_anime_absolute_episode_mismatch_rejection():
    """One Piece 1050 vs One Piece 1051 must be strictly rejected."""
    video = "One Piece 1050 1080p.mkv"
    sub = "One Piece 1051.srt"

    accepted, reason, _ = hard_compatibility_filter(extract_metadata(video), extract_metadata(sub))
    assert accepted is False
    assert "episode mismatch" in reason.lower()


def test_anime_dash_episode_implicit_s01():
    """Show - 01 vs Show S01E01 must be accepted via implicit S01 equivalence."""
    video = "Show - 01.mkv"
    sub = "Show.S01E01.1080p.CR.WEB-DL.srt"

    compat = calculate_compatibility(extract_metadata(video), extract_metadata(sub))
    assert compat.accepted is True
    assert compat.episode_match is True


def test_anime_e02_vs_02_matching():
    """Show.E02 vs Show.02 must be accepted."""
    video = "Show.E02.1080p.mkv"
    sub = "Show 02.srt"

    compat = calculate_compatibility(extract_metadata(video), extract_metadata(sub))
    assert compat.accepted is True
    assert compat.episode_match is True


def test_apollo_13_movie_not_anime_episode():
    """Apollo 13 must not be parsed as anime episode 13."""
    meta = extract_metadata("Apollo.13.1995.1080p.BluRay.x264")
    assert meta["year"] == 1995
    assert meta["episode"] is None
    assert meta["absolute_episode"] is None
    assert is_anime_content("Apollo.13.1995.1080p.BluRay.x264") is False


def test_generic_bracket_group_movie_not_anime():
    """Generic group [FLUX] in movie release must not classify as anime."""
    filename = "[FLUX] Movie.Title.2024.1080p.WEB-DL"
    assert is_anime_content(filename, filename) is False


# ============================================================================
# 4. FPS TESTS
# ============================================================================


def test_fps_relation_exact():
    """23.976 vs 23.976 must be exact FPS relation."""
    assert determine_fps_relation(23.976, 23.976) == "exact"
    assert determine_fps_relation(25.0, 25.0) == "exact"


def test_fps_relation_near():
    """23.976 vs 24.0 or 29.97 vs 30.0 must be near FPS relation."""
    assert determine_fps_relation(23.976, 24.0) == "near"
    assert determine_fps_relation(24.0, 23.976) == "near"
    assert determine_fps_relation(29.97, 30.0) == "near"


def test_fps_relation_drift():
    """23.976 vs 25.0 PAL drift must be drift relation with desync penalty."""
    assert determine_fps_relation(23.976, 25.0) == "drift"
    assert determine_fps_relation(24.0, 25.0) == "drift"

    video = "Movie.2024.1080p.BluRay.23.976fps"
    sub_pal = "Movie.2024.1080p.BluRay.25fps.srt"
    compat = calculate_compatibility(extract_metadata(video), extract_metadata(sub_pal))
    assert compat.fps_relation == "drift"
    assert any("FPS drift" in r for r in compat.reasons)


# ============================================================================
# 5. MEDIA SOURCE TESTS
# ============================================================================


def test_source_matching_and_family():
    """WEB-DL matches WEB-DL (+40), WEBRip is family (+25), BluRay is compatible."""
    v_webdl = extract_metadata("Show.S01E01.1080p.WEB-DL")
    s_webdl = extract_metadata("Show.S01E01.1080p.WEB-DL.srt")
    s_webrip = extract_metadata("Show.S01E01.1080p.WEBRip.srt")
    s_bluray = extract_metadata("Show.S01E01.1080p.BluRay.srt")

    compat_exact = calculate_compatibility(v_webdl, s_webdl)
    compat_family = calculate_compatibility(v_webdl, s_webrip)
    compat_cross = calculate_compatibility(v_webdl, s_bluray)

    assert compat_exact.score > compat_family.score
    assert compat_family.score > compat_cross.score
    # All are accepted (compatible)
    assert compat_exact.accepted is True
    assert compat_family.accepted is True
    assert compat_cross.accepted is True


def test_resolution_only_difference_never_hard_rejected():
    """1080p video vs 720p subtitle must be accepted (soft signal only)."""
    video = "Show.S01E02.1080p.WEB-DL-FLUX"
    sub = "Show.S01E02.720p.WEB-DL-FLUX.srt"

    v_meta = extract_metadata(video)
    s_meta = extract_metadata(sub)

    accepted, reason, _ = hard_compatibility_filter(v_meta, s_meta)
    assert accepted is True
    assert reason is None

    compat = calculate_compatibility(v_meta, s_meta)
    assert compat.accepted is True
    assert compat.hard_reject_reason is None


def test_release_group_only_difference_never_hard_rejected():
    """FLUX video vs NTb subtitle must be accepted (soft penalty only)."""
    video = "Show.S01E02.1080p.WEB-DL-FLUX"
    sub = "Show.S01E02.1080p.WEB-DL-NTb.srt"

    v_meta = extract_metadata(video)
    s_meta = extract_metadata(sub)

    accepted, reason, _ = hard_compatibility_filter(v_meta, s_meta)
    assert accepted is True
    assert reason is None

    compat = calculate_compatibility(v_meta, s_meta)
    assert compat.accepted is True
    assert compat.hard_reject_reason is None
    assert compat.release_group_match is False


def test_resolution_and_group_difference_combined_still_accepted():
    """Both resolution and group differing must still be accepted, ranked lower."""
    video = "Show.S01E02.1080p.WEB-DL-FLUX"
    sub_exact = "Show.S01E02.1080p.WEB-DL-FLUX.srt"
    sub_diff = "Show.S01E02.720p.WEB-DL-NTb.srt"

    compat_exact = calculate_compatibility(extract_metadata(video), extract_metadata(sub_exact))
    compat_diff = calculate_compatibility(extract_metadata(video), extract_metadata(sub_diff))

    assert compat_diff.accepted is True
    assert compat_diff.hard_reject_reason is None
    assert compat_diff.score < compat_exact.score


# ============================================================================
# 6. CODEC AND AUDIO TESTS
# ============================================================================


def test_codec_and_audio_cross_compatibility():
    """H264 vs H265 and AC3 vs AAC are compatible, not rejected."""
    video = "Movie.2024.1080p.BluRay.x264.DTS-HD"
    sub = "Movie.2024.1080p.BluRay.x265.AAC.srt"

    compat = calculate_compatibility(extract_metadata(video), extract_metadata(sub))
    assert compat.accepted is True
    assert compat.score > 40


# ============================================================================
# 7. STREAMING SERVICE TESTS
# ============================================================================


def test_service_mismatch_is_soft_penalty_not_rejection():
    """Netflix video vs Amazon subtitle must be compatible (accepted) with soft penalty."""
    video = "Show.S01E01.1080p.NF.WEB-DL"
    sub = "Show.S01E01.1080p.AMZN.WEB-DL.srt"

    v_meta = extract_metadata(video)
    s_meta = extract_metadata(sub)

    accepted, reason, _ = hard_compatibility_filter(v_meta, s_meta)
    assert accepted is True  # NEVER hard reject streaming service mismatch

    compat = calculate_compatibility(v_meta, s_meta)
    assert compat.accepted is True
    assert compat.service_match is False
    assert any("Different streaming platforms" in r for r in compat.reasons)


# ============================================================================
# 8. HASH MATCH TESTS
# ============================================================================


def test_binary_hash_match_top_priority():
    """Binary hash match must receive identity acceptance (100%, top score)."""
    video = "Random.Movie.2024.mkv"
    sub = "Random Subtitle.srt"

    compat = calculate_compatibility(
        extract_metadata(video), extract_metadata(sub), is_hash_match=True
    )
    assert compat.accepted is True
    assert compat.is_hash_match is True
    assert compat.match_method == "hash"
    assert compat.score >= 500
    assert compat.percentage == 100


# ============================================================================
# 9. SDH AND MULTI-LANGUAGE RANKING TESTS
# ============================================================================


def test_sdh_retention_and_filtering():
    """SDH subtitles are retained by default, filtered only when exclude_sdh=True."""
    video = "Show.S01E01.1080p.WEB-DL"
    sub_regular = SubtitleRelease(
        release_name="Show.S01E01.WEB-DL.srt", download_url="http://1", provider="subdl"
    )
    sub_sdh = SubtitleRelease(
        release_name="Show.S01E01.WEB-DL.SDH.srt", download_url="http://2", provider="subdl"
    )

    # By default, both are retained
    ranked = rank_subtitles(video, [sub_sdh, sub_regular], exclude_sdh=False)
    assert len(ranked) == 2

    # When exclude_sdh=True, SDH is excluded
    ranked_filtered = rank_subtitles(video, [sub_sdh, sub_regular], exclude_sdh=True)
    assert len(ranked_filtered) == 1
    assert ranked_filtered[0].release_name == sub_regular.release_name


def test_multi_language_priority_grouping():
    """Candidates are grouped by preferred language, with internal score ranking."""
    video = "Show.S01E01.1080p.WEB-DL-FLUX"
    sub_ar_best = SubtitleRelease(
        release_name="Show.S01E01.1080p.WEB-DL-FLUX.srt",
        download_url="http://ar1",
        provider="subdl",
        lang="ara",
    )
    sub_ar_low = SubtitleRelease(
        release_name="Show.S01E01.720p.HDTV.srt",
        download_url="http://ar2",
        provider="subdl",
        lang="ara",
    )
    sub_en = SubtitleRelease(
        release_name="Show.S01E01.1080p.WEB-DL-FLUX.srt",
        download_url="http://en1",
        provider="subdl",
        lang="eng",
    )

    # If user prefers Arabic then English
    ranked = rank_subtitles(
        video, [sub_en, sub_ar_low, sub_ar_best], preferred_languages=["ara", "eng"]
    )
    assert len(ranked) == 3
    # Arabic first, ordered by score
    assert ranked[0].release_name == sub_ar_best.release_name
    assert ranked[1].release_name == sub_ar_low.release_name
    # English second
    assert ranked[2].release_name == sub_en.release_name


# ============================================================================
# 10. DEDUPLICATION TESTS
# ============================================================================


def test_subtitle_deduplication():
    """Duplicate subtitles from multiple providers are deduplicated cleanly."""
    sub1 = SubtitleRelease(
        release_name="Show.S01E01.1080p.WEB-DL-FLUX",
        download_url="http://subdl.com/1.srt",
        provider="subdl",
        lang="ara",
    )
    sub2 = SubtitleRelease(
        release_name="Show.S01E01.1080p.WEB-DL-FLUX",
        download_url="http://subdl.com/1.srt",
        provider="subsource",
        lang="ara",
    )  # identical URL
    sub3 = SubtitleRelease(
        release_name="Show.S01E01.1080p.WEB-DL-FLUX.srt",
        download_url="http://subsource.com/2.srt",
        provider="subsource",
        lang="ara",
    )  # identical normalized release + lang
    sub4 = SubtitleRelease(
        release_name="Show.S01E01.1080p.WEB-DL-FLUX.srt",
        download_url="http://subsource.com/3.srt",
        provider="subsource",
        lang="eng",
    )  # different lang

    unique = deduplicate_subtitles([sub1, sub2, sub3, sub4])
    assert len(unique) == 2
    langs = [s.lang for s in unique]
    assert "ara" in langs
    assert "eng" in langs


# ============================================================================
# 8. BAZARR HARD EXCLUSION & SCORING VERIFICATION TESTS
# ============================================================================


def test_bazarr_multi_season_batch_rejection():
    """Multi-season batch/pack (e.g. S01-S05) must be hard rejected for single episode playback."""
    video = "Breaking.Bad.S02E05.1080p.BluRay-FLUX"
    sub_batch = "Breaking.Bad.S01-S05.720p.BluRay.x264.srt"
    sub_season = "Breaking Bad Season 1-4 Complete.srt"

    v_meta = extract_metadata(video)
    s_meta1 = extract_metadata(sub_batch)
    s_meta2 = extract_metadata(sub_season)

    accepted1, reason1, _ = hard_compatibility_filter(v_meta, s_meta1)
    assert accepted1 is False
    assert "batch/complete" in reason1.lower()

    accepted2, reason2, _ = hard_compatibility_filter(v_meta, s_meta2)
    assert accepted2 is False
    assert "batch/complete" in reason2.lower()


def test_bazarr_content_type_mismatch_movie_vs_episodic():
    """Movie target vs candidate with explicit episode must be hard rejected."""
    video = "Inception.2010.1080p.BluRay.x264-FLUX"
    sub_ep = "Inception.E05.1080p.WEB-DL.srt"

    v_meta = extract_metadata(video)
    s_meta = extract_metadata(sub_ep)

    accepted, reason, _ = hard_compatibility_filter(v_meta, s_meta)
    assert accepted is False
    assert "candidate has explicit episode" in reason.lower()


def test_bazarr_content_type_mismatch_episodic_vs_movie():
    """Episodic series target vs explicit movie candidate must be hard rejected."""
    video = "Demon.Slayer.S01E01.1080p.CR.WEB-DL"
    sub_movie = "Demon Slayer The Movie Mugen Train 1080p BluRay.srt"

    v_meta = extract_metadata(video)
    s_meta = extract_metadata(sub_movie)

    accepted, reason, _ = hard_compatibility_filter(v_meta, s_meta)
    assert accepted is False
    assert "explicitly a movie" in reason.lower()


def test_bazarr_special_and_ova_conflicts():
    """Standard episode vs Special/OVA (and vice-versa) must be hard rejected."""
    video_regular = "Attack.on.Titan.S01E01.1080p.BluRay"
    sub_special = "Attack on Titan Special 01 1080p BluRay.srt"
    sub_ova = "Attack on Titan OVA 1 1080p BluRay.srt"

    v_meta_reg = extract_metadata(video_regular)
    s_meta_spec = extract_metadata(sub_special)
    s_meta_ova = extract_metadata(sub_ova)

    # Regular episode target vs Special sub
    accepted1, reason1, _ = hard_compatibility_filter(v_meta_reg, s_meta_spec)
    assert accepted1 is False
    assert "ova/special" in reason1.lower()

    # Regular episode target vs OVA sub
    accepted2, reason2, _ = hard_compatibility_filter(v_meta_reg, s_meta_ova)
    assert accepted2 is False
    assert "ova/special" in reason2.lower()

    # Special target vs regular episode sub
    video_special = "Attack.on.Titan.Special.01.1080p.BluRay"
    sub_regular = "Attack.on.Titan.S01E01.1080p.BluRay.srt"
    v_meta_spec = extract_metadata(video_special)
    s_meta_reg = extract_metadata(sub_regular)

    accepted3, reason3, _ = hard_compatibility_filter(v_meta_spec, s_meta_reg)
    assert accepted3 is False
    assert "ova/special" in reason3.lower()

    # Special target vs Special sub (Accepted!)
    accepted4, reason4, _ = hard_compatibility_filter(v_meta_spec, s_meta_spec)
    assert accepted4 is True


def test_bazarr_edition_conflict_extended_vs_theatrical():
    """Extended vs Theatrical and Director's Cut vs Theatrical must be hard rejected."""
    # Extended video vs Theatrical sub
    v_ext = extract_metadata("Movie.Extended.Cut.1080p.BluRay")
    s_th = extract_metadata("Movie.Theatrical.Cut.1080p.BluRay.srt")
    accepted1, reason1, _ = hard_compatibility_filter(v_ext, s_th)
    assert accepted1 is False
    assert "edition conflict" in reason1.lower()

    # Director's Cut video vs Theatrical sub
    v_dc = extract_metadata("Movie.Directors.Cut.1080p.BluRay")
    accepted2, reason2, _ = hard_compatibility_filter(v_dc, s_th)
    assert accepted2 is False
    assert "edition conflict" in reason2.lower()


def test_bazarr_anime_guarded_implicit_season_1():
    """Anime absolute episode with no season mention aligns with S01 and rejects S02."""
    v_anime = extract_metadata("Frieren - 02.mkv")
    s_s01 = extract_metadata("Frieren.S01E02.1080p.CR.WEB-DL.srt")
    s_s02 = extract_metadata("Frieren.S02E02.1080p.CR.WEB-DL.srt")

    compat1 = calculate_compatibility(v_anime, s_s01)
    assert compat1.accepted is True
    assert compat1.episode_match is True
    assert compat1.season_match is True

    accepted2, reason2, _ = hard_compatibility_filter(v_anime, s_s02)
    assert accepted2 is False
    assert "season mismatch" in reason2.lower()
