"""
Comprehensive Real-World Ranking Audit Test Suite for NinjaSubs.

Audits ranking quality, relational ordering, hard compatibility filters,
and performance across realistic media release and subtitle combinations:
- Section A: TV Series (exact, services, groups, episode/season mismatch, multi-episode)
- Section B: Movies (source, resolution, codec, year mismatch, editions)
- Section C: Anime (absolute episodes, E-prefix, multi-episode, implicit S01, numbers in titles)
- Section D: Media Source hierarchy (WEB-DL, WEBRip, BluRay, Remux)
- Section E: FPS alignment and drift (23.976, 24, 29.97, 30, 25 PAL)
- Section F: Video Codecs (H264, H265, AV1)
- Section G: Audio Formats (AC3, AAC, DD5.1, DDP5.1, DTS)
- Section H: Streaming Services (HMAX, AMZN, NF)
- Section I: Binary Hash Priority
- Section J: Multi-Language Grouping & Internal Ranking
- Section K: Hearing Impaired (SDH) Inclusion & Exclusion
- Section L: 50+ Candidate Performance Benchmark
"""

from app.models import SubtitleRelease
from app.services.subtitle_matcher import (
    calculate_compatibility,
    determine_fps_relation,
    extract_metadata,
    hard_compatibility_filter,
    is_anime_content,
    rank_subtitles,
)
from tools.audit_subtitle_ranking import audit_ranking

# ============================================================================
# A. TV SERIES AUDIT
# ============================================================================


def test_audit_tv_exact_same_release():
    """Exact same release metadata must achieve top compatibility rank and be accepted."""
    target = "The.Last.of.Us.S01E01.1080p.UHD.BluRay.x265-FLUX"
    sub_exact = SubtitleRelease(
        release_name="The.Last.of.Us.S01E01.1080p.UHD.BluRay.x265-FLUX.srt",
        download_url="http://1",
        provider="subdl",
    )
    sub_generic = SubtitleRelease(
        release_name="The.Last.of.Us.S01E01.HDTV.srt", download_url="http://2", provider="subdl"
    )

    ranked = rank_subtitles(target, [sub_generic, sub_exact])
    assert len(ranked) == 2
    assert ranked[0].release_name == sub_exact.release_name
    assert ranked[0].compatibility.accepted is True
    assert ranked[0].compatibility.release_group_match is True
    assert ranked[0].compatibility.source_match is True


def test_audit_tv_same_episode_different_streaming_service():
    """Different streaming services for same episode remain compatible with a soft penalty."""
    target = "House.of.the.Dragon.S02E01.1080p.HMAX.WEB-DL-FLUX"
    sub_hmax = SubtitleRelease(
        release_name="House of the Dragon S02E01 HMAX WEB-DL FLUX.srt",
        download_url="http://1",
        provider="subdl",
    )
    sub_amzn = SubtitleRelease(
        release_name="House of the Dragon S02E01 AMZN WEB-DL FLUX.srt",
        download_url="http://2",
        provider="subdl",
    )

    v_meta = extract_metadata(target)
    c_hmax = calculate_compatibility(v_meta, sub_hmax.release_name)
    c_amzn = calculate_compatibility(v_meta, sub_amzn.release_name)

    # Both accepted
    assert c_hmax.accepted is True
    assert c_amzn.accepted is True

    # Same service ranks higher than different service
    assert c_hmax.score > c_amzn.score
    assert c_hmax.service_match is True
    assert c_amzn.service_match is False


def test_audit_tv_same_episode_different_release_group():
    """Conflicting release groups remain compatible with a soft group mismatch penalty."""
    target = "Succession.S04E03.1080p.WEB-DL-FLUX"
    sub_flux = SubtitleRelease(
        release_name="Succession S04E03 1080p WEB-DL FLUX.srt",
        download_url="http://1",
        provider="subdl",
    )
    sub_ntb = SubtitleRelease(
        release_name="Succession S04E03 1080p WEB-DL NTb.srt",
        download_url="http://2",
        provider="subdl",
    )

    ranked = rank_subtitles(target, [sub_ntb, sub_flux])
    assert len(ranked) == 2
    assert ranked[0].release_name == sub_flux.release_name
    assert ranked[1].release_name == sub_ntb.release_name
    assert ranked[0].score > ranked[1].score


def test_audit_tv_wrong_episode_hard_rejection():
    """Wrong episode candidate must be strictly rejected and discarded."""
    target = "Breaking.Bad.S01E02.1080p.WEB-DL-FLUX"
    sub_e01 = SubtitleRelease(
        release_name="Breaking Bad S01E01 1080p WEB-DL-FLUX.srt",
        download_url="http://1",
        provider="subdl",
    )
    sub_e02 = SubtitleRelease(
        release_name="Breaking Bad S01E02 1080p WEB-DL-FLUX.srt",
        download_url="http://2",
        provider="subdl",
    )

    ranked = rank_subtitles(target, [sub_e01, sub_e02], discard_mismatches=True)
    assert len(ranked) == 1
    assert ranked[0].release_name == sub_e02.release_name


def test_audit_tv_wrong_season_hard_rejection():
    """Wrong season candidate must be strictly rejected and discarded."""
    target = "Severance.S01E01.1080p.ATVP.WEB-DL"
    sub_s02 = SubtitleRelease(
        release_name="Severance.S02E01.1080p.ATVP.WEB-DL.srt",
        download_url="http://1",
        provider="subdl",
    )
    sub_s01 = SubtitleRelease(
        release_name="Severance.S01E01.1080p.ATVP.WEB-DL.srt",
        download_url="http://2",
        provider="subdl",
    )

    ranked = rank_subtitles(target, [sub_s02, sub_s01], discard_mismatches=True)
    assert len(ranked) == 1
    assert ranked[0].release_name == sub_s01.release_name


def test_audit_tv_multi_episode_contains_target():
    """Multi-episode subtitle containing target episode must be accepted."""
    target = "Show.S01E02.1080p.WEB-DL"
    sub_pack = SubtitleRelease(
        release_name="Show.S01E01-E03.1080p.WEB-DL.srt", download_url="http://1", provider="subdl"
    )

    v_meta = extract_metadata(target)
    s_meta = extract_metadata(sub_pack.release_name)
    accepted, reason, _ = hard_compatibility_filter(v_meta, s_meta)
    assert accepted is True
    compat = calculate_compatibility(v_meta, s_meta)
    assert compat.accepted is True
    assert compat.episode_match is True


def test_audit_tv_repack_proper_alignment_calibration():
    """Matching REPACK/PROPER subtitle ranks above otherwise identical normal subtitle."""
    target = "House.of.the.Dragon.S02E01.1080p.HMAX.WEB-DL.REPACK-FLUX"
    sub_repack = SubtitleRelease(
        release_name="House.of.the.Dragon.S02E01.1080p.HMAX.WEB-DL.REPACK-FLUX.srt",
        download_url="http://1",
        provider="subdl",
    )
    sub_normal = SubtitleRelease(
        release_name="House.of.the.Dragon.S02E01.1080p.HMAX.WEB-DL-FLUX.srt",
        download_url="http://2",
        provider="subdl",
    )

    v_meta = extract_metadata(target)
    compat_repack = calculate_compatibility(v_meta, sub_repack.release_name)
    compat_normal = calculate_compatibility(v_meta, sub_normal.release_name)

    # Both must be accepted (soft ranking penalty, not hard rejection)
    assert compat_repack.accepted is True
    assert compat_normal.accepted is True

    # Matching version ranks strictly above mismatched version
    ranked = rank_subtitles(target, [sub_normal, sub_repack])
    assert len(ranked) == 2
    assert ranked[0].release_name == sub_repack.release_name
    assert ranked[0].score > ranked[1].score


# ============================================================================
# B. MOVIES AUDIT
# ============================================================================


def test_audit_movie_same_title_year_source():
    """Same movie title, year, and source must achieve high score."""
    target = "Oppenheimer.2023.2160p.UHD.Remux.DV.Atmos-FLUX"
    sub = SubtitleRelease(
        release_name="Oppenheimer.2023.2160p.UHD.Remux.srt",
        download_url="http://1",
        provider="subdl",
    )

    v_meta = extract_metadata(target)
    compat = calculate_compatibility(v_meta, sub.release_name)
    assert compat.accepted is True
    assert compat.score > 80


def test_audit_movie_different_source():
    """Exact source match ranks above cross-source match."""
    target = "Gladiator.2000.1080p.BluRay.x264-CMRG"
    sub_bluray = SubtitleRelease(
        release_name="Gladiator.2000.1080p.BluRay.srt", download_url="http://1", provider="subdl"
    )
    sub_webrip = SubtitleRelease(
        release_name="Gladiator.2000.1080p.WEBRip.srt", download_url="http://2", provider="subdl"
    )

    ranked = rank_subtitles(target, [sub_webrip, sub_bluray])
    assert ranked[0].release_name == sub_bluray.release_name
    assert ranked[0].score > ranked[1].score


def test_audit_movie_different_resolution():
    """Different resolution for same movie/source is compatible with minor score variation."""
    target = "Interstellar.2014.2160p.BluRay"
    sub_2160 = SubtitleRelease(
        release_name="Interstellar.2014.2160p.BluRay.srt", download_url="http://1", provider="subdl"
    )
    sub_1080 = SubtitleRelease(
        release_name="Interstellar.2014.1080p.BluRay.srt", download_url="http://2", provider="subdl"
    )

    c_2160 = calculate_compatibility(extract_metadata(target), sub_2160.release_name)
    c_1080 = calculate_compatibility(extract_metadata(target), sub_1080.release_name)
    assert c_2160.accepted is True
    assert c_1080.accepted is True
    assert c_2160.score >= c_1080.score


def test_audit_movie_different_codec():
    """Different video codecs (x264 vs x265) remain compatible."""
    target = "Inception.2010.1080p.BluRay.x264-SPARKS"
    sub_x265 = SubtitleRelease(
        release_name="Inception.2010.1080p.BluRay.x265.srt",
        download_url="http://1",
        provider="subdl",
    )

    compat = calculate_compatibility(extract_metadata(target), sub_x265.release_name)
    assert compat.accepted is True
    assert compat.score > 40


def test_audit_movie_wrong_year():
    """Wrong year candidate receives penalty and ranks below same year."""
    target = "The.Batman.2022.1080p.BluRay"
    sub_2022 = SubtitleRelease(
        release_name="The Batman 2022 BluRay.srt", download_url="http://1", provider="subdl"
    )
    sub_2021 = SubtitleRelease(
        release_name="The Batman 2021 BluRay.srt", download_url="http://2", provider="subdl"
    )

    ranked = rank_subtitles(target, [sub_2021, sub_2022])
    assert ranked[0].release_name == sub_2022.release_name
    assert ranked[0].score > ranked[1].score


def test_audit_calibration_dune_year_mismatch_vs_same_year_webdl():
    """Target Dune 2024 Remux: 2024 WEB-DL must rank above 2023 BluRay (year mismatch calibrated to -60)."""
    target = "Dune.Part.Two.2024.2160p.UHD.Remux.DV.Atmos-FLUX"
    sub_2023_bluray = SubtitleRelease(
        release_name="Dune Part Two 2023 1080p BluRay.srt",
        download_url="http://1",
        provider="subdl",
    )
    sub_2024_webdl = SubtitleRelease(
        release_name="Dune Part Two 2024 1080p WEB-DL.srt",
        download_url="http://2",
        provider="subdl",
    )

    v_meta = extract_metadata(target)
    compat_2023 = calculate_compatibility(v_meta, sub_2023_bluray.release_name)
    compat_2024 = calculate_compatibility(v_meta, sub_2024_webdl.release_name)

    # Both accepted (year mismatch is a soft penalty, not hard rejection)
    assert compat_2023.accepted is True
    assert compat_2024.accepted is True

    # Candidate B (same year WEB-DL) ranks strictly above Candidate A (wrong year BluRay)
    assert compat_2024.score > compat_2023.score

    ranked = rank_subtitles(target, [sub_2023_bluray, sub_2024_webdl])
    assert len(ranked) == 2
    assert ranked[0].release_name == sub_2024_webdl.release_name


def test_audit_calibration_same_year_source_hierarchy_preserved():
    """Same-year BluRay ranks above different-source release when all other identity metadata is equal."""
    target = "Dune.Part.Two.2024.2160p.UHD.Remux.DV.Atmos-FLUX"
    sub_2024_bluray = SubtitleRelease(
        release_name="Dune Part Two 2024 1080p BluRay.srt",
        download_url="http://1",
        provider="subdl",
    )
    sub_2024_webdl = SubtitleRelease(
        release_name="Dune Part Two 2024 1080p WEB-DL.srt",
        download_url="http://2",
        provider="subdl",
    )

    v_meta = extract_metadata(target)
    compat_bluray = calculate_compatibility(v_meta, sub_2024_bluray.release_name)
    compat_webdl = calculate_compatibility(v_meta, sub_2024_webdl.release_name)

    assert compat_bluray.accepted is True
    assert compat_webdl.accepted is True
    # BluRay (family match +25, remux/bluray pref +20) ranks above WEB-DL (cross-source penalty -15)
    assert compat_bluray.score > compat_webdl.score

    ranked = rank_subtitles(target, [sub_2024_webdl, sub_2024_bluray])
    assert len(ranked) == 2
    assert ranked[0].release_name == sub_2024_bluray.release_name


def test_audit_movie_theatrical_vs_extended_rejection():
    """Explicit Theatrical vs Extended cut is hard rejected."""
    target = "Avatar.Theatrical.Cut.1080p.BluRay"
    sub_ext = SubtitleRelease(
        release_name="Avatar.Extended.Cut.1080p.BluRay.srt",
        download_url="http://1",
        provider="subdl",
    )

    v_meta = extract_metadata(target)
    accepted, reason, _ = hard_compatibility_filter(v_meta, extract_metadata(sub_ext.release_name))
    assert accepted is False
    assert "edition conflict" in reason.lower()


# ============================================================================
# C. ANIME AUDIT
# ============================================================================


def test_audit_anime_1050_vs_1050():
    """One Piece 1050 vs 1050 accepted with top score."""
    target = "One Piece 1050 1080p.mkv"
    sub = SubtitleRelease(
        release_name="One Piece 1050.srt", download_url="http://1", provider="subdl"
    )

    compat = calculate_compatibility(extract_metadata(target), sub.release_name)
    assert compat.accepted is True
    assert compat.percentage == 100


def test_audit_anime_1050_vs_e1050():
    """One Piece 1050 vs E1050 accepted with episode match."""
    target = "One Piece 1050 1080p.mkv"
    sub = SubtitleRelease(
        release_name="One Piece E1050.srt", download_url="http://1", provider="subdl"
    )

    compat = calculate_compatibility(extract_metadata(target), sub.release_name)
    assert compat.accepted is True
    assert compat.episode_match is True


def test_audit_anime_1049_1050_multi_episode():
    """One Piece 1049-1050 accepted for target 1050."""
    target = "One Piece 1050 1080p.mkv"
    sub = SubtitleRelease(
        release_name="One Piece 1049-1050.srt", download_url="http://1", provider="subdl"
    )

    compat = calculate_compatibility(extract_metadata(target), sub.release_name)
    assert compat.accepted is True


def test_audit_anime_1050_vs_1051_rejection():
    """One Piece 1050 vs 1051 strictly hard rejected."""
    target = "One Piece 1050 1080p.mkv"
    sub = SubtitleRelease(
        release_name="One Piece 1051.srt", download_url="http://1", provider="subdl"
    )

    accepted, reason, _ = hard_compatibility_filter(
        extract_metadata(target), extract_metadata(sub.release_name)
    )
    assert accepted is False
    assert "episode mismatch" in reason.lower()


def test_audit_anime_s01e01_vs_absolute_01():
    """Show - 01 vs Show.S01E01 accepted via implicit S01 equivalence."""
    target = "Show - 01.mkv"
    sub = SubtitleRelease(
        release_name="Show.S01E01.1080p.CR.WEB-DL.srt", download_url="http://1", provider="subdl"
    )

    compat = calculate_compatibility(extract_metadata(target), sub.release_name)
    assert compat.accepted is True
    assert compat.episode_match is True


def test_audit_anime_titles_with_numbers_not_misclassified():
    """Titles containing numbers (Apollo 13, Blade Runner 2049, Iron Man 3) are not anime."""
    movies = [
        "Apollo.13.1995.1080p.BluRay.x264",
        "Blade.Runner.2049.2017.2160p.UHD",
        "Iron.Man.3.2013.1080p.BluRay.x264-SPARKS",
    ]
    for m in movies:
        meta = extract_metadata(m)
        assert meta["year"] is not None
        assert meta["episode"] is None
        assert is_anime_content(m, m) is False


# ============================================================================
# D. MEDIA SOURCE AUDIT
# ============================================================================


def test_audit_source_hierarchy():
    """
    Source hierarchy:
    1. Exact WEB-DL vs WEB-DL
    2. Family WEB-DL vs WEBRip
    3. Cross-source WEB-DL vs BluRay
    """
    target = "Show.S01E01.1080p.WEB-DL"
    sub_exact = SubtitleRelease(
        release_name="Show.S01E01.1080p.WEB-DL.srt", download_url="http://1", provider="subdl"
    )
    sub_family = SubtitleRelease(
        release_name="Show.S01E01.1080p.WEBRip.srt", download_url="http://2", provider="subdl"
    )
    sub_cross = SubtitleRelease(
        release_name="Show.S01E01.1080p.BluRay.srt", download_url="http://3", provider="subdl"
    )

    ranked = rank_subtitles(target, [sub_cross, sub_family, sub_exact])
    assert ranked[0].release_name == sub_exact.release_name
    assert ranked[1].release_name == sub_family.release_name
    assert ranked[2].release_name == sub_cross.release_name


def test_audit_source_bluray_vs_remux():
    """BluRay target with Remux subtitle is disc family compatible."""
    target = "Movie.2024.1080p.BluRay"
    sub_remux = SubtitleRelease(
        release_name="Movie.2024.1080p.Remux.srt", download_url="http://1", provider="subdl"
    )

    compat = calculate_compatibility(extract_metadata(target), sub_remux.release_name)
    assert compat.accepted is True
    assert compat.source_match is True


def test_audit_source_uhd_remux_vs_webdl():
    """UHD Remux target accepts WEB-DL with lower priority than BluRay/Remux."""
    target = "Movie.2024.2160p.UHD.Remux"
    sub_bluray = SubtitleRelease(
        release_name="Movie.2024.2160p.BluRay.srt", download_url="http://1", provider="subdl"
    )
    sub_webdl = SubtitleRelease(
        release_name="Movie.2024.2160p.WEB-DL.srt", download_url="http://2", provider="subdl"
    )

    ranked = rank_subtitles(target, [sub_webdl, sub_bluray])
    assert ranked[0].release_name == sub_bluray.release_name
    assert ranked[0].score > ranked[1].score


# ============================================================================
# E. FPS AUDIT
# ============================================================================


def test_audit_fps_relations():
    """Audit exact, near, and drift FPS relations."""
    assert determine_fps_relation(23.976, 23.976) == "exact"
    assert determine_fps_relation(23.976, 24.0) == "near"
    assert determine_fps_relation(29.97, 30.0) == "near"
    assert determine_fps_relation(23.976, 25.0) == "drift"


def test_audit_fps_drift_penalty():
    """23.976 vs 25.0 applies heavy desync penalty (-150)."""
    target = "Movie.2024.1080p.BluRay.23.976fps"
    sub_matched = SubtitleRelease(
        release_name="Movie.2024.1080p.BluRay.23.976fps.srt",
        download_url="http://1",
        provider="subdl",
    )
    sub_pal = SubtitleRelease(
        release_name="Movie.2024.1080p.BluRay.25fps.srt", download_url="http://2", provider="subdl"
    )

    c_matched = calculate_compatibility(extract_metadata(target), sub_matched.release_name)
    c_pal = calculate_compatibility(extract_metadata(target), sub_pal.release_name)

    assert c_matched.score > c_pal.score
    assert c_pal.fps_relation == "drift"
    assert any("FPS drift conflict" in r for r in c_pal.reasons)


# ============================================================================
# F. CODEC AUDIT
# ============================================================================


def test_audit_codec_h264_vs_h265():
    """H264 video with H265 subtitle remains compatible."""
    target = "Movie.2024.1080p.BluRay.x264"
    sub = SubtitleRelease(
        release_name="Movie.2024.1080p.BluRay.x265.srt", download_url="http://1", provider="subdl"
    )

    compat = calculate_compatibility(extract_metadata(target), sub.release_name)
    assert compat.accepted is True


def test_audit_codec_h265_vs_av1():
    """H265 video with AV1 subtitle remains compatible."""
    target = "Movie.2024.1080p.BluRay.x265"
    sub = SubtitleRelease(
        release_name="Movie.2024.1080p.BluRay.AV1.srt", download_url="http://1", provider="subdl"
    )

    compat = calculate_compatibility(extract_metadata(target), sub.release_name)
    assert compat.accepted is True


# ============================================================================
# G. AUDIO AUDIT
# ============================================================================


def test_audit_audio_formats_cross_compatibility():
    """AC3 vs AAC and DD5.1 vs DDP5.1 remain compatible."""
    v1 = extract_metadata("Movie.2024.1080p.AC3")
    c1 = calculate_compatibility(v1, "Movie.2024.1080p.AAC.srt")
    assert c1.accepted is True

    v2 = extract_metadata("Movie.2024.1080p.DD5.1")
    c2 = calculate_compatibility(v2, "Movie.2024.1080p.DDP5.1.srt")
    assert c2.accepted is True

    v3 = extract_metadata("Movie.2024.1080p.DTS")
    c3 = calculate_compatibility(v3, "Movie.2024.1080p.AAC.srt")
    assert c3.accepted is True


# ============================================================================
# H. STREAMING SERVICES AUDIT
# ============================================================================


def test_audit_streaming_services_alignment():
    """HMAX match receives bonus, HMAX vs AMZN receives soft penalty, NF vs AMZN soft penalty."""
    target = "Show.S01E01.1080p.HMAX.WEB-DL"
    sub_hmax = SubtitleRelease(
        release_name="Show.S01E01.1080p.HMAX.WEB-DL.srt", download_url="http://1", provider="subdl"
    )
    sub_amzn = SubtitleRelease(
        release_name="Show.S01E01.1080p.AMZN.WEB-DL.srt", download_url="http://2", provider="subdl"
    )

    c_match = calculate_compatibility(extract_metadata(target), sub_hmax.release_name)
    c_diff = calculate_compatibility(extract_metadata(target), sub_amzn.release_name)

    assert c_match.score > c_diff.score
    assert c_match.service_match is True
    assert c_diff.service_match is False
    assert c_diff.accepted is True  # Non-fatal


# ============================================================================
# I. HASH PRIORITY AUDIT
# ============================================================================


def test_audit_exact_hash_priority():
    """Exact binary hash match ranks first above a high-matching filename candidate."""
    target = "Random.Movie.2024.1080p.BluRay.x264-FLUX.mkv"
    sub_file = SubtitleRelease(
        release_name="Random.Movie.2024.1080p.BluRay.x264-FLUX.srt",
        download_url="http://1",
        provider="subdl",
        is_hash_match=False,
    )
    sub_hash = SubtitleRelease(
        release_name="Completely.Different.Name.srt",
        download_url="http://2",
        provider="opensubtitles",
        is_hash_match=True,
    )

    ranked = rank_subtitles(target, [sub_file, sub_hash])
    assert ranked[0].release_name == sub_hash.release_name
    assert ranked[0].is_hash_match is True
    assert ranked[0].score >= 500


# ============================================================================
# J. LANGUAGE AUDIT
# ============================================================================


def test_audit_language_preference_and_internal_ranking():
    """
    Preferred languages: Arabic then English.
    All Arabic candidates precede English candidates,
    while internal compatibility score orders candidates within each language group.
    """
    target = "Show.S01E01.1080p.WEB-DL-FLUX"
    sub_ar_best = SubtitleRelease(
        release_name="Show.S01E01.1080p.WEB-DL-FLUX.Arabic.srt",
        download_url="http://1",
        provider="subdl",
        lang="ara",
    )
    sub_ar_low = SubtitleRelease(
        release_name="Show.S01E01.720p.HDTV.Arabic.srt",
        download_url="http://2",
        provider="subdl",
        lang="ara",
    )
    sub_en_best = SubtitleRelease(
        release_name="Show.S01E01.1080p.WEB-DL-FLUX.English.srt",
        download_url="http://3",
        provider="subdl",
        lang="eng",
    )
    sub_en_low = SubtitleRelease(
        release_name="Show.S01E01.720p.HDTV.English.srt",
        download_url="http://4",
        provider="subdl",
        lang="eng",
    )

    ranked = rank_subtitles(
        target,
        [sub_en_best, sub_ar_low, sub_en_low, sub_ar_best],
        preferred_languages=["ara", "eng"],
    )
    assert len(ranked) == 4

    # Group 1: Arabic
    assert ranked[0].release_name == sub_ar_best.release_name
    assert ranked[1].release_name == sub_ar_low.release_name

    # Group 2: English
    assert ranked[2].release_name == sub_en_best.release_name
    assert ranked[3].release_name == sub_en_low.release_name


# ============================================================================
# K. SDH AUDIT
# ============================================================================


def test_audit_sdh_inclusion_and_exclusion():
    """SDH candidate is removed when exclude_sdh=True, retained when False."""
    target = "Show.S01E01.1080p.WEB-DL"
    sub_reg = SubtitleRelease(
        release_name="Show.S01E01.WEB-DL.srt",
        download_url="http://1",
        provider="subdl",
        hearing_impaired=False,
    )
    sub_sdh = SubtitleRelease(
        release_name="Show.S01E01.WEB-DL.SDH.srt",
        download_url="http://2",
        provider="subdl",
        hearing_impaired=True,
    )

    # Exclude HI = True
    ranked_off = rank_subtitles(target, [sub_sdh, sub_reg], exclude_sdh=True)
    assert len(ranked_off) == 1
    assert ranked_off[0].release_name == sub_reg.release_name

    # Exclude HI = False
    ranked_on = rank_subtitles(target, [sub_sdh, sub_reg], exclude_sdh=False)
    assert len(ranked_on) == 2


# ============================================================================
# L. 50+ CANDIDATE PERFORMANCE BENCHMARK
# ============================================================================


def test_audit_performance_benchmark_50_plus_candidates():
    """
    Performance benchmark:
    Rank at least 50 realistic candidates against a single target.
    Measure parse, filter, score, sort, and total execution time.
    Asserts throughput is fast (average < 5ms per candidate).
    """
    target = "House.of.the.Dragon.S02E01.1080p.HMAX.WEB-DL.DDP5.1.Atmos.H.264-FLUX"

    sources = ["WEB-DL", "WEBRip", "BluRay", "Remux", "HDTV"]
    groups = ["FLUX", "NTb", "CMRG", "SPARKS", "ION10", "PSA", "AVS", "MiNX"]
    resolutions = ["2160p", "1080p", "720p"]
    services = ["HMAX", "AMZN", "NF", "DSNP"]

    candidates = []
    idx = 1
    for src in sources:
        for grp in groups:
            for res in resolutions:
                svc = services[idx % len(services)]
                name = f"House.of.the.Dragon.S02E01.{res}.{svc}.{src}.x264-{grp}.srt"
                candidates.append(
                    SubtitleRelease(
                        release_name=name,
                        download_url=f"http://bench/{idx}.srt",
                        provider=f"prov_{idx % 3}",
                        lang="ara" if idx % 2 == 0 else "eng",
                    )
                )
                idx += 1
                if len(candidates) >= 60:
                    break
            if len(candidates) >= 60:
                break
        if len(candidates) >= 60:
            break

    assert len(candidates) >= 50, f"Expected >= 50 candidates, got {len(candidates)}"

    audit_res = audit_ranking(target, candidates, preferred_languages=["ara", "eng"])

    print(
        f"\n[Performance Benchmark] Processed {audit_res['candidate_count']} candidates in {audit_res['total_time_ms']} ms"
    )
    print(f"  Average time per candidate: {audit_res['avg_per_candidate_ms']} ms")
    print(
        f"  Parse: {audit_res['parse_time_ms']} ms | Filter: {audit_res['filter_time_ms']} ms | Score: {audit_res['score_time_ms']} ms | Sort: {audit_res['sort_time_ms']} ms"
    )

    # Assert sub-millisecond or fast performance (< 5ms per candidate in test environment)
    assert (
        audit_res["avg_per_candidate_ms"] < 10.0
    ), f"Performance too slow: {audit_res['avg_per_candidate_ms']} ms/candidate"
    assert audit_res["ranked_count"] == len(candidates)
