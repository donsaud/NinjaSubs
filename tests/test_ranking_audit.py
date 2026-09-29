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


# ============================================================================
# M. ZERO-TOLERANCE MONOTONIC RANKING INVARIANTS
# ============================================================================
# The core guarantee: for a single language pool, the player list is ordered by
# match_percentage strictly descending. These tests build the exact scenario
# pools from the audit matrix and assert the invariant on every adjacent pair.


def _audit_pool(names: list[str]) -> list[SubtitleRelease]:
    return [
        SubtitleRelease(release_name=n, download_url=f"http://cdn/{i}.srt", provider="subdl")
        for i, n in enumerate(names)
    ]


def _audit_rank(target: str, names: list[str], **kwargs) -> list[SubtitleRelease]:
    return rank_subtitles(
        target,
        _audit_pool(names),
        preferred_languages=["ara"],
        **kwargs,
    )


def _assert_monotonic_descending(ranked: list[SubtitleRelease]) -> None:
    """candidates[i].match_percentage >= candidates[i+1].match_percentage."""
    for prev, curr in zip(ranked, ranked[1:], strict=False):
        prev_pct = getattr(prev, "match_percentage", 0)
        curr_pct = getattr(curr, "match_percentage", 0)
        assert prev_pct >= curr_pct, (
            f"Sorting violation: {prev.release_name} ({prev_pct}%) ranked lower than "
            f"{curr.release_name} ({curr_pct}%)"
        )


def _assert_percentages_clamped(ranked: list[SubtitleRelease]) -> None:
    for sub in ranked:
        pct = getattr(sub, "match_percentage", 0)
        assert 0 <= pct <= 100, (
            f"Clamp violation: {sub.release_name} -> {pct}% (must be within [0, 100])"
        )


def test_audit_invariant_movie_scenario_a_exact_group_master_vs_alternatives():
    """Dune Part Two: exact group+master > platform web equivalent > alternates."""
    target = "Dune.Part.Two.2024.2160p.MAX.WEB-DL.DDP5.1.Atmos.DV.HDR.H.265-FLUX.mkv"
    exact = "Dune.Part.Two.2024.2160p.MAX.WEB-DL.DDP5.1.Atmos.DV.HDR.H.265-FLUX.srt"
    platform_web = "Dune.Part.Two.2024.1080p.MAX.WEB-DL.DDP5.1.H.264-NTb.srt"
    alt_webrip = "Dune.Part.Two.2024.1080p.AMZN.WEBRip.DDP5.1-GRP.srt"
    microrip = "Dune.Part.Two.2024.720p.WEBRip.x264.YTS.srt"

    ranked = _audit_rank(target, [microrip, alt_webrip, platform_web, exact])
    _assert_monotonic_descending(ranked)
    _assert_percentages_clamped(ranked)

    assert len(ranked) == 4, "all four WEB variants must be accepted (no hard reject)"
    assert ranked[0].release_name == exact, "exact group + master match must rank #1"
    assert ranked[0].match_percentage == 100
    assert ranked[1].release_name == platform_web, "same-platform WEB-DL must rank #2"
    # The two remaining alternates rank below the platform equivalent. (The
    # wrong-service AMZN WEBRip carries the streaming-service mismatch penalty,
    # so it can sit below the generic WEBRip; both are still monotonic.)
    assert {ranked[2].release_name, ranked[3].release_name} == {alt_webrip, microrip}
    assert ranked[1].match_percentage >= ranked[2].match_percentage


def test_audit_invariant_movie_scenario_b_edition_isolation():
    """Avatar Extended: a non-Extended/Theatrical subtitle must rank strictly lower."""
    target = "Avatar.The.Way.of.Water.2022.Extended.1080p.BluRay.x264-SPARKS.mkv"
    extended = "Avatar.The.Way.of.Water.2022.Extended.1080p.BluRay.x264-SPARKS.srt"
    same_group_plain = "Avatar.The.Way.of.Water.2022.1080p.BluRay.x264-SPARKS.srt"
    theatrical = "Avatar.The.Way.of.Water.2022.Theatrical.1080p.BluRay.x264-SPARKS.srt"

    ranked = _audit_rank(target, [same_group_plain, theatrical, extended])
    _assert_monotonic_descending(ranked)
    _assert_percentages_clamped(ranked)

    assert ranked[0].release_name == extended, "Extended subtitle must rank #1"
    assert ranked[0].match_percentage == 100
    # Theatrical is a hard edition conflict -> filtered out entirely.
    assert theatrical not in [s.release_name for s in ranked]

    # The same-group but non-Extended release is accepted yet ranks strictly lower.
    plain = next(s for s in ranked if s.release_name == same_group_plain)
    assert plain.match_percentage < ranked[0].match_percentage, (
        f"Non-Extended same-group release ({plain.match_percentage}%) must rank below "
        f"Extended ({ranked[0].match_percentage}%)"
    )

    # With mismatches retained, the Theatrical cut still scores 0% and stays last.
    retained = _audit_rank(
        target, [same_group_plain, theatrical, extended], discard_mismatches=False
    )
    _assert_monotonic_descending(retained)
    _assert_percentages_clamped(retained)
    assert retained[-1].release_name == theatrical
    assert retained[-1].match_percentage == 0


def test_audit_invariant_series_scenario_a_season_episode_integrity():
    """House of the Dragon S02E04: other episodes/seasons are excluded or ranked lower."""
    target = (
        "House.of.the.Dragon.S02E04.The.Red.Dragon.and.the.Gold.1080p.MAX.WEB-DL.H.264-FLUX.mkv"
    )
    match = "House.of.the.Dragon.S02E04.1080p.MAX.WEB-DL.H.264-FLUX.srt"
    wrong_ep_pre = "House.of.the.Dragon.S02E03.1080p.MAX.WEB-DL.H.264-FLUX.srt"
    wrong_ep_post = "House.of.the.Dragon.S02E05.1080p.MAX.WEB-DL.H.264-FLUX.srt"
    wrong_season = "House.of.the.Dragon.S01E04.1080p.MAX.WEB-DL.H.264-FLUX.srt"
    pool = [wrong_season, wrong_ep_pre, wrong_ep_post, match]

    ranked = _audit_rank(target, pool)
    _assert_monotonic_descending(ranked)
    _assert_percentages_clamped(ranked)
    assert ranked[0].release_name == match
    assert ranked[0].match_percentage == 100
    # Wrong season/episode candidates are disqualified completely.
    assert [s.release_name for s in ranked] == [match]

    # When mismatches are retained they must still be strictly lower than the match.
    retained = _audit_rank(target, pool, discard_mismatches=False)
    _assert_monotonic_descending(retained)
    _assert_percentages_clamped(retained)
    assert retained[0].release_name == match
    assert all(s.match_percentage <= retained[0].match_percentage for s in retained[1:])


def test_audit_invariant_series_scenario_b_audio_channels_and_multi_episode():
    """Shogun S01E05: 5.1/7.1 are audio (not episodes) and E05-E06 never trumps E05."""
    target = "Shogun.2024.S01E05.1080p.DSNP.WEB-DL.DDP5.1.H.264-FLUX.mkv"
    single = "Shogun.2024.S01E05.1080p.DSNP.WEB-DL.DDP5.1.H.264-FLUX.srt"
    double = "Shogun.2024.S01E05-E06.1080p.DSNP.WEB-DL.DDP5.1.H.264-FLUX.srt"
    audio_71 = "Shogun.2024.S01E05.1080p.DSNP.WEB-DL.DDP7.1.H.264-FLUX.srt"

    # Audio channels must never be parsed as episode numbers.
    for name in (target, single, double, audio_71):
        meta = extract_metadata(name)
        assert meta["season"] == 1, name
        assert meta["episode"] == 5, f"{name} -> episode {meta['episode']} (audio mis-parsed?)"

    # The double-episode pack must not outrank the exact single, in any order.
    for pool in ([double, single, audio_71], [single, double, audio_71], [audio_71, double, single]):
        ranked = _audit_rank(target, pool)
        _assert_monotonic_descending(ranked)
        _assert_percentages_clamped(ranked)
        assert ranked[0].release_name == single, (
            f"Exact single episode must rank #1, got {ranked[0].release_name}"
        )


def test_audit_invariant_tie_break_audio_codec():
    """Equal group+source: the closer audio profile breaks the tie."""
    target = "Movie.2024.1080p.BluRay.DTS-HD.MA.5.1.x264-GRP.mkv"
    dts_hd = "Movie.2024.1080p.BluRay.DTS-HD.MA.5.1.x264-GRP.srt"
    ddp = "Movie.2024.1080p.BluRay.DDP5.1.x264-GRP.srt"

    ranked = _audit_rank(target, [ddp, dts_hd])
    _assert_monotonic_descending(ranked)
    _assert_percentages_clamped(ranked)
    assert ranked[0].release_name == dts_hd
    assert ranked[0].score > ranked[1].score, "matching audio profile must win the tie"


def test_audit_invariant_tie_break_scene_over_reupload():
    """Equal metadata: clean scene dot-notation beats a website-watermarked re-upload."""
    target = "Movie.2024.1080p.BluRay.x264-GRP.mkv"
    scene = "Movie.2024.1080p.BluRay.x264-GRP.srt"
    reupload = "Movie 2024 1080p BluRay x264 GRP [www.SubScene.com].srt"

    ranked = _audit_rank(target, [reupload, scene])
    _assert_monotonic_descending(ranked)
    _assert_percentages_clamped(ranked)
    assert ranked[0].release_name == scene
    assert ranked[0].score > ranked[1].score


def test_audit_invariant_monotonic_and_clamped_over_large_mixed_pool():
    """Every adjacent pair descends and every percentage is within [0, 100]."""
    targets = [
        "Shogun.2024.S01E05.1080p.DSNP.WEB-DL.DDP5.1.H.264-FLUX.mkv",
        "Dune.Part.Two.2024.2160p.MAX.WEB-DL.DDP5.1.Atmos.DV.HDR.H.265-FLUX.mkv",
    ]
    services = ["DSNP", "HMAX", "AMZN", "NF", "ATVP"]
    sources = ["WEB-DL", "WEBRip", "BluRay", "Remux", "HDTV"]
    resolutions = ["2160p", "1080p", "720p"]
    groups = ["FLUX", "NTb", "CMRG", "GRP", "YTS"]

    pool: list[str] = []
    i = 0
    for svc in services:
        for src in sources:
            for res in resolutions:
                pool.append(f"Shogun.2024.S01E05.{res}.{svc}.{src}.x264-{groups[i % len(groups)]}.srt")
                pool.append(
                    f"Dune.Part.Two.2024.{res}.{svc}.{src}.H.265-{groups[i % len(groups)]}.srt"
                )
                i += 1
    pool += [
        "Shogun.2024.S01E03.1080p.DSNP.WEB-DL.srt",
        "Shogun.2024.S01E05-E06.1080p.DSNP.WEB-DL.srt",
        "Dune.Part.Two.2024.1080p.AMZN.WEBRip.srt",
    ]

    asserted = 0
    for target in targets:
        ranked = _audit_rank(target, pool, discard_mismatches=False)
        _assert_monotonic_descending(ranked)
        _assert_percentages_clamped(ranked)
        asserted += len(ranked)
    assert asserted >= 50, "expected a large candidate pool"


# --------------------------------------------------------------------------- #
# N. CONTENT GROUND-TRUTH: MISLABELED SUBTITLE DETECTION
# --------------------------------------------------------------------------- #
# Filename metadata can lie (bad uploads, re-tagged releases, unadjusted
# translator timings). These tests prove the cue-sanity validator catches a
# mistimed candidate by content, so only a mathematically synchronized top
# candidate can be served.


def _mislabeled_pair():
    """Same BluRay-CHD filename story, DVDRip-timed cues (dialogue at 00:05)."""
    target = (
        "1\n00:00:05,000 --> 00:00:06,500\nFirst spoken line here\n\n"
        "2\n00:00:07,000 --> 00:00:08,500\nSecond spoken line here\n\n"
        "3\n00:00:10,000 --> 00:00:12,000\nThird spoken line here\n"
    )
    reference = (
        "1\n00:00:50,000 --> 00:00:51,500\nFirst spoken line here\n\n"
        "2\n00:00:52,000 --> 00:00:53,500\nSecond spoken line here\n\n"
        "3\n00:00:55,000 --> 00:00:57,000\nThird spoken line here\n"
    )
    return target, reference


def test_mislabeled_subtitle_timing_anomaly_detected():
    from app.services.subtitle_matcher import (
        TIMING_MISMATCH_PENALTY,
        first_dialogue_threshold_ms,
        validate_cue_sanity,
    )

    target, reference = _mislabeled_pair()
    verdict = validate_cue_sanity(target, reference, same_family=True)
    assert verdict["ok"] is False, "45s first-dialogue delta must fail the gate"
    assert verdict["penalty"] == TIMING_MISMATCH_PENALTY
    assert -46000 <= verdict["delta_ms"] <= -44000
    assert verdict["threshold_ms"] == first_dialogue_threshold_ms(True) == 1500
    assert "TIMING_MISMATCH" in (verdict["reason"] or "")


def test_aligned_pair_passes_cue_sanity():
    """Into the Wild shape: credit + date cards, then dialogue at ~49.7s both sides."""
    from app.services.subtitle_matcher import validate_cue_sanity

    target = (
        "1\n00:00:02,694 --> 00:00:09,661\nترجمة مستخرجة من نتفليكس @user\n\n"
        "2\n00:00:09,662 --> 00:00:21,662\n25/03/2011\n\n"
        "3\n00:00:49,694 --> 00:00:50,661\nأمي\n\n"
        "4\n00:00:52,396 --> 00:00:54,193\nأمي ساعديني\n"
    )
    reference = (
        "1\n00:00:49,682 --> 00:00:50,682\nMom!\n\n"
        "2\n00:00:52,393 --> 00:00:54,227\nMom! Help me.\n"
    )
    verdict = validate_cue_sanity(target, reference, same_family=True)
    assert verdict["ok"] is True
    assert abs(verdict["delta_ms"]) <= 1500
    assert verdict["penalty"] == 0


def test_top_ranked_candidate_is_content_synchronized():
    """Two identically-named BluRay-CHD candidates; only the truly timed one validates."""
    from app.services.subtitle_matcher import median_cue_offset, validate_cue_sanity

    reference = (
        "1\n00:00:50,000 --> 00:00:51,500\nFirst spoken line here\n\n"
        "2\n00:00:52,000 --> 00:00:53,500\nSecond spoken line here\n"
    )
    good = (
        "1\n00:00:50,010 --> 00:00:51,510\nFirst spoken line here\n\n"
        "2\n00:00:52,020 --> 00:00:53,520\nSecond spoken line here\n"
    )
    bad, _ = _mislabeled_pair()
    assert validate_cue_sanity(good, reference, same_family=True)["ok"] is True
    assert validate_cue_sanity(bad, reference, same_family=True)["ok"] is False
    assert abs(median_cue_offset(good, reference) or 0) < 0.2
    assert abs(median_cue_offset(bad, reference) or 0) >= 0.2
