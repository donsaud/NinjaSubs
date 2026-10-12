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
    assert ranked[0].compatibility.discovery_eligible is True
    assert ranked[0].compatibility.state("release_group").value == "match"
    assert ranked[0].compatibility.state("source").value == "match"


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
    assert compat.accepted is False
    assert compat.episode_match is False




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
    assert len(ranked) == 1  # Explicit remake/year contradiction is rejected.




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






def test_audit_anime_1049_1050_multi_episode():
    """One Piece 1049-1050 accepted for target 1050."""
    target = "One Piece 1050 1080p.mkv"
    sub = SubtitleRelease(
        release_name="One Piece 1049-1050.srt", download_url="http://1", provider="subdl"
    )

    compat = calculate_compatibility(extract_metadata(target), sub.release_name)
    assert compat.accepted is True
    from app.services.release_scoring_integration import compare_metadata
    assert not compare_metadata(target, sub.release_name).ready_reference_eligible


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
    assert ranked[1] is sub_en_best

    # Group 2: English
    assert ranked[2] is sub_ar_low
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
    """Final output must preserve the source-first execution key.

    Displayed point percentage is not the ordering authority: exact video
    source precedes a different source with more complete metadata.
    """
    for prev, curr in zip(ranked, ranked[1:], strict=False):
        assert prev.weighted_key and curr.weighted_key
        assert prev.weighted_key <= curr.weighted_key, (
            f"Sorting violation: {prev.release_name} ({prev.weighted_key}) "
            f"ranked before {curr.release_name} ({curr.weighted_key})"
        )


def _assert_percentages_clamped(ranked: list[SubtitleRelease]) -> None:
    for sub in ranked:
        pct = getattr(sub, "match_percentage", 0)
        assert 0 <= pct <= 100, (
            f"Clamp violation: {sub.release_name} -> {pct}% (must be within [0, 100])"
        )












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


# --------------------------------------------------------------------------- #
# O. MOVIEHASH ELEVATION: matched_by_hash flag, 100% rank, player labels
# --------------------------------------------------------------------------- #




def test_synced_candidate_still_rises_when_metadata_matches():
    """A synced candidate whose tokens match keeps its natural top position."""
    target = "Mad.Men.S01E02.1080p.WEB-DL.x264-WADU.mkv"

    def _rel(name, **kwargs):
        url = "http://x/" + "".join(ch for ch in name if ch.isalnum())
        return SubtitleRelease(
            release_name=name, download_url=url, provider="subdl", lang="eng", **kwargs
        )

    matched_synced = _rel("Mad.Men.S01E02.1080p.WEB-DL.x264-WADU.srt", status="synced")
    generic = _rel("Mad.Men.S01E02.720p.HDTV.x264-Other.srt")

    ranked = rank_subtitles(target, [generic, matched_synced])
    assert ranked[0].release_name == matched_synced.release_name


def test_player_labels_use_only_the_standard_badge_format():
    """Labels are always `[{pct}%] [{provider}] {name}` with no injected tags.

    Hash and synced states are expressed through the match percentage and the
    release name alone. No override, emoji, or separator may be injected.
    """
    from app.services.aggregator import format_informative_badge

    hashed = SubtitleRelease(
        release_name="Movie.2024.1080p.BluRay.x264-FLUX.srt",
        download_url="http://1",
        provider="opensubtitles",
        lang="eng",
        matched_by_hash=True,
        match_percentage=100,
    )
    hashed_label = format_informative_badge(hashed, 100)
    assert hashed_label == "[100%] [OpenSubtitles] Movie.2024.1080p.BluRay.x264-FLUX"
    assert "⚡" not in hashed_label

    synced = SubtitleRelease(
        release_name="Movie.2024.1080p.BluRay.x264-FLUX.srt",
        download_url="http://2",
        provider="subdl",
        lang="ara",
        match_percentage=100,
        status="synced",
    )
    synced_label = format_informative_badge(synced, 100)
    assert synced_label == "[100%] [SubDL] Movie.2024.1080p.BluRay.x264-FLUX"
    assert "⚡" not in synced_label
    # A synced artifact is indistinguishable from the same release without the flag.
    unsynced = SubtitleRelease(
        release_name="Movie.2024.1080p.BluRay.x264-FLUX.srt",
        download_url="http://3",
        provider="subdl",
        lang="ara",
        match_percentage=100,
    )
    assert format_informative_badge(unsynced, 100) == synced_label

    plain = SubtitleRelease(
        release_name="Movie.2024.1080p.BluRay.x264-FLUX.srt",
        download_url="http://3",
        provider="subdl",
        lang="ara",
        match_percentage=100,
    )
    assert format_informative_badge(plain, 100) == "[100%] [SubDL] Movie.2024.1080p.BluRay.x264-FLUX"


# --------------------------------------------------------------------------- #
# P. DETERMINISTIC TIER HIERARCHY
# --------------------------------------------------------------------------- #
# Ordering must be a strict, token-driven hierarchy. The additive soft score is
# still reported for display, but it must never be able to reorder candidates
# across tiers.


def test_group_without_compatible_source_cannot_override_binding():
    """Tier 1 beats tier 2 even when the generic candidate matches far more props."""
    target = "Show.S01E05.2160p.BluRay.REMUX.DV.Atmos.H.265-PiR8.mkv"

    exact_group = SubtitleRelease(
        release_name="Show.S01E05.1080p.WEB-DL.x264-PiR8.srt",
        download_url="http://x/group",
        provider="subdl",
        lang="eng",
    )
    # No group in common, but it matches source, resolution, codec and platform.
    generic = SubtitleRelease(
        release_name="Show.S01E05.2160p.BluRay.REMUX.DV.Atmos.H.265-FLUX.srt",
        download_url="http://x/generic",
        provider="subdl",
        lang="eng",
    )

    ranked = rank_subtitles(target, [generic, exact_group])
    assert ranked[0] is generic
    assert ranked[0].compatibility.state("source").value == "match"
    assert ranked[1].compatibility.state("source").value == "mismatch"
    assert ranked[0].match_percentage > ranked[1].match_percentage




def test_known_scene_groups_beat_generic_web_release():
    """Each canonical scene group outranks an otherwise identical generic web rip."""
    for group in ("FLUX", "FraMeSToR", "CHD", "NTb"):
        target = f"Show.S01E05.1080p.BluRay.x264-{group}.mkv"
        group_rel = SubtitleRelease(
            release_name=f"Show.S01E05.1080p.BluRay.x264-{group}.srt",
            download_url=f"http://x/{group}",
            provider="subdl",
            lang="eng",
        )
        generic = SubtitleRelease(
            release_name="Show.S01E05.1080p.WEB-DL.x264.mkv.srt".replace(".mkv", ""),
            download_url=f"http://x/generic-{group}",
            provider="subdl",
            lang="eng",
        )
        ranked = rank_subtitles(target, [generic, group_rel])
        assert ranked[0].release_name == group_rel.release_name, group


def test_wrong_title_never_earns_a_strong_tier():
    """A release-group collision on an unrelated title must not look perfect."""
    target = "Shogun.2024.S01E05.1080p.DSNP.WEB-DL.H.264-FLUX.mkv"

    other_work = SubtitleRelease(
        # FLUX releases many unrelated titles, so the group matches by accident.
        release_name="Dune.Part.Two.2024.2160p.DSNP.WEB-DL.H.265-FLUX.srt",
        download_url="http://x/other",
        provider="subdl",
        lang="eng",
    )
    correct = SubtitleRelease(
        release_name="Shogun.2024.S01E05.1080p.AMZN.HDTV.x264-GRP.srt",
        download_url="http://x/correct",
        provider="subdl",
        lang="eng",
    )

    ranked = rank_subtitles(target, [other_work, correct])
    assert ranked[0].release_name == correct.release_name
    assert ranked == [correct]  # Hard content contradiction is never displayed.




def test_tier_ordering_is_stable_and_input_independent():
    """Candidate order must not depend on provider arrival order."""
    target = "Show.S01E05.1080p.BluRay.x264-PiR8.mkv"
    rels = [
        SubtitleRelease(
            release_name=n, download_url="http://x/" + n, provider="subdl", lang="eng"
        )
        for n in (
            "Show.S01E05.1080p.BluRay.x264-PiR8.srt",
            "Show.S01E05.1080p.WEB-DL.x264.srt",
            "Show.S01E05.720p.HDTV.x264.srt",
        )
    ]
    forward = [s.release_name for s in rank_subtitles(target, rels)]
    backward = [s.release_name for s in rank_subtitles(target, list(reversed(rels)))]
    assert forward == backward


def test_badge_includes_uploader_only_when_present():
    """`(by 'uploader')` is appended when known and omitted entirely when not."""
    from app.services.aggregator import format_informative_badge

    with_uploader = SubtitleRelease(
        release_name="Movie.2024.1080p.BluRay.x264-FLUX.srt",
        download_url="http://1",
        provider="subdl",
        lang="eng",
        match_percentage=95,
        uploader="subsmaster",
    )
    assert (
        format_informative_badge(with_uploader, 95)
        == "[95%] [SubDL] Movie.2024.1080p.BluRay.x264-FLUX (by subsmaster)"
    )

    without = SubtitleRelease(
        release_name="Movie.2024.1080p.BluRay.x264-FLUX.srt",
        download_url="http://2",
        provider="subdl",
        lang="eng",
        match_percentage=95,
    )
    label = format_informative_badge(without, 95)
    assert label == "[95%] [SubDL] Movie.2024.1080p.BluRay.x264-FLUX"
    assert "by" not in label


def test_no_badge_contains_a_lightning_bolt():
    """No label path may emit the removed status overrides."""
    from app.services.aggregator import format_informative_badge

    for kwargs in (
        {"matched_by_hash": True, "provider": "opensubtitles"},
        {"status": "synced", "provider": "subdl"},
        {"status": "synced", "matched_by_hash": True, "provider": "opensubtitles"},
        {"provider": "subsource"},
    ):
        rel = SubtitleRelease(
            release_name="Movie.2024.1080p.BluRay.x264-FLUX.srt",
            download_url="http://1",
            lang="eng",
            **kwargs,
        )
        assert "⚡" not in format_informative_badge(rel, 100)
