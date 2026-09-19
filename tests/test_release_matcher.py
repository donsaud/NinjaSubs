"""Unit tests for the Release-Matching, Sanitization, and Scoring Engine."""

from app.utils.release_matcher import (
    calculate_compatibility_percentage,
    clean_subtitle_filename,
    extract_target_filename,
    get_source_popularity_percentage,
    levenshtein_ratio,
    parse_release_metadata,
)


def test_clean_subtitle_filename_trailing_hash():
    # Detect and strip trailing hash suffixes appended after extension or release name
    raw1 = "Dexter.2006.S08.1080p.BluRay.x265-ImE.srt_86f1f22e8f1fd5bd"
    assert clean_subtitle_filename(raw1) == "Dexter.2006.S08.1080p.BluRay.x265-ImE"

    raw2 = "file.srt_86f1f22e8f1fd5bd"
    assert clean_subtitle_filename(raw2) == "file"

    raw3 = "Movie.Name.2024.1080p.BluRay.x264-FLUX_86f1f22e8f1fd5bd.srt"
    assert clean_subtitle_filename(raw3) == "Movie.Name.2024.1080p.BluRay.x264-FLUX"

    # Subdl specific hash patterns: _86f1f22e8f1fd5bd, _fd3cebb32c50021a, _e29912b8a8e1a5fa
    assert clean_subtitle_filename("Show.S01E01.1080p_fd3cebb32c50021a") == "Show.S01E01.1080p"
    assert (
        clean_subtitle_filename("Movie.2024.BluRay.x264_e29912b8a8e1a5fa.srt")
        == "Movie.2024.BluRay.x264"
    )


def test_sanitize_release_name_and_format_subtitle_title():
    from app.utils.release_matcher import format_subtitle_title

    # Target format: strictly single space between tags, never underscores
    res1 = format_subtitle_title(
        49, "Subdl", "Dexter.2006.S08.1080p.BluRay.x265-ImE_86f1f22e8f1fd5bd.srt"
    )
    assert res1 == "[49%] [Subdl] Dexter.2006.S08.1080p.BluRay.x265-ImE"
    assert "_" not in res1.split("] ")[0]  # No underscores between badges

    res2 = format_subtitle_title(
        60, "SubSource", "Dexter.S08.1080p.BluRay.x264-ROVERS_fd3cebb32c50021a"
    )
    assert res2 == "[60%] [SubSource] Dexter.S08.1080p.BluRay.x264-ROVERS"


def test_clean_subtitle_filename_redundant_extensions():
    # Strip unnecessary redundant extensions
    raw = "Dexter.S08.1080p.BluRay.x264-ROVERS.srt.srt"
    assert clean_subtitle_filename(raw) == "Dexter.S08.1080p.BluRay.x264-ROVERS"

    raw_single = "Dexter.S08.1080p.BluRay.x264-ROVERS.srt"
    assert clean_subtitle_filename(raw_single) == "Dexter.S08.1080p.BluRay.x264-ROVERS"


def test_clean_subtitle_filename_messy_delimiters():
    # Remove consecutive underscores or messy delimiters
    raw = "Dexter__2006__-__S08"
    assert clean_subtitle_filename(raw) == "Dexter.2006.S08"

    raw2 = "Movie___Title__2022__-__1080p"
    assert clean_subtitle_filename(raw2) == "Movie.Title.2022.1080p"


def test_parse_release_metadata_remux():
    filename = "Oppenheimer.2023.2160p.UHD.BluRay.Remux.HEVC.DV.TrueHD.Atmos.7.1-Framestor.mkv"
    meta = parse_release_metadata(filename)
    assert meta["source"] == "Remux"
    assert meta["resolution"] == "2160p"
    assert meta["codec"] == "HEVC"
    assert meta["group"] == "Framestor"


def test_parse_release_metadata_bluray():
    filename = "Dune.Part.Two.2024.1080p.BluRay.x264-FLUX.mkv"
    meta = parse_release_metadata(filename)
    assert meta["source"] == "BluRay"
    assert meta["resolution"] == "1080p"
    assert meta["codec"] == "x264"
    assert meta["group"] == "FLUX"


def test_parse_release_metadata_webdl():
    filename = "Breaking.Bad.S05E16.1080p.WEB-DL.DD5.1.H.264-NTb.mkv"
    meta = parse_release_metadata(filename)
    assert meta["source"] == "WEB-DL"
    assert meta["resolution"] == "1080p"
    assert meta["codec"] == "x264"
    assert meta["group"] == "NTb"


def test_parse_release_metadata_webrip_bracket_group():
    filename = "The.Batman.2022.720p.WEBRip.x265.AAC5.1-[YTS.MX].mp4"
    meta = parse_release_metadata(filename)
    assert meta["source"] == "WEBRip"
    assert meta["resolution"] == "720p"
    assert meta["codec"] == "x265"
    assert meta["group"] == "YTS.MX"


def test_parse_release_metadata_hdtv():
    filename = "Succession.S04E01.720p.HDTV.x264-AVS[eztv].mkv"
    meta = parse_release_metadata(filename)
    assert meta["source"] == "HDTV"
    assert meta["resolution"] == "720p"
    assert meta["codec"] == "x264"
    assert meta["group"] == "AVS"


def test_parse_release_metadata_cam():
    filename = "Avatar.The.Way.of.Water.2022.CAM.x264-CMRG.mkv"
    meta = parse_release_metadata(filename)
    assert meta["source"] == "CAM"
    assert meta["resolution"] is None
    assert meta["codec"] == "x264"
    assert meta["group"] == "CMRG"


def test_parse_release_metadata_empty_or_none():
    meta = parse_release_metadata("")
    assert meta["source"] is None
    assert meta["resolution"] is None
    assert meta["group"] is None
    assert meta["codec"] is None


def test_levenshtein_ratio():
    assert levenshtein_ratio("Inception.2010.1080p.mkv", "Inception 2010 1080p.srt") == 1.0
    sim = levenshtein_ratio(
        "Dune.Part.Two.2024.1080p-FLUX.mkv", "Dune.Part.Two.2024.1080p-FLUX.srt"
    )
    assert sim == 1.0
    sim_mod = levenshtein_ratio(
        "Dune.Part.Two.2024.1080p.BluRay-FLUX.mkv", "Dune.Part.Two.2024.720p.WEB-DL-NTb.srt"
    )
    assert 0.5 < sim_mod < 0.9
    assert levenshtein_ratio("", "") == 1.0
    assert levenshtein_ratio("test", "") == 0.0


def test_calculate_compatibility_percentage_exact_match():
    target = "Gladiator.2000.1080p.BluRay.x264-SPARKS.mkv"
    sub = "Gladiator.2000.1080p.BluRay.x264-SPARKS.srt"

    score = calculate_compatibility_percentage(target, sub)
    # Source: 40, Group: 35, Resolution: 10, Codec: 5, Similarity: 10 -> 100%
    assert score == 100


def test_calculate_compatibility_percentage_high_compatibility():
    # Remux target with BluRay sub: Source (+30), Group (+35), Res (+10), Codec (+5), Similarity (~9) -> ~89%
    target = "Gladiator.2000.2160p.UHD.BluRay.Remux.HEVC-Framestor.mkv"
    sub = "Gladiator.2000.2160p.BluRay.HEVC-Framestor.srt"

    score = calculate_compatibility_percentage(target, sub)
    assert 80 <= score <= 95


def test_calculate_compatibility_percentage_incompatible_penalty():
    # BluRay target with CAM sub: -30 source penalty
    target = "Dune.2021.1080p.BluRay-FLUX.mkv"
    sub = "Dune.2021.CAM.x264-CMRG.srt"

    score = calculate_compatibility_percentage(target, sub)
    assert score <= 10  # Clamped at >= 0


def test_calculate_compatibility_percentage_clamping():
    # Ensure score never drops below 0% or exceeds 100%
    target = "Movie.2024.2160p.Remux.HEVC-GRP.mkv"
    sub_bad = "Completely.Different.CAM.srt"
    assert calculate_compatibility_percentage(target, sub_bad) == 0


def test_get_source_popularity_percentage_hierarchy():
    remux = get_source_popularity_percentage("Movie.2024.2160p.Remux.srt")
    bluray = get_source_popularity_percentage("Movie.2024.1080p.BluRay.srt")
    webdl = get_source_popularity_percentage("Movie.2024.1080p.WEB-DL.srt")
    webrip = get_source_popularity_percentage("Movie.2024.1080p.WEBRip.srt")
    hdtv = get_source_popularity_percentage("Movie.2024.720p.HDTV.srt")
    cam = get_source_popularity_percentage("Movie.2024.CAM.srt")

    assert remux > bluray > webdl > webrip > hdtv > cam
    assert remux == 60
    assert bluray == 55
    assert webdl == 50
    assert webrip == 45


def test_extract_target_filename_from_extra():
    extra1 = "filename=Dune.Part.Two.2024.2160p.UHD.BluRay.x265-FLUX.mkv"
    assert extract_target_filename(extra1) == "Dune.Part.Two.2024.2160p.UHD.BluRay.x265-FLUX.mkv"

    extra2 = "videoHash=123456789abcdef&videoSize=123456&filename=Oppenheimer.2023.1080p.mkv"
    assert extract_target_filename(extra2) == "Oppenheimer.2023.1080p.mkv"

    extra3 = "filename%3DThe.Batman.2022.1080p.WEBRip.mkv.json"
    assert extract_target_filename(extra3) == "The.Batman.2022.1080p.WEBRip.mkv"

    assert (
        extract_target_filename(None, query_params={"filename": "Inception.2010.mkv"})
        == "Inception.2010.mkv"
    )
    assert extract_target_filename(None) is None
    assert extract_target_filename("") is None


def test_matching_group_and_source_guarantees_minimum_85_percent():
    """Verify that any release with matching group and source scores >= 85%."""
    target = "Gladiator.2000.1080p.BluRay.x264-SPARKS.mkv"

    # 1. Matching group & source, but different resolution and codec
    sub_diff_res_codec = "Gladiator.2000.720p.BluRay.x265-SPARKS.srt"
    score1 = calculate_compatibility_percentage(target, sub_diff_res_codec)
    assert score1 >= 85

    # 2. Matching group & source, no resolution or codec mentioned
    sub_no_tech = "Gladiator.2000.BluRay-SPARKS.srt"
    score2 = calculate_compatibility_percentage(target, sub_no_tech)
    assert score2 >= 85

    # 3. Known scene group matching (playWEB, FLUX, Framestor, ION10, PSA)
    target_web = "Show.S01E01.1080p.WEB-DL.DDP5.1.Atmos-playWEB.mkv"
    sub_web = "Show.S01E01.720p.WEB-DL.AAC-playWEB.srt"
    score3 = calculate_compatibility_percentage(target_web, sub_web)
    assert score3 >= 85

    target_flux = "Dune.Part.Two.2024.1080p.BluRay.x264-FLUX.mkv"
    sub_flux = "Dune.Part.Two.2024.720p.BluRay.x265-FLUX.srt"
    score4 = calculate_compatibility_percentage(target_flux, sub_flux)
    assert score4 >= 85


def test_noisy_domains_stripping_preserves_compatibility_score():
    """Verify that domain brackets, noisy site prefixes, and metadata brackets do not degrade score."""
    target = "Gladiator.2000.1080p.BluRay.x264-SPARKS.mkv"
    clean_sub = "Gladiator.2000.1080p.BluRay.x264-SPARKS.srt"
    clean_score = calculate_compatibility_percentage(target, clean_sub)
    assert clean_score == 100

    # 1. [arabp2p.net] noisy domain bracket
    sub_arabp2p = "[arabp2p.net] - Gladiator.2000.1080p.BluRay.x264-SPARKS.srt"
    assert calculate_compatibility_percentage(target, sub_arabp2p) == 100

    # 2. moappmovies prefix
    sub_moapp = "moappmovies.Gladiator.2000.1080p.BluRay.x264-SPARKS.srt"
    assert calculate_compatibility_percentage(target, sub_moapp) == 100

    # 3. moappmovies.net - prefix
    sub_moapp_dash = "moappmovies.net - Gladiator.2000.1080p.BluRay.x264-SPARKS.srt"
    assert calculate_compatibility_percentage(target, sub_moapp_dash) == 100

    # 4. Unrelated metadata bracket [4k] or (4k)
    sub_meta_bracket = "Gladiator.2000.[4k].1080p.BluRay.x264-SPARKS.srt"
    assert calculate_compatibility_percentage(target, sub_meta_bracket) == 100

    # 5. Trailing hash _86f1f22e8f1fd5bd
    sub_hash = "Gladiator.2000.1080p.BluRay.x264-SPARKS_86f1f22e8f1fd5bd.srt"
    assert calculate_compatibility_percentage(target, sub_hash) == 100


def test_rank_and_sort_subtitles_strict_descending_order():
    """Verify that rank_and_sort_subtitles strictly orders results descending by computed score."""
    from app.models import SubtitleRelease
    from app.services.ranking import rank_and_sort_subtitles

    target = "Gladiator.2000.1080p.BluRay.x264-SPARKS.mkv"
    subs = [
        SubtitleRelease(
            release_name="Gladiator.2000.CAM.x264-CMRG.srt",
            download_url="http://example.com/cam.srt",
            provider="subdl",
        ),
        SubtitleRelease(
            release_name="Gladiator.2000.1080p.BluRay.x264-SPARKS.srt",
            download_url="http://example.com/sparks_100.srt",
            provider="subdl",
        ),
        SubtitleRelease(
            release_name="Gladiator.2000.720p.BluRay.x265-SPARKS.srt",
            download_url="http://example.com/sparks_85.srt",
            provider="subdl",
        ),
        SubtitleRelease(
            release_name="Gladiator.2000.1080p.WEB-DL.x264-NTb.srt",
            download_url="http://example.com/ntb_web.srt",
            provider="subsource",
        ),
        SubtitleRelease(
            release_name="Gladiator (2000).srt",
            download_url="http://example.com/fallback.srt",
            provider="subsource",
        ),
    ]

    ranked = rank_and_sort_subtitles(subs, target_filename=target)

    # All items must be assigned a score
    for s in ranked:
        assert isinstance(s.score, int)

    # Strictly descending order
    scores = [s.score for s in ranked]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == 100
    assert scores[1] >= 85
    assert scores[-1] == 0
    assert ranked[0].release_name == "Gladiator.2000.1080p.BluRay.x264-SPARKS.srt"
    assert ranked[-1].release_name == "Gladiator.2000.CAM.x264-CMRG.srt"
