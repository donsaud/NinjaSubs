"""Tests for the ASS/SSA -> color-preserved SRT converter."""

from app.utils.ass_converter import (
    _ass_color_to_rgb,
    _format_srt_timestamp,
    _parse_ass_timestamp,
    convert_ass_to_srt,
    convert_ass_to_srt_bytes,
)

ASS_SAMPLE = """[Script Info]
Title: Sample
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1
Style: Gold,Arial,20,&H0022CCFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:01:10.00,0:01:14.00,Default,,0,0,0,,{\\c&H0000FF&}Red text{\\c}
Dialogue: 0,0:01:15.50,0:01:18.00,Default,,0,0,0,,{\\pos(400,570)\\an8}Plain line
Dialogue: 0,0:01:19.00,0:01:20.00,Gold,,0,0,0,,Styled gold line
Dialogue: 0,0:01:21.00,0:01:22.00,Default,,0,0,0,,Line one\\NLine two\\htab
Dialogue: 0,0:01:23.00,0:01:24.00,Default,,0,0,0,,{\\p1}m 0 0 l 100 0 100 100 0 100
Comment: 0,0:01:25.00,0:01:26.00,Default,,0,0,0,,a comment
"""


def test_parse_ass_timestamp_centiseconds_to_milliseconds():
    assert _parse_ass_timestamp("0:01:10.00") == 70_000
    assert _parse_ass_timestamp("0:01:15.50") == 75_500
    assert _parse_ass_timestamp("1:02:03.25") == 3_723_250
    assert _parse_ass_timestamp("0:00:01.5") == 1_500
    assert _parse_ass_timestamp("not a time") is None


def test_format_srt_timestamp():
    assert _format_srt_timestamp(70_000) == "00:01:10,000"
    assert _format_srt_timestamp(75_500) == "00:01:15,500"
    assert _format_srt_timestamp(3_723_250) == "01:02:03,250"


def test_ass_bgr_to_rgb():
    assert _ass_color_to_rgb("&H0000FF&") == "#FF0000"  # red
    assert _ass_color_to_rgb("&HFF0000&") == "#0000FF"  # blue
    assert _ass_color_to_rgb("&H00FF00&") == "#00FF00"  # green
    assert _ass_color_to_rgb("&H0022CCFF&") == "#FFCC22"
    assert _ass_color_to_rgb("") is None


def test_converter_output_structure_and_colors():
    out = convert_ass_to_srt(ASS_SAMPLE, apply_rtl=False)

    assert out.startswith("1\n00:01:10,000 --> 00:01:14,000\n")
    assert '<font color="#FF0000">Red text</font>' in out
    # Style default color (non-white) is applied.
    assert '<font color="#FFCC22">Styled gold line</font>' in out
    # Layout/positioning tags fully removed.
    assert "\\pos" not in out and "\\an" not in out and "{" not in out and "}" not in out
    # \N and \h translated.
    assert "Line one\nLine two tab" in out
    # Drawing and Comment cues dropped.
    assert "m 0 0 l 100" not in out
    assert "a comment" not in out
    # Sequential numbering without gaps.
    assert "\n1\n" not in out and out.lstrip().startswith("1\n")


def test_rtl_integration_preserves_font_tags():
    ass = (
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,{\\c&H0000FF&}مرحبا.\n"
    )
    out = convert_ass_to_srt(ass, apply_rtl=True)
    assert '<font color="#FF0000">' in out
    assert "</font>" in out
    assert "مرحبا" in out
    assert out.rstrip().endswith("\u200f") or "\u200f" in out


def test_bytes_wrapper_and_non_ass_passthrough():
    out = convert_ass_to_srt_bytes(ASS_SAMPLE.encode("utf-8"), apply_rtl=False).decode("utf-8")
    assert "<font color=\"#FF0000\">Red text</font>" in out
    srt = "1\n00:00:01,000 --> 00:00:02,000\nHello\n"
    assert convert_ass_to_srt(srt) == srt


def test_config_roundtrip_and_cache_key():
    from app.services.cache import build_cache_key
    from app.utils.config_parser import encode_user_config, parse_user_config

    assert parse_user_config(encode_user_config()).convert_ass_to_srt is True
    assert (
        parse_user_config(
            encode_user_config(convert_ass_to_srt=False)
        ).convert_ass_to_srt
        is False
    )
    base = dict(media_type="movie", imdb_id="tt0111161")
    assert build_cache_key(**base, convert_ass_to_srt=True) != build_cache_key(
        **base, convert_ass_to_srt=False
    )


def test_pipeline_converts_ass_when_enabled_and_raw_when_disabled():
    from app.main import _build_subtitle_response

    enabled = _build_subtitle_response(
        ASS_SAMPLE.encode("utf-8"), "movie.ass", "ass", convert_ass=True
    )
    body = enabled.body.decode("utf-8")
    assert enabled.media_type.startswith("application/x-subrip")
    assert "00:01:10,000 --> 00:01:14,000" in body
    assert "Dialogue:" not in body

    disabled = _build_subtitle_response(
        ASS_SAMPLE.encode("utf-8"), "movie.ass", "ass", convert_ass=False
    )
    assert disabled.media_type.startswith("text/x-ssa")
    assert "Dialogue:" in disabled.body.decode("utf-8")



ASS_E2E = (
    "[Script Info]\n"
    "Title: E2E\n"
    "ScriptType: v4.00+\n\n"
    "[V4+ Styles]\n"
    "Format: Name, Fontname, Fontsize, PrimaryColour\n"
    "Style: Default,Arial,20,&H00CCFF\n\n"
    "[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,{\\c&H00CCFF&}"
    "\u0646\u0639\u0645 , \u0644\u062f\u064a 3 \u0623\u064a\u0627\u0645\n"
)


def test_e2e_ass_converted_then_all_preferences_applied():
    """ASS -> SRT then the full SRT pipeline (RTL, commas, numerals, colors)."""
    from app.main import _build_subtitle_response
    from app.utils.cleaners import CleanOptions

    out = _build_subtitle_response(
        ASS_E2E.encode("utf-8"),
        "movie.ass",
        "ass",
        convert_ass=True,
        clean_options=CleanOptions(clean_commas=True),
        enable_rtl_fix=True,
        strip_diacritics=True,
        eastern_arabic_numerals=True,
    )
    body = out.body.decode("utf-8")
    assert out.media_type.startswith("application/x-subrip")
    # Color preserved (ASS BGR &H00CCFF -> #FFCC00) and hex attributes untouched.
    assert '<font color="#FFCC00">' in body
    # Arabic comma normalization.
    assert "\u0646\u0639\u0645\u060c" in body
    # Western -> Eastern Arabic numerals.
    assert "\u0663" in body and "3 " not in body
    # RTL anchoring.
    assert "\u200f" in body


def test_e2e_ass_color_stripping_removes_generated_font_tags():
    from app.main import _build_subtitle_response
    from app.utils.cleaners import CleanOptions

    out = _build_subtitle_response(
        ASS_E2E.encode("utf-8"),
        "movie.ass",
        "ass",
        convert_ass=True,
        clean_options=CleanOptions(strip_colors=True),
        eastern_arabic_numerals=True,
    )
    body = out.body.decode("utf-8")
    assert "<font" not in body
    assert "#FFCC00" not in body
    assert "\u0663" in body  # numerals still applied after color stripping
