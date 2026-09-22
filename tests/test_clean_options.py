"""Tests for the flattened cleaning toggles (CleanOptions) and encoding fallback."""

from app.models import UserPreferences
from app.utils.cleaners import (
    CleanOptions,
    clean_subtitle,
    fix_subtitle_encoding_bytes,
)

ARABIC = "هذا نص عربي طويل لاختبار الترميز القديم"


def _srt(*cues: tuple[str, str, str]) -> str:
    blocks = [f"{i}\n{start} --> {end}\n{text}" for i, (start, end, text) in enumerate(cues, 1)]
    return "\n\n".join(blocks) + "\n"


# ---------------------------------------------------------------------------
# Encoding fallback
# ---------------------------------------------------------------------------


def test_fix_encoding_decodes_cp1256():
    raw = ARABIC.encode("cp1256")
    out = fix_subtitle_encoding_bytes(raw).decode("utf-8")
    assert out == ARABIC


def test_fix_encoding_decodes_utf8():
    assert fix_subtitle_encoding_bytes(ARABIC.encode("utf-8")).decode("utf-8") == ARABIC


def test_fix_encoding_prefers_cp1256_over_iso8859_6():
    # The documented fallback order tries cp1256 before iso-8859-6, so legacy
    # Arabic bytes decode consistently as Windows-1256.
    raw = ARABIC.encode("cp1256")
    assert fix_subtitle_encoding_bytes(raw).decode("utf-8") == ARABIC


def test_fix_encoding_empty_is_noop():
    assert fix_subtitle_encoding_bytes(b"") == b""


def test_clean_options_from_prefs_reads_each_flag():
    prefs = UserPreferences(
        fix_encoding=False,
        clean_tags=False,
        strip_colors=True,
        clean_spacing=False,
        clean_symbols=False,
        clean_commas=False,
        clean_timing=False,
    )
    options = CleanOptions.from_prefs(prefs)
    assert options.fix_encoding is False
    assert options.clean_tags is False
    assert options.strip_colors is True
    assert options.clean_spacing is False
    assert options.clean_symbols is False
    assert options.clean_commas is False
    assert options.clean_timing is False


# ---------------------------------------------------------------------------
# Individual toggles
# ---------------------------------------------------------------------------


def test_clean_tags_toggle():
    content = _srt(("00:00:01,000", "00:00:03,000", "<i>Hello"))
    assert "</i>" in clean_subtitle(content, CleanOptions(clean_tags=True))
    assert "</i>" not in clean_subtitle(content, CleanOptions(clean_tags=False))


def test_clean_tags_does_not_strip_font_colors():
    content = _srt(("00:00:01,000", "00:00:03,000", '<font color="#ff0"><b>x'))
    out = clean_subtitle(content, CleanOptions(clean_tags=True, strip_colors=False))
    assert '<font color="#ff0">' in out
    assert "<b>x</b>" in out


def test_strip_colors_toggle_removes_ass_font_and_vtt_classes():
    ass = _srt(("00:00:01,000", "00:00:03,000", r"{\c&H00FF00&}Hello"))
    font = _srt(("00:00:01,000", "00:00:03,000", '<font color="#ff0">Hi</font>'))
    vtt = _srt(("00:00:01,000", "00:00:03,000", "<c.yellow>Yo</c>"))
    opts = CleanOptions(strip_colors=True)
    assert r"\c&H00FF00&" not in clean_subtitle(ass, opts)
    assert "<font" not in clean_subtitle(font, opts)
    assert "<c.yellow>" not in clean_subtitle(vtt, opts)
    # Off -> left intact.
    assert "Hello" in clean_subtitle(ass, CleanOptions(strip_colors=False))


def test_clean_spacing_toggle():
    content = _srt(("00:00:01,000", "00:00:03,000", "word   , next"))
    assert "word, next" in clean_subtitle(content, CleanOptions(clean_spacing=True))
    assert "word   , next" in clean_subtitle(content, CleanOptions(clean_spacing=False))


def test_clean_symbols_toggle():
    content = _srt(("00:00:01,000", "00:00:03,000", "Wait -- now<br>next"))
    on = clean_subtitle(content, CleanOptions(clean_symbols=True))
    assert "..." in on
    assert "<br>" not in on
    off = clean_subtitle(content, CleanOptions(clean_symbols=False))
    assert "--" in off


def test_clean_commas_toggle():
    content = _srt(("00:00:01,000", "00:00:03,000", "نص , نص"))
    assert "نص، نص" in clean_subtitle(content, CleanOptions(clean_commas=True))
    assert "نص،" not in clean_subtitle(content, CleanOptions(clean_commas=False))


def test_clean_timing_toggle():
    content = _srt(
        ("00:00:01,000", "00:00:03,000", "one"),
        ("00:00:02,800", "00:00:05,000", "two"),
    )
    clamped = clean_subtitle(content, CleanOptions(clean_timing=True))
    assert "00:00:01,000 --> 00:00:02,800" in clamped
    unclamped = clean_subtitle(content, CleanOptions(clean_timing=False))
    assert "00:00:01,000 --> 00:00:03,000" in unclamped


def test_all_toggles_off_returns_content_unchanged():
    content = _srt(
        ("00:00:01,000", "00:00:03,000", "<i>Hello   , world<br>more -- x"),
        ("00:00:02,800", "00:00:05,000", "<font color='#f00'>two</font>"),
    )
    options = CleanOptions(
        fix_encoding=False,
        clean_tags=False,
        strip_colors=False,
        clean_spacing=False,
        clean_symbols=False,
        clean_commas=False,
        clean_timing=False,
    )
    assert clean_subtitle(content, options) == content
