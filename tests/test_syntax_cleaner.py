"""Tests for the safe subtitle syntax/formatting cleaner (clean_subtitle_syntax)."""

import re

from app.utils.cleaners import clean_subtitle_syntax, clean_subtitle_syntax_bytes


def _srt(*cues: tuple[str, str, str]) -> str:
    blocks = [f"{i}\n{start} --> {end}\n{text}" for i, (start, end, text) in enumerate(cues, 1)]
    return "\n\n".join(blocks) + "\n"


def _indices(text: str) -> list[int]:
    return [int(line) for line in text.splitlines() if re.fullmatch(r"\d+", line.strip())]


def test_repairs_unclosed_tags_and_drops_stray_closers():
    content = _srt(("00:00:01,000", "00:00:03,000", "<i>italic line</i></b> and <u>underline"))

    out = clean_subtitle_syntax(content)

    assert "<i>italic line</i>" in out
    assert "<u>underline</u>" in out
    # Stray closing tag with no opener is dropped.
    assert "</b>" not in out


def test_clean_tags_preserves_font_colors_but_fixes_i_b():
    content = _srt(
        ("00:00:01,000", "00:00:03,000", '<font color="#ff0000"><i>Red text')
    )

    out = clean_subtitle_syntax(content)

    # clean_tags never strips colors on its own.
    assert '<font color="#ff0000">' in out
    # ...but it still repairs the unclosed <i>.
    assert "<i>Red text</i>" in out


def test_collapses_spaces_and_removes_space_before_punctuation():
    content = _srt(("00:00:01,000", "00:00:03,000", "Hello   ,   world .  Next"))

    out = clean_subtitle_syntax(content)

    assert "Hello, world. Next" in out
    assert "  " not in out.split("-->")[1]


def test_converts_loose_double_dash_to_ellipsis():
    content = _srt(("00:00:01,000", "00:00:03,000", "Wait -- what happened?"))

    out = clean_subtitle_syntax(content)

    assert "Wait... what happened?" in out
    assert "--" not in out.split("-->")[1]


def test_removes_empty_lines_and_redundant_breaks():
    content = _srt(("00:00:01,000", "00:00:03,000", "first<br><br>second<br/>third"))

    out = clean_subtitle_syntax(content)

    assert "first\nsecond\nthird" in out
    assert "first\n\nsecond" not in out
    assert "<br" not in out


def test_clamps_negligible_overlap_to_next_start():
    content = _srt(
        ("00:00:01,000", "00:00:03,000", "one"),
        ("00:00:02,800", "00:00:05,000", "two"),
    )

    out = clean_subtitle_syntax(content)

    assert "00:00:01,000 --> 00:00:02,800" in out


def test_leaves_large_overlaps_untouched():
    content = _srt(
        ("00:00:01,000", "00:00:05,000", "one"),
        ("00:00:03,000", "00:00:06,000", "two"),
    )

    out = clean_subtitle_syntax(content)

    assert "00:00:01,000 --> 00:00:05,000" in out


def test_normalizes_latin_commas_in_arabic_text():
    content = _srt(("00:01:23,456", "00:01:25,000", "على الفلك , كل جريمة"))

    out = clean_subtitle_syntax(content)

    assert "على الفلك، كل جريمة" in out
    assert "00:01:23,456" in out
    assert "على الفلك , كل جريمة" not in out


def test_arabic_comma_normalization_leaves_numbers_untouched():
    content = _srt(("00:00:01,000", "00:00:03,000", "الميزانية 1,000 دولار, 2,500 يورو"))

    out = clean_subtitle_syntax(content)

    assert "1,000" in out
    assert "2,500" in out


def test_reindexes_srt_cue_numbers():
    content = (
        "7\n00:00:01,000 --> 00:00:02,000\nHello  world\n\n"
        "9\n00:00:03,000 --> 00:00:04,000\nAgain\n"
    )

    out = clean_subtitle_syntax(content)

    assert _indices(out) == [1, 2]


def test_non_cue_content_is_returned_unchanged():
    content = "This is just plain text\nwithout any cues.\n"
    assert clean_subtitle_syntax(content) == content


def test_clean_syntax_is_idempotent():
    content = _srt(
        ("00:00:01,000", "00:00:03,000", "<i>Hello   ,  world</i> -- <font>x</font>"),
        ("00:00:02,800", "00:00:05,000", "second"),
    )
    once = clean_subtitle_syntax(content)
    assert clean_subtitle_syntax(once) == once


def test_bytes_wrapper_handles_utf8():
    content = _srt(("00:00:01,000", "00:00:03,000", "<i>نص  عربي</i>"))
    out = clean_subtitle_syntax_bytes(content.encode("utf-8")).decode("utf-8")
    assert "نص عربي" in out


def test_strip_kashida_removes_tatweel_globally():
    from app.utils.cleaners import strip_kashida, strip_kashida_bytes

    assert strip_kashida("بـــــلال") == "بلال"
    assert "ـ" not in strip_kashida("aـb")
    assert strip_kashida_bytes("xـy".encode()) == b"xy"


def test_clean_subtitle_strips_kashida_before_matching():
    from app.utils.cleaners import clean_subtitle

    content = _srt(("00:00:01,000", "00:00:03,000", "بـــــلال"))
    out = clean_subtitle(content)
    assert "ـ" not in out
    assert "بلال" in out


def test_strip_intro_credits_removes_pre_speech_branding():
    from app.utils.cleaners import strip_intro_credits

    content = _srt(
        ("00:00:02,000", "00:00:09,000", "ترجمة مستخرجة من نتفليكس @user"),
        ("00:00:09,000", "00:00:21,000", "25/03/2011"),
        ("00:00:49,000", "00:00:51,000", "مرحبا بالعالم"),
    )
    out = strip_intro_credits(content, 49682)
    assert "@user" not in out and "25/03/2011" not in out
    assert "مرحبا بالعالم" in out
    assert _indices(out) == [1]


def test_strip_intro_credits_keeps_later_cues_and_dialogue():
    from app.utils.cleaners import strip_intro_credits

    content = _srt(
        ("00:00:02,000", "00:00:09,000", "ترجمة @user"),
        ("00:00:49,000", "00:00:51,000", "مرحبا"),
        ("00:01:30,000", "00:01:32,000", "Visit www.example.com"),
    )
    out = strip_intro_credits(content, 49682)
    # Pre-speech branding dropped; later cues (even noisy ones) untouched.
    assert "مرحبا" in out and "www.example.com" in out
    assert _indices(out) == [1, 2]
    assert strip_intro_credits(content, None) == content
