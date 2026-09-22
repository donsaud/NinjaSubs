"""Tests for subtitle advertisement stripping (strip_advertisements)."""

import re

from app.utils.cleaners import strip_advertisements

LEADING_AD = "www.adsite.com"
TRAILING_AD = "Subscribe @promo_channel"
MIDDLE_URL = "Check www.example.com for details"

SRT_INPUT = (
    "1\n"
    "00:00:01,000 --> 00:00:03,000\n"
    f"{LEADING_AD}\n"
    "\n"
    "2\n"
    "00:00:10,000 --> 00:00:13,000\n"
    "Hello, this is legitimate dialogue.\n"
    "\n"
    "3\n"
    "00:05:00,000 --> 00:05:03,000\n"
    f"{MIDDLE_URL}\n"
    "\n"
    "4\n"
    "00:10:10,000 --> 00:10:20,000\n"
    f"{TRAILING_AD}\n"
)


def _srt_indices(text: str) -> list[int]:
    """Extract SRT sequence-number lines (pure integers)."""
    return [int(line) for line in text.splitlines() if re.fullmatch(r"\d+", line.strip())]


def test_removes_leading_and_trailing_ads_but_keeps_middle_dialogue():
    out = strip_advertisements(SRT_INPUT)

    # Leading header ad and trailing telegram handle are removed.
    assert LEADING_AD not in out
    assert "@promo_channel" not in out
    # Legitimate dialogue is preserved.
    assert "Hello, this is legitimate dialogue." in out
    # A URL in the middle of the movie (outside the 60s / 90s windows) is kept.
    assert MIDDLE_URL in out

    # SRT indices remain contiguous starting at 1.
    assert _srt_indices(out) == [1, 2]


def test_removes_telegram_and_arabic_channel_markers():
    content = (
        "1\n"
        "00:00:01,000 --> 00:00:03,000\n"
        "قناة الترجمة تيليجرام\n"
        "\n"
        "2\n"
        "00:00:05,000 --> 00:00:08,000\n"
        "حوار حقيقي في منتصف الفيلم\n"
    )
    out = strip_advertisements(content)
    # Promotional Arabic markers are stripped, but the credit line is preserved
    # (keep_translator_credits defaults to True).
    assert "قناة" not in out
    assert "تيليجرام" not in out
    assert "الترجمة" in out
    assert "حوار حقيقي في منتصف الفيلم" in out
    assert _srt_indices(out) == [1, 2]


def test_removes_credit_lines():
    content = (
        "1\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "Sync and Corrected by SomeGuy\n"
        "\n"
        "2\n"
        "00:00:04,000 --> 00:00:06,000\n"
        "Actual subtitle line\n"
    )
    out = strip_advertisements(content)
    assert "Corrected by" not in out
    assert "Sync" not in out
    assert "Actual subtitle line" in out


def test_short_words_and_handles_in_middle_are_not_stripped():
    """Words resembling patterns (e.g. 'coming') must survive outside the windows."""
    content = (
        "1\n"
        "00:00:01,000 --> 00:00:03,000\n"
        "Opening line\n"
        "\n"
        "2\n"
        "00:05:00,000 --> 00:05:03,000\n"
        "I am coming home, don't worry.\n"
        "\n"
        "3\n"
        "00:09:00,000 --> 00:09:03,000\n"
        "Email me at home please.\n"
    )
    out = strip_advertisements(content)
    assert "I am coming home" in out
    assert "Email me at home please." in out
    assert _srt_indices(out) == [1, 2, 3]


def test_vtt_ads_removed_and_header_preserved():
    content = (
        "WEBVTT\n"
        "\n"
        "00:00:01.000 --> 00:00:03.000\n"
        "Encoded by team\n"
        "\n"
        "00:00:05.000 --> 00:00:08.000\n"
        "Legit dialogue\n"
    )
    out = strip_advertisements(content)
    assert out.startswith("WEBVTT")
    assert "Encoded by" not in out
    assert "Legit dialogue" in out
    assert "-->" in out


def test_no_ads_returns_content_unchanged():
    content = (
        "1\n"
        "00:00:01,000 --> 00:00:03,000\n"
        "Just a normal line\n"
        "\n"
        "2\n"
        "00:00:05,000 --> 00:00:08,000\n"
        "Another normal line\n"
    )
    assert strip_advertisements(content) == content


def test_ass_content_is_untouched():
    ass = (
        "[Script Info]\n"
        "Title: Example\n\n"
        "[Events]\n"
        "Dialogue: 0,0:00:01.00,0:00:04.00,Default,,0,0,0,,www.adsite.com\n"
    )
    assert strip_advertisements(ass) == ass


def test_keep_translator_credit_strips_only_ad_link():
    """A credit line with a URL keeps the credit text but drops the URL."""
    content = (
        "1\n"
        "00:00:01,000 --> 00:00:03,000\n"
        "ترجمة فلان www.adsite.com\n"
        "\n"
        "2\n"
        "00:00:05,000 --> 00:00:08,000\n"
        "Hello world\n"
    )
    out = strip_advertisements(content, keep_translator_credits=True)
    assert "ترجمة فلان" in out
    assert "www.adsite.com" not in out
    assert "Hello world" in out
    assert _srt_indices(out) == [1, 2]


def test_keep_translator_credit_disabled_drops_credit_cue():
    """With keep_translator_credits=False the whole credit cue is removed."""
    content = (
        "1\n"
        "00:00:01,000 --> 00:00:03,000\n"
        "ترجمة فلان www.adsite.com\n"
        "\n"
        "2\n"
        "00:00:05,000 --> 00:00:08,000\n"
        "Hello world\n"
    )
    out = strip_advertisements(content, keep_translator_credits=False)
    assert "ترجمة" not in out
    assert "www.adsite.com" not in out
    assert "Hello world" in out
    assert _srt_indices(out) == [1]


def test_trailing_credit_with_telegram_handle_kept():
    content = (
        "1\n"
        "00:00:01,000 --> 00:00:03,000\n"
        "Opening line\n"
        "\n"
        "2\n"
        "00:00:10,000 --> 00:00:12,000\n"
        "Translated by Jane\n"
        "Join t.me/official @promo"
        "\n"
    )
    out = strip_advertisements(content, keep_translator_credits=True)
    assert "Translated by Jane" in out
    assert "t.me/official" not in out
    assert "@promo" not in out
    assert _srt_indices(out) == [1, 2]


def test_crlf_line_endings_are_preserved():
    content = SRT_INPUT.replace("\n", "\r\n")
    out = strip_advertisements(content)
    assert "\r\n" in out
    assert LEADING_AD not in out
