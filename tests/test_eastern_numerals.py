"""Tests for Eastern Arabic numeral conversion (convert_eastern_arabic_numerals)."""

from app.utils.cleaners import (
    convert_eastern_arabic_numerals,
    convert_eastern_arabic_numerals_bytes,
)


def _srt(*cues: tuple[str, str, str]) -> str:
    blocks = [f"{i}\n{start} --> {end}\n{text}" for i, (start, end, text) in enumerate(cues, 1)]
    return "\n\n".join(blocks) + "\n"


def test_converts_digits_in_arabic_dialogue():
    content = _srt(("00:00:01,000", "00:00:03,000", "لدي 3 أيام و 25 ساعة"))

    out = convert_eastern_arabic_numerals(content)

    assert "لدي ٣ أيام و ٢٥ ساعة" in out


def test_does_not_touch_timestamps_or_indices():
    content = _srt(("00:01:23,456", "00:01:25,000", "الحلقة 2"))

    out = convert_eastern_arabic_numerals(content)

    assert "00:01:23,456 --> 00:01:25,000" in out
    assert out.startswith("1\n00:01:23,456")
    assert "الحلقة ٢" in out


def test_english_lines_are_untouched():
    content = _srt(("00:00:01,000", "00:00:03,000", "I have 3 days and 25 hours"))

    out = convert_eastern_arabic_numerals(content)

    assert out == content


def test_html_tag_attributes_are_protected_but_inner_text_converts():
    content = _srt(
        ("00:00:01,000", "00:00:03,000", '<font color="#00FF00">3 أيام</font>')
    )

    out = convert_eastern_arabic_numerals(content)

    assert 'color="#00FF00"' in out
    assert "٣ أيام" in out


def test_ass_override_tags_are_protected():
    content = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "\n"
        "[Events]\n"
        "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,"
        r"{\c&H00FF00&}لدي 3 أيام"
    )

    out = convert_eastern_arabic_numerals(content)

    assert r"{\c&H00FF00&}" in out
    assert "لدي ٣ أيام" in out


def test_alphanumeric_tokens_are_preserved():
    content = _srt(
        ("00:00:01,000", "00:00:03,000", "شغلت MP4 على Windows 11 مع AK-47 و S01E01")
    )

    out = convert_eastern_arabic_numerals(content)

    assert "MP4" in out
    assert "Windows 11" in out
    assert "AK-47" in out
    assert "S01E01" in out


def test_mixed_standalone_number_and_token():
    content = _srt(("00:00:01,000", "00:00:03,000", "الفصل 2 من AK-47"))

    out = convert_eastern_arabic_numerals(content)

    assert "الفصل ٢ من AK-47" in out


def test_non_cue_content_is_unchanged():
    content = "plain text 123\n"
    assert convert_eastern_arabic_numerals(content) == content


def test_bytes_wrapper_handles_utf8():
    content = _srt(("00:00:01,000", "00:00:03,000", "3 أيام"))
    out = convert_eastern_arabic_numerals_bytes(content.encode("utf-8")).decode("utf-8")
    assert "٣ أيام" in out


def test_conversion_is_idempotent():
    content = _srt(("00:00:01,000", "00:00:03,000", "3 أيام"))
    once = convert_eastern_arabic_numerals(content)
    assert convert_eastern_arabic_numerals(once) == once
