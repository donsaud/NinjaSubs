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


def test_milliseconds_after_comma_are_never_modified():
    content = _srt(
        ("00:02:15,000", "00:02:18,500", "وصل بعد 5 دقائق"),
        ("00:02:19,125", "00:02:21,750", "الساعة 5:30"),
    )
    out = convert_eastern_arabic_numerals(content)
    assert "00:02:15,000 --> 00:02:18,500" in out
    assert "00:02:19,125 --> 00:02:21,750" in out
    assert out.splitlines()[0] == "1"
    assert "وصل بعد ٥ دقائق" in out
    assert "الساعة ٥:٣٠" in out


def test_glued_model_numbers_are_preserved():
    cases = [
        "استخدم AK-47 في المعركة",
        "اشترى AR-15 الجديد",
        "مسدس M1911 قديم",
        "شغل ملف MP4 الآن",
        "شاشة 4K واضحة",
        "فيديو 1080p عالي الجودة",
        "كرت RTX-4090 قوي",
        "رصاصة 50cal سريعة",
    ]
    for text in cases:
        out = convert_eastern_arabic_numerals(_srt(("00:00:01,000", "00:00:03,000", text)))
        for token in ("AK-47", "AR-15", "M1911", "MP4", "4K", "1080p", "RTX-4090", "50cal"):
            if token in text:
                assert token in out, f"{token!r} was converted in {text!r}"


def test_underscore_joined_identifiers_are_preserved():
    content = _srt(("00:00:01,000", "00:00:03,000", "الحلقة S01_E01 مع test_123"))
    out = convert_eastern_arabic_numerals(content)
    assert "S01_E01" in out
    assert "test_123" in out


def test_space_separated_latin_context_is_preserved():
    cases = [
        ("طائرة Boeing 747 كبيرة", "Boeing 747"),
        ("ظهر Error 404 على الشاشة", "Error 404"),
        ("اقرأ Page 12 من الكتاب", "Page 12"),
        ("مهمة Apollo 11 ناجحة", "Apollo 11"),
    ]
    for text, token in cases:
        out = convert_eastern_arabic_numerals(_srt(("00:00:01,000", "00:00:03,000", text)))
        assert token in out, f"{token!r} was converted in {text!r}"


def test_arabic_counts_years_and_durations_convert():
    content = _srt(("00:00:01,000", "00:00:03,000", "لدي 3 أولاد منذ سنة 2024 وقضيت 10 أيام"))
    out = convert_eastern_arabic_numerals(content)
    assert "لدي ٣ أولاد منذ سنة ٢٠٢٤ وقضيت ١٠ أيام" in out


def test_percentage_and_price_convert():
    content = _srt(("00:00:01,000", "00:00:03,000", "خصم 95% والسعر 50 دولار"))
    out = convert_eastern_arabic_numerals(content)
    assert "خصم ٩٥% والسعر ٥٠ دولار" in out


def test_standalone_time_converts():
    content = _srt(("00:00:01,000", "00:00:03,000", "الاجتماع الساعة 5:30 مساء"))
    out = convert_eastern_arabic_numerals(content)
    assert "الاجتماع الساعة ٥:٣٠ مساء" in out


def test_mixed_convertible_and_protected_numbers():
    content = _srt(("00:00:01,000", "00:00:03,000", "الفصل 2 من AK-47 و Boeing 747 وصلت 3 مرات"))
    out = convert_eastern_arabic_numerals(content)
    assert "الفصل ٢ من AK-47 و Boeing 747 وصلت ٣ مرات" in out
