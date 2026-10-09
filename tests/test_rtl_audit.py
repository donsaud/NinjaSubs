"""Conservative RTL repairs, content preservation and delivery-stage interactions."""

import unicodedata
from collections import Counter
from dataclasses import replace

import pytest

from app.main import _run_subtitle_optimization_pipeline
from app.utils.cleaners import CleanOptions, convert_eastern_arabic_numerals_bytes
from app.utils.rtl import RLM, fix_rtl_punctuation, fix_rtl_punctuation_bytes


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        (".مرحبا بك", "مرحبا بك."),
        ("!مرحبا بك", "مرحبا بك!"),
        ("؟هل أنت بخير", "هل أنت بخير؟"),
        ("…مرحبا بك", "مرحبا بك…"),
        ('."مرحبا بك"', '"مرحبا بك".'),
        (".(مرحبا بك)", "(مرحبا بك)."),
        (".<i>مرحبا بك</i>", "<i>مرحبا بك</i>."),
        ("?!مرحبا بك", "مرحبا بك?!"),
        ("...مرحبا بك", "مرحبا بك..."),
        ('"...مرحبا بك"', '"...مرحبا بك"'),
        ("- .مرحبا بك", "- مرحبا بك."),
    ],
)
def test_legacy_repairs_preserve_content_and_are_idempotent(original: str, expected: str) -> None:
    fixed = fix_rtl_punctuation(original)
    assert fixed.replace(RLM, "") == expected
    assert fix_rtl_punctuation(fixed) == fixed
    assert Counter(fixed.replace(RLM, "")) == Counter(original)
    assert "".join(c for c in fixed if unicodedata.category(c).startswith("L")) == "".join(
        c for c in original if unicodedata.category(c).startswith("L")
    )


@pytest.mark.parametrize(
    "original",
    [
        "مرحبا بك.", "مرحبا بك!", "هل أنت بخير؟", "مرحبا بك…",
        '"مرحبا بك."', "«مرحبا بك.»", "“مرحبا بك.”",
        "(مرحبا بك.)", "[مرحبا بك.]", "{مرحبا بك.}",
        "- مرحبا بك.", "– مرحبا بك.", "— مرحبا بك.",
        "مرحبا بك?!", "مرحبا بك!!", "ماذا…؟",
        "مرحبا John.", "Windows 11 يعمل جيدًا.", "استخدم MP4.", "AK-47",
        "الساعة 10:30.", "الإصدار 2.0.", "سنة 2026.",
        "<i>مرحبا بك.</i>", "<b>مرحبا بك!</b>",
        '<font color="#fff">مرحبا بك.</font>',
        '<font face="Family (A) [B]" color="#fff">مرحبا بك.</font>',
        ".NET يعمل جيدًا.", "!John مرحبا", ".مرحبا John", "…مرحبا بك.",
        "- .مرحبا بك...", '"مرحبا', "مرحبا-", "؟Hello.",
        "- مرحبا\n- كيف حالك؟",
        ")<font face=\"Family (A)\">مرحبا بك</font>",
        "<i>آسفة يا كوبر)</i>",
    ],
)
def test_correct_or_ambiguous_dialogue_keeps_visible_content(original: str) -> None:
    fixed = fix_rtl_punctuation(original)
    assert fixed.replace(RLM, "") == original
    assert fix_rtl_punctuation(fixed) == fixed


@pytest.mark.parametrize(
    "original",
    ["", " ", "\t\r\n", ".", "؟.", "!؟…", "★♫", "١٢٣.",
     "Hello.", "Hello!", "(Hello.)", "<i>؟.</i>", '<i title="عربي">Hello.</i>'],
)
def test_no_rtl_dialogue_is_exact_noop(original: str) -> None:
    assert fix_rtl_punctuation(original) == original


def test_multiline_srt_keeps_cues_timestamps_and_crlf() -> None:
    source = '1\r\n00:00:01,000 --> 00:00:04,000\r\n.مرحبا بك\r\n"مرحبا بك."\r\n\r\n'
    fixed = fix_rtl_punctuation(source)
    assert fixed.replace(RLM, "") == source.replace(".مرحبا بك\r", "مرحبا بك.\r")
    assert fix_rtl_punctuation(fixed) == fixed


def test_ass_fields_tags_and_hard_breaks_are_preserved() -> None:
    prefix = "Dialogue: 0,0:00:01.00,0:00:04.00,Default,,0,0,0,,"
    source = prefix + r"{\i1}.مرحبا بك{\i0}\N؟هل أنت بخير"
    fixed = fix_rtl_punctuation(source)
    assert fixed.replace(RLM, "") == prefix + r"{\i1}مرحبا بك.{\i0}\Nهل أنت بخير؟"
    assert fix_rtl_punctuation(fixed) == fixed


def test_utf8_wrapper_and_invalid_bytes() -> None:
    assert fix_rtl_punctuation_bytes(b"") == b""
    assert fix_rtl_punctuation_bytes(b"\xff\xfe") == b"\xff\xfe"
    assert fix_rtl_punctuation_bytes("؟هل أنت بخير".encode()).decode() == "هل أنت بخير؟"


def test_toggle_off_disables_only_rtl_repair() -> None:
    source = "1\n00:00:01,000 --> 00:00:04,000\n.مرحبا بك\n".encode()
    options = CleanOptions(False, False, False, False, False, False, False)
    disabled = _run_subtitle_optimization_pipeline(
        source, enable_rtl_fix=False, enable_ad_removal=False, options=options,
    )
    enabled = _run_subtitle_optimization_pipeline(
        source, enable_rtl_fix=True, enable_ad_removal=False, options=options,
    )
    assert disabled == source
    assert enabled == fix_rtl_punctuation_bytes(source)


@pytest.mark.parametrize(
    "options",
    [
        *[
            replace(CleanOptions(False, False, False, False, False, False, False), **{flag: True})
            for flag in ("clean_commas", "clean_tags", "strip_colors", "clean_spacing", "clean_symbols")
        ],
        CleanOptions(),
    ],
)
def test_existing_cleaning_stages_compose_before_rtl(options: CleanOptions) -> None:
    source = (
        '1\n00:00:01,000 --> 00:00:04,000\n'
        '<font color="#fff"><i>.مَرْحَبًا بك</i></font>\n'
        'الساعة 10:30،  مرحبا John.\n[ضحك]\n'
    ).encode()
    preceding = _run_subtitle_optimization_pipeline(
        source, enable_rtl_fix=False, enable_ad_removal=False, options=options,
        strip_hi=True, strip_diacritics=True, eastern_arabic_numerals=False,
    )
    expected = convert_eastern_arabic_numerals_bytes(fix_rtl_punctuation_bytes(preceding))
    actual = _run_subtitle_optimization_pipeline(
        source, enable_rtl_fix=True, enable_ad_removal=False, options=options,
        strip_hi=True, strip_diacritics=True, eastern_arabic_numerals=True,
    )
    assert actual == expected
    assert b"00:00:01,000 --> 00:00:04,000" in actual
    assert b"John" in actual
    assert "الساعة ١٠:٣٠" in actual.decode()


@pytest.mark.parametrize("original", [
    '."<i>مرحبا بك</i>',
    '".مرحبا <font color="fff">بك</font>',
])
def test_ambiguous_legacy_quote_with_markup_preserves_tags_and_characters(original: str) -> None:
    fixed = fix_rtl_punctuation(original)
    assert Counter(fixed.replace(RLM, "")) == Counter(original)
    assert fix_rtl_punctuation(fixed) == fixed
    for tag in ('<i>', '</i>', '<font color="fff">', '</font>'):
        assert fixed.count(tag) == original.count(tag)
