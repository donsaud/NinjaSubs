"""Standalone trailing Arabic dialogue markers use existing wrapper semantics."""

import pytest

from app.utils.rtl import RLM, fix_rtl_punctuation, fix_rtl_punctuation_bytes

DASHES = ["-", "–", "—"]
WRAPPERS = [("", ""), ("<i>", "</i>"), ("<b>", "</b>"),
            ('"', '"'), ("“", "”"), ("«", "»"), ("(", ")"), ("[", "]")]


@pytest.mark.parametrize("dash", DASHES)
@pytest.mark.parametrize("body", ["مرحبا بك", "أجل، في أفضل حال", "تسرّني معرفة ذلك",
                                  "مرحبا بك؟", "مرحبا بك!", "مرحبا بك."])
@pytest.mark.parametrize("opener,closer", WRAPPERS)
def test_trailing_dialogue_dash_relocation_and_idempotence(dash, body, opener, closer):
    original = f"{opener}{body} {dash}{closer}"
    fixed = fix_rtl_punctuation(original)
    assert fixed.replace(RLM, "") == f"{opener}{dash} {body}{closer}"
    assert fix_rtl_punctuation(fixed) == fixed


@pytest.mark.parametrize("dash", DASHES)
def test_correct_prefix_preserved(dash):
    original = f"{dash} مرحبا بك"
    fixed = fix_rtl_punctuation(original)
    assert fixed.replace(RLM, "") == original
    assert fix_rtl_punctuation(fixed) == fixed


@pytest.mark.parametrize("original", [
    "COVID-19", "A - B", "العربية - English", "123 - 456", "2025-2026", "اسم-مركب",
    "مرحبا - بك", "مرحبا بك-", "123 -", "-",
    "مرحبا بك -.", "مرحبا بك --", "- مرحبا بك -", "<i class='x'>مرحبا بك -</i>",
    "<i>مرحبا بك -</b>",
])
def test_ambiguous_and_non_dialogue_hyphens_unchanged(original):
    fixed = fix_rtl_punctuation(original)
    assert fixed.replace(RLM, "") == original
    assert fix_rtl_punctuation(fixed) == fixed


def test_complete_srt_utf8_bytes_preserve_cues_order_timestamps_and_other_text():
    original = ("1\r\n00:00:01,000 --> 00:00:03,000\r\nأجل، في أفضل حال -\r\n\r\n"
                "2\r\n00:00:03,500 --> 00:00:05,500\r\nتسرّني معرفة ذلك -\r\n\r\n"
                "3\r\n00:00:06,000 --> 00:00:07,000\r\nCOVID-19\r\n").encode()
    expected = original.decode().replace("أجل، في أفضل حال -", "- أجل، في أفضل حال").replace(
        "تسرّني معرفة ذلك -", "- تسرّني معرفة ذلك")
    fixed = fix_rtl_punctuation_bytes(original)
    assert fixed.decode().replace(RLM, "") == expected
    assert fix_rtl_punctuation_bytes(fixed) == fixed
    assert fix_rtl_punctuation_bytes(b"\xff") == b"\xff"


def test_ass_text_only_and_hard_line_breaks():
    prefix = "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,"
    original = prefix + r"{\i1}مرحبا بك -{\i0}\Nأجل، في أفضل حال –"
    fixed = fix_rtl_punctuation(original)
    assert fixed.replace(RLM, "") == prefix + r"{\i1}- مرحبا بك{\i0}\N– أجل، في أفضل حال"
    assert fix_rtl_punctuation(fixed) == fixed


def test_malformed_brackets_preserved_without_dash_relocation():
    fixed = fix_rtl_punctuation("(مرحبا بك -]")
    assert fixed.replace(RLM, "") == "(مرحبا بك -]"
    assert fix_rtl_punctuation(fixed) == fixed
