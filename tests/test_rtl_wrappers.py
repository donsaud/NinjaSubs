"""Bounded, non-recursive repair of terminal prefixes inside Arabic wrappers."""

import re
import unicodedata
from collections import Counter

import pytest

from app.main import _run_subtitle_optimization_pipeline
from app.utils.cleaners import CleanOptions
from app.utils.rtl import RLM, fix_rtl_punctuation, fix_rtl_punctuation_bytes

WRAPPERS = [("(", ")"), ("[", "]"), ('"', '"'), ("«", "»"), ("“", "”")]
TERMINALS = [(".", "مرحبا بك"), ("!", "انتبه"), ("؟", "هل أنت بخير"), ("?", "هل أنت بخير")]


def _assert_content_preserved(original: str, fixed: str) -> None:
    visible = fixed.replace(RLM, "")
    assert Counter(visible) == Counter(original.replace(RLM, ""))
    assert re.findall(r"<[^>]*>|\{\\[^}]*\}", visible) == re.findall(
        r"<[^>]*>|\{\\[^}]*\}", original
    )
    assert "".join(c for c in visible if unicodedata.category(c).startswith("L")) == "".join(
        c for c in original if unicodedata.category(c).startswith("L")
    )
    assert fix_rtl_punctuation(fixed) == fixed


@pytest.mark.parametrize(("opener", "closer"), WRAPPERS)
@pytest.mark.parametrize(("punctuation", "dialogue"), TERMINALS)
def test_inner_terminal_stays_inside_wrapper(
    opener: str, closer: str, punctuation: str, dialogue: str,
) -> None:
    original = f"{opener}{punctuation}{dialogue}{closer}"
    fixed = fix_rtl_punctuation(original)
    assert fixed == f"{opener}{dialogue}{punctuation}{closer}{RLM}"
    _assert_content_preserved(original, fixed)


@pytest.mark.parametrize(("opener", "closer"), WRAPPERS)
@pytest.mark.parametrize(("punctuation", "dialogue"), TERMINALS)
def test_correct_wrapped_terminal_keeps_content(
    opener: str, closer: str, punctuation: str, dialogue: str,
) -> None:
    original = f"{opener}{dialogue}{punctuation}{closer}"
    fixed = fix_rtl_punctuation(original)
    assert fixed.replace(RLM, "") == original
    _assert_content_preserved(original, fixed)


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        ("(.<i>مرحبا بك</i>)", "(<i>مرحبا بك.</i>)"),
        ("(!<b>انتبه</b>)", "(<b>انتبه!</b>)"),
        ('"؟<i>هل أنت بخير</i>"', '"<i>هل أنت بخير؟</i>"'),
        ("[?<b>هل أنت بخير</b>]", "[<b>هل أنت بخير?</b>]"),
        ("(<i>.مرحبا بك</i>)", "(<i>مرحبا بك.</i>)"),
        ("<i>.مرحبا بك</i>", "<i>مرحبا بك.</i>"),
        ("<b>!انتبه</b>", "<b>انتبه!</b>"),
        ('“<i>؟هل أنت بخير</i>”', '“<i>هل أنت بخير؟</i>”'),
        ("  (.مرحبا بك) \t", "  (مرحبا بك.) \t"),
        ("( .مرحبا بك )", "( مرحبا بك. )"),
        (f"(.مرحبا بك){RLM}", f"(مرحبا بك.){RLM}"),
    ],
)
def test_simple_emphasis_and_whitespace_preserved(original: str, expected: str) -> None:
    fixed = fix_rtl_punctuation(original)
    assert fixed.replace(RLM, "") == expected.replace(RLM, "")
    _assert_content_preserved(original, fixed)


@pytest.mark.parametrize(("opener", "closer"), WRAPPERS)
@pytest.mark.parametrize(
    "body",
    [
        "...مرحبا", "…مرحبا", ",مرحبا", "،مرحبا", ";مرحبا", "؛مرحبا",
        "?!مرحبا", "!!مرحبا", ".مرحبا.", "!انتبه!", "؟هل أنت بخير؟",
        ".NET", ".NET Core", '"Hello."', '"Windows 11."', "AK-47", "MP4",
        '"مرحبا John."', ".مرحبا John", ".الساعة 10:30 الآن", ".١٢٣",
        "!", "؟", ".", ".مرحبا\u200f بك",
    ],
)
def test_ambiguous_or_mixed_content_is_not_relocated(opener: str, closer: str, body: str) -> None:
    original = f"{opener}{body}{closer}"
    fixed = fix_rtl_punctuation(original)
    assert fixed.replace(RLM, "") == original.replace(RLM, "")
    _assert_content_preserved(original, fixed)


@pytest.mark.parametrize(
    "original",
    [
        "((.مرحبا))", '[".مرحبا"]', '"(.مرحبا)"', "«[!انتبه]»",
        '(<span class="x">.مرحبا</span>)',
        '(.<i title="مرحبا (x)">مرحبا بك</i>)',
        '"؟<i class="x">هل أنت بخير</i>"',
        "(.<i><b>مرحبا بك</b></i>)", "(.<i>مرحبا بك</b>)",
        "(.<I>مرحبا بك</I>)", "(.<font color=\"#fff\">مرحبا بك</font>)",
        "(.<i>مرحبا</i> بك)", "(<i>.مرحبا</i><b>بك</b>)",
        r"(.{\i1}مرحبا بك{\i0})", "{.مرحبا بك}",
        "<i>مرحبا بك.</i>", "(<i>مرحبا بك.</i>)", "<b>انتبه!</b>",
        "<i>…مرحبا</i>", "<b>،مرحبا</b>",
    ],
)
def test_complex_markup_and_nesting_are_unchanged(original: str) -> None:
    fixed = fix_rtl_punctuation(original)
    assert fixed.replace(RLM, "") == original
    _assert_content_preserved(original, fixed)


def test_srt_multiline_keeps_indices_timestamps_line_endings_and_tags() -> None:
    original = (
        '1\r\n00:00:01,000 --> 00:00:04,000\r\n(.مرحبا بك)\r\n'
        '"؟<i>هل أنت بخير</i>"\r\n\r\n'
        '2\r\n00:00:05,000 --> 00:00:08,000\r\n(.NET)\r\n(...مرحبا)\r\n'
    )
    expected = original.replace("(.مرحبا بك)", "(مرحبا بك.)").replace(
        '"؟<i>هل أنت بخير</i>"', '"<i>هل أنت بخير؟</i>"'
    )
    fixed = fix_rtl_punctuation_bytes(original.encode()).decode()
    assert fixed.replace(RLM, "") == expected
    _assert_content_preserved(original, fixed)


def test_ass_text_only_with_override_tags_and_hard_breaks() -> None:
    prefix = "Dialogue: 0,0:00:01.00,0:00:04.00,Default,,0,0,0,,"
    original = prefix + r"{\i1}(.مرحبا بك){\i0}\N[!انتبه]"
    expected = prefix + r"{\i1}(مرحبا بك.){\i0}\N[انتبه!]"
    fixed = fix_rtl_punctuation(original)
    assert fixed.replace(RLM, "") == expected
    _assert_content_preserved(original, fixed)


def test_shared_pipeline_keeps_wrapper_boundary_and_respects_toggle() -> None:
    original = '1\n00:00:01,000 --> 00:00:04,000\n(.<i>مرحبا بك</i>)\n'.encode()
    options = CleanOptions(False, False, False, False, False, False, False)
    enabled = _run_subtitle_optimization_pipeline(
        original, enable_rtl_fix=True, enable_ad_removal=False, options=options,
    )
    disabled = _run_subtitle_optimization_pipeline(
        original, enable_rtl_fix=False, enable_ad_removal=False, options=options,
    )
    assert disabled == original
    assert enabled.decode().replace(RLM, "") == (
        '1\n00:00:01,000 --> 00:00:04,000\n(<i>مرحبا بك.</i>)\n'
    )
