"""Regression tests for subtitle text-processing defects.

Covers the ad cleaner deleting ordinary Arabic dialogue, text lost after a
blank line inside a cue, BOM-prefixed WebVTT, unordered/duplicated ASS events,
Arabic letters outside the base block in the RTL fix, HI keyword substrings,
dialogue dashes lost with speaker labels, and nondeterministic release groups.
"""

import pytest

from app.services.ranking import parse_release_metadata
from app.utils.ass_converter import convert_ass_to_srt
from app.utils.cleaners import (
    clean_subtitle,
    strip_advertisements,
    strip_hi_artifacts,
)
from app.utils.rtl import _ENDS_WITH_ARABIC_LETTER_REGEX


def _srt(*texts: str, start_s: int = 1) -> str:
    blocks = []
    for i, text in enumerate(texts, 1):
        s = start_s + (i - 1) * 3
        blocks.append(f"{i}\n00:00:{s:02d},000 --> 00:00:{s + 2:02d},000\n{text}")
    return "\n\n".join(blocks) + "\n"


# --------------------------------------------------------------------------- #
# Ad / credit cleaner
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "line",
    [
        "مرحبا بكم في قناة الجزيرة",  # "channel" in dialogue
        "رأيت ذلك على تويتر",  # "Twitter" in dialogue
        "سأقوم بتقديم طلب",  # "submission" in dialogue
        "أسباب الترجمة كثيرة",  # "translation" in dialogue
    ],
)
@pytest.mark.parametrize("keep_credits", [True, False])
def test_ordinary_arabic_dialogue_survives_ad_removal(line, keep_credits):
    out = strip_advertisements(_srt(line, "حوار"), keep_translator_credits=keep_credits)
    assert line in out


@pytest.mark.parametrize(
    "line",
    ["تابعونا على قناتنا", "قناة الترجمة على تيليجرام", "تابعنا على تويتر", "تويتر: @subs"],
)
def test_promotional_arabic_lines_are_still_removed(line):
    out = strip_advertisements(_srt(line, "حوار حقيقي"), keep_translator_credits=False)
    assert line not in out
    assert "حوار حقيقي" in out


@pytest.mark.parametrize("line", ["ترجمة: فلان", "ترجمة وتعديل فلان", "إعداد وتنسيق: فلان"])
def test_real_credit_lines_are_removed_when_credits_are_off(line):
    out = strip_advertisements(_srt(line, "حوار حقيقي"), keep_translator_credits=False)
    assert line not in out


# --------------------------------------------------------------------------- #
# Blank line inside a cue
# --------------------------------------------------------------------------- #

_SPLIT_CUE = (
    "1\n00:00:01,000 --> 00:00:02,000\nLine  one\n\nLine two\n\n"
    "2\n00:00:03,000 --> 00:00:04,000\nNext\n"
)


def test_clean_subtitle_keeps_text_after_blank_line_in_cue():
    out = clean_subtitle(_SPLIT_CUE)  # the double space forces a rewrite
    assert "Line two" in out
    assert out.count(" --> ") == 2


def test_strip_hi_keeps_text_after_blank_line_in_cue():
    content = _SPLIT_CUE.replace("Line  one", "[MUSIC] Line one")
    out = strip_hi_artifacts(content)
    assert "Line two" in out
    assert "MUSIC" not in out


def test_vtt_note_block_is_not_folded_into_a_cue():
    content = (
        "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHello  there\n\n"
        "NOTE translator comment\n\n00:00:03.000 --> 00:00:04.000\nBye\n"
    )
    out = clean_subtitle(content)
    assert "translator comment" not in out
    assert "Hello there" in out and "Bye" in out


# --------------------------------------------------------------------------- #
# BOM-prefixed WebVTT
# --------------------------------------------------------------------------- #


def test_bom_prefixed_webvtt_stays_webvtt():
    content = "﻿WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHello  there\n"
    out = clean_subtitle(content)
    assert "00:00:01.000 --> 00:00:02.000" in out
    assert "\n1\n" not in out  # no SRT cue index inserted


# --------------------------------------------------------------------------- #
# ASS -> SRT
# --------------------------------------------------------------------------- #

_ASS_HEAD = (
    "[Script Info]\nScriptType: v4.00+\n\n[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
)


def test_ass_events_are_emitted_in_time_order():
    ass = (
        _ASS_HEAD
        + "Dialogue: 0,0:00:05.00,0:00:06.00,Default,,0,0,0,,Second\n"
        + "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,First\n"
    )
    out = convert_ass_to_srt(ass, apply_rtl=False)
    assert out.index("First") < out.index("Second")
    assert out.startswith("1\n00:00:01,000 --> 00:00:02,000\nFirst")


def test_ass_layered_duplicate_events_are_emitted_once():
    line = "Dialogue: {layer},0:00:01.00,0:00:02.00,Default,,0,0,0,,Same line\n"
    out = convert_ass_to_srt(_ASS_HEAD + line.format(layer=0) + line.format(layer=1), False)
    assert out.count("Same line") == 1


# --------------------------------------------------------------------------- #
# RTL ranges
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("text", ["ماذا تفعلی", "یک", "کتاب", "نصٌّ"])
def test_lines_ending_in_extended_arabic_letters_qualify(text):
    assert _ENDS_WITH_ARABIC_LETTER_REGEX.search(text)


# --------------------------------------------------------------------------- #
# HI stripping
# --------------------------------------------------------------------------- #


def test_hi_keyword_is_matched_as_a_word_not_a_substring():
    out = strip_hi_artifacts(_srt("(لأسباب أمنية) لا نستطيع", "(طرق على الباب) من هناك؟"))
    assert "(لأسباب أمنية)" in out
    assert "الباب" not in out


def test_speaker_label_removal_keeps_dialogue_dash():
    out = strip_hi_artifacts(_srt("- JOHN: Hello\n- MARY: Hi"))
    assert "- Hello" in out and "- Hi" in out


# --------------------------------------------------------------------------- #
# Release groups
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name,group",
    [
        ("Mad.Max.Fury.Road.2015.1080p.BluRay.YIFY", "YIFY"),
        ("Lost.S01E01.720p.HDTV", None),
        ("Show.S01E01.720p.HDTV.x264-LOL", "LOL"),  # the "-GROUP" tail still counts
    ],
)
def test_title_words_are_not_release_groups(name, group):
    assert parse_release_metadata(name)["group"] == group
