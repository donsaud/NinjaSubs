"""Tests for two_point_sync.parser."""

from __future__ import annotations

import pytest

from tests.two_point_sync._helpers import build_dialogue, make_cue
from two_point_sync.parser import (
    Cue,
    SubtitleParseError,
    format_srt,
    format_timestamp,
    parse_srt,
    parse_srt_bytes,
    parse_timestamp,
    validate_cues,
)


def test_parse_timestamp_comma_and_dot() -> None:
    assert parse_timestamp("01:02:03,456") == ((1 * 60 + 2) * 60 + 3) * 1000 + 456
    assert parse_timestamp("01:02:03.456") == parse_timestamp("01:02:03,456")


def test_parse_timestamp_rejects_invalid() -> None:
    with pytest.raises(SubtitleParseError):
        parse_timestamp("not-a-time")
    with pytest.raises(SubtitleParseError):
        parse_timestamp("00:61:00,000")
    with pytest.raises(SubtitleParseError):
        parse_timestamp("00:00:60,000")


def test_format_timestamp_round_trip() -> None:
    for value in (0, 1, 999, 1_000, 3_661_001, 5_610_800):
        assert parse_timestamp(format_timestamp(value)) == value


def test_format_timestamp_rejects_negative() -> None:
    with pytest.raises(SubtitleParseError):
        format_timestamp(-1)


def test_parse_and_format_round_trip_preserves_cues() -> None:
    cues = build_dialogue(5)
    text = format_srt(cues)
    reparsed = parse_srt(text)
    assert len(reparsed) == 5
    for original, parsed in zip(cues, reparsed, strict=False):
        assert parsed.number == original.number
        assert parsed.order == original.order
        assert parsed.start_ms == original.start_ms
        assert parsed.end_ms == original.end_ms
        assert parsed.text == original.text


def test_parse_preserves_arabic_text_and_tags() -> None:
    content = (
        "1\n00:02:22,640 --> 00:02:25,160\n"
        "\u202b- أنا آسف، آسف\n\u202b- لا، انتظر\n\n"
        "2\n00:02:30,840 --> 00:02:33,840\n"
        "{\\an8}مرحبا <i>بالعالم</i>\n"
    )
    cues = parse_srt(content)
    assert len(cues) == 2
    assert "أنا آسف" in cues[0].text
    assert "{\\an8}" in cues[1].text
    assert "<i>" in cues[1].text


def test_parse_handles_bom_and_crlf() -> None:
    cues = build_dialogue(3)
    payload = ("\ufeff" + format_srt(cues)).replace("\n", "\r\n")
    assert len(parse_srt(payload)) == 3
    assert len(parse_srt_bytes(payload.encode("utf-8"))) == 3


def test_parse_bytes_handles_bom_bytes() -> None:
    cues = build_dialogue(2)
    payload = format_srt(cues).encode("utf-8-sig")
    assert payload[:3] == b"\xef\xbb\xbf"
    assert len(parse_srt_bytes(payload)) == 2


def test_parse_rejects_empty_content() -> None:
    with pytest.raises(SubtitleParseError):
        parse_srt("")
    with pytest.raises(SubtitleParseError):
        parse_srt("   \n  ")
    with pytest.raises(SubtitleParseError):
        parse_srt_bytes(b"")


def test_parse_rejects_negative_duration() -> None:
    content = "1\n00:00:02,000 --> 00:00:01,000\nHello\n"
    with pytest.raises(SubtitleParseError):
        parse_srt(content)


def test_parse_rejects_malformed_timing_line() -> None:
    content = "1\n00:00:01 ??? 00:00:02\nHello\n"
    with pytest.raises(SubtitleParseError):
        parse_srt(content)


def test_parse_rejects_malformed_index_with_text() -> None:
    content = "oops\n00:00:01,000 --> 00:00:02,000\nHello\n"
    with pytest.raises(SubtitleParseError):
        parse_srt(content)


def test_parse_rejects_empty_cue_text() -> None:
    content = "1\n00:00:01,000 --> 00:00:02,000\n   \n"
    with pytest.raises(SubtitleParseError):
        parse_srt(content)


def test_parse_skips_stray_formatting_residue() -> None:
    cues = build_dialogue(2)
    text = format_srt(cues) + "\n\u202b\n"
    assert len(parse_srt(text)) == 2


def test_validate_detects_duplicates_and_overlaps() -> None:
    cues = [
        make_cue(1, 1_000, 2_000, "First line here"),
        make_cue(1, 1_000, 2_500, "Second line here"),
        make_cue(3, 2_400, 3_000, "Third line here"),
    ]
    kinds = {issue.kind for issue in validate_cues(cues)}
    assert "duplicate_number" in kinds
    assert "duplicate_timestamp" in kinds
    assert "overlapping_cues" in kinds


def test_validate_clean_cues_have_no_issues() -> None:
    assert validate_cues(build_dialogue(4)) == []


def test_validate_reports_negative_duration_without_mutating() -> None:
    cue = Cue(number=1, order=0, start_ms=5_000, end_ms=1_000, text="Broken timing")
    issues = validate_cues([cue])
    assert any(issue.kind == "negative_duration" for issue in issues)
    assert cue.start_ms == 5_000 and cue.text == "Broken timing"
