"""Structural ASS and RTL regressions; no real evidence is modified."""

import re

import pytest

from app.extractor import is_ass_subtitle
from app.main import _build_subtitle_response, _sync_subtitle_for_response
from app.services.sync.parser_fingerprint import semantic_sha
from app.services.sync_service import sanitize_subtitle
from app.utils.ass_converter import (
    _parse_ass_timestamp,
    convert_ass_result,
    convert_ass_to_srt,
    convert_ass_to_srt_bytes,
)
from app.utils.cleaners import CleanOptions, clean_subtitle
from app.utils.rtl import RLM, fix_rtl_punctuation, fix_rtl_punctuation_bytes


def ass(text, start="0:00:01.00", end="0:00:03.00"):
    return ("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")


@pytest.mark.parametrize("prefix", ["", "\ufeff", "[Script Info]\nScriptType: v4.00\n",
                                    "[V4 Styles]\n"])
def test_structural_ass_without_optional_headers(prefix):
    raw = prefix + ass("Readable")
    assert is_ass_subtitle(raw.encode())
    result = convert_ass_result(raw, False)
    assert result.converted and result.stats.cues_emitted == 1
    assert "Readable" in result.data


@pytest.mark.parametrize("text", ["Dialogue: literal text", "[Events]", "[Script Info]",
                                  "[Events]\nDialogue: literal text"])
def test_srt_literals_survive_every_conversion_routing_pass(text):
    srt = f"1\n00:00:01,000 --> 00:00:03,000\n{text}\n"
    assert not is_ass_subtitle(srt.encode())
    assert convert_ass_to_srt(srt) == srt
    assert sanitize_subtitle(srt) == srt
    response = _build_subtitle_response(srt.encode(), "mislabeled.ass", "ass", enable_rtl_fix=False,
                                       enable_ad_removal=False)
    assert response.body == srt.encode()
    assert response.media_type.startswith("application/x-subrip")


def test_converted_ass_dialogue_literal_remains_srt_on_second_pass():
    raw = ass("Dialogue: literal text [Events]").encode()
    converted = convert_ass_to_srt_bytes(raw, False)
    assert converted != raw and not is_ass_subtitle(converted)
    assert convert_ass_to_srt_bytes(converted, False) == converted


def test_cleaner_does_not_skip_srt_containing_ass_literal_tokens():
    raw = "1\n00:00:01,000 --> 00:00:03,000\nDialogue: literal  text\n[Events]\n"
    assert clean_subtitle(raw) == raw.replace("literal  text", "literal text")


def test_headerless_ssa_default_event_field_order():
    raw = "[Events]\nDialogue: Marked=0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Readable\n"
    assert is_ass_subtitle(raw.encode())
    assert convert_ass_result(raw, False).data == "1\n00:00:01,000 --> 00:00:03,000\nReadable\n"


@pytest.mark.parametrize("raw", ["[Script Info]\nTitle: metadata only\n",
                                "[Events]\nFormat: Start, End, Text\nComment: 0:00:01.00,0:00:03.00,comment\n",
                                ass("Readable", "bad", "bad"),
                                "[Script Info]\n[Events]\nDialogue: malformed\n"])
def test_failed_conversion_preserves_input_and_uses_native_response(raw):
    result = convert_ass_result(raw, False)
    assert not result.converted and result.data == raw
    assert convert_ass_to_srt_bytes(raw.encode()) == raw.encode()
    response = _build_subtitle_response(raw.encode(), "fallback.ass", "ass")
    assert response.body == raw.encode()
    assert response.media_type.startswith("text/x-ssa")
    assert sanitize_subtitle(raw) == ""


def test_requested_ass_with_ambiguous_unstructured_content_preserves_bytes():
    raw = b"Dialogue: malformed, readable text {literal note}\n"
    response = _build_subtitle_response(raw, "fallback.ass", "ass")
    assert response.body == raw and response.media_type.startswith("text/x-ssa")


@pytest.mark.asyncio
async def test_failed_ass_conversion_never_enters_sync(monkeypatch):
    async def forbidden(*args, **kwargs):
        pytest.fail("native ASS conversion failure entered sync")
    monkeypatch.setattr("app.main._maybe_sync_subtitle", forbidden)
    raw = ass("Readable", "bad", "bad").encode()
    assert await _sync_subtitle_for_response(raw, {}, {}, "probe", None, True, True) == raw


@pytest.mark.parametrize("fraction,ms", [("1", 100), ("12", 120), ("123", 123)])
@pytest.mark.parametrize("separator", [".", ",", ":"])
def test_timestamp_fractions_exact(fraction, ms, separator):
    assert _parse_ass_timestamp(f"123456:59:59{separator}{fraction}") == 123456 * 3600000 + 3599000 + ms


@pytest.mark.parametrize("timestamp", ["0:60:01.00", "0:00:60.00", "-1:00:00.00", "0:0:0.1234", "x"])
def test_bad_timestamp_rejected(timestamp):
    assert _parse_ass_timestamp(timestamp) is None
    result = convert_ass_result(ass("bad", timestamp) + ass("good"), False)
    assert result.stats.invalid_skipped == 1 and result.stats.cues_emitted == 1


@pytest.mark.parametrize("end", ["0:00:01.00", "0:00:00.99"])
def test_zero_or_negative_duration_never_emitted(end):
    result = convert_ass_result(ass("bad", end=end) + ass("good"), False)
    assert result.stats.invalid_skipped == 1
    assert "bad" not in result.data and result.stats.cues_emitted == 1


@pytest.mark.parametrize("text,expected", [
    (r"First\N\NLast\n\nEnd\hword", "First\nLast\nEnd word"),
    (r"Text {literal note} {ملاحظة}", "Text {literal note} {ملاحظة}"),
    (r"{\p1}m 0 0 l 1 1{\p0}spoken", "spoken"),
    (r"spoken{\p9}m 0 0", "spoken"),
    (r"before{\p1}m 0 0{\p0}after{\p2}l 1 1{\p0}end", "beforeafterend"),
    (r"Text {\i1}hello{\i0}", "Text hello"),
    (r"before{\i1{\b1}}after", "beforeafter"),
    (r"before{\pos(1,2)", "before"),
    (r"{\unknownthing42}spoken", "spoken"),
    (r"{\t(0,100,\p1\c&H0000FF&)}spoken", "spoken"),
    (r"{\an8\pos(1,2)\move(1,2,3,4)\fad(1,2)\fade(1,2,3,4,5,6,7)\blur3\bord2\shad1\fs40\fnArial\alpha&H00&\t(\fs40)\k10\K20\kf30\ko40}spoken", "spoken"),
    ("😀 e\u0301 عربية", "😀 e\u0301 عربية"),
    ("!", "!"),
])
def test_override_drawing_literal_unicode_and_multiline_policy(text, expected):
    result = convert_ass_result(ass(text), False)
    assert result.data == "1\n00:00:01,000 --> 00:00:03,000\n" + expected + "\n"
    assert convert_ass_to_srt(result.data, False) == result.data
    assert clean_subtitle(result.data).count(" --> ") == 1


@pytest.mark.parametrize("text", [r"{\p1}m 0 0 l 1 1", r"{\c&H0000FF&}",
                                  "<i></i>", "<font color=\"#FF0000\"></font>",
                                  r"{\c&H0000FF&}" + "\u200f\u200e"])
def test_empty_or_drawing_events_dropped_when_readable_events_exist(text):
    result = convert_ass_result(ass(text) + ass("retained"), False)
    assert result.stats.cues_emitted == 1
    assert result.stats.empty_visible + result.stats.drawing_only == 1
    assert "retained" in result.data


@pytest.mark.parametrize("count", [1000, 10000])
def test_large_event_files_no_text_loss(count):
    raw = ass(r"Line {literal} {\i1}spoken\Nsecond") * count
    result = convert_ass_result(raw, False)
    assert result.stats.cues_emitted == count
    assert result.data.count("Line {literal} spoken\nsecond") == count


@pytest.mark.parametrize("text", ["{" * 100000 + "visible", "readable{\\pos(" + "{" * 100000],
                         ids=["literal-braces", "unclosed-command"])
def test_long_malformed_override_no_recursion_or_text_loss(text):
    result = convert_ass_result(ass(text), False)
    assert result.converted
    assert ("visible" if text.startswith("{") else "readable") in result.data
    assert "\\pos" not in result.data


def test_large_blank_header_and_unclosed_literal_markup_are_bounded():
    text = "<font" * 10000 + "readable"
    raw = "\n" * 100000 + ass(text)
    result = convert_ass_result(raw, False)
    assert result.converted and result.data.endswith(text + "\n")


@pytest.mark.parametrize("prefix", ["<font" * 10000, "<i>" * 10000, "{" * 10000],
                         ids=["unclosed-font", "unclosed-emphasis", "literal-braces"])
def test_converter_with_optional_rtl_has_bounded_malformed_markup(prefix):
    result = convert_ass_result(ass(prefix + "مرحبا"), True)
    assert result.converted and prefix + "مرحبا" in result.data


def test_short_form_vtt_timestamp_cannot_be_overridden_by_literal_ass_header():
    raw = b"WEBVTT\n\n00:01.000 --> 00:03.000\n[Script Info]\nDialogue: literal\n"
    assert not is_ass_subtitle(raw)
    assert convert_ass_to_srt_bytes(raw) == raw


@pytest.mark.parametrize("next_start,end,expected_end", [
    (1000, 1300, 1300), (1000, 3000, 3000), (1200, 1300, 1200),
    (2000, 3000, 3000), (3000, 3000, 3000),
])
@pytest.mark.parametrize("converted", [False, True])
def test_overlap_clamp_preserves_strict_duration(next_start, end, expected_end, converted):
    from app.utils.ass_converter import _format_srt_timestamp
    a, b, c = map(_format_srt_timestamp, (end, next_start, next_start + 1000))
    srt = f"1\n00:00:01,000 --> {a}\nFirst\n\n2\n{b} --> {c}\nSecond\n"
    if converted:
        raw = ass("First", end="0:" + a[3:].replace(",", ".")) + ass(
            "Second", "0:" + b[3:].replace(",", "."), "0:" + c[3:].replace(",", "."))
        srt = convert_ass_to_srt(raw, False)
    out = clean_subtitle(srt, CleanOptions())
    assert f"00:00:01,000 --> {_format_srt_timestamp(expected_end)}" in out
    assert out.count(" --> ") == 2


BODIES = [
    "مرحبا بك", "مرحبا بك؟", "مرحبا بك.", "مرحبا بك!", "مرحبا بك...", "مرحبا بك،", "مرحبا بك:", "مرحبا بك؛",
    "مرحبا (بك)", "مرحبا [بك]", "مرحبا بك (جداً)", "صحيح، (فورت) لودرديل", "أجل، في الـ(كيز)",
    "(مرحبا بك)", "[مرحبا بك]", '"مرحبا بك"', "'مرحبا بك'", "«مرحبا بك»", "“مرحبا بك”", "‹مرحبا بك›",
    "<i>مرحبا بك</i>", "<b>مرحبا بك</b>", "<u>مرحبا بك</u>", "<i>مرحبا (بك)</i>",
    '<font color="#FF0000">مرحبا بك</font>', '<font color="#FF0000">مرحبا (بك)</font>',
    '<font color="#FF0000"><i>مرحبا بك</i></font>', '<i><font color="#FF0000">مرحبا بك</font></i>',
    '("مرحبا بك")', "«(مرحبا بك)»", '<i>"مرحبا بك"</i>', "<b>(مرحبا [بك])</b>",
    '"أنا (دب)، أترك رسالة"', '"صندوق بريد المستخدم ممتلئ"',
    "سأعود الساعة 5", "سأعود الساعة ٥", "الموعد 12:30", "الموعد ١٢:٣٠",
    "مرحبا John", "موعد Netflix اليوم", "الإصدار BluRay", "اذهب إلى L.A.",
]


@pytest.mark.parametrize("body", BODIES)
@pytest.mark.parametrize("dash", ["-", "–", "—"])
@pytest.mark.parametrize("space", [" ", "  ", "\t"])
def test_expanded_structural_dialogue_matrix(body, dash, space):
    fixed = fix_rtl_punctuation(body + space + dash)
    assert fixed.replace(RLM, "") == dash + " " + body
    assert fix_rtl_punctuation(fixed) == fixed
    correct = fix_rtl_punctuation(dash + " " + body)
    assert correct.replace(RLM, "") == dash + " " + body
    assert fix_rtl_punctuation(correct) == correct


@pytest.mark.parametrize("original", [
    "COVID-19", "2025-2026", "123 - 456", "١٢٣ - ٤٥٦", "اسم-مركب", "مرحبا - بك", "مرحبا بك-",
    "العربية - English", "English - العربية", "مرحبا - English", "A - B", "ABC -",
    "مرحبا بك --", "مرحبا بك - -", "مرحبا بك — —", "-- مرحبا بك", "- - مرحبا بك",
    "مرحبا بك (-", "(مرحبا بك -", '"مرحبا بك -', "«مرحبا بك -", "<i>مرحبا بك -",
    '<font color="#FF0000">مرحبا بك -', "<i><b>مرحبا بك</i> -", "<i class='x'>مرحبا بك</i> -",
])
def test_ambiguous_or_malformed_structure_never_invented(original):
    fixed = fix_rtl_punctuation(original)
    assert fixed.replace(RLM, "") == original
    assert fix_rtl_punctuation(fixed) == fixed


@pytest.mark.parametrize("mark", [RLM, "\u200e"])
@pytest.mark.parametrize("placement", ["prefix", "body_end", "end"])
def test_bidi_transparent_analysis_preserves_existing_marks(mark, placement):
    body = (mark if placement == "prefix" else "") + "مرحبا بك" + (mark if placement == "body_end" else "")
    original = body + " -" + (mark if placement == "end" else "")
    fixed = fix_rtl_punctuation(original)
    assert fixed.replace(RLM, "").replace("\u200e", "") == "- مرحبا بك"
    assert mark in fixed
    assert fix_rtl_punctuation(fixed) == fixed


def test_full_bytes_real_dialogue_tags_timing_and_newlines():
    bodies = ["صحيح، (فورت) لودرديل", '"أنا (دب)، أترك رسالة"', '<font color="#FF0000">مرحبا بك</font>']
    original = "\r\n\r\n".join(f"{i}\r\n00:00:0{i},000 --> 00:00:0{i},900\r\n{body} -"
                                for i, body in enumerate(bodies, 1)).encode()
    fixed = fix_rtl_punctuation_bytes(original)
    expected = original.decode()
    for body in bodies:
        expected = expected.replace(body + " -", "- " + body)
    assert fixed.decode().replace(RLM, "") == expected
    assert fix_rtl_punctuation_bytes(fixed) == fixed


def test_ass_color_to_served_srt_with_normal_rtl_pass():
    raw = ass(r"{\c&H0000FF&}مرحبا (بك) -").encode()
    converted = convert_ass_to_srt_bytes(raw, False)
    assert RLM.encode() not in converted
    response = _build_subtitle_response(converted, "source.ass", "ass", enable_ad_removal=False)
    text = response.body.decode()
    assert text.count(" --> ") == 1
    assert "00:00:01,000 --> 00:00:03,000" in text
    assert '<font color="#FF0000">' in text and text.count("</font>") == 1
    assert re.sub(r"<[^>]+>|\u200f", "", text).endswith("- مرحبا (بك)\n")
    assert "\\c" not in text and "{" not in text
    assert fix_rtl_punctuation_bytes(response.body) == response.body


def test_ass_routing_does_not_change_capture_srt_parser_semantics():
    # The audited legacy Capture V2 parser closure has no ASS/RTL dependency.
    assert semantic_sha() == "9e4a8192148b51785b71a9b16a70f619b0ffceb70b530162dc2364964bab0bff"


def test_old_sanitized_reference_cache_rejected_without_deleting_artifact(tmp_path):
    import hashlib
    import json

    from app.services.sync.cache import ReferenceDiskCache
    from app.services.sync.query import ReferenceQuery

    query = ReferenceQuery(imdb_id="tt0000001", target_sub_id="synthetic-probe")
    cache = ReferenceDiskCache(tmp_path, min_bytes=1)
    path = tmp_path / (query.cache_stem + "_probe_team.srt")
    payload = "1\n00:00:01,000 --> 00:00:03,000\nReadable\n"
    path.write_text(payload, encoding="utf-8")
    sidecar = path.with_suffix(".srt.json")
    metadata = {"kind": "team", "strict": True, "engine_version": 5,
                "content_sha": hashlib.sha256(payload.encode()).hexdigest()}
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    assert cache.get(query) is None
    assert path.read_text(encoding="utf-8") == payload
    metadata["engine_version"] = 6
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    assert cache.get(query) is not None
