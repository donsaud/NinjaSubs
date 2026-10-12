"""Internal ASS reference conversion preserves existing converter semantics."""
import io
import zipfile

import pytest

from app.services.sync.decode import decode_payload
from app.utils.ass_converter import convert_ass_to_srt_bytes

ASS = b"[Script Info]\nScriptType: v4.00+\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 0,0:00:10.00,0:00:12.00,Default,,0,0,0,,Hello\nDialogue: 0,0:00:20.00,0:00:22.00,Default,,0,0,0,,World\n"


@pytest.mark.parametrize("archived", [False, True])
def test_raw_or_selected_ass_reference_uses_existing_converter(archived):
    raw = ASS
    if archived:
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("Show.S08E01.ass", ASS)
            archive.writestr("Show.S08E02.ass", ASS.replace(b"Hello", b"Wrong"))
        raw = stream.getvalue()
    result = decode_payload(raw, "Show.S08.BluRay.srt", 8, 1, min_bytes=1)
    assert result == convert_ass_to_srt_bytes(ASS, apply_rtl=False)
    assert b"00:00:10,000 --> 00:00:12,000" in result
    assert b"Dialogue:" not in result


def test_srt_reference_bytes_are_preserved():
    raw = b"1\n00:00:10,000 --> 00:00:12,000\nHello\n"
    assert decode_payload(raw, "Show.S08E01.srt", 8, 1, min_bytes=1) == raw


def test_failed_ass_conversion_preserves_original_for_fail_closed_parsing():
    raw = b"[Script Info]\n[Events]\nDialogue: malformed\n"
    assert decode_payload(raw, "Show.S08E01.ass", 8, 1, min_bytes=1) == raw
