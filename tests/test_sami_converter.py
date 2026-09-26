"""Unit tests for SAMI (.smi) to SRT conversion."""

from app.utils.sami_converter import convert_sami_to_srt


def _sami(*cues: str) -> str:
    body = "".join(f"<SYNC Start={start}><P Class=ENCC>{text}\n" for start, text in cues)
    return f"<SAMI><HEAD><TITLE>t</TITLE></HEAD><BODY>\n{body}</BODY></SAMI>"


def test_basic_cues_use_next_start_as_end():
    out = convert_sami_to_srt(_sami((1000, "Hello there"), (3500, "General Kenobi")))
    assert out == (
        "1\n00:00:01,000 --> 00:00:03,500\nHello there\n\n"
        "2\n00:00:03,500 --> 00:00:05,500\nGeneral Kenobi\n"
    )


def test_empty_and_nbsp_cues_dropped_with_renumbering():
    out = convert_sami_to_srt(
        _sami((1000, "First"), (2000, "&nbsp;"), (3000, "   "), (4000, "Second"))
    )
    assert out == (
        "1\n00:00:01,000 --> 00:00:02,000\nFirst\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\nSecond\n"
    )


def test_entities_tags_and_breaks_cleaned():
    out = convert_sami_to_srt(
        _sami((1000, "Fish &amp; Chips<br><i>on tour</i>"))
    )
    assert out == "1\n00:00:01,000 --> 00:00:03,000\nFish & Chips\non tour\n"


def test_english_class_preferred_over_other_languages():
    sami = (
        "<SAMI><BODY>\n"
        '<SYNC Start=1000><P Class=FRCC>Bonjour\n<P Class=ENCC>Hello\n'
        '<SYNC Start=3000><P Class=ENCC>Bye\n'
        "</BODY></SAMI>"
    )
    out = convert_sami_to_srt(sami)
    assert "Bonjour" not in out
    assert "Hello" in out and "Bye" in out


def test_hour_timestamps_and_last_cue_tail():
    out = convert_sami_to_srt(_sami((3723000, "Late"), (3728000, "Last")))
    assert "01:02:03,000 --> 01:02:08,000" in out
    assert "01:02:08,000 --> 01:02:10,000" in out


def test_non_sami_returns_empty():
    assert convert_sami_to_srt("") == ""
    assert convert_sami_to_srt("1\n00:00:01,000 --> 00:00:02,000\nHi\n") == ""
    assert convert_sami_to_srt("<SAMI><BODY></BODY></SAMI>") == ""
