"""Tests for in-dialogue HI artifact stripping (strip_hi_artifacts)."""

from app.utils.cleaners import strip_hi_artifacts, strip_hi_artifacts_bytes


def _srt(*cues: tuple[str, str, str]) -> str:
    blocks = [f"{i}\n{start} --> {end}\n{text}" for i, (start, end, text) in enumerate(cues, 1)]
    return "\n\n".join(blocks) + "\n"


def test_strips_english_bracketed_hi_but_keeps_dialogue():
    content = _srt(
        ("00:00:01,000", "00:00:03,000", "Hello there. [MUSIC PLAYING] How are you? (SIGHS)")
    )

    out = strip_hi_artifacts(content)

    assert "[MUSIC PLAYING]" not in out
    assert "(SIGHS)" not in out
    assert "Hello there." in out
    assert "How are you?" in out
    assert "00:00:01,000 --> 00:00:03,000" in out


def test_strips_arabic_bracketed_hi():
    content = _srt(("00:00:01,000", "00:00:03,000", "[صوت بكاء] أهلاً بك (موسيقى حزينة)"))

    out = strip_hi_artifacts(content)

    assert "صوت بكاء" not in out
    assert "موسيقى حزينة" not in out
    assert "أهلاً بك" in out


def test_removes_english_and_arabic_speaker_labels():
    content = _srt(
        ("00:00:01,000", "00:00:03,000", "JOHN: Hello there\nسارة: مرحبا"),
    )

    out = strip_hi_artifacts(content)

    assert "JOHN:" not in out
    assert "سارة:" not in out
    assert "Hello there" in out
    assert "مرحبا" in out


def test_keeps_lowercase_parenthetical_dialogue():
    content = _srt(("00:00:01,000", "00:00:03,000", "I think (I don't know) maybe."))

    out = strip_hi_artifacts(content)

    assert "(I don't know)" in out


def test_drops_cue_that_becomes_completely_empty():
    content = _srt(
        ("00:00:01,000", "00:00:02,000", "[MUSIC PLAYING]"),
        ("00:00:03,000", "00:00:04,000", "Real dialogue here"),
    )

    out = strip_hi_artifacts(content)

    assert "MUSIC PLAYING" not in out
    assert "Real dialogue here" in out
    # The empty cue's timing is removed with it.
    assert "00:00:01,000" not in out


def test_webvtt_structure_is_preserved():
    content = (
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:03.000\n"
        "(GUNSHOT) Run!\n"
    )

    out = strip_hi_artifacts(content)

    assert out.startswith("WEBVTT")
    assert "(GUNSHOT)" not in out
    assert "Run!" in out
    assert "00:00:01.000 --> 00:00:03.000" in out


def test_non_cue_content_is_returned_unchanged():
    content = "Plain text [MUSIC PLAYING] without cues.\n"
    assert strip_hi_artifacts(content) == content


def test_bytes_wrapper_handles_utf8():
    content = _srt(("00:00:01,000", "00:00:03,000", "[صوت] نص"))
    out = strip_hi_artifacts_bytes(content.encode("utf-8")).decode("utf-8")
    assert "صوت" not in out
    assert "نص" in out
