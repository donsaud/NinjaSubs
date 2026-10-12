"""Tests for two_point_sync.sanitizer."""

from __future__ import annotations

import pathlib

import pytest

from tests.two_point_sync._helpers import make_cue
from two_point_sync.parser import Cue, parse_srt_bytes
from two_point_sync.sanitizer import classify_all, classify_cue, visible_text

FILM_START = 0
FILM_END = 6_000_000


def _cue(text: str, start_ms: int = 600_000) -> Cue:
    return make_cue(1, start_ms, start_ms + 2_000, text)


def test_website_advertisement_ineligible() -> None:
    for text in (
        "Download more subtitles at www.example-subtitles.com",
        "Get it from https://subscene.example/movie",
        "ترجمات أكثر على موقع example.net للتحميل",
    ):
        result = classify_cue(_cue(text), film_start_ms=FILM_START, film_end_ms=FILM_END)
        assert result.eligible is False
        assert result.confidence == 0.0


def test_translation_credits_ineligible() -> None:
    for text in (
        "ترجمة صلاح سليم",
        "قام بسحب الترجمة OzOz",
        "Translated by Example Team",
        "Synced bySomeone",
    ):
        result = classify_cue(_cue(text), film_start_ms=FILM_START, film_end_ms=FILM_END)
        assert result.eligible is False


def test_formatting_only_cues_excluded() -> None:
    for text in ("{\\an8}", "<i></i>", "\u202b", "  "):
        result = classify_cue(_cue(text), film_start_ms=FILM_START, film_end_ms=FILM_END)
        assert result.eligible is False


def test_counting_fragment_ineligible_but_word_counting_eligible() -> None:
    numeric = classify_cue(_cue("3، 4"), film_start_ms=FILM_START, film_end_ms=FILM_END)
    assert numeric.eligible is False
    assert "counting_or_numbering_fragment" in numeric.reasons

    words = classify_cue(
        _cue("واحد، اثنان، ثلاثة، هيا نبدأ"),
        film_start_ms=FILM_START,
        film_end_ms=FILM_END,
    )
    assert words.eligible is True


def test_genuine_dialogue_near_edges_remains_eligible() -> None:
    # First 30 seconds and last two minutes are warnings, not deletions.
    early = classify_cue(
        _cue("أنا آسف، لم أقصد التأخير", start_ms=5_000),
        film_start_ms=FILM_START,
        film_end_ms=FILM_END,
    )
    late = classify_cue(
        _cue("سأخبرك متى تبدأ العزف الآن", start_ms=FILM_END - 60_000),
        film_start_ms=FILM_START,
        film_end_ms=FILM_END,
    )
    assert early.eligible is True
    assert late.eligible is True
    assert any("film_start" in reason for reason in early.reasons)
    assert any("film_end" in reason for reason in late.reasons)


def test_long_dialogue_not_deleted() -> None:
    long_text = "الحوار الطويل يستمر هنا " * 20
    result = classify_cue(
        make_cue(1, 600_000, 612_000, long_text),
        film_start_ms=FILM_START,
        film_end_ms=FILM_END,
    )
    assert result.eligible is True


def test_standalone_sound_effect_ineligible() -> None:
    result = classify_cue(_cue("(ضحك)"), film_start_ms=FILM_START, film_end_ms=FILM_END)
    assert result.eligible is False


def test_bracketed_speaker_prefix_with_dialogue_eligible() -> None:
    result = classify_cue(
        _cue("(أندرو نيمان) يا سيدي، أنا مستعد"),
        film_start_ms=FILM_START,
        film_end_ms=FILM_END,
    )
    assert result.eligible is True


def test_arabic_text_handled_consistently() -> None:
    result = classify_cue(
        _cue("‫- ماذا تفعل؟\n‫- سأخبرك متى تبدأ"),
        film_start_ms=FILM_START,
        film_end_ms=FILM_END,
    )
    assert result.eligible is True
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.reasons, tuple)


def test_classify_all_does_not_mutate_originals() -> None:
    cues = [
        make_cue(1, 600_000, 602_000, "الحوار الأول هنا"),
        make_cue(2, 700_000, 702_000, "www.ads.example"),
    ]
    snapshot = [(cue.number, cue.start_ms, cue.end_ms, cue.text) for cue in cues]
    classify_all(cues)
    assert [(cue.number, cue.start_ms, cue.end_ms, cue.text) for cue in cues] == snapshot


def test_whiplash_counting_fragment_not_terminal_eligible() -> None:
    fixture = (
        pathlib.Path(__file__).resolve().parents[1]
        / "fixtures"
        / "whiplash"
        / "Whiplash (2014) WEBRip.iTunes.ar(1).srt"
    )
    if not fixture.exists():
        pytest.skip("Whiplash fixture unavailable")
    cues = parse_srt_bytes(fixture.read_bytes())
    scores = classify_all(cues)
    by_text = {visible_text(cue.text): score for cue, score in zip(cues, scores, strict=False)}
    assert by_text["3، 4"].eligible is False
    # Genuine neighbouring dialogue stays eligible.
    dialogue_scores = [
        score
        for cue, score in zip(cues, scores, strict=False)
        if visible_text(cue.text).startswith("سأخبرك")
    ]
    assert dialogue_scores and all(score.eligible for score in dialogue_scores)
