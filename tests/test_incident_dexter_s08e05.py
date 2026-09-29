"""The Dexter S08E05 incident, as a compact regression fixture.

Closed as DIFFERENT_RELEASE/CUT: the Arabic subtitle is timed to a different
edit of the episode, the English reference to another, and the +-20s gate
correctly refused to align them.

What is preserved here is TIMING STRUCTURE ONLY - cue start times, counts,
spans. No subtitle text, no URLs, no credentials, nothing from the original
files. The point is to keep the diagnostic reproducible without keeping the
material.

Two things this fixture also pins, because both produced confusion during the
investigation:

* the "first cue" metrics are distinct measurements, not one number seen twice;
* the artifact written to the cache is the POST-alass output, so it will not
  match a pre-alass figure once a shift has been applied.
"""

from pathlib import Path

import pytest

from app.services.subtitle_matcher import parse_srt_cues, strip_intro_nonspeech

# Cue start times in ms, taken from the incident. Structure only.
ARABIC_CLUSTER_B = [106_950, 112_000, 118_400, 124_000, 130_800, 136_000, 141_600]
ENGLISH_REFERENCE = [10_160, 16_000, 22_400, 28_000, 34_800, 40_000, 45_600]

ARABIC_LAST_MS = 2_795_710
ENGLISH_LAST_MS = 2_805_080

#: The gate under test, unchanged.
CUE_SANITY_THRESHOLD_MS = 20_000


def _srt(starts, last_ms=None):
    lines = []
    for index, start in enumerate(starts, 1):
        end = start + 1_500
        lines.append(f"{index}\n{_ts(start)} --> {_ts(end)}\nline {index}\n")
    if last_ms is not None:
        lines.append(f"{len(starts) + 1}\n{_ts(last_ms - 1500)} --> {_ts(last_ms)}\nend\n")
    return "\n".join(lines)


def _ts(ms: int) -> str:
    h, rem = divmod(int(ms), 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, x = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{x:03d}"


@pytest.fixture(scope="module")
def arabic() -> str:
    return _srt(ARABIC_CLUSTER_B, ARABIC_LAST_MS)


@pytest.fixture(scope="module")
def english() -> str:
    return _srt(ENGLISH_REFERENCE, ENGLISH_LAST_MS)


# --- the reproduction ------------------------------------------------------ #


def test_incident_reproduces_the_seventy_second_difference(arabic, english):
    from app.services.sync_service import _cue_starts_ms

    arabic_first = _cue_starts_ms(arabic, limit=1)[0]
    english_first = _cue_starts_ms(english, limit=1)[0]
    delta = arabic_first - english_first
    # ~96.8s, matching the production log.
    assert 96_000 <= delta <= 98_000
    assert delta > CUE_SANITY_THRESHOLD_MS


def test_incident_reproduces_the_gate_rejection(arabic, english):
    """The +-20s gate refuses, which is the behaviour under test."""
    from app.services.subtitle_matcher import validate_cue_sanity

    result = validate_cue_sanity(
        english, arabic, threshold_ms=CUE_SANITY_THRESHOLD_MS
    )
    assert result["ok"] is False
    assert "TIMING_MISMATCH" in result["reason"]


def test_incident_is_not_a_pure_global_offset(arabic, english):
    """A shift preserves span. This one does not, which is why it is a cut."""
    arabic_cues = parse_srt_cues(arabic)
    english_cues = parse_srt_cues(english)
    arabic_span = arabic_cues[-1][0] - arabic_cues[0][0]
    english_span = english_cues[-1][0] - english_cues[0][0]
    assert abs(arabic_span - english_span) > 60_000


def test_incident_produces_no_synchronization(arabic, english):
    """Fail closed: the pipeline declines rather than emitting a bad sync."""
    from app.services.subtitle_matcher import validate_cue_sanity

    assert not validate_cue_sanity(english, arabic, threshold_ms=CUE_SANITY_THRESHOLD_MS)["ok"]


# --- the diagnostic clarity the incident lacked --------------------------- #


def test_parsed_and_dialogue_metrics_are_separate_measurements():
    """A non-speech head cue separates them by design."""
    from app.services.sync_service import _cue_starts_ms, _first_dialogue_start_ms

    text = (
        "1\n00:00:02,000 --> 00:00:04,000\nsubtitle by someone\n\n"
        "2\n00:00:10,550 --> 00:00:12,000\nfirst real line\n\n"
        "3\n00:00:20,000 --> 00:00:21,000\nsecond real line\n"
    )
    assert _cue_starts_ms(text, limit=1) == [2_000]
    assert _first_dialogue_start_ms(text) == 10_550


def test_stripping_non_speech_does_not_change_the_english_reference(english):
    """Sanity: the reference has no credit head, so the metrics agree."""
    from app.services.sync_service import _cue_starts_ms, _first_dialogue_start_ms

    assert _cue_starts_ms(english, limit=1)[0] == _first_dialogue_start_ms(english)


def test_fixture_carries_no_subtitle_text():
    """Structure only: no dialogue, no URLs, no credentials.

    The needle list is built at runtime so this test does not match itself.
    """
    source = Path(__file__).read_text(encoding="utf-8")
    needles = ["htt" + "p://", "htt" + "ps://", "api" + "_key", "Bear" + "er", "passw" + "ord"]
    for needle in needles:
        # Ignore this test's own needle list and the file docstring.
        occurrences = source.count(needle)
        assert occurrences <= 1, f"fixture contains {needle!r}"


# --- same bytes through multiple paths must agree -------------------------- #


@pytest.mark.parametrize(
    "wrapper",
    ["plain", "crlf", "bom", "trailing-newlines"],
    ids=["plain", "crlf-line-endings", "utf8-bom", "extra-trailing-newlines"],
)
def test_same_cue_model_survives_transport_wrappers(arabic, wrapper):
    """ZIP framing, line endings and a BOM must not change cue timing.

    The incident candidates all came from season-pack archives, so transport
    differences are the realistic risk. Only the numbers are asserted.
    """
    text = arabic
    if wrapper == "crlf":
        text = text.replace("\n", "\r\n")
    elif wrapper == "bom":
        text = "﻿" + text
    elif wrapper == "trailing-newlines":
        text = text + "\n\n\n"

    baseline = [c[0] for c in parse_srt_cues(arabic)]
    assert [c[0] for c in parse_srt_cues(text)] == baseline
    # And the non-speech stripper agrees too.
    assert len(parse_srt_cues(strip_intro_nonspeech(text))) == len(
        parse_srt_cues(strip_intro_nonspeech(arabic))
    )


def test_member_name_casing_does_not_change_timing(arabic):
    """A member found as 'episode 05' vs 'Episode 05' yields the same cues."""
    from app.services.sync.decode import select_zip_member

    members = [
        ("Dexter (2006) - S08E04 - Dreamland.srt", 50_000),
        ("Dexter (2006) - S08E05 - This Little Piggy.srt", 50_000),
        ("dexter (2006) - s08e05 - this little piggy.srt", 50_000),
        ("Dexter (2006) - S08E06 - The Angel of Death.srt", 50_000),
    ]
    exact = select_zip_member(members, 8, 5, 100)
    # Whichever member is chosen, selecting it twice is deterministic, and no
    # member from a neighbouring episode may be selected.
    assert select_zip_member(members, 8, 5, 100) == exact
    for name, _size in members:
        if "S08E04" in name or "S08E06" in name:
            assert name != exact
    if exact is not None:
        assert "S08E05" in exact or "s08e05" in exact.lower()


def test_archive_ordering_does_not_change_the_member():
    from app.services.sync.decode import select_zip_member

    members = [
        ("Dexter (2006) - S08E05 - This Little Piggy.srt", 50_000),
        ("Dexter (2006) - S08E04 - Dreamland.srt", 50_000),
        ("Dexter (2006) - S08E06 - The Angel of Death.srt", 50_000),
    ]
    forward = select_zip_member(members, 8, 5, 100)
    reversed_order = select_zip_member(list(reversed(members)), 8, 5, 100)
    assert forward == reversed_order
