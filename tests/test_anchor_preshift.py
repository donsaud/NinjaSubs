"""The anchor-correspondence pre-shift: propose a shift, then prove it.

Two failure modes are pinned here, both observed on real Whiplash subtitles.

1. A first-cue delta is a hypothesis, not a measurement. Against a different
   English edition of the same film it pointed the wrong way and applying it
   made alignment worse.

2. Multi-region anchor corroboration is necessary but not sufficient. The
   production gate accepts the bad hypothesis too -- dispersion measures
   agreement, and a wrong answer can agree with itself perfectly. Acceptance
   therefore also requires the shift to measurably improve correspondence
   against the reference.
"""

from __future__ import annotations

import itertools

import pytest

from app.services.subtitle_matcher import parse_srt_cues
from app.services.sync.anchor_preshift import (
    ANCHOR_PRESHIFT_MAX_ABS_MS,
    ANCHOR_PRESHIFT_MIN_ABS_MS,
    apply_anchor_preshift,
    decide_anchor_preshift,
    preshift_text,
    render_srt_cues,
)

#: Mean gap ~5.9s, which is what dialogue in a feature actually runs at.
#:
#: The spacing is load-bearing, not cosmetic. ``pair_cues`` matches each cue to
#: the nearest unused reference cue within ``RESIDUAL_TOLERANCE_MS`` (5000ms),
#: and the anchor gate searches within ``LARGE_OFFSET_ANCHOR_TOLERANCE_MS``
#: (2500ms). A fixture denser than those tolerances makes every anchor snap to
#: whichever neighbour happens to be closest, so an estimate lands up to half a
#: cue gap away from the truth -- an artefact of the fixture, not a property of
#: the estimator.
_STEPS = itertools.cycle([4200, 6800, 3100, 9100, 5200, 7400, 4600, 8300, 3900, 6100])


def track(count: int = 380, *, music_every: int = 0) -> list[tuple[int, int, str]]:
    """A film-length track with irregular, human-like cue spacing.

    ``music_every`` inserts non-speech cues. They are placed identically in
    every encode of a film, which makes them the most reliable offset anchors
    available -- see the module docstring.
    """
    cues: list[tuple[int, int, str]] = []
    position = 0
    for index in range(count):
        duration = next(_STEPS)
        text = "[Music]" if music_every and index % music_every == 0 else f"line {index} spoken here now"
        cues.append((position, position + duration - 100, text))
        position += duration
    return cues


def shifted(cues, delta_ms: int):
    return [(a + delta_ms, b + delta_ms, t) for a, b, t in cues]


def median_abs_residual(target, reference) -> float:
    from app.services.sync.alignment import RESIDUAL_TOLERANCE_MS, pair_cues

    pairing = pair_cues(list(target), list(reference), tolerance_ms=RESIDUAL_TOLERANCE_MS)
    residuals = sorted(abs(m.delta_ms) for m in pairing.matches)
    if not residuals:
        return float("inf")
    mid = len(residuals) // 2
    if len(residuals) % 2:
        return float(residuals[mid])
    return (residuals[mid - 1] + residuals[mid]) / 2.0


# --- the good case ------------------------------------------------------------


@pytest.mark.parametrize("delta", [6000, 8000, 11000, -6000])
def test_a_real_constant_offset_is_corrected(delta):
    reference = track()
    target = shifted(reference, delta)

    decision = decide_anchor_preshift(target, reference)

    assert decision.accepted is True, decision.reason
    assert decision.preshift is not None
    before = median_abs_residual(target, reference)
    after = median_abs_residual(
        apply_anchor_preshift(target, decision.preshift.shift_ms), reference
    )
    assert after < before
    assert decision.preshift.pairs_after >= decision.preshift.pairs_before
    assert decision.preshift.median_abs_after_ms < decision.preshift.median_abs_before_ms


def test_an_offset_inside_the_pairing_tolerance_abstains_rather_than_guessing():
    """Below ``RESIDUAL_TOLERANCE_MS`` the outcome metric cannot discriminate.

    ``pair_cues`` matches each cue to the nearest unused reference cue within
    5000ms, so a 4000ms error still pairs almost everything and the median
    residual barely moves. The anchors then propose something the metric cannot
    confirm, so the gate abstains. That is the safe direction: no shift, alass
    still runs, and nothing is asserted that was not measured.
    """
    from app.services.sync.alignment import RESIDUAL_TOLERANCE_MS

    assert RESIDUAL_TOLERANCE_MS == 5000
    reference = track()
    target = shifted(reference, 4000)

    decision = decide_anchor_preshift(target, reference)

    assert decision.accepted is False
    assert decision.preshift is None
    assert "did not reduce the median residual" in decision.reason


def test_correcting_a_real_offset_makes_the_subtitle_genuinely_aligned():
    """The production outcome: 11s out becomes aligned, not merely different."""
    reference = track()
    target = shifted(reference, 11_000)

    decision = decide_anchor_preshift(target, reference)

    assert decision.accepted is True
    aligned = apply_anchor_preshift(target, decision.preshift.shift_ms)
    assert median_abs_residual(aligned, reference) < 200


# --- the bad cases the outcome test exists to catch ---------------------------


def test_a_first_cue_delta_that_points_the_wrong_way_is_refused():
    """A different English edition opens on a different line.

    The opening dialogue disagrees by a plausible-looking amount, so the seed
    lands inside the band and anchors are found -- but the shift makes
    correspondence worse, so it must not be applied.
    """
    reference = track()
    # Same film, but the target's dialogue is re-segmented near the head so the
    # first line is 9s away from the reference's first line, while the rest of
    # the track is only 2s out.
    target = list(track())
    target = [(a - 7000 if a < 240_000 else a - 2000, b - 7000 if b < 240_000 else b - 2000, t)
              for a, b, t in target]

    decision = decide_anchor_preshift(target, reference)

    assert decision.accepted is False
    assert decision.preshift is None
    assert decision.reason


def test_a_self_consistent_but_wrong_hypothesis_is_refused():
    """Anchors that agree with each other are not evidence of being right.

    The target is two unrelated films spliced together. Anchors still agree, so
    only the outcome test can catch it.
    """
    first = track(190)
    second = track(190)
    spliced = first + [(a + 3_600_000, b + 3_600_000, t) for a, b, t in second]

    decision = decide_anchor_preshift(spliced, track())

    assert decision.accepted is False


# --- the band -----------------------------------------------------------------


def test_an_offset_too_small_to_matter_is_left_to_alass():
    reference = track()
    decision = decide_anchor_preshift(shifted(reference, 400), reference)

    assert decision.accepted is False
    assert "outside" in decision.reason


def test_an_offset_beyond_the_band_is_refused():
    """A difference of minutes is a different cut, not a shifted encode."""
    reference = track()
    decision = decide_anchor_preshift(shifted(reference, 140_000), reference)

    assert decision.accepted is False
    assert "outside" in decision.reason


def test_the_band_is_two_to_twenty_five_seconds():
    assert ANCHOR_PRESHIFT_MIN_ABS_MS == 2000
    assert ANCHOR_PRESHIFT_MAX_ABS_MS == 25_000


# --- insufficient evidence ----------------------------------------------------


def test_a_handful_of_cues_yields_no_decision():
    decision = decide_anchor_preshift(track(20), track())

    assert decision.accepted is False
    assert "not enough" in decision.reason


def test_a_track_with_no_dialogue_yields_no_decision():
    music_only = [(i * 3000, i * 3000 + 2500, "[Music]") for i in range(200)]
    decision = decide_anchor_preshift(music_only, music_only)

    assert decision.accepted is False


# --- non-speech cues are anchors, not noise -----------------------------------


def test_non_speech_cues_are_used_rather_than_filtered_out():
    """Music cues sit at the same instant in every encode, so they anchor well.

    Filtering them out (as the alignment-consistency gate does, for a different
    question) measurably degraded the real estimate: +11072ms -> +13116ms and
    42ms -> 1210ms median residual on the Whiplash pair.
    """
    reference = track(music_every=7)
    target = shifted(reference, 9000)

    decision = decide_anchor_preshift(target, reference)

    assert decision.accepted is True
    aligned = apply_anchor_preshift(target, decision.preshift.shift_ms)
    assert median_abs_residual(aligned, reference) < 200


# --- the text helpers ---------------------------------------------------------


def test_apply_shift_moves_every_cue():
    cues = [(1000, 2000, "a"), (3000, 4000, "b")]
    assert apply_anchor_preshift(cues, 500) == [(1500, 2500, "a"), (3500, 4500, "b")]


def test_apply_shift_of_zero_is_identity():
    cues = [(1000, 2000, "a")]
    assert apply_anchor_preshift(cues, 0) == cues


def test_cues_pushed_off_the_front_are_dropped_not_clamped():
    """A cue with no positive position has lost its place; clamping invents one."""
    cues = [(1000, 2000, "doomed"), (10_000, 11_000, "kept")]
    assert apply_anchor_preshift(cues, -5000) == [(5000, 6000, "kept")]


def test_render_round_trips_through_the_parser():
    cues = track(40)
    assert parse_srt_cues(render_srt_cues(cues)) == cues


def test_render_renumbers_from_one():
    rendered = render_srt_cues([(0, 1000, "a"), (2000, 3000, "b")])
    assert rendered.splitlines()[0] == "1"
    assert [b.splitlines()[0] for b in rendered.strip().split("\n\n")] == ["1", "2"]


def test_preshift_text_preserves_crlf():
    cues = track(40)
    text = render_srt_cues(cues, newline="\r\n")

    out = preshift_text(text, 5000)

    assert out is not None
    assert "\r\n" in out
    # The round trip is the real assertion: CRLF in, CRLF out, and the cues
    # still parse as separate blocks rather than merging into one.
    assert parse_srt_cues(out) == apply_anchor_preshift(cues, 5000)


def test_preshift_text_returns_none_when_nothing_survives():
    cues = [(1000, 2000, "a"), (2500, 3000, "b")]
    text = render_srt_cues(cues)

    assert preshift_text(text, -5000) is None


def test_preshift_text_ignores_a_zero_shift():
    text = render_srt_cues(track(40))
    assert preshift_text(text, 0) is None


def test_preshift_text_ignores_non_srt():
    assert preshift_text("not a subtitle", 5000) is None
