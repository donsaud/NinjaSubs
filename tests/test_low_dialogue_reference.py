"""Low-dialogue references: sparse is not the same as unusable.

A long-form film carries real music and silence between lines, so its dialogue
coverage estimate is low even when it holds an enormous amount of timing
evidence. The old gate read that estimate as a structural verdict and discarded
such a reference as ``reference_invalid``; these tests pin the corrected rule
and, just as importantly, pin the cases that must still be refused.
"""

from __future__ import annotations

import pytest

from app.services.subtitle_matcher import parse_srt_cues
from app.services.sync.alignment import AlignmentAnalyzer
from app.services.sync.reference import (
    DIALOGUE_PROBE_REGIONS,
    LOW_COVERAGE_MIN_DIALOGUE_CUES,
    MIN_DIALOGUE_COVERAGE,
    MIN_DIALOGUE_REGIONS,
    MIN_REFERENCE_CUES,
    DialogueProfile,
    analyze_reference_health,
)


def srt(rows: list[tuple[int, int, str]]) -> str:
    def ts(ms: int) -> str:
        h, ms = divmod(ms, 3_600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    return "\n\n".join(
        f"{i}\n{ts(start)} --> {ts(end)}\n{text}" for i, (start, end, text) in enumerate(rows, 1)
    ) + "\n"


FILM_SPAN_MS = 5_836_983  # 97 minutes, the shape that triggered this


def spread_over_film(count: int, *, step_ms: int | None = None, start_ms: int = 5_000):
    """`count` dialogue cues spread evenly across a feature-length runtime."""
    step = step_ms or (FILM_SPAN_MS // count)
    return [(start_ms + i * step, start_ms + i * step + 1_500, f"a spoken line number {i}")
            for i in range(count)]


# --------------------------------------------------------------------------- #
# A. sparse but well distributed -> usable                                       #
# --------------------------------------------------------------------------- #


def test_a_sparse_reference_with_distributed_anchors_is_usable():
    """The production shape: 97 minutes, 903 dialogue cues, coverage 0.31.

    Coverage is far below the floor and always will be for anything that runs
    long, but the dialogue reaches every part of the timeline and there are far
    more of it than the bare minimum for any claim. That is usable evidence.
    """
    health = analyze_reference_health(srt(spread_over_film(903)))

    assert health.dialogue_coverage is not None
    assert health.dialogue_coverage < MIN_DIALOGUE_COVERAGE, "fixture must stay sparse"
    assert health.profile is DialogueProfile.LOW_DIALOGUE
    assert health.dialogue_regions == DIALOGUE_PROBE_REGIONS
    assert health.dialogue_cues >= LOW_COVERAGE_MIN_DIALOGUE_CUES
    assert health.healthy is True, health.reasons
    # The measurement is still reported, never silently dropped.
    assert any("covers" in reason for reason in health.reasons)


# --------------------------------------------------------------------------- #
# D. ordinary dense content -> unchanged                                        #
# --------------------------------------------------------------------------- #


def test_d_ordinary_dense_content_is_unaffected():
    """Coverage above the floor keeps the standard profile and healthy verdict."""
    health = analyze_reference_health(srt(spread_over_film(500, step_ms=2_000)))

    assert health.dialogue_coverage is not None
    assert health.dialogue_coverage >= MIN_DIALOGUE_COVERAGE
    assert health.profile is DialogueProfile.STANDARD_DIALOGUE
    assert health.healthy is True, health.reasons


def test_d_dense_short_span_is_unchanged_by_the_new_region_rule():
    """A normal episode fills every region, so the region gate never fires."""
    health = analyze_reference_health(srt(spread_over_film(400, step_ms=2_000)))
    assert health.dialogue_regions >= MIN_DIALOGUE_REGIONS
    assert health.healthy is True


# --------------------------------------------------------------------------- #
# C. sparse AND thin / bunched -> conservative refusal                          #
# --------------------------------------------------------------------------- #


def test_c_low_coverage_with_too_few_cues_is_refused():
    """Sparse is allowed; sparse and thin is not.

    Below :data:`MIN_REFERENCE_CUES` the absolute floor already refuses it.
    """
    health = analyze_reference_health(srt(spread_over_film(8, step_ms=97_000)))

    assert health.profile is DialogueProfile.LOW_DIALOGUE
    assert health.healthy is False
    assert any("need 12" in reason or f"need {MIN_REFERENCE_CUES}" in reason
               for reason in health.reasons)


def test_c_low_coverage_with_a_bare_minimum_of_cues_is_still_refused():
    """The regression guard.

    Exactly ``MIN_REFERENCE_CUES`` cues spread thin used to be refused by the
    coverage floor alone. It must stay refused, because a dozen lines cannot
    corroborate a feature-length alignment -- the count clears the floor for
    *any* claim, not the higher bar a low-coverage reference has to meet.
    """
    health = analyze_reference_health(srt(spread_over_film(MIN_REFERENCE_CUES, step_ms=97_000)))

    assert health.dialogue_cues == MIN_REFERENCE_CUES
    assert health.profile is DialogueProfile.LOW_DIALOGUE
    assert health.healthy is False
    assert any("too few anchors" in reason for reason in health.reasons), health.reasons


def test_c_anchors_bunched_into_one_stretch_are_refused():
    """Many cues in one small part of the film anchor nothing outside it.

    The cue count here clears :data:`LOW_COVERAGE_MIN_DIALOGUE_CUES` on purpose,
    so the *only* thing that can refuse this is the distribution rule.
    """
    rows = [(i * 800, i * 800 + 700, f"early line {i}") for i in range(40)]
    rows += [(900_000 + i * 3_000, 900_000 + i * 3_000 + 2_000, f"tail line {i}")
             for i in range(3)]
    health = analyze_reference_health(srt(rows))

    assert health.dialogue_cues >= LOW_COVERAGE_MIN_DIALOGUE_CUES
    assert health.dialogue_regions < MIN_DIALOGUE_REGIONS
    assert health.healthy is False
    assert any("timeline regions" in reason for reason in health.reasons)
    # Refused for distribution, not for being thin.
    assert not any("too few anchors" in reason for reason in health.reasons)


def test_c_credits_only_reference_is_refused():
    rows = [(i * 4_000, i * 4_000 + 3_000, "[Music]") for i in range(40)]
    health = analyze_reference_health(srt(rows))

    assert health.dialogue_cues == 0
    assert health.healthy is False
    assert any("dialogue cue" in reason for reason in health.reasons)


@pytest.mark.parametrize("bad", [None, "", "not a subtitle"])
def test_c_unparseable_reference_is_refused(bad):
    assert analyze_reference_health(bad).healthy is False


# --------------------------------------------------------------------------- #
# E. safety: the alignment verdict itself is unchanged                          #
# --------------------------------------------------------------------------- #


def test_e_a_badly_misaligned_sparse_subtitle_is_still_rejected():
    """E. The gate change must not make the verifier more forgiving.

    The reference is a healthy low-dialogue one, so this exercises the verifier
    alone. An output that carries a plausible constant shift but whose cues are
    scattered by seconds must still be refused for the same reason dense content
    would be, because none of the thresholds moved.
    """
    reference_text = srt(spread_over_film(200, step_ms=30_000))
    target = parse_srt_cues(reference_text)
    # A ~13s correction (the real-world magnitude) with 4s of scatter on top, so
    # the residual tail is far past the 2000ms limit.
    wrong = [
        (start + 13_000 + (3_500 if index % 3 == 0 else -2_000),
         end + 13_000 + (3_500 if index % 3 == 0 else -2_000), text)
        for index, (start, end, text) in enumerate(target)
    ]

    evaluation = AlignmentAnalyzer().analyze(
        target, wrong, parse_srt_cues(reference_text),
        alass_applied=True, alass_successful=True,
    )

    assert evaluation.sync_state.value == "unverified"
    assert evaluation.rejection_reason is not None


def test_e_p95_and_mad_limits_are_unchanged_by_a_low_dialogue_reference():
    """The thresholds are global constants; this pins that nothing moved them."""
    from app.services.sync import alignment

    assert alignment.MAX_P95_MS_FOR_STABLE == 2000.0
    assert alignment.MAX_MAD_MS_FOR_STABLE == 800.0
