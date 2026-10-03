"""The constant-shift exemption: p95 set aside only on positive evidence.

The Whiplash case that motivates this module:

    #3 HDA REMUX, correctly shifted by +11072ms
        median residual   42ms      <- genuinely aligned
        residual p95    3541ms      <- Arabic/English segmentation drift
        movement mad      0ms       <- one constant shift
        struct          0.89

The general verifier reads the p95 and refuses a well-aligned subtitle. These
tests pin the narrowness of the exemption that replaces it: every criterion must
hold, and every *other* refusal must still return the original.
"""

from __future__ import annotations

import itertools

import pytest

from app.services.sync.alignment import (
    CutVerdict,
    RejectionReason,
    SubtitleEvaluation,
    SyncState,
    VerificationAvailability,
)
from app.services.sync.anchor_preshift import AnchorPreshift, apply_anchor_preshift
from app.services.sync.constant_shift_exemption import (
    CONSTANT_SHIFT_MAX_MEDIAN_RESIDUAL_MS,
    ConstantShiftExemptionState,
    apply_exemption_to_evaluation,
    decide_constant_shift_exemption,
    measure_median_abs_residual,
)
from app.services.sync.large_offset_investigation import AlassOutputValidation

_STEPS = itertools.cycle([4200, 6800, 3100, 9100, 5200, 7400, 4600, 8300, 3900, 6100])


def track(count: int = 380) -> list[tuple[int, int, str]]:
    cues: list[tuple[int, int, str]] = []
    position = 0
    for index in range(count):
        duration = next(_STEPS)
        cues.append((position, position + duration - 100, f"line {index} spoken here now"))
        position += duration
    return cues


def good_preshift() -> AnchorPreshift:
    return AnchorPreshift(
        shift_ms=11_072,
        seed_ms=11_159,
        anchors=5,
        regions_sampled=5,
        dispersion_ms=956.0,
        pairs_before=730,
        pairs_after=915,
        median_abs_before_ms=1939.0,
        median_abs_after_ms=42.0,
    )


def evaluation(**overrides) -> SubtitleEvaluation:
    base = dict(
        median_offset_ms=40.0,
        mad_offset_ms=0.0,
        p95_offset_ms=3541.0,
        drift_ms_per_minute=0.0,
        structural_similarity=0.89,
        alass_applied=True,
        alass_successful=True,
    )
    base.update(overrides)
    ev = SubtitleEvaluation(**base)
    ev.sync_state = SyncState.UNVERIFIED
    ev.verification = VerificationAvailability.UNKNOWN
    ev.rejection_reason = RejectionReason.LOW_CONFIDENCE
    ev.cut_verdict = CutVerdict.SAME_CUT
    return ev


def valid_alass() -> AlassOutputValidation:
    return AlassOutputValidation(ok=True, cue_count=900, input_cue_count=900)


_UNSET = object()


def decide(
    preshift=_UNSET,
    ev=_UNSET,
    alass=_UNSET,
    median=42.0,
):
    """Build a decision from good defaults, overriding only what a test names.

    Explicit ``None`` overrides are meaningful (they mean "no preshift",
    "no evaluation"), so the sentinel cannot be ``None`` itself.
    """
    return decide_constant_shift_exemption(
        good_preshift() if preshift is _UNSET else preshift,
        evaluation() if ev is _UNSET else ev,
        valid_alass() if alass is _UNSET else alass,
        median,
    )


# --- the case that motivates it ----------------------------------------------


def test_a_tightly_aligned_constant_shift_is_exempted_despite_high_p95():
    decision = decide()

    assert decision.exempted is True
    assert decision.state is ConstantShiftExemptionState.EXEMPT_CONSTANT_SHIFT
    assert decision.median_residual_ms == 42.0
    assert decision.anchors == 5


def test_an_exempted_result_is_marked_verified_and_resynced():
    ev = evaluation()
    apply_exemption_to_evaluation(ev)

    assert ev.sync_state is SyncState.VERIFIED_RESYNCED
    assert ev.verification is VerificationAvailability.VERIFIED
    assert ev.rejection_reason is None
    # The substitution has to stay auditable, not silent.
    assert any("constant-shift exemption" in reason for reason in ev.reasons)
    assert any("unverified" in reason for reason in ev.reasons)


# --- criterion by criterion ---------------------------------------------------


def test_no_preshift_evidence_means_no_exemption():
    decision = decide(preshift=None)

    assert decision.exempted is False
    assert decision.reason_codes == ["no_preshift_evidence"]


def test_too_few_anchors_is_refused():
    preshift = AnchorPreshift(
        shift_ms=11_072,
        seed_ms=11_159,
        anchors=2,
        regions_sampled=5,
        dispersion_ms=100.0,
        pairs_before=700,
        pairs_after=900,
        median_abs_before_ms=1900.0,
        median_abs_after_ms=40.0,
    )
    decision = decide(preshift=preshift)

    assert decision.exempted is False
    assert decision.reason_codes == ["insufficient_anchors"]


def test_dispersed_anchors_are_refused():
    preshift = AnchorPreshift(
        shift_ms=11_072,
        seed_ms=11_159,
        anchors=5,
        regions_sampled=5,
        dispersion_ms=5000.0,
        pairs_before=700,
        pairs_after=900,
        median_abs_before_ms=1900.0,
        median_abs_after_ms=40.0,
    )
    decision = decide(preshift=preshift)

    assert decision.exempted is False
    assert decision.reason_codes == ["dispersion_too_high"]


def test_a_loose_median_residual_is_refused_even_with_perfect_anchors():
    decision = decide(median=900.0)

    assert decision.exempted is False
    assert decision.reason_codes == ["median_residual_too_high"]


def test_an_unmeasured_median_residual_is_refused():
    """The whole point is to replace p95 with a measured centre."""
    decision = decide(median=None)

    assert decision.exempted is False
    assert decision.reason_codes == ["median_residual_unmeasured"]


def test_the_median_residual_bar_is_150ms():
    assert CONSTANT_SHIFT_MAX_MEDIAN_RESIDUAL_MS == 150.0


def test_a_boundary_median_residual_is_allowed():
    decision = decide(median=CONSTANT_SHIFT_MAX_MEDIAN_RESIDUAL_MS)

    assert decision.exempted is True


@pytest.mark.parametrize("mad", [None, 801.0, 5000.0])
def test_a_movement_that_is_not_one_constant_shift_is_refused(mad):
    decision = decide(ev=evaluation(mad_offset_ms=mad))

    assert decision.exempted is False
    assert decision.reason_codes == ["movement_not_constant"]


def test_excessive_drift_is_refused():
    decision = decide(ev=evaluation(drift_ms_per_minute=400.0))

    assert decision.exempted is False
    assert decision.reason_codes == ["drift_too_high"]


@pytest.mark.parametrize("structure", [None, 0.2, 0.64])
def test_poor_structural_agreement_is_refused(structure):
    decision = decide(ev=evaluation(structural_similarity=structure))

    assert decision.exempted is False
    assert decision.reason_codes == ["structure_too_low"]


# --- every other refusal still binds ------------------------------------------


@pytest.mark.parametrize(
    "rejection",
    [
        RejectionReason.WRONG_CONTENT,
        RejectionReason.INVALID_SUBTITLE,
        RejectionReason.INVALID_TIMINGS,
        RejectionReason.CUE_LOSS,
        RejectionReason.STRUCTURE_MISMATCH,
        RejectionReason.IMPLAUSIBLE_OFFSET,
    ],
)
def test_a_non_confidence_rejection_is_never_exempted(rejection):
    """These say the subtitles are not the same content.

    No amount of good alignment evidence excuses them: a perfectly timed
    translation of the wrong film is still the wrong film.
    """
    ev = evaluation()
    ev.rejection_reason = rejection

    decision = decide(ev=ev)

    assert decision.exempted is False
    assert decision.reason_codes == ["rejection_not_confidence_only"]


def test_a_different_cut_is_never_exempted():
    ev = evaluation()
    ev.cut_verdict = CutVerdict.DIFFERENT_CUT

    decision = decide(ev=ev)

    assert decision.exempted is False
    assert decision.reason_codes == ["different_cut"]


def test_invalid_alass_output_is_refused():
    decision = decide(alass=AlassOutputValidation(ok=False))

    assert decision.exempted is False
    assert decision.reason_codes == ["alass_output_invalid"]


def test_a_missing_evaluation_is_refused():
    decision = decide(ev=None)

    assert decision.exempted is False
    assert decision.reason_codes == ["no_evaluation"]


def test_no_rejection_reason_is_accepted_as_confidence_only():
    """A clean analyzer run that happens to be unserved can still be exempted."""
    ev = evaluation()
    ev.rejection_reason = None

    assert decide(ev=ev).exempted is True


# --- the measurement helper ---------------------------------------------------


def test_median_residual_is_tightly_zero_for_aligned_tracks():
    reference = track()
    assert measure_median_abs_residual(reference, reference) == 0.0


def test_median_residual_reports_a_real_offset():
    reference = track()
    offset = [(a + 900, b + 900, t) for a, b, t in reference]

    measured = measure_median_abs_residual(offset, reference)

    assert measured == pytest.approx(900.0, abs=1.0)


def test_median_residual_is_none_without_correspondence():
    assert measure_median_abs_residual([], track()) is None


def test_median_residual_is_unmoved_by_a_few_wildly_misplaced_cues():
    """The property the exemption depends on.

    Segmentation disagreement moves a minority of cues far away, which inflates
    p95 and leaves the median alone. That asymmetry is why the median can stand
    in for p95 here, and it is worth pinning rather than assuming.
    """
    from app.services.sync.alignment import RESIDUAL_TOLERANCE_MS, pair_cues

    reference = track()
    scattered = list(reference)
    for index in range(0, len(scattered), 17):  # ~6% of cues
        start, end, text = scattered[index]
        scattered[index] = (start + 4000, end + 4000, text)

    p95_residuals = sorted(
        abs(m.delta_ms)
        for m in pair_cues(scattered, reference, tolerance_ms=RESIDUAL_TOLERANCE_MS).matches
    )
    p95 = p95_residuals[int(round((len(p95_residuals) - 1) * 0.95))]

    median = measure_median_abs_residual(scattered, reference)

    assert p95 > 2000.0
    assert median < CONSTANT_SHIFT_MAX_MEDIAN_RESIDUAL_MS


def test_the_pre_shift_and_the_exemption_measure_the_same_thing():
    """Both numbers come from one pairing implementation, on purpose.

    The shift is authorised by a median residual and the serving decision is
    authorised by a median residual; if those were computed differently the two
    gates could disagree about the same subtitle.
    """
    reference = track()
    offset = [(a + 5000, b + 5000, t) for a, b, t in reference]

    corrected = apply_anchor_preshift(offset, -5000)

    assert measure_median_abs_residual(corrected, reference) == 0.0
