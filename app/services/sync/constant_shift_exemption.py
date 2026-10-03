"""Constant-shift exemption: when residual p95 is the *wrong* measurement.

The problem
-----------
Two releases of one film rarely share cue segmentation. An Arabic track with 958
cues and an English reference with 922 cannot pair index-for-index: every place
the translator merged or split a line shifts all subsequent pairs. The result is
a residual distribution with a tight centre and a long tail -- on real Whiplash
data, 86% of cues inside a 1500-2000ms band while the median sat at 88ms.

``MAX_P95_MS_FOR_STABLE`` reads that tail as desynchronisation and refuses a
subtitle that is in fact well aligned. Worse, it is not merely noisy here, it is
*anti-correlated with truth*: on the same pair a +9500ms shift measures p95
1701ms (a pass) at 1573ms median error, while the correct +11072ms shift measures
p95 3541ms at 42ms median error. Tuning p95 would prefer the worse alignment.

So p95 is exempted -- and only p95 -- for a correction that is positively
evidenced to be a single constant shift, measured four independent ways:

1. **Corroborated multi-region anchors.** :mod:`app.services.sync.anchor_preshift`
   searched five regions of the timeline, agreed on an offset, and only proposed
   it after showing the shift measurably improved correspondence. Dispersion and
   anchor count are re-checked here rather than trusted.
2. **A tight median residual.** The alignment is confirmed positively, at the
   centre of the distribution, not inferred from the tail. This is the criterion
   that replaces p95: it is unmoved by segmentation drift, where p95 is
   dominated by it.
3. **A constant movement.** Median absolute movement within
   ``MAX_MAD_MS_FOR_STABLE`` -- alass moved the film by one offset, not a curve.
4. **Structural agreement.** The corrected track still looks like the reference.

And the exemptions are narrow by construction:

* The analyzer's refusal must be a *confidence* objection. ``WRONG_CONTENT``,
  ``INVALID_TIMINGS``, ``CUE_LOSS``, ``STRUCTURE_MISMATCH`` and a
  ``DIFFERENT_CUT`` verdict all still return the original, because those say the
  subtitles are not the same film rather than that the measurement is unfair.
* alass output must independently validate.
* Every refusal reason is recorded, and a denial says which check failed.

Unlike the Large Offset path, which deliberately leaves the analyzer's verdict
alone, an exemption here does mark the result ``VERIFIED_RESYNCED``/``verified``
so the artifact is cacheable. That is a real loosening and is the reason the
median-residual bar is 150ms rather than something looser: it is doing the work
p95 no longer does.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from .alignment import (
    MAX_DRIFT_MS_PER_MINUTE,
    MAX_MAD_MS_FOR_STABLE,
    CutVerdict,
    RejectionReason,
    SubtitleEvaluation,
    SyncState,
    VerificationAvailability,
)
from .anchor_preshift import AnchorPreshift
from .large_offset import (
    LARGE_OFFSET_MAX_DISPERSION_MS,
    LARGE_OFFSET_MIN_ANCHORS,
)
from .large_offset_investigation import AlassOutputValidation

logger = logging.getLogger(__name__)

#: The median residual an exempted result must still achieve, in milliseconds.
#:
#: This is the load-bearing number in this module. p95 is being set aside, so
#: something has to stand in its place, and it has to be a statistic that
#: segmentation disagreement cannot inflate. The median is that statistic: on the
#: real Whiplash pair it read 42ms while p95 read 3541ms.
#:
#: 150ms is roughly one and a half frames at 24fps. Subtitles are not frame-
#: locked against dialogue in practice, so this is "indistinguishable from
#: aligned" rather than a claim of frame accuracy.
CONSTANT_SHIFT_MAX_MEDIAN_RESIDUAL_MS = 150.0

#: Floor on structural agreement between the corrected track and the reference.
#: Matches the Large Offset path so the two exemptions cannot drift apart.
CONSTANT_SHIFT_MIN_STRUCTURAL_SIMILARITY = 0.65


class ConstantShiftExemptionState(str, Enum):
    """Whether the constant-shift exemption applies to this candidate."""

    ORIGINAL = "original"
    EXEMPT_CONSTANT_SHIFT = "exempt_constant_shift"


# Reason codes, in the order they are checked. The first failure is the one
# reported, so the log names the actual obstacle rather than the last check.
REASON_NO_PRESHIFT = "no_preshift_evidence"
REASON_INSUFFICIENT_ANCHORS = "insufficient_anchors"
REASON_DISPERSION_TOO_HIGH = "dispersion_too_high"
REASON_ALASS_INVALID = "alass_output_invalid"
REASON_NO_EVALUATION = "no_evaluation"
REASON_WRONG_REJECTION = "rejection_not_confidence_only"
REASON_DIFFERENT_CUT = "different_cut"
REASON_MOVEMENT_NOT_CONSTANT = "movement_not_constant"
REASON_DRIFT_TOO_HIGH = "drift_too_high"
REASON_STRUCTURE_TOO_LOW = "structure_too_low"
REASON_MEDIAN_RESIDUAL_UNMEASURED = "median_residual_unmeasured"
REASON_MEDIAN_RESIDUAL_TOO_HIGH = "median_residual_too_high"
REASON_EXEMPTED = "constant_shift_corroborated"


@dataclass
class ConstantShiftExemptionDecision:
    """The exemption verdict, with the evidence that produced it."""

    state: ConstantShiftExemptionState
    reason_codes: list[str] = field(default_factory=list)
    median_residual_ms: float | None = None
    movement_mad_ms: float | None = None
    structural_similarity: float | None = None
    anchors: int | None = None
    dispersion_ms: float | None = None

    @property
    def exempted(self) -> bool:
        return self.state is ConstantShiftExemptionState.EXEMPT_CONSTANT_SHIFT

    def summary(self) -> str:
        if not self.exempted:
            return f"not exempted ({','.join(self.reason_codes) or 'no reason'})"
        return (
            f"exempted ({self.reason_codes[0]}) "
            f"median_residual={self.median_residual_ms:.0f}ms "
            f"movement_mad={self.movement_mad_ms:.0f}ms "
            f"structural={self.structural_similarity:.3f} "
            f"anchors={self.anchors} dispersion={self.dispersion_ms}ms"
        )


def _deny(
    decision: ConstantShiftExemptionDecision, reason: str
) -> ConstantShiftExemptionDecision:
    decision.state = ConstantShiftExemptionState.ORIGINAL
    decision.reason_codes.append(reason)
    logger.info(
        "constant_shift.serving_denied %s %s", reason, decision.summary()
    )
    return decision


def decide_constant_shift_exemption(
    preshift: AnchorPreshift | None,
    evaluation: SubtitleEvaluation | None,
    validation: AlassOutputValidation,
    median_residual_ms: float | None,
) -> ConstantShiftExemptionDecision:
    """May a p95-refused but tightly-aligned result be served after all?

    ``preshift`` is the accepted :class:`AnchorPreshift` from the pre-alass
    stage, or ``None`` when no shift was applied -- in which case there is no
    constant-shift evidence and the answer is always no.

    ``median_residual_ms`` is the median absolute residual of the *final* alass
    output against the reference, measured by the caller. Passing ``None`` denies:
    the whole point is to replace p95 with a measured centre, and an unmeasured
    centre is not a substitute.

    Fails closed at every step.
    """
    decision = ConstantShiftExemptionDecision(
        state=ConstantShiftExemptionState.ORIGINAL
    )

    # 1. Corroborated multi-region anchors.
    if preshift is None:
        return _deny(decision, REASON_NO_PRESHIFT)
    decision.anchors = preshift.anchors
    decision.dispersion_ms = preshift.dispersion_ms
    if preshift.anchors < LARGE_OFFSET_MIN_ANCHORS:
        return _deny(decision, REASON_INSUFFICIENT_ANCHORS)
    if (
        preshift.dispersion_ms is not None
        and preshift.dispersion_ms > LARGE_OFFSET_MAX_DISPERSION_MS
    ):
        return _deny(decision, REASON_DISPERSION_TOO_HIGH)

    if not validation.ok:
        return _deny(decision, REASON_ALASS_INVALID)

    if evaluation is None:
        return _deny(decision, REASON_NO_EVALUATION)

    decision.movement_mad_ms = evaluation.mad_offset_ms
    decision.structural_similarity = evaluation.structural_similarity
    decision.median_residual_ms = median_residual_ms

    # The refusal must be about confidence. Anything else says the subtitles are
    # not the same content, which no amount of good alignment evidence excuses.
    rejection = evaluation.rejection_reason
    if rejection is not None and rejection is not RejectionReason.LOW_CONFIDENCE:
        return _deny(decision, REASON_WRONG_REJECTION)
    # Compared by value, not identity. ``cut_verdict`` is a plain ``str`` field on
    # a model with ``validate_assignment``, so a value assigned anywhere else
    # arrives as ``'different_cut'`` rather than as the enum member -- an
    # identity check silently lets a different-cut subtitle through.
    if evaluation.cut_verdict == CutVerdict.DIFFERENT_CUT.value:
        return _deny(decision, REASON_DIFFERENT_CUT)

    # 3. Constant movement.
    if (
        evaluation.mad_offset_ms is None
        or evaluation.mad_offset_ms > MAX_MAD_MS_FOR_STABLE
    ):
        return _deny(decision, REASON_MOVEMENT_NOT_CONSTANT)

    drift = evaluation.drift_ms_per_minute
    if drift is not None and abs(drift) > MAX_DRIFT_MS_PER_MINUTE:
        return _deny(decision, REASON_DRIFT_TOO_HIGH)

    # 4. Structural agreement.
    if (
        evaluation.structural_similarity is None
        or evaluation.structural_similarity < CONSTANT_SHIFT_MIN_STRUCTURAL_SIMILARITY
    ):
        return _deny(decision, REASON_STRUCTURE_TOO_LOW)

    # 2. Tight median residual -- the criterion standing in for p95.
    if median_residual_ms is None:
        return _deny(decision, REASON_MEDIAN_RESIDUAL_UNMEASURED)
    if median_residual_ms > CONSTANT_SHIFT_MAX_MEDIAN_RESIDUAL_MS:
        return _deny(decision, REASON_MEDIAN_RESIDUAL_TOO_HIGH)

    decision.state = ConstantShiftExemptionState.EXEMPT_CONSTANT_SHIFT
    decision.reason_codes.append(REASON_EXEMPTED)
    logger.info(
        "constant_shift.serving_allowed p95_ms=%s median_residual_ms=%.0f "
        "movement_mad_ms=%.0f structural_similarity=%.3f anchors=%d/%d "
        "dispersion_ms=%.0f",
        evaluation.p95_offset_ms,
        median_residual_ms,
        evaluation.mad_offset_ms or 0.0,
        evaluation.structural_similarity or 0.0,
        preshift.anchors,
        preshift.regions_sampled,
        preshift.dispersion_ms or 0.0,
    )
    return decision


def apply_exemption_to_evaluation(evaluation: SubtitleEvaluation) -> None:
    """Mark an exempted result as a measured, verified re-sync.

    Called only after :func:`decide_constant_shift_exemption` has returned
    ``EXEMPT_CONSTANT_SHIFT``. Unlike the Large Offset path this *does* rewrite
    the verdict, so the artifact becomes cacheable as a finished sync
    (``may_serve_synchronized`` and ``is_reusable_verified`` both follow). The
    original analyzer verdict is preserved in ``reasons`` so the substitution
    stays auditable.
    """
    original_state = evaluation.sync_state
    original_verification = evaluation.verification
    evaluation.reasons.append(
        f"constant-shift exemption: p95 {evaluation.p95_offset_ms:.0f}ms set aside "
        f"as segmentation drift; analyzer recorded "
        f"{original_state.value}/{original_verification.value}"
    )
    # Through ``set_verdict``, not attribute assignment. ``SubtitleEvaluation``
    # enforces ``validate_assignment`` and its structural guard reads
    # ``verification`` from the validation context, which is empty on a plain
    # attribute write -- so ``evaluation.sync_state = VERIFIED_RESYNCED`` is
    # silently downgraded back to UNVERIFIED. ``set_verdict`` sets verification
    # first, which is what makes a verified claim possible at all.
    evaluation.rejection_reason = None
    evaluation.set_verdict(SyncState.VERIFIED_RESYNCED, VerificationAvailability.VERIFIED)


def measure_median_abs_residual(
    target: str | Sequence[tuple[int, int, str]],
    reference: str | Sequence[tuple[int, int, str]],
) -> float | None:
    """Median absolute residual of ``target`` against ``reference``, in ms.

    Delegates to the same pairing the pre-shift gate uses, so the number that
    authorises serving and the number that authorised shifting are produced by
    one implementation.
    """
    from ..subtitle_matcher import parse_srt_cues
    from .alignment import RESIDUAL_TOLERANCE_MS, pair_cues

    target_cues = parse_srt_cues(target) if isinstance(target, str) else list(target)
    reference_cues = (
        parse_srt_cues(reference) if isinstance(reference, str) else list(reference)
    )
    pairing = pair_cues(target_cues, reference_cues, tolerance_ms=RESIDUAL_TOLERANCE_MS)
    residuals = sorted(abs(match.delta_ms) for match in pairing.matches)
    if not residuals:
        return None
    middle = len(residuals) // 2
    if len(residuals) % 2:
        return float(residuals[middle])
    return (residuals[middle - 1] + residuals[middle]) / 2.0
