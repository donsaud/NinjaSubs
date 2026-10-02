"""The Large Offset serving decision: what may be delivered, and what never may.

``decide_large_offset_serving`` is the narrow exception to the invariant that a
synchronization attempt the verifier does not trust must never replace the
original. Every half of that exception is pinned here, in both directions:

  * the case the feature exists for -- a same-episode correction the general
    verifier refuses ONLY on residual p95 -- must be deliverable;
  * every other refusal (content, structure, cue evidence, a non-constant
    movement, a missing measurement, a failed investigation, an invalid alass
    output) must still return the original, because those say something about
    the correction rather than about segmentation differences.

The exception is deliberately scoped to MAD. ``p95`` above the stable limit is
*allowed* here, and only alongside every other condition holding; a p95
exemption on its own, or an exemption for any other measurement, does not exist.

Nothing in this module may make an artifact reusable or verified: that half of
the contract is asserted just as hard as the delivery half.
"""

from __future__ import annotations

import pathlib

from app.services.sync import large_offset_investigation as loi
from app.services.sync.alignment import (
    MAX_MAD_MS_FOR_STABLE,
    MAX_P95_MS_FOR_STABLE,
    RejectionReason,
    SubtitleEvaluation,
    SyncState,
    VerificationAvailability,
    is_reusable_verified,
    may_serve_synchronized,
)
from app.services.sync.large_offset_investigation import (
    AlassOutputValidation,
    LargeOffsetInvestigation,
    LargeOffsetServingState,
    decide_large_offset_serving,
    investigate_large_offset,
    validate_alass_output,
)

FIX = pathlib.Path(__file__).parent / "fixtures" / "large_offset"


def _read(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


def _investigation() -> LargeOffsetInvestigation:
    """The Dexter pair, which establishes same-episode identity.

    Built by the real investigation rather than fabricated, so a change to what
    makes a candidate eligible shows up here instead of being hidden behind a
    hand-assembled object.
    """
    inv = investigate_large_offset(
        _read("dexter_s08e04_target.srt"),
        _read("dexter_s08e04_reference.srt"),
        identity_supported=True,
        reference_trust="high",
    )
    assert inv.eligible_for_alass, (
        "fixture pair no longer establishes eligibility: "
        f"{inv.decision.value} {inv.reason_codes}"
    )
    return inv


def _validation(ok: bool = True) -> AlassOutputValidation:
    """A structural validation result. Stage 2 is a sanity check, not a gate."""
    if ok:
        return AlassOutputValidation(
            ok=True,
            cue_count=560,
            input_cue_count=560,
            retention_ratio=1.0,
            coverage_ratio=1.0,
            residual_offset_ms=-60.0,
            monotonic=True,
            timestamps_valid=True,
            reason_codes=[],
        )
    return AlassOutputValidation(
        ok=False,
        cue_count=0,
        input_cue_count=560,
        retention_ratio=0.0,
        coverage_ratio=0.0,
        residual_offset_ms=None,
        monotonic=False,
        timestamps_valid=True,
        reason_codes=[loi.ALASS_REASON_EMPTY],
    )


def _evaluation(**overrides) -> SubtitleEvaluation:
    """The analyzer's measured verdict for a real large-offset correction.

    Defaults are the measured Dexter EVOLV numbers: MAD 0, p95 3310 (above the
    2000ms stable limit), structural 0.855, refused on confidence alone.
    """
    values: dict = {
        "mad_offset_ms": 0.0,
        "p95_offset_ms": 3310.0,
        "structural_similarity": 0.855,
        "sync_state": SyncState.UNVERIFIED,
        "verification": VerificationAvailability.UNKNOWN,
        "rejection_reason": RejectionReason.LOW_CONFIDENCE,
        "alass_applied": True,
        "alass_successful": True,
        "reasons": ["low_confidence"],
        "change_points": [],
    }
    values.update(overrides)
    return SubtitleEvaluation(**values)


def decide(inv, val, evaluation):
    return decide_large_offset_serving(inv, val, evaluation)


# --------------------------------------------------------------------------- #
# The case the feature exists for
# --------------------------------------------------------------------------- #


def test_segmentation_driven_refusal_is_deliverable():
    """The scoped exception itself: constant movement, structural agreement,
    refused by the general verifier on residual p95 alone -> corrected bytes."""
    state = decide(_investigation(), _validation(), _evaluation())
    assert state is LargeOffsetServingState.ALASS_CORRECTED_LARGE_OFFSET
    assert state.value == "alass_corrected_large_offset"


def test_the_p95_exemption_is_the_only_exemption():
    """p95 above the stable limit is tolerated *here*, and only here.

    This is the measurement the exception exists for. It is allowed while MAD
    stays inside MAX_MAD_MS_FOR_STABLE and every other condition holds; there
    is no comparable exemption for any other measurement.
    """
    assert MAX_P95_MS_FOR_STABLE < 4750.0, (
        "the documented exemption range no longer covers the real cases"
    )
    for p95 in (MAX_P95_MS_FOR_STABLE + 1.0, 4750.0):
        state = decide(
            _investigation(), _validation(), _evaluation(p95_offset_ms=p95)
        )
        assert state is LargeOffsetServingState.ALASS_CORRECTED_LARGE_OFFSET, (
            f"p95={p95} is the segmentation difference the feature exists for"
        )


def test_mad_must_stay_inside_the_stable_limit():
    """The one measurement the exception is bounded by.

    A correction that did not move every cue by the same amount is not a
    constant shift, whatever its structural similarity claims.
    """
    assert MAX_MAD_MS_FOR_STABLE == 800.0
    for mad in (MAX_MAD_MS_FOR_STABLE + 1.0, 11039.0, 4658.0):
        state = decide(
            _investigation(), _validation(), _evaluation(mad_offset_ms=mad)
        )
        assert state is LargeOffsetServingState.ORIGINAL, f"MAD={mad} must refuse"
    # Exactly at the limit is still a constant shift.
    assert (
        decide(
            _investigation(), _validation(), _evaluation(mad_offset_ms=800.0)
        )
        is LargeOffsetServingState.ALASS_CORRECTED_LARGE_OFFSET
    )


def test_an_unmeasured_movement_fails_closed():
    """``mad_offset_ms=None`` is "we could not measure it", not "it is zero"."""
    inv = _investigation()
    state = decide(inv, _validation(), _evaluation(mad_offset_ms=None))
    assert state is LargeOffsetServingState.ORIGINAL
    assert loi.SERVE_REASON_MOVEMENT_NOT_CONSTANT in inv.serving_reason_codes


# --------------------------------------------------------------------------- #
# Failures the verifier reports for reasons that have nothing to do with p95
# --------------------------------------------------------------------------- #


def test_content_and_structural_rejections_are_never_overridden():
    """A refusal ABOUT the correction binds. Only the confidence-only refusal
    is in scope."""
    for reason in (
        RejectionReason.WRONG_CONTENT,
        RejectionReason.STRUCTURE_MISMATCH,
        RejectionReason.CUE_LOSS,
        RejectionReason.IMPLAUSIBLE_OFFSET,
        RejectionReason.INVALID_TIMINGS,
        RejectionReason.ALASS_FAILED,
        RejectionReason.INSUFFICIENT_EVIDENCE,
        RejectionReason.OTHER,
    ):
        inv = _investigation()
        state = decide(inv, _validation(), _evaluation(rejection_reason=reason))
        assert state is LargeOffsetServingState.ORIGINAL, reason
        assert loi.SERVE_REASON_STRUCTURAL_REJECTED in inv.serving_reason_codes


def test_low_confidence_is_the_in_scope_refusal():
    inv = _investigation()
    state = decide(
        inv, _validation(), _evaluation(rejection_reason=RejectionReason.LOW_CONFIDENCE)
    )
    assert state is LargeOffsetServingState.ALASS_CORRECTED_LARGE_OFFSET
    assert loi.SERVE_REASON_CORRECTED in inv.serving_reason_codes


def test_a_structural_rejection_reason_is_never_overridden():
    """Even a low-confidence refusal that ALSO carries a structural reason
    refuses: the structural reason is what the analyzer found."""
    inv = _investigation()
    state = decide(
        inv,
        _validation(),
        _evaluation(
            rejection_reason=RejectionReason.LOW_CONFIDENCE,
            structural_similarity=0.5,
        ),
    )
    assert state is LargeOffsetServingState.ORIGINAL
    assert loi.SERVE_REASON_STRUCTURE_TOO_LOW in inv.serving_reason_codes


def test_structural_similarity_below_the_floor_refuses():
    inv = _investigation()
    state = decide(
        inv, _validation(), _evaluation(structural_similarity=0.64)
    )
    assert state is LargeOffsetServingState.ORIGINAL
    assert loi.SERVE_REASON_STRUCTURE_TOO_LOW in inv.serving_reason_codes
    # The floor is a documented number, not an accident.
    assert loi.LARGE_OFFSET_MIN_STRUCTURAL_SIMILARITY == 0.65
    state = decide(
        inv, _validation(), _evaluation(structural_similarity=0.65)
    )
    assert state is LargeOffsetServingState.ALASS_CORRECTED_LARGE_OFFSET


def test_an_unmeasured_structural_similarity_fails_closed():
    state = decide(
        _investigation(), _validation(), _evaluation(structural_similarity=None)
    )
    assert state is LargeOffsetServingState.ORIGINAL


# --------------------------------------------------------------------------- #
# Preconditions: identity, a valid alass output, an analyzer verdict
# --------------------------------------------------------------------------- #


def test_identity_not_established_refuses():
    """Without reference-first identity there is no authorization at all --
    this is the precondition the whole design rests on."""
    inv = _investigation()
    inv.eligible_for_alass = False
    inv.decision = loi.LargeOffsetDecision.INSUFFICIENT_EVIDENCE
    state = decide(inv, _validation(), _evaluation())
    assert state is LargeOffsetServingState.ORIGINAL
    assert loi.SERVE_REASON_NOT_ELIGIBLE in inv.serving_reason_codes


def test_invalid_alass_output_refuses():
    inv = _investigation()
    state = decide(inv, _validation(ok=False), _evaluation())
    assert state is LargeOffsetServingState.ORIGINAL
    assert loi.SERVE_REASON_ALASS_INVALID in inv.serving_reason_codes


def test_missing_analyzer_verdict_fails_closed():
    """Without the analyzer's own measurement there is no way to tell a
    segmentation-driven refusal from a structural one."""
    inv = _investigation()
    state = decide(inv, _validation(), None)
    assert state is LargeOffsetServingState.ORIGINAL
    assert loi.SERVE_REASON_NO_EVALUATION in inv.serving_reason_codes


def test_the_decision_is_recorded_on_the_investigation():
    inv = _investigation()
    assert inv.serving_state is LargeOffsetServingState.ORIGINAL, (
        "a fresh investigation must default to serving the original"
    )
    decide(inv, _validation(), _evaluation())
    assert inv.serving_state is LargeOffsetServingState.ALASS_CORRECTED_LARGE_OFFSET
    assert inv.serving_reason_codes == [loi.SERVE_REASON_CORRECTED]

    inv2 = _investigation()
    decide(inv2, _validation(), _evaluation(mad_offset_ms=9999.0))
    assert inv2.serving_state is LargeOffsetServingState.ORIGINAL
    assert loi.SERVE_REASON_MOVEMENT_NOT_CONSTANT in inv2.serving_reason_codes


# --------------------------------------------------------------------------- #
# The orchestrator wiring: reaching the path is not authorization
# --------------------------------------------------------------------------- #


def test_orchestrator_override_requires_the_full_decision(caplog):
    """The dangerous wiring mistake: overriding because the large-offset path
    ran at all, instead of because the decision passed.

    This case enters the large-offset path and establishes same-episode identity
    -- so ``large_offset_inv`` exists and the override branch is genuinely
    reached -- but the correction is not a constant shift, so the investigation
    refuses it, ``serving_state`` stays ORIGINAL, and the provider bytes must go
    out. It runs a stub sync service rather than the real alass binary, so the
    guard exists on every platform the suite runs on.
    """
    import asyncio

    from app.services.subtitle_matcher import parse_srt_cues as _parse
    from app.services.sync.orchestrator import SyncOrchestrator
    from app.services.sync.query import ResolvedReference

    reference_text = _read("dexter_s08e04_reference.srt")
    reference_cues = _parse(reference_text)

    # Uniform +96.5s: squarely a large-offset candidate.
    shift = 96_500
    parts = []
    for i, (s, e, t) in enumerate(reference_cues):
        parts.append(f"{i + 1}\n{_ts(s + shift)} --> {_ts(e + shift)}\n{t}")
    target_text = "\n\n".join(parts) + "\n"
    target = target_text.encode("utf-8")

    # The "correction" alass produced: not one constant shift. The analyzer
    # measures it as non-constant and refuses it, and so does the investigation,
    # so delivery is ORIGINAL on both counts.
    drift_total = 6_000
    n = len(reference_cues)
    corrected_parts = []
    for i, (s, e, t) in enumerate(reference_cues):
        extra = int(drift_total * (i / max(1, n - 1)))
        corrected_parts.append(
            f"{i + 1}\n{_ts(s + extra)} --> {_ts(e + extra)}\n{t}"
        )
    corrected_text = "\n\n".join(corrected_parts) + "\n"

    class _Reference:
        def __init__(self) -> None:
            self.calls = 0

        async def resolve_with_provenance(self, query, *args, **kwargs):  # noqa: ANN001
            self.calls += 1
            return ResolvedReference(
                reference_text,
                kind="hash",
                bluray_match=True,
                candidate="x.srt",
                reference_trust="high",
                reference_consensus=1.0,
                reference_independent_sources=1,
            )

    class _StubSyncService:
        async def sync_async(self, target_text_arg, reference, **kwargs):  # noqa: ANN001
            return corrected_text

    meta = {
        "provider": "subdl",
        "release_name": "Dexter.S08E04.Scar.Tissue.1080p.BluRay.x264.srt",
        "language": "ar",
        "lang": "ar",
        "imdb_id": "tt0773262",
        "season": 8,
        "episode": 4,
        "video_fingerprint": "a1b2c3d4" * 8,
        "target_filename": "Dexter.s8e04.Scar.tissue.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv",
        "video_size": 5_414_925_805,
    }

    strategy = _Reference()
    orchestrator = SyncOrchestrator(
        external_strategy=strategy, sync_service=_StubSyncService()
    )
    import logging as _logging

    with caplog.at_level(_logging.INFO, logger="app.services.sync.orchestrator"):
        served = asyncio.run(
            orchestrator.evaluate_and_sync(
                target, meta, "tt0773262:8:4", auto_sync=True
            )
        )

    assert strategy.calls == 1, "reference was never resolved; path not exercised"
    evaluation = orchestrator._last_evaluation
    assert evaluation is not None, "no analyzer verdict was produced"

    # The investigation must actually have run, so the override branch is
    # genuinely reached rather than skipped past.
    decision_lines = [r.getMessage() for r in caplog.records if "large_offset.decision" in r.getMessage()]
    assert decision_lines, (
        "the large-offset investigation never ran; this test would not exercise "
        "the override branch at all"
    )
    assert "serving_state=original" in decision_lines[-1], decision_lines[-1]

    # The normal serving gate must also refuse -- otherwise this test would be
    # asserting nothing about the override at all.
    assert not may_serve_synchronized(
        evaluation.sync_state.value, evaluation.verification.value
    ), (
        "the analyzer unexpectedly accepted this; the test would not distinguish "
        "the override from ordinary delivery"
    )
    assert served == target, (
        "the orchestrator served a correction the investigation refused; the "
        "override must be conditional on the full decision, not on having "
        "entered the large-offset path"
    )


def _ts(ms: int) -> str:
    ms = max(0, int(ms))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# --------------------------------------------------------------------------- #
# The exception must not leak into verification or reuse
# --------------------------------------------------------------------------- #


def test_corrected_state_is_not_a_sync_state_and_not_a_verification():
    """``ALASS_CORRECTED_LARGE_OFFSET`` is a delivery decision only. It must not
    exist anywhere in the analyzer's vocabulary."""
    corrected = LargeOffsetServingState.ALASS_CORRECTED_LARGE_OFFSET
    assert corrected.value not in {s.value for s in SyncState}
    assert corrected.value not in {v.value for v in VerificationAvailability}
    # And it is never what the analyzer is asked to decide with.
    assert corrected is not SyncState.VERIFIED_SYNCED
    assert corrected is not SyncState.VERIFIED_RESYNCED


def test_delivering_a_correction_does_not_verify_or_reuse_it():
    """The whole point: these artifacts are served once and stay non-reusable.

    The delivery decision runs on the analyzer's verdict; it must leave that
    verdict exactly as it found it.
    """
    evaluation = _evaluation()
    before_state = evaluation.sync_state
    before_verification = evaluation.verification

    assert not may_serve_synchronized(
        evaluation.sync_state.value, evaluation.verification.value
    ), "the analyzer must still refuse this on its own terms"
    assert not is_reusable_verified(
        evaluation.sync_state.value, evaluation.verification.value
    )

    inv = _investigation()
    state = decide(inv, _validation(), evaluation)
    assert state is LargeOffsetServingState.ALASS_CORRECTED_LARGE_OFFSET

    # Untouched.
    assert evaluation.sync_state is before_state
    assert evaluation.verification is before_verification
    assert evaluation.sync_state is SyncState.UNVERIFIED
    assert evaluation.verification is VerificationAvailability.UNKNOWN
    # ...so it is still not servable and still not reusable by the normal path.
    assert not may_serve_synchronized(
        evaluation.sync_state.value, evaluation.verification.value
    )
    assert not is_reusable_verified(
        evaluation.sync_state.value, evaluation.verification.value
    )


def test_no_sync_state_becomes_reusable_through_this_path():
    """Whatever the investigation decides, the analyzer's own reuse gate is
    unchanged for every state it can produce."""
    for state in SyncState:
        for verification in VerificationAvailability:
            reusable = is_reusable_verified(state.value, verification.value)
            if reusable:
                assert state in (SyncState.VERIFIED_SYNCED, SyncState.VERIFIED_RESYNCED)
                assert verification in (
                    VerificationAvailability.VERIFIED,
                    VerificationAvailability.CACHED,
                )


# --------------------------------------------------------------------------- #
# Against the real alass outputs: the whole decision, end to end
# --------------------------------------------------------------------------- #


def test_real_alass_outputs_are_routed_as_measured():
    """The measured matrix, asserted as behaviour rather than printed.

    measured on the Dexter target against each candidate reference, exactly as
    ``alass_out/run.sh`` produced them:

      * ``wrong_episode`` and ``drifting`` move inconsistently -> MAD refuses;
      * ``valid_constant`` and the real Dexter reference are constant shifts and
        structural agreements -> corrected bytes are served;
      * ``different_cut`` is the one KNOWN FALSE ACCEPT: MAD 0.0 and structural
        0.762, indistinguishable from the required-positive case on every signal
        available here (structural 0.762 vs 0.729 for the real case). Accepted
        deliberately rather than guessed at -- see the module note in
        ``large_offset_investigation.py``.
    """
    from app.services.subtitle_matcher import parse_srt_cues
    from app.services.sync.alignment import AlignmentAnalyzer

    target = _read("dexter_s08e04_target.srt")

    # reference -> (expected serving state, refusal reason expected)
    cases = {
        "dexter_s08e04_reference": (
            LargeOffsetServingState.ALASS_CORRECTED_LARGE_OFFSET,
            None,
        ),
        "valid_constant_offset": (
            LargeOffsetServingState.ALASS_CORRECTED_LARGE_OFFSET,
            None,
        ),
        "negative_different_cut": (
            LargeOffsetServingState.ALASS_CORRECTED_LARGE_OFFSET,
            None,
        ),
        "negative_wrong_episode": (
            LargeOffsetServingState.ORIGINAL,
            loi.SERVE_REASON_MOVEMENT_NOT_CONSTANT,
        ),
        "negative_drifting": (
            LargeOffsetServingState.ORIGINAL,
            loi.SERVE_REASON_MOVEMENT_NOT_CONSTANT,
        ),
    }

    for ref_name, (expected, expected_reason) in cases.items():
        ref = _read(f"{ref_name}.srt")
        out_path = FIX / "alass_out" / f"out_{ref_name}.srt"
        assert out_path.is_file(), f"missing alass output fixture: {out_path}"
        output = out_path.read_text(encoding="utf-8")

        inv = investigate_large_offset(
            target, ref, identity_supported=True, reference_trust="high"
        )
        val = validate_alass_output(target, output, ref)
        evaluation = AlignmentAnalyzer().analyze(
            parse_srt_cues(target),
            parse_srt_cues(output),
            parse_srt_cues(ref),
            alass_applied=True,
            alass_successful=True,
            max_plausible_offset_ms=loi.LARGE_OFFSET_MAX_MS,
        )
        state = decide(inv, val, evaluation)

        assert state is expected, (
            f"{ref_name}: got {state.value}, expected {expected.value} "
            f"(reasons={inv.serving_reason_codes} mad={evaluation.mad_offset_ms} "
            f"struct={evaluation.structural_similarity} "
            f"rej={evaluation.rejection_reason})"
        )
        if expected_reason is not None:
            assert expected_reason in inv.serving_reason_codes, (
                f"{ref_name}: refused for the wrong reason "
                f"{inv.serving_reason_codes}"
            )

        # Whatever was decided, the analyzer's verdict is untouched and the
        # artifact stays out of the reusable cache.
        assert not is_reusable_verified(
            evaluation.sync_state.value, evaluation.verification.value
        ), ref_name


def test_identity_signals_show_margin_but_are_not_what_refuses_the_negatives():
    """Recorded honestly rather than overstated.

    The reference-first identity signals do NOT separate the wrong-episode
    reference: every fixture is reported ``same_episode`` because the floors are
    set for acceptance (gap 0.60, landmarks 0.50, coverage 0.55) rather than for
    discrimination. The measured margin is large though -- the wrong-episode
    reference scores gap similarity 0.649 against 0.998 for every positive --
    so tightening is a real, evidence-backed follow-up, not a guess.

    What actually refuses ``wrong_episode`` today is movement MAD, which is the
    binding condition of the serving decision. That is asserted here so the
    reliance is explicit instead of assumed away.
    """
    target = _read("dexter_s08e04_target.srt")
    positive = investigate_large_offset(
        target,
        _read("dexter_s08e04_reference.srt"),
        identity_supported=True,
        reference_trust="high",
    )
    negative = investigate_large_offset(
        target,
        _read("negative_wrong_episode.srt"),
        identity_supported=True,
        reference_trust="high",
    )
    # Both are accepted as same-episode today...
    assert positive.same_episode.value == "same_episode"
    assert negative.same_episode.value == "same_episode"
    # ...and the gap signal is where the discrimination would come from.
    pos_gap = positive.evidence.gap_distribution_similarity
    neg_gap = negative.evidence.gap_distribution_similarity
    assert neg_gap is not None and pos_gap is not None
    assert neg_gap < pos_gap, "expected the wrong-episode reference to score lower"
    assert neg_gap < pos_gap * 0.75, (
        "the measured margin has disappeared; re-evaluate whether the gap floor "
        f"can now be tightened (pos={pos_gap} neg={neg_gap})"
    )
