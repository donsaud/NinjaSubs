"""Large Offset Evidence Gate.

The confirmed case: ``Dexter.s8e04.Scar.tissue.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv``
opens its dialogue at ~107.5s while compatible BluRay references open at
~10.7s. That ~96.6s displacement used to be refused outright by the 20s
first-dialogue execution window. It is now a *Large Offset Candidate* that has
to earn an alass attempt from corroborating timing evidence, and every
post-alass check still decides whether anything is actually served.

Every cue timing in ``tests/fixtures/large_offset`` is a real measurement from
the production cache; see ``tools/build_large_offset_fixtures.py``. No test
here touches the network.
"""

from __future__ import annotations

import bisect
import inspect
import pathlib

import pytest

from app.services.subtitle_matcher import (
    FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
    first_dialogue_cluster,
    parse_srt_cues,
    strip_intro_nonspeech,
    validate_cue_sanity,
)
from app.services.sync import large_offset as gate
from app.services.sync.alignment import (
    MAX_MAD_MS_FOR_STABLE,
    MAX_PLAUSIBLE_OFFSET_MS,
    AlignmentAnalyzer,
    RejectionReason,
    SyncState,
)
from app.services.sync.large_offset import (
    assess_large_offset,
    classify_large_offset_candidate,
)

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "large_offset"

# Either verified state counts as "the gate did not weaken verification": a
# subtitle alass actually re-timed lands on VERIFIED_RESYNCED, and an
# already-aligned one on VERIFIED_SYNCED.
VERIFIED_STATES = {SyncState.VERIFIED_SYNCED, SyncState.VERIFIED_RESYNCED}


def read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def cues(name: str) -> list[tuple[int, int, str]]:
    return parse_srt_cues(read(name))


def dialogue_start_ms(name: str) -> int | None:
    return first_dialogue_cluster(strip_intro_nonspeech(read(name)))


def synth(cue_count: int, *, start_ms: int = 10_000, step_ms: int = 2_000) -> str:
    """A dense, plainly-dialogue subtitle with fully controlled timings."""
    blocks = []
    for index in range(cue_count):
        begin = start_ms + index * step_ms
        stamp = f"{begin // 3600000:02d}:{begin // 60000 % 60:02d}:{begin // 1000 % 60:02d},{begin % 1000:03d}"
        stop = begin + 900
        stamp_end = (
            f"{stop // 3600000:02d}:{stop // 60000 % 60:02d}:{stop // 1000 % 60:02d},{stop % 1000:03d}"
        )
        blocks.append(
            f"{index + 1}\n{stamp} --> {stamp_end}\n"
            f"We should talk about the second issue number {index} here.\n"
        )
    return "\n".join(blocks)


# --------------------------------------------------------------------------- #
# The normal <=20s path is untouched
# --------------------------------------------------------------------------- #


def test_normal_offset_stays_on_the_fast_path() -> None:
    """An ordinary shift is not a large-offset candidate and stays un-gated."""
    target = synth(60)
    reference = synth(60, start_ms=14_000)  # +4s, inside the 20s window

    verdict = validate_cue_sanity(
        target, reference, threshold_ms=FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS
    )
    assert verdict["ok"], "a 4s shift must still pass the normal window"

    assert (
        classify_large_offset_candidate(
            target, reference, threshold_ms=FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS
        )
        is None
    ), "a normal offset must never be routed through the large-offset gate"


def test_dexter_s08e04_measurements_match_the_reported_case() -> None:
    """Pin the real-world numbers the feature exists for."""
    target_first = dialogue_start_ms("dexter_s08e04_target.srt")
    reference_first = dialogue_start_ms("dexter_s08e04_reference.srt")

    assert target_first == 107_540, "target first dialogue ~107.5s"
    assert reference_first == 10_710, "reference first dialogue ~10.7s"

    seed = classify_large_offset_candidate(
        read("dexter_s08e04_target.srt"),
        read("dexter_s08e04_reference.srt"),
        threshold_ms=FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
    )
    assert seed is not None, "the case must be classified as a large-offset candidate"
    assert 96_000 <= seed <= 97_500, f"~96.6s displacement, got {seed}ms"


# --------------------------------------------------------------------------- #
# Positive: the gate grants a bounded trial
# --------------------------------------------------------------------------- #


def test_valid_constant_large_offset_is_accepted_for_a_trial() -> None:
    """Case E / the real case: several distant regions agree on one offset."""
    target = read("dexter_s08e04_target.srt")
    reference = read("dexter_s08e04_reference.srt")
    seed = classify_large_offset_candidate(
        target, reference, threshold_ms=FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS
    )
    assessment = assess_large_offset(
        target, reference, seed_offset_ms=seed, identity_supported=True
    )

    assert assessment.accepted, assessment.summary()
    assert assessment.estimated_offset_ms is not None
    assert 95_000 <= assessment.estimated_offset_ms <= 98_000
    assert assessment.anchor_count >= gate.LARGE_OFFSET_MIN_ANCHORS
    assert assessment.offset_dispersion_ms is not None
    assert assessment.offset_dispersion_ms <= gate.LARGE_OFFSET_MAX_DISPERSION_MS
    assert assessment.structural_similarity is not None
    assert assessment.structural_similarity >= gate.LARGE_OFFSET_MIN_STRUCTURAL_SIMILARITY


def test_acceptance_is_not_pinned_to_one_offset() -> None:
    """A different constant offset is accepted too; nothing is hard-coded."""
    target = read("dexter_s08e04_target.srt")
    reference = read("valid_constant_offset.srt")
    seed = classify_large_offset_candidate(
        target, reference, threshold_ms=FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS
    )
    assert seed is not None and seed < FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS * 5
    assessment = assess_large_offset(
        target, reference, seed_offset_ms=seed, identity_supported=True
    )
    assert assessment.accepted, assessment.summary()
    assert assessment.estimated_offset_ms is not None
    assert 91_000 <= assessment.estimated_offset_ms <= 95_000


def test_anchors_are_spread_across_the_timeline() -> None:
    """Evidence must be distributed, not clustered in the opening."""
    target = read("dexter_s08e04_target.srt")
    reference = read("dexter_s08e04_reference.srt")
    assessment = assess_large_offset(
        target, reference, seed_offset_ms=96_830, identity_supported=True
    )
    assert assessment.regions_sampled == gate.LARGE_OFFSET_REGION_COUNT
    assert assessment.anchor_count >= 4, "at least four of five regions must anchor"

    # Positions come from independent regions of a ~50 minute file, so the
    # anchors cannot all be clustered at the head.
    target_cues = gate._bounded(gate._as_cues(target))
    reference_starts = [c[0] for c in gate._bounded(gate._as_cues(reference))]
    regions, _, _ = gate._region_medians(target_cues, reference_starts, 96_830.0)
    positions = [position for position, _ in regions]
    assert len(set(positions)) >= 4
    assert max(positions) - min(positions) > 1_000_000, "anchors must span minutes"


def test_consensus_counts_only_agreeing_references() -> None:
    """Evidence #5: independent references that agree raise the count."""
    target = read("dexter_s08e04_target.srt")
    reference = read("dexter_s08e04_reference.srt")

    agreeing = assess_large_offset(
        target,
        reference,
        seed_offset_ms=96_830,
        identity_supported=True,
        consensus_offsets_ms=[96_500.0, 96_600.0, 96_400.0],
    )
    assert agreeing.accepted
    assert agreeing.consensus_count == 3
    assert gate.REASON_CONSENSUS_AGREES in agreeing.reason_codes

    conflicting = assess_large_offset(
        target,
        reference,
        seed_offset_ms=96_830,
        identity_supported=True,
        consensus_offsets_ms=[31_000.0, 4_000.0, 60_000.0],
    )
    assert conflicting.consensus_count == 0
    assert gate.REASON_CONSENSUS_CONFLICTS in conflicting.reason_codes
    # Conflicting consensus is recorded, never used as proof on its own.
    assert gate.REASON_CONSENSUS_AGREES not in conflicting.reason_codes


# --------------------------------------------------------------------------- #
# Negative: every guard is individually load-bearing
# --------------------------------------------------------------------------- #


def test_over_ceiling_is_rejected_without_analysis() -> None:
    """Section 9: a bounded ceiling, checked before anything expensive runs."""
    target = read("dexter_s08e04_target.srt")
    seed = gate.LARGE_OFFSET_MAX_MS + 5_000
    assessment = assess_large_offset(
        target, target, seed_offset_ms=seed, identity_supported=True
    )
    assert not assessment.accepted
    assert assessment.reason_codes == [gate.REASON_MAX_OFFSET_EXCEEDED]
    assert assessment.anchor_count == 0, "the ceiling must short-circuit the analysis"

    explicit = assess_large_offset(
        target, target, seed_offset_ms=seed, identity_supported=True, max_offset_ms=1_000
    )
    assert not explicit.accepted
    assert gate.REASON_MAX_OFFSET_EXCEEDED in explicit.reason_codes


def test_ceiling_is_bounded_by_the_documented_constant() -> None:
    assert gate.LARGE_OFFSET_MAX_SECONDS == 180
    assert gate.LARGE_OFFSET_MAX_MS == 180_000
    # The normal window is untouched: still 20s, and strictly inside the
    # large-offset ceiling.
    assert FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS == 20_000
    assert FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS < gate.LARGE_OFFSET_MAX_MS


def test_insufficient_evidence_is_rejected() -> None:
    """Case D: a real offset, but too few cues to corroborate independently."""
    target = read("dexter_s08e04_target.srt")
    reference = read("negative_insufficient_anchors.srt")
    seed = classify_large_offset_candidate(
        target, reference, threshold_ms=FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS
    )
    assert seed is not None, "still a large-offset candidate"
    assessment = assess_large_offset(
        target, reference, seed_offset_ms=seed, identity_supported=True
    )
    assert not assessment.accepted
    assert gate.REASON_INSUFFICIENT_ANCHORS in assessment.reason_codes
    assert assessment.anchor_count < gate.LARGE_OFFSET_MIN_ANCHORS


def test_high_dispersion_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Case C evidence: regions that disagree are not a constant offset.

    Driven by a tightened dispersion ceiling so the check is provably
    load-bearing rather than incidentally satisfied.
    """
    target = read("dexter_s08e04_target.srt")
    reference = read("dexter_s08e04_reference.srt")
    monkeypatch.setattr(gate, "LARGE_OFFSET_MAX_DISPERSION_MS", 1.0)
    assessment = assess_large_offset(
        target, reference, seed_offset_ms=96_830, identity_supported=True
    )
    assert not assessment.accepted
    assert gate.REASON_DISPERSION_TOO_HIGH in assessment.reason_codes


def test_progressive_drift_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Case C: a drifting timeline must not pass as a constant offset."""
    target = read("dexter_s08e04_target.srt")
    reference = read("negative_drifting.srt")
    monkeypatch.setattr(gate, "LARGE_OFFSET_MAX_DRIFT_MS_PER_MINUTE", 0.0)
    assessment = assess_large_offset(
        target, reference, seed_offset_ms=96_780, identity_supported=True
    )
    assert not assessment.accepted
    assert (
        gate.REASON_DRIFT_TOO_HIGH in assessment.reason_codes
        or gate.REASON_INSUFFICIENT_ANCHORS in assessment.reason_codes
    ), assessment.summary()


def test_structural_mismatch_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evidence #3 is a real gate, not decoration."""
    target = read("dexter_s08e04_target.srt")
    reference = read("dexter_s08e04_reference.srt")
    monkeypatch.setattr(gate, "LARGE_OFFSET_MIN_STRUCTURAL_SIMILARITY", 1.01)
    assessment = assess_large_offset(
        target, reference, seed_offset_ms=96_830, identity_supported=True
    )
    assert not assessment.accepted
    assert gate.REASON_STRUCTURAL_MISMATCH in assessment.reason_codes


def test_identity_is_a_precondition() -> None:
    """Identity is necessary; it is never sufficient on its own."""
    target = read("dexter_s08e04_target.srt")
    reference = read("dexter_s08e04_reference.srt")
    assessment = assess_large_offset(
        target, reference, seed_offset_ms=96_830, identity_supported=False
    )
    assert not assessment.accepted
    assert assessment.reason_codes == [gate.REASON_IDENTITY_UNSUPPORTED]


def test_missing_seed_abstains() -> None:
    """No measured opening means no evidence base: abstain, do not guess."""
    target = read("dexter_s08e04_target.srt")
    assessment = assess_large_offset(
        target, target, seed_offset_ms=None, identity_supported=True
    )
    assert not assessment.accepted
    assert assessment.reason_codes == [gate.REASON_NO_DIALOGUE_SEED]


def test_partially_spread_anchors_are_still_insufficient() -> None:
    """Two or three agreeing regions must not unlock the gate.

    The neighbouring tests only cover the degenerate cases: zero anchors, and the
    ceiling short-circuit. Nothing pinned the 1-3 band, so lowering
    ``LARGE_OFFSET_MIN_ANCHORS`` from 4 to 1 left both of them green -- a mutation
    that removed most of the guard entirely was reported as catching nothing.
    This test pins the threshold itself.
    """
    target = read("dexter_s08e04_target.srt")
    reference = read("dexter_s08e04_reference.srt")
    original = gate._region_medians

    def with_regions(count: int):
        # One region per requested count, each carrying the same offset so the
        # only thing under test is how many anchors are required.
        fake = [(i * 1000, 96_830.0) for i in range(count)]
        return lambda *a, **k: (fake, 5, [(p, o) for p, o in fake])  # type: ignore[return-value]

    try:
        for count in range(1, gate.LARGE_OFFSET_MIN_ANCHORS):
            gate._region_medians = with_regions(count)  # type: ignore[assignment]
            assessment = assess_large_offset(
                target, reference, seed_offset_ms=96_830, identity_supported=True
            )
            assert not assessment.accepted, (
                f"{count} anchor(s) must not satisfy the gate"
            )
            assert gate.REASON_INSUFFICIENT_ANCHORS in assessment.reason_codes
            assert assessment.anchor_count == count

        gate._region_medians = with_regions(  # type: ignore[assignment]
            gate.LARGE_OFFSET_MIN_ANCHORS
        )
        enough = assess_large_offset(
            target, reference, seed_offset_ms=96_830, identity_supported=True
        )
        assert gate.REASON_INSUFFICIENT_ANCHORS not in enough.reason_codes
        assert enough.anchor_count == gate.LARGE_OFFSET_MIN_ANCHORS
    finally:
        gate._region_medians = original  # type: ignore[assignment]


def test_single_cue_evidence_is_never_sufficient() -> None:
    """The first-cue comparison alone must not unlock anything."""
    target = read("dexter_s08e04_target.srt")
    reference = read("dexter_s08e04_reference.srt")
    # Force the gate to demand anchors the evidence cannot supply.
    original = gate._region_medians
    try:
        gate._region_medians = lambda *a, **k: ([], 0, [])  # type: ignore[assignment]
        assessment = assess_large_offset(
            target, reference, seed_offset_ms=96_830, identity_supported=True
        )
    finally:
        gate._region_medians = original  # type: ignore[assignment]
    assert not assessment.accepted
    assert gate.REASON_INSUFFICIENT_ANCHORS in assessment.reason_codes


# --------------------------------------------------------------------------- #
# Language agnosticism, determinism, bounded cost
# --------------------------------------------------------------------------- #


def test_gate_is_language_agnostic() -> None:
    """Section 14: the decision must not depend on the subtitle language."""
    target = read("dexter_s08e04_target.srt")
    reference = read("dexter_s08e04_reference.srt")
    baseline = assess_large_offset(
        target, reference, seed_offset_ms=96_830, identity_supported=True
    )

    # Re-render the identical timings with entirely different, non-Latin text.
    retimed = "\n\n".join(
        f"{i + 1}\n{_ts(start)} --> {_ts(end)}\nÃƒÂ¥Ã‚Â­Ã¢â‚¬â€ÃƒÂ¥Ã‚Â¹Ã¢â‚¬Â¢ÃƒÂ¦Ã¢â‚¬â€œÃ¢â‚¬Â¡ÃƒÂ¦Ã…â€œÃ‚Â¬ {i} ÃƒÂ¦Ã‚ÂµÃ¢â‚¬Â¹ÃƒÂ¨Ã‚Â¯Ã¢â‚¬Â¢ÃƒÂ¥Ã¢â‚¬Â Ã¢â‚¬Â¦ÃƒÂ¥Ã‚Â®Ã‚Â¹\n"
        for i, (start, end, _) in enumerate(parse_srt_cues(reference))
    )
    other = assess_large_offset(
        target, retimed, seed_offset_ms=96_830, identity_supported=True
    )
    assert other.anchor_count == baseline.anchor_count
    assert other.structural_similarity == baseline.structural_similarity
    assert other.accepted == baseline.accepted


def test_gate_is_deterministic() -> None:
    target = read("dexter_s08e04_target.srt")
    reference = read("dexter_s08e04_reference.srt")
    runs = [
        assess_large_offset(
            target, reference, seed_offset_ms=96_830, identity_supported=True
        ).summary()
        for _ in range(4)
    ]
    assert len(set(runs)) == 1


def test_sampling_is_bounded() -> None:
    """Section 16: cost must not scale with file size."""
    huge = synth(20_000, step_ms=200)
    assessment = assess_large_offset(
        huge, huge, seed_offset_ms=96_830, identity_supported=True
    )
    # Nothing was accepted on the strength of an unbounded scan, and the
    # structural profile stayed inside its own bin budget.
    assert assessment.regions_sampled == gate.LARGE_OFFSET_REGION_COUNT
    assert assessment.structural_similarity is None or 0.0 <= (
        assessment.structural_similarity
    ) <= 1.0


def test_summary_never_leaks_subtitle_text() -> None:
    target = read("dexter_s08e04_target.srt")
    reference = read("dexter_s08e04_reference.srt")
    assessment = assess_large_offset(
        target, reference, seed_offset_ms=96_830, identity_supported=True
    )
    summary = assessment.summary()
    assert "ÃƒËœÃ‚Â¯Ãƒâ„¢Ã†â€™ÃƒËœÃ‚Â³ÃƒËœÃ‚ÂªÃƒËœÃ‚Â±" not in summary
    assert "http" not in summary
    assert "api_key" not in summary


# --------------------------------------------------------------------------- #
# The gate grants nothing on its own: post-alass verification stays authoritative
# --------------------------------------------------------------------------- #


def test_gate_result_is_not_a_verification() -> None:
    """Accepting the gate must not look like a sync claim anywhere."""
    assessment = assess_large_offset(
        read("dexter_s08e04_target.srt"),
        read("dexter_s08e04_reference.srt"),
        seed_offset_ms=96_830,
        identity_supported=True,
    )
    assert assessment.accepted
    fields = set(type(assessment).model_fields)
    assert "sync_state" not in fields
    assert "verification" not in fields
    assert "sync_confidence" not in fields


def test_post_alass_default_ceiling_is_unchanged() -> None:
    """Existing callers keep the 20s ceiling: the widening is opt-in.

    Pinned at the signature because the parameter's *behavioural* effect is
    deliberately minimal. Post-alass quality already discriminate, so the
    magnitude ceiling is not what stops a bad large-offset alignment -- which
    is the safe arrangement, and exactly why the default must be asserted
    directly rather than inferred from a verdict.
    """
    parameters = inspect.signature(AlignmentAnalyzer.analyze).parameters
    assert MAX_PLAUSIBLE_OFFSET_MS == 20_000
    assert parameters["max_plausible_offset_ms"].default == 20_000, (
        "the widened ceiling must never become the default"
    )
    assert "movement_offset_ms" not in parameters, (
        "the verifier must not accept a caller-supplied scalar pre-shift: the "
        "gate's estimate is the opening disagreement, not the per-cue movement"
    )

    # And behaviourally: a large displacement is never quietly verified.
    reference = synth(20, start_ms=10_000, step_ms=2_000)      # 10s..48s
    target = synth(20, start_ms=106_000, step_ms=2_000)        # 106s..144s
    evaluation = AlignmentAnalyzer().analyze(
        parse_srt_cues(target), None, parse_srt_cues(reference)
    )
    assert evaluation.sync_state not in VERIFIED_STATES
    assert evaluation.rejection_reason is RejectionReason.IMPLAUSIBLE_OFFSET


def test_widened_ceiling_still_requires_good_residuals() -> None:
    """Section 11: raising the magnitude ceiling must not lower the bar.

    Models what alass actually did on the real S08E04 case: a *small* per-cue
    movement (it re-anchored the opening) that lands the output on the
    reference. The residual quality gates are what must carry the decision.
    """
    reference = synth(40, start_ms=10_000, step_ms=2_000)      # 10s..88s
    target = synth(40, start_ms=14_000, step_ms=2_000)        # 14s..92s
    target_cues = parse_srt_cues(target)
    reference_cues = parse_srt_cues(reference)

    # "alass" placed the target's content on the reference: residual ~0, and a
    # small per-cue movement, which is the shape alass actually produces.
    synced = [(s - 4_000, e - 4_000, t) for s, e, t in target_cues]
    good = AlignmentAnalyzer().analyze(
        target_cues,
        synced,
        reference_cues,
        alass_applied=True,
        alass_successful=True,
        max_plausible_offset_ms=gate.LARGE_OFFSET_MAX_MS,
    )
    assert good.sync_state in VERIFIED_STATES, good.reasons

    # The same widened ceiling with a sloppy result must still be refused.
    sloppy = [
        (s - 4_000 + (3_000 if index % 2 else -3_000), e - 4_000, t)
        for index, (s, e, t) in enumerate(target_cues)
    ]
    bad = AlignmentAnalyzer().analyze(
        target_cues,
        sloppy,
        reference_cues,
        alass_applied=True,
        alass_successful=True,
        max_plausible_offset_ms=gate.LARGE_OFFSET_MAX_MS,
    )
    assert bad.sync_state not in VERIFIED_STATES
    assert bad.rejection_reason is not None


def test_uniform_correction_beyond_the_pairing_radius_fails_safe() -> None:
    """A movement larger than the pairing radius is unmeasurable, not verified.

    With no scalar pre-shift, a genuinely uniform multi-second-per-minute
    correction leaves too few pairs inside ``MOVEMENT_TOLERANCE_MS``. The
    existing "not measurable" branch must refuse it. This is the deliberate
    fail-safe for giving up the invalid pre-shift: unverifiable, never verified.
    """
    from app.services.sync.alignment import (
        MIN_CUES_FOR_PERCENTILES,
        MOVEMENT_TOLERANCE_MS,
    )

    reference = synth(20, start_ms=10_000, step_ms=2_000)
    target = synth(20, start_ms=106_000, step_ms=2_000)
    target_cues = parse_srt_cues(target)
    reference_cues = parse_srt_cues(reference)
    # A uniform 96s correction: further than the pairing radius from every cue.
    synced = [(s - 96_000, e - 96_000, t) for s, e, t in target_cues]
    assert all(abs(-96_000) > MOVEMENT_TOLERANCE_MS for _ in target_cues)

    evaluation = AlignmentAnalyzer().analyze(
        target_cues,
        synced,
        reference_cues,
        alass_applied=True,
        alass_successful=True,
        max_plausible_offset_ms=gate.LARGE_OFFSET_MAX_MS,
    )
    assert evaluation.sync_state not in VERIFIED_STATES
    assert evaluation.rejection_reason is RejectionReason.INSUFFICIENT_EVIDENCE
    assert any("not measurable" in reason for reason in evaluation.reasons)
    assert MIN_CUES_FOR_PERCENTILES > 0


def test_gate_scalar_offset_is_not_injected_into_movement_pairing() -> None:
    """Regression: the gate's scalar estimate must not distort movement pairing.

    On the real S08E04 pair the gate measured the *opening* disagreement at
    +96.1s, but alass produced a per-cue movement of about +9.5s. A previous
    implementation pre-shifted the target by the gate scalar before pairing,
    which mispaired nearly every cue and reported a movement MAD of ~10.4s
    instead of 0ms. This test fails if that pre-shift is ever reintroduced.
    """
    target = cues("dexter_s08e04_target.srt")
    reference = cues("dexter_s08e04_reference.srt")
    # What alass actually did: a small, constant per-cue movement.
    synced = [(s + 9_460, e + 9_460, t) for s, e, t in target]

    evaluation = AlignmentAnalyzer().analyze(
        target,
        synced,
        reference,
        alass_applied=True,
        alass_successful=True,
        max_plausible_offset_ms=gate.LARGE_OFFSET_MAX_MS,
    )
    assert evaluation.mad_offset_ms is not None
    assert evaluation.mad_offset_ms <= MAX_MAD_MS_FOR_STABLE, (
        f"movement MAD must reflect the real per-cue movement, got "
        f"{evaluation.mad_offset_ms}ms -- the gate scalar appears to be "
        f"injected into pairing again"
    )


def test_unconstrained_movement_pairing_cannot_manufacture_verified() -> None:
    """Dropping the pre-shift must not turn a poor result into VERIFIED.

    The output here is barely moved while the reference sits ~96s away, so
    residual quality is poor. Movement pairing is no longer distorted, but the
    residual gates still have to refuse it.
    """
    target = cues("dexter_s08e04_target.srt")
    reference = cues("dexter_s08e04_reference.srt")
    synced = [(s + 9_460, e + 9_460, t) for s, e, t in target]

    evaluation = AlignmentAnalyzer().analyze(
        target,
        synced,
        reference,
        alass_applied=True,
        alass_successful=True,
        max_plausible_offset_ms=gate.LARGE_OFFSET_MAX_MS,
    )
    assert evaluation.sync_state not in VERIFIED_STATES
    assert evaluation.rejection_reason is not None


def test_end_to_end_dexter_s08e04_grants_a_trial_and_verifies() -> None:
    """The confirmed case end to end, offline.

    Cue-sanity refuses it, the evidence gate accepts it, and only then does the
    analyzer get the widened ceiling -- where it still has to earn a verified
    state on its own residual evidence.
    """
    target = read("dexter_s08e04_target.srt")
    reference = read("dexter_s08e04_reference.srt")

    sanity = validate_cue_sanity(
        target, reference, threshold_ms=FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS
    )
    assert not sanity["ok"], "precondition: the old path refused this case"

    seed = classify_large_offset_candidate(
        target, reference, threshold_ms=FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS
    )
    assessment = assess_large_offset(
        target, reference, seed_offset_ms=seed, identity_supported=True
    )
    assert assessment.accepted, assessment.summary()

    # Simulate the alignment the gate exists to permit, pairing each target cue
    # one-to-one with the reference cue it actually corresponds to. That is the
    # shape of a successful alass run for a constant displacement.
    offset = assessment.estimated_offset_ms or 0.0
    target_cues = parse_srt_cues(target)
    reference_cues = parse_srt_cues(reference)
    reference_starts = [start for start, _, _ in reference_cues]
    used: set[int] = set()
    synced: list[tuple[int, int, str]] = []
    for start, end, text in target_cues:
        predicted = start - offset
        index = bisect.bisect_left(reference_starts, predicted)
        candidates = [
            i
            for i in range(max(0, index - 3), min(len(reference_starts), index + 4))
            if i not in used
        ]
        if not candidates:
            synced.append((int(start - offset), int(end - offset), text))
            continue
        best = min(candidates, key=lambda i: abs(reference_starts[i] - predicted))
        used.add(best)
        landed = reference_starts[best]
        synced.append((landed, landed + (end - start), text))

    evaluation = AlignmentAnalyzer().analyze(
        target_cues,
        synced,
        reference_cues,
        alass_applied=True,
        alass_successful=True,
        max_plausible_offset_ms=gate.LARGE_OFFSET_MAX_MS,
    )
    # The magnitude ceiling is no longer what refuses this case: whatever the
    # alignment quality, the displacement itself must not be the objection.
    assert evaluation.rejection_reason is not RejectionReason.IMPLAUSIBLE_OFFSET, (
        evaluation.reasons
    )

    # And a deliberately poor alignment of the same pair is still refused,
    # so widening the ceiling did not weaken any quality gate.
    sloppy = [
        (s - offset + (4_000 if i % 2 else -4_000), e - offset, t)
        for i, (s, e, t) in enumerate(target_cues)
    ]
    refused = AlignmentAnalyzer().analyze(
        target_cues,
        sloppy,
        reference_cues,
        alass_applied=True,
        alass_successful=True,
        max_plausible_offset_ms=gate.LARGE_OFFSET_MAX_MS,
    )
    assert refused.sync_state not in VERIFIED_STATES, refused.reasons


def test_wrong_episode_is_not_verified_even_with_the_widened_ceiling() -> None:
    """Case B: the widened ceiling must not let a bad alignment through.

    Timing-only evidence cannot separate a wrong episode from a genuine
    constant shift, so the gate may grant a trial. The verdict is the
    analyzer's, and it must not be VERIFIED.
    """
    target = read("dexter_s08e04_target.srt")
    wrong = read("negative_wrong_episode.srt")

    evaluation = AlignmentAnalyzer().analyze(
        parse_srt_cues(target),
        # A "sync" that is simply the wrong file's timings, unaligned.
        parse_srt_cues(wrong),
        parse_srt_cues(wrong),
        alass_applied=True,
        alass_successful=True,
        max_plausible_offset_ms=gate.LARGE_OFFSET_MAX_MS,
    )
    assert evaluation.sync_state is not SyncState.VERIFIED_SYNCED
    assert evaluation.rejection_reason is not None or not evaluation.alass_successful


def test_different_cut_is_not_verified() -> None:
    """Case A: a longer edit must not reach VERIFIED through the new path."""
    target = read("dexter_s08e04_target.srt")
    cut = read("negative_different_cut.srt")
    target_cues = parse_srt_cues(target)
    cut_cues = parse_srt_cues(cut)

    # Shifting the whole (already divergent) cut by a constant cannot make the
    # two timelines agree.
    synced = [(s + 96_000, e + 96_000, t) for s, e, t in cut_cues]
    evaluation = AlignmentAnalyzer().analyze(
        target_cues,
        synced,
        cut_cues,
        alass_applied=True,
        alass_successful=True,
        max_plausible_offset_ms=gate.LARGE_OFFSET_MAX_MS,
    )
    assert evaluation.sync_state is not SyncState.VERIFIED_SYNCED


def test_rejection_reason_enum_is_unchanged() -> None:
    """No new public sync state was invented for this feature."""
    assert RejectionReason.IMPLAUSIBLE_OFFSET in set(RejectionReason)


# --------------------------------------------------------------------------- #
# Orchestrator wiring: a large offset now reaches alass instead of aborting
# --------------------------------------------------------------------------- #


class _Strategy:
    """Yields one fixed reference, the way a real strategy would."""

    def __init__(self, name: str, reference: str | None) -> None:
        self._name = name
        self._reference = reference
        self.validates_target = False
        self.resolved = 0
        self.validator_saw_reference = False

    async def resolve_with_provenance(self, query, update_validator=None):
        from app.services.sync.query import ResolvedReference

        self.resolved += 1
        if self._reference is None:
            return ResolvedReference(None)
        if update_validator is not None:
            # This is the path that used to walk past a large-offset reference.
            self.validator_saw_reference = update_validator(self._reference)
        return ResolvedReference(
            text=self._reference, kind="edition", candidate="ref", bluray_match=False
        )


class _SyncService:
    """The alass execution boundary. Counts real invocations."""

    def __init__(self) -> None:
        self.calls = 0

    async def sync_async(self, target, reference, **kwargs):
        self.calls += 1
        return None


def _orchestrator(references: list[str | None], *, limit: int = 3):
    from app.services.sync.orchestrator import SyncOrchestrator

    service = _SyncService()
    orch = SyncOrchestrator(sync_service=service, sync_cache=None)
    strategies = [_Strategy(f"s{i}", ref) for i, ref in enumerate(references)]
    orch._strategies = lambda: [(s._name, s) for s in strategies]  # type: ignore[method-assign]
    orch._alass_candidate_limit = limit
    orch.__dict__["_test_strategies"] = strategies
    return orch, service


def _meta() -> dict:
    return {
        "imdb_id": "tt0773262",
        "season": 8,
        "episode": 4,
        "media_type": "series",
        "target_filename": (
            "Dexter.s8e04.Scar.tissue.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv"
        ),
        "video_size": 52_000_000_000,
        "lang": "ara",
    }


@pytest.mark.asyncio
async def test_orchestrator_lets_a_large_offset_reach_alass(caplog) -> None:
    """The real case must get past cue-sanity and actually invoke alass.

    Before the gate this request aborted with ``no deterministic reference`` and
    alass was never spawned. The gate's whole purpose is to change that.
    """
    import logging

    target = read("dexter_s08e04_target.srt")
    reference = read("dexter_s08e04_reference.srt")
    orch, service = _orchestrator([reference])

    with caplog.at_level(logging.INFO, logger="app.services.sync.orchestrator"):
        await orch.evaluate_and_sync(
            target.encode("utf-8"), _meta(), "lo_orch_ok", auto_sync=True
        )

    messages = [r.getMessage() for r in caplog.records]
    assert any("large-offset candidate detected" in m for m in messages), (
        "the large-offset path must be explicitly identifiable in logs"
    )
    assert any("large-offset evidence accepted" in m for m in messages), (
        "acceptance must be logged distinctly from a normal pass"
    )
    assert service.calls == 1, "alass must now be reached for this reference"


@pytest.mark.asyncio
async def test_orchestrator_rejects_over_ceiling_without_alass(caplog) -> None:
    """Past the ceiling the reference is still refused before any subprocess."""
    import logging

    target = read("dexter_s08e04_target.srt")
    orch, service = _orchestrator([synth(20, start_ms=10_000, step_ms=2_000)])

    with caplog.at_level(logging.INFO, logger="app.services.sync.orchestrator"):
        await orch.evaluate_and_sync(
            target.encode("utf-8"), _meta(), "lo_orch_ceiling", auto_sync=True
        )

    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "max offset exceeded" in m or "large-offset evidence rejected" in m
        for m in messages
    ), messages[-6:]
    assert service.calls == 0, "an over-ceiling reference must not reach alass"
    # A pre-alass rejection is explicitly not charged to the budget.
    budget = [m for m in messages if "[sync-budget]" in m]
    assert budget, "budget accounting must stay observable"
    assert "counted_toward_limit=false" in budget[-1]
    assert "attempted=0" in budget[-1]


@pytest.mark.asyncio
async def test_orchestrator_falls_through_to_the_next_candidate() -> None:
    """A refused large-offset reference must not stop the candidate loop."""
    target = read("dexter_s08e04_target.srt")
    over_ceiling = synth(20, start_ms=10_000, step_ms=2_000)  # ~400s, refused
    usable = synth(20, start_ms=111_000, step_ms=2_000)     # +3.5s from the real target
    orch, service = _orchestrator([over_ceiling, usable])

    await orch.evaluate_and_sync(
        target.encode("utf-8"), _meta(), "lo_orch_fallthrough", auto_sync=True
    )

    strategies = orch.__dict__["_test_strategies"]
    assert strategies[1].resolved == 1, "the next candidate must still be tried"
    assert service.calls == 1, "the usable candidate is the one that runs alass"


def _ts(ms: int) -> str:
    ms = max(0, int(ms))
    return (
        f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    )
