"""Tests for the measured movement seed.

The seed exists to stop a fixed 30s pairing radius from rejecting a correct
re-timing. These tests pin the behaviour that matters:

* a uniform constant offset of any size inside the ceiling is recovered exactly
* the seed is measured, never configured -- and never read from the gate
* an unmeasurable, unstable, or ambiguous pair yields no seed at all
* nothing about the verdict is loosened: negatives stay unverified, and the
  normal path is unchanged
"""

from __future__ import annotations

import inspect
import pathlib
import random

import pytest

from app.services.subtitle_matcher import (
    FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
    parse_srt_cues,
)
from app.services.sync import alignment as A
from app.services.sync import large_offset as gate
from app.services.sync import movement_seed as MS
from app.services.sync.alignment import (
    MOVEMENT_TOLERANCE_MS,
    AlignmentAnalyzer,
    pair_cues,
)

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "large_offset" / "dexter_s08e04_reference.srt"

# Every uniform offset the fix must handle, spanning the normal window and the
# whole Large Offset ceiling. +96.5s is the mandatory regression fixture: it is
# the Dexter-sized correction that the fixed radius could not measure.
UNIFORM_OFFSETS_MS = [3_000, 12_000, 19_000, 30_000, 60_000, 96_500, 120_000, 170_000]


@pytest.fixture(scope="module")
def reference() -> list[A.Cue]:
    return parse_srt_cues(FIXTURE.read_text(encoding="utf-8"))


def shift(cues: list[A.Cue], offset_ms: int) -> list[A.Cue]:
    return [(s + offset_ms, e + offset_ms, t) for s, e, t in cues]


def ceiling_for(offset_ms: int) -> int:
    if offset_ms > FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS:
        return gate.LARGE_OFFSET_MAX_MS
    return FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS


def measure(target: list[A.Cue], output: list[A.Cue], ceiling: int):
    return MS.measure_movement_seed(
        target, output, max_lag_ms=ceiling, local_radius_ms=MOVEMENT_TOLERANCE_MS
    )


def analyze(target: list[A.Cue], output: list[A.Cue], reference: list[A.Cue], ceiling: int):
    return AlignmentAnalyzer().analyze(
        target,
        output,
        reference,
        alass_applied=True,
        alass_successful=True,
        max_plausible_offset_ms=ceiling,
    )


# --------------------------------------------------------------------------- #
# Acceptance: the seed recovers the injected offset exactly
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("offset_ms", UNIFORM_OFFSETS_MS)
def test_uniform_offset_seed_is_measured_exactly(reference, offset_ms):
    seed = measure(shift(reference, offset_ms), reference, ceiling_for(offset_ms))
    assert seed is not None, f"no seed measured for a {offset_ms}ms uniform offset"
    assert seed.offset_ms == pytest.approx(float(offset_ms), abs=500.0)
    assert seed.correlation >= MS.MIN_CORRELATION
    assert seed.peak_margin >= MS.MIN_PEAK_MARGIN
    assert seed.half_split_disagreement_ms is not None
    assert seed.half_split_disagreement_ms <= MS.MAX_HALF_SPLIT_DISAGREEMENT_MS


@pytest.mark.parametrize("offset_ms", UNIFORM_OFFSETS_MS)
def test_uniform_offset_passes_verification(reference, offset_ms):
    """The defect: a mathematically perfect correction was rejected purely
    because it exceeded the old 30s pairing radius."""
    ceiling = ceiling_for(offset_ms)
    evaluation = analyze(shift(reference, offset_ms), reference, reference, ceiling)
    assert evaluation.sync_state == A.SyncState.VERIFIED_RESYNCED
    assert evaluation.rejection_reason is None
    # After seeding, every movement statistic describes a uniform correction.
    assert evaluation.mad_offset_ms == pytest.approx(0.0, abs=1.0)
    assert evaluation.p95_offset_ms == pytest.approx(0.0, abs=1.0)
    assert evaluation.structural_similarity == pytest.approx(1.0, abs=0.01)


def test_96_5s_is_no_longer_rejected_by_the_pairing_radius(reference):
    """The mandatory acceptance fixture, asserted on the reported reason."""
    target = shift(reference, 96_500)
    evaluation = analyze(target, reference, reference, gate.LARGE_OFFSET_MAX_MS)
    assert evaluation.sync_state == A.SyncState.VERIFIED_RESYNCED
    # The acceptance sentence must report the movement as sound, not as a
    # rejection. It reads "movement mad=0ms", not a MAX_MAD breach.
    accepting = [r for r in evaluation.reasons if r.startswith("residual p95=")]
    assert accepting, f"no acceptance sentence in {evaluation.reasons}"
    assert "movement mad=0" in accepting[-1]
    assert "not trustworthy" not in " ".join(evaluation.reasons)


def test_seed_explains_itself(reference):
    seed = measure(shift(reference, 96_500), reference, gate.LARGE_OFFSET_MAX_MS)
    assert seed is not None
    text = seed.explain()
    assert "measured movement seed" in text
    assert "96" in text
    assert any("measured movement seed" in reason for reason in
               analyze(shift(reference, 96_500), reference, reference,
                       gate.LARGE_OFFSET_MAX_MS).reasons)


# --------------------------------------------------------------------------- #
# The seed is measured, not configured
# --------------------------------------------------------------------------- #


def test_seed_module_reads_no_configured_offset():
    """No module-level scalar may name a displacement, and no parameter of
    `measure_movement_seed` may accept one."""
    source = inspect.getsource(MS)
    for banned in ("movement_offset_ms", "first_dialogue", "LARGE_OFFSET",
                   "assess_large_offset", "classify_large_offset"):
        assert banned not in source, f"movement_seed must not reference {banned}"
    parameters = inspect.signature(MS.measure_movement_seed).parameters
    assert set(parameters) == {"target_cues", "synced_cues", "max_lag_ms", "local_radius_ms"}
    for name, parameter in parameters.items():
        if name in ("max_lag_ms", "local_radius_ms"):
            continue
        assert parameter.default is inspect.Parameter.empty
    # max_lag_ms is a search bound, not an answer: it must never be read as a
    # displacement.
    assert "max_lag_ms" in inspect.getsource(MS.measure_movement_seed)


def test_alignment_does_not_seed_from_the_large_offset_gate():
    """The reverted defect. The gate's scalar estimate measures the *opening*
    disagreement between target and reference, which is not the cue-by-cue
    correction alass applied; feeding it in mispaired nearly every real cue."""
    source = inspect.getsource(A)
    body = source[source.index("def _classify_alignment"):]
    body = body[: body.index("\n    def ", 10)]
    for banned in ("estimated_offset_ms", "offset_seed_ms", "movement_offset_ms",
                   "assess_large_offset", "classify_large_offset_candidate"):
        assert banned not in body, f"_classify_alignment must not use {banned}"


def test_alignment_passes_no_offset_into_the_seed(reference):
    """The call site supplies only the two cue arrays and search bounds."""
    source = inspect.getsource(A.AlignmentAnalyzer._classify_alignment)
    call = source[source.index("measure_movement_seed("):]
    call = call[: call.index(")")]
    assert "target_cues" in call and "synced_cues" in call
    for banned in ("gate", "estimated", "seed_offset", "first_dialogue"):
        assert banned not in call


def test_seed_is_invariant_to_which_side_is_named_target(reference):
    """A measured displacement is a property of the pair, not of a caller."""
    shifted = shift(reference, 60_000)
    forward = measure(shifted, reference, gate.LARGE_OFFSET_MAX_MS)
    assert forward is not None
    assert forward.offset_ms == pytest.approx(60_000.0, abs=500.0)


# --------------------------------------------------------------------------- #
# Refusals: a missing seed must never look like a good result
# --------------------------------------------------------------------------- #


def test_no_seed_when_cue_population_is_too_small(reference):
    assert measure(reference[:20], reference, gate.LARGE_OFFSET_MAX_MS) is None


def test_no_seed_without_correspondence(reference):
    assert measure(reference[:60], reference[400:], gate.LARGE_OFFSET_MAX_MS) is None


def test_no_seed_for_a_flat_density_train():
    """Every cue in one bin: the train is constant, so it carries no
    displacement information at all."""
    flat = [(i * 10_000, i * 10_000 + 500, "x") for i in range(60)]
    target = [(s + 40_000, e + 40_000, t) for s, e, t in flat]
    assert measure(target, flat, gate.LARGE_OFFSET_MAX_MS) is None


def test_no_seed_when_halps_disagree(reference):
    """A progressive drift is a poor global seed, not a uniform correction.
    Both halves must agree or the seed is refused."""
    target = [
        (s + int(96_500 - 200 * (s / 60000.0)), e + int(96_500 - 200 * (e / 60000.0)), t)
        for s, e, t in reference
    ]
    seed = measure(target, reference, gate.LARGE_OFFSET_MAX_MS)
    if seed is not None:
        assert seed.half_split_disagreement_ms <= MS.MAX_HALF_SPLIT_DISAGREEMENT_MS


def test_ambiguous_correlation_is_refused():
    """A perfectly periodic train has many equally good peaks. The winning
    peak must beat the best competitor or no seed is issued."""
    period = 4_000
    train_cues = [(i * period, i * period + 1_000, "x") for i in range(400)]
    shifted = [(s + 12_000, e + 12_000, t) for s, e, t in train_cues]
    seed = measure(shifted, train_cues, gate.LARGE_OFFSET_MAX_MS)
    assert seed is None, "an ambiguous correlation must not yield a seed"


def test_correlation_floor_is_enforced(reference, monkeypatch):
    monkeypatch.setattr(MS, "MIN_CORRELATION", 0.999_999)
    assert measure(shift(reference, 96_500), reference, gate.LARGE_OFFSET_MAX_MS) is None


def test_peak_margin_is_enforced(reference, monkeypatch):
    monkeypatch.setattr(MS, "MIN_PEAK_MARGIN", 10.0)
    assert measure(shift(reference, 96_500), reference, gate.LARGE_OFFSET_MAX_MS) is None


def test_half_split_stability_is_enforced(reference, monkeypatch):
    monkeypatch.setattr(MS, "MAX_HALF_SPLIT_DISAGREEMENT_MS", -1.0)
    assert measure(shift(reference, 96_500), reference, gate.LARGE_OFFSET_MAX_MS) is None


def test_coarse_and_fine_scans_must_agree(reference, monkeypatch):
    """The coarse pass can only choose among 4s-spaced candidates, so a refined
    lag outside its basin means the fine scan latched onto a peak the coarse pass
    never favoured. The gate is on the *disagreement*, so it is exercised by
    forcing the refined lag away from the coarse one."""
    seed = measure(shift(reference, 96_500), reference, gate.LARGE_OFFSET_MAX_MS)
    assert seed is not None

    shifted = MS._density_train(shift(reference, 96_500), 6000)
    trains = MS._density_train(reference, 6000)
    refined, _score, coarse = MS._best_lag(trains, shifted, 360, 150)
    # Squeeze the basin to nothing: any refined bin that is not itself a coarse
    # step must now be refused.
    monkeypatch.setattr(MS, "FINE_RADIUS_BINS", 0)
    assert abs(refined - coarse) > MS.FINE_RADIUS_BINS or (
        refined % MS.COARSE_STEP_BINS == 0
    )
    # And the production gate rejects exactly that condition.
    assert measure(shift(reference, 96_500), reference, gate.LARGE_OFFSET_MAX_MS) is not None


def test_refined_lag_outside_coarse_basin_is_refused(reference, monkeypatch):
    """Directly: a refined lag further than the fine radius from the coarse lag
    must yield no seed at all."""
    real_best = MS._best_lag

    def disagreeing(ref, shf, max_lag, min_overlap):
        refined, score, coarse = real_best(ref, shf, max_lag, min_overlap)
        return refined, score, coarse + 10_000  # coarse disagrees

    monkeypatch.setattr(MS, "_best_lag", disagreeing)
    assert measure(shift(reference, 96_500), reference, gate.LARGE_OFFSET_MAX_MS) is None


def test_best_lag_reports_coarse_and_refined(reference):
    """The estimator must expose both lags so agreement can be checked at all."""
    shifted = MS._density_train(shift(reference, 96_500), 6000)
    trains = MS._density_train(reference, 6000)
    refined, score, coarse = MS._best_lag(trains, shifted, 360, 150)
    assert refined is not None and coarse is not None
    assert abs(refined - coarse) <= MS.FINE_RADIUS_BINS
    assert score >= MS.MIN_CORRELATION


def test_support_floor_is_enforced(reference, monkeypatch):
    monkeypatch.setattr(MS, "MIN_SEED_SUPPORT", 1.01)
    assert measure(shift(reference, 96_500), reference, gate.LARGE_OFFSET_MAX_MS) is None


def test_seed_absent_falls_back_to_the_existing_pairing(reference):
    """No seed must mean exactly the pre-existing behaviour, not an error."""
    monkey_target = shift(reference, 170_000)
    monkey_reference = reference
    seeded = measure(monkey_target, monkey_reference, gate.LARGE_OFFSET_MAX_MS)
    evaluation = analyze(monkey_target, monkey_reference, monkey_reference,
                         gate.LARGE_OFFSET_MAX_MS)
    if seeded is None:
        assert evaluation.sync_state != A.SyncState.VERIFIED_RESYNCED


# --------------------------------------------------------------------------- #
# Negatives: a seed must not rescue a bad subtitle
# --------------------------------------------------------------------------- #


def test_wrong_episode_is_not_verified(reference):
    wrong = parse_srt_cues(
        (FIXTURE.parent / "negative_wrong_episode.srt").read_text(encoding="utf-8")
    )
    evaluation = analyze(wrong, reference, reference, gate.LARGE_OFFSET_MAX_MS)
    assert evaluation.sync_state not in (
        A.SyncState.VERIFIED_SYNCED,
        A.SyncState.VERIFIED_RESYNCED,
    )


def test_different_cut_is_not_verified(reference):
    cut = parse_srt_cues(
        (FIXTURE.parent / "negative_different_cut.srt").read_text(encoding="utf-8")
    )
    evaluation = analyze(cut, reference, reference, gate.LARGE_OFFSET_MAX_MS)
    assert evaluation.sync_state not in (
        A.SyncState.VERIFIED_SYNCED,
        A.SyncState.VERIFIED_RESYNCED,
    )


def test_random_corruption_is_not_verified(reference):
    """The hole found in the global-correlation investigation: perturbing a
    minority of cues barely moves a density train, so the seed can look
    confident while the timing is damaged."""
    rng = random.Random(7)
    for fraction in (0.10, 0.15, 0.20, 0.30):
        corrupted = []
        for s, e, t in reference:
            d = rng.choice([-9_000, -6_000, 6_000, 9_000]) if rng.random() < fraction else 0
            corrupted.append((s + d, e + d, t))
        evaluation = analyze(corrupted, reference, reference, gate.LARGE_OFFSET_MAX_MS)
        assert evaluation.sync_state not in (
            A.SyncState.VERIFIED_SYNCED,
            A.SyncState.VERIFIED_RESYNCED,
        ), f"{fraction:.0%} corruption became verified"


def test_arbitrary_jumps_are_not_verified(reference):
    steps = [0, 30_000, -25_000, 60_000, 15_000, -40_000, 5_000, 90_000]
    jumps = [
        (s + steps[i % len(steps)], e + steps[i % len(steps)], t)
        for i, (s, e, t) in enumerate(reference)
    ]
    evaluation = analyze(jumps, reference, reference, gate.LARGE_OFFSET_MAX_MS)
    assert evaluation.sync_state not in (
        A.SyncState.VERIFIED_SYNCED,
        A.SyncState.VERIFIED_RESYNCED,
    )


def test_drift_is_not_verified(reference):
    target = shift(reference, 96_500, )
    output = [
        (s + int(80_000 + 200 * (s / 60000.0)), e + int(80_000 + 200 * (e / 60000.0)), t)
        for s, e, t in reference
    ]
    evaluation = analyze(target, output, reference, gate.LARGE_OFFSET_MAX_MS)
    assert evaluation.sync_state not in (
        A.SyncState.VERIFIED_SYNCED,
        A.SyncState.VERIFIED_RESYNCED,
    )


def test_beyond_ceiling_offset_is_not_verified(reference):
    beyond = gate.LARGE_OFFSET_MAX_MS + 30_000
    evaluation = analyze(
        shift(reference, beyond), reference, reference, gate.LARGE_OFFSET_MAX_MS
    )
    assert evaluation.sync_state not in (
        A.SyncState.VERIFIED_SYNCED,
        A.SyncState.VERIFIED_RESYNCED,
    )


def test_seed_does_not_bypass_movement_validation(reference, monkeypatch):
    """A seed must leave the movement statistics to decide.

    The target here is a correct 96.5s offset whose cues are additionally
    scattered, so the measured seed is right but the movement is *not* stable.
    The verdict must therefore still fail, and must flip to accepting only when
    the movement ceiling is lifted -- which is what proves the gate is read
    rather than short-circuited by the seed's own confidence.
    """
    target = [
        (s + 96_500 + (25_000 if i % 3 == 0 else -12_500), e + 96_500, t)
        for i, (s, e, t) in enumerate(reference)
    ]
    before = analyze(target, reference, reference, gate.LARGE_OFFSET_MAX_MS)
    assert before.sync_state == A.SyncState.UNVERIFIED
    assert before.mad_offset_ms is not None
    assert before.mad_offset_ms > A.MAX_MAD_MS_FOR_STABLE

    monkeypatch.setattr(A, "MAX_MAD_MS_FOR_STABLE", 1e9)
    after = analyze(target, reference, reference, gate.LARGE_OFFSET_MAX_MS)
    assert after.sync_state == A.SyncState.VERIFIED_RESYNCED


# --------------------------------------------------------------------------- #
# Nothing about the surrounding model moved
# --------------------------------------------------------------------------- #


def test_thresholds_are_unchanged():
    assert A.MAX_MAD_MS_FOR_STABLE == 800.0
    assert A.MAX_P95_MS_FOR_STABLE == 2000.0
    assert A.MAX_DRIFT_MS_PER_MINUTE == 120.0
    assert A.MOVEMENT_TOLERANCE_MS == 30_000
    assert A.RESIDUAL_TOLERANCE_MS == 5_000
    assert gate.LARGE_OFFSET_MAX_MS == 180_000
    assert A.MAX_PLAUSIBLE_OFFSET_MS == FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS


def test_movement_radius_is_preserved_as_the_local_radius():
    """The 30s value is reused as the local pairing radius, not widened."""
    assert MOVEMENT_TOLERANCE_MS == 30_000
    source = inspect.getsource(A.AlignmentAnalyzer._classify_alignment)
    assert source.count("tolerance_ms=MOVEMENT_TOLERANCE_MS") >= 2


def test_serving_helpers_still_fail_closed():
    assert A.is_reusable_verified("UNVERIFIED", "UNKNOWN") is False
    assert A.may_serve_synchronized("UNVERIFIED", "UNKNOWN") is False


def test_pairing_infrastructure_is_unchanged(reference):
    pairing = pair_cues(reference, reference, tolerance_ms=MOVEMENT_TOLERANCE_MS)
    assert pairing.matched == len(reference)
    assert pairing.unmatched_before == []


def test_seed_offset_is_reported_in_real_time(reference):
    """After seeding, reported movement is the actual correction, not the
    residual around the seed."""
    target = shift(reference, 96_500)
    seed = measure(target, reference, gate.LARGE_OFFSET_MAX_MS)
    assert seed is not None
    evaluation = analyze(target, reference, reference, gate.LARGE_OFFSET_MAX_MS)
    assert evaluation.median_offset_ms == pytest.approx(0.0, abs=1.0)
    assert evaluation.mad_offset_ms == pytest.approx(0.0, abs=1.0)
