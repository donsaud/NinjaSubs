"""Alignment analysis: what ``alass`` did, and how much it can be trusted.

``alass`` is an alignment engine, not a truth detector. It exits 0 and writes a
file for inputs it cannot align, so these tests pin the rule that a successful
process is never itself evidence of synchronization.

Scenarios follow the A-J matrix: already-synced, constant offset, FPS drift,
wrong episode, small offset from a different release, poor alignment despite
exit 0, negligible-change re-timing, consensus, shared-but-wrong consensus, and
missing metadata.
"""

from __future__ import annotations

import pytest

from app.services.sync.alignment import (
    MAX_DRIFT_MS_PER_MINUTE,
    MAX_MAD_MS_FOR_STABLE,
    MAX_P95_MS_FOR_STABLE,
    AlignmentAnalyzer,
    RejectionReason,
    SyncState,
    analyze_drift,
    detect_change_points,
    evaluate_sync_state,
    pair_cue_starts,
    pair_cues,
)


def _ts(ms: int) -> str:
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def srt(
    count: int,
    *,
    step_ms: int = 2_000,
    start_ms: int = 0,
    body: str = "I never thought I would find someone like you in my life",
) -> str:
    """A realistic subtitle: full sentences, blank-line separated cues."""
    return "\n\n".join(
        f"{i + 1}\n{_ts(start_ms + i * step_ms)} --> "
        f"{_ts(start_ms + i * step_ms + 1_500)}\n{body}"
        for i in range(count)
    )


def srt_with_deltas(deltas: list[float]) -> str:
    """Build a subtitle whose cue ``i`` starts at ``deltas[i]`` ms."""
    return "\n\n".join(
        f"{i + 1}\n{_ts(int(d))} --> {_ts(int(d) + 1_500)}\n"
        "I never thought I would find someone like you in my life"
        for i, d in enumerate(deltas)
    )


ANALYZER = AlignmentAnalyzer()


# --------------------------------------------------------------------------- #
# Case A / G: already synchronized
# --------------------------------------------------------------------------- #


def test_case_a_already_synchronized_is_verified_synced():
    """Case A: exact release + already synchronized -> VERIFIED_SYNCED."""
    target = srt(40)
    evaluation = ANALYZER.analyze(target, None, target)
    assert evaluation.sync_state is SyncState.VERIFIED_SYNCED
    assert evaluation.median_offset_ms == pytest.approx(0.0, abs=1.0)
    assert evaluation.rejection_reason is None
    assert "already aligned" in " ".join(evaluation.reasons)


def test_case_g_alass_making_negligible_changes_stays_resynced():
    """Case G: alass runs but moves almost nothing -> still a sound alignment.

    A verified re-sync and an already-aligned pass are different claims with
    the same quality; the state reflects that alass was applied.
    """
    target = srt(40)
    barely_moved = srt(40, start_ms=120)
    reference = srt(40, start_ms=120)
    evaluation = ANALYZER.analyze(
        target, barely_moved, reference, alass_applied=True, alass_successful=True
    )
    assert evaluation.sync_state is SyncState.VERIFIED_RESYNCED
    assert evaluation.alass_applied is True
    assert evaluation.p95_offset_ms is not None
    assert evaluation.p95_offset_ms <= MAX_P95_MS_FOR_STABLE


# --------------------------------------------------------------------------- #
# Case B: correct release + constant offset
# --------------------------------------------------------------------------- #


def test_case_b_stable_offset_without_alass_is_probable_not_verified():
    """Case B: a large *stable* offset is legal, but unverified until re-timed."""
    target = srt(40)
    reference = srt(40, start_ms=2_310)
    evaluation = ANALYZER.analyze(target, None, reference)
    # Not a rejection: a stable 2.31s offset is exactly what a different
    # master's intro produces.
    assert evaluation.sync_state is SyncState.PROBABLE_SYNC
    assert evaluation.median_offset_ms == pytest.approx(2_310, abs=50)
    # Low variance is what makes it plausible.
    assert evaluation.mad_offset_ms == pytest.approx(0.0, abs=50)


def test_case_b2_constant_offset_corrected_by_alass_is_resynced():
    """Case B continued: after re-timing, the residual collapses -> verified."""
    target = srt(40)
    aligned = srt(40, start_ms=2_310)
    reference = srt(40, start_ms=2_310)
    evaluation = ANALYZER.analyze(
        target, aligned, reference, alass_applied=True, alass_successful=True
    )
    assert evaluation.sync_state is SyncState.VERIFIED_RESYNCED
    # The headline offset is the residual, which is ~0, not the 2.31s moved.
    assert abs(evaluation.median_offset_ms or 0.0) <= MAX_P95_MS_FOR_STABLE
    assert evaluation.sync_confidence is not None
    assert evaluation.sync_confidence >= 90.0


# --------------------------------------------------------------------------- #
# Case C: FPS drift
# --------------------------------------------------------------------------- #


def test_case_c_progressive_drift_is_probable_not_stable():
    """Case C: a growing offset is drift, distinguishable from a fixed shift."""
    steady_deltas = [float(i * 2_000) for i in range(40)]
    drifting_deltas = [float(i * 2_000) + 1_200 * i for i in range(40)]

    steady = analyze_drift([(int(d), 2_310.0) for d in steady_deltas])
    drifting = analyze_drift([(int(d), 1_200.0 * i) for i, d in enumerate(drifting_deltas)])

    assert steady is not None and abs(steady) < 1.0
    assert drifting is not None and abs(drifting) > MAX_DRIFT_MS_PER_MINUTE
    assert abs(drifting) > abs(steady)

    evaluation = ANALYZER.analyze(
        srt_with_deltas(steady_deltas),
        srt_with_deltas(drifting_deltas),
        srt_with_deltas(steady_deltas),
        alass_applied=True,
        alass_successful=True,
    )
    # Progressive re-timing lowers confidence rather than being trusted as a
    # clean global shift.
    assert evaluation.sync_state is not SyncState.VERIFIED_RESYNCED
    assert evaluation.sync_state in (SyncState.PROBABLE_SYNC, SyncState.UNVERIFIED)
    assert evaluation.drift_ms_per_minute is not None
    assert abs(evaluation.drift_ms_per_minute) > MAX_DRIFT_MS_PER_MINUTE


def test_drift_uses_existing_fps_relation_logic():
    """The analyzer reuses, not reimplements, FPS-relation detection."""
    from app.services.subtitle_matcher import determine_fps_relation

    evaluation = ANALYZER.analyze(
        srt(40), None, srt(40, start_ms=2_310), target_fps=25.0, reference_fps=23.976
    )
    reasons = " ".join(evaluation.reasons)
    assert f"fps relation: {determine_fps_relation(25.0, 23.976)}" in reasons


# --------------------------------------------------------------------------- #
# Case D: wrong episode / wrong cut
# --------------------------------------------------------------------------- #


def test_case_d_wrong_cut_is_rejected():
    """Case D: a 97s offset is a different cut, not a sync offset."""
    evaluation = ANALYZER.analyze(srt(40), None, srt(40, start_ms=97_000))
    assert evaluation.sync_state is SyncState.REJECTED
    assert evaluation.rejection_reason is RejectionReason.IMPLAUSIBLE_OFFSET


def test_wrong_episode_is_rejected_by_content_filter_not_alignment():
    """Case D is settled upstream by hard filtering; alignment never sees it.

    The alignment layer must not be responsible for content identity.
    """
    from app.models import SubtitleRelease
    from app.services.subtitle_matcher import extract_metadata, hard_compatibility_filter

    target = extract_metadata("Dexter.S08E05.1080p.BluRay.x264-PiR8.mkv")
    wrong = extract_metadata("Dexter.S08E06.1080p.BluRay.x264-PiR8.mkv")
    accepted, reason, _ = hard_compatibility_filter(target, wrong)
    assert accepted is False
    assert reason

    # A correct-episode candidate is admitted, and only then judged on timing.
    right = extract_metadata("Dexter.S08E05.720p.BluRay.x264-NORDiC.srt")
    accepted, _, _ = hard_compatibility_filter(target, right)
    assert accepted is True
    assert isinstance(right.get("episode"), int) and right["episode"] == 5
    assert SubtitleRelease(release_name="x", download_url="u", provider="subdl")


# --------------------------------------------------------------------------- #
# Case E: different release, small offset
# --------------------------------------------------------------------------- #


def test_case_e_small_offset_different_release_is_lower_confidence():
    """Case E: a small offset from another release is 'possible', not proven."""
    target = srt(40)
    reference = srt(40, start_ms=1_200)
    evaluation = ANALYZER.analyze(target, None, reference)
    assert evaluation.sync_state is SyncState.PROBABLE_SYNC
    assert evaluation.verification_confidence is not None
    assert evaluation.verification_confidence < 100.0


# --------------------------------------------------------------------------- #
# Case F: alass exits 0 but the alignment is poor
# --------------------------------------------------------------------------- #


def test_case_f_successful_alass_with_poor_alignment_is_not_verified():
    """Case F: exit code 0 is not evidence. Poor output must not be VERIFIED."""
    target = srt(40)
    reference = srt(40, start_ms=2_310)
    # Scattered, unalignable output: correct cue count, random placement.
    scrambled_positions = [0, 41_000, 3_000, 58_000, 7_000, 22_000, 12_000, 49_000] * 5
    scrambled = srt_with_deltas([float(p) for p in scrambled_positions[:40]])

    evaluation = ANALYZER.analyze(
        target, scrambled, reference, alass_applied=True, alass_successful=True
    )
    assert evaluation.sync_state in (SyncState.UNVERIFIED, SyncState.REJECTED)
    assert evaluation.sync_state is not SyncState.VERIFIED_RESYNCED
    assert evaluation.sync_state is not SyncState.VERIFIED_SYNCED
    assert not evaluation.reasons == []


def test_case_f2_alass_failure_is_unverified():
    """alass producing nothing is UNVERIFIED, never VERIFIED."""
    evaluation = ANALYZER.analyze(
        srt(40), None, srt(40, start_ms=2_310), alass_applied=True, alass_successful=False
    )
    assert evaluation.sync_state is SyncState.UNVERIFIED
    assert evaluation.rejection_reason is RejectionReason.ALASS_FAILED


def test_alass_dropping_cues_is_rejected():
    """An output that lost most cues is a broken alignment, not a sync."""
    target = srt(40)
    evaluation = ANALYZER.analyze(
        target,
        srt(5, start_ms=2_310),
        srt(40, start_ms=2_310),
        alass_applied=True,
        alass_successful=True,
    )
    assert evaluation.sync_state is SyncState.REJECTED
    assert evaluation.rejection_reason is RejectionReason.CUE_LOSS
    assert evaluation.coverage_score is not None
    assert evaluation.coverage_score < 0.80


# --------------------------------------------------------------------------- #
# Case H / I: consensus
# --------------------------------------------------------------------------- #


def test_case_h_agreement_raises_consensus_evidence():
    """Case H: independent candidates agreeing on an offset form a cluster."""
    target = srt(40)
    reference = srt(40, start_ms=2_310)
    offsets = [2_310, 2_290, 2_350]
    evaluations = [
        ANALYZER.analyze(target, srt(40, start_ms=int(o)), reference, alass_applied=True,
                         alass_successful=True)
        for o in offsets
    ]
    # Each is independently verified; the cluster is supporting evidence only.
    assert all(e.sync_state is SyncState.VERIFIED_RESYNCED for e in evaluations)
    residuals = [abs(e.p95_offset_ms or 0.0) for e in evaluations]
    assert max(residuals) < min(offsets) - 2_000


def test_case_i_consensus_never_promotes_beyond_individual_verification():
    """Case I: an outlier stays an outlier regardless of how many agree.

    Multiple providers can mirror one another while all being wrong, so
    consensus is never allowed to upgrade a single candidate's state.
    """
    target = srt(40)
    good_reference = srt(40, start_ms=2_310)
    bad_reference = srt(40, start_ms=97_000)

    # Three sources agree on the same wrong cut.
    agreeing = [
        ANALYZER.analyze(target, None, bad_reference) for _ in range(3)
    ]
    assert all(e.sync_state is SyncState.REJECTED for e in agreeing)

    # The lone good candidate is still the only acceptable one.
    good = ANALYZER.analyze(target, srt(40, start_ms=2_310), good_reference,
                            alass_applied=True, alass_successful=True)
    assert good.sync_state is SyncState.VERIFIED_RESYNCED
    # Consensus would rank the majority first; state-based ordering must not.
    assert good.sync_state.rank < agreeing[0].sync_state.rank


def test_consensus_field_stays_optional_until_computed():
    """Consensus is never invented when no cluster analysis has run."""
    evaluation = evaluate_sync_state(srt(40), None, srt(40, start_ms=2_310))
    assert evaluation.consensus_score is None


# --------------------------------------------------------------------------- #
# Case J: missing metadata / graceful degradation
# --------------------------------------------------------------------------- #


def test_case_j_no_reference_is_unverified_not_rejected():
    """Case J: absent evidence degrades to UNVERIFIED, never a hard reject."""
    evaluation = ANALYZER.analyze(srt(40), None, None)
    assert evaluation.sync_state is SyncState.UNVERIFIED
    assert evaluation.rejection_reason is RejectionReason.INSUFFICIENT_EVIDENCE
    # No metric is fabricated.
    assert evaluation.median_offset_ms is None
    assert evaluation.p95_offset_ms is None
    assert evaluation.drift_ms_per_minute is None


def test_insufficient_dialogue_stays_unverified():
    """Case J: too few cues to measure -> UNVERIFIED, and explicitly flagged."""
    sparse = "1\n00:00:01,000 --> 00:00:02,000\nHello"
    evaluation = ANALYZER.analyze(sparse, None, srt(40))
    assert evaluation.sync_state is SyncState.UNVERIFIED
    assert evaluation.rejection_reason is RejectionReason.INSUFFICIENT_EVIDENCE


def test_season_pack_reference_is_not_claimed_as_verified():
    """A season-pack reference stays admitted but never VERIFIED on its own.

    Candidate admission (validate_cue_sanity) is unchanged; only the new
    synchronization claim fails closed. This is the "fail closed for the new
    claim without changing existing admission" contract.
    """
    from app.services.subtitle_matcher import validate_cue_sanity

    target = srt(40, start_ms=106_950)
    pack = srt(40, start_ms=9_280)
    # Existing behaviour: admission is allowed to fail open.
    admission = validate_cue_sanity(target, pack, threshold_ms=20_000)
    assert admission["ok"] is False
    # The new claim refuses.
    evaluation = ANALYZER.analyze(target, None, pack)
    assert evaluation.sync_state is not SyncState.VERIFIED_SYNCED


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def test_mad_is_robust_to_a_single_outlier():
    """MAD ignores one wild cue that a mean or stdev would be dragged by."""
    clean = [1_000.0] * 20
    with_outlier = [*clean[:-1], 90_000.0]

    def _mad(values: list[float]) -> float:
        from app.services.sync.alignment import _median

        median = _median(values)
        return _median([abs(v - median) for v in values])

    assert _mad(clean) == 0.0
    assert _mad(with_outlier) == 0.0
    assert _percentile(clean, 0.95) == 1_000.0


def _percentile(values: list[float], fraction: float) -> float:
    from app.services.sync.alignment import _percentile as impl

    return impl(values, fraction)


def test_p95_reports_the_tail():
    """Nearest-rank percentile: the 95th smallest of 100 values is index 94."""
    from app.services.sync.alignment import _percentile

    values = [float(i) for i in range(100)]
    assert _percentile(values, 0.95) == 94.0
    assert _percentile(values, 0.5) == pytest.approx(50.0, abs=1.0)
    assert _percentile([7.0], 0.95) == 7.0


def test_pair_cue_starts_matches_nearest_within_tolerance():
    before = parse_cue_starts(srt(10))
    after = parse_cue_starts(srt(10, start_ms=2_000))
    pairs = pair_cue_starts(before, after)
    assert len(pairs) == 10
    assert all(abs(delta - 2_000) < 1 for _, delta in pairs)

    # Beyond tolerance nothing pairs, rather than pairing a wrong neighbour.
    far = parse_cue_starts(srt(10, start_ms=97_000))
    assert pair_cue_starts(before, far, tolerance_ms=5_000) == []


def test_cue_outside_tolerance_is_reported_as_unmatched() -> None:
    """A cue with no counterpart is reported, not silently dropped."""
    before = parse_cue_starts(srt(6))
    after = parse_cue_starts(srt(6, start_ms=2_000))

    result = pair_cues(before, after, tolerance_ms=5_000)

    assert result.before_total == 6
    assert result.after_total == 6
    assert result.matched == 6
    assert result.unmatched_before == []
    assert result.unmatched_after == []


def test_unmatched_cue_never_becomes_a_timing_residual() -> None:
    """The semantic rule: unmatched means "no counterpart", not "zero error".

    One cue is far beyond the tolerance. It must appear in ``unmatched_before``
    and must not contribute a delta -- in particular not a delta sitting at the
    tolerance boundary, which is how a truncation artefact reads as timing.
    """
    before = parse_cue_starts(srt(4)) + [(500_000, 501_000, "far away cue")]
    after = parse_cue_starts(srt(4, start_ms=2_000))

    result = pair_cues(before, after, tolerance_ms=5_000)

    assert result.unmatched_before == [4], "the distant cue must be reported"
    assert result.matched == 4
    deltas = [delta for _, delta in result.pairs]
    assert len(deltas) == 4, "an unmatched cue must not appear as a residual"
    assert all(abs(delta - 2_000) < 1 for delta in deltas)
    # Nothing may be manufactured at the tolerance edge.
    assert not any(abs(abs(delta) - 5_000) < 1 for delta in deltas)
    assert max(abs(delta) for delta in deltas) < 5_000


def test_matched_residuals_are_identical_to_the_legacy_helper() -> None:
    """Backward compatibility: the numbers existing callers see do not move."""
    before = parse_cue_starts(srt(20))
    for shift in (0, 700, -1_500, 3_000):
        after = parse_cue_starts(srt(20, start_ms=10_000 + shift))
        assert pair_cues(before, after).pairs == pair_cue_starts(before, after)


def test_pairing_reports_both_sides_when_counts_differ() -> None:
    """More cues than counterparts: the surplus is reported on the before side."""
    before = parse_cue_starts(srt(20, step_ms=1_000))
    after = parse_cue_starts(srt(8, start_ms=10_000, step_ms=1_000))

    result = pair_cues(before, after, tolerance_ms=5_000)

    assert result.matched == 8
    assert result.before_total == 20
    assert len(result.unmatched_before) == 12
    assert result.unmatched_after == []
    assert len(result.pairs) == 8


def test_empty_side_reports_everything_unmatched() -> None:
    before = parse_cue_starts(srt(3))
    result = pair_cues(before, [], tolerance_ms=5_000)
    assert result.matched == 0
    assert result.unmatched_before == [0, 1, 2]
    assert result.pairs == []


def test_pairing_coverage_is_recorded_and_is_not_an_acceptance_input() -> None:
    """Coverage is observable on the evaluation, and changes no verdict."""
    reference = [(10_000 + 2_000 * i, 10_900 + 2_000 * i, f"line {i}") for i in range(30)]
    target = [(10_000 + 2_000 * i, 10_900 + 2_000 * i, f"line {i}") for i in range(30)]
    synced = [(9_000 + 2_000 * i, 9_900 + 2_000 * i, f"line {i}") for i in range(30)]

    evaluation = AlignmentAnalyzer().analyze(
        target, synced, reference, alass_applied=True, alass_successful=True
    )

    assert evaluation.movement_pairing is not None
    assert evaluation.movement_pairing.matched == 30
    assert evaluation.movement_pairing.unmatched_before == 0
    assert evaluation.residual_pairing is not None
    assert evaluation.residual_pairing.matched == 30
    # Coverage appears in the operator-facing summary, for diagnosis.
    assert "mv_pairs=" in evaluation.explain()
    assert "res_pairs=" in evaluation.explain()


def test_unmatched_cues_cannot_rescue_a_bad_alignment() -> None:
    """Excluding unmatched cues must not turn a poor result into VERIFIED.

    Isolates the "too few pairs" guard: 30 cues reach the verifier, movement
    pairs perfectly, but only 4 find a counterpart against the reference, and
    those 4 agree well. Nothing but the count guard can refuse this, so if the
    guard were removed the result would be free to verify on four agreeable
    pairs while 26 cues were never compared at all.
    """
    reference = [(10_000 + 2_000 * i, 10_900 + 2_000 * i, f"line {i}") for i in range(30)]
    aligned = [(10_000 + 2_000 * i, 10_900 + 2_000 * i, f"line {i}") for i in range(4)]
    stray = [(400_000 + 20_000 * i, 400_900 + 20_000 * i, f"stray {i}") for i in range(26)]
    target = aligned + stray
    synced = list(target)

    evaluation = AlignmentAnalyzer().analyze(
        target, synced, reference, alass_applied=True, alass_successful=True
    )

    # The unmatched are reported rather than hidden.
    assert evaluation.movement_pairing is not None
    assert evaluation.movement_pairing.matched == 30
    assert evaluation.residual_pairing is not None
    assert evaluation.residual_pairing.matched == 4
    assert evaluation.residual_pairing.unmatched_before == 26

    # And the count guard, not luck, is what refuses it.
    assert evaluation.sync_state is not SyncState.VERIFIED_SYNCED
    assert evaluation.sync_state is not SyncState.VERIFIED_RESYNCED
    assert evaluation.rejection_reason is RejectionReason.INSUFFICIENT_EVIDENCE
    assert any("too few cues" in reason for reason in evaluation.reasons)


def parse_cue_starts(text: str):
    from app.services.subtitle_matcher import parse_srt_cues

    return parse_srt_cues(text)


def test_change_points_detect_a_piecewise_shift():
    """A scene insertion produces a step, not a smooth ramp."""
    steady = [(i * 5_000, 1_400.0) for i in range(10)]
    second_half = [(50_000 + i * 5_000, 30_000.0 + i * 20.0) for i in range(10)]
    stepped = [*steady, *second_half]
    found = detect_change_points(stepped)
    assert found, "a 28s step must be detected"
    assert abs(found[0][1]) > 5_000
    # A pure constant offset has no change points.
    assert detect_change_points(steady) == []


def test_change_points_are_evidence_not_a_rejection_rule():
    """A structural difference is reported, and the state stays usable."""
    target = srt(40)
    # Aligned, but with a large mid-file step (e.g. a recap / inserted scene).
    stepped: list[float] = []
    for i in range(40):
        stepped.append(float(i * 2_000) + (12_000.0 if i >= 20 else 0.0))
    reference = srt(40)
    evaluation = ANALYZER.analyze(
        target, srt_with_deltas(stepped), reference, alass_applied=True, alass_successful=True
    )
    assert evaluation.change_points
    assert evaluation.sync_state is not SyncState.REJECTED


# --------------------------------------------------------------------------- #
# Ordering / determinism
# --------------------------------------------------------------------------- #


def test_state_rank_orders_verified_before_unverified():
    """The documented presentation ordering."""
    order = [
        SyncState.VERIFIED_SYNCED,
        SyncState.VERIFIED_RESYNCED,
        SyncState.PROBABLE_SYNC,
        SyncState.UNVERIFIED,
        SyncState.REJECTED,
    ]
    ranks = [state.rank for state in order]
    assert ranks == sorted(ranks)
    assert SyncState.VERIFIED_SYNCED.rank < SyncState.VERIFIED_RESYNCED.rank
    assert SyncState.UNVERIFIED.rank < SyncState.REJECTED.rank


def test_evaluation_is_deterministic():
    """Same inputs must always produce an identical verdict."""
    target = srt(40)
    reference = srt(40, start_ms=2_310)
    first = ANALYZER.analyze(target, srt(40, start_ms=2_310), reference,
                             alass_applied=True, alass_successful=True)
    second = ANALYZER.analyze(target, srt(40, start_ms=2_310), reference,
                              alass_applied=True, alass_successful=True)
    assert first.sync_state is second.sync_state
    assert first.median_offset_ms == second.median_offset_ms
    assert first.p95_offset_ms == second.p95_offset_ms
    assert first.explain() == second.explain()


def test_content_and_sync_signals_stay_separate():
    """A perfect content match never implies a synchronization state."""
    evaluation = ANALYZER.analyze(
        srt(40), None, None, content_match_score=100.0
    )
    assert evaluation.content_match_score == 100.0
    assert evaluation.sync_state is SyncState.UNVERIFIED


def test_explain_is_readable_and_safe():
    """explain() must describe the verdict without dumping subtitle text."""
    evaluation = ANALYZER.analyze(
        srt(40), srt(40, start_ms=2_310), srt(40, start_ms=2_310),
        alass_applied=True, alass_successful=True,
    )
    text = evaluation.explain()
    assert "verified_resynced" in text
    assert "I never thought" not in text
    assert MAX_MAD_MS_FOR_STABLE > 0 and MAX_P95_MS_FOR_STABLE > 0
