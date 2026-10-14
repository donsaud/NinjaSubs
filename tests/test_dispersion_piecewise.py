"""Dispersion semantics and piecewise timing stability (Cases A-E + step/drift).

P95/MAD measure dispersion (stability around the shift); the median measures
the shift itself (magnitude). A large constant offset must never read as
unstable, a genuine step between two stable regimes must never read as
progressive drift, and alternating noise must never be disguised as
micro-segments. Timing-v3 remains the authoritative verifier; these tests pin
the measurement layer only and add no acceptance gate.
"""

from __future__ import annotations

import pytest

from app.services.sync import alignment as A


def _pairs(offsets: list[float], start_ms: int = 60_000, step_ms: int = 60_000):
    return [(start_ms + index * step_ms, delta) for index, delta in enumerate(offsets)]


def _srt(n: int, start_ms: int = 60_000, step_ms: int = 60_000, shift_ms: int = 0) -> str:
    def fmt(total: int) -> str:
        total = max(0, total)
        return (f"{total // 3_600_000:02d}:{(total // 60_000) % 60:02d}:"
                f"{(total // 1_000) % 60:02d},{total % 1_000:03d}")

    blocks = []
    for i in range(n):
        start = start_ms + i * step_ms + shift_ms
        blocks.append(f"{i + 1}\n{fmt(start)} --> {fmt(start + 2_000)}\n"
                      f"Dialogue event number {i + 1} spoken aloud")
    return "\n\n".join(blocks) + "\n"


# --------------------------------------------------------------------------- #
# Case A -- constant offset: large median, small dispersion
# --------------------------------------------------------------------------- #

def test_measure_constant_offset_reports_shift_not_instability():
    evaluation = A.SubtitleEvaluation()
    A.AlignmentAnalyzer()._measure(
        evaluation, _pairs([6_000 + ((i * 37) % 61) - 30 for i in range(40)]))
    assert evaluation.median_offset_ms == pytest.approx(6_000, abs=50)
    assert evaluation.mad_offset_ms is not None and evaluation.mad_offset_ms < 100
    # The corrected behaviour: residual dispersion, not shift magnitude.
    assert evaluation.p95_offset_ms is not None and evaluation.p95_offset_ms < 100


def test_stable_offset_above_p95_bound_is_probable_never_rejected():
    target = _srt(40)
    reference = _srt(40, shift_ms=4_000)
    evaluation = A.AlignmentAnalyzer().analyze(target, None, reference)
    assert evaluation.sync_state is A.SyncState.PROBABLE_SYNC
    assert evaluation.median_offset_ms == pytest.approx(4_000, abs=50)
    assert evaluation.mad_offset_ms == pytest.approx(0.0, abs=50)
    assert evaluation.p95_offset_ms is not None
    assert evaluation.p95_offset_ms < 100


# --------------------------------------------------------------------------- #
# Case B -- legitimate two-segment offset
# --------------------------------------------------------------------------- #

def _two_segment() -> list[tuple[int, float]]:
    first = [1_000 + ((i * 53) % 41) - 20 for i in range(30)]
    second = [15_000 + ((i * 47) % 41) - 20 for i in range(30)]
    return _pairs(first + second)


def test_two_stable_segments_are_segmented_with_low_local_dispersion():
    segments = A.segment_pairs(_two_segment())
    assert len(segments) == 2
    assert [len(run) for run in segments] == [30, 30]
    for run in segments:
        deltas = [delta for _, delta in run]
        local_median = A._median(deltas)
        residuals = [abs(delta - local_median) for delta in deltas]
        assert A._median(residuals) < 100
        assert A._percentile(residuals, 0.95) < 100


def test_step_change_is_detected_not_smoothed():
    # The running-median detector reports a ramp of large steps straddling
    # the transition (pre-existing behaviour); what matters is that the step
    # is detected at the true boundary and segment_pairs resolves one split.
    points = A.detect_change_points(_two_segment())
    assert points
    assert all(abs(step) >= A.CHANGE_POINT_MS for _, step in points)
    positions = [position for position, _ in points]
    assert min(positions) <= 1_860_000 <= max(positions)
    assert len(A.segment_pairs(_two_segment())) == 2


# --------------------------------------------------------------------------- #
# Case C -- noisy alternating offsets must not become micro-segments
# --------------------------------------------------------------------------- #

def test_alternating_offsets_refuse_segmentation_and_read_unstable():
    alternating = [(4_000 if i % 2 else 0) + (i % 3) - 1 for i in range(60)]
    pairs = _pairs(alternating, step_ms=30_000)
    segments = A.segment_pairs(pairs)
    assert len(segments) == 1
    deltas = [delta for _, delta in pairs]
    median = A._median(deltas)
    residuals = [abs(delta - median) for delta in deltas]
    assert A._median(residuals) > 1_000
    assert A._percentile(residuals, 0.95) > 1_000


def test_fragmented_blocks_refuse_segmentation():
    fragmented: list[float] = []
    for block in range(15):
        fragmented.extend([6_000 if block % 2 else 0] * 4)
    segments = A.segment_pairs(_pairs(fragmented, step_ms=30_000))
    assert len(segments) == 1


# --------------------------------------------------------------------------- #
# Case D -- actual linear drift stays detectable; step is not drift
# --------------------------------------------------------------------------- #

def test_genuine_drift_detected_without_change_points():
    pairs = _pairs([float(i * 100) for i in range(60)], step_ms=60_000)
    assert A.segment_pairs(pairs) == [sorted(pairs, key=lambda p: p[0])]
    assert A.detect_change_points(pairs) == []
    assert A.analyze_drift(pairs) == pytest.approx(100.0, abs=5.0)


def test_step_change_is_not_classified_as_progressive_drift():
    pairs = _two_segment()
    drift = A.analyze_drift(pairs)
    assert drift is not None and abs(drift) < 5.0
    # The test is only meaningful if a naive global fit would exceed the
    # progressive-drift ceiling used by the legacy verdict.
    xs = [position / 60_000.0 for position, _ in pairs]
    ys = [delta for _, delta in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    naive = (sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=False))
             / sum((x - mean_x) ** 2 for x in xs))
    assert abs(naive) > A.MAX_DRIFT_MS_PER_MINUTE


def test_drift_within_a_supported_segment_still_measured():
    first = [1_000 + i * 50 for i in range(30)]
    second = [15_000 + ((i * 47) % 41) - 20 for i in range(30)]
    drift = A.analyze_drift(_pairs(first + second))
    assert drift == pytest.approx(50.0, abs=5.0)


# --------------------------------------------------------------------------- #
# Outlier / edge guards
# --------------------------------------------------------------------------- #

def test_single_outlier_cannot_form_a_perfect_segment():
    deltas = [1_000.0] * 59 + [15_000.0]
    segments = A.segment_pairs(_pairs(deltas))
    assert len(segments) == 1


def test_too_few_pairs_measure_globally():
    pairs = _pairs([1_000.0] * 6 + [15_000.0] * 6)
    assert A.segment_pairs(pairs) == [sorted(pairs, key=lambda p: p[0])]


def test_chronological_order_preserved_regardless_of_input_order():
    pairs = _two_segment()
    shuffled = list(reversed(pairs))
    segments = A.segment_pairs(shuffled)
    assert len(segments) == 2
    for run in segments:
        positions = [position for position, _ in run]
        assert positions == sorted(positions)


# --------------------------------------------------------------------------- #
# Case E -- low residual alone never overrides missing correspondence
# --------------------------------------------------------------------------- #

def test_low_residual_without_correspondence_is_not_verified():
    from app.services.sync.unified_verification import verify_candidate

    target = _srt(12)
    unrelated = "\n\n".join(
        f"{i + 1}\n00:{i:02d}:00,000 --> 00:{i:02d}:02,000\n"
        f"Unrelated episode line {i + 1} entirely"
        for i in range(12)) + "\n"
    verdict = verify_candidate(target, unrelated, target, candidate_algorithm="two_point")
    assert verdict.proposed_verdict != "VERIFIED"
    assert any("insufficient_independent_groups" in reason
               for reason in verdict.failure_reasons)
