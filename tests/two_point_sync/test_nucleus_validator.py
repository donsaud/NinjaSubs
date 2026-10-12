"""Tests for two_point_sync.validator."""

from __future__ import annotations

from tests.two_point_sync._helpers import apply_affine, build_dialogue
from two_point_sync.anchors import detect_anchors
from two_point_sync.parser import Cue
from two_point_sync.transform import LinearTransform
from two_point_sync.validator import Checkpoint, validate_transform


def _checkpoints(target_ms: list[int], reference_ms: list[int]) -> list[Checkpoint]:
    return [
        Checkpoint(target_number=position + 1, reference_number=position + 1,
                   target_ms=t, reference_ms=r)
        for position, (t, r) in enumerate(zip(target_ms, reference_ms, strict=False))
    ]


def test_accepts_accurate_transform_with_spread_checkpoints() -> None:
    transform = LinearTransform(scale=1.0, offset_ms=5_000.0)
    checkpoints = _checkpoints(
        [60_000, 600_000, 1_200_000, 1_800_000, 2_400_000, 3_000_000],
        [65_000, 605_000, 1_205_000, 1_805_000, 2_405_000, 3_005_000],
    )
    report = validate_transform(transform, checkpoints)
    assert report.accepted
    assert report.n_checkpoints == 6
    assert report.median_abs_ms == 0.0
    assert set(report.region_counts) == {"beginning", "middle", "ending"}


def test_rejects_transform_with_bad_middle_but_good_endpoints() -> None:
    # Endpoints fit; the middle carries an abrupt +30s discontinuity.
    transform = LinearTransform(scale=1.0, offset_ms=0.0)
    checkpoints = _checkpoints(
        [100_000, 200_000, 500_000, 600_000, 900_000, 1_000_000],
        [100_000, 200_000, 530_000, 630_000, 900_000, 1_000_000],
    )
    report = validate_transform(transform, checkpoints)
    assert not report.accepted
    assert report.has_discontinuity
    assert "nonlinear_discontinuity_in_middle" in report.reasons


def test_rejects_clustered_checkpoints_without_coverage() -> None:
    transform = LinearTransform(scale=1.0, offset_ms=0.0)
    checkpoints = _checkpoints(
        [100_000, 110_000, 120_000, 130_000],
        [100_000, 110_000, 120_000, 130_000],
    )
    report = validate_transform(transform, checkpoints, span_start_ms=0, span_end_ms=3_600_000)
    assert not report.accepted
    assert "missing_region_coverage" in report.reasons


def test_rejects_insufficient_checkpoints() -> None:
    report = validate_transform(
        LinearTransform(scale=1.0, offset_ms=0.0),
        _checkpoints([100_000], [100_000]),
    )
    assert not report.accepted
    assert "insufficient_checkpoints" in report.reasons


def test_rejects_large_overall_residual() -> None:
    transform = LinearTransform(scale=1.0, offset_ms=60_000.0)
    checkpoints = _checkpoints(
        [100_000, 500_000, 900_000, 1_300_000, 1_700_000, 2_100_000],
        [100_000, 500_000, 900_000, 1_300_000, 1_700_000, 2_100_000],
    )
    report = validate_transform(transform, checkpoints)
    assert not report.accepted
    assert "median_residual_exceeds_tolerance" in report.reasons


def test_contradictory_region_vetoes_acceptance() -> None:
    transform = LinearTransform(scale=1.0, offset_ms=0.0)
    checkpoints = _checkpoints(
        [100_000, 500_000, 900_000, 1_300_000, 1_700_000, 2_100_000],
        [100_000, 500_000, 900_000, 1_300_000, 1_700_000, 2_100_000],
    )
    assert validate_transform(transform, checkpoints).accepted
    # A coherent contradictory region (all ~+60s off) vetoes the claim.
    contradictory = _checkpoints([200_000, 400_000, 600_000], [260_000, 460_000, 660_000])
    report = validate_transform(transform, checkpoints, contradictory=contradictory)
    assert not report.accepted
    assert report.has_discontinuity
    assert "contradictory_region_rejected" in report.reasons


def test_single_contradictory_point_is_not_a_veto() -> None:
    transform = LinearTransform(scale=1.0, offset_ms=0.0)
    checkpoints = _checkpoints(
        [100_000, 500_000, 900_000, 1_300_000, 1_700_000, 2_100_000],
        [100_000, 500_000, 900_000, 1_300_000, 1_700_000, 2_100_000],
    )
    report = validate_transform(
        transform, checkpoints, contradictory=_checkpoints([200_000], [260_000])
    )
    assert report.accepted


def test_nonlinear_cut_pipeline_reports_discontinuity() -> None:
    """A global Two-Point model must not validate across an abrupt cut."""
    target = build_dialogue(12, step_ms=60_000)
    reference = apply_affine(target, 1.0, 5_000.0)
    # Introduce a +45s discontinuity across the middle third.
    shifted = [
        Cue(number=cue.number, order=cue.order,
            start_ms=cue.start_ms + (45_000 if 4 <= position < 8 else 0),
            end_ms=cue.end_ms + (45_000 if 4 <= position < 8 else 0),
            text=cue.text)
        for position, cue in enumerate(reference)
    ]
    result = detect_anchors(target, shifted)
    assert not result.accepted
    assert "NONLINEAR_DISCONTINUITY" in result.reasons
    assert len(result.regimes) == 2
    primary, alt = result.regimes
    assert abs(alt.implied_offset_ms - primary.offset_ms) > 30_000
    assert len(result.contradictory) >= 3
    # The cut evacuated the middle third, so the primary-regime checkpoints
    # cannot cover the span on their own either.
    model = LinearTransform(scale=primary.scale, offset_ms=primary.offset_ms)
    checkpoints = [
        Checkpoint(target_number=item.target_number, reference_number=item.reference_number,
                   target_ms=item.target_ms, reference_ms=item.reference_ms)
        for item in result.checkpoints
    ]
    report = validate_transform(model, checkpoints)
    assert not report.accepted
    assert "missing_region_coverage" in report.reasons
