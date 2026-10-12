"""Tests for two_point_sync.transform."""

from __future__ import annotations

import pytest

from tests.two_point_sync._helpers import build_dialogue
from two_point_sync.transform import (
    LinearTransform,
    TransformError,
    apply_transform,
    fit_from_anchors,
    fit_least_squares,
)


def test_fit_uses_exact_equations() -> None:
    transform = fit_from_anchors(1_000, 2_000, 3_000, 6_000)
    assert transform.scale == (6_000 - 2_000) / (3_000 - 1_000)
    assert transform.offset_ms == 2_000 - transform.scale * 1_000
    assert transform.apply_ms(1_000) == 2_000
    assert transform.apply_ms(3_000) == 6_000


def test_identity_transform() -> None:
    transform = fit_from_anchors(100_000, 100_000, 5_000_000, 5_000_000)
    assert transform.scale == 1.0
    assert transform.offset_ms == 0.0


def test_constant_offset() -> None:
    transform = fit_from_anchors(60_000, 70_000, 600_000, 610_000)
    assert transform.scale == 1.0
    assert transform.offset_ms == 10_000.0


def test_negative_drift() -> None:
    transform = fit_from_anchors(0, 0, 3_600_000, 3_456_000)
    assert transform.scale == pytest.approx(0.96)
    assert transform.apply_ms(3_600_000) == 3_456_000


def test_combined_offset_and_drift() -> None:
    transform = fit_from_anchors(142_640, 134_000, 5_610_800, 5_842_069)
    assert transform.scale == pytest.approx((5_842_069 - 134_000) / (5_610_800 - 142_640))
    assert transform.apply_ms(142_640) == 134_000
    assert transform.apply_ms(5_610_800) == 5_842_069


def test_apply_preserves_text_order_and_count() -> None:
    cues = build_dialogue(8)
    transform = LinearTransform(scale=1.042755, offset_ms=-8_622.0)
    converted = apply_transform(cues, transform)
    assert len(converted) == len(cues)
    for original, new in zip(cues, converted, strict=False):
        assert new.number == original.number
        assert new.order == original.order
        assert new.text == original.text
        assert new.start_ms == round(1.042755 * original.start_ms - 8_622.0)
        assert new.end_ms == round(1.042755 * original.end_ms - 8_622.0)


def test_degenerate_anchors_fail_explicitly() -> None:
    with pytest.raises(TransformError):
        fit_from_anchors(5_000, 1_000, 5_000, 9_000)  # T1 == T2
    with pytest.raises(TransformError):
        fit_from_anchors(1_000, 9_000, 5_000, 9_000)  # R1 == R2
    with pytest.raises(TransformError):
        fit_from_anchors(1_000, 9_000, 5_000, 1_000)  # reversed reference


def test_negative_output_timestamp_raises_without_clamping() -> None:
    transform = LinearTransform(scale=1.0, offset_ms=-500_000.0)
    cues = build_dialogue(3, start_ms=60_000)
    with pytest.raises(TransformError):
        apply_transform(cues, transform)


def test_rounding_is_deterministic() -> None:
    transform = LinearTransform(scale=1.042755, offset_ms=-8_622.0)
    cues = build_dialogue(5)
    assert apply_transform(cues, transform) == apply_transform(cues, transform)


def test_least_squares_recovers_line() -> None:
    pairs = [(float(t), 1.5 * float(t) + 200.0) for t in (0, 1_000, 2_000, 3_000)]
    transform = fit_least_squares(pairs)
    assert transform.scale == pytest.approx(1.5)
    assert transform.offset_ms == pytest.approx(200.0)


def test_least_squares_rejects_degenerate_input() -> None:
    with pytest.raises(TransformError):
        fit_least_squares([(5_000.0, 1_000.0)])
    with pytest.raises(TransformError):
        fit_least_squares([(5_000.0, 1_000.0), (5_000.0, 2_000.0)])
