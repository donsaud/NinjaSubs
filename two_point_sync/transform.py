"""Strict global affine timestamp transformation.

Implements exactly::

    scale = (R2 - R1) / (T2 - T1)
    offset = R1 - scale * T1
    new_time = scale * old_time + offset

applied identically to every Target cue start/end. No piecewise or nonlinear
behaviour lives in this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from two_point_sync.parser import Cue


class TransformError(ValueError):
    """Raised for degenerate fits or invalid transformed timestamps."""


@dataclass(frozen=True)
class LinearTransform:
    """A global ``new_time = scale * old_time + offset`` mapping (ms)."""

    scale: float
    offset_ms: float

    @property
    def offset_seconds(self) -> float:
        """Offset expressed in seconds (for reporting only)."""
        return self.offset_ms / 1000.0

    def apply_ms(self, value_ms: int) -> int:
        """Apply the transform to one timestamp with deterministic rounding."""
        result = self.scale * float(value_ms) + self.offset_ms
        if not math.isfinite(result):
            raise TransformError(f"Non-finite transform output for {value_ms}")
        rounded = int(round(result))
        if rounded < 0:
            raise TransformError(
                f"Transform yields negative timestamp for {value_ms}: {rounded}"
            )
        return rounded


def fit_from_anchors(t1_ms: int, r1_ms: int, t2_ms: int, r2_ms: int) -> LinearTransform:
    """Derive the affine transform from two verified anchor correspondences."""
    for name, value in (("T1", t1_ms), ("R1", r1_ms), ("T2", t2_ms), ("R2", r2_ms)):
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise TransformError(f"Anchor {name} is not finite: {value!r}")
    if t2_ms == t1_ms:
        raise TransformError("Degenerate anchors: T1 and T2 must be distinct")
    delta_t = float(t2_ms) - float(t1_ms)
    delta_r = float(r2_ms) - float(r1_ms)
    if delta_t == 0.0:
        raise TransformError("Degenerate anchors: zero target separation")
    scale = delta_r / delta_t
    if not math.isfinite(scale) or scale <= 0:
        raise TransformError(
            f"Invalid scale {scale}: anchors must be chronologically consistent "
            f"(T1={t1_ms}, T2={t2_ms}, R1={r1_ms}, R2={r2_ms})"
        )
    if delta_r <= 0:
        raise TransformError("Reference anchors must be chronologically ordered (R2 > R1)")
    offset_ms = float(r1_ms) - scale * float(t1_ms)
    if not math.isfinite(offset_ms):
        raise TransformError("Non-finite offset derived from anchors")
    return LinearTransform(scale=scale, offset_ms=offset_ms)


def fit_least_squares(pairs: list[tuple[float, float]]) -> LinearTransform:
    """Fit ``new_time = scale * old_time + offset`` by ordinary least squares.

    Diagnostic helper used to centre a majority timing regime after robust
    candidate selection; the two-anchor equations in :func:`fit_from_anchors`
    remain the sole estimator for accepted primary anchors.
    """
    if len(pairs) < 2:
        raise TransformError("Least-squares fit needs at least two pairs")
    count = float(len(pairs))
    mean_t = sum(pair[0] for pair in pairs) / count
    mean_r = sum(pair[1] for pair in pairs) / count
    numerator = sum((pair[0] - mean_t) * (pair[1] - mean_r) for pair in pairs)
    denominator = sum((pair[0] - mean_t) ** 2 for pair in pairs)
    if denominator == 0.0:
        raise TransformError("Degenerate fit: zero target variance")
    scale = numerator / denominator
    if not math.isfinite(scale) or scale <= 0:
        raise TransformError(f"Invalid least-squares scale {scale}")
    offset_ms = mean_r - scale * mean_t
    if not math.isfinite(offset_ms):
        raise TransformError("Non-finite least-squares offset")
    return LinearTransform(scale=scale, offset_ms=offset_ms)


def apply_transform(cues: list[Cue], transform: LinearTransform) -> list[Cue]:
    """Apply ``transform`` to every cue start/end, preserving text and order."""
    converted: list[Cue] = []
    for cue in cues:
        new_start = transform.apply_ms(cue.start_ms)
        new_end = transform.apply_ms(cue.end_ms)
        if new_end <= new_start:
            raise TransformError(
                f"Cue {cue.number} collapses under transform "
                f"(start={new_start}, end={new_end})"
            )
        converted.append(
            Cue(
                number=cue.number,
                order=cue.order,
                start_ms=new_start,
                end_ms=new_end,
                text=cue.text,
            )
        )
    for first, second in zip(converted, converted[1:], strict=False):
        if second.start_ms < first.start_ms:
            raise TransformError("Transform broke cue chronological order")
    return converted
