"""Lightweight independent verification of a derived linear transform.

Validation uses subtitle correspondences that were NOT the two estimation
anchors. It reports residuals across the beginning/middle/ending regions and
rejects global linear models that hide a discontinuity behind a good overall
average. Independent of the production NinjaSubs verifier.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from two_point_sync.transform import LinearTransform

# Documented nucleus validation policy (module-owned, production untouched).
MEDIAN_TOLERANCE_MS = 750.0
P90_TOLERANCE_MS = 2000.0
MAX_TOLERANCE_MS = 3000.0
REGION_MEDIAN_TOLERANCE_MS = 1500.0
MIN_CHECKPOINTS = 3
MIN_COVERAGE_FRACTION = 0.5
# A lone mismatching correspondence is noise; a coherent contradictory set is
# evidence. At least this many contradictory points are needed to veto.
MIN_CONTRADICTORY_SUPPORT = 2


@dataclass(frozen=True)
class Checkpoint:
    """One independent correspondence between transformed Target and Reference."""

    target_number: int
    reference_number: int
    target_ms: int
    reference_ms: int


@dataclass(frozen=True)
class CheckpointResidual:
    """A checkpoint annotated with its timing residual."""

    checkpoint: Checkpoint
    transformed_ms: int
    residual_ms: float


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of independent transform verification."""

    accepted: bool
    n_checkpoints: int
    median_abs_ms: float
    max_abs_ms: float
    p90_abs_ms: float
    region_medians_ms: dict[str, float]
    region_counts: dict[str, int]
    has_discontinuity: bool
    reasons: tuple[str, ...] = ()
    residuals: tuple[CheckpointResidual, ...] = ()


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = fraction * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _region_of(target_ms: int, span_start: int, span_end: int) -> str:
    width = max(1, span_end - span_start)
    position = (target_ms - span_start) / width
    if position < 1.0 / 3.0:
        return "beginning"
    if position < 2.0 / 3.0:
        return "middle"
    return "ending"


def validate_transform(
    transform: LinearTransform,
    checkpoints: list[Checkpoint],
    *,
    span_start_ms: int | None = None,
    span_end_ms: int | None = None,
    contradictory: list[Checkpoint] | None = None,
) -> ValidationReport:
    """Verify ``transform`` against independent checkpoints.

    ``span_start_ms``/``span_end_ms`` optionally describe the full Target film
    span so that checkpoints clustered near one anchor are reported as
    missing coverage. When omitted, the checkpoint span itself is used.

    ``contradictory`` optionally carries textually grounded correspondences
    that the candidate model does NOT explain (e.g. a re-timed opening
    region reported by anchor discovery). A coherent contradictory set vetoes
    acceptance: the validator can never bless a global transform while
    ignoring a contradictory region it was told about.
    """
    if len(checkpoints) < MIN_CHECKPOINTS:
        return ValidationReport(
            accepted=False,
            n_checkpoints=len(checkpoints),
            median_abs_ms=0.0,
            max_abs_ms=0.0,
            p90_abs_ms=0.0,
            region_medians_ms={},
            region_counts={},
            has_discontinuity=False,
            reasons=("insufficient_checkpoints",),
            residuals=(),
        )

    annotated: list[CheckpointResidual] = []
    for checkpoint in checkpoints:
        transformed = transform.scale * float(checkpoint.target_ms) + transform.offset_ms
        annotated.append(
            CheckpointResidual(
                checkpoint=checkpoint,
                transformed_ms=int(round(transformed)),
                residual_ms=transformed - float(checkpoint.reference_ms),
            )
        )

    abs_values = sorted(abs(item.residual_ms) for item in annotated)
    median_abs = float(statistics.median(abs_values))
    max_abs = float(abs_values[-1])
    p90_abs = float(_percentile(abs_values, 0.90))

    span_start = span_start_ms
    if span_start is None:
        span_start = min(item.checkpoint.target_ms for item in annotated)
    span_end = span_end_ms
    if span_end is None:
        span_end = max(item.checkpoint.target_ms for item in annotated)
    if span_end <= span_start:
        span_end = span_start + 1
    by_region: dict[str, list[float]] = {"beginning": [], "middle": [], "ending": []}
    for item in annotated:
        region = _region_of(item.checkpoint.target_ms, span_start, span_end)
        by_region[region].append(abs(item.residual_ms))
    region_medians = {
        region: float(statistics.median(values)) if values else 0.0
        for region, values in by_region.items()
    }
    region_counts = {region: len(values) for region, values in by_region.items()}

    reasons: list[str] = []
    # Coverage: checkpoints must span the film, not cluster near one anchor.
    film_span = max(1, span_end - span_start)
    ordered_targets = sorted(item.checkpoint.target_ms for item in annotated)
    coverage = (ordered_targets[-1] - ordered_targets[0]) / film_span
    populated_regions = sum(1 for values in by_region.values() if values)
    if populated_regions < 3:
        reasons.append("missing_region_coverage")
    if coverage < MIN_COVERAGE_FRACTION:
        reasons.append("insufficient_temporal_coverage")

    # Discontinuity: a bad middle hidden by good endpoints must fail.
    has_discontinuity = False
    if (
        populated_regions == 3
        and region_medians["middle"] > REGION_MEDIAN_TOLERANCE_MS
        and region_medians["beginning"] <= REGION_MEDIAN_TOLERANCE_MS
        and region_medians["ending"] <= REGION_MEDIAN_TOLERANCE_MS
    ):
        has_discontinuity = True
        reasons.append("nonlinear_discontinuity_in_middle")
    elif any(
        median > REGION_MEDIAN_TOLERANCE_MS for median in region_medians.values()
    ):
        reasons.append("regional_median_exceeds_tolerance")

    if median_abs > MEDIAN_TOLERANCE_MS:
        reasons.append("median_residual_exceeds_tolerance")
    if p90_abs > P90_TOLERANCE_MS:
        reasons.append("p90_residual_exceeds_tolerance")
    if max_abs > MAX_TOLERANCE_MS:
        reasons.append("max_residual_exceeds_tolerance")

    # Contradictory region veto: textually grounded correspondences that the
    # transform cannot explain reject the global claim, no matter how good
    # the checkpoint statistics look on their own.
    if contradictory is not None and len(contradictory) >= MIN_CONTRADICTORY_SUPPORT:
        contra_abs = sorted(
            abs(
                transform.scale * float(item.target_ms)
                + transform.offset_ms
                - float(item.reference_ms)
            )
            for item in contradictory
        )
        contra_median = float(statistics.median(contra_abs))
        if contra_median > REGION_MEDIAN_TOLERANCE_MS:
            has_discontinuity = True
            reasons.append("contradictory_region_rejected")

    accepted = not reasons
    if accepted:
        reasons.append("within_linear_policy")
    return ValidationReport(
        accepted=accepted,
        n_checkpoints=len(annotated),
        median_abs_ms=median_abs,
        max_abs_ms=max_abs,
        p90_abs_ms=p90_abs,
        region_medians_ms=region_medians,
        region_counts=region_counts,
        has_discontinuity=has_discontinuity,
        reasons=tuple(reasons),
        residuals=tuple(annotated),
    )
