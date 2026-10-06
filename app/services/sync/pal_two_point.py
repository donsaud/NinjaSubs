"""Two-point PAL-speedup synchronization fallback.

Narrowly scoped path for genuine PAL/NTSC-style speed conversion, where
subtitle timing is globally stretched or compressed. It fires only after Alass
and the existing verified paths have failed, and only when the evidence fits
one tightly defined class:

    reference_time = target_time * slope + intercept

with ``slope`` extremely close to ``25 / 23.976`` (PAL speedup), low residual
p95, strong structural agreement, and high cue coverage.

Why this narrow class is safe to accept where a generic linear fit is not: a
wrong-cut pair essentially never lands within ±0.001 of the PAL ratio *and*
simultaneously shows low residuals, high structure, and full coverage. The
slope match is independent corroboration, analogous to how the constant-shift
exemption treats a tight median residual. Nothing here weakens the global
verifier: the analyzer still measures the transformed output with all gates
intact, and only an additional explicit policy acceptance -- recorded in
provenance -- allows serving.

What this module never does:

* derive anchors from metadata (credits, URLs, handles are refused);
* invent correspondence (first/last *dialogue runs*, never first/last cues);
* touch ranking, MatchTier, MovieHash, cache trust, or request budgets;
* implement piecewise fitting (structurally different cuts stay rejected).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from app.services.sync.alignment import (
    MAX_P95_MS_FOR_STABLE,
    AlignmentAnalyzer,
)
from app.services.sync.large_offset_investigation import (
    LARGE_OFFSET_MIN_STRUCTURAL_SIMILARITY,
)
from app.services.sync.ninja_points import (
    MAX_SLOPE,
    MIN_SLOPE,
    LinearSyncModel,
    render_srt,
)

logger = logging.getLogger(__name__)

#: Strategy identifier for provenance, logs, cache decision segment, and
#: shadow telemetry. Stable and versionable.
STRATEGY_ID = "two_point_linear_pal"

#: Model version, baked into provenance so future algorithm changes cannot
#: masquerade as this one in cache or telemetry.
MODEL_VERSION = "pal-two-point-v1"

#: Exact PAL speedup ratio: 25 fps content mastered at 23.976 fps.
PAL_SLOPE = 25.0 / 23.976

#: Maximum |measured slope - PAL| for PAL-class membership. Timing analysis:
#: cue timestamps quantize to milliseconds and anchor placement within a
#: dialogue event carries ~1-2 s of uncertainty; over a ~90-minute span that
#: is at most ~4e-4 of slope error. 0.001 gives ~2.5x headroom while still
#: excluding the nearest non-PAL ratio (25/24 = 1.041667, off by 0.00104) and,
#: overwhelmingly, the 1.0 no-op case (off by 0.0427).
PAL_SLOPE_TOLERANCE = 0.001

#: Minimum span between the two anchors. A short span cannot distinguish drift
#: from placement noise; ~50 minutes makes a 0.001 slope error visible as
#: ~3 s of end displacement, far above cue-timing noise.
MIN_ANCHOR_SPAN_MS = 3_000_000

#: Coverage floor, deliberately above the generic MIN_CUE_RETENTION (0.80): a
#: path that bypasses the movement gate must show a nearly complete timeline.
MIN_COVERAGE = 0.90

#: Residual p95 ceiling: the standard stable-alignment bound, reused exactly.
MAX_RESIDUAL_P95_MS = MAX_P95_MS_FOR_STABLE

#: Structural floor: the large-offset investigation's floor, reused exactly.
MIN_STRUCTURAL_SIMILARITY = LARGE_OFFSET_MIN_STRUCTURAL_SIMILARITY

#: Dialogue-run parameters for endpoint selection, mirroring the existing
#: first-dialogue-cluster semantics (2+ consecutive dialogue cues).
_ENDPOINT_RUN_LENGTH = 2
_ENDPOINT_MAX_GAP_MS = 20_000

Cue = tuple[int, int, str]


@dataclass(frozen=True)
class EndpointAnchor:
    """One end of the two-point correspondence."""

    target_ms: int
    reference_ms: int
    kind: str  # "first" or "last"


@dataclass
class PalTwoPointResult:
    """Outcome of a two-point PAL attempt. Never served on its own."""

    accepted: bool
    slope: float | None = None
    intercept_ms: float | None = None
    first: EndpointAnchor | None = None
    last: EndpointAnchor | None = None
    transformed_text: str | None = None
    evaluation: Any | None = None
    reason_codes: list[str] = field(default_factory=list)


def _is_usable_dialogue(text: str) -> bool:
    """Genuine spoken dialogue only: dialogue text AND not metadata."""
    from app.services.subtitle_matcher import _is_dialogue_text
    from app.services.sync.ninja_points import _is_metadata_cue

    return bool(text and text.strip()) and _is_dialogue_text(text) and not _is_metadata_cue(text)


def find_endpoint_anchors(
    target_cues: list[Cue], reference_cues: list[Cue]
) -> tuple[EndpointAnchor, EndpointAnchor] | None:
    """First and last genuine dialogue cues of each subtitle, paired by position.

    Positional correspondence (first<->first, last<->last) is a hypothesis, not
    proof: same-movie subtitles overwhelmingly open and close on the same
    events, and every downstream gate (PAL slope, residuals, structure,
    coverage, verifier) must still pass. Metadata, credits, URLs, handles, and
    non-dialogue are refused at both ends; each end additionally requires a
    2-cue dialogue run so an isolated blip can never anchor the transform.
    """
    if not target_cues or not reference_cues:
        return None

    def first_run(cues: list[Cue]) -> int | None:
        run: list[int] = []
        prev_end: int | None = None
        for index, (start, end, text) in enumerate(cues):
            if not _is_usable_dialogue(text):
                if len(run) >= _ENDPOINT_RUN_LENGTH:
                    break
                run = []
                prev_end = None
                continue
            if run and prev_end is not None and start - prev_end > _ENDPOINT_MAX_GAP_MS:
                if len(run) >= _ENDPOINT_RUN_LENGTH:
                    break
                run = []
            run.append(index)
            prev_end = end
            if len(run) >= _ENDPOINT_RUN_LENGTH:
                return run[0]
        return run[0] if len(run) >= _ENDPOINT_RUN_LENGTH else None

    def last_run(cues: list[Cue]) -> int | None:
        run: list[int] = []
        prev_start: int | None = None
        for index in range(len(cues) - 1, -1, -1):
            start, end, text = cues[index]
            if not _is_usable_dialogue(text):
                if len(run) >= _ENDPOINT_RUN_LENGTH:
                    break
                run = []
                prev_start = None
                continue
            if run and prev_start is not None and prev_start - end > _ENDPOINT_MAX_GAP_MS:
                if len(run) >= _ENDPOINT_RUN_LENGTH:
                    break
                run = []
            run.append(index)
            prev_start = start
            if len(run) >= _ENDPOINT_RUN_LENGTH:
                return run[0]
        return run[0] if len(run) >= _ENDPOINT_RUN_LENGTH else None

    t_first = first_run(target_cues)
    r_first = first_run(reference_cues)
    t_last = last_run(target_cues)
    r_last = last_run(reference_cues)
    if t_first is None or r_first is None or t_last is None or r_last is None:
        return None
    first = EndpointAnchor(
        target_ms=target_cues[t_first][0], reference_ms=reference_cues[r_first][0],
        kind="first",
    )
    last = EndpointAnchor(
        target_ms=target_cues[t_last][0], reference_ms=reference_cues[r_last][0],
        kind="last",
    )
    if not (last.target_ms > first.target_ms and last.reference_ms > first.reference_ms):
        return None
    return first, last


def is_pal_slope(slope: float) -> bool:
    """True only for slopes within tolerance of the exact PAL ratio."""
    if not math.isfinite(slope):
        return False
    return abs(slope - PAL_SLOPE) <= PAL_SLOPE_TOLERANCE


def attempt_pal_two_point(
    target_text: str,
    reference_text: str,
    analyzer: AlignmentAnalyzer | None = None,
    *,
    max_plausible_offset_ms: int = 20_000,
) -> PalTwoPointResult:
    """Try the two-point PAL transform. Returns acceptance, never serves.

    Pipeline: endpoint anchors -> slope/intercept -> PAL-class check ->
    LinearSyncModel transform -> existing analyzer (all gates intact) ->
    narrow policy acceptance. Any failure at any stage returns
    ``accepted=False`` with reason codes; the caller serves the original.
    """
    reasons: list[str] = []
    try:
        from app.services.subtitle_matcher import parse_srt_cues

        target_cues = parse_srt_cues(target_text)
        reference_cues = parse_srt_cues(reference_text)
    except Exception as exc:
        return PalTwoPointResult(accepted=False, reason_codes=[f"parse_failed:{type(exc).__name__}"])

    endpoints = find_endpoint_anchors(target_cues, reference_cues)
    if endpoints is None:
        return PalTwoPointResult(accepted=False, reason_codes=["no_endpoints"])
    first, last = endpoints

    span_ms = last.target_ms - first.target_ms
    if span_ms < MIN_ANCHOR_SPAN_MS:
        return PalTwoPointResult(
            accepted=False, first=first, last=last,
            reason_codes=["span_too_short"],
        )
    slope = (last.reference_ms - first.reference_ms) / span_ms
    intercept = first.reference_ms - slope * first.target_ms
    if not (MIN_SLOPE <= slope <= MAX_SLOPE):
        return PalTwoPointResult(
            accepted=False, first=first, last=last,
            reason_codes=["slope_out_of_bounds"],
        )
    if not is_pal_slope(slope):
        return PalTwoPointResult(
            accepted=False, first=first, last=last,
            reason_codes=["not_pal_class"],
        )

    model = LinearSyncModel(slope=slope, intercept_ms=intercept, inliers=2)
    transformed = render_srt(model.apply([(s, e, t) for s, e, t in target_cues]))
    analyzer = analyzer or AlignmentAnalyzer()
    try:
        evaluation = analyzer.analyze(
            target_cues,
            transformed,
            reference_text,
            alass_applied=True,
            alass_successful=True,
            max_plausible_offset_ms=max_plausible_offset_ms,
        )
    except Exception as exc:
        return PalTwoPointResult(
            accepted=False, first=first, last=last,
            reason_codes=[f"verification_raised:{type(exc).__name__}"],
        )

    # Narrow PAL policy: every gate must pass. p95/struct/coverage reuse
    # existing bounds; the slope match above is the independent corroboration
    # that justifies accepting despite movement MAD (which necessarily spans
    # the correction magnitude for a true speedup).
    p95 = evaluation.p95_offset_ms
    struct = evaluation.structural_similarity
    coverage = evaluation.coverage_score
    if p95 is None or p95 > MAX_RESIDUAL_P95_MS:
        reasons.append("p95_too_high")
    if struct is None or struct < MIN_STRUCTURAL_SIMILARITY:
        reasons.append("structure_too_low")
    if coverage is None or coverage < MIN_COVERAGE:
        reasons.append("coverage_too_low")
    if reasons:
        return PalTwoPointResult(
            accepted=False, first=first, last=last, slope=slope,
            intercept_ms=intercept, transformed_text=transformed,
            evaluation=evaluation, reason_codes=reasons,
        )
    evaluation.reasons.append(
        f"{STRATEGY_ID}: PAL-speedup two-point transform "
        f"(slope={slope:.6f}); verification measured output only"
    )
    return PalTwoPointResult(
        accepted=True, first=first, last=last, slope=slope,
        intercept_ms=intercept, transformed_text=transformed,
        evaluation=evaluation,
        reason_codes=["pal_accepted"],
    )
