"""NinjaSync Points shadow evaluation: measure, never serve.

This module answers one question without touching user-visible behavior:
"would NinjaSync Points have produced a trustworthy candidate where Alass did
not?" It runs the offline ``HybridSyncEngine``, classifies the outcome, and
logs bounded structured telemetry. It has no code path that serves, caches,
ranks, or otherwise alters production behavior -- the only outputs are a
``ShadowEvaluation`` value and log lines.

Fail-closed contract: every exception inside evaluation becomes a
``SHADOW_ERROR`` classification. A shadow failure is telemetry, never a
production failure, and it can never promote, serve, or cache anything.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Shadow classifications. Deterministic strings; never ambiguous prose.
ALASS_WINS = "ALASS_WINS"
NINJA_POINTS_CONSTANT_VERIFIED = "NINJA_POINTS_CONSTANT_VERIFIED"
NINJA_POINTS_LINEAR_VERIFIED = "NINJA_POINTS_LINEAR_VERIFIED"
NINJA_POINTS_REJECTED = "NINJA_POINTS_REJECTED"
REFERENCE_REJECTED = "REFERENCE_REJECTED"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
UNSUPPORTED_STRUCTURE = "UNSUPPORTED_STRUCTURE"
SHADOW_ERROR = "SHADOW_ERROR"

CLASSIFICATIONS: tuple[str, ...] = (
    ALASS_WINS,
    NINJA_POINTS_CONSTANT_VERIFIED,
    NINJA_POINTS_LINEAR_VERIFIED,
    NINJA_POINTS_REJECTED,
    REFERENCE_REJECTED,
    INSUFFICIENT_EVIDENCE,
    UNSUPPORTED_STRUCTURE,
    SHADOW_ERROR,
)


@dataclass(frozen=True)
class ShadowEvaluation:
    """One shadow run. Counts and metrics only -- never subtitle content."""

    classification: str
    alass_state: str | None
    ninja_best_strategy: str | None
    anchor_count: int
    offset_ms: float | None
    slope: float | None
    median_residual_ms: float | None
    mad_ms: float | None
    p95_ms: float | None
    structural_similarity: float | None
    rejection_reason: str | None
    duration_ms: float
    eligible_for_future_serving: bool
    error: str | None = None
    # Cues refused as translator/uploader metadata during detection. A count
    # only -- never cue text, URLs, or handles.
    metadata_skipped: int = 0


_counters_lock = threading.Lock()
_counters: dict[str, int] = dict.fromkeys(CLASSIFICATIONS, 0)
_shadow_ms_total = 0.0


def counters_snapshot() -> dict[str, Any]:
    """Copy of shadow counters. Tests and diagnostics only."""
    with _counters_lock:
        return {"counts": dict(_counters), "total_ms": _shadow_ms_total}


def reset_counters() -> None:
    """Clear shadow counters. Tests only; production never calls this."""
    global _shadow_ms_total
    with _counters_lock:
        for key in _counters:
            _counters[key] = 0
        _shadow_ms_total = 0.0


def _record_counters(classification: str, duration_ms: float) -> None:
    global _shadow_ms_total
    with _counters_lock:
        if classification in _counters:
            _counters[classification] += 1
        _shadow_ms_total += max(0.0, duration_ms)


def _cut_of(evaluation: Any) -> str | None:
    value = getattr(evaluation, "cut_verdict", None)
    return value if isinstance(value, str) else None


def run_shadow_evaluation(
    target_text: str,
    reference_text: str,
    *,
    alass_state_value: str | None,
    alass_verified: bool,
    analyzer: Any | None = None,
) -> ShadowEvaluation:
    """Evaluate NinjaSync Points in shadow mode. Never raises, never serves.

    When Alass already verified, NinjaSync is not run at all: there is nothing
    left to prove, and redundant work would only add latency. Otherwise the
    offline engine runs, every candidate is verified by the existing analyzer,
    and the outcome is classified. A VERIFIED ninja candidate is recorded as
    *eligible for future serving* -- explicitly not served, not cached.
    """
    started = time.monotonic()

    def finish(
        classification: str,
        *,
        ninja_best_strategy: str | None = None,
        anchor_count: int = 0,
        metadata_skipped: int = 0,
        offset_ms: float | None = None,
        slope: float | None = None,
        median_residual_ms: float | None = None,
        mad_ms: float | None = None,
        p95_ms: float | None = None,
        structural_similarity: float | None = None,
        rejection_reason: str | None = None,
        eligible: bool = False,
        error: str | None = None,
    ) -> ShadowEvaluation:
        duration_ms = (time.monotonic() - started) * 1000.0
        evaluation = ShadowEvaluation(
            classification=classification,
            alass_state=alass_state_value,
            ninja_best_strategy=ninja_best_strategy,
            anchor_count=anchor_count,
            offset_ms=offset_ms,
            slope=slope,
            median_residual_ms=median_residual_ms,
            mad_ms=mad_ms,
            p95_ms=p95_ms,
            structural_similarity=structural_similarity,
            rejection_reason=rejection_reason,
            duration_ms=duration_ms,
            eligible_for_future_serving=eligible,
            error=error,
            metadata_skipped=metadata_skipped,
        )
        _record_counters(classification, duration_ms)
        logger.info(
            "[ninja-shadow] classification=%s alass_state=%s ninja_best=%s "
            "anchors=%d metadata_skipped=%d offset_ms=%s slope=%s "
            "median_residual_ms=%s mad_ms=%s "
            "p95_ms=%s structural=%s rejection=%s duration_ms=%.1f eligible=%s",
            evaluation.classification,
            evaluation.alass_state,
            evaluation.ninja_best_strategy,
            evaluation.anchor_count,
            evaluation.metadata_skipped,
            evaluation.offset_ms,
            evaluation.slope,
            evaluation.median_residual_ms,
            evaluation.mad_ms,
            evaluation.p95_ms,
            evaluation.structural_similarity,
            evaluation.rejection_reason,
            evaluation.duration_ms,
            evaluation.eligible_for_future_serving,
        )
        return evaluation

    try:
        if alass_verified:
            return finish(ALASS_WINS)

        from app.services.sync.ninja_points import HybridSyncEngine

        engine = HybridSyncEngine(analyzer=analyzer)
        result = engine.run(target_text, reference_text)
        anchor_count = result.anchor_count
        skipped = result.metadata_skipped
        best = result.best
        if best is not None and best.verified:
            evaluation = best.evaluation
            strategy = best.strategy_id
            classification = (
                NINJA_POINTS_CONSTANT_VERIFIED
                if strategy == "ninja_points_constant"
                else NINJA_POINTS_LINEAR_VERIFIED
            )
            return finish(
                classification,
                ninja_best_strategy=strategy,
                anchor_count=anchor_count,
                offset_ms=_model_offset(best),
                slope=_model_slope(best),
                median_residual_ms=getattr(evaluation, "median_offset_ms", None),
                mad_ms=getattr(evaluation, "mad_offset_ms", None),
                p95_ms=getattr(evaluation, "p95_offset_ms", None),
                structural_similarity=getattr(evaluation, "structural_similarity", None),
                eligible=True,
                metadata_skipped=skipped,
            )

        # Nothing verified. Distinguish why, honestly.
        if anchor_count == 0:
            return finish(INSUFFICIENT_EVIDENCE, anchor_count=0,
                metadata_skipped=skipped)
        cuts = {
            _cut_of(c.evaluation)
            for c in result.candidates
            if c.evaluation is not None
        } - {None}
        if cuts == {"different_cut"} and result.candidates:
            return finish(
                UNSUPPORTED_STRUCTURE,
                anchor_count=anchor_count,
                metadata_skipped=skipped,
                rejection_reason="different_cut",
            )
        if not result.candidates:
            # Anchors existed but no model could even be fitted: the evidence
            # contradicts itself, which points at the reference, not the fit.
            return finish(
                REFERENCE_REJECTED,
                anchor_count=anchor_count,
                metadata_skipped=skipped,
                rejection_reason="no_model_fitted",
            )
        reasons: list[str] = []
        for candidate in result.candidates:
            reason = getattr(candidate.evaluation, "rejection_reason", None)
            value = getattr(reason, "value", reason)
            if value is not None:
                reasons.append(str(value))
        return finish(
            NINJA_POINTS_REJECTED,
            anchor_count=anchor_count,
            metadata_skipped=skipped,
            rejection_reason=",".join(sorted(set(reasons))) or "unverified",
        )
    except Exception as exc:  # fail closed: shadow errors are telemetry
        message = str(exc)
        return finish(SHADOW_ERROR, error=message[:200] or type(exc).__name__)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        return value
    return None


def _model_offset(candidate: Any) -> float | None:
    return _as_float(getattr(candidate, "offset_ms", None))


def _model_slope(candidate: Any) -> float | None:
    return _as_float(getattr(candidate, "slope", None))
