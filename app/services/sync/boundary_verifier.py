"""Observational verification of a FIXED output with one edition boundary.

No timestamp generation, authorization, cache access or legacy cue pairing.
Target/output correspondence requires preserved usable-cue order and population;
reference evidence uses union occupancy, runs and gaps, across languages/splits.
These conservative shadow gates are not production verification thresholds.
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
import statistics
import threading
import time
from bisect import bisect_left, bisect_right
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial

from app.services.subtitle_matcher import parse_srt_cues
from app.services.sync.alignment import SubtitleEvaluation
from app.services.sync.case_capture import capture_boundary, capture_shadow_status
from app.services.sync.pal_two_point import _is_usable_dialogue
from app.services.sync.timeline_affine import clip, duration, intersection, occupancy_score, union

logger = logging.getLogger(__name__)
Cue = tuple[int, int, str]
MAX_CHARS = 500_000
MAX_CUES = 4000
MIN_SUPPORT = 20
MIN_SPAN_MS = 60_000
MIN_JUMP_MS = 5000
MOVEMENT_RESIDUAL_MS = 350
BOUNDARY_ERROR_MS = 1000
MIN_COVERAGE = 0.90
MIN_WINDOW_SCORE = 0.80
MAX_UNSUPPORTED_MS = 5000
BUDGET_SECONDS = 0.25


@dataclass(frozen=True)
class SegmentEvidence:
    support: int
    span_ms: int
    offset_ms: float
    residual_p95_ms: float
    residual_mad_ms: float
    runs: int
    reference_runs: int
    reference_start_p95_ms: float
    occupancy: float
    forward_coverage: float
    reverse_coverage: float
    windows: tuple[float, ...]


@dataclass(frozen=True)
class BoundaryResult:
    accepted: bool = False
    classification: str = "UNSUPPORTED"
    artifact: str = "raw_transformed"
    boundary_ms: int | None = None
    boundary_interval_ms: tuple[int, int] | None = None
    offset_a_ms: float | None = None
    offset_b_ms: float | None = None
    jump_ms: float | None = None
    gap_disagreement_ms: float | None = None
    improvement: float | None = None
    segments: tuple[SegmentEvidence, ...] = ()
    holdout_pass: bool = False
    holdout_support: tuple[int, int] = (0, 0)
    holdout_p95_ms: tuple[float, float] = (0, 0)
    reasons: tuple[str, ...] = ()
    duration_ms: float = 0


class _Refusal(Exception):
    pass


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise _Refusal(reason)


def _percentile(values: list[float]) -> float:
    return sorted(values)[math.ceil(len(values) * .95) - 1]


def _ordered(text: str) -> list[Cue]:
    # Parsing each block separately preserves file order; the normal parser sorts.
    blocks = re.split(r"\n[ \t]*\n", text.replace("\r\n", "\n").replace("\r", "\n"))
    _require(len(blocks) <= MAX_CUES+2, "input_limit")
    cues = [c for block in blocks for c in parse_srt_cues(block)]
    _require(len(cues) <= MAX_CUES, "input_limit")
    _require(all(s >= 0 and e > s for s, e, _ in cues), "invalid_timing")
    _require(all(a[0] <= b[0] and a[1] <= b[1]
                 for a, b in zip(cues, cues[1:], strict=False)), "ordering_violation")
    return [c for c in cues if _is_usable_dialogue(c[2])]


def _occupied(cues: list[Cue]) -> list[tuple[float, float]]:
    return union([(float(s), float(e)) for s, e, _ in cues], 250)


def _coverage(cues: list[Cue], reference: list[tuple[float, float]]) -> float:
    occupied = _occupied(cues)
    padded = union([(s - 350, e + 350) for s, e in reference])
    return intersection(occupied, padded) / duration(occupied)


def _evidence(cues: list[Cue], reference: list[tuple[float, float]],
              offset: float, residuals: list[float]) -> SegmentEvidence:
    occupied = _occupied(cues)
    lo, hi = occupied[0][0], occupied[-1][1]
    local = clip(reference, lo, hi)
    _require(bool(local), "missing_reference_support")
    windows = tuple(occupancy_score(clip(occupied, lo + (hi-lo)*i/4,
        lo + (hi-lo)*(i+1)/4), clip(local, lo + (hi-lo)*i/4,
        lo + (hi-lo)*(i+1)/4)) for i in range(4))
    reverse = intersection(local, union([(s-350, e+350) for s, e in occupied])) / duration(local)
    reference_starts = [s for s, _ in reference]
    start_errors = []
    for s, _ in occupied:
        index = bisect_left(reference_starts, s)
        start_errors.append(min(abs(s-reference_starts[j])
                               for j in {max(0, index-1), min(index, len(reference)-1)}))
    median = statistics.median(residuals)
    return SegmentEvidence(len(cues), int(hi-lo), offset, _percentile(residuals),
        statistics.median([abs(x-median) for x in residuals]), len(occupied), len(local),
        _percentile(start_errors), occupancy_score(occupied, local),
        _coverage(cues, reference), reverse, windows)


def _boundary_score(target: list[Cue], output: list[Cue],
                    reference: list[tuple[float, float]], k: int,
                    a: float, b: float) -> tuple[float, float]:
    # A real target silence must straddle the boundary. Inspect consecutive
    # reference runs, rather than choosing unrelated nearest starts/ends.
    _require(target[k-1][1] <= target[k][0], "cue_crosses_boundary")
    _require(output[k-1][1] <= output[k][0], "crossing_mapping")
    gaps = [(abs(end-(target[k-1][1]+a)) + abs(start-(target[k][0]+b)),
             start-end) for (_, end), (start, _) in zip(reference, reference[1:], strict=False)]
    _require(bool(gaps), "unsupported_boundary")
    error, gap = min(gaps)
    target_gap = target[k][0] - target[k-1][1]
    disagreement = abs((gap-target_gap) - (b-a))
    return error, disagreement


def verify_boundary(target_text: str, reference_text: str, transformed_text: str,
                    *, global_verified: bool = False,
                    artifact: str = "raw_transformed",
                    cancelled: threading.Event | None = None) -> BoundaryResult:
    """Return separate shadow evidence; never mutate inputs or create timestamps."""
    started = time.monotonic()
    details: dict = {}
    # Only fixed vocabulary can reach logs, even when callers pass arbitrary labels.
    safe_artifact = "final_served" if artifact == "final_served" else "raw_transformed"

    def checkpoint() -> None:
        _require(cancelled is None or not cancelled.is_set(), "cancelled")
        _require(time.monotonic()-started <= BUDGET_SECONDS, "budget_exceeded")

    try:
        _require(not global_verified, "simpler_verified_model")
        _require(max(map(len, (target_text, reference_text, transformed_text))) <= MAX_CHARS,
                 "input_limit")
        checkpoint()
        target, reference_cues, output = map(_ordered, (target_text, reference_text, transformed_text))
        _require(len(target) == len(output), "cue_population_changed")
        _require(len(target) >= 2*MIN_SUPPORT and len(reference_cues) >= 2*MIN_SUPPORT,
                 "sparse_evidence")
        checkpoint()
        shifts = [o[0]-t[0] for t, o in zip(target, output, strict=True)]
        end_shifts = [o[1]-t[1] for t, o in zip(target, output, strict=True)]
        steps = [i for i in range(1, len(shifts)) if abs(shifts[i]-shifts[i-1]) >= MIN_JUMP_MS]
        _require(len(steps) == 1, "not_exactly_one_boundary")
        k = steps[0]
        _require(min(k, len(target)-k) >= MIN_SUPPORT, "tiny_segment")
        a, b = statistics.median(shifts[:k]), statistics.median(shifts[k:])
        _require(b-a >= MIN_JUMP_MS, "downward_or_small_jump")
        details.update(boundary_ms=target[k][0], boundary_interval_ms=(target[k-1][1], target[k][0]),
                       offset_a_ms=a, offset_b_ms=b, jump_ms=b-a)
        residuals = [max(abs(s-offset), abs(e-offset)) for i, (s, e) in
                     enumerate(zip(shifts, end_shifts, strict=True)) for offset in [a if i<k else b]]
        # Max as well as p95: a third small population must not disappear below a percentile.
        _require(max(residuals) <= MOVEMENT_RESIDUAL_MS, "unexplained_regime_or_movement")
        one = statistics.median(shifts)
        loss = sum(abs(s-one) for s in shifts)
        improvement = 1-sum(residuals)/loss if loss else 0
        details["improvement"] = improvement
        _require(improvement >= .90, "insufficient_improvement")
        reference = _occupied(reference_cues)
        error, disagreement = _boundary_score(target, output, reference, k, a, b)
        details["gap_disagreement_ms"] = disagreement
        _require(error <= 2*BOUNDARY_ERROR_MS and disagreement <= BOUNDARY_ERROR_MS,
                 "unsupported_boundary")
        checkpoint()
        segments = tuple(_evidence(output[lo:hi], reference, offset, residuals[lo:hi])
                         for lo, hi, offset in ((0, k, a), (k, len(target), b)))
        details["segments"] = segments
        for evidence in segments:
            _require(evidence.span_ms >= MIN_SPAN_MS and evidence.runs >= 3
                     and evidence.reference_runs >= 3, "insufficient_distributed_support")
            _require(min(evidence.windows) >= MIN_WINDOW_SCORE
                     and min(evidence.forward_coverage, evidence.reverse_coverage) >= MIN_COVERAGE,
                     "poor_reference_agreement")
            _require(evidence.reference_start_p95_ms <= BOUNDARY_ERROR_MS,
                     "unsupported_landmarks")
        # Detect continuous unsupported dialogue regions, not just population quantiles.
        padded = union([(s-350, e+350) for s, e in reference])
        padded_starts = [s for s, _ in padded]
        padded_ends = [e for _, e in padded]
        unsupported_start: int | None = None
        for start, end, _ in output:
            local = padded[bisect_right(padded_ends, start):bisect_left(padded_starts, end)]
            if intersection([(start, end)], local)/(end-start) < .5:
                if unsupported_start is None:
                    unsupported_start = start
                _require(end-unsupported_start <= MAX_UNSUPPORTED_MS, "unsupported_dialogue_interval")
            else:
                unsupported_start = None
        checkpoint()
        # Alternate whole dialogue runs. Neither offsets nor boundary are fitted
        # from withheld output shifts; the reference resolves training ambiguity.
        training: list[int] = []
        held: list[int] = []
        run, run_end = -1, -1
        for i, (s, e, _) in enumerate(target):
            if s > run_end+250:
                run += 1
            (training if run % 2 == 0 else held).append(i)
            run_end = max(run_end, e)
        transitions = [(left, right) for left, right in zip(training, training[1:], strict=False)
                       if abs(shifts[right]-shifts[left]) >= MIN_JUMP_MS]
        _require(len(transitions) == 1, "holdout_ambiguous_regimes")
        left, right = transitions[0]
        _require(right-left <= 32, "holdout_boundary_ambiguity")
        ha = statistics.median([shifts[i] for i in training if i <= left])
        hb = statistics.median([shifts[i] for i in training if i >= right])
        candidates: list[tuple[float, int]] = []
        for candidate in range(left+1, right+1):
            checkpoint()
            if target[candidate-1][1] > target[candidate][0]:
                continue
            score, gap_error = _boundary_score(target, output, reference, candidate, ha, hb)
            if score <= 2*BOUNDARY_ERROR_MS and gap_error <= BOUNDARY_ERROR_MS:
                candidates.append((score, candidate))
        _require(bool(candidates), "holdout_boundary_unsupported")
        hk = min(candidates)[1]
        _require(hk == k, "holdout_boundary_disagreement")
        held_support: list[int] = []
        held_p95: list[float] = []
        for lo, hi, offset in ((0, hk, ha), (hk, len(target), hb)):
            fit = [i for i in training if lo <= i < hi]
            withheld = [i for i in held if lo <= i < hi]
            _require(min(len(fit), len(withheld)) >= 8, "holdout_sparse")
            errors = [max(abs(shifts[i]-offset), abs(end_shifts[i]-offset)) for i in withheld]
            _require(max(errors) <= MOVEMENT_RESIDUAL_MS, "holdout_movement_failed")
            _require(_coverage([output[i] for i in withheld], reference) >= MIN_COVERAGE,
                     "holdout_reference_failed")
            held_support.append(len(withheld))
            held_p95.append(_percentile(errors))
        checkpoint()
        details.update(holdout_pass=True, holdout_support=tuple(held_support),
                       holdout_p95_ms=tuple(held_p95))
        return BoundaryResult(True, "EDGE_DISCONTINUITY", safe_artifact, **details,
                              duration_ms=(time.monotonic()-started)*1000)
    except _Refusal as exc:
        return BoundaryResult(artifact=safe_artifact, reasons=(str(exc),), **details,
                              duration_ms=(time.monotonic()-started)*1000)


@dataclass
class BoundaryShadow:
    """One background job per owner; results cannot participate in serving."""
    tasks: set[asyncio.Task[None]] = field(default_factory=set)
    _closed: bool = False

    def _finished(self, task: asyncio.Task[None]) -> None:
        self.tasks.discard(task)
        if not task.cancelled():
            # Retrieve even unexpected failures (e.g. a logging handler error).
            task.exception()

    def schedule(self, target: str, reference: str, transformed: str,
                 evaluation: SubtitleEvaluation) -> None:
        try:
            if (self._closed or self.tasks or evaluation.sync_state.value.startswith("verified")
                    or not evaluation.alass_successful
                    or not (evaluation.cut_verdict == "piecewise"
                            or evaluation.rejection_reason == "low_confidence")):
                capture_shadow_status(reference, transformed, "not_scheduled")
                return
            if max(map(len, (target, reference, transformed))) > MAX_CHARS:
                capture_shadow_status(reference, transformed, "input_limit_skip")
                return
            loop = asyncio.get_running_loop()
            job = self._run(target, reference, transformed)
            try:
                task = loop.create_task(job)
            except Exception:
                job.close()
                raise
            self.tasks.add(task)
            task.add_done_callback(self._finished)
            capture_shadow_status(reference, transformed, "scheduled")
        except Exception:
            # No exception value: it could contain private input or configuration.
            logger.debug("[boundary-shadow] accepted=false reason=scheduling_failed")
            capture_shadow_status(reference, transformed, "scheduling_failed")

    async def _run(self, target: str, reference: str, transformed: str) -> None:
        cancelled = threading.Event()
        # A private single-worker pool never occupies the shared ALASS executor.
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="boundary-shadow")
        try:
            worker = asyncio.get_running_loop().run_in_executor(executor, partial(
                verify_boundary, target, reference, transformed, cancelled=cancelled))
            try:
                result = await asyncio.shield(worker)
            except asyncio.CancelledError:
                capture_shadow_status(reference, transformed, "cancelled")
                cancelled.set()
                # Cancellation must not free the owner's slot while a worker is
                # still running. Repeated cancellation also cannot orphan it.
                while not worker.done():
                    try:
                        await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if not worker.cancelled():
                    worker.exception()
                raise
            a, b = result.segments if len(result.segments) == 2 else (None, None)
            capture_boundary(reference, transformed, result, target)
            logger.info(
                "[boundary-shadow] accepted=%s classification=%s artifact=%s segments=%d "
                "boundary_ms=%s offset_a_ms=%s offset_b_ms=%s jump_ms=%s "
                "support_a=%s support_b=%s residual_p95_a=%s residual_p95_b=%s "
                "reference_p95_a=%s reference_p95_b=%s coverage_a=%s coverage_b=%s "
                "gap_error_ms=%s "
                "holdout_pass=%s reason=%s duration_ms=%.1f",
                str(result.accepted).lower(), result.classification, result.artifact,
                len(result.segments), result.boundary_ms, result.offset_a_ms, result.offset_b_ms,
                result.jump_ms, a.support if a else None, b.support if b else None,
                a.residual_p95_ms if a else None, b.residual_p95_ms if b else None,
                a.reference_start_p95_ms if a else None, b.reference_start_p95_ms if b else None,
                a.forward_coverage if a else None, b.forward_coverage if b else None,
                result.gap_disagreement_ms,
                str(result.holdout_pass).lower(), ",".join(result.reasons) or "supported",
                result.duration_ms,
            )
        except Exception:
            logger.info("[boundary-shadow] accepted=false reason=evaluation_failed")
            capture_shadow_status(reference, transformed, "evaluation_failed")
        finally:
            # The worker has completed (including cancellation drain) before
            # releasing its private thread. No shared ALASS semaphore is touched.
            executor.shutdown(wait=True, cancel_futures=True)

    async def close(self) -> None:
        self._closed = True
        tasks = list(self.tasks)
        for task in tasks:
            if not task.cancelling():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
