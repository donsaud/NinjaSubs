"""Alass candidate budget: scope of the counter, proven per request.

A production Whiplash trace reported::

    alass candidate limit reached (3) -> deferring remaining candidates
    all sync strategies failed -> serving original subtitle

with NO ``[sync] starting alass``, ``pre-alass timing``, ``executing alass`` or
``verifier input`` line for that request. The request stopped before Alass; the
trace does not show Alass failing, it shows the budget already spent.

The defect: ``alass_attempted`` was read from ``self._metrics``, created once in
``__init__`` on an orchestrator that ``main.py`` deliberately keeps as a
process-wide singleton. Nothing reset it, so the allowance was consumed by
whichever requests arrived first and never replenished. Measured before the
fix: 10 sequential requests produced 3 Alass executions in total.

The counter is now request-local. These tests pin that, and they do so by
counting real ``sync_async`` calls -- the Alass execution boundary -- rather
than by inspecting the counter.

Fixture calibration
-------------------
Reaching the Alass call site means satisfying every pre-Alass gate honestly:

* valid SRT with blank-line separated blocks
* ``HH:MM:SS,mmm`` timestamps, first dialogue past the 60s intro strip
* enough consecutive dialogue for ``first_dialogue_cluster``
* cue-sanity delta inside the 20s execution window
* a median offset at or above ``ALIGNED_OFFSET_THRESHOLD_S`` so the
  already-aligned shortcut is NOT taken

Two non-obvious traps, both hit while writing this file:

* Uniform cue spacing plus a shift that is a multiple of that spacing makes
  ``median_cue_offset`` degenerate. Nearest-neighbour matching finds an exact
  hit for every target cue, so the measured offset is 0.0 and the subtitle
  reads as already aligned. The cue times here are deliberately irregular.
* ``first_dialogue_cluster`` needs a run of >= 2 consecutive dialogue cues
  within ``_DIALOGUE_MAX_GAP_MS``; without that it returns ``None`` and
  cue-sanity fails open, which would make every gate pass vacuously.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from app.services.sync.orchestrator import SyncOrchestrator

# Irregular cue starts, first one past the 60s intro-strip window.
_TIMES: list[int] = [95000]
for _i in range(29):
    _TIMES.append(_TIMES[-1] + 1500 + (_i % 5) * 700)

#: Shifts chosen from measured behaviour (see module docstring).
SHIFT_ELIGIBLE = 6000      # sanity ok, median 0.5s -> NOT already aligned
SHIFT_WRONG_CUT = 30000    # cue-sanity TIMING_MISMATCH, rejected before alass
SHIFT_ALIGNED = 0          # already-aligned shortcut, no subprocess

LIMIT = 3


def _ts(ms: int) -> str:
    total_s, millis = divmod(int(ms), 1000)
    m, s = divmod(total_s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"


def _srt(shift: int = 0, *, cues: int = 30) -> str:
    out = []
    for i in range(cues):
        start = _TIMES[i] + shift
        out.append(
            f"{i + 1}\n{_ts(start)} --> {_ts(start + 900)}\n"
            f"We should talk about the second issue number {i} here.\n\n"
        )
    return "".join(out)


TARGET = _srt(0)


class _Strategy:
    """Yields a fixed reference, or nothing when asked to fail resolution."""

    def __init__(self, name: str, reference: str | None):
        self._name = name
        self._reference = reference
        self.validates_target = False
        self.resolved = 0

    async def resolve_with_provenance(self, query, update_validator=None):
        from app.services.sync.query import ResolvedReference

        self.resolved += 1
        if self._reference is None:
            return ResolvedReference(None)
        return ResolvedReference(
            text=self._reference, kind="edition", candidate="ref", bluray_match=False
        )


class _SyncService:
    """The Alass execution boundary. Counts real invocations."""

    def __init__(self, *, gate: asyncio.Event | None = None):
        self.calls = 0
        self._gate = gate

    async def sync_async(self, target, reference, **kwargs):
        if self._gate is not None:
            # Deterministic interleaving: hold inside the alass boundary so a
            # second request is provably concurrent rather than racing by luck.
            self._gate.set()
            await asyncio.sleep(0)
        self.calls += 1
        return None  # force the loop onward; the budget is what is measured


def _meta() -> dict:
    return {
        "imdb_id": "tt0123456",
        "season": 8,
        "episode": 5,
        "media_type": "series",
        "target_filename": (
            "Whiplash.2014.UHD.BluRay.2160p.TrueHD.Atmos.7.1.DV.HEVC.HYBRID.REMUX-FaMeSToR.mkv"
        ),
        "video_size": 5192387053,
        "lang": "ara",
    }


def _orchestrator(shifts: list[int], *, limit: int = LIMIT, svc=None) -> tuple[SyncOrchestrator, _SyncService]:
    """One long-lived orchestrator whose strategies return `shifts` in order."""
    service = svc or _SyncService()
    orch = SyncOrchestrator(sync_service=service, sync_cache=None)
    strategies = [_Strategy(f"s{i}", _srt(sh)) for i, sh in enumerate(shifts)]
    orch._strategies = lambda: [(s._name, s) for s in strategies]  # type: ignore[method-assign]
    orch._alass_candidate_limit = limit
    orch.__dict__["_test_strategies"] = strategies
    return orch, service


async def _run(orch: SyncOrchestrator, target_id: str) -> bytes:
    return await orch.evaluate_and_sync(
        TARGET.encode("utf-8"), _meta(), target_id, auto_sync=True
    )


def _eligible(count: int) -> list[int]:
    """`count` distinct shifts that all pass every pre-Alass gate."""
    return [SHIFT_ELIGIBLE + i * 100 for i in range(count)]


# --- fixture self-check: the gates are honestly satisfied -------------------


def test_fixture_reaches_the_alass_boundary():
    """Guard against silently regressing into a vacuous fixture.

    If this fails, every other test in the file is measuring the wrong path.
    """
    from app.services.subtitle_matcher import (
        ALIGNED_OFFSET_THRESHOLD_S,
        FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
        first_dialogue_cluster,
        median_cue_offset,
        parse_srt_cues,
        validate_cue_sanity,
    )

    assert len(parse_srt_cues(TARGET)) == 30
    assert first_dialogue_cluster(TARGET) is not None
    ref = _srt(SHIFT_ELIGIBLE)
    offset = median_cue_offset(TARGET, ref)
    assert offset is not None and abs(offset) >= ALIGNED_OFFSET_THRESHOLD_S, (
        "fixture must not take the already-aligned shortcut"
    )
    verdict = validate_cue_sanity(
        TARGET, ref, threshold_ms=FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS
    )
    assert verdict["ok"], f"eligible reference must pass cue sanity: {verdict}"


def test_fixture_wrong_cut_is_rejected_before_alass():
    from app.services.subtitle_matcher import (
        FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
        validate_cue_sanity,
    )

    verdict = validate_cue_sanity(
        TARGET, _srt(SHIFT_WRONG_CUT),
        threshold_ms=FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
    )
    assert not verdict["ok"], "the wrong-cut reference must be refused"


# --- Test A/B: budget scope ------------------------------------------------


@pytest.mark.asyncio
async def test_a_budget_is_per_request():
    """Two requests, one orchestrator: each gets the full allowance."""
    orch, svc = _orchestrator(_eligible(3))

    await _run(orch, "sub-a")
    first = svc.calls
    assert first == LIMIT, f"request 1 should use its allowance, got {first}"

    await _run(orch, "sub-b")
    assert svc.calls == first + LIMIT, (
        f"request 2 inherited request 1's consumption: {svc.calls} total "
        f"instead of {2 * LIMIT}"
    )


@pytest.mark.asyncio
async def test_b_sequential_requests_do_not_share_budget():
    """The Whiplash shape: a long-lived process must not trend to no-sync."""
    orch, svc = _orchestrator(_eligible(1))

    for i in range(10):
        await _run(orch, f"sub-{i}")

    assert svc.calls == 10, (
        f"only {svc.calls} of 10 requests reached alass; the budget leaks "
        "across requests"
    )


# --- Test C: concurrency ---------------------------------------------------


@pytest.mark.asyncio
async def test_c_concurrent_requests_do_not_share_budget():
    """Two overlapping requests each keep their own local allowance.

    A shared counter would let the first request's consumption starve the
    second. The gate makes the overlap deterministic instead of timing luck.
    """
    reached = asyncio.Event()
    svc = _SyncService(gate=reached)
    orch, _ = _orchestrator(_eligible(2), svc=svc)

    first = asyncio.create_task(_run(orch, "sub-a"))
    await reached.wait()
    second = asyncio.create_task(_run(orch, "sub-b"))
    await asyncio.gather(first, second)

    assert svc.calls == 4, (
        f"concurrent requests shared a budget: {svc.calls} executions instead of 4"
    )


# --- Test D: pre-Alass rejection does not consume the budget ---------------


@pytest.mark.asyncio
async def test_d_pre_alass_rejection_does_not_consume_budget():
    """A wrong-cut reference is refused without spending an alass slot.

    The budget exists to bound subprocesses, so a candidate that never reaches
    one must not consume it. The safety gate is unchanged; only the accounting
    is asserted.
    """
    orch, svc = _orchestrator([SHIFT_WRONG_CUT] + _eligible(2))

    await _run(orch, "sub")

    assert svc.calls == 2, (
        f"expected the two eligible references to run, got {svc.calls} alass calls"
    )
    strategies = orch.__dict__["_test_strategies"]
    assert strategies[0].resolved == 1, "the wrong-cut reference was still evaluated"
    assert strategies[1].resolved == 1, "and it was refused, not skipped silently"


# --- Test E: exactly the allowance runs -------------------------------------


@pytest.mark.asyncio
async def test_e_exactly_three_executions_per_request():
    """Four eligible candidates: three run, the fourth is deferred."""
    orch, svc = _orchestrator(_eligible(4))

    await _run(orch, "sub")

    assert svc.calls == LIMIT
    strategies = orch.__dict__["_test_strategies"]
    assert strategies[3].resolved == 1, (
        "the fourth candidate is resolved before the budget is consulted, "
        "because the already-aligned shortcut needs a reference; it is then "
        "deferred without an alass execution"
    )
    assert orch.metrics["alass_runs"] == LIMIT, (
        "the process-wide counter tracks real executions for observability"
    )


# --- Test F: diagnostics ----------------------------------------------------


@pytest.mark.asyncio
async def test_f_budget_diagnostic_is_accurate(caplog):
    """One trace must answer why the limit was reached."""
    orch, svc = _orchestrator([SHIFT_WRONG_CUT] + _eligible(4))

    with caplog.at_level(logging.INFO, logger="app.services.sync.orchestrator"):
        await _run(orch, "sub")

    budget = [r.getMessage() for r in caplog.records if "[sync-budget]" in r.getMessage()]
    assert budget, "budget accounting must be observable"

    deferrals = [m for m in budget if "reason=budget_exhausted" in m]
    assert deferrals, "the deferred candidate must be reported"
    final = deferrals[-1]
    assert f"limit={LIMIT}" in final
    assert "attempted=3" in final
    assert "alass_started=3" in final
    assert "alass_completed=0" in final, (
        "completed must reflect the mock, which returns no output"
    )
    assert "rejected_before_alass=1" in final, (
        "the wrong-cut reference must be counted as a pre-alass rejection"
    )
    assert "deferred=1" in final
    assert "candidate=" in final

    rejections = [m for m in budget if "reason=cue_sanity_rejected" in m]
    assert rejections, "a cue-sanity rejection must be reported separately"
    assert "counted_toward_limit=false" in rejections[0], (
        "a pre-alass rejection must be recorded as not consuming the budget"
    )

    for message in budget:
        for forbidden in ("We should talk", "http", "api_key", "Bearer"):
            assert forbidden not in message


# --- Test G: Whiplash regression -------------------------------------------


@pytest.mark.asyncio
async def test_g_whiplash_regression_process_wide_orchestrator():
    """The production failure mode, end to end.

    Under the old implementation the first few requests consumed the allowance
    process-wide and every later request deferred immediately. Under the fix
    every request starts with a fresh local budget.
    """
    orch, svc = _orchestrator(_eligible(2))

    results = []
    for i in range(6):
        results.append(svc.calls)
        await _run(orch, f"sub-{i}")

    assert svc.calls == 12, (
        f"6 requests x 2 attempts should be 12 executions, saw {svc.calls}. "
        "A process-wide budget would have produced 2 in total."
    )
    assert all(later > earlier for earlier, later in zip(results, results[1:], strict=False)), (
        "no request may be starved by an earlier one"
    )


# --- already-aligned is preserved (Part 8) ----------------------------------


@pytest.mark.asyncio
async def test_already_aligned_shortcut_is_not_budgeted_or_blocked():
    """An already-aligned subtitle serves with no subprocess and no slot.

    The aligned candidate is scheduled AFTER the three that spend the
    allowance. It must still be evaluated and served rather than starved,
    because the already-aligned check runs before the budget gate by design:
    that path costs no subprocess, so a budget that exists to bound subprocesses
    has no reason to block it.
    """
    orch, svc = _orchestrator([SHIFT_ELIGIBLE] * 3 + [SHIFT_ALIGNED])

    out = await _run(orch, "sub")

    assert svc.calls == LIMIT, (
        f"the three eligible candidates ran; the aligned one must not add to it "
        f"(saw {svc.calls})"
    )
    assert out, "the original subtitle was served"
    assert orch.__dict__["_test_strategies"][3].resolved == 1, (
        "the aligned candidate was still evaluated, not skipped by the budget"
    )


@pytest.mark.asyncio
async def test_deterministic_candidate_order():
    """Identical inputs produce an identical attempt sequence."""
    orders = []
    for _ in range(2):
        orch, svc = _orchestrator(_eligible(4))
        await _run(orch, "sub")
        orders.append(svc.calls)
    assert orders[0] == orders[1] == LIMIT
