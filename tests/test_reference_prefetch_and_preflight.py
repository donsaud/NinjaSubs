"""Progressive reference preparation, lightweight preflight, and guarded
Large-Offset replay.

These tests use only local fixtures and fakes: no provider request, no ALASS
process, and no network. The Into the Wild fixtures are the exact cached
references from the confirmed production case (Arabic FSiHD and English FSiHD
against the WEBRip Netflix target).
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

from app.services.sync import orchestrator as om
from app.services.sync.orchestrator import _PlannedPrefetcher
from app.services.sync.query import ReferenceQuery, ResolvedReference
from app.services.sync.reference_preflight import (
    measure_reference_timing,
    sibling_is_materially_tighter,
)
from app.services.sync.telemetry import SyncTiming, current_sync_timing

FIXTURES = Path(__file__).parent / "fixtures" / "into_the_wild"
_FIXTURE_NAMES = ("target.srt", "reference_ara_fsihd.srt", "reference_eng_fsihd.srt")
REAL_FIXTURES_AVAILABLE = all((FIXTURES / name).exists() for name in _FIXTURE_NAMES)
if REAL_FIXTURES_AVAILABLE:
    TARGET = (FIXTURES / "target.srt").read_text("utf-8")
    ARA = (FIXTURES / "reference_ara_fsihd.srt").read_text("utf-8")
    ENG = (FIXTURES / "reference_eng_fsihd.srt").read_text("utf-8")
else:  # real subtitle fixtures are deliberately gitignored; keep mechanics tests runnable
    TARGET = "1\n00:00:01,000 --> 00:00:02,000\nsynthetic target\n"
    ARA = "1\n00:00:02,000 --> 00:00:03,000\nsynthetic arabic reference\n"
    ENG = "1\n00:00:03,000 --> 00:00:04,000\nsynthetic english reference\n"
FAMILY = "BluRay||fsihd"
requires_real_fixtures = pytest.mark.skipif(
    not REAL_FIXTURES_AVAILABLE,
    reason="private Into the Wild subtitle fixtures are not present",
)


class _PlannedStrategy:
    """Minimal planned strategy for exercising the orchestrator generator."""

    cache = None
    validates_target = True

    def __init__(self, payloads: list[str]) -> None:
        self.payloads = payloads
        self.started: list[int] = []
        self._plan = [
            (_release(f"Candidate.{i + 1}.1080p.BluRay.x264-FSiHD.srt"), "subdl")
            for i in range(len(payloads))
        ]

    async def plan_candidates(self, query):
        return list(self._plan)

    async def download_planned_candidate(
        self, release, provider, query, validate, shared, plan_index=None
    ):  # noqa: ANN001
        index = int(plan_index or 1) - 1
        self.started.append(index)
        return ResolvedReference(
            self.payloads[index], kind="edition", candidate=f"candidate-{index + 1}.srt"
        )


class _FakeStrategy:
    """Downloads prepared candidates without any network or provider state."""

    def __init__(self, payloads: list[str | None], delay: float = 0.0) -> None:
        self.payloads = payloads
        self.delay = delay
        self.started: list[int] = []
        self.finished: list[int] = []

    async def download_planned_candidate(
        self, release, provider, query, validate, shared, plan_index=None
    ):  # noqa: ANN001
        index = int(plan_index or 1) - 1
        self.started.append(index)
        if self.delay:
            await asyncio.sleep(self.delay)
        payload = self.payloads[index]
        self.finished.append(index)
        if payload is None:
            return ResolvedReference(None, reference_failure="download_failure")
        return ResolvedReference(
            payload, kind="edition", candidate=f"candidate-{index + 1}.srt"
        )


def _release(name: str):
    from types import SimpleNamespace

    return SimpleNamespace(release_name=name, download_url=f"https://invalid/{name}")


def _prefetcher(strategy: _FakeStrategy, *, concurrency: int = 2, lookahead: int = 2):
    planned = [
        (_release(f"Candidate.{i + 1}.1080p.BluRay.x264-FSiHD.srt"), "subdl")
        for i in range(len(strategy.payloads))
    ]
    return _PlannedPrefetcher(
        strategy=strategy,
        planned=planned,
        query=ReferenceQuery(imdb_id="tt0758758"),
        shared={"downloaded": 0, "seen": set()},
        validate=lambda text: True,
        concurrency=concurrency,
        lookahead=lookahead,
    )


@pytest.mark.asyncio
async def test_prefetch_prepares_next_while_current_is_awaited():
    """The successor is already downloaded when the loop asks for it."""
    strategy = _FakeStrategy(["one", "two", "three"], delay=0.05)
    p = _prefetcher(strategy)
    p.arm_ahead(0, [1, 2])
    timing = SyncTiming()
    token = current_sync_timing.set(timing)
    try:
        first = await p.prepare(0)
        assert first.text == "one"
        # The bounded window (current + one ahead) was scheduled immediately.
        assert 1 in strategy.started
        second = await p.prepare(1)
        assert second.text == "two"
        assert timing.prefetch_reused >= 1, "the successor should already be decoded"
        # Current + two-ahead window at most; never the whole pool.
        assert timing.prefetch_started <= 3, "lookahead must stay bounded"
    finally:
        await p.close()
        current_sync_timing.reset(token)
    assert not any(task for task in p._tasks.values() if not task.done())


@pytest.mark.asyncio
async def test_prefetch_stays_off_for_an_ordinary_candidate():
    """One verifying candidate must still cost exactly one download."""
    strategy = _FakeStrategy(["one", "two", "three"], delay=0.0)
    p = _prefetcher(strategy)
    timing = SyncTiming()
    token = current_sync_timing.set(timing)
    try:
        assert (await p.prepare(0)).text == "one"
        assert strategy.started == [0]
        assert not p.armed
        assert timing.prefetch_started == 1
    finally:
        await p.close()
        current_sync_timing.reset(token)


@pytest.mark.asyncio
async def test_prefetch_auto_advances_after_a_deterministic_duplicate_but_stays_bounded():
    """A duplicate is skipped by the loop, so the next candidate is warmed."""
    strategy = _FakeStrategy(["one", None, "three", "four"])
    p = _prefetcher(strategy)
    p.arm_ahead(0, [1, 2])
    timing = SyncTiming()
    token = current_sync_timing.set(timing)
    try:
        first = await p.prepare(0)
        assert first.text == "one"
        # Task 1 returns None, which auto-schedules task 2. Task 3 must remain
        # unscheduled: only two unconsumed candidates are ever prepared.
        for _ in range(50):
            await asyncio.sleep(0.01)
            if 2 in p._tasks and p._tasks[2].done():
                break
        assert 2 in p._tasks
        assert 3 not in p._tasks
        assert (await p.prepare(1)).text is None
        assert (await p.prepare(2)).text == "three"
        # Candidate 3 was consumed; the window may now prepare candidate 4.
        assert 3 in p._tasks or timing.prefetch_started <= 3
    finally:
        await p.close()
        current_sync_timing.reset(token)


@pytest.mark.asyncio
async def test_prefetch_cancels_unused_downloads_on_success():
    """Once a result is served, speculative work must not keep running."""
    strategy = _FakeStrategy(["one", "two", "three"], delay=0.5)
    p = _prefetcher(strategy, concurrency=1)
    p.arm_ahead(0, [1])
    timing = SyncTiming()
    token = current_sync_timing.set(timing)
    try:
        assert (await p.prepare(0)).text == "one"
        assert 1 in strategy.started
    finally:
        await p.close()
        current_sync_timing.reset(token)
    assert all(task.done() for task in p._tasks.values())
    assert any(task.cancelled() for task in p._tasks.values())
    assert timing.prefetch_cancelled >= 1


@requires_real_fixtures


def test_into_the_wild_preflight_prefers_the_tighter_same_family_reference():
    """Existing gates, not a new verifier, select the proven FSiHD reference."""
    ara = measure_reference_timing(TARGET, ARA, family=FAMILY)
    eng = measure_reference_timing(TARGET, ENG, family=FAMILY)
    assert ara is not None and eng is not None
    assert ara.accepted and eng.accepted
    assert ara.large_offset and eng.large_offset
    # Both independently recover the same ~-23s displacement.
    assert abs((ara.offset_ms or 0) - (eng.offset_ms or 0)) <= 750
    # The English release has the materially tighter anchor dispersion that
    # made it the proven winner; the Arabic one is not excluded.
    assert sibling_is_materially_tighter(ara, eng)
    assert not sibling_is_materially_tighter(eng, ara)


@requires_real_fixtures


def test_preflight_rejects_a_different_release_family():
    ara = measure_reference_timing(TARGET, ARA, family="BluRay||fsihd")
    eng = measure_reference_timing(TARGET, ENG, family="WEBRip||other")
    assert not sibling_is_materially_tighter(ara, eng)


def test_preflight_requires_agreeing_offsets():
    """A tighter reference that proposes a different correction is not preferred."""
    from app.services.sync.reference_preflight import ReferenceTimingEvidence

    current = ReferenceTimingEvidence(
        family=FAMILY, large_offset=True, accepted=True, offset_ms=-23_000,
        dispersion_ms=800.0, drift_ms_per_minute=10.0,
    )
    sibling = ReferenceTimingEvidence(
        family=FAMILY, large_offset=True, accepted=True, offset_ms=-11_000,
        dispersion_ms=50.0, drift_ms_per_minute=1.0,
    )
    assert not sibling_is_materially_tighter(current, sibling)


def _fake_evidence(dispersion_ms: float = 800.0):
    from app.services.sync.reference_preflight import ReferenceTimingEvidence

    return ReferenceTimingEvidence(
        family=FAMILY,
        large_offset=True,
        accepted=True,
        offset_ms=-23_000.0,
        dispersion_ms=dispersion_ms,
        drift_ms_per_minute=1.0,
        structural_similarity=0.85,
    )


@pytest.mark.asyncio
async def test_orchestrator_prefers_the_tighter_sibling_but_keeps_it_in_plan(monkeypatch):
    """The preflight decision is a reorder, never an acceptance or exclusion."""
    monkeypatch.setattr(
        "app.services.sync.reference_preflight.measure_reference_timing",
        lambda *args, **kwargs: _fake_evidence(),
    )
    monkeypatch.setattr(
        "app.services.sync.reference_preflight.sibling_is_materially_tighter",
        lambda *args, **kwargs: True,
    )
    strategy = _FakeStrategy(["first", "second"])
    p = _prefetcher(strategy)
    p.arm_ahead(0, [1])
    timing = SyncTiming()
    token = current_sync_timing.set(timing)
    orchestrator = om.SyncOrchestrator()
    query = ReferenceQuery(imdb_id="tt0758758", target_text=TARGET)
    try:
        assert (await p.prepare(0)).text == "first"
        preferred = await orchestrator._preferred_sibling(
            current_index=0,
            current_text="first",
            pending=[1],
            prefetcher=p,
            query=query,
            timing=timing,
        )
        assert preferred == 1
        assert timing.preflight_runs >= 1
        assert p.peek(1)[1].text == "second"
    finally:
        await p.close()
        current_sync_timing.reset(token)


@pytest.mark.asyncio
async def test_preflight_reorder_keeps_the_deferred_candidate_next(monkeypatch):
    """The slower same-family reference is deferred, not dropped."""
    monkeypatch.setattr(
        "app.services.sync.reference_preflight.measure_reference_timing",
        lambda *args, **kwargs: _fake_evidence(),
    )
    monkeypatch.setattr(
        "app.services.sync.reference_preflight.sibling_is_materially_tighter",
        lambda *args, **kwargs: True,
    )
    strategy = _PlannedStrategy(["first", "second"])
    orchestrator = om.SyncOrchestrator()
    timing = SyncTiming()
    token = current_sync_timing.set(timing)
    query = ReferenceQuery(
        imdb_id="tt0758758",
        target_text=TARGET,
        target_filename="Into.The.Wild.2007.1080p.BluRay.x264-FSiHD",
    )
    generator = orchestrator._reference_attempts(
        [("external-release-reference", strategy)], query, lambda text: True, timing
    )
    try:
        first_name, first_obj = await generator.__anext__()
        first = await first_obj.resolve_with_provenance(query)
        second_name, second_obj = await generator.__anext__()
        second = await second_obj.resolve_with_provenance(query)
        assert first.text == "second"
        assert second.text == "first"
        # Labels keep the planner's positions, so telemetry still identifies
        # which planned release was executed.
        assert first_name.endswith("#candidate2")
        assert second_name.endswith("#candidate1")
    finally:
        await generator.aclose()
        current_sync_timing.reset(token)


@pytest.mark.asyncio
async def test_prefetch_spends_the_speculative_slot_on_a_distinct_release(monkeypatch):
    """A mirrored sibling is deferred, not speculatively downloaded twice."""
    from types import SimpleNamespace

    monkeypatch.setattr(
        "app.services.sync.reference_preflight.measure_reference_timing",
        lambda *args, **kwargs: _fake_evidence(),
    )
    releases = [
        SimpleNamespace(
            release_name="Into.The.Wild.2007.1080p.BluRay.x264-FSiHD.srt",
            download_url="https://invalid/subdl",
            lang="ara",
        ),
        SimpleNamespace(
            release_name="Into.The.Wild.2007.1080p.BluRay.x264-FSiHD.srt",
            download_url="https://invalid/subsource",
            lang="ara",
        ),
        SimpleNamespace(
            release_name="Into.The.Wild.2007.1080p.BluRay.x264-FSiHD.ENG.srt",
            download_url="https://invalid/subdl-eng",
            lang="eng",
        ),
    ]
    strategy = _FakeStrategy(["first", "mirror", "distinct"])
    p = _PlannedPrefetcher(
        strategy=strategy,
        planned=[(release, "subdl") for release in releases],
        query=ReferenceQuery(imdb_id="tt0758758", target_text=TARGET),
        shared={"downloaded": 0, "seen": set()},
        validate=lambda text: True,
        concurrency=2,
        lookahead=2,
    )
    timing = SyncTiming()
    token = current_sync_timing.set(timing)
    try:
        assert (await p.prepare(0)).text == "first"
        orchestrator = om.SyncOrchestrator()
        orchestrator._maybe_arm_prefetch(
            current_index=0,
            current_text="first",
            pending=[1, 2],
            prefetcher=p,
            query=ReferenceQuery(imdb_id="tt0758758", target_text=TARGET),
            timing=timing,
        )
        assert p.armed
        assert p._ahead == [2], "the mirrored sibling must not be prefetched"
        assert 1 not in p._tasks
        assert 2 in p._tasks
        # It remains reachable on demand if the preferred candidates fail.
        assert (await p.prepare(1)).text == "mirror"
        assert 1 in strategy.started
    finally:
        await p.close()
        current_sync_timing.reset(token)


class _PayloadCache:
    is_ephemeral = True

    def __init__(self, payload: bytes, meta: dict) -> None:
        self.payload = payload
        self.meta = meta
        self.set_calls = 0

    async def get(self, key):
        return self.payload

    async def get_meta(self, key):
        return self.meta

    async def is_failed(self, key):
        return False

    async def get_verdict(self, key):
        return None

    async def set(self, key, value):
        self.set_calls += 1

    async def set_meta(self, key, value):
        pass

    async def set_verdict(self, key, value, **kwargs):
        pass

    async def mark_failed(self, key, ttl=None):
        pass

    async def clear_failed(self, key):
        pass

    async def find_synced_for(self, *args, **kwargs):
        return None

    async def find_synced_for_target(self, *args, **kwargs):
        return None


def _into_the_wild_meta() -> dict:
    return {
        "imdb_id": "tt0758758",
        "media_type": "movie",
        "target_filename": "Into.The.Wild.2007.1080p.BluRay.x264-FSiHD",
        "video_size": "13150936000",
        "lang": "ara",
        "release_name": "Into The Wild.WEBRip.Netflix.srt",
        "sub_id": "d8efd4b030f64a1d",
        "year": 2007,
    }


def _corrected_meta(payload: bytes) -> dict:
    return {
        "sync_state": "unverified",
        "verification": "unknown",
        "serving_state": "alass_corrected_large_offset",
        "large_offset_authorization": om.LARGE_OFFSET_CACHE_AUTHORIZATION_VERSION,
        "target_binding": "strong",
        "target_binding_reasons": ["release_group_match", "source_compatible"],
        "reference_sha": hashlib.sha256(ENG.encode("utf-8")).hexdigest(),
        "artifact_sha256": hashlib.sha256(payload).hexdigest(),
    }


@pytest.mark.asyncio
async def test_guarded_large_offset_cache_replays_the_approved_bytes(monkeypatch):
    """Warm request: the confirmed corrected artifact is served, still UNVERIFIED."""
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True, raising=False)
    payload = b"1\n00:00:30,000 --> 00:00:32,000\ncorrected\n"
    cache = _PayloadCache(payload, _corrected_meta(payload))
    strategy = _FakeStrategy([])
    orchestrator = om.SyncOrchestrator(external_strategy=strategy, sync_cache=cache)
    out = await orchestrator.evaluate_and_sync(
        TARGET.encode("utf-8"), _into_the_wild_meta(), "d8efd4b030f64a1d", auto_sync=True
    )
    assert out == payload
    assert orchestrator._last_evaluation is not None
    assert orchestrator._last_evaluation.sync_state.value == "unverified"
    assert orchestrator._last_evaluation.verification.value == "unknown"
    assert orchestrator.metrics["large_offset_cache_hits"] == 1
    assert strategy.started == [], "a guarded replay must not touch a provider"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("large_offset_authorization", None),
        ("serving_state", "original"),
        ("target_binding", "weak"),
        ("artifact_sha256", "0" * 64),
        ("reference_sha", "short"),
        ("sync_state", "verified_synced"),
    ],
)
async def test_guarded_large_offset_cache_refuses_an_incomplete_or_different_decision(
    monkeypatch, field, value
):
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True, raising=False)
    payload = b"corrected bytes"
    meta = _corrected_meta(payload)
    meta[field] = value
    cache = _PayloadCache(payload, meta)
    orchestrator = om.SyncOrchestrator(
        external_strategy=_FakeStrategy([]), sync_cache=cache
    )
    out = await orchestrator.evaluate_and_sync(
        TARGET.encode("utf-8"), _into_the_wild_meta(), "d8efd4b030f64a1d", auto_sync=True
    )
    assert out == TARGET.encode("utf-8"), "an unproven cache entry must fall through"
    assert orchestrator.metrics["large_offset_cache_hits"] == 0


@pytest.mark.asyncio
async def test_guarded_large_offset_cache_is_disabled_under_unified_authority(
    monkeypatch,
):
    """The Large-Offset override is fenced off under Unified authority."""
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True, raising=False)
    monkeypatch.setattr(om, "unified_authority_enabled", lambda meta: True)
    payload = b"corrected bytes"
    cache = _PayloadCache(payload, _corrected_meta(payload))
    orchestrator = om.SyncOrchestrator(
        external_strategy=_FakeStrategy([]), sync_cache=cache
    )
    out = await orchestrator.evaluate_and_sync(
        TARGET.encode("utf-8"), _into_the_wild_meta(), "d8efd4b030f64a1d", auto_sync=True
    )
    assert out == TARGET.encode("utf-8")
    assert orchestrator.metrics["large_offset_cache_hits"] == 0
