"""P0/P1/P2 release regressions; synthetic evidence, no provider requests."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.models import SubtitleRelease
from app.services.sync import hybrid_router as router
from app.services.sync import orchestrator as om
from app.services.sync.cache import ReferenceDiskCache
from app.services.sync.external_strategy import ExternalExactStrategy
from app.services.sync.query import ReferenceQuery, ResolvedReference
from app.services.sync.target_binding import classify_target_binding
from app.services.sync.telemetry import SyncTiming
from app.services.sync_cache import SyncCache
from two_point_sync.parser import Cue, format_srt

_LONG_TEXT = "The lighthouse keeper arrives at dawn and waits for the tide"


def _srt(count: int = 200, step_ms: int = 2_000, *, start: int = 0) -> str:
    cues = [
        Cue(number=index + 1, order=index + 1,
            start_ms=start + index * step_ms, end_ms=start + index * step_ms + 1_500,
            text=f"{_LONG_TEXT} number {index + 1}")
        for index in range(count)
    ]
    return format_srt(cues)


def _plan_result(name: str, score: float, *, source="match", ready=True, unresolved=False):
    candidate = SimpleNamespace(identity=(name,), language="eng",
                                unresolved_pack=unresolved)
    evidence = [SimpleNamespace(field="source",
                                state=SimpleNamespace(value=source),
                                relationship="same")]
    if unresolved:
        evidence.append(SimpleNamespace(field="season",
                                       state=SimpleNamespace(value="match"),
                                       relationship="same"))
    return SimpleNamespace(
        candidate=candidate,
        ranking_score=score, score=score, percentage=score,
        source_priority_points=0, score_without_hash=score, ranking_percentage=score,
        discovery_eligible=True, ready_reference_eligible=ready,
        rejection_reasons=[], readiness_reasons=[],
        evidence=evidence,
    )




def _evidence(**overrides):
    return {"sync_state": "verified_resynced", "verification": "verified",
            "target_binding": "strong", **overrides}


@pytest.mark.parametrize("episode", [1, 2])
def test_dexter_candidate_reuse_never_becomes_trust_without_verification(tmp_path, episode):
    query = ReferenceQuery(
        imdb_id="tt0773262", media_type="series", season=8, episode=episode,
        target_filename=f"Dexter.S08E{episode:02}.1080p.BluRay.x265-d3g.mkv",
        target_text=_srt().replace("lighthouse", "harbour"),
    )
    cache = ReferenceDiskCache(tmp_path, min_bytes=100)
    strategy = ExternalExactStrategy(cache=cache, min_bytes=100)
    result = ResolvedReference(
        _srt(), candidate=f"Dexter.S08E{episode:02}.BluRay-d3g.srt",
        provider="subdl", kind="edition", member_season=8, member_episode=episode,
    )
    strategy._remember_video_reference(query, result)
    assert cache.get_video_scoped(replace(query, target_sub_id="other")) is not None
    assert cache.get_sticky_reference(query) is None
    assert cache.set_sticky_reference(
        query, provider="subdl", text=result.text, member_season=8,
        member_episode=episode, evidence=_evidence())
    assert cache.get_sticky_reference(query) is not None


@pytest.mark.parametrize("evidence", [
    {}, _evidence(sync_state="unverified"), _evidence(verification="unknown"),
    _evidence(target_binding="weak"), _evidence(target_binding="unknown"),
    _evidence(target_binding="conflict"),
])
def test_unverified_or_insufficiently_bound_preference_refused(tmp_path, evidence):
    query = ReferenceQuery(imdb_id="tt0758758", media_type="movie")
    cache = ReferenceDiskCache(tmp_path, min_bytes=100)
    assert not cache.set_sticky_reference(
        query, provider="subdl", text=_srt(), evidence=evidence)
    assert cache.get_sticky_reference(query) is None
    assert not list(tmp_path.iterdir())


def test_self_reference_never_retained_or_promoted(tmp_path):
    query = ReferenceQuery(imdb_id="tt0758758", media_type="movie", target_text=_srt())
    cache = ReferenceDiskCache(tmp_path, min_bytes=100)
    strategy = ExternalExactStrategy(cache=cache, min_bytes=100)
    variant = "\ufeff" + query.target_text.replace("\n", "\r\n")
    strategy._remember_video_reference(query, ResolvedReference(variant, provider="subdl"))
    assert cache.get_video_scoped(query) is None
    assert not cache.set_sticky_reference(
        query, provider="subdl", text=variant, evidence=_evidence())


def test_old_unverified_preference_is_not_reused(tmp_path):
    import json

    query = ReferenceQuery(imdb_id="tt0758758", media_type="movie")
    cache = ReferenceDiskCache(tmp_path, min_bytes=100)
    assert cache.set_sticky_reference(query, provider="subdl", text=_srt(), evidence=_evidence())
    pointer = tmp_path / f"{query.video_scope_stem}.sticky.json"
    record = json.loads(pointer.read_text())
    record["evidence"]["sync_state"] = "unverified"
    pointer.write_text(json.dumps(record))
    assert cache.get_sticky_reference(query) is None
    # Existing bytes remain available only as a candidate, requiring fresh verification.
    assert cache.get_video_scoped(query) is not None


def test_preference_refuses_partial_and_self_reuse(tmp_path):
    query = ReferenceQuery(imdb_id="tt0758758", media_type="movie")
    cache = ReferenceDiskCache(tmp_path, min_bytes=100)
    assert not cache.set_sticky_reference(
        query, provider="subdl", text=_srt(), partial=True, evidence=_evidence())
    assert cache.set_sticky_reference(query, provider="subdl", text=_srt(), evidence=_evidence())
    assert cache.get_sticky_reference(replace(query, target_text=_srt())) is None


def test_guarded_legacy_delivery_does_not_promote(tmp_path, monkeypatch):
    from app.services.sync.alignment import SubtitleEvaluation

    monkeypatch.setattr(router, "_enabled", False)
    cache = ReferenceDiskCache(tmp_path, min_bytes=100)
    orch = om.SyncOrchestrator(sync_cache=SyncCache(), external_strategy=SimpleNamespace(cache=cache))
    evaluation = SubtitleEvaluation()
    evaluation.target_binding = "strong"
    orch._promote_sticky_reference(
        meta={"imdb_id": "tt0758758", "media_type": "movie"}, target_id="sub",
        pristine_text=_srt().replace("lighthouse", "harbour"),
        resolved=ResolvedReference(_srt(), provider="subdl"), evaluation=evaluation,
        serving_state="alass_corrected_large_offset")
    assert not list(tmp_path.iterdir())


async def test_cached_packaging_variant_does_not_repeat_provider_alignment(monkeypatch):
    query = ReferenceQuery(imdb_id="tt0773262", media_type="series", season=8, episode=2)
    reference = _srt()
    cached = ResolvedReference(reference, provider="subdl")
    strategy = SimpleNamespace(
        cache=SimpleNamespace(get=lambda q: None, get_video_scoped=lambda q: (cached,)),
        plan_candidates=AsyncMock(return_value=[(object(), object())]),
    )
    observed = []

    async def download(_candidate, _provider, _query, _validate, shared):
        from app.services.sync.reference import prepared_reference_digest

        variant = "\ufeff" + reference.replace("\n", "\r\n")
        observed.append(prepared_reference_digest(variant) in shared["seen"])
        return ResolvedReference(None)

    strategy.download_planned_candidate = download
    monkeypatch.setattr(om.settings, "REFERENCE_PREFETCH_ENABLED", False)
    orch = om.SyncOrchestrator()
    async for _, attempt in orch._reference_attempts(
            [("external", strategy)], query, lambda text: True, SyncTiming()):
        await attempt.resolve_with_provenance(query)
    assert observed == [True]


def test_into_the_wild_alternative_services_get_slots_without_binding_upgrade(monkeypatch):
    query = ReferenceQuery(
        imdb_id="tt0758758", media_type="movie",
        target_filename="Into.the.Wild.2007.1080p.AMZN.WEB-DL.x264-GPRS.mkv")
    names = ["iTunes.WEB-DL.x264-R1", "iTunes.WEB-DL.x264-R1", "NF.WEB-DL.x264-R2"]
    releases = [SubtitleRelease(
        release_name=f"Into.the.Wild.2007.1080p.{name}.srt", lang="eng", provider="subdl",
        download_url=f"https://provider.invalid/{index}") for index, name in enumerate(names)]
    pairs = [(candidate, _plan_result(str(index), 90-index))
             for index, candidate in enumerate(releases)]
    strategy = ExternalExactStrategy()
    monkeypatch.setattr(type(strategy), "_download_block_reason", lambda *a: None)
    provider = object()
    plan, report = strategy._select_planned_attempts(
        pairs, [], [], {id(r): provider for r in releases}, 2, query)
    assert [r for r, _ in plan] == [releases[0], releases[2]]
    assert report["planned_count"] == 2
    for candidate, _ in plan:
        assert not classify_target_binding(query.target_filename, candidate.release_name).supports_verified


def test_prepared_dedup_keeps_different_dialogue_and_timing_distinct():
    from app.services.sync.reference import prepared_reference_digest

    reference = _srt()
    digest = prepared_reference_digest(reference)
    assert prepared_reference_digest("\ufeff" + reference.replace("\n", "\r\n")) == digest
    assert prepared_reference_digest(reference.replace("lighthouse", "harbour")) != digest
    assert prepared_reference_digest(_srt(start=5_000)) != digest


def test_exploration_reserve_keeps_all_available_download_slots(monkeypatch):
    query = ReferenceQuery(imdb_id="tt1", target_filename="Film.2020.BluRay-GRP.mkv")
    releases = [SubtitleRelease(release_name=f"Film.2020.BluRay-{group}.srt",
                               provider="subdl", lang="eng", download_url=f"/{group}")
                for group in ["A", "B", "C", "PACK"]]
    pairs = [(candidate, _plan_result(str(i), 90-i, ready=i != 3, unresolved=i == 3))
             for i, candidate in enumerate(releases)]
    strategy = ExternalExactStrategy()
    monkeypatch.setattr(type(strategy), "_download_block_reason", lambda *a: None)
    provider = object()
    plan, _ = strategy._select_planned_attempts(
        pairs, [], [], {id(r): provider for r in releases}, 3, query)
    assert len(plan) == 3


async def test_complete_video_candidate_remains_available_after_partial_subtitle_hit():
    partial = ResolvedReference(_srt(count=40), partial=True)
    complete = ResolvedReference(_srt())
    strategy = SimpleNamespace(
        cache=SimpleNamespace(get=lambda q: partial, get_video_scoped=lambda q: (complete,)),
        plan_candidates=AsyncMock(return_value=[]))
    query = ReferenceQuery(imdb_id="tt0773262", media_type="series", season=8, episode=2)
    orch = om.SyncOrchestrator()
    candidates = []
    async for _, attempt in orch._reference_attempts(
            [("external", strategy)], query, lambda text: True, SyncTiming()):
        candidates.append(await attempt.resolve_with_provenance(query))
    assert candidates == [partial, complete]
