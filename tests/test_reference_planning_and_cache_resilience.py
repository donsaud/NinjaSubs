"""Dexter planning, bounded pack exploration, and cache resilience regressions.

The synthetic Dexter PiR8 pool below reproduces the live failure shape: six
ready HDTV references previously outscored two unresolved, source-compatible
BluRay season packs. Source-priority ordering now discovers these packs first;
extraction readiness, bounded planning, safety gates and fallback stay intact.
"""

from __future__ import annotations

import io
import itertools
import json
import zipfile
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.models import SubtitleRelease
from app.services import release_scoring_integration as integration
from app.services.sync.cache import ReferenceDiskCache
from app.services.sync.external_strategy import (
    ExternalExactStrategy,
    _select_candidates_ranked,
    download_source_digest,
)
from app.services.sync.query import ReferenceQuery
from app.services.sync.telemetry import SyncTiming

TARGET = "Dexter.s8e01.A.Beautiful.Day.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv"
READY_NAMES = [
    "Dexter.S08E01.720p.HDTV.x264-EVOLVE.srt",
    "Dexter.S08E01.HDTV.x264-2HD.srt",
    "Dexter.S08E01.HDTV.x264-2HD-HI.srt",
    "Dexter.S08E01.720p.HDTV.x264-EVOLVE-HI.srt",
    "Dexter.S08E01.480p.HDTV.x264-mSD.srt",
    "Dexter.S08E01.480p.HDTV.x264-mSD-HI.srt",
]
FRAMESTOR_PACK = "Dexter.S08.1080p.BluRay.REMUX-FraMeSToR.zip"
ROVERS_PACK = "Dexter.S08.1080p.BluRay.x264-ROVERS.zip"


def _release(name: str, url: str, provider: str = "subdl", lang: str = "eng") -> SubtitleRelease:
    return SubtitleRelease(
        release_name=name,
        download_url=url,
        provider=provider,
        lang=lang,
    )


def _query(**kwargs) -> ReferenceQuery:
    base = {
        "imdb_id": "tt0773262",
        "media_type": "series",
        "season": 8,
        "episode": 1,
        "target_filename": TARGET,
        "languages": ("eng", "ara"),
    }
    base.update(kwargs)
    return ReferenceQuery(**base)


def _dexter_pool() -> list[SubtitleRelease]:
    releases = [
        _release(name, f"https://cdn.example.invalid/hdtv/{index}.srt")
        for index, name in enumerate(READY_NAMES)
    ]
    releases.extend(
        [
            _release(FRAMESTOR_PACK, "https://cdn.example.invalid/packs/framestor.zip"),
            _release(ROVERS_PACK, "https://cdn.example.invalid/packs/rovers.zip"),
        ]
    )
    return releases


def _strategy(pool: list[SubtitleRelease], tmp_path, **providers: Any) -> ExternalExactStrategy:
    subdl = SimpleNamespace(name="subdl", is_breaker_open=lambda: False)
    subsource = SimpleNamespace(name="subsource", is_breaker_open=lambda: False)
    strategy = ExternalExactStrategy(
        subdl_provider=providers.get("subdl", subdl),
        subsource_provider=providers.get("subsource", subsource),
        opensubtitles_provider=providers.get("opensubtitles"),
        cache=ReferenceDiskCache(root=tmp_path / "refs", ttl=3600.0, min_bytes=100),
        min_bytes=100,
    )
    provider_by_name = {
        "subdl": strategy._subdl,
        "subsource": strategy._subsource,
        "opensubtitles": strategy._opensubtitles,
    }
    assert all(release.provider in provider_by_name for release in pool)
    strategy._gather_candidates = AsyncMock(
        return_value=[(release, provider_by_name[release.provider]) for release in pool]
    )
    return strategy


def _plan_report(caplog) -> dict[str, Any]:
    line = next(
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("[reference-plan] ")
    )
    return json.loads(line.partition("[reference-plan] ")[2])


def _srt(tag: str, *, cues: int = 120, start_ms: int = 95_000) -> str:
    def stamp(ms: int) -> str:
        seconds, millis = divmod(int(ms), 1000)
        minutes, secs = divmod(seconds, 60)
        hours, mins = divmod(minutes, 60)
        return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"

    out = []
    for index in range(cues):
        start = start_ms + index * 2_500
        out.append(
            f"{index + 1}\n{stamp(start)} --> {stamp(start + 900)}\n{tag} dialogue {index}\n"
        )
    return "\n".join(out)


def _zip_bytes(members: dict[str, str]) -> bytes:
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w") as archive:
        for name, text in members.items():
            archive.writestr(name, text)
    return raw.getvalue()


@pytest.mark.asyncio
async def test_dexter_source_matching_packs_lead_the_bounded_plan(tmp_path, monkeypatch, caplog):
    """Six ready HDTV releases must not permanently starve a compatible pack."""
    from app.config import settings

    monkeypatch.setattr(settings, "REFERENCE_POOL_LIMIT", 6)
    caplog.set_level("INFO")
    strategy = _strategy(_dexter_pool(), tmp_path)
    plan = await strategy.plan_candidates(_query())
    report = _plan_report(caplog)

    assert len(plan) == 6
    urls = [release.download_url for release, _ in plan]
    pack_urls = {
        "https://cdn.example.invalid/packs/framestor.zip",
        "https://cdn.example.invalid/packs/rovers.zip",
    }
    assert len(set(urls) & pack_urls) == 2
    assert set(urls[:2]) == pack_urls
    assert len(urls) == len(set(urls))
    assert sum(url.startswith("https://cdn.example.invalid/hdtv/") for url in urls) == 4
    assert report["budget_slots"] == 6
    assert report["exploration_slots"] == 0
    assert all(row["role"] == "ordinary" for row in report["planned"])
    assert all(row["unresolved_pack"] and not row["ready_reference_eligible"]
               for row in report["planned"][:2])
    assert all(row["reason"] == "below_pool_limit" for row in report["excluded"])


@pytest.mark.asyncio
async def test_incompatible_pack_does_not_take_an_exploration_slot(tmp_path, monkeypatch):
    """A source-mismatching pack is not a promising video-compatible pack."""
    from app.config import settings

    monkeypatch.setattr(settings, "REFERENCE_POOL_LIMIT", 6)
    pool = [
        _release(name, f"https://cdn.example.invalid/hdtv/{index}.srt")
        for index, name in enumerate(READY_NAMES)
    ]
    pool.append(_release("Dexter.S08.HDTV.x264-GRP.zip", "https://cdn.example.invalid/packs/grp.zip"))
    strategy = _strategy(pool, tmp_path)
    plan = await strategy.plan_candidates(_query())

    assert len(plan) == 6
    assert all(
        release.download_url.startswith("https://cdn.example.invalid/hdtv/")
        for release, _ in plan
    )


@pytest.mark.asyncio
async def test_correct_pack_extraction_promotes_the_exploration_candidate(tmp_path, monkeypatch):
    """Actual extracted metadata—not the archive name—authorizes readiness."""
    strategy = _strategy([], tmp_path)
    provider = SimpleNamespace(name="subdl", is_breaker_open=lambda: False)
    strategy._subdl = provider
    member = "Dexter.S08E01.1080p.BluRay.REMUX-FraMeSToR.srt"
    provider.download_archive = AsyncMock(
        return_value=_zip_bytes(
            {
                member: _srt("REMUX S08E01"),
                "Dexter.S08E02.1080p.BluRay.REMUX-FraMeSToR.srt": _srt("REMUX S08E02"),
            }
        )
    )
    pack = _release(FRAMESTOR_PACK, "https://cdn.example.invalid/packs/framestor.zip")
    monkeypatch.setattr(
        "app.services.sync.external_strategy._REFERENCE_DOWNLOAD_MIN_INTERVAL_SECONDS", 0
    )

    result = await strategy._download_candidate(pack, provider, _query())
    proof = integration.reference_proof(
        pack, _query(), "Dexter.S08E01.1080p.BluRay.REMUX-FraMeSToR.srt"
    )

    assert result.text and "REMUX S08E01" in result.text
    assert result.member_season == 8 and result.member_episode == 1
    assert proof is not None and proof["season"] == 8 and proof["episode"] == 1
    assert proof["score"] > 0


@pytest.mark.asyncio
async def test_wrong_pack_member_fails_closed(tmp_path):
    strategy = _strategy([], tmp_path)
    provider = SimpleNamespace(name="subdl", is_breaker_open=lambda: False)
    strategy._subdl = provider
    provider.download_archive = AsyncMock(
        return_value=_zip_bytes(
            {"Dexter.S08E02.1080p.BluRay.REMUX-FraMeSToR.srt": _srt("REMUX S08E02")}
        )
    )
    pack = _release(FRAMESTOR_PACK, "https://cdn.example.invalid/packs/framestor.zip")

    result = await strategy._download_candidate(pack, provider, _query())

    assert result.text is None
    assert result.reference_failure == "candidate_rejected"


def test_unresolved_pack_receives_no_episode_points():
    query = _query()
    pack = _release(FRAMESTOR_PACK, "https://cdn.example.invalid/packs/framestor.zip")
    (result,) = integration.matches([pack], query)

    assert result.state("episode").value == "unknown"
    episode = next(item for item in result.contributions if item.criterion == "episode")
    assert episode.points == 0
    assert "episode_unresolved" in result.readiness_reasons


@pytest.mark.asyncio
async def test_planning_is_deterministic_for_shuffled_arrivals(tmp_path, monkeypatch):
    """Provider arrival order must not alter global ReferencePolicy order."""
    from app.config import settings

    monkeypatch.setattr(settings, "REFERENCE_POOL_LIMIT", 6)
    pool = _dexter_pool()
    strategy = _strategy(pool, tmp_path)
    expected = [release.download_url for release, _ in await strategy.plan_candidates(_query())]
    checked = 0
    for arrival in itertools.permutations(pool):
        if checked >= 12:
            break
        checked += 1
        strategy._gather_candidates = AsyncMock(
            return_value=[(release, strategy._subdl) for release in arrival]
        )
        assert [release.download_url for release, _ in await strategy.plan_candidates(_query())] == expected


def test_selected_subtitle_release_does_not_change_reference_order(tmp_path):
    """Reference ranking uses the target video, never the chosen subtitle."""
    pool = _dexter_pool()
    first = _query()
    first.target_sub_release_name = "Dexter.S08E01.480p.HDTV.x264-mSD.srt"
    second = _query()
    second.target_sub_release_name = "Dexter.S08E01.1080p.BluRay.REMUX-FraMeSToR.srt"

    assert [id(release) for release in _select_candidates_ranked(pool, first)] == [
        id(release) for release in _select_candidates_ranked(pool, second)
    ]


@pytest.mark.asyncio
async def test_exact_download_source_is_not_planned_twice(tmp_path, monkeypatch, caplog):
    """Pre-download source identity must prevent the same archive from filling slots."""
    from app.config import settings

    monkeypatch.setattr(settings, "REFERENCE_POOL_LIMIT", 6)
    caplog.set_level("INFO")
    first = _release(
        "Dexter.S08E01.720p.HDTV.x264-EVOLVEA.srt",
        "https://cdn.example.invalid/a.zip?token=SECRET",
    )
    second = _release(
        "Dexter.S08E01.720p.HDTV.x264-EVOLVEB.srt",
        "HTTPS://cdn.example.invalid/a.zip#fragment",
    )
    strategy = _strategy([first, second], tmp_path)
    plan = await strategy.plan_candidates(_query())
    report = _plan_report(caplog)

    assert plan == [(plan[0][0], plan[0][1])]
    assert len({download_source_digest("subdl", release.download_url) for release, _ in plan}) == 1
    assert any(row["reason"] == "duplicate_download_source" for row in report["excluded"])
    assert "SECRET" not in caplog.text
    assert download_source_digest("subdl", first.download_url) == download_source_digest(
        "subdl", second.download_url
    )


@pytest.mark.asyncio
async def test_quota_breaker_and_budget_restrictions_are_reported(tmp_path, monkeypatch, caplog):
    """Operational skips must identify quota and breaker state, not invent ranking."""
    from app.config import settings

    monkeypatch.setattr(settings, "REFERENCE_POOL_LIMIT", 6)
    caplog.set_level("INFO")
    ordinary = _release("Dexter.S08E01.720p.HDTV.x264-EVOLVE.srt", "https://cdn.example.invalid/a.srt")
    hashed = _release("Dexter.S08E01.720p.HDTV.x264-EVOLVE.srt", "https://cdn.example.invalid/b.srt")
    hashed.provider = "opensubtitles"
    os_provider = SimpleNamespace(name="opensubtitles", is_breaker_open=lambda: False)
    strategy = _strategy([ordinary, hashed], tmp_path, opensubtitles=os_provider)
    broken = SimpleNamespace(name="subdl", is_breaker_open=lambda: True)
    strategy._subdl = broken
    strategy._gather_candidates = AsyncMock(return_value=[(ordinary, broken), (hashed, strategy._opensubtitles)])
    plan = await strategy.plan_candidates(_query())
    report = _plan_report(caplog)

    assert [release.download_url for release, _ in plan] == [ordinary.download_url]
    assert report["planned"][0]["breaker_open"] is True
    reasons = {row["reason"] for row in report["excluded"]}
    assert "opensubtitles_non_hash_quota" in reasons
    assert report["budget_slots"] == 6
    blocked = await strategy.download_planned_candidate(
        ordinary, broken, _query(), None, {"downloaded": 0, "seen": set()}
    )
    assert blocked.reference_failure == "provider_circuit_breaker_open"


@pytest.mark.asyncio
async def test_planned_positions_follow_actual_lazy_execution(tmp_path, monkeypatch, caplog):
    """Plan indices must identify the releases in their executed attempt order."""
    from app.config import settings
    from app.services.sync.orchestrator import SyncOrchestrator

    monkeypatch.setattr(settings, "REFERENCE_POOL_LIMIT", 6)
    caplog.set_level("INFO")
    query = _query()
    first = _release("Dexter.S08E01.720p.HDTV.x264-EVOLVE.srt", "https://cdn.example.invalid/first.srt")
    second = _release("Dexter.S08E01.1080p.BluRay.REMUX-FraMeSToR.zip", "https://cdn.example.invalid/second.zip")
    seen = []

    async def download(release, provider, request, validate, shared, plan_index=None):
        seen.append((release.download_url, plan_index))
        from app.services.sync.query import ResolvedReference

        return ResolvedReference(None, reference_failure="candidate_rejected")

    strategy = SimpleNamespace(
        cache=SimpleNamespace(get=lambda request: None),
        plan_candidates=AsyncMock(return_value=[(first, "subdl"), (second, "subdl")]),
        download_planned_candidate=download,
    )
    orchestrator = SyncOrchestrator(external_strategy=strategy, sync_service=SimpleNamespace())
    generator = orchestrator._reference_attempts(
        [("external-release-reference", strategy)], query, lambda text: True, SyncTiming()
    )
    first_adapter = (await generator.__anext__())[1]
    second_adapter = (await generator.__anext__())[1]
    await first_adapter.resolve_with_provenance(query)
    await second_adapter.resolve_with_provenance(query)
    await generator.aclose()

    assert seen == [
        ("https://cdn.example.invalid/first.srt", 1),
        ("https://cdn.example.invalid/second.zip", 2),
    ]
    reached = [
        record.getMessage()
        for record in caplog.records
        if "candidate reached" in record.getMessage()
    ]
    assert not reached  # The spy bypasses strategy logging; indices are asserted above.


@pytest.mark.asyncio
async def test_empty_and_rate_limited_downloads_have_distinct_reasons(tmp_path):
    """An empty payload is only a rate-limit refusal when the breaker is open."""
    from app.services.sync.query import ResolvedReference

    provider = SimpleNamespace(name="subdl", is_breaker_open=lambda: False)
    provider.download_archive = AsyncMock(return_value=None)
    strategy = _strategy([], tmp_path)
    strategy._subdl = provider
    release = _release("Dexter.S08E01.720p.HDTV.x264-EVOLVE.srt", "https://cdn.example.invalid/a.srt")

    empty = await strategy._download_candidate(release, provider, _query())
    assert isinstance(empty, ResolvedReference)
    assert empty.reference_failure == "download_failure"

    provider.is_breaker_open = lambda: True
    limited = await strategy._download_candidate(release, provider, _query())
    assert limited.reference_failure == "provider_rate_limited"


@pytest.mark.asyncio
async def test_exhausted_alternatives_serve_original_safely(tmp_path, monkeypatch):
    """Independent-verification failure must fall back to the original bytes."""
    from app.config import settings
    from app.services.sync.orchestrator import SyncOrchestrator

    monkeypatch.setattr(settings, "ENABLE_SUBTITLE_SYNC", True)
    target = _srt("TARGET").encode("utf-8")
    meta = {
        "imdb_id": "tt0773262",
        "lang": "ara",
        "media_type": "series",
        "season": 8,
        "episode": 1,
        "target_filename": TARGET,
    }

    async def download(release, provider, request, validate, shared, plan_index=None):
        from app.services.sync.query import ResolvedReference

        return ResolvedReference(None, reference_failure="candidate_rejected")

    strategy = SimpleNamespace(
        cache=SimpleNamespace(get=lambda request: None),
        plan_candidates=AsyncMock(
            return_value=[(
                _release("Dexter.S08E01.720p.HDTV.x264-EVOLVE.srt", "https://cdn.example.invalid/a.srt"),
                "subdl",
            )]
        ),
        download_planned_candidate=download,
    )
    calls = []

    async def sync_async(*args, **kwargs):
        calls.append(args)
        raise AssertionError("no reference means ALASS must not run")

    orchestrator = SyncOrchestrator(
        external_strategy=strategy, sync_service=SimpleNamespace(sync_async=sync_async)
    )
    assert await orchestrator.evaluate_and_sync(target, meta, "hdtv", auto_sync=True) == target
    assert calls == []


@pytest.mark.asyncio
async def test_reference_planning_does_not_change_display_order(tmp_path, monkeypatch):
    """The reference exploration path must not alter user-facing ranking."""
    from app.config import settings
    from app.services.subtitle_matcher import rank_subtitles

    monkeypatch.setattr(settings, "REFERENCE_POOL_LIMIT", 6)
    pool = _dexter_pool()
    before = [
        (subtitle.release_name, subtitle.score, subtitle.match_percentage)
        for subtitle in rank_subtitles(
            TARGET, pool, preferred_languages=["eng"], discard_mismatches=False
        )
    ]
    strategy = _strategy(pool, tmp_path)
    await strategy.plan_candidates(_query())
    after = [
        (subtitle.release_name, subtitle.score, subtitle.match_percentage)
        for subtitle in rank_subtitles(
            TARGET, pool, preferred_languages=["eng"], discard_mismatches=False
        )
    ]

    assert after == before

@pytest.mark.asyncio
async def test_dexter_hdtv_switch_skips_provider_discovery_after_remux_cached(tmp_path):
    """Reproduce the HDTV-after-REMUX switch without repeating provider work."""
    from app.services.sync.orchestrator import SyncOrchestrator

    cache = ReferenceDiskCache(root=tmp_path / "refs", ttl=3600.0, min_bytes=100)
    remux_query = _query(target_text=_srt("REMUX"))
    cache.set_video_scoped(
        remux_query,
        "subdl",
        _srt("REMUX"),
        candidate="Dexter.S08.REMUX-FraMeSToR",
        member_season=8,
        member_episode=1,
        score=248.0,
        score_without_hash=248.0,
    )

    class _ExplodingProvider:
        async def search_subtitles(self, **kwargs):
            raise AssertionError("provider discovery must be skipped for a valid artifact")

        async def download_archive(self, url, api_key=None):
            raise AssertionError("archive download must be skipped for a valid artifact")

    strategy = ExternalExactStrategy(
        subdl_provider=_ExplodingProvider(),
        subsource_provider=_ExplodingProvider(),
        opensubtitles_provider=_ExplodingProvider(),
        cache=cache,
        min_bytes=100,
    )
    hdtv_query = _query(target_text=_srt("HDTV", start_ms=125_000))
    orchestrator = SyncOrchestrator(external_strategy=strategy, sync_service=SimpleNamespace())
    generator = orchestrator._reference_attempts(
        [("external-release-reference", strategy)], hdtv_query, lambda text: True, SyncTiming()
    )
    name, _ = await generator.__anext__()
    await generator.aclose()

    assert name == "external-release-reference#video_cache"


@pytest.mark.asyncio
async def test_failed_video_cache_continues_to_fresh_candidates(tmp_path):
    """A failed artifact must not become a dead end for alternative references."""
    from app.services.sync.orchestrator import SyncOrchestrator

    cache = ReferenceDiskCache(root=tmp_path / "refs", ttl=3600.0, min_bytes=100)
    query = _query()
    query.target_text = _srt("TARGET")
    cache.set_video_scoped(
        query,
        "subdl",
        _srt("CACHED"),
        candidate="Dexter.S08.REMUX-FraMeSToR",
        member_season=8,
        member_episode=1,
        score=100.0,
        score_without_hash=100.0,
    )
    planned_release = _release("Dexter.S08E01.720p.HDTV.x264-EVOLVE.srt", "https://cdn.example.invalid/a.srt")
    strategy = SimpleNamespace(
        cache=cache,
        plan_candidates=AsyncMock(return_value=[(planned_release, SimpleNamespace(name="subdl"))]),
    )
    orchestrator = SyncOrchestrator(external_strategy=strategy, sync_service=SimpleNamespace())
    generator = orchestrator._reference_attempts(
        [("external-release-reference", strategy)], query, lambda text: True, SyncTiming()
    )

    first = await generator.__anext__()
    assert first[0] == "external-release-reference#video_cache"
    assert strategy.plan_candidates.call_count == 0
    second = await generator.__anext__()
    assert second[0] == "external-release-reference#candidate1"
    assert strategy.plan_candidates.call_count == 1
    await generator.aclose()


@pytest.mark.asyncio
async def test_stronger_video_artifact_is_not_replaced_by_weaker_content(tmp_path):
    """Same source/kind filenames must retain, and prefer, stronger evidence."""
    cache = ReferenceDiskCache(root=tmp_path / "refs", ttl=3600.0, min_bytes=100)
    query = _query()
    strong = _srt("STRONG")
    weak = _srt("WEAK")
    cache.set_video_scoped(
        query, "subdl", strong, kind="hash", candidate="strong",
        member_season=8, member_episode=1, score=250.0, score_without_hash=200.0,
    )
    cache.set_video_scoped(
        query, "subdl", weak, kind="hash", candidate="weak",
        member_season=8, member_episode=1, score=10.0, score_without_hash=10.0,
    )
    cache.set_video_scoped(
        query, "subdl", weak, kind="edition", candidate="weak-edition",
        member_season=8, member_episode=1, score=500.0, score_without_hash=500.0,
    )

    resolved, _ = cache.get_video_scoped(query)

    assert resolved.text == strong
    assert len(list(cache.root.glob(f"{query.video_scope_stem}.video.subdl.*.srt"))) == 3


@pytest.mark.asyncio
async def test_independent_verification_guard_blocks_identical_cache_reuse(tmp_path):
    """Identical normalized cache content is not independent evidence."""
    from app.services.sync.orchestrator import _lookup_video_reference
    from app.services.sync.telemetry import SyncTiming

    cache = ReferenceDiskCache(root=tmp_path / "refs", ttl=3600.0, min_bytes=100)
    stored = _query()
    cache.set_video_scoped(
        stored, "subdl", _srt("SAME"), candidate="same",
        member_season=8, member_episode=1, score=100.0, score_without_hash=100.0,
    )
    current = _query(target_text=_srt("SAME", start_ms=99_000))
    timing = SyncTiming()

    assert _lookup_video_reference(SimpleNamespace(cache=cache), current, timing) is None
    assert timing.reference_outcomes["reference_content_identical_to_target"] == 1
