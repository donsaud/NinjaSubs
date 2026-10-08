"""Download-ineligible metadata must not exhaust the bounded fallback plan."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from test_lazy_candidate_resolution import _EchoSvc, _JitterSvc, _srt

from app.models import SubtitleRelease
from app.services.sync import orchestrator as om
from app.services.sync.external_strategy import (
    TIER_SOURCE_EDITION,
    ExternalExactStrategy,
    _select_candidates_ranked,
    reference_tier,
)
from app.services.sync.orchestrator import SyncOrchestrator
from app.services.sync.query import ReferenceQuery, ResolvedReference
from app.services.sync.telemetry import SyncTiming, current_sync_timing


class Provider:
    def __init__(self, name, payloads):
        self.name = name
        self.payloads = payloads
        self.downloads = []

    async def download_archive(self, url, **kwargs):
        self.downloads.append(url)
        payload = self.payloads[url]
        if isinstance(payload, Exception):
            raise payload
        return payload.encode()


class Strategy(ExternalExactStrategy):
    def __init__(self, references):
        self.os = Provider("opensubtitles", {})
        self.sd = Provider("subdl", {})
        self.ss = Provider("subsource", {})
        super().__init__(self.sd, self.ss, self.os)
        self.cache = SimpleNamespace(get=lambda query: None)
        self.releases = []
        for i in range(6):
            self.add(self.os, f"Show.S01E01.Criterion.Collection.1080p.BluRay.x264-A{i}", "eng", "unused")
        for i, payload in enumerate(references):
            self.add(self.sd if i % 2 == 0 else self.ss,
                     f"Show.S01E01.Criterion.Collection.1080p.BluRay.x264-B{i}", "ara", payload)

    def add(self, provider, name, lang, payload, **kwargs):
        url = f"/{provider.name}/{len(self.releases)}"
        release = SubtitleRelease(
            release_name=name, provider=provider.name, lang=lang,
            download_url=url, **kwargs,
        )
        provider.payloads[url] = payload
        self.releases.append((release, provider))

    async def _gather_candidates(self, providers, query):
        return self.releases


META: dict[str, Any] = {
    "imdb_id": "tt1234567", "media_type": "series", "season": 1, "episode": 1,
    "target_filename": "Show.S01E01.Criterion.Collection.1080p.BluRay.x264-TARGET.mkv", "lang": "ara",
}
QUERY = ReferenceQuery(
    imdb_id=META["imdb_id"], media_type="series", season=1, episode=1,
    target_filename=META["target_filename"], languages=("eng", "ara"),
)
TARGET = _srt(600, "TARGET").encode()


@pytest.fixture(autouse=True)
def gates(monkeypatch):
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True)
    monkeypatch.setattr(om.settings, "REFERENCE_POOL_LIMIT", 6)
    monkeypatch.setattr(
        "app.services.sync.external_strategy._REFERENCE_DOWNLOAD_MIN_INTERVAL_SECONDS", 0,
    )


def pipeline(monkeypatch, strategy, service):
    orchestrator = SyncOrchestrator(external_strategy=strategy, sync_service=service)
    affine_attempts = []

    async def reject_affine(target, resolved, *args):
        affine_attempts.append(resolved)
        return None

    monkeypatch.setattr(orchestrator, "_try_timeline_affine", reject_affine)
    return orchestrator, affine_attempts


@pytest.mark.asyncio
async def test_ineligible_os_metadata_cannot_fill_pool():
    strategy = Strategy([_srt(600, "REFERENCE", 3000)])
    ranked = _select_candidates_ranked([r for r, _ in strategy.releases], QUERY)
    assert all(r.provider == "opensubtitles" for r in ranked[:6])
    plan = await strategy.plan_candidates(QUERY)
    assert len(plan) == 1
    assert plan[0][1] is strategy.sd
    assert reference_tier(QUERY.target_filename, plan[0][0]) == TIER_SOURCE_EDITION
    assert strategy.os.downloads == strategy.sd.downloads == []


@pytest.mark.asyncio
async def test_no_hash_reaches_download_affine_and_alass(monkeypatch):
    strategy = Strategy([_srt(600, "REFERENCE", 3000)] * 3)
    service = _EchoSvc()
    orchestrator, attempts = pipeline(monkeypatch, strategy, service)
    timing = SyncTiming()
    token = current_sync_timing.set(timing)
    try:
        result = await orchestrator._execute(TARGET, dict(META), "target", True)
    finally:
        current_sync_timing.reset(token)
    assert result != TARGET
    assert service.calls == 1
    assert len(attempts) == 1
    assert attempts[0].kind == "edition"
    assert attempts[0].reference_trust is None
    assert timing.candidate_count_planned == 3
    assert timing.candidates_considered == timing.references_downloaded == 1
    assert len(strategy.sd.downloads) == 1
    assert strategy.ss.downloads == strategy.os.downloads == []


@pytest.mark.asyncio
async def test_verifier_rejects_first_then_second_succeeds(monkeypatch):
    strategy = Strategy([_srt(600, "FIRST", 3000), _srt(600, "SECOND", 6000)])

    class Service(_JitterSvc):
        async def sync_async(self, target, reference, **kwargs):
            self.calls += 1
            return self._jittered() if self.calls == 1 else reference

    service = Service()
    orchestrator, attempts = pipeline(monkeypatch, strategy, service)
    result = await orchestrator._execute(TARGET, dict(META), "target", True)
    assert service.calls == len(attempts) == 2
    assert result == strategy.ss.payloads[strategy.ss.downloads[0]].encode()
    assert len(strategy.sd.downloads) == len(strategy.ss.downloads) == 1


@pytest.mark.asyncio
async def test_all_rejected_serve_original(monkeypatch):
    strategy = Strategy([_srt(600, "FIRST", 3000), _srt(600, "SECOND", 6000)])
    service = _JitterSvc()
    orchestrator, attempts = pipeline(monkeypatch, strategy, service)
    assert await orchestrator._execute(TARGET, dict(META), "target", True) == TARGET
    assert service.calls == len(attempts) == 2


@pytest.mark.asyncio
async def test_download_failure_continues_without_logging_url(monkeypatch, caplog):
    strategy = Strategy([RuntimeError("https://private.invalid/?token=secret"),
                         _srt(600, "SECOND", 6000)])
    service = _EchoSvc()
    orchestrator, attempts = pipeline(monkeypatch, strategy, service)
    assert await orchestrator._execute(TARGET, dict(META), "target", True) != TARGET
    assert service.calls == len(attempts) == 1
    assert len(strategy.sd.downloads) == len(strategy.ss.downloads) == 1
    assert "private.invalid" not in caplog.text
    assert "token=secret" not in caplog.text


@pytest.mark.asyncio
async def test_duplicate_content_not_evaluated_twice(monkeypatch):
    same = _srt(600, "SAME", 3000)
    strategy = Strategy([same, same, _srt(600, "THIRD", 6000)])
    service = _JitterSvc()
    orchestrator, attempts = pipeline(monkeypatch, strategy, service)
    assert await orchestrator._execute(TARGET, dict(META), "target", True) == TARGET
    assert service.calls == len(attempts) == 2
    assert len(strategy.sd.downloads) + len(strategy.ss.downloads) == 3


@pytest.mark.asyncio
async def test_exact_hash_reference_still_precedes_external_plan(monkeypatch):
    strategy = Strategy([_srt(600, "EXTERNAL", 3000)])
    service = _EchoSvc()
    orchestrator, attempts = pipeline(monkeypatch, strategy, service)

    async def hashed(query):
        assert not strategy.sd.downloads
        return ResolvedReference(_srt(600, "HASH", 3000), kind="hash")

    orchestrator._hash_reference_strategy = SimpleNamespace(resolve_with_provenance=hashed)
    assert await orchestrator._execute(TARGET, {**META, "video_hash": "1234567890abcdef"}, "target", True) != TARGET
    assert [r.kind for r in attempts] == ["hash"]
    assert strategy.sd.downloads == strategy.os.downloads == []


@pytest.mark.asyncio
async def test_os_hash_candidate_remains_eligible():
    strategy = Strategy([])
    strategy.add(strategy.os, "Show.S01E01.BluRay.x264-HASH", "eng",
                 _srt(600, "HASH"), is_hash_match=True)
    plan = await strategy.plan_candidates(QUERY)
    assert len(plan) == 1
    assert plan[0][0].is_hash_match


@pytest.mark.asyncio
async def test_os_non_hash_download_guard_is_preserved():
    strategy = Strategy([])
    release, provider = strategy.releases[0]
    result = await strategy.download_planned_candidate(release, provider, QUERY, None, {})
    assert result.text is None
    assert strategy.os.downloads == []
