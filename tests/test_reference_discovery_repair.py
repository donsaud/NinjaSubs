"""Reference discovery repair regressions.

The Mad Men WADU candidate was available in Display but disappeared from the
Reference pool when a SubSource season-catalog enrichment timed out. These
tests pin the three independent repairs:

* completed Tier-1 results survive Tier-2 failure/timeout;
* an exact same-release-group pack suppresses unnecessary enrichment while
  still requiring extraction/proof later;
* Display-discovered candidates are reusable only as same-video,
  same-authorization candidate hints.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.models import SubtitleRelease
from app.services import release_scoring_integration as rsi
from app.services.sync import discovery_hints
from app.services.sync.cache import ReferenceDiskCache
from app.services.sync.external_strategy import ExternalExactStrategy
from app.services.sync.query import ReferenceQuery, ResolvedReference

VIDEO = (
    "Mad.Men.S01E01.Smoke.Gets.in.Your.Eyes.REPACK.2160p.HMAX.WEB-DL."
    "DDP5.1.DV.HDR.H.265-WADU.mkv"
)
WADU = "Mad.Men.S01.REPACK.2160p.HMAX.WEB-DL.DDP5.1.DV.HDR.H.265-WADU.srt"
BTN = "Mad.Men.S01.720p.WEB-DL.AAC2.0.H264-BTN.srt"


def _query(**overrides) -> ReferenceQuery:
    base = dict(
        imdb_id="tt0804503",
        media_type="series",
        season=1,
        episode=1,
        target_filename=VIDEO,
        title="Mad Men",
        year=2007,
        video_size=8113838532,
        release_group="WADU",
        languages=("eng", "ara"),
    )
    base.update(overrides)
    return ReferenceQuery(**base)


def _release(name: str, *, provider: str, url: str, lang: str) -> SubtitleRelease:
    release = SubtitleRelease(
        release_name=name, provider=provider, download_url=url, lang=lang
    )
    evidence = rsi.record(release)
    evidence["metadata"].update(imdb_id="tt0804503", imdb_verified=True)
    release.release_evidence = [evidence]
    return release


def _wadu() -> SubtitleRelease:
    return _release(WADU, provider="subsource", url="https://invalid/wadu", lang="ara")


def _btn() -> SubtitleRelease:
    return _release(BTN, provider="subdl", url="https://invalid/btn", lang="eng")


class _Provider:
    name = "subsource"

    def is_breaker_open(self) -> bool:
        return False


@pytest.fixture(autouse=True)
def _clean_hints():
    discovery_hints.clear()
    yield
    discovery_hints.clear()


def test_exact_release_pack_skips_unnecessary_enrichment():
    query = _query()
    assert ExternalExactStrategy._needs_season_enrichment([_wadu()], query) is False
    assert ExternalExactStrategy._needs_season_enrichment([_btn()], query) is True
    assert ExternalExactStrategy._needs_season_enrichment([_btn(), _wadu()], query) is False


def test_wrong_episode_exact_group_does_not_suppress_enrichment():
    wrong_episode = _release(
        "Mad.Men.S01E02.REPACK.2160p.HMAX.WEB-DL.DDP5.1.DV.HDR.H.265-WADU.srt",
        provider="subsource",
        url="https://invalid/wadu-e02",
        lang="ara",
    )
    assert ExternalExactStrategy._needs_season_enrichment([wrong_episode], _query()) is True


@pytest.mark.asyncio
async def test_tier1_survives_tier2_timeout(monkeypatch):
    strategy = ExternalExactStrategy(timeout=0.3)
    tier1 = _btn()
    calls: list[int | None] = []

    async def search(episode):
        calls.append(episode)
        if episode is None:
            await asyncio.sleep(5)
        return [tier1] if episode is not None else []

    # Force the enrichment branch; the exact-pack shortcut is tested above.
    monkeypatch.setattr(
        ExternalExactStrategy,
        "_needs_season_enrichment",
        staticmethod(lambda releases, query, strict=True: True),
    )
    started = time.monotonic()
    result = await strategy._search_tiers(
        search, _query(), "subsource", deadline=started + 1.0
    )
    elapsed = time.monotonic() - started

    assert result == [tier1]
    assert calls == [1, None]
    assert elapsed < 1.5


@pytest.mark.asyncio
async def test_provider_deadline_cannot_discard_completed_tier1():
    provider = _Provider()
    strategy = ExternalExactStrategy(subsource_provider=provider, timeout=0.3)
    tier1 = _btn()

    async def search(episode):
        if episode is None:
            await asyncio.sleep(5)
        return [tier1] if episode is not None else []

    async def provider_search(provider_arg, query, *, deadline=None):
        assert provider_arg is provider
        assert deadline is not None
        return await strategy._search_tiers(search, query, "subsource", deadline=deadline)

    strategy._search_provider = provider_search  # type: ignore[method-assign]
    result = await strategy._gather_candidates([provider], _query())

    assert [release for release, _ in result] == [tier1]


def test_wadu_ranks_above_btn_in_both_shared_policies():
    query = _query()
    releases = [_btn(), _wadu()]

    pairs, _ = rsi.discovery_pairs(releases, query)
    reference_order = [result.candidate.release_claims[0] for _, result in pairs]
    assert "wadu" in reference_order[0].lower()

    display = rsi.display_order(
        VIDEO,
        [_btn(), _wadu()],
        preferred_languages=["ara"],
        season=1,
        episode=1,
        title="Mad Men",
        year=2007,
        imdb_id="tt0804503",
        media_type="series",
        video_size=8113838532,
    )
    assert display
    assert display[0].release_name == WADU


def test_hint_cache_is_video_and_authorization_scoped():
    video_key = discovery_hints.video_scope_key(
        imdb_id="tt0804503", media_type="series", season=1, episode=1,
        target_filename=VIDEO, video_size=8113838532,
    )
    auth = discovery_hints.authorization_digest(subsource_key="user-a")
    other_auth = discovery_hints.authorization_digest(subsource_key="user-b")
    other_video = discovery_hints.video_scope_key(
        imdb_id="tt0804503", media_type="series", season=1, episode=2,
        target_filename=VIDEO, video_size=8113838532,
    )

    assert discovery_hints.register(video_key, auth, [_wadu()]) == 1
    assert len(discovery_hints.lookup(video_key, auth)) == 1
    assert discovery_hints.lookup(video_key, other_auth) == []
    assert discovery_hints.lookup(other_video, auth) == []


@pytest.mark.asyncio
async def test_exact_pack_failure_continues_to_tier1_alternative(tmp_path):
    """P1 must not strand the request when the exact pack cannot be proven."""

    class Provider:
        name = "subsource"

        def __init__(self):
            self.search_calls: list[int | None] = []

        def is_breaker_open(self) -> bool:
            return False

        async def search_subtitles(self, *, episode=None, **kwargs):
            self.search_calls.append(episode)
            return [_wadu(), _btn()] if episode is not None else []

    class Strategy(ExternalExactStrategy):
        def __init__(self, provider, cache):
            super().__init__(subsource_provider=provider, cache=cache, timeout=1.0)
            self.provider = provider
            self.downloads: list[str] = []

        async def _download_candidate(self, best, provider, query):
            name = getattr(best, "release_name", "")
            self.downloads.append(name)
            if name == WADU:
                return ResolvedReference(None, reference_failure="reference_not_resolvable")
            return ResolvedReference(
                "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n"
                "2\n00:00:03,000 --> 00:00:04,000\nWorld\n",
                kind="edition", candidate=name, member_season=1, member_episode=1,
            )

    provider = Provider()
    strategy = Strategy(provider, ReferenceDiskCache(root=tmp_path, min_bytes=1))
    results = [result async for result in strategy.resolve_candidates_lazy(_query())]

    assert provider.search_calls == [1], "P1 should skip Tier 2 for the exact pack"
    assert strategy.downloads == [WADU, BTN], "the exact pack failure must advance to Tier 1 fallback"
    assert len(results) == 1
    assert results[0].candidate == BTN
    assert results[0].text


@pytest.mark.asyncio
async def test_display_hints_recover_candidates_when_provider_fails():
    provider = _Provider()
    query = _query(api_keys={"subsource": "user-a"})
    video_key = discovery_hints.video_scope_key(
        imdb_id=query.imdb_id, media_type=query.media_type, season=query.season,
        episode=query.episode, target_filename=query.target_filename,
        video_hash=query.video_hash, video_size=query.video_size,
    )
    auth = discovery_hints.authorization_digest(subsource_key="user-a")
    hint = _wadu()
    discovery_hints.register(video_key, auth, [hint])

    strategy = ExternalExactStrategy(subsource_provider=provider, timeout=0.3)

    async def fail_provider(provider_arg, query_arg, *, deadline=None):
        raise TimeoutError("season catalog timed out")

    strategy._search_provider = fail_provider  # type: ignore[method-assign]
    result = await strategy._gather_candidates([provider], query)

    assert len(result) == 1
    release, provider_out = result[0]
    assert release.release_name == WADU
    assert provider_out is provider
    assert release is not hint, "hints must be copies, never shared mutable state"
