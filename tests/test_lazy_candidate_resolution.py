"""Lazy candidate resolution: download only what the loop actually reaches.

Previously ``resolve_all_with_provenance`` downloaded the entire pool upfront
(up to REFERENCE_POOL_LIMIT files, paced 0.6s apart) before verification even
started. A request that verified on the first candidate still paid for all the
rest -- wasted provider downloads and avoidable 429 pressure.

Now the strategy only *plans* (search + rank, no downloads) and each candidate
is downloaded inside the verification loop via ``_LazyCandidate``. When an
earlier candidate verifies, the loop returns and the remaining adapters never
download. All per-candidate guards (breaker, quota, pacing, cue-sanity,
content-dedup) still apply, evaluated at download time when that state is
freshest.
"""

from __future__ import annotations

import pytest

from app.services.sync import orchestrator as om
from app.services.sync.external_strategy import ExternalExactStrategy
from app.services.sync.orchestrator import SyncOrchestrator
from app.services.sync.query import ReferenceQuery, ResolvedReference


def _ts(ms: int) -> str:
    h, r = divmod(ms, 3600000)
    m, r = divmod(r, 60000)
    s, ms2 = divmod(r, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms2:03d}"


def _srt(cues: int, tag: str, offset_ms: int = 0) -> str:
    """Cue list with a shifted timeline so it genuinely needs alignment."""
    out = []
    for i in range(cues):
        s = 1000 + offset_ms + i * 3000
        e = s + 2000
        out.append(f"{i + 1}\n{_ts(s)} --> {_ts(e)}\n{tag}{i}\n")
    return "\n".join(out)


def _rel(name: str, **kw):
    from app.models import SubtitleRelease

    base = {"release_name": name, "download_url": f"/sub/x/{name}", "provider": "subdl"}
    base.update(kw)
    return SubtitleRelease(**base)


class _FakeProvider:
    name = "subdl"

    def __init__(self, payloads: dict[str, str]):
        self.payloads = payloads
        self.downloads: list[str] = []

    def is_breaker_open(self) -> bool:
        return False

    async def download_archive(self, release, api_key=None, username=None, password=None):
        name = getattr(release, "release_name", "")
        self.downloads.append(name)
        return self.payloads.get(name, "").encode("utf-8")


class _Strategy(ExternalExactStrategy):
    def __init__(self, payloads, **kw):
        super().__init__(**kw)
        self.provider = _FakeProvider(payloads)
        self._subdl = self.provider

    async def _gather_candidates(self, providers, query):
        rels = [_rel(n) for n in self.provider.payloads]
        return [(r, self.provider) for r in rels]

    async def _download_candidate(self, best, provider, query):
        name = getattr(best, "release_name", "?")
        raw = await provider.download_archive(best)
        if not raw:
            return ResolvedReference(None)
        return ResolvedReference(
            raw.decode("utf-8") if isinstance(raw, bytes) else raw,
            kind="edition", candidate=name, partial=False,
        )


def _strategy(payloads):
    return _Strategy(payloads, cache=None, timeout=5.0)


def _query(**kw):
    base = {
        "imdb_id": "tt2582802",
        "media_type": "movie",
        "target_filename": "Whiplash.2014.Criterion.Collection.2160p.UHD.BluRay.x254-SURCODE.mkv",
        "video_hash": "",
        "video_size": "19571049411",
        "languages": ("ara",),
    }
    base.update(kw)
    return ReferenceQuery(**base)


def _payloads(n: int) -> dict[str, str]:
    # Distinct content, small opening offsets: passes cue-sanity, needs alass.
    return {f"Whiplash.2014.Criterion.Collection.2160p.BluRay.x264-R{i}.mkv": _srt(600, f"C{i}", offset_ms=3000 * (i + 1)) for i in range(n)}


class _Cache:
    is_ephemeral = False

    async def get(self, k):
        return None

    async def get_meta(self, k):
        return None

    async def is_failed(self, k):
        return False

    async def get_verdict(self, k):
        return None

    async def set(self, k, d):
        pass

    async def set_meta(self, k, m):
        pass

    async def set_verdict(self, k, v, **kw):
        pass

    async def mark_failed(self, k, ttl=None):
        pass

    async def clear_failed(self, k):
        pass

    async def find_synced_for(self, *a, **kw):
        return None

    async def find_synced_for_target(self, *a, **kw):
        return None


def _meta():
    return {
        "media_type": "movie",
        "target_filename": "Whiplash.2014.Criterion.Collection.2160p.UHD.BluRay.x264-SURCODE.mkv",
        "video_hash": "",
        "video_size": "19571049411",
        "lang": "ara",
        "imdb_id": "tt2582802",
    }


@pytest.fixture(autouse=True)
def _gate(monkeypatch):
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True, raising=False)


# ===========================================================================
# Strategy level: planning is free, downloads happen on demand
# ===========================================================================


@pytest.mark.asyncio
async def test_plan_performs_no_downloads():
    s = _strategy(_payloads(4))
    plan = await s.plan_candidates(_query())
    assert len(plan) == 4
    assert s.provider.downloads == [], "planning must not touch the network"


@pytest.mark.asyncio
async def test_lazy_generator_downloads_one_at_a_time():
    s = _strategy(_payloads(4))
    gen = s.resolve_candidates_lazy(_query(), update_validator=lambda t: True)
    first = await gen.__anext__()
    assert first.text
    assert len(s.provider.downloads) == 1
    second = await gen.__anext__()
    assert second.text
    assert len(s.provider.downloads) == 2
    await gen.aclose()


@pytest.mark.asyncio
async def test_lazy_generator_stops_without_downloading_the_rest():
    """The whole point: a consumer that stops early pays only for what it used."""
    s = _strategy(_payloads(4))
    gen = s.resolve_candidates_lazy(_query(), update_validator=lambda t: True)
    first = await gen.__anext__()
    assert first.text
    await gen.aclose()
    assert len(s.provider.downloads) == 1


@pytest.mark.asyncio
async def test_lazy_generator_still_dedups_by_content():
    same = _srt(600, "SAME", offset_ms=3000)
    s = _strategy(
        {
            "One.2014.mkv": same,
            "Two.2014.mkv": same,
            "Three.2014.mkv": _srt(600, "OTHER", offset_ms=3000),
        }
    )
    out = [r async for r in s.resolve_candidates_lazy(_query(), update_validator=lambda t: True)]
    assert len(out) == 2, "identical content collapses even when resolved lazily"


@pytest.mark.asyncio
async def test_eager_resolve_all_still_works():
    """Back-compat: the eager API keeps its contract for existing callers."""
    s = _strategy(_payloads(3))
    out = await s.resolve_all_with_provenance(_query(), update_validator=lambda t: True)
    assert len(out) == 3
    assert len(s.provider.downloads) == 3


# ===========================================================================
# Orchestrator level: verification stops the downloads
# ===========================================================================


class _EchoSvc:
    """Returns the reference itself: a perfect alignment that verifies."""

    def __init__(self):
        self.calls = 0

    async def sync_async(self, target_text, reference, **kw):
        self.calls += 1
        return reference


class _JitterSvc:
    """Returns a scrambled timeline that verification must reject."""

    def __init__(self):
        self.calls = 0

    @staticmethod
    def _jittered() -> str:
        out = []
        for i in range(600):
            s = 1000 + i * 3000 + (4000 if i % 2 == 0 else -4000)
            e = s + (900 if i % 2 == 0 else 300)
            out.append(f"{i + 1}\n{_ts(s)} --> {_ts(e)}\nSCR{i}\n")
        return "\n".join(out)

    async def sync_async(self, target_text, reference, **kw):
        self.calls += 1
        return self._jittered()


@pytest.mark.asyncio
async def test_first_candidate_verifying_downloads_only_once():
    """THE regression this work exists for: previously all 4 would download."""
    strategy = _strategy(_payloads(4))
    svc = _EchoSvc()
    o = SyncOrchestrator(sync_cache=_Cache(), external_strategy=strategy, sync_service=svc)
    sub = _srt(600, "T").encode()
    out = await o.evaluate_and_sync(sub, _meta(), "tt2582802", auto_sync=True)
    assert svc.calls == 1, "only the first candidate should reach alass"
    assert len(strategy.provider.downloads) == 1, (
        f"expected exactly 1 download, got {len(strategy.provider.downloads)}"
    )
    assert out != sub, "the verified alignment must actually be served"


@pytest.mark.asyncio
async def test_rejection_downloads_the_next_candidate():
    """A rejected first candidate must still advance -- and download -- the next."""
    strategy = _strategy(_payloads(3))
    svc = _JitterSvc()
    o = SyncOrchestrator(sync_cache=_Cache(), external_strategy=strategy, sync_service=svc)
    sub = _srt(600, "T").encode()
    out = await o.evaluate_and_sync(sub, _meta(), "tt2582802", auto_sync=True)
    assert out == sub, "nothing verified, so the original must be served"
    assert svc.calls == 3, f"all 3 candidates must be attempted, got {svc.calls}"
    assert len(strategy.provider.downloads) == 3


@pytest.mark.asyncio
async def test_cache_hit_stays_first_without_redownload():
    """A proven cached reference keeps priority; the pool then fills in behind it
    without re-offering identical bytes."""

    class _HitStrategyCache:
        """Strategy-level cache returning a proven reference for this query."""

        def get(self, query):
            return ResolvedReference(
                _srt(600, "HIT", offset_ms=3000), kind="edition", candidate="Whiplash.2014.Criterion.Collection.2160p.BluRay.x264-HIT"
            )

    strategy = _strategy(_payloads(2))
    strategy.cache = _HitStrategyCache()
    svc = _EchoSvc()
    o = SyncOrchestrator(sync_cache=_Cache(), external_strategy=strategy, sync_service=svc)
    sub = _srt(600, "T").encode()
    await o.evaluate_and_sync(sub, _meta(), "tt2582802", auto_sync=True)
    # The cache hit verifies first; pool downloads must not happen at all.
    assert svc.calls == 1
    assert strategy.provider.downloads == [], (
        "a verifying cache hit must prevent every pool download"
    )


@pytest.mark.asyncio
async def test_strategy_without_plan_falls_back_to_eager():
    """Strategies that predate plan_candidates keep the old eager path."""

    class _Legacy:
        validates_target = False
        cache = None

        def __init__(self):
            self.calls = 0

        async def resolve_all_with_provenance(self, query, *, update_validator=None):
            self.calls += 1
            return [
                ResolvedReference(_srt(600, "L", offset_ms=3000), kind="edition", candidate="Whiplash.2014.Criterion.Collection.2160p.BluRay.x264-LEGACY")
            ]

    legacy = _Legacy()
    svc = _EchoSvc()
    o = SyncOrchestrator(sync_cache=_Cache(), external_strategy=legacy, sync_service=svc)
    sub = _srt(600, "T").encode()
    out = await o.evaluate_and_sync(sub, _meta(), "tt2582802", auto_sync=True)
    assert legacy.calls == 1
    assert out != sub
