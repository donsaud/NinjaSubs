"""Tests for the sync strategies (hash-exact, embedded) and the orchestrator."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from app.models import SubtitleRelease
from app.services.sync.cache import ReferenceDiskCache
from app.services.sync.embedded_strategy import EmbeddedStrategy
from app.services.sync.hash_strategy import HashExactStrategy
from app.services.sync.orchestrator import SyncOrchestrator, build_synced_cache_key
from app.services.sync.query import ReferenceQuery
from app.services.sync_cache import SyncCache

BIG_REF = (
    "1\n00:00:01,000 --> 00:00:02,000\n" + ("reference line\n" * 600) + "\n"
).encode()


def _os_release(name, file_id, *, hash_match=True, hi=False):
    return SubtitleRelease(
        release_name=name,
        download_url=f"/sub/opensubtitles/{file_id}.srt",
        provider="opensubtitles",
        lang="eng",
        hearing_impaired=hi,
        is_hash_match=hash_match,
    )


class _FakeOSProvider:
    def __init__(self, releases, payload=BIG_REF):
        self._releases = releases
        self._payload = payload
        self.search_calls: list[dict] = []
        self.download_calls: list[str] = []

    async def search_subtitles(self, **kwargs):
        self.search_calls.append(kwargs)
        return list(self._releases)

    async def download_archive(self, download_ref, api_key=None):
        self.download_calls.append(download_ref)
        return self._payload


def _hash_query(**overrides):
    params = {
        "imdb_id": "tt0758758",
        "media_type": "movie",
        "target_filename": "Into.The.Wild.2007.1080p.BluRay.x264-FSiHD",
        "video_hash": "8b2ashafthash1234",
        "languages": ("eng",),
    }
    params.update(overrides)
    return ReferenceQuery(**params)


@pytest.mark.asyncio
async def test_hash_strategy_needs_provider_and_hash():
    assert await HashExactStrategy(None).resolve(_hash_query()) is None

    provider = _FakeOSProvider([_os_release("x.srt", 1)])
    strategy = HashExactStrategy(provider)
    assert await strategy.resolve(_hash_query(video_hash="")) is None
    assert await strategy.resolve(_hash_query(video_hash=None)) is None
    assert provider.search_calls == []


@pytest.mark.asyncio
async def test_hash_strategy_downloads_confirmed_match_and_caches(tmp_path):
    provider = _FakeOSProvider(
        [
            _os_release("Other.srt", 9, hash_match=False),
            _os_release("Into.The.Wild.HI.srt", 7, hi=True),
            _os_release("Into.The.Wild.2007.srt", 8),
        ]
    )
    cache = ReferenceDiskCache(root=tmp_path / "disk", ttl=3600.0, min_bytes=100)
    strategy = HashExactStrategy(provider, cache=cache, timeout=2.0)

    content = await strategy.resolve(_hash_query())
    assert content is not None and "reference line" in content
    # Non-HI hash match preferred; the release search carried the video hash.
    assert provider.download_calls == ["/sub/opensubtitles/8.srt"]
    assert provider.search_calls[0]["video_hash"] == "8b2ashafthash1234"
    assert (tmp_path / "disk" / f"{_hash_query().cache_stem}_opensubtitles_hash.srt").exists()


@pytest.mark.asyncio
async def test_hash_strategy_rejects_unconfirmed_candidates():
    provider = _FakeOSProvider([_os_release("Into.The.Wild.2007.srt", 8, hash_match=False)])
    strategy = HashExactStrategy(provider, timeout=2.0)
    assert await strategy.resolve(_hash_query()) is None
    assert provider.download_calls == []


class _FakeStrategy:
    def __init__(self, result=None, exc=None, kind="team", bluray_match=False):
        self.result = result
        self.exc = exc
        self.kind = kind
        self.bluray_match = bluray_match
        self.calls = 0

    async def resolve(self, query):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return self.result

    async def resolve_with_provenance(self, query):
        from app.services.sync.query import ResolvedReference

        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return ResolvedReference(self.result, kind=self.kind, bluray_match=self.bluray_match)


class _FakeSyncService:
    def __init__(self):
        self.calls = 0
        self.kinds: list[str] = []
        self.flags: list[tuple] = []
        self.relaxed: list[bool] = []
        self.partial: list[bool] = []

    async def sync_async(
        self,
        target,
        reference,
        decision_kind="edition",
        is_series=False,
        source_confirmed=False,
        relaxed=False,
        reference_partial=False,
    ):
        self.calls += 1
        self.kinds.append(decision_kind)
        self.flags.append((is_series, source_confirmed))
        self.relaxed.append(relaxed)
        self.partial.append(reference_partial)
        assert "reference line" in reference
        return "1\n00:00:09,000 --> 00:00:10,000\nsynced\n"


def _arabic_bytes(n=6):
    blocks = []
    for i in range(n):
        blocks.append(f"{i + 1}\n00:00:{i * 2 + 1:02d},000 --> 00:00:{i * 2 + 2:02d},000\nمرحبا")
    return ("\n\n".join(blocks) + "\n").encode()


def _orchestrator(**overrides):
    params = {
        "hash_strategy": _FakeStrategy(None),
        "embedded_strategy": _FakeStrategy(None),
        "sync_service": _FakeSyncService(),
        "sync_cache": SyncCache(),
    }
    params.update(overrides)
    return SyncOrchestrator(**params)


def _meta():
    return {"imdb_id": "tt1", "media_type": "movie", "lang": "ara"}


@pytest.mark.asyncio
async def test_orchestrator_gates_skip_strategies(monkeypatch):
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", False)
    orch = _orchestrator()
    assert await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True) == _arabic_bytes()
    assert orch._hash_strategy.calls == 0

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
    assert await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t", False) == _arabic_bytes()
    assert await orch.evaluate_and_sync(_arabic_bytes(), {"lang": "eng"}, "t", True) == _arabic_bytes()
    assert orch._hash_strategy.calls == 0 and orch._embedded_strategy.calls == 0


@pytest.mark.asyncio
async def test_orchestrator_prefers_embedded_then_caches(monkeypatch):
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
    reference = BIG_REF.decode()
    # Tier order is embedded -> external -> hash: the ground-truth embedded
    # reference (the video's own track) wins.
    orch = _orchestrator(
        embedded_strategy=_FakeStrategy(reference),
        external_strategy=_FakeStrategy("should-not-be-used"),
        hash_strategy=_FakeStrategy("should-not-be-used"),
    )
    out = await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True)
    assert b"synced" in out
    assert orch._embedded_strategy.calls == 1
    assert orch._external_strategy.calls == 0 and orch._hash_strategy.calls == 0
    assert orch._sync_service.calls == 1

    # A warm result is served before reference resolution or alass.
    orch2 = _orchestrator(
        embedded_strategy=_FakeStrategy("x"),
        external_strategy=_FakeStrategy("x"),
        hash_strategy=_FakeStrategy("x"),
        sync_cache=orch._sync_cache,
    )
    out2 = await orch2.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True)
    assert out2 == out
    assert orch2._embedded_strategy.calls == 0
    assert orch2._external_strategy.calls == 0 and orch2._hash_strategy.calls == 0
    assert orch2._sync_service.calls == 0


@pytest.mark.asyncio
async def test_orchestrator_uses_embedded_tier_first(monkeypatch, caplog):
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
    orch = _orchestrator(embedded_strategy=_FakeStrategy(BIG_REF.decode()))
    with caplog.at_level("INFO"):
        out = await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True)
    assert b"synced" in out
    assert orch._embedded_strategy.calls == 1 and orch._hash_strategy.calls == 0

    # No deterministic tier delivers: original served with the abort message.
    orch = _orchestrator()
    with caplog.at_level("INFO"):
        assert await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True) == _arabic_bytes()
    assert "[sync] no deterministic reference available -> aborting sync" in caplog.text


@pytest.mark.asyncio
async def test_orchestrator_survives_strategy_errors(monkeypatch):
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
    orch = _orchestrator(
        hash_strategy=_FakeStrategy(exc=RuntimeError("boom")),
        embedded_strategy=_FakeStrategy(BIG_REF.decode()),
    )
    out = await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True)
    assert b"synced" in out


@pytest.mark.asyncio
async def test_orchestrator_keys_isolate_distinct_payloads(monkeypatch):
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
    orch = _orchestrator(embedded_strategy=_FakeStrategy(BIG_REF.decode()))
    await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True)
    other = _arabic_bytes().replace("مرحبا".encode(), "أهلا".encode())
    await orch.evaluate_and_sync(other, _meta(), "t", True)
    # Same subtitle ID, different bytes: resolved twice, no false cache hit.
    assert orch._embedded_strategy.calls == 2


def test_synced_cache_key_binds_payload():
    meta = {"imdb_id": "tt1", "season": 1, "episode": 1, "video_hash": "VH"}
    key_a = build_synced_cache_key(meta, "s", content_hash="a", decision="team")
    key_b = build_synced_cache_key(meta, "s", content_hash="b", decision="team")
    assert key_a != key_b
    assert key_a == "final_sub:tt1:s1e1:vh:s:team:a"


@pytest.mark.asyncio
async def test_orchestrator_forwards_decision_kind_to_alass(monkeypatch):
    """The edition/team verdict must reach the guardrail."""
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
    orch = _orchestrator(
        hash_strategy=_FakeStrategy(None),
        embedded_strategy=_FakeStrategy(BIG_REF.decode(), kind="edition"),
    )
    await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True)
    assert orch._sync_service.kinds == ["edition"]

    orch = _orchestrator(hash_strategy=_FakeStrategy(BIG_REF.decode(), kind="hash"))
    await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True)
    assert orch._sync_service.kinds == ["hash"]


@pytest.mark.asyncio
async def test_orchestrator_trusts_alass_output(monkeypatch):
    """A zero-exit alass output reaches the client regardless of the shift."""
    import subprocess

    from app.config import settings as app_settings
    from app.services.sync_service import SubtitleSyncService

    ref = "".join(
        f"{i + 1}\n00:00:{i * 2 + 1:02d},000 --> 00:00:{i * 2 + 2:02d},000\nref {i}\n\n"
        for i in range(6)
    )

    def _shifted_run(command, capture_output=True, timeout=None):
        # alass "succeeds" but with a +69s shift on the first cue.
        drifted = "".join(
            f"{i + 1}\n00:01:{10 + i * 2:02d},000 --> 00:01:{12 + i * 2:02d},000\nsynced {i}\n\n"
            for i in range(6)
        )
        with open(command[3], "w", encoding="utf-8") as handle:
            handle.write(drifted)
        return SimpleNamespace(returncode=0, stderr=b"", stdout=b"")

    monkeypatch.setattr(subprocess, "run", _shifted_run)
    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)

    for kind in ("edition", "team"):
        orch = _orchestrator(
            embedded_strategy=_FakeStrategy(ref, kind=kind),
            sync_service=SubtitleSyncService(),
        )
        out = await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True)
        assert b"synced" in out


def _probe_payload(*langs):
    streams = [
        {"index": i, "codec_name": "subrip", "tags": {"language": lang}}
        for i, lang in enumerate(langs)
    ]
    return json.dumps({"streams": streams}).encode()


def _embedded_query(**overrides):
    params = {
        "imdb_id": "tt1",
        "media_type": "movie",
        "stream_url": "https://cdn.example/video.mkv",
    }
    params.update(overrides)
    return ReferenceQuery(**params)


def _big_srt_bytes():
    return ("1\n00:00:01,000 --> 00:00:02,000\n" + ("hello\n" * 600) + "\n").encode()


@pytest.mark.asyncio
async def test_embedded_strategy_extracts_english_track(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0].endswith("ffprobe"):
            return SimpleNamespace(returncode=0, stdout=_probe_payload("spa", "eng"), stderr=b"")
        return SimpleNamespace(returncode=0, stdout=_big_srt_bytes(), stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        "app.services.sync.embedded_strategy.is_safe_public_url", lambda url, **kwargs: (True, "ok")
    )
    cache = ReferenceDiskCache(root=tmp_path / "disk", ttl=3600.0, min_bytes=100)
    strategy = EmbeddedStrategy(
        ffprobe_path="/fake/ffprobe",
        ffmpeg_path="/fake/ffmpeg",
        cache=cache,
        timeout=2.0,
        min_bytes=100,
    )
    content = await strategy.resolve(_embedded_query())
    assert content is not None and "hello" in content
    assert any("ffprobe" in cmd[0] for cmd in calls)
    extract = [cmd for cmd in calls if "ffmpeg" in cmd[0]][0]
    assert extract[extract.index("-map") + 1] == "0:1"
    assert list((tmp_path / "disk").glob("*.srt")) == [
        tmp_path / "disk" / f"{_embedded_query().cache_stem}_embedded_embedded.srt"
    ]


@pytest.mark.asyncio
async def test_embedded_strategy_aborts_without_english_track(monkeypatch):
    def fake_run(cmd, **kwargs):
        return SimpleNamespace(returncode=0, stdout=_probe_payload("spa", "fre"), stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        "app.services.sync.embedded_strategy.is_safe_public_url", lambda url, **kwargs: (True, "ok")
    )
    strategy = EmbeddedStrategy(
        ffprobe_path="/fake/ffprobe", ffmpeg_path="/fake/ffmpeg", timeout=2.0
    )
    assert await strategy.resolve(_embedded_query()) is None


@pytest.mark.asyncio
async def test_embedded_strategy_skips_without_url_or_binaries(monkeypatch):
    monkeypatch.setattr(
        "app.services.sync.embedded_strategy.is_safe_public_url", lambda url, **kwargs: (True, "ok")
    )
    strategy = EmbeddedStrategy(ffprobe_path="/fake/ffprobe", ffmpeg_path="/fake/ffmpeg")
    assert await strategy.resolve(_embedded_query(stream_url="")) is None
    assert await strategy.resolve(_embedded_query(stream_url=None)) is None

    strategy = EmbeddedStrategy(ffprobe_path="/nonexistent/ffprobe-xyz")
    assert await strategy.resolve(_embedded_query()) is None


@pytest.mark.asyncio
async def test_embedded_strategy_blocks_unsafe_stream_url(monkeypatch, caplog):
    """An SSRF/private URL must abort before any ffprobe/ffmpeg subprocess."""
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # pragma: no cover - must never be reached
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    strategy = EmbeddedStrategy(ffprobe_path="/fake/ffprobe", ffmpeg_path="/fake/ffmpeg")
    with caplog.at_level("WARNING"):
        result = await strategy.resolve(_embedded_query(stream_url="file:///etc/passwd"))
    assert result is None
    assert calls == []
    assert "blocked unsafe stream URL" in caplog.text

    monkeypatch.setattr(
        "app.services.sync.embedded_strategy.is_safe_public_url",
        lambda url, **kwargs: (False, "blocked address 127.0.0.1"),
    )
    result = await strategy.resolve(_embedded_query(stream_url="http://127.0.0.1/video.mkv"))
    assert result is None
    assert calls == []


@pytest.mark.asyncio
async def test_precheck_skips_sync_on_group_match(monkeypatch, caplog):
    """Same release group on stream and subtitle: serve original, no alass."""
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
    meta = {
        "imdb_id": "tt1",
        "media_type": "movie",
        "lang": "ara",
        "target_filename": "Into.The.Wild.2007.1080p.BluRay.x264-FSiHD.mkv",
        "release_name": "Into.The.Wild.2007.1080p.BluRay.x264-FSiHD.srt",
    }
    orch = _orchestrator()
    with caplog.at_level("INFO"):
        out = await orch.evaluate_and_sync(_arabic_bytes(), meta, "t", True)
    assert out == _arabic_bytes()
    assert "already matches release group (FSiHD)" in caplog.text
    assert orch._hash_strategy.calls == 0 and orch._embedded_strategy.calls == 0
    assert orch._sync_service.calls == 0


@pytest.mark.asyncio
async def test_precheck_is_case_insensitive_and_requires_both_sides(monkeypatch):
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
    meta = {
        "imdb_id": "tt1",
        "media_type": "movie",
        "lang": "ara",
        "target_filename": "Show.1080p.WEB-DL.x264-chd.mkv",
        "release_name": "Show.1080p.WEB-DL.x264-CHD.srt",
    }
    orch = _orchestrator()
    assert await orch.evaluate_and_sync(_arabic_bytes(), meta, "t", True) == _arabic_bytes()
    assert orch._hash_strategy.calls == 0


@pytest.mark.asyncio
async def test_precheck_proceeds_on_mismatch_or_missing_group(monkeypatch):
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
    mismatched = {
        "imdb_id": "tt1",
        "media_type": "movie",
        "lang": "ara",
        "target_filename": "Show.1080p.WEB-DL.x264-GRP.mkv",
        "release_name": "Show.1080p.WEB-DL.x264-OTHER.srt",
    }
    orch = _orchestrator(hash_strategy=_FakeStrategy(BIG_REF.decode()))
    await orch.evaluate_and_sync(_arabic_bytes(), mismatched, "t", True)
    assert orch._hash_strategy.calls == 1

    # Identical filenames (listing fallback) must not trivially match.
    identical = dict(mismatched, target_filename="Show.1080p.WEB-DL.x264-GRP.srt",
                     release_name="Show.1080p.WEB-DL.x264-GRP.srt")
    orch = _orchestrator(hash_strategy=_FakeStrategy(BIG_REF.decode()))
    await orch.evaluate_and_sync(_arabic_bytes(), identical, "t", True)
    assert orch._hash_strategy.calls == 1

    # No group on either side: normal flow.
    bare = {"imdb_id": "tt1", "media_type": "movie", "lang": "ara"}
    orch = _orchestrator(hash_strategy=_FakeStrategy(BIG_REF.decode()))
    await orch.evaluate_and_sync(_arabic_bytes(), bare, "t", True)
    assert orch._hash_strategy.calls == 1


@pytest.mark.asyncio
async def test_orchestrator_forwards_recap_context(monkeypatch):
    """Series + bluray verdict must reach the guardrail together."""
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
    orch = _orchestrator(
        external_strategy=_FakeStrategy(BIG_REF.decode(), kind="edition", bluray_match=True)
    )
    meta = {
        "imdb_id": "tt0773262", "media_type": "series", "season": 8, "episode": 1,
        "lang": "ara", "target_filename": "Dexter.S08E01.1080p.BluRay.x264.mkv",
    }
    await orch.evaluate_and_sync(_arabic_bytes(), meta, "t", True)
    assert orch._sync_service.kinds == ["edition"]
    assert orch._sync_service.flags == [(True, True)]


@pytest.mark.asyncio
async def test_single_flight_coalesces_identical_requests(monkeypatch):
    """Two simultaneous identical syncs run the pipeline exactly once."""
    import asyncio

    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)

    release = asyncio.Event()

    class _SlowStrategy:
        def __init__(self):
            self.calls = 0

        async def resolve_with_provenance(self, query):
            from app.services.sync.query import ResolvedReference

            self.calls += 1
            await release.wait()
            return ResolvedReference(BIG_REF.decode(), kind="team")

    strategy = _SlowStrategy()
    orch = _orchestrator(external_strategy=strategy)
    first = asyncio.ensure_future(orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True))
    await asyncio.sleep(0)
    second = asyncio.ensure_future(orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True))
    await asyncio.sleep(0)
    release.set()
    out1, out2 = await asyncio.gather(first, second)
    assert strategy.calls == 1
    assert out1 == out2 and b"synced" in out1
    assert orch._inflight == {}


@pytest.mark.asyncio
async def test_single_flight_isolates_distinct_keys(monkeypatch):
    """Different subtitle IDs never share a flight."""
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
    orch = _orchestrator(external_strategy=_FakeStrategy(BIG_REF.decode(), kind="team"))
    await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t1", True)
    await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t2", True)
    assert orch._external_strategy.calls == 2
    assert orch._inflight == {}


@pytest.mark.asyncio
async def test_single_flight_propagates_errors_and_cleans_up(monkeypatch):
    """A failing flight raises for every waiter and leaves no residue."""
    import asyncio

    from app.config import settings as app_settings

    class _BoomService:
        def __init__(self):
            self.calls = 0

        async def sync_async(self, *args, **kwargs):
            self.calls += 1
            raise RuntimeError("boom")

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
    orch = _orchestrator(
        external_strategy=_FakeStrategy(BIG_REF.decode(), kind="team"),
        sync_service=_BoomService(),
    )
    results = await asyncio.gather(
        orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True),
        orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True),
        return_exceptions=True,
    )
    assert orch._external_strategy.calls == 1
    assert orch._sync_service.calls == 1
    assert all(isinstance(item, RuntimeError) for item in results)
    assert orch._inflight == {}


@pytest.mark.asyncio
async def test_sync_cache_sidecar_records_provenance(monkeypatch):
    """A successful sync stores decision metadata beside the payload."""
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
    cache = SyncCache()
    orch = _orchestrator(
        hash_strategy=_FakeStrategy(BIG_REF.decode(), kind="hash"), sync_cache=cache
    )
    out = await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True)
    assert b"synced" in out

    key = next(key for key in cache._local if key.startswith("final_sub:"))
    meta = await cache.get_meta(key)
    assert meta is not None
    assert meta["status"] == "synced"
    assert meta["decision"] == "hash"
    assert isinstance(meta["applied_shift"], list)
    assert len(meta["reference_sha"]) == 16
    assert "timestamp" in meta


@pytest.mark.asyncio
async def test_sync_cache_sidecar_absent_without_sync(monkeypatch):
    """Aborted syncs record no provenance."""
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
    cache = SyncCache()
    orch = _orchestrator(sync_cache=cache)
    await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True)
    assert list(cache._local) == []
    assert await cache.get_meta("final_sub:whatever") is None


# --- Remediation tests: singleton coalescing, negative cache, REMUX symmetry ---


@pytest.mark.asyncio
async def test_app_singleton_orchestrator_coalesces_concurrent_requests(monkeypatch):
    """Two concurrent serve-path requests must share one in-flight future."""
    import asyncio

    from app import main
    from app.config import settings as app_settings
    from app.services.sync.query import ResolvedReference

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
    monkeypatch.setattr(app_settings, "SYNC_TOTAL_REQUEST_BUDGET", 30.0)
    monkeypatch.setattr(main, "_http_client", object())
    monkeypatch.setattr(main, "_sync_cache", SyncCache())

    orch = main._build_sync_orchestrator()
    assert orch is not None

    released = asyncio.Event()
    calls = {"n": 0}

    class _Slow:
        async def resolve_with_provenance(self, query):
            calls["n"] += 1
            await released.wait()
            return ResolvedReference(BIG_REF.decode(), kind="team")

    orch._external_strategy = _Slow()
    orch._sync_service = _FakeSyncService()
    monkeypatch.setattr(main, "_get_sync_orchestrator", lambda: orch)

    first = asyncio.ensure_future(main._maybe_sync_subtitle(_arabic_bytes(), _meta(), "t", True))
    await asyncio.sleep(0)
    second = asyncio.ensure_future(main._maybe_sync_subtitle(_arabic_bytes(), _meta(), "t", True))
    for _ in range(50):
        await asyncio.sleep(0)
        if calls["n"] == 1:
            break

    # Exactly one shared future is in flight; the pipeline ran once.
    assert len(orch._inflight) == 1
    flight = next(iter(orch._inflight.values()))
    assert not flight.done()
    assert calls["n"] == 1

    released.set()
    out1, out2 = await asyncio.gather(first, second)
    assert out1 == out2
    assert b"synced" in out1
    assert orch._inflight == {}


@pytest.mark.asyncio
async def test_request_budget_returns_original_and_warms_cache(monkeypatch):
    """Over-budget sync serves the original but the detached task still caches."""
    import asyncio

    from app import main
    from app.config import settings as app_settings
    from app.services.sync.query import ResolvedReference

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
    monkeypatch.setattr(app_settings, "SYNC_TOTAL_REQUEST_BUDGET", 0.05)
    monkeypatch.setattr(main, "_http_client", object())
    cache = SyncCache()
    monkeypatch.setattr(main, "_sync_cache", cache)

    orch = main._build_sync_orchestrator()
    assert orch is not None

    released = asyncio.Event()

    class _Slow:
        async def resolve_with_provenance(self, query):
            await released.wait()
            return ResolvedReference(BIG_REF.decode(), kind="team")

    orch._external_strategy = _Slow()
    orch._sync_service = _FakeSyncService()
    monkeypatch.setattr(main, "_get_sync_orchestrator", lambda: orch)

    out = await main._maybe_sync_subtitle(_arabic_bytes(), _meta(), "t", True)
    assert out == _arabic_bytes()

    # Let the detached sync finish and verify it populated the cache.
    released.set()
    for _ in range(50):
        await asyncio.sleep(0.01)
        if any(k.startswith("final_sub:") for k in cache._local):
            break
    assert any(k.startswith("final_sub:") for k in cache._local)


@pytest.mark.asyncio
async def test_orchestrator_negative_cache_skips_provider_fanout(monkeypatch):
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
    cache = SyncCache()
    orch = _orchestrator(sync_cache=cache)
    first = await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True)
    assert first == _arabic_bytes()

    resolution_key = orch._flight_key(_meta(), "t", _arabic_bytes())
    assert await cache.is_failed(resolution_key) is True

    orch2 = _orchestrator(
        embedded_strategy=_FakeStrategy(BIG_REF.decode()), sync_cache=cache
    )
    second = await orch2.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True)
    assert second == _arabic_bytes()
    assert orch2._embedded_strategy.calls == 0


@pytest.mark.asyncio
async def test_external_reference_remux_target_is_retail_pair():
    """A REMUX stream against a BluRay reference must set bluray_match."""
    import io
    import zipfile

    from app.services.sync.external_strategy import ExternalExactStrategy

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("Show.S01E01.srt", BIG_REF.decode())

    class _FakeProvider:
        name = "fake"

        async def download_archive(self, url, api_key=None):
            return buffer.getvalue()

    class _Release:
        release_name = "Show.S01E01.2160p.BluRay.REMUX-GRP.srt"
        download_url = "https://example/fake.zip"

    query = ReferenceQuery(
        imdb_id="tt1",
        media_type="series",
        season=1,
        episode=1,
        target_filename="Show.S01E01.2160p.REMUX.HEVC-GRP.mkv",
    )
    strategy = ExternalExactStrategy(subdl_provider=None, subsource_provider=None)
    resolved = await strategy._download_reference([_Release()], _FakeProvider(), None, query)
    assert resolved.text
    assert resolved.kind == "edition"
    assert resolved.bluray_match is True


@pytest.mark.asyncio
async def test_orchestrator_marks_opaque_edition_target_as_relaxed(monkeypatch):
    """A hash-only target must reach the strict timeline gate as relaxed."""
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)

    def _meta_with(filename):
        return {
            "imdb_id": "tt1", "media_type": "series", "season": 1, "episode": 1,
            "lang": "ara", "target_filename": filename,
        }

    orch = _orchestrator(external_strategy=_FakeStrategy(BIG_REF.decode(), kind="edition"))
    await orch.evaluate_and_sync(
        _arabic_bytes(), _meta_with("e4WcFo4Tz5J8PoFwiBfP880XsBuHk4dS.mkv"), "t", True
    )
    assert orch._sync_service.relaxed == [True]

    orch = _orchestrator(external_strategy=_FakeStrategy(BIG_REF.decode(), kind="edition"))
    await orch.evaluate_and_sync(
        _arabic_bytes(), _meta_with("Show.S01E01.1080p.WEB-DL-GRP.mkv"), "t", True
    )
    assert orch._sync_service.relaxed == [False]


@pytest.mark.asyncio
async def test_orchestrator_logs_sync_cache_hit(monkeypatch, caplog):
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
    orch = _orchestrator(external_strategy=_FakeStrategy(BIG_REF.decode()))
    await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "sub123", True)

    orch2 = _orchestrator(
        external_strategy=_FakeStrategy("x"), sync_cache=orch._sync_cache
    )
    with caplog.at_level("INFO"):
        out = await orch2.evaluate_and_sync(_arabic_bytes(), _meta(), "sub123", True)
    assert b"synced" in out
    assert orch2._external_strategy.calls == 0
    assert "cache HIT for sub=sub123" in caplog.text
