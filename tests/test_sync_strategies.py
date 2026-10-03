"""Tests for the sync strategies (external exact-match, hash-exact) and the orchestrator."""

from types import SimpleNamespace

import pytest

from app.services.sync.orchestrator import SyncOrchestrator, build_synced_cache_key
from app.services.sync.query import ReferenceQuery
from app.services.sync_cache import SyncCache

BIG_REF = (
    "1\n00:00:01,000 --> 00:00:02,000\n" + ("reference line\n" * 600) + "\n"
).encode()


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


class _StubAnalyzer:
    """Controls the synchronization outcome so caching tests test *caching*.

    ``test_sync_strategies.py`` asserts orchestration and cache behaviour, but
    the fixtures it uses are synthetic and the real analyzer legitimately
    REJECTS their output (cue loss). Letting the analyzer decide meant these
    tests were really asserting "a rejected artifact is re-served as a warm
    cache hit", which is the defect this stub removes. Analyzer arithmetic has
    its own dedicated tests; the safety direction (a non-positive result is
    never re-served) is proven against the real analyzer in
    ``test_unverified_payload_not_reused.py``.
    """

    def __init__(self, state, verification="verified"):
        from app.services.sync.alignment import SubtitleEvaluation

        self._evaluation = SubtitleEvaluation(
            sync_state=state,
            verification=verification,
            sync_confidence=0.95,
            verification_confidence=0.9,
            median_offset_ms=10.0,
            mad_offset_ms=5.0,
            p95_offset_ms=20.0,
            content_match_score=0.9,
        )

    def analyze(self, *args, **kwargs):
        return self._evaluation


def _stub_analyzer(orch, state=None, verification="verified"):
    """Attach a stub analyzer; defaults to a verified, reusable outcome."""
    from app.services.sync.alignment import SyncState, VerificationAvailability

    if state is None:
        state = SyncState.VERIFIED_SYNCED
    if isinstance(verification, str):
        verification = VerificationAvailability(verification)
    orch._analyzer = _StubAnalyzer(state, verification)
    return orch


def _arabic_bytes(n=6):
    blocks = []
    for i in range(n):
        blocks.append(f"{i + 1}\n00:00:{i * 2 + 1:02d},000 --> 00:00:{i * 2 + 2:02d},000\nمرحبا")
    return ("\n\n".join(blocks) + "\n").encode()


def _orchestrator(**overrides):
    params = {
        "hash_strategy": _FakeStrategy(None),
        "external_strategy": _FakeStrategy(None),
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
    # An unknown language is the only remaining skip: with no language there is
    # nothing to align against. A specific non-Arabic language is NOT skipped --
    # that gate was product scope, not a technical requirement.
    assert await orch.evaluate_and_sync(_arabic_bytes(), {"lang": ""}, "t", True) == _arabic_bytes()
    assert orch._hash_strategy.calls == 0 and orch._external_strategy.calls == 0

    # A non-Arabic target now enters the pipeline and reaches the strategies.
    await orch.evaluate_and_sync(_arabic_bytes(), {"lang": "eng"}, "t", True)
    assert orch._hash_strategy.calls + orch._external_strategy.calls >= 1


@pytest.mark.asyncio
async def test_orchestrator_prefers_external_then_caches(monkeypatch):
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
    reference = BIG_REF.decode()
    # Tier order is external exact-match -> hash: the external reference wins.
    orch = _orchestrator(
        external_strategy=_FakeStrategy(reference),
        hash_strategy=_FakeStrategy("should-not-be-used"),
    )
    _stub_analyzer(orch)
    out = await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True)
    assert b"synced" in out
    assert orch._external_strategy.calls == 1
    assert orch._hash_strategy.calls == 0
    assert orch._sync_service.calls == 1

    # A warm result is served before reference resolution or alass.
    orch2 = _orchestrator(
        external_strategy=_FakeStrategy("x"),
        hash_strategy=_FakeStrategy("x"),
        sync_cache=orch._sync_cache,
    )
    out2 = await orch2.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True)
    assert out2 == out
    assert orch2._external_strategy.calls == 0 and orch2._hash_strategy.calls == 0
    assert orch2._sync_service.calls == 0


@pytest.mark.asyncio
async def test_orchestrator_uses_external_tier_first(monkeypatch, caplog):
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
    orch = _orchestrator(external_strategy=_FakeStrategy(BIG_REF.decode()))
    # Trusted outcome: this test is about tier ordering, not verification. The
    # real analyzer rejects the fake service's 1-cue output (cue loss), and an
    # untrusted result is deliberately not served.
    _stub_analyzer(orch)
    with caplog.at_level("INFO"):
        out = await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True)
    assert b"synced" in out
    assert orch._external_strategy.calls == 1 and orch._hash_strategy.calls == 0

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
        external_strategy=_FakeStrategy(exc=RuntimeError("boom")),
        hash_strategy=_FakeStrategy(BIG_REF.decode()),
    )
    # Trusted outcome: this test is about orchestration, not verification.
    # The real analyzer rejects the fake service's output, and an untrusted
    # result is deliberately not served.
    _stub_analyzer(orch)
    out = await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True)
    assert b"synced" in out


@pytest.mark.asyncio
async def test_orchestrator_keys_isolate_distinct_payloads(monkeypatch):
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
    orch = _orchestrator(external_strategy=_FakeStrategy(BIG_REF.decode()))
    await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True)
    other = _arabic_bytes().replace("مرحبا".encode(), "أهلا".encode())
    await orch.evaluate_and_sync(other, _meta(), "t", True)
    # Same subtitle ID, different bytes: resolved twice, no false cache hit.
    assert orch._external_strategy.calls == 2


@pytest.mark.asyncio
async def test_orchestrator_falls_back_to_next_strategy_on_sync_failure(monkeypatch, caplog):
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)

    class FailingEditionSyncService(_FakeSyncService):
        async def sync_async(self, target_srt, reference, decision_kind="edition", **kwargs):
            self.calls += 1
            if decision_kind == "edition":
                # Reference fails validation (e.g. collapsed cues)
                return None
            return "1\n00:00:09,000 --> 00:00:10,000\nsynced via hash\n"

    sync_svc = FailingEditionSyncService()
    orch = _orchestrator(
        external_strategy=_FakeStrategy(BIG_REF.decode(), kind="edition"),
        hash_strategy=_FakeStrategy(BIG_REF.decode(), kind="team"),
        sync_service=sync_svc,
    )
    # Trusted outcome: this test is about strategy fallback, not verification.
    _stub_analyzer(orch)

    with caplog.at_level("WARNING"):
        out = await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True)

    assert b"synced via hash" in out
    assert orch._external_strategy.calls == 1
    assert orch._hash_strategy.calls == 1
    assert sync_svc.calls == 2
    assert (
        "external exact-match strategy reference failed sync/validation -> falling back to next strategy"
        in caplog.text
    )


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
        external_strategy=_FakeStrategy(BIG_REF.decode(), kind="edition"),
    )
    await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True)
    assert orch._sync_service.kinds == ["edition"]

    orch = _orchestrator(hash_strategy=_FakeStrategy(BIG_REF.decode(), kind="hash"))
    await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True)
    assert orch._sync_service.kinds == ["hash"]


@pytest.mark.asyncio
async def test_orchestrator_withholds_untrusted_alass_output(monkeypatch):
    """A zero-exit alass output the verifier distrusts must NOT reach the client.

    Delivery fail-open, proven on a real Whiplash 2160p REMUX request: alass
    exited 0, the analyzer returned ``unverified``/``unknown`` with the explicit
    reason "alass output not trustworthy" (residual p95 4200 ms against a
    2000 ms bound), yet the mangled output was returned -- displacing a
    serviceable original by up to +293 s.

    The analyzer is the authority on trust. This test uses the REAL analyzer, so
    the rejection is genuine rather than stubbed.
    """
    import subprocess

    from app.config import settings as app_settings
    from app.services.sync_service import SubtitleSyncService

    # Reference dialogue starts 1s after the target's: close enough to pass the
    # first-dialogue sanity gate yet misaligned, so alass runs.
    ref = "".join(
        f"{i + 1}\n00:00:{i * 2 + 2:02d},000 --> 00:00:{i * 2 + 3:02d},000\nref {i}\n\n"
        for i in range(6)
    )

    def _shifted_run(command, capture_output=True, timeout=None):
        # alass "succeeds" but with a +69s shift on the first cue: structurally
        # valid (cue count preserved, monotonic) yet wildly untrustworthy.
        drifted = "".join(
            f"{i + 1}\n00:01:{10 + i * 2:02d},000 --> 00:01:{12 + i * 2:02d},000\nsynced {i}\n\n"
            for i in range(6)
        )
        with open(command[3], "w", encoding="utf-8") as handle:
            handle.write(drifted)
        return SimpleNamespace(returncode=0, stderr=b"", stdout=b"")

    monkeypatch.setattr(subprocess, "run", _shifted_run)
    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)

    original = _arabic_bytes()
    for kind in ("edition", "team"):
        orch = _orchestrator(
            external_strategy=_FakeStrategy(ref, kind=kind),
            sync_service=SubtitleSyncService(),
        )
        out = await orch.evaluate_and_sync(original, _meta(), "t", True)

        ev = orch._last_evaluation
        assert ev is not None
        assert ev.verification.value != "verified", (
            "fixture drifted further than expected; the negative control lost its premise"
        )
        assert b"synced" not in out, "untrusted alass output was served to the client"
        assert out == original, "the original provider subtitle must be served instead"


@pytest.mark.asyncio
async def test_aligned_pair_skips_alass_but_caches(monkeypatch):
    """Content already aligned (median offset < 0.2s): serve original, no alass."""
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
    ref = "".join(
        f"{i + 1}\n00:00:{i * 2 + 1:02d},000 --> 00:00:{i * 2 + 2:02d},000\nreference line {i}\n\n"
        for i in range(6)
    )
    orch = _orchestrator(external_strategy=_FakeStrategy(ref, kind="edition"))
    _stub_analyzer(orch)
    out = await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True)
    assert out == _arabic_bytes()
    assert orch._sync_service.calls == 0
    # The aligned result is cached: a repeat is a positive hit, no providers.
    out2 = await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True)
    assert out2 == _arabic_bytes()
    assert orch._external_strategy.calls == 1


@pytest.mark.asyncio
async def test_group_names_do_not_gate_sync(monkeypatch):
    """Release-group equality never skips resolution; content decides."""
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
    assert orch._hash_strategy.calls == 1 and orch._external_strategy.calls == 1


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
    # Trusted outcome: this test is about request coalescing, not verification.
    _stub_analyzer(orch)
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
    # Trusted outcome: this test is about orchestration, not verification.
    # The real analyzer rejects the fake service's output, and an untrusted
    # result is deliberately not served.
    _stub_analyzer(orch)
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
    # Trusted outcome: this test is about the cache sidecar, not verification.
    _stub_analyzer(orch)
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
    # Trusted outcome: this test is about coalescing on the app singleton.
    _stub_analyzer(orch)
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
    # Trusted outcome: this test is about coalescing on the app singleton.
    _stub_analyzer(orch)
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
async def test_orchestrator_skips_reference_failing_cue_sanity(monkeypatch):
    """A mistimed reference (first dialogue 45s off) is skipped, never alass'd."""
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
    target = (
        "1\n00:00:05,000 --> 00:00:06,500\nمرحبا\n\n"
        "2\n00:00:07,000 --> 00:00:08,500\nمرحبا\n"
    ).encode()
    ref = (
        "1\n00:00:50,000 --> 00:00:51,500\nreference line one\n\n"
        "2\n00:00:52,000 --> 00:00:53,500\nreference line two\n"
    )
    orch = _orchestrator(external_strategy=_FakeStrategy(ref, kind="edition"))
    out = await orch.evaluate_and_sync(target, _meta(), "t", True)
    assert out == target
    assert orch._sync_service.calls == 0


@pytest.mark.asyncio
async def test_orchestrator_allows_realistic_cross_source_offset(monkeypatch):
    """First-dialogue deltas within ±20s run alass (uniform intro/bumper shift)."""
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
    target = (
        "1\n00:00:05,000 --> 00:00:06,500\nمرحبا\n\n"
        "2\n00:00:07,000 --> 00:00:08,500\nمرحبا\n"
    ).encode()
    # Reference dialogue ~4.7s later (Mad Men shape): inside the execution window.
    ref = (
        "1\n00:00:09,700 --> 00:00:11,200\nreference line one\n\n"
        "2\n00:00:11,700 --> 00:00:13,200\nreference line two\n"
    )
    orch = _orchestrator(external_strategy=_FakeStrategy(ref, kind="edition"))
    # Trusted outcome: this test is about orchestration, not verification.
    # The real analyzer rejects the fake service's output, and an untrusted
    # result is deliberately not served.
    _stub_analyzer(orch)
    out = await orch.evaluate_and_sync(target, _meta(), "t", True)
    assert b"synced" in out
    assert orch._sync_service.calls == 1


@pytest.mark.asyncio
async def test_orchestrator_skips_huge_first_dialogue_delta(monkeypatch):
    """First-dialogue deltas beyond ±20s are a different cut: skip, serve original."""
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
    target = (
        "1\n00:00:05,000 --> 00:00:06,500\nمرحبا\n\n"
        "2\n00:00:07,000 --> 00:00:08,500\nمرحبا\n"
    ).encode()
    ref = (
        "1\n00:00:35,000 --> 00:00:36,500\nreference line one\n\n"
        "2\n00:00:37,000 --> 00:00:38,500\nreference line two\n"
    )
    orch = _orchestrator(external_strategy=_FakeStrategy(ref, kind="edition"))
    out = await orch.evaluate_and_sync(target, _meta(), "t", True)
    assert out == target
    assert orch._sync_service.calls == 0


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
        external_strategy=_FakeStrategy(BIG_REF.decode()), sync_cache=cache
    )
    second = await orch2.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True)
    assert second == _arabic_bytes()
    assert orch2._external_strategy.calls == 0


@pytest.mark.asyncio
async def test_failed_reference_short_circuits_repeat_request(monkeypatch, tmp_path):
    """A cue-sanity failure is remembered, so the next request does no downloads."""
    from app.config import settings as app_settings
    from app.models import SubtitleRelease
    from app.services.sync.cache import ReferenceDiskCache
    from app.services.sync.external_strategy import ExternalExactStrategy

    monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
    target = _dialogue(10_160, marker="reference line target", cues=30)
    # 400s out, past LARGE_OFFSET_MAX_SECONDS: the Large Offset Evidence Gate
    # refuses it outright, so this stays a genuine *pre-alass* rejection that
    # the reference-family cache can remember. A merely large-but-plausible
    # offset would instead be granted an alass trial and judged afterwards.
    wrong_cut = _dialogue(410_160, marker="reference line wrong cut", cues=30)
    release = SubtitleRelease(
        release_name="Dexter.S08E05.720p.BluRay.x264-NORDiC.srt",
        download_url="http://wrong-cut",
        provider="subdl",
        lang="eng",
    )

    class _CountingProvider:
        name = "subdl"

        def __init__(self):
            self.searches = 0
            self.downloaded: list[str] = []

        async def search_subtitles(self, **kwargs):
            self.searches += 1
            return [release]

        async def download_archive(self, url, api_key=None):
            self.downloaded.append(url)
            return wrong_cut

    provider = _CountingProvider()
    strategy = ExternalExactStrategy(
        subdl_provider=provider,
        subsource_provider=None,
        opensubtitles_provider=None,
        cache=ReferenceDiskCache(root=tmp_path / "refs", ttl=3600.0, min_bytes=100),
        min_bytes=100,
        timeout=1.0,
    )
    sync_cache = SyncCache()
    orch = SyncOrchestrator(
        hash_strategy=_FakeStrategy(None),
        external_strategy=strategy,
        sync_service=_FakeSyncService(),
        sync_cache=sync_cache,
    )
    meta = {
        "imdb_id": "tt0773262",
        "media_type": "series",
        "season": 8,
        "episode": 5,
        "lang": "ara",
        "target_filename": SEASON_PACK_TARGET,
    }

    first = await orch.evaluate_and_sync(target, meta, "wrong-cut-sub", True)
    assert first == target
    assert provider.searches == 1
    assert provider.downloaded == ["http://wrong-cut"]
    assert orch._sync_service.calls == 0

    resolution_key = orch._flight_key(meta, "wrong-cut-sub", target)
    assert await sync_cache.is_failed(resolution_key) is True

    second = await orch.evaluate_and_sync(target, meta, "wrong-cut-sub", True)
    assert second == target
    assert provider.searches == 1
    assert provider.downloaded == ["http://wrong-cut"]
    assert orch._sync_service.calls == 0


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
    _stub_analyzer(orch)
    await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "sub123", True)

    orch2 = _orchestrator(
        external_strategy=_FakeStrategy("x"), sync_cache=orch._sync_cache
    )
    with caplog.at_level("INFO"):
        out = await orch2.evaluate_and_sync(_arabic_bytes(), _meta(), "sub123", True)
    assert b"synced" in out
    assert orch2._external_strategy.calls == 0
    assert "cache HIT for sub=sub123" in caplog.text


@pytest.mark.asyncio
async def test_reference_selection_prefers_hash_matched_english(tmp_path):
    """A MovieHash-matched English reference beats strong text-only candidates."""
    from app.models import SubtitleRelease
    from app.services.sync.cache import ReferenceDiskCache
    from app.services.sync.external_strategy import TIER_HASH, ExternalExactStrategy, reference_tier

    class _Provider:
        def __init__(self, releases):
            self._releases = releases
            self.downloaded: list[str] = []

        async def search_subtitles(self, **kwargs):
            return list(self._releases)

        async def download_archive(self, url, api_key=None):
            self.downloaded.append(url)
            return BIG_REF

    target = (
        "Mad.Men.S01E02.Ladies.Room.REPACK.2160p.HMAX.WEB-DL.DDP5.1.DV.HDR.H.265-WADU.mkv"
    )
    text_only = SubtitleRelease(
        release_name="Mad.Men.S01E02.2160p.HMAX.WEB-DL.DDP5.1.H.265-WADU.srt",
        download_url="http://text", provider="subdl", lang="eng",
    )
    # Strong text-only match (same group + source), but no byte-exact proof.
    assert reference_tier(target, text_only) > TIER_HASH
    hashed = SubtitleRelease(
        release_name="Mad.Men.S01E02.720p.HDTV.x264-Scene.srt",
        download_url="http://hash", provider="opensubtitles", lang="eng",
        matched_by_hash=True,
    )
    subdl = _Provider([text_only])
    opensubtitles = _Provider([hashed])
    strategy = ExternalExactStrategy(
        subdl_provider=subdl,
        subsource_provider=None,
        opensubtitles_provider=opensubtitles,
        cache=ReferenceDiskCache(root=tmp_path / "refs", ttl=3600.0, min_bytes=100),
        timeout=1.0,
    )
    query = ReferenceQuery(
        imdb_id="tt0804503", media_type="series", season=1, episode=2, target_filename=target
    )
    resolved = await strategy.resolve_with_provenance(query)
    assert resolved.text is not None
    assert resolved.kind == "hash"
    assert opensubtitles.downloaded == ["http://hash"]
    assert subdl.downloaded == []

# --------------------------------------------------------------------------- #
# Season-pack handling: archives are unbundled to the target episode, while
# uncompressed cumulative packs are rejected. Episode-specific candidates stay
# preferred, and a rejected reference never aborts the whole sync.
# --------------------------------------------------------------------------- #

SEASON_PACK_TARGET = "Dexter.s8e05.This.little.piggy.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv"
SEASON_PACK_NAME = "Dexter.S08.BluRay.1080p.TrueHD.5.1.AVC.REMUX-FraMeSToR.srt"


def _ts(ms: int) -> str:
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, msec = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{msec:03d}"


def _dialogue(first_ms: int, marker: str = "E", cues: int = 120) -> bytes:
    """A full-length, sentence-shaped subtitle the cue-sanity gate can read.

    The first cue cluster anchors at ``first_ms``; the remaining cues pad the
    payload past the strategy's minimum size so it is not rejected as too small
    before validation ever runs.
    """
    line = f"I never thought I would find someone like you in my life {marker}"
    blocks = [
        f"{i + 1}\n{_ts(first_ms + i * 2000)} --> {_ts(first_ms + i * 2000 + 1500)}\n{line}\n"
        for i in range(cues)
    ]
    return "\n".join(blocks).encode()


def _season_zip(episodes: range | list[int]) -> bytes:
    """Build an in-memory season-pack ZIP holding one .srt per episode."""
    import io as _io
    import zipfile as _zipfile

    buf = _io.BytesIO()
    with _zipfile.ZipFile(buf, "w", _zipfile.ZIP_DEFLATED) as archive:
        for ep in episodes:
            archive.writestr(
                f"Dexter.S08E{ep:02d}.ar.srt", _dialogue(10_000 + ep * 1000, marker=f"E{ep:02d}")
            )
    return buf.getvalue()


def _provider_for(releases, payload):
    class _Provider:
        def __init__(self):
            self.downloaded: list[str] = []

        async def search_subtitles(self, **kwargs):
            return list(releases)

        async def download_archive(self, url, api_key=None):
            self.downloaded.append(url)
            return payload

    return _Provider()


def _series_query(season: int = 8, episode: int = 5):
    return ReferenceQuery(
        imdb_id="tt0773262",
        media_type="series",
        season=season,
        episode=episode,
        target_filename=SEASON_PACK_TARGET,
    )


# ------------------------- eligibility / ranking ---------------------------- #


def test_season_pack_is_demoted_below_episode_specific_candidate():
    """A pack stays a valid candidate but is ranked below an exact-episode track."""
    from app.models import SubtitleRelease
    from app.services.sync.external_strategy import _select_candidates_ranked

    season_pack = SubtitleRelease(
        release_name=SEASON_PACK_NAME,
        download_url="http://pack", provider="subdl", lang="ara",
    )
    episode_single = SubtitleRelease(
        release_name="Dexter.S08E05.720p.BluRay.x264-NORDiC",
        download_url="http://ep5", provider="opensubtitles", lang="eng",
    )
    other_episode = SubtitleRelease(
        release_name="Dexter.S08E09.720p.BluRay.x264-OTHER",
        download_url="http://ep9", provider="opensubtitles", lang="eng",
    )
    ranked = _select_candidates_ranked([season_pack, episode_single, other_episode], _series_query())
    names = [rel.release_name for rel in ranked]
    # Episode-specific first, pack retained as a demoted fallback, wrong episode gone.
    assert names[0] == episode_single.release_name
    assert names == [episode_single.release_name, season_pack.release_name]


def test_season_pack_still_eligible_when_no_episode_specific_reference_exists():
    """Graceful degradation: with no exact-episode candidate, the pack is still used."""
    from app.models import SubtitleRelease
    from app.services.sync.external_strategy import _select_candidates_ranked

    season_pack = SubtitleRelease(
        release_name=SEASON_PACK_NAME,
        download_url="http://pack", provider="subdl", lang="ara",
    )
    other_episode = SubtitleRelease(
        release_name="Dexter.S08E09.720p.BluRay.x264-OTHER",
        download_url="http://ep9", provider="opensubtitles", lang="eng",
    )
    ranked = _select_candidates_ranked([season_pack, other_episode], _series_query())
    assert [rel.release_name for rel in ranked] == [season_pack.release_name]


# ---------------------- archive unbundling (in-memory) ---------------------- #


@pytest.mark.asyncio
async def test_season_pack_archive_unbundles_target_episode(tmp_path):
    """A season-pack ZIP is sliced to S08E05 and only that episode is served."""
    from app.models import SubtitleRelease
    from app.services.sync.cache import ReferenceDiskCache
    from app.services.sync.external_strategy import ExternalExactStrategy

    pack = SubtitleRelease(
        release_name=SEASON_PACK_NAME,
        download_url="http://pack", provider="subdl", lang="ara",
    )
    subdl = _provider_for([pack], _season_zip(range(1, 13)))
    strategy = ExternalExactStrategy(
        subdl_provider=subdl,
        subsource_provider=None,
        opensubtitles_provider=None,
        cache=ReferenceDiskCache(root=tmp_path / "refs", ttl=3600.0, min_bytes=100),
        min_bytes=100,
        timeout=1.0,
    )
    resolved = await strategy.resolve_with_provenance(_series_query())
    assert resolved.text is not None
    # Only the requested episode's dialogue came out of the 12-member archive.
    assert "E05" in resolved.text
    for other in ("E01", "E02", "E03", "E06", "E12"):
        assert f" {other}" not in resolved.text
    assert subdl.downloaded == ["http://pack"]


@pytest.mark.asyncio
async def test_season_pack_archive_missing_target_episode_falls_back(tmp_path):
    """A pack that lacks the target episode is skipped for the next candidate."""
    from app.models import SubtitleRelease
    from app.services.sync.cache import ReferenceDiskCache
    from app.services.sync.external_strategy import ExternalExactStrategy

    # Higher-scoring pack, but it only contains episodes 1-3.
    pack_missing = SubtitleRelease(
        release_name="Dexter.S08.1080p.BluRay.TrueHD.5.1.AVC.REMUX-FraMeSToR.srt",
        download_url="http://pack-missing", provider="subdl", lang="ara",
    )
    # Lower-scoring pack that does contain S08E05.
    pack_with_ep5 = SubtitleRelease(
        release_name="Dexter.S08.720p.WEB-DL.x264-Other.srt",
        download_url="http://pack-ep5", provider="subdl", lang="ara",
    )
    subdl = _provider_for([pack_missing, pack_with_ep5], b"")
    # Each candidate gets its own payload based on which URL was requested.
    payloads = {
        "http://pack-missing": _season_zip([1, 2, 3]),
        "http://pack-ep5": _season_zip([4, 5, 6]),
    }

    async def _download(url, api_key=None):
        subdl.downloaded.append(url)
        return payloads[url]

    subdl.download_archive = _download

    strategy = ExternalExactStrategy(
        subdl_provider=subdl,
        subsource_provider=None,
        opensubtitles_provider=None,
        cache=ReferenceDiskCache(root=tmp_path / "refs", ttl=3600.0, min_bytes=100),
        min_bytes=100,
        timeout=1.0,
    )
    resolved = await strategy.resolve_with_provenance(_series_query())
    assert resolved.text is not None
    assert resolved.candidate == pack_with_ep5.release_name
    # The incomplete pack was tried, then the next one supplied E05.
    assert subdl.downloaded == ["http://pack-missing", "http://pack-ep5"]
    assert "E05" in resolved.text
    assert "E04" not in resolved.text


@pytest.mark.asyncio
async def test_season_pack_archive_extracts_with_alternate_episode_naming(tmp_path):
    """Members tagged 8x05 / S8E5 are recognised as the target episode too."""
    import io as _io
    import zipfile as _zipfile

    from app.models import SubtitleRelease
    from app.services.sync.cache import ReferenceDiskCache
    from app.services.sync.external_strategy import ExternalExactStrategy

    buf = _io.BytesIO()
    with _zipfile.ZipFile(buf, "w", _zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("Dexter.8x04.ar.srt", _dialogue(10_000, marker="X04"))
        archive.writestr("Dexter.8x05.ar.srt", _dialogue(10_000, marker="X05"))
        archive.writestr("Dexter.S8E06.ar.srt", _dialogue(10_000, marker="X06"))

    pack = SubtitleRelease(
        release_name="Dexter.S08.BluRay.REMUX.ar.zip",
        download_url="http://pack", provider="subdl", lang="ara",
    )
    subdl = _provider_for([pack], buf.getvalue())
    strategy = ExternalExactStrategy(
        subdl_provider=subdl,
        subsource_provider=None,
        opensubtitles_provider=None,
        cache=ReferenceDiskCache(root=tmp_path / "refs", ttl=3600.0, min_bytes=100),
        min_bytes=100,
        timeout=1.0,
    )
    resolved = await strategy.resolve_with_provenance(_series_query())
    assert resolved.text is not None
    assert "X05" in resolved.text
    assert "X04" not in resolved.text
    assert "X06" not in resolved.text


# ------------------- true cumulative (concatenated) packs -------------------- #


def test_cumulative_pack_detector_flags_multi_episode_timelines():
    from app.services.sync.decode import looks_like_cumulative_pack

    assert looks_like_cumulative_pack(_dialogue(10_000).decode()) is False
    # Two episodes merged: the timeline restarts near 00:00 for the second one.
    cumulative = _dialogue(10_000, marker="P1").decode() + _dialogue(5_000, marker="P2").decode()
    assert looks_like_cumulative_pack(cumulative) is True
    # A single very long feature also trips the span guard.
    assert looks_like_cumulative_pack(_dialogue(0, cues=6000).decode()) is True


@pytest.mark.asyncio
async def test_cumulative_single_file_pack_is_rejected(tmp_path):
    """An uncompressed multi-episode timeline is rejected, next candidate used."""
    from app.models import SubtitleRelease
    from app.services.sync.cache import ReferenceDiskCache
    from app.services.sync.external_strategy import ExternalExactStrategy

    # Higher-scoring pack served as one bare .srt spanning two episodes.
    cumulative = SubtitleRelease(
        release_name="Dexter.S08.1080p.BluRay.TrueHD.5.1.AVC.REMUX-FraMeSToR.srt",
        download_url="http://pack-cumulative", provider="subdl", lang="ara",
    )
    # Lower-scoring pack that unbundles cleanly.
    bundled = SubtitleRelease(
        release_name="Dexter.S08.720p.WEB-DL.x264-Other.srt",
        download_url="http://pack-bundled", provider="subdl", lang="ara",
    )
    subdl = _provider_for([cumulative, bundled], b"")
    payloads = {
        # A bare .srt (not an archive) whose timeline runs on into the next episode.
        "http://pack-cumulative": _dialogue(10_000, marker="P1") + _dialogue(4_000, marker="P2"),
        "http://pack-bundled": _season_zip([5]),
    }

    async def _download(url, api_key=None):
        subdl.downloaded.append(url)
        return payloads[url]

    subdl.download_archive = _download

    strategy = ExternalExactStrategy(
        subdl_provider=subdl,
        subsource_provider=None,
        opensubtitles_provider=None,
        cache=ReferenceDiskCache(root=tmp_path / "refs", ttl=3600.0, min_bytes=100),
        min_bytes=100,
        timeout=1.0,
    )
    resolved = await strategy.resolve_with_provenance(_series_query())
    assert resolved.text is not None
    assert resolved.candidate == bundled.release_name
    # The cumulative file was decoded, rejected, and the next pack supplied E05.
    assert subdl.downloaded == ["http://pack-cumulative", "http://pack-bundled"]
    assert "P1" not in resolved.text
    assert "E05" in resolved.text


@pytest.mark.asyncio
async def test_cumulative_pack_allowed_for_movie_queries(tmp_path):
    """Long-form movie references are not subject to the episode-pack guard."""
    from app.models import SubtitleRelease
    from app.services.sync.cache import ReferenceDiskCache
    from app.services.sync.external_strategy import ExternalExactStrategy

    movie = SubtitleRelease(
        release_name="Interstellar.2014.2160p.WEB-DL.DDP5.1.HDR.H.265-FLUX.srt",
        download_url="http://movie", provider="subdl", lang="eng",
    )
    subdl = _provider_for([movie], _dialogue(30_000, cues=4000))
    strategy = ExternalExactStrategy(
        subdl_provider=subdl,
        subsource_provider=None,
        opensubtitles_provider=None,
        cache=ReferenceDiskCache(root=tmp_path / "refs", ttl=3600.0, min_bytes=100),
        min_bytes=100,
        timeout=1.0,
    )
    query = ReferenceQuery(
        imdb_id="tt0816692", media_type="movie", target_filename=movie.release_name
    )
    resolved = await strategy.resolve_with_provenance(query)
    assert resolved.text is not None


# --------------------------- fall-through / caching ------------------------- #


@pytest.mark.asyncio
async def test_rejected_reference_falls_through_to_next_candidate(tmp_path):
    """A weaker reference must not be promoted because it fits the candidate.

    This is the production incident, reproduced exactly: target
    ``Dexter.s8e05.This.little.piggy.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv``, a
    candidate whose first dialogue sits at ~106.95s, an exact-release-group
    reference at ~10.51s, and a weaker 720p reference that happens to sit at
    ~106.95s.

    The old policy returned the 720p reference, because it was the only one that
    satisfied the requested subtitle. That is the circularity: the candidate
    chose the reference. The correct outcome is to refuse, and to say why.
    """
    from app.models import SubtitleRelease
    from app.services.sync.cache import ReferenceDiskCache
    from app.services.sync.external_strategy import ExternalExactStrategy

    bad_ref = _dialogue(10_510)  # first dialogue ~96s before the candidate's
    good_ref = _dialogue(106_950)  # correct for the candidate, wrong for the target

    pack = SubtitleRelease(
        release_name=SEASON_PACK_NAME,
        download_url="http://pack", provider="subdl", lang="ara",
    )
    exact = SubtitleRelease(
        release_name="Dexter.S08E05.1080p.BluRay.x264-PiR8.srt",
        download_url="http://exact", provider="subdl", lang="eng",
    )
    weaker = SubtitleRelease(
        release_name="Dexter.S08E05.720p.BluRay.x264-NORDiC",
        download_url="http://720p", provider="subsource", lang="eng",
    )
    target_text = _dialogue(106_950).decode()

    subdl = _provider_for([pack, exact], bad_ref)
    subsource = _provider_for([weaker], good_ref)
    strategy = ExternalExactStrategy(
        subdl_provider=subdl,
        subsource_provider=subsource,
        opensubtitles_provider=None,
        cache=ReferenceDiskCache(root=tmp_path / "refs", ttl=3600.0, min_bytes=100),
        min_bytes=100,
        timeout=1.0,
    )

    def _validator(reference_text: str) -> bool:
        from app.services.subtitle_matcher import (
            FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
            validate_cue_sanity,
        )

        return bool(
            validate_cue_sanity(
                target_text,
                reference_text,
                threshold_ms=FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
            )["ok"]
        )

    resolved = await strategy.resolve_with_provenance(_series_query(), update_validator=_validator)

    # The exact-group reference is the target's anchor. It disagrees with the
    # candidate, so the candidate is a different cut and nothing is
    # synchronized. The 720p reference that fits the candidate is suppressed.
    assert resolved.text is None
    assert resolved.kind == "abort"
    assert subdl.downloaded == ["http://exact"]
    assert subsource.downloaded == ["http://720p"]


@pytest.mark.asyncio
async def test_equal_strength_reference_still_falls_through(tmp_path):
    """A mislabeled reference must not abort the sync when its equal stands.

    The fall-through behaviour is preserved where it is sound: two references
    of the same target strength (same release group) are interchangeable, so if
    the first is mislabeled the second may serve. What is forbidden is
    substituting a *weaker* reference, which is the previous test.
    """
    from app.models import SubtitleRelease
    from app.services.sync.cache import ReferenceDiskCache
    from app.services.sync.external_strategy import ExternalExactStrategy

    mislabeled = _dialogue(10_510)
    correct = _dialogue(106_950)
    target_text = _dialogue(106_950).decode()

    first = SubtitleRelease(
        release_name="Dexter.S08E05.1080p.BluRay.x264-PiR8.srt",
        download_url="http://first", provider="subdl", lang="eng",
    )
    second = SubtitleRelease(
        release_name="Dexter.S08E05.1080p.BluRay.TrueHD5.1.AVC-PiR8.srt",
        download_url="http://second", provider="subsource", lang="eng",
    )

    subdl = _provider_for([first], mislabeled)
    subsource = _provider_for([second], correct)
    strategy = ExternalExactStrategy(
        subdl_provider=subdl,
        subsource_provider=subsource,
        opensubtitles_provider=None,
        cache=ReferenceDiskCache(root=tmp_path / "refs", ttl=3600.0, min_bytes=100),
        min_bytes=100,
        timeout=1.0,
    )

    def _validator(reference_text: str) -> bool:
        from app.services.subtitle_matcher import (
            FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
            validate_cue_sanity,
        )

        return bool(
            validate_cue_sanity(
                target_text,
                reference_text,
                threshold_ms=FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
            )["ok"]
        )

    resolved = await strategy.resolve_with_provenance(_series_query(), update_validator=_validator)
    assert resolved.text is not None
    assert resolved.candidate == second.release_name


@pytest.mark.asyncio
async def test_stale_cached_reference_is_revalidated_and_replaced(tmp_path):
    """A cached reference that fails cue-sanity is discarded and re-resolved."""
    from app.models import SubtitleRelease
    from app.services.sync.cache import ReferenceDiskCache
    from app.services.sync.external_strategy import ExternalExactStrategy

    bad_ref = _dialogue(10_510)  # first dialogue ~96s before the target's
    good_ref = _dialogue(106_950)  # correct cut

    pack = SubtitleRelease(
        release_name=SEASON_PACK_NAME,
        download_url="http://pack", provider="subdl", lang="ara",
    )
    single = SubtitleRelease(
        release_name="Dexter.S08E05.720p.BluRay.x264-NORDiC",
        download_url="http://ep5", provider="subsource", lang="eng",
    )
    cache = ReferenceDiskCache(root=tmp_path / "refs", ttl=3600.0, min_bytes=100)
    # Seed the cache with a stale reference that fails cue-sanity.
    cache.set(_series_query(), "subdl", bad_ref.decode(), kind="edition", candidate="http://pack")

    subdl = _provider_for([pack], bad_ref)
    subsource = _provider_for([single], good_ref)
    strategy = ExternalExactStrategy(
        subdl_provider=subdl,
        subsource_provider=subsource,
        opensubtitles_provider=None,
        cache=cache,
        min_bytes=100,
        timeout=1.0,
    )
    target_text = _dialogue(106_950).decode()

    def _validator(reference_text: str) -> bool:
        from app.services.subtitle_matcher import (
            FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
            validate_cue_sanity,
        )

        return bool(
            validate_cue_sanity(
                target_text,
                reference_text,
                threshold_ms=FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
            )["ok"]
        )

    resolved = await strategy.resolve_with_provenance(_series_query(), update_validator=_validator)
    assert resolved.text is not None
    assert subsource.downloaded == ["http://ep5"]
    assert resolved.candidate == single.release_name
    # The rejected reference must be gone from disk, not merely bypassed: a
    # stale entry otherwise survives the 30-day TTL and is re-read (and
    # re-rejected) on every later request for this stem.
    assert cache.get(_series_query()) is not None  # the good one replaced it
    assert bad_ref.decode() not in {
        (p.read_text(encoding="utf-8"))
        for p in cache.root.glob(f"{_series_query().cache_stem}_*.srt")
    }


@pytest.mark.asyncio
async def test_rejected_candidate_is_not_redownloaded_after_cache_eviction(tmp_path):
    """A candidate proven bad for this stem is not fetched again after an eviction.

    This is the exact wasteful path seen in production: a cached reference
    fails cue-sanity, is evicted, and the deterministic fan-out then restarts
    from the same ranked list and re-downloads the very candidate that just
    failed. The memo must stop that second download.
    """
    from app.models import SubtitleRelease
    from app.services.sync.cache import ReferenceDiskCache
    from app.services.sync.external_strategy import ExternalExactStrategy

    bad_ref = _dialogue(10_510)  # ~96s off: a different cut
    good_ref = _dialogue(106_950)  # correct cut

    # `bad` is a tighter release-name match to the target (identical group
    # token) so it outranks `good` and is always attempted first, then walked
    # past.
    bad = SubtitleRelease(
        release_name="Dexter.S08E05.1080p.BluRay.TrueHD5.1.AVC-NORDiC.srt",
        download_url="http://bad", provider="subdl", lang="eng",
    )
    good = SubtitleRelease(
        release_name="Dexter.S08E05.720p.BluRay.x264-DEMAND.srt",
        download_url="http://good", provider="subdl", lang="eng",
    )

    class _Provider:
        name = "subdl"

        def __init__(self):
            self.downloaded: list[str] = []

        async def search_subtitles(self, **kwargs):
            return [bad, good]

        async def download_archive(self, url, api_key=None):
            self.downloaded.append(url)
            return bad_ref if url == "http://bad" else good_ref

    provider = _Provider()
    cache = ReferenceDiskCache(root=tmp_path / "refs", ttl=3600.0, min_bytes=100)
    strategy = ExternalExactStrategy(
        subdl_provider=provider,
        cache=cache,
        min_bytes=100,
        timeout=1.0,
    )
    target_text = _dialogue(106_950).decode()

    def _validator(reference_text: str) -> bool:
        from app.services.subtitle_matcher import (
            FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
            validate_cue_sanity,
        )

        return bool(
            validate_cue_sanity(
                target_text,
                reference_text,
                threshold_ms=FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
            )["ok"]
        )

    query = _series_query()
    # Seed the cache with the reference that does NOT fit this target, exactly
    # as a previous target for the same video would have written it.
    cache.set(query, "subdl", bad_ref.decode(), kind="edition", candidate=bad.release_name)

    first = await strategy.resolve_with_provenance(query, update_validator=_validator)
    assert first.text is not None
    # The stale entry is evicted, the bad candidate is tried once and rejected,
    # then the good one wins.
    assert provider.downloaded == ["http://bad", "http://good"]

    # Now simulate the next request: evict the freshly cached good entry (as a
    # differing target would), forcing a full re-resolve. The memo must stop
    # `http://bad` from being downloaded a second time. `http://good` is
    # re-fetched because it succeeded rather than being rejected, and the cache
    # entry that held it was just removed.
    cache.delete(query)
    second = await strategy.resolve_with_provenance(query, update_validator=_validator)
    assert second.text is not None
    assert provider.downloaded.count("http://bad") == 1, (
        f"the rejected candidate was re-downloaded after eviction: {provider.downloaded}"
    )


def test_reference_cache_delete_removes_payload_and_sidecar(tmp_path):
    """`delete` clears the SRT and its verdict sidecar, and reports whether it removed one."""
    from app.services.sync.cache import ReferenceDiskCache

    cache = ReferenceDiskCache(root=tmp_path / "refs", ttl=3600.0, min_bytes=100)
    query = _series_query()
    cache.set(query, "subdl", _dialogue(106_950).decode(), kind="edition", candidate="x")

    assert list(cache.root.glob(f"{query.cache_stem}_*.srt"))
    assert cache.delete(query) is True
    assert not list(cache.root.glob(f"{query.cache_stem}_*.srt"))
    # No orphaned sidecar is left behind to be re-read as provenance.
    assert not list(cache.root.glob(f"{query.cache_stem}_*.json"))
    # Deleting again is a no-op, not an error.
    assert cache.delete(query) is False
    assert cache.get(query) is None
