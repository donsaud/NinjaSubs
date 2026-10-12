"""Deferred scheduling, job ownership, and safe numeric telemetry."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.services.sync import orchestrator as om
from app.services.sync.external_strategy import ExternalExactStrategy
from app.services.sync.orchestrator import SyncOrchestrator
from app.services.sync.query import ResolvedReference
from app.services.sync.telemetry import SyncTiming, current_sync_timing


def subtitle(tag: str) -> str:
    return "\n".join(
        f"{i + 1}\n00:{i // 10:02}:{i % 10 * 6:02},000 --> "
        f"00:{i // 10:02}:{i % 10 * 6 + 2:02},000\n{tag} dialogue {i}\n"
        for i in range(24)
    )


TARGET = subtitle("target").encode()
META = {"imdb_id": "tt1234567", "lang": "ara", "target_filename": "Film.WEB.mkv"}


class External(ExternalExactStrategy):
    def __init__(self, cached=None, error=None):
        super().__init__(cache=None)
        self.cache = SimpleNamespace(get=lambda query: cached)
        self.calls = 0
        self.downloads = []
        self.error = error
        self.entered = asyncio.Event()
        self.release = None
        self.cleaned = False

    async def plan_candidates(self, query):
        self.calls += 1
        self.entered.set()
        try:
            if self.release is not None:
                await self.release.wait()
            if self.error is not None:
                raise self.error
            return [("A", self), ("B", self), ("C", self)]
        finally:
            self.cleaned = True

    def _download_allowed(self, candidate, provider, query):
        # This scheduling double supplies opaque tokens, not provider metadata.
        return True

    async def _download_candidate(self, candidate, provider, query):
        self.downloads.append(candidate)
        return ResolvedReference(subtitle(candidate), kind="edition", candidate=candidate)


class Service:
    async def sync_async(self, *args, **kwargs):
        return None


def setup(monkeypatch, external, accepted="A", hashed=None):
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True)
    monkeypatch.setattr(om, "measure_alignment_consistency", lambda *args, **kw: None)
    monkeypatch.setattr("app.services.sync.external_strategy._REFERENCE_DOWNLOAD_MIN_INTERVAL_SECONDS", 0)
    seen = []
    tails = []
    hash_strategy = None
    if hashed is not None:
        async def resolve(query):
            assert external.calls == 0
            return hashed
        hash_strategy = SimpleNamespace(resolve_with_provenance=resolve)
    orchestrator = SyncOrchestrator(
        external_strategy=external, hash_reference_strategy=hash_strategy, sync_service=Service()
    )

    async def affine(target, resolved, *args):
        seen.append(resolved.candidate)
        if resolved.candidate == accepted:
            timing = current_sync_timing.get()
            timing.winning_sync_strategy = "timeline_affine"
            return b"accepted artifact"
        return None

    async def pal(target, reference, *args, **kwargs):
        tails.append(reference)
        return None

    monkeypatch.setattr(orchestrator, "_try_timeline_affine", affine)
    monkeypatch.setattr(orchestrator, "_try_pal_two_point_fallback", pal)
    return orchestrator, seen, tails


async def run(orchestrator):
    return await orchestrator.evaluate_and_sync(TARGET, dict(META), "target", auto_sync=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["hash", "cache"])
async def test_early_acceptance_avoids_planning(monkeypatch, caplog, source):
    caplog.set_level("INFO")
    ref = ResolvedReference(subtitle("winner"), kind=source if source == "hash" else "edition", candidate="winner")
    external = External(cached=ref if source == "cache" else None)
    orchestrator, seen, _ = setup(monkeypatch, external, "winner", ref if source == "hash" else None)
    assert await run(orchestrator) == b"accepted artifact"
    assert external.calls == 0
    assert external.downloads == []
    assert seen == ["winner"]
    summary = next(r.message for r in caplog.records if "total_ms=" in r.message)
    assert "external_plan_avoided=True" in summary
    assert "external_plan_ms=None" in summary
    assert "result=transformed" in summary
    assert "dialogue" not in summary
    assert "Film" not in summary


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["hash", "cache", "none"])
async def test_failed_early_reference_reaches_one_ranked_plan(monkeypatch, caplog, source):
    caplog.set_level("INFO")
    ref = ResolvedReference(subtitle("prior"), kind="hash" if source == "hash" else "edition", candidate="prior")
    external = External(cached=ref if source == "cache" else None)
    orchestrator, seen, _ = setup(monkeypatch, external, "B", ref if source == "hash" else None)
    assert await run(orchestrator) == b"accepted artifact"
    assert external.calls == 1
    assert external.downloads == ["A", "B"]
    assert seen == (["prior"] if source != "none" else []) + ["A", "B"]
    summary = next(r.message for r in caplog.records if "total_ms=" in r.message)
    assert "candidate_count_planned=3" in summary
    assert "references_downloaded=2" in summary
    assert "external_plan_avoided=False" in summary
    assert "external_plan_ms=None" not in summary


@pytest.mark.asyncio
async def test_cached_sha_dedup_preserves_next_candidate(monkeypatch, caplog):
    caplog.set_level("INFO")
    external = External(cached=ResolvedReference(subtitle("A"), kind="edition", candidate="cached"))
    orchestrator, seen, _ = setup(monkeypatch, external, "B")
    assert await run(orchestrator) == b"accepted artifact"
    assert seen == ["cached", "B"]
    assert external.downloads == ["A", "B"]
    assert external.calls == 1
    summary = next(r.message for r in caplog.records if "total_ms=" in r.message)
    assert "references_deduplicated=1" in summary


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("https://private.invalid/?token=SECRET"), TimeoutError()])
async def test_planning_failure_preserves_original_and_safe_logs(monkeypatch, caplog, error):
    caplog.set_level("INFO")
    external = External(error=error)
    orchestrator, _, _ = setup(monkeypatch, external)
    assert await run(orchestrator) == TARGET
    assert external.calls == 1
    assert external.downloads == []
    assert "SECRET" not in caplog.text
    assert "private.invalid" not in caplog.text
    assert "result=original" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["hash", "cache", "plan"])
async def test_pal_tail_keeps_first_usable_reference(monkeypatch, source):
    prior = ResolvedReference(subtitle("prior"), kind="hash" if source == "hash" else "edition", candidate="prior")
    external = External(cached=prior if source == "cache" else None)
    orchestrator, seen, tails = setup(monkeypatch, external, None, prior if source == "hash" else None)
    assert await run(orchestrator) == TARGET
    assert seen == (["prior"] if source != "plan" else []) + ["A", "B", "C"]
    assert tails == [prior.text if source != "plan" else subtitle("A")]


@pytest.mark.asyncio
async def test_single_flight_waiter_cancellation_does_not_cancel_plan(monkeypatch):
    external = External()
    external.release = asyncio.Event()
    orchestrator, _, _ = setup(monkeypatch, external)
    first = asyncio.create_task(run(orchestrator))
    await external.entered.wait()
    second = asyncio.create_task(run(orchestrator))
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert not external.cleaned
    external.release.set()
    assert await second == b"accepted artifact"
    assert external.calls == 1
    await orchestrator.close()
    assert not orchestrator._inflight


@pytest.mark.asyncio
async def test_shutdown_cancels_and_drains_awaited_plan(monkeypatch):
    external = External()
    external.release = asyncio.Event()
    orchestrator, _, _ = setup(monkeypatch, external)
    waiter = asyncio.create_task(run(orchestrator))
    await external.entered.wait()
    await orchestrator.close()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert external.cleaned
    assert external.calls == 1
    assert external.downloads == []
    assert not orchestrator._inflight


@pytest.mark.asyncio
async def test_provider_timeout_isolated_from_healthy_provider(monkeypatch):
    strategy = ExternalExactStrategy(timeout=0.01)
    slow_cleaned = asyncio.Event()

    async def search(provider, query):
        if provider == "slow":
            try:
                await asyncio.Event().wait()
            finally:
                slow_cleaned.set()
        return ["healthy-result"]

    monkeypatch.setattr(strategy, "_search_provider", search)
    result = await strategy._gather_candidates(["slow", "healthy"], SimpleNamespace())
    assert result == [("healthy-result", "healthy")]
    assert slow_cleaned.is_set()


def test_service_preflight_does_not_count_process_launch(monkeypatch):
    from app.services.sync_service import SubtitleSyncService

    timing = SyncTiming()
    token = current_sync_timing.set(timing)
    try:
        service = SubtitleSyncService()
        monkeypatch.setattr(service, "is_available", lambda: True)
        monkeypatch.setattr("app.services.sync_service.subprocess.run", lambda *a, **kw: pytest.fail("preflight must stop process"))
        assert service.sync("malformed", "malformed") is None
        assert timing.alass_processes_launched == 0
    finally:
        current_sync_timing.reset(token)


def test_service_counts_alignment_process_not_availability_probe(monkeypatch):
    from pathlib import Path

    from app.services.sync_service import SubtitleSyncService

    timing = SyncTiming()
    token = current_sync_timing.set(timing)
    commands = []

    def process(command, **kwargs):
        commands.append(command)
        Path(command[3]).write_text(subtitle("output"), encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr=b"")

    try:
        service = SubtitleSyncService()
        monkeypatch.setattr(service, "is_available", lambda: True)
        monkeypatch.setattr("app.services.sync_service.subprocess.run", process)
        assert service.sync(subtitle("target"), subtitle("reference")) is not None
        assert len(commands) == timing.alass_processes_launched == 1
    finally:
        current_sync_timing.reset(token)


@pytest.mark.asyncio
async def test_repeated_strategy_object_is_planned_once(monkeypatch):
    external = External()
    orchestrator, _, _ = setup(monkeypatch, external, None)
    monkeypatch.setattr(orchestrator, "_strategies", lambda: [("external", external)] * 2)
    assert await run(orchestrator) == TARGET
    assert external.calls == 1


@pytest.mark.parametrize("timeout", [False, True])
def test_process_launch_count_handles_spawn_error_and_timeout(monkeypatch, timeout):
    import subprocess

    from app.services.sync_service import SubtitleSyncService

    timing = SyncTiming()
    token = current_sync_timing.set(timing)

    def process(*args, **kwargs):
        if timeout:
            raise subprocess.TimeoutExpired("alass", 1)
        raise FileNotFoundError("not installed")

    try:
        service = SubtitleSyncService()
        monkeypatch.setattr(service, "is_available", lambda: True)
        monkeypatch.setattr("app.services.sync_service.subprocess.run", process)
        assert service.sync(subtitle("target"), subtitle("reference")) is None
        assert timing.alass_processes_launched == int(timeout)
    finally:
        current_sync_timing.reset(token)


@pytest.mark.parametrize("result", ["error", "cancelled", "original"])
def test_incomplete_or_unconfigured_job_does_not_claim_avoided_planning(caplog, result):
    caplog.set_level("INFO")
    SyncTiming().log(result, 1.0)
    assert "external_plan_avoided=False" in caplog.text


@pytest.mark.parametrize("result", ["error", "cancelled", "original"])
def test_configured_but_unsuccessful_job_does_not_claim_avoided_planning(caplog, result):
    caplog.set_level("INFO")
    SyncTiming(external_plan_available=True).log(result, 1.0)
    assert "external_plan_avoided=False" in caplog.text


@pytest.mark.asyncio
async def test_no_external_strategy_does_not_claim_avoided_planning(monkeypatch, caplog):
    caplog.set_level("INFO")
    external = External()
    ref = ResolvedReference(subtitle("winner"), kind="hash", candidate="winner")
    orchestrator, _, _ = setup(monkeypatch, external, "winner", ref)
    orchestrator._external_strategy = None
    assert await run(orchestrator) == b"accepted artifact"
    assert "external_plan_avoided=False" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["empty", "exception"])
async def test_moviehash_unavailable_reaches_external_fallback(monkeypatch, failure):
    external = External()
    orchestrator, seen, _ = setup(monkeypatch, external)

    async def resolve(query):
        assert external.calls == 0
        if failure == "exception":
            raise RuntimeError("provider unavailable")
        return ResolvedReference(None)

    orchestrator._hash_reference_strategy = SimpleNamespace(resolve_with_provenance=resolve)
    assert await run(orchestrator) == b"accepted artifact"
    assert external.calls == 1
    assert seen == ["A"]


@pytest.mark.asyncio
async def test_optional_hash_exact_stays_after_exhausted_external_candidates(monkeypatch):
    external = External()
    orchestrator, seen, _ = setup(monkeypatch, external, "optional")

    async def resolve(query):
        assert external.calls == 1
        assert external.downloads == ["A", "B", "C"]
        return ResolvedReference(subtitle("optional"), kind="hash", candidate="optional")

    orchestrator._hash_strategy = SimpleNamespace(resolve_with_provenance=resolve)
    assert await run(orchestrator) == b"accepted artifact"
    assert seen == ["A", "B", "C", "optional"]


@pytest.mark.asyncio
async def test_real_whiplash_cached_scheduler_preserves_oracle_and_skips_tail(monkeypatch):
    from pathlib import Path

    from app.services.subtitle_matcher import parse_srt_cues

    folder = Path(__file__).parent / "fixtures" / "whiplash"
    target = (folder / "target.srt").read_text(encoding="utf-8-sig").encode("utf-8")
    reference = (folder / "reference.srt").read_text(encoding="utf-8-sig")
    # Synthetic same-group identity isolates scheduling from the real timing oracle.
    release = "Whiplash.2014.1080p.BluRay.REMUX-FraMeSToR"
    external = External(cached=ResolvedReference(reference, kind="edition", candidate=release))
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True)

    class NoAlass:
        async def sync_async(self, *args, **kwargs):
            pytest.fail("verified Whiplash must not run ALASS")

    orchestrator = SyncOrchestrator(external_strategy=external, sync_service=NoAlass())

    async def no_pal(*args, **kwargs):
        pytest.fail("verified Whiplash must not run PAL tail")

    monkeypatch.setattr(orchestrator, "_try_pal_two_point_fallback", no_pal)
    output = await orchestrator.evaluate_and_sync(target, {**META, "target_filename": release}, "whiplash", auto_sync=True)
    # The oracle remains outside synchronization inputs and is read after generation.
    expected = parse_srt_cues((folder / "expected.srt").read_text(encoding="utf-8-sig"))
    original, generated = parse_srt_cues(target.decode()), parse_srt_cues(output.decode())
    assert len(original) == len(generated) == len(expected) == 825
    assert len(parse_srt_cues(reference)) == 960
    assert [cue[2] for cue in original] == [cue[2] for cue in generated]
    for (start, end, _), (expected_start, expected_end, _) in zip(generated, expected, strict=True):
        assert start == expected_start
        assert abs(end - expected_end) <= 1
    assert external.calls == 0
    assert external.downloads == []


@pytest.mark.asyncio
@pytest.mark.parametrize("cached_before_hash", [False, True])
async def test_moviehash_cache_write_does_not_replace_external_snapshot(monkeypatch, cached_before_hash):
    prior_cache = ResolvedReference(subtitle("old"), kind="edition", candidate="old")
    external = External(cached=prior_cache if cached_before_hash else None)
    orchestrator, seen, _ = setup(monkeypatch, external, "A")
    hashed = ResolvedReference(subtitle("hashed"), kind="hash", candidate="hashed")

    async def resolve(query):
        # Real MovieHash resolution caches a downloaded reference before sync acceptance.
        external.cache = SimpleNamespace(get=lambda query: hashed)
        return hashed

    orchestrator._hash_reference_strategy = SimpleNamespace(resolve_with_provenance=resolve)
    assert await run(orchestrator) == b"accepted artifact"
    assert seen == ["hashed"] + (["old"] if cached_before_hash else []) + ["A"]
    assert external.calls == 1
