"""Sticky Verified Reference: learn once, reuse safely, deliver faster.

A reference that produced a verified, deliverable sync becomes the first
attempt for later subtitles for the SAME video. Promotion happens only on
TRANSFORMED serves; every reuse still re-verifies, re-checks binding, and
re-checks delivery. Failures never erase or oscillate the preference.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.sync import hybrid_router as router
from app.services.sync import orchestrator as om
from app.services.sync.alignment import SyncState, VerificationAvailability
from app.services.sync.cache import ReferenceDiskCache
from app.services.sync.query import ReferenceQuery, ResolvedReference
from app.services.sync_cache import SyncCache

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "penny"
CACHE_SUBS = pathlib.Path(__file__).resolve().parent.parent / "subs_cache"
D3G = "Penny.Dreadful.S01E01.Night.Work.1080p.BluRay.TrueHD5.1.AVC-d3g.mkv"
AOS_NAME = "Penny.Dreadful.S01E01.Bluray-AOS.srt"
ROVERS_NAME = "Penny.Dreadful.S01E01.BluRay.1080p.REMUX-FraMeSToR.srt"


def _read(path: pathlib.Path) -> str:
    from tests.private_fixtures import read_subtitle_text

    return read_subtitle_text(path)


def _hdtv() -> str:
    return _read(FIXTURES / "penny-s01e01-hdtv-ar-2.srt")


def _aos() -> str:
    return _read(FIXTURES / "penny-s01e01-bluray-aos-ar.srt")


def _rovers_en() -> str:
    path = (CACHE_SUBS / "references" /
            "tt2628232_1_1_d3g_v2_601ffb5b21eb105b.video.subdl.edition.8427e4afee8812a2.srt")
    if not path.is_file():
        pytest.skip(f"real-byte fixture unavailable: {path.name}")
    return path.read_bytes().decode("utf-8-sig")


def _webdl() -> str:
    path = CACHE_SUBS / "1e9b2fc7c3fdafbc.srt"
    if not path.is_file():
        pytest.skip(f"real-byte fixture unavailable: {path.name}")
    return path.read_bytes().decode("utf-8-sig")


def _meta() -> dict:
    return {"imdb_id": "tt2628232", "media_type": "series", "season": 1,
            "episode": 1, "lang": "ara", "target_filename": D3G,
            "title": "Penny Dreadful", "video_size": 4836521323}


def _query(video: str = D3G, episode: int = 1) -> ReferenceQuery:
    return ReferenceQuery(
        imdb_id="tt2628232", target_filename=video, media_type="series",
        season=1, episode=episode, target_lang="ara",
        video_size=4836521323, languages=("eng", "ara"))


def _resolved(text: str, candidate: str = AOS_NAME,
              provider: str = "subsource") -> ResolvedReference:
    return ResolvedReference(text, kind="edition", candidate=candidate,
                             provider=provider, member_season=1, member_episode=1)


def _evaluation(verified: bool = True):
    from app.services.sync.alignment import SubtitleEvaluation

    evaluation = SubtitleEvaluation()
    if verified:
        evaluation.set_verdict(SyncState.VERIFIED_RESYNCED,
                               VerificationAvailability.VERIFIED)
    return evaluation


def _gate_on(monkeypatch) -> None:
    monkeypatch.setattr(router, "_enabled", True)
    monkeypatch.delenv(router.TEST_SCOPE_ENV, raising=False)
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True)
    monkeypatch.setattr(om.SyncOrchestrator, "_audit_serve", lambda self, *a, **k: None)
    monkeypatch.setattr(om.SyncOrchestrator, "_record_outcome", lambda self, *a, **k: None)
    monkeypatch.setattr(om.SyncOrchestrator, "_try_timeline_affine",
                        AsyncMock(return_value=None))


def _harness(monkeypatch, strategy, service=None, cache=None):
    _gate_on(monkeypatch)
    service = service or SimpleNamespace(sync_async=AsyncMock(return_value=None))
    orch = om.SyncOrchestrator(sync_cache=cache or SyncCache(),
                              external_strategy=strategy, sync_service=service)
    return orch, service


# ---------------------------------------------------------------- store ---

def test_promote_and_lookup_round_trip(tmp_path: pathlib.Path) -> None:
    cache = ReferenceDiskCache(root=tmp_path)
    query = _query()
    assert cache.set_sticky_reference(
        query, provider="subsource", candidate=AOS_NAME, kind="edition",
        text=_aos(), member_season=1, member_episode=1,
        evidence={"sync_state": "verified_resynced", "groups": 80}) is True
    resolved = cache.get_sticky_reference(query)
    assert resolved is not None
    assert resolved.from_sticky_reference is True
    assert resolved.text == _aos()
    assert resolved.provider == "subsource"
    assert resolved.candidate == AOS_NAME
    assert resolved.member_episode == 1


def test_promotion_replaces_previous_winner(tmp_path: pathlib.Path) -> None:
    cache = ReferenceDiskCache(root=tmp_path)
    query = _query()
    assert cache.set_sticky_reference(
        query, provider="subdl", candidate="old.srt", kind="edition",
        text=_rovers_en(), member_season=1, member_episode=1, evidence={}) is True
    assert cache.set_sticky_reference(
        query, provider="subsource", candidate=AOS_NAME, kind="edition",
        text=_aos(), member_season=1, member_episode=1, evidence={}) is True
    resolved = cache.get_sticky_reference(query)
    assert resolved is not None and resolved.candidate == AOS_NAME


def test_methodology_mismatch_invalidates(tmp_path: pathlib.Path) -> None:
    cache = ReferenceDiskCache(root=tmp_path)
    query = _query()
    assert cache.set_sticky_reference(
        query, provider="subsource", candidate=AOS_NAME, kind="edition",
        text=_aos(), member_season=1, member_episode=1, evidence={}) is True
    pointer = tmp_path / f"{query.video_scope_stem}.sticky.json"
    record = json.loads(pointer.read_text(encoding="utf-8"))
    record["verification_version"] = "unified-v1"
    pointer.write_text(json.dumps(record), encoding="utf-8")
    assert cache.get_sticky_reference(query) is None
    assert not pointer.exists()


def test_tampered_artifact_invalidates(tmp_path: pathlib.Path) -> None:
    cache = ReferenceDiskCache(root=tmp_path)
    query = _query()
    assert cache.set_sticky_reference(
        query, provider="subsource", candidate=AOS_NAME, kind="edition",
        text=_aos(), member_season=1, member_episode=1, evidence={}) is True
    pointer = tmp_path / f"{query.video_scope_stem}.sticky.json"
    record = json.loads(pointer.read_text(encoding="utf-8"))
    artifact = tmp_path / record["artifact"]
    artifact.write_text("corrupted evidence", encoding="utf-8")
    assert cache.get_sticky_reference(query) is None
    assert not pointer.exists()


def test_missing_artifact_invalidates(tmp_path: pathlib.Path) -> None:
    cache = ReferenceDiskCache(root=tmp_path)
    query = _query()
    assert cache.set_sticky_reference(
        query, provider="subsource", candidate=AOS_NAME, kind="edition",
        text=_aos(), member_season=1, member_episode=1, evidence={}) is True
    pointer = tmp_path / f"{query.video_scope_stem}.sticky.json"
    record = json.loads(pointer.read_text(encoding="utf-8"))
    (tmp_path / record["artifact"]).unlink()
    assert cache.get_sticky_reference(query) is None
    assert not pointer.exists()


def test_expired_pointer_invalidates(tmp_path: pathlib.Path) -> None:
    cache = ReferenceDiskCache(root=tmp_path)
    query = _query()
    assert cache.set_sticky_reference(
        query, provider="subsource", candidate=AOS_NAME, kind="edition",
        text=_aos(), member_season=1, member_episode=1, evidence={}) is True
    pointer = tmp_path / f"{query.video_scope_stem}.sticky.json"
    old = 60 * 24 * 3600 * 31
    os.utime(pointer, (pointer.stat().st_mtime - old,) * 2)
    assert cache.get_sticky_reference(query) is None
    assert not pointer.exists()


def test_scope_mismatch_across_video_and_episode(tmp_path: pathlib.Path) -> None:
    cache = ReferenceDiskCache(root=tmp_path)
    assert cache.set_sticky_reference(
        _query(), provider="subsource", candidate=AOS_NAME, kind="edition",
        text=_aos(), member_season=1, member_episode=1, evidence={}) is True
    assert cache.get_sticky_reference(_query(video="Other.Show.S01E01.mkv")) is None
    assert cache.get_sticky_reference(_query(episode=2)) is None


def test_unproven_member_never_promotes(tmp_path: pathlib.Path) -> None:
    cache = ReferenceDiskCache(root=tmp_path)
    query = _query()
    assert cache.set_sticky_reference(
        query, provider="subsource", candidate=AOS_NAME, kind="edition",
        text=_aos(), evidence={}) is False
    assert cache.set_sticky_reference(
        query, provider="subsource", candidate=AOS_NAME, kind="edition",
        text=_aos(), member_season=1, member_episode=2, evidence={}) is False
    assert list(tmp_path.iterdir()) == []


def test_persistence_across_instances(tmp_path: pathlib.Path) -> None:
    query = _query()
    writer = ReferenceDiskCache(root=tmp_path)
    assert writer.set_sticky_reference(
        query, provider="subsource", candidate=AOS_NAME, kind="edition",
        text=_aos(), member_season=1, member_episode=1, evidence={}) is True
    reader = ReferenceDiskCache(root=tmp_path)
    resolved = reader.get_sticky_reference(query)
    assert resolved is not None and resolved.text == _aos()


# ------------------------------------------------------------ promotion ---

def test_no_promotion_without_verified_delivery(monkeypatch, tmp_path: pathlib.Path) -> None:
    _gate_on(monkeypatch)
    cache = ReferenceDiskCache(root=tmp_path)
    strategy = SimpleNamespace(cache=cache)
    orch = om.SyncOrchestrator(sync_cache=SyncCache(), external_strategy=strategy,
                              sync_service=None)
    meta, target = _meta(), _hdtv()
    resolved = _resolved(_aos())
    orch._promote_sticky_reference(meta=meta, target_id="sub", pristine_text=target,
                                   resolved=resolved,
                                   evaluation=_evaluation(verified=False))
    assert list(tmp_path.iterdir()) == []
    orch._promote_sticky_reference(meta=meta, target_id="sub", pristine_text=target,
                                   resolved=ResolvedReference(None),
                                   evaluation=_evaluation(verified=True))
    assert list(tmp_path.iterdir()) == []
    # A self-identical "reference" is never promotable.
    orch._promote_sticky_reference(meta=meta, target_id="sub", pristine_text=target,
                                   resolved=_resolved(target, candidate="same.srt"),
                                   evaluation=_evaluation(verified=True))
    assert list(tmp_path.iterdir()) == []


def test_verified_serve_promotes_with_evidence(monkeypatch, tmp_path: pathlib.Path) -> None:
    _gate_on(monkeypatch)
    cache = ReferenceDiskCache(root=tmp_path)
    strategy = SimpleNamespace(cache=cache)
    orch = om.SyncOrchestrator(sync_cache=SyncCache(), external_strategy=strategy,
                              sync_service=None)
    orch._promote_sticky_reference(
        meta=_meta(), target_id="sub", pristine_text=_hdtv(),
        resolved=_resolved(_aos()), evaluation=_evaluation(verified=True))
    pointer = tmp_path / f"{_query().video_scope_stem}.sticky.json"
    record = json.loads(pointer.read_text(encoding="utf-8"))
    assert record["verification_method"] == "unified"
    assert record["verification_version"] == "timing-v3"
    assert record["candidate"] == AOS_NAME
    assert record["evidence"]["sync_state"] == "verified_resynced"
    assert cache.get_sticky_reference(_query()) is not None


def test_gate_off_never_promotes(monkeypatch, tmp_path: pathlib.Path) -> None:
    monkeypatch.setattr(router, "_enabled", False)
    monkeypatch.delenv(router.TEST_SCOPE_ENV, raising=False)
    cache = ReferenceDiskCache(root=tmp_path)
    strategy = SimpleNamespace(cache=cache)
    orch = om.SyncOrchestrator(sync_cache=SyncCache(), external_strategy=strategy,
                              sync_service=None)
    orch._promote_sticky_reference(
        meta=_meta(), target_id="sub", pristine_text=_hdtv(),
        resolved=_resolved(_aos()), evaluation=_evaluation(verified=True))
    assert list(tmp_path.iterdir()) == []


# ----------------------------------------------------------------- tier ---

def _disk_strategy(monkeypatch, tmp_path: pathlib.Path, **extra):
    """External-style strategy over a tmp on-disk reference store."""
    cache = ReferenceDiskCache(root=tmp_path)

    async def _plan(query):
        return []

    strategy = SimpleNamespace(
        plan_candidates=_plan, cache=cache,
        download_planned_candidate=AsyncMock(return_value=ResolvedReference(None)),
        resolve_with_provenance=AsyncMock(return_value=ResolvedReference(None)),
        **extra)
    return strategy, cache


async def _attempt_names(orch, strategy, query) -> tuple[list[str], object]:
    from app.services.sync.telemetry import SyncTiming

    timing = SyncTiming()
    names = []
    async for name, _adapter in orch._reference_attempts(
            [("external-release-reference", strategy)], query, lambda text: True, timing):
        names.append(name)
    return names, timing


async def test_sticky_attempted_before_video_cache(monkeypatch, tmp_path: pathlib.Path) -> None:
    """The winner takes priority over stale video-cache candidates."""
    _gate_on(monkeypatch)
    strategy, cache = _disk_strategy(monkeypatch, tmp_path)
    query = _query()
    assert cache.set_video_scoped(
        query, "subdl", _rovers_en(), kind="edition", candidate="Old-Stale.srt",
        member_season=1, member_episode=1) is True
    assert cache.set_sticky_reference(
        query, provider="subsource", candidate=AOS_NAME, kind="edition",
        text=_aos(), member_season=1, member_episode=1, evidence={}) is True
    orch = om.SyncOrchestrator(sync_cache=SyncCache(), external_strategy=strategy,
                              sync_service=None)
    names, timing = await _attempt_names(orch, strategy, query)
    assert names[0].endswith("#sticky"), names
    assert timing.sticky_reference_attempted is True
    # A different-byte video-cache candidate may still follow; precedence
    # (sticky first) is what the tier guarantees.


async def test_identical_video_bytes_not_reattempted(monkeypatch, tmp_path: pathlib.Path) -> None:
    """Same bytes via video tier are skipped after the sticky attempt."""
    _gate_on(monkeypatch)
    strategy, cache = _disk_strategy(monkeypatch, tmp_path)
    query = _query()
    assert cache.set_video_scoped(
        query, "subsource", _aos(), kind="edition", candidate="Same-Bytes.srt",
        member_season=1, member_episode=1) is True
    assert cache.set_sticky_reference(
        query, provider="subsource", candidate=AOS_NAME, kind="edition",
        text=_aos(), member_season=1, member_episode=1, evidence={}) is True
    orch = om.SyncOrchestrator(sync_cache=SyncCache(), external_strategy=strategy,
                              sync_service=None)
    names, timing = await _attempt_names(orch, strategy, query)
    assert names[0].endswith("#sticky"), names
    assert not any(name.endswith("#video_cache") for name in names), names


async def test_sticky_hit_skips_planning_and_download(monkeypatch, tmp_path: pathlib.Path) -> None:
    """Seeded AOS first: HDTV serves TRANSFORMED, no search/download/ALASS."""
    _gate_on(monkeypatch)
    strategy, cache = _disk_strategy(monkeypatch, tmp_path)
    query = _query()
    assert cache.set_sticky_reference(
        query, provider="subsource", candidate=AOS_NAME, kind="edition",
        text=_aos(), member_season=1, member_episode=1, evidence={}) is True
    service = SimpleNamespace(sync_async=AsyncMock(return_value=None))
    orch = om.SyncOrchestrator(sync_cache=SyncCache(), external_strategy=strategy,
                              sync_service=service)
    out = await orch.evaluate_and_sync(_hdtv().encode(), _meta(), "sub-hdtv",
                                       auto_sync=True)
    assert out != _hdtv().encode()
    service.sync_async.assert_not_called()
    strategy.download_planned_candidate.assert_not_called()


async def test_self_identical_sticky_skipped(monkeypatch, tmp_path: pathlib.Path) -> None:
    """A sticky record matching the target itself is skipped, flow continues."""
    _gate_on(monkeypatch)
    strategy, cache = _disk_strategy(monkeypatch, tmp_path)
    query = _query()
    assert cache.set_sticky_reference(
        query, provider="subsource", candidate="same.srt", kind="edition",
        text=_hdtv(), member_season=1, member_episode=1, evidence={}) is True
    service = SimpleNamespace(sync_async=AsyncMock(return_value=None))
    orch = om.SyncOrchestrator(sync_cache=SyncCache(), external_strategy=strategy,
                              sync_service=service)
    out = await orch.evaluate_and_sync(_hdtv().encode(), _meta(), "sub-hdtv",
                                       auto_sync=True)
    assert out == _hdtv().encode()


async def test_target_failure_keeps_valid_sticky(monkeypatch, tmp_path: pathlib.Path) -> None:
    """One incompatible subtitle never erases the preference for others."""
    _gate_on(monkeypatch)
    strategy, cache = _disk_strategy(monkeypatch, tmp_path)
    query = _query()
    assert cache.set_sticky_reference(
        query, provider="subsource", candidate=AOS_NAME, kind="edition",
        text=_aos(), member_season=1, member_episode=1, evidence={}) is True
    service = SimpleNamespace(sync_async=AsyncMock(return_value=None))
    orch = om.SyncOrchestrator(sync_cache=SyncCache(), external_strategy=strategy,
                              sync_service=service)
    # Unrelated-content target: sticky attempted, fails verification, ORIGINAL.
    foreign = ("1\n00:01:00,000 --> 00:01:02,000\nCompletely unrelated line one\n\n"
               "2\n00:02:00,000 --> 00:02:02,000\nCompletely unrelated line two\n")
    out = await orch.evaluate_and_sync(foreign.encode(), _meta(), "sub-foreign",
                                       auto_sync=True)
    assert out == foreign.encode()
    pointer = tmp_path / f"{query.video_scope_stem}.sticky.json"
    assert pointer.is_file(), "a target-specific failure must not erase the record"
    # A compatible target still hits the retained preference afterwards.
    out2 = await orch.evaluate_and_sync(_hdtv().encode(), _meta(), "sub-hdtv",
                                        auto_sync=True)
    assert out2 != _hdtv().encode()


async def test_new_winner_replaces_unsuitable_preference(monkeypatch, tmp_path: pathlib.Path) -> None:
    """A newly verified reference safely replaces the preferred one."""
    _gate_on(monkeypatch)
    import app.services.sync.season_pack as pack_module

    monkeypatch.setattr(pack_module, "pack_root",
                        lambda root=None: tmp_path / "packs")
    cache = ReferenceDiskCache(root=tmp_path)
    query = _query()
    assert cache.set_sticky_reference(
        query, provider="subdl", candidate="Old-Stale.srt", kind="edition",
        text=_rovers_en(), member_season=1, member_episode=1, evidence={}) is True
    service = SimpleNamespace(sync_async=AsyncMock(return_value=None))
    strategy = SimpleNamespace(
        plan_candidates=AsyncMock(return_value=[
            (SimpleNamespace(release_name=AOS_NAME), "prov")]),
        download_planned_candidate=AsyncMock(return_value=_resolved(_aos())),
        cache=cache,
        resolve_with_provenance=AsyncMock(return_value=ResolvedReference(None)))
    orch = om.SyncOrchestrator(sync_cache=SyncCache(), external_strategy=strategy,
                              sync_service=service)
    out = await orch.evaluate_and_sync(_hdtv().encode(), _meta(), "sub-hdtv",
                                       auto_sync=True)
    assert out != _hdtv().encode()
    pointer = tmp_path / f"{query.video_scope_stem}.sticky.json"
    record = json.loads(pointer.read_text(encoding="utf-8"))
    assert record["candidate"] == AOS_NAME


async def test_rovers_then_aos_sequence_regression(monkeypatch, tmp_path: pathlib.Path) -> None:
    """Live replay: ROVERS attempted once, AOS wins; next request starts at AOS.

    Request 1 (HDTV): stale ROVERS video artifact attempted, then planned
    downloads (ROVERS fails, AOS verifies via Two-Point) and AOS promotes.
    Request 2 (WEB-DL, different subtitle ID, same video): sticky AOS first,
    no planning, no downloads, no ALASS.
    """
    _gate_on(monkeypatch)
    import app.services.sync.season_pack as pack_module

    monkeypatch.setattr(pack_module, "pack_root",
                        lambda root=None: tmp_path / "packs")
    cache = ReferenceDiskCache(root=tmp_path)
    query = _query()
    assert cache.set_video_scoped(
        query, "subdl", _rovers_en(), kind="edition", candidate="Old-Stale.srt",
        member_season=1, member_episode=1) is True

    rovers = _resolved(_rovers_en(), candidate="Old-Stale.srt", provider="subdl")
    aos = _resolved(_aos())
    downloads: list[str] = []

    async def _download(cand, provider, query, validate, shared, plan_index=None):
        downloads.append(getattr(cand, "release_name", "?"))
        if len(downloads) == 1:
            return rovers
        return aos

    async def _plan(query):
        return [(SimpleNamespace(release_name="Old-Stale.srt"), "prov"),
                (SimpleNamespace(release_name=AOS_NAME), "prov")]

    strategy = SimpleNamespace(
        plan_candidates=_plan, cache=cache,
        download_planned_candidate=_download,
        resolve_with_provenance=AsyncMock(return_value=ResolvedReference(None)))
    service = SimpleNamespace(sync_async=AsyncMock(return_value=None))
    orch = om.SyncOrchestrator(sync_cache=SyncCache(), external_strategy=strategy,
                              sync_service=service)
    first = await orch.evaluate_and_sync(_hdtv().encode(), _meta(), "sub-hdtv",
                                         auto_sync=True)
    assert first != _hdtv().encode()
    assert downloads == ["Old-Stale.srt", AOS_NAME], downloads
    pointer = tmp_path / f"{query.video_scope_stem}.sticky.json"
    assert json.loads(pointer.read_text(encoding="utf-8"))["candidate"] == AOS_NAME

    downloads.clear()
    service.sync_async.reset_mock()
    plans: list = []

    async def _plan2(query):
        plans.append(query)
        return []

    strategy.plan_candidates = _plan2  # type: ignore[assignment]
    second = await orch.evaluate_and_sync(_webdl().encode(), _meta(), "sub-webdl",
                                          auto_sync=True)
    assert second != _webdl().encode()
    assert plans == [], "no provider planning once sticky succeeds"
    assert downloads == [], "no downloads once sticky succeeds"
    service.sync_async.assert_not_called()


async def test_compatible_bypass_ignores_sticky(monkeypatch, tmp_path: pathlib.Path) -> None:
    """An already-compatible subtitle serves ORIGINAL before any reference."""
    _gate_on(monkeypatch)
    strategy, cache = _disk_strategy(monkeypatch, tmp_path)
    query = _query()
    assert cache.set_sticky_reference(
        query, provider="subsource", candidate=AOS_NAME, kind="edition",
        text=_aos(), member_season=1, member_episode=1, evidence={}) is True
    from tests.test_unified_verification import _linear_pair

    target, _ = _linear_pair()
    service = SimpleNamespace(sync_async=AsyncMock(return_value=None))
    orch = om.SyncOrchestrator(sync_cache=SyncCache(), external_strategy=strategy,
                              sync_service=service)
    await orch._sync_cache.set_verdict(
        om.SyncOrchestrator._verdict_key(
            _meta(), hashlib.sha256(target.encode()).hexdigest()[:16]),
        {"sync_state": "verified_synced", "verification": "verified",
         "target_binding": "exact", "verification_method": "unified",
         "verification_version": "timing-v3",
         "artifact_sha256": hashlib.sha256(target.encode()).hexdigest(),
         "reference_sha": "c" * 64})
    out = await orch.evaluate_and_sync(target.encode(), _meta(), "sub-lin",
                                       auto_sync=True)
    assert out == target.encode()
    service.sync_async.assert_not_called()


async def test_sticky_and_pack_cooperate(monkeypatch, tmp_path: pathlib.Path) -> None:
    """Same video → sticky first; pack tier still follows for its own cases."""
    import app.services.sync.season_pack as pack_module

    _gate_on(monkeypatch)
    strategy, cache = _disk_strategy(monkeypatch, tmp_path)
    query = _query()
    assert cache.set_sticky_reference(
        query, provider="subsource", candidate=AOS_NAME, kind="edition",
        text=_aos(), member_season=1, member_episode=1, evidence={}) is True
    pack_root = tmp_path / "packs"
    monkeypatch.setattr(pack_module, "pack_root", lambda root=None: pack_root)
    raw_pack = None
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("Penny.Dreadful.S01E01.Bluray-AOS.srt",
                         _aos().encode("utf-8"))
        archive.writestr("Penny.Dreadful.S01E02.Bluray-AOS.srt",
                         _aos().encode("utf-8"))
    raw_pack = buffer.getvalue()
    from app.services.sync.season_pack import retain_season_pack

    pack_key = retain_season_pack(
        raw=raw_pack, provider_label="subsource",
        pack_candidate="Penny.Dreadful.S01.Bluray-AOS.zip",
        source_download_digest="packdigest9", query=query,
        root=tmp_path / "packs")
    assert pack_key
    assert pack_module.promote_season_pack(query, ResolvedReference(
        _aos(), source_download_digest="packdigest9"), root=pack_root)
    orch = om.SyncOrchestrator(sync_cache=SyncCache(), external_strategy=strategy,
                              sync_service=None)
    names, _timing = await _attempt_names(orch, strategy, query)
    assert names[0].endswith("#sticky"), names
    # Pack tier remains available behind sticky for pack-eligible requests.
    assert any(name.endswith("#season_pack") for name in names), names
