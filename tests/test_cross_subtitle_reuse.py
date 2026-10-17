"""Cross-subtitle Sticky Reference and guarded artifact reuse.

The production Into the Wild case selected two different subtitle IDs whose
original provider bytes were byte-identical. The second request must reuse the
already-authorized corrected artifact without discovery or another ALASS run,
while still reporting UNVERIFIED/UNKNOWN.

These tests use synthetic bytes for mechanics and the private Into the Wild
fixture when available. No provider request and no real ALASS process runs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.sync import orchestrator as om
from app.services.sync.alignment import (
    SubtitleEvaluation,
    SyncState,
    VerificationAvailability,
)
from app.services.sync.cache import ReferenceDiskCache
from app.services.sync.large_offset_investigation import LargeOffsetServingState
from app.services.sync.query import ReferenceQuery, ResolvedReference
from app.services.sync_cache import SyncCache

CORRECTED = b"1\n00:00:30,000 --> 00:00:32,000\ncorrected artifact\n"
REF_SHA = "a" * 64
FIXTURES = Path(__file__).parent / "fixtures" / "into_the_wild"
REAL_TARGET = FIXTURES / "target.srt"
REAL_FIXTURES_AVAILABLE = REAL_TARGET.exists()
requires_real_fixtures = pytest.mark.skipif(
    not REAL_FIXTURES_AVAILABLE,
    reason="private Into the Wild subtitle fixture is not present",
)


def _ts(ms: int) -> str:
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _srt(cues: int, tag: str, offset_ms: int = 0) -> str:
    out = []
    for i in range(cues):
        start = 1000 + offset_ms + i * 3000
        end = start + 2000
        out.append(f"{i + 1}\n{_ts(start)} --> {_ts(end)}\n{tag}{i}\n")
    return "\n".join(out)


def _meta(sub_id: str = "d8efd4b030f64a1d", **kw) -> dict:
    base = {
        "imdb_id": "tt0758758",
        "media_type": "movie",
        "target_filename": "Into.The.Wild.2007.1080p.BluRay.x264-FSiHD",
        "video_size": "13150936000",
        "lang": "ara",
        "release_name": "Into The Wild.WEBRip.Netflix.srt",
        "sub_id": sub_id,
        "year": 2007,
    }
    base.update(kw)
    return base


def _query(sub_id: str = "d8efd4b030f64a1d") -> ReferenceQuery:
    return ReferenceQuery(
        imdb_id="tt0758758",
        media_type="movie",
        target_filename="Into.The.Wild.2007.1080p.BluRay.x264-FSiHD",
        target_sub_id=sub_id,
        target_sub_release_name="Into The Wild.WEBRip.Netflix.srt",
        title="Into the Wild",
        year=2007,
        video_size=13150936000,
        languages=("eng", "ara"),
    )


def _unverified_evaluation() -> SubtitleEvaluation:
    return SubtitleEvaluation(
        sync_state=SyncState.UNVERIFIED,
        verification=VerificationAvailability.UNKNOWN,
        target_binding="strong",
    )


def _guarded_meta(content_hash: str, artifact_key: str) -> dict:
    payload = _unverified_evaluation().model_dump(mode="json")
    payload.update(
        sync_state="unverified",
        verification="unknown",
        serving_state=LargeOffsetServingState.ALASS_CORRECTED_LARGE_OFFSET.value,
        large_offset_authorization=om.LARGE_OFFSET_CACHE_AUTHORIZATION_VERSION,
        target_binding="strong",
        target_binding_reasons=["release_group_match", "source_compatible"],
        reference_sha=REF_SHA,
        artifact_key=artifact_key,
        artifact_sha256=hashlib.sha256(CORRECTED).hexdigest(),
        original_content_hash=content_hash,
    )
    return payload


async def _put_content(
    cache: SyncCache,
    meta: dict,
    content_hash: str,
    *,
    artifact: bytes = CORRECTED,
    **overrides,
) -> str:
    key = om.SyncOrchestrator._content_artifact_key(meta, content_hash)
    record = _guarded_meta(content_hash, key)
    record.update(overrides)
    await cache.set(key, artifact)
    await cache.set_meta(key, record)
    return key


class _Strategy:
    """No provider work; records whether planning was reached."""

    validates_target = False
    cache = None

    def __init__(self) -> None:
        self.plan_calls = 0
        self.resolve_calls = 0

    async def plan_candidates(self, query):
        self.plan_calls += 1
        return []

    async def resolve_with_provenance(self, query, **kwargs):
        self.resolve_calls += 1
        return ResolvedReference(None)


class _SyncService:
    """Never called when planning yields no reference; records accidental calls."""

    def __init__(self) -> None:
        self.calls = 0

    async def sync_async(self, *args, **kwargs):
        self.calls += 1
        return None


def _orchestrator(cache: SyncCache, strategy: _Strategy | None = None):
    return om.SyncOrchestrator(
        external_strategy=strategy or _Strategy(),
        sync_cache=cache,
        sync_service=_SyncService(),
    )


def test_content_key_ignores_subtitle_id_but_binds_video_language_and_bytes():
    meta1 = _meta("d8efd4b030f64a1d")
    meta2 = _meta("92788b7594b7f497")
    assert om.SyncOrchestrator._content_artifact_key(meta1, "abc") == (
        om.SyncOrchestrator._content_artifact_key(meta2, "abc")
    )
    assert om.SyncOrchestrator._content_artifact_key(meta1, "abc") != (
        om.SyncOrchestrator._content_artifact_key(meta1, "def")
    )
    assert om.SyncOrchestrator._content_artifact_key(meta1, "abc") != (
        om.SyncOrchestrator._content_artifact_key(_meta(lang="eng"), "abc")
    )
    assert om.SyncOrchestrator._content_artifact_key(meta1, "abc") != (
        om.SyncOrchestrator._content_artifact_key(
            _meta(target_filename="Other.Release.mkv"), "abc"
        )
    )
    assert om.SyncOrchestrator._content_artifact_key(meta1, "abc") != (
        om.SyncOrchestrator._content_artifact_key(
            _meta(video_size="999999"), "abc"
        )
    )
    assert om.SyncOrchestrator._content_artifact_key(meta1, "abc") != (
        om.SyncOrchestrator._content_artifact_key(
            _meta(subdl_key="a-different-credential"), "abc"
        )
    )


def test_movie_member_normalization_never_weakens_series_matching():
    from app.services.sync.cache import _member_matches

    # Movies: historical 0 and current None describe the same member.
    assert _member_matches(0, None, movie=True) is True
    assert _member_matches(None, None, movie=True) is True
    # Series: season 0 (specials) is not season None; episode 0 is not missing.
    assert _member_matches(0, None, movie=False) is False
    assert _member_matches(None, 0, movie=False) is False
    assert _member_matches(0, 0, movie=False) is True
    assert _member_matches("02", 2, movie=False) is True


@pytest.mark.asyncio
async def test_guarded_artifact_reuses_across_subtitle_ids(monkeypatch):
    """Different subtitle ID + byte-identical bytes -> no discovery, no ALASS."""
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True, raising=False)
    cache = SyncCache()
    strategy = _Strategy()
    orchestrator = _orchestrator(cache, strategy)
    sub_bytes = _srt(600, "TARGET").encode()
    content_hash = hashlib.sha256(sub_bytes).hexdigest()[:16]
    meta1 = _meta("d8efd4b030f64a1d")
    meta2 = _meta("92788b7594b7f497")
    await _put_content(cache, meta1, content_hash)

    served = await orchestrator.evaluate_and_sync(
        sub_bytes, meta2, "92788b7594b7f497", auto_sync=True
    )
    assert served == CORRECTED
    assert strategy.plan_calls == 0, "cross-subtitle reuse must skip discovery"
    assert strategy.resolve_calls == 0
    assert orchestrator.metrics["large_offset_content_cache_hits"] == 1
    assert orchestrator._last_evaluation is not None
    assert orchestrator._last_evaluation.sync_state.value == "unverified"
    assert orchestrator._last_evaluation.verification.value == "unknown"
    # The second subtitle's own request key is now warm too.
    resolution_key = orchestrator._flight_key(meta2, "92788b7594b7f497", sub_bytes)
    assert await cache.get(resolution_key) == CORRECTED


@pytest.mark.asyncio
async def test_unknown_verdict_is_not_cached_but_content_artifact_is_reusable(monkeypatch):
    """SyncCache refuses an UNKNOWN verdict; the content tier is the reuse path."""
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True, raising=False)
    cache = SyncCache()
    orchestrator = _orchestrator(cache)
    sub_bytes = _srt(600, "TARGET").encode()
    content_hash = hashlib.sha256(sub_bytes).hexdigest()[:16]
    meta = _meta()
    verdict_key = orchestrator._verdict_key(meta, content_hash)
    content_key = om.SyncOrchestrator._content_artifact_key(meta, content_hash)
    await orchestrator._store_verdict(
        verdict_key,
        _unverified_evaluation(),
        content_hash,
        meta,
        "d8efd4b030f64a1d",
        artifact_key=content_key,
        artifact_bytes=CORRECTED,
        reference_sha=REF_SHA,
        serving_state=LargeOffsetServingState.ALASS_CORRECTED_LARGE_OFFSET.value,
    )
    assert await cache.get_verdict(verdict_key) is None, (
        "an UNKNOWN verdict must never become a reusable measured verdict"
    )
    await cache.set(content_key, CORRECTED)
    await cache.set_meta(content_key, _guarded_meta(content_hash, content_key))
    hit = await orchestrator._load_guarded_content_artifact(meta, content_hash)
    assert hit is not None
    artifact, record, key = hit
    assert artifact == CORRECTED
    assert record["serving_state"] == "alass_corrected_large_offset"
    assert key == content_key


@pytest.mark.asyncio
async def test_same_subtitle_payload_hit_publishes_content_tier_for_other_ids(monkeypatch):
    """A same-ID guarded replay also seeds the cross-subtitle content entry."""
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True, raising=False)
    cache = SyncCache()
    strategy = _Strategy()
    orchestrator = _orchestrator(cache, strategy)
    sub_bytes = _srt(600, "TARGET").encode()
    content_hash = hashlib.sha256(sub_bytes).hexdigest()[:16]
    meta1 = _meta("d8efd4b030f64a1d")
    meta2 = _meta("92788b7594b7f497")
    resolution_key = orchestrator._flight_key(meta1, "d8efd4b030f64a1d", sub_bytes)
    content_key = om.SyncOrchestrator._content_artifact_key(meta1, content_hash)
    record = _guarded_meta(content_hash, content_key)
    await cache.set(resolution_key, CORRECTED)
    await cache.set_meta(resolution_key, record)

    first = await orchestrator.evaluate_and_sync(
        sub_bytes, meta1, "d8efd4b030f64a1d", auto_sync=True
    )
    assert first == CORRECTED
    assert orchestrator.metrics["large_offset_cache_hits"] == 1
    assert await cache.get(content_key) == CORRECTED

    second = await orchestrator.evaluate_and_sync(
        sub_bytes, meta2, "92788b7594b7f497", auto_sync=True
    )
    assert second == CORRECTED
    assert strategy.plan_calls == 0
    assert orchestrator.metrics["large_offset_content_cache_hits"] == 1


@pytest.mark.asyncio
async def test_content_hit_does_not_bypass_metadata_gate(monkeypatch):
    """FAST_ORIGINAL/IDENTITY_CONFLICT keep their normal serve-original policy."""
    from app.services.sync.metadata_gate import (
        MetadataGateAction,
        MetadataGateDecision,
    )

    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True, raising=False)
    cache = SyncCache()
    strategy = _Strategy()
    orchestrator = _orchestrator(cache, strategy)
    sub_bytes = _srt(600, "TARGET").encode()
    content_hash = hashlib.sha256(sub_bytes).hexdigest()[:16]
    await _put_content(cache, _meta(), content_hash)
    monkeypatch.setattr(
        om,
        "decide_metadata_gate",
        lambda *args, **kwargs: MetadataGateDecision(
            MetadataGateAction.IDENTITY_CONFLICT, ("title_identity_conflict",)
        ),
    )

    served = await orchestrator.evaluate_and_sync(
        sub_bytes, _meta("92788b7594b7f497"), "92788b7594b7f497", auto_sync=True
    )
    assert served == sub_bytes
    assert orchestrator.metrics["large_offset_content_cache_hits"] == 0
    assert strategy.plan_calls == 0


@pytest.mark.asyncio
async def test_different_original_content_does_not_reuse(monkeypatch):
    """A different subtitle payload gets its own content key and normal pipeline."""
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True, raising=False)
    cache = SyncCache()
    strategy = _Strategy()
    orchestrator = _orchestrator(cache, strategy)
    original = _srt(600, "TARGET").encode()
    different = _srt(600, "OTHER", offset_ms=1500).encode()
    content_hash = hashlib.sha256(original).hexdigest()[:16]
    await _put_content(cache, _meta(), content_hash)

    served = await orchestrator.evaluate_and_sync(
        different, _meta("92788b7594b7f497"), "92788b7594b7f497", auto_sync=True
    )
    assert served == different
    assert strategy.plan_calls == 1
    assert orchestrator.metrics["large_offset_content_cache_hits"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        "missing_artifact",
        "tampered_artifact",
        "missing_authorization",
        "wrong_binding",
    ],
)
async def test_invalid_guarded_content_falls_back_to_reference_pipeline(monkeypatch, change):
    """Invalid/incomplete reuse must re-evaluate, never act as a cache rejection."""
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True, raising=False)
    cache = SyncCache()
    strategy = _Strategy()
    orchestrator = _orchestrator(cache, strategy)
    sub_bytes = _srt(600, "TARGET").encode()
    content_hash = hashlib.sha256(sub_bytes).hexdigest()[:16]
    meta = _meta()
    key = om.SyncOrchestrator._content_artifact_key(meta, content_hash)
    record = _guarded_meta(content_hash, key)
    if change == "missing_authorization":
        record.pop("large_offset_authorization", None)
    elif change == "wrong_binding":
        record["target_binding"] = "weak"
    if change != "missing_artifact":
        await cache.set(key, CORRECTED)
    if change == "tampered_artifact":
        await cache.set(key, b"1\n00:00:01,000 --> 00:00:02,000\ntampered\n")
    await cache.set_meta(key, record)

    served = await orchestrator.evaluate_and_sync(
        sub_bytes, _meta("92788b7594b7f497"), "92788b7594b7f497", auto_sync=True
    )
    assert served == sub_bytes
    assert strategy.plan_calls == 1, "incomplete reuse must reach reference planning"
    assert orchestrator.metrics["large_offset_content_cache_hits"] == 0


@pytest.mark.asyncio
async def test_cross_subtitle_cache_is_video_and_language_isolated(monkeypatch):
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True, raising=False)
    cache = SyncCache()
    strategy = _Strategy()
    orchestrator = _orchestrator(cache, strategy)
    sub_bytes = _srt(600, "TARGET").encode()
    content_hash = hashlib.sha256(sub_bytes).hexdigest()[:16]
    await _put_content(cache, _meta(), content_hash)

    other_video = _meta(
        "92788b7594b7f497",
        target_filename="Into.The.Wild.2007.720p.BluRay.x264-OTHER",
        video_size="999999",
    )
    served = await orchestrator.evaluate_and_sync(
        sub_bytes, other_video, "92788b7594b7f497", auto_sync=True
    )
    assert served == sub_bytes
    assert strategy.plan_calls == 1

    other_lang = _meta("92788b7594b7f497", lang="eng")
    orchestrator2 = _orchestrator(SyncCache(), _Strategy())
    served2 = await orchestrator2.evaluate_and_sync(
        sub_bytes, other_lang, "92788b7594b7f497", auto_sync=True
    )
    assert served2 == sub_bytes


def test_movie_video_cache_accepts_legacy_member_zero(tmp_path):
    """Old movie artifacts recorded member 0; requests carry None."""
    cache = ReferenceDiskCache(root=tmp_path)
    query = _query()
    text = _srt(600, "REFERENCE")
    assert cache.set_video_scoped(
        query, "subdl", text, kind="edition", candidate="Into.The.Wild.ENG.srt",
        member_season=0, member_episode=0, score=120.0,
    )
    # New writes normalize movie members to None, which still matches.
    assert cache.get_video_scoped(query) is not None
    # Simulate a legacy sidecar that still says 0.
    for sidecar in tmp_path.glob("*.json"):
        data = json.loads(sidecar.read_text("utf-8"))
        if data.get("scope") == "video":
            data["member_season"] = 0
            data["member_episode"] = 0
            sidecar.write_text(json.dumps(data, sort_keys=True), "utf-8")
    assert cache.get_video_scoped(query) is not None


def test_legacy_large_offset_promotes_sticky_for_movie_and_reuses_across_ids(tmp_path):
    cache = ReferenceDiskCache(root=tmp_path)
    orchestrator = om.SyncOrchestrator(
        external_strategy=SimpleNamespace(cache=cache),
        sync_cache=SyncCache(),
        sync_service=None,
    )
    target = _srt(600, "TARGET")
    reference = _srt(600, "REFERENCE", offset_ms=20_000)
    resolved = ResolvedReference(
        reference, kind="edition", bluray_match=True,
        candidate="Into.The.Wild.2007.1080p.BluRay.x264-FSiHD.ENG.srt",
        provider="subdl", member_season=0, member_episode=0,
    )
    meta = _meta()
    evaluation = _unverified_evaluation()
    orchestrator._promote_sticky_reference(
        meta=meta,
        target_id="d8efd4b030f64a1d",
        pristine_text=target,
        resolved=resolved,
        evaluation=evaluation,
        serving_state=LargeOffsetServingState.ALASS_CORRECTED_LARGE_OFFSET.value,
    )
    pointer = tmp_path / f"{_query().video_scope_stem}.sticky.json"
    assert pointer.is_file()
    record = json.loads(pointer.read_text("utf-8"))
    assert record["verification_method"] == "legacy"
    assert record["verification_version"] == "large-offset-v1"
    assert record["evidence"]["serving_state"] == "alass_corrected_large_offset"

    # A different subtitle ID for the same video hits the sticky pointer.
    hit = cache.get_sticky_reference(
        _query("92788b7594b7f497"), methodology=("legacy", "large-offset-v1")
    )
    assert hit is not None
    assert hit.from_sticky_reference is True
    assert hit.candidate.endswith(".ENG.srt")
    # Unified lookup must not reinterpret the legacy pointer.
    assert cache.get_sticky_reference(_query("92788b7594b7f497")) is None
    # A different video must not hit.
    other = ReferenceQuery(
        imdb_id="tt0758758", media_type="movie",
        target_filename="Into.The.Wild.2007.720p.BluRay.x264-OTHER",
        video_size=999999, languages=("eng", "ara"),
    )
    assert cache.get_sticky_reference(other, methodology=("legacy", "large-offset-v1")) is None


def test_legacy_unverified_without_corrected_serving_never_promotes(tmp_path):
    cache = ReferenceDiskCache(root=tmp_path)
    orchestrator = om.SyncOrchestrator(
        external_strategy=SimpleNamespace(cache=cache),
        sync_cache=SyncCache(), sync_service=None,
    )
    orchestrator._promote_sticky_reference(
        meta=_meta(), target_id="sub", pristine_text=_srt(600, "TARGET"),
        resolved=ResolvedReference(
            _srt(600, "REFERENCE", offset_ms=20_000), kind="edition",
            candidate="Into.The.Wild.ENG.srt", provider="subdl",
        ),
        evaluation=_unverified_evaluation(),
        serving_state=None,
    )
    assert list(tmp_path.glob("*.sticky.json")) == []


def test_legacy_sticky_is_attempted_before_discovery_for_a_new_subtitle(tmp_path):
    """A different subtitle ID for the same video reaches the promoted reference first."""
    cache = ReferenceDiskCache(root=tmp_path)
    orchestrator = om.SyncOrchestrator(
        external_strategy=SimpleNamespace(cache=cache),
        sync_cache=SyncCache(), sync_service=_SyncService(),
    )
    target = _srt(600, "TARGET")
    reference = _srt(600, "REFERENCE", offset_ms=20_000)
    orchestrator._promote_sticky_reference(
        meta=_meta(),
        target_id="d8efd4b030f64a1d",
        pristine_text=target,
        resolved=ResolvedReference(
            reference, kind="edition", bluray_match=True,
            candidate="Into.The.Wild.2007.1080p.BluRay.x264-FSiHD.ENG.srt",
            provider="subdl", member_season=0, member_episode=0,
        ),
        evaluation=_unverified_evaluation(),
        serving_state=LargeOffsetServingState.ALASS_CORRECTED_LARGE_OFFSET.value,
    )

    class _StickyStrategy:
        validates_target = False

        def __init__(self) -> None:
            self.cache = cache
            self.plan_calls = 0
            self.resolve_calls = 0

        async def plan_candidates(self, query):
            self.plan_calls += 1
            return []

        async def resolve_with_provenance(self, query, **kwargs):
            self.resolve_calls += 1
            return ResolvedReference(None)

    strategy = _StickyStrategy()

    async def collect():
        from app.services.sync.telemetry import SyncTiming

        timing = SyncTiming()
        generator = orchestrator._reference_attempts(
            [("external-release-reference", strategy)],
            _query("92788b7594b7f497"),
            lambda text: True,
            timing,
            _meta("92788b7594b7f497"),
        )
        try:
            name, adapter = await generator.__anext__()
            await adapter.resolve_with_provenance(_query("92788b7594b7f497"))
            return [name]
        finally:
            await generator.aclose()

    names = asyncio.run(collect())
    assert names == ["retained-reference#sticky"], names
    assert strategy.plan_calls == 0, "sticky reference must precede discovery"


@pytest.mark.asyncio
@requires_real_fixtures
async def test_into_the_wild_two_subtitle_ids_share_one_guarded_artifact(monkeypatch):
    """The exact two production subtitle IDs have the same content hash."""
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True, raising=False)
    cache = SyncCache()
    strategy = _Strategy()
    orchestrator = _orchestrator(cache, strategy)
    sub_bytes = REAL_TARGET.read_bytes()
    content_hash = hashlib.sha256(sub_bytes).hexdigest()[:16]
    meta1 = _meta("d8efd4b030f64a1d")
    meta2 = _meta("92788b7594b7f497")
    await _put_content(cache, meta1, content_hash)

    served1 = await orchestrator.evaluate_and_sync(
        sub_bytes, meta1, "d8efd4b030f64a1d", auto_sync=True
    )
    served2 = await orchestrator.evaluate_and_sync(
        sub_bytes, meta2, "92788b7594b7f497", auto_sync=True
    )
    assert served1 == served2 == CORRECTED
    assert strategy.plan_calls == 0
    assert orchestrator.metrics["large_offset_content_cache_hits"] == 2
