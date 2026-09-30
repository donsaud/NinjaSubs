"""A non-positive synchronization result must never become reusable as verified.

Defect provenance: a real Whiplash 2160p REMUX request produced an Alass output
that the verifier evaluated as ``UNVERIFIED`` / ``UNKNOWN`` (median +266 ms,
MAD 1106 ms, p95 4200 ms, drift +60.1 ms/min). The orchestrator nevertheless
wrote that transformed output into the payload cache and served it. A second
request re-read it, found the subtitle "already aligned", and served it as
``probable_sync`` -- so an unverified artifact was promoted without ever being
verified.

These tests pin the fail-closed contract:

* only a verified-and-measured entry may be served from the payload fast path;
* the artifact may still exist in the cache (nothing is deleted) -- it simply
  cannot authorize reuse;
* a missing/legacy meta record is treated as not reusable, not as permission.

The positive (warm hit) path is preserved, and ``test_orchestrator_logs_sync_cache_hit``
plus ``test_orchestrator_prefers_external_then_caches`` cover it.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_sync_strategies import (  # noqa: E402
    BIG_REF,
    _arabic_bytes,
    _FakeStrategy,
    _meta,
    _orchestrator,
    _stub_analyzer,
)

from app.services.sync.alignment import (  # noqa: E402
    SyncState,
    VerificationAvailability,
    is_reusable_verified,
)

NON_POSITIVE_STATES = [
    SyncState.UNVERIFIED,
    SyncState.REJECTED,
    SyncState.PROBABLE_SYNC,
]


class TestReusableVerifiedPredicate:
    """The single definition of "may be reused as a verified result"."""

    def test_verified_and_measured_is_reusable(self):
        for state in (SyncState.VERIFIED_SYNCED, SyncState.VERIFIED_RESYNCED):
            for verification in (
                VerificationAvailability.VERIFIED,
                VerificationAvailability.CACHED,
            ):
                assert is_reusable_verified(state.value, verification.value) is True

    def test_every_non_positive_state_is_not_reusable(self):
        for state in NON_POSITIVE_STATES:
            assert is_reusable_verified(state.value, VerificationAvailability.VERIFIED.value) is False

    def test_unmeasured_verification_is_never_reusable(self):
        for verification in (
            VerificationAvailability.UNKNOWN,
            VerificationAvailability.PREDICTED,
        ):
            assert (
                is_reusable_verified(SyncState.VERIFIED_SYNCED.value, verification.value) is False
            )

    def test_missing_state_and_verification_fail_closed(self):
        for state, verification in (
            (None, None),
            (None, VerificationAvailability.VERIFIED.value),
            (SyncState.VERIFIED_SYNCED.value, None),
            ("", ""),
            ("garbage", "garbage"),
        ):
            assert is_reusable_verified(state, verification) is False


class TestUnverifiedPayloadIsNotServed:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("state", NON_POSITIVE_STATES)
    async def test_non_positive_result_is_not_reused_on_repeat(self, monkeypatch, state):
        from app.config import settings as app_settings

        monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)

        orch = _orchestrator(external_strategy=_FakeStrategy(BIG_REF.decode()))
        _stub_analyzer(orch, state=state, verification="unknown")
        await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True)

        # The artifact IS stored -- caching is not deleted...
        assert orch._sync_cache._local, "artifact was deleted instead of being gated"

        # ...but a second request must NOT be answered from the cache.
        orch2 = _orchestrator(
            external_strategy=_FakeStrategy(BIG_REF.decode()),
            sync_cache=orch._sync_cache,
        )
        _stub_analyzer(orch2, state=state, verification="unknown")
        out2 = await orch2.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True)

        assert orch2._external_strategy.calls == 1, "non-positive artifact was served from cache"
        assert orch2._sync_service.calls == 1
        assert out2 is not None

    @pytest.mark.asyncio
    async def test_real_analyzer_rejection_is_not_reused(self, monkeypatch):
        """Uses the REAL analyzer: the synthetic fixture is genuinely rejected."""
        from app.config import settings as app_settings

        monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)

        orch = _orchestrator(external_strategy=_FakeStrategy(BIG_REF.decode()))
        await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True)
        first = orch._last_evaluation
        assert not is_reusable_verified(first.sync_state.value, first.verification.value)

        orch2 = _orchestrator(
            external_strategy=_FakeStrategy(BIG_REF.decode()),
            sync_cache=orch._sync_cache,
        )
        await orch2.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True)
        assert orch2._external_strategy.calls == 1, "rejected artifact was served from cache"

    @pytest.mark.asyncio
    async def test_unverified_never_produces_a_reusable_verdict_or_alias(self, monkeypatch):
        """The verdict store must not authorize the unverified payload either."""
        from app.config import settings as app_settings

        monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)

        orch = _orchestrator(external_strategy=_FakeStrategy(BIG_REF.decode()))
        _stub_analyzer(orch, state=SyncState.UNVERIFIED, verification="unknown")
        await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True)

        assert await orch._sync_cache.get_verdict(orch._verdict_key(_meta(), "x")) is None

    @pytest.mark.asyncio
    async def test_verified_result_is_still_reused_without_providers(self, monkeypatch):
        """The positive path is preserved: caching is narrowed, not removed."""
        from app.config import settings as app_settings

        monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)

        orch = _orchestrator(external_strategy=_FakeStrategy(BIG_REF.decode()))
        _stub_analyzer(orch)
        out = await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True)
        assert is_reusable_verified(
            orch._last_evaluation.sync_state.value,
            orch._last_evaluation.verification.value,
        )

        orch2 = _orchestrator(external_strategy=_FakeStrategy("x"), sync_cache=orch._sync_cache)
        out2 = await orch2.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True)

        assert out2 == out
        assert orch2._external_strategy.calls == 0
        assert orch2._sync_service.calls == 0

    @pytest.mark.asyncio
    async def test_legacy_entry_without_meta_is_not_served(self, monkeypatch):
        """An entry written before this fix has no state; it must not be served."""
        from app.config import settings as app_settings
        from app.services.sync_cache import SyncCache

        monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)

        cache = SyncCache()
        orch = _orchestrator(external_strategy=_FakeStrategy(BIG_REF.decode()), sync_cache=cache)
        _stub_analyzer(orch)
        out = await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True)

        # Simulate a pre-fix cache: payload present, meta absent.
        for key in list(cache._local.keys()):
            cache._local_meta.pop(f"{key}:meta", None)

        orch2 = _orchestrator(
            external_strategy=_FakeStrategy(BIG_REF.decode()),
            sync_cache=cache,
        )
        _stub_analyzer(orch2)
        await orch2.evaluate_and_sync(_arabic_bytes(), _meta(), "t", True)

        assert orch2._external_strategy.calls == 1, "meta-less entry was served as verified"
        assert out is not None


class TestResyncVerdictReuse:
    """A ``verified_resynced`` verdict claims alass DID re-time the subtitle.

    The verdict-reuse branch serves ``sub_bytes`` -- the original. For a
    ``verified_synced`` verdict that is correct (nothing was re-timed). For a
    ``verified_resynced`` verdict it is not: the transformed artifact is the
    synchronized subtitle, and serving the original under a verified-resync
    label hands back an unsynchronized subtitle while claiming success.

    Demonstrated in production flow: with the artifact missing, the branch
    returned the original and reported ``verified_resynced`` / ``cached`` with
    zero provider calls and zero alass runs.
    """

    def _meta(self):
        return {
            "imdb_id": "tt1",
            "media_type": "movie",
            "lang": "ara",
            "target_filename": "Movie.2020.1080p.BluRay.x264-GRP.mkv",
        }

    @staticmethod
    def _verdict(state, artifact_key=None):
        v = {
            "sync_state": state,
            "verification": "verified",
            "alass_applied": state == "verified_resynced",
            "alass_successful": state == "verified_resynced",
            "engine_version": 1,
        }
        if artifact_key is not None:
            v["artifact_key"] = artifact_key
        return v

    @staticmethod
    def _content_hash(sub: bytes) -> str:
        import hashlib

        return hashlib.sha256(sub).hexdigest()[:16]

    @pytest.mark.asyncio
    async def test_resync_verdict_without_artifact_is_not_reused(self, monkeypatch):
        from app.config import settings as app_settings
        from app.services.sync_cache import SyncCache

        monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)

        cache = SyncCache()
        meta = self._meta()
        sub = _arabic_bytes()
        orch0 = _orchestrator(external_strategy=_FakeStrategy(BIG_REF.decode()))
        vkey = orch0._verdict_key(meta, self._content_hash(sub))
        assert vkey, "verdict key must be derivable for this fixture"
        # A resync verdict whose artifact_key points at nothing.
        await cache.set_verdict(vkey, self._verdict("verified_resynced", "final_sub:missing"))

        orch = _orchestrator(
            external_strategy=_FakeStrategy(BIG_REF.decode()), sync_cache=cache
        )
        _stub_analyzer(orch)
        served = await orch.evaluate_and_sync(sub, meta, "release-1", True)
        assert served, "the request must produce a recomputed result, not skip work"

        assert orch._external_strategy.calls == 1, "stale resync verdict skipped re-evaluation"
        ev = orch._last_evaluation
        assert ev is None or ev.sync_state.value != "verified_resynced", (
            "claimed a verified resync while serving bytes that were never re-timed"
        )
        # The verdict was genuinely recomputed, not recalled from cache.
        assert ev is None or ev.verification.value != "cached"

    @pytest.mark.asyncio
    async def test_resync_verdict_with_artifact_serves_transformed_bytes(self, monkeypatch):
        from app.config import settings as app_settings
        from app.services.sync_cache import SyncCache

        monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)

        cache = SyncCache()
        meta = self._meta()
        sub = _arabic_bytes()
        transformed = sub + b"\n2\n00:00:30,000 --> 00:00:31,000\nsynced\n"
        await cache.set("final_sub:resync-artifact", transformed)

        orch0 = _orchestrator(external_strategy=_FakeStrategy(BIG_REF.decode()))
        vkey = orch0._verdict_key(meta, self._content_hash(sub))
        await cache.set_verdict(
            vkey, self._verdict("verified_resynced", "final_sub:resync-artifact")
        )

        orch = _orchestrator(
            external_strategy=_FakeStrategy("x"), sync_cache=cache
        )
        _stub_analyzer(orch)
        served = await orch.evaluate_and_sync(sub, meta, "release-1", True)

        assert served == transformed, "resync reuse must serve the transformed artifact"
        assert orch._external_strategy.calls == 0
        assert orch._last_evaluation.sync_state.value == "verified_resynced"
        assert orch._last_evaluation.verification.value == "cached"

    @pytest.mark.asyncio
    async def test_synced_verdict_still_reuses_without_providers(self, monkeypatch):
        """``verified_synced`` needs no artifact: the original IS the answer."""
        from app.config import settings as app_settings
        from app.services.sync_cache import SyncCache

        monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)

        cache = SyncCache()
        meta = self._meta()
        sub = _arabic_bytes()
        orch0 = _orchestrator(external_strategy=_FakeStrategy(BIG_REF.decode()))
        vkey = orch0._verdict_key(meta, self._content_hash(sub))
        await cache.set_verdict(vkey, self._verdict("verified_synced"))

        orch = _orchestrator(
            external_strategy=_FakeStrategy("x"), sync_cache=cache
        )
        _stub_analyzer(orch)
        served = await orch.evaluate_and_sync(sub, meta, "release-1", True)

        assert served == sub
        assert orch._external_strategy.calls == 0, "no-op verdict should reuse without providers"
        assert orch._last_evaluation.sync_state.value == "verified_synced"


class TestIsolation:
    @pytest.mark.asyncio
    async def test_result_is_not_reused_across_target_identities(self, monkeypatch):
        """Reuse is bound to target identity: a different release must not hit.

        Note ``build_synced_cache_key`` carries no explicit ``lang`` segment, but
        that is unreachable rather than a defect: ``_evaluate_and_sync`` returns
        early for any language that is not Arabic, so every cached entry is
        Arabic by construction. Target identity is the real bound.
        """
        from app.config import settings as app_settings

        monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)

        orch = _orchestrator(external_strategy=_FakeStrategy(BIG_REF.decode()))
        _stub_analyzer(orch)
        await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "ara-release-a", True)

        orch2 = _orchestrator(
            external_strategy=_FakeStrategy(BIG_REF.decode()),
            sync_cache=orch._sync_cache,
        )
        _stub_analyzer(orch2)
        await orch2.evaluate_and_sync(_arabic_bytes(), _meta(), "ara-release-b", True)

        assert orch2._external_strategy.calls == 1, "result was reused across target identities"

    @pytest.mark.asyncio
    async def test_same_target_identity_is_reused_for_verified(self, monkeypatch):
        """The converse guard: the verified warm hit still works for one target."""
        from app.config import settings as app_settings

        monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)

        orch = _orchestrator(external_strategy=_FakeStrategy(BIG_REF.decode()))
        _stub_analyzer(orch)
        await orch.evaluate_and_sync(_arabic_bytes(), _meta(), "ara-release", True)

        orch2 = _orchestrator(
            external_strategy=_FakeStrategy("x"), sync_cache=orch._sync_cache
        )
        _stub_analyzer(orch2)
        await orch2.evaluate_and_sync(_arabic_bytes(), _meta(), "ara-release", True)

        assert orch2._external_strategy.calls == 0, "verified result was not reused"
