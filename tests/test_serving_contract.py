"""The serving contract: a synchronization attempt the verifier does not trust
must never replace the original subtitle in the user response.

Delivery fail-open proven in production flow (Whiplash 2014 2160p REMUX,
``d777e212d36310c7``): alass exited 0, the analyzer returned
``unverified``/``unknown`` with the explicit reason "alass output not
trustworthy" (residual p95 4200 ms against a 2000 ms bound, movement
mad 1106 ms), and the output was served anyway. Timestamp comparison against the
provider bytes showed displacements of +150 748 ms early and +245 289 ms late --
the user received a mangled subtitle in place of a serviceable original.

The analyzer was already correct. The defect was that delivery ignored it.
``_validate_synced_output`` could not catch it because it is structural only
(cue-count preservation, monotonicity, span sanity), and a large non-uniform
shift satisfies all three.

This module pins the contract across every outcome:

* ``VERIFIED`` + ``verified_synced``   -> original (nothing was re-timed)
* ``VERIFIED`` + ``verified_resynced`` -> transformed artifact
* ``VERIFIED`` + ``probable_sync``      -> transformed (accepted, confidence capped)
* ``UNVERIFIED``/``UNKNOWN``            -> original, never the transform
* ``REJECTED``                          -> original, never the transform
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_sync_strategies import (  # noqa: E402
    _arabic_bytes,
    _FakeStrategy,
    _meta,
    _orchestrator,
    _stub_analyzer,
)

from app.services.sync.alignment import (  # noqa: E402
    SyncState,
    is_reusable_verified,
    may_serve_synchronized,
)

BIG = "BIG_REF"


class TestServingContractPredicate:
    """The delivery decision, stated once."""

    def test_verified_outcomes_may_serve_the_transform(self):
        for state in (
            SyncState.VERIFIED_SYNCED,
            SyncState.VERIFIED_RESYNCED,
            SyncState.PROBABLE_SYNC,
        ):
            assert may_serve_synchronized(state.value, "verified") is True

    def test_unverified_never_serves_the_transform(self):
        assert may_serve_synchronized(SyncState.UNVERIFIED.value, "unknown") is False
        assert may_serve_synchronized(SyncState.UNVERIFIED.value, "predicted") is False

    def test_rejected_never_serves_the_transform(self):
        # Even if a measurement is present, REJECTED is not served.
        assert may_serve_synchronized(SyncState.REJECTED.value, "verified") is False
        assert may_serve_synchronized(SyncState.REJECTED.value, "unknown") is False

    def test_missing_verification_fails_closed(self):
        assert may_serve_synchronized(None, None) is False
        assert may_serve_synchronized(SyncState.VERIFIED_RESYNCED.value, None) is False

    def test_delivery_and_reuse_are_deliberately_different_questions(self):
        """``probable_sync`` is served but not reusable -- both are intentional."""
        state, verification = SyncState.PROBABLE_SYNC.value, "verified"
        assert may_serve_synchronized(state, verification) is True
        assert is_reusable_verified(state, verification) is False


class TestUntrustedOutputIsWithheld:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "state,verification",
        [
            (SyncState.UNVERIFIED, "unknown"),
            (SyncState.UNVERIFIED, "predicted"),
            (SyncState.REJECTED, "verified"),
        ],
    )
    async def test_original_is_served_for_untrusted_outcomes(
        self, monkeypatch, state, verification
    ):
        from app.config import settings as app_settings

        monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
        original = _arabic_bytes()

        orch = _orchestrator(external_strategy=_FakeStrategy(BIG_REF_DECODE))
        _stub_analyzer(orch, state=state, verification=verification)
        out = await orch.evaluate_and_sync(original, _meta(), "t", True)

        assert out == original, "untrusted transform replaced the original subtitle"
        assert b"synced" not in out

    @pytest.mark.asyncio
    async def test_untrusted_output_creates_no_reusable_verified_entry(
        self, monkeypatch
    ):
        """The artifact may exist for diagnostics, but must not be reusable."""
        from app.config import settings as app_settings

        monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
        original = _arabic_bytes()
        cache = _orchestrator(external_strategy=_FakeStrategy(BIG_REF_DECODE))._sync_cache

        orch = _orchestrator(
            external_strategy=_FakeStrategy(BIG_REF_DECODE), sync_cache=cache
        )
        _stub_analyzer(orch, state=SyncState.UNVERIFIED, verification="unknown")
        await orch.evaluate_and_sync(original, _meta(), "t", True)

        for key in list(cache._local.keys()):
            meta = cache._local_meta.get(f"{key}:meta")
            assert meta is None or "verified_synced" not in meta

        # A repeat request must not be answered from the cache either.
        orch2 = _orchestrator(
            external_strategy=_FakeStrategy(BIG_REF_DECODE), sync_cache=cache
        )
        _stub_analyzer(orch2, state=SyncState.UNVERIFIED, verification="unknown")
        out2 = await orch2.evaluate_and_sync(original, _meta(), "t", True)
        assert out2 == original
        assert orch2._external_strategy.calls == 1, "cache served an untrusted artifact"


class TestZeroCueTargetFailsClosed:
    """A target the SRT parser cannot read must fail closed, never open.

    Real production case: Dexter S08E03 subtitles ``5eeb876d309a0002`` and
    ``a3ae72e9b3fd15f9`` are Advanced Sub Station Alpha (ASS v4+) scripts that
    are cached under a ``.srt`` name. The sync analyzer parses SRT only, and it
    runs *before* the response-time ``convert_ass_to_srt_bytes`` stage
    (``app/main.py``), so it legitimately sees 0 cues where the response
    contains 550. The result is ``unverified``/``unknown``, the transform is
    withheld, and the original (later converted) is served.

    This is a functional limitation, not a safety defect: an ASS subtitle is
    simply never synchronized. The invariant pinned here is that such a case can
    never produce a trusted synchronization, a reusable verified entry, or
    transformed bytes in the response.
    """

    ASS = (
        "[Script Info]\r\n"
        "; Advanced Sub Station Alpha v4+\r\n"
        "ScriptType: v4.00+\r\n"
        "\r\n"
        "[V4+ Styles]\r\n"
        "Format: Name, Fontname, Fontsize\r\n"
        "Style: Default,Arial,20\r\n"
        "\r\n"
        "[Events]\r\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\r\n"
        + "".join(
            f"Dialogue: 0,0:{i // 60:02d}:{i % 60:02d}.00,0:{i // 60:02d}:{i % 60 + 1:02d}.00,"
            f"Default,,0,0,0,,line {i}\n"
            for i in range(1, 40)
        )
    )

    def test_ass_text_yields_zero_srt_cues(self):
        from app.services.sync.alignment import parse_srt_cues

        assert len(parse_srt_cues(self.ASS)) == 0
        assert self.ASS.count("Dialogue:") == 39

    @pytest.mark.asyncio
    async def test_zero_cue_target_is_unverified_and_original_is_served(self, monkeypatch):
        from app.config import settings as app_settings
        from app.services.sync.alignment import parse_srt_cues

        monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
        original = self.ASS.encode()

        orch = _orchestrator(external_strategy=_FakeStrategy(BIG_REF_DECODE))
        out = await orch.evaluate_and_sync(original, _meta(), "t", True)

        ev = orch._last_evaluation
        assert ev is not None
        assert ev.sync_state.value == "unverified"
        assert ev.verification.value == "unknown"
        assert "0 cues" in " ".join(ev.reasons)
        assert out == original, "a zero-cue target must never yield transformed bytes"
        assert parse_srt_cues(out.decode()) == []

    @pytest.mark.asyncio
    async def test_zero_cue_target_creates_no_reusable_entry(self, monkeypatch):
        from app.config import settings as app_settings

        monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
        cache = _orchestrator(external_strategy=_FakeStrategy(BIG_REF_DECODE))._sync_cache

        orch = _orchestrator(
            external_strategy=_FakeStrategy(BIG_REF_DECODE), sync_cache=cache
        )
        await orch.evaluate_and_sync(self.ASS.encode(), _meta(), "t", True)

        for key in list(cache._local.keys()):
            meta = cache._local_meta.get(f"{key}:meta")
            assert meta is None or "verified_synced" not in meta

        orch2 = _orchestrator(
            external_strategy=_FakeStrategy(BIG_REF_DECODE), sync_cache=cache
        )
        out2 = await orch2.evaluate_and_sync(self.ASS.encode(), _meta(), "t", True)
        assert out2 == self.ASS.encode()
        assert orch2._external_strategy.calls == 1, "a zero-cue entry was served from cache"


class TestTrustedOutputIsServed:
    @pytest.mark.asyncio
    async def test_verified_synced_serves_the_original(self, monkeypatch):
        """Nothing was re-timed, so the original IS the synchronized answer."""
        from app.config import settings as app_settings

        monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
        original = _arabic_bytes()
        # An aligned reference (same cue times as the target) so the already-
        # aligned path runs and no re-timing is attempted.
        aligned_ref = original.decode("utf-8")

        orch = _orchestrator(external_strategy=_FakeStrategy(aligned_ref))
        _stub_analyzer(orch, state=SyncState.VERIFIED_SYNCED)
        out = await orch.evaluate_and_sync(original, _meta(), "t", True)

        assert out == original
        assert orch._sync_service.calls == 0, "an aligned pair must not run alass"

    @pytest.mark.asyncio
    async def test_verified_resync_serves_the_transformed_bytes(self, monkeypatch):
        from app.config import settings as app_settings

        monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
        original = _arabic_bytes()

        orch = _orchestrator(external_strategy=_FakeStrategy(BIG_REF_DECODE))
        _stub_analyzer(orch, state=SyncState.VERIFIED_RESYNCED)
        out = await orch.evaluate_and_sync(original, _meta(), "t", True)

        assert out != original, "verified resync must serve the transformed artifact"
        assert b"synced" in out

    @pytest.mark.asyncio
    async def test_verified_resync_remains_reusable(self, monkeypatch):
        """The fix must not collapse verified reuse into always-serve-original."""
        from app.config import settings as app_settings

        monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
        original = _arabic_bytes()
        cache = _orchestrator(external_strategy=_FakeStrategy(BIG_REF_DECODE))._sync_cache

        orch = _orchestrator(
            external_strategy=_FakeStrategy(BIG_REF_DECODE), sync_cache=cache
        )
        _stub_analyzer(orch, state=SyncState.VERIFIED_RESYNCED)
        first = await orch.evaluate_and_sync(original, _meta(), "t", True)

        orch2 = _orchestrator(
            external_strategy=_FakeStrategy("x"), sync_cache=cache
        )
        _stub_analyzer(orch2, state=SyncState.VERIFIED_RESYNCED)
        second = await orch2.evaluate_and_sync(original, _meta(), "t", True)

        assert first == second, "verified transformed bytes were not reused"
        assert orch2._external_strategy.calls == 0

    @pytest.mark.asyncio
    async def test_probable_sync_keeps_its_serving_semantics(self, monkeypatch):
        """Accepted but confidence-capped: still served, still not reusable.

        Documented deliberately. ``probable_sync`` with a verified measurement
        is an accepted alignment whose confidence was capped (for example by the
        12-cue minimum), not a distrusted one. Serving it preserves the existing
        contract; refusing it would silently disable auto-sync for every
        short-but-valid subtitle.
        """
        from app.config import settings as app_settings

        monkeypatch.setattr(app_settings, "ENABLE_SUBTITLE_SYNC", True)
        original = _arabic_bytes()

        orch = _orchestrator(external_strategy=_FakeStrategy(BIG_REF_DECODE))
        _stub_analyzer(orch, state=SyncState.PROBABLE_SYNC)
        out = await orch.evaluate_and_sync(original, _meta(), "t", True)

        assert out != original, "probable_sync should still serve the transform"
        assert b"synced" in out
        assert is_reusable_verified("probable_sync", "verified") is False


from test_sync_strategies import BIG_REF  # noqa: E402

BIG_REF_DECODE = BIG_REF.decode()
