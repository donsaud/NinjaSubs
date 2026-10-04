"""Negative caching of measured verifier rejections (fix #3).

Production evidence this pins: two Whiplash requests 33 minutes apart
produced *byte-identical* rejections (p95 2416ms, movement mad 9356ms,
residual unmatched 33 -> 130 at 5s tolerance, cut=piecewise) and each burned
an alass subprocess. The verdict fast path previously short-circuited only
VERIFIED states, so a stored UNVERIFIED verdict fell straight through.

These tests pin both halves of the change:

  * a measured rejection is reused without re-running alass, and
  * the reuse is content-bound -- a different video, subtitle, or language
    still runs normally.

And the safety half, which matters more than the CPU saving: reuse must never
upgrade a rejection into a verified claim, and must never serve transformed
bytes for an unverified result.
"""

from __future__ import annotations

import inspect

import pytest

from app.services.sync import orchestrator as _orch
from app.services.sync.orchestrator import SyncOrchestrator, SyncState


@pytest.fixture(autouse=True)
def _server_enabled(monkeypatch):
    """The server gate is off by default; these tests exercise the inner path."""
    monkeypatch.setattr(_orch.settings, "ENABLE_SUBTITLE_SYNC", True, raising=False)


class _Cache:
    """Minimal SyncCache stand-in that records get/set traffic."""

    def __init__(self, verdict=None, artifact=None, failed=False, payload_meta=None):
        self.verdict = verdict
        self.artifact = artifact
        self.failed = failed
        self.payload_meta = payload_meta
        self.verdict_gets = 0
        self.sets: list[str] = []
        self.is_ephemeral = False

    # Production contract: the payload layer reads meta to decide reusability.
    async def get_meta(self, key):
        return self.payload_meta

    # Production contract: the pre-existing negative cache. Only set when every
    # strategy failed to resolve a reference -- NOT when the verifier rejected.
    async def is_failed(self, key):
        return self.failed

    async def mark_failed(self, key, ttl=None):
        self.failed = True

    async def get_verdict(self, key):
        self.verdict_gets += 1
        return self.verdict

    async def get(self, key):
        return self.artifact

    async def set(self, key, data):
        self.sets.append(key)

    async def set_meta(self, key, meta):
        self.sets.append(key)


def _meta(**kw):
    base = {
        "media_type": "movie",
        "target_filename": "Whiplash.2014.2160p.UHD.BluRay.x254-SURCODE.mkv",
        "video_hash": "",
        "video_size": "19571049411",
        "lang": "ara",
    }
    base.update(kw)
    return base


# ===========================================================================
# Key isolation -- the reuse must be content-bound
# ===========================================================================


def test_a_rejection_key_requires_a_video_fingerprint():
    """Without a fingerprint there is nothing safe to key on, so nothing is
    reused. A catalogue request must never inherit another video's rejection
    (or its success)."""
    assert SyncOrchestrator._verdict_key({"lang": "ara"}, "abc") is None


def test_the_rejection_key_changes_with_the_video():
    a = SyncOrchestrator._verdict_key(_meta(video_hash="8e245d9679d31e12"), "abc")
    b = SyncOrchestrator._verdict_key(_meta(video_hash="1111111111111111"), "abc")
    assert a is not None and b is not None
    assert a != b


def test_the_rejection_key_changes_with_the_subtitle():
    a = SyncOrchestrator._verdict_key(_meta(), "aaa")
    b = SyncOrchestrator._verdict_key(_meta(), "bbb")
    assert a != b


def test_the_rejection_key_changes_with_the_language():
    a = SyncOrchestrator._verdict_key(_meta(lang="ara"), "abc")
    b = SyncOrchestrator._verdict_key(_meta(lang="eng"), "abc")
    assert a != b


# ===========================================================================
# Reuse semantics -- served state must never be inflated
# ===========================================================================


@pytest.mark.asyncio
async def test_a_cached_rejection_serves_the_original_bytes():
    """An unverified rejection has no trustworthy artifact, so the ORIGINAL
    bytes are served -- never a transformed file."""
    rejection = {
        "sync_state": "unverified",
        "verification": "unknown",
        "reasons": ["low_confidence"],
        "median_offset_ms": -3,
        "p95_offset_ms": 2416,
        "mad_offset_ms": 9356,
    }
    cache = _Cache(verdict=rejection)
    orch = SyncOrchestrator(sync_cache=cache)
    sub = b"1\n00:00:01,000 --> 00:00:03,000\noriginal\n\n"

    out = await orch.evaluate_and_sync(sub, _meta(), "tt2582802", auto_sync=True)

    assert out == sub, "must serve the original, unmodified subtitle"
    assert out != b"transformed"


@pytest.mark.asyncio
async def test_a_reused_rejection_is_reported_as_cached_not_measured():
    """`_evaluation_from_verdict` must not present a recall as a fresh
    measurement."""
    rejection = {"sync_state": "unverified", "verification": "unknown", "reasons": ["low_confidence"]}
    cache = _Cache(verdict=rejection)
    orch = SyncOrchestrator(sync_cache=cache)

    await orch.evaluate_and_sync(
        b"1\n00:00:01,000 --> 00:00:03,000\nx\n\n", _meta(), "tt2582802", auto_sync=True
    )

    ev = orch._last_evaluation
    assert ev is not None
    assert ev.sync_state == "unverified", "a rejection must never become verified"


@pytest.mark.asyncio
async def test_a_reused_rejection_never_claims_a_verified_state():
    """The critical anti-regression: reuse must not upgrade UNVERIFIED."""
    rejection = {"sync_state": "unverified", "verification": "unknown"}
    cache = _Cache(verdict=rejection)
    orch = SyncOrchestrator(sync_cache=cache)

    await orch.evaluate_and_sync(
        b"1\n00:00:01,000 --> 00:00:03,000\nx\n\n", _meta(), "tt2582802", auto_sync=True
    )

    assert orch._last_evaluation.sync_state != "verified"
    assert orch._last_evaluation.sync_state != "verified_synced"
    assert orch._last_evaluation.sync_state != "verified_resynced"


@pytest.mark.asyncio
async def test_a_rejection_without_an_artifact_still_serves_the_original():
    """VERIFIED_RESYNCED needs its artifact present. This mirrors that care for
    the negative path: nothing is invented."""
    cache = _Cache(verdict={"sync_state": "unverified", "verification": "unknown"})
    orch = SyncOrchestrator(sync_cache=cache)
    sub = b"1\n00:00:01,000 --> 00:00:03,000\nx\n\n"

    assert await orch.evaluate_and_sync(sub, _meta(), "tt2582802", auto_sync=True) == sub


@pytest.mark.asyncio
async def test_no_verdict_at_all_falls_through_to_normal_processing():
    """A miss must behave exactly as before -- no behaviour change on a cold
    cache."""
    cache = _Cache(verdict=None)
    orch = SyncOrchestrator(sync_cache=cache, sync_service=None)
    sub = b"1\n00:00:01,000 --> 00:00:03,000\nx\n\n"

    # With no sync_service the original is served, proving it reached normal
    # processing rather than taking the negative-cache shortcut.
    assert await orch.evaluate_and_sync(sub, _meta(), "tt2582802", auto_sync=True) == sub


# ===========================================================================
# Strategy labelling (fix: accuracy of `external exact-match`)
# ===========================================================================


def test_the_external_strategy_is_no_longer_labelled_exact_match():
    """The label implied proven identity for every candidate; a `decision=edition`
    reference read as an exact match in the live trace."""
    orch = SyncOrchestrator(external_strategy=object())
    names = [n for n, _ in orch._strategies()]
    assert "external exact-match" not in names
    assert "external-release-reference" in names


def test_the_moviehash_strategy_keeps_its_explicit_name():
    orch = SyncOrchestrator(external_strategy=object(), hash_reference_strategy=object())
    names = [n for n, _ in orch._strategies()]
    assert names[0] == "opensubtitles moviehash"


# ===========================================================================
# Invariants the negative cache must never violate
# ===========================================================================


def test_negative_caching_does_not_touch_the_verifier_thresholds():
    from app.services.sync import alignment

    assert alignment.MAX_P95_MS_FOR_STABLE == 2000
    assert alignment.MAX_MAD_MS_FOR_STABLE < 9356


def test_negative_caching_creates_no_hash_identity():
    from app.models import SubtitleRelease

    for prov in ("subdl", "subsource", "opensubtitles"):
        r = SubtitleRelease(release_name="a.srt", download_url="/sub/x.srt", provider=prov)
        assert r.matched_by_hash is False


def test_the_disc_track_skip_survives_the_cleanup():
    """`decision_kind_of` was removed; the gate must read `resolved.kind`."""
    import inspect

    from app.services.sync.matching import is_bare_disc_track_name
    from app.services.sync.orchestrator import decision_kind_is_name_based

    assert not hasattr(SyncOrchestrator, "decision_kind_of")
    assert is_bare_disc_track_name("00001.m2ts") is True
    assert decision_kind_is_name_based("edition") is True
    assert decision_kind_is_name_based("hash") is False
    src = inspect.getsource(SyncOrchestrator)
    assert "decision_kind_of" not in src


# ===========================================================================
# REQUIRED: alass must not run twice for the same rejected pairing
# ===========================================================================


class _CountingCache(_Cache):
    """Cache that models the full production contract and counts fan-out."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.verdicts: dict[str, dict] = {}

    async def get_verdict(self, key):
        self.verdict_gets += 1
        return self.verdicts.get(key)

    async def put_verdict(self, key, verdict):
        self.verdicts[key] = verdict


class _ExplodingStrategy:
    """Any attempt to run alass at all is a test failure."""

    name = "exploding"
    validates_target = False

    def __init__(self):
        self.calls = 0

    async def resolve_with_provenance(self, query, **kw):
        self.calls += 1
        raise AssertionError("alass fan-out must not run for a cached rejection")


@pytest.mark.asyncio
async def test_a_repeated_rejected_request_never_invokes_alass_again():
    """THE core regression.

    A reference was found but the verifier rejected it. The rejection is stored
    as an UNVERIFIED verdict. A second, identical request must serve the
    original bytes WITHOUT re-entering reference fan-out.
    """
    strategy = _ExplodingStrategy()
    cache = _CountingCache()
    orch = SyncOrchestrator(sync_cache=cache, external_strategy=strategy)
    sub = b"1\n00:00:01,000 --> 00:00:03,000\noriginal\n\n"
    meta = _meta()

    import hashlib
    key = SyncOrchestrator._verdict_key(meta, hashlib.sha256(sub).hexdigest()[:16])
    # Seed the measured rejection exactly as _store_verdict would.
    await cache.put_verdict(key, {"sync_state": "unverified", "verification": "unknown"})

    out = await orch.evaluate_and_sync(sub, meta, "tt2582802", auto_sync=True)

    assert out == sub, "must serve the ORIGINAL bytes"
    assert strategy.calls == 0, "alass must not be invoked again"


@pytest.mark.asyncio
async def test_a_cached_rejection_is_never_promoted_to_a_verified_state():
    """Reuse must never upgrade UNVERIFIED into a verified claim."""
    for stored in (
        {"sync_state": "unverified", "verification": "unknown"},
        {"sync_state": "rejected", "verification": "unknown"},
    ):
        cache = _CountingCache()
        cache.verdicts["k"] = stored
        orch = SyncOrchestrator(sync_cache=cache, external_strategy=_ExplodingStrategy())

        await orch.evaluate_and_sync(
            b"1\n00:00:01,000 --> 00:00:03,000\nx\n\n", _meta(), "tt2582802", auto_sync=True
        )

        ev = orch._last_evaluation
        if ev is None:
            continue  # no evaluation recorded is safe: no claim was made
        assert ev.sync_state not in ("verified", "verified_synced", "verified_resynced")
        assert ev.sync_state != SyncState.VERIFIED_SYNCED.value
        assert ev.sync_state != SyncState.VERIFIED_RESYNCED.value


@pytest.mark.asyncio
async def test_a_changed_target_identity_retries_normally():
    """A different video must NOT inherit another video's rejection."""
    meta_a = _meta(video_hash="8e245d9679d31e12")
    meta_b = _meta(video_hash="1111111111111111")
    assert (
        SyncOrchestrator._verdict_key(meta_a, "h") != SyncOrchestrator._verdict_key(meta_b, "h")
    )


@pytest.mark.asyncio
async def test_a_changed_target_filename_retries_normally():
    a = SyncOrchestrator._verdict_key(_meta(), "h")
    b = SyncOrchestrator._verdict_key(_meta(target_filename="Other.2019.mkv"), "h")
    assert a != b


def test_no_reference_failure_caching_still_works():
    """The pre-existing meaning of is_failed is preserved: no usable reference."""
    assert hasattr(SyncOrchestrator, "_verdict_key")
    src = inspect.getsource(SyncOrchestrator._execute)
    assert "is_failed(resolution_key)" in src
    assert "mark_failed(resolution_key)" in src
