"""Cache-consistency regressions from the Dexter S08E05 14:08/14:11 trace.

A production trace recorded, for the same subtitle identity and the same target:

    14:08  [sync] cache HIT for sub=63df03613463c531 -> serving pre-synced ...
    14:11  [sync] evaluate sub=63df03613463c531
           [reference] resolving ...            <- full re-sync, reference re-downloaded

Both halves of the production lookup are exercised here without any network:
``SyncOrchestrator._flight_key`` and ``SyncCache`` itself, using the real
identity shape from the incident.

What the audit established, and what these tests pin:

1. The payload store is IN-MEMORY when no Redis is configured, so any process
   or container restart discards every ``final_sub:`` entry. TTL is 24h, so a
   three-minute gap cannot be expiry.
2. The key is a composite: content hash of the ORIGINAL payload bytes, the
   video fingerprint, the provider-credential digest, and a context digest
   covering (media_type, target_filename, video_hash, video_size, lang). Every
   one of those is an independent input, and a change in any of them is a
   silent miss.
3. With no video hash and no stream URL, the key's "video" segment degrades to
   the target filename, and failing that to the SUBTITLE id, which is not video
   identity at all.

None of these are fixed by changing key semantics; the task forbids that, and
the risk is that they are invisible. These tests make the boundaries explicit
so a future trace can point at the input that moved.
"""

from __future__ import annotations

import pytest

from app.services.sync.orchestrator import (
    _fingerprint_source,
    build_synced_cache_key,
)
from app.services.sync_cache import SyncCache

# The exact identity shape from the incident.
IMDB = "tt0773262"
SEASON = 8
EPISODE = 5
FILENAME = "Dexter.s8e05.This.little.piggy.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv"
VIDEO_SIZE = 5192387053
SUB_ID = "63df03613463c531"

META = {
    "imdb_id": IMDB,
    "season": SEASON,
    "episode": EPISODE,
    "media_type": "series",
    "target_filename": FILENAME,
    "video_size": VIDEO_SIZE,
    "lang": "ara",
    "subdl_key": "k1",
    "subsource_key": "k2",
    "opensubtitles_key": "k3",
}

SUB_BYTES = b"1\n00:00:10,000 --> 00:00:12,000\nline\n"


def _srt(n: int = 20) -> bytes:
    out = []
    for i in range(n):
        s = i * 2000
        out.append(f"{i + 1}\n00:00:{s // 60 % 60:02d},{s % 1000:03d} --> 00:00:12,000\nx\n")
    return "".join(out).encode()


# --- Part 8: repeated identical request, in-process -------------------------


@pytest.mark.asyncio
async def test_repeated_identical_request_is_a_stable_cache_hit():
    """Same process, same inputs: the second request must hit.

    This is the invariant the 14:11 request broke. Within one running process
    with no input drift, a miss on the second identical request is a bug.
    """
    from app.services.sync.orchestrator import SyncOrchestrator

    cache = SyncCache(ttl=3600)
    assert cache.is_ephemeral is True, "no Redis configured, so payloads are in-process"

    orch = SyncOrchestrator(sync_cache=cache)
    key = orch._flight_key(dict(META), SUB_ID, SUB_BYTES)
    assert await cache.get(key) is None

    await cache.set(key, SUB_BYTES)
    first = await cache.get(key)
    second = await cache.get(key)
    assert first == SUB_BYTES
    assert second == first, "an identical in-process lookup must be stable"


@pytest.mark.asyncio
async def test_a_fresh_cache_instance_has_no_payload():
    """Proves the lifecycle cause directly: a new process cannot see it.

    This is the mechanism that would turn a 14:08 hit into a 14:11 miss without
    any key changing. It is asserted rather than assumed, because the two cases
    produce identical log output.
    """
    warm = SyncCache(ttl=3600)
    from app.services.sync.orchestrator import SyncOrchestrator

    key = SyncOrchestrator(sync_cache=warm)._flight_key(dict(META), SUB_ID, SUB_BYTES)
    await warm.set(key, SUB_BYTES)
    assert await warm.get(key) == SUB_BYTES

    cold = SyncCache(ttl=3600)  # simulates a restarted process
    assert await cold.get(key) is None, (
        "payload store is process-local; a restart must look like a miss"
    )


# --- Part 7 B: target binding is mandatory ---------------------------------


@pytest.mark.asyncio
async def test_b_same_sub_id_different_target_must_miss():
    """Test B, mandatory: target A's artifact must never serve target B."""
    cache = SyncCache(ttl=3600)
    from app.services.sync.orchestrator import SyncOrchestrator

    orch = SyncOrchestrator(sync_cache=cache)
    key_a = orch._flight_key(dict(META), SUB_ID, SUB_BYTES)
    await cache.set(key_a, SUB_BYTES)

    other = dict(META, target_filename="Dexter.S08E05.720p.BluRay.x264-OTHER.mkv")
    key_b = orch._flight_key(other, SUB_ID, SUB_BYTES)
    assert key_a != key_b
    assert await cache.get(key_b) is None


@pytest.mark.asyncio
async def test_artifact_lookup_is_fingerprint_scoped():
    """The prefix-scan artifact lookup must respect the video fingerprint."""
    cache = SyncCache(ttl=3600)
    fp_a = "a" * 24
    fp_b = "b" * 24
    key = SyncCache.build_key(IMDB, f"s{SEASON}e{EPISODE}", fp_a, SUB_ID, "edition", "chash")
    await cache.set(key, SUB_BYTES)
    assert await cache.find_synced_for_target(IMDB, f"s{SEASON}e{EPISODE}", fp_a, SUB_ID) == SUB_BYTES
    assert await cache.find_synced_for_target(IMDB, f"s{SEASON}e{EPISODE}", fp_b, SUB_ID) is None


# --- Part 7 D/E: negatives and engine binding ------------------------------


@pytest.mark.asyncio
async def test_d_rejected_verdict_with_artifact_present_is_not_served():
    """Drives the real serving boundary, not the stored string.

    The previous version of this test asserted only that a stored verdict
    CONTAINS the string "rejected". That proved nothing: mutating
    ``set_verdict`` to treat REJECTED as verified did not fail it, because the
    write-side guard is not the decision that matters.

    The decision lives in ``_reusable_verified_fallback_sync``. A matching
    artifact is deliberately seeded under the SAME target and subtitle
    identity, so if the boundary is bypassed the artifact is returned and this
    test fails. Mutating that gate to accept non-positive verdicts fails all
    three states.
    """
    from app.main import (
        _fallback_candidate_ref,
        _reusable_verified_fallback_sync,
        _sync_cache,
    )
    from app.models import SubtitleRelease
    from app.services.sync.alignment import SyncState

    for store in ("_local", "_local_meta", "_local_fail", "_local_verdict"):
        getattr(_sync_cache, store).clear()

    # The fingerprint MUST be derived from exactly the fields the boundary
    # uses, or every key below is unreachable and the test passes for the
    # wrong reason. video_fingerprint_from_meta hashes any extra fields it is
    # given (video_size, video_hash, stream_url), so supplying a wider meta
    # than the boundary does produces a different identity.
    boundary_meta = {
        "imdb_id": IMDB,
        "season": SEASON,
        "episode": EPISODE,
        "target_filename": FILENAME,
    }
    fingerprint = SyncCache.video_fingerprint_from_meta(dict(boundary_meta))
    assert fingerprint
    assert fingerprint == SyncCache.video_fingerprint_from_meta(dict(boundary_meta))

    release = SubtitleRelease(
        release_name="Dexter.S08E05.HDTV.XviD-AFG.srt",
        download_url="https://subsource.test/x.srt",
        provider="subsource",
        lang="ara",
    )
    # The id the boundary actually derives. Seeding any other id would make the
    # lookup miss on the key and pass for the wrong reason, which is exactly
    # how the earlier version of this test was vacuous.
    ref = _fallback_candidate_ref(release, "subsource", SEASON, EPISODE)
    assert ref and ref != SUB_ID

    artifact = b"1\n00:00:10,000 --> 00:00:12,000\nx\n"
    payload_key = SyncCache.build_key(
        IMDB, f"s{SEASON}e{EPISODE}", fingerprint, ref, "edition", "chash"
    )
    await _sync_cache.set(payload_key, artifact)
    assert await _sync_cache.find_synced_for_target(
        IMDB, f"s{SEASON}e{EPISODE}", fingerprint, ref
    ) == artifact, "the artifact must be genuinely reachable"

    async def serve_with(state: str) -> bytes | None:
        for store in ("_local", "_local_verdict"):
            getattr(_sync_cache, store).clear()
        await _sync_cache.set_verdict(
            SyncCache.build_verdict_key(fingerprint, "h", "ara"),
            {"sync_state": state, "verification": "verified", "language": "ara"},
            alias_key=SyncCache.build_verdict_alias_key(fingerprint, ref),
        )
        await _sync_cache.set(payload_key, artifact)
        return await _reusable_verified_fallback_sync(
            release=release, imdb_id=IMDB, season=SEASON, episode=EPISODE,
            lang="ara", target_filename=FILENAME,
        )
    # Positive control: identical setup, verified verdict -> artifact served.
    # Without this, a setup that silently misses the key would make the
    # negative assertion below meaningless.
    assert await serve_with(SyncState.VERIFIED_SYNCED.value) == artifact

    for non_positive in (
        SyncState.REJECTED.value,
        SyncState.UNVERIFIED.value,
        SyncState.PROBABLE_SYNC.value,
    ):
        assert await serve_with(non_positive) is None, (
            f"{non_positive} must never yield a served artifact, even when a "
            "matching artifact exists for the exact same target and subtitle"
        )


@pytest.mark.asyncio
async def test_e_stale_engine_version_is_rejected_not_silently_accepted():
    """Not vacuous: the alias is written under the CURRENT engine version and
    the payload claims a different one, which is what an older build leaves."""
    import json

    from app.services import sync_cache as sc
    from app.services.sync.alignment import SyncState

    cache = SyncCache(ttl=3600)
    fingerprint = SyncCache.video_fingerprint_from_meta(dict(META))
    stale = sc.SYNC_VERDICT_ENGINE_VERSION + 1
    cache._local_verdict[SyncCache.build_verdict_alias_key(fingerprint, SUB_ID)] = json.dumps(
        {
            "sync_state": SyncState.VERIFIED_SYNCED.value,
            "verification": "verified",
            "language": "ara",
            "engine_version": stale,
        }
    )
    assert await cache.get_verdict_by_ref(fingerprint, SUB_ID) is None


# --- Part 7 H: ambiguous artifacts ------------------------------------------


@pytest.mark.asyncio
async def test_h_multiple_artifacts_for_same_target_and_sub_id():
    """Documented selection behaviour; must not depend on iteration order."""
    cache = SyncCache(ttl=3600)
    fp = "c" * 24
    season_ep = f"s{SEASON}e{EPISODE}"
    await cache.set(
        SyncCache.build_key(IMDB, season_ep, fp, SUB_ID, "edition", "hash_a"), b"artifact-a"
    )
    await cache.set(
        SyncCache.build_key(IMDB, season_ep, fp, SUB_ID, "team", "hash_b"), b"artifact-b"
    )
    found = await cache.find_synced_for_target(IMDB, season_ep, fp, SUB_ID)
    assert found in (b"artifact-a", b"artifact-b"), (
        "ambiguous result must still resolve to an artifact of this exact target "
        "and subtitle; the prefix scan is inherently first-match"
    )
    # Repeat lookups are stable, which is the property that actually matters.
    again = await cache.find_synced_for_target(IMDB, season_ep, fp, SUB_ID)
    assert again == found


# --- the composite-key drift surface ----------------------------------------


def test_key_changes_when_original_payload_bytes_change():
    """The key embeds the ORIGINAL payload hash.

    This is a genuine non-determinism: if a provider download fails and the
    fallback returns different bytes, the same logical subtitle and target
    produce a different key and therefore a guaranteed miss. It is a safe
    direction (never a false hit) but it is a silent one.
    """
    from app.services.sync.orchestrator import SyncOrchestrator

    orch = SyncOrchestrator(sync_cache=None)
    k1 = orch._flight_key(dict(META), SUB_ID, b"original bytes")
    k2 = orch._flight_key(dict(META), SUB_ID, b"fallback bytes")
    assert k1 != k2


def test_key_changes_when_provider_credentials_change():
    from app.services.sync.orchestrator import SyncOrchestrator

    orch = SyncOrchestrator(sync_cache=None)
    rotated = dict(META, subdl_key="rotated")
    assert orch._flight_key(dict(META), SUB_ID, SUB_BYTES) != orch._flight_key(
        rotated, SUB_ID, SUB_BYTES
    )


def test_key_changes_when_target_filename_is_absent():
    """Absent filename degrades the binding, and the log must say so."""
    from app.services.sync.orchestrator import SyncOrchestrator

    orch = SyncOrchestrator(sync_cache=None)
    named = orch._flight_key(dict(META), SUB_ID, SUB_BYTES)
    anonymous = orch._flight_key({k: v for k, v in META.items() if k != "target_filename"},
                                 SUB_ID, SUB_BYTES)
    assert named != anonymous
    assert _fingerprint_source(dict(META), SUB_ID) == "target_filename"
    assert _fingerprint_source({k: v for k, v in META.items() if k != "target_filename"},
                               SUB_ID) == "target_id", (
        "the weakest source is the SUBTITLE id, and must be named as such so "
        "it is never mistaken for a cryptographic video fingerprint"
    )


def test_fingerprint_source_hierarchy_is_explicit():
    """Each supported fallback is labelled, strongest first."""
    assert _fingerprint_source({**META, "video_hash": "abc"}, SUB_ID) == "video_hash"
    assert _fingerprint_source({**META, "stream_url": "http://x/y"}, SUB_ID) == "stream_context"
    assert _fingerprint_source(dict(META), SUB_ID) == "target_filename"
    assert _fingerprint_source({"imdb_id": IMDB}, SUB_ID) == "target_id"


def test_weaker_fingerprint_sources_are_still_accepted():
    """Labelling must not become rejection.

    A request with no video hash is ordinary playback, not an error, and must
    still produce a usable key rather than being refused.
    """
    from app.services.sync.orchestrator import SyncOrchestrator

    orch = SyncOrchestrator(sync_cache=None)
    for meta in (
        dict(META),
        {**META, "stream_url": "http://host/stream.m3u8?token=abc"},
        {k: v for k, v in META.items() if k != "target_filename"},
    ):
        assert orch._flight_key(meta, SUB_ID, SUB_BYTES)


def test_key_is_stable_for_identical_inputs():
    from app.services.sync.orchestrator import SyncOrchestrator

    orch = SyncOrchestrator(sync_cache=None)
    a = orch._flight_key(dict(META), SUB_ID, SUB_BYTES)
    b = orch._flight_key(dict(META), SUB_ID, SUB_BYTES)
    assert a == b, "key construction itself must be deterministic"


def test_build_synced_cache_key_shape_is_unchanged():
    """The documented key layout must not drift silently."""
    key = build_synced_cache_key(dict(META), SUB_ID, content_hash="c0ffee", decision="edition")
    assert key.startswith(f"final_sub:{IMDB}:s{SEASON}e{EPISODE}:")
    assert key.endswith(":edition:c0ffee")


# --- Part 9: semantic invariants still hold --------------------------------


@pytest.mark.asyncio
async def test_verdict_lookup_requires_language_match_when_alias_lacks_it():
    from app.services.sync.alignment import SyncState

    cache = SyncCache(ttl=3600)
    fingerprint = SyncCache.video_fingerprint_from_meta(dict(META))
    await cache.set_verdict(
        SyncCache.build_verdict_key(fingerprint, "h", "eng"),
        {"sync_state": SyncState.VERIFIED_SYNCED.value, "verification": "verified",
         "language": "eng"},
        alias_key=SyncCache.build_verdict_alias_key(fingerprint, SUB_ID),
    )
    verdict = await cache.get_verdict_by_ref(fingerprint, SUB_ID)
    assert verdict is not None
    assert verdict["language"] == "eng", (
        "the alias key carries no language, so language must remain readable "
        "from the payload for the caller to reject a mismatch"
    )
    assert verdict["language"] != "ara"
