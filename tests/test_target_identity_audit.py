"""Target-identity construction: the fingerprint input audit (Phase 3).

``SyncCache.video_fingerprint_from_meta(meta)`` hashes whatever fields the
caller supplies:

    imdb_id, season, episode, target_filename, video_hash, video_size

There is no canonical field list, so two call sites that build the dict
differently derive DIFFERENT identities for the SAME target. This file pins the
production call sites against each other, because a divergence here silently
breaks cache reuse in a way no single-site test can see.

Found by the audit, in code added in the previous phase:

    orchestrator.py  writes the verdict alias from the full ``meta``, which
                     carries ``video_hash`` and ``video_size``
    main.py:1288     read the alias from a 4-field dict with neither

Stremio always supplies ``videosize``, so on a normal request the two
fingerprints can never agree and the pre-download reuse lookup can never find
the verdict it is meant to reuse. The reuse path was unreachable in production.
"""

from __future__ import annotations

import pytest

from app.services.sync_cache import SyncCache

IMDB = "tt0773262"
SEASON = 8
EPISODE = 5
FILENAME = "Dexter.s8e05.This.little.piggy.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv"
VIDEO_SIZE = 5192387053
VIDEO_HASH = "0123456789abcdef0123456789abcdef"
SUB_ID = "63df03613463c531"
LANG = "ara"


def _full_meta(**overrides) -> dict:
    """The shape the orchestrator uses: the real request metadata."""
    meta = {
        "imdb_id": IMDB,
        "season": SEASON,
        "episode": EPISODE,
        "target_filename": FILENAME,
        "video_hash": VIDEO_HASH,
        "video_size": VIDEO_SIZE,
    }
    meta.update(overrides)
    return meta


def _reduced_meta() -> dict:
    """The shape main.py:1288 used to build for the reuse lookup."""
    return {
        "imdb_id": IMDB,
        "season": SEASON,
        "episode": EPISODE,
        "target_filename": FILENAME,
    }


# --- the divergence itself -------------------------------------------------


def test_reduced_meta_cannot_reproduce_the_serve_time_fingerprint():
    """The defect, stated as an executable claim.

    A verdict alias written from the real request meta is unreachable through a
    fingerprint built from the reduced dict.
    """
    serve_fp = SyncCache.video_fingerprint_from_meta(_full_meta())
    reduced_fp = SyncCache.video_fingerprint_from_meta(_reduced_meta())
    assert serve_fp != reduced_fp, (
        "if these ever become equal this test is obsolete and the call sites "
        "should be consolidated into one canonical constructor"
    )


@pytest.mark.asyncio
async def test_reuse_lookup_finds_a_verdict_written_by_the_orchestrator():
    """End-to-end proof that the reuse path was unreachable in production."""
    from app.main import (
        _fallback_candidate_ref,
        _reusable_verified_fallback_sync,
        _sync_cache,
    )
    from app.models import SubtitleRelease
    from app.services.sync.alignment import SyncState

    for store in ("_local", "_local_meta", "_local_fail", "_local_verdict"):
        getattr(_sync_cache, store).clear()

    release = SubtitleRelease(
        release_name="Dexter.S08E05.HDTV.XviD-AFG.srt",
        download_url="https://subsource.test/x.srt",
        provider="subsource",
        lang=LANG,
    )
    ref = _fallback_candidate_ref(release, "subsource", SEASON, EPISODE)

    # Written exactly as the orchestrator writes it: full meta, with the
    # video_size Stremio actually sends.
    serve_fp = SyncCache.video_fingerprint_from_meta(_full_meta())
    await _sync_cache.set_verdict(
        SyncCache.build_verdict_key(serve_fp, "h", LANG),
        {
            "sync_state": SyncState.VERIFIED_SYNCED.value,
            "verification": "verified",
            "language": LANG,
        },
        alias_key=SyncCache.build_verdict_alias_key(serve_fp, ref),
    )
    await _sync_cache.set(
        SyncCache.build_key(IMDB, f"s{SEASON}e{EPISODE}", serve_fp, ref, "edition", "chash"),
        b"1\n00:00:10,000 --> 00:00:12,000\nx\n",
    )

    served = await _reusable_verified_fallback_sync(
        release=release, imdb_id=IMDB, season=SEASON, episode=EPISODE,
        lang=LANG, target_filename=FILENAME,
        video_hash=VIDEO_HASH, video_size=VIDEO_SIZE,
    )
    assert served == b"1\n00:00:10,000 --> 00:00:12,000\nx\n", (
        "a verified verdict written from the real request meta must be "
        "reachable by the pre-download reuse lookup; Stremio always sends "
        "videosize, so a divergence here disables reuse in production"
    )


# --- guards against the fix regressing --------------------------------------


def test_orchestrator_and_reuse_now_agree():
    """After the fix both paths must derive the SAME identity."""
    from app.main import _target_fingerprint_meta

    reuse_meta = _target_fingerprint_meta(
        {
            "imdb_id": IMDB, "season": SEASON, "episode": EPISODE,
            "target_filename": FILENAME, "video_hash": VIDEO_HASH,
            "video_size": VIDEO_SIZE,
        }
    )
    assert SyncCache.video_fingerprint_from_meta(
        dict(_full_meta())
    ) == SyncCache.video_fingerprint_from_meta(reuse_meta), (
        "the reuse lookup and the serve path must construct target identity "
        "identically, otherwise cache reuse silently stops working"
    )


def test_fingerprint_is_stable_for_identical_meta():
    assert SyncCache.video_fingerprint_from_meta(
        _full_meta()
    ) == SyncCache.video_fingerprint_from_meta(_full_meta())


@pytest.mark.parametrize(
    "field", ["imdb_id", "season", "episode", "target_filename", "video_hash", "video_size"]
)
def test_every_identity_field_actually_participates(field):
    """Each field must change the identity, or it is dead weight in the hash."""
    base = SyncCache.video_fingerprint_from_meta(_full_meta())
    changed = SyncCache.video_fingerprint_from_meta(
        _full_meta(**{field: "different-value"})
    )
    assert base != changed, f"{field} is not part of the target identity"


def test_absent_video_hash_still_yields_an_identity():
    """A request with no video hash must remain usable, not be refused."""
    meta = _full_meta(video_hash="", video_size=None)
    assert SyncCache.video_fingerprint_from_meta(meta)
