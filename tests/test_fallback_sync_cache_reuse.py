"""Pre-download reuse of an existing verified sync result, on the fallback path.

Scope: avoid spending a provider download and an alass run when this exact
candidate has already been proven synchronized against this exact target.

The audit behind these tests established one thing that constrains everything
else. ``SyncCache`` has two separate stores:

* ``_local_verdict`` -- JSON *verdicts* keyed
  ``verdict:{engine}:{video_fingerprint}:{subtitle_hash}:{lang}``, plus a
  fingerprint+engine alias keyed by the 16-hex candidate id. The alias exists
  precisely so a verdict can be found before the subtitle bytes exist.
* ``_local`` -- the actual subtitle *bytes*, keyed
  ``final_sub:{imdb}:{season_ep}:{fingerprint}:{sub_id}[:{decision}][:{hash}]``.

So a verdict is findable before download, but the payload is content-addressed
and is not. Reuse therefore requires a verified verdict AND a reachable
artifact; a verdict alone must not be treated as a payload.
"""

from __future__ import annotations

import hashlib
from unittest.mock import patch

import pytest

from app.models import SubtitleRelease
from app.services.sync.alignment import SyncState
from app.services.sync_cache import SyncCache

TARGET = "Dexter.s8e05.This.little.piggy.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv"
REQUESTED = "Dexter.S08E05.HDTV.XviD-AFG.srt"
RELEASE = "Dexter.S08E05.1080p.BluRay.x265-ROVERS.srt"
URL = "https://subsource.test/rovers.srt"
LANG = "ara"

META = {
    "imdb_id": "tt0773262",
    "season": 8,
    "episode": 5,
    "target_filename": TARGET,
}
SEASON_EP = "s8e5"


def _ref() -> str:
    """The search-time subtitle id, exactly as the listing path derives it."""
    rel_key = f"subsource:{RELEASE}:{URL}:s8:e5"
    return hashlib.sha256(rel_key.encode("utf-8")).hexdigest()[:16]


def _fingerprint() -> str:
    return SyncCache.video_fingerprint_from_meta(dict(META))


def _release() -> SubtitleRelease:
    return SubtitleRelease(
        release_name=RELEASE, download_url=URL, provider="subsource", lang=LANG
    )


def _cache():
    from app.main import _sync_cache

    return _sync_cache


@pytest.fixture(autouse=True)
def _isolate_sync_cache():
    """The SyncCache is a process-wide singleton.

    Without this, a verdict seeded by one test is inherited by the next, and a
    test written to prove a *miss* silently becomes a hit. Each test here must
    start from a cache it fully controls, or it proves nothing.
    """
    from app.main import _sync_cache

    stores = ("_local", "_local_meta", "_local_fail", "_local_verdict")
    for name in stores:
        getattr(_sync_cache, name).clear()
    yield
    for name in stores:
        getattr(_sync_cache, name).clear()


async def _seed_verified(state: SyncState, *, language: str = LANG, engine: int | None = None):
    """Write a verified verdict plus its synced artifact, the way serve does."""
    cache = _cache()
    fingerprint = _fingerprint()
    payload = b"1\n00:00:10,160 --> 00:00:12,000\ncached synced output\n"
    verdict = {
        "sync_state": state.value,
        "verification": "verified",
        "language": language,
        "sync_confidence": 0.9,
        "reasons": ["seeded for test"],
    }
    if engine is not None:
        verdict["engine_version"] = engine
    await cache.set_verdict(
        SyncCache.build_verdict_key(fingerprint, "content-hash", language),
        verdict,
        alias_key=SyncCache.build_verdict_alias_key(fingerprint, _ref()),
    )
    await cache.set(
        SyncCache.build_key("tt0773262", SEASON_EP, fingerprint, _ref(), "edition", "chash"),
        payload,
    )
    return payload


FRESH = b"1\n00:00:20,000 --> 00:00:22,000\nfresh download\n"


def _install_provider(downloads: list[str], payload: bytes = FRESH):
    class _Provider:
        name = "subsource"

        async def search_subtitles(self, **kwargs):
            return [_release()]

        async def download_archive(self, download_ref, api_key=None):
            downloads.append(download_ref)
            return payload

    return _Provider()


async def _run():
    from app.main import _fallback_download_subsource

    outcome: dict = {}
    meta: dict = {"release_name": REQUESTED, "lang": LANG}
    result = await _fallback_download_subsource(
        imdb_id="tt0773262",
        media_type="series",
        season=8,
        episode=5,
        subsource_key="k",
        target_filename=TARGET,
        lang=LANG,
        client=object(),
        requested_release_name=REQUESTED,
        requested_uploader=None,
        requested_hearing_impaired=None,
        meta=meta,
        outcome=outcome,
    )
    return result, outcome, meta


async def _run_with(provider):
    with patch("app.main.SubsourceProvider", new=lambda client: provider):
        return await _run()


# --- A/B. verified positives are reused without downloading -----------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state", [SyncState.VERIFIED_SYNCED, SyncState.VERIFIED_RESYNCED]
)
async def test_a_b_verified_positive_is_reused(state):
    expected = await _seed_verified(state)
    downloads: list[str] = []
    result, outcome, meta = await _run_with(_install_provider(downloads))
    assert result == expected
    assert downloads == [], "an exact verified result must not be re-downloaded"
    assert outcome["category"] == "SYNC_CACHE_REUSE"
    assert outcome["downloads_skipped"] is True
    assert meta["fallback_sync_cache_reuse"] is True


# --- C. miss follows the existing path --------------------------------------


@pytest.mark.asyncio
async def test_c_miss_downloads_and_returns_the_new_payload():
    downloads: list[str] = []
    result, outcome, _meta = await _run_with(_install_provider(downloads, FRESH))
    assert result == FRESH
    assert downloads == [URL]
    assert outcome["category"] != "SYNC_CACHE_REUSE"


# --- D/E/F. identity must match exactly -------------------------------------


@pytest.mark.asyncio
async def test_d_different_target_video_is_a_miss():
    """A verdict proved for another video must not be reused here.

    Seeded against the *other* target's fingerprint, which is the direction that
    actually matters: a verdict exists, it is verified, and it is for a real
    video -- just not this one.
    """
    cache = _cache()
    other_meta = dict(META, target_filename="Dexter.S08E05.720p.BluRay.x264-OTHER.mkv")
    other_fp = SyncCache.video_fingerprint_from_meta(other_meta)
    assert other_fp != _fingerprint()
    await cache.set_verdict(
        SyncCache.build_verdict_key(other_fp, "h", LANG),
        {
            "sync_state": SyncState.VERIFIED_SYNCED.value,
            "verification": "verified",
            "language": LANG,
        },
        alias_key=SyncCache.build_verdict_alias_key(other_fp, _ref()),
    )
    await cache.set(
        SyncCache.build_key("tt0773262", SEASON_EP, other_fp, _ref(), "edition", "chash"),
        b"other target output\n",
    )
    downloads: list[str] = []
    result, outcome, _meta = await _run_with(_install_provider(downloads, FRESH))
    assert result == FRESH
    assert downloads == [URL]
    assert outcome["category"] != "SYNC_CACHE_REUSE"


@pytest.mark.asyncio
async def test_e_different_subtitle_is_a_miss():
    cache = _cache()
    fingerprint = _fingerprint()
    await cache.set_verdict(
        SyncCache.build_verdict_key(fingerprint, "h", LANG),
        {
            "sync_state": SyncState.VERIFIED_SYNCED.value,
            "verification": "verified",
            "language": LANG,
        },
        alias_key=SyncCache.build_verdict_alias_key(fingerprint, "0000000000000000"),
    )
    downloads: list[str] = []
    result, outcome, _meta = await _run_with(_install_provider(downloads, FRESH))
    assert result == FRESH
    assert downloads == [URL]
    assert outcome["category"] != "SYNC_CACHE_REUSE"


@pytest.mark.asyncio
async def test_f_different_language_is_a_miss():
    await _seed_verified(SyncState.VERIFIED_SYNCED, language="eng")
    downloads: list[str] = []
    _result, outcome, _meta = await _run_with(_install_provider(downloads, FRESH))
    assert downloads == [URL]
    assert outcome["category"] != "SYNC_CACHE_REUSE"


# --- G. engine version -----------------------------------------------------


@pytest.mark.asyncio
async def test_g_stale_engine_version_is_a_miss():
    """An entry persisted by an older engine must not be reused.

    ``set_verdict`` always stamps the *current* engine version, so the read-time
    check in ``get_verdict`` can only be reached by an entry written by an older
    build. Writing the store directly reproduces exactly that: a verified,
    correct-looking verdict whose engine version no longer matches.
    """
    import json

    from app.services import sync_cache as sc

    current = sc.SYNC_VERDICT_ENGINE_VERSION
    stale = current + 1
    cache = _cache()
    fingerprint = _fingerprint()
    alias = SyncCache.build_verdict_alias_key(fingerprint, _ref())
    cache._local_verdict[alias] = json.dumps(
        {
            "sync_state": SyncState.VERIFIED_SYNCED.value,
            "verification": "verified",
            "language": LANG,
            "engine_version": stale,
        }
    )
    # Even though a matching artifact exists, the stale verdict must not
    # authorize reuse.
    await cache.set(
        SyncCache.build_key("tt0773262", SEASON_EP, fingerprint, _ref(), "edition", "chash"),
        b"stale engine output\n",
    )
    downloads: list[str] = []
    result, outcome, _meta = await _run_with(_install_provider(downloads, FRESH))
    assert result == FRESH
    assert downloads == [URL]
    assert outcome["category"] != "SYNC_CACHE_REUSE"


# --- H. a cached negative is never served as output -------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state", [SyncState.REJECTED, SyncState.UNVERIFIED, SyncState.PROBABLE_SYNC]
)
async def test_h_cached_non_positive_is_not_served(state):
    cache = _cache()
    fingerprint = _fingerprint()
    await cache.set_verdict(
        SyncCache.build_verdict_key(fingerprint, "h", LANG),
        {
            "sync_state": state.value,
            "verification": "verified",
            "language": LANG,
        },
        alias_key=SyncCache.build_verdict_alias_key(fingerprint, _ref()),
    )
    # An artifact exists under the same identity. It must still not be served:
    # the negative verdict is the authoritative signal.
    await cache.set(
        SyncCache.build_key("tt0773262", SEASON_EP, fingerprint, _ref(), "edition", "chash"),
        b"must not be served\n",
    )
    downloads: list[str] = []
    result, outcome, _meta = await _run_with(_install_provider(downloads, FRESH))
    assert result == FRESH
    assert outcome["category"] != "SYNC_CACHE_REUSE"


# --- I. the two caches stay separate ----------------------------------------


@pytest.mark.asyncio
async def test_i_search_cache_hit_does_not_imply_sync_cache_hit():
    """A cached candidate list is not a verified verdict."""
    downloads: list[str] = []
    provider = _install_provider(downloads, FRESH)
    first, _o1, _m1 = await _run_with(provider)
    assert downloads == [URL]

    # Second run: provider search is served from the search cache, but there is
    # still no verified verdict, so the subtitle must be downloaded again.
    downloads.clear()
    second, outcome, _m2 = await _run_with(provider)
    assert second == FRESH
    assert downloads == [URL], "search-cache hit must not be mistaken for a sync hit"
    assert outcome["category"] != "SYNC_CACHE_REUSE"


# --- J. eligibility is unchanged; the lookup simply precedes the download ---


@pytest.mark.asyncio
async def test_j_lookup_happens_for_a_compatible_candidate_with_no_exact_release():
    downloads: list[str] = []
    _result, outcome, _meta = await _run_with(_install_provider(downloads, FRESH))
    assert outcome["exact_candidates"] == 0
    assert outcome["compatible_candidates"] == 1
    assert downloads == [URL]


# --- determinism and payload scoping ---------------------------------------


@pytest.mark.asyncio
async def test_repeated_lookups_are_deterministic():
    await _seed_verified(SyncState.VERIFIED_RESYNCED)
    results = []
    for _ in range(3):
        result, _outcome, _meta = await _run_with(_install_provider([], b"unused"))
        results.append(result)
    assert len(set(results)) == 1
    assert results[0] == b"1\n00:00:10,160 --> 00:00:12,000\ncached synced output\n"


@pytest.mark.asyncio
async def test_payload_lookup_is_fingerprint_scoped():
    """The strict lookup must not return another target's artifact."""
    cache = _cache()
    other_fp = "deadbeefdeadbeefdeadbeef"
    await cache.set(
        SyncCache.build_key("tt0773262", SEASON_EP, other_fp, _ref(), "edition", "chash"),
        b"wrong target\n",
    )
    found = await cache.find_synced_for_target("tt0773262", SEASON_EP, other_fp, _ref())
    assert found == b"wrong target\n"
    assert await cache.find_synced_for_target("tt0773262", SEASON_EP, "0" * 24, _ref()) is None


@pytest.mark.asyncio
async def test_verified_verdict_without_artifact_still_downloads():
    """A verdict is not a payload. Content-addressing means no bytes."""
    cache = _cache()
    fingerprint = _fingerprint()
    await cache.set_verdict(
        SyncCache.build_verdict_key(fingerprint, "h", LANG),
        {
            "sync_state": SyncState.VERIFIED_SYNCED.value,
            "verification": "verified",
            "language": LANG,
        },
        alias_key=SyncCache.build_verdict_alias_key(fingerprint, _ref()),
    )
    downloads: list[str] = []
    result, _outcome, _meta = await _run_with(_install_provider(downloads, FRESH))
    assert result == FRESH
    assert downloads == [URL]
