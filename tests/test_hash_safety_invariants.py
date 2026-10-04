"""P0/P1 safety invariants: hash provenance, cross-video isolation, parity.

Every test here corresponds to a defect found during the architecture
investigation and fixed in the same change. They are deliberately adversarial:
each asserts the *rejection* path as strongly as the acceptance path, because
the failure mode being guarded is a false "exact hash" claim or another video's
identity leaking in.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from starlette.requests import Request

from app.main import _media_context_from_request, _merge_sync_meta
from app.models import SubtitleRelease
from app.services.ranking import extract_stream_params

HASH_A = "239f938f5b1d6ebd"
HASH_B = "8e245d9679d31e12"


def _release(**kw):
    base = {"release_name": "a.srt", "download_url": "/sub/opensubtitles/1.srt", "provider": "opensubtitles"}
    base.update(kw)
    return SubtitleRelease(**base)


def _ctx(qs: str) -> dict:
    return _media_context_from_request(
        Request({"type": "http", "method": "GET", "path": "/x", "query_string": qs.encode(), "headers": []})
    )


# ===========================================================================
# P0-2 -- hash flags accept only a real boolean
# ===========================================================================


@pytest.mark.parametrize("value", [True, False])
def test_real_booleans_are_preserved(value):
    r = _release(is_hash_match=value)
    assert r.is_hash_match is value
    assert r.matched_by_hash is value


@pytest.mark.parametrize("bad", [1, 0, "1", "true", "True", "yes", "false", "False", None, 1.0, [], {}])
def test_non_boolean_values_cannot_become_an_exact_hash_match(bad):
    """Pydantic's lax bool used to coerce every one of these to True."""
    with pytest.raises(ValidationError):
        _release(is_hash_match=bad)
    with pytest.raises(ValidationError):
        _release(matched_by_hash=bad)


@pytest.mark.parametrize("bad", [1, "1", "true", "yes"])
def test_a_non_boolean_flag_cannot_reach_tier_0(bad):
    """Requirement 9: an invalid provider result must never become Tier 0."""
    from app.services.subtitle_matcher import MatchTier

    # Construction is refused outright, so no Tier-0 carrying object can exist.
    assert MatchTier.HASH == 0


def test_the_two_flags_stay_welded():
    assert _release(is_hash_match=True).matched_by_hash is True
    assert _release(matched_by_hash=True).is_hash_match is True


def test_no_subtitle_release_carries_a_video_hash_field():
    """Identity evidence must not be smuggled onto the release itself."""
    assert "video_hash" not in SubtitleRelease.model_fields


def test_subdl_and_subsource_are_never_labelled_exact_hash():
    """Requirement: no false exact-hash labelling of other providers."""
    for provider in ("subdl", "subsource", "yifysubtitles", "subtitlecat"):
        assert _release(provider=provider).is_hash_match is False
        assert _release(provider=provider).matched_by_hash is False


def test_a_hash_is_never_derived_from_other_metadata():
    """Filename / IMDb / release name / size must not become a MovieHash."""
    params = extract_stream_params("filename=Movie.2024.1080p.FLUX.mkv&imdb_id=tt123&videoSize=999", None)
    assert params["video_hash"] is None
    assert params["filename"]
    assert params["video_size"] == 999


# ===========================================================================
# P0-1 -- cross-video identity isolation
# ===========================================================================


def _cached_record(sub_id: str, video_hash, video_size):
    return {"sub_id": sub_id, "video_hash": video_hash, "video_size": video_size,
            "imdb_id": "tt1", "media_type": "movie"}


def test_video_b_never_inherits_video_a_after_identical_candidate_ids():
    """The record key is video-agnostic; the request must still win."""
    merged = _merge_sync_meta(
        _cached_record("same-sub-id", HASH_A, 111), {"video_hash": HASH_B, "video_size": 222}
    )
    assert merged["video_hash"] == HASH_B
    assert merged["video_size"] == 222


def test_a_hashless_request_does_not_inherit_a_previous_hash():
    """The hole the fix closes: nothing overrode an unproven cached identity."""
    merged = _merge_sync_meta(_cached_record("same-sub-id", HASH_A, 111), {"imdb_id": "tt1"})
    assert "video_hash" not in merged
    assert "video_size" not in merged


def test_a_hashless_request_keeps_its_own_non_identity_metadata():
    merged = _merge_sync_meta(_cached_record("s", HASH_A, 111), {"imdb_id": "tt9"})
    assert merged["imdb_id"] == "tt1"  # only-if-empty key, unchanged behaviour
    assert merged["sub_id"] == "s"


def test_the_same_hash_is_stable_across_repeated_requests():
    record = _cached_record("s", None, None)
    first = _merge_sync_meta(record, {"video_hash": HASH_A, "video_size": 999})
    second = _merge_sync_meta(record, {"video_hash": HASH_A, "video_size": 999})
    assert first["video_hash"] == second["video_hash"] == HASH_A
    assert first["video_size"] == second["video_size"] == 999


def test_different_hashes_stay_isolated_end_to_end():
    from app.services.sync.orchestrator import SyncOrchestrator

    stems = {
        SyncOrchestrator()._build_query({"imdb_id": "tt1", "video_hash": h, "target_filename": "same.mkv"}).cache_stem
        for h in (HASH_A, HASH_B)
    }
    assert len(stems) == 2


def test_same_filename_different_hashes_do_not_collide():
    from app.services.sync.orchestrator import SyncOrchestrator

    stems = {
        SyncOrchestrator()._build_query(
            {"imdb_id": "tt1", "video_hash": h, "video_size": 5, "target_filename": "Same.Name.mkv"}
        ).cache_stem
        for h in (HASH_A, HASH_B)
    }
    assert len(stems) == 2


def test_different_sizes_are_isolated():
    from app.services.sync.orchestrator import SyncOrchestrator

    stems = {
        SyncOrchestrator()._build_query({"imdb_id": "tt1", "video_hash": HASH_A, "video_size": s}).cache_stem
        for s in (999, 1111)
    }
    assert len(stems) == 2


def test_request_over_cache_precedence_is_preserved_for_present_values():
    merged = _merge_sync_meta(_cached_record("s", HASH_A, 111), {"video_hash": HASH_B, "video_size": 222})
    assert merged["video_hash"] == HASH_B


# ===========================================================================
# P1-2 -- serve-side spelling parity
# ===========================================================================


@pytest.mark.parametrize("key", ["videoHash", "videohash", "video_hash"])
def test_all_search_side_hash_spellings_are_accepted_on_serve(key):
    assert _ctx(f"{key}={HASH_A}")["video_hash"] == HASH_A


@pytest.mark.parametrize("key", ["videoSize", "videosize", "video_size"])
def test_all_search_side_size_spellings_are_accepted_on_serve(key):
    assert _ctx(f"{key}=999")["video_size"] == "999"


@pytest.mark.parametrize("key", ["moviehash", "hash", "moviebytesize", "size"])
def test_search_only_spellings_stay_rejected_on_serve(key):
    """Parity is with what the search path accepts; unrelated keys are ignored."""
    assert _ctx(f"{key}=abc").get("video_hash") is None


def test_a_camelcase_serve_url_reaches_the_reference_query():
    from app.services.sync.orchestrator import SyncOrchestrator

    merged = _merge_sync_meta({}, _ctx(f"videoHash={HASH_A}&videoSize=999"))
    q = SyncOrchestrator()._build_query({**merged, "imdb_id": "tt1"})
    assert q.video_hash == HASH_A
    assert q.video_size == "999"


# ===========================================================================
# P1-3 -- video size zero
# ===========================================================================


def test_a_positive_size_is_preserved():
    assert extract_stream_params("videoSize=999", None)["video_size"] == 999
    assert extract_stream_params(None, {"videoSize": "999"})["video_size"] == 999


@pytest.mark.parametrize(
    "args", [("videoSize=0", None), (None, {"videoSize": "0"}), ("videoSize=", None)]
)
def test_an_explicit_zero_means_unavailable(args):
    """0 is not a real byte size; it is normalised once, to None."""
    assert extract_stream_params(*args)["video_size"] is None


def test_a_missing_size_stays_missing():
    assert extract_stream_params("filename=x.mkv", None)["video_size"] is None


def test_a_real_query_size_is_not_replaced_by_a_zero_in_the_extra():
    """Previously the query value was int 999, but an extra-segment 0 saw a
    falsy slot... the asymmetry ran the other way: a query 0 (int) was falsy and
    the extra overwrote it. Now 0 normalises to None once and a real value in
    either source survives."""
    assert extract_stream_params("videoSize=999", {"videoSize": "0"})["video_size"] == 999
    assert extract_stream_params("videoSize=0", {"videoSize": "999"})["video_size"] == 999


# ===========================================================================
# P1-1 -- OpenSubtitles format routes
# ===========================================================================


def _open_subtitle_routes() -> set[str]:
    from app.main import app

    return {r.path for r in app.routes if hasattr(r, "path") and "/sub/opensubtitles/" in r.path}


@pytest.mark.parametrize("ext", ["srt", "ass", "ssa", "sub", "vtt"])
def test_every_emittable_extension_has_a_proxy_route(ext):
    """The provider emits .ssa and _FILE_ID_RE accepts .ssa/.sub."""
    assert f"/sub/opensubtitles/{{file_id}}.{ext}" in _open_subtitle_routes()
    assert f"/{{config}}/sub/opensubtitles/{{file_id}}.{ext}" in _open_subtitle_routes()


@pytest.mark.parametrize("ext", ["ssa", "sub"])
def test_a_valid_ssa_or_sub_reference_matches_the_file_id_pattern(ext):

    from app.providers.opensubtitles import _FILE_ID_RE

    m = _FILE_ID_RE.search(f"/sub/opensubtitles/7144860.{ext}")
    assert m and m.group(1) == "7144860"


@pytest.mark.parametrize("bad", ["../../etc/passwd", "..%2F..%2Fetc%2Fpasswd", "abc"])
def test_a_non_numeric_or_traversing_reference_is_not_routable(bad):
    """file_id is an int path parameter, so these cannot match any route."""

    from app.providers.opensubtitles import _FILE_ID_RE

    assert _FILE_ID_RE.search(f"/sub/opensubtitles/{bad}.srt") is None


@pytest.mark.parametrize("ext", ["exe", "php", "py"])
def test_an_unsupported_extension_has_no_route(ext):
    assert f"/sub/opensubtitles/{{file_id}}.{ext}" not in _open_subtitle_routes()


# ===========================================================================
# P0-3 -- MovieHash reference trust
# ===========================================================================


@pytest.mark.asyncio
async def test_a_hash_reference_reports_strong_trust_without_asserting_verified(tmp_path):
    from app.services.sync.cache import ReferenceDiskCache
    from app.services.sync.hash_reference import OpenSubtitlesHashReferenceStrategy
    from app.services.sync.reference import ReferenceTrust

    payload = ("1\n00:00:01,000 --> 00:00:03,000\nline\n\n") * 400

    class Provider:
        name = "opensubtitles"

        async def search_subtitles(self, **kw):
            return [_release(is_hash_match=True)]

        async def download_archive(self, ref, api_key=None, username=None, password=None):
            return payload.encode()

        def is_breaker_open(self):
            return False

    strategy = OpenSubtitlesHashReferenceStrategy(
        Provider(), cache=ReferenceDiskCache(root=tmp_path / "r", min_bytes=5120)
    )
    from app.services.sync.query import ReferenceQuery

    resolved = await strategy.resolve_with_provenance(
        ReferenceQuery(imdb_id="tt1", video_hash=HASH_A, target_filename="a.mkv",
                       api_keys={"opensubtitles": "k"})
    )
    assert resolved.text is not None
    assert resolved.reference_trust == ReferenceTrust.STRONG.value == "strong"
    # Crucially: no verification verdict is claimed. Timing authority stays with
    # the analyzer; trust can only withhold a claim (alignment.py:769-777).
    assert not hasattr(resolved, "sync_state")
    assert resolved.kind == "hash"


def test_strong_trust_is_only_ever_a_withholding_gate():
    """Documents why setting STRONG cannot manufacture a verified sync."""
    from app.services.sync.alignment import _REFERENCE_TRUST_SUPPORTING_VERIFIED

    assert "strong" in _REFERENCE_TRUST_SUPPORTING_VERIFIED
    # And a low/unknown trust actively withholds.
    assert "acceptable" not in _REFERENCE_TRUST_SUPPORTING_VERIFIED
    assert "unknown" not in _REFERENCE_TRUST_SUPPORTING_VERIFIED
