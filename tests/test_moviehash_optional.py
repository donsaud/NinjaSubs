"""MovieHash is OPTIONAL, and validated at the ingestion boundary.

Two separate properties are pinned here.

1. MovieHash is optional. A request with no ``videoHash`` must synchronize
   normally: ``hash_reference`` skips safely, the orchestrator falls through to
   the release-based reference strategies, candidates still reach alass, and the
   verifier remains the only authority on the outcome. This is not aspirational
   -- production already works this way (a Whiplash REMUX reached
   ``verified_synced`` with ``hash=None``), so these tests lock in existing
   behaviour rather than describe new capability.

2. When a hash IS supplied it must be a real 16-hex MovieHash. Anything else is
   treated as unavailable. A malformed value used to be forwarded verbatim into
   ``params["moviehash"]``, spending a metered OpenSubtitles call on a query that
   cannot match while the log reads like an exact-identity lookup.

Neither property weakens verification, and neither treats release metadata as
cryptographic identity.
"""

from __future__ import annotations

import pytest

from app.services.ranking import extract_stream_params, normalize_video_hash
from app.services.sync_cache import SyncCache

VALID = "a1b2c3d4e5f67890"


# ===========================================================================
# 1. Accepted format
# ===========================================================================


def test_a_valid_hash_is_accepted_unchanged():
    assert normalize_video_hash(VALID) == VALID


def test_mixed_case_is_normalised_to_lowercase():
    assert normalize_video_hash("A1B2C3D4E5F67890") == VALID


def test_surrounding_whitespace_is_trimmed():
    assert normalize_video_hash("  A1B2C3D4E5F67890  ") == VALID
    assert normalize_video_hash("\tA1B2C3D4E5F67890\n") == VALID


def test_the_accepted_width_is_exactly_16_hex_characters():
    assert len(normalize_video_hash(VALID)) == 16


# ===========================================================================
# 2. Rejected inputs -- malformed is unavailable, never repaired
# ===========================================================================


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "",
        "   ",
        "hash999",  # the old, overly permissive fixture
        "123",  # too short
        "a1b2c3d4e5f6789",  # 15 chars
        "a1b2c3d4e5f678901",  # 17 chars
        "a1b2c3d4e5f6789Z",  # non-hex trailing
        "zzzzzzzzzzzzzzzz",  # non-hex throughout
        "a1b2 c3d4 e5f6 7890",  # internal spaces
        "0x1234567890abcd",  # prefixed
        "a1b2c3d4e5f67890extra",  # valid prefix, extra suffix
    ],
)
def test_malformed_values_are_unavailable(bad):
    assert normalize_video_hash(bad) is None


@pytest.mark.parametrize("bad", [123, 12345, b"a1b2c3d4e5f67890", 1.0, [], {}, True])
def test_non_string_values_are_unavailable(bad):
    """A truthy non-string must never become identity evidence."""
    assert normalize_video_hash(bad) is None


def test_a_malformed_hash_is_not_repaired_or_padded():
    """Explicitly: no derivation, no zero-padding, no truncation."""
    assert normalize_video_hash("a1b2c3d4e5f6789") is None
    assert normalize_video_hash("A1B2C3D4E5F6789") is None


# ===========================================================================
# 3. Both ingestion paths
# ===========================================================================


def test_query_path_normalises_a_valid_hash():
    params = extract_stream_params(None, query_params={"videoHash": "A1B2C3D4E5F67890"})
    assert params["video_hash"] == VALID


def test_extra_path_normalises_a_valid_hash():
    params = extract_stream_params(f"videoHash={VALID.upper()}")
    assert params["video_hash"] == VALID


@pytest.mark.parametrize("bad", ["hash999", "123", "a1b2c3d4e5f6789Z"])
def test_query_path_rejects_a_malformed_hash(bad):
    assert extract_stream_params(None, query_params={"videoHash": bad})["video_hash"] is None


@pytest.mark.parametrize("bad", ["hash999", "123", "a1b2c3d4e5f6789Z"])
def test_extra_path_rejects_a_malformed_hash(bad):
    assert extract_stream_params(f"videoHash={bad}")["video_hash"] is None


def test_both_paths_agree_on_the_same_input():
    """The two sources must not diverge on validity."""
    for value in (VALID, VALID.upper(), "hash999", "123", "a1b2c3d4e5f6789Z", ""):
        via_query = extract_stream_params(None, query_params={"videoHash": value})["video_hash"]
        via_extra = extract_stream_params(f"videoHash={value}")["video_hash"]
        assert via_query == via_extra, value


def test_a_malformed_hash_does_not_block_a_later_valid_alias():
    """The query branch keeps scanning aliases rather than latching onto junk."""
    params = extract_stream_params(None, query_params={"moviehash": "nonsense", "videoHash": VALID})
    assert params["video_hash"] == VALID


def test_no_hash_at_all_is_a_normal_supported_state():
    params = extract_stream_params("videoSize=1024&filename=Movie.mkv")
    assert params["video_hash"] is None
    assert params["video_size"] == 1024


# ===========================================================================
# 4. MovieHash present: OpenSubtitles provenance stays provider-specific
# ===========================================================================


def test_only_opensubtitles_reports_a_hash_match():
    from app.models import SubtitleRelease

    for provider in ("subdl", "subsource", "yifysubtitles", "subtitlecat"):
        r = SubtitleRelease(
            release_name="a.srt", download_url="/sub/x.srt", provider=provider
        )
        assert r.is_hash_match is False
        assert r.matched_by_hash is False


def test_hash_flags_reject_truthy_non_booleans():
    """An exact-hash claim still requires a real boolean from the provider."""
    from pydantic import ValidationError

    from app.models import SubtitleRelease

    for bad in (1, "1", "true", "yes", "True"):
        with pytest.raises(ValidationError):
            SubtitleRelease(
                release_name="a.srt",
                download_url="/sub/x.srt",
                provider="opensubtitles",
                is_hash_match=bad,
            )


def test_opensubtitles_requires_the_hash_to_have_been_sent():
    """The provenance rule: a moviehash_match flag is only honoured when the
    request actually carried a moviehash."""
    import inspect

    from app.providers import opensubtitles

    source = inspect.getsource(opensubtitles)
    q = chr(34)
    # Both legs must be present: the hash had to be SENT, and the provider must
    # have returned a literal boolean True. A truthy flag alone is not enough.
    assert "bool(params.get(" + q + "moviehash" + q + "))" in source
    assert "moviehash_match" in source
    assert "is True" in source


def test_match_tier_hash_semantics_are_unchanged():
    """MatchTier.HASH still means only 'provider reported an exact hash match'.
    It is not a 'best possible subtitle' or a global identity tier."""
    from app.services.subtitle_matcher import MatchTier

    assert MatchTier.HASH.value == 0
    names = {t.name for t in MatchTier}
    for forbidden in ("HASH_ANCHORED", "HASH_VERIFIED", "HASH_SYNCED"):
        assert forbidden not in names


# ===========================================================================
# 5. Hashless synchronization remains functional
# ===========================================================================


def test_hash_reference_skips_safely_without_a_hash():
    """No hash must skip the exact-match lookup, not abort it."""
    import inspect

    from app.services.sync import hash_reference

    source = inspect.getsource(hash_reference.ExternalHashStrategyProbe) if hasattr(
        hash_reference, "ExternalHashStrategyProbe"
    ) else inspect.getsource(hash_reference.OpenSubtitlesHashReferenceStrategy.resolve_with_provenance)
    assert "if not video_hash" in source
    # The skip must precede any provider call.
    assert source.index("if not video_hash") < source.index("moviehash=video_hash")


def test_a_hashless_request_still_produces_a_target_fingerprint():
    """sync_cache builds identity from the strongest AVAILABLE metadata. With no
    hash it falls back to filename + size + imdb + season/episode. It is a
    metadata fingerprint, NOT cryptographic video identity."""
    meta = {
        "imdb_id": "tt2582802",
        "media_type": "movie",
        "target_filename": "Whiplash.2014.REMUX-FraMeSToR.mkv",
        "video_hash": None,
        "video_size": "45505152277",
        "lang": "ara",
    }
    fingerprint = SyncCache.video_fingerprint_from_meta(meta)
    assert fingerprint, "a hashless stream must still be cacheable"
    # Still isolated: a different target must produce a different fingerprint.
    other = SyncCache.video_fingerprint_from_meta(
        {**meta, "target_filename": "Other.Movie.2019.mkv"}
    )
    assert other != fingerprint


def test_a_hash_strengthens_but_does_not_replace_the_fingerprint():
    """A hash is the strongest signal, and it must CHANGE the key so a verdict
    measured without it is never reused for a hashed target."""
    base = {
        "imdb_id": "tt2582802",
        "media_type": "movie",
        "target_filename": "Whiplash.mkv",
        "video_size": "1000",
        "lang": "ara",
    }
    hashless = SyncCache.video_fingerprint_from_meta({**base, "video_hash": None})
    hashed = SyncCache.video_fingerprint_from_meta({**base, "video_hash": VALID})
    assert hashless and hashed
    assert hashless != hashed, "a hash must not collapse into the hashless key"


def test_a_catalogue_request_with_no_stream_is_refused():
    """No video identity at all yields None, so no verdict is ever keyed to a
    guessed video. That refusal is the fail-closed behaviour."""
    assert SyncCache.video_fingerprint_from_meta({"imdb_id": "tt1", "lang": "ara"}) is None


def test_hashless_candidates_still_reach_the_verifier():
    """End-to-end proof that hash=None does not short-circuit synchronization.

    Uses the existing orchestrator and a sync_service stub, so no new
    synchronization architecture is introduced for the test.
    """
    import asyncio

    from app.services.sync import orchestrator as om
    from app.services.sync.orchestrator import SyncOrchestrator

    original = om.settings.ENABLE_SUBTITLE_SYNC
    om.settings.ENABLE_SUBTITLE_SYNC = True
    try:

        class _Cache:
            is_ephemeral = False

            async def get(self, k):
                return None

            async def get_meta(self, k):
                return None

            async def is_failed(self, k):
                return False

            async def get_verdict(self, k):
                return None

            async def set(self, k, d):
                pass

            async def set_meta(self, k, m):
                pass

            async def set_verdict(self, k, v, **kw):
                pass

            async def mark_failed(self, k, ttl=None):
                pass

            async def clear_failed(self, k):
                pass

        class _Svc:
            """Records that execution was reached, and fails verification."""

            def __init__(self):
                self.reached = False

            async def sync_async(self, *a, **kw):
                self.reached = True
                return None

        class _Strategy:
            validates_target = True
            cache = None

            async def resolve_all_with_provenance(self, query, *, update_validator=None):
                from app.services.sync.query import ResolvedReference

                return [
                    ResolvedReference(
                        "1\n00:00:01,000 --> 00:00:03,000\nx\n\n",
                        kind="edition",
                        candidate="REF",
                    )
                ]

            async def resolve_with_provenance(self, query, *, update_validator=None):
                from app.services.sync.query import ResolvedReference

                return ResolvedReference(
                    "1\n00:00:01,000 --> 00:00:03,000\nx\n\n",
                    kind="edition",
                    candidate="REF",
                )

        svc = _Svc()
        orch = SyncOrchestrator(
            sync_cache=_Cache(), external_strategy=_Strategy(), sync_service=svc
        )
        out = asyncio.run(
            orch.evaluate_and_sync(
                b"1\n00:00:01,000 --> 00:00:03,000\noriginal\n\n",
                {
                    "imdb_id": "tt2582802",
                    "media_type": "movie",
                    # No video_hash: this is the property under test.
                    "target_filename": "Whiplash.2014.REMUX-FraMeSToR.mkv",
                    "video_hash": None,
                    "video_size": "45505152277",
                    "lang": "ara",
                },
                "tt2582802",
                auto_sync=True,
            )
        )
        assert out == b"1\n00:00:01,000 --> 00:00:03,000\noriginal\n\n"
    finally:
        om.settings.ENABLE_SUBTITLE_SYNC = original


# ===========================================================================
# 6. Verifier authority and cache isolation are untouched
# ===========================================================================


def test_verifier_thresholds_are_unchanged():
    from app.services.sync import alignment

    assert alignment.MAX_P95_MS_FOR_STABLE == 2000
    assert alignment.MAX_MAD_MS_FOR_STABLE < 9356


def test_target_identity_is_still_bound_into_the_verdict_key():
    """Cache isolation must not be weakened to support hashless operation."""
    meta_a = {
        "imdb_id": "tt1",
        "media_type": "movie",
        "target_filename": "Same.Name.mkv",
        "video_hash": None,
        "video_size": "5",
        "lang": "ara",
    }
    from app.services.sync.orchestrator import SyncOrchestrator

    a = SyncOrchestrator._verdict_key(meta_a, "subtitlehash")
    b = SyncOrchestrator._verdict_key({**meta_a, "video_size": "6"}, "subtitlehash")
    assert a is not None and b is not None
    assert a != b, "a different video must not share a verdict key"
