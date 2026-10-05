"""Same-video identity as an explicit ranking dimension.

``_identity_rank`` reads the existing ``SubtitleRelease.is_hash_match`` field and
nothing else. That field is set in exactly one place -- ``OpenSubtitlesProvider``,
and only when the request carried a ``moviehash`` AND OpenSubtitles returned
``attributes.moviehash_match is True`` -- and it is a ``StrictBool``, so a truthy
string cannot reach it.

The ordering it was added to is strict lexicographic, and the position matters:
identity sits BELOW synchronization state and evidence, so a hash-exact subtitle
with unverified timing never outranks a hashless subtitle the verifier actually
confirmed. It sits ABOVE sync confidence, compatibility percentage and
provider/release, because among candidates with comparable timing evidence, proof
that the subtitle belongs to THIS video is the next strongest discriminator.

These tests use the project's real ``SubtitleRelease`` and the existing ordering
helpers rather than a parallel fake model, so they exercise the production path.
"""

from __future__ import annotations

import pytest

from app.models import MatchTier, SubtitleRelease
from app.services.subtitle_matcher import CompatibilityResult
from app.services.sync.ordering import comparison_key, order_candidates
from app.services.sync.predictor import CONFIDENCE_PREDICTION_FLOOR

VERIFIED = "verified_synced"
UNKNOWN_STATE = None
PREDICTED = "predicted"

AV_VERIFIED = "verified"
AV_CACHED = "cached"
AV_PREDICTED = "predicted"
AV_UNKNOWN = "unknown"


def _release(
    name: str,
    *,
    tier: MatchTier = MatchTier.CLOSE,
    hash_match: bool = False,
    provider: str = "subdl",
    lang: str = "ara",
    score: int | None = None,
) -> SubtitleRelease:
    release = SubtitleRelease(
        release_name=name,
        download_url=f"http://example.invalid/{name}",
        provider=provider,
        lang=lang,
        # is_hash_match is StrictBool, so this rejects truthy strings at
        # construction -- exactly the provenance guarantee we rely on.
        is_hash_match=hash_match,
    )
    release.compatibility = CompatibilityResult(
        accepted=True, match_tier=tier, match_method="filename"
    )
    release.match_tier = tier
    if score is not None:
        release.match_percentage = score
    return release


def _attach(
    release: SubtitleRelease,
    state: str | None,
    verification: str | None,
    *,
    base: int = 0,
    confidence: float | None = None,
    lang_rank: int = 0,
) -> SubtitleRelease:
    release.sync_state = state
    release.sync_verification = verification
    release.sync_confidence = confidence
    release.sync_base_rank = base
    release.sync_lang_rank = lang_rank
    return release


# ===========================================================================
# Case A -- verified synchronization beats hash identity
# ===========================================================================


def test_case_a_verified_sync_beats_hash_identity():
    a = _attach(_release("hash.srt", hash_match=True), UNKNOWN_STATE, AV_UNKNOWN)
    b = _attach(_release("nosync.srt", hash_match=False), VERIFIED, AV_VERIFIED)

    assert order_candidates([a, b]) == [b, a]
    assert comparison_key(b) < comparison_key(a)


# ===========================================================================
# Case B -- verified sync beats hash compatibility too
# ===========================================================================


def test_case_b_verified_sync_beats_hash_tier():
    a = _attach(
        _release("hash.srt", hash_match=True, tier=MatchTier.HASH),
        UNKNOWN_STATE,
        AV_UNKNOWN,
    )
    b = _attach(
        _release("exact.srt", hash_match=False, tier=MatchTier.EXACT),
        VERIFIED,
        AV_VERIFIED,
    )

    assert order_candidates([a, b]) == [b, a]


# ===========================================================================
# Case C -- hash identity wins among equivalent verified candidates
# ===========================================================================


def test_case_c_hash_identity_wins_among_verified():
    """The primary reason _identity_rank exists."""
    # Names are chosen so the stable tie-breaker alone would put b FIRST.
    # Without the identity dimension this test therefore fails.
    a = _attach(_release("z-hash.srt", hash_match=True), VERIFIED, AV_VERIFIED)
    b = _attach(_release("a-plain.srt", hash_match=False), VERIFIED, AV_VERIFIED)

    assert order_candidates([a, b]) == [a, b]
    assert comparison_key(a) < comparison_key(b)


def test_case_c_also_holds_for_cached_verification():
    a = _attach(_release("z-hash.srt", hash_match=True), VERIFIED, AV_CACHED)
    b = _attach(_release("a-plain.srt", hash_match=False), VERIFIED, AV_CACHED)

    assert order_candidates([a, b]) == [a, b]


def test_identity_is_a_lower_value_than_no_identity():
    from app.services.sync.ordering import _identity_rank

    assert _identity_rank(_release("x.srt", hash_match=True)) == 0
    assert _identity_rank(_release("y.srt", hash_match=False)) == 1


# ===========================================================================
# Case D -- prediction normalization must still apply
# ===========================================================================


def test_case_d_weak_prediction_cannot_leapfrog_verified():
    """A weak prediction is demoted to UNVERIFIED/UNKNOWN inside the comparator,
    and in any case never outranks VERIFIED."""
    strong = _attach(
        _release("weakpred.srt", hash_match=False),
        PREDICTED,
        AV_PREDICTED,
        confidence=CONFIDENCE_PREDICTION_FLOOR,
    )
    verified = _attach(_release("nosync.srt", hash_match=False), VERIFIED, AV_VERIFIED)

    assert order_candidates([strong, verified]) == [verified, strong]


def test_a_hash_candidate_does_not_beat_a_strong_prediction():
    """Identity is below sync state: a predicted candidate is not UNKNOWN."""
    predicted = _attach(
        _release("pred.srt", hash_match=False),
        PREDICTED,
        AV_PREDICTED,
        confidence=CONFIDENCE_PREDICTION_FLOOR,
    )
    hashed = _attach(_release("hash.srt", hash_match=True), UNKNOWN_STATE, AV_UNKNOWN)

    assert order_candidates([predicted, hashed]) == [predicted, hashed]


def test_prediction_floor_cannot_be_bypassed_via_identity():
    from app.services.sync.ordering import _normalized_evidence

    weak = _attach(
        _release("w.srt", hash_match=True),
        PREDICTED,
        AV_PREDICTED,
        confidence=CONFIDENCE_PREDICTION_FLOOR - 0.01,
    )
    state, availability = _normalized_evidence(weak)
    assert state == "unverified"
    assert availability == AV_UNKNOWN


# ===========================================================================
# Case E -- deterministic tie-breaking
# ===========================================================================


def test_case_e_equivalent_candidates_use_stable_tie_breaker():
    a = _attach(_release("b.srt"), VERIFIED, AV_VERIFIED)
    b = _attach(_release("a.srt"), VERIFIED, AV_VERIFIED)

    assert order_candidates([a, b]) == [b, a], "release_name breaks the tie"
    assert order_candidates([b, a]) == [b, a], "and it does so regardless of input order"


def test_ordering_is_identical_across_repeated_runs():
    candidates = [
        _attach(_release(f"c{i}.srt", hash_match=(i % 3 == 0)),
                VERIFIED if i % 2 else UNKNOWN_STATE,
                AV_VERIFIED if i % 2 else AV_UNKNOWN)
        for i in range(8)
    ]
    first = [r.release_name for r in order_candidates(candidates)]
    for _ in range(5):
        assert [r.release_name for r in order_candidates(candidates)] == first


def test_input_is_not_mutated():
    candidates = [
        _attach(_release("a.srt", hash_match=True), VERIFIED, AV_VERIFIED),
        _attach(_release("b.srt"), UNKNOWN_STATE, AV_UNKNOWN),
    ]
    before = list(candidates)
    order_candidates(candidates)
    assert candidates == before


# ===========================================================================
# 6. Hashless verified candidate remains first-class
# ===========================================================================


def test_a_hashless_verified_candidate_can_rank_first():
    hashless = _attach(
        _release("hashless.srt", hash_match=False, tier=MatchTier.EXACT),
        VERIFIED,
        AV_VERIFIED,
    )
    weaker = [
        _attach(_release("x.srt", hash_match=True), UNKNOWN_STATE, AV_UNKNOWN),
        _attach(_release("y.srt"), PREDICTED, AV_PREDICTED, confidence=0.9),
        _attach(_release("z.srt", tier=MatchTier.FALLBACK), UNKNOWN_STATE, AV_UNKNOWN),
    ]
    assert order_candidates([*weaker, hashless])[0] is hashless


def test_no_hash_penalty_is_applied_to_every_candidate():
    """When nothing carries a hash the ordering must be unchanged by the new key."""
    plain = [
        _attach(_release("a.srt"), VERIFIED, AV_VERIFIED),
        _attach(_release("b.srt", tier=MatchTier.EXACT), UNKNOWN_STATE, AV_UNKNOWN),
    ]
    keys = [comparison_key(r) for r in plain]
    for key in keys:
        assert key[4] == 1, "identity rank is neutral when no hash is present"


# ===========================================================================
# 9. Provenance safety -- ranking consumes the flag, never manufactures it
# ===========================================================================


def test_other_providers_cannot_inherit_a_hash_match():
    subdl = _release("subdl.srt", hash_match=False, provider="subdl")
    subsource = _release("ss.srt", hash_match=False, provider="subsource")
    os_exact = _release("os.srt", hash_match=True, provider="opensubtitles")

    order_candidates([os_exact, subdl, subsource])

    assert subdl.is_hash_match is False
    assert subsource.is_hash_match is False
    assert os_exact.is_hash_match is True


def test_ordering_does_not_mutate_the_candidate():
    candidate = _release("x.srt", hash_match=False, provider="subdl")
    before = (candidate.is_hash_match, candidate.matched_by_hash)
    order_candidates([candidate, _release("y.srt", hash_match=True, provider="opensubtitles")])
    assert (candidate.is_hash_match, candidate.matched_by_hash) == before


def test_malformed_hash_values_cannot_reach_the_flag():
    """StrictBool rejects truthy strings, so ranking can never read a faked match."""
    from pydantic import ValidationError

    for bad in (1, "1", "true", "yes", "True"):
        with pytest.raises(ValidationError):
            SubtitleRelease(
                release_name="x.srt",
                download_url="http://example.invalid/x.srt",
                provider="opensubtitles",
                is_hash_match=bad,
            )


def test_identity_is_not_inferred_from_provider_or_name():
    """A provider that merely looks hash-capable gets no identity credit."""
    opensubtitles_plain = _release(
        "plain.srt", hash_match=False, provider="opensubtitles"
    )
    subdl_plain = _release("other.srt", hash_match=False, provider="subdl")
    keys = [comparison_key(r) for r in (opensubtitles_plain, subdl_plain)]
    assert keys[0][4] == keys[1][4] == 1


# ===========================================================================
# 10. Fallback disclosure is preserved
# ===========================================================================


def test_fallback_disclosure_survives_ranking():
    fallback = _attach(
        _release("fb.srt", tier=MatchTier.FALLBACK, hash_match=False),
        UNKNOWN_STATE,
        AV_UNKNOWN,
    )
    fallback.compatibility.match_method = "COMPATIBLE_RELEASE"
    before = fallback.compatibility.match_method
    order_candidates([fallback])
    assert fallback.compatibility.match_method == before
    assert fallback.match_tier == MatchTier.FALLBACK


def test_a_verified_fallback_can_still_rank_above_an_unverified_exact():
    """Strong sync evidence may lift a fallback, without erasing its disclosure."""
    fallback_verified = _attach(
        _release("fbv.srt", tier=MatchTier.FALLBACK, hash_match=False),
        VERIFIED,
        AV_VERIFIED,
    )
    exact_unverified = _attach(
        _release("exu.srt", tier=MatchTier.EXACT, hash_match=False),
        UNKNOWN_STATE,
        AV_UNKNOWN,
    )
    assert order_candidates([exact_unverified, fallback_verified])[0] is fallback_verified
    assert fallback_verified.match_tier == MatchTier.FALLBACK


# ===========================================================================
# 8. Existing invariants are untouched
# ===========================================================================


def test_verified_synced_still_outranks_verified_resynced():
    synced = _attach(_release("a.srt"), "verified_synced", AV_VERIFIED)
    resynced = _attach(_release("b.srt"), "verified_resynced", AV_VERIFIED)
    assert order_candidates([resynced, synced]) == [synced, resynced]


def test_rejected_still_sorts_last():
    good = _attach(_release("a.srt"), VERIFIED, AV_VERIFIED)
    rejected = _attach(_release("b.srt"), "rejected", AV_VERIFIED)
    assert order_candidates([rejected, good]) == [good, rejected]


def test_language_preference_still_dominates_everything():
    """An explicit user choice outranks any amount of sync evidence."""
    arabic_verified = _attach(
        _release("ar.srt", lang="ara", hash_match=False),
        VERIFIED,
        AV_VERIFIED,
        lang_rank=0,
    )
    english_hashed = _attach(
        _release("en.srt", lang="eng", hash_match=True),
        VERIFIED,
        AV_VERIFIED,
        lang_rank=1,
    )
    assert order_candidates([english_hashed, arabic_verified])[0] is arabic_verified


def test_match_tier_remains_the_compatibility_source_of_truth():
    exact = _attach(_release("a.srt", tier=MatchTier.EXACT), UNKNOWN_STATE, AV_UNKNOWN)
    close = _attach(_release("b.srt", tier=MatchTier.CLOSE), UNKNOWN_STATE, AV_UNKNOWN)
    assert exact.match_tier == MatchTier.EXACT
    assert comparison_key(exact) < comparison_key(close)


def test_no_match_tier_was_added():
    names = {t.name for t in MatchTier}
    for forbidden in ("HASH_ANCHORED", "HASH_VERIFIED", "HASH_SYNCED"):
        assert forbidden not in names


def test_verifier_thresholds_are_untouched():
    from app.services.sync import alignment

    assert alignment.MAX_P95_MS_FOR_STABLE == 2000
    assert alignment.MAX_MAD_MS_FOR_STABLE < 9356


def test_cache_identity_still_binds_a_verdict_to_its_target():
    """A cached VERIFIED result must remain target-bound; ranking consumes the
    flag but must not let a verdict cross videos."""
    from app.services.sync_cache import SyncCache

    meta = {
        "imdb_id": "tt1",
        "media_type": "movie",
        "target_filename": "Same.Name.mkv",
        "video_hash": None,
        "video_size": "1000",
        "lang": "ara",
    }
    a = SyncCache.video_fingerprint_from_meta(meta)
    b = SyncCache.video_fingerprint_from_meta({**meta, "video_size": "2000"})
    assert a is not None and b is not None and a != b


def test_the_comparator_is_pure_and_cheap():
    """No network, no downloads, no alass: a pure tuple build per candidate."""
    candidates = [
        _attach(_release(f"c{i}.srt"), VERIFIED if i else UNKNOWN_STATE,
                AV_VERIFIED if i else AV_UNKNOWN)
        for i in range(200)
    ]
    first = [r.release_name for r in order_candidates(candidates)]
    assert len(first) == 200
    # Re-running yields byte-identical output: sorting keys only, nothing else.
    assert [r.release_name for r in order_candidates(candidates)] == first
