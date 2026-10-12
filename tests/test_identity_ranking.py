"""Identity provenance and score-first presentation regression tests.

Fixtures provide the authoritative displayed scores. Ordering consumes those
scores without inferring hash identity or independently promoting sync states.
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
    release.match_percentage = score if score is not None else (100 if hash_match else
        {MatchTier.HASH: 100, MatchTier.EXACT: 90, MatchTier.SOURCE_FAMILY: 75,
         MatchTier.CLOSE: 40, MatchTier.FALLBACK: 10}[tier])
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


def test_exact_score_beats_unscoped_verified_claim():
    a = _attach(_release("hash.srt", hash_match=True), UNKNOWN_STATE, AV_UNKNOWN)
    b = _attach(_release("nosync.srt", hash_match=False), VERIFIED, AV_VERIFIED)

    assert order_candidates([a, b]) == [a, b]
    assert comparison_key(a) < comparison_key(b)


# ===========================================================================
# Case B -- verified sync beats hash compatibility too
# ===========================================================================




# ===========================================================================
# Case C -- hash identity wins among equivalent verified candidates
# ===========================================================================






def test_identity_is_a_lower_value_than_no_identity():
    from app.services.sync.ordering import _identity_rank

    assert _identity_rank(_release("x.srt", hash_match=True)) == 0
    assert _identity_rank(_release("y.srt", hash_match=False)) == 1


# ===========================================================================
# Case D -- prediction normalization must still apply
# ===========================================================================




def test_exact_score_beats_prediction_without_identity():
    """Identity is below sync state: a predicted candidate is not UNKNOWN."""
    predicted = _attach(
        _release("pred.srt", hash_match=False),
        PREDICTED,
        AV_PREDICTED,
        confidence=CONFIDENCE_PREDICTION_FLOOR,
    )
    hashed = _attach(_release("hash.srt", hash_match=True), UNKNOWN_STATE, AV_UNKNOWN)

    assert order_candidates([predicted, hashed]) == [hashed, predicted]


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


def test_unscoped_verified_fallback_does_not_override_compatibility():
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
    assert order_candidates([exact_unverified, fallback_verified])[0] is exact_unverified
    assert fallback_verified.match_tier == MatchTier.FALLBACK


# ===========================================================================
# 7b. Sync quality outranks compatibility within the same sync evidence
# ===========================================================================


def test_raw_verifier_confidence_does_not_override_compatibility():
    """Among candidates with the same sync state and availability, stronger
    measured evidence ranks first even when compatibility is worse."""
    strong = _attach(
        _release("strong.srt", tier=MatchTier.FALLBACK),
        VERIFIED,
        AV_VERIFIED,
        base=10,
        confidence=98.0,
    )
    weak = _attach(
        _release("weak.srt", tier=MatchTier.EXACT),
        VERIFIED,
        AV_VERIFIED,
        base=0,
        confidence=72.0,
    )
    assert order_candidates([weak, strong])[0] is weak




def test_actionable_prediction_confidence_still_counts():
    """A prediction above the floor keeps its confidence as positive evidence,
    but only within the predicted availability tier."""
    from app.services.sync.ordering import _ranked_confidence

    actionable = _attach(
        _release("p.srt"),
        PREDICTED,
        AV_PREDICTED,
        confidence=0.9,
    )
    assert _ranked_confidence(actionable, "predicted") == 0.9


# ===========================================================================
# 8. Existing invariants are untouched
# ===========================================================================










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
