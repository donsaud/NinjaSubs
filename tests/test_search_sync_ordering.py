"""Search evidence remains scoped; presentation follows displayed compatibility.

Predictions cannot become verification, missing stream context stays unknown,
cache evidence remains fingerprint-bound, and search never runs synchronization.
Ordering consumes precomputed Compatibility Scores without independently
promoting states or overriding percentage order.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.models import MatchTier, SubtitleRelease
from app.services.subtitle_matcher import CompatibilityResult
from app.services.sync.alignment import SyncState, VerificationAvailability
from app.services.sync.ordering import comparison_key, order_candidates
from app.services.sync.predictor import (
    CONFIDENCE_HASH,
    CONFIDENCE_PREDICTION_FLOOR,
    SyncPrediction,
    SyncPredictor,
    prediction_is_below_floor,
)

PREDICTOR = SyncPredictor()

# A real video fingerprint: without this the predictor refuses to predict.
FINGERPRINT = {
    "imdb_id": "tt0773262",
    "media_type": "series",
    "season": 8,
    "episode": 5,
    "target_filename": "Dexter.s8e05.This.little.piggy.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv",
}
# A catalogue request: title/season/episode only, no stream context.
NO_FINGERPRINT = {
    "imdb_id": "tt0773262",
    "media_type": "series",
    "season": 8,
    "episode": 5,
    "title": "Dexter",
    "year": 2006,
}


def _release(
    name: str,
    *,
    tier: MatchTier = MatchTier.CLOSE,
    accepted: bool = True,
    lang: str = "ara",
    **compat_kwargs,
) -> SubtitleRelease:
    release = SubtitleRelease(
        release_name=name, download_url=f"http://{name}", provider="subdl", lang=lang
    )
    release.compatibility = CompatibilityResult(
        accepted=accepted,
        match_tier=tier,
        match_method="filename",
        **compat_kwargs,
    )
    release.match_tier = tier
    release.match_percentage = {MatchTier.HASH: 100, MatchTier.EXACT: 90,
        MatchTier.SOURCE_FAMILY: 75, MatchTier.CLOSE: 40, MatchTier.FALLBACK: 10}[tier]
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


# --------------------------------------------------------------------------- #
# Predictor rules
# --------------------------------------------------------------------------- #


def test_hash_match_is_the_strongest_prediction():
    prediction = PREDICTOR.predict(
        _release("A.srt", tier=MatchTier.HASH, is_hash_match=True), FINGERPRINT
    )
    assert prediction.availability is VerificationAvailability.PREDICTED
    assert prediction.predicted_state is SyncState.PROBABLE_SYNC
    assert prediction.confidence == CONFIDENCE_HASH
    assert prediction.is_actionable


def test_exact_release_group_predicts():
    prediction = PREDICTOR.predict(
        _release("A.srt", tier=MatchTier.EXACT, release_group_match=True), FINGERPRINT
    )
    assert prediction.is_actionable
    assert prediction.rule == "exact_release"


def test_source_family_with_matching_edition_predicts():
    prediction = PREDICTOR.predict(
        _release(
            "A.srt",
            tier=MatchTier.SOURCE_FAMILY,
            source_match=True,
            edition_match=True,
        ),
        FINGERPRINT,
    )
    assert prediction.is_actionable
    assert prediction.rule == "source_edition"


def test_title_and_episode_only_is_too_weak_to_predict():
    prediction = PREDICTOR.predict(
        _release("A.srt", tier=MatchTier.CLOSE, season_match=True, episode_match=True),
        FINGERPRINT,
    )
    assert prediction.availability is VerificationAvailability.UNKNOWN
    assert prediction.predicted_state is SyncState.UNVERIFIED
    assert not prediction.is_actionable


def test_fallback_tier_predicts_nothing():
    prediction = PREDICTOR.predict(_release("A.srt", tier=MatchTier.FALLBACK), FINGERPRINT)
    assert prediction.availability is VerificationAvailability.UNKNOWN
    assert prediction.rule == "fallback"


# --------------------------------------------------------------------------- #
# C. Prediction can never become VERIFIED
# --------------------------------------------------------------------------- #


def test_predictor_has_no_code_path_to_a_verified_state():
    """Structural guarantee, not just a spot check of the current rules."""
    source = Path(PREDICTOR.__class__.__module__.replace(".", "/") + ".py")
    text = source.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert node.value not in (
                SyncState.VERIFIED_SYNCED.value,
                SyncState.VERIFIED_RESYNCED.value,
            ), f"predictor references a verified state: {node.value}"


@pytest.mark.parametrize("tier", list(MatchTier))
@pytest.mark.parametrize("fingerprint", [FINGERPRINT, NO_FINGERPRINT, None])
def test_no_prediction_ever_returns_a_verified_state(tier, fingerprint):
    for accepted in (True, False):
        prediction = PREDICTOR.predict(
            _release("A.srt", tier=tier, accepted=accepted), fingerprint
        )
        assert prediction.predicted_state not in (
            SyncState.VERIFIED_SYNCED,
            SyncState.VERIFIED_RESYNCED,
        )


def test_weak_prediction_is_below_the_action_floor():
    weak = SyncPrediction(
        availability=VerificationAvailability.PREDICTED,
        predicted_state=SyncState.PROBABLE_SYNC,
        confidence=CONFIDENCE_PREDICTION_FLOOR - 1,
    )
    assert prediction_is_below_floor(weak)


# --------------------------------------------------------------------------- #
# D. Missing fingerprint stays UNKNOWN
# --------------------------------------------------------------------------- #


def test_missing_fingerprint_produces_unknown_prediction():
    prediction = PREDICTOR.predict(
        _release("A.srt", tier=MatchTier.HASH, is_hash_match=True), NO_FINGERPRINT
    )
    assert prediction.availability is VerificationAvailability.UNKNOWN
    assert prediction.rule == "no_fingerprint"
    assert prediction.confidence is None


def test_none_target_meta_produces_unknown_prediction():
    prediction = PREDICTOR.predict(_release("A.srt", tier=MatchTier.HASH), None)
    assert prediction.availability is VerificationAvailability.UNKNOWN


def test_prediction_never_infers_from_a_subtitle_filename_alone():
    """The regression guard for the target_filename contamination bug."""
    # A perfectly matching release name with no video fingerprint predicts
    # nothing, because the target is unknown.
    prediction = PREDICTOR.predict(
        _release("Dexter.S08E05.1080p.BluRay.x264-PiR8.srt", tier=MatchTier.EXACT),
        NO_FINGERPRINT,
    )
    assert prediction.availability is VerificationAvailability.UNKNOWN
    assert prediction.rule == "no_fingerprint"


# --------------------------------------------------------------------------- #
# E/F. Hard rejection is absolute
# --------------------------------------------------------------------------- #


def test_wrong_episode_stays_rejected_even_with_strong_metadata():
    rejected = _release(
        "A.srt",
        tier=MatchTier.HASH,
        is_hash_match=True,
        release_group_match=True,
        source_match=True,
        accepted=False,
        hard_reject_reason="episode mismatch",
    )
    prediction = PREDICTOR.predict(rejected, FINGERPRINT)
    assert prediction.predicted_state is SyncState.REJECTED
    assert prediction.availability is VerificationAvailability.UNKNOWN
    assert not prediction.is_actionable




# --------------------------------------------------------------------------- #
# A/B/G/H. Ordering semantics
# --------------------------------------------------------------------------- #


def test_exact_score_outweighs_unscoped_verified_claim():
    verified = _attach(
        _release("A.srt"),
        SyncState.VERIFIED_SYNCED.value,
        "cached",
        base=5,
        confidence=98.0,
    )
    unverified = _attach(_release("B.srt", tier=MatchTier.HASH), None, None, base=0)
    ordered = order_candidates([unverified, verified])
    assert ordered[0] is unverified


def test_displayed_score_orders_verified_sync_states():
    """Explicit product semantics: re-timing is not better evidence."""
    already = _attach(
        _release("A.srt", tier=MatchTier.EXACT),
        SyncState.VERIFIED_SYNCED.value,
        "cached",
        base=3,
        confidence=70.0,
    )
    resynced = _attach(
        _release("B.srt", tier=MatchTier.HASH),
        SyncState.VERIFIED_RESYNCED.value,
        "cached",
        base=0,
        confidence=100.0,
    )
    ordered = order_candidates([resynced, already])
    assert ordered[0] is resynced






def test_weak_prediction_does_not_outrank_unknown_with_better_content():
    """A low-confidence prediction must not promote a weak candidate."""
    predicted = _attach(
        _release("A.srt", tier=MatchTier.CLOSE),
        SyncState.PROBABLE_SYNC.value,
        VerificationAvailability.PREDICTED.value,
        base=5,
        confidence=40.0,
    )
    unknown = _attach(_release("B.srt", tier=MatchTier.EXACT), None, None, base=0)
    ordered = order_candidates([predicted, unknown])
    assert ordered[0] is unknown


def test_user_language_preference_outranks_sync_evidence():
    """An explicit user choice is never reordered by a heuristic."""
    preferred_unknown = _attach(_release("A.srt", lang="ara"), None, None, base=0, lang_rank=0)
    other_cached = _attach(
        _release("B.srt", lang="eng"),
        SyncState.VERIFIED_SYNCED.value,
        "cached",
        base=5,
        confidence=99.0,
        lang_rank=1,
    )
    ordered = order_candidates([other_cached, preferred_unknown])
    assert ordered[0] is preferred_unknown


# --------------------------------------------------------------------------- #
# Order matrix from the spec
# --------------------------------------------------------------------------- #






# --------------------------------------------------------------------------- #
# I/J/K/L. Guarantees about the search path
# --------------------------------------------------------------------------- #


def test_search_path_performs_zero_alass_executions():
    """Structural check: nothing on the search path may invoke alass."""
    main_source = Path("app/main.py").read_text(encoding="utf-8")
    tree = ast.parse(main_source)

    search_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_fetch_subtitles_handler":
            search_fn = node
            break
    assert search_fn is not None, "search handler not found"

    called = {
        node.func.attr
        for node in ast.walk(search_fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    forbidden = {"sync_async", "sync", "evaluate_and_sync"}
    assert not (called & forbidden), f"search path calls {called & forbidden}"

    # alass is only reachable through SubtitleSyncService, which the search
    # handler never constructs.
    constructed = {
        node.func.id
        for node in ast.walk(search_fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "SubtitleSyncService" not in constructed


def test_serve_time_top_n_behaviour_is_unchanged():
    """The alass candidate limit still applies on the serve path."""
    from app.config import settings

    assert int(getattr(settings, "ALASS_CANDIDATE_LIMIT", 3)) >= 1
    source = Path("app/services/sync/orchestrator.py").read_text(encoding="utf-8")
    assert "alass candidate limit reached" in source
    assert "_alass_candidate_limit" in source


def test_cache_lookup_is_fingerprint_bound():
    from app.services.sync_cache import SyncCache

    a = SyncCache.build_verdict_alias_key("fpA", "cand")
    b = SyncCache.build_verdict_alias_key("fpB", "cand")
    assert a != b


def test_ordering_is_deterministic_across_repeated_runs_and_input_orders():
    candidates = [
        _attach(_release("C.srt"), SyncState.PROBABLE_SYNC.value, "predicted", base=2,
               confidence=80.0),
        _attach(_release("A.srt"), SyncState.VERIFIED_SYNCED.value, "cached", base=5,
               confidence=99.0),
        _attach(_release("B.srt"), None, None, base=0),
        _attach(_release("D.srt"), SyncState.VERIFIED_RESYNCED.value, "cached", base=1,
               confidence=90.0),
    ]
    expected = [r.release_name for r in order_candidates(candidates)]
    for _ in range(10):
        assert [r.release_name for r in order_candidates(list(candidates))] == expected
    assert [r.release_name for r in order_candidates(list(reversed(candidates)))] == expected
    # All keys are distinct here, so the result must not depend on input order.
    assert len({comparison_key(c) for c in candidates}) == len(candidates)


def test_predictor_is_deterministic():
    release = _release("A.srt", tier=MatchTier.EXACT, release_group_match=True)
    results = {PREDICTOR.predict(release, FINGERPRINT).explain() for _ in range(10)}
    assert len(results) == 1


def test_ordering_does_not_mutate_or_depend_on_dict_ordering():
    candidates = [_attach(_release("B.srt"), None, None, base=1),
                  _attach(_release("A.srt"), None, None, base=0)]
    before = [r.release_name for r in candidates]
    order_candidates(candidates)
    assert [r.release_name for r in candidates] == before
