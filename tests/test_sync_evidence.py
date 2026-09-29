"""Evidence availability, verdict caching, ordering, and the false-positive gate.

Covers the layers added after the initial alignment work: VerificationAvailability,
structural comparison, cut classification, the measured-verdict cache, the Top-N
alass limit, deterministic ordering, and the benchmark that guards all of it.
"""

from __future__ import annotations

import json

import pytest
from sync_benchmark import (
    assert_no_known_false_positives,
    build_dataset,
    build_srt,
    even,
    run_benchmark,
)

from app.models import MatchTier, SubtitleRelease
from app.services.sync.alignment import (
    MIN_CUES_FOR_VERIFIED,
    AlignmentAnalyzer,
    RejectionReason,
    SubtitleEvaluation,
    SyncState,
    VerificationAvailability,
)
from app.services.sync.ordering import comparison_key, order_candidates
from app.services.sync.structural import (
    MIN_STRUCTURAL_SIMILARITY,
    CutVerdict,
    StructuralProfile,
    classify_cut,
    compare_structures,
)
from app.services.sync_cache import SYNC_VERDICT_ENGINE_VERSION, SyncCache

ANALYZER = AlignmentAnalyzer()


def _release(name: str, **kwargs) -> SubtitleRelease:
    return SubtitleRelease(
        release_name=name, download_url=f"http://{name}", provider="subdl", lang="ara", **kwargs
    )


# --------------------------------------------------------------------------- #
# 1. Evidence availability
# --------------------------------------------------------------------------- #


def test_predicted_evidence_cannot_back_a_verified_state():
    """Metadata-only inference must never read as a measurement."""
    evaluation = SubtitleEvaluation(
        sync_state=SyncState.VERIFIED_SYNCED, verification=VerificationAvailability.PREDICTED
    )
    assert evaluation.sync_state is SyncState.PROBABLE_SYNC
    assert evaluation.verification is VerificationAvailability.PREDICTED
    assert evaluation.rejection_reason is RejectionReason.INSUFFICIENT_EVIDENCE
    assert any("downgraded" in reason for reason in evaluation.reasons)


def test_unknown_evidence_cannot_back_a_verified_state():
    evaluation = SubtitleEvaluation(
        sync_state=SyncState.VERIFIED_RESYNCED, verification=VerificationAvailability.UNKNOWN
    )
    assert evaluation.sync_state is SyncState.UNVERIFIED
    assert evaluation.measured() is False


def test_measured_and_cached_evidence_are_accepted():
    for availability in (VerificationAvailability.VERIFIED, VerificationAvailability.CACHED):
        evaluation = SubtitleEvaluation(
            sync_state=SyncState.VERIFIED_RESYNCED, verification=availability
        )
        assert evaluation.sync_state is SyncState.VERIFIED_RESYNCED
        assert evaluation.measured() is True


def test_verdict_guard_also_applies_to_in_place_updates():
    evaluation = SubtitleEvaluation()
    evaluation.set_verdict(SyncState.VERIFIED_SYNCED, VerificationAvailability.PREDICTED)
    assert evaluation.sync_state is SyncState.PROBABLE_SYNC

    evaluation.set_verdict(SyncState.VERIFIED_RESYNCED, VerificationAvailability.VERIFIED)
    assert evaluation.sync_state is SyncState.VERIFIED_RESYNCED


def test_availability_and_state_are_independent_dimensions():
    assert SyncState.VERIFIED_SYNCED.rank < SyncState.VERIFIED_RESYNCED.rank
    assert VerificationAvailability.VERIFIED.rank < VerificationAvailability.CACHED.rank
    assert VerificationAvailability.PREDICTED.rank < VerificationAvailability.UNKNOWN.rank


# --------------------------------------------------------------------------- #
# 2. Structural comparison
# --------------------------------------------------------------------------- #


def test_structural_profile_of_a_subtitle():
    profile = StructuralProfile.from_subtitle(build_srt(even(20)))
    assert profile.cue_count == 20
    assert profile.median_duration_ms == 1500
    assert profile.density_cues_per_minute is not None
    assert profile.first_dialogue_ms == 0
    assert profile.cluster_count >= 1


def test_identical_structure_scores_high():
    similarity = compare_structures(build_srt(even(40)), build_srt(even(40)))
    assert similarity.score is not None
    assert similarity.same_structure
    assert similarity.cue_count_ratio == pytest.approx(1.0)


def test_different_structure_scores_low():
    similarity = compare_structures(
        build_srt(even(40)), build_srt(even(12, step_ms=6_000))
    )
    assert similarity.score is not None
    assert similarity.score < similarity_for_near_cue_count()


def similarity_for_near_cue_count():
    """A modest cue-count difference is tolerated, not treated as a new cut."""
    return compare_structures(build_srt(even(40)), build_srt(even(34, step_ms=2_400))).score or 0.0


def test_cue_splitting_is_tolerated():
    """Providers split cues differently; that must not read as a new cut."""
    # 40 cues vs the same content at 2x the cadence.
    similarity = compare_structures(build_srt(even(40)), build_srt(even(80, step_ms=1_000)))
    assert similarity.score is not None
    assert similarity.score > MIN_STRUCTURAL_SIMILARITY


def test_empty_side_reports_no_comparison():
    assert compare_structures(None, build_srt(even(10))).score is None
    assert compare_structures(build_srt(even(10)), None).score is None
    assert compare_structures(None, None).score is None


# --------------------------------------------------------------------------- #
# 3. Cut classification
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"median_offset_ms": 0.0, "p95_offset_ms": 0.0, "mad_offset_ms": 0.0}, CutVerdict.STABLE_OFFSET),
        (
            {"median_offset_ms": 1_800.0, "p95_offset_ms": 1_900.0, "mad_offset_ms": 50.0},
            CutVerdict.STABLE_OFFSET,
        ),
        (
            {"median_offset_ms": 5_000.0, "drift_ms_per_minute": 400.0},
            CutVerdict.DRIFT,
        ),
        (
            {"median_offset_ms": 30_000.0, "change_points": [(600_000, 29_000.0)]},
            CutVerdict.PIECEWISE,
        ),
        (
            {
                "median_offset_ms": 97_000.0,
                "change_points": [(600_000, 97_000.0)],
                "structural": compare_structures(
                    build_srt(even(40)), build_srt(even(10, step_ms=8_000))
                ),
            },
            CutVerdict.DIFFERENT_CUT,
        ),
        ({"median_offset_ms": None}, CutVerdict.UNKNOWN),
    ],
)
def test_cut_classification(kwargs, expected):
    call = {
        "p95_offset_ms": None,
        "mad_offset_ms": None,
        "drift_ms_per_minute": None,
        "change_points": [],
        "structural": None,
        "max_plausible_offset_ms": 20_000.0,
        "max_p95_ms": 2_000.0,
        "max_drift_ms_per_minute": 120.0,
    }
    call.update(kwargs)
    assert classify_cut(**call) is expected


def test_large_stable_offset_is_not_treated_as_an_error():
    """A big but rigid shift is re-timable, not automatically a wrong cut."""
    verdict = classify_cut(
        median_offset_ms=14_000.0,
        p95_offset_ms=14_000.0,
        mad_offset_ms=0.0,
        drift_ms_per_minute=0.0,
        change_points=[],
        structural=None,
        max_plausible_offset_ms=20_000.0,
        max_p95_ms=2_000.0,
        max_drift_ms_per_minute=120.0,
    )
    assert verdict is CutVerdict.STABLE_OFFSET


def test_small_offset_with_foreign_structure_is_not_proof_of_same_cut():
    verdict = classify_cut(
        median_offset_ms=400.0,
        p95_offset_ms=400.0,
        mad_offset_ms=0.0,
        drift_ms_per_minute=0.0,
        change_points=[],
        structural=compare_structures(build_srt(even(40)), build_srt(even(8, step_ms=10_000))),
        max_plausible_offset_ms=20_000.0,
        max_p95_ms=2_000.0,
        max_drift_ms_per_minute=120.0,
    )
    assert verdict is CutVerdict.DIFFERENT_CUT


# --------------------------------------------------------------------------- #
# 4. Evidence floor
# --------------------------------------------------------------------------- #


def test_sparse_cues_cannot_reach_a_verified_claim():
    """The benchmark's sparse case: timings measure cleanly but prove little."""
    base = even(MIN_CUES_FOR_VERIFIED - 6)
    evaluation = ANALYZER.analyze(build_srt(base), None, build_srt(base))
    assert evaluation.sync_state is not SyncState.VERIFIED_SYNCED
    assert evaluation.sync_state is SyncState.PROBABLE_SYNC
    assert any("below the" in reason for reason in evaluation.reasons)


def test_enough_cues_can_reach_verified():
    base = even(40)
    evaluation = ANALYZER.analyze(build_srt(base), None, build_srt(base))
    assert evaluation.sync_state is SyncState.VERIFIED_SYNCED


# --------------------------------------------------------------------------- #
# 5. Verdict cache
# --------------------------------------------------------------------------- #


def test_verdict_key_binds_video_subtitle_language_and_engine():
    a = SyncCache.build_verdict_key("fp1", "sub1", "ara")
    assert SyncCache.build_verdict_key("fp2", "sub1", "ara") != a
    assert SyncCache.build_verdict_key("fp1", "sub2", "ara") != a
    assert SyncCache.build_verdict_key("fp1", "sub1", "eng") != a
    assert SyncCache.build_verdict_key("fp1", "sub1", "ara", 99) != a
    assert f":{SYNC_VERDICT_ENGINE_VERSION}:" in a


def test_fingerprint_requires_real_video_signals():
    assert SyncCache.video_fingerprint_from_meta({"imdb_id": "tt1", "title": "X"}) is None
    assert (
        SyncCache.video_fingerprint_from_meta(
            {"imdb_id": "tt1", "season": 8, "episode": 5, "title": "X", "year": 2006}
        )
        is None
    )
    with_stream = SyncCache.video_fingerprint_from_meta(
        {
            "imdb_id": "tt1",
            "season": 8,
            "episode": 5,
            "target_filename": "Show.S08E05.1080p.BluRay-GRP.mkv",
        }
    )
    assert with_stream
    # A different video must not share a fingerprint.
    other = SyncCache.video_fingerprint_from_meta(
        {
            "imdb_id": "tt1",
            "season": 8,
            "episode": 5,
            "target_filename": "Show.S08E05.1080p.WEB-DL-OTHER.mkv",
        }
    )
    assert other != with_stream


@pytest.mark.asyncio
async def test_only_measured_verdicts_are_cached():
    cache = SyncCache()
    key = SyncCache.build_verdict_key("fp1", "sub1", "ara")

    await cache.set_verdict(key, {"sync_state": "verified_resynced", "verification": "unknown"})
    assert await cache.get_verdict(key) is None

    await cache.set_verdict(key, {"sync_state": "verified_resynced", "verification": "predicted"})
    assert await cache.get_verdict(key) is None

    await cache.set_verdict(
        key, {"sync_state": "verified_resynced", "verification": "verified", "reasons": ["ok"]}
    )
    stored = await cache.get_verdict(key)
    assert stored["sync_state"] == "verified_resynced"
    # A recalled verdict is never presented as a fresh measurement.
    assert stored["verification"] == "cached"


@pytest.mark.asyncio
async def test_alass_process_success_alone_is_not_a_verdict():
    """Exit code 0 with no measured outcome must not be cached as a verdict."""
    cache = SyncCache()
    key = SyncCache.build_verdict_key("fp1", "sub1", "ara")
    await cache.set_verdict(
        key,
        {
            "sync_state": "unverified",
            "verification": "verified",
            "alass_successful": True,
            "alass_applied": True,
        },
    )
    stored = await cache.get_verdict(key)
    # It is stored, but it is stored as unverified - never as synced.
    assert stored["sync_state"] == "unverified"


@pytest.mark.asyncio
async def test_stale_engine_version_is_never_reused():
    cache = SyncCache()
    key = SyncCache.build_verdict_key("fp1", "sub1", "ara")
    cache._local_verdict[key] = json.dumps(
        {"sync_state": "verified_resynced", "verification": "verified", "engine_version": 0}
    )
    assert await cache.get_verdict(key) is None


@pytest.mark.asyncio
async def test_verdict_does_not_leak_across_videos_or_subtitles():
    cache = SyncCache()
    key = SyncCache.build_verdict_key("fp1", "sub1", "ara")
    await cache.set_verdict(key, {"sync_state": "verified_resynced", "verification": "verified"})
    assert await cache.get_verdict(SyncCache.build_verdict_key("fp2", "sub1", "ara")) is None
    assert await cache.get_verdict(SyncCache.build_verdict_key("fp1", "sub2", "ara")) is None


@pytest.mark.asyncio
async def test_alias_index_is_bound_to_the_video_fingerprint():
    cache = SyncCache()
    primary = SyncCache.build_verdict_key("fp1", "sub1", "ara")
    alias = SyncCache.build_verdict_alias_key("fp1", "cand-1")
    await cache.set_verdict(
        primary, {"sync_state": "verified_synced", "verification": "verified"}, alias_key=alias
    )
    assert (await cache.get_verdict_by_ref("fp1", "cand-1"))["sync_state"] == "verified_synced"
    assert await cache.get_verdict_by_ref("fp2", "cand-1") is None


# --------------------------------------------------------------------------- #
# 6. Deterministic ordering
# --------------------------------------------------------------------------- #


def _with_sync(name: str, state: str | None, verification: str | None, **kwargs):
    release = _release(name, **kwargs)
    release.sync_state = state
    release.sync_verification = verification
    return release


def test_verified_synced_never_loses_to_verified_resynced():
    """Re-timing is not better evidence than already being right."""
    already = _with_sync("A.srt", "verified_synced", "verified", match_tier=MatchTier.CLOSE)
    resynced = _with_sync(
        "B.srt", "verified_resynced", "verified", match_tier=MatchTier.HASH
    )
    ordered = order_candidates([resynced, already])
    assert ordered[0] is already
    assert ordered[1] is resynced


def test_sync_evidence_outranks_a_better_match_tier():
    weak_but_verified = _with_sync(
        "A.srt", "verified_resynced", "verified", match_tier=MatchTier.FALLBACK
    )
    strong_but_unverified = _with_sync(
        "B.srt", "unverified", "unknown", match_tier=MatchTier.HASH
    )
    ordered = order_candidates([strong_but_unverified, weak_but_verified])
    assert ordered[0] is weak_but_verified


def test_match_tier_still_decides_within_equal_sync_evidence():
    hash_match = _with_sync("A.srt", None, None, match_tier=MatchTier.HASH, match_percentage=40)
    fallback = _with_sync("B.srt", None, None, match_tier=MatchTier.FALLBACK, match_percentage=99)
    ordered = order_candidates([fallback, hash_match])
    assert ordered[0] is hash_match


def test_missing_claim_never_outranks_a_real_claim():
    no_claim = _with_sync("A.srt", None, None, match_tier=MatchTier.CLOSE)
    probable = _with_sync(
        "B.srt", "probable_sync", "verified", match_tier=MatchTier.FALLBACK
    )
    ordered = order_candidates([no_claim, probable])
    assert ordered[0] is probable


def test_rejected_sorts_last():
    rejected = _with_sync("A.srt", "rejected", "verified", match_tier=MatchTier.HASH)
    unverified = _with_sync("B.srt", "unverified", "unknown", match_tier=MatchTier.FALLBACK)
    ordered = order_candidates([rejected, unverified])
    assert ordered[-1] is rejected


def test_displayed_score_semantics_are_untouched():
    """match_percentage keeps meaning compatibility; sync fields are separate."""
    release = _release("A.srt", match_tier=MatchTier.CLOSE)
    release.match_percentage = 95
    assert release.match_percentage == 95
    # No synchronization claim is implied by the score itself.
    assert release.sync_state is None
    assert release.sync_verification is None


def test_ordering_is_deterministic_and_input_order_independent():
    candidates = [
        _with_sync("C.srt", "probable_sync", "cached", match_tier=MatchTier.CLOSE),
        _with_sync("A.srt", "verified_synced", "verified", match_tier=MatchTier.EXACT),
        _with_sync("B.srt", None, None, match_tier=MatchTier.HASH),
        _with_sync("D.srt", "unverified", "unknown", match_tier=MatchTier.SOURCE_FAMILY),
    ]
    first = [r.release_name for r in order_candidates(candidates)]
    for _ in range(5):
        assert [r.release_name for r in order_candidates(list(candidates))] == first
    # Reversing the input must not change the outcome.
    reversed_order = [r.release_name for r in order_candidates(list(reversed(candidates)))]
    assert reversed_order == first


def test_ordering_does_not_mutate_input():
    candidates = [_with_sync("B.srt", "unverified", "unknown"), _with_sync("A.srt", "verified_synced", "verified")]
    order_candidates(candidates)
    assert [c.release_name for c in candidates] == ["B.srt", "A.srt"]


def test_comparison_key_is_total_and_stable():
    a = _with_sync("A.srt", "verified_synced", "verified")
    b = _with_sync("A.srt", "verified_synced", "verified")
    assert comparison_key(a) == comparison_key(b)


# --------------------------------------------------------------------------- #
# 7. The false-positive gate
# --------------------------------------------------------------------------- #


def test_benchmark_has_no_known_false_positives():
    report = run_benchmark()
    assert_no_known_false_positives(report)
    assert report.false_rejected == [], report.details


def test_benchmark_reports_the_four_tracked_numbers():
    report = run_benchmark()
    assert report.total == len(build_dataset())
    assert report.true_verified_count >= 3
    assert report.false_verified_count == 0
    assert report.precision == 1.0
    # The report is human-readable for regression triage.
    assert "false_verified=0" in report.summary()


def test_benchmark_is_deterministic():
    first = run_benchmark().details
    second = run_benchmark().details
    assert first == second


def test_no_scenario_is_claimed_verified_without_measurement():
    for scenario in build_dataset():
        evaluation = ANALYZER.analyze(
            scenario.target,
            scenario.synced,
            scenario.reference,
            alass_applied=scenario.alass_applied,
            alass_successful=scenario.alass_successful,
        )
        if evaluation.sync_state in (SyncState.VERIFIED_SYNCED, SyncState.VERIFIED_RESYNCED):
            assert evaluation.verification is VerificationAvailability.VERIFIED


# --------------------------------------------------------------------------- #
# 8. target_filename invariant
# --------------------------------------------------------------------------- #


def test_target_filename_invariant_across_the_pipeline():
    """A subtitle name must never become the target video filename."""
    from app.main import _media_context_from_request, _merge_sync_meta, _subtitle_context_query

    subtitle_name = "Dexter.2006.S08E05.srt"
    query = _subtitle_context_query("tt0773262", "series", 8, 5, None, None, None)
    assert "filename" not in query

    class _QueryParams(dict):
        def get(self, key, default=None):
            value = super().get(key)
            return default if value is None else value

    request = type("R", (), {"query_params": _QueryParams(_pairs(query))})()
    context = _media_context_from_request(request)
    assert "target_filename" not in context

    merged = _merge_sync_meta(
        {"imdb_id": "tt1", "release_name": subtitle_name, "season": 8, "episode": 5}, context
    )
    assert merged.get("target_filename") is None
    assert merged["has_video_fingerprint"] is False
    # The display name is untouched.
    assert merged["release_name"] == subtitle_name


def _pairs(query: str) -> list[tuple[str, str]]:
    import urllib.parse

    return urllib.parse.parse_qsl(query)
