"""Target-bound reference selection regressions.

Derived from a production incident: target
``Dexter.s8e05.This.little.piggy.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv``
(videosize 5192387053), requested Arabic subtitle timed ~106.95s, and every
English reference candidate timed ~10.16s / ~9.16s. The resolver downloaded six
English candidates, rejected all six with the same ~+97s delta, and aborted.

The gate was correct. The decision was wrong in shape: the reference was chosen
by which reference happened to fit the requested subtitle, and the run was six
repetitions of one experiment.

Every fixture here is timing and metadata only: cue counts, first/last
dialogue, spans, tiers. No subtitle text.
"""

from __future__ import annotations

import inspect

import pytest

from app.models import MatchTier
from app.services.sync.reference_v2 import (
    CANDIDATE_DIFFERENT_TIMELINE,
    NO_TARGET_BOUND_REFERENCE,
    REFERENCE_LOW_HEALTH,
    TARGET_REFERENCE_SELECTED,
    ReferenceAttempt,
    decide_target_bound_reference,
    reference_family_counts,
    release_family_key,
    summarize_target_bound,
)

TARGET = "Dexter.s8e05.This.little.piggy.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv"

# The two timing models from the incident log. Family A is the 1080p/BluRay
# model the target belongs to; family B is the shorter cut the requested
# Arabic subtitle was timed against.
FAMILY_A = {
    "first_dialogue_ms": 10_160,
    "last_dialogue_ms": 2_805_160,
    "span_ms": 2_795_000,
    "cue_count": 1_142,
}
FAMILY_B = {
    "first_dialogue_ms": 106_950,
    "last_dialogue_ms": 2_795_950,
    "span_ms": 2_689_000,
    "cue_count": 1_098,
}


def attempt(
    name: str,
    *,
    tier: MatchTier,
    ok: bool | None,
    family: str | None = None,
    health: str = "good",
    hard_accepted: bool = True,
    provider: str | None = "p1",
) -> ReferenceAttempt:
    return ReferenceAttempt(
        reference_id=name[:16],
        release_name=name,
        provider=provider,
        match_tier=tier,
        hard_accepted=hard_accepted,
        health_class=health,
        family=family if family is not None else release_family_key(name),
        downloaded=ok is not None,
        cue_sanity_ok=ok,
    )


DEMAND = "Dexter.S08.720p.BluRay.x264-DEMAND.srt"
ROVERS = "Dexter.S08.1080p.BluRay.x265-ROVERS.srt"
WEAK = "Dexter.S08.720p.WEB-DL.x264-GRP.srt"


# --- the causal-order invariant -------------------------------------------


def test_selector_has_no_candidate_parameter():
    """Step A must be structurally incapable of consulting the candidate."""
    params = list(inspect.signature(decide_target_bound_reference).parameters)
    assert params == ["attempts"]


def test_decision_never_reads_a_candidate_field():
    """No attempt field may encode 'fits the requested subtitle'."""
    fields = set(ReferenceAttempt.model_fields)
    for forbidden in {"candidate_first_ms", "candidate_delta_ms", "candidate_family"}:
        assert forbidden not in fields


# --- incident regressions --------------------------------------------------


def test_dexter_different_cut_fails_closed():
    """~107s candidate vs ~10s target family: no synchronization.

    All target-accepted references disagree with the candidate, so the correct
    outcome is an explicit refusal, not a substituted reference.
    """
    decision = decide_target_bound_reference(
        [
            attempt(DEMAND, tier=MatchTier.SOURCE_FAMILY, ok=False),
            attempt(ROVERS, tier=MatchTier.CLOSE, ok=False),
        ]
    )
    assert decision.outcome == CANDIDATE_DIFFERENT_TIMELINE
    assert decision.selected is None
    assert not decision.usable
    assert any(CANDIDATE_DIFFERENT_TIMELINE in r for r in decision.reasons)


def test_dexter_positive_counterpart_proceeds():
    """Same target, candidate in the target's own timing family: proceed."""
    decision = decide_target_bound_reference(
        [attempt(ROVERS, tier=MatchTier.EXACT, ok=True)]
    )
    assert decision.outcome == TARGET_REFERENCE_SELECTED
    assert decision.selected is not None
    assert decision.selected.release_name == ROVERS
    assert decision.usable


def test_candidate_cannot_rescue_weaker_reference():
    """The circularity guard.

    A strong target match fails; a weak one fits the candidate. The weak one
    must not be promoted, because that is exactly how the requested subtitle
    would choose the reference.
    """
    decision = decide_target_bound_reference(
        [
            attempt(ROVERS, tier=MatchTier.EXACT, ok=False),
            attempt(WEAK, tier=MatchTier.FALLBACK, ok=True),
        ]
    )
    assert decision.outcome == CANDIDATE_DIFFERENT_TIMELINE
    assert decision.selected is None
    assert len(decision.suppressed_as_circular) == 1


def test_first_passing_candidate_regression():
    """A weak reference listed first must not win by fitting the candidate.

    Two arrangements, because the old architecture broke on the first pass.
    """
    # Weak reference passes first, strong one fails: fail closed, never promote.
    strict = decide_target_bound_reference(
        [
            attempt(WEAK, tier=MatchTier.FALLBACK, ok=True),
            attempt(ROVERS, tier=MatchTier.EXACT, ok=False),
        ]
    )
    assert strict.outcome == CANDIDATE_DIFFERENT_TIMELINE
    assert strict.selected is None

    # Both pass: target identity decides, and the strong one is chosen even
    # though the weak one appears earlier in provider ordering.
    both = decide_target_bound_reference(
        [
            attempt(WEAK, tier=MatchTier.FALLBACK, ok=True),
            attempt(ROVERS, tier=MatchTier.EXACT, ok=True),
        ]
    )
    assert both.outcome == TARGET_REFERENCE_SELECTED
    assert both.selected.release_name == ROVERS


def test_stronger_target_reference_wins_over_provider_order():
    """B has strong exact target metadata; A is weak. B is chosen."""
    decision = decide_target_bound_reference(
        [
            attempt(WEAK, tier=MatchTier.CLOSE, ok=True, provider="first"),
            attempt(ROVERS, tier=MatchTier.EXACT, ok=True, provider="last"),
        ]
    )
    assert decision.selected is not None
    assert decision.selected.match_tier == MatchTier.EXACT
    assert decision.selected.release_name == ROVERS


def test_outlier_reference_is_not_selected_by_provider_count():
    """A and C share the target family; B is the outlier.

    B is healthy and passes, but A is the stronger target match, so A is the
    anchor. Provider count is never the deciding term.
    """
    decision = decide_target_bound_reference(
        [
            attempt(ROVERS, tier=MatchTier.EXACT, ok=True, provider="p1"),
            attempt(ROVERS + ".copy", tier=MatchTier.EXACT, ok=True, provider="p2"),
            attempt(WEAK, tier=MatchTier.FALLBACK, ok=True, provider="p3"),
        ]
    )
    assert decision.selected is not None
    assert decision.selected.match_tier == MatchTier.EXACT
    assert decision.selected.release_name == ROVERS


def test_provider_count_is_not_independent_evidence():
    """Six providers serving one release is one timing model, not six votes."""
    siblings = [
        attempt(DEMAND, tier=MatchTier.SOURCE_FAMILY, ok=False, provider=f"p{i}")
        for i in range(6)
    ]
    decision = decide_target_bound_reference(siblings)
    counts = reference_family_counts(decision.attempts)
    assert list(counts.values()) == [6]
    assert decision.unique_families == 1
    assert decision.duplicate_families_avoided == 5
    assert decision.outcome == CANDIDATE_DIFFERENT_TIMELINE


# --- fail-closed paths -----------------------------------------------------


def test_no_target_bound_reference_when_everything_mismatches_the_target():
    decision = decide_target_bound_reference(
        [attempt(WEAK, tier=MatchTier.FALLBACK, ok=None, hard_accepted=False)]
    )
    assert decision.outcome == NO_TARGET_BOUND_REFERENCE
    assert decision.selected is None
    assert decision.downloads_used == 0


def test_health_orders_but_does_not_veto():
    """Health is a tie-breaker, not a standalone production gate.

    The shadow thresholds were calibrated for measurement. Promoting them to a
    hard veto was measured to refuse references the previous policy accepted,
    so health ranks equally-strong candidates and is recorded, but does not by
    itself cancel a reference. A target mismatch still cancels it absolutely.
    """
    # Two references of identical target strength: the healthier one wins.
    both = decide_target_bound_reference(
        [
            attempt(DEMAND, tier=MatchTier.SOURCE_FAMILY, ok=True, health="unusable"),
            attempt(ROVERS, tier=MatchTier.SOURCE_FAMILY, ok=True, health="healthy"),
        ]
    )
    assert both.selected is not None
    assert both.selected.release_name == ROVERS
    assert both.outcome == TARGET_REFERENCE_SELECTED

    # Alone, a low-health reference is still usable, and the reason is reported.
    alone = decide_target_bound_reference(
        [attempt(DEMAND, tier=MatchTier.SOURCE_FAMILY, ok=True, health="unusable")]
    )
    assert alone.selected is not None
    assert any(REFERENCE_LOW_HEALTH in r for r in alone.reasons)


def test_hard_target_mismatch_cannot_be_rescued_by_a_passing_gate():
    """A reference rejected against the target is never an anchor, even if it
    fits the candidate perfectly."""
    decision = decide_target_bound_reference(
        [attempt(WEAK, tier=MatchTier.EXACT, ok=True, hard_accepted=False)]
    )
    assert decision.selected is None
    assert decision.outcome == NO_TARGET_BOUND_REFERENCE


# --- observability ---------------------------------------------------------


def test_summary_carries_no_text_or_urls():
    decision = decide_target_bound_reference(
        [attempt(ROVERS, tier=MatchTier.EXACT, ok=True)]
    )
    summary = summarize_target_bound(decision)
    blob = repr(summary)
    for forbidden in ("http", "srt", "www", "api_key", "Bearer", TARGET.lower()):
        assert forbidden.lower() not in blob.lower()
    assert summary["match_tier"] == "EXACT"
    assert summary["outcome"] == TARGET_REFERENCE_SELECTED


def test_decision_is_deterministic():
    attempts = [
        attempt(ROVERS, tier=MatchTier.EXACT, ok=True, provider="p1"),
        attempt(WEAK, tier=MatchTier.CLOSE, ok=True, provider="p2"),
    ]
    first = decide_target_bound_reference(attempts)
    second = decide_target_bound_reference(list(reversed(attempts)))
    assert first.selected is not None and second.selected is not None
    assert first.selected.release_name == second.selected.release_name


def test_timing_families_are_distinct_for_distinct_releases():
    assert release_family_key(DEMAND) == release_family_key(DEMAND + ".en")
    assert release_family_key(DEMAND) != release_family_key(ROVERS)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
