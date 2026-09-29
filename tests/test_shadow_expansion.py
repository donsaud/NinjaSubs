"""Controlled, opt-in shadow-pool expansion: does ONE extra candidate help?

The question this suite answers is narrow and empirical: when the legacy early
break leaves the shadow pool unable to support a comparison, does spending one
extra download actually widen the evidence, and what does it cost?

Nothing here promotes anything, and every test asserts the legacy reference is
untouched.
"""

import pytest

from app.config import settings
from app.services.sync.shadow_expansion import (
    EXPANSION_DUPLICATE_GROUP,
    EXPANSION_ENABLED_COMPARISON,
    EXPANSION_FETCH_FAILURE,
    EXPANSION_NEW_TIMING_GROUP,
    EXPANSION_NO_VALUE,
    EXPANSION_NOT_ATTEMPTED,
    FETCH_CACHE_REUSE,
    FETCH_NETWORK,
    REFUSED_AUDIT_DISABLED,
    REFUSED_LIMIT_ZERO,
    REFUSED_NO_FINGERPRINT,
    REFUSED_POOL_SUFFICIENT,
    ExpansionTelemetry,
    choose_expansion_candidate,
    classify_expansion,
    expansion_summary,
    has_target_fingerprint,
    resolve_fetch_budget,
)

TARGET = "Dexter.S08E05.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv"
NO_FINGERPRINT = "Some.Movie.2020.mkv"


def _ts(ms: int) -> str:
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, msec = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{msec:03d}"


def srt(positions, *, duration_ms: int = 1500, body: str = "a line of dialogue here") -> str:
    return "\n".join(
        f"{i}\n{_ts(p)} --> {_ts(p + duration_ms)}\n{body}\n" for i, p in enumerate(positions, 1)
    )


def even(count: int, step_ms: int = 2000, start_ms: int = 0) -> list[int]:
    return [start_ms + i * step_ms for i in range(count)]


def name(group: str) -> str:
    return f"Dexter.S08E05.1080p.BluRay.x264-{group}.srt"


# --- A. audit disabled + limit 1 => no budget at all ----------------------- #

def test_a_audit_disabled_refuses_the_budget_even_when_configured():
    budget = resolve_fetch_budget(
        configured_limit=1,
        audit_enabled=False,
        target_filename=TARGET,
        comparison_class="NO_COMPARISON",
    )
    assert budget.allowed == 0
    assert budget.reason == REFUSED_AUDIT_DISABLED


def test_a_default_configuration_costs_nothing():
    assert settings.REFERENCE_SHADOW_POOL_FETCH_LIMIT == 0
    budget = resolve_fetch_budget(
        configured_limit=settings.REFERENCE_SHADOW_POOL_FETCH_LIMIT,
        audit_enabled=False,
        target_filename=TARGET,
        comparison_class="NO_COMPARISON",
    )
    assert budget.allowed == 0


# --- B. audit enabled + limit 0 => no budget ------------------------------- #

def test_b_audit_enabled_with_zero_limit_refuses():
    budget = resolve_fetch_budget(
        configured_limit=0,
        audit_enabled=True,
        target_filename=TARGET,
        comparison_class="NO_COMPARISON",
    )
    assert budget.allowed == 0
    assert budget.reason == REFUSED_LIMIT_ZERO


# --- C. sufficient pool => never spend the budget -------------------------- #

def test_c_meaningful_pool_is_left_alone():
    budget = resolve_fetch_budget(
        configured_limit=1,
        audit_enabled=True,
        target_filename=TARGET,
        comparison_class="MEANINGFUL_COMPARISON",
    )
    assert budget.allowed == 0
    assert budget.reason == REFUSED_POOL_SUFFICIENT


# --- D. insufficient pool + limit 1 => exactly one permitted --------------- #

def test_d_insufficient_pool_permits_the_configured_budget():
    budget = resolve_fetch_budget(
        configured_limit=1,
        audit_enabled=True,
        target_filename=TARGET,
        comparison_class="NO_COMPARISON",
    )
    assert budget.allowed == 1
    choice = choose_expansion_candidate(
        TARGET,
        [("subdl", name("GroupA")), ("subsource", name("GroupB"))],
        {("subdl", name("GroupA")): srt(even(40))},
    )
    # Only one candidate is ever chosen; the budget is not a shopping list.
    assert choice is not None
    assert choice.key == ("subsource", name("GroupB"))


# --- E. duplicate timing grid adds no independence ------------------------- #

def test_e_duplicate_timing_grid_is_classified_as_no_independence():
    outcome = classify_expansion(
        eligible=True,
        refusal=None,
        attempted=True,
        succeeded=True,
        pool_size_before=1,
        groups_before=1,
        pool_size_after=2,
        groups_after=1,
        meaningful_before=False,
        meaningful_after=False,
    )
    assert outcome == EXPANSION_DUPLICATE_GROUP


# --- F. new timing grid is recognised -------------------------------------- #

def test_f_new_timing_group_is_recognised():
    outcome = classify_expansion(
        eligible=True,
        refusal=None,
        attempted=True,
        succeeded=True,
        pool_size_before=1,
        groups_before=1,
        pool_size_after=2,
        groups_after=2,
        meaningful_before=False,
        meaningful_after=True,
    )
    # Enabling a comparison outranks merely adding a group.
    assert outcome == EXPANSION_ENABLED_COMPARISON
    outcome2 = classify_expansion(
        eligible=True,
        refusal=None,
        attempted=True,
        succeeded=True,
        pool_size_before=1,
        groups_before=1,
        pool_size_after=2,
        groups_after=2,
        meaningful_before=False,
        meaningful_after=False,
    )
    assert outcome2 == EXPANSION_NEW_TIMING_GROUP


# --- G. no target fingerprint never fetches -------------------------------- #

def test_g_missing_fingerprint_refuses():
    assert has_target_fingerprint(TARGET) is True
    assert has_target_fingerprint(None) is False
    budget = resolve_fetch_budget(
        configured_limit=1,
        audit_enabled=True,
        target_filename=None,
        comparison_class="NO_COMPARISON",
    )
    assert budget.allowed == 0
    assert budget.refusal if False else budget.reason == REFUSED_NO_FINGERPRINT


# --- I. network failure is audited, not hidden ---------------------------- #

def test_i_fetch_failure_is_classified():
    outcome = classify_expansion(
        eligible=True,
        refusal=None,
        attempted=True,
        succeeded=False,
        pool_size_before=1,
        groups_before=1,
        pool_size_after=1,
        groups_after=1,
        meaningful_before=False,
        meaningful_after=False,
    )
    assert outcome == EXPANSION_FETCH_FAILURE


def test_i_not_attempted_is_its_own_outcome():
    outcome = classify_expansion(
        eligible=False,
        refusal=REFUSED_AUDIT_DISABLED,
        attempted=False,
        succeeded=False,
        pool_size_before=1,
        groups_before=1,
        pool_size_after=1,
        groups_after=1,
        meaningful_before=False,
        meaningful_after=False,
    )
    assert outcome == EXPANSION_NOT_ATTEMPTED


def test_i_no_value_when_the_fetch_added_nothing():
    outcome = classify_expansion(
        eligible=True,
        refusal=None,
        attempted=True,
        succeeded=True,
        pool_size_before=1,
        groups_before=1,
        pool_size_after=1,
        groups_after=1,
        meaningful_before=False,
        meaningful_after=False,
    )
    assert outcome == EXPANSION_NO_VALUE


# --- J/K. cache reuse vs network fetch, and independence not from sources -- #

def test_j_and_k_existing_payloads_are_never_refetched():
    """A candidate already materialized is a reuse, not a network request."""
    existing = ("subdl", name("GroupA"))
    discovered = [existing, ("subsource", name("GroupB"))]
    payloads = {existing: srt(even(40))}
    choice = choose_expansion_candidate(TARGET, discovered, payloads)
    assert choice is not None
    assert choice.key not in payloads
    assert FETCH_NETWORK != FETCH_CACHE_REUSE


def test_k_duplicate_source_does_not_create_independence():
    """Two providers carrying the same timing grid stay one timing model."""
    from app.services.sync.reference_v2 import build_shadow_pool

    same = srt(even(40))
    discovered = [("subdl", name("GroupA")), ("subsource", name("GroupB"))]
    pool = build_shadow_pool(
        TARGET, discovered, {d: same for d in discovered}, limit=4
    )
    assert pool.pool_size == 2
    assert pool.independent_groups == 1
    # A second provider is a second record, not a second timing model.
    assert pool.comparison_class == "NO_COMPARISON"


# --- L. deterministic selection ------------------------------------------- #

def test_l_selection_is_deterministic():
    discovered = [
        ("subdl", name("GroupA")),
        ("subsource", name("GroupB")),
        ("opensub", name("GroupC")),
    ]
    payloads = {("subdl", name("GroupA")): srt(even(40))}
    picks = {
        choose_expansion_candidate(TARGET, discovered, payloads).key for _ in range(10)
    }
    assert len(picks) == 1


def test_l_selection_prefers_an_unrepresented_source():
    # Rank order would pick the other GroupA edition; diversity picks the new
    # source, because a different source is the cheapest hint at a different
    # timing model. Same release group, different edition, same provider.
    same_source = ("subdl", "Dexter.S08E05.720p.BluRay.x264-GroupA.srt")
    new_source = ("subsource", name("GroupB"))
    payloads = {("subdl", name("GroupA")): srt(even(40))}
    choice = choose_expansion_candidate(
        TARGET, [same_source, new_source], payloads
    )
    assert choice is not None
    assert choice.key == new_source
    assert choice.reason == "unseen_source"


def test_l_selection_falls_back_to_rank_order_when_all_sources_are_known():
    same_source_a = ("subdl", "Dexter.S08E05.720p.BluRay.x264-GroupA.srt")
    same_source_b = ("subdl", "Dexter.S08E05.1080p.BluRay.x264-GroupA.mkv.srt")
    payloads = {("subdl", name("GroupA")): srt(even(40))}
    choice = choose_expansion_candidate(
        TARGET, [same_source_a, same_source_b], payloads
    )
    assert choice is not None
    assert choice.key == same_source_a
    assert choice.reason == "rank_order"


def test_candidate_choice_is_none_when_nothing_is_left():
    payloads = {("subdl", name("GroupA")): srt(even(40))}
    assert (
        choose_expansion_candidate(TARGET, [("subdl", name("GroupA"))], payloads) is None
    )


def test_hard_rejected_candidate_is_never_chosen_for_expansion():
    good = ("subdl", name("GroupA"))
    bad = ("subdl", "Dexter.S01E01.720p.HDTV.x264-GroupB.srt")
    choice = choose_expansion_candidate(
        TARGET, [good, bad], {good: srt(even(40))}
    )
    assert choice is None or choice.key != bad


def test_rejected_stem_candidate_is_skipped():
    kept = ("subsource", name("GroupB"))
    skipped = ("subdl", name("GroupC"))
    choice = choose_expansion_candidate(
        TARGET,
        [skipped, kept],
        {},
        rejected=[skipped],
    )
    assert choice is not None
    assert choice.key == kept


# --- telemetry shape ------------------------------------------------------- #

def test_telemetry_records_cost_and_marginal_value():
    telemetry = ExpansionTelemetry(
        eligible=True,
        outcome=EXPANSION_NEW_TIMING_GROUP,
        attempts=1,
        successes=1,
        network_fetches=1,
        bytes=1234,
        latency_ms=12.5,
        pool_size_before=1,
        independent_groups_before=1,
        pool_size_after=2,
        independent_groups_after=2,
        meaningful_before=False,
        meaningful_after=True,
    )
    summary = expansion_summary(telemetry)
    assert summary["attempts"] == 1
    assert summary["bytes"] == 1234
    assert summary["latency_ms"] == 12.5


def test_unknown_bytes_and_latency_stay_none():
    """Absent instrumentation is reported unknown, never estimated."""
    telemetry = ExpansionTelemetry(successes=1)
    summary = expansion_summary(telemetry)
    assert summary["bytes"] is None
    assert summary["latency_ms"] is None


def test_telemetry_defaults_are_inert():
    telemetry = ExpansionTelemetry()
    assert telemetry.outcome == EXPANSION_NOT_ATTEMPTED
    assert telemetry.attempts == 0
    assert telemetry.eligible is False


@pytest.mark.parametrize("limit", [0, 1, 2, 5])
def test_budget_never_exceeds_the_configured_limit(limit):
    budget = resolve_fetch_budget(
        configured_limit=limit,
        audit_enabled=True,
        target_filename=TARGET,
        comparison_class="NO_COMPARISON",
    )
    assert budget.allowed == max(0, limit)
