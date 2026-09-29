"""Bounded reference shadow pool: the measurement-bias fix.

The legacy resolver stops at the first candidate that passes cue sanity, so the
policy under evaluation chose the alternatives available to the policy being
evaluated against. These tests pin the pool that removes that bias, and pin the
isolation that keeps the legacy reference authoritative.
"""

from app.services.sync.reference_v2 import (
    COMPARISON_MEANINGFUL,
    COMPARISON_NO,
    build_shadow_pool,
    classify_comparison,
    classify_shadow_switch,
    select_reference_v2,
)

TARGET = "Dexter.S08E05.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv"
CREDITS = "sync by someone"


def _ts(ms: int) -> str:
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, msec = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{msec:03d}"


def srt(positions, *, duration_ms: int = 1500, body: str = "I never thought I would find someone") -> str:
    blocks = []
    for i, start in enumerate(positions, start=1):
        blocks.append(
            f"{i}\n{_ts(start)} --> {_ts(start + duration_ms)}\n{body}\n"
        )
    return "\n".join(blocks)


def even(count: int, step_ms: int = 2000, start_ms: int = 0) -> list[int]:
    return [start_ms + i * step_ms for i in range(count)]


def name(group: str) -> str:
    return f"Dexter.S08E05.1080p.BluRay.x264-{group}.srt"


# --- A: one candidate is not a comparison ---------------------------------- #

def test_pool_a_single_candidate_is_no_comparison():
    text = srt(even(40))
    pool = build_shadow_pool(TARGET, [("subdl", name("PiR8"))], {("subdl", name("PiR8")): text}, limit=4)
    assert pool.pool_size == 1
    assert pool.comparison_class == COMPARISON_NO
    # A one-candidate pool must never be reported as agreement.
    assert classify_comparison(pool) == COMPARISON_NO


# --- B: two genuinely different references can be compared ----------------- #

def test_pool_b_two_independent_candidates_is_meaningful():
    a, b = srt(even(40)), srt(even(40, step_ms=2100))
    pool = build_shadow_pool(
        TARGET,
        [("subdl", name("PiR8")), ("subsource", name("OtherGRP"))],
        {("subdl", name("PiR8")): a, ("subsource", name("OtherGRP")): b},
        limit=4,
    )
    assert pool.pool_size == 2
    assert pool.independent_groups == 2
    assert pool.comparison_class == COMPARISON_MEANINGFUL


# --- C: four copies of one timing grid are one reference ------------------- #

def test_pool_c_four_candidates_same_timing_grid_is_one_group():
    text = srt(even(40))
    discovered = [(p, name(f"G{i}")) for i, p in enumerate(["subdl", "subsource", "opensub", "extra"])]
    pool = build_shadow_pool(TARGET, discovered, {k: text for k in discovered}, limit=4)
    assert pool.pool_size == 4
    assert pool.independent_groups == 1
    # Four payloads, but the policies were only ever offered one timing model.
    assert pool.comparison_class == COMPARISON_NO


# --- D: four candidates spanning three timing grids ----------------------- #

def test_pool_d_four_candidates_three_timing_groups():
    texts = [srt(even(40)), srt(even(40, step_ms=2100)), srt(even(40, step_ms=2300)), srt(even(40))]
    discovered = [(f"p{i}", name(f"G{i}")) for i in range(4)]
    payloads = {k: texts[i] for i, k in enumerate(discovered)}
    pool = build_shadow_pool(TARGET, discovered, payloads, limit=4)
    assert pool.pool_size == 4
    assert pool.independent_groups == 3
    assert pool.comparison_class == COMPARISON_MEANINGFUL


# --- E: shadow differs, synchronizer still receives legacy ----------------- #

def test_pool_e_switch_labels_describe_without_claiming_improvement():
    credits = srt(even(40), body=CREDITS)
    healthy = srt(even(40))
    selection = select_reference_v2(
        TARGET, [("subdl", name("Credits"), credits), ("subdl", name("PiR8"), healthy)]
    )
    assert selection.chosen is not None
    legacy = next((c for c in selection.all_candidates if c.release_name == name("Credits")), None)
    labels = classify_shadow_switch(legacy, selection.chosen)
    assert "SHADOW_SWITCH" in labels
    assert any("health" in label for label in labels)
    assert any("not eligible" in label for label in labels)
    # Never an unqualified quality claim.
    assert not any("improvement" in label.lower() for label in labels)
    # The credits-only reference is the unhealthy one being left behind.
    assert legacy is not None and legacy.health_class == "unusable"
    assert selection.chosen.health_class == "healthy"


# --- F: agreement is only meaningful with a real alternative --------------- #

def test_pool_f_agreement_without_alternative_is_not_meaningful():
    text = srt(even(40))
    pool = build_shadow_pool(TARGET, [("subdl", name("PiR8"))], {("subdl", name("PiR8")): text}, limit=4)
    selection = select_reference_v2(TARGET, pool.references)
    comparison_class = pool.comparison_class
    assert selection.chosen is not None
    # Legacy and shadow necessarily pick the same single reference...
    assert selection.considered == 1
    # ...and that fact must be classified as a non-comparison, not as agreement.
    assert comparison_class == COMPARISON_NO


# --- G: the pool is bounded and the truncation is recorded ------------------ #

def test_pool_g_limit_exceeded_is_truncated_and_recorded():
    discovered = [(f"p{i}", name(f"G{i}")) for i in range(6)]
    payloads = {k: srt(even(40, step_ms=2000 + i * 50)) for i, k in enumerate(discovered)}
    pool = build_shadow_pool(TARGET, discovered, payloads, limit=4)
    assert pool.pool_size == 4
    assert pool.pool_limit == 4
    assert pool.truncated is True
    assert any("truncated" in r for r in pool.reasons)


def test_pool_g_zero_limit_disables_the_pool():
    pool = build_shadow_pool(TARGET, [("subdl", name("PiR8"))], {("subdl", name("PiR8")): srt(even(40))}, limit=0)
    assert pool.pool_size == 0
    assert pool.references == []


# --- H: hard-rejected candidates cannot enter the pool ---------------------- #

def test_pool_h_hard_rejected_candidate_cannot_enter_the_pool():
    good = ("subdl", name("PiR8"))
    # Wrong season/episode: the existing hard filter rejects it.
    bad = ("subdl", "Dexter.S01E01.720p.HDTV.x264-OtherGRP.srt")
    pool = build_shadow_pool(TARGET, [good, bad], {good: srt(even(40)), bad: srt(even(40, step_ms=2100))}, limit=4)
    assert [n for _p, n, _t in pool.references] == [name("PiR8")]
    assert pool.hard_rejected == 1


# --- I: health decides when content compatibility ties --------------------- #

def test_pool_i_better_health_alternative_wins_on_a_tie():
    credits = ("subdl", name("Credits"))
    healthy = ("subdl", name("PiR8"))
    pool = build_shadow_pool(TARGET, [credits, healthy], {credits: srt(even(40), body=CREDITS), healthy: srt(even(40))}, limit=4)
    selection = select_reference_v2(TARGET, pool.references)
    assert selection.chosen is not None
    assert selection.chosen.release_name == name("PiR8")


# --- J: exact verified cache evidence outranks an equal unverified one ------ #

def test_pool_j_verified_cache_candidate_is_preferred():
    # Deliberately equal-tier names: verified cache evidence is a tie-breaker
    # below content identity, so a content-tier difference would mask it.
    a, b = srt(even(40)), srt(even(40, step_ms=2100))
    discovered = [("subdl", name("Alpha")), ("subsource", name("Beta"))]
    payloads = {discovered[0]: a, discovered[1]: b}
    pool = build_shadow_pool(TARGET, discovered, payloads, limit=4)

    from app.services.sync.reference_v2 import _identity

    verified_id = _identity("subsource", name("Beta"), b)
    selection = select_reference_v2(TARGET, pool.references, verified_ids=frozenset({verified_id}))
    assert selection.chosen is not None
    assert selection.chosen.subtitle_id == verified_id
    assert selection.chosen.cache_verified is True


def test_pool_j_content_identity_still_outranks_verified_cache():
    # The exact-verified candidate is only preferred among content-compatible
    # candidates; a higher-tier reference must still win.
    strong = srt(even(40))
    weak = srt(even(40, step_ms=2100))
    refs = [("subdl", name("PiR8"), strong), ("subsource", name("Alpha"), weak)]
    from app.services.sync.reference_v2 import _identity

    verified_id = _identity("subsource", name("Alpha"), weak)
    selection = select_reference_v2(TARGET, refs, verified_ids=frozenset({verified_id}))
    assert selection.chosen is not None
    assert selection.chosen.release_name == name("PiR8")


# --- K: no fingerprint means unknown identity, never an invented one -------- #

def test_pool_k_no_fingerprint_fabricates_nothing():
    text = srt(even(40))
    pool = build_shadow_pool(None, [("subdl", name("PiR8"))], {("subdl", name("PiR8")): text}, limit=4)
    selection = select_reference_v2(None, pool.references)
    assert selection.chosen is not None
    # No target metadata was invented, so no content tier is claimed.
    assert selection.chosen.reasons
    assert any("unknown" in r for r in selection.chosen.reasons)
    assert not any("hard rejection" in r for r in selection.chosen.reasons)


# --- pool does not bias toward the legacy winner --------------------------- #

def test_pool_is_not_built_from_the_legacy_winner_and_its_nearest_neighbour():
    # Four candidates: the legacy winner is first in rank order, but the pool
    # must still span independent timing models rather than the winner's copies.
    discovered = [(f"p{i}", name(f"G{i}")) for i in range(4)]
    payloads = {k: srt(even(40, step_ms=2000 + i * 60)) for i, k in enumerate(discovered)}
    pool = build_shadow_pool(TARGET, discovered, payloads, limit=4)
    assert pool.independent_groups == 4
    assert pool.pool_size == 4


def test_pool_prefers_grid_diversity_over_rank_order():
    # Rank order would put both identical-grid copies first; diversification
    # must still surface the distinct timing models that exist.
    dup_a, dup_b = ("subdl", name("A1")), ("subdl", name("A2"))
    x, y = ("subsource", name("X")), ("opensub", name("Y"))
    payloads = {
        dup_a: srt(even(40)),
        dup_b: srt(even(40)),
        x: srt(even(40, step_ms=2200)),
        y: srt(even(40, step_ms=2400)),
    }
    pool = build_shadow_pool(TARGET, [dup_a, dup_b, x, y], payloads, limit=2)
    grids = {t for _p, _n, t in pool.references}
    assert len(grids) == 2
    assert pool.independent_groups == 2


def test_pool_reports_when_limited_by_available_payloads():
    pool = build_shadow_pool(TARGET, [("subdl", name("PiR8"))], {("subdl", name("PiR8")): srt(even(40))}, limit=4)
    assert pool.limited_by_available_payloads is True
    assert any("shadow_pool_limited_by_available_payloads" in r for r in pool.reasons)


def test_pool_adds_no_fetches_by_default():
    from app.config import settings

    assert settings.REFERENCE_SHADOW_POOL_LIMIT == 4
    # Extra downloads are opt-in; the default must not widen the bandwidth bill.
    assert settings.REFERENCE_SHADOW_POOL_FETCH_LIMIT == 0
    # And the pool limit is a separate concern from the alass candidate limit.
    assert settings.ALASS_CANDIDATE_LIMIT == 3
