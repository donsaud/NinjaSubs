"""Interval correspondence: the experiment's findings, pinned as tests.

DIAGNOSTIC ONLY. ``app/services/sync/interval_correspondence.py`` is imported by
no production module and a test asserts that.

What the experiment measured, and what these tests hold in place:

1. The reason ASAP stayed at p95 ~2277ms was never misalignment. Its per-quarter
   p50 was 77-307ms while p95 was 2067-2357ms in *all four* quarters, because the
   subtitle carries ~28% more cues than the reference at the same per-cue
   duration. A one-to-one matcher cannot express that; it force-pairs the
   surplus.
2. A split/merge-aware model expresses it: 193 of those surplus cues become
   SPLIT groups and the residual drops to ~527ms with zero forced matches.
3. Segmentation variation stops mattering. Finer segmentation at a +96.5s offset
   -- the case that made the previous model collapse to p95 2422ms / coverage
   0.33 -- now yields reference coverage 1.000 and residual 0.
4. Damage is exposed rather than absorbed, with one honest asymmetry: deletion
   and truncation show up in the timing outcome, duplication does not (an
   adjacent duplicate is a legal SPLIT) and is only visible through total speech
   time.
5. Three cases are undetectable by any timing method and are recorded as limits
   rather than papered over: a wrong episode with identical timing, reordered
   text, and a small uniform offset that is simply a valid relationship.
"""

from __future__ import annotations

import pathlib
import random

import pytest

import app.services.sync.interval_correspondence as ic

REPO = pathlib.Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ fixtures #
def ref(n=420, start=8000, span_lo=1600, span_hi=4200, step=220):
    cues = []
    t = start
    for i in range(n):
        length = span_lo + (i * 37) % (span_hi - span_lo)
        cues.append((t, t + length, ""))
        t += length + step
    return cues


def shift(cues, d):
    return [(s + d, e + d, t) for s, e, t in cues]


def split_every(cues, n=2):
    out = []
    for i, (s, e, t) in enumerate(cues):
        if i % n == 0 and e - s >= 4:
            m = s + (e - s) // 2
            out += [(s, m, t), (m, e, t)]
        else:
            out.append((s, e, t))
    return out


def merge_pairs(cues):
    out = []
    for i in range(0, len(cues), 2):
        if i + 1 < len(cues):
            gap = max(0, cues[i + 1][0] - cues[i][1])
            out.append((cues[i][0], max(cues[i][1], cues[i + 1][1] - gap), ""))
        else:
            out.append(cues[i])
    return out


def drop_every(cues, frac):
    keep_every = max(2, int(round(1.0 / (1.0 - frac))))
    return [c for i, c in enumerate(cues) if i % keep_every != 0]


def duplicate_every(cues, frac):
    period = max(2, int(round(1.0 / frac)))
    out = []
    for i, c in enumerate(cues):
        out.append(c)
        if i % period == 0:
            out.append(c)
    return sorted(out, key=lambda c: c[0])


def corrupt(cues, amount, frac, seed=5):
    rng = random.Random(seed)
    return [
        (s + (j if rng.random() < frac else 0),
         e + (j if rng.random() < frac else 0), t)
        for s, e, t in cues
        for j in ([rng.randint(-amount, amount)] if True else [0])
    ]


def run(target, reference, hypothesis=0.0):
    return ic.align_intervals(target, reference, hypothesis)


# ---------------------------------------------------- the central claim ---- #
def test_uniform_offset_of_any_size_is_one_piece_with_zero_residual():
    """Offset size must never create a segment."""
    r = ref()
    for d in (0, 2000, 12000, 30000, 60000, 96500, 120000, 170000):
        rep = run(shift(r, d), r, float(d))
        assert rep.mapping.piece_count == 1, f"+{d}ms produced extra pieces"
        assert not rep.mapping.breakpoints_ms
        assert rep.p95() < 1.0, f"+{d}ms residual {rep.p95()}"
        assert rep.outcome is ic.AlignmentOutcome.ALIGNED
        assert rep.reference_coverage == 1.0
        assert rep.target_coverage == 1.0


def test_finer_segmentation_at_large_offset_does_not_collapse():
    """The previous model's fatal weakness.

    It reached p95 2422ms and coverage 0.33 here. Segmentation is a legal
    relationship and must not degrade the timing.
    """
    r = ref()
    fine = shift(split_every(r, 2), 96500)
    rep = run(fine, r, 96500.0)
    assert rep.outcome is ic.AlignmentOutcome.ALIGNED
    assert rep.reference_coverage > 0.95
    assert rep.target_coverage > 0.95
    assert rep.p95() < 200
    assert rep.mapping.piece_count == 1
    assert rep.split_count() > 100, "the extra cues should be SPLITs, not surplus"


def test_coarser_segmentation_is_accepted_as_merges():
    r = ref()
    rep = run(merge_pairs(r), r)
    assert rep.merge_count() > 0, "merged cues should be MERGE groups"
    assert rep.reference_coverage > 0.5
    assert rep.completeness["active_duration_ratio"] == pytest.approx(1.0, abs=0.1), (
        "coarser segmentation must not look like content duplication"
    )


def test_cue_count_ratio_near_one_is_not_required():
    """Finer segmentation changes cue count a lot and timing stays perfect."""
    r = ref()
    rep = run(split_every(r, 3), r)
    assert rep.completeness["cue_count_ratio"] > 1.2
    assert rep.outcome is ic.AlignmentOutcome.ALIGNED
    assert rep.p95() < 1.0
    assert rep.content_observation is ic.ContentObservation.CONTENT_COMPARABLE


# ------------------------------------------------- surplus is not forced --- #
def test_surplus_target_cues_become_splits_not_forced_matches():
    """A denser target must not be explained by inventing correspondences."""
    r = ref()
    rep = run(split_every(r, 2), r)
    assert rep.surplus_target == [], "extra cues should be absorbed as splits"
    assert rep.matched_target == len(split_every(r, 2))


def test_a_target_far_from_its_hypothesis_is_not_claimed_as_aligned():
    """Wrong release, wrong hypothesis: must refuse, not force."""
    r = ref()
    rep = run(shift(r, 96_500), r, 0.0)
    assert rep.outcome is not ic.AlignmentOutcome.ALIGNED
    assert rep.reference_coverage < 0.9


def test_correct_hypothesis_recovers_the_same_target():
    r = ref()
    rep = run(shift(r, 96_500), r, 96_500.0)
    assert rep.outcome is ic.AlignmentOutcome.ALIGNED
    assert rep.reference_coverage == 1.0


# ------------------------------------------------------------- damage ------ #
def test_deletion_is_exposed_not_absorbed():
    r = ref()
    for frac in (0.20, 0.40, 0.60):
        rep = run(drop_every(r, frac), r)
        assert rep.outcome is ic.AlignmentOutcome.LOW_REFERENCE_COVERAGE, (
            f"deleting {frac:.0%} was reported as {rep.outcome.value}"
        )
        assert rep.content_observation is ic.ContentObservation.TARGET_CONTENT_LOSS


def test_truncation_is_reported_as_surplus_reference():
    r = ref()
    rep = run(r[: int(len(r) * 0.6)], r)
    assert rep.outcome is ic.AlignmentOutcome.SURPLUS_REFERENCE
    assert rep.reference_coverage < 0.8
    assert rep.content_observation is ic.ContentObservation.TARGET_CONTENT_LOSS


def test_sparse_subtitle_is_not_confident():
    r = ref()
    sparse = r[::4]
    rep = run(sparse, r)
    assert rep.outcome is not ic.AlignmentOutcome.ALIGNED
    assert rep.reference_coverage < 0.5


def test_duplication_is_invisible_to_timing_but_visible_to_speech_time():
    """The honest asymmetry, pinned so it is not rediscovered as a surprise.

    An adjacent duplicate of a cue is a legal SPLIT, so the timing outcome is
    ALIGNED with zero residual. Only total speech time separates duplication
    from a legitimate finer segmentation, because splitting preserves total time
    and duplicating adds to it.
    """
    r = ref()
    dup = run(duplicate_every(r, 0.40), r)
    assert dup.outcome is ic.AlignmentOutcome.ALIGNED, (
        "timing cannot see duplication; this documents the limit"
    )
    assert dup.p95() < 1.0
    assert dup.content_observation is ic.ContentObservation.TARGET_CONTENT_EXCESS
    assert dup.completeness["active_duration_ratio"] > 1.3

    fine = run(split_every(r, 2), r)
    assert fine.content_observation is ic.ContentObservation.CONTENT_COMPARABLE
    assert fine.completeness["cue_count_ratio"] > 1.4, (
        "half the cues split must clearly raise the cue count"
    )


def test_heavy_corruption_is_not_labelled_aligned():
    r = ref()
    rep = run(corrupt(r, 8000, 0.30), r)
    assert rep.outcome is not ic.AlignmentOutcome.ALIGNED


def test_small_corruption_is_a_recorded_limit():
    """10-20% corruption stays inside the band. Recorded, not hidden."""
    r = ref()
    for frac in (0.10, 0.20):
        rep = run(corrupt(r, 3000, frac), r)
        assert rep.outcome is ic.AlignmentOutcome.ALIGNED, (
            "10-20%% jitter is not detected; see docs/interval_correspondence.md"
        )


def test_a_wrong_episode_with_identical_timing_cannot_be_detected():
    """A structural limit of every timing method, asserted so it stays honest."""
    r = ref()
    rep = run(shift(r, 1200), r, 1200.0)
    assert rep.outcome is ic.AlignmentOutcome.ALIGNED
    # Detection of this case is the job of identity evidence, not correspondence.
    assert rep.content_observation is ic.ContentObservation.CONTENT_COMPARABLE


def test_reordering_text_is_invisible_by_design():
    """Text is never used, so content order cannot be checked here."""
    r = ref(n=60)
    texts = [f"line-{i}" for i in range(len(r))]
    random.Random(3).shuffle(texts)
    shuffled = [(s, e, texts[i]) for i, (s, e, _) in enumerate(r)]
    rep = run(shuffled, ref(n=60))
    assert rep.outcome is ic.AlignmentOutcome.ALIGNED


# ------------------------------------------------------------ structure ---- #
def test_groups_are_monotonic_and_reference_cues_are_consumed_once():
    """The alignment must never reuse a reference cue or move backwards."""
    r = ref()
    rep = run(shift(split_every(r, 2), 40000), r, 40000.0)
    seen_ref: set[int] = set()
    prev_t = -1
    prev_r = -1
    for g in rep.groups:
        assert g.target_indices[0] > prev_t, "target order went backwards"
        assert g.reference_indices[0] > prev_r, "reference order went backwards"
        for ri in g.reference_indices:
            assert ri not in seen_ref, "a reference cue was matched twice"
            seen_ref.add(ri)
        prev_t = g.target_indices[-1]
        prev_r = g.reference_indices[-1]


def test_correspondence_ceiling_rejects_a_displaced_cue():
    """A cue pushed beyond the ceiling is left unmatched, not paired.

    ``MAX_GROUP_CENTER_MS`` is inside ``SEARCH_WINDOW_MS`` on purpose: the window
    is how far the model is willing to *look*, and the ceiling is how far it is
    willing to *believe*. Without the ceiling a cue displaced by 3000ms would be
    paired against the window alone.
    """
    r = ref()
    assert ic.MAX_GROUP_CENTER_MS < ic.SEARCH_WINDOW_MS, (
        "the ceiling must be tighter than the window or the window is the only bound"
    )
    nudged = list(shift(r, 20_000))
    for idx in (100, 200, 300):
        s, e, t = nudged[idx]
        nudged[idx] = (s + 3000, e + 3000, t)
    rep = run(nudged, r, 20_000.0)
    for g in rep.groups:
        assert abs(g.center_delta_ms) <= ic.MAX_GROUP_CENTER_MS
    assert len(rep.groups) < len(r), "the displaced cues should not all be paired"


def test_group_size_cap_is_enforced_not_merely_hoped_for():
    """Four reference cues collapsed into one target cue must not be explained.

    ``MAX_GROUP`` is 3, so a 4-to-1 collapse is not a correspondence the model
    is allowed to claim. Unlimited grouping would absorb it and report a clean
    alignment, which is how duplication and deletion get mistaken for
    segmentation.
    """
    r = ref()
    collapsed = []
    for i in range(0, len(r), 4):
        chunk = r[i: i + 4]
        gap = max(0, chunk[1][0] - chunk[0][1]) if len(chunk) > 1 else 0
        end = chunk[-1][1] - (gap if len(chunk) > 1 else 0)
        collapsed.append((chunk[0][0], end, ""))
    rep = run(collapsed, r)
    for g in rep.groups:
        assert len(g.reference_indices) <= ic.MAX_GROUP
    assert rep.reference_coverage < 0.9, (
        "a 4-to-1 collapse exceeds MAX_GROUP and must not be fully explained"
    )


def test_overlap_requirement_rejects_near_but_disjoint_spans():
    """Proximity alone must not justify a group.

    Cues 1000ms long, 4000ms apart, with the target pinned 1500ms later by an
    explicitly zero-offset mapping: each target cue sits within the centre
    ceiling of a reference cue but overlaps none of them. The overlap rule is the
    only thing that can reject these -- the centre ceiling alone accepts them.

    The mapping is pinned on purpose. Letting the anchors discover the real 1500ms
    offset would align everything perfectly, which is the correct behaviour and
    would test nothing about the overlap rule.
    """
    from app.services.sync.anchor_correspondence import AnchorMapping, MappingSegment

    short = [(8000 + i * 4000, 8000 + i * 4000 + 1000, "") for i in range(60)]
    shifted = [(s + 1500, e + 1500, t) for s, e, t in short]
    zero = AnchorMapping(
        segments=[MappingSegment(0, 10_000_000, 0.0)],
        regions=[],
        anchor_samples=6,
        time_map_increasing=True,
    )
    rep = ic._align(shifted, short, zero, ic.SEARCH_WINDOW_MS, 1500.0)
    assert rep.reference_coverage < 0.9, (
        "disjoint single-cue pairs must be rejected; coverage "
        f"{rep.reference_coverage:.3f} means they were accepted"
    )


def test_group_sizes_are_bounded():
    r = ref()
    rep = run(shift(split_every(r, 2), 20000), r, 20000.0)
    for g in rep.groups:
        assert len(g.target_indices) <= ic.MAX_GROUP
        assert len(g.reference_indices) <= ic.MAX_GROUP
    spans = [
        r[g.reference_indices[-1]][1] - r[g.reference_indices[0]][0] for g in rep.groups
    ]
    assert max(spans) <= ic.GROUP_SPAN_MS


def test_inconsistent_anchors_abstain_rather_than_propagate():
    """Anchors that disagree must stop the search, not bend it silently.

    The disagreement has to survive the noise filter first: a lone outlier is
    folded into the agreed body, so the offsets here differ by less than
    ``REGION_DEVIATION_MS`` but more than the agreement margin.
    """
    from app.services.sync.anchor_correspondence import AnchorRegion, build_mapping

    mapping = build_mapping(
        [
            AnchorRegion(0, 900_000, 96_000.0, 6),
            AnchorRegion(900_000, 1_900_000, 89_000.0, 6),
        ]
    )
    assert mapping.piece_count == 2, "these offsets should not be folded as noise"
    ok, bad = ic.anchors_consistent(mapping, 1500.0)
    assert not ok
    assert bad == 1


def test_consistent_anchors_are_recognised():
    from app.services.sync.anchor_correspondence import AnchorRegion, build_mapping

    mapping = build_mapping(
        [
            AnchorRegion(0, 900_000, 96_000.0, 6),
            AnchorRegion(900_000, 1_900_000, 96_100.0, 6),
        ]
    )
    ok, bad = ic.anchors_consistent(mapping, 1500.0)
    assert ok
    assert bad == 0


def test_inconsistent_anchors_short_circuit_the_alignment():
    """The gate lives at the DP entry, so no path can bypass it."""
    from app.services.sync.anchor_correspondence import AnchorRegion, build_mapping

    r = ref()
    bad_mapping = build_mapping(
        [
            AnchorRegion(0, 900_000, 96_000.0, 6),
            AnchorRegion(900_000, 1_900_000, 89_000.0, 6),
        ]
    )
    rep = ic._align(r, r, bad_mapping, ic.SEARCH_WINDOW_MS, 1500.0)
    assert rep.outcome is ic.AlignmentOutcome.ANCHORS_INCONSISTENT
    assert rep.groups == []
    assert rep.inconsistent_regions == 1


def test_no_anchor_evidence_means_no_search():
    r = ref()
    rep = ic.align_intervals(shift(r, 5_000_000), r, 0.0)
    assert rep.outcome is ic.AlignmentOutcome.NO_CORRESPONDENCE
    assert rep.groups == []


def test_empty_input_is_handled():
    rep = ic.align_intervals([], ref(10), 0.0)
    assert rep.outcome is ic.AlignmentOutcome.NO_CORRESPONDENCE


# --------------------------------------------------------- completeness ---- #
def test_completeness_is_observed_and_separate():
    r = ref()
    rep = run(r[: int(len(r) * 0.6)], r)
    keys = set(rep.completeness)
    assert {
        "cue_count_ratio",
        "active_duration_ratio",
        "reference_coverage",
        "target_coverage",
        "matched_reference_span_ratio",
        "first_cue_delta_ms",
        "last_cue_delta_ms",
    } <= keys
    # Content observation is reported next to the timing outcome, never inside it.
    assert isinstance(rep.content_observation, ic.ContentObservation)
    assert rep.content["split_groups"] == rep.split_count()


def test_completeness_carries_no_acceptance_threshold():
    src = (REPO / "app/services/sync/interval_correspondence.py").read_text(
        encoding="utf-8"
    )
    code = "\n".join(line.split("#", 1)[0] for line in src.splitlines())
    for forbidden in (
        "MAX_P95_MS_FOR_STABLE",
        "MIN_CUE_RATIO",
        "MIN_ACTIVE_RATIO",
        "VERIFY_THRESHOLD",
    ):
        assert forbidden not in code, f"a completeness threshold crept in: {forbidden}"


def test_content_and_timing_outcomes_are_distinct_fields():
    """Kept separate on purpose, so neither can stand in for the other."""
    r = ref()
    rep = run(duplicate_every(r, 0.40), r)
    assert rep.outcome is ic.AlignmentOutcome.ALIGNED
    assert rep.content_observation is ic.ContentObservation.TARGET_CONTENT_EXCESS
    assert rep.as_row()["outcome"] != rep.as_row()["content_observation"]


# ------------------------------------------------------------ isolation ---- #
def test_no_production_module_imports_the_experiment():
    from tests.test_anchor_correspondence import DIAGNOSTIC_MODULES

    offenders = [
        p.name
        for p in (REPO / "app").rglob("*.py")
        if p.name not in DIAGNOSTIC_MODULES
        and "interval_correspondence" in p.read_text(encoding="utf-8")
    ]
    assert not offenders, f"production imports the experiment: {offenders}"


def test_module_never_touches_production_state():
    import ast

    banned = {
        "SyncCache", "cache_manager", "LRUCacheManager", "SyncOrchestrator",
        "AlignmentAnalyzer", "SyncState", "may_serve_synchronized",
    }
    tree = ast.parse(
        (REPO / "app/services/sync/interval_correspondence.py").read_text(
            encoding="utf-8"
        )
    )
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.update(a.asname or a.name for a in node.names)
    assert not (imported & banned), f"imports production state: {sorted(imported & banned)}"


def test_subtitle_text_is_never_used():
    """The model must be timing-only. Text is ignored entirely."""
    r = ref(n=40)
    plain = run(r, r)
    scrambled = run([(s, e, f"x{i}") for i, (s, e, _) in enumerate(r)], r)
    assert plain.as_row() == scrambled.as_row()


def test_does_not_modify_inputs():
    r = ref()
    t = shift(r, 96500)
    before = list(t)
    ic.align_intervals(t, r, 96500.0)
    assert t == before


def test_deterministic():
    r = ref()
    t = shift(split_every(r, 2), 96500)
    first = run(t, r, 96500.0).as_row()
    for _ in range(2):
        assert run(t, r, 96500.0).as_row() == first


def test_search_window_is_bounded():
    """The window is the anchor bound; it must stay modest."""
    assert ic.SEARCH_WINDOW_MS <= 6000
    assert ic.MAX_GROUP_CENTER_MS <= ic.SEARCH_WINDOW_MS
    assert ic.MAX_GROUP <= 3
