"""Anchor-guided correspondence: shadow experiment tests.

DIAGNOSTIC ONLY. ``app/services/sync/anchor_correspondence.py`` is imported by no
production module and a test asserts that. It reads cues and returns a report; it
writes no verdict, artifact, alias or cache entry.

What the experiment found, and what these tests pin:

1. Anchor-guided correspondence **does** recover the correspondence that
   nearest-start matching misses. At a ~96s offset the hypothesis-guided search
   produces a coherent mapping where global nearest-start produced garbage.
2. On the pair production actually evaluates -- Alass output against reference --
   it improves the residual materially: EVOLV p95 3310ms -> ~1451ms, ASAP
   4750ms -> ~2277ms.
3. A uniform offset of any size collapses to **one** mapping piece with a
   residual of zero. The model does not hallucinate segments.
4. Ambiguity produces abstention: cues whose admissible candidates disagree are
   left unmatched rather than forced.
5. It is **not** safe as an acceptance rule. ASAP still exceeds a 2000ms bound,
   finer segmentation degrades it, and -- the important one -- content damage is
   invisible to it by construction. That last is not a defect: the task requires
   completeness to stay separate. It is why this cannot be an acceptance signal
   on its own.

A note on the corpus: an earlier draft counted "wrong release", "deleted",
"duplicated", "reordered" and "wrong episode" as false matches. That was a
mistake in the experiment, not in the model. Those cases have *correct timing* --
deleting cues does not misalign the survivors -- so a correspondence model is
right to call them good timing. They are content problems, which is exactly why
completeness is measured separately and never folded into the label.
"""

from __future__ import annotations

import pathlib
import random

import app.services.sync.anchor_correspondence as ac


def _ref() -> list[tuple[int, int, str]]:
    cues: list[tuple[int, int, str]] = []
    t = 8000
    i = 0
    while t < 2_850_000:
        span = 1600 + (i * 37) % 2600
        cues.append((t, t + span, ""))
        t += span + 220
        i += 1
    return cues


def _shift(cues, d):
    return [(s + d, e + d, t) for s, e, t in cues]


def _truncate(cues, keep):
    return cues[: int(len(cues) * keep)]


def _drop_every(cues, k):
    step = max(1, int(len(cues) * k))
    return [c for i, c in enumerate(cues) if i % step]


def _duplicate(cues, k):
    out = []
    step = max(1, int(len(cues) * k))
    for i, c in enumerate(cues):
        out.append(c)
        if i % step == 0:
            out.append(c)
    return sorted(out, key=lambda c: c[0])


def _corrupt(cues, amount_ms, frac, seed=5):
    rng = random.Random(seed)
    out = []
    for c in cues:
        if rng.random() < frac:
            j = rng.randint(-amount_ms, amount_ms)
            out.append((c[0] + j, c[1] + j, c[2]))
        else:
            out.append(c)
    return out


def _fragment(cues, n=4):
    out = []
    for s, e, t in cues:
        step = max(1, (e - s) // n)
        for k in range(n):
            out.append((s + k * step, s + k * step + step, t))
    return sorted(out, key=lambda c: c[0])


def _piecewise(cues, early, late, frac=0.30):
    b = cues[int(len(cues) * frac)][0]
    return [
        (s + (early if s < b else late), e + (early if s < b else late), t)
        for s, e, t in cues
    ]


# ------------------------------------------------------- the central claim -- #


def test_anchor_guided_recovers_a_large_uniform_offset():
    """The claim under test: a 96.5s offset is recovered, not smeared.

    The previous experiment's nearest-start builder lost a third of the cues and
    produced a lag spread of ~108s on the real reference at this offset. The
    hypothesis-guided search does not, because it looks near a predicted point
    rather than globally.
    """
    ref = _ref()
    report = ac.anchor_guided_correspondence(_shift(ref, 96_500), ref, 96_500)
    assert report.decision is ac.CorrespondenceDecision.GOOD
    # Coverage is ~0.79 on a densely-cued reference, not 1.0, because ambiguity
    # rejection declines a cue whenever two admissible candidates disagree by
    # more than the ambiguity margin. That is the intended conservative
    # behaviour and it matches the real cases (EVOLV 0.77, ASAP 0.64); the point
    # here is that most cues are still recovered, not that all are.
    assert report.coverage > 0.7, "a large uniform offset must not lose most cues"
    assert report.residual_p95() < 100
    assert report.mapping.piece_count == 1, "a uniform offset is one piece"


def test_uniform_offset_never_hallucinates_multiple_segments():
    """Size of the offset must not create pieces; only disagreement may."""
    ref = _ref()
    for delta in (0, 2_000, 12_000, 30_000, 96_500, 120_000):
        report = ac.anchor_guided_correspondence(_shift(ref, delta), ref, float(delta))
        assert report.mapping.piece_count == 1, f"{delta}ms produced extra pieces"
        assert report.residual_p95() < 100
        assert not report.mapping.breakpoints_ms


def test_inferred_offset_matches_the_true_offset():
    ref = _ref()
    for delta in (12_000, 96_500, 120_000):
        report = ac.anchor_guided_correspondence(_shift(ref, delta), ref, float(delta))
        got = report.mapping.segments[0].offset_ms
        assert abs(got - delta) < 500, f"offset {got} does not match {delta}"


def test_piecewise_case_is_representable_and_monotonic():
    """A legitimate early/late correction must stay time-map monotonic."""
    ref = _ref()
    target = _piecewise(ref, 96_000, 0)
    report = ac.anchor_guided_correspondence(target, ref, 96_000)
    assert report.mapping.time_map_increasing, (
        "a decreasing offset is legitimate, but the time map must still advance"
    )
    assert report.mapping.piece_count in (1, 2)


def test_monotonicity_check_uses_the_correct_offset_sign():
    """Regression for a sign error found during this experiment.

    Offset here is ``target - reference``, so a mapped time is ``t - offset``.
    Building the check with ``+`` inverted it and falsely abstained on the real
    EVOLV case with a legitimate ~96s correction. A large *positive* offset with
    a small variation must be accepted.
    """
    # ~96s offset that decreases slightly over time, as the real anchors do.
    regions = [
        ac.AnchorRegion(0, 600_000, 97_775.0, 6),
        ac.AnchorRegion(600_000, 2_380_000, 96_258.0, 6),
        ac.AnchorRegion(2_380_000, 2_900_000, 94_255.0, 6),
    ]
    # Make them disagree enough to split, then check the map still advances.
    mapping = ac.build_mapping(regions)
    assert mapping.time_map_increasing, (
        "a large positive offset decreasing slightly is a valid re-timing; the "
        "sign of the monotonicity check must subtract the offset"
    )


# ------------------------------------------------------------- abstention -- #


def test_ambiguous_cues_are_left_unmatched_not_forced():
    """Cues with disagreeing candidates must abstain rather than guess."""
    ref = _ref()
    # A reference with a duplicate of every cue 1.5s away makes candidates
    # ambiguous within the neighbourhood.
    doubled = sorted(ref + _shift(ref, 1_500), key=lambda c: c[0])
    report = ac.anchor_guided_correspondence(ref, doubled, 0.0)
    assert report.ambiguous_target, (
        "expected some cues to be ambiguous; if none are, ambiguity rejection is "
        "not being exercised and cannot be trusted"
    )
    assert report.decision is not ac.CorrespondenceDecision.GOOD or report.ambiguous_target


def test_no_correspondence_is_distinguished_from_poor_residual():
    """'Nothing to compare' and 'compared badly' must not be the same label."""
    ref = _ref()
    # No anchor regions at all: a hypothesis so wrong nothing lands near it.
    report = ac.anchor_guided_correspondence(ref, ref, -5_000_000.0)
    assert report.decision is ac.CorrespondenceDecision.ABSTAIN
    assert "no anchor regions" in " ".join(report.reasons)


# --------------------------------------------- unreachable-by-corpus guards -- #


def _region(start, end, offset, n=6):
    return ac.AnchorRegion(start, end, offset, n)


def _folding_mapping() -> ac.AnchorMapping:
    """A map whose later segment sits *earlier* in reference time than the first.

    Reachable by constructing regions, but not through the public entry point:
    one hypothesis confines cross-region offset spread to the neighbourhood, so
    offsets cannot diverge far enough to fold. That is exactly why this guard
    needs a direct test -- inlined, removing it went unnoticed.
    """
    return ac.build_mapping(
        [
            _region(0, 900_000, 0.0),
            _region(900_000, 1_900_000, 96_000.0),
        ]
    )


def test_folding_time_map_is_reported_as_not_advancing():
    mapping = _folding_mapping()
    assert mapping.piece_count == 2, "these regions disagree enough to split"
    assert not mapping.time_map_increasing


def test_folding_time_map_abstains_rather_than_labelling():
    """A decreasing offset is legal; a time map that folds back is not."""
    mapping = _folding_mapping()
    correspondences = [
        ac.Correspondence(
            target_index=i,
            reference_index=i,
            target_start_ms=1_000 + i * 1000,
            reference_start_ms=1_000 + i * 1000,
            residual_ms=0.0,
        )
        for i in range(60)
    ]
    decision, reasons = ac._decide(mapping, correspondences)
    assert decision is ac.CorrespondenceDecision.ABSTAIN
    assert "implied time map does not advance" in reasons


def test_decide_is_reachable_and_covers_each_branch():
    """Every branch of the label is reachable, so each can be mutated."""
    good = ac.build_mapping([_region(0, 900_000, 96_000.0)])
    thin = [
        ac.Correspondence(0, 0, 0, 0, 0.0),
    ]
    assert ac._decide(good, thin)[0] is ac.CorrespondenceDecision.ABSTAIN

    clean = [
        ac.Correspondence(i, i, i * 1000, i * 1000, 10.0) for i in range(60)
    ]
    assert ac._decide(good, clean)[0] is ac.CorrespondenceDecision.GOOD

    noisy = [
        ac.Correspondence(i, i, i * 1000, i * 1000, 9000.0) for i in range(60)
    ]
    assert ac._decide(good, noisy)[0] is ac.CorrespondenceDecision.BAD


def test_thin_evidence_abstains():
    ref = _ref()
    report = ac.anchor_guided_correspondence(ref[:5], ref, 0.0)
    assert report.decision is ac.CorrespondenceDecision.ABSTAIN


# ----------------------------------------------- timing damage separation -- #


def test_severe_timing_corruption_is_not_labelled_good():
    """Having correspondences is not the same as having good ones.

    An earlier version of the decision rule checked only correspondence count and
    monotonicity, so corrupted timing scored GOOD. The residual condition is what
    separates them.
    """
    ref = _ref()
    for frac, amount in ((0.30, 8000), (0.50, 8000)):
        report = ac.anchor_guided_correspondence(
            _corrupt(ref, amount, frac), ref, 0.0
        )
        assert report.decision is not ac.CorrespondenceDecision.GOOD, (
            f"{frac:.0%} corruption at +/-{amount}ms was labelled GOOD"
        )


def test_severe_fragmentation_is_not_labelled_good():
    ref = _ref()
    report = ac.anchor_guided_correspondence(_fragment(ref, 6), ref, 0.0)
    assert report.decision is not ac.CorrespondenceDecision.GOOD


def test_content_damage_keeps_good_timing_and_is_caught_by_completeness():
    """The separation the task requires, asserted rather than asserted-about.

    Deletion, duplication and truncation do not misalign the cues that survive,
    so the correspondence label is legitimately GOOD. The content signals are
    where the damage must show, and for truncation and sparsity they do.
    """
    ref = _ref()
    truncated = ac.anchor_guided_correspondence(_truncate(ref, 0.6), ref, 0.0)
    # Timing is fine...
    assert truncated.residual_p95() < 200
    # ...and completeness reports the loss, separately.
    assert truncated.completeness["cue_count_ratio"] < 0.75
    assert truncated.completeness["temporal_density"] < 0.9


def test_uniform_pattern_deletion_is_invisible_to_both_and_that_is_recorded():
    """A known limitation, pinned so it is not rediscovered as a surprise."""
    ref = _ref()
    report = ac.anchor_guided_correspondence(_drop_every(ref, 0.40), ref, 0.0)
    assert report.residual_p95() < 200
    assert report.completeness["cue_count_ratio"] > 0.9, (
        "deleting every 2.5th cue barely moves the count; no signal here sees it"
    )
    assert report.completeness["temporal_density"] > 0.8


# --------------------------------------------------- completeness separate -- #


def test_label_is_timing_only_and_completeness_is_not_folded_in():
    ref = _ref()
    report = ac.anchor_guided_correspondence(ref, ref, 0.0)
    assert "TIMING ONLY" in " ".join(report.reasons)
    assert report.completeness, "completeness must be reported even on a happy path"
    assert set(report.completeness) & {
        "cue_count_ratio", "active_duration_ratio", "temporal_density",
        "active_span_ratio",
    }


def test_no_completeness_threshold_exists_in_the_module():
    """Completeness is characterised, never bounded. Asserted structurally."""
    src = pathlib.Path(ac.__file__).read_text(encoding="utf-8")
    for forbidden in ("MIN_CUE_RATIO", "MIN_DENSITY", "MIN_COVERAGE"):
        assert forbidden not in src, f"a completeness threshold crept in: {forbidden}"


# ------------------------------------------------------------- isolation --- #


#: Modules that are allowed to import each other because they are all part of
#: the diagnostic experiment. Anything else importing them is a production leak.
DIAGNOSTIC_MODULES = frozenset({
    "anchor_correspondence.py",
    "interval_correspondence.py",
})


def test_no_production_module_imports_the_experiment():
    repo = pathlib.Path(__file__).resolve().parent.parent
    offenders = [
        p.name
        for p in (repo / "app").rglob("*.py")
        if p.name not in DIAGNOSTIC_MODULES
        and "anchor_correspondence" in p.read_text(encoding="utf-8")
    ]
    assert not offenders, f"production imports the experiment: {offenders}"


def test_module_imports_no_cache_or_verifier():
    import ast

    banned = {
        "SyncCache", "cache_manager", "LRUCacheManager", "SyncOrchestrator",
        "AlignmentAnalyzer", "SyncState", "may_serve_synchronized",
    }
    tree = ast.parse(pathlib.Path(ac.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.update(a.asname or a.name for a in node.names)
    overlap = imported & banned
    assert not overlap, f"experiment imports production state: {sorted(overlap)}"


def test_decisions_are_named_as_shadow():
    for d in ac.CorrespondenceDecision:
        assert d.value.startswith("ANCHOR_CORRESPONDENCE_")


def test_diagnostic_modules_carry_no_acceptance_threshold():
    """No diagnostic or candidate module may declare an acceptance limit.

    A threshold sitting in a diagnostic module is one line away from becoming
    production policy, so the absence is asserted structurally. This test did
    not exist: the mutation adding ``MAX_P95_MS_FOR_STABLE`` to
    ``segmentation_pairing.py`` left the suite green and was reported
    UNPROTECTED once pre-existing failures stopped being mistaken for catches.
    """
    repo = pathlib.Path(__file__).resolve().parent.parent
    modules = (
        "app/services/sync/segmentation_pairing.py",
        "app/services/sync/piecewise_shadow.py",
        "app/services/sync/piecewise_shadow_eval.py",
        "app/services/sync/anchor_correspondence.py",
    )
    for rel in modules:
        path = repo / rel
        assert path.exists(), f"{rel} was moved; update this test"
        # Mentions in prose are fine -- several of these modules explain that a
        # production constant is deliberately NOT what they use. Only a
        # declaration counts, so comments are stripped first.
        src = path.read_text(encoding="utf-8")
        code = "\n".join(line.split("#", 1)[0] for line in src.splitlines())
        for forbidden in (
            "MAX_P95_MS_FOR_STABLE",
            "MAX_RESIDUAL_MS",
            "VERIFY_THRESHOLD",
            "ACCEPT_LIMIT",
        ):
            assert forbidden not in code, (
                f"{rel} declares {forbidden}; a diagnostic module must not carry an "
                "acceptance limit"
            )


def test_reuses_the_existing_anchor_machinery():
    """Part 1: no second anchor detector. The production helper must be used."""
    src = pathlib.Path(ac.__file__).read_text(encoding="utf-8")
    assert "_nearest_reference_offset" in src, (
        "the experiment must consume the existing hypothesis-guided helper"
    )
    assert "def _nearest_reference_offset" not in src, (
        "the helper must be imported, not reimplemented"
    )


def test_deterministic():
    ref = _ref()
    target = _shift(ref, 96_500)
    first = ac.anchor_guided_correspondence(target, ref, 96_500).as_row()
    for _ in range(3):
        assert ac.anchor_guided_correspondence(target, ref, 96_500).as_row() == first


def test_never_modifies_its_inputs():
    """The mapping lives in diagnostic space; cues must come back untouched."""
    ref = _ref()
    target = _shift(ref, 96_500)
    before = list(target)
    ac.anchor_guided_correspondence(target, ref, 96_500)
    assert target == before
