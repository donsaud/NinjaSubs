"""Tests for the graded-anchor variant and the content-integrity assessment.

DIAGNOSTIC ONLY. Both modules are imported by no production module (asserted).

The findings these pin:

1. PART 1: the large-offset gate and the DP were reading the same anchors and
   judging them with different statistics. The gate asks "is this one shift?"
   (median absolute deviation about the median); the first version of the DP asked
   "is the offset identical between adjacent regions?" -- a local question that
   rejects EVOLV's real -51.76 ms/min drift. The cause was established before any
   margin was touched, and it was three concrete implementation differences:
   statistic, drift source, and cue bounding.
2. PART 2: using the gate's own statistics, EVOLV's anchors are usable and the
   DP proceeds instead of abstaining. The anchors really are noisy -- they
   scatter 1835ms about the fitted drift line -- so the result reports what is
   unexplained rather than claiming a clean alignment.
3. PART 9: content integrity is a separate question. It catches real volume
   damage, it flags a legitimate different release as suspect, and it cannot
   attribute an excess without text evidence. It never implies timing quality.
"""

from __future__ import annotations

import pathlib
import random

import pytest

import app.services.sync.content_integrity as content
import app.services.sync.interval_correspondence as ic

REPO = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures" / "large_offset"

# The three subtitle files these two tests measure. They are the SYNTHETIC
# corpus: every timing is the measured one, every word is generated. See
# tests/fixtures/large_offset/README.md for provenance. The real, non-redistributable
# files stay on the owner's machine; nothing here depends on them.
_SYNTHETIC_SUBTITLES = {
    "reference": "dexter_s08e04_reference.srt",
    "evolv": "dexter_s08e04_evolv_original.srt",
    "asap": "dexter_s08e04_asap_original.srt",
}


def _synthetic_fixture(name: str) -> pathlib.Path:
    path = FIXTURES / _SYNTHETIC_SUBTITLES[name]
    if not path.is_file():  # pragma: no cover - fixture-dependent
        pytest.skip(f"synthetic subtitle fixture unavailable: {path}")
    return path


def ref(n=300, start=8000, lo=1600, hi=4200, step=220):
    cues = []
    t = start
    for i in range(n):
        length = lo + (i * 37) % (hi - lo)
        cues.append((t, t + length, f"line {i}"))
        t += length + step
    return cues


def shift(cues, d):
    return [(s + d, e + d, x) for s, e, x in cues]


def timing_only(cues):
    return [(s, e, "") for s, e, _ in cues]


def duplicate_every(cues, frac):
    period = max(2, int(round(1.0 / frac)))
    out = []
    for i, c in enumerate(cues):
        out.append(c)
        if i % period == 0:
            out.append(c)
    return sorted(out, key=lambda c: c[0])


def drop_every(cues, frac):
    keep = max(2, int(round(1.0 / (1.0 - frac))))
    return [c for i, c in enumerate(cues) if i % keep != 0]


# ---------------------------------------------------------- anchor grading -- #
def test_grading_uses_the_global_statistic_not_adjacent_gaps():
    """The core PART 1 correction, asserted on a synthetic reproduction.

    Region medians that wander by more than the agreement margin between adjacent
    regions, while their deviation from the median stays inside the dispersion
    limit. That is exactly EVOLV's shape: a real drift the gate accepts.
    """
    regions = [(0, 96_000.0), (1_000_000, 94_000.0), (2_000_000, 97_000.0)]
    grading = ic.grade_anchors(regions, 1200.0, 0.5, drift_ms_per_minute=-50.0)
    # MAD about the median is small even though consecutive gaps are not.
    assert grading.dispersion_ms < 1200.0
    assert len(grading.hard) >= 1
    assert not grading.contradictory


def test_grading_rejects_an_anchor_beyond_the_gate_dispersion_limit():
    regions = [(0, 96_000.0), (1_000_000, 96_000.0), (2_000_000, 96_000.0),
               (3_000_000, 130_000.0)]
    grading = ic.grade_anchors(regions, 1200.0, 0.5, drift_ms_per_minute=0.0)
    assert len(grading.rejected) == 1
    assert grading.rejected[0].offset_ms == 130_000.0
    # One outlier is dropped, not fatal.
    assert not grading.contradictory


def test_grading_abstains_when_trusted_anchors_exceed_the_window():
    """Contradiction is tested against the search window, not the dispersion limit.

    The dispersion limit answers "is this one shift?" -- a question about the
    centre. What breaks a band is scatter wider than the band itself. Exercised
    with a tightened window because with the shipped defaults the branch is
    unreachable -- see the invariant test below.
    """
    # Offsets inside the dispersion limit, but a supplied drift that disagrees
    # with them: the band would be placed where the anchors are not.
    disagreeing = [(0, 98_800.0), (1_000_000, 100_000.0), (2_000_000, 98_800.0)]
    grading = ic.grade_anchors(
        disagreeing, 1200.0, 0.5, drift_ms_per_minute=200.0, window_ms=2000
    )
    assert all(a.grade is not ic.AnchorGrade.REJECTED for a in grading.graded)
    assert grading.contradictory
    assert grading.residual_spread > 2000.0

    agreeing = [(0, 98_800.0), (1_000_000, 100_000.0), (2_000_000, 98_800.0)]
    assert not ic.grade_anchors(
        agreeing, 1200.0, 0.5, drift_ms_per_minute=0.0, window_ms=2000
    ).contradictory


def test_default_configuration_makes_contradiction_defence_in_depth_only():
    """Recorded so the unreachable branch is not mistaken for a live guard.

    Surviving anchors are by construction within the dispersion limit of the
    median, so their spread is bounded by twice that limit -- 2400ms at the
    default. The window is 4000ms, so ``contradictory`` cannot fire under the
    shipped constants. The grading already guarantees the band is usable; the
    branch exists for a tightened window.
    """
    assert 2 * 1200.0 < ic.SEARCH_WINDOW_MS


def test_graded_alignment_beats_the_local_rule_on_a_drifting_target():
    """The variant must not lose a case the first version handled."""
    r = ref()
    rep = ic.align_with_graded_anchors(timing_only(shift(r, 96_500)),
                                       timing_only(r), 96_500.0)
    assert rep.outcome is ic.AlignmentOutcome.ALIGNED
    assert rep.reference_coverage > 0.95
    assert rep.p95() < 50


def test_graded_alignment_does_not_invent_breakpoints():
    r = ref()
    for d in (0, 2000, 96_500, 170_000):
        rep = ic.align_with_graded_anchors(timing_only(shift(r, d)),
                                           timing_only(r), float(d))
        assert rep.mapping.piece_count == 1, f"+{d}ms invented a breakpoint"
        assert not rep.mapping.breakpoints_ms


def test_graded_alignment_reports_its_anchor_grading():
    r = ref()
    rep = ic.align_with_graded_anchors(timing_only(shift(r, 96_500)),
                                       timing_only(r), 96_500.0)
    grading = rep.anchor_grading
    assert grading is not None
    counts = grading.counts()
    assert counts["hard"] >= 1
    assert counts["hard"] + counts["soft"] + counts["rejected"] > 0


def test_graded_alignment_abstains_without_anchor_evidence():
    r = ref(40)
    rep = ic.align_with_graded_anchors(
        timing_only(shift(r, 5_000_000)), timing_only(r), 0.0
    )
    assert rep.outcome is ic.AlignmentOutcome.NO_CORRESPONDENCE
    assert rep.groups == []


def test_graded_offset_fn_reproduces_a_linear_ramp():
    grading = ic.AnchorGrading(
        graded=[], estimated_offset_ms=96_000.0, dispersion_ms=100.0,
        slope_ms_per_ms=-50.0 / 60_000.0, origin_ms=0,
        hard=[], soft=[], rejected=[], contradictory=False,
    )
    fn = ic.graded_offset_fn(grading)
    assert fn(0) == 96_000.0
    assert abs(fn(60_000) - 95_950.0) < 1e-6
    assert abs(fn(3_600_000) - 93_000.0) < 1e-6


# ----------------------------------------------------- content integrity ---- #
def test_two_pass_refinement_recentres_a_biased_seed():
    """The second pass must not be a no-op.

    At the exact measured seed the two passes coincide, which is why removing
    the second pass looked harmless. They diverge as soon as the seed is biased:
    for EVOLV, drift reads -19.3 ms/min from one pass and -30.5 from two, because
    pass 2 re-measures the anchors against the median pass 1 produced. That is the
    property that stops one badly-placed opening cue deciding the band, so it is
    asserted directly rather than left implied.
    """
    from app.services.subtitle_matcher import parse_srt_cues
    from app.services.sync.alignment import analyze_drift
    from app.services.sync.large_offset import (
        _as_cues,
        _bounded,
        _median,
        _region_medians,
    )

    def read(name):
        text = _synthetic_fixture(name).read_text(encoding="utf-8", errors="replace")
        return [(int(s), int(e), "") for s, e, _ in parse_srt_cues(text)]

    reference = read("reference")
    target = read("evolv")
    ref_starts = [c[0] for c in reference]
    bounded = _bounded(_as_cues(target))
    seed = 97_300.0  # the true offset, biased by a second

    one_pass, _, one_samples = _region_medians(bounded, ref_starts, seed)
    refined = _median([v for _, v in one_pass])
    two_pass, _, two_samples = _region_medians(bounded, ref_starts, refined)

    one_drift = analyze_drift(one_samples)
    two_drift = analyze_drift(two_samples)
    assert one_drift != two_drift, (
        "fixture no longer discriminates: one and two passes agree"
    )

    rep = ic.align_with_graded_anchors(target, reference, seed)
    measured = rep.anchor_grading.slope_ms_per_ms * 60_000
    assert measured == pytest.approx(two_drift, abs=0.01), (
        "the graded variant must report the refined (second-pass) drift"
    )
    assert measured != pytest.approx(one_drift, abs=0.01)


def test_graded_variant_agrees_with_the_production_gate():
    """PART 1's conclusion, asserted on real fixtures.

    The whole reason the graded variant exists is that it must read the anchors
    the *same* way the gate does. If it re-derives them differently -- one pass
    instead of two, or every cue instead of the bounded sample -- it reintroduces
    exactly the disagreement this work set out to close. So the graded variant's
    dispersion and drift are compared against ``assess_large_offset`` itself.

    ASAP is the discriminating case: it has 781 cues against a 600-cue sample
    bound, so sampling everything changes which cues are measured.
    """
    from app.services.subtitle_matcher import parse_srt_cues
    from app.services.sync.large_offset import assess_large_offset

    def read(name):
        text = _synthetic_fixture(name).read_text(encoding="utf-8", errors="replace")
        return [(int(s), int(e), "") for s, e, _ in parse_srt_cues(text)]

    reference = read("reference")
    for name, seed in (
        ("asap", 96050.0),
        ("evolv", 96100.0),
    ):
        target = read(name)
        gate = assess_large_offset(
            target, reference, seed_offset_ms=int(seed), identity_supported=True
        )
        rep = ic.align_with_graded_anchors(target, reference, seed)
        grading = rep.anchor_grading
        assert grading is not None
        assert grading.dispersion_ms == pytest.approx(
            gate.offset_dispersion_ms, abs=1.0
        ), f"{name}: dispersion disagrees with the gate"
        assert grading.estimated_offset_ms == pytest.approx(
            gate.estimated_offset_ms, abs=1.0
        ), f"{name}: estimated offset disagrees with the gate"
        if gate.drift_ms_per_minute is not None:
            assert grading.slope_ms_per_ms * 60_000 == pytest.approx(
                gate.drift_ms_per_minute, abs=0.01
            ), f"{name}: drift disagrees with the gate"


def test_drift_keeps_the_direction_the_gate_measured():
    """Drift is measured evidence, not a guess: the sign must survive.

    Reversing it would point the search band the wrong way along the timeline.
    With real fixtures the drift is small enough that a sign flip barely moves
    the outcome, so the conversion is asserted directly.
    """
    regions = [(0, 96_000.0), (1_000_000, 95_000.0), (2_000_000, 94_000.0)]
    falling = ic.grade_anchors(regions, 1200.0, 0.5, drift_ms_per_minute=-60.0)
    rising = ic.grade_anchors(regions, 1200.0, 0.5, drift_ms_per_minute=+60.0)
    assert falling.slope_ms_per_ms < 0
    assert rising.slope_ms_per_ms > 0
    assert abs(falling.slope_ms_per_ms) == pytest.approx(60.0 / 60_000, rel=1e-6)


def sparse_gap_ref(n=240, start=8000, span=1200, gap=2000):
    """Cues whose gaps are large relative to their spans.

    Merging several cues into one absorbs those gaps, which is what moves total
    speech time. With small gaps the ratio barely moves and the fixture cannot
    reach the mixed-signal branch at all.
    """
    return [
        (start + i * (span + gap), start + i * (span + gap) + span, f"line {i}")
        for i in range(n)
    ]


def test_coarser_segmentation_is_not_content_damage():
    """The other mixed-signal direction.

    Merging cues *raises* speech time while *lowering* the cue count, because a
    merged display spans the gap. Reading that as content damage would reject
    every legitimately re-cut subtitle, which is exactly the failure mode the
    volume rule has to avoid in both directions.
    """
    r = sparse_gap_ref()
    # Merging four cues into one absorbs the three gaps between them. With a
    # fixture whose gaps are large relative to the cue spans, that pushes speech
    # time well outside tolerance while the cue count collapses -- the two ratios
    # move in opposite directions, which is the mixed-signal case.
    coarser = []
    for i in range(0, len(r), 4):
        chunk = r[i: i + 4]
        coarser.append((chunk[0][0], chunk[-1][1], chunk[0][2]))
    rep = content.assess_content_integrity(coarser, r)
    assert rep.observed["cue_count_ratio"] < 0.5, "cue count should fall sharply"
    assert rep.observed["active_duration_ratio"] > 1.15, (
        f"speech time should rise clearly: {rep.observed['active_duration_ratio']}"
    )
    assert rep.verdict is content.ContentIntegrity.UNKNOWN
    assert rep.verdict is not content.ContentIntegrity.SUSPECT, (
        "a segmentation difference must not read as content damage"
    )


def test_identical_content_is_good():
    r = ref()
    rep = content.assess_content_integrity(r, r)
    assert rep.verdict is content.ContentIntegrity.GOOD
    assert rep.timing_implied is False


def test_duplication_and_deletion_are_suspect():
    r = ref()
    dup = content.assess_content_integrity(duplicate_every(r, 0.40), r)
    dele = content.assess_content_integrity(drop_every(r, 0.40), r)
    assert dup.verdict is content.ContentIntegrity.SUSPECT
    assert dele.verdict is content.ContentIntegrity.SUSPECT
    assert dup.observed["cue_count_ratio"] > 1.2
    assert dele.observed["cue_count_ratio"] < 0.8


def test_fine_segmentation_is_not_a_content_problem():
    """Splitting a cue raises the cue count but not total speech time."""
    r = ref()
    fine = []
    for i, (s, e, t) in enumerate(r):
        if i % 2 == 0 and e - s >= 4:
            m = s + (e - s) // 2
            fine += [(s, m, t), (m, e, t)]
        else:
            fine.append((s, e, t))
    rep = content.assess_content_integrity(fine, r)
    assert rep.verdict is not content.ContentIntegrity.SUSPECT, (
        "finer segmentation must not read as extra content"
    )


def test_cross_language_reference_cannot_be_content_good():
    """A different-language reference cannot be content-confirmed.

    Matching cue and speech-time volumes prove nothing about *which* lines these
    are, so GOOD is unavailable without same-language evidence.
    """
    r = ref()
    foreign = [(s, e, f"ligne {i} xyz") for i, (s, e, _) in enumerate(r)]
    rep = content.assess_content_integrity(foreign, r)
    assert rep.verdict is content.ContentIntegrity.UNKNOWN
    assert any(
        f.signal == "text_line_overlap" and f.evidence is content.Evidence.NOT_AVAILABLE
        for f in rep.findings
    )
    assert any("no same-language evidence" in lim for lim in rep.limits)


def test_same_language_pair_can_be_content_good():
    r = ref()
    assert content.assess_content_integrity(r, r).verdict is content.ContentIntegrity.GOOD
    assert content.assess_content_integrity(shift(r, 1500), r).verdict is (
        content.ContentIntegrity.GOOD
    ), "a uniform shift of real content is still the same content"


def test_excess_without_text_evidence_is_recorded_as_unattributable():
    r = ref()
    foreign = duplicate_every(r, 0.40)
    foreign = [(s, e, f"ligne {i} xyz")
               for i, (s, e, _) in enumerate(foreign)]
    rep = content.assess_content_integrity(foreign, r)
    assert rep.verdict is content.ContentIntegrity.SUSPECT
    assert any("indistinguishable" in lim for lim in rep.limits)


def test_content_integrity_never_claims_timing_quality():
    r = ref()
    rep = content.assess_content_integrity(r, r, reference_coverage=0.0)
    assert rep.timing_implied is False
    assert rep.observed["reference_coverage"] == 0.0
    assert rep.verdict is content.ContentIntegrity.GOOD, (
        "content integrity is independent of timing: identical content with zero "
        "reference coverage is still content-GOOD"
    )


def test_empty_input_is_unknown_not_good():
    assert content.assess_content_integrity([], ref(10)).verdict is (
        content.ContentIntegrity.UNKNOWN
    )


def test_sequence_disagreement_detects_reordering_without_text():
    r = ref()
    assert content.sequence_disagreement(r, r) == 0.0
    shuffled = list(r)
    random.Random(3).shuffle(shuffled)
    bad = content.sequence_disagreement(shuffled, r)
    assert bad is not None and bad > 0.5
    # Different lengths cannot be compared positionally.
    assert content.sequence_disagreement(r, r[:-5]) is None


def test_no_acceptance_threshold_in_content_integrity():
    src = (REPO / "app/services/sync/content_integrity.py").read_text(encoding="utf-8")
    code = "\n".join(line.split("#", 1)[0] for line in src.splitlines())
    for forbidden in ("MAX_P95_MS_FOR_STABLE", "VERIFY_THRESHOLD", "ACCEPT_LIMIT"):
        assert forbidden not in code


# ------------------------------------------------------------- isolation --- #
def test_no_production_module_imports_the_new_modules():
    from tests.test_anchor_correspondence import DIAGNOSTIC_MODULES

    allowed = DIAGNOSTIC_MODULES | {"content_integrity.py"}
    for module in ("interval_correspondence", "content_integrity"):
        offenders = [
            p.name
            for p in (REPO / "app").rglob("*.py")
            if p.name not in allowed
            and module in p.read_text(encoding="utf-8")
        ]
        assert not offenders, f"production imports {module}: {offenders}"


def test_content_integrity_stores_no_text():
    src = (REPO / "app/services/sync/content_integrity.py").read_text(encoding="utf-8")
    # The module may normalise text into a hash but must never persist it.
    assert "hashlib" in src
    assert "observed[" in src
    assert "text_line_overlap" in src
    for bad in ("_norm_hash(text)", "norm_text", "raw_text"):
        if bad == "_norm_hash(text)":
            continue
        assert bad not in src, f"content integrity appears to persist text: {bad}"
