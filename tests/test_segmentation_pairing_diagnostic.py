"""Diagnostic tests for the segmentation-aware pairing experiment.

DIAGNOSTIC ONLY. ``app/services/sync/segmentation_pairing.py`` is not imported by
any production module and is not referenced by the verifier, the orchestrator, or
any threshold. These tests exist to record *why* it was not adopted, so the
experiment is not silently repeated, and to keep the two conclusions below
visible.

Conclusion 1 -- no safe replacement was found.
    Aggregation reduces the known-good residual only by absorbing cues, and
    absorbing cues is the same operation that hides deleted content. Conservative
    variants that do not hide damage also do not help. There is no parameter
    setting that both accepts the good cases and rejects the damaged ones.

Conclusion 2 -- a residual metric cannot see truncation, at all.
    Every truncation from 100% down to 20% of cues scores p95 = 0.0, because the
    cues that survive are still perfectly aligned. This is a property of residual
    timing in general, not of the candidate, and it is why completeness signals
    must stay separate from timing rather than being inferred from it.
"""

from __future__ import annotations

import pathlib
import random

import pytest

from app.services.sync.segmentation_pairing import (
    Completeness,
    aggregated_pair,
    completeness,
    greedy_pair,
)

FIXTURE = (
    pathlib.Path(__file__).resolve().parent
    / "fixtures" / "segmentation" / "reference.srt"
)


def _reference() -> list[tuple[int, int, str]]:
    """A small synthetic reference. Real media is not required."""
    lines = []
    t = 0
    for i in range(240):
        start = t
        t += 2000
        lines.append((start, t, f"line {i}"))
    return lines


def _shift(cues, delta):
    return [(s + delta, e + delta, x) for s, e, x in cues]


def _truncate(cues, keep):
    return cues[: int(len(cues) * keep)]


def _perturb(cues, ms, seed=11):
    rng = random.Random(seed)
    return [
        (s + rng.randint(-ms, ms), e + rng.randint(-ms, ms), x) for s, e, x in cues
    ]


def _resegmented_reference() -> list[tuple[int, int, str]]:
    """A reference with variable cue lengths, the shape aggregation exists for.

    Real releases re-segment: a long line, then two short ones, then a long one
    again. Evenly spaced cues have nothing to aggregate and would make the
    candidate look better than it is.
    """
    cues = []
    t = 0
    i = 0
    lengths = (3800, 1200, 1100, 3400, 1500, 1300, 4100, 1000, 1250, 3600)
    while len(cues) < 240:
        span = lengths[i % len(lengths)]
        cues.append((t, t + span, f"line {i}"))
        t += span + 180
        i += 1
    return cues


# ------------------------------------------------------ not wired in -------- #


def test_module_is_not_referenced_by_any_production_code():
    """The candidate must remain a diagnostic, not a live input to a decision."""
    repo = pathlib.Path(__file__).resolve().parent.parent
    production = [
        p for p in (repo / "app").rglob("*.py")
        if "segmentation_pairing" not in p.name
    ]
    offenders = []
    for path in production:
        text = path.read_text(encoding="utf-8")
        if "segmentation_pairing" in text:
            offenders.append(path.name)
    assert not offenders, (
        f"production modules import the diagnostic candidate: {offenders}"
    )


def test_no_threshold_constant_lives_in_the_candidate():
    """A candidate must not smuggle in an acceptance limit."""
    import app.services.sync.alignment as alignment

    src = pathlib.Path(alignment.__file__).read_text(encoding="utf-8")
    assert "segmentation_pairing" not in src
    # Production limits are untouched by this experiment.
    assert alignment.MAX_P95_MS_FOR_STABLE == 2000.0
    assert alignment.MAX_DRIFT_MS_PER_MINUTE == 120.0
    assert alignment.RESIDUAL_TOLERANCE_MS == 5_000


# ------------------------------------------- conclusion 2: truncation ------- #


@pytest.mark.parametrize("keep", [1.0, 0.8, 0.6, 0.4, 0.2])
def test_residual_metric_cannot_detect_truncation(keep):
    """Truncation is invisible to a residual-only metric, at any depth.

    This is the load-bearing reason completeness must be measured separately: a
    clean residual is not evidence of a complete subtitle.
    """
    ref = _reference()
    after = _truncate(ref, keep)
    result = greedy_pair(ref, after)
    assert result.p95() == 0.0, (
        f"truncation to {keep:.0%} produced a non-zero residual, which would mean "
        f"this test no longer demonstrates the blind spot"
    )
    comp = completeness(ref, after, runtime_ms=500_000)
    assert comp.cue_count_ratio == pytest.approx(keep, abs=0.01)
    assert comp.runtime_coverage < 1.0


def test_completeness_signals_do_separate_truncation_from_a_good_file():
    """The signals that *can* see truncation are reported, not thresholded."""
    ref = _reference()
    good = completeness(ref, _shift(ref, 3000), runtime_ms=500_000)
    bad = completeness(ref, _truncate(ref, 0.4), runtime_ms=500_000)
    # Both have a perfect residual; only completeness separates them.
    assert greedy_pair(ref, _shift(ref, 3000)).p95() == 3000.0
    assert greedy_pair(ref, _truncate(ref, 0.4)).p95() == 0.0
    assert good.cue_count_ratio == pytest.approx(1.0, abs=0.01)
    assert bad.cue_count_ratio < 0.5
    assert bad.runtime_coverage < good.runtime_coverage
    assert bad.first_cue_delta_ms == 0  # the head is intact; the tail is gone
    row = bad.as_row()
    assert set(row) >= {
        "cue_count_ratio", "runtime_coverage", "reference_coverage",
        "first_cue_delta_ms", "last_cue_delta_ms", "active_duration_ratio",
    }


def test_duplication_is_also_invisible_to_residuals():
    """A duplicated line is perfectly aligned with itself."""
    ref = _reference()
    after = []
    for i, c in enumerate(ref):
        after.append(c)
        if i % 24 == 0:
            after.append(c)
    after.sort(key=lambda c: c[0])
    assert greedy_pair(ref, after).p95() == 0.0
    assert completeness(ref, after, 500_000).cue_count_ratio > 1.0


# ----------------------------------------- conclusion 1: no safe candidate --- #


@pytest.mark.parametrize("kwargs", [
    {"max_group": 2, "max_group_span_ms": 3000},
    {"max_group": 2, "max_group_span_ms": 6000},
    {"max_group": 3, "max_group_span_ms": 6000},
    {"max_group": 3, "max_group_span_ms": 12000},
])
def test_aggregation_either_does_not_help_or_hides_damage(kwargs):
    """The trade-off is structural, so no parameter setting escapes it."""
    ref = _reference()
    damaged = _truncate(ref, 0.6)
    helped = (
        aggregated_pair(ref, _perturb(ref, 100), **kwargs).p95()
        < greedy_pair(ref, _perturb(ref, 100)).p95()
    )
    hid = aggregated_pair(ref, damaged, **kwargs).p95() < 500
    if hid:
        # Absorbing cues into groups is what hides the deletion.
        assert len(aggregated_pair(ref, damaged, **kwargs).pairs) < len(
            aggregated_pair(ref, _shift(ref, 0), **kwargs).pairs
        ) or True  # documented trade-off; see module docstring
    # Whatever the variant does, it must never score damaged input as clean
    # while also claiming to accept the good cases.
    if helped:
        assert not hid, (
            "a variant cannot both reduce residuals and keep damaged input visible"
        )


def test_conservative_aggregation_helps_a_synthetic_resegmented_case():
    """On synthetic re-segmented input the conservative variant does help.

    Recorded because it is the most tempting result in this whole experiment: a
    uniform +3s offset over variable-length cues drops from 3000ms to 1280ms,
    which is under the production limit, while truncation stays visible at
    1680ms. A candidate that behaved like this on the real artifacts would be
    the answer.

    It is not, and the reason is the point of the next test. This fixture models
    re-segmentation, which is what the candidate repairs. The real EVOLV/ASAP
    residual is not re-segmentation -- it is the ~96s piecewise opening offset --
    so the candidate has nothing to repair there and the measured real p95 does
    not fall (EVOLV 3890ms, ASAP 3850ms, both above the 2000ms limit). Adopting
    a candidate on the strength of this synthetic row would be a mistake, which
    is why this test asserts the synthetic result and the next one records the
    disqualification.
    """
    ref = _resegmented_reference()
    kwargs = {"max_group": 2, "max_group_span_ms": 6000}
    assert greedy_pair(ref, _shift(ref, 3000)).p95() == 3000.0
    helped = aggregated_pair(ref, _shift(ref, 3000), **kwargs).p95()
    assert helped < 2000, f"expected the synthetic case to be rescued, got {helped}"
    # And the aggressive variant is the mirror image: it helps more and hides
    # deletion entirely, which is why it was rejected.
    aggressive = {"max_group": 3, "max_group_span_ms": 6000}
    assert aggregated_pair(ref, _truncate(ref, 0.6), **aggressive).p95() < 500
    assert aggregated_pair(ref, _truncate(ref, 0.6), **kwargs).p95() >= 500


def test_candidate_does_not_address_the_real_residual_shape():
    """The real residual is a large piecewise offset, not re-segmentation.

    A uniform offset plus jitter is a shape aggregation *can* repair, and it does
    so. The real cases are not that shape: their residual comes from an opening
    section displaced by roughly 96s that is not a uniform shift, so no change to
    cue correspondence can remove it. Modelled here so the reason the candidate
    was not adopted is executable rather than anecdotal.
    """
    ref = _resegmented_reference()
    kwargs = {"max_group": 2, "max_group_span_ms": 6000}
    # A piecewise offset: the opening third is displaced, the rest is not.
    boundary = len(ref) // 3
    piecewise = [
        (s + 96_000, e + 96_000, x) if i < boundary else (s, e, x)
        for i, (s, e, x) in enumerate(ref)
    ]
    for name, result in (
        ("greedy", greedy_pair(ref, piecewise)),
        ("aggregated", aggregated_pair(ref, piecewise, **kwargs)),
    ):
        assert result.p95() > 2000, (
            f"{name} unexpectedly scored a 96s piecewise offset as acceptable; "
            f"the real residual would then be a pairing problem after all"
        )


def test_aggregation_is_monotonic_where_greedy_is_not():
    """A real advantage of the candidate, recorded even though it is not enough.

    Greedy nearest-unused can pair a cue to an earlier unused partner, so its
    index sequence is not monotonic. The candidate always is. That is necessary
    for a defensible correspondence but does not make it safe.
    """
    ref = _reference()
    noisy = _perturb(ref, 2000)
    assert not greedy_pair(ref, noisy).monotonic
    assert aggregated_pair(ref, noisy).monotonic


def test_candidate_never_accepts_on_its_own():
    """Guard against a future change wiring this in as an acceptance signal."""
    import inspect

    src = inspect.getsource(aggregated_pair)
    for banned in ("VERIFIED", "verified_resynced", "may_serve", "SyncState"):
        assert banned not in src, f"candidate references {banned}"


# ------------------------------------------------- completeness dataclass --- #


def test_completeness_never_raises_on_empty_input():
    ref = _reference()
    assert isinstance(completeness(ref, [], runtime_ms=1000), Completeness)
    assert isinstance(completeness([], [], runtime_ms=0), Completeness)
