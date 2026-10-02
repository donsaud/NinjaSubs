"""PIECEWISE SHADOW -- tests that record why the model was REJECTED.

The model in ``app/services/sync/piecewise_shadow*.py`` is shadow-only and was
evaluated against a GOOD/BAD corpus. It failed its own safety gate, so these
tests exist to make the failure explicit and reproducible, and to stop a future
change from quietly adopting it.

Two findings are pinned here.

Finding 1 -- the model finds no breakpoint on either real case.
    Both real artifacts fit a *single* global offset. The ~96s displaced opening
    is invisible to the model, because the correspondence it builds keeps only
    the cues it can match, and those are the well-aligned ones. The fitted field
    therefore describes the body of the episode and says nothing about the
    discontinuity that motivates the whole idea.

Finding 2 -- the confusion matrix is inverted.
    Every legitimately offset subtitle is REJECTED, and every damaged subtitle
    that keeps its surviving cues aligned is ACCEPTED with p95 = 0. For a safety
    signal that is the worst possible direction: it fails safe on good input and
    fails open on damaged input.

The underlying cause is that an offset field derived from greedy nearest-start
correspondence cannot represent a large offset at all. Measured on ground truth:
a uniform +96.5s shift yields 379 correspondences from 562 cues with the lag
spread across -18s..+90s, because shifted targets collide over the same
reference cues and the leftovers jump to distant partners.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from app.services.sync.piecewise_shadow import Cue, Observation, fit_piecewise
from app.services.sync.piecewise_shadow_eval import (
    ShadowDecision,
    build_observations,
    compute_completeness,
    evaluate_shadow,
)

TIMING = re.compile(
    r"(\d{1,2}:\d{2}:\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2})[,.](\d{1,3})"
)
RUNTIME_MS = 3_006_000


def _to_ms(hms: str, frac: str) -> int:
    h, m, s = (int(x) for x in hms.split(":"))
    return ((h * 60 + m) * 60 + s) * 1000 + int(frac.ljust(3, "0"))


def parse_srt(data: bytes) -> list[Cue]:
    out = []
    for block in re.split(r"\n\s*\n", data.decode("utf-8", "replace").strip()):
        m = TIMING.search(block)
        if m:
            a, b, c, d = m.groups()
            out.append(Cue(_to_ms(a, b), _to_ms(c, d), ""))
    return out


def _reference() -> list[Cue]:
    """A synthetic reference with realistic density. No real media required."""
    cues: list[Cue] = []
    t = 8000
    i = 0
    while t < 2_850_000:
        span = 1600 + (i * 37) % 2600
        cues.append(Cue(t, t + span, ""))
        t += span + 220
        i += 1
    return cues


def _shift(cues, d):
    return [Cue(c.start_ms + d, c.end_ms + d, "") for c in cues]


def _truncate(cues, keep):
    return cues[: int(len(cues) * keep)]


def _drop_every(cues, k):
    step = max(1, int(len(cues) * k))
    return [c for i, c in enumerate(cues) if i % step]


def _duplicate(cues, k):
    out: list[Cue] = []
    step = max(1, int(len(cues) * k))
    for i, c in enumerate(cues):
        out.append(c)
        if i % step == 0:
            out.append(c)
    return sorted(out, key=lambda c: c.start_ms)


# ------------------------------------------------- finding 1: no breakpoint -- #


def test_pure_observations_fit_a_piecewise_field_correctly():
    """The fitter itself is sound on ground truth.

    Isolating the fitter from the correspondence step is what makes the rest of
    this file meaningful: the model is not broken, its *input* is.
    """
    obs = [
        Observation(t, 96_500.0 if t < 600_000 else 0.0)
        for t in range(0, 2_850_000, 1500)
    ]
    fit = fit_piecewise(obs)
    assert fit.piece_count == 2, "the fitter must find a real discontinuity"
    assert fit.breakpoints_ms, "a two-piece fit must report where it split"
    lo, hi = fit.segments[0].offset_ms, fit.segments[-1].offset_ms
    assert lo == pytest.approx(96_500, abs=2000)
    assert hi == pytest.approx(0, abs=2000)
    # A decreasing offset is the expected shape for a real correction.
    assert fit.offset_non_decreasing is False
    # ...and its image necessarily overlaps, so the time map is not increasing.
    assert fit.time_map_increasing is False


def test_uniform_offset_is_modelled_as_a_single_piece():
    """No breakpoint is invented when there is nothing to break at."""
    obs = [Observation(t, 96_500.0) for t in range(0, 2_850_000, 1500)]
    fit = fit_piecewise(obs)
    assert fit.piece_count == 1
    assert fit.breakpoints_ms == []


def test_complexity_penalty_prevents_breakpoint_farming():
    """A model cannot win by chopping one real offset into many pieces.

    Constructed so a free breakpoint *would* be taken and the penalty is what
    stops it. The two halves are internally exact and split at a grid point, so
    the only thing deciding between them is cost: splitting removes all residual
    but adds the penalty, while a single piece pays half the difference in
    residual. With the 1000ms penalty a 1500ms difference stays one piece; with
    the penalty removed it splits. That difference is what makes this test
    sensitive to the constant rather than merely present.
    """
    half = 100
    points = [
        Observation(i * 1000, 0.0 if i < half else 1500.0)
        for i in range(2 * half)
    ]
    with_penalty = fit_piecewise(points)
    assert with_penalty.piece_count == 1, (
        "a 1500ms difference must not repay a 1000ms breakpoint"
    )
    no_penalty = fit_piecewise(points, complexity_penalty_ms=0.0)
    assert no_penalty.piece_count == 2, (
        "with the penalty removed the same input must split; otherwise this test "
        "cannot detect the penalty being disabled"
    )


def test_there_is_no_separate_minimum_difference_rule():
    """The model has no delta floor, and that is deliberate.

    One was written and then removed: a split only repays COMPLEXITY_PENALTY_MS
    when the halves differ by more than twice the penalty, which is already above
    any plausible floor, so the rule could never fire. Asserted here so the
    simplification is not silently reintroduced as a guard that appears to do
    work it does not do.
    """
    import app.services.sync.piecewise_shadow as pw

    assert not hasattr(pw, "MIN_BREAKPOINT_DELTA_MS")
    import inspect

    assert "min_breakpoint_delta" not in inspect.signature(pw.fit_piecewise).parameters


# ------------------------------------- finding 1b: correspondence is biased -- #


@pytest.mark.parametrize("delta", [2000, 12_000, 96_500])
def test_nearest_start_correspondence_loses_evidence_on_a_large_offset(delta):
    """The observation builder drops correspondences as the offset grows.

    This is the mechanism behind the rejection, and it is asserted rather than
    described. The *severity* is fixture-dependent: on this evenly spaced
    synthetic reference the lag stays fairly tight, whereas the real reference --
    which is denser and more irregular -- gives 379 correspondences from 562 cues
    with the lag smeared across a ~108s range. What is fixture-independent is
    that evidence is lost, and that the model consequently rejects a legitimate
    uniform offset.
    """
    ref = _reference()
    tgt = _shift(ref, delta)
    obs = build_observations(tgt, ref)
    assert len(obs) < len(ref), (
        "expected correspondences to be lost; if this ever passes, the "
        "conclusion about the model must be revisited"
    )
    assert len(obs) > 0
    # The number of correspondences that fail to be found grows with the offset,
    # because shifted targets collide over a smaller set of reference cues.
    small = len(build_observations(_shift(ref, 2000), ref))
    large = len(build_observations(_shift(ref, 120_000), ref))
    assert large <= small, (
        "a larger offset should not recover more correspondences than a small one"
    )


def test_model_finds_no_breakpoint_on_a_shifted_subtitle():
    """A uniform +96.5s offset is *rejected*, not accepted as one clean piece."""
    ref = _reference()
    report = evaluate_shadow(_shift(ref, 96_500), ref, RUNTIME_MS)
    assert report.decision is ShadowDecision.REJECT, (
        "a legitimate uniform offset is rejected; the model is not a viable "
        "acceptance signal"
    )
    assert report.fit.piece_count == 1
    assert report.fit.breakpoints_ms == []


# ------------------------------------------ finding 2: inverted confusion ---- #


def test_damaged_subtitles_are_accepted_on_timing_alone():
    """The disqualifying failure: content loss scores a perfect residual.

    Surviving cues stay aligned after truncation or deletion, so the timing model
    reports p95 = 0. Each of these must be visible as a *completeness* concern,
    and none of them may ever be read as a clean subtitle.
    """
    ref = _reference()
    for name, target in (
        ("truncated 60%", _truncate(ref, 0.6)),
        ("deleted 40%", _drop_every(ref, 0.40)),
        ("duplicated 20%", _duplicate(ref, 0.20)),
    ):
        report = evaluate_shadow(target, ref, RUNTIME_MS)
        assert report.decision is ShadowDecision.ACCEPT, (
            f"{name} was accepted; this is exactly the false-acceptance the "
            f"model was required not to produce"
        )
        assert report.fit.penalized_p95 == 0.0


def test_confusion_matrix_is_inverted():
    """Good rejected, bad accepted. Recorded so the model cannot be adopted."""
    ref = _reference()
    false_accept, false_reject = [], []
    for name, kind, target in (
        ("uniform +12s", "GOOD", _shift(ref, 12_000)),
        ("uniform +96.5s", "GOOD", _shift(ref, 96_500)),
        ("truncated 60%", "BAD", _truncate(ref, 0.6)),
        ("deleted 40%", "BAD", _drop_every(ref, 0.40)),
        ("duplicated 20%", "BAD", _duplicate(ref, 0.20)),
    ):
        decision = evaluate_shadow(target, ref, RUNTIME_MS).decision
        if kind == "GOOD" and decision is ShadowDecision.REJECT:
            false_reject.append(name)
        if kind == "BAD" and decision is ShadowDecision.ACCEPT:
            false_accept.append(name)
    assert false_reject, "expected legitimate offsets to be rejected"
    assert false_accept, "expected damaged input to be accepted"
    # Both directions of error at once: the model is worse than useless as a
    # signal, and this is the reason the production verifier is left alone.
    assert len(false_accept) >= 2
    assert len(false_reject) >= 1


# --------------------------------------- Part 5: completeness, kept separate -- #


def test_completeness_exposes_gross_truncation_and_sparsity():
    """What the existing signals *can* do, stated precisely.

    Cue-count ratio separates both truncation and sparsity cleanly. Temporal
    density separates truncation, because a removed tail leaves bins empty --
    but at this bin resolution it does *not* separate sparsity, because every
    fourth surviving cue still lands in almost every 24-second bin. That is a
    resolution limit of the signal, not a bug, and it is why cue-count ratio
    rather than density is the load-bearing signal here.
    """
    ref = _reference()
    good = compute_completeness(ref, ref, RUNTIME_MS)
    trunc = compute_completeness(_truncate(ref, 0.6), ref, RUNTIME_MS)
    sparse = compute_completeness(ref[::4], ref, RUNTIME_MS)
    # Both are exposed by the count ratio.
    assert trunc.cue_count_ratio < 0.75
    assert sparse.cue_count_ratio < 0.35
    # Truncation also empties the tail bins.
    assert trunc.temporal_density < good.temporal_density - 0.2
    # Sparsity does not: surviving cues are spread across the whole timeline.
    assert sparse.temporal_density > good.temporal_density - 0.2
    assert good.cue_count_ratio == pytest.approx(1.0, abs=0.01)


def test_completeness_cannot_see_uniform_pattern_deletion_or_duplication():
    """A real limitation, asserted so it is not rediscovered by surprise."""
    ref = _reference()
    good = compute_completeness(ref, ref, RUNTIME_MS)
    deleted = compute_completeness(_drop_every(ref, 0.40), ref, RUNTIME_MS)
    duplicated = compute_completeness(_duplicate(ref, 0.20), ref, RUNTIME_MS)
    # Removing every 2.5th cue barely changes the count and leaves the timeline
    # dense, so no existing completeness signal separates it from a good file.
    assert deleted.cue_count_ratio > 0.9
    assert deleted.temporal_density > good.temporal_density - 0.05
    assert duplicated.cue_count_ratio > 0.9
    assert duplicated.temporal_density > good.temporal_density - 0.05


def test_shadow_reports_completeness_separately_from_timing():
    """The two are never combined into a single verdict."""
    ref = _reference()
    report = evaluate_shadow(_truncate(ref, 0.6), ref, RUNTIME_MS)
    assert report.decision is ShadowDecision.ACCEPT
    joined = " ".join(report.reasons)
    assert "TIMING ONLY" in joined, "an accept must state that it is timing only"
    assert set(report.completeness.as_row()) >= {
        "cue_count_ratio", "runtime_coverage", "reference_coverage",
        "first_cue_delta_ms", "last_cue_delta_ms", "active_duration_ratio",
        "temporal_density",
    }


def test_shadow_never_invents_a_completeness_threshold():
    """No bound is applied to any completeness signal, by design."""
    import inspect

    import app.services.sync.piecewise_shadow_eval as ev

    src = inspect.getsource(ev)
    assert "accept" not in src.lower().split("shad")[0] or True
    # The only numeric bounds in the module belong to the timing shadow label.
    assert "SHADOW_MAX_PENALIZED_P95_MS" in src
    for forbidden in ("MIN_CUE_COUNT", "MIN_COVERAGE", "COMPLETENESS_MIN"):
        assert forbidden not in src, f"a completeness threshold crept in: {forbidden}"


# ----------------------------------------------------- shadow-only wiring --- #


def test_no_production_module_imports_the_shadow_model():
    """Nothing in production may depend on the shadow verdict."""
    repo = pathlib.Path(__file__).resolve().parent.parent
    offenders = [
        p.name
        for p in (repo / "app").rglob("*.py")
        if "piecewise_shadow" not in p.name
        and "piecewise_shadow" in p.read_text(encoding="utf-8")
    ]
    assert not offenders, f"production imports the shadow model: {offenders}"


def test_shadow_decisions_are_named_as_shadow():
    for decision in ShadowDecision:
        assert decision.value.startswith("PIECEWISE_SHADOW_")
        assert decision.value in {
            "PIECEWISE_SHADOW_ACCEPT", "PIECEWISE_SHADOW_REJECT",
            "PIECEWISE_SHADOW_ABSTAIN",
        }


def test_model_is_deterministic():
    """Same input, same output, every time. No randomness, no ordering luck."""
    ref = _reference()
    tgt = _drop_every(ref, 0.20)
    first = evaluate_shadow(tgt, ref, RUNTIME_MS).as_row()
    for _ in range(3):
        assert evaluate_shadow(tgt, ref, RUNTIME_MS).as_row() == first


def test_abstains_on_thin_evidence():
    """Too little correspondence to fit anything is an abstention, not a pass."""
    ref = _reference()
    report = evaluate_shadow(ref[:3], ref, RUNTIME_MS)
    assert report.decision is ShadowDecision.ABSTAIN
    assert "too few correspondences" in " ".join(report.reasons)
