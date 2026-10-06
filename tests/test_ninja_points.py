"""NinjaSync Points Phase 1: offline prototype tests.

Scope is deliberately the prototype only: anchor detection, model fitting,
candidate generation, and independent verification. There is deliberately NO
test that serves, caches, ranks, or wires anything into the request pipeline,
because the module under test has no serving path by construction.

A background fact these tests rely on: the existing analyzer is the judge.
A clean constant shift of sufficient cues verifies as VERIFIED_* through the
real ``AlignmentAnalyzer``; anything else must yield ``best=None``.
"""

from __future__ import annotations

import pytest

from app.services.sync import ninja_points as np
from app.services.sync.ninja_points import (
    AnchorDetector,
    AnchorFitter,
    AnchorSet,
    HybridResult,
    HybridSyncEngine,
    SyncAnchor,
    render_srt,
)


def _cues(n: int, *, start_ms: int = 1000, step_ms: int = 3000, tag: str = "line"):
    return [
        (start_ms + i * step_ms, start_ms + i * step_ms + 2000, f"{tag} number {i} here")
        for i in range(n)
    ]


def _srt(cues) -> str:
    return render_srt(list(cues))


def _shifted(cues, offset_ms: int):
    return [(s + offset_ms, e + offset_ms, t) for s, e, t in cues]


# ===========================================================================
# Constant model
# ===========================================================================


def test_constant_perfect_offset_verifies():
    target = _cues(30)
    reference = _shifted(target, 2000)
    result = HybridSyncEngine().run(_srt(target), _srt(reference))
    assert result.best is not None
    assert result.best.strategy_id == np.STRATEGY_CONSTANT
    assert result.best.verified


def test_constant_small_noise_verifies():
    import random

    rng = random.Random(7)
    target = _cues(30)
    reference = [(s + 2000 + rng.randint(-120, 120), e + 2000, t) for s, e, t in target]
    result = HybridSyncEngine().run(_srt(target), _srt(reference))
    assert result.best is not None and result.best.verified


def test_constant_outlier_does_not_move_the_median():
    target = _cues(30)
    # 29 cues at +2000ms, one wild anchor far away.
    fitter = AnchorFitter()
    anchors = AnchorSet(
        anchors=tuple(
            SyncAnchor(subtitle_ms=s, reference_ms=(s + 2000 if i else s + 12000),
                       confidence=1.0, support=29 if i else 1)
            for i, (s, _, _) in enumerate(target)
        )
    )
    model = fitter.fit_constant(anchors)
    assert model is not None
    assert model.offset_ms == pytest.approx(2000.0)


def test_single_anchor_never_fits():
    fitter = AnchorFitter()
    lonely = AnchorSet(
        anchors=(SyncAnchor(subtitle_ms=1000, reference_ms=3000, confidence=1.0, support=1),)
    )
    assert fitter.fit_constant(lonely) is None
    assert fitter.fit_linear(lonely) is None


def test_contradictory_anchors_refuse():
    target = _cues(30)
    refs = [
        (s + (2000 if i % 3 == 0 else (-14000 if i % 3 == 1 else 9000)), e, t)
        for i, (s, e, t) in enumerate(target)
    ]
    anchors = AnchorDetector().detect(target, refs)
    # Contradictory offsets cannot corroborate: either no anchors survive or
    # no model fits them.
    fitter = AnchorFitter()
    assert fitter.fit_constant(anchors) is None
    assert fitter.fit_linear(anchors) is None


# ===========================================================================
# Linear model
# ===========================================================================


def _drifted(cues, slope: float, intercept: float):
    return [(slope * s + intercept, slope * e + intercept, t) for s, e, t in cues]


def _wide_anchors(n=250, slope=1.02, intercept=500.0, noise=0, seed=0):
    """Anchors spanning >10min of subtitle time, as MIN_LINEAR_SPAN_MS demands.

    Forty cues at 3s spacing span barely 2 minutes, on which no drift claim is
    measurable -- the span gate exists precisely to refuse such fits.
    """
    import random

    rng = random.Random(seed)
    out = []
    for i in range(n):
        s = float(1000 + i * 3000)
        r = slope * s + intercept + (rng.randint(-noise, noise) if noise else 0)
        out.append(SyncAnchor(subtitle_ms=s, reference_ms=r, confidence=1.0, support=n))
    return AnchorSet(anchors=tuple(out))


def test_linear_perfect_drift_fits():
    model = AnchorFitter().fit_linear(_wide_anchors())
    assert model is not None
    assert model.slope == pytest.approx(1.02, abs=1e-9)
    assert model.intercept_ms == pytest.approx(500.0, abs=1e-6)


def test_linear_small_noise_fits():
    anchors = _wide_anchors(noise=150, seed=11)
    lowered = AnchorSet(
        anchors=tuple(
            SyncAnchor(subtitle_ms=a.subtitle_ms, reference_ms=a.reference_ms,
                       confidence=0.9, support=a.support)
            for a in anchors
        )
    )
    assert AnchorFitter().fit_linear(lowered) is not None


def test_linear_outlier_refuses():
    anchors = list(_wide_anchors())
    bad = anchors[120]
    anchors[120] = SyncAnchor(
        subtitle_ms=bad.subtitle_ms, reference_ms=bad.reference_ms + 30000.0,
        confidence=1.0, support=len(anchors),
    )
    # One anchor 30s off the line: the full-set residual gate must refuse.
    assert AnchorFitter().fit_linear(AnchorSet(anchors=tuple(anchors))) is None


def test_linear_insufficient_separation_refuses():
    # All anchors inside a 60s window: drift is unmeasurable, refuse.
    anchors = AnchorSet(
        anchors=tuple(
            SyncAnchor(subtitle_ms=1000 + i * 2000, reference_ms=3000 + i * 2000,
                       confidence=1.0, support=10)
            for i in range(10)
        )
    )
    assert AnchorFitter().fit_linear(anchors) is None


def test_linear_insufficient_anchors_refuses():
    anchors = AnchorSet(
        anchors=tuple(
            SyncAnchor(subtitle_ms=1000 + i * 600000, reference_ms=3000 + i * 600000,
                       confidence=1.0, support=2)
            for i in range(2)
        )
    )
    assert AnchorFitter().fit_linear(anchors) is None


def test_implausible_slope_refuses():
    anchors = AnchorSet(
        anchors=tuple(
            SyncAnchor(subtitle_ms=float(i * 600000), reference_ms=float(i * 900000),
                       confidence=1.0, support=10)
            for i in range(10)
        )
    )
    # Slope 1.5 is a speed change, not a frame-rate conversion.
    assert AnchorFitter().fit_linear(anchors) is None


# ===========================================================================
# Anchor detection
# ===========================================================================


def test_detection_finds_unique_matches():
    target = _cues(20)
    reference = _shifted(target, 1500)
    anchors = AnchorDetector().detect(target, reference)
    assert len(anchors) == 20
    assert all(a.support == 20 for a in anchors)


def test_detection_ignores_repeated_phrases():
    target = [(1000 + i * 3000, 3000 + i * 3000, "Yes." if i % 2 else f"unique line {i}") for i in range(20)]
    reference = [(s + 1500, e + 1500, t) for s, e, t in target]
    anchors = AnchorDetector().detect(target, reference)
    # "Yes." appears 10 times on each side: never an anchor.
    assert len(anchors) == 10


def test_detection_rejects_duration_mismatch():
    target = [(1000, 3000, "a substantive line of dialogue here")]
    reference = [(2500, 12000, "a substantive line of dialogue here")]
    assert len(AnchorDetector().detect(target, reference)) == 0


def test_detection_requires_corroboration():
    # Exactly one shared line: support 1 < min_support 2, so no anchors.
    target = [(1000, 3000, "only shared line here")] + _cues(9, tag="target-only")
    reference = [(2500, 4500, "only shared line here")] + _cues(9, tag="reference-only")
    assert len(AnchorDetector().detect(target, reference)) == 0


def test_detection_respects_temporal_plausibility():
    target = _cues(20)
    reference = _shifted(target, 120_000)  # 2 minutes: beyond the default window
    assert len(AnchorDetector().detect(target, reference)) == 0


def test_detection_does_not_mutate_inputs():
    target = _cues(10)
    reference = _shifted(target, 1500)
    snapshot_t, snapshot_r = list(target), list(reference)
    AnchorDetector().detect(target, reference)
    assert target == snapshot_t
    assert reference == snapshot_r


# ===========================================================================
# Verification authority and serving safety
# ===========================================================================


def test_unrelated_subtitle_yields_no_best():
    target = _cues(30, tag="alpha")
    reference = _cues(30, tag="completely different beta", start_ms=50000)
    result = HybridSyncEngine().run(_srt(target), _srt(reference))
    assert result.best is None
    # Candidates may exist, but none may be accepted.
    assert all(not c.verified for c in result.candidates)


def test_verified_candidate_is_eligible_and_unverified_is_not():
    target = _cues(30)
    reference = _shifted(target, 2000)
    result = HybridSyncEngine().run(_srt(target), _srt(reference))
    assert result.best is not None and result.best.verified
    bad = HybridSyncEngine().run(
        _srt(_cues(30, tag="alpha")), _srt(_cues(30, tag="beta", start_ms=50000))
    )
    assert bad.best is None


def test_engine_has_no_serving_path():
    """Phase 1 structural guarantee: nothing to call that serves or caches."""
    import app.services.sync.ninja_points as module

    source = open(module.__file__, encoding="utf-8").read()
    for forbidden in ("def serve", "set_verdict", "cache.set", "order_candidates"):
        assert forbidden not in source
    summary = HybridSyncEngine.summarize(HybridResult(best=None, candidates=[]))
    assert summary["served"] is False


def test_provenance_records_the_true_producer():
    target = _cues(30)
    reference = _shifted(target, 2000)
    result = HybridSyncEngine().run(_srt(target), _srt(reference))
    assert result.best is not None
    assert result.best.strategy_id == np.STRATEGY_CONSTANT
    assert result.best.model_version == np.MODEL_VERSION
    reasons = " ".join(result.best.evaluation.reasons)
    assert "ninja_points_constant" in reasons
    assert "without alass" in reasons


def _candidate_with_state(state_value: str):
    from app.services.sync.ninja_points import HybridCandidate

    class _State:
        value = state_value

    class _Eval:
        sync_state = _State()

    return HybridCandidate(
        strategy_id="ninja_points_constant", model_version="x",
        description="x", transformed_text="x", evaluation=_Eval(),
    )


def test_verified_gate_accepts_only_verified_states():
    assert _candidate_with_state("verified_synced").verified is True
    assert _candidate_with_state("verified_resynced").verified is True
    for state in ("probable_sync", "unverified", "rejected", "unknown"):
        assert _candidate_with_state(state).verified is False, state


def test_strategy_ids_and_version_are_stable():
    assert np.STRATEGY_CONSTANT == "ninja_points_constant"
    assert np.STRATEGY_LINEAR == "ninja_points_linear"
    assert np.MODEL_VERSION == "ninja-points-v1"


# ===========================================================================
# Model selection, determinism, immutability
# ===========================================================================


def test_constant_preferred_over_linear_when_both_verify():
    """Simplicity wins structurally: linear is not even attempted."""
    target = _cues(30)
    reference = _shifted(target, 2000)
    result = HybridSyncEngine().run(_srt(target), _srt(reference))
    assert result.best is not None
    assert result.best.strategy_id == np.STRATEGY_CONSTANT
    assert [c.strategy_id for c in result.candidates] == [np.STRATEGY_CONSTANT]


def test_alass_verified_result_is_never_overridden():
    class _FakeEval:
        class _State:
            value = "verified_synced"

        sync_state = _State()

    from app.services.sync.ninja_points import HybridCandidate

    alass_candidate = HybridCandidate(
        strategy_id="alass", model_version="alass",
        description="alass", transformed_text="x", evaluation=_FakeEval(),
    )
    assert alass_candidate.verified
    # The engine returns a verified alass output immediately without fitting.
    target = _cues(30)
    reference = _shifted(target, 2000)
    result = HybridSyncEngine().run(
        _srt(target), _srt(reference), alass_output=_srt(reference)
    )
    assert result.best is not None
    assert result.best.strategy_id == "alass"
    # Ninja models must not even be attempted once Alass verified.
    assert [c.strategy_id for c in result.candidates] == ["alass"]


def test_determinism_across_runs():
    target = _cues(30)
    reference = _shifted(target, 2000)
    first = HybridSyncEngine().run(_srt(target), _srt(reference))
    second = HybridSyncEngine().run(_srt(target), _srt(reference))
    assert (first.best.strategy_id if first.best else None) == (
        second.best.strategy_id if second.best else None
    )
    assert first.best.transformed_text == second.best.transformed_text


def test_anchor_set_immutability():
    import dataclasses

    anchors = AnchorDetector().detect(_cues(10), _shifted(_cues(10), 1500))
    with pytest.raises(dataclasses.FrozenInstanceError):
        anchors.anchors[0].confidence = 0.0  # type: ignore[misc]


def test_models_apply_without_mutating_input():
    from app.services.sync.ninja_points import ConstantSyncModel, LinearSyncModel

    cues = _cues(10)
    snapshot = list(cues)
    constant = ConstantSyncModel(offset_ms=2000.0, inliers=10)
    shifted = constant.apply(cues)
    assert cues == snapshot
    assert [s for s, _, _ in shifted] == [s + 2000 for s, _, _ in cues]
    assert "ninja_points_constant" in constant.describe()
    linear = LinearSyncModel(slope=1.0, intercept_ms=2000.0, inliers=10)
    moved = linear.apply(cues)
    assert cues == snapshot
    assert [s for s, _, _ in moved] == [s + 2000 for s, _, _ in cues]
    assert "ninja_points_linear" in linear.describe()


def test_shadow_summary_is_counts_and_metrics_only():
    target = _cues(30)
    reference = _shifted(target, 2000)
    result = HybridSyncEngine().run(_srt(target), _srt(reference))
    summary = HybridSyncEngine.summarize(result)
    assert summary["served"] is False
    assert summary["best_strategy"] == np.STRATEGY_CONSTANT
    assert summary["best_model_version"] == np.MODEL_VERSION
    assert summary[np.STRATEGY_CONSTANT]["verified"] is True
    blob = str(summary)
    assert "line number" not in blob
