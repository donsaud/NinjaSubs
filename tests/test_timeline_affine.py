"""Temporal equivalence, distributed cut refusal, and verified delivery guards."""

from __future__ import annotations

import random
import re
from pathlib import Path

import pytest

from app.services.subtitle_matcher import parse_srt_cues
from app.services.sync.ninja_points import render_srt
from app.services.sync.timeline_affine import (
    MAX_HYPOTHESES,
    MAX_LANDMARKS,
    Timeline,
    attempt_timeline_affine,
    hypotheses,
    plausible_affine_reference,
    transform_text,
    verify_timeline,
)


def film(seed=37, groups=100):
    rng = random.Random(seed)
    cues = []
    start = 142640
    for i in range(groups):
        length = rng.randrange(2600, 6200)
        cues.append((start, start + length, f"  Speaker {i}\nSecond speaker.  "))
        start += length + rng.randrange(2200, 18000)
    return cues


def reference(cues, slope=1.0, intercept=5000, segmentation="mixed"):
    out = []
    for i, (s, e, _) in enumerate(cues):
        s, e = round(s * slope + intercept), round(e * slope + intercept)
        if segmentation == "split" or segmentation == "mixed" and i % 3 == 0:
            middle = (s + e) // 2
            out.extend([(s, middle, "Bonjour."), (middle + 50, e, "Bonsoir.")])
        else:
            out.append((s, e, "Dialogue in another language."))
    return out


@pytest.mark.parametrize("slope,intercept,kind", [
    (1.0, 5000, "CONSTANT_OFFSET"),
    (25 / 23.976, -8621.595, "PAL_LIKE_AFFINE"),
    (23.976 / 25, 3000, "PAL_LIKE_AFFINE"),
    (1.0123, -1000, "GLOBAL_AFFINE"),
])
@pytest.mark.parametrize("segmentation", ["none", "split", "mixed"])
def test_global_affine_with_different_segmentation(slope, intercept, kind, segmentation):
    target = film()
    ref = reference(target, slope, intercept, segmentation)
    original = render_srt(target).replace("\n", "\r\n")
    result = attempt_timeline_affine(original, render_srt(ref))
    assert result.accepted, result.diagnostics
    assert result.slope == pytest.approx(slope, abs=0.000002)
    assert result.intercept_ms == pytest.approx(intercept, abs=2)
    assert result.diagnostics.classification == kind
    assert result.diagnostics.after - result.diagnostics.before >= 0.06
    generated = parse_srt_cues(result.transformed_text)
    expected = parse_srt_cues(transform_text(original, slope, intercept))
    assert len(generated) == len(target)
    assert [t for _, _, t in generated] == [t for _, _, t in parse_srt_cues(original)]
    assert max(abs(s - es) for (s, _, _), (es, _, _) in zip(generated, expected, strict=True)) <= 2
    assert re.sub(r"\d\d:\d\d:\d\d,\d{3}", "TIME", original) == re.sub(
        r"\d\d:\d\d:\d\d,\d{3}", "TIME", result.transformed_text)
    assert result.evaluation.verification.value == "verified"
    assert not result.evaluation.alass_applied


def test_reverse_many_to_one_and_duplicate_overlap():
    whole = film()
    split = reference(whole, 1, 0, "split")
    ref = reference(whole, 1.013, 4000, "none")
    ref += ref[10:30]  # duplication has no extra occupancy weight
    ref += [(ref[50][0] + 100, ref[50][1] - 100, "Overlap.")]
    result = attempt_timeline_affine(render_srt(split), render_srt(ref))
    assert result.accepted, result.diagnostics
    assert len(parse_srt_cues(result.transformed_text)) == len(split)


def test_metadata_is_never_endpoint():
    target = film()
    ref = reference(target, 1.0123, 5000)
    ref = [(0, 1500, "Translated by: uploader"), (2000, 4000, "https://example.test")] + ref
    ref += [(ref[-1][1] + 100000, ref[-1][1] + 105000, "Subtitles by: uploader")]
    timeline = Timeline.from_cues(ref)
    assert next(m.time_ms for m in timeline.landmarks if m.kind == "first") == ref[2][0]
    assert next(m.time_ms for m in timeline.landmarks if m.kind == "last") == reference(
        target, 1.0123, 5000)[-1][0]
    result = attempt_timeline_affine(render_srt(target), render_srt(ref))
    assert result.accepted, result.diagnostics
    assert result.slope == pytest.approx(1.0123, abs=0.000002)


@pytest.mark.parametrize("mode", ["step", "middle", "wrong_movie", "missing_scene", "plausible_slope"])
def test_real_cuts_and_wrong_references_fail(mode):
    target = film(groups=160)
    ref = reference(target, 1.0123, 5000)
    if mode == "wrong_movie":
        ref = reference(film(seed=92, groups=160), 1.0123, 5000)
    elif mode == "missing_scene":
        ref = ref[:35] + ref[65:]
    elif mode == "plausible_slope":
        ref = [(s + (i % 7) * 2000, e + (i % 7) * 2000, t) for i, (s, e, t) in enumerate(ref)]
    else:
        lo, hi = target[60][0] * 1.0123 + 5000, target[110][0] * 1.0123 + 5000
        ref = [(s + (7000 if s >= lo and (mode == "step" or s < hi) else 0),
                e + (7000 if s >= lo and (mode == "step" or s < hi) else 0), t)
               for s, e, t in ref]
    result = attempt_timeline_affine(render_srt(target), render_srt(ref))
    assert not result.accepted, result.diagnostics
    assert result.transformed_text is None
    assert result.evaluation is None
    if mode == "middle":
        report = verify_timeline(Timeline.from_cues(target), Timeline.from_cues(ref), 1.0123, 5000)
        assert not report.accepted
        assert "local_discontinuity" in report.reasons
        assert "inconsistent_windows" in report.reasons


def test_no_improvement_is_not_accepted():
    target = film()
    result = attempt_timeline_affine(render_srt(target), render_srt(reference(target, 1, 0)))
    assert not result.accepted
    report = verify_timeline(Timeline.from_cues(target), Timeline.from_cues(target), 1, 0)
    assert report.after == 1
    assert "no_meaningful_improvement" in report.reasons


def test_insufficient_evidence_and_bounded_search():
    assert not attempt_timeline_affine(render_srt(film(groups=3)), render_srt(film(groups=3))).accepted
    timeline = Timeline.from_cues(film(groups=2000))
    assert len(timeline.landmarks) <= MAX_LANDMARKS
    assert len(hypotheses(timeline, timeline)) <= MAX_HYPOTHESES
    # Featureless constant occupancy has no distinctive temporal evidence.
    assert not attempt_timeline_affine(render_srt([(0, 200000, "Hello.")]),
                                      render_srt([(5000, 205000, "Bonjour.")])).accepted


def test_noisy_landmarks_do_not_choose_wrong_hypothesis():
    target = film()
    ref = reference(target, 1.0123, 5000)
    ref[40:44] = [(s + 300, e + 300, t) for s, e, t in ref[40:44]]
    result = attempt_timeline_affine(render_srt(target), render_srt(ref))
    assert result.accepted, result.diagnostics
    assert result.slope == pytest.approx(1.0123, abs=0.000002)
    assert result.diagnostics.after > 0.99


def test_display_duration_noise_and_short_gaps():
    target = film()
    ref = reference(target, 1.0123, 5000)
    ref = [(s, e + (i % 5 - 2) * 180, t) for i, (s, e, t) in enumerate(ref)]
    result = attempt_timeline_affine(render_srt(target), render_srt(ref))
    assert result.accepted, result.diagnostics
    assert result.slope == pytest.approx(1.0123, abs=0.00001)
    assert result.intercept_ms == pytest.approx(5000, abs=20)


def test_metadata_does_not_contribute_occupancy():
    cues = film()
    before = Timeline.from_cues(cues)
    after = Timeline.from_cues([(0, 3000, "Translated by: uploader")] + cues + [
        (cues[-1][1] + 100000, cues[-1][1] + 104000, "https://example.test")])
    assert after.occupancy == before.occupancy
    assert after.landmarks == before.landmarks


async def test_orchestrator_affine_cache_and_provenance(monkeypatch):
    from app.services.sync.orchestrator import SyncOrchestrator
    from app.services.sync.query import ResolvedReference
    from app.services.sync_cache import SyncCache

    target = render_srt(film())
    resolved = ResolvedReference(render_srt(reference(film(), 1.0123, 5000)), kind="edition")
    cache = SyncCache()
    orch = SyncOrchestrator(sync_cache=cache)
    served = await orch._try_timeline_affine(target, resolved, {}, "sub", "sha", "rk", None)
    assert served is not None and served != target.encode()
    assert await cache.get("rk") == served
    meta = await cache.get_meta("rk")
    assert meta["verification"] == "verified"
    assert meta["reference_kind"] == "edition"
    assert meta["model_version"] == "timeline-affine-v2"
    assert orch._last_evaluation.sync_state.value == "verified_resynced"


async def test_orchestrator_refuses_unverified_and_partial(monkeypatch):
    from app.services.sync import timeline_affine as module
    from app.services.sync.alignment import SubtitleEvaluation
    from app.services.sync.orchestrator import SyncOrchestrator
    from app.services.sync.query import ResolvedReference

    orch = SyncOrchestrator()
    monkeypatch.setattr(module, "attempt_timeline_affine", lambda *a: module.AffineResult(
        accepted=True, transformed_text="unsafe", evaluation=SubtitleEvaluation()))
    assert await orch._try_timeline_affine("target", ResolvedReference("ref"), {}, "sub", "sha", "rk", None) is None
    assert await orch._try_timeline_affine("target", ResolvedReference("ref", partial=True), {}, "sub", "sha", "rk", None) is None


async def test_multi_reference_affine_delivery_and_reuse(monkeypatch):
    from app.config import settings
    from app.services.sync.orchestrator import SyncOrchestrator
    from app.services.sync.query import ResolvedReference
    from app.services.sync_cache import SyncCache

    monkeypatch.setattr(settings, "ENABLE_SUBTITLE_SYNC", True)
    target = render_srt(film()).encode()
    refs = [ResolvedReference(render_srt(reference(film(seed=92), 1.0123, 5000)), kind="edition"),
            ResolvedReference(render_srt(reference(film(), 1.0123, 35000)), kind="edition")]

    class Multi:
        cache = None
        calls = 0

        async def resolve_all_with_provenance(self, query, *, update_validator=None):
            self.calls += 1
            return [ref for ref in refs if update_validator(ref.text)]

    class Alass:
        calls = 0

        async def sync_async(self, *args, **kwargs):
            self.calls += 1
            return None

    multi, alass, cache = Multi(), Alass(), SyncCache()
    orch = SyncOrchestrator(external_strategy=multi, sync_service=alass, sync_cache=cache)
    meta = {"imdb_id": "tt1234567", "lang": "fra", "target_filename": "Movie.2020.WEB.mkv"}
    served = await orch.evaluate_and_sync(target, meta, "sub", auto_sync=True)
    assert served != target
    assert orch._last_evaluation.verification.value == "verified"
    assert multi.calls == 1
    # The first reference fails; the second independently verifies with a
    # >20s intro displacement that discovery must not discard as a wrong cut.
    assert alass.calls <= 1
    assert await orch.evaluate_and_sync(target, meta, "sub", auto_sync=True) == served
    assert multi.calls == 1
    assert orch.metrics["verification_cache_hits"] >= 1


def test_discovery_span_hint_is_not_identity_proof():
    target = render_srt(film())
    assert plausible_affine_reference(target, render_srt(reference(film(), 1.0123, 35000)))
    # Even a wrong movie can share span; it still fails independent verification.
    wrong = render_srt(reference(film(seed=92), 1.0123, 35000))
    assert not attempt_timeline_affine(target, wrong).accepted


def test_whiplash_gold_standard():
    folder = Path(__file__).parent / "fixtures" / "whiplash"
    missing = [name for name in ("target.srt", "reference.srt", "expected.srt")
               if not (folder / name).is_file()]
    assert not missing, f"Required real Whiplash fixtures missing: {', '.join(missing)}"
    target = (folder / "target.srt").read_text(encoding="utf-8-sig")
    reference_text = (folder / "reference.srt").read_text(encoding="utf-8-sig")
    # The oracle is deliberately read only after automatic generation.
    result = attempt_timeline_affine(target, reference_text)
    assert result.accepted, result.diagnostics
    expected = parse_srt_cues((folder / "expected.srt").read_text(encoding="utf-8-sig"))
    original, generated = parse_srt_cues(target), parse_srt_cues(result.transformed_text)
    assert len(original) == len(generated) == len(expected) == 825
    assert len(parse_srt_cues(reference_text)) == 960
    assert [t for _, _, t in original] == [t for _, _, t in generated]
    assert [t for _, _, t in original] == [t for _, _, t in expected]
    assert result.slope == pytest.approx(1.042755149812734, abs=0.0000003)
    assert result.intercept_ms == pytest.approx(-8621.594569, abs=1)
    for (s, e, _), (es, ee, _) in zip(generated, expected, strict=True):
        assert s == es
        assert abs(e - ee) <= 1


def test_isolated_closing_dialogue_is_verified_with_middle_evidence():
    target = film()
    last = target[-1][1] + 180000
    target.append((last, last + 2600, "One final line."))
    ref = reference(target, 1.017, -2100)
    ref.append((ref[-1][1] + 100000, ref[-1][1] + 105000, "Subtitles by: uploader"))
    result = attempt_timeline_affine(render_srt(target), render_srt(ref))
    assert result.accepted, result.diagnostics
    assert result.slope == pytest.approx(1.017, abs=0.000002)
    assert result.intercept_ms == pytest.approx(-2100, abs=2)
    timeline = Timeline.from_cues(ref)
    assert next(m.time_ms for m in timeline.landmarks if m.kind == "boundary_last") == ref[-2][0]
