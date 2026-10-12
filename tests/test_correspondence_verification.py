"""Adversarial tests for correspondence-aware verification (§9).

A robust verifier accepts genuinely supported synchronization and rejects
convincing-looking but incorrect candidates. Synthetic pairs keep every case
self-contained; real fixtures are used only where the phenomenon needs them
(Dexter T#795), with graceful skips otherwise.
"""

from __future__ import annotations

import hashlib
import pathlib

import pytest

from app.services.sync.alignment import may_serve_synchronized
from app.services.sync.correspondence_verification import (
    METHOD,
    anchor_pairs,
    examine_outlier,
    held_out_checkpoints,
    to_evaluation,
    verify_linear_candidate,
    verify_nonlinear_candidate,
)
from tests.two_point_sync._helpers import apply_affine, build_dialogue, make_cue
from two_point_sync.anchors import Anchor, AnchorResult
from two_point_sync.anchors import detect_anchors as nucleus_detect
from two_point_sync.parser import Cue, format_srt, parse_srt
from two_point_sync.sanitizer import classify_all
from two_point_sync.transform import LinearTransform, apply_transform

DEX_DIR = pathlib.Path(__file__).resolve().parent / "fixtures" / "two_point_sync" / "dexter"
DEX_TARGET = DEX_DIR / "Dexter.S08E01.480p.HDTV.x264-mSD.srt"
DEX_REFERENCE = DEX_DIR / "Dexter.S08E01.A.Beautiful.Day.BluRay.1080p.TrueHD.5.1.AVC.REMUX-FraMeSToR.ar.srt"


def _linear_pair(count: int = 14, step_ms: int = 60_000,
                 scale: float = 1.0, offset_ms: float = 5_000.0):
    target_cues = build_dialogue(count, step_ms=step_ms)
    reference_cues = apply_affine(target_cues, scale, offset_ms)
    return format_srt(target_cues), format_srt(reference_cues)


def _detect(target_text: str, reference_text: str):
    from two_point_sync.parser import parse_srt as nucleus_parse

    target = nucleus_parse(target_text)
    reference = nucleus_parse(reference_text)
    return target, reference, nucleus_detect(
        target, reference,
        target_scores=classify_all(target), reference_scores=classify_all(reference))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------- positive cases ---


def test_legitimate_large_fps_drift_accepted() -> None:
    """A 25fps->23.976fps correction moves late cues ~30s: magnitude is
    telemetry, and the candidate verifies on correspondence residuals."""
    target_text, reference_text = _linear_pair(16, step_ms=45_000,
                                               scale=25.0 / 23.976, offset_ms=-8_000.0)
    target, reference, detection = _detect(target_text, reference_text)
    assert detection.accepted
    candidate = format_srt(apply_transform(
        target,
        LinearTransform(scale=detection.scale or 1.0,
                        offset_ms=detection.offset_ms or 0.0)))
    verdict = verify_linear_candidate(
        target_text=target_text, reference_text=reference_text,
        candidate_text=candidate, anchor_result=detection, reference_sha=_sha(reference_text))
    assert verdict.accepted, verdict.failure_reasons
    assert verdict.method == METHOD
    assert verdict.movement_magnitude_ms > 20_000
    assert verdict.median_residual_ms <= 750.0
    evaluation = to_evaluation(verdict)
    assert may_serve_synchronized(evaluation.sync_state, evaluation.verification)


def test_identity_pair_accepted() -> None:
    target_text, reference_text = _linear_pair()
    target, reference, detection = _detect(target_text, reference_text)
    assert detection.accepted
    candidate = format_srt(apply_transform(
        target, LinearTransform(scale=1.0, offset_ms=5_000.0)))
    verdict = verify_linear_candidate(
        target_text=target_text, reference_text=reference_text,
        candidate_text=candidate, anchor_result=detection, reference_sha=_sha(reference_text))
    assert verdict.accepted, verdict.failure_reasons


# ------------------------------------------------------- negative cases ---


def test_constant_wrong_offset_rejected() -> None:
    target_text, reference_text = _linear_pair()
    _, _, detection = _detect(target_text, reference_text)
    assert detection.accepted
    from two_point_sync.parser import parse_srt as nucleus_parse

    wrong = format_srt(apply_transform(
        nucleus_parse(target_text), LinearTransform(scale=1.0, offset_ms=60_000.0)))
    verdict = verify_linear_candidate(
        target_text=target_text, reference_text=reference_text,
        candidate_text=wrong, anchor_result=detection, reference_sha=_sha(reference_text))
    assert not verdict.accepted
    assert any("median_exceeds_tolerance" in reason for reason in verdict.failure_reasons)


def test_incorrect_linear_scale_rejected() -> None:
    target_text, reference_text = _linear_pair(16, step_ms=45_000,
                                               scale=25.0 / 23.976, offset_ms=-8_000.0)
    _, _, detection = _detect(target_text, reference_text)
    assert detection.accepted
    from two_point_sync.parser import parse_srt as nucleus_parse

    wrong = format_srt(apply_transform(
        nucleus_parse(target_text), LinearTransform(scale=1.0, offset_ms=-8_000.0)))
    verdict = verify_linear_candidate(
        target_text=target_text, reference_text=reference_text,
        candidate_text=wrong, anchor_result=detection, reference_sha=_sha(reference_text))
    assert not verdict.accepted


def test_wrong_episode_reference_insufficient() -> None:
    target_text, _ = _linear_pair()
    other = build_dialogue(14, sentences=[
        "Completely different sentence one", "Completely different sentence two",
        "Completely different sentence three", "Completely different sentence four",
        "Completely different sentence five", "Completely different sentence six",
        "Completely different sentence seven", "Completely different sentence eight",
        "Completely different sentence nine", "Completely different sentence ten",
        "Completely different sentence eleven", "Completely different sentence twelve",
        "Completely different sentence thirteen", "Completely different sentence fourteen",
    ])
    verdict = verify_linear_candidate(
        target_text=target_text, reference_text=format_srt(other),
        candidate_text=target_text,
        anchor_result=nucleus_detect(
            parse_srt(target_text), other,
            target_scores=classify_all(parse_srt(target_text)),
            reference_scores=classify_all(other)),
        reference_sha="x")
    assert not verdict.accepted
    assert any("insufficient_independent_groups" in reason
               for reason in verdict.failure_reasons)


def test_mistimed_opening_region_rejected() -> None:
    target_text, reference_text = _linear_pair(12)
    base = apply_transform(parse_srt(target_text),
                           LinearTransform(scale=1.0, offset_ms=5_000.0))
    bad = [
        Cue(number=c.number, order=c.order,
            start_ms=c.start_ms + (30_000 if c.number <= 4 else 0),
            end_ms=c.end_ms + (30_000 if c.number <= 4 else 0), text=c.text)
        for c in base
    ]
    _, _, detection = _detect(target_text, reference_text)
    verdict = verify_linear_candidate(
        target_text=target_text, reference_text=reference_text,
        candidate_text=format_srt(bad), anchor_result=detection,
        reference_sha=_sha(reference_text))
    assert not verdict.accepted
    assert any("opening" in reason for reason in verdict.failure_reasons)


def test_mistimed_ending_region_rejected() -> None:
    target_text, reference_text = _linear_pair(12)
    base = apply_transform(parse_srt(target_text),
                           LinearTransform(scale=1.0, offset_ms=5_000.0))
    bad = [
        Cue(number=c.number, order=c.order,
            start_ms=c.start_ms + (30_000 if c.number >= 9 else 0),
            end_ms=c.end_ms + (30_000 if c.number >= 9 else 0), text=c.text)
        for c in base
    ]
    _, _, detection = _detect(target_text, reference_text)
    verdict = verify_linear_candidate(
        target_text=target_text, reference_text=reference_text,
        candidate_text=format_srt(bad), anchor_result=detection,
        reference_sha=_sha(reference_text))
    assert not verdict.accepted
    assert any("ending" in reason for reason in verdict.failure_reasons)


def test_concealed_discontinuity_rejected() -> None:
    """Endpoints fit a global line but the middle follows another regime."""
    target = build_dialogue(14, step_ms=60_000)
    reference = apply_affine(target, 1.0, 5_000.0)
    shifted = [
        Cue(number=c.number, order=c.order,
            start_ms=c.start_ms + (45_000 if 5 <= position < 9 else 0),
            end_ms=c.end_ms + (45_000 if 5 <= position < 9 else 0), text=c.text)
        for position, c in enumerate(reference)
    ]
    target_text, shifted_text = format_srt(target), format_srt(shifted)
    _, _, detection = _detect(target_text, shifted_text)
    assert not detection.accepted
    assert "NONLINEAR_DISCONTINUITY" in detection.reasons
    # A candidate following only the endpoints must fail regional review.
    candidate = format_srt(apply_affine(target, 1.0, 5_000.0))
    verdict = verify_nonlinear_candidate(
        target_text=target_text, reference_text=shifted_text,
        candidate_text=candidate, anchor_result=detection,
        reference_sha=_sha(shifted_text))
    assert not verdict.accepted


def test_wrong_reference_similar_timestamps_insufficient() -> None:
    """Same timing, unrelated dialogue: no textual correspondence exists."""
    target_text, _ = _linear_pair(10)
    arabic = [f"جملة عربية مختلفة تماما رقم {position}" for position in range(10)]
    reference = [
        make_cue(position + 1, 60_000 + position * 60_000, 62_000 + position * 60_000, text)
        for position, text in enumerate(arabic)
    ]
    reference_text = format_srt(reference)
    _, _, detection = _detect(target_text, reference_text)
    assert not detection.accepted


def test_repeated_generic_dialogue_insufficient() -> None:
    def _fmt(ms: int) -> str:
        return f"{ms // 3_600_000:02d}:{(ms // 60_000) % 60:02d}:{(ms // 1_000) % 60:02d},{ms % 1000:03d}"

    first = "\n\n".join(
        f"{position + 1}\n{_fmt(60_000 + position * 60_000)} --> "
        f"{_fmt(62_000 + position * 60_000)}\n{['Yes', 'No'][position % 2]}"
        for position in range(8)) + "\n"
    second = "\n\n".join(
        f"{position + 1}\n{_fmt(65_000 + position * 60_000)} --> "
        f"{_fmt(67_000 + position * 60_000)}\n{['No', 'Yes'][position % 2]}"
        for position in range(8)) + "\n"
    _, _, detection = _detect(first, second)
    assert not detection.accepted


def test_overlapping_groups_counted_once() -> None:
    first = Anchor(target_number=5, reference_number=5, target_ms=100_000,
                   reference_ms=105_000, evidence=("x",), confidence=0.9,
                   target_numbers=(5, 6), reference_numbers=(5,))
    second = Anchor(target_number=5, reference_number=5, target_ms=100_000,
                    reference_ms=105_000, evidence=("x",), confidence=0.9,
                    target_numbers=(5,), reference_numbers=(5,))
    result = AnchorResult(accepted=False, reasons=("probe",),
                          checkpoints=(first, second))
    records = anchor_pairs(result)
    assert len(records) == 2  # distinct signatures: (5,6)x(5) vs (5)x(5)
    # Identical signatures collapse.
    result = AnchorResult(accepted=False, reasons=("probe",),
                          checkpoints=(second, second))
    assert len(anchor_pairs(result)) == 1


def test_held_out_excludes_fitting_anchors() -> None:
    target_text, reference_text = _linear_pair()
    detection = nucleus_detect(
        parse_srt(target_text), parse_srt(reference_text),
        target_scores=classify_all(parse_srt(target_text)),
        reference_scores=classify_all(parse_srt(reference_text)))
    assert detection.accepted
    from two_point_sync.anchors import Anchor as NucleusAnchor

    held = held_out_checkpoints(detection)
    assert held, "need held-out checkpoints"
    signatures = {(tuple(i.target_numbers or (i.target_number,)),
                   tuple(i.reference_numbers or (i.reference_number,))) for i in held}
    for primary in (detection.first_anchor, detection.last_anchor):
        assert primary is not None
        signature = (tuple(primary.target_numbers or (primary.target_number,)),
                     tuple(primary.reference_numbers or (primary.reference_number,)))
        assert signature not in signatures
    assert isinstance(held[0], NucleusAnchor)


def test_sparse_coverage_rejected() -> None:
    target_text, reference_text = _linear_pair(20, step_ms=30_000)
    _, _, detection = _detect(target_text, reference_text)
    assert detection.accepted
    # Candidate verified against opening-only pairs: other thirds uncovered.
    from two_point_sync.parser import parse_srt as nucleus_parse

    candidate = format_srt(apply_transform(
        nucleus_parse(target_text), LinearTransform(scale=1.0, offset_ms=5_000.0)))
    verdict = verify_linear_candidate(
        target_text=target_text, reference_text=reference_text,
        candidate_text=candidate,
        anchor_result=AnchorResult(
            accepted=True, first_anchor=detection.first_anchor,
            last_anchor=detection.last_anchor, scale=1.0, offset_ms=5_000.0,
            checkpoints=tuple(
                item for item in detection.checkpoints if item.target_ms < 250_000),
            reasons=("probe",)),
        reference_sha=_sha(reference_text))
    assert not verdict.accepted
    assert any("uncovered" in reason for reason in verdict.failure_reasons)


def test_isolated_outlier_listed_not_hidden() -> None:
    """One ~2.5s deviation among good pairs: accepted, but explicitly listed
    (Dexter T#795 analogue at correspondence level)."""
    target_text, reference_text = _linear_pair(12)
    base = apply_transform(parse_srt(target_text),
                           LinearTransform(scale=1.0, offset_ms=5_000.0))
    # Cue 6 is a held-out checkpoint (primaries 1 and 12 stay clean), so the
    # deviation is verified as listed evidence rather than hidden.
    bad = [
        Cue(number=c.number, order=c.order,
            start_ms=c.start_ms + (2_500 if c.number == 6 else 0),
            end_ms=c.end_ms + (2_500 if c.number == 6 else 0), text=c.text)
        for c in base
    ]
    _, _, detection = _detect(target_text, reference_text)
    verdict = verify_linear_candidate(
        target_text=target_text, reference_text=reference_text,
        candidate_text=format_srt(bad), anchor_result=detection,
        reference_sha=_sha(reference_text))
    assert verdict.accepted, verdict.failure_reasons
    assert verdict.outlier_count >= 1
    assert verdict.max_residual_ms <= 3000.0


def test_error_cluster_rejected() -> None:
    """Six cues ~2.5s off: systematic, median fails."""
    target_text, reference_text = _linear_pair(12)
    base = apply_transform(parse_srt(target_text),
                           LinearTransform(scale=1.0, offset_ms=5_000.0))
    bad = [
        Cue(number=c.number, order=c.order,
            start_ms=c.start_ms + (2_500 if c.number >= 7 else 0),
            end_ms=c.end_ms + (2_500 if c.number >= 7 else 0), text=c.text)
        for c in base
    ]
    _, _, detection = _detect(target_text, reference_text)
    verdict = verify_linear_candidate(
        target_text=target_text, reference_text=reference_text,
        candidate_text=format_srt(bad), anchor_result=detection,
        reference_sha=_sha(reference_text))
    assert not verdict.accepted


def test_non_monotonic_candidate_rejected() -> None:
    target_text, reference_text = _linear_pair(12)
    base = apply_transform(parse_srt(target_text),
                           LinearTransform(scale=1.0, offset_ms=5_000.0))
    bad = list(base)
    bad[5], bad[6] = bad[6], bad[5]
    reordered = [
        Cue(number=position + 1, order=position, start_ms=c.start_ms,
            end_ms=c.end_ms, text=c.text)
        for position, c in enumerate(bad)
    ]
    _, _, detection = _detect(target_text, reference_text)
    verdict = verify_linear_candidate(
        target_text=target_text, reference_text=reference_text,
        candidate_text=format_srt(reordered), anchor_result=detection,
        reference_sha=_sha(reference_text))
    assert not verdict.accepted


def test_missing_reordered_modified_dialogue_rejected() -> None:
    target_text, reference_text = _linear_pair(12)
    base = apply_transform(parse_srt(target_text),
                           LinearTransform(scale=1.0, offset_ms=5_000.0))
    cues = [c for c in base if c.number != 6]
    _, _, detection = _detect(target_text, reference_text)
    verdict = verify_linear_candidate(
        target_text=target_text, reference_text=reference_text,
        candidate_text=format_srt(cues), anchor_result=detection,
        reference_sha=_sha(reference_text))
    assert not verdict.accepted

    tampered = [
        Cue(number=c.number, order=c.order, start_ms=c.start_ms, end_ms=c.end_ms,
            text="tampered line here" if c.number == 3 else c.text) for c in cues
    ]
    verdict = verify_linear_candidate(
        target_text=target_text, reference_text=reference_text,
        candidate_text=format_srt(tampered), anchor_result=detection,
        reference_sha=_sha(reference_text))
    assert not verdict.accepted


def test_dialogue_clamping_vetoes_but_ad_clamping_flags() -> None:
    target = build_dialogue(8)
    target_text = format_srt(target)
    reference_text = format_srt(apply_affine(target, 1.0, 50_000.0))
    _, _, detection = _detect(target_text, reference_text)
    assert detection.accepted
    cues = parse_srt(target_text)

    def _move(number: int) -> str:
        moved = [
            Cue(number=c.number, order=c.order,
                start_ms=0 if c.number == number else c.start_ms + 50_000,
                end_ms=(c.end_ms - c.start_ms) if c.number == number else c.end_ms + 50_000,
                text=c.text)
            for c in cues
        ]
        return format_srt(moved)

    verdict = verify_linear_candidate(
        target_text=target_text, reference_text=reference_text,
        candidate_text=_move(1), anchor_result=detection,
        reference_sha=_sha(reference_text))
    assert not verdict.accepted
    assert any("dialogue_clamped" in reason for reason in verdict.failure_reasons)

    ad = [Cue(number=c.number, order=c.order, start_ms=c.start_ms, end_ms=c.end_ms,
              text="Visit www.example-subtitles.com" if c.number == 1 else c.text)
          for c in cues]
    ad_text = format_srt(ad)
    ad_ref = format_srt(apply_affine(ad, 1.0, 50_000.0))
    _, _, detection = _detect(ad_text, ad_ref)
    # Ten cues keep every third covered once the ineligible ad cue drops out.
    long_target = build_dialogue(10)
    long_text = format_srt(long_target)
    long_cues = parse_srt(long_text)
    long_ad = [Cue(number=c.number, order=c.order, start_ms=c.start_ms, end_ms=c.end_ms,
                   text="Visit www.example-subtitles.com" if c.number == 1 else c.text)
               for c in long_cues]
    long_ad_text = format_srt(long_ad)
    long_ad_ref = format_srt(apply_affine(long_ad, 1.0, 50_000.0))
    _, _, long_detection = _detect(long_ad_text, long_ad_ref)
    moved = [
        Cue(number=c.number, order=c.order,
            start_ms=0 if c.number == 1 else c.start_ms + 50_000,
            end_ms=(c.end_ms - c.start_ms) if c.number == 1 else c.end_ms + 50_000,
            text=c.text)
        for c in parse_srt(long_ad_text)
    ]
    verdict = verify_linear_candidate(
        target_text=long_ad_text, reference_text=long_ad_ref,
        candidate_text=format_srt(moved), anchor_result=long_detection,
        reference_sha=_sha(long_ad_ref))
    assert verdict.accepted, verdict.failure_reasons


def test_to_evaluation_never_rewrites_legacy() -> None:
    target_text, reference_text = _linear_pair()
    _, _, detection = _detect(target_text, reference_text)
    from two_point_sync.parser import parse_srt as nucleus_parse

    candidate = format_srt(apply_transform(
        nucleus_parse(target_text), LinearTransform(scale=1.0, offset_ms=5_000.0)))
    verdict = verify_linear_candidate(
        target_text=target_text, reference_text=reference_text,
        candidate_text=candidate, anchor_result=detection,
        reference_sha=_sha(reference_text))
    assert verdict.accepted
    from app.services.sync.alignment import SubtitleEvaluation

    legacy = SubtitleEvaluation()
    before = (legacy.sync_state, legacy.verification, tuple(legacy.reasons))
    fresh = to_evaluation(verdict)
    assert fresh is not legacy
    assert (legacy.sync_state, legacy.verification, tuple(legacy.reasons)) == before
    assert may_serve_synchronized(fresh.sync_state, fresh.verification)


def test_telemetry_has_no_dialogue_text() -> None:
    target_text, reference_text = _linear_pair()
    _, _, detection = _detect(target_text, reference_text)
    from two_point_sync.parser import parse_srt as nucleus_parse

    candidate = format_srt(apply_transform(
        nucleus_parse(target_text), LinearTransform(scale=1.0, offset_ms=5_000.0)))
    verdict = verify_linear_candidate(
        target_text=target_text, reference_text=reference_text,
        candidate_text=candidate, anchor_result=detection,
        reference_sha=_sha(reference_text))
    blob = repr((verdict.coverage_thirds, verdict.regional_discontinuities,
                 verdict.outliers, verdict.clamped, verdict.unmatched_reference_spans,
                 verdict.failure_reasons, verdict.transformation_model))
    assert "lighthouse" not in blob and "S08E01" not in blob


def test_dexter_t795_is_segmentation_overlap() -> None:
    if not DEX_TARGET.exists() or not DEX_REFERENCE.exists():
        pytest.skip("Dexter fixtures unavailable")
    from two_point_sync.parser import parse_srt_bytes

    reference = parse_srt_bytes(DEX_REFERENCE.read_bytes())
    # T#795 ("Ignore her", t=2531551; candidate 2541873) sits inside R#582
    # (2539101->2543063): a segmentation difference, with neighbours
    # supporting the candidate timeline.
    examination = examine_outlier(2541873, reference, 2539101)
    assert examination["kind"] == "within_reference_span"
    assert any(entry["number"] == 582 for entry in examination["inside_reference_spans"])
