"""Tests for two_point_sync.anchors (core correspondence discovery)."""

from __future__ import annotations

from tests.two_point_sync._helpers import apply_affine, build_dialogue, make_cue
from two_point_sync.anchors import AnchorResult, detect_anchors, normalize_text
from two_point_sync.parser import Cue


def _reference(
    target: list[Cue],
    scale: float,
    offset_ms: float,
    *,
    drop: set[int] | None = None,
    prefix: list[Cue] | None = None,
    suffix: list[Cue] | None = None,
) -> list[Cue]:
    ref = apply_affine(target, scale, offset_ms)
    if drop:
        ref = [cue for cue in ref if cue.number not in drop]
    out: list[Cue] = []
    number = 1
    for cue in (prefix or []) + ref + (suffix or []):
        out.append(Cue(number=number, order=number - 1, start_ms=cue.start_ms,
                       end_ms=cue.end_ms, text=cue.text))
        number += 1
    return out


def test_normalize_text_arabic_variants() -> None:
    assert normalize_text("أحمد") == normalize_text("احمد")
    assert normalize_text("مدرسة") == normalize_text("مدرسه")
    assert normalize_text("  مرحبا،  يا  صديقي! ") == normalize_text("مرحبا يا صديقي")


def test_normalize_strips_tags_and_bidi() -> None:
    assert normalize_text("{\\an8}مرحبا") == normalize_text("‫مرحبا")
    assert normalize_text("<i>مرحبا</i>") == normalize_text("مرحبا")


def test_identity_timing_recovered() -> None:
    target = build_dialogue(12)
    reference = _reference(target, 1.0, 0.0)
    result = detect_anchors(target, reference)
    assert result.accepted
    assert result.scale == 1.0
    assert result.offset_ms == 0.0
    assert result.first_anchor is not None and result.last_anchor is not None
    assert result.temporal_separation_ms > 0


def test_pure_offset_recovered() -> None:
    target = build_dialogue(12)
    reference = _reference(target, 1.0, 10_000.0)
    result = detect_anchors(target, reference)
    assert result.accepted
    assert abs(result.scale - 1.0) < 1e-9
    assert abs(result.offset_ms - 10_000.0) < 2.0


def test_negative_offset_recovered() -> None:
    target = build_dialogue(12, start_ms=120_000)
    reference = _reference(target, 1.0, -45_000.0)
    result = detect_anchors(target, reference)
    assert result.accepted
    assert abs(result.offset_ms - (-45_000.0)) < 2.0


def test_progressive_drift_recovered() -> None:
    target = build_dialogue(16, step_ms=60_000)
    reference = _reference(target, 25.0 / 23.976, -8_000.0)
    result = detect_anchors(target, reference)
    assert result.accepted
    assert abs(result.scale - 25.0 / 23.976) < 5e-6
    assert abs(result.offset_ms - (-8_000.0)) < 5.0


def test_combined_offset_and_drift_recovered() -> None:
    target = build_dialogue(16, step_ms=45_000)
    reference = _reference(target, 1.042755, -8_622.0)
    result = detect_anchors(target, reference)
    assert result.accepted
    assert abs(result.scale - 1.042755) < 1e-6
    assert abs(result.offset_ms - (-8_622.0)) < 5.0


def test_missing_cues_and_different_counts_tolerated() -> None:
    target = build_dialogue(16, step_ms=45_000)
    ads = [make_cue(999, 5_000, 9_000, "Download subtitles at www.example-subs.com")]
    credits = [make_cue(999, 5_000_000, 5_010_000, "Translated by Example Team")]
    reference = _reference(target, 1.0, 7_500.0, drop={3, 7, 11}, prefix=ads, suffix=credits)
    result = detect_anchors(target, reference)
    assert result.accepted
    assert abs(result.scale - 1.0) < 1e-6
    assert abs(result.offset_ms - 7_500.0) < 5.0
    assert len(result.checkpoints) >= 2


def test_repeated_generic_dialogue_does_not_corrupt_model() -> None:
    target = build_dialogue(14, step_ms=60_000)
    # Inject a repeated generic phrase at several positions.
    target = [
        make_cue(cue.number, cue.start_ms, cue.end_ms, "Yes" if cue.number in (4, 8, 12) else cue.text)
        for cue in target
    ]
    reference = _reference(target, 1.0, 5_000.0)
    result = detect_anchors(target, reference)
    assert result.accepted
    assert abs(result.scale - 1.0) < 1e-6
    assert abs(result.offset_ms - 5_000.0) < 5.0


def test_advertising_and_credits_never_become_anchors() -> None:
    target = build_dialogue(12, step_ms=60_000)
    ad = make_cue(999, 1_000, 4_000, "Visit www.free-subtitles.example for more")
    credit = make_cue(999, 800_000, 802_000, "3, 4")
    target_with_edges = [ad] + target + [credit]
    reference = _reference(target, 1.0, 3_000.0)
    result = detect_anchors(target_with_edges, reference)
    assert result.accepted
    assert result.first_anchor is not None and result.last_anchor is not None
    by_start = {cue.start_ms: cue.text for cue in target_with_edges}
    first_text = by_start[result.first_anchor.target_ms]
    last_text = by_start[result.last_anchor.target_ms]
    assert "www.free-subtitles" not in first_text
    assert last_text.strip() != "3, 4"


def test_wrong_terminal_candidate_rejected() -> None:
    target = build_dialogue(12, step_ms=60_000)
    reference = _reference(target, 1.0, 4_000.0)
    # Append an unrelated dialogue tail to the reference that mimics a
    # plausible-but-wrong endpoint (different text, inconsistent timing).
    wrong_tail = make_cue(999, 4_000_000, 4_002_000, "A completely unrelated closing remark")
    reference = reference + [wrong_tail]
    result = detect_anchors(target, reference)
    assert result.accepted
    assert result.last_anchor is not None
    assert result.last_anchor.reference_number != wrong_tail.number
    assert abs(result.scale - 1.0) < 1e-6


def test_different_segmentation_tolerated() -> None:
    target = build_dialogue(12, step_ms=60_000)
    reference = _reference(target, 1.0, 6_000.0)
    # Merge two reference cues into one re-segmented cue (combined text).
    merged_text = reference[4].text + " " + reference[5].text
    merged = Cue(number=reference[4].number, order=reference[4].order,
                 start_ms=reference[4].start_ms, end_ms=reference[5].end_ms,
                 text=merged_text)
    reference = reference[:4] + [merged] + reference[6:]
    result = detect_anchors(target, reference)
    assert result.accepted
    assert abs(result.scale - 1.0) < 1e-6
    assert abs(result.offset_ms - 6_000.0) < 5.0


def test_insufficient_evidence_fails_safely() -> None:
    target = build_dialogue(4)
    other = build_dialogue(4, sentences=[
        "Completely different sentence one",
        "Completely different sentence two",
        "Completely different sentence three",
        "Completely different sentence four",
    ])
    result = detect_anchors(target, other)
    assert not result.accepted
    assert result.first_anchor is None
    assert "insufficient_correspondence_evidence" in result.reasons


def test_cross_language_without_correspondence_fails_safely() -> None:
    target = build_dialogue(10)
    arabic = [
        "مرحبا بك في هذه المدينة الجميلة",
        "ذهبنا إلى السوق في الصباح الباكر",
        "كان الطقس حارا جدا ذلك اليوم",
        "اجتمع الأصدقاء في المساء",
        "قرأ الكتاب حتى منتصف الليل",
        "سافرنا بالقطار إلى العاصمة",
        "أعدت الأم وجبة لذيذة",
        "لعب الأطفال في الحديقة",
        "اشترى الخبز من المخبز",
        "شاهدنا الغروب من الشاطئ",
    ]
    reference = [
        make_cue(position + 1, 60_000 + position * 30_000, 62_000 + position * 30_000, text)
        for position, text in enumerate(arabic)
    ]
    result = detect_anchors(target, reference)
    assert not result.accepted
    assert result.first_anchor is None


def test_detection_is_deterministic() -> None:
    target = build_dialogue(16, step_ms=45_000)
    reference = _reference(target, 1.042755, -8_622.0, drop={5, 9})
    first = detect_anchors(target, reference)
    second = detect_anchors(target, reference)
    assert first.accepted and second.accepted
    assert first.scale == second.scale
    assert first.offset_ms == second.offset_ms
    assert first.first_anchor == second.first_anchor
    assert first.last_anchor == second.last_anchor


def _has_evidence(result: AnchorResult, fragment: str) -> bool:
    anchors = list(result.checkpoints) + list(result.contradictory)
    if result.first_anchor is not None:
        anchors.append(result.first_anchor)
    if result.last_anchor is not None:
        anchors.append(result.last_anchor)
    return any(fragment in part for item in anchors for part in item.evidence)


def test_grouped_one_to_two_segmentation_found() -> None:
    # Adjacent target cues (small gaps) jointly match one reference cue.
    target = build_dialogue(12, step_ms=5_000, duration_ms=2_000)
    reference = _reference(target, 1.0, 6_000.0)
    merged_text = reference[4].text + " " + reference[5].text
    merged = Cue(number=reference[4].number, order=reference[4].order,
                 start_ms=reference[4].start_ms, end_ms=reference[5].end_ms,
                 text=merged_text)
    reference = reference[:4] + [merged] + reference[6:]
    result = detect_anchors(target, reference)
    assert result.accepted
    assert abs(result.scale - 1.0) < 1e-6
    assert abs(result.offset_ms - 6_000.0) < 5.0
    assert _has_evidence(result, "grouped_exact_match:2v1")


def test_grouped_two_to_one_split_target_found() -> None:
    # One reference cue matches two adjacent target cues (N:1 direction).
    sentences = [
        "The lighthouse keeper arrives at dawn with the tide",
        "We missed the last train to Alexandria",
        "Her brother repairs old transistor radios",
        "The committee postponed the hearing again",
        "A sandstorm buried the northern highway",
        "He traded his watch for a paper map",
        "The orchestra tuned their instruments quietly",
        "She keeps a garden behind the bakery",
    ]
    target: list[Cue] = []
    for position, text in enumerate(sentences):
        if position == 4:
            base = 60_000 + position * 5_000
            target.append(make_cue(5, base, base + 2_000, "A sandstorm buried the"))
            target.append(make_cue(6, base + 2_500, base + 4_500, "northern highway"))
        else:
            number = position + 1 if position < 4 else position + 2
            base = 60_000 + position * 5_000
            target.append(make_cue(number, base, base + 2_000, text))
    reference = _reference(
        [cue for cue in target if cue.number != 6], 1.0, 6_000.0
    )
    # Reference cue 5 carries the combined segmentation of target cues 5+6.
    reference = [
        make_cue(cue.number, cue.start_ms, cue.end_ms,
                 "A sandstorm buried the northern highway" if cue.number == 5 else cue.text)
        for cue in reference
    ]
    assert normalize_text(reference[4].text) == normalize_text(
        target[4].text + " " + target[5].text
    )
    result = detect_anchors(target, reference)
    assert result.accepted
    assert abs(result.scale - 1.0) < 1e-6
    assert _has_evidence(result, "grouped_exact_match:2v1")


def test_dash_led_lines_match_across_segmentation() -> None:
    # Two dash-led speaker lines in one target cue match two reference cues.
    target = build_dialogue(12, step_ms=60_000)
    two_voices = make_cue(
        6, target[5].start_ms, target[5].end_ms,
        "- The ferry crossing was delayed by fog\n- She keeps a garden behind the bakery",
    )
    target = target[:5] + [two_voices] + target[6:]
    line_a = make_cue(100, target[5].start_ms + 4_000, target[5].start_ms + 5_000,
                      "The ferry crossing was delayed by fog")
    line_b = make_cue(101, target[5].start_ms + 5_500, target[5].start_ms + 7_000,
                      "She keeps a garden behind the bakery")
    reference = _reference([cue for cue in target if cue.number != 6], 1.0, 4_000.0)
    reference = reference[:5] + [line_a, line_b] + reference[5:]
    result = detect_anchors(target, reference)
    assert result.accepted
    assert abs(result.scale - 1.0) < 1e-6
    assert _has_evidence(result, "cue_line_match")


def test_wrapped_lines_stay_single_event() -> None:
    # Plain wrapped lines (no speaker dashes) are one event, not two.
    target = build_dialogue(12, step_ms=60_000)
    wrapped = make_cue(
        6, target[5].start_ms, target[5].end_ms,
        "The ferry crossing was delayed by\nfog and heavy rain",
    )
    target = target[:5] + [wrapped] + target[6:]
    reference = _reference(target, 1.0, 4_000.0)
    result = detect_anchors(target, reference)
    assert result.accepted
    assert not _has_evidence(result, "cue_line_match")


def test_weak_generic_terminal_anchor_rejected() -> None:
    # A trailing generic singleton must not become the terminal anchor.
    target = build_dialogue(12, step_ms=60_000)
    target = [
        make_cue(cue.number, cue.start_ms, cue.end_ms, "Thanks" if cue.number == 12 else cue.text)
        for cue in target
    ]
    reference = _reference(target, 1.0, 5_000.0)
    result = detect_anchors(target, reference)
    assert result.accepted
    assert result.first_anchor is not None and result.last_anchor is not None
    assert result.first_anchor.target_number == 1
    assert result.last_anchor.target_number == 11
    assert abs(result.scale - 1.0) < 1e-6
    assert abs(result.offset_ms - 5_000.0) < 5.0


def test_two_regimes_return_nonlinear_discontinuity() -> None:
    # Four distinctive middle cues follow a +45s regime inside a +5s film.
    target = build_dialogue(14, step_ms=60_000)
    reference = apply_affine(target, 1.0, 5_000.0)
    shifted = [
        Cue(number=cue.number, order=cue.order,
            start_ms=cue.start_ms + (45_000 if 5 <= position < 9 else 0),
            end_ms=cue.end_ms + (45_000 if 5 <= position < 9 else 0),
            text=cue.text)
        for position, cue in enumerate(reference)
    ]
    result = detect_anchors(target, shifted)
    assert not result.accepted
    assert "NONLINEAR_DISCONTINUITY" in result.reasons
    assert result.first_anchor is None and result.last_anchor is None
    assert result.scale is None and result.offset_ms is None
    assert len(result.regimes) == 2
    primary, alt = result.regimes
    assert abs(primary.offset_ms - 5_000.0) < 1_500.0
    assert abs(alt.implied_offset_ms - primary.offset_ms) > 30_000
    assert len(result.contradictory) >= 3
