"""Dexter S08E01 real-world divergence test (FraMeSToR reference).

Both files contain the "Previously on Dexter" recap, but the recap follows a
different timing regime (~-96.3s) than the main body (~+10.5s). The nucleus
must therefore return NONLINEAR_DISCONTINUITY for one global linear
transformation: no synced output is produced here.

This reference is NOT the user-confirmed ROVERS/PiR8 artifact; nothing here
asserts synchronization against the PiR8 video.
"""

from __future__ import annotations

import pathlib

import pytest

from two_point_sync.anchors import detect_anchors
from two_point_sync.parser import parse_srt_bytes
from two_point_sync.sanitizer import classify_all
from two_point_sync.transform import LinearTransform
from two_point_sync.validator import Checkpoint, validate_transform

FIXTURE_DIR = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "two_point_sync" / "dexter"
TARGET_NAME = "Dexter.S08E01.480p.HDTV.x264-mSD.srt"
REFERENCE_NAME = (
    "Dexter.S08E01.A.Beautiful.Day.BluRay.1080p.TrueHD.5.1.AVC.REMUX-FraMeSToR.ar.srt"
)

# Documented observations (test expectations only, derived by the nucleus).
EXPECTED_PRIMARY_OFFSET_SECONDS = 10.468
EXPECTED_RECAP_OFFSET_SECONDS = -96.346


def _load() -> tuple[object, object]:
    target_path = FIXTURE_DIR / TARGET_NAME
    reference_path = FIXTURE_DIR / REFERENCE_NAME
    if not target_path.exists() or not reference_path.exists():
        pytest.skip("Dexter fixtures unavailable")
    return parse_srt_bytes(target_path.read_bytes()), parse_srt_bytes(reference_path.read_bytes())


def test_dexter_grouped_recap_and_nonlinear_discontinuity() -> None:
    target, reference = _load()  # type: ignore[misc]
    assert len(target) == 921  # type: ignore[arg-type]
    assert len(reference) == 684  # type: ignore[arg-type]
    target_by_number = {cue.number: cue for cue in target}  # type: ignore[union-attr]
    reference_by_number = {cue.number: cue for cue in reference}  # type: ignore[union-attr]

    target_scores = classify_all(target)  # type: ignore[arg-type]
    reference_scores = classify_all(reference)  # type: ignore[arg-type]

    # Opening/closing ads and translator credits are never anchor candidates.
    assert not target_scores[0].eligible
    assert not target_scores[-1].eligible
    assert not reference_scores[-1].eligible

    # Verified segmentation example: Target cues #2+#3 are grouped inside
    # Reference cue #1 (offsets in milliseconds).
    assert reference_by_number[1].start_ms - target_by_number[2].start_ms == -96460
    # Verified regime examples: recap pair vs main-body pairs.
    assert reference_by_number[61].start_ms - target_by_number[82].start_ms == 10030
    assert reference_by_number[79].start_ms - target_by_number[104].start_ms == 10288

    result = detect_anchors(target, reference,  # type: ignore[arg-type]
                            target_scores=target_scores, reference_scores=reference_scores)

    # No single global linear transformation may be claimed.
    assert not result.accepted
    assert "NONLINEAR_DISCONTINUITY" in result.reasons
    assert result.first_anchor is None and result.last_anchor is None
    assert result.scale is None and result.offset_ms is None

    # Two incompatible timing regimes are reported across the full span.
    assert len(result.regimes) == 2
    primary, recap = result.regimes
    assert abs(primary.scale - 1.0) < 0.002
    assert abs(primary.offset_ms / 1000.0 - EXPECTED_PRIMARY_OFFSET_SECONDS) < 1.5
    assert primary.support >= 10
    assert abs(recap.implied_offset_ms / 1000.0 - EXPECTED_RECAP_OFFSET_SECONDS) < 5.0
    assert recap.support >= 3
    assert abs(recap.implied_offset_ms - primary.offset_ms) > 30_000
    assert recap.span_start_ms < 150_000
    assert recap.span_end_ms - recap.span_start_ms >= 60_000

    # The grouped recap correspondence Target #2+#3 <-> Reference #1 is
    # demonstrated with anchored evidence (marker seed + exact line match).
    groups = [
        item for item in result.contradictory
        if item.target_numbers == (2, 3) and item.reference_numbers == (1,)
    ]
    assert len(groups) == 1
    assert any("anchored_group_match" in part for part in groups[0].evidence)
    # Its timing agrees with the recap regime, not the primary one.
    assert abs(groups[0].target_ms - 106993) < 2000
    assert abs(groups[0].reference_ms - 10533) < 2000

    # Main-body checkpoints still validate at module level on their own ...
    main_model = LinearTransform(scale=primary.scale, offset_ms=primary.offset_ms)
    main_checkpoints = [
        Checkpoint(target_number=item.target_number, reference_number=item.reference_number,
                   target_ms=item.target_ms, reference_ms=item.reference_ms)
        for item in result.checkpoints
    ]
    assert len(main_checkpoints) >= 10
    film_start = min(cue.start_ms for cue in target)  # type: ignore[union-attr]
    film_end = max(cue.end_ms for cue in target)  # type: ignore[union-attr]
    accepted = validate_transform(main_model, main_checkpoints,
                                  span_start_ms=film_start, span_end_ms=film_end)
    assert accepted.accepted, accepted.reasons

    # ... but the validator must NOT accept while ignoring the contradictory
    # opening region it was told about.
    contra_checkpoints = [
        Checkpoint(target_number=item.target_number, reference_number=item.reference_number,
                   target_ms=item.target_ms, reference_ms=item.reference_ms)
        for item in result.contradictory
    ]
    rejected = validate_transform(main_model, main_checkpoints,
                                  span_start_ms=film_start, span_end_ms=film_end,
                                  contradictory=contra_checkpoints)
    assert not rejected.accepted
    assert "contradictory_region_rejected" in rejected.reasons
    assert rejected.has_discontinuity
