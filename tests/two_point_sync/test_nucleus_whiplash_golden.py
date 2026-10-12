"""Whiplash (2014) golden benchmark for the Two-Point Auto-Sync nucleus.

End-to-end pipeline over the ORIGINAL supplied fixtures (byte-for-byte
unchanged)::

    WEBRip.iTunes Arabic subtitle (Target) + REMUX BluRay subtitle (Reference)

The expected scale (~1.042755) and offset (~-8.622s) below are TEST
expectations only: every parameter is derived independently by the nucleus
from the original inputs. Nothing is hardcoded in production code.
"""

from __future__ import annotations

import pathlib

import pytest

from two_point_sync.anchors import detect_anchors
from two_point_sync.parser import format_srt, parse_srt_bytes
from two_point_sync.sanitizer import classify_all, visible_text
from two_point_sync.transform import apply_transform, fit_from_anchors
from two_point_sync.validator import Checkpoint, validate_transform

FIXTURE_DIR = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "whiplash"
TARGET_NAME = "Whiplash (2014) WEBRip.iTunes.ar(1).srt"
REFERENCE_NAME = "Whiplash.2014.1080p.REMUX.BluRay.AVC.DTS.HD.MA.5.1-HDA(1).srt"

EXPECTED_SCALE = 1.042755
EXPECTED_OFFSET_SECONDS = -8.622
SCALE_TOLERANCE = 0.0005
OFFSET_TOLERANCE_SECONDS = 0.75


def _load_fixtures() -> tuple[pathlib.Path, pathlib.Path]:
    target_path = FIXTURE_DIR / TARGET_NAME
    reference_path = FIXTURE_DIR / REFERENCE_NAME
    if not target_path.exists() or not reference_path.exists():
        pytest.skip("Whiplash original fixtures unavailable")
    return target_path, reference_path


def test_whiplash_golden_end_to_end(tmp_path: pathlib.Path) -> None:
    target_path, reference_path = _load_fixtures()

    # -- Parse ------------------------------------------------------------
    target = parse_srt_bytes(target_path.read_bytes())
    reference = parse_srt_bytes(reference_path.read_bytes())
    assert len(target) == 825
    assert len(reference) == 960

    # -- Anchor sanitization ----------------------------------------------
    target_scores = classify_all(target)
    reference_scores = classify_all(reference)
    target_by_text = {visible_text(cue.text): score for cue, score in zip(target, target_scores, strict=False)}
    assert target_by_text["3، 4"].eligible is False
    assert sum(1 for score in target_scores if score.eligible) > 700

    # -- Correspondence detection ------------------------------------------
    result = detect_anchors(target, reference,
                            target_scores=target_scores, reference_scores=reference_scores)
    assert result.accepted
    assert result.first_anchor is not None and result.last_anchor is not None

    # Anchors must be genuine dialogue: no advertisement, credit, watermark
    # or counting fragment may serve as a primary anchor.
    for anchor in (result.first_anchor, result.last_anchor):
        combined_evidence = ";".join(anchor.evidence)
        assert "advertisement" not in combined_evidence
        assert "credit" not in combined_evidence
        assert "watermark" not in combined_evidence
        assert "counting" not in combined_evidence
    forbidden_texts = {"3، 4", "3، 4.", "115", "OzOz"}
    target_by_number = {cue.number: cue for cue in target}
    reference_by_number = {cue.number: cue for cue in reference}
    anchor_texts = set()
    for anchor in (result.first_anchor, result.last_anchor):
        anchor_texts.add(visible_text(target_by_number[anchor.target_number].text))
        anchor_texts.add(visible_text(reference_by_number[anchor.reference_number].text))
    assert not (anchor_texts & forbidden_texts)

    # First anchor early in the film, terminal anchor in the known final
    # dialogue region (external diagnostic evidence, asserted as regions).
    film_start = min(cue.start_ms for cue in target)
    film_end = max(cue.end_ms for cue in target)
    assert result.first_anchor.target_ms - film_start < 600_000
    assert abs(result.last_anchor.target_ms - 5_610_800) < 5_000
    assert abs(result.last_anchor.reference_ms - 5_842_069) < 5_000
    assert result.temporal_separation_ms > 0.8 * (film_end - film_start)
    assert len(result.checkpoints) >= 10

    # -- Linear transform ---------------------------------------------------
    assert result.scale is not None and result.offset_ms is not None
    assert abs(result.scale - EXPECTED_SCALE) <= SCALE_TOLERANCE
    assert abs(result.offset_ms / 1000.0 - EXPECTED_OFFSET_SECONDS) <= OFFSET_TOLERANCE_SECONDS
    transform = fit_from_anchors(
        result.first_anchor.target_ms, result.first_anchor.reference_ms,
        result.last_anchor.target_ms, result.last_anchor.reference_ms,
    )
    assert abs(transform.scale - result.scale) < 1e-12

    # -- Independent validation ---------------------------------------------
    checkpoints = [
        Checkpoint(target_number=item.target_number, reference_number=item.reference_number,
                   target_ms=item.target_ms, reference_ms=item.reference_ms)
        for item in result.checkpoints
    ]
    report = validate_transform(transform, checkpoints,
                                span_start_ms=film_start, span_end_ms=film_end)
    assert report.accepted, report.reasons
    # Comparison targets from the earlier benchmark (not hardcoded derivations).
    assert report.median_abs_ms <= 500.0
    assert report.max_abs_ms <= 3000.0
    for region in ("beginning", "middle", "ending"):
        assert report.region_counts[region] >= 2
        assert report.region_medians_ms[region] <= 1000.0

    # -- Synchronized output --------------------------------------------------
    synced = apply_transform(target, transform)
    assert len(synced) == len(target) == 825
    for original, converted in zip(target, synced, strict=False):
        assert converted.text == original.text
        assert converted.number == original.number
    # Final dialogue timing consistent with the REMUX reference.
    assert abs(synced[-1].start_ms - result.last_anchor.reference_ms) <= 1_500

    output_path = tmp_path / "whiplash.two-point.synced.srt"
    output_path.write_text(format_srt(synced), encoding="utf-8")
    report_path = tmp_path / "whiplash.two-point.report.txt"
    report_path.write_text(
        "Two-Point Auto-Sync validation report (Whiplash 2014)\n"
        f"first_anchor: target#{result.first_anchor.target_number} "
        f"t={result.first_anchor.target_ms} "
        f"reference#{result.first_anchor.reference_number} "
        f"r={result.first_anchor.reference_ms} "
        f"evidence={';'.join(result.first_anchor.evidence)}\n"
        f"last_anchor: target#{result.last_anchor.target_number} "
        f"t={result.last_anchor.target_ms} "
        f"reference#{result.last_anchor.reference_number} "
        f"r={result.last_anchor.reference_ms} "
        f"evidence={';'.join(result.last_anchor.evidence)}\n"
        f"scale={transform.scale:.6f} offset_s={transform.offset_seconds:.3f}\n"
        f"checkpoints={report.n_checkpoints} "
        f"median_abs_ms={report.median_abs_ms:.1f} "
        f"max_abs_ms={report.max_abs_ms:.1f} p90_abs_ms={report.p90_abs_ms:.1f}\n"
        f"regions={report.region_medians_ms} counts={report.region_counts}\n"
        f"verdict={'ACCEPTED' if report.accepted else 'REJECTED'}\n",
        encoding="utf-8",
    )
    assert output_path.exists() and report_path.exists()
