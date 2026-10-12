"""Offline ALASS nonlinear-handling regression test (synthetic pair).

Runs the existing ALASS binary directly (never the live service) on a small
synthetic pair with a mid-file timing discontinuity, then validates the
output independently with the isolated Two-Point nucleus modules.

Skips when no ALASS binary is available, following the repository convention
(``NINJASUBS_TEST_ALASS`` override or ``alass`` on PATH).
"""

from __future__ import annotations

import os
import pathlib
import shutil
import statistics
import subprocess

import pytest

from tests.two_point_sync._helpers import apply_affine, build_dialogue
from two_point_sync.anchors import normalize_text
from two_point_sync.parser import Cue, format_srt, parse_srt, parse_srt_bytes
from two_point_sync.transform import LinearTransform
from two_point_sync.validator import Checkpoint, validate_transform


def _alass_binary() -> str:
    binary = os.getenv("NINJASUBS_TEST_ALASS") or shutil.which("alass")
    if binary is None:
        pytest.skip("alass not available in this environment")
    return binary


def test_alass_handles_mid_file_discontinuity_offline(tmp_path: pathlib.Path) -> None:
    binary = _alass_binary()
    target = build_dialogue(14, step_ms=60_000)
    reference = apply_affine(target, 1.0, 5_000.0)
    shifted = [
        Cue(number=cue.number, order=cue.order,
            start_ms=cue.start_ms + (45_000 if 5 <= position < 9 else 0),
            end_ms=cue.end_ms + (45_000 if 5 <= position < 9 else 0),
            text=cue.text)
        for position, cue in enumerate(reference)
    ]
    ref_path = tmp_path / "ref.srt"
    tgt_path = tmp_path / "tgt.srt"
    out_path = tmp_path / "out.srt"
    # Reference carries the true shifted times of the nonlinear pair.
    ref_path.write_text(format_srt(shifted), encoding="utf-8")
    tgt_path.write_text(format_srt(target), encoding="utf-8")

    completed = subprocess.run(
        [binary, str(ref_path), str(tgt_path), str(out_path), "--split-penalty", "7.0"],
        capture_output=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")[-500:]
    assert out_path.exists() and out_path.stat().st_size > 0

    synced = parse_srt_bytes(out_path.read_bytes())
    assert len(synced) == len(target)
    # Dialogue text preserved cue-for-cue.
    assert [normalize_text(c.text) for c in synced] == [
        normalize_text(c.text) for c in target
    ]
    # Independent check: every exact correspondence lands near its reference.
    ref_by_text = {normalize_text(c.text): c.start_ms for c in shifted}
    residuals = [
        abs(c.start_ms - ref_by_text[normalize_text(c.text)]) for c in synced
    ]
    assert statistics.median(residuals) < 1500.0
    # And the Two-Point validator accepts the aligned middle checkpoints
    # under an identity residual audit (all residuals within policy).
    checkpoints = [
        Checkpoint(target_number=c.number, reference_number=c.number,
                   target_ms=c.start_ms,
                   reference_ms=ref_by_text[normalize_text(c.text)])
        for c in synced
    ]
    report = validate_transform(LinearTransform(scale=1.0, offset_ms=0.0), checkpoints)
    assert report.accepted, report.reasons
    assert parse_srt(out_path.read_text(encoding="utf-8")) == synced
