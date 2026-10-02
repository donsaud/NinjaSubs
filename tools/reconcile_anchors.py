"""PART 1: reconcile the large-offset evidence with the DP's anchors.

Answers, per anchor record: what the production gate saw, what the DP sees, and
whether the anchor survives the DP path. Nothing here is Dexter-specific; the
case is chosen by argument position and the fixtures are read from disk.

Run: python tools/reconcile_anchors.py
"""

from __future__ import annotations

import os
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.subtitle_matcher import parse_srt_cues  # noqa: E402
from app.services.sync.anchor_correspondence import (  # noqa: E402
    build_mapping,
    collect_anchor_regions,
)
from app.services.sync.interval_correspondence import (  # noqa: E402
    align_intervals,
    anchors_consistent,
)
from app.services.sync.large_offset import (  # noqa: E402
    LARGE_OFFSET_ANCHOR_TOLERANCE_MS,
    LARGE_OFFSET_CUES_PER_REGION,
    LARGE_OFFSET_MAX_DISPERSION_MS,
    LARGE_OFFSET_REGION_COUNT,
    _median,
    _region_medians,
    assess_large_offset,
)

#: Working diagnostics folder for this experiment (Dexter S08E04 working
#: files). Deliberately NOT committed: point NINJASUBS_DIAGNOSTICS_DIR at your
#: own copy. The committed equivalents the test suite uses live in
#: tests/fixtures/large_offset/.
_DIAGNOSTICS_DIR = os.environ.get("NINJASUBS_DIAGNOSTICS_DIR", "")
if not _DIAGNOSTICS_DIR:
    raise SystemExit(
        "Set NINJASUBS_DIAGNOSTICS_DIR to the folder holding this experiment's "
        "input files (01_EVOLV_original.srt, 03_ASAP_original.srt, "
        "05_PiR8_reference.srt, ...). They are not committed; see "
        "tests/fixtures/large_offset/ for the committed equivalents."
    )
FIXTURES = pathlib.Path(_DIAGNOSTICS_DIR)


def cues(name: str) -> list[tuple[int, int, str]]:
    text = (FIXTURES / name).read_text(encoding="utf-8", errors="replace")
    return [(int(s), int(e), "") for s, e, _ in parse_srt_cues(text)]


def anchor_records(target, reference_starts, hypothesis):
    """Every (position, target_start, offset) the production sampler can produce."""
    regions, _, samples = _region_medians(target, reference_starts, hypothesis)
    return regions, samples


def mad(values):
    if not values:
        return float("nan")
    med = statistics.median(values)
    return statistics.median([abs(v - med) for v in values])


def analyse(label, target, reference, seed):
    print("=" * 96)
    print(f"{label}   seed hypothesis = {seed}ms")
    print("=" * 96)
    ref_starts = [s for s, _, _ in reference]

    # ---- A: the production gate, run exactly as the analyzer runs it -------- #
    assessment = assess_large_offset(
        target,
        reference,
        seed_offset_ms=int(seed),
        identity_supported=True,
    )
    print("\n  A. LARGE-OFFSET GATE (production logic, two passes)")
    print(f"     accepted                 : {assessment.accepted}")
    print(f"     reason codes             : {assessment.reason_codes}")
    print(f"     anchor_count             : {assessment.anchor_count} "
          f"(min {4})")
    print(f"     regions_sampled          : {assessment.regions_sampled}")
    print(f"     estimated_offset_ms      : {assessment.estimated_offset_ms}")
    print(f"     offset_dispersion_ms     : {assessment.offset_dispersion_ms} "
          f"(max {LARGE_OFFSET_MAX_DISPERSION_MS})")
    print(f"     drift_ms_per_minute      : {assessment.drift_ms_per_minute}")
    print(f"     structural_similarity    : {assessment.structural_similarity}")

    # ---- B: anchor-guided correspondence ---------------------------------- #
    ag_regions = collect_anchor_regions(target, ref_starts, seed)
    ag_map = build_mapping(ag_regions)
    ag_offsets = [r.offset_ms for r in ag_regions]
    print("\n  B. ANCHOR-GUIDED CORRESPONDENCE (one pass, raw seed)")
    print(f"     regions with matches     : {len(ag_regions)}")
    print(f"     region medians           : {[round(o, 1) for o in ag_offsets]}")
    print(f"     median absolute dev      : {round(mad(ag_offsets), 1)}")
    print(f"     mapping pieces           : {ag_map.piece_count}")
    print(f"     consecutive differences  : "
          f"{[round(ag_offsets[i+1]-ag_offsets[i], 1) for i in range(len(ag_offsets)-1)]}")

    # ---- C: the monotonic DP ------------------------------------------------ #
    ok, bad = anchors_consistent(ag_map, 1500.0)
    dp = align_intervals(target, reference, seed)
    print("\n  C. MONOTONIC DP")
    print(f"     anchor consistency(1500) : ok={ok} bad_boundaries={bad}")
    print(f"     outcome                  : {dp.outcome.value}")
    print(f"     reference coverage       : {dp.reference_coverage:.3f}")
    print(f"     p95                      : {dp.p95():.0f}ms")

    # ---- the reconciliation ------------------------------------------------ #
    print("\n  RECONCILIATION")
    p1_regions, p1_samples = anchor_records(target, ref_starts, seed)
    p1_offsets = [v for _, v in p1_regions]
    refined = _median(p1_offsets) if p1_offsets else seed
    p2_regions, p2_samples = anchor_records(target, ref_starts, refined)
    p2_offsets = [v for _, v in p2_regions]
    print(f"     pass 1 hypothesis        : {seed}  -> {len(p1_offsets)} medians "
          f"{[round(o, 1) for o in p1_offsets]}  MAD {mad(p1_offsets):.1f}")
    print(f"     refined hypothesis       : {refined:.1f}")
    print(f"     pass 2 hypothesis        : {refined:.1f}  -> {len(p2_offsets)} "
          f"medians {[round(o, 1) for o in p2_offsets]}  MAD {mad(p2_offsets):.1f}")
    print("     production uses pass 2; DP/anchor-guided use pass 1")
    consecutive = [
        abs(p2_offsets[i + 1] - p2_offsets[i]) for i in range(len(p2_offsets) - 1)
    ]
    print(f"     pass-2 consecutive diffs: {[round(c, 1) for c in consecutive]} "
          f"(max {max(consecutive) if consecutive else float('nan'):.1f})")

    # ---- per-anchor survival through the DP -------------------------------- #
    print("\n  PER-ANCHOR SURVIVAL (pass 2 anchors vs the DP's chosen mapping)")
    survivors = 0
    for position, offset in p2_regions:
        # which reference cue did this anchor imply?
        implied_ref = position - offset
        nearest = min(
            range(len(ref_starts)),
            key=lambda i: abs(ref_starts[i] - implied_ref),
        )
        matched = any(
            nearest in g.reference_indices for g in dp.groups
        )
        if matched:
            survivors += 1
        print(f"     t={position:>9} offset={offset:>9.1f} implied_ref={implied_ref:>9}"
              f" ref_cue={nearest:>4} in_dp={matched}")
    print(f"     anchors surviving DP     : {survivors}/{len(p2_regions)}")

    # ambiguity: is a competing reference cue nearby?
    amb = 0
    for position, offset in p2_regions:
        implied_ref = position - offset
        near = [
            i for i, s in enumerate(ref_starts)
            if abs(s - implied_ref) <= LARGE_OFFSET_ANCHOR_TOLERANCE_MS
        ]
        if len(near) > 1:
            amb += 1
    print(f"     anchors with >1 candidate reference cue within tolerance: {amb}"
          f"/{len(p2_regions)}")
    print()
    return dict(
        label=label,
        accepted=assessment.accepted,
        gate_dispersion=assessment.offset_dispersion_ms,
        gate_offset=assessment.estimated_offset_ms,
        pass1_offsets=[round(o, 1) for o in p1_offsets],
        pass1_mad=round(mad(p1_offsets), 1),
        pass2_offsets=[round(o, 1) for o in p2_offsets],
        pass2_mad=round(mad(p2_offsets), 1),
        refined_hypothesis=round(refined, 1),
        pass2_consecutive=[round(c, 1) for c in consecutive],
        dp_outcome=dp.outcome.value,
        dp_coverage=round(dp.reference_coverage, 4),
        survivors=survivors,
        anchors=len(p2_regions),
    )


def main() -> int:
    ref = cues("05_PiR8_reference.srt")
    out = []
    out.append(analyse("EVOLV original", cues("01_EVOLV_original.srt"), ref, 96100.0))
    out.append(analyse("ASAP original", cues("03_ASAP_original.srt"), ref, 96050.0))
    print(
        f"  CONSTANTS: regions={LARGE_OFFSET_REGION_COUNT} "
        f"cues_per_region={LARGE_OFFSET_CUES_PER_REGION} "
        f"tolerance={LARGE_OFFSET_ANCHOR_TOLERANCE_MS}ms "
        f"max_dispersion={LARGE_OFFSET_MAX_DISPERSION_MS:.0f}ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
