"""PART 2 diagnostic: why does ASAP stay at p95 ~2277ms?

Measures, per episode quarter, the quantities needed to attribute the residual
to a specific mechanism. Reports no conclusions the numbers do not support.

Run:  python tools/diagnose_asap_residual.py
"""

from __future__ import annotations

import json
import os
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.subtitle_matcher import parse_srt_cues  # noqa: E402
from app.services.sync.anchor_correspondence import (  # noqa: E402
    anchor_guided_correspondence,
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


def cues(path: pathlib.Path):
    parsed = parse_srt_cues(
        path.read_text(encoding="utf-8", errors="replace")
    )
    return [(int(s), int(e)) for s, e, _ in parsed]


def pct(vals, q):
    if not vals:
        return float("nan")
    o = sorted(vals)
    k = max(0, min(len(o) - 1, int(round(q * (len(o) - 1)))))
    return float(o[k])


def load(name):
    return cues(FIXTURES / name)


def _quarter_bounds(n: int, qi: int) -> tuple[int, int]:
    """Index bounds of quarter ``qi`` of ``n`` cues, matching ``quarters()``."""
    edges = [0, n // 4, n // 2, (3 * n) // 4, n]
    return edges[qi], edges[qi + 1]


def quarters(seq):
    """Split a list of (start,end) cues into four equal-count temporal blocks."""
    n = len(seq)
    edges = [0, n // 4, n // 2, (3 * n) // 4, n]
    out = []
    for i in range(4):
        out.append(seq[edges[i]: edges[i + 1]])
    return out


def diagnose(label, target, reference, hypothesis, anchor_hypothesis):
    rep = anchor_guided_correspondence(
        [(s, e, "") for s, e in target],
        [(s, e, "") for s, e in reference],
        hypothesis,
    )

    print("=" * 78)
    print(f"{label}   hypothesis={hypothesis:.0f}ms   "
          f"decision={rep.decision.value}")
    print("=" * 78)
    m_start = min(t[0] for t in target)
    m_end = max(t[0] for t in target)
    span = m_end - m_start

    tq = quarters(target)
    rq = quarters(reference)

    print(f"\n  mapping: {rep.mapping.piece_count} piece(s) "
          f"offsets={[round(s.offset_ms) for s in rep.mapping.segments]}")
    print(f"  overall: corr={len(rep.correspondences)} "
          f"amb={len(rep.ambiguous_target)} "
          f"cov={rep.coverage:.4f} "
          f"p50={pct([c.residual_ms for c in rep.correspondences],0.5):.0f} "
          f"p95={pct([c.residual_ms for c in rep.correspondences],0.95):.0f} "
          f"p99={pct([c.residual_ms for c in rep.correspondences],0.99):.0f}")

    header = (
        f"  {'region':<9}{'tgt':>5}{'ref':>5}{'T/R':>6}"
        f"{'corr':>6}{'cov':>7}{'p50':>7}{'p95':>7}{'p99':>7}"
        f"{'unm':>5}{'amb':>5}{'medOff':>9}{'seg?':>7}"
    )
    print("\n" + header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    for qi in range(4):
        t_block = tq[qi]
        r_block = rq[qi]
        lo = t_block[0][0]
        hi = t_block[-1][0] if t_block else 0
        corr = [c for c in rep.correspondences if lo <= c.target_start_ms < hi]
        res = [c.residual_ms for c in corr]
        offs = [c.target_start_ms - c.reference_start_ms for c in corr]

        block_indices = set(range(*_quarter_bounds(len(target), qi)))
        q_unmatched = len(block_indices & set(rep.unmatched_target))
        q_ambiguous = len(block_indices & set(rep.ambiguous_target))

        t_active = sum(e - s for s, e in t_block)
        t_dur = [e - s for s, e in t_block]
        r_active = sum(e - s for s, e in r_block)
        r_dur = [e - s for s, e in r_block]

        # Segmentation evidence: is the target denser than the reference here?
        ratio = len(t_block) / max(1, len(r_block))
        seg = "same"
        if ratio > 1.6:
            seg = "finer"
        elif ratio > 1.15:
            seg = "slightly+"
        elif ratio < 0.62:
            seg = "coarser"
        elif ratio < 0.87:
            seg = "slightly-"

        print(
            f"  Q{qi+1:<8}{len(t_block):>5}{len(r_block):>5}{ratio:>6.2f}"
            f"{len(corr):>6}{len(corr)/max(1,len(t_block)):>7.3f}"
            f"{pct(res,0.5):>7.0f}{pct(res,0.95):>7.0f}{pct(res,0.99):>7.0f}"
            f"{q_unmatched:>5}{q_ambiguous:>5}"
            f"{(statistics.median(offs) if offs else float('nan')):>9.0f}{seg:>7}"
        )
        rows.append(
            dict(
                quarter=qi + 1,
                target_cues=len(t_block),
                reference_cues=len(r_block),
                cue_ratio=round(ratio, 3),
                correspondences=len(corr),
                coverage=round(len(corr) / max(1, len(t_block)), 4),
                unmatched=q_unmatched,
                ambiguous=q_ambiguous,
                p50=round(pct(res, 0.5), 1),
                p95=round(pct(res, 0.95), 1),
                p99=round(pct(res, 0.99), 1),
                median_local_offset=round(statistics.median(offs), 1) if offs else None,
                median_target_duration=round(statistics.median(t_dur), 1),
                median_reference_duration=round(statistics.median(r_dur), 1),
                target_active_ms=t_active,
                reference_active_ms=r_active,
                segmentation=seg,
            )
        )

    # ---- mechanism probes ------------------------------------------------- #
    print("\n  MECHANISM PROBES")

    # 1. Is the residual spread across the whole file, or clustered?
    res_all = [c.residual_ms for c in rep.correspondences]
    over = [c for c in rep.correspondences if abs(c.residual_ms) > 1000]
    print(f"    |residual| > 1000ms : {len(over)}/{len(res_all)} "
          f"({100*len(over)/max(1,len(res_all)):.1f}%)")
    if over:
        times = sorted((c.target_start_ms - m_start) / span for c in over)
        # are they clustered or spread?
        buckets = [0, 0, 0, 0]
        for t in times:
            buckets[min(3, int(t * 4))] += 1
        print(f"    their distribution over quarters: {buckets}")

    # 2. One-to-many / many-to-one: do target cues collide on reference cues?
    ref_hits: dict[int, list[int]] = {}
    for c in rep.correspondences:
        ref_hits.setdefault(c.reference_index, []).append(c.target_index)
    many_to_one = sum(1 for v in ref_hits.values() if len(v) > 1)
    # A reference cue is "shared" when exactly one target cue claims it but its
    # immediate neighbour in target order is also matched to a nearby reference
    # cue -- the signature of one reference cue split across two target cues.
    one_to_many = 0
    for c in rep.correspondences:
        nxt = c.reference_index + 1
        if nxt in ref_hits and any(
            t > c.target_index for t in ref_hits[nxt]
        ):
            one_to_many += 1
    print(f"    reference cues hit by >1 target cue (many-to-one): {many_to_one}")
    print(f"    target cues with an adjacent reference cue also matched "
          f"(one-to-many signature): {one_to_many}")

    # 3. Do rejected neighbours exist just outside the window?
    print(f"    ambiguous target cues (rejected, not forced): "
          f"{len(rep.ambiguous_target)}")
    print(f"    unmatched target cues (no candidate at all): "
          f"{len(rep.unmatched_target)}")

    # 4. Duration relationship
    t_dur_all = [e - s for s, e in target]
    r_dur_all = [e - s for s, e in reference]
    print(f"    median target cue duration : "
          f"{statistics.median(t_dur_all):.0f}ms")
    print(f"    median reference duration  : "
          f"{statistics.median(r_dur_all):.0f}ms")

    # 5. Active-duration accounting
    print(f"    total active target   : {sum(t_dur_all)}ms")
    print(f"    total active reference: {sum(r_dur_all)}ms")

    return dict(label=label, hypothesis=hypothesis, rows=rows,
                decision=rep.decision.value,
                correspondences=len(rep.correspondences),
                coverage=round(rep.coverage, 4),
                p95=round(pct(res_all, 0.95), 1),
                p50=round(pct(res_all, 0.5), 1))


def main() -> int:
    ref = load("05_PiR8_reference.srt")
    results = []
    results.append(
        diagnose("ASAP alass vs reference", load("04_ASAP_alass_output.srt"), ref, 0.0, 0.0)
    )
    results.append(
        diagnose("EVOLV alass vs reference", load("02_EVOLV_alass_output.srt"), ref, 0.0, 0.0)
    )
    results.append(
        diagnose("ASAP original vs reference", load("03_ASAP_original.srt"), ref, 96050.0, 96050.0)
    )
    results.append(
        diagnose("EVOLV original vs reference", load("01_EVOLV_original.srt"), ref, 96100.0, 96100.0)
    )
    out = ROOT / "reports" / "asap_residual_breakdown.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
