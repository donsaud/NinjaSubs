"""PART 7/9/10/12 runner: current vs anchor-guided vs interval alignment.

Diagnostic only. Prints tables; writes JSON under reports/.
"""

from __future__ import annotations

import json
import os
import pathlib
import random
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.subtitle_matcher import parse_srt_cues  # noqa: E402
from app.services.sync.anchor_correspondence import (  # noqa: E402
    anchor_guided_correspondence,
)
from app.services.sync.interval_correspondence import (  # noqa: E402
    AlignmentOutcome,
    align_intervals,
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


# ---------------------------------------------------------------- fixtures -- #
def load(name: str) -> list[tuple[int, int, str]]:
    text = (FIXTURES / name).read_text(encoding="utf-8", errors="replace")
    return [(int(s), int(e), t) for s, e, t in parse_srt_cues(text)]


def strip_text(cues):
    """Timing-only. The model must never need text."""
    return [(s, e, "") for s, e, _ in cues]


# ------------------------------------------------------------- generators -- #
def shift(cues, d):
    return [(s + d, e + d, t) for s, e, t in cues]


def drift(cues, ms_per_min):
    if not cues:
        return cues
    t0 = cues[0][0]
    out = []
    for s, e, t in cues:
        d = int(round((s - t0) / 60000.0 * ms_per_min))
        out.append((s + d, e + d, t))
    return out


def split_every(cues, n=2):
    """Split every ``n``-th cue in two, leaving the rest alone.

    Must not truncate: slicing to the original length would silently delete the
    tail, which reads as "reference only half explained" rather than as finer
    segmentation.
    """
    out = []
    for i, (s, e, t) in enumerate(cues):
        if i % n == 0 and e - s >= 4:
            mid = s + (e - s) // 2
            out.append((s, mid, t))
            out.append((mid, e, t))
        else:
            out.append((s, e, t))
    return out


def merge_pairs(cues):
    """Coarser segmentation: one cue covers what two did.

    The merged cue spans the pair but does NOT absorb the silent gap between
    them, so total speech time is preserved. Including the gap would inflate
    active time and masquerade as content duplication.
    """
    out = []
    for i in range(0, len(cues), 2):
        if i + 1 < len(cues):
            gap = max(0, cues[i + 1][0] - cues[i][1])
            out.append((cues[i][0], max(cues[i][1], cues[i + 1][1] - gap), ""))
        else:
            out.append(cues[i])
    return out


def piecewise(cues, first_frac, early, late):
    if not cues:
        return cues
    b = cues[int(len(cues) * first_frac)][0]
    return [
        (s + (early if s < b else late), e + (early if s < b else late), t)
        for s, e, t in cues
    ]


def drop_every(cues, frac):
    """Delete exactly ``frac`` of the cues, spread evenly.

    The previous version computed a stride from ``len*frac`` and so deleted only
    a couple of cues -- "deleted 60%" removed 0.4% and the model was right to
    see nothing. This keeps every ``1/(1-frac)``-th cue.
    """
    if frac >= 1:
        return []
    keep_every = max(2, int(round(1.0 / (1.0 - frac))))
    return [c for i, c in enumerate(cues) if i % keep_every != 0]


def duplicate_every(cues, frac):
    """Insert ``frac`` extra cues, each an exact adjacent twin of a real cue."""
    if frac <= 0:
        return list(cues)
    period = max(2, int(round(1.0 / frac)))
    out = []
    for i, c in enumerate(cues):
        out.append(c)
        if i % period == 0:
            out.append(c)
    return sorted(out, key=lambda c: c[0])


def corrupt(cues, amount, frac, seed=5):
    rng = random.Random(seed)
    out = []
    for s, e, t in cues:
        if rng.random() < frac:
            j = rng.randint(-amount, amount)
            out.append((s + j, e + j, t))
        else:
            out.append((s, e, t))
    return out


def truncate(cues, keep):
    return cues[: int(len(cues) * keep)]


def reorder_text(cues, seed=11):
    """Shuffle the *text* of adjacent cues, keeping every timestamp.

    A time-ordered subtitle cannot be "reordered" in time without becoming
    corrupt, so the only meaningful reordering is content order. A timing model
    that ignores text is blind to this by construction, and the case is kept to
    demonstrate exactly that.
    """
    texts = [t for _, _, t in cues]
    rng = random.Random(seed)
    texts = list(texts)
    rng.shuffle(texts)
    return [(s, e, texts[i]) for i, (s, e, _) in enumerate(cues)]


def middle_cut(cues, frac=0.3, shift_ms=4000):
    """Different cut: a middle stretch sits at a different offset."""
    if not cues:
        return cues
    a = cues[int(len(cues) * (0.5 - frac / 2))][0]
    b = cues[int(len(cues) * (0.5 + frac / 2))][0]
    return [
        (s + (shift_ms if a <= s < b else 0), e + (shift_ms if a <= s < b else 0), t)
        for s, e, t in cues
    ]


def sparsify(cues, keep):
    step = max(1, int(1 / max(1e-6, keep)))
    return [c for i, c in enumerate(cues) if i % step == 0]


# ---------------------------------------------------------------- running -- #
def pct(vals, q):
    if not vals:
        return float("inf")
    o = sorted(vals)
    k = max(0, min(len(o) - 1, int(round(q * (len(o) - 1)))))
    return float(o[k])


def run_case(name, target, reference, hypothesis, category):
    """Current production pairing, anchor-guided, and interval alignment."""
    row = {
        "case": name,
        "category": category,
        "target_cues": len(target),
        "reference_cues": len(reference),
    }

    # --- anchor-guided (previous experiment) --- #
    try:
        ag = anchor_guided_correspondence(target, reference, hypothesis)
        row["anchor_guided"] = {
            "correspondences": len(ag.correspondences),
            "coverage": round(ag.coverage, 4),
            "p50": round(pct([c.residual_ms for c in ag.correspondences], 0.5), 1),
            "p95": round(pct([c.residual_ms for c in ag.correspondences], 0.95), 1),
            "ambiguous": len(ag.ambiguous_target),
            "unmatched": len(ag.unmatched_target),
            "pieces": ag.mapping.piece_count,
            "decision": ag.decision.value,
        }
    except Exception as exc:  # pragma: no cover - diagnostic tool
        row["anchor_guided"] = {"error": str(exc)}

    # --- interval alignment (new model) --- #
    try:
        iv = align_intervals(target, reference, hypothesis)
        d = iv.as_row()
        d["splits"] = iv.split_count()
        d["merges"] = iv.merge_count()
        d["completeness"] = iv.completeness
        d["content"] = iv.content
        d["content_observation"] = iv.content_observation.value
        d["notes"] = iv.notes
        row["interval"] = d
    except Exception as exc:  # pragma: no cover - diagnostic tool
        row["interval"] = {"error": str(exc)}

    return row


HDR = (
    f"  {'case':<28}{'cat':<19}{'grps':>5}{'spl':>4}{'mrg':>4}"
    f"{'tcov':>6}{'rcov':>6}{'p50':>6}{'p95':>7}{'p99':>7}"
    f"{'amb':>5}{'surT':>6}{'surR':>6}{'pcs':>4}{'actR':>6}  {'timing':<28}{'content':<24}"
)


def show(rows, title):
    print("=" * 176)
    print(title)
    print("=" * 176)
    print(HDR)
    print("  " + "-" * (len(HDR) - 2))
    for r in rows:
        iv = r.get("interval", {})
        if "error" in iv:
            print(f"  {r['case']:<28}{r['category']:<19}  ERROR {iv['error'][:50]}")
            continue
        act = iv.get("completeness", {}).get("active_duration_ratio", float("nan"))
        print(
            f"  {r['case']:<28}{r['category']:<19}"
            f"{iv['groups']:>5}{iv['splits']:>4}{iv['merges']:>4}"
            f"{iv['target_coverage']:>6.3f}{iv['reference_coverage']:>6.3f}"
            f"{iv['p50']:>6.0f}{iv['p95']:>7.0f}{iv['p99']:>7.0f}"
            f"{iv['ambiguous']:>5}{iv['surplus_target']:>6}{iv['surplus_reference']:>6}"
            f"{iv['mapping_pieces']:>4}{act:>6.2f}  "
            f"{iv['outcome'].replace('INTERVAL_', ''):<28}"
            f"{str(iv.get('content_observation', '')).replace('TARGET_CONTENT_', 'T_'):<24}"
        )
    print()


def main() -> int:
    ref = load("05_PiR8_reference.srt")
    evolv = load("01_EVOLV_original.srt")
    evolv_al = load("02_EVOLV_alass_output.srt")
    asap = load("03_ASAP_original.srt")
    asap_al = load("04_ASAP_alass_output.srt")

    all_rows = []

    # ---------------- PART 7: real cases ---------------- #
    real = [
        ("EVOLV original", evolv, 96100.0, "LARGE_OFFSET"),
        ("EVOLV alass", evolv_al, 0.0, "GOOD_ALIGNMENT"),
        ("ASAP original", asap, 96050.0, "LARGE_OFFSET"),
        ("ASAP alass", asap_al, 0.0, "GOOD_ALIGNMENT"),
    ]
    rows = [run_case(n, strip_text(t), strip_text(ref), h, c)
            for n, t, h, c in real]
    show(rows, "PART 7  REAL CASES")
    all_rows += rows

    # ---------------- PART 8: uniform offsets ---------------- #
    base = strip_text(ref)
    rows = []
    for d in (0, 2000, 12000, 30000, 60000, 96500, 120000, 170000):
        rows.append(run_case(f"uniform +{d/1000:g}s", shift(base, d), base, float(d),
                             "GOOD_ALIGNMENT"))
    show(rows, "PART 8a  UNIFORM LARGE OFFSETS")
    all_rows += rows

    # ---------------- PART 8: piecewise ---------------- #
    rows = [
        run_case("opening +96s", piecewise(base, 0.22, 96000, 0), base, 96000.0,
                 "PIECEWISE_ALIGNMENT"),
        run_case("closing +96s", piecewise(base, 0.78, 0, 96000), base, 96000.0,
                 "PIECEWISE_ALIGNMENT"),
        run_case("first-third +96s", piecewise(base, 1 / 3, 96000, 0), base, 96000.0,
                 "PIECEWISE_ALIGNMENT"),
        run_case("two-region 96s/0", piecewise(base, 0.5, 96000, 0), base, 96000.0,
                 "PIECEWISE_ALIGNMENT"),
        run_case("three-region", split_piecewise(base), base, 96000.0,
                 "PIECEWISE_ALIGNMENT"),
    ]
    show(rows, "PART 8b  PIECEWISE")
    all_rows += rows

    # ---------------- PART 8: drift ---------------- #
    rows = [run_case(f"drift {m}ms/min", drift(base, m), base, 0.0, "DRIFT")
            for m in (30, 60, 100)]
    show(rows, "PART 8c  DRIFT")
    all_rows += rows

    # ---------------- PART 9: segmentation ---------------- #
    rows = [
        run_case("identical", base, base, 0.0, "SEGMENTATION_VARIATION"),
        run_case("slightly finer", split_every(base, 2), base, 0.0,
                 "SEGMENTATION_VARIATION"),
        run_case("much finer", split_every(base, 4), base, 0.0,
                 "SEGMENTATION_VARIATION"),
        run_case("slightly coarser", merge_pairs(base), base, 0.0,
                 "SEGMENTATION_VARIATION"),
        run_case("mixed split/merge", mixed_seg(base), base, 0.0,
                 "SEGMENTATION_VARIATION"),
        run_case("finer + 96s", shift(split_every(base, 2), 96500), base, 96500.0,
                 "SEGMENTATION_VARIATION"),
    ]
    show(rows, "PART 9  SEGMENTATION")
    all_rows += rows

    # ---------------- PART 10: damage ---------------- #
    rows = [
        run_case("deleted 20%", drop_every(base, 0.20), base, 0.0, "CONTENT_DAMAGE"),
        run_case("deleted 40%", drop_every(base, 0.40), base, 0.0, "CONTENT_DAMAGE"),
        run_case("deleted 60%", drop_every(base, 0.60), base, 0.0, "CONTENT_DAMAGE"),
        run_case("duplicated 20%", duplicate_every(base, 0.20), base, 0.0,
                 "CONTENT_DAMAGE"),
        run_case("duplicated 40%", duplicate_every(base, 0.40), base, 0.0,
                 "CONTENT_DAMAGE"),
        run_case("corrupt 10% +/-3s", corrupt(base, 3000, 0.10), base, 0.0,
                 "CONTENT_DAMAGE"),
        run_case("corrupt 20% +/-3s", corrupt(base, 3000, 0.20), base, 0.0,
                 "CONTENT_DAMAGE"),
        run_case("corrupt 30% +/-8s", corrupt(base, 8000, 0.30), base, 0.0,
                 "CONTENT_DAMAGE"),
        run_case("text reordered", reorder_text(base), base, 0.0,
                 "CONTENT_DAMAGE"),
        run_case("uniform +700ms", shift(base, 700), base, 0.0,
                 "GOOD_ALIGNMENT"),
        run_case("wrong cut (mid shift)", middle_cut(base), base, 0.0,
                 "WRONG_CUT"),
        run_case("wrong release +96s", shift(base, 96_500), base, 0.0,
                 "WRONG_RELEASE"),
        run_case("wrong release +96s (hyp)", shift(base, 96_500), base, 96500.0,
                 "GOOD_ALIGNMENT"),
        run_case("wrong episode (timing twin)", shift(base, 1200), base, 1200.0,
                 "WRONG_EPISODE"),
        run_case("truncated 60%", truncate(base, 0.6), base, 0.0, "TRUNCATED"),
        run_case("sparse 25%", sparsify(base, 0.25), base, 0.0, "SPARSE"),
    ]
    show(rows, "PART 10  DAMAGE")
    all_rows += rows

    # ---------------- PART 12: confusion ---------------- #
    confusion(all_rows)

    out = ROOT / "reports" / "interval_model_matrix.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(f"  written {out}")
    return 0


def split_piecewise(cues):
    """Three regions: +96s, 0, -96s."""
    if not cues:
        return cues
    a = cues[int(len(cues) * 0.34)][0]
    b = cues[int(len(cues) * 0.67)][0]
    out = []
    for s, e, t in cues:
        d = 96000 if s < a else (0 if s < b else -96000)
        out.append((s + d, e + d, t))
    return out


def mixed_seg(cues):
    """Alternate: split some cues, merge others."""
    if not cues:
        return cues
    out = []
    i = 0
    n = len(cues)
    while i < n:
        if i % 4 == 0 and i + 1 < n:
            a, b = cues[i], cues[i + 1]
            step = max(1, (b[1] - a[0]) // 3)
            for k in range(3):
                out.append((a[0] + k * step, a[0] + k * step + step, ""))
            i += 2
        elif i % 4 == 1 and i + 2 < n:
            out.append((cues[i][0], cues[i + 2][1], ""))
            i += 3
        else:
            out.append(cues[i])
            i += 1
    return sorted(out, key=lambda c: c[0])


def confusion(rows):
    """PART 12/13: does the model separate damage, and does it abstain?"""
    print("=" * 150)
    print("PART 12  CATEGORY TABLE  (interval model)")
    print("=" * 150)
    hdr = (f"  {'category':<24}{'n':>3}{'ALIGNED':>9}{'SURP_T':>8}{'SURP_R':>8}"
           f"{'AMBIG':>7}{'LOWCOV':>8}{'NONE':>7}{'ANCH':>6}  medP95  medTcov")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    cats = {}
    for r in rows:
        cats.setdefault(r["category"], []).append(r.get("interval", {}))
    for cat, items in sorted(cats.items()):
        def count(o, rows_=items):
            return sum(1 for i in rows_ if i.get("outcome") == o.value)
        p95 = [i["p95"] for i in items if "p95" in i]
        tc = [i["target_coverage"] for i in items if "target_coverage" in i]
        print(
            f"  {cat:<24}{len(items):>3}"
            f"{count(AlignmentOutcome.ALIGNED):>9}"
            f"{count(AlignmentOutcome.SURPLUS_TARGET):>8}"
            f"{count(AlignmentOutcome.SURPLUS_REFERENCE):>8}"
            f"{count(AlignmentOutcome.AMBIGUOUS):>7}"
            f"{count(AlignmentOutcome.LOW_REFERENCE_COVERAGE):>8}"
            f"{count(AlignmentOutcome.NO_CORRESPONDENCE):>7}"
            f"{count(AlignmentOutcome.ANCHORS_INCONSISTENT):>6}"
            f"  {(statistics.median(p95) if p95 else float('nan')):7.0f}"
            f"  {(statistics.median(tc) if tc else float('nan')):.3f}"
        )
    print()


if __name__ == "__main__":
    raise SystemExit(main())
