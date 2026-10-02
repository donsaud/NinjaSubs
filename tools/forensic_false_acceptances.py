"""PART 3/5/6/7/8: forensic comparison of the five false acceptances.

For each known false acceptance, asks one question: what evidence that already
exists in this system separates it from a legitimately synchronized subtitle?

Only existing signals are measured. Nothing here is used to accept anything, and
no threshold is fitted to these fixtures -- the point is to find out which
signals *can* discriminate, not to tune one until they do.

Text is hashed and discarded; only counts and numbers are reported.

Run: python tools/forensic_false_acceptances.py
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import random
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.subtitle_matcher import parse_srt_cues  # noqa: E402
from app.services.sync.interval_correspondence import (  # noqa: E402
    align_with_graded_anchors,
)
from app.services.sync.large_offset import (  # noqa: E402
    _shifted_profile_similarity,
)
from app.services.sync.structural import compare_structures  # noqa: E402

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
_NON_WORD = re.compile(r"[^\w]+")


def load(name: str) -> list[tuple[int, int, str]]:
    text = (FIXTURES / name).read_text(encoding="utf-8", errors="replace")
    return [(int(s), int(e), t) for s, e, t in parse_srt_cues(text)]


def timing_only(cues):
    return [(s, e, "") for s, e, _ in cues]


def norm_hash(text: str) -> str:
    cleaned = _NON_WORD.sub(" ", text.lower()).strip()
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:12] if cleaned else ""


def sequence_signature(cues) -> list[str]:
    """Per-cue structural digest: duration bucket + gap bucket. No text survives."""
    sig = []
    prev_end = None
    for s, e, _ in cues:
        dur = max(0, e - s)
        gap = max(0, s - prev_end) if prev_end is not None else 0
        sig.append(f"{dur // 250}:{gap // 500}")
        prev_end = e
    return sig


def order_agreement(a: list[str], b: list[str]) -> float:
    """Fraction of positions where two equal-length signatures agree."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(1 for i in range(n) if a[i] == b[i]) / n


def multiset_text(cues) -> dict[str, int]:
    out: dict[str, int] = {}
    for _, _, t in cues:
        h = norm_hash(t)
        if h:
            out[h] = out.get(h, 0) + 1
    return out


def corrupt(cues, amount, frac, seed=5):
    rng = random.Random(seed)
    out = []
    for s, e, t in cues:
        j = rng.randint(-amount, amount) if rng.random() < frac else 0
        out.append((s + j, e + j, t))
    return out


def middle_cut(cues, frac=0.3, shift_ms=4000):
    a = cues[int(len(cues) * (0.5 - frac / 2))][0]
    b = cues[int(len(cues) * (0.5 + frac / 2))][0]
    return [
        (s + (shift_ms if a <= s < b else 0), e + (shift_ms if a <= s < b else 0), t)
        for s, e, t in cues
    ]


def shift(cues, d):
    return [(s + d, e + d, t) for s, e, t in cues]


def reorder_text(cues, seed=11):
    texts = [t for _, _, t in cues]
    random.Random(seed).shuffle(texts)
    return [(s, e, texts[i]) for i, (s, e, _) in enumerate(cues)]


def rows(name, target, reference, hypothesis, good):
    st = compare_structures(timing_only(target), timing_only(reference))
    t_active = sum(max(0, e - s) for s, e, _ in target)
    r_active = sum(max(0, e - s) for s, e, _ in reference)
    rep = align_with_graded_anchors(timing_only(target), timing_only(reference),
                                    hypothesis)
    prof = _shifted_profile_similarity(
        [(s, e, "") for s, e, _ in target],
        [(s, e, "") for s, e, _ in reference],
        0.0,
    )
    tt = multiset_text(target)
    rt = multiset_text(reference)
    matched = sum(min(c, rt.get(h, 0)) for h, c in tt.items())
    return {
        "case": name,
        "kind": "GOOD" if good else "FALSE-ACCEPT",
        "timing_outcome": rep.outcome.value.replace("INTERVAL_", ""),
        "ref_coverage": round(rep.reference_coverage, 3),
        "p95": round(rep.p95(), 1),
        "splits": rep.split_count(),
        "merges": rep.merge_count(),
        "struct_score": None if st.score is None else round(st.score, 3),
        "struct_cue_ratio": (
            None if st.cue_count_ratio is None else round(st.cue_count_ratio, 3)
        ),
        "struct_similar": st.same_structure,
        "cue_ratio": round(len(target) / len(reference), 3),
        "active_ratio": round(t_active / r_active, 3) if r_active else None,
        "profile_similarity": None if prof is None else round(prof, 3),
        "first_cue_delta": target[0][0] - reference[0][0],
        "last_cue_delta": target[-1][1] - reference[-1][1],
        "text_order_agreement": round(
            order_agreement(sequence_signature(target), sequence_signature(reference)),
            3,
        ),
        "text_line_overlap": round(matched / max(1, sum(tt.values())), 3),
    }


COLS = [
    ("case", "case", 26),
    ("kind", "kind", 12),
    ("timing_outcome", "timing", 22),
    ("ref_coverage", "refCov", 7),
    ("p95", "p95", 7),
    ("struct_score", "struct", 7),
    ("struct_cue_ratio", "sCRatio", 8),
    ("struct_similar", "sSim", 6),
    ("cue_ratio", "cueR", 6),
    ("active_ratio", "actR", 6),
    ("profile_similarity", "profSim", 8),
    ("first_cue_delta", "dFirst", 8),
    ("last_cue_delta", "dLast", 7),
    ("text_order_agreement", "seqAgr", 7),
    ("text_line_overlap", "txtOvl", 7),
]


def show(rs, title):
    print("=" * 168)
    print(title)
    print("=" * 168)
    print("  " + "".join(f"{h:>{w}}" for _, h, w in COLS))
    print("  " + "-" * 164)
    for r in rs:
        cells = []
        for key, _, w in COLS:
            v = r[key]
            if isinstance(v, bool):
                v = "yes" if v else "NO"
            elif isinstance(v, float):
                v = f"{v:.3f}"
            cells.append(f"{str(v):>{w}}")
        print("  " + "".join(cells))
    print()


def main() -> int:
    ref = load("05_PiR8_reference.srt")
    ref_t = timing_only(ref)

    good = [
        ("EVOLV alass (good)", timing_only(load("02_EVOLV_alass_output.srt")), 0.0),
        ("ASAP alass (good)", timing_only(load("04_ASAP_alass_output.srt")), 0.0),
        ("uniform +96.5s", shift(ref_t, 96_500), 96_500.0),
    ]
    bad = [
        ("FA1 wrong episode", shift(ref, 1200), 1200.0),
        ("FA2 text reordered", reorder_text(ref), 0.0),
        ("FA3 corrupt 10%", corrupt(ref, 3000, 0.10), 0.0),
        ("FA4 corrupt 20%", corrupt(ref, 3000, 0.20), 0.0),
        ("FA5 wrong cut +4s", middle_cut(ref), 0.0),
    ]

    g_rows = [rows(n, t, ref, h, True) for n, t, h in good]
    b_rows = [rows(n, t, ref, h, False) for n, t, h in bad]
    show(g_rows, "LEGITIMATE CASES (reference for comparison)")
    show(b_rows, "THE FIVE FALSE ACCEPTANCES")

    print("=" * 168)
    print("SIGNAL SEPARATION: does the signal differ between the good and the false-accept group?")
    print("=" * 168)
    keys = [
        "struct_score", "struct_cue_ratio", "struct_similar", "cue_ratio",
        "active_ratio", "profile_similarity", "first_cue_delta",
        "last_cue_delta", "text_order_agreement", "text_line_overlap",
        "ref_coverage", "p95",
    ]
    for k in keys:
        gv = [r[k] for r in g_rows if r[k] is not None]
        bv = [r[k] for r in b_rows if r[k] is not None]
        if not gv or not bv:
            verdict = "NOT AVAILABLE (no values on one side)"
            print(f"  {k:<24} {verdict}")
            continue
        if all(isinstance(x, bool) for x in gv + bv):
            same = set(gv) == set(bv)
            verdict = "USELESS (identical on both sides)" if same else "SEPARATES"
        else:
            overlap = (
                max(gv) >= min(bv) and min(gv) <= max(bv)
            )
            verdict = "AMBIGUOUS (ranges overlap)" if overlap else "SEPARATES"
        print(f"  {k:<24} good=[{min(gv)}..{max(gv)}]  bad=[{min(bv)}..{max(bv)}]"
              f"  -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
