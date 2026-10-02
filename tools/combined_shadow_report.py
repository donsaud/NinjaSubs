"""PART 9/10/11: combined shadow report and corpus scoring.

DIAGNOSTIC ONLY. Runs every corpus case through the graded-anchor DP and the
content-integrity assessment, assembles the combined shadow verdict, and scores
correct/false acceptances, rejections and abstentions. Measures runtime.

The combined verdict deliberately requires identity evidence. Where that evidence
does not exist the answer is ABSTAIN, not a pass.

Run: python tools/combined_shadow_report.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tools.run_interval_matrices as mx  # noqa: E402
from app.services.sync.content_integrity import (  # noqa: E402
    ContentIntegrity,
    assess_content_integrity,
    sequence_disagreement,
)
from app.services.sync.interval_correspondence import (  # noqa: E402
    AlignmentOutcome,
    align_with_graded_anchors,
)
from app.services.sync.reference import (  # noqa: E402
    assess_consensus,
    detect_duplicate_groups,
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


def load(name):
    return mx.load(name)


def timing_only(cues):
    return [(s, e, "") for s, e, _ in cues]


class ShadowVerdict(str):
    pass


ACCEPT = "ACCEPT-CANDIDATE"
REJECT = "REJECT-CANDIDATE"
ABSTAIN = "ABSTAIN"


def identity_evidence(reference_text: str) -> dict:
    """What identity evidence actually exists for the local reference set."""
    groups = detect_duplicate_groups([("provA", "ref-a", reference_text)])
    score, grp, agrees, reasons = assess_consensus(groups)
    return {
        "reference_groups": len(groups),
        "consensus_score": score,
        "consensus_agrees": agrees,
        "consensus_available": score is not None,
        "reasons": reasons,
    }


def combined_shadow(name, category, expect, target, reference, hypothesis, ref_text):
    st = compare_structures(timing_only(target), timing_only(reference))

    t0 = time.perf_counter()
    rep = align_with_graded_anchors(timing_only(target), timing_only(reference), hypothesis)
    dp_ms = (time.perf_counter() - t0) * 1000.0

    integrity = assess_content_integrity(
        target, reference, reference_coverage=rep.reference_coverage
    )
    identity = identity_evidence(ref_text)
    seq = sequence_disagreement(target, reference)

    # --- the combined decision --------------------------------------------- #
    # Order matters. Identity evidence that does not exist means there is nothing
    # to decide on, so it abstains before any quality reading is consulted.
    if not identity["consensus_available"]:
        verdict = ABSTAIN
        because = (
            "no independent second reference: identity evidence unavailable, so no "
            "case can be an accept-candidate"
        )
    elif rep.outcome in (
        AlignmentOutcome.ANCHORS_INCONSISTENT,
        AlignmentOutcome.NO_CORRESPONDENCE,
    ):
        verdict = ABSTAIN
        because = f"timing abstained: {rep.outcome.value}"
    elif integrity.verdict is ContentIntegrity.SUSPECT:
        verdict = REJECT
        because = "content integrity suspect"
    elif rep.outcome is not AlignmentOutcome.ALIGNED:
        verdict = REJECT
        because = f"timing not aligned: {rep.outcome.value}"
    elif integrity.verdict is ContentIntegrity.UNKNOWN:
        verdict = ABSTAIN
        because = "content integrity unknown"
    elif not st.same_structure:
        verdict = REJECT
        because = "structural similarity below the existing minimum"
    else:
        verdict = ACCEPT
        because = "timing aligned, content good, identity and structure present"

    return {
        "case": name,
        "category": category,
        "expected": expect,
        "verdict": verdict,
        "because": because,
        "TIMING": {
            "dp_outcome": rep.outcome.value,
            "dp_reference_coverage": round(rep.reference_coverage, 4),
            "dp_p95": round(rep.p95(), 1),
            "dp_splits": rep.split_count(),
            "dp_merges": rep.merge_count(),
            "anchors": (
                rep.anchor_grading.counts() if rep.anchor_grading else None
            ),
            "anchor_dispersion_ms": (
                rep.anchor_grading.dispersion_ms if rep.anchor_grading else None
            ),
            "anchor_spread_ms": (
                rep.anchor_grading.residual_spread if rep.anchor_grading else None
            ),
            "dp_runtime_ms": round(dp_ms, 1),
        },
        "CONTENT": {
            "cue_count_ratio": integrity.observed.get("cue_count_ratio"),
            "active_duration_ratio": integrity.observed.get("active_duration_ratio"),
            "runtime_coverage": rep.completeness.get("matched_reference_span_ratio"),
            "reference_coverage": rep.reference_coverage,
            "integrity": integrity.verdict.value,
            "text_line_overlap": integrity.observed.get("text_line_overlap"),
            "sequence_disagreement": None if seq is None else round(seq, 4),
            "limits": integrity.limits,
        },
        "IDENTITY": identity,
        "STRUCTURE": {
            "structural_similarity": st.score,
            "cue_count_ratio": st.cue_count_ratio,
            "same_structure": st.same_structure,
            "split_count": rep.split_count(),
            "merge_count": rep.merge_count(),
            "monotonic": True,
        },
        "_runtime_ms": round(dp_ms, 1),
    }


def main() -> int:
    ref = load("05_PiR8_reference.srt")
    ref_text = (FIXTURES / "05_PiR8_reference.srt").read_text(
        encoding="utf-8", errors="replace"
    )


    evolv = load("01_EVOLV_original.srt")
    evolv_al = load("02_EVOLV_alass_output.srt")
    asap = load("03_ASAP_original.srt")
    asap_al = load("04_ASAP_alass_output.srt")

    cases = [
        # GOOD
        ("EVOLV alass", "GOOD_ALIGNMENT", "ACCEPT", evolv_al, 0.0),
        ("ASAP alass", "GOOD_ALIGNMENT", "ACCEPT", asap_al, 0.0),
        ("EVOLV original", "LARGE_OFFSET", "ACCEPT", evolv, 96100.0),
        ("ASAP original", "LARGE_OFFSET", "ACCEPT", asap, 96050.0),
        # These derive from the reference, so they carry its text. Building them
        # from timing-only cues made CONTENT_INTEGRITY_GOOD unreachable, because
        # GOOD requires positive same-language evidence.
        ("uniform +0s", "GOOD_ALIGNMENT", "ACCEPT", ref, 0.0),
        ("uniform +96.5s", "GOOD_ALIGNMENT", "ACCEPT", mx.shift(ref, 96_500), 96500.0),
        ("uniform +170s", "GOOD_ALIGNMENT", "ACCEPT", mx.shift(ref, 170_000), 170000.0),
        ("seg: slightly finer", "SEGMENTATION_VARIATION", "ACCEPT",
         mx.split_every(ref, 2), 0.0),
        ("seg: much finer", "SEGMENTATION_VARIATION", "ACCEPT",
         mx.split_every(ref, 4), 0.0),
        ("seg: mixed split/merge", "SEGMENTATION_VARIATION", "ACCEPT",
         mx.mixed_seg(ref), 0.0),
        ("seg: finer +96.5s", "SEGMENTATION_VARIATION", "ACCEPT",
         mx.shift(mx.split_every(ref, 2), 96_500), 96500.0),
        ("drift 30ms/min", "DRIFT", "ACCEPT", mx.drift(ref, 30), 0.0),
        # BAD
        ("wrong episode (timing twin)", "WRONG_EPISODE", "REJECT", mx.shift(ref, 1200), 1200.0),
        ("text reordered", "WRONG_EPISODE", "REJECT", mx.reorder_text(ref), 0.0),
        ("wrong cut +4s", "WRONG_CUT", "REJECT", mx.middle_cut(ref), 0.0),
        ("wrong release +96s (bad hyp)", "WRONG_RELEASE", "REJECT", mx.shift(ref, 96_500), 0.0),
        ("truncated 60%", "TRUNCATED", "REJECT", mx.truncate(ref, 0.6), 0.0),
        ("sparse 25%", "SPARSE", "REJECT", mx.sparsify(ref, 0.25), 0.0),
        ("deleted 40%", "CONTENT_DAMAGE", "REJECT", mx.drop_every(ref, 0.40), 0.0),
        ("deleted 60%", "CONTENT_DAMAGE", "REJECT", mx.drop_every(ref, 0.60), 0.0),
        ("duplicated 40%", "CONTENT_DAMAGE", "REJECT", mx.duplicate_every(ref, 0.40), 0.0),
        ("corrupt 10%", "CONTENT_DAMAGE", "REJECT", mx.corrupt(ref, 3000, 0.10), 0.0),
        ("corrupt 20%", "CONTENT_DAMAGE", "REJECT", mx.corrupt(ref, 3000, 0.20), 0.0),
        ("corrupt 30% +/-8s", "CONTENT_DAMAGE", "REJECT", mx.corrupt(ref, 8000, 0.30), 0.0),
    ]

    rows = [
        combined_shadow(n, c, e, t, ref, h, ref_text)
        for n, c, e, t, h in cases
    ]

    print("=" * 150)
    print("PART 10  COMBINED SHADOW REPORT")
    print("=" * 150)
    hdr = (f"  {'case':<28}{'expect':>8}{'verdict':>18}{'dpOutcome':>24}"
           f"{'refCov':>7}{'p95':>7}{'content':>28}{'consensus':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        t = r["TIMING"]
        c = r["CONTENT"]
        print(
            f"  {r['case']:<28}{r['expected']:>8}{r['verdict']:>18}"
            f"{t['dp_outcome'].replace('INTERVAL_',''):>24}"
            f"{t['dp_reference_coverage']:>7.3f}{t['dp_p95']:>7.0f}"
            f"{c['integrity'].replace('CONTENT_INTEGRITY_',''):>28}"
            f"{str(r['IDENTITY']['consensus_available']):>10}"
        )

    print()
    print("=" * 150)
    print("PART 11  SCORING")
    print("=" * 150)

    def score(label, verdict_of):
        acc = [r for r in rows if verdict_of(r) == ACCEPT]
        rej = [r for r in rows if verdict_of(r) == REJECT]
        absn = [r for r in rows if verdict_of(r) == ABSTAIN]
        ca = [r for r in acc if r["expected"] == "ACCEPT"]
        fa = [r for r in acc if r["expected"] == "REJECT"]
        cr = [r for r in rej if r["expected"] == "REJECT"]
        fr = [r for r in rej if r["expected"] == "ACCEPT"]
        print(f"  -- {label}")
        print(f"     correct acceptance : {len(ca)}")
        print(f"     FALSE acceptance   : {len(fa)}")
        for r in fa:
            print(f"         {r['case']}")
        print(f"     correct rejection  : {len(cr)}")
        print(f"     FALSE rejection    : {len(fr)}")
        for r in fr:
            print(f"         {r['case']}")
        print(f"     abstentions        : {len(absn)}")
        print(f"     total              : {len(rows)}")
        return len(fa), len(fr), len(absn)

    def full(r):
        return r["verdict"]

    def timing_content_only(r):
        """What the correspondence and integrity layers decide on their own.

        The consensus gate is lifted hypothetically. This exists to separate two
        very different questions: whether the overall gate is safe (it is, because
        it abstains) and whether the timing/content layers are discriminative on
        their own.
        """
        if not r["IDENTITY"]["consensus_available"]:
            t = r["TIMING"]
            c = r["CONTENT"]
            if t["dp_outcome"] in ("INTERVAL_ANCHORS_INCONSISTENT",
                                   "INTERVAL_NO_CORRESPONDENCE"):
                return ABSTAIN
            if c["integrity"] == "CONTENT_INTEGRITY_SUSPECT":
                return REJECT
            if t["dp_outcome"] != "INTERVAL_ALIGNED":
                return REJECT
            if c["integrity"] == "CONTENT_INTEGRITY_UNKNOWN":
                return ABSTAIN
            if not r["STRUCTURE"]["same_structure"]:
                return REJECT
            return ACCEPT
        return r["verdict"]

    print("  MODE A: combined gate, identity evidence required (as designed)")
    score("full gate", full)
    print()
    print("  MODE B: consensus gate lifted, timing + content + structure only")
    score("timing/content only", timing_content_only)
    print()

    print()
    print("  DP runtime:")
    for r in sorted(rows, key=lambda x: -x["_runtime_ms"])[:6]:
        print(f"      {r['case']:<28}{r['_runtime_ms']:>8}ms "
              f"(target {r['CONTENT']['cue_count_ratio']}x reference)")
    total = sum(r["_runtime_ms"] for r in rows)
    print(f"      total across {len(rows)} cases: {total:.0f}ms "
          f"(mean {total/len(rows):.1f}ms)")

    out = ROOT / "reports" / "combined_shadow_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"\n  written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
