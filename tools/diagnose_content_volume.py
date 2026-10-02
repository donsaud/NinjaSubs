"""PART 2 follow-up: is ASAP's surplus content duplication or additional lines?

Counts only. No subtitle text is ever printed or stored -- normalised text is
hashed immediately and discarded. The question decides whether the surplus is a
content problem or a segmentation problem, which changes what the model should do.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.subtitle_matcher import parse_srt_cues  # noqa: E402

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


def norm_hash(text: str) -> str:
    """Deterministic normalisation, then hash. The text never leaves here."""
    cleaned = _NON_WORD.sub(" ", text.lower()).strip()
    if not cleaned:
        return ""
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:16]


def hashes(path: pathlib.Path) -> list[str]:
    parsed = parse_srt_cues(path.read_text(encoding="utf-8", errors="replace"))
    return [norm_hash(t) for _, _, t in parsed]


def multiset(path: pathlib.Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for h in hashes(path):
        if h:
            out[h] = out.get(h, 0) + 1
    return out


def compare(label: str, target_name: str, ref_name: str) -> None:
    tgt = multiset(FIXTURES / target_name)
    ref = multiset(FIXTURES / ref_name)
    tgt_n = sum(tgt.values())
    ref_n = sum(ref.values())

    # reference lines present in target, respecting multiplicity
    matched_ref = sum(min(c, tgt.get(h, 0)) for h, c in ref.items())
    # target lines present in reference, respecting multiplicity
    matched_tgt = sum(min(c, ref.get(h, 0)) for h, c in tgt.items())

    ref_dupes = sum(c - 1 for c in ref.values() if c > 1)
    tgt_dupes = sum(c - 1 for c in tgt.values() if c > 1)

    print("=" * 70)
    print(f"{label}")
    print("=" * 70)
    print(f"  target cues                 : {tgt_n}")
    print(f"  reference cues              : {ref_n}")
    print(f"  reference lines found in target     : {matched_ref} "
          f"({100*matched_ref/max(1,ref_n):.1f}%)")
    print(f"  target lines found in reference     : {matched_tgt} "
          f"({100*matched_tgt/max(1,tgt_n):.1f}%)")
    print(f"  target cues with NO reference text  : {tgt_n - matched_tgt}")
    print(f"  reference cues with NO target text  : {ref_n - matched_ref}")
    print(f"  repeated lines within target        : {tgt_dupes}")
    print(f"  repeated lines within reference     : {ref_dupes}")
    surplus = tgt_n - matched_tgt
    if surplus > 0:
        print(f"  surplus explained by REPEATS        : "
              f"{min(surplus, tgt_dupes)} of {surplus}")
    print()


def main() -> int:
    print("Text is hashed and discarded; only counts are shown.\n")
    compare("ASAP original vs PiR8 reference", "03_ASAP_original.srt",
            "05_PiR8_reference.srt")
    compare("ASAP alass vs PiR8 reference", "04_ASAP_alass_output.srt",
            "05_PiR8_reference.srt")
    compare("EVOLV original vs PiR8 reference", "01_EVOLV_original.srt",
            "05_PiR8_reference.srt")
    compare("EVOLV alass vs PiR8 reference", "02_EVOLV_alass_output.srt",
            "05_PiR8_reference.srt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
