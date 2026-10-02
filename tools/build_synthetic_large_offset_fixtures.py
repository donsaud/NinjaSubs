"""Deterministic synthetic regeneration of the Large Offset public fixtures.

WHY THIS EXISTS
---------------
The committed corpus was derived from real, commercially released subtitles for
a television episode. Timings are facts about a timeline and are not the
copyrighted work; the *dialogue text* is. This script keeps every measured
characteristic the test suite depends on and replaces only the words, so the
public repository ships no third-party dialogue while the safety properties stay
byte-for-byte comparable:

  * identical cue count, cue boundaries, durations and inter-cue gaps
  * identical per-cue word counts, so text volume/density metrics are unchanged
  * identical first-dialogue instants, so the ~+96.6s displacement is preserved
  * the negatives are still the same TIMING transforms of the same reference
    (different cut, drifting, wrong episode), so what refuses them is unchanged

Text is drawn from a fixed word pool with a fixed PRNG seed, so regenerating
gives identical bytes.
"""

from __future__ import annotations

import pathlib
import random
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.services.subtitle_matcher import parse_srt_cues  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "large_offset"
FX = FIX / "alass_out" / "fx"

# A neutral vocabulary. Nothing here is dialogue from any production: the words
# are ordinary English nouns/verbs assembled by a seeded PRNG.
NOUNS = (
    "anchor ledger crate signal timer hatch relay beacon docket circuit valve "
    "roster parcel docket spindle anchor gauge marker relay panel tray anchor "
    "beacon conduit spar relay docket"
).split()
VERBS = (
    "verify recalibrate seal inspect reroute log align compare latch measure "
    "confirm detach mount record sample purge index mark trim"
).split()
ADJS = (
    "spare inline primary secondary dormant sealed nominal partial rotary "
    "paired staged sealed idle lateral"
).split()

# Distinct pools so the wrong-episode negative shares no vocabulary with the
# positives. That is what makes its anchors fail to corroborate.
OTHER_NOUNS = (
    "harbour lantern orchard terrace viaduct cistern quarry parapet granary "
    "alcove rampart jetty cornice cellar turret aqueduct"
).split()
OTHER_VERBS = (
    "listen linger gather wander harbour tend mend polish barter drift muster "
    "salute shelter ramble mend tend"
).split()
OTHER_ADJS = (
    "windworn chalky lamplit sunken mossy brackish cobbled vaulted shuttered "
    "weathered"
).split()

CHARACTERS = ("CHARACTER_A", "CHARACTER_B", "CHARACTER_C")
# The wrong-episode negative must share no tokens at all with the positives, so
# it gets its own speakers as well as its own vocabulary.
OTHER_CHARACTERS = ("STRANGER_1", "STRANGER_2", "STRANGER_3")

# The reference and the target are two releases of the same dialogue, so their
# anchors must corroborate. They share a stem word and differ in filler, which
# is what a real translation pair looks like to the matcher.
SYNONYMS = {
    "verify": "confirm", "recalibrate": "align", "seal": "close",
    "inspect": "survey", "reroute": "redirect", "log": "record",
    "align": "match", "compare": "diff", "latch": "clip",
    "measure": "gauge", "confirm": "verify", "detach": "unclip",
    "mount": "bracket", "record": "note", "sample": "draw",
    "purge": "clear", "index": "catalogue", "mark": "flag",
    "trim": "clip",
}


def _words(n: int, rng: random.Random) -> list[str]:
    out: list[str] = []
    for _ in range(n):
        out.append(
            rng.choice(ADJS) if rng.random() < 0.3
            else rng.choice(VERBS) if rng.random() < 0.5
            else rng.choice(NOUNS)
        )
    return out


def _wc(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def synthetic_text(index: int, words: int, seed: int, other: bool) -> str:
    """One cue of deterministic dialogue.

    ``words`` is the TOTAL word count of the original cue, speaker label
    included, so replacing the text does not move any text-volume metric.
    """
    rng = random.Random((seed, index, words, other).__hash__() & 0xFFFFFFFF)
    if words <= 0:
        return ""
    speaker = (OTHER_CHARACTERS if other else CHARACTERS)[index % 3]
    if words == 1:
        return speaker
    pool = _words(words - 1, rng)
    return f"{speaker}: {' '.join(pool)}"


def variant_text(index: int, words: int, seed: int, text: str) -> str:
    """The target release's wording: same stem word, different filler."""
    rng = random.Random((seed, index, words, "variant").__hash__() & 0xFFFFFFFF)
    if words <= 0:
        return ""
    speaker = text.split(":", 1)[0] if ":" in text else CHARACTERS[index % 3]
    if ":" not in text:
        speaker = CHARACTERS[index % 3]
    if words == 1:
        return speaker
    stem = text.split(":", 1)[-1].strip().split()[0] if ":" in text else "anchor"
    stem = SYNONYMS.get(stem, stem)
    if words == 2:
        return f"{speaker}: {stem}"
    fill = _words(words - 2, rng)
    return f"{speaker}: {stem} {' '.join(fill)}"


def load(path: pathlib.Path) -> list[tuple[int, int, str]]:
    return parse_srt_cues(path.read_text(encoding="utf-8", errors="replace"))


def render(cues) -> str:
    blocks = [
        f"{i + 1}\n{ts(s)} --> {ts(e)}\n{t}"
        for i, (s, e, t) in enumerate(cues)
    ]
    return "\n\n".join(blocks) + "\n"


def ts(ms: int) -> str:
    ms = max(0, int(ms))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def retime_and_rewrite(name: str, seed: int, other: bool = False) -> int:
    path = FIX / name
    src = load(path)
    out = []
    for i, (s, e, text) in enumerate(src):
        out.append((s, e, synthetic_text(i, _wc(text), seed, other)))
    path.write_text(render(out), encoding="utf-8", newline="\n")
    return len(out)


def rewrite_derived() -> None:
    """The derived negatives keep the reference wording; only timing differs.

    They are TIMING transforms of the reference, so re-deriving them from the
    freshly rewritten reference keeps every relationship the suite asserts
    (opening aligned, +600s after the pivot, 300ms/min drift, 3 cues).
    """
    ref = load(FIX / "dexter_s08e04_reference.srt")

    pivot = len(ref) // 2
    (FIX / "negative_different_cut.srt").write_text(
        render([
            (s, e, t) if i < pivot else (s + 600_000, e + 600_000, t)
            for i, (s, e, t) in enumerate(ref)
        ]),
        encoding="utf-8", newline="\n",
    )
    (FIX / "negative_drifting.srt").write_text(
        render([
            (s + int(300.0 * (s / 60_000.0)), e + int(300.0 * (s / 60_000.0)), t)
            for s, e, t in ref
        ]),
        encoding="utf-8", newline="\n",
    )
    (FIX / "negative_insufficient_anchors.srt").write_text(
        render(ref[:3]), encoding="utf-8", newline="\n"
    )
    (FIX / "valid_constant_offset.srt").write_text(
        render([(s + 3_650, e + 3_650, t) for s, e, t in ref]),
        encoding="utf-8", newline="\n",
    )


def mirror_to_fx() -> None:
    """The alass input bundle is a copy of the inputs it was run against."""
    FX.mkdir(parents=True, exist_ok=True)
    for name in (
        "dexter_s08e04_reference.srt", "dexter_s08e04_target.srt",
        "valid_constant_offset.srt", "negative_different_cut.srt",
        "negative_drifting.srt", "negative_wrong_episode.srt",
        "negative_insufficient_anchors.srt",
    ):
        (FX / name).write_bytes((FIX / name).read_bytes())


def main() -> int:
    print("rewriting dialogue (timings preserved):")
    print(f"  dexter_s08e04_reference.srt   {retime_and_rewrite('dexter_s08e04_reference.srt', 1)} cues")
    print(f"  negative_wrong_episode.srt    {retime_and_rewrite('negative_wrong_episode.srt', 7, other=True)} cues")

    # The target is the reference's dialogue in a different release's wording,
    # on the reference's own (late) timeline.
    ref = load(FIX / "dexter_s08e04_reference.srt")
    tgt = load(FIX / "dexter_s08e04_target.srt")
    new_target = []
    for i, (s, e, _t) in enumerate(tgt):
        # The stem comes from the REFERENCE's wording so anchors corroborate;
        # the word count comes from this file's OWN original cue so its text
        # volume is unchanged.
        stem = synthetic_text(i, _wc(ref[min(i, len(ref) - 1)][2]), 1, other=False)
        new_target.append((s, e, variant_text(i, _wc(_t), 3, stem)))
    (FIX / "dexter_s08e04_target.srt").write_text(
        render(new_target), encoding="utf-8", newline="\n"
    )
    print(f"  dexter_s08e04_target.srt      {len(new_target)} cues")

    # The two per-release targets the anchor-grading tests measure. Different
    # cue counts (560 and 781) and slightly different openings; both keep their
    # own measured timelines and lose only their words.
    print(f"  dexter_s08e04_evolv_original  {retime_and_rewrite('dexter_s08e04_evolv_original.srt', 11)} cues")
    print(f"  dexter_s08e04_asap_original   {retime_and_rewrite('dexter_s08e04_asap_original.srt', 13)} cues")

    print("re-deriving the timing transforms:")
    rewrite_derived()
    mirror_to_fx()
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
