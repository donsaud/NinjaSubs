"""Measure the Large Offset Investigation signals on the real fixtures.

Bounded and deterministic: reads only tests/fixtures/large_offset, prints one
row per case, exits. Used to set the signal floors against evidence.
"""
from __future__ import annotations

import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.subtitle_matcher import (  # noqa: E402
    FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
    parse_srt_cues,
)
from app.services.sync.large_offset_investigation import (  # noqa: E402
    LargeOffsetSameEpisode,
    investigate_large_offset,
    validate_alass_output,
)

FIX = pathlib.Path("tests") / "fixtures" / "large_offset"


def diagnostics_dir() -> pathlib.Path:
    """The Dexter S08E04 working files this calibration reads, or exit.

    Deliberately not committed: the Alass outputs are per-machine artefacts.
    The committed equivalents used by the test suite live under FIX.
    """
    raw = os.environ.get("NINJASUBS_DIAGNOSTICS_DIR", "")
    if not raw:
        raise SystemExit(
            "Set NINJASUBS_DIAGNOSTICS_DIR to the folder holding this "
            "calibration's working files (01_EVOLV_original.srt, "
            "02_EVOLV_alass_output.srt, ...)."
        )
    return pathlib.Path(raw)


def read(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


def shift(text: str, delta_ms: int) -> str:
    from app.services.subtitle_matcher import parse_srt_cues

    out = []
    for i, (s, e, t) in enumerate(parse_srt_cues(text), start=1):
        s2, e2 = max(0, s + delta_ms), max(0, e + delta_ms)

        def ts(ms: int) -> str:
            return (
                f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:"
                f"{ms // 1000 % 60:02d},{ms % 1000:03d}"
            )

        out.append(f"{i}\n{ts(s2)} --> {ts(e2)}\n{t}\n")
    return "\n".join(out)


def corrupt(text: str, fraction: float, seed: int = 7) -> str:
    """Drop a deterministic fraction of cues and perturb their timing."""
    from app.services.subtitle_matcher import parse_srt_cues

    cues = parse_srt_cues(text)
    keep = []
    for i, (s, e, t) in enumerate(cues):
        if (i * seed) % 100 < fraction * 100:
            continue
        keep.append((s + (i % 5) * 700, e + (i % 3) * 900, t))
    return "\n".join(
        f"{i}\n{_ts(s)} --> {_ts(e)}\n{t}\n" for i, (s, e, t) in enumerate(keep, 1)
    )


def duplicate(text: str, factor: int = 2) -> str:
    from app.services.subtitle_matcher import parse_srt_cues

    cues = parse_srt_cues(text)
    out = []
    n = 1
    for _ in range(factor):
        for s, e, t in cues:
            out.append(f"{n}\n{_ts(s)} --> {_ts(e)}\n{t}\n")
            n += 1
    return "\n".join(out)


def _ts(ms: int) -> str:
    ms = max(0, int(ms))
    return (
        f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:"
        f"{ms // 1000 % 60:02d},{ms % 1000:03d}"
    )


def sparse(text: str, keep_every: int = 25) -> str:
    from app.services.subtitle_matcher import parse_srt_cues

    cues = parse_srt_cues(text)[::keep_every]
    return "\n".join(
        f"{i}\n{_ts(s)} --> {_ts(e)}\n{t}\n" for i, (s, e, t) in enumerate(cues, 1)
    )


def main() -> int:
    target = read("dexter_s08e04_target.srt")
    reference = read("dexter_s08e04_reference.srt")

    cases: list[tuple[str, str, str]] = [
        ("POS dexter (real)", target, reference),
        ("POS dexter vs valid_constant", target, read("valid_constant_offset.srt")),
        ("NEG wrong_episode", target, read("negative_wrong_episode.srt")),
        ("NEG different_cut", target, read("negative_different_cut.srt")),
        ("NEG drifting", target, read("negative_drifting.srt")),
        ("NEG insufficient_anchors", target, read("negative_insufficient_anchors.srt")),
        ("POS synthetic +96.5s", shift(reference, 96_500), reference),
        ("POS synthetic +30s", shift(reference, 30_000), reference),
        ("POS synthetic +120s", shift(reference, 120_000), reference),
        ("POS synthetic +170s", shift(reference, 170_000), reference),
        ("NEG corrupt 10%", corrupt(reference, 0.10), reference),
        ("NEG corrupt 20%", corrupt(reference, 0.20), reference),
        ("NEG duplicated x2", duplicate(reference, 2), reference),
        ("NEG sparse", sparse(reference), reference),
        ("NEG empty ref", target, ""),
    ]

    header = (
        f"{'case':<30} {'seed':>8} {'same_episode':<22} "
        f"{'cons':>5} {'gaps':>5} {'lmk':>5} {'cov':>5} {'dens':>5} {'anc':>7}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for name, tgt, ref in cases:
        inv = investigate_large_offset(
            tgt, ref, identity_supported=True, reference_trust="strong"
        )
        ev = inv.evidence

        def f(v, w=5):
            return "n/a" if v is None else f"{v:.{w - 2}f}"

        print(
            f"{name:<30} {str(inv.seed_offset_ms):>8} "
            f"{inv.same_episode.value:<22} "
            f"{f(ev.offset_consistency_score):>5} "
            f"{f(ev.gap_distribution_similarity):>5} "
            f"{f(ev.silence_landmark_agreement):>5} "
            f"{f(ev.temporal_coverage):>5} "
            f"{f(ev.density_profile_similarity):>5} "
            f"{ev.anchor_count}/{ev.regions_sampled:<4}"
        )

    print()
    print("-- same-episode tallies --")
    for verdict in LargeOffsetSameEpisode:
        names = [
            n
            for n, t, r in cases
            if investigate_large_offset(
                t, r, identity_supported=True, reference_trust="strong"
            ).same_episode
            is verdict
        ]
        print(f"  {verdict.value:<24} {len(names):>2}  {', '.join(names)}")

    print()
    print("-- post-alass validation on the known-good alass outputs --")
    diag = diagnostics_dir()
    pairs = [
        ("EVOLV alass", "01_EVOLV_original.srt", "02_EVOLV_alass_output.srt"),
        ("ASAP alass", "03_ASAP_original.srt", "04_ASAP_alass_output.srt"),
        ("synthetic alass", "06_synthetic_plus96.500s.srt",
         "07_synthetic_plus96.500s_alass_output.srt"),
    ]
    ref_text = (diag / "05_PiR8_reference.srt").read_text(encoding="utf-8", errors="replace")
    for label, orig_name, out_name in pairs:
        orig = (diag / orig_name).read_text(encoding="utf-8", errors="replace")
        out = (diag / out_name).read_text(encoding="utf-8", errors="replace")
        v = validate_alass_output(orig, out, ref_text)
        print(f"  {label:<20} {v.summary()}")

    print()
    print("-- control: validating the ORIGINAL as if alass had produced it --")
    for label, orig_name in [
        ("EVOLV original", "01_EVOLV_original.srt"),
        ("ASAP original", "03_ASAP_original.srt"),
    ]:
        orig = (diag / orig_name).read_text(encoding="utf-8", errors="replace")
        v = validate_alass_output(orig, orig, ref_text)
        print(f"  {label:<20} {v.summary()}")

    print()
    print(f"normal window = {FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS}ms")
    measure_serving_boundary(target, reference)
    end_to_end_serving(target)
    return 0


def end_to_end_serving(target_text: str) -> None:
    """Run investigation -> alass output -> analyzer -> serving decision.

    Uses the *real* alass outputs produced inside Docker, so the movement MAD
    fed to the decision is measured rather than simulated. Expected outcomes are
    the ones agreed from the evidence table: both real cases and both good
    fixture pairs are served, the structurally broken and drifting negatives are
    refused, and ``different_cut`` is the documented known false accept.
    """
    from app.services.sync import large_offset as gate
    from app.services.sync.alignment import AlignmentAnalyzer
    from app.services.sync.large_offset_investigation import (
        LargeOffsetServingState,
        decide_large_offset_serving,
        investigate_large_offset,
        validate_alass_output,
    )

    FIX = pathlib.Path("tests") / "fixtures" / "large_offset"
    OUT = FIX / "alass_out"
    D = diagnostics_dir()

    def read(p):
        return pathlib.Path(p).read_text(encoding="utf-8", errors="replace")

    pairs: list[tuple[str, str, str, str]] = []
    for label, refname in [
        ("GOOD dexter_ref", "dexter_s08e04_reference.srt"),
        ("GOOD valid_const", "valid_constant_offset.srt"),
        ("BAD wrong_episode", "negative_wrong_episode.srt"),
        ("BAD different_cut", "negative_different_cut.srt"),
        ("BAD drifting", "negative_drifting.srt"),
    ]:
        pairs.append(
            (
                label,
                target_text,
                read(FIX / refname),
                read(OUT / ("out_" + pathlib.Path(refname).stem + ".srt")),
            )
        )
    ref_real = read(D / "05_PiR8_reference.srt")
    for label, orig, outn in [
        ("REAL EVOLV", "01_EVOLV_original.srt", "02_EVOLV_alass_output.srt"),
        ("REAL ASAP", "03_ASAP_original.srt", "04_ASAP_alass_output.srt"),
        (
            "REAL synth+96.5",
            "06_synthetic_plus96.500s.srt",
            "07_synthetic_plus96.500s_alass_output.srt",
        ),
    ]:
        pairs.append((label, read(D / orig), ref_real, read(D / outn)))

    print()
    print("-- end to end: investigation -> validation -> analyzer -> serving --")
    print(
        f"{'case':<20} {'eligible':<9} {'valid':<7} {'MAD':>9} "
        f"{'struct':>7}  serving_state"
    )
    print("-" * 92)

    for label, tgt, ref, out in pairs:
        inv = investigate_large_offset(
            tgt, ref, identity_supported=True, reference_trust="strong"
        )
        val = validate_alass_output(tgt, out, ref)
        ev = AlignmentAnalyzer().analyze(
            parse_srt_cues(tgt),
            parse_srt_cues(out),
            parse_srt_cues(ref),
            alass_applied=True,
            alass_successful=True,
            max_plausible_offset_ms=gate.LARGE_OFFSET_MAX_MS,
        )
        state = decide_large_offset_serving(inv, val, ev)
        st = ev.structural_similarity
        print(
            f"{label:<20} {str(inv.eligible_for_alass):<9} {str(val.ok):<7} "
            f"{str(ev.mad_offset_ms):>9} "
            f"{str(round(st, 3) if st else None):>7}  "
            f"{state.value:<30} {','.join(inv.serving_reason_codes)}"
        )

    print()
    print("  (KNOWN FALSE ACCEPT by design: BAD different_cut)")
    print(
        f"  normal window = {FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS}ms; "
        f"ALASS_CORRECTED_LARGE_OFFSET is a serving state, never a SyncState"
    )
    assert LargeOffsetServingState.ALASS_CORRECTED_LARGE_OFFSET != "verified_synced"


def _simulate_good_alignment(target: str, reference: str, offset: float):
    """Pair each target cue with the reference cue it corresponds to.

    The shape of a successful alass run for a constant displacement, and the
    shape alass would produce for a *wrong* episode just as happily -- which is
    precisely why timing evidence alone cannot separate the two.
    """
    import bisect

    from app.services.subtitle_matcher import parse_srt_cues

    target_cues = parse_srt_cues(target)
    reference_cues = parse_srt_cues(reference)
    starts = [start for start, _, _ in reference_cues]
    used: set[int] = set()
    synced = []
    for start, end, text in target_cues:
        predicted = start - offset
        index = bisect.bisect_left(starts, predicted)
        options = [
            i
            for i in range(max(0, index - 3), min(len(starts), index + 4))
            if i not in used
        ]
        if not options:
            synced.append((int(start - offset), int(end - offset), text))
            continue
        best = min(options, key=lambda i: abs(starts[i] - predicted))
        used.add(best)
        landed = starts[best]
        synced.append((landed, landed + (end - start), text))
    return target_cues, synced, reference_cues


def measure_serving_boundary(target: str, reference: str) -> None:
    """Where does the existing verifier sit on the good cases vs the bad ones?

    This is the boundary the new serving decision has to respect rather than
    override. If the analyzer refuses the good Dexter alignment for a reason
    that is *not* the segmentation-driven p95, no override is defensible; if it
    refuses the bad ones for structural reasons, those reasons must stay
    load-bearing.
    """
    from app.services.subtitle_matcher import parse_srt_cues
    from app.services.sync import large_offset as gate
    from app.services.sync.alignment import AlignmentAnalyzer

    print()
    print("-- analyzer verdicts (the boundary the new path must respect) --")

    cases: list[tuple[str, str, str, float | None]] = [
        ("GOOD dexter aligned", target, reference, None),
        ("GOOD dexter sloppy", target, reference, "sloppy"),
        ("BAD wrong_episode", target, read("negative_wrong_episode.srt"), "raw"),
        ("BAD different_cut", target, read("negative_different_cut.srt"), "shifted"),
        ("BAD drifting", target, read("negative_drifting.srt"), "raw"),
    ]

    seed = 96_830
    for label, tgt, ref, mode in cases:
        if mode in (None, "sloppy"):
            tc, synced, rc = _simulate_good_alignment(tgt, ref, seed)
            if mode == "sloppy":
                synced = [
                    (s - seed + (4_000 if i % 2 else -4_000), e - seed, t)
                    for i, (s, e, t) in enumerate(tc)
                ]
        elif mode == "raw":
            tc, synced, rc = (
                parse_srt_cues(tgt),
                parse_srt_cues(ref),
                parse_srt_cues(ref),
            )
        else:  # shifted
            tc = parse_srt_cues(tgt)
            rc = parse_srt_cues(ref)
            synced = [(s + 96_000, e + 96_000, t) for s, e, t in rc]

        evaluation = AlignmentAnalyzer().analyze(
            tc,
            synced,
            rc,
            alass_applied=True,
            alass_successful=True,
            max_plausible_offset_ms=gate.LARGE_OFFSET_MAX_MS,
        )
        print(f"  {label:<22} state={evaluation.sync_state.value:<18} "
              f"reason={str(evaluation.rejection_reason):<32} "
              f"p95={getattr(evaluation, 'p95_offset_ms', 'n/a')}")

    print()
    print("-- stage 2 on a plausible alass output for each pair --")
    for label, ref in [
        ("GOOD dexter", reference),
        ("BAD wrong_episode", read("negative_wrong_episode.srt")),
        ("BAD different_cut", read("negative_different_cut.srt")),
        ("BAD drifting", read("negative_drifting.srt")),
    ]:
        _, synced, _ = _simulate_good_alignment(target, ref, seed)
        out = "\n".join(
            f"{i}\n{_ts(s)} --> {_ts(e)}\n{t}\n"
            for i, (s, e, t) in enumerate(sorted(synced), 1)
        )
        v = validate_alass_output(target, out, ref)
        print(f"  {label:<20} ok={v.ok} {v.summary()}")


if __name__ == "__main__":
    raise SystemExit(main())
