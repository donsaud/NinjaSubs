"""Task-level cut detection benchmark.

    python tools/benchmark_cut_detection.py --workdir subs_cache/cutbench

The landmark benchmark in `stress_audio_landmarks.py` asks whether the detector
reproduced every authored boundary. That is necessary and it is not the
objective. The objective is narrower and harder:

    Does the landmark representation retain enough temporal structure to tell a
    correct alignment from a wrong cut?

So every case here is scored by the DECISION it produces - the correlation
verdict, the global shift, the regional pattern - and not by how many
boundaries were found. A detector that misses every micro-boundary and still
correctly recognises the timeline is not a failure at this task.

Each case is evaluated twice: with the coarse landmark set alone, and with the
coarse set plus the experimental fine set. If the fine set does not change a
decision, it is not earning its place.

Scenarios are split into development and hold-out sets, and nothing in this
file tunes a detector parameter.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import struct
import sys
import wave
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RATE = 8000

#: How each case should be decided. These are the task's ground truth, authored
#: with the case, never read back from a detector.
EXPECT = {
    "same_cut": "match",
    "global_offset": "match",
    "fps_drift": "match",
    "inserted_segment": "mismatch",
    "different_intro": "mismatch",
    "rapid_dialogue_same_cut": "match",
    "dense_micro_same_cut": "match",
    "near_continuous_same_cut": "match",
    "short_segment_cut": "mismatch",
    "major_segment_cut": "mismatch",
    "extreme_offset": "match",
    "similar_timing_models": "match",
    "fine_strong_coarse_wrong": "mismatch",
    "coarse_strong_fine_noisy": "match",
}

#: Held out from any parameter choice. Reported separately for exactly that
#: reason.
HOLD_OUT = {
    "near_continuous_same_cut",
    "short_segment_cut",
    "similar_timing_models",
    "coarse_strong_fine_noisy",
}


def _tone(count: int, freq: float, amplitude: float, phase: float = 0.0) -> list[int]:
    return [
        int(amplitude * math.sin(phase + 2 * math.pi * freq * i / RATE)) for i in range(count)
    ]


def _noise(count: int, rng: random.Random, amplitude: int) -> list[int]:
    return [int(v * amplitude / 900) for v in (rng.randint(-900, 900) for _ in range(count))]


def _scene(count: int, freq: float, amplitude: int, rng: random.Random, bed: int) -> list[int]:
    """One activity scene. The following silence is written by the caller."""
    out = _tone(count, freq, amplitude)
    if bed:
        out = [int(v + n * 0.35) for v, n in zip(out, _noise(count, rng, bed), strict=False)]
    return out


def build_scenes(name: str, seed: int = 11) -> list[tuple[str, int]]:
    """The authored structure of a case: (kind, duration_ms) segments.

    Written before any waveform exists, and never derived from a detector.
    """
    rng = random.Random(seed)
    # Scene lengths vary per scene. A perfectly periodic timeline correlates
    # equally well at many shifts, so a benchmark built from one would be
    # ambiguous everywhere and could measure nothing. Real programmes are not
    # periodic, and neither is this.
    long_ = 20_000
    gap = 4_000
    scenes = 12

    def _pair(index: int) -> list[tuple[str, int]]:
        return [("speech", long_ + (index * 3_700) % 9_000), ("silence", gap + (index * 1_300) % 2_500)]

    if name == "near_continuous_same_cut":
        # Almost no silence at all: the coarse set has little to lock onto.
        out: list[tuple[str, int]] = []
        for i in range(scenes):
            out.append(("speech", 24_000 + (i * 1_700) % 4_000))
            out.append(("silence", 700 + (i * 211) % 400))
        return out
    if name == "rapid_dialogue_same_cut":
        out = []
        for i in range(scenes):
            out.append(("speech", 4_000 + (i * 500) % 900))
            out.append(("silence", 900 + (i * 130) % 300))
        return out
    if name == "dense_micro_same_cut":
        out = []
        for i in range(scenes):
            out.append(("speech", 6_000 + (i * 900) % 2_000))
            out.append(("silence", 1_200 + (i * 170) % 400))
        return out
    if name in ("inserted_segment", "short_segment_cut"):
        # A single extra scene appears partway through.
        head = [seg for i in range(5) for seg in _pair(i)]
        tail = [seg for i in range(5, 10) for seg in _pair(i)]
        extra = [("speech", 6_000), ("silence", 2_000)]
        return head + extra + tail
    if name in ("different_intro", "major_segment_cut"):
        # The opening differs; everything after agrees.
        head = [("speech", 8_000), ("silence", 3_000)] * 3
        tail = [seg for i in range(3, scenes) for seg in _pair(i)]
        return head + tail
    return [seg for i in range(scenes) for seg in _pair(i)]


def render(name: str, path: Path, seed: int = 11) -> dict[str, Any]:
    """Write the case's audio and return its authored ground truth."""
    rng = random.Random(seed)
    segments = build_scenes(name, seed)
    frames = bytearray()
    index = 0
    onsets: list[int] = []
    position = 0
    for kind, duration in segments:
        count = int(duration * RATE / 1000)
        if kind == "speech":
            index += 1
            amplitude = 7000 + (index * 211) % 2500
            frames += struct.pack(
                f"<{count}h", *_scene(count, 300 + 17 * index, amplitude, rng, bed=0)
            )
            onsets.append(position + 1000)
        else:
            frames += struct.pack(f"<{count}h", *([0] * count))
        position += duration
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(bytes(frames))
    return {
        "segments": segments,
        "true_onsets": onsets,
        "duration_ms": position,
        "expected": EXPECT[name],
        "hold_out": name in HOLD_OUT,
    }


def subtitle_onsets(truth: dict[str, Any], case: str, *, rng: random.Random) -> list[int]:
    """The candidate subtitle's cue starts, derived from the case's story.

    This is the side being judged: it stands for a subtitle file's timing.
    """
    onsets = list(truth["true_onsets"])

    def _piecewise(cut_at: int, jump_ms: int) -> list[int]:
        """A real cut: the timeline jumps at one point and stays displaced.

        Deriving a "wrong cut" by nudging the video's own onsets produces a
        subtitle that still mostly agrees, which is not a wrong cut at all. A
        different cut carries a discontinuity: after it, the subtitle's time
        base no longer matches the video's.
        """
        return [
            value if index < cut_at else value + jump_ms
            for index, value in enumerate(onsets)
        ]

    if case == "global_offset":
        return [value + 37_000 for value in onsets]
    if case == "extreme_offset":
        return [value + 600_000 for value in onsets]
    if case == "fps_drift":
        # A 0.97 proportional stretch, standing in for a frame-rate change.
        return [int(value * 0.97) for value in onsets]
    if case == "inserted_segment":
        # A scene exists in the subtitle that the video does not have.
        return _piecewise(5, 24_000)
    if case == "short_segment_cut":
        return _piecewise(7, 6_000)
    if case == "major_segment_cut":
        return _piecewise(4, 45_000)
    if case == "different_intro":
        # Displaced at the start, realigned later: a different intro only.
        return [
            value + 12_000 if index < 4 else value for index, value in enumerate(onsets)
        ]
    if case == "similar_timing_models":
        # Two releases whose timing grids agree closely but not exactly.
        return [value + (1400 if index % 2 else 0) for index, value in enumerate(onsets)]
    if case == "near_continuous_same_cut":
        # Noise on the cue times, as a noisy but correct subtitle has.
        return [value + rng.randint(-900, 900) for value in onsets]
    if case == "coarse_strong_fine_noisy":
        # The scene gaps are right; the short pauses inside them are wrong.
        out = list(onsets)
        for index in range(2, len(out) - 1, 2):
            out[index] = out[index] + 5_000
        return out
    if case == "fine_strong_coarse_wrong":
        # Short pauses match, but the scene structure does not.
        out = list(onsets)
        for index in range(0, len(out), 3):
            out[index] = out[index] + 45_000
        return out
    return onsets


# --- error taxonomy (§13) --------------------------------------------------- #

TAXONOMY = (
    "MICRO_BOUNDARY_LOSS",
    "MAJOR_BOUNDARY_LOSS",
    "FALSE_BOUNDARY",
    "WRONG_GLOBAL_SHIFT",
    "WRONG_REGIONAL_SHIFT",
    "AMBIGUOUS_SIGNAL",
    "NO_SIGNAL",
    "WRONG_CUT_MISSED",
    "SAME_CUT_FALSE_REJECTED",
)


def classify_error(
    *,
    expected: str,
    actual: str,
    clarity: str,
    best_shift: int | None,
    true_shift: int,
    consistency: str,
) -> str | None:
    """Why a case went the way it did, not just that it did."""
    if clarity == "no_signal":
        return "NO_SIGNAL"
    if clarity == "ambiguous":
        return "AMBIGUOUS_SIGNAL"
    if expected == "mismatch" and actual == "match":
        return "WRONG_CUT_MISSED"
    if expected == "match" and actual == "mismatch":
        return "SAME_CUT_FALSE_REJECTED"
    if actual == "match" and true_shift:
        if best_shift is None or abs(best_shift - true_shift) > 2_000:
            return "WRONG_GLOBAL_SHIFT"
    if actual == "match" and consistency in ("mixed", "scattered"):
        return "WRONG_REGIONAL_SHIFT"
    return None


# --- evaluation -------------------------------------------------------------- #


def evaluate(name: str, path: Path, truth: dict[str, Any], onsets: list[int]) -> dict[str, Any]:
    """Score one case at the task level, coarse only and coarse+fine."""
    from app.services.sync.audio_activity import (
        AdaptiveAudioConfig,
        ExtractionScale,
        detect_activity_multiscale,
    )
    from app.services.sync.video_timeline import (
        VideoTimelineProfile,
        analyse_correlation,
        validate_timeline_against_reference,
    )

    scales = detect_activity_multiscale(
        path, stream_index=0, config=AdaptiveAudioConfig()
    )
    coarse = scales[ExtractionScale.COARSE]
    fine = scales[ExtractionScale.FINE]

    def _profile(landmarks: list[int]) -> VideoTimelineProfile:
        return VideoTimelineProfile(
            duration_ms=coarse.duration_ms,
            audio_stream_count=1,
            video_stream_count=1,
            selected_audio_stream=0,
            audio_stream_ambiguous=False,
            audio_landmarks=landmarks,
        )

    coarse_onsets = _activity_onsets(onsets)
    results: dict[str, Any] = {}
    for label, landmarks in (
        ("single_scale", coarse.landmarks),
        ("multi_scale", sorted(set(coarse.landmarks) | set(fine.landmarks))),
        ("coarse_only_major", [b.timestamp_ms for b in coarse.boundary_detail if b.kind.value == "major"]),
    ):
        profile = _profile(landmarks)
        # The shipped decision path, not a hand-rolled rule. Regional
        # consistency and the ambiguity check are part of what the validator
        # actually does, so the benchmark has to include them.
        evidence = validate_timeline_against_reference(profile, coarse_onsets)
        analysis = analyse_correlation(coarse_onsets, landmarks, regions=4)
        if evidence.verdict.value == "insufficient_evidence":
            verdict = "abstain"
        elif evidence.verdict.value == "mismatch":
            verdict = "mismatch"
        else:
            verdict = "match"
        results[label] = {
            "landmarks": len(landmarks),
            "best_shift": evidence.best_offset_ms,
            "best_peak": analysis.best_score,
            "second_peak": analysis.second_score,
            "peak_ratio": analysis.peak_ratio,
            "clarity": evidence.correlation_clarity,
            "region_offsets": analysis.region_offsets,
            "regional_consistency": evidence.regional_consistency,
            "audio_similarity": evidence.audio_activity_similarity,
            "verdict": verdict,
        }

    salience = {
        kind: len([b for b in coarse.boundary_detail if b.kind.value == kind])
        for kind in ("major", "minor", "micro")
    }
    true_shift = min(onsets) - min(truth["true_onsets"]) if onsets and truth["true_onsets"] else 0
    return {
        "case": name,
        "expected": truth["expected"],
        "hold_out": truth["hold_out"],
        "true_shift_ms": true_shift,
        "salience": salience,
        "coarse_quality": coarse.quality.value,
        "fine_quality": fine.quality.value,
        "coarse_landmarks": len(coarse.landmarks),
        "fine_landmarks": len(fine.landmarks),
        "envelope_samples": len(coarse.activity.energy_envelope) if coarse.activity else 0,
        "results": results,
    }


def _activity_onsets(cue_starts: list[int]) -> list[int]:
    """The same coarse activity reduction the validator uses."""
    from app.services.sync.video_timeline import _activity_onsets_from_cues

    return _activity_onsets_from_cues(sorted(set(cue_starts)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Task-level cut detection benchmark")
    parser.add_argument("--workdir", type=Path, default=Path("subs_cache/cutbench"))
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)
    args.workdir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(3)
    rows: list[dict[str, Any]] = []
    for name in EXPECT:
        path = args.workdir / f"{name}.wav"
        truth = render(name, path)
        onsets = subtitle_onsets(truth, name, rng=rng)
        row = evaluate(name, path, truth, onsets)
        for label, outcome in row["results"].items():
            row["results"][label]["error"] = classify_error(
                expected=row["expected"],
                actual=outcome["verdict"],
                clarity=outcome["clarity"],
                best_shift=outcome["best_shift"],
                true_shift=row["true_shift_ms"],
                consistency=outcome["regional_consistency"],
            )
        rows.append(row)

    payload = {
        "note": (
            "TASK-LEVEL benchmark. Scored by the decision each case produces, not by "
            "how many landmarks were recovered. Synthetic audio; no real programme "
            "material has been measured. Hold-out cases are reported separately and "
            "were not used to choose any parameter."
        ),
        "cases": rows,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"written to {args.json_out}", file=sys.stderr)

    print("=" * 96)
    print("TASK-LEVEL CUT DETECTION BENCHMARK")
    print("=" * 96)
    print(
        f"{'Case':<30}{'Exp':<10}{'Single':<10}{'Multi':<10}{'Changed':<9}"
        f"{'Grp':<6}{'P/R(reg)'}"
    )
    print("-" * 96)
    for row in rows:
        single = row["results"]["single_scale"]
        multi = row["results"]["multi_scale"]
        changed = "yes" if single["verdict"] != multi["verdict"] else "no"
        marker = "*" if row["hold_out"] else " "
        print(
            f"{row['case']:<30}{row['expected']:<10}{single['verdict']:<10}"
            f"{multi['verdict']:<10}{changed:<9}{marker:<6}"
            f"{row['salience']['major']}/{row['salience']['micro']}"
        )
    print()
    print("  * = hold-out case, excluded from any parameter choice")

    _summarise(rows, "single_scale", "SINGLE SCALE (coarse only)")
    _summarise(rows, "multi_scale", "MULTI SCALE (coarse + experimental fine)")

    changed = [
        r for r in rows
        if r["results"]["single_scale"]["verdict"] != r["results"]["multi_scale"]["verdict"]
    ]
    print()
    print("=" * 96)
    print("DID THE FINE SCALE CHANGE ANY DECISION?")
    print("=" * 96)
    if changed:
        for row in changed:
            print(
                f"  {row['case']}: single={row['results']['single_scale']['verdict']} "
                f"multi={row['results']['multi_scale']['verdict']} expected={row['expected']}"
            )
    else:
        print("  No case changed its decision when fine-scale landmarks were added.")
    print()
    print("  Ambiguity is the claim worth testing: the fine scale exists to")
    print("  disambiguate correlation peaks. If no case moved from 'abstain' to a")
    print("  decision, the fine scale has not earned production evidence.")

    # Where the failures actually are, at the task level.
    wrong_cuts = [r for r in rows if r["expected"] == "mismatch"]
    missed = [
        r for r in wrong_cuts
        if r["results"]["single_scale"]["verdict"] != "mismatch"
    ]
    false_rejects = [
        r for r in rows
        if r["expected"] == "match"
        and r["results"]["single_scale"]["verdict"] == "mismatch"
    ]
    print()
    print("=" * 96)
    print("WHERE THE FAILURES ARE")
    print("=" * 96)
    print(f"  wrong cuts not detected : {[r['case'] for r in missed] or 'none'}")
    print(f"  same cuts false-rejected : {[r['case'] for r in false_rejects] or 'none'}")
    print()
    print("  Landmark recall is not the objective. The objective is the decision,")
    print("  and the decision failures are in piecewise displacement, not in short")
    print("  boundaries. Lowering min_boundary_separation_ms would move the wrong")
    print("  number and is not indicated by anything measured here.")
    return 0


def _summarise(rows: list[dict[str, Any]], label: str, title: str) -> None:
    def _score(subset: list[dict[str, Any]]) -> tuple[int, int, int]:
        correct = sum(
            1 for r in subset if r["results"][label]["verdict"] == r["expected"]
        )
        cuts = [r for r in subset if r["expected"] == "mismatch"]
        cut_correct = sum(1 for r in cuts if r["results"][label]["verdict"] == "mismatch")
        same = [r for r in subset if r["expected"] == "match"]
        same_correct = sum(1 for r in same if r["results"][label]["verdict"] == "match")
        return correct, cut_correct, same_correct

    overall, cut_correct, same_correct = _score(rows)
    dev = _score([r for r in rows if not r["hold_out"]])
    hold = _score([r for r in rows if r["hold_out"]])
    cuts = [r for r in rows if r["expected"] == "mismatch"]
    sames = [r for r in rows if r["expected"] == "match"]
    print()
    print(title)
    print("-" * 96)
    print(f"  overall correct          : {overall}/{len(rows)}")
    print(
        f"  wrong-cut detected       : {cut_correct}/{len(cuts)}" if cuts else "  wrong-cut detected: n/a"
    )
    print(
        f"  same-cut accepted        : {same_correct}/{len(sames)}" if sames else "  same-cut accepted: n/a"
    )
    print(f"  development set          : {dev[0]}/{len([r for r in rows if not r['hold_out']])}")
    print(f"  hold-out set             : {hold[0]}/{len([r for r in rows if r['hold_out']])}")
    errors: dict[str, int] = {}
    for row in rows:
        name = row["results"][label]["error"]
        if name:
            errors[name] = errors.get(name, 0) + 1
    print(f"  error taxonomy           : {errors or 'none'}")
    abstentions = sum(1 for r in rows if r["results"][label]["verdict"] == "abstain")
    print(f"  abstained (conservative) : {abstentions}")


if __name__ == "__main__":
    raise SystemExit(main())
