"""Characterise the audio landmark extractor against hostile audio.

    python tools/stress_audio_landmarks.py --generate

Real programme audio is not available to this repository, so the question §8
asks - is the extractor stable, over-sensitive or under-sensitive on dialogue
over music, ambience, crowd scenes and uneven loudness? - cannot be answered
with real material here. What *can* be done is attack the extractor's actual
assumptions with audio that reproduces the conditions real content creates, and
measure how the landmarks move.

This is a PROXY, and is labelled as one everywhere it reports. Silence
detection depends on an amplitude floor, so the honest question is not "does it
work on tone bursts" but "how far can the noise floor and dynamics move before
the structure is lost, and in which direction".

Audio is generated, never committed. Results are written to a JSON artifact that
is committed, so the numbers are reviewable without the media.
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
#: The ground-truth structure every generated clip follows: a dialogue block
#: then a silence, repeated. The extractor's job is to recover the silences.
BLOCK_S = 20
GAP_S = 4
BLOCKS = 12
TRUE_SILENCE_STARTS = [i * (BLOCK_S + GAP_S) * 1000 for i in range(BLOCKS)]
TRUE_SILENCE_ENDS = [value + GAP_S * 1000 for value in TRUE_SILENCE_STARTS]

SCENARIOS: dict[str, str] = {
    "clean": "Idealised speech-like tone bursts and true silence. The baseline.",
    "ambience": "Continuous low-level noise floor, as in an outdoor or room scene.",
    "music_bed": "A continuous musical bed under the dialogue, as in a scored scene.",
    "crowd": "Many overlapping voices, as in a market or stadium.",
    "effects": "Sharp transient sound effects inside otherwise silent blocks.",
    "uneven_gain": "Loudness varying block to block, as after a scene change.",
    "rapid_dialogue": "Short gaps between lines instead of long ones.",
    "long_silence": "Much longer quiet stretches than a typical scene break.",
    # --- §16: harder conditions, amplitude dynamics only, never speech ------
    "drifting_noise_floor": "A noise floor that rises and falls across the file.",
    "dynamic_music": "A musical bed whose own level moves over time.",
    "dialogue_envelope": "Speech-like amplitude envelopes, not speech content.",
    "overlapping_activity": "Two independent activity sources overlapping.",
    "effects_over_ambience": "Sparse transients over a continuous room tone.",
    "sudden_gain_change": "A step change in level partway through.",
    "quiet_scene_tone": "A long quiet scene sitting on a nonzero room tone.",
    "compressed_dynamics": "Heavy dynamic range reduction, as in loudness-normalised TV.",
    "scene_level_steps": "The background level steps between scenes.",
}


def _noise(rng: random.Random, count: int) -> list[int]:
    return [rng.randint(-900, 900) for _ in range(count)]


def _tone(count: int, freq: float, amplitude: float = 8000.0, phase: float = 0.0) -> list[int]:
    out: list[int] = []
    for i in range(count):
        out.append(int(amplitude * math.sin(phase + 2 * math.pi * freq * i / RATE)))
    return out


def build_scenario(name: str, seed: int = 7) -> list[tuple[str, int]]:
    """Return (kind, duration_ms) segments for one scenario.

    ``kind`` is ``speech`` or ``silence``; the waveform itself is written
    separately so the ground-truth silences stay explicit and checkable.
    """
    rng = random.Random(seed)
    block = BLOCK_S * 1000
    gap = GAP_S * 1000
    segments: list[tuple[str, int]] = []
    if name == "rapid_dialogue":
        block, gap = 4_000, 900
    elif name == "long_silence":
        block, gap = 12_000, 12_000
    for _ in range(BLOCKS):
        segments.append(("speech", block))
        segments.append(("silence", gap))
    return segments


def render_scenario(name: str, path: Path, seed: int = 7) -> list[tuple[str, int]]:
    """Write a WAV for one scenario and return its ground-truth segments.

    The returned segments are the independently authored ground truth. They are
    built before any waveform is synthesised and are never derived from a
    detector's output.
    """
    segments = build_scenario(name, seed)
    rng = random.Random(seed)
    frames = bytearray()
    block_index = 0
    # Level of the continuous background, in dB relative to full scale.
    bed_db = -36.0
    if name in ("drifting_noise_floor", "scene_level_steps"):
        bed_db = -36.0
    elif name == "compressed_dynamics":
        bed_db = -30.0
    elif name == "quiet_scene_tone":
        bed_db = -44.0
    for kind, duration in segments:
        count = int(duration * RATE / 1000)
        if kind == "silence":
            frames += struct.pack(
                f"<{count}h", *_background(name, count, rng, bed_db, block_index, phase)
            )
            continue

        block_index += 1
        gain = 1.0
        if name == "uneven_gain":
            gain = 0.15 + 0.85 * ((block_index * 37) % 11) / 10.0
        phase = rng.random()
        base = _tone(count, 300 + 30 * block_index, amplitude=8000 * gain, phase=phase)
        if name == "music_bed":
            base = [
                int(value + 2600 * math.sin(2 * math.pi * 220 * i / RATE))
                for i, value in enumerate(base)
            ]
        elif name == "ambience":
            base = [int(value + noise * 0.4) for value, noise in zip(base, _noise(rng, count), strict=False)]
        elif name == "crowd":
            for extra in (620, 940, 1310):
                base = [
                    int(value * 0.5 + 3000 * math.sin(2 * math.pi * extra * i / RATE))
                    for i, value in enumerate(base)
                ]
        elif name == "effects":
            # Two sharp transients inside the block.
            for at in (count // 3, 2 * count // 3):
                for k in range(200):
                    if at + k < count:
                        base[at + k] = int(base[at + k] + 12000 * math.exp(-k / 30))
        extra = _overlay(name, count, rng, bed_db, block_index, phase)
        if extra is not None:
            base = [int(value + e) for value, e in zip(base, extra, strict=False)]
        frames += struct.pack(f"<{count}h", *[max(-32000, min(32000, v)) for v in base])

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(bytes(frames))
    return segments


def _background(
    name: str, count: int, rng: random.Random, bed_db: float, block_index: int, phase: float
) -> list[int]:
    """The continuous level present during a 'silence' for this scenario."""
    if name in ("clean", "effects", "uneven_gain", "rapid_dialogue", "long_silence"):
        return [0] * count
    if name in ("ambience", "crowd", "drifting_noise_floor", "scene_level_steps",
                "quiet_scene_tone", "compressed_dynamics", "effects_over_ambience"):
        # Amplitude is derived from the requested dB directly. _noise already
        # produces a known level, so scaling it again would double-attenuate
        # and quietly turn a hard case into a trivially silent one.
        level = bed_db
        if name == "drifting_noise_floor":
            level += math.sin(block_index / 2.0) * 8.0
        if name == "scene_level_steps":
            level += 6.0 if block_index >= 6 else 0.0
        peak = int(32768 * (10 ** (level / 20.0)) * math.sqrt(3))
        peak = max(1, min(32000, peak))
        return [int(v * peak / 900) for v in _noise(rng, count)]
    if name in ("music_bed", "dynamic_music"):
        level = -22.0
        if name == "dynamic_music":
            level += math.sin(block_index / 1.7) * 5.0
        amplitude = max(1, min(20000, int(32768 * (10 ** (level / 20.0)) * 0.707)))
        return _tone(count, 220, amplitude=amplitude, phase=phase)
    return [0] * count


def _overlay(
    name: str, count: int, rng: random.Random, bed_db: float, block_index: int, phase: float
) -> list[int] | None:
    """Extra signal layered on top of the active block, if the scenario has one."""
    if name in ("clean", "uneven_gain", "rapid_dialogue", "long_silence", "compressed_dynamics",
                "dialogue_envelope"):
        return None
    if name in ("ambience", "crowd", "drifting_noise_floor", "scene_level_steps",
                "quiet_scene_tone", "effects_over_ambience"):
        level = bed_db
        if name == "drifting_noise_floor":
            level += math.sin(block_index / 2.0) * 8.0
        if name == "scene_level_steps":
            level += 6.0 if block_index >= 6 else 0.0
        peak = max(1, min(32000, int(32768 * (10 ** (level / 20.0)) * math.sqrt(3))))
        return [int(v * peak / 900 * 0.4) for v in _noise(rng, count)]
    if name in ("music_bed", "dynamic_music"):
        level = -22.0
        if name == "dynamic_music":
            level += math.sin(block_index / 1.7) * 5.0
        amplitude = max(1, min(20000, int(32768 * (10 ** (level / 20.0)) * 0.707)))
        return _tone(count, 220, amplitude=amplitude, phase=phase + 0.7)
    if name == "overlapping_activity":
        return [int(v * 0.5 + 3000 * math.sin(2 * math.pi * 700 * i / RATE))
                for i, v in enumerate(_tone(count, 430, amplitude=5000, phase=phase))]
    if name == "dialogue_envelope":
        # An amplitude envelope that rises and falls within the block, so the
        # level is not constant the way a pure tone is.
        out: list[int] = []
        for i in range(count):
            envelope = 0.35 + 0.65 * abs(math.sin(2 * math.pi * (i / max(1, count)) * 3.0))
            out.append(int(envelope * v))
        return out
    if name == "sudden_gain_change":
        return None
    return None


def _expected_silences(segments: list[tuple[str, int]]) -> tuple[list[int], list[int]]:
    starts: list[int] = []
    ends: list[int] = []
    position = 0
    for kind, duration in segments:
        if kind == "silence":
            starts.append(position)
            ends.append(position + duration)
        position += duration
    return starts, ends


def analyse(name: str, path: Path, timeout: float) -> dict[str, Any]:
    from app.services.sync.video_timeline import extract_video_profile

    profile = extract_video_profile(path, timeout=timeout)
    if profile is None:
        return {"scenario": name, "status": "PROFILE_UNAVAILABLE"}
    landmarks = profile.audio_landmarks
    return {
        "scenario": name,
        "status": "ok",
        "description": SCENARIOS.get(name, ""),
        "duration_ms": profile.duration_ms,
        "audio_stream_count": profile.audio_stream_count,
        "selected_audio_stream": profile.selected_audio_stream,
        "stream_selection": profile.audio_stream_selection,
        "landmark_count": len(landmarks),
        "landmarks": landmarks,
        "extraction_ms": profile.extraction_ms,
        "landmarks_present": bool(landmarks),
    }


def score(name: str, result: dict[str, Any], segments: list[tuple[str, int]], tolerance_ms: int) -> dict[str, Any]:
    """Compare detected silence boundaries against the known structure."""
    if result.get("status") != "ok":
        return {**result, "verdict": result.get("status", "unknown")}
    starts, ends = _expected_silences(segments)
    detected = result["landmarks"]
    expected = sorted(starts + ends)

    matched_start = 0
    for value in starts:
        if any(abs(value - d) <= tolerance_ms for d in detected):
            matched_start += 1
    matched_end = 0
    for value in ends:
        if any(abs(value - d) <= tolerance_ms for d in detected):
            matched_end += 1
    recall = (matched_start + matched_end) / max(1, len(expected))
    # Precision: how many detections correspond to a real boundary.
    real = 0
    for d in detected:
        if any(abs(d - e) <= tolerance_ms for e in expected):
            real += 1
    precision = real / max(1, len(detected))
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    if not detected:
        verdict = "UNDER_SENSITIVE: structure lost entirely"
    elif recall < 0.5:
        verdict = "UNDER_SENSITIVE: most boundaries missed"
    elif precision < 0.5:
        verdict = "OVER_SENSITIVE: most detections are not real boundaries"
    else:
        verdict = "STABLE: boundaries recovered"
    return {
        **result,
        "expected_boundary_count": len(expected),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "verdict": verdict,
    }


def sensitivity(path: Path, floors: list[str], timeout: float) -> dict[str, int]:
    """How many silence boundaries each amplitude floor recovers.

    A single fixed floor cannot serve every programme: a music bed needs a much
    higher floor than room tone, and a higher floor starts inventing silences
    in genuinely quiet-but-not-silent material. Recording the whole curve is
    more useful than any one number, and it is why the floor is not tuned here.
    """
    import subprocess

    out: dict[str, int] = {}
    for floor in floors:
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
                    "-af", f"silencedetect=noise={floor}dB:d=0.700",
                    "-f", "null", "-",
                ],
                capture_output=True, text=True, timeout=timeout, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            out[floor] = -1
            continue
        # Count starts only: each silence emits silence_start, silence_end and
        # silence_duration, so a loose substring count triples the result.
        out[floor] = (result.stderr or "").count("silence_start")
    return out


def analyse_adaptive(path: Path, timeout: float) -> dict[str, Any]:
    """Run the per-file adaptive detector and report what it derived."""
    from app.services.sync.audio_activity import (
        AdaptiveAudioConfig,
        detect_activity_adaptive,
    )

    profile = detect_activity_adaptive(
        path,
        stream_index=0,
        config=AdaptiveAudioConfig(),
        timeout=timeout,
    )
    return {
        "detector": "adaptive",
        "status": "ok",
        "description": SCENARIOS.get(path.stem, ""),
        "duration_ms": profile.duration_ms,
        "landmark_count": len(profile.landmarks),
        "landmarks": profile.landmarks,
        "baseline_db": profile.baseline_db,
        "threshold_db": profile.enter_threshold_db,
        "exit_threshold_db": profile.exit_threshold_db,
        "split_gap_db": profile.split_gap_db,
        "activity_ratio": profile.activity_ratio,
        "segment_count": profile.segment_count,
        "quality": profile.quality.value,
        "stability_score": profile.stability_score,
        "extraction_ms": profile.extraction_ms,
        "reasons": profile.reasons[:4],
    }


def compare(name: str, fixed: dict[str, Any], adaptive: dict[str, Any], segments) -> dict[str, Any]:
    """Fixed vs adaptive for one scenario, against the authored ground truth."""
    starts, ends = _expected_silences(segments)
    expected = sorted(starts + ends)

    def _score(result: dict[str, Any]) -> dict[str, Any]:
        if result.get("status") != "ok":
            return {"landmarks": 0, "recall": 0.0, "precision": 0.0, "f1": 0.0}
        detected = result["landmarks"]
        matched = sum(
            1 for e in expected if any(abs(e - d) <= 700 for d in detected)
        )
        recall = matched / max(1, len(expected))
        real = sum(
            1 for d in detected if any(abs(d - e) <= 700 for e in expected)
        )
        precision = real / max(1, len(detected))
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return {
            "landmarks": len(detected),
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
        }

    fixed_score = _score(fixed)
    adaptive_score = _score(adaptive)
    difference = round(adaptive_score["f1"] - fixed_score["f1"], 4)
    if difference > 0.05:
        decision = "adaptive recovers structure the fixed floor cannot"
    elif difference < -0.05:
        decision = "REGRESSION: the adaptive detector scores lower here"
    elif fixed_score["f1"] == 0.0 and adaptive_score["f1"] == 0.0:
        decision = "both abstain: no defensible structure in this clip"
    else:
        decision = "comparable"
    return {
        "scenario": name,
        "expected_boundaries": len(expected),
        "fixed": fixed_score,
        "adaptive": adaptive_score,
        "f1_difference": difference,
        "adaptive_quality": adaptive.get("quality"),
        "adaptive_baseline_db": adaptive.get("baseline_db"),
        "adaptive_threshold_db": adaptive.get("threshold_db"),
        "adaptive_split_gap_db": adaptive.get("split_gap_db"),
        "adaptive_extraction_ms": adaptive.get("extraction_ms"),
        "fixed_extraction_ms": fixed.get("extraction_ms"),
        "decision": decision,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stress the audio landmark extractor with hostile audio"
    )
    parser.add_argument("--workdir", type=Path, default=Path("subs_cache/stress_audio"))
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--scenario", action="append", help="only these scenarios")
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="also run the per-file adaptive detector",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="fixed vs adaptive, per scenario (implies --adaptive)",
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="also record the amplitude-floor sensitivity curve (measurement only)",
    )
    args = parser.parse_args(argv)

    args.workdir.mkdir(parents=True, exist_ok=True)
    names = args.scenario or list(SCENARIOS)
    results: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for name in names:
        path = args.workdir / f"{name}.wav"
        segments = render_scenario(name, path)
        result = analyse(name, path, args.timeout)
        row = score(name, result, segments, tolerance_ms=700)
        if args.sensitivity:
            row["floor_sensitivity"] = sensitivity(
                path, ["-45", "-35", "-30", "-25", "-20"], args.timeout
            )
        results.append(row)
        if args.compare or args.adaptive:
            adaptive = analyse_adaptive(path, args.timeout)
            if args.compare:
                comparisons.append(compare(name, result, adaptive, segments))
            else:
                row["adaptive"] = {
                    k: adaptive[k]
                    for k in (
                        "landmark_count", "baseline_db", "threshold_db", "quality",
                        "stability_score", "activity_ratio",
                    )
                }

    payload = {
        "note": (
            "PROXY DATA, not real programme audio. Generated clips that reproduce the "
            "conditions real content creates (ambience, music beds, crowd noise, "
            "transients, uneven loudness, rapid dialogue, long silences) so the "
            "extractor's assumptions can be stress-tested. No conclusion about real "
            "programmes may be drawn from this file."
        ),
        "silence_detector": "noise=-45dB:d=0.700",
        "sensitivity_recorded": bool(args.sensitivity),
        "finding": (
            "The -45dB floor in use recovers boundaries only from audio containing "
            "true digital silence. Room tone, crowd noise and a music bed all yield "
            "ZERO landmarks at that floor, and the validator then correctly abstains. "
            "The structure is recoverable at a higher floor, but no single floor "
            "serves all content: a music bed needs a markedly higher one, and a "
            "higher floor begins inventing silences in quiet-but-not-silent "
            "material. This is a measurement. No threshold was changed."
        ),
        "rate_hz": RATE,
        "scenarios": results,
        "comparisons": comparisons,
        "adaptive_config": None,
    }
    if args.compare or args.adaptive:
        from app.services.sync.audio_activity import AdaptiveAudioConfig as _AdaptiveCfg

        payload["adaptive_config"] = _AdaptiveCfg().model_dump()
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
        print(f"written to {args.json_out}", file=sys.stderr)

    print("=" * 74)
    print("AUDIO LANDMARK EXTRACTOR STRESS (PROXY DATA, NOT REAL PROGRAMMES)")
    print("=" * 74)
    for row in results:
        if row.get("status") != "ok":
            print(f"  {row['scenario']:16s} {row.get('status')}")
            continue
        print(
            f"  {row['scenario']:16s} landmarks={row['landmark_count']:>3} "
            f"expected={row['expected_boundary_count']:>3} "
            f"P={row['precision']:.2f} R={row['recall']:.2f} F1={row['f1']:.2f}  "
            f"{row['verdict']}"
        )
    if comparisons:
        print()
        print("=" * 92)
        print("FIXED vs ADAPTIVE")
        print("=" * 92)
        print(
            f"{'Scenario':<22}{'Fixed P/R':>13}{'Adaptive P/R':>15}"
            f"{'dF1':>8}  {'Quality':<12}Decision"
        )
        print("-" * 92)
        for row in comparisons:
            fixed_p = row["fixed"]
            adaptive_p = row["adaptive"]
            print(
                f"{row['scenario']:<22}"
                f"{fixed_p['precision']:.2f}/{fixed_p['recall']:.2f}".ljust(35)
                .rjust(35)
                + f"{adaptive_p['precision']:.2f}/{adaptive_p['recall']:.2f}".rjust(15)
                + f"{row['f1_difference']:>+8.2f}"
                + f"  {str(row['adaptive_quality']):<12}{row['decision']}"
            )
        print()
        print("  dF1 is a measured difference, reported as a difference and not as an")
        print("  improvement claim. Scenarios are held out: the adaptive parameters were")
        print("  not chosen to maximise this table.")
    print()
    print("  This measures whether the extractor survives the conditions real audio")
    print("  creates. It does not measure real audio, and cannot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
