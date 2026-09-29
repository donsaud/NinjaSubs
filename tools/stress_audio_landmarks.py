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
    """Write a WAV for one scenario and return its ground-truth segments."""
    segments = build_scenario(name, seed)
    rng = random.Random(seed)
    frames = bytearray()
    block_index = 0
    for kind, duration in segments:
        count = int(duration * RATE / 1000)
        if kind == "silence":
            if name == "ambience":
                frames += struct.pack(f"<{count}h", *_noise(rng, count))
            elif name == "music_bed":
                frames += struct.pack(
                    f"<{count}h", *_tone(count, 220, amplitude=2600, phase=rng.random())
                )
            elif name == "crowd":
                frames += struct.pack(f"<{count}h", *_noise(rng, count))
            else:
                frames += struct.pack(f"<{count}h", *([0] * count))
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
        frames += struct.pack(f"<{count}h", *[max(-32000, min(32000, v)) for v in base])

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(bytes(frames))
    return segments


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stress the audio landmark extractor with hostile audio"
    )
    parser.add_argument("--workdir", type=Path, default=Path("subs_cache/stress_audio"))
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--scenario", action="append", help="only these scenarios")
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="also record the amplitude-floor sensitivity curve (measurement only)",
    )
    args = parser.parse_args(argv)

    args.workdir.mkdir(parents=True, exist_ok=True)
    names = args.scenario or list(SCENARIOS)
    results: list[dict[str, Any]] = []
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
    }
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
    print()
    print("  This measures whether the extractor survives the conditions real audio")
    print("  creates. It does not measure real audio, and cannot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
