"""Per-file adaptive audio activity detection.

This is an **activity** detector, not a speech recogniser and not VAD. It never
inspects words, phonemes, speakers, language or meaning. It measures short-time
energy and asks one question about each frame: is this file loud or quiet
*relative to its own distribution*?

The fixed `silencedetect` floor it replaces worked only on audio containing true
digital silence, and no global constant can serve both a music bed and a
quiet-but-not-silent scene. See docs/audio_landmark_pipeline.md for the audit
that established this and for why re-tuning was not an option.

The shape is deliberately conservative:

    decoded mono samples
      -> short-time energy in dB
      -> robust per-file baseline
      -> bounded adaptive threshold
      -> smoothing + hysteresis
      -> minimum activity / silence durations
      -> boundary separation and merging
      -> landmarks, with a quality verdict

Every intermediate is retained on the result, so a reader can see the
measurements rather than a single opaque number. Pathological output is
detected and labelled, and a detector that cannot establish structure says so
instead of inventing landmarks.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

#: Bumped whenever extraction or comparison changes in a way that invalidates a
#: cached profile. Adaptive extraction is a different algorithm, so a profile
#: produced by it must never be served to a caller expecting the fixed one.
ADAPTIVE_PROFILE_VERSION = 2

#: Decode ceiling. A long film is sampled rather than fully materialised, and
#: the energy envelope of a programme does not need full resolution.
MAX_DECODE_SECONDS = 7_200.0
DECODE_RATE = 8_000
DEFAULT_TIMEOUT = 180.0


class LandmarkQuality(str, Enum):
    """Whether the detector's output can be trusted as evidence."""

    STRONG = "strong"
    USABLE = "usable"
    WEAK = "weak"
    INSUFFICIENT = "insufficient"
    INVALID = "invalid"


class AdaptiveAudioConfig(BaseModel):
    """Extraction parameters. Not synchronization thresholds."""

    window_ms: int = 50
    hop_ms: int = 25
    smoothing_ms: int = 250
    enter_margin_db: float = 6.0
    exit_margin_db: float = 3.0
    min_threshold_db: float = -60.0
    max_threshold_db: float = 0.0
    min_silence_ms: int = 700
    min_activity_ms: int = 400
    min_boundary_separation_ms: int = 1_500
    baseline_percentile: float = 20.0
    min_landmarks: int = 4
    max_landmarks: int = 400

    @classmethod
    def from_settings(cls, settings: object) -> AdaptiveAudioConfig:
        """Read the project's settings model, tolerating absence."""

        def _get(name: str, fallback):
            value = getattr(settings, name, fallback)
            return fallback if value is None else value

        return cls(
            window_ms=int(_get("ADAPTIVE_AUDIO_WINDOW_MS", 50)),
            hop_ms=int(_get("ADAPTIVE_AUDIO_HOP_MS", 25)),
            smoothing_ms=int(_get("ADAPTIVE_AUDIO_SMOOTHING_MS", 250)),
            enter_margin_db=float(_get("ADAPTIVE_AUDIO_ENTER_MARGIN_DB", 6.0)),
            exit_margin_db=float(_get("ADAPTIVE_AUDIO_EXIT_MARGIN_DB", 3.0)),
            min_threshold_db=float(_get("ADAPTIVE_AUDIO_MIN_THRESHOLD_DB", -60.0)),
            max_threshold_db=float(_get("ADAPTIVE_AUDIO_MAX_THRESHOLD_DB", -18.0)),
            min_silence_ms=int(_get("ADAPTIVE_AUDIO_MIN_SILENCE_MS", 700)),
            min_activity_ms=int(_get("ADAPTIVE_AUDIO_MIN_ACTIVITY_MS", 400)),
            min_boundary_separation_ms=int(
                _get("ADAPTIVE_AUDIO_MIN_BOUNDARY_SEPARATION_MS", 1_500)
            ),
            baseline_percentile=float(_get("ADAPTIVE_AUDIO_BASELINE_PERCENTILE", 20.0)),
            min_landmarks=int(_get("ADAPTIVE_AUDIO_MIN_LANDMARKS", 4)),
            max_landmarks=int(_get("ADAPTIVE_AUDIO_MAX_LANDMARKS", 400)),
        )


class AdaptiveAudioProfile(BaseModel):
    """Landmarks plus every measurement that produced them."""

    landmarks: list[int] = Field(default_factory=list)
    #: Activity intervals, which is the detector's real output; landmarks are
    #: derived from their edges.
    activity_intervals: list[tuple[int, int]] = Field(default_factory=list)
    duration_ms: int = 0
    baseline_db: float | None = None
    #: How decisively the energy distribution split into floor and bulk. A
    #: small value means the file has no clean quiet floor.
    split_gap_db: float | None = None
    threshold_db: float | None = None
    enter_threshold_db: float | None = None
    exit_threshold_db: float | None = None
    activity_ratio: float | None = None
    segment_count: int = 0
    frame_count: int = 0
    quality: LandmarkQuality = LandmarkQuality.INSUFFICIENT
    #: 0..1, how cleanly the output separates from a pathological shape.
    stability_score: float | None = None
    reasons: list[str] = Field(default_factory=list)
    profile_version: int = ADAPTIVE_PROFILE_VERSION
    extraction_ms: float | None = None

    @property
    def usable(self) -> bool:
        return self.quality in (LandmarkQuality.STRONG, LandmarkQuality.USABLE)


def _decode_mono(path: Path, stream_index: int, timeout: float) -> tuple[list[float], int]:
    """Decode to mono float samples, bounded in time.

    Decoding happens once per file. Everything downstream works on the energy
    envelope, so per-candidate comparison never re-decodes.
    """
    if not shutil.which("ffmpeg"):
        return [], 0
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "error",
        "-t",
        f"{MAX_DECODE_SECONDS:.0f}",
        "-i",
        str(path),
        "-map",
        f"0:{stream_index}",
        "-ac",
        "1",
        "-ar",
        str(DECODE_RATE),
        "-f",
        "f32le",
        "-",
    ]
    with tempfile.TemporaryDirectory(prefix="adaptive_audio_") as workdir:
        raw_path = Path(workdir) / "mono.f32"
        # Replace only the trailing "-" output target with a real file, so the
        # format and codec flags survive.
        argv = [*command[:-1], str(raw_path)]
        try:
            result = subprocess.run(  # noqa: S603 - fixed argv, no shell
                argv,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return [], 0
        if result.returncode != 0 or not raw_path.is_file():
            return [], 0
        data = raw_path.read_bytes()

    count = len(data) // 4
    if count == 0:
        return [], 0
    import struct

    samples = struct.unpack(f"<{count}f", data[: count * 4])
    return list(samples), count


def _energy_db(samples: list[float], window: int, hop: int) -> list[tuple[int, float]]:
    """Short-time RMS in dBFS, with the frame's start time.

    An RMS window over dB is the same quantity `silencedetect` compares
    against its floor, so the two detectors are measuring the same thing and
    differ only in how the threshold is chosen.
    """
    frames: list[tuple[int, float]] = []
    if window <= 0 or hop <= 0 or not samples:
        return frames
    index = 0
    while index + window <= len(samples):
        chunk = samples[index : index + window]
        mean_square = sum(value * value for value in chunk) / window
        rms = math.sqrt(mean_square)
        db = 20.0 * math.log10(rms) if rms > 1e-9 else -120.0
        frames.append((int(index * 1000 / DECODE_RATE), round(db, 3)))
        index += hop
    return frames


def _percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolated percentile of a 0..1 fraction. Deterministic."""
    if not values:
        return -120.0
    fraction = min(1.0, max(0.0, fraction))
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _median_absolute_deviation(values: list[float], centre: float) -> float:
    if not values:
        return 0.0
    return _percentile([abs(value - centre) for value in values], 0.5)


def _lower_mode_baseline(
    energy_db: list[float], *, search_fraction: float = 0.6
) -> tuple[float, float, float]:
    """Split the energy distribution into a quiet floor and a bulk.

    Returns ``(baseline, bulk_floor, gap_db)``: the level below the widest gap
    in the lower half of the sorted distribution, the level above that gap, and
    the width of the gap.

    A fixed low percentile was tried first and measured, and it is the wrong
    shape of estimator here. The quiet fraction of a file is not stable: a clip
    that is 84% active puts its 20th percentile inside the *active* level, so
    the derived threshold rises above the audio and nothing is ever detected,
    while a clip that is 50% silent puts the same percentile deep in the
    silence. The floor has to be found, not sampled at a fixed rank.

    Splitting at the widest gap is the simplest method indifferent to that
    ratio, and it is the classic one-cluster/one-cluster split of a 1-D
    distribution. Returning both edges also lets the threshold be constrained
    to sit *inside* the gap rather than wherever the margin happens to land.
    """
    if not energy_db:
        return -120.0, -119.0, 0.0
    ordered = sorted(energy_db)
    count = len(ordered)
    if count < 4:
        return ordered[0], ordered[min(1, count - 1)], 0.0
    limit = max(2, int(count * search_fraction))
    window = max(4, count // 200)
    best_gap = -1.0
    best_index = 0
    for index in range(0, max(1, limit - window)):
        gap = ordered[index + window] - ordered[index]
        if gap > best_gap:
            best_gap = gap
            best_index = index
    return (
        ordered[max(0, best_index)],
        ordered[min(count - 1, best_index + window)],
        max(0.0, best_gap),
    )


def derive_threshold(
    energy_db: list[float], config: AdaptiveAudioConfig
) -> tuple[float, float, float, list[str]]:
    """Baseline, enter threshold and exit threshold, all bounded.

    The baseline is this file's own quiet floor rather than its quietest
    moment, because §6: the lowest energy in a file is not necessarily
    silence. A room tone or a quiet scene sits near the bottom of the
    distribution without being silence.

    The threshold is then constrained to sit inside the gap between that floor
    and the bulk. That constraint is what makes the method work across
    programmes rather than only those with a wide dynamic range: a music bed
    with 12dB between bed and dialogue cannot be given a threshold that sits
    above its dialogue, whatever the margin says.
    """
    reasons: list[str] = []
    if not energy_db:
        return (-60.0, -60.0, -60.0, ["no audio energy measured"])

    baseline, bulk_floor, gap = _lower_mode_baseline(energy_db)
    loud_reference = _percentile(energy_db, 0.90)
    spread = _median_absolute_deviation(energy_db, baseline)
    dynamic = max(0.0, loud_reference - baseline)

    # A wide gap is a decisive split, so a small margin suffices. A narrow one
    # means the file is nearly uniform, so the margin grows rather than the
    # threshold landing in the middle of the noise.
    margin = 2.0 if gap >= 12.0 else 6.0
    margin += min(4.0, spread * 0.25)
    if dynamic < 3.0:
        reasons.append("little dynamic range; margin widened to avoid splitting the noise")

    enter = baseline + config.enter_margin_db + margin
    exit_ = baseline + config.exit_margin_db + margin * 0.5

    # Keep both thresholds strictly inside the gap.
    if gap >= 3.0:
        ceiling = bulk_floor - 1.0
        floor_for_enter = baseline + 1.0
        if enter > ceiling:
            enter = ceiling
            reasons.append(
                f"enter clamped into the {gap:.1f}dB gap between floor and bulk"
            )
        if enter < floor_for_enter:
            enter = floor_for_enter
        if exit_ > ceiling:
            exit_ = ceiling
        if exit_ < baseline + 0.5:
            exit_ = baseline + 0.5

    if enter > config.max_threshold_db:
        enter = config.max_threshold_db
        reasons.append("enter threshold clamped down to the maximum")
    if exit_ > config.max_threshold_db:
        exit_ = config.max_threshold_db
        reasons.append("exit threshold clamped down to the maximum")
    if exit_ < config.min_threshold_db:
        exit_ = config.min_threshold_db
        reasons.append("exit threshold clamped up to the minimum")
    if enter < config.min_threshold_db:
        enter = config.min_threshold_db
    if enter < exit_:
        enter = exit_
        reasons.append("enter raised to the exit threshold to keep hysteresis ordered")
    return (
        round(baseline, 3),
        round(enter, 3),
        round(exit_, 3),
        reasons,
    )


def _smooth(values: list[float], window_frames: int) -> list[float]:
    """Median smoothing. Cheap, and it removes single-frame spikes without the
    lag a moving average would introduce."""
    if window_frames <= 1 or not values:
        return list(values)
    half = window_frames // 2
    out: list[float] = []
    for index in range(len(values)):
        start = max(0, index - half)
        stop = min(len(values), index + half + 1)
        chunk = sorted(values[start:stop])
        out.append(chunk[len(chunk) // 2])
    return out


def _classify_activity(
    smoothed: list[float], enter_db: float, exit_db: float
) -> list[bool]:
    """Hysteresis: crossing upward needs `enter_db`, falling back needs `exit_db`.

    This is what stops a level sitting exactly on the threshold from chattering
    into dozens of spurious boundaries.
    """
    state = False
    out: list[bool] = []
    for index, value in enumerate(smoothed):
        if not state and value >= enter_db:
            state = True
        elif state and value < exit_db:
            state = False
        out.append(state)
    return out


def _intervals_from_mask(
    frames: list[tuple[int, float]], mask: list[bool], duration_ms: int
) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(mask):
        time_ms = frames[index][0]
        if active and start is None:
            start = time_ms
        elif not active and start is not None:
            intervals.append((start, time_ms))
            start = None
    if start is not None:
        intervals.append((start, duration_ms))
    return intervals


def _apply_minimums(
    intervals: list[tuple[int, int]],
    silence_intervals: list[tuple[int, int]],
    config: AdaptiveAudioConfig,
) -> tuple[list[tuple[int, int]], list[str]]:
    """Drop activity too short to be real, and silence too short to be a gap.

    A 30ms dip should not become a boundary, and a 50ms pause inside a sentence
    should not split an activity segment.
    """
    reasons: list[str] = []
    kept = [
        (start, end)
        for start, end in intervals
        if (end - start) >= config.min_activity_ms
    ]
    dropped_activity = len(intervals) - len(kept)
    if dropped_activity:
        reasons.append(f"dropped {dropped_activity} activity segments below the minimum")
    kept_silence = [
        (start, end)
        for start, end in silence_intervals
        if (end - start) >= config.min_silence_ms
    ]
    dropped_silence = len(silence_intervals) - len(kept_silence)
    if dropped_silence:
        reasons.append(f"dropped {dropped_silence} silence segments below the minimum")
    return kept + kept_silence, reasons


def _merge_close(intervals: list[tuple[int, int]], separation_ms: int) -> list[tuple[int, int]]:
    """Merge intervals whose gap is too small to be a meaningful boundary."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start - last_end < separation_ms:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _quality(
    profile: AdaptiveAudioProfile, config: AdaptiveAudioConfig
) -> LandmarkQuality:
    """Classify the output, including the pathological shapes.

    A detector that produced nonsense must be labelled invalid rather than
    allowed to become evidence for a cut mismatch.
    """
    landmarks = profile.landmarks
    duration = max(1, profile.duration_ms)
    if not landmarks:
        profile.reasons.append("no landmarks: the file has no temporal structure to compare")
        return LandmarkQuality.INSUFFICIENT
    if len(landmarks) == 1:
        profile.reasons.append("a single landmark cannot support a comparison")
        return LandmarkQuality.INSUFFICIENT
    if len(landmarks) > config.max_landmarks:
        profile.reasons.append(
            f"{len(landmarks)} landmarks exceeds the pathological-density limit"
        )
        return LandmarkQuality.INVALID
    if len(landmarks) < config.min_landmarks:
        profile.reasons.append(
            f"only {len(landmarks)} landmarks, below the minimum of {config.min_landmarks}"
        )
        return LandmarkQuality.WEAK

    # Repeated micro-boundaries inside one span indicate chatter.
    gaps = [b - a for a, b in zip(landmarks, landmarks[1:], strict=False)]
    chatter = sum(1 for gap in gaps if gap < config.min_boundary_separation_ms)
    chatter_ratio = chatter / max(1, len(gaps))
    # One enormous gap means the whole file looked like a single event.
    largest = max(gaps) if gaps else 0
    dominant_ratio = largest / duration

    score = 1.0
    score -= 0.5 * chatter_ratio
    if dominant_ratio > 0.85:
        score -= 0.4
        profile.reasons.append("one dominant interval covers most of the timeline")
    if profile.activity_ratio is not None:
        ratio = profile.activity_ratio
        if ratio < 0.05 or ratio > 0.98:
            score -= 0.3
            profile.reasons.append(f"activity ratio {ratio:.2f} is not a plausible programme")
    profile.stability_score = round(max(0.0, min(1.0, score)), 4)
    profile.reasons.append(f"boundary chatter ratio {chatter_ratio:.2f}")

    if score >= 0.8:
        return LandmarkQuality.STRONG
    if score >= 0.5:
        return LandmarkQuality.USABLE
    return LandmarkQuality.WEAK


def detect_activity_adaptive(
    path: str | Path,
    *,
    stream_index: int,
    config: AdaptiveAudioConfig | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    duration_ms: int | None = None,
) -> AdaptiveAudioProfile:
    """Detect activity boundaries relative to this file's own distribution.

    Never raises for bad media. A file it cannot read comes back as
    ``INSUFFICIENT`` with a reason, because "I could not tell" must stay
    distinguishable from "it does not match".
    """
    import time

    started = time.monotonic()
    profile = AdaptiveAudioProfile()
    if config is None:
        from app.config import settings

        config = AdaptiveAudioConfig.from_settings(settings)

    samples, count = _decode_mono(Path(path), stream_index, timeout)
    if not samples:
        profile.quality = LandmarkQuality.INSUFFICIENT
        profile.reasons.append("audio could not be decoded; abstaining")
        profile.extraction_ms = round((time.monotonic() - started) * 1000, 3)
        return profile

    total_ms = int(count * 1000 / DECODE_RATE)
    profile.duration_ms = duration_ms or total_ms
    profile.frame_count = 0

    frames = _energy_db(samples, int(config.window_ms * DECODE_RATE / 1000), int(config.hop_ms * DECODE_RATE / 1000))
    if not frames:
        profile.quality = LandmarkQuality.INSUFFICIENT
        profile.reasons.append("audio too short to frame")
        profile.extraction_ms = round((time.monotonic() - started) * 1000, 3)
        return profile
    profile.frame_count = len(frames)

    energy = [value for _time, value in frames]
    baseline, enter_db, exit_db, reasons = derive_threshold(energy, config)
    profile.baseline_db = baseline
    profile.split_gap_db = round(_lower_mode_baseline(energy)[2], 3)
    profile.enter_threshold_db = enter_db
    profile.exit_threshold_db = exit_db
    profile.threshold_db = enter_db
    profile.reasons.extend(reasons)

    smoothing_frames = max(1, int(config.smoothing_ms / max(1, config.hop_ms)))
    smoothed = _smooth(energy, smoothing_frames)
    mask = _classify_activity(smoothed, enter_db, exit_db)

    activity = _intervals_from_mask(frames, mask, profile.duration_ms)
    silence = _invert_intervals(activity, profile.duration_ms)
    combined, more = _apply_minimums(activity, silence, config)
    profile.reasons.extend(more)
    activity, silence = _split_by_kind(combined, _frame_times(frames), mask)

    merged = _merge_close(activity, config.min_boundary_separation_ms)
    profile.activity_intervals = merged
    profile.segment_count = len(merged)
    active_ms = sum(end - start for start, end in merged)
    profile.activity_ratio = round(active_ms / max(1, profile.duration_ms), 4)

    edges: list[int] = []
    for start, end in merged:
        edges.append(start)
        edges.append(end)
    profile.landmarks = sorted(set(edges))

    profile.quality = _quality(profile, config)
    profile.extraction_ms = round((time.monotonic() - started) * 1000, 3)
    return profile


def _invert_intervals(
    activity: list[tuple[int, int]], duration_ms: int
) -> list[tuple[int, int]]:
    silence: list[tuple[int, int]] = []
    cursor = 0
    for start, end in sorted(activity):
        if start > cursor:
            silence.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration_ms:
        silence.append((cursor, duration_ms))
    return silence


def _split_by_kind(
    combined: list[tuple[int, int]],
    frame_times: list[int],
    mask: list[bool],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Separate surviving activity intervals from surviving silence ones.

    ``combined`` mixes both kinds, so membership is decided by asking whether
    the interval's midpoint was classified active in the original mask.
    """
    kept_activity: list[tuple[int, int]] = []
    kept_silence: list[tuple[int, int]] = []
    for start, end in combined:
        midpoint = (start + end) // 2
        index = _frame_index_at(frame_times, midpoint)
        active = bool(mask[index]) if index is not None else True
        (kept_activity if active else kept_silence).append((start, end))
    return kept_activity, kept_silence


def _frame_times(frames: list[tuple[int, float]]) -> list[int]:
    """Frame start times only, which is all the time lookup needs."""
    return [frame[0] for frame in frames]


def _frame_index_at(frame_times: list[int], time_ms: int) -> int | None:
    if not frame_times:
        return None
    low, high = 0, len(frame_times) - 1
    while low <= high:
        mid = (low + high) // 2
        if frame_times[mid] < time_ms:
            low = mid + 1
        else:
            high = mid - 1
    return min(max(0, high), len(frame_times) - 1)
