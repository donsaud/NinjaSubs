"""Video-derived timeline validation: an evidence source outside the subtitle loop.

Every check built so far compares subtitle artifacts to each other. That loop
is closed, and a closed loop cannot detect that both of its members are wrong
in the same way. The golden benchmark's circular case is exactly that: a
candidate and a reference sharing one incorrect timing model produce a perfect
residual, total provider consensus, and a clean verification.

This module introduces the first independent witness: the target video's own
timeline. It never inspects subtitle text, and it never asserts anything it
has not measured.

Two design rules are load-bearing:

* **Each signal stays separate.** There is no combined ``video_score``. A
  mismatch in one signal and a match in another must remain visible, because
  which signal fired is the diagnosis.
* **Absence of evidence is not evidence.** A missing video, a missing binary,
  or too few landmarks yields an abstain. Abstaining never withholds a
  verification and never creates one.

See docs/video_timeline_capabilities.md for the investigation that chose these
signals, including why keyframes were rejected and why this cannot run on the
production request path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import tempfile
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

#: Bumped when extraction or comparison changes in a way that invalidates a
#: cached profile. Cached profiles from another version are never reused.
VIDEO_PROFILE_VERSION = 1

#: Silence shorter than this is not a landmark; it is a breath.
MIN_SILENCE_MS = 700
#: A scene boundary is only counted if it is this far from the last one, which
#: keeps a fast-cut sequence from producing dozens of near-identical marks.
MIN_SCENE_GAP_MS = 5_000


class VideoVerdict(str, Enum):
    """What the video timeline says about a subtitle's timing model."""

    #: Landmarks correspond across the whole timeline.
    MATCH = "match"
    #: Landmarks actively disagree; this is negative evidence.
    MISMATCH = "mismatch"
    #: A profile exists but carries too few landmarks to compare.
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    #: No video, or no tooling to read it. Never negative evidence.
    UNAVAILABLE = "unavailable"


class VideoFailureReason(str, Enum):
    """Machine-readable reasons. Absence and disagreement are kept apart."""

    NO_VIDEO = "VIDEO_PROFILE_UNAVAILABLE"
    EXTRACTION_FAILED = "VIDEO_PROFILE_EXTRACTION_FAILED"
    NO_AUDIO_STREAM = "VIDEO_AUDIO_STREAM_MISSING"
    AUDIO_STREAM_AMBIGUOUS = "AUDIO_STREAM_AMBIGUOUS"
    TOO_FEW_LANDMARKS = "VIDEO_EVIDENCE_INSUFFICIENT"
    DURATION_MISMATCH = "VIDEO_DURATION_MISMATCH"
    AUDIO_ACTIVITY_MISMATCH = "AUDIO_ACTIVITY_MISMATCH"
    SCENE_STRUCTURE_MISMATCH = "SCENE_STRUCTURE_MISMATCH"
    NONE = "VIDEO_EVIDENCE_NONE"


class AudioStreamChoice(BaseModel):
    """Which audio stream the timeline is read from, and why.

    Real files carry commentary, description, alternate-language and SDH
    tracks. Reading the wrong one produces landmarks for the wrong content, so
    the choice is made from container metadata only, recorded explicitly, and
    refused when it cannot be established.
    """

    index: int | None = None
    reason: str = "unknown"
    #: Disposition and tag signals that drove the decision, for diagnosis.
    signals: list[str] = Field(default_factory=list)
    ambiguous: bool = False


#: Disposition/tags that identify a track as NOT the programme audio.
_NON_PROGRAMME_HINTS = (
    "commentary",
    "description",
    "sdh",
    "hearing impaired",
    "audio description",
    "narration",
    "alternate",
)


def select_audio_stream(streams: list[dict]) -> AudioStreamChoice:
    """Pick the programme audio deterministically, or refuse to guess.

    Uses only container metadata. No speech recognition, and no assumption that
    the first track is the right one. When the choice cannot be established
    confidently the result is ambiguous, and the caller abstains.
    """
    audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if not audio:
        return AudioStreamChoice(reason="no_audio_stream", ambiguous=True)
    if len(audio) == 1:
        return AudioStreamChoice(index=0, reason="single_audio_stream")

    signals: list[str] = []
    # Container metadata first: the default disposition is the main track.
    for position, stream in enumerate(audio):
        disposition = stream.get("disposition") or {}
        if disposition.get("default"):
            signals.append(f"stream {position} has default disposition")
            return AudioStreamChoice(
                index=position, reason="default_disposition", signals=signals
            )

    # Otherwise exclude anything tagged as not-programme audio.
    programme: list[int] = []
    for position, stream in enumerate(audio):
        tags = {
            str(value).lower()
            for value in (stream.get("tags") or {}).values()
        }
        label = tags | {
            str(value).lower() for value in (stream.get("tags") or {}).keys()
        }
        if any(hint in tag for hint in _NON_PROGRAMME_HINTS for tag in label):
            signals.append(f"stream {position} excluded: tagged {sorted(label)[:3]}")
            continue
        programme.append(position)

    if len(programme) == 1:
        return AudioStreamChoice(
            index=programme[0],
            reason="only_untagged_programme_stream",
            signals=signals,
        )
    signals.append(
        f"{len(programme)} plausible programme streams and no default disposition"
    )
    # Refusing is the correct outcome: an arbitrary pick would silently compare
    # against the wrong audio.
    return AudioStreamChoice(
        reason="ambiguous", ambiguous=True, signals=signals
    )


class VideoTimelineProfile(BaseModel):
    """A deterministic, text-free summary of one video file's timeline.

    Contains landmarks and counts only. No frames, no samples, no media data.
    """

    duration_ms: int | None = None
    audio_stream_count: int | None = None
    video_stream_count: int | None = None
    #: Which audio stream the landmarks were read from, and whether that
    #: choice was confident. An ambiguous choice yields no landmarks at all.
    selected_audio_stream: int | None = None
    audio_stream_selection: str = "unknown"
    audio_stream_ambiguous: bool = False
    #: Container start time, in ms. Landmarks are normalised to be relative to
    #: the programme start rather than the container start, because a file with
    #: intro padding or a non-zero start_time would otherwise shift every
    #: landmark and be mistaken for a different cut.
    audio_start_time_ms: int = 0
    #: Onset/offset times of coarse audio activity, in ms, programme-relative.
    audio_landmarks: list[int] = Field(default_factory=list)
    #: Coarse scene-change times, in ms, programme-relative.
    video_landmarks: list[int] = Field(default_factory=list)
    profile_version: int = VIDEO_PROFILE_VERSION
    #: Wall-clock cost of extraction, for the performance report.
    extraction_ms: int | None = None

    @property
    def digest(self) -> str:
        """Identity of the profile's content, for caching and reporting."""
        seed = json.dumps(
            {
                "v": self.profile_version,
                "d": self.duration_ms,
                "a": self.audio_stream_count,
                "v2": self.video_stream_count,
                "al": self.audio_landmarks,
                "vl": self.video_landmarks,
            },
            sort_keys=True,
        )
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


class VideoTimelineEvidence(BaseModel):
    """Per-signal results. Deliberately not collapsed into one number."""

    available: bool = False
    verdict: VideoVerdict = VideoVerdict.UNAVAILABLE
    reason: VideoFailureReason = VideoFailureReason.NO_VIDEO
    profile_version: int = VIDEO_PROFILE_VERSION
    profile_digest: str | None = None
    duration_similarity: float | None = None
    audio_activity_similarity: float | None = None
    scene_boundary_similarity: float | None = None
    # Counts, so a reader can see how thin the evidence was.
    audio_landmark_count: int = 0
    scene_landmark_count: int = 0
    reference_span_ms: int | None = None
    video_duration_ms: int | None = None
    # Raw extraction context, recorded so a reader can see what the signal was
    # actually built from rather than a single collapsed number.
    audio_stream_count: int | None = None
    selected_audio_stream: int | None = None
    audio_stream_selection: str | None = None
    audio_landmarks: list[int] = Field(default_factory=list)
    scene_landmarks: list[int] = Field(default_factory=list)
    # The global shift that best explains the subtitle's structure, and how
    # well each half of the timeline agrees with it. A real match or a real
    # global offset keeps the halves consistent; a spliced edit does not.
    best_offset_ms: int = 0
    #: Whole-timeline and per-region alignment, kept as separate measurements.
    correlation_peak: float | None = None
    correlation_second_peak: float | None = None
    correlation_peak_ratio: float | None = None
    correlation_clarity: str = "no_signal"
    region_offsets: list[int] = Field(default_factory=list)
    region_scores: list[float] = Field(default_factory=list)
    regional_consistency: str = "unknown"
    #: Which detector produced the landmarks, and what it derived.
    audio_detector: str = "fixed"
    adaptive_threshold_db: float | None = None
    adaptive_baseline_db: float | None = None
    landmark_quality: str | None = None
    first_half_similarity: float | None = None
    second_half_similarity: float | None = None
    detail: list[str] = Field(default_factory=list)


class VideoProfileCache:
    """Profile cache bound to exact video identity and profile version.

    Deliberately small and local. A profile is only reused when the video's
    identity, the profile version and the extraction parameters all match, and
    a failed extraction is cached as a failure so a broken file is not retried
    on every request.
    """

    def __init__(self, root: str | Path, *, ttl_seconds: float = 86400.0) -> None:
        self.root = Path(root)
        self.ttl_seconds = ttl_seconds
        self.root.mkdir(parents=True, exist_ok=True)

    def _key(self, video_identity: str) -> str:
        seed = f"{VIDEO_PROFILE_VERSION}|{video_identity}"
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]

    def _path(self, video_identity: str) -> Path:
        return self.root / f"video_profile_{self._key(video_identity)}.json"

    def get(self, video_identity: str) -> VideoTimelineProfile | None:
        path = self._path(video_identity)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        # A cached extraction failure is a failure, not an empty profile.
        # Without this, a failed extraction validates into a default profile
        # with no landmarks, which downstream would read as "no evidence"
        # rather than "we already know this file cannot be profiled".
        if payload.get("error"):
            return None
        if int(payload.get("profile_version") or 0) != VIDEO_PROFILE_VERSION:
            return None
        try:
            return VideoTimelineProfile.model_validate(payload)
        except Exception:
            return None

    def set(self, video_identity: str, profile: VideoTimelineProfile) -> None:
        try:
            self._path(video_identity).write_text(
                profile.model_dump_json(), encoding="utf-8"
            )
        except OSError:  # pragma: no cover - a cache write failure is not fatal
            logger.debug("[video-timeline] profile cache write failed", exc_info=True)

    def record_failure(self, video_identity: str, reason: str) -> None:
        """Cache the failure so a broken file is not re-probed repeatedly."""
        try:
            self._path(video_identity).write_text(
                json.dumps({"profile_version": VIDEO_PROFILE_VERSION, "error": reason}),
                encoding="utf-8",
            )
        except OSError:  # pragma: no cover
            logger.debug("[video-timeline] failure cache write failed", exc_info=True)


def _run(command: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def probe_container(path: Path, *, timeout: float) -> dict | None:
    """ffprobe stream and duration facts. Header only; no decoding."""
    if not shutil.which("ffprobe"):
        return None
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        timeout,
    )
    if result.returncode != 0 or not result.stdout:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def detect_audio_activity(
    path: Path,
    *,
    duration_ms: int | None,
    timeout: float,
    stream_index: int | None = None,
) -> list[int]:
    """Coarse activity boundaries via silencedetect.

    This is not speech recognition and not VAD. It asks one cheap question:
    where are the long silences. A different cut inserts or removes material,
    which moves these boundaries; a global subtitle offset does not.
    """
    if not shutil.which("ffmpeg"):
        return []
    if stream_index is None:
        return []
    # A coarse activity floor keeps the cost bounded on long files. The
    # threshold is deliberately generous: only long silences count.
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(path),
        "-map",
        f"0:{stream_index}",
        "-af",
        f"silencedetect=noise=-45dB:d={MIN_SILENCE_MS / 1000:.3f}",
        "-f",
        "null",
        "-",
    ]
    try:
        result = _run(command, timeout)
    except (OSError, subprocess.SubprocessError):
        return []
    text = f"{result.stdout}\n{result.stderr or ''}"
    landmarks: list[int] = []
    for line in text.splitlines():
        if "silence_start:" not in line and "silence_end:" not in line:
            continue
        _, _, value = line.partition(":")
        try:
            seconds = float(value.strip().split(" ")[0])
        except (ValueError, IndexError):
            continue
        landmarks.append(int(seconds * 1000))
    return sorted(set(landmarks))


def detect_scene_boundaries(
    path: Path, *, duration_ms: int | None, timeout: float, limit: int = 400
) -> list[int]:
    """Coarse scene changes via scdet, thinned to major boundaries.

    Low-weight evidence: re-encoding perturbs scene detection, so a mismatch
    here alone is never decisive. It is recorded, and it is reported.
    """
    if not shutil.which("ffmpeg"):
        return []
    # Decode at a tiny resolution. Scene structure survives downscaling; the
    # pixel cost does not.
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(path),
        "-vf",
        "scale=160:-2,scdet=threshold=12",
        "-an",
        "-f",
        "null",
        "-",
    ]
    try:
        result = _run(command, timeout)
    except (OSError, subprocess.SubprocessError):
        return []
    text = f"{result.stdout}\n{result.stderr or ''}"
    landmarks: list[int] = []
    last = -MIN_SCENE_GAP_MS
    for line in text.splitlines():
        if "lavfi.scd.score" not in line and "scdet" not in line:
            continue
        marker = " t: "
        if marker not in line:
            continue
        tail = line.split(marker, 1)[1]
        try:
            seconds = float(tail.split(" ")[0])
        except ValueError:
            continue
        ms = int(seconds * 1000)
        if ms - last < MIN_SCENE_GAP_MS:
            continue
        last = ms
        landmarks.append(ms)
        if len(landmarks) >= limit:
            break
    if duration_ms:
        landmarks = [m for m in landmarks if 0 <= m <= duration_ms]
    return landmarks


def extract_video_profile(
    path: str | Path,
    *,
    timeout: float = 120.0,
    include_video_landmarks: bool = True,
) -> VideoTimelineProfile | None:
    """Build a profile from a real file, or return None to mean abstain.

    ``None`` is not an error to be raised: it means this file cannot be
    examined here, which is a routine outcome on a machine without ffmpeg.
    """
    import time

    media = Path(path)
    if not media.is_file():
        return None
    started = time.monotonic()
    container = probe_container(media, timeout=timeout)
    if not container:
        return None

    streams = container.get("streams") or []
    fmt = container.get("format") or {}
    duration_ms: int | None = None
    raw_duration = fmt.get("duration")
    if raw_duration not in (None, "N/A"):
        try:
            duration_ms = int(float(raw_duration) * 1000)
        except (TypeError, ValueError):
            duration_ms = None

    # §12: a container start time is not a programme start time. Landmarks are
    # normalised to be programme-relative, or a file with intro padding would
    # look like a different cut.
    start_ms = 0
    for key in ("start_time",):
        raw_start = fmt.get(key)
        if raw_start not in (None, "N/A"):
            try:
                start_ms = int(float(raw_start) * 1000)
            except (TypeError, ValueError):
                start_ms = 0
            break

    choice = select_audio_stream(streams)
    audio_count = sum(1 for s in streams if s.get("codec_type") == "audio")
    video_count = sum(1 for s in streams if s.get("codec_type") == "video")

    audio_landmarks: list[int] = []
    video_landmarks: list[int] = []
    if choice.index is not None and not choice.ambiguous:
        stream = streams[choice.index] if choice.index < len(streams) else None
        # Prefer the stream's own start time; fall back to the container's.
        stream_start = start_ms
        if stream is not None:
            raw = (stream.get("start_time"))
            if raw not in (None, "N/A"):
                try:
                    stream_start = int(float(raw) * 1000)
                except (TypeError, ValueError):
                    stream_start = start_ms
        audio_landmarks = [
            max(0, value - stream_start)
            for value in detect_audio_activity(
                media, duration_ms=duration_ms, timeout=timeout, stream_index=choice.index
            )
        ]
        if include_video_landmarks and video_count:
            video_landmarks = [
                max(0, value - start_ms)
                for value in detect_scene_boundaries(
                    media, duration_ms=duration_ms, timeout=timeout
                )
            ]

    return VideoTimelineProfile(
        duration_ms=duration_ms,
        audio_stream_count=audio_count,
        video_stream_count=video_count,
        selected_audio_stream=choice.index,
        audio_stream_selection=choice.reason,
        audio_stream_ambiguous=choice.ambiguous,
        audio_start_time_ms=start_ms,
        audio_landmarks=sorted(audio_landmarks),
        video_landmarks=sorted(video_landmarks),
        extraction_ms=int((time.monotonic() - started) * 1000),
    )


# --- comparison ------------------------------------------------------------ #


def _activity_onsets_from_cues(cue_starts: list[int]) -> list[int]:
    """Coarse activity onsets implied by subtitle timing.

    A gap longer than the silence threshold is treated as a quiet stretch, so
    the subtitle's own structure yields landmarks of the same *kind* as the
    video's silencedetect output. That is what makes the two comparable
    without pretending to understand either one.
    """
    if not cue_starts:
        return []
    landmarks: list[int] = [cue_starts[0]]
    for previous, current in zip(cue_starts, cue_starts[1:], strict=False):
        if current - previous >= MIN_SILENCE_MS:
            landmarks.append(previous)
    return landmarks


def _similarity(series_a: list[int], series_b: list[int], tolerance_ms: int) -> float | None:
    """Fraction of A's landmarks matched by a B landmark within tolerance.

    Symmetric in intent but computed on A so the number answers "how much of
    this subtitle's structure does the video corroborate". A global offset is
    *not* forgiven here: that is deliberate, because a subtitle that sits 95s
    away from the video's activity is a different edit, not a shifted copy.
    Callers that want offset tolerance must use ``_best_aligned_similarity``.
    """
    if not series_a or not series_b:
        return None
    matched = 0
    remaining = list(series_b)
    for point in series_a:
        best_index = -1
        best_delta = tolerance_ms + 1
        for index, other in enumerate(remaining):
            delta = abs(point - other)
            if delta <= tolerance_ms and delta < best_delta:
                best_delta = delta
                best_index = index
        if best_index >= 0:
            matched += 1
            remaining.pop(best_index)
    return round(matched / len(series_a), 4)


def _best_aligned_similarity(
    series_a: list[int],
    series_b: list[int],
    tolerance_ms: int,
    *,
    prefer_offset: int = 0,
    tie_epsilon: float = 1e-9,
) -> tuple[float | None, int]:
    """Best similarity over global offsets, and the offset that achieved it.

    §8 requires that a large stable offset not be confused with a wrong cut. A
    constant shift moves every landmark by the same amount, so some single
    offset explains the structure. A wrong cut does not: no offset explains
    the whole timeline.

    Candidate offsets come from the actual cross-correlation of the two
    landmark sets, not from an arbitrary grid. A fixed grid silently failed to
    represent a 95s shift on shorter timelines, which would have turned a
    legitimate offset into a false mismatch.

    Ties are broken toward ``prefer_offset`` rather than toward whichever
    offset happens to sort first. Without that, a short region can tie at a low
    score across thousands of offsets and report an arbitrary one, which then
    reads as regional inconsistency that is not in the audio at all.
    """
    if not series_a or not series_b:
        return None, 0
    if len(series_a) * len(series_b) <= 4096:
        candidates = {b - a for a in series_a for b in series_b}
    else:
        span = series_a[-1] - series_a[0] if len(series_a) > 1 else 0
        step = max(1, span // 200)
        candidates = set(range(-span, span + 1, step)) if span > 0 else {0}
    best_score = -1.0
    best_offset = 0
    for offset in sorted(candidates):
        score = _similarity([value + offset for value in series_a], series_b, tolerance_ms)
        if score is None:
            continue
        if score > best_score + tie_epsilon:
            best_score = score
            best_offset = offset
        elif score >= best_score - tie_epsilon and (
            abs(offset - prefer_offset) < abs(best_offset - prefer_offset)
        ):
            best_score = max(best_score, score)
            best_offset = offset
    return (best_score if best_score >= 0 else None), best_offset


def _correlation_settings() -> tuple[int, float]:
    """Region count and ambiguity ratio, from the project's settings model."""
    try:
        from app.config import settings

        regions = int(getattr(settings, "VIDEO_TIMELINE_REGIONS", 4) or 4)
        ratio = float(getattr(settings, "VIDEO_TIMELINE_AMBIGUITY_RATIO", 0.85) or 0.85)
    except Exception:  # pragma: no cover - settings always present in practice
        regions, ratio = 4, 0.85
    return max(2, regions), ratio


def _splice_signature(
    series_a: list[int], series_b: list[int], offset: int, tolerance_ms: int
) -> tuple[float | None, float | None]:
    """How well each half of the timeline agrees at one global offset.

    This is the structural test for a wrong cut. A genuine match, and a genuine
    global offset, are both explained by ONE shift, so the two halves agree. An
    edit that splices different material leaves one half explained and the
    other unexplained, and no threshold on the overall average can see that,
    because the average is a number in the middle.

    Returns ``(first_half, second_half)`` similarity, or ``(None, None)`` when
    there are too few landmarks to split meaningfully.
    """
    if len(series_a) < 6:
        return None, None
    midpoint = len(series_a) // 2
    first = [value + offset for value in series_a[:midpoint]]
    second = [value + offset for value in series_a[midpoint:]]
    return (
        _similarity(first, series_b, tolerance_ms),
        _similarity(second, series_b, tolerance_ms),
    )


class CorrelationClarity(str, Enum):
    """Whether one offset can be trusted to explain the timeline."""

    CLEAR = "clear"
    AMBIGUOUS = "ambiguous"
    NO_SIGNAL = "no_signal"


class RegionalConsistency(str, Enum):
    """Whether the best shift stays the same across the timeline."""

    CONSISTENT = "consistent"
    MIXED = "mixed"
    SCATTERED = "scattered"
    UNKNOWN = "unknown"


class CorrelationAnalysis(BaseModel):
    """Whole-timeline and per-region alignment, kept separate.

    A global offset and a structural cut both produce a best shift. What
    separates them is whether that shift holds everywhere, so the regional
    shifts are recorded rather than averaged into one number.
    """

    best_offset_ms: int = 0
    best_score: float | None = None
    second_score: float | None = None
    #: How close the runner-up peak is to the winner, relative to the winner.
    peak_ratio: float | None = None
    clarity: CorrelationClarity = CorrelationClarity.NO_SIGNAL
    regions: int = 2
    region_offsets: list[int] = Field(default_factory=list)
    region_scores: list[float] = Field(default_factory=list)
    consistency: RegionalConsistency = RegionalConsistency.UNKNOWN


def analyse_correlation(
    series_a: list[int],
    series_b: list[int],
    *,
    tolerance_ms: int = 1_500,
    regions: int = 2,
    ambiguity_ratio: float = 0.85,
    min_region_points: int = 3,
) -> CorrelationAnalysis:
    """Find the best global shift, then check whether it holds everywhere.

    The best peak is not accepted on its own. A timeline that correlates about
    as well at two different shifts has not been aligned, it has been guessed
    at, and guessing must not become a verification.
    """
    analysis = CorrelationAnalysis(regions=max(2, regions))
    if len(series_a) < 4 or len(series_b) < 4:
        analysis.clarity = CorrelationClarity.NO_SIGNAL
        return analysis

    best_score, best_offset = _best_aligned_similarity(series_a, series_b, tolerance_ms)
    analysis.best_score = best_score
    analysis.best_offset_ms = best_offset
    if best_score is None:
        analysis.clarity = CorrelationClarity.NO_SIGNAL
        return analysis

    # The runner-up: the best score achievable at any offset OTHER than the
    # winner, and at least one region-width away from it, so a near-identical
    # neighbouring shift is not mistaken for a competing explanation.
    competitors: list[float] = []
    span = max(1, series_b[-1] - series_b[0])
    separation = max(tolerance_ms * 4, span // 20)
    shift = max(1, tolerance_ms)
    probe = -span
    while probe <= span:
        if abs(probe - best_offset) > separation:
            score = _similarity(
                [value + probe for value in series_a], series_b, tolerance_ms
            )
            if score is not None:
                competitors.append(score)
        probe += shift
    second = max(competitors) if competitors else None
    analysis.second_score = second
    analysis.peak_ratio = round(second / best_score, 4) if second is not None and best_score else None
    analysis.clarity = (
        CorrelationClarity.CLEAR
        if (analysis.peak_ratio is None or analysis.peak_ratio < ambiguity_ratio)
        else CorrelationClarity.AMBIGUOUS
    )

    # Regional shifts, for the structural question.
    count = analysis.regions
    size = len(series_a) // count
    if size < min_region_points:
        analysis.consistency = RegionalConsistency.UNKNOWN
        return analysis
    for region in range(count):
        chunk = series_a[region * size : (region + 1) * size]
        # Regions are measured relative to the global alignment, so their
        # offsets are deviations from one answer and are comparable.
        score, offset = _best_aligned_similarity(
            chunk, series_b, tolerance_ms, prefer_offset=best_offset
        )
        analysis.region_scores.append(round(score, 4) if score is not None else 0.0)
        analysis.region_offsets.append(offset)

    spread = max(analysis.region_offsets) - min(analysis.region_offsets)
    if spread <= tolerance_ms:
        analysis.consistency = RegionalConsistency.CONSISTENT
    elif spread <= tolerance_ms * 8:
        analysis.consistency = RegionalConsistency.MIXED
    else:
        analysis.consistency = RegionalConsistency.SCATTERED
    return analysis


def validate_timeline_against_reference(
    profile: VideoTimelineProfile | None,
    reference_cue_starts: list[int],
    *,
    landmark_tolerance_ms: int = 1_500,
    min_landmarks: int = 4,
    require_offset_invariant: bool = True,
    # Declared, not tuned. A subtitle the video corroborates shares at least
    # this much of its silence structure with it. Nothing adapts these values,
    # and while the layer is observational they cannot change a user-visible
    # outcome.
    audio_match_floor: float = 0.6,
    scene_match_floor: float = 0.35,
    # How far apart the two halves must be before the timeline is called
    # spliced. A real offset leaves them within a hair of each other.
    splice_gap: float = 0.4,
) -> VideoTimelineEvidence:
    """Compare a video's timeline with a subtitle's implied timeline.

    The video side comes only from the file. The subtitle side is reduced to
    the same kind of landmark, so the comparison is structural over the whole
    timeline rather than a single offset check.
    """
    if profile is None:
        return VideoTimelineEvidence(
            available=False,
            verdict=VideoVerdict.UNAVAILABLE,
            reason=VideoFailureReason.NO_VIDEO,
            detail=["no video profile; abstaining rather than guessing"],
        )

    evidence = VideoTimelineEvidence(
        available=True,
        profile_version=profile.profile_version,
        profile_digest=profile.digest,
        video_duration_ms=profile.duration_ms,
        audio_landmark_count=len(profile.audio_landmarks),
        scene_landmark_count=len(profile.video_landmarks),
        audio_stream_count=profile.audio_stream_count,
        selected_audio_stream=profile.selected_audio_stream,
        audio_stream_selection=profile.audio_stream_selection,
        audio_landmarks=list(profile.audio_landmarks),
        scene_landmarks=list(profile.video_landmarks),
    )
    cue_starts = sorted({int(value) for value in reference_cue_starts if value >= 0})
    if cue_starts:
        evidence.reference_span_ms = cue_starts[-1] - cue_starts[0]
    if profile.duration_ms and cue_starts:
        # Computed before any early return so the signal stays visible even
        # when another signal has already settled the verdict. Duration is
        # supporting evidence only: a different edit can share a runtime, and
        # a global offset does not change one.
        span = max(cue_starts[-1], 1)
        ratio = min(span, profile.duration_ms) / max(span, profile.duration_ms)
        evidence.duration_similarity = round(ratio, 4)

    if not profile.audio_stream_count:
        evidence.verdict = VideoVerdict.INSUFFICIENT_EVIDENCE
        evidence.reason = VideoFailureReason.NO_AUDIO_STREAM
        evidence.detail.append("target has no audio stream to compare against")
        return evidence
    if profile.audio_stream_ambiguous:
        # Several plausible programme tracks and no default disposition. Reading
        # the wrong one would compare the subtitle against the wrong audio, so
        # the correct outcome is to abstain rather than pick.
        evidence.verdict = VideoVerdict.INSUFFICIENT_EVIDENCE
        evidence.reason = VideoFailureReason.AUDIO_STREAM_AMBIGUOUS
        evidence.detail.append(
            "programme audio track not identifiable from container metadata"
        )
        return evidence

    subtitle_landmarks = _activity_onsets_from_cues(cue_starts)
    if len(profile.audio_landmarks) < min_landmarks or len(subtitle_landmarks) < min_landmarks:
        evidence.verdict = VideoVerdict.INSUFFICIENT_EVIDENCE
        evidence.reason = VideoFailureReason.TOO_FEW_LANDMARKS
        evidence.detail.append(
            f"video landmarks={len(profile.audio_landmarks)}, "
            f"subtitle landmarks={len(subtitle_landmarks)}, need {min_landmarks}"
        )
        return evidence

    regions, ambiguity_ratio = _correlation_settings()
    correlation = analyse_correlation(
        subtitle_landmarks,
        profile.audio_landmarks,
        tolerance_ms=landmark_tolerance_ms,
        regions=regions,
        ambiguity_ratio=ambiguity_ratio,
    )
    evidence.correlation_peak = correlation.best_score
    evidence.correlation_second_peak = correlation.second_score
    evidence.correlation_peak_ratio = correlation.peak_ratio
    evidence.correlation_clarity = correlation.clarity.value
    evidence.region_offsets = list(correlation.region_offsets)
    evidence.region_scores = list(correlation.region_scores)
    evidence.regional_consistency = correlation.consistency.value

    # Ambiguity only applies to a STRONG peak that is not unique. A weak best
    # peak is positive evidence that the timeline does not correspond, so it is
    # a mismatch rather than a failure to align: refusing to call that
    # "ambiguous" would let a badly wrong cut pass as merely unclear.
    strong_enough = (
        correlation.best_score is not None
        and correlation.best_score >= audio_match_floor
    )
    if correlation.clarity is CorrelationClarity.AMBIGUOUS and strong_enough:
        # Two shifts explain this about equally well. Nothing has been aligned,
        # and guessing must not become a verification or a mismatch.
        evidence.verdict = VideoVerdict.INSUFFICIENT_EVIDENCE
        evidence.reason = VideoFailureReason.TOO_FEW_LANDMARKS
        evidence.detail.append(
            f"ambiguous correlation: peak={correlation.best_score} "
            f"runner-up={correlation.second_score} ratio={correlation.peak_ratio}"
        )
        return evidence
    if correlation.clarity is CorrelationClarity.NO_SIGNAL:
        evidence.verdict = VideoVerdict.INSUFFICIENT_EVIDENCE
        evidence.reason = VideoFailureReason.TOO_FEW_LANDMARKS
        evidence.detail.append("no usable correlation peak")
        return evidence
    if correlation.clarity is CorrelationClarity.AMBIGUOUS:
        evidence.detail.append(
            "competing peaks exist but the best is weak; treated as disagreement "
            "rather than as ambiguity"
        )

    if require_offset_invariant:
        audio_score = correlation.best_score
        offset = correlation.best_offset_ms
    else:
        audio_score = _similarity(
            subtitle_landmarks, profile.audio_landmarks, landmark_tolerance_ms
        )
        offset = 0
    evidence.audio_activity_similarity = audio_score
    evidence.best_offset_ms = offset
    evidence.detail.append(
        f"audio activity similarity={audio_score} at offset={offset}ms "
        f"({len(subtitle_landmarks)} subtitle landmarks vs "
        f"{len(profile.audio_landmarks)} video)"
    )

    first_half, second_half = _splice_signature(
        subtitle_landmarks, profile.audio_landmarks, offset, landmark_tolerance_ms
    )
    evidence.first_half_similarity = first_half
    evidence.second_half_similarity = second_half
    if first_half is not None and second_half is not None:
        gap = abs(first_half - second_half)
        evidence.detail.append(
            f"half-timeline agreement: first={first_half} second={second_half} "
            f"gap={round(gap, 4)}"
        )
        if gap >= splice_gap:
            # One shift explains part of the timeline and not the rest. That is
            # an edit, not an offset.
            evidence.verdict = VideoVerdict.MISMATCH
            evidence.reason = VideoFailureReason.AUDIO_ACTIVITY_MISMATCH
            evidence.detail.append(
                "splice signature: a single offset explains only part of the timeline"
            )
            return evidence

    scene_score: float | None = None
    if profile.video_landmarks and cue_starts:
        scene_cues = [cue_starts[0]] + [
            cue for cue in cue_starts[1:] if cue - cue_starts[0] >= MIN_SCENE_GAP_MS
        ]
        if require_offset_invariant:
            scene_score, _ = _best_aligned_similarity(
                scene_cues, profile.video_landmarks, max(landmark_tolerance_ms, 2_000)
            )
        else:
            scene_score = _similarity(
                scene_cues, profile.video_landmarks, max(landmark_tolerance_ms, 2_000)
            )
        evidence.scene_boundary_similarity = scene_score
        evidence.detail.append(f"scene structure similarity={scene_score}")

    signals = [s for s in (audio_score, scene_score) if s is not None]
    if not signals:
        evidence.verdict = VideoVerdict.INSUFFICIENT_EVIDENCE
        evidence.reason = VideoFailureReason.TOO_FEW_LANDMARKS
        evidence.detail.append("no comparable timeline signal available")
        return evidence

    audio_mismatch = audio_score is not None and audio_score < audio_match_floor
    scene_mismatch = scene_score is not None and scene_score < scene_match_floor
    if audio_mismatch or scene_mismatch:
        evidence.verdict = VideoVerdict.MISMATCH
        if audio_mismatch:
            evidence.reason = VideoFailureReason.AUDIO_ACTIVITY_MISMATCH
        else:
            evidence.reason = VideoFailureReason.SCENE_STRUCTURE_MISMATCH
        return evidence

    evidence.verdict = VideoVerdict.MATCH
    evidence.reason = VideoFailureReason.NONE
    return evidence


def is_withholding_evidence(evidence: VideoTimelineEvidence) -> bool:
    """Whether this evidence may block a verified claim.

    Only a positive, measured mismatch qualifies. Abstention, unavailability
    and a thin sample never withhold anything, and none of them can ever
    create a verification either.
    """
    return evidence.available and evidence.verdict is VideoVerdict.MISMATCH


def temp_probe_target(path: str | Path) -> str:  # pragma: no cover - test seam
    """Copy a file to a temp location. Kept for parity with alass lifecycle."""
    with tempfile.NamedTemporaryFile(suffix=Path(path).suffix, delete=False) as handle:
        shutil.copyfile(path, handle.name)
        return handle.name


def to_audit_dict(evidence: VideoTimelineEvidence) -> dict:
    """Serialise video evidence for the audit record.

    Measurements and counts only. No audio content, no landmarks list, nothing
    that could carry programme content.
    """
    return {
        "available": evidence.available,
        "verdict": evidence.verdict.value,
        "reason": evidence.reason.value,
        "profile_version": evidence.profile_version,
        "profile_digest": evidence.profile_digest,
        "duration_similarity": evidence.duration_similarity,
        "audio_activity_similarity": evidence.audio_activity_similarity,
        "scene_boundary_similarity": evidence.scene_boundary_similarity,
        "audio_stream_count": evidence.audio_stream_count,
        "selected_audio_stream": evidence.selected_audio_stream,
        "audio_stream_selection": evidence.audio_stream_selection,
        "audio_landmark_count": evidence.audio_landmark_count,
        "scene_landmark_count": evidence.scene_landmark_count,
        "best_offset_ms": evidence.best_offset_ms,
        "first_half_similarity": evidence.first_half_similarity,
        "second_half_similarity": evidence.second_half_similarity,
        "correlation_peak": evidence.correlation_peak,
        "correlation_second_peak": evidence.correlation_second_peak,
        "correlation_peak_ratio": evidence.correlation_peak_ratio,
        "correlation_clarity": evidence.correlation_clarity,
        "regional_consistency": evidence.regional_consistency,
        "audio_detector": evidence.audio_detector,
        "adaptive_threshold_db": evidence.adaptive_threshold_db,
        "adaptive_baseline_db": evidence.adaptive_baseline_db,
        "landmark_quality": evidence.landmark_quality,
    }
