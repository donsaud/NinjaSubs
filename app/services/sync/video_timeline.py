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
    TOO_FEW_LANDMARKS = "VIDEO_EVIDENCE_INSUFFICIENT"
    DURATION_MISMATCH = "VIDEO_DURATION_MISMATCH"
    AUDIO_ACTIVITY_MISMATCH = "AUDIO_ACTIVITY_MISMATCH"
    SCENE_STRUCTURE_MISMATCH = "SCENE_STRUCTURE_MISMATCH"
    NONE = "VIDEO_EVIDENCE_NONE"


class VideoTimelineProfile(BaseModel):
    """A deterministic, text-free summary of one video file's timeline.

    Contains landmarks and counts only. No frames, no samples, no media data.
    """

    duration_ms: int | None = None
    audio_stream_count: int | None = None
    video_stream_count: int | None = None
    #: Onset/offset times of coarse audio activity, in ms.
    audio_landmarks: list[int] = Field(default_factory=list)
    #: Coarse scene-change times, in ms.
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
    # The global shift that best explains the subtitle's structure, and how
    # well each half of the timeline agrees with it. A real match or a real
    # global offset keeps the halves consistent; a spliced edit does not.
    best_offset_ms: int = 0
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
    path: Path, *, duration_ms: int | None, timeout: float
) -> list[int]:
    """Coarse activity boundaries via silencedetect.

    This is not speech recognition and not VAD. It asks one cheap question:
    where are the long silences. A different cut inserts or removes material,
    which moves these boundaries; a global subtitle offset does not.
    """
    if not shutil.which("ffmpeg"):
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
        "0:a:0?",
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
    if duration_ms:
        result.stdout = f"{result.stdout}\n{result.stderr or ''}"
        text = result.stdout
    else:
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
    audio_count = sum(1 for s in streams if s.get("codec_type") == "audio")
    video_count = sum(1 for s in streams if s.get("codec_type") == "video")
    fmt = container.get("format") or {}
    duration_ms: int | None = None
    raw_duration = fmt.get("duration")
    if raw_duration not in (None, "N/A"):
        try:
            duration_ms = int(float(raw_duration) * 1000)
        except (TypeError, ValueError):
            duration_ms = None

    audio_landmarks = (
        detect_audio_activity(media, duration_ms=duration_ms, timeout=timeout)
        if audio_count
        else []
    )
    video_landmarks = (
        detect_scene_boundaries(media, duration_ms=duration_ms, timeout=timeout)
        if (include_video_landmarks and video_count)
        else []
    )
    return VideoTimelineProfile(
        duration_ms=duration_ms,
        audio_stream_count=audio_count,
        video_stream_count=video_count,
        audio_landmarks=audio_landmarks,
        video_landmarks=video_landmarks,
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
    series_a: list[int], series_b: list[int], tolerance_ms: int
) -> tuple[float | None, int]:
    """Best similarity over global offsets, and the offset that achieved it.

    §8 requires that a large stable offset not be confused with a wrong cut. A
    constant shift moves every landmark by the same amount, so some single
    offset explains the structure. A wrong cut does not: no offset explains
    the whole timeline.

    Candidate offsets come from the actual cross-correlation of the two
    landmark sets, not from an arbitrary grid. A fixed grid silently failed to
    represent a 95s shift on shorter timelines, which would have turned a
    legitimate offset into a false mismatch. With many landmarks the exact
    product is too large, so it falls back to a coarse sweep.
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
        if score is not None and score > best_score:
            best_score = score
            best_offset = offset
    return (best_score if best_score >= 0 else None), best_offset


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

    subtitle_landmarks = _activity_onsets_from_cues(cue_starts)
    if len(profile.audio_landmarks) < min_landmarks or len(subtitle_landmarks) < min_landmarks:
        evidence.verdict = VideoVerdict.INSUFFICIENT_EVIDENCE
        evidence.reason = VideoFailureReason.TOO_FEW_LANDMARKS
        evidence.detail.append(
            f"video landmarks={len(profile.audio_landmarks)}, "
            f"subtitle landmarks={len(subtitle_landmarks)}, need {min_landmarks}"
        )
        return evidence

    if require_offset_invariant:
        audio_score, offset = _best_aligned_similarity(
            subtitle_landmarks, profile.audio_landmarks, landmark_tolerance_ms
        )
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
