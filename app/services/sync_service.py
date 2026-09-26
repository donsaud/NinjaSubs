"""Subtitle synchronization via the ``alass`` engine.

``alass`` (Rust) aligns a target subtitle against a trusted reference, handling
discontinuous offsets (intro/recap, commercial cuts) that naive global-offset
math cannot. It is executed as a subprocess with a strict timeout so a slow or
missing binary can never hang a player request.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import subprocess
import tempfile
import time

from app.config import settings

logger = logging.getLogger(__name__)

# SDH / accessibility indicators stripped from the REFERENCE file only.
# The negated classes exclude newlines on purpose: an unclosed bracket on one
# cue must never eat whole cue blocks further down the file (that silently
# deleted over half of a real episode file and collapsed its runtime).
_SDH_BRACKETS_REGEX = re.compile(r"\[[^\]\n]*\]|\([^)\n]*\)|\u266a[^\u266a\n]*\u266a")
_SDH_MUSIC_REGEX = re.compile(r"\u266a[^\u266a\n]*\u266a")
_BLANK_LINES_REGEX = re.compile(r"\n{3,}")
# Inline ASS/SSA style overrides that sometimes leak into SRT files (e.g.
# ``{\fs36\fad(300,1500)\c&HEDE829&Comic Sans Ms}``) and crash alass's parser.
# Same single-line discipline as the SDH patterns: a stray "{" must not eat
# subsequent cues while looking for its closing brace.
_INLINE_ASS_TAG_REGEX = re.compile(r"\{[^}\n]*\}")

# A SubRip timespan line, tolerating trailing positioning/styling tokens that
# some generators append (e.g. ``00:03:03,508 --> 00:03:04,592 X1:0 Y1:0``).
_TIMESPAN_LINE_REGEX = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})"
)

# Standalone positioning/style token lines to drop from cue text.
_POSITION_TOKEN_LINE_REGEX = re.compile(
    r"^\s*(?:x[12]|y[12]|w\d*|h\d*|line|position|align|margin[LVR]?|start|end)\s*[:=].*$",
    re.IGNORECASE,
)

# A reference/target must contain at least this many well-formed cues to be
# worth handing to alass.
_MIN_VALID_CUES = 5


def strip_sdh(text: str) -> str:
    """Remove SDH indicators (``[...]``, ``(...)``, ``♪...♪``) from reference text."""
    cleaned = _SDH_BRACKETS_REGEX.sub(" ", text)
    cleaned = _SDH_MUSIC_REGEX.sub(" ", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return _BLANK_LINES_REGEX.sub("\n\n", cleaned)


def _vtt_to_srt(text: str) -> str:
    """Minimal WebVTT -> SRT conversion (timestamps + cue text)."""
    blocks = re.split(r"\n[ \t]*\n", text.replace("\r\n", "\n"))
    rendered: list[str] = []
    for block in blocks:
        block = block.strip("\n")
        if not block or block.upper().startswith("WEBVTT") or block.startswith("NOTE"):
            continue
        lines = [line for line in block.split("\n")]
        timestamp_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timestamp_index is None:
            continue
        raw_ts = lines[timestamp_index]
        start, _, end_and_settings = raw_ts.partition("-->")
        end = end_and_settings.strip().split(" ")[0]
        start = start.strip().replace(".", ",")
        end = end.replace(".", ",")
        cue_text = "\n".join(lines[timestamp_index + 1 :]).strip()
        if not cue_text:
            continue
        rendered.append(f"{len(rendered) + 1}\n{start} --> {end}\n{cue_text}")
    return "\n\n".join(rendered) + ("\n" if rendered else "")


def _parse_timestamp_ms(value: str) -> int:
    """Convert ``HH:MM:SS,mmm`` (or ``.`` separator) to milliseconds."""
    time_part, _, fraction = value.replace(",", ".").partition(".")
    hours, minutes, seconds = (int(part) for part in time_part.split(":"))
    millis = int(fraction.ljust(3, "0")[:3])
    return ((hours * 3600) + (minutes * 60) + seconds) * 1000 + millis


def _format_timestamp(value: str) -> str:
    """Re-format a parsed timestamp as strict ``HH:MM:SS,mmm``."""
    time_part, _, fraction = value.replace(",", ".").partition(".")
    hours, minutes, seconds = (int(part) for part in time_part.split(":"))
    millis = fraction.ljust(3, "0")[:3]
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis}"


def _cue_spans(text: str) -> list[tuple[int, int]]:
    """``(start_ms, end_ms)`` for cues with well-formed, ordered timespans."""
    spans: list[tuple[int, int]] = []
    for match in _TIMESPAN_LINE_REGEX.finditer(text or ""):
        start = _parse_timestamp_ms(match.group("start"))
        end = _parse_timestamp_ms(match.group("end"))
        if start < end:
            spans.append((start, end))
    return spans


def _valid_cue_count(text: str) -> int:
    """Number of well-formed cues whose start timestamp is before its end."""
    return len(_cue_spans(text))


# Strict full-line timespan: catches minus signs and trailing garbage that a
# searching regex would silently skip over.
_STRICT_TIMESPAN_LINE_REGEX = re.compile(
    r"^\s*(?P<sneg>-)?(?P<sh>\d{1,2}):(?P<smin>\d{2}):(?P<ss>\d{2})[,.](?P<sf>\d{1,3})"
    r"\s*-->\s*(?P<eneg>-)?(?P<eh>\d{1,2}):(?P<em>\d{2}):(?P<es>\d{2})[,.](?P<ef>\d{1,3})\s*$"
)


def _validate_synced_output(target_srt: str, synced: str) -> str | None:
    """Reject structurally unsound alass output; return a reason or ``None`` if sound.

    Guards: cue retention (>= 85% of the target's cues), per-cue integrity
    (no negative or inverted timestamps, no malformed timespan lines), and
    total duration within [0.5x, 2.0x] of the target (no collapse/blowup).
    """
    target_spans = _cue_spans(target_srt)
    if not target_spans:
        return "no valid target cues to compare against"
    out_spans: list[tuple[int, int]] = []
    for line in synced.splitlines():
        if "-->" not in line:
            continue
        match = _STRICT_TIMESPAN_LINE_REGEX.match(line.strip())
        if match is None:
            return f"malformed timespan line: {line.strip()[:60]!r}"
        parts = match.groupdict()
        if parts["sneg"] or parts["eneg"]:
            return "negative timestamp in output"
        start = (
            (int(parts["sh"]) * 3600 + int(parts["smin"]) * 60 + int(parts["ss"])) * 1000
            + int(parts["sf"].ljust(3, "0")[:3])
        )
        end = (
            (int(parts["eh"]) * 3600 + int(parts["em"]) * 60 + int(parts["es"])) * 1000
            + int(parts["ef"].ljust(3, "0")[:3])
        )
        if not start < end:
            return f"inverted cue ({start} >= {end} ms)"
        out_spans.append((start, end))
    if len(out_spans) < len(target_spans) * 0.85:
        return f"cue retention {len(out_spans)}/{len(target_spans)} below 85%"
    target_duration = max(end for _, end in target_spans) - min(start for start, _ in target_spans)
    out_duration = max(end for _, end in out_spans) - min(start for start, _ in out_spans)
    if out_duration < 0.5 * target_duration or out_duration > 2.0 * target_duration:
        return (
            f"duration {out_duration / 1000.0:.1f}s outside [0.5x, 2.0x] "
            f"of target {target_duration / 1000.0:.1f}s"
        )
    return None


def _cue_starts_ms(text: str, limit: int = 3) -> list[int]:
    """First ``limit`` cue start timestamps (milliseconds) from an SRT string."""
    starts: list[int] = []
    for match in _TIMESPAN_LINE_REGEX.finditer(text or ""):
        starts.append(_parse_timestamp_ms(match.group("start")))
        if len(starts) >= limit:
            break
    return starts


def _percentile_cue_end_ms(text: str, percentile: float = 0.98) -> int:
    """Cue end timestamp at the given percentile (milliseconds).

    Translator credits and outro cards routinely sit minutes past the last
    dialogue cue; the 98th percentile measures the content runtime while
    ignoring that trailing tail. Returns 0 when no cue ends parse.
    """
    ends = sorted(
        _parse_timestamp_ms(match.group("end"))
        for match in _TIMESPAN_LINE_REGEX.finditer(text or "")
    )
    if not ends:
        return 0
    index = min(len(ends) - 1, max(0, math.ceil(percentile * len(ends)) - 1))
    return ends[index]


# Runtime tolerance bands for the duration compatibility gate. Films get a
# flat 5-minute floor (credit/outro drift on top of the percentile measure);
# series get a 2-minute floor. Both scale to 8% of the reference runtime.
_FILM_RUNTIME_SECONDS = 3600.0
_FILM_GAP_FLOOR_SECONDS = 300.0
_SERIES_GAP_FLOOR_SECONDS = 120.0
_DURATION_GAP_RATIO = 0.08
# When target and reference are a bilaterally confirmed retail-disc pair
# (BluRay/REMUX) they share a master, so a larger runtime gap is far more
# likely an extended/recap cut than a wrong one; allow a wider band and let
# alass + the shift guardrail decide.
_SOURCE_CONFIRMED_GAP_RATIO = 0.15

# A shift whose per-cue spread stays within this band (after trimming a single
# extreme sample) is a *uniform constant offset* — an intro/logo/bumper length
# difference — not a drifting (wrong-cut) alignment.
_UNIFORM_SHIFT_SPREAD_SECONDS = 3.0


def _uniform_shift_spread(shifts: list[float], min_samples: int = 8) -> float:
    """Spread of the sampled shifts with a single extreme outlier trimmed.

    A lone early-cue artifact (translator credit/pre-roll) must not mark an
    otherwise-constant offset as variable.
    """
    ordered = sorted(shifts)
    if len(ordered) >= min_samples:
        ordered = ordered[1:-1]
    if not ordered:
        return 0.0
    return ordered[-1] - ordered[0]

# Relaxed fallback (uninformative target) has no edition confirmation to lean
# on, so the timeline must agree much more tightly: 5% of the runtime (1-minute
# floor) instead of 8%, and the cue population must be comparable.
_RELAXED_GAP_FLOOR_SECONDS = 60.0
_RELAXED_GAP_RATIO = 0.05
# A different cut of the same content routinely differs in cue count; a whole
# season/multi-episode pack or a substantially different master diverges far
# more, so reject beyond this ratio (larger / smaller cue count).
_MAX_CUE_COUNT_RATIO = 2.5
# A reference whose cue starts jump backwards by more than a minute is a
# multi-episode/season pack (each episode restarts near zero), not a single.
_PACK_RESET_THRESHOLD_MS = 60_000
_MIN_PACK_RESETS = 2


def _backward_jumps(spans: list[tuple[int, int]]) -> int:
    """Count cue starts that jump backwards relative to the running max end."""
    jumps = 0
    running_end = 0
    for start, end in spans:
        if running_end and start + _PACK_RESET_THRESHOLD_MS < running_end:
            jumps += 1
        running_end = max(running_end, end)
    return jumps


def _timeline_rejection(
    target_srt: str,
    reference: str,
    *,
    decision_kind: str,
    relaxed: bool,
    source_confirmed: bool = False,
    reference_partial: bool = False,
) -> str | None:
    """Return a rejection reason when target/reference timelines cannot match.

    Multi-signal pre-alass gate: percentile runtime (as before), plus — for
    best-effort ``edition`` references only, where no exact cut is proven —
    a cue-count sanity ratio and a season-pack/multi-episode reset detector.
    Relaxed fallbacks tighten the runtime band further. A ``reference_partial``
    (sampled prefix, e.g. the first 15 min of an embedded track) is exempt from
    the runtime/cue-count gates because its end is intentionally truncated.
    """
    if reference_partial:
        return None
    target_duration = _percentile_cue_end_ms(target_srt) / 1000.0
    ref_duration = _percentile_cue_end_ms(reference) / 1000.0
    if ref_duration >= _FILM_RUNTIME_SECONDS:
        tolerance = max(_FILM_GAP_FLOOR_SECONDS, _DURATION_GAP_RATIO * ref_duration)
    else:
        tolerance = max(_SERIES_GAP_FLOOR_SECONDS, _DURATION_GAP_RATIO * ref_duration)
    if source_confirmed:
        tolerance = max(
            tolerance,
            _SOURCE_CONFIRMED_GAP_RATIO * max(ref_duration, target_duration),
        )
    if relaxed:
        tolerance = min(
            tolerance,
            max(
                _RELAXED_GAP_FLOOR_SECONDS,
                _RELAXED_GAP_RATIO * max(ref_duration, target_duration),
            ),
        )
    if abs(target_duration - ref_duration) > tolerance:
        return (
            f"duration mismatch detected (target: {target_duration:.1f}s, "
            f"ref: {ref_duration:.1f}s)"
        )

    # Structural checks only for best-effort edition references: exact kinds
    # (hash/embedded/team) legitimately differ in cue population.
    if decision_kind != "edition":
        return None

    target_spans = _cue_spans(target_srt)
    ref_spans = _cue_spans(reference)
    if not target_spans or not ref_spans:
        return None
    ratio = max(len(target_spans), len(ref_spans)) / max(
        1, min(len(target_spans), len(ref_spans))
    )
    if ratio > _MAX_CUE_COUNT_RATIO:
        return (
            f"cue-count mismatch (target: {len(target_spans)}, ref: {len(ref_spans)})"
        )
    resets = _backward_jumps(ref_spans)
    if resets >= _MIN_PACK_RESETS:
        return f"reference looks like a season pack ({resets} timestamp resets)"
    return None


def sanitize_subtitle(text: str) -> str:
    """
    Normalise a subtitle string into strict UTF-8-safe SRT before ``alass``:

    - strip a UTF-8 BOM and NUL bytes,
    - normalise CRLF/CR line endings to ``\\n``,
    - convert ASS/SSA, WebVTT, and SAMI payloads to SRT,
    - strip inline ASS/SSA style override tags (``{\\fs36 ...}``) that leak into SRT,
    - drop orphan/empty cues, re-index contiguously, and guarantee a trailing newline.
    """
    if not text:
        return ""
    text = text.lstrip("\ufeff").replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    leading = text.lstrip()[:200].lower()
    if (
        leading.startswith("[script info]")
        or leading.startswith("[v4")
        or "dialogue:" in text.lower()
    ):
        from app.utils.ass_converter import convert_ass_to_srt

        text = convert_ass_to_srt(text, apply_rtl=False)
    elif leading.startswith("webvtt"):
        text = _vtt_to_srt(text)
    elif "<sami" in leading or re.search(r"<sync\s+start\s*=", leading):
        from app.utils.sami_converter import convert_sami_to_srt

        text = convert_sami_to_srt(text)

    text = _INLINE_ASS_TAG_REGEX.sub("", text)
    return normalize_srt_blocks(text)


def normalize_srt_blocks(text: str) -> str:
    """Strip inline tags, normalise timespans, drop empty cues, and re-index SRT blocks."""
    text = _INLINE_ASS_TAG_REGEX.sub("", text)
    blocks = re.split(r"\n[ \t]*\n", text)
    rendered: list[str] = []
    for block in blocks:
        lines = [line.rstrip() for line in block.strip("\n").split("\n")]
        timestamp_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timestamp_index is None:
            continue
        # Strictly re-emit the timespan, discarding trailing position/style tokens
        # (``X1:0``, ``Y1:0``, ``line:...``, ``position:...``) that crash alass.
        match = _TIMESPAN_LINE_REGEX.search(lines[timestamp_index])
        if match is None:
            continue
        timespan = (
            f"{_format_timestamp(match.group('start'))} --> "
            f"{_format_timestamp(match.group('end'))}"
        )
        text_lines = [
            line.strip()
            for line in lines[timestamp_index + 1 :]
            if line.strip() and not _POSITION_TOKEN_LINE_REGEX.match(line)
        ]
        if not text_lines:
            continue
        rendered.append(f"{len(rendered) + 1}\n{timespan}\n" + "\n".join(text_lines))
    return "\n\n".join(rendered) + ("\n" if rendered else "")


# Dynamic ``alass`` execution budget by payload size (max target/reference chars).
_TIMEOUT_TIERS: tuple[tuple[int, float], ...] = (
    (100_000, 30.0),  # full-length movies: the DP alignment matrix needs room
    (50_000, 6.0),  # long episodes / extras
    (0, 4.0),  # 20-45 min TV episodes
)


class SubtitleSyncService:
    """Synchronize an Arabic subtitle against a reference subtitle with ``alass``."""

    def __init__(
        self,
        alass_path: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.alass_path = alass_path or getattr(settings, "ALASS_PATH", "alass") or "alass"
        configured = timeout
        if configured is None:
            try:
                configured = float(getattr(settings, "ALASS_TIMEOUT_SECONDS", 10.0))
            except (TypeError, ValueError):
                configured = 10.0
        self.timeout = configured if configured and configured > 0 else 10.0
        self._semaphore = asyncio.Semaphore(max(1, settings.ALASS_MAX_CONCURRENT_SYNCS))

    def _effective_timeout(self, target_srt: str, reference_srt: str) -> float:
        """Scale the alass budget with payload size, never below the configured floor."""
        largest = max(len(target_srt or ""), len(reference_srt or ""))
        tier = _TIMEOUT_TIERS[-1][1]
        for threshold, seconds in _TIMEOUT_TIERS:
            if largest >= threshold:
                tier = seconds
                break
        return max(self.timeout, tier)

    def is_available(self) -> bool:
        """True when the ``alass`` binary can be resolved/executed."""
        try:
            probe = subprocess.run(
                [self.alass_path, "--version"],
                capture_output=True,
                timeout=self.timeout,
            )
            return probe.returncode in (0, 1)  # alass may exit 1 for --version
        except (OSError, subprocess.SubprocessError):
            return False

    # Maximum plausible shift for a non-exact (edition) reference. Hash,
    # embedded, and team references identify the same cut, so large shifts
    # are impossible; an edition reference is only timing-adjacent, and a
    # shift beyond this bound means alass aligned the wrong content.
    MAX_ALLOWED_EDITION_SHIFT = 6.0
    # Broadcast/WEB airings carry recap/intro segments (typically 30-120s)
    # that retail BluRay editions omit. A large UNIFORM shift on series
    # content with a bilaterally confirmed BluRay source is that recap cut,
    # not a sync collapse, so the ceiling is raised for exactly that case.
    MAX_ALLOWED_RECAP_SHIFT = 120.0
    # A *uniform* constant offset (e.g. a longer studio logo/bumper) may exceed
    # the strict edition ceiling up to this bound; variable/split shifts stay
    # on the strict ceiling.
    MAX_ALLOWED_UNIFORM_SHIFT = 30.0

    def sync(
        self,
        target_srt: str,
        reference_srt: str,
        decision_kind: str = "edition",
        is_series: bool = False,
        source_confirmed: bool = False,
        relaxed: bool = False,
        reference_partial: bool = False,
    ) -> str | None:
        """
        Return the synced target SRT, or ``None`` on any failure.

        Runs ``alass <reference> <target> <output> --split-penalty 0.5`` inside
        secure temporary files with a strict subprocess timeout.

        ``decision_kind`` (``hash`` / ``embedded`` / ``team`` / ``edition``)
        selects the safety guardrail: for a non-exact ``edition`` reference a
        first-cue shift beyond ``MAX_ALLOWED_EDITION_SHIFT`` discards the
        output so a mis-aligned sync is never served — except for series
        content with a confirmed BluRay source pair, where recap cuts of up
        to ``MAX_ALLOWED_RECAP_SHIFT`` are legitimate.
        """
        if not target_srt or not reference_srt:
            logger.info(
                "[sync] skipped: empty input (target=%d, reference=%d)",
                len(target_srt or ""),
                len(reference_srt or ""),
            )
            return None

        target_srt = sanitize_subtitle(target_srt)
        reference = sanitize_subtitle(strip_sdh(reference_srt))

        # Pre-alass validation: never spawn the subprocess for degenerate input.
        target_cues = _valid_cue_count(target_srt)
        reference_cues = _valid_cue_count(reference)
        if target_cues < _MIN_VALID_CUES or reference_cues < _MIN_VALID_CUES:
            logger.warning(
                "[SRT_VALIDATION_ERROR] insufficient valid cues "
                "(target=%d, reference=%d, min=%d) -> fallback",
                target_cues,
                reference_cues,
                _MIN_VALID_CUES,
            )
            return None

        # Duration + timeline compatibility gate: files whose runtimes or cue
        # populations diverge cannot be the same cut (season pack vs episode,
        # wrong episode, extended vs broadcast, trailer, ...). Percentile
        # durations ignore trailing credit/outro cues; edition references are
        # additionally checked for cue-count and season-pack structure.
        rejection = _timeline_rejection(
            target_srt,
            reference,
            decision_kind=decision_kind,
            relaxed=relaxed,
            source_confirmed=source_confirmed,
            reference_partial=reference_partial,
        )
        if rejection is not None:
            logger.warning(
                "[sync] %s -> aborting alass, serving original", rejection
            )
            return None

        effective_timeout = self._effective_timeout(target_srt, reference)
        logger.info(
            "[sync] starting alass: target=%d chars, reference=%d chars "
            "(sanitized), timeout=%.1fs",
            len(target_srt),
            len(reference),
            effective_timeout,
        )
        # Raw first-cue offset before alignment: makes a wrong-cut reference
        # obvious (a seconds-level gap is an intro offset; minutes is a cut).
        target_first = _cue_starts_ms(target_srt, limit=1)
        reference_first = _cue_starts_ms(reference, limit=1)
        logger.info(
            "[sync] first cue before alass: target=%s reference=%s",
            f"{target_first[0] / 1000.0:.2f}s" if target_first else "n/a",
            f"{reference_first[0] / 1000.0:.2f}s" if reference_first else "n/a",
        )

        ref_file = tgt_file = out_file = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".srt", delete=False, encoding="utf-8"
            ) as ref_handle:
                ref_handle.write(reference)
                ref_file = ref_handle.name
            with tempfile.NamedTemporaryFile(
                "w", suffix=".srt", delete=False, encoding="utf-8"
            ) as tgt_handle:
                tgt_handle.write(target_srt)
                tgt_file = tgt_handle.name

            out_file = tgt_file + ".synced.srt"

            # alass syntax: alass <REFERENCE_FILE> <TARGET_TO_FIX> <OUTPUT_FILE> [options]
            # argv[1] = trusted reference (English), argv[2] = target (Arabic).
            command = [
                self.alass_path,
                ref_file,
                tgt_file,
                out_file,
                # Upstream alass default for movies: a small penalty fragments
                # the timeline into unnecessary splits.
                "--split-penalty",
                "7.0",
            ]
            logger.info("[sync] executing: %s", " ".join(command))
            started = time.monotonic()
            result = subprocess.run(
                command,
                capture_output=True,
                timeout=effective_timeout,
            )
            elapsed = time.monotonic() - started
            logger.info(
                "[sync] alass finished in %.2fs with exit code %s",
                elapsed,
                result.returncode,
            )
            if result.returncode != 0:
                stderr = (result.stderr or b"").decode("utf-8", "replace").strip()
                stdout = (result.stdout or b"").decode("utf-8", "replace").strip()
                logger.warning(
                    "[sync] alass exited with code %s | stderr=%r | stdout=%r -> fallback",
                    result.returncode,
                    stderr[:500],
                    stdout[:500],
                )
                return None

            if not os.path.exists(out_file) or os.path.getsize(out_file) == 0:
                logger.warning("[sync] alass produced an empty output file -> fallback")
                return None

            with open(out_file, encoding="utf-8") as handle:
                synced = handle.read()
            if not synced:
                logger.warning("[sync] alass output was empty text -> fallback")
                return None
            rejection = _validate_synced_output(target_srt, synced)
            if rejection is not None:
                logger.warning("[sync] synced output rejected: %s -> fallback", rejection)
                return None
            logger.info("[sync] success: produced %d chars of synced subtitle", len(synced))

            # Prove whether alass actually shifted the timecodes.
            target_starts = _cue_starts_ms(target_srt, limit=target_cues)
            synced_starts = _cue_starts_ms(synced, limit=target_cues)
            shifts = [
                (after - before) / 1000.0
                for before, after in zip(target_starts, synced_starts, strict=False)
            ]
            if shifts:
                peak = max(shifts, key=abs)
                spread = _uniform_shift_spread(shifts)
                logger.info(
                    "[sync] applied shift - sampled %d cue shift(s): peak %+.2fs, "
                    "spread %.2fs",
                    len(shifts),
                    peak,
                    spread,
                )
                if decision_kind == "edition" or reference_partial:
                    if is_series and source_confirmed:
                        # Broadcast/retail recap cut: allow a large uniform shift.
                        if abs(peak) > self.MAX_ALLOWED_RECAP_SHIFT:
                            logger.warning(
                                "[sync] safety guardrail triggered: recap shift "
                                "(%+.2fs, spread %.2fs) exceeds the recap ceiling; "
                                "rejecting sync to prevent mis-timing",
                                peak,
                                spread,
                            )
                            return None
                    elif (
                        spread <= _UNIFORM_SHIFT_SPREAD_SECONDS
                        and abs(peak) <= self.MAX_ALLOWED_UNIFORM_SHIFT
                    ):
                        # Constant intro/logo/bumper offset: no drift, safe.
                        logger.info(
                            "[sync] uniform constant offset %+.2fs accepted "
                            "(intro/bumper difference, spread %.2fs, no drift)",
                            peak,
                            spread,
                        )
                    elif abs(peak) > self.MAX_ALLOWED_EDITION_SHIFT:
                        logger.warning(
                            "[sync] safety guardrail triggered: variable shift "
                            "(%+.2fs, spread %.2fs) exceeds max safe threshold for "
                            "generic edition; rejecting sync to prevent mis-timing "
                            "(reference likely a different cut of the video)",
                            peak,
                            spread,
                        )
                        return None
            else:
                logger.info("[sync] applied shift: no cues parsed for comparison")
            return synced
        except subprocess.TimeoutExpired:
            logger.warning("[sync] alass timed out after %.1fs -> fallback", effective_timeout)
            return None
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[sync] alass sync failed (%s) -> fallback", exc)
            return None
        finally:
            for path in (ref_file, tgt_file, out_file):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:  # pragma: no cover - best effort cleanup
                        pass

    async def sync_async(
        self,
        target_srt: str,
        reference_srt: str,
        decision_kind: str = "edition",
        is_series: bool = False,
        source_confirmed: bool = False,
        relaxed: bool = False,
        reference_partial: bool = False,
    ) -> str | None:
        """Non-blocking wrapper around :meth:`sync` for use in async routes."""
        async with self._semaphore:
            worker = asyncio.create_task(asyncio.to_thread(
                self.sync, target_srt, reference_srt, decision_kind,
                is_series, source_confirmed, relaxed, reference_partial,
            ))
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                # to_thread cannot stop a running process. Hold its slot until
                # subprocess.run completes or kills it at the configured timeout.
                try:
                    await worker
                finally:
                    raise
