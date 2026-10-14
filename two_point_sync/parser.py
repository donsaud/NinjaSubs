"""SRT subtitle parsing with timestamp preservation.

Supports SRT fully. Timestamps are stored as integer milliseconds to avoid
accumulating floating-point errors. Cue text (including any inline formatting
tags) is preserved exactly as decoded; only timestamp fields are ever
rewritten by the synchronisation pipeline.

Extension boundary for ASS/SSA: add ``parse_ass``/``parse_ssa`` constructors
that return the same :class:`Cue` model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TIMESTAMP_RE = re.compile(r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})[,.](?P<ms>\d{3})")

TIMING_LINE_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{3})"
)


@dataclass(frozen=True)
class Cue:
    """A single subtitle cue with preserved identity and text."""

    number: int
    order: int
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class CueIssue:
    """A non-fatal structural observation about parsed cues."""

    kind: str
    message: str
    cue_number: int | None = None
    order: int | None = None


class SubtitleParseError(ValueError):
    """Raised when subtitle input cannot be parsed into cues."""


def parse_timestamp(value: str) -> int:
    """Parse ``HH:MM:SS,mmm`` (or ``HH:MM:SS.mmm``) into integer milliseconds."""
    match = TIMESTAMP_RE.fullmatch(value.strip())
    if match is None:
        raise SubtitleParseError(f"Invalid timestamp: {value!r}")
    hours = int(match.group("h"))
    minutes = int(match.group("m"))
    seconds = int(match.group("s"))
    millis = int(match.group("ms"))
    if minutes >= 60 or seconds >= 60:
        raise SubtitleParseError(f"Timestamp out of range: {value!r}")
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def format_timestamp(total_ms: int) -> str:
    """Format integer milliseconds as ``HH:MM:SS,mmm`` (deterministic)."""
    if total_ms < 0:
        raise SubtitleParseError(f"Cannot format negative timestamp: {total_ms}")
    millis = total_ms % 1000
    total_seconds = total_ms // 1000
    seconds = total_seconds % 60
    total_minutes = total_seconds // 60
    minutes = total_minutes % 60
    hours = total_minutes // 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


_RESIDUE_RE = re.compile(r"[A-Za-z0-9\u0600-\u06FF\u0660-\u0669\u06F0-\u06F9]")


def _is_formatting_residue(block: str) -> bool:
    """Return True for blocks with no textual payload (stray bidi/tags)."""
    cleaned = block.strip().lstrip("\ufeff")
    cleaned = re.sub(r"\{[^}]*\}|<[^>]*>", "", cleaned)
    cleaned = re.sub(r"[‎‏‪‫‬‭‮⁦⁧⁨⁩﻿⁠\s]", "", cleaned)
    return cleaned == "" or _RESIDUE_RE.search(cleaned) is None


def _split_blocks(content: str) -> list[str]:
    normalised = content.replace("\r\n", "\n").replace("\r", "\n")
    raw_blocks = re.split(r"\n\s*\n", normalised.strip("\n"))
    return [block for block in raw_blocks if block.strip() != ""]


def parse_srt(content: str) -> list[Cue]:
    """Parse SRT ``content`` into an ordered list of :class:`Cue`.

    Raises :class:`SubtitleParseError` when no valid cue can be recovered or
    when a block is structurally malformed (bad index, bad timing line,
    invalid timestamps, or empty cue text). Nothing is silently dropped:
    every returned cue preserves its original number, order and text.
    """
    if content is None or content.strip() == "":
        raise SubtitleParseError("Empty subtitle content")
    # Tolerate callers that read the file without utf-8-sig.
    content = content.lstrip("\ufeff")
    blocks = _split_blocks(content)
    cues: list[Cue] = []
    for position, block in enumerate(blocks):
        lines = block.split("\n")
        if len(lines) < 2 and _is_formatting_residue(block):
            # Stray formatting residue (e.g. a trailing bidi mark with no
            # timing line) carries no dialogue and is not a cue.
            continue
        if len(lines) < 2:
            raise SubtitleParseError(f"Malformed block {position + 1}: {block[:80]!r}")
        timing_present = TIMING_LINE_RE.search(block) is not None
        if not timing_present and _is_formatting_residue(block):
            continue
        try:
            number = int(lines[0].strip())
        except ValueError:
            if _is_formatting_residue(block):
                continue
            raise SubtitleParseError(
                f"Malformed cue index in block {position + 1}: {lines[0][:40]!r}"
            ) from None
        timing = TIMING_LINE_RE.search(lines[1])
        if timing is None:
            raise SubtitleParseError(
                f"Malformed timing line in block {position + 1}: {lines[1][:80]!r}"
            )
        start_ms = parse_timestamp(timing.group("start"))
        end_ms = parse_timestamp(timing.group("end"))
        if end_ms <= start_ms:
            raise SubtitleParseError(
                f"Non-positive cue duration in block {position + 1} "
                f"(start={start_ms}, end={end_ms})"
            )
        text = "\n".join(lines[2:])
        if text.strip() == "":
            raise SubtitleParseError(f"Empty cue text in block {position + 1}")
        cues.append(
            Cue(number=number, order=position, start_ms=start_ms, end_ms=end_ms, text=text)
        )
    if not cues:
        raise SubtitleParseError("No subtitle cues found")
    return cues


def parse_srt_bytes(data: bytes) -> list[Cue]:
    """Decode raw subtitle bytes (BOM/encoding tolerant) and parse as SRT."""
    if not data:
        raise SubtitleParseError("Empty subtitle payload")
    text: str | None = None
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "windows-1256", "windows-1252"):
        try:
            text = data.decode(encoding)
            break
        except (UnicodeDecodeError, LookupError) as exc:
            errors.append(f"{encoding}: {exc}")
    if text is None:
        raise SubtitleParseError(f"Undecodable subtitle bytes ({'; '.join(errors)})")
    return parse_srt(text)


@dataclass(frozen=True)
class VerificationParseReport:
    """Accounting for verification-input leniency (see parse_srt_for_verification)."""

    excluded: tuple[tuple[int, str], ...] = ()
    ambiguous: tuple[int, ...] = ()
    renumbered: bool = False


def parse_srt_for_verification(
    content: str,
    *, exclude_ambiguous: bool = False,
) -> tuple[list[Cue] | None, VerificationParseReport]:
    """Parse SRT ``content`` for Unified verification measurement.

    Verification-input boundary ONLY: the serving/strict ``parse_srt`` is
    untouched. Differences from strict parsing, all fail-safe:

    * Cue-shaped blocks (number + timing) with no dialogue text, or with a
      non-positive duration, carry no correspondence evidence — the same
      blocks the ALASS sanitizer drops before aligning. They are excluded
      and recorded in ``report.excluded`` as ``(block_number, kind)`` with
      kind ``"empty_text"`` or ``"non_positive_duration"``. This is
      accounted exclusion, never confused with missing dialogue: genuine
      dialogue loss still changes cue counts/text and fails integrity.
    * A text-bearing block that cannot be placed (no timing line, bad cue
      index, malformed timing) is AMBIGUOUS: attaching it would invent cue
      boundaries and dropping it would discard dialogue. Returns
      ``(None, report)`` with the 1-based block numbers in
      ``report.ambiguous`` so the caller abstains explicitly.
    * Kept cues are densely renumbered (``renumbered=True``) so the number
      space matches the sanitized working copy ALASS aligns; order follows
      document position and timings/text are verbatim.

    Raises :class:`SubtitleParseError` for empty input or when no dialogue
    cue survives (parity with strict parsing: nothing to measure).
    """
    if content is None or content.strip() == "":
        raise SubtitleParseError("Empty subtitle content")
    content = content.lstrip("\ufeff")
    blocks = _split_blocks(content)
    cues: list[Cue] = []
    excluded: list[tuple[int, str]] = []
    ambiguous: list[int] = []
    for position, block in enumerate(blocks):
        lines = block.split("\n")
        if len(lines) < 2 and _is_formatting_residue(block):
            continue
        timing_present = TIMING_LINE_RE.search(block) is not None
        if not timing_present:
            if _is_formatting_residue(block):
                continue
            ambiguous.append(position + 1)
            continue
        try:
            int(lines[0].strip())
        except ValueError:
            if _is_formatting_residue(block):
                continue
            ambiguous.append(position + 1)
            continue
        timing = TIMING_LINE_RE.search(lines[1])
        if timing is None:
            ambiguous.append(position + 1)
            continue
        try:
            start_ms = parse_timestamp(timing.group("start"))
            end_ms = parse_timestamp(timing.group("end"))
        except SubtitleParseError:
            ambiguous.append(position + 1)
            continue
        text = "\n".join(lines[2:])
        if end_ms <= start_ms:
            excluded.append((position + 1, "non_positive_duration"))
            continue
        if text.strip() == "":
            excluded.append((position + 1, "empty_text"))
            continue
        cues.append(
            Cue(number=len(cues) + 1, order=position,
                start_ms=start_ms, end_ms=end_ms, text=text)
        )
    if ambiguous and not exclude_ambiguous:
        return None, VerificationParseReport(
            excluded=tuple(excluded), ambiguous=tuple(ambiguous),
            renumbered=True,
        )
    if not cues:
        raise SubtitleParseError("No subtitle cues found")
    return cues, VerificationParseReport(
        excluded=tuple(excluded), ambiguous=tuple(ambiguous), renumbered=True,
    )


def validate_cues(cues: list[Cue]) -> list[CueIssue]:
    """Report structural issues without removing or mutating any cue."""
    issues: list[CueIssue] = []
    seen_numbers: set[int] = set()
    seen_starts: dict[int, int] = {}
    previous: Cue | None = None
    for cue in cues:
        if cue.end_ms <= cue.start_ms:
            issues.append(
                CueIssue(
                    kind="negative_duration",
                    message=f"Cue {cue.number} has non-positive duration",
                    cue_number=cue.number,
                    order=cue.order,
                )
            )
        if cue.text.strip() == "":
            issues.append(
                CueIssue(
                    kind="empty_text",
                    message=f"Cue {cue.number} has empty text",
                    cue_number=cue.number,
                    order=cue.order,
                )
            )
        if cue.number in seen_numbers:
            issues.append(
                CueIssue(
                    kind="duplicate_number",
                    message=f"Duplicate cue number {cue.number}",
                    cue_number=cue.number,
                    order=cue.order,
                )
            )
        seen_numbers.add(cue.number)
        if cue.start_ms in seen_starts:
            issues.append(
                CueIssue(
                    kind="duplicate_timestamp",
                    message=(
                        f"Cue {cue.number} shares start time with cue "
                        f"{seen_starts[cue.start_ms]}"
                    ),
                    cue_number=cue.number,
                    order=cue.order,
                )
            )
        else:
            seen_starts[cue.start_ms] = cue.number
        if previous is not None and cue.start_ms < previous.end_ms:
            issues.append(
                CueIssue(
                    kind="overlapping_cues",
                    message=(
                        f"Cue {cue.number} starts before previous cue "
                        f"{previous.number} ends"
                    ),
                    cue_number=cue.number,
                    order=cue.order,
                )
            )
        previous = cue
    return issues


def format_srt(cues: list[Cue]) -> str:
    """Serialise cues back to SRT, preserving numbers, order and text."""
    blocks: list[str] = []
    for cue in cues:
        blocks.append(
            f"{cue.number}\n"
            f"{format_timestamp(cue.start_ms)} --> {format_timestamp(cue.end_ms)}\n"
            f"{cue.text}"
        )
    return "\n\n".join(blocks) + "\n"
