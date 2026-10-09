"""ASS/SSA -> SubRip (.srt) converter with primary color preservation.

Many players (ExoPlayer on Android TV/Stremio, Nuvio) fail to render native
``.ass``/``.ssa`` subtitles, exposing raw override tags (``{\\pos(..)}``,
``{\\an8}``) and breaking Arabic BiDi alignment. This module converts those
files into standard SRT, translating ASS BGR color overrides
(``{\\c&HBBGGRR&}`` / ``{\\1c&H...&}``) into HTML ``<font color="#RRGGBB">``
tags while stripping all positioning/animation/formatting commands.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field

from app.utils.rtl import fix_rtl_punctuation

logger = logging.getLogger(__name__)

# ASS timestamps: H:MM:SS.cc (centiseconds) or H:MM:SS,mmm.
_ASS_TIMESTAMP_REGEX = re.compile(r"^\s*(\d+):(\d{1,2}):(\d{1,2})[.,:](\d{1,3})\s*$")
# Primary color command: \c&H...& or \1c&H...&
_ASS_PRIMARY_COLOR_REGEX = re.compile(r"\\(?:1)?c\s*&H([0-9A-Fa-f]{1,8})&?", re.IGNORECASE)
# Style reset: \r (optionally \rStyleName)
_ASS_RESET_REGEX = re.compile(r"\\r(?:[^\\}]*)?", re.IGNORECASE)


def _parse_ass_timestamp(value: str) -> int | None:
    """Return milliseconds for an ASS timestamp (centiseconds -> milliseconds)."""
    match = _ASS_TIMESTAMP_REGEX.match(value)
    if not match:
        return None
    hours, minutes, seconds, fraction = match.groups()
    # Fractional digits are a decimal fraction of a second (ASS uses centiseconds).
    if int(minutes) > 59 or int(seconds) > 59 or len(hours) > 1000:
        return None
    fraction_ms = int(fraction.ljust(3, "0"))
    return (int(hours) * 3600 + int(minutes) * 60 + int(seconds)) * 1000 + fraction_ms


def _format_srt_timestamp(milliseconds: int) -> str:
    milliseconds = max(0, milliseconds)
    hours = milliseconds // 3_600_000
    minutes = (milliseconds % 3_600_000) // 60_000
    seconds = (milliseconds % 60_000) // 1000
    millis = milliseconds % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _ass_color_to_rgb(value: str) -> str | None:
    """Convert an ASS ``&HAABBGGRR&`` (or ``&HBBGGRR&``) color into ``#RRGGBB``."""
    raw = (value or "").strip().upper().removeprefix("&H").rstrip("&")
    if not raw or not re.fullmatch(r"[0-9A-F]{1,8}", raw):
        return None
    raw = raw.zfill(8)  # AABBGGRR
    _alpha, blue, green, red = raw[0:2], raw[2:4], raw[4:6], raw[6:8]
    return f"#{red}{green}{blue}"


def _parse_style_colors(content: str) -> dict[str, str]:
    """Map ``[V4(+) Styles]`` style names to their primary color as ``#RRGGBB``."""
    styles: dict[str, str] = {}
    in_styles = False
    fields: list[str] | None = None

    for raw_line in content.splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if lowered.startswith("[") and lowered.endswith("]"):
            in_styles = lowered in ("[v4+ styles]", "[v4 styles]", "[v4++ styles]")
            fields = None
            continue
        if not in_styles:
            continue
        if lowered.startswith("format:"):
            fields = [part.strip().lower() for part in line.split(":", 1)[1].split(",")]
        elif lowered.startswith("style:") and fields:
            parts = line.split(":", 1)[1].split(",", len(fields) - 1)
            data = dict(zip(fields, parts, strict=False))
            name = data.get("name", "").strip()
            rgb = _ass_color_to_rgb(data.get("primarycolour", ""))
            if name and rgb:
                styles[name] = rgb
    return styles


@dataclass
class ConversionStats:
    events_seen: int = 0
    cues_emitted: int = 0
    invalid_skipped: int = 0
    drawing_only: int = 0
    mixed_drawing: int = 0
    malformed_overrides: int = 0
    empty_visible: int = 0


@dataclass
class ConversionResult:
    data: str
    converted: bool
    reason: str
    stats: ConversionStats = field(default_factory=ConversionStats)


_DEFAULT_FIELDS = ["layer", "start", "end", "style", "name",
                   "marginl", "marginr", "marginv", "effect", "text"]
_SRT_TIME_LINE = re.compile(r"^[ \t]*(?:\d+:)?\d{1,2}:\d{2}[,.]\d+[ \t]*-->", re.MULTILINE)
_VISIBLE_WRAPPERS = re.compile(r"</?(?:font\b[^<>]*|[ibu])>", re.IGNORECASE)
_DRAWING_COMMAND = re.compile(r"\\p\s*(\d+)(?![\w.])", re.IGNORECASE)


def is_ass_text(content: str) -> bool:
    """Require section/event structure; a SubRip timespan takes precedence."""
    if _SRT_TIME_LINE.search(content):
        return False
    in_events = False
    header = False
    for raw in content.lstrip("\ufeff").splitlines():
        line = raw.strip().lower()
        if line in {"[script info]", "[v4 styles]", "[v4+ styles]", "[v4++ styles]"}:
            header = True
        if line.startswith("[") and line.endswith("]"):
            in_events = line == "[events]"
        elif in_events and line.startswith("format:"):
            names = [part.strip() for part in line.split(":", 1)[1].split(",")]
            if {"start", "end", "text"}.issubset(names) and names[-1] == "text":
                return True
        elif in_events and line.startswith(("dialogue:", "comment:")):
            parts = line.split(":", 1)[1].split(",", 9)
            if len(parts) == 10 and _parse_ass_timestamp(parts[1]) is not None:
                return True
    return header


def _override_segments(text: str, stats: ConversionStats) -> Iterator[tuple[bool, str]]:
    """Single forward scan. Only braces beginning with a command are overrides.

    Nested/unclosed command blocks are discarded as ambiguous syntax, without
    interpreting state from them. Unclosed blocks consume the remaining span;
    preceding readable text is retained. Literal braces remain literal.
    """
    cursor = 0
    literal_start = 0
    while cursor < len(text):
        if text[cursor] != "{":
            cursor += 1
            continue
        command_start = cursor + 1
        while command_start < len(text) and text[command_start].isspace():
            command_start += 1
        if command_start >= len(text) or text[command_start] != "\\":
            cursor += 1
            continue
        yield False, text[literal_start:cursor]
        start = cursor
        depth = 1
        nested = False
        cursor = command_start + 1
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
                nested = True
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth or nested:
            stats.malformed_overrides += 1
        else:
            yield True, text[start + 1:cursor - 1]
        literal_start = cursor
    yield False, text[literal_start:]


def _top_level_commands(block: str) -> Iterator[str]:
    """Do not execute state changes inside transform/other command parameters."""
    start: int | None = None
    depth = 0
    for index, char in enumerate(block):
        if char == "\\" and depth == 0:
            if start is not None:
                yield block[start:index]
            start = index
        elif char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
    if start is not None:
        yield block[start:]


def _has_visible_text(html: str) -> bool:
    return any(not char.isspace() and unicodedata.category(char) != "Cf"
               for char in _VISIBLE_WRAPPERS.sub("", html))


def _ass_text_to_html(text: str, default_rgb: str | None,
                      stats: ConversionStats | None = None) -> str:
    """Discard presentation syntax/drawing spans, preserving literal content."""
    stats = stats if stats is not None else ConversionStats()
    base_rgb = default_rgb if default_rgb and default_rgb.upper() != "#FFFFFF" else None
    rgb = base_rgb
    open_rgb: str | None = None
    pieces: list[str] = []
    drawing = False
    had_drawing = False
    for command, segment in _override_segments(text, stats):
        if command:
            # Preserve the existing simple first-color/reset policy; transforms
            # are presentation instructions, not immediate drawing/color state.
            commands = list(_top_level_commands(segment))
            for instruction in commands:
                mode = _DRAWING_COMMAND.match(instruction)
                if mode:
                    drawing = bool(mode.group(1).strip("0"))
                    had_drawing |= drawing
            color = next((match for instruction in commands
                          if (match := _ASS_PRIMARY_COLOR_REGEX.match(instruction))), None)
            if color is not None:
                rgb = _ass_color_to_rgb(color.group(1)) or base_rgb
            elif any(_ASS_RESET_REGEX.match(instruction) for instruction in commands):
                rgb = base_rgb
            continue
        if drawing or not segment:
            continue
        if open_rgb != rgb:
            if open_rgb:
                pieces.append("</font>")
            if rgb:
                pieces.append(f'<font color="{rgb}">')
            open_rgb = rgb
        pieces.append(segment.replace("\\N", "\n").replace("\\n", "\n").replace("\\h", " "))
    if open_rgb:
        pieces.append("</font>")
    output = "".join(pieces)
    # Blank lines inside an SRT cue are block delimiters, never text separators.
    output = "\n".join(line for line in output.splitlines() if line.strip())
    visible = _has_visible_text(output)
    if had_drawing:
        if visible:
            stats.mixed_drawing += 1
        else:
            stats.drawing_only += 1
    return output


def _apply_rtl(text: str) -> str:
    """Use the shared RTL pipeline, which understands generated font wrappers."""
    return fix_rtl_punctuation(text)


def convert_ass_result(ass_content: str, apply_rtl: bool = True) -> ConversionResult:
    """Explicit conversion contract: unsuccessful conversion preserves input."""
    stats = ConversionStats()
    if not is_ass_text(ass_content):
        return ConversionResult(ass_content, False, "not_ass", stats)
    styles = _parse_style_colors(ass_content.lstrip("\ufeff"))
    in_events = False
    fields: list[str] | None = None
    blocks: list[str] = []
    for raw_line in ass_content.lstrip("\ufeff").splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if lowered.startswith("[") and lowered.endswith("]"):
            in_events = lowered == "[events]"
            fields = None
            continue
        if not in_events:
            continue
        if lowered.startswith("format:"):
            fields = [part.strip().lower() for part in line.split(":", 1)[1].split(",")]
            continue
        if not lowered.startswith("dialogue:"):
            continue
        stats.events_seen += 1
        names = fields if fields is not None else _DEFAULT_FIELDS
        if not {"start", "end", "text"}.issubset(names) or names[-1] != "text" or len(set(names)) != len(names):
            stats.invalid_skipped += 1
            continue
        parts = line.split(":", 1)[1].split(",", len(names) - 1)
        if len(parts) != len(names):
            stats.invalid_skipped += 1
            continue
        data = dict(zip(names, parts, strict=True))
        start = _parse_ass_timestamp(data["start"])
        end = _parse_ass_timestamp(data["end"])
        if start is None or end is None or end <= start:
            stats.invalid_skipped += 1
            continue
        drawing_only_before = stats.drawing_only
        html = _ass_text_to_html(data["text"], styles.get(data.get("style", "").strip()), stats)
        if not _has_visible_text(html):
            if stats.drawing_only == drawing_only_before:
                stats.empty_visible += 1
            continue
        if apply_rtl:
            html = _apply_rtl(html)
        timestamp = f"{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}"
        blocks.append(f"{len(blocks) + 1}\n{timestamp}\n{html.strip()}")
    stats.cues_emitted = len(blocks)
    reason = "converted" if blocks else "no_readable_cues"
    logger.debug("[ass-conversion] format=ass status=%s stats=%s", reason, stats)
    return ConversionResult("\n\n".join(blocks) + "\n" if blocks else ass_content,
                            bool(blocks), reason, stats)


def convert_ass_to_srt(ass_content: str, apply_rtl: bool = True) -> str:
    return convert_ass_result(ass_content, apply_rtl).data


def convert_ass_to_srt_bytes(content: bytes, apply_rtl: bool = True) -> bytes:
    """UTF-8 wrapper; failed conversion preserves the exact original bytes."""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return content
    result = convert_ass_result(text, apply_rtl)
    return result.data.encode("utf-8") if result.converted else content
