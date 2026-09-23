"""ASS/SSA -> SubRip (.srt) converter with primary color preservation.

Many players (ExoPlayer on Android TV/Stremio, Nuvio) fail to render native
``.ass``/``.ssa`` subtitles, exposing raw override tags (``{\\pos(..)}``,
``{\\an8}``) and breaking Arabic BiDi alignment. This module converts those
files into standard SRT, translating ASS BGR color overrides
(``{\\c&HBBGGRR&}`` / ``{\\1c&H...&}``) into HTML ``<font color="#RRGGBB">``
tags while stripping all positioning/animation/formatting commands.
"""

from __future__ import annotations

import re

from app.utils.rtl import _append_trailing_rlm, fix_rtl_punctuation

# ASS timestamps: H:MM:SS.cc (centiseconds) or H:MM:SS,mmm.
_ASS_TIMESTAMP_REGEX = re.compile(r"^\s*(\d+):(\d{1,2}):(\d{1,2})[.,:](\d{1,3})\s*$")
# Any override block: {\...}
_ASS_OVERRIDE_REGEX = re.compile(r"\{([^}]*)\}")
# Primary color command: \c&H...& or \1c&H...&
_ASS_PRIMARY_COLOR_REGEX = re.compile(r"\\(?:1)?c\s*&H([0-9A-Fa-f]{1,8})&?", re.IGNORECASE)
# Style reset: \r (optionally \rStyleName)
_ASS_RESET_REGEX = re.compile(r"\\r(?:[^\\}]*)?", re.IGNORECASE)
# Vector-drawing mode: \p1 .. \p9
_ASS_DRAWING_REGEX = re.compile(r"\\p\s*[1-9]", re.IGNORECASE)

_FONT_TAG_REGEX = re.compile(r"</?font\b[^>]*>", re.IGNORECASE)


def _parse_ass_timestamp(value: str) -> int | None:
    """Return milliseconds for an ASS timestamp (centiseconds -> milliseconds)."""
    match = _ASS_TIMESTAMP_REGEX.match(value)
    if not match:
        return None
    hours, minutes, seconds, fraction = match.groups()
    # Fractional digits are a decimal fraction of a second (ASS uses centiseconds).
    fraction_ms = round(int(fraction) / (10 ** len(fraction)) * 1000)
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


def _ass_text_to_html(text: str, default_rgb: str | None) -> str:
    """Translate ASS dialogue text (with overrides) into color-preserving HTML."""
    text = text.replace("\\N", "\n").replace("\\n", "\n").replace("\\h", " ")
    # A non-white style primary color is applied as the base color.
    base_rgb = default_rgb if default_rgb and default_rgb.upper() != "#FFFFFF" else None

    pieces: list[str] = []
    font_open = False

    def close_font() -> None:
        nonlocal font_open
        if font_open:
            pieces.append("</font>")
            font_open = False

    def open_font(rgb: str | None) -> None:
        nonlocal font_open
        if rgb:
            pieces.append(f'<font color="{rgb}">')
            font_open = True

    open_font(base_rgb)

    cursor = 0
    for match in _ASS_OVERRIDE_REGEX.finditer(text):
        pieces.append(text[cursor : match.start()])
        block = match.group(1)

        color_match = _ASS_PRIMARY_COLOR_REGEX.search(block)
        if color_match:
            close_font()
            open_font(_ass_color_to_rgb(color_match.group(1)))
        elif _ASS_RESET_REGEX.search(block):
            close_font()
            open_font(base_rgb)
        # Every other command (\pos, \an, \move, \fade, \clip, \fs, \fn, \b, \i,
        # \p, ...) is simply dropped.
        cursor = match.end()

    pieces.append(text[cursor:])
    close_font()
    return "".join(pieces)


def _shield_font_tags(text: str) -> tuple[str, dict[str, str]]:
    tokens: dict[str, str] = {}

    def _replace(match: re.Match[str]) -> str:
        token = f"\x00FT{len(tokens)}\x00"
        tokens[token] = match.group(0)
        return token

    return _FONT_TAG_REGEX.sub(_replace, text), tokens


def _apply_rtl(text: str) -> str:
    """Run the RTL pipeline while shielding ``<font>`` tags from corruption."""
    shielded, tokens = _shield_font_tags(text)
    shielded = fix_rtl_punctuation(shielded)
    for token, tag in tokens.items():
        shielded = shielded.replace(token, tag)
    # The RLM cannot be appended while the closing tag was shielded, so anchor
    # trailing neutral punctuation now that the tags are restored.
    return "\n".join(_append_trailing_rlm(line) for line in shielded.split("\n"))


def convert_ass_to_srt(ass_content: str, apply_rtl: bool = True) -> str:
    """
    Convert ASS/SSA subtitle content into SubRip (.srt).

    - Parses ``[Events]`` ``Dialogue:`` lines and converts ``H:MM:SS.cc``
      timestamps into ``HH:MM:SS,mmm``.
    - Preserves primary text colors as ``<font color="#RRGGBB">`` (ASS uses
      BGR ordering), resolving style defaults and ``\\c``/``\\1c`` overrides.
    - Strips all remaining layout/animation tags, converts ``\\N``/``\\n``/``\\h``,
      and drops vector-drawing cues.
    - Optionally applies the Arabic RTL normalization pipeline.
    """
    if not ass_content or not re.search(r"(?im)^\s*dialogue\s*:", ass_content):
        return ass_content

    style_colors = _parse_style_colors(ass_content)

    in_events = False
    fields: list[str] | None = None
    blocks: list[str] = []

    for raw_line in ass_content.splitlines():
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

        body = line.split(":", 1)[1]
        field_names = fields or [
            "layer", "start", "end", "style", "name",
            "marginl", "marginr", "marginv", "effect", "text",
        ]
        parts = body.split(",", len(field_names) - 1)
        if len(parts) < len(field_names):
            continue
        data = dict(zip(field_names, parts, strict=False))

        start_ms = _parse_ass_timestamp(data.get("start", ""))
        end_ms = _parse_ass_timestamp(data.get("end", ""))
        if start_ms is None or end_ms is None:
            continue

        raw_text = data.get("text", "")
        # Vector-drawing cues carry no spoken dialogue -> drop entirely.
        if _ASS_DRAWING_REGEX.search(raw_text):
            continue

        style = data.get("style", "").strip()
        html = _ass_text_to_html(raw_text, style_colors.get(style))
        if apply_rtl:
            html = _apply_rtl(html)
        html = html.strip("\n").strip()
        if not html:
            continue

        index = len(blocks) + 1
        timestamp = f"{_format_srt_timestamp(start_ms)} --> {_format_srt_timestamp(end_ms)}"
        blocks.append(f"{index}\n{timestamp}\n{html}")

    if not blocks:
        return ""

    return "\n\n".join(blocks) + "\n"


def convert_ass_to_srt_bytes(content: bytes, apply_rtl: bool = True) -> bytes:
    """UTF-8 byte wrapper around :func:`convert_ass_to_srt`."""
    if not content:
        return content
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return content
    return convert_ass_to_srt(text, apply_rtl=apply_rtl).encode("utf-8")
