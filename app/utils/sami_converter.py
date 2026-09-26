"""SAMI (.smi) to SubRip (.srt) conversion.

SAMI (Synchronized Accessible Media Interchange) stores cues as
``<SYNC Start={milliseconds}>`` blocks with HTML-ish ``<P>`` text, often
carrying several language classes and ``&nbsp;`` filler cues. This converter
extracts the English track when labelled, drops empty cues, and emits strict
SRT so downstream sync/validation can consume the payload.
"""

from __future__ import annotations

import html
import re

_SYNC_REGEX = re.compile(r"<\s*sync\b[^>]*>", re.IGNORECASE)
_START_REGEX = re.compile(r"start\s*=\s*(\d+)", re.IGNORECASE)
_PARAGRAPH_REGEX = re.compile(
    r"<\s*p\b([^>]*)>(.*?)(?=<\s*p\b|<\s*sync\b|$)", re.IGNORECASE | re.DOTALL
)
_CLASS_REGEX = re.compile(r"class\s*=\s*[\"']?([^\"'\s>]+)", re.IGNORECASE)
_TAG_REGEX = re.compile(r"<[^>]+>")
_WS_REGEX = re.compile(r"[ \t]+")

# A trailing cue with no following SYNC gets this tail duration.
_LAST_CUE_MS = 2000


def _format_timestamp(milliseconds: int) -> str:
    """Format milliseconds as strict ``HH:MM:SS,mmm``."""
    total = max(0, int(milliseconds))
    hours, remainder = divmod(total, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _clean_text(fragment: str) -> str:
    """Strip tags/entities from a SAMI paragraph, preserving line breaks."""
    fragment = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", fragment)
    fragment = _TAG_REGEX.sub("", fragment)
    fragment = html.unescape(fragment).replace("\xa0", " ")
    lines = [_WS_REGEX.sub(" ", line).strip() for line in fragment.split("\n")]
    return "\n".join(line for line in lines if line)


def _paragraph_texts(sync_body: str) -> list[tuple[str, str]]:
    """Split a SYNC block into ``(class, text)`` paragraphs, dropping empties."""
    paragraphs: list[tuple[str, str]] = []
    for match in _PARAGRAPH_REGEX.finditer(sync_body):
        class_match = _CLASS_REGEX.search(match.group(1) or "")
        text = _clean_text(match.group(2) or "")
        if text:
            paragraphs.append(((class_match.group(1) if class_match else ""), text))
    return paragraphs


def convert_sami_to_srt(content: str) -> str:
    """Convert SAMI subtitle text to SubRip; returns ``""`` when no cues survive."""
    if not content or "<sync" not in content.lower():
        return ""
    text = content.replace("\r\n", "\n").replace("\r", "\n")

    boundaries: list[tuple[int, int]] = []
    for match in _SYNC_REGEX.finditer(text):
        start_match = _START_REGEX.search(match.group(0))
        if start_match:
            boundaries.append((match.start(), int(start_match.group(1))))
    if not boundaries:
        return ""

    cues: list[str] = []
    for index, (tag_start, start_ms) in enumerate(boundaries):
        body_end = boundaries[index + 1][0] if index + 1 < len(boundaries) else len(text)
        next_start = boundaries[index + 1][1] if index + 1 < len(boundaries) else None
        paragraphs = _paragraph_texts(text[tag_start:body_end])
        if not paragraphs:
            continue
        # Prefer English-labelled paragraphs when the track carries several
        # languages; otherwise keep everything that survived cleaning.
        english_classes = [text for cls, text in paragraphs if "en" in cls.lower()]
        cue_lines = english_classes or [text for _, text in paragraphs]
        end_ms = next_start if next_start is not None else start_ms + _LAST_CUE_MS
        if end_ms <= start_ms:
            continue
        cues.append(
            f"{len(cues) + 1}\n{_format_timestamp(start_ms)} --> {_format_timestamp(end_ms)}\n"
            + "\n".join(cue_lines)
        )
    return "\n\n".join(cues) + ("\n" if cues else "")
