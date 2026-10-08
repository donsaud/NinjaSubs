"""Diagnostic projection of boundary parser semantics; no production decisions."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from app.services.subtitle_matcher import parse_srt_cues
from app.services.sync.ninja_points import render_srt
from app.services.sync.pal_two_point import _is_usable_dialogue
from app.services.sync.parser_fingerprint import semantic_sha

VERSION = "boundary-cues-v1"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def implementation_sha() -> str:
    return semantic_sha()


def project(text: str) -> dict[str, Any]:
    if len(text) > 500_000:
        raise ValueError("projection input limit")
    blocks = re.split(r"\n[ \t]*\n", text.replace("\r\n", "\n").replace("\r", "\n"))
    if len(blocks) > 4002:
        raise ValueError("projection cue limit")
    cues = []
    for block in blocks:
        # Refuse ambiguous blocks rather than invent an interpretation.
        if block.count("-->") > 1:
            raise ValueError("ambiguous timing block")
        parsed = parse_srt_cues(block)
        if "-->" in block and not parsed and block.split("-->", 1)[1].strip():
            # Empty timestamped cues are safely omitted by the production parser.
            lines = block.strip().splitlines()
            timing = next(i for i, line in enumerate(lines) if "-->" in line)
            if any(line.strip() for line in lines[timing + 1:]):
                raise ValueError("unparsed timing block")
        for start, end, text_body in parsed:
            if start < 0 or end <= start:
                raise ValueError("invalid projected timing")
            cues.append([start, end, _is_usable_dialogue(text_body)])
    if len(cues) > 4000 or any(a[0] > b[0] or a[1] > b[1]
                             for a, b in zip(cues, cues[1:], strict=False)):
        raise ValueError("unsafe projected order")
    return {"version": VERSION, "cues": cues}


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def render_projection(value: dict[str, Any]) -> str:
    if value.get("version") != VERSION or not isinstance(value.get("cues"), list):
        raise ValueError("unsupported projection")
    cues = value["cues"]
    if len(cues) > 4000:
        raise ValueError("projection cue limit")
    for cue in cues:
        if (not isinstance(cue, list) or len(cue) != 3 or type(cue[0]) is not int
                or type(cue[1]) is not int or type(cue[2]) is not bool
                or cue[0] < 0 or cue[1] <= cue[0]):
            raise ValueError("invalid projection")
    if any(a[0] > b[0] or a[1] > b[1] for a, b in zip(cues, cues[1:], strict=False)):
        raise ValueError("unsafe projected order")
    return render_srt([(s, e, "Dialogue event." if usable else "Translated by: uploader")
                       for s, e, usable in cues])
