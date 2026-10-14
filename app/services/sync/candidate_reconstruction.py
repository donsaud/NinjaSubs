"""Timing-only candidate reconstruction across the ALASS sanitization boundary.

Live regression (Dexter S08E01 PiR8): ``SubtitleSyncService.sync`` aligns a
*sanitized* working copy of the target (advertisement lines removed, blocks
re-indexed), while Unified Verifier judges the ALASS output against the *raw*
original target. The single sanitization-explained difference (cue 921 lost
its ``facebook.com`` promo line; the translator credit was kept) tripped the
``dialogue_text_modified:1`` integrity rejection even though ALASS preserved
every working-copy cue byte-for-byte and timing evidence was strong.

The invariant is: Auto-Sync may alter timestamps, never user-facing dialogue.
The correction therefore transplants ONLY verified ALASS timings onto the
original user-facing cues, and only when an unambiguous 1:1 cue mapping plus
a proven sanitized-input/ALASS-output text relation hold. Anything else
returns ``None`` so the caller keeps the existing fail-closed behavior
(verify the raw ALASS bytes, which then reject as before).

What is proven before reconstructing:

* equal, non-zero cue counts across original, sanitized working copy, and
  ALASS output, with identical cue numbers in order (positional 1:1 mapping);
* ALASS fidelity: every sanitized cue's normalized text equals the
  corresponding ALASS output cue's normalized text (ALASS invented, dropped,
  or rewrote nothing);
* every original-vs-ALASS normalized difference is exactly explained by
  sanitization (sanitized text equals ALASS text wherever original differs).

The rebuilt candidate keeps original cue numbers/texts verbatim with ALASS
start/end timings, and must re-parse to the same cue count or it is
discarded. Genuine ALASS mutations, dropped/duplicated/reordered cues, and
ambiguous mappings all yield ``None``.

Fallback: when sanitization excluded content (a dropped translator credit,
empty cues, or an unplaceable stray line) no 1:1 mapping exists. If the
sanitized working copy vs ALASS output proves the alignment IS one rigid
global shift — start and end shifts constant within
``OFFSET_CONSISTENCY_TOLERANCE_MS``, dialogue untouched — that exact shift is
applied to every timed block of the original. Untimed blocks survive
byte-identical: no cue ownership is invented and no timestamp is guessed.
Anything else (nonlinear drift, ALASS dialogue changes, negative results)
yields ``None``. A rigid shift never approximates a nonlinear transform.
"""

from __future__ import annotations

import statistics

from two_point_sync.anchors import normalize_text
from two_point_sync.parser import (
    TIMING_LINE_RE,
    SubtitleParseError,
    _split_blocks,
    format_timestamp,
    parse_srt_for_verification,
    parse_timestamp,
)
from two_point_sync.parser import Cue as NucleusCue

#: Maximum per-cue deviation from the median shift for a validated global
#: offset (ms). A passing file moves as one rigid block; anything looser
#: would approximate nonlinear drift with a constant, which is forbidden.
OFFSET_CONSISTENCY_TOLERANCE_MS = 250
#: Minimum sanitized-domain cues required to establish a global offset.
MIN_OFFSET_CUES = 10


def _lenient(text: str) -> list[NucleusCue] | None:
    """Verification-input parse; ``None`` on ambiguity or parse failure."""
    try:
        cues, report = parse_srt_for_verification(text, exclude_ambiguous=True)
    except SubtitleParseError:
        return None
    except Exception:
        return None
    if cues is None:
        return None
    return cues


def reconstruct_with_original_dialogue(
    original_text: str, alass_text: str
) -> str | None:
    """Rebuild ``alass_text`` timings over ``original_text`` dialogue.

    Returns the reconstructed SRT, or ``None`` when the mapping is ambiguous
    or ALASS altered dialogue (caller must then keep fail-closed behavior).
    Never raises: every parse/shape failure maps to ``None``.
    """
    try:
        exact = _reconstruct_exact(original_text, alass_text)
    except Exception:
        exact = None
    if exact is not None:
        return exact
    try:
        grouped = _reconstruct_split_output(original_text, alass_text)
    except Exception:
        grouped = None
    if grouped is not None:
        return grouped
    try:
        return _reconstruct_by_offset(original_text, alass_text)
    except Exception:
        return None


def _reconstruct_split_output(original_text: str, alass_text: str) -> str | None:
    """Rejoin a uniquely partitioned ALASS split onto each pristine cue.

    A merged output cannot determine intermediate original cue boundaries, so
    this path declines it. Every output cue must belong to exactly one exact
    working-copy text group in sequence; missing/invented words cannot pass.
    """
    from app.services.sync_service import prepare_alass_target

    original = _lenient(original_text)
    working = _lenient(prepare_alass_target(original_text))
    produced = _lenient(alass_text)
    if (not original or not working or not produced or len(original) != len(working)
            or len(produced) <= len(working)):
        return None
    blocks = _split_blocks(original_text.lstrip("\ufeff"))
    position = 0
    for old, clean in zip(original, working, strict=True):
        expected = normalize_text(clean.text)
        matches = [end for end in range(position + 1, min(len(produced), position + 4) + 1)
                   if normalize_text(" ".join(c.text for c in produced[position:end])) == expected]
        if len(matches) != 1:
            return None
        end = matches[0]
        group = produced[position:end]
        if any(b.start_ms < a.end_ms for a, b in zip(group, group[1:], strict=False)):
            return None
        blocks[old.order] = TIMING_LINE_RE.sub(
            f"{format_timestamp(group[0].start_ms)} --> {format_timestamp(group[-1].end_ms)}",
            blocks[old.order], count=1)
        position = end
    if position != len(produced):
        return None
    return "\n\n".join(blocks) + "\n"


def _reconstruct_exact(original_text: str, alass_text: str) -> str | None:
    try:
        original_cues = _lenient(original_text)
        alass_cues = _lenient(alass_text)
    except Exception:
        return None
    if not original_cues or not alass_cues or len(original_cues) != len(alass_cues):
        return None
    if any(a.number != b.number for a, b in zip(original_cues, alass_cues, strict=False)):
        return None
    try:
        from app.services.sync_service import prepare_alass_target

        sanitized_text = prepare_alass_target(original_text)
        sanitized_cues = _lenient(sanitized_text)
    except Exception:
        return None
    if sanitized_cues is None or len(sanitized_cues) != len(original_cues):
        # Sanitization dropped or split cues: no unambiguous mapping.
        return None
    if any(a.number != b.number for a, b in zip(sanitized_cues, original_cues, strict=False)):
        return None
    blocks = _split_blocks(original_text.lstrip("\ufeff"))
    for original, sanitized, produced in zip(original_cues, sanitized_cues, alass_cues, strict=False):
        if (normalize_text(sanitized.text) != normalize_text(produced.text)
                and normalize_text(original.text) != normalize_text(produced.text)):
            # ALASS rewrote working-copy dialogue: not a sanitization
            # artifact, so reconstruction would hide real corruption.
            return None
        # Any residual original-vs-produced difference is sanitization-born:
        # counts/numbers are equal (no cue drops or splits), and every other
        # sanitize transform (BOM, newlines, kashida, tags, whitespace) is
        # invisible to normalize_text. Only ad-line removal remains visible,
        # which is working-copy policy, not candidate corruption.
        # Replace only the known timing line. Empty and unplaceable blocks
        # stay in their original position with their dialogue untouched.
        blocks[original.order] = TIMING_LINE_RE.sub(
            f"{format_timestamp(produced.start_ms)} --> {format_timestamp(produced.end_ms)}",
            blocks[original.order], count=1,
        )
    try:
        rendered = "\n\n".join(blocks) + "\n"
        check = _lenient(rendered)
    except Exception:
        return None
    if check is None or len(check) != len(original_cues):
        return None
    if any(a.number != b.number for a, b in zip(check, original_cues, strict=False)):
        return None
    return rendered


def _reconstruct_by_offset(original_text: str, alass_text: str) -> str | None:
    """Apply a validated constant ALASS shift to every original timing line.

    See the module docstring for the safety contract. Never raises.
    """
    try:
        from app.services.sync_service import prepare_alass_target

        sanitized_text = prepare_alass_target(original_text)
        sanitized_cues = _lenient(sanitized_text)
        produced_cues = _lenient(alass_text)
    except Exception:
        return None
    if (not sanitized_cues or not produced_cues
            or len(sanitized_cues) < MIN_OFFSET_CUES
            or len(sanitized_cues) != len(produced_cues)):
        return None
    if any(a.number != b.number
           for a, b in zip(sanitized_cues, produced_cues, strict=False)):
        return None
    for clean, produced in zip(sanitized_cues, produced_cues, strict=False):
        if normalize_text(clean.text) != normalize_text(produced.text):
            # ALASS rewrote working-copy dialogue: its timings prove nothing
            # about the original, so no shift may be transferred.
            return None
    start_shifts = [p.start_ms - s.start_ms
                    for s, p in zip(sanitized_cues, produced_cues, strict=False)]
    end_shifts = [p.end_ms - s.end_ms
                  for s, p in zip(sanitized_cues, produced_cues, strict=False)]
    offset = statistics.median(start_shifts)
    if (max(abs(shift - offset) for shift in start_shifts) > OFFSET_CONSISTENCY_TOLERANCE_MS
            or max(abs(shift - offset) for shift in end_shifts) > OFFSET_CONSISTENCY_TOLERANCE_MS
            or abs(statistics.median(end_shifts) - offset) > OFFSET_CONSISTENCY_TOLERANCE_MS):
        # Nonlinear drift or duration reshaping: a constant must never
        # approximate it. The exact path already declined; stay fail-closed.
        return None
    offset_ms = int(round(offset))
    rendered = apply_linear_timings(original_text, 1.0, offset_ms)
    if rendered is None:
        return None
    try:
        check = _lenient(rendered)
        original_cues = _lenient(original_text)
    except Exception:
        return None
    if (check is None or original_cues is None or len(check) != len(original_cues)
            or any(a.text != b.text
                   for a, b in zip(check, original_cues, strict=False))):
        # Dialogue loss or gain (including dropped unmeasurable content).
        return None
    return rendered


def apply_linear_timings(
    original_text: str, scale: float, offset_ms: float
) -> str | None:
    """Apply ``t_new = scale * t + offset`` to every timed block, in place.

    The single timing-transfer primitive for the original subtitle: cue
    numbers, dialogue text, formatting, ordering, and unplaceable blocks all
    survive byte-identical; only timestamp spans change. Untimed or
    unparseable blocks pass through untouched (still ambiguous, still
    preserved). Returns ``None`` when any result is unrepresentable
    (negative start, degenerate duration), when the scale is not positive,
    or when anything but timestamp spans changed. Never raises.
    """
    if not isinstance(scale, int | float) or not isinstance(offset_ms, int | float):
        return None
    if not scale > 0:
        return None
    try:
        blocks = _split_blocks(original_text.lstrip("﻿"))
        shifted_blocks: list[str] = []
        for block in blocks:
            lines = block.split("\n")
            if len(lines) >= 2:
                try:
                    int(lines[0].strip())
                except ValueError:
                    shifted_blocks.append(block)
                    continue
                timing = TIMING_LINE_RE.search(lines[1])
                if timing is None:
                    shifted_blocks.append(block)
                    continue
                try:
                    start_ms = parse_timestamp(timing.group("start"))
                    end_ms = parse_timestamp(timing.group("end"))
                except SubtitleParseError:
                    # Unparseable timing stays verbatim: still ambiguous,
                    # still preserved, still contributes no timing evidence.
                    shifted_blocks.append(block)
                    continue
                new_start = int(round(start_ms * scale + offset_ms))
                new_end = int(round(end_ms * scale + offset_ms))
                if new_start < 0 or new_end <= new_start:
                    # Unrepresentable result: never clamp or invent, abstain.
                    return None
                lines[1] = TIMING_LINE_RE.sub(
                    f"{format_timestamp(new_start)} --> {format_timestamp(new_end)}",
                    lines[1], count=1,
                )
                shifted_blocks.append("\n".join(lines))
            else:
                # Untimed blocks (stray lines, residues) pass through untouched.
                shifted_blocks.append(block)
        for old_block, new_block in zip(blocks, shifted_blocks, strict=True):
            if old_block != new_block and (
                    TIMING_LINE_RE.sub("", old_block) != TIMING_LINE_RE.sub("", new_block)):
                # Only timestamp spans may change; anything else is corruption.
                return None
        return (("﻿" if original_text.startswith("﻿") else "")
                + "\n\n".join(shifted_blocks) + "\n")
    except Exception:
        return None
