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
"""

from __future__ import annotations

from two_point_sync.anchors import normalize_text
from two_point_sync.parser import Cue as NucleusCue
from two_point_sync.parser import SubtitleParseError, format_srt, parse_srt


def _cue_key(cue: NucleusCue) -> tuple[int, int, int]:
    return (cue.number, cue.start_ms, cue.end_ms)


def reconstruct_with_original_dialogue(
    original_text: str, alass_text: str
) -> str | None:
    """Rebuild ``alass_text`` timings over ``original_text`` dialogue.

    Returns the reconstructed SRT, or ``None`` when the mapping is ambiguous
    or ALASS altered dialogue (caller must then keep fail-closed behavior).
    Never raises: every parse/shape failure maps to ``None``.
    """
    try:
        original_cues = parse_srt(original_text)
        alass_cues = parse_srt(alass_text)
    except SubtitleParseError:
        return None
    except Exception:
        return None
    if not original_cues or len(original_cues) != len(alass_cues):
        return None
    if any(a.number != b.number for a, b in zip(original_cues, alass_cues, strict=False)):
        return None
    try:
        from app.services.sync_service import prepare_alass_target

        sanitized_text = prepare_alass_target(original_text)
        sanitized_cues = parse_srt(sanitized_text)
    except Exception:
        return None
    if len(sanitized_cues) != len(original_cues):
        # Sanitization dropped or split cues: no unambiguous mapping.
        return None
    if any(a.number != b.number for a, b in zip(sanitized_cues, original_cues, strict=False)):
        return None
    rebuilt: list[NucleusCue] = []
    for original, sanitized, produced in zip(original_cues, sanitized_cues, alass_cues, strict=False):
        if normalize_text(sanitized.text) != normalize_text(produced.text):
            # ALASS rewrote working-copy dialogue: not a sanitization
            # artifact, so reconstruction would hide real corruption.
            return None
        # Any residual original-vs-produced difference is sanitization-born:
        # counts/numbers are equal (no cue drops or splits), and every other
        # sanitize transform (BOM, newlines, kashida, tags, whitespace) is
        # invisible to normalize_text. Only ad-line removal remains visible,
        # which is working-copy policy, not candidate corruption.
        rebuilt.append(
            NucleusCue(
                number=original.number,
                order=original.order,
                start_ms=produced.start_ms,
                end_ms=produced.end_ms,
                text=original.text,
            )
        )
    try:
        rendered = format_srt(rebuilt)
        check = parse_srt(rendered)
    except (SubtitleParseError, Exception):
        return None
    if len(check) != len(original_cues):
        return None
    if any(a.number != b.number for a, b in zip(check, original_cues, strict=False)):
        return None
    return rendered
