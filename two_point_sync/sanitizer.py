"""Anchor-suitability classification for subtitle cues.

This module never deletes or rewrites subtitle content. It only classifies
cues according to their suitability as synchronisation anchors. A cue that is
unsuitable as an anchor still remains in the synchronised subtitle output.

Duration/position heuristics (long cues, film edges, short or repeated
dialogue, musical content) are confidence penalties or candidacy flags, never
unconditional deletion rules.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from two_point_sync.parser import Cue

# ---------------------------------------------------------------------------
# Public contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnchorEligibility:
    """Classification result for one cue's anchor candidacy."""

    eligible: bool
    confidence: float
    reasons: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Pattern tables (Arabic + English)
# ---------------------------------------------------------------------------

_URL_RE = re.compile(
    r"(https?://|www\.|\.com\b|\.net\b|\.org\b|\.tv\b|\.me\b|@\w|"
    r"opensubtitle|subscene|yify|yts|mydream|egybest|mycima|fasel|akwam)",
    re.IGNORECASE,
)

_DOWNLOAD_RE = re.compile(
    r"(download|subtitle\s*download|get\s+more|sync\s*by|resync|re-sync|"
    r"uploaded?\s*by|encoder|encode\s*by|release\s*by|rip\s*by|"
    r"\b vaso\b|\bozoz\b|\boz+oz\b|"
    r"تحميل|حمّل|لتحميل|الترجمات|ترجمات|مزامنة|مزامنه|رفع|إعداد|اعداد)",
    re.IGNORECASE,
)

_CREDIT_RE = re.compile(
    r"(translator|translated\s*by|translation\s*by|translators?|subtitles?\s*by|"
    r"synced?\s*by|corrected?\s*by|edited?\s*by|adjusted?\s*by|timed?\s*by|"
    r"ترجم[ةه]|مترجم|قام\s*بسحب|سحب\s*الترجم|تنسيق|تدقيق|مراجعة|"
    r"إعداد\s*الترجم|اعداد\s*الترجم|فريق\s*الترجم|تمت\s*الترجم|"
    r"صلاح\s*سليم)",
    re.IGNORECASE,
)

_WATERMARK_RESIDUE_RE = re.compile(r"(ozoz|omarbaga|arabs?cene|arabseed)", re.IGNORECASE)

_TAG_RE = re.compile(r"\{[^}]*\}|<[^>]*>")
_BIDI_RE = re.compile(r"[‎‏‪‫‬‭‮⁦⁧⁨⁩﻿⁠]")
_MUSIC_SYMBOL_RE = re.compile(r"[♪♫♩♬🎵🎶]")
_ARABIC_LETTER_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
_DIGIT_RE = re.compile(r"[0-9\u0660-\u0669\u06F0-\u06F9]")

# A cue whose visible payload is only digits/punctuation is a counting or
# slate fragment (e.g. "3، 4", "115", "1 2 3"), not dialogue evidence.
_PURE_COUNT_RE = re.compile(
    r"^[\s0-9\u0660-\u0669\u06F0-\u06F9\u200e\u200f\u202a-\u202e"
    r"\.,،:;!\?\-\–\—_~'\"`´‘’“”«»\(\)\[\]\{\}…/\\|]+$"
)

_BRACKETED_RE = re.compile(r"[\(\[\{][^\)\]\}]*[\)\]\}]")

_GENERIC_SINGLETONS = frozenset(
    {
        "نعم",
        "لا",
        "اجل",
        "حسنا",
        "حاضر",
        "شكرا",
        "عفوا",
        "مرحبا",
        "اهلا",
        "اسف",
        "yes",
        "no",
        "ok",
        "okay",
        "thanks",
        "thank you",
        "please",
        "hey",
        "hi",
        "wow",
    }
)

# Duration/position signals are penalties only (section 5.2 of the task).
LONG_DURATION_MS = 8000
EDGE_HEAD_MS = 30_000
EDGE_TAIL_MS = 120_000
LONG_STATIONARY_CHARS = 220


def visible_text(text: str) -> str:
    """Return the human-visible payload of a cue (tags/bidi stripped)."""
    cleaned = _TAG_RE.sub(" ", text)
    cleaned = _BIDI_RE.sub("", cleaned)
    cleaned = _MUSIC_SYMBOL_RE.sub(" ", cleaned)
    return cleaned.strip()


def _has_letters(payload: str) -> bool:
    return bool(_ARABIC_LETTER_RE.search(payload) or _LATIN_LETTER_RE.search(payload))


def _basic_norm(text: str) -> str:
    cleaned = _TAG_RE.sub(" ", text)
    cleaned = _BIDI_RE.sub("", cleaned)
    cleaned = unicodedata.normalize("NFKC", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().casefold()
    return cleaned


def _is_credit_position(cue: Cue, film_start_ms: int, film_end_ms: int) -> bool:
    return cue.start_ms < film_start_ms + EDGE_HEAD_MS or cue.start_ms > film_end_ms - EDGE_TAIL_MS


def classify_cue(
    cue: Cue,
    *,
    film_start_ms: int,
    film_end_ms: int,
    repetition_count: int = 1,
) -> AnchorEligibility:
    """Classify a single cue's suitability as a synchronisation anchor."""
    reasons: list[str] = []
    confidence = 1.0

    payload = visible_text(cue.text)
    duration_ms = cue.end_ms - cue.start_ms

    # --- Hard exclusions (non-dialogue content) ---------------------------
    if payload == "" or not (_has_letters(payload) or _DIGIT_RE.search(payload)):
        return AnchorEligibility(
            eligible=False, confidence=0.0, reasons=("formatting_only_or_empty",)
        )

    if _URL_RE.search(cue.text) or _URL_RE.search(payload):
        return AnchorEligibility(
            eligible=False, confidence=0.0, reasons=("website_advertisement",)
        )

    if _WATERMARK_RESIDUE_RE.search(cue.text):
        return AnchorEligibility(
            eligible=False, confidence=0.0, reasons=("uploader_watermark",)
        )

    if _CREDIT_RE.search(payload) or _DOWNLOAD_RE.search(payload):
        # Translator/download credits cluster at the film edges, but the
        # lexical evidence alone is decisive wherever they appear.
        return AnchorEligibility(
            eligible=False, confidence=0.0, reasons=("translation_credit_or_ad",)
        )

    if _PURE_COUNT_RE.match(payload):
        # Standalone numbering / countdown slate (e.g. "3، 4"). Word-form
        # counting ("واحد، اثنان") still carries letters and stays eligible.
        return AnchorEligibility(
            eligible=False, confidence=0.05, reasons=("counting_or_numbering_fragment",)
        )

    without_bracketed = _BRACKETED_RE.sub(" ", payload).strip(" \t\n-–—:؛،,.!?؟\"'“”‘’")
    if without_bracketed == "" and _BRACKETED_RE.search(payload):
        return AnchorEligibility(
            eligible=False, confidence=0.05, reasons=("standalone_sound_effect",)
        )

    if _MUSIC_SYMBOL_RE.search(cue.text) and not _has_letters(payload):
        return AnchorEligibility(
            eligible=False, confidence=0.05, reasons=("music_only",)
        )

    # --- Soft penalties (warning signals, never deletions) -----------------
    if duration_ms > LONG_DURATION_MS:
        confidence -= 0.15
        reasons.append("long_duration_warning")
    if len(payload) >= LONG_STATIONARY_CHARS and duration_ms > LONG_DURATION_MS:
        confidence -= 0.10
        reasons.append("long_stationary_text_warning")
    if cue.start_ms < film_start_ms + EDGE_HEAD_MS:
        confidence -= 0.10
        reasons.append("near_film_start_warning")
    if cue.start_ms > film_end_ms - EDGE_TAIL_MS:
        confidence -= 0.10
        reasons.append("near_film_end_warning")
    if len(payload) < 4:
        confidence -= 0.20
        reasons.append("very_short_warning")
    if repetition_count > 2:
        confidence -= 0.25
        reasons.append("repeated_dialogue_warning")
    if _MUSIC_SYMBOL_RE.search(cue.text):
        confidence -= 0.15
        reasons.append("musical_content_warning")

    norm = _basic_norm(payload)
    if norm in _GENERIC_SINGLETONS:
        confidence -= 0.30
        reasons.append("generic_phrase_warning")

    confidence = max(0.0, min(1.0, confidence))
    if not reasons:
        reasons.append("dialogue_candidate")
    return AnchorEligibility(eligible=True, confidence=confidence, reasons=tuple(reasons))


def classify_all(cues: list[Cue]) -> list[AnchorEligibility]:
    """Classify every cue; input cue objects are never mutated."""
    if not cues:
        return []
    film_start_ms = min(cue.start_ms for cue in cues)
    film_end_ms = max(cue.end_ms for cue in cues)
    counts: dict[str, int] = {}
    norms = [_basic_norm(visible_text(cue.text)) for cue in cues]
    for norm in norms:
        counts[norm] = counts.get(norm, 0) + 1
    return [
        classify_cue(
            cue,
            film_start_ms=film_start_ms,
            film_end_ms=film_end_ms,
            repetition_count=counts[norm],
        )
        for cue, norm in zip(cues, norms, strict=False)
    ]
