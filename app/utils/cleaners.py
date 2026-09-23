"""Subtitle advertisement cleaning utilities.

``strip_advertisements`` removes promotional cues (website links, telegram/twitter
handles, "encoded by" credits, ...) from SRT and WebVTT files while leaving
legitimate dialogue untouched. Detection is intentionally restricted to the
opening (first 60s) and closing (last 90s) windows, where ad cues are almost
always placed, to minimise false positives.

The function is format-preserving and re-indexes SRT cue numbers so the output
stays strictly spec-compliant.
"""

import re
from dataclasses import dataclass
from typing import Any

# Opening / closing windows (seconds) in which ad cues are detected.
_LEADING_WINDOW_SECONDS = 60.0
_TRAILING_WINDOW_SECONDS = 90.0

# High-signal promotional patterns (case-insensitive for Latin text).
_AD_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"https?://", re.IGNORECASE),
    re.compile(r"www\.", re.IGNORECASE),
    re.compile(r"\.(?:com|org|net|me)\b", re.IGNORECASE),
    re.compile(r"t\.me/", re.IGNORECASE),
    re.compile(r"subscene", re.IGNORECASE),
    re.compile(r"opensubtitles", re.IGNORECASE),
    re.compile(r"sync\s+by", re.IGNORECASE),
    re.compile(r"corrected\s+by", re.IGNORECASE),
    re.compile(r"encoded\s+by", re.IGNORECASE),
    re.compile(r"translated\s+by", re.IGNORECASE),
    re.compile(r"قناة"),
    re.compile(r"تيليجرام"),
    re.compile(r"تويتر"),
    re.compile(r"@\w+", re.IGNORECASE),
)

# Direct translation credits that are whitelisted when keep_translator_credits is on.
_CREDIT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ترجمة"),
    re.compile(r"تعريب"),
    re.compile(r"تعديل التوقيت"),
    re.compile(r"ضبط التوقيت"),
    re.compile(r"تقديم"),
    re.compile(r"دمج"),
    re.compile(r"إعداد"),
    re.compile(r"translated\s+by", re.IGNORECASE),
    re.compile(r"translation\s+by", re.IGNORECASE),
    re.compile(r"synced\s+by", re.IGNORECASE),
    re.compile(r"timed\s+by", re.IGNORECASE),
    re.compile(r"subtitles?\s+by", re.IGNORECASE),
)

# Translator-attribution triggers after which a social handle (@username) is a
# credit, not spam. A handle is protected only when it DIRECTLY follows a
# trigger (tolerating colons/dashes/spaces, EN case-insensitive), e.g.
# "Translated by: @D700mka", "ترجمة: @D700mka", "تعديل التوقيت: @D700mka".
# Standalone promo handles ("Follow @x", "Join @x", "@promo") never match.
_TRANSLATOR_TRIGGER_CORE = (
    r"(?:"
    r"translated\s+by"
    r"|translation\s+by"
    r"|synced\s+by"
    r"|timed\s+by"
    r"|subtitles?\s+by"
    r"|ترجمة"
    r"|تعديل التوقيت"
    r"|ضبط التوقيت"
    r"|تقديم"
    r"|دمج"
    r"|إعداد"
    r")"
)
_TRANSLATOR_TRIGGER_AT_END_REGEX = re.compile(
    _TRANSLATOR_TRIGGER_CORE + r"\s*[:\u061b\uff1a\-_\u2013\u2014]*\s*$",
    re.IGNORECASE,
)
# Separator run between co-translator handles in a credit list:
# whitespace, commas (Latin/Arabic), slashes, ampersands, plus signs,
# Arabic "و" or English "and" (e.g. "@a, @b", "@a & @b", "@a و @b",
# "@a and @b"). Anchored at the end of the preceding context.
_HANDLE_CHAIN_SEPARATOR_AT_END_REGEX = re.compile(
    r"(?:[\s,،;/&+]|\s*و\s*|(?<!\w)and(?!\w)\s*)+$",
    re.IGNORECASE,
)
_HANDLE_TOKEN_REGEX = re.compile(r"@\w+", re.IGNORECASE)

# Promotional tokens stripped from a credit line (URLs, domains, handles, sites).
_AD_TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"https?://\S+", re.IGNORECASE),
    re.compile(r"www\.\S+", re.IGNORECASE),
    re.compile(r"\b[\w.-]+\.(?:com|org|net|me)\b\S*", re.IGNORECASE),
    re.compile(r"t\.me/\S+", re.IGNORECASE),
    re.compile(r"subscene", re.IGNORECASE),
    re.compile(r"opensubtitles", re.IGNORECASE),
    re.compile(r"@\w+", re.IGNORECASE),
    re.compile(r"قناة"),
    re.compile(r"تيليجرام"),
    re.compile(r"تويتر"),
)

_TIMESTAMP_REGEX = re.compile(r"(\d{1,3}):(\d{2}):(\d{2})[,.](\d{1,3})")
_BLANK_LINE_REGEX = re.compile(r"\n[ \t]*\n")
_EDGE_SEPARATORS_REGEX = re.compile(r"^[\s\-|:•·,]+|[\s\-|:•·,]+$")

# --- Syntax cleaning -------------------------------------------------------
_ALLOWED_TAG_NAMES = frozenset({"i", "b", "u"})
_ANY_TAG_REGEX = re.compile(r"<[^>]*>")
_BR_TAG_REGEX = re.compile(r"(?i)<br\s*/?>")
_MULTISPACE_REGEX = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT_REGEX = re.compile(r"\s+([,.!?;:،؛؟])")
_LOOSE_DOUBLE_DASH_REGEX = re.compile(r"(?<!-)--(?!-)")
# Latin comma sitting right after an Arabic letter (optionally spaced) -> Arabic comma.
# Requires an Arabic character before the comma, so digit groups ("1,000") and
# timestamp fragments ("00:01:23,456") are never touched.
_ARABIC_RANGE = "\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF"
_LATIN_COMMA_IN_ARABIC_REGEX = re.compile(rf"([{_ARABIC_RANGE}])\s*,\s*")
_TIMESTAMP_LINE_REGEX = re.compile(
    r"^(?P<start>\d{1,3}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
    r"(?P<end>\d{1,3}:\d{2}:\d{2}[,.]\d{1,3})(?P<rest>.*)$"
)
_MAX_OVERLAP_MS = 500

# Color stripping (opt-in): ASS override commands, <font ...> tags, WebVTT classes.
_ASS_COLOR_TAG_REGEX = re.compile(r"\{\\[1-4]?[cC](?:&H[0-9A-Fa-f]+&)?\}")
_FONT_TAG_REGEX = re.compile(r"(?i)</?font\b[^>]*>")
_VTT_CLASS_TAG_REGEX = re.compile(r"(?i)</?c(?:\.[A-Za-z0-9_-]+)*>")
_VTT_VOICE_TAG_REGEX = re.compile(r"(?i)</?v(?:\.[A-Za-z0-9_-]+)*>")

# --- Eastern Arabic numerals (opt-in) --------------------------------------
_WESTERN_TO_EASTERN_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
# HTML tags and ASS override blocks whose attribute digits must never change.
_PROTECTED_TAG_REGEX = re.compile(r"<[^>]*>|\{[^}]*\}")
_DIGIT_RUN_REGEX = re.compile(r"\d+")
_ASCII_LETTER_REGEX = re.compile(r"[A-Za-z]")
# Glued identifier characters: Latin letters, digits, dots, hyphens and
# underscores (AK-47, MP4, RTX-4090, S01_E01). Used only by the
# Eastern-Arabic numeral guard below.
_LATIN_TOKEN_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" + "_"
)
_ASS_DIALOGUE_LINE_REGEX = re.compile(r"^(?:Dialogue|Comment):", re.IGNORECASE)

# Arabic Harakat to strip: Fatha, Damma, Kasra, Sukun, superscript Alef and the
# small high rounded zero. U+0651 SHADDA and every Tanween type (Fathatan U+064B,
# Dammatan U+064C, Kasratan U+064D) are excluded so they are always preserved.
_ARABIC_DIACRITICS_REGEX = re.compile("[\u064e\u064f\u0650\u0652\u0670\u06df]")
# Temporary placeholder that shields a meaningful feminine Kasra during stripping.
_PROTECTED_KASRA_PLACEHOLDER = "__KASRA__"
_ARABIC_LETTERS = "\u0621-\u064a"

# Feminine endings whose Kasra is meaningful and must survive (matched at word
# boundaries using Arabic-letter lookarounds rather than ASCII \b).
_PROTECTED_KASRA_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "أنتِ" may carry internal vowels (e.g. "أَنْتِ"): allow optional harakat
    # between the letters while still requiring the final Kasra.
    re.compile(
        r"(?<![\u0621-\u064A])([أإ][\u064E\u064F\u0650]?ن[\u064E\u064F\u0650\u0652]?ت)ِ"
        r"(?![\u0621-\u064A])"
    ),
    # Attached feminine Kaf pronouns (لكِ, منكِ, ...): intermediate harakat /
    # sukun do not break matching (e.g. "لَكِ", "مِنْكِ", "مَعَكِ", "إِلَيْكِ").
    re.compile(
        rf"(?<![{_ARABIC_LETTERS}])"
        r"("
        r"ل[\u064B-\u0652]?ك"
        r"|م[\u064B-\u0652]?ن[\u064B-\u0652]?ك"
        r"|ع[\u064B-\u0652]?ن[\u064B-\u0652]?ك"
        r"|ف[\u064B-\u0652]?ي[\u064B-\u0652]?ك"
        r"|م[\u064B-\u0652]?ع[\u064B-\u0652]?ك"
        r"|إ[\u064B-\u0652]?ل[\u064B-\u0652]?ي[\u064B-\u0652]?ك"
        r"|ع[\u064B-\u0652]?ل[\u064B-\u0652]?ي[\u064B-\u0652]?ك"
        r"|ب[\u064B-\u0652]?ه[\u064B-\u0652]?ـ?[\u064B-\u0652]?ك"
        r")ِ"
        rf"(?![{_ARABIC_LETTERS}])"
    ),
)

# Feminine suffix Ta' (تاء الفاعل للمخاطبة): a word-final "تِ" whose Kasra is
# meaningful ("علّمتِ", "فعلتِ", "قلتِ", "عَلَّمْتِ"). The preceding letter (and an
# optional diacritic) is captured so the Kasra can be masked without using a
# variable-width lookbehind (unsupported by Python's ``re``).
_PROTECTED_TA_KASRA_REGEX = re.compile(r"([\u0621-\u064A][\u064B-\u0652]?)تِ(?![\u0621-\u064A])")

# Feminine attached Kaf (كاف المخاطبة): any word-final "كِ" keeps its Kasra
# ("إنّكِ", "كأنّكِ", "كتابكِ", "رأيتكِ"). ``*`` tolerates combined Shadda + Haraka
# before the Kaf (e.g. إِنَّكِ).
_PROTECTED_KAF_KASRA_REGEX = re.compile(r"([\u0621-\u064A][\u064B-\u0652]*)كِ(?![\u0621-\u064A])")

# --- In-dialogue HI artifact stripping -------------------------------------
# Bracketed sound descriptions: "[MUSIC PLAYING]", "(GUNSHOT)", "[صوت بكاء]".
_BRACKET_REGEX = re.compile(r"[\[\(][^\[\]\(\)]*[\]\)]")
_HI_ARABIC_KEYWORDS: tuple[str, ...] = (
    "صوت",
    "موسيقى",
    "موسيقي",
    "بكاء",
    "ضحك",
    "صراخ",
    "صرخة",
    "تنهيد",
    "خطوات",
    "رنين",
    "تصفيق",
    "صمت",
    "همس",
    "أنفاس",
    "صرير",
    "زفير",
    "شهيق",
    "باب",
    "هاتف",
    "قفل",
)
_LATIN_UPPERCASE_REGEX = re.compile(r"^[A-Z0-9\s'’\-\.!?…&/]+$")
_HAS_LATIN_LETTER_REGEX = re.compile(r"[A-Za-z]")
_HAS_LOWERCASE_REGEX = re.compile(r"[a-z]")
_HAS_ARABIC_REGEX = re.compile(rf"[{_ARABIC_RANGE}]")
# Leading speaker labels: "JOHN: ", "سارة: ", optionally preceded by a dash.
# Restricted to a single token so normal dialogue ("Wait a minute: ...") is untouched.
_SPEAKER_LABEL_REGEX = re.compile(
    rf"^\s*[-–—]?\s*[A-Za-z{_ARABIC_RANGE}][A-Za-z0-9{_ARABIC_RANGE}_'’\-]{{0,30}}:\s*"
)


def _looks_like_ass(content: str) -> bool:
    """ASS/SSA is not cue-based like SRT/VTT; skip it to avoid corrupting styles."""
    lowered = content[:8192].lower()
    return (
        "[script info]" in lowered
        or "[v4+ styles]" in lowered
        or "[v4 styles]" in lowered
        or "dialogue:" in lowered
    )


# Encoding fallback chain for legacy Arabic subtitle files.
_ENCODING_FALLBACKS: tuple[str, ...] = ("utf-8-sig", "cp1256", "iso-8859-6", "cp1252")


@dataclass(frozen=True)
class CleanOptions:
    """Independent, user-toggleable subtitle cleaning switches."""

    fix_encoding: bool = True
    clean_tags: bool = True
    strip_colors: bool = False
    clean_spacing: bool = True
    clean_symbols: bool = True
    clean_commas: bool = True
    clean_timing: bool = True

    @property
    def any_text_cleaning(self) -> bool:
        return any(
            (
                self.clean_tags,
                self.strip_colors,
                self.clean_spacing,
                self.clean_symbols,
                self.clean_commas,
            )
        )

    @classmethod
    def from_prefs(cls, prefs: Any) -> "CleanOptions":
        """Build options from a ``UserPreferences``-like object (attribute access)."""

        def _flag(name: str, default: bool) -> bool:
            return bool(getattr(prefs, name, default))

        return cls(
            fix_encoding=_flag("fix_encoding", True),
            clean_tags=_flag("clean_tags", True),
            strip_colors=_flag("strip_colors", False),
            clean_spacing=_flag("clean_spacing", True),
            clean_symbols=_flag("clean_symbols", True),
            clean_commas=_flag("clean_commas", True),
            clean_timing=_flag("clean_timing", True),
        )


def fix_subtitle_encoding_bytes(content: bytes) -> bytes:
    """
    Decode legacy-encoded subtitle bytes and re-encode as clean UTF-8.

    Tries ``utf-8`` (BOM-aware), ``cp1256`` (Windows Arabic), ``iso-8859-6``
    (ISO Arabic) and finally ``cp1252`` (with replacement) so no byte sequence
    ever raises.
    """
    if not content:
        return content
    for encoding in _ENCODING_FALLBACKS:
        try:
            return content.decode(encoding).encode("utf-8")
        except (UnicodeDecodeError, LookupError):
            continue
    return content.decode("cp1252", errors="replace").encode("utf-8")


def strip_arabic_diacritics(content: str) -> str:
    """
    Removal of Arabic Tashkeel (Harakat).

    Preserved:
    - Bare Shadda (U+0651) on all letters, without any bundled vowel.
    - All Tanween types: Fathatan (U+064B), Dammatan (U+064C), Kasratan (U+064D).
    - Meaningful feminine Kasra on pronouns/attachments (``أنتِ``, ``لكِ``,
      ``إليكِ``, ...) and the feminine suffix Ta' (``علّمتِ``, ``عَلَّمْتِ``).

    Protected Kasras are shielded behind a placeholder, the generic Harakat are
    stripped, then the placeholder is restored as Kasra (fully reversible).

    Only applied when Arabic text and strippable diacritics are present, so
    non-Arabic (and already-clean) subtitles are returned byte-for-byte
    unchanged. Timestamps and tags contain only ASCII and are never affected.
    """
    if not content or not _HAS_ARABIC_REGEX.search(content):
        return content
    if not _ARABIC_DIACRITICS_REGEX.search(content):
        return content

    text = content
    for pattern in _PROTECTED_KASRA_PATTERNS:
        text = pattern.sub(
            lambda match: match.group(1) + _PROTECTED_KASRA_PLACEHOLDER, text
        )
    text = _PROTECTED_TA_KASRA_REGEX.sub(
        lambda match: match.group(1) + "ت" + _PROTECTED_KASRA_PLACEHOLDER, text
    )
    text = _PROTECTED_KAF_KASRA_REGEX.sub(
        lambda match: match.group(1) + "ك" + _PROTECTED_KASRA_PLACEHOLDER, text
    )
    text = _ARABIC_DIACRITICS_REGEX.sub("", text)
    return text.replace(_PROTECTED_KASRA_PLACEHOLDER, "\u0650")


def strip_arabic_diacritics_bytes(content: bytes) -> bytes:
    """UTF-8 byte wrapper around :func:`strip_arabic_diacritics`."""
    if not content:
        return content
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return content
    return strip_arabic_diacritics(text).encode("utf-8")


def tashkeel_remove(content: str) -> str:
    """Public alias for the contextual Tashkeel (Harakat) removal routine."""
    return strip_arabic_diacritics(content)


def tashkeel_remove_bytes(content: bytes) -> bytes:
    """UTF-8 byte wrapper around :func:`tashkeel_remove`."""
    return strip_arabic_diacritics_bytes(content)


def _parse_seconds(timestamp_line: str) -> tuple[float, float] | None:
    """Return (start_seconds, end_seconds) for an SRT/VTT timestamp line."""
    matches = _TIMESTAMP_REGEX.findall(timestamp_line)
    if len(matches) < 2:
        return None

    def _to_seconds(parts: tuple[str, str, str, str]) -> float:
        hours, minutes, seconds, fraction = parts
        # Normalise 1-2 digit fractions (e.g. ",5" == 500ms).
        millis = int(fraction.ljust(3, "0")[:3])
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + millis / 1000.0

    return _to_seconds(matches[0]), _to_seconds(matches[1])


def _has_ad(text: str) -> bool:
    return any(pattern.search(text) for pattern in _AD_PATTERNS)


def _has_credit(text: str) -> bool:
    return any(pattern.search(text) for pattern in _CREDIT_PATTERNS)


def _translator_protected_handle_spans(line: str) -> list[tuple[int, int]]:
    """    Spans of ``@handle`` tokens directly following a translator trigger.

    Only handles whose preceding text on the line ends with an attribution
    trigger (plus optional colon/dash/spaces) are protected, so
    ``Translated by: @D700mka`` keeps its handle while
    ``... @D700mka Follow @spammer`` still loses ``@spammer``. Chained
    co-translator lists (``ترجمة: @a, @b``, ``@a & @b``, ``@a و @b``,
    ``@a and @b``) protect each subsequent handle too.
    """
    spans: list[tuple[int, int]] = []
    for match in _HANDLE_TOKEN_REGEX.finditer(line):
        context = line[: match.start()]
        if _TRANSLATOR_TRIGGER_AT_END_REGEX.search(context):
            spans.append(match.span())
            continue
        if spans:
            # Strip one trailing separator run, then require the remainder
            # to end with an already-protected handle (recursive chains work
            # because each link leaves a separator-terminated prefix).
            core = _HANDLE_CHAIN_SEPARATOR_AT_END_REGEX.sub("", context)
            for start, end in reversed(spans):
                if not core.endswith(line[start:end]):
                    continue
                before = core[: -len(line[start:end])]
                if _TRANSLATOR_TRIGGER_AT_END_REGEX.search(
                    before
                ) or _HANDLE_CHAIN_SEPARATOR_AT_END_REGEX.search(before):
                    spans.append(match.span())
                break
    return spans


def _strip_promotional_tokens(line: str) -> str:
    """Remove URLs/domains/handles from a line while keeping the credit text.

    Handles directly following a translator-attribution trigger are shielded
    behind placeholders so ``ترجمة: @D700mka`` survives while standalone
    promo handles and ad links are still stripped.
    """
    cleaned = line
    placeholders: dict[str, str] = {}
    spans = _translator_protected_handle_spans(cleaned)
    # Replace from the end so earlier spans stay valid.
    for reversed_index, (start, end) in enumerate(reversed(spans)):
        index = len(spans) - 1 - reversed_index
        handle = cleaned[start:end]
        token = f"\x00TRHANDLE{index}\x00"
        placeholders[token] = handle
        cleaned = cleaned[:start] + token + cleaned[end:]
    for pattern in _AD_TOKEN_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    for token, handle in placeholders.items():
        cleaned = cleaned.replace(token, handle)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = _EDGE_SEPARATORS_REGEX.sub("", cleaned)
    return cleaned.strip()


def _process_cue_text(
    text_lines: list[str], keep_translator_credits: bool
) -> tuple[bool, list[str]]:
    """
    Decide whether a cue should be dropped and, when kept, return its cleaned lines.

    - keep_translator_credits: retain whitelisted credit lines (stripping only the
      promotional URLs/handles glued to them) and drop pure-ad lines.
    - Otherwise: drop every line that is promotional or credited.
    Returns (drop_cue, processed_lines).
    """
    processed: list[str] = []
    for line in text_lines:
        has_ad = _has_ad(line)
        has_credit = _has_credit(line)

        if not keep_translator_credits:
            if has_ad or has_credit:
                continue
            processed.append(line)
            continue

        if has_credit:
            cleaned = _strip_promotional_tokens(line) if has_ad else line
            if cleaned.strip():
                processed.append(cleaned)
            continue

        if has_ad:
            continue

        processed.append(line)

    if not processed:
        return True, []
    return False, processed


def strip_advertisements(content: str, keep_translator_credits: bool = True) -> str:
    """
    Remove advertisement cues from SRT/WebVTT subtitle text.

    Only cues whose start time falls inside the first 60 seconds or the last
    90 seconds are candidates, and only when they contain a known promotional
    pattern or credit. SRT sequence numbers are re-generated contiguously.
    Non-SRT/VTT (e.g. ASS) input is returned unchanged.

    When ``keep_translator_credits`` is True, whitelisted credit lines (e.g.
    ``ترجمة``, ``تعريب``, ``Translated by``, ``Subtitles by``) are retained, with
    only the promotional URLs/handles glued to them removed. When False, all
    promotional and credited metadata is discarded.
    """
    if not content or "-->" not in content or _looks_like_ass(content):
        return content

    newline = "\r\n" if "\r\n" in content else "\n"
    normalised = content.replace("\r\n", "\n")
    had_trailing_newline = normalised.endswith("\n")

    is_vtt = normalised.lstrip()[:6].upper().startswith("WEBVTT")

    header_blocks: list[str] = []
    cues: list[dict] = []
    seen_cue = False

    for block in _BLANK_LINE_REGEX.split(normalised.strip("\n")):
        if not block.strip():
            continue
        lines = block.split("\n")
        timestamp_index = next((i for i, line in enumerate(lines) if "-->" in line), None)

        if timestamp_index is None:
            # Keep header/metadata blocks (WEBVTT, Kind:, Language:, ...) only.
            if not seen_cue:
                header_blocks.append(block.strip("\n"))
            continue

        seen_cue = True
        timestamp_line = lines[timestamp_index].strip()
        time_span = _parse_seconds(timestamp_line)
        cues.append(
            {
                "timestamp": timestamp_line,
                "text": lines[timestamp_index + 1 :],
                "start": time_span[0] if time_span else None,
                "end": time_span[1] if time_span else None,
            }
        )

    if not cues:
        return content

    timed_cues = [cue for cue in cues if cue["start"] is not None and cue["end"] is not None]
    total_end = max((cue["end"] for cue in timed_cues), default=0.0)

    trailing_start = max(0.0, total_end - _TRAILING_WINDOW_SECONDS)

    kept: list[dict] = []
    text_modified = False
    for cue in cues:
        start = cue["start"]
        in_window = False
        if start is not None:
            in_window = start <= _LEADING_WINDOW_SECONDS or (
                total_end > 0 and start >= trailing_start
            )
        if in_window:
            drop, cleaned_lines = _process_cue_text(
                cue["text"], keep_translator_credits
            )
            if drop:
                continue
            if cleaned_lines != cue["text"]:
                text_modified = True
                cue = {**cue, "text": cleaned_lines}
        kept.append(cue)

    if len(kept) == len(cues) and not text_modified:
        # Nothing removed; keep the original bytes exactly.
        return content

    rendered: list[str] = []
    index = 1
    for cue in kept:
        block_lines: list[str] = []
        if not is_vtt:
            block_lines.append(str(index))
            index += 1
        block_lines.append(cue["timestamp"])
        block_lines.extend(cue["text"])
        rendered.append("\n".join(block_lines))

    sections = header_blocks + rendered
    result = "\n\n".join(section.strip("\n") for section in sections if section is not None)
    if had_trailing_newline:
        result += "\n"

    return result.replace("\n", newline)


def _timestamp_to_ms(timestamp: str) -> int | None:
    match = _TIMESTAMP_REGEX.match(timestamp.strip())
    if not match:
        return None
    hours, minutes, seconds, fraction = match.groups()
    millis = int(fraction.ljust(3, "0")[:3])
    return int(hours) * 3_600_000 + int(minutes) * 60_000 + int(seconds) * 1000 + millis


def _ms_to_timestamp(milliseconds: int, is_vtt: bool) -> str:
    milliseconds = max(0, milliseconds)
    hours = milliseconds // 3_600_000
    minutes = (milliseconds % 3_600_000) // 60_000
    seconds = (milliseconds % 60_000) // 1000
    millis = milliseconds % 1000
    separator = "." if is_vtt else ","
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{millis:03d}"


def _strip_colors(text: str) -> str:
    """Remove ASS override colors, ``<font [color]>`` tags and WebVTT class/voice tags."""
    text = _ASS_COLOR_TAG_REGEX.sub("", text)
    text = _FONT_TAG_REGEX.sub("", text)
    text = _VTT_CLASS_TAG_REGEX.sub("", text)
    return _VTT_VOICE_TAG_REGEX.sub("", text)


def _balance_html_tags(text: str, *, keep_font_tags: bool = False) -> str:
    """
    Balance i/b/u tags and drop unsupported/broken tags (keeping inner text).

    ``<font>`` wrappers are preserved untouched when ``keep_font_tags`` is True so
    ``clean_tags`` never strips colors on its own; color removal is owned by the
    separate ``strip_colors`` option.
    """
    pieces: list[str] = []
    stack: list[str] = []
    cursor = 0

    for match in _ANY_TAG_REGEX.finditer(text):
        pieces.append(text[cursor : match.start()])
        raw_tag = match.group(0)
        normalized = raw_tag.strip().lower()
        closing = normalized.startswith("</")
        inner = normalized[1:-1].strip().lstrip("/")
        name = inner.split()[0] if inner else ""

        if name == "font" and keep_font_tags:
            # Preserve color/font wrappers verbatim (colors are opt-in via strip_colors).
            pieces.append(raw_tag)
        elif name in _ALLOWED_TAG_NAMES:
            if closing:
                if stack and stack[-1] == name:
                    stack.pop()
                    pieces.append(f"</{name}>")
                # Stray closing tag -> drop it.
            else:
                stack.append(name)
                pieces.append(f"<{name}>")
        # Other unsupported tags are dropped, inner text is kept.
        cursor = match.end()

    pieces.append(text[cursor:])
    result = "".join(pieces)
    for name in reversed(stack):
        result += f"</{name}>"
    return result


def _normalize_arabic_commas(line: str) -> str:
    """Convert a Latin comma following Arabic text into an Arabic comma (``،``)."""
    return _LATIN_COMMA_IN_ARABIC_REGEX.sub(r"\1، ", line)


def _clean_dialogue_line(line: str, options: CleanOptions) -> str:
    if options.clean_commas:
        line = _normalize_arabic_commas(line)
    if options.clean_spacing:
        line = _MULTISPACE_REGEX.sub(" ", line)
    if options.clean_symbols:
        line = _LOOSE_DOUBLE_DASH_REGEX.sub("...", line)
    if options.clean_spacing:
        line = _SPACE_BEFORE_PUNCT_REGEX.sub(r"\1", line)
    return line.strip()


def _clean_cue_text(text_lines: list[str], options: CleanOptions) -> list[str]:
    merged = "\n".join(text_lines)
    if options.strip_colors:
        merged = _strip_colors(merged)
    if options.clean_symbols:
        merged = _BR_TAG_REGEX.sub("\n", merged)
    if options.clean_tags:
        # Never strip <font> here: color removal is solely owned by strip_colors.
        merged = _balance_html_tags(merged, keep_font_tags=not options.strip_colors)
    cleaned = [_clean_dialogue_line(part, options) for part in merged.split("\n")]
    return [line for line in cleaned if line]


def clean_subtitle(content: str, options: CleanOptions | None = None) -> str:
    """
    Apply the selected "Fix common errors" cleanups to SRT/WebVTT subtitles:

    - ``clean_tags``: repair unbalanced ``<i>/<b>/<u>`` and drop unsupported tags
      (e.g. legacy ``<font color=...>``) while keeping their inner text,
    - ``strip_colors``: remove ASS ``{\\c&H...&}``, ``<font ...>`` and WebVTT
      class tags,
    - ``clean_spacing``: collapse duplicate spaces, remove spaces before punctuation,
    - ``clean_symbols``: convert loose ``--`` into ``...`` and drop redundant
      ``<br>`` breaks,
    - ``clean_commas``: convert Latin commas in Arabic text into ``،``,
    - ``clean_timing``: clamp negligible (< 500ms) cue overlaps to the next start.

    Empty cues are dropped and SRT cue numbers re-indexed. Non-SRT/VTT (e.g. ASS)
    input, and content with nothing to fix, is returned unchanged.
    """
    if not content or "-->" not in content or _looks_like_ass(content):
        return content

    options = options or CleanOptions()
    if not options.any_text_cleaning and not options.clean_timing:
        return content

    newline = "\r\n" if "\r\n" in content else "\n"
    normalised = content.replace("\r\n", "\n")
    had_trailing_newline = normalised.endswith("\n")

    is_vtt = normalised.lstrip()[:6].upper().startswith("WEBVTT")

    header_blocks: list[str] = []
    cues: list[dict] = []
    seen_cue = False

    for block in _BLANK_LINE_REGEX.split(normalised.strip("\n")):
        if not block.strip():
            continue
        lines = block.split("\n")
        timestamp_index = next(
            (i for i, line in enumerate(lines) if _TIMESTAMP_LINE_REGEX.match(line.strip())),
            None,
        )

        if timestamp_index is None:
            if not seen_cue:
                header_blocks.append(block.strip("\n"))
            continue

        seen_cue = True
        match = _TIMESTAMP_LINE_REGEX.match(lines[timestamp_index].strip())
        assert match is not None  # guaranteed by the search above
        start_ms = _timestamp_to_ms(match.group("start"))
        end_ms = _timestamp_to_ms(match.group("end"))
        cues.append(
            {
                "start": start_ms,
                "end": end_ms,
                "rest": match.group("rest"),
                "text": lines[timestamp_index + 1 :],
            }
        )

    if not cues:
        return content

    changed = False

    # 1. Clean cue text and drop cues that become empty.
    cleaned_cues: list[dict] = []
    for cue in cues:
        if options.any_text_cleaning:
            cleaned_text = _clean_cue_text(cue["text"], options)
        else:
            cleaned_text = list(cue["text"])
        if cleaned_text != cue["text"]:
            changed = True
        if not cleaned_text:
            changed = True
            continue
        cue["text"] = cleaned_text
        if cue["start"] is None or cue["end"] is None:
            cue["start"] = cue["start"] or 0
            cue["end"] = cue["end"] or 0
        cleaned_cues.append(cue)

    # 2. Clamp negligible overlaps to the following cue's start.
    if options.clean_timing:
        for index in range(len(cleaned_cues) - 1):
            current = cleaned_cues[index]
            following = cleaned_cues[index + 1]
            overlap = current["end"] - following["start"]
            if 0 < overlap < _MAX_OVERLAP_MS:
                current["end"] = following["start"]
                changed = True

    if not changed:
        return content

    if not cleaned_cues:
        return ""

    rendered: list[str] = []
    index = 1
    for cue in cleaned_cues:
        block_lines: list[str] = []
        if not is_vtt:
            block_lines.append(str(index))
            index += 1
        start_ts = _ms_to_timestamp(cue["start"], is_vtt)
        end_ts = _ms_to_timestamp(cue["end"], is_vtt)
        block_lines.append(f"{start_ts} --> {end_ts}{cue['rest']}")
        block_lines.extend(cue["text"])
        rendered.append("\n".join(block_lines))

    sections = header_blocks + rendered
    result = "\n\n".join(section.strip("\n") for section in sections if section is not None)
    if had_trailing_newline:
        result += "\n"
    return result.replace("\n", newline)


def clean_subtitle_syntax(content: str) -> str:
    """Backward-compatible wrapper enabling every cleaning toggle (colors off)."""
    return clean_subtitle(content, CleanOptions())


def clean_subtitle_bytes(content: bytes, options: CleanOptions | None = None) -> bytes:
    """UTF-8 byte wrapper around :func:`clean_subtitle`."""
    if not content:
        return content
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return content
    return clean_subtitle(text, options).encode("utf-8")


def clean_subtitle_syntax_bytes(content: bytes) -> bytes:
    """UTF-8 byte wrapper around :func:`clean_subtitle_syntax` (all toggles on)."""
    return clean_subtitle_bytes(content, CleanOptions())


def _is_hi_segment(inner: str) -> bool:
    """Heuristically decide whether bracketed text is a non-dialogue HI cue."""
    stripped = inner.strip()
    if not stripped:
        return False
    if _HAS_ARABIC_REGEX.search(stripped):
        return any(keyword in stripped for keyword in _HI_ARABIC_KEYWORDS)
    if not _HAS_LATIN_LETTER_REGEX.search(stripped):
        return False
    # Latin descriptions are written in ALL CAPS (e.g. "MUSIC PLAYING", "GUNSHOT").
    return _LATIN_UPPERCASE_REGEX.match(stripped) is not None and not _HAS_LOWERCASE_REGEX.search(
        stripped
    )


def _strip_hi_from_line(line: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        return "" if _is_hi_segment(match.group(0)[1:-1]) else match.group(0)

    line = _BRACKET_REGEX.sub(_replace, line)
    line = _SPEAKER_LABEL_REGEX.sub("", line)
    return _MULTISPACE_REGEX.sub(" ", line).strip()


def strip_hi_artifacts(content: str) -> str:
    """
    Remove in-dialogue Hearing-Impaired artifacts from SRT/WebVTT subtitles:

    - bracketed sound descriptions such as ``[MUSIC PLAYING]``, ``(GUNSHOT)``,
      ``[صوت بكاء]`` or ``(موسيقى حزينة)``,
    - leading speaker labels such as ``JOHN:`` or ``سارة:``.

    Empty cues are dropped and timestamps/line structure are preserved. Non-SRT/VTT
    input, and content with nothing to strip, is returned unchanged.
    """
    if not content or "-->" not in content or _looks_like_ass(content):
        return content

    newline = "\r\n" if "\r\n" in content else "\n"
    normalised = content.replace("\r\n", "\n")
    had_trailing_newline = normalised.endswith("\n")

    header_blocks: list[str] = []
    cue_blocks: list[str] = []
    seen_cue = False
    changed = False

    for block in _BLANK_LINE_REGEX.split(normalised.strip("\n")):
        if not block.strip():
            continue
        lines = block.split("\n")
        timestamp_index = next(
            (i for i, line in enumerate(lines) if _TIMESTAMP_LINE_REGEX.match(line.strip())),
            None,
        )

        if timestamp_index is None:
            if not seen_cue:
                header_blocks.append(block.strip("\n"))
            continue

        seen_cue = True
        original_text = lines[timestamp_index + 1 :]
        cleaned_text = [_strip_hi_from_line(line) for line in original_text]
        cleaned_text = [line for line in cleaned_text if line]

        if cleaned_text != original_text:
            changed = True
        if not cleaned_text:
            # Cue held only HI artifacts -> drop it entirely.
            continue

        cue_blocks.append("\n".join(lines[: timestamp_index + 1] + cleaned_text))

    if not changed:
        return content

    sections = header_blocks + cue_blocks
    result = "\n\n".join(section.strip("\n") for section in sections)
    if had_trailing_newline:
        result += "\n"
    return result.replace("\n", newline)


def strip_hi_artifacts_bytes(content: bytes) -> bytes:
    """UTF-8 byte wrapper around :func:`strip_hi_artifacts`."""
    if not content:
        return content
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return content
    return strip_hi_artifacts(text).encode("utf-8")


def _digit_run_is_latin_context(text: str, start: int, end: int) -> bool:
    """
    True when a digit run belongs to a Latin/alphanumeric token and must be kept.

    Covers glued identifiers (``AK-47``, ``MP4``, ``S01E01``, ``S01_E01``),
    hyphen/underscore-joined codes (``RTX-4090``, ``AR-15``) and
    space-separated brand/model numbers (``Windows 11``, ``Boeing 747``,
    ``Error 404``).
    """
    left = start
    while left > 0 and text[left - 1] in _LATIN_TOKEN_CHARS:
        left -= 1
    right = end
    while right < len(text) and text[right] in _LATIN_TOKEN_CHARS:
        right += 1
    if _ASCII_LETTER_REGEX.search(text[left:right]):
        return True

    index = start - 1
    while index >= 0 and text[index].isspace():
        index -= 1
    if index >= 0 and _ASCII_LETTER_REGEX.match(text[index]):
        return True

    index = end
    while index < len(text) and text[index].isspace():
        index += 1
    return index < len(text) and _ASCII_LETTER_REGEX.match(text[index]) is not None


def _convert_segment_numerals(segment: str) -> str:
    """Convert Western digits to Eastern Arabic numerals in one un-tagged segment."""
    if not _HAS_ARABIC_REGEX.search(segment) or not _DIGIT_RUN_REGEX.search(segment):
        return segment

    def _replace(match: re.Match[str]) -> str:
        if _digit_run_is_latin_context(segment, match.start(), match.end()):
            return match.group(0)
        return match.group(0).translate(_WESTERN_TO_EASTERN_DIGITS)

    return _DIGIT_RUN_REGEX.sub(_replace, segment)


def _convert_numerals_in_text(text: str) -> str:
    """Convert digits outside protected HTML/ASS tags."""
    pieces: list[str] = []
    cursor = 0
    for match in _PROTECTED_TAG_REGEX.finditer(text):
        pieces.append(_convert_segment_numerals(text[cursor : match.start()]))
        pieces.append(match.group(0))
        cursor = match.end()
    pieces.append(_convert_segment_numerals(text[cursor:]))
    return "".join(pieces)


def _convert_ass_numerals(content: str) -> str:
    newline = "\r\n" if "\r\n" in content else "\n"
    normalised = content.replace("\r\n", "\n")
    changed = False
    lines: list[str] = []
    for line in normalised.split("\n"):
        if _ASS_DIALOGUE_LINE_REGEX.match(line):
            fields = line.split(",", 9)
            if len(fields) == 10:
                converted = _convert_numerals_in_text(fields[9])
                if converted != fields[9]:
                    changed = True
                    line = ",".join(fields[:9]) + "," + converted
        lines.append(line)
    if not changed:
        return content
    return "\n".join(lines).replace("\n", newline)


def convert_eastern_arabic_numerals(content: str) -> str:
    """
    Convert Western digits (0-9) to Eastern Arabic numerals (٠-٩) in Arabic dialogue.

    - Only lines/fields containing Arabic text are touched; cue indices and
      timestamps (including milliseconds) are never modified.
    - HTML (``<font ...>``) and ASS (``{\\c&H...&}``) tags are left untouched.
    - Glued Latin/alphanumeric tokens are kept: ``AK-47``, ``AR-15``,
      ``M1911``, ``MP4``, ``4K``, ``1080p``, ``RTX-4090``, ``50cal``,
      ``S01_E01``.
    - Space-separated Latin context is kept: ``Boeing 747``, ``Error 404``,
      ``Page 12``, ``Apollo 11``.
    - Plain numbers in Arabic context convert: counts, years/dates,
      percentages (``95%``), prices, durations and standalone times
      (``5:30``).
    """
    if not content:
        return content
    if _looks_like_ass(content):
        return _convert_ass_numerals(content)
    if "-->" not in content:
        return content

    newline = "\r\n" if "\r\n" in content else "\n"
    normalised = content.replace("\r\n", "\n")
    had_trailing_newline = normalised.endswith("\n")

    changed = False
    sections: list[str] = []
    for block in _BLANK_LINE_REGEX.split(normalised.strip("\n")):
        if not block.strip():
            continue
        lines = block.split("\n")
        timestamp_index = next(
            (i for i, line in enumerate(lines) if _TIMESTAMP_LINE_REGEX.match(line.strip())),
            None,
        )
        if timestamp_index is None:
            sections.append(block.strip("\n"))
            continue

        original_text = lines[timestamp_index + 1 :]
        converted_text = [_convert_numerals_in_text(line) for line in original_text]
        if converted_text != original_text:
            changed = True
        sections.append("\n".join(lines[: timestamp_index + 1] + converted_text))

    if not changed:
        return content

    result = "\n\n".join(section.strip("\n") for section in sections)
    if had_trailing_newline:
        result += "\n"
    return result.replace("\n", newline)


def convert_eastern_arabic_numerals_bytes(content: bytes) -> bytes:
    """UTF-8 byte wrapper around :func:`convert_eastern_arabic_numerals`."""
    if not content:
        return content
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return content
    return convert_eastern_arabic_numerals(text).encode("utf-8")


def strip_advertisements_bytes(content: bytes, keep_translator_credits: bool = True) -> bytes:
    """UTF-8 byte wrapper around :func:`strip_advertisements`."""
    if not content:
        return content
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return content
    return strip_advertisements(text, keep_translator_credits).encode("utf-8")
