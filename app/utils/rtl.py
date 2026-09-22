"""Unicode Bidirectional Algorithm (UBA) normalization for RTL subtitles.

Three related problems are handled here:

1. **Trailing punctuation flipping (UBA base direction).**
   With a Left-to-Right base direction, a trailing bidi-neutral character
   (period, ellipsis, question/exclamation mark, comma, ...) at the end of an
   otherwise RTL line inherits the paragraph direction (UBA rules N1/N2) and
   visually jumps to the far *right*. Appending a Right-to-Left Mark (RLM,
   U+200F) after it gives it a strong-RTL neighbour so it resolves to RTL and
   stays at the visual far left.

2. **Legacy "Reverse RTL start/end" hacks.**
   Some translators prepended trailing punctuation to the start of the Arabic
   string (e.g. ``".والدي كان مزارعاً"``). Leading neutral punctuation/symbols are
   moved to the logical end and inverted paired brackets are mirrored back.

3. **Mirrored / orphaned parentheses and brackets.**
   Legacy files frequently contain manually flipped bracket glyphs, including
   lines whose two boundary brackets use the *same* direction
   (``"(نص("`` / ``")نص)"``) or an embedded bracket replacing the real closing
   bracket (``"(آسفة ... (كوبر"``). Brackets are rebalanced and mirrored using the
   Unicode ``Bidi_Mirrored`` counterparts.

This module operates on decoded subtitle text and is safe to run on SRT, WebVTT,
and ASS/SSA content. It is idempotent.
"""

import re
import unicodedata

# Right-to-Left Mark: a zero-width strong-RTL character.
RLM = "\u200f"

# Bidi classes that are direction-neutral (can be misresolved at line end).
_NEUTRAL_BIDI_CLASSES = frozenset({"ON", "CS", "ET", "ES"})

# Strong RTL bidi classes.
_RTL_BIDI_CLASSES = frozenset({"R", "AL", "AN"})

# Dialogue dashes that open a spoken line (``- مرحبا``).
_DIALOGUE_DASHES = frozenset({"-", "–", "—"})

# Neutral punctuation/symbols that legacy files wrongly prepend.
_LEADING_PUNCTUATION = frozenset(".!?,:;…،؛۔")

# Bidi-mirrored bracket pairs (logical opening -> logical closing).
_BRACKET_PAIRS = {
    "(": ")",
    "[": "]",
    "{": "}",
    "«": "»",
    "“": "”",
}
_OPENING_TO_CLOSING = dict(_BRACKET_PAIRS)
_CLOSING_TO_OPENING = {closing: opening for opening, closing in _BRACKET_PAIRS.items()}
_OPENING_BRACKETS = frozenset(_OPENING_TO_CLOSING)
_CLOSING_BRACKETS = frozenset(_CLOSING_TO_OPENING)
_BRACKET_CHARS = _OPENING_BRACKETS | _CLOSING_BRACKETS
_MOVABLE_LEADING = _LEADING_PUNCTUATION | _CLOSING_BRACKETS

# Quote characters and their pairing counterparts.
_QUOTE_PAIRS = {
    '"': '"',
    "'": "'",
    "“": "”",
    "”": "“",
    "«": "»",
    "»": "«",
}
_QUOTE_CHARS = frozenset(_QUOTE_PAIRS)
_QUOTE_OPENERS = frozenset({"“", "«"})
_QUOTE_CLOSERS = frozenset({"”", "»"})

# Leading cluster of legacy-prepended punctuation/quotes/dots (e.g. ``."``, ``".``).
_LEADING_CLUSTER_REGEX = re.compile(r"^[\s.\"'“”«»…]+")
# Whitespace immediately preceding a trailing run of dots/ellipsis.
_TRAILING_DOTS_SPACE_REGEX = re.compile(r"\s+([.…]+)\s*$")

# Quote wrappers that delimit a fully quoted line (opening -> closing).
_QUOTE_WRAPPERS: tuple[tuple[str, str], ...] = (
    ('"', '"'),
    ("«", "»"),
    ("“", "”"),
)
# Misplaced edge punctuation inside a quoted phrase.
_EDGE_DOTS_LEADING_REGEX = re.compile(r"^[.…]+")
_EDGE_DOTS_TRAILING_REGEX = re.compile(r"[.…]+$")

# ASS/SSA inline override tags: {\i1}, {\pos(1,2)}, etc.
_LEADING_TAGS_REGEX = re.compile(r"^(?:\{[^}]*\})+")
_TRAILING_TAGS_REGEX = re.compile(r"(?:\{[^}]*\})+$")
_ASS_DIALOGUE_REGEX = re.compile(r"^(?:Dialogue|Comment):", re.IGNORECASE)


def _bidi_class(char: str) -> str:
    try:
        return unicodedata.bidirectional(char)
    except (TypeError, ValueError):
        return ""


def contains_rtl(text: str) -> bool:
    """Return True if the text contains any strong Right-to-Left character."""
    return any(_bidi_class(ch) in _RTL_BIDI_CLASSES for ch in text)


# ---------------------------------------------------------------------------
# Bracket normalization
# ---------------------------------------------------------------------------


def _visible_bracket_positions(core: str) -> list[tuple[int, str]]:
    """Bracket positions outside ASS ``{...}`` override blocks."""
    positions: list[tuple[int, str]] = []
    depth = 0
    for index, char in enumerate(core):
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth = max(0, depth - 1)
            continue
        if depth == 0 and char in _BRACKET_CHARS:
            positions.append((index, char))
    return positions


def _brackets_balanced(sequence: list[str]) -> bool:
    """Return True when the bracket sequence is a valid nesting in logical order."""
    stack: list[str] = []
    for char in sequence:
        if char in _OPENING_BRACKETS:
            stack.append(char)
        elif char in _CLOSING_BRACKETS:
            if not stack:
                return False
            if _OPENING_TO_CLOSING.get(stack.pop()) != char:
                return False
    return not stack


_WORD_LETTER_BIDI_CLASSES = frozenset({"L", "R", "AL", "AN", "EN", "NSM"})


def _is_word_letter(char: str) -> bool:
    """A letter/digit/combining-mark character (excludes spaces and punctuation)."""
    return bool(char) and _bidi_class(char) in _WORD_LETTER_BIDI_CLASSES


def _first_word_span(core: str) -> tuple[int, int] | None:
    """Span of the first word (letters only), skipping leading brackets/punctuation."""
    index = 0
    while index < len(core) and not _is_word_letter(core[index]):
        index += 1
    start = index
    while index < len(core) and _is_word_letter(core[index]):
        index += 1
    return (start, index) if index > start else None


def _last_word_span(core: str) -> tuple[int, int] | None:
    """Span of the last word, skipping trailing punctuation, if any."""
    index = len(core.rstrip()) - 1
    while index >= 0 and not _is_word_letter(core[index]):
        index -= 1
    if index < 0:
        return None
    end = index + 1
    while index - 1 >= 0 and _is_word_letter(core[index - 1]):
        index -= 1
    return (index, end)


def _attached_word_span(core: str, index: int, char: str) -> tuple[int, int] | None:
    """Span of the word glued to the bracket (``(كوبر`` / ``كوبر)``), if any."""
    if char in _OPENING_BRACKETS:
        start = index + 1
        end = start
        while end < len(core) and _is_word_letter(core[end]):
            end += 1
        return (start, end) if end > start else None

    end = index
    start = index
    while start - 1 >= 0 and _is_word_letter(core[start - 1]):
        start -= 1
    return (start, end) if end > start else None


def _wrap_word_span(
    core: str,
    bracket_positions: list[tuple[int, str]],
    span: tuple[int, int],
    open_char: str,
) -> str:
    """Drop all brackets and wrap the given word span in ``open_char``...pair."""
    bracket_indices = {index for index, _ in bracket_positions}
    chars: list[str] = []
    new_start: int | None = None
    new_end: int | None = None
    for index, char in enumerate(core):
        if index in bracket_indices:
            continue
        if index == span[0]:
            new_start = len(chars)
        chars.append(char)
        if index == span[1] - 1:
            new_end = len(chars)

    if new_start is None or new_end is None:
        return core

    rebuilt = "".join(chars)
    close_char = _OPENING_TO_CLOSING.get(open_char, ")")
    return f"{rebuilt[:new_start]}{open_char}{rebuilt[new_start:new_end]}{close_char}{rebuilt[new_end:]}"


def _normalize_brackets(core: str) -> str:
    """
    Context-aware repair of legacy-inverted parentheses/brackets.

    - A well-formed nesting is left untouched.
    - A bracket glued to an inner/trailing word is restored strictly around that
      word. The sentence-starting word is never wrapped when a distinct trailing
      entity exists (e.g. ``)آسفة ... يا كوبر`` -> ``آسفة ... يا (كوبر)``).
    - A stray bracket with a distinct trailing entity wraps that trailing entity.
    """
    positions = _visible_bracket_positions(core)
    if not positions:
        return core

    if _brackets_balanced([char for _, char in positions]):
        return core

    first_span = _first_word_span(core)
    last_span = _last_word_span(core)

    # 1. A bracket glued to a non-first word (inner/trailing name or side mention).
    for index, char in positions:
        span = _attached_word_span(core, index, char)
        if span is None:
            continue
        if (
            first_span is not None
            and span == first_span
            and last_span is not None
            and last_span != first_span
        ):
            continue  # never wrap the sentence-starting word when a trailing entity exists
        open_char = char if char in _OPENING_BRACKETS else _CLOSING_TO_OPENING[char]
        return _wrap_word_span(core, positions, span, open_char)

    # 2. A stray bracket plus a distinct trailing entity -> wrap the trailing entity.
    if first_span is not None and last_span is not None and last_span != first_span:
        return _wrap_word_span(core, positions, last_span, "(")

    return core


# ---------------------------------------------------------------------------
# Legacy leading punctuation + RLM
# ---------------------------------------------------------------------------


def _leading_movable_run(core: str) -> str:
    """
    Return the leading run of legacy-prepended punctuation/symbols.

    A leading hyphen is only movable when glued to another punctuation mark
    (e.g. ``-.``); a normal speaker dash (``- مرحبا``) is left intact. Opening
    brackets terminate the run (they are legitimate).
    """
    index = 0
    length = len(core)
    while index < length:
        char = core[index]
        if char in _MOVABLE_LEADING:
            index += 1
            continue
        if char == "-":
            following = core[index + 1] if index + 1 < length else ""
            if following and (following in _MOVABLE_LEADING or following == "-"):
                index += 1
                continue
            break
        break
    return core[:index]


def _mirror_leading_run(run: str) -> str:
    """Mirror closing brackets in a leading run to their RTL opening equivalents."""
    return "".join(_CLOSING_TO_OPENING.get(char, char) for char in run)


def _isolate_leading_dash(core: str) -> str:
    """
    Prepend RLM before a leading dialogue dash so it renders at the visual start.

    With an LTR paragraph base direction a leading ``-`` is bidi-neutral and would
    otherwise be resolved LTR (far left / end of the Arabic sentence). Giving it a
    strong-RTL left neighbour (``\\u200F- مرحبا``) locks it to the visual right.
    """
    index = 0
    while index < len(core) and core[index].isspace():
        index += 1
    if index >= len(core) or core[index] == RLM or core[index] not in _DIALOGUE_DASHES:
        return core

    remainder = core[index + 1 :].lstrip()
    if not remainder or not contains_rtl(remainder):
        return core

    return f"{core[:index]}{RLM}{core[index:]}"


def _append_trailing_rlm(core: str) -> str:
    """Append RLM after a trailing bidi-neutral character (idempotent)."""
    stripped = core.rstrip()
    if stripped and not stripped.endswith(RLM):
        if _bidi_class(stripped[-1]) in _NEUTRAL_BIDI_CLASSES:
            trailing_whitespace = core[len(stripped) :]
            return f"{stripped}{RLM}{trailing_whitespace}"
    return core


def _is_quote_opening(text: str, index: int, char: str) -> bool:
    """Heuristically decide whether a quote opens (before a word) or closes."""
    if char in _QUOTE_OPENERS:
        return True
    if char in _QUOTE_CLOSERS:
        return False
    previous = text[index - 1] if index > 0 else ""
    return not _is_word_letter(previous)


def _quotes_are_balanced(text: str) -> bool:
    """True when all quotes nest as open/close pairs."""
    stack: list[str] = []
    for index, char in enumerate(text):
        if char not in _QUOTE_CHARS:
            continue
        if _is_quote_opening(text, index, char):
            stack.append(char)
        elif not stack:
            return False
        else:
            opener = stack.pop()
            if _QUOTE_PAIRS.get(opener, opener) != char:
                return False
    return not stack


def _pair_inner_quote(text: str) -> str:
    """Close the first unmatched opening quote around the word it introduces."""
    stack: list[tuple[int, str]] = []
    for index, char in enumerate(text):
        if char not in _QUOTE_CHARS:
            continue
        if _is_quote_opening(text, index, char):
            stack.append((index, char))
        elif stack:
            stack.pop()

    if not stack:
        return text

    index, char = stack[0]
    word_end = index + 1
    while word_end < len(text) and _is_word_letter(text[word_end]):
        word_end += 1
    if word_end == index + 1:
        return text

    closing = _QUOTE_PAIRS.get(char, char)
    return f"{text[:word_end]}{closing}{text[word_end:]}"


def _relocate_leading_quote(core: str) -> str:
    """
    Fix legacy leading quote/dot clusters such as ``."حول هراء "أبولو ...``.

    When the line's quotes are unbalanced, the leading quote is treated as the
    missing closing quote for the first unmatched inner quote (paired around its
    word), and leading dots are absorbed by an existing trailing ellipsis or moved
    to the logical end.
    """
    match = _LEADING_CLUSTER_REGEX.match(core)
    if not match:
        return core

    cluster = match.group(0)
    if not any(char in _QUOTE_CHARS for char in cluster):
        return core
    if _quotes_are_balanced(core):
        return core

    rest = core[len(cluster) :]
    if not rest:
        return core

    rest = _pair_inner_quote(rest)

    cluster_dots = "".join(char for char in cluster if char in ".…")
    if cluster_dots:
        stripped_rest = rest.rstrip()
        if not stripped_rest.endswith((".", "…")):
            rest = f"{stripped_rest}{cluster_dots}"

    return rest


def _detect_quote_wrapper(text: str) -> tuple[str, str] | None:
    """Return the (opener, closer) pair when ``text`` is fully wrapped in quotes."""
    for opener, closer in _QUOTE_WRAPPERS:
        if (
            len(text) > len(opener) + len(closer)
            and text.startswith(opener)
            and text.endswith(closer)
        ):
            return opener, closer
    return None


def _normalize_quoted_edges(core: str) -> str:
    """
    Fix misplaced edge punctuation inside a fully quoted Arabic phrase.

    Legacy files render ``".نطلق عليه صندوق السماء"`` where the sentence-final
    period visually flips to the front of the line. The period is moved across
    the closing quote (``"نطلق عليه صندوق السماء".``); the trailing RLM is then
    appended by :func:`_append_trailing_rlm` so the dot stays locked on the left.
    Also normalises the mirrored ``"نص."`` form and ``«...»`` quotation marks.
    """
    leading_ws = core[: len(core) - len(core.lstrip())]
    trailing_ws = core[len(core.rstrip()) :]
    stripped = core.strip()

    wrapper = _detect_quote_wrapper(stripped)
    if wrapper is None:
        return core

    opener, closer = wrapper
    inner = stripped[len(opener) : len(stripped) - len(closer)]
    body = inner.strip()
    if not body:
        return core

    has_leading_dot = _EDGE_DOTS_LEADING_REGEX.match(body) is not None
    has_trailing_dot = _EDGE_DOTS_TRAILING_REGEX.search(body) is not None
    if not (has_leading_dot or has_trailing_dot):
        return core

    body = _EDGE_DOTS_LEADING_REGEX.sub("", body)
    body = _EDGE_DOTS_TRAILING_REGEX.sub("", body).strip()
    if not body or not contains_rtl(body):
        return core

    return f"{leading_ws}{opener}{body}{closer}.{trailing_ws}"


_DIALOGUE_DASH_CHARS = "-\u2013\u2014"
# Trailing dialogue dash: "…نص -" / "…؟ -".
_TRAILING_DASH_REGEX = re.compile(rf"^(.+?)\s*([{_DIALOGUE_DASH_CHARS}])\s*$")
# Leading dash glued to a stray leading dot: "- .نص".
_LEADING_DASH_DOT_REGEX = re.compile(rf"^([{_DIALOGUE_DASH_CHARS}])\s*\.\s*(.+)$")
# Stray leading dot on the dialogue body: ".نص".
_LEADING_DOT_REGEX = re.compile(r"^\.\s*(.+)$")


def _normalize_pre_reversed_dialogue(core: str) -> str:
    """
    Repair legacy manually-reversed dialogue dashes/punctuation.

    Wild SRT files often move the dialogue dash to the end of the line (and
    sometimes the sentence's period to the front) to "compensate" for broken
    players. With native BiDi rendering those appear doubled:

    - ``"نص عربي... -"``      -> ``"- نص عربي..."``
    - ``"هل ...؟ -"``         -> ``"- هل ...؟"``
    - ``".أجل -"``            -> ``"- أجل."``
    - ``"- .أجل"``            -> ``"- أجل."``

    Already-correct leading-dash lines are left untouched.
    """
    if not contains_rtl(core):
        return core

    leading_ws = core[: len(core) - len(core.lstrip())]
    trailing_ws = core[len(core.rstrip()) :]
    text = core.strip()
    if not text:
        return core

    # Pattern B (leading dash + stray leading dot): "- .أجل" -> "- أجل."
    match = _LEADING_DASH_DOT_REGEX.match(text)
    if match:
        body = match.group(2).strip().rstrip(".")
        if body and contains_rtl(body):
            return f"{leading_ws}{match.group(1)} {body}.{trailing_ws}"
        return core

    # Patterns A/C (trailing dash): move it to the front.
    match = _TRAILING_DASH_REGEX.match(text)
    if not match or text.startswith(tuple(_DIALOGUE_DASH_CHARS)):
        return core

    body = match.group(1).strip()
    dash = match.group(2)

    # Pattern B (stray leading dot on the body): ".أجل" -> "أجل."
    dot_prefix = _LEADING_DOT_REGEX.match(body)
    if dot_prefix:
        body = dot_prefix.group(1).strip().rstrip(".")
        body = f"{body}."

    if not body or not contains_rtl(body):
        return core

    return f"{leading_ws}{dash} {body}{trailing_ws}"


def _normalize_core(core: str) -> str:
    """Normalize brackets/quotes, move legacy leading punctuation, apply the RLM suffix."""
    if not contains_rtl(core):
        return core

    core = _normalize_brackets(core)
    # Fix quoted edge punctuation before the legacy leading-quote relocation, which
    # would otherwise misread the trailing quote of ``"نص."`` as an opening quote.
    core = _normalize_quoted_edges(core)
    core = _relocate_leading_quote(core)

    run = _leading_movable_run(core)
    if run:
        remainder = core[len(run) :].lstrip()
        if contains_rtl(remainder):
            stripped_remainder = remainder.rstrip()
            trailing_whitespace = remainder[len(stripped_remainder) :]

            # Paired inversion, e.g. ")مرحبا(" -> "(مرحبا)".
            if (
                len(run) == 1
                and run in _CLOSING_TO_OPENING
                and stripped_remainder
                and stripped_remainder[-1] == _CLOSING_TO_OPENING[run]
            ):
                core = (
                    _CLOSING_TO_OPENING[run]
                    + stripped_remainder[:-1]
                    + _OPENING_TO_CLOSING[stripped_remainder[-1]]
                    + trailing_whitespace
                )
            else:
                # Lone leading punctuation/bracket -> move (and mirror) to the end.
                core = f"{stripped_remainder}{_mirror_leading_run(run)}{trailing_whitespace}"

    core = _TRAILING_DOTS_SPACE_REGEX.sub(r"\1", core)
    core = _isolate_leading_dash(core)
    return _append_trailing_rlm(core)


# ---------------------------------------------------------------------------
# Line / payload processing
# ---------------------------------------------------------------------------


def _process_segment(segment: str, is_ass: bool) -> str:
    """Normalize one visual segment (a line, or an ASS ``\\N`` sub-segment)."""
    if not contains_rtl(segment):
        return segment

    leading_tags = ""
    trailing_tags = ""
    core = segment

    if is_ass:
        leading_match = _LEADING_TAGS_REGEX.match(core)
        if leading_match:
            leading_tags = leading_match.group(0)
            core = core[len(leading_tags) :]

        trailing_match = _TRAILING_TAGS_REGEX.search(core)
        if trailing_match:
            trailing_tags = trailing_match.group(0)
            core = core[: trailing_match.start()]

    core = _normalize_pre_reversed_dialogue(core)
    return f"{leading_tags}{_normalize_core(core)}{trailing_tags}"


def _process_text(text: str, is_ass: bool) -> str:
    """Normalize a full text payload, honouring ASS ``\\N`` hard line breaks."""
    if "\\N" in text:
        return "\\N".join(_process_segment(part, is_ass) for part in text.split("\\N"))
    return _process_segment(text, is_ass)


def fix_rtl_punctuation(text: str) -> str:
    """
    Normalize RTL subtitle text:
    - rebalance/mirror inverted parentheses and brackets,
    - move legacy leading punctuation to the logical end,
    - append RLM after trailing neutral punctuation.

    Handles SRT/WebVTT lines, CRLF endings, ASS ``Dialogue:``/``Comment:`` lines
    (only the text field is touched) and ASS inline override tags. Idempotent.
    """
    if not text or not contains_rtl(text):
        return text

    fixed_lines: list[str] = []
    for raw_line in text.split("\n"):
        carriage_return = ""
        line = raw_line
        if line.endswith("\r"):
            carriage_return = "\r"
            line = line[:-1]

        if not contains_rtl(line):
            fixed_lines.append(raw_line)
            continue

        dialogue_match = _ASS_DIALOGUE_REGEX.match(line)
        if dialogue_match:
            # ASS: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
            fields = line.split(",", 9)
            if len(fields) == 10:
                prefix = ",".join(fields[:9]) + ","
                fixed = f"{prefix}{_process_text(fields[9], is_ass=True)}"
                fixed_lines.append(f"{fixed}{carriage_return}")
                continue

        fixed = _process_text(line, is_ass=False)
        fixed_lines.append(f"{fixed}{carriage_return}")

    return "\n".join(fixed_lines)


def fix_rtl_punctuation_bytes(content: bytes) -> bytes:
    """UTF-8 byte wrapper around :func:`fix_rtl_punctuation`."""
    if not content:
        return content
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return content
    return fix_rtl_punctuation(text).encode("utf-8")
