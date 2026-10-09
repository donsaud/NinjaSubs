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
_LEADING_PUNCTUATION = frozenset(".!?؟,:;…،؛۔")

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
# Inner-wrapper repair is deliberately narrower than the historical outer-prefix rule.
_INNER_TERMINAL_PUNCTUATION = frozenset(".!?؟")
_INNER_WRAPPERS = _QUOTE_WRAPPERS + (("(", ")"), ("[", "]"))
_SIMPLE_EMPHASIS_REGEX = re.compile(
    r"(?P<opening><(?P<tag>[ib])>)(?P<body>[^<>]*)(?P<closing></(?P=tag)>)"
)

# ASS/SSA inline override tags: {\i1}, {\pos(1,2)}, etc.
_LEADING_TAGS_REGEX = re.compile(r"^(?:\{[^{}]*\})+")
_ASS_TAG_REGEX = re.compile(r"\{[^{}]*\}")
_ASS_DIALOGUE_REGEX = re.compile(r"^(?:Dialogue|Comment):", re.IGNORECASE)

# Inverted leading quote + terminal punctuation (editor glitches), e.g. `".نص`, `."نص`.
_INVERTED_LEADING_REGEX = re.compile(
    r'^(?:'
    r'(?P<quote1>["\u201c\u201d\u00ab\u00bb])(?P<punct1>[.!?\u2026])'
    r"|(?P<punct2>[.!?\u2026])(?P<quote2>[\"\u201c\u201d\u00ab\u00bb])"
    r")(?=[\u0600-\u06FF])"
)
# HTML/`<font>` and ASS `{...}` tags that may wrap a dialogue line.
_ANY_TAG_PART = r"(?:\{[^{}]*\}|<[^<>]*>)"
_LEADING_ANY_TAG_REGEX = re.compile(rf"^(?:{_ANY_TAG_PART})+")
_ANY_TAG_REGEX = re.compile(_ANY_TAG_PART)
_FORMATTING_TAG_REGEX = re.compile(r"<[^<>]*>|\{\\[^{}]*\}")
# A line qualified for the inverted-leading-quote fix must end on an Arabic letter
# (so properly quoted lines ending in a quote/ellipsis keep their existing handling).
_ENDS_WITH_ARABIC_LETTER_REGEX = re.compile(r"[\u0621-\u064a][\u064b-\u0652]*$")


def _bidi_class(char: str) -> str:
    try:
        return unicodedata.bidirectional(char)
    except (TypeError, ValueError):
        return ""


def contains_rtl(text: str) -> bool:
    """Return True if the text contains any strong Right-to-Left character."""
    return any(_bidi_class(ch) in _RTL_BIDI_CLASSES for ch in text)


def _split_trailing_tags(text: str, *, ass_only: bool = False) -> tuple[str, str]:
    """Find a consecutive suffix in one scan, without retrying tag runs."""
    pattern = _ASS_TAG_REGEX if ass_only else _ANY_TAG_REGEX
    start = end = 0
    for match in pattern.finditer(text):
        if match.start() != end:
            start = match.start()
        end = match.end()
    return (text[:start], text[start:]) if end == len(text) and end else (text, "")


def _arabic_dialogue_only(text: str) -> bool:
    """Require actual Arabic letters and refuse mixed-script relocation.

    Formatting attributes, RTL punctuation and Arabic digits alone are not
    evidence of legacy dialogue. This is an admission guard, not language or
    author-intent detection.
    """
    visible = _FORMATTING_TAG_REGEX.sub("", text)
    letters = [char for char in visible if unicodedata.category(char).startswith("L")]
    return bool(letters) and all("ARABIC" in unicodedata.name(char, "") for char in letters)


def _legacy_punctuation_body(text: str) -> bool:
    """Arabic-only body with no already-correct terminal punctuation.

    First/last Arabic letters and wrappers constrain the existing prefix
    heuristic. A leading pause on an already-terminated line stays untouched.
    """
    if not _arabic_dialogue_only(text):
        return False
    visible = _FORMATTING_TAG_REGEX.sub("", text).strip().strip(RLM).strip()
    visible = visible.lstrip("([{\"'«“ ").rstrip(")]}\"'»” ")
    visible = visible.rstrip(RLM).rstrip()
    visible = visible.rstrip("\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652")
    return bool(visible) and all(
        unicodedata.category(char).startswith("L") and "ARABIC" in unicodedata.name(char, "")
        for char in (visible[0], visible[-1])
    )


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
    # Word-span heuristics do not distinguish tag attributes from dialogue.
    # Refuse ambiguous bracket repairs rather than risk damaging markup.
    if _FORMATTING_TAG_REGEX.search(core) or not _arabic_dialogue_only(core):
        return core
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
    if (
        not any(char in ".…" for char in cluster)
        or not _arabic_dialogue_only(core)
        or _FORMATTING_TAG_REGEX.search(core)
    ):
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


def _relocate_inner_terminal(text: str) -> str:
    """Move one terminal prefix in plain Arabic; never recurse into wrappers."""
    stripped = text.strip()
    if not stripped or stripped[0] not in _INNER_TERMINAL_PUNCTUATION:
        return text
    body = stripped[1:]
    # Nested wrappers, markup, numbers and controls are ambiguous here. The
    # existing Arabic-only/first-last-letter guard also rejects mixed scripts,
    # already-terminated dialogue and leading ellipses/multiple punctuation.
    if (
        any(char in _BRACKET_CHARS | _QUOTE_CHARS or char in "<>\\" for char in body)
        or any(unicodedata.category(char)[0] in {"N", "C"} for char in body)
        or not _legacy_punctuation_body(body)
    ):
        return text
    leading_ws = text[: len(text) - len(text.lstrip())]
    trailing_ws = text[len(text.rstrip()) :]
    return f"{leading_ws}{body}{stripped[0]}{trailing_ws}"


def _repair_wrapped_body(inner: str) -> str:
    """Repair plain content or one attribute-free i/b emphasis layer only."""
    if "<" not in inner and ">" not in inner:
        return _relocate_inner_terminal(_normalize_pre_reversed_dialogue(inner))
    stripped = inner.strip()
    prefix = stripped[:1] if stripped[:1] in _INNER_TERMINAL_PUNCTUATION else ""
    emphasis = _SIMPLE_EMPHASIS_REGEX.fullmatch(stripped[len(prefix):])
    if emphasis is None:
        return inner
    content = prefix + emphasis.group("body")
    repaired = _relocate_inner_terminal(_normalize_pre_reversed_dialogue(content))
    if repaired == content:
        return inner
    leading_ws = inner[: len(inner) - len(inner.lstrip())]
    trailing_ws = inner[len(inner.rstrip()) :]
    return (
        f"{leading_ws}{emphasis.group('opening')}{repaired}"
        f"{emphasis.group('closing')}{trailing_ws}"
    )


def _normalize_wrapped_punctuation(core: str) -> str | None:
    """Handle one outer wrapper; return None when there is no supported wrapper.

    Recognized but ambiguous content is returned unchanged so legacy quote and
    bracket heuristics cannot reinterpret it. Curly braces, arbitrary nesting,
    attributes and non-emphasis tags are deliberately outside this repair rule.
    """
    leading_ws = core[: len(core) - len(core.lstrip())]
    trailing_ws = core[len(core.rstrip()) :]
    stripped = core.strip()
    marker = RLM if stripped.endswith(RLM) else ""
    if marker:
        stripped = stripped[:-1]
    for opener, closer in _INNER_WRAPPERS:
        if len(stripped) >= 2 and stripped.startswith(opener) and stripped.endswith(closer):
            inner = stripped[len(opener):len(stripped) - len(closer)]
            repaired = _repair_wrapped_body(inner)
            return f"{leading_ws}{opener}{repaired}{closer}{marker}{trailing_ws}"
    if _SIMPLE_EMPHASIS_REGEX.fullmatch(stripped):
        return f"{leading_ws}{_repair_wrapped_body(stripped)}{marker}{trailing_ws}"
    return None


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
        body = match.group(2).strip()
        if _legacy_punctuation_body(body):
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
        if not _legacy_punctuation_body(dot_prefix.group(1)):
            return core
        body = dot_prefix.group(1).strip()
        body = f"{body}."
    elif not (_arabic_dialogue_only(body) and body.endswith(tuple(_LEADING_PUNCTUATION))):
        # New unpunctuated dialogue form: require a separated, single marker
        # and plain Arabic-only text. Numbers, markup and nested wrappers are
        # too ambiguous to relocate. Preserve the legacy punctuated rule above.
        if (
            not text[:match.start(2)].endswith(tuple(" \t"))
            or not _arabic_dialogue_only(body)
            or any(char in _DIALOGUE_DASH_CHARS or char in _BRACKET_CHARS | _QUOTE_CHARS
                   or char in "<>" or unicodedata.category(char)[0] in {"N", "C"}
                   for char in body)
            or not _legacy_punctuation_body(body)
        ):
            return core

    if not body or not contains_rtl(body):
        return core

    return f"{leading_ws}{dash} {body}{trailing_ws}"


def fix_inverted_leading_quote(line: str) -> str:
    """
    Repair lines whose leading quote + terminal punctuation were inverted by an
    editor glitch, e.g. ``".نادِني بـ"عزيزتي`` -> ``نادِني بـ"عزيزتي".\\u200F``.

    Guardrails (return the line unchanged):
    - dialogue lines starting with ``-`` or the tatweel ``ـ`` are never touched;
    - a legitimate standalone leading quote (``"أهلاً" قال الرجل.``) is preserved,
      because the quote must be *immediately* followed by terminal punctuation
      and an Arabic letter to qualify;
    - leading/trailing HTML (``<font>``, ``<i>``) and ASS (``{\\pos...}``) tags are
      isolated and restored verbatim;
    - lines that do not exactly match the inverted pattern are left as-is.

    A Right-to-Left Mark (RLM) is appended to anchor the relocated punctuation in
    BiDi-unaware players.
    """
    if not line or not contains_rtl(line):
        return line

    leading_tags = ""
    trailing_tags = ""
    core = line

    leading_match = _LEADING_ANY_TAG_REGEX.match(core)
    if leading_match:
        leading_tags = leading_match.group(0)
        core = core[len(leading_tags) :]

    core, trailing_tags = _split_trailing_tags(core)

    if core.lstrip().startswith(("-", "\u0640")):
        return line

    match = _INVERTED_LEADING_REGEX.match(core)
    if not match:
        return line

    quote = match.group("quote1") or match.group("quote2")
    punct = match.group("punct1") or match.group("punct2")
    body = core[match.end() :]
    if not body or not _arabic_dialogue_only(body) or _FORMATTING_TAG_REGEX.search(body):
        return line

    # Only relocate when the line ends on an Arabic letter. Lines ending in a
    # quote/ellipsis are legitimate quoted phrases handled by the quote normalizer.
    if not _ENDS_WITH_ARABIC_LETTER_REGEX.search(core.rstrip()):
        return line

    # Avoid doubling the closing quote when the body already ends with one.
    closing_quote = "" if body.endswith(quote) else quote
    return f"{leading_tags}{body}{closing_quote}{punct}{trailing_tags}{RLM}"


def _normalize_core(core: str) -> str:
    """Normalize brackets/quotes, move legacy leading punctuation, apply the RLM suffix."""
    if not contains_rtl(core):
        return core

    wrapped = _normalize_wrapped_punctuation(core)
    if wrapped is not None:
        return _append_trailing_rlm(wrapped)

    core = _normalize_brackets(core)
    core = _relocate_leading_quote(core)

    run = _leading_movable_run(core)
    if run:
        remainder = core[len(run) :].lstrip()
        paired_inversion = (
            len(run) == 1
            and run in _CLOSING_TO_OPENING
            and remainder.rstrip().endswith(_CLOSING_TO_OPENING[run])
            and _legacy_punctuation_body(remainder.rstrip()[:-1])
        )
        markup_bracket = bool(
            _FORMATTING_TAG_REGEX.search(core) and any(char in _BRACKET_CHARS for char in run)
        )
        if not markup_bracket and (paired_inversion or _legacy_punctuation_body(remainder)):
            stripped_remainder = remainder.rstrip()
            trailing_whitespace = remainder[len(stripped_remainder) :]

            # Paired inversion, e.g. ")مرحبا(" -> "(مرحبا)".
            if paired_inversion:
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


_SAFE_DIALOGUE_TAG = re.compile(
    r"</?(?:[ibu]|font)\b[^<>]*>", re.IGNORECASE
)
_SAFE_FONT_OPEN = re.compile(r"<font\s+color\s*=\s*([\"'])#[0-9a-f]{6}\1\s*>", re.IGNORECASE)
_TRANSPARENT_BIDI = frozenset({RLM, "\u200e"})
_DIALOGUE_PAIRS = {"(": ")", "[": "]", '"': '"', "'": "'", "«": "»", "“": "”", "‹": "›"}


def _safe_dialogue_tag(text: str, cursor: int) -> tuple[str, bool, int] | None:
    """Shared token grammar for safe emphasis and generated color markup."""
    match = _SAFE_DIALOGUE_TAG.match(text, cursor)
    if match is None:
        return None
    token = match.group().lower()
    closing = token.startswith("</")
    name = token[2 if closing else 1:].split(">", 1)[0].split()[0]
    if token not in {f"<{name}>", f"</{name}>"} and not (
        name == "font" and not closing and _SAFE_FONT_OPEN.fullmatch(token)
    ):
        return None
    return name, closing, match.end()


def _dialogue_structure(text: str) -> tuple[bool, str, dict[int, tuple[int, int, int]]]:
    """Validate wrapper nesting and derive visible text without rewriting markup.

    Maps each opening offset to (content start, closing start, closing end),
    allowing an existing inside-wrapper dialogue marker to stay inside it.
    Unknown HTML/attributes are conservatively refused.
    """
    stack: list[tuple[str, int, int]] = []
    spans: dict[int, tuple[int, int, int]] = {}
    visible: list[str] = []
    cursor = 0
    while cursor < len(text):
        char = text[cursor]
        if char in _TRANSPARENT_BIDI:
            cursor += 1
            continue
        if char == "<":
            tag = _safe_dialogue_tag(text, cursor)
            if tag is None:
                return False, "", spans
            name, closing, tag_end = tag
            if closing:
                if not stack or stack[-1][0] != f"</{name}>":
                    return False, "", spans
                _, start, content_start = stack.pop()
                spans[start] = content_start, cursor, tag_end
            else:
                stack.append((f"</{name}>", cursor, tag_end))
            cursor = tag_end
            continue
        if char == ">":
            return False, "", spans
        if stack and char == stack[-1][0]:
            _, start, content_start = stack.pop()
            spans[start] = content_start, cursor, cursor + 1
        elif char in _DIALOGUE_PAIRS:
            stack.append((_DIALOGUE_PAIRS[char], cursor, cursor + 1))
        elif char in _DIALOGUE_PAIRS.values():
            return False, "", spans
        visible.append(char)
        cursor += 1
    return not stack, "".join(visible), spans


def _structural_dialogue(core: str) -> str | None:
    """Handle separated markers before legacy punctuation heuristics can guess.

    None delegates non-dialogue input to existing punctuation repair. Any
    malformed/ambiguous dialogue structure is preserved, with bidi anchoring
    only. A single marker can move within an existing whole-line wrapper.
    """
    transparent = "".join(char for char in core if char not in _TRANSPARENT_BIDI)
    if not any(char in _DIALOGUE_DASHES for char in transparent):
        return None
    balanced, visible, spans = _dialogue_structure(core)
    if not balanced:
        return _append_trailing_rlm(core)
    # Keep existing special leading-dot repair, constrained by its old guard.
    legacy = _normalize_pre_reversed_dialogue(core)
    if legacy != core and core.strip().startswith((".", "- .", "– .", "— .")):
        return _append_trailing_rlm(_isolate_leading_dash(legacy))
    has_arabic = any(unicodedata.category(char).startswith("L")
                     and "ARABIC" in unicodedata.name(char, "") for char in visible)
    if not has_arabic:
        return core
    start, end = 0, len(core)

    def trim(left: int, right: int) -> tuple[int, int]:
        while left < right and (core[left].isspace() or core[left] in _TRANSPARENT_BIDI):
            left += 1
        while right > left and (core[right - 1].isspace() or core[right - 1] in _TRANSPARENT_BIDI):
            right -= 1
        return left, right

    start, end = trim(start, end)
    while start < end:
        if core[end - 1] in _DIALOGUE_DASHES:
            break
        span = spans.get(start)
        if span is None or span[2] != end:
            break
        start, end = trim(span[0], span[1])
    marker = core[end - 1] if end > start else ""
    # Refuse ranges, glued hyphens, internal separators, or multiple markers.
    if marker in _DIALOGUE_DASHES and sum(char in _DIALOGUE_DASHES for char in visible) == 1:
        separator = end - 2
        while separator >= start and core[separator] in _TRANSPARENT_BIDI:
            separator -= 1
        if separator >= start and core[separator].isspace():
            body_end = end - 1
            while body_end > start and core[body_end - 1].isspace():
                body_end -= 1
            body = core[start:body_end]
            fixed = core[:start] + _isolate_leading_dash(marker + " " + body) + core[end:]
            return _append_trailing_rlm(_isolate_leading_dash(fixed))
    # Correct leading markers and internal/ambiguous separators retain structure.
    return _append_trailing_rlm(_isolate_leading_dash(core))


def _legacy_visible_text(core: str) -> tuple[str, list[int], bool]:
    """Project safe markup to visible offsets; never repair or rewrite tags.

    Unpaired boundary tags are permitted for the evidenced orphan emphasis
    forms. Crossed tags, unknown attributes and interior orphan tags fail closed.
    """
    visible: list[str] = []
    offsets: list[int] = []
    stack: list[tuple[str, bool]] = []
    seen_text = False
    orphan_suffix = False
    index = 0
    while index < len(core):
        char = core[index]
        if char == "{" and core[index + 1:index + 2] == "\\":
            override = _ASS_TAG_REGEX.match(core, index)
            if override is None:
                return "", [], False
            index = override.end()
            continue
        if char == "<":
            tag = _safe_dialogue_tag(core, index)
            if tag is None:
                return "", [], False
            name, closing, tag_end = tag
            if closing:
                if stack:
                    if stack[-1][0] != name:
                        return "", [], False
                    stack.pop()
                else:
                    orphan_suffix = True
            else:
                if orphan_suffix:
                    return "", [], False
                stack.append((name, seen_text))
            index = tag_end
            continue
        if char == ">":
            return "", [], False
        if char not in _TRANSPARENT_BIDI:
            if not char.isspace():
                if orphan_suffix:
                    return "", [], False
                seen_text = True
            visible.append(char)
            offsets.append(index)
        index += 1
    return "".join(visible), offsets, not any(interior for _, interior in stack)


def _legacy_visual_order(core: str, preserve_ellipsis: bool = False) -> tuple[str, bool]:
    """Normalize evidenced two-marker mirrors; flag unsafe legacy candidates.

    The flag prevents a declined candidate from reaching old generic bracket
    guesses. Existing balanced text and punctuation-only repairs are unaffected.
    """
    if "(" not in core and ")" not in core:
        return core, False
    visible, offsets, safe = _legacy_visible_text(core)
    if not safe:
        return core, True
    imbalance = visible.count("(") - visible.count(")")
    if imbalance == 0:
        return core, False
    if imbalance not in {-2, 2}:
        return core, True
    left, right = 0, len(visible)

    def trim(start: int, end: int) -> tuple[int, int]:
        while start < end and visible[start].isspace():
            start += 1
        while end > start and visible[end - 1].isspace():
            end -= 1
        return start, end

    left, right = trim(left, right)
    if right - left >= 2 and visible[right - 1] in _DIALOGUE_DASHES:
        if not visible[right - 2].isspace() or sum(char in _DIALOGUE_DASHES for char in visible) != 1:
            return core, True
        left, right = trim(left, right - 1)
    quote_chars = frozenset("\"'«»“”‹›")
    quotes = [(index, char) for index, char in enumerate(visible[left:right], left)
              if char in quote_chars]
    if quotes:
        if (len(quotes) != 2 or quotes[0][0] != left or quotes[1][0] != right - 1
                or _DIALOGUE_PAIRS.get(quotes[0][1]) != quotes[1][1]):
            return core, True
        left, right = trim(left + 1, right - 1)
    body = visible[left:right]
    if not any(unicodedata.category(char).startswith("L")
               and "ARABIC" in unicodedata.name(char, "") for char in body):
        return core, True
    punctuation = next((token for token in ("...", "…", "،") if body.startswith(token)), "")
    punctuation_start = left
    left, right = trim(left + len(punctuation), right)
    if left >= right:
        return core, True
    body = visible[left:right]
    if imbalance == 2:
        if (not body.startswith("(") or punctuation == "،"
                or not body[1:].lstrip()
                or unicodedata.category(body[1:].lstrip()[0])[0] not in {"L", "N"}):
            return core, True
        proposal = body[1:] + ")"
    else:
        if (not body.endswith(")") or body[:-1].rstrip().endswith(")")
                or unicodedata.category(body[0])[0] not in {"L", "N"}):
            return core, True
        proposal = "(" + body[:-1]
    balanced, _, spans = _dialogue_structure(proposal)
    depth = 0
    for char in proposal:
        if char == "(":
            depth += 1
            if depth > 1:
                return core, True
        elif char == ")":
            depth -= 1
    if not balanced or any(
        not any(unicodedata.category(char)[0] in {"L", "N"}
                for char in proposal[start:end])
        for opening, (start, end, _) in spans.items() if proposal[opening] == "("
    ):
        return core, True
    # The strict dialogue rule owns ellipsis placement after parenthesis repair.
    if preserve_ellipsis and punctuation in {"...", "…"}:
        punctuation = ""
    # Offset edits place new punctuation inside the original style boundary.
    edits = {offsets[index]: "" for index in range(punctuation_start,
                                                  punctuation_start + len(punctuation))}
    inserts: dict[int, str] = {}
    if imbalance == 2:
        edits[offsets[left]] = ""
        terminal = next((token for token in ("...", "…") if body.endswith(token)), "")
        end = right
        if punctuation and terminal:
            end -= len(terminal)
            for index in range(end, right):
                edits[offsets[index]] = ""
        inserts[offsets[end - 1] + 1] = ")" + (terminal if punctuation and terminal else punctuation)
    else:
        edits[offsets[right - 1]] = ""
        inserts[offsets[left]] = "("
        if punctuation:
            first_close = visible.index(")", left, right - 1)
            destination = visible[first_close + 1:right - 1].lstrip()
            cluster_end = 0
            while cluster_end < len(destination) and destination[cluster_end] in "،.…":
                cluster_end += 1
            cluster = destination[:cluster_end]
            existing = "،" in cluster if punctuation == "،" else "..." in cluster or "…" in cluster
            if not existing:
                inserts[offsets[first_close] + 1] = punctuation
    pieces: list[str] = []
    for index, char in enumerate(core):
        pieces.append(inserts.get(index, ""))
        pieces.append(edits.get(index, char))
    pieces.append(inserts.get(len(core), ""))
    return "".join(pieces), False


def normalize_legacy_visual_order(text: str) -> str:
    """Pure line-level legacy normalization without bidi anchoring or dash moves."""
    return _legacy_visual_order(text)[0]


def _safe_style_at(text: str, stop: int) -> tuple[str, ...]:
    stack: list[str] = []
    cursor = 0
    while cursor < stop:
        if text[cursor] == "<":
            tag = _safe_dialogue_tag(text, cursor)
            if tag is not None:
                _, closing, tag_end = tag
                if closing:
                    if stack:
                        stack.pop()
                else:
                    stack.append(text[cursor:tag_end].lower())
                cursor = tag_end
                continue
        cursor += 1
    return tuple(stack)


def _ellipsis_dialogue_shape(core: str) -> tuple[str, int, int, list[int]] | None:
    """Recognize exact leading ellipsis and one separated terminal marker."""
    visible, offsets, safe = _legacy_visible_text(core)
    if not safe or "{\\" in core or any(char in "\"'«»“”‹›" for char in visible):
        return None
    left = len(visible) - len(visible.lstrip())
    right = len(visible.rstrip())
    body = visible[left:right]
    token = next((item for item in ("...", "…") if body.startswith(item)), "")
    if not token or body[len(token):].startswith((".", "…")):
        return None
    if (len(body) <= len(token) + 2 or body[-1] not in _DIALOGUE_DASHES
            or not body[-2].isspace() or sum(char in _DIALOGUE_DASHES for char in body) != 1):
        return None
    end = right - 1
    while end > left and visible[end - 1].isspace():
        end -= 1
    if not any(unicodedata.category(char).startswith("L")
               and "ARABIC" in unicodedata.name(char, "")
               for char in visible[left + len(token):end]):
        return None
    if core[offsets[left]:offsets[left] + len(token)] != token:
        return None
    return token, left, end, offsets


def _normalize_dialogue_ellipsis(core: str) -> str:
    """Move only evidenced dialogue ellipsis; dash movement stays downstream."""
    shape = _ellipsis_dialogue_shape(core)
    if shape is None:
        return core
    token, left, end, offsets = shape
    visible, _, _ = _legacy_visible_text(core)
    if not _dialogue_structure(visible[left + len(token):end])[0]:
        return core
    source = offsets[left]
    destination = offsets[end - 1] + 1
    if _safe_style_at(core, source) != _safe_style_at(core, destination):
        return core
    # Preserve existing destination punctuation instead of duplicating it.
    suffix = "" if visible[left:end].endswith(("...", "…")) else token
    content_start = left + len(token)
    while content_start < end and visible[content_start].isspace():
        content_start += 1
    # Delete only the leading token and its whitespace inside one style span.
    removal_end = offsets[content_start]
    if _safe_style_at(core, removal_end) != _safe_style_at(core, source):
        removal_end = source + len(token)
    if "<" in core[source:removal_end] or "{" in core[source:removal_end]:
        removal_end = source + len(token)
    return core[:source] + core[removal_end:destination] + suffix + core[destination:]


def normalize_legacy_dialogue_ellipsis(text: str) -> str:
    """Structural parenthesis + strict dialogue ellipsis pass, without dash moves."""
    shape = _ellipsis_dialogue_shape(text)
    if shape is None:
        return text
    normalized, ambiguous = _legacy_visual_order(text, preserve_ellipsis=True)
    return text if ambiguous else _normalize_dialogue_ellipsis(normalized)


def _quote_boundary(line: str, opening: bool) -> str | None:
    """Relocate one visible quote only within the same safe style context."""
    visible, offsets, safe = _legacy_visible_text(line)
    if not safe or visible.count('"') != 1 or "{\\" in line:
        return None
    left = len(visible) - len(visible.lstrip())
    right = len(visible.rstrip())
    if left >= right:
        return None
    quote = visible.index('"')
    if quote not in {left, right - 1}:
        return None
    remaining = line[:offsets[quote]] + line[offsets[quote] + 1:]
    body, positions, _ = _legacy_visible_text(remaining)
    start = len(body) - len(body.lstrip())
    end = len(body.rstrip())
    if start >= end:
        return None
    destination = positions[start] if opening else positions[end - 1] + 1

    if _safe_style_at(line, offsets[quote]) != _safe_style_at(remaining, destination):
        return None
    return remaining[:destination] + '"' + remaining[destination:]


def _multiline_quote_body(line: str, mirrored_comma: bool, process: bool) -> str:
    """Reuse line-level mirrors while shielding continuation ellipses."""
    if process and _ellipsis_dialogue_shape(line) is not None:
        return _process_segment(line, is_ass=False)
    visible, offsets, _ = _legacy_visible_text(line)
    start = len(visible) - len(visible.lstrip())
    token = next((item for item in ("...", "…") if visible[start:].startswith(item)), "")
    prefix = ""
    if token:
        # Contiguous punctuation only; retain its original style boundary.
        position = offsets[start]
        if line[position:position + len(token)] != token:
            return line
        prefix = line[:position + len(token)]
        line = line[position + len(token):]
    visible, offsets, _ = _legacy_visible_text(line)
    start = len(visible) - len(visible.lstrip())
    end = len(visible.rstrip())
    if mirrored_comma and visible[start:end].startswith("،") and (
        visible.count("(") == visible.count(")")
        and _dialogue_structure(visible[start + 1:end])[0]
    ):
        position = offsets[start]
        suffix = "" if visible[start:end].endswith("،") else "،"
        insertion = offsets[end - 1] + 1
        line = line[:position] + line[position + 1:insertion] + suffix + line[insertion:]
    line = normalize_legacy_visual_order(line)
    if process:
        line = _process_segment(line, is_ass=False)
    return prefix + line


def _multiline_quote_pair(first: str, second: str, process: bool = False) -> tuple[str, str] | None:
    projections = [_legacy_visible_text(line) for line in (first, second)]
    if any(not safe or visible.count('"') != 1 or not any(
        unicodedata.category(char).startswith("L") and "ARABIC" in unicodedata.name(char, "")
        for char in visible
    ) for visible, _, safe in projections):
        return None
    a, b = (projection[0].strip() for projection in projections)
    reversed_quotes = a.endswith('"') and b.startswith('"')
    modern_quotes = a.startswith('"') and b.endswith('"')
    if not reversed_quotes and not (process and modern_quotes):
        return None
    moved = (_quote_boundary(first, True), _quote_boundary(second, False))
    if any(line is None for line in moved):
        return None
    result: list[str] = []
    for index, line in enumerate(moved):
        assert line is not None
        visible, offsets, _ = _legacy_visible_text(line)
        quote = offsets[visible.index('"')]
        body = line[:quote] + line[quote + 1:]
        body = _multiline_quote_body(body, reversed_quotes and index == 0, process)
        visible, offsets, _ = _legacy_visible_text(body)
        start = len(visible) - len(visible.lstrip())
        end = len(visible.rstrip())
        destination = offsets[start] if index == 0 else offsets[end - 1] + 1
        restored = body[:destination] + '"' + body[destination:]
        result.append(_append_trailing_rlm(restored) if process else restored)
    return result[0], result[1]


_SRT_CUE_TIME = re.compile(
    r"[0-9]{2,}:[0-9]{2}:[0-9]{2},[0-9]{3} --> "
    r"[0-9]{2,}:[0-9]{2}:[0-9]{2},[0-9]{3}"
)


def _srt_quote_lines(text: str, process: bool) -> tuple[list[str], set[int]]:
    """Select adjacent text lines inside complete numbered SRT blocks only."""
    lines = text.split("\n")
    protected: set[int] = set()
    cursor = 0
    while cursor < len(lines):
        end = cursor
        while end < len(lines) and lines[end].strip():
            end += 1
        if (end - cursor >= 4 and lines[cursor].lstrip("\ufeff").strip().isdigit()
                and _SRT_CUE_TIME.fullmatch(lines[cursor + 1].rstrip("\r"))):
            index = cursor + 2
            while index + 1 < end:
                endings = ["\r" if lines[i].endswith("\r") else "" for i in (index, index + 1)]
                pair = _multiline_quote_pair(lines[index].rstrip("\r"),
                                             lines[index + 1].rstrip("\r"), process)
                if pair is not None:
                    for offset, body in enumerate(pair):
                        lines[index + offset] = body + endings[offset]
                        protected.add(index + offset)
                    index += 2
                else:
                    index += 1
        cursor = end + 1
    return lines, protected


def normalize_legacy_multiline_quotes(text: str) -> str:
    """Pure SRT cue-level quote normalization; never cross cue boundaries."""
    return "\n".join(_srt_quote_lines(text, process=False)[0])


def _process_segment(segment: str, is_ass: bool) -> str:
    """Normalize one visual segment (a line, or an ASS ``\\N`` sub-segment)."""
    visible = _FORMATTING_TAG_REGEX.sub("", segment)
    if not any(
        unicodedata.category(char).startswith("L") and _bidi_class(char) in {"R", "AL"}
        for char in visible
    ):
        return segment

    leading_tags = ""
    trailing_tags = ""
    core = segment

    if is_ass:
        leading_match = _LEADING_TAGS_REGEX.match(core)
        if leading_match:
            leading_tags = leading_match.group(0)
            core = core[len(leading_tags) :]

        core, trailing_tags = _split_trailing_tags(core, ass_only=True)

    core, ambiguous_legacy = _legacy_visual_order(
        core, preserve_ellipsis=_ellipsis_dialogue_shape(core) is not None
    )
    if ambiguous_legacy:
        return f"{leading_tags}{_append_trailing_rlm(core)}{trailing_tags}"
    core = _normalize_dialogue_ellipsis(core)
    structural = _structural_dialogue(core)
    if structural is not None:
        return f"{leading_tags}{structural}{trailing_tags}"
    core = fix_inverted_leading_quote(core)
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
    cue_lines, protected = _srt_quote_lines(text, process=True)
    for index, raw_line in enumerate(cue_lines):
        if index in protected:
            fixed_lines.append(raw_line)
            continue
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
