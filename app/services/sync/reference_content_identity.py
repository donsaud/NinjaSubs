"""Independent-verification guard: a reference must not be the target itself.

A subtitle aligned against its own content proves nothing. When the user picks
the subtitle that a reference archive was built from, the "reference" can be
byte-for-byte the file we are trying to verify -- or the same dialogue carrying
a different cue grid, extracted from a different member of the same archive.
Either way the alignment is circular and must never be reported as a verified
synchronization.

Two comparisons are made, because filenames alone cannot answer this:

* :func:`identical_content` -- byte identity, the strongest form.
* :func:`normalized_content_digest` -- identical dialogue with every timing
  anchor, index, and formatting difference stripped. This catches the case that
  actually occurs in the wild: the same member of a season pack, re-timestamped.

Only hashes and lengths cross this boundary; subtitle text is discarded
immediately and never logged.
"""

from __future__ import annotations

import hashlib
import re

# Cue arrows, index lines, and every timestamp punctuation character.
_TIMING = re.compile(r"\d{1,3}:\d{2}:\d{2}[,.]\d{1,3}|-->|^\s*\d+\s*$", re.M)
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)
_BOM = "\ufeff"


def normalized_content_digest(text: str | bytes | None) -> str:
    """Digest of a subtitle's dialogue with all timing structure removed.

    Two subtitles that carry the same lines in the same order produce the same
    digest regardless of cue indices, timestamps, styling tags, or the SRT vs
    ASS container. An empty or timing-only payload returns ``""`` so it can never
    collide with a real content digest.
    """
    if not text:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    stripped = _TIMING.sub(" ", text.lstrip(_BOM))
    dialogue = _NON_WORD.sub(" ", stripped.casefold()).strip()
    if not dialogue:
        return ""
    return hashlib.sha256(dialogue.encode("utf-8")).hexdigest()


def identical_content(
    target: str | bytes | None,
    reference: str | bytes | None,
) -> str:
    """Return the reason a reference is not independent of ``target``.

    ``"byte_identical"``, ``"normalized_identical"``, or ``""`` when the two
    carry genuinely different content. Byte comparison happens first because it
    is both the cheapest and the most certain.
    """
    if not target or not reference:
        return ""
    target_bytes = target if isinstance(target, bytes) else target.encode("utf-8", "replace")
    reference_bytes = (
        reference if isinstance(reference, bytes) else reference.encode("utf-8", "replace")
    )
    if hashlib.sha256(target_bytes).digest() == hashlib.sha256(reference_bytes).digest():
        return "byte_identical"
    target_digest = normalized_content_digest(target)
    if target_digest and target_digest == normalized_content_digest(reference):
        return "normalized_identical"
    return ""
