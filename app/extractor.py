"""In-memory ZIP extractor with Zip Slip defense and Arabic character transcoding."""

import difflib
import io
import os
import re
import zipfile
from pathlib import Path


class SubtitleExtractionError(Exception):
    """Raised when subtitle extraction fails."""

    pass


def is_zip_slip_attempt(filename: str) -> bool:
    """
    Detect Zip Slip path traversal vulnerability attempts.
    Returns True if the filename attempts to traverse out of target or is absolute.
    """
    # Check for absolute paths (POSIX or Windows style)
    if os.path.isabs(filename):
        return True
    if re.match(r"^[a-zA-Z]:", filename):
        return True

    # Normalize path parts and look for directory traversal '..'
    normalized_parts = Path(filename).parts
    if ".." in normalized_parts:
        return True

    # Also check raw string for suspicious traversal patterns
    if "../" in filename or "..\\" in filename:
        return True

    return False


def is_ass_subtitle(raw_bytes: bytes) -> bool:
    """Detect if raw bytes represent ASS or SSA subtitle format."""
    if not raw_bytes:
        return False
    sample = raw_bytes[:4096].lower()
    return (
        b"[script info]" in sample
        or b"[v4+ styles]" in sample
        or b"[v4 styles]" in sample
        or b"dialogue:" in sample
    )


def is_vtt_subtitle(raw_bytes: bytes) -> bool:
    """Detect if raw bytes represent WebVTT subtitle format."""
    if not raw_bytes:
        return False
    sample = raw_bytes[:64].strip()
    return sample.startswith(b"WEBVTT") or sample.startswith(b"\xef\xbb\xbfWEBVTT")


def transcode_to_utf8(raw_bytes: bytes) -> bytes:
    """
    Transcode subtitle content to clean UTF-8.
    Handles standard UTF-8, UTF-8-BOM, Arabic legacy encodings (CP1256, ISO-8859-6).
    Preserves ASS/SSA, SRT, and VTT subtitles in their native format without conversion.

    The optional RTL normalization pass (``fix_rtl_punctuation``) is intentionally
    applied later, at serve time, so it can be gated per user via ``enable_rtl_fix``
    and so the on-disk cache always stores the raw original text.
    """
    if not raw_bytes:
        return b""

    text: str | None = None

    # 1. Try UTF-8 with BOM removal
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass

    # 2. Try Windows-1256 (standard Arabic Windows encoding)
    if text is None:
        try:
            text = raw_bytes.decode("cp1256")
        except UnicodeDecodeError:
            pass

    # 3. Try ISO-8859-6 (Arabic standard)
    if text is None:
        try:
            text = raw_bytes.decode("iso-8859-6")
        except UnicodeDecodeError:
            pass

    # 4. Fallback with replacement to avoid crash
    if text is None:
        text = raw_bytes.decode("latin1", errors="replace")

    return text.encode("utf-8")


def extract_srt_from_zip(
    zip_bytes: bytes,
    target_filename: str | None = None,
    season: int | None = None,
    episode: int | None = None,
) -> bytes:
    """
    Extract best matching subtitle (.srt, .ass, .ssa) file from a ZIP archive entirely in-memory.
    Preserves ASS/SSA files in their native format without conversion.

    Args:
        zip_bytes: Raw binary bytes of ZIP archive.
        target_filename: Optional video release name for similarity scoring.
        season: Optional season number for episode matching.
        episode: Optional episode number for episode matching.

    Returns:
        UTF-8 encoded bytes of the extracted subtitle file.

    Raises:
        SubtitleExtractionError: If archive is invalid or contains no safe subtitle files.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except (zipfile.BadZipFile, Exception) as e:
        raise SubtitleExtractionError(f"Corrupt or invalid ZIP archive: {e}") from e

    valid_entries: list[zipfile.ZipInfo] = []

    for entry in zf.infolist():
        # Ignore directory entries
        if entry.is_dir():
            continue

        filename = entry.filename

        # Zip Slip Protection: discard any path traversal or absolute path entries
        if is_zip_slip_attempt(filename):
            continue

        # Ignore macOS and system metadata
        if "__MACOSX" in filename or Path(filename).name.startswith("._"):
            continue

        # Accept .srt, .ass, .ssa, and .vtt files
        f_lower = filename.lower()
        if not (
            f_lower.endswith(".srt")
            or f_lower.endswith(".ass")
            or f_lower.endswith(".ssa")
            or f_lower.endswith(".vtt")
        ):
            continue

        valid_entries.append(entry)

    if not valid_entries:
        raise SubtitleExtractionError("No valid subtitle files (.srt, .ass, .ssa, .vtt) found in the archive.")

    # If single entry, extract and transcode directly
    if len(valid_entries) == 1:
        raw_sub = zf.read(valid_entries[0])
        return transcode_to_utf8(raw_sub)

    # If multiple entries exist, select the best candidate
    best_entry = _select_best_srt(
        valid_entries,
        target_filename=target_filename,
        season=season,
        episode=episode,
    )
    raw_sub = zf.read(best_entry)
    return transcode_to_utf8(raw_sub)


def _select_best_srt(
    entries: list[zipfile.ZipInfo],
    target_filename: str | None = None,
    season: int | None = None,
    episode: int | None = None,
) -> zipfile.ZipInfo:
    """Score and pick the most appropriate subtitle file from multiple entries."""
    target_ext = Path(target_filename).suffix.lower() if target_filename else ""

    if episode is not None:
        # Priority 1: Match entry via metadata extraction (supports 01, 02, 2, S01E02, Anime - 02, etc.)
        from app.services.subtitle_matcher import extract_metadata

        matching_entries: list[zipfile.ZipInfo] = []
        for entry in entries:
            f_name = Path(entry.filename).name
            m = extract_metadata(f_name)
            m_ep = m.get("episode")
            m_abs = m.get("absolute_episode")
            m_eps = m.get("episodes") or set()
            m_s = m.get("season")
            if season is not None and m_s is not None and m_s != season:
                continue
            if m_ep == episode or m_abs == episode or episode in m_eps:
                matching_entries.append(entry)

        if matching_entries:

            def _matching_key(e: zipfile.ZipInfo):
                e_ext = Path(e.filename).suffix.lower()
                ext_match = 1 if (target_ext and e_ext == target_ext) else (1 if e_ext == ".srt" else 0)
                sim = 0.0
                if target_filename:
                    sim = difflib.SequenceMatcher(
                        None, Path(target_filename).stem.lower(), Path(e.filename).stem.lower()
                    ).ratio()
                return (ext_match, sim, e.file_size)

            matching_entries.sort(key=_matching_key, reverse=True)
            return matching_entries[0]

        # Priority 2: Fallback regex patterns for episode e.g. 02, 2, [02], - 02, S01E02
        ep_patterns = [
            rf"(?i)(?:^|[._\-\s\[#])0*{episode}(?:$|[._\-\s\]#])",
            rf"(?i)e0*{episode}\b",
            rf"(?i)ep0*{episode}\b",
        ]
        if season is not None:
            ep_patterns.insert(0, rf"(?i)s0*{season}[._\-\s]*e0*{episode}\b")
            ep_patterns.insert(1, rf"(?i){season}x0*{episode}\b")

        regex_matches: list[zipfile.ZipInfo] = []
        for entry in entries:
            name_lower = Path(entry.filename).name.lower()
            if any(re.search(pat, name_lower) for pat in ep_patterns):
                regex_matches.append(entry)

        if regex_matches:

            def _regex_key(e: zipfile.ZipInfo):
                e_ext = Path(e.filename).suffix.lower()
                ext_match = 1 if (target_ext and e_ext == target_ext) else (1 if e_ext == ".srt" else 0)
                return (ext_match, e.file_size)

            regex_matches.sort(key=_regex_key, reverse=True)
            return regex_matches[0]

    if target_filename:
        target_stem = Path(target_filename).stem.lower()

        def score_entry(e: zipfile.ZipInfo) -> tuple[float, float, int]:
            e_ext = Path(e.filename).suffix.lower()
            ext_match = 1.0 if (target_ext and e_ext == target_ext) else 0.0
            entry_stem = Path(e.filename).stem.lower()
            sim = difflib.SequenceMatcher(None, target_stem, entry_stem).ratio()
            return (ext_match, sim, e.file_size)

        entries.sort(key=score_entry, reverse=True)
        return entries[0]

    # Default fallback: prefer target extension if any, else .srt, then largest file
    def _default_key(e: zipfile.ZipInfo):
        e_ext = Path(e.filename).suffix.lower()
        ext_match = 1 if (target_ext and e_ext == target_ext) else (1 if e_ext == ".srt" else 0)
        return (ext_match, e.file_size)

    entries.sort(key=_default_key, reverse=True)
    return entries[0]
