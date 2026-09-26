"""Archive extractor with Zip Slip defense and Arabic character transcoding."""

import difflib
import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Archive format signatures. ``tar`` is detected by the POSIX ``ustar`` magic at
# byte 257; ``gzip`` (e.g. .tar.gz) by its two-byte header.
_ZIP_MAGIC = b"PK\x03\x04"
_RAR_MAGIC = b"Rar!\x1a\x07"
_SEVENZ_MAGIC = b"7z\xbc\xaf\x27\x1c"
_GZIP_MAGIC = b"\x1f\x8b"
_TAR_MAGIC_OFFSET = 257
_TAR_MAGIC = b"ustar"

# Subtitle file extensions accepted from an archive.
_SUBTITLE_EXTENSIONS = (".srt", ".ass", ".ssa", ".vtt")

# Markers that prove a decoded payload is a real subtitle (never binary/HTML).
_SUBTITLE_MARKERS = ("-->", "WEBVTT", "[Script Info]", "Dialogue:")


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


# Arabic script ranges used to validate a candidate decoding.
_ARABIC_SCRIPT_REGEX = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)

# Maximum share of replacement characters tolerated before a UTF-8 repair
# attempt is rejected in favour of the legacy single-byte decodings.
_UTF8_REPAIR_TOLERANCE = 0.10


def _is_arabic_lang(lang: str | None) -> bool:
    return (lang or "").strip().lower().startswith("ar")


def decode_subtitle_bytes(raw_bytes: bytes, lang: str | None = None) -> str:
    """Decode downloaded subtitle bytes to text. Never raises.

    Priority order: ``utf-8-sig``/``utf-8`` (must be valid), UTF-16 (BOM only),
    ``cp1256`` (Windows Arabic), ``iso-8859-6``, and finally a ``replace``
    fallback. For Arabic subtitles an extra repair step runs first: if strict
    UTF-8 fails but a ``replace`` decode yields real Arabic script with only a
    few damaged bytes, that repair wins — otherwise a single corrupt byte
    would flip the whole file into single-byte mojibake.
    """
    if not raw_bytes:
        return ""

    # Unambiguous byte-order marks win over every heuristic.
    if raw_bytes.startswith(b"\xff\xfe"):
        try:
            return raw_bytes.decode("utf-16")
        except UnicodeDecodeError:
            pass
    elif raw_bytes.startswith(b"\xfe\xff"):
        try:
            return raw_bytes.decode("utf-16-be")
        except UnicodeDecodeError:
            pass

    # 1. UTF-8 (BOM-aware). Strict: any invalid sequence falls through.
    try:
        return raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass

    # 2. Arabic repair: mostly-valid UTF-8 with a few damaged bytes.
    if _is_arabic_lang(lang):
        repaired = raw_bytes.decode("utf-8", errors="replace")
        damage = repaired.count("�") / max(1, len(repaired))
        if damage < _UTF8_REPAIR_TOLERANCE and _ARABIC_SCRIPT_REGEX.search(repaired):
            return repaired

    # 3. Legacy single-byte Arabic encodings, cp1256 first.
    for encoding in ("cp1256", "iso-8859-6"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue

    # 4. Last resort: cp1256 with replacement keeps Arabic readable; this
    # cannot raise, so some text is always returned.
    return raw_bytes.decode("cp1256", errors="replace")


def transcode_to_utf8(raw_bytes: bytes, lang: str | None = None) -> bytes:
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
    return decode_subtitle_bytes(raw_bytes, lang=lang).encode("utf-8")


def extract_srt_from_zip(
    zip_bytes: bytes,
    target_filename: str | None = None,
    season: int | None = None,
    episode: int | None = None,
    lang: str | None = None,
) -> bytes:
    """
    Extract best matching subtitle (.srt, .ass, .ssa) file from a ZIP archive entirely in-memory.
    Preserves ASS/SSA files in their native format without conversion.

    Args:
        zip_bytes: Raw binary bytes of the ZIP archive.
        target_filename: Optional video release name for similarity scoring.
        season: Optional season number for episode matching.
        episode: Optional episode number for episode matching.
        lang: Optional ISO-639-2 language code; enables Arabic-aware decoding.

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
        members = [entry.filename for entry in zf.infolist()][:20]
        raise SubtitleExtractionError(
            "No valid subtitle files (.srt, .ass, .ssa, .vtt) found in the archive. "
            f"Members: {members}"
        )

    # If single entry, extract and transcode directly
    if len(valid_entries) == 1:
        raw_sub = zf.read(valid_entries[0])
        return transcode_to_utf8(raw_sub, lang=lang)

    # If multiple entries exist, select the best candidate
    best_entry = _select_best_srt(
        valid_entries,
        target_filename=target_filename,
        season=season,
        episode=episode,
    )
    raw_sub = zf.read(best_entry)
    return transcode_to_utf8(raw_sub, lang=lang)


def detect_archive_kind(data: bytes) -> str | None:
    """Classify a download by magic bytes: zip / rar / 7z / tar / gzip.

    Returns ``None`` for a raw (non-archive) payload such as a bare .srt.
    """
    if not data:
        return None
    if data[:4] == _ZIP_MAGIC:
        return "zip"
    if data[:4] == b"Rar!" or data[:7] == _RAR_MAGIC:
        return "rar"
    if data[:6] == _SEVENZ_MAGIC:
        return "7z"
    if data[_TAR_MAGIC_OFFSET : _TAR_MAGIC_OFFSET + 5] == _TAR_MAGIC:
        return "tar"
    if data[:2] == _GZIP_MAGIC:
        return "gzip"
    return None


def looks_like_subtitle(data: bytes) -> bool:
    """True when a decoded payload contains real subtitle markup.

    Guards the non-archive path: a binary archive, an HTML error page, a
    provider notice (.txt), or an image must never be cached or served as a
    subtitle.
    """
    if not data or not data.strip():
        return False
    text = data.decode("utf-8", "replace")
    return any(marker in text for marker in _SUBTITLE_MARKERS)


def extract_subtitle_from_archive(
    raw_bytes: bytes,
    target_filename: str | None = None,
    season: int | None = None,
    episode: int | None = None,
    lang: str | None = None,
) -> bytes:
    """Extract the best subtitle from a ZIP/RAR/7z/tar/gzip archive (in-memory).

    ZIP is handled natively; other formats are unpacked with ``bsdtar``.
    Raises :class:`SubtitleExtractionError` for an unknown format, an
    unextractable archive, or one with no recognized subtitle file.
    """
    kind = detect_archive_kind(raw_bytes)
    if kind == "zip":
        return extract_srt_from_zip(
            raw_bytes,
            target_filename=target_filename,
            season=season,
            episode=episode,
            lang=lang,
        )
    if kind in ("rar", "7z", "tar", "gzip"):
        members = _extract_nonzip_members(raw_bytes, kind)
        if not members:
            raise SubtitleExtractionError(
                f"No valid subtitle files (.srt, .ass, .ssa, .vtt) found in the {kind} archive."
            )
        _, payload = _select_nonzip_member(members, season, episode)
        return transcode_to_utf8(payload, lang=lang)
    raise SubtitleExtractionError("Unrecognised archive format")


def _extract_nonzip_members(raw_bytes: bytes, kind: str) -> list[tuple[str, bytes]]:
    """Extract subtitle members from a RAR/7z/tar/gzip archive via ``bsdtar``."""
    suffix = {"rar": ".rar", "7z": ".7z", "tar": ".tar", "gzip": ".tar.gz"}.get(kind, ".bin")
    tool = (
        shutil.which("bsdtar")
        or shutil.which("unrar")
        or shutil.which("7z")
        or shutil.which("7za")
    )
    if tool is None:
        raise SubtitleExtractionError(
            f"No extractor available for {kind} archives (install bsdtar/unrar)"
        )

    tool_name = Path(tool).name.lower()
    with tempfile.TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / f"archive{suffix}"
        archive_path.write_bytes(raw_bytes)
        out_dir = Path(tmp) / "out"
        out_dir.mkdir()
        if tool_name.startswith("unrar"):
            command = [tool, "x", "-y", str(archive_path), str(out_dir) + os.sep]
        elif tool_name.startswith("7z"):
            command = [tool, "x", "-y", f"-o{out_dir}", str(archive_path)]
        else:  # bsdtar
            command = [tool, "-x", "-f", str(archive_path), "-C", str(out_dir)]
        try:
            proc = subprocess.run(command, capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as exc:
            raise SubtitleExtractionError(f"Failed to extract {kind} archive: {exc}") from exc
        if proc.returncode != 0:
            detail = (proc.stderr or b"").decode("utf-8", "replace").strip()[:200]
            raise SubtitleExtractionError(
                f"Failed to extract {kind} archive (exit {proc.returncode}): {detail}"
            )
        members: list[tuple[str, bytes]] = []
        for path in out_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in _SUBTITLE_EXTENSIONS:
                members.append((str(path.relative_to(out_dir)), path.read_bytes()))
        return members


def _select_nonzip_member(
    members: list[tuple[str, bytes]],
    season: int | None,
    episode: int | None,
) -> tuple[str, bytes]:
    """Pick the best episode/season-matching member, else the largest .srt."""
    from app.services.sync.matching import _member_episode_number, _season_number

    def _key(item: tuple[str, bytes]) -> tuple[int, int, int, int]:
        name, data = item
        candidate_episode = _member_episode_number(name)
        candidate_season = _season_number(name)
        season_bad = (
            1
            if (season is not None and candidate_season is not None and candidate_season != season)
            else 0
        )
        episode_bad = (
            1
            if (episode is not None and candidate_episode is not None and candidate_episode != episode)
            else 0
        )
        not_srt = 0 if name.lower().endswith(".srt") else 1
        return (season_bad, episode_bad, not_srt, -len(data))

    return sorted(members, key=_key)[0]


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
