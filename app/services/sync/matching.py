"""Pure release-name matching primitives for reference selection.

No I/O, no provider knowledge: season/episode parsing, source/codec/
resolution classification, and scene release-group extraction.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from functools import lru_cache
from typing import Any
from urllib.parse import unquote_plus

_guessit: Callable[..., dict[str, Any]] | None = None
try:
    _guessit = importlib.import_module("guessit").guessit
except ImportError:  # pragma: no cover - guessit is a declared dependency
    _guessit = None


@lru_cache(maxsize=2048)
def guess_metadata(name: str | None) -> dict:
    """Parse filename using guessit with LRU cache."""
    if not name or _guessit is None:
        return {}
    try:
        clean = unquote_plus(str(name)).rsplit("/", 1)[-1].strip()
        return dict(_guessit(clean))
    except Exception:
        return {}

# Scene/release tokens that make a filename useful for matching a reference.
_RELEASE_TOKEN_REGEX = re.compile(
    r"(1080p|720p|2160p|480p|4k|web[\s._-]?dl|webrip|blu[\s._-]?ray|brrip|bdrip|hdtv|"
    r"dvdrip|x264|x265|h\.?264|h\.?265|hevc|aac|dts|proper|repack|remux|"
    r"s\d{1,2}[\s._-]?e\d{1,2}|season[\s._-]?\d+|19\d{2}|20\d{2})",
    re.IGNORECASE,
)

_EPISODE_PATTERNS = (
    re.compile(r"(?i)\bS\d{1,2}[\s._-]?E(\d{1,3})\b"),
    re.compile(r"(?i)\bE(?:P)?[\s._-]?(\d{1,3})\b"),
    re.compile(r"(?i)\bEpisode[\s._-]?(\d{1,3})\b"),
    re.compile(r"(?i)\bPart[\s._-]?(\d{1,3})\b"),
    re.compile(r"(?i)\b\d{1,2}x(\d{1,3})\b"),
)
# A bare 1-2 digit number, not preceded by a letter/digit (S08, x265) and not
# followed by a codec/resolution token (1080p).
_BARE_EPISODE_REGEX = re.compile(r"(?i)(?<![A-Za-z0-9])(\d{1,2})(?![\dxp])")
_NON_EPISODE_TOKENS = frozenset(
    {"0", "480", "576", "720", "1080", "2160", "264", "265", "60", "50", "25", "24", "30", "10"}
)
_INFO_TOKENS = ("info", "nfo", "readme", "sample", "trailer", "credit", "cover")


_SEASON_PATTERNS = (
    re.compile(r"(?i)\bS(\d{1,2})[\s._-]?E\d{1,3}\b"),
    re.compile(r"(?i)\bS(\d{1,2})\b"),
    re.compile(r"(?i)\bSeason[\s._-]?(\d{1,2})\b"),
    re.compile(r"(?i)\b(\d{1,2})x\d{1,3}\b"),
)

_WORD_SEASONS: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12,
}
_SEASON_WORD_AFTER_REGEX = re.compile(r"(?i)\bseason[\s._-]?([a-z]+)\b")
_SEASON_WORD_BEFORE_REGEX = re.compile(r"(?i)\b([a-z]+)[\s._-]?season\b")


def _season_number(name: str | None) -> int | None:
    """Best-effort season number parsed from a release/member name.

    Understands ``S08``, ``Season 8``, ``8x01`` and worded seasons
    (``Season One``, ``Season Eight``).
    """
    if not name:
        return None
    meta = guess_metadata(name)
    s = meta.get("season")
    if s is not None:
        try:
            val = int(s[0] if isinstance(s, list) and s else s)
            if 0 < val <= 99:
                return val
        except (ValueError, TypeError):
            pass
    base = (name or "").rsplit("/", 1)[-1]
    for pattern in _SEASON_PATTERNS:
        match = pattern.search(base)
        if match:
            return int(match.group(1))
    # "Season One" (season first) takes priority over "One Season".
    after = _SEASON_WORD_AFTER_REGEX.search(base)
    if after and after.group(1).lower() in _WORD_SEASONS:
        return _WORD_SEASONS[after.group(1).lower()]
    before = _SEASON_WORD_BEFORE_REGEX.search(base)
    if before and before.group(1).lower() in _WORD_SEASONS:
        return _WORD_SEASONS[before.group(1).lower()]
    return None


# Audio channel tags (e.g. 5.1, 7.1, 2.0, 1.0, 5.1.4, 7.1.2) must never be
# parsed as episode numbers.
_AUDIO_CHANNEL_REGEX = re.compile(r"(?<!\d)[1-9]\.[0-2](?:\.[0-4])?(?!\d)")


def _member_episode_number(name: str | None) -> int | None:
    """Best-effort episode number parsed from a ZIP member filename."""
    base = (name or "").rsplit("/", 1)[-1]
    for pattern in _EPISODE_PATTERNS:
        match = pattern.search(base)
        if match:
            return int(match.group(1))
    cleaned = _AUDIO_CHANNEL_REGEX.sub("", base)
    for match in _BARE_EPISODE_REGEX.finditer(cleaned):
        value = int(match.group(1))
        if value > 0 and str(value) not in _NON_EPISODE_TOKENS:
            return value
    return None


def _is_info_member(name: str) -> bool:
    lowered = name.rsplit("/", 1)[-1].lower()
    return any(token in lowered for token in _INFO_TOKENS)


# Release source/quality tokens used to match a reference to the target stream.
_SOURCE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("remux", r"\b(bdremux|remux)\b"),
    ("bluray", r"\b(blu[\s._-]?ray|bdrip|brrip|bd25|bd50)\b"),
    ("webdl", r"\b(web[\s._-]?dl|webrip|web)\b"),
    ("stream", r"\b(nf|netflix|amzn|amazon|dsnp|disney|hmax|max|atvp|appletv|itunes|hulu|pcok)\b"),
    ("hdtv", r"\b(hdtv|pdtv|dsr|satrip)\b"),
    ("dvd", r"\b(dvdrip|dvd|dvdr)\b"),
    ("cam", r"\b(cam|ts|tc|hdts|hdcam)\b"),
)
_RESOLUTION_REGEX = re.compile(r"\b(2160p|1080p|720p|480p|4k)\b", re.IGNORECASE)

# Codec families used for edition-equivalence checks.
_CODEC_PATTERNS: tuple[tuple[str, str], ...] = (
    ("h265", r"\b(x265|h\.?265|hevc)\b"),
    ("h264", r"\b(x264|h\.?264|avc)\b"),
    ("xvid", r"\b(xvid|divx)\b"),
    ("av1", r"\b(av1)\b"),
)


def _source_kind(name: str | None) -> str | None:
    """Classify a release name's source (bluray / webdl / hdtv / ...)."""
    if not name:
        return None
    meta = guess_metadata(name)
    other = meta.get("other")
    if other and ("Remux" in other if isinstance(other, list) else other == "Remux"):
        return "remux"
    src = str(meta.get("source") or "").lower()
    if "blu-ray" in src:
        return "bluray"
    if "web" in src:
        return "webdl"
    if "hdtv" in src:
        return "hdtv"
    if "dvd" in src:
        return "dvd"
    lowered = (name or "").lower()
    for kind, pattern in _SOURCE_PATTERNS:
        if re.search(pattern, lowered):
            return kind
    return None


def _codec_kind(name: str | None) -> str | None:
    """Classify a release name's video codec family (h264 / h265 / ...)."""
    if not name:
        return None
    meta = guess_metadata(name)
    codec = str(meta.get("video_codec") or "").lower()
    if "h.265" in codec or "hevc" in codec:
        return "h265"
    if "h.264" in codec or "avc" in codec:
        return "h264"
    if "xvid" in codec or "divx" in codec:
        return "xvid"
    if "av1" in codec:
        return "av1"
    lowered = (name or "").lower()
    for kind, pattern in _CODEC_PATTERNS:
        if re.search(pattern, lowered):
            return kind
    return None


def _sources_compatible(target_source: str | None, candidate_source: str | None) -> bool:
    """True unless both sources are known and different families.

    Unknowns pass through here (strictness is enforced by the caller);
    BluRay and REMUX count as one family (same retail-disc master).
    """
    if not target_source or not candidate_source:
        return True
    return candidate_source == target_source or {target_source, candidate_source} == {
        "bluray",
        "remux",
    }


# Explicit edition/cut markers. A target and candidate carrying different
# sets (US vs International, theatrical vs Extended) are different timings
# and must never align. Note: "remux" is deliberately absent — it describes
# the encoding method, not the cut (a remux IS the disc content), and the
# source patterns already model the BluRay/REMUX relationship.
_EDITION_TAGS = frozenset(
    {
        "us", "uk", "intl", "international",
        "unrated", "rated", "unextended", "extended", "directors", "theatrical",
        "uncut", "proper", "repack", "rerip",
        # Additional cut/edit markers that change timing or framing.
        "imax", "remastered", "criterion", "hybrid",
    }
)

# Multi-word cut markers. Tokenising on words would split these into generic
# tokens ("special", "open") that cause false positives on ordinary titles, so
# they are matched as phrases and surfaced with an underscore-joined tag key.
_EDITION_PHRASES: tuple[str, ...] = ("final cut", "special edition", "open matte")


def _edition_tags(name: str | None) -> frozenset[str]:
    """Explicit edition markers present in a release name.

    Single-word markers are matched as whole tokens; known multi-word cut
    markers (``final cut``, ``special edition``, ``open matte``) are matched as
    phrases so bare ``special``/``open`` never register as an edition.
    """
    normalized = re.sub(r"[^a-z0-9]+", " ", (name or "").lower())
    tags = {token for token in normalized.split() if token in _EDITION_TAGS}
    for phrase in _EDITION_PHRASES:
        if re.search(r"\b" + phrase.replace(" ", r"\s+") + r"\b", normalized):
            tags.add(phrase.replace(" ", "_"))
    return frozenset(tags)


# Regional audio and distribution markers (MULTI, FRENCH, GERMAN, etc.) that
# frequently carry distinct distributor intros or bumper offsets.
_REGIONAL_TAGS = frozenset(
    {
        "multi", "french", "vff", "truefrench", "vfq",
        "german", "deutsch", "italian", "ita",
        "castellano", "latino", "spanish", "nordic",
    }
)


def _regional_tags(name: str | None) -> frozenset[str]:
    """Regional audio/release markers present in a release name."""
    normalized = re.sub(r"[^a-z0-9]+", " ", (name or "").lower())
    return frozenset(token for token in normalized.split() if token in _REGIONAL_TAGS)


# Retail-disc source families: a BluRay and its REMUX are the same physical
# master (same cut/timing), so they are symmetric for recap-allowance purposes.
_RETAIL_DISC_SOURCES = frozenset({"bluray", "remux"})


def is_retail_disc_source(kind: str | None) -> bool:
    """True when a classified source is a retail disc (BluRay or REMUX)."""
    return kind in _RETAIL_DISC_SOURCES


def _resolution(name: str | None) -> str | None:
    """Normalised resolution token (``1080p``) or ``None``."""
    if not name:
        return None
    meta = guess_metadata(name)
    res = meta.get("screen_size")
    if res:
        res_str = str(res).lower()
        if res_str in ("2160p", "4k", "1080p", "720p", "480p", "576p"):
            return res_str
    match = _RESOLUTION_REGEX.search(name or "")
    return match.group(1).lower() if match else None


# Trailing tokens that are episode markers rather than release groups
# (e.g. the ``E013`` in ``S01E01-E013``).
_EPISODE_LIKE_REGEX = re.compile(r"(?i)^(s\d{1,2}e\d{1,3}|e\d{1,3}|s\d{1,2}|\d{1,2}x\d{1,3})$")

# Tokens that look like a trailing release group but are actually codec,
# source, language, or container markers (e.g. the ``DL`` in ``WEB-DL``).
_NON_GROUP_TOKENS = frozenset(
    {
        "1080p", "720p", "2160p", "480p", "4k", "2160", "1080", "720", "480",
        "web", "dl", "webdl", "webrip", "bluray", "brrip", "bdrip", "hdtv",
        "dvdrip", "dvd", "cam", "x264", "x265", "h264", "h265", "hevc",
        "xvid", "divx", "aac", "ac3", "dts", "dtshd", "dd", "ddp", "eac3",
        "truehd", "atmos", "flac", "mp3", "proper", "repack", "rerip",
        "remux", "unrated", "extended", "complete", "hybrid", "dubbed",
        "subbed", "hi", "sdh", "eng", "en", "ar",
        "srt", "ass", "ssa", "sub", "vtt", "mkv", "mp4", "avi",
    }
)


def _release_group(name: str | None) -> str | None:
    """Extract a scene release-group tag (``FSiHD``, ``mSD``, ``ImE``) or ``None``.

    Understands trailing ``-GROUP`` scene tags as well as groups tucked at the
    end of a parenthetical/bracketed suffix (``(1080p BluRay x265 ImE)``).
    """
    if not name:
        return None
    meta = guess_metadata(name)
    grp = meta.get("release_group")
    if grp and isinstance(grp, str) and len(grp) >= 2 and grp.lower() not in _NON_GROUP_TOKENS:
        return grp
    base = re.sub(r"\.(srt|ass|ssa|sub|vtt|mkv|mp4|avi)$", "", name.rsplit("/", 1)[-1].strip(), flags=re.IGNORECASE)
    tokens: list[str] = []
    bracket = re.search(r"[\[\(]([^\[\]()]+)[\]\)]\s*$", base)
    if bracket:
        tokens = re.findall(r"[A-Za-z0-9]+", bracket.group(1))
    if not tokens:
        dash = re.search(r"-\s*([^-\s]+)\s*$", base)
        if not dash:
            return None
        tokens = re.findall(r"[A-Za-z0-9]+", dash.group(1))
    # Drop trailing codec/language/episode markers (``-FSiHD.ENG`` -> ``FSiHD``).
    while tokens and (
        tokens[-1].lower() in _NON_GROUP_TOKENS
        or tokens[-1].isdigit()
        or _EPISODE_LIKE_REGEX.match(tokens[-1])
    ):
        tokens.pop()
    if not tokens:
        return None
    segment = tokens[-1]
    if not re.fullmatch(r"[A-Za-z0-9]{2,}", segment):
        return None
    return segment


# Scene retail encodes that preserve the original studio bumpers/intro, so
# their subtitle timeline matches the disc master. Micro-rip / custom-repack
# groups frequently trim bumpers or re-encode intros, shifting the whole
# timeline; prefer the former when several references pass the edition gates.
_PREFERRED_REFERENCE_GROUPS = frozenset(
    {
        "chd", "hdc", "ebp", "fsihd", "surcode", "framestor", "don",
        "ctrlhd", "flux", "sparks", "geck", "tayto", "kogi", "trollhd",
        "swtyblz", "fgt", "d-z0n", "dz0n", "playweb", "sometv",
    }
)
_DEPRIORITIZED_REFERENCE_GROUPS = frozenset(
    {
        "tigole", "yify", "yts", "rarbg", "ganool", "jyk", "ozlem",
        "mmkv", "blackjesus", "psa", "pahe", "galaxyrg", "sujaid",
        "axxo", "shaanig", "evo", "qman", "utr", "vxt", "d3g",
        "afm72", "ano", "rmteam", "chotab", "judas", "nepu",
        "mkvhub", "bone", "minihd", "bhdstudio", "joy", "klaxxon",
    }
)


def reference_group_rank(name: str | None) -> int:
    """Rank a reference's release group by timeline reliability.

    ``0`` = known scene retail encode (bumpers preserved), ``1`` = neutral /
    unknown, ``2`` = micro-rip / custom repack (bumpers often cut). Groups are
    matched as whole tokens so dot-delimited names (``….Tigole.srt``) count too.
    """
    tokens = set(re.findall(r"[a-z0-9]+", (name or "").lower()))
    group = tokens | {(_release_group(name) or "").lower()}
    if group & _PREFERRED_REFERENCE_GROUPS:
        return 0
    if group & _DEPRIORITIZED_REFERENCE_GROUPS:
        return 2
    return 1


def is_informative_release_name(name: str | None) -> bool:
    """True when a stream filename carries scene/release tokens worth matching on.

    Arbitrary debrid names (``wVwm.mkv``) return ``False`` so the resolver
    falls back to the top-rated official reference for the exact episode.
    """
    candidate = (name or "").strip()
    if not candidate:
        return False
    return bool(_RELEASE_TOKEN_REGEX.search(candidate))


_MULTI_SEASON_RANGE_REGEX = re.compile(r"\bs\d{1,2}\s*-\s*s?\d{1,2}\b", re.IGNORECASE)

# A *reliable* single-episode marker (SxxExx, Exx/EPxx, Episode N, Part N,
# NxNN). Unlike ``_BARE_EPISODE_REGEX`` it never mistakes a season number
# ("Season 1", "S01") or a technical token ("1080p") for an episode.
_EXPLICIT_EPISODE_REGEX = re.compile(
    r"(?i)(?:s\d{1,2}[\s._-]?e\d{1,3}"
    r"|\bep?[\s._-]?\d{1,3}\b"
    r"|\bepisode[\s._-]?\d{1,3}\b"
    r"|\bpart[\s._-]?\d{1,3}\b"
    r"|\b\d{1,2}x\d{1,3}\b)"
)


def looks_like_season_pack(name: str | None) -> bool:
    """True when a release name is a whole-season/multi-season pack.

    A raw season-pack subtitle cannot be sliced to one episode, so episode
    matching must prefer an explicitly-tagged single-episode release over
    these names.
    """
    candidate = str(name or "")
    lowered = candidate.lower()
    if not lowered:
        return False
    meta = guess_metadata(candidate)
    if meta.get("type") == "episode" and meta.get("season") is not None and meta.get("episode") is None:
        return True
    other = meta.get("other")
    if other and ("Complete" in other if isinstance(other, list) else other == "Complete"):
        return True
    if "complete" in lowered:
        return True
    if _MULTI_SEASON_RANGE_REGEX.search(lowered):
        return True
    if _EXPLICIT_EPISODE_REGEX.search(candidate):
        return False
    # No explicit episode marker but a season number ("Season 1", "S01",
    # "First Season") => a season pack, not a single episode.
    return _season_number(candidate) is not None


def candidate_episode_number(name: str | None) -> int | None:
    """Episode number for matching, treating season packs as untagged.

    ``_member_episode_number`` is intentionally loose and will read the "1" in
    "Season 1" as episode 1, which would wrongly exclude a whole-season pack
    from every episode except the first. Packs are therefore reported as
    untagged so they remain sliceable by ``select_zip_member``.
    """
    if not name or looks_like_season_pack(name):
        return None
    meta = guess_metadata(name)
    if meta.get("type") == "movie":
        return None
    ep = meta.get("episode")
    if ep is not None:
        if isinstance(ep, list) and ep:
            return int(ep[0])
        try:
            return int(ep)
        except (ValueError, TypeError):
            pass
    return _member_episode_number(name)


def _path_segments(value: str | None) -> list[str]:
    """Split a filename/path on both separators after URL-decoding."""
    raw = unquote_plus(str(value or "")).replace("\\", "/")
    return [segment for segment in raw.split("/") if segment]


def prefer_meaningful_release_name(value: str | None) -> str:
    """Pick the most edition-informative segment of a filename or path.

    Usenet/debrid proxies (e.g. AIOStreams) hand out an obfuscated basename
    inside a folder that carries the real scene name::

        Suits.S01E01...-playWEB/e4WcFo4Tz5J8PoFwiBfP880XsBuHk4dS.mkv

    Matching on the bare hash fails every edition gate. When the basename is
    not informative but an ancestor directory is, return that directory name;
    otherwise return the basename unchanged (preserving direct-filename input).
    """
    segments = _path_segments(value)
    if not segments:
        return str(value or "").strip()
    basename = segments[-1]
    if is_informative_release_name(basename):
        return basename
    for ancestor in reversed(segments[:-1]):
        if is_informative_release_name(ancestor):
            return ancestor
    return basename


_TITLE_SPLIT_REGEX = re.compile(
    r"(?i)(?:^|[\s._\-()\[\]])(?:"
    r"s\d{1,2}[\s._-]*e\d{1,3}(?:[\s._-]*e\d{1,3})*|"
    r"season[\s._-]*(?:\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)|"
    r"(?<!\d)\d{1,2}x\d{1,3}(?!\d)|"
    r"episode[\s._-]*\d{1,3}"
    r")(?:$|[\s._\-()\[\]])"
)
_TITLE_NOISE_REGEX = re.compile(
    r"(?i)\b(1080p|720p|2160p|480p|2160|1080|720|480|4k|web[\s._-]?dl|webrip|bluray|"
    r"brrip|bdrip|hdtv|dvdrip|cam|x264|x265|h\.?264|h\.?265|hevc|aac|dts|remux|"
    r"proper|repack|complete|dual|unrated|extended|directors|cut|season|episode)\b"
)


def _title_from_filename(filename: str | None) -> str | None:
    """Derive a Podnapisi keyword title from a scene-style playback filename."""
    if not filename:
        return None
    base = unquote_plus(str(filename)).replace("\\", "/").rsplit("/", 1)[-1].strip()
    base = re.sub(r"\.[A-Za-z0-9]{2,5}$", "", base).strip()
    if not base:
        return None
    marker = _TITLE_SPLIT_REGEX.search(base)
    head = base[: marker.start()] if marker else base
    head = re.sub(r"\([^()]*\)|\[[^\[\]]*\]", " ", head)
    head = re.sub(r"[._\-+]+", " ", head)
    head = _TITLE_NOISE_REGEX.sub(" ", head)
    head = re.sub(r"\b(19\d{2}|20\d{2})\b", " ", head)
    title = re.sub(r"\s+", " ", head).strip(" -")
    if len(title) < 3 or not re.search(r"[A-Za-z]{2,}", title):
        return None
    if not re.search(r"\s", title) and not _RELEASE_TOKEN_REGEX.search(base) and len(title) < 5:
        return None
    return title
