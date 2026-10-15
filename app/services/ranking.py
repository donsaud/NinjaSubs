"""Attribute-Based Weighted Scoring and Subtitle Ranking Engine for Stremio/Nuvio."""

import re
import urllib.parse
from typing import Any

# Known technical/spec tokens that are NOT scene/p2p release groups
_INVALID_GROUPS = {
    "1080p",
    "720p",
    "2160p",
    "4k",
    "uhd",
    "576p",
    "480p",
    "bluray",
    "blu-ray",
    "bdrip",
    "brrip",
    "webdl",
    "web-dl",
    "webrip",
    "web-rip",
    "remux",
    "bdremux",
    "hdtv",
    "pdtv",
    "dvd",
    "dvdrip",
    "cam",
    "telesync",
    "ts",
    "x264",
    "x265",
    "hevc",
    "av1",
    "h264",
    "h265",
    "avc",
    "aac",
    "ac3",
    "dts",
    "truehd",
    "atmos",
    "ddp",
    "dd",
    "sub",
    "subs",
    "srt",
    "mkv",
    "mp4",
    "avi",
    "extended",
    "remastered",
    "unrated",
    "proper",
    "repack",
    "multi",
    "dual",
    "10bit",
    "hdr",
    "dv",
    "dovi",
    "dl",
    "rip",
    "hd",
    "bd",
}

# Known scene and P2P release groups
KNOWN_GROUPS = {
    "RARBG",
    "FRAMESTOR",
    "PLAYWEB",
    "FLUX",
    "ION10",
    "PSA",
    "AMZN",
    "SPARKS",
    "GECKOS",
    "ROVERS",
    "NTB",
    "CMRG",
    "IME",
    "YTS",
    "YTS.MX",
    "YIFY",
    "QXR",
    "TGX",
    "CTRLHD",
    "DON",
    "D-ZON3",
    "SBR",
    "SMURF",
    "CHD",
    "WIKI",
    "TAYTO",
    "SINNERS",
    "EVO",
    "AMIABLE",
    "BLOW",
    "COCOS",
    "DEPTH",
    "DRONES",
    "FGT",
    "GHOULS",
    "HIDT",
    "JEBI",
    "LOST",
    "MAX",
    "NERDHD",
    "OFT",
    "PAHE",
    "PTH",
    "RAV1NE",
    "SANTI",
    "SEV",
    "SURCODE",
    "TELLY",
    "USURY",
    "VXT",
    "WAF",
    "YOLOW",
    "AVS",
    "MINX",
    "DIMENSION",
    "LOL",
    "KILLERS",
    "FLEET",
    "BATV",
    "TROPIC",
    "ETHEL",
    "SNEAKY",
    "GALAXYRG",
    "BONE",
    "KOGI",
    "GLHF",
    "PLZ",
    "PIGNUS",
    "TOMMY",
    "NAISU",
    "EDITH",
    "SURF",
    "POW4",
    "TRUMP",
    "DEFLATE",
    "SHORTBREHD",
    "WELP",
    "CAFFEINE",
    "MZABI",
    "NOGRP",
    "ELITE",
    "DRG",
    "BLUTORRENT",
}

# Group names that are also ordinary title words; matching them anywhere in a
# name labels "Mad.Max.Fury.Road" or "Lost.S01" with a release group.
_AMBIGUOUS_GROUP_WORDS = frozenset({"LOST", "MAX", "DON", "LOL"})


def _scan_order(groups) -> tuple[str, ...]:
    """Deterministic, longest-first scan order for substring group detection."""
    return tuple(sorted(groups, key=lambda g: (-len(g), g.upper())))


KNOWN_GROUPS_SCAN_ORDER = _scan_order(
    g for g in KNOWN_GROUPS if g.upper() not in _AMBIGUOUS_GROUP_WORDS
)

# Regex patterns for junk token stripping
_DOMAIN_BRACKETS = re.compile(
    r"\[[^\]]*?\.(?:com|net|org|blog|site|tv|me|to|is|io|cc|co|xyz|online|top|info|biz)[^\]]*?\]",
    re.IGNORECASE,
)
_DOMAIN_PARENS = re.compile(
    r"\([^)]*?\.(?:com|net|org|blog|site|tv|me|to|is|io|cc|co|xyz|online|top|info|biz)[^)]*?\)",
    re.IGNORECASE,
)
_STANDALONE_DOMAINS = re.compile(
    r"\b(?:www\.)?[a-zA-Z0-9_\-]+\.(?:com|net|org|blog|site|tv|me|to|is|io|cc|co|xyz|online|top|info|biz)\b",
    re.IGNORECASE,
)
_NOISY_SITE_PREFIXES = re.compile(
    r"\b(?:moappmovies|arabp2p)(?:\.[a-z]{2,4})?\b",
    re.IGNORECASE,
)
_NOISY_PREFIX_START = re.compile(
    r"^moappmovies[._\-\s]*",
    re.IGNORECASE,
)
_NOISY_BRACKET_SITES = re.compile(
    r"\[(?:moappmovies|arabp2p).*?\]",
    re.IGNORECASE,
)

# Trailing hashes / hex signatures: e.g. _86f1f22e8f1fd5bd, _fd3cebb32c50021a
_HEX_HASHES = re.compile(r"_[a-f0-9]{8,}\b", re.IGNORECASE)
_TRAILING_HASH = re.compile(r"(?:_|\.)[a-f0-9]{8,}(?=[._\s]|$)", re.IGNORECASE)

# Unrelated metadata brackets: e.g. [4k], (4k), [1080p], (1080p)
_METADATA_BRACKETS = re.compile(
    r"\[\s*(4k|2160p|1080p|1080i|720p|576p|480p|uhd|hdr|10bit|dv|dolby\s*vision)\s*\]",
    re.IGNORECASE,
)
_METADATA_PARENS = re.compile(
    r"\(\s*(4k|2160p|1080p|1080i|720p|576p|480p|uhd|hdr|10bit|dv|dolby\s*vision)\s*\)",
    re.IGNORECASE,
)

# Regex patterns for core scene attribute extraction
_RES_REGEX = re.compile(r"(?i)\b(2160p|4k|uhd|1080p|1080i|720p|576p|576i|480p|480i)\b")

_SOURCE_REGEXES = [
    (re.compile(r"(?i)\b(uhd[-._ ]?remux|bdremux|remux)\b"), "Remux"),
    (re.compile(r"(?i)\b(bluray|blu-ray|bdrip|brrip)\b"), "BluRay"),
    (re.compile(r"(?i)\b(web-?dl|webdl)\b"), "WEB-DL"),
    (re.compile(r"(?i)\b(web-?rip|webrip)\b"), "WEBRip"),
    (re.compile(r"(?i)\b(web)\b"), "WEB-DL"),
    (re.compile(r"(?i)\b(hdtv|pdtv|dsr|tvrip)\b"), "HDTV"),
    (re.compile(r"(?i)\b(dvd|dvdrip|r5)\b"), "DVD"),
    (re.compile(r"(?i)\b(cam|camrip|hdcam|ts|telesync|hd-ts|hdts|tc|telecine)\b"), "CAM"),
]

_CODEC_REGEXES = [
    (re.compile(r"(?i)\b(x265)\b"), "x265"),
    (re.compile(r"(?i)\b(hevc|h\.?265)\b"), "HEVC"),
    (re.compile(r"(?i)\b(x264)\b"), "x264"),
    (re.compile(r"(?i)\b(avc|h\.?264)\b"), "x264"),
    (re.compile(r"(?i)\b(av1)\b"), "AV1"),
    (re.compile(r"(?i)\b(10bit)\b"), "10bit"),
    (re.compile(r"(?i)\b(hdr10\+|hdr10|hdr)\b"), "HDR"),
    (re.compile(r"(?i)\b(dolby[-._ ]?vision|dovi|dv)\b"), "DV"),
]

_AUDIO_REGEXES = [
    (re.compile(r"(?i)\b(dts-hd(?:[-._ ]?ma)?)\b"), "DTS-HD"),
    (re.compile(r"(?i)\b(dts)\b"), "DTS"),
    (re.compile(r"(?i)\b(truehd(?:[-._ ]?atmos)?)\b"), "TrueHD"),
    (re.compile(r"(?i)\b(atmos)\b"), "Atmos"),
    (re.compile(r"(?i)\b(ddp5\.1|dd\+5\.1|eac3(?:[-._ ]?5\.1)?)\b"), "DDP5.1"),
    (re.compile(r"(?i)\b(ddp|dd\+|eac3)\b"), "DDP"),
    (re.compile(r"(?i)\b(dd5\.1|ac3(?:[-._ ]?5\.1)?)\b"), "DD5.1"),
    (re.compile(r"(?i)\b(ac3|dd)\b"), "AC3"),
    (re.compile(r"(?i)\b(aac(?:[0-9]\.[0-9])?)\b"), "AAC"),
    (re.compile(r"(?i)\b(flac)\b"), "FLAC"),
]


def sanitize_release_name(raw_name: str) -> str:
    r"""
    Robust release name sanitization stripping:
    - Domain names and URL brackets: \[.*?\.?(com|net|org|blog|site|tv).*?\]
    - Standalone URL domains and noisy prefixes (arabp2p.net, moappmovies...)
    - Trailing hashes/hex signatures: _[a-f0-9]{8,}
    - Unrelated metadata brackets: [4k], (4k), [1080p], (1080p)
    - Redundant subtitle/video extensions (.srt, .vtt, .mkv)
    - Messy separators and leading/trailing punctuation.
    """
    if not raw_name:
        return ""

    name = str(raw_name).strip()

    # 1. Strip file extension if attached
    name = re.sub(r"\.(srt|vtt|sub|mkv|mp4|avi|ts|webm)$", "", name, flags=re.IGNORECASE)

    # 2. Strip trailing Subdl/provider hash IDs and hex signatures (e.g. _86f1f22e8f1fd5bd)
    name = _HEX_HASHES.sub("", name)
    name = _TRAILING_HASH.sub("", name)

    # Secondary extension strip if chained (e.g. .srt_86f1f22e8f1fd5bd)
    name = re.sub(r"\.(srt|vtt|sub|mkv|mp4|avi|ts|webm)$", "", name, flags=re.IGNORECASE)
    name = _HEX_HASHES.sub("", name)
    name = _TRAILING_HASH.sub("", name)

    # 3. Strip domain names, URL brackets, and noisy site tags
    name = _DOMAIN_BRACKETS.sub("", name)
    name = _DOMAIN_PARENS.sub("", name)
    name = _NOISY_BRACKET_SITES.sub("", name)
    name = _STANDALONE_DOMAINS.sub("", name)
    name = _NOISY_SITE_PREFIXES.sub("", name)
    name = _NOISY_PREFIX_START.sub("", name)

    # 4. Strip unrelated metadata brackets (e.g. [4k], (4k))
    name = _METADATA_BRACKETS.sub("", name)
    name = _METADATA_PARENS.sub("", name)

    # 5. Clean off trailing standard indexer tags like [eztv], [rarbg], [rartv], [tgx]
    name = re.sub(r"\[(eztv|rarbg|rartv|tgx|10bit|hdr|dv)\]$", "", name, flags=re.IGNORECASE)

    # 6. Normalize ugly double/triple underscores, dashes, and duplicate dots
    name = re.sub(r"__-__", ".", name)
    name = re.sub(r"_{2,}", ".", name)
    name = re.sub(r"\.-+\.", ".", name)
    name = re.sub(r"\.\.+", ".", name)

    # 7. Clean up any trailing/leading dots, underscores, dashes, or spaces
    name = name.strip("._ -")

    return name


clean_subtitle_filename = sanitize_release_name


def extract_clean_title(filename: str) -> str:
    """
    Extract pure movie/series title (and year/season) stripped of technical tokens,
    groups, and junk for high-accuracy title similarity comparison.
    """
    name = sanitize_release_name(filename)
    # Strip trailing scene group e.g. -SPARKS, -FLUX
    name = re.sub(r"-[A-Za-z0-9_.]+$", "", name)

    # Strip technical tags
    tech_pattern = re.compile(
        r"(?i)\b(2160p|4k|1080p|1080i|720p|576p|480p|uhd|remux|bdremux|bluray|blu-ray|bdrip|brrip|"
        r"web-?dl|webdl|web-?rip|webrip|hdtv|pdtv|dvd|dvdrip|cam|telesync|ts|"
        r"x265|hevc|h\.?265|x264|h\.?264|avc|av1|10bit|hdr|hdr10\+?|dv|dovi|dolby[-._ ]?vision|"
        r"dts-hd(?:[-._ ]?ma)?|dts|truehd|atmos|ddp5\.1|dd\+5\.1|eac3|ddp|dd\+|dd5\.1|ac3|aac|flac|"
        r"repack|proper|extended|unrated|directors[-._ ]?cut|remastered|multi|dual)\b"
    )
    name = tech_pattern.sub("", name)
    name = re.sub(r"[\s._\-+]+", " ", name).strip().lower()
    return name


def format_subtitle_title(score: int, source: str, raw_filename: str) -> str:
    """
    Format subtitle title with strictly single spaces between badges:
    e.g. '[95%] [Subdl] Dexter.2006.S08.1080p.BluRay.x265-ImE'
    """
    clean_name = sanitize_release_name(raw_filename)
    return f"[{score}%] [{source}] {clean_name}"


def parse_release_metadata(filename: str) -> dict[str, str | None]:
    """
    Parse a media stream or subtitle filename into core scene attributes:
    - group: Release group (e.g. RARBG, FraMeSToR, playWEB, FLUX, ION10, PSA, AMZN)
    - source: Remux, BluRay, UHD, WEB-DL, WEBRip, HDTV, DVD, CAM
    - resolution: 2160p, 1080p, 720p, 576p, 480p
    - codec: x265, HEVC, x264, AV1, 10bit, HDR, DV
    - audio: DTS-HD, DTS, TrueHD, Atmos, DDP5.1, DDP, DD5.1, AC3, AAC, FLAC
    - raw_filename: Original unparsed input string
    """
    if not filename:
        return {
            "source": None,
            "resolution": None,
            "group": None,
            "codec": None,
            "audio": None,
            "raw_filename": "",
        }

    raw_str = str(filename)
    name = sanitize_release_name(raw_str)

    # 1. Resolution
    resolution: str | None = None
    res_match = _RES_REGEX.search(name) or _RES_REGEX.search(raw_str)
    if res_match:
        val = res_match.group(1).lower()
        if val in ("2160p", "4k", "uhd"):
            resolution = "2160p"
        elif val in ("1080p", "1080i"):
            resolution = "1080p"
        elif val == "720p":
            resolution = "720p"
        elif val in ("576p", "576i"):
            resolution = "576p"
        elif val in ("480p", "480i"):
            resolution = "480p"

    # 2. Source
    source: str | None = None
    for pattern, src_name in _SOURCE_REGEXES:
        if pattern.search(name) or pattern.search(raw_str):
            source = src_name
            break

    # 3. Codec
    codec: str | None = None
    for pattern, c_name in _CODEC_REGEXES:
        if pattern.search(name) or pattern.search(raw_str):
            codec = c_name
            break

    # 4. Audio
    audio: str | None = None
    for pattern, a_name in _AUDIO_REGEXES:
        if pattern.search(name) or pattern.search(raw_str):
            audio = a_name
            break

    # 5. Group
    group: str | None = None

    # Step A: Scene convention: last segment following a hyphen '-'
    if "-" in name:
        candidate = name.rsplit("-", 1)[-1].strip()
        candidate = re.sub(r"\[[^\]]*\]$", "", candidate).strip()
        candidate = candidate.strip(" ._-[]()")
        if candidate and candidate.lower() not in _INVALID_GROUPS and len(candidate) >= 2:
            group = candidate

    # Step B: Bracketed group e.g. [FLUX], [YTS.MX], [FraMeSToR]
    if not group:
        bracket_match = re.search(r"\[([A-Za-z0-9_.]+)\]", name)
        if bracket_match:
            candidate = bracket_match.group(1).strip()
            if candidate and candidate.lower() not in _INVALID_GROUPS and len(candidate) >= 2:
                group = candidate

    # Step C: Match known release groups (case-insensitive word boundaries).
    # Scanned in a fixed order (longest first) so the winner never depends on
    # set hash order; ordinary words ("Lost", "Max") are only trusted as the
    # ``-GROUP`` tail handled above, never anywhere in a title.
    if not group:
        for kg in KNOWN_GROUPS_SCAN_ORDER:
            if re.search(rf"\b{re.escape(kg)}\b", name, re.IGNORECASE):
                group = kg
                break

    return {
        "source": source,
        "resolution": resolution,
        "group": group,
        "codec": codec,
        "audio": audio,
        "raw_filename": raw_str,
    }


def levenshtein_ratio(s1: str, s2: str) -> float:
    """
    Compute normalized Levenshtein similarity ratio between two strings [0.0, 1.0].
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    clean1 = (
        re.sub(r"\s+", " ", re.sub(r"[._\-+]", " ", clean_subtitle_filename(s1))).strip().lower()
    )
    clean2 = (
        re.sub(r"\s+", " ", re.sub(r"[._\-+]", " ", clean_subtitle_filename(s2))).strip().lower()
    )

    if clean1 == clean2:
        return 1.0

    len1, len2 = len(clean1), len(clean2)
    if len1 == 0 or len2 == 0:
        return 0.0

    if len1 > len2:
        clean1, clean2 = clean2, clean1
        len1, len2 = len2, len1

    prev_row = list(range(len1 + 1))
    curr_row = [0] * (len1 + 1)

    for j, c2 in enumerate(clean2, start=1):
        curr_row[0] = j
        for i, c1 in enumerate(clean1, start=1):
            cost = 0 if c1 == c2 else 1
            curr_row[i] = min(
                curr_row[i - 1] + 1,
                prev_row[i] + 1,
                prev_row[i - 1] + cost,
            )
        prev_row = curr_row[:]

    distance = prev_row[len1]
    max_len = max(len(clean1), len(clean2))
    return max(0.0, 1.0 - (distance / max_len))








def calculate_compatibility_percentage(target_meta, sub_meta) -> int:
    """Historical API alias for the sole weighted compatibility engine."""
    from app.services.release_scoring_integration import compare_metadata
    return round(compare_metadata(target_meta, sub_meta).ranking_percentage)


calculate_match_score = calculate_compatibility_percentage


def get_source_popularity_percentage(subtitle_name: str) -> int:
    """No invented compatibility when target release evidence is unavailable."""
    return calculate_compatibility_percentage("", subtitle_name)


get_source_popularity_score = get_source_popularity_percentage


_VIDEO_HASH_RE = re.compile(r"^[0-9a-f]{16}$")


def normalize_video_hash(value: Any) -> str | None:
    """Normalise a client-supplied MovieHash, or return ``None``.

    MovieHash is OPTIONAL. A missing hash is a normal, fully supported state:
    the pipeline continues through release-compatibility reference selection and
    the verifier remains the sole authority on synchronisation. Nothing here
    derives, computes, or guesses a hash -- it only validates one that a client
    actually supplied.

    A malformed value is treated as UNAVAILABLE rather than passed through,
    because ``extract_stream_params`` previously accepted any non-empty string.
    A truthy-but-invalid value would otherwise reach
    ``OpenSubtitlesProvider``'s ``params["moviehash"]``, spending a metered
    OpenSubtitles call on a query that can never match and logging a request
    that looks like an exact-identity lookup.

    Strictly 16 hex characters, which is the width OpenSubtitles returns and
    the width ``opensubtitles.py`` documents. Case and surrounding whitespace
    are normalised; nothing else is coerced.
    """
    if value is None:
        return None
    # ``bytes`` is truthy and would stringify to "b'...'"; reject explicitly
    # rather than letting a non-string reach int()/re.
    if not isinstance(value, str):
        return None
    candidate = value.strip().lower()
    if not candidate:
        return None
    return candidate if _VIDEO_HASH_RE.match(candidate) else None


def extract_stream_params(
    extra: str | None,
    query_params: Any | None = None,
) -> dict[str, Any]:
    """
    Extract stream parameters (filename, video_hash, video_size) from {extra} route path or query parameters.
    """
    params: dict[str, Any] = {
        "filename": None,
        "video_hash": None,
        "video_size": None,
    }

    if query_params:
        filename_query = query_params.get("filename")
        if filename_query:
            params["filename"] = urllib.parse.unquote(filename_query).strip()

        for hk in ("videoHash", "video_hash", "moviehash", "hash"):
            val = query_params.get(hk)
            if val:
                # Malformed values are dropped, not forwarded. The loop keeps
                # scanning so a later alias can still supply a valid hash.
                normalized = normalize_video_hash(urllib.parse.unquote(str(val)))
                if normalized:
                    params["video_hash"] = normalized
                    break

        for sk in ("videoSize", "video_size", "moviebytesize", "size"):
            val = query_params.get(sk)
            if val:
                raw_sz = urllib.parse.unquote(str(val)).strip()
                # P1-3: same normalisation as the extra-segment branch. A byte
                # size of 0 is not a real file size for this API, so it means
                # "unavailable" and is normalised to None exactly once, here,
                # rather than surviving as int 0 that later `or ""` fallbacks
                # silently erase at different hops.
                if raw_sz in ("", "0"):
                    params["video_size"] = None
                else:
                    try:
                        params["video_size"] = int(raw_sz)
                    except (ValueError, TypeError):
                        params["video_size"] = raw_sz
                break

    if extra:
        decoded = urllib.parse.unquote(extra).strip()
        if decoded.endswith(".json"):
            decoded = decoded[:-5].strip()

        # Known Stremio extra parameter keys for subtitles
        known_keys = (
            "filename|videoHash|video_hash|moviehash|hash|videoSize|video_size|moviebytesize|size"
        )
        param_pattern = re.compile(rf"(?:^|[&/])({known_keys})=", re.IGNORECASE)

        matches = list(param_pattern.finditer(decoded))
        if matches:
            for i, m in enumerate(matches):
                key = m.group(1).lower()
                val_start = m.end()
                val_end = matches[i + 1].start() if i + 1 < len(matches) else len(decoded)
                val = decoded[val_start:val_end].strip(" '\"")

                if key == "filename" and not params["filename"]:
                    params["filename"] = val
                elif (
                    key in ("videohash", "video_hash", "moviehash", "hash")
                    and not params["video_hash"]
                ):
                    params["video_hash"] = normalize_video_hash(val)
                elif (
                    key in ("videosize", "video_size", "moviebytesize", "size")
                    and params["video_size"] in (None, "")
                ):
                # P1-3: presence, not truthiness. A supplied size of 0 used to be
                # treated as absent (`not params["video_size"]` sees int 0 as
                # falsy) and silently overwritten by the extra segment, while
                # its sibling hash -- stored as str "0" -- was immune. 0 is not a
                # meaningful byte size for this API, so it is normalised to
                # "missing" once, here, instead of meaning different things at
                # different hops.
                    raw = val.strip()
                    if raw in ("", "0"):
                        params["video_size"] = None
                    else:
                        try:
                            params["video_size"] = int(raw)
                        except (ValueError, TypeError):
                            params["video_size"] = raw
        else:
            if not params["filename"] and re.search(r"(?i)\.(mkv|mp4|avi|ts|m2ts|webm)$", decoded):
                # Preserve a meaningful parent folder when the basename is an
                # obfuscated debrid hash (e.g. ``Scene.Release.Name/hash.mkv``).
                from app.services.sync.matching import prefer_meaningful_release_name

                stripped = decoded.strip(" '\"")
                params["filename"] = prefer_meaningful_release_name(stripped) or stripped

    return params


def extract_target_filename(
    extra: str | None,
    query_params: Any | None = None,
) -> str | None:
    """
    Extract the playing stream filename from {extra} route path or query parameters.
    """
    return extract_stream_params(extra, query_params)["filename"]


def rank_and_sort_subtitles(subtitles, target_filename=None, preferred_languages=None):
    """Historical API delegates to authoritative weighted DisplayPolicy."""
    from app.services.release_scoring_integration import display_order
    return display_order(target_filename, subtitles, preferred_languages=preferred_languages)
