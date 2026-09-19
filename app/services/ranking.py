"""Attribute-Based Weighted Scoring and Subtitle Ranking Engine for Stremio/Nuvio."""

import re
import urllib.parse
from typing import Any

from app.models import SubtitleRelease

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

    # Step C: Match known release groups (case-insensitive word boundaries)
    if not group:
        for kg in KNOWN_GROUPS:
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


def is_group_match(g1: str | None, g2: str | None) -> bool:
    """Check if two release groups match (case-insensitive and alias-aware)."""
    if not g1 or not g2:
        return False
    norm1 = g1.strip().lower().replace(".", "")
    norm2 = g2.strip().lower().replace(".", "")
    if norm1 == norm2:
        return True
    if {norm1, norm2} <= {"yts", "ytsmx", "yify"}:
        return True
    return False


def is_source_match(s1: str | None, s2: str | None) -> tuple[bool, bool, bool]:
    """
    Check source compatibility.
    Returns: (exact_match, high_compat_match, is_incompatible_penalty)
    """
    if not s1 or not s2:
        return False, False, False
    s1_norm = s1.strip().lower()
    s2_norm = s2.strip().lower()
    if s1_norm == s2_norm:
        return True, False, False
    # High compatibility: Remux with BluRay/UHD, WEB-DL with WEBRip
    if {s1_norm, s2_norm} <= {"remux", "bluray", "uhd"}:
        return False, True, False
    if {s1_norm, s2_norm} <= {"web-dl", "webrip"}:
        return False, True, False
    # Incompatible penalty: Disc/WEB target with CAM or HDTV
    if (
        s1_norm in ("remux", "bluray", "uhd", "web-dl", "webrip") and s2_norm in ("cam", "hdtv")
    ) or (s2_norm == "cam" and s1_norm != "cam"):
        return False, False, True
    return False, False, False


def is_codec_audio_match(
    t_codec: str | None, s_codec: str | None, t_audio: str | None, s_audio: str | None
) -> bool:
    """Check if codec or audio attributes match."""
    if t_codec and s_codec:
        c1 = t_codec.strip().lower()
        c2 = s_codec.strip().lower()
        if c1 == c2:
            return True
        if {c1, c2} <= {"x265", "hevc"}:
            return True
        if {c1, c2} <= {"x264", "h264", "avc"}:
            return True
        if {c1, c2} <= {"dv", "dovi", "dolby vision"}:
            return True
    if t_audio and s_audio:
        a1 = t_audio.strip().lower()
        a2 = s_audio.strip().lower()
        if a1 == a2:
            return True
        if {a1, a2} <= {"dts-hd", "dts"}:
            return True
        if {a1, a2} <= {"ddp5.1", "ddp", "dd5.1", "eac3"}:
            return True
        if {a1, a2} <= {"truehd", "atmos"}:
            return True
    return False


def calculate_compatibility_percentage(
    target_meta: dict[str, str | None] | str,
    sub_meta: dict[str, str | None] | str,
) -> int:
    """
    Attribute-Based Weighted Scoring calculation (0 - 100%):
    - Group Match: +40% (single strongest sync indicator)
    - Source Match: +25% (exact: +25%, high-compat: +20%, incompatible: -30%)
    - Resolution Match: +15% (e.g., 2160p with 2160p)
    - Codec / Audio Match: +10% (e.g., x265/HEVC, DTS-HD, DDP5.1)
    - Clean Title Match: +10% (normalized title similarity ratio)

    Edge-Case Rules:
    - If both group and source match, minimum compatibility score guarantees 85%+.
    - Retain fallback compatibility score for items with no identifiable attributes.
    - Clamped strictly between 0% and 100%.
    """
    if isinstance(target_meta, str):
        target_meta = parse_release_metadata(target_meta)
    if isinstance(sub_meta, str):
        sub_meta = parse_release_metadata(sub_meta)

    score = 0

    # 1. Group Match (+40%)
    t_group = target_meta.get("group")
    s_group = sub_meta.get("group")
    group_match = is_group_match(t_group, s_group)
    if group_match:
        score += 40

    # 2. Source Match (+25% / +20% / -30%)
    t_source = target_meta.get("source")
    s_source = sub_meta.get("source")
    exact_src, high_compat_src, src_penalty = is_source_match(t_source, s_source)
    if exact_src:
        score += 25
    elif high_compat_src:
        score += 20
    elif src_penalty:
        score -= 30

    # 3. Resolution Match (+15%)
    t_res = (target_meta.get("resolution") or "").strip().lower()
    s_res = (sub_meta.get("resolution") or "").strip().lower()
    res_match = bool(t_res and s_res and t_res == s_res)
    if res_match:
        score += 15

    # 4. Codec / Audio Match (+10%)
    t_codec = target_meta.get("codec")
    s_codec = sub_meta.get("codec")
    t_audio = target_meta.get("audio")
    s_audio = sub_meta.get("audio")
    if is_codec_audio_match(t_codec, s_codec, t_audio, s_audio):
        score += 10

    # 5. Clean Title Match (+10%)
    t_raw = target_meta.get("raw_filename") or ""
    s_raw = sub_meta.get("raw_filename") or ""
    sim_title = levenshtein_ratio(extract_clean_title(t_raw), extract_clean_title(s_raw))
    sim_full = levenshtein_ratio(clean_subtitle_filename(t_raw), clean_subtitle_filename(s_raw))
    sim = max(sim_title, sim_full)
    score += int(round(sim * 10))

    # Edge-case rule 1: If both group and source match, minimum compatibility score guarantees 85%+
    if group_match and (exact_src or high_compat_src):
        score = max(score, 85)

    # Edge-case rule 2: Retain fallback compatibility score for items with no identifiable attributes
    has_sub_attributes = bool(
        sub_meta.get("group")
        or sub_meta.get("source")
        or sub_meta.get("resolution")
        or sub_meta.get("codec")
    )
    if not has_sub_attributes and not src_penalty:
        fallback = get_source_popularity_percentage(s_raw)
        score = max(score, fallback)

    return max(0, min(100, score))


calculate_match_score = calculate_compatibility_percentage


def get_source_popularity_percentage(subtitle_name: str) -> int:
    """
    Fallback baseline score (centered around 50%) based on source tier
    when no target stream filename is available.
    """
    meta = parse_release_metadata(subtitle_name)
    src = (meta.get("source") or "").lower()

    if src == "remux":
        return 60
    if src == "bluray":
        return 55
    if src == "web-dl":
        return 50
    if src == "webrip":
        return 45
    if src == "hdtv":
        return 35
    if src == "cam":
        return 10
    return 40


get_source_popularity_score = get_source_popularity_percentage


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
                params["video_hash"] = urllib.parse.unquote(str(val)).strip()
                break

        for sk in ("videoSize", "video_size", "moviebytesize", "size"):
            val = query_params.get(sk)
            if val:
                raw_sz = urllib.parse.unquote(str(val)).strip()
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
                    params["video_hash"] = val
                elif (
                    key in ("videosize", "video_size", "moviebytesize", "size")
                    and not params["video_size"]
                ):
                    try:
                        params["video_size"] = int(val)
                    except (ValueError, TypeError):
                        params["video_size"] = val
        else:
            if not params["filename"] and re.search(r"(?i)\.(mkv|mp4|avi|ts|m2ts|webm)$", decoded):
                params["filename"] = decoded.split("/")[-1].strip(" '\"")

    return params


def extract_target_filename(
    extra: str | None,
    query_params: Any | None = None,
) -> str | None:
    """
    Extract the playing stream filename from {extra} route path or query parameters.
    """
    return extract_stream_params(extra, query_params)["filename"]


def rank_and_sort_subtitles(
    subtitles: list[SubtitleRelease],
    target_filename: str | None = None,
    preferred_languages: list[str] | None = None,
) -> list[SubtitleRelease]:
    """
    Score and strictly sort subtitle releases in descending order by compatibility score.
    If multiple languages are requested, groups by language order while preserving
    strict descending score order within each language group.
    """
    target_meta = parse_release_metadata(target_filename) if target_filename else None

    for sub in subtitles:
        if target_meta:
            sub.score = calculate_compatibility_percentage(target_meta, sub.release_name)
        else:
            sub.score = get_source_popularity_percentage(sub.release_name)

    pref_langs = [lang.lower() for lang in (preferred_languages or [])]
    if len(pref_langs) > 1:
        lang_order = {lang: i for i, lang in enumerate(pref_langs)}
        return sorted(
            subtitles,
            key=lambda sub: (
                lang_order.get((getattr(sub, "lang", "") or "").lower(), 999),
                -sub.score,
            ),
        )

    # Strictly descending by computed score
    return sorted(subtitles, key=lambda sub: sub.score, reverse=True)
