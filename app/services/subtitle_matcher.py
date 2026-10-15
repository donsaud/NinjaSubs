"""Comprehensive Zero-Error Subtitle Matcher Engine for NinjaSubs.

Supports:
- Global, Regional, Asian/Anime, and FAST platforms:
  HMAX, AMZN, DSNP, NF, ATVP, HULU, PCOK, PMTP, CRAV, STAN, CR, IT, SHO, STARZ,
  BBC, CBC, ALL4, ROKU, WOW, BCORE, MGM+, DCU, AMCP, AMC, BINGE, FOXT, NOW, SKY,
  CPLUS, VIAPLAY, HIDIVE, TVING, WAVVE, IQIYI, WETV, BILIBILI, VIKI, TUBI, PLUTO,
  FREEVEE, DISC, HIST, SYFY, CW, FX, PBS.
- Editions & Cuts: THEATRICAL, EXTENDED, DIRECTORS_CUT, UNRATED, SPECIAL_EDITION, REMASTERED, IMAX.
- Remux, Repack, Proper, and Rerip detection & alignment.
- FPS Drift Detection: 23.976, 24.000, 25.000 (PAL), 29.970, 30.0, 50.0, 59.94, 60.0 fps.
- SDH / Hearing Impaired detection.
- Media Sources & Hierarchical Source Families:
  DISC_HI (Remux, BluRay, UHD), DISC_LO (BDRip, BRRip, DVDRip, DVD),
  WEB (WEB-DL, WEBRip, WEB), TV (HDTV, PDTV, DSR), CAM (CAM, TS).
- Resolutions: 2160p (4K/UHD), 1080p, 720p, 576p, 480p.
- Codecs & Profiles: x265 (HEVC), x264 (AVC), AV1, XviD, DivX, 10bit, HDR, DV (Dolby Vision).
- Audio Profiles: Atmos, TrueHD, DTS-HD, DTS, DDP5.1, DDP, DD5.1, AC3, AAC, FLAC, Opus.
- Scene & P2P release groups: FLUX, NTb, CMRG, FraMeSToR, RARBG, Sparks, etc.
- Two-Stage Architecture:
  * Stage 0: Binary MovieHash deterministic short-circuit (Tier 0, score=500, 100%).
  * Stage 1: Hard exclusion filter rejecting fatal desyncs (score=-1000).
  * Stage 2: Additive soft scoring matrix assigning MatchTier 1 to 4.
"""

import bisect
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.models import MatchTier, SubtitleRelease
from app.services.ranking import (
    _INVALID_GROUPS,
    KNOWN_GROUPS,
    clean_subtitle_filename,
    sanitize_release_name,
)

logger = logging.getLogger(__name__)


def is_debug_ranking_enabled() -> bool:
    """Check if safe diagnostic ranking logging is enabled via environment or config."""
    val = os.getenv("NINJASUBS_DEBUG_RANKING", "").strip().lower()
    if val in ("true", "1", "yes"):
        return True
    try:
        from app.config import settings

        return bool(getattr(settings, "NINJASUBS_DEBUG_RANKING", False))
    except Exception:
        return False


# =======================================================
# 1. STRUCTURED COMPATIBILITY & MATCH RESULT MODELS
# =======================================================


@dataclass
class CompatibilityResult:
    """
    Explicit, structured result for subtitle compatibility assessment.
    Separates hard rejection decisions from soft weighted ranking.
    """

    accepted: bool = True
    score: int = 0
    confidence: float | str = 0.5  # 0.0 to 1.0 or "deterministic"
    match_method: str = "filename"  # "hash", "filename", "fallback"
    match_tier: MatchTier = MatchTier.FALLBACK
    hard_reject_reason: str | None = None
    title_match: float = 0.0
    season_match: bool | None = None
    episode_match: bool | None = None
    release_group_match: bool | None = None
    service_match: bool | None = None
    source_match: bool | None = None
    edition_match: bool | None = None
    fps_relation: str = "unknown"  # "exact", "near", "drift", "unknown"
    is_hash_match: bool = False
    is_sdh: bool = False
    percentage: int = 0
    reasons: list[str] = field(default_factory=list)
    binding: str | None = None
    evidence_confidence: str | None = None
    parsed_target: Any = None
    parsed_candidate: Any = None
    timing_variant_match: bool = False

    def __post_init__(self):
        if self.match_tier == MatchTier.HASH or self.is_hash_match:
            self.is_hash_match = True
            if self.match_method == "filename":
                self.match_method = "hash"


class MatchResult(int):
    """
    Composite match score integer supporting arithmetic comparison (e.g. `res > 130`, `res < -500`),
    tuple unpacking `(score, percentage) = calculate_match_score(...)`, attribute access (`.score`, `.percentage`),
    and structured inspection via `.compatibility`.
    """

    score: int
    percentage: int
    compatibility: CompatibilityResult | None

    def __new__(
        cls, score: int, percentage: int = 0, compatibility: CompatibilityResult | None = None
    ) -> "MatchResult":
        obj = super().__new__(cls, score)
        obj.score = score
        obj.percentage = percentage
        obj.compatibility = compatibility
        return obj

    def __iter__(self):
        yield self.score
        yield self.percentage

    def __getitem__(self, item):
        return (self.score, self.percentage)[item]

    def __repr__(self):
        return f"MatchResult(score={self.score}, percentage={self.percentage}%)"


# =======================================================
# 2. SHARED PARSING VOCABULARY
# =======================================================

# Language tokens and streaming services to never confuse with scene release groups
_LANGUAGE_TOKENS = {
    "arabic",
    "ara",
    "ar",
    "english",
    "eng",
    "en",
    "french",
    "fre",
    "fra",
    "fr",
    "spanish",
    "spa",
    "es",
    "german",
    "ger",
    "deu",
    "de",
    "italian",
    "ita",
    "it",
    "japanese",
    "jap",
    "jpn",
    "ja",
    "korean",
    "kor",
    "ko",
    "chinese",
    "chi",
    "zho",
    "zh",
    "russian",
    "rus",
    "ru",
    "portuguese",
    "por",
    "pt",
    "hindi",
    "hin",
    "hi",
    "subtitle",
    "subtitles",
    "sub",
    "subs",
    "srt",
    "ass",
    "ssa",
    "vtt",
    "amzn",
    "amazon",
    "hmax",
    "hbomax",
    "max",
    "dsnp",
    "disney",
    "netflix",
    "nf",
    "atvp",
    "hulu",
}
_SUBTITLE_PROVIDERS = {
    "subdl",
    "subsource",
    "opensubtitles",
    "opensubtitlesorg",
    "os",
    "addic7ed",
    "yts",
    "ytsmx",
    "yify",
}
_TRANSLATION_PLATFORMS = {
    "animeiat",
    "okanime",
    "fassub",
    "animetitan",
    "cimaclub",
    "phantom",
    "subdl",
    "subsource",
    "opensubtitles",
    "os",
}
_ALL_INVALID_GROUPS = set(_INVALID_GROUPS) | _LANGUAGE_TOKENS | _SUBTITLE_PROVIDERS
_CLEANED_KNOWN_GROUPS = {
    g
    for g in KNOWN_GROUPS
    if g.upper() not in {"AMZN", "MAX", "NF", "HMAX", "DSNP", "ATVP", "HULU", "ARABIC", "ENGLISH"}
    and g.lower() not in _SUBTITLE_PROVIDERS
}
# Fixed scan order (see ranking.KNOWN_GROUPS_SCAN_ORDER): longest first, and
# title words such as "Lost" are only trusted as the trailing "-GROUP" token.
_CLEANED_KNOWN_GROUPS_SCAN_ORDER = tuple(
    sorted(
        (g for g in _CLEANED_KNOWN_GROUPS if g.upper() not in {"LOST", "DON", "LOL"}),
        key=lambda g: (-len(g), g.upper()),
    )
)

# Source hierarchy and family classification
SOURCE_FAMILIES: dict[str, str] = {
    "Remux": "DISC_HI",
    "BluRay": "DISC_HI",
    "UHD": "DISC_HI",
    "BDRip": "DISC_LO",
    "BRRip": "DISC_LO",
    "DVDRip": "DISC_LO",
    "DVD": "DISC_LO",
    "WEB-DL": "WEB",
    "WEBRip": "WEB",
    "WEB": "WEB",
    "HDTV": "TV",
    "PDTV": "TV",
    "DSR": "TV",
    "CAM": "CAM",
}

# =======================================================
# 3. COMPILED REGEX PATTERNS (PRE-COMPILED AT MODULE LEVEL)
# =======================================================

SERVICES = [
    # Global Majors
    (re.compile(r"(?i)\b(hmax|hbomax|max)\b"), "HMAX"),
    (re.compile(r"(?i)\b(amzn|amazon|primevideo)\b"), "AMZN"),
    (re.compile(r"(?i)(?:\b|[._\-])(dsnp|disney(?:\+|plus)?)(?:$|[._\-])"), "DSNP"),
    (re.compile(r"(?i)\b(netflix|nf)\b"), "NF"),
    (re.compile(r"(?i)(?:\b|[._\-])(atvp|appletv(?:\+|plus)?)(?:$|[._\-])"), "ATVP"),
    (re.compile(r"(?i)\b(hulu)\b"), "HULU"),
    (re.compile(r"(?i)\b(pcok|peacock)\b"), "PCOK"),
    (re.compile(r"(?i)(?:\b|[._\-])(pmtp|paramount(?:\+|plus)?)(?:$|[._\-])"), "PMTP"),
    (re.compile(r"(?i)\b(crav|crave)\b"), "CRAV"),
    (re.compile(r"(?i)\b(stan)\b"), "STAN"),
    (re.compile(r"(?i)\b(crunchyroll|cr)\b"), "CR"),
    (re.compile(r"(?i)\b(itunes|it)\b"), "IT"),
    (re.compile(r"(?i)\b(sho|showtime)\b"), "SHO"),
    (re.compile(r"(?i)\b(starz)\b"), "STARZ"),
    (re.compile(r"(?i)\b(bbc|iplayer)\b"), "BBC"),
    (re.compile(r"(?i)\b(cbc|cbcgem)\b"), "CBC"),
    (re.compile(r"(?i)\b(all4|channel4)\b"), "ALL4"),
    (re.compile(r"(?i)\b(roku)\b"), "ROKU"),
    (re.compile(r"(?i)\b(wow|wowpresents)\b"), "WOW"),
    (re.compile(r"(?i)\b(bcore|braviacore|sonycore)\b"), "BCORE"),
    (re.compile(r"(?i)(?:\b|[._\-])(mgm\+|mgmplus|epix)(?:$|[._\-])"), "MGM+"),
    (re.compile(r"(?i)\b(dcu|dcuniverse)\b"), "DCU"),
    # Regional / Premium Cable
    (re.compile(r"(?i)(?:\b|[._\-])(amcp|amc\+|amcplus)(?:$|[._\-])"), "AMCP"),
    (re.compile(r"(?i)\b(amc)\b"), "AMC"),
    (re.compile(r"(?i)\b(binge)\b"), "BINGE"),
    (re.compile(r"(?i)\b(foxt|foxtel)\b"), "FOXT"),
    (re.compile(r"(?i)\b(now|nowtv)\b"), "NOW"),
    (re.compile(r"(?i)\b(skyshowtime|sky)\b"), "SKY"),
    (re.compile(r"(?i)(?:\b|[._\-])(cplus|canal\+|canalplus)(?:$|[._\-])"), "CPLUS"),
    (re.compile(r"(?i)\b(viaplay)\b"), "VIAPLAY"),
    # Asian & Anime Platforms
    (re.compile(r"(?i)\b(hidive)\b"), "HIDIVE"),
    (re.compile(r"(?i)\b(tving)\b"), "TVING"),
    (re.compile(r"(?i)\b(wavve)\b"), "WAVVE"),
    (re.compile(r"(?i)\b(iqiyi|iq)\b"), "IQIYI"),
    (re.compile(r"(?i)\b(wetv)\b"), "WETV"),
    (re.compile(r"(?i)\b(bilibili|bili)\b"), "BILIBILI"),
    (re.compile(r"(?i)\b(viki)\b"), "VIKI"),
    # FAST / AVOD Platforms
    (re.compile(r"(?i)\b(tubi|tubitv)\b"), "TUBI"),
    (re.compile(r"(?i)\b(pluto|plutotv)\b"), "PLUTO"),
    (re.compile(r"(?i)\b(freevee)\b"), "FREEVEE"),
    # Cable & US Networks
    (
        re.compile(r"(?i)(?:\b|[._\-])(disc|discovery\+|discoveryplus|discovery)(?:$|[._\-])"),
        "DISC",
    ),
    (re.compile(r"(?i)\b(hist|history)\b"), "HIST"),
    (re.compile(r"(?i)\b(syfy)\b"), "SYFY"),
    (re.compile(r"(?i)\b(cw)\b"), "CW"),
    (re.compile(r"(?i)\b(fx)\b"), "FX"),
    (re.compile(r"(?i)\b(pbs)\b"), "PBS"),
]

STREAMING_SERVICES = SERVICES

SOURCES = [
    (re.compile(r"(?i)\b(uhd[-._ ]?remux|bdremux|remux)\b"), "Remux"),
    (re.compile(r"(?i)\b(bluray|blu-ray|bd)\b"), "BluRay"),
    (re.compile(r"(?i)\b(uhd|4k-uhd|4k[-._ ]uhd)\b"), "UHD"),
    (re.compile(r"(?i)\b(bdrip|brrip)\b"), "BDRip"),
    (re.compile(r"(?i)\b(web-?dl|webdl)\b"), "WEB-DL"),
    (re.compile(r"(?i)\b(web-?rip|webrip)\b"), "WEBRip"),
    (re.compile(r"(?i)\b(web)\b"), "WEB"),
    (re.compile(r"(?i)\b(hdtv|pdtv|dsr|tvrip)\b"), "HDTV"),
    (re.compile(r"(?i)\b(dvdrip)\b"), "DVDRip"),
    (re.compile(r"(?i)\b(dvd|r5)\b"), "DVD"),
    (re.compile(r"(?i)\b(cam|camrip|hdcam|ts|telesync|hd-ts|hdts|tc|telecine)\b"), "CAM"),
]

_RES_REGEX = re.compile(r"(?i)\b(2160p|4k|uhd|1080p|1080i|720p|576p|576i|480p|480i)\b")

VIDEO_CODECS = [
    (re.compile(r"(?i)\b(x265|hevc|h\.?265)\b"), "x265"),
    (re.compile(r"(?i)\b(x264|avc|h\.?264)\b"), "x264"),
    (re.compile(r"(?i)\b(av1)\b"), "AV1"),
    (re.compile(r"(?i)\b(xvid)\b"), "XviD"),
    (re.compile(r"(?i)\b(divx)\b"), "DivX"),
]

BIT_DEPTHS = [
    (re.compile(r"(?i)\b(10[-._ ]?bit|10b)\b"), "10bit"),
    (re.compile(r"(?i)\b(8[-._ ]?bit|8b)\b"), "8bit"),
]

HDR_PROFILES = [
    (re.compile(r"(?i)\b(dolby[-._ ]?vision|dovi|dv)\b"), "DV"),
    (re.compile(r"(?i)\b(hdr10\+)\b"), "HDR10+"),
    (re.compile(r"(?i)\b(hdr10|hdr)\b"), "HDR"),
    (re.compile(r"(?i)\b(sdr)\b"), "SDR"),
]

AUDIO = [
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
    (re.compile(r"(?i)\b(opus)\b"), "Opus"),
]

EDITIONS = [
    (
        re.compile(
            r"(?i)\b(director'?s?[-._ ]?cut|directors[-._ ]?cut|directors|dc[-._ ]?cut|dir[-._ ]?cut)\b"
        ),
        "DIRECTORS_CUT",
    ),
    (re.compile(r"(?i)\b(extended[-._ ]?(?:cut|edition)?|ext[-._ ]?cut|ext)\b"), "EXTENDED"),
    (re.compile(r"(?i)\b(unrated[-._ ]?(?:cut|edition)?)\b"), "UNRATED"),
    (re.compile(r"(?i)\b(theatrical[-._ ]?(?:cut|edition)?)\b"), "THEATRICAL"),
    (re.compile(r"(?i)\b(imax[-._ ]?(?:enhanced|edition|cut)?)\b"), "IMAX"),
    (re.compile(r"(?i)\b(special[-._ ]?edition|se[-._ ]?cut)\b"), "SPECIAL_EDITION"),
    (re.compile(r"(?i)\b(remastered|remaster)\b"), "REMASTERED"),
]

_REPACK_REGEX = re.compile(r"(?i)\b(repack\d?|proper|rerip)\b")
_REMUX_REGEX = re.compile(r"(?i)\b(uhd[-._ ]?remux|bdremux|remux)\b")
# Re-upload / malformed markers (website URLs, tracker domains) that indicate a
# non-scene name which should lose a tie to a clean scene dot-notation release.
_REUPLOAD_REGEX = re.compile(r"(?i)(?:https?://|www\.|\.(?:com|net|org|info|tv|cc|me|link|club|xyz)\b)")

_FPS_PATTERNS = [
    (re.compile(r"(?i)\b(23\.976(?:fps)?|23\.98(?:fps)?)\b"), 23.976),
    (re.compile(r"(?i)\b(24\.000(?:fps)?|24\.0(?:fps)?|24fps)\b"), 24.0),
    (re.compile(r"(?i)\b(25\.000(?:fps)?|25\.0(?:fps)?|25fps)\b"), 25.0),
    (re.compile(r"(?i)\b(29\.970(?:fps)?|29\.97(?:fps)?|29\.97fps)\b"), 29.97),
    (re.compile(r"(?i)\b(30\.000(?:fps)?|30\.0(?:fps)?|30fps)\b"), 30.0),
    (re.compile(r"(?i)\b(50\.000(?:fps)?|50\.0(?:fps)?|50fps)\b"), 50.0),
    (re.compile(r"(?i)\b(59\.940(?:fps)?|59\.94(?:fps)?|59.94fps)\b"), 59.94),
    (re.compile(r"(?i)\b(60\.000(?:fps)?|60\.0(?:fps)?|60fps)\b"), 60.0),
]

_SDH_REGEX = re.compile(r"(?i)(?:\b(sdh|hearing[-._ ]impaired)\b|\[sdh\]|\.sdh\.)")
_OVA_ONLY_REGEX = re.compile(r"(?i)\b(ova\d*|oad\d*)\b")
_SPECIAL_REGEX = re.compile(
    r"(?i)(?:\b(special|specials)\b|\bsp\s*\d+\b|\bsp\b(?=\s*[-–—_\[0-9]))"
)
_OVA_REGEX = re.compile(
    r"(?i)(?:\b(ova\d*|oad\d*|special|specials)\b|\bsp\s*\d+\b|\bsp\b(?=\s*[-–—_\[0-9]))"
)
_COMPLETE_BATCH_REGEX = re.compile(
    r"(?i)(?:\b(complete|batch|full[-._ ]?season|season[-._ ]?complete|complete[-._ ]?series)\b"
    r"|(?<!E)\bS\d{1,2}\s*[-–—~_]\s*S?\d{1,2}\b(?![._\s-]*E\d)"
    r"|\bseasons?\s*\d{1,2}\s*[-–—~_]\s*\d{1,2}\b)"
)

_SE_EP_REGEX = re.compile(
    r"(?i)\b(?:se|season)[-._ ]?(\d{1,2})[-._ ]*(?:ep|episode|e)[-._ ]?(\d{1,4})\b"
)
# Episode numbers run to four digits (long-running anime: S01E1085).
_S_E_REGEX = re.compile(r"(?i)\bS(\d{1,2})[-._ ]?E(\d{1,4})\b")
_X_E_REGEX = re.compile(r"(?i)\b(\d{1,2})x(\d{1,3})\b")
_SEASON_EP_REGEX = re.compile(r"(?i)\bSeason[-._ ]?(\d{1,2})[-._ ]?Episode[-._ ]?(\d{1,2})\b")
_MULTI_EP_REGEX = re.compile(
    r"(?i)\bS(\d{1,2})[-._ ]?E(\d{1,4})(?:[-–—]\d{1,4}|(?:[-._ ]?E\d{1,4})+)\b"
)
_ORDINAL_SEASON_REGEX = re.compile(r"(?i)\b(\d{1,2})(?:st|nd|rd|th)[-._ ]?season\b")
# Not after an apostrophe: the possessive "s" in "Ocean's.11" is no season.
_SEASON_ONLY_REGEX = re.compile(r"(?i)(?<!['’])\b(?:se|season|s)[-._ ]?(\d{1,2})\b")
_EPISODE_ONLY_REGEX = re.compile(r"(?i)\b(?:ep|episode|e)[-._ ]?(\d{1,3})\b")
_SEASON_MENTION_REGEX = re.compile(
    r"(?i)(?:"
    r"\bS\d{1,2}(?:[-._ ]?E\d{1,3})?"
    r"|"
    r"\b(?:se|season)\s*[-._ ]?\s*\d{1,2}"
    r"|"
    r"\b\d{1,2}x\d{1,3}"
    r"|"
    r"\b\d{1,2}(?:st|nd|rd|th)[-._ ]?season"
    r"|"
    r"\b(?:first|second|third|fourth|fifth|final)[-._ ]?season"
    r")"
)
_YEAR_REGEX = re.compile(r"\b(19\d{2}|20\d{2})\b")

_ANIME_FANSUB_GROUPS = {
    "subsplease",
    "erai-raws",
    "erairaws",
    "kaylith",
    "horriblesubs",
    "judas",
    "asw",
    "commie",
    "coalgirls",
    "fff",
    "doki",
    "hatsuyuki",
    "chyu",
    "neosubs",
    "damedesuyo",
    "cleo",
    "lostyears",
    "dragsterps",
    "smokin",
    "davinci",
    "animerg",
    "ember",
    "kametsu",
    "scy",
    "tlcat",
    "bluraydesu",
    "beetv",
}

_SPEC_TAGS = {
    "1080p",
    "720p",
    "480p",
    "576p",
    "2160p",
    "4k",
    "uhd",
    "hevc",
    "x264",
    "x265",
    "h264",
    "h265",
    "bluray",
    "remux",
    "web-dl",
    "webdl",
    "webrip",
    "web",
    "hdtv",
    "dvd",
    "yts",
    "ytsmx",
    "yify",
    "rarbg",
    "aac",
    "ac3",
    "dts",
    "10bit",
}


def has_season_mention(filename: str) -> bool:
    """Check if filename explicitly contains any season indicator."""
    if not filename:
        return False
    return bool(_SEASON_MENTION_REGEX.search(str(filename)))


def determine_fps_relation(v_fps: float | None, s_fps: float | None) -> str:
    """
    Determine explicit FPS relation between target video and candidate subtitle:
    - 'exact': e.g. 23.976 vs 23.976, 25 vs 25, 29.97 vs 29.97 (diff < 0.005)
    - 'near': e.g. 23.976 vs 24.0, 29.97 vs 30.0, 59.94 vs 60.0 (diff <= 0.06)
    - 'drift': e.g. 23.976 vs 25.0, 24 vs 25 (diff > 0.06)
    - 'unknown': when either FPS is missing
    """
    if v_fps is None or s_fps is None:
        return "unknown"
    diff = abs(v_fps - s_fps)
    if diff < 0.005:
        return "exact"
    pairs = [
        (23.976, 24.0),
        (29.97, 30.0),
        (59.94, 60.0),
    ]
    for p1, p2 in pairs:
        if (abs(v_fps - p1) < 0.05 and abs(s_fps - p2) < 0.05) or (
            abs(v_fps - p2) < 0.05 and abs(s_fps - p1) < 0.05
        ):
            return "near"
    if diff <= 0.06:
        return "near"
    return "drift"


# =======================================================
# 4. METADATA EXTRACTION & HELPER FUNCTIONS
# =======================================================


def extract_leading_bracket_group(filename: str) -> str | None:
    """Extract fansub group strictly from leading bracket tag e.g. [SubsPlease], [Erai-raws], [A&C]."""
    if not filename:
        return None
    m = re.match(r"^\[([^\]]+)\]", str(filename).strip())
    if m:
        g = m.group(1).strip()
        g_low = g.lower()
        if (
            g_low not in _SPEC_TAGS
            and g_low not in _ALL_INVALID_GROUPS
            and g_low not in _SUBTITLE_PROVIDERS
            and not re.match(r"^[0-9A-Fa-f]{8}$", g)
            and not re.match(
                r"^(?:s\d+(?:e\d+)?|e\d+|ep\d+|se\d+|\d+x\d+|season\d+|episode\d+|\d+)$",
                g_low,
            )
        ):
            return g
    return None


def extract_metadata(filename: str) -> dict[str, Any]:
    """
    Comprehensive single-pass parsing of video/torrent or subtitle filename into structured metadata:
    - title: Clean movie/series title
    - year: 4-digit release year
    - content_type: 'movie' or 'series'
    - season: Season integer
    - episode: Primary episode integer
    - episodes: Set of all episodes (for multi-episode files)
    - absolute_episode: Standalone episode number (Anime / continuous numbering)
    - absolute_episode_confidence: 'high', 'medium', 'low', or None
    - service: Streaming platform (HMAX, AMZN, DSNP, NF, etc.)
    - source: Remux, BluRay, UHD, BDRip, WEB-DL, WEBRip, WEB, HDTV, DVD, CAM
    - source_family: DISC_HI, DISC_LO, WEB, TV, CAM
    - resolution: 2160p, 1080p, 720p, 576p, 480p
    - video_codec: x265, x264, AV1, XviD, DivX
    - bit_depth: 10bit, 8bit
    - hdr: DV, HDR10+, HDR, SDR
    - codec: Backward-compatible codec/profile token
    - audio: Atmos, TrueHD, DTS-HD, DTS, DDP5.1, DDP, DD5.1, AC3, AAC, FLAC, Opus
    - audio_codec: Same as audio
    - group: Scene / P2P / Fansub release group
    - edition: THEATRICAL, EXTENDED, DIRECTORS_CUT, UNRATED, SPECIAL_EDITION, REMASTERED, IMAX
    - is_remux: bool
    - is_repack: bool
    - fps: float or None
    - is_sdh: bool
    - is_ova: bool
    - is_special: bool
    - is_complete: bool
    - raw_filename: Original unparsed input string
    """
    if not filename:
        return {
            "title": "",
            "year": None,
            "content_type": "movie",
            "season": None,
            "episode": None,
            "episodes": set(),
            "absolute_episode": None,
            "absolute_episode_confidence": None,
            "service": None,
            "source": None,
            "source_family": None,
            "resolution": None,
            "video_codec": None,
            "bit_depth": None,
            "hdr": None,
            "codec": None,
            "audio": None,
            "audio_codec": None,
            "group": None,
            "edition": None,
            "is_remux": False,
            "is_repack": False,
            "fps": None,
            "is_sdh": False,
            "is_ova": False,
            "is_special": False,
            "is_complete": False,
            "raw_filename": "",
        }

    raw_str = str(filename).strip()
    name = sanitize_release_name(raw_str)

    # 1. Season and Episode
    season: int | None = None
    episode: int | None = None
    episodes: set[int] = set()
    absolute_episode: int | None = None
    absolute_episode_confidence: str | None = None

    m_multi = _MULTI_EP_REGEX.search(name) or _MULTI_EP_REGEX.search(raw_str)
    if m_multi:
        season = int(m_multi.group(1))
        m_range = re.search(r"(?i)E(\d{1,3})\s*[-–—]\s*(?:E)?(\d{1,3})", m_multi.group(0))
        if m_range:
            s_ep, e_ep = int(m_range.group(1)), int(m_range.group(2))
            if 0 < e_ep - s_ep <= 50:
                episodes.update(range(s_ep, e_ep + 1))
        for ep_m in re.finditer(r"(?i)E(\d{1,3})", m_multi.group(0)):
            episodes.add(int(ep_m.group(1)))
        if episodes:
            episode = sorted(episodes)[0]

    if season is None or episode is None:
        m_se_ep = _SE_EP_REGEX.search(name) or _SE_EP_REGEX.search(raw_str)
        if m_se_ep:
            season = int(m_se_ep.group(1))
            episode = int(m_se_ep.group(2))
            episodes = {episode}
        else:
            m_se = _S_E_REGEX.search(name) or _S_E_REGEX.search(raw_str)
            if m_se:
                season = int(m_se.group(1))
                episode = int(m_se.group(2))
                episodes = {episode}
            else:
                m_xe = _X_E_REGEX.search(name) or _X_E_REGEX.search(raw_str)
                if m_xe:
                    season = int(m_xe.group(1))
                    episode = int(m_xe.group(2))
                    episodes = {episode}
                else:
                    m_sep = _SEASON_EP_REGEX.search(name) or _SEASON_EP_REGEX.search(raw_str)
                    if m_sep:
                        season = int(m_sep.group(1))
                        episode = int(m_sep.group(2))
                        episodes = {episode}

    if season is None:
        m_ord = _ORDINAL_SEASON_REGEX.search(name) or _ORDINAL_SEASON_REGEX.search(raw_str)
        if m_ord:
            season = int(m_ord.group(1))
        else:
            m_s = _SEASON_ONLY_REGEX.search(name) or _SEASON_ONLY_REGEX.search(raw_str)
            if m_s:
                season = int(m_s.group(1))

    if episode is None:
        m_e = _EPISODE_ONLY_REGEX.search(name) or _EPISODE_ONLY_REGEX.search(raw_str)
        if m_e:
            episode = int(m_e.group(1))
            episodes = {episode}

    # 2. Release Year
    year: int | None = None
    for y_m in _YEAR_REGEX.finditer(name):
        val = int(y_m.group(1))
        if 1900 <= val <= 2099:
            year = val
            break

    # 3. Absolute Episode Parsing for Anime
    if season is None or episode is None:
        anime_str = name.replace("_", " ")
        anime_str = re.sub(r"\[[0-9A-Fa-f]{8}\]", "", anime_str)
        anime_str = re.sub(
            r"(?i)\b(2160p?|1080p?|720p?|576p?|480p?|x264|x265|h\.?264|h\.?265|av1|10bit|8bit|[0-9]\.[0-9]|ddp5\.1|dd5\.1)\b",
            "",
            anime_str,
        )
        if year:
            anime_str = re.sub(rf"\b{year}\b", "", anime_str)

        # High-confidence standalone bare or bracketed episode (e.g. "02", "2", "[Animeiat] 02", "02 - Arabic", "2 - Arabic")
        bare = re.sub(r"^\s*\[[^\]]+\]\s*", "", anime_str).strip()
        m_bare = re.match(r"^([1-9]\d{0,3})\b", bare)
        # A leading number followed later by the release year is a title
        # ("300.2006", "21.Jump.Street.2012"), not an episode.
        if m_bare and year and re.search(
            rf"(?<!\d){re.escape(m_bare.group(1))}\D.*\b{year}\b", name
        ):
            m_bare = None

        m_dash = re.search(r"[-–—]\s*(?:e|ep|episode)?\s*(\d{1,4})\b", anime_str, re.IGNORECASE)
        m_tag = re.search(r"(?i)(?:^|[\s._\-\[#])(?:ep|episode|e)\s*0*(\d{1,4})\b", anime_str)
        m_hash = re.search(r"#\s*(\d{1,4})\b", anime_str)
        m_brack = re.search(r"\[\s*0*(\d{1,4})\s*\]", raw_str) or re.search(
            r"\[\s*0*(\d{1,4})\s*\]", anime_str
        )
        m_leading_zero = re.search(r"(?:^|[\s._\-])0([1-9])\b", anime_str)
        m_end_digit = re.search(r"[\s._\-]([1-9])\s*$", anime_str)
        m_multi_digit = re.search(r"(?<!\d)(\d{3,4})(?!\d)", anime_str)
        m_two_digit = re.search(r"(?:^|[\s._\-])([1-9]\d)\b", anime_str)

        cand_val: int | None = None
        cand_conf: str | None = None
        range_eps: set[int] | None = None

        m_anime_range = re.search(
            r"[-–—~]\s*(?:ep|episode|e)?\s*(\d{1,4})\s*[-–—~]\s*(?:ep|episode|e)?\s*(\d{1,4})\b",
            anime_str,
            re.IGNORECASE,
        )
        if not m_anime_range:
            m_anime_range = re.search(
                r"\b(?:ep|episode|e)\s*(\d{1,4})\s*[-–—~]\s*(?:ep|episode|e)?\s*(\d{1,4})\b",
                anime_str,
                re.IGNORECASE,
            )
        if not m_anime_range:
            m_anime_range = re.search(
                r"(?<!\d)(\d{1,4})\s*[-–—~]\s*(\d{1,4})(?!\d)",
                anime_str,
            )

        if m_anime_range and (
            0 < int(m_anime_range.group(2)) - int(m_anime_range.group(1)) <= 200
        ):
            cand_val = int(m_anime_range.group(1))
            cand_conf = "high"
            range_eps = set(range(int(m_anime_range.group(1)), int(m_anime_range.group(2)) + 1))
        elif (
            m_bare
            and int(m_bare.group(1)) not in (2160, 1080, 720, 576, 480, 264, 265)
            and not (1900 <= int(m_bare.group(1)) <= 2099)
        ):
            cand_val = int(m_bare.group(1))
            cand_conf = "high"
        elif m_dash:
            cand_val = int(m_dash.group(1))
            cand_conf = "high"
        elif m_tag:
            cand_val = int(m_tag.group(1))
            cand_conf = "high"
        elif m_hash:
            cand_val = int(m_hash.group(1))
            cand_conf = "high"
        elif m_brack:
            cand_val = int(m_brack.group(1))
            cand_conf = "high"
        elif m_leading_zero:
            cand_val = int(m_leading_zero.group(1))
            cand_conf = "high"
        elif m_end_digit:
            cand_val = int(m_end_digit.group(1))
            cand_conf = "high"
        elif not year and season is None:
            if m_multi_digit:
                val = int(m_multi_digit.group(1))
                if val not in (2160, 1080, 720, 576, 480, 264, 265) and not (1900 <= val <= 2099):
                    cand_val = val
                    cand_conf = "medium"
            elif m_two_digit:
                val = int(m_two_digit.group(1))
                if val not in (2160, 1080, 720, 576, 480, 264, 265):
                    cand_val = val
                    cand_conf = "medium"

        if cand_val is not None:
            if (
                not (year and cand_val == year)
                and cand_val not in (2160, 1080, 720, 576, 480, 264, 265)
                and not (1900 <= cand_val <= 2099)
            ):
                absolute_episode = cand_val
                absolute_episode_confidence = cand_conf
                if episode is None:
                    episode = cand_val
                    episodes = range_eps if range_eps else {cand_val}

    # 4. Streaming Service
    service: str | None = None
    for pattern, s_key in SERVICES:
        if pattern.search(name) or pattern.search(raw_str):
            service = s_key
            break

    # 5. Media Source & Source Family
    source: str | None = None
    for pattern, src_key in SOURCES:
        if pattern.search(name) or pattern.search(raw_str):
            source = src_key
            break

    is_remux = bool(_REMUX_REGEX.search(name) or _REMUX_REGEX.search(raw_str))
    if is_remux:
        source = "Remux"

    if service and not source:
        source = "WEB-DL"

    source_family = SOURCE_FAMILIES.get(source) if source else None
    is_repack = bool(_REPACK_REGEX.search(name) or _REPACK_REGEX.search(raw_str))

    # 6. Edition / Cut
    edition: str | None = None
    for pattern, ed_key in EDITIONS:
        if pattern.search(name) or pattern.search(raw_str):
            edition = ed_key
            break

    # 7. FPS
    fps: float | None = None
    for pattern, fps_val in _FPS_PATTERNS:
        if pattern.search(name) or pattern.search(raw_str):
            fps = fps_val
            break

    # 8. SDH, OVA, Special, Complete/Batch
    is_sdh = bool(_SDH_REGEX.search(name) or _SDH_REGEX.search(raw_str))
    is_ova = bool(_OVA_ONLY_REGEX.search(name) or _OVA_ONLY_REGEX.search(raw_str))
    is_special = bool(_SPECIAL_REGEX.search(name) or _SPECIAL_REGEX.search(raw_str))
    is_complete = bool(_COMPLETE_BATCH_REGEX.search(name) or _COMPLETE_BATCH_REGEX.search(raw_str))

    # 9. Resolution
    resolution: str | None = None
    res_m = _RES_REGEX.search(name) or _RES_REGEX.search(raw_str)
    if res_m:
        res_tok = res_m.group(1).lower()
        if res_tok in ("2160p", "4k", "uhd"):
            resolution = "2160p"
        elif res_tok in ("1080p", "1080i"):
            resolution = "1080p"
        elif res_tok in ("720p",):
            resolution = "720p"
        elif res_tok in ("576p", "576i"):
            resolution = "576p"
        elif res_tok in ("480p", "480i"):
            resolution = "480p"

    # 10. Video Codecs, Bit Depth & HDR Profiles
    video_codec: str | None = None
    for pattern, c_key in VIDEO_CODECS:
        if pattern.search(name) or pattern.search(raw_str):
            video_codec = c_key
            break

    bit_depth: str | None = None
    for pattern, bd_key in BIT_DEPTHS:
        if pattern.search(name) or pattern.search(raw_str):
            bit_depth = bd_key
            break

    hdr: str | None = None
    for pattern, hdr_key in HDR_PROFILES:
        if pattern.search(name) or pattern.search(raw_str):
            hdr = hdr_key
            break

    codec: str | None = video_codec or hdr or bit_depth

    # 11. Audio Profile
    audio: str | None = None
    for pattern, a_key in AUDIO:
        if pattern.search(name) or pattern.search(raw_str):
            audio = a_key
            break

    # 12. Release Group
    group: str | None = None
    leading_g = extract_leading_bracket_group(raw_str)
    if leading_g:
        group = leading_g
    elif "-" in name:
        cand = name.rsplit("-", 1)[-1].strip()
        cand = re.sub(r"\[[^\]]*\]$", "", cand).strip()
        cand = cand.strip(" ._-[]()")
        cand_tokens = [t.lower() for t in re.split(r"[\s._\-+]+", cand) if t]
        if (
            cand
            and cand.lower() not in _ALL_INVALID_GROUPS
            and len(cand) >= 2
            and not cand.isdigit()
            and " " not in cand
            and not cand[0].isdigit()
            and not any(t in _SPEC_TAGS or t in _ALL_INVALID_GROUPS for t in cand_tokens)
        ):
            group = cand

    if not group:
        b_m = re.search(r"\[([A-Za-z0-9_.]+)\]", name)
        if b_m:
            cand = b_m.group(1).strip(" ._-\t[]()")
            if (
                cand
                and cand.lower() not in _ALL_INVALID_GROUPS
                and cand.lower() not in _SUBTITLE_PROVIDERS
                and cand.lower() not in _SPEC_TAGS
                and len(cand) >= 2
                and not cand.isdigit()
                and not re.match(
                    r"(?i)^(?:s\d+(?:e\d+)?|e\d+|ep\d+|se\d+|\d+x\d+|season\d+|episode\d+|bd|dvd|\d+)$",
                    cand,
                )
            ):
                group = cand

    if not group:
        tokens = [t for t in re.split(r"[\s._\-+]+", name) if t]
        if tokens:
            last_tok = tokens[-1].strip(" ._-\t[]()")
            has_media_specs = bool(source or resolution or video_codec or service or is_remux)
            if last_tok.upper() in _CLEANED_KNOWN_GROUPS or (
                has_media_specs
                and len(last_tok) >= 2
                and last_tok.lower() not in _ALL_INVALID_GROUPS
                and last_tok.lower() not in _SUBTITLE_PROVIDERS
                and last_tok.lower() not in _SPEC_TAGS
                and not last_tok.isdigit()
                and not re.match(
                    r"(?i)^(?:s\d+(?:e\d+)?|e\d+|ep\d+|se\d+|\d+x\d+|season\d+|episode\d+|bd|dvd|\d+)$",
                    last_tok,
                )
                and not any(last_tok.lower() == s_key.lower() for _, s_key in SERVICES)
            ):
                group = last_tok

    if not group:
        for kg in _CLEANED_KNOWN_GROUPS_SCAN_ORDER:
            if re.search(rf"\b{re.escape(kg)}\b", name, re.IGNORECASE):
                group = kg
                break

    if group:
        group = group.strip(" ._-\t[]()")
        if (
            re.match(
                r"(?i)^(?:s\d+(?:e\d+)?|e\d+|ep\d+|se\d+|\d+x\d+|season\d+|episode\d+|bd|dvd|\d+)$",
                group,
            )
            or group.lower() in _ALL_INVALID_GROUPS
            or group.lower() in _SUBTITLE_PROVIDERS
            or group.lower() in _SPEC_TAGS
        ):
            group = None

    # Supplemental metadata extraction using guessit when fields are missing
    needs_guessit = (
        not group
        or not source
        or not resolution
        or not video_codec
        or (season is None and episode is None and absolute_episode is None)
    )

    g_meta = {}
    if needs_guessit:
        try:
            from app.services.sync.matching import guess_metadata

            g_meta = guess_metadata(raw_str)
        except Exception:
            g_meta = {}

    if g_meta:
        if not group and g_meta.get("release_group"):
            g_cand = g_meta.get("release_group")
            if isinstance(g_cand, list):
                g_cand = g_cand[0]
            if isinstance(g_cand, str):
                g_cand = g_cand.strip(" ._-\t[]()")
                if (
                    g_cand
                    and g_cand.lower() not in _ALL_INVALID_GROUPS
                    and g_cand.lower() not in _SUBTITLE_PROVIDERS
                    and g_cand.lower() not in _SPEC_TAGS
                    and len(g_cand) >= 2
                    and not g_cand.isdigit()
                    and not re.match(
                        r"(?i)^(?:s\d+(?:e\d+)?|e\d+|ep\d+|se\d+|\d+x\d+|season\d+|episode\d+|bd|dvd|\d+)$",
                        g_cand,
                    )
                ):
                    group = g_cand

        if not edition and g_meta.get("edition"):
            ed_raw = str(g_meta.get("edition")).lower()
            for pattern, ed_key in EDITIONS:
                if pattern.search(ed_raw):
                    edition = ed_key
                    break

        if not service and g_meta.get("streaming_service"):
            srv_raw = str(g_meta.get("streaming_service")).lower()
            for pattern, s_key in SERVICES:
                if pattern.search(srv_raw):
                    service = s_key
                    break

        if not resolution and g_meta.get("screen_size"):
            sc_raw = str(g_meta.get("screen_size")).lower()
            res_m = _RES_REGEX.search(sc_raw)
            if res_m:
                res_tok = res_m.group(1).lower()
                if res_tok in ("2160p", "4k", "uhd"):
                    resolution = "2160p"
                elif res_tok in ("1080p", "1080i"):
                    resolution = "1080p"
                elif res_tok in ("720p",):
                    resolution = "720p"
                elif res_tok in ("576p", "576i"):
                    resolution = "576p"
                elif res_tok in ("480p", "480i"):
                    resolution = "480p"

        if not video_codec and g_meta.get("video_codec"):
            vc_raw = str(g_meta.get("video_codec")).lower()
            for pattern, c_key in VIDEO_CODECS:
                if pattern.search(vc_raw):
                    video_codec = c_key
                    break
            if video_codec and not codec:
                codec = video_codec

        if not source:
            other = g_meta.get("other")
            if other and ("Remux" in other if isinstance(other, list) else other == "Remux"):
                source = "Remux"
                is_remux = True
            elif g_meta.get("source"):
                src_raw = str(g_meta.get("source")).lower()
                for pattern, src_key in SOURCES:
                    if pattern.search(src_raw):
                        source = src_key
                        break
            if source:
                source_family = SOURCE_FAMILIES.get(source)

        if year is None and g_meta.get("year"):
            try:
                y_val = int(g_meta["year"])
                if 1900 <= y_val <= 2099:
                    year = y_val
            except (ValueError, TypeError):
                pass

        if season is None and absolute_episode is None and not episodes and g_meta.get("season"):
            try:
                s_val = int(
                    g_meta["season"][0]
                    if isinstance(g_meta["season"], list)
                    else g_meta["season"]
                )
                if 0 < s_val <= 99:
                    season = s_val
            except (ValueError, TypeError):
                pass

        if episode is None and absolute_episode is None and not episodes and g_meta.get("episode"):
            try:
                ep_val = int(
                    g_meta["episode"][0]
                    if isinstance(g_meta["episode"], list)
                    else g_meta["episode"]
                )
                if ep_val > 0:
                    episode = ep_val
                    episodes = {ep_val}
            except (ValueError, TypeError):
                pass

    if service and not source:
        source = "WEB-DL"
        source_family = "WEB"

    # 13. Clean Title Extraction
    title = _extract_pure_title(raw_str, year, season, episode)

    # 14. Content Type Identification
    content_type = (
        "series"
        if (
            season is not None
            or episode is not None
            or absolute_episode is not None
            or is_complete
            or has_season_mention(raw_str)
        )
        else "movie"
    )

    return {
        "title": title,
        "year": year,
        "content_type": content_type,
        "season": season,
        "episode": episode,
        "episodes": episodes,
        "absolute_episode": absolute_episode,
        "absolute_episode_confidence": absolute_episode_confidence,
        "service": service,
        "source": source,
        "source_family": source_family,
        "resolution": resolution,
        "video_codec": video_codec,
        "bit_depth": bit_depth,
        "hdr": hdr,
        "codec": codec,
        "audio": audio,
        "audio_codec": audio,
        "group": group,
        "edition": edition,
        "is_remux": is_remux,
        "is_repack": is_repack,
        "fps": fps,
        "is_sdh": is_sdh,
        "is_ova": is_ova,
        "is_special": is_special,
        "is_complete": is_complete,
        "raw_filename": raw_str,
    }


parse_video_metadata = extract_metadata


def _extract_pure_title(
    filename: str,
    year: int | None = None,
    season: int | None = None,
    episode: int | None = None,
) -> str:
    """Extract clean title stripped of year, season, episode, spec tokens, and release groups."""
    clean = sanitize_release_name(filename)
    clean = clean.replace("_", " ")
    clean = re.sub(r"^\s*\[?[A-Za-z0-9_.\-\s]{2,25}\]\s*", "", clean)
    clean = re.sub(r"-[A-Za-z0-9_.]+$", "", clean)

    tech_tags = (
        r"(?i)\b(2160p|4k|1080p|1080i|720p|576p|480p|uhd|remux|bdremux|bluray|blu-ray|bdrip|brrip|"
        r"web-?dl|webdl|web-?rip|webrip|hdtv|pdtv|dvd|dvdrip|cam|telesync|ts|"
        r"x265|hevc|h\.?265|x264|h\.?264|avc|av1|xvid|divx|10bit|hdr|hdr10\+?|dv|dovi|dolby[-._ ]?vision|"
        r"dts-hd(?:[-._ ]?ma)?|dts|truehd|atmos|ddp5\.1|dd\+5\.1|eac3|ddp|dd\+|dd5\.1|ac3|aac|flac|opus|"
        r"hmax|hbomax|amzn|amazon|dsnp|disney|netflix|nf|atvp|appletv|hulu|pcok|peacock|pmtp|paramount|"
        r"crav|crave|stan|crunchyroll|cr|itunes|it|sho|showtime|starz|bbc|iplayer|cbc|all4|roku|wow|bcore|"
        r"amcp|amc|binge|foxt|foxtel|now|nowtv|skyshowtime|sky|cplus|canal|viaplay|hidive|tving|wavve|"
        r"iqiyi|wetv|bilibili|viki|tubi|tubitv|pluto|plutotv|freevee|disc|discovery|hist|history|syfy|"
        r"cw|fx|pbs|mgm|epix|dcu|"
        r"repack\d?|proper|rerip|extended|unrated|directors|theatrical|imax|special[-._ ]?edition|remastered|"
        r"multi|dual|ova\d*|oad\d*|special|specials|complete|batch)\b"
    )
    clean = re.sub(tech_tags, "", clean)
    clean = re.sub(
        r"(?i)\b(?:se|season)[-._ ]?\d{1,2}[-._ ]*(?:ep|episode|e)[-._ ]?\d{1,3}\b", "", clean
    )
    clean = re.sub(r"(?i)\bS\d{1,2}[-._ ]?E\d{1,2}(?:[-._ ]?E\d{1,2})*\b", "", clean)
    clean = re.sub(r"(?i)\b\d{1,2}x\d{1,2}\b", "", clean)
    clean = re.sub(r"(?i)\b\d{1,2}(?:st|nd|rd|th)[-._ ]?season\b", "", clean)
    clean = re.sub(r"(?i)\b(?:se|season|s)[-._ ]?\d{1,2}\b", "", clean)
    clean = re.sub(r"(?i)\b(?:ep|episode|e)[-._ ]?\d{1,3}\b", "", clean)
    clean = re.sub(r"(?i)(?:[-–—]\s*)\d{1,4}\b", "", clean)
    clean = re.sub(r"#\s*\d{1,4}\b", "", clean)
    clean = re.sub(
        r"(?i)\b(arabic|english|spanish|french|german|italian|russian|portuguese|hindi|japanese|korean|chinese)\b",
        "",
        clean,
    )

    if year:
        clean = re.sub(rf"\b{year}\b", "", clean)
    elif episode is not None:
        clean = re.sub(rf"\b0*{episode}\b", "", clean)

    clean = re.sub(r"\b[0-9A-Fa-f]{8}\b", "", clean)
    clean = re.sub(r"[\[\](){}]", " ", clean)
    clean = re.sub(r"[\s._\-+]+", " ", clean).strip().lower()
    return clean


def is_anime_content(
    video_filename: str | dict[str, Any] = "",
    sub_filename: str | dict[str, Any] | SubtitleRelease = "",
    metadata: dict[str, Any] | None = None,
) -> bool:
    """
    Detect whether the target video or subtitle release is Anime content.
    Requires reliable anime indicators:
    - Known anime/fansub groups
    - Anime streaming services (CR, HIDIVE, BILIBILI, VIKI)
    - Explicit anime keywords
    - OVA/OAD indicators
    - Reliable anime-style episode numbering
    Never classifies generic [Group] (e.g. [FLUX]), [WEB], [1080p], or normal movie releases as anime.
    """
    if isinstance(video_filename, dict):
        v_raw = video_filename.get("raw_filename", "")
        v_meta = video_filename
    else:
        v_raw = str(video_filename or "")
        v_meta = metadata if metadata else (extract_metadata(v_raw) if v_raw else {})

    if isinstance(sub_filename, dict):
        s_raw = sub_filename.get("raw_filename", "")
        s_meta = sub_filename
    elif hasattr(sub_filename, "release_name"):
        s_raw = str(getattr(sub_filename, "release_name", ""))
        s_meta = extract_metadata(s_raw) if s_raw else {}
    else:
        s_raw = str(sub_filename or "")
        s_meta = extract_metadata(s_raw) if s_raw else {}

    combined_text = f"{v_raw} {s_raw}".strip()
    if not combined_text:
        return False

    # 1. Known fansub groups in leading brackets or text
    for fn in (v_raw, s_raw):
        bg = extract_leading_bracket_group(fn)
        if bg:
            bg_lower = bg.lower()
            if bg_lower in _ANIME_FANSUB_GROUPS or re.search(r"(?:subs|raws|fansub)", bg_lower):
                return True

    for fg in _ANIME_FANSUB_GROUPS:
        if re.search(rf"\b{re.escape(fg)}\b", combined_text, re.IGNORECASE):
            return True

    # 2. Anime streaming services & keywords
    services = {v_meta.get("service"), s_meta.get("service")}
    if services & {"CR", "HIDIVE", "BILIBILI", "VIKI"}:
        return True

    if re.search(r"(?i)\b(anime|fansub|crunchyroll|hidive|bilibili)\b", combined_text):
        return True

    # 3. OVA / OAD indicators
    if v_meta.get("is_ova") or s_meta.get("is_ova"):
        return True
    if re.search(r"(?i)\b(?:ova|oad)\d*\b", combined_text):
        return True

    # Guard: Standard TV with explicit season mention on both files without anime tags is regular TV
    if has_season_mention(v_raw) and has_season_mention(s_raw):
        return False

    # 4. Reliable anime numbering patterns:
    for m in re.finditer(r"[-–—]\s*(?:e|ep|episode)?\s*(\d{1,4})\b", combined_text, re.IGNORECASE):
        n = int(m.group(1))
        if n not in (2160, 1080, 720, 576, 480, 264, 265) and not (1900 <= n <= 2099):
            return True

    if re.search(r"#\s*\d{1,4}\b", combined_text):
        return True

    for m in re.finditer(r"\[\s*0*(\d{1,4})\s*\]", combined_text):
        n = int(m.group(1))
        if n not in (2160, 1080, 720, 576, 480, 264, 265) and not (1900 <= n <= 2099):
            return True

    for m in re.finditer(r"(?i)(?:^|[\s._\-\[#])(?:ep|episode|e)\s*0*(\d{1,4})\b", combined_text):
        n = int(m.group(1))
        if n not in (2160, 1080, 720, 576, 480, 264, 265) and not (1900 <= n <= 2099):
            return True

    # A bare number is only an episode number if it is not part of a codec
    # token. ``x254``/``x264``/``x266`` are codec names, and reading them as
    # episodes classified ordinary Hollywood features as anime purely because of
    # the encoder in their release name -- which sent them down the anime scoring
    # path and demoted a UHD BluRay release below a 1080p micro-rip.
    for m in re.finditer(r"(?<![\dxX])(\d{3,4})(?!\d)", combined_text):
        n = int(m.group(1))
        if n not in (2160, 1080, 720, 576, 480, 264, 265) and not (1900 <= n <= 2099):
            return True

    if (
        v_meta.get("absolute_episode_confidence") == "high"
        or s_meta.get("absolute_episode_confidence") == "high"
    ):
        return True

    return False


# =======================================================
# 5. STAGE 1: HARD COMPATIBILITY FILTER
# =======================================================


def hard_compatibility_filter(
    target_meta: dict[str, Any],
    cand_meta: dict[str, Any],
    is_hash_match: bool = False,
) -> tuple[bool, str | None, str]:
    """
    Stage 1: Hard Compatibility Filter.
    Rejects only when the mismatch is explicit, unambiguous, and reliable.
    Returns: (accepted: bool, hard_reject_reason: Optional[str], match_method: str)
    """
    if is_hash_match:
        return True, None, "hash"

    v_raw = target_meta.get("raw_filename", "")
    s_raw = cand_meta.get("raw_filename", "")

    v_year = target_meta.get("year")
    v_season = target_meta.get("season")
    v_ep = target_meta.get("episode")
    v_eps = target_meta.get("episodes") or set()
    v_abs = target_meta.get("absolute_episode")
    v_abs_conf = target_meta.get("absolute_episode_confidence")
    v_is_ova = target_meta.get("is_ova", False)
    v_is_special = target_meta.get("is_special", False)
    v_is_spec = v_is_ova or v_is_special or bool(_OVA_REGEX.search(v_raw))

    s_season = cand_meta.get("season")
    s_ep = cand_meta.get("episode")
    s_eps = cand_meta.get("episodes") or set()
    s_abs = cand_meta.get("absolute_episode")
    s_abs_conf = cand_meta.get("absolute_episode_confidence")
    s_is_ova = cand_meta.get("is_ova", False)
    s_is_special = cand_meta.get("is_special", False)
    s_is_spec = s_is_ova or s_is_special or bool(_OVA_REGEX.search(s_raw))
    s_is_complete = cand_meta.get("is_complete", False)

    is_anime = is_anime_content(target_meta, cand_meta)
    target_has_ep = v_ep is not None or v_abs is not None

    # 1. Batch / Complete release vs individual episode
    if target_has_ep and not v_is_spec:
        if s_is_complete or bool(_COMPLETE_BATCH_REGEX.search(s_raw)):
            target_num = v_abs if v_abs is not None else v_ep
            if not (
                target_num is not None
                and (
                    target_num in s_eps
                    or (s_season is not None and v_season == s_season and target_num in s_eps)
                )
            ):
                return (
                    False,
                    "Batch/complete release cannot represent individual episode",
                    "filename",
                )

    # 2. OVA / Special vs regular episode mismatch
    if target_has_ep:
        if v_is_spec and not s_is_spec:
            return False, "Target is OVA/Special but candidate is regular episode", "filename"
        if not v_is_spec and s_is_spec:
            return False, "Candidate is OVA/Special but target is regular episode", "filename"

    # 3. Content Type Mismatch (Movie vs TV Episode and vice-versa)
    _MOVIE_TAG_REGEX = re.compile(r"(?i)\b(the\s+movie|theatrical\s+film)\b")

    is_target_movie = v_season is None and v_ep is None and v_abs is None
    if is_target_movie:
        if s_season is not None and s_ep is not None:
            return False, "Target is a movie but candidate is an explicit TV episode", "filename"
        if s_season is not None and not has_season_mention(v_raw):
            return False, "Target is a movie but candidate has explicit season", "filename"
        if s_ep is not None and not has_season_mention(v_raw) and not (v_year and s_ep == v_year):
            return False, "Target is a movie but candidate has explicit episode", "filename"

    # Vice-versa: Target is episodic series, but candidate is explicitly a movie
    if (target_has_ep or v_season is not None) and not _MOVIE_TAG_REGEX.search(v_raw):
        if _MOVIE_TAG_REGEX.search(s_raw):
            target_num = v_abs if v_abs is not None else v_ep
            if target_num is None or (target_num not in s_eps and s_ep != target_num):
                return (
                    False,
                    "Target is an episodic series but candidate is explicitly a movie",
                    "filename",
                )

    # 4. Explicit season mismatch (e.g. S01 vs S02)
    v_is_single_abs = is_anime and (v_abs is not None) and not has_season_mention(v_raw)
    s_is_single_abs = is_anime and (s_abs is not None) and not has_season_mention(s_raw)
    v_eff_season = v_season if v_season is not None else (1 if v_is_single_abs else None)
    s_eff_season = s_season if s_season is not None else (1 if s_is_single_abs else None)

    if v_eff_season is not None and s_eff_season is not None and v_eff_season != s_eff_season:
        return (
            False,
            f"Explicit season mismatch (Season {v_eff_season} vs Season {s_eff_season})",
            "filename",
        )

    if v_eff_season is not None and s_eff_season is None and has_season_mention(s_raw):
        return False, "Candidate mentions conflicting season", "filename"
    if s_eff_season is not None and v_eff_season is None and has_season_mention(v_raw):
        return False, "Target mentions conflicting season", "filename"

    # 5. Explicit episode mismatch & multi-episode acceptance
    if is_anime:
        v_target_ep = v_abs if v_abs is not None else v_ep
        s_target_ep = s_abs if s_abs is not None else s_ep
        if v_target_ep is not None and s_target_ep is not None:
            if v_target_ep != s_target_ep and v_target_ep not in s_eps and s_target_ep not in v_eps:
                if (
                    v_abs is not None
                    and s_abs is not None
                    and v_abs_conf == "high"
                    and s_abs_conf == "high"
                ):
                    return (
                        False,
                        f"Anime absolute episode mismatch ({v_abs} vs {s_abs})",
                        "filename",
                    )
                return False, f"Anime episode mismatch ({v_target_ep} vs {s_target_ep})", "filename"
    else:
        if v_ep is not None and s_ep is not None:
            if v_ep != s_ep and v_ep not in s_eps and s_ep not in v_eps:
                return False, f"Explicit episode mismatch (E{v_ep} vs E{s_ep})", "filename"

    # 6. Explicit edition conflict (e.g. Extended vs Theatrical)
    v_edition = target_meta.get("edition")
    s_edition = cand_meta.get("edition")
    if v_edition and s_edition and v_edition != s_edition:
        return False, f"Explicit edition conflict ({v_edition} vs {s_edition})", "filename"

    return True, None, "filename"


# =======================================================
# 6. STAGE 2: SOFT COMPATIBILITY RANKING
# =======================================================


# A candidate whose title shares essentially nothing with the video is a
# different work, so it can never earn a source/group tier no matter how well
# its medium happens to line up. Only the generic fallback tier is reachable.
# Season/episode identity exempts a candidate because long series titles often
# dilute the ratio (a subtitle carrying only "Show.S02E04" legitimately scores
# low against a full episode title) while still being the same episode.
MIN_TIER_TITLE_RATIO = 0.5


def _normalize_group(group: str | None) -> str:
    """Case- and separator-insensitive release-group key.

    Delegates to the shared implementation used by reference selection so both
    paths agree on what "same group" means (``PiR8`` == ``pir8`` == ``Pi_R8``).
    Imported lazily because ``app.services.sync`` pulls in the orchestrator,
    which imports this module.
    """
    try:
        from app.services.sync.matching import normalize_release_group
    except ImportError:  # pragma: no cover - defensive
        return re.sub(r"[^a-z0-9]", "", (group or "").strip().lower())
    return normalize_release_group(group)


def _is_different_work(
    title_ratio: float,
    episode_match: bool | None,
    season_match: bool | None,
) -> bool:
    """True when the candidate almost certainly belongs to a different work."""
    if title_ratio >= MIN_TIER_TITLE_RATIO:
        return False
    return episode_match is not True and season_match is not True


# Release groups whose output is a re-encode/mirror of someone else's release
# rather than its own capture. Used only to *demote* a candidate one tier (see
# ``determine_match_tier``); never to reject one, and never by itself. Derived
# from the existing KNOWN_GROUPS vocabulary so no group name is duplicated here.
_MICRO_RIP_GROUPS = frozenset({"YIFY", "YTS", "YTS.MX", "RARBG"})


def _is_micro_rip_group(group: str | None) -> bool:
    """True for a known re-encode/mirror release group."""
    if not group:
        return False
    normalized = re.sub(r"[^A-Z0-9.]", "", group.strip().upper())
    return normalized in _MICRO_RIP_GROUPS


def _micro_rip_release(release_group: str | None, raw_name: str | None) -> bool:
    """Micro-rip detection over both parsers.

    ``extract_metadata`` only records a group when it is dash-delimited, so a
    trailing ``...x264.YIFY`` arrives here as ``None``. The shared
    ``ranking.parse_release_metadata`` does resolve it, so it is consulted rather
    than adding a second regex for the same vocabulary.
    """
    if _is_micro_rip_group(release_group):
        return True
    if not raw_name:
        return False
    try:
        from app.services.ranking import parse_release_metadata

        return _is_micro_rip_group(parse_release_metadata(raw_name).get("group"))
    except Exception:  # pragma: no cover - defensive
        return False


def _is_high_tier_target(resolution: str | None, source: str | None) -> bool:
    """True when the playback file is a UHD/REMUX-class source.

    Evidence-based and narrow: 2160p/4K/UHD, or an explicit REMUX. A 1080p
    WEB-DL target is deliberately *not* high tier, so ordinary streaming
    releases are unaffected by the micro-rip demotion below.
    """
    res = (resolution or "").strip().lower()
    src = (source or "").strip().lower()
    return res in {"2160p", "4k", "uhd"} or src == "remux"


def determine_match_tier(video_meta, sub_meta, is_hash_match=False):
    """Compatibility label only, never a ranking ladder."""
    return calculate_compatibility(video_meta, sub_meta, is_hash_match).match_tier





def calculate_compatibility(video_meta, sub_meta, is_hash_match=False) -> CompatibilityResult:
    """Compatibility API backed only by the weighted authority; a flag is not hash proof."""
    from app.services.release_scoring_integration import compare_metadata
    result = compare_metadata(video_meta, sub_meta)
    states = {e.field: e.state.value for e in result.evidence}
    # A hard-rejected candidate is labelled FALLBACK, never a compatibility tier:
    # it carries a -1000 score and must not read as a near match.
    tier = (MatchTier.HASH if result.verified_hash else MatchTier.FALLBACK
            if not result.discovery_eligible else MatchTier.EXACT
            if states.get("release_group") == "match" and states.get("source") == "match"
            else MatchTier.SOURCE_FAMILY if states.get("source") == "match" else MatchTier.CLOSE)
    return CompatibilityResult(accepted=result.discovery_eligible, score=result.ranking_score if result.discovery_eligible else -1000,
        percentage=round(result.ranking_percentage) if result.discovery_eligible else 0, match_tier=tier,
        hard_reject_reason=','.join(result.rejection_reasons) or None,
        reasons=list(result.rejection_reasons), is_hash_match=result.verified_hash,
        binding="exact" if result.verified_hash else "strong" if tier == MatchTier.EXACT else "weak" if tier == MatchTier.SOURCE_FAMILY else "unknown",
        source_match=states.get("source") == "match", release_group_match=states.get("release_group") == "match",
        season_match=states.get("season") == "match", episode_match=states.get("episode") == "match",
        service_match=states.get("service_disc_family") == "match", edition_match=states.get("cut") == "match",
        fps_relation=determine_fps_relation(extract_metadata(video_meta).get("fps") if isinstance(video_meta, str) else video_meta.get("fps"),
            extract_metadata(sub_meta).get("fps") if isinstance(sub_meta, str) else sub_meta.get("fps") if isinstance(sub_meta, dict) else None))


def calculate_match_score(
    video_meta: dict[str, Any] | str,
    sub_meta: dict[str, Any] | str | SubtitleRelease,
    is_hash_match: bool = False,
) -> MatchResult:
    """
    Public entrypoint computing match score and compatibility percentage.
    Backwards-compatible with integer operations, tuple unpacking (score, percentage),
    and exposes structured diagnostics via .compatibility.
    """
    compat = calculate_compatibility(video_meta, sub_meta, is_hash_match=is_hash_match)
    if not compat.accepted:
        return MatchResult(-1000, 0, compatibility=compat)
    return MatchResult(compat.score, compat.percentage, compatibility=compat)


def _calculate_anime_match_score(
    video_meta: dict[str, Any],
    sub_meta: dict[str, Any],
    is_hash_match: bool = False,
) -> MatchResult:
    return calculate_match_score(video_meta, sub_meta, is_hash_match=is_hash_match)


def _calculate_standard_match_score(
    video_meta: dict[str, Any],
    sub_meta: dict[str, Any],
    is_hash_match: bool = False,
) -> MatchResult:
    return calculate_match_score(video_meta, sub_meta, is_hash_match=is_hash_match)


# =======================================================
# 7. DEDUPLICATION & MULTI-LANGUAGE RANKING
# =======================================================


def deduplicate_subtitles(subtitles):
    """Merge only duplicate provider records using shared evidence normalization."""
    from app.services.release_scoring_integration import display_order
    return display_order(None, subtitles)


def _is_hearing_impaired(sub: Any) -> bool:
    """Safely detect hearing-impaired/SDH subtitles from a release or dict."""
    if isinstance(sub, dict):
        if sub.get("hearing_impaired"):
            return True
        name = sub.get("release_name", "") or ""
    else:
        if getattr(sub, "hearing_impaired", False):
            return True
        name = getattr(sub, "release_name", "") or ""
    return bool(_SDH_REGEX.search(name))


def rank_subtitles(video_filename, subtitles, preferred_languages=None, discard_mismatches=True,
                   exclude_sdh=False, season=None, episode=None, title=None, year=None,
                   hi_preference="neutral", **video_facts):
    """Compatibility entrypoint: production DisplayPolicy is the sole ranker."""
    from app.services.release_scoring_integration import display_order
    return display_order(video_filename, subtitles, preferred_languages=preferred_languages,
        discard_mismatches=discard_mismatches, exclude_sdh=exclude_sdh, season=season,
        episode=episode, title=title, year=year, hi_preference=hi_preference, **video_facts)


def refresh_visible_compatibility(candidates, target_filename=None, **preferences):
    """Sync status is diagnostic only and cannot inflate release compatibility."""
    return rank_subtitles(target_filename, candidates, **preferences)


# =======================================================
# 7. CONTENT-AWARE DIALOGUE ALIGNMENT VALIDATION
# =======================================================
# Ground-truth content check: filename metadata can be mislabeled or carry
# unadjusted translator timings, but actual cue timings cannot lie. These
# helpers locate each file's first substantive dialogue cluster and compare
# them, so a mistimed reference is penalized no matter how pristine its
# release name looks.

# First-dialogue sanity thresholds (milliseconds): same retail master must open
# within 1.5s; across different sources/intros within 3.0s. These strict bounds
# are for RANKING (penalizing unknown-offset candidates). Sync EXECUTION uses
# the wider FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS below so alass can still fix
# realistic uniform intro/bumper shifts (3-15s) between Web and BluRay masters.
FIRST_DIALOGUE_SAME_FAMILY_THRESHOLD_MS = 1500
FIRST_DIALOGUE_CROSS_FAMILY_THRESHOLD_MS = 3000
FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS = 20000
# Intro cards/credits are only stripped within the first minute of a file.
INTRO_NONSPEECH_WINDOW_MS = 60_000
# Median global offset below which a subtitle counts as already aligned.
ALIGNED_OFFSET_THRESHOLD_S = 0.2
# Heavy penalty reserved for a proven timing mismatch.
TIMING_MISMATCH_PENALTY = -150

_DIALOGUE_MIN_LETTERS = 3
_DIALOGUE_MAX_GAP_MS = 20_000

_NON_SPEECH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)(?:https?://|www\.|@[\w.]+|\.(?:com|net|org|info|tv|cc|me|link|club|xyz)\b)"),
    re.compile(
        # Latin words are bounded: "credit" must not match "incredible".
        r"(?i)(?:ترجمة|مترجم|\btranslation\b|translated\s+by|\bsubtitles?\b|\bencoded\b|"
        r"sync(?:ed)?\s+by|\bcredits?\b|corrected\s+by|timed\s+by)"
    ),
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b"),
    re.compile(r"♪|\[music\]|\(music\)"),
)


def parse_srt_cues(text: str) -> list[tuple[int, int, str]]:
    """Parse ``(start_ms, end_ms, text)`` cues from an SRT string (tolerant)."""
    cues: list[tuple[int, int, str]] = []
    if not text or "-->" not in text:
        return cues
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    def _to_ms(value: str) -> int | None:
        value = value.strip().replace(",", ".")
        time_part, _, fraction = value.partition(".")
        parts = [p for p in time_part.split(":") if p.strip()]
        try:
            if len(parts) == 3:
                h, m, s = (int(p) for p in parts)
            elif len(parts) == 2:
                h, m, s = 0, int(parts[0]), int(parts[1])
            elif len(parts) == 1:
                h, m, s = 0, 0, int(parts[0])
            else:
                return None
            ms = int(fraction.ljust(3, "0")[:3]) if fraction else 0
        except ValueError:
            return None
        return ((h * 3600) + (m * 60) + s) * 1000 + ms

    for block in re.split(r"\n[ \t]*\n", normalized.strip("\n")):
        lines = [line for line in block.strip("\n").split("\n")]
        ts_idx = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if ts_idx is None:
            continue
        # The hours field is optional here, exactly as it is in the sync path's
        # own ``_TIMESPAN_LINE_REGEX`` and in alass's input handling. A real
        # provider SRT using ``MM:SS,mmm`` used to parse to ZERO cues here
        # while every other timing consumer in the system counted all of them,
        # so the verifier was handed an empty target for a subtitle the sync
        # service had just parsed successfully. ``_to_ms`` already understood
        # the two-part form; this regex was what made that unreachable.
        m = re.search(
            r"((?:\d{1,2}:)?\d{1,2}:\d{2}[,.]\d{1,3})\s*-->\s*((?:\d{1,2}:)?\d{1,2}:\d{2}[,.]\d{1,3})",
            lines[ts_idx],
        )
        if not m:
            continue
        start_ms = _to_ms(m.group(1))
        end_ms = _to_ms(m.group(2))
        if start_ms is None or end_ms is None:
            continue
        body = "\n".join(lines[ts_idx + 1 :]).strip()
        if body:
            cues.append((start_ms, end_ms, body))
    cues.sort(key=lambda c: c[0])
    return cues


def _strip_markup_for_speech(text: str) -> str:
    t = (text or "").replace("ـ", "")
    t = re.sub(r"\{[^{}]*\}", " ", t)
    t = re.sub(r"<[^<>]*>", " ", t)
    return t


def _is_non_speech_text(text: str) -> bool:
    """True for credits, URLs/handles, date cards, music-only or empty cues."""
    cleaned = _strip_markup_for_speech(text)
    if not cleaned.strip():
        return True
    if any(pattern.search(cleaned) for pattern in _NON_SPEECH_PATTERNS):
        return True
    # Remove bracketed asides/music; require substantive letters for dialogue.
    remainder = re.sub(r"\[[^\]]*\]|\([^()]*\)|♪", " ", cleaned)
    letters = sum(1 for ch in remainder if ch.isalpha())
    return letters < _DIALOGUE_MIN_LETTERS


def _is_dialogue_text(text: str) -> bool:
    return not _is_non_speech_text(text)


def strip_intro_nonspeech(
    cues: str | list[tuple[int, int, str]],
    window_ms: int = INTRO_NONSPEECH_WINDOW_MS,
) -> list[tuple[int, int, str]]:
    """Drop non-speech cues (credits, social links, date cards) inside the intro window."""
    parsed = parse_srt_cues(cues) if isinstance(cues, str) else list(cues)
    return [c for c in parsed if not (c[0] < window_ms and _is_non_speech_text(c[2]))]


def first_dialogue_cluster(
    cues: str | list[tuple[int, int, str]],
    *,
    min_consecutive: int = 2,
    max_gap_ms: int = _DIALOGUE_MAX_GAP_MS,
) -> int | None:
    """Start (ms) of the first run of ``min_consecutive`` dialogue cues, else ``None``."""
    parsed = parse_srt_cues(cues) if isinstance(cues, str) else sorted(cues, key=lambda c: c[0])
    run_start: int | None = None
    run_len = 0
    prev_end: int | None = None
    for start, end, text in parsed:
        if not _is_dialogue_text(text):
            run_start, run_len, prev_end = None, 0, None
            continue
        if run_start is None:
            run_start, run_len, prev_end = start, 1, end
        elif prev_end is not None and start - prev_end <= max_gap_ms:
            run_len += 1
            prev_end = max(prev_end, end)
        else:
            run_start, run_len, prev_end = start, 1, end
        if run_len >= min_consecutive:
            return run_start
    return None


def median_cue_offset(
    target: str | list[tuple[int, int, str]],
    reference: str | list[tuple[int, int, str]],
) -> float | None:
    """Median ``(reference - target)`` cue-start offset in seconds, else ``None``."""
    t = parse_srt_cues(target) if isinstance(target, str) else sorted(target, key=lambda c: c[0])
    r = parse_srt_cues(reference) if isinstance(reference, str) else sorted(reference, key=lambda c: c[0])
    if not t or not r:
        return None
    r_starts = [s for s, _, _ in r]
    diffs: list[float] = []
    for s, _, _ in t:
        i = bisect.bisect_left(r_starts, s)
        cands = [r_starts[j] for j in (i - 1, i, i + 1) if 0 <= j < len(r_starts)]
        nearest = min(cands, key=lambda x: abs(x - s))
        diffs.append((nearest - s) / 1000.0)
    diffs.sort()
    mid = len(diffs) // 2
    if len(diffs) % 2:
        return diffs[mid]
    return (diffs[mid - 1] + diffs[mid]) / 2.0


# --------------------------------------------------------------------------- #
# "Is this subtitle already aligned?"                                           #
# --------------------------------------------------------------------------- #
# A nearest-neighbour median cannot answer this. In a densely cued subtitle every
# target cue has *some* reference cue within a second or two, so the median
# collapses towards zero no matter how far apart the two timelines really are.
# Production proved it: three real releases measured +0.06s / +0.14s / +0.06s and
# were served as "already aligned", while their residual p95 against the same
# reference was 4401ms / 4588ms / 4401ms -- over twice the tolerance.
#
# So the claim is only made when three independent things hold: the cues
# actually correspond, the disagreement is small, and it is small in more than
# one place. Any of them failing hands the subtitle to alass instead.

#: Sections the timeline is split into. One median over a whole film can hide a
#: subtitle that starts aligned and diverges later, so agreement has to be shown
#: in several independent places.
ALIGNED_SECTION_COUNT = 5
#: A section only counts as evidence when at least this share of its cues found a
#: counterpart inside the residual tolerance.
ALIGNED_SECTION_MIN_COVERAGE = 0.5
#: How many sections must corroborate before an alignment claim is allowed at all.
#: Unanimity among those sections is then required; see ``measure_alignment_consistency``.
ALIGNED_MIN_SECTIONS = 3
#: Absolute floor on matched dialogue cues for making any alignment claim. Kept
#: low on purpose: the protection against the production bug was section
#: unanimity plus residual spread, not raw cue count, and a high floor here
#: would cost short-but-genuinely-aligned subtitles their alass shortcut. It
#: only needs to reject a one- or two-cue coincidence.
ALIGNED_MIN_MATCHED_CUES = 5


@dataclass
class AlignmentConsistency:
    """Whether one near-zero offset explains a target against its reference."""

    median_offset_s: float | None = None
    p95_ms: float | None = None
    coverage: float = 0.0
    sections_measured: int = 0
    sections_agreeing: int = 0
    #: ``(position_ms, target_text, reference_text)`` for the first matches, so a
    #: log line can show what the comparison actually looked at.
    samples: list[tuple[int, str, str]] = field(default_factory=list)
    reason: str = ""

    @property
    def aligned(self) -> bool:
        return self.reason == ""


def measure_alignment_consistency(
    target: str | list[tuple[int, int, str]],
    reference: str | list[tuple[int, int, str]],
    *,
    is_dialogue: Callable[[str], bool] | None = None,
) -> AlignmentConsistency | None:
    """Measure whether a subtitle genuinely needs no re-timing.

    Pairs cues with the same one-to-one, tolerance-bounded matcher the verifier
    uses, drops non-speech cues first, then splits the timeline into
    :data:`ALIGNED_SECTION_COUNT` sections and measures each one separately. The
    result is only ``aligned`` when the overall median is inside
    :data:`ALIGNED_OFFSET_THRESHOLD_S`, the residual spread is inside the
    verifier's own :data:`~app.services.sync.alignment.MAX_P95_MS_FOR_STABLE`
    bound, and at least :data:`ALIGNED_MIN_SECTIONS` sections independently agree.

    Known boundary: the evidence here is purely temporal. A segment that has
    been displaced by roughly a whole number of cue intervals can still find
    distinct reference cues at a near-zero offset, and timing alone cannot tell
    that apart from a genuine match. Catching that case needs content
    comparison, which belongs to the verifier -- the gate's job is to stop the
    far more common failure, where a real offset is reported as zero because
    every cue had *some* neighbour nearby.
    """
    from app.services.sync.alignment import (
        MAX_P95_MS_FOR_STABLE,
        RESIDUAL_TOLERANCE_MS,
        pair_cues,
    )

    t = parse_srt_cues(target) if isinstance(target, str) else sorted(target, key=lambda c: c[0])
    r = parse_srt_cues(reference) if isinstance(reference, str) else sorted(reference, key=lambda c: c[0])
    if is_dialogue is not None:
        # Credits, intros and site tags are not speech and must not be allowed
        # to stand in for timing evidence.
        t = [c for c in t if is_dialogue(c[2])]
        r = [c for c in r if is_dialogue(c[2])]
    if len(t) < 2 or len(r) < 2:
        return None

    pairing = pair_cues(t, r, tolerance_ms=RESIDUAL_TOLERANCE_MS)
    pairs = pairing.pairs
    if not pairs:
        return AlignmentConsistency(reason="no cue correspondence within tolerance")
    if len(pairs) < ALIGNED_MIN_MATCHED_CUES:
        # An absolute floor, deliberately not scaled down for short inputs. A
        # handful of cues is trivial to match -- any offset lands within
        # tolerance of something -- so agreeing on a few of them says nothing
        # about a film-length subtitle. Refusing to claim alignment here is the
        # safe direction: the subtitle simply goes to alass.
        return None

    deltas = [delta for _, delta in pairs]
    magnitude = [abs(d) for d in deltas]
    ordered = sorted(magnitude)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    values = sorted(deltas)
    mid = len(values) // 2
    median = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2.0

    lo = min(c[0] for c in t)
    hi = max(c[1] for c in t)
    width = max(1, hi - lo) / ALIGNED_SECTION_COUNT

    # Section membership comes from each match's real target-cue index. Pair
    # ordinals would put a late cue's delta in an early section and hide a tail
    # that drifted away entirely.
    matched_by_section: list[list[float]] = [[] for _ in range(ALIGNED_SECTION_COUNT)]
    cues_by_section: list[int] = [0] * ALIGNED_SECTION_COUNT
    for cue in t:
        position = cue[0]
        slot = min(ALIGNED_SECTION_COUNT - 1, max(0, int((position - lo) / width)))
        cues_by_section[slot] += 1
    for match in pairing.matches:
        position = t[match.before_index][0]
        slot = min(ALIGNED_SECTION_COUNT - 1, max(0, int((position - lo) / width)))
        matched_by_section[slot].append(match.delta_ms)

    # Every section that actually contains dialogue is a claim about part of the
    # film, so each one is either corroborated or it counts against the verdict.
    # A section whose cues find no counterpart is NOT skipped: skipping it let a
    # subtitle that lines up for the first half and drifts away for the second
    # pass on the strength of the half that happened to match.
    measured = sum(1 for count in cues_by_section if count)
    agreeing = 0
    for index in range(ALIGNED_SECTION_COUNT):
        section_cues = cues_by_section[index]
        if not section_cues:
            continue
        if len(matched_by_section[index]) / section_cues < ALIGNED_SECTION_MIN_COVERAGE:
            continue
        section_values = sorted(matched_by_section[index])
        m = len(section_values) // 2
        section_median = (
            section_values[m]
            if len(section_values) % 2
            else (section_values[m - 1] + section_values[m]) / 2.0
        )
        if abs(section_median) < ALIGNED_OFFSET_THRESHOLD_S * 1000.0:
            agreeing += 1

    result = AlignmentConsistency(
        median_offset_s=median / 1000.0,
        p95_ms=p95,
        coverage=len(pairs) / len(t),
        sections_measured=measured,
        sections_agreeing=agreeing,
    )
    # Show what the comparison actually looked at, so a suspicious "already
    # aligned" line can be checked by eye instead of taken on trust.
    for match in sorted(pairing.matches, key=lambda mch: mch.position_ms)[:2]:
        result.samples.append(
            (match.position_ms, t[match.before_index][2][:40], r[match.after_index][2][:40])
        )

    if abs(median) >= ALIGNED_OFFSET_THRESHOLD_S * 1000.0:
        result.reason = f"median offset {median / 1000.0:+.2f}s is not negligible"
    elif p95 > MAX_P95_MS_FOR_STABLE:
        result.reason = (
            f"residual p95 {p95:.0f}ms exceeds the {MAX_P95_MS_FOR_STABLE:.0f}ms "
            "tolerance; the timelines disagree well past the median"
        )
    elif measured < ALIGNED_MIN_SECTIONS:
        result.reason = (
            f"only {measured} section(s) of the timeline carry enough dialogue to "
            "corroborate anything"
        )
    elif agreeing < measured:
        # Unanimity, not a majority. "Already aligned" asserts the subtitle needs
        # no re-timing across its whole duration, so one section that cannot be
        # corroborated -- because its cues drifted, jumped, or have no counterpart
        # within tolerance -- is already a reason to hand the subtitle to alass.
        disagreeing = measured - agreeing
        result.reason = (
            f"{disagreeing} of {measured} dialogue sections disagree or cannot be "
            "corroborated; alignment is not consistent across the whole timeline"
        )
    return result


def first_dialogue_threshold_ms(same_family: bool | None) -> int:
    """Sanity threshold: 1.5s on one master, 3.0s across different sources."""
    if same_family:
        return FIRST_DIALOGUE_SAME_FAMILY_THRESHOLD_MS
    return FIRST_DIALOGUE_CROSS_FAMILY_THRESHOLD_MS


def validate_cue_sanity(
    candidate: str | list[tuple[int, int, str]],
    reference: str | list[tuple[int, int, str]],
    *,
    same_family: bool | None = None,
    threshold_ms: int | None = None,
) -> dict[str, Any]:
    """Content ground-truth check on first substantive dialogue clusters.

    Intro non-dialogue (credits/links/date cards in the first 60s) is stripped
    from both sides, then the first run of >=2 dialogue cues in each file is
    compared. Returns ``ok``/``delta_ms``/``penalty``/``reason``. Missing
    dialogue fails OPEN (``ok=True``) so alass still gets a chance; a proven
    mismatch carries ``TIMING_MISMATCH_PENALTY``.
    """
    limit = threshold_ms if threshold_ms is not None else first_dialogue_threshold_ms(same_family)
    cand_first = first_dialogue_cluster(strip_intro_nonspeech(candidate))
    ref_first = first_dialogue_cluster(strip_intro_nonspeech(reference))
    if cand_first is None or ref_first is None:
        return {
            "ok": True,
            "delta_ms": None,
            "candidate_first_ms": cand_first,
            "reference_first_ms": ref_first,
            "threshold_ms": limit,
            "penalty": 0,
            "reason": "insufficient dialogue to validate",
        }
    delta = cand_first - ref_first
    if abs(delta) > limit:
        return {
            "ok": False,
            "delta_ms": delta,
            "candidate_first_ms": cand_first,
            "reference_first_ms": ref_first,
            "threshold_ms": limit,
            "penalty": TIMING_MISMATCH_PENALTY,
            "reason": (
                f"TIMING_MISMATCH: candidate first dialogue at {cand_first / 1000.0:.2f}s vs "
                f"reference at {ref_first / 1000.0:.2f}s (delta {delta / 1000.0:+.2f}s > {limit / 1000.0:.1f}s)"
            ),
        }
    return {
        "ok": True,
        "delta_ms": delta,
        "candidate_first_ms": cand_first,
        "reference_first_ms": ref_first,
        "threshold_ms": limit,
        "penalty": 0,
        "reason": None,
    }


# Backwards-compatible aliases
clean_subtitle_filename = clean_subtitle_filename
sanitize_release_name = sanitize_release_name
parse_release_metadata = extract_metadata
