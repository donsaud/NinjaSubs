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
from dataclasses import dataclass, field
from typing import Any

from app.models import MatchTier, SubtitleRelease
from app.services.ranking import (
    _INVALID_GROUPS,
    KNOWN_GROUPS,
    clean_subtitle_filename,
    get_source_popularity_percentage,
    levenshtein_ratio,
    sanitize_release_name,
)
from app.utils.language import normalize_to_iso639_2

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
# 2. CONFIGURABLE RANKING WEIGHTS & CONSTANTS
# =======================================================

WEIGHT_EXACT_HASH = 500
WEIGHT_TITLE_MAX = 25
WEIGHT_YEAR_MATCH = 10
WEIGHT_YEAR_MISMATCH = -60
WEIGHT_SEASON_MATCH = 15
WEIGHT_EPISODE_MATCH_TV = 20
WEIGHT_EPISODE_MATCH_ANIME = 115
WEIGHT_EDITION_MATCH = 40
WEIGHT_EDITION_UNMARKED_PENALTY = -25
WEIGHT_GROUP_MATCH_TV = 45
WEIGHT_GROUP_MISMATCH_TV = -30
WEIGHT_GROUP_MATCH_ANIME = 45
WEIGHT_GROUP_MISMATCH_ANIME = -30
WEIGHT_GROUP_UNSHARED_FANSUB = -25
WEIGHT_GROUP_CLEAN_ANIME = 25
WEIGHT_FPS_EXACT = 15
WEIGHT_FPS_NEAR = 5
WEIGHT_FPS_DRIFT = -150
WEIGHT_SOURCE_MATCH = 40
WEIGHT_SOURCE_FAMILY_MATCH = 25
WEIGHT_SOURCE_CROSS_PENALTY = -15
WEIGHT_SOURCE_CAM_PENALTY = -60
WEIGHT_REMUX_PREFERENCE = 20
WEIGHT_SERVICE_MATCH = 50
WEIGHT_SERVICE_MISMATCH = -40
WEIGHT_REPACK_MATCH = 20
WEIGHT_REPACK_MISMATCH = -20
WEIGHT_RESOLUTION_MATCH = 15
WEIGHT_RESOLUTION_CLOSE = 5
WEIGHT_CODEC_MATCH = 10
WEIGHT_AUDIO_MATCH = 10
# Prefer the exact single episode over a multi-episode/range release (e.g.
# S01E05-E06) for a single-episode target: identical content, weaker match.
WEIGHT_MULTI_EP_PENALTY = 20
# Tie-breaker: a re-uploaded/malformed filename (website URL, tracker domain)
# must rank below a clean scene dot-notation release of otherwise equal metadata.
WEIGHT_REUPLOAD_PENALTY = 10

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
    r"(?i)\b(?:se|season)[-._ ]?(\d{1,2})[-._ ]*(?:ep|episode|e)[-._ ]?(\d{1,3})\b"
)
_S_E_REGEX = re.compile(r"(?i)\bS(\d{1,2})[-._ ]?E(\d{1,2})\b")
_X_E_REGEX = re.compile(r"(?i)\b(\d{1,2})x(\d{1,2})\b")
_SEASON_EP_REGEX = re.compile(r"(?i)\bSeason[-._ ]?(\d{1,2})[-._ ]?Episode[-._ ]?(\d{1,2})\b")
_MULTI_EP_REGEX = re.compile(
    r"(?i)\bS(\d{1,2})[-._ ]?E(\d{1,2})(?:[-–—]\d{1,2}|(?:[-._ ]?E\d{1,2})+)\b"
)
_ORDINAL_SEASON_REGEX = re.compile(r"(?i)\b(\d{1,2})(?:st|nd|rd|th)[-._ ]?season\b")
_SEASON_ONLY_REGEX = re.compile(r"(?i)\b(?:se|season|s)[-._ ]?(\d{1,2})\b")
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
        for kg in _CLEANED_KNOWN_GROUPS:
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

    for m in re.finditer(r"(?<!\d)(\d{3,4})(?!\d)", combined_text):
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


def determine_match_tier(
    is_hash_match: bool,
    accepted: bool,
    release_group_match: bool | None,
    has_source_match: bool,
    service_match: bool | None,
    episode_match: bool | None,
    season_match: bool | None,
    edition_match: bool | None = None,
    edition_unverified: bool = False,
    has_year_mismatch: bool = False,
    is_unshared_fansub: bool = False,
    title_ratio: float = 1.0,
) -> MatchTier:
    """Classify a candidate into the deterministic MatchTier hierarchy.

    Classification is purely token-based; the additive soft score is deliberately
    *not* consulted. A strict hierarchy is what guarantees an exact release-group
    match always sorts above a generic candidate, no matter how many incidental
    properties (resolution, codec, audio, platform) the latter happens to match.

    - Tier 0 (HASH): exact binary hash match
    - Tier 1 (EXACT): exact release group token match
    - Tier 2 (SOURCE_FAMILY): source/edition match, or an exact group match whose
      cut could not be verified
    - Tier 3 (CLOSE): season/episode match
    - Tier 4 (FALLBACK): everything else, including any candidate whose title
      does not match the video
    """
    if is_hash_match:
        return MatchTier.HASH
    if not accepted:
        return MatchTier.FALLBACK

    # Title identity is the precondition for every stronger tier. Without it the
    # candidate is a different work, and a coincidental source/platform/group
    # match is meaningless.
    if _is_different_work(title_ratio, episode_match, season_match):
        return MatchTier.FALLBACK

    if is_unshared_fansub:
        base = (
            MatchTier.CLOSE
            if episode_match is True or season_match is True
            else MatchTier.FALLBACK
        )
    elif release_group_match is True:
        # An exact group match is the strongest identity signal available. The
        # one exception is an unverifiable cut: if the video declares an edition
        # and the subtitle is unmarked, the group match is real but the cut is
        # unknown, so it is capped at the source/edition tier. (An *explicit*
        # edition conflict is already hard-rejected during candidate filtering.)
        base = MatchTier.SOURCE_FAMILY if edition_unverified else MatchTier.EXACT
    elif has_source_match or service_match is True or edition_match is True:
        base = MatchTier.SOURCE_FAMILY
    elif episode_match is True or season_match is True:
        base = MatchTier.CLOSE
    else:
        base = MatchTier.FALLBACK

    # A year mismatch means we probably matched a different film entirely, so it
    # invalidates every identity claim above (group, source, edition) rather
    # than merely weakening them. It is a demotion, never a promotion: the
    # candidate drops to the generic fallback tier and is decided by the
    # remaining signals. A hash match is exempt because it is byte-exact truth.
    if has_year_mismatch:
        return MatchTier.FALLBACK
    return base


def calculate_compatibility(
    video_meta: dict[str, Any] | str,
    sub_meta: dict[str, Any] | str | SubtitleRelease,
    is_hash_match: bool = False,
) -> CompatibilityResult:
    """
    Two-stage subtitle compatibility evaluation:
    1. Tier 0 (Deterministic Hash Match): Short-circuits immediately.
    2. Tier 1 to 4: Stage 1 Hard Exclusion Filter -> Stage 2 Soft Scoring.
    """
    if hasattr(sub_meta, "release_name") and hasattr(sub_meta, "is_hash_match"):
        is_hash_match = is_hash_match or bool(
            getattr(sub_meta, "is_hash_match", False)
            or getattr(sub_meta, "matched_by_hash", False)
        )
        s_meta = extract_metadata(sub_meta.release_name)
    elif isinstance(sub_meta, dict):
        is_hash_match = is_hash_match or bool(
            sub_meta.get("is_hash_match", False) or sub_meta.get("matched_by_hash", False)
        )
        if "title" not in sub_meta and "release_name" in sub_meta:
            s_meta = extract_metadata(sub_meta["release_name"])
        elif "title" not in sub_meta and isinstance(sub_meta.get("name"), str):
            s_meta = extract_metadata(sub_meta["name"])
        else:
            s_meta = sub_meta
    elif isinstance(sub_meta, str):
        s_meta = extract_metadata(sub_meta)
    else:
        s_meta = extract_metadata(str(sub_meta))

    is_sdh = bool(s_meta.get("is_sdh", False)) if isinstance(s_meta, dict) else False

    # -------------------------------------------------------------------
    # TIER 0: DETERMINISTIC BINARY HASH MATCH SHORT-CIRCUIT
    # -------------------------------------------------------------------
    if is_hash_match:
        logger.debug(
            "[Hash Match SHORT-CIRCUIT] Sub: '%s'",
            s_meta.get("raw_filename") if isinstance(s_meta, dict) else str(s_meta),
        )
        return CompatibilityResult(
            accepted=True,
            score=WEIGHT_EXACT_HASH,
            percentage=100,
            match_tier=MatchTier.HASH,
            confidence="deterministic",
            match_method="hash",
            title_match=1.0,
            season_match=True,
            episode_match=True,
            is_hash_match=True,
            is_sdh=is_sdh,
            reasons=["Exact binary MovieHash match (100% sync guaranteed)"],
        )

    if isinstance(video_meta, str):
        v_meta = extract_metadata(video_meta)
    else:
        v_meta = video_meta

    accepted, reject_reason, match_method = hard_compatibility_filter(
        v_meta, s_meta, is_hash_match=False
    )

    reasons: list[str] = []

    # Handle Hard Rejection
    if not accepted:
        logger.debug(
            "[Hard Filter REJECT] Target: '%s' | Sub: '%s' | Reason: %s",
            v_meta.get("raw_filename"),
            s_meta.get("raw_filename"),
            reject_reason,
        )
        return CompatibilityResult(
            accepted=False,
            score=-1000,
            confidence=0.0,
            match_method=match_method,
            match_tier=MatchTier.FALLBACK,
            hard_reject_reason=reject_reason,
            is_hash_match=False,
            is_sdh=is_sdh,
            percentage=0,
            reasons=[f"Hard rejection: {reject_reason}"],
        )

    score = 0
    is_anime = is_anime_content(v_meta, s_meta)

    # 1. Title Similarity Matching
    v_title = v_meta.get("title") or ""
    s_title = s_meta.get("title") or ""
    v_clean_title = re.sub(r"^(?:\[[^\]]*\]|[^\]]*\])\s*", "", v_title).strip()
    s_clean_title = re.sub(r"^(?:\[[^\]]*\]|[^\]]*\])\s*", "", s_title).strip()
    title_ratio = 1.0

    if is_anime and (not s_clean_title or not v_clean_title):
        title_ratio = 1.0
        score += WEIGHT_TITLE_MAX
        reasons.append(f"Anime episode title alignment (+{WEIGHT_TITLE_MAX})")
    elif v_clean_title and s_clean_title:
        v_low = v_clean_title.lower()
        s_low = s_clean_title.lower()
        if s_low.startswith(v_low) or v_low.startswith(s_low) or (is_anime and (s_low in v_low or v_low in s_low)):
            title_ratio = 1.0
            score += WEIGHT_TITLE_MAX
            reasons.append(f"Title prefix match (+{WEIGHT_TITLE_MAX})")
        else:
            title_ratio = levenshtein_ratio(v_clean_title, s_clean_title)
            pts = int(round(title_ratio * WEIGHT_TITLE_MAX))
            score += pts
            reasons.append(f"Title similarity {title_ratio:.2f} (+{pts})")
    elif v_title and s_title:
        title_ratio = levenshtein_ratio(v_title, s_title)
        pts = int(round(title_ratio * WEIGHT_TITLE_MAX))
        score += pts
        reasons.append(f"Title similarity {title_ratio:.2f} (+{pts})")
    else:
        score += 20
        reasons.append("Default title base score (+20)")

    # 2. Release Year Matching
    v_year = v_meta.get("year")
    s_year = s_meta.get("year")
    if v_year and s_year:
        if v_year == s_year:
            score += WEIGHT_YEAR_MATCH
            reasons.append(f"Release year match {v_year} (+{WEIGHT_YEAR_MATCH})")
        else:
            score += WEIGHT_YEAR_MISMATCH
            reasons.append(f"Release year mismatch ({v_year} vs {s_year}) ({WEIGHT_YEAR_MISMATCH})")

    # 3. Season & Episode Matching
    season_match: bool | None = None
    episode_match: bool | None = None

    if is_anime:
        v_target_ep = (
            v_meta.get("absolute_episode")
            if v_meta.get("absolute_episode") is not None
            else v_meta.get("episode")
        )
        s_target_ep = (
            s_meta.get("absolute_episode")
            if s_meta.get("absolute_episode") is not None
            else s_meta.get("episode")
        )
        s_eps = s_meta.get("episodes") or set()

        is_anime_ep_match = False
        if (
            v_target_ep is not None
            and s_target_ep is not None
            and (v_target_ep == s_target_ep or v_target_ep in s_eps)
        ):
            is_anime_ep_match = True
            episode_match = True
            score += WEIGHT_EPISODE_MATCH_ANIME
            reasons.append(f"Anime episode {v_target_ep} match (+{WEIGHT_EPISODE_MATCH_ANIME})")

        # Anime Season Alignment Check
        v_raw_fn = v_meta.get("raw_filename", "")
        s_raw_fn = s_meta.get("raw_filename", "")
        v_is_single_abs = (v_meta.get("absolute_episode") is not None) and not has_season_mention(
            v_raw_fn
        )
        s_is_single_abs = (s_meta.get("absolute_episode") is not None) and not has_season_mention(
            s_raw_fn
        )
        v_eff_season = (
            v_meta.get("season")
            if v_meta.get("season") is not None
            else (1 if v_is_single_abs else None)
        )
        s_eff_season = (
            s_meta.get("season")
            if s_meta.get("season") is not None
            else (1 if s_is_single_abs else None)
        )
        if v_eff_season is not None and s_eff_season is not None and v_eff_season == s_eff_season:
            season_match = True
        elif v_eff_season == 1 and s_eff_season is None and not has_season_mention(s_raw_fn):
            season_match = True

        # Leading Bracket / Group Matching for Anime
        v_lead = extract_leading_bracket_group(v_meta.get("raw_filename", "")) or v_meta.get(
            "group"
        )
        s_lead = extract_leading_bracket_group(s_meta.get("raw_filename", "")) or s_meta.get(
            "group"
        )
        v_norm = v_lead.strip().lower().replace(".", "") if v_lead else None
        s_norm = s_lead.strip().lower().replace(".", "") if s_lead else None

        is_unshared_fansub = False
        release_group_match = None
        if v_norm and s_norm:
            if v_norm == s_norm:
                release_group_match = True
                score += WEIGHT_GROUP_MATCH_ANIME
                reasons.append(f"Fansub group match [{v_lead}] (+{WEIGHT_GROUP_MATCH_ANIME})")
            elif s_norm in _TRANSLATION_PLATFORMS or s_norm in _SUBTITLE_PROVIDERS:
                score += WEIGHT_GROUP_CLEAN_ANIME
                reasons.append(f"Translation provider subtitle [{s_lead}] (+{WEIGHT_GROUP_CLEAN_ANIME})")
            else:
                release_group_match = False
                is_unshared_fansub = True
                score += WEIGHT_GROUP_MISMATCH_ANIME
                reasons.append(
                    f"Fansub group conflict [{v_lead}] vs [{s_lead}] ({WEIGHT_GROUP_MISMATCH_ANIME})"
                )
        elif s_norm and not v_norm:
            if s_norm in _TRANSLATION_PLATFORMS or s_norm in _SUBTITLE_PROVIDERS:
                score += WEIGHT_GROUP_CLEAN_ANIME
                reasons.append(f"Translation provider subtitle [{s_lead}] (+{WEIGHT_GROUP_CLEAN_ANIME})")
            else:
                is_unshared_fansub = True
                score += WEIGHT_GROUP_UNSHARED_FANSUB
                reasons.append(
                    f"Subtitle has fansub [{s_lead}], video generic ({WEIGHT_GROUP_UNSHARED_FANSUB})"
                )
        elif not s_norm:
            score += WEIGHT_GROUP_CLEAN_ANIME
            reasons.append(f"Clean/generic anime episode subtitle (+{WEIGHT_GROUP_CLEAN_ANIME})")

        # Single episode priority over batch / complete packs
        v_is_batch = bool(v_meta.get("is_complete") or (len(v_meta.get("episodes") or set()) > 1))
        s_is_batch = bool(s_meta.get("is_complete") or (len(s_eps) > 1))
        if not v_is_batch and s_is_batch:
            score -= 20
            reasons.append("Multi-episode/batch pack preference adjustment (-20)")
        elif not v_is_batch and not s_is_batch:
            score += 10
            reasons.append("Direct single-episode alignment (+10)")
    else:
        # Standard TV Season & Episode
        v_season = v_meta.get("season")
        v_ep = v_meta.get("episode")
        s_season = s_meta.get("season")
        s_ep = s_meta.get("episode")
        s_eps = s_meta.get("episodes") or set()

        if v_ep is not None and (v_ep == s_ep or v_ep in s_eps):
            episode_match = True
            if v_season is not None and s_season is not None and v_season == s_season:
                season_match = True
                score += WEIGHT_EPISODE_MATCH_TV
                reasons.append(
                    f"TV Season {v_season} & Episode {v_ep} match (+{WEIGHT_EPISODE_MATCH_TV})"
                )
            elif v_season == 1 and s_season is None:
                season_match = True
                score += WEIGHT_EPISODE_MATCH_TV
                reasons.append(
                    f"TV Episode {v_ep} match (implicit S1) (+{WEIGHT_EPISODE_MATCH_TV})"
                )
            else:
                score += 10
                reasons.append(f"TV Episode {v_ep} match (+10)")
        elif v_season is not None and s_season is not None and v_season == s_season:
            season_match = True
            score += 5
            reasons.append(f"TV Season {v_season} match (+5)")

        # Prefer the exact single episode over a multi-episode/range release
        # (S01E05-E06) for a single-episode target: same content, weaker match.
        if v_ep is not None and len(s_eps) > 1 and v_ep in s_eps:
            score -= WEIGHT_MULTI_EP_PENALTY
            reasons.append(
                f"Multi-episode release {sorted(s_eps)} (-{WEIGHT_MULTI_EP_PENALTY})"
            )

        # Release Group Matching for Standard TV & Movies
        v_grp = v_meta.get("group")
        s_grp = s_meta.get("group")
        release_group_match = None
        if v_grp and s_grp:
            v_norm = _normalize_group(v_grp)
            s_norm = _normalize_group(s_grp)
            if v_norm == s_norm or {v_norm, s_norm} <= {"yts", "ytsmx", "yify"}:
                release_group_match = True
                score += WEIGHT_GROUP_MATCH_TV
                reasons.append(f"Release group match [{v_grp}] (+{WEIGHT_GROUP_MATCH_TV})")
            elif v_grp.upper() in KNOWN_GROUPS and s_grp.upper() in KNOWN_GROUPS:
                release_group_match = False
                score += WEIGHT_GROUP_MISMATCH_TV
                reasons.append(
                    f"Different known scene groups [{v_grp}] vs [{s_grp}] ({WEIGHT_GROUP_MISMATCH_TV})"
                )

        is_unshared_fansub = False
        is_anime_ep_match = False

    # 4. Streaming Service Alignment
    v_svc = (v_meta.get("service") or "").upper()
    s_svc = (s_meta.get("service") or "").upper()
    service_match: bool | None = None
    if v_svc and s_svc:
        if v_svc == s_svc:
            service_match = True
            score += WEIGHT_SERVICE_MATCH
            reasons.append(f"Streaming platform match {v_svc} (+{WEIGHT_SERVICE_MATCH})")
        else:
            service_match = False
            score += WEIGHT_SERVICE_MISMATCH
            reasons.append(
                f"Different streaming platforms ({v_svc} vs {s_svc}) ({WEIGHT_SERVICE_MISMATCH})"
            )

    # 5. Media Source Alignment & Source Families
    v_src = (v_meta.get("source") or "").lower()
    s_src = (s_meta.get("source") or "").lower()
    v_family = v_meta.get("source_family") or SOURCE_FAMILIES.get(v_meta.get("source") or "")
    s_family = s_meta.get("source_family") or SOURCE_FAMILIES.get(s_meta.get("source") or "")
    source_match: bool | None = None
    has_source_match = False

    if v_src and s_src:
        if v_src == s_src:
            source_match = True
            has_source_match = True
            score += WEIGHT_SOURCE_MATCH
            reasons.append(f"Media source match {v_src.upper()} (+{WEIGHT_SOURCE_MATCH})")
        elif v_family and s_family and v_family == s_family:
            source_match = True
            has_source_match = True
            score += WEIGHT_SOURCE_FAMILY_MATCH
            if v_family in ("DISC_HI", "DISC_LO"):
                reasons.append(
                    f"Disc source family ({v_src} ~ {s_src}) (+{WEIGHT_SOURCE_FAMILY_MATCH})"
                )
            elif v_family == "WEB":
                reasons.append(
                    f"Web source family ({v_src} ~ {s_src}) (+{WEIGHT_SOURCE_FAMILY_MATCH})"
                )
            else:
                reasons.append(
                    f"Source family alignment ({v_src} ~ {s_src}) (+{WEIGHT_SOURCE_FAMILY_MATCH})"
                )
        elif (v_family in ("DISC_HI", "DISC_LO") and s_family == "WEB") or (
            v_family == "WEB" and s_family in ("DISC_HI", "DISC_LO")
        ):
            score += WEIGHT_SOURCE_CROSS_PENALTY
            reasons.append(f"Cross-source (disc vs web) ({WEIGHT_SOURCE_CROSS_PENALTY})")
        elif v_src in ("remux", "bluray", "uhd", "web-dl", "webrip", "web") and s_src == "hdtv":
            score -= 25
            reasons.append("HDTV subtitle on high-quality source (-25)")
        elif s_src == "cam" and v_src != "cam":
            score += WEIGHT_SOURCE_CAM_PENALTY
            reasons.append(f"CAM subtitle on non-CAM video ({WEIGHT_SOURCE_CAM_PENALTY})")
        elif v_src == "cam" and s_src != "cam":
            score += WEIGHT_SOURCE_CAM_PENALTY
            reasons.append(f"Non-CAM subtitle on CAM video ({WEIGHT_SOURCE_CAM_PENALTY})")

    # Remux Preference
    v_is_remux = v_meta.get("is_remux", False)
    s_is_remux = s_meta.get("is_remux", False)
    if v_is_remux:
        if s_is_remux or s_src in ("bluray", "remux"):
            score += WEIGHT_REMUX_PREFERENCE
            reasons.append(f"Remux/BluRay alignment (+{WEIGHT_REMUX_PREFERENCE})")
        elif s_src in ("webrip", "hdtv", "cam"):
            score -= 30
            reasons.append("WebRip/HDTV subtitle on Remux video (-30)")

    # Repack Alignment
    v_is_repack = v_meta.get("is_repack", False)
    s_is_repack = s_meta.get("is_repack", False)
    if v_is_repack and s_is_repack:
        score += WEIGHT_REPACK_MATCH
        reasons.append(f"Repack/Proper alignment (+{WEIGHT_REPACK_MATCH})")
    elif v_is_repack and not s_is_repack:
        score += WEIGHT_REPACK_MISMATCH
        reasons.append(f"Repack video with regular subtitle ({WEIGHT_REPACK_MISMATCH})")

    # 6. Edition / Cut Alignment
    v_edition = v_meta.get("edition")
    s_edition = s_meta.get("edition")
    edition_match: bool | None = None
    if v_edition and s_edition:
        if v_edition == s_edition:
            edition_match = True
            score += WEIGHT_EDITION_MATCH
            reasons.append(f"Edition match {v_edition} (+{WEIGHT_EDITION_MATCH})")
    elif v_edition and not s_edition:
        score += WEIGHT_EDITION_UNMARKED_PENALTY
        reasons.append(
            f"Video has edition {v_edition}, subtitle unmarked ({WEIGHT_EDITION_UNMARKED_PENALTY})"
        )
    elif not v_edition and s_edition and s_edition in ("EXTENDED", "DIRECTORS_CUT", "UNRATED"):
        score -= 200
        reasons.append("Theatrical video with extended cut subtitle (-200)")

    # 7. FPS Relation
    v_fps = v_meta.get("fps")
    s_fps = s_meta.get("fps")
    fps_relation = determine_fps_relation(v_fps, s_fps)
    if fps_relation == "exact":
        score += WEIGHT_FPS_EXACT
        reasons.append(f"Exact FPS match {v_fps}fps (+{WEIGHT_FPS_EXACT})")
    elif fps_relation == "near":
        score += WEIGHT_FPS_NEAR
        reasons.append(f"Near FPS alignment ({v_fps}fps vs {s_fps}fps) (+{WEIGHT_FPS_NEAR})")
    elif fps_relation == "drift":
        score += WEIGHT_FPS_DRIFT
        reasons.append(f"FPS drift conflict ({v_fps}fps vs {s_fps}fps) ({WEIGHT_FPS_DRIFT})")

    # 8. Resolution Alignment (Weak Signal)
    v_res = v_meta.get("resolution")
    s_res = s_meta.get("resolution")
    if v_res and s_res:
        if v_res == s_res:
            score += WEIGHT_RESOLUTION_MATCH
            reasons.append(f"Resolution match {v_res} (+{WEIGHT_RESOLUTION_MATCH})")
        elif {v_res, s_res} <= {"2160p", "1080p"}:
            score += WEIGHT_RESOLUTION_CLOSE
            reasons.append(f"Close resolution (4K ~ 1080p) (+{WEIGHT_RESOLUTION_CLOSE})")

    # 9. Codec Alignment (Weak Signal)
    v_cdc = (v_meta.get("video_codec") or v_meta.get("codec") or "").lower()
    s_cdc = (s_meta.get("video_codec") or s_meta.get("codec") or "").lower()
    if v_cdc and s_cdc:
        if v_cdc == s_cdc:
            score += WEIGHT_CODEC_MATCH
            reasons.append(f"Codec match {v_cdc.upper()} (+{WEIGHT_CODEC_MATCH})")
        elif {v_cdc, s_cdc} <= {"x265", "hevc"}:
            score += WEIGHT_CODEC_MATCH
            reasons.append("HEVC/x265 match (+10)")
        elif {v_cdc, s_cdc} <= {"x264", "avc"}:
            score += WEIGHT_CODEC_MATCH
            reasons.append("AVC/x264 match (+10)")

    # 10. Audio Alignment (Very Weak Signal)
    v_aud = (v_meta.get("audio") or "").lower()
    s_aud = (s_meta.get("audio") or "").lower()
    if v_aud and s_aud:
        if v_aud == s_aud:
            score += WEIGHT_AUDIO_MATCH
            reasons.append(f"Audio profile match {v_aud.upper()} (+{WEIGHT_AUDIO_MATCH})")
        elif {v_aud, s_aud} <= {"truehd", "atmos"}:
            score += WEIGHT_AUDIO_MATCH
            reasons.append("TrueHD/Atmos compatibility (+10)")
        elif {v_aud, s_aud} <= {"dts-hd", "dts"}:
            score += WEIGHT_AUDIO_MATCH
            reasons.append("DTS profile compatibility (+10)")
        elif {v_aud, s_aud} <= {"ddp5.1", "ddp", "dd5.1", "ac3"}:
            score += WEIGHT_AUDIO_MATCH
            reasons.append("Dolby Digital profile compatibility (+10)")

    # 11. Re-upload / malformed filename penalty (website or tracker domain marker).
    if _REUPLOAD_REGEX.search(s_meta.get("raw_filename", "")):
        score -= WEIGHT_REUPLOAD_PENALTY
        reasons.append(
            f"Re-upload/website marker in filename (-{WEIGHT_REUPLOAD_PENALTY})"
        )

    # Special Anime Elevation: Guarantee compatible anime episode subtitles reach 100% (score >= 135)
    if is_anime and is_anime_ep_match and not is_unshared_fansub and score > 0:
        score = max(score, 135)

    # Percentage Normalization
    # A candidate for a different work must never read as a strong match, no
    # matter how many media properties happen to line up. Without this a release
    # group collision (the same group naming unrelated titles) can push an
    # unrelated subtitle to 100%.
    if _is_different_work(title_ratio, episode_match, season_match):
        score = min(score, 0)
        percentage = 0
        reasons.append(
            f"Different work (title similarity {title_ratio:.2f} "
            f"< {MIN_TIER_TITLE_RATIO}, no season/episode match) - capped at 0%"
        )
    elif not is_unshared_fansub and score >= 130:
        percentage = 100
    elif score > 0:
        percentage = max(1, min(99, int(round((score / 135.0) * 100))))
        if is_anime and is_anime_ep_match:
            percentage = max(percentage, 95)
    else:
        percentage = 0

    # Confidence Calculation (0.05 to 0.99)
    conf = 0.50
    if title_ratio >= 0.85:
        conf += 0.20
    if season_match or episode_match:
        conf += 0.20
    if has_source_match or service_match:
        conf += 0.08
    if fps_relation == "drift":
        conf -= 0.20
    confidence = max(0.05, min(0.99, round(conf, 2)))

    has_year_mismatch = bool(v_year and s_year and v_year != s_year)
    tier = determine_match_tier(
        is_hash_match=False,
        accepted=True,
        release_group_match=release_group_match,
        has_source_match=has_source_match,
        service_match=service_match,
        episode_match=episode_match,
        season_match=season_match,
        edition_match=edition_match,
        edition_unverified=bool(v_meta.get("edition")) and not s_meta.get("edition"),
        has_year_mismatch=has_year_mismatch,
        is_unshared_fansub=is_unshared_fansub,
        title_ratio=title_ratio,
    )

    # Deterministic tier floor: an exact hash or exact release-group match is a
    # 100% match by definition and must never be dragged down by soft score
    # arithmetic.
    if tier == MatchTier.EXACT:
        percentage = 100

    logger.debug(
        "[Soft Rank] Target: '%s' | Sub: '%s' | Tier: %s | Score: %d | Pct: %d%% | Conf: %.2f | Reasons: %s",
        v_meta.get("raw_filename"),
        s_meta.get("raw_filename"),
        tier,
        score,
        percentage,
        confidence,
        ", ".join(reasons),
    )

    return CompatibilityResult(
        accepted=True,
        score=score,
        confidence=confidence,
        match_method=match_method,
        match_tier=tier,
        title_match=title_ratio,
        season_match=season_match,
        episode_match=episode_match,
        release_group_match=release_group_match,
        service_match=service_match,
        source_match=source_match,
        edition_match=edition_match,
        fps_relation=fps_relation,
        is_hash_match=False,
        is_sdh=is_sdh,
        percentage=percentage,
        reasons=reasons,
    )


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


def deduplicate_subtitles(subtitles: list[SubtitleRelease]) -> list[SubtitleRelease]:
    """
    Deduplicate subtitle releases from multiple providers (SubDL, SubSource, OpenSubtitles).
    Retains the candidate with the highest compatibility quality and metadata completeness.
    """
    if not subtitles:
        return []

    seen: dict[str, SubtitleRelease] = {}
    result: list[SubtitleRelease] = []

    def _get_quality_tuple(s: SubtitleRelease) -> tuple[int, int, int]:
        is_hash = (
            1
            if (getattr(s, "is_hash_match", False) or getattr(s, "matched_by_hash", False))
            else 0
        )
        score_val = getattr(s, "score", 0) or 0
        name_len = len(getattr(s, "release_name", "") or "")
        return (is_hash, score_val, name_len)

    for sub in subtitles:
        url = getattr(sub, "download_url", "") or ""
        r_name = getattr(sub, "release_name", "") or ""
        lang = getattr(sub, "lang", "ara") or "ara"
        norm_name = sanitize_release_name(r_name).lower()
        sub_id = getattr(sub, "id", None)

        keys: list[str] = []
        if url and not url.startswith("/sub/"):
            keys.append(f"url:{url}")
        if sub_id:
            provider = getattr(sub, "provider", "")
            keys.append(f"id:{provider}:{sub_id}")
        if norm_name:
            keys.append(f"name:{lang}:{norm_name}")

        existing_key = None
        for k in keys:
            if k in seen:
                existing_key = k
                break

        if existing_key:
            prev = seen[existing_key]
            if _get_quality_tuple(sub) > _get_quality_tuple(prev):
                idx = result.index(prev)
                result[idx] = sub
                for k in keys:
                    seen[k] = sub
        else:
            result.append(sub)
            for k in keys:
                seen[k] = sub

    return result


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


def rank_subtitles(
    video_filename: str | None,
    subtitles: list[SubtitleRelease],
    preferred_languages: list[str] | None = None,
    discard_mismatches: bool = True,
    exclude_sdh: bool = False,
    season: int | None = None,
    episode: int | None = None,
    title: str | None = None,
    year: int | None = None,
    hi_preference: str = "neutral",
) -> list[SubtitleRelease]:
    """
    Ranks subtitle releases against a playback stream/video filename.
    Sorting priority:
    1. Language preference group (e.g. Arabic first, then English)
    2. Hard Filter Accepted status (accepted candidates before rejected)
    3. Hash match priority
    4. Compatibility score descending
    5. Confidence descending
    6. Deterministic tie-breaker
    """
    if not subtitles:
        return []

    # HI preference: "exclude" behaves like exclude_sdh; "prefer" boosts HI in sorting
    if hi_preference == "exclude":
        exclude_sdh = True
    prefer_hi = hi_preference == "prefer"

    # Optional SDH exclusion
    if exclude_sdh:
        subtitles = [s for s in subtitles if not _is_hearing_impaired(s)]

    # Deduplicate candidates across upstream providers
    deduped_subs = deduplicate_subtitles(subtitles)

    if video_filename:
        target_meta = extract_metadata(video_filename)
        if season is not None and target_meta.get("season") is None:
            target_meta["season"] = season
        if (
            episode is not None
            and target_meta.get("episode") is None
            and not target_meta.get("episodes")
        ):
            target_meta["episode"] = episode
            target_meta["episodes"] = {episode}
            target_meta["absolute_episode"] = episode
            target_meta["absolute_episode_confidence"] = "high"
        if year is not None and target_meta.get("year") is None:
            target_meta["year"] = year
        if title and not target_meta.get("title"):
            target_meta["title"] = sanitize_release_name(title)
    elif title or season is not None or episode is not None:
        target_meta = extract_metadata(title or "")
        if season is not None:
            target_meta["season"] = season
        if episode is not None:
            target_meta["episode"] = episode
            target_meta["episodes"] = {episode}
            target_meta["absolute_episode"] = episode
            target_meta["absolute_episode_confidence"] = "high"
        if year is not None:
            target_meta["year"] = year
        if title:
            target_meta["title"] = sanitize_release_name(title)
    else:
        target_meta = None

    scored_subs: list[SubtitleRelease] = []

    for sub in deduped_subs:
        rel_name = getattr(sub, "release_name", None) or (
            sub.get("release_name") if isinstance(sub, dict) else ""
        )
        is_hash = bool(
            getattr(sub, "is_hash_match", False)
            or getattr(sub, "matched_by_hash", False)
            or (sub.get("is_hash_match", False) if isinstance(sub, dict) else False)
            or (sub.get("matched_by_hash", False) if isinstance(sub, dict) else False)
        )

        if target_meta:
            scored_compat = calculate_compatibility(target_meta, rel_name, is_hash_match=is_hash)
            score_val = scored_compat.score
            pct_val = scored_compat.percentage
            tier_val = scored_compat.match_tier

            if isinstance(sub, dict):
                sub["score"] = score_val
                sub["match_percentage"] = pct_val
                sub["match_tier"] = tier_val
                sub["compatibility"] = scored_compat
            else:
                sub.score = score_val
                sub.match_percentage = pct_val
                sub.match_tier = tier_val
                if hasattr(sub, "compatibility"):
                    sub.compatibility = scored_compat

            if discard_mismatches and not scored_compat.accepted:
                if is_debug_ranking_enabled():
                    logger.info(
                        "[DEBUG_RANKING] DISCARDED: [%s][%s] '%s' | HardReject: %s",
                        getattr(sub, "provider", "unknown")
                        if not isinstance(sub, dict)
                        else sub.get("provider", "unknown"),
                        getattr(sub, "lang", "ara")
                        if not isinstance(sub, dict)
                        else sub.get("lang", "ara"),
                        rel_name,
                        scored_compat.hard_reject_reason,
                    )
                continue
        else:
            pop_score = get_source_popularity_percentage(rel_name)
            if is_hash:
                score_val = WEIGHT_EXACT_HASH + pop_score
                pct_val = 100
                tier_val = MatchTier.HASH
            else:
                score_val = pop_score
                pct_val = pop_score
                tier_val = MatchTier.FALLBACK

            if isinstance(sub, dict):
                sub["score"] = score_val
                sub["match_percentage"] = pct_val
                sub["match_tier"] = tier_val
            else:
                sub.score = score_val
                sub.match_percentage = pct_val
                sub.match_tier = tier_val

        scored_subs.append(sub)

    pref_langs = [normalize_to_iso639_2(lang) for lang in (preferred_languages or [])]
    lang_order = {lang: i for i, lang in enumerate(pref_langs)} if pref_langs else {}

    def _sort_key(s):
        s_lang = normalize_to_iso639_2(
            getattr(s, "lang", None) or (s.get("lang") if isinstance(s, dict) else "ara")
        )
        l_idx = lang_order.get(s_lang, 999)

        compat = (
            getattr(s, "compatibility", None) if not isinstance(s, dict) else s.get("compatibility")
        )
        accepted = getattr(compat, "accepted", True) if compat else (getattr(s, "score", 0) > -500)
        acc_idx = 0 if accepted else 1

        is_h = (
            0
            if (
                getattr(s, "is_hash_match", False)
                or getattr(s, "matched_by_hash", False)
                or (compat and compat.is_hash_match)
            )
            else 1
        )
        # Deterministic MatchTier drives the order. Cached ``status == "synced"``
        # is deliberately excluded: a stale sync artifact must never be promoted
        # above a candidate whose release metadata actually matches. A synced
        # candidate still floats up naturally whenever its own tokens earn a
        # better tier.
        raw_tier = getattr(compat, "match_tier", None) if compat else None
        if raw_tier is None and not isinstance(s, dict):
            raw_tier = getattr(s, "match_tier", None)
        if raw_tier is None and isinstance(s, dict):
            raw_tier = s.get("match_tier")
        try:
            tier_val = int(getattr(raw_tier, "value", raw_tier) or 0)
        except (TypeError, ValueError):
            tier_val = 0 if is_h == 0 else 4

        sc = getattr(s, "score", None) if not isinstance(s, dict) else s.get("score", 0)
        sc_val = sc if sc is not None else 0

        r_name = (
            getattr(s, "release_name", "") if not isinstance(s, dict) else s.get("release_name", "")
        )

        # HI preference: prefer hearing-impaired tracks when requested
        hi_rank = 0 if (prefer_hi and _is_hearing_impaired(s)) else 1

        conf = getattr(compat, "confidence", 0.5) if compat else 0.5
        if conf == "deterministic":
            conf_val = 1.0
        elif isinstance(conf, int | float):
            conf_val = float(conf)
        else:
            try:
                conf_val = float(conf)
            except (ValueError, TypeError):
                conf_val = 0.5

        pct = (
            getattr(s, "match_percentage", None)
            if not isinstance(s, dict)
            else s.get("match_percentage", 0)
        )
        pct_val = pct if pct is not None else 0

        return (
            l_idx,
            acc_idx,
            is_h,
            tier_val,
            -pct_val,
            -sc_val,
            hi_rank,
            -conf_val,
            r_name,
        )

    ranked = sorted(scored_subs, key=_sort_key)

    if is_debug_ranking_enabled():
        logger.info(
            "[DEBUG_RANKING] === Ranking Evaluation for Target: '%s' (Total Candidates: %d) ===",
            video_filename or "None",
            len(ranked),
        )
        for idx, item in enumerate(ranked, 1):
            rel_name = (
                getattr(item, "release_name", "")
                if not isinstance(item, dict)
                else item.get("release_name", "")
            )
            prov = (
                getattr(item, "provider", "unknown")
                if not isinstance(item, dict)
                else item.get("provider", "unknown")
            )
            lang = (
                getattr(item, "lang", "ara")
                if not isinstance(item, dict)
                else item.get("lang", "ara")
            )
            compat = (
                getattr(item, "compatibility", None)
                if not isinstance(item, dict)
                else item.get("compatibility")
            )
            tier_display = (
                getattr(compat, "match_tier", None)
                if compat
                else getattr(item, "match_tier", None)
            )
            sc = getattr(item, "score", 0) if not isinstance(item, dict) else item.get("score", 0)
            pct = (
                getattr(item, "match_percentage", 0)
                if not isinstance(item, dict)
                else item.get("match_percentage", 0)
            )

            conf = getattr(compat, "confidence", None) if compat else None
            accepted = getattr(compat, "accepted", True) if compat else True
            hard_reason = getattr(compat, "hard_reject_reason", None) if compat else None
            match_method = getattr(compat, "match_method", "filename") if compat else "unknown"
            is_hash = bool(
                getattr(compat, "is_hash_match", False)
                if compat
                else getattr(item, "is_hash_match", False)
            )
            source_match = getattr(compat, "source_match", None) if compat else None
            service_match = getattr(compat, "service_match", None) if compat else None
            ep_match = getattr(compat, "episode_match", None) if compat else None
            se_match = getattr(compat, "season_match", None) if compat else None
            fps_rel = getattr(compat, "fps_relation", "unknown") if compat else "unknown"
            top_reasons = getattr(compat, "reasons", [])[:3] if (compat and compat.reasons) else []

            logger.info(
                "[DEBUG_RANKING] #%d [%s][%s] '%s' | Tier: %s | Score: %s (%d%%) | Conf: %s | Accepted: %s | Method: %s | Hash: %s | SourceMatch: %s | ServiceMatch: %s | EpMatch: %s | SeMatch: %s | FPS: %s | Reject: %s | Reasons: %s",
                idx,
                prov,
                lang,
                rel_name,
                tier_display,
                sc,
                pct,
                conf,
                accepted,
                match_method,
                is_hash,
                source_match,
                service_match,
                ep_match,
                se_match,
                fps_rel,
                hard_reason,
                top_reasons,
            )
        logger.info("[DEBUG_RANKING] === End Ranking Evaluation ===")
    elif logger.isEnabledFor(logging.DEBUG):
        for idx, item in enumerate(ranked, 1):
            r_name = (
                getattr(item, "release_name", "")
                if not isinstance(item, dict)
                else item.get("release_name")
            )
            sc = getattr(item, "score", 0) if not isinstance(item, dict) else item.get("score")
            pct = (
                getattr(item, "match_percentage", 0)
                if not isinstance(item, dict)
                else item.get("match_percentage")
            )
            logger.debug("Final Rank #%d: [%d%%] (Score: %d) %s", idx, pct, sc, r_name)

    return ranked


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
        r"(?i)(?:ترجمة|مترجم|translation|translated\s+by|subtitle|encoded|"
        r"sync(?:ed)?\s+by|credit|corrected\s+by|timed\s+by)"
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
