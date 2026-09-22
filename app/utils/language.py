"""Language normalization and provider mapping utilities."""

from typing import Any

# Comprehensive list of supported languages for the Multi-Select configuration UI
AVAILABLE_LANGUAGES: list[dict[str, Any]] = [
    {
        "code": "ara",
        "iso639_1": "ar",
        "name": "Arabic",
        "subdl": "AR",
        "subsource": "Arabic",
        "aliases": ["arabic", "ar", "ar-sa"],
    },
    {
        "code": "eng",
        "iso639_1": "en",
        "name": "English",
        "subdl": "EN",
        "subsource": "English",
        "aliases": ["english", "en", "en-us", "en-gb"],
    },
    {
        "code": "fra",
        "iso639_1": "fr",
        "name": "French",
        "subdl": "FR",
        "subsource": "French",
        "aliases": ["french", "fr", "fre"],
    },
    {
        "code": "spa",
        "iso639_1": "es",
        "name": "Spanish",
        "subdl": "ES",
        "subsource": "Spanish",
        "aliases": ["spanish", "es"],
    },
    {
        "code": "deu",
        "iso639_1": "de",
        "name": "German",
        "subdl": "DE",
        "subsource": "German",
        "aliases": ["german", "de", "ger"],
    },
    {
        "code": "ita",
        "iso639_1": "it",
        "name": "Italian",
        "subdl": "IT",
        "subsource": "Italian",
        "aliases": ["italian", "it"],
    },
    {
        "code": "por",
        "iso639_1": "pt",
        "name": "Portuguese",
        "subdl": "PT",
        "subsource": "Portuguese",
        "aliases": ["portuguese", "pt", "pt-br"],
    },
    {
        "code": "tur",
        "iso639_1": "tr",
        "name": "Turkish",
        "subdl": "TR",
        "subsource": "Turkish",
        "aliases": ["turkish", "tr"],
    },
    {
        "code": "fas",
        "iso639_1": "fa",
        "name": "Persian (Farsi)",
        "subdl": "FA",
        "subsource": "Farsi",
        "aliases": ["persian", "farsi", "fa", "per"],
    },
    {
        "code": "ind",
        "iso639_1": "id",
        "name": "Indonesian",
        "subdl": "ID",
        "subsource": "Indonesian",
        "aliases": ["indonesian", "id"],
    },
    {
        "code": "rus",
        "iso639_1": "ru",
        "name": "Russian",
        "subdl": "RU",
        "subsource": "Russian",
        "aliases": ["russian", "ru"],
    },
    {
        "code": "nld",
        "iso639_1": "nl",
        "name": "Dutch",
        "subdl": "NL",
        "subsource": "Dutch",
        "aliases": ["dutch", "nl", "dut"],
    },
    {
        "code": "pol",
        "iso639_1": "pl",
        "name": "Polish",
        "subdl": "PL",
        "subsource": "Polish",
        "aliases": ["polish", "pl"],
    },
    {
        "code": "swe",
        "iso639_1": "sv",
        "name": "Swedish",
        "subdl": "SV",
        "subsource": "Swedish",
        "aliases": ["swedish", "sv"],
    },
    {
        "code": "nor",
        "iso639_1": "no",
        "name": "Norwegian",
        "subdl": "NO",
        "subsource": "Norwegian",
        "aliases": ["norwegian", "no"],
    },
    {
        "code": "dan",
        "iso639_1": "da",
        "name": "Danish",
        "subdl": "DA",
        "subsource": "Danish",
        "aliases": ["danish", "da"],
    },
    {
        "code": "fin",
        "iso639_1": "fi",
        "name": "Finnish",
        "subdl": "FI",
        "subsource": "Finnish",
        "aliases": ["finnish", "fi"],
    },
    {
        "code": "ell",
        "iso639_1": "el",
        "name": "Greek",
        "subdl": "EL",
        "subsource": "Greek",
        "aliases": ["greek", "el", "gre"],
    },
    {
        "code": "heb",
        "iso639_1": "he",
        "name": "Hebrew",
        "subdl": "HE",
        "subsource": "Hebrew",
        "aliases": ["hebrew", "he"],
    },
    {
        "code": "hin",
        "iso639_1": "hi",
        "name": "Hindi",
        "subdl": "HI",
        "subsource": "Hindi",
        "aliases": ["hindi", "hi"],
    },
    {
        "code": "urd",
        "iso639_1": "ur",
        "name": "Urdu",
        "subdl": "UR",
        "subsource": "Urdu",
        "aliases": ["urdu", "ur"],
    },
    {
        "code": "ben",
        "iso639_1": "bn",
        "name": "Bengali",
        "subdl": "BN",
        "subsource": "Bengali",
        "aliases": ["bengali", "bn"],
    },
    {
        "code": "vie",
        "iso639_1": "vi",
        "name": "Vietnamese",
        "subdl": "VI",
        "subsource": "Vietnamese",
        "aliases": ["vietnamese", "vi"],
    },
    {
        "code": "tha",
        "iso639_1": "th",
        "name": "Thai",
        "subdl": "TH",
        "subsource": "Thai",
        "aliases": ["thai", "th"],
    },
    {
        "code": "kor",
        "iso639_1": "ko",
        "name": "Korean",
        "subdl": "KO",
        "subsource": "Korean",
        "aliases": ["korean", "ko"],
    },
    {
        "code": "jpn",
        "iso639_1": "ja",
        "name": "Japanese",
        "subdl": "JA",
        "subsource": "Japanese",
        "aliases": ["japanese", "ja"],
    },
    {
        "code": "zho",
        "iso639_1": "zh",
        "name": "Chinese",
        "subdl": "ZH",
        "subsource": "Chinese",
        "aliases": ["chinese", "zh", "chi"],
    },
    {
        "code": "ron",
        "iso639_1": "ro",
        "name": "Romanian",
        "subdl": "RO",
        "subsource": "Romanian",
        "aliases": ["romanian", "ro", "rum"],
    },
    {
        "code": "ces",
        "iso639_1": "cs",
        "name": "Czech",
        "subdl": "CS",
        "subsource": "Czech",
        "aliases": ["czech", "cs", "cze"],
    },
    {
        "code": "hun",
        "iso639_1": "hu",
        "name": "Hungarian",
        "subdl": "HU",
        "subsource": "Hungarian",
        "aliases": ["hungarian", "hu"],
    },
    {
        "code": "ukr",
        "iso639_1": "uk",
        "name": "Ukrainian",
        "subdl": "UK",
        "subsource": "Ukrainian",
        "aliases": ["ukrainian", "uk"],
    },
    {
        "code": "msa",
        "iso639_1": "ms",
        "name": "Malay",
        "subdl": "MS",
        "subsource": "Malay",
        "aliases": ["malay", "ms", "may"],
    },
    {
        "code": "hrv",
        "iso639_1": "hr",
        "name": "Croatian",
        "subdl": "HR",
        "subsource": "Croatian",
        "aliases": ["croatian", "hr", "scr"],
    },
    {
        "code": "srp",
        "iso639_1": "sr",
        "name": "Serbian",
        "subdl": "SR",
        "subsource": "Serbian",
        "aliases": ["serbian", "sr", "scc"],
    },
    {
        "code": "bul",
        "iso639_1": "bg",
        "name": "Bulgarian",
        "subdl": "BG",
        "subsource": "Bulgarian",
        "aliases": ["bulgarian", "bg"],
    },
]

# Build lookup dictionaries from AVAILABLE_LANGUAGES
_ISO_639_2_MAP: dict[str, str] = {}
_SUBDL_LANG_MAP: dict[str, str] = {}
_SUBSOURCE_LANG_MAP: dict[str, str] = {}
_DISPLAY_PREFIX_MAP: dict[str, str] = {}

for entry in AVAILABLE_LANGUAGES:
    code = entry["code"]
    _ISO_639_2_MAP[code] = code
    _ISO_639_2_MAP[entry["iso639_1"]] = code
    for alias in entry.get("aliases", []):
        _ISO_639_2_MAP[alias.lower()] = code

    _SUBDL_LANG_MAP[code] = entry["subdl"]
    _SUBSOURCE_LANG_MAP[code] = entry["subsource"]
    _DISPLAY_PREFIX_MAP[code] = entry["subdl"]


def normalize_to_iso639_2(raw_lang: str | None, default: str = "ara") -> str:
    """
    Normalize any language string/code to standard 3-letter ISO-639-2 code.
    Defaults to 'ara' if unresolved.
    """
    if not raw_lang:
        return default
    cleaned = raw_lang.strip().lower()
    if cleaned in _ISO_639_2_MAP:
        return _ISO_639_2_MAP[cleaned]
    # Fallback to 3 letters if length allows
    if len(cleaned) == 3:
        return cleaned
    return default


def get_subdl_lang_code(iso_code: str) -> str:
    """Get Subdl query language code for an ISO-639-2 language."""
    iso_norm = normalize_to_iso639_2(iso_code)
    return _SUBDL_LANG_MAP.get(iso_norm, iso_norm.upper()[:2])


def get_subsource_lang_name(iso_code: str) -> str:
    """Get SubSource official API v1 language name for an ISO-639-2 language."""
    iso_norm = normalize_to_iso639_2(iso_code)
    return _SUBSOURCE_LANG_MAP.get(iso_norm, iso_norm.capitalize())


def get_display_prefix(iso_code: str) -> str:
    """Get Stremio hybrid track label prefix (e.g., 'AR', 'EN', 'FR')."""
    iso_norm = normalize_to_iso639_2(iso_code)
    return _DISPLAY_PREFIX_MAP.get(iso_norm, iso_norm.upper()[:2])


_OPENSUBTITLES_LANG_MAP: dict[str, str] = {
    entry["code"]: entry["iso639_1"] for entry in AVAILABLE_LANGUAGES
}
for entry in AVAILABLE_LANGUAGES:
    _OPENSUBTITLES_LANG_MAP[entry["iso639_1"]] = entry["iso639_1"]
    for alias in entry.get("aliases", []):
        _OPENSUBTITLES_LANG_MAP[alias.lower()] = entry["iso639_1"]


def get_opensubtitles_lang_code(iso_code: str) -> str:
    """Get OpenSubtitles API query language code (e.g., 'ar', 'en') for an ISO-639-2 language."""
    iso_norm = normalize_to_iso639_2(iso_code)
    return _OPENSUBTITLES_LANG_MAP.get(iso_norm, iso_norm[:2])


# SubtitleCat uses ISO-639-1-ish codes in its .srt filenames, with a few legacy
# (ISO-639-2/B) variants such as 'iw' for Hebrew and regional suffixes like 'pt-BR'.
_SUBTITLECAT_LANG_OVERRIDES: dict[str, set[str]] = {
    "heb": {"he", "iw"},
    "por": {"pt", "pt-br", "pt-pt"},
    "zho": {"zh", "zh-cn", "zh-tw", "zh-hans", "zh-hant"},
    "srp": {"sr", "scc"},
    "ron": {"ro", "rum"},
    "fas": {"fa", "per"},
    "msa": {"ms", "may"},
    "nld": {"nl", "dut"},
    "ces": {"cs", "cze"},
    "ell": {"el", "gre"},
    "fra": {"fr", "fre"},
    "deu": {"de", "ger"},
}


def get_subtitlecat_lang_codes(iso_code: str) -> set[str]:
    """
    Get the set of lowercase SubtitleCat language codes for an ISO-639-2 language.
    Includes ISO-639-1 code plus known regional/legacy variants (e.g. ``pt-BR``, ``iw``).
    """
    iso_norm = normalize_to_iso639_2(iso_code)
    base = get_opensubtitles_lang_code(iso_norm)
    codes = {base.lower()} if base else set()
    codes |= _SUBTITLECAT_LANG_OVERRIDES.get(iso_norm, set())
    return {code.lower() for code in codes if code}


def get_language_name(code: str, default: str = "Arabic") -> str:
    """Resolve clean display name for language code (e.g. 'ara' -> 'Arabic', 'eng' -> 'English')."""
    if not code:
        return default
    norm = normalize_to_iso639_2(code)
    for entry in AVAILABLE_LANGUAGES:
        if entry["code"] == norm:
            return entry["name"].split(" (")[0].strip()
    return default
