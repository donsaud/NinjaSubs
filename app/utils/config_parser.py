"""User configuration encoder and decoder for Stremio per-user manifest URLs."""

import base64
import json
import logging
import urllib.parse
from typing import Any

from app.config import settings
from app.models import (
    DEFAULT_BADGE_PARTS,
    UserPreferences,
    badge_format_to_parts,
    normalize_badge_parts,
)

logger = logging.getLogger(__name__)


def encode_user_config(
    subdl_key: str | None = None,
    subsource_key: str | None = None,
    languages: list[str] | None = None,
    exclude_hi: bool = False,
    nuvio_mode: bool = False,
    opensubtitles_key: str | None = None,
    opensubtitles_api_key: str | None = None,
    hi_preference: str | None = None,
    badge_parts: list[str] | None = None,
    badge_format: str | None = None,
    enable_subdl: bool = True,
    enable_subsource: bool = True,
    enable_opensubtitles: bool = False,
    enable_yifysubtitles: bool = False,
    enable_subtitlecat: bool = False,
    enable_rtl_fix: bool = True,
    enable_ad_removal: bool = True,
    keep_translator_credits: bool = True,
    fix_encoding: bool = True,
    clean_tags: bool = True,
    strip_colors: bool = False,
    clean_spacing: bool = True,
    clean_symbols: bool = True,
    clean_commas: bool = True,
    clean_timing: bool = True,
    strip_hi: bool = False,
    eastern_arabic_numerals: bool = False,
    strip_diacritics: bool = False,
) -> str:
    """
    Encode user configuration into a URL-safe base64 string matching community addons.
    """
    payload: dict[str, Any] = {}
    if subdl_key and subdl_key.strip():
        payload["subdl_key"] = subdl_key.strip()
    if subsource_key and subsource_key.strip():
        payload["subsource_key"] = subsource_key.strip()
    os_key = opensubtitles_key or opensubtitles_api_key
    if os_key and os_key.strip():
        payload["opensubtitles_key"] = os_key.strip()
    if languages:
        payload["languages"] = [
            language.strip().lower() for language in languages if language.strip()
        ]
    else:
        payload["languages"] = ["ara"]
    if exclude_hi:
        payload["exclude_hi"] = True
    if nuvio_mode:
        payload["nuvio_mode"] = True

    if hi_preference in ("prefer", "exclude"):
        payload["hi_preference"] = hi_preference

    if enable_subdl is False:
        payload["enable_subdl"] = False
    if enable_subsource is False:
        payload["enable_subsource"] = False
    if enable_opensubtitles:
        payload["enable_opensubtitles"] = True
    if enable_yifysubtitles:
        payload["enable_yifysubtitles"] = True
    if enable_subtitlecat:
        payload["enable_subtitlecat"] = True
    if enable_rtl_fix is False:
        payload["enable_rtl_fix"] = False
    if enable_ad_removal is False:
        payload["enable_ad_removal"] = False
    if keep_translator_credits is False:
        payload["keep_translator_credits"] = False
    # fix_encoding is baked into the core engine (always active) — never serialized.
    if clean_tags is False:
        payload["clean_tags"] = False
    if strip_colors:
        payload["strip_colors"] = True
    if clean_spacing is False:
        payload["clean_spacing"] = False
    if clean_symbols is False:
        payload["clean_symbols"] = False
    if clean_commas is False:
        payload["clean_commas"] = False
    if clean_timing is False:
        payload["clean_timing"] = False
    if strip_hi:
        payload["strip_hi"] = True
    if eastern_arabic_numerals:
        payload["eastern_arabic_numerals"] = True
    if strip_diacritics:
        payload["strip_diacritics"] = True

    if badge_parts is not None:
        resolved_parts = normalize_badge_parts(badge_parts)
    elif badge_format is not None:
        resolved_parts = badge_format_to_parts(badge_format)
    else:
        resolved_parts = list(DEFAULT_BADGE_PARTS)

    if resolved_parts != DEFAULT_BADGE_PARTS:
        payload["badge_parts"] = resolved_parts

    json_bytes = json.dumps(payload).encode("utf-8")
    return base64.urlsafe_b64encode(json_bytes).decode("utf-8").rstrip("=")


def parse_user_config(
    config_str: str | None = None,
    query_subdl: str | None = None,
    query_subsource: str | None = None,
    query_opensubtitles: str | None = None,
) -> UserPreferences:
    """
    Extract UserPreferences from URL path or query params.
    Gracefully handles malformed Base64/JSON without crashing.
    Falls back to server-wide environment settings if not provided by user.
    """
    user_subdl: str | None = query_subdl.strip() if query_subdl else None
    user_subsource: str | None = query_subsource.strip() if query_subsource else None
    user_opensubtitles: str | None = query_opensubtitles.strip() if query_opensubtitles else None
    languages: list[str] = []
    exclude_hi: bool = False
    nuvio_mode: bool = False
    hi_preference: str = "neutral"
    badge_parts: list[str] | None = None
    enable_subdl: bool = True
    enable_subsource: bool = True
    enable_opensubtitles: bool = False
    enable_yifysubtitles: bool = False
    enable_subtitlecat: bool = False
    enable_rtl_fix: bool = True
    enable_ad_removal: bool = True
    keep_translator_credits: bool = True
    clean_tags: bool = True
    strip_colors: bool = False
    clean_spacing: bool = True
    clean_symbols: bool = True
    clean_commas: bool = True
    clean_timing: bool = True
    strip_hi: bool = False
    eastern_arabic_numerals: bool = False
    strip_diacritics: bool = False

    def _as_bool(value: Any, default: bool = True) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() not in ("0", "false", "no", "off", "")

    if config_str:
        clean_config = config_str.strip()

        # 1. Try Base64 JSON (primary standard format)
        try:
            padded = clean_config + "=" * (-len(clean_config) % 4)
            decoded_bytes = base64.urlsafe_b64decode(padded.encode("utf-8"))
            data = json.loads(decoded_bytes.decode("utf-8"))
            if isinstance(data, dict):
                if not user_subdl:
                    user_subdl = data.get("subdl_key") or data.get("subdl") or data.get("subdlKey")
                if not user_subsource:
                    user_subsource = (
                        data.get("subsource_key")
                        or data.get("subsource")
                        or data.get("subsourceKey")
                    )
                if not user_opensubtitles:
                    user_opensubtitles = (
                        data.get("opensubtitles_key")
                        or data.get("opensubtitles_api_key")
                        or data.get("opensubtitles")
                        or data.get("opensubtitlesKey")
                    )

                cfg_langs = data.get("languages") or data.get("langs")
                if isinstance(cfg_langs, list):
                    languages = [
                        str(language).strip().lower()
                        for language in cfg_langs
                        if str(language).strip()
                    ]
                elif isinstance(cfg_langs, str):
                    languages = [
                        language.strip().lower()
                        for language in cfg_langs.split(",")
                        if language.strip()
                    ]

                exclude_hi = bool(
                    data.get("exclude_hi") or data.get("excludeHI") or data.get("no_hi")
                )
                if "nuvio_mode" in data:
                    nuvio_mode = bool(data["nuvio_mode"])
                elif "stremio_mode" in data:
                    nuvio_mode = not bool(data["stremio_mode"])

                raw_hi = str(data.get("hi_preference") or "").strip().lower()
                if raw_hi in ("prefer", "exclude"):
                    hi_preference = raw_hi

                if "badge_parts" in data:
                    badge_parts = normalize_badge_parts(data.get("badge_parts"))
                else:
                    raw_badge = str(
                        data.get("badge_format") or data.get("badge_style") or ""
                    ).strip().lower()
                    if raw_badge:
                        badge_parts = badge_format_to_parts(raw_badge)

                if "enable_subdl" in data:
                    enable_subdl = _as_bool(data.get("enable_subdl"), True)
                if "enable_subsource" in data:
                    enable_subsource = _as_bool(data.get("enable_subsource"), True)
                if "enable_opensubtitles" in data:
                    enable_opensubtitles = _as_bool(data.get("enable_opensubtitles"), False)
                if "enable_yifysubtitles" in data or "yifysubtitles" in data:
                    enable_yifysubtitles = _as_bool(
                        data.get("enable_yifysubtitles", data.get("yifysubtitles")), False
                    )
                if "enable_subtitlecat" in data or "subtitlecat" in data:
                    enable_subtitlecat = _as_bool(
                        data.get("enable_subtitlecat", data.get("subtitlecat")), False
                    )
                if "enable_rtl_fix" in data or "rtl_fix" in data:
                    enable_rtl_fix = _as_bool(
                        data.get("enable_rtl_fix", data.get("rtl_fix")), True
                    )
                if "enable_ad_removal" in data or "ad_removal" in data or "remove_ads" in data:
                    enable_ad_removal = _as_bool(
                        data.get(
                            "enable_ad_removal",
                            data.get("ad_removal", data.get("remove_ads")),
                        ),
                        True,
                    )
                if "keep_translator_credits" in data or "keep_credits" in data:
                    keep_translator_credits = _as_bool(
                        data.get("keep_translator_credits", data.get("keep_credits")), True
                    )
                # fix_encoding is baked into the core engine (always active) — legacy
                # payload values are intentionally ignored.
                if "clean_tags" in data:
                    clean_tags = _as_bool(data.get("clean_tags"), True)
                if "strip_colors" in data:
                    strip_colors = _as_bool(data.get("strip_colors"), False)
                if "clean_spacing" in data:
                    clean_spacing = _as_bool(data.get("clean_spacing"), True)
                if "clean_symbols" in data:
                    clean_symbols = _as_bool(data.get("clean_symbols"), True)
                if "clean_commas" in data:
                    clean_commas = _as_bool(data.get("clean_commas"), True)
                if "clean_timing" in data:
                    clean_timing = _as_bool(data.get("clean_timing"), True)
                # Legacy pre-flattening key: apply to every text-cleaning toggle.
                if "clean_syntax" in data or "clean_formatting" in data:
                    legacy = _as_bool(
                        data.get("clean_syntax", data.get("clean_formatting")), True
                    )
                    clean_tags = legacy
                    clean_spacing = legacy
                    clean_symbols = legacy
                    clean_commas = legacy
                    clean_timing = legacy
                if "strip_hi" in data or "strip_hi_labels" in data:
                    strip_hi = _as_bool(
                        data.get("strip_hi", data.get("strip_hi_labels")), False
                    )
                if "eastern_arabic_numerals" in data or "eastern_numerals" in data:
                    eastern_arabic_numerals = _as_bool(
                        data.get(
                            "eastern_arabic_numerals", data.get("eastern_numerals")
                        ),
                        False,
                    )
                if "strip_diacritics" in data or "stripDiacritics" in data:
                    strip_diacritics = _as_bool(
                        data.get("strip_diacritics", data.get("stripDiacritics")),
                        False,
                    )
        except Exception as e:
            logger.debug(f"Base64 JSON decode skipped for config string: {e}")

        # 2. Try URL query string format (subdl=KEY&subsource=KEY&opensubtitles=KEY&languages=ara&exclude_hi=1)
        if not user_subdl or not user_subsource or not user_opensubtitles:
            try:
                parsed_qs = urllib.parse.parse_qs(clean_config)
                if not user_subdl:
                    for k in ("subdl", "subdl_key", "subdlKey"):
                        if k in parsed_qs:
                            user_subdl = parsed_qs[k][0]
                            break
                if not user_subsource:
                    for k in ("subsource", "subsource_key", "subsourceKey"):
                        if k in parsed_qs:
                            user_subsource = parsed_qs[k][0]
                            break
                if not user_opensubtitles:
                    for k in (
                        "opensubtitles",
                        "opensubtitles_key",
                        "opensubtitles_api_key",
                        "opensubtitlesKey",
                    ):
                        if k in parsed_qs:
                            user_opensubtitles = parsed_qs[k][0]
                            break
                if not languages and "languages" in parsed_qs:
                    languages = [
                        language.strip().lower()
                        for language in parsed_qs["languages"][0].split(",")
                        if language.strip()
                    ]
                if not exclude_hi and "exclude_hi" in parsed_qs:
                    exclude_hi = parsed_qs["exclude_hi"][0].lower() in ("1", "true", "yes")
                if "nuvio_mode" in parsed_qs:
                    nuvio_mode = parsed_qs["nuvio_mode"][0].lower() in ("1", "true", "yes")
                if hi_preference == "neutral" and "hi_preference" in parsed_qs:
                    raw_hi = parsed_qs["hi_preference"][0].strip().lower()
                    if raw_hi in ("prefer", "exclude"):
                        hi_preference = raw_hi
                if badge_parts is None and "badge_parts" in parsed_qs:
                    badge_parts = normalize_badge_parts(parsed_qs["badge_parts"][0])
                if badge_parts is None and "badge_format" in parsed_qs:
                    raw_badge = parsed_qs["badge_format"][0].strip().lower()
                    if raw_badge:
                        badge_parts = badge_format_to_parts(raw_badge)
                if "enable_subdl" in parsed_qs:
                    enable_subdl = _as_bool(parsed_qs["enable_subdl"][0], True)
                if "enable_subsource" in parsed_qs:
                    enable_subsource = _as_bool(parsed_qs["enable_subsource"][0], True)
                if "enable_opensubtitles" in parsed_qs:
                    enable_opensubtitles = _as_bool(parsed_qs["enable_opensubtitles"][0], False)
                if "enable_yifysubtitles" in parsed_qs:
                    enable_yifysubtitles = _as_bool(parsed_qs["enable_yifysubtitles"][0], False)
                if "enable_subtitlecat" in parsed_qs:
                    enable_subtitlecat = _as_bool(parsed_qs["enable_subtitlecat"][0], False)
                if "enable_rtl_fix" in parsed_qs:
                    enable_rtl_fix = _as_bool(parsed_qs["enable_rtl_fix"][0], True)
                if "enable_ad_removal" in parsed_qs:
                    enable_ad_removal = _as_bool(parsed_qs["enable_ad_removal"][0], True)
                if "keep_translator_credits" in parsed_qs:
                    keep_translator_credits = _as_bool(
                        parsed_qs["keep_translator_credits"][0], True
                    )
                # fix_encoding is baked into the core engine (always active) — ignored.
                if "clean_tags" in parsed_qs:
                    clean_tags = _as_bool(parsed_qs["clean_tags"][0], True)
                if "strip_colors" in parsed_qs:
                    strip_colors = _as_bool(parsed_qs["strip_colors"][0], False)
                if "clean_spacing" in parsed_qs:
                    clean_spacing = _as_bool(parsed_qs["clean_spacing"][0], True)
                if "clean_symbols" in parsed_qs:
                    clean_symbols = _as_bool(parsed_qs["clean_symbols"][0], True)
                if "clean_commas" in parsed_qs:
                    clean_commas = _as_bool(parsed_qs["clean_commas"][0], True)
                if "clean_timing" in parsed_qs:
                    clean_timing = _as_bool(parsed_qs["clean_timing"][0], True)
                if "clean_syntax" in parsed_qs:
                    legacy_qs = _as_bool(parsed_qs["clean_syntax"][0], True)
                    clean_tags = legacy_qs
                    clean_spacing = legacy_qs
                    clean_symbols = legacy_qs
                    clean_commas = legacy_qs
                    clean_timing = legacy_qs
                if "strip_hi" in parsed_qs:
                    strip_hi = _as_bool(parsed_qs["strip_hi"][0], False)
                if "eastern_arabic_numerals" in parsed_qs:
                    eastern_arabic_numerals = _as_bool(
                        parsed_qs["eastern_arabic_numerals"][0], False
                    )
                if "strip_diacritics" in parsed_qs:
                    strip_diacritics = _as_bool(parsed_qs["strip_diacritics"][0], False)
            except Exception:
                pass

        # 3. Try pipe/colon format (subdl:KEY|subsource:KEY|opensubtitles:KEY) or positional (k1|k2|ara|0|0|k3)
        if not user_subdl or not user_subsource or not user_opensubtitles:
            if "|" in clean_config:
                parts = clean_config.split("|")
                has_colons = any(":" in p for p in parts)
                if has_colons:
                    for part in parts:
                        if ":" in part:
                            k, v = part.split(":", 1)
                            k_lower = k.lower().strip()
                            if k_lower in ("subdl", "subdl_key") and not user_subdl:
                                user_subdl = v.strip()
                            elif k_lower in ("subsource", "subsource_key") and not user_subsource:
                                user_subsource = v.strip()
                            elif (
                                k_lower
                                in ("opensubtitles", "opensubtitles_key", "opensubtitles_api_key")
                                and not user_opensubtitles
                            ):
                                user_opensubtitles = v.strip()
                else:
                    if len(parts) >= 1 and not user_subdl and parts[0].strip():
                        user_subdl = parts[0].strip()
                    if len(parts) >= 2 and not user_subsource and parts[1].strip():
                        user_subsource = parts[1].strip()
                    if len(parts) >= 3 and not languages and parts[2].strip():
                        languages = [
                            language.strip().lower()
                            for language in parts[2].split(",")
                            if language.strip()
                        ]
                    if len(parts) >= 4 and not exclude_hi and parts[3].strip():
                        exclude_hi = parts[3].strip().lower() in ("1", "true", "yes")
                    if len(parts) >= 5 and not nuvio_mode and parts[4].strip():
                        nuvio_mode = parts[4].strip().lower() in ("1", "true", "yes")
                    if len(parts) >= 6 and not user_opensubtitles and parts[5].strip():
                        user_opensubtitles = parts[5].strip()

    # Fallback to server-wide environment settings
    effective_subdl = (user_subdl or settings.SUBDL_API_KEY or "").strip()
    effective_subsource = (user_subsource or settings.SUBSOURCE_API_KEY or "").strip()
    effective_opensubtitles = (
        user_opensubtitles or getattr(settings, "OPENSUBTITLES_API_KEY", "") or ""
    ).strip()

    if not languages:
        languages = ["ara"]

    logger.info(
        f"[Config Check] Exclude HI: {exclude_hi} | Languages: {languages} | OpenSubtitles: {'Yes' if effective_opensubtitles else 'No'}"
    )

    if hi_preference == "neutral" and exclude_hi:
        hi_preference = "exclude"

    return UserPreferences(
        subdl_key=effective_subdl,
        subsource_key=effective_subsource,
        opensubtitles_key=effective_opensubtitles,
        languages=languages,
        exclude_hi=exclude_hi,
        nuvio_mode=nuvio_mode,
        hi_preference=hi_preference,
        badge_parts=normalize_badge_parts(badge_parts),
        enable_subdl=enable_subdl,
        enable_subsource=enable_subsource,
        enable_opensubtitles=enable_opensubtitles,
        enable_yifysubtitles=enable_yifysubtitles,
        enable_subtitlecat=enable_subtitlecat,
        enable_rtl_fix=enable_rtl_fix,
        enable_ad_removal=enable_ad_removal,
        keep_translator_credits=keep_translator_credits,
        # Baked into the core engine — always active regardless of payload.
        fix_encoding=True,
        clean_tags=clean_tags,
        strip_colors=strip_colors,
        clean_spacing=clean_spacing,
        clean_symbols=clean_symbols,
        clean_commas=clean_commas,
        clean_timing=clean_timing,
        strip_hi=strip_hi,
        eastern_arabic_numerals=eastern_arabic_numerals,
        strip_diacritics=strip_diacritics,
    )
