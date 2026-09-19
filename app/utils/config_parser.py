"""User configuration encoder and decoder for Stremio per-user manifest URLs."""

import base64
import json
import logging
import urllib.parse
from typing import Any

from app.config import settings
from app.models import UserPreferences

logger = logging.getLogger(__name__)


def encode_user_config(
    subdl_key: str | None = None,
    subsource_key: str | None = None,
    languages: list[str] | None = None,
    exclude_hi: bool = False,
    nuvio_mode: bool = False,
    opensubtitles_key: str | None = None,
    opensubtitles_api_key: str | None = None,
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

    return UserPreferences(
        subdl_key=effective_subdl,
        subsource_key=effective_subsource,
        opensubtitles_key=effective_opensubtitles,
        languages=languages,
        exclude_hi=exclude_hi,
        nuvio_mode=nuvio_mode,
    )
