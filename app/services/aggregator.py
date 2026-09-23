"""Subtitle Aggregation Engine for NinjaSubs with In-Memory Caching."""

import asyncio
import logging
import re

import httpx

from app.config import settings
from app.models import (
    SubtitleRelease,
    UserPreferences,
    badge_format_to_parts,
    normalize_badge_parts,
)
from app.providers.base import BaseSubtitleProvider
from app.providers.opensubtitles import OpenSubtitlesProvider
from app.providers.subdl import SubdlProvider
from app.providers.subsource import SubsourceProvider
from app.providers.subtitlecat import SubtitlecatProvider
from app.providers.yifysubtitles import YifysubtitlesProvider
from app.services.cache import (
    build_cache_key,
    get_cached_subtitles,
    set_cached_subtitles,
)
from app.services.subtitle_matcher import rank_subtitles
from app.utils.language import normalize_to_iso639_2

logger = logging.getLogger("uvicorn.error")


def clean_subtitle_display_name(name: str) -> str:
    """
    Clean filename for subtitle display label:
    - Strip subtitle file extensions (.srt, .vtt, .sub, .ass, .ssa, .mkv, .mp4, etc.)
    - Strip trailing hexadecimal provider hashes (e.g. _6e4b079e36dd0457)
    - Strip leading/trailing orphan brackets and artifacts (e.g. leading NRN], trailing ])
    - Preserve natural dots, hyphens, and balanced brackets.
    """
    if not name:
        return ""
    clean = re.sub(
        r"\.(?:srt|vtt|sub|ass|ssa|mkv|mp4|avi|webm|ts|iso|zip)$",
        "",
        name.strip(),
        flags=re.IGNORECASE,
    )
    clean = re.sub(r"_[a-f0-9]{8,32}$", "", clean, flags=re.IGNORECASE)
    clean = re.sub(
        r"\.(?:srt|vtt|sub|ass|ssa|mkv|mp4|avi|webm|ts|iso|zip)$", "", clean, flags=re.IGNORECASE
    )
    # Strip leading anime release / fansub group tags like [MSRT Fansub], [Shiniori-Raws], [Kaylith]
    while True:
        m = re.match(r"^\s*\[[^\]]+\]\s*(.+)$", clean)
        if m and m.group(1).strip():
            candidate = m.group(1).strip()
            clean_sep = re.sub(r"^[-_–—]\s*", "", candidate)
            if clean_sep.strip():
                clean = clean_sep.strip()
            else:
                clean = candidate
            if clean.startswith("[") and re.match(r"^\s*\[[^\]]+\]\s*(.+)$", clean):
                continue
            break
        else:
            break

    clean = re.sub(r"^[A-Za-z0-9_\-]{1,10}\]\s*", "", clean)
    clean = clean.lstrip("])} ")
    clean = clean.rstrip("[({ ")
    while clean.count("]") > clean.count("["):
        idx = clean.rfind("]")
        clean = clean[:idx] + clean[idx + 1 :]
    while clean.count("[") > clean.count("]"):
        idx = clean.find("[")
        clean = clean[:idx] + clean[idx + 1 :]
    while clean.count(")") > clean.count("("):
        idx = clean.rfind(")")
        clean = clean[:idx] + clean[idx + 1 :]
    while clean.count("(") > clean.count(")"):
        idx = clean.find("(")
        clean = clean[:idx] + clean[idx + 1 :]
    return clean.strip()


def clean_final_label(label: str) -> str:
    """
    Strictly eliminate trailing random hex IDs/hashes and leftover extensions from display label:
    1. Strip standard subtitle extensions if present.
    2. Strip trailing hexadecimal hashes/IDs (e.g., _b5f1bc3a5ec49533 or _01703060d61e07a7).
    3. Strip trailing orphan underscores or hyphens.
    """
    if not label:
        return ""
    # 1. Strip standard subtitle extensions if present
    label = re.sub(r"\.(?:srt|vtt|sub|ass|ssa)$", "", str(label).strip(), flags=re.IGNORECASE)
    # 2. Strip trailing hexadecimal hashes/IDs (e.g., _b5f1bc3a5ec49533 or _01703060d61e07a7)
    # Matches an underscore followed by 8 to 32 alphanumeric/hex characters at the end of the string
    label = re.sub(r"_[a-fA-F0-9]{8,32}$", "", label)
    label = re.sub(r"\.(?:srt|vtt|sub|ass|ssa)$", "", label, flags=re.IGNORECASE)
    label = re.sub(r"_[a-fA-F0-9]{8,32}$", "", label)
    # 3. Strip trailing orphan underscores or hyphens
    label = label.rstrip(" _-")
    return label


def format_informative_badge(
    release: SubtitleRelease,
    display_score: int,
    lang_name: str = "Arabic",
    source_tag: str = "SubDL",
    badge_format: str = "score_provider",
    badge_parts: list[str] | None = None,
) -> str:
    """
    Format a subtitle label by composing user-selected components in canonical order:
      - "score"    -> "[{pct}%]"
      - "provider" -> "[{provider}]"
      - "filename" -> "{clean_filename}"
      - "uploader" -> "(by {uploader})" (only when an uploader is known)

    `badge_parts` takes precedence. When omitted, the legacy `badge_format` preset is
    mapped to its component list for backward compatibility.

    Specifications:
    - pct: Calculated integer match percentage (e.g. 100, 95, 61).
    - source_provider: Canonical provider name (OpenSubtitles, SubDL, SubSource).
    - clean_filename: Full, natural release name / subtitle filename:
      * Strip file extension (.srt, .vtt, .ass).
      * Strip internal hexadecimal hash suffixes (e.g. trailing _6e4b079e36dd0457).
      * Leave dots, spaces, and hyphens intact without shortening or tokenizing.
    - No extra prefixes like "[NinjaSubs]" or language words like "Arabic".
    """
    pct = getattr(release, "match_percentage", None)
    if pct is None:
        pct = display_score
    if pct is None:
        pct = getattr(release, "score", 0)

    try:
        pct_int = int(round(float(pct))) if pct is not None else 0
    except (ValueError, TypeError):
        pct_int = 0

    if getattr(release, "is_hash_match", False):
        pct_int = 100

    prov = (getattr(release, "provider", None) or source_tag or "SubDL").strip().lower()
    if prov == "subsource":
        source_provider = "SubSource"
    elif prov in ("opensubtitles", "opensubtitlesv3", "os"):
        source_provider = "OpenSubtitles"
    elif prov == "yifysubtitles":
        source_provider = "YIFY"
    elif prov == "subtitlecat":
        source_provider = "SubtitleCat"
    else:
        source_provider = "SubDL"

    raw_filename = getattr(release, "release_name", "") or ""
    clean_filename = clean_subtitle_display_name(raw_filename)
    clean_name = re.sub(r"_[a-fA-F0-9]{8,32}$", "", clean_filename)
    clean_name = clean_final_label(clean_name)

    parts = normalize_badge_parts(badge_parts) if badge_parts is not None else badge_format_to_parts(
        badge_format
    )
    uploader = clean_final_label(str(getattr(release, "uploader", "") or "").strip())

    label = ""
    if "score" in parts:
        label = f"[{pct_int}%]"
    if "provider" in parts:
        label = f"{label} [{source_provider}]".strip()
    if "filename" in parts and clean_name:
        label = f"{label} {clean_name}".strip()
    if "uploader" in parts and uploader:
        label = f"{label} (by {uploader})".strip()

    if not label:
        label = clean_name or f"[{pct_int}%]"

    label = re.sub(r"_[a-fA-F0-9]{8,32}$", "", label)
    return clean_final_label(label)


format_subtitle_label = format_informative_badge


async def aggregate_subtitles(
    imdb_id: str,
    media_type: str = "movie",
    season: int | str | None = None,
    episode: int | str | None = None,
    filename: str | None = None,
    video_hash: str | None = None,
    video_size: int | str | None = None,
    languages: list[str] | None = None,
    exclude_hi: bool = False,
    user_preferences: UserPreferences | None = None,
    subdl_key: str | None = None,
    subsource_key: str | None = None,
    opensubtitles_key: str | None = None,
    title: str | None = None,
    year: int | None = None,
    http_client: httpx.AsyncClient | None = None,
    subdl_provider: BaseSubtitleProvider | None = None,
    subsource_provider: BaseSubtitleProvider | None = None,
    opensubtitles_provider: BaseSubtitleProvider | None = None,
    yifysubtitles_provider: BaseSubtitleProvider | None = None,
    subtitlecat_provider: BaseSubtitleProvider | None = None,
    use_cache: bool = True,
) -> list[SubtitleRelease]:
    """
    Fetch, deduplicate, rank, and cache subtitles from all configured upstream providers.
    Uses in-memory TTLCache to eliminate redundant queries for seeks, pauses, and client reconnections.
    """
    # Resolve preferences & API keys
    prefs = user_preferences or UserPreferences()
    effective_subdl_key = subdl_key if subdl_key is not None else prefs.subdl_key
    effective_subsource_key = subsource_key if subsource_key is not None else prefs.subsource_key
    effective_opensubtitles_key = (
        opensubtitles_key if opensubtitles_key is not None else prefs.opensubtitles_key
    )
    effective_langs = languages if languages is not None else prefs.languages
    hi_preference = getattr(prefs, "hi_preference", "neutral") or "neutral"
    effective_exclude_hi = (
        exclude_hi if exclude_hi is not None else prefs.exclude_hi
    ) or hi_preference == "exclude"

    # Normalize season / episode
    parsed_season: int | None = None
    parsed_episode: int | None = None
    if season is not None and str(season).strip() != "":
        try:
            parsed_season = int(season)
        except (ValueError, TypeError):
            pass
    if episode is not None and str(episode).strip() != "":
        try:
            parsed_episode = int(episode)
        except (ValueError, TypeError):
            pass

    is_series = (
        str(media_type).lower() in ("series", "tv", "anime") and parsed_episode is not None
    ) or (parsed_season is not None)
    if str(media_type).lower() == "anime":
        if parsed_episode is not None:
            is_series = True
            if parsed_season is None:
                parsed_season = 1
        elif is_series and parsed_season is None:
            parsed_season = 1

    # Check in-memory cache
    cache_key = build_cache_key(
        media_type=media_type,
        imdb_id=imdb_id,
        season=parsed_season,
        episode=parsed_episode,
        filename=filename,
        video_hash=video_hash,
        video_size=video_size,
        languages=effective_langs,
        exclude_hi=effective_exclude_hi,
        subdl_key=effective_subdl_key,
        subsource_key=effective_subsource_key,
        opensubtitles_key=effective_opensubtitles_key,
        hi_preference=hi_preference,
        enable_subdl=bool(getattr(prefs, "enable_subdl", True)),
        enable_subsource=bool(getattr(prefs, "enable_subsource", True)),
        enable_opensubtitles=bool(getattr(prefs, "enable_opensubtitles", False)),
        enable_yifysubtitles=bool(getattr(prefs, "enable_yifysubtitles", False)),
        enable_subtitlecat=bool(getattr(prefs, "enable_subtitlecat", False)),
        enable_rtl_fix=bool(getattr(prefs, "enable_rtl_fix", True)),
        enable_ad_removal=bool(getattr(prefs, "enable_ad_removal", True)),
        keep_translator_credits=bool(getattr(prefs, "keep_translator_credits", True)),
        fix_encoding=bool(getattr(prefs, "fix_encoding", True)),
        clean_tags=bool(getattr(prefs, "clean_tags", True)),
        strip_colors=bool(getattr(prefs, "strip_colors", False)),
        clean_spacing=bool(getattr(prefs, "clean_spacing", True)),
        clean_symbols=bool(getattr(prefs, "clean_symbols", True)),
        clean_commas=bool(getattr(prefs, "clean_commas", True)),
        clean_timing=bool(getattr(prefs, "clean_timing", True)),
        strip_hi=bool(getattr(prefs, "strip_hi", False)),
        eastern_arabic_numerals=bool(
            getattr(prefs, "eastern_arabic_numerals", False)
        ),
        strip_diacritics=bool(getattr(prefs, "strip_diacritics", False)),
        convert_ass_to_srt=bool(getattr(prefs, "convert_ass_to_srt", True)),
    )

    if use_cache:
        cached_results = get_cached_subtitles(cache_key)
        if cached_results is not None:
            logger.info(
                f"[Aggregator] Cache HIT for {imdb_id} (key: {cache_key[:12]}). Returning {len(cached_results)} subtitles."
            )
            return cached_results

    # Instantiate providers if not provided
    def resolve_provider(
        provider: BaseSubtitleProvider | None,
        provider_type: type[BaseSubtitleProvider],
    ) -> BaseSubtitleProvider:
        if provider is not None:
            return provider
        if http_client is None:
            raise ValueError("HTTP client is required when a provider is not injected")
        return provider_type(http_client)

    tasks = []

    # SubDL (per-user enabled + API key)
    if bool(getattr(prefs, "enable_subdl", True)):
        p_subdl = resolve_provider(subdl_provider, SubdlProvider)
        tasks.append(
            p_subdl.search_subtitles(
                imdb_id=imdb_id,
                is_series=is_series,
                season=parsed_season,
                episode=parsed_episode,
                title=title,
                year=year,
                api_key=effective_subdl_key,
                languages=effective_langs,
                exclude_hi=effective_exclude_hi,
            )
        )

    # SubSource (per-user enabled + API key)
    if bool(getattr(prefs, "enable_subsource", True)):
        p_subsource = resolve_provider(subsource_provider, SubsourceProvider)
        tasks.append(
            p_subsource.search_subtitles(
                imdb_id=imdb_id,
                is_series=is_series,
                season=parsed_season,
                episode=parsed_episode,
                title=title,
                year=year,
                api_key=effective_subsource_key,
                languages=effective_langs,
                exclude_hi=effective_exclude_hi,
                target_filename=filename,
            )
        )

    # Include OpenSubtitles if enabled and an API key/provider is available
    if bool(getattr(prefs, "enable_opensubtitles", False)) and (
        effective_opensubtitles_key
        or getattr(settings, "OPENSUBTITLES_API_KEY", "").strip()
        or opensubtitles_provider is not None
    ):
        p_opensubtitles = resolve_provider(opensubtitles_provider, OpenSubtitlesProvider)
        tasks.append(
            p_opensubtitles.search_subtitles(
                imdb_id=imdb_id,
                is_series=is_series,
                season=parsed_season,
                episode=parsed_episode,
                title=title,
                year=year,
                api_key=effective_opensubtitles_key,
                languages=effective_langs,
                exclude_hi=effective_exclude_hi,
                video_hash=video_hash,
                video_size=video_size,
            )
        )

    # YIFYSubtitles (no API key; movies only)
    if yifysubtitles_provider is not None or (
        http_client is not None
        and getattr(settings, "ENABLE_YIFYSUBTITLES", True)
        and bool(getattr(prefs, "enable_yifysubtitles", False))
    ):
        p_yify = resolve_provider(yifysubtitles_provider, YifysubtitlesProvider)
        tasks.append(
            p_yify.search_subtitles(
                imdb_id=imdb_id,
                is_series=is_series,
                season=parsed_season,
                episode=parsed_episode,
                title=title,
                year=year,
                languages=effective_langs,
                exclude_hi=effective_exclude_hi,
            )
        )

    # SubtitleCat (no API key; text search by title/year or SxxExx)
    if subtitlecat_provider is not None or (
        http_client is not None
        and getattr(settings, "ENABLE_SUBTITLECAT", True)
        and bool(getattr(prefs, "enable_subtitlecat", False))
    ):
        p_cat = resolve_provider(subtitlecat_provider, SubtitlecatProvider)
        tasks.append(
            p_cat.search_subtitles(
                imdb_id=imdb_id,
                is_series=is_series,
                season=parsed_season,
                episode=parsed_episode,
                title=title,
                year=year,
                languages=effective_langs,
                exclude_hi=effective_exclude_hi,
                target_filename=filename,
            )
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_releases: list[SubtitleRelease] = []
    for r in results:
        if isinstance(r, list):
            all_releases.extend(r)
        elif isinstance(r, Exception):
            logger.error(f"[Aggregator] Upstream provider query failed: {r}")

    # Deduplicate results based on release_name and normalized language (prioritizing hash matches)
    all_releases.sort(key=lambda r: 0 if getattr(r, "is_hash_match", False) else 1)
    seen_keys = set()
    deduped_releases: list[SubtitleRelease] = []
    for rel in all_releases:
        norm_lang = normalize_to_iso639_2(getattr(rel, "lang", "ara"))
        dedup_key = (rel.release_name.strip().lower(), norm_lang)
        if dedup_key not in seen_keys:
            seen_keys.add(dedup_key)
            deduped_releases.append(rel)

    # Score and rank subtitle releases using comprehensive Subtitle Matcher Engine
    ranked_releases = rank_subtitles(
        video_filename=filename,
        subtitles=deduped_releases,
        preferred_languages=effective_langs,
        discard_mismatches=True if (filename or episode is not None) else False,
        exclude_sdh=effective_exclude_hi,
        season=parsed_season,
        episode=parsed_episode,
        title=title,
        year=year,
        hi_preference=hi_preference,
    )

    if use_cache:
        set_cached_subtitles(cache_key, ranked_releases)

    return ranked_releases
