"""Subtitle Aggregation Engine for NinjaSubs with In-Memory Caching."""

import asyncio
import logging
import re

import httpx

from app.config import settings
from app.models import SubtitleRelease, UserPreferences
from app.providers.base import BaseSubtitleProvider
from app.providers.opensubtitles import OpenSubtitlesProvider
from app.providers.subdl import SubdlProvider
from app.providers.subsource import SubsourceProvider
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
) -> str:
    """
    Format subtitle label strictly as:
    f"[{pct}%] [{source_provider}] {clean_filename}"

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
    else:
        source_provider = "SubDL"

    raw_filename = getattr(release, "release_name", "") or ""
    clean_filename = clean_subtitle_display_name(raw_filename)
    clean_name = re.sub(r"_[a-fA-F0-9]{8,32}$", "", clean_filename)
    clean_name = clean_final_label(clean_name)

    if clean_name:
        label = f"[{pct_int}%] [{source_provider}] {clean_name}"
    else:
        label = f"[{pct_int}%] [{source_provider}]"

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
    effective_exclude_hi = exclude_hi if exclude_hi is not None else prefs.exclude_hi

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

    p_subdl = resolve_provider(subdl_provider, SubdlProvider)
    p_subsource = resolve_provider(subsource_provider, SubsourceProvider)
    p_opensubtitles = None
    if (
        effective_opensubtitles_key
        or getattr(settings, "OPENSUBTITLES_API_KEY", "").strip()
        or opensubtitles_provider is not None
    ):
        p_opensubtitles = resolve_provider(opensubtitles_provider, OpenSubtitlesProvider)

    tasks = [
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
        ),
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
        ),
    ]

    # Include OpenSubtitles if API key is present, server default is configured, or mock provider injected
    if p_opensubtitles is not None:
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
    )

    if use_cache:
        set_cached_subtitles(cache_key, ranked_releases)

    return ranked_releases
