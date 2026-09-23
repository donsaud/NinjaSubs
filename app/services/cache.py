"""High-Performance In-Memory Caching for NinjaSubs subtitle aggregation.

Implements an in-memory TTL cache with capacity for 1,000 entries and a TTL
of 6 hours (21,600s) using cachetools.TTLCache.
"""

import hashlib
import logging

from cachetools import TTLCache

from app.models import SubtitleRelease

logger = logging.getLogger("uvicorn.error")

# Global in-memory TTL cache: maxsize=1000, ttl=21600 (6 hours)
SUBTITLE_CACHE: TTLCache = TTLCache(maxsize=1000, ttl=21600)


def build_cache_key(
    media_type: str,
    imdb_id: str,
    season: int | str | None = None,
    episode: int | str | None = None,
    filename: str | None = None,
    video_hash: str | None = None,
    video_size: int | str | None = None,
    languages: list[str] | None = None,
    exclude_hi: bool = False,
    subdl_key: str | None = None,
    subsource_key: str | None = None,
    opensubtitles_key: str | None = None,
    hi_preference: str = "neutral",
    enable_subdl: bool = True,
    enable_subsource: bool = True,
    enable_opensubtitles: bool = True,
    enable_yifysubtitles: bool = True,
    enable_subtitlecat: bool = True,
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
    convert_ass_to_srt: bool = True,
    **kwargs,
) -> str:
    """
    Build a deterministic cache key for a subtitle aggregation request.
    Accounts for media type, ID, season/ep, playback filename, video hash/size,
    languages, exclude_hi, active API keys, and Phase-2 ranking preferences.
    """
    # Clean IMDb / Kitsu ID
    clean_id = str(imdb_id or "").strip().lower()

    # Clean season & episode
    s_str = str(season).strip() if season is not None and str(season).strip() != "" else ""
    e_str = str(episode).strip() if episode is not None and str(episode).strip() != "" else ""

    # Clean target stream parameters
    fn_clean = str(filename or "").strip().lower()
    vh_clean = str(video_hash or "").strip().lower()
    vs_clean = str(video_size or "").strip().lower()

    # Clean language preferences (sorted to ensure list ordering does not affect cache hit rate)
    langs = sorted(
        [
            str(language).strip().lower()
            for language in (languages or ["ara"])
            if str(language).strip()
        ]
    )
    lang_str = ",".join(langs)

    # API key hashes (preserves stateless multi-user isolation without leaking plain text keys)
    k1 = hashlib.sha256((subdl_key or "").strip().encode("utf-8")).hexdigest()[:8]
    k2 = hashlib.sha256((subsource_key or "").strip().encode("utf-8")).hexdigest()[:8]
    k3 = hashlib.sha256((opensubtitles_key or "").strip().encode("utf-8")).hexdigest()[:8]

    parts = [
        str(media_type or "").strip().lower(),
        clean_id,
        s_str,
        e_str,
        fn_clean,
        vh_clean,
        vs_clean,
        lang_str,
        "1" if exclude_hi else "0",
        k1,
        k2,
        k3,
        str(hi_preference or "neutral").strip().lower(),
        "1" if enable_subdl else "0",
        "1" if enable_subsource else "0",
        "1" if enable_opensubtitles else "0",
        "1" if enable_yifysubtitles else "0",
        "1" if enable_subtitlecat else "0",
        "1" if enable_rtl_fix else "0",
        "1" if enable_ad_removal else "0",
        "1" if keep_translator_credits else "0",
        "1" if fix_encoding else "0",
        "1" if clean_tags else "0",
        "1" if strip_colors else "0",
        "1" if clean_spacing else "0",
        "1" if clean_symbols else "0",
        "1" if clean_commas else "0",
        "1" if clean_timing else "0",
        "1" if strip_hi else "0",
        "1" if eastern_arabic_numerals else "0",
        "1" if strip_diacritics else "0",
        "1" if convert_ass_to_srt else "0",
    ]
    raw_key = ":".join(parts)
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def get_cached_subtitles(cache_key: str) -> list[SubtitleRelease] | None:
    """Retrieve deep-copied cached subtitle releases if present."""
    cached = SUBTITLE_CACHE.get(cache_key)
    if cached is None:
        return None
    logger.debug(f"[SubtitleCache] HIT for key {cache_key[:12]}")
    return [sub.model_copy(deep=True) for sub in cached]


def set_cached_subtitles(cache_key: str, subtitles: list[SubtitleRelease]) -> None:
    """Store deep-copied subtitle releases into the in-memory TTL cache."""
    SUBTITLE_CACHE[cache_key] = [sub.model_copy(deep=True) for sub in subtitles]
    logger.debug(f"[SubtitleCache] SET for key {cache_key[:12]} ({len(subtitles)} releases)")


def clear_subtitle_cache() -> None:
    """Clear all entries from the in-memory subtitle cache."""
    SUBTITLE_CACHE.clear()
    logger.info("[SubtitleCache] In-memory subtitle cache cleared.")
