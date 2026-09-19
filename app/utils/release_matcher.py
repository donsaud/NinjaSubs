"""Release-Matching, Sanitization, and Scoring Engine for Stremio/Nuvio Subtitles Addon.

Re-exports from app.services.ranking for backward compatibility.
"""

from app.services.ranking import (
    _INVALID_GROUPS,
    KNOWN_GROUPS,
    calculate_compatibility_percentage,
    calculate_match_score,
    clean_subtitle_filename,
    extract_clean_title,
    extract_stream_params,
    extract_target_filename,
    format_subtitle_title,
    get_source_popularity_percentage,
    get_source_popularity_score,
    is_codec_audio_match,
    is_group_match,
    is_source_match,
    levenshtein_ratio,
    parse_release_metadata,
    rank_and_sort_subtitles,
    sanitize_release_name,
)
from app.services.subtitle_matcher import (
    parse_video_metadata,
    rank_subtitles,
)

__all__ = [
    "sanitize_release_name",
    "clean_subtitle_filename",
    "extract_clean_title",
    "format_subtitle_title",
    "parse_release_metadata",
    "parse_video_metadata",
    "levenshtein_ratio",
    "is_group_match",
    "is_source_match",
    "is_codec_audio_match",
    "calculate_compatibility_percentage",
    "calculate_match_score",
    "get_source_popularity_percentage",
    "get_source_popularity_score",
    "extract_target_filename",
    "extract_stream_params",
    "rank_and_sort_subtitles",
    "rank_subtitles",
    "_INVALID_GROUPS",
    "KNOWN_GROUPS",
]
