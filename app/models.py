"""Data models for Stremio protocol and internal subtitle representations."""

from enum import IntEnum
from typing import Any

from pydantic import BaseModel, Field


class MatchTier(IntEnum):
    """
    Tier-based matching hierarchy:
    Tier 0 (HASH): Deterministic binary moviehash match (100% sync guaranteed).
    Tier 1 (EXACT): Exact release name, release group, or exact source+service match.
    Tier 2 (SOURCE_FAMILY): Compatible source family match (disc / web) or streaming service match.
    Tier 3 (CLOSE): Partial or close match (same episode, close resolution/codec).
    Tier 4 (FALLBACK): General fallback match (title-only or default heuristics).
    """

    HASH = 0
    EXACT = 1
    SOURCE_FAMILY = 2
    CLOSE = 3
    FALLBACK = 4

    def __str__(self) -> str:
        return self.name.lower()

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.name.lower() == other.lower() or str(self.value) == other
        return super().__eq__(other)

    def __hash__(self) -> int:
        return super().__hash__()


class Manifest(BaseModel):
    """Stremio Addon v3 Manifest model."""

    id: str = "org.ninjasubs.addon"
    name: str = "NinjaSubs"
    version: str = "1.0.0"
    description: str = (
        "A fast pass-through proxy that fetches, extracts, and streams native subtitles directly to Stremio."
    )
    logo: str | None = None
    icon: str | None = None
    resources: list[str | dict[str, Any]] = Field(default_factory=lambda: list[str | dict[str, Any]](["subtitles"]))
    types: list[str] = ["movie", "series", "anime"]
    idPrefixes: list[str] = ["tt", "kitsu"]
    catalogs: list[dict[str, Any]] = Field(default_factory=list)
    behaviorHints: dict[str, Any] | None = Field(
        default_factory=lambda: {"configurable": True, "configurationRequired": False}
    )


class SubtitleItem(BaseModel):
    """Stremio Subtitle entry schema."""

    id: str = Field(description="Unique hash or identifier for the subtitle")
    url: str = Field(description="Direct accessible URL to fetch the .srt file")
    lang: str = Field(
        description="Language tag: 'ara | <release_filename>' for frontend visibility"
    )
    title: str = Field(
        description="Display title formatted as '[{score}%] [{Provider}] {clean_filename}'"
    )
    format: str = Field(default="srt", description="Subtitle format extension")


class SubtitlesResponse(BaseModel):
    """Stremio Addon Subtitles endpoint response schema."""

    subtitles: list[SubtitleItem] = Field(default_factory=list)


class ParsedMediaID(BaseModel):
    """Parsed representation of Stremio media identifier."""

    raw_id: str
    imdb_id: str
    is_series: bool = False
    season: int | None = None
    episode: int | None = None


class SubtitleRelease(BaseModel):
    """Normalized upstream subtitle metadata representation."""

    release_name: str
    download_url: str
    provider: str  # "subdl" or "subsource"
    format: str = "srt"
    hearing_impaired: bool = False
    lang: str = "ara"
    score: int = 0
    match_percentage: int = 0
    is_hash_match: bool = False
    match_tier: MatchTier | None = None
    compatibility: Any | None = None


class UserPreferences(BaseModel):
    """User preferences decoded from stateless URL-based configuration payload."""

    subdl_key: str = ""
    subsource_key: str = ""
    opensubtitles_key: str = ""
    languages: list[str] = Field(default_factory=lambda: ["ara"])
    exclude_hi: bool = False
    nuvio_mode: bool = False  # True: Clean ISO code ("ara") & Nuvio ID; False: Stremio format ("ara | ...", "AR [★ Match] | ...")

    @property
    def opensubtitles_api_key(self) -> str:
        return self.opensubtitles_key

    def __iter__(self):
        """Allow backward-compatible tuple unpacking: subdl, subsource = parse_user_config(...)."""
        yield self.subdl_key
        yield self.subsource_key
