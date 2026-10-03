"""Data models for Stremio protocol and internal subtitle representations."""

from enum import IntEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


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
    version: str = "1.1.0"
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
    matched_by_hash: bool = False  # explicit OpenSubtitles MovieHash flag (synced with is_hash_match)
    status: str = ""  # e.g. "synced" when a synced artifact exists for the current stream
    match_tier: MatchTier | None = None
    compatibility: Any | None = None
    uploader: str = ""  # Subtitle uploader/author username, when provided upstream

    # --- synchronization evidence (search-time) ------------------------- #
    # These are separate from match_percentage, which continues to mean exactly
    # what it always has: content compatibility. A high match_percentage never
    # implies synchronization; these fields carry that separate claim, and both
    # default to "no claim made" rather than to anything positive.
    sync_state: str | None = None  # e.g. "verified_synced" / "probable_sync"
    sync_verification: str | None = None  # "verified" / "cached" / "predicted" / "unknown"
    sync_confidence: float | None = None
    sync_reasons: list[str] = Field(default_factory=list)
    # Position assigned by the existing matcher, captured before any
    # sync-evidence re-sort so content ordering has a single source of truth.
    sync_base_rank: int | None = None
    # Position of this candidate's language within the user's preferred list.
    # User language preference is an explicit choice, so it outranks any
    # synchronization evidence and is never overridden by it.
    sync_lang_rank: int | None = None

    @model_validator(mode="after")
    def _sync_hash_flags(self):
        """Keep the two MovieHash flags consistent (either implies the other)."""
        if self.matched_by_hash or self.is_hash_match:
            self.matched_by_hash = True
            self.is_hash_match = True
        return self


# Subtitle badge components, in canonical display order.
#   "score"    -> "[100%]"
#   "provider" -> "[SubDL]"
#   "filename" -> "Release.Name"
#   "uploader" -> "(by username)"
CANONICAL_BADGE_PARTS: tuple[str, ...] = ("score", "provider", "filename", "uploader")
DEFAULT_BADGE_PARTS: list[str] = ["score", "provider", "filename", "uploader"]

_LEGACY_BADGE_FORMATS: dict[str, list[str]] = {
    "score_provider": ["score", "provider", "filename"],
    "score": ["score", "filename"],
    "plain": ["filename"],
    "uploader": ["score", "filename", "uploader"],
}


def normalize_badge_parts(raw: Any) -> list[str]:
    """
    Normalize a raw badge-parts payload into a validated list in canonical order.

    Accepts a list or comma-separated string; unknown tokens are dropped.
    An empty/invalid result falls back to the default (score + provider + filename +
    uploader).
    """
    if raw is None:
        return list(DEFAULT_BADGE_PARTS)

    if isinstance(raw, str):
        items = [part.strip().lower() for part in raw.split(",")]
    elif isinstance(raw, list | tuple | set):
        items = [str(part).strip().lower() for part in raw]
    else:
        return list(DEFAULT_BADGE_PARTS)

    selected = {item for item in items if item in CANONICAL_BADGE_PARTS}
    ordered = [part for part in CANONICAL_BADGE_PARTS if part in selected]
    return ordered or list(DEFAULT_BADGE_PARTS)


def badge_format_to_parts(badge_format: str | None) -> list[str]:
    """Map a legacy preset badge format name to its component list."""
    return list(_LEGACY_BADGE_FORMATS.get((badge_format or "").strip().lower(), DEFAULT_BADGE_PARTS))


class UserPreferences(BaseModel):
    """User preferences decoded from stateless URL-based configuration payload."""

    subdl_key: str = ""
    subsource_key: str = ""
    opensubtitles_key: str = ""
    #: OpenSubtitles account credentials. Used only to exchange for a user JWT,
    #: which in turn raises the *download* quota. They do not gate search or
    #: hash matching -- see app.providers.opensubtitles_auth for why.
    #:
    #: Carried in the stateless config token like the API keys. That token is
    #: base64, not encryption: anything able to read the manifest URL can
    #: recover these. That is the existing trade-off for the API keys too, but a
    #: password is a far more reusable secret than a provider key, so the UI says
    #: so plainly and the value is never echoed back or written to disk.
    opensubtitles_username: str = ""
    opensubtitles_password: str = ""
    #: Where the effective provider credentials above came from:
    #: "manifest" (supplied in this request's config), "environment",
    #: "mixed", or "default" (none configured). Recorded at the point the
    #: precedence rule is applied, because after the values are merged their
    #: origin cannot be recovered. Diagnostic only; never part of any digest.
    credential_source: str = "default"
    languages: list[str] = Field(default_factory=lambda: ["ara"])
    exclude_hi: bool = False
    nuvio_mode: bool = False  # True: Clean ISO code ("ara") & Nuvio ID; False: Stremio format ("ara | ...", "AR [★ Match] | ...")
    hi_preference: str = "neutral"  # "neutral", "prefer", or "exclude"
    # Provider enable/disable toggles (SubDL/SubSource on by default; the rest
    # stay off until the user explicitly enables them)
    enable_subdl: bool = True
    enable_subsource: bool = True
    enable_opensubtitles: bool = False
    enable_yifysubtitles: bool = False
    enable_subtitlecat: bool = False
    # Arabic RTL normalization pipeline (punctuation/brackets/quotes + RLM)
    enable_rtl_fix: bool = True
    # Remove promotional/advertisement cues (websites, telegram handles, credits)
    enable_ad_removal: bool = True
    # Preserve whitelisted translator credit lines while stripping ad links
    keep_translator_credits: bool = True
    # Independent subtitle cleaning toggles (flattened "Clean syntax & formatting").
    # Baked into the core engine (always active; no UI toggle). Legacy encodings -> UTF-8.
    fix_encoding: bool = True
    clean_tags: bool = True  # repair unclosed <i>/<b>, strip unsupported tags
    strip_colors: bool = False  # remove ASS/inline font colors
    clean_spacing: bool = True  # collapse multi-spaces, trim pre-punctuation spaces
    clean_symbols: bool = True  # '--' -> '...', drop stray <br>
    clean_commas: bool = True  # Latin commas in Arabic text -> '،'
    clean_timing: bool = True  # clamp cue overlaps < 500ms
    # Strip in-dialogue HI artifacts ([MUSIC], (SIGHS), "JOHN:" speaker labels)
    strip_hi: bool = False
    # Convert Western digits to Eastern Arabic numerals in Arabic dialogue
    eastern_arabic_numerals: bool = False
    # Strip Arabic diacritics (keeping Shadda) from Arabic dialogue
    strip_diacritics: bool = False
    # Convert ASS/SSA subtitles to color-preserved SRT for playback stability
    convert_ass_to_srt: bool = True
    # Auto-sync Arabic subtitles against a trusted reference (alass)
    auto_sync: bool = True
    # Subtitle badge components to display (order-independent; canonical order applied):
    #   "score"    -> "[100%]"
    #   "provider" -> "[SubDL]"
    #   "filename" -> "Release.Name"
    #   "uploader" -> "(by username)"
    badge_parts: list[str] = Field(default_factory=lambda: list(DEFAULT_BADGE_PARTS))

    @property
    def resolved_badge_parts(self) -> list[str]:
        """Return validated badge components in canonical display order."""
        return normalize_badge_parts(self.badge_parts)

    @property
    def badge_format(self) -> str:
        """Legacy preset name derived from the selected components (backward compatibility)."""
        parts = self.resolved_badge_parts
        for preset, mapped in _LEGACY_BADGE_FORMATS.items():
            if parts == mapped:
                return preset
        return "custom"

    @property
    def resolved_badge_format(self) -> str:
        """Backward-compatible alias for the derived legacy preset name."""
        return self.badge_format

    @property
    def opensubtitles_api_key(self) -> str:
        return self.opensubtitles_key

    def __iter__(self):
        """Allow backward-compatible tuple unpacking: subdl, subsource = parse_user_config(...)."""
        yield self.subdl_key
        yield self.subsource_key
