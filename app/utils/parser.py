"""Parser for Stremio media IDs (movies and TV series)."""

import re

from app.models import ParsedMediaID


def parse_stremio_id(raw_id: str) -> ParsedMediaID:
    """
    Parse Stremio media ID into IMDb ID and optional season/episode.

    Supported patterns:
      - Movie: "tt1234567" or "tt1234567.json"
      - Series: "tt1234567:1:5" or "tt1234567:01:05" or "tt1234567:1:5.json"
      - Series with extra params: "tt1234567:1:5/videoHash=..."

    Raises:
        ValueError: If ID does not contain a valid IMDb tt pattern.
    """
    if not raw_id:
        raise ValueError("Media ID cannot be empty.")

    # Strip .json extension if attached to ID string
    cleaned = raw_id
    if cleaned.endswith(".json"):
        cleaned = cleaned[:-5]

    # If ID contains extra path segments (e.g. extra/videoHash=...), take first part
    if "/" in cleaned:
        cleaned = cleaned.split("/", 1)[0]

    # Support Kitsu anime ID format (e.g. kitsu:12345 or kitsu:12345:1)
    if cleaned.startswith("kitsu:"):
        parts = cleaned.split(":")
        kitsu_id = f"{parts[0]}:{parts[1]}"
        if len(parts) >= 3:
            try:
                episode = int(parts[2])
                return ParsedMediaID(
                    raw_id=raw_id,
                    imdb_id=kitsu_id,
                    is_series=True,
                    season=1,
                    episode=episode,
                )
            except ValueError:
                pass
        return ParsedMediaID(
            raw_id=raw_id,
            imdb_id=kitsu_id,
            is_series=False,
            season=None,
            episode=None,
        )

    # Split by colon
    parts = cleaned.split(":")
    imdb_id = parts[0].strip()

    # Validate IMDb ID format (e.g., tt0111161)
    if not re.match(r"^tt\d+$", imdb_id):
        raise ValueError(f"Invalid IMDb ID format: '{imdb_id}'. Expected 'tt' followed by digits.")

    if len(parts) >= 3:
        # TV series format: tt1234567:season:episode
        try:
            season = int(parts[1])
            episode = int(parts[2])
            return ParsedMediaID(
                raw_id=raw_id,
                imdb_id=imdb_id,
                is_series=True,
                season=season,
                episode=episode,
            )
        except ValueError as e:
            raise ValueError(f"Invalid season/episode integers in series ID: '{raw_id}'") from e
    elif len(parts) == 2:
        # Anime/single-season episode format: tt1234567:episode
        try:
            episode = int(parts[1])
            return ParsedMediaID(
                raw_id=raw_id,
                imdb_id=imdb_id,
                is_series=True,
                season=1,
                episode=episode,
            )
        except ValueError:
            pass

    # Single movie format: tt1234567
    return ParsedMediaID(
        raw_id=raw_id,
        imdb_id=imdb_id,
        is_series=False,
        season=None,
        episode=None,
    )
