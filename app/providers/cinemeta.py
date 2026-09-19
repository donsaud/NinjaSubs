"""Cinemeta metadata client for resolving IMDb IDs to titles and release years."""

import logging
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# Lightweight in-memory LRU-like cache for metadata resolution
_CINEMETA_CACHE: dict[str, dict[str, Any]] = {}


class CinemetaClient:
    """Helper client to resolve Stremio IMDb IDs via official Cinemeta API."""

    BASE_URL = "https://v3-cinemeta.strem.io"

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def get_metadata(self, media_type: str, imdb_id: str) -> dict[str, Any] | None:
        """
        Fetch title and year for a movie or TV show.
        Returns dict with keys: 'title', 'year', 'type'.
        """
        cache_key = f"{media_type}:{imdb_id}"
        if cache_key in _CINEMETA_CACHE:
            return _CINEMETA_CACHE[cache_key]

        target_type = "series" if media_type in ("series", "tv") else "movie"
        url = f"{self.BASE_URL}/meta/{target_type}/{imdb_id}.json"

        try:
            resp = await self.client.get(
                url,
                timeout=settings.UPSTREAM_TIMEOUT,
                headers={"User-Agent": "StremioArabicSubs/1.0.0"},
            )
            if resp.status_code != 200:
                return None

            data = resp.json()
            meta = data.get("meta")
            if not meta:
                return None

            title = meta.get("name")
            raw_year = meta.get("year")
            year = None
            if raw_year:
                try:
                    # Year might be "2022" or "2018-2023"
                    year_str = str(raw_year).split("-")[0].strip()
                    year = int(year_str)
                except ValueError:
                    year = None

            result = {
                "title": title,
                "year": year,
                "type": target_type,
            }
            # Limit cache size to 1000 items
            if len(_CINEMETA_CACHE) > 1000:
                _CINEMETA_CACHE.pop(next(iter(_CINEMETA_CACHE)))
            _CINEMETA_CACHE[cache_key] = result
            return result
        except Exception as e:
            logger.debug(f"Cinemeta lookup failed for {imdb_id}: {e}")
            return None
