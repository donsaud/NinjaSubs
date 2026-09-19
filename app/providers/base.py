"""Base subtitle provider interface and abstractions."""

from abc import ABC, abstractmethod

import httpx

from app.models import SubtitleRelease


class BaseSubtitleProvider(ABC):
    """Abstract interface for upstream subtitle providers."""

    name: str

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    @abstractmethod
    async def search_subtitles(
        self,
        imdb_id: str,
        is_series: bool = False,
        season: int | None = None,
        episode: int | None = None,
        title: str | None = None,
        year: int | None = None,
        api_key: str | None = None,
        languages: list[str] | None = None,
        exclude_hi: bool = False,
        *,
        types: list[str] | None = None,
        **kwargs,
    ) -> list[SubtitleRelease]:
        """Search upstream for subtitles."""
        pass

    @abstractmethod
    async def download_archive(
        self,
        download_ref: str,
        api_key: str | None = None,
    ) -> bytes | None:
        """Download subtitle ZIP or SRT file from upstream."""
        pass
