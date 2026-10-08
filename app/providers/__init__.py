"""Subtitle providers package."""

from app.providers.base import BaseSubtitleProvider
from app.providers.cinemeta import CinemetaClient
from app.providers.opensubtitles import OpenSubtitlesProvider
from app.providers.subdl import SubdlProvider
from app.providers.subsource import SubsourceProvider, SubSourceService

__all__ = [
    "BaseSubtitleProvider",
    "SubdlProvider",
    "SubsourceProvider",
    "SubSourceService",
    "OpenSubtitlesProvider",
    "CinemetaClient",
]

# Retired names are recognized only to reject historical cached entries.
RETIRED_PROVIDERS = frozenset({"gestdown", "isubtitles", "yifysubtitles", "subtitlecat"})
