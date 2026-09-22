"""Subtitle providers package."""

from app.providers.base import BaseSubtitleProvider
from app.providers.cinemeta import CinemetaClient
from app.providers.opensubtitles import OpenSubtitlesProvider
from app.providers.subdl import SubdlProvider
from app.providers.subsource import SubsourceProvider, SubSourceService
from app.providers.subtitlecat import SubtitlecatProvider
from app.providers.yifysubtitles import YifysubtitlesProvider

__all__ = [
    "BaseSubtitleProvider",
    "SubdlProvider",
    "SubsourceProvider",
    "SubSourceService",
    "OpenSubtitlesProvider",
    "SubtitlecatProvider",
    "YifysubtitlesProvider",
    "CinemetaClient",
]
