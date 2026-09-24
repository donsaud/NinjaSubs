"""Configuration settings for Stremio Arabic Subtitles Addon."""

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Microservice environment and performance settings."""

    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 7000
    BASE_URL: str | None = None  # e.g., http://192.168.1.50:7000
    ADDON_BASE_URL: str | None = None
    HOST_IP: str | None = None
    LAN_IP: str | None = None

    # Upstream Provider API Keys
    SUBDL_API_KEY: str = ""
    SUBSOURCE_API_KEY: str = ""
    OPENSUBTITLES_API_KEY: str = ""

    # Keyless scraper providers (no API key required)
    ENABLE_YIFYSUBTITLES: bool = True
    ENABLE_SUBTITLECAT: bool = True

    # Disk Cache limits
    CACHE_DIR: str = os.getenv("CACHE_DIR", str(Path.cwd() / "subs_cache"))
    CACHE_MAX_BYTES: int = 1 * 1024 * 1024 * 1024  # 1 GB
    CACHE_MAX_FILES: int = 500

    # Networking & Resource Constraints (<100MB RAM, strict 6s timeout)
    UPSTREAM_TIMEOUT: float = 6.0
    MAX_KEEP_ALIVE_CONNECTIONS: int = 20
    MAX_CONNECTIONS: int = 50

    # Diagnostic / Observability
    NINJASUBS_DEBUG_RANKING: bool = False

    # Administrative secret used to protect privileged endpoints such as
    # ``/cache/clear`` and authorized cache bypass.  When empty/None these
    # privileged operations are *disabled* rather than publicly open.
    # Loaded from the NINJASUBS_ADMIN_TOKEN environment variable.
    NINJASUBS_ADMIN_TOKEN: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
