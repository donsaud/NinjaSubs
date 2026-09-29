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

    # Subtitle auto-synchronization (alass) & result cache
    ENABLE_SUBTITLE_SYNC: bool = True
    ALASS_PATH: str = "alass"
    ALASS_TIMEOUT_SECONDS: float = 10.0
    ALASS_MAX_CONCURRENT_SYNCS: int = 1
    REDIS_URL: str | None = None
    # Hard inline budget for sync during a player request. Downloads from the
    # reference providers + alass routinely take ~10s, so the budget is generous;
    # past it we serve the original and let the (still-running) sync warm the
    # cache for the next request.
    SYNC_TOTAL_REQUEST_BUDGET: float = 15.0

    # Persistent English-reference disk cache (30 days) for the sync pipeline.
    REFERENCE_CACHE_DIR: str | None = None
    REFERENCE_CACHE_TTL_SECONDS: float = 2592000.0  # 30 days

    # Legacy reference strictness flag (kept for configuration compatibility;
    # the sync pipeline now accepts any season/episode-matched reference).
    SYNC_REQUIRE_EXACT_MATCH: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

# Activate secret redaction for every log record as early as possible: all
# providers import this module, so outbound ``httpx`` URL logs get scrubbed.
from app.utils.log_redaction import install_log_redaction  # noqa: E402

install_log_redaction()
