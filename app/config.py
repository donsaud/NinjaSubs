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
    # How many promising candidates may be taken through alass per sync request.
    # The ranked candidate list is usually far larger, so this is what stops one
    # subtitle request from turning into dozens of subprocess runs. Configurable
    # rather than hard-coded so it can be tuned against the false-positive
    # benchmark without editing logic.
    ALASS_CANDIDATE_LIMIT: int = 3
    # --- reference selection shadow pool (measurement only) -------------- #
    # The legacy resolver stops at the first candidate that passes cue sanity,
    # so it usually offers the shadow selector a single alternative and a low
    # "disagreement" rate that reflects that, not policy agreement. This pool
    # lets the shadow selector inspect a small, diversified set instead.
    #
    # Deliberately separate from ALASS_CANDIDATE_LIMIT: populating the pool
    # never runs alass and never changes the production reference.
    REFERENCE_SHADOW_POOL_LIMIT: int = 4
    # Extra downloads permitted purely to materialize pool payloads. Defaults to
    # 0, so the default install adds no bandwidth and the telemetry reports
    # `shadow_pool_limited_by_available_payloads` rather than pretending the
    # pool is representative.
    REFERENCE_SHADOW_POOL_FETCH_LIMIT: int = 0
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

    # --- shadow audit trail (observability only) ------------------------ #
    # Disabled by default: a default install records nothing and writes no
    # files. When enabled, every synchronization decision is written to a
    # sanitized JSONL file for offline analysis. Telemetry never influences a
    # ranking decision, a threshold, or a served payload.
    SYNC_AUDIT_ENABLED: bool = False
    SYNC_AUDIT_PATH: str | None = None
    SYNC_AUDIT_MAX_RECORDS: int = 2000

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
