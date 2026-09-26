"""Pytest fixtures and configuration."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from app.cache import LRUCacheManager

# Prune any oversized environment variables injected by subshell to prevent Windows 32767-char limit error on patch.dict
for k, v in list(os.environ.items()):
    if len(v) > 30000:
        del os.environ[k]


@pytest.fixture(autouse=True)
def disable_keyless_scraper_providers(monkeypatch):
    """Disable keyless scraper providers (YIFYSubtitles/SubtitleCat) during tests
    so the suite never makes real outbound HTTP requests to them."""
    from app.config import settings

    monkeypatch.setattr(settings, "ENABLE_YIFYSUBTITLES", False, raising=False)
    monkeypatch.setattr(settings, "ENABLE_SUBTITLECAT", False, raising=False)
    yield


@pytest.fixture(autouse=True)
def default_sync_strict_mode(monkeypatch):
    """Pin the reference decision policy to strict for deterministic tests.

    The developer's local ``.env`` may set ``SYNC_REQUIRE_EXACT_MATCH=false``;
    tests that exercise relaxed behavior set it explicitly themselves.
    """
    from app.config import settings

    monkeypatch.setattr(settings, "SYNC_REQUIRE_EXACT_MATCH", True, raising=False)
    yield


@pytest.fixture(autouse=True)
def reset_subdl_circuit_breaker():
    """Reset the process-wide SubDL breaker so a 429 in one test cannot
    fast-bypass SubDL in later tests."""
    from app.providers.subdl import SUBDL_BREAKER

    SUBDL_BREAKER.reset()
    yield
    SUBDL_BREAKER.reset()


@pytest.fixture(autouse=True)
def reset_in_memory_cache():
    """Reset in-memory subtitle aggregation + failure caches between tests."""
    try:
        from app.cache import cache_manager
        from app.services.cache import clear_subtitle_cache

        clear_subtitle_cache()
        cache_manager.clear_failures()
    except ImportError:
        pass
    yield
    try:
        from app.cache import cache_manager
        from app.services.cache import clear_subtitle_cache

        clear_subtitle_cache()
        cache_manager.clear_failures()
    except ImportError:
        pass


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache testing."""
    tmp = tempfile.mkdtemp(prefix="test_subs_cache_")
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def test_cache_manager(temp_cache_dir):
    """Provide an isolated LRUCacheManager instance."""
    return LRUCacheManager(
        cache_dir=str(temp_cache_dir),
        max_bytes=10000,  # small size for quick testing
        max_files=5,
    )


@pytest.fixture
def client():
    """Shared TestClient fixture for FastAPI application testing."""
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)
