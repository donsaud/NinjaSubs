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
def reset_in_memory_cache():
    """Reset in-memory subtitle aggregation cache between tests."""
    try:
        from app.services.cache import clear_subtitle_cache

        clear_subtitle_cache()
    except ImportError:
        pass
    yield
    try:
        from app.services.cache import clear_subtitle_cache

        clear_subtitle_cache()
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
