"""Pytest fixtures and configuration."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Redirect the on-disk cache BEFORE any application module is imported.
#
# ``app.config`` resolves CACHE_DIR at import time and ``LRUCacheManager``
# captures it in ``__init__``, so a fixture that patches ``settings`` later is
# too late for any singleton built during import. Setting the environment
# variable here is the only point early enough to be reliable.
#
# The default is ``<repo>/subs_cache`` - the same directory a local run of the
# app uses, inside the repository. Test output written there is
# indistinguishable from real operational state.
# ---------------------------------------------------------------------------
_TESTS_ROOT = Path(__file__).resolve().parent
_TEST_CACHE_DIR = _TESTS_ROOT / "cache"
os.environ["CACHE_DIR"] = str(_TEST_CACHE_DIR)
os.environ.setdefault("REFERENCE_CACHE_DIR", str(_TEST_CACHE_DIR / "references"))

from app.cache import LRUCacheManager  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def isolated_sync_cache():
    """Clean the test-owned cache once, before and after the session.

    Scoped strictly to ``tests/cache``. A real cache is never deleted: this
    fixture will not touch the production path even if CACHE_DIR is overridden.
    """
    if _TEST_CACHE_DIR.exists():
        shutil.rmtree(_TEST_CACHE_DIR, ignore_errors=True)
    _TEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    yield _TEST_CACHE_DIR
    shutil.rmtree(_TEST_CACHE_DIR, ignore_errors=True)


@pytest.fixture(autouse=True)
def settings_cache_points_at_tests():
    """Keep the live settings object pointed at the test cache.

    The environment variable covers modules imported before this ran; this
    covers objects constructed during a test, which read ``settings`` live.
    """
    from app.config import settings

    previous_cache = settings.CACHE_DIR
    previous_reference = settings.REFERENCE_CACHE_DIR
    settings.CACHE_DIR = str(_TEST_CACHE_DIR)
    settings.REFERENCE_CACHE_DIR = str(_TEST_CACHE_DIR / "references")
    try:
        yield
    finally:
        settings.CACHE_DIR = previous_cache
        settings.REFERENCE_CACHE_DIR = previous_reference


@pytest.fixture(autouse=True)
def reset_request_context():
    """No sync log line may inherit a request id from another test."""
    from app.logging_context import current_request_id, set_request_id

    token = set_request_id(None)
    try:
        yield
    finally:
        from app.logging_context import reset_request_id

        reset_request_id(token)
        assert current_request_id() is None


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
