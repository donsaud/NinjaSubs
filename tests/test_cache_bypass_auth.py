"""Tests verifying that cache bypass endpoints require admin authorization.

Ensures anonymous `nocache`, `refresh`, `bypass_cache`, and `x-bypass-cache` cannot bypass cache.
Also verifies authorized admin cache bypass works.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_admin_token():
    original = getattr(settings, "NINJASUBS_ADMIN_TOKEN", "")
    yield
    settings.NINJASUBS_ADMIN_TOKEN = original

def test_anonymous_bypass_does_not_disable_cache():
    # Mock aggregate_subtitles to check cache usage flag
    async def mock_aggregate(*args, **kwargs):
        # Verify use_cache flag is True despite bypass request
        assert kwargs.get("use_cache") is True, "Cache was disabled for anonymous bypass"
        return []

    with patch(
        "app.services.aggregator.aggregate_subtitles",
        new=AsyncMock(side_effect=mock_aggregate),
    ):
        # Test with query parameters
        for param in ["nocache", "refresh", "bypass_cache"]:
            resp = client.get("/subtitles/movie/tt0111161.json?" + param + "=true")
            assert resp.status_code == 200
        # Test with header
        resp = client.get("/subtitles/movie/tt0111161.json", headers={"x-bypass-cache": "true"})
        assert resp.status_code == 200

def test_admin_bypass_disables_cache():
    # Setup admin token
    from app.config import settings
    settings.NINJASUBS_ADMIN_TOKEN = "admintoken123"
    async def mock_aggregate(*args, **kwargs):
        # Verify use_cache flag is False when admin bypass is provided
        assert kwargs.get("use_cache") is False, "Cache was not disabled for admin bypass"
        return []

    with patch(
        "app.services.aggregator.aggregate_subtitles",
        new=AsyncMock(side_effect=mock_aggregate),
    ):
        # Test with query parameters + admin token header
        resp = client.get(
            "/subtitles/movie/tt0111161.json?bypass_cache=true",
            headers={"X-NinjaSubs-Admin-Token": "admintoken123"},
        )
        assert resp.status_code == 200
        # Test with header only
        resp = client.get(
            "/subtitles/movie/tt0111161.json",
            headers={
                "X-NinjaSubs-Admin-Token": "admintoken123",
                "x-bypass-cache": "true",
            },
        )
        assert resp.status_code == 200

def test_admin_token_required_for_cache_clear():
    # Already covered in test_cache_clear.py, but verify integration
    settings.NINJASUBS_ADMIN_TOKEN = "admintoken123"
    resp = client.post("/cache/clear", headers={"X-NinjaSubs-Admin-Token": "admintoken123"})
    assert resp.status_code == 200
    resp = client.post("/cache/clear", headers={"X-NinjaSubs-Admin-Token": "wrong"})
    assert resp.status_code == 403
