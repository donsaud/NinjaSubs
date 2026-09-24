"""Tests for administrative cache clearing endpoint.

Verifies:
- GET /cache/clear returns 405 (Method Not Allowed)
- POST /cache/clear without admin token returns 404/403
- POST /cache/clear with missing token returns 404/403
- POST /cache/clear with wrong token returns 403
- POST /cache/clear with correct token clears caches
"""

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


@pytest.fixture(autouse=True)
def reset_settings():
    # Ensure admin token is cleared for tests
    original_token = getattr(settings, "NINJASUBS_ADMIN_TOKEN", "")
    yield
    settings.NINJASUBS_ADMIN_TOKEN = original_token

client = TestClient(app)

def test_get_cache_clear_returns_405():
    response = client.get("/cache/clear")
    assert response.status_code == 405

def test_post_without_token_returns_404():
    settings.NINJASUBS_ADMIN_TOKEN = ""
    response = client.post("/cache/clear")
    assert response.status_code == 404

def test_post_with_missing_header_returns_403():
    settings.NINJASUBS_ADMIN_TOKEN = "testtoken123"
    response = client.post("/cache/clear")
    assert response.status_code == 403

def test_post_with_wrong_token_returns_403():
    settings.NINJASUBS_ADMIN_TOKEN = "testtoken123"
    response = client.post("/cache/clear", headers={"X-NinjaSubs-Admin-Token": "wrong"})
    assert response.status_code == 403

def test_post_with_valid_token_clears_cache():
    # Setup non-empty cache state
    import asyncio

    from app.cache import cache_manager
    from app.services.cache import SUBTITLE_CACHE
    from app.services.credentials import credential_store
    # Store a dummy metadata file
    cache_manager.store_metadata("test_id", {"sub_id": "test_id"})
    meta_path = cache_manager.get_meta_path("test_id")
    assert meta_path.exists()
    # Populate credential store
    asyncio.run(credential_store.store("test_id", {"subdl_key": "x"}))
    assert credential_store.size() > 0
    # Clear via admin endpoint
    settings.NINJASUBS_ADMIN_TOKEN = "validtoken123"
    response = client.post("/cache/clear", headers={"X-NinjaSubs-Admin-Token": "validtoken123"})
    assert response.status_code == 200
    # Verify cache cleared
    assert len(SUBTITLE_CACHE) == 0
    assert not meta_path.exists()
    # Verify credential store cleared
    assert credential_store.size() == 0

