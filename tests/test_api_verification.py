"""Tests for API Key real-time verification endpoints and configure badges."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_verify_subdl_missing_key(client):
    """Subdl verification without key returns valid=False."""
    resp = client.get("/api/verify/subdl")
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is False
    assert "required" in data["message"].lower()

    resp_empty = client.get("/api/verify/subdl?api_key=  ")
    assert resp_empty.status_code == 200
    assert resp_empty.json()["valid"] is False


@pytest.mark.asyncio
async def test_verify_subdl_valid_key(client):
    """Subdl verification with valid key (HTTP 200 and status: True)."""
    mock_resp = httpx.Response(
        200,
        json={"status": True, "subtitles": []},
        request=httpx.Request("GET", "https://api.subdl.com"),
    )
    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_resp)):
        resp = client.get("/api/verify/subdl?api_key=valid_token_123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True


@pytest.mark.asyncio
async def test_verify_subdl_invalid_key(client):
    """Subdl verification with invalid key (HTTP 403 or status: False)."""
    mock_resp = httpx.Response(
        403,
        json={"status": False, "error": "not_authorized", "message": "Not Authorized"},
        request=httpx.Request("GET", "https://api.subdl.com"),
    )
    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_resp)):
        resp = client.get("/api/verify/subdl?api_key=bad_token")
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False


@pytest.mark.asyncio
async def test_verify_subdl_connection_error(client):
    """Subdl verification returns valid=False on network failure."""
    with patch(
        "httpx.AsyncClient.get", new=AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
    ):
        resp = client.get("/api/verify/subdl?api_key=some_key")
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False
        assert data.get("error") == "Connection error"


def test_verify_subsource_missing_key(client):
    """Subsource verification without key returns valid=False."""
    resp = client.get("/api/verify/subsource")
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is False
    assert "required" in data["message"].lower()


@pytest.mark.asyncio
async def test_verify_subsource_valid_key(client):
    """Subsource verification with valid key (HTTP 200)."""
    mock_resp = httpx.Response(
        200,
        json={"success": True, "data": []},
        request=httpx.Request("GET", "https://api.subsource.net"),
    )
    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_resp)):
        resp = client.get("/api/verify/subsource?api_key=valid_subsource_key")
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True


@pytest.mark.asyncio
async def test_verify_subsource_valid_404_key(client):
    """Subsource verification returns valid=True on HTTP 404 (valid auth, no subs found)."""
    mock_resp = httpx.Response(
        404,
        json={"error": "Not found", "message": "No subtitles found for this query"},
        request=httpx.Request("GET", "https://api.subsource.net"),
    )
    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_resp)):
        resp = client.get("/api/verify/subsource?api_key=valid_subsource_key")
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True


@pytest.mark.asyncio
async def test_verify_subsource_invalid_key(client):
    """Subsource verification with invalid key (HTTP 401)."""
    mock_resp = httpx.Response(
        401,
        json={"error": "Invalid API key", "message": "The provided API key is invalid"},
        request=httpx.Request("GET", "https://api.subsource.net"),
    )
    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_resp)):
        resp = client.get("/api/verify/subsource?api_key=bad_subsource_key")
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False


@pytest.mark.asyncio
async def test_verify_subsource_connection_error(client):
    """Subsource verification returns valid=False on upstream timeout."""
    with patch(
        "httpx.AsyncClient.get",
        new=AsyncMock(side_effect=httpx.TimeoutException("Upstream timeout")),
    ):
        resp = client.get("/api/verify/subsource?api_key=some_key")
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False
        assert "timeout" in data.get("error_snippet", "").lower() or "error" in str(data).lower()


def test_verify_opensubtitles_missing_key(client):
    """OpenSubtitles verification without key returns valid=False."""
    resp = client.get("/api/verify/opensubtitles")
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is False
    assert "required" in data["message"].lower()


@pytest.mark.asyncio
async def test_verify_opensubtitles_valid_key(client):
    """OpenSubtitles verification with valid key (HTTP 200)."""
    mock_resp = httpx.Response(
        200,
        json={"data": [{"id": "1"}]},
        request=httpx.Request(
            "GET", "https://api.opensubtitles.com/api/v1/subtitles?imdb_id=0111161&languages=en"
        ),
    )
    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_resp)):
        resp = client.get("/api/verify/opensubtitles?api_key=valid_os_key")
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True


@pytest.mark.asyncio
async def test_verify_opensubtitles_invalid_key(client):
    """OpenSubtitles verification with invalid key (HTTP 401)."""
    mock_resp = httpx.Response(
        401,
        json={"message": "Invalid API key"},
        request=httpx.Request(
            "GET", "https://api.opensubtitles.com/api/v1/subtitles?imdb_id=0111161&languages=en"
        ),
    )
    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_resp)):
        resp = client.get("/api/verify/opensubtitles?api_key=invalid_os_key")
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False


@pytest.mark.asyncio
async def test_verify_opensubtitles_connection_error(client):
    """OpenSubtitles verification returns valid=False on connection error."""
    with patch(
        "httpx.AsyncClient.get", new=AsyncMock(side_effect=httpx.ConnectError("Connection failed"))
    ):
        resp = client.get("/api/verify/opensubtitles?api_key=some_key")
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False
        assert data.get("error") == "Connection error"


def test_configure_page_contains_badges_and_debounce(client):
    """Assert configure HTML has status badges, IDs, and debounce listeners."""
    resp = client.get("/configure")
    assert resp.status_code == 200
    html = resp.text

    # Inputs and status badges
    assert 'id="subdl-key"' in html
    assert 'id="subdl-status"' in html
    assert 'id="subsource-key"' in html
    assert 'id="subsource-status"' in html
    assert 'id="opensubtitles-key"' in html
    assert 'id="opensubtitles-status"' in html

    # API key input wrappers (with inline "Get API Key" action)
    assert "provider-key-wrap" in html
    assert "provider-key-action" in html

    # Status badge labels and debounce timer
    assert "Valid API" in html
    assert "Invalid API" in html
    assert "Checking..." in html
    assert "subdlDebounceTimer" in html
    assert "subsourceDebounceTimer" in html
    assert "/api/verify/subdl" in html
    assert "/api/verify/subsource" in html
    assert "/api/verify/opensubtitles" in html


def test_configure_page_stremio_installation_card(client):
    """Assert configure HTML has dedicated Stremio installation card and both action buttons."""
    resp = client.get("/configure")
    assert resp.status_code == 200
    html = resp.text

    # Stremio Card Header & Branding (monochrome logo, kit.betterer.cc style)
    assert "Stremio" in html
    assert "Install directly to Stremio or copy the manifest link." in html
    assert "#f2f2f2" in html

    # Buttons
    assert 'id="installStremioApp"' in html
    assert 'id="installStremioWeb"' in html
    assert "Install to Stremio" in html
    assert "Stremio Web" in html
    assert "Install directly to Stremio or copy the manifest link." in html

    # Client-side wiring
    assert "getManifestUrl()" in html
    assert "stremio://" in html
    assert "https://web.stremio.com/#/addons?addon=" in html
    assert "installAppBtn.addEventListener" in html
    assert "installWebBtn.addEventListener" in html
