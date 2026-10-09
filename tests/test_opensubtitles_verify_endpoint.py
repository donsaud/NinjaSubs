"""Key validation ignores legacy account fields and never attempts login."""

from unittest.mock import AsyncMock

import httpx
import pytest


@pytest.mark.parametrize("method", ["get", "post"])
@pytest.mark.parametrize("status", [200, 401, 403, 429])
def test_validation_is_key_only(client, monkeypatch, method, status):
    upstream = AsyncMock(spec=httpx.AsyncClient)
    upstream.__aenter__.return_value = upstream
    upstream.get.return_value = httpx.Response(status, json={"data": []})
    monkeypatch.setattr("app.main.httpx.AsyncClient", lambda **kw: upstream)
    if method == "post":
        response = client.post("/api/verify/opensubtitles", json={
            "api_key": "example-key", "username": "ignored", "password": "ignored"})
    else:
        response = client.get("/api/verify/opensubtitles", params={"api_key": "example-key"})
    assert response.json()["valid"] is (status == 200)
    upstream.get.assert_awaited_once()
    assert upstream.get.call_args.kwargs["headers"]["Api-Key"] == "example-key"
    assert "Authorization" not in upstream.get.call_args.kwargs["headers"]
    upstream.post.assert_not_awaited()
    assert "account_checked" not in response.json()


def test_missing_key_validation_never_opens_a_client(client, monkeypatch):
    def forbidden(**kw):
        raise AssertionError("no network client without an API key")
    monkeypatch.setattr("app.main.httpx.AsyncClient", forbidden)
    assert client.post("/api/verify/opensubtitles", json={}).json()["valid"] is False
