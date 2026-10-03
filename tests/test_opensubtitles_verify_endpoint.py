"""The OpenSubtitles verify endpoint must not carry a password in a URL.

Query strings end up in the server access log, browser history, and any
intermediate proxy log. A password is the credential here a user is most likely
to have reused elsewhere, so the account check is a POST with a JSON body and
the GET route takes the API key only.
"""

from __future__ import annotations

import inspect

import httpx
import pytest

from app.main import verify_opensubtitles_key


class _Awaitable:
    """Minimal stand-in so a canned httpx.Response can be returned from a
    patched async method without needing an event loop per call."""

    def __init__(self, value: httpx.Response) -> None:
        self._value = value

    def __await__(self):
        async def _inner():
            return self._value

        return _inner().__await__()


def _search_ok() -> httpx.Response:
    return httpx.Response(
        200,
        json={"data": []},
        request=httpx.Request("GET", "https://api.opensubtitles.com/api/v1/subtitles"),
    )


def _login(status: int) -> httpx.Response:
    return httpx.Response(
        status,
        json={"token": "jwt", "base_url": "api.opensubtitles.com"},
        request=httpx.Request("POST", "https://api.opensubtitles.com/api/v1/login"),
    )


def _patch(monkeypatch, login_status: int | None = None) -> None:
    search = _search_ok()
    monkeypatch.setattr(
        httpx.AsyncClient, "get", lambda self, *a, **k: _Awaitable(search)
    )
    if login_status is not None:
        login = _login(login_status)
        monkeypatch.setattr(
            httpx.AsyncClient, "post", lambda self, *a, **k: _Awaitable(login)
        )


# --- structural guarantee -----------------------------------------------------


def test_the_get_route_takes_no_account_credentials():
    """The password is structurally unable to travel in a query string.

    Asserted on the signature rather than on behaviour, because a future
    parameter added back would otherwise pass every behavioural test while
    quietly reintroducing the leak.
    """
    assert set(inspect.signature(verify_opensubtitles_key).parameters) == {"api_key"}


def test_the_configure_page_posts_the_account_in_a_body(client):
    html = client.get("/configure").text

    assert 'fetch("/api/verify/opensubtitles", {' in html
    assert 'method: "POST"' in html
    assert "Content-Type\": \"application/json" in html
    # The password must never be assembled into a URL on the page.
    assert "password=${encodeURIComponent" not in html


# --- behaviour ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_valid_key_and_account(client, monkeypatch):
    _patch(monkeypatch, login_status=200)

    resp = client.post(
        "/api/verify/opensubtitles",
        json={"api_key": "K", "username": "me@example.com", "password": "pw"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is True
    assert data["account_checked"] is True
    assert data["account_valid"] is True


@pytest.mark.asyncio
async def test_a_rejected_account_does_not_invalidate_a_working_key(client, monkeypatch):
    """Two different failures, and the user needs them told apart.

    A flat "invalid" would push someone to delete an API key that is perfectly
    good, in order to fix an account problem the key has nothing to do with.
    """
    _patch(monkeypatch, login_status=401)

    resp = client.post(
        "/api/verify/opensubtitles",
        json={"api_key": "K", "username": "me@example.com", "password": "wrong"},
    )

    data = resp.json()
    assert data["valid"] is True
    assert data["account_checked"] is True
    assert data["account_valid"] is False
    assert data["account_status"] == 401
    assert "login was rejected" in data["message"]


@pytest.mark.asyncio
async def test_a_key_only_post_skips_the_account_check(client, monkeypatch):
    _patch(monkeypatch)

    resp = client.post("/api/verify/opensubtitles", json={"api_key": "K"})

    assert resp.json() == {"valid": True, "account_checked": False}


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [{"api_key": "K", "username": "me"}, {"api_key": "K", "password": "pw"}])
async def test_a_half_pair_skips_the_account_check(client, monkeypatch, body):
    """Half a login can only fail, and each attempt burns OpenSubtitles' 30/hour
    login budget."""
    _patch(monkeypatch)

    resp = client.post("/api/verify/opensubtitles", json=body)

    assert resp.json()["account_checked"] is False


@pytest.mark.asyncio
async def test_an_invalid_key_fails_before_any_login(client, monkeypatch):
    bad = httpx.Response(
        401,
        json={"message": "Invalid API key"},
        request=httpx.Request("GET", "https://api.opensubtitles.com/api/v1/subtitles"),
    )
    monkeypatch.setattr(httpx.AsyncClient, "get", lambda self, *a, **k: _Awaitable(bad))

    def explode(self, *a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("login attempted despite an invalid key")

    monkeypatch.setattr(httpx.AsyncClient, "post", explode)

    resp = client.post(
        "/api/verify/opensubtitles",
        json={"api_key": "bad", "username": "me", "password": "pw"},
    )

    assert resp.json()["valid"] is False


def test_a_post_without_a_key_is_rejected(client):
    resp = client.post("/api/verify/opensubtitles", json={})

    assert resp.status_code == 200
    assert resp.json()["valid"] is False


def test_a_malformed_body_is_rejected_rather_than_crashing(client):
    resp = client.post(
        "/api/verify/opensubtitles",
        content=b"not json",
        headers={"Content-Type": "application/json"},
    )

    assert resp.status_code in (200, 422)


@pytest.mark.asyncio
async def test_the_get_route_still_validates_a_key(client, monkeypatch):
    """Unchanged behaviour for the key-only path."""
    _patch(monkeypatch)

    resp = client.get("/api/verify/opensubtitles?api_key=K")

    assert resp.status_code == 200
    assert resp.json()["valid"] is True


@pytest.mark.asyncio
async def test_account_credentials_on_the_get_route_are_ignored(client, monkeypatch):
    """Even if someone hand-crafts the URL, the password is not read."""
    _patch(monkeypatch)

    resp = client.get(
        "/api/verify/opensubtitles?api_key=K&username=me&password=hunter2"
    )

    assert resp.json() == {"valid": True, "account_checked": False}
