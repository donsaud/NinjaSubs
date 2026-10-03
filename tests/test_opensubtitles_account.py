"""OpenSubtitles account credentials: login, caching, and credential hygiene.

Three properties matter and are easy to lose:

* **Login is rate-limited** (1/s, 10/min, 30/hour), so a token must be cached and
  concurrent callers must share one login rather than each spending a slot.
* **Search does not need the token.** Hash matching already worked with an API
  key alone, so authentication must never become a precondition for it.
* **The password must not outlive the request.** The config token is base64, so
  the password arrives in cleartext; it is also the credential a user is most
  likely to have reused, which is why it is kept out of logs and off disk.
"""

from __future__ import annotations

import asyncio

import pytest

from app.cache import sanitize_metadata
from app.config import settings
from app.providers.opensubtitles import OpenSubtitlesProvider
from app.providers.opensubtitles_auth import (
    OPENSUBTITLES_TOKENS,
    TOKEN_TTL_SECONDS,
    OpenSubtitlesToken,
    OpenSubtitlesTokenCache,
    _mask,
    credential_digest,
)
from app.utils.config_parser import encode_user_config, parse_user_config


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


class RecordingClient:
    """Counts login calls and replays a scripted response."""

    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.login_calls = 0
        self.bodies: list[dict] = []

    async def post(self, url: str, headers=None, json=None, **kwargs) -> FakeResponse:
        if url.endswith("/login"):
            self.login_calls += 1
            self.bodies.append(dict(json or {}))
        return self.response


@pytest.fixture(autouse=True)
def _clear_shared_cache():
    OPENSUBTITLES_TOKENS._tokens.clear()
    OPENSUBTITLES_TOKENS._locks.clear()
    yield
    OPENSUBTITLES_TOKENS._tokens.clear()
    OPENSUBTITLES_TOKENS._locks.clear()


# --- login and caching --------------------------------------------------------


@pytest.mark.asyncio
async def test_a_valid_login_yields_a_token():
    cache = OpenSubtitlesTokenCache()
    client = RecordingClient(
        FakeResponse(200, {"token": "jwt-abc", "base_url": "api.opensubtitles.com"})
    )

    token = await cache.get_token(
        client,
        api_key="KEY",
        username="me@example.com",
        password="pw",
        base_url="https://api.opensubtitles.com/api/v1",
        user_agent="test",
    )

    assert token is not None
    assert token.token == "jwt-abc"
    assert token.base_url == "https://api.opensubtitles.com/api/v1"
    assert client.login_calls == 1


@pytest.mark.asyncio
async def test_the_token_is_reused_instead_of_logging_in_again():
    """/login allows 30 calls an hour; a second search must not spend one."""
    cache = OpenSubtitlesTokenCache()
    client = RecordingClient(FakeResponse(200, {"token": "jwt"}))

    for _ in range(5):
        await cache.get_token(
            client,
            api_key="KEY",
            username="me@example.com",
            password="pw",
            base_url="https://api.opensubtitles.com/api/v1",
            user_agent="test",
        )

    assert client.login_calls == 1


@pytest.mark.asyncio
async def test_concurrent_callers_share_one_login():
    """Single-flight. Without it a burst of subtitle requests each logs in,
    because at cold start no token exists yet."""
    cache = OpenSubtitlesTokenCache()
    client = RecordingClient(FakeResponse(200, {"token": "jwt"}))

    results = await asyncio.gather(
        *(
            cache.get_token(
                client,
                api_key="KEY",
                username="me@example.com",
                password="pw",
                base_url="https://api.opensubtitles.com/api/v1",
                user_agent="test",
            )
            for _ in range(8)
        )
    )

    assert client.login_calls == 1
    assert all(r is not None and r.token == "jwt" for r in results)


@pytest.mark.asyncio
async def test_a_rejected_login_is_not_retried():
    """A 401 must not loop: OpenSubtitles counts wrong credentials against the
    30/hour budget, so retrying makes the situation worse."""
    cache = OpenSubtitlesTokenCache()
    client = RecordingClient(FakeResponse(401, {"message": "invalid"}))

    for _ in range(3):
        token = await cache.get_token(
            client,
            api_key="KEY",
            username="me@example.com",
            password="wrong",
            base_url="https://api.opensubtitles.com/api/v1",
            user_agent="test",
        )
        assert token is None

    assert client.login_calls == 3  # one per explicit call, never retried internally


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "username,password,api_key",
    [("", "pw", "KEY"), ("me", "", "KEY"), ("me", "pw", "")],
)
async def test_incomplete_credentials_never_attempt_a_login(username, password, api_key):
    cache = OpenSubtitlesTokenCache()
    client = RecordingClient(FakeResponse(200, {"token": "jwt"}))

    token = await cache.get_token(
        client,
        api_key=api_key,
        username=username,
        password=password,
        base_url="https://api.opensubtitles.com/api/v1",
        user_agent="test",
    )

    assert token is None
    assert client.login_calls == 0


@pytest.mark.asyncio
async def test_a_response_without_a_token_is_refused():
    cache = OpenSubtitlesTokenCache()
    client = RecordingClient(FakeResponse(200, {"base_url": "api.opensubtitles.com"}))

    token = await cache.get_token(
        client,
        api_key="KEY",
        username="me@example.com",
        password="pw",
        base_url="https://api.opensubtitles.com/api/v1",
        user_agent="test",
    )

    assert token is None


@pytest.mark.asyncio
async def test_a_hostile_base_url_is_ignored():
    """An unexpected base_url must never become a request target."""
    cache = OpenSubtitlesTokenCache()
    client = RecordingClient(
        FakeResponse(200, {"token": "jwt", "base_url": "file:///etc/passwd"})
    )

    token = await cache.get_token(
        client,
        api_key="KEY",
        username="me@example.com",
        password="pw",
        base_url="https://api.opensubtitles.com/api/v1",
        user_agent="test",
    )

    assert token is not None
    assert token.base_url == "https://api.opensubtitles.com/api/v1"


@pytest.mark.asyncio
async def test_a_vip_base_url_is_honoured():
    """VIP accounts get a different host that requests must continue against."""
    cache = OpenSubtitlesTokenCache()
    client = RecordingClient(
        FakeResponse(200, {"token": "jwt", "base_url": "vip-api.opensubtitles.com"})
    )

    token = await cache.get_token(
        client,
        api_key="KEY",
        username="me@example.com",
        password="pw",
        base_url="https://api.opensubtitles.com/api/v1",
        user_agent="test",
    )

    assert token is not None
    assert token.base_url == "https://vip-api.opensubtitles.com/api/v1"


@pytest.mark.asyncio
async def test_invalidate_forces_a_fresh_login():
    cache = OpenSubtitlesTokenCache()
    client = RecordingClient(FakeResponse(200, {"token": "jwt"}))

    await cache.get_token(
        client, api_key="K", username="u", password="p",
        base_url="https://x", user_agent="t",
    )
    cache.invalidate("K", "u", "p")
    await cache.get_token(
        client, api_key="K", username="u", password="p",
        base_url="https://x", user_agent="t",
    )

    assert client.login_calls == 2


def test_the_ttl_leaves_margin_on_the_24_hour_token():
    assert TOKEN_TTL_SECONDS < 24 * 3600
    assert TOKEN_TTL_SECONDS >= 12 * 3600


# --- credential hygiene -------------------------------------------------------


def test_the_digest_does_not_contain_the_password():
    digest = credential_digest("KEY", "me@example.com", "hunter2")

    assert "hunter2" not in digest
    assert "me@example.com" not in digest


def test_the_digest_changes_when_the_password_changes():
    assert credential_digest("K", "u", "a") != credential_digest("K", "u", "b")


def test_the_password_never_reaches_disk_metadata():
    safe, changed = sanitize_metadata(
        {
            "sub_id": "x",
            "opensubtitles_username": "me@example.com",
            "opensubtitles_password": "hunter2",
        }
    )

    assert changed is True
    assert "hunter2" not in str(safe)
    assert safe["has_opensubtitles_password"] is True
    # The username is not a secret and stays, so diagnostics can still show which
    # account a result came from.
    assert safe["opensubtitles_username"] == "me@example.com"


def test_masking_hides_a_username_but_keeps_it_correlatable():
    masked = _mask("someone@example.com")

    assert masked.startswith("s")
    assert masked.endswith("m")
    assert "someone" not in masked


def test_masking_short_and_empty_usernames():
    assert _mask("") == "<empty>"
    assert _mask("ab") == "**"


# --- config round trip --------------------------------------------------------


def test_credentials_survive_the_config_token():
    token = encode_user_config(
        opensubtitles_key="KEY",
        opensubtitles_username="me@example.com",
        opensubtitles_password="hunter2",
        enable_opensubtitles=True,
    )
    prefs = parse_user_config(token)

    assert prefs.opensubtitles_key == "KEY"
    assert prefs.opensubtitles_username == "me@example.com"
    assert prefs.opensubtitles_password == "hunter2"
    assert prefs.credential_source == "manifest"


def test_a_half_pair_is_not_emitted():
    """Half a login can only fail, and each attempt burns the hourly login budget."""
    token = encode_user_config(
        opensubtitles_key="KEY", opensubtitles_username="me@example.com"
    )
    prefs = parse_user_config(token)

    assert prefs.opensubtitles_username == ""
    assert prefs.opensubtitles_password == ""


def test_credentials_decode_from_the_query_string_branch():
    prefs = parse_user_config(
        "opensubtitles_username=me%40example.com&opensubtitles_password=hunter2"
    )

    assert prefs.opensubtitles_username == "me@example.com"
    assert prefs.opensubtitles_password == "hunter2"


def test_a_password_is_not_trimmed_by_the_decoder():
    prefs = parse_user_config(
        "opensubtitles_username=me&opensubtitles_password=" + "%20pw%20"
    )

    assert prefs.opensubtitles_password == " pw "


def test_the_environment_supplies_credentials_when_the_user_omits_them(
    monkeypatch,
):
    monkeypatch.setattr(settings, "OPENSUBTITLES_USERNAME", "envuser", raising=False)
    monkeypatch.setattr(settings, "OPENSUBTITLES_PASSWORD", "envpw", raising=False)

    prefs = parse_user_config(None)

    assert prefs.opensubtitles_username == "envuser"
    assert prefs.opensubtitles_password == "envpw"
    assert prefs.credential_source == "environment"


def test_a_user_password_wins_over_the_environment(monkeypatch):
    monkeypatch.setattr(settings, "OPENSUBTITLES_USERNAME", "envuser", raising=False)
    monkeypatch.setattr(settings, "OPENSUBTITLES_PASSWORD", "envpw", raising=False)

    prefs = parse_user_config(
        encode_user_config(opensubtitles_username="me", opensubtitles_password="pw")
    )

    assert prefs.opensubtitles_username == "me"
    assert prefs.opensubtitles_password == "pw"
    assert prefs.credential_source == "mixed"


# --- provider wiring ----------------------------------------------------------


def test_provider_credentials_fall_back_to_the_environment(monkeypatch):
    monkeypatch.setattr(settings, "OPENSUBTITLES_API_KEY", "envkey", raising=False)
    monkeypatch.setattr(settings, "OPENSUBTITLES_USERNAME", "envuser", raising=False)
    monkeypatch.setattr(settings, "OPENSUBTITLES_PASSWORD", "envpw", raising=False)
    provider = OpenSubtitlesProvider(RecordingClient(FakeResponse(200, {})))

    key, user, password = provider._resolve_credentials(None)

    assert (key, user, password) == ("envkey", "envuser", "envpw")


def test_per_request_credentials_win():
    provider = OpenSubtitlesProvider(RecordingClient(FakeResponse(200, {})))

    key, user, password = provider._resolve_credentials("K", "me", "pw")

    assert (key, user, password) == ("K", "me", "pw")


def test_headers_carry_the_api_key_and_optional_token():
    provider = OpenSubtitlesProvider(RecordingClient(FakeResponse(200, {})))

    assert "Authorization" not in provider._get_headers("KEY")
    assert provider._get_headers("KEY", "jwt")["Authorization"] == "jwt"
    assert provider._get_headers("KEY", "jwt")["Api-Key"] == "KEY"


@pytest.mark.asyncio
async def test_search_works_without_any_account_credentials():
    """The premise this feature had to respect: hash matching never needed a login."""
    captured: dict = {}

    class Client:
        async def get(self, url, params=None, headers=None, **kwargs):
            captured["url"] = url
            captured["params"] = params
            captured["headers"] = headers
            return FakeResponse(200, {"data": []})

    provider = OpenSubtitlesProvider(Client())
    provider._get_headers = lambda api_key, token=None: {
        "Api-Key": api_key,
        **({"Authorization": token} if token else {}),
    }

    releases = await provider.search_subtitles(
        imdb_id="tt2582802", api_key="KEY", moviehash="8e245d9679d31e12", moviebytesize=19571049411
    )

    assert releases == []
    assert captured["params"]["moviehash"] == "8e245d9679d31e12"
    assert captured["params"]["moviebytesize"] == "19571049411"
    assert "Authorization" not in captured["headers"]


@pytest.mark.asyncio
async def test_search_sends_the_token_when_an_account_is_configured():
    OPENSUBTITLES_TOKENS._tokens[credential_digest("KEY", "me@example.com", "pw")] = (
        OpenSubtitlesToken(
            token="jwt-123",
            base_url="https://api.opensubtitles.com/api/v1",
            expires_at=9_999_999_999.0,
        )
    )
    captured: dict = {}

    class Client:
        async def get(self, url, params=None, headers=None, **kwargs):
            captured["headers"] = headers
            return FakeResponse(200, {"data": []})

    provider = OpenSubtitlesProvider(Client())

    await provider.search_subtitles(imdb_id="tt1", api_key="KEY", username="me@example.com", password="pw")

    assert captured["headers"]["Authorization"] == "jwt-123"
