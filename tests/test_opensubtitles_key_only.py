"""API-key-only transport, legacy config and actual Stremio delivery."""

import base64
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from app.config import settings
from app.models import UserPreferences
from app.providers.opensubtitles import OPENSUBTITLES_BREAKER, OpenSubtitlesProvider
from app.utils.config_parser import encode_user_config, parse_user_config

SRT = b"1\n00:00:01,000 --> 00:00:03,000\nSubtitle delivered successfully\n"


@pytest.mark.parametrize("encoding", ["token", "query"])
def test_legacy_account_config_is_ignored_and_not_re_emitted(encoding):
    legacy = {"opensubtitles_key": "example-key", "opensubtitles_username": "ignored-user",
              "opensubtitles_password": "ignored-password"}
    config = (base64.urlsafe_b64encode(json.dumps(legacy).encode()).decode()
              if encoding == "token" else "&".join(f"{k}={v}" for k, v in legacy.items()))
    prefs = parse_user_config(config)
    assert prefs.opensubtitles_key == "example-key"
    assert not hasattr(prefs, "opensubtitles_username")
    assert not hasattr(prefs, "opensubtitles_password")
    token = encode_user_config(opensubtitles_key=prefs.opensubtitles_key)
    decoded = json.loads(base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)))
    assert "opensubtitles_username" not in decoded
    assert "opensubtitles_password" not in decoded
    assert UserPreferences().enable_opensubtitles is True
    assert parse_user_config(encode_user_config(enable_opensubtitles=False)).enable_opensubtitles is False


@pytest.mark.asyncio
async def test_search_and_download_transport_never_login_or_send_authorization():
    seen = []
    def transport(request):
        seen.append(request)
        assert "authorization" not in request.headers
        if request.url.path == "/api/v1/subtitles":
            assert request.headers["Api-Key"] == "example-key"
            return httpx.Response(200, json={"data": []})
        if request.url.path == "/api/v1/download":
            assert request.headers["Api-Key"] == "example-key"
            assert json.loads(request.content) == {"file_id": 123}
            return httpx.Response(200, json={"link": "https://example.invalid/subtitle.srt"})
        assert str(request.url) == "https://example.invalid/subtitle.srt"
        return httpx.Response(200, content=SRT)
    OPENSUBTITLES_BREAKER.reset()
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as upstream:
        provider = OpenSubtitlesProvider(upstream)
        assert await provider.search_subtitles("tt1234567", api_key="example-key") == []
        assert await provider.download_archive("/sub/opensubtitles/123.srt", api_key="example-key") == SRT
    assert [r.method for r in seen] == ["GET", "POST", "GET"]
    assert all(not r.url.path.endswith("/login") for r in seen)


@pytest.mark.asyncio
async def test_missing_key_prevents_all_provider_requests(monkeypatch):
    monkeypatch.setattr(settings, "OPENSUBTITLES_API_KEY", "")
    upstream = AsyncMock(spec=httpx.AsyncClient)
    provider = OpenSubtitlesProvider(upstream)
    assert await provider.search_subtitles("tt1234567") == []
    assert await provider.get_download_url(123) is None
    assert await provider.download_archive("/sub/opensubtitles/123.srt") is None
    assert await provider.download_archive("https://example.invalid/subtitle.srt") is None
    upstream.get.assert_not_awaited()
    upstream.post.assert_not_awaited()


def test_stremio_route_uses_real_key_only_download_pipeline(client, monkeypatch):
    import app.main as main

    calls = []
    upstream = AsyncMock(spec=httpx.AsyncClient)
    def post(url, **kwargs):
        calls.append(("POST", url))
        assert url.endswith("/download")
        assert kwargs["headers"]["Api-Key"] == "example-key"
        assert "Authorization" not in kwargs["headers"]
        assert kwargs["json"] == {"file_id": 123}
        return httpx.Response(200, json={"link": "https://example.invalid/subtitle.srt"})
    def get(url, **kwargs):
        calls.append(("GET", url))
        assert url == "https://example.invalid/subtitle.srt"
        return httpx.Response(200, content=SRT)
    upstream.post.side_effect = post
    upstream.get.side_effect = get
    monkeypatch.setattr(main, "_http_client", upstream)
    monkeypatch.setattr(settings, "OPENSUBTITLES_API_KEY", "example-key")
    monkeypatch.setattr(settings, "ENABLE_SUBTITLE_SYNC", False)
    monkeypatch.setattr(main.cache_manager, "get_subtitle", AsyncMock(return_value=None))
    monkeypatch.setattr(main.cache_manager, "is_failed", lambda _: False)
    monkeypatch.setattr(main.cache_manager, "get_metadata", lambda _: {
        "provider": "opensubtitles", "download_url": "/sub/opensubtitles/123.srt",
        "release_name": "Example.1080p.BluRay", "lang": "eng",
        "opensubtitles_username": "ignored", "opensubtitles_password": "ignored"})
    OPENSUBTITLES_BREAKER.reset()
    response = client.get("/sub/key_only_test.srt")
    assert response.status_code == 200
    assert b"Subtitle delivered successfully" in response.content
    assert len(calls) == 2
