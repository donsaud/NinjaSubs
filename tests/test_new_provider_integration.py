"""Configuration, aggregation and actual delivery integration for optional providers."""
from unittest.mock import AsyncMock

import httpx
import pytest

from app.models import SubtitleRelease, UserPreferences
from app.providers import GestdownProvider, IsubtitlesProvider
from app.services.aggregator import aggregate_subtitles, format_informative_badge
from app.services.cache import build_cache_key
from app.utils.config_parser import encode_user_config, parse_user_config


@pytest.mark.parametrize("flag", ["enable_gestdown", "enable_isubtitles"])
def test_config_default_and_roundtrip(flag):
    assert not getattr(UserPreferences(), flag)
    assert not getattr(parse_user_config(None), flag)
    assert getattr(parse_user_config(encode_user_config(**{flag: True})), flag)
    assert getattr(parse_user_config(f"{flag}=true"), flag)
    assert not getattr(parse_user_config(f"{flag}=false"), flag)
    legacy = parse_user_config(encode_user_config(enable_subtitlecat=True, enable_yifysubtitles=True))
    assert legacy.enable_subtitlecat and legacy.enable_yifysubtitles
    assert not legacy.enable_gestdown and not legacy.enable_isubtitles


def test_registry_and_cache_isolation_without_changing_legacy_keys():
    import app.providers as registry
    assert registry.GestdownProvider is GestdownProvider
    assert registry.IsubtitlesProvider is IsubtitlesProvider
    assert "SubtitlecatProvider" not in registry.__all__
    assert "YifysubtitlesProvider" not in registry.__all__
    args = {"media_type": "series", "imdb_id": "tt0903747", "season": 1, "episode": 1}
    original = build_cache_key(**args)
    assert original == build_cache_key(**args, enable_gestdown=False, enable_isubtitles=False)
    enabled = build_cache_key(**args, enable_gestdown=True)
    assert original != enabled
    assert enabled != build_cache_key(**args, enable_gestdown=True, tvdb_id=81189)
    assert original != build_cache_key(**args, enable_isubtitles=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("movie", [True, False])
async def test_aggregator_skips_movie_and_preserves_other_provider_results(movie):
    good = SubtitleRelease(release_name="Breaking.Bad.S01E01.HDTV-GROUP", provider="subdl",
                           download_url="/safe.zip", lang="eng")
    subdl = AsyncMock()
    subdl.search_subtitles.return_value = [good]
    gest = AsyncMock()
    gest.search_subtitles.side_effect = httpx.ConnectError("mock")
    results = await aggregate_subtitles(
        "tt0903747", media_type="movie" if movie else "series",
        season=None if movie else 1, episode=None if movie else 1,
        title="Breaking Bad", tvdb_id=81189, languages=["eng"],
        user_preferences=UserPreferences(enable_subsource=False, enable_gestdown=True),
        subdl_provider=subdl, gestdown_provider=gest, use_cache=False)
    assert results and results[0].provider == "subdl"
    assert gest.search_subtitles.await_count == (0 if movie else 1)
    if not movie:
        assert gest.search_subtitles.call_args.kwargs["tvdb_id"] == 81189
        assert gest.search_subtitles.call_args.kwargs["season"] == 1
        assert gest.search_subtitles.call_args.kwargs["episode"] == 1


@pytest.mark.parametrize("provider,label", [("gestdown", "Gestdown"), ("isubtitles", "iSubtitles")])
def test_badge(provider, label):
    result = SubtitleRelease(release_name="Release", provider=provider, download_url="/id")
    assert f"[{label}]" in format_informative_badge(result, 50)


@pytest.mark.parametrize("provider", ["gestdown", "isubtitles"])
def test_download_route_uses_existing_response_pipeline(client, monkeypatch, provider):
    import app.main as main
    raw = b"1\n00:00:01,000 --> 00:00:02,000\nHello.\n"
    cls = GestdownProvider if provider == "gestdown" else IsubtitlesProvider
    download = AsyncMock(return_value=raw)
    monkeypatch.setattr(cls, "download_archive", download)
    monkeypatch.setattr(main, "_http_client", AsyncMock(spec=httpx.AsyncClient))
    monkeypatch.setattr(main.cache_manager, "get_subtitle", AsyncMock(return_value=None))
    monkeypatch.setattr(main.cache_manager, "get_metadata", lambda _: {
        "provider": provider, "download_url": "/verified-id", "release_name": "Release", "lang": "eng"})
    response = client.get("/sub/0123456789abcdef.srt")
    assert response.status_code == 200 and b"Hello." in response.content
    download.assert_awaited_once_with("/verified-id")


def test_configure_prefill_and_explicit_opt_in(client):
    config = encode_user_config(enable_gestdown=True, enable_isubtitles=True)
    body = client.get(f"/{config}/configure").text
    assert '"enable_gestdown": true' in body and '"enable_isubtitles": true' in body
    assert "if (catEnabled) configObj.enable_gestdown = true" in body
    assert "if (yifyEnabled) configObj.enable_isubtitles = true" in body
    for toggle in ("enableGestdown", "enableIsubtitles"):
        element = body.split(f'id="{toggle}"', 1)[1].split(">", 1)[0]
        assert "checked" not in element
