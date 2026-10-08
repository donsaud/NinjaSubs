"""Three-provider product surface and safe compatibility with retired tokens."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock
from urllib.parse import urlsplit

import httpx
import pytest

import app.providers as registry
from app.models import SubtitleRelease, UserPreferences
from app.services import aggregator
from app.services.cache import build_cache_key
from app.utils.config_parser import encode_user_config, parse_user_config

RETIRED = ("gestdown", "isubtitles", "yifysubtitles", "subtitlecat")


@pytest.mark.parametrize("flag", ["enable_" + name for name in RETIRED])
@pytest.mark.parametrize("encoding", ["token", "query"])
def test_retired_flags_are_parseable_but_cannot_schedule(flag, encoding):
    token = encode_user_config(**{flag: True}) if encoding == "token" else flag + "=true"
    prefs = parse_user_config(token)
    assert isinstance(prefs, UserPreferences)
    assert getattr(prefs, flag) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("media", ["movie", "series"])
async def test_legacy_flags_only_query_the_three_active_providers(media):
    providers = [AsyncMock() for _ in range(3)]
    for provider in providers:
        provider.search_subtitles.return_value = []
    prefs = parse_user_config(encode_user_config(
        enable_subdl=True, enable_subsource=True, enable_opensubtitles=True,
        enable_gestdown=True, enable_isubtitles=True,
        enable_yifysubtitles=True, enable_subtitlecat=True))
    await aggregator.aggregate_subtitles(
        "tt1234567", media_type=media, season=1 if media == "series" else None,
        episode=2 if media == "series" else None, user_preferences=prefs,
        subdl_provider=providers[0], subsource_provider=providers[1],
        opensubtitles_provider=providers[2], use_cache=False)
    for provider in providers:
        provider.search_subtitles.assert_awaited_once()
    # No optional client is supplied: any extra scheduled provider would fail.


def test_registry_exports_only_the_three_subtitle_provider_classes():
    classes = {name for name in registry.__all__ if name.endswith("Provider")}
    assert classes == {"BaseSubtitleProvider", "SubdlProvider", "SubsourceProvider", "OpenSubtitlesProvider"}
    assert not hasattr(registry, "GestdownProvider")
    assert not hasattr(registry, "IsubtitlesProvider")


@pytest.mark.parametrize("provider", ["gestdown", "isubtitles"])
def test_removed_cache_dimensions_are_ignored(provider):
    args = {"media_type": "series", "imdb_id": "tt1234567", "season": 1, "episode": 2}
    assert build_cache_key(**args) == build_cache_key(**args, **{"enable_" + provider: True}, tvdb_id=123)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", RETIRED)
async def test_historical_aggregation_cache_cannot_list_retired_providers(monkeypatch, provider):
    current = SubtitleRelease(provider="subdl", release_name="Current", download_url="/core")
    retired = SubtitleRelease(provider=provider, release_name="Retired", download_url="/old")
    monkeypatch.setattr(aggregator, "get_cached_subtitles", lambda _: [retired, current])
    result = await aggregator.aggregate_subtitles("tt1234567")
    assert [r.provider for r in result] == ["subdl"]


@pytest.mark.asyncio
async def test_cache_containing_only_retired_entries_refetches_active_results(monkeypatch):
    retired = SubtitleRelease(provider="gestdown", release_name="Retired", download_url="/old")
    monkeypatch.setattr(aggregator, "get_cached_subtitles", lambda _: [retired])
    monkeypatch.setattr(aggregator, "set_cached_subtitles", lambda *a: None)
    subdl = AsyncMock()
    subdl.search_subtitles.return_value = []
    await aggregator.aggregate_subtitles("tt1234567",
        user_preferences=UserPreferences(enable_subsource=False), subdl_provider=subdl)
    subdl.search_subtitles.assert_awaited_once()


@pytest.mark.parametrize("provider", RETIRED)
@pytest.mark.parametrize("cached", [False, True])
def test_retired_download_routes_never_call_upstream(client, monkeypatch, provider, cached):
    import app.main as main
    upstream = AsyncMock(spec=httpx.AsyncClient)
    monkeypatch.setattr(main, "_http_client", upstream)
    monkeypatch.setattr(main.cache_manager, "get_subtitle", AsyncMock(return_value=b"cached" if cached else None))
    monkeypatch.setattr(main.cache_manager, "is_failed", lambda _: False)
    monkeypatch.setattr(main.cache_manager, "get_metadata", lambda _: {
        "provider": provider, "download_url": "/retired", "release_name": "Retired", "lang": "eng"})
    response = client.get("/sub/0123456789abcdef.srt")
    assert response.status_code == 400
    upstream.get.assert_not_awaited()
    upstream.post.assert_not_awaited()


def test_configure_legacy_token_shows_exactly_three_key_required_providers(client):
    token = encode_user_config(enable_gestdown=True, enable_isubtitles=True,
                              enable_yifysubtitles=True, enable_subtitlecat=True)
    body = client.get(f"/{token}/configure").text
    assert re.findall(r'<span class="provider-name">([^<]+)</span>', body) == [
        "SubDL", "SubSource", "OpenSubtitles"]
    assert body.count("Requires Key") == 2
    assert body.count("API Key + Account") == 1
    assert "Free / No Key" not in body
    for toggle in ("enableGestdown", "enableIsubtitles", "enableYifysubtitles", "enableSubtitlecat"):
        assert toggle not in body


@pytest.mark.parametrize("name", ["gestdown", "isubtitles"])
def test_removed_svg_assets_are_not_served(client, name):
    assert client.get(f"/static/{name}.svg").status_code == 404


@pytest.mark.parametrize("language", ["ara", "eng"])
def test_actual_install_serializer_only_emits_the_three_active_providers(language):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required to execute the actual configuration serializer")
    template = Path("app/templates/configure.html").read_text(encoding="utf-8")
    functions = template[template.index("function serializeConfig()"):
                         template.index("function updateUrls()")]
    setup = r"""
    const document={getElementById:(id)=>({checked:true,value:''})};
    const getSubdlInput=getSubsourceInput=getOpenSubtitlesInput=getOpenSubtitlesUsernameInput=getOpenSubtitlesPasswordInput=()=>({value:''});
    const DEFAULT_BADGE_PARTS=['score','provider','filename','uploader'];
    const getBadgeParts=()=>DEFAULT_BADGE_PARTS;
    const getEffectiveBaseUrl=()=> 'http://test.invalid';
    const btoa=s=>Buffer.from(s,'binary').toString('base64');
    """
    setup += "const getSelectedLanguages=()=>['" + language + "'];"
    result = subprocess.run([node, "-e", setup + functions + "console.log(getManifestUrl());"],
                            capture_output=True, text=True, check=True, timeout=10)
    token = urlsplit(result.stdout.strip()).path.split("/")[1]
    prefs = parse_user_config(token)
    assert prefs.languages == [language]
    assert all(not getattr(prefs, "enable_" + name) for name in RETIRED)
