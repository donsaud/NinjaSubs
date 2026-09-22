"""Tests for configuration UI preferences:
- HI preference
- Selectable subtitle badge format
- Live badge preview markup on /configure
"""

import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models import SubtitleRelease, UserPreferences
from app.services.aggregator import aggregate_subtitles, format_informative_badge
from app.services.subtitle_matcher import rank_subtitles
from app.utils.config_parser import encode_user_config, parse_user_config

# ============================================================================
# 1. PREFERENCE SERIALIZATION
# ============================================================================


def test_encode_parse_phase2_preferences_roundtrip():
    """New preferences must round-trip through URL-safe base64."""
    encoded = encode_user_config(
        subdl_key="k1",
        subsource_key="k2",
        languages=["ara", "eng"],
        hi_preference="prefer",
    )
    prefs = parse_user_config(encoded)

    assert prefs.hi_preference == "prefer"


def test_defaults_when_phase2_fields_absent():
    """Legacy payloads without new keys fall back to safe defaults."""
    encoded = encode_user_config("k1", "k2")
    prefs = parse_user_config(encoded)

    assert prefs.hi_preference == "neutral"


def test_parse_query_string_phase2_preferences():
    """Query-string config format supports new preferences."""
    prefs = parse_user_config("subdl=k1&hi_preference=exclude")
    assert prefs.hi_preference == "exclude"


def test_encode_parse_keyless_provider_toggles():
    """Keyless provider enable/disable toggles round-trip and default to enabled."""
    encoded = encode_user_config("k1", "k2", enable_yifysubtitles=False, enable_subtitlecat=True)
    prefs = parse_user_config(encoded)
    assert prefs.enable_yifysubtitles is False
    assert prefs.enable_subtitlecat is True

    default = parse_user_config(encode_user_config("k1", "k2"))
    assert default.enable_yifysubtitles is False
    assert default.enable_subtitlecat is False

    qs = parse_user_config("subdl=k1&enable_subtitlecat=true")
    assert qs.enable_subtitlecat is True


def test_encode_parse_rtl_fix_toggle():
    """enable_rtl_fix round-trips and defaults to True for legacy configs."""
    encoded_off = encode_user_config("k1", "k2", enable_rtl_fix=False)
    prefs_off = parse_user_config(encoded_off)
    assert prefs_off.enable_rtl_fix is False

    encoded_on = encode_user_config("k1", "k2", enable_rtl_fix=True)
    prefs_on = parse_user_config(encoded_on)
    assert prefs_on.enable_rtl_fix is True

    # Legacy config without the field defaults to enabled.
    assert parse_user_config(encode_user_config("k1", "k2")).enable_rtl_fix is True
    assert UserPreferences().enable_rtl_fix is True

    # Query-string form.
    assert parse_user_config("subdl=k1&enable_rtl_fix=false").enable_rtl_fix is False


def test_cache_key_includes_rtl_fix():
    from app.services.cache import build_cache_key

    base = dict(media_type="movie", imdb_id="tt0111161")
    on = build_cache_key(**base, enable_rtl_fix=True)
    off = build_cache_key(**base, enable_rtl_fix=False)
    assert on != off


def test_configure_page_shows_rtl_fix_toggle(client):
    body = client.get("/configure").text
    assert 'id="enableRtlFix"' in body
    assert "Arabic RTL Alignment Fix" in body
    assert (
        "Automatically fix inverted punctuation, brackets, and quotes in Arabic subtitles"
        in body
    )


def test_configure_page_prefills_rtl_fix(client):
    encoded = encode_user_config(subdl_key="k", enable_rtl_fix=False)
    body = client.get(f"/{encoded}/configure").text
    assert '"enable_rtl_fix": false' in body


@pytest.mark.asyncio
async def test_rtl_fix_toggle_controls_served_content(client):
    """When disabled, the raw text is served; when enabled, the RLM fix is applied."""
    rlm = "\u200f".encode()
    raw = "1\n00:00:01,000 --> 00:00:04,000\nمرحبا بك.\n".encode()

    for enabled, expect_rlm in ((True, True), (False, False)):
        cfg = encode_user_config(opensubtitles_key="k", enable_rtl_fix=enabled)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = raw
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        with (
            patch(
                "app.providers.opensubtitles.OpenSubtitlesProvider.get_download_url",
                new=AsyncMock(return_value="https://download.opensubtitles.com/temp/x.srt"),
            ),
            patch("app.cache.cache_manager.get_subtitle", new=AsyncMock(return_value=None)),
            patch("app.main._http_client", new=mock_http),
        ):
            resp = client.get(f"/{cfg}/sub/opensubtitles/{700 + int(enabled)}.srt")
            assert resp.status_code == 200
            if expect_rlm:
                assert rlm in resp.content
            else:
                assert rlm not in resp.content


def test_encode_parse_provider_enable_flags():
    """Provider enable flags round-trip and default to True."""
    encoded = encode_user_config(
        "k1", "k2", enable_subdl=False, enable_subsource=True, enable_opensubtitles=False
    )
    prefs = parse_user_config(encoded)
    assert prefs.enable_subdl is False
    assert prefs.enable_subsource is True
    assert prefs.enable_opensubtitles is False

    defaults = parse_user_config(encode_user_config("k1", "k2"))
    assert defaults.enable_subdl is True
    assert defaults.enable_subsource is True
    assert defaults.enable_opensubtitles is False

    assert parse_user_config("subdl=k1&enable_subsource=false").enable_subsource is False


@pytest.mark.asyncio
async def test_disabled_key_provider_pref_skips_provider():
    """A disabled key-based provider must not be queried."""
    p_subdl = AsyncMock()
    p_subdl.search_subtitles = AsyncMock(return_value=[])
    p_subsource = AsyncMock()
    p_subsource.search_subtitles = AsyncMock(return_value=[])

    await aggregate_subtitles(
        imdb_id="tt0111161",
        media_type="movie",
        user_preferences=UserPreferences(enable_subdl=False, enable_subsource=True),
        subdl_provider=p_subdl,
        subsource_provider=p_subsource,
        use_cache=False,
    )

    assert p_subdl.search_subtitles.await_count == 0
    assert p_subsource.search_subtitles.await_count == 1


def test_encode_parse_ad_removal_toggle():
    """enable_ad_removal round-trips and defaults to True for legacy configs."""
    assert parse_user_config(encode_user_config("k1", "k2", enable_ad_removal=False)).enable_ad_removal is False
    assert parse_user_config(encode_user_config("k1", "k2", enable_ad_removal=True)).enable_ad_removal is True
    assert parse_user_config(encode_user_config("k1", "k2")).enable_ad_removal is True
    assert UserPreferences().enable_ad_removal is True
    assert parse_user_config("subdl=k1&enable_ad_removal=false").enable_ad_removal is False


def test_cache_key_includes_ad_removal():
    from app.services.cache import build_cache_key

    base = dict(media_type="movie", imdb_id="tt0111161")
    assert build_cache_key(**base, enable_ad_removal=True) != build_cache_key(
        **base, enable_ad_removal=False
    )


def test_configure_page_shows_ad_removal_toggle(client):
    body = client.get("/configure").text
    assert 'id="enableAdRemoval"' in body
    assert "Remove ads" in body
    assert "Clean promotional links, website domains, and social media handles" in body


@pytest.mark.asyncio
async def test_ad_removal_toggle_controls_served_content(client):
    """When enabled, ad cues are stripped; when disabled they are preserved."""
    raw = (
        b"1\n00:00:01,000 --> 00:00:03,000\nwww.adsite.com\n\n"
        b"2\n00:00:05,000 --> 00:00:08,000\nHello world\n"
    )

    for enabled, expect_ad in ((True, False), (False, True)):
        cfg = encode_user_config(opensubtitles_key="k", enable_ad_removal=enabled)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = raw
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        with (
            patch(
                "app.providers.opensubtitles.OpenSubtitlesProvider.get_download_url",
                new=AsyncMock(return_value="https://download.opensubtitles.com/temp/y.srt"),
            ),
            patch("app.cache.cache_manager.get_subtitle", new=AsyncMock(return_value=None)),
            patch("app.main._http_client", new=mock_http),
        ):
            resp = client.get(f"/{cfg}/sub/opensubtitles/{900 + int(enabled)}.srt")
            assert resp.status_code == 200
            if expect_ad:
                assert b"www.adsite.com" in resp.content
            else:
                assert b"www.adsite.com" not in resp.content
            assert b"Hello world" in resp.content


def test_encode_parse_keep_translator_credits():
    """keep_translator_credits round-trips and defaults to True."""
    assert parse_user_config(
        encode_user_config("k1", "k2", keep_translator_credits=False)
    ).keep_translator_credits is False
    assert parse_user_config(
        encode_user_config("k1", "k2", keep_translator_credits=True)
    ).keep_translator_credits is True
    assert parse_user_config(encode_user_config("k1", "k2")).keep_translator_credits is True
    assert UserPreferences().keep_translator_credits is True
    assert parse_user_config(
        "subdl=k1&keep_translator_credits=false"
    ).keep_translator_credits is False


def test_cache_key_includes_keep_translator_credits():
    from app.services.cache import build_cache_key

    base = dict(media_type="movie", imdb_id="tt0111161")
    assert build_cache_key(**base, keep_translator_credits=True) != build_cache_key(
        **base, keep_translator_credits=False
    )


def test_configure_page_shows_keep_translator_credits_toggle(client):
    body = client.get("/configure").text
    assert 'id="keepTranslatorCredits"' in body
    assert "Keep translator credits" in body
    assert "Preserve translator names while removing spam, ads, and website links." in body
    # Standalone preference: fully decoupled from the "Remove ads" toggle.
    assert 'id="translatorCreditsRow" class="pref-row"' in body
    assert "updateAdRemovalDependents" not in body


@pytest.mark.asyncio
async def test_keep_translator_credits_toggle_controls_served_content(client):
    """When enabled, credit text survives with the ad stripped; when disabled it is removed."""
    raw = (
        b"1\n00:00:01,000 --> 00:00:03,000\n\xd8\xaa\xd8\xb1\xd8\xac\xd9\x85\xd8\xa9 "
        b"\xd9\x81\xd9\x84\xd8\xa7\xd9\x86 www.adsite.com\n\n"
        b"2\n00:00:05,000 --> 00:00:08,000\nHello world\n"
    )

    for keep, expect_credit in ((True, True), (False, False)):
        cfg = encode_user_config(
            opensubtitles_key="k", enable_opensubtitles=True, keep_translator_credits=keep
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = raw
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        with (
            patch(
                "app.providers.opensubtitles.OpenSubtitlesProvider.get_download_url",
                new=AsyncMock(return_value="https://download.opensubtitles.com/temp/z.srt"),
            ),
            patch("app.cache.cache_manager.get_subtitle", new=AsyncMock(return_value=None)),
            patch("app.main._http_client", new=mock_http),
        ):
            resp = client.get(f"/{cfg}/sub/opensubtitles/{1200 + int(keep)}.srt")
            assert resp.status_code == 200
            assert b"www.adsite.com" not in resp.content
            if expect_credit:
                assert "ترجمة".encode() in resp.content
            else:
                assert "ترجمة".encode() not in resp.content
            assert b"Hello world" in resp.content


def test_exclude_hi_implies_exclude_hi_preference():
    """exclude_hi=True is reflected as hi_preference='exclude' when unset."""
    encoded = encode_user_config("k1", "k2", exclude_hi=True)
    prefs = parse_user_config(encoded)
    assert prefs.hi_preference == "exclude"


# ============================================================================
# 2. PROVIDER QUERYING
# ============================================================================


@pytest.mark.asyncio
async def test_all_providers_queried_unconditionally():
    """SubDL and SubSource are always queried (no provider-priority feature)."""
    p_subdl = AsyncMock()
    p_subdl.search_subtitles = AsyncMock(return_value=[])
    p_subsource = AsyncMock()
    p_subsource.search_subtitles = AsyncMock(return_value=[])

    prefs = UserPreferences(subdl_key="k1", subsource_key="k2")

    await aggregate_subtitles(
        imdb_id="tt0111161",
        media_type="movie",
        user_preferences=prefs,
        subdl_provider=p_subdl,
        subsource_provider=p_subsource,
        use_cache=False,
    )

    assert p_subdl.search_subtitles.await_count == 1
    assert p_subsource.search_subtitles.await_count == 1


# ============================================================================
# 3. HI PREFERENCE
# ============================================================================


def test_hi_preference_exclude_and_prefer():
    """hi_preference='exclude' removes SDH; 'prefer' ranks it first on ties."""
    video = "Show.S01E01.mkv"
    regular = SubtitleRelease(
        release_name="Show.S01E01.WEB-DL.srt", download_url="http://r", provider="subdl"
    )
    sdh = SubtitleRelease(
        release_name="Show.S01E01.WEB-DL.SDH.srt", download_url="http://s", provider="subdl"
    )

    excl = rank_subtitles(video, [regular, sdh], hi_preference="exclude")
    assert len(excl) == 1
    assert "SDH" not in excl[0].release_name

    pref = rank_subtitles(video, [regular, sdh], hi_preference="prefer")
    assert len(pref) == 2
    assert "SDH" in pref[0].release_name


# ============================================================================
# 5. CONFIGURATION UI MARKUP
# ============================================================================


# ============================================================================
# 4b. SUBTITLE BADGE FORMAT
# ============================================================================


def test_format_informative_badge_styles():
    """The user-selectable badge styles render the expected labels."""
    sub = SubtitleRelease(
        release_name="Movie.Title.2024.1080p.BluRay.x264-FLUX.srt",
        download_url="http://x",
        provider="subdl",
        match_percentage=95,
    )

    assert (
        format_informative_badge(sub, 95, source_tag="SubDL", badge_format="score_provider")
        == "[95%] [SubDL] Movie.Title.2024.1080p.BluRay.x264-FLUX"
    )
    assert (
        format_informative_badge(sub, 95, source_tag="SubDL", badge_format="score")
        == "[95%] Movie.Title.2024.1080p.BluRay.x264-FLUX"
    )
    assert (
        format_informative_badge(sub, 95, source_tag="SubDL", badge_format="plain")
        == "Movie.Title.2024.1080p.BluRay.x264-FLUX"
    )


def test_format_informative_badge_uploader_style():
    """The uploader style appends ' (by username)' and falls back when absent."""
    sub = SubtitleRelease(
        release_name="Movie.Title.2024.1080p.BluRay.x264-FLUX.srt",
        download_url="http://x",
        provider="opensubtitles",
        match_percentage=95,
        uploader="subsmaster",
    )
    assert (
        format_informative_badge(sub, 95, badge_format="uploader")
        == "[95%] Movie.Title.2024.1080p.BluRay.x264-FLUX (by subsmaster)"
    )

    # No uploader -> score-only fallback (no dangling "(by ...)")
    sub.uploader = ""
    assert (
        format_informative_badge(sub, 95, badge_format="uploader")
        == "[95%] Movie.Title.2024.1080p.BluRay.x264-FLUX"
    )


def test_uploader_preserved_through_ranking_and_badge():
    """Uploader survives ranking and appears in the rendered subtitle title."""
    video = "Movie.Title.2024.1080p.BluRay.x264-FLUX.mkv"
    sub = SubtitleRelease(
        release_name="Movie.Title.2024.1080p.BluRay.x264-FLUX.srt",
        download_url="http://x",
        provider="opensubtitles",
        lang="ara",
        uploader="uploader_joe",
    )
    ranked = rank_subtitles(video, [sub])
    assert ranked[0].uploader == "uploader_joe"
    assert (
        format_informative_badge(ranked[0], 100, badge_format="uploader")
        .endswith("(by uploader_joe)")
    )


@pytest.mark.asyncio
async def test_keyless_scraper_providers_queried_when_injected():
    """YIFYSubtitles and SubtitleCat are queried when their providers are injected."""
    p_subdl = AsyncMock()
    p_subdl.search_subtitles = AsyncMock(return_value=[])
    p_subsource = AsyncMock()
    p_subsource.search_subtitles = AsyncMock(return_value=[])
    p_yify = AsyncMock()
    p_yify.search_subtitles = AsyncMock(return_value=[])
    p_cat = AsyncMock()
    p_cat.search_subtitles = AsyncMock(return_value=[])

    await aggregate_subtitles(
        imdb_id="tt0111161",
        media_type="movie",
        title="Gladiator",
        year=2000,
        user_preferences=UserPreferences(),
        subdl_provider=p_subdl,
        subsource_provider=p_subsource,
        yifysubtitles_provider=p_yify,
        subtitlecat_provider=p_cat,
        use_cache=False,
    )

    assert p_yify.search_subtitles.await_count == 1
    assert p_cat.search_subtitles.await_count == 1


def test_badge_parts_encode_parse_roundtrip():
    """badge_parts serializes only when non-default and parses back in canonical order."""
    encoded = encode_user_config("k1", "k2", badge_parts=["uploader", "score"])
    prefs = parse_user_config(encoded)
    # Canonical order applied; provider + filename omitted
    assert prefs.badge_parts == ["score", "uploader"]
    assert prefs.resolved_badge_parts == ["score", "uploader"]

    # Default is omitted and resolves to score + provider + filename + uploader
    prefs_default = parse_user_config(encode_user_config("k1", "k2"))
    assert prefs_default.badge_parts == ["score", "provider", "filename", "uploader"]

    # Empty/invalid selections fall back to the default
    assert UserPreferences(badge_parts=[]).resolved_badge_parts == [
        "score",
        "provider",
        "filename",
        "uploader",
    ]
    assert UserPreferences(badge_parts=["bogus"]).resolved_badge_parts == [
        "score",
        "provider",
        "filename",
        "uploader",
    ]


def test_legacy_badge_format_still_supported():
    """Legacy badge_format presets map back to their component lists."""
    prefs_plain = parse_user_config(encode_user_config("k1", "k2", badge_format="plain"))
    assert prefs_plain.resolved_badge_parts == ["filename"]
    assert prefs_plain.badge_format == "plain"

    prefs_uploader = parse_user_config(encode_user_config("k1", "k2", badge_format="uploader"))
    assert prefs_uploader.resolved_badge_parts == ["score", "filename", "uploader"]
    assert prefs_uploader.badge_format == "uploader"

    assert parse_user_config("subdl=k1&badge_format=score").resolved_badge_parts == [
        "score",
        "filename",
    ]


@pytest.mark.asyncio
async def test_endpoint_respects_badge_format(client):
    """The /subtitles endpoint must serve titles using the configured badge style."""
    sub = SubtitleRelease(
        release_name="Shawshank.1994.1080p.BluRay.x264-FLUX.srt",
        download_url="/sub/x.zip",
        provider="subdl",
        lang="ara",
    )
    config = encode_user_config(subdl_key="k", badge_format="plain")

    with (
        patch("app.main._http_client", new_callable=AsyncMock),
        patch(
            "app.providers.subdl.SubdlProvider.search_subtitles",
            new=AsyncMock(return_value=[sub]),
        ),
        patch(
            "app.providers.subsource.SubsourceProvider.search_subtitles",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "app.providers.cinemeta.CinemetaClient.get_metadata",
            new=AsyncMock(return_value=None),
        ),
    ):
        extra = "filename=Shawshank.1994.1080p.BluRay.x264-FLUX.mkv"
        resp = client.get(f"/{config}/subtitles/movie/tt0111161/{extra}.json")
        assert resp.status_code == 200
        subs = resp.json()["subtitles"]
        assert len(subs) == 1
        assert subs[0]["title"] == "Shawshank.1994.1080p.BluRay.x264-FLUX"


@pytest.mark.asyncio
async def test_endpoint_respects_badge_parts_uploader(client):
    """The endpoint composes the title from selected badge parts including the uploader."""
    sub = SubtitleRelease(
        release_name="Shawshank.1994.1080p.BluRay.x264-FLUX.srt",
        download_url="/sub/y.zip",
        provider="opensubtitles",
        lang="ara",
        uploader="subsmaster",
    )
    config = encode_user_config(
        opensubtitles_key="k",
        enable_opensubtitles=True,
        badge_parts=["score", "filename", "uploader"],
    )

    with (
        patch("app.main._http_client", new_callable=AsyncMock),
        patch(
            "app.providers.subdl.SubdlProvider.search_subtitles",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "app.providers.subsource.SubsourceProvider.search_subtitles",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "app.providers.opensubtitles.OpenSubtitlesProvider.search_subtitles",
            new=AsyncMock(return_value=[sub]),
        ),
        patch(
            "app.providers.cinemeta.CinemetaClient.get_metadata",
            new=AsyncMock(return_value=None),
        ),
    ):
        extra = "filename=Shawshank.1994.1080p.BluRay.x264-FLUX.mkv"
        resp = client.get(f"/{config}/subtitles/movie/tt0111161/{extra}.json")
        assert resp.status_code == 200
        subs = resp.json()["subtitles"]
        assert len(subs) == 1
        assert subs[0]["title"] == (
            "[100%] Shawshank.1994.1080p.BluRay.x264-FLUX (by subsmaster)"
        )


@pytest.mark.asyncio
async def test_endpoint_track_id_has_no_hash(client):
    """The response id must be the clean label (no internal 16-hex sub_id) because
    some clients such as Nuvio render the id directly."""
    sub = SubtitleRelease(
        release_name="The.Shawshank.Redemption.1994.1080p.BluRay.srt",
        download_url="/sub/z.zip",
        provider="subdl",
        lang="ara",
        uploader="MostafaNegm",
    )
    config = encode_user_config(subdl_key="k", badge_parts=["score", "provider", "filename", "uploader"])

    with (
        patch("app.main._http_client", new_callable=AsyncMock),
        patch(
            "app.providers.subdl.SubdlProvider.search_subtitles",
            new=AsyncMock(return_value=[sub]),
        ),
        patch(
            "app.providers.subsource.SubsourceProvider.search_subtitles",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "app.providers.cinemeta.CinemetaClient.get_metadata",
            new=AsyncMock(return_value=None),
        ),
    ):
        extra = "filename=The.Shawshank.Redemption.1994.1080p.BluRay.mkv"
        resp = client.get(f"/{config}/subtitles/movie/tt0111161/{extra}.json")
        assert resp.status_code == 200
        items = resp.json()["subtitles"]
        assert len(items) == 1
        item = items[0]
        assert item["id"] == item["title"]
        assert " (by MostafaNegm)" in item["id"]
        # No trailing/internal 16-hex hash anywhere in the id
        assert not re.search(r"[0-9a-f]{16}", item["id"])


def test_configure_page_contains_badge_builder(client):
    """The config page exposes the four toggleable badge components."""
    body = client.get("/configure").text
    assert 'id="badgeBuilder"' in body
    for part in ("score", "provider", "filename", "uploader"):
        assert f'data-part="{part}"' in body
    assert "const initialPrefs =" in body


def test_configure_page_prefills_badge_parts(client):
    """/{config}/configure injects the previously selected badge components."""
    encoded = encode_user_config(subdl_key="k", badge_parts=["filename"])
    body = client.get(f"/{encoded}/configure").text
    assert '"badge_parts": ["filename"]' in body


def test_cache_key_includes_keyless_provider_toggles():
    """Cache key must vary with the keyless provider toggles to avoid cross-user leakage."""
    from app.services.cache import build_cache_key

    base = dict(media_type="movie", imdb_id="tt0111161")
    on = build_cache_key(**base, enable_yifysubtitles=True, enable_subtitlecat=True)
    yify_off = build_cache_key(**base, enable_yifysubtitles=False, enable_subtitlecat=True)
    cat_off = build_cache_key(**base, enable_yifysubtitles=True, enable_subtitlecat=False)
    assert on != yify_off
    assert on != cat_off
    assert yify_off != cat_off


def test_configure_page_shows_provider_icons(client):
    """Each provider label is preceded by its locally-served icon."""
    body = client.get("/configure").text
    for icon in (
        "subdl.png",
        "subsource.png",
        "opensubtitles.png",
        "yifysubtitles.png",
        "subtitlecat.png",
    ):
        assert f'/static/{icon}' in body
        assert client.get(f"/static/{icon}").status_code == 200


def test_configure_page_shows_keyless_provider_toggles(client):
    """YIFYSubtitles and SubtitleCat appear with enable/disable toggles."""
    body = client.get("/configure").text
    assert "YIFYSubtitles" in body
    assert "SubtitleCat" in body
    assert 'id="enableYifysubtitles"' in body
    assert 'id="enableSubtitlecat"' in body
    assert "Free / No Key" in body
    assert "Machine Translated" in body


def test_configure_page_shows_unified_provider_rows(client):
    """All five providers render consistent rows with badges and toggles."""
    body = client.get("/configure").text
    assert "Subtitle Providers" in body
    for toggle in (
        "enableSubdl",
        "enableSubsource",
        "enableOpensubtitles",
        "enableYifysubtitles",
        "enableSubtitlecat",
    ):
        assert f'id="{toggle}"' in body
    assert body.count("Requires Key") == 3
    assert 'id="subdlKey"' in body
    assert 'id="subsourceKey"' in body
    assert 'id="opensubtitlesKey"' in body
    # Semantic badge colors
    assert "bg-amber-500/10 text-amber-400 border-amber-500/20" in body
    assert "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" in body
    assert "bg-purple-500/10 text-purple-400 border-purple-500/20" in body


@pytest.mark.asyncio
async def test_disabled_scraper_pref_skips_provider(monkeypatch):
    """A per-user disabled keyless provider must not be queried."""
    from app.config import settings

    monkeypatch.setattr(settings, "ENABLE_YIFYSUBTITLES", True, raising=False)
    monkeypatch.setattr(settings, "ENABLE_SUBTITLECAT", True, raising=False)

    p_subdl = AsyncMock()
    p_subdl.search_subtitles = AsyncMock(return_value=[])
    p_subsource = AsyncMock()
    p_subsource.search_subtitles = AsyncMock(return_value=[])

    yify_mock = AsyncMock(return_value=[])
    cat_mock = AsyncMock(return_value=[])

    prefs = UserPreferences(enable_yifysubtitles=False, enable_subtitlecat=True)

    with (
        patch(
            "app.providers.yifysubtitles.YifysubtitlesProvider.search_subtitles", new=yify_mock
        ),
        patch("app.providers.subtitlecat.SubtitlecatProvider.search_subtitles", new=cat_mock),
    ):
        await aggregate_subtitles(
            imdb_id="tt0111161",
            media_type="movie",
            title="Gladiator",
            year=2000,
            user_preferences=prefs,
            http_client=AsyncMock(),
            subdl_provider=p_subdl,
            subsource_provider=p_subsource,
            use_cache=False,
        )

    assert yify_mock.await_count == 0
    assert cat_mock.await_count == 1


def test_configure_page_contains_phase2_controls(client):
    """The /configure page must expose the live badge preview and no removed controls."""
    resp = client.get("/configure")
    assert resp.status_code == 200
    body = resp.text

    assert "Subtitle Badge Preview" in body
    assert 'id="badgePreview"' in body
    assert "Provider Priorities" not in body
    assert 'id="providerList"' not in body
    assert "Minimum Match Threshold" not in body
    assert 'id="minThreshold"' not in body


def test_configured_manifest_preserves_phase2_config(client):
    """The generated manifest URL must remain valid when Phase 2 prefs are encoded."""
    encoded = encode_user_config(
        subdl_key="k1",
        hi_preference="prefer",
    )
    resp = client.get(f"/{encoded}/manifest.json")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "org.ninjasubs.addon"
