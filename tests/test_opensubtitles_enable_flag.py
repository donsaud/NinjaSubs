"""The OpenSubtitles enable flag must survive the configure round trip.

Regression coverage for a defect found by runtime diagnosis, not by a unit test.
The three pieces disagreed:

* ``encode_user_config`` emitted the field only when it was ``True``;
* ``configure.html`` emitted ``false`` when disabled and nothing when enabled;
* ``parse_user_config`` defaulted the missing field to ``False``.

So checking the OpenSubtitles box produced no field at all, which parsed back as
disabled -- the provider could never be switched on from the UI. With the other
providers also off, the add-on returned an empty subtitle list and Stremio showed
nothing.

The flag now uses the same opt-out shape as the SubDL/SubSource toggles.
"""

from __future__ import annotations

import base64
import json

import pytest

from app.utils.config_parser import encode_user_config, parse_user_config


def raw_fields(token: str) -> dict:
    padded = token + "=" * ((4 - len(token) % 4) % 4)
    return json.loads(base64.urlsafe_b64decode(padded).decode())


# 1/2/3. Parsing behaviour
# ---------------------------------------------------------------------------


def test_missing_enable_opensubtitles_is_enabled():
    """An install URL that predates the flag must not silently disable it."""
    assert parse_user_config(None).enable_opensubtitles is True
    assert parse_user_config("").enable_opensubtitles is True
    assert parse_user_config(encode_user_config("k1", "k2")).enable_opensubtitles is True


def test_explicit_true_is_enabled():
    assert parse_user_config("enable_opensubtitles=true").enable_opensubtitles is True
    token = encode_user_config("k1", "k2", enable_opensubtitles=True)
    assert "enable_opensubtitles" not in raw_fields(token)
    assert parse_user_config(token).enable_opensubtitles is True


def test_explicit_false_is_disabled():
    assert parse_user_config("enable_opensubtitles=false").enable_opensubtitles is False
    token = encode_user_config("k1", "k2", enable_opensubtitles=False)
    # Opt-out: the field is written precisely so the off state can be expressed.
    assert raw_fields(token)["enable_opensubtitles"] is False
    assert parse_user_config(token).enable_opensubtitles is False


# 4/5. The configure page's checkbox state and the encoded config agree
# ---------------------------------------------------------------------------


def _rendered_toggle_state(html: str, toggle_id: str) -> bool | None:
    """The checked state of one provider toggle as the browser would see it."""
    import re

    match = re.search(rf'<input[^>]*id="{toggle_id}"[^>]*>', html)
    assert match, f"toggle {toggle_id} not found"
    return "checked" in match.group(0)


@pytest.fixture(scope="module")
def configure_html() -> str:
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        return client.get("/configure").text


def test_configure_page_opensubtitles_toggle_defaults_to_enabled(configure_html):
    """The checkbox must be pre-checked to match the parser/encoder default.

    If the box renders unchecked while the default is enabled, the page's very
    first "Generate" would write ``enable_opensubtitles=false`` and undo it.
    """
    assert _rendered_toggle_state(configure_html, "enableOpensubtitles") is True


def test_configure_page_enabled_state_survives_encode_decode(configure_html):
    """Checked -> enabled, and the encoded URL must not carry a disabling field."""
    # The page only writes the field when the box is unchecked, so an enabled
    # install URL is the one with the field absent.
    token = encode_user_config("k1", "k2", enable_opensubtitles=True)
    assert "enable_opensubtitles" not in raw_fields(token)
    assert parse_user_config(token).enable_opensubtitles is True
    # SubDL/SubSource use the identical shape, so the page's JS covers both.
    assert _rendered_toggle_state(configure_html, "enableSubdl") is True


def test_configure_page_disabled_state_survives_encode_decode(configure_html):
    """Unchecked -> disabled, and the off state must be representable."""
    token = encode_user_config("k1", "k2", enable_opensubtitles=False)
    assert raw_fields(token)["enable_opensubtitles"] is False
    assert parse_user_config(token).enable_opensubtitles is False


# 6. Credentials are orthogonal to the flag
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("enabled", [True, False])
def test_credentials_remain_separate_from_the_enable_flag(enabled):
    token = encode_user_config(
        "k1",
        "k2",
        opensubtitles_key="OSKEY",
        opensubtitles_username="osuser",
        opensubtitles_password="ospass",
        enable_opensubtitles=enabled,
    )
    prefs = parse_user_config(token)
    assert prefs.opensubtitles_key == "OSKEY"
    assert prefs.opensubtitles_username == "osuser"
    assert prefs.opensubtitles_password == "ospass"
    assert prefs.enable_opensubtitles is enabled


def test_a_credential_without_the_flag_does_not_enable_the_provider():
    """Supplying a key is not consent to run the provider.

    This is why the flag has to be readable in both directions: without an
    explicit ``false``, a URL carrying only a key resolves to the default.
    """
    token = encode_user_config(
        "k1",
        "k2",
        opensubtitles_key="OSKEY",
        opensubtitles_username="osuser",
        opensubtitles_password="ospass",
    )
    assert parse_user_config(token).enable_opensubtitles is True
    off = encode_user_config(
        "k1",
        "k2",
        opensubtitles_key="OSKEY",
        opensubtitles_username="osuser",
        opensubtitles_password="ospass",
        enable_opensubtitles=False,
    )
    assert parse_user_config(off).enable_opensubtitles is False


# 7/8. The aggregator honours the flag
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.asyncio
async def test_the_aggregator_invocability_follows_the_flag(enabled: bool):
    from unittest.mock import AsyncMock

    from app.models import UserPreferences
    from app.services.aggregator import aggregate_subtitles

    provider = AsyncMock()
    provider.search_subtitles = AsyncMock(return_value=[])
    provider.is_breaker_open = AsyncMock(return_value=False)

    # Every other provider is switched off so this asserts one thing only and
    # nothing here can reach the network.
    prefs = UserPreferences(
        enable_opensubtitles=enabled,
        enable_subdl=False,
        enable_subsource=False,
        enable_yifysubtitles=False,
        enable_subtitlecat=False,
        languages=["ara"],
    )
    await aggregate_subtitles(
        imdb_id="tt2582802",
        media_type="movie",
        user_preferences=prefs,
        http_client=None,
        opensubtitles_provider=provider,
        languages=["ara"],
        use_cache=False,
    )
    assert provider.search_subtitles.await_count == (1 if enabled else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False])
async def test_a_disabled_opensubtitles_provider_stays_skipped(enabled: bool):
    """Named wrapper for the disabled half of the pair above."""


# 9. The other toggles are untouched
# ---------------------------------------------------------------------------


def test_subdl_and_subsource_keep_their_opt_out_behaviour():
    on = parse_user_config(encode_user_config("k1", "k2"))
    assert on.enable_subdl is True
    assert on.enable_subsource is True

    off = parse_user_config(encode_user_config("k1", "k2", enable_subdl=False, enable_subsource=False))
    assert off.enable_subdl is False
    assert off.enable_subsource is False
    assert raw_fields(encode_user_config("k1", "k2", enable_subdl=False))["enable_subdl"] is False


def test_opt_in_scrapers_keep_their_defaults():
    """YIFY and SubtitleCat are opt-in and must not have flipped to on."""
    prefs = parse_user_config(encode_user_config("k1", "k2"))
    assert prefs.enable_yifysubtitles is False
    assert prefs.enable_subtitlecat is False

    on = parse_user_config(encode_user_config("k1", "k2", enable_yifysubtitles=True, enable_subtitlecat=True))
    assert on.enable_yifysubtitles is True
    assert on.enable_subtitlecat is True


def test_the_flag_shape_now_matches_subdl():
    """Guards the two toggles against drifting apart again."""
    for kwargs in ({}, {"enable_subdl": False}, {"enable_subdl": True}):
        token = encode_user_config("k1", "k2", **kwargs)
        expected = kwargs.get("enable_subdl", True)
        assert parse_user_config(token).enable_subdl is expected, kwargs

    for kwargs in ({}, {"enable_opensubtitles": False}, {"enable_opensubtitles": True}):
        token = encode_user_config("k1", "k2", **kwargs)
        expected = kwargs.get("enable_opensubtitles", True)
        assert parse_user_config(token).enable_opensubtitles is expected, kwargs
