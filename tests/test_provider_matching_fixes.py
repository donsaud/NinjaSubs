"""Regression tests for provider and release-matching defects.

Covers rate-limit handling (SubDL season-pack retry after a 429, SubSource
search ignoring its breaker, breakers being shortened), episode parsing
(release groups and years read as episodes, four-digit
episodes, movie titles that start with a number), language labelling, and
malformed provider items.
"""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.providers.opensubtitles import OPENSUBTITLES_BREAKER, OpenSubtitlesProvider
from app.providers.subdl import SUBDL_BREAKER, SubdlProvider
from app.providers.subsource import (
    SUBSOURCE_BREAKER,
    SubsourceProvider,
    _matches_series_episode,
)
from app.services.subtitle_matcher import extract_metadata
from app.utils.sami_converter import convert_sami_to_srt

# --------------------------------------------------------------------------- #
# Rate limits
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 401, 403])
async def test_subdl_skips_season_pack_fallback_after_refusal(status):
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = httpx.Response(status, text="refused")
    provider = SubdlProvider(client)

    result = await provider.search_subtitles(
        imdb_id="tt0773262", is_series=True, season=1, episode=5, api_key="k"
    )

    assert result == []
    assert client.get.await_count == 1  # no second, season-pack request


@pytest.mark.asyncio
async def test_subdl_still_tries_season_pack_on_plain_empty_result():
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = httpx.Response(200, json={"status": True, "subtitles": []})
    provider = SubdlProvider(client)

    await provider.search_subtitles(
        imdb_id="tt0773262", is_series=True, season=1, episode=5, api_key="k"
    )

    assert client.get.await_count == 2


@pytest.mark.asyncio
async def test_subsource_search_429_trips_breaker_and_next_search_is_skipped():
    client = AsyncMock(spec=httpx.AsyncClient)
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 429
    resp.text = "rate limited"
    client.get.return_value = resp
    provider = SubsourceProvider(client)

    assert await provider.search_subtitles(imdb_id="tt1", api_key="k") == []
    assert SUBSOURCE_BREAKER.is_open()

    client.get.reset_mock()
    assert await provider.search_subtitles(imdb_id="tt1", api_key="k") == []
    client.get.assert_not_awaited()


@pytest.mark.parametrize("breaker", [SUBDL_BREAKER, SUBSOURCE_BREAKER, OPENSUBTITLES_BREAKER])
def test_breaker_trip_never_shortens_an_open_window(breaker):
    try:
        if breaker is SUBDL_BREAKER:
            breaker.cooldown, original = 3600.0, breaker.cooldown
            breaker.trip()
            breaker.cooldown = 1.0
            breaker.trip()
            breaker.cooldown = original
        else:
            breaker.trip(3600.0)
            breaker.trip(1.0)
        assert breaker.remaining > 3000
    finally:
        breaker.reset()


# --------------------------------------------------------------------------- #
# SubSource episode filter
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name,season,episode,expected",
    [
        ("Dexter.Season.Two.Complete.720p", 1, 5, False),
        ("Show.S01.1080p.BluRay.x264-2HD", 1, 5, True),  # "-2HD" is a group
        ("Doctor Who 2005 - 05", 1, 5, True),  # "2005" is a year
        ("Doctor Who 2005 - 06", 1, 5, False),
        ("Hunter X Hunter-91", 1, 91, True),
        ("Show 001-025", 1, 12, True),
        ("Show 001-025", 1, 30, False),
    ],
)
def test_subsource_episode_filter(name, season, episode, expected):
    assert _matches_series_episode({}, season, episode, name) is expected


# --------------------------------------------------------------------------- #
# Filename metadata
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name,season,episode",
    [
        ("Show.S01E100.720p.HDTV", 1, 100),
        ("One.Piece.S01E1085.1080p.WEB", 1, 1085),
    ],
)
def test_three_and_four_digit_episodes_keep_their_season(name, season, episode):
    meta = extract_metadata(name)
    assert (meta["season"], meta["episode"]) == (season, episode)


@pytest.mark.parametrize(
    "name",
    [
        "300.2006.1080p.BluRay.x264-AMIABLE",
        "21.Jump.Street.2012.1080p.BluRay.x264",
        "Ocean's.11.2001.1080p.BluRay.x264",
    ],
)
def test_movie_title_numbers_are_not_seasons_or_episodes(name):
    meta = extract_metadata(name)
    assert meta["season"] is None
    assert meta["episode"] is None
    assert meta["content_type"] == "movie"


def test_bare_anime_episode_is_still_detected():
    assert extract_metadata("[Animeiat] 02 - Arabic.srt")["episode"] == 2


# --------------------------------------------------------------------------- #
# Language labelling
# --------------------------------------------------------------------------- #


def _subsource_client(items):
    client = AsyncMock(spec=httpx.AsyncClient)

    def side_effect(url, params=None, headers=None, timeout=None):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.text = ""
        if "movies/search" in url:
            resp.json.return_value = {"movieId": 1}
        else:
            resp.json.return_value = {"data": items}
        return resp

    client.get.side_effect = side_effect
    return client


@pytest.mark.asyncio
async def test_subsource_persian_result_is_labelled_persian_not_arabic():
    client = _subsource_client(
        [{"subtitleId": "s1", "releaseInfo": ["Movie.2020.1080p.srt"], "language": "farsi_persian"}]
    )
    subs = await SubsourceProvider(client).search_subtitles(
        imdb_id="tt1", api_key="k", languages=["fas"]
    )
    assert [s.lang for s in subs] == ["fas"]


@pytest.mark.asyncio
async def test_subsource_language_filter_compares_whole_words():
    client = _subsource_client(
        [{"subtitleId": "s1", "releaseInfo": ["Movie.2020.1080p.srt"], "language": "Malayalam"}]
    )
    subs = await SubsourceProvider(client).search_subtitles(
        imdb_id="tt1", api_key="k", languages=["may"]
    )
    assert subs == []


# --------------------------------------------------------------------------- #
# Malformed provider items
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_opensubtitles_malformed_item_does_not_drop_the_list():
    good = {
        "attributes": {
            "release": "Movie.2020.1080p",
            "language": "ar",
            "files": [{"file_id": 7, "file_name": "Movie.2020.1080p.srt"}],
        }
    }
    payload = {"data": [{"attributes": None}, {"attributes": {"files": ["oops"]}}, good]}
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = httpx.Response(
        200, json=payload, request=httpx.Request("GET", "https://api.opensubtitles.com/")
    )

    subs = await OpenSubtitlesProvider(client).search_subtitles(imdb_id="tt1", api_key="k")

    assert [s.release_name for s in subs] == ["Movie.2020.1080p"]


@pytest.mark.asyncio
async def test_subdl_non_string_url_item_is_skipped():
    items = [
        {"release_name": "Bad", "url": 12345},
        {"release_name": "Good.Release", "url": "/subtitle/1-2.zip", "lang": "AR"},
    ]
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = httpx.Response(200, json={"status": True, "subtitles": items})

    subs = await SubdlProvider(client).search_subtitles(imdb_id="tt1", api_key="k")

    assert [s.release_name for s in subs] == ["Good.Release.srt"]


# --------------------------------------------------------------------------- #
# SAMI
# --------------------------------------------------------------------------- #

_SAMI = (
    "<SAMI><BODY><SYNC Start=1000><P Class=ARCC>مرحبا<P Class=ENCC>Hello"
    "<SYNC Start=3000><P Class=ENCC>&nbsp;</BODY></SAMI>"
)


def test_sami_prefers_requested_language_track():
    assert "مرحبا" in convert_sami_to_srt(_SAMI, lang="ara")
    assert "Hello" not in convert_sami_to_srt(_SAMI, lang="ara")


def test_sami_defaults_to_english_track():
    assert "Hello" in convert_sami_to_srt(_SAMI)
