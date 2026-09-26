"""Tests for the SubDL 429 circuit breaker (fast-bypass without network)."""

from unittest.mock import AsyncMock

import httpx
import pytest

from app.providers.subdl import SUBDL_BREAKER, SubdlProvider


@pytest.mark.asyncio
async def test_breaker_open_fast_bypasses_without_network():
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = httpx.Response(200, json={"status": True, "subtitles": []})
    provider = SubdlProvider(client)

    SUBDL_BREAKER.trip()
    try:
        assert await provider.search_subtitles(imdb_id="tt1", api_key="k") == []
        assert await provider.download_archive("https://dl.subdl.com/x.zip", api_key="k") is None
        client.get.assert_not_awaited()
    finally:
        SUBDL_BREAKER.reset()


@pytest.mark.asyncio
async def test_429_trips_breaker_and_bypasses_next_call():
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = httpx.Response(429, text="rate limited")
    provider = SubdlProvider(client)

    assert await provider.search_subtitles(imdb_id="tt1", api_key="k") == []
    assert SUBDL_BREAKER.is_open() is True
    assert SUBDL_BREAKER.remaining > 0

    client.get.reset_mock()
    assert await provider.search_subtitles(imdb_id="tt1", api_key="k") == []
    client.get.assert_not_awaited()
    SUBDL_BREAKER.reset()


@pytest.mark.asyncio
async def test_download_429_trips_breaker():
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = httpx.Response(429, text="rate limited")
    provider = SubdlProvider(client)

    assert await provider.download_archive("https://dl.subdl.com/x.zip", api_key="k") is None
    assert SUBDL_BREAKER.is_open() is True
    SUBDL_BREAKER.reset()


def test_breaker_reset_closes():
    SUBDL_BREAKER.trip()
    assert SUBDL_BREAKER.is_open() is True
    SUBDL_BREAKER.reset()
    assert SUBDL_BREAKER.is_open() is False
