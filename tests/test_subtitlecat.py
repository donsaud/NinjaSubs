"""Unit tests for the SubtitleCat (subtitlecat.com) provider."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.providers.subtitlecat import SubtitlecatProvider
from app.utils.language import get_subtitlecat_lang_codes

SEARCH_HTML = """
<html><body>
<table class="sub-table">
<tbody>
<tr>
<td><a href="subs/1634/The.Shawshank.Redemption.1994.1080p.x264.YIFY-eng.html">The.Shawshank.Redemption.1994.1080p.x264.YIFY-eng</a> (translated from English)</td>
<td class="sub-table__stars">&nbsp;</td>
</tr>
<tr>
<td><a href="subs/8/The.Shawshank.Redemption.1994.1080p.x264.YIFY.html">The.Shawshank.Redemption.1994.1080p.x264.YIFY</a></td>
<td class="sub-table__stars">&nbsp;</td>
</tr>
</tbody>
</table>
</body></html>
"""

DETAIL_HTML = """
<html><body>
<div class="all-sub">
  <div class="sub-single">
    <span>Arabic</span>
    <a href="/subs/1657/The.Shawshank.Redemption.1994.1080p.x264.YIFY-eng-ar.srt">Download</a>
  </div>
  <div class="sub-single">
    <span>English</span>
    <a href="/subs/1642/The.Shawshank.Redemption.1994.1080p.x264.YIFY-eng-en.srt">Download</a>
  </div>
</div>
</body></html>
"""


def _response(status_code: int = 200, text: str = "", content: bytes = b"") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.content = content
    return resp


def _mock_client(search_html: str = SEARCH_HTML, detail_html: str = DETAIL_HTML) -> AsyncMock:
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    def side_effect(url, params=None, headers=None, timeout=None, **kwargs):
        if "index.php" in url:
            return _response(200, text=search_html)
        if url.endswith(".srt"):
            return _response(200, content=b"1\n00:00:01,000 --> 00:00:02,000\nHello\n")
        # Only the first search candidate (id 1634) exposes translations.
        if "/subs/1634/" in url:
            return _response(200, text=detail_html)
        return _response(200, text="<html><body>no translations</body></html>")

    mock_client.get.side_effect = side_effect
    return mock_client


def test_subtitlecat_lang_codes():
    """Language-code mapping includes ISO-639-1 and legacy variants."""
    assert "ar" in get_subtitlecat_lang_codes("ara")
    assert "en" in get_subtitlecat_lang_codes("eng")
    assert "iw" in get_subtitlecat_lang_codes("heb")
    assert "pt-br" in get_subtitlecat_lang_codes("por")


@pytest.mark.asyncio
async def test_subtitlecat_search_arabic():
    """Arabic search returns the pre-generated -ar.srt translation."""
    mock_client = _mock_client()
    provider = SubtitlecatProvider(mock_client)

    subs = await provider.search_subtitles(
        imdb_id="tt0111161",
        is_series=False,
        title="The Shawshank Redemption",
        year=1994,
        languages=["ara"],
    )

    assert len(subs) == 1
    sub = subs[0]
    assert sub.provider == "subtitlecat"
    assert sub.lang == "ara"
    assert sub.release_name == "The.Shawshank.Redemption.1994.1080p.x264.YIFY-eng"
    assert sub.download_url.endswith(".srt")
    assert sub.download_url.endswith("-eng-ar.srt")
    assert sub.download_url.startswith("https://www.subtitlecat.com/subs/")


@pytest.mark.asyncio
async def test_subtitlecat_search_english():
    """English search picks the -en.srt translation."""
    mock_client = _mock_client()
    provider = SubtitlecatProvider(mock_client)

    subs = await provider.search_subtitles(
        imdb_id="tt0111161",
        is_series=False,
        title="The Shawshank Redemption",
        year=1994,
        languages=["eng"],
    )

    assert len(subs) == 1
    assert subs[0].lang == "eng"
    assert subs[0].download_url.endswith("-eng-en.srt")


@pytest.mark.asyncio
async def test_subtitlecat_no_title_returns_empty():
    """Without a title (e.g. Cinemeta failed) no request is made."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    provider = SubtitlecatProvider(mock_client)

    subs = await provider.search_subtitles(imdb_id="tt0111161", is_series=False, title=None)
    assert subs == []
    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_subtitlecat_no_results():
    """A search page with no result rows yields no releases."""
    mock_client = _mock_client(search_html="<html><body>No results</body></html>")
    provider = SubtitlecatProvider(mock_client)

    subs = await provider.search_subtitles(
        imdb_id="tt0111161", is_series=False, title="Unknown Movie", languages=["ara"]
    )
    assert subs == []


@pytest.mark.asyncio
async def test_subtitlecat_series_query_and_download():
    """Series queries build an SxxExx query and downloads return raw srt bytes."""
    captured = {}
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    def side_effect(url, params=None, headers=None, timeout=None, **kwargs):
        if "index.php" in url:
            captured["params"] = params
            return _response(200, text=SEARCH_HTML)
        if url.endswith(".srt"):
            return _response(200, content=b"1\n00:00:01,000 --> 00:00:02,000\nHello\n")
        if "/subs/1634/" in url:
            return _response(200, text=DETAIL_HTML)
        return _response(200, text="<html><body>no translations</body></html>")

    mock_client.get.side_effect = side_effect
    provider = SubtitlecatProvider(mock_client)

    subs = await provider.search_subtitles(
        imdb_id="tt0903747",
        is_series=True,
        season=5,
        episode=16,
        title="Breaking Bad",
        languages=["ara"],
    )
    assert captured["params"]["search"] == "Breaking Bad S05E16"
    assert len(subs) == 1

    data = await provider.download_archive(subs[0].download_url)
    assert data is not None
    assert b"Hello" in data
