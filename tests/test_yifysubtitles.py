"""Unit tests for the YIFYSubtitles (yifysubtitles.ch) provider."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.providers.yifysubtitles import YifysubtitlesProvider

MOVIE_HTML = """
<html><body>
<table class="table">
<tbody>
<tr data-id="125583">
  <td class="rating-cell"><span class="label">0</span></td>
  <td class="flag-cell"><span class="flag flag-sa"></span><span class="sub-lang">Arabic</span></td>
  <td><a href="/subtitles/the-shawshank-redemption-1994-arabic-yify-125583"><span class="text-muted">subtitle</span> The.Shawshank.Redemption.1994.1080p.BluRay.x264.[YTS.AG]</a></td>
  <td class="other-cell"></td>
  <td class="uploader-cell"><a href="/user/sub">sub</a></td>
</tr>
<tr data-id="125590">
  <td class="rating-cell"><span class="label">5</span></td>
  <td class="flag-cell"><span class="flag flag-us"></span><span class="sub-lang">English</span></td>
  <td><a href="/subtitles/the-shawshank-redemption-1994-english-yify-125590">The.Shawshank.Redemption.1994.720p.WEB-DL</a></td>
  <td class="other-cell"></td>
  <td class="uploader-cell"><a href="/user/joe">joe</a></td>
</tr>
</tbody>
</table>
</body></html>
"""


def _response(status_code: int = 200, text: str = "", content: bytes = b"") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.content = content
    return resp


@pytest.mark.asyncio
async def test_yifysubtitles_movie_parsing_filters_language():
    """Arabic-only request keeps just the Arabic row and maps metadata correctly."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get.return_value = _response(200, text=MOVIE_HTML)

    provider = YifysubtitlesProvider(mock_client)
    subs = await provider.search_subtitles(
        imdb_id="tt0111161",
        is_series=False,
        languages=["ara"],
    )

    assert len(subs) == 1
    sub = subs[0]
    assert sub.provider == "yifysubtitles"
    assert sub.lang == "ara"
    assert sub.release_name == "The.Shawshank.Redemption.1994.1080p.BluRay.x264.[YTS.AG]"
    assert sub.download_url == (
        "https://yifysubtitles.ch/subtitle/"
        "the-shawshank-redemption-1994-arabic-yify-125583.zip"
    )
    assert sub.uploader == "sub"


@pytest.mark.asyncio
async def test_yifysubtitles_multi_release_uses_first_name():
    """Rows listing several releases separated by <br /> keep only the first name."""
    html = """
    <table><tbody>
    <tr data-id="357712">
      <td class="rating-cell"><span class="label">0</span></td>
      <td class="flag-cell"><span class="flag flag-sa"></span><span class="sub-lang">Arabic</span></td>
      <td><a href="/subtitles/the-shawshank-redemption-1994-arabic-yify-357712"><span class="text-muted">subtitle</span> First.Release.1080p.BluRay<br />
Second.Release.720p.BRRip<br />
Third.Release.x264</a></td>
      <td class="other-cell"></td>
      <td class="uploader-cell"><a href="/user/x">x</a></td>
    </tr>
    </tbody></table>
    """
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get.return_value = _response(200, text=html)

    provider = YifysubtitlesProvider(mock_client)
    subs = await provider.search_subtitles(imdb_id="tt0111161", is_series=False, languages=["ara"])

    assert len(subs) == 1
    assert subs[0].release_name == "First.Release.1080p.BluRay"
    assert "\n" not in subs[0].release_name


@pytest.mark.asyncio
async def test_yifysubtitles_multi_language():
    """Requesting Arabic + English returns both rows."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get.return_value = _response(200, text=MOVIE_HTML)

    provider = YifysubtitlesProvider(mock_client)
    subs = await provider.search_subtitles(
        imdb_id="tt0111161",
        is_series=False,
        languages=["ara", "eng"],
    )

    assert {s.lang for s in subs} == {"ara", "eng"}


@pytest.mark.asyncio
async def test_yifysubtitles_series_unsupported():
    """YIFYSubtitles is movies-only: series must short-circuit without requests."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    provider = YifysubtitlesProvider(mock_client)

    subs = await provider.search_subtitles(imdb_id="tt0944947", is_series=True)
    assert subs == []
    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_yifysubtitles_page_not_found():
    """A 404 / 'Page not found' listing returns no results."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get.return_value = _response(404, text="Page not found")

    provider = YifysubtitlesProvider(mock_client)
    subs = await provider.search_subtitles(imdb_id="tt0000000", is_series=False)
    assert subs == []


@pytest.mark.asyncio
async def test_yifysubtitles_download_zip():
    """download_archive returns raw zip bytes on success."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get.return_value = _response(200, content=b"PK\x03\x04zipdata")

    provider = YifysubtitlesProvider(mock_client)
    data = await provider.download_archive(
        "https://yifysubtitles.ch/subtitle/the-shawshank-redemption-1994-arabic-yify-125583.zip"
    )
    assert data == b"PK\x03\x04zipdata"


@pytest.mark.asyncio
async def test_yifysubtitles_download_cloudflare_warmup():
    """On a Cloudflare 403 the provider warms up the session once and retries."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    zip_attempts = {"count": 0}

    def side_effect(url, **kwargs):
        if url.endswith(".zip"):
            zip_attempts["count"] += 1
            if zip_attempts["count"] == 1:
                return _response(403, text="<cf challenge>")
            return _response(200, content=b"PK\x03\x04retried")
        return _response(200, text="<home>")

    mock_client.get.side_effect = side_effect

    provider = YifysubtitlesProvider(mock_client)
    data = await provider.download_archive("https://yifysubtitles.ch/subtitle/x.zip")

    assert data == b"PK\x03\x04retried"
    assert zip_attempts["count"] == 2
