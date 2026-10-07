"""Observed iSubtitles HTML shapes, with no live test dependency."""
import io
import zipfile

import httpx
import pytest

from app.extractor import extract_subtitle_from_archive
from app.providers.isubtitles import IsubtitlesProvider

SEARCH = '<h3><a href="/media-subtitles">Media (2014)</a></h3><a href="https://www.imdb.com/title/tt2582802">IMDb</a>'


def listing(release="Media.2014.BluRay-GROUP", language="Arabic", imdb="tt2582802"):
    return f'''<a href="https://www.imdb.com/title/{imdb}">IMDb</a><table><tr>
    <td data-title="Download"><a href="/download/media/arabic/123">Download</a></td>
    <td data-title="Language"><a>{language}</a></td>
    <td class="movie-release" data-title="Release / Movie"><a>{release}</a></td>
    </tr></table>'''


def archive():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("media.srt", "1\n00:00:01,000 --> 00:00:02,000\nHello.\n")
    return buffer.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize("language,iso", [("Arabic", "ara"), ("English", "eng")])
async def test_movie_metadata_language_release_and_zip(language, iso):
    def handle(request):
        if request.url.path == "/search":
            assert request.url.params["kwd"] == "Media"
            return httpx.Response(200, text=SEARCH)
        if request.url.path == "/media-subtitles":
            return httpx.Response(200, text=listing(language=language))
        return httpx.Response(200, content=archive())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        provider = IsubtitlesProvider(client)
        results = await provider.search_subtitles("tt2582802", title="Media", year=2014,
                                                  languages=[iso])
        assert len(results) == 1
        result = results[0]
        assert result.lang == iso and result.release_name == "Media.2014.BluRay-GROUP"
        assert not result.is_hash_match and not result.matched_by_hash
        raw = await provider.download_archive(result.download_url)
        assert raw == archive()
        assert b"-->" in extract_subtitle_from_archive(raw)


@pytest.mark.asyncio
@pytest.mark.parametrize("release,expected", [("Media.S01E01.HDTV-GROUP", 1),
                                             ("Media.S01E02.HDTV-GROUP", 0),
                                             ("Media.S01E01-E02.HDTV-GROUP", 0),
                                             ("Media.Season.1.Pack", 0)])
async def test_episode_identity(release, expected):
    def handle(request):
        return httpx.Response(200, text=SEARCH if request.url.path == "/search" else listing(release))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        results = await IsubtitlesProvider(client).search_subtitles(
            "tt2582802", True, 1, 1, title="Media", languages=["ara"])
        assert len(results) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("response", ["", "unexpected HTML", "<h3>broken", "captcha"])
async def test_empty_malformed_challenge(response):
    async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda _: httpx.Response(200, text=response))) as client:
        assert not await IsubtitlesProvider(client).search_subtitles("tt2582802", title="Media")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [404, 429, httpx.ReadTimeout, httpx.ConnectError])
async def test_fail_open(failure):
    def handle(request):
        if isinstance(failure, int):
            return httpx.Response(failure)
        raise failure("mock", request=request)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        provider = IsubtitlesProvider(client)
        assert not await provider.search_subtitles("tt2582802", title="Media")
        assert await provider.download_archive("/download/media/english/123") is None


@pytest.mark.asyncio
async def test_wrong_media_and_unsupported_download():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(
            200, text=SEARCH if request.url.path == "/search" else listing(imdb="tt12345")))) as client:
        provider = IsubtitlesProvider(client)
        assert not await provider.search_subtitles("tt2582802", title="Media")
        assert await provider.download_archive("/download/media/english/123") is None
        assert await provider.download_archive("https://private.invalid/sub.zip") is None


@pytest.mark.asyncio
async def test_raw_srt_download():
    raw = b"1\n00:00:01,000 --> 00:00:02,000\nHello.\n"
    async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda _: httpx.Response(200, content=raw))) as client:
        assert await IsubtitlesProvider(client).download_archive(
            "/download/media/english/123") == raw
