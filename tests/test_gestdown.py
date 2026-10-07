"""Gestdown contract tests; all network responses are deterministic mocks."""
import copy

import httpx
import pytest

from app.extractor import transcode_to_utf8
from app.providers.gestdown import GestdownProvider

SHOW = "31ffb6ce-c000-4079-8912-b3f72057baed"
SID = "69cf7d79-052c-4f12-a57d-995d77de43ad"
ROW = {"subtitleId": SID, "version": "0tv", "release": None, "completed": True,
       "hearingImpaired": False, "language": "English",
       "downloadUri": f"/subtitles/download/{SID}"}
SRT = b"1\n00:00:01,000 --> 00:00:02,000\nHello.\n"


def transport(*, episode=None, shows=None, failure=None):
    calls = []

    def handle(request):
        calls.append(request.url.path)
        if failure is not None:
            if isinstance(failure, int):
                return httpx.Response(failure)
            if failure == "invalid_json":
                return httpx.Response(200, content=b"not JSON")
            raise failure("mock", request=request)
        if request.url.path.startswith("/shows/"):
            return httpx.Response(200, json={"shows": shows if shows is not None else [
                {"id": SHOW, "name": "Breaking Bad", "tvDbId": 81189}]})
        if request.url.path.startswith("/subtitles/get/"):
            return httpx.Response(200, json=episode if episode is not None else {
                "matchingSubtitles": [ROW]})
        return httpx.Response(200, content=SRT)

    return httpx.MockTransport(handle), calls


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["en", "eng", "English"])
async def test_tvdb_episode_language_release_and_download(language):
    mock, calls = transport()
    async with httpx.AsyncClient(transport=mock) as client:
        provider = GestdownProvider(client)
        results = await provider.search_subtitles("tt0903747", True, 1, 1,
                                                  tvdb_id=81189, languages=[language])
        assert len(results) == 1
        result = results[0]
        assert result.release_name == "0tv" and result.lang == "eng"
        assert result.provider == "gestdown" and result.format == "srt"
        assert not result.is_hash_match and not result.matched_by_hash
        assert calls == ["/shows/external/tvdb/81189", f"/subtitles/get/{SHOW}/1/1/eng"]
        raw = await provider.download_archive(result.download_url)
        assert raw == SRT and transcode_to_utf8(raw, lang="eng") == SRT


@pytest.mark.asyncio
async def test_exact_title_fallback_and_ambiguous_show_refusal():
    mock, calls = transport()
    async with httpx.AsyncClient(transport=mock) as client:
        assert await GestdownProvider(client).search_subtitles(
            "tt0903747", True, 1, 1, title="Breaking Bad", languages=["eng"])
        assert calls[0] == "/shows/search/Breaking Bad"
    mock, calls = transport(shows=[{"id": SHOW, "name": "Breaking Bad"}] * 2)
    async with httpx.AsyncClient(transport=mock) as client:
        assert not await GestdownProvider(client).search_subtitles(
            "tt0903747", True, 1, 1, title="Breaking Bad")
        assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [404, 429, 500, "invalid_json", httpx.ReadTimeout,
                                     httpx.ConnectError])
async def test_search_and_download_fail_open(failure):
    mock, _ = transport(failure=failure)
    async with httpx.AsyncClient(transport=mock) as client:
        provider = GestdownProvider(client)
        assert not await provider.search_subtitles("tt0903747", True, 1, 1, tvdb_id=81189)
        assert await provider.download_archive(f"/subtitles/download/{SID}") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("rows", [[], None, "unexpected", [{}], [{**ROW, "completed": False}],
                                  [{**ROW, "language": "Arabic"}],
                                  [{**ROW, "language": {"unexpected": True}}],
                                  [{**ROW, "downloadUri": "https://private.invalid"}]])
async def test_empty_malformed_or_wrong_language(rows):
    mock, _ = transport(episode={"matchingSubtitles": rows})
    async with httpx.AsyncClient(transport=mock) as client:
        assert not await GestdownProvider(client).search_subtitles(
            "tt0903747", True, 1, 1, tvdb_id=81189, languages=["eng"])


@pytest.mark.asyncio
async def test_release_and_hi_filter():
    row = copy.deepcopy(ROW)
    row.update(release="Breaking.Bad.S01E01.HDTV-GROUP", hearingImpaired=True)
    mock, _ = transport(episode={"matchingSubtitles": [row]})
    async with httpx.AsyncClient(transport=mock) as client:
        provider = GestdownProvider(client)
        args = dict(imdb_id="tt0903747", is_series=True, season=1, episode=1, tvdb_id=81189)
        assert (await provider.search_subtitles(**args))[0].release_name == row["release"]
        assert not await provider.search_subtitles(**args, exclude_hi=True)


@pytest.mark.asyncio
async def test_movies_missing_identity_and_unsafe_download_make_no_requests():
    mock, calls = transport()
    async with httpx.AsyncClient(transport=mock) as client:
        provider = GestdownProvider(client)
        assert not await provider.search_subtitles("tt2582802", title="Whiplash", tvdb_id=81189)
        assert not await provider.search_subtitles("tt0903747", True, 1, 1)
        assert await provider.download_archive("https://private.invalid/file.srt") is None
        assert calls == []


@pytest.mark.asyncio
async def test_unknown_tvdb_show():
    mock, calls = transport(shows=[])
    async with httpx.AsyncClient(transport=mock) as client:
        assert not await GestdownProvider(client).search_subtitles("tt0903747", True, 1, 1,
                                                                  tvdb_id=81189)
    assert len(calls) == 1
