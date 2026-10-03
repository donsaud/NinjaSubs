"""OpenSubtitles through Stremio's keyless v3 endpoint.

The endpoint's contract, verified against the live service:
``GET https://opensubtitles-v3.strem.io/subtitles/{type}/{id}/{extra}.json``
returns ``{"subtitles": [...]}`` where each entry carries a **direct** ``url``.

Two behaviours these tests pin, because both were confirmed by probing and
neither is obvious from the docs:

* it does **not** filter by language, so the filter is ours to apply;
* it does **not** filter on the hash and reports no hash, so no hash match may
  be claimed from it.

Every test here mocks the endpoint. None of them reach the network.
"""

from __future__ import annotations

import httpx
import pytest

from app.models import SubtitleRelease
from app.providers.opensubtitles import (
    OPENSUBTITLES_BREAKER,
    OpenSubtitlesProvider,
    _normalize_language,
)


@pytest.fixture(autouse=True)
def _reset_breaker():
    OPENSUBTITLES_BREAKER.reset()
    yield
    OPENSUBTITLES_BREAKER.reset()


def _item(
    *,
    sub_id: str = "1",
    lang: str = "ara",
    name: str = "Movie.2024.1080p.WEB-DL-GROUP.srt",
    release: str | None = None,
    url: str | None = None,
) -> dict:
    return {
        "id": sub_id,
        "url": url or f"https://subs5.strem.io/en/download/file/{sub_id}",
        "lang": lang,
        "subtitleFileName": name,
        "movieReleaseName": release if release is not None else name.rsplit(".srt", 1)[0],
        "releaseGroup": "GROUP",
        "releaseFormat": "WEB-DL",
        "SubEncoding": "CP1256",
        "fpsMilli": 23976,
        "season": 0,
        "episode": 0,
    }


class StubClient:
    """Records the requested URL and returns a canned v3 payload."""

    def __init__(self, payload: dict, status: int = 200) -> None:
        self.payload = payload
        self.status = status
        self.urls: list[str] = []

    async def get(self, url: str, headers=None, **kwargs) -> httpx.Response:
        self.urls.append(url)
        return httpx.Response(
            self.status,
            json=self.payload,
            request=httpx.Request("GET", url),
        )


def provider(client) -> OpenSubtitlesProvider:
    return OpenSubtitlesProvider(client)


# --- URL construction ---------------------------------------------------------


@pytest.mark.asyncio
async def test_the_url_targets_the_keyless_v3_endpoint():
    client = StubClient({"subtitles": []})
    await provider(client).search_subtitles(imdb_id="tt2582802", languages=["ara"])

    url = client.urls[0]
    assert url.startswith("https://opensubtitles-v3.strem.io/subtitles/movie/tt2582802")
    assert url.endswith(".json")
    # No credential anywhere in the request.
    assert "Api-Key" not in (provider(client)._get_headers())
    assert "apikey" not in url.lower()
    assert "api_key" not in url.lower()


@pytest.mark.asyncio
async def test_stream_parameters_are_passed_through():
    """filename / videoSize / videoHash reach the endpoint, as the spec requires."""
    client = StubClient({"subtitles": []})
    await provider(client).search_subtitles(
        imdb_id="tt2582802",
        languages=["ara"],
        video_hash="8e245d9679d31e12",
        video_size=19571049411,
        filename="Whiplash.2014.2160p.UHD.mkv",
    )

    url = client.urls[0]
    assert "videoHash%3D8e245d9679d31e12" in url
    assert "videoSize%3D19571049411" in url
    assert "filename%3DWhiplash" in url


@pytest.mark.asyncio
async def test_caller_extra_params_are_passed_through_untouched():
    client = StubClient({"subtitles": []})
    await provider(client).search_subtitles(
        imdb_id="tt2582802", languages=["ara"], extra_params={"encoding": "cp1256"}
    )

    assert "encoding%3Dcp1256" in client.urls[0]


@pytest.mark.asyncio
async def test_a_series_request_uses_the_series_type_and_episode_query():
    client = StubClient({"subtitles": []})
    await provider(client).search_subtitles(
        imdb_id="tt0111161", is_series=True, season=2, episode=3, languages=["eng"]
    )

    url = client.urls[0]
    assert "/subtitles/series/tt0111161" in url
    assert "season=2" in url and "episode=3" in url


@pytest.mark.asyncio
async def test_a_compound_imdb_id_is_reduced_to_the_base_id():
    client = StubClient({"subtitles": []})
    await provider(client).search_subtitles(
        imdb_id="tt0111161:2:3", is_series=True, season=2, episode=3, languages=["eng"]
    )

    assert "/subtitles/series/tt0111161" in client.urls[0]


@pytest.mark.asyncio
async def test_a_stale_api_key_is_accepted_and_ignored():
    """An old install URL may still carry a key. That must not break search."""
    client = StubClient({"subtitles": [_item()]})
    releases = await provider(client).search_subtitles(
        imdb_id="tt1", api_key="legacy-key", languages=["ara"]
    )

    assert len(releases) == 1


# --- language filtering, which the endpoint does not do -----------------------


@pytest.mark.asyncio
async def test_results_are_filtered_to_the_requested_language():
    """The endpoint returns every language; filtering happens here."""
    client = StubClient(
        {
            "subtitles": [
                _item(sub_id="1", lang="ara"),
                _item(sub_id="2", lang="eng"),
                _item(sub_id="3", lang="fre"),
                _item(sub_id="4", lang="ara"),
            ]
        }
    )
    releases = await provider(client).search_subtitles(
        imdb_id="tt1", languages=["ara"]
    )

    assert [r.download_url.split("/")[-1] for r in releases] == ["1", "4"]
    assert all(r.lang == "ara" for r in releases)


@pytest.mark.asyncio
async def test_multiple_requested_languages_are_all_kept():
    client = StubClient(
        {"subtitles": [_item(sub_id="1", lang="ara"), _item(sub_id="2", lang="eng")]}
    )
    releases = await provider(client).search_subtitles(
        imdb_id="tt1", languages=["ara", "eng"]
    )

    assert len(releases) == 2


def test_language_normalisation_handles_the_codes_the_endpoint_emits():
    assert _normalize_language("ara") == "ara"
    assert _normalize_language("ARA") == "ara"
    assert _normalize_language("") == ""
    # Three-letter codes the endpoint uses, including ones this codebase does not
    # have a mapping for, must still round-trip rather than collapse.
    assert _normalize_language("zho") == "zho"
    assert _normalize_language("srp") == "srp"


# --- mapping to SubtitleRelease ----------------------------------------------


@pytest.mark.asyncio
async def test_a_result_maps_to_a_directly_downloadable_release():
    client = StubClient(
        {
            "subtitles": [
                _item(
                    sub_id="1954576302",
                    name="Whiplash.2014.720p.WEB-DL-RARBG.srt",
                    release="Whiplash.2014.720p.WEB-DL.AAC2.0.H264-RARBG",
                )
            ]
        }
    )
    release = (await provider(client).search_subtitles(imdb_id="tt1", languages=["ara"]))[0]

    assert release.provider == "opensubtitles"
    assert release.download_url == "https://subs5.strem.io/en/download/file/1954576302"
    assert release.release_name == "Whiplash.2014.720p.WEB-DL.AAC2.0.H264-RARBG"
    assert release.format == "srt"
    assert release.lang == "ara"


@pytest.mark.asyncio
async def test_the_release_name_falls_back_to_the_filename_then_the_id():
    client = StubClient({"subtitles": [{"id": "7", "url": "https://x/y", "lang": "ara"}]})
    release = (await provider(client).search_subtitles(imdb_id="tt1", languages=["ara"]))[0]

    assert "tt1" in release.release_name


@pytest.mark.asyncio
async def test_no_hash_match_is_claimed():
    """The endpoint neither filters on the hash nor reports one.

    Claiming a match would promote the release into the hash tier and to the top
    of the list on evidence that does not exist.
    """
    client = StubClient({"subtitles": [_item(lang="ara")]})
    release = (
        await provider(client).search_subtitles(
            imdb_id="tt1", languages=["ara"], video_hash="8e245d9679d31e12",
            video_size=19571049411,
        )
    )[0]

    assert release.is_hash_match is False
    assert release.matched_by_hash is False


@pytest.mark.asyncio
async def test_entries_without_a_usable_url_are_dropped():
    client = StubClient(
        {
            "subtitles": [
                _item(sub_id="1", lang="ara"),
                {"id": "2", "lang": "ara"},  # no url
                {"id": "3", "url": "javascript:alert(1)", "lang": "ara"},
            ]
        }
    )
    releases = await provider(client).search_subtitles(imdb_id="tt1", languages=["ara"])

    assert len(releases) == 1


@pytest.mark.asyncio
async def test_hearing_impaired_releases_are_excluded_on_request():
    client = StubClient(
        {
            "subtitles": [
                _item(sub_id="1", name="Movie.sdh.srt", release="Movie.SDH"),
                _item(sub_id="2", name="Movie.srt", release="Movie"),
            ]
        }
    )
    releases = await provider(client).search_subtitles(
        imdb_id="tt1", languages=["ara"], exclude_hi=True
    )

    assert [r.release_name for r in releases] == ["Movie"]


@pytest.mark.asyncio
async def test_a_malformed_payload_yields_no_results_rather_than_raising():
    for payload in ({}, {"subtitles": None}, {"subtitles": "nope"}, []):
        client = StubClient(payload)
        assert await provider(client).search_subtitles(imdb_id="t", languages=["ara"]) == []


@pytest.mark.asyncio
async def test_a_429_trips_the_breaker_and_a_later_call_short_circuits():
    client = StubClient({"subtitles": []}, status=429)
    assert await provider(client).search_subtitles(imdb_id="t", languages=["ara"]) == []
    assert OPENSUBTITLES_BREAKER.is_open() is True

    calls = len(client.urls)
    assert await provider(client).search_subtitles(imdb_id="t", languages=["ara"]) == []
    assert len(client.urls) == calls, "breaker should have prevented the second request"


@pytest.mark.asyncio
async def test_a_transport_error_yields_no_results():
    class Broken:
        async def get(self, url, headers=None, **kwargs):
            raise httpx.ConnectError("boom")

    assert await provider(Broken()).search_subtitles(imdb_id="t", languages=["ara"]) == []


# --- download -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_fetches_the_direct_url():
    payload = b"1\n00:00:01,000 --> 00:00:02,000\nhello\n"

    class DownloadClient:
        def __init__(self) -> None:
            self.urls: list[str] = []

        async def get(self, url, headers=None, **kwargs) -> httpx.Response:
            self.urls.append(url)
            return httpx.Response(
                200, content=payload, request=httpx.Request("GET", url)
            )

    client = DownloadClient()
    got = await OpenSubtitlesProvider(client).download_archive(
        "https://subs5.strem.io/en/download/file/1954576302"
    )

    assert got == payload
    assert client.urls == ["https://subs5.strem.io/en/download/file/1954576302"]


@pytest.mark.asyncio
async def test_a_legacy_numeric_reference_cannot_be_resolved():
    """A bare file_id came from the old authenticated API.

    The v3 endpoint has no file_id to negotiate, so this has to be a clean miss
    rather than a request to the wrong host.
    """
    client = StubClient({"subtitles": []})
    assert await OpenSubtitlesProvider(client).download_archive("1954576302") is None


@pytest.mark.asyncio
async def test_get_download_url_is_a_no_op():
    assert await OpenSubtitlesProvider(StubClient({})).get_download_url(123) is None


def test_the_release_model_requires_a_download_url():
    """A release without one cannot be served, so the field is not optional."""
    assert "download_url" in SubtitleRelease.model_fields
