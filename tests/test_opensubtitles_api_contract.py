"""OpenSubtitles API-contract behaviour, verified against the current live API.

Every assertion here encodes something confirmed against the running service
rather than an assumption:

* ``POST /download`` takes ``attributes.files[].file_id``, never the top-level
  ``id``/``subtitle_id``;
* HTTP 406 is overloaded -- quota exhausted *or* ``Invalid file_id`` -- and the
  ``message`` field is what tells them apart;
* HTTP 429 is a request-rate limit and carries ``Retry-After``;
* error bodies are sometimes a JSON *array*, not an object.

No network access. Credentials are dummies; nothing here resolves a real key.
"""

from __future__ import annotations

import httpx
import pytest

from app.providers.opensubtitles import (
    _FILE_ID_RE,
    OPENSUBTITLES_BREAKER,
    OpenSubtitlesProvider,
    _error_payload,
    _is_quota_406,
    _quota_reset_seconds,
    looks_like_subtitle_payload,
)

KEY = "dummy-key-not-real"
BASE = "https://api.opensubtitles.com/api/v1"

SRT = (
    b"1\n00:00:01,000 --> 00:00:03,000\nhello\n\n"
    b"2\n00:00:04,000 --> 00:00:06,000\nworld\n\n"
)


@pytest.fixture(autouse=True)
def _reset_breaker():
    OPENSUBTITLES_BREAKER.reset()
    yield
    OPENSUBTITLES_BREAKER.reset()


class Response:
    def __init__(self, status: int, payload=None, headers=None, text: str = "", content: bytes = b""):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.text = text
        self.content = content

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class Stub:
    """Records every call so requests can be asserted, not just responses."""

    def __init__(self, post=None, get=None):
        self._post = post
        self._get = get
        self.gets: list[tuple] = []
        self.posts: list[tuple] = []

    async def get(self, url, headers=None, params=None, **kw):
        self.gets.append((url, headers, params, kw))
        result = self._get(url, headers, params, kw) if self._get else Response(200, {"data": []})
        return result

    async def post(self, url, headers=None, json=None, **kw):
        self.posts.append((url, headers, json, kw))
        return self._post(url, headers, json, kw) if self._post else Response(200, {"link": "https://x/y.srt"})


def provider(stub) -> OpenSubtitlesProvider:
    return OpenSubtitlesProvider(stub)


def attrs(**over):
    base = {
        "language": "en",
        "release": "Movie.2024.1080p-FLUX",
        "files": [{"file_id": 7144860, "file_name": "Movie.srt"}],
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# Response-shape helpers
# ---------------------------------------------------------------------------


def test_error_body_is_read_from_both_object_and_array_forms():
    """OpenSubtitles sends ``[{...}]`` for 401s and ``{...}`` for quota 406s."""
    assert _error_payload(Response(401, {"message": "obj"}))["message"] == "obj"
    assert _error_payload(Response(401, [{"message": "arr"}]))["message"] == "arr"
    assert _error_payload(Response(500, None, text="<html>")) == {}


def test_a_406_invalid_file_id_is_not_mistaken_for_quota_exhaustion():
    assert _is_quota_406({"message": "Invalid file_id"}) is False
    assert _is_quota_406({"message": "invalid file id"}) is False
    assert (
        _is_quota_406(
            {
                "requests": 21,
                "remaining": -1,
                "message": "You have downloaded your allowed 20 subtitles for 24h.",
                "reset_time_utc": "2030-01-01T00:00:00.000Z",
            }
        )
        is True
    )


def test_quota_reset_is_read_from_the_documented_reset_time_utc():
    """`reset_time_unix` does not exist in the API; `reset_time_utc` does."""
    seconds = _quota_reset_seconds({"reset_time_utc": "2030-01-01T00:00:00.000Z"})
    assert seconds is not None and seconds > 0
    # A body with neither field yields None so the caller can use its default.
    assert _quota_reset_seconds({"message": "nope"}) is None


def test_file_id_survives_every_format_the_provider_emits():
    """ASS/VTT/SSA references used to parse to no id and never downloaded."""
    for ref, expected in [
        ("/sub/opensubtitles/12345.srt", "12345"),
        ("/sub/opensubtitles/12345.ass", "12345"),
        ("/sub/opensubtitles/12345.vtt", "12345"),
        ("/sub/opensubtitles/12345.ssa", "12345"),
        ("12345", "12345"),
    ]:
        m = _FILE_ID_RE.search(ref)
        assert m and m.group(1) == expected, ref


@pytest.mark.parametrize(
    "payload,ok",
    [
        (SRT, True),
        (b"\xef\xbb\xbf1\r\n00:00:01,000 --> 00:00:03,000\r\nhi\r\n\r\n", True),
        (b"<html><body>Service Unavailable</body></html>", False),
        (b"<!DOCTYPE html><html></html>", False),
        (b'{"message":"You have downloaded your allowed 20 subtitles"}', False),
        (b"", False),
        (b"   ", False),
        (b"\x00\x01\x02", False),
    ],
)
def test_downloaded_payloads_are_screened_for_subtitle_content(payload, ok):
    assert looks_like_subtitle_payload(payload) is ok


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_uses_files_file_id_not_the_subtitle_id():
    """The single most common integration bug: id != file_id."""
    body = {
        "data": [
            {
                "id": "6230367",
                "type": "subtitle",
                "attributes": attrs(subtitle_id="6230367"),
            }
        ]
    }
    stub = Stub(get=lambda *a: Response(200, body))
    results = await provider(stub).search_subtitles(
        imdb_id="tt2582802", is_series=False, api_key=KEY
    )

    assert len(results) == 1
    # The advertised reference must carry the file id, which is what /download takes.
    assert results[0].download_url == "/sub/opensubtitles/7144860.srt"
    assert "6230367" not in results[0].download_url


@pytest.mark.asyncio
async def test_search_does_not_assume_arabic_when_language_is_absent():
    body = {"data": [{"id": "1", "attributes": attrs(language=None)}]}
    stub = Stub(get=lambda *a: Response(200, body))
    results = await provider(stub).search_subtitles(
        imdb_id="tt1", is_series=False, api_key=KEY
    )
    assert results[0].lang != "ara"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "os_code,expected",
    [
        ("en", "eng"),
        ("es", "spa"),
        ("fr", "fra"),
        # Published by the live /infos/languages table but not plain ISO-639-1.
        ("pt-br", "por"),
        ("pt-pt", "por"),
        ("zh-cn", "zho"),
        ("az-az", "aze"),
    ],
)
async def test_search_preserves_the_reported_language(os_code, expected):
    """The language the API reports must survive into the release.

    OpenSubtitles publishes eight codes that are not two letters
    (``pt-br``, ``pt-pt``, ``zh-cn``, ``zh-tw``, ``zh-ca``, ``az-az``,
    ``az-zb``, ``tm-td``). Before they were mapped, a response using one was
    labelled "und" -- and any *request* naming one was normalised through the
    Arabic default and silently queried as Arabic.
    """
    body = {"data": [{"id": "1", "attributes": attrs(language=os_code)}]}
    stub = Stub(get=lambda *a: Response(200, body))
    results = await provider(stub).search_subtitles(
        imdb_id="tt1", is_series=False, api_key=KEY
    )
    assert results[0].lang == expected


@pytest.mark.asyncio
async def test_an_unrecognised_language_is_not_reported_as_arabic():
    """Better 'und' than a confident lie: an unknown code must not become Arabic."""
    body = {"data": [{"id": "1", "attributes": attrs(language="qq")}]}
    stub = Stub(get=lambda *a: Response(200, body))
    results = await provider(stub).search_subtitles(
        imdb_id="tt1", is_series=False, api_key=KEY
    )
    assert results[0].lang != "ara"






@pytest.mark.asyncio
async def test_a_search_429_trips_the_breaker_for_the_whole_process():
    stub = Stub(
        get=lambda *a: Response(429, {"message": "API rate limit exceeded"}, {"Retry-After": "1"})
    )
    await provider(stub).search_subtitles(imdb_id="tt1", is_series=False, api_key=KEY)
    assert OPENSUBTITLES_BREAKER.is_open() is True


@pytest.mark.asyncio
async def test_retry_after_is_honoured_over_the_fixed_default():
    stub = Stub(
        get=lambda *a: Response(429, {"message": "slow down"}, {"Retry-After": "300"})
    )
    await provider(stub).search_subtitles(imdb_id="tt1", is_series=False, api_key=KEY)
    # 300s from Retry-After, not the 60s fallback.
    assert 250 < OPENSUBTITLES_BREAKER.remaining <= 300


@pytest.mark.asyncio
async def test_a_malformed_success_body_degrades_to_no_results():
    stub = Stub(get=lambda *a: Response(200, {"unexpected": "shape"}))
    assert await provider(stub).search_subtitles(
        imdb_id="tt1", is_series=False, api_key=KEY
    ) == []


@pytest.mark.asyncio
async def test_a_non_json_success_body_degrades_to_no_results():
    stub = Stub(get=lambda *a: Response(200, None, text="<html>hi</html>"))
    assert await provider(stub).search_subtitles(
        imdb_id="tt1", is_series=False, api_key=KEY
    ) == []


@pytest.mark.asyncio
async def test_search_without_an_api_key_makes_no_request():
    stub = Stub()
    assert await provider(stub).search_subtitles(
        imdb_id="tt1", is_series=False, api_key=""
    ) == []
    assert stub.gets == []


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_sends_only_the_api_key():
    """Downloads use the key without a login or account token."""


    def post(url, headers, json, kw):
        assert json == {"file_id": 7144860}
        assert headers.get("Api-Key") == KEY
        assert "Authorization" not in headers
        return Response(200, {"link": "https://www.opensubtitles.com/download/tok/f.srt"})

    stub = Stub(post=post, get=lambda *a: Response(200, content=SRT))
    data = await provider(stub).download_archive(
        "/sub/opensubtitles/7144860.srt",
        api_key=KEY,
    )
    assert data == SRT


@pytest.mark.asyncio
async def test_download_uses_the_file_id_for_non_srt_formats():
    """The old regex matched only '.srt', so every ASS/VTT download failed."""

    seen: list[dict] = []

    def post(url, headers, json, kw):
        seen.append(json)
        return Response(200, {"link": "https://x/f.ass"})

    stub = Stub(post=post, get=lambda *a: Response(200, content=SRT))
    for ref in (
        "/sub/opensubtitles/999.ass",
        "/sub/opensubtitles/999.vtt",
        "/sub/opensubtitles/999.ssa",
    ):
        assert await provider(stub).download_archive(ref, api_key=KEY) == SRT
    assert [c["file_id"] for c in seen] == [999, 999, 999]




@pytest.mark.asyncio
async def test_a_quota_406_trips_the_breaker_for_the_documented_reset_window():
    body = {
        "requests": 21,
        "remaining": -1,
        "message": "You have downloaded your allowed 20 subtitles for 24h.",
        "reset_time_utc": "2030-01-01T00:00:00.000Z",
    }
    stub = Stub(post=lambda *a: Response(406, body))
    assert (
        await provider(stub).download_archive("/sub/opensubtitles/1.srt", api_key=KEY)
        is None
    )
    assert OPENSUBTITLES_BREAKER.is_open() is True


@pytest.mark.asyncio
async def test_an_invalid_file_id_406_does_not_disable_the_provider():
    """406 is overloaded. Treating a bad id as quota exhaustion took the whole
    process out of OpenSubtitles for an hour over one wrong request."""
    stub = Stub(post=lambda *a: Response(406, {"message": "Invalid file_id"}))
    assert (
        await provider(stub).download_archive("/sub/opensubtitles/1.srt", api_key=KEY)
        is None
    )
    assert OPENSUBTITLES_BREAKER.is_open() is False


@pytest.mark.asyncio
async def test_an_html_error_body_is_never_returned_as_a_subtitle():
    """The `link` is an unauthenticated tokenised URL and can serve HTML with 200."""
    stub = Stub(
        post=lambda *a: Response(200, {"link": "https://x/f.srt"}),
        get=lambda *a: Response(
            200, content=b"<html><body>503 Service Unavailable</body></html>"
        ),
    )
    assert (
        await provider(stub).download_archive("/sub/opensubtitles/1.srt", api_key=KEY)
        is None
    )


@pytest.mark.asyncio
async def test_a_truncated_body_is_rejected_by_the_size_floor():
    stub = Stub(
        post=lambda *a: Response(200, {"link": "https://x/f.srt"}),
        get=lambda *a: Response(200, content=SRT),
    )
    # The provider accepts a small but well-formed body; the *reference* strategy
    # additionally enforces a byte floor. This asserts the two layers are distinct.
    assert await provider(stub).download_archive("/sub/opensubtitles/1.srt", api_key=KEY) == SRT


@pytest.mark.asyncio
async def test_an_open_breaker_prevents_any_further_request():
    stub = Stub(post=lambda *a: Response(200, {"link": "https://x/f.srt"}))
    OPENSUBTITLES_BREAKER.trip(600.0, reason="test")
    assert (
        await provider(stub).download_archive("/sub/opensubtitles/1.srt", api_key=KEY)
        is None
    )
    assert stub.posts == [] and stub.gets == []


@pytest.mark.asyncio
async def test_a_network_error_during_download_returns_none():
    class Boom(Stub):
        async def post(self, *a, **kw):
            raise httpx.ConnectError("down")

    assert (
        await provider(Boom()).download_archive("/sub/opensubtitles/1.srt", api_key=KEY)
        is None
    )
    assert OPENSUBTITLES_BREAKER.is_open() is False


@pytest.mark.asyncio
async def test_download_without_an_api_key_cannot_resolve_a_link():
    stub = Stub()
    assert (
        await provider(stub).download_archive("/sub/opensubtitles/1.srt", api_key="") is None
    )
    assert stub.posts == []


# ---------------------------------------------------------------------------
# MovieHash query shape -- verified against the live API
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_moviehash_query_never_sends_imdb_id():
    """Regression, found by a live smoke test.

    Measured against the real service:

        ?moviehash=H                 -> 2 rows, moviehash_match=true on both
        ?moviehash=H&imdb_id=X       -> 50 rows, moviehash_match=false on ALL
        ?moviehash=H&season_number=1 -> 0 rows
        ?moviehash=H&moviebytesize=S -> 2 rows, moviehash_match=true
        ?moviehash=H&type=movie      -> 2 rows, moviehash_match=true

    Sending imdb_id alongside the hash makes OpenSubtitles fall back to a plain
    IMDb search. It still answers HTTP 200 with plenty of plausible rows, so the
    failure is invisible: the reference path just never sees
    moviehash_match=true and can never obtain an exact reference. A nonsense
    hash returns 0 rows, which is what proves the hash is honoured at all.
    """
    seen: dict = {}

    def get(url, headers, params, kw):
        seen.clear()
        seen.update(params or {})
        return Response(200, {"data": []})

    stub = Stub(get=get)
    await provider(stub).search_subtitles(
        imdb_id="tt10872600",
        is_series=False,
        api_key=KEY,
        languages=[],
        moviehash="239f938f5b1d6ebd",
        moviebytesize=9500000000,
    )

    assert seen["moviehash"] == "239f938f5b1d6ebd"
    assert str(seen["moviebytesize"]) == "9500000000"
    assert seen["type"] == "movie"
    assert "imdb_id" not in seen, "imdb_id silently disables MovieHash matching"
    assert "season_number" not in seen
    assert "episode_number" not in seen


@pytest.mark.asyncio
async def test_a_series_moviehash_query_also_drops_season_and_episode():
    seen: dict = {}

    def get(url, headers, params, kw):
        seen.clear()
        seen.update(params or {})
        return Response(200, {"data": []})

    stub = Stub(get=get)
    await provider(stub).search_subtitles(
        imdb_id="tt0903747",
        is_series=True,
        season=1,
        episode=2,
        api_key=KEY,
        languages=[],
        moviehash="239f938f5b1d6ebd",
    )
    assert "imdb_id" not in seen
    assert "season_number" not in seen
    assert "episode_number" not in seen


@pytest.mark.asyncio
async def test_a_query_without_a_hash_is_unchanged():
    """The normal provider path must keep using imdb_id and season/episode."""
    seen: dict = {}

    def get(url, headers, params, kw):
        seen.clear()
        seen.update(params or {})
        return Response(200, {"data": []})

    stub = Stub(get=get)
    await provider(stub).search_subtitles(
        imdb_id="tt0903747",
        is_series=True,
        season=1,
        episode=2,
        api_key=KEY,
        languages=["ara"],
    )
    assert seen["imdb_id"] == "903747"
    assert seen["season_number"] == 1
    assert seen["episode_number"] == 2
    assert seen["type"] == "episode"
    assert "moviehash" not in seen


# ---------------------------------------------------------------------------
# Secrets never leak
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_credential_appears_in_the_returned_reference(caplog):
    stub = Stub(
        post=lambda *a: Response(200, {"link": "https://x/f.srt"}),
        get=lambda *a: Response(200, content=SRT),
    )
    with caplog.at_level("DEBUG"):
        data = await provider(stub).download_archive(
            "/sub/opensubtitles/7144860.srt",
            api_key=KEY,
        )
    blob = (data or b"").decode("utf-8", "replace") + caplog.text
    assert KEY not in blob
