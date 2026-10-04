"""MovieHash propagation into the AutoSync reference strategy.

Traced conclusion: the data flow is correct and the input is simply absent.

Stremio's subtitle request carries ``filename`` and sometimes ``videoSize``, but
it does not transmit the OpenSubtitles MovieHash for most streams. That hash is
MD5(file_size + first 64 KiB of the file), so producing it means reading the
video's bytes -- something the client does and a server-side add-on never can.
These tests pin both halves of that: the values propagate faithfully when the
client supplies them, and nothing weaker is ever accepted when it does not.
"""

from __future__ import annotations

import pytest
from starlette.requests import Request

from app.main import _media_context_from_request, _merge_sync_meta
from app.models import SubtitleRelease
from app.services.ranking import extract_stream_params
from app.services.sync.hash_reference import _is_explicit_hash_match
from app.services.sync.orchestrator import SyncOrchestrator
from app.services.sync.query import ReferenceQuery

HASH = "239f938f5b1d6ebd"
SIZE = 9500000000


def _serve_request(query: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/x",
            "query_string": query.encode(),
            "headers": [],
        }
    )


# --- 1. Hash and size propagate when the client supplies them ---------------


def test_hash_and_size_are_extracted_from_the_subtitle_extra():
    params = extract_stream_params(
        f"filename=Dexter.s8e04.1080p.mkv&videoHash={HASH}&videoSize={SIZE}", None
    )
    assert params["video_hash"] == HASH
    assert params["video_size"] == SIZE
    assert params["filename"] == "Dexter.s8e04.1080p.mkv"


def test_hash_and_size_survive_the_serve_request_merge():
    context = _media_context_from_request(
        _serve_request(
            f"imdb=tt0773262&type=series&season=8&episode=4"
            f"&filename=Dexter.s8e04.1080p.mkv&videohash={HASH}&videosize={SIZE}"
        )
    )
    assert context["video_hash"] == HASH
    assert context["video_size"] == str(SIZE)

    merged = _merge_sync_meta({"video_hash": None, "video_size": None}, context)
    assert merged["video_hash"] == HASH
    assert merged["video_size"] == str(SIZE)


def test_hash_reaches_the_reference_query():
    context = _media_context_from_request(
        _serve_request(f"videohash={HASH}&videosize={SIZE}&filename=x.mkv")
    )
    merged = _merge_sync_meta({"video_hash": None, "video_size": None}, context)
    query = SyncOrchestrator()._build_query(
        {**merged, "imdb_id": "tt0773262", "season": 8, "episode": 4}
    )
    assert query.video_hash == HASH
    assert query.video_size == str(SIZE)
    assert query.season == 8 and query.episode == 4


def test_the_cache_identity_separates_different_videos():
    """A reference must never be reused across a different hash or size."""
    base = SyncOrchestrator()._build_query(
        {"imdb_id": "tt1", "video_hash": HASH, "video_size": SIZE, "target_filename": "a.mkv"}
    )
    other_hash = SyncOrchestrator()._build_query(
        {"imdb_id": "tt1", "video_hash": "0" * 16, "video_size": SIZE, "target_filename": "a.mkv"}
    )
    other_size = SyncOrchestrator()._build_query(
        {"imdb_id": "tt1", "video_hash": HASH, "video_size": 1, "target_filename": "a.mkv"}
    )
    assert base.cache_stem != other_hash.cache_stem
    assert base.cache_stem != other_size.cache_stem


# --- 2. Missing hash skips the exact-hash lookup -----------------------------


@pytest.mark.asyncio
async def test_missing_hash_skips_the_reference_lookup_entirely(tmp_path):
    from app.services.sync.cache import ReferenceDiskCache
    from app.services.sync.hash_reference import OpenSubtitlesHashReferenceStrategy

    class Provider:
        name = "opensubtitles"

        def __init__(self) -> None:
            self.searches = 0

        async def search_subtitles(self, **kw):
            self.searches += 1
            return []

        async def download_archive(self, *a, **kw):  # pragma: no cover
            raise AssertionError("must not download without a hash")

        def is_breaker_open(self) -> bool:
            return False

    provider = Provider()
    strategy = OpenSubtitlesHashReferenceStrategy(
        provider, cache=ReferenceDiskCache(root=tmp_path / "r", min_bytes=5120)
    )
    resolved = await strategy.resolve_with_provenance(
        ReferenceQuery(imdb_id="tt1", video_hash=None, video_size=SIZE)
    )
    assert resolved.text is None
    assert provider.searches == 0


def test_the_request_stremio_actually_sends_yields_no_hash():
    """The observed real request: filename and videoSize, but no videoHash."""
    params = extract_stream_params(
        "filename=Dexter.s8e04.Scar.tissue.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv&videoSize=123",
        None,
    )
    assert params["filename"]
    assert params["video_size"] == 123
    # Absent -- and that is the whole reason the strategy refuses.
    assert params["video_hash"] is None


# --- 3. Fallback results are never treated as hash matches ------------------


def test_an_imdb_or_title_fallback_result_is_not_a_hash_match():
    release = SubtitleRelease(
        release_name="Dexter.S08E04.1080p.BluRay.x264-PiR8.srt",
        download_url="/sub/opensubtitles/1.srt",
        provider="opensubtitles",
        lang="eng",
    )
    assert release.is_hash_match is False
    assert _is_explicit_hash_match(release) is False


# --- 4. Only an explicit true is accepted -----------------------------------


@pytest.mark.parametrize(
    "attrs,expect",
    [
        ({"moviehash_match": True}, True),
        ({"moviehash_match": False}, False),
        ({}, False),
        ({"moviehash_match": None}, False),
        ({"moviehash_match": "true"}, False),  # a string is not a boolean true
        ({"moviehash_match": 1}, False),
    ],
)
def test_only_a_literal_true_counts_as_a_hash_match(attrs, expect):
    from app.providers.opensubtitles import OpenSubtitlesProvider

    provider = OpenSubtitlesProvider.__new__(OpenSubtitlesProvider)
    provider.client = None
    params = {"moviehash": HASH}
    attribute = {"language": "en", "files": [{"file_id": 1, "file_name": "a.srt"}]}
    attribute.update(attrs)

    import httpx

    class Stub:
        async def get(self, url, headers=None, **kw):
            return httpx.Response(
                200,
                json={"data": [{"id": "1", "attributes": attribute}]},
                request=httpx.Request("GET", url),
            )

    provider.client = Stub()
    results = _run(provider, params)
    assert bool(results and results[0].is_hash_match) is expect


def _run(provider, params):
    import asyncio

    async def go():
        return await provider.search_subtitles(
            imdb_id="tt1", is_series=False, api_key="k", languages=[], **params
        )

    return asyncio.run(go())


def test_a_hash_match_requires_a_hash_to_have_been_sent():
    """The API only reports moviehash_match when a hash was supplied, and the
    provider must not manufacture one for a search that carried no hash."""
    import asyncio

    import httpx

    from app.providers.opensubtitles import OpenSubtitlesProvider

    provider = OpenSubtitlesProvider.__new__(OpenSubtitlesProvider)

    class Stub:
        async def get(self, url, headers=None, **kw):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "1",
                            "attributes": {
                                "language": "en",
                                "moviehash_match": True,
                                "files": [{"file_id": 1, "file_name": "a.srt"}],
                            },
                        }
                    ]
                },
                request=httpx.Request("GET", url),
            )

    provider.client = Stub()
    results = asyncio.run(
        provider.search_subtitles(imdb_id="tt1", is_series=False, api_key="k", languages=[])
    )
    # No moviehash was sent, so the flag must not be honoured.
    assert results and results[0].is_hash_match is False


# --- 5. A genuine explicit hash match is selectable -------------------------


@pytest.mark.asyncio
async def test_a_genuine_hash_match_is_selected_and_downloaded(tmp_path):
    from app.services.sync.cache import ReferenceDiskCache
    from app.services.sync.hash_reference import OpenSubtitlesHashReferenceStrategy

    payload = ("1\n00:00:01,000 --> 00:00:03,000\nline\n\n") * 400
    download_urls: list = []

    class Provider:
        name = "opensubtitles"

        async def search_subtitles(self, **kw):
            assert kw["moviehash"] == HASH
            assert str(kw["moviebytesize"]) == str(SIZE)
            assert kw["languages"] == []  # language-agnostic
            return [
                SubtitleRelease(
                    release_name="match.srt",
                    download_url="/sub/opensubtitles/7144860.srt",
                    provider="opensubtitles",
                    lang="spa",
                    is_hash_match=True,
                )
            ]

        async def download_archive(self, ref, api_key=None, username=None, password=None):
            download_urls.append(ref)
            return payload.encode()

        def is_breaker_open(self) -> bool:
            return False

    strategy = OpenSubtitlesHashReferenceStrategy(
        Provider(), cache=ReferenceDiskCache(root=tmp_path / "r", min_bytes=5120)
    )
    resolved = await strategy.resolve_with_provenance(
        ReferenceQuery(
            imdb_id="tt1",
            video_hash=HASH,
            video_size=SIZE,
            target_filename="a.mkv",
            api_keys={"opensubtitles": "k"},
        )
    )
    assert resolved.text is not None
    assert resolved.kind == "hash"
    assert download_urls == ["/sub/opensubtitles/7144860.srt"]


# --- 6. AutoSync failure does not break normal subtitle delivery -------------


@pytest.mark.asyncio
async def test_reference_failure_still_yields_normal_subtitles():
    """A refused hash lookup must not empty the Stremio response."""
    from unittest.mock import AsyncMock

    from app.models import UserPreferences
    from app.services.aggregator import aggregate_subtitles

    os_provider = AsyncMock()
    os_provider.search_subtitles = AsyncMock(return_value=[])
    os_provider.is_breaker_open = AsyncMock(return_value=False)

    subdl = AsyncMock()
    subdl.search_subtitles = AsyncMock(
        return_value=[
            SubtitleRelease(
                release_name="Show.S01E01.1080p.srt",
                download_url="/sub/abc.srt",
                provider="subdl",
                lang="ara",
                format="srt",
            )
        ]
    )
    subdl.is_breaker_open = AsyncMock(return_value=False)

    prefs = UserPreferences(
        enable_opensubtitles=True,
        enable_subsource=False,
        enable_subtitlecat=False,
        enable_yifysubtitles=False,
        languages=["ara"],
    )
    results = await aggregate_subtitles(
        imdb_id="tt1",
        media_type="series",
        season=1,
        episode=1,
        user_preferences=prefs,
        http_client=None,
        subdl_provider=subdl,
        opensubtitles_provider=os_provider,
        languages=["ara"],
        use_cache=False,
    )
    assert results, "the OpenSubtitles lookup returned nothing and must not empty the list"
    assert any(r.provider == "subdl" for r in results)
