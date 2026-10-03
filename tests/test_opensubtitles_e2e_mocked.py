"""End-to-end OpenSubtitles -> ALASS, fully mocked at the HTTP boundary.

Traces the complete path with the real provider, the real token cache, the real
hash-reference strategy, the real orchestrator and the real ALASS argv
construction:

    Stremio request params
      -> OpenSubtitlesProvider.search_subtitles(moviehash, moviebytesize)
      -> attributes.moviehash_match == true  (the only accepted proof)
      -> POST /login  (Api-Key plus username/password)
      -> POST /download  {"file_id": files[].file_id}
      -> GET the returned link
      -> payload screened as real subtitle content
      -> ReferenceDiskCache
      -> strategy ResolvedReference
      -> orchestrator
      -> alass <reference> <target> <output>

Only the network is faked. No live OpenSubtitles call is made and no real
credential is used; if none is configured this flow cannot run at all, which is
asserted at the end so the suite never implies the live API was exercised.
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest

from app.providers.opensubtitles import OPENSUBTITLES_BREAKER, OpenSubtitlesProvider
from app.providers.opensubtitles_auth import OPENSUBTITLES_TOKENS
from app.services.sync.cache import ReferenceDiskCache
from app.services.sync.hash_reference import OpenSubtitlesHashReferenceStrategy
from app.services.sync.orchestrator import SyncOrchestrator
from app.services.sync_service import SubtitleSyncService

KEY = "dummy-key-not-real"
USER = "dummy-user"
PASSWORD = "dummy-password"
HASH = "239f938f5b1d6ebd"
SIZE = 8410835756
FILE_ID = 7144860
DOWNLOAD_LINK = "https://www.opensubtitles.com/download/abc123/subfile/movie.srt"

_SPACING = (2870, 4310, 1990, 5380, 2410, 3620, 4790, 2180)


def _ts(ms: int) -> str:
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def _srt(cues: int, offset_ms: int, label: str) -> str:
    blocks = []
    cursor = 4000 + offset_ms
    for i in range(cues):
        step = _SPACING[i % len(_SPACING)]
        blocks.append(
            f"{i + 1}\n{_ts(cursor)} --> {_ts(cursor + step - 400)}\n"
            f"{label} line {i + 1} carrying enough words to look like dialogue\n"
        )
        cursor += step
    return "\n".join(blocks)


def _json_response(status: int, payload, headers=None) -> httpx.Response:
    return httpx.Response(
        status,
        json=payload,
        headers=headers or {},
        request=httpx.Request("GET", "https://api.opensubtitles.com/api/v1/x"),
    )


def _search_item(*, moviehash_match: bool, language: str, file_id: int, release: str):
    """A response item shaped exactly as the live API returns one."""
    return {
        "id": str(file_id + 1),  # deliberately != file_id, the classic bug
        "type": "subtitle",
        "attributes": {
            "subtitle_id": str(file_id + 1),
            "language": language,
            "release": release,
            "hearing_impaired": False,
            "moviehash_match": moviehash_match,
            "files": [{"file_id": file_id, "file_name": f"{file_id}.srt"}],
        },
    }


class FakeOpenSubtitles:
    """A scripted OpenSubtitles API over the real provider's HTTP surface."""

    def __init__(self, items, link_body: bytes = b"", link_status: int = 200) -> None:
        self.items = items
        self.link_body = link_body
        self.link_status = link_status
        self.searches: list[dict] = []
        self.downloads: list[dict] = []
        self.logins = 0
        self.link_fetches = 0

    async def get(self, url, headers=None, params=None, **kw):
        if url.endswith("/subtitles"):
            self.searches.append(dict(params or {}))
            return _json_response(200, {"total_count": len(self.items), "data": self.items})
        if url == DOWNLOAD_LINK:
            self.link_fetches += 1
            return httpx.Response(
                self.link_status,
                content=self.link_body,
                headers={"content-type": "text/plain"},
                request=httpx.Request("GET", url),
            )
        raise AssertionError(f"unexpected GET {url}")

    async def post(self, url, headers=None, json=None, **kw):
        if url.endswith("/login"):
            self.logins += 1
            return _json_response(
                200, {"token": "JWT-abc", "base_url": "api.opensubtitles.com"}
            )
        if url.endswith("/download"):
            self.downloads.append({"body": json, "headers": headers})
            return _json_response(200, {"link": DOWNLOAD_LINK})
        raise AssertionError(f"unexpected POST {url}")


@pytest.fixture(autouse=True)
def _reset():
    OPENSUBTITLES_BREAKER.reset()
    OPENSUBTITLES_TOKENS.invalidate(KEY, USER, PASSWORD)
    OPENSUBTITLES_TOKENS.invalidate(KEY, "", "")
    yield
    OPENSUBTITLES_BREAKER.reset()
    OPENSUBTITLES_TOKENS.invalidate(KEY, USER, PASSWORD)


class RecordingSyncService(SubtitleSyncService):
    """Captures the real argv roles instead of spawning alass."""

    def __init__(self, result: str) -> None:
        super().__init__()
        self.result = result
        self.calls: list[dict] = []

    async def sync_async(self, target_srt, reference_srt, **kwargs):
        self.calls.append({"target": target_srt, "reference": reference_srt, **kwargs})
        return self.result


async def _run(tmp_path: Path, fake: FakeOpenSubtitles, target_text: str):
    """Drive the whole pipeline and return (orchestrator, sync, reference text)."""
    provider = OpenSubtitlesProvider(fake)
    strategy = OpenSubtitlesHashReferenceStrategy(
        provider, cache=ReferenceDiskCache(root=tmp_path / "refs", min_bytes=5120)
    )
    sync = RecordingSyncService(target_text)
    orch = SyncOrchestrator(hash_reference_strategy=strategy, sync_service=sync)
    out = await orch.evaluate_and_sync(
        target_text.encode("utf-8"),
        {
            "imdb_id": "tt1375666",
            "target_filename": "Inception.2010.2160p.UHD.BluRay.x265-SURCODE.mkv",
            "media_type": "movie",
            "title": "Inception",
            "year": 2010,
            "lang": "ara",
            "video_hash": HASH,
            "video_size": SIZE,
            "opensubtitles_key": KEY,
            "opensubtitles_username": USER,
            "opensubtitles_password": PASSWORD,
        },
        "target-sub-id",
        True,
    )
    return orch, sync, out


@pytest.mark.asyncio
async def test_full_flow_reaches_alass_with_the_reference_in_argv1(tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.settings.ENABLE_SUBTITLE_SYNC", True)
    reference = _srt(80, 0, "english reference")
    target = _srt(80, 2337, "arabic target")
    fake = FakeOpenSubtitles(
        [_search_item(moviehash_match=True, language="en", file_id=FILE_ID, release="Inception.srt")],
        link_body=reference.encode("utf-8"),
    )

    _orch, sync, out = await _run(tmp_path, fake, target)

    # 1. Searched by hash and size, with no language restriction.
    assert fake.searches, "no search was issued"
    sent = fake.searches[0]
    assert sent["moviehash"] == HASH
    assert str(sent["moviebytesize"]) == str(SIZE)
    assert "languages" not in sent, "the reference search must not filter by language"

    # 2. Logged in, then downloaded with BOTH required headers and the file_id.
    assert fake.logins == 1
    assert len(fake.downloads) == 1
    assert fake.downloads[0]["body"] == {"file_id": FILE_ID}
    headers = fake.downloads[0]["headers"]
    assert headers["Api-Key"] == KEY
    assert headers["Authorization"] == "JWT-abc"

    # 3. The link was fetched exactly once.
    assert fake.link_fetches == 1

    # 4. The reference text actually reached ALASS, in the reference slot.
    assert len(sync.calls) == 1
    call = sync.calls[0]
    assert call["reference"].startswith("1\n00:00:04,000")
    assert "english reference line 1" in call["reference"]
    assert call["target"] == target
    assert "arabic target line 1" in call["target"]
    assert call["decision_kind"] == "hash"

    # 5. The output is the target. The reference never leaks in as the result.
    assert b"arabic target" in out
    assert b"english reference" not in out


@pytest.mark.asyncio
async def test_a_result_without_moviehash_match_never_becomes_a_reference(tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.settings.ENABLE_SUBTITLE_SYNC", True)
    target = _srt(80, 2337, "arabic target")
    fake = FakeOpenSubtitles(
        [_search_item(moviehash_match=False, language="en", file_id=FILE_ID, release="Inception.srt")],
        link_body=_srt(80, 0, "english reference").encode("utf-8"),
    )

    _orch, sync, _out = await _run(tmp_path, fake, target)

    # A search-time login is expected and harmless (the JWT raises the search
    # quota). What must not happen is spending a *download*, or running alass.
    assert fake.downloads == []
    assert sync.calls == []


@pytest.mark.asyncio
async def test_only_the_explicit_hash_match_among_mixed_results_is_used(tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.settings.ENABLE_SUBTITLE_SYNC", True)
    target = _srt(80, 2337, "arabic target")
    fake = FakeOpenSubtitles(
        [
            _search_item(moviehash_match=False, language="ar", file_id=111, release="A.srt"),
            _search_item(moviehash_match=True, language="es", file_id=222, release="B.srt"),
            _search_item(moviehash_match=False, language="fr", file_id=333, release="C.srt"),
        ],
        link_body=_srt(80, 0, "spanish reference").encode("utf-8"),
    )

    _orch, sync, _out = await _run(tmp_path, fake, target)

    assert len(fake.downloads) == 1
    # The only explicit hash match, and the reference need not be English.
    assert fake.downloads[0]["body"] == {"file_id": 222}
    assert len(sync.calls) == 1
    assert "spanish reference line 1" in sync.calls[0]["reference"]


@pytest.mark.asyncio
async def test_an_html_body_from_the_link_is_rejected_before_alass(tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.settings.ENABLE_SUBTITLE_SYNC", True)
    target = _srt(80, 2337, "arabic target")  # noqa: F841 - read by _run below
    fake = FakeOpenSubtitles(
        [_search_item(moviehash_match=True, language="en", file_id=FILE_ID, release="Inception.srt")],
        link_body=b"<html><body>503 Service Unavailable</body></html>",
    )

    _orch, sync, out = await _run(tmp_path, fake, target)

    assert sync.calls == [], "an HTML error page must never reach ALASS"
    # The request still succeeds with the untouched target.
    assert b"arabic target" in out


@pytest.mark.asyncio
async def test_a_second_request_reuses_the_cached_reference_without_any_call(tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.settings.ENABLE_SUBTITLE_SYNC", True)
    target = _srt(80, 2337, "arabic target")
    fake = FakeOpenSubtitles(
        [_search_item(moviehash_match=True, language="en", file_id=FILE_ID, release="Inception.srt")],
        link_body=_srt(80, 0, "english reference").encode("utf-8"),
    )

    await _run(tmp_path, fake, target)
    assert len(fake.searches) == 1 and len(fake.downloads) == 1

    # Same video, same target: served entirely from the reference cache.
    await _run(tmp_path, fake, target)
    assert len(fake.searches) == 1, "cache hit must not re-search OpenSubtitles"
    assert len(fake.downloads) == 1, "cache hit must not re-download"
    assert fake.link_fetches == 1
    assert fake.logins == 1


@pytest.mark.asyncio
async def test_the_cache_does_not_leak_across_videos(tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.settings.ENABLE_SUBTITLE_SYNC", True)
    fake = FakeOpenSubtitles(
        [_search_item(moviehash_match=True, language="en", file_id=FILE_ID, release="Inception.srt")],
        link_body=_srt(80, 0, "english reference").encode("utf-8"),
    )

    provider = OpenSubtitlesProvider(fake)
    strategy = OpenSubtitlesHashReferenceStrategy(
        provider, cache=ReferenceDiskCache(root=tmp_path / "refs", min_bytes=5120)
    )

    from app.services.sync.query import ReferenceQuery

    base = {
        "imdb_id": "tt1375666",
        "target_filename": "Inception.2010.mkv",
        "media_type": "movie",
        "video_size": SIZE,
        "api_keys": {
            "opensubtitles": KEY,
            "opensubtitles_username": USER,
            "opensubtitles_password": PASSWORD,
        },
    }
    await strategy.resolve_with_provenance(ReferenceQuery(video_hash=HASH, **base))
    await strategy.resolve_with_provenance(ReferenceQuery(video_hash="0000000000000000", **base))
    await strategy.resolve_with_provenance(
        ReferenceQuery(video_hash=HASH, **{**base, "video_size": 1})
    )

    # Three distinct videos, three separate retrievals.
    assert len(fake.downloads) == 3


@pytest.mark.asyncio
async def test_opensubtitles_failing_never_breaks_the_request(tmp_path, monkeypatch):
    """A dead upstream must fall through, not fail the Stremio request."""
    monkeypatch.setattr("app.config.settings.ENABLE_SUBTITLE_SYNC", True)
    target = _srt(80, 2337, "arabic target")

    class Dead(FakeOpenSubtitles):
        async def get(self, url, headers=None, params=None, **kw):
            raise httpx.ConnectError("opensubtitles unreachable")

    _orch, sync, out = await _run(tmp_path, Dead([]), target)
    assert sync.calls == []
    assert b"arabic target" in out

    # And an exhausted quota behaves the same way. The search must still return a
    # genuine hash match, otherwise nothing ever reaches /download and the 406
    # path is never exercised.
    class Quota(FakeOpenSubtitles):
        async def post(self, url, headers=None, json=None, **kw):
            if url.endswith("/login"):
                self.logins += 1
                return _json_response(200, {"token": "JWT", "base_url": "api.opensubtitles.com"})
            self.downloads.append({"body": json, "headers": headers})
            return _json_response(
                406,
                {
                    "requests": 21,
                    "remaining": -1,
                    "message": "You have downloaded your allowed 20 subtitles for 24h.",
                    "reset_time_utc": "2030-01-01T00:00:00.000Z",
                },
            )

    quota_dir = tmp_path / "quota"
    quota_dir.mkdir(parents=True, exist_ok=True)
    quota_fake = Quota(
        [_search_item(moviehash_match=True, language="en", file_id=FILE_ID, release="I.srt")],
        link_body=_srt(80, 0, "english reference").encode("utf-8"),
    )
    _orch2, sync2, out2 = await _run(quota_dir, quota_fake, target)
    assert len(quota_fake.downloads) == 1, "the 406 path was not reached"
    assert sync2.calls == []
    assert b"arabic target" in out2
    assert OPENSUBTITLES_BREAKER.is_open() is True


@pytest.mark.asyncio
async def test_no_hash_metadata_means_no_opensubtitles_call_at_all(tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.settings.ENABLE_SUBTITLE_SYNC", True)
    fake = FakeOpenSubtitles(
        [_search_item(moviehash_match=True, language="en", file_id=FILE_ID, release="I.srt")],
        link_body=_srt(80, 0, "english reference").encode("utf-8"),
    )

    provider = OpenSubtitlesProvider(fake)
    strategy = OpenSubtitlesHashReferenceStrategy(
        provider, cache=ReferenceDiskCache(root=tmp_path / "refs", min_bytes=5120)
    )
    from app.services.sync.query import ReferenceQuery

    resolved = await strategy.resolve_with_provenance(
        ReferenceQuery(
            imdb_id="tt1375666",
            video_hash=None,
            api_keys={"opensubtitles": KEY},
        )
    )
    assert resolved.text is None
    assert fake.searches == [] and fake.downloads == []


def test_no_real_credential_is_present_in_this_suite():
    """Guards the claim boundary: these tests never touch the live API.

    Asserting on the literal JWT header prefix cannot live in this file, because
    the assertion itself would contain it. The structural check is what matters:
    every credential here is an obvious dummy.
    """
    source = Path(__file__).read_text(encoding="utf-8")
    assert KEY == "dummy-key-not-real"
    assert PASSWORD == "dummy-password"
    for value in (KEY, USER, PASSWORD, "JWT-abc", "JWT"):
        assert value.lower().startswith(("dummy", "jwt"))
        assert f'"{value}"' in source
    # Nothing resembling a real 40+ character OpenSubtitles consumer key.
    assert not re.search(r'"[A-Za-z0-9]{40,}"', source)
