"""Serve-time verdicts for OpenSubtitles proxy subtitles must reach ranking.

Root cause, proven live (Whiplash tt2582802): an OpenSubtitles subtitle
("Whiplash 2014 netflix.srt") verified with a perfect score --
``verified_resynced``, median=0, mad=0, p95=0, struct=1.00, sync_conf=100 --
yet stayed at the bottom of the Stremio list on every request.

Why: serve-time verdicts are stored under the proxy route's ``file_id``
(``/sub/opensubtitles/8251770.srt`` -> alias key ``...:8251770``), but the
listing-time lookup only tried ``rel_sub_id`` (a sha256 of
provider:release_name:download_url). Those keys can never match, so the
verdict was orphaned and the subtitle ranked on its generic release name
alone.

``_lookup_search_verdict`` tries the candidate id first, then the provider
file_id for OpenSubtitles entries -- the identical value both sides key on.
"""

from __future__ import annotations

import pytest

from app.main import _lookup_search_verdict, _opensubtitles_file_id


@pytest.mark.parametrize(
    ("provider", "url", "expected"),
    [
        ("opensubtitles", "https://dl.opensubtitles.org/sub/8251770/x.srt", "8251770"),
        ("OpenSubtitles", "https://dl.opensubtitles.org/sub/8251770/x.srt", "8251770"),
        ("opensubtitles", "https://example.invalid/no-digits-here.srt", None),
        ("opensubtitles", "", None),
        ("subdl", "https://dl.subdl.com/subtitle/12345.zip", None),
        ("subsource", "https://api.subsource.net/dl/12345", None),
    ],
)
def test_file_id_extraction_is_provider_scoped(provider, url, expected):
    assert _opensubtitles_file_id(_Rel(provider, url)) == expected


class _Rel:
    def __init__(self, provider: str, download_url: str):
        self.provider = provider
        self.download_url = download_url


class _Cache:
    def __init__(self, verdicts: dict[tuple[str, str], dict]):
        self._verdicts = verdicts
        self.calls: list[tuple[str, str]] = []
        self.raise_on: set[tuple[str, str]] = set()

    async def get_verdict_by_ref(self, fingerprint: str, ref: str):
        self.calls.append((fingerprint, ref))
        if (fingerprint, ref) in self.raise_on:
            raise RuntimeError("cache backend down")
        return self._verdicts.get((fingerprint, ref))


FP = "fingerprint-abc123"
VERDICT = {"sync_state": "verified_resynced", "sync_confidence": 100}


def _os_rel():
    return _Rel(
        "opensubtitles",
        "https://dl.opensubtitles.org/sub/8251770/Whiplash 2014 netflix.srt",
    )


@pytest.mark.asyncio
async def test_candidate_id_hit_is_used_directly():
    cache = _Cache({(FP, "relsubid00000001"): VERDICT})
    out = await _lookup_search_verdict(cache, FP, "relsubid00000001", _os_rel())
    assert out == VERDICT
    assert cache.calls == [(FP, "relsubid00000001")]


@pytest.mark.asyncio
async def test_opensubtitles_falls_back_to_file_id():
    """THE regression: the verdict lives under the serve-time file_id."""
    cache = _Cache({(FP, "8251770"): VERDICT})
    out = await _lookup_search_verdict(cache, FP, "relsubid00000001", _os_rel())
    assert out == VERDICT, "must find the verdict stored under the proxy file_id"
    assert cache.calls == [(FP, "relsubid00000001"), (FP, "8251770")]


@pytest.mark.asyncio
async def test_candidate_id_wins_when_both_exist():
    cache = _Cache({(FP, "relsubid00000001"): {"sync_state": "unverified"}, (FP, "8251770"): VERDICT})
    out = await _lookup_search_verdict(cache, FP, "relsubid00000001", _os_rel())
    assert out == {"sync_state": "unverified"}
    assert cache.calls == [(FP, "relsubid00000001")]


@pytest.mark.asyncio
async def test_no_verdict_anywhere_returns_none():
    cache = _Cache({})
    assert await _lookup_search_verdict(cache, FP, "relsubid00000001", _os_rel()) is None


@pytest.mark.asyncio
async def test_non_opensubtitles_never_gets_a_second_lookup():
    """SubDL/SubSource serve under their own rel_sub_id, so a second lookup
    would only risk cross-key confusion."""
    cache = _Cache({(FP, "8251770"): VERDICT})
    rel = _Rel("subdl", "https://dl.subdl.com/subtitle/12345.zip")
    assert await _lookup_search_verdict(cache, FP, "relsubid00000001", rel) is None
    assert cache.calls == [(FP, "relsubid00000001")]


@pytest.mark.asyncio
async def test_missing_fingerprint_touches_nothing():
    cache = _Cache({(FP, "8251770"): VERDICT})
    assert await _lookup_search_verdict(cache, None, "relsubid00000001", _os_rel()) is None
    assert cache.calls == []


@pytest.mark.asyncio
async def test_cache_failure_degrades_to_prediction():
    cache = _Cache({})
    cache.raise_on = {(FP, "relsubid00000001"), (FP, "8251770")}
    assert await _lookup_search_verdict(cache, FP, "relsubid00000001", _os_rel()) is None
