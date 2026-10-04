"""Provider download pacing for the multi-candidate reference pool.

Regression under test: resolving the pool eagerly fired up to
REFERENCE_POOL_LIMIT downloads back-to-back. Against SubDL that tripped the
provider rate limiter and opened the circuit breaker *mid-request*:

    GET dl.subdl.com/...3610179-8527589.zip  ->  HTTP 429 Too Many Requests
    WARNING: Subdl circuit breaker tripped (HTTP 429); skipping SubDL for 60s
    subdl returned 0 candidate(s)  ->  resolve_all: 0 distinct candidate(s)
    external-release-reference produced no usable reference candidate

A later candidate in the same request then got no reference and fell back to
the original. These tests pin that downloads are spaced, and -- just as
important -- that pacing does not weaken verification or slow the common
single-candidate case.
"""

from __future__ import annotations

import time

import pytest

from app.models import SubtitleRelease
from app.services.sync import external_strategy as es
from app.services.sync.external_strategy import ExternalExactStrategy
from app.services.sync.query import ReferenceQuery


def _rel(name: str):
    return SubtitleRelease(
        release_name=name, download_url=f"/sub/x/{name}", provider="subdl"
    )


def _srt(cues: int, tag: str) -> str:
    out = []
    for i in range(cues):
        s = 1000 + i * 3000
        out.append(f"{i + 1}\n00:00:{s // 1000:02d},{s % 1000:03d} --> 00:00:{(s + 2000) // 1000:02d},000\n{tag}{i}\n")
    return "\n".join(out)


def _query():
    return ReferenceQuery(
        imdb_id="tt2582802",
        media_type="movie",
        target_filename="Whiplash.2014.REMUX-FraMeSToR.mkv",
        languages=("ara", "eng"),
    )


class _Provider:
    name = "subdl"

    def __init__(self, payloads):
        self.payloads = payloads
        self.stamps: list[float] = []

    def is_breaker_open(self):
        return False

    async def download_archive(self, release, api_key=None, username=None, password=None):
        name = getattr(release, "release_name", "")
        self.stamps.append(time.monotonic())
        return self.payloads.get(name, "").encode()


class _Strategy(ExternalExactStrategy):
    def __init__(self, payloads, **kw):
        super().__init__(**kw)
        self.provider = _Provider(payloads)
        self._subdl = self.provider

    async def _gather_candidates(self, providers, query):
        return [(_rel(n), self.provider) for n in self.provider.payloads]

    async def _download_candidate(self, best, provider, query):
        from app.services.sync.query import ResolvedReference

        raw = await provider.download_archive(best)
        name = getattr(best, "release_name", "?")
        if not raw:
            return ResolvedReference(None)
        return ResolvedReference(
            raw.decode(), kind="edition", candidate=name, partial=False
        )


def _strategy(n: int):
    payloads = {f"Rel{i}.2014.2160p.mkv": _srt(500 + i, f"T{i}") for i in range(n)}
    return _Strategy(payloads, cache=None, timeout=5.0)


@pytest.mark.asyncio
async def test_consecutive_downloads_are_spaced(monkeypatch):
    """THE regression: back-to-back downloads tripped a provider 429."""
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(es.asyncio, "sleep", fake_sleep)

    s = _strategy(4)
    out = await s.resolve_all_with_provenance(_query(), update_validator=lambda t: True)

    assert len(out) == 4, "all four candidates still resolve"
    # One pause between each pair: 4 downloads => 3 pauses. The FIRST download
    # pays no latency.
    assert len(slept) == 3, f"expected 3 inter-download pauses, got {len(slept)}"
    assert all(d >= 0.5 for d in slept), f"pause too short: {slept}"


@pytest.mark.asyncio
async def test_the_single_candidate_case_pays_no_throttle(monkeypatch):
    """The overwhelmingly common path must not gain latency."""
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(es.asyncio, "sleep", fake_sleep)

    s = _strategy(1)
    out = await s.resolve_all_with_provenance(_query(), update_validator=lambda t: True)

    assert len(out) == 1
    assert slept == [], "a single download must not sleep"


@pytest.mark.asyncio
async def test_pacing_does_not_reduce_the_candidate_pool(monkeypatch):
    async def fake_sleep(seconds):
        return None

    monkeypatch.setattr(es.asyncio, "sleep", fake_sleep)

    s = _strategy(6)
    out = await s.resolve_all_with_provenance(_query(), update_validator=lambda t: True)
    assert len(out) == 6, "throttling must cost candidates, not correctness"


@pytest.mark.asyncio
async def test_pacing_preserves_content_dedup(monkeypatch):
    """Duplicate payloads must still collapse despite the pacing."""

    async def fake_sleep(seconds):
        return None

    monkeypatch.setattr(es.asyncio, "sleep", fake_sleep)

    same = _srt(500, "SAME")
    s = _Strategy(
        {"A.2014.mkv": same, "B.2014.mkv": same, "C.2014.mkv": _srt(400, "OTHER")},
        cache=None,
        timeout=5.0,
    )
    out = await s.resolve_all_with_provenance(_query(), update_validator=lambda t: True)
    assert len(out) == 2


def test_the_interval_is_a_module_constant_not_a_magic_number():
    assert es._REFERENCE_DOWNLOAD_MIN_INTERVAL_SECONDS >= 0.5
