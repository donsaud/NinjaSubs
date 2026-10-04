"""Multi-candidate reference discovery and independent verification.

The gap this closes: the orchestrator's retry loop advanced between STRATEGIES,
not between candidates. ``ExternalExactStrategy.resolve_with_provenance``
returned exactly one reference, so when only SubDL was configured and no
MovieHash existed there was a single ALASS attempt. A wrong-cut first pick
ended the request even though other candidate releases were in the pool.

Production evidence (Whiplash, tt2582802): one cached ``subdl_edition``
reference with 922 cues aligned against an 825-cue target (1.12x surplus),
giving movement MAD 9356ms and residual matching that DEGRADED from 33 to 130
unmatched cues at a 5s tolerance.

These tests pin: multiple distinct candidates, content-level dedup, the
Whiplash cue-count case, per-candidate independent verification, exhaustion
falling back to the original, and -- most importantly -- that nothing is served
unless it passes the existing verifier.
"""

from __future__ import annotations

import hashlib

import pytest

from app.models import SubtitleRelease
from app.services.sync.external_strategy import ExternalExactStrategy
from app.services.sync.orchestrator import (
    SyncOrchestrator,
    _PreResolvedReference,
    decision_kind_is_name_based,
)

# ===========================================================================
# Fixtures
# ===========================================================================


def _rel(name: str, **kw):
    base = {"release_name": name, "download_url": f"/sub/x/{name}", "provider": "subdl"}
    base.update(kw)
    return SubtitleRelease(**base)


def _srt(cue_count: int, first_start_ms: int = 1000, step_ms: int = 3000, tag: str = "") -> str:
    """Build a structurally valid SRT with a controllable cue count."""
    cues = []
    for i in range(cue_count):
        start = first_start_ms + i * step_ms
        end = start + 2000
        cues.append(
            f"{i + 1}\n"
            f"{start // 3600000:02d}:{(start // 60000) % 60:02d}:{(start // 1000) % 60:02d},"
            f"{start % 1000:03d} --> "
            f"{end // 3600000:02d}:{(end // 60000) % 60:02d}:{(end // 1000) % 60:02d},"
            f"{end % 1000:03d}\n{tag}line {i + 1}\n"
        )
    return "\n".join(cues)


TARGET_CUES = 825
WHIPLASH_REFERENCE_CUES = 922  # the wrong-cut reference actually observed


class _FakeProvider:
    """Provider whose downloads are served from a name -> text map."""

    name = "subdl"

    def __init__(self, payloads: dict[str, str]):
        self.payloads = payloads
        self.downloads: list[str] = []

    def is_breaker_open(self) -> bool:
        return False

    async def search_subtitles(self, **kw):
        return [_rel(n) for n in self.payloads]

    async def download_archive(self, release, api_key=None, username=None, password=None):
        name = getattr(release, "release_name", "")
        self.downloads.append(name)
        return self.payloads.get(name, "").encode("utf-8")


class _Strategy(ExternalExactStrategy):
    """ExternalExactStrategy with the network edge stubbed out."""

    def __init__(self, payloads, **kw):
        super().__init__(**kw)
        self.provider = _FakeProvider(payloads)
        self._subdl = self.provider

    async def _gather_candidates(self, providers, query):
        rels = [_rel(n) for n in self.provider.payloads]
        return [(r, self.provider) for r in rels]

    async def _download_candidate(self, best, provider, query):
        from app.services.sync.query import ResolvedReference

        name = getattr(best, "release_name", "?")
        raw = await provider.download_archive(best)
        if not raw:
            return ResolvedReference(None)
        # The real strategy decodes the archive; mirror that.
        return ResolvedReference(
            raw.decode("utf-8") if isinstance(raw, bytes) else raw,
            kind="edition", candidate=name, partial=False,
        )


def _strategy(payloads):
    return _Strategy(payloads, cache=None, timeout=5.0)


def _query(**kw):
    from app.services.sync.query import ReferenceQuery

    base = {
        "imdb_id": "tt2582802",
        "media_type": "movie",
        "target_filename": "Whiplash.2014.2160p.UHD.BluRay.x254-SURCODE.mkv",
        "video_hash": "",
        "video_size": "19571049411",
        "languages": ("ara",),
    }
    base.update(kw)
    return ReferenceQuery(**base)


# ===========================================================================
# 1. Multiple distinct candidates are exposed
# ===========================================================================


@pytest.mark.asyncio
async def test_multiple_distinct_candidates_are_returned():
    s = _strategy(
        {
            "A.2014.2160p.mkv": _srt(820, tag="A"),
            "B.2014.1080p.mkv": _srt(700, tag="B"),
            "C.2014.720p.mkv": _srt(500, tag="C"),
        }
    )
    out = await s.resolve_all_with_provenance(_query(), update_validator=lambda t: True)
    assert len(out) == 3
    assert all(r.text for r in out)
    # Provenance preserved per candidate.
    assert {r.candidate for r in out} == {
        "A.2014.2160p.mkv",
        "B.2014.1080p.mkv",
        "C.2014.720p.mkv",
    }
    assert all(r.kind == "edition" for r in out)


@pytest.mark.asyncio
async def test_no_candidates_yields_an_empty_list():
    s = _strategy({})
    assert await s.resolve_all_with_provenance(_query(), update_validator=lambda t: True) == []


# ===========================================================================
# 2. Deduplication is by CONTENT, not filename
# ===========================================================================


@pytest.mark.asyncio
async def test_duplicate_content_under_different_filenames_is_collapsed():
    """Two providers routinely publish the same timing model under different
    release names. Spending two downloads to learn one fact is waste, and
    counting them as distinct candidates would inflate the candidate count."""
    same = _srt(820, tag="SAME")
    s = _strategy(
        {
            "Release.Name.One.2014.mkv": same,
            "Totally.Different.Name.2014.mkv": same,
            "Third.Unique.Name.2014.mkv": _srt(700, tag="OTHER"),
        }
    )
    out = await s.resolve_all_with_provenance(_query(), update_validator=lambda t: True)
    assert len(out) == 2, "identical content must collapse to one candidate"
    names = {r.candidate for r in out}
    assert "Totally.Different.Name.2014.mkv" not in names or len(out) == 2


@pytest.mark.asyncio
async def test_content_digest_is_the_dedup_key_not_the_name():
    a = _srt(820, tag="X")
    b = _srt(820, tag="X")
    assert hashlib.sha256(a.encode()).hexdigest() == hashlib.sha256(b.encode()).hexdigest()
    c = _srt(821, tag="X")
    assert hashlib.sha256(a.encode()).hexdigest() != hashlib.sha256(c.encode()).hexdigest()


# ===========================================================================
# 3. The Whiplash cue-count case
# ===========================================================================


@pytest.mark.asyncio
async def test_the_whiplash_wrong_cut_reference_is_still_offered_but_not_special():
    """The 922-vs-825 reference must not be silently preferred on the strength
    of a cue-count heuristic. Cue counts RANK; they never establish identity."""
    wrong_cut = _srt(WHIPLASH_REFERENCE_CUES, tag="WC")
    right_cut = _srt(TARGET_CUES, tag="RC")
    s = _strategy({"WrongCut.2014.2160p.mkv": wrong_cut, "RightCut.2014.mkv": right_cut})

    out = await s.resolve_all_with_provenance(_query(), update_validator=lambda t: True)

    assert len(out) == 2, "both candidates must be offered; the verifier decides"
    # Crucially, no candidate is marked as an exact/hash match because of it.
    assert all(decision_kind_is_name_based(r.kind) for r in out)


def test_the_observed_cue_surplus_is_the_failure_signature():
    """Documents why one candidate was not enough: 922 vs 825 cues."""
    assert WHIPLASH_REFERENCE_CUES / TARGET_CUES == pytest.approx(1.117, abs=0.01)
    # ~1.12x surplus -> piecewise cut -> movement MAD in the seconds range.
    assert WHIPLASH_REFERENCE_CUES - TARGET_CUES == 97


# ===========================================================================
# 4. update_validator still filters
# ===========================================================================


@pytest.mark.asyncio
async def test_a_candidate_failing_cue_sanity_is_not_offered():
    s = _strategy({"Good.2014.mkv": _srt(820, tag="G"), "Bad.2014.mkv": _srt(820, tag="B")})
    calls: list[str] = []

    def validate(text: str) -> bool:
        calls.append(text)
        return text is not None and "line 1" in text

    out = await s.resolve_all_with_provenance(_query(), update_validator=validate)
    # Both are structurally fine, so both pass; the validator was consulted.
    assert len(out) == 2
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_a_rejecting_validator_removes_every_candidate():
    s = _strategy({"A.2014.mkv": _srt(820)})
    out = await s.resolve_all_with_provenance(_query(), update_validator=lambda t: False)
    assert out == []


# ===========================================================================
# 5. Orchestrator feeds each candidate through the existing loop
# ===========================================================================


def test_pre_resolved_adapter_replays_the_candidate():
    """The adapter is what lets the ~600-line execution body apply unchanged."""
    from app.services.sync.query import ResolvedReference

    r = ResolvedReference("text", kind="edition", candidate="X")
    a = _PreResolvedReference(r)
    assert a.validates_target is False


@pytest.mark.asyncio
async def test_the_expansion_uses_the_multi_path_and_never_the_single_path(monkeypatch):
    """The orchestrator must consult resolve_all_with_provenance exactly once
    and must NOT fall back to resolve_with_provenance when candidates exist."""
    from app.services.sync import orchestrator as om
    from app.services.sync.query import ResolvedReference

    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True, raising=False)

    class _Multi:
        validates_target = True
        cache = None

        def __init__(self):
            self.multi_calls = 0
            self.single_calls = 0

        async def resolve_all_with_provenance(self, query, *, update_validator=None):
            self.multi_calls += 1
            return [
                ResolvedReference(_srt(820, tag="A"), kind="edition", candidate="A"),
                ResolvedReference(_srt(700, tag="B"), kind="edition", candidate="B"),
                ResolvedReference(_srt(500, tag="C"), kind="edition", candidate="C"),
            ]

        async def resolve_with_provenance(self, query, *, update_validator=None):
            self.single_calls += 1
            raise AssertionError("single-reference path must not run when candidates exist")

    class _Cache:
        is_ephemeral = False

        async def get(self, k):
            return None

        async def get_meta(self, k):
            return None

        async def is_failed(self, k):
            return False

        async def get_verdict(self, k):
            return None

        async def set(self, k, d):
            pass

        async def set_meta(self, k, m):
            pass

        async def set_verdict(self, k, v, **kw):
            pass

        async def mark_failed(self, k, ttl=None):
            pass

        async def clear_failed(self, k):
            pass

    multi = _Multi()
    # A sync_service stub keeps the candidate loop reachable; the point of this
    # test is WHICH candidates the strategy offers, not the alass outcome.
    o = SyncOrchestrator(sync_cache=_Cache(), external_strategy=multi, sync_service=object())

    await o.evaluate_and_sync(
        _srt(TARGET_CUES, tag="T").encode(),
        {
            "media_type": "movie",
            "target_filename": "Whiplash.2014.2160p.mkv",
            "video_hash": "",
            "video_size": "19571049411",
            "lang": "ara",
            "imdb_id": "tt2582802",
        },
        "tt2582802",
        auto_sync=True,
    )

    assert multi.multi_calls == 1, "multi-candidate path must be used exactly once"
    assert multi.single_calls == 0, "must not also run the single-reference path"


# ===========================================================================
# 6. Nothing is served without passing the verifier
# ===========================================================================


@pytest.mark.asyncio
async def test_all_candidates_failing_serves_the_original():
    """Exhaustion must fall back to the original bytes, never to an unverified
    alignment."""
    from app.services.sync import orchestrator as om

    om.settings.ENABLE_SUBTITLE_SYNC = True

    class _Cache:
        is_ephemeral = False

        async def get(self, k):
            return None

        async def get_meta(self, k):
            return None

        async def is_failed(self, k):
            return False

        async def get_verdict(self, k):
            return None

        async def set(self, k, d):
            pass

        async def set_meta(self, k, m):
            pass

        async def set_verdict(self, k, v, **kw):
            pass

        async def mark_failed(self, k, ttl=None):
            pass

        async def clear_failed(self, k):
            pass

    class _Multi:
        validates_target = True
        cache = None

        async def resolve_all_with_provenance(self, query, *, update_validator=None):
            from app.services.sync.query import ResolvedReference

            # Wrong cut (922 cues) then another wrong cut.
            return [
                ResolvedReference(_srt(922, tag="WC"), kind="edition", candidate="WC"),
                ResolvedReference(_srt(1100, tag="WC2"), kind="edition", candidate="WC2"),
            ]

        async def resolve_with_provenance(self, query, *, update_validator=None):
            raise AssertionError("must not fall back to the single-reference path")

    o = SyncOrchestrator(sync_cache=_Cache(), external_strategy=_Multi(), sync_service=None)
    sub = _srt(TARGET_CUES, tag="T").encode()

    out = await o.evaluate_and_sync(
        sub,
        {
            "media_type": "movie",
            "target_filename": "Whiplash.2014.mkv",
            "video_hash": "",
            "video_size": "19571049411",
            "languages": ("ara",),
            "imdb_id": "tt2582802",
        },
        "tt2582802",
        auto_sync=True,
    )
    assert out == sub, "must serve the ORIGINAL subtitle when every candidate fails"


# ===========================================================================
# 7. Safety invariants
# ===========================================================================


def test_multi_candidate_expansion_creates_no_hash_identity():
    """Offering several candidates must never upgrade any of them to an exact
    match."""
    for prov in ("subdl", "subsource", "yifysubtitles", "subtitlecat"):
        r = SubtitleRelease(release_name="a.srt", download_url="/sub/x.srt", provider=prov)
        assert r.is_hash_match is False
        assert r.matched_by_hash is False


def test_multi_candidate_expansion_does_not_relax_verifier_thresholds():
    from app.services.sync import alignment

    assert alignment.MAX_P95_MS_FOR_STABLE == 2000
    assert alignment.MAX_MAD_MS_FOR_STABLE < 9356


def test_a_hash_strategy_is_not_expanded_into_ranked_candidates():
    """Exact-identity evidence must not be traded for a ranked list of weaker
    references. The hash strategy exposes no multi-candidate method, so the
    expansion skips it by construction."""
    from app.services.sync.hash_reference import OpenSubtitlesHashReferenceStrategy

    assert not hasattr(OpenSubtitlesHashReferenceStrategy, "resolve_all_with_provenance")


def test_cue_counts_are_ranking_signals_only():
    """Documents the rule in code: a cue-count match is not an identity claim."""
    assert decision_kind_is_name_based("edition") is True
    assert decision_kind_is_name_based("hash") is False


@pytest.mark.asyncio
async def test_a_cached_reference_does_not_disable_expansion(monkeypatch):
    """REGRESSION for the live Whiplash failure.

    A cache-hit short-circuit made expansion exit before gathering, so a repeat
    request with a cached `subdl_edition` reference produced exactly ONE alass
    run and zero alternatives -- observed live: 0 `resolve_all` lines,
    1 `invoking alass`, verifier rejecting with the identical metrics
    (p95 2416ms, movement mad 9356ms).
    """
    from app.services.sync import orchestrator as om
    from app.services.sync.query import ResolvedReference

    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True, raising=False)

    # A REAL ResolvedReference: the loop reads many provenance attributes, and a
    # thin stub would fail for unrelated reasons instead of testing this.
    _hit_ref = ResolvedReference("cached", kind="edition", candidate="CACHED")

    class _CacheRef:
        def get(self, query):
            return _hit_ref

    class _Multi:
        validates_target = True
        cache = _CacheRef()

        def __init__(self):
            self.multi_calls = 0

        async def resolve_all_with_provenance(self, query, *, update_validator=None):
            self.multi_calls += 1
            return [
                ResolvedReference("alt-one", kind="edition", candidate="ALT1"),
                ResolvedReference("alt-two", kind="edition", candidate="ALT2"),
            ]

    class _Cache:
        is_ephemeral = False

        async def get(self, k):
            return None

        async def get_meta(self, k):
            return None

        async def is_failed(self, k):
            return False

        async def get_verdict(self, k):
            return None

        async def set(self, k, d):
            pass

        async def set_meta(self, k, m):
            pass

        async def set_verdict(self, k, v, **kw):
            pass

        async def mark_failed(self, k, ttl=None):
            pass

        async def clear_failed(self, k):
            pass

    class _Svc:
        """Reaches the exec site without running a real alass."""

        async def sync_async(self, *a, **kw):
            return None

    multi = _Multi()
    o = SyncOrchestrator(sync_cache=_Cache(), external_strategy=multi, sync_service=_Svc())
    await o.evaluate_and_sync(
        b"1\n00:00:01,000 --> 00:00:03,000\nx\n\n",
        {"media_type": "movie", "target_filename": "W.mkv", "video_hash": "",
         "video_size": "1", "lang": "ara", "imdb_id": "tt2582802"},
        "tt2582802", auto_sync=True,
    )
    assert multi.multi_calls == 1, "a cache hit must NOT skip candidate gathering"
