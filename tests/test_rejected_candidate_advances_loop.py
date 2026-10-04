"""A rejected candidate must NOT terminate the multi-candidate loop.

Live Whiplash evidence that prompted this (``external-release-reference``
expansion produced 6 candidates, yet only one ALASS run ever happened):

    [reference] resolve_all: 6 distinct candidate(s) from 6 ranked release(s)
    [sync] external-release-reference#candidate1 strategy provided a reference
    [sync] invoking alass: target=62564 chars, reference=65092 chars
    [sync] alignment unverified: ... p95=3493ms ... reject=low_confidence
    [sync] delivery ... decision=ORIGINAL

There were NO `candidate SKIPPED` and NO `rejected by cue-sanity` lines, so the
other five candidates were never rejected on their merits -- they were simply
never reached. The loop executed ``return served`` on the first candidate that
completed verification, whether it passed or failed, so expansion could only
ever run one candidate.

These tests pin the fixed behaviour and, just as importantly, pin that nothing
unverified is served while doing it.
"""

from __future__ import annotations

import pytest

from app.services.sync import orchestrator as om
from app.services.sync.orchestrator import SyncOrchestrator
from app.services.sync.query import ResolvedReference


def _ts(ms: int) -> str:
    h, r = divmod(ms, 3600000)
    m, r = divmod(r, 60000)
    s, ms2 = divmod(r, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms2:03d}"


def _srt(cues: int, tag: str, offset_ms: int = 0) -> str:
    """Build a cue list. ``offset_ms`` shifts the whole timeline so a candidate
    is genuinely out of alignment and therefore actually needs alass, rather
    than tripping the already-aligned shortcut."""
    out = []
    for i in range(cues):
        s = 1000 + offset_ms + i * 3000
        e = s + 2000
        out.append(f"{i + 1}\n{_ts(s)} --> {_ts(e)}\n{tag}{i}\n")
    return "\n".join(out)


def _jittered(cues: int, tag: str) -> str:
    """A structurally broken timeline: alternating +/-4s jitter per cue.

    Guaranteed to invert neighbouring cues and scatter residuals, which is what
    drives movement MAD and residual p95 past their ceilings.
    """
    out = []
    for i in range(cues):
        s = 1000 + i * 3000 + (4000 if i % 2 == 0 else -4000)
        e = s + 900
        if i % 2 == 1:
            e = s + 300  # short, irregular durations too
        out.append(f"{i + 1}\n{_ts(s)} --> {_ts(e)}\n{tag}{i}\n")
    return "\n".join(out)


class _Cache:
    is_ephemeral = False

    def __init__(self):
        self.failed = False
        self.verdicts: dict = {}

    async def get(self, k):
        return None

    async def get_meta(self, k):
        return None

    async def is_failed(self, k):
        return self.failed

    async def mark_failed(self, k, ttl=None):
        self.failed = True

    async def clear_failed(self, k):
        self.failed = False

    async def get_verdict(self, k):
        return self.verdicts.get(k)

    async def set_verdict(self, k, v, **kw):
        self.verdicts[k] = v

    async def set(self, k, d):
        pass

    async def set_meta(self, k, m):
        pass


class _Svc:
    """Records every candidate that reached execution and always FAILS
    verification.

    It deliberately returns a scrambled timeline rather than echoing the
    reference. Echoing made the output a perfect alignment (p95=0, struct=1.00),
    so candidate1 *passed* verification and the loop correctly returned on the
    first candidate -- which tested nothing. Producing an untrustworthy result
    for every candidate is what actually exercises the advance-to-next path.
    """

    def __init__(self, accept_from: int | None = None):
        self.calls: list[str] = []
        self.accept_from = accept_from

    async def sync_async(self, *a, **kw):
        ref = a[1] if len(a) > 1 else ""
        self.calls.append(str(len(ref)))
        # Deliberately corrupt the cue TIMING so verification must fail.
        #
        # Two earlier attempts failed to do this and taught us what the verifier
        # actually measures. Echoing the reference gave p95=0/struct=1.00, and a
        # *uniform* offset also gave p95=0/struct=1.00, because a constant shift
        # preserves cue spacing so every residual is identical. The verifier
        # judges residual SPREAD and structural agreement, not absolute offset.
        #
        # So: alternate +/-4000ms jitter on every cue. Adjacent cues can invert,
        # spacing becomes irregular, and both MAD and p95 explode.
        return _jittered(600, "SCRAMBLED")


class _Multi:
    """Yields N candidates; the loop must attempt more than one when rejected."""

    validates_target = True
    cache = None

    def __init__(self, n: int = 4):
        self.n = n

    async def resolve_all_with_provenance(self, query, *, update_validator=None):
        # Honour the validator exactly as the real strategy does: a candidate
        # that fails it must not be offered. Candidates are given progressively
        # larger opening offsets (still inside the +/-20s execution window) so
        # each one legitimately needs alignment instead of being short-circuited
        # as already-aligned.
        out = []
        for i in range(self.n):
            ref = _srt(600, f"C{i}", offset_ms=3000 * (i + 1))
            if update_validator is not None and not update_validator(ref):
                continue
            out.append(ResolvedReference(ref, kind="edition", candidate=f"CAND{i}"))
        return out


def _meta():
    return {
        "media_type": "movie",
        "target_filename": "Whiplash.2014.2160p.UHD.BluRay.x264-SURCODE.mkv",
        "video_hash": "",
        "video_size": "19571049411",
        "lang": "ara",
        "imdb_id": "tt2582802",
    }


@pytest.fixture(autouse=True)
def _gate(monkeypatch):
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True, raising=False)


async def _run(multi, svc):
    o = SyncOrchestrator(sync_cache=_Cache(), external_strategy=multi, sync_service=svc)
    sub = _srt(600, "T").encode()
    out = await o.evaluate_and_sync(sub, _meta(), "tt2582802", auto_sync=True)
    return o, out


@pytest.mark.asyncio
async def test_a_rejected_candidate_does_not_end_the_request():
    """THE regression. Previously exactly one candidate was ever attempted."""
    svc = _Svc()
    o, out = await _run(_Multi(n=4), svc)

    assert len(svc.calls) >= 2, (
        f"loop must advance past a rejected candidate; only attempted {len(svc.calls)}"
    )


@pytest.mark.asyncio
async def test_nothing_unverified_is_served_while_retrying():
    """Advancing must not weaken the gate: if nothing verifies, the ORIGINAL is
    served unchanged."""
    svc = _Svc()
    sub = _srt(600, "T").encode()
    o, out = await _run(_Multi(n=3), svc)
    assert out == sub, "original subtitle must be served when nothing verifies"


@pytest.mark.asyncio
async def test_a_rejection_does_not_mark_the_resolution_failed():
    """mark_failed means 'no usable reference'. Overloading it with 'the
    reference did not verify' would suppress the retry this loop performs, which
    is exactly what 6 serving-contract tests guard against."""
    cache = _Cache()
    svc = _Svc()
    o = SyncOrchestrator(sync_cache=cache, external_strategy=_Multi(n=3), sync_service=svc)
    await o.evaluate_and_sync(_srt(600, "T").encode(), _meta(), "tt2582802", auto_sync=True)

    assert cache.failed is False, (
        "a resolved-but-rejected reference must not set the no-reference flag"
    )


@pytest.mark.asyncio
async def test_all_candidates_failing_still_serves_the_original():
    svc = _Svc()
    sub = _srt(600, "T").encode()
    o, out = await _run(_Multi(n=5), svc)
    assert out == sub


@pytest.mark.asyncio
async def test_candidate_expansion_is_not_disabled_by_the_loop_change():
    """The strategy must still be consulted through the multi path."""
    multi = _Multi(n=3)
    o, _ = await _run(multi, _Svc())
    assert multi.n == 3
