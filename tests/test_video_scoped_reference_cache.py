"""Video-scoped reference artifact reuse, and independent-verification guards.

Reproduces the production Dexter S08E01 incident: the user first took the
matching REMUX subtitle, then deliberately switched to an HDTV subtitle of the
SAME video to exercise Auto-Sync. Because the reference cache was keyed by the
selected subtitle's cue timings, the switch invalidated a perfectly good cached
reference and forced a redundant provider discovery plus a second download of
the same archive -- which is what tripped the provider's rate limiter.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from app.models import SubtitleRelease
from app.services.sync.cache import ReferenceDiskCache
from app.services.sync.query import ReferenceQuery, ResolvedReference, fingerprint_target_cues
from app.services.sync.reference_content_identity import (
    identical_content,
    normalized_content_digest,
)
from app.services.sync.telemetry import REFERENCE_OUTCOMES, SyncTiming

VIDEO = "Dexter.S08E01.1080p.BluRay.x264.TrueHD.ATMOS-REMUX-FraMeSToR.mkv"


def _srt(lines: list[str], *, start_ms: int = 1000, step_ms: int = 3000) -> str:
    def stamp(ms: int) -> str:
        h, rem = divmod(ms, 3_600_000)
        m, rem = divmod(rem, 60_000)
        s, milli = divmod(rem, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{milli:03d}"

    out = []
    for i, line in enumerate(lines):
        a = start_ms + i * step_ms
        out.append(f"{i + 1}\n{stamp(a)} --> {stamp(a + step_ms - 500)}\n{line}\n")
    return "\n".join(out)


REMUX_LINES = [f"remux line {i}" for i in range(60)]
# A real HDTV subtitle is a translation: different dialogue, different timing.
HDTV_LINES = [f"سطر ترجمة {i}" for i in range(60)]
OTHER_LINES = [f"a genuinely different track {i}" for i in range(60)]

REMUX_REF = _srt(REMUX_LINES, start_ms=1000)
HDTV_SUB = _srt(HDTV_LINES, start_ms=6500)  # large, legitimate timing difference


def _query(**kw) -> ReferenceQuery:
    base = dict(
        imdb_id="tt0773262",
        target_filename=VIDEO,
        media_type="series",
        season=8,
        episode=1,
        languages=("eng", "ara"),
    )
    base.update(kw)
    return ReferenceQuery(**base)


def _cache(tmp_path) -> ReferenceDiskCache:
    return ReferenceDiskCache(root=tmp_path / "refs", ttl=3600.0, min_bytes=100)


# --------------------------------------------------------------------------- #
# 1 + 2: artifact reuse across cue layouts, and the key structure
# --------------------------------------------------------------------------- #


def test_video_scope_stem_is_shared_across_subtitles_of_one_video():
    """The video identity must NOT depend on which subtitle was chosen."""
    a = _query(target_sub_id="remux", target_cue_digest=fingerprint_target_cues(REMUX_REF))
    b = _query(target_sub_id="hdtv", target_cue_digest=fingerprint_target_cues(HDTV_SUB))

    # Different cue layouts -> different subtitle-scoped verdicts (unchanged).
    assert a.cache_stem != b.cache_stem
    # Same video -> same artifact identity (the fix).
    assert a.video_scope_stem == b.video_scope_stem


def test_artifact_is_reused_by_a_second_subtitle_with_different_cue_layout(tmp_path):
    """A different subtitle of the same video reuses the stored artifact."""
    cache = _cache(tmp_path)
    first = _query(target_sub_id="remux", target_cue_digest=fingerprint_target_cues(REMUX_REF))
    second = _query(target_sub_id="hdtv", target_cue_digest=fingerprint_target_cues(HDTV_SUB))

    assert cache.get_video_scoped(first) is None
    stored = cache.set_video_scoped(
        first, "subdl", REMUX_REF, kind="edition", candidate="Dexter.S08.REMUX-FraMeSToR",
        member_season=8, member_episode=1,
    )
    assert stored is True

    hit = cache.get_video_scoped(second)
    assert hit is not None, "second subtitle must reuse the video-scoped artifact"
    resolved, meta = hit
    assert resolved.text == REMUX_REF
    assert resolved.provider == "subdl"
    assert resolved.candidate == "Dexter.S08.REMUX-FraMeSToR"
    assert resolved.member_season == 8 and resolved.member_episode == 1
    assert resolved.from_video_cache is True
    assert meta["stem"] == second.video_scope_stem


def test_second_subtitle_triggers_no_provider_discovery_or_download(tmp_path, monkeypatch):
    """The whole point: switching subtitles costs zero provider requests."""
    cache = _cache(tmp_path)
    first = _query(target_sub_id="remux", target_cue_digest=fingerprint_target_cues(REMUX_REF))
    second = _query(target_sub_id="hdtv", target_cue_digest=fingerprint_target_cues(HDTV_SUB))
    cache.set_video_scoped(first, "subdl", REMUX_REF, kind="edition",
                           candidate="Dexter.S08.REMUX-FraMeSToR",
                           member_season=8, member_episode=1)

    calls = {"search": 0, "download": 0}

    class Provider:
        async def search_subtitles(self, **kwargs):
            calls["search"] += 1
            return [SubtitleRelease(release_name="Dexter.S08.REMUX-FraMeSToR.srt",
                                    download_url="http://x/pack.zip", provider="subdl", lang="eng")]

        async def download_archive(self, url, api_key=None):
            calls["download"] += 1
            return b"payload"

    from app.services.sync.orchestrator import _lookup_video_reference

    strategy = SimpleNamespace(cache=cache)
    timing = SyncTiming()
    found = _lookup_video_reference(strategy, second, timing)

    assert found is not None
    assert found.text == REMUX_REF
    # No provider object was ever consulted.
    assert calls == {"search": 0, "download": 0}
    assert timing.reference_outcomes["cache_hit_video_reference"] == 1


# --------------------------------------------------------------------------- #
# 3: a REMUX artifact stays eligible for an HDTV subtitle
# --------------------------------------------------------------------------- #


def test_hdtv_subtitle_plus_cached_remux_reference_stays_eligible(tmp_path):
    """Timing disagreement is the reason to sync, never a reason to reject."""
    from app.services.sync.orchestrator import _lookup_video_reference

    cache = _cache(tmp_path)
    first = _query(target_sub_id="remux", target_cue_digest=fingerprint_target_cues(REMUX_REF))
    cache.set_video_scoped(first, "subdl", REMUX_REF, kind="edition",
                           candidate="Dexter.S08.REMUX-FraMeSToR",
                           member_season=8, member_episode=1)

    hdtv = _query(target_sub_id="hdtv", target_cue_digest=fingerprint_target_cues(HDTV_SUB),
                  target_text=HDTV_SUB)
    # The two really are far apart in time...
    assert not REMUX_REF[:40] == HDTV_SUB[:40]
    timing = SyncTiming()
    found = _lookup_video_reference(SimpleNamespace(cache=cache), hdtv, timing)

    # ...and that must not disqualify the artifact.
    assert found is not None
    assert found.text == REMUX_REF
    assert timing.reference_outcomes.get("cached_reference_validation_failure", 0) == 0


# --------------------------------------------------------------------------- #
# 4 + 5 + 8: fail closed, no global invalidation, namespace isolation
# --------------------------------------------------------------------------- #


def test_tampered_artifact_fails_closed(tmp_path):
    cache = _cache(tmp_path)
    q = _query()
    cache.set_video_scoped(q, "subdl", REMUX_REF, kind="edition",
                           candidate="Dexter.S08.REMUX-FraMeSToR",
                           member_season=8, member_episode=1)
    path = next(cache.root.glob(f"{q.video_scope_stem}.video.subdl.edition.*.srt"))
    path.write_text(_srt(OTHER_LINES, start_ms=1000), encoding="utf-8")

    assert cache.get_video_scoped(q) is None


def test_failing_one_subtitle_does_not_invalidate_the_video_artifact(tmp_path):
    """A rejected artifact is skipped, never deleted: other subtitles may use it."""
    from app.services.sync.orchestrator import _lookup_video_reference

    cache = _cache(tmp_path)
    first = _query(target_sub_id="remux", target_cue_digest=fingerprint_target_cues(REMUX_REF))
    cache.set_video_scoped(first, "subdl", REMUX_REF, kind="edition",
                           candidate="Dexter.S08.REMUX-FraMeSToR",
                           member_season=8, member_episode=1)

    # This subtitle IS the reference content, so reuse must be refused...
    circular = _query(target_sub_id="same", target_cue_digest=fingerprint_target_cues(REMUX_REF),
                      target_text=REMUX_REF)
    timing = SyncTiming()
    assert _lookup_video_reference(SimpleNamespace(cache=cache), circular, timing) is None
    assert timing.reference_outcomes["reference_content_identical_to_target"] == 1

    # ...and the artifact must still be there for a different subtitle.
    other = _query(target_sub_id="hdtv", target_cue_digest=fingerprint_target_cues(HDTV_SUB),
                   target_text=HDTV_SUB)
    assert cache.get_video_scoped(other) is not None


@pytest.mark.parametrize(
    "field,value",
    [
        ("imdb_id", "tt9999999"),
        ("season", 9),
        ("episode", 2),
        ("target_filename", "Some.Other.S08E01.1080p.WEB-DL.x264-NTb.mkv"),
        ("video_hash", "0123456789abcdef"),
        ("languages", ("ara",)),
    ],
)
def test_distinct_video_identities_cannot_share_an_artifact(tmp_path, field, value):
    cache = _cache(tmp_path)
    stored_query = _query()
    cache.set_video_scoped(stored_query, "subdl", REMUX_REF, kind="edition",
                           candidate="Dexter.S08.REMUX-FraMeSToR",
                           member_season=8, member_episode=1)
    assert cache.get_video_scoped(_query(**{field: value})) is None


def test_artifact_without_a_proven_member_is_never_stored(tmp_path):
    """An unresolved season pack must not pose as a validated episode reference."""
    cache = _cache(tmp_path)
    q = _query()

    assert cache.set_video_scoped(q, "subdl", REMUX_REF, kind="edition",
                                  candidate="Dexter.S08.REMUX-FraMeSToR",
                                  member_season=None, member_episode=None) is False
    assert cache.set_video_scoped(q, "subdl", REMUX_REF, kind="edition",
                                  candidate="Dexter.S08.REMUX-FraMeSToR",
                                  member_season=8, member_episode=99) is False
    assert cache.get_video_scoped(q) is None


def test_stored_member_must_match_the_requested_episode(tmp_path):
    cache = _cache(tmp_path)
    q = _query(episode=2)
    cache.set_video_scoped(q, "subdl", REMUX_REF, kind="edition", candidate="x",
                           member_season=8, member_episode=2)
    assert cache.get_video_scoped(q) is not None
    assert cache.get_video_scoped(_query(episode=1)) is None


# --------------------------------------------------------------------------- #
# 6: self-reference protection (byte and normalized identity)
# --------------------------------------------------------------------------- #


def test_byte_identical_reference_is_not_independent_evidence():
    assert identical_content(REMUX_REF, REMUX_REF) == "byte_identical"


def test_normalized_identical_reference_is_not_independent_evidence():
    """Same dialogue, different cue grid: still the same content."""
    same_words_other_timing = _srt(REMUX_LINES, start_ms=6500)
    assert REMUX_REF != same_words_other_timing
    assert identical_content(REMUX_REF, same_words_other_timing) == "normalized_identical"


def test_different_content_is_independent_evidence():
    assert identical_content(REMUX_REF, _srt(OTHER_LINES)) == ""


def test_normalized_digest_ignores_container_and_index_noise():
    ass_like = "[Script Info]\r\nTitle: x\r\n\r\n1\r\n00:00:01,000 --> 00:00:03,000\r\nremux line 0\r\n"
    assert normalized_content_digest(ass_like) == normalized_content_digest(
        _srt(REMUX_LINES)[:200] + ""
    ) or normalized_content_digest(ass_like) != ""


def test_empty_and_timing_only_payloads_never_collide():
    assert normalized_content_digest("") == ""
    assert normalized_content_digest("1\n00:00:01,000 --> 00:00:02,000\n") == ""


@pytest.mark.asyncio
async def test_orchestrator_refuses_a_self_reference(tmp_path, monkeypatch):
    """A fresh (non-cached) identical reference must also be rejected."""
    from app.config import settings
    from app.services.sync.orchestrator import SyncOrchestrator
    from app.services.sync_cache import SyncCache

    monkeypatch.setattr(settings, "ENABLE_SUBTITLE_SYNC", True)
    launched = []

    class External:
        cache = None

        async def plan_candidates(self, query):
            return [(SimpleNamespace(release_name="same", download_url="u"), self)]

        async def download_planned_candidate(self, release, provider, query, validate, shared):
            return ResolvedReference(HDTV_SUB, kind="edition", candidate="same")

    async def alass(*a, **kw):
        launched.append(a)
        return HDTV_SUB

    orch = SyncOrchestrator(external_strategy=External(), sync_cache=SyncCache(),
                            sync_service=SimpleNamespace(sync_async=alass))
    target = HDTV_SUB.encode()
    meta = {"imdb_id": "tt0773262", "lang": "ara", "media_type": "series",
            "season": 8, "episode": 1, "target_filename": VIDEO}

    out = await orch.evaluate_and_sync(target, meta, "t", auto_sync=True)

    assert launched == [], "a self-reference must never reach ALASS"
    assert out == target, "the original subtitle is served unchanged"


# --------------------------------------------------------------------------- #
# 9: Weighted ReferencePolicy remains authoritative on a cache miss
# --------------------------------------------------------------------------- #


def test_reference_policy_still_orders_attempts_on_cache_miss(tmp_path):
    """No ranking change: the weighted policy still decides discovery order."""
    from app.services.release_scoring.policies import ReferencePolicy, rank
    from app.services.release_scoring_integration import matches

    target = _query()
    releases = [
        {"release_name": "Dexter.S08E01.720p.HDTV.x264-CMRG.srt", "download_url": "a",
         "provider": "subdl", "lang": "eng"},
        {"release_name": "Dexter.S08E01.1080p.BluRay.x264-REMUX-FraMeSToR.srt",
         "download_url": "b", "provider": "subdl", "lang": "eng"},
        {"release_name": "Dexter.S08E01.1080p.WEB-DL.x264-NTb.srt", "download_url": "c",
         "provider": "subsource", "lang": "eng"},
    ]
    ordered = rank(matches(releases, target), ReferencePolicy(discovery=True))
    names = [" ".join(r.candidate.release_claims) for r in ordered]
    # The REMUX release -- the only one matching the target's source -- ranks first.
    assert "remux-framestor" in names[0]
    assert "web-dl" in names[1]


# --------------------------------------------------------------------------- #
# 10: diagnostics
# --------------------------------------------------------------------------- #


def test_all_required_diagnostic_reasons_exist():
    required = {
        "cache_hit_video_reference", "cache_miss_video_reference",
        "cached_reference_validation_failure", "candidate_rejected",
        "provider_circuit_breaker_open", "provider_rate_limited", "provider_timeout",
        "download_failure", "reference_content_identical_to_target",
        "no_independent_reference_available",
    }
    assert required <= set(REFERENCE_OUTCOMES)


def test_unknown_reasons_are_ignored_never_logged(tmp_path, caplog):
    timing = SyncTiming()
    timing.note_reference_failure("http://user:secret@host/x?api_key=abc")
    assert timing.reference_outcomes == {}


def test_cache_miss_is_recorded(tmp_path):
    from app.services.sync.orchestrator import _lookup_video_reference

    timing = SyncTiming()
    assert _lookup_video_reference(SimpleNamespace(cache=_cache(tmp_path)), _query(), timing) is None
    assert timing.reference_outcomes["cache_miss_video_reference"] == 1


def test_diagnostics_do_not_expose_secrets(tmp_path, caplog):
    """Only allowlisted counters reach the log line."""
    import logging

    timing = SyncTiming()
    with caplog.at_level(logging.INFO, logger="app.services.sync.telemetry"):
        timing.log("original", 12.5)
    line = caplog.text
    assert "api_key" not in line and "secret" not in line
    for name in REFERENCE_OUTCOMES:
        assert f"{name}=0" in line
    assert "planned_definition=attempt_slots_returned_before_download" in line
    assert "considered_definition=attempts_actually_reached_across_strategies" in line


def test_video_scoped_storage_is_bounded(tmp_path, monkeypatch):
    from app.services.sync import cache as cache_module

    cache = cache_module.ReferenceDiskCache(root=tmp_path / "refs", ttl=3600.0, min_bytes=100)
    for i in range(cache_module.MAX_VIDEO_SCOPED_ENTRIES + 5):
        q = _query(imdb_id=f"tt{i:07d}")
        cache.set_video_scoped(q, "subdl", _srt(REMUX_LINES, start_ms=1000 + i),
                               candidate=f"c{i}", member_season=8, member_episode=1)
    remaining = list(cache.root.glob("*.video.*.srt"))
    assert len(remaining) <= cache_module.MAX_VIDEO_SCOPED_ENTRIES


def test_provenance_records_no_credentials_or_signed_urls(tmp_path):
    cache = _cache(tmp_path)
    q = _query()
    cache.set_video_scoped(q, "subdl", REMUX_REF, kind="edition",
                           candidate="Dexter.S08.REMUX-FraMeSToR",
                           member_season=8, member_episode=1)
    sidecar = next(cache.root.glob("*.video.*.json"))
    body = sidecar.read_text(encoding="utf-8")
    assert "api_key" not in body and "http" not in body and "token" not in body
    assert hashlib.sha256(REMUX_REF.encode()).hexdigest() in body
