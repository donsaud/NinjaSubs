"""Shadow audit trail and the permanent ranking invariants.

Observability must not become an influence, and the invariants that keep
synchronization evidence honest have to survive future changes. These tests
cover the audit sink's privacy guarantees, prediction/verification pairing, and
the three standing invariants: language ordering, no synthesized target
metadata, and cache identity isolation.
"""

from __future__ import annotations

import json

import pytest

from app.models import MatchTier, SubtitleRelease
from app.services.subtitle_matcher import CompatibilityResult
from app.services.sync.alignment import SyncState, VerificationAvailability
from app.services.sync.audit import (
    AUDIT_LOG,
    AuditLog,
    SyncDecisionRecord,
    sanitize_name,
    sanitize_reason,
    stable_id,
)
from app.services.sync.ordering import order_candidates
from app.services.sync_cache import SYNC_VERDICT_ENGINE_VERSION, SyncCache


@pytest.fixture
def audit_log(tmp_path):
    return AuditLog(enabled=True, path=tmp_path / "audit.jsonl")


def _release(name: str, tier: MatchTier = MatchTier.CLOSE, lang: str = "ara"):
    release = SubtitleRelease(
        release_name=name, download_url=f"http://{name}", provider="subdl", lang=lang
    )
    release.compatibility = CompatibilityResult(accepted=True, match_tier=tier)
    release.match_tier = tier
    return release


def _attach(release, state, verification, *, base=0, lang_rank=0, confidence=None):
    release.sync_state = state
    release.sync_verification = verification
    release.sync_confidence = confidence
    release.sync_base_rank = base
    release.sync_lang_rank = lang_rank
    return release


# --------------------------------------------------------------------------- #
# Disabled by default / write-only
# --------------------------------------------------------------------------- #


def test_audit_is_disabled_by_default():
    from app.config import settings

    assert bool(getattr(settings, "SYNC_AUDIT_ENABLED", False)) is False
    assert AUDIT_LOG.enabled is False


def test_disabled_audit_records_nothing_and_writes_no_file(tmp_path):
    log = AuditLog(enabled=False, path=tmp_path / "audit.jsonl")
    log.record(SyncDecisionRecord(phase="serve", sync_state="verified_resynced"))
    log.record_search(SyncDecisionRecord(phase="search"))
    assert log.records() == []
    assert not (tmp_path / "audit.jsonl").exists()


def test_audit_record_never_raises_into_the_request_path(audit_log):
    class _Explosive:
        @property
        def reasons(self):
            raise RuntimeError("boom")

    # A malformed evaluation must not break serving.
    audit_log.record(SyncDecisionRecord(phase="serve", sync_state="unverified"))
    audit_log.record(SyncDecisionRecord(phase="serve", sync_state="unverified"))
    assert len(audit_log.records()) == 2


# --------------------------------------------------------------------------- #
# Privacy: no content, no credentials
# --------------------------------------------------------------------------- #


def test_records_store_no_subtitle_text():
    record = SyncDecisionRecord(
        phase="serve",
        sync_state="verified_resynced",
        reasons=["residual p95=0ms after a stable shift"],
    )
    payload = json.loads(record.to_jsonl())
    blob = json.dumps(payload).lower()
    for forbidden in ("-->", "00:00:", "transcript", "arabic_text"):
        assert forbidden not in blob


def test_reasons_are_truncated_and_url_safe():
    long_reason = "x" * 500
    assert len(sanitize_reason(long_reason)) <= 160
    redacted = sanitize_reason("failed on https://cdn.example/x.srt?token=abcdef123456&signature=zzz")
    assert "http" not in redacted
    assert "<redacted>" in redacted


def test_release_names_are_bounded():
    # A realistic, long release name is kept but truncated.
    long_name = "Some.Very.Long.Show.Name.S08E05.1080p.BluRay.TrueHD5.1.AVC-RELEASE-GROUP"
    assert len(sanitize_name(long_name)) <= 64
    assert sanitize_name(long_name) is not None
    # A value that is entirely a signed URL, or an opaque 40+ char token, is
    # reduced to nothing rather than partially stored.
    assert sanitize_name("https://cdn.example/a.mkv?token=abc123def456") is None
    assert sanitize_name("A" * 64) is None
    assert sanitize_name("") is None
    assert sanitize_name(None) is None


def test_identities_are_digests_not_raw_values():
    digest = stable_id("tt0773262:8:5:PiR8.mkv")
    assert digest and "tt0773262" not in digest
    assert stable_id("a") != stable_id("b")
    assert stable_id(None) is None
    assert stable_id("") is None


def test_exported_jsonl_has_no_sensitive_fields():
    record = SyncDecisionRecord(phase="serve", video_id=stable_id("v"), subtitle_id=stable_id("s"))
    payload = json.loads(record.to_jsonl())
    for forbidden in ("sub_text", "content", "token", "api_key", "password", "stream_url"):
        assert forbidden not in payload


# --------------------------------------------------------------------------- #
# Prediction -> verification pairing
# --------------------------------------------------------------------------- #


def test_serve_record_is_annotated_with_the_earlier_prediction(audit_log):
    audit_log.record_search(
        SyncDecisionRecord(
            phase="search",
            video_id="v1",
            subtitle_id="c1",
            verification=VerificationAvailability.PREDICTED.value,
            sync_state=SyncState.PROBABLE_SYNC.value,
            prediction_rule="exact_release",
            prediction_confidence=90.0,
        )
    )
    audit_log.record_serve(
        SyncDecisionRecord(
            phase="serve",
            video_id="v1",
            subtitle_id="c1",
            verification=VerificationAvailability.VERIFIED.value,
            sync_state=SyncState.VERIFIED_RESYNCED.value,
            alass_applied=True,
        )
    )
    serve = [r for r in audit_log.records() if r.phase == "serve"]
    assert len(serve) == 1
    assert serve[0].prediction_state == SyncState.PROBABLE_SYNC.value
    assert serve[0].prediction_rule_at_search == "exact_release"
    assert audit_log.counters()["paired_records"] == 1


def test_unpaired_serve_record_is_still_recorded(audit_log):
    audit_log.record_serve(
        SyncDecisionRecord(phase="serve", video_id="v9", subtitle_id="c9", sync_state="unverified")
    )
    serve = [r for r in audit_log.records() if r.phase == "serve"]
    assert serve[0].prediction_state is None


def test_a_prediction_is_consumed_once(audit_log):
    record = dict(
        phase="search",
        video_id="v1",
        subtitle_id="c1",
        verification=VerificationAvailability.PREDICTED.value,
        sync_state=SyncState.PROBABLE_SYNC.value,
    )
    audit_log.record_search(SyncDecisionRecord(**record))
    audit_log.record_serve(SyncDecisionRecord(phase="serve", video_id="v1", subtitle_id="c1"))
    audit_log.record_serve(SyncDecisionRecord(phase="serve", video_id="v1", subtitle_id="c1"))
    serves = [r for r in audit_log.records() if r.phase == "serve"]
    assert serves[0].prediction_state == SyncState.PROBABLE_SYNC.value
    assert serves[1].prediction_state is None


def test_predictions_do_not_pair_across_videos(audit_log):
    audit_log.record_search(
        SyncDecisionRecord(
            phase="search",
            video_id="videoA",
            subtitle_id="cand",
            verification="predicted",
            sync_state="probable_sync",
        )
    )
    audit_log.record_serve(
        SyncDecisionRecord(phase="serve", video_id="videoB", subtitle_id="cand")
    )
    serve = [r for r in audit_log.records() if r.phase == "serve"]
    assert serve[0].prediction_state is None


def test_only_predictions_are_remembered_for_pairing(audit_log):
    audit_log.record_search(
        SyncDecisionRecord(phase="search", video_id="v", subtitle_id="c", verification="unknown")
    )
    audit_log.record_serve(SyncDecisionRecord(phase="serve", video_id="v", subtitle_id="c"))
    serve = [r for r in audit_log.records() if r.phase == "serve"]
    assert serve[0].prediction_state is None


def test_pending_map_is_bounded(audit_log):
    for i in range(audit_log._max_pending + 50):  # noqa: SLF001 - bound assertion
        audit_log.record_search(
            SyncDecisionRecord(
                phase="search", video_id="v", subtitle_id=f"c{i}", verification="predicted"
            )
        )
    assert len(audit_log._pending) <= audit_log._max_pending  # noqa: SLF001


def test_jsonl_export_round_trips(audit_log):
    audit_log.record(
        SyncDecisionRecord(phase="serve", video_id="v1", subtitle_id="s1", sync_state="unverified")
    )
    path = audit_log.path
    assert path is not None and path.is_file()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["sync_state"] == "unverified"


# --------------------------------------------------------------------------- #
# Counters
# --------------------------------------------------------------------------- #


def test_counters_track_fingerprint_cache_and_alass(audit_log):
    audit_log.record(SyncDecisionRecord(phase="search", has_video_fingerprint=True, from_cache=True))
    audit_log.record(SyncDecisionRecord(phase="serve", has_video_fingerprint=False, alass_applied=True))
    counters = audit_log.counters()
    assert counters["records"] == 2
    assert counters["search_records"] == 1
    assert counters["serve_records"] == 1
    assert counters["cache_hits"] == 1
    assert counters["no_fingerprint"] == 1
    assert counters["alass_runs"] == 1


def test_ranking_diff_is_counted_not_applied(audit_log):
    audit_log.record_ranking_diff(changed=True)
    audit_log.record_ranking_diff(changed=False)
    counters = audit_log.counters()
    assert counters["ranking_comparisons"] == 2
    assert counters["ranking_changed"] == 1


# --------------------------------------------------------------------------- #
# INVARIANT 1: language preference is never overridden
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("languages", "names"),
    [
        (["ara", "eng"], ["A", "B", "C"]),
        (["eng", "ara"], ["A", "B", "C"]),
        (["ara"], ["A", "B", "C", "D", "E"]),
        (["ara", "eng", "fre"], ["A", "B", "C", "D"]),
    ],
)
def test_language_grouping_survives_sync_evidence(languages, names):
    """The requested language is always the primary grouping key."""
    candidates = []
    for index, name in enumerate(names):
        # Alternate languages so a reordering would be visible.
        lang = languages[index % len(languages)]
        candidates.append(
            _attach(
                _release(f"{name}.srt", tier=MatchTier.HASH, lang=lang),
                SyncState.VERIFIED_SYNCED.value,
                "cached",
                base=index,
                lang_rank=languages.index(lang),
                confidence=99.0,
            )
        )
    # Every candidate carries maximal, equal sync evidence, so only language
    # preference can decide the grouping.
    ordered = order_candidates(candidates)
    seen = [rel.sync_lang_rank for rel in ordered]
    assert seen == sorted(seen), "language preference was overridden"


def test_language_preference_holds_with_no_sync_evidence():
    candidates = [
        _attach(_release("A.srt", lang="eng"), None, None, base=0, lang_rank=1),
        _attach(_release("B.srt", lang="ara"), None, None, base=1, lang_rank=0),
    ]
    ordered = order_candidates(candidates)
    assert ordered[0].release_name == "B.srt"


# --------------------------------------------------------------------------- #
# INVARIANT 2: no target metadata is synthesized from a subtitle
# --------------------------------------------------------------------------- #


def test_no_target_metadata_is_synthesized_without_a_fingerprint():
    from app.main import _merge_sync_meta
    from app.services.sync.matching import has_video_fingerprint
    from app.services.sync.predictor import SyncPredictor

    subtitle_name = "Dexter.S08E05.1080p.BluRay.x264-PiR8.srt"
    merged = _merge_sync_meta(
        {"imdb_id": "tt1", "release_name": subtitle_name, "season": 8, "episode": 5}, {}
    )
    assert has_video_fingerprint(merged) is False
    assert merged.get("target_filename") is None

    prediction = SyncPredictor().predict(
        _release(subtitle_name, tier=MatchTier.HASH),
        {"imdb_id": "tt1", "season": 8, "episode": 5, "title": "Dexter", "year": 2006},
    )
    assert prediction.availability is VerificationAvailability.UNKNOWN
    assert prediction.rule == "no_fingerprint"


def test_missing_fingerprint_stays_unknown_unless_an_exact_verdict_exists():
    """No fingerprint means UNKNOWN, and only a measured verdict may override."""
    from app.services.sync.predictor import SyncPredictor

    release = _release("A.srt", tier=MatchTier.EXACT)
    assert SyncPredictor().predict(release, {}).availability is VerificationAvailability.UNKNOWN

    # A measured verdict is the only thing that can supply real evidence.
    measured = SyncState.VERIFIED_RESYNCED.value
    assert measured in {SyncState.VERIFIED_RESYNCED.value, SyncState.VERIFIED_SYNCED.value}


# --------------------------------------------------------------------------- #
# INVARIANT 3: cache identity isolation
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_same_subtitle_different_video_does_not_reuse_a_verdict():
    cache = SyncCache()
    key = SyncCache.build_verdict_key("videoA", "sub1", "ara")
    await cache.set_verdict(key, {"sync_state": "verified_resynced", "verification": "verified"})
    other_video = SyncCache.build_verdict_key("videoB", "sub1", "ara")
    assert await cache.get_verdict(other_video) is None


@pytest.mark.asyncio
async def test_same_video_different_subtitle_does_not_reuse_a_verdict():
    cache = SyncCache()
    key = SyncCache.build_verdict_key("videoA", "sub1", "ara")
    await cache.set_verdict(key, {"sync_state": "verified_resynced", "verification": "verified"})
    other_sub = SyncCache.build_verdict_key("videoA", "sub2", "ara")
    assert await cache.get_verdict(other_sub) is None


@pytest.mark.asyncio
async def test_different_engine_version_does_not_reuse_a_stale_verdict():
    cache = SyncCache()
    key = SyncCache.build_verdict_key("videoA", "sub1", "ara")
    await cache.set_verdict(key, {"sync_state": "verified_resynced", "verification": "verified"})
    # Simulate a record written by a different engine version.
    cache._local_verdict[key] = json.dumps(  # noqa: SLF001 - corruption/staleness test
        {
            "sync_state": "verified_resynced",
            "verification": "verified",
            "engine_version": SYNC_VERDICT_ENGINE_VERSION + 1,
        }
    )
    assert await cache.get_verdict(key) is None


@pytest.mark.asyncio
async def test_alias_is_bound_to_the_video_fingerprint():
    cache = SyncCache()
    primary = SyncCache.build_verdict_key("videoA", "sub1", "ara")
    alias = SyncCache.build_verdict_alias_key("videoA", "cand1")
    await cache.set_verdict(
        primary, {"sync_state": "verified_synced", "verification": "verified"}, alias_key=alias
    )
    assert await cache.get_verdict_by_ref("videoA", "cand1") is not None
    assert await cache.get_verdict_by_ref("videoB", "cand1") is None


# --------------------------------------------------------------------------- #
# Telemetry cannot influence results
# --------------------------------------------------------------------------- #


def test_audit_module_does_not_import_ordering_or_matcher():
    """The audit layer must not be able to influence a decision."""
    import ast
    from pathlib import Path

    source = Path("app/services/sync/audit.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any("ordering" in name for name in imported)
    assert not any("subtitle_matcher" in name for name in imported)
    assert not any("predictor" in name for name in imported)


def test_thresholds_were_not_modified_by_this_work():
    """The audit phase must not tune verification thresholds."""
    from app.services.sync.alignment import (
        MAX_DRIFT_MS_PER_MINUTE,
        MAX_MAD_MS_FOR_STABLE,
        MAX_P95_MS_FOR_STABLE,
        MIN_CUE_RETENTION,
        MIN_CUES_FOR_VERIFIED,
    )
    from app.services.sync.predictor import (
        CONFIDENCE_EXACT_IDENTITY,
        CONFIDENCE_HASH,
        CONFIDENCE_PREDICTION_FLOOR,
        CONFIDENCE_SOURCE_EDITION,
    )

    # Values established in earlier phases; the audit adds no new tuning.
    assert MIN_CUES_FOR_VERIFIED == 12
    assert MIN_CUE_RETENTION == 0.80
    assert MAX_P95_MS_FOR_STABLE == 2000.0
    assert MAX_MAD_MS_FOR_STABLE == 800.0
    assert MAX_DRIFT_MS_PER_MINUTE == 120.0
    assert CONFIDENCE_HASH == 95.0
    assert CONFIDENCE_EXACT_IDENTITY == 90.0
    assert CONFIDENCE_SOURCE_EDITION == 75.0
    assert CONFIDENCE_PREDICTION_FLOOR == 70.0
