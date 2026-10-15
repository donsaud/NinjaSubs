"""Regression tests for sync-engine defects.

Covers NUL bytes reaching alass, per-request stream tokens splitting the
payload cache, explicit sync requests joining a non-explicit flight, source
exclusions overwritten between reference tiers, cues shifted before zero
becoming zero-length, the listing badge trusting unverified payloads, and the
unbounded rejected-candidate memo.
"""

import pytest
from cachetools import TTLCache

from app.services.sync.external_strategy import ExternalExactStrategy
from app.services.sync.ninja_points import ConstantSyncModel, LinearSyncModel
from app.services.sync.orchestrator import (
    SyncOrchestrator,
    _add_known_source,
    build_synced_cache_key,
)
from app.services.sync.query import ReferenceQuery
from app.services.sync_cache import SyncCache
from app.services.sync_service import sanitize_subtitle

META = {"imdb_id": "tt0773262", "season": 8, "episode": 5, "target_filename": "Dexter.S08E05.mkv"}


def test_sanitize_strips_nul_bytes():
    raw = "1\n00:00:01,000 --> 00:00:02,000\nhel\x00lo\n"
    assert "\x00" not in sanitize_subtitle(raw)
    assert "hello" in sanitize_subtitle(raw)


def test_stream_token_does_not_change_synced_cache_key():
    a = build_synced_cache_key({**META, "stream_url": "https://cdn.x/v/abc.mkv?token=1111"}, "sub")
    b = build_synced_cache_key({**META, "stream_url": "https://cdn.x/v/abc.mkv?token=2222"}, "sub")
    c = build_synced_cache_key({**META, "stream_url": "https://cdn.x/v/other.mkv?token=1111"}, "sub")
    assert a == b
    assert a != c


def test_explicit_sync_request_does_not_join_a_non_explicit_flight():
    plain = SyncOrchestrator._flight_key(dict(META), "sub", b"bytes")
    explicit = SyncOrchestrator._flight_key({**META, "force_sync": True}, "sub", b"bytes")
    assert plain != explicit
    # Non-explicit keys are unchanged in shape.
    assert not plain.endswith(":explicit")


def test_known_source_digests_accumulate_across_tiers():
    query = ReferenceQuery(imdb_id="tt1")
    query.known_source_digests = _add_known_source(query, "sticky")
    query.known_source_digests = _add_known_source(query, "pack")
    query.known_source_digests = _add_known_source(query, "sticky")
    assert query.known_source_digests == ("sticky", "pack")


@pytest.mark.parametrize(
    "model",
    [ConstantSyncModel(offset_ms=-5000, inliers=3), LinearSyncModel(1.0, -5000, inliers=3)],
)
def test_cues_shifted_entirely_before_zero_are_dropped_not_zero_length(model):
    out = model.apply([(1000, 2000, "gone"), (4000, 6000, "straddles"), (8000, 9000, "kept")])
    assert out == [(0, 1000, "straddles"), (3000, 4000, "kept")]
    assert all(end > start for start, end, _ in out)


@pytest.mark.asyncio
async def test_synced_badge_ignores_unverified_payloads():
    cache = SyncCache()
    key = SyncCache.build_key("tt1", "movie", "video", "sub", "edition", "hash")
    await cache.set(key, b"rejected alass output")
    await cache.set_meta(key, {"sync_state": "rejected", "verification": "verified"})
    assert await cache.find_synced_for("tt1", "sub") is None

    await cache.set_meta(key, {"sync_state": "verified_resynced", "verification": "verified"})
    assert await cache.find_synced_for("tt1", "sub") == b"rejected alass output"


def test_rejected_candidate_memo_is_bounded_and_expiring():
    strategy = ExternalExactStrategy()
    assert isinstance(strategy._rejected, TTLCache)
    assert strategy._rejected.maxsize > 0 and strategy._rejected.ttl > 0
