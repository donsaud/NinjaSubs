"""Per-title sync-outcome telemetry.

The orchestrator logs every request's outcome, but no single line answers "is
Dexter syncing better than Whiplash?". ``SyncOutcomeTracker`` keeps bounded
in-memory counters keyed by title so the answer is one periodic log line.

These tests pin: exactly-one-bucket classification, per-title isolation,
bounded memory, summary cadence, and that recording is observability only
(never changes serving).
"""

from __future__ import annotations

from app.services.sync.outcomes import (
    LARGE_OFFSET_SERVED,
    NO_REFERENCE,
    REJECTED,
    VERIFIED,
    SyncOutcomeTracker,
    classify_outcome,
)

# ===========================================================================
# classify_outcome: exactly one bucket per terminal decision
# ===========================================================================


def test_verified_serving_classifies_as_verified():
    assert classify_outcome(True, "original", False) == VERIFIED


def test_large_offset_serving_stays_distinct_from_verified():
    """The large-offset path serves with state UNVERIFIED by design. Collapsing
    it into VERIFIED in aggregates would manufacture verified claims."""
    assert (
        classify_outcome(True, "alass_corrected_large_offset", False)
        == LARGE_OFFSET_SERVED
    )


def test_large_offset_matching_is_case_insensitive():
    assert classify_outcome(True, "ALASS_Corrected_Large_Offset", False) == LARGE_OFFSET_SERVED


def test_unserved_with_rejections_classifies_as_rejected():
    assert classify_outcome(False, None, True) == REJECTED


def test_unserved_without_rejections_classifies_as_no_reference():
    assert classify_outcome(False, None, False) == NO_REFERENCE


# ===========================================================================
# Tracker: per-title isolation
# ===========================================================================


def test_counts_are_isolated_per_title():
    tracker = SyncOutcomeTracker(summary_every=10_000)
    tracker.record("tt2582802", None, None, VERIFIED)
    tracker.record("tt2582802", None, None, VERIFIED)
    tracker.record("tt0773262", 8, 4, REJECTED)
    snap = tracker.snapshot()
    assert snap["total"] == 3
    assert snap["totals"][VERIFIED] == 2
    assert snap["totals"][REJECTED] == 1
    assert snap["titles"][("tt2582802", "movie", "x")][VERIFIED] == 2
    assert snap["titles"][("tt0773262", "8", "4")][REJECTED] == 1


def test_movie_and_series_keys_do_not_collide():
    tracker = SyncOutcomeTracker(summary_every=10_000)
    tracker.record("tt1", None, None, VERIFIED)
    tracker.record("tt1", 1, 2, VERIFIED)
    snap = tracker.snapshot()
    assert len(snap["titles"]) == 2


def test_title_key_normalizes_case_and_whitespace():
    tracker = SyncOutcomeTracker(summary_every=10_000)
    tracker.record("TT2582802", None, None, VERIFIED)
    tracker.record("tt2582802", None, None, VERIFIED)
    snap = tracker.snapshot()
    assert len(snap["titles"]) == 1
    assert snap["total"] == 2


def test_unknown_outcome_is_ignored_not_counted():
    tracker = SyncOutcomeTracker(summary_every=10_000)
    tracker.record("tt1", None, None, "bogus")
    assert tracker.snapshot()["total"] == 0


# ===========================================================================
# Tracker: bounded memory and summary cadence
# ===========================================================================


def test_title_table_is_bounded():
    tracker = SyncOutcomeTracker(summary_every=10_000, max_titles=3)
    for i in range(10):
        tracker.record(f"tt{i}", None, None, VERIFIED)
    snap = tracker.snapshot()
    assert snap["total"] == 10
    assert snap["totals"][VERIFIED] == 10
    assert len(snap["titles"]) <= 4  # 3 titles + overflow bucket


def test_summary_logs_on_cadence(caplog):
    import logging

    tracker = SyncOutcomeTracker(summary_every=5)
    with caplog.at_level(logging.INFO, logger="app.services.sync.outcomes"):
        for _ in range(4):
            tracker.record("tt1", None, None, VERIFIED)
        assert not [r for r in caplog.records if "sync-outcomes" in r.message]
        tracker.record("tt1", None, None, REJECTED)
    assert any("sync-outcomes" in r.message for r in caplog.records)


def test_summary_names_weakest_titles(caplog):
    import logging

    tracker = SyncOutcomeTracker(summary_every=2)
    with caplog.at_level(logging.INFO, logger="app.services.sync.outcomes"):
        tracker.record("tt2582802", None, None, REJECTED)
        tracker.record("tt0773262", 8, 4, VERIFIED)
    messages = " ".join(r.message for r in caplog.records if "sync-outcomes" in r.message)
    assert "tt2582802" in messages
    assert "rejected=1" in messages


def test_summary_contains_no_subtitle_content(caplog):
    """Only title keys and counters may appear; never filenames or text."""
    import logging

    tracker = SyncOutcomeTracker(summary_every=1)
    with caplog.at_level(logging.INFO, logger="app.services.sync.outcomes"):
        tracker.record("tt1", 1, 2, VERIFIED)
    for record in caplog.records:
        assert ".mkv" not in record.message
        assert ".srt" not in record.message


def test_recording_never_raises():
    tracker = SyncOutcomeTracker()
    tracker.record(None, None, None, VERIFIED)
    tracker.record("tt1", None, None, VERIFIED)
    assert tracker.snapshot()["total"] == 2


def test_snapshot_is_a_copy():
    tracker = SyncOutcomeTracker(summary_every=10_000)
    tracker.record("tt1", None, None, VERIFIED)
    snap = tracker.snapshot()
    snap["totals"][VERIFIED] = 999
    assert tracker.snapshot()["totals"][VERIFIED] == 1
