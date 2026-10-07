"""Temporary simulated inputs test the scout contract, never REAL corpus evidence."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.sync import case_capture as cc
from app.services.sync.boundary_verifier import BoundaryResult
from app.services.sync.ninja_points import render_srt
from app.services.sync.telemetry import current_sync_timing
from app.services.sync.timeline_affine import AffineResult
from tools import boundary_scout as scout


@pytest.fixture
def pair(tmp_path):
    text = render_srt([(10000 + i * 5000, 11000 + i * 5000, "Test speech.") for i in range(50)])
    a, b = tmp_path / "a.srt", tmp_path / "b.srt"
    a.write_bytes(text.encode())
    b.write_bytes(text.replace("Test speech.", "Reference speech.").encode())
    meta = {"imdb_id": "tt1796960", "season": 2, "episode": 2, "provider": "subdl"}
    return (scout.payload(a, {**meta, "release_name": "Homeland.S02E02.HDTV.x264-EVOLVE"}),
            scout.payload(b, {**meta, "release_name": "Homeland.S02E02.1080p.BluRay.x264-GROUP"}))


class FakeAlass:
    def __init__(self, *, failure=False, invalid=False):
        self.calls = 0
        self.failure = failure
        self.invalid = invalid

    def sync(self, target, reference, **kwargs):
        self.calls += 1
        current_sync_timing.get().alass_processes_launched += 1
        if self.failure:
            return None
        cc.capture_alass_bytes(target.encode(), reference, reference, target)
        return None if self.invalid else target


@pytest.fixture
def non_global(monkeypatch):
    monkeypatch.setattr(scout, "attempt_timeline_affine", lambda *a: AffineResult())
    monkeypatch.setattr(scout, "measure_alignment_consistency", lambda *a, **kw: SimpleNamespace(aligned=False))


def test_duplicate_sha_rejected(pair):
    a, b = pair
    b = replace(b, sha256=a.sha256)
    assert scout.pairs([a, b]) == []
    assert scout.evaluate(a, b, FakeAlass())["error_type"] == "ValueError"


def test_cross_episode_rejected(pair):
    a, b = pair
    b = replace(b, episode=3)
    assert scout.pairs([a, b]) == []
    assert scout.evaluate(a, b, FakeAlass())["status"] == "ERROR"


def test_affine_success_never_runs_alass(pair, monkeypatch):
    monkeypatch.setattr(scout, "attempt_timeline_affine", lambda *a: AffineResult(accepted=True))
    service = FakeAlass()
    row = scout.evaluate(*pair, service)
    assert row["status"] == "GLOBAL_SOLVED"
    assert service.calls == 0
    assert row["alass_launched"] is False


def test_alass_failure_safe(pair, non_global):
    row = scout.evaluate(*pair, FakeAlass(failure=True))
    assert row["status"] == "ALASS_FAILED"
    assert row["alass_launched"] is True
    assert row["boundary"] is None
    assert cc.current_capture.get() is None
    assert current_sync_timing.get() is None


def test_unmeasurable_consistency_does_not_abort(pair, non_global, monkeypatch):
    monkeypatch.setattr(scout, "measure_alignment_consistency", lambda *a, **kw: None)
    row = scout.evaluate(*pair, FakeAlass(failure=True))
    assert row["alass_launched"] is True
    assert row["status"] == "ALASS_FAILED"


def test_boundary_acceptance_always_derived(pair, non_global, monkeypatch):
    monkeypatch.setattr(scout, "verify_boundary", lambda *a, **kw: BoundaryResult(accepted=True, classification="EDGE_DISCONTINUITY", holdout_pass=True))
    row = scout.evaluate(*pair, FakeAlass())
    assert row["status"] == "EDGE_CANDIDATE"
    assert row["evidence_class"] == "DERIVED_SCOUTING_ONLY"
    assert row["requires_live_stremio_confirmation"] is True
    rendered = scout.table(scout.report([row], scout.Counter()))
    assert "EVIDENCE_CLASS=DERIVED_SCOUTING_ONLY" in rendered
    assert "REQUIRES_LIVE_STREMIO_CONFIRMATION=true" in rendered


def test_rejected_service_output_never_candidate(pair, non_global, monkeypatch):
    monkeypatch.setattr(scout, "verify_boundary", lambda *a, **kw: BoundaryResult(accepted=True, classification="EDGE_DISCONTINUITY"))
    row = scout.evaluate(*pair, FakeAlass(invalid=True))
    assert row["status"] == "ALASS_FAILED"
    assert scout.report([row], scout.Counter())["summary"]["boundary_EDGE_positives"] == 0


def test_source_bytes_unchanged_and_production_cache_untouched(pair, non_global, monkeypatch, tmp_path):
    from app.services.sync_cache import SyncCache
    def forbidden(*a, **kw):
        pytest.fail("production cache accessed")
    monkeypatch.setattr(SyncCache, "__init__", forbidden)
    before = {p.path: p.path.read_bytes() for p in pair}
    sentinel = tmp_path / "subs_cache" / "sentinel"
    sentinel.parent.mkdir()
    sentinel.write_bytes(b"production cache sentinel")
    scout.evaluate(*pair, FakeAlass())
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert sentinel.read_bytes() == b"production cache sentinel"


@pytest.mark.parametrize("change", [
    {"episode": None}, {"episode": 3}, {"imdb_id": "unknown"},
    {"imdb_id": "tt0773262", "season": 8}, {"imdb_id": "tt0758758"},
    {"release_name": "Homeland.S02E02-E03.HDTV.x264-EVOLVE"},
])
def test_identity_fails_closed(pair, change):
    meta = {"imdb_id": "tt1796960", "season": 2, "episode": 2, "provider": "subdl",
            "release_name": "Homeland.S02E02.HDTV.x264-EVOLVE", **change}
    with pytest.raises(ValueError):
        scout.payload(pair[0].path, meta)


def test_changed_source_refused(pair):
    pair[0].path.write_bytes(b"changed")
    row = scout.evaluate(*pair, FakeAlass())
    assert row["status"] == "ERROR"
    assert not row["alass_launched"]


def test_metadata_privacy():
    for value in ("https://example.invalid/?token=secret", r"C:\Users\someone", "api_key=secret"):
        assert scout.label(value) == "withheld"


def test_discovery_hash_conflict_and_synthetic_manifest(pair, tmp_path):
    import json
    root = tmp_path / "discovery"
    (root / "subs_cache/_meta").mkdir(parents=True)
    for name, episode in (("a", 2), ("b", 3)):
        (root / "subs_cache" / f"{name}.srt").write_bytes(pair[0].path.read_bytes())
        meta = {"imdb_id": "tt1796960", "season": 2, "episode": episode, "provider": "subdl",
                "release_name": f"Homeland.S02E{episode:02d}.HDTV.x264-EVOLVE"}
        (root / "subs_cache/_meta" / f"{name}.json").write_text(json.dumps(meta))
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"payloads": [{"provenance": "synthetic", "path": "a.srt"}]}))
    items, skips = scout.discover(root, [], [manifest])
    assert not items
    assert skips["conflicting_sha_identity"] == 2
    assert skips["non_pristine_manifest_entry"] == 1


def test_production_alass_bridge_argv_raw_bytes_and_cleanup(pair, monkeypatch):
    from app.services import sync_service
    seen = {}
    def process(argv, **kwargs):
        # Invoke the real production wrapper; substitute only the executable in this test.
        assert argv[4:] == ["--split-penalty", "7.0"]
        paths = [Path(p) for p in argv[1:4]]
        seen["paths"] = paths
        assert "Reference speech." in paths[0].read_text()
        assert "Test speech." in paths[1].read_text()
        raw = paths[1].read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        paths[2].write_bytes(raw)
        seen["raw"] = raw
        return SimpleNamespace(returncode=0, stderr=b"", stdout=b"")
    monkeypatch.setattr(sync_service.subprocess, "run", process)
    observer = scout.RawObserver()
    token = cc.current_capture.set(observer)
    try:
        output = sync_service.SubtitleSyncService(alass_path="test-only").sync(
            pair[0].path.read_text(), pair[1].path.read_text(), is_series=True,
            source_confirmed=True)
    finally:
        cc.current_capture.reset(token)
    assert output is not None
    assert observer.raw == seen["raw"]
    assert "\r" not in output
    assert all(not p.exists() for p in seen["paths"])


def test_production_alass_spawn_failure_safe(pair, non_global, monkeypatch):
    from app.services import sync_service
    def fail(*a, **kw):
        raise FileNotFoundError("simulated missing binary")
    monkeypatch.setattr(sync_service.subprocess, "run", fail)
    monkeypatch.setattr(scout.shutil, "which", lambda *a: "test-binary")
    row = scout.evaluate(*pair, sync_service.SubtitleSyncService(alass_path="missing"))
    assert row["status"] == "INELIGIBLE"
    assert row["alass_launched"] is False
    assert row["boundary"] is None


def test_unavailable_executable_reported(pair, non_global, monkeypatch):
    monkeypatch.setattr(scout.shutil, "which", lambda *a: None)
    row = scout.evaluate(*pair, scout.SubtitleSyncService(alass_path="missing"))
    assert row["status"] == "ALASS_UNAVAILABLE"
    assert not row["alass_launched"]
    assert scout.report([row], scout.Counter())["summary"]["errors"] == 1
