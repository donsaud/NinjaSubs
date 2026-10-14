"""Isolated temporary test corpus; no provider requests or real evidence generated."""
import asyncio
import io
import json
import tarfile
import zipfile
from types import SimpleNamespace

import pytest

from app.services.sync.ninja_points import render_srt
from tools import boundary_scout as scout
from tools import boundary_scout_intake as intake


@pytest.fixture
def real_shape():
    raw = render_srt([(10000 + i * 5000, 11000 + i * 5000, "Test speech.") for i in range(50)]).encode()
    rel = SimpleNamespace(provider="subdl", release_name="Homeland.S02E03.HDTV.x264-GROUP.srt",
                          download_url="https://dl.subdl.com/subtitles/123-456.zip", lang="ara")
    return raw, rel, ("tt1796960", 2, 3)


def test_duplicate_sha_deduplicated(tmp_path, real_shape):
    raw, rel, identity = real_shape
    corpus = intake.Corpus(tmp_path / "output")
    assert corpus.store(raw, rel, identity, "target")
    assert not corpus.store(raw, rel, identity, "target")
    assert len(corpus.entries) == 1
    assert corpus.stats["duplicate_sha"] == 1


def test_source_bytes_immutable_and_no_secret_metadata(tmp_path, real_shape):
    raw, rel, identity = real_shape
    raw = raw.replace(b"\n", b"\r\n")
    corpus = intake.Corpus(tmp_path / "output")
    corpus.store(raw, rel, identity, "target")
    entry = corpus.entries[0]
    assert (corpus.root / entry["path"]).read_bytes() == raw
    assert entry["sha256"] == scout.digest(raw)
    before = (corpus.root / entry["path"]).read_bytes()
    intake.scout.discover(tmp_path / "empty", [], [corpus.root / "intake-manifest.json"])
    assert (corpus.root / entry["path"]).read_bytes() == before
    manifest = (corpus.root / "intake-manifest.json").read_text()
    assert "https:" not in manifest
    assert "download_url" not in manifest
    assert entry["source_metadata"]["provider_listing_id"] == "123-456"


def test_intake_cannot_write_production_or_evidence_paths(tmp_path):
    for path in (scout.ROOT / "outputs", scout.ROOT / "subs_cache", scout.ROOT / "tests/replay",
                 tmp_path / "cache", tmp_path / "NinjaSubs-Captures"):
        with pytest.raises(ValueError):
            intake.Corpus(path)


def test_cross_episode_prohibited(tmp_path, real_shape):
    raw, rel, identity = real_shape
    corpus = intake.Corpus(tmp_path / "output")
    with pytest.raises(ValueError):
        corpus.store(raw, rel, (identity[0], 2, 4), "target")
    assert not corpus.entries


def test_failed_provider_download_leaves_no_partial_entry(tmp_path, real_shape, monkeypatch):
    raw, rel, identity = real_shape
    reference = SimpleNamespace(provider="subdl", release_name="Homeland.S02E03.BluRay.x264-GROUP.srt",
                                download_url="https://dl.subdl.com/subtitles/789-987.zip", lang="eng")
    class Provider:
        async def search_subtitles(self, **kw):
            return [rel, reference]
        async def download_archive(self, ref):
            return None
    async def no_wait(*a):
        pass
    monkeypatch.setattr(intake.asyncio, "sleep", no_wait)
    corpus = intake.Corpus(tmp_path / "output")
    asyncio.run(intake.acquire(corpus, [Provider()], [identity]))
    assert not corpus.entries
    assert corpus.stats["provider_download_or_payload_errors"] == 2
    assert not list(corpus.root.rglob("source.srt"))
    assert json.loads((corpus.root / "intake-manifest.json").read_text())["payloads"] == []


def test_zip_member_bytes_retained(tmp_path, real_shape):
    raw, rel, identity = real_shape
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(rel.release_name, raw)
    corpus = intake.Corpus(tmp_path / "output")
    corpus.store(stream.getvalue(), rel, identity, "target")
    entry = corpus.entries[0]
    source = corpus.root / entry["path"]
    assert source.read_bytes() == raw
    assert (source.parent / "download.bin").read_bytes() == stream.getvalue()


def test_docker_unavailable_fails_before_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(intake.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1))
    with pytest.raises(intake.IntakeUnavailable, match="Docker access"):
        intake.docker_run("ninjasubs", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_alass_version_failure_is_explicit(monkeypatch):
    monkeypatch.setattr(intake.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1))
    with pytest.raises(intake.IntakeUnavailable, match="version check failed"):
        intake.alass_preflight("production-alass")


def test_derived_results_and_no_evidence_manifest_mutation(tmp_path, real_shape):
    raw, rel, identity = real_shape
    corpus = intake.Corpus(tmp_path / "output")
    evidence = tmp_path / "formal-evidence.json"
    evidence.write_bytes(b"formal evidence sentinel")
    corpus.store(raw, rel, identity, "target")
    assert scout.report([], scout.Counter())["evidence_class"] == "DERIVED_SCOUTING_ONLY"
    assert scout.report([], scout.Counter())["requires_live_stremio_confirmation"] is True
    assert evidence.read_bytes() == b"formal evidence sentinel"


def test_docker_snapshot_isolated_and_has_no_env(tmp_path, monkeypatch):
    calls = []
    def run(argv, **kw):
        calls.append(argv)
        if "input" in kw:
            with tarfile.open(fileobj=io.BytesIO(kw["input"])) as archive:
                names = archive.getnames()
                assert all(name.startswith(("app/", "tools/")) for name in names)
                assert not any(".env" in name or "cache" == name.split("/")[0] for name in names)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(intake.subprocess, "run", run)
    intake.docker_run("ninjasubs", tmp_path / "output")
    execution = next(c for c in calls if "--in-container" in c)
    workdir = execution[execution.index("-w") + 1]
    out = execution[execution.index("--output-dir") + 1]
    assert workdir.startswith("/tmp/ninjasubs-boundary-scout-")
    assert workdir.endswith("/snapshot")
    assert not out.startswith(workdir)
    assert not any("/app/cache" in arg for c in calls for arg in c)


def test_seed_corpus_is_broad_and_excludes_exhausted_and_known_global():
    assert len(set(intake.SEEDS)) >= 20
    assert ("tt1796960", 2, 2) not in intake.SEEDS
    assert not any(i[0] == "tt0758758" or i[0] == "tt0773262" and i[1] == 8 for i in intake.SEEDS)


def test_conflicting_media_sha_removed_from_usable_manifest(tmp_path, real_shape):
    raw, rel, identity = real_shape
    corpus = intake.Corpus(tmp_path / "output")
    corpus.store(raw, rel, identity, "target")
    original = corpus.root / corpus.entries[0]["path"]
    with pytest.raises(ValueError, match="conflicting_media"):
        corpus.store(raw, rel, ("tt0475784", 2, 3), "target")
    assert not corpus.entries
    assert original.read_bytes() == raw
    assert not json.loads((corpus.root / "intake-manifest.json").read_text())["payloads"]


def test_nonempty_existing_corpus_never_overwritten(tmp_path, real_shape):
    raw, rel, identity = real_shape
    corpus = intake.Corpus(tmp_path / "output")
    corpus.store(raw, rel, identity, "target")
    before = (corpus.root / "intake-manifest.json").read_bytes()
    with pytest.raises(ValueError, match="new isolated output directory"):
        intake.Corpus(corpus.root)
    assert (corpus.root / "intake-manifest.json").read_bytes() == before


def test_rank_prefers_coverage_on_both_sides():
    segment = {"support": 100, "forward_coverage": .99, "reverse_coverage": .95,
               "residual_p95_ms": 50, "reference_start_p95_ms": 100}
    row = {"status": "EDGE_CANDIDATE", "alass_launched": True,
           "legacy": {"sync_state": "unverified"},
           "boundary": {"holdout_pass": True, "segments": [segment, segment]}}
    weaker = {**row, "boundary": {**row["boundary"], "segments": [
        {**segment, "forward_coverage": 1, "reverse_coverage": .70}, segment]}}
    assert scout.rank(row) > scout.rank(weaker)
