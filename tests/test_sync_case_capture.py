"""Diagnostic export must never become a serving decision or leak config."""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.sync import case_capture as cc
from app.services.sync.alignment import SubtitleEvaluation
from app.services.sync.boundary_verifier import BoundaryResult, BoundaryShadow

TARGET = b"1\n00:00:02,000 --> 00:00:03,000\nA dialogue line.\n"
REF = "1\n00:00:04,000 --> 00:00:05,000\nAnother dialogue line.\n"
RAW = "1\n00:00:04,000 --> 00:00:05,000\nA dialogue line.\n"
META = {"imdb_id": "tt0773262", "season": 8, "episode": 2,
        "release_name": "Dexter.S08E02.HDTV-ASAP"}


@pytest.fixture(autouse=True)
def isolated_capture(monkeypatch):
    token = cc.current_capture.set(None)
    monkeypatch.delenv(cc.SELECTION_ENV, raising=False)
    monkeypatch.delenv(cc.ROOT_ENV, raising=False)
    yield
    assert cc.flush_captures()
    asyncio.run(cc.close_captures())
    cc.current_capture.reset(token)


def enable(monkeypatch, tmp_path, *, target=TARGET, case_id="selected-case", meta=None):
    selector = {k: (meta or META)[k] for k in ("imdb_id", "season", "episode")}
    selector.update(case_id=case_id, target_sha256=cc.digest(target))
    monkeypatch.setenv(cc.SELECTION_ENV, json.dumps([selector]))
    monkeypatch.setenv(cc.ROOT_ENV, str(tmp_path / "capture"))


def metadata(capture):
    assert cc.flush_captures()
    return json.loads((capture.directory / "metadata.json").read_text())


def test_off_by_default_does_no_io(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("disabled capture touched the filesystem")
    monkeypatch.setattr(Path, "mkdir", fail)
    cc.begin_capture(TARGET, META, True)
    cc.capture_reference(REF, "release", "edition")
    cc.capture_transformed(REF, RAW, "alass")
    cc.capture_final(TARGET, "srt")
    assert cc.current_capture.get() is None


@pytest.mark.parametrize("change", ["sha", "imdb", "season", "episode", "preference"])
def test_exact_selection_required(monkeypatch, tmp_path, change):
    enable(monkeypatch, tmp_path)
    meta = dict(META)
    target, enabled = TARGET, True
    if change == "sha":
        target += b"\n"
    elif change == "preference":
        enabled = False
    else:
        meta[{"imdb": "imdb_id"}.get(change, change)] = "tt1234567" if change == "imdb" else 9
    cc.begin_capture(target, meta, enabled)
    assert cc.current_capture.get() is None
    assert not (tmp_path / "capture").exists()


@pytest.mark.parametrize("selection", ["{", "{}", "[]", '[{"case_id":"../escape"}]'])
def test_invalid_selection_is_inert(monkeypatch, tmp_path, selection):
    monkeypatch.setenv(cc.SELECTION_ENV, selection)
    monkeypatch.setenv(cc.ROOT_ENV, str(tmp_path / "capture"))
    cc.begin_capture(TARGET, META, True)
    assert cc.current_capture.get() is None
    assert not (tmp_path / "capture").exists()


def test_complete_capture_exact_bytes_hashes_and_no_private_metadata(monkeypatch, tmp_path, caplog):
    enable(monkeypatch, tmp_path)
    private = {**META, "password": "private-password", "stream_url": "https://private.test/token",
               "subdl_key": "private-api-key", "config": {"secret": "never serialize"}}
    cc.begin_capture(TARGET, private, True)
    capture = cc.current_capture.get()
    cc.capture_provider(REF, "subdl")
    cc.capture_reference(REF, "BluRay DEMAND", "edition")
    evaluation = SubtitleEvaluation()
    evaluation.reasons.append("private-password https://private.test/token")
    before = evaluation.model_dump()
    cc.capture_transformed(REF, RAW, "alass", evaluation, RAW)
    cc.capture_boundary(REF, RAW, BoundaryResult(False, "UNSUPPORTED", reasons=("tiny_segment",)))
    cc.capture_output(RAW.encode(), "alass", False)
    cc.capture_response_input(RAW.encode())
    final = RAW.replace("05,000", "04,900").encode()
    cc.capture_final(final, "srt")
    assert cc.flush_captures()
    assert evaluation.model_dump() == before
    assert (capture.directory / "pristine_target.srt").read_bytes() == TARGET
    assert (capture.directory / "selected_reference.srt").read_bytes() == REF.encode()
    assert (capture.directory / "raw_transformed.srt").read_bytes() == RAW.encode()
    assert (capture.directory / "final_served.srt").read_bytes() == final
    record = metadata(capture)
    assert record["final_served_sha256"] == cc.digest(final)
    assert record["final_artifact_verified"] is False
    attempt = record["attempts"][0]
    assert attempt["reference_sha256"] == cc.digest(REF.encode())
    assert attempt["raw_transformed_sha256"] == cc.digest(RAW.encode())
    assert attempt["alignment_input_sha256"] == cc.digest(RAW.encode())
    assert attempt["provider"] == "subdl"
    assert attempt["boundary_shadow"]["reasons"] == ["tiny_segment"]
    safe = (capture.directory / "metadata.json").read_text()
    assert all(secret not in safe for secret in ("private-password", "private-api-key",
                                                "https://", str(tmp_path), "never serialize"))
    assert not caplog.records
    assert not list(capture.directory.rglob(".capture-*"))


def test_disabled_capture_starts_no_writer():
    assert cc._writer is None
    cc.begin_capture(TARGET, META, True)
    assert cc._writer is None
    assert cc.current_capture.get() is None


@pytest.mark.parametrize("private", ["https://provider.test/signed", "token=abc", "Bearer abc",
                                    "C:\\Users\\private", "api\u200b_key=abc"])
def test_private_artifacts_refused_without_rewriting(monkeypatch, tmp_path, private):
    target = TARGET + private.encode()
    enable(monkeypatch, tmp_path, target=target)
    cc.begin_capture(target, META, True)
    assert cc.current_capture.get() is None
    enable(monkeypatch, tmp_path)
    cc.begin_capture(TARGET, META, True)
    capture = cc.current_capture.get()
    cc.capture_transformed(REF, RAW + private, "alass")
    assert metadata(capture)["attempts"] == []
    cc.capture_final(target, "srt")
    assert not (capture.directory / "final_served.srt").exists()


def test_bad_labels_are_omitted(monkeypatch, tmp_path):
    enable(monkeypatch, tmp_path)
    cc.begin_capture(TARGET, {**META, "release_name": "https://signed.test"}, True)
    capture = cc.current_capture.get()
    cc.capture_reference(REF, "C:\\Users\\private", "edition")
    cc.capture_transformed(REF, RAW, "alass")
    record = metadata(capture)
    assert record["target_release_label"] is None
    assert record["attempts"][0]["release_label"] is None


def test_export_write_failure_is_fail_open(monkeypatch, tmp_path):
    enable(monkeypatch, tmp_path)
    cc.begin_capture(TARGET, META, True)
    capture = cc.current_capture.get()
    assert cc.flush_captures()
    original = (capture.directory / "metadata.json").read_bytes()
    def fail(*args):
        raise OSError("private exception https://secret.test")
    monkeypatch.setattr(cc.os, "replace", fail)
    cc.capture_transformed(REF, RAW, "alass")
    cc.capture_final(TARGET, "srt")
    assert cc.flush_captures()
    assert (capture.directory / "metadata.json").read_bytes() == original
    assert not list(capture.directory.rglob(".capture-*"))
    assert cc.current_capture.get() is None


def test_capture_setup_failure_is_fail_open(monkeypatch, tmp_path):
    enable(monkeypatch, tmp_path)
    def fail(*args, **kwargs):
        raise OSError("permission denied")
    monkeypatch.setattr(Path, "mkdir", fail)
    cc.begin_capture(TARGET, META, True)
    capture = cc.current_capture.get()
    assert cc.flush_captures()
    assert capture.failed
    cc.capture_final(TARGET, "srt")
    assert cc.current_capture.get() is None


def test_rejected_attempts_kept_separate_from_actual_original_delivery(monkeypatch, tmp_path):
    enable(monkeypatch, tmp_path)
    cc.begin_capture(TARGET, META, True)
    capture = cc.current_capture.get()
    for ref in (REF, REF.replace("Another", "Different")):
        cc.capture_reference(ref, "BluRay", "edition")
        cc.capture_transformed(ref, RAW, "alass")
    cc.capture_output(TARGET, "original", False)
    cc.capture_final(TARGET, "srt")
    record = metadata(capture)
    assert len(record["attempts"]) == 2
    assert record["output_matching_attempts"] == []
    assert not (capture.directory / "raw_transformed.srt").exists()
    assert (capture.directory / "attempt-01/raw_transformed.srt").read_bytes() == RAW.encode()
    assert record["final_served_sha256"] == cc.digest(TARGET)


async def test_concurrent_requests_and_late_shadow_are_isolated(monkeypatch, tmp_path):
    enable(monkeypatch, tmp_path)
    gate = asyncio.Event()
    async def request():
        cc.begin_capture(TARGET, META, True)
        capture = cc.current_capture.get()
        cc.capture_reference(REF, "BluRay", "edition")
        cc.capture_transformed(REF, RAW, "alass")
        async def late():
            await gate.wait()
            cc.capture_boundary(REF, RAW, BoundaryResult(False, "UNSUPPORTED"))
        shadow = asyncio.create_task(late())
        cc.capture_output(RAW.encode(), "alass", False)
        cc.capture_final(RAW.encode(), "srt")
        return capture, shadow
    records = await asyncio.gather(request(), request())
    assert records[0][0].directory != records[1][0].directory
    gate.set()
    await asyncio.gather(*(job for _, job in records))
    for capture, _ in records:
        record = metadata(capture)
        assert record["delivery"] == "response_constructed"
        assert record["attempts"][0]["boundary_shadow"]["accepted"] is False


async def test_existing_shadow_result_is_exported_without_verdict_mutation(monkeypatch, tmp_path):
    from app.services.sync import boundary_verifier as bv
    enable(monkeypatch, tmp_path)
    cc.begin_capture(TARGET, META, True)
    capture = cc.current_capture.get()
    cc.capture_transformed(REF, RAW, "alass")
    verdict = SubtitleEvaluation(alass_applied=True, alass_successful=True,
                                 cut_verdict="piecewise", rejection_reason="low_confidence")
    before = verdict.model_dump()
    monkeypatch.setattr(bv, "verify_boundary", lambda *a, **kw:
                        BoundaryResult(False, "UNSUPPORTED", reasons=("unsupported_boundary",)))
    owner = BoundaryShadow()
    owner.schedule(TARGET.decode(), REF, RAW, verdict)
    await asyncio.gather(*owner.tasks)
    await owner.close()
    assert verdict.model_dump() == before
    assert metadata(capture)["attempts"][0]["boundary_shadow"]["reasons"] == ["unsupported_boundary"]


@pytest.mark.parametrize("output_format", ["srt", "vtt"])
async def test_response_pipeline_exports_exact_cleaned_body(monkeypatch, tmp_path, output_format):
    from app import main
    enable(monkeypatch, tmp_path)
    async def sync(*args):
        cc.capture_reference(REF, "BluRay", "edition")
        cc.capture_transformed(REF, RAW, "alass")
        cc.capture_output(RAW.encode(), "alass", False)
        return RAW.encode()
    monkeypatch.setattr(main, "_maybe_sync_subtitle", sync)
    payload = await main._sync_subtitle_for_response(TARGET, META, {}, "id", None, True, True)
    capture = cc.current_capture.get()
    monkeypatch.setattr(main, "_run_subtitle_optimization_pipeline",
                        lambda raw, **kw: raw.replace(b"05,000", b"04,900"))
    response = main._build_subtitle_response(payload, "safe release", req_format=output_format)
    assert cc.flush_captures()
    assert (capture.directory / f"final_served.{output_format}").read_bytes() == response.body
    assert metadata(capture)["final_served_sha256"] == cc.digest(response.body)
    assert (capture.directory / "raw_transformed.srt").read_bytes() == RAW.encode()
    assert metadata(capture)["response_input_matching_attempts"] == ["attempt-01"]


async def test_real_affine_output_capture_does_not_change_cache_or_result(monkeypatch, tmp_path):
    from app.services.sync.orchestrator import SyncOrchestrator
    from app.services.sync.query import ResolvedReference
    from app.services.sync_cache import SyncCache
    fixture = Path(__file__).parent / "fixtures/whiplash"
    target = (fixture / "target.srt").read_text(encoding="utf-8-sig")
    reference = (fixture / "reference.srt").read_text(encoding="utf-8-sig")
    # Synthetic release identity isolates capture from the unchanged real timing oracle.
    release = "Whiplash.2014.1080p.BluRay.REMUX-FraMeSToR"
    resolved = ResolvedReference(reference, kind="edition", candidate=release)
    outputs, caches = [], []
    for selected in (False, True):
        if selected:
            enable(monkeypatch, tmp_path, target=target.encode())
            cc.begin_capture(target.encode(), META, True)
            cc.capture_reference(reference, resolved.candidate, resolved.kind)
        cache = SyncCache()
        orch = SyncOrchestrator(sync_cache=cache)
        outputs.append(await orch._try_timeline_affine(target, resolved, {"target_filename": release}, "id", "sha", "rk", None))
        caches.append(await cache.get_meta("rk"))
        await orch.close()
    assert outputs[0] == outputs[1] and outputs[0] is not None
    assert caches[0] == caches[1]
    capture = cc.current_capture.get()
    record = metadata(capture)
    assert record["timeline_affine_attempts"][0]["classification"] == "PAL_LIKE_AFFINE"
    assert record["attempts"][0]["strategy"] == "timeline_affine"
    assert record["attempts"][0]["raw_transformed_sha256"] == cc.digest(outputs[0])


def test_bounded_attempt_count_and_no_background_jobs(monkeypatch, tmp_path):
    assert cc.numeric_evidence({"residual_p95_ms": 58, "holdout_p95_ms": [0, 1]}) == {
        "residual_p95_ms": 58, "holdout_p95_ms": [0, 1]}
    enable(monkeypatch, tmp_path)
    cc.begin_capture(TARGET, META, True)
    capture = cc.current_capture.get()
    for _ in range(cc.MAX_ATTEMPTS + 3):
        cc.capture_transformed(REF, RAW, "alass")
    assert len(metadata(capture)["attempts"]) == cc.MAX_ATTEMPTS


def test_observation_failure_cannot_escape(monkeypatch, tmp_path):
    enable(monkeypatch, tmp_path)
    cc.begin_capture(TARGET, META, True)
    def fail(*a):
        raise RuntimeError("private details")
    monkeypatch.setattr(cc.CaseCapture, "final", fail)
    cc.capture_final(TARGET, "srt")
    assert cc.current_capture.get() is None


def test_rejected_affine_observation_preserves_truth(monkeypatch, tmp_path):
    enable(monkeypatch, tmp_path)
    cc.begin_capture(TARGET, META, True)
    capture = cc.current_capture.get()
    cc.capture_affine(REF, SimpleNamespace(accepted=False, slope=None, intercept_ms=None,
                       diagnostics=SimpleNamespace(classification="DISCONTINUITY_OR_DIFFERENT_CUT")))
    assert metadata(capture)["timeline_affine_attempts"][0]["accepted"] is False


def test_exact_alass_file_and_normalized_verifier_bytes_bound_separately(monkeypatch, tmp_path):
    enable(monkeypatch, tmp_path)
    cc.begin_capture(TARGET, META, True)
    capture = cc.current_capture.get()
    cc.capture_reference(REF, "BluRay", "edition")
    raw = RAW.replace("\n", "\r\n").encode()
    source = tmp_path / "alass.srt"
    source.write_bytes(raw)
    cc.capture_alass_bytes(source.read_bytes(), REF, REF, TARGET.decode())
    cc.capture_transformed(REF, RAW, "alass")
    cc.capture_verifier(REF, RAW, SubtitleEvaluation())
    cc.capture_boundary(REF, RAW, BoundaryResult(False, "UNSUPPORTED"))
    cc.capture_output(RAW.encode(), "alass", False)
    cc.capture_response_input(RAW.encode())
    record = metadata(capture)
    assert len(record["attempts"]) == 1
    attempt = record["attempts"][0]
    assert attempt["producer_observation"] == "alass_output_file"
    assert attempt["raw_transformed_sha256"] == cc.digest(raw)
    assert attempt["verifier_transformed_sha256"] == cc.digest(RAW.encode())
    assert attempt["boundary_shadow"]["accepted"] is False
    assert (capture.directory / "raw_transformed.srt").read_bytes() == raw
    assert (capture.directory / "attempt-01/verifier_transformed.srt").read_bytes() == RAW.encode()


async def test_service_capture_preserves_timing_proposal_and_argv(monkeypatch, tmp_path):
    from app.services import sync_service as service_module
    from app.services.sync_service import SubtitleSyncService
    target = (Path(__file__).parent / "fixtures/large_offset/dexter_s08e04_reference.srt").read_text()
    enable(monkeypatch, tmp_path, target=target.encode())
    cc.begin_capture(target.encode(), META, True)
    capture = cc.current_capture.get()
    raw = target.replace("\n", "\r\n").encode()
    def run(command, **kwargs):
        assert Path(command[1]).read_text() != ""
        assert Path(command[2]).read_text() != ""
        assert command[3].endswith(".synced.srt")
        Path(command[3]).write_bytes(raw)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
    monkeypatch.setattr(service_module.subprocess, "run", run)
    result = await SubtitleSyncService(alass_path="test-alass").sync_async(target, target)
    assert result == target.replace("\r\n", "\n")
    attempt = metadata(capture)["attempts"][0]
    assert attempt["raw_transformed_sha256"] == cc.digest(raw)
    assert attempt["producer_observation"] == "alass_output_file"


async def test_late_background_output_cannot_overwrite_response_evidence(monkeypatch, tmp_path):
    from app import main
    enable(monkeypatch, tmp_path)
    cc.begin_capture(TARGET, META, True)
    capture = cc.current_capture.get()
    gate = asyncio.Event()
    async def late():
        await gate.wait()
        cc.capture_transformed(REF, RAW, "alass")
        cc.capture_output(RAW.encode(), "alass", False)
    job = asyncio.create_task(late())
    response = main._build_subtitle_response(TARGET, "safe release", enable_rtl_fix=False)
    gate.set()
    await job
    record = metadata(capture)
    assert record["orchestrator_completed_after_response"] is True
    assert record["response_input_is_pristine"] is True
    assert record["response_input_matching_attempts"] == []
    assert record["final_served_sha256"] == cc.digest(response.body)
    assert (capture.directory / "final_served.srt").read_bytes() == response.body
    assert not (capture.directory / "selected_reference.srt").exists()
    assert not (capture.directory / "raw_transformed.srt").exists()


async def test_slow_storage_does_not_hold_response_or_sync_budget(monkeypatch, tmp_path):
    from app import main
    enable(monkeypatch, tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = cc.CaseCapture._write_atomic
    def slow(self, *args):
        entered.set()
        release.wait(3)
        original(self, *args)
    monkeypatch.setattr(cc.CaseCapture, "_write_atomic", slow)
    async def sync(*args):
        return TARGET
    monkeypatch.setattr(main, "_maybe_sync_subtitle", sync)
    try:
        payload = await asyncio.wait_for(main._sync_subtitle_for_response(
            TARGET, META, {}, "id", None, True, True), timeout=0.5)
        capture = cc.current_capture.get()
        assert await asyncio.to_thread(entered.wait, 1)
        response = main._build_subtitle_response(payload, "safe", enable_rtl_fix=False)
        assert response.body
        assert not release.is_set()  # Response completed while disk was still blocked.
    finally:
        release.set()
    record = metadata(capture)
    assert record["final_served_sha256"] == cc.digest(response.body)


def test_backpressure_is_bounded_and_fail_open(monkeypatch, tmp_path):
    enable(monkeypatch, tmp_path)
    monkeypatch.setattr(cc, "MAX_PENDING_WRITES", 4)
    entered, release = threading.Event(), threading.Event()
    original = cc.CaseCapture._write_atomic
    def slow(self, *args):
        entered.set()
        release.wait(3)
        original(self, *args)
    monkeypatch.setattr(cc.CaseCapture, "_write_atomic", slow)
    cc.begin_capture(TARGET, META, True)
    capture = cc.current_capture.get()
    try:
        assert entered.wait(1)
        for _ in range(20):
            cc.capture_transformed(REF, RAW, "alass")
        assert capture.failed
        assert cc._writer.pending.qsize() <= 4
        assert len([t for t in threading.enumerate() if t.name == "sync-capture-writer"]) == 1
        cc.capture_final(TARGET, "srt")
        assert cc.current_capture.get() is None
    finally:
        release.set()


def test_selected_reference_is_explicit_for_identical_outputs(monkeypatch, tmp_path):
    enable(monkeypatch, tmp_path)
    cc.begin_capture(TARGET, META, True)
    capture = cc.current_capture.get()
    second = REF.replace("Another", "Different")
    for reference, label in ((REF, "first rejected"), (second, "second selected")):
        cc.capture_reference(reference, label, "edition")
        cc.capture_transformed(reference, RAW, "alass")
    cc.capture_selected(second, RAW, "alass")
    cc.capture_output(RAW.encode(), "alass", False)
    cc.capture_response_input(RAW.encode())
    cc.capture_final(RAW.encode(), "srt")
    record = metadata(capture)
    assert record["response_selected_attempt"] == "attempt-02"
    assert (capture.directory / "selected_reference.srt").read_bytes() == second.encode()
    for name, sha in record["artifact_sha256"].items():
        assert cc.digest((capture.directory / name).read_bytes()) == sha


def test_provider_and_reference_metadata_have_hard_bounds(monkeypatch, tmp_path):
    enable(monkeypatch, tmp_path)
    cc.begin_capture(TARGET, META, True)
    capture = cc.current_capture.get()
    for i in range(100):
        ref = REF + str(i)
        cc.capture_provider(ref, "subdl")
        cc.capture_reference(ref, "edition", "edition")
    assert len(capture.providers) == cc.MAX_ATTEMPTS
    assert len(capture.references) == cc.MAX_ATTEMPTS


def test_path_traversal_and_unsafe_final_format_refused(monkeypatch, tmp_path):
    enable(monkeypatch, tmp_path)
    cc.begin_capture(TARGET, META, True)
    capture = cc.current_capture.get()
    cc.capture_final(TARGET, "../../escaped")
    capture.write("../../escaped", TARGET)
    assert cc.flush_captures()
    assert capture.failed
    assert not (capture.directory.parent.parent / "escaped").exists()


async def test_normal_shutdown_drains_writer_and_stops_admission(monkeypatch, tmp_path):
    enable(monkeypatch, tmp_path)
    cc.begin_capture(TARGET, META, True)
    capture = cc.current_capture.get()
    cc.capture_response_input(TARGET)
    cc.capture_final(TARGET, "srt")
    await cc.close_captures()
    assert cc._writer is None
    record = metadata(capture)
    assert (capture.directory / "final_served.srt").read_bytes() == TARGET
    assert record["final_served_sha256"] == cc.digest(TARGET)
    assert not list(capture.directory.rglob(".capture-*"))
    capture.write("late.srt", TARGET)
    assert capture.failed
    assert cc._writer is None
    assert not (capture.directory / "late.srt").exists()


@pytest.mark.parametrize("newline", ["\r\n", "\r", "\n"])
async def test_service_return_is_identical_enabled_and_disabled(monkeypatch, tmp_path, newline):
    from app.services import sync_service as sm
    fixture = Path(__file__).parent / "fixtures/large_offset/dexter_s08e04_reference.srt"
    text = fixture.read_text()
    raw = text.replace("\n", newline).encode()
    calls = []
    def run(command, **kw):
        calls.append(command[4:])
        Path(command[3]).write_bytes(raw)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
    monkeypatch.setattr(sm.subprocess, "run", run)
    service = sm.SubtitleSyncService(alass_path="test-alass")
    before = await service.sync_async(text, text)
    enable(monkeypatch, tmp_path, target=text.encode())
    cc.begin_capture(text.encode(), META, True)
    capture = cc.current_capture.get()
    after = await service.sync_async(text, text)
    assert before == after and before is not None
    assert calls[0] == calls[1]
    record = metadata(capture)
    assert record["attempts"][0]["raw_transformed_sha256"] == cc.digest(raw)
