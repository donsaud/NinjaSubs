"""Explicitly selected, fail-open diagnostic artifacts; never production trust."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import queue
import re
import tempfile
import threading
import unicodedata
import uuid
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SELECTION_ENV = "NINJASUBS_CASE_CAPTURE_SELECTION"
ROOT_ENV = "NINJASUBS_CASE_CAPTURE_ROOT"
MAX_BYTES = 2_000_000
MAX_ATTEMPTS = 16
MAX_PENDING_WRITES = 64
_UNSAFE = re.compile(
    r"(?i)(?:[a-z][a-z0-9+.-]*://|\bwww\.|[a-z]:[/\\]|\\\\|"
    r"/(?:home|users|tmp)/|\b(?:api[_-]?key|(?:access[_-]?|refresh[_-]?)?token|"
    r"password|secret|authorization)[\"']?\s*[:=]|\bBearer\s+\S+|"
    r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+|"
    r"\bAKIA[A-Z0-9]{16}\b|\bsk-[A-Za-z0-9_-]{16,})"
)
current_capture: ContextVar[CaseCapture | None] = ContextVar("sync_case_capture", default=None)
_writer_lock = threading.Lock()
_writer: CaptureWriter | None = None


@dataclass(frozen=True)
class ProjectionJob:
    target: str
    reference: str
    transformed: str


class CaptureWriter:
    """One bounded process-wide writer, created only for selected captures."""
    def __init__(self) -> None:
        self.pending: queue.Queue[tuple[CaseCapture, str, bytes | tuple[str, str] | ProjectionJob]] = queue.Queue(
            MAX_PENDING_WRITES)
        self.closing = threading.Event()
        self.thread = threading.Thread(target=self._run, name="sync-capture-writer", daemon=True)
        self.thread.start()

    def submit(self, owner: CaseCapture, name: str, data: bytes | tuple[str, str] | ProjectionJob) -> None:
        if self.closing.is_set() or owner.failed:
            owner.failed = True
            return
        try:
            self.pending.put_nowait((owner, name, data))
        except queue.Full:
            owner.failed = True

    def _run(self) -> None:
        while not self.closing.is_set() or not self.pending.empty():
            try:
                owner, name, data = self.pending.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                if not owner.failed:
                    if isinstance(data, ProjectionJob):
                        owner._write_projection(name, data)
                        continue
                    if isinstance(data, tuple):
                        source, expected_sha = data
                        with (owner.directory / source).open("rb") as stream:
                            data = stream.read(MAX_BYTES + 1)
                        if len(data) > MAX_BYTES or digest(data) != expected_sha:
                            raise ValueError("capture source digest mismatch")
                    if name == "metadata.json":
                        with owner.lock:
                            data = (json.dumps(owner.metadata, indent=2, allow_nan=False)
                                    + "\n").encode()
                    owner._write_atomic(name, data)
            except Exception:
                owner.failed = True  # Fixed status only; never log exception details.
            finally:
                self.pending.task_done()

    def drain(self, timeout: float = 5) -> bool:
        # Condition wait is bounded; no queue.join() that can hold shutdown forever.
        import time
        deadline = time.monotonic() + timeout
        with self.pending.all_tasks_done:
            while self.pending.unfinished_tasks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.pending.all_tasks_done.wait(remaining)
        return True


def flush_captures(timeout: float = 5) -> bool:
    """Local diagnostic/test barrier, never called from subtitle delivery."""
    return _writer is None or _writer.drain(timeout)


async def close_captures() -> None:
    global _writer
    writer = _writer
    if writer is None:
        return
    writer.closing.set()
    await asyncio.to_thread(writer.thread.join, 5)
    if not writer.thread.is_alive():
        with _writer_lock:
            if _writer is writer:
                _writer = None


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def safe_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = unicodedata.normalize("NFKC", value)
    value = "".join(c for c in value if unicodedata.category(c) != "Cf")
    if (not value or len(value) > 160 or _UNSAFE.search(value)
            or re.search(r"[A-Za-z0-9_-]{40,}", value)
            or not re.fullmatch(r"[\w .+()\[\]-]+", value)):
        return None
    return value


def safe_payload(raw: bytes) -> bool:
    if not raw or len(raw) > MAX_BYTES:
        return False
    text = raw.decode("utf-8-sig")
    inspected = "".join(c for c in unicodedata.normalize("NFKC", text)
                        if unicodedata.category(c) != "Cf")
    return "\x00" not in inspected and not _UNSAFE.search(inspected)


def numeric_evidence(value: Any) -> Any:
    """Boundary dataclass fields only: numbers, booleans and fixed code strings."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value if re.fullmatch(r"[A-Za-z_]{1,80}", value) else None
    if isinstance(value, (tuple, list)):
        return [numeric_evidence(v) for v in value]
    if isinstance(value, dict):
        return {k: numeric_evidence(v) for k, v in value.items()
                if re.fullmatch(r"[a-z_][a-z0-9_]{0,79}", k)}
    return None


def evaluation_evidence(evaluation: Any) -> dict[str, Any]:
    # Never copy model_dump(), free-text reasons, or request/reference objects.
    fields = ("sync_confidence", "verification_confidence", "median_offset_ms",
              "mad_offset_ms", "p95_offset_ms", "drift_ms_per_minute",
              "coverage_score", "structural_similarity", "alass_applied", "alass_successful")
    result = {k: numeric_evidence(getattr(evaluation, k, None)) for k in fields}
    for k in ("sync_state", "verification", "cut_verdict", "rejection_reason"):
        value = getattr(evaluation, k, None)
        result[k] = numeric_evidence(getattr(value, "value", value))
    return result


class CaseCapture:
    def __init__(self, directory: Path, identity: dict[str, Any], raw: bytes, label: Any, language: Any = None):
        self.directory = directory
        self.failed = False
        self.writer: CaptureWriter | None = None
        self.lock = threading.RLock()
        self.references: dict[str, dict[str, Any]] = {}
        self.providers: dict[str, str | None] = {}
        self.metadata: dict[str, Any] = {
            "schema_version": 2, "provenance": "live_capture_unreviewed",
            "timestamp": datetime.now(UTC).isoformat(), **identity,
            "target_sha256": digest(raw), "target_release_label": safe_text(label),
            "parser_language": language if isinstance(language, str) and re.fullmatch(
                r"[a-z]{2,3}", language) else None,
            "target_representation": "sync_ingress_before_intro_stripping_or_preshift",
            "attempts": [], "final_served_sha256": None,
            "boundary_shadow": "not_observed", "delivery": "not_observed",
        }
        self.write("pristine_target.srt", raw)
        self.save()

    def write(self, name: str, raw: bytes) -> None:
        global _writer
        if len(raw) > MAX_BYTES:
            self.failed = True
            return
        with _writer_lock:
            if self.writer is None:
                if _writer is None:
                    _writer = CaptureWriter()
                self.writer = _writer
            self.writer.submit(self, name, raw)
        if name != "metadata.json":
            self.metadata.setdefault("artifact_sha256", {})[name] = digest(raw)

    def _write_atomic(self, name: str, raw: bytes) -> None:
        relative = Path(name)
        if len(raw) > MAX_BYTES or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe capture filename")
        destination = self.directory / name
        parent = destination.parent
        if any(p.is_symlink() for p in (destination, *destination.parents)):
            raise ValueError("unsafe capture destination")
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if parent.is_symlink() or destination.is_symlink():
            raise ValueError("unsafe capture destination")
        fd, temporary = tempfile.mkstemp(prefix=".capture-", dir=parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def save(self) -> None:
        self.write("metadata.json", (json.dumps(self.metadata, indent=2, allow_nan=False)
                                     + "\n").encode())

    def reference(self, reference: str, label: Any, kind: Any) -> None:
        raw = reference.encode("utf-8")
        sha = digest(raw)
        if (not safe_payload(raw)
                or (sha not in self.references and len(self.references) >= MAX_ATTEMPTS)):
            return
        self.references[sha] = {"release_label": safe_text(label), "provenance": safe_text(kind),
                                "provider": self.providers.get(sha)}

    def transformed(self, reference: str, transformed: str, strategy: str,
                    evaluation: Any = None, alignment_input: str | None = None,
                    raw_bytes: bytes | None = None) -> None:
        ref = reference.encode("utf-8")
        verifier_raw = transformed.encode("utf-8")
        raw = raw_bytes if raw_bytes is not None else verifier_raw
        if raw_bytes is None and strategy == "alass":
            for attempt in reversed(self.metadata["attempts"]):
                if (attempt["reference_sha256"] == digest(ref)
                        and attempt["verifier_transformed_sha256"] == digest(verifier_raw)
                        and attempt["producer_observation"] == "alass_output_file"):
                    if evaluation is not None:
                        attempt["verifier"] = evaluation_evidence(evaluation)
                        self.save()
                    return
        if (len(self.metadata["attempts"]) >= MAX_ATTEMPTS
                or not safe_payload(ref) or not safe_payload(raw)):
            return
        index = len(self.metadata["attempts"]) + 1
        folder = f"attempt-{index:02d}"
        attempt = {"directory": folder, "strategy": safe_text(strategy),
                   "reference_sha256": digest(ref), "raw_transformed_sha256": digest(raw),
                   "verifier_transformed_sha256": digest(verifier_raw),
                   "producer_observation": ("alass_output_file" if raw_bytes is not None
                                            else "orchestrator_transformed_text"),
                   **self.references.get(digest(ref), {"provenance": None, "provider": None,
                                                     "release_label": None}),
                   "verifier": evaluation_evidence(evaluation),
                   "boundary_shadow": "not_observed",
                   "boundary_shadow_status": "not_observed"}
        self.write(f"{folder}/selected_reference.srt", ref)
        self.write(f"{folder}/raw_transformed.srt", raw)
        if raw != verifier_raw:
            self.write(f"{folder}/verifier_transformed.srt", verifier_raw)
        if alignment_input is not None:
            source = alignment_input.encode("utf-8")
            if digest(source) != self.metadata["target_sha256"] and safe_payload(source):
                self.write(f"{folder}/alignment_input.srt", source)
                attempt["alignment_input_sha256"] = digest(source)
        self.metadata["attempts"].append(attempt)
        self.save()

    def alass_bytes(self, raw: bytes, selected_reference: str, reference_input: str,
                   alignment_input: str) -> None:
        if len(raw) > MAX_BYTES:
            return
        normalized = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        count = len(self.metadata["attempts"])
        self.transformed(selected_reference, normalized, "alass",
                         alignment_input=alignment_input, raw_bytes=raw)
        if len(self.metadata["attempts"]) > count:
            attempt = self.metadata["attempts"][-1]
            ref = reference_input.encode()
            if safe_payload(ref):
                self.write(f"{attempt['directory']}/alass_reference_input.srt", ref)
                attempt["alass_reference_input_sha256"] = digest(ref)
            self.save()

    def _write_projection(self, folder: str, job: ProjectionJob) -> None:
        # Runs on the bounded disk writer, never on the request/shadow event loop.
        from app.extractor import decode_subtitle_bytes
        from app.services.sync.parser_fingerprint import FINGERPRINT_SCHEMA
        from app.services.sync.parser_projection import canonical_bytes, implementation_sha, project
        original = (self.directory / "pristine_target.srt").read_bytes()
        if len(original) > MAX_BYTES or digest(original) != self.metadata["target_sha256"]:
            raise ValueError("capture original changed")
        decoded = decode_subtitle_bytes(original, lang=self.metadata["parser_language"])
        if decoded != job.target:
            raise ValueError("shadow target differs from ingress decoder")
        texts = {"target": job.target, "reference": job.reference, "transformed": job.transformed}
        manifest = {"version": 2, "parser_fingerprint_schema": FINGERPRINT_SCHEMA,
                    "parser_implementation_sha256": implementation_sha(),
                    "original_target_sha256": digest(original),
                    "projections": {role: project(text) for role, text in texts.items()},
                    "input_sha256": {role: digest(text.encode()) for role, text in texts.items()}}
        raw = canonical_bytes(manifest)
        name = f"{folder}/boundary_parser_projection.json"
        # Inputs are exact verifier representations, distinct from original/raw bytes.
        artifacts = {f"{folder}/boundary_{role}_input.srt": text.encode()
                     for role, text in texts.items()}
        if not all(safe_payload(value) for value in artifacts.values()):
            raise ValueError("capture projection privacy refusal")
        artifacts[name] = raw
        for path, value in artifacts.items():
            self._write_atomic(path, value)
        with self.lock:
            self.metadata.setdefault("artifact_sha256", {}).update(
                {path: digest(value) for path, value in artifacts.items()})
            for attempt in self.metadata["attempts"]:
                if attempt["directory"] == folder:
                    attempt["parser_projection"] = name
                    attempt["parser_projection_sha256"] = digest(raw)
            metadata_raw = (json.dumps(self.metadata, indent=2, allow_nan=False) + "\n").encode()
        self._write_atomic("metadata.json", metadata_raw)

    def boundary(self, reference: str, transformed: str, result: Any,
                 target: str | None = None) -> None:
        for attempt in reversed(self.metadata["attempts"]):
            if (attempt["reference_sha256"] == digest(reference.encode())
                    and attempt["verifier_transformed_sha256"] == digest(transformed.encode())):
                attempt["boundary_shadow"] = numeric_evidence(asdict(result))
                attempt["boundary_shadow_status"] = "completed"
                self.metadata["boundary_shadow"] = "observed"
                if target is not None and self.writer is not None:
                    self.writer.submit(self, attempt["directory"],
                                       ProjectionJob(target, reference, transformed))
                self.save()
                break

    def shadow_status(self, reference: str, transformed: str, status: str) -> None:
        for attempt in reversed(self.metadata["attempts"]):
            if (attempt["reference_sha256"] == digest(reference.encode())
                    and attempt["verifier_transformed_sha256"] == digest(transformed.encode())):
                attempt["boundary_shadow_status"] = numeric_evidence(status)
                self.save()
                break

    def verifier(self, reference: str, transformed: str, evaluation: Any) -> None:
        for attempt in reversed(self.metadata["attempts"]):
            if (attempt["reference_sha256"] == digest(reference.encode())
                    and attempt["verifier_transformed_sha256"] == digest(transformed.encode())):
                attempt["verifier"] = evaluation_evidence(evaluation)
                self.save()
                break

    def affine(self, reference: str, result: Any) -> None:
        attempts = self.metadata.setdefault("timeline_affine_attempts", [])
        if len(attempts) >= MAX_ATTEMPTS:
            return
        attempts.append({"reference_sha256": digest(reference.encode()),
                         "accepted": bool(result.accepted),
                         "classification": numeric_evidence(result.diagnostics.classification),
                         "slope": numeric_evidence(result.slope),
                         "intercept_ms": numeric_evidence(result.intercept_ms)})
        self.save()

    def output(self, raw: bytes, strategy: Any, cache_hit: bool) -> None:
        self.metadata["orchestrator_output_sha256"] = digest(raw)
        self.metadata["orchestrator_completed_after_response"] = (
            self.metadata["final_served_sha256"] is not None)
        self.metadata["winning_strategy"] = safe_text(strategy)
        self.metadata["verified_cache_hit"] = cache_hit
        matches = [a["directory"] for a in self.metadata["attempts"]
                   if a["verifier_transformed_sha256"] == digest(raw)]
        self.metadata["output_matching_attempts"] = matches
        self.save()

    def selected(self, reference: str, transformed: str, strategy: str) -> None:
        matches = [a for a in self.metadata["attempts"]
                   if a["reference_sha256"] == digest(reference.encode())
                   and a["verifier_transformed_sha256"] == digest(transformed.encode())
                   and a["strategy"] == strategy]
        if matches:
            self.metadata["selected_attempt"] = matches[-1]["directory"]

    def response_input(self, raw: bytes) -> None:
        self.metadata["response_input_sha256"] = digest(raw)
        self.metadata["response_input_is_pristine"] = digest(raw) == self.metadata["target_sha256"]
        self.metadata["response_input_matching_attempts"] = [
            a["directory"] for a in self.metadata["attempts"]
            if a["verifier_transformed_sha256"] == digest(raw)]
        matches = self.metadata["response_input_matching_attempts"]
        selected = self.metadata.get("selected_attempt")
        if selected not in matches:
            selected = matches[0] if len(matches) == 1 else None
        self.metadata["response_selected_attempt"] = selected
        if selected is not None:
            attempt = next(a for a in self.metadata["attempts"] if a["directory"] == selected)
            with _writer_lock:
                if self.writer is not None:
                    for name, sha in (("selected_reference.srt", attempt["reference_sha256"]),
                                      ("raw_transformed.srt", attempt["raw_transformed_sha256"])):
                        self.writer.submit(self, name, (f"{selected}/{name}", sha))
                        self.metadata.setdefault("artifact_sha256", {})[name] = sha

    def final(self, raw: bytes, output_format: str) -> None:
        if output_format not in {"srt", "vtt", "ass", "ssa"}:
            return
        if not safe_payload(raw):
            self.metadata["delivery"] = "export_refused_privacy_or_size"
            self.save()
            return
        # Preserve actual body; never rename VTT/ASS bytes as an SRT artifact.
        name = "final_served.srt" if output_format == "srt" else f"final_served.{output_format}"
        self.write(name, raw)
        self.metadata["final_served_sha256"] = digest(raw)
        self.metadata["final_format"] = output_format
        self.metadata["delivery"] = "response_constructed"
        self.metadata["final_artifact_verified"] = False
        self.save()


def _observe(method: str, *args: Any) -> None:
    capture = current_capture.get()
    if capture is None or capture.failed:
        return
    try:
        with capture.lock:
            getattr(capture, method)(*args)
    except Exception:
        # No exception details or subtitle bodies in logs. No delivery coupling.
        pass


def begin_capture(raw: bytes, meta: dict[str, Any], enabled: bool) -> None:
    current_capture.set(None)
    try:
        selection = os.environ.get(SELECTION_ENV, "")
        if not enabled or not selection or len(selection) > 16_000:
            return
        selectors = json.loads(selection)
        if not isinstance(selectors, list) or len(selectors) > 32:
            return
        identity = {k: meta.get(k) for k in ("imdb_id", "season", "episode")}
        if not re.fullmatch(r"tt\d{5,12}", str(identity["imdb_id"])):
            return
        if any(v is not None and (type(v) is not int or not 1 <= v <= 999)
               for v in (identity["season"], identity["episode"])):
            return
        sha = digest(raw)
        for s in selectors:
            if (not isinstance(s, dict) or set(s) != {*identity, "case_id", "target_sha256"}
                    or any(s.get(k) != v for k, v in identity.items())
                    or s.get("target_sha256") != sha
                    or not isinstance(s.get("case_id"), str)
                    or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", s["case_id"])
                    or len(s["case_id"]) > 80 or not safe_payload(raw)):
                continue
            root = Path(os.environ.get(ROOT_ENV, "/tmp/ninjasubs_case_capture"))
            case = root / s["case_id"]
            if not root.is_absolute():
                return
            directory = case / uuid.uuid4().hex
            current_capture.set(CaseCapture(directory, identity, raw,
                                           meta.get("release_name") or meta.get("display_release_name"),
                                           meta.get("lang")))
            return
    except Exception:
        pass


def capture_reference(reference: str, label: Any, kind: Any) -> None:
    _observe("reference", reference, label, kind)


def capture_provider(reference: str, provider: Any) -> None:
    capture = current_capture.get()
    if capture is not None:
        try:
            with capture.lock:
                sha = digest(reference.encode())
                if sha not in capture.providers and len(capture.providers) >= MAX_ATTEMPTS:
                    return
                capture.providers[sha] = safe_text(provider)
        except Exception:
            pass


def capture_transformed(reference: str, transformed: str, strategy: str,
                        evaluation: Any = None, alignment_input: str | None = None) -> None:
    _observe("transformed", reference, transformed, strategy, evaluation, alignment_input)


def capture_alass_bytes(raw: bytes, selected_reference: str, reference_input: str,
                       alignment_input: str) -> None:
    _observe("alass_bytes", raw, selected_reference, reference_input, alignment_input)


def capture_enabled() -> bool:
    capture = current_capture.get()
    return capture is not None and not capture.failed


def capture_selected(reference: str, transformed: str, strategy: str) -> None:
    _observe("selected", reference, transformed, strategy)


def capture_verifier(reference: str, transformed: str, evaluation: Any) -> None:
    _observe("verifier", reference, transformed, evaluation)


def capture_boundary(reference: str, transformed: str, result: Any,
                     target: str | None = None) -> None:
    _observe("boundary", reference, transformed, result, target)


def capture_shadow_status(reference: str, transformed: str, status: str) -> None:
    _observe("shadow_status", reference, transformed, status)


def capture_affine(reference: str, result: Any) -> None:
    _observe("affine", reference, result)


def capture_output(raw: bytes, strategy: Any, cache_hit: bool) -> None:
    _observe("output", raw, strategy, cache_hit)


def capture_final(raw: bytes, output_format: str) -> None:
    try:
        _observe("final", raw, output_format)
    finally:
        current_capture.set(None)


def capture_response_input(raw: bytes) -> None:
    _observe("response_input", raw)
