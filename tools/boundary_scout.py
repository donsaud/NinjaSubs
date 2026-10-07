"""Offline real-payload scouting; never live evidence, providers or production caches."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
import tempfile
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.extractor import decode_subtitle_bytes  # noqa: E402
from app.services.subtitle_matcher import (  # noqa: E402
    FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
    measure_alignment_consistency,
    parse_srt_cues,
    validate_cue_sanity,
)
from app.services.sync.alignment import MAX_PLAUSIBLE_OFFSET_MS, AlignmentAnalyzer  # noqa: E402
from app.services.sync.anchor_preshift import decide_anchor_preshift, preshift_text  # noqa: E402
from app.services.sync.boundary_verifier import verify_boundary  # noqa: E402
from app.services.sync.case_capture import CaseCapture, current_capture  # noqa: E402
from app.services.sync.large_offset import (  # noqa: E402
    LARGE_OFFSET_MAX_MS,
    assess_large_offset,
    classify_large_offset_candidate,
)
from app.services.sync.large_offset_investigation import investigate_large_offset  # noqa: E402
from app.services.sync.matching import _source_kind  # noqa: E402
from app.services.sync.parser_projection import project  # noqa: E402
from app.services.sync.reference import is_dialogue_cue  # noqa: E402
from app.services.sync.telemetry import SyncTiming, current_sync_timing  # noqa: E402
from app.services.sync.timeline_affine import attempt_timeline_affine  # noqa: E402
from app.services.sync_service import SubtitleSyncService  # noqa: E402
from app.utils.cleaners import strip_intro_credits  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_CLASS = "DERIVED_SCOUTING_ONLY"
MAX_BYTES = 2_000_000


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def label(value: Any) -> str:
    """Reject unsafe metadata; omit embedded episode titles from release labels."""
    s = str(value or "")
    if len(s) > 240 or re.search(r"[:/\\@?=&]|(?i:token|password|api.?key|jwt)", s):
        return "withheld"
    s = re.sub(r"(?i)(S\d{1,2}E\d{1,3})[. _-]+.*?(?=(?:\d{3,4}p|HDTV|WEB|BluRay|BDRip|REMUX)\b)", r"\1.", s)
    return s.replace("\n", " ").replace("\r", " ").replace("|", " ")


@dataclass(frozen=True)
class Payload:
    path: Path
    imdb: str
    season: int | None
    episode: int | None
    provider: str
    release: str
    subtitle_id: str
    sha256: str
    language: str | None = None

    @property
    def identity(self) -> tuple[str, int | None, int | None]:
        return self.imdb, self.season, self.episode

    @property
    def family(self) -> str | None:
        return _source_kind(self.release)

    def record(self) -> dict[str, Any]:
        return {"provider": self.provider, "release": label(self.release),
                "id": label(self.subtitle_id), "sha256": self.sha256}


def payload(path: Path, meta: dict[str, Any], *, expected: str | None = None) -> Payload:
    """Metadata must identify a single movie/episode; filenames may contradict it."""
    imdb = meta.get("imdb_id")
    if not isinstance(imdb, str) or not re.fullmatch(r"tt\d{7,10}", imdb):
        raise ValueError("ambiguous_imdb")
    season, episode = meta.get("season"), meta.get("episode")
    series = meta.get("media_type") in {"series", "tv", "episode"} or season is not None or episode is not None
    if not series and meta.get("media_type") != "movie":
        raise ValueError("ambiguous_media_type")
    if series and (type(season) is not int or type(episode) is not int or season < 1 or episode < 1):
        raise ValueError("ambiguous_episode")
    if imdb == "tt0758758" or (imdb == "tt0773262" and season == 8):
        raise ValueError("exhausted_media")
    release = str(meta.get("release_name") or "")
    matches = re.findall(r"(?i)S(\d{1,2})E(\d{1,3})", release)
    if series and (not matches or any((int(s), int(e)) != (season, episode) for s, e in matches)):
        raise ValueError("release_episode_ambiguous")
    if series and re.search(r"(?i)S\d{1,2}E\d{1,3}[-+]E?\d", release):
        raise ValueError("multi_episode_release")
    if label(release) == "withheld" or not _source_kind(release):
        raise ValueError("unknown_release_family")
    provider = meta.get("provider")
    if provider not in {"subdl", "subsource", "opensubtitles", "embedded"}:
        raise ValueError("unknown_provider")
    if path.suffix.lower() != ".srt" or path.stat().st_size > MAX_BYTES:
        raise ValueError("unsupported_payload")
    raw = path.read_bytes()
    sha = digest(raw)
    if expected is not None and sha != expected:
        raise ValueError("source_sha_mismatch")
    return Payload(path, imdb, season, episode, provider, release,
                   label(meta.get("sub_id") or path.stem), sha, meta.get("lang"))


def discover(root: Path, capture_roots: list[Path], manifests: list[Path]) -> tuple[list[Payload], Counter]:
    found: list[Payload] = []
    skipped: Counter = Counter()

    def add(path: Path, meta: dict[str, Any], expected: str | None = None) -> None:
        try:
            found.append(payload(path, meta, expected=expected))
        except (ValueError, OSError, TypeError):
            # Categorized errors, never provider-controlled exception text.
            skipped["excluded_or_ambiguous_payload"] += 1

    for sidecar in sorted((root / "subs_cache/_meta").glob("*.json")):
        path = root / "subs_cache" / (sidecar.stem + ".srt")
        if not path.is_file():
            skipped["metadata_without_payload"] += 1
            continue
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            add(path, meta, meta.get("sha256"))
        except (ValueError, TypeError, AttributeError):
            skipped["invalid_metadata"] += 1
    corpus = root / "tests/replay/corpus.json"
    if corpus.is_file():
        for case in json.loads(corpus.read_text(encoding="utf-8")).get("cases", []):
            if case.get("provenance") != "real":
                skipped["derived_replay_case"] += 1
                continue
            if "capture_v2" in case:
                continue  # Captures are inventoried below, without regenerating timings.
            for role in ("target", "reference"):
                spec = case.get("fixtures", {}).get(role, {})
                if spec.get("role") or spec.get("operations"):
                    skipped["generated_or_timing_only_replay"] += 1
                    continue
                identity = case.get("media_identity", [])
                if len(identity) != 3 or not spec.get("sha256"):
                    skipped["unbound_replay_fixture"] += 1
                    continue
                relative = Path(spec.get("path", ""))
                path = (root / relative).resolve()
                if relative.is_absolute() or not path.is_relative_to(root.resolve()):
                    skipped["unsafe_artifact_path"] += 1
                    continue
                add(path, {"imdb_id": identity[0], "season": identity[1], "episode": identity[2],
                           "media_type": case.get("media_type"), "provider": spec.get("provider"),
                           "release_name": spec.get("release_name")}, spec["sha256"])
    # Capture references need per-attempt provider/release identity, never a borrowed stream name.
    for folder in capture_roots + [root / "tests/replay/cases"]:
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("metadata.json")):
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
                if meta.get("provenance") != "live_capture_unreviewed":
                    skipped["unproven_capture"] += 1
                    continue
                if meta.get("imdb_id") == "tt0773262" and meta.get("season") == 8:
                    skipped["exhausted_capture"] += 1
                    continue
                hashes = meta.get("artifact_sha256", {})
                target_meta = {**meta, "provider": meta.get("target_provider"),
                               "release_name": meta.get("target_release_label")}
                add(path.parent / "pristine_target.srt", target_meta, meta.get("target_sha256"))
                for name, sha in hashes.items():
                    if name.endswith("selected_reference.srt"):
                        relative = Path(name)
                        ref = (path.parent / relative).resolve()
                        if relative.is_absolute() or not ref.is_relative_to(path.parent.resolve()):
                            skipped["unsafe_artifact_path"] += 1
                            continue
                        attempts = meta.get("attempts", [])
                        attempt: dict[str, Any] = next((a for a in attempts if a.get("reference_sha256") == sha), {})
                        add(ref, {**meta, "provider": attempt.get("provider"),
                                  "release_name": attempt.get("release_label")}, sha)
            except (ValueError, OSError, TypeError, AttributeError):
                skipped["invalid_capture"] += 1
    # Additional provider files need explicit portable, hash-bound REAL provenance.
    # Timing-only replay fixtures, transformed/served outputs and synthetic cases are never inputs.
    for manifest in manifests:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for row in data.get("payloads", []):
            if row.get("provenance") != "real" or row.get("role") not in {"target", "reference"}:
                skipped["non_pristine_manifest_entry"] += 1
                continue
            relative = Path(row.get("path", ""))
            path = (manifest.parent / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(manifest.parent.resolve()) or not re.fullmatch(r"[a-f0-9]{64}", str(row.get("sha256", ""))):
                skipped["unbound_manifest_entry"] += 1
                continue
            add(path, row, row["sha256"])
    unique: dict[tuple[Any, ...], Payload] = {}
    identities: dict[str, set] = {}
    for item in found:
        identities.setdefault(item.sha256, set()).add(item.identity)
        unique.setdefault((*item.identity, item.sha256), item)
    safe = [p for p in unique.values() if len(identities[p.sha256]) == 1]
    skipped["conflicting_sha_identity"] += len(unique) - len(safe)
    skipped["duplicate_payload"] += len(found) - len(unique)
    return safe, skipped


def pairs(items: list[Payload]) -> list[tuple[Payload, Payload]]:
    return [(t, r) for t in items for r in items
            if t.identity == r.identity and t.sha256 != r.sha256
            and t.family in {"hdtv", "webdl", "stream"} and r.family in {"bluray", "remux"}]


class RawObserver:
    """Request-local memory only; production diagnostic hook calls alass_bytes."""
    failed = False

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.raw: bytes | None = None
        self.input: str | None = None
        self.reference_input: str | None = None

    def alass_bytes(self, raw: bytes, selected: str, reference: str, target: str) -> None:
        self.raw, self.input, self.reference_input = raw, target, reference


def evaluate(target: Payload, reference: Payload, service: SubtitleSyncService) -> dict[str, Any]:
    started = time.perf_counter()
    row: dict[str, Any] = {"imdb_id": target.imdb, "season": target.season,
        "episode": target.episode, "target": target.record(), "reference": reference.record(),
        "evidence_class": EVIDENCE_CLASS, "requires_live_stremio_confirmation": True,
        "alass_launched": False, "legacy": None, "boundary": None, "status": "ERROR"}
    try:
        if target.identity != reference.identity or target.sha256 == reference.sha256:
            raise ValueError("unsafe_pair")
        raw_t, raw_r = target.path.read_bytes(), reference.path.read_bytes()
        if digest(raw_t) != target.sha256 or digest(raw_r) != reference.sha256:
            raise ValueError("source_changed")
        t = decode_subtitle_bytes(raw_t, lang=target.language)
        r = decode_subtitle_bytes(raw_r, lang=reference.language)
        project(t)
        project(r)
        row.update(target_cue_count=len(parse_srt_cues(t)), reference_cue_count=len(parse_srt_cues(r)))
        affine = attempt_timeline_affine(t, r)
        row["timeline_affine"] = {"accepted": affine.accepted,
                                  "hypotheses_tested": affine.hypotheses_tested,
                                  "diagnostics": asdict(affine.diagnostics)}
        if affine.accepted:
            row["status"] = "GLOBAL_SOLVED"
            return row
        sanity = validate_cue_sanity(t, r, threshold_ms=FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS)
        large = None
        ceiling = MAX_PLAUSIBLE_OFFSET_MS
        if not sanity["ok"]:
            seed = classify_large_offset_candidate(t, r, threshold_ms=FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS)
            large = assess_large_offset(t, r, seed_offset_ms=seed, identity_supported=True) if seed is not None else None
            row["large_offset_assessment"] = large.model_dump(mode="json") if large else None
            if large is None or not large.accepted:
                row["status"] = "INELIGIBLE"
                row["rejection_reason"] = "production_cue_sanity_or_large_offset_gate"
                return row
            ceiling = LARGE_OFFSET_MAX_MS
        if sanity["reference_first_ms"] is not None:
            t = strip_intro_credits(t, sanity["reference_first_ms"])
        consistency = measure_alignment_consistency(t, r, is_dialogue=is_dialogue_cue) if large is None else None
        if consistency is not None and consistency.aligned:
            row["status"] = "ALREADY_ALIGNED"
            return row
        if large is not None:
            inv = investigate_large_offset(t, r, identity_supported=True, independent_reference_count=1)
            row["large_offset_investigation"] = inv.model_dump(mode="json")
        pre = decide_anchor_preshift(t, r)
        if pre.accepted and pre.preshift is not None:
            t = preshift_text(t, pre.preshift.shift_ms) or t
        if isinstance(service, SubtitleSyncService) and not shutil.which(service.alass_path):
            row["status"] = "ALASS_UNAVAILABLE"
            row["rejection_reason"] = "production_executable_unavailable"
            return row
        observer, timing = RawObserver(), SyncTiming()
        capture_token = current_capture.set(cast(CaseCapture, observer))
        timing_token = current_sync_timing.set(timing)
        try:
            output = service.sync(t, r, decision_kind="edition", is_series=target.season is not None,
                                  source_confirmed=True, relaxed=False, reference_partial=False)
        finally:
            current_capture.reset(capture_token)
            current_sync_timing.reset(timing_token)
            row["alass_launched"] = timing.alass_processes_launched > 0
        row["alass_service_accepted"] = output is not None
        if observer.raw is None:
            row["status"] = "ALASS_FAILED" if row["alass_launched"] else "INELIGIBLE"
            row["rejection_reason"] = "production_service_no_raw_output"
            return row
        # Exact raw producer artifact, only the same universal-newline representation as production.
        transformed = observer.raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        row["raw_transformed_sha256"] = digest(observer.raw)
        row["verifier_target_sha256"] = digest(t.encode())
        row["alass_input_sha256"] = digest((observer.input or "").encode())
        row["alass_reference_input_sha256"] = digest((observer.reference_input or "").encode())
        legacy = AlignmentAnalyzer().analyze(parse_srt_cues(t), transformed, r,
            alass_applied=True, alass_successful=True, max_plausible_offset_ms=ceiling)
        row["legacy"] = legacy.model_dump(mode="json")
        verified = legacy.sync_state.value.startswith("verified")
        boundary = verify_boundary(t, r, transformed, global_verified=verified)
        row["boundary"] = asdict(boundary)
        row["cue_population_mismatch"] = len(parse_srt_cues(t)) != len(parse_srt_cues(transformed))
        row["status"] = ("ALASS_FAILED" if output is None else
                         "EDGE_CANDIDATE" if boundary.accepted and boundary.classification == "EDGE_DISCONTINUITY" else "BOUNDARY_REJECT")
        row["rejection_reason"] = list(boundary.reasons)
        return row
    except Exception as exc:
        row["error_type"] = type(exc).__name__
        return row
    finally:
        row["runtime_ms"] = round((time.perf_counter() - started) * 1000, 3)


def rank(row: dict[str, Any]) -> tuple:
    b = row.get("boundary") or {}
    seg = b.get("segments", [])
    legacy = row.get("legacy") or {}
    return (row.get("status") == "EDGE_CANDIDATE", row.get("alass_launched", False)
            and bool(legacy) and not str(legacy.get("sync_state", "")).startswith("verified"),
            len(seg) == 2, not row.get("cue_population_mismatch", False), b.get("holdout_pass", False),
            min((s["support"] for s in seg), default=0),
            min((min(s["forward_coverage"], s["reverse_coverage"]) for s in seg), default=0),
            -max((s["residual_p95_ms"] for s in seg), default=1e9),
            -max((s["reference_start_p95_ms"] for s in seg), default=1e9),
            -abs(b.get("gap_disagreement_ms") or 0), abs(b.get("jump_ms") or 0))


def report(rows: list[dict[str, Any]], skipped: Counter) -> dict[str, Any]:
    summary = {"pairs_evaluated": len(rows), "global_solutions": sum(r["status"] == "GLOBAL_SOLVED" for r in rows),
        "ALASS_runs": sum(r["alass_launched"] for r in rows),
        "legacy_verified": sum(str((r.get("legacy") or {}).get("sync_state", "")).startswith("verified") for r in rows),
        "legacy_unverified": sum(bool(r.get("legacy")) and not str(r["legacy"].get("sync_state", "")).startswith("verified") for r in rows),
        "boundary_EDGE_positives": sum(r["status"] == "EDGE_CANDIDATE" for r in rows),
        "boundary_rejects": sum(r["status"] == "BOUNDARY_REJECT" for r in rows),
        "errors": sum(r["status"] in {"ERROR", "ALASS_FAILED", "ALASS_UNAVAILABLE"} for r in rows)}
    ranked = sorted(rows, key=rank, reverse=True)
    return {"evidence_class": EVIDENCE_CLASS, "requires_live_stremio_confirmation": True,
            "summary": summary, "inventory_skips": dict(skipped), "pairs": rows,
            "top_candidates": [r for r in ranked if r["status"] not in {"GLOBAL_SOLVED", "ALREADY_ALIGNED", "ERROR", "INELIGIBLE", "ALASS_FAILED", "ALASS_UNAVAILABLE"}][:10]}


def table(data: dict[str, Any]) -> str:
    lines = ["SCOUT SUMMARY", "---------------", *[f"{k}: {v}" for k, v in data["summary"].items()], "",
             "| Rank | IMDb | S/E | Target ID / release | Reference release | Status |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(data["top_candidates"], 1):
        lines.append(f"| {i} | {r['imdb_id']} | {r['season']}/{r['episode']} | {r['target']['id']} / {r['target']['release']} | {r['reference']['release']} | {r['status']} |")
        if r["status"] == "EDGE_CANDIDATE":
            lines.extend(["EVIDENCE_CLASS=DERIVED_SCOUTING_ONLY", "REQUIRES_LIVE_STREMIO_CONFIRMATION=true"])
    if not data["top_candidates"]:
        lines.append("No eligible ranked live-confirmation candidate found.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, action="append", default=[])
    parser.add_argument("--payload-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alass", default=None)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.is_relative_to(ROOT):
        parser.error("output directory must be outside the repository")
    # Production logs can contain dialogue or subprocess stderr: suppress them entirely.
    logging.disable(logging.CRITICAL)
    items, skipped = discover(ROOT, args.capture_root, args.payload_manifest)
    with tempfile.TemporaryDirectory(prefix="boundary-scout-") as working:
        service = SubtitleSyncService(alass_path=args.alass)
        rows = [evaluate(t, r, service) for t, r in pairs(items)]
        data = report(rows, skipped)
        data["inventory_payloads"] = len(items)
        # Reports are built in private temporary storage; only reports survive the run.
        staged = Path(working) / "report.json"
        staged.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
        json_report = staged.read_text(encoding="utf-8")
    out.mkdir(parents=True, exist_ok=True)
    for name, content in (("boundary-scout.json", json_report),
                          ("boundary-scout.md", table(data))):
        temporary = out / (name + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(out / name)
    print(table(data))


if __name__ == "__main__":
    main()
