"""Validate local captured SRT artifacts and admit a replay case only on PASS."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.subtitle_matcher import parse_srt_cues  # noqa: E402
from app.services.sync.boundary_verifier import MAX_CHARS, MAX_CUES  # noqa: E402
from app.services.sync.pal_two_point import _is_usable_dialogue  # noqa: E402
from tools.sync_replay import ROOT, case_mismatches, fixture_path, replay_case  # noqa: E402

CASE_TYPES = {"GLOBAL_AFFINE", "CONSTANT_OFFSET", "EDGE_DISCONTINUITY",
              "DIFFERENT_CUT_REJECT", "WRONG_REFERENCE_REJECT",
              "MULTI_DISCONTINUITY_REJECT", "INSUFFICIENT_EVIDENCE_REJECT"}
SAFE_METADATA = {"target_release_family", "reference_release_family", "media_type",
                 "season", "episode"}
UNSAFE = re.compile(
    r"(?i)(?:[a-z][a-z0-9+.-]*://|\bwww\.|[a-z]:[/\\]|\\\\|/(?:home|users|tmp)/|"
    r"\b(?:api[_-]?key|(?:access[_-]?|refresh[_-]?)?token|password|secret|authorization)[\"']?\s*[:=]|"
    r"\bBearer\s+\S+|\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+|"
    r"\bAKIA[A-Z0-9]{16}\b|\bsk-[A-Za-z0-9_-]{16,})"
)
STAMP = r"(?:\d{1,2}:)?\d{1,2}:\d{2}[,.]\d{1,3}"
TIMING = re.compile(rf"^({STAMP})\s*-->\s*({STAMP})(?:[ \t]+[^\n]*)?$")


def safe_id(value: str) -> str:
    if (not re.fullmatch(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", value) or len(value) > 80 or UNSAFE.search(value)
            or re.fullmatch(r"con|prn|aux|nul|com[1-9]|lpt[1-9]", value)):
        raise ValueError("invalid case_id")
    return value


def safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(metadata, dict) or set(metadata) - SAFE_METADATA:
        raise ValueError("metadata contains unsupported fields")
    for key, value in metadata.items():
        if key in {"season", "episode"}:
            if type(value) is not int or not 1 <= value <= 999:
                raise ValueError("invalid episode metadata")
        elif (not isinstance(value, str) or not value or len(value) > 160
              or UNSAFE.search(value) or re.search(r"[A-Za-z0-9_-]{40,}", value)
              or not re.fullmatch(r"[\w .+()\[\]-]+", value)):
            raise ValueError("unsafe metadata label")
    if metadata.get("media_type", "series") not in {"movie", "series"}:
        raise ValueError("invalid media_type")
    return dict(metadata)


def validate_srt(raw: bytes) -> tuple[str, list[tuple[int, int, str]]]:
    if not raw or len(raw) > MAX_CHARS * 4 + 3:
        raise ValueError("empty or oversized subtitle")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeError:
        raise ValueError("subtitle must be UTF-8; convert locally before import") from None
    safety_text = "".join(c for c in unicodedata.normalize("NFKC", text)
                          if unicodedata.category(c) != "Cf")
    if len(text) > MAX_CHARS or UNSAFE.search(safety_text) or "\x00" in text:
        raise ValueError("unsafe or oversized subtitle content")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    blocks = [block for block in re.split(r"\n[ \t]*\n", normalized)
              if any(not c.isspace() and unicodedata.category(c) != "Cf" for c in block)]
    if len(blocks) > MAX_CUES:
        raise ValueError("too many cues")
    cues = []
    for block in blocks:
        lines = block.splitlines()
        index = 1 if lines and lines[0].strip().isdigit() else 0
        timing = TIMING.fullmatch(lines[index].strip()) if len(lines) > index else None
        if timing is None or not "\n".join(lines[index + 1:]).strip():
            raise ValueError("invalid SRT block or negative timestamp")
        for stamp in timing.groups():
            pieces = stamp.replace(",", ".").split(".")[0].split(":")
            if int(pieces[-1]) >= 60 or int(pieces[-2]) >= 60:
                raise ValueError("invalid timestamp fields")
        parsed = parse_srt_cues(block)
        if len(parsed) != 1 or parsed[0][0] < 0 or parsed[0][1] <= parsed[0][0]:
            raise ValueError("invalid cue timing")
        cues.extend(parsed)  # Parsing each block preserves original file order.
    if not cues or any(a[0] > b[0] or a[1] > b[1] for a, b in zip(cues, cues[1:], strict=False)):
        raise ValueError("invalid cue ordering")
    return text, cues


def dialogue(cues: list[tuple[int, int, str]]) -> list[str]:
    return [re.sub(r"\s+", " ", re.sub(r"<[^>]*>", "", text)).strip()
            for _, _, text in cues if _is_usable_dialogue(text)]


def artifact_index(root: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    for folder in (root / "tests/fixtures", root / "tests/replay/cases"):
        if folder.exists():
            for path in sorted(folder.rglob("*.srt")):
                relative = path.relative_to(root).as_posix()
                if (path.is_symlink() or not path.resolve().is_relative_to(root.resolve())
                        or UNSAFE.search(relative) or path.stat().st_size > MAX_CHARS * 4 + 3):
                    continue
                index.setdefault(hashlib.sha256(path.read_bytes()).hexdigest(),
                                 path.relative_to(root).as_posix())
    return index


def import_case(**kwargs: Any) -> dict[str, Any]:
    """Serialize imports; never overwrite another import's manifest update."""
    root = kwargs.get("root", ROOT)
    lock = fixture_path("tests/replay/.case-import.lock", root)
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError("another import owns the corpus lock") from None
    try:
        os.close(handle)
        return _import_case(**kwargs)
    finally:
        lock.unlink(missing_ok=True)


def _import_case(*, case_id: str, inputs: dict[str, Path], case_type: str,
                expected_boundary_accepted: bool | None,
                metadata: dict[str, Any] | None = None,
                expected_boundary_run: bool | None = None,
                expected_parent_model: str | None = None,
                expected_parent_model_failure: bool | None = None,
                expected_reason: str = "", provenance: str = "real",
                source_case_id: str | None = None,
                allow_identical_target_reference: bool = False,
                root: Path = ROOT) -> dict[str, Any]:
    safe_id(case_id)
    metadata = safe_metadata(metadata or {})
    if case_type not in CASE_TYPES or type(expected_boundary_accepted) is not bool:
        raise ValueError("explicit supported case type and boundary outcome required")
    if expected_boundary_accepted != (case_type == "EDGE_DISCONTINUITY"):
        raise ValueError("case type contradicts expected outcome")
    if provenance not in {"real", "derived", "synthetic"}:
        raise ValueError("invalid provenance")
    if provenance == "real" and source_case_id is not None:
        raise ValueError("real cases cannot claim a derived source")
    if provenance == "derived" and source_case_id is None:
        raise ValueError("derived case requires source_case_id")
    if source_case_id is not None:
        safe_id(source_case_id)
    if expected_parent_model not in {None, "PAL_LIKE_AFFINE", "GLOBAL_AFFINE", "CONSTANT_OFFSET"}:
        raise ValueError("invalid expected parent model")
    if not re.fullmatch(r"[a-z_]*", expected_reason):
        raise ValueError("invalid expected reason")
    for value in (expected_boundary_run, expected_parent_model_failure):
        if value is not None and type(value) is not bool:
            raise ValueError("expectation must be boolean")
    if set(inputs) - {"target", "reference", "transformed", "served"} or not {
            "target", "reference", "transformed"}.issubset(inputs):
        raise ValueError("target/reference/transformed files required")
    manifest_path = fixture_path("tests/replay/corpus.json", root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    if any(case["case_id"] == case_id for case in cases):
        raise ValueError("case_id already exists; existing cases are never overwritten")
    if source_case_id and not any(case["case_id"] == source_case_id for case in cases):
        raise ValueError("unknown source_case_id")
    raw_by_role, parsed_by_role, hashes = {}, {}, {}
    for role in ("target", "reference", "transformed", "served"):
        if role not in inputs:
            continue
        path = inputs[role]
        try:
            if path.stat().st_size > MAX_CHARS * 4 + 3:
                raise ValueError(f"{role} subtitle oversized")
            raw = path.read_bytes()
        except OSError:
            raise ValueError(f"{role} input unavailable") from None
        _, cues = validate_srt(raw)
        raw_by_role[role], parsed_by_role[role] = raw, cues
        hashes[role] = hashlib.sha256(raw).hexdigest()
    if hashes["target"] == hashes["reference"] and not allow_identical_target_reference:
        raise ValueError("identical target/reference requires explicit allowance")
    speech = dialogue(parsed_by_role["target"])
    if not speech or any(dialogue(parsed_by_role[role]) != speech
                         for role in ("transformed", "served") if role in parsed_by_role):
        raise ValueError("target/output dialogue population or ordering differs")
    destination = fixture_path(f"tests/replay/cases/{case_id}", root)
    if destination.exists():
        raise ValueError("case directory already exists")
    expected_run = (case_type not in {"GLOBAL_AFFINE", "CONSTANT_OFFSET"}
                    if expected_boundary_run is None else expected_boundary_run)
    contract = {"expected_boundary_run": expected_run,
                "expected_boundary_accepted": expected_boundary_accepted,
                "expected_classification": "EDGE_DISCONTINUITY" if expected_boundary_accepted else "UNSUPPORTED"}
    if expected_parent_model is not None:
        contract["expected_parent_model"] = expected_parent_model
    if expected_parent_model_failure is not None:
        contract["expected_parent_model_failure"] = expected_parent_model_failure
    entry = {
        "case_id": case_id, "case_type": case_type, "provenance": provenance,
        "source_case_id": source_case_id, "media_type": metadata.get("media_type", "series"),
        "target_release_family": metadata.get("target_release_family", "unspecified"),
        "reference_release_family": metadata.get("reference_release_family", "unspecified"),
        "expected_model": case_type if case_type in {"GLOBAL_AFFINE", "CONSTANT_OFFSET", "EDGE_DISCONTINUITY"} else "UNSUPPORTED",
        "expected_shadow_outcome": expected_boundary_accepted,
        "expected_reason": expected_reason, "boundary_should_run": expected_run,
        "expectations": contract, "fixtures": {}, "artifact_metadata": {},
        "notes": "Local captured artifacts; no input paths retained.",
        "allow_identical_target_reference": allow_identical_target_reference,
        "production_context": {"alass_applied": True, "alass_successful": True},
        **metadata,
    }
    index = artifact_index(root)
    # Temporary artifacts and reports are outside the visible corpus until replay passes.
    with tempfile.TemporaryDirectory(prefix=".sync-import-", dir=manifest_path.parent) as temporary:
        stage = Path(temporary)
        staged_fixtures = {}
        for role, raw in raw_by_role.items():
            sha = hashes[role]
            shared = index.get(sha)
            name = "final_served.srt" if role == "served" else role + ".srt"
            final_path = shared or (destination / name).relative_to(root).as_posix()
            if shared is None:
                (stage / name).write_bytes(raw)
                staged_path = (stage / name).relative_to(root).as_posix()
                index[sha] = final_path
            elif final_path.startswith(destination.relative_to(root).as_posix() + "/"):
                staged_path = (stage / Path(final_path).name).relative_to(root).as_posix()
            else:
                staged_path = shared
            entry["fixtures"][role] = {"path": final_path}
            staged_fixtures[role] = {"path": staged_path}
            entry["artifact_metadata"][role] = {"sha256": sha, "cue_count": len(parsed_by_role[role]),
                                                   "reused": shared is not None}
        def check_identity() -> None:
            for role, spec in staged_fixtures.items():
                actual = hashlib.sha256(fixture_path(spec["path"], root).read_bytes()).hexdigest()
                if actual != hashes[role]:
                    raise ValueError("artifact changed during import")

        check_identity()
        result = replay_case({**entry, "fixtures": staged_fixtures}, root)
        check_identity()
        mismatch = case_mismatches(entry, result["evidence"])
        if mismatch:
            actual_reason = ",".join(result["evidence"]["shadow"]["reasons"]) or "accepted"
            raise ValueError("replay expectation mismatch: " + ",".join(mismatch) + "; reason=" + actual_reason)
        report = {"case_id": case_id, "provenance": provenance,
                  "artifacts": entry["artifact_metadata"], "replay": result, "passed": True}
        (stage / "case.json").write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (stage / "diagnostic.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # Re-check after evaluation. Do not clobber concurrent/manual manifest edits.
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("manifest changed during import; retry")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ValueError("case directory appeared during import")
        os.rename(stage, destination)
        pending: Path | None = None
        try:
            updated = {**manifest, "cases": [*cases, entry]}
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=manifest_path.parent,
                                             delete=False, suffix=".tmp") as handle:
                pending = Path(handle.name)
                handle.write(json.dumps(updated, indent=2) + "\n")
            if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
                raise ValueError("manifest changed during import; retry")
            os.replace(pending, manifest_path)
        except BaseException:
            # Only the just-created, contained case directory is owned by this import.
            if destination.resolve().is_relative_to((root / "tests/replay/cases").resolve()):
                shutil.rmtree(destination)
            raise
        finally:
            if pending is not None:
                pending.unlink(missing_ok=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    for role in ("target", "reference", "transformed", "served"):
        parser.add_argument("--" + role, type=Path, required=role != "served")
    parser.add_argument("--expected-model", choices=sorted(CASE_TYPES), required=True)
    parser.add_argument("--expected-boundary-accepted", choices=("true", "false"), required=True)
    parser.add_argument("--expected-boundary-run", choices=("true", "false"))
    parser.add_argument("--expected-parent-model")
    parser.add_argument("--expected-parent-model-failure", choices=("true", "false"))
    parser.add_argument("--expected-reason", default="")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--provenance", choices=("real", "derived", "synthetic"), default="real")
    parser.add_argument("--source-case-id")
    parser.add_argument("--allow-identical-target-reference", action="store_true")
    args = parser.parse_args()
    try:
        metadata = json.loads(args.metadata.read_text(encoding="utf-8")) if args.metadata else {}
        report = import_case(case_id=args.case_id,
            inputs={role: getattr(args, role) for role in ("target", "reference", "transformed", "served") if getattr(args, role)},
            case_type=args.expected_model, expected_boundary_accepted=args.expected_boundary_accepted == "true",
            metadata=metadata, expected_boundary_run=None if args.expected_boundary_run is None else args.expected_boundary_run == "true",
            expected_parent_model=args.expected_parent_model,
            expected_parent_model_failure=None if args.expected_parent_model_failure is None else args.expected_parent_model_failure == "true",
            expected_reason=args.expected_reason, provenance=args.provenance, source_case_id=args.source_case_id,
            allow_identical_target_reference=args.allow_identical_target_reference)
    except ValueError as exc:
        # Validation exceptions are fixed safe vocabulary; never echo artifact text/paths.
        print("FAIL " + str(exc))
        return 1
    except (OSError, TypeError, KeyError):
        print("FAIL import unavailable or malformed input; no private error details")
        return 1
    print(report["case_id"] + " PASS expected behavior reproduced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
