"""Isolated provider intake and Docker scouting. Never production cache/evidence writes."""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import re
import subprocess
import sys
import tarfile
import tempfile
import uuid
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import boundary_scout as scout  # noqa: E402

# Explicit series identities, not title searches. Four ordinary episodes per series.
# The globally solved Homeland S02E02 pair is omitted from acquisition.
SEEDS = [(imdb, season, episode) for imdb, season, episodes in (
    ("tt1796960", 2, (3, 4, 6, 7)), ("tt0475784", 1, (2, 4, 6, 8)),
    ("tt0944947", 2, (1, 2, 3, 5)), ("tt1839578", 2, (2, 3, 4, 5)),
    ("tt0903747", 3, (2, 3, 4, 5)), ("tt0773262", 4, (2, 3, 4, 5)),
) for episode in episodes]
MAX_DOWNLOAD = 8_000_000
MAX_PER_ROLE = 3


class IntakeUnavailable(RuntimeError):
    pass


def output_root(path: Path) -> Path:
    root = path.resolve()
    if root.is_relative_to(scout.ROOT) or any(part.lower() in {
        "subs_cache", "cache", "references", "replay", "ninjasubs-captures"
    } for part in root.parts):
        raise ValueError("isolated external output directory required")
    return root


def listing_id(provider: str, download_ref: str) -> str:
    patterns = {"subdl": r"/(\d+(?:-\d+)?)\.zip(?:$|\?)",
                "subsource": r"/subtitles/(\d+)(?:/|$)",
                "opensubtitles": r"/opensubtitles/(\d+)\."}
    match = re.search(patterns.get(provider, r"(?!)"), download_ref)
    if not match:
        raise ValueError("provider_listing_id_unavailable")
    return match[1]


def check_release(release: str, identity: tuple[str, int, int]) -> None:
    if scout.label(release) == "withheld" or not release:
        raise ValueError("unsafe_release_metadata")
    matches = re.findall(r"(?i)S(\d{1,2})E(\d{1,3})", release)
    if not matches or any((int(s), int(e)) != identity[1:] for s, e in matches):
        raise ValueError("episode_identity_ambiguous")
    if re.search(r"(?i)(extended|director.?s?.?cut|unrated|special.?edition|S\d+E\d+[-+]E?\d+)", release):
        raise ValueError("multi_edit_or_multi_episode_release")
    if identity[0] == "tt0758758" or identity[0] == "tt0773262" and identity[1] == 8:
        raise ValueError("exhausted_media")


def extract_pristine(raw: bytes, release: str, identity: tuple[str, int, int]) -> tuple[bytes, str | None]:
    """Retain download and exact unmodified member bytes; never transcode source artifacts."""
    from app.services.sync.decode import select_zip_member
    if not raw or len(raw) > MAX_DOWNLOAD:
        raise ValueError("download_size_limit")
    if zipfile.is_zipfile(io.BytesIO(raw)):
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            infos = archive.infolist()
            if len(infos) > 1000 or sum(i.file_size for i in infos) > 40_000_000:
                raise ValueError("archive_limit")
            chosen = select_zip_member([(i.filename, i.file_size) for i in infos if not i.is_dir()],
                                       identity[1], identity[2], min_bytes=0)
            if not chosen or not chosen.lower().endswith(".srt"):
                raise ValueError("no_pristine_srt_member")
            info = archive.getinfo(chosen)
            if info.file_size > scout.MAX_BYTES:
                raise ValueError("member_size_limit")
            # Never extract a filesystem path; archive.read returns bytes only.
            named = re.findall(r"(?i)S(\d+)E(\d+)", chosen)
            if named and any((int(s), int(e)) != identity[1:] for s, e in named):
                raise ValueError("member_episode_mismatch")
            return archive.read(chosen), scout.label(chosen.rsplit("/", 1)[-1])
    if raw.startswith((b"Rar!", b"7z", b"\x1f\x8b")) or len(raw) > scout.MAX_BYTES:
        raise ValueError("unsupported_pristine_archive")
    return raw, None


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class Corpus:
    def __init__(self, directory: Path):
        self.root = output_root(directory)
        previous = self.root / "intake-manifest.json"
        if previous.exists() and json.loads(previous.read_text()).get("payloads"):
            raise ValueError("nonempty intake corpus requires a new isolated output directory")
        self.entries: list[dict[str, Any]] = []
        self.hash_identities: dict[str, tuple[str, int, int]] = {}
        self.ambiguous_hashes: set[str] = set()
        self.stats: Counter = Counter()
        self.root.mkdir(parents=True, exist_ok=True)
        self.save()

    def save(self) -> None:
        atomic_json(self.root / "intake-manifest.json", {
            "schema_version": 1, "evidence_class": scout.EVIDENCE_CLASS,
            "requires_live_stremio_confirmation": True, "payloads": self.entries,
            "intake_statistics": dict(self.stats)})

    def store(self, raw: bytes, rel: Any, identity: tuple[str, int, int], role: str) -> bool:
        check_release(rel.release_name, identity)
        sid = listing_id(rel.provider, rel.download_url)
        original, member = extract_pristine(raw, rel.release_name, identity)
        text = scout.decode_subtitle_bytes(original, lang=rel.lang)
        scout.project(text)
        if not scout.parse_srt_cues(text):
            raise ValueError("no_parseable_cues")
        sha = scout.digest(original)
        if sha in self.ambiguous_hashes:
            raise ValueError("previously_ambiguous_sha")
        if sha in self.hash_identities:
            if self.hash_identities[sha] != identity:
                self.entries = [e for e in self.entries if e["sha256"] != sha]
                self.hash_identities.pop(sha)
                self.ambiguous_hashes.add(sha)
                self.stats["conflicting_media_sha_refused"] += 1
                self.save()
                raise ValueError("duplicate_sha_conflicting_media")
            self.stats["duplicate_sha"] += 1
            return False
        entry = {"provenance": "real", "role": role, "provider": rel.provider,
                 "sub_id": sid, "imdb_id": identity[0], "season": identity[1], "episode": identity[2],
                 "media_type": "series", "release_name": rel.release_name,
                 "lang": rel.lang, "sha256": sha, "path": f"intake/{sha}/source.srt",
                 "download_sha256": scout.digest(raw), "download_timestamp": datetime.now(UTC).isoformat(),
                 "source_metadata": {"query_imdb_id": identity[0], "query_season": identity[1],
                                     "query_episode": identity[2], "query_languages": ["ar", "en"],
                                     "archive_member": member, "provider_listing_id": sid}}
        intake = self.root / "intake"
        intake.mkdir(exist_ok=True)
        destination = intake / sha
        if destination.exists():
            raise ValueError("existing_payload_requires_explicit_new_run")
        with tempfile.TemporaryDirectory(prefix=".pending-", dir=intake) as temporary:
            stage = Path(temporary) / "complete"
            stage.mkdir()
            (stage / "source.srt").write_bytes(original)
            (stage / "download.bin").write_bytes(raw)
            stage.rename(destination)
        self.entries.append(entry)
        self.hash_identities[sha] = identity
        self.stats["payloads_downloaded"] += 1
        try:
            self.save()
        except Exception:
            self.entries.pop()
            self.hash_identities.pop(sha)
            raise
        return True


async def acquire(corpus: Corpus, providers: list[Any], identities: list[tuple[str, int, int]]) -> None:
    # Sequential requests and bounded role pools preserve provider quotas and memory.
    for identity in identities:
        if identity[1] <= 0 or identity[2] <= 0 or not re.fullmatch(r"tt\d{7,10}", identity[0]):
            corpus.stats["invalid_identity"] += 1
            continue
        corpus.stats["media_identities_queried"] += 1
        pools: dict[str, list[tuple[Any, Any]]] = {"target": [], "reference": []}
        for provider in providers:
            try:
                releases = await provider.search_subtitles(imdb_id=identity[0], is_series=True,
                    season=identity[1], episode=identity[2], languages=["ar", "en"], exclude_hi=True)
                for rel in releases:
                    try:
                        check_release(rel.release_name, identity)
                        listing_id(rel.provider, rel.download_url)
                    except ValueError:
                        corpus.stats["metadata_refused"] += 1
                        continue
                    family = scout._source_kind(rel.release_name)
                    role = ("target" if rel.lang in {"ar", "ara"} and family in {"hdtv", "webdl", "stream"}
                            else "reference" if rel.lang in {"en", "eng"} and family in {"bluray", "remux"} else None)
                    if role:
                        pools[role].append((provider, rel))
            except Exception:
                corpus.stats["provider_search_errors"] += 1
            await asyncio.sleep(0.7)
        if not all(pools.values()):
            corpus.stats["no_safe_cross_release_pool"] += 1
            continue
        corpus.stats["media_identities_discovered"] += 1
        for role, pool in pools.items():
            pool.sort(key=lambda p: (scout._source_kind(p[1].release_name) != "hdtv" if role == "target" else
                                    scout._source_kind(p[1].release_name) != "remux", p[1].provider, p[1].release_name))
            stored = 0
            for provider, rel in pool[:12]:
                if stored >= MAX_PER_ROLE:
                    break
                try:
                    raw = await provider.download_archive(rel.download_url)
                    if not raw:
                        raise ValueError("provider_download_failed")
                    stored += corpus.store(raw, rel, identity, role)
                except Exception:
                    corpus.stats["provider_download_or_payload_errors"] += 1
                await asyncio.sleep(0.7)
        corpus.save()


def alass_preflight(executable: str) -> None:
    try:
        result = subprocess.run([executable, "--version"], capture_output=True, timeout=20)
        if result.returncode != 0:
            raise IntakeUnavailable("production ALASS version check failed")
    except (OSError, subprocess.SubprocessError) as exc:
        raise IntakeUnavailable("production ALASS unavailable") from exc


async def scan_inside(directory: Path) -> None:
    import httpx

    from app.config import settings
    from app.providers.opensubtitles import OpenSubtitlesProvider
    from app.providers.subdl import SubdlProvider
    from app.providers.subsource import SubsourceProvider
    alass_preflight(settings.ALASS_PATH)
    corpus = Corpus(directory)
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        providers = [cls(client) for cls, key in (
            (SubdlProvider, settings.SUBDL_API_KEY), (SubsourceProvider, settings.SUBSOURCE_API_KEY),
            (OpenSubtitlesProvider, settings.OPENSUBTITLES_API_KEY)) if key]
        if not providers:
            raise IntakeUnavailable("no configured provider API credentials")
        await acquire(corpus, providers, SEEDS)
    # Empty inventory root prevents any production/replay-cache discovery.
    with tempfile.TemporaryDirectory(prefix="scout-inventory-") as empty:
        items, skipped = scout.discover(Path(empty), [], [corpus.root / "intake-manifest.json"])
    pair_list = scout.pairs(items)
    rows = [scout.evaluate(t, r, scout.SubtitleSyncService()) for t, r in pair_list]
    data = scout.report(rows, skipped)
    reasons: Counter = Counter()
    for row in rows:
        if row["status"] == "BOUNDARY_REJECT":
            reasons.update((row.get("boundary") or {}).get("reasons", []))
    data["run_status"] = "COMPLETED"
    data["expanded_summary"] = {**dict(corpus.stats), "unique_subtitle_sha256": len(items),
        "eligible_cross_release_pairs": len(pair_list), "GLOBAL_SOLVED": data["summary"]["global_solutions"],
        "ALASS_eligible": sum(r.get("alass_launched", False) or r["status"] in {"ALASS_FAILED", "ALASS_UNAVAILABLE"} for r in rows),
        **data["summary"], "boundary_reject_reason_histogram": dict(reasons)}
    atomic_json(corpus.root / "boundary-scout-expanded.json", data)
    expanded = "\n".join(f"{k}: {v}" for k, v in data["expanded_summary"].items())
    (corpus.root / "boundary-scout-expanded.md").write_text(expanded + "\n\n" + scout.table(data), encoding="utf-8")


def docker_run(container: str, destination: Path) -> None:
    """Run a snapshot under /tmp in the existing container, leaving /app untouched."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", container):
        raise ValueError("invalid_container_name")
    destination = output_root(destination)
    previous = destination / "intake-manifest.json"
    if previous.exists() and json.loads(previous.read_text()).get("payloads"):
        raise ValueError("nonempty intake corpus requires a new isolated output directory")
    code = ("import subprocess;from app.config import settings;"
            "r=subprocess.run([settings.ALASS_PATH,'--version'],capture_output=True,timeout=20);"
            "raise SystemExit(r.returncode)")
    probe = subprocess.run(["docker", "exec", container, "python", "-c", code], capture_output=True, timeout=30)
    if probe.returncode:
        raise IntakeUnavailable("Docker access or production ALASS preflight unavailable")
    isolated = "/tmp/ninjasubs-boundary-scout-" + uuid.uuid4().hex
    snapshot = isolated + "/snapshot"
    copied = False
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path in sorted((scout.ROOT / "app").rglob("*.py")):
            archive.add(path, arcname=path.relative_to(scout.ROOT).as_posix(), recursive=False)
        for name in ("boundary_scout.py", "boundary_scout_intake.py"):
            archive.add(scout.ROOT / "tools" / name, arcname="tools/" + name)
    try:
        unpack = "import os,sys,tarfile;os.makedirs(sys.argv[1]);tarfile.open(fileobj=sys.stdin.buffer,mode='r|').extractall(sys.argv[1],filter='data')"
        r = subprocess.run(["docker", "exec", "-i", container, "python", "-c", unpack, snapshot],
                           input=buffer.getvalue(), capture_output=True, timeout=60)
        if r.returncode:
            raise IntakeUnavailable("isolated Docker snapshot failed")
        r = subprocess.run(["docker", "exec", "-w", snapshot, "-e", "PYTHONPATH=" + snapshot,
            container, "python", "tools/boundary_scout_intake.py", "--in-container",
            "--output-dir", isolated + "/outputs/boundary-scout"], capture_output=True, timeout=3600)
        if r.returncode:
            raise IntakeUnavailable("isolated provider/scout execution failed; no completed scan")
        destination.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["docker", "cp", container + ":" + isolated + "/outputs/boundary-scout/.",
                            str(destination)], capture_output=True, timeout=60)
        if r.returncode:
            raise IntakeUnavailable("isolated report copy failed")
        copied = True
    finally:
        cleanup = "import shutil,sys;shutil.rmtree(sys.argv[1],ignore_errors=True)"
        # On copy/scan failure preserve any acquired corpus under the unique /tmp
        # workspace for recovery; remove only the Python snapshot in that case.
        subprocess.run(["docker", "exec", container, "python", "-c", cleanup,
                        isolated if copied else snapshot], capture_output=True, timeout=30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--container", default="ninjasubs")
    parser.add_argument("--in-container", action="store_true")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    try:
        if args.in_container:
            asyncio.run(scan_inside(args.output_dir))
        else:
            docker_run(args.container, args.output_dir)
    except (IntakeUnavailable, OSError, subprocess.SubprocessError, ValueError) as exc:
        reason = str(exc) if isinstance(exc, IntakeUnavailable) else type(exc).__name__
        print("INTAKE/SCOUT NOT COMPLETED: " + reason)
        try:
            root = output_root(args.output_dir)
            root.mkdir(parents=True, exist_ok=True)
            data = scout.report([], Counter())
            data.update(run_status="NOT_COMPLETED", blocker=reason, expanded_summary={
                "media_identities_discovered": 0, "subtitle_payloads_acquired": 0,
                "unique_subtitle_sha256": 0, "eligible_cross_release_pairs": 0,
                "GLOBAL_SOLVED": 0, "ALASS_eligible": 0, "actual_ALASS_runs": 0,
                "legacy_verified": 0, "legacy_unverified": 0,
                "EDGE_DISCONTINUITY_accepted": 0, "boundary_reject_reason_histogram": {}, "errors": 1})
            # Never overwrite a previous corpus or completed report on failure.
            if not (root / "intake-manifest.json").exists():
                atomic_json(root / "intake-manifest.json", {"schema_version": 1,
                    "payloads": [], "run_status": "NOT_COMPLETED", "blocker": reason,
                    "evidence_class": scout.EVIDENCE_CLASS})
            if not (root / "boundary-scout-expanded.json").exists():
                atomic_json(root / "boundary-scout-expanded.json", data)
                (root / "boundary-scout-expanded.md").write_text(
                    "INTAKE/SCOUT NOT COMPLETED: " + reason + "\nNo provider acquisition or scan completed.\n",
                    encoding="utf-8")
        except (OSError, ValueError):
            pass
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
