"""Mad Men S01E01 quality-fallback benchmark (DBO and SA89).

Uses the real cached target/reference bytes and real ALASS in the container.
The two candidate references are the cached WEB-DL BTN reference (weak binding,
the production failure) and the cached SA89 BluRay reference (source-incompatible
with the WADU video; used to prove the fallback does not blindly serve it).

Run with ITW_APP_ROOT=/app for the pre-quality deployed image and ITW_APP_ROOT=/work
for the working tree.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = Path(os.environ.get("ITW_APP_ROOT", str(ROOT))).resolve()
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

VIDEO = "Mad.Men.S01E01.Smoke.Gets.in.Your.Eyes.REPACK.2160p.HMAX.WEB-DL.DDP5.1.DV.HDR.H.265-WADU.mkv"
TARGETS = {
    "DBO": ("3d449ce4ce4d0d28", "Mad.Men.S01.720p.BluRay.DTS.x264-DBO.srt"),
    "SA89": ("a437fb8d5539353a", "Mad.Men.S01.1080p.BluRay.DD5.1.x264-SA89.srt"),
}
REFS = [
    ("BTN", "Mad.Men.S01.720p.WEB-DL.AAC2.0.H264-BTN.srt",
     "/app/cache/references/tt0804503_1_1_WADU_v2_e3f3e720de3ca96d.video.subdl.edition.71f906776825a9c2.srt"),
    ("SA89REF", "Mad.Men.S01.1080p.BluRay.DD5.1.x264-SA89.srt",
     "/app/cache/references/tt0804503_1_1_SA89_v2_fa0011efbe0acc3e.video.subdl.edition.89f334377eb99168.srt"),
]


async def _main(runs: int) -> int:
    import app.services.sync.orchestrator as om  # noqa: F401  (import-order guard)
    from app.models import SubtitleRelease
    from app.services.release_scoring_integration import record
    from app.services.sync.cache import ReferenceDiskCache
    from app.services.sync.external_strategy import ExternalExactStrategy
    from app.services.sync.orchestrator import SyncOrchestrator
    from app.services.sync.query import ResolvedReference
    from app.services.sync.telemetry import SyncTiming
    from app.services.sync_cache import SyncCache
    from app.services.sync_service import SubtitleSyncService

    svc = SubtitleSyncService()
    if not svc.is_available():
        print("alass unavailable", file=sys.stderr)
        return 3
    ref_text = {key: Path(path).read_text("utf-8") for key, _, path in REFS}

    class _Prov:
        name = "subdl"

        def __init__(self) -> None:
            self.downloads = 0

        def is_breaker_open(self) -> bool:
            return False

        async def download_archive(self, release, api_key=None, **kwargs):
            return b""

    class _Strategy(ExternalExactStrategy):
        def __init__(self, root: Path) -> None:
            super().__init__(cache=ReferenceDiskCache(root=root, min_bytes=1), timeout=5.0)
            self.provider = _Prov()
            self._subdl = self.provider
            self._subsource = None
            self._opensubtitles = None

        async def _gather_candidates(self, providers, query):
            out = []
            for key, name, _path in REFS:
                release = SubtitleRelease(
                    release_name=name,
                    download_url=f"https://invalid/{key}",
                    provider="subdl", lang="eng",
                )
                evidence = record(release)
                evidence["metadata"].update(
                    imdb_id="tt0804503", imdb_verified=True,
                    season=1, episode=1,
                    source="webdl" if key == "BTN" else "bluray",
                )
                release.release_evidence = [evidence]
                out.append((release, self.provider))
            return out

        async def _download_candidate(self, best, provider, query):
            name = str(getattr(best, "release_name", ""))
            provider.downloads += 1
            key = "BTN" if "BTN" in name else "SA89REF"
            return ResolvedReference(
                ref_text[key], kind="edition",
                bluray_match=False, candidate=name, provider="subdl",
                member_season=1, member_episode=1,
            )

    def meta(sub_id: str, release_name: str) -> dict:
        return {
            "imdb_id": "tt0804503", "media_type": "series",
            "season": 1, "episode": 1, "target_filename": VIDEO,
            "video_size": "8113838532", "lang": "ara",
            "release_name": release_name, "sub_id": sub_id, "year": 2007,
        }

    captured: list[SyncTiming] = []
    orig = SyncTiming.log

    def capture(self, result, total_ms):
        self.total_ms = total_ms
        captured.append(self)
        return orig(self, result, total_ms)

    SyncTiming.log = capture

    results = []
    for _ in range(max(1, runs)):
        for label, (sub_id, release_name) in TARGETS.items():
            target = Path(f"/app/cache/{sub_id}.srt").read_bytes()
            with tempfile.TemporaryDirectory() as cache_dir:
                strategy = _Strategy(Path(cache_dir))
                orchestrator = SyncOrchestrator(
                    external_strategy=strategy,
                    sync_service=svc,
                    sync_cache=SyncCache(),
                )
                captured.clear()
                started = time.monotonic()
                served = await orchestrator.evaluate_and_sync(
                    target, meta(sub_id, release_name), sub_id, auto_sync=True
                )
                elapsed = (time.monotonic() - started) * 1000
                timing = captured[-1] if captured else SyncTiming()
                content_keys = [
                    key for key in getattr(orchestrator._sync_cache, "_local", {})
                    if "large_offset_content_v1" in key
                ]
                results.append({
                    "case": label,
                    "elapsed_ms": round(elapsed, 1),
                    "internal_ms": round(float(getattr(timing, "total_ms", elapsed)), 1),
                    "downloads": int(timing.references_downloaded),
                    "alass_runs": int(timing.alass_processes_launched),
                    "strategy": timing.winning_sync_strategy,
                    "served_sha": hashlib.sha256(served).hexdigest()[:16],
                    "served_original": served == target,
                    "content_artifacts": len(content_keys),
                    "quality_triggered": bool(getattr(timing, "quality_fallback_triggered", False)),
                    "quality_attempts": int(getattr(timing, "quality_fallback_attempts", 0)),
                    "quality_served": bool(getattr(timing, "quality_fallback_served", False)),
                    "quality_exhausted": bool(getattr(timing, "quality_fallback_exhausted", False)),
                    "first_p95": getattr(timing, "quality_fallback_first_p95_ms", None),
                    "first_binding": getattr(timing, "quality_fallback_first_binding", None),
                })
    print(json.dumps({"runs": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=2)
    raise SystemExit(asyncio.run(_main(p.parse_args().runs)))
