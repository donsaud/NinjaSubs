"""Two-subtitle reuse benchmark for the Into the Wild video.

First call: production-shaped discovery/download/ALASS for subtitle ID 1.
Second call: a different subtitle ID with byte-identical original content.

The script prints per-call downloads, ALASS executions, cache state and served
SHA. It uses the real orchestrator/verifier/ALASS and mocked providers with the
exact cached reference bytes. No provider network request is made.
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
FIXTURES = ROOT / "tests" / "fixtures" / "into_the_wild"
APP_ROOT = Path(os.environ.get("ITW_APP_ROOT", str(ROOT))).resolve()
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

TARGET = FIXTURES / "target.srt"
ARA = FIXTURES / "reference_ara_fsihd.srt"
ENG = FIXTURES / "reference_eng_fsihd.srt"
IMDB = "tt0758758"
VIDEO = "Into.The.Wild.2007.1080p.BluRay.x264-FSiHD"
SUBTITLE = "Into The Wild.WEBRip.Netflix.srt"
ID1 = "d8efd4b030f64a1d"
ID2 = "92788b7594b7f497"
DELAYS = {VIDEO + ".srt": 1.0, VIDEO + ".ENG.srt": 0.65}


async def _main(runs: int) -> int:
    import app.services.sync.orchestrator as om
    from app.models import SubtitleRelease
    from app.services.release_scoring_integration import record
    from app.services.sync.cache import ReferenceDiskCache
    from app.services.sync.external_strategy import ExternalExactStrategy
    from app.services.sync.orchestrator import SyncOrchestrator
    from app.services.sync.query import ResolvedReference
    from app.services.sync.telemetry import SyncTiming
    from app.services.sync_cache import SyncCache
    from app.services.sync_service import SubtitleSyncService

    if not all(path.exists() for path in (TARGET, ARA, ENG)):
        print(f"missing fixtures at {FIXTURES}", file=sys.stderr)
        return 2
    target_bytes = TARGET.read_bytes()
    ara = ARA.read_text("utf-8")
    eng = ENG.read_text("utf-8")
    ara_digest = hashlib.sha256(ara.encode("utf-8")).hexdigest()
    svc = SubtitleSyncService()
    if not svc.is_available():
        print("alass unavailable", file=sys.stderr)
        return 3

    real_validate = om.validate_cue_sanity

    def production_shaped_validate(candidate, reference, **kwargs):
        if hashlib.sha256(str(reference).encode("utf-8")).hexdigest() == ara_digest:
            return {
                "ok": True, "delta_ms": 0, "candidate_first_ms": 2000,
                "reference_first_ms": 2000, "threshold_ms": kwargs.get("threshold_ms"),
                "penalty": 0, "reason": None,
            }
        return real_validate(candidate, reference, **kwargs)

    om.validate_cue_sanity = production_shaped_validate

    class _Provider:
        def __init__(self, name: str) -> None:
            self.name = name
            self.downloads = 0

        def is_breaker_open(self) -> bool:
            return False

        async def download_archive(self, release, api_key=None, **kwargs):
            return b""

    class _Strategy(ExternalExactStrategy):
        def __init__(self, root: Path) -> None:
            super().__init__(cache=ReferenceDiskCache(root=root, min_bytes=1), timeout=5.0)
            self.providers = {"subdl": _Provider("subdl"), "subsource": _Provider("subsource")}
            self._subdl = self.providers["subdl"]
            self._subsource = self.providers["subsource"]
            self._opensubtitles = None
            self.gather_calls = 0

        async def _gather_candidates(self, providers, query):
            self.gather_calls += 1
            await asyncio.sleep(1.9)
            specs = [
                (VIDEO + ".srt", "subdl", "https://invalid/subdl/fsihd"),
                (VIDEO + ".srt", "subsource", "https://invalid/subsource/fsihd"),
                (VIDEO + ".ENG.srt", "subdl", "https://invalid/subdl/fsihd-eng"),
            ]
            out = []
            for name, provider_name, url in specs:
                release = SubtitleRelease(
                    release_name=name, download_url=url, provider=provider_name,
                    lang="ara" if "ENG" not in name else "eng",
                )
                evidence = record(release)
                evidence["metadata"].update(imdb_id=IMDB, imdb_verified=True, source="bluray")
                release.release_evidence = [evidence]
                out.append((release, self.providers[provider_name]))
            return out

        async def _download_candidate(self, best, provider, query):
            name = str(getattr(best, "release_name", ""))
            await asyncio.sleep(DELAYS.get(name, 0.5))
            provider.downloads += 1
            return ResolvedReference(
                ara if "ENG" not in name else eng,
                kind="edition", bluray_match=True, candidate=name, provider=provider.name,
                # Production's movie proof records season/episode 0, the legacy
                # value that exposed the video-cache 0-vs-None mismatch.
                member_season=0, member_episode=0,
            )

    def meta(sub_id: str) -> dict:
        return {
            "imdb_id": IMDB, "media_type": "movie", "target_filename": VIDEO,
            "video_size": "13150936000", "lang": "ara", "release_name": SUBTITLE,
            "sub_id": sub_id, "year": 2007,
        }

    captured: list[SyncTiming] = []
    original_log = SyncTiming.log

    def capture_log(self, result, total_ms):
        self.total_ms = total_ms
        captured.append(self)
        return original_log(self, result, total_ms)

    SyncTiming.log = capture_log

    async def call(orchestrator, sub_id: str, payload: bytes = target_bytes) -> dict:
        captured.clear()
        started = time.monotonic()
        served = await orchestrator.evaluate_and_sync(
            payload, meta(sub_id), sub_id, auto_sync=True
        )
        elapsed = (time.monotonic() - started) * 1000
        timing = captured[-1] if captured else SyncTiming()
        return {
            "sub_id": sub_id,
            "elapsed_ms": round(elapsed, 1),
            "internal_ms": round(float(getattr(timing, "total_ms", elapsed)), 1),
            "downloads": int(timing.references_downloaded),
            "alass_runs": int(timing.alass_processes_launched),
            "cache_hit": bool(timing.cache_hit),
            "strategy": timing.winning_sync_strategy,
            "served_sha": hashlib.sha256(served).hexdigest()[:16],
            "transformed": served != payload,
        }

    results = []
    for _ in range(max(1, runs)):
        with tempfile.TemporaryDirectory() as cache_dir:
            strategy = _Strategy(Path(cache_dir))
            cache = SyncCache()
            orchestrator = SyncOrchestrator(
                external_strategy=strategy, sync_service=svc, sync_cache=cache
            )
            first = await call(orchestrator, ID1)
            second = await call(orchestrator, ID2)
            third = await call(orchestrator, "aaaaaaaaaaaaaaaa", target_bytes + b"\n")
            first["provider_downloads"] = {
                "subdl": strategy.providers["subdl"].downloads,
                "subsource": strategy.providers["subsource"].downloads,
            }
            results.append({
                "first": first,
                "second": second,
                "third_different_content": third,
                "discovery_calls": strategy.gather_calls,
            })

    print(json.dumps({"runs": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    raise SystemExit(asyncio.run(_main(parser.parse_args().runs)))
