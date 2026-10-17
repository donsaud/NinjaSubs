"""Mad Men discovery-repair benchmark with the real cached bytes and ALASS.

Modes exercise the repairs independently:

* ``tier1_timeout``: Tier 1 returns only BTN; Tier 2 times out. P0 must keep
  the completed Tier 1 candidate. No better candidate exists, so the quality
  fallback must return the original rather than serve the weak correction.
* ``wadu_exact``: Tier 1 contains the exact WADU pack and BTN. P1 must skip
  the unnecessary season catalog; WADU is planned, downloaded, and verified.
* ``wadu_hint``: the provider search fails entirely. P2 recovers WADU from the
  same-video Display candidate hint; it is still downloaded and verified.

Run in the image with ``ITW_APP_ROOT=/work`` and the repository mounted at
``/work`` plus ``subs_cache`` at ``/app/cache``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if ROOT.name == "tools":
    ROOT = ROOT.parent
APP_ROOT = Path(os.environ.get("ITW_APP_ROOT", str(ROOT))).resolve()
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

VIDEO = "Mad.Men.S01E01.Smoke.Gets.in.Your.Eyes.REPACK.2160p.HMAX.WEB-DL.DDP5.1.DV.HDR.H.265-WADU.mkv"
TARGETS = {
    "DBO": ("3d449ce4ce4d0d28", "Mad.Men.S01.720p.BluRay.DTS.x264-DBO.srt"),
    "SA89": ("a437fb8d5539353a", "Mad.Men.S01.1080p.BluRay.DD5.1.x264-SA89.srt"),
}
WADU_NAME = "Mad.Men.S01.REPACK.2160p.HMAX.WEB-DL.DDP5.1.DV.HDR.H.265-WADU.srt"
BTN_NAME = "Mad.Men.S01.720p.WEB-DL.AAC2.0.H264-BTN.srt"
REFS = {
    "WADU": (WADU_NAME, "/app/cache/ba4ceee32aab31d3.srt"),
    "BTN": (BTN_NAME, "/app/cache/references/tt0804503_1_1_WADU_v2_e3f3e720de3ca96d.video.subdl.edition.71f906776825a9c2.srt"),
    "SA89REF": ("Mad.Men.S01.1080p.BluRay.DD5.1.x264-SA89.srt",
                "/app/cache/references/tt0804503_1_1_SA89_v2_fa0011efbe0acc3e.video.subdl.edition.89f334377eb99168.srt"),
}


async def _main(runs: int, modes: list[str]) -> int:
    import app.services.sync.orchestrator as om  # noqa: F401  (import-order guard)
    from app.models import SubtitleRelease
    from app.services.release_scoring_integration import record
    from app.services.sync import discovery_hints
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
    ref_text = {key: Path(path).read_text("utf-8") for key, (_, path) in REFS.items()}

    def release(key: str):
        name = REFS[key][0]
        rel = SubtitleRelease(
            release_name=name,
            download_url=f"https://invalid/{key}",
            provider="subsource",
            lang="ara" if key == "WADU" else "eng",
        )
        evidence = record(rel)
        evidence["metadata"].update(
            imdb_id="tt0804503", imdb_verified=True, source="webdl",
        )
        rel.release_evidence = [evidence]
        return rel

    class _Prov:
        name = "subsource"

        def __init__(self, mode: str) -> None:
            self.mode = mode
            self.search_calls: list[int | None] = []

        def is_breaker_open(self) -> bool:
            return False

        async def download_archive(self, release, api_key=None, **kwargs):
            return b""

        async def search_subtitles(self, *, episode=None, **kwargs):
            self.search_calls.append(episode)
            if self.mode in {"wadu_hint", "wadu_cached_btn"}:
                raise TimeoutError("provider fan-out unavailable")
            if episode is not None:
                keys = ["WADU", "BTN"] if self.mode == "wadu_exact" else ["BTN"]
                return [release(key) for key in keys]
            await asyncio.sleep(5)
            return []

    class _Strategy(ExternalExactStrategy):
        def __init__(self, root: Path, mode: str) -> None:
            super().__init__(
                cache=ReferenceDiskCache(root=root, min_bytes=1), timeout=0.25
            )
            self.provider = _Prov(mode)
            self._subsource = self.provider
            self._subdl = None
            self._opensubtitles = None
            self.downloaded: list[str] = []

        async def _download_candidate(self, best, provider, query):
            name = str(getattr(best, "release_name", ""))
            provider.downloads = getattr(provider, "downloads", 0) + 1
            self.downloaded.append(name)
            if name == WADU_NAME:
                key = "WADU"
            elif name == BTN_NAME:
                key = "BTN"
            else:
                key = "SA89REF"
            return ResolvedReference(
                ref_text[key], kind="edition", candidate=name,
                member_season=1, member_episode=1,
            )

    def meta(sub_id: str, release_name: str) -> dict:
        return {
            "imdb_id": "tt0804503", "media_type": "series",
            "season": 1, "episode": 1, "target_filename": VIDEO,
            "video_size": "8113838532", "lang": "ara", "year": 2007,
            "release_name": release_name, "sub_id": sub_id,
            "subsource_key": "benchmark-auth",
        }

    captured: list[SyncTiming] = []
    orig = SyncTiming.log

    def capture(self, result, total_ms):
        self.total_ms = total_ms
        captured.append(self)
        return orig(self, result, total_ms)

    SyncTiming.log = capture

    evidence_lines: list[str] = []
    handler = logging.Handler()
    handler.setLevel(logging.INFO)

    def emit(record):
        message = record.getMessage()
        if any(token in message for token in (
            "target_binding", "verification=", "binding=", "selected reference",
        )):
            evidence_lines.append(message)

    handler.emit = emit  # type: ignore[method-assign]
    sync_logger = logging.getLogger("app.services.sync.orchestrator")
    sync_logger.setLevel(logging.INFO)
    sync_logger.addHandler(handler)

    results = []
    try:
        for mode in modes:
            for _ in range(max(1, runs)):
                for label, (sub_id, release_name) in TARGETS.items():
                    discovery_hints.clear()
                    target = Path(f"/app/cache/{sub_id}.srt").read_bytes()
                    if mode == "wadu_hint":
                        key = discovery_hints.video_scope_key(
                            imdb_id="tt0804503", media_type="series", season=1,
                            episode=1, target_filename=VIDEO,
                            video_size=8113838532,
                        )
                        auth = discovery_hints.authorization_digest(
                            subsource_key="benchmark-auth"
                        )
                        discovery_hints.register(key, auth, [release("WADU")])
                    with tempfile.TemporaryDirectory() as cache_dir:
                        strategy = _Strategy(Path(cache_dir), mode)
                        orchestrator = SyncOrchestrator(
                            external_strategy=strategy,
                            sync_service=svc,
                            sync_cache=SyncCache(),
                        )
                        if mode == "wadu_cached_btn":
                            key = discovery_hints.video_scope_key(
                                imdb_id="tt0804503", media_type="series", season=1,
                                episode=1, target_filename=VIDEO,
                                video_size=8113838532,
                            )
                            auth = discovery_hints.authorization_digest(
                                subsource_key="benchmark-auth"
                            )
                            discovery_hints.register(key, auth, [release("WADU")])
                            cached_query = orchestrator._build_query(
                                meta(sub_id, release_name), sub_id, target.decode("utf-8")
                            )
                            strategy.cache.set_video_scoped(
                                cached_query, "subsource", ref_text["BTN"],
                                kind="edition", candidate=BTN_NAME,
                                member_season=1, member_episode=1,
                                score=1025.0, score_without_hash=215.0,
                            )
                        evidence_lines.clear()
                        captured.clear()
                        started = time.monotonic()
                        run_meta = meta(sub_id, release_name)
                        served = await orchestrator.evaluate_and_sync(
                            target, run_meta, sub_id, auto_sync=True
                        )
                        elapsed = (time.monotonic() - started) * 1000
                        timing = captured[-1] if captured else SyncTiming()
                        binding = None
                        verification = None
                        if strategy.downloaded and served != target:
                            served_name = strategy.downloaded[-1]
                            if served_name == WADU_NAME:
                                ref_key = "WADU"
                            elif served_name == BTN_NAME:
                                ref_key = "BTN"
                            else:
                                ref_key = "SA89REF"
                            resolved = ResolvedReference(
                                ref_text[ref_key], kind="edition", candidate=served_name,
                                member_season=1, member_episode=1,
                            )
                            try:
                                assessment = orchestrator._target_binding(resolved, run_meta)
                                binding = assessment.state.value
                            except Exception:  # noqa: BLE001 - benchmark evidence only
                                binding = None
                            evaluation = getattr(orchestrator, "_last_evaluation", None)
                            if evaluation is not None:
                                verification = evaluation.verification.value
                        content_keys = [
                            key for key in getattr(orchestrator._sync_cache, "_local", {})
                            if "large_offset_content_v1" in key
                        ]
                        results.append({
                            "mode": mode,
                            "case": label,
                            "elapsed_ms": round(elapsed, 1),
                            "internal_ms": round(float(getattr(timing, "total_ms", elapsed)), 1),
                            "downloads": int(timing.references_downloaded),
                            "alass_runs": int(timing.alass_processes_launched),
                            "strategy": timing.winning_sync_strategy,
                            "reference_kind": timing.reference_kind,
                            "binding": binding,
                            "verification": verification,
                            "downloaded": strategy.downloaded,
                            "served_sha": hashlib.sha256(served).hexdigest()[:16],
                            "served_original": served == target,
                            "content_artifacts": len(content_keys),
                            "quality_triggered": bool(getattr(timing, "quality_fallback_triggered", False)),
                            "quality_attempts": int(getattr(timing, "quality_fallback_attempts", 0)),
                            "quality_served": bool(getattr(timing, "quality_fallback_served", False)),
                            "quality_exhausted": bool(getattr(timing, "quality_fallback_exhausted", False)),
                            "first_p95": getattr(timing, "quality_fallback_first_p95_ms", None),
                            "first_binding": getattr(timing, "quality_fallback_first_binding", None),
                            "search_calls": strategy.provider.search_calls,
                            "evidence": evidence_lines[-6:],
                        })
    finally:
        sync_logger.removeHandler(handler)
        SyncTiming.log = orig
    print(json.dumps({"runs": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=1)
    p.add_argument(
        "--modes",
        default="tier1_timeout,wadu_exact,wadu_hint,wadu_cached_btn",
        help="comma-separated: tier1_timeout,wadu_exact,wadu_hint,wadu_cached_btn",
    )
    args = p.parse_args()
    raise SystemExit(asyncio.run(_main(args.runs, [m.strip() for m in args.modes.split(",") if m.strip()])))
