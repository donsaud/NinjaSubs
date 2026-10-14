"""Large-offset bulk correction: constant ±96 s, drift, two-regime files.

A subtitle that differs by a large constant offset must synchronize like any
other offset; a two-regime file (differently-cut recap + bulk episode) must
serve its verifiable bulk without forcing the recap. Genuine cut/episode
mismatches must stay rejected. All dialogue here is synthetic; real-byte
coverage uses gitignored runtime cache with skip guards.
"""

from __future__ import annotations

import logging
import pathlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.services.sync import hybrid_router as router
from app.services.sync import orchestrator as om
from app.services.sync.query import ResolvedReference
from app.services.sync_cache import SyncCache
from tests.test_two_point_primary import _sentences
from two_point_sync.parser import parse_srt_for_verification

D3G_TARGET = ("Dexter.s8e02.Every.silver.lining1080p.BluRay.10Bit.Dts-HDMa5.1.HEVC-d3g")
D3G_CANDIDATE = "Dexter.S08E02.1080p.BluRay.x264-GRP.srt"


def _require(path: pathlib.Path) -> str:
    import pytest

    if not path.is_file():
        pytest.skip(f"real-byte fixture unavailable: {path.name}")
    return path.read_bytes().decode("utf-8-sig")


def _harness(monkeypatch, resolved, alass_outcome=None):
    """Gate ON, strong binding stub, affine off, stubbed ALASS."""
    from app.services.sync.target_binding import classify_target_binding

    monkeypatch.setattr(router, "_enabled", True)
    monkeypatch.delenv(router.TEST_SCOPE_ENV, raising=False)
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True)
    monkeypatch.setattr(om.SyncOrchestrator, "_audit_serve", lambda self, *a, **k: None)
    monkeypatch.setattr(om.SyncOrchestrator, "_record_outcome", lambda self, *a, **k: None)
    monkeypatch.setattr(om.SyncOrchestrator, "_try_timeline_affine",
                        AsyncMock(return_value=None))
    strong = classify_target_binding(
        "Some.Show.S01E01.480p.BluRay.x264-GRP", "Some.Show.S01E01.480p.BluRay.x264-GRP")
    assert strong.supports_verified
    monkeypatch.setattr(om.SyncOrchestrator, "_target_binding",
                        staticmethod(lambda resolved, meta: strong))
    strategy = SimpleNamespace(
        resolve_with_provenance=AsyncMock(return_value=resolved))
    service = SimpleNamespace(sync_async=AsyncMock(return_value=alass_outcome))
    cache = SyncCache()
    return (om.SyncOrchestrator(sync_cache=cache, external_strategy=strategy,
                               sync_service=service), service)


def _meta() -> dict:
    return {"imdb_id": "tt0773262", "media_type": "series", "season": 8,
            "episode": 2, "lang": "ara", "target_filename": D3G_TARGET,
            "title": "Dexter", "video_size": 5712655000}


def _resolved(reference: str) -> ResolvedReference:
    return ResolvedReference(reference, kind="edition", candidate=D3G_CANDIDATE,
                             provider="subdl")


async def test_constant_offset_plus96_serves(monkeypatch) -> None:
    target, reference = _sentences(24, shift_ms=96_390, paraphrase=(4, 8, 12, 16, 20, 24))
    orch, service = _harness(monkeypatch, _resolved(reference))
    out = await orch.evaluate_and_sync(target.encode(), _meta(), "sub", auto_sync=True)
    assert out != target.encode()
    service.sync_async.assert_not_called()
    cues, _ = parse_srt_for_verification(out.decode(), exclude_ambiguous=True)
    original, _ = parse_srt_for_verification(target, exclude_ambiguous=True)
    assert cues is not None and original is not None
    assert [c.text for c in cues] == [c.text for c in original]


async def test_constant_offset_minus96_serves(monkeypatch) -> None:
    target, reference = _sentences(24, start_ms=200_000, shift_ms=-96_390,
                                   paraphrase=(4, 8, 12, 16, 20, 24))
    orch, service = _harness(monkeypatch, _resolved(reference))
    out = await orch.evaluate_and_sync(target.encode(), _meta(), "sub", auto_sync=True)
    assert out != target.encode()
    service.sync_async.assert_not_called()


async def test_large_offset_with_drift_serves(monkeypatch) -> None:
    target, reference = _sentences(24, shift_ms=96_000, drift=0.0003,
                                   paraphrase=(4, 8, 12, 16, 20, 24))
    orch, service = _harness(monkeypatch, _resolved(reference))
    out = await orch.evaluate_and_sync(target.encode(), _meta(), "sub", auto_sync=True)
    assert out != target.encode()
    service.sync_async.assert_not_called()


def _two_regime_pair() -> tuple[str, str]:
    """Bulk (+9 s) interrupted by a contradictory block (+96 s).

    The block's exact lines are two tokens: strong enough for anchor
    candidacy (so the discontinuity veto fires) but excluded from timing
    measurement (short utterances need neighbour context), mirroring the
    real S08E02 recap evidence. One 5-token repeated line sits at
    order-violating positions to prove the chronological filter.
    First dialogue agrees within the normal window, so no large-offset
    machinery is involved: this isolates veto + proposal + measurement.
    """
    alternates = [
        "Completely different sentence alpha",
        "Completely different sentence beta",
        "Completely different sentence gamma",
        "Completely different sentence delta",
    ]
    generics = ["After you", "No way", "Stay back"]

    def cue(number: int, start: int, text: str) -> str:
        return f"{number}\n{_fmt(start)} --> {_fmt(start + 2_000)}\n{text}"

    def bulk_text(number: int) -> str:
        base = f"Episode dialogue event number {number} spoken aloud"
        return base if number % 4 else alternates[number % 4]

    t_blocks, r_blocks = [], []
    # Bulk part 1: cues 1-24 at realistic 18 s spacing, reference 9 s earlier.
    # Dense dialogue forms first-dialogue clusters, as real releases do.
    for position in range(24):
        number = position + 1
        start = 60_000 + position * 18_000
        text = f"Episode dialogue event number {number} spoken aloud"
        t_blocks.append(cue(number, start, text))
        r_blocks.append(cue(number, start - 9_000, bulk_text(number)))
    # Contradictory block: cues 25-34; exact generics at 25/27/29 sit 96 s
    # later in target than in reference; the rest is paraphrased (no match).
    for position in range(10):
        number = 25 + position
        start = 700_000 + position * 9_000
        if position in (0, 2, 4):
            text = generics[position // 2]
            t_blocks.append(cue(number, start, text))
            r_blocks.append(cue(number, start - 96_000, text))
        else:
            t_blocks.append(cue(number, start, f"Inserted aside number {number} aside"))
            r_blocks.append(cue(number, start - 9_000, f"Reworded aside {number} variant"))
    # Bulk part 2: cues 35-58 from the middle into the ending third,
    # including the wrong-instance repeated line.
    for position in range(24):
        number = 35 + position
        start = 800_000 + position * 18_000
        if number == 55:
            text = "What are you doing here"
        else:
            text = f"Episode dialogue event number {number} spoken aloud"
        t_blocks.append(cue(number, start, text))
        if number == 55:
            # Same line, but placed early in the reference (wrong instance).
            r_blocks.append(cue(number, 63_000, text))
        else:
            r_blocks.append(cue(number, start - 9_000, bulk_text(number)))
    return "\n\n".join(t_blocks) + "\n", "\n\n".join(r_blocks) + "\n"


def _fmt(total_ms: int) -> str:
    total_ms = max(0, total_ms)
    millis = total_ms % 1000
    seconds = (total_ms // 1000) % 60
    minutes = (total_ms // 60_000) % 60
    hours = total_ms // 3_600_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


async def test_two_regime_bulk_fix_serves(monkeypatch) -> None:
    """Recap regime vetoes the global claim; verifiable bulk still serves."""
    target, reference = _two_regime_pair()
    orch, service = _harness(monkeypatch, _resolved(reference))
    out = await orch.evaluate_and_sync(target.encode(), _meta(), "sub", auto_sync=True)
    assert out != target.encode()
    service.sync_async.assert_not_called()
    cues, _ = parse_srt_for_verification(out.decode(), exclude_ambiguous=True)
    original, _ = parse_srt_for_verification(target, exclude_ambiguous=True)
    assert cues is not None and original is not None
    assert [c.text for c in cues] == [c.text for c in original]
    evaluation = orch._last_evaluation
    assert evaluation is not None
    assert str(evaluation.sync_state.value).startswith("verified")
    assert any("primary_regime_only" in reason for reason in evaluation.reasons)


async def test_different_cut_stays_rejected(monkeypatch) -> None:
    """Second half shifted +600 s: no global claim survives verification."""
    target, _ = _sentences(18, paraphrase=(4, 8, 12, 16))
    lines = []
    for position in range(18):
        start = 60_000 + position * 60_000
        if position >= 9:
            start += 600_000
        text = (f"Distinctive dialogue event number {position + 1} spoken aloud")
        if (position + 1) in (4, 8, 12, 16):
            text = f"Completely different sentence {(position + 1) % 4}"
        lines.append(f"{position + 1}\n{_fmt(start)} --> {_fmt(start + 2_000)}\n{text}")
    reference = "\n\n".join(lines) + "\n"
    orch, service = _harness(monkeypatch, _resolved(reference))
    out = await orch.evaluate_and_sync(target.encode(), _meta(), "sub", auto_sync=True)
    assert out == target.encode()


async def test_wrong_episode_stays_rejected(monkeypatch) -> None:
    """Disjoint vocabulary: insufficient evidence, never a forced verdict."""
    target, _ = _sentences(10)
    other = "\n\n".join(
        f"{position + 1}\n{_fmt(60_000 + position * 60_000)} --> "
        f"{_fmt(62_000 + position * 60_000)}\nUnrelated episode line {position + 1} entirely"
        for position in range(10)) + "\n"
    orch, service = _harness(monkeypatch, _resolved(other))
    out = await orch.evaluate_and_sync(target.encode(), _meta(), "sub", auto_sync=True)
    assert out == target.encode()


def test_warped_output_with_dropped_cue_abstains() -> None:
    """Nonlinear ALASS-style warp plus a dropped cue: reconstruction refuses."""
    from app.services.sync.candidate_reconstruction import (
        reconstruct_with_original_dialogue,
    )
    from two_point_sync.parser import (
        TIMING_LINE_RE,
        _split_blocks,
        format_timestamp,
        parse_timestamp,
    )

    target, _ = _sentences(20, paraphrase=(4, 8, 12, 16, 20))
    blocks = _split_blocks(target)
    warped = []
    for position, block in enumerate(blocks):
        lines = block.split("\n")
        timing = TIMING_LINE_RE.search(lines[1])
        assert timing is not None
        shift = 96_000 if position < 5 else 9_000
        start = parse_timestamp(timing.group("start")) + shift
        end = parse_timestamp(timing.group("end")) + shift
        lines[1] = TIMING_LINE_RE.sub(
            f"{format_timestamp(start)} --> {format_timestamp(end)}", lines[1], count=1)
        warped.append("\n".join(lines))
    # Plus a dropped cue: counts and constancy both fail.
    assert reconstruct_with_original_dialogue(
        target, "\n\n".join(warped[:-1]) + "\n") is None


def test_order_filter_drops_wrong_instance() -> None:
    """LIS keeps the chronological run; the inverted pairing is dropped."""
    from two_point_sync.anchors import _longest_chronological_run

    records = [(1000, 1000, 1000, (1,), (1,)),
               (2000, 2000, 2000, (2,), (2,)),
               (3000, 500, 3000, (3,), (3,)),
               (4000, 4000, 4000, (4,), (4,))]
    kept = _longest_chronological_run(records)
    assert [r[0] for r in kept] == [1000, 2000, 4000]
    assert _longest_chronological_run([]) == []
    assert len(_longest_chronological_run(records[:2])) == 2


async def test_detection_timeout_falls_back_to_alass(monkeypatch, caplog) -> None:
    """A stalled anchor search abstains fast; fallback proceeds, once only."""
    import time

    import two_point_sync.anchors as anchors_module

    target, reference = _sentences(16, shift_ms=5_000, paraphrase=(4, 8, 12))
    original = anchors_module.detect_anchors
    monkeypatch.setattr(om, "TWO_POINT_DETECTION_TIMEOUT_SECONDS", 0.05)

    def _slow(*args, **kwargs):
        time.sleep(0.3)
        return original(*args, **kwargs)

    monkeypatch.setattr(anchors_module, "detect_anchors", _slow)
    orch, service = _harness(monkeypatch, _resolved(reference))
    with caplog.at_level(logging.INFO, logger="app.services.sync.orchestrator"):
        await orch.evaluate_and_sync(target.encode(), _meta(), "sub", auto_sync=True)
    assert any("error=TimeoutError" in r.message for r in caplog.records)
    # Two-Point abstained; exactly one ALASS attempt followed (no duplicate
    # work, no hang). A later tail may still serve; the timeout itself must
    # only ever cost one fallback attempt.
    assert service.sync_async.await_count == 1


async def test_s08e02_real_bytes_serve_bulk(monkeypatch, caplog) -> None:
    """Live d3g bytes: large-offset gate engages, bulk still serves TRANSFORMED."""
    root = pathlib.Path(__file__).resolve().parent.parent / "subs_cache"
    target = _require(root / "2fc09c4f89a18d3b.srt")
    reference = _require(
        root / "references" / ("tt0773262_8_2_d3g_v2_9ea29ec90715209f.video.subdl.edition."
                               "bfdedc48d3e9b3d2.srt"))
    from app.services.sync.query import ResolvedReference as _Resolved

    resolved = _Resolved(reference, kind="edition",
                         candidate="Dexter.s8.BluRay.10Bit.1080p.DD5.1.H265-d3g.srt",
                         provider="subdl")
    orch, service = _harness(monkeypatch, resolved)
    with caplog.at_level(logging.INFO, logger="app.services.sync.orchestrator"):
        out = await orch.evaluate_and_sync(target.encode(), _meta(), "sub-s08e02",
                                           auto_sync=True)
    assert out != target.encode()
    service.sync_async.assert_not_called()
    cues, _ = parse_srt_for_verification(out.decode(), exclude_ambiguous=True)
    original, _ = parse_srt_for_verification(target, exclude_ambiguous=True)
    assert cues is not None and original is not None
    assert len(cues) == len(original) == 894
    assert [c.text for c in cues] == [c.text for c in original]
    assert not any("large-offset evidence accepted" in r.message for r in caplog.records)
    assert any("primary_regime_only" in r for r in orch._last_evaluation.reasons)
    # Final HTTP delivery authorizes the served bytes.
    from app import main
    from app.services.sync.delivery import DeliveryPayload

    assert isinstance(out, DeliveryPayload)
    assert out.authorization.may_serve is True
    with caplog.at_level(logging.INFO, logger="app.services.sync.delivery"):
        response = main._build_subtitle_response(
            out, "Dexter.s8e02.Every.silver.lining1080p.BluRay.10Bit.Dts-HDMa5.1.HEVC-d3g",
            "srt", auto_sync=True)
    assert response.status_code == 200
    records = [r.message for r in caplog.records if "[unified-delivery]" in r.message]
    assert any("decision=TRANSFORMED" in message and "final_integrity=True" in message
               for message in records)


def test_s08e02_global_96s_refused() -> None:
    """The user's literal +96 s global claim must not verify (bulk vetoes it)."""
    root = pathlib.Path(__file__).resolve().parent.parent / "subs_cache"
    target = _require(root / "2fc09c4f89a18d3b.srt")
    reference = _require(
        root / "references" / ("tt0773262_8_2_d3g_v2_9ea29ec90715209f.video.subdl.edition."
                               "bfdedc48d3e9b3d2.srt"))
    from app.services.sync.candidate_reconstruction import apply_linear_timings
    from app.services.sync.unified_verification import verify_candidate

    candidate = apply_linear_timings(target, 1.0, 96_390)
    assert candidate is not None
    verdict = verify_candidate(target, reference, candidate,
                               candidate_algorithm="two_point")
    assert verdict.proposed_verdict == "REJECTED"
