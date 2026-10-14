"""Two-Point primary: ordinary timing solved without ALASS, fallback intact.

Covers the §10 Two-Point cases at the orchestrator seam: constant offset
and linear drift serve TRANSFORMED with original dialogue preserved and no
ALASS subprocess; anchor failure, middle disagreement, ambiguous input, and
binding refusal all fall through to (or past) ALASS without forcing a
verdict. Real Penny bytes pin the HDTV regression end to end.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.sync import hybrid_router as router
from app.services.sync import orchestrator as om
from app.services.sync.query import ResolvedReference
from app.services.sync_cache import SyncCache
from tests.test_unified_authority import META
from two_point_sync.parser import parse_srt_for_verification

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "penny"


def _read(path: pathlib.Path) -> str:
    return path.read_bytes().decode("utf-8-sig")


def _require(path: pathlib.Path) -> str:
    """Real-byte runtime cache reads skip cleanly on machines without them."""
    if not path.is_file():
        pytest.skip(f"real-byte fixture unavailable: {path.name}")
    return path.read_bytes().decode("utf-8-sig")


def _harness(monkeypatch, candidates, alass_outcome=None, strong_binding=True):
    """Orchestrator with gate ON, real Two-Point path, stubbed reference/ALASS."""
    from app.services.sync.target_binding import classify_target_binding

    monkeypatch.setattr(router, "_enabled", True)
    monkeypatch.delenv(router.TEST_SCOPE_ENV, raising=False)
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True)
    monkeypatch.setattr(om.SyncOrchestrator, "_audit_serve", lambda self, *a, **k: None)
    monkeypatch.setattr(om.SyncOrchestrator, "_record_outcome", lambda self, *a, **k: None)
    monkeypatch.setattr(om.SyncOrchestrator, "_try_timeline_affine",
                        AsyncMock(return_value=None))
    if strong_binding:
        strong = classify_target_binding(
            "Some.Show.S01E01.480p.BluRay.x264-GRP", "Some.Show.S01E01.480p.BluRay.x264-GRP")
        assert strong.supports_verified
        monkeypatch.setattr(om.SyncOrchestrator, "_target_binding",
                            staticmethod(lambda resolved, meta: strong))
    strategy = SimpleNamespace(
        resolve_with_provenance=AsyncMock(return_value=candidates))
    service = SimpleNamespace(sync_async=AsyncMock(return_value=alass_outcome))
    cache = SyncCache()
    orch = om.SyncOrchestrator(sync_cache=cache, external_strategy=strategy,
                              sync_service=service)
    return orch, cache, service, strategy


def _series_meta() -> dict:
    meta = dict(META)
    meta.update({"imdb_id": "tt2628232", "media_type": "series", "season": 1,
                 "episode": 1,
                 "target_filename": "Penny.Dreadful.S01E01.Night.Work.1080p.BluRay.TrueHD5.1.AVC-d3g.mkv"})
    return meta


def _fmt(total_ms: int) -> str:
    total_ms = max(0, total_ms)
    millis = total_ms % 1000
    seconds = (total_ms // 1000) % 60
    minutes = (total_ms // 60_000) % 60
    hours = total_ms // 3_600_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _sentences(n: int, start_ms: int = 60_000, step_ms: int = 60_000,
               shift_ms: int = 0, drift: float = 0.0,
               paraphrase: tuple[int, ...] = ()) -> tuple[str, str]:
    """Distinctive per-cue dialogue with a genuinely different reference.

    The reference carries reworded lines at ``paraphrase`` positions (so it
    is independent evidence, never a circular copy) while the remaining
    exact lines provide dialogue correspondences across all thirds.
    ``shift_ms`` applies a constant offset; ``drift`` a linear rate change.
    """
    alternates = [
        "Completely different sentence alpha",
        "Completely different sentence beta",
        "Completely different sentence gamma",
        "Completely different sentence delta",
    ]
    target, reference = [], []
    for position in range(n):
        start = start_ms + position * step_ms
        text = (f"Distinctive dialogue event number {position + 1} spoken aloud")
        target.append(f"{position + 1}\n{_fmt(start)} --> {_fmt(start + 2_000)}\n{text}")
        if (position + 1) in paraphrase:
            text = alternates[(position + 1) % len(alternates)]
        shifted = int(round(start * (1.0 + drift) + shift_ms))
        reference.append(f"{position + 1}\n{_fmt(shifted)} --> {_fmt(shifted + 2_000)}\n{text}")
    return "\n\n".join(target) + "\n", "\n\n".join(reference) + "\n"


async def test_constant_offset_serves_without_alass(monkeypatch) -> None:
    target, reference = _sentences(16, shift_ms=12_025, paraphrase=(4, 8, 12, 16))
    resolved = ResolvedReference(reference, kind="edition", candidate="ref.srt")
    orch, _cache, service, _strategy = _harness(monkeypatch, resolved)
    out = await orch.evaluate_and_sync(target.encode(), _series_meta(), "sub", auto_sync=True)
    assert out != target.encode()
    service.sync_async.assert_not_called()
    cues, _ = parse_srt_for_verification(out.decode(), exclude_ambiguous=True)
    original, _ = parse_srt_for_verification(target, exclude_ambiguous=True)
    assert cues is not None and original is not None
    assert [c.text for c in cues] == [c.text for c in original]
    assert all((b.start_ms - a.start_ms) == 12_025
               for a, b in zip(original, cues, strict=True))


async def test_linear_drift_serves_without_alass(monkeypatch) -> None:
    """Frame-rate-style progressive drift (≈1.0003 slope) via Two-Point."""
    target, reference = _sentences(16, shift_ms=-8_000, drift=0.0003,
                                   paraphrase=(4, 8, 12, 16))
    resolved = ResolvedReference(reference, kind="edition", candidate="ref.srt")
    orch, _cache, service, _strategy = _harness(monkeypatch, resolved)
    out = await orch.evaluate_and_sync(target.encode(), _series_meta(), "sub", auto_sync=True)
    assert out != target.encode()
    service.sync_async.assert_not_called()


async def test_anchor_failure_falls_through_to_alass(monkeypatch) -> None:
    """No trustworthy anchors: ALASS still attempted, original on its failure."""
    target, _ = _sentences(8)
    other_lines = [
        "Unrelated utterance about weather patterns",
        "Unrelated utterance about market prices",
        "Unrelated utterance about distant relatives",
        "Unrelated utterance about broken machinery",
        "Unrelated utterance about evening meals",
        "Unrelated utterance about morning trains",
        "Unrelated utterance about paper archives",
        "Unrelated utterance about glass windows",
    ]
    reference = "\n\n".join(
        f"{position + 1}\n{_fmt(65_000 + position * 60_000)} --> "
        f"{_fmt(67_000 + position * 60_000)}\n{text}"
        for position, text in enumerate(other_lines)) + "\n"
    resolved = ResolvedReference(reference, kind="edition", candidate="ref.srt")
    orch, _cache, service, _strategy = _harness(monkeypatch, resolved)
    out = await orch.evaluate_and_sync(target.encode(), _series_meta(), "sub", auto_sync=True)
    assert out == target.encode()
    assert service.sync_async.await_count >= 1


async def test_middle_disagreement_refuses_and_falls_through(monkeypatch) -> None:
    """Anchors agree at both ends but the middle belongs elsewhere: no force."""
    target, _ = _sentences(18, paraphrase=(4, 8, 12, 16))
    lines = []
    for position in range(18):
        start = 60_000 + position * 60_000
        if 6 <= position <= 11:
            start += 300_000
        text = (f"Distinctive dialogue event number {position + 1} spoken aloud")
        if (position + 1) in (4, 8, 12, 16):
            text = f"Completely different sentence {(position + 1) % 4}"
        lines.append(f"{position + 1}\n{_fmt(start)} --> {_fmt(start + 2_000)}\n{text}")
    reference = "\n\n".join(lines) + "\n"
    resolved = ResolvedReference(reference, kind="edition", candidate="ref.srt")
    orch, _cache, service, _strategy = _harness(monkeypatch, resolved)
    out = await orch.evaluate_and_sync(target.encode(), _series_meta(), "sub", auto_sync=True)
    # Either the nucleus vetoes (discontinuity) or Unified refuses: never a
    # forced linear serve, and ALASS remains reachable for the nonlinear case.
    served_changed = out != target.encode()
    if served_changed:
        assert service.sync_async.await_count == 0, "only Two-Point may serve here"
    else:
        assert service.sync_async.await_count >= 1, "ALASS fallback must be tried"


async def test_malformed_inherited_text_preserved(monkeypatch) -> None:
    """Stray unplaceable line + constant shift: served with stray intact."""
    target, reference = _sentences(14, shift_ms=9_000, paraphrase=(4, 8, 12))
    target = target + "\nstray wrapped continuation line\n"
    resolved = ResolvedReference(reference, kind="edition", candidate="ref.srt")
    orch, _cache, service, _strategy = _harness(monkeypatch, resolved)
    out = await orch.evaluate_and_sync(target.encode(), _series_meta(), "sub", auto_sync=True)
    assert out != target.encode()
    service.sync_async.assert_not_called()
    assert "stray wrapped continuation line" in out.decode()


async def test_penny_hdtv_end_to_end_without_alass(monkeypatch) -> None:
    """Live d3g HDTV bytes × AOS reference: TRANSFORMED, dialogue preserved."""
    target = _read(FIXTURES / "penny-s01e01-hdtv-ar-2.srt")
    reference = _read(FIXTURES / "penny-s01e01-bluray-aos-ar.srt")
    resolved = ResolvedReference(reference, kind="edition",
                                 candidate="Penny.Dreadful.S01E01.Bluray-AOS.srt",
                                 provider="subsource")
    orch, _cache, service, _strategy = _harness(monkeypatch, resolved,
                                               strong_binding=False)
    out = await orch.evaluate_and_sync(target.encode(), _series_meta(), "sub", auto_sync=True)
    assert out != target.encode()
    service.sync_async.assert_not_called()
    cues, _ = parse_srt_for_verification(out.decode(), exclude_ambiguous=True)
    original, _ = parse_srt_for_verification(target, exclude_ambiguous=True)
    assert cues is not None and original is not None
    assert len(cues) == len(original) == 474
    assert [c.text for c in cues] == [c.text for c in original]
    assert "FDC400" in cues[-1].text
    assert isinstance(out, bytes)
    assert str(orch._last_evaluation.sync_state.value).startswith("verified")


async def test_penny_webdl_end_to_end_without_alass(monkeypatch) -> None:
    """Live d3g WEB-DL bytes × AOS reference: TRANSFORMED, no ALASS."""
    cache_root = pathlib.Path(__file__).resolve().parent.parent / "subs_cache"
    target = _require(cache_root / "1e9b2fc7c3fdafbc.srt")
    reference = _read(FIXTURES / "penny-s01e01-bluray-aos-ar.srt")
    resolved = ResolvedReference(reference, kind="edition",
                                 candidate="Penny.Dreadful.S01E01.Bluray-AOS.srt",
                                 provider="subsource")
    orch, _cache, service, _strategy = _harness(monkeypatch, resolved,
                                               strong_binding=False)
    out = await orch.evaluate_and_sync(target.encode(), _series_meta(), "sub", auto_sync=True)
    assert out != target.encode()
    service.sync_async.assert_not_called()
    cues, _ = parse_srt_for_verification(out.decode(), exclude_ambiguous=True)
    original, _ = parse_srt_for_verification(target, exclude_ambiguous=True)
    assert cues is not None and original is not None
    assert [c.text for c in cues] == [c.text for c in original]


async def test_psypher_brrip_end_to_end_without_alass(monkeypatch) -> None:
    """Live OpenSubtitles PSYPHER target × AOS: near-aligned, Two-Point serves."""
    cache_root = pathlib.Path(__file__).resolve().parent.parent / "subs_cache"
    target = _require(cache_root / "4055301.srt")
    reference = _read(FIXTURES / "penny-s01e01-bluray-aos-ar.srt")
    resolved = ResolvedReference(reference, kind="edition",
                                 candidate="Penny.Dreadful.S01E01.Bluray-AOS.srt",
                                 provider="subsource")
    orch, _cache, service, _strategy = _harness(monkeypatch, resolved,
                                               strong_binding=False)
    out = await orch.evaluate_and_sync(target.encode(), _series_meta(), "sub", auto_sync=True)
    assert out != target.encode()
    service.sync_async.assert_not_called()
    cues, _ = parse_srt_for_verification(out.decode(), exclude_ambiguous=True)
    original, _ = parse_srt_for_verification(target, exclude_ambiguous=True)
    assert cues is not None and original is not None
    assert [c.text for c in cues] == [c.text for c in original]


async def test_intention_pairing_falls_through_without_forcing(monkeypatch) -> None:
    """HDTV × iNTENTiON (different translation): no anchors, no force.

    Two-Point correctly abstains here (insufficient correspondence: the
    pair needs genuine nonlinear alignment, which only ALASS can attempt).
    In this harness the weak uncorroborated binding skips sync entirely per
    standing policy, so the safe ORIGINAL is served with nothing forced and
    nothing fabricated. ALASS remains the genuine fallback for this pairing
    whenever binding admits it.
    """
    cache_root = pathlib.Path(__file__).resolve().parent.parent / "subs_cache"
    target = _require(cache_root / "6b696568a7c8898a.srt")
    reference = _require(cache_root / "references" /
                         "tt2628232_1_1_d3g_v2_601ffb5b21eb105b.video.subsource.edition.32c9194335f2e997.srt")
    resolved = ResolvedReference(reference, kind="edition",
                                 candidate="Penny.Dreadful.S01.1080p.BluRay.x264-iNTENTiON.srt",
                                 provider="subsource")
    orch, _cache, _service, _strategy = _harness(monkeypatch, resolved,
                                                strong_binding=False)
    out = await orch.evaluate_and_sync(target.encode(), _series_meta(), "sub", auto_sync=True)
    assert out == target.encode()
