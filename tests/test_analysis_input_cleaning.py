"""Temporary analysis inputs vs pristine delivered content (§3.6).

Cleaning (advertisements, translator credits, HI descriptions) is an
internal technique for extracting better timing evidence. It must never
remove content from the delivered subtitle: the original stays immutable,
alignment engines see temporary copies, verified timings are transferred
onto the pristine original, and final integrity is checked. Timing-v3
remains the authoritative verifier; these tests pin the pipeline.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.services.sync import hybrid_router as router
from app.services.sync import orchestrator as om
from app.services.sync.candidate_reconstruction import (
    apply_linear_timings,
    reconstruct_with_original_dialogue,
)
from app.services.sync.query import ResolvedReference
from app.services.sync_cache import SyncCache
from app.services.sync_service import prepare_alass_target, strip_sdh
from app.utils.cleaners import strip_advertisements, strip_intro_credits
from two_point_sync.anchors import detect_anchors
from two_point_sync.parser import parse_srt_for_verification
from two_point_sync.sanitizer import classify_all


def _fmt(total_ms: int) -> str:
    total_ms = max(0, total_ms)
    millis = total_ms % 1000
    seconds = (total_ms // 1000) % 60
    minutes = (total_ms // 60_000) % 60
    hours = total_ms // 3_600_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _cue(number: int, start_ms: int, text: str, duration: int = 2_000) -> str:
    return f"{number}\n{_fmt(start_ms)} --> {_fmt(start_ms + duration)}\n{text}"


def _dialogue(number: int, start_ms: int) -> str:
    return _cue(number, start_ms, f"Distinctive dialogue event number {number} spoken aloud")


def _parse(text: str):
    cues, _ = parse_srt_for_verification(text, exclude_ambiguous=True)
    assert cues is not None
    return cues


def _eligibility(text: str) -> dict[str, bool]:
    cues = _parse(text)
    return {cue.text: score.eligible
            for cue, score in zip(cues, classify_all(cues), strict=True)}


# --------------------------------------------------------------------------- #
# 1. Translator credit before the first dialogue
# --------------------------------------------------------------------------- #

def test_credit_before_first_dialogue_excluded_from_anchors_but_kept():
    target = "\n\n".join([
        _cue(1, 2_000, "ترجمة: فريق العمل"),
        *(_dialogue(n, 60_000 + (n - 2) * 60_000) for n in range(2, 18)),
    ]) + "\n"
    assert _eligibility(target)["ترجمة: فريق العمل"] is False
    cleaned = strip_intro_credits(target, first_speech_ms=60_000)
    assert "ترجمة" not in cleaned
    assert "ترجمة: فريق العمل" in target  # original immutable
    assert len(_parse(cleaned)) == len(_parse(target)) - 1


# --------------------------------------------------------------------------- #
# 2. Advertisement banner near the ending
# --------------------------------------------------------------------------- #

def test_trailing_ad_banner_removed_from_analysis_copy_only():
    body = [_dialogue(n, 60_000 + (n - 1) * 60_000) for n in range(1, 17)]
    # Tail ad sits inside the trailing window (last 90 s of ~1080 s).
    body.append(_cue(17, 1_020_000, "www.example-subs.com"))
    target = "\n\n".join(body) + "\n"
    cleaned = strip_advertisements(target, keep_translator_credits=True)
    assert "example-subs" not in cleaned
    assert "www.example-subs.com" in target
    assert len(_parse(cleaned)) == 16


# --------------------------------------------------------------------------- #
# 3. Telegram/website promotion among legitimate dialogue cues
# --------------------------------------------------------------------------- #

def test_mid_file_promo_kept_by_windowed_cleaner_but_ineligible_as_anchor():
    body = [_dialogue(n, 60_000 + (n - 1) * 60_000) for n in range(1, 17)]
    body.insert(8, _cue(99, 540_000, "تابعنا على تيليجرام @channel"))
    target = "\n\n".join(body) + "\n"
    # The cue-windowed cleaner deliberately leaves mid-file cues alone ...
    assert "@channel" in strip_advertisements(target, keep_translator_credits=True)
    # ... while anchor selection still refuses them wherever they appear.
    assert _eligibility(target)["تابعنا على تيليجرام @channel"] is False
    target_cues = _parse(target)
    reference_cues = _parse(
        "\n\n".join(_dialogue(n, 60_000 + (n - 1) * 60_000 + 9_000)
                     for n in range(1, 17)) + "\n")
    result = detect_anchors(target_cues, reference_cues)
    assert result.accepted  # dialogue evidence suffices without the promo


# --------------------------------------------------------------------------- #
# 4. HI-only sound descriptions
# --------------------------------------------------------------------------- #

def test_hi_only_cues_excluded_from_anchors_and_reference_copy():
    for hi_text in ("[door slams]", "(dramatic music)"):
        target = "\n\n".join([
            _cue(1, 60_000, hi_text),
            *(_dialogue(n, 60_000 + (n - 1) * 60_000) for n in range(2, 18)),
        ]) + "\n"
        assert _eligibility(target)[hi_text] is False, hi_text
    # A music-marked cue that still carries words is penalised, not deleted:
    # the wording may be needed for correspondence, so it stays eligible.
    target = "\n\n".join([
        _cue(1, 60_000, "♪ background music ♪"),
        *(_dialogue(n, 60_000 + (n - 1) * 60_000) for n in range(2, 18)),
    ]) + "\n"
    cues = _parse(target)
    score = classify_all(cues)[0]
    assert score.eligible is True
    assert "musical_content_warning" in score.reasons
    # ... while the reference working copy still drops the music markers.
    assert "♪" not in strip_sdh(_cue(1, 60_000, "♪ background music ♪"))
    reference = _cue(1, 60_000, "[door slams]") + "\n\n" + _dialogue(2, 120_000) + "\n"
    assert "[door slams]" not in strip_sdh(reference)
    assert "Distinctive dialogue event number 2 spoken aloud" in strip_sdh(reference)
    assert "[door slams]" in reference


# --------------------------------------------------------------------------- #
# 5. Real dialogue that merely resembles an HI cue
# --------------------------------------------------------------------------- #

def test_bracketed_dialogue_stays_eligible():
    target = "\n\n".join([
        _cue(1, 60_000, "[door slams] Come in, quickly!"),
        _cue(2, 120_000, "(whispering) I saw him there"),
        *(_dialogue(n, 60_000 + (n - 1) * 60_000) for n in range(3, 18)),
    ]) + "\n"
    flags = _eligibility(target)
    assert flags["[door slams] Come in, quickly!"] is True
    assert flags["(whispering) I saw him there"] is True


# --------------------------------------------------------------------------- #
# 6-8. Counts, reconstruction, pristine transfer
# --------------------------------------------------------------------------- #

def test_offset_transfer_preserves_text_across_different_counts():
    target = "\n\n".join(_dialogue(n, 60_000 + (n - 1) * 60_000)
                         for n in range(1, 17)) + "\n"
    shifted = apply_linear_timings(target, 1.0, 9_000)
    assert shifted is not None
    assert [c.text for c in _parse(shifted)] == [c.text for c in _parse(target)]
    original, moved = _parse(target), _parse(shifted)
    assert all((b.start_ms - a.start_ms) == 9_000
               for a, b in zip(original, moved, strict=True))


def _shift_block_timings(text: str, shift_ms: int) -> str:
    moved = apply_linear_timings(text, 1.0, shift_ms)
    assert moved is not None
    return moved


def test_alass_working_copy_reconstructs_onto_pristine_original():
    ad = _cue(1, 2_000, "Download more at www.example-subs.com")
    dialogue = "\n\n".join(_dialogue(n, 60_000 + (n - 2) * 60_000)
                            for n in range(2, 18))
    target = ad + "\n\n" + dialogue + "\n"
    working = prepare_alass_target(target)
    assert "example-subs" not in working  # temp copy drops the ad ...
    assert "example-subs" in target  # ... the original keeps it
    produced = _shift_block_timings(working, 9_000)
    rebuilt = reconstruct_with_original_dialogue(target, produced)
    assert rebuilt is not None
    assert "www.example-subs.com" in rebuilt
    assert [c.text for c in _parse(rebuilt)] == [c.text for c in _parse(target)]
    original, moved = _parse(target), _parse(rebuilt)
    assert all((b.start_ms - a.start_ms) == 9_000
               for a, b in zip(original, moved, strict=True))


# --------------------------------------------------------------------------- #
# 9. Malformed or unusual non-dialogue blocks
# --------------------------------------------------------------------------- #

def test_malformed_block_passes_through_timing_transfer_verbatim():
    blocks = [_dialogue(n, 60_000 + (n - 1) * 60_000) for n in range(1, 13)]
    blocks.insert(6, "stray wrapped continuation line")
    target = "\n\n".join(blocks) + "\n"
    moved = apply_linear_timings(target, 1.0, 9_000)
    assert moved is not None
    assert "stray wrapped continuation line" in moved
    cues, report = parse_srt_for_verification(target, exclude_ambiguous=True)
    assert cues is not None and len(cues) == 12


# --------------------------------------------------------------------------- #
# 10. A subtitle with no advertising or HI cues is not needlessly altered
# --------------------------------------------------------------------------- #

def test_clean_subtitle_analysis_copies_preserve_everything():
    target = "\n\n".join(_dialogue(n, 60_000 + (n - 1) * 60_000)
                         for n in range(1, 17)) + "\n"
    assert strip_advertisements(target, keep_translator_credits=True) == target
    working = prepare_alass_target(target)
    assert [c.text for c in _parse(working)] == [c.text for c in _parse(target)]
    assert [(c.start_ms, c.end_ms) for c in _parse(working)] == [
        (c.start_ms, c.end_ms) for c in _parse(target)]
    assert all(score.eligible for score in classify_all(_parse(target)))


# --------------------------------------------------------------------------- #
# End to end: filtered analysis, pristine delivery (Two-Point, no ALASS)
# --------------------------------------------------------------------------- #

def _orchestrator(monkeypatch, resolved):
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
    service = SimpleNamespace(sync_async=AsyncMock(return_value=None))
    orch = om.SyncOrchestrator(sync_cache=SyncCache(), external_strategy=strategy,
                              sync_service=service)
    return orch, service


async def test_served_subtitle_keeps_every_original_line(monkeypatch) -> None:
    """Credit + promo + HI + tail ad survive; only timings move (Two-Point)."""
    alternates = ["Completely different sentence alpha",
                  "Completely different sentence beta"]
    target_blocks = [_cue(1, 2_000, "ترجمة: فريق العمل")]
    reference_blocks = []
    number = 2
    for position in range(16):
        start = 60_000 + position * 60_000
        text = f"Distinctive dialogue event number {position + 1} spoken aloud"
        if position in (7,):
            target_blocks.append(_cue(number, start, "تابعنا على تيليجرام @channel"))
            reference_blocks.append(_cue(number, start + 9_000, alternates[0]))
            number += 1
        if position in (3,):
            target_blocks.append(_cue(number, start + 5_000, "[door slams]"))
            reference_blocks.append(_cue(number, start + 5_000 + 9_000, alternates[1]))
            number += 1
        target_blocks.append(_cue(number, start, text))
        shown = alternates[position % 2] if position in (4, 8, 12) else text
        reference_blocks.append(_cue(number, start + 9_000, shown))
        number += 1
    target_blocks.append(_cue(number, start + 55_000, "www.example-subs.com"))
    target = "\n\n".join(target_blocks) + "\n"
    reference = "\n\n".join(reference_blocks) + "\n"
    resolved = ResolvedReference(reference, kind="edition", candidate="ref.srt")
    orch, service = _orchestrator(monkeypatch, resolved)
    meta = {"imdb_id": "tt0000001", "media_type": "series", "season": 1,
            "episode": 1, "lang": "ara", "target_filename": "Some.Show.S01E01.mkv",
            "title": "Some Show", "video_size": 1000}
    out = await orch.evaluate_and_sync(target.encode(), meta, "sub", auto_sync=True)
    assert out != target.encode()
    service.sync_async.assert_not_called()
    assert [c.text for c in _parse(out.decode())] == [c.text for c in _parse(target)]
    assert "ترجمة: فريق العمل" in out.decode()
    assert "@channel" in out.decode()
    assert "[door slams]" in out.decode()
    assert "www.example-subs.com" in out.decode()
