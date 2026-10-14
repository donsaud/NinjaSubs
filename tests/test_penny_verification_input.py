"""Penny Dreadful verification-input regressions (live HDTV inputs).

Live failure: both HDTV Arabic targets were unparseable by Unified's strict
SRT grammar, so every candidate abstained with
``unparseable:SubtitleParseError`` and Auto-Sync never engaged — even though
the serving pipeline plays these files and ALASS aligns them.

* ``penny-s01e01-hdtv-ar.srt`` — timing-only cue (block 7) aborted parsing.
  The lenient verification boundary skips it with accounting; the Arabic
  reference path then genuinely VERIFIED (371 groups), and the English path
  correctly abstains (cross-language, no correspondences).
* ``penny-s01e01-hdtv-ar-2.srt`` — stray continuation line (block 353) with
  genuinely ambiguous ownership. The boundary must abstain explicitly rather
  than invent cue boundaries or discard dialogue.

Fixtures are byte copies of the live-cached inputs; do NOT edit them.
"""

from __future__ import annotations

import hashlib
import pathlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.sync import hybrid_router as router
from app.services.sync.candidate_reconstruction import (
    reconstruct_with_original_dialogue,
)
from app.services.sync.query import ResolvedReference
from app.services.sync.unified_verification import _check_integrity, verify_candidate
from app.services.sync_cache import SyncCache
from two_point_sync.anchors import normalize_text
from two_point_sync.parser import (
    Cue,
    SubtitleParseError,
    format_srt,
    parse_srt,
    parse_srt_for_verification,
)

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "penny"
TARGET = FIXTURES / "penny-s01e01-hdtv-ar.srt"
TARGET_2 = FIXTURES / "penny-s01e01-hdtv-ar-2.srt"
AR_REFERENCE = FIXTURES / "penny-s01e01-bluray-aos-ar.srt"
EN_REFERENCE = FIXTURES / "penny-s01e01-bluray-rovers-en.srt"
AR_OUTPUT = FIXTURES / "penny-s01e01-alass-aos-output.srt"


def _read(path: pathlib.Path) -> str:
    return path.read_bytes().decode("utf-8-sig")


def test_strict_parser_still_rejects_both_inputs() -> None:
    """Boundary documentation: the serving-strict grammar is unchanged."""
    with pytest.raises(SubtitleParseError):
        parse_srt(_read(TARGET))
    with pytest.raises(SubtitleParseError):
        parse_srt(_read(TARGET_2))


def test_lenient_parse_accounts_textless_cues() -> None:
    cues, report = parse_srt_for_verification(_read(TARGET))
    assert cues is not None and len(cues) == 383
    assert report.ambiguous == ()
    assert report.renumbered is True
    kinds = [kind for _, kind in report.excluded]
    assert "empty_text" in kinds
    assert [c.number for c in cues] == list(range(1, len(cues) + 1))


def test_lenient_parse_abstains_on_ambiguous_block() -> None:
    cues, report = parse_srt_for_verification(_read(TARGET_2))
    assert cues is None
    assert 353 in report.ambiguous


def test_arabic_path_measures_and_verifies() -> None:
    target, reference = _read(TARGET), _read(AR_REFERENCE)
    candidate = _read(AR_OUTPUT)
    reconstructed = reconstruct_with_original_dialogue(target, candidate)
    assert reconstructed is not None
    verdict = verify_candidate(target, reference, reconstructed,
                               candidate_algorithm="alass")
    assert verdict.proposed_verdict == "VERIFIED", verdict.failure_reasons
    # Endpoint dialogue events used by Two-Point fitting are held out of
    # timing measurement, so measurable groups are total minus holdout.
    assert verdict.independent_evidence_count == 370
    assert "unparseable" not in ";".join(verdict.failure_reasons)


def test_english_path_abstains_on_merits_not_parse() -> None:
    target, reference = _read(TARGET), _read(EN_REFERENCE)
    verdict = verify_candidate(target, reference, target,
                               candidate_algorithm="already_aligned")
    assert verdict.proposed_verdict == "UNVERIFIED"
    assert "cross_language_insufficient_evidence" in verdict.failure_reasons
    assert not any("unparseable" in r or "ambiguous" in r
                   for r in verdict.failure_reasons)


def test_already_aligned_real_target_verifies() -> None:
    """No destructive transformation needed: the original itself verifies."""
    target, reference = _read(TARGET), _read(AR_REFERENCE)
    verdict = verify_candidate(target, reference, target,
                               candidate_algorithm="already_aligned")
    assert verdict.proposed_verdict == "VERIFIED", verdict.failure_reasons


def test_ambiguous_target_measured_not_vetoed() -> None:
    """Block 353 contributes no evidence but never vetoes measurable timing.

    The HDTV block-353 input is genuinely ~12s mistimed throughout (user
    confirmed NOT synchronized): the timing-first verifier measures the
    remaining 80 independent dialogue groups across all thirds and REJECTS
    on timing disagreement, instead of aborting the whole document on the
    ambiguous block.
    """
    from app.services.sync.correspondence_verification import MIN_INDEPENDENT_GROUPS

    target, reference = _read(TARGET_2), _read(AR_REFERENCE)
    verdict = verify_candidate(target, reference, target,
                               candidate_algorithm="already_aligned")
    assert verdict.proposed_verdict == "REJECTED", verdict.failure_reasons
    assert verdict.independent_evidence_count >= MIN_INDEPENDENT_GROUPS
    assert "conflicting_dialogue_timing" in verdict.failure_reasons
    assert verdict.coverage.get("opening", 0) >= 2
    assert verdict.coverage.get("middle", 0) >= 2
    assert verdict.coverage.get("ending", 0) >= 2
    # The ambiguous block is accounted as unmeasurable dialogue, never as
    # a successful point and never as an automatic veto of the rest.
    assert verdict.integrity_result.get("unmeasurable_blocks", 0) >= 1


def test_block353_alass_timing_genuinely_fixed() -> None:
    """Real container ALASS output: timing fixed; raw bytes still not servable.

    The sanitized working-copy triple (what ALASS was actually given)
    VERIFIES with the same 80 groups at small residuals: ALASS genuinely
    corrected the ~12s offset. But the raw ALASS bytes cannot be served as
    a verified synchronization of the pristine original: sanitization
    dropped the translator-credit cue and the ambiguous block-353 line, so
    the raw candidate REJECTS on output integrity. Timing success must
    never override dialogue preservation; reconstruction (tested below)
    must bridge the gap.
    """
    import app.services.sync.orchestrator  # noqa: F401  (import-order: avoids sync_service cycle)
    from app.services.sync_service import (
        prepare_alass_target,
        sanitize_subtitle,
        strip_sdh,
    )
    from app.utils.cleaners import strip_advertisements

    target, reference = _read(TARGET_2), _read(AR_REFERENCE)
    alass_output = _read(FIXTURES / "penny-s01e01-block353-alass-output.srt")
    sanitized_target = prepare_alass_target(target)
    sanitized_reference = sanitize_subtitle(
        strip_advertisements(strip_sdh(reference), keep_translator_credits=False))
    timing = verify_candidate(sanitized_target, sanitized_reference, alass_output,
                              candidate_algorithm="alass")
    assert timing.proposed_verdict == "VERIFIED", timing.failure_reasons
    assert timing.independent_evidence_count >= 50
    assert timing.residual_statistics["median"] < 750.0
    assert timing.residual_statistics["max"] < 3000.0

    delivery = verify_candidate(target, reference, alass_output,
                                candidate_algorithm="alass")
    assert delivery.proposed_verdict == "REJECTED", delivery.failure_reasons


def test_block353_reconstructed_final_verifies() -> None:
    """Acceptance target C: offset-transferred final subtitle VERIFIES.

    ALASS proved a rigid +12025ms shift in the sanitized domain; the final
    artifact carries that shift on every original timed block while keeping
    all original dialogue byte-identical. The same 80 correspondence groups
    now agree at small residuals, so Timing-First returns VERIFIED.
    """
    target, reference = _read(TARGET_2), _read(AR_REFERENCE)
    alass_output = _read(FIXTURES / "penny-s01e01-block353-alass-output.srt")
    final = reconstruct_with_original_dialogue(target, alass_output)
    assert final is not None
    assert final == _read(FIXTURES / "penny-s01e01-block353-reconstructed.srt")
    verdict = verify_candidate(target, reference, final,
                               candidate_algorithm="alass")
    assert verdict.proposed_verdict == "VERIFIED", verdict.failure_reasons
    assert verdict.independent_evidence_count >= 50
    assert verdict.residual_statistics["median"] < 750.0
    assert verdict.residual_statistics["max"] < 3000.0
    assert verdict.integrity_result.get("accepted") is True


def test_block353_final_preserves_all_content() -> None:
    """No dialogue lost, invented, or re-owned in the final artifact."""
    from two_point_sync.parser import _split_blocks

    target = _read(TARGET_2)
    final = _read(FIXTURES / "penny-s01e01-block353-reconstructed.srt")
    original_cues, _ = parse_srt_for_verification(target, exclude_ambiguous=True)
    final_cues, _ = parse_srt_for_verification(final, exclude_ambiguous=True)
    assert original_cues is not None and final_cues is not None
    assert len(final_cues) == len(original_cues) == 474
    # Timestamp differences expected; dialogue differences are not.
    assert [c.text for c in final_cues] == [c.text for c in original_cues]
    assert all((b.start_ms - a.start_ms) == 12025 and (b.end_ms - a.end_ms) == 12025
               for a, b in zip(original_cues, final_cues, strict=True))
    # Translator credit preserved and shifted with everything else.
    assert "FDC400" in final_cues[-1].text
    # Ambiguous block-353 text preserved byte-identical, still unowned.
    old_blocks = _split_blocks(target.lstrip("﻿"))
    new_blocks = _split_blocks(final.lstrip("﻿"))
    assert len(old_blocks) == len(new_blocks) == 475
    assert old_blocks[352] == new_blocks[352]
    assert "مالكولم" in new_blocks[352]


def test_block353_final_delivery_integrity(caplog) -> None:
    """The final artifact survives the real HTTP delivery path as TRANSFORMED."""
    import hashlib
    import logging

    from app import main
    from app.services.sync.alignment import may_serve_synchronized
    from app.services.sync.delivery import DeliveryAuthorization, DeliveryPayload
    from app.services.sync.unified_verification import to_evaluation

    caplog.set_level(logging.INFO)
    target, reference = _read(TARGET_2), _read(AR_REFERENCE)
    final = _read(FIXTURES / "penny-s01e01-block353-reconstructed.srt")
    verdict = verify_candidate(target, reference, final,
                               candidate_algorithm="alass")
    assert verdict.proposed_verdict == "VERIFIED"
    evaluation = to_evaluation(verdict)
    evaluation.target_binding = "exact"
    candidate_bytes = final.encode("utf-8")
    authorization = DeliveryAuthorization(
        "req-penny-test", hashlib.sha256(candidate_bytes).hexdigest(),
        hashlib.sha256(reference.encode("utf-8")).hexdigest(), "exact", "VERIFIED",
        may_serve_synchronized(evaluation.sync_state, evaluation.verification),
        target.encode("utf-8"))
    response = main._build_subtitle_response(
        DeliveryPayload(candidate_bytes, authorization),
        "Penny.Dreadful.S01E01.MULTi.1080p.BluRay.x264-PANZeR",
        "srt", auto_sync=True)
    assert response.status_code == 200
    records = [r.message for r in caplog.records if "[unified-delivery]" in r.message]
    assert any("decision=TRANSFORMED" in message and "final_integrity=True" in message
               for message in records)
    # Served bytes carry corrected timing, not the original grid.
    assert bytes(response.body) != target.encode("utf-8")


def test_empty_cue_accounting_synthetic() -> None:
    target = ("1\n00:00:01,000 --> 00:00:03,000\nHello\n\n"
              "2\n00:00:04,000 --> 00:00:05,000\n\n"
              "3\n00:00:06,000 --> 00:00:08,000\nWorld\n")
    cues, report = parse_srt_for_verification(target)
    assert cues is not None and len(cues) == 2
    assert report.excluded == ((2, "empty_text"),)
    assert [c.text for c in cues] == ["Hello", "World"]


def test_ambiguous_block_synthetic() -> None:
    target = ("1\n00:00:01,000 --> 00:00:03,000\nHello\n\n"
              "stray wrapped continuation line\n\n"
              "2\n00:00:04,000 --> 00:00:06,000\nWorld\n")
    cues, report = parse_srt_for_verification(target)
    assert cues is None
    assert report.ambiguous == (2,)


def test_deletion_still_rejects() -> None:
    target, reference = _read(TARGET), _read(AR_REFERENCE)
    cues = parse_srt_for_verification(target)[0]
    assert cues is not None
    damaged = [c for c in cues if c.number != 6]
    verdict = verify_candidate(target, reference, format_srt(damaged),
                               candidate_algorithm="alass")
    assert verdict.proposed_verdict in ("REJECTED", "UNVERIFIED")


def test_modification_still_rejects() -> None:
    target, reference = _read(TARGET), _read(AR_REFERENCE)
    cues = parse_srt_for_verification(target)[0]
    assert cues is not None
    tampered = [
        Cue(number=c.number, order=c.order, start_ms=c.start_ms, end_ms=c.end_ms,
            text="completely corrupted dialogue line" if c.number == 10 else c.text)
        for c in cues
    ]
    verdict = verify_candidate(target, reference, format_srt(tampered),
                               candidate_algorithm="alass")
    assert verdict.proposed_verdict == "REJECTED"
    assert any("dialogue_text_modified" in r for r in verdict.failure_reasons)


def test_identity_mismatch_still_rejects() -> None:
    target, reference = _read(TARGET), _read(AR_REFERENCE)
    cues = parse_srt_for_verification(target)[0]
    assert cues is not None
    shifted = [
        Cue(number=c.number, order=c.order, start_ms=c.start_ms + 5000,
            end_ms=c.end_ms + 5000, text="unrelated dialogue line")
        for c in cues
    ]
    verdict = verify_candidate(target, reference, format_srt(shifted),
                               candidate_algorithm="alass")
    assert verdict.proposed_verdict in ("REJECTED", "UNVERIFIED")


def test_inherited_disorder_accepted_but_new_disorder_rejects() -> None:
    base = [Cue(number=i + 1, order=i, start_ms=i * 2000,
                end_ms=i * 2000 + 1500, text=f"line {i + 1}") for i in range(6)]
    # Source disorder at pair (2, 3): candidate preserving it is not scrambling.
    disordered = list(base)
    disordered[2] = Cue(number=3, order=2, start_ms=6000, end_ms=7500, text="line 3")
    disordered[3] = Cue(number=4, order=3, start_ms=4000, end_ms=5500, text="line 4")
    assert _check_integrity(disordered, disordered)["accepted"] is True
    assert _check_integrity(disordered, disordered)["inherited_order_violations"] == 1
    # Freshly scrambling an ordered target still rejects.
    scrambled = list(base)
    scrambled[2], scrambled[3] = scrambled[3], scrambled[2]
    renumbered = [Cue(number=i + 1, order=c.order, start_ms=c.start_ms,
                      end_ms=c.end_ms, text=c.text) for i, c in enumerate(scrambled)]
    result = _check_integrity(base, renumbered)
    assert result["accepted"] is False
    assert any("candidate_order_violation" in r for r in result["reasons"])


def test_insufficient_evidence_fallback_preserved() -> None:
    tiny = ("1\n00:00:01,000 --> 00:00:03,000\nHello\n\n"
            "2\n00:01:00,000 --> 00:01:02,000\nWorld\n")
    verdict = verify_candidate(tiny, tiny.replace("Hello", "Different"), tiny,
                               candidate_algorithm="alass")
    assert verdict.proposed_verdict in ("REJECTED", "UNVERIFIED")


def test_penny_reconstruction_preserves_dialogue() -> None:
    target = _read(TARGET)
    candidate = _read(AR_OUTPUT)
    reconstructed = reconstruct_with_original_dialogue(target, candidate)
    assert reconstructed is not None
    original = parse_srt_for_verification(target)[0]
    produced = parse_srt_for_verification(candidate)[0]
    rebuilt = parse_srt_for_verification(reconstructed)[0]
    assert original is not None and produced is not None and rebuilt is not None
    assert [c.text for c in rebuilt] == [c.text for c in original]
    assert [(c.start_ms, c.end_ms) for c in rebuilt] == [
        (c.start_ms, c.end_ms) for c in produced]
    assert all(normalize_text(a.text) == normalize_text(b.text)
               for a, b in zip(rebuilt, produced, strict=False))


def _intro_target() -> tuple[str, str]:
    """Pristine target with a leading translator credit plus a shifted twin."""
    lines = ["1\n00:00:02,000 --> 00:00:04,000\nTranslated by ExampleFan"]
    shifted = []
    for number in range(2, 14):
        start = 60_000 + (number - 2) * 60_000
        lines.append(f"{number}\n{_fmt(start)} --> {_fmt(start + 2_000)}\nDialogue line {number}")
        shifted.append(f"{number - 1}\n{_fmt(start + 5_000)} --> {_fmt(start + 7_000)}"
                       f"\nDialogue line {number}")
    return "\n\n".join(lines) + "\n", "\n\n".join(shifted) + "\n"


def _fmt(total_ms: int) -> str:
    total_ms = max(0, total_ms)
    millis = total_ms % 1000
    seconds = (total_ms // 1000) % 60
    minutes = (total_ms // 60_000) % 60
    hours = total_ms // 3_600_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def test_pristine_identity_matches_and_mismatches() -> None:
    from app.services.sync.reference_content_identity import PristineTargetIdentity

    identity = PristineTargetIdentity.precompute("hello\nworld")
    assert identity.matches("hello\nworld") == "byte_identical"
    assert identity.matches("  hello   world  ") == "normalized_identical"
    assert identity.matches("completely different") == ""
    assert identity.matches("") == ""
    assert identity.matches(None) == ""
    assert PristineTargetIdentity.precompute("").matches("hello") == ""
    assert PristineTargetIdentity.precompute(None).matches("hello") == ""


def _identity_harness(monkeypatch, candidates: list):
    from app.services.sync import orchestrator as om
    from app.services.sync.target_binding import classify_target_binding

    monkeypatch.setattr(om.SyncOrchestrator, "_audit_serve", lambda self, *a, **k: None)
    monkeypatch.setattr(om.SyncOrchestrator, "_record_outcome", lambda self, *a, **k: None)
    monkeypatch.setattr(om.SyncOrchestrator, "_try_timeline_affine",
                        AsyncMock(return_value=None))
    monkeypatch.setattr(om.SyncOrchestrator, "_try_pal_two_point_fallback",
                        AsyncMock(return_value=None))
    monkeypatch.setattr(om.SyncOrchestrator, "_try_hybrid_sync",
                        AsyncMock(return_value=None))
    monkeypatch.setattr(router, "_enabled", False)
    monkeypatch.delenv(router.TEST_ENABLED_ENV, raising=False)
    monkeypatch.delenv(router.TEST_SCOPE_ENV, raising=False)
    strong = classify_target_binding(
        "Film.2020.1080p.BluRay.x264-GRP", "Film.2020.1080p.BluRay.x264-GRP"
    )
    assert strong.supports_verified
    monkeypatch.setattr(
        om.SyncOrchestrator, "_target_binding", staticmethod(lambda resolved, meta: strong)
    )
    strategy = SimpleNamespace(
        resolve_all_with_provenance=AsyncMock(return_value=candidates)
    )
    service = SimpleNamespace(sync_async=AsyncMock(return_value=None))
    cache = SyncCache()
    return (om.SyncOrchestrator(sync_cache=cache, external_strategy=strategy,
                               sync_service=service), cache, service, strategy)


async def test_self_reference_skipped_after_intro_strip(monkeypatch) -> None:
    """Pristine guard fires even after working-copy preparation (live miss)."""
    from app.services.sync import orchestrator as om
    from tests.test_unified_authority import META

    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True)
    target, shifted_twin = _intro_target()
    first = ResolvedReference(shifted_twin, kind="edition", candidate="other.srt")
    identical = ResolvedReference(target, kind="edition", candidate="same.srt")
    orch, _cache, service, _strategy = _identity_harness(monkeypatch, [first, identical])
    out = await orch.evaluate_and_sync(target.encode(), dict(META), "sub", auto_sync=True)
    assert out == target.encode()
    # Exactly one ALASS invocation, against the distinct candidate only: the
    # byte-identical download that follows intro stripping never reaches it.
    assert service.sync_async.await_count == 1
    assert service.sync_async.await_args[0][1] == shifted_twin


async def test_no_circular_success_on_identical_reference(monkeypatch) -> None:
    """An identical reference alone can never produce a transformed serve."""
    from app.services.sync import orchestrator as om
    from tests.test_unified_authority import META

    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True)
    target, _ = _intro_target()
    identical = ResolvedReference(target, kind="edition", candidate="same.srt")
    orch, cache, service, _strategy = _identity_harness(monkeypatch, [identical])
    out = await orch.evaluate_and_sync(target.encode(), dict(META), "sub", auto_sync=True)
    assert out == target.encode()
    service.sync_async.assert_not_called()
    verdict_key = om.SyncOrchestrator._verdict_key(
        META, hashlib.sha256(target.encode()).hexdigest()[:16]
    )
    assert await cache.get_verdict(verdict_key) is None


async def test_video_cache_identical_artifact_skipped() -> None:
    """Reused cached references face the same independence guard."""
    from app.services.sync.orchestrator import _lookup_video_reference
    from app.services.sync.query import ReferenceQuery
    from app.services.sync.telemetry import SyncTiming

    target = "1\n00:00:01,000 --> 00:00:03,000\nHello\n"
    query = ReferenceQuery(imdb_id="tt1", target_text=target)
    timing = SyncTiming()
    same = ResolvedReference(target, candidate="same.srt", provider="subdl")
    cache = SimpleNamespace(get_video_scoped=lambda query: [same])
    strategy = SimpleNamespace(cache=cache)
    timing = SyncTiming()
    assert _lookup_video_reference(strategy, query, timing) is None
    assert timing.reference_outcomes.get("reference_content_identical_to_target") == 1
    other = ResolvedReference(target.replace("Hello", "Different"), candidate="o.srt",
                              provider="subdl")
    cache2 = SimpleNamespace(get_video_scoped=lambda query: [other])
    assert _lookup_video_reference(SimpleNamespace(cache=cache2), query, SyncTiming()) is not None


async def test_unmarked_unparseable_negative_reevaluates(monkeypatch) -> None:
    """Pre-leniency parse-abort verdicts refresh exactly once, then reuse."""
    from app.services.sync import orchestrator as om
    from tests.test_unified_authority import META

    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True)
    target, _ = _intro_target()
    orch, cache, service, strategy = _identity_harness(monkeypatch, [])
    # The harness pins the gate OFF; this stale-cache case needs it ON.
    monkeypatch.setattr(router, "_enabled", True)
    monkeypatch.delenv(router.TEST_SCOPE_ENV, raising=False)
    verdict_key = om.SyncOrchestrator._verdict_key(
        META, hashlib.sha256(target.encode()).hexdigest()[:16]
    )
    await cache.set_verdict(verdict_key, {
        "sync_state": "unverified", "verification": "verified",
        "reasons": ["unified_unverified: unparseable:SubtitleParseError"],
        "target_binding": "exact",
        "verification_method": "unified", "verification_version": "unified-v1",
    })
    out = await orch.evaluate_and_sync(target.encode(), dict(META), "sub", auto_sync=True)
    assert out == target.encode()
    assert strategy.resolve_all_with_provenance.await_count == 1
