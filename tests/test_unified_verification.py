"""Unified verifier tests: benchmarks, shadow comparison, adversarial cases.

Shadow mode: everything here measures; nothing serves, caches, or runs
ALASS (except the binary-gated regeneration test, which only recomputes
bytes). Real fixtures skip gracefully when absent.
"""

from __future__ import annotations

import ast
import hashlib
import pathlib

import pytest

from app.services.sync.unified_verification import (
    METHOD,
    VERSION,
    shadow_compare,
    to_evaluation,
    verify_candidate,
)
from tests.two_point_sync._helpers import apply_affine, build_dialogue
from two_point_sync.parser import Cue, format_srt, parse_srt
from two_point_sync.transform import LinearTransform, apply_transform

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"
WHIP_TARGET = FIXTURES / "whiplash" / "Whiplash (2014) WEBRip.iTunes.ar(1).srt"
WHIP_REFERENCE = (
    FIXTURES / "whiplash" / "Whiplash.2014.1080p.REMUX.BluRay.AVC.DTS.HD.MA.5.1-HDA(1).srt"
)
DEX_DIR = FIXTURES / "two_point_sync" / "dexter"
DEX_TARGET = DEX_DIR / "Dexter.S08E01.480p.HDTV.x264-mSD.srt"
DEX_FRAMESTOR = DEX_DIR / "Dexter.S08E01.A.Beautiful.Day.BluRay.1080p.TrueHD.5.1.AVC.REMUX-FraMeSToR.ar.srt"
ROVERS_DIR = FIXTURES / "hybrid_rovers"
ROVERS_REFERENCE = ROVERS_DIR / "dexter-s08e01-rovers-634.srt"
ROVERS_CANDIDATE = ROVERS_DIR / "dexter-msd-rovers-alass-synced.srt"
DEX_VERIFIED_CANDIDATE = (
    FIXTURES / ".." / ".." / "reports" / "two_point_sync" / "dexter.alass.verified.srt"
).resolve()


def _read(path: pathlib.Path) -> str:
    if not path.exists():
        pytest.skip(f"fixture unavailable: {path.name}")
    return path.read_bytes().decode("utf-8-sig")


def _linear_pair(count: int = 14, step_ms: int = 60_000,
                 scale: float = 1.0, offset_ms: float = 5_000.0):
    """Overlapping-but-different pair: same timing model, reworded lines, so
    the reference carries genuinely different content (independent timing
    proof) while keeping exact correspondences for the nucleus."""
    target_cues = build_dialogue(count, step_ms=step_ms)
    first, second = max(2, count // 3), max(3, 2 * count // 3)
    alternates = {first: "Completely different sentence one",
                  second: "Completely different sentence two"}
    reference_cues = []
    for cue in apply_affine(target_cues, scale, offset_ms):
        text = alternates.get(cue.number, cue.text)
        reference_cues.append(Cue(number=cue.number, order=cue.order,
                                  start_ms=cue.start_ms, end_ms=cue.end_ms, text=text))
    return format_srt(target_cues), format_srt(reference_cues)


def _identical_pair(count: int = 14, step_ms: int = 60_000,
                    scale: float = 1.0, offset_ms: float = 5_000.0):
    """Same translation on both sides (triggers same-content protection)."""
    target_cues = build_dialogue(count, step_ms=step_ms)
    return format_srt(target_cues), format_srt(apply_affine(target_cues, scale, offset_ms))


def _diverged_pair(count: int = 14, step_ms: int = 60_000,
                   scale: float = 1.0, offset_ms: float = 5_000.0):
    """Alias kept for readability at call sites needing independence."""
    return _linear_pair(count, step_ms=step_ms, scale=scale, offset_ms=offset_ms)


def _true_candidate(target_text: str, scale: float, offset_ms: float) -> str:
    return format_srt(apply_transform(
        parse_srt(target_text), LinearTransform(scale=scale, offset_ms=offset_ms)))


# ------------------------------------------------------------ benchmarks ---


def test_whiplash_unified_verified() -> None:
    target, reference = _read(WHIP_TARGET), _read(WHIP_REFERENCE)
    cues = parse_srt(target)
    candidate = _true_candidate(target, 1.04277053, -8707.9)
    verdict = verify_candidate(target, reference, candidate, candidate_algorithm="two_point")
    assert verdict.proposed_verdict == "VERIFIED", verdict.failure_reasons
    assert verdict.verification_method == METHOD
    assert verdict.verification_version == VERSION
    assert verdict.independent_evidence_count >= 5
    assert all(count >= 2 for count in verdict.coverage.values())
    assert verdict.residual_statistics["median"] <= 750.0
    assert verdict.candidate_sha256 == hashlib.sha256(candidate.encode()).hexdigest()
    assert verdict.target_identity["cues"] == len(cues) == 825
    assert verdict.temporal_result["movement_magnitude_ms"] > 60_000
    assert verdict.structural_result["available"] is True


def test_dexter_framestor_unified_verified() -> None:
    target, reference = _read(DEX_TARGET), _read(DEX_FRAMESTOR)
    candidate = _read(DEX_VERIFIED_CANDIDATE)
    verdict = verify_candidate(target, reference, candidate, candidate_algorithm="alass")
    assert verdict.proposed_verdict == "VERIFIED", verdict.failure_reasons
    assert len(verdict.temporal_result.get("regimes", [])) >= 1
    assert verdict.independent_evidence_count >= 5
    assert any(entry.get("kind") == "within_reference_span"
               for entry in verdict.risk_findings)
    assert verdict.residual_statistics["median"] <= 750.0


def test_dexter_rovers_unified_verified() -> None:
    target, reference = _read(DEX_TARGET), _read(ROVERS_REFERENCE)
    candidate = _read(ROVERS_CANDIDATE)
    verdict = verify_candidate(target, reference, candidate, candidate_algorithm="alass")
    assert verdict.proposed_verdict == "VERIFIED", verdict.failure_reasons
    assert verdict.independent_evidence_count >= 5


def test_already_aligned_verified_without_transformation() -> None:
    target_cues = build_dialogue(14)
    target = format_srt(target_cues)
    alternates = ["Completely different sentence one", "Completely different sentence two",
                  "Completely different sentence three"]
    reference_cues = [
        Cue(number=c.number, order=c.order, start_ms=c.start_ms, end_ms=c.end_ms,
            text=alternates[(c.number // 4) - 1] if c.number in (4, 8, 12) else c.text)
        for c in target_cues
    ]
    reference = format_srt(reference_cues)
    verdict = verify_candidate(target, reference, target, candidate_algorithm="two_point")
    assert verdict.proposed_verdict == "VERIFIED", verdict.failure_reasons
    assert verdict.residual_statistics["median"] == 0.0


def test_pal_affine_output_verified() -> None:
    target_text, reference_text = _linear_pair(16, step_ms=45_000,
                                               scale=25.0 / 23.976, offset_ms=-8_000.0)
    candidate = _true_candidate(target_text, 25.0 / 23.976, -8_000.0)
    verdict = verify_candidate(target_text, reference_text, candidate,
                               candidate_algorithm="two_point")
    assert verdict.proposed_verdict == "VERIFIED", verdict.failure_reasons
    # Integer-ms rounding of reference times bounds the fitted scale.
    assert abs(verdict.temporal_result["scale"] - 25.0 / 23.976) < 5e-6


def test_shadow_compare_explains_dexter_disagreement() -> None:
    target, reference = _read(DEX_TARGET), _read(DEX_FRAMESTOR)
    candidate = _read(DEX_VERIFIED_CANDIDATE)
    comparison = shadow_compare(target, reference, candidate, candidate_algorithm="alass")
    assert comparison["legacy"]["evaluated"] is True
    assert comparison["legacy"]["sync_state"] == "unverified"
    assert comparison["v2"]["evaluated"] is True
    assert comparison["unified"]["proposed_verdict"] == "VERIFIED"
    assert comparison["agreement"] is False
    assert comparison["elapsed_ms"] >= 0


def test_shadow_compare_agrees_on_clean_pair() -> None:
    target_text, reference_text = _linear_pair(14)
    candidate = _true_candidate(target_text, 1.0, 5_000.0)
    comparison = shadow_compare(target_text, reference_text, candidate,
                                candidate_algorithm="two_point")
    assert comparison["unified"]["proposed_verdict"] == "VERIFIED"
    assert comparison["agreement"] is True


# ------------------------------------------------------------ adversarial ---


def test_identical_content_refused() -> None:
    target_text, reference_text = _identical_pair()
    verdict = verify_candidate(target_text, reference_text, target_text,
                               candidate_algorithm="two_point")
    assert verdict.proposed_verdict == "UNVERIFIED"
    assert any("identical" in reason for reason in verdict.failure_reasons)


def test_wrong_episode_rejected_or_unverified() -> None:
    target_text, _ = _linear_pair()
    other = build_dialogue(14, sentences=[f"Completely different sentence {w}" for w in
                                          ("one", "two", "three", "four", "five", "six",
                                           "seven", "eight", "nine", "ten", "eleven",
                                           "twelve", "thirteen", "fourteen")])
    reference_text = format_srt(other)
    verdict = verify_candidate(target_text, reference_text, target_text,
                               candidate_algorithm="two_point")
    assert verdict.proposed_verdict in ("UNVERIFIED", "REJECTED")
    assert verdict.failure_reasons


def test_incorrect_scale_rejected() -> None:
    target_text, reference_text = _linear_pair(16, step_ms=45_000,
                                               scale=25.0 / 23.976, offset_ms=-8_000.0)
    candidate = _true_candidate(target_text, 1.0, -8_000.0)
    verdict = verify_candidate(target_text, reference_text, candidate,
                               candidate_algorithm="two_point")
    assert verdict.proposed_verdict == "REJECTED"
    assert any("drift" in reason or "median" in reason or "max" in reason
               for reason in verdict.failure_reasons)


def test_incorrect_offset_rejected() -> None:
    target_text, reference_text = _linear_pair()
    candidate = _true_candidate(target_text, 1.0, 60_000.0)
    verdict = verify_candidate(target_text, reference_text, candidate,
                               candidate_algorithm="two_point")
    assert verdict.proposed_verdict == "REJECTED"


def test_mistimed_opening_rejected() -> None:
    target_text, reference_text = _linear_pair(12)
    base = apply_transform(parse_srt(target_text), LinearTransform(scale=1.0, offset_ms=5_000.0))
    bad = [Cue(number=c.number, order=c.order,
               start_ms=c.start_ms + (30_000 if c.number <= 4 else 0),
               end_ms=c.end_ms + (30_000 if c.number <= 4 else 0), text=c.text) for c in base]
    verdict = verify_candidate(target_text, reference_text, format_srt(bad),
                               candidate_algorithm="two_point")
    assert verdict.proposed_verdict == "REJECTED"
    assert any("opening" in reason for reason in verdict.failure_reasons)


def test_mistimed_ending_rejected() -> None:
    target_text, reference_text = _linear_pair(12)
    base = apply_transform(parse_srt(target_text), LinearTransform(scale=1.0, offset_ms=5_000.0))
    bad = [Cue(number=c.number, order=c.order,
               start_ms=c.start_ms + (30_000 if c.number >= 9 else 0),
               end_ms=c.end_ms + (30_000 if c.number >= 9 else 0), text=c.text) for c in base]
    verdict = verify_candidate(target_text, reference_text, format_srt(bad),
                               candidate_algorithm="two_point")
    assert verdict.proposed_verdict == "REJECTED"
    assert any("ending" in reason for reason in verdict.failure_reasons)


def test_hidden_second_regime_rejected() -> None:
    """Dominant constant movement (MAD≈0) hiding a mistimed opening."""
    target = build_dialogue(14, step_ms=60_000)
    reference = apply_affine(target, 1.0, 5_000.0)
    reworded = {5: "Completely different sentence one",
                12: "Completely different sentence two"}
    shifted = [Cue(number=c.number, order=c.order,
                   start_ms=c.start_ms + (50_000 if position < 3 else 0),
                   end_ms=c.end_ms + (50_000 if position < 3 else 0),
                   text=reworded.get(c.number, c.text))
               for position, c in enumerate(reference)]
    target_text, shifted_text = format_srt(target), format_srt(shifted)
    candidate = _true_candidate(target_text, 1.0, 5_000.0)
    verdict = verify_candidate(target_text, shifted_text, candidate,
                               candidate_algorithm="alass")
    assert verdict.proposed_verdict == "REJECTED", verdict.failure_reasons
    assert len(verdict.temporal_result.get("regimes", [])) >= 1


def test_concentrated_cluster_rejected_but_scattered_listed() -> None:
    base_text, reference_text = _linear_pair(30, step_ms=60_000)
    base = apply_transform(parse_srt(base_text), LinearTransform(scale=1.0, offset_ms=5_000.0))

    def shifted(numbers):
        return format_srt([
            Cue(number=c.number, order=c.order,
                start_ms=c.start_ms + (2_500 if c.number in numbers else 0),
                end_ms=c.end_ms + (2_500 if c.number in numbers else 0), text=c.text)
            for c in base])

    clustered = verify_candidate(base_text, reference_text, shifted({13, 14, 15}),
                                 candidate_algorithm="two_point")
    assert clustered.proposed_verdict == "REJECTED"
    assert any("concentrated_error_cluster" in reason
               for reason in clustered.failure_reasons)
    scattered = verify_candidate(base_text, reference_text, shifted({6, 16, 26}),
                                 candidate_algorithm="two_point")
    assert scattered.proposed_verdict == "VERIFIED", scattered.failure_reasons
    assert scattered.residual_statistics["max"] <= 3000.0


def test_generic_repeat_trap_insufficient() -> None:
    def _fmt(ms: int) -> str:
        return f"{ms // 3_600_000:02d}:{(ms // 60_000) % 60:02d}:{(ms // 1_000) % 60:02d},{ms % 1000:03d}"

    def _generic(order: str) -> str:
        words = ["Yes", "No"] if order == "yn" else ["No", "Yes"]
        return "\n\n".join(
            f"{position + 1}\n{_fmt(60_000 + position * 60_000)} --> "
            f"{_fmt(62_000 + position * 60_000)}\n{words[position % 2]}"
            for position in range(8)) + "\n"

    verdict = verify_candidate(_generic("yn"), _generic("ny"), _generic("yn"),
                               candidate_algorithm="two_point")
    assert verdict.proposed_verdict == "UNVERIFIED"
    assert verdict.failure_reasons


def test_sparse_evidence_unverified() -> None:
    target_text, reference_text = _linear_pair(6)
    verdict = verify_candidate(target_text, reference_text, target_text,
                               candidate_algorithm="two_point")
    assert verdict.proposed_verdict == "UNVERIFIED"


def test_tampered_and_missing_dialogue_rejected() -> None:
    target_text, reference_text = _linear_pair(12)
    cues = parse_srt(target_text)
    tampered = [Cue(number=c.number, order=c.order, start_ms=c.start_ms, end_ms=c.end_ms,
                    text="tampered line here" if c.number == 3 else c.text) for c in cues]
    verdict = verify_candidate(target_text, reference_text, format_srt(tampered),
                               candidate_algorithm="two_point")
    assert verdict.proposed_verdict == "REJECTED"
    assert any("dialogue_text_modified" in reason for reason in verdict.failure_reasons)

    base = apply_transform(parse_srt(target_text), LinearTransform(scale=1.0, offset_ms=5_000.0))
    dropped = [c for c in base if c.number != 6]
    verdict = verify_candidate(target_text, reference_text, format_srt(dropped),
                               candidate_algorithm="two_point")
    assert verdict.proposed_verdict in ("REJECTED", "UNVERIFIED")


def test_non_monotonic_rejected() -> None:
    target_text, reference_text = _linear_pair(12)
    base = list(apply_transform(parse_srt(target_text),
                                LinearTransform(scale=1.0, offset_ms=5_000.0)))
    bad = list(base)
    bad[5], bad[6] = bad[6], bad[5]
    reordered = [Cue(number=position + 1, order=position, start_ms=c.start_ms,
                     end_ms=c.end_ms, text=c.text) for position, c in enumerate(bad)]
    verdict = verify_candidate(target_text, reference_text, format_srt(reordered),
                               candidate_algorithm="two_point")
    assert verdict.proposed_verdict == "REJECTED"


def _reworded(reference_cues, numbers=(3, 7)):
    """Reword selected reference cues so target/reference differ in content
    (independent timing proof) while keeping exact correspondences."""
    alternates = {number: f"Completely different sentence {position + 1}"
                  for position, number in enumerate(numbers)}
    return [Cue(number=c.number, order=c.order, start_ms=c.start_ms, end_ms=c.end_ms,
                text=alternates.get(c.number, c.text)) for c in reference_cues]


def test_dialogue_clamp_rejected_ad_clamp_flagged() -> None:
    target = build_dialogue(12)
    target_text = format_srt(target)
    # Reword a late cue only: the opening must keep full coverage.
    reference_text = format_srt(_reworded(apply_affine(target, 1.0, 50_000.0), numbers=(11,)))

    def _move(cues, number: int):
        return format_srt([
            Cue(number=c.number, order=c.order,
                start_ms=0 if c.number == number else c.start_ms + 50_000,
                end_ms=(c.end_ms - c.start_ms) if c.number == number else c.end_ms + 50_000,
                text=c.text) for c in cues])

    verdict = verify_candidate(target_text, reference_text,
                               _move(parse_srt(target_text), 1),
                               candidate_algorithm="two_point")
    assert verdict.proposed_verdict == "REJECTED"
    assert any("dialogue_clamped" in reason for reason in verdict.failure_reasons)

    ad = [Cue(number=c.number, order=c.order, start_ms=c.start_ms, end_ms=c.end_ms,
              text="Visit www.example-subtitles.com" if c.number == 1 else c.text)
          for c in parse_srt(target_text)]
    ad_text = format_srt(ad)
    ad_ref = format_srt(_reworded(apply_affine(parse_srt(ad_text), 1.0, 50_000.0),
                                 numbers=(11,)))
    verdict = verify_candidate(ad_text, ad_ref, _move(parse_srt(ad_text), 1),
                               candidate_algorithm="two_point")
    assert verdict.proposed_verdict == "VERIFIED", verdict.failure_reasons


def test_cross_language_abstains_with_evidence() -> None:
    arabic = [f"جملة عربية مختلفة تماما رقم {position}" for position in range(10)]
    target = "\n\n".join(
        f"{position + 1}\n00:{1 + position:02d}:00,000 --> 00:{1 + position:02d}:02,000\n{text}"
        for position, text in enumerate(arabic)) + "\n"
    english = build_dialogue(10)
    reference = format_srt(apply_affine(english, 1.0, 2_000.0))
    candidate = format_srt(apply_transform(
        parse_srt(target), LinearTransform(scale=1.0, offset_ms=2_000.0)))
    verdict = verify_candidate(target, reference, candidate, candidate_algorithm="alass")
    assert verdict.proposed_verdict == "UNVERIFIED"
    assert any("cross_language" in reason for reason in verdict.failure_reasons)
    assert verdict.structural_result.get("available") is True


def test_structural_false_high_never_verifies_alone() -> None:
    target_text, _ = _linear_pair(10)
    cues = parse_srt(target_text)
    other = build_dialogue(10, sentences=[f"Unrelated line number {position}" for position in range(10)])
    reference = format_srt([
        Cue(number=c.number, order=c.order, start_ms=c.start_ms, end_ms=c.end_ms, text=o.text)
        for c, o in zip(cues, other, strict=True)])
    verdict = verify_candidate(target_text, reference, target_text,
                               candidate_algorithm="two_point")
    assert verdict.proposed_verdict == "UNVERIFIED"
    assert verdict.structural_result.get("available") is True


def test_wrong_timestamp_similar_reference_insufficient() -> None:
    from tests.two_point_sync._helpers import make_cue as _make_cue

    target, _ = _linear_pair(10)
    arabic = [f"جملة عربية مختلفة تماما رقم {position}" for position in range(10)]
    reference_cues = [
        _make_cue(position + 1, 60_000 + position * 60_000, 62_000 + position * 60_000, text)
        for position, text in enumerate(arabic)
    ]
    verdict = verify_candidate(target, format_srt(reference_cues), target,
                               candidate_algorithm="two_point")
    assert verdict.proposed_verdict == "UNVERIFIED"


def test_alass_success_wrong_output_rejected() -> None:
    target = build_dialogue(14, step_ms=60_000)
    reference = apply_affine(target, 1.0, 5_000.0)
    reworded = {2: "Completely different sentence one",
                13: "Completely different sentence two"}
    shifted = [Cue(number=c.number, order=c.order,
                   start_ms=c.start_ms + (45_000 if 5 <= position < 9 else 0),
                   end_ms=c.end_ms + (45_000 if 5 <= position < 9 else 0),
                   text=reworded.get(c.number, c.text))
               for position, c in enumerate(reference)]
    target_text, shifted_text = format_srt(target), format_srt(shifted)
    candidate = _true_candidate(target_text, 1.0, 5_000.0)
    verdict = verify_candidate(target_text, shifted_text, candidate,
                               candidate_algorithm="alass")
    assert verdict.proposed_verdict == "REJECTED", verdict.failure_reasons


def test_two_point_right_class_wrong_timing_rejected() -> None:
    target_text, reference_text = _linear_pair()
    candidate = _true_candidate(target_text, 1.0, 60_000.0)
    verdict = verify_candidate(target_text, reference_text, candidate,
                               candidate_algorithm="two_point")
    assert verdict.proposed_verdict == "REJECTED"


def test_internal_error_fails_closed() -> None:
    target_text, reference_text = _linear_pair()
    import app.services.sync.unified_verification as module

    original = module.detect_anchors
    try:
        def _boom(*args, **kwargs):
            raise RuntimeError("boom")

        module.detect_anchors = _boom
        verdict = verify_candidate(target_text, reference_text, target_text,
                                   candidate_algorithm="two_point")
    finally:
        module.detect_anchors = original
    assert verdict.proposed_verdict == "UNVERIFIED"
    assert any("correspondence_error" in reason for reason in verdict.failure_reasons)


def test_to_evaluation_needs_binding_and_may_serve() -> None:
    from app.services.sync.alignment import may_serve_synchronized

    target_text, reference_text = _linear_pair()
    candidate = _true_candidate(target_text, 1.0, 5_000.0)
    verdict = verify_candidate(target_text, reference_text, candidate,
                               candidate_algorithm="two_point")
    assert verdict.proposed_verdict == "VERIFIED"
    evaluation = to_evaluation(verdict)
    assert may_serve_synchronized(evaluation.sync_state, evaluation.verification)
    bad = verify_candidate(target_text, reference_text,
                           _true_candidate(target_text, 1.0, 60_000.0),
                           candidate_algorithm="two_point")
    assert bad.proposed_verdict == "REJECTED"
    denied = to_evaluation(bad)
    assert not may_serve_synchronized(denied.sync_state, denied.verification)


def test_cache_identity_segments_methodology() -> None:
    from app.services.sync.unified_verification import cache_identity

    target_text, reference_text = _linear_pair()
    candidate = _true_candidate(target_text, 1.0, 5_000.0)
    first = cache_identity(target_text=target_text, reference_text=reference_text,
                           candidate_text=candidate,
                           video_binding={"video": "tt0773262", "binding": "strong"})
    assert first["verification_method"] == "unified"
    assert first["verification_version"] == "unified-v1"
    assert first["candidate_sha256"] == hashlib.sha256(candidate.encode()).hexdigest()
    altered = cache_identity(target_text=target_text, reference_text=reference_text,
                             candidate_text=candidate + " ",
                             video_binding={"video": "tt0773262", "binding": "strong"})
    assert altered["candidate_sha256"] != first["candidate_sha256"]


def test_module_has_no_provider_or_alass_imports() -> None:
    tree = ast.parse(pathlib.Path("app/services/sync/unified_verification.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    blob = "\n".join(sorted(imported))
    assert "provider" not in blob and "subprocess" not in blob and "httpx" not in blob
    assert "alass" not in blob.replace("candidate_algorithm", "")


def test_telemetry_carries_no_dialogue() -> None:
    target_text, reference_text = _linear_pair()
    candidate = _true_candidate(target_text, 1.0, 5_000.0)
    verdict = verify_candidate(target_text, reference_text, candidate,
                               candidate_algorithm="two_point")
    blob = repr((verdict.coverage, verdict.regional_result, verdict.risk_findings,
                 verdict.failure_reasons, verdict.temporal_result, verdict.structural_result))
    assert "lighthouse" not in blob


def test_single_parse_per_document() -> None:
    import app.services.sync.unified_verification as module

    target_text, reference_text = _linear_pair(12)
    candidate = _true_candidate(target_text, 1.0, 5_000.0)
    calls: list[str] = []
    original = module.parse_srt

    def _counting(text: str):
        calls.append(text[:16])
        return original(text)

    module.parse_srt = _counting
    try:
        verdict = module.verify_candidate(target_text, reference_text, candidate,
                                          candidate_algorithm="two_point")
    finally:
        module.parse_srt = original
    assert verdict.proposed_verdict == "VERIFIED"
    assert len(calls) == 3, f"expected one parse per document, got {len(calls)}"


def test_runtime_within_shadow_budget() -> None:
    target, reference = _read(DEX_TARGET), _read(DEX_FRAMESTOR)
    candidate = _true_candidate  # placeholder replaced below
    _ = candidate
    import time as _time

    target_cues = parse_srt(target)
    from two_point_sync.parser import format_srt as _format
    from two_point_sync.transform import LinearTransform as _LT
    from two_point_sync.transform import apply_transform as _apply

    shifted = _format(_apply(target_cues, _LT(scale=1.0, offset_ms=10_000.0)))
    started = _time.monotonic()
    verdict = verify_candidate(target, reference, shifted, candidate_algorithm="alass")
    elapsed_ms = (_time.monotonic() - started) * 1000.0
    assert elapsed_ms < 10_000.0, f"unified verification took {elapsed_ms:.0f}ms"
    assert verdict.proposed_verdict in ("VERIFIED", "REJECTED", "UNVERIFIED")
