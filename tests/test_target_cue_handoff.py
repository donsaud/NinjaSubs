"""Target cue data flow: sync service and verifier must agree on the target.

A real Dexter S08E05 production trace contained two contradictory facts about
one subtitle:

    [sync] pre-alass timing: target_cue_count=584
    [sync] verifier input:  target has 0 cues, below the 5 needed to verify

The same string was measured by both, so the target was not lost between them.

Root cause: two parsers with two accepted timestamp grammars. The sync path
parses via ``_TIMESPAN_LINE_REGEX`` / ``_cue_starts_ms``, which treat the hours
field as optional (``(?:\\d{1,2}:)?\\d{1,2}:\\d{2}``). ``parse_srt_cues`` matched
``\\d{1,2}:\\d{2}:\\d{2}`` with hours MANDATORY, so a provider SRT written as
``MM:SS,mmm`` counted as every cue upstream and as ZERO cues at the verifier.
``_to_ms`` inside that same function already handled the two-part form; the
regex was what made its tolerance unreachable.

The repair is two-part and both halves are load-bearing:

* ``parse_srt_cues`` accepts the same grammar as the rest of the sync path.
* the orchestrator parses the target ONCE, at the verifier boundary, and hands
  the collection over instead of the text, so the analyzer never reparses.

These tests assert the fix without asserting any particular verdict. Nothing
here expects the Dexter subtitle to become VERIFIED: the whole point is that the
verifier must be handed the real target and reach its own conclusion.
"""

from __future__ import annotations

import pytest

from app.services.subtitle_matcher import parse_srt_cues
from app.services.sync.alignment import AlignmentAnalyzer
from app.services.sync_service import _cue_starts_ms


def srt(count: int, *, with_hours: bool = True, start_ms: int = 0, step_ms: int = 2000) -> str:
    blocks = []
    for i in range(count):
        start = start_ms + i * step_ms
        end = start + 900
        if with_hours:
            t0 = _hms(start)
            t1 = _hms(end)
        else:
            # MM:SS,mmm - legal input the sync path has always accepted.
            t0 = _ms(start)
            t1 = _ms(end)
        blocks.append(f"{i + 1}\n{t0} --> {t1}\nline {i}\n")
    return "\n".join(blocks)


def _hms(ms: int) -> str:
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{milli:03d}"


def _ms(ms: int) -> str:
    m, rem = divmod(ms, 60_000)
    s, milli = divmod(rem, 1_000)
    return f"{m:02d}:{s:02d},{milli:03d}"


# --- the defect itself ------------------------------------------------------


@pytest.mark.parametrize("with_hours", [True, False])
def test_parse_srt_cues_agrees_with_the_sync_path_parser(with_hours):
    """The invariant that was broken: two parsers, one grammar."""
    text = srt(24, with_hours=with_hours)
    assert len(parse_srt_cues(text)) == 24
    assert len(parse_srt_cues(text)) == len(_cue_starts_ms(text, limit=1000))


def test_two_part_timestamps_used_to_parse_to_zero():
    """Pins the exact regression: MM:SS must not silently vanish."""
    text = srt(24, with_hours=False)
    assert len(parse_srt_cues(text)) == 24
    starts = [cue[0] for cue in parse_srt_cues(text)]
    assert starts == sorted(starts)
    assert starts[0] == 0
    assert starts[-1] == 23 * 2000


# --- Test A: parsed target reaches the verifier -----------------------------


def test_a_parsed_target_reaches_verifier():
    analyzer = AlignmentAnalyzer()
    target = srt(40, with_hours=False)  # the incident's shape
    synced = srt(40, with_hours=False, start_ms=250)

    evaluation = analyzer.analyze(parse_srt_cues(target), synced, target, alass_applied=True)

    assert len(parse_srt_cues(target)) == 40
    # The insufficient-evidence branch is the one the bug tripped.
    assert evaluation.rejection_reason is None or evaluation.rejection_reason.value != (
        "insufficient_evidence"
    )
    assert not any("target has 0 cues" in r for r in evaluation.reasons)


def test_a_orchestrator_hands_cues_not_text():
    """The analyzer receives the collection, so it cannot reparse to zero.

    ``analyze`` accepts either text or cues. Feeding it the list exercises the
    branch the orchestrator now uses, and proves a two-part-timestamp target
    survives the handoff.
    """
    analyzer = AlignmentAnalyzer()
    target = srt(40, with_hours=False)
    synced = srt(40, with_hours=False, start_ms=250)

    from_text = analyzer.analyze(target, synced, target, alass_applied=True)
    from_cues = analyzer.analyze(parse_srt_cues(target), synced, target, alass_applied=True)

    assert from_text.reasons == from_cues.reasons
    assert from_text.sync_state == from_cues.sync_state


# --- Test B: genuinely thin target is still rejected ------------------------


def test_b_thin_target_still_fails_the_evidence_gate():
    analyzer = AlignmentAnalyzer()
    target = srt(3, with_hours=False)  # fewer than min_cues (5)
    synced = srt(3, with_hours=False, start_ms=250)

    evaluation = analyzer.analyze(parse_srt_cues(target), synced, target, alass_applied=True)

    assert evaluation.sync_state.value == "unverified"
    assert evaluation.verification.value == "unknown"
    assert evaluation.rejection_reason is not None
    assert any("target has 3 cues" in r for r in evaluation.reasons)


def test_b_threshold_is_unchanged():
    """The minimum-cue requirement must not have been lowered."""
    from app.services.sync.alignment import MIN_CUES_FOR_VERIFICATION, MIN_CUES_FOR_VERIFIED

    assert MIN_CUES_FOR_VERIFICATION == 5
    assert MIN_CUES_FOR_VERIFIED == 12
    assert AlignmentAnalyzer().min_cues == 5


# --- Test C: genuinely empty target stays unverified ------------------------


def test_c_empty_target_stays_unverified():
    analyzer = AlignmentAnalyzer()
    for target in ("", "not a subtitle at all", srt(0)):
        evaluation = analyzer.analyze(
            parse_srt_cues(target), srt(30), srt(30), alass_applied=True
        )
        assert evaluation.sync_state.value == "unverified"
        assert evaluation.verification.value == "unknown"
        assert evaluation.rejection_reason is not None
        assert any("target has 0 cues" in r for r in evaluation.reasons)


def test_c_malformed_timestamps_stay_unverified():
    analyzer = AlignmentAnalyzer()
    target = "1\nnot-a-timestamp --> also-not\nline\n"
    evaluation = analyzer.analyze(
        parse_srt_cues(target), srt(30), srt(30), alass_applied=True
    )
    assert evaluation.sync_state.value == "unverified"
    assert any("target has 0 cues" in r for r in evaluation.reasons)


# --- Test D: post-Alass handoff --------------------------------------------


def test_d_post_alass_handoff_uses_real_target_cues():
    """Alass produced output; verification must judge it against the real target."""
    analyzer = AlignmentAnalyzer()
    target_cues = parse_srt_cues(srt(40, with_hours=False))
    synced = srt(40, with_hours=False, start_ms=250)

    assert len(target_cues) == 40
    evaluation = analyzer.analyze(
        target_cues, synced, srt(40, with_hours=True), alass_applied=True, alass_successful=True
    )

    # The verifier reached a real measurement, not the empty-target shortcut.
    assert not any("target has 0 cues" in r for r in evaluation.reasons)
    assert evaluation.alass_applied is True
    assert evaluation.alass_successful is True
    assert evaluation.sync_state.value != "unverified" or evaluation.rejection_reason is None


def test_d_synced_cues_are_also_reused_not_reparsed():
    """Both sides of the comparison come from the tolerant grammar."""
    analyzer = AlignmentAnalyzer()
    target = srt(40, with_hours=False)
    synced = srt(40, with_hours=False, start_ms=250)

    assert len(parse_srt_cues(synced)) == 40
    evaluation = analyzer.analyze(
        parse_srt_cues(target), parse_srt_cues(synced), target, alass_applied=True
    )
    assert not any("has 0 cues" in r for r in evaluation.reasons)


# --- no evidence is manufactured --------------------------------------------


def test_empty_cues_list_is_still_rejected():
    """Passing an empty collection must not be a way around the gate."""
    analyzer = AlignmentAnalyzer()
    evaluation = analyzer.analyze([], srt(40), srt(40), alass_applied=True)
    assert evaluation.sync_state.value == "unverified"
    assert any("target has 0 cues" in r for r in evaluation.reasons)


def test_cue_count_alone_is_not_proof():
    """A large count with no synced output still cannot reach VERIFIED."""
    analyzer = AlignmentAnalyzer()
    evaluation = analyzer.analyze(
        parse_srt_cues(srt(600, with_hours=False)), None, srt(600), alass_applied=False
    )
    # No alignment was applied, so the strongest defensible claim is not
    # VERIFIED even with hundreds of cues available.
    assert evaluation.sync_state.value != "verified_resynced"
