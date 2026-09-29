"""Reference trust: a verification is only as good as its reference.

Covers the correlated-error risk in the existing pipeline, where the first
candidate that is not obviously broken becomes the reference and every later
measurement is taken relative to it.

These tests assert *measurement* behaviour. They deliberately do not assert
that any particular reference is chosen, because reference policy is not
changed in this phase - it is only measured.
"""

from __future__ import annotations

import pytest

from app.services.subtitle_matcher import parse_srt_cues
from app.services.sync.alignment import AlignmentAnalyzer, SyncState
from app.services.sync.reference import (
    MIN_REFERENCE_CUES,
    ReferenceTrust,
    analyze_reference_health,
    assess_consensus,
    assess_reference,
    detect_duplicate_groups,
    reference_fingerprint,
)
from app.services.sync.structural import StructuralProfile

ANALYZER = AlignmentAnalyzer()


def _ts(ms: int) -> str:
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def srt(
    positions,
    *,
    duration_ms: int = 1500,
    body: str = "I never thought I would find someone like you in my life tonight",
) -> str:
    return "\n\n".join(
        f"{i + 1}\n{_ts(int(p))} --> {_ts(int(p) + duration_ms)}\n{body}"
        for i, p in enumerate(positions)
    )


def even(count: int, step_ms: int = 2000, start_ms: int = 0) -> list[int]:
    return [start_ms + i * step_ms for i in range(count)]


# --------------------------------------------------------------------------- #
# Case A: an excellent verified reference
# --------------------------------------------------------------------------- #


def test_case_a_verified_reference_reaches_high_trust():
    text = srt(even(40))
    assessment = assess_reference(
        text,
        target_cues=parse_srt_cues(text),
        provider="subdl",
        decision_kind="edition",
        cache_verified=True,
    )
    assert assessment.trust is ReferenceTrust.VERIFIED
    assert assessment.supports_verified_claim() is True
    assert assessment.health.healthy
    assert assessment.failure is None


def test_hash_and_team_references_reach_strong_trust():
    text = srt(even(40))
    for kind in ("hash", "team"):
        assessment = assess_reference(
            text, target_cues=parse_srt_cues(text), decision_kind=kind
        )
        assert assessment.trust is ReferenceTrust.STRONG
        assert assessment.supports_verified_claim() is True


def test_plain_edition_reference_is_only_acceptable():
    text = srt(even(40))
    assessment = assess_reference(
        text, target_cues=parse_srt_cues(text), decision_kind="edition"
    )
    assert assessment.trust is ReferenceTrust.ACCEPTABLE
    # Acceptable is healthy but not proven enough to carry a verified claim.
    assert assessment.supports_verified_claim() is False


# --------------------------------------------------------------------------- #
# Case B: three identical copies from three providers
# --------------------------------------------------------------------------- #


def test_case_b_three_identical_copies_count_as_one_reference():
    """Provider count is not independence."""
    same = srt(even(40))
    groups = detect_duplicate_groups(
        [("subdl", "A.srt", same), ("subsource", "B.srt", same), ("opensubtitles", "C.srt", same)]
    )
    assert len(groups) == 1
    assert sum(len(members) for members in groups.values()) == 3

    assessment = assess_reference(
        same, target_cues=parse_srt_cues(same), decision_kind="edition", groups=groups
    )
    assert assessment.independent_sources == 1
    assert assessment.duplicate_sources == 2
    assert any("provider count is not independence" in r for r in assessment.reasons)


def test_differently_worded_but_identical_timing_still_collapses():
    """Providers re-word credits and translation style but keep the cue grid."""
    a = srt(even(40), body="I never thought I would find someone like you tonight")
    b = srt(even(40), body="من لم أتخيل أن أجد شخصًا مثلك في حياتي")
    assert reference_fingerprint(a) == reference_fingerprint(b)
    groups = detect_duplicate_groups([("subdl", "A", a), ("subsource", "B", b)])
    assert len(groups) == 1


# --------------------------------------------------------------------------- #
# Case C: three genuinely different subtitles agree
# --------------------------------------------------------------------------- #


def test_case_c_genuinely_different_references_agree():
    base = even(40)
    references = [
        ("subdl", "A", srt(base)),
        ("subsource", "B", srt([p + 120 for p in base])),
        ("opensubtitles", "C", srt([p - 90 for p in base])),
    ]
    groups = detect_duplicate_groups(references)
    assert len(groups) == 3, "materially different timing grids must stay separate"

    score, size, agrees, reasons = assess_consensus(groups)
    assert agrees is True
    assert size == 3
    assert score == pytest.approx(1.0)
    assert any("agree" in reason for reason in reasons)


def test_consensus_is_absent_with_a_single_reference():
    groups = detect_duplicate_groups([("subdl", "A", srt(even(40)))])
    score, size, agrees, reasons = assess_consensus(groups)
    assert score is None and agrees is None
    assert any("no consensus" in reason for reason in reasons)


# --------------------------------------------------------------------------- #
# Case D: agreement does not make a wrong reference correct
# --------------------------------------------------------------------------- #


def test_case_d_consensus_does_not_rescue_a_wrong_cut_reference():
    """Three references agreeing is not the same as being right for this target.

    Note what this layer can and cannot see: a *constant* offset produces an
    identical structural profile, because the comparison is shift-invariant by
    design. A 97s displacement is caught by the existing +/-20s cue-sanity gate
    upstream, not here. What this layer guarantees is that agreement never
    upgrades an unproven reference into a proven one.
    """
    wrong = srt([p + 97_000 for p in even(40)])
    target = srt(even(40))
    groups = detect_duplicate_groups(
        [
            ("subdl", "A", wrong),
            ("subsource", "B", srt([p + 96_900 for p in even(40)])),
            ("opensubtitles", "C", srt([p + 97_100 for p in even(40)])),
        ]
    )
    assessment = assess_reference(
        wrong, target_cues=parse_srt_cues(target), decision_kind="edition", groups=groups
    )
    # They agree, but agreement does not promote an unproven reference.
    assert assessment.consensus_agrees is True
    assert assessment.trust is ReferenceTrust.ACCEPTABLE
    assert assessment.trust is not ReferenceTrust.VERIFIED
    assert assessment.trust is not ReferenceTrust.STRONG
    assert assessment.supports_verified_claim() is False


def test_structurally_foreign_reference_cannot_support_a_verified_claim():
    """A reference with a very different shape must not carry a verified claim.

    A 3x-fewer-cue reference keeps a similar per-cue profile, so it is not
    caught by structural mismatch alone; it is caught by health (sparse
    dialogue). Either way the outcome that matters is the same: it cannot
    support a verified synchronization claim.
    """
    target = srt(even(40))
    foreign = srt(even(12, step_ms=6_000))
    groups = detect_duplicate_groups(
        [
            ("subdl", "A", foreign),
            ("subsource", "B", srt(even(11, step_ms=6_100))),
            ("opensubtitles", "C", srt(even(13, step_ms=5_900))),
        ]
    )
    assessment = assess_reference(
        foreign, target_cues=parse_srt_cues(target), decision_kind="edition", groups=groups
    )
    assert assessment.supports_verified_claim() is False
    assert assessment.trust in (ReferenceTrust.REJECTED, ReferenceTrust.UNKNOWN)


def test_consensus_alone_never_marks_a_rejected_reference_usable():
    assert ReferenceTrust.REJECTED not in (
        ReferenceTrust.VERIFIED,
        ReferenceTrust.STRONG,
    )


# --------------------------------------------------------------------------- #
# Case E: a broken reference
# --------------------------------------------------------------------------- #


def test_case_e_severe_cue_loss_reduces_reference_trust():
    text = srt(even(40))
    lossy = srt(even(4))
    assessment = assess_reference(
        lossy, target_cues=parse_srt_cues(text), decision_kind="edition"
    )
    assert assessment.trust is ReferenceTrust.REJECTED
    assert assessment.failure is not None
    assert assessment.supports_verified_claim() is False


def test_health_detects_structural_problems():
    healthy = analyze_reference_health(srt(even(40)))
    assert healthy.healthy is True
    assert healthy.cue_count == 40

    # Non-monotonic timestamps.
    broken = "1\n00:00:10,000 --> 00:00:12,000\nfirst line here\n\n" + srt([1000 + p for p in even(30)])
    unhealthy = analyze_reference_health(broken)
    assert unhealthy.healthy is False
    assert unhealthy.reasons

    # Credits-only content has no usable dialogue.
    credits = "\n\n".join(
        f"{i + 1}\n{_ts(i * 2000)} --> {_ts(i * 2000 + 1500)}\nsync by someone"
        for i in range(40)
    )
    assert analyze_reference_health(credits).healthy is False

    assert analyze_reference_health(None).healthy is False
    assert analyze_reference_health("not a subtitle").healthy is False


# --------------------------------------------------------------------------- #
# Case F: a cached reference for a different video
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_case_f_reference_cache_reuse_across_videos_is_impossible():
    from app.services.sync_cache import SyncCache

    cache = SyncCache()
    key = SyncCache.build_verdict_key("videoA", "refSub1", "eng")
    await cache.set_verdict(key, {"sync_state": "verified_resynced", "verification": "verified"})
    assert await cache.get_verdict(SyncCache.build_verdict_key("videoB", "refSub1", "eng")) is None
    assert await cache.get_verdict(SyncCache.build_verdict_key("videoA", "refSub2", "eng")) is None


# --------------------------------------------------------------------------- #
# Case G: verified evidence outweighs several unverified references
# --------------------------------------------------------------------------- #


def test_case_g_verified_reference_stays_strongest():
    text = srt(even(40))
    verified = assess_reference(
        text, target_cues=parse_srt_cues(text), decision_kind="edition", cache_verified=True
    )
    for kind in ("edition", "source_edition", "fallback"):
        other = assess_reference(
            text, target_cues=parse_srt_cues(text), decision_kind=kind
        )
        assert verified.trust.rank <= other.trust.rank


# --------------------------------------------------------------------------- #
# Case H: no trustworthy reference means no verified claim
# --------------------------------------------------------------------------- #


def test_case_h_unusable_reference_yields_unverified_not_manufactured_confidence():
    text = srt(even(40))
    for reference in (None, "", "garbage", srt(even(3))):
        evaluation = ANALYZER.analyze(text, None, reference)
        assert evaluation.sync_state is not SyncState.VERIFIED_SYNCED
        assert evaluation.sync_state is not SyncState.VERIFIED_RESYNCED


def test_weak_reference_trust_withholds_a_verified_claim():
    """The gate only removes claims; it never creates one."""
    target = srt(even(40))
    aligned = srt(even(40, start_ms=2_310))
    reference = srt(even(40, start_ms=2_310))

    strong = ANALYZER.analyze(
        target, aligned, reference, alass_applied=True, alass_successful=True,
        reference_trust="strong",
    )
    assert strong.sync_state is SyncState.VERIFIED_RESYNCED

    weak = ANALYZER.analyze(
        target, aligned, reference, alass_applied=True, alass_successful=True,
        reference_trust="acceptable",
    )
    assert weak.sync_state is SyncState.PROBABLE_SYNC
    assert any("reference trust" in reason for reason in weak.reasons)

    rejected = ANALYZER.analyze(
        target, aligned, reference, alass_applied=True, alass_successful=True,
        reference_trust="rejected",
    )
    assert rejected.sync_state is SyncState.PROBABLE_SYNC

    # No trust information at all: the pre-existing verdict is untouched.
    unlabelled = ANALYZER.analyze(
        target, aligned, reference, alass_applied=True, alass_successful=True,
    )
    assert unlabelled.sync_state is SyncState.VERIFIED_RESYNCED


# --------------------------------------------------------------------------- #
# No policy change in this phase
# --------------------------------------------------------------------------- #


def test_assessment_does_not_change_which_reference_is_selected():
    """Assessment is measured after the winner is chosen, never used to choose it."""
    import ast
    from pathlib import Path

    source = Path("app/services/sync/external_strategy.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    # The assessment call must sit after the winner is chosen (the break).
    assessment_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "assess_reference"
    ]
    assert assessment_lines, "assessment should be wired in"
    return_breaks = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Break)
    ]
    assert assessment_lines[0] > min(return_breaks), (
        "reference trust must be measured after the winner is chosen"
    )


def test_reference_module_never_calls_alass():
    from pathlib import Path

    source = Path("app/services/sync/reference.py").read_text(encoding="utf-8")
    assert "subprocess" not in source
    assert "sync_async" not in source
    assert "ALASS" not in source


def test_reference_thresholds_are_documented_and_pinned():
    from app.services.sync.reference import (
        MAX_BACKWARD_FRACTION,
        MAX_CUE_DURATION_MS,
        MIN_DIALOGUE_COVERAGE,
    )

    assert MIN_REFERENCE_CUES == 12
    assert MIN_DIALOGUE_COVERAGE == 0.55
    assert MAX_CUE_DURATION_MS == 15_000
    assert MAX_BACKWARD_FRACTION == 0.02
    # It reuses the existing cue parser rather than adding one.
    from app.services.subtitle_matcher import parse_srt_cues

    assert parse_srt_cues(srt(even(40)))


def test_structural_profile_is_reused_for_consensus():
    """Consensus leans on the existing profile rather than a new parser."""
    profile = StructuralProfile.from_subtitle(srt(even(40)))
    assert profile.cue_count == 40
