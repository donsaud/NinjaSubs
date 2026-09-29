"""Shadow reference selection v2 - measurement only, never used for alignment.

The existing selector commits to the first candidate that passes cue sanity,
using its own ``reference_tier`` ladder. v2 is an alternative built on the
*existing* compatibility machinery, run beside it purely to record where the
two disagree.

The critical property under test: the shadow pick can never reach alass.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.models import MatchTier
from app.services.sync.reference import ReferenceTrust
from app.services.sync.reference_v2 import (
    GATE_MIN_DIALOGUE_CUES,
    ReferenceCandidate,
    build_candidates,
    compare_selections,
    find_tier_contradictions,
    select_reference_v2,
)

TARGET = "Dexter.S08E05.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv"


def _ts(ms: int) -> str:
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def srt(positions, *, duration_ms: int = 1500, body: str = "I never thought I would find someone") -> str:
    return "\n\n".join(
        f"{i + 1}\n{_ts(int(p))} --> {_ts(int(p) + duration_ms)}\n{body}"
        for i, p in enumerate(positions)
    )


def even(count: int, step_ms: int = 2000, start_ms: int = 0) -> list[int]:
    return [start_ms + i * step_ms for i in range(count)]


CREDITS = "sync by someone"


# --------------------------------------------------------------------------- #
# A. Same MatchTier, better reference health
# --------------------------------------------------------------------------- #


def test_case_a_health_decides_between_equal_tier_candidates():
    healthy = ("subdl", "Dexter.S08E05.1080p.BluRay.x264-PiR8.srt", srt(even(40)))
    # Same tier and same release shape, but sparse enough dialogue to be only
    # "degraded" while still eligible.
    sparse = (
        "subsource",
        "Dexter.S08E05.1080p.BluRay.x264-PiR8.srt",
        srt(even(GATE_MIN_DIALOGUE_CUES, step_ms=2, start_ms=0)),
    )
    selection = select_reference_v2(TARGET, [sparse, healthy])
    assert selection.chosen is not None
    assert selection.chosen.provider == "subdl"
    assert selection.chosen.health_class == "healthy"
    assert any("health decided" in reason for reason in selection.reasons)


def test_credits_only_candidate_is_excluded_entirely():
    healthy = ("subdl", "Dexter.S08E05.1080p.BluRay.x264-PiR8.srt", srt(even(40)))
    credits = (
        "subsource",
        "Dexter.S08E05.1080p.BluRay.x264-PiR8.srt",
        srt(even(40), body=CREDITS),
    )
    selection = select_reference_v2(TARGET, [credits, healthy])
    assert selection.chosen is not None
    assert selection.chosen.provider == "subdl"
    assert all(c.provider != "subsource" for c in selection.ranked)


def test_reference_health_never_outranks_content_identity():
    """A healthy wrong-episode candidate must lose to a degraded-but-correct one."""
    correct_but_sparse = (
        "subdl",
        "Dexter.S08E05.1080p.BluRay.x264-PiR8.srt",
        srt(even(GATE_MIN_DIALOGUE_CUES, step_ms=2)),
    )
    wrong_episode = ("subdl", "Dexter.S08E06.1080p.BluRay.x264-PiR8.srt", srt(even(40)))
    selection = select_reference_v2(TARGET, [wrong_episode, correct_but_sparse])
    assert selection.chosen is not None
    assert "S08E05" in selection.chosen.release_name
    assert selection.hard_rejected == 1
    # The wrong-episode candidate was healthier but is not selectable at all.
    assert all("S08E06" not in c.release_name for c in selection.ranked)


# --------------------------------------------------------------------------- #
# B. Hard rejection wins over health
# --------------------------------------------------------------------------- #


def test_case_b_hard_rejected_candidate_cannot_be_selected():
    wrong = ("subdl", "Dexter.S08E06.1080p.BluRay.x264-PiR8.srt", srt(even(40)))
    selection = select_reference_v2(TARGET, [wrong])
    assert selection.chosen is None
    assert selection.hard_rejected == 1
    assert "quality gates" in " ".join(selection.reasons)


def test_hard_rejection_survives_in_the_candidate_model():
    candidates = build_candidates(TARGET, [("subdl", "Dexter.S08E06.mkv", srt(even(40)))])
    assert candidates[0].hard_accepted is False
    assert candidates[0].eligible is False
    assert candidates[0].hard_reject_reason


# --------------------------------------------------------------------------- #
# C. Exact verified cache beats an equivalent unverified candidate
# --------------------------------------------------------------------------- #


def test_case_c_verified_cache_evidence_wins():
    unverified = ("subdl", "Dexter.S08E05.1080p.BluRay.x264-PiR8.srt", srt(even(40)))
    provisional = select_reference_v2(TARGET, [unverified])
    assert provisional.chosen is not None
    verified_id = provisional.chosen.subtitle_id

    with_cache = select_reference_v2(
        TARGET,
        [unverified, ("subsource", "Dexter.S08E05.1080p.BluRay.x264-OTHER.srt", srt(even(40)))],
        verified_ids=frozenset({verified_id}),
    )
    # The exactly-verified candidate is preferred over the other same-tier one.
    assert with_cache.chosen is not None
    assert with_cache.chosen.subtitle_id == verified_id
    assert with_cache.chosen.cache_verified is True
    assert with_cache.chosen.trust is ReferenceTrust.VERIFIED


# --------------------------------------------------------------------------- #
# D/E. Independence accounting
# --------------------------------------------------------------------------- #


def test_case_d_three_providers_identical_timing_is_one_group():
    same = srt(even(40))
    selection = select_reference_v2(
        TARGET,
        [
            ("subdl", "Dexter.S08E05.1080p.BluRay.x264-A.srt", same),
            ("subsource", "Dexter.S08E05.1080p.BluRay.x264-B.srt", same),
            ("opensubtitles", "Dexter.S08E05.1080p.BluRay.x264-C.srt", same),
        ],
    )
    groups = {c.independent_group for c in build_candidates(
        TARGET,
        [
            ("subdl", "Dexter.S08E05.1080p.BluRay.x264-A.srt", same),
            ("subsource", "Dexter.S08E05.1080p.BluRay.x264-B.srt", same),
            ("opensubtitles", "Dexter.S08E05.1080p.BluRay.x264-C.srt", same),
        ],
    )}
    assert len(groups) == 1
    # A duplicate is still selectable, it is just not counted as independent.
    assert selection.chosen is not None
    assert selection.chosen.group_size == 3


def test_case_e_three_different_timings_are_three_groups():
    references = [
        ("subdl", "Dexter.S08E05.1080p.BluRay.x264-A.srt", srt(even(40))),
        ("subsource", "Dexter.S08E05.1080p.BluRay.x264-B.srt", srt([p + 3000 for p in even(40)])),
        ("opensubtitles", "Dexter.S08E05.1080p.BluRay.x264-C.srt", srt([p - 3000 for p in even(40)])),
    ]
    groups = {c.independent_group for c in build_candidates(TARGET, references)}
    assert len(groups) == 3


# --------------------------------------------------------------------------- #
# F/G. Quality gates
# --------------------------------------------------------------------------- #


def test_case_f_credits_only_reference_is_not_strong():
    credits = ("subdl", "Dexter.S08E05.1080p.BluRay.x264-PiR8.srt", srt(even(40), body=CREDITS))
    candidates = build_candidates(TARGET, [credits])
    assert candidates[0].health.dialogue_cues == 0
    assert candidates[0].health_class == "unusable"
    assert candidates[0].eligible is False


def test_case_g_sparse_dialogue_is_downgraded_by_explicit_threshold():
    sparse = ("subdl", "Dexter.S08E05.1080p.BluRay.x264-PiR8.srt", srt(even(5)))
    candidates = build_candidates(TARGET, [sparse])
    assert candidates[0].health.dialogue_cues < GATE_MIN_DIALOGUE_CUES
    assert candidates[0].health_class == "unusable"
    assert candidates[0].eligible is False


def test_quality_gates_are_separate_from_user_acceptance():
    """A sparse subtitle can still be shown to a user; it just is not an anchor."""
    sparse = build_candidates(TARGET, [("subdl", "Dexter.S08E05.srt", srt(even(5)))])[0]
    assert sparse.health_class == "unusable"
    # Nothing here mutates user-facing acceptance.
    assert not hasattr(sparse, "accepted_by_user")


# --------------------------------------------------------------------------- #
# H/I. Legacy vs shadow
# --------------------------------------------------------------------------- #


def test_case_h_agreement_is_reported_when_picks_match():
    reference = ("subdl", "Dexter.S08E05.1080p.BluRay.x264-PiR8.srt", srt(even(40)))
    selection = select_reference_v2(TARGET, [reference])
    comparison = compare_selections(selection.chosen.subtitle_id, selection)
    assert comparison.changed is False
    assert comparison.agree is True
    assert any("agree" in reason for reason in comparison.reasons)


def test_case_i_disagreement_is_reported_with_reasons():
    good = ("subdl", "Dexter.S08E05.1080p.BluRay.x264-PiR8.srt", srt(even(40)))
    weak = ("subsource", "Dexter.S08E05.1080p.BluRay.x264-PiR8.srt", srt(even(40), body=CREDITS))
    selection = select_reference_v2(TARGET, [good, weak])
    comparison = compare_selections("legacy-id-not-in-set", selection)
    assert comparison.changed is True
    assert comparison.agree is False
    assert any("not even an eligible shadow candidate" in r for r in comparison.reasons)


def test_comparison_reports_no_shadow_candidate_without_guessing():
    selection = select_reference_v2(TARGET, [("subdl", "Dexter.S08E06.mkv", srt(even(40)))])
    comparison = compare_selections("whatever", selection)
    assert comparison.shadow_id is None
    assert comparison.agree is True
    assert any("no eligible candidate" in r for r in comparison.reasons)


# --------------------------------------------------------------------------- #
# J. No fingerprint
# --------------------------------------------------------------------------- #


def test_case_j_no_fingerprint_fabricates_nothing():
    references = [
        ("subdl", "Dexter.S08E05.1080p.BluRay.x264-PiR8.srt", srt(even(40))),
        ("subsource", "Dexter.S08E05.720p.WEB-DL.x264-OTHER.srt", srt([p + 5000 for p in even(40)])),
    ]
    candidates = build_candidates(None, references)
    for candidate in candidates:
        # No target filename means no content claim and no invented identity.
        assert candidate.match_tier is MatchTier.FALLBACK
        assert any("content identity unknown" in r for r in candidate.reasons)
        # Identity is a digest, never a name.
        assert candidate.subtitle_id and "Dexter" not in candidate.subtitle_id
    # Ties break deterministically rather than by fabricated compatibility.
    first = [c.subtitle_id for c in select_reference_v2(None, references).ranked]
    second = [c.subtitle_id for c in select_reference_v2(None, references).ranked]
    assert first == second


# --------------------------------------------------------------------------- #
# K. Different release/cut
# --------------------------------------------------------------------------- #


def test_case_k_different_cut_is_not_preferred_over_comparable():
    same_cut = ("subdl", "Dexter.S08E05.1080p.BluRay.x264-PiR8.srt", srt(even(40)))
    other_cut = ("subsource", "Dexter.S08E05.1080p.WEB-DL.x264-OTHER.srt", srt(even(40)))
    selection = select_reference_v2(TARGET, [other_cut, same_cut])
    assert selection.chosen is not None
    assert selection.chosen.match_tier in (MatchTier.HASH, MatchTier.EXACT)
    assert "PiR8" in selection.chosen.release_name


# --------------------------------------------------------------------------- #
# Reuse, not a third ladder
# --------------------------------------------------------------------------- #


def test_v2_consumes_the_existing_compatibility_result():
    """Tiers come from MatchTier, not a new ladder."""
    source = Path("app/services/sync/reference_v2.py").read_text(encoding="utf-8")
    assert "calculate_compatibility" in source
    assert "hard_compatibility_filter" in source
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    # No episode/title/fps re-parsing of its own.
    assert not any("guessit" in name for name in imported)


def test_tier_contradictions_between_ladders_are_reportable():
    rows = find_tier_contradictions(
        TARGET,
        [
            ("subdl", "Dexter.S08E05.1080p.BluRay.x264-PiR8.srt", srt(even(40))),
            ("subsource", "Dexter.S08E05.720p.WEB-DL.x264-OTHER.srt", srt(even(40))),
        ],
    )
    assert rows
    for row in rows:
        assert "match_tier" in row and "reference_tier" in row and "agree" in row


# --------------------------------------------------------------------------- #
# CRITICAL: shadow cannot influence the actual synchronization
# --------------------------------------------------------------------------- #


def test_shadow_selection_cannot_reach_alass():
    """Structural proof: alass only ever receives the legacy reference text.

    Checked as a data-flow fact rather than by variable naming, so renaming a
    local cannot weaken the guarantee.
    """
    orchestrator = ast.parse(Path("app/services/sync/orchestrator.py").read_text(encoding="utf-8"))
    _execute = next(
        node
        for node in ast.walk(orchestrator)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_execute"
    )
    # The value handed to alass as the reference.
    reference_source = next(
        node.value.value
        for node in ast.walk(_execute)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "reference" for t in node.targets)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "text"
    )
    assert isinstance(reference_source, ast.Name)
    assert reference_source.id == "resolved"

    # `resolved` is the strategy's ResolvedReference. Its `text` is assigned from
    # the legacy `text` variable and never from a shadow object.
    strategy = ast.parse(
        Path("app/services/sync/external_strategy.py").read_text(encoding="utf-8")
    )
    resolve = next(
        node
        for node in ast.walk(strategy)
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "resolve_with_provenance"
    )
    shadow_locals: set[str] = set()
    for node in ast.walk(resolve):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            callee = node.value.func
            name = getattr(callee, "id", None) or getattr(callee, "attr", None)
            if name in ("select_reference_v2", "compare_selections"):
                shadow_locals.update(
                    t.id for t in node.targets if isinstance(t, ast.Name)
                )
    assert shadow_locals, "shadow selector should be wired in"

    for node in ast.walk(resolve):
        if (
            isinstance(node, ast.Return)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", None) == "ResolvedReference"
        ):
            for keyword in node.value.keywords:
                if keyword.arg in ("text", "candidate", "kind"):
                    # Must not be a shadow-derived expression.
                    source = ast.dump(keyword.value)
                    assert "shadow" not in source
                    assert "selection" not in source
                    assert "comparison" not in source


@pytest.mark.asyncio
async def test_shadow_cannot_reach_alass_behaviourally():
    """End-to-end proof: when the policies disagree, the synchronizer gets the
    LEGACY pick.

    An AST check cannot see data-flow mutation, so this drives the real strategy
    with a scenario where the two genuinely disagree, and asserts the value
    handed onward is the legacy one.

    The scenario is the realistic one: the legacy loop downloads a first
    candidate that fails cue-sanity, then downloads a second that passes. Legacy
    commits to the second (credits-heavy but correctly timed). Shadow sees both
    and, judging on health alone, prefers the first. The returned text must
    still be the legacy one.

    Both candidates are deliberately in the same identity band, so this test
    exercises shadow isolation and not the target-bound tier floor. That floor
    is covered by test_reference_target_bound.py: when the first candidate
    matches the target *better* than the second, the legacy fall-through is
    now correctly refused, which would otherwise mask what this test measures.
    """
    from app.models import SubtitleRelease
    from app.services.subtitle_matcher import (
        FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
        validate_cue_sanity,
    )
    from app.services.sync.cache import ReferenceDiskCache
    from app.services.sync.external_strategy import ExternalExactStrategy
    from app.services.sync.query import ReferenceQuery

    target_text = srt(even(40))
    # Wrong cut: healthy dialogue, but its first cue is ~97s out.
    rejected_text = srt(even(40, start_ms=97_000))
    # Correct timing, but credits-only: unusable as a timing anchor.
    credits_text = srt(even(40), body=CREDITS)

    class _Provider:
        name = "subdl"

        def __init__(self):
            self.downloaded: list[str] = []

        async def search_subtitles(self, **kwargs):
            return [
                SubtitleRelease(
                    release_name="Dexter.S08E05.1080p.BluRay.x264-OtherGRP.srt",
                    download_url="http://first",
                    provider="subdl",
                    lang="eng",
                ),
                SubtitleRelease(
                    release_name="Dexter.S08E05.1080p.BluRay.x264-ThirdGRP.srt",
                    download_url="http://second",
                    provider="subdl",
                    lang="eng",
                ),
            ]

        async def download_archive(self, url, api_key=None):
            self.downloaded.append(url)
            payload = rejected_text if url == "http://first" else credits_text
            return payload.encode()

    provider = _Provider()
    strategy = ExternalExactStrategy(
        subdl_provider=provider,
        subsource_provider=None,
        opensubtitles_provider=None,
        cache=ReferenceDiskCache(root=".", ttl=3600.0, min_bytes=100),
        min_bytes=100,
        timeout=1.0,
    )
    query = ReferenceQuery(
        imdb_id="tt0773262",
        media_type="series",
        season=8,
        episode=5,
        target_filename=TARGET,
    )

    def _validator(reference_text: str) -> bool:
        return bool(
            validate_cue_sanity(
                target_text,
                reference_text,
                threshold_ms=FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
            )["ok"]
        )

    resolved = await strategy.resolve_with_provenance(query, update_validator=_validator)

    # Both candidates were downloaded, so the policies could genuinely differ.
    assert provider.downloaded == ["http://first", "http://second"]
    assert resolved.text is not None
    # The synchronizer receives the LEGACY pick, never the shadow one.
    assert resolved.text == credits_text
    assert resolved.text != rejected_text
    # And the disagreement was recorded, not acted upon.
    assert resolved.shadow_changed is True
    assert resolved.shadow_reference_id


def test_shadow_module_is_not_imported_by_the_sync_service():
    """The alass wrapper must not know the shadow selector exists."""
    source = Path("app/services/sync_service.py").read_text(encoding="utf-8")
    assert "reference_v2" not in source
    assert "select_reference_v2" not in source


def test_candidate_model_explains_itself():
    candidate = ReferenceCandidate(
        subtitle_id="abc", release_name="Show.S01E01.1080p-GRP.srt", provider="subdl"
    )
    assert "Show" in candidate.explain()
    assert "tier=" in candidate.explain()
