"""NinjaSync Points shadow mode: measure everything, change nothing.

These tests pin the Phase 2 contract: shadow evaluation observes the exact
(target, reference) pair of a rejected Alass candidate, classifies the NinjaSync
outcome honestly, and cannot alter serving, ranking, cache, or verification --
because it has no code path that does.
"""

from __future__ import annotations

import logging

import pytest

from app.services.sync.ninja_shadow import (
    ALASS_WINS,
    INSUFFICIENT_EVIDENCE,
    NINJA_POINTS_CONSTANT_VERIFIED,
    NINJA_POINTS_LINEAR_VERIFIED,
    NINJA_POINTS_REJECTED,
    REFERENCE_REJECTED,
    SHADOW_ERROR,
    UNSUPPORTED_STRUCTURE,
    counters_snapshot,
    reset_counters,
    run_shadow_evaluation,
)


def _ts(ms: int) -> str:
    h, r = divmod(max(0, int(ms)), 3600000)
    m, r = divmod(r, 60000)
    s, ms2 = divmod(r, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms2:03d}"


def _srt(cues: int, tag: str, offset_ms: int = 0, step_ms: int = 3000) -> str:
    out = []
    for i in range(cues):
        s = 1000 + offset_ms + i * step_ms
        out.append(f"{i + 1}\n{_ts(s)} --> {_ts(s + 2000)}\n{tag} line {i}\n")
    return "\n\n".join(out) + "\n"


@pytest.fixture(autouse=True)
def _clean_counters():
    reset_counters()
    yield
    reset_counters()


# ===========================================================================
# Classification matrix
# ===========================================================================


def test_alass_verified_skips_ninja_entirely():
    target = _srt(30, "alpha")
    reference = _srt(30, "alpha", offset_ms=2000)
    result = run_shadow_evaluation(
        target, reference, alass_state_value="verified_synced", alass_verified=True
    )
    assert result.classification == ALASS_WINS
    assert result.anchor_count == 0
    assert result.eligible_for_future_serving is False


def test_constant_verified_shadow_case():
    target = _srt(30, "alpha")
    reference = _srt(30, "alpha", offset_ms=2000)
    result = run_shadow_evaluation(
        target, reference, alass_state_value="unverified", alass_verified=False
    )
    assert result.classification == NINJA_POINTS_CONSTANT_VERIFIED
    assert result.eligible_for_future_serving is True
    assert result.anchor_count >= 2
    assert result.offset_ms == pytest.approx(2000.0, abs=1.0)


def test_rejected_when_nothing_verifies():
    target = _srt(30, "alpha")
    reference = _srt(30, "beta", offset_ms=5000)
    result = run_shadow_evaluation(
        target, reference, alass_state_value="unverified", alass_verified=False
    )
    assert result.classification in (NINJA_POINTS_REJECTED, INSUFFICIENT_EVIDENCE)
    assert result.eligible_for_future_serving is False


def test_sparse_dialogue_is_insufficient_evidence():
    # Exactly one shared line: support 1 < 2, so detection yields nothing.
    target = (
        "1\n00:00:01,000 --> 00:00:03,000\nthe only shared line here\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\ntarget only line\n\n"
    )
    reference = (
        "1\n00:00:03,000 --> 00:00:05,000\nthe only shared line here\n\n"
        "2\n00:00:07,000 --> 00:00:09,000\nreference only line\n\n"
    )
    result = run_shadow_evaluation(
        target, reference, alass_state_value="unverified", alass_verified=False
    )
    assert result.classification == INSUFFICIENT_EVIDENCE
    assert result.eligible_for_future_serving is False


def test_few_corroborated_anchors_still_fail_verification_safely():
    # Three identical lines DO corroborate into anchors and a model, but the
    # analyzer refuses verified claims below its cue minimum. Safe either way.
    target = _srt(3, "alpha")
    reference = _srt(3, "alpha", offset_ms=2000)
    result = run_shadow_evaluation(
        target, reference, alass_state_value="unverified", alass_verified=False
    )
    assert result.classification in (NINJA_POINTS_REJECTED, INSUFFICIENT_EVIDENCE)
    assert result.eligible_for_future_serving is False


def test_repeated_dialogue_cannot_anchor():
    target = "\n\n".join(
        f"{i + 1}\n{_ts(1000 + i * 3000)} --> {_ts(3000 + i * 3000)}\nYes."
        for i in range(30)
    ) + "\n"
    reference = "\n\n".join(
        f"{i + 1}\n{_ts(3000 + i * 3000)} --> {_ts(5000 + i * 3000)}\nYes."
        for i in range(30)
    ) + "\n"
    result = run_shadow_evaluation(
        target, reference, alass_state_value="unverified", alass_verified=False
    )
    assert result.classification == INSUFFICIENT_EVIDENCE


def test_structural_cut_does_not_verify():
    cues = 120
    target = _srt(cues, "cut")
    ref_lines = []
    for i in range(cues):
        shift = 2000 if i < cues // 2 else 15000
        s = 1000 + i * 3000 + shift
        ref_lines.append(f"{i + 1}\n{_ts(s)} --> {_ts(s + 2000)}\ncut line {i}\n")
    reference = "\n\n".join(ref_lines) + "\n"
    result = run_shadow_evaluation(
        target, reference, alass_state_value="unverified", alass_verified=False
    )
    assert result.classification in (
        NINJA_POINTS_REJECTED,
        REFERENCE_REJECTED,
        UNSUPPORTED_STRUCTURE,
    )
    assert result.eligible_for_future_serving is False


def test_shadow_errors_are_telemetry_not_exceptions():
    class _Boom:
        def analyze(self, *a, **kw):
            raise RuntimeError("verifier exploded")

    # None inputs are handled gracefully by the parser (no anchors); a
    # catastrophic engine failure is what must become SHADOW_ERROR.
    calm = run_shadow_evaluation(
        None, None, alass_state_value="unverified", alass_verified=False  # type: ignore[arg-type]
    )
    assert calm.classification == INSUFFICIENT_EVIDENCE
    import app.services.sync.ninja_points as engine_module

    real_run = engine_module.HybridSyncEngine.run

    def _explodes(self, *a, **kw):
        raise RuntimeError("engine exploded")

    engine_module.HybridSyncEngine.run = _explodes
    try:
        broken = run_shadow_evaluation(
            _srt(30, "alpha"),
            _srt(30, "alpha", offset_ms=2000),
            alass_state_value="unverified",
            alass_verified=False,
        )
    finally:
        engine_module.HybridSyncEngine.run = real_run
    assert broken.classification == SHADOW_ERROR
    assert broken.error
    assert broken.eligible_for_future_serving is False


def test_linear_classification_exists_for_genuine_drift():
    # A drifted pair exercises the linear path; whatever the verifier decides,
    # the classification must be an honest member of the matrix.
    cues = [(1000 + i * 3000, 3000 + i * 3000, f"drift line {i}") for i in range(250)]
    from app.services.sync.ninja_points import render_srt

    target = render_srt(cues)
    reference = render_srt([(1.02 * s + 500.0, 1.02 * e + 500.0, t) for s, e, t in cues])
    result = run_shadow_evaluation(
        target, reference, alass_state_value="unverified", alass_verified=False
    )
    assert result.classification in (
        NINJA_POINTS_LINEAR_VERIFIED,
        NINJA_POINTS_REJECTED,
        REFERENCE_REJECTED,
        INSUFFICIENT_EVIDENCE,
    )
    if result.classification == NINJA_POINTS_LINEAR_VERIFIED:
        assert result.slope is not None and 0.95 <= result.slope <= 1.05


def test_determinism_across_runs():
    target = _srt(30, "alpha")
    reference = _srt(30, "alpha", offset_ms=2000)
    first = run_shadow_evaluation(
        target, reference, alass_state_value="unverified", alass_verified=False
    )
    second = run_shadow_evaluation(
        target, reference, alass_state_value="unverified", alass_verified=False
    )
    assert first.classification == second.classification
    assert first.anchor_count == second.anchor_count
    assert first.offset_ms == second.offset_ms


# ===========================================================================
# Telemetry is bounded and secret-free
# ===========================================================================


def test_counters_track_classifications_only(caplog):
    run_shadow_evaluation(
        _srt(30, "alpha"), _srt(30, "alpha", offset_ms=2000),
        alass_state_value="unverified", alass_verified=False,
    )
    snap = counters_snapshot()
    assert snap["counts"][NINJA_POINTS_CONSTANT_VERIFIED] == 1
    assert set(snap["counts"]) == {
        ALASS_WINS, NINJA_POINTS_CONSTANT_VERIFIED, NINJA_POINTS_LINEAR_VERIFIED,
        NINJA_POINTS_REJECTED, REFERENCE_REJECTED, INSUFFICIENT_EVIDENCE,
        UNSUPPORTED_STRUCTURE, SHADOW_ERROR,
    }
    assert isinstance(snap["total_ms"], float)


def test_shadow_logs_carry_no_subtitle_text_or_secrets(caplog):
    target = _srt(10, "supersecret dialogue line")
    reference = _srt(10, "supersecret dialogue line", offset_ms=2000)
    with caplog.at_level(logging.INFO, logger="app.services.sync.ninja_shadow"):
        run_shadow_evaluation(
            target, reference, alass_state_value="unverified", alass_verified=False
        )
    blob = "\n".join(r.message for r in caplog.records)
    assert "supersecret" not in blob
    assert "api_key" not in blob.lower()
    assert "bearer" not in blob.lower()
    assert "token" not in blob.lower()


def test_shadow_module_touches_no_ranking_or_hash():
    import app.services.sync.ninja_shadow as module

    source = open(module.__file__, encoding="utf-8").read()
    for forbidden in ("order_candidates", "MatchTier", "moviehash", "video_hash", "set_verdict"):
        assert forbidden not in source


# ===========================================================================
# Integration: shadow changes nothing observable
# ===========================================================================


def _make_harness():
    from app.services.sync import orchestrator as om
    from app.services.sync.orchestrator import SyncOrchestrator
    from app.services.sync.query import ResolvedReference

    class _Cache:
        is_ephemeral = False

        def __init__(self):
            self.writes: list[str] = []

        async def get(self, k):
            return None

        async def get_meta(self, k):
            return None

        async def is_failed(self, k):
            return False

        async def mark_failed(self, k, ttl=None):
            self.writes.append(("mark_failed", k))

        async def clear_failed(self, k):
            pass

        async def get_verdict(self, k):
            return None

        async def set_verdict(self, k, v, **kw):
            self.writes.append(("set_verdict", k))

        async def set(self, k, d):
            self.writes.append(("set", k))

        async def set_meta(self, k, m):
            self.writes.append(("set_meta", k))

    class _Svc:
        async def sync_async(self, *a, **kw):
            ref = a[1] if len(a) > 1 else ""
            _ = ref
            out = []
            for i in range(600):
                s = 1000 + i * 3000 + (4000 if i % 2 == 0 else -4000)
                out.append(f"{i + 1}\n{_ts(s)} --> {_ts(s + 900)}\nSCRAMBLED{i}\n")
            return "\n\n".join(out) + "\n"

    class _Multi:
        validates_target = True
        cache = None

        def __init__(self):
            self.calls = 0

        async def resolve_all_with_provenance(self, query, *, update_validator=None):
            self.calls += 1
            refs = [
                _srt(600, f"C{i}", offset_ms=3000 * (i + 1)) for i in range(2)
            ]
            out = []
            for i, ref in enumerate(refs):
                if update_validator is not None and not update_validator(ref):
                    continue
                out.append(ResolvedReference(ref, kind="edition", candidate=f"CAND{i}"))
            return out

    def meta():
        return {
            "media_type": "movie",
            "target_filename": "Whiplash.2014.2160p.UHD.BluRay.x264-SURCODE.mkv",
            "video_hash": "",
            "video_size": "19571049411",
            "lang": "ara",
            "imdb_id": "tt2582802",
        }

    return om, SyncOrchestrator, _Cache, _Svc, _Multi, meta


@pytest.mark.asyncio
async def test_shadow_does_not_alter_serving(monkeypatch, caplog):
    om, SyncOrchestrator, _Cache, _Svc, _Multi, meta = _make_harness()
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True, raising=False)
    sub = _srt(600, "T").encode()

    monkeypatch.setattr(om.settings, "NINJA_SHADOW_ENABLED", False, raising=False)
    o_off = SyncOrchestrator(sync_cache=_Cache(), external_strategy=_Multi(), sync_service=_Svc())
    out_off = await o_off.evaluate_and_sync(bytes(sub), meta(), "tt2582802", auto_sync=True)

    monkeypatch.setattr(om.settings, "NINJA_SHADOW_ENABLED", True, raising=False)
    cache_on = _Cache()
    o_on = SyncOrchestrator(sync_cache=cache_on, external_strategy=_Multi(), sync_service=_Svc())
    with caplog.at_level(logging.INFO, logger="app.services.sync.ninja_shadow"):
        out_on = await o_on.evaluate_and_sync(bytes(sub), meta(), "tt2582802", auto_sync=True)

    assert out_on == out_off == bytes(sub)
    assert any("ninja-shadow" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_shadow_writes_no_production_cache_entries(monkeypatch):
    om, SyncOrchestrator, _Cache, _Svc, _Multi, meta = _make_harness()
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True, raising=False)
    monkeypatch.setattr(om.settings, "NINJA_SHADOW_ENABLED", True, raising=False)
    sub = _srt(600, "T").encode()

    cache_off = _Cache()
    o_off = SyncOrchestrator(sync_cache=cache_off, external_strategy=_Multi(), sync_service=_Svc())
    monkeypatch.setattr(om.settings, "NINJA_SHADOW_ENABLED", False, raising=False)
    await o_off.evaluate_and_sync(bytes(sub), meta(), "tt2582802", auto_sync=True)

    monkeypatch.setattr(om.settings, "NINJA_SHADOW_ENABLED", True, raising=False)
    cache_on = _Cache()
    o_on = SyncOrchestrator(sync_cache=cache_on, external_strategy=_Multi(), sync_service=_Svc())
    await o_on.evaluate_and_sync(bytes(sub), meta(), "tt2582802", auto_sync=True)

    assert sorted(map(str, cache_on.writes)) == sorted(map(str, cache_off.writes))


@pytest.mark.asyncio
async def test_shadow_verified_candidate_is_recorded_not_served(monkeypatch):
    """Even when NinjaSync would verify, the served bytes are the original."""
    om, SyncOrchestrator, _Cache, _Svc, _Multi, meta = _make_harness()
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True, raising=False)
    monkeypatch.setattr(om.settings, "NINJA_SHADOW_ENABLED", True, raising=False)

    class _GoodMulti(_Multi):
        async def resolve_all_with_provenance(self, query, *, update_validator=None):
            ref = _srt(600, "G", offset_ms=3000)
            if update_validator is not None and not update_validator(ref):
                return []
            from app.services.sync.query import ResolvedReference

            return [ResolvedReference(ref, kind="edition", candidate="GOOD")]

    sub = _srt(600, "T").encode()
    o = SyncOrchestrator(sync_cache=_Cache(), external_strategy=_GoodMulti(), sync_service=_Svc())
    out = await o.evaluate_and_sync(bytes(sub), meta(), "tt2582802", auto_sync=True)
    # The sync service returns scrambled output, so production rejects and the
    # original is served -- regardless of what shadow measured.
    assert out == bytes(sub)


# ===========================================================================
# Shadow matrix M-W: translator metadata cases (real credit text)
# ===========================================================================


def _credit_srt(entries) -> str:
    entries = list(entries)
    out = []
    for i, (s, text) in enumerate(entries):
        out.append(f"{i + 1}\n{_ts(s)} --> {_ts(s + 2000)}\n{text}")
    return "\n\n".join(out) + "\n"


def _dialogue_entries(n, start_ms=60000, step_ms=4000, offset_ms=0):
    return [
        (start_ms + i * step_ms + offset_ms, f"dialogue line number {i} here")
        for i in range(n)
    ]


def test_shadow_arabic_translator_credit():
    """Matrix M: credit filtered, dialogue verifies, telemetry counts it."""
    target = _credit_srt([(1000, "\u062a\u0631\u062c\u0645\u0629 \u0641\u0644\u0627\u0646")] + _dialogue_entries(14))
    reference = _credit_srt([(1000, "\u062a\u0631\u062c\u0645\u0629 \u0641\u0644\u0627\u0646")] + _dialogue_entries(14, offset_ms=2000))
    result = run_shadow_evaluation(
        target, reference, alass_state_value="unverified", alass_verified=False
    )
    assert result.classification == NINJA_POINTS_CONSTANT_VERIFIED
    assert result.metadata_skipped >= 1
    assert result.eligible_for_future_serving is True


def test_shadow_english_and_timing_credits():
    """Matrix N/O: English and timing attribution filtered the same way."""
    target = _credit_srt(
        [(2000, "Translated by XYZ"), (5000, "\u062a\u0639\u062f\u064a\u0644 \u0627\u0644\u062a\u0648\u0642\u064a\u062a: XYZ")]
        + _dialogue_entries(14)
    )
    reference = _credit_srt(
        [(2000, "Translated by XYZ"), (5000, "\u062a\u0639\u062f\u064a\u0644 \u0627\u0644\u062a\u0648\u0642\u064a\u062a: XYZ")]
        + _dialogue_entries(14, offset_ms=2000)
    )
    result = run_shadow_evaluation(
        target, reference, alass_state_value="unverified", alass_verified=False
    )
    assert result.classification == NINJA_POINTS_CONSTANT_VERIFIED
    assert result.metadata_skipped >= 2


def test_shadow_url_and_social_handles():
    """Matrix P/Q: website and social metadata never anchor."""
    target = _credit_srt(
        [(1500, "Translated by XYZ www.example.com"), (4000, "Telegram: @example")]
        + _dialogue_entries(14)
    )
    reference = _credit_srt(
        [(1500, "Translated by XYZ www.example.com"), (4000, "Telegram: @example")]
        + _dialogue_entries(14, offset_ms=2000)
    )
    result = run_shadow_evaluation(
        target, reference, alass_state_value="unverified", alass_verified=False
    )
    assert result.classification == NINJA_POINTS_CONSTANT_VERIFIED
    assert result.metadata_skipped >= 2


def test_shadow_release_group_credit():
    """Matrix R: release-group credit lines are metadata, not dialogue."""
    target = _credit_srt([(1000, "Synced by XYZ")] + _dialogue_entries(14))
    reference = _credit_srt([(1000, "Synced by XYZ")] + _dialogue_entries(14, offset_ms=2000))
    result = run_shadow_evaluation(
        target, reference, alass_state_value="unverified", alass_verified=False
    )
    assert result.classification == NINJA_POINTS_CONSTANT_VERIFIED
    assert result.metadata_skipped >= 1


def test_shadow_misleading_credit_timestamp():
    """Matrix S: the strongest false-anchor scenario. Credit at +35s, dialogue
    at +2s. The shadow result must follow the dialogue, never the credit."""
    target = _credit_srt([(5000, "\u062a\u0631\u062c\u0645\u0629 \u0641\u0644\u0627\u0646")] + _dialogue_entries(14))
    reference = _credit_srt([(40000, "\u062a\u0631\u062c\u0645\u0629 \u0641\u0644\u0627\u0646")] + _dialogue_entries(14, offset_ms=2000))
    result = run_shadow_evaluation(
        target, reference, alass_state_value="unverified", alass_verified=False
    )
    assert result.classification == NINJA_POINTS_CONSTANT_VERIFIED
    assert result.offset_ms == pytest.approx(2000.0, abs=250.0)
    assert abs((result.offset_ms or 0) - 35000.0) > 20000.0
    assert result.metadata_skipped >= 1


def test_shadow_legitimate_first_dialogue():
    """Matrix T: a real first cue (Run!) stays eligible as an anchor."""
    target = _credit_srt([(2000, "Run!"), (8000, "Stop right there now")])
    reference = _credit_srt([(4000, "Run!"), (10000, "Stop right there now")])
    result = run_shadow_evaluation(
        target, reference, alass_state_value="unverified", alass_verified=False
    )
    # Two cues cannot verify through the analyzer, but the run must attempt
    # (or reject on verification) -- never report the cues as metadata.
    assert result.metadata_skipped == 0
    assert result.classification in (NINJA_POINTS_REJECTED, INSUFFICIENT_EVIDENCE)


def test_shadow_mixed_metadata_and_dialogue():
    """Matrix U: dialogue carries the fit while credits sit alongside."""
    target = _credit_srt(
        [(1000, "\u062a\u0631\u062c\u0645\u0629 \u0648\u062a\u0639\u062f\u064a\u0644: \u0641\u0644\u0627\u0646"), (3000, "Synced by XYZ")]
        + _dialogue_entries(14)
    )
    reference = _credit_srt(
        [(1000, "\u062a\u0631\u062c\u0645\u0629 \u0648\u062a\u0639\u062f\u064a\u0644: \u0641\u0644\u0627\u0646"), (3000, "Synced by XYZ")]
        + _dialogue_entries(14, offset_ms=2000)
    )
    result = run_shadow_evaluation(
        target, reference, alass_state_value="unverified", alass_verified=False
    )
    assert result.classification == NINJA_POINTS_CONSTANT_VERIFIED
    assert result.metadata_skipped >= 2


def test_shadow_metadata_both_sides_different_timing():
    """Matrix V: shared credit, divergent timing, too little dialogue to
    verify. Must fail safe and stay ineligible."""
    target = _credit_srt([(5000, "\u062a\u0631\u062c\u0645\u0629 \u0641\u0644\u0627\u0646"), (60000, "only dialogue here")])
    reference = _credit_srt([(40000, "\u062a\u0631\u062c\u0645\u0629 \u0641\u0644\u0627\u0646"), (62000, "only dialogue here")])
    result = run_shadow_evaluation(
        target, reference, alass_state_value="unverified", alass_verified=False
    )
    assert result.eligible_for_future_serving is False
    assert result.metadata_skipped >= 1


def test_shadow_metadata_one_side_only():
    """Matrix W: an unmatched credit cannot anchor by construction."""
    target = _credit_srt([(1000, "\u062a\u0631\u062c\u0645\u0629 \u0641\u0644\u0627\u0646"), (60000, "dialogue line one here")])
    reference = _credit_srt([(62000, "dialogue line one here")])
    result = run_shadow_evaluation(
        target, reference, alass_state_value="unverified", alass_verified=False
    )
    assert result.eligible_for_future_serving is False
    assert result.metadata_skipped >= 1


def test_shadow_metadata_count_in_telemetry_log(caplog):
    """The bounded log line reports the refusal count, never cue text."""
    import logging

    target = _credit_srt([(1000, "\u062a\u0631\u062c\u0645\u0629 \u0641\u0644\u0627\u0646")] + _dialogue_entries(6))
    reference = _credit_srt([(1000, "\u062a\u0631\u062c\u0645\u0629 \u0641\u0644\u0627\u0646")] + _dialogue_entries(6, offset_ms=2000))
    with caplog.at_level(logging.INFO, logger="app.services.sync.ninja_shadow"):
        run_shadow_evaluation(
            target, reference, alass_state_value="unverified", alass_verified=False
        )
    import re

    shadow_lines = [r.message for r in caplog.records if r.message.startswith("[ninja-shadow]")]
    assert shadow_lines, "shadow evaluation must log exactly one structured line"
    counts = [
        int(m.group(1))
        for line in shadow_lines
        for m in [re.search(r"metadata_skipped=(\d+)", line)]
        if m
    ]
    assert counts and all(c >= 1 for c in counts)
    assert not any("ترجمة" in line for line in shadow_lines)
