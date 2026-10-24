"""Evidence-driven sync gate: fast delivery requires positive release evidence.

The gate decides only whether to ENTER the expensive pipeline (reference
discovery, verification, ALASS). Tests assert actual execution behavior
with spies, never just the returned enum. Timing-v3 authority, pristine
reconstruction, caches, and delivery integrity are unchanged.
"""

from __future__ import annotations

import hashlib
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.sync import hybrid_router as router
from app.services.sync import orchestrator as om
from app.services.sync.metadata_gate import (
    MetadataGateAction,
    decide_metadata_gate,
)
from app.services.sync.query import ResolvedReference
from app.services.sync_cache import SyncCache

FAST = MetadataGateAction.FAST_ORIGINAL
INVESTIGATE = MetadataGateAction.INVESTIGATE
IDENTITY_CONFLICT = MetadataGateAction.IDENTITY_CONFLICT

VIDEO_BLURAY_FM = "Movie.2024.1080p.BluRay.REMUX-FraMeSToR.mkv"


def decide(video, subtitle, **kwargs):
    return decide_metadata_gate(video, subtitle, **kwargs)


# --------------------------------------------------------------------------- #
# Decision matrix (mission §9)
# --------------------------------------------------------------------------- #

def test_same_release_fast_original():
    decision = decide(VIDEO_BLURAY_FM, "Movie.2024.1080p.BluRay.REMUX-FraMeSToR.srt")
    assert decision.action is FAST
    assert decision.reasons == ("same_release_lineage",)


@pytest.mark.parametrize("video_group,subtitle_group", [("HDC", "CHD"), ("CHD", "HDC")])
def test_into_the_wild_cross_group_requires_investigation(video_group, subtitle_group):
    decision = decide(
        f"Into.the.Wild.2007.1080p.BluRay.x264-{video_group}.mkv",
        f"Into.the.Wild.2007.1080p.BluRay.x264-{subtitle_group}.srt",
    )
    assert decision.action is INVESTIGATE
    assert decision.reasons == ("release_compatibility_unproven",)


@pytest.mark.parametrize("video,subtitle", [
    ("Dexter.s8e02.Every.silver.lining1080p.BluRay.10Bit.Dts-HDMa5.1.HEVC-d3g.mkv",
     "Dexter.s8e02.Every.silver.lining.BluRay.10Bit.1080p.DD5.1.H265-d3g.srt"),
    ("Into.the.Wild.2007.1080p.BluRay.x264-HDC.mkv",
     "Into.the.Wild.2007.720p.BluRay.x264-HDC.srt"),
])
def test_confirmed_release_lineage_keeps_fast_original(video, subtitle):
    decision = decide(video, subtitle)
    assert decision.action is FAST
    assert decision.reasons == ("same_release_lineage",)


def test_compatible_family_without_group_investigates():
    assert decide(VIDEO_BLURAY_FM, "Movie.2024.1080p.BluRay.srt").action is INVESTIGATE


def test_group_difference_is_uncertain_not_a_source_conflict():
    decision = decide(VIDEO_BLURAY_FM, "Movie.2024.1080p.BluRay.HDA.srt")
    assert decision.action is INVESTIGATE
    assert "source_family_conflict" not in decision.reasons


def test_source_family_conflict_investigates_both_directions():
    assert decide(VIDEO_BLURAY_FM, "Movie.2024.WEB-DL.AMZN.srt").action is INVESTIGATE
    assert decide("Movie.2024.WEB-DL.AMZN.mkv",
                  "Movie.2024.1080p.BluRay.FraMeSToR.srt").action is INVESTIGATE
    assert decide(VIDEO_BLURAY_FM,
                  "Movie.2024.WEB-DL.AMZN.srt").reasons == ("source_family_conflict",)


def test_matching_parsed_web_lineage_stays_fast():
    assert decide("Movie.2024.WEB-DL.AMZN.mkv", "Movie.2024.WEB-DL.AMZN.srt").action is FAST


def test_unknown_service_without_lineage_investigates():
    assert decide("Movie.2024.WEB-DL.mkv", "Movie.2024.WEB-DL.srt").action is INVESTIGATE


def test_bare_filename_invents_no_mismatch():
    decision = decide("Movie.2024.1080p.BluRay.x264-GROUP.mkv", "Arabic.srt")
    assert decision.action is INVESTIGATE
    assert decision.reasons == ("metadata_insufficient",)


def test_confirmed_incompatible_cut_investigates():
    decision = decide("Movie.2024.1080p.BluRay.srt",
                      "Movie.2024.1080p.BluRay.Extended.srt")
    assert decision.action is INVESTIGATE
    assert decision.reasons == ("edition_or_cut_conflict",)


def test_missing_lineage_is_not_an_edition_conflict():
    decision = decide("Movie.2024.1080p.BluRay.srt", "Movie.2024.1080p.BluRay.srt")
    assert decision.action is INVESTIGATE
    assert decision.reasons == ("release_compatibility_unproven",)


def test_resolution_codec_remux_alone_never_investigate():
    assert decide("Movie.2024.2160p.BluRay.REMUX-FraMeSToR.mkv",
                  "Movie.2024.1080p.BluRay.x264-FraMeSToR.srt").action is FAST
    # ... while a genuine source difference on a clean name still triggers.
    decision = decide(VIDEO_BLURAY_FM, "Movie.2024.HDTV.x264-GRP.srt")
    assert decision.action is INVESTIGATE
    assert decision.reasons == ("source_family_conflict",)


def test_wrong_episode_preserves_identity_protection():
    decision = decide("Show.S01E01.1080p.BluRay-GRP.mkv", "Show.S01E02.1080p.BluRay-GRP.srt")
    assert decision.action is IDENTITY_CONFLICT
    assert decision.reasons == ("episode_identity_conflict",)


def test_wrong_title_preserves_identity_protection():
    decision = decide("Movie.2024.1080p.BluRay-GRP.mkv", "Other.2024.1080p.BluRay-GRP.srt")
    assert decision.action is IDENTITY_CONFLICT


def test_remake_year_conflict_is_identity_not_timing():
    decision = decide("Movie.2024.1080p.BluRay-GRP.mkv", "Movie.2001.1080p.BluRay-GRP.srt")
    assert decision.action is IDENTITY_CONFLICT
    assert decision.reasons == ("year_identity_conflict",)


def test_missing_names_abstain_to_legacy_investigation():
    assert decide(None, "Movie.2024.1080p.BluRay.srt").reasons == (
        "insufficient_metadata_for_gate",)
    assert decide(VIDEO_BLURAY_FM, None).action is INVESTIGATE
    assert decide("", "").action is INVESTIGATE


def test_source_aliases_uhd_bluray_compatible():
    assert decide("Movie.2024.2160p.UHD.BluRay-FraMeSToR.mkv",
                  "Movie.2024.1080p.BluRay-FraMeSToR.srt").action is FAST


def test_pal_frame_rate_conflict_investigates():
    decision = decide("Show.S01E01.1080p.BluRay.25fps-GRP.mkv",
                      "Show.S01E01.1080p.BluRay.23.976fps-GRP.srt")
    assert decision.action is INVESTIGATE
    assert decision.reasons == ("frame_rate_conflict",)


def test_explicit_request_bypasses_compatible_metadata():
    decision = decide(VIDEO_BLURAY_FM, "Movie.2024.1080p.BluRay.REMUX-FraMeSToR.srt",
                      explicit_sync_request=True)
    assert decision.action is INVESTIGATE
    assert decision.reasons == ("explicit_sync_request",)


# --------------------------------------------------------------------------- #
# Execution harness with spies
# --------------------------------------------------------------------------- #

def _fmt(total_ms: int) -> str:
    total_ms = max(0, total_ms)
    millis = total_ms % 1000
    seconds = (total_ms // 1000) % 60
    minutes = (total_ms // 60_000) % 60
    hours = total_ms // 3_600_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _sentences(n: int, shift_ms: int = 0, paraphrase: tuple[int, ...] = (),
               start_ms: int = 60_000, step_ms: int = 60_000):
    alternates = ["Completely different sentence alpha",
                  "Completely different sentence beta"]
    target, reference = [], []
    for position in range(n):
        start = start_ms + position * step_ms
        text = f"Distinctive dialogue event number {position + 1} spoken aloud"
        target.append(f"{position + 1}\n{_fmt(start)} --> {_fmt(start + 2_000)}\n{text}")
        shown = alternates[position % 2] if (position + 1) in paraphrase else text
        moved = start + shift_ms
        reference.append(f"{position + 1}\n{_fmt(moved)} --> {_fmt(moved + 2_000)}\n{shown}")
    return "\n\n".join(target) + "\n", "\n\n".join(reference) + "\n"


def _harness(monkeypatch, resolved, cache=None):
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
    orch = om.SyncOrchestrator(sync_cache=cache or SyncCache(),
                              external_strategy=strategy, sync_service=service)
    return orch, strategy, service


def _meta(video=VIDEO_BLURAY_FM, release=None, **extra):
    meta = {"imdb_id": "tt1234567", "media_type": "movie", "lang": "ara",
            "target_filename": video, "title": "Movie", "video_size": 4200000000}
    if release is not None:
        meta["release_name"] = release
    meta.update(extra)
    return meta


@pytest.mark.parametrize("video,release", [
    (VIDEO_BLURAY_FM, "Movie.2024.1080p.BluRay.REMUX-FraMeSToR.srt"),
    ("Dexter.s8e02.Every.silver.lining1080p.BluRay.10Bit.Dts-HDMa5.1.HEVC-d3g.mkv",
     "Dexter.s8e02.Every.silver.lining.BluRay.10Bit.1080p.DD5.1.H265-d3g.srt"),
])
async def test_fast_path_avoids_discovery_download_and_alass(monkeypatch, caplog, video, release):
    target, _ = _sentences(16)
    orch, strategy, service = _harness(monkeypatch, None)
    with caplog.at_level("INFO", logger="app.services.sync.orchestrator"):
        out = await orch.evaluate_and_sync(
            target.encode(),
            _meta(video=video, release=release),
            "sub", auto_sync=True)
    assert out == target.encode()
    strategy.resolve_with_provenance.assert_not_called()
    service.sync_async.assert_not_called()
    assert any("metadata-gate decision=fast_original" in r.message for r in caplog.records)


async def test_into_the_wild_without_timing_witness_reaches_discovery(monkeypatch):
    target, _ = _sentences(16)
    orch, strategy, _service = _harness(monkeypatch, ResolvedReference(None))
    monkeypatch.setattr(orch, "_timing_witness_decision", lambda *a: om.TimingWitnessDecision(
        False, "timing_witness_unavailable"))
    out = await orch.evaluate_and_sync(
        target.encode(),
        _meta(video="Into.the.Wild.2007.1080p.BluRay.x264-HDC.mkv",
              release="Into.the.Wild.2007.1080p.BluRay.x264-CHD.srt"),
        "sub", auto_sync=True,
    )
    strategy.resolve_with_provenance.assert_awaited()
    # Investigation alone cannot authorize a correction without a reference.
    assert out == target.encode()


async def test_investigate_path_reaches_reference_discovery(monkeypatch):
    target, reference = _sentences(16, paraphrase=(4, 8, 12, 16))
    resolved = ResolvedReference(reference, kind="edition", candidate="ref.srt")
    orch, strategy, service = _harness(monkeypatch, resolved)
    out = await orch.evaluate_and_sync(
        target.encode(), _meta(release="Movie.2024.WEB-DL.AMZN.srt"),
        "sub", auto_sync=True)
    strategy.resolve_with_provenance.assert_awaited()
    # Zero shift: the pipeline runs, verifies alignment, and serves the
    # original bytes without ever invoking ALASS.
    assert out == target.encode()
    service.sync_async.assert_not_called()
    assert str(orch._last_evaluation.sync_state.value).startswith("verified")


async def test_identity_conflict_serves_original_without_investigation(monkeypatch, caplog):
    target, reference = _sentences(16, paraphrase=(4, 8, 12, 16))
    resolved = ResolvedReference(reference, kind="edition", candidate="ref.srt")
    orch, strategy, service = _harness(monkeypatch, resolved)
    with caplog.at_level("INFO", logger="app.services.sync.orchestrator"):
        out = await orch.evaluate_and_sync(
            target.encode(),
            _meta(video="Show.S01E01.1080p.BluRay-GRP.mkv",
                   release="Show.S01E02.1080p.BluRay-GRP.srt",
                   media_type="series", season=1, episode=2),
            "sub", auto_sync=True)
    assert out == target.encode()
    strategy.resolve_with_provenance.assert_not_called()
    service.sync_async.assert_not_called()
    assert any("identity conflict" in r.message for r in caplog.records)


async def test_explicit_request_bypasses_fast_delivery(monkeypatch):
    target, reference = _sentences(16, paraphrase=(4, 8, 12, 16))
    resolved = ResolvedReference(reference, kind="edition", candidate="ref.srt")
    orch, strategy, service = _harness(monkeypatch, resolved)
    await orch.evaluate_and_sync(
        target.encode(),
        _meta(release="Movie.2024.1080p.BluRay.REMUX-FraMeSToR.srt",
              force_sync=True),
        "sub", auto_sync=True)
    strategy.resolve_with_provenance.assert_awaited()


async def test_fast_delivery_writes_no_verified_verdict_and_promotes_nothing(monkeypatch):
    target, _ = _sentences(16)
    cache = SyncCache()
    promoted = []
    monkeypatch.setattr(om.SyncOrchestrator, "_promote_sticky_reference",
                        lambda self, **k: promoted.append(k))
    orch, _, _ = _harness(monkeypatch, None, cache=cache)
    meta = _meta(release="Movie.2024.1080p.BluRay.REMUX-FraMeSToR.srt",
                 video_hash="0123456789abcdef")
    out = await orch.evaluate_and_sync(target.encode(), meta, "sub", auto_sync=True)
    assert out == target.encode()
    assert promoted == []
    key = om.SyncOrchestrator._verdict_key(
        meta, hashlib.sha256(target.encode()).hexdigest()[:16])
    assert key is not None
    assert await cache.get_verdict(key) is None


async def test_unknown_names_preserve_legacy_investigation(monkeypatch):
    target, reference = _sentences(16, paraphrase=(4, 8, 12, 16))
    resolved = ResolvedReference(reference, kind="edition", candidate="ref.srt")
    orch, strategy, _ = _harness(monkeypatch, resolved)
    meta = {"imdb_id": "tt1234567", "media_type": "movie", "lang": "ara",
            "title": "Movie", "video_size": 4200000000}
    await orch.evaluate_and_sync(target.encode(), meta, "sub", auto_sync=True)
    strategy.resolve_with_provenance.assert_awaited()


# --------------------------------------------------------------------------- #
# Sync scenarios on source mismatch (Stages 3-4)
# --------------------------------------------------------------------------- #

async def test_mismatch_but_aligned_serves_original_verified_without_alass(monkeypatch):
    target, reference = _sentences(16, paraphrase=(4, 8, 12, 16))
    resolved = ResolvedReference(reference, kind="edition", candidate="ref.srt")
    orch, strategy, service = _harness(monkeypatch, resolved)
    out = await orch.evaluate_and_sync(
        target.encode(), _meta(release="Movie.2024.WEB-DL.AMZN.srt"),
        "sub", auto_sync=True)
    assert out == target.encode()
    service.sync_async.assert_not_called()
    evaluation = orch._last_evaluation
    assert evaluation is not None
    assert str(evaluation.sync_state.value).startswith("verified")


async def test_mismatch_with_offset_serves_transformed_when_verified(monkeypatch):
    target, reference = _sentences(16, shift_ms=9_000, paraphrase=(4, 8, 12, 16))
    resolved = ResolvedReference(reference, kind="edition", candidate="ref.srt")
    orch, _, service = _harness(monkeypatch, resolved)
    out = await orch.evaluate_and_sync(
        target.encode(), _meta(release="Movie.2024.WEB-DL.AMZN.srt"),
        "sub", auto_sync=True)
    assert out != target.encode()
    service.sync_async.assert_not_called()


async def test_mismatch_without_evidence_falls_back_to_original(monkeypatch):
    target, _ = _sentences(10)
    other = "\n\n".join(
        f"{i + 1}\n{_fmt(60_000 + i * 60_000)} --> {_fmt(62_000 + i * 60_000)}\n"
        f"Unrelated episode line {i + 1} entirely" for i in range(10)) + "\n"
    resolved = ResolvedReference(other, kind="edition", candidate="ref.srt")
    orch, _, _ = _harness(monkeypatch, resolved)
    out = await orch.evaluate_and_sync(
        target.encode(), _meta(release="Movie.2024.WEB-DL.AMZN.srt"),
        "sub", auto_sync=True)
    assert out == target.encode()


# --------------------------------------------------------------------------- #
# Telemetry and latency evidence
# --------------------------------------------------------------------------- #

def test_gate_decision_is_submillisecond_local_parse():
    started = time.perf_counter()
    for _ in range(50):
        decide(VIDEO_BLURAY_FM, "Movie.2024.1080p.BluRay.REMUX-FraMeSToR.srt")
    elapsed_ms = (time.perf_counter() - started) * 1000 / 50
    assert elapsed_ms < 5.0


async def test_fast_path_reports_gate_telemetry_and_avoidance(monkeypatch, caplog):
    target, _ = _sentences(16)
    orch, _, _ = _harness(monkeypatch, None)
    with caplog.at_level("INFO", logger="app.services.sync.telemetry"):
        await orch.evaluate_and_sync(
            target.encode(),
            _meta(release="Movie.2024.1080p.BluRay.REMUX-FraMeSToR.srt"),
            "sub", auto_sync=True)
    summary = next(r.message for r in caplog.records if "total_ms=" in r.message)
    assert "metadata_gate_decision=fast_original" in summary
    assert "metadata_gate_reason=same_release_lineage" in summary
    assert "metadata_gate_ms=" in summary
    assert "references_downloaded=0" in summary
    assert "ALASS_processes_launched=0" in summary


async def test_investigate_path_reports_gate_telemetry(monkeypatch, caplog):
    target, reference = _sentences(16, paraphrase=(4, 8, 12, 16))
    resolved = ResolvedReference(reference, kind="edition", candidate="ref.srt")
    orch, _, _ = _harness(monkeypatch, resolved)
    with caplog.at_level("INFO", logger="app.services.sync.telemetry"):
        await orch.evaluate_and_sync(
            target.encode(), _meta(release="Movie.2024.WEB-DL.AMZN.srt"),
            "sub", auto_sync=True)
    summary = next(r.message for r in caplog.records if "total_ms=" in r.message)
    assert "metadata_gate_decision=investigate" in summary
    assert "metadata_gate_reason=source_family_conflict" in summary


def test_gate_reasons_are_closed_allowlist():
    import app.services.sync.metadata_gate as gate_module

    assert set(gate_module.REASONS) == {
        "same_release_lineage", "no_mismatch_evidence", "metadata_insufficient",
        "release_compatibility_unproven",
        "insufficient_metadata_for_gate", "source_family_conflict",
        "edition_or_cut_conflict", "frame_rate_conflict",
        "episode_identity_conflict", "title_identity_conflict",
        "year_identity_conflict", "explicit_sync_request",
        # Phase 6: the timing-witness vocabulary, reported beside the metadata
        # verdict so "metadata compatible" and "timing compatible" stay
        # distinguishable. Still a closed set; no provider-controlled string.
        "timing_witness_aligned", "timing_witness_insufficient",
        "timing_witness_unavailable", "timing_witness_conflict",
    }
