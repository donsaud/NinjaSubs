"""Behavioral isolation of the bounded shadow pool, and report honesty.

An AST check cannot see data-flow, so the isolation proof here drives the real
strategy and asserts on the value the synchronizer actually receives.
"""
import importlib.util
import json
import sys

import pytest

from app.services.sync.reference_v2 import COMPARISON_MEANINGFUL, COMPARISON_NO

TARGET = "Dexter.S08E05.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv"
CREDITS = "sync by someone"


def _ts(ms: int) -> str:
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, msec = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{msec:03d}"


def srt(positions, *, duration_ms: int = 1500, body: str = "I never thought I would find someone") -> str:
    return "\n".join(
        f"{i}\n{_ts(p)} --> {_ts(p + duration_ms)}\n{body}\n"
        for i, p in enumerate(positions, start=1)
    )


def even(count: int, step_ms: int = 2000, start_ms: int = 0) -> list[int]:
    return [start_ms + i * step_ms for i in range(count)]


def _strategy(monkeypatch, tmp_path, groups, payloads, *, fetch_limit=0):
    from app.config import settings
    from app.models import SubtitleRelease
    from app.services.sync.cache import ReferenceDiskCache
    from app.services.sync.external_strategy import ExternalExactStrategy
    from app.services.sync.query import ReferenceQuery

    monkeypatch.setattr(settings, "REFERENCE_SHADOW_POOL_FETCH_LIMIT", fetch_limit, raising=False)

    class _Provider:
        name = "subdl"

        def __init__(self):
            self.downloaded: list[str] = []

        async def search_subtitles(self, **kwargs):
            return [
                SubtitleRelease(
                    release_name=f"Dexter.S08E05.1080p.BluRay.x264-{grp}.srt",
                    download_url=f"http://{grp}",
                    provider="subdl",
                    lang="eng",
                )
                for grp in groups
            ]

        async def download_archive(self, url, api_key=None):
            self.downloaded.append(url)
            return payloads[url].encode()

    provider = _Provider()
    strategy = ExternalExactStrategy(
        subdl_provider=provider,
        subsource_provider=None,
        opensubtitles_provider=None,
        cache=ReferenceDiskCache(root=str(tmp_path), ttl=3600.0, min_bytes=100),
        min_bytes=100,
        timeout=1.0,
    )
    return strategy, provider, ReferenceQuery(
        imdb_id="tt0773262", media_type="series", season=8, episode=5, target_filename=TARGET
    )


def _validator(target_text):
    from app.services.subtitle_matcher import (
        FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
        validate_cue_sanity,
    )

    def _validate(reference_text: str) -> bool:
        return bool(
            validate_cue_sanity(
                target_text,
                reference_text,
                threshold_ms=FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
            )["ok"]
        )

    return _validate


@pytest.mark.asyncio
async def test_meaningful_disagreement_never_reaches_the_synchronizer(monkeypatch, tmp_path):
    """The core proof: a real disagreement, and the legacy text still wins.

    Candidate 1 is healthy dialogue from a different cut, so the legacy loop
    rejects it on cue-sanity and downloads candidate 2, which is credits-only
    but correctly timed. Legacy commits to candidate 2. Shadow sees both and
    prefers candidate 1. The synchronizer must still receive candidate 2.
    """
    target_text = srt(even(40))
    wrong_cut = srt(even(40, start_ms=97_000))
    credits = srt(even(40), body=CREDITS)
    groups = ["GroupA", "GroupB"]
    payloads = {"http://GroupA": wrong_cut, "http://GroupB": credits}

    strategy, provider, query = _strategy(monkeypatch, tmp_path, groups, payloads)
    resolved = await strategy.resolve_with_provenance(
        query, update_validator=_validator(target_text)
    )

    assert provider.downloaded == ["http://GroupA", "http://GroupB"]
    assert resolved.text is not None
    # The synchronizer receives the LEGACY reference...
    assert resolved.text == credits
    assert resolved.text != wrong_cut
    # ...while the shadow genuinely disagreed, over a real pool.
    assert resolved.shadow_changed is True
    assert resolved.shadow_comparison_class == COMPARISON_MEANINGFUL
    assert resolved.shadow_pool_size == 2
    assert resolved.shadow_pool_independent_groups == 2
    assert resolved.shadow_additional_fetches == 0
    assert "SHADOW_SWITCH" in resolved.shadow_switch_labels


@pytest.mark.asyncio
async def test_single_candidate_pool_is_recorded_as_no_comparison(monkeypatch, tmp_path):
    """The early-break bias is now visible instead of flattering."""
    target_text = srt(even(40))
    credits = srt(even(40), body=CREDITS)
    strategy, _provider, query = _strategy(
        monkeypatch, tmp_path, ["GroupB"], {"http://GroupB": credits}
    )

    resolved = await strategy.resolve_with_provenance(
        query, update_validator=_validator(target_text)
    )
    assert resolved.text == credits
    assert resolved.shadow_pool_size == 1
    assert resolved.shadow_comparison_class == COMPARISON_NO
    # Agreement here is an artifact of there being no alternative.
    assert resolved.shadow_changed is False
    assert resolved.shadow_pool_limited_by_payloads is True


@pytest.mark.asyncio
async def test_optional_pool_fetch_is_bounded_and_measured(monkeypatch, tmp_path):
    """Opt-in extra downloads stay bounded and are counted.

    Audit must actually be recording, otherwise the budget is refused even
    though it is configured.
    """
    from app.services.sync.audit import AUDIT_LOG

    monkeypatch.setattr(AUDIT_LOG, "enabled", True, raising=False)
    target_text = srt(even(40))
    credits = srt(even(40), body=CREDITS)
    other = srt(even(40, step_ms=2200))
    groups = ["GroupA", "GroupB", "GroupC"]
    payloads = {
        "http://GroupA": credits,
        "http://GroupB": other,
        "http://GroupC": srt(even(40, step_ms=2400)),
    }

    strategy, provider, query = _strategy(
        monkeypatch, tmp_path, groups, payloads, fetch_limit=1
    )
    resolved = await strategy.resolve_with_provenance(
        query, update_validator=_validator(target_text)
    )

    # Legacy still stops at its own first passing candidate.
    assert resolved.text == credits
    # At most the configured number of extra fetches happened.
    assert resolved.shadow_extra_fetch_attempts <= 1
    assert resolved.shadow_extra_fetch_successes <= 1
    assert len(provider.downloaded) <= 2
    assert resolved.shadow_expansion_eligible is True
    # Marginal value is judged, not assumed.
    assert resolved.shadow_expansion_outcome in (
        "ADDED_NEW_TIMING_GROUP",
        "ADDED_DUPLICATE_GROUP",
        "ENABLED_MEANINGFUL_COMPARISON",
    )
    assert resolved.shadow_independent_groups_after >= resolved.shadow_independent_groups_before
    assert resolved.shadow_meaningful_after is True


def test_raw_agreement_is_reported_separately_from_meaningful(tmp_path):
    """A report that quotes 94% agreement off trivial cases must not exist."""
    sys_path = str(tmp_path)
    log = tmp_path / "sync_audit.jsonl"
    records = []
    # 720 one-candidate requests: legacy == shadow by necessity.
    for i in range(720):
        records.append(
            {
                "phase": "serve",
                "reference_trust": "acceptable",
                "shadow_reference_id": f"id{i}",
                "shadow_reference_changed": False,
                "shadow_pool_materialized": True,
                "shadow_pool_size": 1,
                "shadow_pool_independent_groups": 1,
                "shadow_comparison_class": "NO_COMPARISON",
            }
        )
    # 225 real agreements over a real pool.
    for i in range(225):
        records.append(
            {
                "phase": "serve",
                "reference_trust": "acceptable",
                "shadow_reference_id": f"m{i}",
                "shadow_reference_changed": False,
                "shadow_pool_materialized": True,
                "shadow_pool_size": 3,
                "shadow_pool_independent_groups": 3,
                "shadow_comparison_class": "MEANINGFUL_COMPARISON",
            }
        )
    # 55 real disagreements.
    for i in range(55):
        records.append(
            {
                "phase": "serve",
                "reference_trust": "acceptable",
                "shadow_reference_id": f"d{i}",
                "shadow_reference_changed": True,
                "shadow_pool_materialized": True,
                "shadow_pool_size": 3,
                "shadow_pool_independent_groups": 3,
                "shadow_comparison_class": "MEANINGFUL_COMPARISON",
                "shadow_switch_labels": ["SHADOW_SWITCH", "same_match_tier"],
                "shadow_reasons": ["health: unusable -> healthy"],
            }
        )
    log.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    spec = importlib.util.spec_from_file_location(
        "calibrate_sync", "tools/calibrate_sync.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    analysis = mod.shadow_comparison_analysis(records)
    assert analysis["meaningful_comparisons"] == 280
    assert analysis["meaningful_changed"] == 55
    assert analysis["meaningful_agreed"] == 225
    assert analysis["meaningful_agreement_rate"] == 0.8036
    assert analysis["non_comparable"] == 720
    assert analysis["pool_coverage"]["one_candidate_only"] == 720
    assert analysis["pool_coverage"]["two_or_more_candidates"] == 280
    # The raw figure is inflated by the 720 trivial cases; it must not be the
    # headline and must not be conflated with the meaningful rate.
    assert analysis["agreement_rate"] == 0.945
    assert analysis["meaningful_agreement_rate"] != analysis["agreement_rate"]

    rendered = mod.render_shadow_report(analysis)
    assert "Shadow Coverage" in rendered
    assert "Meaningful agreement" in rendered
    assert "Raw agreement" in rendered
    assert sys_path  # keep the fixture referenced


def test_report_classifies_legacy_records_without_pool_telemetry():
    spec = importlib.util.spec_from_file_location(
        "calibrate_sync_legacy", "tools/calibrate_sync.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    # A record written before pool telemetry existed must not be silently
    # counted as a meaningful comparison.
    analysis = mod.shadow_comparison_analysis(
        [{"phase": "serve", "reference_trust": "acceptable", "shadow_reference_id": "x"}]
    )
    assert analysis["meaningful_comparisons"] == 0
    assert analysis["non_comparable"] == 1
    assert analysis["meaningful_agreement_rate"] is None


def test_shadow_replay_reports_simulation_unavailable(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "calibrate_sync_replay", "tools/calibrate_sync.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    analysis, rendered = mod.shadow_replay_analysis([], "missing-case", tmp_path / "a.jsonl")
    assert analysis["available"] is False
    assert analysis["case_found"] is False
    assert "SIMULATION_UNAVAILABLE" in analysis["reason"]
    assert "SIMULATION_UNAVAILABLE" in rendered
