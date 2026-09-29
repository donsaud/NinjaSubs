"""Behavioral proof that an expanded shadow pool stays out of production.

Drives the real strategy with the audit recording and a budget of one, in a
scenario where the extra candidate changes what the shadow would choose. The
assertion that matters is the last one: the synchronizer still receives the
legacy reference.
"""
import json
import sys

import pytest

TARGET = "Dexter.S08E05.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv"
CREDITS = "sync by someone"


def _ts(ms: int) -> str:
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, msec = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{msec:03d}"


def srt(positions, *, duration_ms: int = 1500, body: str = "a line of dialogue here") -> str:
    return "\n".join(
        f"{i}\n{_ts(p)} --> {_ts(p + duration_ms)}\n{body}\n"
        for i, p in enumerate(positions, 1)
    )


def even(count: int, step_ms: int = 2000, start_ms: int = 0) -> list[int]:
    return [start_ms + i * step_ms for i in range(count)]


def _validator(target_text):
    from app.services.subtitle_matcher import (
        FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
        validate_cue_sanity,
    )

    def _validate(reference_text: str) -> bool:
        return bool(
            validate_cue_sanity(
                target_text, reference_text, threshold_ms=FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS
            )["ok"]
        )

    return _validate


def _build(monkeypatch, tmp_path, groups, payloads, *, fetch_limit, audit_enabled):
    from app.config import settings
    from app.models import SubtitleRelease
    from app.services.sync.audit import AUDIT_LOG
    from app.services.sync.cache import ReferenceDiskCache
    from app.services.sync.external_strategy import ExternalExactStrategy
    from app.services.sync.query import ReferenceQuery

    monkeypatch.setattr(settings, "REFERENCE_SHADOW_POOL_FETCH_LIMIT", fetch_limit, raising=False)
    monkeypatch.setattr(AUDIT_LOG, "enabled", audit_enabled, raising=False)

    class _Provider:
        name = "subdl"

        def __init__(self):
            self.downloaded: list[str] = []

        async def search_subtitles(self, **kwargs):
            return [
                SubtitleRelease(
                    release_name=f"Dexter.S08E05.1080p.BluRay.x264-{g}.srt",
                    download_url=f"http://{g}",
                    provider="subdl",
                    lang="eng",
                )
                for g in groups
            ]

        async def download_archive(self, url, api_key=None):
            self.downloaded.append(url)
            return payloads[url]

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


@pytest.mark.asyncio
async def test_a_audit_disabled_makes_no_extra_request(monkeypatch, tmp_path):
    """Configured limit, audit off: the budget must be ignored entirely."""
    target_text = srt(even(40))
    groups = ["GroupA", "GroupB"]
    payloads = {
        "http://GroupA": srt(even(40), body=CREDITS).encode(),
        "http://GroupB": srt(even(40, step_ms=2200)).encode(),
    }
    strategy, provider, query = _build(
        monkeypatch, tmp_path, groups, payloads, fetch_limit=1, audit_enabled=False
    )
    resolved = await strategy.resolve_with_provenance(
        query, update_validator=_validator(target_text)
    )
    # Legacy downloaded one candidate and stopped.
    assert len(provider.downloaded) == 1
    assert resolved.shadow_extra_fetch_attempts == 0
    assert resolved.shadow_expansion_eligible is False
    assert resolved.shadow_expansion_refusal == "audit_disabled"


@pytest.mark.asyncio
async def test_b_zero_limit_makes_no_extra_request(monkeypatch, tmp_path):
    target_text = srt(even(40))
    groups = ["GroupA", "GroupB"]
    payloads = {
        "http://GroupA": srt(even(40), body=CREDITS).encode(),
        "http://GroupB": srt(even(40, step_ms=2200)).encode(),
    }
    strategy, provider, query = _build(
        monkeypatch, tmp_path, groups, payloads, fetch_limit=0, audit_enabled=True
    )
    resolved = await strategy.resolve_with_provenance(
        query, update_validator=_validator(target_text)
    )
    assert len(provider.downloaded) == 1
    assert resolved.shadow_extra_fetch_attempts == 0
    assert resolved.shadow_expansion_refusal == "fetch_limit_zero"


@pytest.mark.asyncio
async def test_g_missing_fingerprint_makes_no_extra_request(monkeypatch, tmp_path):
    from app.config import settings
    from app.models import SubtitleRelease
    from app.services.sync.audit import AUDIT_LOG
    from app.services.sync.cache import ReferenceDiskCache
    from app.services.sync.external_strategy import ExternalExactStrategy
    from app.services.sync.query import ReferenceQuery

    monkeypatch.setattr(settings, "REFERENCE_SHADOW_POOL_FETCH_LIMIT", 1, raising=False)
    monkeypatch.setattr(AUDIT_LOG, "enabled", True, raising=False)

    class _Provider:
        name = "subdl"

        def __init__(self):
            self.downloaded: list[str] = []

        async def search_subtitles(self, **kwargs):
            return [
                SubtitleRelease(
                    release_name="Some.Movie.2020.1080p.WEB-DL.x264-GroupA.mkv",
                    download_url="http://GroupA",
                    provider="subdl",
                    lang="eng",
                ),
                SubtitleRelease(
                    release_name="Some.Movie.2020.720p.WEB-DL.x264-GroupB.mkv",
                    download_url="http://GroupB",
                    provider="subdl",
                    lang="eng",
                ),
            ]

        async def download_archive(self, url, api_key=None):
            self.downloaded.append(url)
            return srt(even(40, step_ms=2200)).encode()

    provider = _Provider()
    strategy = ExternalExactStrategy(
        subdl_provider=provider,
        subsource_provider=None,
        opensubtitles_provider=None,
        cache=ReferenceDiskCache(root=str(tmp_path), ttl=3600.0, min_bytes=100),
        min_bytes=100,
        timeout=1.0,
    )
    query = ReferenceQuery(
        imdb_id="tt1234567",
        media_type="movie",
        target_filename=None,
    )
    resolved = await strategy.resolve_with_provenance(query)
    assert resolved.shadow_extra_fetch_attempts == 0
    assert resolved.shadow_expansion_refusal in (
        "no_target_fingerprint",
        "pool_already_comparable",
    )


@pytest.mark.asyncio
async def test_h_expanded_pool_never_reaches_the_synchronizer(monkeypatch, tmp_path):
    """One extra candidate changes the shadow, not the production reference.

    The legacy pick is credits-only but correctly timed, so it passes
    cue-sanity and wins. The pool starts with a single reference, which cannot
    support a comparison, so exactly one extra candidate is fetched. That
    candidate is healthy dialogue from a different cut. The shadow can now
    prefer it - and the synchronizer must still receive the legacy text.
    """
    target_text = srt(even(40))
    legacy_text = srt(even(40), body=CREDITS)
    extra_text = srt(even(40, start_ms=91_000))
    groups = ["GroupA", "GroupB"]
    payloads = {
        "http://GroupA": legacy_text.encode(),
        "http://GroupB": extra_text.encode(),
    }
    strategy, provider, query = _build(
        monkeypatch, tmp_path, groups, payloads, fetch_limit=1, audit_enabled=True
    )
    resolved = await strategy.resolve_with_provenance(
        query, update_validator=_validator(target_text)
    )

    # Exactly one extra request, and no more.
    assert resolved.shadow_expansion_eligible is True
    assert resolved.shadow_extra_fetch_attempts == 1
    assert resolved.shadow_extra_fetch_network_fetches == 1
    assert len(provider.downloaded) == 2

    # The extra candidate really did widen the evidence.
    assert resolved.shadow_independent_groups_before == 1
    assert resolved.shadow_independent_groups_after >= 1
    assert resolved.shadow_extra_fetch_bytes and resolved.shadow_extra_fetch_bytes > 0
    assert resolved.shadow_extra_fetch_latency_ms is not None
    assert resolved.shadow_expansion_outcome in (
        "ADDED_NEW_TIMING_GROUP",
        "ADDED_DUPLICATE_GROUP",
        "ENABLED_MEANINGFUL_COMPARISON",
    )
    # Candidate metadata is recorded as safe facts, never text or a URL.
    assert resolved.shadow_extra_candidate_provider in (None, "subdl")
    assert resolved.shadow_extra_candidate_match_tier is None or isinstance(
        resolved.shadow_extra_candidate_match_tier, str
    )

    # The production reference is untouched: legacy won, and legacy is served.
    assert resolved.text == legacy_text
    assert resolved.text != extra_text


@pytest.mark.asyncio
async def test_i_network_failure_is_recorded_and_synchronization_unaffected(
    monkeypatch, tmp_path
):
    """A failing extra fetch must not damage the request."""
    target_text = srt(even(40))
    legacy_text = srt(even(40), body=CREDITS)
    groups = ["GroupA", "GroupB"]
    payloads = {
        "http://GroupA": legacy_text.encode(),
        # GroupB fails to decode into a usable reference.
        "http://GroupB": b"",
    }
    strategy, provider, query = _build(
        monkeypatch, tmp_path, groups, payloads, fetch_limit=1, audit_enabled=True
    )
    resolved = await strategy.resolve_with_provenance(
        query, update_validator=_validator(target_text)
    )
    assert resolved.text == legacy_text
    assert resolved.shadow_extra_fetch_attempts == 1
    assert resolved.shadow_extra_fetch_successes == 0
    assert resolved.shadow_expansion_outcome in ("CAUSED_FETCH_FAILURE", "NOT_ATTEMPTED")


@pytest.mark.asyncio
async def test_c_comparable_pool_spends_no_budget(monkeypatch, tmp_path):
    """When two references are already available, nothing is fetched."""
    target_text = srt(even(40))
    # Legacy rejects the first on cue-sanity, so two are already materialized.
    groups = ["GroupA", "GroupB", "GroupC"]
    payloads = {
        "http://GroupA": srt(even(40, start_ms=97_000)).encode(),
        "http://GroupB": srt(even(40)).encode(),
        "http://GroupC": srt(even(40, step_ms=2400)).encode(),
    }
    strategy, provider, query = _build(
        monkeypatch, tmp_path, groups, payloads, fetch_limit=1, audit_enabled=True
    )
    resolved = await strategy.resolve_with_provenance(
        query, update_validator=_validator(target_text)
    )
    assert len(provider.downloaded) == 2
    assert resolved.shadow_extra_fetch_attempts == 0
    assert resolved.shadow_expansion_refusal == "pool_already_comparable"
    assert resolved.shadow_comparison_class == "MEANINGFUL_COMPARISON"


def test_report_expansion_section_states_facts_without_recommending():
    import importlib.util

    spec = importlib.util.spec_from_file_location("cal_exp", "tools/calibrate_sync.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    records = [
        {
            "phase": "serve",
            "reference_trust": "acceptable",
            "shadow_reference_id": f"x{i}",
            "shadow_reference_changed": False,
            "shadow_pool_materialized": True,
            "shadow_pool_size": 1,
            "shadow_pool_independent_groups": 1,
            "shadow_comparison_class": "NO_COMPARISON",
            "shadow_expansion_eligible": True,
            "shadow_expansion_outcome": "ENABLED_MEANINGFUL_COMPARISON",
            "shadow_expansion_refusal": None,
            "shadow_extra_fetch_attempts": 1,
            "shadow_extra_fetch_successes": 1,
            "shadow_extra_fetch_network_fetches": 1,
            "shadow_extra_fetch_bytes": 4096,
            "shadow_extra_fetch_latency_ms": 15.0,
            "shadow_pool_size_before": 1,
            "shadow_independent_groups_before": 1,
            "shadow_pool_size_after": 2,
            "shadow_independent_groups_after": 2,
            "shadow_meaningful_before": False,
            "shadow_meaningful_after": True,
            "shadow_extra_duplicate_timing_grid": False,
            "shadow_extra_candidate_provider": "subdl",
            "shadow_extra_candidate_match_tier": "EXACT",
        }
        for i in range(40)
    ]
    analysis = mod.shadow_comparison_analysis(records)
    exp = analysis["expansion"]
    assert exp["eligible_requests"] == 40
    assert exp["network_fetches"] == 40
    assert exp["outcomes"]["ENABLED_MEANINGFUL_COMPARISON"] == 40
    assert exp["independent_groups_before"] == 1.0
    assert exp["independent_groups_after"] == 2.0
    assert exp["meanful_after_count"] == 40
    assert exp["marginal_meaningful_rate"] == 1.0

    rendered = mod.render_shadow_report(analysis)
    assert "Shadow Pool Expansion" in rendered
    assert "Independent groups before" in rendered
    assert "These are observations" in rendered


def test_report_refuses_a_rate_on_a_thin_sample():
    import importlib.util

    spec = importlib.util.spec_from_file_location("cal_thin", "tools/calibrate_sync.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    records = [
        {
            "phase": "serve",
            "reference_trust": "acceptable",
            "shadow_reference_id": f"x{i}",
            "shadow_reference_changed": False,
            "shadow_pool_materialized": True,
            "shadow_pool_size": 1,
            "shadow_pool_independent_groups": 1,
            "shadow_comparison_class": "NO_COMPARISON",
            "shadow_expansion_eligible": True,
            "shadow_expansion_outcome": "ADDED_NEW_TIMING_GROUP",
            "shadow_extra_fetch_attempts": 1,
            "shadow_extra_fetch_successes": 1,
        }
        for i in range(3)
    ]
    analysis = mod.shadow_comparison_analysis(records)
    assert analysis["expansion"]["marginal_meaningful_rate"] is None
    assert "INSUFFICIENT_SAMPLE" in mod.render_shadow_report(analysis)

    comparison = mod.fetch_limit_comparison(records)
    assert comparison["status"] == "INSUFFICIENT_SAMPLE"
    assert "INSUFFICIENT_SAMPLE" in mod._render_fetch_limit_comparison(comparison)


def test_report_treats_unknown_cost_as_unknown():
    import importlib.util

    spec = importlib.util.spec_from_file_location("cal_unk", "tools/calibrate_sync.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    records = [
        {
            "phase": "serve",
            "reference_trust": "acceptable",
            "shadow_reference_id": f"x{i}",
            "shadow_reference_changed": False,
            "shadow_pool_materialized": True,
            "shadow_pool_size": 1,
            "shadow_pool_independent_groups": 1,
            "shadow_comparison_class": "NO_COMPARISON",
            "shadow_expansion_eligible": True,
            "shadow_expansion_outcome": "ADDED_DUPLICATE_GROUP",
            "shadow_extra_fetch_attempts": 1,
            "shadow_extra_fetch_successes": 1,
            "shadow_extra_fetch_bytes": None,
            "shadow_extra_fetch_latency_ms": None,
        }
        for i in range(40)
    ]
    analysis = mod.shadow_comparison_analysis(records)
    exp = analysis["expansion"]
    assert exp["mean_bytes"] is None
    assert exp["mean_latency_ms"] is None
    assert exp["bytes_unknown"] == 40
    assert exp["latency_unknown"] == 40
    rendered = mod.render_shadow_report(analysis)
    assert "unknown" in rendered
    assert json.dumps(analysis, default=str)
