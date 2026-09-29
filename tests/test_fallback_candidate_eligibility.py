"""Fallback candidate eligibility: "not exact" is not "not compatible".

From a production incident. Target
``Dexter.s8e05.This.little.piggy.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv``,
requested ``Dexter.S08E05.HDTV.XviD-AFG.srt``. The SubDL download returned 502
and tripped its circuit breaker; SubSource was searched, 500 rows scanned, 23
final Arabic candidates returned, and the request still failed with

    No exact-equivalent release for 'Dexter.S08E05.HDTV.XviD-AFG.srt' ...

followed by another 502.

The gate was conflating two different questions. Whether the fallback provider
happens to carry the identical release-name string is not the same question as
whether a subtitle can be synchronized to this target. Answering only the first
one made a synchronizable candidate unreachable.

These tests fix eligibility only. They do not assert that anything synchronizes:
that decision still belongs to the target-bound reference, the +/-20s gate,
alass and verification, none of which are touched here.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.models import SubtitleRelease

TARGET = "Dexter.s8e05.This.little.piggy.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv"
REQUESTED = "Dexter.S08E05.HDTV.XviD-AFG.srt"


def _release(name: str, url: str) -> SubtitleRelease:
    return SubtitleRelease(
        release_name=name, download_url=url, provider="subsource", lang="ara"
    )


def _provider_for(releases, payload: bytes, calls: list[str] | None = None):
    class _Provider:
        name = "subsource"

        def __init__(self):
            self.searches = 0

        async def search_subtitles(self, **kwargs):
            self.searches += 1
            return list(releases)

        async def download_archive(self, download_ref, api_key=None):
            if calls is not None:
                calls.append(download_ref)
            return payload

    return _Provider()


async def _run(provider, *, requested=REQUESTED, target=TARGET, outcome=None, meta=None):
    from app.main import _fallback_download_subsource

    with patch("app.main.SubsourceProvider", new=lambda client: provider):
        return await _fallback_download_subsource(
            imdb_id="tt0773262",
            media_type="series",
            season=8,
            episode=5,
            subsource_key="k",
            target_filename=target,
            lang="ara",
            client=object(),
            requested_release_name=requested,
            requested_uploader=None,
            requested_hearing_impaired=None,
            meta=meta if meta is not None else {"release_name": requested},
            outcome=outcome,
        )


# --- A. exact equivalent exists: still preferred ----------------------------


@pytest.mark.asyncio
async def test_a_exact_release_is_preferred():
    payload = b"1\n00:00:01,000 --> 00:00:04,000\nx\n"
    exact = SubtitleRelease(
        release_name="Dexter.S08E05.HDTV.XviD-AFG",
        download_url="https://s/exact",
        provider="subsource",
        lang="ara",
    )
    calls: list[str] = []
    outcome: dict = {}
    result = await _run(
        _provider_for([_release("Dexter.S08.720p.BluRay.x264-OTHER.srt", "https://s/other"), exact],
                      payload, calls),
        outcome=outcome,
    )
    assert result == payload
    assert calls == ["https://s/exact"]
    assert outcome["category"] == "EXACT_RELEASE_SELECTED"


# --- B. the incident: no exact release, but a compatible candidate ----------


@pytest.mark.asyncio
async def test_b_dexter_no_exact_release_still_reaches_synchronization():
    """The production failure, and the behaviour that must replace it.

    Previously this returned None before any download. Now the compatible
    candidate is fetched and handed to the pipeline.
    """
    payload = b"1\n00:00:10,160 --> 00:00:12,000\nx\n"
    compatible = _release("Dexter.S08E05.1080p.BluRay.x265-ROVERS.srt", "https://s/rovers")
    calls: list[str] = []
    outcome: dict = {}
    result = await _run(_provider_for([compatible], payload, calls), outcome=outcome)

    assert result == payload, "a target-compatible candidate must not be discarded"
    assert calls == ["https://s/rovers"]
    assert outcome["exact_candidates"] == 0
    assert outcome["compatible_candidates"] == 1
    assert outcome["category"] == "NO_EXACT_RELEASE"
    # ROVERS shares the target's 1080p/BluRay source family but not the PiR8
    # group, so SOURCE_FAMILY is the honest class. This is the realistic
    # incident: no provider offers the exact release, and the best available
    # candidate is same-source rather than identical.
    assert outcome["reason"] == "COMPATIBLE_RELEASE"


# --- C. weak but not hard-rejected: attempted, never auto-verified ----------


@pytest.mark.asyncio
async def test_c_weak_candidate_is_eligible_but_reports_its_weakness():
    payload = b"1\n00:00:01,000 --> 00:00:04,000\nx\n"
    weak = _release("Dexter.S08E05.720p.WEB-DL.x264-GRP.srt", "https://s/weak")
    outcome: dict = {}
    result = await _run(_provider_for([weak], payload), outcome=outcome)
    assert result == payload
    assert outcome["reason"] in {"WEAK_COMPATIBLE", "COMPATIBLE_RELEASE"}


# --- D/E. hard rejections are absolute --------------------------------------


@pytest.mark.asyncio
async def test_d_wrong_episode_is_hard_rejected():
    calls: list[str] = []
    wrong = _release("Dexter.S08E09.1080p.BluRay.x265-ROVERS.srt", "https://s/wrong")
    outcome: dict = {}
    result = await _run(_provider_for([wrong], b"", calls), outcome=outcome)
    assert result is None
    assert calls == []
    assert outcome["category"] == "NO_COMPATIBLE_CANDIDATE"


@pytest.mark.asyncio
async def test_e_wrong_season_is_hard_rejected():
    calls: list[str] = []
    wrong = _release("Dexter.S01E05.1080p.BluRay.x265-ROVERS.srt", "https://s/wrong")
    outcome: dict = {}
    result = await _run(_provider_for([wrong], b"", calls), outcome=outcome)
    assert result is None
    assert calls == []


# --- F. nothing compatible at all: clean, distinct failure ------------------


@pytest.mark.asyncio
async def test_f_no_compatible_candidate_is_a_distinct_fact_from_no_exact():
    outcome: dict = {}
    result = await _run(
        _provider_for([_release("Dexter.S08E09.1080p.BluRay.x265-ROVERS.srt", "https://s/x")], b""),
        outcome=outcome,
    )
    assert result is None
    assert outcome["category"] == "NO_COMPATIBLE_CANDIDATE"
    # Zero exact and zero compatible are reported as two separate counts, so a
    # cached "no exact match" is never mistaken for "no subtitles exist".
    assert outcome["exact_candidates"] == 0
    assert outcome["compatible_candidates"] == 0


# --- G/H/I. search cost and deduplication -----------------------------------


@pytest.mark.asyncio
async def test_g_repeated_identical_query_searches_once():
    provider = _provider_for(
        [_release("Dexter.S08E05.1080p.BluRay.x265-ROVERS.srt", "https://s/rovers")], b"x"
    )
    with patch("app.main.SubsourceProvider", new=lambda client: provider):
        from app.main import _fallback_download_subsource

        for _ in range(3):
            await _fallback_download_subsource(
                imdb_id="tt0773262", media_type="series", season=8, episode=5,
                subsource_key="k", target_filename=TARGET, lang="ara", client=object(),
                requested_release_name=REQUESTED, requested_uploader=None,
                requested_hearing_impaired=None, meta={},
            )
    assert provider.searches == 1, "the 5x100 page scan must not repeat"


@pytest.mark.asyncio
async def test_h_materially_different_queries_are_not_shared():
    calls: list[str] = []
    provider = _provider_for(
        [_release("Dexter.S08E05.1080p.BluRay.x265-ROVERS.srt", "https://s/rovers")], b"x", calls
    )
    with patch("app.main.SubsourceProvider", new=lambda client: provider):
        from app.main import _fallback_download_subsource

        for lang in ("ara", "eng"):
            await _fallback_download_subsource(
                imdb_id="tt0773262", media_type="series", season=8, episode=5,
                subsource_key="k", target_filename=TARGET, lang=lang, client=object(),
                requested_release_name=REQUESTED, requested_uploader=None,
                requested_hearing_impaired=None, meta={},
            )
    assert provider.searches == 2


@pytest.mark.asyncio
async def test_i_cache_stores_candidates_not_a_verdict():
    """A cached empty exact-match must not poison a later compatible lookup."""
    first = _provider_for([_release("Dexter.S08E09.1080p.BluRay.x265-ROVERS.srt", "https://s/x")], b"")
    calls: list[str] = []
    await _run(first, outcome={})
    # Same query, but now a genuinely compatible candidate exists upstream.
    second = _provider_for(
        [_release("Dexter.S08E05.1080p.BluRay.x265-ROVERS.srt", "https://s/rovers")], b"y", calls
    )
    result = await _run(second, outcome={})
    assert second.searches == 0, "the candidate list is cached"
    # ...and the cached list is re-classified, not remembered as a verdict.
    assert result is None
    assert calls == []


# --- classification is a label over existing machinery ---------------------


def test_classification_reuses_match_tier_bands():
    from app.main import _classify_fallback_release

    base = TARGET
    same_family = SimpleNamespace(release_name="Dexter.S08E05.1080p.BluRay.x265-ROVERS.srt")
    other_source = SimpleNamespace(release_name="Dexter.S08E05.720p.WEB-DL.x264-GRP.srt")
    wrong_ep = SimpleNamespace(release_name="Dexter.S08E09.1080p.BluRay.x265-ROVERS.srt")

    assert _classify_fallback_release(base, same_family, None)[0] in {
        "SAME_RELEASE_FAMILY",
        "COMPATIBLE_RELEASE",
    }
    assert _classify_fallback_release(base, other_source, None)[0] in {
        "COMPATIBLE_RELEASE",
        "WEAK_COMPATIBLE",
    }
    assert _classify_fallback_release(base, wrong_ep, None)[0] == "INCOMPATIBLE"


def test_classification_without_an_identity_is_incompatible():
    from app.main import _classify_fallback_release

    assert _classify_fallback_release(None, SimpleNamespace(release_name="x.srt"), None)[0] == (
        "INCOMPATIBLE"
    )
