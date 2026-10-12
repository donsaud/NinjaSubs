"""General weighted reference discovery, execution, and archive-readiness contracts."""

import io
import zipfile
from unittest.mock import AsyncMock

import pytest

from app.models import SubtitleRelease
from app.services import release_scoring_integration as integration
from app.services.sync.external_strategy import ExternalExactStrategy, _select_candidates_ranked
from app.services.sync.query import ReferenceQuery

SOURCES = [
    ("BluRay", "D3G", "WEB-DL"),
    ("WEB-DL", "NTb", "HDTV"),
    ("HDTV", "AFG", "BluRay"),
    ("WEBRip", "GROUP", "HDTV"),
    ("DVDRip", "GROUP", "WEB-DL"),
]


def release(source="BluRay", group="OTHER", *, pack=False, provider="subdl", episode=2):
    name = f"Show.{'S01' if pack else f'S01E{episode:02d}'}.1080p.{source}.x264-{group}"
    return SubtitleRelease(
        release_name=name, download_url="/" + name, provider=provider, lang="eng"
    )


def query(source="BluRay", group="D3G"):
    return ReferenceQuery(
        imdb_id="tt1234567",
        media_type="series",
        season=1,
        episode=2,
        target_filename=f"Show.S01E02.1080p.{source}.x264-{group}.mkv",
        target_sub_release_name="Unrelated.S01E02.HDTV-AFG",
    )


@pytest.mark.parametrize("source,group,wrong", SOURCES)
def test_resolved_same_source_beats_cross_source_even_with_wrong_group(source, group, wrong):
    same, cross, exact = release(source), release(wrong, group), release(source, group)
    assert _select_candidates_ranked([cross, same, exact], query(source, group)) == [
        exact,
        same,
        cross,
    ]


def test_unresolved_pack_discovery_and_actual_selected_member_readiness():
    pack = release(pack=True)
    q = query()
    assert _select_candidates_ranked([pack], q) == [pack]
    assert not integration.reference_ready(pack, q)
    assert integration.reference_ready(pack, q, "Show.S01E02.1080p.BluRay.x264-OTHER.srt")
    assert not integration.reference_ready(pack, q, "Show.S01E03.1080p.BluRay.x264-OTHER.srt")
    assert not integration.reference_ready(pack, q, "generic.srt")
    assert integration.reference_ready(pack, q, "2.srt")
    assert not integration.reference_ready(pack, q, "3.srt")


def test_moviehash_requires_actual_independent_query_provenance():
    q = query()
    q.video_hash = "1ec51ed4797c63ff"
    q.video_size = 5727011301
    ordinary = release()
    exact = release("WEB-DL", provider="opensubtitles")
    exact.is_hash_match = True
    assert _select_candidates_ranked([exact, ordinary], q)[0] is ordinary
    integration.hash_proof(exact, {"moviehash": q.video_hash, "moviebytesize": q.video_size}, True)
    assert _select_candidates_ranked([ordinary, exact], q)[0] is exact


@pytest.mark.parametrize("source,group,wrong", SOURCES)
def test_season_enrichment_only_when_no_ready_source_compatible_episode(source, group, wrong):
    q = query(source, group)
    assert ExternalExactStrategy._needs_season_enrichment([release(wrong)], q)
    assert not ExternalExactStrategy._needs_season_enrichment([release(source)], q)
    assert ExternalExactStrategy._needs_season_enrichment([release(source, pack=True)], q)


@pytest.mark.asyncio
async def test_weighted_plan_directly_controls_real_download_attempts(tmp_path, monkeypatch):
    from app.services.sync.cache import ReferenceDiskCache

    q = query()
    same = release()
    cross = release("WEB-DL", "D3G")
    provider = AsyncMock()
    provider.is_breaker_open = lambda: False
    strategy = ExternalExactStrategy(
        subdl_provider=provider, cache=ReferenceDiskCache(tmp_path), min_bytes=1
    )
    strategy._gather_candidates = AsyncMock(return_value=[(cross, provider), (same, provider)])
    body = "1\n00:00:01,000 --> 00:00:02,000\nDialogue.\n"
    provider.download_archive = AsyncMock(return_value=body.encode())
    monkeypatch.setattr(
        "app.services.sync.external_strategy._REFERENCE_DOWNLOAD_MIN_INTERVAL_SECONDS", 0
    )
    plan = await strategy.plan_candidates(q)
    assert [c for c, _ in plan] == [same, cross]
    shared = {"downloaded": 0, "seen": set()}
    for c, p in plan:
        await strategy.download_planned_candidate(c, p, q, None, shared)
    assert [call.args[0] for call in provider.download_archive.call_args_list] == [
        same.download_url,
        cross.download_url,
    ]


@pytest.mark.asyncio
async def test_actual_archive_member_rechecked_before_reference_return(tmp_path):
    from app.services.sync.cache import ReferenceDiskCache

    provider = AsyncMock()
    pack = release(pack=True)
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w") as z:
        z.writestr(
            "Show.S01E02.1080p.BluRay.x264-OTHER.srt", "1\n00:00:01,000 --> 00:00:02,000\nHello.\n"
        )
    provider.download_archive = AsyncMock(return_value=raw.getvalue())
    strategy = ExternalExactStrategy(
        subdl_provider=provider, cache=ReferenceDiskCache(tmp_path), min_bytes=1
    )
    result = await strategy._download_candidate(pack, provider, query())
    assert result.text and result.kind == "edition"


def test_wrong_identity_and_unknown_source_cannot_be_ready():
    q = query()
    wrong = release(episode=3)
    assert _select_candidates_ranked([wrong], q) == []
    unknown = SubtitleRelease(
        release_name="Show.S01E02", download_url="/unknown", provider="subdl", lang="ara"
    )
    assert _select_candidates_ranked([unknown], q) == [unknown]
    assert not integration.reference_ready(unknown, q)
