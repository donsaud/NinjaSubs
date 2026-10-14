"""Season-pack reference reuse: same video-release lineage only.

A pack archive downloaded for one episode is indexed once; later episodes
from video files of the same lineage (series + season + source + group)
reuse their episode member locally with no provider search and no download.
Unknown lineage never matches; different source/group never matches; each
episode uses its own member; verification downstream is unchanged; the first
episode is never delayed for pack acquisition (demand-driven only).
"""

from __future__ import annotations

import io
import pathlib
import zipfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.services.sync import hybrid_router as router
from app.services.sync import orchestrator as om
from app.services.sync.query import ReferenceQuery, ResolvedReference
from app.services.sync.season_pack import (
    lineages_match,
    lookup_pack_member,
    promote_season_pack,
    retain_season_pack,
    season_pack_reference,
    video_lineage,
)
from app.services.sync_cache import SyncCache
from tests.test_two_point_primary import _sentences

ROVERS_E01 = "Penny.Dreadful.S01E01.1080p.BluRay.x264-ROVERS.mkv"
ROVERS_E02 = "Penny.Dreadful.S01E02.1080p.BluRay.x264-ROVERS.mkv"
NTB_E03 = "Penny.Dreadful.S01E03.1080p.WEB-DL-NTb.mkv"


def _srt(text: str, start_ms: int = 60_000) -> str:
    return (f"1\n00:01:00,000 --> 00:01:02,000\n{text}\n\n"
            f"2\n00:02:00,000 --> 00:02:02,000\nSecond line {text}\n")


def _pack_zip(members: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, text in members.items():
            archive.writestr(name, text.encode("utf-8"))
    return buffer.getvalue()


def _pack_members() -> dict[str, str]:
    return {
        "Penny.Dreadful.S01E01.Bluray-AOS.srt": _srt("Episode one dialogue here"),
        "Penny.Dreadful.S01E02.Bluray-AOS.srt": _srt("Episode two dialogue here"),
    }


def _query(video: str, episode: int, lang: str = "ara") -> ReferenceQuery:
    return ReferenceQuery(
        imdb_id="tt2628232", target_filename=video, media_type="series",
        season=1, episode=episode, target_lang=lang,
        languages=("eng", "ara"))


def _retain(tmp_path: pathlib.Path, video: str = ROVERS_E01,
            episode: int = 1,
            members: dict[str, str] | None = None) -> str | None:
    raw = _pack_zip(members if members is not None else _pack_members())
    query = _query(video, episode)
    key = retain_season_pack(
        raw=raw, provider_label="subsource",
        pack_candidate="Penny.Dreadful.S01.Bluray-AOS.zip",
        source_download_digest="packdigest1234", query=query,
        root=tmp_path / "references")
    if key:
        reference = ResolvedReference((members or _pack_members())[next(name for name in (members or _pack_members()) if f"E{episode:02d}" in name)], source_download_digest="packdigest1234")
        assert promote_season_pack(query, reference, root=tmp_path / "references")
    return key


# ------------------------------------------------------------- lineage ---

def test_lineage_parses_source_and_group() -> None:
    lineage = video_lineage(ROVERS_E01)
    assert lineage is not None
    assert lineage["source"] == "bluray"
    assert lineage["group"] == "rovers"
    webdl = video_lineage("Penny.Dreadful.S01E03.1080p.WEB-DL-NTb.mkv")
    assert webdl is not None
    assert webdl["source"] == "webdl"
    assert webdl["group"] == "ntb"


def test_lineage_missing_filename_is_none() -> None:
    assert video_lineage(None) is None
    assert video_lineage("") is None


def test_match_requires_known_equal_source_and_group() -> None:
    rovers = video_lineage(ROVERS_E01)
    assert lineages_match(rovers, video_lineage(ROVERS_E02)) is True
    assert lineages_match(rovers, video_lineage(NTB_E03)) is False
    assert lineages_match(rovers, {"source": "bluray", "group": None}) is False
    assert lineages_match(rovers, {"source": None, "group": "rovers"}) is False
    assert lineages_match(None, rovers) is False
    assert lineages_match(rovers, None) is False
    assert lineages_match({"source": "bluray", "group": "rovers"},
                           {"source": "bluray", "group": "ROVERS"}) is True


# --------------------------------------------------------------- store ---

def test_retain_and_lookup_round_trip(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "references"
    key = _retain(tmp_path)
    assert key
    found = lookup_pack_member(_query(ROVERS_E02, 2), root=root)
    assert found is not None
    assert "Episode two dialogue here" in found["text"]
    assert found["member_episode"] == 2
    assert found["member_season"] == 1
    assert found["provider"] == "subsource"
    assert found["source_download_digest"] == "packdigest1234"
    # Episode one member is its own file, never episode two's text.
    one = lookup_pack_member(_query(ROVERS_E01, 1), root=root)
    assert one is not None and "Episode one dialogue here" in one["text"]


def test_lookup_rejects_different_lineage(tmp_path: pathlib.Path) -> None:
    _retain(tmp_path)
    assert lookup_pack_member(_query(NTB_E03, 3),
                              root=tmp_path / "references") is None


def test_lookup_rejects_unknown_lineage(tmp_path: pathlib.Path) -> None:
    _retain(tmp_path)
    assert lookup_pack_member(_query("random_video_file.mkv", 2),
                              root=tmp_path / "references") is None


def test_lookup_rejects_missing_episode(tmp_path: pathlib.Path) -> None:
    _retain(tmp_path)
    assert lookup_pack_member(_query(ROVERS_E02, 7),
                              root=tmp_path / "references") is None


def test_lookup_empty_store_is_none(tmp_path: pathlib.Path) -> None:
    assert lookup_pack_member(_query(ROVERS_E02, 2),
                              root=tmp_path / "references") is None


def test_retain_rejects_single_file_and_non_archive(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "references"
    query = _query(ROVERS_E01, 1)
    single = _pack_zip({"only.srt": _srt("lonely")})
    assert retain_season_pack(raw=single, provider_label="subsource",
                              pack_candidate="single.zip",
                              source_download_digest="d1", query=query,
                              root=root) is None
    assert retain_season_pack(raw=_srt("not an archive").encode(),
                              provider_label="subsource", pack_candidate="x.srt",
                              source_download_digest="d2", query=query,
                              root=root) is None
    assert retain_season_pack(raw=b"PK\x03\x04corrupt",
                              provider_label="subsource", pack_candidate="bad.zip",
                              source_download_digest="d3", query=query,
                              root=root) is None
    assert not (root / "season_packs").exists()


def test_retain_rejects_unknown_first_video_lineage(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "references"
    query = _query("mystery.mkv", 1)
    assert retain_season_pack(raw=_pack_zip(_pack_members()),
                              provider_label="subsource", pack_candidate="pack.zip",
                              source_download_digest="d9", query=query,
                              root=root) is None


def test_pack_reference_shape(tmp_path: pathlib.Path, monkeypatch) -> None:
    import app.services.sync.season_pack as pack_module

    _retain(tmp_path)
    monkeypatch.setattr(pack_module, "pack_root",
                        lambda root=None: tmp_path / "references")
    resolved = season_pack_reference(_query(ROVERS_E02, 2))
    assert resolved is not None
    assert isinstance(resolved, ResolvedReference)
    assert resolved.kind == "edition"
    assert resolved.from_season_pack is True
    assert resolved.member_episode == 2
    assert "Episode two dialogue here" in (resolved.text or "")


# ---------------------------------------------------------- orchestrator ---

def _orchestrator_harness(monkeypatch, candidates, pack_dir=None):
    """Gate ON; strategy with observable planning; stubbed ALASS."""
    import app.services.sync.season_pack as pack_module

    monkeypatch.setattr(router, "_enabled", True)
    monkeypatch.delenv(router.TEST_SCOPE_ENV, raising=False)
    monkeypatch.setattr(om.settings, "ENABLE_SUBTITLE_SYNC", True)
    monkeypatch.setattr(om.SyncOrchestrator, "_audit_serve", lambda self, *a, **k: None)
    monkeypatch.setattr(om.SyncOrchestrator, "_record_outcome", lambda self, *a, **k: None)
    monkeypatch.setattr(om.SyncOrchestrator, "_try_timeline_affine",
                        AsyncMock(return_value=None))
    if pack_dir is not None:
        monkeypatch.setattr(pack_module, "pack_root", lambda root=None: pack_dir)
    planned: list = []

    async def _plan(query):
        planned.append(query)
        return []

    strategy = SimpleNamespace(
        plan_candidates=_plan,
        download_planned_candidate=AsyncMock(side_effect=AssertionError("no download")),
        cache=None,
        resolve_with_provenance=AsyncMock(return_value=candidates))
    service = SimpleNamespace(sync_async=AsyncMock(return_value=None))
    orch = om.SyncOrchestrator(sync_cache=SyncCache(), external_strategy=strategy,
                              sync_service=service)
    return orch, planned, service


def _series_meta(video: str, episode: int) -> dict:
    return {"imdb_id": "tt2628232", "media_type": "series", "season": 1,
            "episode": episode, "lang": "ara", "target_filename": video,
            "title": "Penny Dreadful"}


async def test_pack_hit_skips_planning_and_download(monkeypatch, tmp_path) -> None:
    """E02 same lineage: pack member serves via Two-Point, no search/download/ALASS."""
    target, member = _sentences(14, shift_ms=5_000, paraphrase=(4, 8, 12))
    _retain(tmp_path, video=ROVERS_E02, episode=2, members={
        "Penny.Dreadful.S01E01.Bluray-AOS.srt": _srt("Episode one dialogue here"),
        "Penny.Dreadful.S01E02.Bluray-AOS.srt": member,
    })
    orch, planned, service = _orchestrator_harness(
        monkeypatch, ResolvedReference(None), pack_dir=tmp_path / "references")
    out = await orch.evaluate_and_sync(target.encode(), _series_meta(ROVERS_E02, 2),
                                       "sub", auto_sync=True)
    assert out != target.encode()
    assert planned == [], "provider planning must not run on a pack hit"
    service.sync_async.assert_not_called()


async def test_lineage_mismatch_falls_back_to_planning(monkeypatch, tmp_path) -> None:
    """E03 WEB-DL-NTb: pack ignored, normal discovery proceeds."""
    _retain(tmp_path)
    orch, planned, _service = _orchestrator_harness(
        monkeypatch, ResolvedReference(None), pack_dir=tmp_path / "references")
    target, _ = _sentences(6)
    out = await orch.evaluate_and_sync(target.encode(), _series_meta(NTB_E03, 3),
                                       "sub", auto_sync=True)
    assert out == target.encode()
    assert len(planned) >= 1, "planning must proceed without a lineage match"


async def test_missing_episode_falls_back_to_planning(monkeypatch, tmp_path) -> None:
    """Pack without E07: no member, normal discovery proceeds."""
    _retain(tmp_path)
    orch, planned, _service = _orchestrator_harness(
        monkeypatch, ResolvedReference(None), pack_dir=tmp_path / "references")
    target, _ = _sentences(6)
    out = await orch.evaluate_and_sync(target.encode(), _series_meta(ROVERS_E02, 7),
                                       "sub", auto_sync=True)
    assert out == target.encode()
    assert len(planned) >= 1
