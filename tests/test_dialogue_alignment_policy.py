"""Shared accuracy policy and retained-pack lifecycle, using actual bytes."""
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.services.sync.candidate_reconstruction import reconstruct_with_original_dialogue
from app.services.sync.query import ResolvedReference
from app.services.sync.season_pack import (
    lineages_match,
    lookup_pack_member,
    promote_season_pack,
    retain_season_pack,
    video_lineage,
)
from app.services.sync.unified_verification import verify_candidate
from tests.test_season_pack import _orchestrator_harness, _pack_zip, _query, _series_meta
from tests.test_two_point_primary import _sentences
from tests.test_unified_verification import _linear_pair, _true_candidate
from two_point_sync.parser import Cue, format_srt, parse_srt


@pytest.mark.parametrize("offset,scale", [(96_000, 1), (-30_000, 1), (5000, 1.001)])
@pytest.mark.parametrize("algorithm", ["timeline_affine", "two_point", "primary_regime", "alass"])
def test_algorithm_and_movement_do_not_veto(offset, scale, algorithm):
    target, reference = _linear_pair(20, scale=scale, offset_ms=offset)
    candidate = _true_candidate(target, scale, offset)
    assert verify_candidate(target, reference, candidate, candidate_algorithm=algorithm).proposed_verdict == "VERIFIED"


def test_missing_opening_has_explicit_scope():
    target, reference = _linear_pair(30)
    reference = format_srt(parse_srt(reference)[10:])
    verdict = verify_candidate(target, reference, _true_candidate(target, 1, 5000))
    assert verdict.proposed_verdict == "VERIFIED", verdict.failure_reasons
    assert verdict.coverage["opening"] == 0
    assert verdict.regional_result["unmeasured_cues"] > 0
    assert verdict.regional_result["global_alignment_claim"] is False


def test_negative_shift_to_genuinely_aligned_zero_is_not_a_clamp():
    target, reference = _linear_pair(20, offset_ms=-60_000)
    candidate = _true_candidate(target, 1, -60_000)
    assert parse_srt(candidate)[0].start_ms == 0
    verdict = verify_candidate(target, reference, candidate)
    assert verdict.proposed_verdict == "VERIFIED", verdict.failure_reasons


def test_nonlinear_candidate_measured_without_constant_shift_requirement():
    target, reference = _linear_pair(30)
    def move(text):
        return format_srt([Cue(number=c.number, order=c.order, start_ms=c.start_ms + (i // 10) * 1000,
                              end_ms=c.end_ms + (i // 10) * 1000, text=c.text)
                           for i, c in enumerate(parse_srt(text))])
    candidate = move(_true_candidate(target, 1, 5000))
    assert verify_candidate(target, move(reference), candidate).proposed_verdict == "VERIFIED"
    assert verify_candidate(target, move(reference), target).proposed_verdict == "REJECTED"


def test_alass_split_rejoins_pristine_content():
    original = "1\n00:01:00,000 --> 00:01:04,000\nA distinctive sentence about the old lighthouse\n"
    split = ("1\n00:01:05,000 --> 00:01:07,000\nA distinctive sentence\n\n"
             "2\n00:01:07,000 --> 00:01:09,000\nabout the old lighthouse\n")
    restored = reconstruct_with_original_dialogue(original, split)
    assert restored
    assert parse_srt(restored)[0].text == parse_srt(original)[0].text
    assert len(parse_srt(restored)) == 1
    assert reconstruct_with_original_dialogue(original, split.replace("lighthouse", "castle")) is None


@pytest.mark.parametrize("episode", [4, 5])
def test_real_dexter_dialogue_verdict(episode):
    from tests.private_fixtures import read_subtitle_text

    folder = Path(__file__).parent / "fixtures" / "dialogue_alignment" / f"dexter_s08e{episode:02}"
    target, reference, candidate = (read_subtitle_text(folder / name)
                                    for name in ("target.srt", "reference.srt", "candidate.srt"))
    verdict = verify_candidate(target, reference, candidate, candidate_algorithm="two_point")
    assert verdict.proposed_verdict == "VERIFIED", verdict.failure_reasons
    assert verdict.residual_statistics["max"] < 2000
    assert verdict.regional_result["global_alignment_claim"] is False
    assert [c.text for c in parse_srt(candidate)] == [c.text for c in parse_srt(target)]
    assert verify_candidate(target, reference, _true_candidate(target, 1, 50_000)).proposed_verdict == "REJECTED"


def test_real_e02_requires_more_than_original_or_guessed_constant_shift():
    from app.services.sync.hybrid_router import classify_pair
    from tests.private_fixtures import read_subtitle_text

    folder = Path(__file__).parent / "fixtures" / "dialogue_alignment" / "dexter_s08e02"
    target, reference = (read_subtitle_text(folder / name)
                         for name in ("target.srt", "reference.srt"))
    assert classify_pair(target, reference)["route"] == "nonlinear"
    assert verify_candidate(target, reference, target).proposed_verdict != "VERIFIED"


def test_legacy_verified_seed_recovers_pack_read_only(tmp_path):
    import hashlib
    import json

    from app.services.sync_service import sanitize_subtitle
    root = tmp_path / "references"
    _, reference = _sentences(14, shift_ms=5000, paraphrase=(4, 8, 12))
    reference = reference.replace("\n", "\r\n")
    query = _query("Penny.Dreadful.S01E01.BluRay-d3g.mkv", 1)
    key = retain_season_pack(raw=_pack_zip({f"Penny.Dreadful.S01E{ep:02}.srt": reference for ep in (1, 2)}),
        query=query, provider_label="subdl", pack_candidate="pack", source_download_digest="legacy", root=root)
    assert key
    index = root / "season_packs" / key / "index.json"
    before = index.read_bytes()
    stem = query.video_scope_stem
    artifact = stem + ".video.subdl.edition.reference.srt"
    payload = sanitize_subtitle(reference).encode()
    (root / artifact).write_bytes(payload)
    metadata = dict(member_episode=1, member_season=1, source_download_digest="legacy")
    (root / (artifact + ".json")).write_text(json.dumps(metadata))
    sticky = dict(artifact=artifact, stem=stem, verification_method="unified",
                  evidence=dict(sync_state="verified_resynced"), content_sha=hashlib.sha256(payload).hexdigest())
    (root / (stem + ".sticky.json")).write_text(json.dumps(sticky))
    later = _query("Penny.Dreadful.S01E02.BluRay-d3g.mkv", 2)
    assert lookup_pack_member(later, root=root)
    assert index.read_bytes() == before
    metadata["source_download_digest"] = "different"
    (root / (artifact + ".json")).write_text(json.dumps(metadata))
    assert lookup_pack_member(later, root=root) is None


async def test_pack_first_does_not_require_provider_planning_interface(monkeypatch, tmp_path):
    from tests.test_season_pack import ROVERS_E02, _retain
    target, reference = _sentences(20, shift_ms=5000, paraphrase=(4, 8, 12))
    _retain(tmp_path, members={"Penny.Dreadful.S01E01.Bluray-AOS.srt": reference,
                              "Penny.Dreadful.S01E02.Bluray-AOS.srt": reference})
    orch, _, service = _orchestrator_harness(monkeypatch, ResolvedReference(None),
                                            pack_dir=tmp_path / "references")
    del orch._external_strategy.plan_candidates
    resolver = orch._external_strategy.resolve_with_provenance
    out = await orch.evaluate_and_sync(target.encode(), _series_meta(ROVERS_E02, 2), "no-planner", auto_sync=True)
    assert out.authorization.may_serve and out != target.encode()
    resolver.assert_not_called()
    service.sync_async.assert_not_called()


async def test_one_acquisition_winner_promotion_and_e01_through_e05_reuse(monkeypatch, tmp_path):
    """Recreate orchestrators; no episode becomes a new prerequisite seed."""
    root = tmp_path / "references"
    targets, members = {}, {}
    for episode in range(1, 6):
        target, reference = _sentences(20, shift_ms=5000, paraphrase=(4, 8, 12))
        # Different actual dialogue per episode, so wrong extraction cannot pass.
        target = target.replace("spoken aloud", f"spoken aloud in episode {episode}")
        reference = reference.replace("spoken aloud", f"spoken aloud in episode {episode}")
        targets[episode] = target
        members[f"Penny.Dreadful.S01E{episode:02}.BluRay-d3g.srt"] = reference
    def video(episode):
        return f"Penny.Dreadful.S01E{episode:02}.1080p.BluRay.x264-d3g.mkv"
    raw = _pack_zip(members)
    query = _query(video(1), 1)
    download = AsyncMock(return_value=raw)
    key = retain_season_pack(raw=await download(), provider_label="subdl", pack_candidate="pack.zip",
                             source_download_digest="one-pack", query=query, root=root)
    assert key
    assert (root / "season_packs" / key / "archive.zip").read_bytes() == raw
    assert lookup_pack_member(_query(video(4), 4), root=root) is None  # acquisition is not proof
    resolved = ResolvedReference(members[next(iter(members))], kind="edition",
        candidate=next(iter(members)), member_season=1, member_episode=1,
        source_download_digest="one-pack")
    orch, planned, _ = _orchestrator_harness(monkeypatch, resolved, pack_dir=root)
    orch._external_strategy.plan_candidates = AsyncMock(return_value=[("release", "provider")])
    orch._external_strategy.download_planned_candidate = AsyncMock(return_value=resolved)
    out = await orch.evaluate_and_sync(targets[1].encode(), _series_meta(video(1), 1), "seed", auto_sync=True)
    assert out != targets[1].encode()
    assert lookup_pack_member(_query(video(5), 5), root=root)
    for episode in range(2, 6):
        for subtitle_id in ("first", "second"):
            orch, planned, service = _orchestrator_harness(monkeypatch, ResolvedReference(None), pack_dir=root)
            out = await orch.evaluate_and_sync(targets[episode].encode(), _series_meta(video(episode), episode), subtitle_id, auto_sync=True)
            assert out != targets[episode].encode()
            assert out.authorization.may_serve
            assert planned == []
            service.sync_async.assert_not_called()
    download.assert_awaited_once()
    member_path = root / "season_packs" / key / "ep04.srt"
    member_path.write_text("corrupted", encoding="utf-8")
    assert lookup_pack_member(_query(video(4), 4), root=root) is None
    assert lookup_pack_member(_query(video(5).replace("d3g", "OTHER"), 5), root=root) is None


def test_cut_identity_changes_lineage():
    regular = video_lineage("Show.S01E01.1080p.BluRay.x264-d3g.mkv")
    extended = video_lineage("Show.S01E02.Extended.1080p.BluRay.x264-d3g.mkv")
    assert not lineages_match(regular, extended)


@pytest.mark.parametrize("source", ["HDTV", "BluRay", "WEB-DL"])
async def test_known_episode_identity_and_dialogue_override_release_labels(monkeypatch, source):
    from app.services.sync import hybrid_router as router
    from app.services.sync import orchestrator as om
    from tests.test_two_point_primary import _harness

    target, reference = _sentences(20, shift_ms=96_000, paraphrase=(4, 8, 12))
    metadata = dict(_series_meta("Penny.Dreadful.S01E01.BluRay-d3g.mkv", 1))
    resolved = ResolvedReference(reference, kind="edition", member_season=1, member_episode=1,
        candidate=f"Penny.Dreadful.S01E01.{source}-OTHER.srt", reference_trust="acceptable")
    orch, _, service, _ = _harness(monkeypatch, resolved, strong_binding=False)
    assert router.unified_authority_enabled(metadata)
    assert om.SyncOrchestrator._target_binding(resolved, metadata).supports_verified
    out = await orch.evaluate_and_sync(target.encode(), metadata, "cross-release", auto_sync=True)
    assert out != target.encode() and out.authorization.may_serve
    service.sync_async.assert_not_called()
    wrong = ResolvedReference(reference, candidate=f"Penny.Dreadful.S01E02.{source}-OTHER.srt",
                              member_season=1, member_episode=2)
    assert not om.SyncOrchestrator._target_binding(wrong, metadata).supports_verified


def test_crlf_members_survive_retention_and_checksum(tmp_path):
    """Real archive line endings must not become a false corruption verdict."""
    root = tmp_path / "references"
    _, reference = _sentences(14, shift_ms=5000, paraphrase=(4, 8, 12))
    reference = reference.replace("\n", "\r\n")
    query = _query("Penny.Dreadful.S01E01.BluRay-d3g.mkv", 1)
    raw = _pack_zip({f"Penny.Dreadful.S01E{ep:02}.srt": reference for ep in (1, 2)})
    key = retain_season_pack(raw=raw, query=query, provider_label="subdl", pack_candidate="pack",
                             source_download_digest="crlf", root=root)
    assert key
    assert promote_season_pack(query, ResolvedReference(reference, source_download_digest="crlf"), root=root)
    found = lookup_pack_member(_query("Penny.Dreadful.S01E02.BluRay-d3g.mkv", 2), root=root)
    assert found and found["text"] == reference
    assert (root / "season_packs" / key / "ep02.srt").read_bytes() == reference.encode()
    assert lookup_pack_member(_query("Penny.Dreadful.S01E02.BluRay-d3g.mkv", 2, lang="eng"), root=root)
    # A fresh interpreter has no request/orchestrator state to help it reuse
    # this pack. This exercises the state retained by the existing Docker bind.
    import hashlib
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-c",
        "import sys,hashlib; from app.services.sync.season_pack import lookup_pack_member; "
        "from app.services.sync.query import ReferenceQuery; "
        "q=ReferenceQuery(imdb_id='tt2628232',season=1,episode=2,target_lang='ara',"
        "target_filename='Penny.Dreadful.S01E02.BluRay-d3g.mkv'); "
        "hit=lookup_pack_member(q,root=sys.argv[1]); assert hit; "
        "print(hashlib.sha256(hit['text'].encode()).hexdigest())", str(root)],
        capture_output=True, text=True, check=True)
    assert result.stdout.strip() == hashlib.sha256(reference.encode()).hexdigest()


async def test_real_e04_e05_cached_d3g_pack_delivers_without_provider_work(monkeypatch, tmp_path):
    root = tmp_path / "references"
    target_seed, reference_seed = _sentences(20, shift_ms=5000, paraphrase=(4, 8, 12))
    def video(ep):
        return f"Dexter.s8e{ep:02}.1080p.BluRay.10Bit.Dts-HDMa5.1.HEVC-d3g.mkv"
    def meta(ep):
        return dict(_series_meta(video(ep), ep), imdb_id="tt0773262", season=8, title="Dexter")
    from app.services.sync.query import ReferenceQuery
    query = ReferenceQuery(imdb_id="tt0773262", target_filename=video(1), season=8,
                           episode=1, target_lang="ara", languages=("eng", "ara"))
    members = {"Dexter.S08E01.BluRay-d3g.srt": reference_seed}
    folder = Path(__file__).parent / "fixtures" / "dialogue_alignment"
    from tests.private_fixtures import read_subtitle_text

    for ep in (2, 4, 5):
        members[f"Dexter.S08E{ep:02}.BluRay-d3g.srt"] = read_subtitle_text(
            folder / f"dexter_s08e{ep:02}" / "reference.srt")
    key = retain_season_pack(raw=_pack_zip(members), query=query, provider_label="subdl",
                             pack_candidate="Dexter.S08.BluRay-d3g.zip",
                             source_download_digest="real-members", root=root)
    assert key
    resolved = ResolvedReference(reference_seed, candidate="Dexter.S08E01.BluRay-d3g.srt",
                                 kind="edition", member_season=8, member_episode=1,
                                 source_download_digest="real-members")
    orch, _, _ = _orchestrator_harness(monkeypatch, resolved, pack_dir=root)
    orch._external_strategy.plan_candidates = AsyncMock(return_value=[("release", "provider")])
    orch._external_strategy.download_planned_candidate = AsyncMock(return_value=resolved)
    seed = await orch.evaluate_and_sync(target_seed.encode(), meta(1), "seed", auto_sync=True)
    assert seed.authorization.may_serve
    for ep in (4, 5):
        target = read_subtitle_text(
            folder / f"dexter_s08e{ep:02}" / "target.srt").encode()
        orch, planned, service = _orchestrator_harness(monkeypatch, ResolvedReference(None), pack_dir=root)
        result = await orch.evaluate_and_sync(target, meta(ep), f"real-{ep}", auto_sync=True)
        assert result != target and result.authorization.may_serve
        assert [c.text for c in parse_srt(result.decode())] == [c.text for c in parse_srt(target.decode("utf-8-sig"))]
        assert planned == []
        service.sync_async.assert_not_called()
