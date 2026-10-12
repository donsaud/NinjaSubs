"""Video-source ordering, independent of weighted metadata completeness."""

from dataclasses import replace
from itertools import permutations

import pytest

from app.models import SubtitleRelease
from app.services import release_scoring_integration as integration
from app.services.release_scoring.engine import evaluate
from app.services.release_scoring.policies import (
    DisplayPolicy,
    ReferencePolicy,
    equivalent_peers,
    rank,
)
from app.services.release_scoring.profiles import candidate_profiles, target_profile
from app.services.sync.external_strategy import _select_candidates_ranked
from app.services.sync.query import ReferenceQuery


def target(source="BluRay"):
    return {"title": "Show", "media_type": "series", "season": 1, "episode": 2,
            "year": 2020, "source": source, "video_codec": "H264"}


def match(identity, metadata, video):
    candidate = candidate_profiles([{"id": identity, "provider": "subdl",
                                     "provider_id": identity, "language": "ara",
                                     "metadata": metadata}])[0]
    return evaluate(target_profile({"metadata": video}), candidate)


@pytest.mark.parametrize("source", ["BluRay", "BDRip", "REMUX", "WEB-DL", "WEBRip", "HDTV", "DVDRip"])
@pytest.mark.parametrize("policy", [DisplayPolicy(), ReferencePolicy(discovery=True)])
@pytest.mark.parametrize("media_type", ["series", "movie"])
def test_exact_video_source_beats_more_complete_cross_source(source, policy, media_type):
    video = target(source)
    if media_type == "movie":
        video = {k: v for k, v in video.items() if k not in {"season", "episode"}}
        video["media_type"] = "movie"
    incomplete = {k: v for k, v in video.items() if k != "year"}
    same = match("same", incomplete, video)
    wrong = match("wrong", {**video, "source": "HDTV" if source != "HDTV" else "BluRay"}, video)
    assert wrong.score > same.score  # Reproduces the old additive-only defect.
    for order in permutations((same, wrong)):
        assert rank(order, policy) == (same, wrong)
    assert same.percentage < wrong.percentage  # Point totals are not rewritten.


@pytest.mark.parametrize("source,family", [("BluRay", "REMUX"), ("REMUX", "BDRip"),
                                           ("BDRip", "BluRay"), ("WEB-DL", "WEBRip"),
                                           ("WEBRip", "WEB-DL")])
def test_exact_then_family_then_cross_source_then_unknown(source, family):
    video = target(source)
    unknown = {k: v for k, v in video.items() if k != "source"}
    results = (match("unknown", unknown, video), match("wrong", {**video, "source": "HDTV"}, video),
               match("family", {**video, "source": family}, video), match("exact", video, video))
    for policy in (DisplayPolicy(), ReferencePolicy(discovery=True)):
        assert [r.candidate.identity for r in rank(results, policy)] == ["exact", "family", "wrong", "unknown"]


def test_unknown_video_source_has_no_invented_preference():
    video = {k: v for k, v in target().items() if k != "source"}
    results = tuple(match(name, {**video, "source": source}, video)
                    for name, source in (("a", "HDTV"), ("b", "BluRay")))
    assert DisplayPolicy().key(results[0])[:4] == DisplayPolicy().key(results[1])[:4]
    assert equivalent_peers(results) == {"a": ("b",), "b": ("a",)}


def test_verified_moviehash_remains_first_and_wrong_episode_still_rejects():
    hm = {"moviehash": "a" * 16, "video_size": 1000, "hash_verified": True}
    video = {**target(), **hm}
    raw = {"id": "hash", "provider": "opensubtitles", "provider_id": "123", "language": "ara",
           "hash_provenance": "opensubtitles_hash_query_literal_true",
           "metadata": {**video, "source": "HDTV"}}
    proof = evaluate(target_profile({"metadata": video}), candidate_profiles([raw])[0])
    same = match("same", target(), video)
    assert proof.verified_hash
    for policy in (DisplayPolicy(), ReferencePolicy()):
        assert rank((same, proof), policy)[0] == proof
        assert rank((same, replace(proof, verified_hash=False, score=same.score)), policy)[0] == same
    wrong = match("episode", {**target(), "episode": 3}, video)
    assert wrong.rejection_reasons and not rank((wrong,), DisplayPolicy())


def test_source_distinctions_are_not_false_evidence_equivalence():
    video = target()
    exact = match("exact", video, video)
    family = match("family", {**video, "source": "REMUX"}, video)
    assert exact.score == family.score
    assert equivalent_peers((exact, family)) == {"exact": (), "family": ()}


def test_display_refresh_and_reference_adapter_use_video_not_selected_subtitle():
    video = "Show.S01E02.1080p.BluRay.x264-GROUP.mkv"
    same = SubtitleRelease(release_name="Show.S01.BluRay.x264-OTHER.zip", download_url="/same", provider="subdl")
    wrong = SubtitleRelease(release_name="Show.S01E02.1080p.HDTV.x264-GROUP.srt", download_url="/wrong", provider="subdl")
    q = ReferenceQuery(imdb_id="tt1234567", target_filename=video, media_type="series",
                       season=1, episode=2, target_sub_release_name=wrong.release_name)
    assert _select_candidates_ranked([wrong, same], q) == [same, wrong]
    assert integration.display_order(video, [wrong, same], preferred_languages=["ara"]) == [same, wrong]
    # Repeated final display refresh cannot undo the source-first ordering.
    assert integration.display_order(video, [same, wrong], preferred_languages=["ara"]) == [same, wrong]
    assert not integration.reference_ready(same, q)
    assert integration.reference_ready(same, q, "Show.S01E02.BluRay.x264-OTHER.srt")
