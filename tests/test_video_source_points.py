"""Numerical source evidence, consistent UI/reference points and safety basis."""

import json
from itertools import permutations

import pytest

from app.models import SubtitleRelease
from app.services import release_scoring_integration as integration
from app.services.release_scoring.engine import evaluate
from app.services.release_scoring.policies import DisplayPolicy, ReferencePolicy, rank
from app.services.release_scoring.profiles import candidate_profiles, target_profile
from app.services.sync.external_strategy import ExternalExactStrategy
from app.services.sync.query import ReferenceQuery


@pytest.mark.parametrize("source,wrong", [("BluRay", "HDTV"), ("REMUX", "WEB-DL"),
                                          ("BDRip", "HDTV"), ("WEB-DL", "BluRay"),
                                          ("WEBRip", "HDTV"), ("HDTV", "WEB-DL"),
                                          ("DVDRip", "WEB-DL")])
@pytest.mark.parametrize("media_type", ["series", "movie"])
def test_source_points_overcome_incomplete_metadata_without_changing_raw_evidence(source, wrong, media_type):
    video = {"title": "Example", "year": 2020, "source": source, "media_type": media_type}
    if media_type == "series":
        video.update(season=1, episode=2)
    same_metadata = {k: v for k, v in video.items() if k != "year"}
    rows = [{"id": "same", "provider_id": "same", "provider": "subdl", "language": "ara", "metadata": same_metadata},
            {"id": "wrong", "provider_id": "wrong", "provider": "subdl", "language": "ara", "metadata": {**video, "source": wrong}}]
    results = tuple(evaluate(target_profile({"metadata": video}), c) for c in candidate_profiles(rows))
    by_id = {r.candidate.identity: r for r in results}
    same, other = by_id["same"], by_id["wrong"]
    assert same.score < other.score and same.percentage < other.percentage
    assert same.ranking_score > other.ranking_score
    assert round(same.ranking_percentage) > round(other.ranking_percentage)
    for result in results:
        assert sum(c.points for c in result.ranking_contributions) == result.ranking_score
        assert result.source_priority_points > 0
    for policy in (DisplayPolicy(), ReferencePolicy(discovery=True)):
        for order in permutations(results):
            assert rank(order, policy)[0] == same
    if media_type == "movie":
        assert "movie_year_unresolved" in same.readiness_reasons
        assert not ReferencePolicy().eligible(same)


def test_ui_and_actual_reference_plan_telemetry_report_identical_points(tmp_path, caplog):
    video = "Dexter.S08E01.1080p.BluRay.TrueHD.AVC-PiR8.mkv"
    same = SubtitleRelease(release_name="Dexter.S08.1080p.BluRay.x264-ROVERS.zip", provider="subdl", download_url="/same", lang="ara")
    wrong = SubtitleRelease(release_name="Dexter.S08E01.720p.HDTV.x264-EVOLVE.srt", provider="subdl", download_url="/wrong", lang="ara")
    q = ReferenceQuery(imdb_id="tt0773262", target_filename=video, media_type="series", season=8, episode=1)
    displayed = integration.display_order(video, [wrong, same], preferred_languages=["ara"])
    assert displayed == [same, wrong]
    assert same.score > wrong.score
    caplog.set_level("INFO")
    ordered, _ = integration.discovery_pairs([wrong, same], q, log_phase="discovery")
    strategy = ExternalExactStrategy(cache=None)
    logged = [json.loads(r.message.split("[weighted-reference] ", 1)[1])
              for r in caplog.records if "[weighted-reference] " in r.message]
    for (release, result), telemetry in zip(ordered, logged, strict=True):
        row = strategy._plan_row(result=result, provider=None, role="ordinary", status="planned")
        assert telemetry["score"] == row["score"] == result.ranking_score
        assert telemetry["percentage"] == row["percentage"] == result.ranking_percentage
        assert release.score == round(telemetry["percentage"])
        assert row["metadata_score"] == result.score
        assert row["metadata_percentage"] == result.percentage
    assert not ordered[0][1].ready_reference_eligible
    assert not integration.reference_ready(same, q)
    assert integration.reference_ready(same, q, "Dexter.S08E01.BluRay.x264-ROVERS.srt")


def test_unknown_video_source_keeps_original_percentage():
    video = {"title": "Example", "year": 2020, "media_type": "movie"}
    candidate = candidate_profiles([{"id": "a", "provider": "subdl", "provider_id": "a", "metadata": {**video, "source": "HDTV"}}])[0]
    result = evaluate(target_profile({"metadata": video}), candidate)
    assert result.source_priority_points == 0
    assert result.ranking_score == result.score_without_hash
    assert result.ranking_percentage == result.percentage
