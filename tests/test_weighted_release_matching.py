"""Independent weighted semantics; metadata fixtures are not REAL timing proof."""
import ast
import copy
import json
import sys
from itertools import permutations
from pathlib import Path

import pytest

from app.services.release_scoring.engine import EPISODE_WEIGHTS, MOVIE_WEIGHTS, evaluate
from app.services.release_scoring.model import MatchState
from app.services.release_scoring.policies import (
    DisplayPolicy,
    ReferencePolicy,
    equivalent_peers,
    rank,
)
from app.services.release_scoring.profiles import (
    candidate_profiles,
    stable_identity,
    target_profile,
)

TV = {"media_type": "series", "title": "Show", "year": 2020, "season": 1, "episode": 1,
      "source": "BluRay", "release_group": "ALPHA", "resolution": "1080p",
      "video_codec": "H264", "audio_codec": "AC3", "service_disc_family": "BD"}
MOVIE = {"media_type": "movie", "title": "Movie", "year": 2020, "source": "BluRay",
         "release_group": "ALPHA", "resolution": "1080p", "video_codec": "H264",
         "audio_codec": "AC3", "service_disc_family": "BD", "cut": "theatrical"}


def record(identity="a", metadata=None, **kw):
    return {"id": identity, "provider": "subdl", "provider_id": identity,
            "metadata": dict(TV if metadata is None else metadata), "language": "ara", **kw}


def result(candidate=None, target=None):
    return evaluate(target_profile({"metadata": TV if target is None else target}),
                    candidate_profiles([record() if candidate is None else candidate])[0])


def test_shared_core_is_pure_and_not_the_obsolete_ranker():
    forbidden = {"logging", "httpx", "requests", "pydantic", "asyncio"}
    for path in Path("app/services/release_scoring").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                assert all(a.name.split(".")[0] in sys.stdlib_module_names - forbidden for a in node.names)
            if isinstance(node, ast.ImportFrom) and not node.level:
                assert node.module in {"app.services.release_parsing.normalize"} or node.module.split(".")[0] in sys.stdlib_module_names - forbidden


def test_numeric_weight_facts_and_exact_metadata():
    assert dict(EPISODE_WEIGHTS)["source"] == 25
    assert dict(EPISODE_WEIGHTS)["release_group"] == 20
    assert dict(MOVIE_WEIGHTS)["source"] == 30
    assert dict(MOVIE_WEIGHTS)["release_group"] == 15
    assert result().score == result().maximum == 359
    movie = result(record(metadata=MOVIE), MOVIE)
    assert movie.score == movie.maximum == 179


@pytest.mark.parametrize("source", ["BluRay", "WEB-DL", "HDTV", "DVDRip", "WEBRip"])
def test_source_mismatch_costs_more_than_group_mismatch(source):
    target = {**TV, "source": source}
    a = record("same_source", {**target, "release_group": "BETA"})
    b = record("same_group", {**target, "source": "HDTV" if source != "HDTV" else "BluRay"})
    results = tuple(evaluate(target_profile({"metadata": target}), c) for c in candidate_profiles([b, a]))
    assert [r.candidate.identity for r in rank(results, ReferencePolicy())] == ["same_source", "same_group"]
    assert results[0].score != results[1].score


@pytest.mark.parametrize("left,right", [("REMUX", "BluRay"), ("BluRay", "BDRip"), ("BDRip", "REMUX"),
                                        ("WEB-DL", "WEBRip"), ("WEBRip", "WEB-DL")])
def test_source_subtype_preserved_but_family_credit_is_weighted(left, right):
    r = result(record(metadata={**TV, "source": right}), {**TV, "source": left})
    assert r.state("source") is MatchState.MATCH
    assert next(e for e in r.evidence if e.field == "source").relationship == "source_family"
    assert r.candidate.fact("source").value != target_profile({"metadata": {**TV, "source": left}}).fact("source").value
    assert r.score == 359


@pytest.mark.parametrize("field,value", [("season", 2), ("episode", 2), ("media_type", "movie"), ("title", "Other")])
def test_content_contradictions_cannot_win_even_with_hash(field, value):
    hash_metadata = {"moviehash": "a" * 16, "video_size": 123456, "hash_verified": True}
    r = result(record(metadata={**TV, **hash_metadata, field: value}, provider="opensubtitles",
                      hash_provenance="opensubtitles_hash_query_literal_true"), {**TV, **hash_metadata})
    assert r.rejection_reasons
    assert not rank((r,), ReferencePolicy())
    assert not rank((r,), DisplayPolicy())


def test_verified_imdb_tolerates_aliases_but_never_wrong_episode():
    base = {**TV, "imdb_id": "tt1234567", "imdb_verified": True}
    r = result(record(metadata={**base, "title": "Localized", "year": 2021}), base)
    assert r.state("title") is MatchState.MISMATCH
    assert not r.rejection_reasons
    assert r.score == 359
    wrong = result(record(metadata={**base, "title": "Localized", "episode": 2}), base)
    assert "episode_mismatch" in wrong.rejection_reasons
    mismatch = result(record(metadata={**base, "imdb_id": "tt7654321"}), base)
    assert "imdb_id_mismatch" in mismatch.rejection_reasons


def test_remake_year_conflict_without_verified_identity():
    r = result(record(metadata={**MOVIE, "year": 1990}), MOVIE)
    assert "year_mismatch" in r.rejection_reasons


@pytest.mark.parametrize("field,value", [("release_group", None), ("source", None), ("video_codec", None)])
def test_unknown_is_not_mismatch(field, value):
    r = result(record(metadata={**TV, field: value}))
    assert r.state(field) is MatchState.UNKNOWN
    assert not r.rejection_reasons
    assert r.score < 359
    if field == "source":
        assert not r.ready_reference_eligible


def test_candidate_only_fields_do_not_reward_metadata_quantity():
    target = {**TV, "audio_codec": None, "resolution": None}
    sparse = result(record(metadata={**TV, "audio_codec": None, "resolution": None}), target)
    verbose = result(record(metadata={**TV, "audio_codec": "DTS", "resolution": "2160p"}), target)
    assert sparse.score == verbose.score == sparse.maximum == 357
    assert sparse.contributions == verbose.contributions


@pytest.mark.parametrize("codec", ["H264", "H.264", "x264", "AVC"])
def test_codec_normalization_is_explicit(codec):
    assert result(record(metadata={**TV, "video_codec": codec})).state("video_codec") is MatchState.MATCH


def test_different_group_is_no_gate_and_language_follows_score():
    english = result(record("en", {**TV, "release_group": "BETA"}, language="eng"))
    arabic = result(record("ar", {**TV, "source": "WEB-DL"}, language="ara"))
    assert [r.candidate.identity for r in rank((arabic, english), ReferencePolicy())] == ["en", "ar"]
    assert [r.candidate.identity for r in rank((arabic, english), DisplayPolicy(("ara",)))] == ["ar"]
    stronger_arabic = result(record("ar_strong", TV, language="ara"))
    assert rank((english, stronger_arabic), ReferencePolicy())[0] is stronger_arabic


def test_hi_preference_and_language_only_break_score_ties():
    hi = result(record("hi", hearing_impaired=True))
    normal = result(record("normal", hearing_impaired=False))
    assert [r.candidate.identity for r in rank((hi, normal), DisplayPolicy(exclude_hi=True))] == ["normal"]
    assert rank((normal, hi), DisplayPolicy(prefer_hi=True))[0] is hi
    assert hi.score == normal.score


def test_reference_discovery_and_ready_season_pack_lifecycle():
    raw = record("pack", {**TV, "episode": None, "scope": "season_pack"})
    pack = result(raw)
    assert rank((pack,), ReferencePolicy(discovery=True))
    assert not rank((pack,), ReferencePolicy())
    selected = {**raw, "archive_member": "Show.S01E01.srt", "archive_member_verified": True}
    ready = result(selected)
    assert ready.ready_reference_eligible
    assert ready.score == 359
    wrong = result({**raw, "archive_member": "Show.S01E02.srt", "archive_member_verified": True})
    assert "episode_mismatch" in wrong.rejection_reasons


def test_unvalidated_archive_member_never_becomes_ready():
    r = result(record(archive_member="Show.S01E01.srt", archive_member_verified=False))
    assert "archive_member_unresolved" in r.readiness_reasons


@pytest.mark.parametrize("raw", ["TRUE", "true", 1, False, None])
def test_truthy_hash_claims_never_establish_proof(raw):
    hm = {"moviehash": "a" * 16, "video_size": 1000, "hash_verified": raw}
    r = result(record(metadata={**TV, **hm}, provider="opensubtitles",
                      hash_provenance="opensubtitles_hash_query_literal_true"), {**TV, **hm})
    assert not r.verified_hash
    assert r.state("moviehash") is MatchState.UNKNOWN


def test_verified_hash_wins_and_unverified_mismatch_remains_unknown():
    hm = {"moviehash": "a" * 16, "video_size": 1000, "hash_verified": True}
    target = {**TV, **hm}
    exact = result(record("exact", {**TV, **hm, "release_group": "BETA"}, provider="opensubtitles",
                          hash_provenance="opensubtitles_hash_query_literal_true"), target)
    metadata = result(record("metadata"), target)
    assert exact.verified_hash and exact.score == 698 and exact.score_without_hash == 339
    assert rank((metadata, exact), ReferencePolicy())[0] is exact
    unknown = result(record(metadata={**TV, "moviehash": "b" * 16}), target)
    assert unknown.state("moviehash") is MatchState.UNKNOWN
    assert not unknown.rejection_reasons
    bad = result(record(metadata={**TV, **hm, "moviehash": "b" * 16}, provider="opensubtitles",
                        hash_provenance="opensubtitles_hash_query_literal_true"), target)
    assert "verified_moviehash_conflict" in bad.rejection_reasons


def test_source_conflict_is_preserved_with_provenance():
    r = result(record(release_names=["Show.S01E01.1080p.HDTV.x264-ALPHA"], metadata=TV))
    assert r.state("source") is MatchState.CONFLICT
    assert r.candidate.fact("source").conflict
    assert len(r.candidate.fact("source").provenance) >= 2
    assert not r.ready_reference_eligible


@pytest.mark.parametrize("variant", ["Criterion", "IMAX", "Remastered"])
def test_presentation_labels_never_become_hard_cut_conflicts(variant):
    r = result(record(metadata={**MOVIE, "edition": variant}), MOVIE)
    assert not r.rejection_reasons


def test_explicit_narrative_cut_conflict_rejects():
    r = result(record(metadata={**MOVIE, "cut": "directorscut"}), MOVIE)
    assert "cut_version_conflict" in r.rejection_reasons


def test_generic_descriptive_suffix_is_not_a_release_group():
    target = target_profile({"filename": "Show.2020.S01E01.1080p.BluRay.REMUX-FraMeSToR-AsRequested.mkv"})
    assert target.fact("release_group").value == "framestor"


def test_permutations_dedup_and_content_peers_preserve_evidence():
    rows = [record("a", provider="subdl", content_sha256="a" * 64),
            record("b", provider="subsource", content_sha256="a" * 64),
            record("alias", provider_id="a", metadata={**TV, "audio_codec": None})]
    before = copy.deepcopy(rows)
    expected = None
    for order in permutations(rows):
        profiles = candidate_profiles(list(order))
        results = tuple(evaluate(target_profile({"metadata": TV}), c) for c in profiles)
        report = (profiles, rank(results, ReferencePolicy()), equivalent_peers(results))
        if expected is None:
            expected = report
        assert report == expected
    assert rows == before
    assert len(expected[0]) == 2
    assert expected[0][0].content_peers


def test_correlation_id_is_not_semantic_evidence_or_stable_key():
    from dataclasses import replace
    candidate = candidate_profiles([record()])[0]
    assert stable_identity(candidate) == stable_identity(replace(candidate, identity="other", aliases=("other",)))


def test_technical_extensions_are_explainable_and_zero_weight():
    candidate = record(metadata={**TV, "hdr": "HDR10", "dolby_vision": "DV", "bit_depth": 10})
    r = result(candidate)
    assert r.candidate.fact("hdr").value == "hdr10"
    assert r.candidate.fact("dolby_vision").value == "dv"
    assert r.score == 359


@pytest.mark.parametrize("path", sorted(Path("tests/fixtures/weighted_release_matching").glob("*.json")))
def test_independent_golden_fixture(path):
    from tools.weighted_release_replay import run_case
    report = run_case(json.loads(path.read_text()))
    assert not report["differences"], report


def test_unknown_provider_ids_preserve_distinct_equivalent_release_records():
    rows = [record("a", provider_id="<missing-provider-id>",
                   release_names=["Show.S01E01.1080p.BluRay.H264-ALPHA"]),
            record("b", provider_id="<missing-provider-id>",
                   release_names=["Show.S01E01.Alternate.Label.1080p.BluRay.H264-ALPHA"])]
    left = candidate_profiles(rows)
    right = candidate_profiles(list(reversed(rows)))
    assert left == right and len(left) == 2
    assert stable_identity(left[0]) != stable_identity(left[1])
    results = tuple(evaluate(target_profile({"metadata": TV}), c) for c in left)
    assert results[0].score == results[1].score
    assert equivalent_peers(results)


def test_technical_aliases_preserve_hdr_plus_without_false_conflicts():
    r = result(record(metadata={**TV, "hdr": "HDR10+", "dolby_vision": "Dolby Vision", "bit_depth": "10-bit"},
                      release_names=["Show.S01E01.1080p.BluRay.HDR10+.DoVi.10bit.H264-ALPHA"]))
    assert r.candidate.fact("hdr").value == "hdr10plus"
    assert not r.candidate.fact("hdr").conflict
    assert r.candidate.fact("dolby_vision").value == "dv"
    assert not r.candidate.fact("dolby_vision").conflict
    assert r.candidate.fact("bit_depth").value == "10"
    assert r.score == 359
