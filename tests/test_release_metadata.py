"""Local release knowledge regressions; these fixtures are not REAL evidence."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from app.services.sync import matching, release_metadata
from app.services.sync.external_strategy import _select_candidates_ranked
from app.services.sync.matching import parse_release_identity as parse
from app.services.sync.query import ReferenceQuery
from app.services.sync.release_knowledge import Comparison, MetadataState, compare_field
from app.services.sync.target_binding import TargetBinding, classify_target_binding


@pytest.mark.parametrize("title", ["Final Cut", "Uncut", "Special", "Extended",
                                  "The Criterion Collection", "Restoration", "Open Matte",
                                  "The Directors Cut", "International Version", "IMAX",
                                  "The BluRay Final Cut"])
def test_title_words_are_not_edition_evidence(title):
    metadata = parse(f"{title}.2020.1080p.BluRay.x264-GRP")
    assert metadata.title == title.lower()
    assert metadata.cut_tags == metadata.edition_tags == ()


@pytest.mark.parametrize("episode_title", ["Final.Cut", "Restoration"])
def test_recognized_episode_title_is_not_edition_evidence(episode_title):
    metadata = parse(f"Show.S01E01.{episode_title}.1080p.BluRay.x264-GRP")
    assert metadata.title == "show"
    assert metadata.cut_tags == metadata.edition_tags == ()


@pytest.mark.parametrize("suffix, field, expected", [
    ("Director’s Cut", "cut_tags", ("directors_cut",)),
    ("Alternate.Version", "cut_tags", ("alternate_version",)),
    ("Alternate-Cut", "cut_tags", ("alternate_cut",)),
    ("Criterion.Collection", "label_tags", ("criterion_collection",)),
    ("Arrow.Video", "label_tags", ("arrow",)),
    ("BFI", "label_tags", ("bfi",)),
    ("Shout.Factory", "label_tags", ("shout_factory",)),
    ("Open.Matte", "presentation_tags", ("open_matte",)),
    ("Fullscreen", "presentation_tags", ("fullscreen",)),
    ("Widescreen", "presentation_tags", ("widescreen",)),
    ("Remastered", "master_tags", ("remastered",)),
    ("Restoration", "master_tags", ("restored",)),
    ("Collector’s.Edition", "packaging_tags", ("collectors_edition",)),
    ("25th.Anniversary.Edition", "packaging_tags", ("anniversary_edition",)),
    ("Special.Edition", "packaging_tags", ("special_edition",)),
    ("AMZN", "service", "amazon"), ("Amazon", "service", "amazon"),
    ("NF", "service", "netflix"), ("Netflix", "service", "netflix"),
    ("DSNP", "service", "disney_plus"), ("Disney+", "service", "disney_plus"),
    ("ATVP", "service", "apple_tv_plus"), ("AppleTV+", "service", "apple_tv_plus"),
    ("HMAX", "service", "hbo_max"), ("HBO.Max", "service", "hbo_max"),
    ("DTS.TrueHD.Atmos", "audio_tags", ("atmos", "dts", "truehd")),
])
def test_categories_and_aliases(suffix, field, expected):
    metadata = parse(f"Film.2020.{suffix}.1080p.BluRay.x264-GRP")
    assert getattr(metadata, field) == expected
    assert metadata.state(field) == MetadataState.KNOWN
    assert metadata.evidence
    assert all(item.source in {"knowledge", "filename", "guessit"} for item in metadata.evidence)


def test_unknown_ambiguous_and_conflicting_evidence():
    generic = parse("Film.2020.1080p.BluRay.x264-GRP")
    criterion = parse("Film.2020.Criterion.1080p.BluRay.x264-GRP")
    arrow = parse("Film.2020.Arrow.1080p.BluRay.x264-GRP")
    assert generic.state("label_tags") == MetadataState.UNKNOWN
    assert compare_field(generic, criterion, "label_tags") == Comparison.UNKNOWN
    assert compare_field(criterion, criterion, "label_tags") == Comparison.MATCH
    assert compare_field(criterion, arrow, "label_tags") == Comparison.CONFLICT
    ambiguous = parse("Film.2020.NF.AMZN.1080p.WEB-DL.x264-GRP")
    assert ambiguous.state("service") == MetadataState.AMBIGUOUS
    assert compare_field(ambiguous, generic, "service") == Comparison.UNKNOWN
    cuts = parse("Film.2020.Directors.Cut.Theatrical.1080p.BluRay.x264-GRP")
    assert cuts.state("cut_tags") == MetadataState.AMBIGUOUS
    assert classify_target_binding(None, None, target_metadata=cuts,
                                   reference_metadata=cuts).state == TargetBinding.UNKNOWN


def test_unicode_and_episode_projection():
    metadata = parse("Ｆｉｌｍ＿Ｓ０２Ｅ０３＿Criterion＿1080p＿BluRay＿x264-GRP")
    assert (metadata.title, metadata.season, metadata.episode) == ("film", 2, 3)
    assert metadata.label_tags == ("criterion_collection",)
    assert parse("Show.2x03.720p.HDTV.x264-GRP").episode == 3
    assert parse("Show.Season.2.1080p.BluRay.x264-GRP").episode is None
    assert parse("Show.S00E01.1080p.BluRay.x264-GRP").state("season") == MetadataState.KNOWN


@pytest.mark.parametrize("name, title, year", [
    ("1917.2019.1080p.BluRay.x264-GRP", "1917", 2019),
    ("2001.A.Space.Odyssey.1968.1080p.BluRay.x264-GRP", "2001 a space odyssey", 1968),
    ("1984.Uncut.1080p.BluRay.x264-GRP", "1984 uncut", None),
])
def test_numeric_titles_do_not_create_year_or_cut_evidence(name, title, year):
    metadata = parse(name)
    assert metadata.title == title and metadata.year == year
    assert metadata.cut_tags == ()


def test_master_duration_and_fps_are_separate():
    metadata = parse("Film.1990.2020.Restoration.1080p.BluRay.x264-GRP [1h30m00s] 23.976fps")
    assert metadata.year == 1990 and metadata.restoration_year == 2020
    assert metadata.duration_hint_seconds == 5400 and metadata.fps == 23.976
    assert metadata.cut_tags == ()


def test_provider_hints_are_whitelisted_and_conflicts_are_ambiguous():
    metadata = parse("Film.2020.1080p.BluRay.x264-GRP", hints={
        "year": 2021, "episode": 2, "source_family": "bluray", "api_key": "never retain",
    })
    assert metadata.year == 2020 and metadata.state("year") == MetadataState.AMBIGUOUS
    assert metadata.episode == 2
    assert "never retain" not in repr(metadata)
    assert classify_target_binding(None, None, target_metadata=metadata,
                                   reference_metadata=metadata).state == TargetBinding.UNKNOWN
    assert parse("Film.2020.1080p.BluRay.x264-GRP").state("year") == MetadataState.KNOWN


@pytest.mark.parametrize("name", [None, "", "()", "a" * 2049, 123, "\x00\x01"])
def test_malformed_names_fail_safely(name):
    assert not parse(name).cut_tags


def test_cache_is_immutable_bounded_and_semantically_versioned(monkeypatch):
    release_metadata._parse_cached.cache_clear()
    name = "Film.2020.Criterion.1080p.BluRay.x264-GRP"
    a = parse(name)
    assert parse(name) is a
    with pytest.raises(FrozenInstanceError):
        a.year = 2021
    monkeypatch.setattr(release_metadata, "SEMANTIC_VERSION", "test-new-semantics")
    assert parse(name) is not a
    for i in range(2050):
        parse(f"Film{i}.2020.1080p.BluRay.x264-GRP")
    assert release_metadata._parse_cached.cache_info().currsize == 2048


def test_fast_path_and_cached_guessit_fallback(monkeypatch):
    calls = []
    def guess(name):
        calls.append(name)
        return {"title": "Ambiguous Film", "year": 2020}
    release_metadata._parse_cached.cache_clear()
    monkeypatch.setattr(matching, "guess_metadata", guess)
    parse("Film.2020.Criterion.1080p.BluRay.x264-GRP")
    assert not calls
    assert parse("Ambiguous Film").title == "ambiguous film"
    parse("Ambiguous Film")
    assert calls == ["Ambiguous Film"]




def test_semantic_rank_precedes_limit_and_preserves_raw_release():
    query = ReferenceQuery(imdb_id="tt2358891", target_filename="The.Great.Beauty.2013.Criterion.1080p.BluRay.x265-SARTRE")
    names = [f"The.Great.Beauty.2013.1080p.BluRay.x265-PublicHD{i}" for i in range(30)]
    best = "The.Great.Beauty.2013.Criterion.Collection.720p.BluRay.x264-OTHER"
    names.append(best)
    releases = [SimpleNamespace(release_name=name, provider="subdl", lang="en") for name in names]
    ranked = _select_candidates_ranked(releases, query)
    assert len(ranked) == 31
    assert ranked[0].release_name != best  # Collection labels cannot override weighted comparable technical points.
    assert ranked.index(next(r for r in ranked if r.release_name == best)) == 30
    assert [release.release_name for release in releases] == names
    assert classify_target_binding(query.target_filename, names[0]).state == TargetBinding.UNKNOWN
    assert classify_target_binding(query.target_filename, best).state == TargetBinding.STRONG


def test_disc_group_identity_survives_different_encodes():
    target = "Whiplash.2014.1080p.BluRay.REMUX-FraMeSToR"
    reference = "Whiplash.2014.2160p.UHD.BluRay.x265.10bit-FraMeSToR"
    assert classify_target_binding(target, reference).state == TargetBinding.STRONG


def test_release_group_is_not_a_cut_or_collection_tag():
    metadata = parse("Film.2020.1080p.BluRay.x264-Criterion")
    assert metadata.release_group == "Criterion"
    assert metadata.label_tags == ()
    assert parse("Film.2020.1080p.WEB-DL.srt").release_group is None
    web = parse("Movie.2020.1080p.WEB-DL-TESTGROUP")
    assert (web.source_family, web.release_group) == ("webdl", "TESTGROUP")
    assert parse("Movie.2020.1080p.BluRay.x264-D-Z0N").release_group == "D-Z0N"
    assert classify_target_binding("Movie.2020.1080p.BluRay.x264-D-Z0N",
                                   "Movie.2020.1080p.BluRay.x264-E-Z0N").state == TargetBinding.WEAK


def test_no_new_movie_or_language_specific_rules():
    from app.services.sync.release_knowledge import RULES
    assert not any("beauty" in rule["pattern"] or "whiplash" in rule["pattern"] for rule, _ in RULES)
