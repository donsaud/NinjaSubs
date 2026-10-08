"""Calibration and safety contract for the single visible Compatibility Score."""

from dataclasses import replace
from unittest.mock import patch

import pytest

from app.models import SubtitleRelease
from app.services.aggregator import format_informative_badge
from app.services.cache import build_cache_key
from app.services.subtitle_matcher import (
    calculate_compatibility,
    rank_subtitles,
    refresh_visible_compatibility,
)
from app.services.sync.matching import parse_release_identity
from app.services.sync.target_binding import TargetBinding
from app.services.visible_compatibility import (
    BANDS,
    compatibility_score,
    presentation_key,
    score_names,
)

TARGET = "Movie.2020.Directors.Cut.2160p.BluRay.x265-GROUP"
SAME = "Movie.2020.Directors.Cut.1080p.BluRay.x264-GROUP"
GENERIC = "Movie.2020.1080p.BluRay-OTHER"
CONFLICT = "Movie.2020.Theatrical.Cut.2160p.BluRay.x265-GROUP"
BEAUTY = "The.Great.Beauty.2013.Criterion.1080p.BluRay.x265.10bit-SARTRE"
BEAUTY_NAMES = [
    "The.Great.Beauty.2013.Criterion.Collection.1080p.BluRay-SARTRE",
    "The.Great.Beauty.2013.Criterion.Collection.720p.BluRay",
    "The.Great.Beauty.2013.720p.BluRay-PublicHD",
    "The.Great.Beauty.2013.WEB-DL",
]


def release(name, **kwargs):
    return SubtitleRelease(
        release_name=name,
        download_url="https://example.invalid/" + name + str(kwargs.get("lang", "ara")),
        provider=kwargs.pop("provider", "subdl"),
        **kwargs,
    )


@pytest.mark.parametrize(
    "target,candidate,binding",
    [
        (TARGET, SAME, TargetBinding.STRONG),
        (TARGET, GENERIC, TargetBinding.UNKNOWN),
        (TARGET, CONFLICT, TargetBinding.CONFLICT),
        ("Movie.2020.2160p.BluRay.x265-GROUP", GENERIC, TargetBinding.WEAK),
        ("Movie.2160p.HEVC", "Movie.2160p.HEVC", TargetBinding.UNKNOWN),
        (TARGET, "Movie", TargetBinding.UNKNOWN),
        (
            "Movie.2020.Criterion.1080p.BluRay",
            "Movie.2020.Criterion.720p.BluRay",
            TargetBinding.STRONG,
        ),
        (
            "Movie.2020.Remastered.1080p.BluRay",
            "Movie.2020.Remastered.720p.BluRay",
            TargetBinding.STRONG,
        ),
        (
            "Movie.2020.IMAX.1080p.BluRay",
            "Movie.2020.Open.Matte.1080p.BluRay",
            TargetBinding.UNKNOWN,
        ),
    ],
)
def test_calibration_bands(target, candidate, binding):
    score = score_names(target, candidate)
    assert score.binding == binding
    assert BANDS[binding][0] <= score.percentage <= BANDS[binding][1]
    assert score.percentage < 100


def test_exact_identity_is_only_route_to_100():
    assert score_names(TARGET, SAME, exact_moviehash=True).percentage == 100
    for candidate in (TARGET, SAME, GENERIC):
        assert score_names(TARGET, candidate).percentage < 100


@pytest.mark.parametrize("bad", [1, "true", "True", "yes", None, False])
def test_hash_requires_literal_true_even_for_dictionary_inputs(bad):
    result = calculate_compatibility(TARGET, {"release_name": SAME, "is_hash_match": bad})
    assert result.percentage < 100
    assert not result.is_hash_match


def test_metadata_only_does_not_treat_missing_names_as_identical():
    target = parse_release_identity("Movie.2020.2160p.BluRay.x265-GROUP")
    candidate = parse_release_identity(GENERIC)
    assert compatibility_score(target, candidate).binding == TargetBinding.WEAK


def test_great_beauty_order_and_publichd_safety():
    ranked = rank_subtitles(BEAUTY, [release(n) for n in reversed(BEAUTY_NAMES)])
    assert [s.release_name for s in ranked] == BEAUTY_NAMES
    assert [s.match_percentage for s in ranked] == sorted(
        [s.match_percentage for s in ranked], reverse=True
    )
    assert ranked[2].compatibility.binding == "unknown"
    assert ranked[2].match_percentage <= 64
    assert ranked[0].match_percentage < 100


def test_whiplash_framestor_no_hash_remains_strong():
    score = score_names(
        "Whiplash.2014.2160p.UHD.BluRay.REMUX-FraMeSToR",
        "Whiplash.2014.1080p.BluRay.x264-FraMeSToR",
    )
    assert score.binding == TargetBinding.STRONG
    assert 85 <= score.percentage < 100


@pytest.mark.parametrize(
    "target",
    [TARGET, "Movie.Directors.Cut.2160p.BluRay.x265-GROUP", "Movie.Directors.Cut.2160p.HDR.HEVC"],
)
def test_cut_dominates_technical_and_group_matches(target):
    a = (
        target.replace("2160p", "1080p")
        .replace("x265", "x264")
        .replace("HEVC", "AVC")
        .replace("HDR.", "")
    )
    b = target.replace("Directors", "Theatrical")
    assert score_names(target, a).percentage > score_names(target, b).percentage
    assert score_names(target, b).binding == TargetBinding.CONFLICT


def test_missing_is_neutral_and_class_order_is_strict():
    strong = score_names(TARGET, SAME)
    unknown = score_names(TARGET, GENERIC)
    conflict = score_names(TARGET, CONFLICT)
    weak = score_names("Movie.2020.2160p.BluRay.x265-GROUP", GENERIC)
    exact = score_names(TARGET, SAME, exact_moviehash=True)
    assert (
        exact.percentage
        > strong.percentage
        > weak.percentage
        > unknown.percentage
        > conflict.percentage
    )


def test_technical_only_advantage_is_small():
    target = "Movie.2160p.BluRay.HEVC"
    a, b = score_names(target, target), score_names(target, "Movie.1080p.BluRay.AVC")
    assert a.percentage >= b.percentage
    assert a.percentage - b.percentage <= 2
    assert a.percentage < 100


@pytest.mark.parametrize(
    "state,verification,trusted,binding,boost",
    [
        (None, None, False, None, False),
        ("verified_synced", "verified", True, "strong", True),
        ("verified_resynced", "verified", True, "strong", True),
        ("verified_resynced", "cached", True, "exact", True),
        ("verified_resynced", "verified", False, "strong", False),
        ("verified_resynced", "verified", True, "unknown", False),
        ("unverified", "verified", True, "strong", False),
        ("probable_sync", "predicted", True, "strong", False),
        ("rejected", "cached", True, "strong", False),
        ("unverified", "unknown", True, "unknown", False),  # Large Offset delivery exception
    ],
)
def test_actual_sync_state_matrix(state, verification, trusted, binding, boost):
    evidence = dict(
        sync_state=state,
        sync_verification=verification,
        sync_evidence_trusted=trusted,
        sync_target_binding=binding,
    )
    before, after = score_names(TARGET, SAME), score_names(TARGET, SAME, evidence=evidence)
    assert (after.percentage > before.percentage) is boost
    assert after.binding == before.binding
    assert after.percentage < 100


def test_unknown_reference_cannot_promote_binding_even_with_verified_claim():
    score = score_names(
        TARGET,
        GENERIC,
        evidence=dict(
            sync_state="verified_synced",
            sync_verification="verified",
            sync_evidence_trusted=True,
            sync_target_binding="strong",
        ),
    )
    assert score.binding == TargetBinding.UNKNOWN and score.percentage <= 64


def test_preferences_and_provider_do_not_change_percentage():
    ar = release(SAME, lang="ara", hearing_impaired=False)
    en = release(SAME, lang="eng", hearing_impaired=True, provider="subsource")
    ranked = rank_subtitles(
        TARGET, [en, ar], preferred_languages=["ar", "en"], hi_preference="prefer"
    )
    assert ranked == [ar, en]
    assert ar.match_percentage == en.match_percentage
    assert rank_subtitles(TARGET, [en, ar], preferred_languages=["en", "ar"])[0] is en
    assert rank_subtitles(TARGET, [en, ar], hi_preference="exclude") == [ar]


def test_hi_preference_only_breaks_equal_scores():
    good, hi = release(SAME), release(GENERIC, hearing_impaired=True)
    assert rank_subtitles(TARGET, [hi, good], hi_preference="prefer")[0] is good


def test_refresh_reuses_metadata_and_does_not_run_parser_or_sync():
    ranked = rank_subtitles(TARGET, [release(SAME), release(GENERIC)])
    ranked[0].sync_state = "verified_resynced"
    ranked[0].sync_verification = "cached"
    ranked[0].sync_target_binding = "strong"
    ranked[0].sync_evidence_trusted = True
    with patch(
        "app.services.sync.matching.parse_release_identity",
        side_effect=AssertionError("extra parse"),
    ):
        refreshed = refresh_visible_compatibility(ranked, TARGET)
    assert refreshed[0].match_percentage == refreshed[0].compatibility.percentage == 99


def test_deterministic_ties_before_truncation():
    names = [GENERIC, CONFLICT, SAME]
    a = rank_subtitles(TARGET, [release(n) for n in names])
    b = rank_subtitles(TARGET, [release(n) for n in reversed(names)])
    assert [s.release_name for s in a] == [s.release_name for s in b]
    assert a[:1][0].release_name == SAME
    assert sorted(a, key=presentation_key) == a


@pytest.mark.parametrize(
    "number,expected",
    [(-4, 0), (101, 100), (0, 0), (100, 100), (float("nan"), 0), (float("inf"), 0)],
)
def test_existing_badge_bounds(number, expected):
    sub = release(SAME)
    sub.match_percentage = number
    assert format_informative_badge(sub, 0).startswith(f"[{expected}%]")
    assert presentation_key(sub)[0] == -expected


def test_badge_does_not_independently_infer_hash_score():
    sub = release(SAME, is_hash_match=True)
    sub.match_percentage = 93
    assert format_informative_badge(sub, 0).startswith("[93%] [SubDL]")


def test_aggregation_cache_is_visible_semantic_versioned():
    with patch("app.services.visible_compatibility.VISIBLE_COMPATIBILITY_VERSION", "old"):
        before = build_cache_key("movie", "tt0000000")
    assert before != build_cache_key("movie", "tt0000000")


def test_group_and_cut_positive_monotonicity():
    generic = score_names(TARGET, GENERIC)
    group = score_names(TARGET, GENERIC.replace("OTHER", "GROUP"))
    cut = score_names(TARGET, SAME)
    assert cut.percentage >= group.percentage >= generic.percentage
    assert score_names(TARGET, CONFLICT).percentage < generic.percentage


def test_whole_title_final_cut_is_not_an_edition():
    assert not parse_release_identity("Final.Cut.1080p.BluRay-GROUP").cut_tags


def test_low_signal_fields_cannot_cross_binding_band():
    target = parse_release_identity(TARGET)
    unknown = replace(
        parse_release_identity(GENERIC),
        resolution=target.resolution,
        codec=target.codec,
        hdr_flags=target.hdr_flags,
        bit_depth=10,
    )
    score = compatibility_score(target, unknown)
    assert score.binding == TargetBinding.UNKNOWN and score.percentage <= 64


def test_explicit_cut_filter_is_preserved_and_retained_conflict_is_low():
    assert rank_subtitles(TARGET, [release(CONFLICT)]) == []
    retained = rank_subtitles(TARGET, [release(CONFLICT)], discard_mismatches=False)
    assert retained[0].compatibility.binding == "conflict"
    assert retained[0].match_percentage == 10


def test_guessed_year_cannot_be_an_explicit_season():
    metadata = parse_release_identity("Hunter X Hunter.2011.E2.srt")
    assert metadata.year == 2011 and metadata.episode == 2
    assert metadata.season is None


def test_whitespace_and_underscore_group_suffixes_are_preserved():
    for name in ("Movie 2020 1080p BluRay GROUP", "Movie.2020.1080p.BluRay-Pi_R8"):
        metadata = parse_release_identity(name)
        assert metadata.release_group in {"GROUP", "Pi_R8"}


def test_multi_episode_pack_never_claims_single_episode_binding():
    target = "Show.S01E02.1080p.WEB-DL-GROUP"
    pack = release("Show.S01E01-E03.1080p.WEB-DL-GROUP")
    ranked = rank_subtitles(target, [pack])
    assert len(ranked) == 1
    assert ranked[0].compatibility.binding == "unknown"
    assert ranked[0].match_percentage <= 64


def test_order_key_and_badge_share_rounding():
    sub = release(SAME)
    sub.match_percentage = 93.8
    assert presentation_key(sub)[0] == -94
    assert format_informative_badge(sub, 0).startswith("[94%]")
