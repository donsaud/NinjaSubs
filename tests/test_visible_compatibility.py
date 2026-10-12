"""Production display contract for weighted-release; sync status is separate."""

from itertools import permutations

from app.models import SubtitleRelease
from app.services.aggregator import format_informative_badge
from app.services.cache import build_cache_key
from app.services.release_scoring import CONTRACT_VERSION
from app.services.subtitle_matcher import rank_subtitles, refresh_visible_compatibility

TARGET = "Movie.2020.1080p.BluRay.x264-GROUP"


def release(name, **kwargs):
    return SubtitleRelease(release_name=name, download_url="/" + name, provider="subdl", **kwargs)


def test_display_badge_and_order_share_weighted_score():
    same = release("Movie.2020.1080p.BluRay.x264-OTHER")
    cross = release("Movie.2020.1080p.WEB-DL.x264-GROUP")
    ranked = rank_subtitles(TARGET, [cross, same])
    assert ranked == [same, cross]
    assert same.compatibility.score > cross.compatibility.score
    for item in ranked:
        assert item.match_percentage == round(item.compatibility.ranking_percentage)
        assert format_informative_badge(item, item.score).startswith(f"[{item.match_percentage}%]")


def test_sync_state_never_inflates_weighted_release_evidence():
    a = release("Movie.2020.1080p.BluRay.x264-GROUP")
    b = release("Movie.2020.1080p.WEB-DL.x264-OTHER")
    before = rank_subtitles(TARGET, [b, a])
    scores = [r.score for r in before]
    b.sync_state = "verified_synced"
    b.sync_verification = "cached"
    b.sync_evidence_trusted = True
    b.sync_target_binding = "strong"
    assert refresh_visible_compatibility(before, TARGET) == [a, b]
    assert [r.score for r in before] == scores


def test_display_identity_and_cut_conflicts_reject_even_when_discard_flag_false():
    wrong = release("Another.Movie.2020.1080p.BluRay.x264-GROUP")
    conflict = release("Movie.2020.Theatrical.Cut.BluRay-GROUP")
    assert rank_subtitles(TARGET, [wrong], discard_mismatches=False) == []
    assert rank_subtitles("Movie.2020.Directors.Cut.BluRay-GROUP", [conflict]) == []


def test_language_hi_and_deterministic_ties():
    ar = release("Movie.2020.1080p.BluRay.x264-A", lang="ara")
    en = release("Movie.2020.1080p.BluRay.x264-B", lang="eng")
    hi = release("Movie.2020.1080p.BluRay.x264-C", lang="ara", hearing_impaired=True)
    assert rank_subtitles(TARGET, [en, ar], preferred_languages=["ara"]) == [ar]
    assert rank_subtitles(TARGET, [ar, hi], hi_preference="prefer")[0] is hi
    assert rank_subtitles(TARGET, [hi, ar], hi_preference="exclude") == [ar]
    expected = [r.release_name for r in rank_subtitles(TARGET, [ar, en, hi])]
    for order in permutations([ar, en, hi]):
        assert [r.release_name for r in rank_subtitles(TARGET, list(order))] == expected


def test_matcher_contract_versions_display_cache():
    from app.services.visible_compatibility import VISIBLE_COMPATIBILITY_VERSION

    assert VISIBLE_COMPATIBILITY_VERSION == CONTRACT_VERSION
    assert build_cache_key("movie", "tt1234567", filename=TARGET)
