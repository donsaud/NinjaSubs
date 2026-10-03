"""Release compatibility must outweigh the generic percentage.

A micro-rip rendition carries a high generic match score simply by sharing the
title, the year and a source token. Those tests are about *release
compatibility*, and the tier hierarchy already exists to express that; these
cases pin the precedence so a lower percentage can never outrank a materially
more compatible release.

None of this rejects anything. A micro-rip stays in the result set and is still
returned when nothing better exists -- that is asserted explicitly below.
"""

from __future__ import annotations

from app.models import MatchTier, SubtitleRelease
from app.services.subtitle_matcher import calculate_compatibility, is_anime_content, rank_subtitles

UHD_TARGET = "Whiplash.2014.2160p.UHD.BluRay.x254-SURCODE.mkv"
YIFY = "Whiplash.2014.1080p.BluRay.x264.YIFY"
YTS = "Whiplash.2014.1080p.BluRay.x264.YTS"
RARBG = "Whiplash.2014.1080p.BluRay.x264-RARBG"
UHD_ALT_GROUP = "Whiplash.2014.2160p.UHD.BluRay.x264-FraMeSToR"
BLURAY_1080 = "Whiplash.2014.1080p.BluRay.x264.anoXmous"


def _order(target: str, names: list[str]) -> list[str]:
    pool = [
        SubtitleRelease(release_name=n, download_url=f"http://cdn/{i}.srt", provider="subdl")
        for i, n in enumerate(names)
    ]
    return [s.release_name for s in rank_subtitles(target, pool, preferred_languages=["ara"])]


def _pct(target: str, name: str) -> int:
    return calculate_compatibility(target, name).percentage


# --------------------------------------------------------------------------- #
# 1. UHD target: a compatible release outranks a micro-rip                     #
# --------------------------------------------------------------------------- #


def test_uhd_target_prefers_compatible_bluray_over_microrip():
    """The reported case: a UHD BluRay target must not list a 1080p rip first.

    Every micro-rip here scores *higher* on the generic percentage than the
    compatible releases do, which is exactly the inversion being fixed.
    """
    order = _order(UHD_TARGET, [YIFY, YTS, RARBG, UHD_ALT_GROUP, BLURAY_1080])

    assert order.index(UHD_ALT_GROUP) < order.index(YIFY)
    assert order.index(BLURAY_1080) < order.index(YIFY)
    assert order.index(BLURAY_1080) < order.index(RARBG)
    # And the inversion is real, not vacuous: percentage alone would disagree.
    assert _pct(UHD_TARGET, YIFY) > _pct(UHD_TARGET, UHD_ALT_GROUP)


def test_microrip_is_demoted_one_tier_not_rejected():
    """A demotion, never a block: still accepted, still scored, just lower tier."""
    good = calculate_compatibility(UHD_TARGET, UHD_ALT_GROUP)
    rip = calculate_compatibility(UHD_TARGET, YIFY)

    assert rip.accepted is True, "a micro-rip must stay a selectable candidate"
    assert rip.match_tier in (MatchTier.CLOSE, MatchTier.FALLBACK)
    assert good.match_tier in (MatchTier.EXACT, MatchTier.SOURCE_FAMILY)
    assert int(good.match_tier.value) < int(rip.match_tier.value)


# --------------------------------------------------------------------------- #
# 2. WEB-DL target: compatible WEB outranks an unrelated BluRay rip             #
# --------------------------------------------------------------------------- #


def test_webdl_target_prefers_web_over_bluray_rip():
    target = "The.Mandalorian.S02E01.1080p.WEB-DL.x264-NTb.mkv"
    web = "The.Mandalorian.S02E01.1080p.WEB-DL.x264-NTb"
    bluray_rip = "The.Mandalorian.S02E01.1080p.BluRay.x264-RARBG"
    yify_rip = "The.Mandalorian.S02E01.1080p.BluRay.x264.YIFY"

    order = _order(target, [bluray_rip, yify_rip, web])

    assert order[0] == web
    assert order.index(web) < order.index(bluray_rip)
    assert order.index(web) < order.index(yify_rip)


def test_webdl_target_is_not_high_tier_so_no_microrip_demotion():
    """The demotion is scoped to UHD/REMUX targets; ordinary streaming is untouched."""
    target = "The.Mandalorian.S02E01.1080p.WEB-DL.x264-NTb.mkv"
    same_res_rip = "The.Mandalorian.S02E01.1080p.WEBRip.x264-YTS"

    # Same resolution as the target -> the resolution precondition is not met, so
    # a micro-rip keeps whatever tier its own evidence earns.
    assert int(calculate_compatibility(target, same_res_rip).match_tier.value) <= int(
        MatchTier.SOURCE_FAMILY.value
    )


# --------------------------------------------------------------------------- #
# 3. Exact group / edition beats generic source-token matches                   #
# --------------------------------------------------------------------------- #


def test_exact_group_outranks_generic_source_match():
    target = "Dune.2021.2160p.BluRay.x265-SPARKS.mkv"
    exact = "Dune.2021.2160p.BluRay.x265-SPARKS"
    rip = "Dune.2021.1080p.BluRay.x264.YIFY"

    order = _order(target, [rip, exact])

    assert order[0] == exact
    assert calculate_compatibility(target, exact).match_tier is MatchTier.EXACT
    assert _pct(target, exact) >= _pct(target, rip)


# --------------------------------------------------------------------------- #
# 4. YIFY remains selectable when nothing better exists                        #
# --------------------------------------------------------------------------- #


def test_microrip_remains_available_when_it_is_the_only_option():
    """Not a blacklist: with no better candidate the rip is still returned."""
    order = _order(UHD_TARGET, [YIFY])

    assert order == [YIFY]


def test_microrip_still_available_alongside_worse_candidates():
    """It also survives when it outranks the genuinely poor options."""
    unrelated = "Whiplash.2014.720p.HDTV.x264-RARBG"

    order = _order(UHD_TARGET, [YIFY, unrelated])

    assert YIFY in order
    assert order.index(YIFY) < order.index(unrelated)


# --------------------------------------------------------------------------- #
# 5. Root cause: a codec token must not make a film look like anime             #
# --------------------------------------------------------------------------- #


def test_codec_token_does_not_classify_a_feature_as_anime():
    """``x254`` is an encoder, not an episode number.

    Reading it as one routed ordinary features through the anime scoring path,
    where a release-group conflict marks the candidate an unshared fansub and
    forces the fallback tier -- which is what pushed a UHD BluRay release below
    a 1080p micro-rip in the first place.
    """
    assert is_anime_content(UHD_TARGET, UHD_ALT_GROUP) is False
    assert is_anime_content(UHD_TARGET, YIFY) is False


def test_genuine_anime_is_still_detected():
    """The guard must keep doing its job for real anime."""
    for video, sub in (
        ("[SubsPlease] Some Anime - 05 [1080p].mkv", "[SubsPlease] Some Anime - 05 [1080p].ass"),
        ("Naruto Shippuuden - 012 [720p].mkv", "Naruto Shippuuden - 012.ass"),
        ("One Piece - 1044 [1080p].mkv", "One Piece - 1044.ass"),
    ):
        assert is_anime_content(video, sub) is True, video


def test_ordinary_feature_stays_out_of_the_anime_path():
    for video, sub in (
        ("Dune.2021.2160p.BluRay.x265-SPARKS.mkv", "Dune.2021.2160p.BluRay.x265-SPARKS.srt"),
        ("Some.Movie.2015.1080p.BluRay.x264-GROUP.mkv", "Some.Movie.2015.1080p.BluRay.x264-GROUP.srt"),
        ("Arrival.2016.2160p.UHD.BluRay.x265-FraMeSToR.mkv", "Arrival.2016.1080p.BluRay.x264-NTb.srt"),
    ):
        assert is_anime_content(video, sub) is False, video
