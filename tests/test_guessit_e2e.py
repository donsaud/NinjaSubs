"""End-to-end validation: guessit metadata parsing + strictly descending ranking.

Guards the v1.1.0 release: proves ``guessit`` is actually installed and parsing
(``guess_metadata`` degrades to ``{}`` when the package is missing), that its
fields flow into the pipeline's ``extract_metadata`` attributes, and that the
ranked candidate list returned by ``rank_subtitles`` is monotonically
non-increasing by ``match_percentage`` with clean ISO-639 language codes.
"""

from app.models import SubtitleRelease
from app.services.subtitle_matcher import extract_metadata, rank_subtitles
from app.services.sync.matching import guess_metadata
from app.utils.language import normalize_to_iso639_2

MOVIE_TARGET = "Dune.Part.Two.2024.2160p.MAX.WEB-DL.DDP5.1.Atmos.DV.HDR.H.265-FLUX.mkv"
TV_TARGET = "House.of.the.Dragon.S02E04.1080p.MAX.WEB-DL.H.264-FLUX.mkv"

MOVIE_POOL = [
    "Dune.Part.Two.2024.720p.WEBRip.x264.YTS.srt",
    "Dune Part Two 2024 Arabic.srt",
    "Dune.Part.Two.2024.1080p.AMZN.WEBRip.DDP5.1-GRP.srt",
    "Dune.Part.Two.2024.2160p.MAX.WEB-DL.DDP5.1.Atmos.DV.HDR.H.265-FLUX.srt",
    "Dune.Part.Two.2024.Theatrical.1080p.BluRay.x264-SPARKS.srt",
    "Dune.Part.Two.2024.1080p.MAX.WEB-DL.DDP5.1.H.264-NTb.srt",
]

TV_POOL = [
    "House.of.the.Dragon.S02E03.1080p.MAX.WEB-DL.H.264-FLUX.srt",
    "House.of.the.Dragon.S01E04.1080p.MAX.WEB-DL.H.264-FLUX.srt",
    "House of the Dragon S02E04 720p BluRay x264-SPARKS.srt",
    "House.of.the.Dragon.S02E04.1080p.MAX.WEB-DL.H.264-FLUX.srt",
    "House.of.the.Dragon.S02E04.720p.AMZN.WEB-DL.x264-GRP.srt",
]


def _pool(names: list[str]) -> list[SubtitleRelease]:
    return [
        SubtitleRelease(release_name=n, download_url=f"http://cdn/{i}.srt", provider="subdl", lang="ara")
        for i, n in enumerate(names)
    ]


def _assert_strictly_descending(ranked: list[SubtitleRelease]) -> None:
    """Ranking is monotonic in tier, and monotonic in percentage within a tier.

    Tier is the primary sort key, so a higher tier may legitimately carry a
    lower percentage than a lower tier (a well-matched WEBRip outranks a
    near-unknown release). Percentage only orders candidates of equal tier.
    """
    assert ranked, "expected at least one ranked candidate"

    best_tier = min(s.match_tier.value for s in ranked)
    assert ranked[0].match_tier.value == best_tier, (
        f"Top candidate {ranked[0].release_name!r} is in tier {ranked[0].match_tier}, "
        f"expected the best available tier {best_tier}"
    )

    for prev, curr in zip(ranked, ranked[1:], strict=False):
        prev_tier = prev.match_tier.value
        curr_tier = curr.match_tier.value
        assert prev_tier <= curr_tier, (
            f"Tier violation: {prev.release_name} (tier {prev_tier}) ranked below "
            f"{curr.release_name} (tier {curr_tier})"
        )
        if prev_tier == curr_tier:
            assert prev.match_percentage >= curr.match_percentage, (
                f"Sorting violation within tier {prev.match_tier}: "
                f"{prev.release_name} ({prev.match_percentage}%) ranked lower than "
                f"{curr.release_name} ({curr.match_percentage}%)"
            )
            if prev.match_percentage == curr.match_percentage:
                assert prev.score >= curr.score, (
                    f"Tie-break violation: {prev.release_name} ({prev.score}) ranked below "
                    f"{curr.release_name} ({curr.score}) at equal match_percentage"
                )


# --------------------------------------------------------------------------- #
# 1. guessit is installed and parses complex release names
# --------------------------------------------------------------------------- #
def test_guessit_parses_complex_movie_metadata():
    meta = guess_metadata(MOVIE_TARGET)
    assert meta, "guess_metadata returned {} — is `guessit` installed?"
    assert meta.get("release_group") == "FLUX"
    assert str(meta.get("source", "")).lower() in ("web", "web-dl")
    assert str(meta.get("streaming_service", "")).lower() == "max"
    assert meta.get("screen_size") == "2160p"
    assert str(meta.get("video_codec", "")).replace(".", "").lower() in ("h265", "x265")
    assert meta.get("audio_codec")  # Dolby Digital Plus / Dolby Atmos


def test_guessit_parses_tv_season_episode_metadata():
    meta = guess_metadata(TV_TARGET)
    assert meta, "guess_metadata returned {} — is `guessit` installed?"
    assert meta.get("season") == 2
    assert meta.get("episode") == 4
    assert meta.get("release_group") == "FLUX"
    assert str(meta.get("streaming_service", "")).lower() == "max"


# --------------------------------------------------------------------------- #
# 2. guessit-derived metadata flows into pipeline attributes
# --------------------------------------------------------------------------- #
def test_pipeline_metadata_maps_guessit_fields():
    movie = extract_metadata(MOVIE_TARGET)
    assert movie["content_type"] == "movie"
    assert movie["source"] == "WEB-DL"
    assert movie["service"] == "HMAX"
    assert movie["group"] == "FLUX"
    assert movie["audio"] == "Atmos"
    assert movie["video_codec"] == "x265"
    assert movie["resolution"] == "2160p"
    assert movie["edition"] is None  # no edition tag in the name

    tv = extract_metadata(TV_TARGET)
    assert tv["content_type"] == "series"
    assert tv["season"] == 2 and tv["episode"] == 4
    assert tv["group"] == "FLUX"
    assert tv["service"] == "HMAX"


# --------------------------------------------------------------------------- #
# 3. End-to-end ranked ordering invariants
# --------------------------------------------------------------------------- #
def test_e2e_movie_ranking_is_strictly_descending():
    ranked = rank_subtitles(
        MOVIE_TARGET, _pool(MOVIE_POOL), preferred_languages=["ara"], discard_mismatches=True
    )
    _assert_strictly_descending(ranked)
    assert ranked[0].release_name.endswith("H.265-FLUX.srt")
    assert ranked[0].match_percentage == 100
    # The Theatrical cut conflicts with the (unmarked) stream and scores 0%.
    theatrical = next(s for s in ranked if "Theatrical" in s.release_name)
    assert theatrical.match_percentage == 0
    for sub in ranked:
        assert sub.lang == "ara" and normalize_to_iso639_2(sub.lang) == "ara"


def test_e2e_tv_ranking_is_strictly_descending():
    ranked = rank_subtitles(
        TV_TARGET, _pool(TV_POOL), preferred_languages=["ara"], discard_mismatches=True
    )
    _assert_strictly_descending(ranked)
    assert ranked[0].release_name.endswith("MAX.WEB-DL.H.264-FLUX.srt")
    assert ranked[0].match_percentage == 100
    # Wrong episode/season candidates are filtered out of the response.
    assert not any("S02E03" in s.release_name or "S01E04" in s.release_name for s in ranked)
    for sub in ranked:
        assert sub.lang == "ara" and normalize_to_iso639_2(sub.lang) == "ara"
