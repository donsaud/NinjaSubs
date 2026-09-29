"""Regression and feature tests for auto-sync and reference resolution improvements."""

from unittest.mock import AsyncMock

import pytest

from app.models import SubtitleRelease
from app.services.sync.decode import select_zip_member
from app.services.sync.external_strategy import (
    TIER_EXACT_GROUP,
    TIER_FALLBACK,
    TIER_SOURCE_EDITION,
    ExternalExactStrategy,
    _select_candidates_ranked,
    reference_tier,
)
from app.services.sync.query import ReferenceQuery
from app.services.sync_service import _validate_synced_output


def test_select_zip_member_prefers_non_sdh_dialogue():
    """Verify that clean dialogue subtitles are preferred over larger SDH files."""
    members = [
        ("Mad.Men.S01E02.Ladies.Room.720p.WEB-DL.sdh.srt", 60000),  # larger, but SDH
        ("Mad.Men.S01E02.Ladies.Room.720p.WEB-DL.srt", 45000),      # clean dialogue
        ("Mad.Men.S01E02.Ladies.Room.720p.WEB-DL.cc.srt", 58000),   # CC
    ]
    chosen = select_zip_member(members, season=1, episode=2, min_bytes=5120)
    assert chosen == "Mad.Men.S01E02.Ladies.Room.720p.WEB-DL.srt"


def test_exact_release_group_beats_known_micro_rip_group():
    """An exact group token match must win regardless of group reputation."""
    target = "Movie.2024.1080p.BluRay.x264-RANDOM.mkv"

    # Same group as the target: same mastering pass, so cues line up exactly.
    random_rel = SubtitleRelease(
        release_name="Movie.2024.1080p.BluRay.x264-RANDOM.srt",
        download_url="http://cdn/random",
        provider="subdl",
        lang="eng",
    )
    # Retail scene release with no group in common with the target.
    chd_rel = SubtitleRelease(
        release_name="Movie.2024.1080p.BluRay.x264-CHD.srt",
        download_url="http://cdn/chd",
        provider="subdl",
        lang="eng",
    )

    assert reference_tier(target, random_rel) == TIER_EXACT_GROUP
    # CHD shares the BluRay medium but not the group, so it can only reach the
    # source/edition tier and can never be promoted into the exact-group tier.
    assert reference_tier(target, chd_rel) == TIER_SOURCE_EDITION
    assert reference_tier(target, random_rel) < reference_tier(target, chd_rel)

    ranked = _select_candidates_ranked(
        [chd_rel, random_rel], ReferenceQuery(imdb_id="tt1", target_filename=target)
    )
    assert ranked[0] is random_rel


def test_validate_synced_output_rejects_zero_collapse():
    """Verify that output where timestamps collapsed to 0 is rejected."""
    target = (
        "1\n00:01:00,000 --> 00:01:02,000\nHello\n\n"
        "2\n00:01:05,000 --> 00:01:07,000\nWorld\n\n"
        "3\n00:01:10,000 --> 00:01:12,000\nTest\n\n"
        "4\n00:01:15,000 --> 00:01:17,000\nFour\n\n"
        "5\n00:01:20,000 --> 00:01:22,000\nFive\n\n"
        "6\n00:01:25,000 --> 00:01:27,000\nSix\n"
    )
    # Collapsed output: all cues start at 00:00:00
    collapsed = (
        "1\n00:00:00,000 --> 00:00:01,000\nHello\n\n"
        "2\n00:00:00,000 --> 00:00:01,000\nWorld\n\n"
        "3\n00:00:00,000 --> 00:00:01,000\nTest\n\n"
        "4\n00:00:00,000 --> 00:00:01,000\nFour\n\n"
        "5\n00:00:00,000 --> 00:00:01,000\nFive\n\n"
        "6\n00:00:00,000 --> 00:00:01,000\nSix\n"
    )
    assert not _validate_synced_output(target, collapsed)


def test_validate_synced_output_accepts_valid_shifted_output():
    """Verify that a legitimate linear shift passes post-sync validation."""
    target = (
        "1\n00:01:00,000 --> 00:01:02,000\nHello\n\n"
        "2\n00:01:05,000 --> 00:01:07,000\nWorld\n\n"
        "3\n00:01:10,000 --> 00:01:12,000\nTest\n\n"
        "4\n00:01:15,000 --> 00:01:17,000\nFour\n\n"
        "5\n00:01:20,000 --> 00:01:22,000\nFive\n\n"
        "6\n00:01:25,000 --> 00:01:27,000\nSix\n"
    )
    synced = (
        "1\n00:01:04,500 --> 00:01:06,500\nHello\n\n"
        "2\n00:01:09,500 --> 00:01:11,500\nWorld\n\n"
        "3\n00:01:14,500 --> 00:01:16,500\nTest\n\n"
        "4\n00:01:19,500 --> 00:01:21,500\nFour\n\n"
        "5\n00:01:24,500 --> 00:01:26,500\nFive\n\n"
        "6\n00:01:29,500 --> 00:01:31,500\nSix\n"
    )
    assert _validate_synced_output(target, synced)


@pytest.mark.asyncio
async def test_external_strategy_passes_moviehash_to_opensubtitles():
    """Verify that video_hash and video_size are forwarded to OpenSubtitles provider."""
    mock_os = AsyncMock()
    mock_os.search_subtitles.return_value = []

    strategy = ExternalExactStrategy(opensubtitles_provider=mock_os)
    query = ReferenceQuery(
        imdb_id="tt0804503",
        target_filename="Mad.Men.S01E02.mkv",
        video_hash="0123456789abcdef",
        video_size=7415742622,
        season=1,
        episode=2,
    )

    await strategy._search_opensubtitles(query)

    assert mock_os.search_subtitles.called
    kwargs = mock_os.search_subtitles.call_args.kwargs
    assert kwargs.get("moviehash") == "0123456789abcdef"
    assert kwargs.get("moviebytesize") == 7415742622


@pytest.mark.asyncio
async def test_external_strategy_multi_candidate_fallback(tmp_path):
    """Verify that if candidate 1 fails download, candidate 2 is downloaded and used."""
    good_srt = ("1\n00:00:01,000 --> 00:00:02,000\n" + ("x" * 6000) + "\n").encode()

    class _FailFirstProvider:
        name = "failfirst"

        async def search_subtitles(self, **kwargs):
            return [
                SubtitleRelease(
                    release_name="Movie.2024.1080p.WEB-DL-GRP1.srt",
                    download_url="http://cdn/fail",
                    provider=self.name,
                    lang="eng",
                ),
                SubtitleRelease(
                    release_name="Movie.2024.1080p.WEB-DL-GRP2.srt",
                    download_url="http://cdn/good",
                    provider=self.name,
                    lang="eng",
                ),
            ]

        async def download_archive(self, url, api_key=None):
            if "fail" in url:
                return b""  # Empty payload, fails min_bytes check
            return good_srt

    provider = _FailFirstProvider()
    strategy = ExternalExactStrategy(subdl_provider=provider)

    query = ReferenceQuery(imdb_id="tt9999999", target_filename="Movie.2024.1080p.WEB-DL-GRP1.mkv")
    resolved = await strategy.resolve_with_provenance(query)

    assert resolved.text is not None
    assert resolved.candidate == "Movie.2024.1080p.WEB-DL-GRP2.srt"


@pytest.mark.asyncio
async def test_opensubtitles_quota_exhaustion_trips_breaker_and_falls_back(tmp_path):
    """Verify that when OpenSubtitles hits 406 (quota exhausted), subsequent OS candidates
    are skipped immediately and the resolver falls back to SubSource."""
    from app.providers.opensubtitles import OPENSUBTITLES_BREAKER
    from app.services.sync.cache import ReferenceDiskCache
    OPENSUBTITLES_BREAKER.reset()

    good_srt = ("1\n00:00:01,000 --> 00:00:02,000\n" + ("x" * 6000) + "\n").encode()

    class _MockOpenSubtitles:
        name = "opensubtitles"
        def __init__(self):
            self.download_calls = 0

        def is_breaker_open(self):
            return OPENSUBTITLES_BREAKER.is_open()

        async def search_subtitles(self, **kwargs):
            return [
                SubtitleRelease(
                    release_name="Dexter.S08E03.720p.BluRay.x264-PiR8.srt",
                    download_url="/sub/opensubtitles/1.srt",
                    provider=self.name,
                    lang="eng",
                ),
                SubtitleRelease(
                    release_name="Dexter.S08E03.1080p.BluRay.x265-ImE.srt",
                    download_url="/sub/opensubtitles/2.srt",
                    provider=self.name,
                    lang="eng",
                ),
            ]

        async def download_archive(self, url, api_key=None):
            self.download_calls += 1
            # Trip breaker as if HTTP 406 was encountered
            OPENSUBTITLES_BREAKER.trip(3600.0, reason="HTTP 406 Quota Exceeded")
            return None

    class _MockSubsource:
        name = "subsource"
        def __init__(self):
            self.download_calls = 0

        def is_breaker_open(self):
            return False

        async def search_subtitles(self, **kwargs):
            return [
                SubtitleRelease(
                    release_name="Dexter.S08E03.720p.BluRay.x264-DEMAND.srt",
                    download_url="https://subsource/download/3",
                    provider=self.name,
                    lang="eng",
                ),
            ]

        async def download_archive(self, url, api_key=None):
            self.download_calls += 1
            return good_srt

    os_provider = _MockOpenSubtitles()
    subsource_provider = _MockSubsource()

    strategy = ExternalExactStrategy(
        opensubtitles_provider=os_provider,
        subsource_provider=subsource_provider,
        cache=ReferenceDiskCache(root=tmp_path),
    )

    query = ReferenceQuery(
        imdb_id="tt0773262",
        target_filename="Dexter.s8e03.Whats.Eating.Dexter.Morgan.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv",
        season=8,
        episode=3,
        media_type="series",
    )

    resolved = await strategy.resolve_with_provenance(query)

    assert resolved.text is not None
    # SubSource candidate was used
    assert resolved.candidate == "Dexter.S08E03.720p.BluRay.x264-DEMAND.srt"
    # OpenSubtitles was called only once before tripping breaker, second OS candidate was skipped!
    assert os_provider.download_calls == 1
    assert subsource_provider.download_calls == 1

    OPENSUBTITLES_BREAKER.reset()


def test_candidates_ranked_tie_priority():
    """Verify that on equal scores, subsource/subdl are favored over rate-capped opensubtitles."""
    from app.providers.opensubtitles import OPENSUBTITLES_BREAKER
    OPENSUBTITLES_BREAKER.reset()

    target = "Movie.2024.1080p.BluRay.x264-RANDOM.mkv"
    os_rel = SubtitleRelease(
        release_name="Movie.2024.1080p.BluRay.x264-GRP.srt",
        download_url="http://os",
        provider="opensubtitles",
        lang="eng",
    )
    subsource_rel = SubtitleRelease(
        release_name="Movie.2024.1080p.BluRay.x264-GRP.srt",
        download_url="http://subsource",
        provider="subsource",
        lang="eng",
    )

    query = ReferenceQuery(imdb_id="tt123", target_filename=target)
    # Passed in order [os_rel, subsource_rel]
    ranked = _select_candidates_ranked([os_rel, subsource_rel], query)
    assert len(ranked) == 2
    # SubSource should be ranked first due to tie priority
    assert ranked[0].provider == "subsource"


def test_vtt_to_srt_supports_short_timestamps():
    """Verify that WebVTT with 2-part MM:SS.mmm timestamps converts to strict SRT."""
    from app.services.sync_service import _TIMESPAN_LINE_REGEX, _parse_timestamp_ms, _vtt_to_srt

    vtt = (
        "WEBVTT\n\n"
        "01:15.200 --> 01:18.500\n"
        "First dialogue cue\n\n"
        "02:00.000 --> 02:05.123\n"
        "Second dialogue cue\n"
    )
    srt = _vtt_to_srt(vtt)
    assert "00:01:15,200 --> 00:01:18,500" in srt
    assert "00:02:00,000 --> 00:02:05,123" in srt

    # Check that _TIMESPAN_LINE_REGEX and _parse_timestamp_ms handle short timestamps
    matches = list(_TIMESPAN_LINE_REGEX.finditer("01:15.200 --> 01:18.500"))
    assert len(matches) == 1
    assert _parse_timestamp_ms("01:15.200") == 75200


def test_timeline_rejection_allows_linear_offset():
    """Verify that a target with a 70s intro delay passes the duration gate."""
    from app.services.sync_service import _timeline_rejection

    # Target starts at 70s and ends at 2070s (span: 2000s)
    target = (
        "1\n00:01:10,000 --> 00:01:12,000\nHello\n\n"
        "2\n00:34:28,000 --> 00:34:30,000\nWorld\n"
    )
    # Ref starts at 2s and ends at 2002s (span: 2000s)
    ref = (
        "1\n00:00:02,000 --> 00:00:04,000\nHello\n\n"
        "2\n00:33:20,000 --> 00:33:22,000\nWorld\n"
    )
    # Raw diff is ~68s (which would exceed relaxed floor of 60s), but span diff is ~0s!
    rejection = _timeline_rejection(target, ref, decision_kind="edition", relaxed=True)
    assert rejection is None


def test_matching_movie_cut_outranks_mismatched_cut():
    """A different cut of the same medium shifts every cue, so it must not tie."""
    target = "The.Lord.of.the.Rings.The.Fellowship.of.the.Ring.2001.Extended.1080p.BluRay.x264-FSiHD.mkv"

    ext_rel = SubtitleRelease(
        release_name="The.Lord.of.the.Rings.2001.Extended.Cut.1080p.BluRay.x264.srt",
        download_url="http://ext",
        provider="subdl",
        lang="eng",
    )
    theatrical_rel = SubtitleRelease(
        release_name="The.Lord.of.the.Rings.2001.Theatrical.1080p.BluRay.x264.srt",
        download_url="http://theatrical",
        provider="subdl",
        lang="eng",
    )

    # Both share the BluRay medium; the extended cut is a token match, the
    # theatrical cut is a different edition and therefore demoted.
    assert reference_tier(target, ext_rel) == TIER_SOURCE_EDITION
    assert reference_tier(target, ext_rel) < reference_tier(target, theatrical_rel)

    ranked = _select_candidates_ranked(
        [theatrical_rel, ext_rel], ReferenceQuery(imdb_id="tt1", target_filename=target)
    )
    assert ranked[0] is ext_rel


def test_episodic_candidate_is_not_promoted_for_movie_target():
    """Episode markers in a candidate name must not earn a movie target any rank."""
    target = "Inception.2010.1080p.BluRay.x264-SPARKS.mkv"

    movie_rel = SubtitleRelease(
        release_name="Inception.2010.1080p.BluRay.x264.srt",
        download_url="http://movie",
        provider="subdl",
        lang="eng",
    )
    tv_rel = SubtitleRelease(
        release_name="Inception.S01E01.1080p.BluRay.x264.srt",
        download_url="http://tv",
        provider="subdl",
        lang="eng",
    )

    # Neither name shares the SPARKS group, and both share the BluRay medium, so
    # the tiers are equal: an episodic marker grants no advantage.
    assert reference_tier(target, movie_rel) == reference_tier(target, tv_rel)
    assert reference_tier(target, movie_rel) == TIER_SOURCE_EDITION


def test_merge_sync_meta_falls_back_to_subtitle_release_name():
    """Verify that debrid generic stream URLs fall back to subtitle's release_name."""
    from app.main import _merge_sync_meta

    meta = {
        "target_filename": "videoplayback.mp4",
        "stream_url": "https://debrid.example.com/d/xyz/videoplayback.mp4",
        "release_name": "Gladiator.2000.Extended.1080p.BluRay.x264-FSiHD.srt",
    }
    context = {}

    merged = _merge_sync_meta(meta, context)
    assert merged["target_filename"] == "Gladiator.2000.Extended.1080p.BluRay.x264-FSiHD.srt"


def test_reference_query_is_series_for_anime_and_tv():
    """Verify that ReferenceQuery.is_series detects anime, tv, and episode presence."""
    anime_q = ReferenceQuery(imdb_id="tt123", media_type="anime")
    assert anime_q.is_series is True

    tv_q = ReferenceQuery(imdb_id="tt123", media_type="tv")
    assert tv_q.is_series is True

    ep_q = ReferenceQuery(imdb_id="tt123", media_type="movie", episode=1)
    assert ep_q.is_series is True

    movie_q = ReferenceQuery(imdb_id="tt123", media_type="movie")
    assert movie_q.is_series is False


def test_exact_group_4k_retail_beats_microrip_with_matching_title():
    """A 4K retail release sharing the target group must beat a micro-rip."""
    target = "Whiplash.2014.MULTI.DV.2160p.WEB.H265-LOST.mkv"

    surcode_rel = SubtitleRelease(
        release_name="Whiplash.2014.2160p.UHD.BluRay.x254-SURCODE.srt",
        download_url="http://cdn/surcode",
        provider="subdl",
        lang="eng",
        hearing_impaired=False,
    )
    jyk_rel = SubtitleRelease(
        release_name="Whiplash 2014 1080p WEB-DL x264 AAC-JYK.srt",
        download_url="http://cdn/jyk",
        provider="subdl",
        lang="eng",
        hearing_impaired=False,
    )

    # Neither release shares the LOST group with the target. Source medium is the
    # deciding property because that is what keeps the cue grid aligned: the
    # WEB-DL micro-rip outranks the 4K BluRay retail release, and resolution is
    # never allowed to override that.
    assert reference_tier(target, surcode_rel) == TIER_FALLBACK
    assert reference_tier(target, jyk_rel) == TIER_SOURCE_EDITION
    assert reference_tier(target, jyk_rel) < reference_tier(target, surcode_rel)


def test_normalize_srt_blocks_drops_dummy_punctuation_and_inverted_timespans():
    """Verify that normalize_srt_blocks discards dummy punctuation cues (.., ---, ***) and inverted cues."""
    from app.services.sync_service import normalize_srt_blocks

    raw_srt = (
        "1\n"
        "00:00:00,300 --> 00:02:17,917\n"
        "..\n\n"
        "2\n"
        "00:02:31,250 --> 00:02:34,500\n"
        "Are you in one of the core bands?\n\n"
        "3\n"
        "00:05:00,000 --> 00:04:30,000\n"
        "Inverted timestamp should be dropped\n\n"
        "4\n"
        "00:05:10,000 --> 00:05:15,000\n"
        "---\n\n"
        "5\n"
        "00:05:20,000 --> 00:05:25,000\n"
        "No, not yet.\n\n"
        "6\n"
        "01:41:30,630 --> 01:59:47,505\n"
        "...\n"
    )

    cleaned = normalize_srt_blocks(raw_srt)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]

    # Only cues 2 and 5 should survive (re-indexed to 1 and 2)
    assert "Are you in one of the core bands?" in cleaned
    assert "No, not yet." in cleaned
    assert ".." not in lines
    assert "---" not in lines
    assert "..." not in lines
    assert "Inverted timestamp" not in cleaned
    assert cleaned.startswith("1\n00:02:31,250 --> 00:02:34,500\nAre you in one of the core bands?")



