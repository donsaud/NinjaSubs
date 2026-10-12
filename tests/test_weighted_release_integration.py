"""Runtime adaptation regressions: evidence, privacy, and authoritative policies."""

import asyncio
import hashlib
from itertools import permutations

import pytest

from app.models import SubtitleRelease
from app.services import release_scoring_integration as integration
from app.services.sync.external_strategy import _select_candidates_ranked
from app.services.sync.query import ReferenceQuery


def subtitle(name, language="ara", provider="opensubtitles"):
    return SubtitleRelease(
        release_name=name,
        download_url=f"/sub/opensubtitles/{int(hashlib.sha256(name.encode()).hexdigest()[:8],16)}.srt",
        provider=provider,
        lang=language,
    )


def test_episode_imdb_is_not_compared_with_series_imdb_and_wrong_episode_rejects():
    q = ReferenceQuery(
        imdb_id="tt2661044",
        media_type="series",
        season=1,
        episode=1,
        target_filename="The.100.S01E01.1080p.BluRay.x264-GROUP",
    )
    good = subtitle("The.100.S01E01.1080p.BluRay.x264-OTHER")
    integration.provider_evidence(
        good,
        {
            "feature_details": {
                "feature_type": "Episode",
                "imdb_id": 2912494,
                "year": 2014,
                "season_number": 1,
                "episode_number": 1,
            }
        },
    )
    assert not any(
        "imdb_id_mismatch" in r.rejection_reasons for r in integration.matches([good], q)
    )
    bad = subtitle("The.100.S01E02.1080p.BluRay.x264-OTHER")
    integration.provider_evidence(
        bad,
        {
            "feature_details": {
                "feature_type": "Episode",
                "parent_imdb_id": 2661044,
                "season_number": 1,
                "episode_number": 2,
            }
        },
    )
    assert _select_candidates_ranked([bad, good], q) == [good]


def test_explicit_wrong_parent_imdb_and_movie_imdb_still_reject():
    q = ReferenceQuery(
        imdb_id="tt2661044",
        media_type="series",
        season=1,
        episode=1,
        target_filename="The.100.S01E01.BluRay-GROUP",
    )
    bad = subtitle("The.100.S01E01.BluRay-GROUP")
    integration.provider_evidence(
        bad,
        {
            "feature_details": {
                "feature_type": "Episode",
                "parent_imdb_id": 9999999,
                "season_number": 1,
                "episode_number": 1,
            }
        },
    )
    assert _select_candidates_ranked([bad], q) == []
    movie = subtitle("Movie.2020.BluRay-GROUP")
    integration.provider_evidence(
        movie, {"feature_details": {"feature_type": "Movie", "imdb_id": 9999999}}
    )
    assert (
        integration.reference_order(
            [movie], {"imdb_id": "tt1234567", "filename": "Movie.2020.BluRay-GROUP"}
        )
        == []
    )


def test_bilingual_references_independent_of_arabic_display_and_selected_subtitle():
    video = "Show.S01E02.1080p.BluRay.x264-GROUP"
    english = subtitle("Show.S01E02.1080p.BluRay.x264-OTHER", "eng", "subdl")
    arabic = subtitle("Show.S01E02.1080p.WEB-DL.x264-GROUP", "ara", "subsource")
    q = ReferenceQuery(
        imdb_id="tt1234567",
        target_filename=video,
        media_type="series",
        season=1,
        episode=2,
        languages=("ara",),
        target_sub_release_name=arabic.release_name,
    )
    assert _select_candidates_ranked([arabic, english], q) == [english, arabic]
    assert integration.display_order(video, [english, arabic], preferred_languages=["ara"]) == [
        arabic
    ]


def test_duplicate_records_merge_metadata_without_merging_different_release_records():
    one = subtitle("Show.S01E02.BluRay-A")
    two = subtitle("Show.S01E02.BluRay-A")
    integration.provider_evidence(one, {"resolution": "1080p"})
    integration.provider_evidence(two, {"video_codec": "x264"})
    results = integration.matches(
        [one, two], {"filename": "Show.S01E02.1080p.BluRay.x264-A", "media_type": "series"}
    )
    assert len(results) == 1
    assert results[0].candidate.fact("resolution").value == "1080p"
    assert results[0].candidate.fact("video_codec").value == "h264"
    other = subtitle("Show.S01E02.BluRay-B", provider="subsource")
    assert (
        len(
            integration.matches(
                [one, other], {"filename": "Show.S01E02.BluRay-A", "media_type": "series"}
            )
        )
        == 2
    )


def test_no_switch_required_and_input_order_never_changes_reference_order():
    q = ReferenceQuery(
        imdb_id="tt1234567",
        target_filename="Dexter.S08E01.1080p.BluRay.H264.TrueHD-PiR8",
        media_type="series",
        season=8,
        episode=1,
    )
    candidates = [
        subtitle("Dexter.S08E01.1080p.BluRay.H264.TrueHD-FraMeSToR", "eng", "subdl"),
        subtitle("Dexter.S08E01.1080p.BluRay.x264-ROVERS", "ara", "subsource"),
        subtitle("Dexter.S08E01.1080p.WEB-DL.x264-PiR8", "eng", "subdl"),
    ]
    expected = _select_candidates_ranked(candidates, q)
    assert [r.release_name for r in expected] == [c.release_name for c in candidates]
    for order in permutations(candidates):
        assert _select_candidates_ranked(list(order), q) == expected


@pytest.mark.asyncio
async def test_extraction_context_isolated_and_restored_after_exception():
    async def one(name):
        with integration.extracting_member() as selected:
            await asyncio.sleep(0)
            integration.member_selected(name)
            await asyncio.sleep(0)
            assert selected == [name]
        return selected

    assert await asyncio.gather(one("A.S01E01.srt"), one("B.S02E02.srt")) == [
        ["A.S01E01.srt"],
        ["B.S02E02.srt"],
    ]
    with pytest.raises(ValueError):
        with integration.extracting_member():
            raise ValueError("fixture")
    assert integration._selected_member.get() is None


def test_reference_telemetry_redacts_raw_values(caplog):
    caplog.set_level("INFO")
    sub = subtitle("Private.Title.S01E02.BluRay-GROUP", provider="subdl")
    sub.download_url = "https://example.invalid/?api_key=PRIVATE_SECRET"
    integration.reference_order(
        [sub], {"filename": "Private.Title.S01E02.BluRay-GROUP", "media_type": "series"}
    )
    assert "[weighted-reference]" in caplog.text
    assert (
        "PRIVATE_SECRET" not in caplog.text
        and "Private.Title" not in caplog.text
        and "https://" not in caplog.text
    )


def test_optional_metadata_failure_does_not_change_provider_result():
    sub = subtitle("Show.S01E02.BluRay-A")
    integration.provider_evidence(sub, None)
    assert sub.release_name == "Show.S01E02.BluRay-A" and sub.release_evidence == []


def test_attempt_telemetry_failure_is_consumed(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("optional telemetry fixture")
    monkeypatch.setattr(integration, "matches", fail)
    integration.log_attempt(object(), {})
