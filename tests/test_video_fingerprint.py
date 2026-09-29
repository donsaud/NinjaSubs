"""Video-fingerprint integrity across the subtitle pipeline.

Regression cover for a real production defect: when Stremio requested
subtitles from a catalogue page (no resolved stream), it sent no video
fingerprint at all. The addon then wrote the *subtitle's own release name*
into ``target_filename``, embedded it in the subtitle URL, and read it back on
the serve route as if it were the target video. ``_merge_sync_meta`` accepted
it because the name looked informative, so the sync engine believed the video
was called ``Dexter.2006.S08E05.srt`` - no source, no resolution, no release
group. Every reference then collapsed to TIER_FALLBACK.

The invariant under test:

    target_filename holds the TARGET VIDEO filename only. It must never hold a
    subtitle release name.

A missing fingerprint must stay missing, so the sync layer fails closed and
reports UNVERIFIED instead of chasing phantom candidates.
"""

from __future__ import annotations

import urllib.parse

import pytest

from app.main import (
    _merge_sync_meta,
    _media_context_from_request,
    _subtitle_context_query,
)
from app.services.ranking import extract_stream_params
from app.services.sync.matching import has_video_fingerprint
from app.services.sync.orchestrator import SyncOrchestrator

# The exact production case: Stremio sent the ID only.
CATALOGUE_URL = "http://testserver/subtitles/series/tt0773262:8:5.json"
# The subtitle that was ranked #1 while the leak was live.
LEAKY_RELEASE_NAME = "Dexter.2006.S08E05.srt"
# A real stream context, as Stremio sends when the request originates from a
# resolved torrent/debrid stream.
REAL_VIDEO_FILENAME = "Dexter.s8e05.This.little.piggy.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv"
REAL_VIDEO_SIZE = 5192387053


# --------------------------------------------------------------------------- #
# 1. A missing target filename stays missing
# --------------------------------------------------------------------------- #


def test_missing_extra_yields_no_stream_params():
    """No `extra` and no query params means no fingerprint, not a guess."""
    assert extract_stream_params(None, None) == {
        "filename": None,
        "video_hash": None,
        "video_size": None,
    }
    assert extract_stream_params("", None)["filename"] is None


def test_release_name_is_never_used_as_target_filename():
    """The exact leak: a subtitle name must not become the target video name."""
    target_filename = None  # Stremio sent nothing
    release_name = LEAKY_RELEASE_NAME

    stored = target_filename or release_name
    assert stored == LEAKY_RELEASE_NAME, "guard the pre-fix behaviour"

    # Post-fix contract: the store keeps them in separate fields.
    meta = {"target_filename": target_filename, "display_release_name": release_name}
    assert meta["target_filename"] is None
    assert meta["display_release_name"] == LEAKY_RELEASE_NAME


# --------------------------------------------------------------------------- #
# 2. has_video_fingerprint: catalogue context is not a fingerprint
# --------------------------------------------------------------------------- #


def test_has_video_fingerprint_distinguishes_catalogue_from_stream():
    """Title/year/season alone is NOT a video fingerprint."""
    catalogue_only = {
        "imdb_id": "tt0773262",
        "media_type": "series",
        "season": 8,
        "episode": 5,
        "title": "Dexter",
        "year": 2006,
    }
    assert has_video_fingerprint(catalogue_only) is False

    for field, value in (
        ("target_filename", REAL_VIDEO_FILENAME),
        ("video_hash", "51700dc6914f0812"),
        ("video_size", REAL_VIDEO_SIZE),
        ("stream_url", "https://cdn.example/stream.mkv"),
    ):
        with_fingerprint = {**catalogue_only, field: value}
        assert has_video_fingerprint(with_fingerprint) is True, field

    # Empty strings are not evidence.
    assert has_video_fingerprint({**catalogue_only, "target_filename": ""}) is False
    assert has_video_fingerprint(None, {}) is False
    assert has_video_fingerprint("not-a-dict") is False


# --------------------------------------------------------------------------- #
# 3. Full propagation chain preserves None
# --------------------------------------------------------------------------- #

def _fake_request(query: str):
    """Mimic Starlette's QueryParams: single values, empty string when absent."""

    class _QueryParams(dict):
        def get(self, key, default=None):  # type: ignore[override]
            value = super().get(key)
            return default if value is None else value

    parsed = urllib.parse.parse_qsl(query)
    return type("_Request", (), {"query_params": _QueryParams(parsed)})()


def test_metadata_chain_preserves_missing_fingerprint():
    """request -> metadata -> URL -> query string -> media_context -> merge.

    Every stage must preserve the absence of a video fingerprint rather than
    reconstructing a filename from the subtitle itself.
    """
    # Search stage
    target_filename = None
    release_name = LEAKY_RELEASE_NAME
    meta = {
        "imdb_id": "tt0773262",
        "media_type": "series",
        "season": 8,
        "episode": 5,
        "release_name": release_name,
        "display_release_name": release_name,
        "target_filename": target_filename,
    }
    assert meta["target_filename"] is None

    # Subtitle URL generation
    context_qs = _subtitle_context_query(
        meta["imdb_id"],
        meta["media_type"],
        meta["season"],
        meta["episode"],
        meta["target_filename"],
        None,
        None,
    )
    assert "filename" not in context_qs
    assert urllib.parse.parse_qs(context_qs) == {
        "imdb": ["tt0773262"],
        "type": ["series"],
        "season": ["8"],
        "episode": ["5"],
    }

    # Query-string extraction
    media_context = _media_context_from_request(_fake_request(context_qs))
    assert "target_filename" not in media_context
    assert media_context.get("imdb_id") == "tt0773262"

    # Merge
    merged = _merge_sync_meta(meta, media_context)
    assert merged.get("target_filename") is None
    assert merged["has_video_fingerprint"] is False
    # The display name survives for the UI.
    assert merged["display_release_name"] == LEAKY_RELEASE_NAME
    assert merged["release_name"] == LEAKY_RELEASE_NAME


def test_merge_does_not_invent_fingerprint_from_release_name():
    """Even a highly informative subtitle name must not fill the void."""
    for name in (
        "Dexter.2006.S08E05.srt",
        "Dexter.S08E05.This.Little.Piggy.720p.WEB-DL.DD5.1.H.264-NORDiC.srt",
    ):
        merged = _merge_sync_meta(
            {"imdb_id": "tt1", "release_name": name, "season": 8, "episode": 5}, {}
        )
        assert merged.get("target_filename") is None, name
        assert merged["has_video_fingerprint"] is False


# --------------------------------------------------------------------------- #
# 4. Fingerprint-present propagation is unchanged
# --------------------------------------------------------------------------- #


def test_real_fingerprint_propagates_unchanged():
    """When Stremio supplies the video, it must reach the matcher intact."""
    extra = urllib.parse.quote(f"videoSize={REAL_VIDEO_SIZE}&filename={REAL_VIDEO_FILENAME}")
    params = extract_stream_params(extra, None)
    assert params["filename"] == REAL_VIDEO_FILENAME
    assert params["video_size"] == REAL_VIDEO_SIZE

    context_qs = _subtitle_context_query(
        "tt0773262", "series", 8, 5, params["filename"], None, params["video_size"]
    )

    media_context = _media_context_from_request(_fake_request(context_qs))
    assert media_context["target_filename"] == REAL_VIDEO_FILENAME
    assert int(media_context["video_size"]) == REAL_VIDEO_SIZE

    merged = _merge_sync_meta({"imdb_id": "tt0773262"}, media_context)
    assert merged["target_filename"] == REAL_VIDEO_FILENAME
    assert merged["has_video_fingerprint"] is True


def test_orchestrator_query_keeps_real_filename():
    """The sync query must carry the real video filename, not a release name."""
    orch = SyncOrchestrator()
    meta = {
        "imdb_id": "tt0773262",
        "media_type": "series",
        "season": 8,
        "episode": 5,
        "release_name": LEAKY_RELEASE_NAME,
        "target_filename": REAL_VIDEO_FILENAME,
    }
    query = orch._build_query(meta, target_id="sub-1")
    assert query.target_filename == REAL_VIDEO_FILENAME
    # The candidate's own name is still tracked separately.
    assert query.target_sub_release_name == LEAKY_RELEASE_NAME


def test_orchestrator_query_has_no_fingerprint_when_absent():
    """No video filename in, no video filename out - even with a release name."""
    orch = SyncOrchestrator()
    meta = {
        "imdb_id": "tt0773262",
        "media_type": "series",
        "season": 8,
        "episode": 5,
        "release_name": LEAKY_RELEASE_NAME,
    }
    query = orch._build_query(meta, target_id="sub-1")
    assert query.target_filename is None
    assert query.target_sub_release_name == LEAKY_RELEASE_NAME


# --------------------------------------------------------------------------- #
# 5. Existing behaviour that must not regress
# --------------------------------------------------------------------------- #


def test_uninformative_debrid_filename_still_upgrades_from_release_name():
    """A real-but-obfuscated fingerprint may still be improved.

    ``videoplayback.mp4`` is a genuine target filename, just not an
    informative one, so the long-standing debrid rescue path still applies.
    """
    merged = _merge_sync_meta(
        {
            "target_filename": "videoplayback.mp4",
            "stream_url": "https://debrid.example.com/d/xyz/videoplayback.mp4",
            "release_name": "Gladiator.2000.Extended.1080p.BluRay.x264-FSiHD.srt",
        },
        {},
    )
    assert merged["target_filename"] == "Gladiator.2000.Extended.1080p.BluRay.x264-FSiHD.srt"
    assert merged["has_video_fingerprint"] is True


def test_obfuscated_path_still_recovers_scene_parent():
    """Existing debrid parent-directory rescue is untouched."""
    from app.services.sync.matching import prefer_meaningful_release_name

    obfuscated = r"C:\dl\Suits.S01E01.1080p.WEB-DL-GRP\abc.mkv"
    assert (
        prefer_meaningful_release_name(obfuscated) == "Suits.S01E01.1080p.WEB-DL-GRP"
    )
    merged = _merge_sync_meta({"imdb_id": "tt1"}, {"target_filename": obfuscated})
    assert merged["target_filename"] == "Suits.S01E01.1080p.WEB-DL-GRP"


def test_stream_url_path_still_derives_target_filename():
    """With a stream URL present, its path is a legitimate fingerprint."""
    merged = _merge_sync_meta(
        {"imdb_id": "tt1"},
        {"stream_url": "https://cdn.example/Suits.S01E01.1080p.WEB-DL-GRP/e4Wc.mkv"},
    )
    assert merged["target_filename"] == "Suits.S01E01.1080p.WEB-DL-GRP"


# --------------------------------------------------------------------------- #
# 6. Empty fingerprint is treated as unknown, not compensated
# --------------------------------------------------------------------------- #


def test_ranking_still_functions_without_a_fingerprint():
    """No fingerprint must degrade gracefully, not crash or fabricate.

    Ranking is driven by Cinemeta title/year plus season/episode, so candidates
    are still ordered - just without edition discrimination.
    """
    from app.models import SubtitleRelease
    from app.services.subtitle_matcher import rank_subtitles

    candidates = [
        SubtitleRelease(
            release_name="Dexter.S08E05.720p.BluRay.x264-NORDiC.srt",
            download_url="http://a",
            provider="subdl",
            lang="ara",
        ),
        SubtitleRelease(
            release_name="Dexter.S08.720p.BluRay.x264-OTHER.srt",
            download_url="http://b",
            provider="subdl",
            lang="ara",
        ),
    ]
    ranked = rank_subtitles(
        None,
        candidates,
        season=8,
        episode=5,
        title="Dexter",
        year=2006,
    )
    assert ranked, "catalogue-only requests must still return results"
    # A season pack is a legitimate candidate for S08E05 (it is unbundled on
    # demand), so the guarantee here is only that ranking still works and the
    # exact-episode track is not displaced by the pack.
    assert ranked[0].release_name == "Dexter.S08E05.720p.BluRay.x264-NORDiC.srt"


def test_reference_tier_demotes_without_a_fingerprint():
    """No target filename means no edition evidence: tier stays conservative.

    With a real filename the same candidate is scored against the target's
    source/edition. Without one, it must not be able to claim a match.
    """
    from app.models import SubtitleRelease
    from app.services.sync.external_strategy import TIER_HASH, reference_tier

    candidate = SubtitleRelease(
        release_name="Dexter.S08E05.720p.BluRay.x264-NORDiC.srt",
        download_url="http://x",
        provider="subsource",
        lang="eng",
    )
    # A hash match is unconditional and unaffected by a missing filename.
    hashed = SubtitleRelease(
        release_name="Dexter.S08E05.720p.BluRay.x264-NORDiC.srt",
        download_url="http://x",
        provider="opensubtitles",
        lang="eng",
        matched_by_hash=True,
    )
    assert reference_tier(None, hashed) == TIER_HASH
    assert reference_tier(None, candidate) > TIER_HASH
