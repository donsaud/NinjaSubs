"""Tests for media-context embedding + reference-matching heuristics used by sync."""

from starlette.requests import Request

from app.main import (
    _media_context_from_request,
    _merge_sync_meta,
    _subtitle_context_query,
)
from app.services.reference_resolver import is_informative_release_name
from app.services.sync.orchestrator import build_synced_cache_key as _sync_cache_key


def test_subtitle_context_query_encodes_present_fields_only():
    qs = _subtitle_context_query("tt0773262", "series", 8, 2, "Dexter.S08E02.1080p.WEB-DL.mkv")
    assert "imdb=tt0773262" in qs
    assert "type=series" in qs
    assert "season=8" in qs
    assert "episode=2" in qs
    assert "filename=Dexter.S08E02.1080p.WEB-DL.mkv" in qs

    assert _subtitle_context_query("tt1", None, None, None, None) == "imdb=tt1"
    assert _subtitle_context_query(None, None, None, None, None) == ""


def _make_request(query: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/sub/x.srt",
            "query_string": query.encode(),
            "headers": [],
        }
    )


def test_media_context_from_request_parses_and_coerces():
    request = _make_request("imdb=tt0773262&type=series&season=8&episode=2&filename=wVwm.mkv")
    ctx = _media_context_from_request(request)
    assert ctx["imdb_id"] == "tt0773262"
    assert ctx["media_type"] == "series"
    assert ctx["season"] == 8
    assert ctx["episode"] == 2
    assert ctx["target_filename"] == "wVwm.mkv"

    empty = _media_context_from_request(_make_request(""))
    assert empty == {}


def test_merge_sync_meta_fills_from_context():
    merged = _merge_sync_meta(None, {"imdb_id": "tt1", "season": 8, "target_filename": "wVwm.mkv"})
    assert merged["imdb_id"] == "tt1"
    assert merged["season"] == 8
    assert merged["target_filename"] == "wVwm.mkv"
    assert merged["lang"] == "ara"

    # Existing metadata wins over context.
    merged2 = _merge_sync_meta({"imdb_id": "ttMet"}, {"imdb_id": "ttCtx"})
    assert merged2["imdb_id"] == "ttMet"


def test_sync_cache_key_is_bound_to_exact_payload():
    meta = {"imdb_id": "tt1", "season": 1, "episode": 1}
    key_a = _sync_cache_key(meta, "sub1", content_hash="hash-of-file-a", decision="team")
    key_b = _sync_cache_key(meta, "sub1", content_hash="hash-of-file-b", decision="team")
    assert key_a.startswith("final_sub:tt1:s1e1:")
    assert ":sub1:team:hash-of-file-a" in key_a
    assert key_a != key_b


def test_sync_cache_key_scoped_to_media_fingerprint():
    hashed = _sync_cache_key({"imdb_id": "tt1", "video_hash": "VH"}, "s", "h", "team")
    assert hashed == "final_sub:tt1:movie:vh:s:team:h"
    named = _sync_cache_key({"imdb_id": "tt1", "target_filename": "Show S01E01.mkv"}, "s", "h", "team")
    assert named == "final_sub:tt1:movie:show s01e01.mkv:s:team:h"
    assert hashed != named
    # Same bytes, different verdicts: team and edition syncs never collide.
    edition = _sync_cache_key({"imdb_id": "tt1", "video_hash": "VH"}, "s", "h", "edition")
    assert edition != hashed


def test_video_hash_roundtrips_through_context():
    qs = _subtitle_context_query("tt1", "movie", None, None, "M.mkv", "ab12cd34", 99)
    assert "videohash=ab12cd34" in qs
    assert "videosize=99" in qs
    ctx = _media_context_from_request(_make_request(qs))
    assert ctx["video_hash"] == "ab12cd34"
    assert ctx["video_size"] == "99"
    merged = _merge_sync_meta({"imdb_id": "tt1"}, ctx)
    assert merged["video_hash"] == "ab12cd34"
    assert merged["video_size"] == "99"


def test_is_informative_release_name_heuristic():
    assert is_informative_release_name("Dexter.S08E02.1080p.WEB-DL.x264-FLUX.mkv") is True
    assert is_informative_release_name("Movie 2024 BluRay") is True
    assert is_informative_release_name("wVwm.mkv") is False
    assert is_informative_release_name("aB3d") is False
    assert is_informative_release_name("") is False
    assert is_informative_release_name(None) is False


OBFUSCATED = (
    "Suits.S01E01.Pilot.Part.1.2.1080p.NF.WEB-DL.DDP5.1.H.264-playWEB/"
    "e4WcFo4Tz5J8PoFwiBfP880XsBuHk4dS.mkv"
)
SCENE_PARENT = "Suits.S01E01.Pilot.Part.1.2.1080p.NF.WEB-DL.DDP5.1.H.264-playWEB"


def test_prefer_meaningful_release_name_prefers_scene_parent():
    from app.services.sync.matching import prefer_meaningful_release_name

    assert prefer_meaningful_release_name(OBFUSCATED) == SCENE_PARENT
    # Direct informative filename is returned untouched.
    direct = "Dune.Part.Two.2024.2160p.UHD.BluRay.x265-FLUX.mkv"
    assert prefer_meaningful_release_name(direct) == direct
    # Windows separators are handled too.
    assert (
        prefer_meaningful_release_name(r"C:\dl\Suits.S01E01.1080p.WEB-DL-GRP\abc.mkv")
        == "Suits.S01E01.1080p.WEB-DL-GRP"
    )
    # No informative ancestor: basename stands (no regression).
    assert prefer_meaningful_release_name("downloads/e4WcFo4Tz5J8PoF.mkv") == "e4WcFo4Tz5J8PoF.mkv"
    assert prefer_meaningful_release_name(None) == ""
    assert prefer_meaningful_release_name("") == ""


def test_merge_sync_meta_recovers_parent_from_obfuscated_path():
    merged = _merge_sync_meta({"imdb_id": "tt1"}, {"target_filename": OBFUSCATED})
    assert merged["target_filename"] == SCENE_PARENT
    # The cache fingerprint is now derived from the meaningful name.
    key = _sync_cache_key(merged, "s1", content_hash="h", decision="team")
    assert "suits.s01e01" in key and "web-dl" in key

    # URL context rescues an obfuscated cached basename.
    rescued = _merge_sync_meta(
        {"imdb_id": "tt1", "target_filename": "e4Wc.mkv"}, {"target_filename": OBFUSCATED}
    )
    assert rescued["target_filename"] == SCENE_PARENT

    # No filename anywhere: the stream URL path is used.
    derived = _merge_sync_meta(
        {"imdb_id": "tt1"},
        {"stream_url": "https://cdn.example/Suits.S01E01.1080p.WEB-DL-GRP/e4Wc.mkv"},
    )
    assert derived["target_filename"] == "Suits.S01E01.1080p.WEB-DL-GRP"



