"""Unit tests for Stremio ID parser."""

import pytest

from app.utils.parser import parse_stremio_id


def test_parse_movie_id_standard():
    """Test standard IMDb movie ID."""
    result = parse_stremio_id("tt0111161")
    assert result.imdb_id == "tt0111161"
    assert result.is_series is False
    assert result.season is None
    assert result.episode is None


def test_parse_movie_id_with_json_extension():
    """Test movie ID with .json suffix."""
    result = parse_stremio_id("tt0111161.json")
    assert result.imdb_id == "tt0111161"
    assert result.is_series is False
    assert result.season is None
    assert result.episode is None


def test_parse_series_id_standard():
    """Test standard series ID format (tt1234567:season:episode)."""
    result = parse_stremio_id("tt0944947:1:5")
    assert result.imdb_id == "tt0944947"
    assert result.is_series is True
    assert result.season == 1
    assert result.episode == 5


def test_extract_stream_params_with_ampersand_in_filename():
    """Test extract_stream_params when release group or title contains ampersands."""
    from app.services.ranking import extract_stream_params

    # Case 1: Real Stremio production URL for FMA Brotherhood
    extra_prod = (
        "videoHash=51700dc6914f0812&videoSize=4638564680&"
        "filename=[A&C] Fullmetal Alchemist Brotherhood - 03 [BDRip 1080p] [Multi-Audio] [Multi-Subs] [V2] [BEB049B8].mkv.json"
    )
    res = extract_stream_params(extra_prod)
    assert res["video_hash"] == "51700dc6914f0812"
    assert res["video_size"] == 4638564680
    assert (
        res["filename"]
        == "[A&C] Fullmetal Alchemist Brotherhood - 03 [BDRip 1080p] [Multi-Audio] [Multi-Subs] [V2] [BEB049B8].mkv"
    )

    # Case 2: Filename first, followed by hash and size
    extra_first = (
        "filename=[A&C] Fast & Furious 10.mkv&videoHash=abcdef1234567890&videoSize=1234567"
    )
    res2 = extract_stream_params(extra_first)
    assert res2["filename"] == "[A&C] Fast & Furious 10.mkv"
    assert res2["video_hash"] == "abcdef1234567890"
    assert res2["video_size"] == 1234567


def test_parse_series_id_with_padded_zeroes():
    """Test series ID with zero-padded season and episode."""
    result = parse_stremio_id("tt0944947:02:08")
    assert result.imdb_id == "tt0944947"
    assert result.is_series is True
    assert result.season == 2
    assert result.episode == 8


def test_parse_series_id_with_json_suffix():
    """Test series ID ending with .json."""
    result = parse_stremio_id("tt0944947:3:10.json")
    assert result.imdb_id == "tt0944947"
    assert result.is_series is True
    assert result.season == 3
    assert result.episode == 10


def test_parse_series_id_with_extra_path():
    """Test series ID with extra routing parameters attached."""
    result = parse_stremio_id("tt0944947:4:1/videoHash=1234567890abcdef")
    assert result.imdb_id == "tt0944947"
    assert result.is_series is True
    assert result.season == 4
    assert result.episode == 1


def test_parse_invalid_empty():
    """Test empty ID raises ValueError."""
    with pytest.raises(ValueError, match="Media ID cannot be empty"):
        parse_stremio_id("")


def test_parse_invalid_non_imdb():
    """Test non-IMDb ID format raises ValueError."""
    with pytest.raises(ValueError, match="Invalid IMDb ID format"):
        parse_stremio_id("custom_id_1234")


def test_parse_invalid_series_non_integer():
    """Test series ID with non-integer season or episode raises ValueError."""
    with pytest.raises(ValueError, match="Invalid season/episode integers"):
        parse_stremio_id("tt1234567:one:two")


def test_parse_kitsu_movie():
    """Test Kitsu movie ID format (kitsu:12345)."""
    result = parse_stremio_id("kitsu:12345")
    assert result.imdb_id == "kitsu:12345"
    assert result.is_series is False
    assert result.season is None
    assert result.episode is None


def test_parse_kitsu_series_episode():
    """Test Kitsu anime series episode format (kitsu:12345:12)."""
    result = parse_stremio_id("kitsu:12345:12")
    assert result.imdb_id == "kitsu:12345"
    assert result.is_series is True
    assert result.season == 1
    assert result.episode == 12


def test_parse_anime_two_part_id():
    """Test 2-part anime ID format (tt1234567:5)."""
    result = parse_stremio_id("tt0409591:5")
    assert result.imdb_id == "tt0409591"
    assert result.is_series is True
    assert result.season == 1
    assert result.episode == 5
