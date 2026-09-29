import json
from unittest.mock import MagicMock, patch

import pytest

from app.services.sync.embedded_strategy import LocalEmbeddedStrategy
from app.services.sync.query import ReferenceQuery, ResolvedReference


@pytest.mark.asyncio
async def test_embedded_strategy_returns_none_when_no_filename(tmp_path):
    strategy = LocalEmbeddedStrategy(
        media_dirs=[str(tmp_path)],
        ffprobe_path="ffprobe",
        ffmpeg_path="ffmpeg",
    )
    query = ReferenceQuery(imdb_id="tt1234567", target_filename="")
    res = await strategy.resolve_with_provenance(query)
    assert res.text is None


@pytest.mark.asyncio
async def test_embedded_strategy_returns_cached_reference(tmp_path):
    strategy = LocalEmbeddedStrategy(
        media_dirs=[str(tmp_path)],
        ffprobe_path="ffprobe",
        ffmpeg_path="ffmpeg",
    )
    query = ReferenceQuery(imdb_id="tt1234567", target_filename="Test.Movie.2024.1080p.mkv")

    # Mock cache hit
    mock_cached = ResolvedReference("1\n00:01:00,000 --> 00:01:02,000\nHello\n", kind="embedded", partial=True)
    with patch.object(strategy.cache, "get", return_value=mock_cached):
        res = await strategy.resolve_with_provenance(query)
        assert res.text == mock_cached.text
        assert res.kind == "embedded"
        assert res.partial is True


@pytest.mark.asyncio
async def test_embedded_strategy_extracts_from_local_media(tmp_path):
    # Create fake media file
    media_file = tmp_path / "Whiplash.2014.MULTI.DV.2160p.WEB.H265-LOST.mkv"
    media_file.write_bytes(b"FAKE_VIDEO_CONTENT")

    strategy = LocalEmbeddedStrategy(
        media_dirs=[str(tmp_path)],
        ffprobe_path="ffprobe",
        ffmpeg_path="ffmpeg",
    )
    query = ReferenceQuery(
        imdb_id="tt2582802",
        target_filename="Whiplash.2014.MULTI.DV.2160p.WEB.H265-LOST.mkv",
        video_size=18,
    )

    fake_srt = "1\n00:02:32,194 --> 00:02:33,320\nDesole...\n\n2\n00:02:33,529 --> 00:02:35,197\nNon, reste la.\n\n3\n00:02:40,869 --> 00:02:41,995\nTon nom ?\n\n4\n00:02:42,663 --> 00:02:43,705\nAndrew Neiman.\n\n5\n00:02:44,414 --> 00:02:45,457\nQuelle annee ?\n\n6\n00:02:46,375 --> 00:02:47,584\n1re annee.\n\n7\n00:02:48,168 --> 00:02:49,294\nTu sais qui je suis ?\n\n8\n00:02:49,962 --> 00:02:51,004\nOui.\n"

    with patch.object(strategy, "_probe_best_subtitle_stream", return_value=4), \
         patch.object(strategy, "_extract_partial_reference", return_value=fake_srt):
        res = await strategy.resolve_with_provenance(query)
        assert res.text is not None
        assert "00:02:32,194" in res.text
        assert res.kind == "embedded"
        assert res.partial is False


@pytest.mark.asyncio
async def test_embedded_strategy_probes_subtitles_filters_commentary(tmp_path):
    strategy = LocalEmbeddedStrategy(
        media_dirs=[str(tmp_path)],
        ffprobe_path="ffprobe",
        ffmpeg_path="ffmpeg",
    )

    fake_ffprobe_json = {
        "streams": [
            {"index": 2, "codec_name": "subrip", "tags": {"language": "eng", "title": "Director Commentary"}},
            {"index": 3, "codec_name": "subrip", "tags": {"language": "fre", "title": "French (forced)"}},
            {"index": 4, "codec_name": "subrip", "tags": {"language": "fre", "title": "French"}},
        ]
    }

    mock_run = MagicMock()
    mock_run.returncode = 0
    mock_run.stdout = json.dumps(fake_ffprobe_json)

    with patch("subprocess.run", return_value=mock_run):
        best_idx = strategy._probe_best_subtitle_stream(str(tmp_path / "dummy.mkv"))
        assert best_idx == 4


@pytest.mark.asyncio
async def test_embedded_strategy_extracts_from_remote_stream_when_not_local(tmp_path):
    strategy = LocalEmbeddedStrategy(
        media_dirs=[str(tmp_path)],
        ffprobe_path="ffprobe",
        ffmpeg_path="ffmpeg",
    )
    query = ReferenceQuery(
        imdb_id="tt2582802",
        target_filename="Whiplash.2014.1080p.BluRay.x264.DTS-WiHD.mkv",
        stream_url="https://debrid.example.com/stream/whiplash.mkv",
    )

    fake_srt = "1\n00:02:32,194 --> 00:02:33,320\nSorry...\n\n2\n00:02:33,529 --> 00:02:35,197\nNo, stay there.\n\n3\n00:02:40,869 --> 00:02:41,995\nYour name?\n\n4\n00:02:42,663 --> 00:02:43,705\nAndrew Neiman.\n\n5\n00:02:44,414 --> 00:02:45,457\nWhat year?\n\n6\n00:02:46,375 --> 00:02:47,584\nFirst year.\n\n7\n00:02:48,168 --> 00:02:49,294\nDo you know who I am?\n\n8\n00:02:49,962 --> 00:02:51,004\nYes.\n"

    with patch.object(strategy, "_probe_best_subtitle_stream", return_value=1) as mock_probe, \
         patch.object(strategy, "_extract_partial_reference", return_value=fake_srt) as mock_extract:
        res = await strategy.resolve_with_provenance(query)
        assert res.text is not None
        assert res.kind == "embedded"
        assert res.partial is True
        mock_probe.assert_called_once_with("https://debrid.example.com/stream/whiplash.mkv")
        mock_extract.assert_called_once_with("https://debrid.example.com/stream/whiplash.mkv", 1, is_remote=True)

