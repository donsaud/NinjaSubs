"""Unit tests for the range-based Matroska subtitle extractor."""

import logging
import struct

import pytest

from app.services.sync import mkv_range as M


# --------------------------------------------------------------------------- #
# Tiny EBML builder (fixed-width sizes/payloads keep lengths deterministic)
# --------------------------------------------------------------------------- #
def _vint_size(n: int) -> bytes:
    return b"\x01" + n.to_bytes(7, "big")  # 8-byte VINT


def _eid(value: int) -> bytes:
    length = max(1, (value.bit_length() + 7) // 8)
    return value.to_bytes(length, "big")


def _el(eid: int, payload: bytes) -> bytes:
    return _eid(eid) + _vint_size(len(payload)) + payload


def _u8(n: int) -> bytes:
    return n.to_bytes(8, "big")


def _block(track: int, rel_ms: int, text: bytes, duration_ms: int | None) -> bytes:
    block = bytes([0x80 | track]) + struct.pack(">h", rel_ms) + b"\x00" + text
    if duration_ms is None:
        return _el(M._ID_SIMPLE_BLOCK, block)
    return _el(M._ID_BLOCK_GROUP, _el(M._ID_BLOCK, block) + _el(M._ID_BLOCK_DURATION, _u8(duration_ms)))


def _build_mkv() -> bytes:
    track = _el(
        M._ID_TRACK_ENTRY,
        _el(M._ID_TRACK_NUMBER, _u8(5))
        + _el(M._ID_TRACK_TYPE, _u8(0x11))
        + _el(M._ID_CODEC_ID, b"S_TEXT/UTF8")
        + _el(M._ID_LANGUAGE, b"eng"),
    )
    tracks = _el(M._ID_TRACKS, track)
    info = _el(M._ID_INFO, _el(M._ID_TIMESTAMP_SCALE, _u8(1_000_000)))

    cluster = _el(
        M._ID_CLUSTER,
        _el(M._ID_TIMESTAMP, _u8(1000))
        + b"".join(
            _block(5, i * 2000, f"Line {i}".encode(), 1500) for i in range(6)
        ),
    )

    def cue(cluster_pos: int) -> bytes:
        return _el(
            M._ID_CUE_POINT,
            _el(M._ID_CUE_TIME, _u8(1000))
            + _el(
                M._ID_CUE_TRACK_POSITIONS,
                _el(0xF7, _u8(1)) + _el(M._ID_CUE_CLUSTER_POSITION, _u8(cluster_pos)),
            ),
        )

    # Build SeekHead with placeholder positions to learn element sizes.
    def seek_head(tracks_off: int, cues_off: int) -> bytes:
        def seek(eid: int, pos: int) -> bytes:
            return _el(M._ID_SEEK, _el(M._ID_SEEK_ID, _eid(eid)) + _el(M._ID_SEEK_POSITION, _u8(pos)))

        return _el(M._ID_SEEK_HEAD, seek(M._ID_TRACKS, tracks_off) + seek(M._ID_CUES, cues_off))

    sh = seek_head(0, 0)
    tracks_off = len(sh) + len(info)
    cues_placeholder = _el(M._ID_CUES, cue(0))
    cues_off = tracks_off + len(tracks)
    cluster_off = cues_off + len(cues_placeholder)
    cues = _el(M._ID_CUES, cue(cluster_off))
    sh = seek_head(tracks_off, cues_off)

    segment = sh + info + tracks + cues + cluster
    return _el(0x1A45DFA3, b"") + _el(M._ID_SEGMENT, segment)


def test_find_segment_and_tracks():
    data = _build_mkv()
    seg_start, seg_end = M.find_segment(data)
    assert seg_end > seg_start
    tracks = []
    for eid, ds, de in M._children(data, seg_start, seg_end):
        if eid == M._ID_TRACKS:
            tracks = M.parse_tracks(data, ds, de)
    assert tracks == [{"number": 5, "codec": "S_TEXT/UTF8", "language": "eng"}]


def test_parse_cues_and_cluster_block():
    data = _build_mkv()
    seg_start, _ = M.find_segment(data)
    sees = {}
    for eid, ds, de in M._children(data, seg_start, len(data)):
        if eid == M._ID_SEEK_HEAD:
            sees.update(M.parse_seek_head(data, ds, de))
    cue_blob = data[seg_start + sees[M._ID_CUES] :]
    contents = M._element_contents(cue_blob, M._ID_CUES)
    cues = M.parse_cues(cue_blob, contents[0], contents[1])
    assert cues and cues[0][0] == 1000
    cluster_rel = cues[0][1]

    cluster = data[seg_start + cluster_rel :]
    decoded = M.parse_cluster(cluster, 0, len(cluster), 5)
    assert len(decoded) == 6
    assert decoded[0] == (1000, 1500, "Line 0")
    assert decoded[5] == (1000 + 10_000, 1500, "Line 5")


@pytest.mark.asyncio
async def test_extract_embedded_srt_end_to_end(monkeypatch):
    data = _build_mkv()

    class _Client:
        pass

    async def fake_fetch(client, url, start, size):
        return data[start : start + size]

    monkeypatch.setattr(M, "_fetch", fake_fetch)
    out = await M.extract_embedded_srt("http://host/movie.mkv", _Client(), timeout=5.0)
    assert out is not None
    text = out.decode("utf-8")
    assert "00:00:01,000 --> 00:00:02,500" in text
    assert "Line 0" in text and "Line 5" in text


def test_srt_timestamp():
    assert M._srt_timestamp(1000) == "00:00:01,000"
    assert M._srt_timestamp(3_661_500) == "01:01:01,500"
    assert M._srt_timestamp(-5) == "00:00:00,000"


# --------------------------------------------------------------------------- #
# Track discovery diagnostics
# --------------------------------------------------------------------------- #
def _track_entry(number: int, track_type: int, codec: bytes, lang: bytes) -> bytes:
    return _el(
        M._ID_TRACK_ENTRY,
        _el(M._ID_TRACK_NUMBER, _u8(number))
        + _el(M._ID_TRACK_TYPE, _u8(track_type))
        + _el(M._ID_CODEC_ID, codec)
        + _el(M._ID_LANGUAGE, lang),
    )


def _build_mkv_with_tracks(tracks: list[bytes]) -> bytes:
    """Minimal MKV whose Tracks element is inlined in the header (no cues)."""
    tracks_el = _el(M._ID_TRACKS, b"".join(tracks))
    info = _el(M._ID_INFO, _el(M._ID_TIMESTAMP_SCALE, _u8(1_000_000)))
    segment = info + tracks_el
    return _el(0x1A45DFA3, b"") + _el(M._ID_SEGMENT, segment)


def test_parse_all_tracks_reports_type_codec_language():
    data = _build_mkv_with_tracks(
        [
            _track_entry(1, M._TRACK_TYPE_VIDEO, b"V_MPEG4/ISO/AVC", b"und"),
            _track_entry(2, M._TRACK_TYPE_AUDIO, b"A_AAC", b"eng"),
            _track_entry(3, M._TRACK_TYPE_SUBTITLE, b"S_HDMV/PGS", b"eng"),
            _track_entry(4, M._TRACK_TYPE_SUBTITLE, b"S_TEXT/ASS", b"ara"),
            _track_entry(5, M._TRACK_TYPE_SUBTITLE, b"S_TEXT/UTF8", b"eng"),
        ]
    )
    seg_start, seg_end = M.find_segment(data)
    tracks_range = next(
        (ds, de) for eid, ds, de in M._children(data, seg_start, seg_end) if eid == M._ID_TRACKS
    )
    parsed = M.parse_all_tracks(data, *tracks_range)
    assert [(t["number"], t["type"], t["codec"], t["language"]) for t in parsed] == [
        (1, M._TRACK_TYPE_VIDEO, "V_MPEG4/ISO/AVC", "und"),
        (2, M._TRACK_TYPE_AUDIO, "A_AAC", "eng"),
        (3, M._TRACK_TYPE_SUBTITLE, "S_HDMV/PGS", "eng"),
        (4, M._TRACK_TYPE_SUBTITLE, "S_TEXT/ASS", "ara"),
        (5, M._TRACK_TYPE_SUBTITLE, "S_TEXT/UTF8", "eng"),
    ]
    # parse_tracks keeps the text-codec family (ASS + UTF8), dropping PGS.
    assert M.parse_tracks(data, *tracks_range) == [
        {"number": 4, "codec": "S_TEXT/ASS", "language": "ara"},
        {"number": 5, "codec": "S_TEXT/UTF8", "language": "eng"},
    ]


@pytest.mark.asyncio
async def test_extract_logs_unsupported_subtitle_tracks(monkeypatch, caplog):
    data = _build_mkv_with_tracks(
        [
            _track_entry(1, M._TRACK_TYPE_VIDEO, b"V_MPEG4/ISO/AVC", b"und"),
            _track_entry(2, M._TRACK_TYPE_AUDIO, b"A_AAC", b"eng"),
            _track_entry(3, M._TRACK_TYPE_SUBTITLE, b"S_HDMV/PGS", b"eng"),
            _track_entry(4, M._TRACK_TYPE_SUBTITLE, b"S_TEXT/ASS", b"ara"),
        ]
    )

    async def fake_fetch(client, url, start, size):
        return data[start : start + size]

    monkeypatch.setattr(M, "_fetch", fake_fetch)
    with caplog.at_level(logging.INFO, logger="app.services.sync.mkv_range"):
        out = await M.extract_embedded_srt("http://host/movie.mkv", object(), timeout=5.0)

    assert out is None
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "detected 2 subtitle track(s)" in messages
    assert "Track 3: CodecID='S_HDMV/PGS', Lang='eng'" in messages
    assert "Track 4: CodecID='S_TEXT/ASS', Lang='ara'" in messages
    assert "no S_TEXT/UTF8 track found" in messages
    assert "Total video/audio/sub tracks: 4" in messages


@pytest.mark.asyncio
async def test_extract_logs_zero_subtitle_tracks(monkeypatch, caplog):
    data = _build_mkv_with_tracks(
        [
            _track_entry(1, M._TRACK_TYPE_VIDEO, b"V_MPEG4/ISO/AVC", b"und"),
            _track_entry(2, M._TRACK_TYPE_AUDIO, b"A_AAC", b"eng"),
        ]
    )

    async def fake_fetch(client, url, start, size):
        return data[start : start + size]

    monkeypatch.setattr(M, "_fetch", fake_fetch)
    with caplog.at_level(logging.INFO, logger="app.services.sync.mkv_range"):
        out = await M.extract_embedded_srt("http://host/movie.mkv", object(), timeout=5.0)

    assert out is None
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "file contains 0 subtitle tracks (only video/audio detected)" in messages
    assert "no S_TEXT/UTF8 track found" not in messages


@pytest.mark.asyncio
async def test_extract_logs_detected_tracks_on_supported_srt(monkeypatch, caplog):
    data = _build_mkv()

    async def fake_fetch(client, url, start, size):
        return data[start : start + size]

    monkeypatch.setattr(M, "_fetch", fake_fetch)
    with caplog.at_level(logging.INFO, logger="app.services.sync.mkv_range"):
        out = await M.extract_embedded_srt("http://host/movie.mkv", object(), timeout=5.0)

    assert out is not None
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "detected 1 subtitle track(s)" in messages
    assert "Track 5: CodecID='S_TEXT/UTF8', Lang='eng'" in messages
    assert "no S_TEXT/UTF8 track found" not in messages
