"""Unit tests for the single-window head-sampling Matroska extractor."""

import logging
import struct
from types import SimpleNamespace

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
    return _el(
        M._ID_BLOCK_GROUP,
        _el(M._ID_BLOCK, block) + _el(M._ID_BLOCK_DURATION, _u8(duration_ms)),
    )


def _blocks_for(track: int, count: int, prefix: str) -> bytes:
    """100 ms spacing keeps the int16 relative timestamp in range for 300+ cues."""
    return b"".join(_block(track, i * 100, f"{prefix}{i}".encode(), 90) for i in range(count))


def _track_entry(
    number: int,
    track_type: int,
    codec: bytes,
    lang: bytes,
    name: bytes | None = None,
    forced: bool | None = None,
) -> bytes:
    payload = (
        _el(M._ID_TRACK_NUMBER, _u8(number))
        + _el(M._ID_TRACK_TYPE, _u8(track_type))
        + _el(M._ID_CODEC_ID, codec)
        + _el(M._ID_LANGUAGE, lang)
    )
    if name is not None:
        payload += _el(M._ID_NAME, name)
    if forced is not None:
        payload += _el(M._ID_FLAG_FORCED, _u8(1 if forced else 0))
    return _el(M._ID_TRACK_ENTRY, payload)


def _build_mkv(track_entries: list[bytes], clusters: list[tuple[int, bytes]]) -> bytes:
    """Info-free MKV: Tracks + one Cluster per ``(cluster_time_ms, block_bytes)``.

    The single-window extractor only needs ``Tracks`` and ``Cluster`` elements,
    so no SeekHead/Cues are emitted.
    """
    tracks = _el(M._ID_TRACKS, b"".join(track_entries))
    body = b"".join(
        _el(M._ID_CLUSTER, _el(M._ID_TIMESTAMP, _u8(cluster_time)) + blocks)
        for cluster_time, blocks in clusters
    )
    return _el(0x1A45DFA3, b"") + _el(M._ID_SEGMENT, tracks + body)


def _build_mkv_with_tracks(track_entries: list[bytes]) -> bytes:
    """MKV with only a Tracks element (no clusters)."""
    tracks = _el(M._ID_TRACKS, b"".join(track_entries))
    return _el(0x1A45DFA3, b"") + _el(M._ID_SEGMENT, tracks)


def _build_mkv_without_tracks() -> bytes:
    cluster = _el(M._ID_CLUSTER, _el(M._ID_TIMESTAMP, _u8(0)))
    return _el(0x1A45DFA3, b"") + _el(M._ID_SEGMENT, cluster)


def test_find_segment_and_tracks():
    data = _build_mkv(
        [_track_entry(5, M._TRACK_TYPE_SUBTITLE, b"S_TEXT/UTF8", b"eng")],
        [(1000, b"")],
    )
    seg_start, seg_end = M.find_segment(data)
    assert seg_end > seg_start
    tracks = []
    for eid, ds, de in M._children(data, seg_start, seg_end):
        if eid == M._ID_TRACKS:
            tracks = M.parse_tracks(data, ds, de)
    assert tracks == [{"number": 5, "codec": "S_TEXT/UTF8", "language": "eng"}]


def test_parse_cluster_decodes_blocks():
    data = _build_mkv(
        [_track_entry(5, M._TRACK_TYPE_SUBTITLE, b"S_TEXT/UTF8", b"eng")],
        [(1000, _blocks_for(5, 6, "L"))],
    )
    seg_start, seg_end = M.find_segment(data)
    cluster_range = next(
        (ds, de) for eid, ds, de in M._children(data, seg_start, seg_end) if eid == M._ID_CLUSTER
    )
    decoded = M._parse_cluster_contents(data, cluster_range[0], cluster_range[1], 5)
    assert len(decoded) == 6
    assert decoded[0] == (1000, 90, "L0")


def test_srt_timestamp():
    assert M._srt_timestamp(1000) == "00:00:01,000"
    assert M._srt_timestamp(3_661_500) == "01:01:01,500"
    assert M._srt_timestamp(-5) == "00:00:00,000"


# --------------------------------------------------------------------------- #
# Track parsing + diagnostics
# --------------------------------------------------------------------------- #
def _tracks_from(data: bytes) -> tuple[int, int, list[dict]]:
    seg_start, seg_end = M.find_segment(data)
    tracks_range = next(
        (ds, de) for eid, ds, de in M._children(data, seg_start, seg_end) if eid == M._ID_TRACKS
    )
    return seg_start, seg_end, M.parse_all_tracks(data, *tracks_range)


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
    _, _, parsed = _tracks_from(data)
    assert [(t["number"], t["type"], t["codec"], t["language"]) for t in parsed] == [
        (1, M._TRACK_TYPE_VIDEO, "V_MPEG4/ISO/AVC", "und"),
        (2, M._TRACK_TYPE_AUDIO, "A_AAC", "eng"),
        (3, M._TRACK_TYPE_SUBTITLE, "S_HDMV/PGS", "eng"),
        (4, M._TRACK_TYPE_SUBTITLE, "S_TEXT/ASS", "ara"),
        (5, M._TRACK_TYPE_SUBTITLE, "S_TEXT/UTF8", "eng"),
    ]


def test_parse_all_tracks_extracts_name_and_flag_forced():
    data = _build_mkv_with_tracks(
        [
            _track_entry(
                3,
                M._TRACK_TYPE_SUBTITLE,
                b"S_TEXT/UTF8",
                b"eng",
                name=b"Signs & Songs (forced)",
                forced=True,
            ),
            _track_entry(
                4,
                M._TRACK_TYPE_SUBTITLE,
                b"S_TEXT/UTF8",
                b"eng",
                name=b"English SDH",
                forced=False,
            ),
        ]
    )
    _, _, parsed = _tracks_from(data)
    assert parsed[0]["name"] == "Signs & Songs (forced)"
    assert parsed[0]["forced"] is True
    assert parsed[1]["name"] == "English SDH"
    assert parsed[1]["forced"] is False
    # No FlagForced element -> defaults to False.
    _, _, plain = _tracks_from(
        _build_mkv_with_tracks([_track_entry(5, M._TRACK_TYPE_SUBTITLE, b"S_TEXT/UTF8", b"eng")])
    )
    assert plain[0]["forced"] is False and plain[0]["name"] == ""


def test_is_low_priority_and_track_priority():
    forced = {"number": 3, "language": "eng", "name": "", "forced": True}
    signs = {"number": 4, "language": "eng", "name": "Signs", "forced": False}
    commentary = {"number": 5, "language": "eng", "name": "Commentary", "forced": False}
    plain = {"number": 6, "language": "eng", "name": "English", "forced": False}
    sdh = {"number": 7, "language": "eng", "name": "English SDH", "forced": False}
    other_lang = {"number": 8, "language": "ara", "name": "Arabic", "forced": False}

    assert M._is_low_priority_track(forced)
    assert M._is_low_priority_track(signs)
    assert M._is_low_priority_track(commentary)
    assert not M._is_low_priority_track(plain)
    assert not M._is_low_priority_track(sdh)

    ordered = sorted([other_lang, plain, sdh], key=M._track_priority)
    assert [t["number"] for t in ordered] == [7, 6, 8]  # SDH, then English, then non-Eng


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


# --------------------------------------------------------------------------- #
# Single-window head extraction
# --------------------------------------------------------------------------- #
def _single_track_mkv(cues: int = 80) -> bytes:
    return _build_mkv(
        [_track_entry(5, M._TRACK_TYPE_SUBTITLE, b"S_TEXT/UTF8", b"eng")],
        [(1000, _blocks_for(5, cues, "L"))],
    )


class _RangeClient:
    """Serves ``data[start:end+1]`` and records every Range header."""

    def __init__(self, data: bytes):
        self.data = data
        self.ranges: list[str] = []

    async def get(self, url, headers=None, follow_redirects=True):
        rng = (headers or {}).get("Range", "")
        self.ranges.append(rng)
        if rng == "bytes=0-0":
            return SimpleNamespace(
                status_code=206,
                headers={"content-range": f"bytes 0-0/{len(self.data)}"},
                content=b"\x00",
            )
        start, end = rng.replace("bytes=", "").split("-")
        return SimpleNamespace(status_code=206, headers={}, content=self.data[int(start) : int(end) + 1])


def test_choose_head_bytes_thresholds():
    assert M._choose_head_bytes(2 * 1024**3) == M._STANDARD_HEAD_BYTES
    assert M._choose_head_bytes(4 * 1024**3) == M._STANDARD_HEAD_BYTES  # <= 4 GiB
    assert M._choose_head_bytes(4 * 1024**3 + 1) == M._LARGE_HEAD_BYTES
    assert M._choose_head_bytes(None) == M._LARGE_HEAD_BYTES


@pytest.mark.asyncio
async def test_extract_uses_single_head_request(monkeypatch):
    data = _single_track_mkv()
    calls: list[tuple[int, int]] = []

    async def fake_fetch(client, url, start, size):
        calls.append((start, size))
        return data[start : start + size]

    monkeypatch.setattr(M, "_fetch", fake_fetch)
    out = await M.extract_embedded_srt(
        "http://host/movie.mkv", object(), timeout=5.0, file_size=2 * 1024**3
    )

    assert out is not None
    assert calls == [(0, M._STANDARD_HEAD_BYTES)]


@pytest.mark.asyncio
async def test_head_probe_range_headers_adaptive():
    data = _single_track_mkv()

    standard = _RangeClient(data)
    out_std = await M.extract_embedded_srt(
        "http://host/movie.mkv", standard, timeout=5.0, file_size=2 * 1024**3
    )
    assert standard.ranges == ["bytes=0-31457280"]
    assert out_std is not None

    large = _RangeClient(data)
    out_4k = await M.extract_embedded_srt(
        "http://host/movie.mkv", large, timeout=5.0, file_size=8 * 1024**3
    )
    assert large.ranges == ["bytes=0-78643200"]
    assert out_4k is not None


@pytest.mark.asyncio
async def test_head_probe_large_when_size_unknown(monkeypatch):
    data = _single_track_mkv()
    calls: list[tuple[int, int]] = []

    async def fake_fetch(client, url, start, size):
        calls.append((start, size))
        return data[start : start + size]

    monkeypatch.setattr(M, "_fetch", fake_fetch)
    # ``object()`` has no ``.get`` -> the Content-Range probe fails -> unknown.
    out = await M.extract_embedded_srt("http://host/movie.mkv", object(), timeout=5.0)

    assert out is not None
    assert calls == [(0, M._LARGE_HEAD_BYTES)]


@pytest.mark.asyncio
async def test_head_probe_uses_content_range_when_unknown(monkeypatch):
    data = _single_track_mkv()
    calls: list[tuple[int, int]] = []

    async def fake_fetch(client, url, start, size):
        calls.append((start, size))
        return data[start : start + size]

    async def fake_probe(client, url):
        return 2 * 1024**3

    monkeypatch.setattr(M, "_fetch", fake_fetch)
    monkeypatch.setattr(M, "_probe_file_size", fake_probe)
    out = await M.extract_embedded_srt("http://host/movie.mkv", object(), timeout=5.0)

    assert out is not None
    assert calls == [(0, M._STANDARD_HEAD_BYTES)]


@pytest.mark.asyncio
async def test_probe_file_size_parses_content_range():
    client = _RangeClient(b"x" * 12345)
    assert await M._probe_file_size(client, "http://host/movie.mkv") == 12345


@pytest.mark.asyncio
async def test_probe_file_size_degrades_to_none():
    assert await M._probe_file_size(object(), "http://host/movie.mkv") is None


@pytest.mark.asyncio
async def test_extract_embedded_srt_end_to_end(monkeypatch, caplog):
    data = _build_mkv(
        [_track_entry(5, M._TRACK_TYPE_SUBTITLE, b"S_TEXT/UTF8", b"eng")],
        [(1000, _blocks_for(5, 80, "L"))],
    )

    async def fake_fetch(client, url, start, size):
        return data[start : start + size]

    monkeypatch.setattr(M, "_fetch", fake_fetch)
    with caplog.at_level(logging.INFO, logger="app.services.sync.mkv_range"):
        out = await M.extract_embedded_srt("http://host/movie.mkv", object(), timeout=5.0)

    assert out is not None
    text = out.decode("utf-8")
    assert text.count("-->") == 80
    assert "L0" in text and "L79" in text
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "accepted with 80 contiguous cue(s)" in messages


@pytest.mark.asyncio
async def test_extract_demuxes_across_clusters(monkeypatch):
    data = _build_mkv(
        [_track_entry(4, M._TRACK_TYPE_SUBTITLE, b"S_TEXT/UTF8", b"eng", name=b"English SDH")],
        [
            (0, _blocks_for(4, 25, "A")),
            (10_000, _blocks_for(4, 25, "B")),
            (20_000, _blocks_for(4, 25, "C")),
        ],
    )

    async def fake_fetch(client, url, start, size):
        return data[start : start + size]

    monkeypatch.setattr(M, "_fetch", fake_fetch)
    out = await M.extract_embedded_srt("http://host/movie.mkv", object(), timeout=5.0)

    assert out is not None
    text = out.decode("utf-8")
    assert text.count("-->") == 75
    assert "A0" in text and "C24" in text


@pytest.mark.asyncio
async def test_extract_returns_none_below_cue_threshold(monkeypatch, caplog):
    data = _build_mkv(
        [_track_entry(4, M._TRACK_TYPE_SUBTITLE, b"S_TEXT/UTF8", b"eng", name=b"English")],
        [(1000, _blocks_for(4, 12, "FEW"))],
    )

    async def fake_fetch(client, url, start, size):
        return data[start : start + size]

    monkeypatch.setattr(M, "_fetch", fake_fetch)
    with caplog.at_level(logging.INFO, logger="app.services.sync.mkv_range"):
        out = await M.extract_embedded_srt("http://host/movie.mkv", object(), timeout=5.0)

    assert out is None
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "yielded 12 cue(s)" in messages
    assert "no S_TEXT/UTF8 track yielded" in messages


@pytest.mark.asyncio
async def test_extract_returns_none_without_tracks(monkeypatch, caplog):
    data = _build_mkv_without_tracks()

    async def fake_fetch(client, url, start, size):
        return data[start : start + size]

    monkeypatch.setattr(M, "_fetch", fake_fetch)
    with caplog.at_level(logging.INFO, logger="app.services.sync.mkv_range"):
        out = await M.extract_embedded_srt("http://host/movie.mkv", object(), timeout=5.0)

    assert out is None
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "no Tracks element within the head sample" in messages


@pytest.mark.asyncio
async def test_forced_track_filtered_for_full_sdh_track(monkeypatch, caplog):
    data = _build_mkv(
        [
            _track_entry(
                3, M._TRACK_TYPE_SUBTITLE, b"S_TEXT/UTF8", b"eng",
                name=b"Signs (forced)", forced=True,
            ),
            _track_entry(
                4, M._TRACK_TYPE_SUBTITLE, b"S_TEXT/UTF8", b"eng",
                name=b"English SDH", forced=False,
            ),
        ],
        [(1000, _blocks_for(3, 3, "SIGNS") + _blocks_for(4, 70, "FULL"))],
    )

    async def fake_fetch(client, url, start, size):
        return data[start : start + size]

    monkeypatch.setattr(M, "_fetch", fake_fetch)
    with caplog.at_level(logging.INFO, logger="app.services.sync.mkv_range"):
        out = await M.extract_embedded_srt("http://host/movie.mkv", object(), timeout=5.0)

    assert out is not None
    text = out.decode("utf-8")
    assert "FULL69" in text and "SIGNS0" not in text
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "candidate track(s) in priority order: Track 4" in messages


@pytest.mark.asyncio
async def test_small_track_bypassed_for_full_track(monkeypatch, caplog):
    # Track 3 is named SDH (top priority) but its sample is too small; track 4 is
    # a genuine full track and must win after the cue bar rejects track 3.
    data = _build_mkv(
        [
            _track_entry(
                3, M._TRACK_TYPE_SUBTITLE, b"S_TEXT/UTF8", b"eng",
                name=b"English SDH", forced=False,
            ),
            _track_entry(
                4, M._TRACK_TYPE_SUBTITLE, b"S_TEXT/UTF8", b"eng",
                name=b"English", forced=False,
            ),
        ],
        [(1000, _blocks_for(3, 10, "SMALL") + _blocks_for(4, 70, "FULL"))],
    )

    async def fake_fetch(client, url, start, size):
        return data[start : start + size]

    monkeypatch.setattr(M, "_fetch", fake_fetch)
    with caplog.at_level(logging.INFO, logger="app.services.sync.mkv_range"):
        out = await M.extract_embedded_srt("http://host/movie.mkv", object(), timeout=5.0)

    assert out is not None
    text = out.decode("utf-8")
    assert "FULL69" in text and "SMALL0" not in text
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "trying next candidate" in messages
