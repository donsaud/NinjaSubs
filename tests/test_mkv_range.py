"""Unit tests for the range-based Matroska subtitle extractor."""

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
    # 5 clusters x 40 cues = 200 cues, Cues index at the tail.
    data = _build_tail_cues_mkv(
        [_track_entry(5, M._TRACK_TYPE_SUBTITLE, b"S_TEXT/UTF8", b"eng")],
        [(c * 10_000, b"".join(_blocks_for(5, 40, f"C{c}_"))) for c in range(5)],
    )

    class _Client:
        pass

    async def fake_fetch(client, url, start, size):
        return data[start : start + size]

    monkeypatch.setattr(M, "_fetch", fake_fetch)
    out = await M.extract_embedded_srt("http://host/movie.mkv", _Client(), timeout=5.0)
    assert out is not None
    text = out.decode("utf-8")
    assert "C0_0" in text and "C4_39" in text
    # Full timeline: the last cluster's cue is present.
    assert "00:00:43,900 --> 00:00:43,990" in text


def test_srt_timestamp():
    assert M._srt_timestamp(1000) == "00:00:01,000"
    assert M._srt_timestamp(3_661_500) == "01:01:01,500"
    assert M._srt_timestamp(-5) == "00:00:00,000"


# --------------------------------------------------------------------------- #
# Track discovery diagnostics
# --------------------------------------------------------------------------- #
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


def _build_mkv_with_tracks(tracks: list[bytes]) -> bytes:
    """Minimal MKV whose Tracks element is inlined in the header (no cues)."""
    tracks_el = _el(M._ID_TRACKS, b"".join(tracks))
    info = _el(M._ID_INFO, _el(M._ID_TIMESTAMP_SCALE, _u8(1_000_000)))
    segment = info + tracks_el
    return _el(0x1A45DFA3, b"") + _el(M._ID_SEGMENT, segment)


def _build_mkv_custom(track_entries: list[bytes], blocks: list[bytes]) -> bytes:
    """Build a full MKV (SeekHead/Info/Tracks/Cues/Cluster) with custom tracks."""
    tracks = _el(M._ID_TRACKS, b"".join(track_entries))
    info = _el(M._ID_INFO, _el(M._ID_TIMESTAMP_SCALE, _u8(1_000_000)))
    cluster = _el(
        M._ID_CLUSTER, _el(M._ID_TIMESTAMP, _u8(1000)) + b"".join(blocks)
    )

    def _seek_head(tracks_off: int, cues_off: int) -> bytes:
        def seek(eid: int, pos: int) -> bytes:
            return _el(
                M._ID_SEEK,
                _el(M._ID_SEEK_ID, _eid(eid)) + _el(M._ID_SEEK_POSITION, _u8(pos)),
            )

        return _el(M._ID_SEEK_HEAD, seek(M._ID_TRACKS, tracks_off) + seek(M._ID_CUES, cues_off))

    def cue(cluster_pos: int) -> bytes:
        return _el(
            M._ID_CUE_POINT,
            _el(M._ID_CUE_TIME, _u8(1000))
            + _el(
                M._ID_CUE_TRACK_POSITIONS,
                _el(0xF7, _u8(1)) + _el(M._ID_CUE_CLUSTER_POSITION, _u8(cluster_pos)),
            ),
        )

    sh = _seek_head(0, 0)
    tracks_off = len(sh) + len(info)
    cues_placeholder = _el(M._ID_CUES, cue(0))
    cues_off = tracks_off + len(tracks)
    cluster_off = cues_off + len(cues_placeholder)
    cues = _el(M._ID_CUES, cue(cluster_off))
    sh = _seek_head(tracks_off, cues_off)
    segment = sh + info + tracks + cues + cluster
    return _el(0x1A45DFA3, b"") + _el(M._ID_SEGMENT, segment)


def _build_tail_cues_mkv(
    track_entries: list[bytes],
    clusters: list[tuple[int, bytes]],
    *,
    cue_track: int = 1,
) -> bytes:
    """MKV with one cluster per ``(cue_time_ms, payload)`` and Cues at the tail.

    All payloads use fixed 8-byte field widths, so the Cues element size (and
    therefore the recorded cluster offsets) is independent of the position
    values — no iterative sizing is needed.
    """
    tracks = _el(M._ID_TRACKS, b"".join(track_entries))
    info = _el(M._ID_INFO, _el(M._ID_TIMESTAMP_SCALE, _u8(1_000_000)))

    def seek(eid: int, pos: int) -> bytes:
        return _el(
            M._ID_SEEK, _el(M._ID_SEEK_ID, _eid(eid)) + _el(M._ID_SEEK_POSITION, _u8(pos))
        )

    sh = _el(M._ID_SEEK_HEAD, seek(M._ID_TRACKS, 0) + seek(M._ID_CUES, 0))
    tracks_off = len(sh) + len(info)

    cluster_elems: list[bytes] = []
    cluster_offsets: list[int] = []
    off = tracks_off + len(tracks)
    for cue_time, payload in clusters:
        elem = _el(M._ID_CLUSTER, _el(M._ID_TIMESTAMP, _u8(cue_time)) + payload)
        cluster_elems.append(elem)
        cluster_offsets.append(off)
        off += len(elem)

    def cuepoint(time_ms: int, cluster_pos: int) -> bytes:
        return _el(
            M._ID_CUE_POINT,
            _el(M._ID_CUE_TIME, _u8(time_ms))
            + _el(
                M._ID_CUE_TRACK_POSITIONS,
                _el(M._ID_CUE_TRACK, _u8(cue_track))
                + _el(M._ID_CUE_CLUSTER_POSITION, _u8(cluster_pos)),
            ),
        )

    cues = _el(
        M._ID_CUES,
        b"".join(cuepoint(t, p) for (t, _), p in zip(clusters, cluster_offsets, strict=True)),
    )
    cues_off = tracks_off + len(tracks) + sum(len(elem) for elem in cluster_elems)
    sh = _el(M._ID_SEEK_HEAD, seek(M._ID_TRACKS, tracks_off) + seek(M._ID_CUES, cues_off))
    segment = sh + info + tracks + b"".join(cluster_elems) + cues
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
    data = _build_tail_cues_mkv(
        [_track_entry(5, M._TRACK_TYPE_SUBTITLE, b"S_TEXT/UTF8", b"eng")],
        [(c * 10_000, b"".join(_blocks_for(5, 40, f"C{c}_"))) for c in range(5)],
    )

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
    assert "located Cues index" in messages


# --------------------------------------------------------------------------- #
# Track flags / names + smart track selection
# --------------------------------------------------------------------------- #
def _parse_tracks_from(data: bytes) -> list[dict]:
    seg_start, seg_end = M.find_segment(data)
    tracks_range = next(
        (ds, de) for eid, ds, de in M._children(data, seg_start, seg_end) if eid == M._ID_TRACKS
    )
    return M.parse_all_tracks(data, *tracks_range)


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
    parsed = _parse_tracks_from(data)
    assert parsed[0]["name"] == "Signs & Songs (forced)"
    assert parsed[0]["forced"] is True
    assert parsed[1]["name"] == "English SDH"
    assert parsed[1]["forced"] is False
    # No FlagForced element -> defaults to False.
    plain = _parse_tracks_from(
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


def test_is_incomplete_embedded_thresholds():
    assert M._is_incomplete_embedded(149, 20_000, 0) is True
    assert M._is_incomplete_embedded(150, 20_000, 0) is False
    # Long video + tiny payload is incomplete even with many cues.
    assert M._is_incomplete_embedded(200, 9_000, 20 * 60 * 1000) is True
    assert M._is_incomplete_embedded(200, 20_000, 20 * 60 * 1000) is False
    # Short video: the byte floor does not apply.
    assert M._is_incomplete_embedded(200, 5_000, 5 * 60 * 1000) is False


def _blocks_for(track: int, count: int, prefix: str) -> list[bytes]:
    # 100 ms spacing keeps the int16 relative timestamp within range for 300+ cues.
    return [_block(track, i * 100, f"{prefix}{i}".encode(), 90) for i in range(count)]


@pytest.mark.asyncio
async def test_forced_track_filtered_for_full_sdh_track(monkeypatch, caplog):
    data = _build_mkv_custom(
        [
            _track_entry(
                3,
                M._TRACK_TYPE_SUBTITLE,
                b"S_TEXT/UTF8",
                b"eng",
                name=b"Signs (forced)",
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
        ],
        _blocks_for(3, 3, "SIGNS") + _blocks_for(4, 200, "FULL"),
    )

    async def fake_fetch(client, url, start, size):
        return data[start : start + size]

    monkeypatch.setattr(M, "_fetch", fake_fetch)
    with caplog.at_level(logging.INFO, logger="app.services.sync.mkv_range"):
        out = await M.extract_embedded_srt("http://host/movie.mkv", object(), timeout=5.0)

    assert out is not None
    text = out.decode("utf-8")
    assert "FULL199" in text and "SIGNS0" not in text
    # The forced track never becomes a candidate.
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "candidate track(s) in priority order: Track 4" in messages


@pytest.mark.asyncio
async def test_small_track_bypassed_for_full_track(monkeypatch, caplog):
    # Track 3 is named SDH (top priority) but its sample is too small; track 4 is
    # a genuine full track and must win after the threshold rejects track 3.
    data = _build_mkv_custom(
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
        _blocks_for(3, 10, "SMALL") + _blocks_for(4, 200, "FULL"),
    )

    async def fake_fetch(client, url, start, size):
        return data[start : start + size]

    monkeypatch.setattr(M, "_fetch", fake_fetch)
    with caplog.at_level(logging.INFO, logger="app.services.sync.mkv_range"):
        out = await M.extract_embedded_srt("http://host/movie.mkv", object(), timeout=5.0)

    assert out is not None
    text = out.decode("utf-8")
    assert "FULL199" in text and "SMALL0" not in text
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "trying next candidate" in messages


# --------------------------------------------------------------------------- #
# Tail Cues parsing + full extraction (no truncated references)
# --------------------------------------------------------------------------- #
def test_parse_cues_targets_specific_track():
    def cuepoint(time_ms: int, video_pos: int, sub_pos: int) -> bytes:
        video = _el(
            M._ID_CUE_TRACK_POSITIONS,
            _el(M._ID_CUE_TRACK, _u8(1)) + _el(M._ID_CUE_CLUSTER_POSITION, _u8(video_pos)),
        )
        sub = _el(
            M._ID_CUE_TRACK_POSITIONS,
            _el(M._ID_CUE_TRACK, _u8(4)) + _el(M._ID_CUE_CLUSTER_POSITION, _u8(sub_pos)),
        )
        return _el(M._ID_CUE_POINT, _el(M._ID_CUE_TIME, _u8(time_ms)) + video + sub)

    blob = _el(M._ID_CUES, cuepoint(1000, 111, 222) + cuepoint(2000, 333, 444))
    contents = M._element_contents(blob, M._ID_CUES)
    assert contents is not None
    every = M.parse_cues(blob, contents[0], contents[1])
    assert (1000, 111) in every and (1000, 222) in every
    # Entries for the target track are preferred when present.
    assert M.parse_cues(blob, contents[0], contents[1], target_track=4) == [(1000, 222), (2000, 444)]
    # A track with no entries falls back to every CuePoint (video-only index).
    assert M.parse_cues(blob, contents[0], contents[1], target_track=9) == every


@pytest.mark.asyncio
async def test_extract_full_track_across_clusters_from_tail_cues(monkeypatch, caplog):
    # 160 clusters, one subtitle cue each -> complete timeline spanning the file.
    data = _build_tail_cues_mkv(
        [
            _track_entry(
                4, M._TRACK_TYPE_SUBTITLE, b"S_TEXT/UTF8", b"eng",
                name=b"English SDH", forced=False,
            )
        ],
        [(i * 1000, b"".join(_blocks_for(4, 1, f"CUE{i}_"))) for i in range(160)],
    )

    async def fake_fetch(client, url, start, size):
        return data[start : start + size]

    monkeypatch.setattr(M, "_fetch", fake_fetch)
    with caplog.at_level(logging.INFO, logger="app.services.sync.mkv_range"):
        out = await M.extract_embedded_srt("http://host/movie.mkv", object(), timeout=5.0)

    assert out is not None
    text = out.decode("utf-8")
    assert "CUE0_0" in text and "CUE159_0" in text
    assert text.count("-->") == 160
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "located Cues index" in messages
    assert "complete with 160 cue(s)" in messages


@pytest.mark.asyncio
async def test_extract_returns_none_when_cues_missing(monkeypatch, caplog):
    # Tracks inline but no SeekHead/Cues element at all.
    data = _build_mkv_with_tracks(
        [_track_entry(4, M._TRACK_TYPE_SUBTITLE, b"S_TEXT/UTF8", b"eng", name=b"English")]
    )

    async def fake_fetch(client, url, start, size):
        return data[start : start + size]

    monkeypatch.setattr(M, "_fetch", fake_fetch)
    with caplog.at_level(logging.INFO, logger="app.services.sync.mkv_range"):
        out = await M.extract_embedded_srt("http://host/movie.mkv", object(), timeout=5.0)

    assert out is None
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "no Cues index" in messages


@pytest.mark.asyncio
async def test_extract_returns_none_when_track_incomplete(monkeypatch, caplog):
    # Only 12 cues: below the 150-cue bar -> no partial/truncated reference.
    data = _build_tail_cues_mkv(
        [_track_entry(4, M._TRACK_TYPE_SUBTITLE, b"S_TEXT/UTF8", b"eng", name=b"English")],
        [(i * 1000, b"".join(_blocks_for(4, 1, f"CUE{i}_"))) for i in range(12)],
    )

    async def fake_fetch(client, url, start, size):
        return data[start : start + size]

    monkeypatch.setattr(M, "_fetch", fake_fetch)
    with caplog.at_level(logging.INFO, logger="app.services.sync.mkv_range"):
        out = await M.extract_embedded_srt("http://host/movie.mkv", object(), timeout=5.0)

    assert out is None
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "incomplete (12 cue(s)" in messages
    assert "no complete S_TEXT/UTF8 track" in messages


class _SizeClient:
    def __init__(self, content_range: str):
        self._content_range = content_range

    async def get(self, url, headers=None, follow_redirects=True):
        return SimpleNamespace(headers={"content-range": self._content_range}, status_code=206)


@pytest.mark.asyncio
async def test_probe_total_size_parses_content_range():
    client = _SizeClient("bytes 0-0/19445753080")
    assert await M._probe_total_size(client, "http://host/movie.mkv") == 19445753080


@pytest.mark.asyncio
async def test_probe_total_size_degrades_to_none():
    class _Boom:
        async def get(self, url, headers=None, follow_redirects=True):
            raise RuntimeError("no headers")

    assert await M._probe_total_size(_Boom(), "http://host/movie.mkv") is None
