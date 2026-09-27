"""Range-based Matroska (MKV) subtitle extraction.

Remote debrid/usenet streams serve HTTP Range requests quickly but feed
``ffmpeg`` sequentially at a crawl (subtitles are interleaved with video, so
ffmpeg must read tens of MB of video to reach the first cue). This module
instead parses the EBML container over a handful of small Range requests:

* read the header + ``SeekHead`` to locate ``Tracks`` and ``Cues``;
* read ``Tracks`` to find the desired text-subtitle track number;
* read ``Cues`` to build a keyframe-cluster index;
* fetch a sample of clusters spread across the timeline and decode the
  subtitle blocks they contain into a partial SRT reference.

Only text codecs (``S_TEXT/UTF8`` / ``S_TEXT/ASS`` / ``S_TEXT/SSA``) are
handled; image-based tracks (PGS/VobSub) are reported unsupported.
"""

from __future__ import annotations

import logging
import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import httpx

logger = logging.getLogger(__name__)

# EBML element IDs we care about.
_ID_SEGMENT = 0x18538067
_ID_SEEK_HEAD = 0x114D9B74
_ID_SEEK = 0x4DBB
_ID_SEEK_ID = 0x53AB
_ID_SEEK_POSITION = 0x53AC
_ID_INFO = 0x1549A966
_ID_TIMESTAMP_SCALE = 0x2AD7B1
_ID_DURATION = 0x4489
_ID_TRACKS = 0x1654AE6B
_ID_TRACK_ENTRY = 0xAE
_ID_TRACK_NUMBER = 0xD7
_ID_TRACK_TYPE = 0x83
_ID_CODEC_ID = 0x86
_ID_LANGUAGE = 0x22B59C
_ID_CUES = 0x1C53BB6B
_ID_CUE_POINT = 0xBB
_ID_CUE_TIME = 0xB3
_ID_CUE_TRACK_POSITIONS = 0xB7
_ID_CUE_CLUSTER_POSITION = 0xF1
_ID_CLUSTER = 0x1F43B675
_ID_TIMESTAMP = 0xE7
_ID_SIMPLE_BLOCK = 0xA3
_ID_BLOCK_GROUP = 0xA0
_ID_BLOCK = 0xA1
_ID_BLOCK_DURATION = 0x9B

_TRACK_TYPE_SUBTITLE = 0x11
_TEXT_CODECS = ("S_TEXT/UTF8", "S_TEXT/ASS", "S_TEXT/SSA", "S_TEXT/WEBVTT")
_ENGLISH = frozenset({"eng", "en", "english"})


class MKVRangeError(Exception):
    """Raised when the remote container cannot be parsed via ranges."""


# --------------------------------------------------------------------------- #
# EBML primitives
# --------------------------------------------------------------------------- #
def _read_vint(buf: bytes, pos: int, *, keep_marker: bool) -> tuple[int, int]:
    if pos >= len(buf):
        return 0, 0
    first = buf[pos]
    if first == 0:
        return 0, 0
    length = 1
    mask = 0x80
    while not (first & mask):
        mask >>= 1
        length += 1
        if length > 8:
            return 0, 0
    value = first if keep_marker else (first & (mask - 1))
    for i in range(1, length):
        if pos + i >= len(buf):
            return 0, 0
        value = (value << 8) | buf[pos + i]
    return value, length


def _children(buf: bytes, start: int, end: int):
    """Yield ``(element_id, data_start, data_end)`` for children in ``buf``.

    Positions are absolute file offsets; ``buf`` is treated as starting at
    ``0`` (callers fetch a chunk and pass its base offset by slicing).
    """
    pos = start
    while pos < end:
        element_start = pos
        eid, idlen = _read_vint(buf, pos, keep_marker=True)
        if not idlen:
            break
        pos += idlen
        size, sizelen = _read_vint(buf, pos, keep_marker=False)
        if not sizelen:
            break
        pos += sizelen
        dstart = pos
        dend = pos + size
        if dend > end:
            dend = end
        yield eid, dstart, dend
        if dend <= element_start:
            break  # no forward progress: corrupt stream
        pos = dend


def _read_uint(buf: bytes, start: int, end: int) -> int:
    value = 0
    for i in range(start, min(end, len(buf))):
        value = (value << 8) | buf[i]
    return value


def _read_float(buf: bytes, start: int, end: int) -> float:
    raw = buf[start:end]
    if len(raw) == 4:
        return struct.unpack(">f", raw)[0]
    if len(raw) == 8:
        return struct.unpack(">d", raw)[0]
    return 0.0


# --------------------------------------------------------------------------- #
# Container parsing
# --------------------------------------------------------------------------- #
def find_segment(buf: bytes) -> tuple[int, int]:
    """Return ``(segment_data_start, segment_data_end)`` for a Segment in ``buf``."""
    for eid, dstart, dend in _children(buf, 0, len(buf)):
        if eid == _ID_SEGMENT:
            return dstart, dend
    raise MKVRangeError("no Segment element in the MKV header")


def parse_seek_head(buf: bytes, start: int, end: int) -> dict[int, int]:
    """Return ``{element_id: absolute_position}`` from a SeekHead."""
    positions: dict[int, int] = {}
    for eid, ds, de in _children(buf, start, end):
        if eid != _ID_SEEK:
            continue
        seek_id = seek_pos = None
        for cid, cs, ce in _children(buf, ds, de):
            if cid == _ID_SEEK_ID:
                seek_id = _read_uint(buf, cs, ce)
            elif cid == _ID_SEEK_POSITION:
                seek_pos = _read_uint(buf, cs, ce)
        if seek_id is not None and seek_pos is not None:
            positions[seek_id] = seek_pos
    return positions


def parse_info(buf: bytes, start: int, end: int) -> tuple[int, float]:
    """Return ``(timestamp_scale_ns, duration_ms)`` (0 duration if absent)."""
    scale = 1_000_000
    duration_ticks = 0.0
    for eid, ds, de in _children(buf, start, end):
        if eid == _ID_TIMESTAMP_SCALE:
            scale = _read_uint(buf, ds, de) or scale
        elif eid == _ID_DURATION:
            duration_ticks = _read_float(buf, ds, de)
    return scale, duration_ticks * scale / 1_000_000.0


def parse_tracks(buf: bytes, start: int, end: int) -> list[dict]:
    """Return subtitle TrackEntry dicts: number / codec / language."""
    tracks: list[dict] = []
    for eid, ds, de in _children(buf, start, end):
        if eid != _ID_TRACK_ENTRY:
            continue
        number = track_type = 0
        codec = ""
        language = ""
        for cid, cs, ce in _children(buf, ds, de):
            if cid == _ID_TRACK_NUMBER:
                number = _read_uint(buf, cs, ce)
            elif cid == _ID_TRACK_TYPE:
                track_type = _read_uint(buf, cs, ce)
            elif cid == _ID_CODEC_ID:
                codec = buf[cs:ce].decode("ascii", "replace").strip("\x00")
            elif cid == _ID_LANGUAGE:
                language = buf[cs:ce].decode("ascii", "replace").strip("\x00")
        if track_type == _TRACK_TYPE_SUBTITLE and codec in _TEXT_CODECS:
            tracks.append({"number": number, "codec": codec, "language": language.lower()})
    return tracks


def parse_cues(buf: bytes, start: int, end: int) -> list[tuple[int, int]]:
    """Return ``[(time_ms, cluster_relative_position), ...]`` from Cues."""
    cues: list[tuple[int, int]] = []
    for eid, ds, de in _children(buf, start, end):
        if eid != _ID_CUE_POINT:
            continue
        time_ms = None
        cluster_pos = None
        for cid, cs, ce in _children(buf, ds, de):
            if cid == _ID_CUE_TIME:
                time_ms = _read_uint(buf, cs, ce)
            elif cid == _ID_CUE_TRACK_POSITIONS:
                for pid, ps, pe in _children(buf, cs, ce):
                    if pid == _ID_CUE_CLUSTER_POSITION:
                        cluster_pos = _read_uint(buf, ps, pe)
        if time_ms is not None and cluster_pos is not None:
            cues.append((time_ms, cluster_pos))
    return cues


def parse_cluster(buf: bytes, start: int, end: int, track: int) -> list[tuple[int, int, str]]:
    """Return ``[(start_ms, duration_ms, text), ...]`` for one subtitle track.

    ``buf`` must begin at a Cluster element; its contents are decoded.
    """
    results: list[tuple[int, int, str]] = []
    for eid, ds, de in _children(buf, start, end):
        if eid == _ID_CLUSTER:
            results.extend(_parse_cluster_contents(buf, ds, de, track))
    return results


def _parse_cluster_contents(
    buf: bytes, start: int, end: int, track: int
) -> list[tuple[int, int, str]]:
    cluster_time = 0
    results: list[tuple[int, int, str]] = []
    for eid, ds, de in _children(buf, start, end):
        if eid == _ID_TIMESTAMP:
            cluster_time = _read_uint(buf, ds, de)
        elif eid == _ID_SIMPLE_BLOCK:
            decoded = _decode_block(buf, ds, de, track)
            if decoded is not None:
                rel, text = decoded
                results.append((cluster_time + rel, 0, text))
        elif eid == _ID_BLOCK_GROUP:
            block_rel = None
            block_text = None
            duration = 0
            for cid, cs, ce in _children(buf, ds, de):
                if cid == _ID_BLOCK:
                    decoded = _decode_block(buf, cs, ce, track)
                    if decoded is not None:
                        block_rel, block_text = decoded
                elif cid == _ID_BLOCK_DURATION:
                    duration = _read_uint(buf, cs, ce)
            if block_text is not None and block_rel is not None:
                results.append((cluster_time + block_rel, duration, block_text))
    return results


def _element_contents(buf: bytes, element_id: int) -> tuple[int, int] | None:
    """Return the ``(start, end)`` contents of the first element with ``id``."""
    for eid, ds, de in _children(buf, 0, len(buf)):
        if eid == element_id:
            return ds, de
    return None


def _decode_block(buf: bytes, start: int, end: int, track: int) -> tuple[int, str] | None:
    """Decode a SimpleBlock/Block for ``track`` -> ``(relative_ms, text)``."""
    value, length = _read_vint(buf, start, keep_marker=False)
    if not length:
        return None
    if value != track:
        return None
    pos = start + length
    if pos + 3 > end:
        return None
    rel = struct.unpack(">h", buf[pos : pos + 2])[0]
    pos += 3  # int16 timestamp + flags byte
    text = buf[pos:end].decode("utf-8", "replace").strip("\x00")
    return rel, text


# --------------------------------------------------------------------------- #
# Range fetch + high level extraction
# --------------------------------------------------------------------------- #
async def _fetch(client: httpx.AsyncClient, url: str, start: int, size: int) -> bytes:
    resp = await client.get(
        url, headers={"Range": f"bytes={start}-{start + size - 1}"}, follow_redirects=True
    )
    if resp.status_code not in (200, 206):
        raise MKVRangeError(f"range fetch failed with HTTP {resp.status_code}")
    return resp.content or b""


def _srt_timestamp(ms: int) -> str:
    ms = max(0, ms)
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


async def extract_embedded_srt(
    stream_url: str,
    client: httpx.AsyncClient,
    *,
    want_language: str = "eng",
    windows: int = 2,
    window_bytes: int = 40 * 1024 * 1024,
    target_cues: int = 8,
    timeout: float = 8.5,
) -> bytes | None:
    """Extract a partial text-subtitle reference from a remote MKV via ranges.

    Returns raw SRT bytes (partial, sampled) or ``None`` when the container is
    unsupported/absent. Never raises.
    """
    import asyncio

    async def _run() -> bytes | None:
        header = await _fetch(client, stream_url, 0, 1_048_576)
        seg_start, seg_end = find_segment(header)
        base = seg_start
        window_end = min(seg_end, len(header)) if seg_end <= len(header) else len(header)

        seeks: dict[int, int] = {}
        info_scale, duration_ms = 1_000_000, 0.0
        tracks: list[dict] = []
        for eid, ds, de in _children(header, base, window_end):
            if eid == _ID_SEEK_HEAD:
                seeks.update(parse_seek_head(header, ds, de))
            elif eid == _ID_INFO:
                info_scale, duration_ms = parse_info(header, ds, de)
            elif eid == _ID_TRACKS:
                tracks = parse_tracks(header, ds, de)

        async def _fetch_at(rel_pos: int, size: int) -> bytes:
            return await _fetch(client, stream_url, base + rel_pos, size)

        if not tracks and _ID_TRACKS in seeks:
            blob = await _fetch_at(seeks[_ID_TRACKS], 262_144)
            contents = _element_contents(blob, _ID_TRACKS)
            if contents:
                tracks = parse_tracks(blob, contents[0], contents[1])
        if not tracks:
            raise MKVRangeError("no text subtitle track found in Tracks")

        track = next(
            (t for t in tracks if t["language"] in _ENGLISH),
            tracks[0],
        )

        if _ID_CUES not in seeks:
            raise MKVRangeError("no Cues index (cannot range-seek clusters)")
        cue_blob = await _fetch_at(seeks[_ID_CUES], 1_048_576)
        cue_contents = _element_contents(cue_blob, _ID_CUES)
        if cue_contents is None:
            raise MKVRangeError("Cues element not found at its SeekHead position")
        cues = parse_cues(cue_blob, cue_contents[0], cue_contents[1])
        if not cues:
            raise MKVRangeError("empty Cues index")

        cues.sort()
        # Subtitle blocks are interleaved with video, so we cannot cheaply seek
        # to individual cues. A single large contiguous range is fast on the
        # proxy; fetch several evenly-spread windows and decode every cluster.
        n_windows = max(1, min(windows, 16))
        fractions = [(i + 1) / (n_windows + 1) for i in range(n_windows)]
        collected: list[tuple[int, int, str]] = []
        for frac in fractions:
            if len(collected) >= target_cues:
                break
            index = min(len(cues) - 1, int(len(cues) * frac))
            cluster_rel = cues[index][1]
            try:
                blob = await _fetch_at(cluster_rel, window_bytes)
            except MKVRangeError:
                continue
            collected.extend(parse_cluster(blob, 0, len(blob), track["number"]))

        if not collected:
            raise MKVRangeError("no subtitle blocks decoded from sampled windows")

        collected.sort(key=lambda item: item[0])
        # De-duplicate identical start times (a cluster can be sampled twice).
        seen: set[int] = set()
        blocks: list[str] = []
        for start_ms, duration_ms, text in collected:
            if start_ms in seen or not text.strip():
                continue
            seen.add(start_ms)
            end_ms = start_ms + (duration_ms or 2500)
            blocks.append(f"{len(blocks) + 1}\n{_srt_timestamp(start_ms)} --> {_srt_timestamp(end_ms)}\n{text}")
        if len(blocks) < 5:
            raise MKVRangeError(f"only {len(blocks)} subtitle cue(s) decoded")
        return ("\n\n".join(blocks) + "\n").encode("utf-8")

    try:
        return await asyncio.wait_for(_run(), timeout)
    except Exception as exc:  # noqa: BLE001 - degrade to the ffmpeg tier
        logger.info("[reference] range MKV extraction unavailable: %s", exc)
        return None
