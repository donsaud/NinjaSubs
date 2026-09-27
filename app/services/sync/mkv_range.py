"""Range-based Matroska (MKV) subtitle extraction.

Remote debrid/usenet streams serve HTTP Range requests quickly but feed
``ffmpeg`` sequentially at a crawl (subtitles are interleaved with video, so
ffmpeg must read tens of MB of video to reach the first cue). This module
instead parses the EBML container over a handful of small Range requests:

* read the header + ``SeekHead`` to locate ``Tracks`` and ``Cues``;
* read ``Tracks`` to find the desired text-subtitle track number;
* read ``Cues`` (a targeted range, usually at the tail) to build the fuller
  keyframe-cluster index;
* walk every cue cluster and decode the target track's blocks, reconstructing
  the complete SRT timeline (never a truncated/partial reference).

Only plain-text ``S_TEXT/UTF8`` subtitle tracks are extractable here;
image-based (PGS/VobSub) and styled (ASS/SSA/WebVTT) tracks are reported but
not used. Forced/signs/songs/commentary tracks are deprioritised, and a sample
that looks forced/partial falls back to the next candidate track.
"""

from __future__ import annotations

import logging
import re
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
_ID_NAME = 0x536E
_ID_FLAG_FORCED = 0x55AA
_ID_CUES = 0x1C53BB6B
_ID_CUE_POINT = 0xBB
_ID_CUE_TIME = 0xB3
_ID_CUE_TRACK_POSITIONS = 0xB7
_ID_CUE_TRACK = 0xF7
_ID_CUE_CLUSTER_POSITION = 0xF1
_ID_CLUSTER = 0x1F43B675
_ID_TIMESTAMP = 0xE7
_ID_SIMPLE_BLOCK = 0xA3
_ID_BLOCK_GROUP = 0xA0
_ID_BLOCK = 0xA1
_ID_BLOCK_DURATION = 0x9B

_TRACK_TYPE_VIDEO = 0x01
_TRACK_TYPE_AUDIO = 0x02
_TRACK_TYPE_SUBTITLE = 0x11
_TEXT_CODECS = ("S_TEXT/UTF8", "S_TEXT/ASS", "S_TEXT/SSA", "S_TEXT/WEBVTT")
# The range parser turns raw Matroska text blocks straight into SRT, so only
# S_TEXT/UTF8 is guaranteed to be plain, timestamp-able text. ASS/SSA dialogue
# blocks and image codecs (PGS/VobSub) are detected/logged but never extracted
# here — they resume via the ffmpeg or external tiers instead.
_RANGE_SUPPORTED_TEXT_CODECS = frozenset({"S_TEXT/UTF8"})
_ENGLISH = frozenset({"eng", "en", "english"})

# Track-selection heuristics. A forced/signs/songs/commentary track holds only a
# handful of cues and is useless as a sync reference, so it is deprioritised (or
# dropped); a genuine English (optionally SDH/full) track is preferred.
_LOW_PRIORITY_NAME_MARKERS = ("forced", "stripped", "signs", "songs", "commentary")
_PREFERRED_NAME_MARKERS = ("sdh", "full")

# Completeness bar for an extracted embedded track. A full feature subtitle has
# hundreds of cues; anything under these bounds looks forced/partial and the
# extractor tries the next S_TEXT/UTF8 candidate (or returns None).
_MIN_EMBEDDED_CUES = 150
_MIN_EMBEDDED_BYTES = 10 * 1024
_LONG_VIDEO_MS = 15 * 60 * 1000

# Cues index lives near the tail of most muxes, so read it with one small
# targeted range request. Full extraction then walks every cue cluster.
_TAIL_CUES_BYTES = 512 * 1024
_CUE_CLUSTER_BYTES = 256 * 1024
_MAX_EXTRACT_BYTES = 256 * 1024 * 1024


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


def parse_all_tracks(buf: bytes, start: int, end: int) -> list[dict]:
    """Return every TrackEntry dict: number / type / codec / language / name / forced.

    Unlike :func:`parse_tracks`, this keeps video, audio, and unsupported
    subtitle codecs so callers can log a complete picture of the container and
    apply track-selection heuristics (forced/signs vs. full/SDH).
    """
    tracks: list[dict] = []
    for eid, ds, de in _children(buf, start, end):
        if eid != _ID_TRACK_ENTRY:
            continue
        number = track_type = 0
        codec = ""
        language = ""
        name = ""
        forced = False
        for cid, cs, ce in _children(buf, ds, de):
            if cid == _ID_TRACK_NUMBER:
                number = _read_uint(buf, cs, ce)
            elif cid == _ID_TRACK_TYPE:
                track_type = _read_uint(buf, cs, ce)
            elif cid == _ID_CODEC_ID:
                codec = buf[cs:ce].decode("ascii", "replace").strip("\x00")
            elif cid == _ID_LANGUAGE:
                language = buf[cs:ce].decode("ascii", "replace").strip("\x00")
            elif cid == _ID_NAME:
                name = buf[cs:ce].decode("utf-8", "replace").strip("\x00").strip()
            elif cid == _ID_FLAG_FORCED:
                forced = _read_uint(buf, cs, ce) != 0
        tracks.append(
            {
                "number": number,
                "type": track_type,
                "codec": codec,
                "language": language.lower(),
                "name": name,
                "forced": forced,
            }
        )
    return tracks


def parse_tracks(buf: bytes, start: int, end: int) -> list[dict]:
    """Return supported text-subtitle TrackEntry dicts: number / codec / language."""
    return [
        {"number": track["number"], "codec": track["codec"], "language": track["language"]}
        for track in parse_all_tracks(buf, start, end)
        if track["type"] == _TRACK_TYPE_SUBTITLE and track["codec"] in _TEXT_CODECS
    ]


def _format_subtitle_tracks(tracks: list[dict]) -> str:
    """Render detected subtitle tracks for the diagnostic log line."""
    return "; ".join(
        f"Track {track['number']}: CodecID='{track['codec']}', Lang='{track['language']}'"
        for track in tracks
    )


def _is_low_priority_track(track: dict) -> bool:
    """True for forced/signs/songs/commentary tracks, useless as a reference."""
    if track.get("forced"):
        return True
    name = (track.get("name") or "").lower()
    return any(marker in name for marker in _LOW_PRIORITY_NAME_MARKERS)


def _track_priority(track: dict) -> tuple[int, int, int, int]:
    """Sort key: prefer non-forced English tracks, then ones named sdh/full."""
    name = (track.get("name") or "").lower()
    forced_rank = 1 if track.get("forced") else 0
    lang_rank = 0 if track.get("language") in _ENGLISH else 1
    name_rank = 0 if any(marker in name for marker in _PREFERRED_NAME_MARKERS) else 1
    return (forced_rank, lang_rank, name_rank, int(track.get("number") or 0))


def _is_incomplete_embedded(
    cue_count: int,
    text_bytes: int,
    duration_ms: float,
    *,
    min_cues: int = _MIN_EMBEDDED_CUES,
    min_bytes: int = _MIN_EMBEDDED_BYTES,
) -> bool:
    """True when a sampled embedded track looks forced/partial, not full.

    A complete subtitle track has hundreds of cues; a signs/songs track has a
    handful. Very small payloads on a feature-length (``> 15 min``) video are
    likewise treated as incomplete so the next candidate track can be tried.
    """
    if cue_count < min_cues:
        return True
    if duration_ms >= _LONG_VIDEO_MS and text_bytes < min_bytes:
        return True
    return False


def parse_cues(
    buf: bytes, start: int, end: int, *, target_track: int | None = None
) -> list[tuple[int, int]]:
    """Return ``[(time_ms, cluster_relative_position), ...]`` from Cues.

    Each ``CuePoint`` may carry several ``CueTrackPositions`` (one per indexed
    track). When ``target_track`` is given and the index actually contains
    entries for it, only those cluster positions are returned; otherwise every
    ``CuePoint`` is used, since most muxes index only the video track.
    """
    entries: list[tuple[int, int, int]] = []  # (time_ms, cluster_pos, track)
    for eid, ds, de in _children(buf, start, end):
        if eid != _ID_CUE_POINT:
            continue
        time_ms = None
        positions: list[tuple[int, int]] = []  # (track, cluster_pos)
        for cid, cs, ce in _children(buf, ds, de):
            if cid == _ID_CUE_TIME:
                time_ms = _read_uint(buf, cs, ce)
            elif cid == _ID_CUE_TRACK_POSITIONS:
                track = 0
                cluster_pos = None
                for pid, ps, pe in _children(buf, cs, ce):
                    if pid == _ID_CUE_TRACK:
                        track = _read_uint(buf, ps, pe)
                    elif pid == _ID_CUE_CLUSTER_POSITION:
                        cluster_pos = _read_uint(buf, ps, pe)
                if cluster_pos is not None:
                    positions.append((track, cluster_pos))
        if time_ms is None:
            continue
        entries.extend((time_ms, cluster_pos, track) for track, cluster_pos in positions)

    if target_track is not None:
        targeted = [(time_ms, pos) for time_ms, pos, track in entries if track == target_track]
        if targeted:
            return targeted
    return [(time_ms, pos) for time_ms, pos, _track in entries]


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


async def _probe_total_size(client: httpx.AsyncClient, url: str) -> int | None:
    """Best-effort total file size from a zero-byte range response.

    Used only to locate a tail ``Cues`` element when the SeekHead position is
    stale. Never raises: an unsupported client/response simply returns ``None``.
    """
    try:
        resp = await client.get(url, headers={"Range": "bytes=0-0"}, follow_redirects=True)
        headers = resp.headers
    except Exception:  # noqa: BLE001 - size is only an optimisation
        return None
    try:
        content_range = headers.get("content-range") or ""
    except Exception:  # noqa: BLE001 - headers may be absent/mocked
        content_range = ""
    match = re.search(r"/(\d+)\s*$", content_range)
    if match:
        return int(match.group(1))
    try:
        length = headers.get("content-length")
        if length is not None and int(length) > 1:
            return int(length)
    except (TypeError, ValueError):
        pass
    return None


def _srt_timestamp(ms: int) -> str:
    ms = max(0, ms)
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _render_srt_blocks(collected: list[tuple[int, int, str]]) -> list[str]:
    """Sort, de-duplicate, and render decoded blocks as numbered SRT cues."""
    seen: set[int] = set()
    blocks: list[str] = []
    for start_ms, duration_ms, text in sorted(collected, key=lambda item: item[0]):
        if start_ms in seen or not text.strip():
            continue
        seen.add(start_ms)
        end_ms = start_ms + (duration_ms or 2500)
        blocks.append(
            f"{len(blocks) + 1}\n"
            f"{_srt_timestamp(start_ms)} --> {_srt_timestamp(end_ms)}\n{text}"
        )
    return blocks


async def extract_embedded_srt(
    stream_url: str,
    client: httpx.AsyncClient,
    *,
    want_language: str = "eng",
    timeout: float = 8.5,
    min_cues: int = _MIN_EMBEDDED_CUES,
    min_bytes: int = _MIN_EMBEDDED_BYTES,
    tail_cues_bytes: int = _TAIL_CUES_BYTES,
    cluster_bytes: int = _CUE_CLUSTER_BYTES,
    max_extract_bytes: int = _MAX_EXTRACT_BYTES,
) -> bytes | None:
    """Extract a complete text-subtitle reference from a remote MKV via ranges.

    Locates the ``Cues`` index through the ``SeekHead`` (reading it with one
    small targeted range request — it usually sits at the tail), then walks
    every cue cluster and decodes the target track's blocks to reconstruct the
    full timeline. Returns raw SRT bytes, or ``None`` when the container is
    unsupported, the Cues index is missing, or a complete track cannot be
    collected (never a truncated/partial reference). Never raises.
    """
    import asyncio

    async def _run() -> bytes | None:
        header = await _fetch(client, stream_url, 0, 1_048_576)
        seg_start, seg_end = find_segment(header)
        base = seg_start
        window_end = min(seg_end, len(header)) if seg_end <= len(header) else len(header)

        seeks: dict[int, int] = {}
        info_scale, duration_ms = 1_000_000, 0.0
        all_tracks: list[dict] = []
        for eid, ds, de in _children(header, base, window_end):
            if eid == _ID_SEEK_HEAD:
                seeks.update(parse_seek_head(header, ds, de))
            elif eid == _ID_INFO:
                info_scale, duration_ms = parse_info(header, ds, de)
            elif eid == _ID_TRACKS:
                all_tracks = parse_all_tracks(header, ds, de)

        async def _fetch_at(rel_pos: int, size: int) -> bytes:
            return await _fetch(client, stream_url, base + rel_pos, size)

        if not all_tracks and _ID_TRACKS in seeks:
            blob = await _fetch_at(seeks[_ID_TRACKS], 262_144)
            contents = _element_contents(blob, _ID_TRACKS)
            if contents:
                all_tracks = parse_all_tracks(blob, contents[0], contents[1])
        if not all_tracks:
            raise MKVRangeError("no Tracks element found in the container")

        # Diagnostic: surface every detected subtitle track (number/codec/lang)
        # before deciding whether a usable S_TEXT/UTF8 track exists.
        subtitle_tracks = [t for t in all_tracks if t["type"] == _TRACK_TYPE_SUBTITLE]
        text_tracks = [
            t for t in subtitle_tracks if t["codec"] in _RANGE_SUPPORTED_TEXT_CODECS
        ]
        if subtitle_tracks:
            logger.info(
                "[reference] range MKV extraction: detected %d subtitle track(s): %s",
                len(subtitle_tracks),
                _format_subtitle_tracks(subtitle_tracks),
            )
        if not text_tracks:
            if not subtitle_tracks:
                logger.info(
                    "[reference] range MKV extraction: file contains 0 subtitle "
                    "tracks (only video/audio detected)"
                )
            else:
                logger.info(
                    "[reference] range MKV extraction: no S_TEXT/UTF8 track found. "
                    "Detected subtitle tracks: [%s] (Total video/audio/sub tracks: %d)",
                    _format_subtitle_tracks(subtitle_tracks),
                    len(all_tracks),
                )
            raise MKVRangeError("no supported text subtitle track found in Tracks")

        # Prefer genuine English tracks (optionally SDH/full) and drop
        # forced/signs/songs/commentary tracks that would only ever sync a
        # handful of cues.
        candidates = [t for t in text_tracks if not _is_low_priority_track(t)]
        if not candidates:
            logger.info(
                "[reference] range MKV extraction: only forced/signs/commentary "
                "subtitle tracks present -> skipping embedded reference"
            )
            raise MKVRangeError("only low-priority (forced/signs) subtitle tracks present")
        candidates.sort(key=_track_priority)
        logger.info(
            "[reference] range MKV extraction: candidate track(s) in priority order: %s",
            ", ".join(
                f"Track {t['number']}({t['language']}"
                + (f", '{t['name']}'" if t.get("name") else "")
                + ")"
                for t in candidates
            ),
        )

        # Locate the Cues index via SeekHead and read it with one targeted range
        # request; it normally sits near the end of the file.
        if _ID_CUES not in seeks:
            raise MKVRangeError("no Cues index (cannot enumerate clusters)")
        cue_blob = await _fetch_at(seeks[_ID_CUES], tail_cues_bytes)
        cue_contents = _element_contents(cue_blob, _ID_CUES)
        if cue_contents is None:
            # Stale SeekHead or a Cues element in the final bytes: retry against
            # the tail using the file's total size (when it can be probed).
            total = await _probe_total_size(client, stream_url)
            if total is not None:
                start = max(0, total - tail_cues_bytes)
                cue_blob = await _fetch(client, stream_url, start, tail_cues_bytes)
                cue_contents = _element_contents(cue_blob, _ID_CUES)
                if cue_contents is not None:
                    logger.info(
                        "[reference] range MKV extraction: read tail Cues at byte %d of %d",
                        start,
                        total,
                    )
        if cue_contents is None:
            raise MKVRangeError("Cues element not found at its SeekHead position")
        cue_range: tuple[int, int] = cue_contents
        logger.info(
            "[reference] range MKV extraction: located Cues index (%d bytes)",
            cue_range[1] - cue_range[0],
        )

        async def _extract_track(track_number: int) -> list[tuple[int, int, str]]:
            """Walk every cue cluster and decode the target track's blocks."""
            cues = parse_cues(
                cue_blob, cue_range[0], cue_range[1], target_track=track_number
            )
            if not cues:
                return []
            positions = sorted({cluster_pos for _time, cluster_pos in cues})
            collected: list[tuple[int, int, str]] = []
            decoded_spans: list[tuple[int, int]] = []
            read = 0
            for pos in positions:
                if any(start <= pos < stop for start, stop in decoded_spans):
                    continue
                if read >= max_extract_bytes:
                    raise MKVRangeError(
                        "embedded extraction exceeded the byte budget before completion"
                    )
                try:
                    blob = await _fetch_at(pos, cluster_bytes)
                except MKVRangeError:
                    continue
                # Record the *actual* decoded span so overlapping windows are
                # not re-fetched (parse_cluster decodes every cluster in a blob).
                decoded_spans.append((pos, pos + len(blob)))
                read += len(blob)
                collected.extend(parse_cluster(blob, 0, len(blob), track_number))
            logger.info(
                "[reference] range MKV extraction: track %d decoded %d block(s) across "
                "%d/%d cue cluster(s), %d bytes read",
                track_number,
                len(collected),
                len(decoded_spans),
                len(positions),
                read,
            )
            return collected

        # Walk candidates best-first. A complete track wins; anything below the
        # completeness bar is rejected (never cached as a truncated reference).
        for track in candidates:
            blocks = _render_srt_blocks(await _extract_track(track["number"]))
            payload = ("\n\n".join(blocks) + "\n").encode("utf-8") if blocks else b""
            if _is_incomplete_embedded(
                len(blocks), len(payload), duration_ms, min_cues=min_cues, min_bytes=min_bytes
            ):
                logger.info(
                    "[reference] range MKV extraction: track %d incomplete "
                    "(%d cue(s), %d bytes) -> trying next candidate",
                    track["number"],
                    len(blocks),
                    len(payload),
                )
                continue
            logger.info(
                "[reference] range MKV extraction: track %d complete with %d cue(s)/%d bytes -> using",
                track["number"],
                len(blocks),
                len(payload),
            )
            return payload

        raise MKVRangeError(
            "no complete S_TEXT/UTF8 track could be extracted (no truncated reference)"
        )

    try:
        return await asyncio.wait_for(_run(), timeout)
    except Exception as exc:  # noqa: BLE001 - degrade to the ffmpeg tier
        logger.info("[reference] range MKV extraction unavailable: %s", exc)
        return None
