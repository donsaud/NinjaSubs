"""Range-based Matroska (MKV) subtitle extraction.

Remote debrid/usenet streams serve HTTP Range requests quickly but feed
``ffmpeg`` sequentially at a crawl (subtitles are interleaved with video, so
ffmpeg must read tens of MB of video to reach the first cue). This module takes
a single-shot head sample instead:

* fetch one contiguous range from offset 0 (``head_bytes``);
* parse the EBML container linearly in memory: ``Tracks``, then every
  ``Cluster`` in the sample;
* decode the prioritized ``S_TEXT/UTF8`` candidate track's subtitle blocks;
* accept the first track with at least ``min_cues`` cues (a partial reference),
  otherwise return ``None`` so the caller falls back to external providers.

Only plain-text ``S_TEXT/UTF8`` subtitle tracks are extractable here;
image-based (PGS/VobSub) and styled (ASS/SSA/WebVTT) tracks are reported but
not used. Forced/signs/songs/commentary tracks are deprioritised.
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
_ID_TRACKS = 0x1654AE6B
_ID_TRACK_ENTRY = 0xAE
_ID_TRACK_NUMBER = 0xD7
_ID_TRACK_TYPE = 0x83
_ID_CODEC_ID = 0x86
_ID_LANGUAGE = 0x22B59C
_ID_NAME = 0x536E
_ID_FLAG_FORCED = 0x55AA
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

# Single-window head probe: one contiguous range read from offset 0, demuxed
# linearly in memory. This issues ``Range: bytes=0-26214400`` (~25 MiB). A track
# that yields at least ``_MIN_EMBEDDED_CUES`` cues is accepted as a partial
# reference; anything less returns None so we fall back to external providers.
_HEAD_SAMPLE_BYTES = 25 * 1024 * 1024 + 1
_MIN_EMBEDDED_CUES = 60


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


# --------------------------------------------------------------------------- #
# Container parsing
# --------------------------------------------------------------------------- #
def find_segment(buf: bytes) -> tuple[int, int]:
    """Return ``(segment_data_start, segment_data_end)`` for a Segment in ``buf``."""
    for eid, dstart, dend in _children(buf, 0, len(buf)):
        if eid == _ID_SEGMENT:
            return dstart, dend
    raise MKVRangeError("no Segment element in the MKV header")


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
    head_bytes: int = _HEAD_SAMPLE_BYTES,
    min_cues: int = _MIN_EMBEDDED_CUES,
) -> bytes | None:
    """Sample the head of a remote MKV and demux its subtitle track in memory.

    A single contiguous range request (offset 0, ``head_bytes`` long) is fetched
    and parsed linearly; the first prioritized ``S_TEXT/UTF8`` candidate that
    yields at least ``min_cues`` cues is returned as raw (partial) SRT bytes.
    Returns ``None`` when the container is unsupported or no candidate reaches
    the cue bar — never a truncated reference. Never raises.
    """
    import asyncio

    async def _run() -> bytes | None:
        # Single-window head probe: one contiguous range request, then linear
        # in-memory EBML demuxing (no Cues traversal, no iterative requests).
        sample = await _fetch(client, stream_url, 0, head_bytes)
        seg_start, seg_end = find_segment(sample)
        base = seg_start
        window_end = min(seg_end, len(sample)) if seg_end <= len(sample) else len(sample)

        all_tracks: list[dict] = []
        for eid, ds, de in _children(sample, base, window_end):
            if eid == _ID_TRACKS:
                all_tracks = parse_all_tracks(sample, ds, de)
        if not all_tracks:
            raise MKVRangeError("no Tracks element within the head sample")

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

        # Demux the head sample linearly for each candidate (best first) and
        # accept the first track whose contiguous cue run reaches the bar.
        for track in candidates:
            collected: list[tuple[int, int, str]] = []
            for eid, ds, de in _children(sample, base, window_end):
                if eid == _ID_CLUSTER:
                    collected.extend(_parse_cluster_contents(sample, ds, de, track["number"]))
            blocks = _render_srt_blocks(collected)
            if len(blocks) < min_cues:
                logger.info(
                    "[reference] range MKV extraction: track %d yielded %d cue(s) "
                    "(< %d) -> trying next candidate",
                    track["number"],
                    len(blocks),
                    min_cues,
                )
                continue
            payload = ("\n\n".join(blocks) + "\n").encode("utf-8")
            logger.info(
                "[reference] range MKV extraction: track %d accepted with %d "
                "contiguous cue(s)/%d bytes from the head sample",
                track["number"],
                len(blocks),
                len(payload),
            )
            return payload

        raise MKVRangeError(
            f"no S_TEXT/UTF8 track yielded >= {min_cues} cues in the head sample"
        )

    try:
        return await asyncio.wait_for(_run(), timeout)
    except Exception as exc:  # noqa: BLE001 - degrade to the ffmpeg tier
        logger.info("[reference] range MKV extraction unavailable: %s", exc)
        return None
