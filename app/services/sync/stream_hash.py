"""OpenSubtitles ``MovieHash`` computation from a remote video stream.

Only two 64 KiB blocks (the first and the last) are fetched via HTTP Range
requests, so the media itself is never downloaded. This lets the sync engine
do an exact-bytes hash lookup against OpenSubtitles even when the player
supplies no ``videoHash``.
"""

from __future__ import annotations

import asyncio
import logging
import struct
from typing import TYPE_CHECKING

from app.utils.network_security import is_safe_public_url

if TYPE_CHECKING:  # pragma: no cover - typing only
    import httpx

logger = logging.getLogger(__name__)

_CHUNK = 65536  # 64 KiB, per the OpenSubtitles moviehash spec
_MIN_SIZE = _CHUNK * 2
# A server that ignores ``Range`` answers 200 with the whole movie; refuse to
# buffer anything larger than a single chunk + slack.
_MAX_RANGE_RESPONSE = _CHUNK * 4


def _opensubtitles_hash(first: bytes, last: bytes, size: int) -> str:
    """64-bit OpenSubtitles hash: file size + sum of little-endian uint64s."""
    total = size
    for chunk in (first, last):
        for offset in range(0, len(chunk) - 7, 8):
            total += struct.unpack_from("<Q", chunk, offset)[0]
    return f"{total & 0xFFFFFFFFFFFFFFFF:016x}"


async def _content_length(client: httpx.AsyncClient, stream_url: str) -> int | None:
    try:
        head = await client.head(stream_url, follow_redirects=True)
        length = head.headers.get("content-length")
        if length and length.isdigit():
            return int(length)
        content_range = head.headers.get("content-range")
        if content_range and "/" in content_range:
            total = content_range.rsplit("/", 1)[-1]
            if total.isdigit():
                return int(total)
    except Exception as exc:  # noqa: BLE001 - best effort
        logger.debug("[reference] hash: content-length probe failed: %s", exc)
    # Fall back to a one-byte range probe.
    try:
        resp = await client.get(stream_url, headers={"Range": "bytes=0-0"}, follow_redirects=True)
        content_range = resp.headers.get("content-range")
        if content_range and "/" in content_range:
            total = content_range.rsplit("/", 1)[-1]
            if total.isdigit():
                return int(total)
    except Exception as exc:  # noqa: BLE001 - best effort
        logger.debug("[reference] hash: range probe failed: %s", exc)
    return None


async def _range(
    client: httpx.AsyncClient, stream_url: str, start: int, end: int
) -> bytes | None:
    try:
        resp = await client.get(
            stream_url,
            headers={"Range": f"bytes={start}-{end}"},
            follow_redirects=True,
        )
    except Exception as exc:  # noqa: BLE001 - best effort
        logger.debug("[reference] hash: range fetch failed: %s", exc)
        return None
    body = resp.content or b""
    if resp.status_code == 206 and body:
        if len(body) > _MAX_RANGE_RESPONSE:
            return None
        return body
    # A 200 means the server ignored Range and returned the whole file.
    if resp.status_code == 200 and 0 < len(body) <= _MAX_RANGE_RESPONSE:
        return body
    return None


async def fetch_stream_moviehash(
    stream_url: str,
    client: httpx.AsyncClient,
    *,
    timeout: float = 1.5,
) -> tuple[str, int] | None:
    """Return ``(moviehash_hex, size)`` for a remote stream, or ``None``.

    Bounded by ``timeout`` seconds (range fetch + probe combined). Never
    raises; any failure degrades to ``None`` so the caller can fall through to
    the next tier.
    """
    url = (stream_url or "").strip()
    if not url:
        return None
    safe, reason = await asyncio.to_thread(is_safe_public_url, url)
    if not safe:
        logger.warning("[reference] hash: blocked unsafe stream URL (%s)", reason)
        return None

    async def _compute() -> tuple[str, int] | None:
        size = await _content_length(client, url)
        if size is None or size < _MIN_SIZE:
            return None
        first = await _range(client, url, 0, _CHUNK - 1)
        last = await _range(client, url, size - _CHUNK, size - 1)
        if not first or not last:
            return None
        return _opensubtitles_hash(first, last, size), size

    try:
        result = await asyncio.wait_for(_compute(), timeout)
    except Exception as exc:  # noqa: BLE001 - never raise to caller
        logger.info("[reference] hash: stream hash unavailable (%s)", exc)
        return None
    if result:
        logger.info("[reference] hash: computed moviehash %s (%d bytes)", result[0], result[1])
    return result
