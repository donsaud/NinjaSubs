"""Secondary sync strategy: embedded English subtitle track extraction.

When a playable stream URL is available, ``ffprobe`` inspects the container
headers (via HTTP range requests — the media itself is never downloaded) for
an English subtitle track. If one exists, ``ffmpeg`` extracts that track to
an SRT payload which becomes the ground-truth reference.

Anything unconfirmed aborts: no ``ffprobe`` binary, no stream URL, no
language-tagged English track, or an extraction that fails validation all
return ``None`` so the orchestrator serves the original subtitle.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
from typing import TYPE_CHECKING

from app.config import settings
from app.services.sync.cache import ReferenceDiskCache
from app.services.sync.query import ResolvedReference
from app.utils.network_security import is_safe_public_url

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.services.sync.query import ReferenceQuery

logger = logging.getLogger(__name__)

_MIN_REFERENCE_BYTES = 5120
_ENGLISH_TAGS = frozenset({"eng", "en"})
# Only text subtitle codecs can be converted to SRT; image-based tracks
# (PGS/VobSub/DVB) are unusable as a reference and must fall through to the
# MovieHash tier.
_TEXT_SUBTITLE_CODECS = frozenset(
    {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text", "hdmv_text_subtitle"}
)


class EmbeddedStrategy:
    """Ground-truth reference from an embedded English subtitle track."""

    name = "embedded"

    def __init__(
        self,
        *,
        ffprobe_path: str | None = None,
        ffmpeg_path: str | None = None,
        timeout: float = 15.0,
        min_bytes: int = _MIN_REFERENCE_BYTES,
        cache: ReferenceDiskCache | None = None,
    ) -> None:
        self._ffprobe_path = ffprobe_path
        self._ffmpeg_path = ffmpeg_path
        self.timeout = timeout
        self.min_bytes = min_bytes
        self.cache = cache if cache is not None else ReferenceDiskCache(min_bytes=min_bytes)

    def _binary(self, kind: str) -> str | None:
        """Resolve the tool binary: explicit path wins, else PATH lookup."""
        if kind == "ffprobe":
            override = self._ffprobe_path or getattr(settings, "FFPROBE_PATH", None)
            return override or shutil.which("ffprobe")
        override = self._ffmpeg_path or getattr(settings, "FFMPEG_PATH", None)
        return override or shutil.which("ffmpeg")

    async def resolve(self, query: ReferenceQuery) -> str | None:
        """Return the extracted embedded English track or ``None``."""
        resolved = await self.resolve_with_provenance(query)
        return resolved.text

    async def resolve_with_provenance(
        self, query: ReferenceQuery
    ) -> ResolvedReference:
        """Resolve a reference, reporting the ``embedded`` decision kind with it."""
        cached = self.cache.get(query)
        if cached is not None:
            return cached
        stream_url = (query.stream_url or "").strip()
        if not stream_url:
            logger.info("[reference] embedded strategy skipped: no stream URL supplied")
            return ResolvedReference(None)
        allow_private = bool(getattr(settings, "ALLOW_PRIVATE_STREAM_URLS", True))
        safe, reason = await asyncio.to_thread(
            is_safe_public_url, stream_url, allow_private=allow_private
        )
        if not safe:
            logger.warning(
                "[reference] embedded strategy blocked unsafe stream URL (%s) -> skipping",
                reason,
            )
            return ResolvedReference(None)
        ffprobe = self._binary("ffprobe")
        ffmpeg = self._binary("ffmpeg")
        if not ffprobe or not ffmpeg:
            logger.info("[reference] embedded strategy skipped: ffprobe/ffmpeg unavailable")
            return ResolvedReference(None)
        try:
            text = await asyncio.wait_for(
                self._run(query, stream_url, ffprobe, ffmpeg), self.timeout * 2
            )
        except TimeoutError:
            logger.warning(
                "[reference] embedded strategy timed out after %.1fs", self.timeout * 2
            )
            return ResolvedReference(None)
        if text is None:
            return ResolvedReference(None)
        return ResolvedReference(text, kind="embedded")

    async def _run(
        self, query: ReferenceQuery, stream_url: str, ffprobe: str, ffmpeg: str
    ) -> str | None:
        track_index = await self._english_track_index(ffprobe, stream_url)
        if track_index is None:
            logger.info("[reference] embedded strategy: no English subtitle track found")
            return None
        logger.info(
            "[reference] embedded strategy: extracting English track %d", track_index
        )
        raw = await self._extract_track(ffmpeg, stream_url, track_index)
        if not raw or len(raw) <= self.min_bytes or b"-->" not in raw:
            logger.warning("[reference] embedded strategy: extraction failed validation")
            return None

        from app.extractor import transcode_to_utf8

        text = transcode_to_utf8(raw, lang="eng").decode("utf-8", "replace")
        from app.services.sync_service import sanitize_subtitle

        sanitized = sanitize_subtitle(text)
        if sanitized:
            self.cache.set(query, "embedded", sanitized, kind="embedded")
            return sanitized
        return text

    async def _english_track_index(self, ffprobe: str, stream_url: str) -> int | None:
        """Probe container headers only; return the absolute English sub stream index."""
        command = [
            ffprobe,
            "-v", "error",
            "-rw_timeout", "8000000",
            "-probesize", "4000000",
            "-analyzeduration", "8000000",
            "-select_streams", "s",
            "-show_entries", "stream=index,codec_name:stream_tags=language",
            "-of", "json",
            stream_url,
        ]
        try:
            completed = await asyncio.to_thread(
                subprocess.run, command, capture_output=True, timeout=self.timeout
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("[reference] embedded probe failed: %s", exc)
            return None
        if completed.returncode != 0:
            logger.warning("[reference] embedded probe exited with code %s", completed.returncode)
            return None
        try:
            payload = json.loads((completed.stdout or b"").decode("utf-8", "replace"))
        except ValueError:
            logger.warning("[reference] embedded probe returned invalid JSON")
            return None
        for stream in payload.get("streams", []) or []:
            tags = stream.get("tags") or {}
            language = str(tags.get("language") or "").strip().lower()
            codec = str(stream.get("codec_name") or "").strip().lower()
            if language in _ENGLISH_TAGS and codec in _TEXT_SUBTITLE_CODECS:
                if isinstance(stream.get("index"), int):
                    return int(stream["index"])
            elif language in _ENGLISH_TAGS:
                logger.info(
                    "[reference] embedded strategy: skipping image-based %s track", codec or "?"
                )
        return None

    async def _extract_track(self, ffmpeg: str, stream_url: str, index: int) -> bytes | None:
        """Extract one subtitle stream to SRT bytes (stdout)."""
        command = [
            ffmpeg,
            "-v", "error",
            "-rw_timeout", "8000000",
            "-i", stream_url,
            "-map", f"0:{index}",
            "-c:s", "srt",
            "-f", "srt",
            "-",
        ]
        try:
            completed = await asyncio.to_thread(
                subprocess.run, command, capture_output=True, timeout=self.timeout
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("[reference] embedded extraction failed: %s", exc)
            return None
        if completed.returncode != 0 or not completed.stdout:
            logger.warning(
                "[reference] embedded extraction exited with code %s",
                completed.returncode,
            )
            return None
        return bytes(completed.stdout)
