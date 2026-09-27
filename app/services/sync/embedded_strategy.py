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

# A sampled/partial internal-track reference can be tiny (the extractor already
# enforces a minimum of 5 cues), so the embedded cache floor is far below the
# external tier's 5 kB.
_MIN_REFERENCE_BYTES = 100
_ENGLISH_TAGS = frozenset({"eng", "en"})


def _coerce_file_size(value: object) -> int | None:
    """Parse a ``video_size`` value (int or numeric string) to a positive int."""
    try:
        size = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return size if size > 0 else None
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
        client=None,
        timeout: float = 15.0,
        extract_timeout: float = 8.0,
        range_timeout: float = 8.5,
        inline_range: bool = False,
        warm_timeout: float = 60.0,
        min_bytes: int = _MIN_REFERENCE_BYTES,
        cache: ReferenceDiskCache | None = None,
    ) -> None:
        self._ffprobe_path = ffprobe_path
        self._ffmpeg_path = ffmpeg_path
        self._client = client
        self.timeout = timeout
        self.extract_timeout = extract_timeout
        self.range_timeout = range_timeout
        # Inline range extraction is off by default: a sparse sample can starve
        # the faster external tier. Range extraction runs in the background
        # warm-up instead (see ``warm_reference``).
        self.inline_range = inline_range
        self.warm_timeout = warm_timeout
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

        # Optional fast path: decode the internal text track purely via HTTP
        # Range requests (no sequential media read). Disabled by default so a
        # sparse sample never starves the external tier; the background
        # warm-up populates the cache densely instead.
        if self.inline_range and self._client is not None:
            from app.services.sync.mkv_range import extract_embedded_srt

            try:
                ranged = await asyncio.wait_for(
                    extract_embedded_srt(
                        stream_url,
                        self._client,
                        timeout=self.range_timeout,
                        file_size=_coerce_file_size(query.video_size),
                    ),
                    self.range_timeout + 1.0,
                )
            except Exception as exc:  # noqa: BLE001 - fall back to ffmpeg
                logger.info("[reference] range MKV extraction skipped: %s", exc)
                ranged = None
            if ranged:
                from app.services.sync_service import sanitize_subtitle

                sanitized = sanitize_subtitle(ranged.decode("utf-8", "replace"))
                if sanitized:
                    logger.info(
                        "[reference] embedded strategy: range-extracted %d chars of internal track",
                        len(sanitized),
                    )
                    self.cache.set(query, "embedded-range", sanitized, kind="embedded")
                    return ResolvedReference(sanitized, kind="embedded", partial=True)

        if self._client is not None:
            # Production: a remote demux would blow the player deadline. The
            # inline tier is cache-only; the background warm-up extracts and
            # caches the internal track for the next request.
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
        # Extracted only the first ~15 min of the media, so the reference is a
        # sampled prefix: mark it partial so duration gates don't reject it.
        return ResolvedReference(text, kind="embedded", partial=True)

    async def warm_reference(self, query: ReferenceQuery) -> ResolvedReference | None:
        """Background: densely extract the internal track and cache it.

        Runs without the player deadline. A later request finds the cached
        reference and syncs against the video's own subtitle track instantly.
        """
        if self._client is None:
            return None
        # Only an *embedded* cached reference should short-circuit the warm-up:
        # the disk cache is shared with the external tier's edition reference.
        cached = self.cache.get(query)
        if cached is not None and cached.kind == "embedded":
            return cached
        stream_url = (query.stream_url or "").strip()
        if not stream_url:
            return None
        allow_private = bool(getattr(settings, "ALLOW_PRIVATE_STREAM_URLS", True))
        safe, reason = await asyncio.to_thread(
            is_safe_public_url, stream_url, allow_private=allow_private
        )
        if not safe:
            logger.warning("[reference] embedded warm-up blocked unsafe URL (%s)", reason)
            return None

        from app.services.sync.mkv_range import extract_embedded_srt

        try:
            ranged = await extract_embedded_srt(
                stream_url,
                self._client,
                timeout=self.warm_timeout,
                file_size=_coerce_file_size(query.video_size),
            )
        except Exception as exc:  # noqa: BLE001 - background best effort
            logger.info("[reference] embedded warm-up extraction failed: %s", exc)
            return None
        if not ranged:
            return None
        from app.services.sync_service import sanitize_subtitle

        sanitized = sanitize_subtitle(ranged.decode("utf-8", "replace"))
        if not sanitized:
            return None
        self.cache.set(query, "embedded", sanitized, kind="embedded")
        logger.info(
            "[reference] embedded warm-up cached the internal track (%d chars)", len(sanitized)
        )
        return ResolvedReference(sanitized, kind="embedded", partial=True)

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
        """Extract one subtitle stream to SRT bytes (stdout).

        Reads only the first ~15 minutes of the media (``-t 900``) with a
        strict subprocess timeout, so a 20 GB remote stream is never read to
        EOF. If the timeout fires, whatever cues ffmpeg already emitted are
        used as a partial/sampled reference.
        """
        command = [
            ffmpeg,
            "-v", "error",
            "-nostdin",
            "-threads", "1",
            "-fflags", "+nobuffer+fastseek",
            "-analyzeduration", "10000000",
            "-probesize", "10000000",
            "-rw_timeout", "8000000",
            "-copyts",
            "-i", stream_url,
            # ``-t`` must come *after* ``-i`` (output option): before the input
            # it triggers seeking problems on remote HTTP MKV streams.
            "-t", "900",
            "-map", f"0:{index}",
            "-c:s", "srt",
            "-f", "srt",
            "-",
        ]
        try:
            completed = await asyncio.to_thread(
                subprocess.run, command, capture_output=True, timeout=self.extract_timeout
            )
        except subprocess.TimeoutExpired as exc:
            partial = bytes(exc.stdout or b"")
            logger.warning(
                "[reference] embedded extraction hit the %.1fs cap -> using %d partial bytes",
                self.extract_timeout,
                len(partial),
            )
            return partial or None
        except OSError as exc:
            logger.warning("[reference] embedded extraction failed: %s", exc)
            return None
        if completed.returncode != 0 or not completed.stdout:
            logger.warning(
                "[reference] embedded extraction exited with code %s",
                completed.returncode,
            )
            return None
        return bytes(completed.stdout)
