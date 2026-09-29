"""Local embedded subtitle reference strategy.

Extracts ground-truth reference subtitle tracks directly from local media files
(mounted via Docker volumes from AIOStreams, Sonarr, Radarr, or local storage).
Because the embedded track is muxed into the exact video file being played, its
speech timing matches the audio with 0-millisecond error, regardless of custom
platform bumpers, regional distributor intros (MULTI/French/German), or retail
discrepancies.

Fast streaming: probes the container headers with ffprobe in ~150ms, then streams
only the first 35-50 dialogue cues via ffmpeg into stdout in ~3s.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import time
from typing import TYPE_CHECKING

from app.config import settings
from app.services.sync.cache import ReferenceDiskCache
from app.services.sync.query import ResolvedReference

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.services.sync.query import ReferenceQuery

logger = logging.getLogger(__name__)

_DEFAULT_MEDIA_DIRS = ("/mnt/aiostreams", "/data/movies", "/data/tv")
_TEXT_SUBTITLE_CODECS = frozenset(
    {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text", "hdmv_text_subtitle"}
)
_PREFERRED_LANGUAGES = ("eng", "en", "fre", "fr", "spa", "es", "deu", "de", "ita", "it", "ara", "ar")
_MIN_PARTIAL_CUES = 8


class LocalEmbeddedStrategy:
    """Ground-truth reference from embedded subtitle tracks in local/mounted media files."""

    name = "embedded"

    def __init__(
        self,
        *,
        media_dirs: list[str] | tuple[str, ...] | None = None,
        ffprobe_path: str | None = None,
        ffmpeg_path: str | None = None,
        probe_timeout: float = 4.0,
        extract_timeout: float = 6.0,
        target_cues: int = 40,
        min_bytes: int = 500,
        cache: ReferenceDiskCache | None = None,
    ) -> None:
        if media_dirs is not None:
            self.media_dirs = [d.strip() for d in media_dirs if d and d.strip()]
        else:
            configured = getattr(settings, "LOCAL_MEDIA_DIRS", None)
            if configured is None:
                configured = os.environ.get("LOCAL_MEDIA_DIRS")
            if configured is not None:
                self.media_dirs = [d.strip() for d in configured.split(",") if d.strip()]
            else:
                self.media_dirs = list(_DEFAULT_MEDIA_DIRS)

        self._ffprobe_path = ffprobe_path or getattr(settings, "FFPROBE_PATH", None) or shutil.which("ffprobe")
        self._ffmpeg_path = ffmpeg_path or getattr(settings, "FFMPEG_PATH", None) or shutil.which("ffmpeg")
        self.probe_timeout = probe_timeout
        self.extract_timeout = extract_timeout
        self.target_cues = target_cues
        self.min_bytes = min_bytes
        self.cache = cache if cache is not None else ReferenceDiskCache(min_bytes=min_bytes)

        # In-memory fast path cache: target_filename -> (timestamp, file_path)
        self._path_cache: dict[str, tuple[float, str]] = {}
        self._path_cache_ttl = 600.0  # 10 minutes

    async def resolve(self, query: ReferenceQuery) -> str | None:
        """Return the extracted embedded track text or None."""
        resolved = await self.resolve_with_provenance(query)
        return resolved.text

    async def resolve_with_provenance(self, query: ReferenceQuery) -> ResolvedReference:
        """Resolve a ground-truth reference from local media with provenance."""
        # 1. Check disk cache first (exact kind beats external edition)
        cached = self.cache.get(query)
        if cached is not None and cached.kind == "embedded" and cached.text:
            return cached

        target_filename = str(query.target_filename or "").strip()
        if not target_filename:
            return ResolvedReference(None)

        if not self._ffprobe_path or not self._ffmpeg_path:
            logger.debug("[reference] embedded strategy skipped: ffprobe/ffmpeg not installed")
            return ResolvedReference(None)

        # 2. Locate media file in local media directories (or stream URL)
        raw_size = query.video_size
        video_size: int | None = None
        if raw_size is not None:
            try:
                parsed_size = int(str(raw_size).strip())
            except (TypeError, ValueError):
                parsed_size = 0
            video_size = parsed_size if parsed_size > 0 else None
        file_path = await asyncio.to_thread(self._find_media_file, target_filename, video_size)
        if not file_path:
            if query.stream_url and str(query.stream_url).startswith(("http://", "https://")):
                file_path = str(query.stream_url).strip()
                logger.info("[reference] embedded strategy: probing remote stream URL '%s'", file_path)
            else:
                logger.debug("[reference] embedded strategy: file '%s' not found in local media dirs", target_filename)
                return ResolvedReference(None)
        else:
            logger.info("[reference] embedded strategy: matched local media file '%s'", file_path)

        # 3. Probe subtitle streams
        stream_index = await asyncio.to_thread(self._probe_best_subtitle_stream, file_path)
        if stream_index is None:
            logger.info("[reference] embedded strategy: no usable text subtitle stream in '%s'", file_path)
            return ResolvedReference(None)

        # 4. Extract reference cues (full for local files, fast head-probe for remote streams)
        is_remote = str(file_path).startswith(("http://", "https://"))
        raw_text = await asyncio.to_thread(self._extract_partial_reference, file_path, stream_index, is_remote=is_remote)
        if not raw_text or "-->" not in raw_text:
            logger.warning("[reference] embedded strategy: extraction yielded no valid cues")
            return ResolvedReference(None)

        from app.services.sync_service import sanitize_subtitle

        sanitized = sanitize_subtitle(raw_text)
        if not sanitized or "-->" not in sanitized:
            return ResolvedReference(None)

        # 5. Persist to reference disk cache
        self.cache.set(
            query,
            source="embedded",
            text=sanitized,
            kind="embedded",
            bluray_match=False,
            candidate=target_filename,
            partial=is_remote,
        )
        logger.info(
            "[reference] embedded strategy: successfully extracted %d chars of ground-truth reference (partial=%s)",
            len(sanitized),
            is_remote,
        )
        return ResolvedReference(sanitized, kind="embedded", partial=is_remote)

    def _find_media_file(self, filename: str, video_size: int | None = None) -> str | None:
        """Search configured media directories for a matching video file."""
        now = time.monotonic()
        cache_key = f"{filename}:{video_size or 0}"
        if cache_key in self._path_cache:
            ts, cached_path = self._path_cache[cache_key]
            if now - ts < self._path_cache_ttl and os.path.exists(cached_path):
                return cached_path
            self._path_cache.pop(cache_key, None)

        target_clean = os.path.basename(filename).strip().lower()
        target_stem = os.path.splitext(target_clean)[0]

        for base_dir in self.media_dirs:
            if not os.path.exists(base_dir):
                continue
            try:
                for root, _dirs, files in os.walk(base_dir):
                    parent_clean = os.path.basename(root).strip().lower()
                    for f in files:
                        f_clean = f.strip().lower()
                        if not f_clean.endswith((".mkv", ".mp4", ".avi", ".webm")):
                            continue
                        f_stem = os.path.splitext(f_clean)[0]

                        # Exact filename match
                        if f_clean == target_clean or f_stem == target_stem:
                            matched = os.path.join(root, f)
                            self._path_cache[cache_key] = (now, matched)
                            return matched

                        # Usenet / scene directory match: parent folder has the release name
                        if parent_clean == target_stem:
                            matched = os.path.join(root, f)
                            self._path_cache[cache_key] = (now, matched)
                            return matched

                        # Video size match if provided
                        if video_size:
                            try:
                                if os.path.getsize(os.path.join(root, f)) == video_size:
                                    matched = os.path.join(root, f)
                                    self._path_cache[cache_key] = (now, matched)
                                    return matched
                            except OSError:
                                pass
            except OSError as exc:
                logger.debug("[reference] error scanning media dir %s: %s", base_dir, exc)

        return None

    def _probe_best_subtitle_stream(self, file_path: str) -> int | None:
        """Run fast ffprobe to identify the best text dialogue subtitle stream."""
        ffprobe = self._ffprobe_path
        if not ffprobe:
            return None
        cmd: list[str] = [
            ffprobe,
            "-v", "error",
            "-probesize", "3000000",
            "-analyzeduration", "3000000",
        ]
        if file_path.startswith(("http://", "https://")):
            cmd.extend([
                "-rw_timeout", "5000000",
                "-user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            ])
        cmd.extend([
            "-select_streams", "s",
            "-show_entries", "stream=index,codec_name:stream_tags=language,title",
            "-of", "json",
            file_path,
        ])
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=self.probe_timeout)
            if res.returncode != 0 or not res.stdout:
                return None
            data = json.loads(res.stdout)
        except Exception as exc:
            logger.debug("[reference] embedded ffprobe failed: %s", exc)
            return None

        streams = data.get("streams", [])
        if not streams:
            return None

        candidates: list[tuple[int, int, int]] = []
        for s in streams:
            codec = str(s.get("codec_name") or "").lower()
            if codec not in _TEXT_SUBTITLE_CODECS:
                continue
            tags = s.get("tags") or {}
            lang = str(tags.get("language") or "").lower()
            title = str(tags.get("title") or "").lower()

            # Rank indicators:
            # 1. Non-commentary (commentary is completely wrong speech timing)
            if "commentary" in title or "director" in title:
                continue
            # 2. Non-forced dialogue
            is_forced = "forced" in title or "signs" in title
            forced_rank = 0 if is_forced else 1

            # 3. Language preference
            lang_rank = 0
            if lang in _PREFERRED_LANGUAGES:
                lang_rank = len(_PREFERRED_LANGUAGES) - _PREFERRED_LANGUAGES.index(lang)

            candidates.append((forced_rank, lang_rank, int(s["index"])))

        if not candidates:
            return None

        # Sort: dialogue first, then language preference
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return candidates[0][2]

    def _extract_partial_reference(self, file_path: str, stream_index: int, is_remote: bool = False) -> str | None:
        """Stream subtitle cues from ffmpeg directly (full for local, head-probe for remote)."""
        ffmpeg = self._ffmpeg_path
        if not ffmpeg:
            return None
        cmd: list[str] = [
            ffmpeg,
            "-v", "error",
            "-nostdin",
            "-y",
            "-vn", "-an",
        ]
        if is_remote:
            cmd.extend([
                "-rw_timeout", "5000000",
                "-user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
                "-analyzeduration", "3000000",
                "-probesize", "3000000",
                "-to", "00:02:30",
            ])
        cmd.extend([
            "-i", file_path,
            "-map", f"0:{stream_index}",
            "-c:s", "text",
            "-f", "srt",
            "-",
        ])
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as exc:
            logger.debug("[reference] embedded ffmpeg popen failed: %s", exc)
            return None
        if proc.stdout is None:
            return None

        collected_lines: list[str] = []
        cues_count = 0
        target_cues = 15 if is_remote else self.target_cues
        timeout = self.extract_timeout if is_remote else 20.0
        deadline = time.monotonic() + timeout

        try:
            for raw_line in proc.stdout:
                line = raw_line.decode("utf-8", errors="ignore")
                collected_lines.append(line)
                if "-->" in line:
                    cues_count += 1
                    if cues_count >= target_cues:
                        proc.terminate()
                        break
                if time.monotonic() > deadline:
                    logger.info(
                        "[reference] embedded ffmpeg reached extract timeout (%.1fs, cues=%d)",
                        timeout,
                        cues_count,
                    )
                    proc.terminate()
                    break
        except Exception as exc:
            logger.debug("[reference] embedded ffmpeg stream read error: %s", exc)
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=1.5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        content = "".join(collected_lines)
        if cues_count >= _MIN_PARTIAL_CUES and "-->" in content:
            return content
        return None


# Alias for orchestrator compatibility
EmbeddedStrategy = LocalEmbeddedStrategy
