"""Automated LRU Disk Cache Manager for subtitles."""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


class LRUCacheManager:
    """
    Manages cached .srt files on disk with automated LRU cleanup (1GB / 500 files limit).
    Also manages persistent subtitle download metadata for on-demand fetching.
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        max_bytes: int = settings.CACHE_MAX_BYTES,
        max_files: int = settings.CACHE_MAX_FILES,
    ):
        self.cache_dir = Path(cache_dir or settings.CACHE_DIR)
        self.max_bytes = max_bytes
        self.max_files = max_files
        self.meta_dir = self.cache_dir / "_meta"
        self._lock = asyncio.Lock()

        # Ensure directories exist
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.meta_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create cache directory {self.cache_dir}: {e}")

    def get_subtitle_path(self, sub_id: str) -> Path:
        """Return path for a given subtitle ID."""
        # Sanitize sub_id to avoid path traversal
        clean_id = os.path.basename(sub_id).replace("..", "")
        return self.cache_dir / f"{clean_id}.srt"

    def get_meta_path(self, sub_id: str) -> Path:
        """Return path for subtitle metadata JSON."""
        clean_id = os.path.basename(sub_id).replace("..", "")
        return self.meta_dir / f"{clean_id}.json"

    async def get_subtitle(self, sub_id: str) -> bytes | None:
        """
        Retrieve cached subtitle file if it exists and update its access time (LRU).
        """
        file_path = self.get_subtitle_path(sub_id)
        if not file_path.is_file():
            return None

        async with self._lock:
            try:
                # Update atime to now for LRU tracking (best-effort on container mounts)
                now = time.time()
                try:
                    os.utime(file_path, (now, now))
                except OSError:
                    pass
                return file_path.read_bytes()
            except OSError as e:
                logger.warning(f"Error reading cached subtitle {file_path}: {e}")
                return None

    async def save_subtitle(self, sub_id: str, data: bytes) -> bool:
        """
        Atomically save subtitle file to disk and trigger LRU cleanup.
        """
        file_path = self.get_subtitle_path(sub_id)
        tmp_path = file_path.with_suffix(".srt.tmp")

        async with self._lock:
            try:
                tmp_path.write_bytes(data)
                tmp_path.replace(file_path)
                now = time.time()
                try:
                    os.utime(file_path, (now, now))
                except OSError:
                    pass
            except OSError as e:
                logger.error(f"Failed to write subtitle {file_path}: {e}")
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
                return False

            # Enforce limits after saving
            self._enforce_limits()
            return True

    # Provider credential keys that must NEVER be written to persistent disk.
    _SENSITIVE_METADATA_KEYS: frozenset[str] = frozenset(
        {"subdl_key", "subsource_key", "opensubtitles_key"}
    )

    def store_metadata(self, sub_id: str, metadata: dict[str, Any]) -> None:
        """Store download metadata for on-demand retrieval.

        Provider API keys are stripped before persistence so that no raw
        credentials are ever written to ``subs_cache/_meta/*.json``.  User-specific
        credentials for on-demand downloads are held separately in an expiring
        in-memory store (see ``app.services.credentials``).
        """
        safe_metadata = {
            k: v
            for k, v in metadata.items()
            if k not in self._SENSITIVE_METADATA_KEYS
        }
        meta_path = self.get_meta_path(sub_id)
        try:
            meta_path.write_text(json.dumps(safe_metadata), encoding="utf-8")
        except OSError as e:
            logger.warning(f"Failed to store metadata for {sub_id}: {e}")

    def get_metadata(self, sub_id: str) -> dict[str, Any] | None:
        """Retrieve stored download metadata."""
        meta_path = self.get_meta_path(sub_id)
        if not meta_path.is_file():
            return None
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to read metadata for {sub_id}: {e}")
            return None

    def clear_metadata(self) -> None:
        """Invalidate and wipe all on-disk metadata cache entries."""
        try:
            if self.meta_dir.is_dir():
                for p in self.meta_dir.glob("*.json"):
                    try:
                        p.unlink()
                    except OSError:
                        pass
        except Exception as e:
            logger.warning(f"Failed to clear metadata cache directory {self.meta_dir}: {e}")

    def _enforce_limits(self) -> None:
        """
        Enforce max_bytes (1GB) and max_files (500) via Least Recently Used (LRU) eviction.
        """
        try:
            entries: list[Path] = [
                p for p in self.cache_dir.iterdir() if p.is_file() and p.suffix == ".srt"
            ]
        except OSError:
            return

        if not entries:
            return

        # Gather stat info: (access_time, size, path)
        file_stats = []
        total_size = 0
        for p in entries:
            try:
                stat = p.stat()
                atime = getattr(stat, "st_atime", stat.st_mtime)
                file_stats.append((atime, stat.st_size, p))
                total_size += stat.st_size
            except OSError:
                continue

        # Check if cleanup needed
        if len(file_stats) <= self.max_files and total_size <= self.max_bytes:
            return

        # Sort by access time ascending (oldest accessed first)
        file_stats.sort(key=lambda item: item[0])

        logger.info(
            f"LRU Cache cleanup triggered: {len(file_stats)} files ({total_size / (1024 * 1024):.2f}MB). "
            f"Limits: {self.max_files} files, {self.max_bytes / (1024 * 1024):.2f}MB."
        )

        current_count = len(file_stats)
        for _atime, size, p in file_stats:
            if current_count <= self.max_files and total_size <= self.max_bytes:
                break
            try:
                p.unlink()
                # Also delete associated metadata if present
                sub_id = p.stem
                meta_p = self.get_meta_path(sub_id)
                if meta_p.exists():
                    meta_p.unlink()

                total_size -= size
                current_count -= 1
                logger.debug(f"Evicted LRU subtitle cache entry: {p.name}")
            except OSError as e:
                logger.warning(f"Could not delete cache file {p}: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Return cache health and usage statistics."""
        try:
            entries = [p for p in self.cache_dir.iterdir() if p.is_file() and p.suffix == ".srt"]
            total_size = sum(p.stat().st_size for p in entries)
            return {
                "cache_dir": str(self.cache_dir),
                "file_count": len(entries),
                "max_files": self.max_files,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "max_bytes": self.max_bytes,
                "max_mb": round(self.max_bytes / (1024 * 1024), 2),
            }
        except Exception as e:
            return {"error": str(e)}


cache_manager = LRUCacheManager()
