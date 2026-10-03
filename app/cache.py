"""Automated LRU Disk Cache Manager for subtitles."""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from cachetools import TTLCache

from app.config import settings

logger = logging.getLogger(__name__)

# A failed download/extraction is retried by Stremio; remember it briefly so a
# broken upstream entry cannot trigger a full provider fan-out on every retry.
_FAILURE_TTL_SECONDS = 600

#: Metadata fields that hold a provider credential verbatim. These are stripped
#: before anything is written to ``_meta/*.json`` and again on read, so a
#: pre-existing file cannot re-expose a secret. The credential is not needed for
#: cache correctness: the download path resolves it per request from the URL
#: config, falling back to the environment (see ``parse_user_config``).
_SECRET_METADATA_FIELDS = (
    "subdl_key",
    "subsource_key",
    "opensubtitles_key",
)

#: Query parameters that carry a credential inside a URL. The value is dropped
#: but the URL stays usable: providers re-attach the credential themselves
#: (``SubdlProvider.download_archive`` sets both the ``x-api-key`` header and the
#: ``api_key`` parameter from the resolved key).
_SECRET_URL_PARAMS = frozenset(
    {
        "api_key",
        "apikey",
        "key",
        "token",
        "access_token",
        "accesstoken",
        "auth",
        "authorization",
        "password",
        "secret",
        "sig",
        "signature",
    }
)

#: Presence flags written in place of the secret values, so diagnostics keep
#: showing whether a credential was configured without revealing it.
_SECRET_PRESENCE_FLAGS = {
    "subdl_key": "has_subdl_key",
    "subsource_key": "has_subsource_key",
    "opensubtitles_key": "has_opensubtitles_key",
}


def _redact_url_secrets(url: Any) -> Any:
    """Strip credential-bearing query parameters from a URL, keeping it usable."""
    if not isinstance(url, str) or not url:
        return url
    if "?" not in url:
        return url
    base, _, query = url.partition("?")
    kept = []
    changed = False
    for part in query.split("&"):
        if not part:
            continue
        name, eq, _ = part.partition("=")
        if eq and name.strip().lower() in _SECRET_URL_PARAMS:
            changed = True
            continue
        kept.append(part)
    if not changed:
        return url
    # No dangling "?" when the query held nothing but credentials.
    return f"{base}?{'&'.join(kept)}" if kept else base


def sanitize_metadata(metadata: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Return ``(safe_metadata, changed)`` with every credential removed.

    Applied on write *and* on read. The read-side pass is what protects files
    written before this existed: a legacy ``_meta/*.json`` holding a plaintext
    key is rewritten in the safe form instead of being handed back to callers.
    """
    if not isinstance(metadata, dict):
        return metadata, False
    safe = dict(metadata)
    changed = False
    for field in _SECRET_METADATA_FIELDS:
        if field in safe:
            # A presence flag replaces the value, so diagnostics can still show
            # that a credential was configured. Metadata that never held a
            # credential is left completely untouched.
            value = safe.pop(field)
            safe[_SECRET_PRESENCE_FLAGS[field]] = bool(value and str(value).strip())
            changed = True
    for field in ("download_url", "stream_url"):
        if field in safe:
            redacted = _redact_url_secrets(safe[field])
            if redacted != safe[field]:
                safe[field] = redacted
                changed = True
    return safe, changed


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
        # Short-TTL negative cache for sub_ids whose upstream archive proved
        # unusable (all providers failed), keyed by sub_id.
        self._failed: TTLCache[str, str] = TTLCache(maxsize=4096, ttl=_FAILURE_TTL_SECONDS)

        # Ensure directories exist
        self.ensure_dirs()

    def ensure_dirs(self) -> bool:
        """Create the cache + metadata directories if missing (idempotent)."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.meta_dir.mkdir(parents=True, exist_ok=True)
            return True
        except OSError as e:
            logger.error(f"Failed to create cache directories under {self.cache_dir}: {e}")
            return False

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

        Empty/whitespace payloads are refused so an invalid archive extraction
        can never be persisted and replayed on the next request.
        """
        if not data or not data.strip():
            logger.warning(f"Refusing to cache empty subtitle payload for {sub_id}")
            return False

        file_path = self.get_subtitle_path(sub_id)
        tmp_path = file_path.with_suffix(".srt.tmp")

        async with self._lock:
            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
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

    def store_metadata(self, sub_id: str, metadata: dict[str, Any]) -> None:
        """Store download metadata for on-demand retrieval.

        Credentials are stripped first: a provider API key has no business being
        written to disk, and the download path re-resolves it per request.
        """
        meta_path = self.get_meta_path(sub_id)
        try:
            safe, _ = sanitize_metadata(metadata)
            # A fresh/re-mounted cache volume can be missing ``_meta``; recreate
            # it so metadata writes never fail with ENOENT.
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(safe), encoding="utf-8")
        except OSError as e:
            logger.warning(f"Failed to store metadata for {sub_id}: {e}")

    def get_metadata(self, sub_id: str) -> dict[str, Any] | None:
        """Retrieve stored download metadata.

        Sanitized on read, so a legacy file written before credentials were
        stripped cannot re-expose one. Such a file is rewritten in the safe form
        as a side effect; subtitle payload bytes are never touched.
        """
        meta_path = self.get_meta_path(sub_id)
        if not meta_path.is_file():
            return None
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to read metadata for {sub_id}: {e}")
            return None
        safe, changed = sanitize_metadata(raw)
        if changed:
            # Best-effort in-place repair of the metadata only. A failure here
            # must not break the request, and the payload is left untouched.
            try:
                meta_path.write_text(json.dumps(safe), encoding="utf-8")
            except OSError as e:
                logger.warning(f"Failed to sanitize metadata for {sub_id}: {e}")
        return safe

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

    def mark_failed(self, sub_id: str) -> None:
        """Remember (short TTL) that a sub_id's upstream archive was unusable."""
        self._failed[sub_id] = "1"

    def is_failed(self, sub_id: str) -> bool:
        """True when sub_id recently failed download/extraction (retry guard)."""
        return sub_id in self._failed

    def clear_failed(self, sub_id: str) -> None:
        self._failed.pop(sub_id, None)

    def clear_failures(self) -> None:
        self._failed.clear()

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
