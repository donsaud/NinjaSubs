"""Persistent on-disk cache of resolved English reference subtitles."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
import tempfile
import time
from pathlib import Path

from app.config import settings
from app.services.sync.query import ReferenceQuery, ResolvedReference

logger = logging.getLogger(__name__)

_DEFAULT_TTL_SECONDS = 2592000.0  # 30 days

# Decision kinds that prove the exact edition (vs. best-effort edition timing).
EXACT_KINDS = frozenset({"team", "hash", "embedded"})
_KNOWN_KINDS = ("team", "edition", "hash", "embedded")


class ReferenceDiskCache:
    """Persistent on-disk cache of resolved English reference subtitles.

    Files live at ``{root}/{query.cache_stem}_{source}_{kind}.srt`` and expire
    after a generous TTL so repeated subtitle switches for the same episode
    never re-hit (rate-limited) download APIs. The group segment keeps
    different stream editions (``FSiHD`` vs ``YIFY``) on separate timings, the
    target scope keeps different subtitle cue layouts separate, and the kind
    segment preserves the tree verdict so a cached team reference is never
    downgraded to ``edition`` on recovery.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        ttl: float | None = None,
        min_bytes: int = 5120,
    ) -> None:
        configured_root = root or getattr(settings, "REFERENCE_CACHE_DIR", None)
        if configured_root:
            self.root = Path(configured_root)
        else:
            self.root = Path(settings.CACHE_DIR) / "references"
        if ttl is None:
            try:
                ttl = float(getattr(settings, "REFERENCE_CACHE_TTL_SECONDS", _DEFAULT_TTL_SECONDS))
            except (TypeError, ValueError):
                ttl = _DEFAULT_TTL_SECONDS
        self.ttl = ttl
        self.min_bytes = min_bytes

    @staticmethod
    def _safe_source(source: str | None) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "", source or "ref") or "ref"

    @staticmethod
    def _safe_kind(kind: str | None) -> str:
        """Normalise a decision kind; unknown kinds fail closed to ``edition``."""
        lowered = (kind or "").strip().lower()
        return lowered if lowered in _KNOWN_KINDS else "edition"

    @classmethod
    def _parse_kind(cls, file_name: str, stem: str) -> str:
        """Recover the persisted decision kind from a cache filename.

        Unknown kinds receive conservative edition guardrails.
        """
        suffix = file_name[len(stem) + 1 :] if file_name.startswith(stem + "_") else ""
        if suffix.endswith(".srt"):
            suffix = suffix[: -len(".srt")]
        for kind in _KNOWN_KINDS:
            if suffix == kind or suffix.endswith("_" + kind):
                return kind
        return "edition"

    def _paths(self, query: ReferenceQuery) -> list[Path]:
        if not self.root.is_dir():
            return []
        return sorted(self.root.glob(f"{query.cache_stem}_*.srt"))

    @staticmethod
    def _sidecar_path(path: Path) -> Path:
        """Decision-metadata sidecar paired with a cached reference file."""
        return path.with_suffix(path.suffix + ".json")

    @classmethod
    def _remove(cls, path: Path) -> bool:
        """Delete a payload and its sidecar; ``True`` when the payload existed."""
        try:
            existed = path.is_file()
            path.unlink(missing_ok=True)
            cls._sidecar_path(path).unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("[reference] failed to evict %s: %s", path.name, exc)
            return False
        return existed

    def delete(self, query: ReferenceQuery) -> bool:
        """Evict every cached entry for a target-scoped query stem.

        A cached reference is only ever proven against the target cue layout
        that triggered its write. A different candidate for the same video can
        share the unscoped stem yet have an unrelated cue layout, so a
        reference that fails target cue-sanity must be dropped rather than
        re-read and re-rejected on every subsequent request.
        """
        removed = False
        for path in self._paths(query):
            if self._remove(path):
                removed = True
                logger.info("[reference] evicted unusable cache entry: %s", path.name)
        return removed

    def _read_verdict(self, path: Path, stem: str, text: str) -> tuple[str, bool, str, bool] | None:
        """Recover ``(kind, bluray_match, candidate, partial)`` for a cache file.

        The filename kind segment is authoritative; a sidecar written by
        :meth:`set` additionally restores whether both sides were verified
        BluRay plus the winning release name and partial status. Anything missing or
        inconsistent fails closed.
        """
        kind = self._parse_kind(path.name, stem)
        try:
            meta_path = self._sidecar_path(path)
            if not meta_path.is_file():
                raise FileNotFoundError(meta_path)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if (
            not isinstance(meta, dict)
            or meta.get("kind") != kind
            or meta.get("content_sha") != hashlib.sha256(text.encode("utf-8")).hexdigest()
            or (getattr(settings, "SYNC_REQUIRE_EXACT_MATCH", True) and meta.get("strict") is not True)
            or int(meta.get("engine_version", 1)) < 5
        ):
            return None
        return (
            kind,
            meta.get("bluray_match") is True,
            str(meta.get("candidate", "")),
            meta.get("partial") is True,
        )

    def get(self, query: ReferenceQuery) -> ResolvedReference | None:
        """Return the best valid cached reference or ``None`` on a miss.

        When several files share a stem, an exact-kind entry (team / hash /
        embedded) wins over a best-effort ``edition`` one. Missing or mismatched
        provenance is a miss, including a concurrent write between payload and
        metadata replacement. Old group-only entries never match the v2 stem.
        """
        best: tuple[int, str, ResolvedReference] | None = None  # (rank, name, ref)
        for path in self._paths(query):
            try:
                age = time.time() - path.stat().st_mtime
            except OSError:
                continue
            if age > self.ttl:
                if self._remove(path):
                    logger.info("[reference] expired cache entry removed: %s", path.name)
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            if not (text and len(text.encode("utf-8")) > self.min_bytes and "-->" in text):
                continue
            verdict = self._read_verdict(path, query.cache_stem, text)
            if verdict is None:
                continue
            kind, bluray_match, candidate, partial = verdict
            resolved = ResolvedReference(
                text=text,
                kind=kind,
                bluray_match=bluray_match and kind != "abort",
                candidate=candidate,
                partial=partial,
            )
            rank = 0 if kind in EXACT_KINDS else 1
            if best is None or (rank, path.name) < (best[0], best[1]):
                best = (rank, path.name, resolved)
        if best is None:
            return None
        logger.info(
            "[reference] cache hit on disk: %s (%d bytes, kind=%s)",
            best[1],
            len(best[2].text.encode("utf-8")) if best[2].text else 0,
            best[2].kind,
        )
        return best[2]

    def set(
        self,
        query: ReferenceQuery,
        source: str | None,
        text: str,
        kind: str = "edition",
        *,
        bluray_match: bool = False,
        candidate: str = "",
        partial: bool = False,
    ) -> None:
        """Atomically persist a resolved reference, dropping stale variants.

        A best-effort ``edition`` save never evicts an exact-kind entry
        (team / hash / embedded) for the same target-scoped stem. The verdict
        sidecar records what the filename alone cannot (notably
        ``bluray_match``).
        """
        if not text:
            return
        kind = self._safe_kind(kind)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            stem = query.cache_stem
            path = self.root / f"{stem}_{self._safe_source(source)}_{kind}.srt"
            for stale in self.root.glob(f"{stem}_*.srt"):
                if stale == path:
                    continue
                if kind == "edition" and self._parse_kind(stale.name, stem) in EXACT_KINDS:
                    continue
                with contextlib.suppress(OSError):
                    stale.unlink()
                    self._sidecar_path(stale).unlink()
            self._atomic_write(path, text)
            meta = {
                "kind": kind,
                "bluray_match": bool(bluray_match),
                "candidate": candidate,
                "partial": bool(partial),
                "strict": bool(getattr(settings, "SYNC_REQUIRE_EXACT_MATCH", True)),
                "content_sha": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "engine_version": 5,
            }
            self._atomic_write(self._sidecar_path(path), json.dumps(meta))
            logger.info(
                "[reference] cached reference on disk: %s (%d bytes)",
                path.name,
                len(text.encode("utf-8")),
            )
        except OSError as exc:
            logger.warning("[reference] failed to write reference cache: %s", exc)

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        """Use distinct temporary files so concurrent saves cannot steal a writer's file."""
        tmp = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                             suffix=".tmp", delete=False) as handle:
                tmp = Path(handle.name)
                handle.write(text)
            tmp.replace(path)
        finally:
            if tmp is not None:
                with contextlib.suppress(OSError):
                    tmp.unlink()
