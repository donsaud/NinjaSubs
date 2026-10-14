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
# Version 6 rejects references already sanitized by the pre-safety ASS converter.
# Version 7 marks the video-scoped artifact tier written alongside it.
ENGINE_VERSION = 7

# Video-scoped artifacts are bounded so a long-lived cache directory cannot grow
# without limit. Pruning is oldest-first across the video-scoped namespace only.
MAX_VIDEO_SCOPED_ENTRIES = 512

# Decision kinds that prove the exact edition (vs. best-effort edition timing).
EXACT_KINDS = frozenset({"team", "hash", "embedded"})
_KNOWN_KINDS = ("team", "edition", "hash", "embedded")

#: Filename marker for the video-scoped tier. Distinct from the subtitle-scoped
#: stem so the two namespaces can never collide on disk.
_VIDEO_SCOPE_MARKER = "video"

#: Filename marker for the sticky verified-reference tier. One pointer per
#: exact video stem, so the namespace can never collide with artifact files.
_STICKY_MARKER = "sticky"

#: Sticky pointers are tiny metadata; the bound only stops unbounded growth
#: across videos. Artifacts themselves stay under the video-tier lifecycle.
MAX_STICKY_ENTRIES = 128


def _safe_score(value: object) -> float:
    """Return a numeric artifact score without letting malformed metadata win."""
    try:
        score = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return score if score == score and abs(score) != float("inf") else 0.0


def _cache_member_number(value: object) -> int | str | None:
    """Normalize a stored/requested season/episode value for strict comparison.

    Canonical parser values are strings, while request metadata commonly uses
    integers. Normalizing only decimal representations preserves episode
    binding and lets those two layers agree; anything else compares exactly.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return value if isinstance(value, str | int) or value is None else str(value)


def _video_artifact_sort_key(meta: dict, path_name: str) -> tuple[int, float, float, str]:
    """Order retained artifacts by established verdict strength, then score.

    Exact decision kinds outrank best-effort edition timing. Within the same
    kind, a larger weighted proof score wins. The filename only breaks ties
    deterministically; it never invents a quality claim.
    """
    kind_rank = 0 if meta.get("kind") in EXACT_KINDS else 1
    return (
        kind_rank,
        -_safe_score(meta.get("score")),
        -_safe_score(meta.get("score_without_hash")),
        path_name,
    )


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
            or int(meta.get("engine_version", 1)) < 6
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
                "engine_version": ENGINE_VERSION,
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

    # ------------------------------------------------------------------ #
    # Video-scoped artifact tier
    #
    # Which reference FILE was downloaded is a property of the target video.
    # Whether it synchronizes well is a property of the chosen subtitle. The
    # subtitle-scoped tier above answers the second question; this one answers
    # the first, so switching subtitles reuses the artifact instead of
    # re-downloading the same archive from a rate-limited provider.
    #
    # Nothing here relaxes verification. A reused artifact is still handed to
    # the normal ALASS / affine / target-binding / verifier path, and it is
    # still rejected outright when it cannot serve as independent evidence.
    # ------------------------------------------------------------------ #

    def _video_scope_path(
        self,
        query: ReferenceQuery,
        source: str,
        kind: str,
        artifact_id: str,
    ) -> Path:
        stem = query.video_scope_stem
        safe_artifact = re.sub(r"[^A-Za-z0-9]+", "", artifact_id or "")[:16] or "artifact"
        return self.root / (
            f"{stem}.{_VIDEO_SCOPE_MARKER}.{self._safe_source(source)}.{kind}.{safe_artifact}.srt"
        )

    def _video_scope_paths(self, query: ReferenceQuery) -> list[Path]:
        if not self.root.is_dir():
            return []
        return sorted(self.root.glob(f"{query.video_scope_stem}.{_VIDEO_SCOPE_MARKER}.*.srt"))

    def _prune_video_scoped(self) -> None:
        """Keep the video-scoped namespace bounded, oldest evicted first.

        The globally strongest retained artifact is exempt from pruning: capacity
        control must not discard the best available evidence merely because a
        weaker artifact arrived later.
        """
        try:
            entries = [p for p in self.root.glob(f"*.{_VIDEO_SCOPE_MARKER}.*.srt") if p.is_file()]
        except OSError:
            return
        if len(entries) <= MAX_VIDEO_SCOPED_ENTRIES:
            return
        ranked: list[tuple[tuple[int, float, float, str], Path]] = []
        for path in entries:
            try:
                meta = json.loads(self._sidecar_path(path).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                meta = {}
            if not isinstance(meta, dict):
                meta = {}
            ranked.append((_video_artifact_sort_key(meta, path.name), path))
        ranked.sort(key=lambda item: item[0])
        # Preserve the single strongest entry; evict the oldest of the rest.
        candidates = [path for _, path in ranked[1:]]
        candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
        for stale in candidates[: len(entries) - MAX_VIDEO_SCOPED_ENTRIES]:
            self._remove(stale)

    def get_video_scoped(
        self, query: ReferenceQuery
    ) -> tuple[ResolvedReference, dict] | None:
        """Return a reusable reference artifact for this target VIDEO, or ``None``.

        Validation boundaries, all of which fail closed to ``None``:

        * the entry must live under this video's stem, so a different imdb id,
          season, episode, release group, target filename, video hash/size or
          language set can never share it;
        * the payload digest must match the recorded provenance, and the
          payload must still be a usable subtitle;
        * ``member_season`` / ``member_episode`` must equal the requested
          episode, so an unresolved season pack can never pose as a validated
          episode reference;
        * the engine version and strict-match configuration must still match.

        Note what is deliberately NOT checked here: whether the artifact's
        timing already agrees with the target subtitle. A REMUX reference
        legitimately disagrees with an HDTV subtitle -- that disagreement is
        precisely what Auto-Sync exists to correct. Synchronization quality is
        decided downstream by the real verifier, never by this cache.
        """
        best: tuple[tuple[int, float, float, str], str, ResolvedReference, dict] | None = None
        for path in self._video_scope_paths(query):
            try:
                age = time.time() - path.stat().st_mtime
            except OSError:
                continue
            if age > self.ttl:
                if self._remove(path):
                    logger.info("[reference] expired video-scoped entry removed: %s", path.name)
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            if not (text and len(text.encode("utf-8")) > self.min_bytes and "-->" in text):
                continue
            meta_path = self._sidecar_path(path)
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(meta, dict):
                continue
            kind = self._safe_kind(meta.get("kind"))
            if (
                meta.get("scope") != _VIDEO_SCOPE_MARKER
                or meta.get("stem") != query.video_scope_stem
                or meta.get("kind") != kind
                or int(meta.get("engine_version", 0)) < ENGINE_VERSION
                or meta.get("content_sha") != hashlib.sha256(text.encode("utf-8")).hexdigest()
                or (getattr(settings, "SYNC_REQUIRE_EXACT_MATCH", True) and meta.get("strict") is not True)
            ):
                continue
            # An artifact is only reusable when the member it actually proves is
            # the member that was requested. Never promote an unresolved pack.
            if (
                _cache_member_number(meta.get("member_season")) != _cache_member_number(query.season)
                or _cache_member_number(meta.get("member_episode")) != _cache_member_number(query.episode)
            ):
                continue
            resolved = ResolvedReference(
                text=text,
                kind=kind,
                bluray_match=meta.get("bluray_match") is True and kind != "abort",
                candidate=str(meta.get("candidate", "")),
                provider=str(meta.get("provider", "")) or None,
                member_season=meta.get("member_season"),
                member_episode=meta.get("member_episode"),
                weighted_score=_safe_score(meta.get("score")),
                weighted_score_without_hash=_safe_score(meta.get("score_without_hash")),
                source_download_digest=str(meta.get("source_download_digest") or "") or None,
                from_video_cache=True,
                partial=meta.get("partial") is True,
            )
            key = _video_artifact_sort_key({**meta, "kind": kind}, path.name)
            if best is None or key < best[0]:
                best = (key, path.name, resolved, meta)
        if best is None:
            return None
        _, name, resolved, meta = best
        logger.info(
            "[reference] video-scoped cache hit: file=%s provider=%s candidate=%s "
            "season=%s episode=%s bytes=%d",
            name,
            meta.get("provider", "?"),
            meta.get("candidate", "?"),
            meta.get("member_season", "?"),
            meta.get("member_episode", "?"),
            len(resolved.text.encode("utf-8")) if resolved.text else 0,
        )
        return resolved, meta

    def set_video_scoped(
        self,
        query: ReferenceQuery,
        source: str | None,
        text: str,
        kind: str = "edition",
        *,
        bluray_match: bool = False,
        candidate: str = "",
        partial: bool = False,
        member_season: int | str | None = None,
        member_episode: int | str | None = None,
        score: float = 0.0,
        score_without_hash: float = 0.0,
        source_download_digest: str | None = None,
    ) -> bool:
        """Persist a validated reference artifact against the target VIDEO.

        The write is refused unless ``member_season`` and ``member_episode``
        are both present and equal the requested episode. That is what keeps an
        unresolved season pack -- one whose correct member has not been
        extracted and re-scored -- from being stored as though it were an
        already validated episode reference.
        """
        if not text:
            return False
        member_season = _cache_member_number(member_season)
        member_episode = _cache_member_number(member_episode)
        if query.season is not None and member_season != _cache_member_number(query.season):
            return False
        if query.episode is not None and member_episode != _cache_member_number(query.episode):
            return False
        if query.episode is not None and member_episode is None:
            return False
        kind = self._safe_kind(kind)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            content_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            # Distinct provider/kind entries are already separated by filename.
            # Distinct candidates and payloads receive distinct artifact files as
            # well, so storing a weaker reference can never silently replace a
            # stronger validated reference that happens to share its source.
            artifact_id = hashlib.sha256(
                json.dumps([
                    self._safe_source(source),
                    candidate,
                    content_sha,
                ]).encode("utf-8")
            ).hexdigest()[:16]
            path = self._video_scope_path(query, source or "ref", kind, artifact_id)
            self._atomic_write(path, text)
            meta = {
                "scope": _VIDEO_SCOPE_MARKER,
                "stem": query.video_scope_stem,
                "kind": kind,
                "provider": self._safe_source(source),
                "candidate": candidate,
                "partial": bool(partial),
                "bluray_match": bool(bluray_match),
                "member_season": member_season,
                "member_episode": member_episode,
                "score": _safe_score(score),
                "score_without_hash": _safe_score(score_without_hash),
                "source_download_digest": source_download_digest,
                "languages": list(query.languages),
                "strict": bool(getattr(settings, "SYNC_REQUIRE_EXACT_MATCH", True)),
                "content_sha": content_sha,
                "engine_version": ENGINE_VERSION,
            }
            self._atomic_write(self._sidecar_path(path), json.dumps(meta, sort_keys=True))
            self._prune_video_scoped()
            logger.info(
                "[reference] cached video-scoped reference: provider=%s candidate=%s "
                "season=%s episode=%s bytes=%d",
                meta["provider"],
                candidate or "?",
                member_season,
                member_episode,
                len(text.encode("utf-8")),
            )
            return True
        except OSError as exc:
            logger.warning("[reference] failed to write video-scoped reference cache: %s", exc)
            return False

    # ---------------- sticky verified reference (one pointer per video) --- #

    def _sticky_path(self, query: ReferenceQuery) -> Path:
        """Pointer file for this exact video stem (never an artifact)."""
        return self.root / f"{query.video_scope_stem}.{_STICKY_MARKER}.json"

    def _prune_sticky(self) -> None:
        """Keep the sticky namespace bounded, oldest evicted first.

        Only pointers are evicted here; artifact bytes stay under the
        video-tier lifecycle so other tiers keep working.
        """
        try:
            entries = [p for p in self.root.glob(f"*.{_STICKY_MARKER}.json") if p.is_file()]
        except OSError:
            return
        if len(entries) <= MAX_STICKY_ENTRIES:
            return
        entries.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
        for stale in entries[: len(entries) - MAX_STICKY_ENTRIES]:
            try:
                stale.unlink(missing_ok=True)
            except OSError:
                pass

    def _sticky_methodology(self) -> tuple[str, str]:
        """Current verification methodology; sticky records are bound to it."""
        try:
            from app.services.sync.unified_verification import METHOD, VERSION

            return str(METHOD), str(VERSION)
        except Exception:
            return "unified", "timing-v3"

    def set_sticky_reference(
        self,
        query: ReferenceQuery,
        *,
        provider: str | None,
        candidate: str = "",
        kind: str = "edition",
        text: str,
        member_season: int | str | None = None,
        member_episode: int | str | None = None,
        score: float = 0.0,
        score_without_hash: float = 0.0,
        source_download_digest: str | None = None,
        bluray_match: bool = False,
        partial: bool = False,
        evidence: dict | None = None,
    ) -> bool:
        """Promote a verified, deliverable reference as this video's sticky first choice.

        Only ever called on the TRANSFORMED serve path, after identity,
        distinctness, Timing-First verification, reconstruction, binding,
        may_serve, and delivery-eligibility all passed. Last verified serve
        wins (replacement); failures never reach this method, so one bad
        request can neither erase nor oscillate the preference.

        The artifact bytes are reused when already present under this stem
        (matched by content SHA); otherwise they are written through the
        same :meth:`set_video_scoped` gate, which refuses unproven members.
        The pointer records methodology, so a methodology change can never
        reinterpret an old promotion.
        """
        if not text:
            return False
        member_season = _cache_member_number(member_season)
        member_episode = _cache_member_number(member_episode)
        if query.season is not None and member_season != _cache_member_number(query.season):
            return False
        if query.episode is not None and member_episode != _cache_member_number(query.episode):
            return False
        if query.episode is not None and member_episode is None:
            return False
        kind = self._safe_kind(kind)
        method, version = self._sticky_methodology()
        try:
            evidence_json = json.dumps(evidence or {}, sort_keys=True)
        except (TypeError, ValueError):
            return False
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            content_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            path = self._sticky_artifact_path(query, content_sha)
            if path is None:
                wrote = self.set_video_scoped(
                    query, provider, text, kind=kind, bluray_match=bluray_match,
                    candidate=candidate, partial=partial,
                    member_season=member_season, member_episode=member_episode,
                    score=score, score_without_hash=score_without_hash,
                    source_download_digest=source_download_digest,
                )
                if not wrote:
                    return False
                path = self._sticky_artifact_path(query, content_sha)
            if path is None:
                return False
            pointer = self._sticky_path(query)
            replaced = pointer.is_file()
            record = {
                "scope": _STICKY_MARKER,
                "stem": query.video_scope_stem,
                "artifact": path.name,
                "content_sha": content_sha,
                "provider": self._safe_source(provider),
                "candidate": candidate,
                "kind": kind,
                "verification_method": method,
                "verification_version": version,
                "evidence": json.loads(evidence_json),
                "promoted_at": time.time(),
            }
            self._atomic_write(pointer, json.dumps(record, sort_keys=True))
            self._prune_sticky()
            logger.info(
                "[reference-sticky] promoted=%s replaced=%s video=%s provider=%s "
                "candidate=%s sha=%s",
                True, replaced, query.video_scope_stem,
                self._safe_source(provider), candidate or "?",
                content_sha[:12],
            )
            return True
        except OSError as exc:
            logger.warning("[reference-sticky] promotion write failed: %s", exc)
            return False

    def _sticky_artifact_path(
        self, query: ReferenceQuery, content_sha: str
    ) -> Path | None:
        """Existing video-tier artifact file with these exact bytes, if any."""
        if not self.root.is_dir():
            return None
        for path in sorted(self.root.glob(f"{query.video_scope_stem}.{_VIDEO_SCOPE_MARKER}.*.srt")):
            try:
                meta = json.loads(self._sidecar_path(path).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(meta, dict) and meta.get("content_sha") == content_sha:
                return path
        return None

    def get_sticky_reference(
        self, query: ReferenceQuery
    ) -> ResolvedReference | None:
        """Return this video's sticky verified reference artifact, or ``None``.

        Validation boundaries, all fail closed (a bad pointer is deleted, the
        artifact files themselves are left to the video-tier lifecycle):

        * pointer exists, parses, is fresh (TTL), and names this exact stem;
        * methodology still matches, so an old generation can never be
          reinterpreted as a current promotion;
        * the artifact file exists with valid subtitle bytes whose SHA matches
          the pointer AND its own sidecar, with a matching engine version and
          the requested episode member.

        Timing agreement is deliberately NOT checked: the reference is only
        ever attempted first; synchronization quality is decided downstream
        by the real verifier on every request.
        """
        stem = query.video_scope_stem
        pointer = self._sticky_path(query)
        if not pointer.is_file():
            logger.debug("[reference-sticky] lookup=not_found video=%s", stem)
            return None
        try:
            age = time.time() - pointer.stat().st_mtime
        except OSError:
            return None
        if age > self.ttl:
            self._remove_sticky(pointer)
            logger.info("[reference-sticky] lookup=expired video=%s", stem)
            return None
        try:
            record = json.loads(pointer.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._remove_sticky(pointer)
            logger.info("[reference-sticky] lookup=artifact_invalid video=%s reason=unparseable", stem)
            return None
        if not isinstance(record, dict) or record.get("stem") != stem:
            self._remove_sticky(pointer)
            logger.info("[reference-sticky] lookup=scope_mismatch video=%s", stem)
            return None
        method, version = self._sticky_methodology()
        if (record.get("verification_method") != method
                or record.get("verification_version") != version):
            self._remove_sticky(pointer)
            logger.info("[reference-sticky] lookup=methodology_mismatch video=%s", stem)
            return None
        artifact_name = str(record.get("artifact") or "")
        if not artifact_name or "/" in artifact_name or "\\" in artifact_name:
            self._remove_sticky(pointer)
            logger.info("[reference-sticky] lookup=artifact_invalid video=%s reason=bad_name", stem)
            return None
        path = self.root / artifact_name
        if not path.is_file() or not str(path.name).startswith(stem + "."):
            self._remove_sticky(pointer)
            logger.info("[reference-sticky] lookup=artifact_invalid video=%s reason=missing", stem)
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            self._remove_sticky(pointer)
            logger.info("[reference-sticky] lookup=artifact_invalid video=%s reason=unreadable", stem)
            return None
        if not (text and len(text.encode("utf-8")) > self.min_bytes and "-->" in text):
            self._remove_sticky(pointer)
            logger.info("[reference-sticky] lookup=artifact_invalid video=%s reason=unusable", stem)
            return None
        content_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if content_sha != record.get("content_sha"):
            self._remove_sticky(pointer)
            logger.info("[reference-sticky] lookup=artifact_invalid video=%s reason=sha_mismatch", stem)
            return None
        try:
            meta = json.loads(self._sidecar_path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        if not isinstance(meta, dict):
            meta = {}
        if (meta.get("content_sha") != content_sha
                or int(meta.get("engine_version", 0)) < ENGINE_VERSION
                or (getattr(settings, "SYNC_REQUIRE_EXACT_MATCH", True)
                    and meta.get("strict") is not True)):
            self._remove_sticky(pointer)
            logger.info("[reference-sticky] lookup=artifact_invalid video=%s reason=sidecar", stem)
            return None
        if (
            _cache_member_number(meta.get("member_season")) != _cache_member_number(query.season)
            or _cache_member_number(meta.get("member_episode")) != _cache_member_number(query.episode)
        ):
            self._remove_sticky(pointer)
            logger.info("[reference-sticky] lookup=scope_mismatch video=%s reason=episode", stem)
            return None
        kind = self._safe_kind(meta.get("kind") or record.get("kind"))
        logger.info(
            "[reference-sticky] lookup=hit video=%s provider=%s candidate=%s sha=%s",
            stem, meta.get("provider", "?"), meta.get("candidate", "?"),
            content_sha[:12],
        )
        return ResolvedReference(
            text=text,
            kind=kind,
            bluray_match=meta.get("bluray_match") is True and kind != "abort",
            candidate=str(meta.get("candidate", "")),
            provider=str(meta.get("provider", "")) or None,
            member_season=meta.get("member_season"),
            member_episode=meta.get("member_episode"),
            weighted_score=_safe_score(meta.get("score")),
            weighted_score_without_hash=_safe_score(meta.get("score_without_hash")),
            source_download_digest=str(meta.get("source_download_digest") or "") or None,
            from_sticky_reference=True,
            partial=meta.get("partial") is True,
        )

    @staticmethod
    def _remove_sticky(pointer: Path) -> None:
        """Delete a bad pointer; artifact bytes stay for the video tier."""
        try:
            pointer.unlink(missing_ok=True)
        except OSError:
            pass
