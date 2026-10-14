"""Demand-driven season-pack reference reuse across same-lineage episodes.

When episode N's sync downloads a season-pack archive, the pack is indexed
once and its episode members retained on disk. When episode N+1 arrives from
a video file of the SAME release lineage (series + season + source + group),
its member is served locally with zero provider search and zero downloads.
Timing suitability is still established independently per episode: this tier
only skips discovery, never verification.

Strict rules, all fail-closed:

* Unknown video lineage (missing source or group on either side) never
  matches. Unknown is not equivalent to a match.
* Different source, different group, different series/season/language, or a
  missing episode member → ``None`` (normal discovery proceeds).
* Only ZIP archives are retained natively (no new extractor dependencies);
  anything else is ignored. Raw archives are size-capped, member-capped,
  TTL-expired, and prune-bounded.
* No prefetch: records are created only from downloads the current request
  already needed. The first episode is never delayed for pack acquisition.
* Zip-Slip names, absolute paths, and non-subtitle members are never stored.

Storage reuses the reference-cache root (``REFERENCE_CACHE_DIR``,
else ``CACHE_DIR/references``) under ``season_packs/<key>/`` with an
``index.json`` sidecar — the same directory/TTL discipline as the
video-scoped artifact tier, but a separate key domain and lifecycle.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import time
import zipfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Refuse archives larger than this (a subtitle season pack is ~1MB; anything
#: bigger is not worth retaining for latency).
MAX_PACK_BYTES = 8_000_000
#: Refuse packs with more subtitle members than this (not a season pack).
MAX_PACK_MEMBERS = 40
#: Oldest-first prune bound on retained packs.
MAX_PACKS = 32
#: Pack records expire like video-scoped artifacts.
PACK_TTL_SECONDS = 30 * 24 * 3600

_SUBTITLE_EXTENSIONS = (".srt", ".ass", ".ssa", ".vtt")
_ZIP_MAGIC = b"PK\x03\x04"


def video_lineage(target_filename: str | None) -> dict[str, Any] | None:
    """Release lineage of a video file: normalized ``source`` + ``group``.

    Returns ``None`` when there is no filename at all. Individual fields
    may be ``None`` (unknown); callers must treat unknown as non-matching.
    """
    if not target_filename:
        return None
    from app.services.sync.matching import (
        _release_group,
        _source_kind,
        normalize_release_group,
    )
    from app.services.sync.release_metadata import parse_identity
    identity = parse_identity(target_filename)
    return {
        "source": _source_kind(target_filename),
        "group": normalize_release_group(_release_group(target_filename)) or None,
        "cut": list(identity.cut_tags),
        "edition": list(identity.edition_tags),
    }


def lineages_match(
    stored: dict[str, Any] | None, current: dict[str, Any] | None
) -> bool:
    """True only when both lineages are fully known and equal.

    Source AND group must match (e.g. ROVERS BluRay E01 vs ROVERS BluRay
    E02). A WEB-DL-NTb video never matches a ROVERS-BluRay record, and an
    unknown on either side never matches anything.
    """
    if not stored or not current:
        return False
    for field in ("source", "group"):
        left = (stored.get(field) or "").strip().lower()
        right = (current.get(field) or "").strip().lower()
        if not left or not right or left != right:
            return False
    if sorted(stored.get("cut") or []) != sorted(current.get("cut") or []):
        return False
    if sorted(stored.get("edition") or []) != sorted(current.get("edition") or []):
        return False
    return True


def pack_root(root: str | Path | None = None) -> Path:
    """Reference-cache root, same resolution as the video-scoped tier."""
    from app.config import settings

    if root is not None:
        return Path(root)
    configured = getattr(settings, "REFERENCE_CACHE_DIR", None)
    if configured:
        return Path(configured)
    return Path(settings.CACHE_DIR) / "references"


def _safe_token(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(value or "")) or "unknown"


def pack_key(
    *,
    imdb_id: str | None,
    season: int | None,
    lang: str | None,
    lineage: dict[str, Any] | None,
    provider: str | None,
    pack_digest: str | None,
) -> str:
    """Filesystem-safe identity for one retained season pack."""
    lineage_fp = hashlib.sha256(json.dumps({
        "source": (lineage or {}).get("source"),
        "group": (lineage or {}).get("group"),
        "cut": (lineage or {}).get("cut", []),
        "edition": (lineage or {}).get("edition", []),
    }, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return ".".join([
        _safe_token(imdb_id),
        f"s{season}" if season is not None else "sx",
        _safe_token(lang or "und"),
        f"lin{lineage_fp}",
        _safe_token(provider),
        f"pack{(pack_digest or '')[:16] or 'unknown'}",
    ])


def _pack_dir(key: str, root: str | Path | None = None) -> Path:
    return pack_root(root) / "season_packs" / key


def _member_episode(name: str) -> int | None:
    from app.services.sync.matching import _member_episode_number

    try:
        value = _member_episode_number(name)
    except Exception:
        return None
    return value if isinstance(value, int) and value > 0 else None


def _member_season(name: str) -> int | None:
    from app.services.sync.matching import _season_number

    try:
        value = _season_number(name)
    except Exception:
        return None
    return value if isinstance(value, int) and value > 0 else None


def _is_slip(name: str) -> bool:
    from app.extractor import is_zip_slip_attempt

    try:
        return bool(is_zip_slip_attempt(name))
    except Exception:
        return True


def _transcode(raw: bytes) -> str | None:
    from app.extractor import transcode_to_utf8

    try:
        text = transcode_to_utf8(raw).decode("utf-8")
    except Exception:
        return None
    return text if "-->" in text else None


def retain_season_pack(
    *,
    raw: bytes | None,
    provider_label: str | None,
    pack_candidate: str | None,
    source_download_digest: str | None,
    query: Any,
    root: str | Path | None = None,
) -> str | None:
    """Index and retain a downloaded season-pack archive. Never raises.

    Called only with bytes the current request already downloaded for its
    own episode. Returns the pack key, or ``None`` when there is nothing
    worth retaining (not an archive, single member, unparseable episodes,
    unknown lineage, over caps, corrupt).
    """
    try:
        return _retain_season_pack(
            raw=raw, provider_label=provider_label, pack_candidate=pack_candidate,
            source_download_digest=source_download_digest, query=query, root=root)
    except Exception as exc:  # retention must never break sync
        logger.debug("[season-pack] retain skipped: %s", type(exc).__name__)
        return None


def _retain_season_pack(
    *,
    raw: bytes | None,
    provider_label: str | None,
    pack_candidate: str | None,
    source_download_digest: str | None,
    query: Any,
    root: str | Path | None = None,
) -> str | None:
    if not raw or len(raw) > MAX_PACK_BYTES or raw[:4] != _ZIP_MAGIC:
        return None
    season = getattr(query, "season", None)
    episode = getattr(query, "episode", None)
    imdb_id = getattr(query, "imdb_id", None)
    lang = getattr(query, "target_lang", None) or "und"
    if season is None or episode is None:
        return None
    lineage = video_lineage(getattr(query, "target_filename", None))
    if not lineage or not lineage.get("source") or not lineage.get("group"):
        # First-video lineage unknown: nothing safe to key future reuse on.
        return None
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
        infos = archive.infolist()
    except Exception:
        return None
    members: list[tuple[str, bytes]] = []
    for info in infos:
        if info.is_dir():
            continue
        name = info.filename
        if _is_slip(name):
            continue
        if not name.lower().endswith(_SUBTITLE_EXTENSIONS):
            continue
        member_season = _member_season(name)
        if member_season is not None and member_season != season:
            continue
        member_episode = _member_episode(name)
        if member_episode is None:
            continue
        try:
            payload = archive.read(info)
        except Exception:
            continue
        members.append((name, payload))
    if len(members) < 2 or len(members) > MAX_PACK_MEMBERS:
        # Not a pack (single file) or not worth retaining.
        return None
    by_episode: dict[int, tuple[str, bytes]] = {}
    for name, payload in sorted(members, key=lambda item: item[0]):
        episode_no = _member_episode(name)
        if episode_no is None or episode_no in by_episode:
            continue
        by_episode[episode_no] = (name, payload)
    if len(by_episode) < 2:
        return None
    key = pack_key(imdb_id=imdb_id, season=season, lang=lang, lineage=lineage,
                   provider=provider_label, pack_digest=source_download_digest)
    directory = _pack_dir(key, root)
    texts: dict[int, str] = {}
    for episode_no, (_name, payload) in by_episode.items():
        text = _transcode(payload)
        if text is None:
            continue
        texts[episode_no] = text
    if len(texts) < 2:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    previous_index = _read_index(directory) or {}
    atomic: list[Path] = []
    try:
        archive_path = directory / "archive.zip"
        archive_tmp = directory / "archive.zip.tmp"
        archive_tmp.write_bytes(raw)
        archive_tmp.replace(archive_path)
        for episode_no, text in texts.items():
            member_path = directory / f"ep{episode_no:02d}.srt"
            tmp_path = member_path.with_suffix(".srt.tmp")
            tmp_path.write_bytes(text.encode("utf-8"))
            tmp_path.replace(member_path)
            atomic.append(member_path)
        index = {
            "key": key,
            "imdb_id": imdb_id,
            "season": season,
            "lang": lang,
            "video_lineage": lineage,
            "proven": bool(previous_index.get("proven") and
                           previous_index.get("archive_sha256") == hashlib.sha256(raw).hexdigest()),
            "archive_sha256": hashlib.sha256(raw).hexdigest(),
            "provider": provider_label,
            "pack_candidate": pack_candidate,
            "source_download_digest": source_download_digest,
            "episodes": {
                str(episode_no): {
                    "member": name,
                    "content_sha": hashlib.sha256(
                        texts[episode_no].encode("utf-8")).hexdigest(),
                }
                for episode_no, (name, _payload) in sorted(
                    ((ep, by_episode[ep]) for ep in texts), key=lambda item: item[0])
            },
            "created": time.time(),
        }
        index_path = directory / "index.json"
        tmp_index = directory / "index.json.tmp"
        tmp_index.write_text(json.dumps(index), encoding="utf-8")
        tmp_index.replace(index_path)
    except OSError:
        for path in atomic:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        return None
    _prune_packs(root)
    logger.info("[season-pack] retained pack=%s provider=%s episodes=%s",
                key, provider_label, sorted(texts))
    return key


def _read_index(directory: Path) -> dict[str, Any] | None:
    try:
        index_path = directory / "index.json"
        if not index_path.is_file():
            return None
        if time.time() - index_path.stat().st_mtime > PACK_TTL_SECONDS:
            return None
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _prune_packs(root: str | Path | None = None) -> None:
    try:
        base = pack_root(root) / "season_packs"
        if not base.is_dir():
            return
        entries = sorted(
            (entry for entry in base.iterdir() if entry.is_dir()),
            key=lambda entry: entry.stat().st_mtime)
        for stale in entries[: max(0, len(entries) - MAX_PACKS)]:
            for path in sorted(stale.iterdir()):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            try:
                stale.rmdir()
            except OSError:
                pass
    except OSError:
        pass


def lookup_pack_member(query: Any, root: str | Path | None = None) -> dict[str, Any] | None:
    """Find this episode's member in a retained same-lineage pack. Never raises.

    Returns ``{"text", "provider", "candidate", "source_download_digest",
    "member_season", "member_episode"}`` or ``None`` (normal discovery must
    proceed). Pure local reads: no search, no download, no prefetch.
    """
    try:
        return _lookup_pack_member(query, root)
    except Exception as exc:  # lookup must never break sync
        logger.debug("[season-pack] lookup skipped: %s", type(exc).__name__)
        return None


def _lookup_pack_member(query: Any, root: str | Path | None) -> dict[str, Any] | None:
    season = getattr(query, "season", None)
    episode = getattr(query, "episode", None)
    imdb_id = getattr(query, "imdb_id", None)
    if season is None or episode is None or not imdb_id:
        return None
    current = video_lineage(getattr(query, "target_filename", None))
    if not current:
        return None
    base = pack_root(root) / "season_packs"
    if not base.is_dir():
        return None
    for directory in sorted(base.iterdir()):
        if not directory.is_dir():
            continue
        index = _read_index(directory)
        if not index:
            continue
        if index.get("imdb_id") != imdb_id or index.get("season") != season:
            continue
        if not lineages_match(index.get("video_lineage"), current):
            continue
        if not index.get("proven") and not _legacy_winning_seed(index, directory, pack_root(root)):
            continue
        episodes = index.get("episodes") or {}
        entry = episodes.get(str(episode))
        if not isinstance(entry, dict):
            continue
        member_path = directory / f"ep{episode:02d}.srt"
        try:
            text = member_path.read_bytes().decode("utf-8")
        except OSError:
            continue
        if not text or "-->" not in text:
            continue
        content_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if entry.get("content_sha") and entry["content_sha"] != content_sha:
            continue
        logger.info("[season-pack] HIT pack=%s episode=%s member=%s",
                    directory.name, episode, entry.get("member"))
        return {
            "text": text,
            "provider": index.get("provider"),
            "candidate": entry.get("member") or index.get("pack_candidate"),
            "source_download_digest": index.get("source_download_digest"),
            "member_season": season,
            "member_episode": episode,
            "proven_video_lineage": index["video_lineage"],
        }
    return None


def _legacy_winning_seed(index: dict, directory: Path, root: Path) -> bool:
    """Recover an existing verified seed without reacquiring its archive.

    Old retention had no winner marker. A checksum-bound Unified sticky record
    for the same series/season/group and exact source/member supplies that proof.
    This is read-only compatibility; newly acquired packs use promotion directly.
    """
    from app.services.sync.reference_content_identity import identical_content
    from app.services.sync_service import sanitize_subtitle

    prefix = f"{_safe_token(index.get('imdb_id'))}_{index.get('season')}_"
    for sticky_path in root.glob(f"{prefix}*.sticky.json"):
        try:
            sticky = json.loads(sticky_path.read_text(encoding="utf-8"))
            if (sticky.get("verification_method") != "unified"
                    or not str((sticky.get("evidence") or {}).get("sync_state", "")).startswith("verified")):
                continue
            name = sticky.get("artifact", "")
            if not name or Path(name).name != name:
                continue
            metadata = json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
            episode = metadata.get("member_episode")
            expected_stem = f"{prefix}{episode}_{index['video_lineage']['group']}_v2_"
            if (not str(sticky.get("stem", "")).lower().startswith(expected_stem.lower())
                    or metadata.get("member_season") != index.get("season")
                    or metadata.get("source_download_digest") != index.get("source_download_digest")):
                continue
            payload = (root / name).read_bytes()
            if hashlib.sha256(payload).hexdigest() != sticky.get("content_sha"):
                continue
            entry = (index.get("episodes") or {}).get(str(episode))
            if not entry:
                continue
            member = (directory / f"ep{int(episode):02d}.srt").read_bytes()
            if hashlib.sha256(member).hexdigest() != entry.get("content_sha"):
                continue
            if identical_content(sanitize_subtitle(member.decode("utf-8")), payload.decode("utf-8")):
                return True
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return False


def promote_season_pack(query: Any, resolved: Any, root: str | Path | None = None) -> bool:
    """Associate the complete pack only after an independently verified serve.

    The winner's source and exact episode bytes identify the acquired archive;
    merely downloading another pack cannot prove its relation to this video.
    Previously retained indexes can be promoted by their winning cached member.
    """
    from app.services.sync.reference_content_identity import identical_content

    digest = getattr(resolved, "source_download_digest", None)
    if not digest or not getattr(resolved, "text", None):
        return False
    base = pack_root(root) / "season_packs"
    if not base.is_dir():
        return False
    promoted = False
    for directory in base.iterdir():
        index = _read_index(directory)
        if not index or index.get("source_download_digest") != digest:
            continue
        if (index.get("imdb_id") != query.imdb_id or index.get("season") != query.season
                or not lineages_match(index.get("video_lineage"), video_lineage(query.target_filename))):
            continue
        entry = (index.get("episodes") or {}).get(str(query.episode))
        if not entry:
            continue
        try:
            text = (directory / f"ep{query.episode:02d}.srt").read_bytes().decode("utf-8")
            if hashlib.sha256(text.encode()).hexdigest() != entry.get("content_sha"):
                continue
            if not identical_content(text, resolved.text):
                continue
            if not index.get("proven"):
                index.update(proven=True, proven_episode=query.episode, proven_at=time.time())
                temporary = directory / "index.json.tmp"
                temporary.write_text(json.dumps(index), encoding="utf-8")
                temporary.replace(directory / "index.json")
            promoted = True
        except (OSError, ValueError):
            continue
    return promoted


def season_pack_reference(query: Any) -> Any | None:
    """Build a ``ResolvedReference`` from a retained pack member, or ``None``."""
    from app.services.sync.query import ResolvedReference

    found = lookup_pack_member(query)
    if not found or not found.get("text"):
        return None
    return ResolvedReference(
        found["text"],
        kind="edition",
        candidate=str(found.get("candidate") or "season_pack_member"),
        provider=found.get("provider"),
        member_season=found.get("member_season"),
        member_episode=found.get("member_episode"),
        source_download_digest=found.get("source_download_digest"),
        from_season_pack=True,
        proven_video_lineage=found.get("proven_video_lineage"),
    )
