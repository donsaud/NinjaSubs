"""Short-lived Display-to-Reference candidate hints.

Display already discovers provider candidates for the same video. Reference
selection may reuse those candidates when the provider fan-out is slow or one
provider times out, but a hint is only a *candidate*: it is downloaded,
extracted, re-scored, Target-Bound, and verified exactly like a fresh result.

The cache is deliberately narrow:

* keyed by the target video identity (not by the selected subtitle);
* keyed by a digest of the provider credentials, so one authorization context
  can never reuse another user's discoveries;
* in-memory and short lived (provider listings are transient);
* never an artifact cache and never a verification claim.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
from collections.abc import Iterable
from typing import Any

from cachetools import TTLCache

from app.models import SubtitleRelease

logger = logging.getLogger(__name__)

#: Display often runs tens of seconds before auto-sync. Ten minutes covers the
#: normal click path without turning a stale listing into a long-lived cache.
DEFAULT_TTL_SECONDS = 600.0
#: One entry is a whole listing; the value is small compared with artifacts.
DEFAULT_MAX_ENTRIES = 256

_HINTS: TTLCache[str, tuple[SubtitleRelease, ...]] = TTLCache(
    maxsize=DEFAULT_MAX_ENTRIES, ttl=DEFAULT_TTL_SECONDS
)


def _text(value: object) -> str:
    return str(value or "").strip()


def authorization_digest(
    *,
    subdl_key: str | None = None,
    subsource_key: str | None = None,
    opensubtitles_key: str | None = None,
) -> str:
    """Digest the effective provider credentials without retaining them."""
    material = "\x1f".join(
        (subdl_key or "", subsource_key or "", opensubtitles_key or "")
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def video_scope_key(
    *,
    imdb_id: str | None,
    media_type: str | None,
    season: int | str | None,
    episode: int | str | None,
    target_filename: str | None,
    video_hash: str | None = None,
    video_size: int | str | None = None,
) -> str:
    """Identity of the target video, independent of the chosen subtitle."""
    identity = json.dumps(
        [
            _text(imdb_id).lower(),
            _text(media_type).lower(),
            _text(season),
            _text(episode),
            re.sub(r"\s+", " ", _text(target_filename).lower()),
            _text(video_hash).lower(),
            _text(video_size),
        ],
        ensure_ascii=False,
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]


def _hint_key(video_key: str, auth_digest: str) -> str:
    return f"{video_key}:{auth_digest}"


def _snapshot(release: Any) -> SubtitleRelease | None:
    """Copy only the provider facts Reference needs; never capture user state."""
    url = _text(getattr(release, "download_url", None))
    name = _text(getattr(release, "release_name", None))
    provider = _text(getattr(release, "provider", None)).lower()
    if not url or not name or provider not in {"subdl", "subsource", "opensubtitles"}:
        return None
    evidence = getattr(release, "release_evidence", None)
    return SubtitleRelease(
        release_name=name,
        download_url=url,
        provider=provider,
        format=_text(getattr(release, "format", "srt")) or "srt",
        hearing_impaired=bool(getattr(release, "hearing_impaired", False)),
        lang=_text(getattr(release, "lang", "")) or "ara",
        uploader=_text(getattr(release, "uploader", "")),
        release_evidence=copy.deepcopy(evidence) if isinstance(evidence, list) else [],
    )


def register(
    video_key: str,
    auth_digest: str,
    releases: Iterable[Any],
) -> int:
    """Remember a Display listing as candidate hints.

    Returns the number of distinct provider candidates stored. Content is not
    trusted, downloaded, or classified here.
    """
    if not video_key or not auth_digest:
        return 0
    snapshots: list[SubtitleRelease] = []
    seen: set[str] = set()
    for release in releases or ():
        hint = _snapshot(release)
        if hint is None or hint.download_url in seen:
            continue
        seen.add(hint.download_url)
        snapshots.append(hint)
    if not snapshots:
        return 0
    _HINTS[_hint_key(video_key, auth_digest)] = tuple(snapshots)
    logger.info(
        "[reference-hints] stored %d Display candidate(s) for video=%s",
        len(snapshots),
        video_key[:12],
    )
    return len(snapshots)


def lookup(video_key: str, auth_digest: str) -> list[SubtitleRelease]:
    """Return candidate hints for this video + authorization context.

    The returned releases are copies; callers may attach evidence or mutate
    ranking fields without affecting the stored listing. No artifact is
    returned and no validation is implied.
    """
    if not video_key or not auth_digest:
        return []
    stored = _HINTS.get(_hint_key(video_key, auth_digest))
    if not stored:
        return []
    return [hint.model_copy(deep=True) for hint in stored]


def clear() -> None:
    """Test/support hook: drop every in-process hint."""
    _HINTS.clear()
