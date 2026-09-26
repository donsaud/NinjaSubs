"""Reference lookup parameters shared by all sync strategies."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

from app.services.sync.matching import _release_group


@dataclass
class ReferenceQuery:
    """Parameters used to locate a matching reference subtitle."""

    imdb_id: str
    target_filename: str | None = None
    release_group: str | None = None
    media_type: str = "movie"
    title: str | None = None
    year: int | None = None
    video_hash: str | None = None
    video_size: int | str | None = None
    stream_url: str | None = None
    season: int | None = None
    episode: int | None = None
    api_keys: dict[str, str] = field(default_factory=dict)
    languages: tuple[str, ...] = ("eng",)

    @property
    def is_series(self) -> bool:
        return self.media_type == "series" or self.season is not None

    @property
    def effective_group(self) -> str:
        """Release group for cache identity: explicit field, else filename-derived."""
        group = self.release_group or _release_group(self.target_filename) or "unknown"
        return re.sub(r"[^A-Za-z0-9]+", "", group) or "unknown"

    @property
    def cache_stem(self) -> str:
        """Versioned identity bound to the video, filename, and reference language.

        A release group alone cannot distinguish that group's WEB, disc,
        extended, or remastered releases. The digest also prevents old,
        group-only cache entries from being reused as exact references.
        """
        imdb = re.sub(r"[^A-Za-z0-9]+", "", self.imdb_id or "unknown") or "unknown"
        season = str(self.season) if self.season is not None else "movie"
        episode = str(self.episode) if self.episode is not None else "x"
        identity = json.dumps([
            self.media_type.lower(),
            re.sub(r"\s+", " ", (self.target_filename or "").strip().lower()),
            (self.video_hash or "").strip().lower(),
            str(self.video_size or ""),
            (self.stream_url or "").strip(),
            self.languages,
        ], ensure_ascii=False)
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        return f"{imdb}_{season}_{episode}_{self.effective_group}_v2_{digest}"


@dataclass(frozen=True)
class ResolvedReference:
    """A strategy's verdict: subtitle text plus what was proven about it.

    ``kind`` is one of ``hash`` / ``embedded`` / ``team`` / ``edition`` /
    ``abort``. ``bluray_match`` is True only when the target and reference
    sources were both verified as BluRay; combined with series content it
    unlocks the recap allowance in the shift guardrail (recap cuts shift
    whole episodes by tens of seconds without any mistiming). ``candidate``
    names the winning release for logs and cache sidecars.
    """

    text: str | None
    kind: str = "abort"
    bluray_match: bool = False
    candidate: str = ""
    # A sampled/partial reference (e.g. only the first 15 min extracted from a
    # remote stream). Its end runtime is deliberately shorter than the target,
    # so duration-mismatch gates must not reject it.
    partial: bool = False
