"""Reference lookup parameters shared by all sync strategies."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

from app.services.sync.matching import _release_group

_TARGET_CUE_RANGE_RE = re.compile(
    r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})[,.](\d{1,3})"
    r"\s*-->\s*"
    r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})[,.](\d{1,3})"
)
_TARGET_CUE_SAMPLE = 8


def _cue_timestamp_ms(hours: str | None, minutes: str, seconds: str, millis: str) -> int | None:
    """Convert one timestamp to milliseconds, or ``None`` when out of range."""
    try:
        total = (
            int(hours or 0) * 3_600_000
            + int(minutes) * 60_000
            + int(seconds) * 1_000
            + int(millis.ljust(3, "0"))
        )
    except (TypeError, ValueError):
        return None
    if int(minutes) > 59 or int(seconds) > 59:
        return None
    return total


def fingerprint_target_cues(target_text: str | bytes | None) -> str | None:
    """Fingerprint a target subtitle's initial cue timings, ignoring dialogue text.

    Only timing anchors matter for reference reuse: two subtitle IDs with the
    same initial cue grid may safely share a validated reference, while any
    timing difference must isolate their cache entries. Cue numbers, speaker
    text, whitespace, and SRT/VTT timestamp punctuation do not affect the digest.
    """
    if target_text is None:
        return None
    if isinstance(target_text, bytes):
        text = target_text.decode("utf-8", errors="ignore")
    else:
        text = target_text
    pairs: list[str] = []
    for match in _TARGET_CUE_RANGE_RE.finditer(text.lstrip("\ufeff")):
        start = _cue_timestamp_ms(match.group(1), match.group(2), match.group(3), match.group(4))
        end = _cue_timestamp_ms(match.group(5), match.group(6), match.group(7), match.group(8))
        if start is None or end is None or end <= start:
            continue
        pairs.append(f"{start}-{end}")
        if len(pairs) >= _TARGET_CUE_SAMPLE:
            break
    if not pairs:
        return None
    return hashlib.sha256("|".join(pairs).encode("utf-8")).hexdigest()[:16]


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
    target_download_url: str | None = None
    target_sub_id: str | None = None
    target_sub_release_name: str | None = None
    target_cue_digest: str | None = None

    @property
    def is_series(self) -> bool:
        return (
            str(self.media_type or "").lower() in ("series", "anime", "tv")
            or self.season is not None
            or self.episode is not None
        )

    @property
    def effective_group(self) -> str:
        """Release group for cache identity: explicit field, else filename-derived."""
        group = self.release_group or _release_group(self.target_filename) or "unknown"
        return re.sub(r"[^A-Za-z0-9]+", "", group) or "unknown"

    @staticmethod
    def _target_scope(target_cue_digest: str | None, target_sub_id: str | None) -> str:
        """Scope a cache stem to the target subtitle's cue layout when known.

        A matching cue-layout digest allows reference reuse across subtitle IDs
        with identical timing. A subtitle ID alone is a safe fallback scope when
        cues cannot be fingerprinted; an empty scope preserves legacy stems for
        queries that predate target-aware caching.
        """
        digest = re.sub(r"[^0-9a-f]+", "", (target_cue_digest or "").strip().lower())
        if digest:
            return f"t{digest[:16]}"
        sub_id = (target_sub_id or "").strip().lower()
        if sub_id:
            return "i" + hashlib.sha256(sub_id.encode("utf-8")).hexdigest()[:12]
        return ""

    @property
    def cache_stem(self) -> str:
        """Versioned identity bound to the video and the target cue layout.

        A release group alone cannot distinguish that group's WEB, disc,
        extended, or remastered releases. The digest also prevents old,
        group-only cache entries from being reused as exact references. When the
        target subtitle's initial cue timings are known, they form the final
        stem segment; a reference proven against one cue layout can then be
        reused for an identical layout but never for a different one.
        """
        imdb = re.sub(r"[^A-Za-z0-9]+", "", self.imdb_id or "unknown") or "unknown"
        season = str(self.season) if self.season is not None else "movie"
        episode = str(self.episode) if self.episode is not None else "x"
        # NB: stream_url is deliberately excluded — debrid/AIOStreams hand out a
        # freshly-signed token per request, which would otherwise change the
        # stem every time and defeat the reference cache.
        identity = json.dumps([
            self.media_type.lower(),
            re.sub(r"\s+", " ", (self.target_filename or "").strip().lower()),
            (self.video_hash or "").strip().lower(),
            str(self.video_size or ""),
            self.languages,
        ], ensure_ascii=False)
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        stem = f"{imdb}_{season}_{episode}_{self.effective_group}_v2_{digest}"
        scope = self._target_scope(self.target_cue_digest, self.target_sub_id)
        return f"{stem}_{scope}" if scope else stem


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
