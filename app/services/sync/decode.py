"""Shared payload decoding for reference subtitles (ZIP-aware)."""

from __future__ import annotations

import io
import logging
import re
import zipfile

from app.services.sync.matching import (
    _is_info_member,
    _member_episode_number,
    _season_number,
)

logger = logging.getLogger(__name__)

_SDH_MEMBER_PATTERN = re.compile(r"(?i)(?:^|[\s._\-\[])(?:sdh|cc|hi)(?:$|[\s._\-\]])|hearing[\s._-]?impaired")


def _is_sdh_member(name: str) -> bool:
    """True when a ZIP member filename indicates Hearing Impaired / SDH content."""
    base = name.rsplit("/", 1)[-1].strip()
    return bool(_SDH_MEMBER_PATTERN.search(base))


_CUE_TIMING_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)

# A genuine single-episode subtitle is monotonically increasing. A concatenated
# season pack restarts near 00:00 for every episode it contains, which surfaces
# as a large backward jump in the cue timeline.
_CUMULATIVE_BACKWARD_JUMP_MS = 60_000
# Single episodes run well under three hours; a multi-hour span on an episode
# query means several episodes were merged into one continuous timeline.
_CUMULATIVE_MAX_SPAN_MS = 3 * 60 * 60 * 1000


def _ms(hours: str, minutes: str, seconds: str, fraction: str) -> int:
    return (
        int(hours) * 3_600_000
        + int(minutes) * 60_000
        + int(seconds) * 1000
        + int(fraction.ljust(3, "0")[:3])
    )


def cue_times_ms(text: str) -> list[tuple[int, int]]:
    """Return ``(start_ms, end_ms)`` for every cue found in ``text``."""
    return [
        (
            _ms(m.group(1), m.group(2), m.group(3), m.group(4)),
            _ms(m.group(5), m.group(6), m.group(7), m.group(8)),
        )
        for m in _CUE_TIMING_RE.finditer(text or "")
    ]


def looks_like_cumulative_pack(
    text: str,
    *,
    backward_jump_ms: int = _CUMULATIVE_BACKWARD_JUMP_MS,
    max_span_ms: int = _CUMULATIVE_MAX_SPAN_MS,
) -> bool:
    """True when one subtitle file is really several episodes concatenated.

    Season *archives* are legitimate because each episode is extracted on its own;
    what must never reach ``alass`` is an uncompressed cumulative file whose
    timeline runs on past the end of one episode into the next. Two signals are
    used: the timeline jumping backwards (each episode restarts near zero), and
    an implausibly long total span.
    """
    timings = cue_times_ms(text)
    if len(timings) < 4:
        return False
    highest = timings[0][0]
    for start, _end in timings[1:]:
        if start < highest - backward_jump_ms:
            return True
        highest = max(highest, start)
    span = max(end for _start, end in timings) - timings[0][0]
    return span > max_span_ms



def select_zip_member(
    members: list[tuple[str, int]],
    season: int | None,
    episode: int | None,
    min_bytes: int,
) -> str | None:
    """Pick the best subtitle member from a (possibly season-pack) ZIP.

    Strictly matches the requested season (and episode) when the member names
    carry those tags; it never falls back to a different season's episode.
    """
    subtitles = [
        (name, size)
        for name, size in members
        if name.lower().rsplit(".", 1)[-1] in ("srt", "vtt", "ass", "ssa", "sub", "smi")
    ]
    if not subtitles:
        return None

    # Ignore intro/nfo/credit files and anything below the reference size floor.
    usable = [
        (name, size)
        for name, size in subtitles
        if size >= min_bytes and not _is_info_member(name)
    ]
    pool = usable or subtitles

    # Season: keep only the requested season, allowing names with no season tag.
    if season is not None:
        season_matched = [
            (name, size) for name, size in pool if _season_number(name) == season
        ]
        if season_matched:
            pool = season_matched
        else:
            untagged = [(name, size) for name, size in pool if _season_number(name) is None]
            if not untagged:
                logger.warning(
                    "[reference] rejecting ZIP: no member matches season %s", season
                )
                return None
            pool = untagged

    # Episode: keep only the requested episode; if all candidates carry a
    # different episode tag, reject rather than aligning the wrong episode.
    if episode is not None:
        episode_matched = [
            (name, size) for name, size in pool if _member_episode_number(name) == episode
        ]
        if episode_matched:
            pool = episode_matched
        elif any(_member_episode_number(name) is not None for name, _ in pool):
            logger.warning(
                "[reference] rejecting ZIP: no member matches season %s episode %s",
                season,
                episode,
            )
            return None

    # Non-SDH dialogue first, then .srt format, then largest size.
    pool.sort(
        key=lambda item: (
            _is_sdh_member(item[0]),
            not item[0].lower().endswith(".srt"),
            -item[1],
        )
    )
    return pool[0][0] if pool else None


def decode_payload(
    raw: bytes,
    release_name: str,
    season: int | None = None,
    episode: int | None = None,
    min_bytes: int = 5120,
) -> bytes | None:
    """Return subtitle bytes from a raw download, extracting ZIP/RAR/7z/tar archives."""
    from app.extractor import (
        SubtitleExtractionError,
        detect_archive_kind,
        extract_subtitle_from_archive,
        transcode_to_utf8,
    )

    kind = detect_archive_kind(raw)
    if kind in ("rar", "7z", "tar", "gzip"):
        try:
            return extract_subtitle_from_archive(
                raw, target_filename=release_name, season=season, episode=episode
            )
        except SubtitleExtractionError as exc:
            logger.warning(
                "[reference] %s archive for %r unusable: %s", kind, release_name, exc
            )
            return None

    if zipfile.is_zipfile(io.BytesIO(raw)):
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            infos = [info for info in archive.infolist() if not info.is_dir()]
            if not infos:
                logger.warning("[reference] ZIP for %r is empty", release_name)
                return None
            members = [(info.filename, info.file_size) for info in infos]
            chosen = select_zip_member(members, season, episode, min_bytes)
            if chosen is None:
                logger.warning(
                    "[reference] ZIP for %r has no usable subtitle member: %s",
                    release_name,
                    [name for name, _ in members],
                )
                return None
            logger.info(
                "[reference] selected %r from %d ZIP member(s) (S%sE%s)",
                chosen,
                len(members),
                season,
                episode,
            )
            payload = archive.read(chosen)
        return transcode_to_utf8(payload)
    return transcode_to_utf8(raw)
