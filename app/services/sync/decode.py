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
