"""Bounded contextual enrichment of the canonical release identity parser."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import replace
from functools import lru_cache
from importlib.metadata import version
from typing import Any
from urllib.parse import unquote

from app.services.sync.release_knowledge import (
    EXCLUSIVE_CUTS,
    KNOWLEDGE_FINGERPRINT,
    KNOWLEDGE_VERSION,
    RULES,
    Evidence,
    ReleaseMetadata,
)

SCHEMA_VERSION = "1"
PARSER_VERSION = "2"
GUESSIT_VERSION = version("guessit")
SEMANTIC_VERSION = f"release-{SCHEMA_VERSION}-{PARSER_VERSION}-{KNOWLEDGE_VERSION}-{KNOWLEDGE_FINGERPRINT}-{GUESSIT_VERSION}"
MAX_NAME_LENGTH = 2048
_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")
_EPISODE = re.compile(r"\bs(\d{1,2})(?:\s*e(\d{1,3}))?\b|\b(\d{1,2})x(\d{1,3})\b|\bseason (\d{1,2})\b", re.I)
_TECH = re.compile(r"\b(?:2160p|1080[pi]|720p|480p|4k|blu ?ray|bd ?rip|br ?rip|web ?dl|web ?rip|hdtv|dvd ?rip|remux)\b", re.I)
_RESOLUTION = re.compile(r"\b(2160p|1080[pi]|720p|480p|4k)\b", re.I)
_CODEC = re.compile(r"\b(x26[45]|h ?26[45]|hevc|avc|av1|xvid)\b", re.I)
_BITS = re.compile(r"\b(8|10|12) ?bit\b", re.I)
_GROUP = re.compile(r"-([a-z0-9][a-z0-9_-]{1,40})(?:\.[a-z]{2,3})?$", re.I)
_TECHNICAL_DASH_PAIRS = (("web", "dl"), ("bd", "rip"), ("br", "rip"),
                         ("dts", "hd"), ("h", "264"), ("h", "265"),
                         ("e", "ac"), ("e-ac", "3"), ("ac", "3"))
_EXT = re.compile(r"\.(?:srt|ass|ssa|sub|vtt|mkv|mp4|avi)$", re.I)
_DURATION = re.compile(r"\[(\d{1,2})h(\d{1,2})m(\d{1,2})[ms]\]", re.I)
_FPS = re.compile(r"\b(23\.976|24|25|29\.97|30|50|60)\s*fps\b", re.I)
_UHD = re.compile(r"\b(?:uhd|ultra hd)\b", re.I)
_REMUX = re.compile(r"\b(?:bd)?remux\b", re.I)
_PAREN = re.compile(r"[\[(]([^\[\]()]+)[\])]\s*$")
_NON_GROUP = re.compile(r"(?:\d+bit|edition|collection|cut|hdr\d*|dv|x26[45]|hevc|avc|eng|ara|dl|rip)", re.I)
_RESTORATION = re.compile(r"\b((?:19|20)\d{2}) (?:restoration|restored|remaster(?:ed)?)\b|\b(?:restoration|restored|remaster(?:ed)?) ((?:19|20)\d{2})\b")
_HDR = (("hdr", re.compile(r"\bhdr\b", re.I)),
        ("hdr10", re.compile(r"\bhdr10(?!\+)\b", re.I)),
        ("hdr10_plus", re.compile(r"\bhdr10\+", re.I)),
        ("dolby_vision", re.compile(r"\b(?:dolby vision|dv|dovi)\b", re.I)))


def normalize_name(name: str) -> str:
    return unicodedata.normalize("NFKC", unquote(name)).replace("’", "'").replace("‘", "'")


def words(name: str) -> str:
    return re.sub(r"(?:[^\w+]|_)+", " ", name.casefold()).strip()


def parse_identity(name: str | None, *, hints: Mapping[str, object] | None = None) -> ReleaseMetadata:
    if not isinstance(name, str) or len(name) > MAX_NAME_LENGTH:
        return ReleaseMetadata()
    # Only a bounded basename enters the cache; no URLs/configuration are retained.
    base = normalize_name(name).split("?", 1)[0].replace("\\", "/").rsplit("/", 1)[-1]
    if len(base) > MAX_NAME_LENGTH:
        return ReleaseMetadata()
    metadata = _parse_cached(base, SEMANTIC_VERSION)
    if not hints:
        return metadata
    # Explicit, narrow provider hints only. Never cache/store raw provider data.
    changes: dict[str, Any] = {}
    ambiguous = set(metadata.ambiguous_fields)
    evidence = list(metadata.evidence)
    for field in ("title", "year", "season", "episode", "source_family"):
        value = hints.get(field)
        if field in {"year", "season", "episode"}:
            if type(value) is not int or not 0 <= value <= 9999:
                continue
        elif not isinstance(value, str) or len(value) > 160:
            continue
        elif field == "source_family" and value not in {"bluray", "uhd_bluray", "webdl", "webrip", "hdtv", "dvd"}:
            continue
        elif field == "title":
            value = words(normalize_name(value))
        current = getattr(metadata, field)
        if current is not None and current != value:
            ambiguous.add(field)
        elif field not in ambiguous:
            changes[field] = value
        evidence.append(Evidence(field, str(value), "provider", f"provider.{field}", "medium"))
    return replace(metadata, **changes, ambiguous_fields=tuple(sorted(ambiguous)), evidence=tuple(evidence))


@lru_cache(maxsize=2048)
def _parse_cached(base: str, semantic_version: str) -> ReleaseMetadata:
    from app.services.sync.matching import guess_metadata

    stem = _EXT.sub("", base)
    group_match = _GROUP.search(stem)
    group = group_match.group(1) if group_match and _TECH.search(words(stem)) else None
    group_start = group_match.start() if group_match else len(stem)
    if group:
        # Keep genuine hyphenated groups intact, while keeping technical pairs
        # such as WEB-DL out of the group (WEB-DL-TESTGROUP -> TESTGROUP).
        for prefix, suffix in _TECHNICAL_DASH_PAIRS:
            if stem[:group_start].casefold().endswith(prefix) and group.casefold().startswith(suffix + "-"):
                group_start += len(suffix) + 1
                group = group[len(suffix) + 1:]
    if group and (_NON_GROUP.fullmatch(group) or not any(char.isalpha() for char in group)):
        group = None
    # Scene groups are their own field, never cut/collection vocabulary.
    text = words(stem[:group_start] if group_match and group else stem)
    # A leading year-shaped token can be a numeric title (1917, 1984, 2001).
    # It cannot delimit its own title; the following release year may do so.
    year_match = next((match for match in _YEAR.finditer(text) if match.start() > 0), None)
    episode_match, tech_match = _EPISODE.search(text), _TECH.search(text)
    anchors = [match for match in (year_match, episode_match) if match]
    if not anchors and tech_match and tech_match.start() > 0:
        anchors = [tech_match]
    first = min(anchors, key=lambda match: match.start()) if anchors else None
    title = text[:first.start()].strip() if first else None
    # Tags are allowed only AFTER an identity/technical delimiter, never inside
    # the title. An unstructured title alone is intentionally not a tag source.
    region = text[first.start() if first is tech_match else first.end():].strip() if first else ""
    evidence = []
    guess = None
    if first is tech_match and title:
        guess = guess_metadata(base)
        guessed_title = guess.get("title")
        # For yearless structured releases, a recognized edition after a
        # nonempty GuessIt title establishes the metadata delimiter. Whole
        # titles such as Final Cut remain titles; no free substring scan.
        if guess.get("edition") and isinstance(guessed_title, str):
            title_words = words(guessed_title)
            if title_words and title.startswith(title_words + " "):
                region = text[len(title_words):].strip()
                title = title_words
                evidence.append(Evidence("title", title[:160], "guessit", "guessit.edition_context", "medium"))
    region_tech = _TECH.search(region)
    if episode_match and region_tech and any(
        pattern.search(region[:region_tech.start()])
        for rule, pattern in RULES
        if rule["category"] in {"cut", "label", "presentation", "master", "packaging"}
    ):
        # Episode titles occur after SxxExx. Resolve only potentially confusing
        # vocabulary; ordinary technical episode names keep the cheap path.
        guess = guess_metadata(base)
        episode_title = guess.get("episode_title")
        if isinstance(episode_title, str):
            title_words = words(episode_title)
            if title_words and region.startswith(title_words + " "):
                region = region[len(title_words):].strip()
                evidence.append(Evidence("episode_title", title_words[:160], "guessit", "guessit.title_guard", "medium"))
    values: dict[str, set[str]] = {}
    for rule, pattern in RULES:
        if pattern.search(region):
            category, value = rule["category"], rule["canonical"]
            values.setdefault(category, set()).add(value)
            evidence.append(Evidence(category, value, "knowledge", rule["id"], rule["confidence"]))
    source_values = values.get("source", set())
    if "uhd_bluray" in source_values:
        source_values.discard("bluray")
    elif "bluray" in source_values and _UHD.search(region):
        source_values.discard("bluray")
        source_values.add("uhd_bluray")
    if _REMUX.search(region) and not source_values:
        source_values.add("bluray")
    season = episode = None
    if episode_match:
        a, b, c, d, e = episode_match.groups()
        season = int(a or c or e or 0)
        episode = int(b or d) if b or d else None
    year = int(year_match.group(1)) if year_match and (not episode_match or year_match.start() < episode_match.start()) else None
    # Measured fast path: conventional year/episode + technical filenames do
    # not need GuessIt. Ambiguous/incomplete names are locally enriched once.
    if not (title and (year or season is not None) and tech_match):
        guess = guess if guess is not None else guess_metadata(base)
        if (not title or (episode_match is None and year is None and isinstance(guess.get("episode"), int))) and isinstance(guess.get("title"), str):
            title = words(guess["title"])
        if (year is None and isinstance(guess.get("year"), int)
                and str(guess["year"]) not in (title or "").split()):
            year = guess["year"]
        if season is None and isinstance(guess.get("season"), int) and guess["season"] != year:
            season = guess["season"]
        if episode is None and isinstance(guess.get("episode"), int):
            episode = guess["episode"]
        for field in ("title", "year", "season", "episode"):
            if field in guess:
                evidence.append(Evidence(field, str(guess[field])[:160], "guessit", "guessit.enrichment", "medium"))
    resolution = _RESOLUTION.search(region)
    codec_match = _CODEC.search(region)
    codec = None
    if codec_match:
        token = codec_match.group(1).replace(" ", "").lower()
        codec = "h265" if token in {"x265", "h265", "hevc"} else "h264" if token in {"x264", "h264", "avc"} else token
    # Include the first technical delimiter for technical projection only.
    technical_match = _TECH.search(region)
    technical = region[technical_match.start():] if technical_match else region
    resolution = _RESOLUTION.search(technical)
    bits = _BITS.search(technical)
    if not group:
        parenthetical = _PAREN.search(stem)
        if parenthetical and _RESOLUTION.search(words(parenthetical.group(1))) and _CODEC.search(words(parenthetical.group(1))):
            group = parenthetical.group(1).split()[-1]
    if not group and first and tech_match:
        # Reuse cached enrichment for conventional whitespace/dot group suffixes.
        # Only an explicit GuessIt release_group at the filename suffix counts;
        # technical/edition tokens remain excluded by the existing guard.
        guess = guess if guess is not None else guess_metadata(base)
        guessed_group = guess.get("release_group")
        if (isinstance(guessed_group, str) and 1 < len(guessed_group) <= 40
                and words(stem).endswith(" " + words(guessed_group))
                and not _NON_GROUP.fullmatch(guessed_group)):
            group = guessed_group
            evidence.append(Evidence("release_group", group, "guessit", "guessit.group_suffix", "medium"))
    if group and _NON_GROUP.fullmatch(group):
        group = None
    def tags(category: str) -> tuple[str, ...]:
        return tuple(sorted(values.get(category, ())))

    duration, fps = _DURATION.search(base), _FPS.search(base)
    restoration = _RESTORATION.search(region)
    ambiguous = tuple(field for field, category in (("source_family", "source"), ("service", "service")) if len(values.get(category, ())) > 1)
    if len(values.get("cut", set()) & EXCLUSIVE_CUTS) > 1:
        ambiguous += ("cut_tags",)
    for field, value in (("title", title), ("year", year), ("season", season),
                         ("episode", episode), ("release_group", group),
                         ("resolution", resolution.group(1) if resolution else None), ("codec", codec)):
        if value is not None:
            evidence.append(Evidence(field, str(value)[:160], "filename", f"parser.{field}.v1"))
    return ReleaseMetadata(
        title=title, year=year, season=season, episode=episode,
        source_family=next(iter(source_values)) if len(source_values) == 1 else None,
        resolution=resolution.group(1).lower().replace("4k", "2160p") if resolution else None,
        codec=codec, bit_depth=int(bits.group(1)) if bits else None,
        remux=bool(_REMUX.search(technical)),
        hdr_flags=tuple(tag for tag, pattern in _HDR if pattern.search(technical)),
        release_group=group, cut_tags=tags("cut"), label_tags=tags("label"),
        presentation_tags=tags("presentation"), master_tags=tags("master"), packaging_tags=tags("packaging"),
        edition_tags=tuple(sorted(set().union(*(values.get(category, set()) for category in ("label", "presentation", "master", "packaging"))))),
        service=next(iter(values["service"])) if len(values.get("service", ())) == 1 else None,
        audio_tags=tags("audio"), fps=float(fps.group(1)) if fps else None,
        duration_hint_seconds=(int(duration.group(1))*3600 + int(duration.group(2))*60 + int(duration.group(3))) if duration else None,
        restoration_year=int(restoration.group(1) or restoration.group(2)) if restoration else None,
        evidence=tuple(evidence), ambiguous_fields=ambiguous,
    )
