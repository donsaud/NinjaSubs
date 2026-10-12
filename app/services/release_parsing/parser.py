"""Extract observations only. No canonical arbitration or production imports."""

import re

from .rules import CUT_ALIASES, FIELDS, GROUP_SUFFIX_PATTERN, VERSION_LABELS
from .schema import Observation


def parse_name(name: str, origin: str, precedence: int,
               evidence_origin: str = "input_record") -> dict[str, list[Observation]]:
    out: dict[str, list[Observation]] = {field: [] for field in FIELDS}

    def add(field: str, raw: str) -> None:
        out[field].append(Observation(raw, origin, precedence, evidence_origin))

    name = re.sub(r"\.(mkv|mp4|avi|srt|ass|ssa|zip)$", "", name, flags=re.I)
    episode = (re.search(r"\bS(\d{1,2})(?:[ ._-]*E(\d{1,3}))?", name, re.I)
               or re.search(r"\b(\d{1,2})x(\d{1,3})\b", name, re.I))
    if episode:
        add("season", episode[1])
        add("media_type", "series")
        if episode[2]:
            add("episode", episode[2])
            add("scope", "episode")
            compound = re.match(r"(?:[ ._-]*(?:E|-)(\d{1,3}))+", name[episode.end():], re.I)
            if compound:
                for extra in re.findall(r"\d{1,3}", compound[0]):
                    add("episode", extra)
        else:
            add("scope", "season_pack")
    absolute = None
    span = None
    if not episode:
        # An explicit span/collection ("1 ~ 64", "1-28", "01-28") declares a batch,
        # never a single episode. Reading its first number as episode 1 would
        # hard-reject a pack that does contain the requested episode.
        span = re.search(r"(?i)(?<![\d])0*\d{1,3}\s*(?:~|--?|–|to)\s*0*\d{1,4}(?![\d])", name)
        if span:
            add("scope", "season_pack")
            add("media_type", "series")
        # Established anime absolute forms; years/resolutions/codecs are excluded.
        # The number must end at a delimiter or end-of-name. Without this an
        # opaque hash filename ("e4Wc...P880Xs...") reads its leading "e" as an
        # episode marker and its digits as an episode, which would hard-reject
        # every candidate for a hashed target.
        for found in ([] if span else re.finditer(
                r"(?i)(?:^|[ ._\-#\[])(?:E|EP|episode)?0*(\d{1,4})(?![A-Za-z0-9])", name)):
            number = int(found[1])
            if re.search(r"\b(?:19|20)\d{2}\b", name) and not re.search(r"(?i)[E#\-]", found[0]):
                continue
            if number not in {2160, 1080, 720, 576, 480, 264, 265} and not 1900 <= number <= 2099:
                absolute = found
                if re.match(r"[ ]*(?:-|~)[ ]*\d+", name[found.end():]):
                    add("scope", "season_pack")
                else:
                    add("episode", str(number))
                    add("scope", "episode")
                add("media_type", "series")
                break
    # A batch still names its show: the span bounds the title just as an
    # episode marker does.
    title_boundary = (episode.start() if episode else
                      absolute.start() if absolute else
                      span.start() if span else None)
    year = re.search(r"\b(?:19|20)\d{2}\b", name)
    if year:
        add("year", year[0])
        title_boundary = min(title_boundary, year.start()) if title_boundary is not None else year.start()
        if not episode and not absolute:
            add("media_type", "movie")
    if title_boundary is not None and name[:title_boundary].strip(" ._-"):
        title = re.sub(r"^\[[^\]]+\]\s*", "", name[:title_boundary]).strip(" ._-")
        if title.casefold() not in {"episode", "ep", "e"}:
            add("title", title)
    release_region = name[episode.end():] if episode else name[year.end():] if year else name
    sources = list(re.finditer(
        r"(?<![a-z0-9])(Blu[ ._-]?Ray|BDRip|BD[ ._-]?Rip|REMUX|WEB[ ._-]?DL|"
        r"WEB[ ._-]?Rip|HD[ ._-]?TV|SD[ ._-]?TV|DVD[ ._-]?Rip|HD[ ._-]?CAM|HDTS|"
        r"Telecine|Telesync|DVDScr|Screener|CAM|TS|TC|DSR|Sat[ ._-]?Rip|SCR)(?![a-z0-9])",
        release_region, re.I))
    # BluRay REMUX is one disc subtype declaration, not contradictory evidence.
    remux = [m for m in sources if m[0].lower() == "remux"]
    for source in sources:
        if remux and re.fullmatch(r"blu[ ._-]?ray", source[0], re.I):
            continue
        add("source", source[0])
    group = re.search(GROUP_SUFFIX_PATTERN, name, re.I) or re.search(r"-([a-z0-9]+)$", name, re.I)
    if group:
        # End-of-name group suffix: avoid mistaking WEB-DL alone for a group.
        if not re.fullmatch(r"(?:DL|Rip)(?:[ ._-].*)?", group[1], re.I):
            add("release_group", group[1])
    resolution = re.search(r"\b(2160|1080|720|576|480)[pi]\b", release_region, re.I)
    if resolution:
        add("resolution", resolution[0])
    patterns = {
        "video_codec": r"(?<![a-z0-9])(h[.]?264|x264|avc|h[.]?265|x265|hevc|xvid|divx|av1)(?![a-z0-9])",
        "audio_codec": r"(?<![a-z0-9])(truehd|dts[ ._-]?hd[ ._-]?ma|dts|ddp|eac3|ac3|aac|flac)(?![a-z])",
        "variant_labels": r"\b(Criterion|IMAX|Remastered|Collection)\b",
        "version": r"\b(Extended|Unrated)\b",
        "cut": r"\b(Director[’']?s[ ._-]*Cut|Theatrical[ ._-]*Cut|Broadcast[ ._-]*Cut|Final[ ._-]*Cut)\b",
        "service_disc_family": r"\b(NF|Netflix|AMZN|Amazon|HMAX|DSNP|ATVP|OSN|UHD)\b",
    }
    for field, pattern in patterns.items():
        for match in re.finditer(pattern, release_region, re.I):
            add(field, match[0])
    return out


def parse(record: dict) -> dict[str, tuple[Observation, ...]]:
    out: dict[str, list[Observation]] = {field: [] for field in FIELDS}
    record_origin = record.get("evidence_origin") or (
        f"provider:{record['provider'].casefold()}:{record['provider_id']}"
        if "provider" in record else "target_input")
    for field, value in record.get("metadata", {}).items():
        if value is not None and str(value).strip():
            normalized = re.sub(r"[\W_]+", "", str(value).casefold())
            projected_field = field
            if field == "edition":
                projected_field = ("cut" if normalized in CUT_ALIASES else
                                   "version" if normalized in VERSION_LABELS else "variant_labels")
            elif field in {"cut", "version"}:
                if normalized in CUT_ALIASES:
                    projected_field = "cut"
                elif normalized in VERSION_LABELS:
                    projected_field = "version"
            evidence_origin = record.get("metadata_origins", {}).get(field, record_origin)
            out[projected_field].append(Observation(str(value), f"metadata.{field}", 0, evidence_origin))
    names = [(name, f"release_name:{name}", 2) for name in record.get("release_names", [])]
    if record.get("filename"):
        names.append((record["filename"], "filename", 2))
    if record.get("archive_member"):
        names.append((record["archive_member"], "archive_member", 1))
        if record.get("archive_member_verified") and record.get("archive_member_identity"):
            member_origin = f"validated_member:{record_origin}:{record['archive_member']}"
            for field, value in record["archive_member_identity"].items():
                out[field].append(Observation(str(value), f"archive_member_identity.{field}", 1, member_origin))
            out["scope"].append(Observation("episode", "archive_member_identity.scope", 1, member_origin))
    for name, origin, precedence in names:
        evidence_origin = record_origin
        observations = parse_name(name, origin, precedence, evidence_origin)
        for field, evidence in observations.items():
            if origin == "archive_member" and record.get("archive_member_verified") and field in {"season", "episode"}:
                evidence = [Observation(o.raw, o.origin, o.precedence, f"validated_member:{record_origin}:{name}")
                            for o in evidence]
            out[field].extend(evidence)
    return {field: tuple(sorted(set(evidence), key=lambda o: (o.precedence, o.origin, o.raw, o.evidence_origin)))
            for field, evidence in out.items()}
