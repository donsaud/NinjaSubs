"""Canonicalize every observation without discarding contradictory evidence."""

import hashlib
import json
import re
import unicodedata

from .parser import parse
from .rules import (
    AUDIO_ALIASES,
    CUT_ALIASES,
    FIELDS,
    SERVICE_FAMILIES,
    SOURCE_ALIASES,
    VERSION_LABELS,
    VIDEO_ALIASES,
)
from .schema import Candidate, CanonicalField, CanonicalRelease, Observation


def compact(value: str) -> str:
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFC", value).casefold())


def normalize_value(field: str, value: str) -> str | None:
    value = unicodedata.normalize("NFC", value).strip().casefold()
    if value in {"", "unknown", "n/a", "none"}:
        return None
    aliases = {"source": SOURCE_ALIASES, "video_codec": VIDEO_ALIASES,
               "audio_codec": AUDIO_ALIASES, "service_disc_family": SERVICE_FAMILIES}
    if field in aliases:
        return {compact(k): v for k, v in aliases[field].items()}.get(compact(value))
    if field in {"title", "release_group"}:
        return compact(value) or None
    if field == "cut":
        return CUT_ALIASES.get(compact(value))
    if field == "version":
        labels = {token for token in re.findall(r"[a-z]+", value) if token in VERSION_LABELS}
        return "+".join(sorted(labels)) or None
    if field == "variant_labels":
        return "+".join(sorted({compact(label) for label in re.split(r"[+,;|]+", value) if compact(label)})) or None
    if field in {"season", "episode", "video_size"}:
        return str(int(value)) if value.isdecimal() else None
    if field in {"hash_verified", "imdb_verified"}:
        return "true" if value == "true" else "false" if value == "false" else None
    if field == "moviehash":
        return value if re.fullmatch(r"[0-9a-f]{16}", value) else None
    if field == "resolution":
        return value if re.fullmatch(r"(?:2160|1080|720|576|480)[pi]", value) else None
    if field == "imdb_id":
        return value if re.fullmatch(r"tt\d+", value) else None
    if field == "year":
        return value if re.fullmatch(r"\d{4}", value) and 1700 <= int(value) <= 2199 else None
    if field == "scope":
        return value if value in {"episode", "season_pack", "movie"} else None
    if field == "media_type":
        return value if value in {"movie", "series"} else None
    raise ValueError(f"unknown canonical field: {field}")


def normalize(observations: dict[str, tuple[Observation, ...]]) -> CanonicalRelease:
    fields = []
    for name in FIELDS:
        evidence = tuple((o.raw, normalize_value(name, o.raw) or "unknown", o.origin, o.precedence, o.evidence_origin)
                         for o in observations[name])
        known = [(value, precedence) for _, value, _, precedence, _ in evidence if value != "unknown"]
        chosen = min(known, key=lambda pair: (pair[1], pair[0]))[0] if known else None
        distinct = {value for value, _ in known}
        # A member can supply episode scope for a pack; the two scopes coexist.
        conflict = len(distinct) > 1 and not (name == "scope" and distinct == {"episode", "season_pack"})
        if name in {"variant_labels", "version"}:
            chosen = "+".join(sorted({label for value in distinct for label in value.split("+")})) or None
            conflict = False  # non-exclusive labels retain all observations
        fields.append((name, CanonicalField(chosen, evidence, conflict)))
    return CanonicalRelease(tuple(fields))


def canonical_record(record: dict) -> CanonicalRelease:
    return normalize(parse(record))


def candidates(records: list[dict]) -> tuple[Candidate, ...]:
    """Merge duplicate provider records, not different releases sharing bytes.

    Record identity is (provider, provider_id). Every observation and fixture
    alias is retained. Content equality is separately reported through peers;
    differing release claims never become interchangeable by content hash alone.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        groups.setdefault((record["provider"].casefold(), record["provider_id"]), []).append(record)
    out = []
    for key, group in sorted(groups.items()):
        merged: dict[str, list[Observation]] = {field: [] for field in FIELDS}
        for record in group:
            for field, evidence in parse(record).items():
                # Attach provider/record identity so merged provenance is explicit.
                merged[field].extend(Observation(o.raw, f"{record['id']}:{o.origin}", o.precedence, o.evidence_origin)
                                     for o in evidence)
        aliases = tuple(sorted(record["id"] for record in group))
        hashes = {record["content_sha256"].lower() for record in group if record.get("content_sha256")}
        if any(not re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes):
            raise ValueError("content_sha256 must be a 64-character SHA-256")
        if len(hashes) > 1:
            raise ValueError("duplicate provider record has conflicting content hashes")
        observations = {name: tuple(sorted(set(evidence), key=lambda o: (o.precedence, o.origin, o.raw, o.evidence_origin)))
                        for name, evidence in merged.items()}
        pack = any(o.raw == "season_pack" for o in observations["scope"])
        members = {r["archive_member"] for r in group if r.get("archive_member")}
        verified_members = {r["archive_member"] for r in group if r.get("archive_member_verified") and r.get("archive_member")}
        out.append(Candidate(aliases[0], aliases, tuple((key[0], key[1], alias) for alias in aliases),
                             normalize(observations), aliases[1:], (), next(iter(hashes), None),
                             tuple(sorted({r["archive_member"] for r in group if r.get("archive_member")})),
                             tuple(sorted({name for r in group for name in r.get("release_names", [])} |
                                          {r["filename"] for r in group if r.get("filename")})),
                             pack and not (members and members <= verified_members),
                             bool(members and members <= verified_members),
                             tuple(sorted({(r["archive_member"], str(int(r["archive_member_identity"]["season"])),
                                            str(int(r["archive_member_identity"]["episode"])))
                                           for r in group if r.get("archive_member_verified")
                                           and r.get("archive_member_identity")}))))
    peers: dict[str, list[str]] = {}
    for candidate in out:
        if candidate.content_sha256:
            peers.setdefault(candidate.content_sha256, []).append(candidate.id)
    return tuple(Candidate(c.id, c.aliases, c.records, c.canonical, c.record_duplicates,
                           tuple(sorted(x for x in peers.get(c.content_sha256 or "", []) if x != c.id)),
                           c.content_sha256, c.archive_members, c.release_names,
                           c.unresolved_pack, c.independently_resolved_member, c.member_identities) for c in out)


def stable_identity(candidate: Candidate) -> tuple[str, str, str]:
    values = {name: field.value for name, field in candidate.canonical.fields}
    digest = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
    provider, provider_id, _ = candidate.records[0]
    return provider, provider_id, digest
