"""Replaceable parser adapter reusing only NinjaSubs' pure evidence primitives."""
import hashlib
import json
import re
import unicodedata
from dataclasses import replace

from app.services.release_parsing.normalize import candidates, canonical_record

from .model import Fact, SubtitleCandidateProfile, TargetVideoProfile

LANGUAGES = {"en": "eng", "eng": "eng", "ar": "ara", "ara": "ara"}


def language(value: str) -> str:
    return LANGUAGES.get(value.casefold(), value.casefold() or "unknown")


def _parser_record(record: dict) -> dict:
    # Generic descriptive suffixes are not scene groups. Retain the original
    # unmodified input in the caller; this parsing view only removes that suffix.
    result = {**record}
    for key in ("filename", "archive_member"):
        if record.get(key):
            result[key] = _release_name(record[key])
    result["release_names"] = [_release_name(n) for n in record.get("release_names", [])]
    return result


def _release_name(name: str) -> str:
    return re.sub(r"-([A-Za-z0-9]+)-(?:AsRequested|Internal)(?=\.(?:mkv|srt|ass|ssa|zip)$|$)",
                  r"-\1", name, flags=re.I)


def _technical_value(field: str, value: str) -> str:
    compact = re.sub(r"[^a-z0-9+]", "", value.casefold())
    if field == "dolby_vision":
        return "dv" if compact in {"dv", "dovi", "dolbyvision"} else "unknown"
    if field == "hdr":
        return {"hdr": "hdr", "hdr10": "hdr10", "hdr10+": "hdr10plus", "hdr10plus": "hdr10plus", "hlg": "hlg"}.get(compact, "unknown")
    compact = compact.removesuffix("bit")
    return compact if compact in {"8", "10", "12"} else "unknown"


def _facts(canonical, records: list[dict]) -> tuple[tuple[str, Fact], ...]:
    facts = {name: Fact(field.value, field.observations, field.conflict)
             for name, field in canonical.fields}
    # Technical observations have zero scoring weight. Keep all contradictory
    # evidence rather than rewarding mere metadata quantity.
    for field, pattern in (("hdr", r"\b(HDR10\+?|HDR|HLG)(?![A-Za-z0-9])"),
                           ("dolby_vision", r"\b(DV|DoVi|Dolby[ ._-]?Vision)\b"),
                           ("bit_depth", r"\b(8|10|12)[ ._-]?bit\b")):
        observations = []
        for record in records:
            origin = str(record.get("evidence_origin") or "provider_record")
            supplied = record.get("metadata", {}).get(field)
            if supplied is not None:
                value = _technical_value(field, str(supplied))
                observations.append((str(supplied), value, f"metadata.{field}", 0, origin))
            for name in [record.get("filename", ""), *record.get("release_names", [])]:
                for found in re.finditer(pattern, name, re.I):
                    value = _technical_value(field, found[1])
                    observations.append((found[0], value, "release_name", 2, origin))
        evidence = tuple(sorted(set(observations)))
        values = {o[1] for o in evidence if o[1] != "unknown"}
        facts[field] = Fact(min(values) if values else None, evidence, len(values) > 1)
    source = facts["source"]
    facts["remux"] = Fact("true" if source.value == "remux" else
                          "false" if source.value else None, source.provenance, source.conflict)
    return tuple(sorted(facts.items()))


def _core_record(record: dict) -> dict:
    result = _parser_record(record)
    # Extension fields are parsed independently, not passed into the old parser.
    result["metadata"] = {k: v for k, v in record.get("metadata", {}).items()
                          if k not in {"hdr", "dolby_vision", "bit_depth"}}
    if str(result.get("provider_id", "")).startswith("<missing-provider-id>"):
        canonical = canonical_record(result)
        values: dict[str, object] = {name: {"value": field.value, "conflict": field.conflict} for name, field in canonical.fields}
        values.update(language=language(str(record.get("language", ""))), hearing_impaired=record.get("hearing_impaired"),
                      release_claims=sorted(unicodedata.normalize("NFC", n).casefold().strip()
                                            for n in result.get("release_names", [])))
        result["provider_id"] = "<missing-provider-id>" + hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
    return result


def _valid_hash(record: dict) -> bool:
    metadata = record.get("metadata", {})
    return (metadata.get("hash_verified") is True
            and bool(re.fullmatch(r"[0-9a-fA-F]{16}", str(metadata.get("moviehash", ""))))
            and str(metadata.get("video_size", "")).isdecimal()
            and int(metadata["video_size"]) > 0)


def target_profile(record: dict) -> TargetVideoProfile:
    return TargetVideoProfile(_facts(canonical_record(_core_record(record)), [record]),
                              record.get("metadata", {}).get("imdb_verified") is True,
                              _valid_hash(record))


def candidate_profiles(records: list[dict]) -> tuple[SubtitleCandidateProfile, ...]:
    # Existing parser-only dedup retains contradictions and separates equal
    # content from duplicate provider records. No old matching/policy is used.
    parsed = candidates([_core_record(r) for r in records])
    result = []
    by_alias: dict[str, list[dict]] = {}
    for record in records:
        by_alias.setdefault(record["id"], []).append(record)
    for candidate in parsed:
        group = [r for a in sorted(set(candidate.aliases)) for r in by_alias[a]]
        languages = {language(str(r.get("language", ""))) for r in group}
        hi = {r.get("hearing_impaired") for r in group if isinstance(r.get("hearing_impaired"), bool)}
        authoritative_hash = all(_valid_hash(r) and r.get("hash_provenance") ==
                                 "opensubtitles_hash_query_literal_true" and r["provider"].casefold() ==
                                 "opensubtitles" for r in group)
        # Provider identity comes from its own payload, never search arguments.
        imdb_authoritative = all(r.get("imdb_authoritative") is True or
                                 r.get("metadata", {}).get("imdb_verified") is True for r in group)
        result.append(SubtitleCandidateProfile(
            _facts(candidate.canonical, group), imdb_authoritative, authoritative_hash,
            candidate.id, candidate.records[0][0], candidate.records[0][1], candidate.aliases,
            next(iter(languages)) if len(languages) == 1 else "unknown",
            next(iter(hi)) if len(hi) == 1 else None, candidate.unresolved_pack,
            candidate.archive_members, candidate.independently_resolved_member, candidate.content_peers,
            tuple(sorted(unicodedata.normalize("NFC", n).casefold().strip() for n in candidate.release_names))))
    return tuple(result)


def stable_identity(candidate: SubtitleCandidateProfile) -> tuple[str, str, str]:
    data: dict[str, object] = {name: {"value": fact.value, "conflict": fact.conflict} for name, fact in candidate.facts}
    data.update(language=candidate.language, hearing_impaired=candidate.hearing_impaired,
                release_claims=candidate.release_claims)
    digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    pid = candidate.provider_id
    if pid.startswith("<missing-provider-id>"):
        pid = "<missing-provider-id>"
    return candidate.provider, pid, digest


def with_language(candidate: SubtitleCandidateProfile, value: str) -> SubtitleCandidateProfile:
    return replace(candidate, language=language(value))
