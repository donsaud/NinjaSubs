"""Local, independently authored release vocabulary and immutable evidence model."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class MetadataState(str, Enum):
    KNOWN = "known"
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"


class Comparison(str, Enum):
    MATCH = "match"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class Evidence:
    category: str
    value: str
    source: str
    rule_id: str
    confidence: str = "high"


@dataclass(frozen=True)
class ReleaseMetadata:
    title: str | None = None
    year: int | None = None
    season: int | None = None
    episode: int | None = None
    source_family: str | None = None
    resolution: str | None = None
    codec: str | None = None
    bit_depth: int | None = None
    remux: bool = False
    hdr_flags: tuple[str, ...] = ()
    release_group: str | None = None
    edition_tags: tuple[str, ...] = ()
    cut_tags: tuple[str, ...] = ()
    label_tags: tuple[str, ...] = ()
    presentation_tags: tuple[str, ...] = ()
    master_tags: tuple[str, ...] = ()
    packaging_tags: tuple[str, ...] = ()
    service: str | None = None
    audio_tags: tuple[str, ...] = ()
    fps: float | None = None
    duration_hint_seconds: int | None = None
    restoration_year: int | None = None
    evidence: tuple[Evidence, ...] = ()
    ambiguous_fields: tuple[str, ...] = ()

    def state(self, field: str) -> MetadataState:
        if field in self.ambiguous_fields:
            return MetadataState.AMBIGUOUS
        value = getattr(self, field, None)
        missing = value is None or value is False or value == "" or value == ()
        return MetadataState.UNKNOWN if missing else MetadataState.KNOWN


_CONTENT = Path(__file__).with_name("release_knowledge.json").read_bytes()
_DATA = json.loads(_CONTENT)
KNOWLEDGE_VERSION: str = _DATA["version"]
KNOWLEDGE_FINGERPRINT = hashlib.sha256(_CONTENT).hexdigest()[:16]
RULES = tuple((rule, re.compile(r"(?<!\w)(?:" + rule["pattern"] + r")(?!\w)", re.I))
              for rule in _DATA["rules"])
EXCLUSIVE_CUTS = frozenset(rule["canonical"] for rule, _ in RULES
                           if rule["category"] == "cut" and rule["conflict_policy"] == "exclusive")


def compare_field(a: ReleaseMetadata, b: ReleaseMetadata, field: str) -> Comparison:
    """Unknown/ambiguous evidence cannot establish equality or contradiction."""
    if a.state(field) != MetadataState.KNOWN or b.state(field) != MetadataState.KNOWN:
        return Comparison.UNKNOWN
    return Comparison.MATCH if getattr(a, field) == getattr(b, field) else Comparison.CONFLICT


def compatible_source(a: str | None, b: str | None) -> bool:
    disc = {"bluray", "uhd_bluray"}
    return bool(a and b and (a == b or a in disc and b in disc))
