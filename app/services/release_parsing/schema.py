"""Immutable release evidence; standard library only."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    raw: str
    origin: str
    precedence: int
    evidence_origin: str


@dataclass(frozen=True)
class CanonicalField:
    value: str | None
    observations: tuple[tuple[str, str, str, int, str], ...]
    conflict: bool = False


@dataclass(frozen=True)
class CanonicalRelease:
    fields: tuple[tuple[str, CanonicalField], ...]

    def field(self, name: str) -> CanonicalField:
        return dict(self.fields)[name]

    def value(self, name: str) -> str | None:
        return self.field(name).value


@dataclass(frozen=True)
class Candidate:
    id: str
    aliases: tuple[str, ...]
    records: tuple[tuple[str, str, str], ...]  # provider, provider ID, fixture ID
    canonical: CanonicalRelease
    record_duplicates: tuple[str, ...] = ()
    content_peers: tuple[str, ...] = ()
    content_sha256: str | None = None
    archive_members: tuple[str, ...] = ()
    release_names: tuple[str, ...] = ()
    unresolved_pack: bool = False
    independently_resolved_member: bool = False
    member_identities: tuple[tuple[str, str, str], ...] = ()
