"""Immutable profiles, evidence, and independently inspectable score results."""
from dataclasses import dataclass
from enum import StrEnum


class MatchState(StrEnum):
    MATCH = "match"
    MISMATCH = "mismatch"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class Fact:
    value: str | None
    provenance: tuple[tuple[str, str, str, int, str], ...] = ()
    conflict: bool = False


@dataclass(frozen=True)
class TargetVideoProfile:
    facts: tuple[tuple[str, Fact], ...]
    imdb_authoritative: bool = False
    hash_authoritative: bool = False

    def fact(self, name: str) -> Fact:
        return dict(self.facts).get(name, Fact(None))


@dataclass(frozen=True)
class SubtitleCandidateProfile(TargetVideoProfile):
    identity: str = ""
    provider: str = "unknown"
    provider_id: str = "<missing-provider-id>"
    aliases: tuple[str, ...] = ()
    language: str = "unknown"
    hearing_impaired: bool | None = None
    unresolved_pack: bool = False
    archive_members: tuple[str, ...] = ()
    independently_resolved_member: bool = False
    content_peers: tuple[str, ...] = ()
    release_claims: tuple[str, ...] = ()


@dataclass(frozen=True)
class MatchEvidence:
    field: str
    state: MatchState
    target: Fact
    candidate: Fact
    relationship: str = "direct"


@dataclass(frozen=True)
class Contribution:
    criterion: str
    weight: int
    points: int
    basis: str


@dataclass(frozen=True)
class MatchResult:
    candidate: SubtitleCandidateProfile
    evidence: tuple[MatchEvidence, ...]
    contributions: tuple[Contribution, ...]
    score: int
    score_without_hash: int
    maximum: int
    rejection_reasons: tuple[str, ...]
    readiness_reasons: tuple[str, ...]
    verified_hash: bool = False

    @property
    def source_priority(self) -> int:
        """Exact subtype, compatible family, cross-source, then unresolved."""
        source = next((e for e in self.evidence if e.field == "source"), None)
        if source is None or source.target.value is None or source.target.conflict:
            return 0
        if source.state is MatchState.MATCH:
            return 0 if source.relationship == "direct" else 1
        return 2 if source.state is MatchState.MISMATCH else 3

    @property
    def source_priority_points(self) -> int:
        """Actual source evidence earns an ordering bonus; no identity is inferred.

        One band's width exceeds every comparable metadata point combined.
        Consequently metadata completeness cannot reverse the source policy.
        An unknown/conflicting video source earns no invented preference.
        """
        source = next((e for e in self.evidence if e.field == "source"), None)
        if source is None or source.target.value is None or source.target.conflict:
            return 0
        return (3 - self.source_priority) * (self.maximum + 1)

    @property
    def ranking_score(self) -> int:
        """Shared display/reference points, separate from admission evidence."""
        return (self.score_without_hash + self.source_priority_points
                + (4 * (self.maximum + 1) if self.verified_hash else 0))

    @property
    def ranking_maximum(self) -> int:
        source = next((e for e in self.evidence if e.field == "source"), None)
        known = source is not None and source.target.value is not None and not source.target.conflict
        return 4 * (self.maximum + 1) - 1 if known else self.maximum

    @property
    def ranking_percentage(self) -> float:
        """Relative release score, never a probability or timing verdict."""
        return round(100 * min(self.ranking_score, self.ranking_maximum) / self.ranking_maximum, 2)

    @property
    def ranking_contributions(self) -> tuple[Contribution, ...]:
        return (tuple(c for c in self.contributions if c.criterion != "moviehash")
                + (Contribution("source_priority", 3 * (self.maximum + 1),
                                self.source_priority_points, "video_source_evidence"),
                   Contribution("moviehash", 4 * (self.maximum + 1),
                                4 * (self.maximum + 1) if self.verified_hash else 0,
                                "verified_query" if self.verified_hash else "no_hash_proof")))

    @property
    def percentage(self) -> float:
        return round(100 * min(self.score, self.maximum) / self.maximum, 2)

    @property
    def percentage_without_hash(self) -> float:
        return round(100 * self.score_without_hash / self.maximum, 2)

    def state(self, field: str) -> MatchState:
        return next(e.state for e in self.evidence if e.field == field)

    @property
    def discovery_eligible(self) -> bool:
        return not self.rejection_reasons

    @property
    def ready_reference_eligible(self) -> bool:
        return self.discovery_eligible and not self.readiness_reasons
