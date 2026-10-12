"""Separate eligibility policies over the exact same weighted result."""
from dataclasses import dataclass

from .model import MatchResult
from .profiles import language, stable_identity


def semantic_key(result: MatchResult) -> tuple[int, int, int, int]:
    """Video-source priority shared by display and reference execution.

    Verified MovieHash proof wins. With a known, unambiguous video source,
    exact subtype precedes compatible family, cross-source, then unresolved
    evidence. Missing video source imposes no invented source preference.
    Existing weighted points break ties within each evidence class.
    """
    return (0 if result.verified_hash else 1, result.source_priority,
            -result.ranking_score, -result.score_without_hash)


@dataclass(frozen=True)
class DisplayPolicy:
    languages: tuple[str, ...] = ("ara",)
    exclude_hi: bool = False
    prefer_hi: bool | None = None
    minimum_percent: int = 0

    def eligible(self, result: MatchResult) -> bool:
        return (result.discovery_eligible and result.candidate.language in {language(v) for v in self.languages}
                and not (self.exclude_hi and result.candidate.hearing_impaired is True)
                and result.score * 100 >= result.maximum * self.minimum_percent)

    def key(self, result: MatchResult) -> tuple:
        langs = tuple(language(v) for v in self.languages)
        hi = 0 if self.prefer_hi is None or result.candidate.hearing_impaired is self.prefer_hi else 1
        return (*semantic_key(result),
                langs.index(result.candidate.language) if result.candidate.language in langs else len(langs),
                hi, stable_identity(result.candidate))


@dataclass(frozen=True)
class ReferencePolicy:
    discovery: bool = False

    def eligible(self, result: MatchResult) -> bool:
        return (result.candidate.language in {"eng", "ara"}
                and (result.discovery_eligible if self.discovery else result.ready_reference_eligible))

    def key(self, result: MatchResult) -> tuple:
        return (*semantic_key(result),
                0 if result.candidate.language == "eng" else 1, stable_identity(result.candidate))


def rank(results: tuple[MatchResult, ...], policy: DisplayPolicy | ReferencePolicy) -> tuple[MatchResult, ...]:
    return tuple(sorted((r for r in results if policy.eligible(r)), key=policy.key))


def equivalent_peers(results: tuple[MatchResult, ...]) -> dict[str, tuple[str, ...]]:
    groups: dict[tuple[int, int, int, int], list[str]] = {}
    for result in results:
        if result.discovery_eligible:
            groups.setdefault(semantic_key(result), []).append(result.candidate.identity)
    return {identity: tuple(sorted(peer for peer in identities if peer != identity))
            for identities in groups.values() for identity in identities}
