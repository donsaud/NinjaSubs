"""Reference Selection v2 - SHADOW ONLY. Not used for production references.

The current selector commits to the first candidate that passes cue sanity, in
its own ``reference_tier`` ladder. That ladder is a *second* ranking system
alongside ``MatchTier``, and the two can disagree. This module builds the
alternative policy, runs it beside the existing one, and records where they
diverge - without changing which reference the synchronizer actually uses.

The objective is deliberately not "the subtitle the user should watch". It is
"the subtitle that is safest to use as a timing anchor". A provider with
excellent translation may be a poor reference; a mediocre user-facing subtitle
may be an excellent reference.

Design constraints honoured here:

* **No new content matcher.** Content identity comes from the existing
  ``hard_compatibility_filter`` and ``calculate_compatibility``, so v2 consumes
  ``MatchTier`` rather than inventing a third ladder.
* **Hard rejections are absolute.** A wrong episode cannot be rescued by health,
  consensus, independence, or provider reputation.
* **Lexicographic, not a mega-score.** Content identity dominates; health only
  breaks ties between candidates that are already comparable on content.
* **Reference-specific quality gates.** A subtitle acceptable to show a user may
  still be unsafe as a timing anchor (credits-only, sparse dialogue).
* **Duplicate timing grids are recorded, not rejected.** Three providers
  serving one file is one timing model, but it is still a usable reference.
* **Nothing here can create verification.** Reference trust is an input to
  selection, never proof on its own.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from app.models import MatchTier
from app.services.subtitle_matcher import (
    calculate_compatibility,
    extract_metadata,
    hard_compatibility_filter,
)
from app.services.sync.reference import (
    ReferenceHealth,
    ReferenceTrust,
    analyze_reference_health,
    reference_fingerprint,
)
from app.services.sync.structural import StructuralProfile

logger = logging.getLogger(__name__)

# --- Reference-specific quality gates --------------------------------------- #
# Deliberately separate from user-facing acceptance: a subtitle with little
# dialogue is still worth showing, but it is unsafe to align against.
GATE_MIN_DIALOGUE_CUES = 12
GATE_MIN_TOTAL_CUES = 12
GATE_MAX_INVALID_RATE = 0.0
GATE_MAX_BACKWARD_FRACTION = 0.02
GATE_MIN_DIALOGUE_COVERAGE = 0.55

# Health ranking, best first. Health never outranks content identity; it only
# separates candidates that already tie on content.
_HEALTH_ORDER = {"healthy": 0, "degraded": 1, "unusable": 2}
_TRUST_ORDER = {
    ReferenceTrust.VERIFIED: 0,
    ReferenceTrust.STRONG: 1,
    ReferenceTrust.ACCEPTABLE: 2,
    ReferenceTrust.UNKNOWN: 3,
    ReferenceTrust.REJECTED: 4,
}


def _health_class(health: ReferenceHealth) -> str:
    if health.dialogue_cues >= GATE_MIN_DIALOGUE_CUES and health.cue_count >= GATE_MIN_TOTAL_CUES:
        if (
            health.invalid_timing_rate <= GATE_MAX_INVALID_RATE
            and health.backward_fraction <= GATE_MAX_BACKWARD_FRACTION
            and (
                health.dialogue_coverage is None
                or health.dialogue_coverage >= GATE_MIN_DIALOGUE_COVERAGE
            )
        ):
            return "healthy"
        return "degraded"
    return "unusable"


class ReferenceCandidate(BaseModel):
    """One downloaded candidate, judged for use as a timing anchor."""

    subtitle_id: str
    release_name: str
    provider: str | None = None
    is_hash_match: bool = False
    match_tier: MatchTier = MatchTier.FALLBACK
    hard_accepted: bool = True
    hard_reject_reason: str | None = None
    health: ReferenceHealth = Field(default_factory=ReferenceHealth)
    health_class: str = "unusable"
    trust: ReferenceTrust = ReferenceTrust.UNKNOWN
    cache_verified: bool = False
    independent_group: str | None = None
    group_size: int = 1
    structural_profile: StructuralProfile | None = None
    reasons: list[str] = Field(default_factory=list)

    @property
    def eligible(self) -> bool:
        """Only hard-accepted candidates with usable dialogue can be selected."""
        return self.hard_accepted and self.health_class != "unusable"

    def sort_key(self) -> tuple:
        """Lexicographic ordering. Content identity dominates everything else."""
        return (
            0 if self.eligible else 1,
            int(self.match_tier),
            0 if self.cache_verified else 1,
            _TRUST_ORDER.get(self.trust, 3),
            _HEALTH_ORDER.get(self.health_class, 2),
            -(self.health.dialogue_cues or 0),
            str(self.provider or ""),
            self.release_name,
        )

    def explain(self) -> str:
        return (
            f"{self.release_name[:48]} provider={self.provider} "
            f"tier={self.match_tier.name} health={self.health_class} "
            f"trust={self.trust.value} eligible={self.eligible}"
        )


class ReferenceSelection(BaseModel):
    """Outcome of the shadow selector, including why it beat the alternatives."""

    chosen: ReferenceCandidate | None = None
    ranked: list[ReferenceCandidate] = Field(default_factory=list)
    considered: int = 0
    hard_rejected: int = 0
    reasons: list[str] = Field(default_factory=list)


class SelectionComparison(BaseModel):
    """Legacy vs shadow. Recorded for measurement only."""

    legacy_id: str | None = None
    shadow_id: str | None = None
    changed: bool = False
    agree: bool = True
    reasons: list[str] = Field(default_factory=list)
    legacy_trust: str | None = None
    shadow_trust: str | None = None
    legacy_health: str | None = None
    shadow_health: str | None = None
    independent_groups: int = 0


def _identity(provider: str | None, release_name: str, text: str | None) -> str:
    """Sanitized candidate identity: a digest, never a name or URL."""
    seed = f"{provider or ''}|{release_name}|{reference_fingerprint(text) or ''}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def build_candidates(
    target_filename: str | None,
    references: Sequence[tuple[str, str | None, str | None]],
    *,
    verified_ids: frozenset[str] | None = None,
) -> list[ReferenceCandidate]:
    """Turn already-downloaded references into scored candidates.

    Reuses the existing compatibility machinery for both the hard gate and the
    tier, so v2 never maintains its own content ladder. ``references`` is
    ``(provider, release_name, text)`` for material already fetched in this
    request, so this costs no additional download.
    """
    verified_ids = verified_ids or frozenset()
    target_meta = extract_metadata(target_filename) if target_filename else None
    groups: dict[str, int] = {}
    for provider, name, text in references:
        groups[reference_fingerprint(text) or f"raw:{provider}:{name}"] = (
            groups.get(reference_fingerprint(text) or f"raw:{provider}:{name}", 0) + 1
        )

    candidates: list[ReferenceCandidate] = []
    for provider, name, text in references:
        name = name or ""
        candidate = ReferenceCandidate(
            subtitle_id=_identity(provider, name, text),
            release_name=name,
            provider=provider,
        )
        candidate.structural_profile = StructuralProfile.from_subtitle(text)
        candidate.health = analyze_reference_health(text)
        candidate.health_class = _health_class(candidate.health)
        candidate.independent_group = reference_fingerprint(text)
        candidate.group_size = groups.get(candidate.independent_group or f"raw:{provider}:{name}", 1)

        # --- content identity: the existing machinery, not a new ladder ------
        if target_meta:
            cand_meta = extract_metadata(name)
            accepted, reason, _method = hard_compatibility_filter(target_meta, cand_meta)
            candidate.hard_accepted = accepted
            candidate.hard_reject_reason = reason
            compatibility = calculate_compatibility(
                target_meta, name, is_hash_match=False
            )
            candidate.match_tier = compatibility.match_tier
            candidate.is_hash_match = bool(compatibility.is_hash_match)
            if not accepted:
                candidate.reasons.append(f"hard rejection: {reason}")
        else:
            # No fingerprint: identity is unknown, so nothing is claimed about it.
            candidate.reasons.append(
                "no target filename; content identity unknown and not inferred"
            )

        candidate.cache_verified = candidate.subtitle_id in verified_ids
        if candidate.cache_verified:
            candidate.trust = ReferenceTrust.VERIFIED
            candidate.reasons.append("exact verified cache hit")
        elif candidate.is_hash_match or candidate.match_tier is MatchTier.HASH:
            candidate.trust = ReferenceTrust.STRONG
        elif candidate.match_tier in (MatchTier.EXACT, MatchTier.SOURCE_FAMILY):
            candidate.trust = ReferenceTrust.ACCEPTABLE
        else:
            candidate.trust = ReferenceTrust.UNKNOWN

        if candidate.health_class == "healthy":
            candidate.reasons.append(
                f"healthy: {candidate.health.dialogue_cues} dialogue cues"
            )
        else:
            candidate.reasons.extend(candidate.health.reasons[:2] or ["insufficient dialogue"])
        if candidate.group_size > 1:
            candidate.reasons.append(
                f"one of {candidate.group_size} providers sharing a timing grid"
            )
        candidates.append(candidate)
    return candidates


def select_reference_v2(
    target_filename: str | None,
    references: Sequence[tuple[str, str | None, str | None]],
    *,
    verified_ids: frozenset[str] | None = None,
) -> ReferenceSelection:
    """Shadow selection: the safest timing anchor among already-downloaded refs."""
    candidates = build_candidates(
        target_filename, references, verified_ids=verified_ids
    )
    selection = ReferenceSelection(
        considered=len(candidates),
        hard_rejected=sum(1 for c in candidates if not c.hard_accepted),
    )
    eligible = [c for c in candidates if c.eligible]
    if not eligible:
        selection.reasons.append("no candidate passed the reference quality gates")
        return selection

    ranked = sorted(eligible, key=lambda c: c.sort_key())
    selection.ranked = ranked
    selection.chosen = ranked[0]
    selection.reasons.append(f"selected {ranked[0].explain()}")
    if len(ranked) > 1:
        runner = ranked[1]
        if runner.match_tier == ranked[0].match_tier:
            selection.reasons.append(
                f"same match tier as {runner.subtitle_id}; health decided"
            )
        else:
            selection.reasons.append(
                f"match tier {ranked[0].match_tier.name} outranks "
                f"{runner.subtitle_id} on {runner.match_tier.name}"
            )
    return selection


def compare_selections(
    legacy_id: str | None,
    selection: ReferenceSelection,
    *,
    legacy_trust: str | None = None,
    legacy_health: str | None = None,
) -> SelectionComparison:
    """Compare legacy and shadow picks. Purely observational."""
    shadow = selection.chosen
    comparison = SelectionComparison(
        legacy_id=legacy_id,
        shadow_id=shadow.subtitle_id if shadow else None,
        independent_groups=len({c.independent_group for c in selection.ranked if c.independent_group}),
        legacy_trust=legacy_trust,
        legacy_health=legacy_health,
    )
    if shadow is None:
        comparison.agree = True
        comparison.reasons.append("shadow selector found no eligible candidate")
        return comparison
    comparison.shadow_trust = shadow.trust.value
    comparison.shadow_health = shadow.health_class

    if legacy_id is None or shadow.subtitle_id == legacy_id:
        comparison.reasons.append("legacy and shadow agree")
        return comparison

    comparison.changed = True
    comparison.agree = False
    legacy_match = next(
        (c for c in selection.ranked if c.subtitle_id == legacy_id), None
    )
    comparison.reasons.append("legacy and shadow disagree")
    if legacy_match is None:
        comparison.reasons.append(
            "legacy pick is not even an eligible shadow candidate (hard gate or health)"
        )
    else:
        if legacy_match.match_tier != shadow.match_tier:
            comparison.reasons.append(
                f"match tier differs: legacy {legacy_match.match_tier.name} vs "
                f"shadow {shadow.match_tier.name}"
            )
        if legacy_match.health_class != shadow.health_class:
            comparison.reasons.append(
                f"health differs: legacy {legacy_match.health_class} vs "
                f"shadow {shadow.health_class}"
            )
        if not legacy_match.cache_verified and shadow.cache_verified:
            comparison.reasons.append("shadow candidate has an exact verified cache hit")
    return comparison


def find_tier_contradictions(
    target_filename: str | None,
    references: Sequence[tuple[str, str | None, str | None]],
) -> list[dict[str, Any]]:
    """Where ``MatchTier`` and the legacy ``reference_tier`` disagree.

    Reported, never acted on. Two ladders disagreeing is exactly the drift risk
    that motivated consuming ``MatchTier`` here.
    """
    from app.services.sync.external_strategy import reference_tier

    if not target_filename:
        return []
    target_meta = extract_metadata(target_filename)
    rows: list[dict[str, Any]] = []
    for provider, name, _text in references:
        name = name or ""
        compat = calculate_compatibility(target_meta, name)
        legacy_tier = reference_tier(target_filename, _Legacy(name, provider))
        rows.append(
            {
                "provider": provider,
                "release_name": name,
                "match_tier": compat.match_tier.name,
                "reference_tier": int(legacy_tier),
                "agree": _tiers_agree(compat.match_tier, legacy_tier),
            }
        )
    return rows


class _Legacy:
    """Minimal shim so ``reference_tier`` can read a bare candidate."""

    def __init__(self, release_name: str, provider: str | None) -> None:
        self.release_name = release_name
        self.provider = provider or ""
        self.is_hash_match = False
        self.matched_by_hash = False
        self.hearing_impaired = False
        self.lang = "eng"


def _tiers_agree(match_tier: MatchTier, legacy_tier: int) -> bool:
    """Map both ladders onto a coarse scale and compare.

    MatchTier: 0 HASH, 1 EXACT, 2 SOURCE_FAMILY, 3 CLOSE, 4 FALLBACK.
    reference_tier: 0 HASH, 1 EXACT_GROUP, 2 SOURCE_EDITION, 3 FALLBACK.
    HASH and the strong-identity bands correspond; the tails are coarser in
    reference_tier, so CLOSE and FALLBACK are treated as equivalent.
    """
    band = {
        MatchTier.HASH: 0,
        MatchTier.EXACT: 1,
        MatchTier.SOURCE_FAMILY: 2,
        MatchTier.CLOSE: 3,
        MatchTier.FALLBACK: 3,
    }[match_tier]
    return band == min(legacy_tier, 3)
