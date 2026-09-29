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
from collections.abc import Mapping, Sequence
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
    """Outcome of a shadow selection pass."""

    chosen: ReferenceCandidate | None = None
    ranked: list[ReferenceCandidate] = Field(default_factory=list)
    # Every candidate considered, including ineligible ones. The legacy pick is
    # frequently NOT in `ranked` (e.g. a credits-only reference), and that is
    # exactly the case worth explaining, so it is looked up here.
    all_candidates: list[ReferenceCandidate] = Field(default_factory=list)
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
        all_candidates=candidates,
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


# --- bounded shadow reference pool --------------------------------------- #
#
# The legacy resolver stops at the first candidate that passes cue sanity, so
# the pool of alternatives it hands the shadow selector is chosen BY the policy
# under evaluation. A low disagreement rate then measures the early break, not
# agreement. The pool below gives the shadow selector a small, diversified set
# drawn from the SAME discovered candidate list, without altering which
# candidate the legacy resolver commits to.

COMPARISON_NO = "NO_COMPARISON"
COMPARISON_MEANINGFUL = "MEANINGFUL_COMPARISON"

SHADOW_SWITCH = "SHADOW_SWITCH"


class ReferenceShadowPool(BaseModel):
    """A bounded, diversified set of references the shadow may consider."""

    references: list[tuple[str, str | None, str | None]] = Field(default_factory=list)
    pool_size: int = 0
    pool_limit: int = 0
    available_count: int = 0
    hard_rejected: int = 0
    independent_groups: int = 0
    truncated: bool = False
    limited_by_available_payloads: bool = False
    additional_fetches: int = 0
    comparison_class: str = COMPARISON_NO
    reasons: list[str] = Field(default_factory=list)


def _hard_accepts(target_filename: str | None, release_name: str) -> tuple[bool, str | None]:
    """Cheap metadata-only admissibility. No download, no alignment."""
    if not target_filename:
        return True, None
    try:
        target_meta = extract_metadata(target_filename)
        cand_meta = extract_metadata(release_name)
    except Exception:  # pragma: no cover - malformed names are not fatal
        return False, "unparseable release name"
    if target_meta is None or cand_meta is None:
        return True, None
    accepted, reason, _method = hard_compatibility_filter(target_meta, cand_meta)
    return accepted, reason


def build_shadow_pool(
    target_filename: str | None,
    discovered: Sequence[tuple[str, str]],
    payloads: Mapping[tuple[str, str], str],
    *,
    limit: int,
) -> ReferenceShadowPool:
    """Assemble a bounded reference pool from already-materialized payloads.

    ``discovered`` is the legacy-ranked ``(provider, release_name)`` list;
    ``payloads`` maps that same pair to subtitle text already fetched in this
    request. Nothing here downloads, and nothing here is fed back into the
    legacy decision.

    Grid diversification is two-pass: one representative per distinct timing
    grid first, then any remaining slots filled in rank order. That prevents
    four same-grid copies from masquerading as four independent references
    while still filling the pool when genuinely distinct candidates exist.
    """
    pool = ReferenceShadowPool(pool_limit=limit)
    if limit <= 0:
        pool.reasons.append("reference shadow pool disabled (limit 0)")
        return pool

    seen_keys: set[tuple[str, str]] = set()
    eligible: list[tuple[str, str, str, str | None]] = []
    for provider, name in discovered:
        key = (provider, name)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        text = payloads.get(key)
        if not text:
            continue
        pool.available_count += 1
        accepted, reason = _hard_accepts(target_filename, name)
        if not accepted:
            pool.hard_rejected += 1
            continue
        eligible.append((provider, name, text, reference_fingerprint(text)))

    if not eligible:
        pool.reasons.append("no materializable reference payload available")
        return pool

    chosen: list[tuple[str, str, str, str | None]] = []
    seen_grids: set[str] = set()
    for entry in eligible:
        grid = entry[3] or ""
        if grid and grid in seen_grids:
            continue
        if grid:
            seen_grids.add(grid)
        chosen.append(entry)
        if len(chosen) >= limit:
            break
    if len(chosen) < limit:
        for entry in eligible:
            if entry in chosen:
                continue
            chosen.append(entry)
            if len(chosen) >= limit:
                break

    pool.references = [(p, n, t) for p, n, t, _g in chosen]
    pool.pool_size = len(pool.references)
    pool.independent_groups = len({g for _p, _n, _t, g in chosen if g})
    pool.truncated = len(eligible) > limit
    pool.limited_by_available_payloads = pool.pool_size < limit and not pool.truncated
    pool.comparison_class = classify_comparison(pool)
    if pool.truncated:
        pool.reasons.append(
            f"pool truncated: {len(eligible)} admissible candidates, limit {limit}"
        )
    if pool.limited_by_available_payloads:
        pool.reasons.append(
            "shadow_pool_limited_by_available_payloads: "
            "fewer payloads were materialized than the pool limit allows"
        )
    return pool


def classify_comparison(pool: ReferenceShadowPool) -> str:
    """A comparison only means something when a real alternative existed.

    ``legacy == shadow`` when the pool held a single candidate is not evidence
    of policy agreement; it is an artifact of the legacy early break.
    """
    if pool.pool_size < 2 or pool.independent_groups < 2:
        return COMPARISON_NO
    return COMPARISON_MEANINGFUL


def classify_shadow_switch(
    legacy: ReferenceCandidate | None, shadow: ReferenceCandidate | None
) -> list[str]:
    """Describe why the two policies differ, without claiming v2 is better."""
    labels: list[str] = []
    if shadow is None:
        return ["shadow produced no eligible reference"]
    if legacy is None:
        return ["legacy reference absent from the pool"]
    if legacy.subtitle_id == shadow.subtitle_id:
        return ["same reference"]
    labels.append(SHADOW_SWITCH)
    if legacy.match_tier is shadow.match_tier:
        labels.append("same_match_tier")
    else:
        labels.append(
            f"different_match_tier: {legacy.match_tier.value} -> {shadow.match_tier.value}"
        )
    if not legacy.eligible:
        labels.append(
            f"legacy reference not eligible as an anchor ({legacy.health_class})"
        )
    if legacy.health_class != shadow.health_class:
        labels.append(f"health: {legacy.health_class} -> {shadow.health_class}")
    if legacy.cache_verified != shadow.cache_verified:
        labels.append("verified_evidence: differs")
    if legacy.independent_group and legacy.independent_group != shadow.independent_group:
        labels.append("different_independent_group")
    if shadow.health.dialogue_cues < legacy.health.dialogue_cues:
        labels.append("lower_dialogue_quality")
    return labels


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


_TIER_BANDS: Mapping[MatchTier, int] = {
    MatchTier.HASH: 0,
    MatchTier.EXACT: 1,
    MatchTier.SOURCE_FAMILY: 2,
    MatchTier.CLOSE: 3,
    MatchTier.FALLBACK: 3,
}


def tier_band(match_tier: MatchTier) -> int:
    """Coarse identity band: 0 HASH, 1 EXACT, 2 SOURCE_FAMILY, 3 weak.

    Two references in the same band are interchangeable as far as *content
    identity* is concerned. Adjacent tiers inside one band (CLOSE vs FALLBACK,
    or two arbitrary release groups that both merely sit in SOURCE_FAMILY) are
    metadata noise, not evidence, and must never be allowed to drive a
    decision on their own.
    """
    return _TIER_BANDS.get(match_tier, 3)


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

# ---------------------------------------------------------------------------
# TARGET-BOUND SELECTION (promoted policy)
# ---------------------------------------------------------------------------
#
# The shadow selector above already makes a target-bound choice: its ordering
# is content identity, health and trust, and it never sees a subtitle the user
# asked for. What it lacked was a rule for what happens when a reference is
# *rejected* against that candidate, and that rule is the whole production bug.
#
# The legacy loop treated "fails against this candidate" as "this reference is
# wrong", then walked down the list until some reference happened to fit. That
# inverts the causal order. A reference belongs to the target or it does not;
# the requested subtitle is evidence about the *candidate*, not about the
# reference.

NO_TARGET_BOUND_REFERENCE = "NO_TARGET_BOUND_REFERENCE"
REFERENCE_TARGET_MISMATCH = "REFERENCE_TARGET_MISMATCH"
REFERENCE_LOW_HEALTH = "REFERENCE_LOW_HEALTH"
CANDIDATE_DIFFERENT_TIMELINE = "CANDIDATE_DIFFERENT_TIMELINE"
TARGET_REFERENCE_SELECTED = "TARGET_REFERENCE_SELECTED"

_MAX_REASON_LEN = 120


def release_family_key(release_name: str) -> str:
    """Coarse release identity, used to collapse one timing model to one slot."""
    name = (release_name or "").lower()
    try:
        meta = extract_metadata(name) or {}
    except Exception:  # pragma: no cover - a malformed name is not fatal
        meta = {}
    source = str(meta.get("source") or "").strip()
    edition = str(meta.get("edition") or "").strip()
    group = str(meta.get("group") or "").strip()
    if source or group:
        return "|".join((source, edition, group))
    return name.rsplit(".", 1)[0].strip()


class ReferenceAttempt(BaseModel):
    """One reference the resolver fetched, and what happened to it."""

    reference_id: str
    release_name: str
    provider: str | None = None
    match_tier: MatchTier = MatchTier.FALLBACK
    hard_accepted: bool = True
    health_class: str = "unknown"
    family: str = ""
    downloaded: bool = False
    cue_sanity_ok: bool | None = None
    reasons: list[str] = Field(default_factory=list)

    @property
    def anchorable(self) -> bool:
        """Eligible to be the target's anchor.

        Health is deliberately *not* a veto here. The health thresholds were
        calibrated for shadow measurement, and promoting them to a hard
        production gate was measured to refuse references the previous policy
        accepted: reference loss is a worse failure than a weak anchor. Health
        still orders otherwise-equal candidates, and a reference rejected
        against the target can never be rescued by it.
        """
        return self.hard_accepted and self.downloaded and self.cue_sanity_ok is not None


class TargetBoundDecision(BaseModel):
    """Step A: the target's reference, decided without reference to a candidate."""

    selected: ReferenceAttempt | None = None
    attempts: list[ReferenceAttempt] = Field(default_factory=list)
    outcome: str = NO_TARGET_BOUND_REFERENCE
    reasons: list[str] = Field(default_factory=list)
    downloads_used: int = 0
    unique_families: int = 0
    duplicate_families_avoided: int = 0
    suppressed_as_circular: list[str] = Field(default_factory=list)
    mode: str = "target_bound"

    @property
    def usable(self) -> bool:
        return self.selected is not None


def decide_target_bound_reference(
    attempts: Sequence[ReferenceAttempt],
) -> TargetBoundDecision:
    """Choose the target's reference from a set of attempted references.

    The anti-circularity rule, stated once: a reference that fits the candidate
    better may never displace a reference that matches the target better, even
    if the better-fitting reference is the only one the candidate agrees with.

    Implemented as a tier floor. If any target-accepted reference failed against
    the candidate, only references at least as strong against the *target* may
    still be selected. A weaker reference that merely fits is suppressed, and
    the decision fails closed.
    """
    decision = TargetBoundDecision(attempts=list(attempts))
    decision.downloads_used = sum(1 for a in attempts if a.downloaded)
    families = [a.family or a.reference_id for a in attempts if a.downloaded]
    decision.unique_families = len(set(families))
    decision.duplicate_families_avoided = max(0, len(families) - len(set(families)))

    hard_rejected = [a for a in attempts if not a.hard_accepted]
    low_health = [a for a in attempts if a.health_class == "unusable"]
    if hard_rejected:
        decision.reasons.append(
            f"{REFERENCE_TARGET_MISMATCH}: {len(hard_rejected)} reference(s) "
            "did not match the target video"
        )
    if low_health:
        decision.reasons.append(
            f"{REFERENCE_LOW_HEALTH}: {len(low_health)} reference(s) unusable as anchors"
        )

    anchorable = [a for a in attempts if a.anchorable]
    if not anchorable:
        decision.outcome = NO_TARGET_BOUND_REFERENCE
        decision.reasons.append(
            f"{NO_TARGET_BOUND_REFERENCE}: no target-accepted reference was usable"
        )
        return decision

    failed = [a for a in anchorable if a.cue_sanity_ok is False]
    best_failed_tier = min((int(a.match_tier) for a in failed), default=None)
    best_failed_band = min((tier_band(a.match_tier) for a in failed), default=None)

    passing = [a for a in anchorable if a.cue_sanity_ok is True]
    admissible: list[ReferenceAttempt] = []
    for attempt in passing:
        if best_failed_band is not None and tier_band(attempt.match_tier) > best_failed_band:
            decision.suppressed_as_circular.append(attempt.reference_id)
            attempt.reasons.append(
                "suppressed: fits the candidate but matches the target worse than a "
                "reference already proven incompatible; candidate timing may not "
                "choose the reference"
            )
            continue
        admissible.append(attempt)

    if admissible:
        selected = min(
            admissible,
            key=lambda a: (
                int(a.match_tier),
                0 if a.health_class != "unusable" else 1,
                a.reference_id,
            ),
        )
        decision.selected = selected
        decision.outcome = TARGET_REFERENCE_SELECTED
        decision.reasons.append(
            f"{TARGET_REFERENCE_SELECTED}: match_tier={selected.match_tier.name}"
        )
        if selected.family:
            decision.reasons.append(
                f"reference family={selected.family[:_MAX_REASON_LEN]}"
            )
        if best_failed_tier is not None:
            decision.reasons.append(
                "a stronger target match disagreed with the candidate "
                f"(tier<={MatchTier(best_failed_tier).name}); the candidate is the outlier"
            )
        return decision

    if best_failed_tier is not None:
        decision.outcome = CANDIDATE_DIFFERENT_TIMELINE
        decision.reasons.append(
            f"{CANDIDATE_DIFFERENT_TIMELINE}: the target-bound reference "
            f"(tier<={MatchTier(best_failed_tier).name}) does not share this "
            "candidate's timeline; refusing to substitute a weaker reference "
            "that merely fits"
        )
        return decision

    decision.outcome = NO_TARGET_BOUND_REFERENCE
    decision.reasons.append(
        f"{NO_TARGET_BOUND_REFERENCE}: no reference could be anchored to the target"
    )
    return decision


def reference_family_counts(attempts: Sequence[ReferenceAttempt]) -> dict[str, int]:
    """Per-family download counts, for the duplicate-collapse metric."""
    counts: dict[str, int] = {}
    for attempt in attempts:
        if not attempt.downloaded:
            continue
        key = attempt.family or attempt.reference_id
        counts[key] = counts.get(key, 0) + 1
    return counts


def summarize_target_bound(decision: TargetBoundDecision) -> dict[str, Any]:
    """Audit shape. Counts and hashed identities only: no text, URLs or keys."""
    return {
        "selected": decision.selected.reference_id if decision.selected else None,
        "selection_mode": decision.mode,
        "match_tier": decision.selected.match_tier.name if decision.selected else None,
        "reference_family": decision.selected.family if decision.selected else None,
        "outcome": decision.outcome,
        "downloads_used": decision.downloads_used,
        "unique_families": decision.unique_families,
        "duplicate_families_avoided": decision.duplicate_families_avoided,
        "suppressed_as_circular": len(decision.suppressed_as_circular),
        "reasons": [r[:_MAX_REASON_LEN] for r in decision.reasons[:8]],
    }
