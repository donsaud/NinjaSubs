"""Controlled, opt-in expansion of the reference shadow pool.

This module decides whether the shadow experiment may spend an extra network
request to widen its own evidence. It is strictly experimental.

The boundary, in one place:

    Production     legacy reference selection -> legacy synchronization
    Experimental   shadow pool -> optional extra fetch -> v2 -> audit

Nothing here returns a reference, and nothing here is called by the production
selection path. The only thing this module can produce is permission to fetch
one more candidate *for measurement*, plus the metadata needed to judge whether
that fetch was worth anything.

Three rules keep the default cost at zero:

1. The budget is zero unless the audit log is actually recording. A stale or
   mistyped setting cannot by itself buy extra production traffic.
2. The budget is only spent when the existing pool cannot support a comparison.
   A pool that already offers two independent timing models is left alone.
3. A request with no target fingerprint never fetches. Without release-level
   evidence the extra candidate cannot answer the question being asked, so
   spending bandwidth on it would be waste rather than evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field

from app.services.subtitle_matcher import extract_metadata
from app.services.sync.reference_v2 import COMPARISON_MEANINGFUL, _hard_accepts

# Outcome vocabulary. These describe what an extra fetch did to the evidence.
# None of them claims quality: only a verified outcome could, and this
# experiment deliberately does not produce one.
EXPANSION_NOT_ATTEMPTED = "NOT_ATTEMPTED"
EXPANSION_NO_VALUE = "NO_ADDITIONAL_VALUE"
EXPANSION_DUPLICATE_GROUP = "ADDED_DUPLICATE_GROUP"
EXPANSION_NEW_TIMING_GROUP = "ADDED_NEW_TIMING_GROUP"
EXPANSION_ENABLED_COMPARISON = "ENABLED_MEANINGFUL_COMPARISON"
EXPANSION_FETCH_FAILURE = "CAUSED_FETCH_FAILURE"

FETCH_CACHE_REUSE = "CACHE_REUSE"
FETCH_NETWORK = "NETWORK_FETCH"

# Why no fetch was permitted, for telemetry.
REFUSED_AUDIT_DISABLED = "audit_disabled"
REFUSED_LIMIT_ZERO = "fetch_limit_zero"
REFUSED_NO_FINGERPRINT = "no_target_fingerprint"
REFUSED_POOL_SUFFICIENT = "pool_already_comparable"
REFUSED_NO_CANDIDATE = "no_eligible_candidate"


class FetchBudget(BaseModel):
    """Permission to spend extra network requests, and why."""

    allowed: int = 0
    reason: str = REFUSED_LIMIT_ZERO


class ExpansionChoice(BaseModel):
    """The one extra candidate to consider, with the reason it was picked."""

    provider: str
    release_name: str
    key: tuple[str, str]
    reason: str = "unseen_source"
    # Sanitized identity for telemetry; never the release name or URL.
    reference_id: str | None = None


class ExpansionTelemetry(BaseModel):
    """What one extra fetch did to the evidence, and what it cost."""

    eligible: bool = False
    refusal: str | None = None
    outcome: str = EXPANSION_NOT_ATTEMPTED
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    network_fetches: int = 0
    cache_reuses: int = 0
    bytes: int | None = None
    latency_ms: float | None = None
    pool_size_before: int = 0
    independent_groups_before: int = 0
    pool_size_after: int = 0
    independent_groups_after: int = 0
    meaningful_before: bool = False
    meaningful_after: bool = False
    duplicate_timing_grid: bool | None = None
    # Safe per-candidate facts. No text, no URL, no credentials.
    candidate_provider: str | None = None
    candidate_match_tier: str | None = None
    candidate_source_class: str | None = None
    changed_shadow_reference: bool | None = None
    reasons: list[str] = Field(default_factory=list)


def has_target_fingerprint(target_filename: str | None) -> bool:
    """True when the target carries release-level evidence.

    Without it the experiment cannot distinguish a good reference from a
    coincidentally well-timed one, so no extra bandwidth is spent.
    """
    if not target_filename:
        return False
    try:
        meta = extract_metadata(target_filename)
    except Exception:
        return False
    if not meta:
        return False
    return bool(meta.get("title"))


def resolve_fetch_budget(
    *,
    configured_limit: int,
    audit_enabled: bool,
    target_filename: str | None,
    comparison_class: str,
) -> FetchBudget:
    """Decide how many extra fetches this request may spend, if any."""
    if not audit_enabled:
        # The audit log is the reason to measure at all. Without it the fetch
        # would cost bandwidth and produce no record, so it is refused even if
        # the limit is configured.
        return FetchBudget(allowed=0, reason=REFUSED_AUDIT_DISABLED)
    if configured_limit <= 0:
        return FetchBudget(allowed=0, reason=REFUSED_LIMIT_ZERO)
    if not has_target_fingerprint(target_filename):
        return FetchBudget(allowed=0, reason=REFUSED_NO_FINGERPRINT)
    if comparison_class == COMPARISON_MEANINGFUL:
        # The pool already supports a comparison; an extra candidate would add
        # traffic without adding evidence.
        return FetchBudget(allowed=0, reason=REFUSED_POOL_SUFFICIENT)
    return FetchBudget(allowed=int(configured_limit), reason="expansion permitted")


def _source_key(provider: str, release_name: str) -> tuple[str, str]:
    """Provider plus release group: a cheap proxy for a distinct source.

    Used only to order candidates. Independence is never inferred from this;
    it is measured afterwards from the actual timing grid.
    """
    group = ""
    try:
        meta = extract_metadata(release_name) or {}
        group = str(meta.get("group") or "")
    except Exception:
        group = ""
    return (provider, group)


def choose_expansion_candidate(
    target_filename: str | None,
    discovered: Sequence[tuple[str, str]],
    payloads: Mapping[tuple[str, str], str],
    *,
    rejected: Sequence[tuple[str, str]] = (),
) -> ExpansionChoice | None:
    """Pick the single extra candidate most likely to fill an evidence gap.

    Reuses the existing normalized machinery: provider/rank order from the same
    discovery pass, the existing hard compatibility filter, and release
    metadata. This is a tie-breaker inside an experiment, not a second content
    selection system.

    Candidates whose provider and release group are already represented in the
    pool are considered last, because a different source is the best cheap
    signal that a different timing model might be available. Selection is
    deterministic: the first surviving candidate in discovery order wins.
    """
    from app.services.sync.reference_v2 import _identity

    rejected_keys = set(rejected)
    represented = {_source_key(p, n) for (p, n) in payloads}

    eligible: list[tuple[tuple[int, int], tuple[str, str]]] = []
    seen: set[tuple[str, str]] = set()
    for index, (provider, name) in enumerate(discovered):
        key = (provider, name)
        if key in seen or key in rejected_keys or key in payloads:
            continue
        seen.add(key)
        accepted, _reason = _hard_accepts(target_filename, name)
        if not accepted:
            continue
        already = 0 if _source_key(provider, name) in represented else 1
        eligible.append(((-already, index), key))

    if not eligible:
        return None
    eligible.sort()
    provider, name = eligible[0][1]
    source = _source_key(provider, name)
    return ExpansionChoice(
        provider=provider,
        release_name=name,
        key=(provider, name),
        reason="unseen_source" if source not in represented else "rank_order",
        reference_id=_identity(provider, name, None),
    )


def classify_expansion(
    *,
    eligible: bool,
    refusal: str | None,  # noqa: ARG001 - carried in telemetry, not classification
    attempted: bool,
    succeeded: bool,
    pool_size_before: int,
    groups_before: int,
    pool_size_after: int,
    groups_after: int,
    meaningful_before: bool,
    meaningful_after: bool,
) -> str:
    """Name what the extra fetch did to the evidence.

    Ordering matters: enabling a comparison is the outcome the experiment cares
    about most, then a genuinely new timing model, then a duplicate that added
    a payload but no independence, then nothing at all.
    """
    if not eligible:
        return EXPANSION_NOT_ATTEMPTED
    if attempted and not succeeded:
        return EXPANSION_FETCH_FAILURE
    if not attempted:
        return EXPANSION_NO_VALUE  # pragma: no cover - defensive
    if meaningful_after and not meaningful_before:
        return EXPANSION_ENABLED_COMPARISON
    if groups_after > groups_before:
        return EXPANSION_NEW_TIMING_GROUP
    if pool_size_after > pool_size_before:
        # A payload was added but it shared an existing timing grid, so it is
        # not independent evidence.
        return EXPANSION_DUPLICATE_GROUP
    return EXPANSION_NO_VALUE


def expansion_summary(telemetry: ExpansionTelemetry) -> dict[str, Any]:
    """Flat view for the audit record."""
    return {
        "eligible": telemetry.eligible,
        "refusal": telemetry.refusal,
        "outcome": telemetry.outcome,
        "attempts": telemetry.attempts,
        "successes": telemetry.successes,
        "failures": telemetry.failures,
        "network_fetches": telemetry.network_fetches,
        "cache_reuses": telemetry.cache_reuses,
        "bytes": telemetry.bytes,
        "latency_ms": telemetry.latency_ms,
    }
