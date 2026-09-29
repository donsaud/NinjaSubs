"""Deterministic hierarchical ordering for subtitle results.

Existing behaviour is preserved, not replaced. ``MatchTier`` and
``calculate_compatibility()`` remain the authority on *which release this
subtitle belongs to*; this module only decides how to present a set that may
additionally carry synchronization evidence.

The comparator is a strict lexicographic ordering, never a weighted sum, so no
threshold change can silently reshuffle results:

1. rejected candidates last, and only when something better exists;
2. synchronization evidence, strongest first;
3. the existing ``MatchTier`` hierarchy;
4. synchronization confidence;
5. the existing compatibility percentage;
6. stable, content-derived tie-breakers.

Two deliberate refusals:

* A ``VERIFIED_RESYNCED`` subtitle never outranks a ``VERIFIED_SYNCED`` one.
  They are different claims, and "we re-timed this" is not better evidence than
  "it was already right".
* Synchronization state never overrides hard compatibility. A wrong episode is
  dropped upstream by ``hard_compatibility_filter``; nothing here can promote
  it.
"""

from __future__ import annotations

from typing import Any

from app.models import MatchTier

# Sync-state presentation order (lower sorts first). Mirrors SyncState.rank but
# is declared here so ordering does not depend on enum import order.
_STATE_ORDER = {
    "verified_synced": 0,
    "verified_resynced": 1,
    "probable_sync": 2,
    "unverified": 3,
    "rejected": 4,
    None: 3,  # no claim made at all ranks with UNVERIFIED, never as verified
}

# Evidence strength (lower is stronger). UNKNOWN sorts last so a candidate with
# no claim never outranks one that was actually checked.
_AVAILABILITY_ORDER = {
    "verified": 0,
    "cached": 1,
    "predicted": 2,
    "unknown": 3,
    None: 3,
}


def _state_of(item: Any) -> str | None:
    value = getattr(item, "sync_state", None)
    if value is None and isinstance(item, dict):
        value = item.get("sync_state")
    return value if isinstance(value, str) else None


def _availability_of(item: Any) -> str | None:
    value = getattr(item, "sync_verification", None)
    if value is None and isinstance(item, dict):
        value = item.get("sync_verification")
    return value if isinstance(value, str) else None


def _tier_of(item: Any) -> int:
    tier = getattr(item, "match_tier", None)
    if tier is None and isinstance(item, dict):
        tier = item.get("match_tier")
    if tier is None:
        return int(MatchTier.FALLBACK)
    try:
        return int(tier)
    except (TypeError, ValueError):
        return int(MatchTier.FALLBACK)


def _score_of(item: Any) -> int:
    for name in ("match_percentage", "score"):
        value = getattr(item, name, None)
        if value is None and isinstance(item, dict):
            value = item.get(name)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return 0


def _sync_confidence_of(item: Any) -> float:
    for name in ("sync_confidence", "sync_confidence_score"):
        value = getattr(item, name, None)
        if value is None and isinstance(item, dict):
            value = item.get(name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    # No claim made: the lowest possible confidence, never a middling default.
    return -1.0


def _identity(item: Any) -> tuple[str, str]:
    """Content-derived tie-breaker; independent of dict/provider ordering."""
    provider = getattr(item, "provider", None) or (
        item.get("provider") if isinstance(item, dict) else ""
    )
    name = getattr(item, "release_name", None) or (
        item.get("release_name") if isinstance(item, dict) else ""
    )
    return (str(provider or ""), str(name or ""))


def comparison_key(item: Any) -> tuple:
    """Full sort key. Exposed so the ordering is directly testable."""
    state = _state_of(item)
    return (
        _STATE_ORDER.get(state, _STATE_ORDER[None]),
        _AVAILABILITY_ORDER.get(_availability_of(item), _AVAILABILITY_ORDER[None]),
        _tier_of(item),
        -_sync_confidence_of(item),
        -_score_of(item),
        *_identity(item),
    )


def order_candidates(candidates: list[Any]) -> list[Any]:
    """Return candidates in deterministic presentation order.

    The input is not mutated, and equal keys keep their input order, so two runs
    over the same set always produce the same output.
    """
    return sorted(candidates, key=comparison_key)


def has_stronger_sync_evidence(left: Any, right: Any) -> bool:
    """True when ``left`` presents stronger synchronization evidence."""
    left_key = (_STATE_ORDER.get(_state_of(left), 3), _AVAILABILITY_ORDER.get(_availability_of(left), 3))
    right_key = (_STATE_ORDER.get(_state_of(right), 3), _AVAILABILITY_ORDER.get(_availability_of(right), 3))
    return left_key < right_key
