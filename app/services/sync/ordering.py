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
4. same-video identity (``is_hash_match``);
5. synchronization confidence;
6. the existing compatibility percentage;
7. stable, content-derived tie-breakers.

Same-video identity is deliberately subordinate to synchronization state. A
MovieHash match proves WHICH FILE a subtitle belongs to; it says nothing about
whether its timings are correct. A candidate the verifier confirmed always
outranks a hash-exact candidate whose timing is unknown, and a prediction never
counts as verification.

Two deliberate refusals:

* A ``VERIFIED_RESYNCED`` subtitle never outranks a ``VERIFIED_SYNCED`` one.
  They are different claims, and "we re-timed this" is not better evidence than
  "it was already right".
* Synchronization state never overrides hard compatibility. A wrong episode is
  dropped upstream by ``hard_compatibility_filter``; nothing here can promote
  it.

**The existing rank is reused, not recomputed.** Level 3 defers to the position
the matcher already assigned, which is its own encoding of MatchTier ->
confidence -> compatibility score -> tie-breakers. Re-deriving that here would
create a second, divergent implementation of ordering that could disagree with
the matcher. When no candidate carries synchronization evidence every level
above is identical, so the output is byte-identical to the matcher's own order.
"""

from __future__ import annotations

from typing import Any

from app.models import MatchTier
from app.services.sync.alignment import VerificationAvailability
from app.services.sync.predictor import CONFIDENCE_PREDICTION_FLOOR as PREDICTION_ACTION_FLOOR

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
    """Content tier, exposed for inspection; ordering defers to ``_base_rank``."""
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


def _identity_rank(item: Any) -> int:
    """Same-video identity evidence: lower is stronger.

    Consumes ``is_hash_match`` and nothing else. That flag is set in exactly one
    place -- ``OpenSubtitlesProvider`` sets it only when the request actually
    carried a ``moviehash`` AND OpenSubtitles returned
    ``attributes.moviehash_match is True`` -- and it is a ``StrictBool``, so no
    truthy string can reach it. SubDL, SubSource and the other providers cannot
    inherit it. This function therefore never *infers* identity: it reads a
    provenance flag that was already earned, and it never mutates the candidate.

    Identity is deliberately NOT inferred from provider name, filename
    similarity, release name, MatchTier, video size, language, prediction, or
    sync state. A release-name guess is compatibility evidence, not
    same-video identity, and conflating the two is exactly the mistake this
    ranking exists to avoid.

    Note the deliberate absence of any hash *computation*: the addon receives a
    hash when the client supplies one and never derives one from video bytes.
    """
    value = getattr(item, "is_hash_match", None)
    if value is None and isinstance(item, dict):
        value = item.get("is_hash_match")
    return 0 if value is True else 1


def _base_rank(item: Any) -> int:
    """Position assigned by the existing matcher, when recorded.

    This is the canonical encoding of MatchTier -> confidence -> compatibility
    score -> tie-breakers, so deferring to it keeps a single source of truth for
    content ordering. Large default so unranked items sort after ranked ones.
    """
    for name in ("sync_base_rank", "base_rank"):
        value = getattr(item, name, None)
        if value is None and isinstance(item, dict):
            value = item.get(name)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return 1_000_000


def _identity(item: Any) -> tuple[str, str]:
    """Content-derived tie-breaker; independent of dict/provider ordering."""
    provider = getattr(item, "provider", None) or (
        item.get("provider") if isinstance(item, dict) else ""
    )
    name = getattr(item, "release_name", None) or (
        item.get("release_name") if isinstance(item, dict) else ""
    )
    return (str(provider or ""), str(name or ""))


def _lang_rank(item: Any) -> int:
    """Position of the candidate's language in the user's preferred list.

    Language preference is an explicit user choice, not a heuristic, so it is
    the leading key: no synchronization evidence may reorder languages.
    """
    value = getattr(item, "sync_lang_rank", None)
    if value is None and isinstance(item, dict):
        value = item.get("sync_lang_rank")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    return 0


def _normalized_evidence(item: Any) -> tuple[str | None, str | None]:
    """Return ``(state, availability)`` after applying the prediction floor.

    A ``PREDICTED`` claim below the action floor is demoted to
    ``UNVERIFIED``/``UNKNOWN``. Enforcing this in the comparator rather than only
    at the call site means a weak prediction cannot promote a candidate even if
    a caller forgets to check it, so "prediction is evidence, never a claim"
    holds structurally.
    """
    state = _state_of(item)
    availability = _availability_of(item)
    if availability != VerificationAvailability.PREDICTED:
        return state, availability
    if _sync_confidence_of(item) < PREDICTION_ACTION_FLOOR:
        return "unverified", "unknown"
    return state, availability


def comparison_key(item: Any) -> tuple:
    """Full sort key. Exposed so the ordering is directly testable.

    Precedence: user language preference, sync state, evidence availability,
    the matcher's own rank, same-video identity, sync confidence, compatibility
    score, then stable identity tie-breakers.

    Same-video identity sits deliberately BELOW synchronization state and
    evidence, so a hash-exact subtitle with unverified timing never outranks a
    hashless subtitle the verifier actually confirmed. It sits ABOVE sync
    confidence, the compatibility percentage and provider/release, because among
    candidates with comparable timing evidence, proof that the subtitle belongs
    to THIS video is the next strongest discriminator.
    """
    state, availability = _normalized_evidence(item)
    return (
        _lang_rank(item),
        _STATE_ORDER.get(state, _STATE_ORDER[None]),
        _AVAILABILITY_ORDER.get(availability, _AVAILABILITY_ORDER[None]),
        _base_rank(item),
        _identity_rank(item),
        -_sync_confidence_of(item),
        -_score_of(item),
        *_identity(item),
    )


def order_candidates(candidates: list[Any]) -> list[Any]:
    """Return candidates in deterministic presentation order.

    The input is not mutated. ``sorted`` is stable, so candidates with equal
    keys keep their input order, making repeated runs identical.
    """
    return sorted(candidates, key=comparison_key)


def has_stronger_sync_evidence(left: Any, right: Any) -> bool:
    """True when ``left`` presents stronger synchronization evidence."""
    left_key = (_STATE_ORDER.get(_state_of(left), 3), _AVAILABILITY_ORDER.get(_availability_of(left), 3))
    right_key = (_STATE_ORDER.get(_state_of(right), 3), _AVAILABILITY_ORDER.get(_availability_of(right), 3))
    return left_key < right_key
