"""Presentation ordering delegates to the authoritative Compatibility Score.

Eligibility is enforced upstream. Trusted target-bound evidence is incorporated
into that score before sorting; sync states and predictions cannot override its
displayed value. Evidence inspection helpers remain available for diagnostics.
"""

from __future__ import annotations

from typing import Any

from app.services.sync.alignment import VerificationAvailability
from app.services.sync.predictor import CONFIDENCE_PREDICTION_FLOOR as PREDICTION_ACTION_FLOOR

# Diagnostic evidence order only; never a visible presentation key.
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


def _normalized_evidence(item: Any) -> tuple[str | None, str | None]:
    """Return ``(state, availability)`` after applying the prediction floor.

    A ``PREDICTED`` claim below the action floor is demoted to
    ``UNVERIFIED``/``UNKNOWN``. This diagnostic normalization preserves the action-floor interpretation.
    Presentation never independently promotes a prediction, regardless of
    confidence.
    """
    state = _state_of(item)
    availability = _availability_of(item)
    if availability != VerificationAvailability.PREDICTED:
        return state, availability
    if _sync_confidence_of(item) < PREDICTION_ACTION_FLOOR:
        return "unverified", "unknown"
    return state, availability


def _ranked_confidence(item: Any, availability: str | None) -> float:
    """Sync confidence, counted only when attached to a real evidence claim.

    A prediction demoted below the action floor keeps its stored confidence
    number, but ``_normalized_evidence`` already stripped its claim. Counting
    that orphaned number would let a weak prediction outrank an unknown
    candidate purely on a score the system itself refused to trust, so
    availability ``unknown`` (or missing) always ranks at the floor (-1.0).
    """
    if availability is None or availability == "unknown":
        return -1.0
    return _sync_confidence_of(item)


def comparison_key(item: Any) -> tuple:
    """Displayed score descending, then preferences, quality and stable identity."""
    from app.services.visible_compatibility import presentation_key

    return presentation_key(item)


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
