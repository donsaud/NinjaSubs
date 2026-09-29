"""Metadata-only prediction of how likely a subtitle is already synchronized.

This is a *ranking aid*, never a synchronization claim. It consumes the
compatibility verdict the matcher already produced
(:class:`~app.services.subtitle_matcher.CompatibilityResult`) rather than
re-parsing filenames, re-deriving episodes, or duplicating the FPS logic.

The single invariant it must never break:

    a prediction can only ever produce ``PROBABLE_SYNC`` or a non-positive
    state, and always with ``VerificationAvailability.PREDICTED``.

``VERIFIED_SYNCED`` and ``VERIFIED_RESYNCED`` are reachable only from a
measurement, so this module has no code path to them at all.

Two hard refusals:

* **No fingerprint, no prediction.** A catalogue request carries only
  title/season/episode. Inferring synchronization from a subtitle filename
  there is exactly the bug that produced a fabricated target video earlier.
* **No rescuing a hard rejection.** ``accepted=False`` stays rejected however
  strong the metadata looks.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from app.models import MatchTier
from app.services.sync.alignment import SyncState, VerificationAvailability
from app.services.sync.matching import has_video_fingerprint

logger = logging.getLogger(__name__)

# Confidence floors per deterministic rule. These are not weights to be summed;
# each rule is a separate branch, and the strongest matching rule wins outright.
CONFIDENCE_HASH = 95.0
CONFIDENCE_EXACT_IDENTITY = 90.0
CONFIDENCE_SOURCE_EDITION = 75.0
# Below this, even a matching source family is too weak to predict anything.
# Rule observed: the "title + episode only" case stays UNVERIFIED.
CONFIDENCE_PREDICTION_FLOOR = 70.0


class SyncPrediction(BaseModel):
    """A ranking hint. Never a verification."""

    availability: VerificationAvailability = VerificationAvailability.UNKNOWN
    predicted_state: SyncState = SyncState.UNVERIFIED
    confidence: float | None = None
    reasons: list[str] = Field(default_factory=list)
    # Which rule fired, for logs and for asserting determinism in tests.
    rule: str = "none"

    @property
    def is_actionable(self) -> bool:
        """True when this prediction may influence ordering."""
        return (
            self.availability is VerificationAvailability.PREDICTED
            and self.predicted_state is SyncState.PROBABLE_SYNC
        )

    def explain(self) -> str:
        head = f"{self.availability.value}/{self.predicted_state.value}"
        if self.confidence is not None:
            head += f" conf={self.confidence:.0f}"
        return f"{head} [{self.rule}]" + (f" | {'; '.join(self.reasons)}" if self.reasons else "")


def _compatibility(release: Any) -> Any | None:
    compat = getattr(release, "compatibility", None)
    if compat is None and isinstance(release, dict):
        compat = release.get("compatibility")
    return compat


def _tier(release: Any) -> MatchTier:
    compat = _compatibility(release)
    tier = getattr(compat, "match_tier", None) if compat is not None else None
    if tier is None:
        tier = getattr(release, "match_tier", None)
    if tier is None and isinstance(release, dict):
        tier = release.get("match_tier")
    try:
        return MatchTier(int(tier))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return MatchTier.FALLBACK


def _is_hash(release: Any) -> bool:
    compat = _compatibility(release)
    if compat is not None and bool(getattr(compat, "is_hash_match", False)):
        return True
    return bool(
        getattr(release, "is_hash_match", False)
        or getattr(release, "matched_by_hash", False)
        or (release.get("is_hash_match", False) if isinstance(release, dict) else False)
    )


def _accepted(release: Any) -> tuple[bool, str | None]:
    compat = _compatibility(release)
    if compat is not None:
        return bool(getattr(compat, "accepted", True)), getattr(compat, "hard_reject_reason", None)
    return True, None


def _flag(compat: Any, name: str) -> bool | None:
    if compat is None:
        return None
    value = getattr(compat, name, None)
    return value if value is None or isinstance(value, bool) else None


class SyncPredictor:
    """Estimate synchronization likelihood from metadata alone.

    Deterministic: the same inputs always yield the same prediction, and each
    branch is a rule rather than an accumulated score, so adding a signal can
    never quietly re-weight the others.
    """

    def predict(self, release: Any, target_meta: dict[str, Any] | None) -> SyncPrediction:
        """Return a ranking hint for one candidate."""
        accepted, reject_reason = _accepted(release)

        # Hard compatibility is absolute. Prediction never resurrects a
        # candidate the matcher already rejected, however plausible the release
        # relationship looks.
        if not accepted:
            return SyncPrediction(
                availability=VerificationAvailability.UNKNOWN,
                predicted_state=SyncState.REJECTED,
                confidence=None,
                rule="hard_rejected",
                reasons=[
                    f"hard compatibility rejection ({reject_reason or 'rejected'}); "
                    "prediction cannot rescue a hard rejection"
                ],
            )

        # No real video identity means no basis for a synchronization claim.
        # This is the guard that keeps the earlier target_filename contamination
        # from ever becoming a synchronization prediction.
        if not has_video_fingerprint(target_meta or {}):
            return SyncPrediction(
                availability=VerificationAvailability.UNKNOWN,
                predicted_state=SyncState.UNVERIFIED,
                confidence=None,
                rule="no_fingerprint",
                reasons=["no target video fingerprint; synchronization cannot be predicted"],
            )

        compat = _compatibility(release)
        tier = _tier(release)

        # Rule 1: byte-exact identity. Strongest metadata evidence available
        # without a measurement.
        if _is_hash(release) or tier is MatchTier.HASH:
            return SyncPrediction(
                availability=VerificationAvailability.PREDICTED,
                predicted_state=SyncState.PROBABLE_SYNC,
                confidence=CONFIDENCE_HASH,
                rule="hash",
                reasons=["byte-exact movie hash match; same video file"],
            )

        # Rule 2: exact release identity - same group means the same mastering
        # pass, so the cue grid lines up by construction.
        if tier is MatchTier.EXACT or _flag(compat, "release_group_match") is True:
            return SyncPrediction(
                availability=VerificationAvailability.PREDICTED,
                predicted_state=SyncState.PROBABLE_SYNC,
                confidence=CONFIDENCE_EXACT_IDENTITY,
                rule="exact_release",
                reasons=["exact release identity (matching release group)"],
            )

        # Rule 3: same source medium and compatible cut.
        source_match = _flag(compat, "source_match")
        edition_match = _flag(compat, "edition_match")
        if tier is MatchTier.SOURCE_FAMILY and source_match is not False:
            if edition_match is not False:
                return SyncPrediction(
                    availability=VerificationAvailability.PREDICTED,
                    predicted_state=SyncState.PROBABLE_SYNC,
                    confidence=CONFIDENCE_SOURCE_EDITION,
                    rule="source_edition",
                    reasons=["same source medium and compatible edition"],
                )

        # Rule 4: title + episode only. Real evidence that this is the right
        # episode, but nothing about timing, so it predicts nothing.
        if tier is MatchTier.CLOSE:
            return SyncPrediction(
                availability=VerificationAvailability.UNKNOWN,
                predicted_state=SyncState.UNVERIFIED,
                confidence=CONFIDENCE_SOURCE_EDITION,
                rule="content_only",
                reasons=[
                    "content and episode match only; no release evidence about timing"
                ],
            )

        return SyncPrediction(
            availability=VerificationAvailability.UNKNOWN,
            predicted_state=SyncState.UNVERIFIED,
            confidence=None,
            rule="fallback",
            reasons=["insufficient release evidence to predict synchronization"],
        )


def prediction_is_below_floor(prediction: SyncPrediction) -> bool:
    """True when a prediction is too weak to act on even if marked PREDICTED."""
    return (
        prediction.confidence is None
        or prediction.confidence < CONFIDENCE_PREDICTION_FLOOR
    )
