"""Lightweight reference preflight built from existing timing evidence.

This module deliberately does NOT verify, accept, or exclude a reference. It
re-uses the project's existing cue-sanity and Large Offset Evidence Gate on the
*same sanitized text ALASS would receive*, so the orchestrator can answer one
narrow question: when two already-downloaded references belong to the same
release family and independently propose the same large offset, which one has
tighter anchor agreement?

Only the ordering of already-planned attempts may change. The deferred
candidate remains in the plan immediately behind the preferred one, the ALASS /
Two-Point engines still produce the candidate bytes, and the authoritative
verifier still decides whether anything may be served.

No new timing model is introduced: every number here comes from
``classify_large_offset_candidate`` / ``assess_large_offset``, which already
feed the Large Offset path.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.subtitle_matcher import FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS


@dataclass(frozen=True)
class ReferenceTimingEvidence:
    """Bounded, language-agnostic timing description of one reference.

    ``family`` is the normalized release family (source + release group) from
    the existing reference policy. ``large_offset`` says only that the first
    dialogue lies outside the normal window; ``accepted`` is the existing
    evidence gate's answer and is still not a sync claim.
    """

    family: str
    large_offset: bool
    accepted: bool
    offset_ms: float | None = None
    dispersion_ms: float | None = None
    drift_ms_per_minute: float | None = None
    structural_similarity: float | None = None


def measure_reference_timing(
    target_text: str | None,
    reference_text: str | None,
    *,
    family: str,
) -> ReferenceTimingEvidence | None:
    """Measure one reference with the existing gates, or ``None`` on failure.

    The reference is sanitized first because that is the exact representation
    ALASS receives (``SubtitleSyncService.sync`` sanitizes before invoking the
    binary). Measuring the raw provider payload can therefore see an intro cue
    that ALASS never sees -- the precise confusion that kept the proven Into the
    Wild reference out of the large-offset path.
    """
    if not target_text or not reference_text:
        return None
    try:
        # Lazy imports keep the sync package's existing import order intact.
        from app.services.sync.large_offset import (
            assess_large_offset,
            classify_large_offset_candidate,
        )
        from app.services.sync_service import sanitize_subtitle

        sanitized = sanitize_subtitle(reference_text) or reference_text
        seed = classify_large_offset_candidate(
            target_text,
            sanitized,
            threshold_ms=FIRST_DIALOGUE_EXECUTION_THRESHOLD_MS,
        )
        if seed is None:
            return ReferenceTimingEvidence(family=family, large_offset=False, accepted=False)
        assessment = assess_large_offset(
            target_text,
            sanitized,
            seed_offset_ms=seed,
            identity_supported=True,
        )
    except Exception:
        # Preflight is an optimisation. Any failure must leave the original
        # order and every existing safeguard untouched.
        return None
    return ReferenceTimingEvidence(
        family=family,
        large_offset=True,
        accepted=bool(assessment.accepted),
        offset_ms=assessment.estimated_offset_ms,
        dispersion_ms=assessment.offset_dispersion_ms,
        drift_ms_per_minute=assessment.drift_ms_per_minute,
        structural_similarity=assessment.structural_similarity,
    )


def sibling_is_materially_tighter(
    current: ReferenceTimingEvidence | None,
    sibling: ReferenceTimingEvidence | None,
    *,
    agreement_ms: int = 750,
    dispersion_ratio: float = 0.6,
) -> bool:
    """True when ``sibling`` is a same-family, same-offset, tighter reference.

    Deliberately conservative. All of the following must hold:

    * both measurements exist and the evidence gate accepted both;
    * the normalized release family is identical (same source and group, so
      this is not a cross-release preference);
    * the independently estimated offsets agree within ``agreement_ms``;
    * the sibling's anchor dispersion is at most ``dispersion_ratio`` of the
      current candidate's (materially tighter, not merely lower);
    * the sibling's absolute drift is not worse.

    A deferred candidate is never removed from the plan, so a wrong preference
    still falls back to it on the next loop iteration.
    """
    if current is None or sibling is None:
        return False
    if not (current.accepted and sibling.accepted and current.large_offset and sibling.large_offset):
        return False
    if not current.family or current.family != sibling.family:
        return False
    if current.offset_ms is None or sibling.offset_ms is None:
        return False
    if abs(current.offset_ms - sibling.offset_ms) > agreement_ms:
        return False
    if current.dispersion_ms is None or sibling.dispersion_ms is None:
        return False
    if sibling.dispersion_ms > current.dispersion_ms * dispersion_ratio:
        return False
    current_drift = abs(current.drift_ms_per_minute or 0.0)
    sibling_drift = abs(sibling.drift_ms_per_minute or 0.0)
    if sibling_drift > current_drift:
        return False
    return True
