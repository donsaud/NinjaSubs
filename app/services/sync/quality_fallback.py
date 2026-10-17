"""Bounded quality gate for guarded Large-Offset delivery.

The Large-Offset investigation deliberately exempts residual ``p95`` for a
constant shift, but its delivery decision did not distinguish a reference bound
strongly to the target video (the confirmed Into the Wild shape) from a
source-family-only guess (the Mad Men DBT/SA89 failures). Both could be
``UNVERIFIED/UNKNOWN`` and both were deliverable.

This module answers one narrow question: should a guarded corrected result be
served immediately, or held provisionally while the existing multi-candidate
loop tries a better, independent reference?

It never verifies, promotes, or transforms anything. It only decides whether
the existing verifier's refusal is enough to keep looking. Strong/exact binding
keeps the existing Large-Offset behavior; weak/unknown/conflict binding does
not get an immediate override.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class QualityFallbackAssessment:
    """Outcome of the quality gate for one corrected candidate."""

    defer: bool
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        binding = self.evidence.get("binding") or "?"
        p95 = self.evidence.get("p95_ms")
        mad = self.evidence.get("mad_ms")
        structural = self.evidence.get("structural_similarity")
        verifier = self.evidence.get("verification") or "?"
        rejection = self.evidence.get("rejection_reason") or "none"
        return (
            f"binding={binding} p95={p95} mad={mad} struct={structural} "
            f"verification={verifier} rejection={rejection}"
        )


def assess_quality_fallback(
    *,
    serving_state: str,
    binding_state: str,
    would_serve_without_override: bool,
    sync_state: str,
    verification: str,
    rejection_reason: str | None,
    p95_ms: float | None,
    mad_ms: float | None,
    structural_similarity: float | None,
) -> QualityFallbackAssessment:
    """Decide whether a guarded correction may be served immediately.

    Rules, in order:

    1. Only the guarded Large-Offset state is in scope. Everything else keeps
       its existing serving decision.
    2. A result the normal serving policy already accepts is not deferred.
       This preserves normally verified weak-binding results that earned
       independent verification.
    3. A strong/exact binding guarded result is not deferred. This is the
       confirmed Into the Wild/Dexter shape.
    4. Any other binding (weak/unknown/conflict) relying on the Large-Offset
       override is provisional. The caller should try the next independent
       reference; if none is better, serve the original rather than this
       correction.
    """
    evidence = {
        "binding": binding_state,
        "sync_state": sync_state,
        "verification": verification,
        "rejection_reason": rejection_reason,
        "p95_ms": p95_ms,
        "mad_ms": mad_ms,
        "structural_similarity": structural_similarity,
    }
    if serving_state != "alass_corrected_large_offset":
        return QualityFallbackAssessment(False, "not_guarded_large_offset", evidence)
    if would_serve_without_override:
        return QualityFallbackAssessment(False, "normal_verification_accepted", evidence)
    if binding_state in {"exact", "strong"}:
        return QualityFallbackAssessment(False, "strong_binding_guarded_result", evidence)
    return QualityFallbackAssessment(
        True,
        "weak_binding_guarded_result_requires_better_evidence",
        evidence,
    )
