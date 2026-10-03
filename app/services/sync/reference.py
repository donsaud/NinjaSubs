"""Reference trust: is the subtitle we align against actually trustworthy?

Verification is only as trustworthy as the reference it verifies against. A
perfect residual against a bad reference is still a bad verification, because
alass will faithfully align the target onto whatever it is handed.

Current reference selection, as traced before this module existed
------------------------------------------------------------
1. ``ExternalExactStrategy._gather_candidates`` queries SubDL, SubSource and
   OpenSubtitles concurrently, pools the results, and de-duplicates **by
   download URL only** - two providers serving byte-identical content under
   different URLs survive as two candidates.
2. ``_select_candidates_ranked`` filters (target's own URL, target's own
   release name, wrong season, wrong episode) then sorts by a strict tier
   ladder - ``reference_tier``'s HASH / EXACT_GROUP / SOURCE_EDITION /
   FALLBACK, which is a *different* ladder from ``MatchTier`` - followed by
   deterministic tie-breakers (healthy provider, exact episode over pack,
   English anchor, non-HI, scene naming, provider priority, release name).
3. ``resolve_with_provenance`` walks that list and **commits to the first
   candidate that passes the caller's cue-sanity validator**, then breaks.

So yes: the effective policy is "best-looking subtitle that is not obviously
broken becomes the reference". That is a correlated-error path. Every
subsequent measurement - residual, MAD, p95, drift - is computed *relative to
that choice*, so a wrong reference produces a confidently wrong verification.
Two further properties of the current design:

* A reference is never itself verified. Its provenance is never checked
  against anything; it is only used as the thing to align to.
* Reference and candidate may come from the same provider family, and the
  cross-provider redundancy that exists is not accounted for.

What this module adds
---------------------
Measurement, not policy. :func:`assess_reference` grades a reference's health
and trust, :func:`detect_duplicate_groups` collapses byte-identical or
timing-identical candidates from different providers into one evidence group,
and :func:`assess_consensus` reports agreement between the references that were
*already downloaded* during this request - it never triggers an extra download.

Provider count is explicitly not treated as independence, and consensus is
never allowed to override a hard gate.
"""

from __future__ import annotations

import hashlib
import logging
import re
from enum import Enum

from pydantic import BaseModel, Field

from app.services.subtitle_matcher import parse_srt_cues
from app.services.sync.large_offset import LARGE_OFFSET_CUES_PER_REGION
from app.services.sync.structural import StructuralProfile, compare_structures

logger = logging.getLogger(__name__)

# A reference needs real dialogue before it can anchor anything. Below this it
# cannot support a verified claim, however confident the alignment looks.
MIN_REFERENCE_CUES = 12
# Fraction of the target's runtime that must contain reference dialogue.
#
# This is now a PROFILE SELECTOR, not a usability gate. The figure is
# ``len(dialogue) * 2_000 / span``, an estimate built on a 2s mean cue, so it
# under-reports everything that runs long: a 97-minute feature carrying 903
# dialogue cues scores 0.31 and was being discarded as ``reference_invalid``
# while holding far more timing evidence than a dense 22-minute episode. It is
# still measured and still reported, and it still separates dense content from
# sparse content -- it just no longer decides usability on its own.
MIN_DIALOGUE_COVERAGE = 0.55
# How many equal portions of its own dialogue span a reference must populate.
# Reuses the region count the large-offset gate already corroborates across, so
# "spread across the runtime" means the same thing in both places.
DIALOGUE_PROBE_REGIONS = 5
# ...and how many of those portions must actually carry dialogue. Requiring a
# majority rejects a reference whose cues are numerous but bunched into one
# stretch of the film, which anchors nothing outside that stretch.
MIN_DIALOGUE_REGIONS = 3
# A LOW_DIALOGUE reference has to make up for the coverage it lacks with raw
# evidence, so it needs enough cues to corroborate independently rather than
# merely to exist: the same per-region bar the large-offset gate already
# samples across, over the regions it requires. Twelve cues is the floor for
# *any* claim, not a substitute for coverage -- a 12-cue file spread thin is an
# anecdote, not a sample, while a feature-length one clears this comfortably.
LOW_COVERAGE_MIN_DIALOGUE_CUES = (
    LARGE_OFFSET_CUES_PER_REGION * MIN_DIALOGUE_REGIONS
)
# A cue longer than this is usually a stuck cue, not dialogue.
MAX_CUE_DURATION_MS = 15_000
# Fraction of cues allowed to be non-monotonic before the file is treated as
# structurally broken.
MAX_BACKWARD_FRACTION = 0.02
# Duplicates collapsing into one group changes nothing on its own; it only
# matters when the groups disagree.
MIN_CONSENSUS_CLUSTER = 2


class ReferenceTrust(str, Enum):
    """How much a reference can be trusted as the thing to align against."""

    # Previously verified for this exact video + subtitle + engine.
    VERIFIED = "verified"
    # Byte-exact hash match, or an exact release identity.
    STRONG = "strong"
    # Usable: right content, compatible cut, healthy structure.
    ACCEPTABLE = "acceptable"
    # Not enough evidence to trust, though not contradicted.
    UNKNOWN = "unknown"
    # Contradicted: wrong content, broken, or structurally incompatible.
    REJECTED = "rejected"

    @property
    def rank(self) -> int:
        """Strength of evidence; lower is stronger."""
        return _TRUST_RANK[self]


_TRUST_RANK: dict[ReferenceTrust, int] = {
    ReferenceTrust.VERIFIED: 0,
    ReferenceTrust.STRONG: 1,
    ReferenceTrust.ACCEPTABLE: 2,
    ReferenceTrust.UNKNOWN: 3,
    ReferenceTrust.REJECTED: 4,
}


class ReferenceFailure(str, Enum):
    """Why a reference was not usable. Recorded, never silently swallowed."""

    UNAVAILABLE = "reference_unavailable"
    INSUFFICIENT = "reference_insufficient"
    INVALID = "reference_invalid"
    CONFLICT = "reference_conflict"
    LOW_TRUST = "reference_low_trust"
    HARD_REJECTED = "reference_hard_rejected"


class DialogueProfile(str, Enum):
    """How much of a reference's own runtime carries observable dialogue.

    ``STANDARD_DIALOGUE`` is the ordinary case. ``LOW_DIALOGUE`` marks a
    reference that is genuinely sparse -- long feature-length films carry real
    music and silence between lines -- and is deliberately *not* a verdict:
    the same reference can still hold plenty of usable timing evidence, and the
    decision on that is made by evidence count and spread, not by this label.
    """

    STANDARD_DIALOGUE = "standard_dialogue"
    LOW_DIALOGUE = "low_dialogue"


class ReferenceHealth(BaseModel):
    """Lightweight health signals, reusing the existing cue parser."""

    cue_count: int = 0
    dialogue_cues: int = 0
    first_dialogue_ms: int | None = None
    last_dialogue_ms: int | None = None
    median_cue_duration_ms: float | None = None
    dialogue_coverage: float | None = None
    # Equal portions of the reference's own dialogue span that contain dialogue.
    dialogue_regions: int = 0
    profile: DialogueProfile = DialogueProfile.STANDARD_DIALOGUE
    invalid_timing_rate: float = 0.0
    duplicate_cue_rate: float = 0.0
    backward_fraction: float = 0.0
    long_cue_fraction: float = 0.0
    healthy: bool = False
    reasons: list[str] = Field(default_factory=list)


class ReferenceAssessment(BaseModel):
    """Explainable verdict for one reference. The evidence is kept, not summed."""

    trust: ReferenceTrust = ReferenceTrust.UNKNOWN
    health: ReferenceHealth = Field(default_factory=ReferenceHealth)
    provider: str | None = None
    decision_kind: str | None = None
    from_cache: bool = False
    cache_verified: bool = False
    independent_sources: int = 0
    duplicate_sources: int = 0
    consensus_score: float | None = None
    consensus_group: int | None = None
    consensus_agrees: bool | None = None
    failure: ReferenceFailure | None = None
    reasons: list[str] = Field(default_factory=list)

    def explain(self) -> str:
        head = f"ref_trust={self.trust.value}"
        if self.provider:
            head += f" provider={self.provider}"
        head += f" indep={self.independent_sources} dup={self.duplicate_sources}"
        if self.consensus_score is not None:
            head += f" consensus={self.consensus_score:.2f}"
        if self.failure is not None:
            head += f" failure={self.failure.value}"
        return head + (f" | {'; '.join(self.reasons)}" if self.reasons else "")

    def supports_verified_claim(self) -> bool:
        """False whenever the reference itself is unproven or broken.

        This is the one place reference trust is allowed to *withhold* a claim.
        It can never create one, so it cannot make the system more aggressive.
        """
        return self.trust in (ReferenceTrust.VERIFIED, ReferenceTrust.STRONG)


_NON_SPEECH = re.compile(
    r"(?i)(https?://|www\.|@[\w.]+|♪|\[music\]|\(music\)|subtitle|subtitles|"
    r"translated|translation|ترجمة|مترجم|sync(?:ed)?\s+by|timed\s+by)"
)


def _is_dialogue(text: str) -> bool:
    """Reuse the matcher's own non-speech rule where it is cheap to do so."""
    cleaned = re.sub(r"\[[^\]]*\]|\([^()]*\)|\{[^{}]*\}", " ", text or "")
    if _NON_SPEECH.search(cleaned):
        return False
    return sum(1 for ch in cleaned if ch.isalpha()) >= 3


def is_dialogue_cue(text: str) -> bool:
    """Public form of the non-speech rule.

    Credits, translator intros, site tags, music cues and SDH brackets are not
    speech and must not be treated as timing evidence. Exposed so the alignment
    gate can filter on exactly the same definition the reference health check
    uses, instead of keeping a second copy of the pattern.
    """
    return _is_dialogue(text)


def analyze_reference_health(
    reference: str | None, *, require_dialogue_coverage: bool = True
) -> ReferenceHealth:
    """Measure a reference's structural health from its cues.

    Uses :func:`parse_srt_cues` rather than adding a parser. A reference that
    cannot be parsed is reported as unhealthy rather than silently accepted.

    ``require_dialogue_coverage`` controls only the runtime-coverage floor. It
    exists because that floor answers a *selection* question -- "is this dense
    enough to be worth trying" -- rather than a structural one, and the estimate
    (:data:`MIN_DIALOGUE_COVERAGE` applied to ``len(dialogue) * 2_000 / span``)
    is calibrated against a 2s mean cue. A full-length episode subtitle whose
    cues average well under 2s cannot clear it however correct its structure,
    so callers judging structural usability pass ``False`` and read
    :attr:`ReferenceHealth.dialogue_coverage` for themselves. Default ``True``
    keeps every existing caller's behaviour unchanged.
    """
    health = ReferenceHealth()
    if not reference:
        health.reasons.append("reference produced no parseable cues")
        return health

    cues = parse_srt_cues(reference)
    health.cue_count = len(cues)
    if not cues:
        health.reasons.append("reference produced no parseable cues")
        return health

    invalid = sum(1 for start, end, _ in cues if end <= start or start < 0)
    health.invalid_timing_rate = invalid / len(cues)

    texts = [text.strip() for _, _, text in cues]
    health.duplicate_cue_rate = (
        (len(texts) - len(set(texts))) / len(texts) if texts else 0.0
    )

    backward = 0
    running_end = 0
    for start, end, _ in cues:
        if running_end and start < running_end:
            backward += 1
        running_end = max(running_end, end)
    health.backward_fraction = backward / len(cues)

    long_cues = sum(1 for start, end, _ in cues if end - start > MAX_CUE_DURATION_MS)
    health.long_cue_fraction = long_cues / len(cues)

    dialogue = [
        (start, end) for start, end, text in cues if _is_dialogue(text)
    ]
    health.dialogue_cues = len(dialogue)
    if dialogue:
        health.first_dialogue_ms = dialogue[0][0]
        health.last_dialogue_ms = dialogue[-1][1]
        span = max(1, dialogue[-1][1] - dialogue[0][0])
        health.dialogue_coverage = len(dialogue) * 2_000 / span
        # How far the dialogue actually reaches. Cue count alone cannot tell a
        # well-distributed reference from one whose lines are all bunched into
        # the first act, and only the first can corroborate an alignment
        # anywhere else in the film.
        populated: set[int] = set()
        for start, _end in dialogue:
            offset = int((start - dialogue[0][0]) / span * DIALOGUE_PROBE_REGIONS)
            populated.add(min(DIALOGUE_PROBE_REGIONS - 1, max(0, offset)))
        health.dialogue_regions = len(populated)
        durations = sorted(end - start for start, end in dialogue)
        mid = len(durations) // 2
        health.median_cue_duration_ms = (
            durations[mid] if len(durations) % 2 else (durations[mid - 1] + durations[mid]) / 2
        )

    reasons = health.reasons
    problems: list[str] = []

    if health.cue_count < MIN_REFERENCE_CUES:
        problems.append(f"only {health.cue_count} cues (need {MIN_REFERENCE_CUES})")
    # A reference with no usable dialogue cannot anchor a speech alignment, no
    # matter how many cues it has. Credits-only and music-only files land here.
    if health.dialogue_cues < MIN_REFERENCE_CUES:
        problems.append(
            f"only {health.dialogue_cues} dialogue cue(s) of {health.cue_count}; "
            "not enough speech to anchor an alignment"
        )
    if health.invalid_timing_rate > 0:
        problems.append(f"{invalid} cues have non-positive durations")
    if health.backward_fraction > MAX_BACKWARD_FRACTION:
        problems.append(f"{health.backward_fraction:.1%} of cues are non-monotonic")
    if health.long_cue_fraction > 0.10:
        problems.append(f"{health.long_cue_fraction:.1%} of cues are implausibly long")
    # Coverage is reported and used to label the profile, never to reject on its
    # own: the estimate assumes a 2s mean cue, so a long feature with plenty of
    # dialogue scores low purely for running long. Usability is decided by the
    # absolute cue floor above plus how far the dialogue reaches, which is what
    # "enough observable evidence" actually means.
    if (
        health.dialogue_coverage is not None
        and health.dialogue_coverage < MIN_DIALOGUE_COVERAGE
    ):
        health.profile = DialogueProfile.LOW_DIALOGUE
        health.reasons.append(
            f"dialogue covers {health.dialogue_coverage:.0%} of the runtime "
            f"across {health.dialogue_regions} of {DIALOGUE_PROBE_REGIONS} regions "
            f"({health.dialogue_cues} dialogue cues)"
        )
    if health.dialogue_regions < MIN_DIALOGUE_REGIONS:
        problems.append(
            f"dialogue occupies only {health.dialogue_regions} of "
            f"{DIALOGUE_PROBE_REGIONS} timeline regions; there is nothing to "
            "anchor against away from that stretch"
        )
    if (
        health.profile is DialogueProfile.LOW_DIALOGUE
        and health.dialogue_cues < LOW_COVERAGE_MIN_DIALOGUE_CUES
    ):
        problems.append(
            f"dialogue covers {health.dialogue_coverage:.0%} of the runtime with "
            f"only {health.dialogue_cues} cues (need "
            f"{LOW_COVERAGE_MIN_DIALOGUE_CUES} to make up for it); too few "
            "anchors to corroborate an alignment"
        )

    # Repeated dialogue is normal - refrains, recurring phrases, a name spoken
    # often - so a high duplicate rate is recorded as a quality note rather than
    # treated as disqualifying.
    if health.duplicate_cue_rate > 0.5:
        reasons.append(f"{health.duplicate_cue_rate:.0%} of cues repeat text")

    # Computed from explicit conditions, not by matching reason text.
    health.healthy = not problems
    reasons.extend(problems)
    return health


def reference_fingerprint(text: str | None) -> str | None:
    """Content identity for a reference, for duplicate detection.

    Uses the timing grid rather than the words: two providers serving the same
    subtitle usually differ in dialogue text (credits, translation style) but
    not in when the cues fire. Quantized to 100ms so trivial re-encodings
    collapse together.
    """
    if not text:
        return None
    cues = parse_srt_cues(text)
    if len(cues) < MIN_REFERENCE_CUES:
        return None
    grid = "|".join(f"{start // 100}:{max(0, end - start) // 100}" for start, end, _ in cues)
    return hashlib.sha256(grid.encode("utf-8")).hexdigest()[:16]


def detect_duplicate_groups(
    references: list[tuple[str, str | None, str | None]],
) -> dict[str, list[tuple[str, str | None, str | None]]]:
    """Group ``(provider, name, text)`` references by content identity.

    Provider count is not independence: three providers serving the same file
    are one piece of evidence. Only text that was already downloaded is
    fingerprinted, so this costs no extra fetch.
    """
    groups: dict[str, list[tuple[str, str | None, str | None]]] = {}
    for provider, name, text in references:
        fingerprint = reference_fingerprint(text)
        # An unfingerprinted reference gets its own group; absence of evidence is
        # never read as evidence of duplication.
        key = fingerprint or f"unfingerprinted:{provider}:{name}"
        groups.setdefault(key, []).append((provider, name, text))
    return groups


def assess_consensus(
    groups: dict[str, list[tuple[str, str | None, str | None]]],
    *,
    target_profile: StructuralProfile | None = None,
) -> tuple[float | None, int | None, bool | None, list[str]]:
    """Agreement between independent reference groups.

    A cluster is the largest set of groups whose timing grids agree. Agreement
    is reported as evidence only: two groups agreeing does not make either
    correct, and it is never allowed to promote a rejected reference.
    """
    reasons: list[str] = []
    fingerprinted = {
        key: members for key, members in groups.items() if not key.startswith("unfingerprinted:")
    }
    if len(fingerprinted) < MIN_CONSENSUS_CLUSTER:
        reasons.append(
            f"only {len(fingerprinted)} independent reference(s); no consensus available"
        )
        return None, None, None, reasons

    profiles = {
        key: StructuralProfile.from_subtitle(members[0][2]) for key, members in fingerprinted.items()
    }
    best_cluster = [next(iter(profiles))]
    for key in profiles:
        if key == best_cluster[0]:
            continue
        similarity = compare_structures(
            [c for c in _cues_of(fingerprinted[key])], None
        )
        structural_agreement = _structural_agreement(profiles[best_cluster[0]], profiles[key])
        if structural_agreement >= 0.75:
            best_cluster.append(key)
        del similarity  # structural comparison is used via the profile ratio
    if target_profile is not None:
        target_agreement = [
            _structural_agreement(target_profile, profiles[key]) for key in best_cluster
        ]
        mean_agreement = sum(target_agreement) / len(target_agreement)
        if mean_agreement < 0.5:
            reasons.append(
                "reference cluster disagrees with the target's own structure; "
                "consensus does not make it correct"
            )
            return None, len(best_cluster), False, reasons
    score = len(best_cluster) / len(fingerprinted)
    reasons.append(
        f"{len(best_cluster)} of {len(fingerprinted)} independent reference group(s) agree"
    )
    return round(score, 3), len(best_cluster), True, reasons


def _cues_of(members: list[tuple[str, str | None, str | None]]):
    text = members[0][2] if members else None
    return parse_srt_cues(text) if text else []


def _structural_agreement(left: StructuralProfile, right: StructuralProfile) -> float:
    """Shift-invariant structural agreement in 0..1."""
    if not left.cue_count or not right.cue_count:
        return 0.0
    count_ratio = min(left.cue_count, right.cue_count) / max(left.cue_count, right.cue_count)
    if left.median_duration_ms and right.median_duration_ms:
        duration_ratio = min(left.median_duration_ms, right.median_duration_ms) / max(
            left.median_duration_ms, right.median_duration_ms
        )
    else:
        duration_ratio = 0.5
    if left.cluster_count and right.cluster_count:
        cluster_ratio = min(left.cluster_count, right.cluster_count) / max(
            left.cluster_count, right.cluster_count
        )
    else:
        cluster_ratio = 0.5
    return round((count_ratio + duration_ratio + cluster_ratio) / 3, 3)


def assess_reference(
    reference: str | None,
    *,
    target_cues: list | None = None,
    provider: str | None = None,
    decision_kind: str | None = None,
    from_cache: bool = False,
    cache_verified: bool = False,
    hard_rejected: bool = False,
    groups: dict[str, list[tuple[str, str | None, str | None]]] | None = None,
) -> ReferenceAssessment:
    """Grade one reference. Conservative: anything unproven lands on UNKNOWN."""
    assessment = ReferenceAssessment(
        provider=provider,
        decision_kind=decision_kind,
        from_cache=from_cache,
        cache_verified=cache_verified,
    )
    if hard_rejected:
        assessment.trust = ReferenceTrust.REJECTED
        assessment.failure = ReferenceFailure.HARD_REJECTED
        assessment.reasons.append("reference failed a hard gate; trust cannot be restored")
        return assessment

    health = analyze_reference_health(reference)
    assessment.health = health

    target_profile = StructuralProfile.from_cues(target_cues) if target_cues else None
    if target_cues:
        similarity = compare_structures(target_cues, reference)
        if similarity.score is not None and similarity.score < 0.45:
            assessment.trust = ReferenceTrust.REJECTED
            assessment.failure = ReferenceFailure.CONFLICT
            assessment.reasons.append(
                f"reference structure does not correspond to the target ({similarity.explain()})"
            )
            return assessment

    if not health.healthy:
        assessment.trust = ReferenceTrust.REJECTED if health.cue_count < MIN_REFERENCE_CUES else ReferenceTrust.UNKNOWN
        assessment.failure = (
            ReferenceFailure.INSUFFICIENT
            if health.cue_count < MIN_REFERENCE_CUES
            else ReferenceFailure.INVALID
        )
        assessment.reasons.extend(health.reasons)
        return assessment

    # Trust ladder: exact measured proof first, then deterministic identity.
    if cache_verified:
        assessment.trust = ReferenceTrust.VERIFIED
        assessment.reasons.append("exact verified cache hit for this video + subtitle + engine")
    elif decision_kind == "hash":
        assessment.trust = ReferenceTrust.STRONG
        assessment.reasons.append("byte-exact movie hash match")
    elif decision_kind == "team":
        assessment.trust = ReferenceTrust.STRONG
        assessment.reasons.append("exact release-group (team) match")
    else:
        assessment.trust = ReferenceTrust.ACCEPTABLE
        assessment.reasons.append("healthy reference; exact cut not proven")
    if health.reasons:
        assessment.reasons.extend(health.reasons)

    if groups:
        total_providers = sum(len(members) for members in groups.values())
        independent = len(groups)
        duplicates = total_providers - independent
        assessment.independent_sources = independent
        assessment.duplicate_sources = duplicates
        if duplicates:
            assessment.reasons.append(
                f"{duplicates} provider copy/copies collapsed; provider count is not independence"
            )
        score, size, agrees, consensus_reasons = assess_consensus(
            groups, target_profile=target_profile
        )
        assessment.consensus_score = score
        assessment.consensus_group = size
        assessment.consensus_agrees = agrees
        assessment.reasons.extend(consensus_reasons)
    return assessment
