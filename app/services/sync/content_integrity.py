"""Content integrity: a shadow assessment that is explicitly not about timing.

DIAGNOSTIC ONLY. Imported by no production module. Writes nothing. Never stores
or logs subtitle text -- when text is consulted it is hashed and discarded
immediately, and only counts survive.

The point of this module is that the system already knows timing and content are
different questions, and that a subtitle can be perfectly timed and still be the
wrong content. Nothing here is allowed to imply timing quality, and nothing here
is an acceptance threshold: it reports what was observed and whether the evidence
was sufficient to say anything at all.

Where the evidence runs out it says UNKNOWN rather than guessing. That is the
common case for a genuine cross-language reference, where measured line overlap
was 3.7% (ASAP) and 6.2% (EVOLV) against the local reference.
"""

from __future__ import annotations

import hashlib
import re
import statistics
from dataclasses import dataclass, field
from enum import Enum

_NON_WORD = re.compile(r"[^\w]+")

#: How far a volume ratio may sit from 1.0 before it is called a difference.
#: Diagnostic only. Deliberately NOT an acceptance threshold and not derived from
#: any verifier limit: it exists so "comparable" and "different volume" are
#: distinguishable at all.
VOLUME_TOLERANCE = 0.15


class ContentIntegrity(str, Enum):
    GOOD = "CONTENT_INTEGRITY_GOOD"
    SUSPECT = "CONTENT_INTEGRITY_SUSPECT"
    #: Not enough evidence to say. Not a pass.
    UNKNOWN = "CONTENT_INTEGRITY_UNKNOWN"


class Evidence(str, Enum):
    AVAILABLE = "AVAILABLE"
    USEFUL = "USEFUL"
    AMBIGUOUS = "AMBIGUOUS"
    INSUFFICIENT = "INSUFFICIENT"
    NOT_AVAILABLE = "NOT_AVAILABLE"


@dataclass
class IntegrityFinding:
    signal: str
    value: float | None
    evidence: Evidence
    note: str = ""


@dataclass
class ContentIntegrityReport:
    verdict: ContentIntegrity
    #: Signal name -> measured value. Reported whether or not the verdict is GOOD.
    observed: dict[str, float] = field(default_factory=dict)
    findings: list[IntegrityFinding] = field(default_factory=list)
    #: Why the verdict is not better than it is.
    limits: list[str] = field(default_factory=list)
    #: Explicitly recorded so no reader mistakes this for a timing result.
    timing_implied: bool = False

    def as_row(self) -> dict[str, object]:
        return {
            "verdict": self.verdict.value,
            "observed": dict(self.observed),
            "limits": list(self.limits),
            "timing_implied": self.timing_implied,
        }


def _norm_hash(text: str) -> str:
    cleaned = _NON_WORD.sub(" ", text.lower()).strip()
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:12] if cleaned else ""


def text_evidence(
    target: list[tuple[int, int, str]],
    reference: list[tuple[int, int, str]],
) -> tuple[float, Evidence, str]:
    """Fraction of target lines present in the reference. Hashes only.

    For a cross-language reference this collapses to near zero, which is the
    signal that text cannot arbitrate here -- not that the content is wrong.
    """
    ref_counts: dict[str, int] = {}
    for _, _, t in reference:
        h = _norm_hash(t)
        if h:
            ref_counts[h] = ref_counts.get(h, 0) + 1
    tgt = [_norm_hash(t) for _, _, t in target]
    tgt = [h for h in tgt if h]
    if not tgt or not ref_counts:
        return 0.0, Evidence.NOT_AVAILABLE, "no usable text on one side"
    matched = sum(min(1, ref_counts.get(h, 0)) for h in tgt)
    overlap = matched / len(tgt)
    if overlap >= 0.5:
        return overlap, Evidence.USEFUL, "same-language evidence available"
    return overlap, Evidence.NOT_AVAILABLE, (
        f"only {overlap:.1%} of target lines occur in the reference; the reference "
        "is a different release and/or language, so text cannot arbitrate"
    )


def assess_content_integrity(
    target: list[tuple[int, int, str]],
    reference: list[tuple[int, int, str]],
    *,
    reference_coverage: float | None = None,
) -> ContentIntegrityReport:
    """Observed content volume, kept strictly separate from timing quality."""
    report = ContentIntegrityReport(verdict=ContentIntegrity.UNKNOWN)

    if not target or not reference:
        report.findings.append(
            IntegrityFinding("input", None, Evidence.NOT_AVAILABLE, "empty input")
        )
        report.limits.append("no cues on one side")
        return report

    t_active = sum(max(0, e - s) for s, e, _ in target)
    r_active = sum(max(0, e - s) for s, e, _ in reference)
    cue_ratio = len(target) / len(reference)
    active_ratio = t_active / r_active if r_active else 0.0

    overlap, text_ev, text_note = text_evidence(target, reference)

    observed = {
        "cue_count_ratio": round(cue_ratio, 4),
        "active_duration_ratio": round(active_ratio, 4),
        "text_line_overlap": round(overlap, 4),
    }
    if reference_coverage is not None:
        observed["reference_coverage"] = round(reference_coverage, 4)
    report.observed = observed

    report.findings.extend([
        IntegrityFinding("cue_count_ratio", cue_ratio, Evidence.AVAILABLE),
        IntegrityFinding("active_duration_ratio", active_ratio, Evidence.AVAILABLE),
        IntegrityFinding("text_line_overlap", overlap, text_ev, text_note),
        IntegrityFinding(
            "reference_coverage",
            reference_coverage,
            Evidence.AVAILABLE if reference_coverage is not None else Evidence.NOT_AVAILABLE,
            "from the timing alignment; reported here, never used to judge timing",
        ),
    ])

    # --- volume reading ----------------------------------------------------- #
    # Both ratios near 1 is the comparable case. Both far from 1 in the same
    # direction is a content volume difference. A mixed signal means a
    # segmentation difference, which is not a content problem.
    cue_off = abs(cue_ratio - 1.0)
    act_off = abs(active_ratio - 1.0)

    if cue_off <= VOLUME_TOLERANCE and act_off <= VOLUME_TOLERANCE:
        report.verdict = ContentIntegrity.GOOD
    elif cue_off > VOLUME_TOLERANCE and act_off > VOLUME_TOLERANCE:
        if (cue_ratio > 1.0) == (active_ratio > 1.0):
            report.verdict = ContentIntegrity.SUSPECT
            report.limits.append(
                f"both cue count ({cue_ratio:.2f}x) and speech time "
                f"({active_ratio:.2f}x) differ from the reference in the same "
                "direction: more or less content than the reference carries"
            )
        else:
            # One up, one down: the segmentation changed, not the content volume.
            report.verdict = ContentIntegrity.UNKNOWN
            report.limits.append(
                f"cue count {cue_ratio:.2f}x but speech time {active_ratio:.2f}x "
                "move in opposite directions; this is a segmentation difference, "
                "which timing evidence cannot adjudicate"
            )
    else:
        report.verdict = ContentIntegrity.UNKNOWN
        report.limits.append(
            f"cue count {cue_ratio:.2f}x, speech time {active_ratio:.2f}x: one is "
            "within tolerance and one is not, so the readings disagree"
        )

    # A volume difference is not attributable. Duplication and a genuinely
    # different release look identical here, and without usable text there is
    # nothing that separates them.
    if text_ev is not Evidence.USEFUL and report.verdict is ContentIntegrity.GOOD:
        # GOOD requires *positive* evidence that the content corresponds. When the
        # reference is a different language there is none -- matching cue and
        # speech-time volumes prove nothing about which lines these are. So GOOD
        # is unavailable and the honest answer is UNKNOWN. This is the common case
        # for a real cross-language reference, and it is why a legitimately
        # synchronized subtitle from another release cannot be content-confirmed.
        report.verdict = ContentIntegrity.UNKNOWN
        report.limits.append(
            "no same-language evidence: matching volumes cannot confirm that these "
            "are the same lines, so GOOD is unavailable"
        )

    if report.verdict is ContentIntegrity.SUSPECT and text_ev is not Evidence.USEFUL:
        report.limits.append(
            "duplication and a different release are indistinguishable without "
            "text evidence, so the excess cannot be attributed"
        )

    return report


def sequence_disagreement(
    target: list[tuple[int, int, str]],
    reference: list[tuple[int, int, str]],
) -> float | None:
    """Fraction of positions where a text-free structural signature disagrees.

    Each cue becomes ``duration_bucket:gap_bucket``. No text survives. Returns
    ``None`` when the two files have different lengths, because a positional
    comparison is meaningless then.
    """
    if len(target) != len(reference):
        return None

    def sig(cues):
        out = []
        prev_end = None
        for s, e, _ in cues:
            gap = max(0, s - prev_end) if prev_end is not None else 0
            out.append(f"{max(0, e - s) // 250}:{gap // 500}")
            prev_end = e
        return out

    a, b = sig(target), sig(reference)
    if not a:
        return None
    agree = sum(1 for x, y in zip(a, b, strict=False) if x == y)
    return 1.0 - agree / len(a)


def median_of(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0
