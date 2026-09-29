"""Structural comparison of two subtitles, and classification of the difference.

A timing offset alone cannot distinguish two very different situations:

* the *same cut* re-timed by a constant (a WEB intro versus a BluRay intro);
* a *different cut*, where a scene was inserted or removed and every later cue
  sits at a different position.

Both can present a large "offset". What separates them is the shape of the
timeline: a constant shift moves the whole curve rigidly, while an edited
release leaves a discontinuity partway through and a different structure
afterwards.

This module measures that shape without audio. It deliberately compares
*distributions* rather than absolute cue positions, because providers legitimately
split or merge cues differently, so an exact cue count is never required.
Every result is supporting evidence: low similarity lowers confidence and
narrows the possible cut verdicts, it does not by itself reject a candidate.
"""

from __future__ import annotations

import logging
import math
from enum import Enum

from pydantic import BaseModel, Field

from app.services.subtitle_matcher import parse_srt_cues

logger = logging.getLogger(__name__)

Cue = tuple[int, int, str]

# A gap longer than this separates two dialogue clusters (a scene break).
CLUSTER_GAP_MS = 20_000
# Relative tolerance when comparing distributions; subtitle splits/merges mean
# counts are never expected to match exactly.
SIMILARITY_TOLERANCE = 0.35
# Below this structural similarity the two timelines are unlikely to be the
# same cut, whatever the timing says.
MIN_STRUCTURAL_SIMILARITY = 0.45


class CutVerdict(str, Enum):
    """How the candidate's timeline relates to the reference's."""

    # Same cut, same timings.
    SAME_CUT = "same_cut"
    # Same structure shifted rigidly: re-timing should fix it.
    STABLE_OFFSET = "stable_offset"
    # Offsets that grow over time (PAL speed-up, progressive drift).
    DRIFT = "drift"
    # A discontinuity partway through: recap, inserted/edited scene.
    PIECEWISE = "piecewise"
    # Structurally different, or offset beyond anything re-timing explains.
    DIFFERENT_CUT = "different_cut"
    # Not enough cues to tell.
    UNKNOWN = "unknown"


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("median of empty sequence")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _ratio_similarity(a: float, b: float) -> float:
    """Similarity of two positive magnitudes in 0..1, tolerant of splitting."""
    if a <= 0 and b <= 0:
        return 1.0
    larger, smaller = max(a, b), min(a, b)
    if larger <= 0:
        return 0.0
    return smaller / larger


def _relative_close(a: float, b: float, tolerance: float = SIMILARITY_TOLERANCE) -> bool:
    larger, smaller = max(a, b), min(a, b)
    if larger <= 0:
        return True
    return (larger - smaller) / larger <= tolerance


class StructuralProfile(BaseModel):
    """Shift-invariant description of one subtitle's timing shape."""

    cue_count: int = 0
    median_duration_ms: float | None = None
    density_cues_per_minute: float | None = None
    span_ms: int = 0
    first_dialogue_ms: int | None = None
    last_dialogue_ms: int | None = None
    # Distribution of inter-cue gaps, as quantiles of the *gaps* only.
    gap_p50_ms: float | None = None
    gap_p90_ms: float | None = None
    long_gap_count: int = 0
    # Contiguous dialogue runs separated by a long gap.
    cluster_count: int = 0
    cluster_cue_counts: list[int] = Field(default_factory=list)

    @classmethod
    def from_cues(cls, cues: list[Cue]) -> StructuralProfile:
        if not cues:
            return cls()
        ordered = sorted(cues, key=lambda cue: cue[0])
        starts = [start for start, _, _ in ordered]
        ends = [end for _, end, _ in ordered]
        durations = [float(end - start) for start, end, _ in ordered if end > start]
        gaps = [float(starts[i] - ends[i - 1]) for i in range(1, len(ordered))]
        span = max(1, ends[-1] - starts[0])

        clusters: list[list[Cue]] = [[ordered[0]]]
        for previous, cue in zip(ordered, ordered[1:], strict=False):
            if cue[0] - previous[1] > CLUSTER_GAP_MS:
                clusters.append([cue])
            else:
                clusters[-1].append(cue)

        long_gaps = sorted(g for g in gaps if g > 0)
        return cls(
            cue_count=len(ordered),
            median_duration_ms=round(_median(durations), 1) if durations else None,
            density_cues_per_minute=round(len(ordered) * 60_000 / span, 2),
            span_ms=span,
            first_dialogue_ms=starts[0],
            last_dialogue_ms=ends[-1],
            gap_p50_ms=round(_median(long_gaps), 1) if long_gaps else None,
            gap_p90_ms=round(
                long_gaps[min(len(long_gaps) - 1, math.ceil(0.9 * len(long_gaps)) - 1)], 1
            )
            if long_gaps
            else None,
            long_gap_count=sum(1 for gap in gaps if gap > CLUSTER_GAP_MS),
            cluster_count=len(clusters),
            cluster_cue_counts=[len(cluster) for cluster in clusters],
        )

    @classmethod
    def from_subtitle(cls, value: str | list[Cue] | None) -> StructuralProfile:
        if value is None:
            return cls()
        cues = parse_srt_cues(value) if isinstance(value, str) else sorted(value, key=lambda c: c[0])
        return cls.from_cues(cues)


class StructuralSimilarity(BaseModel):
    """How alike two subtitles' timing structures are, in 0..1."""

    score: float | None = None
    cue_count_ratio: float | None = None
    duration_ratio: float | None = None
    density_ratio: float | None = None
    cluster_count_delta: int | None = None
    long_gap_count_delta: int | None = None
    # Cluster size sequences are compared positionally; differing splits
    # (one cue vs two) are tolerated by a per-element ratio.
    cluster_profile_ratio: float | None = None
    reasons: list[str] = Field(default_factory=list)

    @property
    def same_structure(self) -> bool:
        return self.score is not None and self.score >= MIN_STRUCTURAL_SIMILARITY

    def explain(self) -> str:
        if self.score is None:
            return "structural similarity unavailable"
        return (
            f"struct={self.score:.2f} cues={self.cue_count_ratio:.2f} "
            f"dur={self.duration_ratio:.2f} density={self.density_ratio:.2f} "
            f"clusters=+/-{self.cluster_count_delta}"
        )


def compare_structures(
    candidate: str | list[Cue] | None,
    reference: str | list[Cue] | None,
) -> StructuralSimilarity:
    """Compare two subtitles' cue structure, shift-invariantly.

    Deliberately tolerant: providers split and merge cues differently, so cue
    count and per-cluster sizes are compared as ratios, never as equality.
    """
    candidate_profile = StructuralProfile.from_subtitle(candidate)
    reference_profile = StructuralProfile.from_subtitle(reference)

    if not candidate_profile.cue_count or not reference_profile.cue_count:
        return StructuralSimilarity(reasons=["one side has no cues to compare"])

    similarity = StructuralSimilarity(
        cue_count_ratio=round(
            _ratio_similarity(candidate_profile.cue_count, reference_profile.cue_count), 3
        )
    )

    if (
        candidate_profile.median_duration_ms is not None
        and reference_profile.median_duration_ms is not None
    ):
        similarity.duration_ratio = round(
            _ratio_similarity(
                candidate_profile.median_duration_ms, reference_profile.median_duration_ms
            ),
            3,
        )
    if (
        candidate_profile.density_cues_per_minute is not None
        and reference_profile.density_cues_per_minute is not None
    ):
        similarity.density_ratio = round(
            _ratio_similarity(
                candidate_profile.density_cues_per_minute,
                reference_profile.density_cues_per_minute,
            ),
            3,
        )
    similarity.cluster_count_delta = abs(
        candidate_profile.cluster_count - reference_profile.cluster_count
    )
    similarity.long_gap_count_delta = abs(
        candidate_profile.long_gap_count - reference_profile.long_gap_count
    )

    candidate_clusters = candidate_profile.cluster_cue_counts
    reference_clusters = reference_profile.cluster_cue_counts
    if candidate_clusters and reference_clusters:
        pairs = min(len(candidate_clusters), len(reference_clusters))
        matched = [
            _ratio_similarity(float(candidate_clusters[i]), float(reference_clusters[i]))
            for i in range(pairs)
        ]
        # A differing cluster total is itself informative, so scale the score.
        completeness = pairs / max(len(candidate_clusters), len(reference_clusters))
        similarity.cluster_profile_ratio = round(
            (sum(matched) / pairs) * completeness, 3
        )

    components: list[float] = [similarity.cue_count_ratio or 0.0]
    if similarity.duration_ratio is not None:
        components.append(similarity.duration_ratio)
    if similarity.density_ratio is not None:
        components.append(similarity.density_ratio)
    if similarity.cluster_profile_ratio is not None:
        components.append(similarity.cluster_profile_ratio)
    similarity.score = round(sum(components) / len(components), 3)

    # A large disagreement in where the scene breaks fall is strong evidence of
    # a structural difference, so it is surfaced explicitly.
    if similarity.cluster_count_delta and similarity.cluster_count_delta >= 2:
        similarity.reasons.append(
            f"scene-break count differs by {similarity.cluster_count_delta} "
            f"({candidate_profile.cluster_count} vs {reference_profile.cluster_count})"
        )
    if similarity.long_gap_count_delta >= 2:
        similarity.reasons.append(
            f"long-gap count differs by {similarity.long_gap_count_delta}"
        )
    return similarity


def classify_cut(
    *,
    median_offset_ms: float | None,
    p95_offset_ms: float | None,
    mad_offset_ms: float | None,
    drift_ms_per_minute: float | None,
    change_points: list[tuple[int, float]],
    structural: StructuralSimilarity | None,
    max_plausible_offset_ms: float,
    max_p95_ms: float,
    max_drift_ms_per_minute: float,
) -> CutVerdict:
    """Separate a re-timable offset from a genuinely different cut.

    Not every large offset is an error, and a small one does not prove the
    subtitle shares the cut, so the verdict needs the structural signal too:

    * a rigid shift with a matching structure is ``STABLE_OFFSET`` - re-timable;
    * offsets that grow past the drift ceiling are ``DRIFT``;
    * a discontinuity with a structure that still lines up is ``PIECEWISE``
      (recap / edited scene) and stays usable;
    * an offset beyond the plausibility ceiling, or a structure that does not
      correspond at all, is ``DIFFERENT_CUT``.
    """
    if median_offset_ms is None:
        return CutVerdict.UNKNOWN

    magnitude = abs(median_offset_ms)
    structure_ok = structural is None or structural.same_structure

    if change_points:
        # A step change in an otherwise corresponding timeline is structural,
        # not a failed alignment.
        if structure_ok:
            return CutVerdict.PIECEWISE
        return CutVerdict.DIFFERENT_CUT

    # Drift is judged against the same ceiling the analyzer uses, so the two
    # layers never disagree about whether a shift is progressive. Over a full
    # episode even a modest slope accumulates into seconds, so this must not
    # depend on the magnitude of the starting offset.
    if drift_ms_per_minute is not None and abs(drift_ms_per_minute) > max_drift_ms_per_minute:
        return CutVerdict.DRIFT

    if magnitude <= max_plausible_offset_ms:
        return CutVerdict.STABLE_OFFSET if structure_ok else CutVerdict.DIFFERENT_CUT

    # Beyond the ceiling, only a matching structure argues for salvage.
    if structure_ok and p95_offset_ms is not None and p95_offset_ms <= max_p95_ms:
        return CutVerdict.STABLE_OFFSET
    return CutVerdict.DIFFERENT_CUT
