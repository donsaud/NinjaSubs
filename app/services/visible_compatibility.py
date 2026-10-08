"""Single local compatibility index for subtitle percentages and presentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.sync.matching import normalize_release_group, parse_release_identity
from app.services.sync.release_knowledge import (
    Comparison,
    ReleaseMetadata,
    compare_field,
    compatible_source,
)
from app.services.sync.release_metadata import SEMANTIC_VERSION
from app.services.sync.target_binding import TargetBinding, classify_target_binding

VISIBLE_COMPATIBILITY_VERSION = "visible-compatibility-v1:" + SEMANTIC_VERSION
BANDS = {
    TargetBinding.EXACT: (100, 100),
    TargetBinding.STRONG: (85, 99),
    TargetBinding.WEAK: (65, 84),
    TargetBinding.UNKNOWN: (25, 64),
    TargetBinding.CONFLICT: (0, 24),
}


@dataclass(frozen=True)
class CompatibilityScore:
    percentage: int
    binding: TargetBinding
    evidence_confidence: str
    reason_codes: tuple[str, ...]


def value(item: Any, field: str, default: Any = None) -> Any:
    return item.get(field, default) if isinstance(item, dict) else getattr(item, field, default)


def exact_flag(item: Any) -> bool:
    return value(item, "is_hash_match") is True or value(item, "matched_by_hash") is True


def compatibility_score(
    target: ReleaseMetadata,
    candidate: ReleaseMetadata,
    *,
    exact_moviehash: bool = False,
    evidence: Any = None,
    timing_variant_match: bool = False,
    target_name: str | None = None,
    candidate_name: str | None = None,
) -> CompatibilityScore:
    binding = classify_target_binding(
        target_name or "<target metadata>",
        candidate_name or "<candidate metadata>",
        exact_moviehash=exact_moviehash is True,
        target_metadata=target,
        reference_metadata=candidate,
    )
    reasons = list(binding.reasons)
    if binding.state == TargetBinding.EXACT:
        return CompatibilityScore(100, binding.state, "exact", tuple(reasons))
    if binding.state == TargetBinding.CONFLICT:
        return CompatibilityScore(
            0 if "cut_conflict" not in reasons else 10, binding.state, "conflict", tuple(reasons)
        )

    def matched(field: str) -> bool:
        result = compare_field(target, candidate, field)
        if result != Comparison.UNKNOWN:
            reasons.append(f"{field}_{result.value}")
        return result == Comparison.MATCH

    # Hierarchy within a binding band; weak technical evidence has only one
    # small slot and never buys a stronger target-binding class.
    level = max(
        (
            weight
            for field, weight in (
                ("master_tags", 1),
                ("presentation_tags", 2),
                ("label_tags", 3),
                ("cut_tags", 4),
            )
            if matched(field)
        ),
        default=0,
    )
    source = compatible_source(target.source_family, candidate.source_family)
    group = bool(
        target.release_group
        and normalize_release_group(target.release_group)
        == normalize_release_group(candidate.release_group)
    )
    identity = sum(matched(field) for field in ("title", "year", "season", "episode"))
    technical = min(
        2,
        sum(
            matched(field)
            for field in (
                "resolution",
                "codec",
                "bit_depth",
                "hdr_flags",
                "audio_tags",
                "fps",
                "duration_hint_seconds",
            )
        ),
    )
    service = matched("service")
    if source:
        reasons.append("source_compatible")
    if group:
        reasons.append("release_group_match")
    if timing_variant_match:
        reasons.append("repack_proper_match")
    if binding.state == TargetBinding.STRONG:
        score = min(
            96,
            85
            + level * 2
            + 2 * group
            + source
            + min(identity, 2)
            + min(technical, 1)
            + service
            + timing_variant_match,
        )
    else:
        score = (
            BANDS[binding.state][0]
            + level * 4
            + 8 * source
            + 4 * group
            + min(identity, 4) * 2
            + technical
            + service
            + timing_variant_match
        )

    # An unconfirmed title cannot collect same-group/technical credit. This is
    # insufficient identity, not a newly invented hard cut conflict.
    if compare_field(target, candidate, "title") == Comparison.CONFLICT:
        score = BANDS[binding.state][0]

    trusted = (
        value(evidence, "sync_evidence_trusted") is True
        and value(evidence, "sync_target_binding") in {"strong", "exact"}
        and value(evidence, "sync_verification") in {"verified", "cached"}
        and value(evidence, "sync_state") in {"verified_synced", "verified_resynced"}
    )
    if trusted:
        score += 3
        reasons.append("target_bound_verified")
    low, high = BANDS[binding.state]
    return CompatibilityScore(
        max(low, min(high, score)),
        binding.state,
        "verified" if trusted else "metadata",
        tuple(reasons),
    )


def score_names(
    target_name: str | None,
    candidate_name: str | None,
    *,
    evidence: Any = None,
    exact_moviehash: bool = False,
) -> CompatibilityScore:
    return compatibility_score(
        parse_release_identity(target_name),
        parse_release_identity(candidate_name),
        exact_moviehash=exact_moviehash,
        evidence=evidence,
        target_name=target_name,
        candidate_name=candidate_name,
    )


def displayed_score(number: Any) -> int:
    """The same bounded integer conversion for ordering and badge formatting."""
    try:
        return max(0, min(100, int(round(float(number)))))
    except (TypeError, ValueError, OverflowError):
        return 0


def presentation_key(item: Any) -> tuple:
    """Compatibility first; preferences/quality never inflate the percentage."""
    return (
        -displayed_score(value(item, "match_percentage", 0)),
        int(value(item, "sync_lang_rank", 0) or 0),
        int(value(item, "compatibility_hi_rank", 1) or 0),
        int(value(item, "compatibility_quality_rank", 0) or 0),
        str(value(item, "release_name", "")),
        str(value(item, "provider", "")),
        str(value(item, "download_url", "")),
    )
