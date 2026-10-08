"""Semantic reference ordering; hints never grant synchronization verification."""

from __future__ import annotations

from app.services.sync.matching import normalize_release_group
from app.services.sync.release_knowledge import (
    Comparison,
    ReleaseMetadata,
    compare_field,
    compatible_source,
)
from app.services.sync.target_binding import TargetBinding, classify_target_binding


def semantic_rank(target: ReleaseMetadata, reference: ReleaseMetadata, *, hash_match: bool = False) -> tuple[int, ...]:
    """Lexicographic precedence: identity, variants, master, source, group, quality.

    MovieHash here is only a search hint, as before. The orchestrator still
    requires its independent strict hash strategy before EXACT verification.
    Conflicts are ranked last, not deleted or treated as search constraints.
    """
    binding = classify_target_binding(None, None, target_metadata=target, reference_metadata=reference)
    def agreement(field: str) -> int:
        return {Comparison.MATCH: 0, Comparison.UNKNOWN: 1, Comparison.CONFLICT: 2}[compare_field(target, reference, field)]

    strong = ("cut_tags", "label_tags", "presentation_tags")
    return (
        0 if hash_match else 1,
        1 if binding.state == TargetBinding.CONFLICT else 0,
        sum(agreement(field) for field in ("title", "year", "season", "episode")),
        sum(agreement(field) == 2 for field in strong),
        sum(agreement(field) for field in strong),
        agreement("master_tags"), agreement("restoration_year"),
        agreement("packaging_tags"),
        0 if compatible_source(target.source_family, reference.source_family) else 1,
        agreement("service"),
        0 if target.release_group and normalize_release_group(target.release_group) == normalize_release_group(reference.release_group) else 1,
        sum(agreement(field) for field in ("resolution", "codec", "bit_depth", "hdr_flags", "audio_tags")),
    )
