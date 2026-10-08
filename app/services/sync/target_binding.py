"""Target identity policy, independent of subtitle alignment measurements."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from app.services.sync.matching import (
    ReleaseMetadata,
    _member_episode_number,
    _season_number,
    normalize_release_group,
    parse_release_identity,
)


class TargetBinding(str, Enum):
    EXACT = "exact"
    STRONG = "strong"
    WEAK = "weak"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class BindingAssessment:
    state: TargetBinding
    reasons: tuple[str, ...]
    target: ReleaseMetadata
    reference: ReleaseMetadata

    @property
    def supports_verified(self) -> bool:
        return self.state in {TargetBinding.EXACT, TargetBinding.STRONG}


def classify_target_binding(
    target_name: str | None, reference_name: str | None, *,
    exact_moviehash: bool = False, embedded: bool = False,
) -> BindingAssessment:
    """Names are hints; only corroborated identity supports strict promotion.

    Callers may assert exact_moviehash only after the existing strict hash
    strategy proves it. Embedded means extraction from this target video.
    Different technical encodes and missing labels are never cut conflicts.
    """
    target, reference = parse_release_identity(target_name), parse_release_identity(reference_name)

    def result(state: TargetBinding, *reasons: str) -> BindingAssessment:
        return BindingAssessment(state, reasons, target, reference)

    if exact_moviehash or embedded:
        return result(TargetBinding.EXACT, "moviehash_exact" if exact_moviehash else "target_embedded")
    if target.year and reference.year and target.year != reference.year:
        return result(TargetBinding.CONFLICT, "year_conflict")
    for parser in (_season_number, _member_episode_number):
        a, b = parser(target_name), parser(reference_name)
        if a is not None and b is not None and a != b:
            return result(TargetBinding.CONFLICT, "episode_identity_conflict")
    cuts_a, cuts_b = set(target.cut_tags), set(reference.cut_tags)
    # Remaster, IMAX and Criterion can coexist. Only explicit incompatible cut
    # labels are contradictions; absence and different packaging remain unknown.
    exclusive_cuts = {"directors_cut", "extended", "theatrical", "final_cut", "international_cut"}
    if cuts_a & exclusive_cuts and cuts_b & exclusive_cuts and (
        cuts_a & exclusive_cuts != cuts_b & exclusive_cuts
    ):
        return result(TargetBinding.CONFLICT, "cut_conflict")
    editions_a, editions_b = set(target.edition_tags), set(reference.edition_tags)
    if (cuts_a - cuts_b or editions_a - editions_b
            or cuts_b - cuts_a or editions_b - editions_a):
        return result(TargetBinding.UNKNOWN, "edition_or_cut_identity_incomplete")
    sources_match = bool(target.source_family and target.source_family == reference.source_family)
    if not sources_match:
        return result(TargetBinding.UNKNOWN, "source_identity_insufficient")
    # References are discovered within the request's IMDb scope. Names still
    # need enough title information to avoid a bare technical tag earning trust.
    if not target.title or not reference.title:
        return result(TargetBinding.UNKNOWN, "release_identity_insufficient")
    def title_key(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.lower())

    if title_key(target.title) != title_key(reference.title):
        return result(TargetBinding.UNKNOWN, "title_identity_unconfirmed")
    if cuts_a or editions_a:
        return result(TargetBinding.STRONG, "edition_cut_match", "source_compatible")
    target_group = normalize_release_group(target.release_group)
    reference_group = normalize_release_group(reference.release_group)
    if target_group and target_group == reference_group:
        return result(TargetBinding.STRONG, "release_group_match", "source_compatible")
    def name_key(value: str | None) -> str:
        base = re.sub(r"\.(?:srt|ass|ssa|vtt|mkv|mp4|avi)$", "", value or "", flags=re.I)
        return re.sub(r"[^a-z0-9]", "", base.lower())

    if (target.year and target.year == reference.year
            and (target.resolution or target.codec)
            and name_key(target_name) == name_key(reference_name)):
        return result(TargetBinding.STRONG, "release_name_identity", "source_compatible")
    return result(TargetBinding.WEAK, "source_family_only")
