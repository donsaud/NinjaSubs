"""Evidence-driven sync gate: investigate only on mismatch evidence.

Philosophy: assume the selected subtitle needs no modification unless
release evidence gives a strong reason to suspect a timing mismatch.
Synchronization is the last resort, never the default execution path.

This module decides only whether to ENTER the expensive synchronization
workflow (reference discovery, verification, ALASS). It performs no timing
measurement and overrides no verifier: when investigation runs, the
existing pipeline (already-aligned check, Timeline Affine, Two-Point,
ALASS fallback, timing-v3 authority, pristine reconstruction, final HTTP
integrity) behaves exactly as before.

Evidence comes from the canonical release parser (GuessIt + TRaSH-rules
vocabulary via :func:`parse_release_identity`); no second parser or
scoring framework is introduced, and the user-facing ranking percentage
is never consulted. UNKNOWN is never a mismatch: missing platform, group,
resolution, codec, or service information cannot trigger investigation.
A release-group difference alone never triggers either.

Outcomes:

* ``FAST_ORIGINAL`` -- no mismatch evidence; serve the pristine original
  immediately with no reference discovery, download, or ALASS. This is
  metadata-based delivery, NOT verified synchronization: no verified
  cache entry is written and no Sticky Reference is promoted.
* ``INVESTIGATE`` -- credible mismatch evidence (or an explicit request,
  or too little metadata to judge); run the normal pipeline.
* ``IDENTITY_CONFLICT`` -- the subtitle demonstrably belongs to different
  media (title/year/episode conflict); serve the original without
  investigation. Syncing across media would be wrong, not merely slow.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.services.sync.matching import (
    is_informative_release_name,
    normalize_release_group,
    parse_release_identity,
)
from app.services.sync.release_knowledge import (
    EXCLUSIVE_CUTS,
    Comparison,
    compare_field,
    compatible_source,
)

#: Frames-per-second gap above which two releases cannot share a timeline
#: (PAL 25 vs film 23.976 is the classic case). Filenames rarely carry FPS,
#: so this fires rarely and only on explicit evidence.
FPS_CONFLICT_THRESHOLD = 1.0


class MetadataGateAction(str, Enum):
    FAST_ORIGINAL = "fast_original"
    INVESTIGATE = "investigate"
    IDENTITY_CONFLICT = "identity_conflict"


#: Closed reason allowlist so only bounded category strings reach logs.
REASONS = (
    "same_release_lineage",
    "no_mismatch_evidence",
    "metadata_insufficient",
    "insufficient_metadata_for_gate",
    "source_family_conflict",
    "edition_or_cut_conflict",
    "frame_rate_conflict",
    "episode_identity_conflict",
    "title_identity_conflict",
    "year_identity_conflict",
    "explicit_sync_request",
)


@dataclass(frozen=True)
class MetadataGateDecision:
    action: MetadataGateAction
    reasons: tuple[str, ...] = ()

    @property
    def investigate(self) -> bool:
        return self.action is MetadataGateAction.INVESTIGATE

    @property
    def fast(self) -> bool:
        return self.action is MetadataGateAction.FAST_ORIGINAL


def _reason(code: str) -> tuple[str, ...]:
    assert code in REASONS, code
    return (code,)


def decide_metadata_gate(
    video_name: str | None,
    subtitle_name: str | None,
    *,
    explicit_sync_request: bool = False,
) -> MetadataGateDecision:
    """Decide whether sync investigation is warranted for this pairing."""
    if explicit_sync_request:
        return MetadataGateDecision(
            MetadataGateAction.INVESTIGATE, _reason("explicit_sync_request"))
    if not video_name or not subtitle_name:
        # The gate compares two names; with fewer it cannot judge, so the
        # legacy investigation path runs unchanged (fail open to accuracy).
        return MetadataGateDecision(
            MetadataGateAction.INVESTIGATE, _reason("insufficient_metadata_for_gate"))
    if not is_informative_release_name(subtitle_name):
        # A bare label ("Arabic.srt") or debrid junk carries no contrary
        # signal: unknown metadata is not evidence of incompatibility.
        return MetadataGateDecision(
            MetadataGateAction.FAST_ORIGINAL, _reason("metadata_insufficient"))
    if not is_informative_release_name(video_name):
        # The video is the ground-truth timeline; without a readable video
        # lineage compatibility cannot be asserted, so abstain to legacy.
        return MetadataGateDecision(
            MetadataGateAction.INVESTIGATE, _reason("insufficient_metadata_for_gate"))
    video = parse_release_identity(video_name)
    subtitle = parse_release_identity(subtitle_name)

    for field, reason in (("season", "episode_identity_conflict"),
                          ("episode", "episode_identity_conflict")):
        if compare_field(video, subtitle, field) is Comparison.CONFLICT:
            return MetadataGateDecision(
                MetadataGateAction.IDENTITY_CONFLICT, _reason(reason))
    if compare_field(video, subtitle, "title") is Comparison.CONFLICT:
        return MetadataGateDecision(
            MetadataGateAction.IDENTITY_CONFLICT, _reason("title_identity_conflict"))
    if compare_field(video, subtitle, "year") is Comparison.CONFLICT:
        return MetadataGateDecision(
            MetadataGateAction.IDENTITY_CONFLICT, _reason("year_identity_conflict"))

    video_cuts = set(video.cut_tags) & EXCLUSIVE_CUTS
    subtitle_cuts = set(subtitle.cut_tags) & EXCLUSIVE_CUTS
    if video_cuts != subtitle_cuts and (video_cuts or subtitle_cuts):
        # An exclusive cut tag (Extended, Director's Cut, Theatrical, ...)
        # is never cosmetic: it asserts a different timeline, on either
        # side. Coexistable labels (Remaster, IMAX, Criterion) are not
        # exclusive cuts and never reach this branch.
        return MetadataGateDecision(
            MetadataGateAction.INVESTIGATE, _reason("edition_or_cut_conflict"))
    video_ed = set(video.cut_tags) | set(video.edition_tags)
    subtitle_ed = set(subtitle.cut_tags) | set(subtitle.edition_tags)
    if video_ed and subtitle_ed and video_ed != subtitle_ed:
        return MetadataGateDecision(
            MetadataGateAction.INVESTIGATE, _reason("edition_or_cut_conflict"))

    if (video.source_family and subtitle.source_family
            and not compatible_source(video.source_family, subtitle.source_family)):
        return MetadataGateDecision(
            MetadataGateAction.INVESTIGATE, _reason("source_family_conflict"))

    if (video.fps is not None and subtitle.fps is not None
            and abs(video.fps - subtitle.fps) > FPS_CONFLICT_THRESHOLD):
        return MetadataGateDecision(
            MetadataGateAction.INVESTIGATE, _reason("frame_rate_conflict"))

    # Group, resolution, codec, remux, service, and missing fields never
    # trigger: a different group alone is not a timing signal, and UNKNOWN
    # is not evidence of incompatibility.
    if (video.source_family and subtitle.source_family
            and compatible_source(video.source_family, subtitle.source_family)
            and normalize_release_group(video.release_group)
            and normalize_release_group(video.release_group)
            == normalize_release_group(subtitle.release_group)):
        return MetadataGateDecision(
            MetadataGateAction.FAST_ORIGINAL, _reason("same_release_lineage"))
    return MetadataGateDecision(
        MetadataGateAction.FAST_ORIGINAL, _reason("no_mismatch_evidence"))
