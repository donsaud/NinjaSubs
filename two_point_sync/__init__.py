"""Two-Point Auto-Sync nucleus (isolated, additive).

Independent linear synchronisation engine. Does not replace ALASS and is not
wired into the production pipeline. See the ``two_point_sync`` package modules
for the staged pipeline: parser -> sanitizer -> anchors -> transform ->
validator.
"""

from two_point_sync.anchors import AnchorResult, Regime, detect_anchors
from two_point_sync.parser import Cue, SubtitleParseError, format_srt, parse_srt, parse_srt_bytes
from two_point_sync.sanitizer import AnchorEligibility, classify_all
from two_point_sync.transform import (
    LinearTransform,
    TransformError,
    apply_transform,
    fit_from_anchors,
    fit_least_squares,
)
from two_point_sync.validator import Checkpoint, ValidationReport, validate_transform

__all__ = [
    "AnchorEligibility",
    "AnchorResult",
    "Checkpoint",
    "Cue",
    "LinearTransform",
    "Regime",
    "SubtitleParseError",
    "TransformError",
    "ValidationReport",
    "apply_transform",
    "classify_all",
    "detect_anchors",
    "fit_from_anchors",
    "fit_least_squares",
    "format_srt",
    "parse_srt",
    "parse_srt_bytes",
    "validate_transform",
]
