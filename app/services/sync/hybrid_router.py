"""Controlled Two-Point/ALASS routing and single-authority verification.

The default-OFF branch preserves the Legacy/correspondence routing contract.
Inside the scoped feature gate, algorithms only propose candidate bytes and
Unified Verifier alone judges synchronization correctness. Legacy results are
recorded as diagnostics. The orchestrator retains reference eligibility,
Target Binding, may_serve, caches, and original fallback authorization.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from app.services.sync.alignment import (
    AlignmentAnalyzer,
    may_serve_synchronized,
)
from app.services.sync.reference_content_identity import identical_content
from two_point_sync.anchors import AnchorResult, detect_anchors
from two_point_sync.parser import (
    Cue as NucleusCue,
)
from two_point_sync.parser import (
    SubtitleParseError,
    format_srt,
    parse_srt,
)
from two_point_sync.sanitizer import classify_all
from two_point_sync.transform import LinearTransform, apply_transform, fit_from_anchors

logger = logging.getLogger(__name__)

#: Strategy identifiers for provenance, logs, cache decision segments.
STRATEGY_ID_LINEAR = "hybrid_two_point_linear"
STRATEGY_ID_NONLINEAR = "hybrid_alass_nonlinear"

#: Model version, baked into provenance so future algorithm changes cannot
#: masquerade as this one in cache or telemetry. Bumped to v2 with the
#: correspondence-aware verification methodology: v1-era artifacts (legacy
#: positional verification only) never collide with v2 decisions, and cached
#: v1 UNVERIFIED outcomes cannot silently flip under the new methodology.
MODEL_VERSION = "hybrid-v2"

#: Feature gate name (log/telemetry label).
GATE_SETTING = "HYBRID_TWO_POINT_ALASS_ENABLED"

#: Module-level activation flag. Deliberately NOT a public configuration
#: setting (and not in ``.env``): it defaults to False so live serving is
#: unchanged, and isolated tests enable it with ``monkeypatch`` on this
#: module. Any future live activation requires separate approval.
_enabled = False

#: Environment opt-in for the controlled local test deployment. Read live on
#: every call (no restart needed to flip it back off): the container runs
#: with this variable present only via a temporary Compose override file
#: that is never part of the default ``docker compose up`` path.
TEST_ENABLED_ENV = "HYBRID_TEST_ENABLED"

#: Optional scope restriction ``"imdb:season:episode"`` (e.g.
#: ``"tt0773262:8:1"``) limiting test activation to one video identity.
#: The exact value ``"ALL"`` selects explicit local Unified-primary mode
#: (see ``scope_allows``); anything else malformed fails closed.
TEST_SCOPE_ENV = "HYBRID_TEST_SCOPE"

#: Exact scope value selecting explicit local Unified-primary operation.
#: Must be matched exactly (after stripping); ``"all"``, ``"ALL:*"``, or any
#: other variant is malformed and fails closed. Never the default: an absent
#: or empty scope never means ALL.
ALL_SCOPE = "ALL"

#: Emergency OFF switch. When ``"1"``, Unified/hybrid authority is disabled
#: process-wide regardless of every other flag. Server-side only: no request,
#: subtitle, or user-preference path can set it. Removing the variable (or
#: the temporary Compose override that carries it) restores the configured
#: behavior with no code change.
KILL_ENV = "HYBRID_UNIFIED_DISABLED"


def is_enabled() -> bool:
    """True only with an explicit in-process or scoped local-test opt-in."""
    if os.environ.get(KILL_ENV) == "1":
        return False
    if _enabled:
        return True
    # Environment opt-in additionally requires a parseable scope: an enabled
    # flag with an empty or malformed scope must never mean global activation.
    return os.environ.get(TEST_ENABLED_ENV) == "1" and test_scope() is not None


def unified_authority_enabled(meta: dict) -> bool:
    """Use the single authority only inside the existing controlled scope."""
    return is_enabled() and scope_allows(meta)


def activation_mode() -> str:
    """Operational mode label for telemetry and reporting (no policy effect).

    ``"OFF"`` (disabled), ``"SCOPED"`` (explicit valid identity scope), or
    ``"ALL"`` (explicit local Unified-primary scope). The kill switch always
    reports OFF.
    """
    if os.environ.get(KILL_ENV) == "1":
        return "OFF"
    scope = test_scope()
    if (
        scope is not None
        and scope[0] == ALL_SCOPE
        and scope[1] is None
        and scope[2] is None
        and (_enabled or os.environ.get(TEST_ENABLED_ENV) == "1")
    ):
        return "ALL"
    if is_enabled():
        return "SCOPED"
    return "OFF"


def test_scope() -> tuple[str | None, int | None, int | None] | None:
    """Parse ``HYBRID_TEST_SCOPE`` into ``(imdb, season, episode)``.

    The exact value ``"ALL"`` parses as the explicit local-primary sentinel
    ``("ALL", None, None)``; every other shape follows the identity rules.
    """
    raw = (os.environ.get(TEST_SCOPE_ENV) or "").strip()
    if not raw:
        return None
    if raw == ALL_SCOPE:
        return ALL_SCOPE, None, None
    parts = raw.split(":")
    if len(parts) not in {1, 3} or not re.fullmatch(r"tt\d+", parts[0].strip()):
        return None
    try:
        imdb = parts[0].strip() or None
        season = int(parts[1]) if len(parts) > 1 and parts[1].strip() else None
        episode = int(parts[2]) if len(parts) > 2 and parts[2].strip() else None
    except ValueError:
        return None
    if len(parts) == 3 and (season is None or episode is None or season < 1 or episode < 1):
        return None
    if imdb is None and season is None and episode is None:
        return None
    return imdb, season, episode


def scope_allows(meta: dict) -> bool:
    """True when no scope is configured or the request matches the scope.

    Configured identity fields are mandatory; filenames cannot replace them.
    The exact ``"ALL"`` scope authorizes evaluation for any request meta, but
    authorizes nothing by itself: Target Binding, may_serve, and final HTTP
    integrity still decide every delivery downstream.
    """
    scope = test_scope()
    if scope is None:
        return not bool((os.environ.get(TEST_SCOPE_ENV) or "").strip())
    imdb, season, episode = scope
    if imdb == ALL_SCOPE and season is None and episode is None:
        return True
    meta_imdb = str(meta.get("imdb_id") or meta.get("imdb") or "")
    meta_season = meta.get("season")
    meta_episode = meta.get("episode")
    if imdb and imdb.lower() != meta_imdb.strip().lower():
        return False
    if season is None and episode is None:
        # IMDb-only (film) scope: a legitimate film request carries neither
        # season nor episode. Their presence marks misrouted series metadata,
        # so fail closed rather than admit a partial series identity.
        for present in (meta_season, meta_episode):
            if present is not None and str(present).strip() != "":
                return False
        return True
    if season is not None:
        try:
            if not re.fullmatch(r"[0-9]+", str(meta_season)) or int(str(meta_season)) != season:
                return False
        except (TypeError, ValueError):
            return False
    if episode is not None:
        try:
            if not re.fullmatch(r"[0-9]+", str(meta_episode)) or int(str(meta_episode)) != episode:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _filename_scope_allows(meta: dict) -> bool:
    """Filename fallback when request metadata lacks identity fields."""
    name = str(meta.get("target_filename") or meta.get("filename") or "").lower()
    if not name or "dexter" not in name:
        return False
    compact = re.sub(r"[^a-z0-9]", "", name)
    return bool(re.search(r"s0?8e0?1", compact))


#: Minimum target-cue retention for an ALASS candidate (mirrors the
#: production MIN_CUE_RETENTION floor without redefining it).
MIN_TARGET_RETENTION = 0.80

#: Type of the injected ALASS runner: original target + reference in,
#: candidate text (or None) out. Production passes a thin wrapper around
#: ``SubtitleSyncService.sync_async``; tests inject stubs or the real binary.
AlassRunner = Callable[[str, str], Awaitable[str | None]]


@dataclass(frozen=True)
class HybridRouteResult:
    """Outcome of one hybrid routing evaluation. Never serves by itself."""

    evaluated: bool
    route: str  # linear | nonlinear | insufficient | disabled | failed
    accepted: bool
    transformed_text: str | None = None
    scale: float | None = None
    offset_ms: float | None = None
    evaluation: Any | None = None
    regional: dict[str, Any] = field(default_factory=dict)
    correspondence: dict[str, Any] = field(default_factory=dict)
    correspondence_evaluation: Any | None = None
    reason_codes: tuple[str, ...] = ()
    timings_ms: dict[str, float] = field(default_factory=dict)
    reference_sha: str = ""




def _reference_sha(reference_text: str) -> str:
    return hashlib.sha256(reference_text.encode("utf-8")).hexdigest()


def classify_pair(target_text: str, reference_text: str) -> dict[str, Any]:
    """Run the isolated Two-Point nucleus; return a plain classification dict.

    Raises nothing: every failure mode maps to ``route="insufficient"`` with
    an explicit reason, preserving legacy fallback paths.
    """
    started = time.monotonic()

    def _timings() -> dict[str, float]:
        return {"classify_ms": (time.monotonic() - started) * 1000.0}

    try:
        target_cues = parse_srt(target_text)
        reference_cues = parse_srt(reference_text)
    except SubtitleParseError as exc:
        return {
            "route": "insufficient",
            "reason": f"nucleus_parse_failed:{type(exc).__name__}",
            "timings_ms": _timings(),
        }
    identity = identical_content(target_text, reference_text)
    if identity:
        return {
            "route": "insufficient",
            "reason": f"reference_content_identical_to_target:{identity}",
            "timings_ms": _timings(),
        }
    try:
        result: AnchorResult = detect_anchors(
            target_cues,
            reference_cues,
            target_scores=classify_all(target_cues),
            reference_scores=classify_all(reference_cues),
        )
    except Exception as exc:  # fail closed, always
        return {
            "route": "insufficient",
            "reason": f"nucleus_error:{type(exc).__name__}",
            "timings_ms": _timings(),
        }
    if result.accepted and result.first_anchor is not None and result.last_anchor is not None:
        return {
            "route": "linear",
            "anchor_result": result,
            "target_cues": target_cues,
            "reference_cues": reference_cues,
            "timings_ms": _timings(),
        }
    if "NONLINEAR_DISCONTINUITY" in result.reasons:
        return {
            "route": "nonlinear",
            "anchor_result": result,
            "target_cues": target_cues,
            "reference_cues": reference_cues,
            "timings_ms": _timings(),
        }
    return {
        "route": "insufficient",
        "reason": f"nucleus_rejected:{';'.join(result.reasons) or 'unknown'}",
        "anchor_result": result,
        "timings_ms": _timings(),
    }


def build_linear_output(
    target_cues: list[NucleusCue], anchor_result: AnchorResult
) -> tuple[str, LinearTransform] | None:
    """Render the nucleus transform; None unless structure is preserved."""
    first = anchor_result.first_anchor
    last = anchor_result.last_anchor
    if first is None or last is None:
        return None
    try:
        transform = fit_from_anchors(
            first.target_ms, first.reference_ms, last.target_ms, last.reference_ms
        )
        converted = apply_transform(target_cues, transform)
    except Exception:
        return None
    if len(converted) != len(target_cues):
        return None
    if any(new.text != old.text for new, old in zip(converted, target_cues, strict=True)):
        return None
    return format_srt(converted), transform


async def attempt_hybrid(
    target_text: str,
    reference_text: str,
    *,
    analyzer: AlignmentAnalyzer | None = None,
    alass_runner: AlassRunner | None = None,
    reference_partial: bool = False,
    max_plausible_offset_ms: int = 20_000,
    unified_authority: bool | None = None,
) -> HybridRouteResult:
    """Evaluate one target/reference pair through the hybrid routing table.

    Never serves, never discovers references, never edits subtitles. Returns
    a result the caller serves only when every gate (including Target
    Binding, enforced by the caller) passes.
    """

    timings_ms: dict[str, float] = {}
    reference_sha = _reference_sha(reference_text)
    classification = classify_pair(target_text, reference_text)
    timings_ms.update(classification.get("timings_ms", {}))
    route = classification.get("route", "insufficient")

    # Algorithms propose bytes; only Unified judges their correctness.
    candidate = None
    scale = offset_ms = None
    try:
        if route == "linear":
            built = build_linear_output(classification["target_cues"], classification["anchor_result"])
            if built:
                candidate, transform = built
                scale, offset_ms = transform.scale, transform.offset_ms
        elif route == "nonlinear" and not reference_partial and alass_runner is not None:
            candidate = await alass_runner(target_text, reference_text)
    except Exception as exc:
        reason = (f"alass_failed:{type(exc).__name__}" if route == "nonlinear"
                  else f"candidate_error:{type(exc).__name__}")
        return HybridRouteResult(True, route, False, reason_codes=(reason,),
                                 reference_sha=reference_sha, timings_ms=timings_ms)
    if not candidate:
        return HybridRouteResult(
            True, route, False,
            reason_codes=(str(classification.get("reason") or "no_candidate"),),
            reference_sha=reference_sha, timings_ms=timings_ms)
    if route == "nonlinear":
        # Same sanitization boundary as the ordinary ALASS path: the
        # runner aligns a sanitized working copy while Unified judges
        # against the original. Reconstruct original dialogue under the
        # verified ALASS timings when unambiguous; otherwise verify the
        # raw bytes (fail closed). Linear candidates preserve target text
        # by construction and need no reconstruction.
        from app.services.sync.candidate_reconstruction import (
            reconstruct_with_original_dialogue,
        )

        try:
            rebuilt = reconstruct_with_original_dialogue(target_text, candidate)
        except Exception:
            rebuilt = None
        if rebuilt is not None:
            candidate = rebuilt
    evaluation, telemetry = await unified_review_async(
        target_text=target_text, reference_text=reference_text,
        synced_text=candidate, candidate_algorithm=f"hybrid_{route}",
    )
    accepted = may_serve_synchronized(evaluation.sync_state, evaluation.verification)
    return HybridRouteResult(
        True, route, accepted,
        # Unverified bytes never leave this function: callers may only serve
        # transformed_text when accepted is True (enforced again downstream).
        transformed_text=candidate if accepted else None, evaluation=evaluation,
        correspondence=telemetry, reason_codes=tuple(telemetry.get("reasons", ())),
        reference_sha=reference_sha, scale=scale, offset_ms=offset_ms, timings_ms=timings_ms,
    )



#: Bound for one pre-delivery verification pass. Detection on full-episode
#: pairs measures ~0.3-1.5s; 10s is generous yet keeps the serving budget
#: intact. Exceeding it fails closed (see policy table below).
V2_TIMEOUT_SECONDS = 10.0


def unified_review(*, target_text: str, reference_text: str, synced_text: str,
                   candidate_algorithm: str):
    """Fresh single-authority evaluation and privacy-safe decision telemetry."""
    from app.services.sync.unified_verification import to_evaluation, verify_candidate

    started = time.monotonic()
    verdict = verify_candidate(target_text, reference_text, synced_text,
                               candidate_algorithm=candidate_algorithm)
    evaluation = to_evaluation(verdict)
    if verdict.proposed_verdict == "VERIFIED" and target_text == synced_text:
        from app.services.sync.alignment import SyncState, VerificationAvailability

        evaluation.set_verdict(SyncState.VERIFIED_SYNCED, VerificationAvailability.VERIFIED)
    accepted = verdict.proposed_verdict == "VERIFIED"
    telemetry = {
        "evaluated": True, "verification_method": verdict.verification_method,
        "verification_version": verdict.verification_version,
        "verdict": "ACCEPTED" if accepted else (
            "REJECTED" if verdict.proposed_verdict == "REJECTED" else "INSUFFICIENT_EVIDENCE"),
        "route": verdict.temporal_result.get("route", verdict.correspondence_result.get("route", "unknown")),
        "candidate_sha": verdict.candidate_sha256,
        "reference_sha": _reference_sha(reference_text),
        "served_sha": verdict.candidate_sha256 if accepted else "not_served",
        "transformed_sha": verdict.candidate_sha256[:16],
        "proceed": accepted, "reasons": verdict.failure_reasons,
        "independent_groups_used": verdict.independent_evidence_count,
        "temporal_coverage_thirds": verdict.coverage,
        "dialogue_scope": verdict.regional_result,
        "median_residual_ms": verdict.residual_statistics.get("median"),
        "timings_ms": {"unified_ms": (time.monotonic() - started) * 1000.0},
    }
    return evaluation, telemetry


async def unified_review_async(*, target_text: str, reference_text: str, synced_text: str,
                               candidate_algorithm: str, timeout_s: float = V2_TIMEOUT_SECONDS):
    """Bounded, fail-closed verification, including unexpected internal errors."""
    from app.services.sync.alignment import SubtitleEvaluation

    try:
        return await asyncio.wait_for(asyncio.to_thread(
            unified_review, target_text=target_text, reference_text=reference_text,
            synced_text=synced_text, candidate_algorithm=candidate_algorithm,
        ), timeout_s)
    except Exception as exc:
        evaluation = SubtitleEvaluation()
        reason = "unified_timeout" if isinstance(exc, TimeoutError) else f"unified_error:{type(exc).__name__}"
        evaluation.reasons.append(reason)
        return evaluation, {
            "evaluated": True, "verification_method": "unified",
            "verification_version": "timing-v3", "verdict": "ERROR",
            "proceed": False, "served_sha": "not_served", "reasons": (reason,),
            "candidate_sha": hashlib.sha256(synced_text.encode()).hexdigest(),
        }


def pre_delivery_review(
    *,
    target_text: str,
    reference_text: str,
    synced_text: str,
    meta: dict,
    exemption: str,
    legacy_unverified: bool = True,
) -> tuple[bool, dict[str, Any]]:
    """Review any proposed delivery with Unified inside the controlled gate.

    Legacy acceptance cannot rescue abstention or an error. Outside the gate
    this is a no-op. ``legacy_unverified`` is retained for API compatibility.
    """
    if not unified_authority_enabled(meta):
        return True, {"evaluated": False, "reason": "hybrid_disabled_or_scoped_out"}
    try:
        _, telemetry = unified_review(
            target_text=target_text, reference_text=reference_text,
            synced_text=synced_text, candidate_algorithm=exemption,
        )
    except Exception as exc:
        return False, {"evaluated": True, "verification_method": "unified",
                       "verdict": "ERROR", "proceed": False,
                       "served_sha": "not_served",
                       "reasons": (f"unified_error:{type(exc).__name__}",)}
    return telemetry["proceed"], telemetry


async def pre_delivery_review_async(
    *,
    target_text: str,
    reference_text: str,
    synced_text: str,
    meta: dict,
    exemption: str,
    legacy_unverified: bool = True,
    timeout_s: float = V2_TIMEOUT_SECONDS,
) -> tuple[bool, dict[str, Any]]:
    """Bounded Unified review; timeout/error always blocks inside the gate."""
    if not unified_authority_enabled(meta):
        return True, {"evaluated": False, "reason": "hybrid_disabled_or_scoped_out"}
    _, telemetry = await unified_review_async(
        target_text=target_text, reference_text=reference_text, synced_text=synced_text,
        candidate_algorithm=exemption, timeout_s=timeout_s,
    )
    return telemetry["proceed"], telemetry
