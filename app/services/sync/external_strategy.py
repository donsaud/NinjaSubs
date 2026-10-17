"""External reference resolution for subtitle auto-sync.

All providers (SubDL / SubSource / OpenSubtitles) are queried concurrently and
their candidates are pooled, then scored against the *video's* filename. The
single best-scoring candidate is downloaded and used as the ``alass`` reference.

``alass`` only anchors speech-timing intervals, so any high-accuracy official
track works â€” English and Arabic are searched. The score
favours a reference whose release quality matches the playing video (same source
family, same streaming service family, same release group, resolution and
codec), so a BluRay-quality Arabic subtitle played against a WEB-DL video is
re-timed against a WEB-DL reference rather than a mismatched edition. Unresolved references fail closed; synchronization still owns timing safety.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import time
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from cachetools import TTLCache

from app.services import release_scoring_integration as release_scoring
from app.services.subtitle_matcher import extract_metadata
from app.services.sync import discovery_hints
from app.services.sync.cache import ReferenceDiskCache
from app.services.sync.case_capture import capture_provider
from app.services.sync.decode import (
    decode_payload,
    looks_like_cumulative_pack,
    select_zip_member,
)
from app.services.sync.matching import (
    ReleaseMetadata,
    _source_kind,
    is_informative_release_name,
    is_retail_disc_source,
    normalize_release_group,
    parse_release_identity,
)
from app.services.sync.query import ResolvedReference
from app.services.sync.reference_v2 import release_family_key as _release_family_key
from app.services.sync.telemetry import current_sync_timing

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.services.sync.query import ReferenceQuery

# Minimum pause between consecutive reference DOWNLOADS from the same
# provider within one request.
#
# Why this exists: resolving the candidate pool eagerly meant up to
# REFERENCE_POOL_LIMIT downloads fired back-to-back. Against SubDL that tripped
# the provider's rate limiter and opened the circuit breaker mid-request:
#
#   GET dl.subdl.com/...3610179-8527589.zip  ->  HTTP 429 Too Many Requests
#   WARNING: Subdl circuit breaker tripped (HTTP 429); skipping SubDL for 60s
#   subdl returned 0 candidate(s)  ->  resolve_all: 0 distinct candidate(s)
#
# A later candidate in that same request then had no reference at all and fell
# back to serving the original. Pacing restores politeness without weakening
# any verification gate.
_REFERENCE_DOWNLOAD_MIN_INTERVAL_SECONDS = 0.6

logger = logging.getLogger(__name__)

#: Bounds for the per-process memo of candidates rejected for a target.
_REJECTED_MEMO_MAX_STEMS = 4096
_REJECTED_MEMO_TTL_SECONDS = 6 * 3600

# Below the request budget (SYNC_TOTAL_REQUEST_BUDGET, default 7.5s) so
# reference resolution finishes inline and still leaves room for alass.
_DEFAULT_TIMEOUT = 5.0
_MIN_REFERENCE_BYTES = 5120
# Leave enough room for a completed Tier 1 result to be returned before the
# provider deadline. Tier 2 is opportunistic enrichment, never a precondition
# for keeping candidates that are already in hand.
_TIER2_RETURN_MARGIN_SECONDS = 0.05
_TIER2_MIN_SECONDS = 0.05
# Final cancellation guard for a provider coroutine that ignores its own
# per-tier deadline. Real provider searches return or raise before the deadline;
# the guard must never fire ahead of a completed Tier 1 return.
_PROVIDER_HARD_TIMEOUT_GRACE_SECONDS = 0.25

# Language codes searched/fetched for the *reference* track (independent of the
# user's subtitle-language preference): English is the primary universal retail
# timing reference anchor, with Arabic as secondary reference. Foreign European
# tracks (it/de/fr/es) are excluded from the reference search pool because they
# often carry distributor intro offsets, dubbed audio pacing, or PAL 25fps shifts.
_REFERENCE_LANGUAGES = ("en", "ar")

# Historical diagnostic labels only. ReferencePolicy owns all ordering.
TIER_HASH = 0  # byte-exact OpenSubtitles MovieHash: undisputed ground truth
TIER_EXACT_GROUP = 1  # exact release-group token match (PiR8, FLUX, CHD, NTb, FraMeSToR...)
TIER_SOURCE_EDITION = 2  # same source medium and compatible cut (BluRay REMUX == BluRay REMUX)
TIER_FALLBACK = 3  # everything else that matched the basic title/episode query

_TIER_LABELS = {
    TIER_HASH: "hash",
    TIER_EXACT_GROUP: "exact-group",
    TIER_SOURCE_EDITION: "source-edition",
    TIER_FALLBACK: "fallback",
}

# Planning reserves one download slot for a promising unresolved season pack
# when the ordinary weighted budget is otherwise full. One slot is the minimum
# that gives extraction a chance without adding network traffic: the final plan
# still contains no more candidates than REFERENCE_POOL_LIMIT.
_REFERENCE_EXPLORATION_SLOTS = 1

# Planning diagnostics are bounded: explain the selected plan, a compact set of
# nearby exclusions, and the most relevant policy-ineligible responses without
# dumping or logging an entire provider pool.
_REFERENCE_PLAN_REPORT_SELECTED_LIMIT = 12
_REFERENCE_PLAN_REPORT_EXCLUDED_LIMIT = 6
_REFERENCE_PLAN_REPORT_UNRANKED_LIMIT = 4


def tier_label(tier: int) -> str:
    """Human-readable name for a reference tier (used in logs and cache keys)."""
    return _TIER_LABELS.get(tier, "fallback")


def _release_name(release) -> str:
    return str(getattr(release, "release_name", "") or "")


def _plan_evidence_states(result) -> dict[str, dict[str, str]]:
    """Return bounded source/group compatibility evidence for diagnostics."""
    states = {item.field: item for item in result.evidence if item.field in {"source", "release_group"}}
    return {
        field: {
            "state": states[field].state.value if field in states else "missing",
            "relation": states[field].relationship if field in states else "missing",
        }
        for field in ("source", "release_group")
    }


def _candidate_source_matches(result) -> bool:
    """Whether weighted evidence says this candidate shares the target medium.

    This uses the authoritative matcher's source comparison directly, so it
    works for BluRay, WEB-DL, HDTV, DVD and the other supported families
    without adding a hardcoded release preference or a second scoring ladder.
    """
    return _plan_evidence_states(result)["source"]["state"] == "match"


def _exploration_candidate(result) -> bool:
    """Whether an unresolved pack is eligible for one bounded extraction chance.

    Readiness is deliberately not invented here. The pack must already be
    discovery-eligible and must not already be ready; its episode identity is
    still established only by downloading it, extracting the requested member,
    and re-scoring that member through the existing weighted engine.
    """
    season = next((item for item in result.evidence if item.field == "season"), None)
    return (
        result.discovery_eligible
        and not result.ready_reference_eligible
        and result.candidate.unresolved_pack
        and season is not None
        and season.state.value == "match"
        and _candidate_source_matches(result)
    )


def download_source_digest(provider_label: str | None, download_url: str | None) -> str | None:
    """Digest a download source without retaining its address.

    Query strings, fragments, and embedded userinfo are discarded before
    hashing, so signed URLs and credentials never reach the cache, logs, or
    telemetry. The digest only identifies an exact source already in hand.
    """
    label = str(provider_label or "").casefold()
    raw = str(download_url or "").strip()
    if not label or not raw:
        return None
    try:
        parts = urlsplit(raw)
        host = parts.hostname or ""
        if parts.port:
            host = f"{host}:{parts.port}"
        canonical = f"{parts.scheme.casefold()}://{host.casefold()}{parts.path}" if parts.scheme or host else parts.path
    except Exception:
        canonical = raw
    return hashlib.sha256(f"{label}|{canonical}".encode()).hexdigest()[:16]








def reference_tier(target_name: str | None, release) -> int:
    """Diagnostic label/quota compatibility only; never candidate ordering."""
    if getattr(release, "is_hash_match", None) is True or getattr(release, "matched_by_hash", None) is True:
        return TIER_HASH
    result = release_scoring.compare_metadata(target_name or "", _release_name(release))
    states = {e.field: e.state.value for e in result.evidence}
    if states.get("source") == "match":
        return TIER_EXACT_GROUP if states.get("release_group") == "match" else TIER_SOURCE_EDITION
    return TIER_FALLBACK





def _is_rel_provider_broken(rel) -> bool:
    """True when the candidate's provider circuit breaker is currently open."""
    prov = str(getattr(rel, "provider", "") or "").lower()
    if prov == "opensubtitles":
        from app.providers.opensubtitles import OPENSUBTITLES_BREAKER
        return OPENSUBTITLES_BREAKER.is_open()
    if prov == "subdl":
        from app.providers.subdl import SUBDL_BREAKER
        return SUBDL_BREAKER.is_open()
    if prov == "subsource":
        from app.providers.subsource import SUBSOURCE_BREAKER
        return SUBSOURCE_BREAKER.is_open()
    return False


def _planning_metadata(name: str | None) -> ReleaseMetadata:
    """Enrich missing source hints with the existing general source extractor.

    Some supported legacy spellings (e.g. DVDRip) are absent from the semantic
    vocabulary. Never override known or ambiguous evidence, and never pass this
    planning-only enrichment to binding or verification.
    """
    metadata = parse_release_identity(name)
    if not metadata.source_family and "source_family" not in metadata.ambiguous_fields:
        source = _source_kind(name)
        if source:
            metadata = replace(metadata, source_family=source)
    return metadata


def _ordered_discovery_pairs(releases, query: ReferenceQuery) -> list:
    """Weighted ReferencePolicy discovery pairs; target VIDEO is authoritative."""
    pool = [rel for rel in (releases or []) if not query.target_download_url
            or getattr(rel, "download_url", None) != query.target_download_url]
    ordered, _ = release_scoring.discovery_pairs(pool, query, log_phase="discovery")
    return ordered


def _select_candidates_ranked(releases, query: ReferenceQuery) -> list:
    """Weighted ReferencePolicy discovery order; target VIDEO is authoritative."""
    return [release for release, _ in _ordered_discovery_pairs(releases, query)]


def _select_reference(releases, query: ReferenceQuery):
    """Pick the highest weighted discovery candidate."""
    ranked = _select_candidates_ranked(releases, query)
    return ranked[0] if ranked else None


class ExternalExactStrategy:
    """External reference acquisition across SubDL, SubSource, and OpenSubtitles.

    All providers are queried concurrently; their candidates are pooled and
    scored globally, so the fastest provider no longer wins by default. Series
    episode queries are enriched with a season-wide query only when Tier 1 has
    no ready source-compatible episode or exact-release candidate.
    """

    name = "external"
    # Signals to the orchestrator that ``resolve_with_provenance`` can validate
    # each candidate against the target subtitle and walk down its ranked list
    # instead of committing to the first download.
    validates_target = True

    def __init__(
        self,
        subdl_provider=None,
        subsource_provider=None,
        opensubtitles_provider=None,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        min_bytes: int = _MIN_REFERENCE_BYTES,
        cache: ReferenceDiskCache | None = None,
    ) -> None:
        self._subdl = subdl_provider
        self._subsource = subsource_provider
        self._opensubtitles = opensubtitles_provider
        self.timeout = timeout
        self.min_bytes = min_bytes
        self.cache = cache if cache is not None else ReferenceDiskCache(min_bytes=min_bytes)
        # Candidate releases already proven unusable for a target-scoped query.
        # Keyed by the full cache stem so one subtitle's rejections never leak
        # into another subtitle with a different cue layout. Bounded and
        # expiring: the strategy lives for the whole process, so a plain dict
        # grew with every subtitle ever synced, and one transient bad
        # validation excluded a candidate until restart.
        self._rejected: TTLCache[str, set[str]] = TTLCache(
            maxsize=_REJECTED_MEMO_MAX_STEMS, ttl=_REJECTED_MEMO_TTL_SECONDS
        )

    @staticmethod
    def _candidate_key(candidate) -> str:
        """Stable identity for a candidate release across provider fan-outs."""
        return str(getattr(candidate, "download_url", "") or getattr(candidate, "release_name", ""))

    def _is_rejected(self, stem: str, candidate) -> bool:
        return self._candidate_key(candidate) in self._rejected.get(stem, set())

    def _mark_rejected(self, stem: str, candidate) -> None:
        self._rejected.setdefault(stem, set()).add(self._candidate_key(candidate))

    async def resolve(self, query: ReferenceQuery) -> str | None:
        """Return the best reference subtitle (> ``min_bytes``) or ``None``."""
        resolved = await self.resolve_with_provenance(query)
        return resolved.text

    async def resolve_with_provenance(
        self,
        query: ReferenceQuery,
        *,
        update_validator: Callable[[str], bool] | None = None,
    ) -> ResolvedReference:
        """Query every provider, pool the candidates, download the best one.

        The weighted score is applied across all providers together, so the
        reference that best matches the video's release quality wins regardless
        of which provider responded first. Cache hits recover the persisted
        decision kind.

        ``update_validator`` (when supplied) is handed each downloaded
        reference; a candidate that fails it is skipped in favour of the
        next-ranked one, so a mislabeled season pack can no longer abort the
        whole sync while usable candidates are still available.
        """
        validate = update_validator
        cached = self.cache.get(query)
        if cached is not None:
            if validate is None or validate(cached.text or ""):
                return cached
            # The cached reference was proven against whichever target
            # subtitle wrote it, and that proof is not transferable: this
            # candidate can be a different cut, so the same reference now
            # fails. Evict it, otherwise every later request re-reads the same
            # unusable entry and re-runs the full provider fan-out only to
            # fail identically.
            logger.warning(
                "[reference] cached reference %r failed target cue-sanity; "
                "evicting and re-resolving",
                cached.candidate,
            )
            self.cache.delete(query)

        providers = [
            p for p in (self._subdl, self._subsource, self._opensubtitles) if p is not None
        ]
        if not providers:
            return ResolvedReference(None)
        logger.info(
            "[reference] resolving imdb=%s S%sE%s filename=%r across %s (timeout=%.1fs)",
            query.imdb_id,
            query.season,
            query.episode,
            query.target_filename,
            [self._provider_label(p) for p in providers],
            self.timeout,
        )

        candidates = await self._gather_candidates(providers, query)
        if not candidates:
            logger.warning(
                "[reference] no candidates from %s", [self._provider_label(p) for p in providers]
            )
            return ResolvedReference(None)

        provider_of = {id(rel): provider for rel, provider in candidates}
        ranked = _select_candidates_ranked([rel for rel, _ in candidates], query)
        if not ranked:
            logger.warning("[reference] aborting: no season/episode-matched reference")
            return ResolvedReference(None)

        result = ResolvedReference(None)
        winning_provider = None
        exhausted_providers: set[str] = set()

        max_attempts = 6
        try:
            from app.config import settings as _settings

            max_attempts = max(1, int(getattr(_settings, "REFERENCE_POOL_LIMIT", 6) or 6))
        except Exception:  # pragma: no cover - keep the legacy ceiling
            pass
        attempts = 0
        stem = query.cache_stem
        # References already downloaded during THIS request, as
        # (provider, name, text). Used for duplicate collapse and consensus,
        # both of which are measurements over data we already have.
        observed_references: list[tuple[str, str | None, str | None]] = []
        candidate_decision_kinds: dict[str, str] = {}
        # Cue-sanity outcome per release name, and the release families already
        # proven incompatible with this candidate. Both feed the target-bound
        # decision after the loop.
        cue_sanity_by_release: dict[str, bool] = {}
        family_rejected: set[str] = set()
        family = ""

        for cand in ranked:
            cand_provider = provider_of.get(id(cand))
            if cand_provider is None:
                continue

            # A candidate that already failed cue-sanity for this exact target
            # scope is not re-downloaded: the verdict is deterministic, so
            # retrying it burns a download and provider rate limit to reach the
            # same result.
            if self._is_rejected(stem, cand):
                continue

            provider_lbl = self._provider_label(cand_provider)
            if provider_lbl in exhausted_providers:
                continue

            # Quota protection: every OpenSubtitles download is metered against
            # the provider's download budget, and a speculative reference
            # evaluation can walk through many candidates before one validates.
            # Speculating on a metered download burns that allowance for a guess
            # we have no proof for, so it is only spent when OpenSubtitles has
            # already proven the file byte-exact via MovieHash. Text and series
            # reference sourcing relies on the unlimited SubSource/SubDL tier.
            if (
                not self._download_allowed(cand, cand_provider, query)
            ):
                logger.info(
                    "[reference] skipping non-hash OpenSubtitles candidate %r; "
                    "quota is reserved for MovieHash-exact matches",
                    getattr(cand, "release_name", "?"),
                )
                continue

            if hasattr(cand_provider, "is_breaker_open") and cand_provider.is_breaker_open():
                logger.info(
                    "[reference] provider %s circuit breaker is open; skipping candidate %r",
                    provider_lbl,
                    getattr(cand, "release_name", "?"),
                )
                exhausted_providers.add(provider_lbl)
                continue

            # One timing model advertised by several providers is one model.
            # Once a release family has been proven incompatible with this
            # candidate, a sibling from another provider will be too, so it is
            # not downloaded just to rediscover the same delta.
            family = _release_family_key(_release_name(cand))
            if family in family_rejected:
                logger.info(
                    "[reference] skipping %r: release family %r already proven "
                    "incompatible with this candidate",
                    getattr(cand, "release_name", "?"),
                    family,
                )
                continue

            attempts += 1
            cand_result = await self._download_candidate(cand, cand_provider, query)
            if cand_result.text:
                observed_references.append(
                    (
                        self._provider_label(cand_provider),
                        _release_name(cand),
                        cand_result.text,
                    )
                )
                candidate_decision_kinds[_release_name(cand)] = cand_result.kind
                if validate is not None and not validate(cand_result.text):
                    cue_sanity_by_release[_release_name(cand)] = False
                    family_rejected.add(family)
                    self._mark_rejected(stem, cand)
                    logger.warning(
                        "[reference] candidate %r (provider=%s, family=%s) failed target "
                        "cue-sanity; this reference does not share the candidate's timeline",
                        getattr(cand, "release_name", "?"),
                        provider_lbl,
                        family,
                    )
                else:
                    cue_sanity_by_release[_release_name(cand)] = True
                    result = cand_result
                    winning_provider = cand_provider
                    break
            else:
                # If download failed and tripped the breaker (e.g. quota 406 or
                # rate limit 429), mark this provider exhausted so all its
                # remaining candidates are skipped.
                if hasattr(cand_provider, "is_breaker_open") and cand_provider.is_breaker_open():
                    logger.info(
                        "[reference] provider %s tripped circuit breaker; skipping remaining candidates from this provider",
                        provider_lbl,
                    )
                    exhausted_providers.add(provider_lbl)

                logger.warning(
                    "[reference] candidate %r (provider=%s) failed download/decode; trying next candidate",
                    getattr(cand, "release_name", "?"),
                    provider_lbl,
                )
            if attempts >= max_attempts:
                logger.warning(
                    "[reference] reached maximum candidate download attempts (%d)", max_attempts
                )
                break

        text = result.text
        if text is None or winning_provider is None:
            logger.warning("[reference] all candidate attempts failed to produce a usable reference")
            return ResolvedReference(None)

        # --- TARGET-BOUND REFERENCE DECISION --------------------------------- #
        # The legacy loop above breaks on the first candidate that satisfies the
        # candidate-derived validator. That answer is now treated as a claim,
        # not a verdict: Step A re-derives the reference from the TARGET alone,
        # using the same compatibility machinery (MatchTier, hard filter) and the
        # same ReferenceHealth/ReferenceTrust the shadow selector already used.
        #
        # Step A has no candidate parameter. It can only confirm the legacy pick
        # or reject it, and rejection means "this reference does not belong to
        # the target as strongly as one already proven incompatible", so we fail
        # closed rather than resynchronize against a reference the target
        # evidence does not support.
        reference_decision = None
        # Step A only has something to correct when a candidate gate actually
        # ran. Without one, no candidate influenced the choice and the legacy
        # ranking is already target-derived, so it is left alone.
        if validate is not None:
            try:
                from app.services.sync.reference_v2 import (
                    ReferenceAttempt,
                    build_candidates,
                    decide_target_bound_reference,
                    release_family_key,
                    summarize_target_bound,
                )

                scored = build_candidates(query.target_filename, observed_references)
                logger.debug(
                    "[reference] decision inputs: observed=%d scored=%d cue_map=%s",
                    len(observed_references),
                    len(scored),
                    sorted(cue_sanity_by_release),
                )
                reference_decision = decide_target_bound_reference(
                    [
                        ReferenceAttempt(
                            reference_id=hashlib.sha256(
                                f"{(c.provider or '').lower()}|{c.release_name.lower()}".encode()
                            ).hexdigest()[:16],
                            release_name=c.release_name,
                            weighted_key=c.weighted_key,
                            provider=c.provider,
                            match_tier=c.match_tier,
                            hard_accepted=c.hard_accepted,
                            health_class=c.health_class,
                            family=release_family_key(c.release_name),
                            downloaded=True,
                            cue_sanity_ok=cue_sanity_by_release.get(c.release_name),
                        )
                        for c in scored
                    ]
                )
                legacy_winner = next(
                    (a for a in reference_decision.attempts if a.cue_sanity_ok is True), None
                )
                confirmed = (
                    reference_decision.selected is not None
                    and legacy_winner is not None
                    and reference_decision.selected.release_name == legacy_winner.release_name
                )
                logger.info(
                    "[reference] target-bound decision: outcome=%s confirmed_legacy=%s "
                    "downloads=%d unique_families=%d duplicates_avoided=%d suppressed=%d",
                    reference_decision.outcome,
                    confirmed,
                    reference_decision.downloads_used,
                    reference_decision.unique_families,
                    reference_decision.duplicate_families_avoided,
                    len(reference_decision.suppressed_as_circular),
                )
                for reason in reference_decision.reasons:
                    logger.info("[reference] decision reason: %s", reason)

                try:
                    logger.info(
                        "[reference] selection_summary=%s",
                        json.dumps(summarize_target_bound(reference_decision), sort_keys=True),
                    )
                except Exception as exc:  # pragma: no cover - telemetry is optional
                    logger.debug("[reference] summary unavailable: %s", exc)

                if not confirmed:
                    logger.warning(
                        "[reference] refusing reference: %s", reference_decision.outcome
                    )
                    return ResolvedReference(None)
            except Exception as exc:  # pragma: no cover - policy must not break sync
                logger.debug("[reference] target-bound decision unavailable: %s", exc)

        # Persist a sanitised copy so later switches for this episode are instant.
        from app.services.sync_service import sanitize_subtitle

        sanitized = sanitize_subtitle(text)
        if sanitized:
            self.cache.set(
                query,
                self._provider_label(winning_provider),
                sanitized,
                kind=result.kind,
                bluray_match=result.bluray_match,
                candidate=result.candidate,
            )
            text = sanitized
        self._remember_video_reference(query, result)
        logger.info("[reference] selected reference: %d bytes", len(text.encode("utf-8")))

        # Reference trust is measured here and attached to the result. It does
        # NOT change which reference was chosen: the winner is decided above,
        # unchanged. It exists so a later layer can see how trustworthy the
        # reference actually was, and so telemetry can be compared against the
        # existing policy.
        assessment = None
        try:
            from app.services.sync.reference import assess_reference, detect_duplicate_groups

            groups = detect_duplicate_groups(observed_references)
            assessment = assess_reference(
                text,
                provider=self._provider_label(winning_provider),
                decision_kind=result.kind,
                from_cache=False,
                cache_verified=False,
                groups=groups,
            )
            logger.info("[reference] trust assessment: %s", assessment.explain())
        except Exception as exc:  # pragma: no cover - measurement must not break sync
            logger.debug("[reference] trust assessment unavailable: %s", exc)

        # --- SHADOW reference selection v2 -------------------------------- #
        # Runs beside the existing policy, purely to measure disagreement. The
        # `text` returned above is the LEGACY reference and is what the sync
        # engine receives; the shadow pick is recorded for analysis and is never
        # substituted into the response. `ReferenceSelection` is local and goes
        # no further than the telemetry below.
        #
        # The legacy loop above breaks on the first candidate that passes, so it
        # usually materializes exactly one reference and the shadow has nothing
        # to choose between. That is a property of the legacy policy, and it
        # would show up as a flattering "agreement" rate. To measure the policy
        # rather than the early break, the shadow is given a small bounded pool
        # drawn from the SAME ranked candidates. It never runs alass, never
        # re-ranks user-facing subtitles, and never feeds back into the choice
        # above.
        shadow_comparison = None
        shadow_pool = None
        shadow_switch_labels: list[str] = []
        try:
            from app.config import settings as app_settings
            from app.services.sync.audit import AUDIT_LOG as sync_audit
            from app.services.sync.reference_v2 import (
                COMPARISON_MEANINGFUL,
                build_shadow_pool,
                classify_shadow_switch,
                compare_selections,
                select_reference_v2,
            )
            from app.services.sync.reference_v2 import (
                _identity as _reference_identity,
            )
            from app.services.sync.shadow_expansion import (
                ExpansionTelemetry,
                choose_expansion_candidate,
                classify_expansion,
                resolve_fetch_budget,
            )

            pool_limit = int(getattr(app_settings, "REFERENCE_SHADOW_POOL_LIMIT", 4) or 0)
            fetch_limit = int(
                getattr(app_settings, "REFERENCE_SHADOW_POOL_FETCH_LIMIT", 0) or 0
            )

            # Payloads already in hand for this request. Nothing is re-fetched
            # to populate the pool; extra fetches are opt-in and metered below.
            payloads: dict[tuple[str, str], str] = {}
            for provider, name, ref_text in observed_references:
                if ref_text and name:
                    payloads[(provider, name)] = ref_text

            discovered = [
                (self._provider_label(provider_of[id(c)]), _release_name(c))
                for c in ranked
                if id(c) in provider_of
            ]

            # --- experimental pool expansion (measurement only) ------------- #
            # Everything above is unchanged production behaviour. What follows
            # runs only when the audit log is actually recording, the operator
            # configured a budget, the target is fingerprinted, and the existing
            # pool cannot support a comparison. Its only output is a payload for
            # the shadow pool plus telemetry; the legacy `text` is never touched.
            expansion = ExpansionTelemetry()
            pool_before = build_shadow_pool(
                query.target_filename if query else None,
                discovered,
                payloads,
                limit=pool_limit,
            )
            budget = resolve_fetch_budget(
                configured_limit=fetch_limit,
                audit_enabled=bool(sync_audit.enabled),
                target_filename=query.target_filename if query else None,
                comparison_class=pool_before.comparison_class,
            )
            expansion.eligible = budget.allowed > 0
            expansion.refusal = budget.reason
            expansion.pool_size_before = pool_before.pool_size
            expansion.independent_groups_before = pool_before.independent_groups
            expansion.meaningful_before = (
                pool_before.comparison_class == COMPARISON_MEANINGFUL
            )

            if budget.allowed > 0:
                choice = choose_expansion_candidate(
                    query.target_filename if query else None,
                    discovered,
                    payloads,
                    rejected=[
                        (self._provider_label(provider_of[id(c)]), _release_name(c))
                        for c in ranked
                        if id(c) in provider_of and self._is_rejected(stem, c)
                    ],
                )
                if choice is not None:
                    cand = next(
                        (
                            c
                            for c in ranked
                            if _release_name(c) == choice.release_name
                            and id(c) in provider_of
                            and self._provider_label(provider_of[id(c)]) == choice.provider
                        ),
                        None,
                    )
                    cand_provider = provider_of.get(id(cand)) if cand is not None else None
                    if cand is not None and cand_provider is not None:
                        expansion.attempts = 1
                        expansion.candidate_provider = choice.provider
                        try:
                            from app.services.subtitle_matcher import (
                                calculate_compatibility,
                            )

                            if query.target_filename:
                                compat = calculate_compatibility(
                                    extract_metadata(query.target_filename),
                                    choice.release_name,
                                )
                                expansion.candidate_match_tier = compat.match_tier.name
                                expansion.candidate_source_class = str(
                                    (extract_metadata(choice.release_name) or {}).get("source")
                                    or "unknown"
                                )
                        except Exception:
                            expansion.candidate_match_tier = None

                        started = time.monotonic()
                        try:
                            fetched = await self._download_candidate(
                                cand, cand_provider, query
                            )
                        except Exception:
                            fetched = None
                        elapsed_ms = round((time.monotonic() - started) * 1000, 3)
                        expansion.latency_ms = elapsed_ms
                        expansion.network_fetches = 1
                        if fetched is not None and fetched.text:
                            payloads[choice.key] = fetched.text
                            expansion.successes = 1
                            expansion.bytes = len(fetched.text.encode("utf-8"))
                            expansion.reasons.append(choice.reason)
                        else:
                            expansion.failures = 1
                            expansion.reasons.append("extra fetch produced no payload")

            shadow_pool = build_shadow_pool(
                query.target_filename if query else None,
                discovered,
                payloads,
                limit=pool_limit,
            )

            selection = select_reference_v2(
                query.target_filename if query else None, shadow_pool.references
            )
            # What the extra fetch did to the evidence, judged after the fact
            # from the real timing grids rather than assumed from the source.
            expansion.pool_size_after = shadow_pool.pool_size
            expansion.independent_groups_after = shadow_pool.independent_groups
            expansion.meaningful_after = (
                shadow_pool.comparison_class == COMPARISON_MEANINGFUL
            )
            expansion.outcome = classify_expansion(
                eligible=expansion.eligible,
                refusal=expansion.refusal,
                attempted=expansion.attempts > 0,
                succeeded=expansion.successes > 0,
                pool_size_before=expansion.pool_size_before,
                groups_before=expansion.independent_groups_before,
                pool_size_after=expansion.pool_size_after,
                groups_after=expansion.independent_groups_after,
                meaningful_before=expansion.meaningful_before,
                meaningful_after=expansion.meaningful_after,
            )
            if expansion.successes:
                expansion.duplicate_timing_grid = (
                    expansion.independent_groups_after
                    == expansion.independent_groups_before
                )
            # Same identity function on both sides, otherwise every request
            # would look like a disagreement.
            legacy_identity = _reference_identity(
                self._provider_label(winning_provider), result.candidate, text
            )
            shadow_comparison = compare_selections(
                legacy_identity,
                selection,
                legacy_trust=assessment.trust.value if assessment else None,
                legacy_health=(
                    "healthy" if (assessment and assessment.health.healthy) else "degraded"
                ),
            )
            if shadow_comparison.changed:
                # Looked up in all_candidates, not `ranked`: a legacy pick that
                # is itself ineligible (credits-only) is the interesting case,
                # and `ranked` would hide it.
                legacy_candidate = next(
                    (c for c in selection.all_candidates if c.subtitle_id == legacy_identity),
                    None,
                )
                shadow_switch_labels = classify_shadow_switch(
                    legacy_candidate, selection.chosen
                )
                shadow_comparison.reasons.extend(shadow_switch_labels)
                if shadow_pool is not None and shadow_pool.comparison_class != "MEANINGFUL_COMPARISON":
                    # A difference observed when no real alternative existed is
                    # not evidence about the policies.
                    logger.info(
                        "[reference-shadow] differs but NOT a meaningful comparison: %s",
                        shadow_pool.comparison_class,
                    )
                logger.info(
                    "[reference-shadow] policy differs: %s",
                    "; ".join(shadow_comparison.reasons),
                )
            else:
                logger.info(
                    "[reference-shadow] policies agree on %s (%s)",
                    shadow_comparison.legacy_id,
                    shadow_pool.comparison_class if shadow_pool else "unknown",
                )
        except Exception as exc:  # pragma: no cover - shadow must never break sync
            logger.debug("[reference-shadow] unavailable: %s", exc)

        return ResolvedReference(
            text,
            kind=result.kind,
            bluray_match=result.bluray_match,
            candidate=result.candidate,
            provider=result.provider,
            member_season=result.member_season,
            member_episode=result.member_episode,
            weighted_score=result.weighted_score,
            weighted_score_without_hash=result.weighted_score_without_hash,
            source_download_digest=result.source_download_digest,
            from_video_cache=result.from_video_cache,
            reference_trust=assessment.trust.value if assessment else None,
            reference_consensus=assessment.consensus_score if assessment else None,
            reference_independent_sources=assessment.independent_sources if assessment else 0,
            reference_failure=assessment.failure.value if assessment and assessment.failure else None,
            reference_reasons=list(assessment.reasons) if assessment else [],
            shadow_reference_id=shadow_comparison.shadow_id if shadow_comparison else None,
            shadow_changed=bool(shadow_comparison.changed) if shadow_comparison else False,
            shadow_reasons=list(shadow_comparison.reasons) if shadow_comparison else [],
            shadow_trust=shadow_comparison.shadow_trust if shadow_comparison else None,
            shadow_health=shadow_comparison.shadow_health if shadow_comparison else None,
            shadow_independent_groups=(
                shadow_comparison.independent_groups if shadow_comparison else 0
            ),
            shadow_pool_size=shadow_pool.pool_size if shadow_pool else 0,
            shadow_pool_limit=shadow_pool.pool_limit if shadow_pool else 0,
            shadow_pool_materialized=bool(shadow_pool and shadow_pool.references),
            shadow_pool_truncated=bool(shadow_pool and shadow_pool.truncated),
            shadow_pool_limited_by_payloads=bool(
                shadow_pool and shadow_pool.limited_by_available_payloads
            ),
            shadow_pool_independent_groups=(
                shadow_pool.independent_groups if shadow_pool else 0
            ),
            shadow_comparison_class=shadow_pool.comparison_class if shadow_pool else None,
            shadow_additional_fetches=expansion.successes if expansion else 0,
            shadow_switch_labels=shadow_switch_labels,
            shadow_expansion_eligible=bool(expansion.eligible) if expansion else False,
            shadow_expansion_refusal=expansion.refusal if expansion else None,
            shadow_expansion_outcome=expansion.outcome if expansion else None,
            shadow_extra_fetch_attempts=expansion.attempts if expansion else 0,
            shadow_extra_fetch_successes=expansion.successes if expansion else 0,
            shadow_extra_fetch_failures=expansion.failures if expansion else 0,
            shadow_extra_fetch_network_fetches=(
                expansion.network_fetches if expansion else 0
            ),
            shadow_extra_fetch_bytes=expansion.bytes if expansion else None,
            shadow_extra_fetch_latency_ms=expansion.latency_ms if expansion else None,
            shadow_pool_size_before=expansion.pool_size_before if expansion else 0,
            shadow_independent_groups_before=(
                expansion.independent_groups_before if expansion else 0
            ),
            shadow_pool_size_after=expansion.pool_size_after if expansion else 0,
            shadow_independent_groups_after=(
                expansion.independent_groups_after if expansion else 0
            ),
            shadow_meaningful_before=expansion.meaningful_before if expansion else False,
            shadow_meaningful_after=expansion.meaningful_after if expansion else False,
            shadow_extra_duplicate_timing_grid=(
                expansion.duplicate_timing_grid if expansion else None
            ),
            shadow_extra_candidate_provider=(
                expansion.candidate_provider if expansion else None
            ),
            shadow_extra_candidate_match_tier=(
                expansion.candidate_match_tier if expansion else None
            ),
            shadow_extra_candidate_source_class=(
                expansion.candidate_source_class if expansion else None
            ),
        )

    def _provider_label(self, provider) -> str:
        """Stable provider label for logs and the reference cache filename."""
        if provider is self._subdl:
            return "subdl"
        if provider is self._subsource:
            return "subsource"
        if provider is self._opensubtitles:
            return "opensubtitles"
        return getattr(provider, "name", type(provider).__name__)

    async def _gather_candidates(self, providers, query: ReferenceQuery) -> list[tuple]:
        """Pool ``(release, provider)`` candidates from every provider.

        Each provider owns a single total deadline. Tier 2 is opportunistic:
        once Tier 1 has completed, a slow or failed season catalog must not
        discard it. Display-discovered candidates for the same video and
        authorization context are merged as hints, but they still go through
        download, extraction, proof, binding and verification.
        """

        async def one(provider) -> list[tuple]:
            name = self._provider_label(provider)
            deadline = time.monotonic() + self.timeout
            search = self._search_provider
            try:
                accepts_deadline = "deadline" in inspect.signature(search).parameters
            except (TypeError, ValueError):  # pragma: no cover - exotic callable
                accepts_deadline = False
            try:
                if accepts_deadline:
                    # _search_tiers owns the provider deadline and guarantees
                    # that a completed Tier 1 is returned before it expires.
                    releases = await search(provider, query, deadline=deadline)
                else:
                    # Compatibility with test/legacy overrides that predate the
                    # explicit deadline. The outer guard still bounds them.
                    releases = await asyncio.wait_for(
                        search(provider, query),
                        self.timeout + _PROVIDER_HARD_TIMEOUT_GRACE_SECONDS,
                    )
            except Exception as exc:  # noqa: BLE001 - one provider must not sink the rest
                logger.warning("[reference] %s search failed: %s", name, type(exc).__name__)
                return []
            logger.info("[reference] %s returned %d candidate(s)", name, len(releases or []))
            return [(rel, provider) for rel in (releases or [])]

        batches = await asyncio.gather(*(one(p) for p in providers), return_exceptions=True)
        merged: list[tuple] = []
        for batch in batches:
            if isinstance(batch, list):
                merged.extend(batch)

        # Display already paid for provider discovery on this video. Reuse those
        # candidates only as hints for the same video + credential context; the
        # normal planning/download/proof path decides whether any is usable.
        api_keys = getattr(query, "api_keys", {}) or {}
        hint_key = discovery_hints.video_scope_key(
            imdb_id=getattr(query, "imdb_id", None),
            media_type=getattr(query, "media_type", None),
            season=getattr(query, "season", None),
            episode=getattr(query, "episode", None),
            target_filename=getattr(query, "target_filename", None),
            video_hash=getattr(query, "video_hash", None),
            video_size=getattr(query, "video_size", None),
        )
        hints = discovery_hints.lookup(
            hint_key,
            discovery_hints.authorization_digest(
                subdl_key=api_keys.get("subdl"),
                subsource_key=api_keys.get("subsource"),
                opensubtitles_key=api_keys.get("opensubtitles"),
            ),
        )
        if hints:
            by_label = {self._provider_label(provider): provider for provider in providers}
            reused = 0
            for hint in hints:
                provider = by_label.get(str(getattr(hint, "provider", "")).lower())
                if provider is None:
                    continue
                merged.append((hint, provider))
                reused += 1
            if reused:
                logger.info("[reference] reused %d Display candidate hint(s)", reused)

        # De-duplicate identical downloads returned by more than one provider.
        seen: set[str] = set()
        deduped: list[tuple] = []
        for rel, provider in merged:
            url = getattr(rel, "download_url", None)
            if url and url in seen:
                continue
            if url:
                seen.add(url)
            deduped.append((rel, provider))
        return deduped

    async def _search_provider(self, provider, query: ReferenceQuery, *, deadline: float | None = None) -> list:
        """Route a provider to its search implementation (Identity, not name)."""
        if provider is self._subdl:
            return await self._search_subdl(query, deadline=deadline)
        if provider is self._subsource:
            return await self._search_subsource(query, deadline=deadline)
        if provider is self._opensubtitles:
            return await self._search_opensubtitles(query, deadline=deadline)
        raise ValueError(f"unknown provider {provider!r}")

    def _provider_api_key(self, provider, query: ReferenceQuery) -> str | None:
        if provider is self._subdl:
            return query.api_keys.get("subdl")
        if provider is self._subsource:
            return query.api_keys.get("subsource")
        if provider is self._opensubtitles:
            return query.api_keys.get("opensubtitles")
        return None

    async def resolve_all_with_provenance(
        self,
        query: ReferenceQuery,
        *,
        update_validator: Callable[[str], bool] | None = None,
    ) -> list[ResolvedReference]:
        """Download several DISTINCT usable references, best-ranked first.

        Why this exists: `resolve_with_provenance` returns exactly one
        reference. The orchestrator's retry loop advances between *strategies*,
        not between candidates, so when only SubDL is configured and no
        MovieHash exists there is a single strategy able to yield a reference
        and therefore a single ALASS attempt. If that one candidate is the
        wrong cut, the request ends unsynchronized even though other candidate
        releases for the same film were available all along.

        Production evidence (Whiplash, tt2582802): one cached `subdl_edition`
        reference with 922 cues was aligned against an 825-cue target (1.12x
        surplus), giving movement MAD 9356ms and residual matching that
        DEGRADED from 33 to 130 unmatched cues at a 5s tolerance. The verifier
        rejected it correctly -- but no second candidate was ever tried.

        Deduplication is by SUBTITLE CONTENT SHA-256, never by filename: two
        providers routinely publish the same timing model under different
        release names, and treating those as distinct candidates would spend
        two downloads to learn one fact.

        Ranking beyond the existing provider/release signals: cue-count
        distance and first-cue proximity against the target are used to ORDER
        candidates, because a reference cut from a different edit has a
        different cue count and a different opening dialogue. These are ranking
        signals ONLY. They are never treated as proof that two files are the
        same cut, never establish identity, and never bypass the verifier --
        every returned candidate still has to survive the existing verification
        gates on its own merits.
        """
        out: list[ResolvedReference] = []
        async for result in self.resolve_candidates_lazy(query, update_validator=update_validator):
            out.append(result)
        return out

    async def _ranked_release_plan(
        self, query: ReferenceQuery
    ) -> tuple[list[tuple[Any, Any]], list[tuple[Any, Any]], list[dict[str, Any]], dict[int, Any], int]:
        """Gather and rank releases WITHOUT downloading anything.

        Returns ``(ordered, ineligible, targeted_excluded, provider_of, limit)``:
        ReferencePolicy-ordered release/result pairs, policy-ineligible pairs,
        releases excluded because they are the target download, a map from
        ``id(release)`` to its provider, and the pool limit. Search and ranking
        are cheap metadata operations; every download happens later, on demand,
        so a request that verifies on the first candidate never pays for the
        rest of the pool.
        """
        providers = [
            p for p in (self._subdl, self._subsource, self._opensubtitles) if p is not None
        ]
        if not providers:
            return [], [], [], {}, 0

        candidates = await self._gather_candidates(providers, query)
        if not candidates:
            logger.warning(
                "[reference] no candidates from %s",
                [self._provider_label(p) for p in providers],
            )
            return [], [], [], {}, 0

        provider_of = {id(rel): provider for rel, provider in candidates}
        pool = [rel for rel, _ in candidates]
        targeted_excluded = [
            {
                "provider": self._provider_label(provider),
                "source": download_source_digest(
                    self._provider_label(provider), getattr(rel, "download_url", None)
                ),
                "reason": "same_target_download",
            }
            for rel, provider in candidates
            if query.target_download_url
            and getattr(rel, "download_url", None) == query.target_download_url
        ]
        ordered_pairs, ineligible_pairs = release_scoring.discovery_pairs(pool, query, log_phase="discovery")
        if not ordered_pairs:
            logger.warning("[reference] resolve_all: no season/episode-matched reference")
            return [], [], targeted_excluded, provider_of, 0

        try:
            from app.config import settings as _settings

            limit = max(1, int(getattr(_settings, "REFERENCE_POOL_LIMIT", 6) or 6))
        except Exception:  # pragma: no cover
            limit = 6
        return ordered_pairs, ineligible_pairs, targeted_excluded, provider_of, limit

    def _pool_limit(self) -> int:
        try:
            from app.config import settings as _settings

            return max(1, int(getattr(_settings, "REFERENCE_POOL_LIMIT", 6) or 6))
        except Exception:  # pragma: no cover
            return 6

    def _download_allowed(self, candidate: Any, provider: Any, query: ReferenceQuery) -> bool:
        """Apply the existing quota policy before allocating a bounded plan slot.

        Non-hash OpenSubtitles candidates cannot be downloaded. They must not
        crowd out downloadable references. Rechecked at download time; neither
        ranking nor reference trust is changed.
        """
        return self._quota_block_reason(candidate, provider, query) is None

    def _quota_block_reason(self, candidate: Any, provider: Any, query: ReferenceQuery) -> str | None:
        """Default production reason why a candidate cannot occupy a slot."""
        if not release_scoring.reference_resolvable(candidate, query):
            return "reference_not_resolvable"
        if provider is self._opensubtitles and not any(
            result.verified_hash for result in release_scoring.matches([candidate], query)
        ):
            return "opensubtitles_non_hash_quota"
        return None

    def _download_block_reason(self, candidate: Any, provider: Any, query: ReferenceQuery) -> str | None:
        """Distinguish why a candidate cannot occupy a bounded download slot."""
        if self._download_allowed(candidate, provider, query):
            return None
        return self._quota_block_reason(candidate, provider, query) or "download_not_allowed"

    @staticmethod
    def _provider_breaker_open(provider: Any) -> bool:
        """Snapshot a provider's operational restriction without network use."""
        checker = getattr(provider, "is_breaker_open", None)
        if not callable(checker):
            return False
        try:
            return bool(checker())
        except Exception:
            return False

    def _plan_row(
        self,
        *,
        result,
        provider: Any,
        role: str,
        status: str,
        planned_position: int | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        provider_label = self._provider_label(provider)
        return {
            "safe_id": release_scoring.digest(result.candidate.identity)[:16],
            "provider": provider_label,
            "language": result.candidate.language,
            "score": result.ranking_score,
            "metadata_score": result.score,
            "metadata_percentage": result.percentage,
            "source_priority_points": result.source_priority_points,
            "score_without_hash": result.score_without_hash,
            "percentage": result.ranking_percentage,
            "discovery_eligible": result.discovery_eligible,
            "ready_reference_eligible": result.ready_reference_eligible,
            "unresolved_pack": result.candidate.unresolved_pack,
            "evidence": _plan_evidence_states(result),
            "rejections": list(result.rejection_reasons),
            "readiness": list(result.readiness_reasons),
            "breaker_open": self._provider_breaker_open(provider),
            "role": role,
            "status": status,
            "planned_position": planned_position,
            "reason": reason,
        }

    def _select_planned_attempts(
        self,
        ordered_pairs: list[tuple[Any, Any]],
        ineligible_pairs: list[tuple[Any, Any]],
        targeted_excluded: list[dict[str, Any]],
        provider_of: dict[int, Any],
        limit: int,
        query: ReferenceQuery,
    ) -> tuple[list[tuple[Any, Any]], dict[str, Any]]:
        """Fill at most ``limit`` actual download slots in weighted order.

        The global ReferencePolicy order is never re-sorted. The only
        intentional departure is a single reserved exploration opportunity: when
        ordinary candidates would otherwise saturate the budget, the lowest
        source-mismatching slot can be given to the highest-ranked unresolved
        pack that already shares the target's release medium. Extraction and
        readiness are still earned afterward; no points or rank are invented.
        """
        stem = query.cache_stem
        known_sources = {digest for digest in getattr(query, "known_source_digests", ()) or () if digest}
        ordinary: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        selected_ids: set[int] = set()
        seen_sources: set[str] = set()

        for candidate, result in ordered_pairs:
            if len(ordinary) >= limit:
                break
            provider = provider_of.get(id(candidate))
            if provider is None:
                excluded.append(self._plan_row(
                    result=result, provider=None, role="ordinary",
                    status="excluded", reason="provider_missing",
                ))
                continue
            if id(candidate) in selected_ids:
                excluded.append(self._plan_row(
                    result=result, provider=provider, role="ordinary",
                    status="excluded", reason="duplicate_release",
                ))
                continue
            if self._is_rejected(stem, candidate):
                excluded.append(self._plan_row(
                    result=result, provider=provider, role="ordinary",
                    status="excluded", reason="already_rejected",
                ))
                continue
            source = download_source_digest(self._provider_label(provider), getattr(candidate, "download_url", None))
            if source is not None and source in known_sources:
                excluded.append(self._plan_row(
                    result=result, provider=provider, role="ordinary",
                    status="excluded", reason="duplicate_known_source",
                ))
                continue
            if source is not None and source in seen_sources:
                excluded.append(self._plan_row(
                    result=result, provider=provider, role="ordinary",
                    status="excluded", reason="duplicate_download_source",
                ))
                continue
            block = self._download_block_reason(candidate, provider, query)
            if block is not None:
                excluded.append(self._plan_row(
                    result=result, provider=provider, role="ordinary",
                    status="excluded", reason=block,
                ))
                continue
            if source is not None:
                seen_sources.add(source)
            selected_ids.add(id(candidate))
            ordinary.append({"candidate": candidate, "provider": provider, "result": result})

        normal_capacity = max(0, limit - _REFERENCE_EXPLORATION_SLOTS)
        base = ordinary[:normal_capacity]
        plan_entries = list(base)
        explorer: dict[str, Any] | None = None

        if len(ordinary) >= limit and limit > 0:
            qualifying_normal = any(_exploration_candidate(entry["result"]) for entry in ordinary)
            if not qualifying_normal:
                for candidate, result in ordered_pairs:
                    if id(candidate) in selected_ids:
                        continue
                    if not _exploration_candidate(result):
                        continue
                    provider = provider_of.get(id(candidate))
                    if provider is None or self._is_rejected(stem, candidate):
                        continue
                    source = download_source_digest(
                        self._provider_label(provider), getattr(candidate, "download_url", None)
                    )
                    if source is not None and (source in known_sources or source in seen_sources):
                        continue
                    if self._download_block_reason(candidate, provider, query) is not None:
                        continue
                    explorer = {"candidate": candidate, "provider": provider, "result": result}
                    break
            bottom = base[-1] if base else None
            if (
                explorer is not None
                and bottom is not None
                and not _candidate_source_matches(bottom["result"])
            ):
                source = download_source_digest(
                    self._provider_label(explorer["provider"]),
                    getattr(explorer["candidate"], "download_url", None),
                )
                if source is not None:
                    seen_sources.add(source)
                selected_ids.add(id(explorer["candidate"]))
                plan_entries.append({**explorer, "exploration": True})
                excluded.append(self._plan_row(
                    result=ordinary[-1]["result"],
                    provider=ordinary[-1]["provider"],
                    role="ordinary",
                    status="excluded",
                    reason="exploration_reserve",
                ))
            elif explorer is not None and bottom is None and not qualifying_normal:
                plan_entries.append({**explorer, "exploration": True})
        if explorer is None:
            # No qualifying exploration candidate: retain the ordinary budget
            # exactly as before. Reservation must not shrink a valid plan.
            plan_entries = list(ordinary)

        plan = [(entry["candidate"], entry["provider"]) for entry in plan_entries]
        planned_ids = {id(candidate) for candidate, _ in plan}
        rows: list[dict[str, Any]] = []
        for position, entry in enumerate(plan_entries, start=1):
            rows.append(self._plan_row(
                result=entry["result"],
                provider=entry["provider"],
                role="exploration" if entry.get("exploration") else "ordinary",
                status="planned",
                planned_position=position,
                reason="exploration_slot" if entry.get("exploration") else None,
            ))

        below_limit_rows = 0
        for candidate, result in ordered_pairs:
            if id(candidate) in planned_ids:
                continue
            if any(row["safe_id"] == release_scoring.digest(result.candidate.identity)[:16] for row in excluded):
                continue
            below_limit_rows += 1
            if below_limit_rows <= _REFERENCE_PLAN_REPORT_EXCLUDED_LIMIT:
                excluded.append(self._plan_row(
                    result=result,
                    provider=provider_of.get(id(candidate)),
                    role="ordinary",
                    status="excluded",
                    reason="below_pool_limit",
                ))

        unranked_rows: list[dict[str, Any]] = []
        seen_unranked: set[str] = set()
        for candidate, result in ineligible_pairs:
            safe_id = release_scoring.digest(result.candidate.identity)[:16]
            if safe_id in seen_unranked:
                continue
            seen_unranked.add(safe_id)
            if len(unranked_rows) >= _REFERENCE_PLAN_REPORT_UNRANKED_LIMIT:
                break
            unranked_rows.append(self._plan_row(
                result=result,
                provider=provider_of.get(id(candidate)),
                role="unranked",
                status="excluded",
                reason="reference_policy_ineligible",
            ))

        excluded_total = len(excluded)
        report = {
            "budget_slots": limit,
            "normal_slots": len(base),
            "exploration_slots": 1 if any(entry.get("exploration") for entry in plan_entries) else 0,
            "ranked_releases": len(ordered_pairs),
            "unranked_releases": len(ineligible_pairs),
            "target_download_excluded": len(targeted_excluded),
            "planned_count": len(rows),
            "excluded_count": excluded_total,
            "planned": rows[:_REFERENCE_PLAN_REPORT_SELECTED_LIMIT],
            "excluded": excluded[:_REFERENCE_PLAN_REPORT_SELECTED_LIMIT],
            "unranked_ineligible": unranked_rows,
        }
        return plan, report

    def _log_reference_plan(self, report: dict[str, Any]) -> None:
        """Emit one compact, privacy-safe line describing the planned attempts."""
        try:
            logger.info("[reference-plan] %s", json.dumps(report, sort_keys=True))
        except Exception:
            pass  # Explainability cannot affect subtitle delivery.

    async def plan_candidates(
        self, query: ReferenceQuery
    ) -> list[tuple[Any, Any]]:
        """Ranked ``(release, provider)`` pairs with NO downloads performed.

        The orchestrator wraps each pair in a lazy adapter so the existing
        verification loop downloads a candidate only when it actually reaches
        it. A request that verifies on the first candidate therefore performs
        exactly one download instead of filling the whole pool upfront.
        Previously-rejected releases (for this query's cache stem) are filtered
        here since that check needs no network; breaker and quota state is
        re-checked at download time, when it is freshest.
        """
        ordered_pairs, ineligible_pairs, targeted_excluded, provider_of, limit = await self._ranked_release_plan(query)
        plan, report = self._select_planned_attempts(
            ordered_pairs,
            ineligible_pairs,
            targeted_excluded,
            provider_of,
            limit,
            query,
        )
        self._log_reference_plan(report)
        return plan

    async def download_planned_candidate(
        self,
        cand: Any,
        cand_provider: Any,
        query: ReferenceQuery,
        validate: Callable[[str], bool] | None,
        shared: dict[str, Any],
        plan_index: int | None = None,
    ) -> ResolvedReference:
        """Download ONE planned candidate, honouring pool-wide guards.

        ``shared`` carries ``{"downloaded": int, "seen": set[str]}`` across the
        candidates of one request so pacing, the pool limit accounting, and
        content-dedup behave exactly as they did under eager resolution.
        Returns ``ResolvedReference(None)`` when this candidate is unusable, in
        which case the caller simply advances to the next planned candidate.
        """
        logger.info(
            "[reference] candidate reached provider=%s tier=%d plan_index=%s",
            self._provider_label(cand_provider),
            reference_tier(query.target_filename, cand),
            plan_index if plan_index is not None else "unplanned",
        )
        if hasattr(cand_provider, "is_breaker_open") and cand_provider.is_breaker_open():
            logger.warning(
                "[reference] candidate skipped: provider=%s reason=provider_circuit_breaker_open",
                self._provider_label(cand_provider),
            )
            return ResolvedReference(None, reference_failure="provider_circuit_breaker_open")
        # Preserve the existing quota rule exactly: a non-hash OpenSubtitles
        # download consumes provider quota, so it stays
        # reserved for MovieHash-exact references.
        block = self._download_block_reason(cand, cand_provider, query)
        if block is not None:
            logger.info("[reference] download skipped reason=%s", block)
            return ResolvedReference(None, reference_failure=(
                "candidate_rejected" if block == "reference_not_resolvable" else "download_failure"
            ))

        # Safe, bounded correlation of actual execution with weighted planning.
        release_scoring.log_attempt(cand, query)

        # Pace consecutive downloads from the SAME provider so one request
        # cannot burst it into a 429 / circuit break. Providers are independent
        # rate-limit domains, so the pacing state is per provider: two different
        # providers may transfer concurrently (which is the latency win), while
        # downloads from one provider still start at least 0.6s apart. The lock
        # also makes the start-time accounting safe under bounded prefetch.
        provider_key = self._provider_label(cand_provider) or "unknown"
        pacing_locks = shared.setdefault("pacing_locks", {})
        pacing_last = shared.setdefault("pacing_last", {})
        pacing_lock = pacing_locks.get(provider_key)
        if pacing_lock is None:
            pacing_lock = asyncio.Lock()
            pacing_locks[provider_key] = pacing_lock
        async with pacing_lock:
            now = time.monotonic()
            previous = pacing_last.get(provider_key)
            if previous is not None:
                delay = _REFERENCE_DOWNLOAD_MIN_INTERVAL_SECONDS - (now - previous)
                if delay > 0:
                    await asyncio.sleep(delay)
            pacing_last[provider_key] = time.monotonic()
        shared["downloaded"] = int(shared.get("downloaded") or 0) + 1

        result = await self._download_candidate(cand, cand_provider, query)
        text = result.text
        if not text:
            return result
        timing = current_sync_timing.get()
        if timing is not None:
            timing.references_downloaded += 1
            if result.reference_failure:
                timing.note_reference_failure(result.reference_failure)
        if validate is not None and not validate(text):
            # The validator only returns a bool, so re-measure here purely
            # to attribute the verdict to a NAMED candidate. Without this
            # the log cannot say which release was dropped or why.
            # Name the candidate so the orchestrator's adjacent
            # cue-sanity diagnostic (which carries the measured delta and
            # first-dialogue offsets) can be attributed to a release.
            logger.info(
                "[reference] resolve_all: candidate SKIPPED name=%r lang=%s cues=%d",
                getattr(cand, "release_name", "?"),
                getattr(cand, "lang", "?"),
                text.count(" --> "),
            )
            if timing is not None:
                timing.note_reference_failure("candidate_rejected")
            return ResolvedReference(None, reference_failure="candidate_rejected")

        seen: set[str] = shared.setdefault("seen", set())
        digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
        if digest in seen:
            if timing is not None:
                timing.references_deduplicated += 1
            logger.info(
                "[reference] resolve_all: candidate %r has duplicate content (%s); skipping",
                getattr(cand, "release_name", "?"),
                digest[:12],
            )
            return ResolvedReference(None)
        seen.add(digest)
        self._remember_video_reference(query, result)
        return result

    def _remember_video_reference(self, query: ReferenceQuery, result: ResolvedReference) -> None:
        """Persist a request-validated reference artifact against the target VIDEO.

        Callers invoke this only after the candidate has passed both weighted
        extraction/readiness and request-specific cue validation, so a wrong-cut
        download is not promoted merely because it decoded successfully.

        Which file was downloaded is a property of the video, not of the
        subtitle the user picked. Storing it here is what lets the next
        subtitle for the same episode reuse the artifact instead of
        re-downloading the same archive from a rate-limited provider.

        The cache refuses the write unless a specific episode member was proven
        (see ``set_video_scoped``), so an unresolved season pack is never
        promoted to a validated episode reference. An artifact that is identical
        to the current target is still retained: identity makes it unusable for
        this request, but it may be legitimate independent evidence for a
        different subtitle. Every later use still applies the independent-content
        check before verification.
        """
        if not result.text or query is None:
            return
        try:
            from app.services.sync_service import sanitize_subtitle

            sanitized = sanitize_subtitle(result.text)
            if not sanitized:
                return
            self.cache.set_video_scoped(
                query,
                result.provider,
                sanitized,
                kind=result.kind,
                bluray_match=result.bluray_match,
                candidate=result.candidate,
                member_season=result.member_season,
                member_episode=result.member_episode,
                score=result.weighted_score,
                score_without_hash=result.weighted_score_without_hash,
                source_download_digest=result.source_download_digest,
            )
        except Exception as exc:  # pragma: no cover - caching must never break sync
            logger.debug("[reference] video-scoped cache write skipped: %s", type(exc).__name__)

    async def resolve_candidates_lazy(
        self,
        query: ReferenceQuery,
        *,
        update_validator: Callable[[str], bool] | None = None,
    ) -> Any:
        """Yield DISTINCT usable references one at a time, best-ranked first.

        Async generator: each ``__anext__`` performs at most one download, so a
        consumer that stops early (because a candidate verified) never pays for
        the rest of the pool. All guards from eager resolution apply unchanged
        per candidate: breaker, quota, pacing, validation, content-dedup.
        """
        ordered_pairs, _, _, provider_of, limit = await self._ranked_release_plan(query)
        stem = query.cache_stem
        shared: dict[str, Any] = {"downloaded": 0, "seen": set()}
        yielded = 0
        for cand, _ in ordered_pairs:
            if yielded >= limit:
                break
            cand_provider = provider_of.get(id(cand))
            if cand_provider is None:
                continue
            if self._is_rejected(stem, cand):
                continue
            result = await self.download_planned_candidate(
                cand, cand_provider, query, update_validator, shared
            )
            if not result.text:
                continue
            yielded += 1
            yield result

        logger.info(
            "[reference] resolve_all: %d distinct candidate(s) from %d ranked release(s) "
            "(limit=%d) digests=%s",
            yielded,
            shared.get("downloaded") or 0,
            limit,
            sorted(d[:12] for d in shared.get("seen") or ()),
        )

    async def _download_candidate(
        self, best, provider, query: ReferenceQuery
    ) -> ResolvedReference:
        """Download and decode the selected candidate (at most one file)."""
        best_name = getattr(best, "release_name", "?")
        api_key = self._provider_api_key(provider, query)
        decision_kind = (
            "hash"
            if any(r.verified_hash for r in release_scoring.matches([best], query))
            else "edition"
        )
        tier = reference_tier(query.target_filename, best)
        logger.info(
            "[reference] downloading reference %r (tier=%d/%s, lang=%s) via %s (kind=%s)",
            best_name,
            tier,
            tier_label(tier),
            getattr(best, "lang", "?"),
            self._provider_label(provider),
            decision_kind,
        )
        try:
            network_started = time.monotonic()
            raw = await provider.download_archive(best.download_url, api_key=api_key)
            network_ms = (time.monotonic() - network_started) * 1000
            timing = current_sync_timing.get()
            if timing is not None:
                timing.reference_network_ms += network_ms
        except TimeoutError:
            logger.warning("[reference] download timed out: candidate=%r provider=%s",
                           best_name, self._provider_label(provider))
            return ResolvedReference(None, reference_failure="provider_timeout")
        except Exception as exc:
            logger.warning("[reference] download failed: candidate=%r provider=%s error=%s",
                           best_name, self._provider_label(provider), type(exc).__name__)
            return ResolvedReference(None, reference_failure="download_failure")
        if not raw:
            # A rate-limited provider returns no payload rather than raising, so
            # the breaker state is what distinguishes "refused" from "empty".
            rate_limited = (
                hasattr(provider, "is_breaker_open") and provider.is_breaker_open()
            )
            logger.warning(
                "[reference] empty download payload: candidate=%r provider=%s rate_limited=%s",
                best_name, self._provider_label(provider), rate_limited,
            )
            return ResolvedReference(
                None, reference_failure="provider_rate_limited" if rate_limited else "download_failure"
            )

        with release_scoring.extracting_member() as selected_members:
            # Inflate/bsdtar are blocking; run them off the event loop. The
            # member list is shared by reference, so selections still record.
            decode_started = time.monotonic()
            decoded = await asyncio.to_thread(
                self._decode_payload,
                raw, best_name, season=query.season if query else None,
                episode=query.episode if query else None,
            )
            timing = current_sync_timing.get()
            if timing is not None:
                timing.reference_decode_ms += (time.monotonic() - decode_started) * 1000
        selected_member = selected_members[-1] if selected_members else None
        # An unresolved pack cannot authorize a synchronization reference.
        proof = release_scoring.reference_proof(best, query, selected_member)
        if proof is None:
            return ResolvedReference(None, reference_failure="candidate_rejected")
        if not decoded or len(decoded) <= self.min_bytes:
            logger.warning(
                "[reference] decoded content too small/empty for %r (%d bytes)",
                best_name,
                len(decoded) if decoded else 0,
            )
            return ResolvedReference(None)
        text = decoded.decode("utf-8", "replace")
        # An archive is unbundled to the single target episode, but an
        # uncompressed season pack is one continuous multi-episode timeline that
        # can never be cue-aligned to a single episode. Reject it here so the
        # next-ranked candidate is tried instead of poisoning the sync.
        if query is not None and query.season is not None and query.episode is not None:
            if looks_like_cumulative_pack(text):
                logger.warning(
                    "[reference] rejecting %r: decoded content is a cumulative "
                    "multi-episode pack, not episode S%02dE%02d",
                    best_name,
                    query.season,
                    query.episode,
                )
                return ResolvedReference(None)
        logger.info("[reference] decoded %d bytes for %r", len(decoded), best_name)
        # BluRay and REMUX share the same retail-disc master, so a REMUX target
        # against a BluRay (or REMUX) reference is a confirmed retail pair.
        bluray_match = is_retail_disc_source(
            _source_kind(query.target_filename)
        ) and is_retail_disc_source(_source_kind(best_name))
        source_digest = download_source_digest(
            self._provider_label(provider), getattr(best, "download_url", None)
        )
        # Demand-driven season-pack retention: if this download was a pack
        # archive, index it once so later same-lineage episodes reuse their
        # member locally. Never prefetches; never delays this request.
        try:
            from app.services.sync.season_pack import retain_season_pack

            retain_season_pack(
                raw=raw,
                provider_label=self._provider_label(provider),
                pack_candidate=best_name,
                source_download_digest=source_digest,
                query=query,
            )
        except Exception:  # retention must never break sync
            logger.debug("[season-pack] retain skipped: hook failed")
        capture_provider(text, self._provider_label(provider))
        return ResolvedReference(
            text,
            kind=decision_kind,
            bluray_match=bluray_match,
            candidate=best_name,
            provider=self._provider_label(provider),
            member_season=proof["season"],
            member_episode=proof["episode"],
            weighted_score=float(proof["score"]),
            weighted_score_without_hash=float(proof["score_without_hash"]),
            source_download_digest=source_digest,
        )

    async def _download_reference(
        self, releases, provider, api_key: str | None, query: ReferenceQuery
    ) -> ResolvedReference:
        """Backward-compatible helper: select + download from one provider's list."""
        if not releases:
            logger.info("[reference] provider returned no candidate releases")
            return ResolvedReference(None)
        best = _select_reference(list(releases), query)
        if best is None:
            logger.warning("[reference] aborting: no season/episode-matched reference")
            return ResolvedReference(None)
        return await self._download_candidate(best, provider, query)

    def _select_zip_member(
        self,
        members: list[tuple[str, int]],
        season: int | None,
        episode: int | None,
    ) -> str | None:
        """Pick the best subtitle member from a (possibly season-pack) ZIP."""
        return select_zip_member(members, season, episode, self.min_bytes)

    def _decode_payload(
        self,
        raw: bytes,
        release_name: str,
        season: int | None = None,
        episode: int | None = None,
    ) -> bytes | None:
        """Return subtitle bytes from a raw download, extracting ZIP archives."""
        return decode_payload(raw, release_name, season, episode, self.min_bytes)

    def _match_filename(self, query: ReferenceQuery) -> str | None:
        """Use the filename only when it carries scene tokens; otherwise match
        on IMDb + season/episode alone (top-rated official reference)."""
        if is_informative_release_name(query.target_filename):
            return query.target_filename
        if query.release_group:
            return None  # keep episode-level matching; release group is advisory
        return None

    @staticmethod
    def _has_match(result, field: str) -> bool:
        from app.services.release_scoring.model import MatchState

        return any(
            evidence.field == field and evidence.state is MatchState.MATCH
            for evidence in result.evidence
        )

    @staticmethod
    def _exact_release_identity(result, query: ReferenceQuery) -> bool:
        """True for a source-exact, same-group release for this video.

        This is a discovery shortcut only. An unresolved season pack still has
        to be downloaded and re-scored after member extraction; nothing here
        makes it ready or trusted.
        """
        if not result.discovery_eligible or result.source_priority != 0:
            return False
        group = query.release_group
        if not group:
            group = parse_release_identity(query.target_filename).release_group
        target_group = normalize_release_group(group)
        fact = result.candidate.fact("release_group")
        candidate_group = None if fact.conflict else normalize_release_group(fact.value)
        if not target_group or candidate_group != target_group:
            return False
        # A season pack must at least prove the requested season; the episode
        # member is resolved later. A single-episode file must prove the
        # requested episode, otherwise skipping Tier 2 would strand the plan on
        # an incompatible episode.
        if query.season is not None and not ExternalExactStrategy._has_match(result, "season"):
            return False
        if (
            query.episode is not None
            and not result.candidate.unresolved_pack
            and not ExternalExactStrategy._has_match(result, "episode")
        ):
            return False
        return True

    @staticmethod
    def _needs_season_enrichment(releases, query: ReferenceQuery, strict: bool = True) -> bool:
        """Discover packs unless a ready or exact release candidate exists."""
        if query.season is None:
            return False
        if not releases:
            return True
        from app.services.release_scoring.model import MatchState
        results = release_scoring.matches(list(releases), query)
        if not any(e.target.value for result in results for e in result.evidence if e.field == "source"):
            return False
        for candidate in releases:
            if _is_rel_provider_broken(candidate):
                continue
            if getattr(candidate, "provider", "").lower() == "opensubtitles" and not any(
                    result.verified_hash for result in release_scoring.matches([candidate], query)):
                continue
            if any(result.ready_reference_eligible and (result.verified_hash or result.state("source") is MatchState.MATCH)
                   for result in release_scoring.matches([candidate], query)):
                return False
        # A source-exact, same-release-group pack is already the best shape of
        # candidate tier 2 could find. It remains unresolved until extraction,
        # but that is a download-time proof, not a reason to spend another
        # provider search (and risk losing Tier 1 to a shared timeout).
        if any(ExternalExactStrategy._exact_release_identity(result, query) for result in results):
            logger.info(
                "[reference] Tier 1 already has an exact-release candidate; "
                "skipping season %s catalog enrichment",
                query.season,
            )
            return False
        return True

    async def _search_tiers(
        self,
        search,
        query: ReferenceQuery,
        provider_name: str,
        *,
        deadline: float | None = None,
    ) -> list:
        """Tier 1 episode query, enriched by a Tier 2 season query when needed.

        Tier 2 results are appended after Tier 1 (deduplicated by download URL)
        before the combined pool is ranked against the target video. Tier 2 is
        opportunistic: a timeout or provider failure returns the completed
        Tier 1 batch instead of discarding it.
        """
        deadline = deadline if deadline is not None else time.monotonic() + self.timeout
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"{provider_name} reference search budget exhausted")
        tier1 = await asyncio.wait_for(search(episode=query.episode), remaining)
        if query.episode is None or not self._needs_season_enrichment(tier1, query):
            return list(tier1)
        logger.info(
            "[reference] %s Tier 1 lacks %s-family candidates; querying season %s catalog",
            provider_name,
            _source_kind(query.target_filename),
            query.season,
        )
        tier2_budget = deadline - time.monotonic() - _TIER2_RETURN_MARGIN_SECONDS
        if tier2_budget < _TIER2_MIN_SECONDS:
            logger.warning(
                "[reference] %s Tier 2 skipped: %.2fs left; keeping %d Tier 1 candidate(s)",
                provider_name,
                max(0.0, tier2_budget),
                len(tier1),
            )
            return list(tier1)
        try:
            tier2 = await asyncio.wait_for(search(episode=None), tier2_budget)
        except Exception as exc:  # noqa: BLE001 - enrichment is optional
            logger.warning(
                "[reference] %s Tier 2 failed: %s; keeping %d Tier 1 candidate(s)",
                provider_name,
                type(exc).__name__,
                len(tier1),
            )
            return list(tier1)
        seen = {getattr(rel, "download_url", None) for rel in tier1}
        return list(tier1) + [
            rel for rel in tier2 if getattr(rel, "download_url", None) not in seen
        ]

    async def _search_subdl(self, query: ReferenceQuery, *, deadline: float | None = None) -> list:
        api_key = query.api_keys.get("subdl")

        async def _search(episode: int | None):
            return await self._subdl.search_subtitles(
                imdb_id=query.imdb_id,
                title=query.title,
                year=query.year,
                is_series=query.is_series,
                season=query.season,
                episode=episode,
                api_key=api_key,
                languages=list(_REFERENCE_LANGUAGES),
                target_filename=self._match_filename(query),
            )

        return await self._search_tiers(_search, query, "subdl", deadline=deadline)

    async def _search_subsource(self, query: ReferenceQuery, *, deadline: float | None = None) -> list:
        api_key = query.api_keys.get("subsource")

        async def _search(episode: int | None):
            return await self._subsource.search_subtitles(
                imdb_id=query.imdb_id,
                title=query.title,
                year=query.year,
                is_series=query.is_series,
                season=query.season,
                episode=episode,
                api_key=api_key,
                languages=list(_REFERENCE_LANGUAGES),
                target_filename=self._match_filename(query),
            )

        return await self._search_tiers(_search, query, "subsource", deadline=deadline)

    async def _search_opensubtitles(self, query: ReferenceQuery, *, deadline: float | None = None) -> list:
        api_key = query.api_keys.get("opensubtitles")

        async def _search(episode: int | None):
            return await self._opensubtitles.search_subtitles(
                imdb_id=query.imdb_id,
                title=query.title,
                year=query.year,
                is_series=query.is_series,
                season=query.season,
                episode=episode,
                api_key=api_key,
                languages=list(_REFERENCE_LANGUAGES),
                target_filename=self._match_filename(query),
                moviehash=query.video_hash,
                moviebytesize=query.video_size,
            )

        return await self._search_tiers(_search, query, "opensubtitles", deadline=deadline)

