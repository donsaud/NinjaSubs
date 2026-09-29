"""External reference resolution for subtitle auto-sync.

All providers (SubDL / SubSource / OpenSubtitles) are queried concurrently and
their candidates are pooled, then scored against the *video's* filename. The
single best-scoring candidate is downloaded and used as the ``alass`` reference.

``alass`` only anchors speech-timing intervals, so any high-accuracy official
track works â€” regardless of language (en/es/fr/de/it are searched). The score
favours a reference whose release quality matches the playing video (same source
family, same streaming service family, same release group, resolution and
codec), so a BluRay-quality Arabic subtitle played against a WEB-DL video is
re-timed against a WEB-DL reference rather than a mismatched edition. When no
candidate scores, the first season/episode match is used instead of aborting.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from app.services.sync.cache import ReferenceDiskCache
from app.services.sync.decode import (
    decode_payload,
    looks_like_cumulative_pack,
    select_zip_member,
)
from app.services.sync.matching import (
    _release_group,
    _season_number,
    _source_kind,
    candidate_episode_number,
    guess_metadata,
    is_informative_release_name,
    is_retail_disc_source,
    normalize_release_group,
)
from app.services.sync.query import ResolvedReference

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.services.sync.query import ReferenceQuery

logger = logging.getLogger(__name__)

# Below the request budget (SYNC_TOTAL_REQUEST_BUDGET, default 7.5s) so
# reference resolution finishes inline and still leaves room for alass.
_DEFAULT_TIMEOUT = 5.0
_MIN_REFERENCE_BYTES = 5120

# Language codes searched/fetched for the *reference* track (independent of the
# user's subtitle-language preference): English is the primary universal retail
# timing reference anchor, with Arabic as secondary reference. Foreign European
# tracks (it/de/fr/es) are excluded from the reference search pool because they
# often carry distributor intro offsets, dubbed audio pacing, or PAL 25fps shifts.
_REFERENCE_LANGUAGES = ("en", "ar")

# Deterministic reference tiers. Selection is a strict hierarchy: a candidate in a
# better tier always beats every candidate in a worse tier, so no amount of
# accumulated property points (resolution, codec, audio, platform) can promote a
# generic release above an exact scene-group match.
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


def tier_label(tier: int) -> str:
    """Human-readable name for a reference tier (used in logs and cache keys)."""
    return _TIER_LABELS.get(tier, "fallback")


_EDITION_PATTERNS = [
    (
        re.compile(
            r"(?i)\b(director'?s?[-._ ]?cut|directors[-._ ]?cut|directors|dc[-._ ]?cut|dir[-._ ]?cut)\b"
        ),
        "DIRECTORS_CUT",
    ),
    (
        re.compile(r"(?i)\b(extended[-._ ]?(?:cut|edition)?|ext[-._ ]?cut)\b"),
        "EXTENDED",
    ),
    (
        re.compile(r"(?i)\b(unrated[-._ ]?(?:cut|edition)?)\b"),
        "UNRATED",
    ),
    (
        re.compile(r"(?i)\b(theatrical[-._ ]?(?:cut|edition)?)\b"),
        "THEATRICAL",
    ),
    (
        re.compile(r"(?i)\b(imax[-._ ]?(?:enhanced|edition|cut)?)\b"),
        "IMAX",
    ),
    (
        re.compile(r"(?i)\b(special[-._ ]?edition|se[-._ ]?cut)\b"),
        "SPECIAL_EDITION",
    ),
    (
        re.compile(r"(?i)\b(remastered|remaster)\b"),
        "REMASTERED",
    ),
]


def _edition_kind(name: str | None) -> str | None:
    """Recognize standard movie/series cut editions from release filename."""
    if not name:
        return None
    meta = guess_metadata(name)
    ed = meta.get("edition")
    if ed:
        ed_str = str(ed).lower()
        if "director" in ed_str:
            return "DIRECTORS_CUT"
        if "extended" in ed_str:
            return "EXTENDED"
        if "unrated" in ed_str:
            return "UNRATED"
        if "theatrical" in ed_str:
            return "THEATRICAL"
        if "imax" in ed_str:
            return "IMAX"
        if "special" in ed_str:
            return "SPECIAL_EDITION"
        if "remaster" in ed_str:
            return "REMASTERED"
    for pattern, edition in _EDITION_PATTERNS:
        if pattern.search(name):
            return edition
    return None

def _release_name(release) -> str:
    return str(getattr(release, "release_name", "") or "")


def _groups_match(target_name: str | None, cand_name: str) -> bool:
    """True when candidate and target share the exact release group.

    Compared as a case- and separator-insensitive token, so ``PiR8`` matches
    ``pir8`` and ``Pi_R8``. No other property can substitute for this: an exact
    group means the same mastering pass, so the cue grid lines up by
    construction.
    """
    target_group = normalize_release_group(_release_group(target_name) or "")
    if not target_group:
        return False
    cand_group = normalize_release_group(_release_group(cand_name) or "")
    if not cand_group:
        return False
    if cand_group == target_group:
        return True
    # A candidate may carry the tag as a bracketed prefix rather than a suffix.
    return target_group in re.sub(r"[^a-z0-9]", "", cand_name.lower())


def _source_edition_match(target_name: str | None, cand_name: str) -> bool:
    """True when both sides share a source medium and a compatible cut."""
    target_source = _source_kind(target_name)
    cand_source = _source_kind(cand_name)
    if not target_source or not cand_source or target_source != cand_source:
        return False
    # Same medium but a different cut (Theatrical vs Extended) shifts every cue
    # by the missing footage, so it is treated as a mismatch rather than a match.
    target_ed = _edition_kind(target_name)
    cand_ed = _edition_kind(cand_name)
    if target_ed and cand_ed and target_ed != cand_ed:
        return False
    return True


def reference_tier(target_name: str | None, release) -> int:
    """Classify a reference candidate into the deterministic tier hierarchy.

    This is a strict ordering, not a score. A generic release that happens to
    match resolution, codec and audio can never outrank an exact release-group
    match, because it is compared in a strictly worse tier.
    """
    if getattr(release, "is_hash_match", False) or getattr(release, "matched_by_hash", False):
        return TIER_HASH
    cand_name = _release_name(release)
    if not target_name or not cand_name:
        return TIER_FALLBACK
    if _groups_match(target_name, cand_name):
        return TIER_EXACT_GROUP
    if _source_edition_match(target_name, cand_name):
        return TIER_SOURCE_EDITION
    return TIER_FALLBACK


def _is_scene_named(cand_name: str) -> int:
    clean = cand_name.rsplit("/", 1)[-1].strip()
    return 1 if ("." in clean and " " not in clean) else 0


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


_PROVIDER_TIE_PRIORITY = {
    "subsource": 2,
    "subdl": 2,
    "opensubtitles": 1,
}


def _is_exact_episode(release, query: ReferenceQuery) -> bool:
    """True when the candidate names the queried episode rather than a whole season.

    Season packs are still eligible references (they get unbundled at extraction
    time), but an already-episode-specific track is preferred because it needs no
    slicing and its own episode number is guaranteed correct.
    """
    if query is None or query.episode is None:
        return True
    return candidate_episode_number(_release_name(release)) == query.episode


def _select_candidates_ranked(releases, query: ReferenceQuery) -> list:
    """Return all season/episode-matched reference candidates sorted by rank descending.

    Eligible candidates (matching season/episode) are sorted by score
    descending; ties keep provider order. A byte-exact MovieHash match always
    wins.
    """
    pool = list(releases or [])
    if not pool:
        return []
    if query and query.target_download_url:
        pool = [rel for rel in pool if getattr(rel, "download_url", None) != query.target_download_url]
    if query and query.target_sub_release_name:
        tgt_group = (_release_group(query.target_sub_release_name) or "").lower()
        stream_group = (_release_group(query.target_filename) or "").lower()
        if not (tgt_group and stream_group and tgt_group == stream_group):
            pool = [rel for rel in pool if _release_name(rel) != query.target_sub_release_name]
    if not pool:
        return []
    if query.season is not None:
        matched = [rel for rel in pool if _season_number(_release_name(rel)) == query.season]
        if not matched:
            return []
        pool = matched
    if query.episode is not None:
        # A candidate tagged with a *different* episode can never be sliced down to
        # the target, so it is dropped. Season packs stay eligible on purpose: they
        # are unbundled per-episode at extraction time (see select_zip_member), so
        # a high-tier retail pack is still the best available reference. They are
        # only ranked below exact-episode candidates (see the sort key below).
        pool = [
            rel
            for rel in pool
            if candidate_episode_number(_release_name(rel)) in (None, query.episode)
        ]
    if not pool:
        return []

    tiered = [(reference_tier(query.target_filename, rel), rel) for rel in pool]
    # Strict tier hierarchy first; every remaining key is a deterministic
    # tie-breaker used only *within* a tier:
    #   tier > healthy provider > exact-episode over season pack >
    #   English reference anchor > non-HI > standard scene naming >
    #   provider quota priority > release name.
    target_langs = tuple(
        str(lang).lower()
        for lang in (query.languages if query and query.languages else ("ara", "ar"))
    )
    tiered.sort(
        key=lambda item: (
            item[0],
            1 if _is_rel_provider_broken(item[1]) else 0,
            0 if _is_exact_episode(item[1], query) else 1,
            0 if str(getattr(item[1], "lang", "")).lower().startswith("en") else 1,
            0 if str(getattr(item[1], "lang", "")).lower() in target_langs else 1,
            1 if getattr(item[1], "hearing_impaired", False) else 0,
            0 if _is_scene_named(_release_name(item[1])) else 1,
            -_PROVIDER_TIE_PRIORITY.get(
                str(getattr(item[1], "provider", "") or "").lower(), 0
            ),
            _release_name(item[1]),
        )
    )
    for tier, rel in tiered[:5]:
        logger.info(
            "[reference] candidate tier=%d (%s) lang=%s hash=%s provider=%s %r",
            tier,
            tier_label(tier),
            getattr(rel, "lang", "?"),
            bool(getattr(rel, "is_hash_match", False)),
            getattr(rel, "provider", "?"),
            _release_name(rel),
        )
    return [rel for _, rel in tiered]


def _select_reference(releases, query: ReferenceQuery):
    """Pick the single best reference by deterministic tier then tie-breakers."""
    ranked = _select_candidates_ranked(releases, query)
    return ranked[0] if ranked else None


class ExternalExactStrategy:
    """External reference acquisition across SubDL, SubSource, and OpenSubtitles.

    All providers are queried concurrently; their candidates are pooled and
    scored globally, so the fastest provider no longer wins by default. Series
    episode queries are enriched with a season-wide query only when the episode
    query returned nothing.
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
        # into another subtitle with a different cue layout.
        self._rejected: dict[str, set[str]] = {}

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
        attempts = 0
        stem = query.cache_stem
        # References already downloaded during THIS request, as
        # (provider, name, text). Used for duplicate collapse and consensus,
        # both of which are measurements over data we already have.
        observed_references: list[tuple[str, str | None, str | None]] = []
        candidate_decision_kinds: dict[str, str] = {}

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
            # the account's daily allowance, and a speculative reference
            # evaluation can walk through many candidates before one validates.
            # Speculating on a metered download burns that allowance for a guess
            # we have no proof for, so it is only spent when OpenSubtitles has
            # already proven the file byte-exact via MovieHash. Text and series
            # reference sourcing relies on the unlimited SubSource/SubDL tier.
            if (
                cand_provider is self._opensubtitles
                and reference_tier(query.target_filename, cand) != TIER_HASH
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
                    self._mark_rejected(stem, cand)
                    logger.warning(
                        "[reference] candidate %r (provider=%s) failed target cue-sanity; trying next candidate",
                        getattr(cand, "release_name", "?"),
                        provider_lbl,
                    )
                else:
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

        return ResolvedReference(
            text,
            kind=result.kind,
            bluray_match=result.bluray_match,
            candidate=result.candidate,
            reference_trust=assessment.trust.value if assessment else None,
            reference_consensus=assessment.consensus_score if assessment else None,
            reference_independent_sources=assessment.independent_sources if assessment else 0,
            reference_failure=assessment.failure.value if assessment and assessment.failure else None,
            reference_reasons=list(assessment.reasons) if assessment else [],
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

        Each provider is bounded by the resolver timeout; a slow/failing provider
        simply contributes nothing.
        """

        async def one(provider) -> list[tuple]:
            name = self._provider_label(provider)
            try:
                releases = await asyncio.wait_for(
                    self._search_provider(provider, query), self.timeout
                )
            except Exception as exc:  # noqa: BLE001 - one provider must not sink the rest
                logger.warning("[reference] %s search failed: %s", name, exc)
                return []
            logger.info("[reference] %s returned %d candidate(s)", name, len(releases or []))
            return [(rel, provider) for rel in (releases or [])]

        batches = await asyncio.gather(*(one(p) for p in providers), return_exceptions=True)
        merged: list[tuple] = []
        for batch in batches:
            if isinstance(batch, list):
                merged.extend(batch)

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

    async def _search_provider(self, provider, query: ReferenceQuery) -> list:
        """Route a provider to its search implementation (Identity, not name)."""
        if provider is self._subdl:
            return await self._search_subdl(query)
        if provider is self._subsource:
            return await self._search_subsource(query)
        if provider is self._opensubtitles:
            return await self._search_opensubtitles(query)
        raise ValueError(f"unknown provider {provider!r}")

    def _provider_api_key(self, provider, query: ReferenceQuery) -> str | None:
        if provider is self._subdl:
            return query.api_keys.get("subdl")
        if provider is self._subsource:
            return query.api_keys.get("subsource")
        if provider is self._opensubtitles:
            return query.api_keys.get("opensubtitles")
        return None

    async def _download_candidate(
        self, best, provider, query: ReferenceQuery
    ) -> ResolvedReference:
        """Download and decode the selected candidate (at most one file)."""
        best_name = getattr(best, "release_name", "?")
        api_key = self._provider_api_key(provider, query)
        decision_kind = (
            "hash"
            if (getattr(best, "is_hash_match", False) or getattr(best, "matched_by_hash", False))
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
            raw = await provider.download_archive(best.download_url, api_key=api_key)
        except Exception as exc:
            logger.warning("[reference] download failed: %s", exc)
            return ResolvedReference(None)
        if not raw:
            logger.warning("[reference] empty download payload for %r", best_name)
            return ResolvedReference(None)

        decoded = self._decode_payload(
            raw,
            best_name,
            season=query.season if query else None,
            episode=query.episode if query else None,
        )
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
        return ResolvedReference(
            text,
            kind=decision_kind,
            bluray_match=bluray_match,
            candidate=best_name,
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
    def _needs_season_enrichment(releases, query: ReferenceQuery, strict: bool = True) -> bool:
        """True when a season-wide Tier 2 query is warranted.

        Only when the episode query returned nothing for a series: a season pack
        is the last remaining way to resolve a reference. If Tier 1 already
        produced any candidate, it is used as-is.
        """
        return query.season is not None and not releases

    async def _search_tiers(self, search, query: ReferenceQuery, provider_name: str) -> list:
        """Tier 1 episode query, enriched by a Tier 2 season query when needed.

        Tier 2 results are appended after Tier 1 (deduplicated by download URL)
        so per-episode files keep priority.
        """
        tier1 = await search(episode=query.episode)
        if not self._needs_season_enrichment(tier1, query):
            return list(tier1)
        logger.info(
            "[reference] %s Tier 1 lacks %s-family candidates; querying season %s catalog",
            provider_name,
            _source_kind(query.target_filename),
            query.season,
        )
        tier2 = await search(episode=None)
        seen = {getattr(rel, "download_url", None) for rel in tier1}
        return list(tier1) + [
            rel for rel in tier2 if getattr(rel, "download_url", None) not in seen
        ]

    async def _search_subdl(self, query: ReferenceQuery) -> list:
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

        return await self._search_tiers(_search, query, "subdl")

    async def _search_subsource(self, query: ReferenceQuery) -> list:
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

        return await self._search_tiers(_search, query, "subsource")

    async def _search_opensubtitles(self, query: ReferenceQuery) -> list:
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

        return await self._search_tiers(_search, query, "opensubtitles")

