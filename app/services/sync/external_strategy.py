"""Secondary sync strategy: external exact-match reference download.

Races SubDL and SubSource for the first *valid* result (a fast failure such as
SubDL HTTP 429 never cancels a slower provider), applies the
explicit decision tree (team > edition > abort), downloads at most one file,
and persists it to the reference disk cache. Anything the tree cannot confirm
aborts here so the orchestrator serves the original subtitle.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING

from app.config import settings
from app.services.sync.cache import ReferenceDiskCache
from app.services.sync.decode import decode_payload, select_zip_member
from app.services.sync.matching import (
    _source_kind,
    is_informative_release_name,
    is_retail_disc_source,
)
from app.services.sync.query import ResolvedReference
from app.services.sync.tree import decide

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.services.sync.query import ReferenceQuery

logger = logging.getLogger(__name__)

# Below the request budget (SYNC_TOTAL_REQUEST_BUDGET, default 7.5s) so
# reference resolution finishes inline and still leaves room for alass.
_DEFAULT_TIMEOUT = 5.0
_MIN_REFERENCE_BYTES = 5120


class ExternalExactStrategy:
    """External reference acquisition across SubDL and SubSource.

    Series episode queries run in two tiers: the episode query first, then a
    season-wide enrichment query when the episode results lack the stream's
    source family (retail season packs are indexed at season level).
    """

    name = "external"

    def __init__(
        self,
        subdl_provider=None,
        subsource_provider=None,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        min_bytes: int = _MIN_REFERENCE_BYTES,
        cache: ReferenceDiskCache | None = None,
    ) -> None:
        self._subdl = subdl_provider
        self._subsource = subsource_provider
        self.timeout = timeout
        self.min_bytes = min_bytes
        self.cache = cache if cache is not None else ReferenceDiskCache(min_bytes=min_bytes)

    async def resolve(self, query: ReferenceQuery) -> str | None:
        """Return the first valid reference subtitle (> ``min_bytes``) or ``None``."""
        resolved = await self.resolve_with_provenance(query)
        return resolved.text

    async def resolve_with_provenance(
        self, query: ReferenceQuery
    ) -> ResolvedReference:
        """Resolve a reference, reporting the tree decision kind with it.

        Kinds: ``team`` / ``edition`` for fresh tree verdicts; cache hits
        recover the persisted (or filename-inferred) kind. ``abort`` on
        failure.
        """
        # 0. Persistent disk cache: never re-hit external providers for the same episode.
        # The recovered kind is preserved so a cached team reference keeps its
        # verdict instead of being downgraded to "edition".
        cached = self.cache.get(query)
        if cached is not None:
            return cached

        tasks: list[asyncio.Task[ResolvedReference]] = []
        if self._subdl is not None:
            tasks.append(asyncio.create_task(self._fetch_subdl(query), name="subdl"))
        if self._subsource is not None:
            tasks.append(asyncio.create_task(self._fetch_subsource(query), name="subsource"))
        logger.info(
            "[reference] resolving imdb=%s S%sE%s filename=%r across %s (timeout=%.1fs)",
            query.imdb_id,
            query.season,
            query.episode,
            query.target_filename,
            [t.get_name() for t in tasks] or "no-providers",
            self.timeout,
        )
        if not tasks:
            return ResolvedReference(None)

        # Wait for the FIRST *valid* result rather than the first task to finish:
        # a fast failure (e.g. SubDL HTTP 429) must not cancel a slower provider
        # that would have delivered a usable reference.
        loop = asyncio.get_event_loop()
        deadline = loop.time() + self.timeout
        pending: set[asyncio.Task[ResolvedReference]] = set(tasks)
        result: ResolvedReference | None = None
        winner: str | None = None

        try:
            while pending:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                done, pending = await asyncio.wait(
                    pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    try:
                        resolved = task.result()
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.warning("[reference] %s task failed: %s", task.get_name(), exc)
                        resolved = ResolvedReference(None)
                    size = len(resolved.text.encode("utf-8")) if resolved.text else 0
                    logger.info("[reference] %s returned %d bytes", task.get_name(), size)
                    if resolved.text and size > self.min_bytes and result is None:
                        result = resolved
                        winner = task.get_name()
                if result is not None:
                    break
        finally:
            # Drain all children, including completed losers, on cancellation
            # as well as success. Provider HTTP calls must not outlive the resolver.
            for task in tasks:
                if not task.done():
                    task.cancel()
            with contextlib.suppress(Exception):
                await asyncio.gather(*tasks, return_exceptions=True)

        if result is None or result.text is None:
            logger.warning(
                "[reference] no valid reference within %.1fs (min=%d bytes)",
                self.timeout,
                self.min_bytes,
            )
            return ResolvedReference(None)
        result_text = result.text

        # Persist a sanitised copy so later switches for this episode are instant.
        from app.services.sync_service import sanitize_subtitle

        sanitized = sanitize_subtitle(result_text)
        if sanitized:
            self.cache.set(
                query,
                winner,
                sanitized,
                kind=result.kind,
                bluray_match=result.bluray_match,
                candidate=result.candidate,
            )
            result_text = sanitized
        logger.info("[reference] selected reference: %d bytes", len(result_text.encode("utf-8")))
        return ResolvedReference(result_text, kind=result.kind, bluray_match=result.bluray_match,
                                 candidate=result.candidate)

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

    async def _download_reference(
        self, releases, provider, api_key: str | None, query: ReferenceQuery
    ) -> ResolvedReference:
        """Download the tree-selected candidate and decode it (at most one file).

        Returns the subtitle text plus the tree decision kind (``team`` /
        ``edition`` / ``abort``) so callers can apply kind-aware guardrails.
        ``bluray_match`` records a bilaterally verified BluRay source pair,
        which unlocks the series recap allowance downstream.
        """
        if not releases:
            logger.info("[reference] provider returned no candidate releases")
            return ResolvedReference(None)

        # Explicit decision tree (team > edition > abort) replaces scoring.
        # Fail-safe policy: anything the tree cannot confirm aborts here and
        # the caller serves the original subtitle — never a guessed edition.
        strict = bool(getattr(settings, "SYNC_REQUIRE_EXACT_MATCH", True))
        decision = decide(
            list(releases),
            query,
            strict=strict,
            provider_name=getattr(provider, "name", type(provider).__name__),
        )
        if decision.kind == "abort" or decision.release is None:
            logger.warning("[reference] aborting: %s", decision.reason)
            return ResolvedReference(None)
        best = decision.release
        best_name = getattr(best, "release_name", "?")
        logger.info(
            "[reference] decision=%s (%s); downloading candidate %r via %s",
            decision.kind,
            decision.reason,
            getattr(best, "release_name", "?"),
            getattr(provider, "name", type(provider).__name__),
        )
        try:
            raw = await provider.download_archive(best.download_url, api_key=api_key)
        except Exception as exc:
            logger.warning("[reference] download failed: %s", exc)
            return ResolvedReference(None)
        if not raw:
            logger.warning("[reference] empty download payload for %r", best.release_name)
            return ResolvedReference(None)

        decoded = self._decode_payload(
            raw,
            best_name,
            season=query.season if query else None,
            episode=query.episode if query else None,
        )
        if not decoded:
            logger.warning("[reference] no subtitle content decoded for %r", best.release_name)
            return ResolvedReference(None)
        logger.info(
            "[reference] decoded %d bytes for %r",
            len(decoded),
            getattr(best, "release_name", "?"),
        )
        # BluRay and REMUX share the same retail-disc master, so a REMUX target
        # against a BluRay (or REMUX) reference is a confirmed retail pair and
        # must unlock the series recap allowance just like BluRay/BluRay.
        bluray_match = is_retail_disc_source(
            _source_kind(query.target_filename)
        ) and is_retail_disc_source(_source_kind(best_name))
        return ResolvedReference(
            decoded.decode("utf-8", "replace"),
            kind=decision.kind,
            bluray_match=bluray_match,
            candidate=best_name,
        )

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

        For series with a known stream source family that Tier 1 failed to
        cover (e.g. a BluRay stream whose episode query returned HDTV rips
        only): the season catalog may hold the full-season retail pack.

        When the target is uninformative (an obfuscated debrid hash) and the
        episode query returned nothing, relaxed mode still queries the season
        catalog — a season pack is the only remaining way to resolve a
        reference. If Tier 1 already produced candidates, they are used as-is
        rather than fanning out further.
        """
        if query.season is None:
            return False
        target_source = _source_kind(query.target_filename)
        if not target_source:
            return not strict and not releases
        for rel in releases or []:
            candidate_source = _source_kind(getattr(rel, "release_name", "") or "")
            if candidate_source and (
                candidate_source == target_source
                or {target_source, candidate_source} == {"bluray", "remux"}
            ):
                return False
        return True

    async def _search_tiers(self, search, query: ReferenceQuery, provider_name: str) -> list:
        """Tier 1 episode query, enriched by a Tier 2 season query when needed.

        Tier 2 results are appended after Tier 1 (deduplicated by download
        URL) so per-episode files keep priority and the tree still decides.
        """
        strict = bool(getattr(settings, "SYNC_REQUIRE_EXACT_MATCH", True))
        tier1 = await search(episode=query.episode)
        if not self._needs_season_enrichment(tier1, query, strict):
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

    async def _fetch_subdl(self, query: ReferenceQuery) -> ResolvedReference:
        api_key = query.api_keys.get("subdl")

        async def _search(episode: int | None):
            return await self._subdl.search_subtitles(
                imdb_id=query.imdb_id,
                is_series=query.is_series,
                season=query.season,
                episode=episode,
                api_key=api_key,
                languages=list(query.languages),
                target_filename=self._match_filename(query),
            )

        releases = await self._search_tiers(_search, query, "subdl")
        return await self._download_reference(releases, self._subdl, api_key, query)

    async def _fetch_subsource(self, query: ReferenceQuery) -> ResolvedReference:
        api_key = query.api_keys.get("subsource")

        async def _search(episode: int | None):
            return await self._subsource.search_subtitles(
                imdb_id=query.imdb_id,
                is_series=query.is_series,
                season=query.season,
                episode=episode,
                api_key=api_key,
                languages=list(query.languages),
                target_filename=self._match_filename(query),
            )

        releases = await self._search_tiers(_search, query, "subsource")
        return await self._download_reference(releases, self._subsource, api_key, query)
