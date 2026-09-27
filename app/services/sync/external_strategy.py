"""Secondary sync strategy: external reference download.

Races SubDL, SubSource, and OpenSubtitles for the first *valid* result (a fast
failure such as SubDL HTTP 429 never cancels a slower provider), scores the
candidates against the target video, downloads the best one, and persists it to
the reference disk cache.

``alass`` only anchors speech-timing intervals, so any high-accuracy official
track is a usable reference — not just English. Candidates are ranked by a
weighted score (release group, source family, streaming service, resolution,
codec, language priority, non-HI). If no candidate scores, the first
season/episode match is used rather than aborting.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from typing import TYPE_CHECKING

from app.config import settings
from app.services.sync.cache import ReferenceDiskCache
from app.services.sync.decode import decode_payload, select_zip_member
from app.services.sync.matching import (
    _codec_kind,
    _release_group,
    _resolution,
    _season_number,
    _source_kind,
    _sources_compatible,
    candidate_episode_number,
    is_informative_release_name,
    is_retail_disc_source,
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
# user's subtitle-language preference): common retail tracks are all valid
# timing anchors for alass. ISO-639-1 here; providers normalize to their own.
_REFERENCE_LANGUAGES = ("en", "es", "fr", "de", "it")

# Weighted-scoring weights for candidate reference selection.
_GROUP_SCORE = 50  # exact release group
_SOURCE_SCORE = 30  # shared source family (web-dl/web, bluray/remux)
_PLATFORM_SCORE = 20  # shared streaming service (HMAX/ATVP/NF/AMZN/DSNP)
_RESOLUTION_SCORE = 10  # shared resolution
_CODEC_SCORE = 5  # shared codec (x265/HEVC/x264)
_EN_SCORE = 10  # English preferred on ties
_ALT_LANG_SCORE = 5  # es / fr / de / it
_NON_HI_SCORE = 5  # standard dialogue over hearing-impaired

_LANG_SCORES = {
    "eng": _EN_SCORE,
    "en": _EN_SCORE,
    "spa": _ALT_LANG_SCORE,
    "es": _ALT_LANG_SCORE,
    "fra": _ALT_LANG_SCORE,
    "fr": _ALT_LANG_SCORE,
    "deu": _ALT_LANG_SCORE,
    "de": _ALT_LANG_SCORE,
    "ita": _ALT_LANG_SCORE,
    "it": _ALT_LANG_SCORE,
}

_PLATFORM_PATTERNS = {
    "hmax": re.compile(r"(?i)\b(?:hmax|hbo[\s._-]?max)\b"),
    "atvp": re.compile(r"(?i)\b(?:atvp|apple[\s._-]?tv)\b"),
    "nf": re.compile(r"(?i)\bnf\b"),
    "amzn": re.compile(r"(?i)\bamzn\b"),
    "dsnp": re.compile(r"(?i)\bdsnp\b"),
    "hulu": re.compile(r"(?i)\bhulu\b"),
    "pcok": re.compile(r"(?i)\bpcok\b"),
    "itunes": re.compile(r"(?i)\bitunes\b"),
    "stan": re.compile(r"(?i)\bstan\b"),
}


def _release_name(release) -> str:
    return str(getattr(release, "release_name", "") or "")


def _platform_tags(name: str | None) -> frozenset[str]:
    """Streaming-service tags present in a release name (HMAX/NF/AMZN/...)."""
    lowered = name or ""
    return frozenset(tag for tag, pattern in _PLATFORM_PATTERNS.items() if pattern.search(lowered))


def score_candidate(target_name: str | None, release) -> int:
    """Weighted score for one candidate reference against the target video.

    Weights: exact release group ``+50``, matching source family ``+30``, shared
    streaming service ``+20``, resolution ``+10``, codec ``+5``, language
    priority ``+10`` (English) / ``+5`` (es/fr/de/it), non-HI ``+5``.
    """
    cand_name = _release_name(release)
    if not cand_name:
        return 0
    score = 0

    target_group = (_release_group(target_name) or "").lower()
    if target_group:
        cand_group = (_release_group(cand_name) or "").lower()
        cand_tokens = set(re.findall(r"[a-z0-9]+", cand_name.lower()))
        if cand_group == target_group or target_group in cand_tokens:
            score += _GROUP_SCORE

    target_source = _source_kind(target_name)
    cand_source = _source_kind(cand_name)
    if target_source and cand_source and _sources_compatible(target_source, cand_source):
        score += _SOURCE_SCORE

    if _platform_tags(target_name) & _platform_tags(cand_name):
        score += _PLATFORM_SCORE

    target_res = _resolution(target_name)
    if target_res and target_res == _resolution(cand_name):
        score += _RESOLUTION_SCORE

    target_codec = _codec_kind(target_name)
    if target_codec and target_codec == _codec_kind(cand_name):
        score += _CODEC_SCORE

    score += _LANG_SCORES.get(str(getattr(release, "lang", "") or "").strip().lower(), 0)

    if not getattr(release, "hearing_impaired", False):
        score += _NON_HI_SCORE
    return score


def _select_reference(releases, query: ReferenceQuery):
    """Pick the best season/episode-matched reference by weighted score.

    Eligible candidates (matching season/episode) are sorted by score
    descending; ties keep provider order. A byte-exact MovieHash match always
    wins. An all-zero field falls back to the first available candidate rather
    than aborting — ``None`` is returned only when nothing matches S/E.
    """
    pool = list(releases or [])
    if not pool:
        return None
    if query.season is not None:
        matched = [rel for rel in pool if _season_number(_release_name(rel)) == query.season]
        if not matched:
            return None
        pool = matched
    if query.episode is not None:
        pool = [
            rel
            for rel in pool
            if candidate_episode_number(_release_name(rel)) in (None, query.episode)
        ]
    if not pool:
        return None

    scored = [(score_candidate(query.target_filename, rel), rel) for rel in pool]
    # Hash-confirmed tracks are byte-exact ground truth; otherwise the highest
    # weighted score wins, preserving provider order on ties.
    scored.sort(
        key=lambda item: (1 if getattr(item[1], "is_hash_match", False) else 0, item[0]),
        reverse=True,
    )
    for score, rel in scored[:5]:
        logger.info(
            "[reference] candidate score=%d lang=%s hash=%s %r",
            score,
            getattr(rel, "lang", "?"),
            bool(getattr(rel, "is_hash_match", False)),
            _release_name(rel),
        )
    return scored[0][1]


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
        if self._opensubtitles is not None:
            tasks.append(
                asyncio.create_task(self._fetch_opensubtitles(query), name="opensubtitles")
            )
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
        """Download the best-scoring season/episode-matched reference.

        Selection is permissive: any matching season/episode is eligible and the
        highest weighted score wins (never abort on a low score). A candidate is
        not rejected for release group/source/resolution/edition. ``bluray_match``
        records a bilaterally verified BluRay/REMUX source pair for recap handling.
        """
        if not releases:
            logger.info("[reference] provider returned no candidate releases")
            return ResolvedReference(None)

        best = _select_reference(releases, query)
        if best is None:
            logger.warning(
                "[reference] aborting: no season/episode-matched reference"
            )
            return ResolvedReference(None)
        best_name = getattr(best, "release_name", "?")
        decision_kind = "hash" if getattr(best, "is_hash_match", False) else "edition"
        logger.info(
            "[reference] selected reference %r (score=%d, lang=%s) via %s (kind=%s)",
            best_name,
            score_candidate(query.target_filename, best),
            getattr(best, "lang", "?"),
            getattr(provider, "name", type(provider).__name__),
            decision_kind,
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
            kind=decision_kind,
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

        Only when the episode query returned nothing for a series: a season pack
        is the last remaining way to resolve a reference. If Tier 1 already
        produced any candidate, it is used as-is.
        """
        return query.season is not None and not releases

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
                languages=list(_REFERENCE_LANGUAGES),
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
                languages=list(_REFERENCE_LANGUAGES),
                target_filename=self._match_filename(query),
            )

        releases = await self._search_tiers(_search, query, "subsource")
        return await self._download_reference(releases, self._subsource, api_key, query)

    async def _fetch_opensubtitles(self, query: ReferenceQuery) -> ResolvedReference:
        api_key = query.api_keys.get("opensubtitles")

        async def _search(episode: int | None):
            return await self._opensubtitles.search_subtitles(
                imdb_id=query.imdb_id,
                is_series=query.is_series,
                season=query.season,
                episode=episode,
                api_key=api_key,
                languages=list(_REFERENCE_LANGUAGES),
                target_filename=self._match_filename(query),
            )

        releases = await self._search_tiers(_search, query, "opensubtitles")
        return await self._download_reference(releases, self._opensubtitles, api_key, query)
