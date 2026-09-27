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
from typing import TYPE_CHECKING

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
    looks_like_season_pack,
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

# Weighted-scoring weights for candidate reference selection. The goal is to pick
# the reference whose release quality matches the playing video file.
_GROUP_SCORE = 50  # exact release group
_SOURCE_SCORE = 30  # shared source family (web-dl/web, bluray/remux)
_PLATFORM_SCORE = 20  # identical streaming service (HMAX/ATVP/NF/AMZN/DSNP)
_STREAMING_FAMILY_SCORE = 10  # different streaming service, same streaming family
_RESOLUTION_SCORE = 10  # identical resolution
_RESOLUTION_NEAR_SCORE = 5  # 1080p neighbour of a 4K target
_CODEC_SCORE = 5  # shared codec (x265/HEVC/x264)
_EPISODE_SCORE = 10  # explicit episode single (vs a whole-season pack)
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

    Weights: exact release group ``+50``, matching source family ``+30``,
    identical streaming service ``+20`` (else same streaming family ``+10``),
    resolution ``+10`` (or ``+5`` for a 1080p neighbour of a 4K target), codec
    ``+5``, episode single ``+10`` (season pack ``-10``), language priority
    ``+10`` (English) / ``+5`` (es/fr/de/it), non-HI ``+5``.
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

    target_platforms = _platform_tags(target_name)
    cand_platforms = _platform_tags(cand_name)
    if target_platforms & cand_platforms:
        score += _PLATFORM_SCORE
    elif target_platforms and cand_platforms:
        # Both are streaming releases but different services: still a much better
        # timing anchor than a disc/encode of another edition.
        score += _STREAMING_FAMILY_SCORE

    target_res = _resolution(target_name)
    cand_res = _resolution(cand_name)
    if target_res and cand_res:
        if cand_res == target_res:
            score += _RESOLUTION_SCORE
        elif target_res in ("2160p", "4k") and cand_res in ("1080p", "2160p", "4k"):
            score += _RESOLUTION_NEAR_SCORE

    target_codec = _codec_kind(target_name)
    if target_codec and target_codec == _codec_kind(cand_name):
        score += _CODEC_SCORE

    if candidate_episode_number(cand_name) is not None:
        score += _EPISODE_SCORE
    elif looks_like_season_pack(cand_name):
        score -= _EPISODE_SCORE

    score += _LANG_SCORES.get(str(getattr(release, "lang", "") or "").strip().lower(), 0)

    if not getattr(release, "hearing_impaired", False):
        score += _NON_HI_SCORE
    return score


def _select_reference(releases, query: ReferenceQuery):
    """Pick the best season/episode-matched reference by weighted score.

    Eligible candidates (matching season/episode) are sorted by score
    descending; ties keep provider order. A byte-exact MovieHash match always
    wins. An all-zero field falls back to the first available candidate rather
    than aborting â€” ``None`` is returned only when nothing matches S/E.
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
    """External reference acquisition across SubDL, SubSource, and OpenSubtitles.

    All providers are queried concurrently; their candidates are pooled and
    scored globally, so the fastest provider no longer wins by default. Series
    episode queries are enriched with a season-wide query only when the episode
    query returned nothing.
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
        """Return the best reference subtitle (> ``min_bytes``) or ``None``."""
        resolved = await self.resolve_with_provenance(query)
        return resolved.text

    async def resolve_with_provenance(
        self, query: ReferenceQuery
    ) -> ResolvedReference:
        """Query every provider, pool the candidates, download the best one.

        The weighted score is applied across all providers together, so the
        reference that best matches the video's release quality wins regardless
        of which provider responded first. Cache hits recover the persisted
        decision kind.
        """
        cached = self.cache.get(query)
        if cached is not None:
            return cached

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
        best = _select_reference([rel for rel, _ in candidates], query)
        if best is None:
            logger.warning("[reference] aborting: no season/episode-matched reference")
            return ResolvedReference(None)
        provider = provider_of.get(id(best))

        result = await self._download_candidate(best, provider, query)
        text = result.text
        if text is None:
            return ResolvedReference(None)

        # Persist a sanitised copy so later switches for this episode are instant.
        from app.services.sync_service import sanitize_subtitle

        sanitized = sanitize_subtitle(text)
        if sanitized:
            self.cache.set(
                query,
                self._provider_label(provider),
                sanitized,
                kind=result.kind,
                bluray_match=result.bluray_match,
                candidate=result.candidate,
            )
            text = sanitized
        logger.info("[reference] selected reference: %d bytes", len(text.encode("utf-8")))
        return ResolvedReference(
            text,
            kind=result.kind,
            bluray_match=result.bluray_match,
            candidate=result.candidate,
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
        decision_kind = "hash" if getattr(best, "is_hash_match", False) else "edition"
        logger.info(
            "[reference] downloading reference %r (score=%d, lang=%s) via %s (kind=%s)",
            best_name,
            score_candidate(query.target_filename, best),
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
        logger.info("[reference] decoded %d bytes for %r", len(decoded), best_name)
        # BluRay and REMUX share the same retail-disc master, so a REMUX target
        # against a BluRay (or REMUX) reference is a confirmed retail pair.
        bluray_match = is_retail_disc_source(
            _source_kind(query.target_filename)
        ) and is_retail_disc_source(_source_kind(best_name))
        return ResolvedReference(
            decoded.decode("utf-8", "replace"),
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
                is_series=query.is_series,
                season=query.season,
                episode=episode,
                api_key=api_key,
                languages=list(_REFERENCE_LANGUAGES),
                target_filename=self._match_filename(query),
            )

        return await self._search_tiers(_search, query, "opensubtitles")
