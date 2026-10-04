"""OpenSubtitles MovieHash reference sourcing for ALASS auto-sync.

This is deliberately a separate path from the normal OpenSubtitles provider:

* the user toggle ``enable_opensubtitles`` decides whether OpenSubtitles
  results appear in Stremio listings. This module is reached from the auto-sync
  orchestrator instead, so turning the provider off does not disable hash
  reference retrieval, and turning auto-sync off means this is never consulted
  at all;
* a candidate is accepted only when OpenSubtitles itself reports
  ``attributes.moviehash_match is True``. A title, IMDb ID, filename or
  release-name resemblance can therefore never be mistaken for a byte-exact
  reference, and when the request carries no hash metadata no match is claimed
  and nothing is downloaded;
* selection is language-agnostic. MovieHash identifies the exact file, so the
  reference may be in any language the API returns and none is preferred over
  another. The reference supplies timing only: the user's subtitle remains the
  ALASS target and keeps its own language.

Everything expensive is metered, because OpenSubtitles downloads draw on a
daily quota: at most one candidate is downloaded, the search is skipped
entirely when the circuit breaker is open (HTTP 429 / HTTP 406 quota
exhaustion), and results are cached on the shared reference disk cache so a
repeated request for the same video never re-spends the quota.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from app.config import settings
from app.services.sync.cache import ReferenceDiskCache
from app.services.sync.decode import decode_payload, looks_like_cumulative_pack
from app.services.sync.query import ReferenceQuery, ResolvedReference
from app.services.sync.reference import ReferenceTrust

logger = logging.getLogger("uvicorn.error")

#: Search budget for the hash lookup. A MovieHash query is a single indexed
#: lookup, so this is short; exceeding it simply falls through to the existing
#: reference strategies rather than delaying the subtitle response.
_HASH_SEARCH_TIMEOUT = 5.0

#: Floor on the decoded reference, mirroring ``ExternalExactStrategy``. A tiny
#: payload cannot align anything and would only waste an ALASS run.
_MIN_REFERENCE_BYTES = 5120

#: Formats ALASS can consume directly. Anything else would need a conversion
#: guess on the reference path; the existing ``sanitize_subtitle`` at serve time
#: still converts whatever arrives, so this is only a preference, not a filter.
_ALASS_NATIVE_FORMATS = frozenset({"", "srt", "subrip"})


def _is_explicit_hash_match(release: Any) -> bool:
    """True only when OpenSubtitles reported a MovieHash match for this result.

    ``OpenSubtitlesProvider`` sets both flags from
    ``attributes.moviehash_match is True``, and only when a ``moviehash`` was
    actually sent with the request. A search issued without hash metadata can
    therefore never manufacture a match.
    """
    return bool(
        getattr(release, "is_hash_match", False)
        or getattr(release, "matched_by_hash", False)
    )


def _selection_key(release: Any) -> tuple[int, int, str]:
    """Rank hash-matched candidates.

    Only secondary criteria appear here. The hash match has already been proven
    by the time this runs, so format, HI/SDH status and release name merely pick
    between equally valid references. Language is deliberately absent: the
    reference language must never influence anything.
    """
    fmt = str(getattr(release, "format", "") or "").lower()
    not_native = 0 if fmt in _ALASS_NATIVE_FORMATS else 1
    hearing_impaired = 1 if getattr(release, "hearing_impaired", False) else 0
    return (not_native, hearing_impaired, str(getattr(release, "release_name", "") or ""))


class OpenSubtitlesHashReferenceStrategy:
    """Resolve a byte-exact OpenSubtitles reference for the sync pipeline.

    Implements the same shape as the other reference strategies
    (``resolve_with_provenance`` returning a :class:`ResolvedReference`), so the
    orchestrator's existing loop drives it: the returned text flows into the
    unchanged ALASS invocation as the reference while the user's subtitle stays
    the target, and returning ``ResolvedReference(None)`` falls straight through
    to the next strategy.
    """

    name = "opensubtitles-hash"
    validates_target = True

    def __init__(
        self,
        provider: Any,
        *,
        cache: ReferenceDiskCache | None = None,
        timeout: float = _HASH_SEARCH_TIMEOUT,
        min_bytes: int = _MIN_REFERENCE_BYTES,
    ) -> None:
        self._provider = provider
        self.timeout = timeout
        self.min_bytes = min_bytes
        self._cache = cache

    @property
    def cache(self) -> ReferenceDiskCache:
        """Lazily built so constructing the strategy never touches the disk."""
        if self._cache is None:
            self._cache = ReferenceDiskCache(min_bytes=self.min_bytes)
        return self._cache

    async def resolve(self, query: ReferenceQuery) -> str | None:
        resolved = await self.resolve_with_provenance(query)
        return resolved.text

    async def resolve_with_provenance(
        self,
        query: ReferenceQuery,
        *,
        update_validator: Callable[[str], bool] | None = None,
    ) -> ResolvedReference:
        # 1. Hash metadata is mandatory. Without it an exact match cannot be
        #    proven, so do not claim one and do not spend quota on a guess.
        video_hash = str(query.video_hash or "").strip()
        if not video_hash:
            logger.info(
                "[hashref] request carries no video hash -> skipping OpenSubtitles "
                "reference lookup (no exact match can be proven)"
            )
            return ResolvedReference(None)

        api_key, username, password = self._credentials(query)

        # 2. No API key means the upstream search cannot succeed at all. Skip
        #    before the request rather than after.
        if not api_key:
            logger.info(
                "[hashref] no OpenSubtitles API key configured -> skipping reference lookup"
            )
            return ResolvedReference(None)

        # 3. Quota / rate limiting. HTTP 429 and HTTP 406 (quota exhausted) trip
        #    the shared breaker; honour it instead of retrying into a wall.
        if self._breaker_open():
            logger.info(
                "[hashref] OpenSubtitles circuit breaker open -> falling back to the "
                "existing reference strategies"
            )
            return ResolvedReference(None)

        # 4. Cache first: a reference proven for this exact video and target cue
        #    layout must not be downloaded again.
        cached = self.cache.get(query)
        if cached is not None and cached.text:
            if update_validator is None or update_validator(cached.text):
                logger.info(
                    "[hashref] reusing cached OpenSubtitles reference (%d bytes, "
                    "candidate=%s)",
                    len(cached.text.encode("utf-8")),
                    cached.candidate or "?",
                )
                return cached
            self.cache.delete(query)

        # 5. Search by hash and size. Languages is empty on purpose: the hash
        #    identifies the file, so the reference language is irrelevant.
        try:
            releases = await asyncio.wait_for(
                self._provider.search_subtitles(
                    imdb_id=query.imdb_id,
                    is_series=query.is_series,
                    season=query.season,
                    episode=query.episode,
                    title=query.title,
                    year=query.year,
                    api_key=api_key,
                    languages=[],
                    exclude_hi=False,
                    moviehash=video_hash,
                    moviebytesize=query.video_size,
                    username=username,
                    password=password,
                ),
                self.timeout,
            )
        except TimeoutError:
            logger.warning(
                "[hashref] OpenSubtitles hash search timed out after %.1fs -> fallback",
                self.timeout,
            )
            return ResolvedReference(None)
        except Exception as exc:  # noqa: BLE001 - one provider must never sink the request
            logger.warning("[hashref] OpenSubtitles hash search failed: %s", exc)
            return ResolvedReference(None)

        # 6. Strict acceptance. Anything the API did not explicitly mark as a
        #    hash match is discarded here and never downloaded.
        candidates = [r for r in (releases or []) if _is_explicit_hash_match(r)]
        rejected = len(releases or []) - len(candidates)
        if rejected:
            logger.info(
                "[hashref] discarded %d non-hash-matched OpenSubtitles result(s)",
                rejected,
            )
        if not candidates:
            logger.info(
                "[hashref] OpenSubtitles returned no result marked "
                "moviehash_match=true for hash=%s -> fallback",
                video_hash[:12],
            )
            return ResolvedReference(None)

        candidates.sort(key=_selection_key)
        best = candidates[0]
        release_name = str(getattr(best, "release_name", "") or "") or "?"
        logger.info(
            "[hashref] %d hash-matched candidate(s); selected %r (lang=%s, format=%s)",
            len(candidates),
            release_name,
            getattr(best, "lang", "?"),
            getattr(best, "format", "?"),
        )

        # 7. Download exactly one candidate. The full credential set is
        #    forwarded: an API key alone authenticates search, but the download
        #    endpoint exchanges a session token and is quota-metered per account.
        raw = await self._download(best, api_key, username, password)
        if not raw:
            logger.warning(
                "[hashref] download of hash-matched reference %r failed -> fallback",
                release_name,
            )
            return ResolvedReference(None)

        decoded = decode_payload(
            raw, release_name, query.season, query.episode, self.min_bytes
        )
        # decode_payload only applies min_bytes when picking a member out of an
        # archive; a bare SRT comes straight through. Enforce the floor here so a
        # truncated or error-page download cannot become a reference.
        if not decoded or len(decoded) <= self.min_bytes:
            logger.warning(
                "[hashref] downloaded reference %r was empty or under %d bytes "
                "(got %d) -> fallback",
                release_name,
                self.min_bytes,
                len(decoded or b""),
            )
            return ResolvedReference(None)

        # The reference is a timing input only. It is stored exactly as decoded:
        # no Arabic processing, no numeral conversion, no dialogue rewriting.
        # Format normalisation for ALASS already happens in the sync service.
        text = decoded.decode("utf-8", "replace")
        if query.season is not None and query.episode is not None:
            if looks_like_cumulative_pack(text):
                logger.warning(
                    "[hashref] reference %r looks like a cumulative season pack -> fallback",
                    release_name,
                )
                return ResolvedReference(None)

        if update_validator is not None and not update_validator(text):
            logger.warning(
                "[hashref] hash-matched reference %r failed cue sanity -> fallback",
                release_name,
            )
            return ResolvedReference(None)

        self.cache.set(
            query,
            "opensubtitles",
            text,
            kind="hash",
            candidate=release_name,
            partial=True,
        )
        logger.info(
            "[hashref] resolved byte-exact OpenSubtitles reference %r (%d bytes)",
            release_name,
            len(text.encode("utf-8")),
        )
        return ResolvedReference(
            text,
            kind="hash",
            candidate=release_name,
            partial=True,
            # P0-3: record WHY this reference is trustworthy. ReferenceTrust.STRONG
            # is defined as "Byte-exact hash match, or an exact release identity"
            # (reference.py), which is precisely the evidence here: OpenSubtitles
            # reported moviehash_match for the hash this request supplied.
            #
            # What this does NOT do is assert that the target's timings are
            # correct. AlignmentAnalyzer still measures every cue; reference trust
            # is only ever allowed to WITHHOLD a verified claim
            # (alignment.py:769-777), never to create one. A MovieHash match
            # proves current-video identity, not subtitle timing correctness, so
            # the timing verifier remains the sole authority on synchronisation.
            reference_trust=ReferenceTrust.STRONG.value,
        )

    def _credentials(self, query: ReferenceQuery) -> tuple[str, str, str]:
        """Resolve the effective key/account, request config winning over env.

        The settings attribute names are spelled out rather than derived from
        the query keys: the query uses ``opensubtitles`` while the setting is
        ``OPENSUBTITLES_API_KEY``, so an upper-cased lookup would silently find
        nothing and disable the whole path.
        """
        keys = query.api_keys or {}

        def pick(key: str, attr: str) -> str:
            return str(keys.get(key) or getattr(settings, attr, "") or "").strip()

        return (
            pick("opensubtitles", "OPENSUBTITLES_API_KEY"),
            pick("opensubtitles_username", "OPENSUBTITLES_USERNAME"),
            pick("opensubtitles_password", "OPENSUBTITLES_PASSWORD"),
        )

    def _breaker_open(self) -> bool:
        checker = getattr(self._provider, "is_breaker_open", None)
        try:
            return bool(checker()) if callable(checker) else False
        except Exception:  # noqa: BLE001 - never let telemetry sink the request
            return False

    async def _download(
        self, release: Any, api_key: str, username: str, password: str
    ) -> bytes | None:
        """Download the single selected reference, or ``None`` on any failure."""
        reference = getattr(release, "download_url", None)
        if not reference:
            return None
        try:
            return await self._provider.download_archive(
                reference,
                api_key=api_key,
                username=username,
                password=password,
            )
        except TypeError:
            # A provider or test double that predates the credential keywords.
            try:
                return await self._provider.download_archive(reference, api_key=api_key)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[hashref] reference download failed: %s", exc)
                return None
        except Exception as exc:  # noqa: BLE001 - fall back, never fail the request
            logger.warning("[hashref] reference download failed: %s", exc)
            return None
