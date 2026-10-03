"""OpenSubtitles via Stremio's keyless v3 add-on endpoint.

Why this endpoint
-----------------
The official ``api.opensubtitles.com`` REST API needs an ``Api-Key``, and its
``/download`` endpoint additionally needs a user JWT -- so a working install
required the user to paste a credential, and downloads were capped at the
anonymous quota (5/day) without an account.

``opensubtitles-v3.strem.io`` is the same catalogue exposed as a Stremio
add-on. It needs no key, and every result carries a **direct download URL**, so
the authenticated ``/download`` round-trip disappears entirely. Nothing in this
module sends a credential, and none is accepted.

Contract, verified against the live endpoint
--------------------------------------------
``GET /subtitles/{type}/{id}/{extra}.json`` returns
``{"subtitles": [{id, url, lang, subtitleFileName, movieReleaseName,
releaseGroup, releaseFormat, SubEncoding, fpsMilli, season, episode, ...}]}``.

Two behaviours worth knowing, both confirmed by probing rather than assumed:

* **The endpoint does not filter.** It returns every language for the title
  regardless of ``extra``, and ignores ``language=`` in the query string. So the
  language filter is applied here, client-side, against the requested set.
* **It does not filter on the hash either.** A request carrying ``videoHash`` +
  ``videoSize`` returned the same 95 objects as a bare IMDb request, and the
  response carries no hash field to confirm a match. The stream parameters are
  still passed through, because the endpoint is free to start honouring them and
  a narrower result set is strictly better. But no hash match is *claimed*
  here: :attr:`SubtitleRelease.is_hash_match` stays False, because an unverified
  claim would promote a subtitle to the top tier on evidence that does not
  exist. Hash-tier ranking therefore remains available for providers that can
  actually prove a match.
"""

from __future__ import annotations

import logging
import re
import time
import urllib.parse
from typing import Any

import httpx

from app.config import settings
from app.models import SubtitleRelease
from app.providers.base import BaseSubtitleProvider
from app.utils.language import normalize_to_iso639_2
from app.utils.uploader import extract_uploader

logger = logging.getLogger("uvicorn.error")


class OpenSubtitlesCircuitBreaker:
    """Process-wide cooldown after the upstream rate-limits (HTTP 429) or fails."""

    def __init__(self, default_cooldown: float = 3600.0) -> None:
        self.default_cooldown = default_cooldown
        self._open_until = 0.0
        self._reason = ""

    def trip(self, cooldown: float | None = None, reason: str = "") -> None:
        duration = cooldown if cooldown is not None else self.default_cooldown
        self._open_until = time.monotonic() + duration
        self._reason = reason
        logger.warning(
            "[OpenSubtitles] Circuit breaker tripped (%s); skipping OpenSubtitles requests for %.0fs",
            reason or "Quota/RateLimit",
            duration,
        )

    def is_open(self) -> bool:
        return time.monotonic() < self._open_until

    @property
    def remaining(self) -> float:
        return max(0.0, self._open_until - time.monotonic())

    def reset(self) -> None:
        self._open_until = 0.0
        self._reason = ""


OPENSUBTITLES_BREAKER = OpenSubtitlesCircuitBreaker()

#: Maps an upstream ISO-639-2 code onto the code the endpoint actually returns,
#: for the cases where they differ. Anything absent falls through to a
#: normalised comparison, so this is an optimisation for correctness rather than
#: a lookup table that has to be kept exhaustive.
_LANG_ALIASES = {
    "zho": "zho",
    "chi": "zho",
    "srp": "srp",
    "slv": "slv",
    "may": "may",
    "msa": "may",
}

#: ``hearing_impaired`` is not a field in the v3 payload. These markers are what
#: the upstream actually uses in release names for SDH tracks, so that is where
#: the distinction has to come from.
_HI_MARKERS = re.compile(r"(?i)\b(sdh|hi\b|hearing\s*impaired|for\s+the\s+deaf)")

#: Extensions the endpoint serves. Used to label the format when the release
#: name does not say.
_FORMAT_BY_SUFFIX = {
    "srt": "srt",
    "sub": "sub",
    "ass": "ass",
    "ssa": "ssa",
    "vtt": "vtt",
    "smi": "smi",
    "ttml": "ttml",
}


def _normalize_language(code: str | None) -> str:
    """Upstream code -> the ISO-639-2 form this codebase filters on."""
    raw = (code or "").strip().lower()
    if not raw:
        return ""
    return _LANG_ALIASES.get(raw) or normalize_to_iso639_2(raw, default=raw)


class OpenSubtitlesProvider(BaseSubtitleProvider):
    """Subtitle provider backed by Stremio's keyless OpenSubtitles add-on."""

    name = "opensubtitles"
    BASE_URL = "https://opensubtitles-v3.strem.io"
    USER_AGENT = "StremioArabicSubs v1.0.0"

    def is_breaker_open(self) -> bool:
        return OPENSUBTITLES_BREAKER.is_open()

    def _get_headers(self) -> dict[str, str]:
        """No credential header.

        The endpoint is keyless by design. Accepting an ``api_key`` argument here
        would be actively misleading, so none is taken.
        """
        return {
            "User-Agent": self.USER_AGENT,
            "Accept": "application/json",
        }

    @staticmethod
    def _build_extra(
        *,
        video_hash: str | None,
        video_size: int | str | None,
        filename: str | None,
        extra_params: dict[str, Any] | None,
    ) -> str:
        """URL-encoded extra segment for the path.

        Stremio's convention puts the query string in the path, URL-encoded as a
        single segment: ``videoSize%3D123%26filename%3Dx``. ``urlencode`` alone
        is not enough -- it escapes the values but leaves ``=`` and ``&``
        literal, which would split the request's own query string at the ``&``
        and silently drop everything after it.
        """
        params: dict[str, str] = {}
        if filename:
            params["filename"] = str(filename)
        if video_size not in (None, ""):
            params["videoSize"] = str(video_size)
        if video_hash:
            params["videoHash"] = str(video_hash)
        for key, value in (extra_params or {}).items():
            if value in (None, ""):
                continue
            params[str(key)] = str(value)
        if not params:
            return ""
        return urllib.parse.quote(urllib.parse.urlencode(params), safe="")

    def _build_url(
        self,
        *,
        media_type: str,
        imdb_id: str,
        extra: str,
        query: dict[str, Any] | None,
    ) -> str:
        path = f"/subtitles/{media_type}/{imdb_id}"
        if extra:
            path += f"/{extra}"
        url = f"{self.BASE_URL}{path}.json"
        # Anything the caller could not fit in `extra` still goes on the query
        # string, which the endpoint accepts and ignores.
        if query:
            url += "?" + urllib.parse.urlencode(
                {k: v for k, v in query.items() if v not in (None, "")}
            )
        return url

    @staticmethod
    def _looks_hi(*texts: str | None) -> bool:
        return any(bool(t and _HI_MARKERS.search(t)) for t in texts)

    @staticmethod
    def _format_for(name: str) -> str:
        suffix = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        return _FORMAT_BY_SUFFIX.get(suffix, "srt")

    async def search_subtitles(
        self,
        imdb_id: str,
        is_series: bool = False,
        season: int | None = None,
        episode: int | None = None,
        title: str | None = None,
        year: int | None = None,
        api_key: str | None = None,
        languages: list[str] | None = None,
        exclude_hi: bool = False,
        *,
        video_hash: str | None = None,
        video_size: int | str | None = None,
        filename: str | None = None,
        extra_params: dict[str, Any] | None = None,
        **kwargs,
    ) -> list[SubtitleRelease]:
        """Query the keyless v3 endpoint and filter the result client-side.

        ``api_key`` is part of the base-class signature and is accepted purely so
        a caller that still passes one does not fail. It is ignored, with a log
        line: the credential is no longer meaningful, and refusing would turn a
        working provider into an error for anyone whose saved config still
        carries a key from before the move to v3.
        """
        if api_key:
            logger.info(
                "[OpenSubtitles] Ignoring a configured API key: this provider is keyless."
            )

        # Legacy aliases, still honoured so older call sites keep working.
        v_hash = video_hash or kwargs.get("moviehash") or kwargs.get("videoHash")
        v_size = video_size or kwargs.get("moviebytesize") or kwargs.get("videoSize")

        clean_imdb = str(imdb_id).split(":")[0].strip()
        media_type = "series" if is_series else "movie"

        # The breaker is consulted before building anything. Without this a
        # rate-limited upstream keeps absorbing a request per subtitle lookup for
        # the whole cooldown.
        if OPENSUBTITLES_BREAKER.is_open():
            logger.info(
                "[OpenSubtitles] Circuit breaker open (%.0fs remaining); skipping search for %s",
                OPENSUBTITLES_BREAKER.remaining,
                clean_imdb,
            )
            return []

        query: dict[str, Any] = {}
        if is_series:
            if season is not None:
                query["season"] = season
            if episode is not None:
                query["episode"] = episode

        extra = self._build_extra(
            video_hash=v_hash,
            video_size=v_size,
            filename=filename,
            extra_params=extra_params,
        )
        url = self._build_url(
            media_type=media_type, imdb_id=clean_imdb, extra=extra, query=query
        )

        wanted = {
            _normalize_language(code) for code in (languages or ["ara"])
        }
        wanted.discard("")

        try:
            logger.info(f"[OpenSubtitles Request] Outbound URL: {url}")
            resp = await self.client.get(
                url,
                headers=self._get_headers(),
                timeout=settings.UPSTREAM_TIMEOUT,
                follow_redirects=True,
            )
        except Exception as exc:
            logger.error(
                f"[OpenSubtitles] Request error for {imdb_id}: {type(exc).__name__}"
            )
            return []

        if resp.status_code == 429:
            OPENSUBTITLES_BREAKER.trip(60.0, reason="HTTP 429")
            return []
        if resp.status_code != 200:
            logger.warning(
                f"[OpenSubtitles] Unexpected status {resp.status_code} for {clean_imdb}"
            )
            return []

        try:
            payload = resp.json()
        except Exception as exc:
            logger.warning(f"[OpenSubtitles] JSON decode error: {type(exc).__name__}")
            return []

        raw_items = payload.get("subtitles") if isinstance(payload, dict) else None
        if not isinstance(raw_items, list):
            return []
        logger.info(f"[OpenSubtitles Response] Total items received: {len(raw_items)}")

        results: list[SubtitleRelease] = []
        skipped_language = 0
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            direct_url = str(item.get("url") or "").strip()
            if not direct_url.lower().startswith(("http://", "https://")):
                continue

            norm_lang = _normalize_language(item.get("lang"))
            if wanted and norm_lang not in wanted:
                # The endpoint returns every language for the title, so this is
                # where the language filter actually happens.
                skipped_language += 1
                continue

            file_name = str(item.get("subtitleFileName") or "").strip()
            release_name = (
                str(item.get("movieReleaseName") or "").strip()
                or file_name
                or f"{clean_imdb}.OpenSubtitles.{item.get('id', '')}".strip(".")
            )

            if exclude_hi and self._looks_hi(release_name, file_name):
                continue

            results.append(
                SubtitleRelease(
                    release_name=release_name,
                    # The v3 endpoint's whole point: a direct, credential-free URL.
                    download_url=direct_url,
                    provider=self.name,
                    lang=norm_lang or "ara",
                    format=self._format_for(file_name or release_name),
                    uploader=extract_uploader(release_name),
                    # No hash match is claimed. The endpoint does not filter on
                    # the hash and does not report one, so asserting a match would
                    # promote this to the hash tier on evidence that is absent.
                    is_hash_match=False,
                    matched_by_hash=False,
                )
            )

        logger.info(
            "[OpenSubtitles] %d result(s) after filtering (skipped %d for language, "
            "requested %s)",
            len(results),
            skipped_language,
            ",".join(sorted(wanted)) or "any",
        )
        return results

    async def download_archive(
        self,
        download_ref: str,
        api_key: str | None = None,
        **kwargs,
    ) -> bytes | None:
        """Fetch a subtitle from the direct URL the search returned.

        ``download_ref`` may be that URL, or an internal ``/sub/opensubtitles/...``
        reference; both are resolved here. ``api_key`` is accepted and ignored.
        """
        if OPENSUBTITLES_BREAKER.is_open():
            logger.info(
                "[OpenSubtitles Download] Circuit breaker open (%.0fs remaining); skipping %s",
                OPENSUBTITLES_BREAKER.remaining,
                download_ref,
            )
            return None

        target_url = self._resolve_download_url(download_ref)
        if not target_url:
            logger.warning(
                f"[OpenSubtitles Download] No usable URL for reference {download_ref}"
            )
            return None

        client = (
            self.client
            if self.client is not None
            else httpx.AsyncClient(timeout=settings.UPSTREAM_TIMEOUT, follow_redirects=True)
        )
        should_close = self.client is None
        try:
            resp = await client.get(
                target_url,
                headers={"User-Agent": self.USER_AGENT},
                follow_redirects=True,
                timeout=settings.UPSTREAM_TIMEOUT,
            )
            if resp.status_code == 200 and resp.content:
                return resp.content
            if resp.status_code == 429:
                OPENSUBTITLES_BREAKER.trip(60.0, reason="HTTP 429")
            logger.warning(
                f"[OpenSubtitles Download] Status {resp.status_code} for {target_url}"
            )
            return None
        except Exception as exc:
            logger.error(f"[OpenSubtitles Download] Exception: {type(exc).__name__}")
            return None
        finally:
            if should_close:
                await client.aclose()

    def _resolve_download_url(self, download_ref: str) -> str | None:
        """Turn whatever the caller holds into a fetchable URL."""
        ref = str(download_ref or "").strip()
        if ref.lower().startswith(("http://", "https://")):
            return ref
        # An internal reference carries the upstream URL itself. Older cached
        # entries stored a bare numeric file_id, which the v3 endpoint cannot
        # resolve at all -- returning None makes that a clean miss rather than a
        # request to the wrong host.
        m = re.fullmatch(r"/sub/opensubtitles/([^/]+)\.srt", ref)
        if m:
            candidate = urllib.parse.unquote(m.group(1))
            if candidate.lower().startswith(("http://", "https://")):
                return candidate
        return None
