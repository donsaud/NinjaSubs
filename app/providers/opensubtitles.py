"""OpenSubtitles.com v1 REST API subtitle provider integration.

Endpoint behaviour encoded here was verified against the current live API, not
assumed:

* ``GET /subtitles`` needs ``Api-Key``. ``moviehash`` (16 hex chars) may be sent
  on its own -- ``imdb_id`` is not required -- and the response then carries
  ``attributes.moviehash_match``, a real boolean, which is the ONLY thing this
  provider will ever treat as an exact match.
* ``POST /download`` needs ``Api-Key`` AND ``Authorization: Bearer <jwt>``. The
  payload is ``{"file_id": ...}`` taken from ``attributes.files[].file_id`` --
  never the top-level ``id``/``subtitle_id``, which is a different number.
* The login ``base_url`` is a bare host (``api.opensubtitles.com`` or
  ``vip-api.opensubtitles.com``), not a URL; see ``_normalise_base_url``.
* HTTP 406 is overloaded: it means download quota exhausted *or* "Invalid
  file_id". The two are told apart by the ``message`` field, because only the
  first deserves a long cooldown.
* HTTP 429 is a request-rate limit and carries ``Retry-After``.
* The JWT lives 24 hours with no refresh endpoint; re-``POST /login``.
* Error bodies are sometimes a JSON *array* (``[{"message": ...}]``), so
  ``body["message"]`` silently yields nothing on those.
"""

import logging
import re
import time
from datetime import UTC, datetime
from typing import Any

import httpx

from app.config import settings
from app.models import SubtitleRelease
from app.providers.base import BaseSubtitleProvider
from app.providers.opensubtitles_auth import OPENSUBTITLES_TOKENS, OpenSubtitlesToken
from app.utils.language import get_opensubtitles_lang_code, normalize_to_iso639_2
from app.utils.uploader import extract_uploader

logger = logging.getLogger("uvicorn.error")

#: Cooldown used when the API tells us nothing better.
_QUOTA_COOLDOWN_SECONDS = 3600.0
_RATE_LIMIT_COOLDOWN_SECONDS = 60.0

#: A 406 whose message looks like this is a bad request, not an exhausted quota.
#: Retrying it would burn a whole hour of the circuit breaker for nothing.
_INVALID_FILE_ID_MARKERS = ("invalid file_id", "invalid file id")

#: Matches the trailing ``file_id`` in our own ``/sub/opensubtitles/<id>.<ext>``
#: references. The extension set must cover every format this provider emits:
#: matching only ``.srt`` meant ASS/VTT/SSA references parsed to no id at all and
#: every such download silently failed.
_FILE_ID_RE = re.compile(r"(\d+)(?:\.(?:srt|ass|ssa|vtt|sub))?$")


def _error_payload(resp: httpx.Response) -> dict[str, Any]:
    """Return an error body as a dict whether the API sent an object or an array.

    OpenSubtitles is inconsistent here: 401s arrive as ``[{"message": ...}]``
    while quota 406s arrive as a bare object. Reading ``json()["message"]``
    directly therefore fails silently on the array form and the real reason for
    the failure is lost.
    """
    try:
        data = resp.json()
    except Exception:
        return {}
    if isinstance(data, list):
        merged: dict[str, Any] = {}
        for item in data:
            if isinstance(item, dict):
                for key, value in item.items():
                    merged.setdefault(key, value)
        return merged
    return data if isinstance(data, dict) else {}


def _cooldown_from_retry_after(resp: httpx.Response, fallback: float) -> float:
    """Honour ``Retry-After`` (seconds or HTTP-date) before guessing."""
    raw = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
    if not raw:
        return fallback
    raw = str(raw).strip()
    try:
        return max(1.0, min(float(raw), 86400.0))
    except (TypeError, ValueError):
        pass
    try:
        when = datetime.strptime(raw, "%a, %d %b %Y %H:%M:%S %Z").replace(
            tzinfo=UTC
        )
    except (TypeError, ValueError):
        return fallback
    return max(1.0, min((when - datetime.now(UTC)).total_seconds(), 86400.0))


def _quota_reset_seconds(payload: dict[str, Any]) -> float | None:
    """Seconds until the quota resets, from the documented ``reset_time_utc``.

    The quota response documents ``reset_time_utc`` (ISO-8601) and a human
    ``reset_time``. There is no ``reset_time_unix`` field, so a client looking
    for one silently falls back to its fixed default and can retry long after the
    quota was restored. Both spellings are accepted; the ISO form is preferred
    because it is unambiguous.
    """
    iso = payload.get("reset_time_utc") or payload.get("resetTimeUtc")
    if iso:
        try:
            when = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
            if when.tzinfo is None:
                when = when.replace(tzinfo=UTC)
            delta = (when - datetime.now(UTC)).total_seconds()
            return max(60.0, min(delta, 86400.0))
        except (TypeError, ValueError):
            pass
    raw = payload.get("reset_time_unix")
    if raw:
        try:
            return max(60.0, min(float(raw) - time.time(), 86400.0))
        except (TypeError, ValueError):
            pass
    return None


def _is_quota_406(payload: dict[str, Any]) -> bool:
    """True when a 406 really is quota exhaustion rather than a bad request."""
    message = str(payload.get("message") or "").strip().lower()
    if any(marker in message for marker in _INVALID_FILE_ID_MARKERS):
        return False
    # A quota 406 always carries the remaining counter or a reset hint.
    return bool(
        message
        or payload.get("remaining") is not None
        or payload.get("reset_time_utc")
        or payload.get("reset_time")
    )


def looks_like_subtitle_payload(content: bytes) -> bool:
    """Cheap guard that a download is subtitle text and not an error page.

    A quota wall or gateway hiccup can return HTML or JSON with HTTP 200. Such a
    body would otherwise be handed to the sync pipeline as if it were a
    subtitle, so it is rejected here at the provider boundary.
    """
    if not content or not content.strip():
        return False
    head = content[:512].lstrip().lower()
    if head.startswith((b"<", b"<!doctype", b"<?xml", b"<html")):
        return False
    if b"--> " in content[:4096] or b"-->" in content[:4096]:
        return True
    # A bare SRT cue index plus a timestamp is the minimum viable subtitle.
    return bool(re.search(rb"\d{2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->", content[:4096]))


class OpenSubtitlesCircuitBreaker:
    """Process-wide cooldown after OpenSubtitles rate-limits (HTTP 429) or quota exhaustion (HTTP 406)."""

    def __init__(self, default_cooldown: float = 3600.0) -> None:
        self.default_cooldown = default_cooldown
        self._open_until = 0.0
        self._reason = ""

    def trip(self, cooldown: float | None = None, reason: str = "") -> None:
        duration = cooldown if cooldown is not None else self.default_cooldown
        self._open_until = time.monotonic() + duration
        self._reason = reason
        logger.warning(
            "[OpenSubtitles] Circuit breaker tripped (%s); skipping OpenSubtitles downloads for %.0fs",
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


class OpenSubtitlesProvider(BaseSubtitleProvider):
    """Subtitle provider implementation for OpenSubtitles.com v1 REST API."""

    name = "opensubtitles"
    BASE_URL = "https://api.opensubtitles.com/api/v1"
    USER_AGENT = "StremioArabicSubs v1.0.0"

    def is_breaker_open(self) -> bool:
        return OPENSUBTITLES_BREAKER.is_open()

    def _get_headers(self, api_key: str, token: str | None = None) -> dict[str, str]:
        """Request headers for the v1 API.

        ``Api-Key`` identifies the application and is mandatory on every call.
        ``Authorization`` carries the user JWT when one has been obtained; the
        API recommends sending it on every request once authenticated, and a VIP
        account's alternate host rejects requests that omit it.
        """
        headers = {
            "User-Agent": self.USER_AGENT,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if api_key:
            headers["Api-Key"] = api_key
        if token:
            headers["Authorization"] = token
        return headers

    def _resolve_api_key(self, api_key: str | None) -> str:
        return (api_key or getattr(settings, "OPENSUBTITLES_API_KEY", "") or "").strip()

    def _resolve_credentials(
        self,
        api_key: str | None,
        username: str | None = None,
        password: str | None = None,
    ) -> tuple[str, str, str]:
        """Effective ``(api_key, username, password)`` for this request.

        Per-request values win over the environment, which is what lets a user
        supply their own account on a shared instance. The API key stays
        mandatory: it identifies the consumer and cannot be replaced by a user
        login.
        """
        return (
            self._resolve_api_key(api_key),
            (username or getattr(settings, "OPENSUBTITLES_USERNAME", "") or "").strip(),
            password or getattr(settings, "OPENSUBTITLES_PASSWORD", "") or "",
        )

    async def _authenticate(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        username: str,
        password: str,
    ) -> OpenSubtitlesToken | None:
        """Fetch (or reuse) a user JWT. ``None`` means carry on unauthenticated."""
        if not username or not password:
            return None
        return await OPENSUBTITLES_TOKENS.get_token(
            client,
            api_key=api_key,
            username=username,
            password=password,
            base_url=self.BASE_URL,
            user_agent=self.USER_AGENT,
        )

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
        video_hash: str | None = None,
        video_size: int | str | None = None,
        moviehash: str | None = None,
        moviebytesize: int | str | None = None,
        username: str | None = None,
        password: str | None = None,
        **kwargs,
    ) -> list[SubtitleRelease]:
        """
        Query OpenSubtitles.com v1 REST API for subtitles by IMDb ID and optional MovieHash.
        Gracefully skips if no API key is configured.
        """
        effective_key, effective_user, effective_pass = self._resolve_credentials(
            api_key, username, password
        )
        if not effective_key:
            logger.info("OpenSubtitles API key is not configured. Skipping OpenSubtitles search.")
            return []

        # Clean IMDb ID: strip compound series markers (:s:e) and leading "tt"
        clean_imdb = str(imdb_id).split(":")[0].strip()
        try:
            numeric_id = str(int(re.sub(r"[^0-9]", "", clean_imdb)))
        except (ValueError, TypeError):
            numeric_id = clean_imdb.replace("tt", "").lstrip("0") or "0"

        # Languages mapped from ISO-639-2 to ISO-639-1 (e.g. "ara" -> "ar")
        #
        # ``languages=[]`` explicitly requests NO language filter. The MovieHash
        # reference path uses that: the hash already identifies the exact file,
        # so which language the reference happens to be in is irrelevant to it
        # and filtering would only discard usable references. ``None`` keeps the
        # historical Arabic default the normal provider path relies on.
        if languages is not None and len(languages) == 0:
            mapped_langs: list[str] = []
        elif languages:
            mapped_langs = list(
                dict.fromkeys([get_opensubtitles_lang_code(lang) for lang in languages])
            )
        else:
            mapped_langs = ["ara"]

        # Resolve the MovieHash before building params, because it decides the
        # shape of the whole query.
        v_hash = (
            moviehash
            or video_hash
            or kwargs.get("videoHash")
            or kwargs.get("moviehash")
            or kwargs.get("video_hash")
        )
        v_size = (
            moviebytesize
            or video_size
            or kwargs.get("videoSize")
            or kwargs.get("moviebytesize")
            or kwargs.get("video_size")
        )
        v_hash = str(v_hash).strip() if v_hash else ""

        params: dict[str, Any] = {}

        # A MovieHash identifies one exact file. Verified against the live API:
        #
        #   ?moviehash=H                       -> 2 rows, moviehash_match=true on both
        #   ?moviehash=H&imdb_id=X            -> 50 rows, moviehash_match=false on ALL
        #   ?moviehash=H&season_number=1      -> 0 rows
        #   ?moviehash=H&moviebytesize=S      -> 2 rows, moviehash_match=true
        #   ?moviehash=H&type=movie           -> 2 rows, moviehash_match=true
        #
        # Adding imdb_id makes the API quietly fall back to a plain IMDb search and
        # return entirely unrelated rows with moviehash_match=false, which looks
        # like a successful query but can never be an exact match. Sending
        # imdb_id and season/episode alongside the hash therefore silently
        # disabled the whole AutoSync reference path against the real service --
        # while every mocked test still passed, because the trap is a property of
        # the API, not of the code under test.
        #
        # So: with a hash present the hash alone decides. imdb_id and
        # season/episode are omitted, which is also what the reference path
        # requires -- identifiers must not stand in for hash verification.
        if v_hash:
            params["moviehash"] = v_hash
            if v_size:
                params["moviebytesize"] = str(v_size).strip()
            params["type"] = "episode" if is_series else "movie"
        else:
            params["imdb_id"] = numeric_id
            params["type"] = "episode" if is_series else "movie"
            if is_series:
                if season is not None:
                    params["season_number"] = int(season)
                if episode is not None:
                    params["episode_number"] = int(episode)

        if mapped_langs:
            params["languages"] = ",".join(mapped_langs)

        if exclude_hi:
            params["hearing_impaired"] = "exclude"

        # A user JWT is optional here -- search is unlimited without one -- but
        # sending it is recommended by the API and required for a VIP host, and
        # it is what raises the download quota the results will later draw on.
        token = await self._authenticate(
            self.client, effective_key, effective_user, effective_pass
        )
        headers = self._get_headers(effective_key, token.token if token else None)
        base_url = token.base_url if token else self.BASE_URL
        url = f"{base_url}/subtitles"

        try:
            logger.info(f"[OpenSubtitles Request] Outbound URL: {url} | Params: {params}")
            resp = await self.client.get(
                url,
                params=params,
                headers=headers,
                timeout=settings.UPSTREAM_TIMEOUT,
                follow_redirects=True,
            )

            if resp.status_code == 200:
                try:
                    data = resp.json()
                except Exception as json_err:
                    logger.warning(f"[OpenSubtitles] JSON decode error: {json_err}")
                    return []

                raw_items = data.get("data", []) if isinstance(data, dict) else []
                logger.info(f"[OpenSubtitles Response] Total items received: {len(raw_items)}")

                results: list[SubtitleRelease] = []
                for item in raw_items:
                    if not isinstance(item, dict):
                        continue
                    attributes = item.get("attributes", {})
                    files = attributes.get("files", [])
                    file_id = None
                    file_name = None
                    if isinstance(files, list) and files:
                        file_id = files[0].get("file_id")
                        file_name = files[0].get("file_name")

                    if not file_id:
                        continue

                    # Extract release name
                    release_name = (
                        attributes.get("release")
                        or file_name
                        or f"{imdb_id}.OpenSubtitles.{file_id}"
                    )
                    release_name = str(release_name).strip()

                    # Extract language. Never assume Arabic: the API answers in
                    # the subtitle's own language, and defaulting to "ara" here
                    # would mislabel a non-Arabic result as Arabic.
                    raw_lang = attributes.get("language") or ""
                    norm_lang = normalize_to_iso639_2(raw_lang, default="und")

                    # Hearing impaired handling
                    is_hi = bool(attributes.get("hearing_impaired"))
                    if exclude_hi and is_hi:
                        continue

                    # Uploader/author username (e.g. attributes.uploader.name)
                    uploader = extract_uploader(attributes)

                    is_hash_match = (
                        bool(params.get("moviehash")) and attributes.get("moviehash_match") is True
                    )

                    file_name_lower = str(file_name or "").lower()
                    sub_fmt = "srt"
                    if file_name_lower.endswith(".ass"):
                        sub_fmt = "ass"
                    elif file_name_lower.endswith(".ssa"):
                        sub_fmt = "ssa"
                    elif file_name_lower.endswith(".vtt"):
                        sub_fmt = "vtt"
                    elif attributes.get("format"):
                        fmt_attr = str(attributes.get("format")).lower()
                        if fmt_attr in ("ass", "ssa", "vtt", "srt"):
                            sub_fmt = fmt_attr

                    url = f"/sub/opensubtitles/{file_id}.{sub_fmt}"
                    results.append(
                        SubtitleRelease(
                            release_name=release_name,
                            download_url=url,
                            provider="opensubtitles",
                            format=sub_fmt,
                            hearing_impaired=is_hi,
                            lang=norm_lang,
                            is_hash_match=is_hash_match,
                            matched_by_hash=is_hash_match,
                            uploader=uploader,
                        )
                    )

                logger.info(f"[OpenSubtitles Response] Total items found: {len(results)}")
                return results

            elif resp.status_code == 401:
                # 401 here means the bearer was missing, expired or revoked --
                # the JWT is good for 24h and there is no refresh endpoint. Drop
                # it so the next attempt logs in again instead of replaying a
                # dead token until its TTL elapses.
                OPENSUBTITLES_TOKENS.invalidate(
                    effective_key, effective_user, effective_pass
                )
                logger.warning(
                    "[OpenSubtitles] Authentication rejected (HTTP 401); cached token "
                    "dropped so the next request re-authenticates"
                )
                return []
            elif resp.status_code == 403:
                # 403 is gateway-level: the Api-Key itself is missing or invalid.
                # Re-authenticating cannot help, so the token is left alone.
                logger.warning(
                    "[OpenSubtitles] Request refused (HTTP 403); check "
                    "OPENSUBTITLES_API_KEY."
                )
                return []
            elif resp.status_code == 429:
                # Request-rate limit (5/s on the free tiers). Trip the shared
                # breaker so the other OpenSubtitles call sites stop hammering
                # it too, rather than each rediscovering the limit independently.
                delay = _cooldown_from_retry_after(resp, _RATE_LIMIT_COOLDOWN_SECONDS)
                OPENSUBTITLES_BREAKER.trip(delay, reason="HTTP 429")
                logger.warning("[OpenSubtitles] Rate limit reached (HTTP 429); cooling down.")
                return []
            else:
                logger.warning(
                    f"[OpenSubtitles] Unexpected HTTP {resp.status_code} for {imdb_id}: {resp.text[:300]}"
                )
                return []

        except httpx.TimeoutException:
            logger.warning(f"[OpenSubtitles] Query timed out for {imdb_id}")
            return []
        except Exception as e:
            logger.error(f"[OpenSubtitles] Search error for {imdb_id}: {e}", exc_info=True)
            return []

    async def get_download_url(
        self,
        file_id: int,
        api_key: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ) -> str | None:
        """
        Request temporary download link for subtitle file_id via POST /api/v1/download.

        This is one of the two endpoints OpenSubtitles documents as requiring
        user authentication, so the JWT is not optional in practice: without it
        the call falls back to the anonymous quota (5/day) instead of the
        account's. Search does not have this requirement, which is why the
        credential only becomes load-bearing here.
        """
        if OPENSUBTITLES_BREAKER.is_open():
            logger.info(
                "[OpenSubtitles] Circuit breaker open (%.0fs remaining); skipping download request for file %s",
                OPENSUBTITLES_BREAKER.remaining,
                file_id,
            )
            return None

        effective_key, effective_user, effective_pass = self._resolve_credentials(
            api_key, username, password
        )
        if not effective_key:
            logger.error("[OpenSubtitles Download Fail] Missing API key")
            return None

        client = (
            self.client
            if self.client is not None
            else httpx.AsyncClient(timeout=10.0, follow_redirects=True)
        )
        should_close = self.client is None
        try:
            token = await self._authenticate(
                client, effective_key, effective_user, effective_pass
            )
            headers = self._get_headers(effective_key, token.token if token else None)
            base_url = token.base_url if token else self.BASE_URL
            res = await client.post(
                f"{base_url}/download",
                headers=headers,
                json={"file_id": int(file_id)},
                follow_redirects=True,
            )
            if res.status_code in (200, 201):
                return res.json().get("link")

            if res.status_code == 401 and token is not None:
                # A cached token can be revoked or expire early. Drop it so the
                # next attempt logs in again rather than reusing a dead token.
                OPENSUBTITLES_TOKENS.invalidate(
                    effective_key, effective_user, effective_pass
                )
                logger.warning(
                    "[OpenSubtitles Download Fail] Cached token rejected; will re-authenticate"
                )

            if res.status_code in (406, 429):
                payload = _error_payload(res)
                if res.status_code == 406 and not _is_quota_406(payload):
                    # Overloaded status: this is "Invalid file_id", not an
                    # exhausted quota. Cooling down for an hour would disable
                    # OpenSubtitles for the whole process over one bad id.
                    logger.error(
                        "[OpenSubtitles Download Fail] HTTP 406 with message %r -- "
                        "treating as an invalid file_id, not quota exhaustion",
                        str(payload.get("message") or "")[:120],
                    )
                    return None

                fallback = (
                    _QUOTA_COOLDOWN_SECONDS
                    if res.status_code == 406
                    else _RATE_LIMIT_COOLDOWN_SECONDS
                )
                if res.status_code == 429:
                    delay = _cooldown_from_retry_after(res, fallback)
                else:
                    delay = _quota_reset_seconds(payload) or fallback
                OPENSUBTITLES_BREAKER.trip(delay, reason=f"HTTP {res.status_code}")

            logger.error(
                f"[OpenSubtitles Download Fail] Status: {res.status_code} | {res.text[:200]}"
            )
            return None
        except Exception as e:
            logger.error(f"[OpenSubtitles Download Fail] Exception: {e}")
            return None
        finally:
            if should_close:
                await client.aclose()

    async def download_archive(
        self,
        download_ref: str,
        api_key: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ) -> bytes | None:
        """
        Download subtitle file from OpenSubtitles.
        Resolves direct download URL via get_download_url using file_id.
        """
        if OPENSUBTITLES_BREAKER.is_open():
            logger.info(
                "[OpenSubtitles Download] Circuit breaker open (%.0fs remaining); skipping download for %s",
                OPENSUBTITLES_BREAKER.remaining,
                download_ref,
            )
            return None

        # Resolve the full credential set. Discarding username/password here
        # meant that credentials configured only in settings were dropped and the
        # download silently fell back to the anonymous per-IP quota (5/24h)
        # instead of the account's.
        effective_key, effective_user, effective_pass = self._resolve_credentials(
            api_key, username, password
        )

        # Cover every format this provider emits; matching only ".srt" left
        # ASS/VTT/SSA references with no parseable id and every such download
        # failed quietly.
        m = _FILE_ID_RE.search(str(download_ref).strip())
        file_id = int(m.group(1)) if m else None

        direct_link: str | None = None
        if file_id and effective_key:
            direct_link = await self.get_download_url(
                file_id,
                effective_key,
                username=effective_user,
                password=effective_pass,
            )

        target_url = direct_link or (download_ref if str(download_ref).startswith("http") else None)
        if not target_url:
            logger.warning(
                f"[OpenSubtitles Download] No valid download URL resolved for {download_ref}"
            )
            return None

        client = (
            self.client
            if self.client is not None
            else httpx.AsyncClient(timeout=settings.UPSTREAM_TIMEOUT, follow_redirects=True)
        )
        should_close = self.client is None
        try:
            logger.info(f"[OpenSubtitles Download] Downloading content from {target_url}")
            dl_resp = await client.get(target_url, follow_redirects=True)
            if dl_resp.status_code == 200 and dl_resp.content:
                if not looks_like_subtitle_payload(dl_resp.content):
                    # The `link` is unauthenticated and tokenised, so a stale or
                    # intercepted one can serve HTML or JSON with HTTP 200.
                    # Returning that would hand an error page to the pipeline as
                    # if it were a subtitle.
                    logger.warning(
                        "[OpenSubtitles Download] Response was not subtitle content "
                        "(%d bytes, content-type=%s); discarding",
                        len(dl_resp.content),
                        dl_resp.headers.get("content-type", "?"),
                    )
                    return None
                return dl_resp.content
            logger.warning(
                f"[OpenSubtitles Download] Download failed with HTTP {dl_resp.status_code}"
            )
            return None
        except Exception as dl_err:
            logger.warning(
                f"[OpenSubtitles Download] Error downloading content from {target_url}: {dl_err}"
            )
            return None
        finally:
            if should_close:
                await client.aclose()
