"""OpenSubtitles.com v1 REST API subtitle provider integration."""

import logging
import re
import time
from typing import Any

import httpx

from app.config import settings
from app.models import SubtitleRelease
from app.providers.base import BaseSubtitleProvider
from app.providers.opensubtitles_auth import OPENSUBTITLES_TOKENS, OpenSubtitlesToken
from app.utils.language import get_opensubtitles_lang_code, normalize_to_iso639_2
from app.utils.uploader import extract_uploader

logger = logging.getLogger("uvicorn.error")


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
        target_langs = languages or ["ara"]
        mapped_langs = list(dict.fromkeys([get_opensubtitles_lang_code(lang) for lang in target_langs]))

        params: dict[str, Any] = {
            "imdb_id": numeric_id,
            "languages": ",".join(mapped_langs) if mapped_langs else "ar",
            "type": "episode" if is_series else "movie",
        }

        if is_series:
            if season is not None:
                params["season_number"] = int(season)
            if episode is not None:
                params["episode_number"] = int(episode)

        if exclude_hi:
            params["hearing_impaired"] = "exclude"

        # Support OpenSubtitles MovieHash matching
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

        if v_hash:
            params["moviehash"] = str(v_hash).strip()
        if v_size:
            params["moviebytesize"] = str(v_size).strip()

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

                    # Extract language
                    raw_lang = attributes.get("language") or "ar"
                    norm_lang = normalize_to_iso639_2(raw_lang, default="ara")

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

            elif resp.status_code in (401, 403):
                logger.warning(
                    f"[OpenSubtitles] Authentication failed (HTTP {resp.status_code}). Check OPENSUBTITLES_API_KEY."
                )
                return []
            elif resp.status_code == 429:
                logger.warning("[OpenSubtitles] Rate limit reached (HTTP 429).")
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
                delay = 3600.0 if res.status_code == 406 else 60.0
                try:
                    data = res.json()
                    reset_unix = data.get("reset_time_unix")
                    if reset_unix:
                        diff = float(reset_unix) - time.time()
                        delay = max(60.0, min(diff, 86400.0))
                except Exception:
                    pass
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

        effective_key, _, _ = self._resolve_credentials(api_key, username, password)

        m = re.search(r"(\d+)(?:\.srt)?$", str(download_ref).strip())
        file_id = int(m.group(1)) if m else None

        direct_link: str | None = None
        if file_id and effective_key:
            direct_link = await self.get_download_url(
                file_id, effective_key, username=username, password=password
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
