"""Subdl API provider integration."""

import logging

import httpx

from app.config import settings
from app.models import SubtitleRelease
from app.providers.base import BaseSubtitleProvider
from app.utils.language import get_subdl_lang_code, normalize_to_iso639_2

logger = logging.getLogger("uvicorn.error")


class SubdlProvider(BaseSubtitleProvider):
    """Subtitle provider implementation for Subdl.com."""

    name = "subdl"
    BASE_URL = "https://api.subdl.com/api/v1/subtitles"
    DOWNLOAD_BASE = "https://dl.subdl.com"

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
        **kwargs,
    ) -> list[SubtitleRelease]:
        """
        Query Subdl for subtitles by IMDb ID.
        Requires a valid Subdl API key.
        """
        effective_key = (api_key or settings.SUBDL_API_KEY or "").strip()
        if not effective_key:
            logger.warning(
                "Subdl API key is not configured. Subdl requires an API key for queries. "
                "Configure SUBDL_API_KEY via environment variable or user config link."
            )
            return []

        target_langs = [normalize_to_iso639_2(lang) for lang in languages] if languages else ["ara"]
        subdl_langs = list(dict.fromkeys([get_subdl_lang_code(lang) for lang in target_langs]))
        if not subdl_langs:
            subdl_langs = ["AR"]

        params = {
            "api_key": effective_key,
            "imdb_id": imdb_id,
            "languages": ",".join(subdl_langs),
            "type": "tv" if is_series else "movie",
            "subs_per_page": 30,
            "limit": 100,
            "per_page": 100,
        }

        if is_series:
            if season is not None:
                params["season_number"] = str(season)
            if episode is not None:
                params["episode_number"] = str(episode)

        headers = {
            "Accept": "application/json",
            "User-Agent": "StremioArabicSubs/1.0.0",
        }

        all_subtitles_raw: list[dict] = []
        seen_urls = set()
        page = 1
        max_pages = 5

        while page <= max_pages:
            page_params = dict(params)
            if page > 1:
                page_params["page"] = page

            try:
                resp = await self.client.get(
                    self.BASE_URL,
                    params=page_params,
                    headers=headers,
                    timeout=settings.UPSTREAM_TIMEOUT,
                )

                if resp.status_code in (401, 403):
                    logger.warning(
                        f"Subdl API authentication failed (HTTP {resp.status_code}). "
                        f"Please verify your SUBDL_API_KEY: {resp.text[:200]}"
                    )
                    break

                if resp.status_code == 429:
                    logger.warning("Subdl API rate limit reached (HTTP 429).")
                    break

                if resp.status_code != 200:
                    logger.warning(
                        f"Subdl API returned HTTP {resp.status_code} for IMDb {imdb_id}: {resp.text[:200]}"
                    )
                    break

                data = resp.json()
                if not data.get("status") and not data.get("subtitles"):
                    logger.debug(f"Subdl returned no subtitles for {imdb_id} on page {page}")
                    break

                subtitles_raw = data.get("subtitles") or []
                if not subtitles_raw:
                    break

                new_count = 0
                for item in subtitles_raw:
                    if not isinstance(item, dict):
                        continue
                    url_key = (
                        item.get("url")
                        or item.get("download_link")
                        or item.get("release_name")
                        or item.get("name")
                    )
                    if url_key and url_key in seen_urls:
                        continue
                    if url_key:
                        seen_urls.add(url_key)
                    all_subtitles_raw.append(item)
                    new_count += 1

                if new_count == 0:
                    break

                total_pages = data.get("totalPages") or data.get("total_pages") or data.get("pages")
                if total_pages is not None:
                    try:
                        if page >= int(total_pages):
                            break
                    except (ValueError, TypeError):
                        pass

                if len(subtitles_raw) < 30:
                    break

                page += 1
            except httpx.TimeoutException:
                logger.warning(f"Subdl search request timed out for {imdb_id} on page {page}")
                break
            except Exception as e:
                logger.warning(f"Error querying Subdl page {page} for {imdb_id}: {e}")
                break

        # Fallback to season pack if episode-specific query returned 0 items
        if is_series and episode is not None and not all_subtitles_raw:
            logger.info(
                f"No episode-specific subtitles on Subdl for {imdb_id} S{season}E{episode}. Trying season pack fallback..."
            )
            fallback_params = dict(params)
            fallback_params.pop("episode_number", None)
            try:
                resp = await self.client.get(
                    self.BASE_URL,
                    params=fallback_params,
                    headers=headers,
                    timeout=settings.UPSTREAM_TIMEOUT,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data.get("subtitles") or []:
                        if isinstance(item, dict):
                            all_subtitles_raw.append(item)
            except Exception as fb_err:
                logger.debug(f"Subdl season pack fallback error: {fb_err}")

        total_fetched = len(all_subtitles_raw)
        logger.info(f"[SubDL Response] Total items received from API: {total_fetched}")

        dropped_by_hi = 0
        results: list[SubtitleRelease] = []
        for item in all_subtitles_raw:
            if not isinstance(item, dict):
                continue

            # If exclude_hi is enabled, skip hearing impaired subtitles
            is_hi = bool(
                item.get("hi") or item.get("hearing_impaired") or item.get("hearingImpaired")
            )
            if exclude_hi and is_hi:
                dropped_by_hi += 1
                continue

            raw_name = item.get("release_name") or item.get("name") or f"{imdb_id}.srt"
            raw_url = item.get("url") or item.get("download_link") or ""

            if not raw_url:
                continue

            if raw_url.startswith("http://") or raw_url.startswith("https://"):
                full_download_url = raw_url
            else:
                if not raw_url.startswith("/"):
                    raw_url = "/" + raw_url
                full_download_url = f"{self.DOWNLOAD_BASE}{raw_url}"

            # Append .srt extension to release name if missing
            clean_release_name = raw_name.strip()
            if clean_release_name.lower().endswith(".zip"):
                clean_release_name = clean_release_name[:-4]

            if clean_release_name.lower().endswith(".ass"):
                sub_fmt = "ass"
            elif clean_release_name.lower().endswith(".ssa"):
                sub_fmt = "ssa"
            elif clean_release_name.lower().endswith(".vtt"):
                sub_fmt = "vtt"
            else:
                sub_fmt = "srt"
                if not clean_release_name.lower().endswith(".srt"):
                    clean_release_name += ".srt"

            raw_lang = item.get("lang") or item.get("language")
            item_lang = normalize_to_iso639_2(
                raw_lang, default=target_langs[0] if target_langs else "ara"
            )

            results.append(
                SubtitleRelease(
                    release_name=clean_release_name,
                    download_url=full_download_url,
                    provider=self.name,
                    format=sub_fmt,
                    hearing_impaired=is_hi,
                    lang=item_lang,
                )
            )

        logger.info(
            f"[SubDL Filter Verification] Total: {total_fetched} | "
            f"Dropped (HI): {dropped_by_hi} | "
            f"Final Passed: {len(results)}"
        )
        logger.info(
            f"Subdl returned {len(results)} subtitles for {imdb_id} (Langs: {target_langs})"
        )
        return results

    async def download_archive(self, download_ref: str, api_key: str | None = None) -> bytes | None:
        """
        Download subtitle archive (.zip or .srt) from Subdl.
        """
        effective_key = (api_key or settings.SUBDL_API_KEY or "").strip()
        headers = {
            "User-Agent": "StremioArabicSubs/1.0.0",
        }
        params = {}
        if effective_key:
            headers["x-api-key"] = effective_key
            params["api_key"] = effective_key

        try:
            resp = await self.client.get(
                download_ref,
                params=params,
                headers=headers,
                timeout=settings.UPSTREAM_TIMEOUT,
                follow_redirects=True,
            )
            if resp.status_code == 200 and resp.content:
                return resp.content
            if resp.status_code in (401, 403):
                logger.warning(
                    f"Subdl download authentication failed (HTTP {resp.status_code}). Check SUBDL_API_KEY."
                )
                return None
            if resp.status_code == 429:
                logger.warning("Subdl download rate limit reached (HTTP 429).")
                return None
            logger.warning(f"Subdl download returned HTTP {resp.status_code} for {download_ref}")
            return None
        except httpx.TimeoutException:
            logger.warning(f"Subdl download timed out for {download_ref}")
            return None
        except Exception as e:
            logger.warning(f"Failed to download from Subdl ({download_ref}): {e}")
            return None
