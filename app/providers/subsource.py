"""Subsource official REST API v1 provider integration."""

import asyncio
import json
import logging
import re
from typing import Any

import httpx

from app.config import settings
from app.models import SubtitleRelease
from app.providers.base import BaseSubtitleProvider
from app.utils.language import get_subsource_lang_name, normalize_to_iso639_2

logger = logging.getLogger("uvicorn.error")


def _mask_key(val: str) -> str:
    """Mask sensitive API key strings for logs."""
    if not val:
        return "<empty>"
    v = str(val).strip()
    if len(v) <= 8:
        return "***"
    return f"{v[:4]}...{v[-4:]}"


def _matches_series_episode(item: dict[str, Any], season: int, episode: int, raw_name: str) -> bool:
    """
    Check if a SubSource subtitle item matches the target series season and episode.
    Handles explicit fields, SxxExx, episode ranges (e.g. 001-025), explicit episode
    markers (e.g. E2, Ep.2), anime dash numbering (-91, -02), and whole-season packs.
    """
    try:
        target_s = int(season)
        target_e = int(episode)
    except (ValueError, TypeError):
        return True

    # 1. Explicit metadata fields on item
    item_s = item.get("season")
    item_e = item.get("episode")
    if item_s is not None:
        try:
            if int(item_s) != target_s:
                return False
        except (ValueError, TypeError):
            pass
    if item_e is not None:
        try:
            if int(item_e) != target_e:
                return False
        except (ValueError, TypeError):
            pass
    if item_s is not None and item_e is not None:
        try:
            return int(item_s) == target_s and int(item_e) == target_e
        except (ValueError, TypeError):
            pass

    # 2. Gather all candidate release name strings
    candidates = []
    rel_info = item.get("releaseInfo") or item.get("release_info")
    if isinstance(rel_info, list):
        candidates.extend([str(x) for x in rel_info if x])
    elif isinstance(rel_info, str) and rel_info:
        candidates.append(rel_info)
    for k in ("release_name", "releaseName", "name", "release"):
        val = item.get(k)
        if val and isinstance(val, str):
            candidates.append(val)
    if raw_name:
        candidates.append(raw_name)

    if not candidates:
        return True

    has_specific_indicator = False
    for r_name in candidates:
        r_lower = r_name.lower()

        # a) Standard SxxExx / SeXX.EpXX
        m_se = re.search(
            r"(?:s|se|season)[-._ ]?(\d{1,2})[-._ ]*(?:e|ep|episode)[-._ ]?(\d{1,3})", r_lower
        )
        if m_se:
            has_specific_indicator = True
            if int(m_se.group(1)) == target_s and int(m_se.group(2)) == target_e:
                return True
            continue

        # b) Episode range: e.g. 001-025, 001-148, 01-12
        m_range = re.search(r"(?:ep|e|episode)?\s*0*(\d{1,4})\s*[-~–—]\s*0*(\d{1,4})", r_lower)
        if m_range:
            s_ep, e_ep = int(m_range.group(1)), int(m_range.group(2))
            if s_ep not in (2160, 1080, 720, 576, 480, 264, 265) and e_ep not in (
                2160,
                1080,
                720,
                576,
                480,
                264,
                265,
            ):
                has_specific_indicator = True
                if s_ep <= target_e <= e_ep:
                    return True
                continue

        # c) Explicit episode marker: e.g. E2, Ep.2, Episode 2
        m_ep = re.search(r"(?:^|[\s._\-\[#])(?:ep|episode|e)\s*0*(\d{1,4})(?:[^\d]|$)", r_lower)
        if m_ep:
            ep_val = int(m_ep.group(1))
            if ep_val not in (2160, 1080, 720, 576, 480, 264, 265) and not (1900 <= ep_val <= 2099):
                has_specific_indicator = True
                if ep_val == target_e:
                    return True
                continue

        # d) Bracket episode number: e.g. [02], [91]
        m_bracket = re.search(r"\[\s*0*(\d{1,4})\s*\]", r_lower)
        if m_bracket:
            ep_val = int(m_bracket.group(1))
            if ep_val not in (2160, 1080, 720, 576, 480, 264, 265) and not (1900 <= ep_val <= 2099):
                has_specific_indicator = True
                if ep_val == target_e:
                    return True
                continue

        # e) Dash episode numbering: e.g. Hunter X Hunter-91, Show - 02
        m_dash = re.search(r"[-–—]\s*0*(\d{1,4})(?:[^\d]|$)", r_lower)
        if m_dash:
            ep_val = int(m_dash.group(1))
            if ep_val not in (2160, 1080, 720, 576, 480, 264, 265) and not (1900 <= ep_val <= 2099):
                has_specific_indicator = True
                if ep_val == target_e:
                    return True
                continue

        # f) Whole season pack without specific episode: e.g. S01 Complete, Season 1
        m_s_only = re.search(r"\b(?:s|season)[-._ ]?0*(\d+)\b", r_lower)
        if m_s_only and int(m_s_only.group(1)) == target_s:
            if not re.search(r"\be\d+", r_lower):
                return True

    if has_specific_indicator:
        return False

    return True


class SubSourceService:
    """Service client for SubSource REST API following production Stremio addon querying pipeline."""

    BASE_URL = "https://api.subsource.net/api/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key.strip()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "Authorization": f"Bearer {self.api_key}",
            "X-API-Key": self.api_key,
            "Accept": "application/json",
        }

    async def resolve_movie_id(
        self,
        client: httpx.AsyncClient,
        imdb_id: str,
        title: str | None = None,
        year: int | None = None,
    ) -> int | None:
        """
        Step 1: Resolve internal SubSource movieId using IMDb ID or text title search.
        Tries candidate query parameters on /movies/search.
        """
        endpoint = f"{self.BASE_URL}/movies/search"

        def _extract_id(data: Any) -> int | None:
            if isinstance(data, dict):
                for id_key in ("movieId", "movie_id", "id", "_id"):
                    if data.get(id_key) is not None:
                        try:
                            return int(data[id_key])
                        except (ValueError, TypeError):
                            pass
                for nested_key in ("data", "movies", "results"):
                    if nested_key in data:
                        found = _extract_id(data[nested_key])
                        if found is not None:
                            return found
            elif isinstance(data, list):
                for item in data:
                    found = _extract_id(item)
                    if found is not None:
                        return found
            return None

        # 1. Primary lookup: direct IMDb ID query
        imdb_params = {"searchType": "imdb", "imdb": imdb_id}
        try:
            logger.info(
                f"[SubSource Movie Search] Outbound URL: {endpoint} | Params: {imdb_params}"
            )
            resp = await client.get(
                endpoint,
                headers=self.headers,
                params=imdb_params,
                timeout=settings.UPSTREAM_TIMEOUT,
            )
            raw_snippet = resp.text[:500] if resp.text else "<empty>"
            logger.info(
                f"[SubSource Movie Search Response] Status: {resp.status_code} | Body[:500]: {raw_snippet}"
            )

            if resp.status_code == 200:
                try:
                    data = resp.json()
                    mid = _extract_id(data)
                    if mid:
                        logger.info(
                            f"[SubSource Movie Search] Successfully resolved movieId={mid} for '{imdb_id}'"
                        )
                        return mid
                except Exception as json_err:
                    logger.warning(
                        f"[SubSource Movie Search] Failed to decode JSON response: {json_err}"
                    )
            elif resp.status_code in (401, 403):
                logger.warning(
                    f"[SubSource Movie Search] Authentication failed (HTTP {resp.status_code}). Check SUBSOURCE_API_KEY."
                )
                return None
        except Exception as e:
            logger.warning(f"[SubSource Movie Search] IMDb lookup error: {e}")

        # 2. Fallback lookup: text search if title is provided
        if title:
            text_params: dict[str, Any] = {"searchType": "text", "q": title}
            if year:
                text_params["year"] = year
            try:
                logger.info(
                    f"[SubSource Movie Search Fallback] Outbound URL: {endpoint} | Params: {text_params}"
                )
                resp = await client.get(
                    endpoint,
                    headers=self.headers,
                    params=text_params,
                    timeout=settings.UPSTREAM_TIMEOUT,
                )
                raw_snippet = resp.text[:500] if resp.text else "<empty>"
                logger.info(
                    f"[SubSource Movie Search Fallback Response] Status: {resp.status_code} | Body[:500]: {raw_snippet}"
                )

                if resp.status_code == 200:
                    try:
                        data = resp.json()
                        mid = _extract_id(data)
                        if mid:
                            logger.info(
                                f"[SubSource Movie Search] Successfully resolved movieId={mid} via text fallback for '{title}'"
                            )
                            return mid
                    except Exception as json_err:
                        logger.warning(
                            f"[SubSource Movie Search Fallback] Failed to decode JSON response: {json_err}"
                        )
                elif resp.status_code in (401, 403):
                    logger.warning(
                        f"[SubSource Movie Search] Authentication failed (HTTP {resp.status_code}). Check SUBSOURCE_API_KEY."
                    )
                    return None
            except Exception as e:
                logger.warning(f"[SubSource Movie Search Fallback] Text fallback error: {e}")

        logger.warning(
            f"[SubSource Movie Search] Could not resolve movieId for IMDb ID '{imdb_id}'"
        )
        return None

    async def get_subtitles(
        self,
        client: httpx.AsyncClient,
        media_type: str,
        imdb_id: str,
        season: int | None = None,
        episode: int | None = None,
        language: str = "Arabic",
        title: str | None = None,
        year: int | None = None,
        release_info: str | None = None,
        exclude_hi: bool = False,
        **kwargs,
    ) -> list:
        try:
            subtitles_endpoint = f"{self.BASE_URL}/subtitles"
            sanitized_headers = {
                k: (
                    f"Bearer {_mask_key(self.api_key)}"
                    if k.lower() == "authorization"
                    else (_mask_key(v) if k.lower() == "x-api-key" else v)
                )
                for k, v in self.headers.items()
            }
            logger.info(
                f"[SubSource Request] Outbound URL: {subtitles_endpoint} | Sanitized Headers: {sanitized_headers}"
            )

            # Step 1: Resolve internal SubSource movieId using IMDb ID
            movie_id = await self.resolve_movie_id(
                client=client, imdb_id=imdb_id, title=title, year=year
            )

            async def _fetch_pages(base_params: dict[str, Any]) -> list[Any]:
                all_items: list[Any] = []
                seen_ids = set()
                page = 1
                max_pages = 5

                while page <= max_pages:
                    p = dict(base_params)
                    p["page"] = page

                    try:
                        resp = await client.get(
                            subtitles_endpoint,
                            params=p,
                            headers=self.headers,
                            timeout=settings.UPSTREAM_TIMEOUT,
                        )
                        logger.info(f"[SubSource Request Page {page}] URL: {resp.request.url}")
                        logger.info(
                            f"[SubSource Response Page {page}] Status Code: {resp.status_code}"
                        )

                        if resp.status_code != 200:
                            logger.warning(
                                f"[SubSource] HTTP {resp.status_code} on page {page} for {imdb_id}: {resp.text[:300]}"
                            )
                            break

                        try:
                            data = resp.json()
                        except Exception:
                            data = json.loads(resp.text)

                        batch: list[Any] = []
                        if isinstance(data, list):
                            batch = data
                        elif isinstance(data, dict):
                            for key in ("subtitles", "data", "results", "items"):
                                if key in data and isinstance(data[key], list):
                                    batch = data[key]
                                    break
                            if not batch and "id" in data:
                                batch = [data]

                        if not batch:
                            break

                        new_in_page = 0
                        for it in batch:
                            if not isinstance(it, dict):
                                continue
                            item_id = (
                                it.get("id")
                                or it.get("_id")
                                or it.get("subtitleId")
                                or it.get("sub_id")
                            )
                            if item_id and item_id in seen_ids:
                                continue
                            if item_id:
                                seen_ids.add(item_id)
                            all_items.append(it)
                            new_in_page += 1

                        if new_in_page == 0:
                            break

                        if isinstance(data, dict):
                            total_pages = (
                                data.get("totalPages")
                                or data.get("total_pages")
                                or data.get("pages")
                            )
                            if total_pages is not None:
                                try:
                                    if page >= int(total_pages):
                                        break
                                except (ValueError, TypeError):
                                    pass

                            if "hasMore" in data and not bool(data["hasMore"]):
                                break

                        if len(batch) < 3:
                            break

                        page += 1

                    except Exception as page_err:
                        logger.warning(
                            f"[SubSource] Pagination fetch error on page {page}: {page_err}"
                        )
                        break

                return all_items

            items: list[Any] = []

            # Step 2: Fetch subtitles using movieId with high limit & pagination
            if movie_id:
                params: dict[str, Any] = {
                    "movieId": movie_id,
                    "limit": 100,
                    "perPage": 100,
                    "take": 100,
                }
                if media_type == "series":
                    if season is not None:
                        params["season"] = int(season)
                    if episode is not None:
                        params["episode"] = int(episode)

                items = await _fetch_pages(params)

            # Fallback or secondary query using releaseInfo if movie_id query returned nothing or failed
            if release_info and not items:
                params_rel = {
                    "releaseInfo": release_info,
                    "limit": 100,
                    "perPage": 100,
                    "take": 100,
                }
                logger.info(
                    f"[SubSource Request Fallback] Querying /subtitles with releaseInfo params: {params_rel}"
                )
                items = await _fetch_pages(params_rel)

            if not items:
                logger.warning(f"[SubSource] No subtitle items found for {imdb_id}")
                return []

            logger.info(f"[SubSource Response] Total candidate items extracted: {len(items)}")

            # Step 3: Filter & Parse results locally by season, episode, and language
            lang_lower = language.lower()
            lang_full = get_subsource_lang_name(language).lower()
            raw_items = [
                it
                for it in items
                if isinstance(it, dict)
                and (
                    lang_lower in str(it.get("lang") or it.get("language") or "").lower()
                    or lang_full in str(it.get("lang") or it.get("language") or "").lower()
                )
            ]

            total_fetched = len(raw_items)
            dropped_by_hi = 0
            passed = 0

            matched_subs = []
            for idx, item in enumerate(raw_items):
                try:
                    # Keep Hearing Impaired handling strictly bound to item.get("hearingImpaired") is True
                    is_hi = item.get("hearingImpaired") is True
                    if is_hi and exclude_hi:
                        dropped_by_hi += 1
                        continue

                    sub_id = (
                        item.get("id")
                        or item.get("_id")
                        or item.get("subtitleId")
                        or item.get("sub_id")
                    )
                    if not sub_id:
                        continue

                    # Extract release name
                    raw_name = (
                        item.get("release_name")
                        or item.get("releaseName")
                        or item.get("releaseInfo")
                        or item.get("release_info")
                        or item.get("name")
                        or item.get("release")
                        or f"{imdb_id}.SubSource.{sub_id}"
                    )
                    if isinstance(raw_name, list):
                        raw_name = raw_name[0] if raw_name else f"{imdb_id}.SubSource.{sub_id}"
                    raw_name = str(raw_name)

                    # Filter series episodes
                    if media_type == "series" and season is not None and episode is not None:
                        if not _matches_series_episode(item, season, episode, raw_name):
                            continue

                    item_lang = str(item.get("lang") or item.get("language") or "").lower()

                    matched_subs.append(
                        {
                            "id": str(sub_id),
                            "raw_name": raw_name,
                            "lang": item.get("lang_code")
                            or normalize_to_iso639_2(item_lang, default="ara"),
                            "download_path": item.get("download_path")
                            or item.get("downloadUrl")
                            or f"/subtitles/{sub_id}/download",
                            "hearing_impaired": is_hi,
                            "files": item.get("files") or [],
                        }
                    )
                    passed += 1
                except Exception as parse_err:
                    logger.error(
                        f"[SubSource Item Parse Error] Error parsing item #{idx}: {parse_err} | Item: {item}"
                    )
                    continue

            logger.info(
                f"[Filter Verification] Total: {total_fetched} | Dropped (HI): {dropped_by_hi} | Final Passed: {passed}"
            )

            logger.info(
                f"[SubSource] Successfully matched {len(matched_subs)} subtitles for {imdb_id} (Target Lang: '{language}', Media: '{media_type}', S:{season} E:{episode})"
            )
            return matched_subs

        except Exception as e:
            logger.error(f"[SubSource Fetch Error] {str(e)}", exc_info=True)
            return []


class SubsourceProvider(BaseSubtitleProvider):
    """Subtitle provider implementation for Subsource.net official REST API v1."""

    name = "subsource"
    BASE_URL = "https://api.subsource.net/api/v1"
    API_SUBTITLES_PATH = "/subtitles"
    API_DOWNLOAD_PATH = "/subtitles/{subtitle_id}/download"

    def _get_headers(self, api_key: str | None = None) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/plain, */*",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        }
        effective_key = (api_key or settings.SUBSOURCE_API_KEY or "").strip()
        if effective_key:
            headers["X-API-Key"] = effective_key
            headers["Authorization"] = f"Bearer {effective_key}"
        return headers

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
        target_filename: str | None = None,
        **kwargs,
    ) -> list[SubtitleRelease]:
        """
        Search SubSource official REST API v1 for subtitles.
        Resolves internal movieId via /movies/search and queries /subtitles.
        """
        effective_key = (api_key or settings.SUBSOURCE_API_KEY or "").strip()
        if not effective_key:
            logger.warning(
                "Subsource API key is not configured. Subsource requires an API key for queries. "
                "Configure SUBSOURCE_API_KEY via environment variable or user config link."
            )
            return []

        # Parse compound Stremio ID if formatted as tt1234567:season:episode
        clean_imdb_id = imdb_id
        if ":" in clean_imdb_id:
            parts = clean_imdb_id.split(":")
            clean_imdb_id = parts[0].strip()
            if len(parts) >= 3:
                is_series = True
                try:
                    if season is None:
                        season = int(parts[1])
                    if episode is None:
                        episode = int(parts[2])
                except (ValueError, TypeError):
                    pass
        else:
            clean_imdb_id = clean_imdb_id.strip()

        if clean_imdb_id.endswith(".json"):
            clean_imdb_id = clean_imdb_id[:-5]

        media_type = "series" if is_series else "movie"
        service = SubSourceService(effective_key)
        target_languages = languages if languages else ["ara"]

        if len(target_languages) == 1:
            lang_arg = get_subsource_lang_name(target_languages[0])
            raw_subs = await service.get_subtitles(
                client=self.client,
                media_type=media_type,
                imdb_id=clean_imdb_id,
                season=season,
                episode=episode,
                language=lang_arg,
                title=title,
                year=year,
                release_info=target_filename,
                exclude_hi=exclude_hi,
            )
        else:
            tasks = [
                service.get_subtitles(
                    client=self.client,
                    media_type=media_type,
                    imdb_id=clean_imdb_id,
                    season=season,
                    episode=episode,
                    language=get_subsource_lang_name(lang),
                    title=title,
                    year=year,
                    release_info=target_filename,
                    exclude_hi=exclude_hi,
                )
                for lang in target_languages
            ]
            lang_results = await asyncio.gather(*tasks, return_exceptions=True)
            raw_subs = []
            for r in lang_results:
                if isinstance(r, list):
                    raw_subs.extend(r)
                elif isinstance(r, Exception):
                    logger.warning(f"[SubSource] Language fetch task failed: {r}")

        results: list[SubtitleRelease] = []
        for item in raw_subs:
            try:
                is_hi = bool(item.get("hearing_impaired", False))
                if exclude_hi and is_hi:
                    continue

                raw_release = str(item.get("raw_name") or f"{clean_imdb_id}.srt").strip()
                if raw_release.lower().endswith(".zip"):
                    raw_release = raw_release[:-4]

                # Determine subtitle format (.ass, .ssa, .vtt, .srt)
                files_list = item.get("files") or []
                raw_lower = raw_release.lower()
                if raw_lower.endswith(".ass") or any(
                    str(f).lower().endswith(".ass") for f in files_list
                ):
                    sub_fmt = "ass"
                elif raw_lower.endswith(".ssa") or any(
                    str(f).lower().endswith(".ssa") for f in files_list
                ):
                    sub_fmt = "ssa"
                elif raw_lower.endswith(".vtt") or any(
                    str(f).lower().endswith(".vtt") for f in files_list
                ):
                    sub_fmt = "vtt"
                else:
                    sub_fmt = "srt"

                if not raw_release.lower().endswith(f".{sub_fmt}"):
                    raw_release = re.sub(
                        r"\.(?:srt|vtt|sub|ass|ssa)$", "", raw_release, flags=re.IGNORECASE
                    )
                    raw_release += f".{sub_fmt}"

                download_path = item.get("download_path") or f"/subtitles/{item['id']}/download"
                if not download_path.startswith("http"):
                    download_url = f"{self.BASE_URL}/{download_path.lstrip('/')}"
                else:
                    download_url = download_path

                item_lang = normalize_to_iso639_2(item.get("lang"), default="ara")

                results.append(
                    SubtitleRelease(
                        release_name=raw_release,
                        download_url=download_url,
                        provider=self.name,
                        format=sub_fmt,
                        hearing_impaired=is_hi,
                        lang=item_lang,
                    )
                )
            except Exception as map_err:
                logger.error(
                    f"[SubSource Mapping Error] Error mapping item to SubtitleRelease: {map_err} | Item: {item}"
                )
                continue

        logger.info(
            f"Subsource returned {len(results)} subtitles for {clean_imdb_id} (Langs: {target_languages})"
        )
        return results

    async def download_archive(self, download_ref: str, api_key: str | None = None) -> bytes | None:
        """
        Download subtitle archive (.zip or .srt) from Subsource using X-API-Key.
        Handles both direct binary streams and JSON redirects with downloadUrl.
        """
        effective_key = (api_key or settings.SUBSOURCE_API_KEY or "").strip()
        headers = self._get_headers(effective_key)
        headers["Accept"] = "*/*"

        url = download_ref
        if not url.startswith("http"):
            url = f"{self.BASE_URL}/subtitles/{download_ref}/download"

        try:
            resp = await self.client.get(
                url,
                headers=headers,
                timeout=settings.UPSTREAM_TIMEOUT,
                follow_redirects=True,
            )
            if resp.status_code == 200 and resp.content:
                raw_content = resp.content
                # Handle possible JSON payload (e.g. {"downloadUrl": "https://..."} or error)
                if raw_content.strip().startswith(b"{"):
                    try:
                        json_data = json.loads(raw_content.decode("utf-8"))
                        cdn_url = (
                            json_data.get("downloadUrl")
                            or json_data.get("download_url")
                            or json_data.get("url")
                        )
                        if cdn_url:
                            cdn_resp = await self.client.get(
                                cdn_url,
                                follow_redirects=True,
                                timeout=settings.UPSTREAM_TIMEOUT,
                            )
                            if cdn_resp.status_code == 200 and cdn_resp.content:
                                return cdn_resp.content
                        logger.warning(f"Subsource returned non-binary JSON: {json_data}")
                        return None
                    except Exception as e:
                        logger.debug(f"JSON check parse error in Subsource download: {e}")

                return raw_content

            if resp.status_code in (401, 403):
                logger.warning(
                    f"Subsource download authentication failed (HTTP {resp.status_code}). Check SUBSOURCE_API_KEY."
                )
                return None
            if resp.status_code == 429:
                logger.warning("Subsource download rate limit reached (HTTP 429).")
                return None
            logger.warning(f"Subsource download returned HTTP {resp.status_code} for {url}")
            return None
        except httpx.TimeoutException:
            logger.warning(f"Subsource download timed out for {url}")
            return None
        except Exception as e:
            logger.warning(f"Failed to download from Subsource ({url}): {e}")
            return None
