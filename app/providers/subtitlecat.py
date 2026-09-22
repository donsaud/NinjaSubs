"""SubtitleCat (subtitlecat.com) provider integration.

SubtitleCat is a translation-focused subtitle site. There is no IMDb lookup, so
a text search is performed (title + year for movies, title + SxxExx for series).
Each search result links to a detail page which exposes direct ``.srt`` files for
the languages that have already been translated, e.g.
``/subs/1657/Movie.2024.1080p-eng-ar.srt``.

Only pre-generated translations are surfaced; on-the-fly translation buttons are
not invoked.
"""

import asyncio
import html
import logging
import re
from urllib.parse import unquote, urljoin

import httpx

from app.config import settings
from app.models import SubtitleRelease
from app.providers.base import BaseSubtitleProvider
from app.utils.language import get_subtitlecat_lang_codes, normalize_to_iso639_2

logger = logging.getLogger("uvicorn.error")

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

_SEARCH_LINK_REGEX = re.compile(
    r'href="(subs/\d+/[^"]+?\.html)"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL
)
_SRT_LINK_REGEX = re.compile(r'href="(/subs/\d+/[^"]+?\.srt)"', re.IGNORECASE)
_TARGET_LANG_REGEX = re.compile(r"-([A-Za-z]{2}(?:-[A-Za-z]{2})?)\.srt$", re.IGNORECASE)
_TAG_REGEX = re.compile(r"<[^>]+>")

MAX_CANDIDATES = 6


class SubtitlecatProvider(BaseSubtitleProvider):
    """Subtitle provider implementation for subtitlecat.com."""

    name = "subtitlecat"
    BASE_URL = "https://www.subtitlecat.com"

    def _headers(self) -> dict[str, str]:
        return {
            "User-Agent": _UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": f"{self.BASE_URL}/",
        }

    @staticmethod
    def _build_query(
        title: str | None,
        is_series: bool,
        season: int | None,
        episode: int | None,
        year: int | None,
    ) -> str:
        parts: list[str] = []
        if title:
            parts.append(str(title).strip())
        if is_series and season is not None and episode is not None:
            parts.append(f"S{int(season):02d}E{int(episode):02d}")
        elif year:
            parts.append(str(year))
        return " ".join(p for p in parts if p).strip()

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
        """Search SubtitleCat by text query and collect translated .srt links."""
        query = self._build_query(title, is_series, season, episode, year)
        if not query:
            return []

        wanted_codes: set[str] = set()
        for lang in languages or []:
            wanted_codes |= get_subtitlecat_lang_codes(lang)
        if not wanted_codes:
            return []

        try:
            resp = await self.client.get(
                f"{self.BASE_URL}/index.php",
                params={"search": query},
                headers=self._headers(),
                timeout=settings.UPSTREAM_TIMEOUT,
                follow_redirects=True,
            )
        except httpx.TimeoutException:
            logger.warning(f"[SubtitleCat] Search timed out for '{query}'")
            return []
        except Exception as e:
            logger.warning(f"[SubtitleCat] Search error for '{query}': {e}")
            return []

        if resp.status_code != 200:
            logger.info(f"[SubtitleCat] Search HTTP {resp.status_code} for '{query}'")
            return []

        candidates: list[tuple[str, str]] = []
        seen_paths: set[str] = set()
        for match in _SEARCH_LINK_REGEX.finditer(resp.text):
            rel_path = match.group(1).strip()
            if rel_path in seen_paths:
                continue
            seen_paths.add(rel_path)
            release_name = html.unescape(_TAG_REGEX.sub("", match.group(2)))
            release_name = re.sub(r"\s+", " ", release_name).strip()
            detail_url = urljoin(f"{self.BASE_URL}/", rel_path)
            candidates.append((detail_url, release_name))
            if len(candidates) >= MAX_CANDIDATES:
                break

        if not candidates:
            logger.info(f"[SubtitleCat] No search results for '{query}'")
            return []

        # Fetch candidate detail pages concurrently (bounded), then collect .srt links.
        detail_pages = await asyncio.gather(
            *(self._fetch_detail(url) for url, _ in candidates),
            return_exceptions=True,
        )

        results: list[SubtitleRelease] = []
        seen_keys: set[tuple[str, str]] = set()
        for (_detail_url, release_name), page_html in zip(
            candidates, detail_pages, strict=False
        ):
            if isinstance(page_html, BaseException) or not page_html:
                continue
            for srt_match in _SRT_LINK_REGEX.finditer(page_html):
                srt_path = srt_match.group(1).strip()
                code_match = _TARGET_LANG_REGEX.search(srt_path)
                if not code_match:
                    continue
                target_code = code_match.group(1).lower()
                if target_code not in wanted_codes or target_code == "orig":
                    continue

                # Prefer the human-readable search result name; fall back to the file slug.
                name = release_name
                if not name:
                    slug = unquote(srt_path.rsplit("/", 1)[-1])
                    name = re.sub(r"-[A-Za-z]{2}(?:-[A-Za-z]{2})?\.srt$", "", slug)
                    name = re.sub(r"\s+", " ", name).strip()

                dedup_key = (name.lower(), target_code)
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)

                results.append(
                    SubtitleRelease(
                        release_name=name,
                        download_url=f"{self.BASE_URL}{srt_path}",
                        provider=self.name,
                        format="srt",
                        hearing_impaired=False,
                        lang=self._iso_from_code(target_code),
                    )
                )

        logger.info(
            f"[SubtitleCat] Found {len(results)} subtitles for '{query}' "
            f"(Candidates: {len(candidates)}, Langs: {languages})"
        )
        return results

    async def _fetch_detail(self, url: str) -> str:
        """Fetch a SubtitleCat detail page and return its HTML (empty on failure)."""
        try:
            resp = await self.client.get(
                url,
                headers=self._headers(),
                timeout=settings.UPSTREAM_TIMEOUT,
                follow_redirects=True,
            )
            if resp.status_code == 200:
                return resp.text
        except Exception as e:
            logger.debug(f"[SubtitleCat] Detail fetch failed for {url}: {e}")
        return ""

    @staticmethod
    def _iso_from_code(code: str) -> str:
        """Map a SubtitleCat language code (e.g. 'ar', 'pt-BR') to ISO-639-2."""
        base = code.split("-")[0].lower()
        return normalize_to_iso639_2(base, default="ara")

    async def download_archive(self, download_ref: str, api_key: str | None = None) -> bytes | None:
        """Download a plain ``.srt`` file (SubtitleCat returns raw text, not a zip)."""
        url = download_ref
        if not url.startswith("http"):
            url = f"{self.BASE_URL}/{url.lstrip('/')}"

        headers = dict(self._headers())
        headers["Accept"] = "text/plain,application/octet-stream,*/*"

        try:
            resp = await self.client.get(
                url,
                headers=headers,
                timeout=settings.UPSTREAM_TIMEOUT,
                follow_redirects=True,
            )
            if resp.status_code == 200 and resp.content:
                return resp.content
            logger.warning(f"[SubtitleCat] Download returned HTTP {resp.status_code} for {url}")
            return None
        except httpx.TimeoutException:
            logger.warning(f"[SubtitleCat] Download timed out for {url}")
            return None
        except Exception as e:
            logger.warning(f"[SubtitleCat] Download error for {url}: {e}")
            return None
