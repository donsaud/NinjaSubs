"""No-auth iSubtitles HTML search and downloads, isolated from aggregation."""
from __future__ import annotations

import html
import re
from urllib.parse import quote

import httpx

from app.extractor import looks_like_subtitle
from app.models import SubtitleRelease
from app.providers.base import BaseSubtitleProvider
from app.utils.language import normalize_to_iso639_2


def _text(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", value)).strip()


def _imdb_matches(page: str, imdb_id: str) -> bool:
    wanted = re.fullmatch(r"tt(\d+)", imdb_id)
    return wanted is not None and any(int(v) == int(wanted[1])
                                     for v in re.findall(r"imdb\.com/title/tt(\d+)", page))


class IsubtitlesProvider(BaseSubtitleProvider):
    name = "isubtitles"
    BASE_URL = "https://isubtitles.org"
    TIMEOUT = 8.0

    async def _page(self, path: str) -> str:
        response = await self.client.get(self.BASE_URL + path, timeout=self.TIMEOUT,
                                         follow_redirects=False)
        if response.status_code != 200 or len(response.content) > 4_000_000:
            return ""
        if re.search(r"cf-chl|captcha|just a moment", response.text, re.I):
            return ""
        return response.text

    async def search_subtitles(
        self, imdb_id: str, is_series: bool = False, season: int | None = None,
        episode: int | None = None, title: str | None = None, year: int | None = None,
        api_key: str | None = None, languages: list[str] | None = None,
        exclude_hi: bool = False, **kwargs,
    ) -> list[SubtitleRelease]:
        if not title or not re.fullmatch(r"tt\d+", imdb_id):
            return []
        if is_series and (type(season) is not int or type(episode) is not int
                          or season < 0 or episode < 0):
            return []
        wanted = {normalize_to_iso639_2(v, default="") for v in (languages or ["ara"])} - {""}
        if not wanted:
            return []
        try:
            search = await self._page("/search?kwd=" + quote(title.strip(), safe=""))
            candidates = []
            for block in re.split(r"<h3\b", search, flags=re.I)[1:]:
                link = re.search(r'href="(/[a-z0-9-]+-subtitles)"', block, re.I)
                if link and _imdb_matches(block, imdb_id):
                    candidates.append(link[1])
            if len(set(candidates)) != 1:
                return []
            listing = await self._page(candidates[0])
            if not _imdb_matches(listing, imdb_id):
                return []
            results: list[SubtitleRelease] = []
            seen: set[tuple[str, str]] = set()
            for row in re.findall(r"<tr\b[^>]*>(.*?)</tr>", listing, re.I | re.S):
                download = re.search(r'href="(/download/[a-z0-9-]+/[a-z-]+/\d+)"', row, re.I)
                language = re.search(r'<td[^>]*data-title="Language"[^>]*>(.*?)</td>', row, re.I | re.S)
                releases = re.search(r'<td[^>]*data-title="Release / Movie"[^>]*>(.*?)</td>', row, re.I | re.S)
                if not download or not language or not releases:
                    continue
                lang = normalize_to_iso639_2(_text(language[1]), default="")
                if lang not in wanted:
                    continue
                for release in re.findall(r"<a\b[^>]*>(.*?)</a>", releases[1], re.I | re.S):
                    release = _text(release)
                    episodes = [(int(s), int(e)) for s, e in re.findall(
                        r"(?i)\bS(\d{1,3})[ ._-]*E(\d{1,3})\b", release)]
                    # Only explicitly identified episode files; never guess a pack's episode.
                    if is_series and (episodes != [(season, episode)] or re.search(
                            r"(?i)E\d+[-._ ]+E?\d+\b", release)):
                        continue
                    if not is_series and episodes:
                        continue
                    key = (download[1], release)
                    if not release or key in seen:
                        continue
                    results.append(SubtitleRelease(release_name=release,
                                   download_url=self.BASE_URL + download[1], provider=self.name,
                                   lang=lang))
                    seen.add(key)
                    if len(results) >= 200:
                        return results
            return results
        except (httpx.HTTPError, ValueError, TypeError):
            return []

    async def download_archive(self, download_ref: str, api_key: str | None = None) -> bytes | None:
        path = download_ref.removeprefix(self.BASE_URL)
        if not re.fullmatch(r"/download/[a-z0-9-]+/[a-z-]+/\d+", path, re.I):
            return None
        try:
            response = await self.client.get(self.BASE_URL + path, timeout=self.TIMEOUT,
                                             follow_redirects=False)
            raw = response.content
            if response.status_code == 200 and len(raw) <= 4_000_000:
                if raw.startswith(b"PK\x03\x04") or looks_like_subtitle(raw):
                    return raw  # Shared extractor validates archive members/encoding.
        except httpx.HTTPError:
            pass
        return None
