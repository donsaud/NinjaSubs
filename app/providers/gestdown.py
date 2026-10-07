"""Public Gestdown TV API. Metadata matches only; no MovieHash claims."""
from __future__ import annotations

import re
from urllib.parse import quote
from uuid import UUID

import httpx

from app.extractor import looks_like_subtitle
from app.models import SubtitleRelease
from app.providers.base import BaseSubtitleProvider
from app.utils.language import normalize_to_iso639_2


class GestdownProvider(BaseSubtitleProvider):
    name = "gestdown"
    BASE_URL = "https://api.gestdown.info"
    TIMEOUT = 8.0

    async def _json(self, path: str) -> dict:
        response = await self.client.get(self.BASE_URL + path, timeout=self.TIMEOUT,
                                         follow_redirects=False)
        if response.status_code != 200 or len(response.content) > 2_000_000:
            return {}
        data = response.json()
        return data if isinstance(data, dict) else {}

    async def search_subtitles(
        self, imdb_id: str, is_series: bool = False, season: int | None = None,
        episode: int | None = None, title: str | None = None, year: int | None = None,
        api_key: str | None = None, languages: list[str] | None = None,
        exclude_hi: bool = False, *, tvdb_id: int | str | None = None, **kwargs,
    ) -> list[SubtitleRelease]:
        if not is_series or type(season) is not int or type(episode) is not int:
            return []
        if season < 0 or episode < 0:
            return []
        wanted = list(dict.fromkeys(normalize_to_iso639_2(v, default="")
                                    for v in (languages or ["eng"])))
        wanted = [v for v in wanted if v]
        if not wanted:
            return []
        try:
            if tvdb_id is not None:
                if not re.fullmatch(r"[1-9]\d{0,9}", str(tvdb_id)):
                    return []
                shows = (await self._json(f"/shows/external/tvdb/{tvdb_id}")).get("shows", [])
                shows = [s for s in shows if isinstance(s, dict)
                         and str(s.get("tvDbId")) == str(tvdb_id)]
            elif title and title.strip():
                shows = (await self._json("/shows/search/" + quote(title.strip(), safe=""))).get("shows", [])
                shows = [s for s in shows if isinstance(s, dict)
                         and str(s.get("name", "")).strip().casefold() == title.strip().casefold()]
            else:
                return []
            # No fuzzy selection between similarly named shows.
            if len(shows) != 1:
                return []
            if not isinstance(shows[0].get("id"), str):
                return []
            show_id = str(UUID(shows[0]["id"]))
        except (httpx.HTTPError, ValueError, TypeError, KeyError):
            return []
        results: list[SubtitleRelease] = []
        seen: set[str] = set()
        for language in wanted[:8]:
            try:
                data = await self._json(f"/subtitles/get/{show_id}/{season}/{episode}/{language}")
                rows = data.get("matchingSubtitles", [])
                if not isinstance(rows, list):
                    continue
                for row in rows[:200]:
                    if not isinstance(row, dict) or row.get("completed") is not True:
                        continue
                    if not isinstance(row.get("language"), str):
                        continue
                    lang = normalize_to_iso639_2(row["language"], default="")
                    sid = str(row.get("subtitleId", ""))
                    if not re.fullmatch(r"[a-fA-F0-9-]{36}", sid):
                        continue
                    UUID(sid)
                    if lang != language or sid in seen:
                        continue
                    hi = row.get("hearingImpaired") is True
                    if exclude_hi and hi:
                        continue
                    # Preserve upstream release/version, without fabricating a filename.
                    release = row.get("release") or row.get("version")
                    if not isinstance(release, str) or not release.strip():
                        continue
                    path = f"/subtitles/download/{sid}"
                    if row.get("downloadUri") != path:
                        continue
                    results.append(SubtitleRelease(release_name=release.strip(),
                                   download_url=self.BASE_URL + path, provider=self.name,
                                   lang=lang, hearing_impaired=hi))
                    seen.add(sid)
            except (httpx.HTTPError, ValueError, TypeError, KeyError):
                continue
        return results

    async def download_archive(self, download_ref: str, api_key: str | None = None) -> bytes | None:
        path = download_ref.removeprefix(self.BASE_URL)
        if not re.fullmatch(r"/subtitles/download/[a-fA-F0-9-]{36}", path):
            return None
        try:
            response = await self.client.get(self.BASE_URL + path, timeout=self.TIMEOUT,
                                             follow_redirects=False)
            raw = response.content
            if response.status_code == 200 and len(raw) <= 2_000_000 and looks_like_subtitle(raw):
                return raw
        except httpx.HTTPError:
            pass
        return None
