"""Generic Stremio stream resolver: find the direct URL of the playing file.

NinjaSubs does not receive a ``streamUrl`` directly from Stremio for subtitle requests.
Given the base URL of any Stremio-compliant stream addon (Torrentio, MediaFusion, Comet,
AIOStreams, etc.), this client queries the standard ``/stream/{type}/{id}.json`` endpoint,
matches the stream whose title/description/filename corresponds to the playing file
(release-group and resolution aware), and returns its direct URL for embedded subtitle probing.
"""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING

from app.services.sync.matching import _release_group, _resolution

if TYPE_CHECKING:  # pragma: no cover - typing only
    import httpx

logger = logging.getLogger(__name__)

_MIN_MATCH_SCORE = 0.50
_MANIFEST_SUFFIX = "/manifest.json"


def clean_stream_addon_url(url: str | None) -> str:
    """Normalize any user-supplied Stremio stream addon URL to a clean base URL.

    Handles:
    - Strips leading/trailing whitespace, newlines, and accidental quotes (" or ')
    - Converts 'stremio://' protocol to 'https://'
    - Auto-prepends 'https://' if protocol is missing (e.g. 'torrentio.strem.fun/manifest.json')
    - Strips trailing '/manifest.json', 'manifest.json', and trailing slashes
    - Rewrites local container host alias (aiostreams:4000 -> aiostreams:3000)
    """
    cleaned = (url or "").strip().strip("\"'").strip()
    if not cleaned:
        return ""

    # Normalize protocol
    if cleaned.startswith("stremio://"):
        cleaned = "https://" + cleaned[len("stremio://"):]
    elif not cleaned.startswith(("http://", "https://")):
        cleaned = "https://" + cleaned

    cleaned = cleaned.rstrip("/")
    if cleaned.endswith(_MANIFEST_SUFFIX):
        cleaned = cleaned[: -len(_MANIFEST_SUFFIX)]
    elif cleaned.endswith("manifest.json"):
        cleaned = cleaned[: -len("manifest.json")]
    cleaned = cleaned.rstrip("/")

    # Internal Docker network alias: inside Docker network, aiostreams listens on 3000,
    # though it is mapped to 4000 on the host.
    if "aiostreams:4000" in cleaned:
        cleaned = cleaned.replace("aiostreams:4000", "aiostreams:3000")
    return cleaned


def _tokens(text: str | None) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _match_score(filename: str | None, candidate_text: str) -> float:
    """Score how well a candidate stream describes the requested filename.

    Starts from token overlap, then rewards a matching release group and
    resolution and penalises a mismatched group/resolution.
    """
    wanted = _tokens(filename)
    if not wanted:
        return 0.0
    candidate = _tokens(candidate_text)
    if not candidate:
        return 0.0
    score = len(wanted & candidate) / len(wanted)

    wanted_group = (_release_group(filename) or "").lower()
    if wanted_group:
        candidate_group = (_release_group(candidate_text) or "").lower()
        if candidate_group and candidate_group == wanted_group:
            score += 0.2
        elif wanted_group in candidate:
            score += 0.15
        else:
            score -= 0.25

    wanted_res = _resolution(filename)
    if wanted_res:
        candidate_res = _resolution(candidate_text)
        if candidate_res == wanted_res:
            score += 0.15
        elif candidate_res:
            score -= 0.15

    return score


class StreamResolver:
    """Resolve a direct stream URL from any Stremio stream addon."""

    name = "stream-resolver"

    def __init__(
        self,
        client: httpx.AsyncClient,
        default_base_url: str = "",
        *,
        timeout: float = 4.0,
    ) -> None:
        self._client = client
        self._default_base_url = clean_stream_addon_url(default_base_url)
        self.timeout = timeout
        self._cache: dict[str, tuple[float, str]] = {}
        self._cache_ttl = 600.0  # 10 minutes

    def _base_url(self, base_url: str | None = None) -> str:
        """User-supplied addon URL wins; else the configured env fallback."""
        return clean_stream_addon_url(base_url) or self._default_base_url

    @staticmethod
    def _stream_id(imdb_id: str, media_type: str, season: int | None, episode: int | None) -> str:
        ident = (imdb_id or "").strip().lstrip("/")
        if ident.endswith(".json"):
            ident = ident[: -len(".json")]
        if media_type in ("series", "tv", "anime") and season is not None and episode is not None:
            if ":" not in ident:
                ident = f"{ident}:{season}:{episode}"
        return ident

    async def resolve_stream_url(
        self,
        imdb_id: str,
        media_type: str,
        filename: str | None,
        *,
        season: int | None = None,
        episode: int | None = None,
        base_url: str | None = None,
    ) -> str | None:
        """Return the direct URL of the stream matching ``filename`` or ``None``.

        ``base_url`` overrides the instance's default addon URL.
        Never raises: an unreachable/slow addon or a non-matching list degrades
        gracefully to ``None``.
        """
        base = self._base_url(base_url)
        if not base or not imdb_id:
            return None

        # Check short-term cache
        now = time.monotonic()
        cache_key = f"{base}:{imdb_id}:{media_type}:{season}:{episode}:{filename}"
        if cache_key in self._cache:
            ts, cached_url = self._cache[cache_key]
            if now - ts < self._cache_ttl:
                return cached_url
            self._cache.pop(cache_key, None)

        ident = self._stream_id(imdb_id, media_type, season, episode)
        m_type = "series" if (media_type in ("series", "tv") or (season is not None and episode is not None)) else media_type
        url = f"{base}/stream/{m_type}/{ident}.json"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
        }

        resp = None
        for attempt in range(2):
            try:
                resp = await self._client.get(
                    url,
                    headers=headers,
                    timeout=self.timeout,
                    follow_redirects=True,
                )
                if resp.status_code == 200:
                    break
                if resp.status_code == 404 and media_type == "anime" and m_type == "series":
                    # Try anime endpoint if series returned 404
                    alt_url = f"{base}/stream/anime/{ident}.json"
                    resp = await self._client.get(
                        alt_url,
                        headers=headers,
                        timeout=self.timeout,
                        follow_redirects=True,
                    )
                    if resp.status_code == 200:
                        break
            except Exception as exc:  # noqa: BLE001 - never raise to caller
                logger.info(
                    "[reference] stream addon request failed (attempt %d): %r", attempt + 1, exc
                )

        if resp is None or resp.status_code != 200:
            logger.info("[reference] stream addon returned HTTP %s for %s", getattr(resp, "status_code", "None"), url)
            return None

        try:
            data = resp.json()
        except Exception:  # noqa: BLE001
            logger.info("[reference] stream addon returned non-JSON")
            return None

        streams = (data.get("streams") if isinstance(data, dict) else None) or []
        if not isinstance(streams, list):
            return None

        best_url: str | None = None
        best_score = 0.0
        for stream in streams:
            if not isinstance(stream, dict):
                continue
            direct = stream.get("url")
            if not isinstance(direct, str) or not direct.startswith(("http://", "https://")):
                continue
            hints = stream.get("behaviorHints") or {}
            candidates = [
                hints.get("filename") if isinstance(hints, dict) else None,
                stream.get("title"),
                stream.get("name"),
                stream.get("description"),
            ]
            score = max(
                (_match_score(filename, text) for text in candidates if text),
                default=0.0,
            )
            if score > best_score:
                best_score = score
                best_url = direct

        if best_url is None or best_score < _MIN_MATCH_SCORE:
            logger.info(
                "[reference] stream addon: no stream matched %r among %d (best=%.2f)",
                filename,
                len(streams),
                best_score,
            )
            return None

        logger.info(
            "[reference] stream addon matched stream for %r (score=%.2f)", filename, best_score
        )
        self._cache[cache_key] = (now, best_url)
        return best_url
