"""Internal AIOStreams bridge: resolve a direct stream URL for a request.

NinjaSubs does not normally receive a ``streamUrl`` from Stremio, so the
embedded-track and MovieHash tiers have nothing to probe. When ``AIOSTREAMS_URL``
is configured (internal Docker network), this client asks AIOStreams for the
streams of the requested media, matches the one whose title/filename equals the
playing file, and returns its direct URL for range/probe use.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from app.services.sync.matching import _release_group

if TYPE_CHECKING:  # pragma: no cover - typing only
    import httpx

logger = logging.getLogger(__name__)

_MIN_MATCH_SCORE = 0.6


def _tokens(text: str | None) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _match_score(filename: str | None, candidate_text: str) -> float:
    """Fraction of the requested filename's tokens present in a candidate."""
    wanted = _tokens(filename)
    if not wanted:
        return 0.0
    candidate = _tokens(candidate_text)
    if not candidate:
        return 0.0
    overlap = len(wanted & candidate) / len(wanted)
    wanted_group = (_release_group(filename) or "").lower()
    if wanted_group:
        if wanted_group in candidate:
            overlap += 0.15
        else:
            overlap -= 0.25
    return overlap


class AIOStreamsClient:
    """Resolve a direct stream URL from an internal AIOStreams instance."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        *,
        timeout: float = 1.2,
    ) -> None:
        self._client = client
        self._base_url = (base_url or "").rstrip("/")
        self.timeout = timeout

    def _stream_id(self, imdb_id: str, media_type: str, season, episode) -> str:
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
    ) -> str | None:
        """Return the direct URL of the stream matching ``filename`` or ``None``.

        Never raises: an unreachable/slow AIOStreams or a non-matching list
        degrades to ``None`` so the caller falls through to the next tier.
        """
        if not self._base_url or not imdb_id:
            return None
        ident = self._stream_id(imdb_id, media_type, season, episode)
        url = f"{self._base_url}/stream/{media_type}/{ident}.json"
        resp = None
        for attempt in range(2):  # one retry: the bridge is occasionally flaky
            try:
                resp = await self._client.get(url, timeout=self.timeout, follow_redirects=True)
                break
            except Exception as exc:  # noqa: BLE001 - never raise to caller
                logger.info(
                    "[reference] AIOStreams request failed (attempt %d): %r", attempt + 1, exc
                )
        if resp is None:
            return None
        if resp.status_code != 200:
            logger.info("[reference] AIOStreams returned HTTP %s", resp.status_code)
            return None
        try:
            data = resp.json()
        except Exception:  # noqa: BLE001
            logger.info("[reference] AIOStreams returned non-JSON")
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
                stream.get("title"),
                stream.get("name"),
                stream.get("description"),
                hints.get("filename") if isinstance(hints, dict) else None,
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
                "[reference] AIOStreams: no stream matched %r among %d (best=%.2f)",
                filename,
                len(streams),
                best_score,
            )
            return None
        logger.info(
            "[reference] AIOStreams matched stream for %r (score=%.2f)", filename, best_score
        )
        return best_url
