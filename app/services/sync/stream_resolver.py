"""Generic Stremio stream resolver: find the direct URL of the playing file.

NinjaSubs does not normally receive a ``streamUrl`` from Stremio, so the
embedded-track and MovieHash tiers have nothing to probe. Given the base URL of
*any* Stremio-compliant stream addon (Torrentio, MediaFusion, Comet, AIOStreams,
...), this client queries the standard ``/stream/{type}/{id}.json`` endpoint,
matches the stream whose title/description/filename corresponds to the playing
file (release-group and resolution aware), and returns its direct URL for
range/probe use.

The resolver is intentionally stateless beyond the httpx client and an optional
default base URL (the ``AIOSTREAMS_URL`` environment fallback), so a per-user
addon URL can be supplied on each call.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from app.services.sync.matching import _release_group, _resolution

if TYPE_CHECKING:  # pragma: no cover - typing only
    import httpx

logger = logging.getLogger(__name__)

_MIN_MATCH_SCORE = 0.6
_MANIFEST_SUFFIX = "/manifest.json"


def clean_stream_addon_url(url: str | None) -> str:
    """Normalise a user-supplied Stremio addon URL to a bare base URL.

    Strips surrounding whitespace, a trailing ``/manifest.json`` (with or
    without a leading slash), and any trailing slashes. Returns ``""`` for an
    empty/blank input.
    """
    cleaned = (url or "").strip()
    if not cleaned:
        return ""
    cleaned = cleaned.rstrip("/")
    if cleaned.endswith(_MANIFEST_SUFFIX):
        cleaned = cleaned[: -len(_MANIFEST_SUFFIX)]
    elif cleaned.endswith("manifest.json"):
        cleaned = cleaned[: -len("manifest.json")]
    return cleaned.rstrip("/")


def _tokens(text: str | None) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _match_score(filename: str | None, candidate_text: str) -> float:
    """Score how well a candidate stream describes the requested filename.

    Starts from token overlap, then rewards a matching release group and
    resolution and penalises a mismatched group/resolution — the two signals
    that most reliably separate the correct release in a addon stream list.
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
        timeout: float = 1.2,
    ) -> None:
        self._client = client
        self._default_base_url = clean_stream_addon_url(default_base_url)
        self.timeout = timeout

    def _base_url(self, base_url: str | None = None) -> str:
        """User-supplied addon URL wins; else the configured env fallback."""
        return clean_stream_addon_url(base_url) or self._default_base_url

    @staticmethod
    def _stream_id(imdb_id: str, media_type: str, season, episode) -> str:
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

        ``base_url`` overrides the instance's default addon URL (the
        ``AIOSTREAMS_URL`` fallback). Never raises: an unreachable/slow addon or
        a non-matching list degrades to ``None`` so the caller falls through to
        the external providers.
        """
        base = self._base_url(base_url)
        if not base or not imdb_id:
            return None
        ident = self._stream_id(imdb_id, media_type, season, episode)
        url = f"{base}/stream/{media_type}/{ident}.json"
        resp = None
        for attempt in range(2):  # one retry: stream addons are occasionally flaky
            try:
                resp = await self._client.get(url, timeout=self.timeout, follow_redirects=True)
                break
            except Exception as exc:  # noqa: BLE001 - never raise to caller
                logger.info(
                    "[reference] stream addon request failed (attempt %d): %r", attempt + 1, exc
                )
        if resp is None:
            return None
        if resp.status_code != 200:
            logger.info("[reference] stream addon returned HTTP %s", resp.status_code)
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
        return best_url
