"""Backward-compatible AIOStreams name for the generic stream resolver.

The stream resolution logic was generalised to query any Stremio-compliant
stream addon (Torrentio, MediaFusion, Comet, AIOStreams, ...). This module
keeps the historical ``AIOStreamsClient`` name working for callers and tests;
new code should use :class:`app.services.sync.stream_resolver.StreamResolver`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.services.sync.stream_resolver import StreamResolver

if TYPE_CHECKING:  # pragma: no cover - typing only
    import httpx

__all__ = ["AIOStreamsClient"]


class AIOStreamsClient(StreamResolver):
    """Deprecated alias for :class:`StreamResolver` (kept for compatibility)."""

    def __init__(self, client: httpx.AsyncClient, base_url: str, *, timeout: float = 1.2) -> None:
        super().__init__(client, base_url, timeout=timeout)
