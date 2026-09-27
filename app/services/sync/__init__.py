"""Auto-sync subsystem: English reference resolution and alignment.

- ``matching``: pure release-name matching primitives.
- ``query`` / ``cache``: lookup parameters and persistent reference disk cache.
- ``decode``: shared ZIP-aware payload decoding.
- ``external_strategy``: the single external English reference resolver
  (SubDL / SubSource / OpenSubtitles by IMDb id + season/episode).
- ``orchestrator``: ``SyncOrchestrator`` facade that resolves a reference and
  runs ``alass``, falling back to the original subtitle on any failure.
"""

from app.services.sync.orchestrator import SyncOrchestrator, build_synced_cache_key

__all__ = [
    "SyncOrchestrator",
    "build_synced_cache_key",
]
