"""Auto-sync subsystem: reference matching, decision tree, and alignment.

- ``matching``: pure release-name matching primitives.
- ``tree``: explicit reference decision tree (team > edition > abort).
- ``query`` / ``cache``: lookup parameters and persistent reference disk cache.
- ``decode``: shared ZIP-aware payload decoding.
- ``hash_strategy``: Tier 1 ground truth (OpenSubtitles video-hash match).
- ``embedded_strategy``: Tier 2 ground truth (embedded English track extract).
- ``orchestrator``: ``SyncOrchestrator`` facade routing each request through
  the deterministic tiers with fail-safe fallback to the original.
"""

from app.services.sync.embedded_strategy import EmbeddedStrategy
from app.services.sync.hash_strategy import HashExactStrategy
from app.services.sync.orchestrator import SyncOrchestrator, build_synced_cache_key

__all__ = [
    "EmbeddedStrategy",
    "HashExactStrategy",
    "SyncOrchestrator",
    "build_synced_cache_key",
]
