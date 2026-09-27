"""Backward-compatible entry point for the auto-sync subsystem.

Canonical homes (Phase 2 layout):

- :mod:`app.services.sync.query` — :class:`ReferenceQuery`
- :mod:`app.services.sync.cache` — :class:`ReferenceDiskCache`
- :mod:`app.services.sync.matching` — release-name matching primitives
- :mod:`app.services.sync.external_strategy` — external English reference resolver
- :mod:`app.services.sync.orchestrator` — request routing facade

All names below are re-exported with ``as`` aliases so existing imports keep
working without modification.
"""

from app.services.sync.cache import ReferenceDiskCache as ReferenceDiskCache
from app.services.sync.external_strategy import (
    ExternalExactStrategy as DualReferenceResolver,
)
from app.services.sync.matching import (
    _member_episode_number as _member_episode_number,
)
from app.services.sync.matching import (
    _release_group as _release_group,
)
from app.services.sync.matching import (
    _season_number as _season_number,
)
from app.services.sync.matching import (
    _title_from_filename as _title_from_filename,
)
from app.services.sync.matching import (
    is_informative_release_name as is_informative_release_name,
)
from app.services.sync.query import ReferenceQuery as ReferenceQuery

__all__ = [
    "DualReferenceResolver",
    "ReferenceDiskCache",
    "ReferenceQuery",
    "_member_episode_number",
    "_release_group",
    "_season_number",
    "_title_from_filename",
    "is_informative_release_name",
]
