"""Bounded numeric telemetry owned by one synchronization job."""
from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class SyncTiming:
    job_id: str = field(default_factory=lambda: uuid4().hex[:12])
    external_plan_available: bool = False
    external_plan_started: bool = False
    external_plan_ms: float | None = None
    cached_reference_present: bool = False
    cached_reference_attempted: bool = False
    candidate_count_planned: int = 0
    candidates_considered: int = 0
    references_downloaded: int = 0
    references_deduplicated: int = 0
    alass_processes_launched: int = 0
    winning_candidate_index: int | None = None
    pal_reference_kind: str | None = None
    pal_reference_index: int | None = None
    reference_kind: str | None = None
    winning_sync_strategy: str | None = None
    cache_hit: bool = False

    def log(self, result: str, total_ms: float) -> None:
        # Avoided means successful completion with a configured planner that was not run.
        # Errors, cancellations, and ordinary failure fallbacks are not latency savings.
        avoided = (
            result in {"original", "transformed", "cached"}
            and self.external_plan_available
            and not self.external_plan_started
            and (self.cache_hit or self.winning_sync_strategy is not None)
        )
        # Provenance is an allowlisted category, never a provider-controlled string.
        kind = self.reference_kind if self.reference_kind in {
            "hash", "edition", "group", "embedded", "bluray_candidate", "team"
        } else "other"
        logger.info(
            "[sync-latency] job=%s external_plan_started=%s external_plan_avoided=%s "
            "external_plan_ms=%s cached_reference_present=%s cached_reference_attempted=%s "
            "candidate_count_planned=%d candidates_considered=%d references_downloaded=%d "
            "references_deduplicated=%d ALASS_processes_launched=%d "
            "winning_candidate_index=%s winning_reference_kind=%s winning_sync_strategy=%s "
            "total_ms=%.3f result=%s",
            self.job_id, self.external_plan_started, avoided,
            self.external_plan_ms, self.cached_reference_present, self.cached_reference_attempted,
            self.candidate_count_planned, self.candidates_considered, self.references_downloaded,
            self.references_deduplicated, self.alass_processes_launched,
            self.winning_candidate_index, kind if self.winning_sync_strategy is not None else None,
            self.winning_sync_strategy, total_ms, result,
        )


current_sync_timing: ContextVar[SyncTiming | None] = ContextVar("sync_timing", default=None)
