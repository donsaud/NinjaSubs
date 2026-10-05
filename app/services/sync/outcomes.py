"""Per-title sync-outcome telemetry: which titles actually get synchronized.

Every sync request already logs its own outcome, but no single line answers
"is Dexter syncing better than Whiplash this week?". Raw logs require a human
to correlate delivery lines across requests; this module keeps bounded
in-memory counters keyed by title so the answer is one log line.

Design rules:

* **Counts only, never content.** Keys are ``(imdb_id, season, episode)``;
  values are outcome counters. No subtitle text, filenames, URLs, credentials,
  or digests are stored or logged.
* **Terminal outcomes only.** Exactly one record per request, at the point the
  final bytes are decided -- never per candidate, so multi-candidate requests
  cannot inflate the numbers.
* **Bounded memory.** At most ``MAX_TITLES`` title keys; beyond that new titles
  still advance the global totals but are not broken out. A subtitle addon is
  long-lived and must not grow a dict forever.
* **Observability only.** Recording an outcome never changes what is served,
  cached, or verified. The tracker has no reference to caches, strategies, or
  the verifier.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

#: Terminal outcomes. ``verified`` covers both fresh verification and the
#: constant-shift exemption (which rewrites the verdict to VERIFIED_RESYNCED);
#: ``large_offset_served`` stays distinct because its state remains UNVERIFIED
#: by design and must never be confused with a verified claim in aggregates.
VERIFIED = "verified"
LARGE_OFFSET_SERVED = "large_offset_served"
REJECTED = "rejected"
NO_REFERENCE = "no_reference"

OUTCOMES: tuple[str, ...] = (VERIFIED, LARGE_OFFSET_SERVED, REJECTED, NO_REFERENCE)

#: At most this many distinct titles are broken out; see module docstring.
MAX_TITLES = 500

#: Emit the summary line every this many recorded requests.
SUMMARY_EVERY = 25


def classify_outcome(
    serve_synchronized: bool,
    serving_state_value: str | None,
    rejected_candidates_seen: bool,
) -> str:
    """Map a terminal serving decision to exactly one outcome bucket.

    Pure function so the mapping itself is unit-testable without an
    orchestrator. ``serving_state_value`` is the ``LargeOffsetServingState``
    value (or equivalent string); anything that is not the large-offset
    corrected state and still served synchronized is a verifier-approved
    result.
    """
    if serve_synchronized:
        if (serving_state_value or "").strip().lower() == "alass_corrected_large_offset":
            return LARGE_OFFSET_SERVED
        return VERIFIED
    if rejected_candidates_seen:
        return REJECTED
    return NO_REFERENCE


class SyncOutcomeTracker:
    """Thread-safe per-title outcome counters with periodic summary logging."""

    def __init__(
        self,
        summary_every: int = SUMMARY_EVERY,
        max_titles: int = MAX_TITLES,
    ) -> None:
        self._summary_every = max(1, int(summary_every))
        self._max_titles = max(1, int(max_titles))
        self._lock = threading.Lock()
        self._totals: dict[str, int] = dict.fromkeys(OUTCOMES, 0)
        self._titles: dict[tuple[str, str, str], dict[str, int]] = {}
        self._recorded = 0

    @staticmethod
    def title_key(imdb_id: Any, season: Any, episode: Any) -> tuple[str, str, str]:
        """Normalized title key. ``None`` season/episode means a movie."""
        imdb = str(imdb_id or "").strip().lower()
        season_key = "movie" if season is None else str(season).strip()
        episode_key = "x" if episode is None else str(episode).strip()
        return (imdb, season_key, episode_key)

    def record(
        self,
        imdb_id: Any,
        season: Any,
        episode: Any,
        outcome: str,
    ) -> None:
        """Record one terminal request outcome; log a summary when due."""
        if outcome not in OUTCOMES:
            return
        key = self.title_key(imdb_id, season, episode)
        with self._lock:
            self._totals[outcome] += 1
            self._recorded += 1
            if key not in self._titles:
                if len(self._titles) >= self._max_titles:
                    key = ("", "", "")
                    if key not in self._titles:
                        self._titles[key] = dict.fromkeys(OUTCOMES, 0)
                else:
                    self._titles[key] = dict.fromkeys(OUTCOMES, 0)
            self._titles[key][outcome] += 1
            due = self._recorded % self._summary_every == 0
            if due:
                self._log_summary_locked()

    def snapshot(self) -> dict[str, Any]:
        """Copy of current counts, for tests and diagnostics."""
        with self._lock:
            return {
                "total": self._recorded,
                "totals": dict(self._totals),
                "titles": {k: dict(v) for k, v in self._titles.items()},
            }

    def reset(self) -> None:
        """Clear all counts. Tests only; production never calls this."""
        with self._lock:
            self._totals = dict.fromkeys(OUTCOMES, 0)
            self._titles = {}
            self._recorded = 0

    def _log_summary_locked(self) -> None:
        totals = " ".join(f"{name}={self._totals[name]}" for name in OUTCOMES)
        weak = sorted(
            (
                (key, counts[REJECTED] + counts[NO_REFERENCE], sum(counts.values()))
                for key, counts in self._titles.items()
                if key[0] and counts[REJECTED] + counts[NO_REFERENCE] > 0
            ),
            key=lambda item: (-item[1], item[0]),
        )[:5]
        weak_part = "; ".join(
            f"{imdb or '?'}:S{season}E{episode} failing={failing}/{total}"
            for (imdb, season, episode), failing, total in weak
        ) or "none"
        logger.info(
            "[sync-outcomes] last=%d totals(%s) weakest=[%s]",
            self._recorded,
            totals,
            weak_part,
        )


#: Process-wide tracker. Recording is observability only: it never changes
#: what is served, cached, or verified.
SYNC_OUTCOMES = SyncOutcomeTracker()
