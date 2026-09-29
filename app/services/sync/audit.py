"""Shadow audit trail for synchronization decisions.

Observability only. Nothing in this module may influence a ranking decision, a
threshold, or a served payload: records are written *after* the decision has
already been made, and reading them back is impossible from the request path.
The point is to measure real behaviour before changing anything, so that any
future threshold change can cite a before/after number.

Privacy: no subtitle text, no file contents, no credentials, and no full signed
URLs are ever stored. Identities are truncated SHA-256 digests, and release
names are reduced to a bounded, sanitized form so a log file cannot leak a
private stream name. Reasons are the analyzer's own short metric strings.

The interesting measurement is *prediction vs. later verification*. A search-time
prediction and a serve-time verification for the same video + candidate are
paired by :class:`AuditLog`, which is what makes a confusion matrix (and a
false-positive rate) computable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.services.sync.alignment import SyncState, VerificationAvailability
from app.services.sync.structural import CutVerdict

logger = logging.getLogger(__name__)

# Bumped when the record shape changes, so old exports are not mixed in.
AUDIT_SCHEMA_VERSION = 1
# Identifies the verification engine generation that produced a record.
ENGINE_VERSION = "sync-eval-1"
# Outcomes that count as a verified synchronization claim.
VERIFIED_OUTCOMES = ("verified_synced", "verified_resynced")

# Reason strings are truncated: they must stay metrics, not subtitle content.
_MAX_REASON_CHARS = 160
_MAX_RELEASE_NAME_CHARS = 64
# Anything that looks like a URL, token, or long opaque blob is never recorded.
_SENSITIVE_PATTERN = re.compile(r"(?i)(https?://|[?&](token|signature|auth|key)=|\b[A-Za-z0-9_-]{40,}\b)")


def stable_id(value: Any, *, length: int = 16) -> str | None:
    """Truncated digest of an identifier, or ``None`` when absent."""
    text = str(value or "").strip()
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def sanitize_reason(reason: str) -> str:
    """Bound a reason string and strip anything URL- or token-shaped."""
    text = _SENSITIVE_PATTERN.sub("<redacted>", str(reason or "")).strip()
    return text[:_MAX_REASON_CHARS]


def sanitize_name(name: str | None) -> str | None:
    """Reduce a release name to a bounded, non-identifying form.

    Kept because source/edition class is a required dimension of the audit
    (which release types generate drift or different cuts), but truncated so a
    private stream title cannot be recovered from a log file. A value that is a
    URL is discarded entirely: removing only the scheme would still leak the
    host and path.
    """
    text = str(name or "").strip()
    if not text:
        return None
    if "://" in text:
        return None
    # An opaque 40+ char blob is a token or a hash, not a release name.
    if re.fullmatch(r"[A-Za-z0-9_-]{40,}", text):
        return None
    return text[:_MAX_RELEASE_NAME_CHARS]


class SyncDecisionRecord(BaseModel):
    """One shadow observation of a decision. Never used to make one."""

    schema_version: int = AUDIT_SCHEMA_VERSION
    # Which build of the verification engine produced this observation. Lets an
    # analysis detect records written by a different engine generation.
    engine_version: str = ENGINE_VERSION
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # "search" (metadata prediction / cache evidence) or "serve" (measured).
    phase: str = "search"

    # Digests only; never raw identifiers.
    video_id: str | None = None
    subtitle_id: str | None = None
    language: str = "und"
    has_video_fingerprint: bool = False
    # Only what the application explicitly knows, never inferred.
    request_context: str | None = None

    match_tier: str | None = None
    compatibility_score: float | None = None
    # Which predictor rule fired, so rules can be compared against outcomes.
    prediction_rule: str | None = None
    prediction_confidence: float | None = None

    verification: str = VerificationAvailability.UNKNOWN.value
    sync_state: str = SyncState.UNVERIFIED.value
    sync_confidence: float | None = None

    median_offset_ms: float | None = None
    mad_offset_ms: float | None = None
    p95_offset_ms: float | None = None
    drift_ms_per_minute: float | None = None
    coverage_score: float | None = None
    structural_similarity: float | None = None
    cut_verdict: str | None = None

    alass_applied: bool = False
    alass_successful: bool = False
    from_cache: bool = False

    # Populated only on a serve record that resolves an earlier prediction.
    prediction_state: str | None = None
    prediction_rule_at_search: str | None = None

    # --- reference evidence behind this decision ------------------------ #
    # "Which reference was used, and was it trustworthy?" A verified decision
    # is only as good as this, so it is recorded alongside the outcome.
    reference_trust: str | None = None
    reference_provider: str | None = None
    reference_from_cache: bool = False
    reference_independent_sources: int | None = None
    reference_consensus: float | None = None
    reference_failure: str | None = None
    # --- shadow reference selection v2 (measurement only) ---------------- #
    # Which candidate the alternative policy would have chosen. Never used for
    # the decision itself; recorded so the two policies can be compared.
    shadow_reference_id: str | None = None
    shadow_reference_changed: bool = False
    shadow_reference_trust: str | None = None
    shadow_reference_health: str | None = None
    shadow_independent_groups: int | None = None
    shadow_reasons: list[str] = Field(default_factory=list)
    # Legacy pick was only acceptable/unknown, the shadow pick was stronger,
    # and the decision did not verify. A flag to investigate, never a cause.
    potential_reference_selection_issue: bool = False

    # Normalized release class, reused from existing metadata extraction.
    release_source: str | None = None
    release_resolution: str | None = None
    release_edition: str | None = None
    release_group: str | None = None
    fps_relation: str | None = None
    media_type: str | None = None
    # Provider identity is a short enum-like label, not sensitive, and is what
    # makes provider-level confounding detectable during analysis.
    provider: str | None = None

    reasons: list[str] = Field(default_factory=list)

    def to_jsonl(self) -> str:
        payload = self.model_dump(mode="json")
        payload["timestamp"] = self.timestamp.isoformat()
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _release_meta(release: Any) -> dict[str, Any]:
    """Pull normalized release facts from the matcher's own metadata.

    Deliberately reads the fields ``extract_metadata``/``guessit`` already
    produced. No new parser exists solely for telemetry.
    """
    compat = getattr(release, "compatibility", None)
    out: dict[str, Any] = {
        "match_tier": None,
        "compatibility_score": None,
        "fps_relation": None,
        "provider": None,
    }
    if compat is not None:
        tier = getattr(compat, "match_tier", None)
        out["match_tier"] = getattr(tier, "name", None) or (str(tier) if tier else None)
        percentage = getattr(compat, "percentage", None)
        out["compatibility_score"] = float(percentage) if percentage is not None else None
        out["fps_relation"] = getattr(compat, "fps_relation", None)

    provider = getattr(release, "provider", None)
    if provider is None and isinstance(release, dict):
        provider = release.get("provider")
    # Provider labels are a small fixed set; anything unexpected is dropped
    # rather than stored, so this cannot become a free-text leak.
    text = str(provider or "").strip().lower()[:24]
    out["provider"] = text if text.isalnum() else None

    meta = getattr(release, "target_meta", None)
    if isinstance(meta, dict):
        out["release_source"] = meta.get("source")
        out["release_resolution"] = meta.get("resolution")
        out["release_edition"] = meta.get("edition")
        out["release_group"] = meta.get("release_group")
        out["media_type"] = meta.get("media_type")
    return out


class AuditLog:
    """Bounded, thread-safe, write-only audit sink.

    Disabled by default. When enabled, records go to an in-memory ring (for
    tests and ``/diagnostics``) and optionally to a JSONL file. There is no read
    path used by request handling, which is what makes the log shadow-only.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        path: str | Path | None = None,
        max_records: int = 2000,
        max_pending: int = 2000,
    ) -> None:
        self.enabled = enabled
        self.path = Path(path) if path else None
        self._records: deque[SyncDecisionRecord] = deque(maxlen=max_records)
        # Pending search-time predictions awaiting a serve-time verification.
        self._pending: dict[str, SyncDecisionRecord] = {}
        self._max_pending = max_pending
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {
            "records": 0,
            "search_records": 0,
            "serve_records": 0,
            "paired_records": 0,
            "no_fingerprint": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "alass_runs": 0,
            "ranking_changed": 0,
            "ranking_comparisons": 0,
        }

    # ------------------------- recording ------------------------------- #

    def record(self, record: SyncDecisionRecord) -> None:
        """Persist one record. Never raises into the request path."""
        if not self.enabled:
            return
        try:
            with self._lock:
                self._counters["records"] += 1
                self._counters[f"{record.phase}_records"] += 1
                if not record.has_video_fingerprint:
                    self._counters["no_fingerprint"] += 1
                if record.from_cache:
                    self._counters["cache_hits"] += 1
                else:
                    self._counters["cache_misses"] += 1
                if record.alass_applied:
                    self._counters["alass_runs"] += 1
                self._records.append(record)
            if self.path is not None:
                self._append_file(record)
        except Exception as exc:  # pragma: no cover - telemetry must never break serving
            logger.debug("[audit] record dropped: %s", exc)

    def _append_file(self, record: SyncDecisionRecord) -> None:
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.to_jsonl() + "\n")

    # --------------------- prediction / verification -------------------- #

    @staticmethod
    def pair_key(video_id: str | None, subtitle_id: str | None) -> str:
        return f"{video_id or '-'}|{subtitle_id or '-'}"

    def record_search(self, record: SyncDecisionRecord) -> None:
        """Record a search-time prediction and remember it for pairing."""
        self.record(record)
        if not self.enabled or record.verification != VerificationAvailability.PREDICTED.value:
            return
        try:
            with self._lock:
                if len(self._pending) >= self._max_pending:
                    # Bounded: drop the oldest rather than growing without limit.
                    self._pending.pop(next(iter(self._pending)), None)
                self._pending[self.pair_key(record.video_id, record.subtitle_id)] = record
        except Exception as exc:  # pragma: no cover
            logger.debug("[audit] pending pair dropped: %s", exc)

    def record_serve(self, record: SyncDecisionRecord) -> None:
        """Record a serve-time verification, annotated with any earlier prediction."""
        try:
            with self._lock:
                prior = self._pending.pop(
                    self.pair_key(record.video_id, record.subtitle_id), None
                )
        except Exception:  # pragma: no cover
            prior = None
        if prior is not None:
            record.prediction_state = prior.sync_state
            record.prediction_rule_at_search = prior.prediction_rule
            if prior.prediction_confidence is not None:
                record.prediction_confidence = prior.prediction_confidence
            try:
                with self._lock:
                    self._counters["paired_records"] += 1
            except Exception:  # pragma: no cover
                pass
        self.record(record)

    def record_ranking_diff(self, changed: bool) -> None:
        """Count whether sync evidence changed the user-visible order."""
        if not self.enabled:
            return
        try:
            with self._lock:
                self._counters["ranking_comparisons"] += 1
                if changed:
                    self._counters["ranking_changed"] += 1
        except Exception:  # pragma: no cover
            pass

    # ---------------------------- reading ------------------------------- #

    def records(self) -> list[SyncDecisionRecord]:
        with self._lock:
            return list(self._records)

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._pending.clear()
            for key in self._counters:
                self._counters[key] = 0


# Process-wide sink. Disabled unless explicitly enabled, so a default install
# records nothing and writes no files.
AUDIT_LOG = AuditLog()


def configure_audit(*, enabled: bool, path: str | Path | None, max_records: int) -> AuditLog:
    """Apply settings to the process-wide sink. Returns it for convenience."""
    AUDIT_LOG.enabled = enabled
    AUDIT_LOG.path = Path(path) if path else None
    AUDIT_LOG._records = deque(maxlen=max_records)  # noqa: SLF001 - configuration point
    return AUDIT_LOG


def record_for_evaluation(
    evaluation: Any,
    *,
    phase: str,
    release: Any = None,
    video_id: str | None = None,
    subtitle_id: str | None = None,
    language: str = "und",
    has_video_fingerprint: bool = False,
    request_context: str | None = None,
    from_cache: bool = False,
    prediction_rule: str | None = None,
    prediction_confidence: float | None = None,
    reference: Any = None,
) -> SyncDecisionRecord:
    """Build a record from an existing evaluation. Reuses its fields verbatim."""
    release_facts = _release_meta(release) if release is not None else {}
    record = SyncDecisionRecord(
        phase=phase,
        video_id=video_id,
        subtitle_id=subtitle_id,
        language=language,
        has_video_fingerprint=has_video_fingerprint,
        request_context=request_context,
        match_tier=release_facts.get("match_tier"),
        compatibility_score=release_facts.get("compatibility_score"),
        prediction_rule=prediction_rule,
        prediction_confidence=prediction_confidence,
        verification=getattr(evaluation, "verification", VerificationAvailability.UNKNOWN).value
        if hasattr(getattr(evaluation, "verification", None), "value")
        else str(getattr(evaluation, "verification", VerificationAvailability.UNKNOWN.value)),
        sync_state=getattr(evaluation, "sync_state", SyncState.UNVERIFIED).value
        if hasattr(getattr(evaluation, "sync_state", None), "value")
        else str(getattr(evaluation, "sync_state", SyncState.UNVERIFIED.value)),
        sync_confidence=getattr(evaluation, "sync_confidence", None),
        median_offset_ms=getattr(evaluation, "median_offset_ms", None),
        mad_offset_ms=getattr(evaluation, "mad_offset_ms", None),
        p95_offset_ms=getattr(evaluation, "p95_offset_ms", None),
        drift_ms_per_minute=getattr(evaluation, "drift_ms_per_minute", None),
        coverage_score=getattr(evaluation, "coverage_score", None),
        structural_similarity=getattr(evaluation, "structural_similarity", None),
        cut_verdict=getattr(evaluation, "cut_verdict", None)
        or (CutVerdict.UNKNOWN.value),
        alass_applied=bool(getattr(evaluation, "alass_applied", False)),
        alass_successful=bool(getattr(evaluation, "alass_successful", False)),
        from_cache=from_cache,
        reference_trust=getattr(evaluation, "reference_trust", None),
        reference_provider=release_facts.get("provider"),
        reference_from_cache=bool(getattr(evaluation, "reference_trust", None) == "verified"),
        reference_independent_sources=getattr(evaluation, "reference_independent_sources", None),
        reference_consensus=getattr(evaluation, "reference_consensus", None),
        shadow_reference_id=getattr(reference, "shadow_reference_id", None),
        shadow_reference_changed=bool(getattr(reference, "shadow_changed", False)),
        shadow_reference_trust=getattr(reference, "shadow_trust", None),
        shadow_reference_health=getattr(reference, "shadow_health", None),
        shadow_independent_groups=getattr(reference, "shadow_independent_groups", 0) or None,
        shadow_reasons=[
            sanitize_reason(r) for r in (getattr(reference, "shadow_reasons", None) or [])
        ],
        release_source=release_facts.get("release_source"),
        release_resolution=release_facts.get("release_resolution"),
        release_edition=release_facts.get("release_edition"),
        release_group=release_facts.get("release_group"),
        fps_relation=release_facts.get("fps_relation"),
        media_type=release_facts.get("media_type"),
        provider=release_facts.get("provider"),
        reasons=[sanitize_reason(r) for r in (getattr(evaluation, "reasons", None) or [])],
    )
    state_value = getattr(getattr(evaluation, "sync_state", None), "value", None) or str(
        getattr(evaluation, "sync_state", SyncState.UNVERIFIED.value)
    )
    # A candidate for investigation, never a proven cause: the legacy pick was
    # weaker, the shadow pick was stronger, and the decision did not verify.
    record.potential_reference_selection_issue = bool(
        record.shadow_reference_changed
        and record.shadow_reference_id
        and (getattr(reference, "reference_trust", None) or "")
        in ("acceptable", "unknown", "rejected")
        and (record.shadow_reference_trust or "") in ("verified", "strong")
        and state_value not in VERIFIED_OUTCOMES
    )
    return record
