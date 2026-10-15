"""NinjaSync Points: anchor-based synchronization fallback (Phase 1: offline prototype).

This module generates synchronization *candidates* from cue-level anchors. It
never serves, never caches, and never claims a result is trustworthy: every
candidate must still pass the existing verifier (``AlignmentAnalyzer``), which
remains the final authority. Alass remains the primary aligner; this engine
only runs where Alass already failed or is unavailable.

Pipeline::

    AnchorDetector  ->  AnchorSet  ->  AnchorFitter  ->  SyncModel
        ->  transform  ->  AlignmentAnalyzer.analyze  ->  best VERIFIED or nothing

What is deliberately NOT here (Phase 1 scope):

* **Serving.** There is no serve path, no cache write, and no ranking hook.
  ``HybridResult.best`` is set only for a candidate the analyzer measured as
  ``VERIFIED_*``; anything else yields ``best=None``, the engine-level
  equivalent of "serve the original".
* **Piecewise fitting.** Deferred explicitly: piecewise needs holdout
  validation and cut evidence to avoid memorizing anchors, and that
  evaluation set does not exist yet. Constant explains uniform shifts;
  linear explains drift; anything structural stays rejected.
* **ASS synchronization.** Out of scope; transforms operate on SRT cue lists.

Provenance honesty: candidates are verified with
``analyze(target, transformed, reference, alass_applied=True,
alass_successful=True)`` because that is the measurement path for "a
transformation was applied and produced usable output". Alass did NOT run for
these candidates, so every evaluation additionally carries an explicit reason
recording the true producer (``ninja_points_constant`` /
``ninja_points_linear``). The flags describe that output exists; the reason
describes where it came from. Nothing downstream may read the flags as "alass
ran" -- and in Phase 1 there is no downstream at all.
"""

from __future__ import annotations

import logging
import re
import statistics
from dataclasses import dataclass, field
from typing import Any

from app.services.sync.alignment import AlignmentAnalyzer, SyncState

logger = logging.getLogger(__name__)

#: Model/algorithm version. Baked into every candidate's provenance so a future
#: cache can distinguish results across anchor-extraction, fitting, threshold,
#: or transformation changes. Never reuse a version string after changing any
#: of those.
MODEL_VERSION = "ninja-points-v1"

#: Stable strategy identifiers for provenance and (future) cache keys.
STRATEGY_CONSTANT = "ninja_points_constant"
STRATEGY_LINEAR = "ninja_points_linear"

#: Hard cap on anchors processed per request. Fitting is O(n^2) worst case in
#: the clustering step, so this bounds CPU without touching accuracy: anchors
#: are taken earliest-first and 400 corroborated anchors already over-determine
#: any constant or linear fit.
MAX_ANCHORS = 400

#: Two anchors must agree within this to corroborate each other.
CLUSTER_TOLERANCE_MS = 1000

#: Minimum corroborating anchors for a constant fit. A single strong anchor can
#: define an offset arithmetically, but one anchor cannot distinguish a real
#: shift from a coincidence, so one anchor is never enough to fit.
MIN_CONSTANT_ANCHORS = 2

#: Minimum anchors and span for a linear fit. Two points always define a line,
#: which is exactly why two points alone are not trustworthy: the span and
#: residual requirements below carry the actual evidentiary weight.
MIN_LINEAR_ANCHORS = 3
MIN_LINEAR_SPAN_MS = 600_000  # 10 minutes of subtitle time

#: Plausible drift slope bounds. 23.976 <-> 25 fps conversions span
#: 0.959..1.043; the bounds admit those with margin and refuse anything that
#: looks like a speed change rather than a frame-rate conversion.
MIN_SLOPE = 0.95
MAX_SLOPE = 1.05

#: Worst residual any anchor may show against a fitted model. Above this the
#: anchor set is contradictory and the fit is refused rather than averaged.
MAX_FIT_RESIDUAL_MS = 2000.0

#: Minimum mean anchor confidence for a fit. Below this the anchors are guesses
#: wearing a confidence costume.
MIN_MEAN_CONFIDENCE = 0.5

#: Duration similarity gate for anchor candidacy. A 200ms blip and a 9s
#: exposition card that share normalized text are not the same cue.
MAX_DURATION_RATIO = 2.0

#: Timeout budget for the whole detect+fit+transform pass, in cues processed.
#: Fitting itself is pure Python arithmetic on at most MAX_ANCHORS anchors, so
#: this is inherently bounded; the constant documents the guarantee.

Cue = tuple[int, int, str]

_TAG_RE = re.compile(r"<[^>]*>|\{[^}]*\}|\[[^\]]*\]")

#: Maximum length of a cue that can be classified as translator/uploader
#: metadata. Attribution lines are short ("Translated by XYZ"); ordinary
#: dialogue that happens to contain an attribution-adjacent word is longer.
#: The cap keeps the classifier conservative: long lines are never metadata.
_METADATA_MAX_CHARS = 80

_URL_RE = re.compile(r"https?://|www\.|\bt\.me/", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}")
_SOCIAL_RE = re.compile(
    r"\b(?:telegram|discord|whatsapp|twitter|facebook|instagram|tiktok|youtube)\b"
    r"|discord\.gg",
    re.IGNORECASE,
)
_HANDLE_RE = re.compile(r"(?:^|[\s:(\[])(@[\w.]{2,})")
_TLD_RE = re.compile(
    r"\b[a-z0-9-]+\.(?:com|net|org|io|tv|me|info|biz|xyz|site|online|to|cc)\b",
    re.IGNORECASE,
)
_ATTRIB_EN_RE = re.compile(
    r"\b(?:translat(?:ed|ion|or|ing)|synced?|synchronis(?:ed|ation)"
    r"|timing|timed|timer|subtitles?|subbed|encoded|ripped|releases?d?"
    r"|edit(?:ed|or|ing)?|credits?)\b",
    re.IGNORECASE,
)
_BY_RE = re.compile(r"\bby\b", re.IGNORECASE)
# Arabic translator/timing/credit terminology. Bare names are deliberately NOT
# matched: "فلان" alone is ambiguous and could be dialogue. The attribution
# word plus brevity is what makes the classification high-confidence.
_ATTRIB_AR_RE = re.compile(
    r"ترجم[ةه]|مترجم|توقيت|مزامنة|تعديل|إعداد|اعداد|تنسيق|ترميز|سينك"
)


def _is_metadata_cue(text: str) -> bool:
    """True only for high-confidence translator/uploader metadata cues.

    Internal-only: nothing outside anchor filtering may import this, and it
    never leaves this module as a claim. Anything it does not recognize stays
    usable -- UNKNOWN means usable, and corroboration plus the verifier still
    judge every surviving anchor. Deliberately NOT detected: bare personal or
    group names (ambiguous), bare "thanks", episode titles, continuations.

    Checked against RAW cue text (not normalized text): normalization strips
    ``[...]`` spans, which is exactly where URLs and handles usually hide.

    Reuse investigation (deliberately NOT reusing
    ``subtitle_matcher._is_non_speech_text``): that classifier is tuned for
    intro-window stripping, where over-classification is contained (cues under
    60s, cluster-reset semantics). It matches bare words like "translation" or
    "subtitle" anywhere, so adopting it for whole-file anchor filtering would
    refuse legitimate mid-film dialogue such as "The translation was bad" --
    evidence destruction beyond this task's mandate. Anchors need the opposite
    trade: refuse only high-confidence metadata, preserve everything else. The
    overlap (URLs, handles, translation-by forms, music markers) is covered
    below independently; music-only cues never reach anchors anyway because
    normalization already empties them.
    """
    if not text or not text.strip():
        return False
    raw = text.strip()
    if _URL_RE.search(raw) or _EMAIL_RE.search(raw) or _SOCIAL_RE.search(raw):
        return True
    if _HANDLE_RE.search(raw) or _TLD_RE.search(raw):
        return True
    if len(raw) > _METADATA_MAX_CHARS:
        return False
    if _ATTRIB_AR_RE.search(raw):
        return True
    # English attribution needs a second signal (authorship or labeling):
    # "by" as a standalone word or a colon. "Stand by me" has no attribution
    # word; "the translation was bad" has no by/colon. Both stay usable.
    if _ATTRIB_EN_RE.search(raw) and (_BY_RE.search(raw) or ":" in raw):
        return True
    return False


def normalize_anchor_text(text: str) -> str:
    """Normalized cue text for anchor matching.

    Lowercase, markup stripped, whitespace collapsed. Deliberately deaf to
    hearing-impaired markers and styling: two cues that differ only in ``<i>``
    or ``[Music]`` are the same dialogue for timing purposes.
    """
    cleaned = _TAG_RE.sub(" ", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


@dataclass(frozen=True)
class SyncAnchor:
    """One measured correspondence between a subtitle cue and a reference cue."""

    subtitle_ms: int
    reference_ms: int
    confidence: float
    support: int
    residual_ms: float | None = None
    # Normalized cue text that produced this anchor. Defense in depth: the
    # detector already refuses metadata cues, and the fitters re-check this
    # field so a metadata cue can never reach a fit even if a future caller
    # bypasses the detector filter. Never logged; memory-only.
    source_text: str | None = None

    @property
    def offset_ms(self) -> float:
        return float(self.reference_ms - self.subtitle_ms)


@dataclass(frozen=True)
class AnchorSet:
    """An immutable collection of anchors. Fitting never mutates this."""

    anchors: tuple[SyncAnchor, ...] = ()

    def __len__(self) -> int:
        return len(self.anchors)

    def __iter__(self):
        return iter(self.anchors)

    def offsets(self) -> list[float]:
        return [a.offset_ms for a in self.anchors]

    def with_residuals(self, residuals: list[float | None]) -> AnchorSet:
        """Copy with per-anchor residuals attached. The original is untouched."""
        return AnchorSet(
            anchors=tuple(
                SyncAnchor(
                    subtitle_ms=a.subtitle_ms,
                    reference_ms=a.reference_ms,
                    confidence=a.confidence,
                    support=a.support,
                    residual_ms=r,
                    source_text=a.source_text,
                )
                for a, r in zip(self.anchors, residuals, strict=False)
            )
        )


class AnchorDetector:
    """Extract synchronization anchors from cue-level evidence.

    A candidate anchor needs ALL of: identical normalized text, uniqueness on
    both sides (repeated phrases like "Yes." / "Thanks." are excluded, not
    scored), duration similarity, and temporal plausibility. Survivors earn
    support from mutually consistent neighbours; anchors without corroboration
    are dropped, so a lone coincidence can never anchor a fit.
    """

    def __init__(
        self,
        *,
        max_offset_ms: int = 20_000,
        min_support: int = 2,
        max_anchors: int = MAX_ANCHORS,
    ) -> None:
        self.max_offset_ms = max_offset_ms
        self.min_support = min_support
        self.max_anchors = max_anchors

    @staticmethod
    def _index_cues(
        cues: list[Cue], stats: dict | None = None
    ) -> dict[str, list[int]]:
        """Index cues by normalized text, refusing metadata cues first.

        A translator credit, URL, or handle that happens to match on both
        sides looks exactly like a legitimate anchor to everything downstream,
        so it must never enter the index at all -- not as an anchor, not as
        corroboration support, not as a fitting point. ``stats``, when given,
        records how many cues were refused, so shadow telemetry can report the
        filter's effect without ever logging cue text.
        """
        index: dict[str, list[int]] = {}
        for i, (_, _, text) in enumerate(cues):
            if _is_metadata_cue(text):
                if stats is not None:
                    stats["metadata_skipped"] = stats.get("metadata_skipped", 0) + 1
                continue
            norm = normalize_anchor_text(text)
            if norm:
                index.setdefault(norm, []).append(i)
        return index

    def detect(
        self, target_cues: list[Cue], reference_cues: list[Cue],
        stats: dict | None = None,
    ) -> AnchorSet:
        """Return the corroborated anchor set. Inputs are never mutated."""
        ref_index = self._index_cues(reference_cues, stats)
        tgt_index = self._index_cues(target_cues, stats)

        raw: list[SyncAnchor] = []
        for norm, tgt_ids in tgt_index.items():
            ref_ids = ref_index.get(norm)
            # Uniqueness on BOTH sides: a phrase appearing twice anywhere is
            # not an anchor, however tempting.
            if ref_ids is None or len(tgt_ids) != 1 or len(ref_ids) != 1:
                continue
            ti = tgt_ids[0]
            ri = ref_ids[0]
            t_start, t_end, _ = target_cues[ti]
            r_start, r_end, _ = reference_cues[ri]
            t_dur = max(1, t_end - t_start)
            r_dur = max(1, r_end - r_start)
            if max(t_dur, r_dur) / min(t_dur, r_dur) > MAX_DURATION_RATIO:
                continue
            offset = r_start - t_start
            if abs(offset) > self.max_offset_ms:
                continue
            # Duration agreement feeds confidence; it never promotes alone.
            dur_agree = min(t_dur, r_dur) / max(t_dur, r_dur)
            confidence = round(0.5 + 0.5 * dur_agree, 3)
            raw.append(
                SyncAnchor(
                    subtitle_ms=t_start, reference_ms=r_start,
                    confidence=confidence, support=0, source_text=norm,
                )
            )

        # Corroboration: support = anchors (including self) within tolerance.
        # Deterministic order throughout: earliest subtitle time first.
        # Metadata cannot bootstrap a cluster: it never entered the index, so
        # it is absent from both the anchors and their support counts.
        raw.sort(key=lambda a: (a.subtitle_ms, a.reference_ms))
        corroborated: list[SyncAnchor] = []
        for anchor in raw:
            support = sum(
                1 for other in raw
                if abs(other.offset_ms - anchor.offset_ms) <= CLUSTER_TOLERANCE_MS
            )
            if support >= self.min_support:
                corroborated.append(
                    SyncAnchor(
                        subtitle_ms=anchor.subtitle_ms,
                        reference_ms=anchor.reference_ms,
                        confidence=anchor.confidence,
                        support=support,
                        source_text=anchor.source_text,
                    )
                )
        return AnchorSet(anchors=tuple(corroborated[: self.max_anchors]))


def _place_on_timeline(cues) -> list[Cue]:
    """Drop cues moved entirely before zero; clamp only a straddling start.

    Same policy as ``anchor_preshift``: clamping both ends produced ``(0, 0)``
    cues -- zero-duration entries piled at the start that the film never had.
    """
    return [(max(0, s), e, text) for s, e, text in cues if e > 0]


@dataclass(frozen=True)
class ConstantSyncModel:
    """``t' = t + offset`` with the offset being a robust cluster median."""

    offset_ms: float
    inliers: int
    strategy_id: str = STRATEGY_CONSTANT
    model_version: str = MODEL_VERSION

    def apply(self, cues: list[Cue]) -> list[Cue]:
        shift = int(round(self.offset_ms))
        return _place_on_timeline((s + shift, e + shift, text) for s, e, text in cues)

    def describe(self) -> str:
        return (
            f"constant offset={self.offset_ms:+.0f}ms "
            f"inliers={self.inliers} ({self.strategy_id}/{self.model_version})"
        )


@dataclass(frozen=True)
class LinearSyncModel:
    """``t' = a * t + b`` for frame-rate drift. Constant preferred on ties."""

    slope: float
    intercept_ms: float
    inliers: int
    strategy_id: str = STRATEGY_LINEAR
    model_version: str = MODEL_VERSION

    def apply(self, cues: list[Cue]) -> list[Cue]:
        return _place_on_timeline(
            (
                int(round(self.slope * s + self.intercept_ms)),
                int(round(self.slope * e + self.intercept_ms)),
                text,
            )
            for s, e, text in cues
        )

    def describe(self) -> str:
        return (
            f"linear a={self.slope:.6f} b={self.intercept_ms:+.0f}ms "
            f"inliers={self.inliers} ({self.strategy_id}/{self.model_version})"
        )


class AnchorFitter:
    """Fit sync models to an anchor set. Refusal is a normal outcome."""

    @staticmethod
    def _mean_confidence(anchors: list[SyncAnchor]) -> float:
        if not anchors:
            return 0.0
        return sum(a.confidence for a in anchors) / len(anchors)

    @staticmethod
    def _without_metadata(anchors: list[SyncAnchor]) -> list[SyncAnchor]:
        """Defense in depth: the detector already refuses metadata cues, and a
        fit must never depend on that having happened. Any anchor whose source
        text classifies as metadata is dropped here too, so metadata can reach
        neither an offset cluster nor a slope estimate even if a future caller
        constructs an AnchorSet by hand."""
        return [
            a for a in anchors
            if not (a.source_text and _is_metadata_cue(a.source_text))
        ]

    def fit_constant(self, anchor_set: AnchorSet) -> ConstantSyncModel | None:
        """Largest corroborating cluster; median offset; never the mean.

        The median is immune to a single wild anchor (the +12000ms example
        moves a mean to +4000ms and leaves the median at +2000ms). A cluster
        smaller than MIN_CONSTANT_ANCHORS is not evidence, it is coincidence.
        """
        anchors = self._without_metadata(list(anchor_set.anchors))
        if len(anchors) < MIN_CONSTANT_ANCHORS:
            return None
        if self._mean_confidence(anchors) < MIN_MEAN_CONFIDENCE:
            return None
        ordered = sorted(a.offset_ms for a in anchors)
        # Largest cluster within tolerance (deterministic: earliest wins ties).
        best: list[float] = []
        for i, start in enumerate(ordered):
            cluster = [o for o in ordered[i:] if o - start <= CLUSTER_TOLERANCE_MS]
            if len(cluster) > len(best):
                best = cluster
        if len(best) < MIN_CONSTANT_ANCHORS:
            return None
        offset = float(statistics.median(best))
        return ConstantSyncModel(offset_ms=offset, inliers=len(best))

    def fit_linear(self, anchor_set: AnchorSet) -> LinearSyncModel | None:
        """Least-squares drift fit with conservative acceptance gates.

        Two points always define a line, which is exactly why two points are
        not enough: acceptance additionally requires span, slope plausibility,
        confidence, and bounded residuals across the FULL anchor set (not just
        the fitted points), so contradictory evidence refuses rather than
        averages.
        """
        anchors = self._without_metadata(list(anchor_set.anchors))
        if len(anchors) < MIN_LINEAR_ANCHORS:
            return None
        if self._mean_confidence(anchors) < MIN_MEAN_CONFIDENCE:
            return None
        span = max(a.subtitle_ms for a in anchors) - min(a.subtitle_ms for a in anchors)
        if span < MIN_LINEAR_SPAN_MS:
            return None
        n = len(anchors)
        sx = sum(a.subtitle_ms for a in anchors)
        sy = sum(a.reference_ms for a in anchors)
        sxx = sum(a.subtitle_ms * a.subtitle_ms for a in anchors)
        sxy = sum(a.subtitle_ms * a.reference_ms for a in anchors)
        denom = n * sxx - sx * sx
        if denom == 0:
            return None
        slope = (n * sxy - sx * sy) / denom
        if not (MIN_SLOPE <= slope <= MAX_SLOPE):
            return None
        intercept = (sy - slope * sx) / n
        worst = max(
            abs(slope * a.subtitle_ms + intercept - a.reference_ms)
            for a in anchors
        )
        if worst > MAX_FIT_RESIDUAL_MS:
            return None
        return LinearSyncModel(
            slope=slope, intercept_ms=intercept, inliers=len(anchors)
        )


def render_srt(cues: list[Cue]) -> str:
    """Render cue tuples back to SRT text. Input list is never mutated."""

    def stamp(ms: int) -> str:
        ms = max(0, int(ms))
        h, rem = divmod(ms, 3_600_000)
        m, rem = divmod(rem, 60_000)
        s, ms = divmod(rem, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    if not cues:
        return ""
    return "\n\n".join(
        f"{i}\n{stamp(s)} --> {stamp(e)}\n{text}"
        for i, (s, e, text) in enumerate(cues, start=1)
    ) + "\n"


@dataclass
class HybridCandidate:
    """One transformed subtitle plus its independent verification."""

    strategy_id: str
    model_version: str
    description: str
    transformed_text: str
    evaluation: Any = None
    offset_ms: float | None = None
    slope: float | None = None

    @property
    def verified(self) -> bool:
        state = getattr(getattr(self.evaluation, "sync_state", None), "value", None)
        return state in (
            SyncState.VERIFIED_SYNCED.value,
            SyncState.VERIFIED_RESYNCED.value,
        )


@dataclass
class HybridResult:
    """Engine outcome. ``best`` is set ONLY for a VERIFIED candidate.

    There is no serving path here by construction: the only way a result
    leaves this object as "accepted" is through the verifier, and the caller
    (a future shadow/serving integration, not this module) decides what that
    means. Unverified output is never promoted, never cached, never served.
    """

    best: HybridCandidate | None
    candidates: list[HybridCandidate] = field(default_factory=list)
    anchor_count: int = 0
    # Non-dialogue cues refused before fitting: production-style intro
    # stripping plus this module's metadata filter. Reported so shadow
    # telemetry can measure the filter effect; never cue text.
    metadata_skipped: int = 0


class HybridSyncEngine:
    """Offline fallback engine: Alass first (when provided), then Ninja models.

    ``run`` accepts an optional already-computed Alass output and NEVER runs
    Alass itself: in the serving pipeline Alass runs first and this engine only
    sees the request when Alass failed. Model priority is structural, not
    scored: constant is attempted before linear, and linear is only attempted
    when constant did not verify, so a more complex model can never displace a
    simpler verified one. Every candidate is verified independently with the
    existing analyzer; the first VERIFIED candidate in that order wins.
    """

    def __init__(self, analyzer: AlignmentAnalyzer | None = None) -> None:
        self._analyzer = analyzer or AlignmentAnalyzer()
        self._fitter = AnchorFitter()

    def run(
        self,
        target_text: str,
        reference_text: str,
        *,
        alass_output: str | None = None,
        max_plausible_offset_ms: int = 20_000,
    ) -> HybridResult:
        """Generate, verify, and select. Pure w.r.t. serving and cache."""
        from app.services.subtitle_matcher import (
            first_dialogue_cluster,
            parse_srt_cues,
            strip_intro_nonspeech,
        )
        from app.utils.cleaners import strip_intro_credits

        # Mirror production preprocessing exactly: the serving pipeline strips
        # intro non-speech cues (translator branding, URLs, date cards) before
        # alignment, so shadow must measure the same documents production
        # would align -- otherwise a shifted credit cue pollutes residuals and
        # shadow systematically under-reports what production would verify.
        # Reference is never stripped: it is the trusted timing.
        stripped_count = 0
        try:
            ref_first_speech = first_dialogue_cluster(
                strip_intro_nonspeech(reference_text)
            )
            stripped_text = strip_intro_credits(target_text, ref_first_speech)
        except Exception:  # fail open to unstripped: verification still judges
            logger.debug("[ninja-points] intro strip skipped", exc_info=True)
            stripped_text = target_text
        target_cues = parse_srt_cues(stripped_text)
        try:
            stripped_count = max(
                0, len(parse_srt_cues(target_text)) - len(target_cues)
            )
        except Exception:
            stripped_count = 0
        candidates: list[HybridCandidate] = []

        if alass_output:
            evaluation = self._verify(target_cues, alass_output, reference_text,
                                      "alass", max_plausible_offset_ms)
            candidates.append(
                HybridCandidate(
                    strategy_id="alass",
                    model_version="alass",
                    description="alass output (verified here, never overridden)",
                    transformed_text=alass_output,
                    evaluation=evaluation,
                )
            )
            best_alass = next((c for c in candidates if c.verified), None)
            if best_alass is not None:
                # A verified Alass result is authoritative. Ninja models are
                # not even attempted: there is nothing left to prove.
                return HybridResult(best=best_alass, candidates=candidates,
                                    anchor_count=0)

        # Seed with production-stripped intro cues so telemetry reports the
        # full non-dialogue refusal count, not just this module's filter.
        detection_stats: dict = {"metadata_skipped": stripped_count}
        anchors = AnchorDetector().detect(
            [(s, e, t) for s, e, t in target_cues],
            parse_srt_cues(reference_text),
            stats=detection_stats,
        )
        anchor_count = len(anchors)
        metadata_skipped = int(detection_stats.get("metadata_skipped", 0))
        if len(anchors) >= MIN_CONSTANT_ANCHORS:
            model: ConstantSyncModel | LinearSyncModel | None = self._fitter.fit_constant(anchors)
            if model is not None:
                transformed = render_srt(model.apply(target_cues))
                evaluation = self._verify(
                    target_cues, transformed, reference_text,
                    model.strategy_id, max_plausible_offset_ms,
                )
                candidates.append(
                    HybridCandidate(
                        strategy_id=model.strategy_id,
                        model_version=model.model_version,
                        description=model.describe(),
                        transformed_text=transformed,
                        evaluation=evaluation,
                        offset_ms=model.offset_ms
                        if isinstance(model, ConstantSyncModel) else None,
                        slope=model.slope
                        if isinstance(model, LinearSyncModel) else None,
                    )
                )
                best_constant = next((c for c in candidates if c.verified), None)
                if best_constant is not None:
                    return HybridResult(best=best_constant, candidates=candidates,
                                        anchor_count=anchor_count,
                                        metadata_skipped=metadata_skipped)

        if len(anchors) >= MIN_LINEAR_ANCHORS:
            model = self._fitter.fit_linear(anchors)
            if model is not None:
                transformed = render_srt(model.apply(target_cues))
                evaluation = self._verify(
                    target_cues, transformed, reference_text,
                    model.strategy_id, max_plausible_offset_ms,
                )
                candidates.append(
                    HybridCandidate(
                        strategy_id=model.strategy_id,
                        model_version=model.model_version,
                        description=model.describe(),
                        transformed_text=transformed,
                        evaluation=evaluation,
                        offset_ms=model.offset_ms
                        if isinstance(model, ConstantSyncModel) else None,
                        slope=model.slope
                        if isinstance(model, LinearSyncModel) else None,
                    )
                )
                best_linear = next((c for c in candidates if c.verified), None)
                if best_linear is not None:
                    return HybridResult(best=best_linear, candidates=candidates,
                                        anchor_count=anchor_count,
                                        metadata_skipped=metadata_skipped)

        return HybridResult(best=None, candidates=candidates,
                            anchor_count=anchor_count,
                                        metadata_skipped=metadata_skipped)

    def _verify(
        self,
        target_cues: list[Cue],
        transformed: str,
        reference_text: str,
        strategy_id: str,
        max_plausible_offset_ms: int,
    ):
        """Verify one candidate with the existing analyzer. No exceptions."""
        try:
            evaluation = self._analyzer.analyze(
                target_cues,
                transformed,
                reference_text,
                alass_applied=True,
                alass_successful=True,
                max_plausible_offset_ms=max_plausible_offset_ms,
            )
        except Exception as exc:  # fail closed: verification failure is not verification
            logger.warning("[ninja-points] verification raised: %s", exc)
            return None
        if evaluation is not None:
            evaluation.reasons.append(
                f"{strategy_id}: transformed without alass; "
                "verification measures output only"
            )
        return evaluation

    @staticmethod
    def summarize(result: HybridResult) -> dict[str, Any]:
        """Shadow-mode structured summary. Counts and metrics only."""
        summary: dict[str, Any] = {
            "candidates": len(result.candidates),
            "served": False,  # Phase 1 never serves; the key exists so a
            # future integration must flip it deliberately, in review.
            "best_strategy": result.best.strategy_id if result.best else None,
            "best_model_version": result.best.model_version if result.best else None,
        }
        for cand in result.candidates:
            evaluation = cand.evaluation
            summary[cand.strategy_id] = {
                "verified": cand.verified,
                "state": getattr(getattr(evaluation, "sync_state", None), "value", None),
                "median_residual": getattr(evaluation, "median_offset_ms", None),
                "mad": getattr(evaluation, "mad_offset_ms", None),
                "p95": getattr(evaluation, "p95_offset_ms", None),
                "structural": getattr(evaluation, "structural_similarity", None),
                "coverage": getattr(evaluation, "coverage_score", None),
            }
        return summary
