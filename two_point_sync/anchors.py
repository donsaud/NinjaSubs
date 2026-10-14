"""Dialogue-anchor discovery for Two-Point Auto-Sync.

Finds verified corresponding dialogue events (Target <-> Reference) using
sanitizer eligibility, normalized dialogue text, dialogue sequence and
neighbouring context, distinctiveness, chronological ordering, temporal
separation, and consistency with independently discovered correspondences.

Segmentation differences are first-class: besides 1:1 cue matches the
discovery supports bounded 1:N and N:1 grouped correspondences over dialogue
event units. A cue holding several dash-led speaker lines contributes one
event unit per line; plain wrapped lines stay a single event. Groups are
accepted on exact joined-text equality or on anchored evidence (an exact unit
match extended by adjacent marker/exact units), never on timestamp proximity
alone.

When strong correspondences across the full span support two incompatible
timing regimes, discovery returns an explicit NONLINEAR_DISCONTINUITY result
instead of a forced global transform. It never assumes equal cue indices or
counts, and returns insufficient-evidence rather than guessing.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, replace

from two_point_sync.parser import Cue
from two_point_sync.sanitizer import AnchorEligibility, classify_all
from two_point_sync.transform import (
    LinearTransform,
    TransformError,
    fit_from_anchors,
    fit_least_squares,
)

# ---------------------------------------------------------------------------
# Matching configuration (documented, deterministic)
# ---------------------------------------------------------------------------

INLIER_TOLERANCE_MS = 2000.0
MIN_SEPARATION_FRACTION = 0.25
MIN_SEPARATION_FLOOR_MS = 30_000
MIN_INLIERS = 3
MAX_COMBINATIONS_PER_KEY = 4
MIN_SCALE = 0.85
MAX_SCALE = 1.18
TERMINAL_WINDOW = 6
TERMINAL_SHARED_PREFIX_TOKENS = 2
TERMINAL_MIN_TOKENS = 3
# A primary anchor must clear this confidence bar and must not be a bare
# generic singleton without distinguishing neighbour context.
STRONG_CONFIDENCE = 0.40
# Bounded grouping: at most this many event units per side, all adjacent with
# small inter-cue gaps (a re-segmented utterance stays contiguous).
MAX_GROUP_UNITS = 3
MAX_GROUP_GAP_MS = 10_000
MIN_GROUP_TEXT_LEN = 12
# Regime analysis: clusters of strong, order-preserving correspondences whose
# implied offset disagrees with the primary model by more than this are a
# contradictory timing regime, not noise.
REGIME_OFFSET_TOLERANCE_MS = 5000.0
REGIME_MIN_SUPPORT = 3
REGIME_MIN_SPAN_MS = 10_000
CONTRADICTION_MIN_OFFSET_DIFF_MS = 30_000.0

GENERIC_SINGLETON_NORMS = frozenset(
    {
        "نعم",
        "لا",
        "اجل",
        "حسنا",
        "حاضر",
        "شكرا",
        "عفوا",
        "مرحبا",
        "اهلا",
        "اسف",
        "لماذا",
        "yes",
        "no",
        "ok",
        "okay",
        "thanks",
        "thank you",
        "please",
        "hey",
        "hi",
        "wow",
        "why",
    }
)

# Structural recap openers ("Previously on …") are matched marker-to-marker
# only; they are alignment context, never dialogue evidence on their own.
RECAP_MARKER_TOKENS = frozenset(
    {
        "سابقا",
        "السابقه",
        "السابق",
        "previously",
        "recap",
    }
)
RECAP_MARKER_MAX_TOKENS = 6

_TAG_RE = re.compile(r"\{[^}]*\}|<[^>]*>")
_BIDI_RE = re.compile(r"[‎‏‪‫‬‭‮⁦⁧⁨⁩﻿⁠]")
_PUNCT_RE = re.compile(r"[‐‑‒–—―_~*\"'`´‘’“”«»()\[\]{}:;،,。.!?؟…/\\|@#&+=\-–—]")
_DASH_LEADER_RE = re.compile(r"^[\-–—―]+\s*(\S.*)$")


def normalize_text(value: str) -> str:
    """Conservatively normalise cue text for matching (originals untouched)."""
    text = _TAG_RE.sub(" ", value)
    text = _BIDI_RE.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("ـ", "")
    text = re.sub(r"[\u064b-\u0652\u0670]", "", text)
    for variant in ("\u0623", "\u0625", "\u0622"):
        text = text.replace(variant, "\u0627")
    text = text.replace("\u0629", "\u0647").replace("\u0649", "\u064a")
    text = text.replace("\u0624", "\u0648").replace("\u0626", "\u064a")
    text = _PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text


def _visual_lines(text: str) -> list[str]:
    """Split a cue body into cleaned visual lines (tags/bidi stripped)."""
    cleaned = _TAG_RE.sub(" ", text)
    cleaned = _BIDI_RE.sub("", cleaned)
    return [line.strip() for line in cleaned.split("\n") if line.strip() != ""]


def _is_recap_marker(norm: str) -> bool:
    """Return True for short structural recap openers (never dialogue)."""
    tokens = norm.split()
    if not tokens or len(tokens) > RECAP_MARKER_MAX_TOKENS:
        return False
    return any(token in RECAP_MARKER_TOKENS for token in tokens)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Anchor:
    """One verified Target <-> Reference dialogue correspondence.

    Grouped correspondences list every member cue number; ``target_number``
    / ``reference_number`` (and the timestamps) always describe the first
    member of each side, i.e. the dialogue event onset.
    """

    target_number: int
    reference_number: int
    target_ms: int
    reference_ms: int
    evidence: tuple[str, ...]
    confidence: float
    target_numbers: tuple[int, ...] = ()
    reference_numbers: tuple[int, ...] = ()


@dataclass(frozen=True)
class Regime:
    """One timing regime: a coherent set of correspondences sharing an offset."""

    label: str
    scale: float
    offset_ms: float
    implied_offset_ms: float
    support: int
    span_start_ms: int
    span_end_ms: int


@dataclass(frozen=True)
class CandidateRejection:
    """Why a candidate correspondence was not used as a primary anchor."""

    target_number: int | None
    reference_number: int | None
    reason: str


@dataclass(frozen=True)
class AnchorResult:
    """Outcome of automatic anchor discovery."""

    accepted: bool
    first_anchor: Anchor | None = None
    last_anchor: Anchor | None = None
    scale: float | None = None
    offset_ms: float | None = None
    checkpoints: tuple[Anchor, ...] = ()
    inlier_count: int = 0
    temporal_separation_ms: int = 0
    reasons: tuple[str, ...] = ()
    rejections: tuple[CandidateRejection, ...] = ()
    diagnostics: dict[str, str] = field(default_factory=dict)
    # Timing regimes found across the full span. On NONLINEAR_DISCONTINUITY
    # the first regime is the primary (majority) model and the rest are
    # contradictory; ``checkpoints`` then documents primary-regime support
    # only and must not be mistaken for global validation.
    regimes: tuple[Regime, ...] = ()
    contradictory: tuple[Anchor, ...] = ()


@dataclass(frozen=True)
class _Unit:
    """One dialogue event unit: a whole cue, or one dash-led line of a cue."""

    cue_idx: int
    line_idx: int | None
    norm: str
    start_ms: int
    marker: bool


@dataclass(frozen=True)
class _Candidate:
    target_units: tuple[int, ...]
    reference_units: tuple[int, ...]
    key: str
    confidence: float
    evidence: tuple[str, ...]
    grouped: bool

    @property
    def target_idx(self) -> int:
        return self.target_units[0]

    @property
    def reference_idx(self) -> int:
        return self.reference_units[0]


def _distinctiveness_confidence(key: str, combos: int) -> float:
    tokens = key.split()
    base = 0.55
    base += min(0.30, 0.05 * max(0, len(tokens) - 1))
    base += min(0.10, 0.01 * max(0, len(key) - 12))
    if key in GENERIC_SINGLETON_NORMS:
        base -= 0.30
    if combos > 1:
        base -= 0.10 * combos
    return max(0.05, min(0.95, base))


def _build_units(
    cues: list[Cue], scores: list[AnchorEligibility]
) -> tuple[list[_Unit], list[str]]:
    """Split cues into dialogue event units.

    A cue with several dash-led speaker lines contributes one unit per line
    (separate dialogue events); anything else — including plain wrapped
    lines — stays a single event unit. Ineligible cues contribute no units.
    """
    units: list[_Unit] = []
    norms: list[str] = []
    for cue_idx, cue in enumerate(cues):
        if not scores[cue_idx].eligible:
            continue
        lines = _visual_lines(cue.text)
        dashed = [match.group(1) for line in lines if (match := _DASH_LEADER_RE.match(line))]
        if len(lines) >= 2 and len(dashed) >= 2:
            for line_idx, event in enumerate(dashed):
                norm = normalize_text(event)
                if not norm:
                    continue
                units.append(
                    _Unit(
                        cue_idx=cue_idx,
                        line_idx=line_idx,
                        norm=norm,
                        start_ms=cue.start_ms,
                        marker=_is_recap_marker(norm),
                    )
                )
                norms.append(norm)
        else:
            norm = normalize_text(cue.text)
            if not norm:
                continue
            units.append(
                _Unit(
                    cue_idx=cue_idx,
                    line_idx=None,
                    norm=norm,
                    start_ms=cue.start_ms,
                    marker=_is_recap_marker(norm),
                )
            )
            norms.append(norm)
    return units, norms


def _neighbour_norms(norms: list[str], position: int) -> set[str]:
    neighbours: set[str] = set()
    for offset in (-1, 1):
        neighbour = position + offset
        if 0 <= neighbour < len(norms) and norms[neighbour]:
            neighbours.add(norms[neighbour])
    return neighbours


def _unit_key(units: list[_Unit], positions: tuple[int, ...]) -> str:
    return " ".join(units[position].norm for position in positions)


def _windows(units: list[_Unit], size: int) -> list[tuple[int, ...]]:
    """Consecutive unit windows of ``size`` with small inter-cue gaps."""
    found: list[tuple[int, ...]] = []
    for start in range(len(units) - size + 1):
        window = tuple(range(start, start + size))
        ok = True
        for first, second in zip(window, window[1:], strict=False):
            gap = units[second].start_ms - units[first].start_ms
            if gap < 0 or gap > MAX_GROUP_GAP_MS:
                ok = False
                break
            if units[second].cue_idx == units[first].cue_idx:
                continue
            if units[second].cue_idx != units[first].cue_idx + 1:
                ok = False
                break
        if ok:
            found.append(window)
    return found


def _build_candidates(
    target_cues: list[Cue],
    reference_cues: list[Cue],
    target_scores: list[AnchorEligibility],
    reference_scores: list[AnchorEligibility],
    rejections: list[CandidateRejection],
) -> list[_Candidate]:
    target_units, target_norms = _build_units(target_cues, target_scores)
    reference_units, reference_norms = _build_units(reference_cues, reference_scores)

    # Record why ineligible cues never became candidates (diagnostics, capped).
    for position, cue in enumerate(target_cues):
        if normalize_text(cue.text) and not target_scores[position].eligible:
            if len(rejections) < 50:
                rejections.append(
                    CandidateRejection(
                        target_number=cue.number,
                        reference_number=None,
                        reason=f"target_ineligible:{';'.join(target_scores[position].reasons)}",
                    )
                )
    for position, cue in enumerate(reference_cues):
        if normalize_text(cue.text) and not reference_scores[position].eligible:
            if len(rejections) < 100:
                rejections.append(
                    CandidateRejection(
                        target_number=None,
                        reference_number=cue.number,
                        reason=(
                            "reference_ineligible:"
                            f"{';'.join(reference_scores[position].reasons)}"
                        ),
                    )
                )

    candidates: list[_Candidate] = []

    def _confidence_for(
        key: str,
        combos: int,
        target_pos: int,
        reference_pos: int,
        extra_evidence: list[str],
    ) -> tuple[float, tuple[str, ...]]:
        evidence = [f"exact_normalized_match:{key[:48]}", *extra_evidence]
        if _neighbour_norms(target_norms, target_pos) & _neighbour_norms(
            reference_norms, reference_pos
        ):
            evidence.append("neighbour_context_match")
        confidence = _distinctiveness_confidence(key, combos)
        if "neighbour_context_match" in evidence:
            confidence = min(0.98, confidence + 0.10)
        return max(0.05, confidence), tuple(evidence)

    # -- 1:1 unit matches (whole cues, or individual dash-led cue lines) ----
    target_index: dict[str, list[int]] = {}
    for position, unit in enumerate(target_units):
        if not unit.marker:
            target_index.setdefault(unit.norm, []).append(position)
    reference_index: dict[str, list[int]] = {}
    for position, unit in enumerate(reference_units):
        if not unit.marker:
            reference_index.setdefault(unit.norm, []).append(position)
    for key in sorted(set(target_index) & set(reference_index)):
        left = target_index[key]
        right = reference_index[key]
        combos = len(left) * len(right)
        if combos > MAX_COMBINATIONS_PER_KEY:
            for target_pos in left:
                for reference_pos in right:
                    if len(rejections) >= 500:
                        break
                    rejections.append(
                        CandidateRejection(
                            target_number=target_cues[target_units[target_pos].cue_idx].number,
                            reference_number=reference_cues[
                                reference_units[reference_pos].cue_idx
                            ].number,
                            reason=f"ambiguous_repeated_phrase:{key[:48]}",
                        )
                    )
            continue
        for target_pos in left:
            for reference_pos in right:
                extra: list[str] = []
                if (
                    target_units[target_pos].line_idx is not None
                    or reference_units[reference_pos].line_idx is not None
                ):
                    extra.append("cue_line_match")
                confidence, evidence = _confidence_for(
                    key, combos, target_pos, reference_pos, extra
                )
                # Fold sanitizer eligibility confidence in (eligibility itself
                # was already enforced when building units).
                confidence = min(
                    confidence,
                    (
                        target_scores[target_units[target_pos].cue_idx].confidence
                        + reference_scores[reference_units[reference_pos].cue_idx].confidence
                    )
                    / 2.0
                    + 0.25,
                )
                candidates.append(
                    _Candidate(
                        target_units=(target_pos,),
                        reference_units=(reference_pos,),
                        key=key,
                        confidence=max(0.05, confidence),
                        evidence=evidence,
                        grouped=False,
                    )
                )

    # -- Bounded exact-join groups (1:N / N:1 re-segmentation) ---------------
    exact_group_windows: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for size_t in range(1, MAX_GROUP_UNITS + 1):
        for size_r in range(1, MAX_GROUP_UNITS + 1):
            if size_t == 1 and size_r == 1:
                continue
            target_windows = _windows(target_units, size_t)
            reference_windows = _windows(reference_units, size_r)
            target_keys: dict[str, list[tuple[int, ...]]] = {}
            for window in target_windows:
                if any(target_units[position].marker for position in window):
                    continue
                joined = _unit_key(target_units, window)
                if len(joined) >= MIN_GROUP_TEXT_LEN:
                    target_keys.setdefault(joined, []).append(window)
            reference_key_map: dict[str, list[tuple[int, ...]]] = {}
            for window in reference_windows:
                if any(reference_units[position].marker for position in window):
                    continue
                joined = _unit_key(reference_units, window)
                if len(joined) >= MIN_GROUP_TEXT_LEN:
                    reference_key_map.setdefault(joined, []).append(window)
            for joined in sorted(set(target_keys) & set(reference_key_map)):
                left_windows = target_keys[joined]
                right_windows = reference_key_map[joined]
                if len(left_windows) * len(right_windows) > MAX_COMBINATIONS_PER_KEY:
                    continue
                for target_window in left_windows:
                    for reference_window in right_windows:
                        exact_group_windows.add((target_window, reference_window))
                        evidence = (
                            f"exact_normalized_match:{joined[:48]}",
                            f"grouped_exact_match:{len(target_window)}v{len(reference_window)}",
                        )
                        confidence = min(
                            0.95, _distinctiveness_confidence(joined, 1) + 0.10
                        )
                        candidates.append(
                            _Candidate(
                                target_units=target_window,
                                reference_units=reference_window,
                                key=joined,
                                confidence=confidence,
                                evidence=evidence,
                                grouped=True,
                            )
                        )

    # -- Marker-anchored groups (exact unit + adjacent marker context) -------
    # Seeds are exact 1:1 unit matches plus marker-class pairings (structural
    # recap openers pair by class, since reworded markers never match exactly).
    target_markers = [pos for pos, unit in enumerate(target_units) if unit.marker]
    reference_markers = [pos for pos, unit in enumerate(reference_units) if unit.marker]
    candidates.extend(
        _anchored_groups(
            target_units,
            reference_units,
            candidates,
            exact_group_windows,
            target_markers,
            reference_markers,
        )
    )

    candidates.sort(
        key=lambda item: (
            target_units[item.target_idx].start_ms,
            reference_units[item.reference_idx].start_ms,
            item.key,
        )
    )
    return candidates


def _anchored_groups(
    target_units: list[_Unit],
    reference_units: list[_Unit],
    seeds: list[_Candidate],
    exact_windows: set[tuple[tuple[int, ...], tuple[int, ...]]],
    target_markers: list[int],
    reference_markers: list[int],
) -> list[_Candidate]:
    """Extend exact unit matches with adjacent marker/exact unit context.

    Covers re-segmented dialogue where one side groups a recap marker (or a
    reworded line) together with an exactly matching line, e.g. Target cues
    #2+#3 ("Previously on Dexter" + "Oh my god") inside Reference cue #1.
    Bounded to MAX_GROUP_UNITS cues per side; deterministic.
    """
    # (target unit, reference unit, seed is exact pair)
    seed_pairs: list[tuple[int, int, bool]] = []
    for seed in seeds:
        if seed.grouped or len(seed.target_units) != 1 or len(seed.reference_units) != 1:
            continue
        seed_target = seed.target_units[0]
        seed_reference = seed.reference_units[0]
        if target_units[seed_target].marker or reference_units[seed_reference].marker:
            continue
        seed_pairs.append((seed_target, seed_reference, True))
    # Marker-class seeds: structural recap openers pair by class. Bounded:
    # too many markers means no distinctive structural anchor exists.
    if len(target_markers) <= 6 and len(reference_markers) <= 6:
        for target_pos in target_markers:
            for reference_pos in reference_markers:
                seed_pairs.append((target_pos, reference_pos, False))
    grouped: list[_Candidate] = []
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for seed_target, seed_reference, seed_exact in seed_pairs:
        exact_pairs = 1 if seed_exact else 0
        # Grow cue windows around the seed while neighbours pair up as
        # exact-or-marker unit correspondences, in order.
        target_positions = [seed_target]
        reference_positions = [seed_reference]
        for direction in (-1, 1):
            while True:
                next_target = target_positions[0] + direction if direction < 0 else target_positions[-1] + direction
                next_reference = (
                    reference_positions[0] + direction
                    if direction < 0
                    else reference_positions[-1] + direction
                )
                if not (
                    0 <= next_target < len(target_units)
                    and 0 <= next_reference < len(reference_units)
                ):
                    break
                unit_t = target_units[next_target]
                unit_r = reference_units[next_reference]
                pair_exact = unit_t.norm == unit_r.norm and not unit_t.marker
                pair_marker = unit_t.marker and unit_r.marker
                if not (pair_exact or pair_marker):
                    break
                cue_span_t = {target_units[position].cue_idx for position in target_positions} | {
                    unit_t.cue_idx
                }
                cue_span_r = {reference_units[position].cue_idx for position in reference_positions} | {
                    unit_r.cue_idx
                }
                if len(cue_span_t) > MAX_GROUP_UNITS or len(cue_span_r) > MAX_GROUP_UNITS:
                    break
                edge_t = target_positions[0] if direction < 0 else target_positions[-1]
                edge_r = reference_positions[0] if direction < 0 else reference_positions[-1]
                if (
                    abs(target_units[next_target].start_ms - target_units[edge_t].start_ms)
                    > MAX_GROUP_GAP_MS
                    or abs(
                        reference_units[next_reference].start_ms
                        - reference_units[edge_r].start_ms
                    )
                    > MAX_GROUP_GAP_MS
                ):
                    break
                if direction < 0:
                    target_positions.insert(0, next_target)
                    reference_positions.insert(0, next_reference)
                else:
                    target_positions.append(next_target)
                    reference_positions.append(next_reference)
                if pair_exact:
                    exact_pairs += 1
        target_window = tuple(target_positions)
        reference_window = tuple(reference_positions)
        if len(target_window) == 1 and len(reference_window) == 1:
            continue
        if exact_pairs < 1:
            # A marker seed alone is not evidence; the group must anchor on
            # at least one exact unit correspondence.
            continue
        if (target_window, reference_window) in seen:
            continue
        seen.add((target_window, reference_window))
        if (target_window, reference_window) in exact_windows:
            continue
        joined = _unit_key(target_units, target_window)
        if len(joined) < MIN_GROUP_TEXT_LEN:
            continue
        cue_count_t = len({target_units[position].cue_idx for position in target_window})
        cue_count_r = len({reference_units[position].cue_idx for position in reference_window})
        if cue_count_t == 1 and cue_count_r == 1:
            continue
        seed_norm = target_units[seed_target].norm
        evidence = [
            f"exact_normalized_match:{seed_norm[:48]}",
            f"anchored_group_match:{cue_count_t}v{cue_count_r}_cues",
            "adjacent_marker_or_exact_context",
        ]
        if not seed_exact:
            evidence.append("marker_seeded_group")
        grouped.append(
            _Candidate(
                target_units=target_window,
                reference_units=reference_window,
                key=joined,
                confidence=min(0.90, _distinctiveness_confidence(joined, 1) + 0.05),
                evidence=tuple(evidence),
                grouped=True,
            )
        )
    return grouped


def _is_strong_candidate(candidate: _Candidate) -> bool:
    """Primary anchors need confidence plus distinguishing context."""
    if candidate.confidence < STRONG_CONFIDENCE:
        return False
    if "neighbour_context_match" in candidate.evidence:
        return True
    if candidate.grouped:
        return True
    return candidate.key not in GENERIC_SINGLETON_NORMS


def _model_for(
    first: _Candidate,
    second: _Candidate,
    target_units: list[_Unit],
    reference_units: list[_Unit],
) -> LinearTransform | None:
    t1 = target_units[first.target_idx].start_ms
    r1 = reference_units[first.reference_idx].start_ms
    t2 = target_units[second.target_idx].start_ms
    r2 = reference_units[second.reference_idx].start_ms
    if t2 <= t1 or r2 <= r1:
        return None
    try:
        model = fit_from_anchors(t1, r1, t2, r2)
    except TransformError:
        return None
    if not (MIN_SCALE <= model.scale <= MAX_SCALE):
        return None
    return model


def _inliers_for(
    model: LinearTransform,
    candidates: list[_Candidate],
    target_units: list[_Unit],
    reference_units: list[_Unit],
) -> list[_Candidate]:
    inliers: list[_Candidate] = []
    for candidate in candidates:
        predicted = model.scale * float(target_units[candidate.target_idx].start_ms)
        predicted += model.offset_ms
        residual = abs(predicted - float(reference_units[candidate.reference_idx].start_ms))
        if residual <= INLIER_TOLERANCE_MS:
            inliers.append(candidate)
    return inliers


def _cue_numbers(
    units: list[_Unit], positions: tuple[int, ...], cues: list[Cue]
) -> tuple[int, ...]:
    """Member cue numbers in order, consecutive duplicates collapsed."""
    numbers: list[int] = []
    for position in positions:
        number = cues[units[position].cue_idx].number
        if not numbers or numbers[-1] != number:
            numbers.append(number)
    return tuple(numbers)


def _to_anchor(
    candidate: _Candidate,
    target_units: list[_Unit],
    reference_units: list[_Unit],
    target_cues: list[Cue],
    reference_cues: list[Cue],
) -> Anchor:
    first_target = target_cues[target_units[candidate.target_units[0]].cue_idx]
    first_reference = reference_cues[reference_units[candidate.reference_units[0]].cue_idx]
    return Anchor(
        target_number=first_target.number,
        reference_number=first_reference.number,
        target_ms=first_target.start_ms,
        reference_ms=first_reference.start_ms,
        evidence=candidate.evidence,
        confidence=candidate.confidence,
        target_numbers=_cue_numbers(target_units, candidate.target_units, target_cues),
        reference_numbers=_cue_numbers(
            reference_units, candidate.reference_units, reference_cues
        ),
    )


def _scan_regimes(
    best_model: LinearTransform,
    candidates: list[_Candidate],
    target_units: list[_Unit],
    reference_units: list[_Unit],
    target_cues: list[Cue],
    reference_cues: list[Cue],
) -> tuple[Regime, list[Regime], list[Anchor]]:
    """Cluster strong correspondences by implied offset over the full span.

    Returns the primary regime plus any contradictory regimes (coherent,
    order-preserving correspondence sets with a materially different implied
    offset). Contradictory anchors (including grouped matches) are returned
    for explicit reporting and validator hardening.
    """
    strong = [item for item in candidates if _is_strong_candidate(item)]
    by_offset = sorted(
        (
            float(reference_units[item.reference_idx].start_ms)
            - best_model.scale * float(target_units[item.target_idx].start_ms),
            position,
            item,
        )
        for position, item in enumerate(strong)
    )
    clusters: list[list[tuple[float, int, _Candidate]]] = []
    for implied, _, item in by_offset:
        placed = False
        for cluster in clusters:
            if abs(implied - cluster[0][0]) <= REGIME_OFFSET_TOLERANCE_MS:
                cluster.append((implied, _, item))
                placed = True
                break
        if not placed:
            clusters.append([(implied, _, item)])
    primary_offset = best_model.offset_ms
    main_members = [
        item for implied, _, item in by_offset if abs(implied - primary_offset) <= INLIER_TOLERANCE_MS
    ]
    main_targets = [target_units[item.target_idx].start_ms for item in main_members]
    primary = Regime(
        label="primary",
        scale=best_model.scale,
        offset_ms=best_model.offset_ms,
        implied_offset_ms=primary_offset,
        support=len(main_members),
        span_start_ms=min(main_targets) if main_targets else 0,
        span_end_ms=max(main_targets) if main_targets else 0,
    )
    contradictory_regimes: list[Regime] = []
    contradictory_anchors: list[Anchor] = []
    for cluster in clusters:
        if len(cluster) < REGIME_MIN_SUPPORT:
            continue
        median_offset = sorted(implied for implied, _, _ in cluster)[len(cluster) // 2]
        if abs(median_offset - primary_offset) <= CONTRADICTION_MIN_OFFSET_DIFF_MS:
            continue
        members = [item for _, _, item in cluster]
        seen_signatures: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
        unique_members: list[_Candidate] = []
        for item in members:
            signature = (item.target_units, item.reference_units)
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_members.append(item)
        if len(unique_members) < REGIME_MIN_SUPPORT:
            continue
        members = unique_members

        def _order_key(item: _Candidate) -> tuple[tuple[int, int, int], ...]:
            key: list[tuple[int, int, int]] = []
            for units, pos in (
                (target_units, item.target_units[0]),
                (target_units, item.target_units[-1]),
                (reference_units, item.reference_units[0]),
                (reference_units, item.reference_units[-1]),
            ):
                line = units[pos].line_idx
                key.append((units[pos].start_ms, units[pos].cue_idx, -1 if line is None else line))
            return tuple(key)

        target_order = [_order_key(item)[:2] for item in members]
        reference_order = [_order_key(item)[2:] for item in members]
        # Order preservation: sorting by target position must yield
        # reference positions in the same (strictly increasing) order.
        by_target = sorted(range(len(members)), key=lambda i: target_order[i])
        reference_in_target_order = [reference_order[i] for i in by_target]
        if any(
            first >= second
            for first, second in zip(
                reference_in_target_order, reference_in_target_order[1:], strict=False
            )
        ):
            continue
        span_start = min(key[0][0] for key in target_order)
        span_end = max(key[0][0] for key in target_order)
        if span_end - span_start < REGIME_MIN_SPAN_MS:
            continue
        contradictory_regimes.append(
            Regime(
                label=f"contradictory_{len(contradictory_regimes) + 1}",
                scale=best_model.scale,
                offset_ms=median_offset,
                implied_offset_ms=median_offset,
                support=len(cluster),
                span_start_ms=span_start,
                span_end_ms=span_end,
            )
        )
        for _, _, item in sorted(cluster, key=lambda pair: target_units[pair[2].target_idx].start_ms):
            contradictory_anchors.append(
                _to_anchor(item, target_units, reference_units, target_cues, reference_cues)
            )
    return primary, contradictory_regimes, contradictory_anchors


def _terminal_extension(
    model: LinearTransform,
    target_cues: list[Cue],
    reference_cues: list[Cue],
    target_scores: list[AnchorEligibility],
    reference_scores: list[AnchorEligibility],
    current_last_target_ms: int,
) -> Anchor | None:
    """Confirm a terminal dialogue pair consistent with the global model.

    The pair must carry its own textual evidence (several shared opening
    tokens over a multi-token utterance) and agree with the independently
    fitted model; timestamp proximity alone is never sufficient. Bare generic
    singletons can never become terminal anchors this way.
    """
    eligible_target = [
        position
        for position in range(len(target_cues) - 1, -1, -1)
        if target_scores[position].eligible and normalize_text(target_cues[position].text)
    ][:TERMINAL_WINDOW]
    eligible_reference = [
        position
        for position in range(len(reference_cues) - 1, -1, -1)
        if reference_scores[position].eligible
        and normalize_text(reference_cues[position].text)
    ][:TERMINAL_WINDOW]
    best: Anchor | None = None
    best_residual = INLIER_TOLERANCE_MS
    for target_pos in eligible_target:
        target_tokens = normalize_text(target_cues[target_pos].text).split()
        if len(target_tokens) < TERMINAL_MIN_TOKENS:
            continue
        for reference_pos in eligible_reference:
            reference_tokens = normalize_text(reference_cues[reference_pos].text).split()
            shared = 0
            for left, right in zip(target_tokens, reference_tokens, strict=False):
                if left == right:
                    shared += 1
                else:
                    break
            if shared < TERMINAL_SHARED_PREFIX_TOKENS:
                continue
            predicted = (
                model.scale * float(target_cues[target_pos].start_ms) + model.offset_ms
            )
            residual = abs(predicted - float(reference_cues[reference_pos].start_ms))
            if residual > best_residual:
                continue
            if target_cues[target_pos].start_ms <= current_last_target_ms:
                continue
            best_residual = residual
            best = Anchor(
                target_number=target_cues[target_pos].number,
                reference_number=reference_cues[reference_pos].number,
                target_ms=target_cues[target_pos].start_ms,
                reference_ms=reference_cues[reference_pos].start_ms,
                evidence=(
                    f"terminal_dialogue_shared_prefix:{shared}_tokens",
                    f"model_consistent_residual_ms:{residual:.1f}",
                    "neighbourhood_last_dialogue",
                ),
                confidence=0.90,
                target_numbers=(target_cues[target_pos].number,),
                reference_numbers=(reference_cues[reference_pos].number,),
            )
    return best


def dialogue_correspondences(
    target_cues: list[Cue], reference_cues: list[Cue],
) -> tuple[tuple[int, int, int, tuple[int, ...], tuple[int, ...]], ...]:
    """Unique textual events for timing measurement, without fitting a clock.

    Reuse bounded exact unit/concatenated-group matching. Repeated text with
    multiple placements is excluded, never resolved by timestamp proximity.
    Marker-only and approximate adjacent groups cannot verify a timestamp.
    Later speaker lines have no measured start within their containing cue.
    Exclude those boundaries rather than treating them as zero-error evidence.
    """
    target_scores, reference_scores = classify_all(target_cues), classify_all(reference_cues)
    target_units, _ = _build_units(target_cues, target_scores)
    reference_units, _ = _build_units(reference_cues, reference_scores)
    candidates = _build_candidates(target_cues, reference_cues, target_scores, reference_scores, [])
    placements: dict[str, set[tuple[tuple[int, ...], tuple[int, ...]]]] = {}
    for item in candidates:
        placements.setdefault(item.key, set()).add((item.target_units, item.reference_units))
    used_target: set[int] = set()
    used_reference: set[int] = set()
    records = []
    for item in candidates:
        if len(placements[item.key]) != 1 or not _is_strong_candidate(item):
            continue
        # A unique short utterance in these two files is not necessarily unique
        # dialogue in the film (e.g. Arabic "hello" or "stop"). Require lexical
        # context, not a plausible clock, for such matches.
        if len(set(item.key.split())) < 3 and "neighbour_context_match" not in item.evidence:
            continue
        if not any(e.startswith("exact_normalized_match:") for e in item.evidence):
            continue
        if item.grouped and not any(e.startswith("grouped_exact_match:") for e in item.evidence):
            continue
        left, right = target_units[item.target_idx], reference_units[item.reference_idx]
        if left.line_idx not in (None, 0) or right.line_idx not in (None, 0):
            continue
        target_numbers = _cue_numbers(target_units, item.target_units, target_cues)
        reference_numbers = _cue_numbers(reference_units, item.reference_units, reference_cues)
        if used_target.intersection(target_numbers) or used_reference.intersection(reference_numbers):
            continue
        used_target.update(target_numbers)
        used_reference.update(reference_numbers)
        records.append((left.start_ms, right.start_ms, left.start_ms, target_numbers, reference_numbers))
    # Order-consistent measurement only: dialogue is sequential, so a pairing
    # whose reference onset runs backward against the target order is a
    # wrong-instance match (the same sentence subtitled at two story points,
    # with each side keeping a different one), never timing evidence. Keep
    # the longest chronologically consistent run; the choice is residual-blind
    # (text order only) and deterministic, so it can never manufacture a pass:
    # whatever remains must still clear every sufficiency threshold downstream.
    records = _longest_chronological_run(records)
    # The verifier holds fitting endpoints out of its timing statistics. Keep
    # the full correspondence available for output-integrity checks too.
    return tuple(records)


def _longest_chronological_run(
    records: list[tuple[int, int, int, tuple[int, ...], tuple[int, ...]]],
) -> list[tuple[int, int, int, tuple[int, ...], tuple[int, ...]]]:
    """Longest strictly reference-increasing subsequence (target order kept).

    Records arrive in target-onset order. A repeated line matched to the
    wrong story instance is the only way a later target onset pairs with an
    earlier-or-equal reference onset; dropping those pairings (not cues, not
    dialogue) keeps the majority story order. Ties keep the earliest target
    pairing; overall first-maximal reconstruction makes it deterministic.
    Bounded O(n^2); correspondence sets are small by construction.
    """
    n = len(records)
    if n < 2:
        return records
    best_len = [1] * n
    prev = [-1] * n
    for i in range(n):
        ref_i = records[i][1]
        for j in range(i):
            if records[j][1] < ref_i and best_len[j] + 1 > best_len[i]:
                best_len[i] = best_len[j] + 1
                prev[i] = j
    end = max(range(n), key=lambda i: best_len[i])
    keep: list[int] = []
    while end != -1:
        keep.append(end)
        end = prev[end]
    return [records[i] for i in sorted(keep)]


def detect_anchors(
    target_cues: list[Cue],
    reference_cues: list[Cue],
    *,
    target_scores: list[AnchorEligibility] | None = None,
    reference_scores: list[AnchorEligibility] | None = None,
) -> AnchorResult:
    """Automatically discover two verified dialogue anchors."""
    if len(target_cues) < 2 or len(reference_cues) < 2:
        return AnchorResult(accepted=False, reasons=("insufficient_cues",))

    if target_scores is None:
        target_scores = classify_all(target_cues)
    if reference_scores is None:
        reference_scores = classify_all(reference_cues)

    rejections: list[CandidateRejection] = []
    candidates = _build_candidates(
        target_cues, reference_cues, target_scores, reference_scores, rejections
    )
    if len(candidates) < MIN_INLIERS:
        return AnchorResult(
            accepted=False,
            reasons=("insufficient_correspondence_evidence",),
            rejections=tuple(rejections),
            diagnostics={
                "candidates": str(len(candidates)),
                "detail": "no_textual_correspondence",
            },
        )

    target_units, _ = _build_units(target_cues, target_scores)
    reference_units, _ = _build_units(reference_cues, reference_scores)

    target_span = max(cue.start_ms for cue in target_cues) - min(
        cue.start_ms for cue in target_cues
    )
    min_separation = max(MIN_SEPARATION_FLOOR_MS, int(MIN_SEPARATION_FRACTION * target_span))

    best_model: LinearTransform | None = None
    best_inliers: list[_Candidate] = []
    best_pair: tuple[_Candidate, _Candidate] | None = None
    # Deterministic RANSAC over candidate pairs.
    for left_pos in range(len(candidates)):
        for right_pos in range(left_pos + 1, len(candidates)):
            first = candidates[left_pos]
            second = candidates[right_pos]
            separation = (
                target_units[second.target_idx].start_ms
                - target_units[first.target_idx].start_ms
            )
            if separation < min_separation:
                continue
            model = _model_for(first, second, target_units, reference_units)
            if model is None:
                continue
            inliers = _inliers_for(model, candidates, target_units, reference_units)
            if len(inliers) < MIN_INLIERS:
                continue
            inlier_span = (
                max(target_units[item.target_idx].start_ms for item in inliers)
                - min(target_units[item.target_idx].start_ms for item in inliers)
            )
            best_span = 0
            if best_inliers:
                best_span = (
                    max(target_units[item.target_idx].start_ms for item in best_inliers)
                    - min(target_units[item.target_idx].start_ms for item in best_inliers)
                )
            if (len(inliers) > len(best_inliers)) or (
                len(inliers) == len(best_inliers) and inlier_span > best_span
            ):
                best_model = model
                best_inliers = inliers
                best_pair = (first, second)
            if len(best_inliers) >= len(candidates):
                # Exact early exit (no behavior change): full coverage cannot
                # be beaten — no later model can hold more inliers, and no
                # subset spans wider than the full candidate set. Without
                # this, near-identical documents (hundreds of mutually
                # matching units) evaluate every pair and burn minutes of CPU
                # inside one verification call.
                break

    if best_model is None or best_pair is None or len(best_inliers) < MIN_INLIERS:
        return AnchorResult(
            accepted=False,
            reasons=("no_consistent_linear_model",),
            rejections=tuple(rejections),
            diagnostics={"candidates": str(len(candidates))},
        )

    # Full-span regime check BEFORE accepting: a coherent contradictory
    # timing region (e.g. a re-timed recap) vetoes the global claim.
    primary, contradictory_regimes, contradictory_anchors = _scan_regimes(
        best_model, candidates, target_units, reference_units, target_cues, reference_cues
    )
    ordered = sorted(best_inliers, key=lambda item: target_units[item.target_idx].start_ms)
    main_checkpoints = tuple(
        _to_anchor(item, target_units, reference_units, target_cues, reference_cues)
        for item in ordered[1:-1]
    )
    if contradictory_regimes:
        # Centre the primary regime on its majority support (least squares
        # over RANSAC inliers) instead of reporting a tilted pair artefact.
        try:
            centred = fit_least_squares(
                [
                    (
                        float(target_units[item.target_idx].start_ms),
                        float(reference_units[item.reference_idx].start_ms),
                    )
                    for item in ordered
                ]
            )
            primary = replace(
                primary,
                scale=centred.scale,
                offset_ms=centred.offset_ms,
                implied_offset_ms=centred.offset_ms,
            )
        except TransformError:
            pass
        diagnostics = {
            "candidates": str(len(candidates)),
            "inliers": str(len(ordered)),
            "primary_offset_ms": f"{primary.offset_ms:.1f}",
            "contradictory_offsets_ms": ",".join(
                f"{regime.implied_offset_ms:.0f}" for regime in contradictory_regimes
            ),
            "sanitizer_target_eligible": str(sum(1 for score in target_scores if score.eligible)),
            "sanitizer_reference_eligible": str(
                sum(1 for score in reference_scores if score.eligible)
            ),
        }
        return AnchorResult(
            accepted=False,
            checkpoints=main_checkpoints,
            inlier_count=len(ordered),
            reasons=("NONLINEAR_DISCONTINUITY",),
            rejections=tuple(rejections),
            diagnostics=diagnostics,
            regimes=(primary, *contradictory_regimes),
            contradictory=tuple(contradictory_anchors),
        )

    strong_inliers = [item for item in ordered if _is_strong_candidate(item)]
    usable = strong_inliers if len(strong_inliers) >= 2 else list(ordered)
    first_anchor = _to_anchor(usable[0], target_units, reference_units, target_cues, reference_cues)
    last_inlier_anchor = _to_anchor(usable[-1], target_units, reference_units, target_cues, reference_cues)

    # Prefer a verified terminal dialogue pair when the evidence supports it.
    terminal = _terminal_extension(
        best_model,
        target_cues,
        reference_cues,
        target_scores,
        reference_scores,
        current_last_target_ms=last_inlier_anchor.target_ms,
    )
    last_anchor = terminal if terminal is not None else last_inlier_anchor
    # The reported transform is always derived from the two accepted anchors
    # via the strict two-point equations (never a RANSAC-pair artefact).
    try:
        best_model = fit_from_anchors(
            first_anchor.target_ms,
            first_anchor.reference_ms,
            last_anchor.target_ms,
            last_anchor.reference_ms,
        )
    except TransformError:
        if terminal is not None:
            last_anchor = last_inlier_anchor

    separation_ms = last_anchor.target_ms - first_anchor.target_ms
    if separation_ms < min_separation:
        return AnchorResult(
            accepted=False,
            reasons=("insufficient_temporal_separation",),
            rejections=tuple(rejections),
            diagnostics={"candidates": str(len(candidates))},
        )

    first_key = (usable[0].target_idx, usable[0].reference_idx)
    last_key = (usable[-1].target_idx, usable[-1].reference_idx)
    checkpoints = tuple(
        _to_anchor(item, target_units, reference_units, target_cues, reference_cues)
        for item in ordered
        if (item.target_idx, item.reference_idx) not in {first_key, last_key}
    )
    diagnostics = {
        "candidates": str(len(candidates)),
        "inliers": str(len(ordered)),
        "strong_inliers": str(len(strong_inliers)),
        "first_evidence": ";".join(first_anchor.evidence),
        "last_evidence": ";".join(last_anchor.evidence),
        "terminal_extension": "accepted" if terminal is not None else "not_found",
        "sanitizer_target_eligible": str(sum(1 for score in target_scores if score.eligible)),
        "sanitizer_reference_eligible": str(
            sum(1 for score in reference_scores if score.eligible)
        ),
    }
    primary = replace(
        primary,
        scale=best_model.scale,
        offset_ms=best_model.offset_ms,
        implied_offset_ms=best_model.offset_ms,
    )
    return AnchorResult(
        accepted=True,
        first_anchor=first_anchor,
        last_anchor=last_anchor,
        scale=best_model.scale,
        offset_ms=best_model.offset_ms,
        checkpoints=checkpoints,
        inlier_count=len(ordered) + (1 if terminal is not None else 0),
        temporal_separation_ms=separation_ms,
        reasons=("consistent_linear_model_with_independent_checkpoints",),
        rejections=tuple(rejections),
        diagnostics=diagnostics,
        regimes=(primary,),
        contradictory=(),
    )
