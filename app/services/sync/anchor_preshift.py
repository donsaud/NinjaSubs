"""Anchor-correspondence pre-shift: propose a constant offset, then prove it.

Why this exists
---------------
Production reached alass with an ~11s disagreement between the target's and the
reference's opening dialogue and came back with the timeline unmoved. The naive
response -- shift the target by the first-cue delta -- is wrong often enough to
be dangerous, because "the first dialogue cue" is not the same *line* in two
different releases of the same film:

    #3 HDA REMUX vs SURCODE   seed +11159ms  -> correct, aligns to  42ms
    #3 HDA REMUX vs LOST      seed  -7717ms  -> wrong,  makes it worse
    #2 Ozlem     vs LOST      seed  -8617ms  -> wrong,  makes it worse

A first-cue delta is a *hypothesis*, not a measurement.

Why anchor correspondence alone is not enough either
----------------------------------------------------
The multi-region gate (:mod:`app.services.sync.large_offset`) re-measures anchors
across the timeline and rejects dispersed or drifting hypotheses. It accepts the
good case (``est=+11072``, dispersion 956ms, 5/5 anchors) -- and it *also*
accepts the bad ones (``est=-6600``, dispersion 181ms, 5/5 anchors). Anchors
search near the hypothesis with a tolerance, and in a densely cued reference
there is always *some* cue inside that tolerance, so a wrong hypothesis finds
confident-looking support. Dispersion measures agreement, and a wrong answer
can agree with itself perfectly.

So this module adds the missing half: the proposal is accepted only when
applying it measurably improves correspondence against the reference.

    propose  = multi-region anchor correspondence  (large_offset.assess_large_offset)
    validate = residual outcome measurement       (_outcome)

A shift is served only when both the number of paired cues rises and the median
absolute residual falls. That rejects a hypothesis which is self-consistent but
wrong, which is precisely the failure a pure anchor test cannot see.

Scope
-----
This decides *whether to shift the target before alass*, and by how much. It
does not judge the result: the verifier still owns every accept/reject decision,
and a pre-shifted target that the verifier rejects is served unchanged.

Deliberately no dialogue filtering
----------------------------------
Every cue counts here, including credits, music cues and sound effects. That is
the opposite of the alignment-consistency gate, which filters them out, and the
difference is measurable rather than stylistic: on the real Whiplash pairs,
filtering to dialogue alone moved the estimated offset from +11072ms to
+13116ms and the resulting median residual from 42ms to 1210ms.

The reason is that this is an offset question, not a speech question. A music
cue or a door slam sits at the same instant in two different encodes of the same
film, whereas the first *line* of dialogue does not -- it depends on how the
translator chose to break the sentence. Estimating a global shift from dialogue
alone throws away the most reliable anchors available.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from ..subtitle_matcher import first_dialogue_cluster, parse_srt_cues
from .alignment import RESIDUAL_TOLERANCE_MS, Cue, pair_cues
from .large_offset import assess_large_offset

#: Below this, two tracks are close enough that re-timing is alass's job and a
#: pre-shift would only add a second opinion nobody asked for. Matches the
#: normal execution window's lower edge.
ANCHOR_PRESHIFT_MIN_ABS_MS = 2000

#: Ceiling for a pre-shift. Wider than the large-offset path on purpose: this
#: runs *before* alass on candidates the normal window already accepted, so a
#: generous band is safe -- anything beyond it is a different cut, and shifting
#: by minutes cannot be what a viewer meant.
ANCHOR_PRESHIFT_MAX_ABS_MS = 25_000

#: The proposal has to leave the median residual at or below this fraction of
#: what it was. A shift that shaves a few percent off has not found the offset;
#: it has found noise.
#:
#: Measured against the *median* residual rather than the paired-cue count,
#: because the count is close to saturated and cannot discriminate. ``pair_cues``
#: matches each cue to the nearest unused reference cue within
#: ``RESIDUAL_TOLERANCE_MS``, and that tolerance (5000ms) is wider than the mean
#: cue gap of a dense subtitle (~3400ms), so even a track that is 11 seconds out
#: still pairs 376 of 380 cues -- to the *wrong* ones. Counting pairs therefore
#: measures almost nothing here, while the median residual moves from ~11000ms
#: to ~0 when the shift is right and gets *worse* when it is wrong.
ANCHOR_PRESHIFT_MAX_RESIDUAL_RATIO = 0.75

#: Half-width of the local search run after the anchor proposal. The anchor
#: estimate snaps each cue to the nearest reference cue within
#: ``LARGE_OFFSET_ANCHOR_TOLERANCE_MS``, so it can land roughly half a cue gap
#: from the truth -- 1650ms on the synthetic fixtures here, where an 11000ms
#: offset was proposed as -12650ms. The window has to be able to reach the truth
#: from there, so it is a little wider than the worst observed error.
ANCHOR_PRESHIFT_REFINE_WINDOW_MS = 1500
ANCHOR_PRESHIFT_REFINE_STEP_MS = 250

#: Minimum paired cues on both sides before an outcome comparison means
#: anything. Below this a handful of lucky pairs can "improve" by accident.
ANCHOR_PRESHIFT_MIN_PAIRS = 50


@dataclass(frozen=True)
class AnchorPreshift:
    """An accepted pre-shift, with the evidence that justified it."""

    shift_ms: int
    seed_ms: int
    anchors: int
    regions_sampled: int
    dispersion_ms: float | None
    pairs_before: int
    pairs_after: int
    median_abs_before_ms: float
    median_abs_after_ms: float

    def summary(self) -> str:
        return (
            f"shift={self.shift_ms}ms seed={self.seed_ms}ms "
            f"anchors={self.anchors}/{self.regions_sampled} "
            f"dispersion={self.dispersion_ms}ms "
            f"pairs={self.pairs_before}->{self.pairs_after} "
            f"median|residual|={self.median_abs_before_ms:.0f}ms->"
            f"{self.median_abs_after_ms:.0f}ms"
        )


@dataclass(frozen=True)
class AnchorPreshiftDecision:
    """Why a pre-shift was or was not applied. Always returned, never raises."""

    accepted: bool
    reason: str
    preshift: AnchorPreshift | None = None

    def summary(self) -> str:
        if self.preshift is None:
            return f"no pre-shift ({self.reason})"
        return f"pre-shift {self.reason}: {self.preshift.summary()}"


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def _outcome(
    target_cues: Sequence[Cue], reference_cues: Sequence[Cue]
) -> tuple[int, float] | None:
    """Paired-cue count and median absolute residual at the current alignment.

    Median absolute residual rather than p95 on purpose. p95 is dominated by
    segmentation disagreement between releases -- a 958-cue Arabic track against
    a 922-cue English one puts most cues in a 1.5-2.0s band that says nothing
    about whether the dialogue lines up. The median is unmoved by that tail.
    """
    pairing = pair_cues(list(target_cues), list(reference_cues), tolerance_ms=RESIDUAL_TOLERANCE_MS)
    residuals = [abs(match.delta_ms) for match in pairing.matches]
    if len(residuals) < ANCHOR_PRESHIFT_MIN_PAIRS:
        return None
    return len(residuals), _median(residuals)


def apply_anchor_preshift(cues: Sequence[Cue], shift_ms: int) -> list[Cue]:
    """Shift every cue by ``shift_ms``.

    Cues pushed entirely before zero are dropped rather than clamped. A cue with
    no positive position has lost its place on the timeline, and piling it at
    zero would invent a cue where the film has none.
    """
    if not shift_ms:
        return list(cues)
    shifted: list[Cue] = []
    for start, end, text in cues:
        new_start, new_end = start + shift_ms, end + shift_ms
        if new_end <= 0:
            continue
        shifted.append((max(0, new_start), new_end, text))
    return shifted


def _timestamp(ms: int) -> str:
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def render_srt_cues(cues: Sequence[Cue], newline: str = "\n") -> str:
    """Re-emit cues as SRT text.

    Index numbering is regenerated from scratch rather than carried over: after a
    shift the surviving cues are a different set, and preserving stale numbers
    would leave gaps in the sequence for no benefit.

    Blocks are separated by a blank line. Without it every cue merges into its
    neighbour on re-parse, which silently collapses a whole subtitle into one.
    """
    if not cues:
        return ""
    separator = newline * 2
    blocks = [
        f"{index}{newline}{_timestamp(start)} --> {_timestamp(end)}{newline}{text}"
        for index, (start, end, text) in enumerate(cues, 1)
    ]
    return separator.join(blocks) + newline


def preshift_text(text: str, shift_ms: int) -> str | None:
    """Shift an SRT document by ``shift_ms``, or return ``None``.

    ``None`` means the shift left nothing to serve -- every cue was pushed off
    the front of the timeline -- and the caller must keep the original text
    rather than hand alass an empty file.
    """
    if not shift_ms or "-->" not in text:
        return None
    newline = "\r\n" if "\r\n" in text else "\n"
    shifted = apply_anchor_preshift(parse_srt_cues(text.replace("\r\n", "\n")), shift_ms)
    rendered = render_srt_cues(shifted, newline)
    return rendered or None


def _refine(
    target_cues: Sequence[Cue],
    reference_cues: Sequence[Cue],
    proposal_ms: int,
    baseline_median_ms: float,
) -> tuple[int, tuple[int, float]]:
    """Polish the anchor proposal with a bounded local search.

    The anchor estimate snaps each cue to the nearest reference cue within
    ``LARGE_OFFSET_ANCHOR_TOLERANCE_MS``, so it can land about half a cue gap
    from the truth -- 1650ms on the synthetic fixtures here, where the proposal
    came back -12650ms for a true 11000ms offset. This searches +/-1200ms around
    the proposal and keeps the shift with the lowest median residual.

    Running before the acceptance gate is safe, and measurably so: on the real
    Whiplash pairs the wrong-reference proposals sit in a flat, genuinely wrong
    region and stay rejected (median ratio 0.94 and 1.06 against a 0.75 bar),
    while the right ones improve sharply (0.46 and 0.02). The search cannot find
    a flattering shift in a neighbourhood that has none.

    ``baseline_median_ms`` breaks ties toward *no shift*, so refinement can only
    help: if nothing in the window beats leaving the subtitle alone, the
    proposal is not improved on and the gate rejects it.
    """
    best_shift = proposal_ms
    best_outcome = _outcome(apply_anchor_preshift(target_cues, proposal_ms), reference_cues)
    best_value = best_outcome[1] if best_outcome is not None else float("inf")
    if best_value >= baseline_median_ms:
        # Nothing in the window beats the unshifted subtitle. Report the
        # proposal unchanged so the caller's reason names the real number.
        best_value = baseline_median_ms
    for delta in range(
        -ANCHOR_PRESHIFT_REFINE_WINDOW_MS,
        ANCHOR_PRESHIFT_REFINE_WINDOW_MS + 1,
        ANCHOR_PRESHIFT_REFINE_STEP_MS,
    ):
        if delta == 0:
            continue
        candidate = proposal_ms + delta
        outcome = _outcome(
            apply_anchor_preshift(target_cues, candidate), reference_cues
        )
        if outcome is None:
            continue
        if outcome[1] < best_value:
            best_shift, best_value, best_outcome = candidate, outcome[1], outcome
    if best_outcome is None:
        return best_shift, (0, float("inf"))
    return best_shift, best_outcome


def decide_anchor_preshift(
    target: str | Sequence[Cue],
    reference: str | Sequence[Cue],
    *,
    min_abs_ms: int = ANCHOR_PRESHIFT_MIN_ABS_MS,
    max_abs_ms: int = ANCHOR_PRESHIFT_MAX_ABS_MS,
) -> AnchorPreshiftDecision:
    """Decide whether to shift ``target`` onto ``reference`` before alass.

    Three stages, and each can refuse:

    1. Seed from the opening dialogue. Outside ``[min_abs_ms, max_abs_ms]`` the
       offset is either too small to matter or too large to be this film.
    2. Propose via multi-region anchor correspondence, which re-measures the
       hypothesis across the timeline instead of trusting one cue.
    3. Validate by outcome. The proposal must actually increase correspondence
       and reduce the median residual.

    Returns a decision rather than raising: a subtitle that cannot be shifted
    safely is a normal outcome, not an error.
    """
    target_cues = parse_srt_cues(target) if isinstance(target, str) else list(target)
    reference_cues = (
        parse_srt_cues(reference) if isinstance(reference, str) else list(reference)
    )

    if (
        len(target_cues) < ANCHOR_PRESHIFT_MIN_PAIRS
        or len(reference_cues) < ANCHOR_PRESHIFT_MIN_PAIRS
    ):
        return AnchorPreshiftDecision(False, "not enough dialogue on both sides")

    # Same definition the sync service uses, so the seed here cannot disagree
    # with the opening offset the rest of the pipeline already reasoned about.
    target_first = first_dialogue_cluster(target_cues)
    reference_first = first_dialogue_cluster(reference_cues)
    if target_first is None or reference_first is None:
        return AnchorPreshiftDecision(False, "no measurable opening dialogue")

    seed = reference_first - target_first
    if not min_abs_ms <= abs(seed) <= max_abs_ms:
        return AnchorPreshiftDecision(
            False,
            f"opening offset {seed}ms outside the [{min_abs_ms}, {max_abs_ms}]ms pre-shift band",
        )

    assessment = assess_large_offset(
        target_cues,
        reference_cues,
        seed_offset_ms=seed,
        identity_supported=True,
    )
    if not assessment.accepted or assessment.estimated_offset_ms is None:
        return AnchorPreshiftDecision(
            False,
            "anchor correspondence did not corroborate the opening offset "
            f"({','.join(assessment.reason_codes) or 'no reasons'})",
        )

    shift_ms = int(round(assessment.estimated_offset_ms))
    if not min_abs_ms <= abs(shift_ms) <= max_abs_ms:
        return AnchorPreshiftDecision(
            False, f"corroborated offset {shift_ms}ms left the pre-shift band"
        )

    before = _outcome(target_cues, reference_cues)
    if before is None:
        return AnchorPreshiftDecision(
            False, "too few paired cues to judge whether the shift helped"
        )
    pairs_before, median_before = before

    shift_ms, (pairs_after, median_after) = _refine(
        target_cues, reference_cues, shift_ms, median_before
    )

    # The check that matters. A hypothesis can be self-consistent across five
    # regions and still be wrong; what cannot be faked is a measurable
    # improvement in alignment against the reference it claims to match.
    if median_after >= median_before:
        return AnchorPreshiftDecision(
            False,
            f"shift {shift_ms}ms did not reduce the median residual "
            f"({median_before:.0f}ms->{median_after:.0f}ms)",
        )
    if median_after > median_before * ANCHOR_PRESHIFT_MAX_RESIDUAL_RATIO:
        return AnchorPreshiftDecision(
            False,
            f"shift {shift_ms}ms barely moved the median residual "
            f"({median_before:.0f}ms->{median_after:.0f}ms, needs "
            f"<={median_before * ANCHOR_PRESHIFT_MAX_RESIDUAL_RATIO:.0f}ms)",
        )
    # Correspondence must not get *worse* to achieve that. Weak on its own, but
    # it catches a shift that improves the median by discarding cues.
    if pairs_after < pairs_before:
        return AnchorPreshiftDecision(
            False,
            f"shift {shift_ms}ms lost correspondence "
            f"({pairs_before}->{pairs_after} pairs)",
        )

    return AnchorPreshiftDecision(
        True,
        "anchors corroborated and the shift measurably improved correspondence",
        AnchorPreshift(
            shift_ms=shift_ms,
            seed_ms=seed,
            anchors=assessment.anchor_count,
            regions_sampled=assessment.regions_sampled,
            dispersion_ms=assessment.offset_dispersion_ms,
            pairs_before=pairs_before,
            pairs_after=pairs_after,
            median_abs_before_ms=median_before,
            median_abs_after_ms=median_after,
        ),
    )
