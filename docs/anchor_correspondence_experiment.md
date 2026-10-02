# Anchor-guided correspondence (shadow experiment)

DIAGNOSTIC ONLY. `app/services/sync/anchor_correspondence.py` is imported by no
production module (asserted by test) and changes no verifier behaviour, state,
threshold, cache, ranking or provider call. No subtitle bytes are written.

Question: **can the existing accepted large-offset anchors establish safe, monotonic
cue-correspondence neighbourhoods before residual is measured?** This replaces the
piecewise fitter, which the previous experiment rejected.

## Verdict: B - useful diagnostic only

The correspondence it finds is genuinely better than nearest-start matching, and
that is worth keeping. It is **not** yet a safe acceptance design.

## What it does

1. Reuses the accepted large-offset evidence and the production
   `_nearest_reference_offset()` helper - no second anchor detector.
2. Buckets anchor samples into regions, merges regions whose offsets agree, and
   emits one mapping segment per genuine disagreement.
3. Predicts each target cue's reference position from the mapping, accepts only
   candidates inside a local neighbourhood, and **rejects a cue as ambiguous**
   when candidates disagree by more than the ambiguity margin.
4. Reports a timing-only label plus, separately, completeness characterisations.

No polynomials, no title-specific rules, no forced one-to-one matching, no text.

## The central claim holds

On the pair production actually evaluates - Alass output against reference - the
residual improves materially over what production measures today:

| case | production p95 | anchor-guided p95 | median |
|---|---|---|---|
| EVOLV alass vs ref | 3310 ms | **1451 ms** | 58.8 ms |
| ASAP alass vs ref | 4750 ms | **2277 ms** | 213 ms |

At a uniform ~96.5 s offset it recovers 447/562 correspondences with p95 **0** and
one mapping piece. The offset size alone never invents a segment: every uniform
offset from 0 to +120 s reduces to exactly one piece.

At the large offset the *previous* experiment lost a third of the cues and
produced a lag spread of ~108 s. This recovers a coherent mapping because it
looks near a predicted point rather than globally.

## Why it is only B

- **ASAP still exceeds a 2000 ms bound** (2277 ms). EVOLV clears it; ASAP does
  not. So it does not resolve the case it was most needed for.
- **Segmentation-sensitive.** Finer segmentation degrades it badly
  (p95 2422 ms, coverage 0.33). Coarser segmentation is stable.
- **Content damage is invisible to it, by construction.** Deleting, duplicating
  or reordering cues does not misalign the survivors, so the timing label is
  legitimately GOOD. This is required - completeness stays separate - but it
  means the label can never be an acceptance signal on its own.
- The label's residual condition is close to the existing p95 metric. The gain is
  a **better correspondence**, not a new kind of signal.

## The decision rule must look at residual

A first version labelled on correspondence count plus monotonicity alone. It
called **all 15** damaged corpus fixtures GOOD, including 30 % timing corruption.
Having correspondences is not having good ones. Adding the residual condition
separates severe timing damage (`corrupt 30%` -> BAD, fragmentation -> BAD) while
leaving genuine timing relationships GOOD.

This is pinned by test and by mutation.

## Three bugs found while building it

All three were real, and two of them the earlier harness could not see.

1. **Monotonicity sign inverted.** Offset is `target - reference`, so a mapped
   time is `t - offset` and the check must *subtract*. Using `+` falsely abstained
   on the real EVOLV case with a legitimate ~96 s correction.
2. **Deviation filter erased genuine breakpoints.** A region deviating from the
   median was folded onto "the closest region". With two regions 96 s apart both
   were folded onto whichever sorted first, collapsing a real disagreement into a
   single segment - the exact opposite of the filter's intent. A region may now
   only be folded into an actual consensus cluster, never a lone reading.
3. **An unreachable guard.** A single hypothesis confines cross-region offset
   spread to the search neighbourhood, so a *folding* time map cannot be produced
   through the public API. The guard is correct but was untestable inline; the
   decision step is now a separate function so every branch is reachable.

## The mutation harness was reporting false catches

`tools/run_safety_mutations.py` ran `pytest -x` on a suite with three
pre-existing `guessit` failures. Because `-x` stops at the first failure and
those tests sort early, **any guard in a later module was credited with that
pre-existing failure** while staying green.

The harness now measures the unmutated baseline once, deselects it, and credits a
mutation only for a failure it introduced. Immediately, three guards were exposed
as genuinely unprotected:

- `movement-seed-from-large-offset-gate` and
  `movement-seed-bypasses-movement-validation` were **malformed mutations** - one
  inserted only a comment, the other appended `and True`. Semantic no-ops can
  never be caught. Both now remove real behaviour.
- `diagnostic-candidate-declares-an-acceptance-limit` had **no test at all**.

Result: **65/65 caught**, exit 0.

## Completeness stays separate

Reported, never bounded. No completeness threshold exists in the module
(asserted). For truncation and sparsity the signals show clearly
(`cue_count_ratio` 0.60 / 0.25). Uniform pattern deletion - dropping every 2.5th
cue - is invisible to both timing and count, and is recorded as a known
limitation rather than papered over.

## Not concluded

No threshold was moved. Production is unchanged: EVOLV and ASAP remain
`UNVERIFIED -> ORIGINAL` at p95 3310 ms / 4750 ms. The manual
`MANUAL_PLAYBACK_WITNESS` evidence remains manual and never becomes an automatic
`VERIFIED` label.