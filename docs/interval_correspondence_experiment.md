# Interval correspondence: anchor-bounded, monotonic, split/merge aware

DIAGNOSTIC ONLY. `app/services/sync/interval_correspondence.py` is imported by no
production module (asserted by test) and changes no verifier behaviour, state,
threshold, cache, ranking or provider call. No subtitle bytes are written. No
subtitle text is ever read by the model, logged, or stored.

## Classification: B - USEFUL DIAGNOSTIC ONLY

The correspondence is materially better than anything measured before, and it
fixes the specific weakness that made the previous experiment a B. It is still
not safe to promote: two real failure classes are not detected at all.

## PART 2 - why ASAP was at 2277ms (measured, not inferred)

`tools/diagnose_asap_residual.py`, per episode quarter:

| quarter | tgt | ref | T/R | corr | cov | p50 | p95 | p99 | unm | amb |
|---|---|---|---|---|---|---|---|---|---|---|
| Q1 | 195 | 140 | 1.39 | 125 | 0.641 | 77 | 2147 | 2287 | 47 | 23 |
| Q2 | 195 | 141 | 1.38 | 127 | 0.651 | 167 | 2357 | 2487 | 59 | 9 |
| Q3 | 195 | 140 | 1.39 | 117 | 0.600 | 307 | 2337 | 2477 | 60 | 17 |
| Q4 | 195 | 141 | 1.38 | 129 | 0.662 | 97 | 2067 | 2447 | 42 | 23 |

**p50 is 77-307ms in every quarter while p95 is 2067-2357ms in every quarter.**
So this is not drift, not a bad anchor, not one bad neighbourhood, and not a bad
cut: the timing is good everywhere and the tail is spread evenly.

The reason is volume. ASAP has 780 cues at median 2300ms; the reference has 562 at
2250ms. The per-cue durations are nearly identical and the target simply has
**39% more of them** (active duration 1.374x). 31.2% of correspondences carried
|residual| > 1000ms, which matches the 28% surplus fraction (780-562)/780.

EVOLV is the mirror image: 559 cues against 562 - the same count - but each cue
is longer (2830ms vs 2250ms). Its p50 is -19ms and only 6.7% of correspondences
exceed 1000ms.

**Mechanism: ASAP's subtitle carries ~28% more dialogue than the reference. Its
residual was never misalignment. The one-to-one matcher had no way to express
"this cue is part of a reference cue that became three lines", so it force-paired
the surplus and the surplus became the p95.**

Text could not arbitrate: measured line overlap between the reference and the
targets is 3.7% (ASAP) and 6.2% (EVOLV). The reference is
`tt0773262_8_4_PiR8_v2_..._edition.srt`, a different release and language. Text
is unusable here, exactly as PART 5 anticipated.

## The model

Monotonic dynamic programming over (target index, reference index). Each move
pairs a contiguous run of target cues with a contiguous run of reference cues,
which expresses 1:1, split (1 ref -> k target) and merge (k ref -> 1 target)
without a bespoke rule each. Search is banded around the anchor mapping, so once
anchors exist the model cannot wander the reference.

The primary safety property is built into the objective, not bolted on:
`SKIP_TARGET_COST` (900) is cheaper than `MATCH_COST_PER_MS` (1.0) times a 2.5s
displacement, so declaring a cue unmatchable always beats a weak match.
Grouping costs `GROUP_EXTRA_COST` per extra cue, so it is not free.

Anchors come from the existing production helper via
`collect_anchor_regions`. No second anchor detector.

## PART 7 - real cases

| case | current | anchor-guided | **interval** |
|---|---|---|---|
| EVOLV alass p95 | 3310ms | 1451ms | **296ms** |
| EVOLV alass ref coverage | - | - | 0.801 |
| ASAP alass p95 | 4750ms | 2277ms | **527ms** |
| ASAP alass ref coverage | - | - | 0.847 |
| ASAP original p95 (96s offset) | - | 2374ms | **599ms** |
| EVOLV original (96s offset) | - | 2179ms | abstains |

ASAP alass resolves into 405 groups of which **193 are SPLITS** - the surplus is
now expressed rather than forced. EVOLV alass: 411 groups, 73 splits, 14 merges.

EVOLV original abstains (`ANCHORS_INCONSISTENT`): its regions disagree by 1516ms
and 2004ms across a fixed 1500ms margin. Worth noting that the accepted
large-offset assessment calls the *same* anchors valid (5/5, dispersion 950ms).
The two modules disagree about the same evidence. The abstention is the safe
answer, but the disagreement is a real finding.

## PART 8 - uniform offsets: no breakpoint invention

+0, +2, +12, +30, +60, +96.5, +120, +170s all give **1 mapping piece**, reference
coverage 1.000, target coverage 1.000, residual p50/p95/p99 = 0/0/0. Offset size
alone never creates a segment.

**Piecewise is the model's real failure.** Opening +96s, first-third +96s,
two-region and three-region all abstain (`ANCHORS_INCONSISTENT`); closing +96s
partially aligns and reports `SURPLUS_REFERENCE`. Cause: anchor sampling only
succeeds where the supplied hypothesis happens to be right, so a two-region
target yields evidence for one region and the breakpoint is never observed.
Recovering it would need per-region offset detection, which risks inventing
breakpoints - the exact thing to avoid. The model abstains instead.

Drift: 30 and 100 ms/min align with low reference coverage; 60 ms/min abstains.

## PART 9 - segmentation: the previous model's failure is fixed

| case | tgt/ref | interval result |
|---|---|---|
| identical | 562/562 | cov 1.000, p95 0 |
| slightly finer | 843/562 | cov 1.000, p95 0, 281 splits |
| much finer | 703/562 | cov 1.000, p95 0, 141 splits |
| slightly coarser | 281/562 | 240 merges, surplus reference |
| mixed split/merge | 703/562 | cov 0.950, p95 183 |
| **finer + 96.5s** | 843/562 | **cov 1.000, p95 0** |

The previous model reached p95 2422ms and coverage 0.33 on the last row. It now
aligns it perfectly. Cue-count ratio is nowhere near 1 in the finer cases and
that is explicitly not treated as a problem.

Coarser segmentation is the weakest case: reference coverage 0.584, reported as
`SURPLUS_REFERENCE` rather than claimed as aligned.

## PART 10/12 - damage, and the honest asymmetries

| case | timing | content |
|---|---|---|
| deleted 20/40/60% | `LOW_REFERENCE_COVERAGE` | `TARGET_CONTENT_LOSS` |
| truncated 60% | `SURPLUS_REFERENCE` | `TARGET_CONTENT_LOSS` |
| sparse 25% | `SURPLUS_REFERENCE` | `TARGET_CONTENT_LOSS` |
| corrupt 30% +/-8s | `LOW_REFERENCE_COVERAGE` | comparable |
| corrupt 10/20% +/-3s | **ALIGNED** | comparable |
| duplicated 20/40% | **ALIGNED** | `TARGET_CONTENT_EXCESS` |
| wrong release +96s, wrong hypothesis | `SURPLUS_REFERENCE` | comparable |
| wrong release +96s, right hypothesis | `ALIGNED` | comparable |

Deletion and truncation are caught by timing. **Duplication is not**: an adjacent
duplicate of a cue is a legal SPLIT, so timing sees a clean alignment with zero
residual. The only timing-side signal is total speech time, because splitting a
cue preserves it and duplicating one adds to it:

| case | cue ratio | active ratio | content |
|---|---|---|---|
| identical | 1.00 | 1.00 | COMPARABLE |
| finer segmentation | 1.50 | 1.00 | COMPARABLE |
| duplicated 40% | 1.50 | 1.51 | EXCESS |
| ASAP alass (real) | 1.39 | 1.37 | EXCESS |

Coarser segmentation raises active time while *lowering* cue count, so both
ratios are read together; a mixed signal is reported comparable rather than
guessed at.

**Undetectable, recorded rather than hidden:**
1. **A wrong episode with identical timing.** No timing method can see this. It
   is the job of identity evidence.
2. **Reordered text.** Text is never used. Also identity evidence's problem.
3. **10-20% cue jitter** stays inside the band.
4. **A wrong cut with a 4s middle shift** aligned at coverage 0.881 / p95 158.
   This is a genuine false acceptance and is not explained away.

Duplication and a legitimately different release are indistinguishable here -
both show content excess. That is why ASAP alass is reported as EXCESS even
though it is a real, correctly-aligned release.

## PART 14 - mutations

**77/77 caught**, exit 0, harness unchanged in its handling of crashes, imports,
collection errors and pre-existing baseline failures.

Fourteen mutations cover: production wiring, search window, anchor offset
ignored, group span, correspondence ceiling, reference consumption (monotonicity),
coverage accounting, the aligned-coverage bar, surplus-reference reporting,
content-volume observation, free skipping, and free grouping.

Two guards - `MAX_GROUP` and `MIN_GROUP_OVERLAP_RATIO` - are **deliberately not
given mutations**, and the reason is recorded in the harness. The objective
already forbids the matches they forbid: skipping is cheaper than a 2.5s
displacement, and grouping is expensive, so removing either constant changed no
result. They are kept as defence in depth and would bite if the weights were
retuned. A mutation that cannot be caught is not evidence, and reporting
UNPROTECTED for a correctly redundant guard would bury the real findings.

## Not concluded

No threshold moved. Production unchanged: EVOLV and ASAP remain
`UNVERIFIED -> ORIGINAL`. Manual `MANUAL_PLAYBACK_WITNESS` evidence stays manual.
`ENABLE_AUTOSYNC_TEST_OVERRIDE=false`. No verdict, artifact, alias, negative cache
or provider-cache write; the EVOLV and ASAP provider files remain byte-identical
(`e82586e4...`, `30246e82...`).