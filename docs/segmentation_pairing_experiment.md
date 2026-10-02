# Segmentation-aware correspondence: diagnostic experiment

**DIAGNOSTIC ONLY.** Nothing described here is wired to any verification state,
threshold, or production decision. `app/services/sync/segmentation_pairing.py` is
imported by no production module; a test asserts that, and a mutation that wires
it in is caught.

## Hypothesis under test

Production pairs cues greedily: each cue takes its nearest unused partner. If a
subtitle is re-segmented -- one reference line rendered as two output cues, or
two merged into one -- a strictly one-to-one match cannot represent that
correspondence, so the closest partner is chosen and the residual inherits the
*segmentation* error instead of the true offset. The hypothesis is that a
segmentation-aware correspondence would remove that artificial residual.

The candidate (`aggregated_pair`) builds a monotonic alignment over cue starts
and lets one anchor span up to `max_group` consecutive cues on the other side,
taking the residual of the group as a whole. Greedy nearest-unused is used only
to choose among admissible candidates, and monotonicity is what stops it
double-counting.

## Result: no safe replacement

Measured on the two real validated artifacts, then on a synthetic GOOD/BAD matrix.

| variant | EVOLV p95 | ASAP p95 | damaged input scored clean? |
|---|---|---|---|
| greedy (production-like) | 4050 ms | 1810 ms | truncated 60% → **0.0** |
| agg `max_group=2 span=3000` | 4180 ms | 4680 ms | no |
| agg `max_group=2 span=6000` | 3890 ms | 3850 ms | no |
| agg `max_group=3 span=6000` | 3130 ms | 3680 ms | deleted → **0.0**, truncated → **0.0** |
| agg `max_group=3 span=12000` | 4380 ms | 4060 ms | truncated → **0.0** |

Production limit is 2000 ms. **No variant brings either real case under it.**

Two distinct failure modes, and the trade-off between them is structural:

1. **Conservative variants do not help.** `max_group=2` keeps deleted content
   visible, but also leaves EVOLV at 3890 ms and ASAP at 3850 ms. Aggregation
   has nothing to repair.
2. **Aggressive variants help by hiding damage.** `max_group=3` cuts EVOLV to
   3130 ms -- and scores a subtitle with cues deleted as p95 0.0. Absorbing cues
   into groups is the *same operation* that hides deletion. There is no setting
   that both accepts the good cases and rejects the damaged ones.

## Why the hypothesis was wrong about these cases

On a *synthetic* re-segmented fixture the conservative variant does look good: a
uniform +3 s offset falls from 3000 ms to 1280 ms, under the limit, while
truncation stays visible at 1680 ms. Both behaviours are asserted in
`tests/test_segmentation_pairing_diagnostic.py`.

That result does not transfer, and the reason matters. The real residual is not
re-segmentation. It is an opening section displaced by roughly 96 s that is *not*
a uniform shift, which the movement seed declines (correlation below its floor
for a piecewise track) and which no change to cue correspondence can remove. A test
applies a 96 s offset to the first third of a re-segmented fixture and asserts
that both greedy and aggregated pairing still score it above the limit, so the
disqualification is executable rather than anecdotal.

Earlier work attributed 13.5% (EVOLV) and 58.7% (ASAP) of residuals to greedy
pairing. This experiment does not contradict that arithmetic; it shows that
repairing that share does not bring either case under the limit, so it is not the
binding constraint.

## Text-assisted pairing: abandoned

Tested and rejected on its own merits.

| dialogue | textually ambiguous positions |
|---|---|
| distinct lines | 208 / 562 |
| repeated identical lines | **517 / 562** |
| yes/no answers | 132 / 562 |
| short dialogue | 0 / 562 |

With repeated or yes/no dialogue almost every position is textually tied, so a
text term breaks ties arbitrarily and adds no discrimination exactly where a
verifier needs it most. Across languages the overlap is identically zero, so the
term contributes nothing and **silently degrades to the timing-only case** rather
than failing loudly -- the worst failure mode for a safety signal. No text
pairing was implemented, and no subtitle text is logged anywhere in the
experiment.

## Completeness is a separate question (Part 8)

Kept strictly apart from timing, and the separation is now justified by
measurement rather than caution.

A residual metric **cannot detect truncation at all**:

| cues kept | greedy p95 | cue_count_ratio | coverage |
|---|---|---|---|
| 100% | 0.0 | 1.000 | 0.946 |
| 80% | 0.0 | 0.799 | 0.742 |
| 60% | 0.0 | 0.600 | 0.550 |
| 40% | 0.0 | 0.399 | 0.382 |
| 20% | **0.0** | 0.199 | 0.183 |

A cue that survives truncation is still perfectly aligned, so every matched pair
has zero offset. Duplication is likewise invisible: a duplicated line is perfectly
aligned with itself.

So a clean residual is **not** evidence of a complete subtitle, and must never be
read as one. `completeness()` inventories the signals that do bear on content --
cue-count ratio, runtime coverage, reference coverage, first/last cue delta,
active-duration ratio -- and applies **no threshold** to any of them. Inventing one
without evidence is the same class of mistake as raising the p95 limit.

This is the strongest argument for leaving acceptance as it is: the current
verifier is conservative *because* residual p95 alone is not a sufficient
criterion, and this experiment demonstrated that directly.

## Caveat on the numbers

`greedy_pair` here is a reimplementation for comparison, not the production
pairer. It reproduces EVOLV at 4050 ms against production's 3310 ms, and ASAP at
1810 ms against production's 4750 ms. The absolute values are therefore **not**
production's, and in particular the greedy-ASAP figure is well under the limit
while production rejects the case. Only the *relative* comparison between greedy
and aggregated within this reimplementation carries meaning. No claim here rests
on an absolute threshold crossing.

## Conclusion

No safe replacement exists. Production acceptance is unchanged:
`MAX_P95_MS_FOR_STABLE = 2000.0`, all other thresholds untouched, and both real
cases remain `UNVERIFIED → ORIGINAL`.

The next engineering step is not a better pairing rule. It is the piecewise
offset itself -- a correspondence model that tolerates a *discontinuity* in the
mapping, rather than one that repairs cue-boundary differences.
