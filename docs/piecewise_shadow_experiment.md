# Piecewise offset model: shadow experiment

**Diagnostic only.** `app/services/sync/piecewise_shadow.py` and
`piecewise_shadow_eval.py` are imported by no production module; a test asserts
that, and a mutation that wires the model in is caught. Nothing here changes an
artifact, a verification state, a cache entry, a ranking, or a provider result.

## Question

The two real Dexter S08E04 cases are not explained by a single global offset:
their opening sits roughly 96 s from the rest, the movement seed declines the
track (poor correlation for a piecewise mapping), and the residual tail is heavy.
A piecewise constant offset field is the simplest family that can express "early
section at one offset, later section at another", so: **does such a model
describe the demonstrably good Alass output in a stable, interpretable way, and
could it be an acceptance signal?**

## Model

A deterministic piecewise-constant offset field over cue time, fitted by
exhaustive search over a bounded breakpoint grid. One segment is the special case,
so a uniform offset needs no separate path. Bounded `MAX_SEGMENTS = 2`, a
complexity penalty of 1000 ms per breakpoint, a minimum segment share of 12%, and
a minimum segment length of 20 s. Offsets are medians, so the fit is robust to
outliers. Polynomial models are deliberately excluded: an unbounded-degree fit
can fit anything, including damaged input.

Two constants were written and then **removed after measurement**:

* a "minimum difference between adjacent segments" rule (2000 ms). A split only
  repays the 1000 ms penalty when the halves differ by more than twice the
  penalty, which is already above the rule, so it could never fire. A guard that
  cannot reject anything implies a rule the model does not use.
* an anticipated third segment: no input in the corpus ever fitted one, because a
  third piece never repaid its penalty.

## Result: REJECTED (outcome C)

The model fails its own safety gate, in both directions at once.

| case | corpus label | shadow label | p95 |
|---|---|---|---|
| already aligned | GOOD | ACCEPT | 0 |
| uniform +2 s | GOOD | REJECT | 10840 |
| uniform +12 s | GOOD | REJECT | 12140 |
| uniform +30 s | GOOD | REJECT | 12220 |
| uniform +96.5 s | GOOD | REJECT | 11340 |
| uniform +120 s | GOOD | REJECT | 11820 |
| opening-only +96 s | GOOD | REJECT | 2600 |
| closing-only +96 s | GOOD | REJECT | 2100 |
| piecewise 2-segment | GOOD | REJECT | 4060 |
| moderate drift | GOOD | REJECT | 10858 |
| wrong cut (+0.7 s) | BAD | REJECT | 10840 |
| wrong release | BAD | **ACCEPT** | 0 |
| truncated 60% | BAD | **ACCEPT** | 0 |
| sparse 25% | BAD | **ACCEPT** | 0 |
| deleted 20% / 40% / 60% | BAD | **ACCEPT** | 0 |
| duplicated 20% / 40% | BAD | **ACCEPT** | 0 |
| corrupt 10% ±3 s | BAD | **ACCEPT** | 1749 |
| corrupt 20% ±3 s | BAD | REJECT | 2116 |
| corrupt 30% ±8 s | BAD | REJECT | 4261 |
| reordered | BAD | **ACCEPT** | 0 |
| severe fragmentation | BAD | REJECT | 10422 |

* **false_acceptance = 9** of 14 BAD cases, including truncation and deletion at
  every depth.
* **false_rejection = 9** of 10 GOOD cases.

For a safety signal this is the worst possible shape: it fails closed on good
input and fails open on damaged input.

## Why — the mechanism, not a tuning problem

**A large offset cannot be represented by the correspondence the model builds.**
Observations are nearest-start matches. For a uniform +96.5 s shift the shifted
targets collide over the same reference cues, so correspondences are lost and the
survivors jump to distant partners. On the real reference: 559 target cues give
521 observations for EVOLV and 559 for ASAP; for a synthetic uniform +96.5 s shift
379 of 562 survive with the lag smeared across a ~108 s range. The offset field is
therefore not a measurement of the correction — it is a survivor-biased sample of
the well-aligned part.

**Surviving cues stay aligned when content is deleted**, so damaged input scores
p95 = 0. No residual-based model can see it.

## Real cases: no breakpoint is found

| | EVOLV | ASAP |
|---|---|---|
| target cues | 559 | 780 |
| observations | 521 | 559 |
| lag median | +130 ms | +460 ms |
| fitted pieces | **1** | **1** |
| breakpoints | none | none |
| shadow p95 | 2250 ms | 7360 ms |
| shadow label | REJECT | REJECT |
| production p95 | 3310 ms | 4750 ms |
| production | UNVERIFIED → ORIGINAL | UNVERIFIED → ORIGINAL |

**The model finds no discontinuity on either real case.** It fits a single global
offset at the body's timing and never represents the ~96 s displaced opening that
motivates the whole idea. So even if it were safe, it would not describe the cases
it was built for.

## Completeness, kept separate

Timing and completeness are two independent verdicts and are never combined; each
ACCEPT states "TIMING ONLY". Measured capability:

| signal | truncation | sparsity | uniform-pattern deletion | duplication |
|---|---|---|---|---|
| cue-count ratio | detects | detects | **blind** (~0.99) | blind (~1.01) |
| temporal density | detects | blind at this resolution | blind | blind |
| runtime coverage | weakly | no | no | no |

Cue-count ratio is the load-bearing signal. No threshold is applied to any
completeness signal, here or in the model: inventing one to make a known case pass
is the failure mode under test, and inventing one is the same class of mistake as
raising the p95 limit.

## Conclusion and decision

1. **Does the model describe EVOLV and ASAP accurately?** No — it fits one
   segment and finds no breakpoint on either. *(1)*
2. **Robust to cue segmentation?** Not applicable in the sense that matters: it
   cannot represent the offset it would need to be robust about. *(1)*
3. **Rejects obvious timing corruption?** Split — it rejects large jitter, but
   accepts truncation, deletion, duplication and reordering. *(0)*
4. **Exposes deletion/truncation separately?** Partly. Cue-count ratio catches
   truncation and sparsity; uniform-pattern deletion and duplication are
   invisible to every signal inventoried. *(0.5)*
5. **false_verified / false_acceptance?** 9 of 14 BAD cases. *(0)*
6. **false_rejected?** 9 of 10 GOOD cases. *(0)*
7. **Outperforms the existing diagnostic without weakening safety?** No. The
   existing verifier is conservative and multi-signal; this is neither. *(0)*
8. **Evidence for a production verifier change?** No. *(0)*

**Outcome: C — UNSAFE / REJECTED.** Production acceptance is unchanged:
`MAX_P95_MS_FOR_STABLE = 2000.0` and every other threshold untouched, both real
cases still `UNVERIFIED → ORIGINAL`, and the DEV override is back to `false`.

## What "piecewise" needed, and did not get

Modelling the discontinuity is not the hard part — the fitter is sound, and given
*true* correspondences it recovers a 96.5 s→0 step exactly and places the
breakpoint correctly. The hard part is obtaining true correspondences when the
offset is large. Nearest-start matching cannot; that is the same reason the
movement seed declines these tracks. A future attempt needs a correspondence
method that tolerates a large *and* discontinuous offset before any acceptance
question is worth asking again.
