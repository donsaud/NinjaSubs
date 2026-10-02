# Residual pairing: candidate evaluation (diagnostic only)

Investigation result. **No candidate is safe to adopt, and the production
verifier is unchanged.** This documents why, so the work is not repeated.

## The problem, confirmed

The production pairing (`pair_cues` in `app/services/sync/alignment.py`) is greedy
nearest-unused with no monotonicity constraint. When the output is segmented more
finely than the reference, the greedy assignment hands later cues a later
reference cue and records a boundary disagreement as timing error.

Measured directly: **58.7%** of ASAP's residual distribution and **13.5%** of
EVOLV's is pairing ambiguity, and the over-threshold tail is entirely one
directional (306 of 306 ASAP pairs are *late*). The residual also moves the wrong
way relative to video accuracy — EVOLV cues the video calls on-time carry median
150 ms while the six it flags off-time carry median 90 ms.

So the diagnosis in [residual_p95_characterisation.md](residual_p95_characterisation.md)
is right: this is a pairing defect, not a threshold problem.

## Candidates evaluated

| id | approach | intent |
|---|---|---|
| CUR | current greedy nearest-unused | baseline |
| A | monotonic nearest, order-preserving DP | remove the ordering artefact |
| B/E | monotonic, scored on interval overlap | score agreement not start distance |
| C | many-to-one, sharing allowed when the output interval lies inside the reference | handle finer output segmentation |
| F | two-pass: coarse overlap correspondence, residual inside it | separate correspondence from measurement |

Evaluated on 10 GOOD and 11 BAD fixtures, including the two real cases.

## Result: no candidate separates GOOD from BAD

| candidate | worst GOOD p95 | best BAD p95 | verdict |
|---|---|---|---|
| CUR current | 4775 ms | **0 ms** | overlap |
| A mono-nearest | 4770 ms | **0 ms** | overlap |
| B/E mono-overlap | 692620 ms | **0 ms** | overlap, and numerically broken |
| C many-to-one | 4760 ms | **0 ms** | overlap |
| F two-pass | 3230 ms | **0 ms** | overlap, and unsound |

The floor is the same in every column: **0 ms on BAD fixtures**. `wrong cut
(half)`, `truncated 5min`, `sparse (1/4)` and `deleted 40%` all score 0 under
every candidate, because a subtitle that covers less of the reference is
trivially well-matched to the part it does cover. This is the same
content-completeness wall the ground-truth study found, and it is not a pairing
problem — it is a property of any residual computed against a reference.

### Two candidates are additionally unsound

**B/E (monotonic overlap)** produces 63665 ms on EVOLV and 692620 ms on ASAP,
against 3310 / 4750 for the current metric. Zero-cost overlap makes the DP prefer
matching cues that do not overlap at all. The cost function needs a distance term
for the non-overlapping case, and a correctly-costed version was not achieved
here.

**F (two-pass)** scores **0 ms on `random corrupt 30%` and `duplicated 40%`** —
damaged inputs reading as perfect. Its coarse pass, which is supposed to be the
conservative part, is the pass that decides. On the real cases it pairs only 11 of
559 (EVOLV) and 0 of 780 (ASAP) cues, so it is not measuring those files at all.
A metric that cannot pair a demonstrably synchronized subtitle while calling
30%-corrupted input perfect is not a candidate.

### What the honest best outcome looks like

A (monotonic nearest) is the only candidate that is *directionally* sensible: it
removes the ordering artefact, is deterministic, and stays within a few hundred
ms of the current metric on the real cases. But it does not help — EVOLV goes
3310 → 4340 ms, i.e. slightly *worse*, and its BAD floor is still 0 ms. It fixes
the mechanism without improving the verdict, and on its own evidence it is not
worth changing production for.

## Safety mutations

The seven required mutations (wrong reference cue chosen, non-monotonic pairing,
many-to-one over-acceptance, tolerance widened, coverage check removed, residual
calculation bypassed, corrupted cue ignored) are not added, because there is no
candidate worth protecting. Adding mutation coverage for a metric that failed
safety would imply a commitment the evidence does not support. The existing
31/31 mutation suite continues to guard the production verifier, which is
unchanged.

## Conclusion

- The production verifier is **unchanged and still conservative**.
- The pairing defect is **real and now precisely characterised**, which is worth
  keeping.
- No candidate residual pairing is safe to adopt. The blocker is not the pairing
  algorithm; it is that no residual against a reference can distinguish "correct
  and complete" from "correct but truncated". That requires content-completeness
  evidence, which is a different and larger problem.
