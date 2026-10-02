# Synchronization ground truth: what this project accepts

This document records what the project treats as acceptable synchronization,
established from measurement rather than assertion. It exists because the
classifier investigation could not separate good alignments from damaged ones
until the accept/reject boundary was written down.

Nothing here changes production behaviour. The verifier is unchanged.

## The measurement

Two independent sources of truth were compared over 33 controlled fixtures
derived from the real PiR8 S08E04 reference:

- **Video witness** — the fraction of a candidate's cue starts falling within
  1.5s of an actual subtitle presentation in the video's own embedded English
  PGS track. Objective, independent of the pipeline, offline evaluation only.
  It is never a production signal.
- **Current verifier** — the production `AlignmentAnalyzer` decision.

## What the study found

**The video witness cannot define the boundary.** Across the 11 fixtures the
project considers acceptable the witness ranges 71.4%–98.9%. Across the 13 that
must be rejected it ranges 67.6%–99.3%. The ranges overlap almost completely.

The reason is structural rather than incidental. Fixtures that remove or
duplicate content while leaving timing alone score *higher* than a perfect
alignment, because deleting half an episode still leaves every surviving cue in
the right place:

| fixture | witness | should be |
|---|---|---|
| already aligned | 98.9% | accepted |
| different cut (first half) | **99.3%** | rejected |
| sparse (every 4th cue) | **99.3%** | rejected |
| duplicated 40% cues | **99.1%** | rejected |
| deleted 40% cues | 98.8% | rejected |
| truncated (first 5 min) | 97.9% | rejected |
| nearly empty (45 cues) | 97.8% | rejected |

So a content-complete subtitle aligned within a second, and a quarter of an
episode with nothing in it, are indistinguishable by timing agreement alone.
**Coverage, not timing, is what separates those cases** — which is why the
project's existing structural and cue-retention checks exist and why removing
them would be unsafe.

**The witness also saturates on real media.** The two manually-confirmed-good
real Alass outputs score 97.0% and 99.9%. The two manually-confirmed-poor
originals score 67.0% and 71.3%. That is a real 30-point separation and it
correctly ranks all four. But it is measured against one episode's PGS track,
so it establishes a ranking, not a portable threshold.

## Where the witness and the verifier disagree

No fixture scored ≥90% on the witness and was rejected by the verifier.

Four fixtures scored <70% and were rejected: the two wrong releases and the two
real originals. The verifier and the witness agree on all of those.

The disagreement is one-directional and specific: `uniform +96.5s` scores 63.5%
on the witness and is rejected. That is a limitation of the witness, not of the
verifier — a uniformly shifted subtitle is genuinely misaligned to the video,
and the reference the verifier compares against is the correct timeline.

## Provisional policy

The boundary below is stated as the project's position, with its evidence. It
is not a threshold any component currently applies.

| class | definition | evidence |
|---|---|---|
| **ACCEPTABLE** | content complete, and aligned to the target timeline within the project's existing stability rules (p95 ≤ 2000ms, MAD ≤ 800ms, drift ≤ 120ms/min, structural ≥ 0.55, cue retention ≥ 0.80) | 11 synthetic fixtures, 2 real Alass outputs |
| **ACCEPTABLE (large)** | as above, with a uniform or piecewise correction larger than the 20s normal window, when every region is independently stable | `opening +96.5s only`, `closing +96.5s only`, `piecewise 3-region` — all three verified `VERIFIED_RESYNCED` with residual p95 = 0 |
| **REJECT** | wrong release, wrong cut, truncated, sparse, nearly empty | detected by coverage, structural and cue-retention, never by timing alone |
| **REJECT** | duplicated, deleted, or randomly corrupted cues | detected by coverage and mass, not by timing |
| **AMBIGUOUS** | anything where content loss and timing error are confounded and the witness is not available | not classifiable from timing evidence |

Two facts from this study are load-bearing:

1. `opening +96.5s only`, `closing +96.5s only` and `piecewise 3-region` are
   **already accepted today** with residual p95 = 0. The verifier does support
   legitimate piecewise corrections when each region is genuinely stable. What it
   cannot do is accept a *damaged* one, and Phase 3 showed no available signal
   can.
2. The two real Dexter Alass outputs are the hard case: the witness says they
   are well aligned (97.0%, 99.9%), the verifier rejects them (residual p95
   3310ms and 4750ms). The service serves the original. That is the documented
   conservative outcome, and this study does not overturn it.

## Consequence for further work

A new verification metric is **not** justified on this evidence. A timing-based
metric cannot see content loss (the witness scores a quarter-episode at 99.3%),
and no combination of coverage, structural, mass, count, monotonicity or lag
variance separates all acceptable fixtures from all damaged ones — an
exhaustive search over 142,186 conjunctions left five damaged fixtures passing.

Any future attempt should attack the specific measurement problem, not the
threshold problem: the residual statistic disagrees with the video for two
subtitles the video says are correct, and no signal available in production
explains why.
