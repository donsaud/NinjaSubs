# Residual p95: characterisation and cause

This records *why* two subtitles the video says are correctly synchronized are
rejected by the residual check. It is characterisation only. No threshold is
proposed and no production behaviour changes.

## The observation

| case | video witness | residual p95 | verdict |
|---|---|---|---|
| EVOLV Alass output | 97.0% of cues within 1.5s | 3310 ms | UNVERIFIED |
| ASAP Alass output | 99.9% of cues within 1.5s | 4750 ms | UNVERIFIED |
| EVOLV original | 67.0% | 4630 ms | UNVERIFIED |
| ASAP original | 71.3% | 4850 ms | UNVERIFIED |

The Alass outputs are indistinguishable from their originals on the residual
metric while being 26–29 points better on the video. The residual statistic is
therefore not tracking the same property the witness measures.

## Where the residual actually is

| | median | MAD | p90 | p95 | p99 | max | matched | unmatched out |
|---|---|---|---|---|---|---|---|---|
| EVOLV | 150 ms | 70 | 2470 | 3310 | 4820 | 4990 | 527 | 32 |
| ASAP | 2420 ms | 1860 | 4510 | 4750 | 4970 | 5000 | 557 | 223 |

Two distinct shapes. EVOLV's *median* is excellent (150 ms) and the failure is
entirely in the tail — 13.7% of matched pairs exceed 2 s. ASAP's median is itself
elevated (2420 ms) with 54.9% over 2 s, and its signed deltas run +2226 ms on
average, all in one direction.

## Cause 1: pairing ambiguity dominates

For each match, the analysis asked whether a *nearer* reference cue existed that
the greedy one-to-one search had already consumed.

| | not the globally nearest | of those, a nearer cue inside the 5 s tolerance | share of the residual distribution |
|---|---|---|---|
| EVOLV | 71 / 527 | 71 | **13.5%** |
| ASAP | 327 / 557 | 327 | **58.7%** |

This is the mechanism. ASAP carries 780 cues against the reference's 562. When
several reference cues cover the same span, the greedy nearest-unused assignment
systematically picks the later one, and the residual records the reference's cue
*boundary* rather than any timing error. Both tails are one-directional — 70/72
and 306/306 of the over-2 s pairs are *late*, not early — which is the signature
of a segmentation artefact rather than of drifting dialogue.

It is not one-to-many segmentation in the usual sense: matched cues span at most
one reference cue, and only 5 of 562 reference cues go unmatched for ASAP. The
problem is the *inverse* case — the output has more, finer cues than the
reference, so each one gets matched to whichever reference cue is nearest in
start time, which is systematically offset from that cue's true start.

## Cause 2: the residual is inversely related to the video witness

Splitting each file's matched cues by whether the video says that cue is
on-time:

| | video on-time | video off-time | ratio |
|---|---|---|---|
| EVOLV | n=521, median 150 ms, p95 3310 ms | n=6, median 90 ms, p95 450 ms | **0.1×** |
| ASAP | n=557, median 2420 ms, p95 4750 ms | n=0 | — |

Cues the video identifies as correctly timed carry *more* residual than the
handful the video flags as late. For ASAP every matched cue is video-on-time, so
the residual cannot be falsified against the witness at all on that case. A
statistic that rises as timing accuracy improves cannot be used as a timing-accuracy
gate.

## Cause 3: not an opening artefact

| position | EVOLV median / p95 | ASAP median / p95 |
|---|---|---|
| 0–20% | 80 / 2380 ms | 2890 / 4820 ms |
| 20–40% | 100 / 3500 ms | 3130 / 4840 ms |
| 40–60% | 210 / 3590 ms | 2285 / 4620 ms |
| 60–80% | 160 / 4280 ms | 2750 / 4850 ms |
| 80–100% | 180 / 3020 ms | 360 / 4540 ms |

Flat across the episode for EVOLV. ASAP's last fifth improves (median 360 ms) —
the 223 unmatched cues are concentrated in the opening, where the reference and
output disagree most. The known opening discontinuity is real but is not what
produces the p95.

## Conclusion

The residual p95 on these two cases measures **agreement with the reference's cue
boundaries**, not synchronization with the video. The dominant mechanism is
greedy pairing ambiguity when the output carries more, finer cues than the
reference — 58.7% of the distribution for ASAP, and one-directional throughout.

This is the same finding the ground-truth study reached from a different angle,
and it is why B2 is a *measurement* limitation rather than a threshold problem.
Raising or lowering the limit does not address it: the statistic is measuring the
wrong quantity.

What would address it is a pairing rule that does not force one-to-one matching
on start times — but that is a new residual model, and [the ground-truth
document](sync_ground_truth.md) records why no available combination of signals
can be adopted safely yet. Nothing here justifies changing the threshold.
