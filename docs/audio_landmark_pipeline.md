# Audio landmark pipeline audit

Performed before modifying the extractor, per the phase specification. Every
claim here was verified against the project's own container, not assumed.

## What the current pipeline does

```
ffprobe
  ├─ container duration, stream list, start_time
  └─ select_audio_stream()  → programme audio or AUDIO_STREAM_AMBIGUOUS
        ↓
ffmpeg -af silencedetect=noise=-45dB:d=0.700
  └─ parses "silence_start:" / "silence_end:" from stderr
        ↓
landmarks = sorted(set(values - stream_start_time))
```

## Parameter-by-parameter

| stage | parameter | value | who controls it |
|---|---|---|---|
| decode | stream selection | `select_audio_stream()` | **ours** |
| decode | channel mix | default downmix to mono | ffmpeg's |
| energy | RMS window | internal to `silencedetect` | **ffmpeg's, not configurable** |
| energy | hop size | internal to `silencedetect` | **ffmpeg's, not configurable** |
| activity test | amplitude floor | `noise=-45dB` | **ours** |
| activity test | minimum silence | `d=0.700` (700ms) | **ours** |
| smoothing | none | — | nothing exists |
| hysteresis | none | — | nothing exists |
| minimum activity | none | — | nothing exists |
| boundary merging | none beyond `set()` | — | nothing exists |
| normalisation | programme start | `value - stream_start_time` | **ours** |
| dedup | `sorted(set(...))` | — | **ours** |

## The decisive audit finding

`silencedetect` decides activity **per sample-ish window**, not on a long
integration. Verified empirically: a tone burst of 20ms, 50ms, 100ms, 200ms,
400ms, 700ms and 1000ms inside digital silence all produce
`silence_start` at exactly the burst's end.

```
burst    20ms -> silence_start: 0.02
burst    50ms -> silence_start: 0.05
burst   100ms -> silence_start: 0.1
burst  1000ms -> silence_start: 1
```

So `d=0.700` is **not** a smoothing window. It is a gate on how long silence
must persist before an event is *emitted*. Temporal behaviour is already
adequate; the resolution is not the problem.

**Therefore the amplitude floor is the only thing standing between this pipeline
and working on real audio.** The stress run confirmed it: at `-45dB` the
pipeline finds 24/24 boundaries on clean audio and **0/24** on ambience, crowd
noise and a music bed, while `-30dB` recovers 12/12 on two of those three.

## Why the fixed detector cannot simply be re-tuned

Raising the floor globally does not work, and the stress run shows why:

| scenario | -45dB | -35dB | -30dB | -25dB | -20dB |
|---|---|---|---|---|---|
| ambience | 0 | 0 | 12 | 12 | 12 |
| music bed | 0 | 0 | 0 | 0 | 12 |
| uneven gain | 12 | 12 | 12 | 11 | 8 |

A music bed needs `-20dB`; uneven gain starts losing real boundaries at
`-20dB`. There is no single constant that serves both, because the two
requirements are opposites: one needs a permissive floor, the other a strict
one. A fixed dB value is the wrong *shape* of parameter, not merely the wrong
value.

Three further gaps make re-tuning insufficient even if a magic number existed:

1. **No smoothing or hysteresis.** Small fluctuations around the floor will
   chatter, producing repeated micro-boundaries that a landmark comparison then
   has to treat as real structure.
2. **No minimum activity duration.** Only silence is gated. An asymmetric
   filter invites a long trail of spurious activity boundaries.
3. **Window and hop are not ours.** `silencedetect` exposes neither, so none of
   §7's smoothing or hysteresis can be layered on top of it. Implementing our
   own detector over decoded samples is the only route to the required
   behaviour.

## What the adaptive detector must add

Per the specification, and mapped onto what the audit shows was missing:

- a **per-file baseline** from robust statistics of short-time energy, replacing
  the absolute floor;
- **bounded** threshold derivation, so an extreme recording cannot produce a
  nonsensical value;
- **smoothing and hysteresis** (enter/exit activity thresholds), which the
  current pipeline structurally cannot provide;
- **symmetric minimum durations** for both activity and silence;
- **boundary separation and merging**, plus pathological-output detection;
- a **quality verdict** so thin or unstable output abstains instead of becoming
  evidence.

## Status of the two detectors

Both remain available. The fixed detector is unchanged and still the default;
the adaptive one runs in benchmark and shadow modes only. Neither replaces
production behaviour, and no production threshold moves.

## The adaptive detector: design and measured result

`app/services/sync/audio_activity.py` derives its threshold from each file's
own energy distribution:

```
decoded mono samples (once per file, cached)
  -> short-time RMS in dBFS
  -> split the distribution at its widest gap in the lower half
  -> threshold constrained to sit INSIDE that gap
  -> median smoothing + hysteresis (enter/exit)
  -> minimum activity and silence durations
  -> boundary separation and merging
  -> landmarks + a quality verdict
```

### Two things that were tried and measured first

**A fixed low percentile is the wrong estimator.** It was implemented and
measured. The quiet fraction of a file is not stable: a clip that is 84%
active puts its 20th percentile inside the *active* level, so the derived
threshold rises above the audio and nothing is ever detected, while a clip that
is 50% silent puts the same percentile deep in the silence. The floor has to
be *found*, not sampled at a fixed rank. Splitting at the widest gap is
indifferent to that ratio.

**A margin alone is not enough either.** With a music bed the gap between bed
and dialogue is only ~12dB, and `enter = baseline + 6dB + margin` landed
*above* the dialogue, so nothing registered as active. The fix is not a
smaller margin but a constraint: the threshold is clamped to lie strictly
between the floor and the bulk. That works regardless of how narrow the
recording's dynamics are.

### Measured result, fixed versus adaptive

Seventeen scenarios, amplitude dynamics only, ground truth authored before
synthesis and never read from a detector:

| scenario | fixed P/R | adaptive P/R | dF1 |
|---|---|---|---|
| ambience | 0.00/0.00 | 0.96/0.96 | **+0.96** |
| music bed | 0.00/0.00 | 0.96/0.96 | **+0.96** |
| dynamic music | 0.00/0.00 | 0.96/0.96 | **+0.96** |
| crowd noise | 0.00/0.00 | 0.96/0.96 | **+0.96** |
| drifting noise floor | 0.00/0.00 | 0.96/0.96 | **+0.96** |
| effects over ambience | 0.00/0.00 | 0.96/0.96 | **+0.96** |
| quiet scene with tone | 0.00/0.00 | 0.96/0.96 | **+0.96** |
| compressed dynamics | 0.00/0.00 | 0.96/0.96 | **+0.96** |
| scene level steps | 0.00/0.00 | 0.96/0.96 | **+0.96** |
| **rapid dialogue** | 1.00/1.00 | **0.50/0.04** | **-0.92** |
| clean, effects, uneven gain, long silence, dialogue envelope, overlapping activity, sudden gain change | 1.00/1.00 | 0.96/0.96 | -0.04 |

dF1 is reported as a **difference**, not an improvement claim, and the
scenarios were not used to choose the parameters.

### The regression is real and was not tuned away

`rapid_dialogue` regresses badly. The cause is deliberate: its 900ms gaps sit
below `min_boundary_separation_ms = 1500`, so the detector correctly declines to
emit boundaries for sub-1.5s fluctuations. The fixed detector emits them only
because it has no such rule. This is a parameter trade-off, not a defect, and
relaxing the separation constant to win this row would be exactly the
overfitting §26 warns against. It is recorded in the artifact and asserted by a
test so it cannot disappear quietly.

## Correlation: whole-timeline and regional, with ambiguity

Best-offset alignment is unchanged in principle but now records what it did:

- `correlation_peak`, `correlation_second_peak`, `correlation_peak_ratio` and a
  `clarity` of clear / ambiguous / no-signal;
- per-region shifts and a `regional_consistency` of consistent / mixed /
  scattered, kept separate rather than averaged.

**A strong peak with a near-equal runner-up is ambiguous and abstains.** A
timeline has not been aligned, it has been guessed at.

**A weak best peak is disagreement, not ambiguity.** Refusing to call a badly
wrong cut "ambiguous" would let it pass as merely unclear, so the precedence is
deliberate: weak means mismatched, strong-but-not-unique means insufficient.

Two things this exposed, both fixed in the code rather than the tests:

- A **weak peak with many tied competitors** reads as `peak_ratio == 1.0`. With
  the precedence above, that is correctly a mismatch rather than a refusal.
- **Region ties were broken by offset ordering.** A short region ties at a low
  score across thousands of offsets and reported an arbitrary one, which then
  looked like regional inconsistency that was not in the audio. Ties are now
  broken toward the global best offset, so regional shifts are *deviations from
  one answer* and are comparable.

## Status of the VAD gate

The five-category gate is unchanged and no VAD was added. Against the previous
phase's categories:

- **A — fixed-floor problem: solved** on the nine scenarios above.
- **B — adaptive threshold: this phase**, and it resolves the measured cases.
- **C — landmark extraction: partly open.** The `rapid_dialogue` separation
  trade-off sits here.
- **D — ambiguous/no-signal: still handled by abstention.**
- **E — requires semantic/audio understanding: no evidence.** No measured
  failure needed it.

Synthetic results alone do not justify VAD, and no real programme audio has been
measured. The detector remains benchmark- and shadow-only.
