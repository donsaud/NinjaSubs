# Large Offset fixture corpus

## What this is

A deterministic, **entirely synthetic** subtitle corpus that reproduces the
measured timing structure of the confirmed Large Offset case (a release running
~+96.6 s late against a same-episode reference) so the serving safety properties
can be asserted offline, on any machine, with no third-party content.

## Why the dialogue is synthetic

The timings were measured on real, commercially released subtitles for a
television episode. A timeline is a set of facts and is not the copyrighted
work; the dialogue text is. This repository is public, so **no third-party
dialogue is committed here.** Every cue carries generated text built from a
closed vocabulary by a fixed-seed PRNG.

The real, non-redistributable files are not in this repository. They stay on the
owner's machine under `test-media/` (gitignored) and are used only for private
production validation. See `docs/stremio_playback_validation.md`.

## What is preserved, and why it matters

Substituting the text changes nothing that any test depends on:

| Preserved | Value | What it protects |
|---|---|---|
| cue count | 562 / 560 / 714 / 3 | sampling bounds, region counts |
| first dialogue | 10710 ms / 14570 ms / 93840 ms | the ~+96.6 s displacement the gate must measure |
| per-cue word counts | exact | text-volume / density metrics |
| cue durations and gaps | exact | `p95`, residual and gap-similarity signals |
| negatives | same TIMING transforms of the same reference | *what refuses them is unchanged* |

The three negatives are not re-authored; they are re-derived from the rewritten
reference, so the property that matters — that `different_cut`, `wrong_episode`
and `drifting` are refused for the stated structural reasons — is structurally
identical to the corpus the measurements were taken on.

## Files

| File | Role |
|---|---|
| `dexter_s08e04_reference.srt` | the good same-episode reference |
| `dexter_s08e04_target.srt` | the late release; the thing being corrected |
| `dexter_s08e04_evolv_original.srt` | per-release target, 560 cues |
| `dexter_s08e04_asap_original.srt` | per-release target, 781 cues |
| `valid_constant_offset.srt` | a *second* valid large offset (~+93 s) |
| `negative_different_cut.srt` | +600 s after the midpoint: different edit |
| `negative_wrong_episode.srt` | different episode, disjoint vocabulary |
| `negative_drifting.srt` | opening agrees, then walks away at 300 ms/min |
| `negative_insufficient_anchors.srt` | 3 cues: not enough to corroborate |
| `alass_out/out_*.srt` | real `alass` 2.0.0 output for each reference |
| `alass_out/fx/` | the exact inputs `alass_out/run.sh` was run against |
| `alass_out/run.sh` | regenerates the `out_*.srt` files (needs the image) |

## Regenerating

```sh
python tools/build_synthetic_large_offset_fixtures.py   # dialogue, from timings
docker compose run --rm --entrypoint sh ninjasubs \
    -v "$PWD/tests/fixtures/large_offset/alass_out:/tmp/lo_mad" \
    -w /tmp/lo_mad ninjasubs /tmp/lo_mad/run.sh          # alass outputs
```

Both steps are deterministic: same inputs, same bytes. The generator reads only
the timings already committed here, so it never needs the real media.

`tools/build_large_offset_fixtures.py` reports the original measurements the
timings came from, for a machine that still holds them. It prints timings only.

## Expected serving outcomes

Asserted by `tests/test_large_offset_serving.py::test_real_alass_outputs_are_routed_as_measured`:

| Reference | Serving state | Refused because |
|---|---|---|
| `dexter_s08e04_reference` | `alass_corrected_large_offset` | — |
| `valid_constant_offset` | `alass_corrected_large_offset` | — |
| `negative_different_cut` | `alass_corrected_large_offset` | — (**known false accept**, documented in `large_offset_investigation.py`) |
| `negative_wrong_episode` | `original` | `movement_not_a_constant_shift` |
| `negative_drifting` | `original` | `movement_not_a_constant_shift` |

In every row the analyzer's own verdict stays `unverified`/`unknown` and the
artifact stays out of the reusable cache. Serving a correction is not a
verification.