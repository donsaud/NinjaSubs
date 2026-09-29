# Golden synchronization annotation protocol

This document defines how a case in `tests/fixtures/golden_sync/manifest.json`
receives its label. It exists so that "correct" means one specific thing, and so
that two people annotating the same case would write the same values.

## The rule that matters most

**Ground truth is never produced by the system under test.**

You may not label a case by asking `SyncState`, `AlignmentAnalyzer`,
`SyncPredictor`, `ReferenceTrust`, or the result of running Alass whether it is
correct. Those are the components being evaluated. Using their output as the
answer key would make the benchmark measure internal consistency, which is
exactly the failure this harness exists to avoid.

Acceptable sources of a label:

- Exact identity between the video and the subtitle (byte-identical subtitle
  shipped with the release, or a hash match).
- A timing relationship established independently of the pipeline — for example
  a subtitle whose cue starts were compared by hand against the video.
- Manual review of synchronization, watching the video.
- A manually confirmed release and cut relationship.
- Known-good reference material: a subtitle that is authoritative for the
  target and whose timing is not in question.

Unacceptable sources, without exception:

- "Alass exited 0."
- "The residuals looked small."
- "The subtitle looks plausible."
- "The provider said it matches."
- Anything the analyzer, predictor or trust model reported.

## The five questions

Annotate in this order. Each is a separate field; do not collapse them.

### A. Does the subtitle belong to the same content?

`content`:

| value | meaning |
|---|---|
| `same_release` | Same release of the same work. |
| `same_content_different_release` | Same work, different encode or source. |
| `different_content` | Different work, or a different episode. |
| `unknown` | Not established. |

### B. Is it the same release and cut?

`cut`: `same_cut`, `different_cut`, or `unknown`.

Same release does **not** imply same cut. A director's cut, an extended cut, a
different edit of a TV episode, or a regional variant are all
`different_cut` while still being `same_release`. This is the single most
important distinction in the dataset, because a wrong-cut subtitle is
internally perfectly consistent and still wrong for the target.

### C. Was the subtitle originally synchronized?

`original_sync`: `already_synced`, `resyncable`, `not_resyncable`, `unknown`.

Answer this about the subtitle as shipped, relative to the video, **before**
any alignment is attempted.

### D. Can it be correctly synchronized to the target?

`resyncable`: `true`, `false`, or `null`.

`true` only if some alignment of this subtitle onto this target produces
correct timing. `false` if no such alignment exists — for example a different
episode, or a different cut whose content genuinely does not correspond.

### E. After synchronization, is the resulting timing correct?

`final_alignment`: `correct`, `incorrect`, or `unknown`.

This is the label the false-positive analysis is measured against. A
successful alignment that lands on the wrong cut is `incorrect`. A refused
alignment is `correct` only if the refusal was the right call.

## Label provenance

`annotation_source` on each case:

- `objective` — established by identity or by an independent known-good
  relationship. No human watched anything.
- `human_reviewed` — a person inspected synchronization against the actual
  video.

These are tracked separately so a later analysis can tell which conclusions rest
on proof and which rest on judgement.

## Declared alignment outcomes

`ground_truth[].alignment` records the alignment result the harness hands to the
verifier, because the verifier's contract is to judge a subtitle that has
already been re-timed.

| mode | meaning |
|---|---|
| `as_target` | A correct aligner landed on the true timing. `residual_ms` adds ordinary jitter. |
| `as_target_with_drift` | Aligned, but a rate mismatch leaves `drift_per_minute_ms` of progressive drift. |
| `partial_misalign` | Exit 0, small residuals, but the timeline jumps at `displaced_from_index` by `displaced_by_ms`. This is the shape an aligner produces on a different cut. |
| `no_alignment` | The aligner produced nothing usable. |

Every field is written down so a reader can see exactly what the verifier was
shown. The harness does not invent alignment results; it materialises the
declared one.

**This is not a measurement of Alass.** The harness scores the verification
layer against known-correct and known-incorrect alignments. Whether Alass
produces those alignments is a separate question, and for synthetic fixtures it
is not measured at all.

## Hard negatives are mandatory

A dataset of only well-behaved cases proves nothing. Every revision must retain
cases that look plausible by metadata and are wrong:

- Same title, season, episode, release group and resolution; different cut.
- Same release metadata with a large timing displacement.
- Excellent structural similarity, wrong episode.
- Multiple providers carrying byte-different copies of one incorrect timing grid.

That last one is the circular case. Two providers agreeing is not evidence:
consensus is 100% and the answer is still wrong. The benchmark must catch it,
and the manifest includes `circular_shared_wrong_timing` for exactly that
purpose.

## Multiple valid references

Where more than one candidate is a legitimate synchronization anchor, list them
in `valid_reference_keys` and do **not** force a single "correct reference".
Such a case is a `VALID_REFERENCE_SET`. `duplicate_timing_grid` is one.

## Adding a case

1. Add the fixture under `tests/fixtures/golden_sync/subtitles/`.
2. Add a case with `python tools/generate_golden_fixtures.py`, or edit the
   generator and re-run it. The generator refuses to invent a label.
3. Record a `sha256` for every fixture.
4. If the case uses real media, keep the media local and out of band. Only
   manifests, small text fixtures and metadata belong in the repository.
5. Re-run the evaluator. If the new case changes a rate, the rate was too small
   to mean much; that is a fact about the dataset, not about the system.

## Reading the results

The benchmark reports counts, error classes, rates with Wilson intervals, and a
set of `INVESTIGATION CANDIDATES`. It never recommends a threshold change, and
it never promotes anything. Promotion decisions require measured evidence
across a dataset large enough for the interval to be meaningful; see §21 of the
phase specification.
