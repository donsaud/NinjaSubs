# Known limitations, classified

Each entry states what kind of thing it is, so a limitation is not later
mistaken for a bug and fixed without evidence.

| # | limitation | kind | status |
|---|---|---|---|
| 1 | segmentation sensitivity | measurement limitation | characterised, not fixable with a threshold |
| 2 | weak drift detection | measurement limitation | documented, thresholds unchanged |
| 3 | single-episode video witness | evidence gap | corpus expanded to 8 real cases, 3 witnessed |
| 4 | conservative fallback on good Alass artifacts | **intended** behaviour | correct |
| 5 | ASS conversion boundary | **intended** limitation | now test-covered |
| 6 | `guessit` environment dependency | environment | documented with install steps |

## 1. Segmentation sensitivity — measurement limitation

A subtitle that is well synchronized but segmented differently from the reference
can fail verification, because the residual measures agreement with the
*reference's cue boundaries*. Mechanism characterised in
[residual_pairing_candidates.md](residual_pairing_candidates.md): greedy pairing
assigns later cues to later reference cues when the output is more finely
segmented (58.7% of ASAP's residual, 13.5% of EVOLV's).

**Not a bug.** Five candidate pairings were evaluated; none separates acceptable
from damaged input, because a truncated subtitle matches the part of the reference
it covers perfectly. Fixing it needs content-completeness evidence, not a
different matching rule.

Future experiment: a residual computed against the *activity* the reference
covers rather than its individual cue starts — but that must not weaken the
completeness checks that currently catch truncation.

## 2. Weak drift detection — measurement limitation

Drift of 300 ms/min still produces a well-behaved movement median. The existing
drift check and its 120 ms/min limit are unchanged and correct for what they
measure; they simply are not sensitive to slow ramps in a small sample.

**Not a bug, and not changed.** Raising or lowering the limit is not supported by
evidence.

Future experiment: a per-region drift estimate with an independently validated
reliability floor, which the earlier region investigation could not demonstrate.

## 3. Single-episode video witness — evidence gap

Only S08E04 has a real video fixture, so the offline witness exists for 3 of the
8 real corpus cases. The other 5 are labelled `AMBIGUOUS_NO_WITNESS_MATCH` rather
than guessed.

**Test/evidence gap, not a bug.** Conclusions do not generalise to other releases
yet.

Future experiment: obtain one more episode's media. Not required for the current
decision, which is to change nothing.

## 4. Conservative fallback on good Alass artifacts — intended behaviour

EVOLV (video witness 97.0%) and ASAP (99.9%) are rejected by the verifier and the
original is served. The Alass artifacts are demonstrably better aligned to the
video, but the verifier cannot measure that today.

**This is correct and must not be changed.** The verifier fails closed; serving a
transformed artifact it cannot vouch for would be the unsafe direction. The
conservative outcome is the designed outcome, and the alternative — accepting on
a manual or offline judgement — is explicitly out of scope for production
acceptance.

Future experiment: only after a measurement exists that tracks video accuracy
*and* rejects damaged input. See limitation 1.

## 5. ASS conversion boundary — intended limitation, now covered

A native ASS/SSA track is aligned only when the user opts into ASS→SRT
conversion. Alass emits SRT, so aligning ASS would destroy its styling; the
alternative is to serve the file untouched.

**Intended, and now pinned by tests** (`tests/test_ass_sync_boundary.py`): ASS is
detected by content regardless of extension, native ASS is returned byte-identical
with the sync path never entered, conversion is deterministic, and nothing
transformed reaches the provider cache.

## 6. `guessit` environment dependency — environment

`guessit>=3.8.0` is declared in `requirements.txt` and present in the container
(4.4.0), but absent from a bare host Python. Three tests fail without it,
including one that only appears to be a release-parsing bug: without `guessit`,
`extract_metadata` returns nothing and a `.en` language suffix leaks into the
release-family key.

**Not a repository defect.** Install the declared requirements to run the suite
green. No code change is warranted. See
[baseline_test_failures.md](baseline_test_failures.md).
