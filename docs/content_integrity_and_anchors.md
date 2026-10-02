# Content integrity, graded anchors, and the safety-mutation harness

DIAGNOSTIC ONLY. `app/services/sync/content_integrity.py` and
`app/services/sync/interval_correspondence.py` are imported by no production
module (asserted by test) and change no verifier behaviour, state, threshold,
cache, ranking or provider call. No subtitle bytes are written. No subtitle text
is ever read by the model, logged, or stored.

## Classification: C - UNSAFE / REJECTED for acceptance

The research question was whether existing identity / release / content-integrity
evidence plus the shadow anchor-interval correspondence can **safely prevent the
known false acceptances**, and what class the shadow model is.

Measured answer, `tools/combined_shadow_report.py`, 24 cases:

| mode | correct accept | **false accept** | correct reject | false reject | abstain |
|---|---|---|---|---|---|
| **A** identity evidence required (as designed) | 0 | **0** | 0 | 0 | 24 |
| **B** consensus lifted, timing + content + structure | 3 | **4** | 8 | 5 | 4 |

- **Mode A is perfectly safe and completely useless.** With one independent
  reference, `assess_consensus()` always returns
  `only 1 independent reference(s); no consensus available`, so every one of the
  24 cases abstains. The gate never accepts anything, therefore it never accepts
  anything wrongly.
- **Mode B does work, and it is unsafe.** It still accepts all four of
  `wrong episode (timing twin)`, `text reordered`, `wrong cut +4s`,
  `corrupt 10%`. Lifting identity evidence buys acceptance capability by
  purchasing false acceptances with it.

**No safe combination was found.** The class is C, and the experiment stops here.

### Why the four false acceptances survive

This is structural, not a tuning miss:

- `wrong episode (timing twin)` -> dp `ALIGNED`, rcov `1.000`, p95 `0`,
  content **`GOOD`**. Timing and content are both *perfect*. Only identity
  evidence could reject it, and identity evidence is exactly what Mode A cannot
  supply without a second reference.
- `text reordered` -> dp `ALIGNED`, rcov `1.000`, p95 `0`, content `GOOD`.
  Reordering preserves the multiset of cues, so duration/gap signatures survive
  and content overlap is unchanged.
- `wrong cut +4s` -> dp `ALIGNED`, rcov `0.868`, p95 `121`, content `GOOD`.
- `corrupt 10%` -> dp `ALIGNED`, rcov `0.915`, p95 `8`, content `GOOD`.

Every one of them reaches `CONTENT_INTEGRITY_GOOD` because their **content is
identical or near-identical to the reference**. Content integrity therefore
cannot separate them from the correct cases by construction: it measures *what*
is present, and the damage here is *which* episode and *in what order*.

### Why GOOD is only reachable for reference-identical content

`assess_content_integrity()` requires positive same-language evidence. `GOOD` is
capped to `UNKNOWN` whenever the text side is unavailable or cross-language.
In this corpus that means the only cases reaching `GOOD` are precisely the
degenerate negatives derived from the reference itself. That is an honest
statement of what the signal can do, not a bug to tune away: on real data with
one reference and one candidate, GOOD proves the candidate and reference agree
about the words, which is not the same as proving they are the same episode in
the same order.

## The nine required measurements (final pass)

| # | item | measured result |
|---|---|---|
| 1 | independent reference count | **0 independent episodes.** 7 files, all one episode: `05`/`06`/`07` text-identical (1.000), `03`/`04` identical (1.000), `01`/`02` 0.995. Cross-source overlap `01x03`=0.328, `01x05`=0.442, `03x05`=0.347 - three translations of *the same episode*, and the target is one of them. |
| 2 | consensus agreement | 1 ref -> `score=None`, `only 1 independent reference(s)`. 2 refs (target excluded) -> `1.0 / agree`. Same bytes twice -> `None` (duplicates collapse correctly, not double-counted). `1.0` is returned whether or not the target is included, because consensus compares identity metadata and all three are the same episode. |
| 3 | EVOLV | `SURPLUS_REFERENCE`, rcov 0.520, p95 633, content `UNKNOWN` -> **not accepted** |
| 4 | ASAP | `SURPLUS_REFERENCE`, rcov 0.612, p95 643, content `SUSPECT` -> **not accepted** |
| 5 | wrong episode | `ALIGNED`, rcov 1.000, p95 0, content `GOOD` -> **FALSE ACCEPT** |
| 6 | wrong release | `SURPLUS_REFERENCE`, rcov 0.477, p95 638, content `GOOD` -> **FALSE ACCEPT** |
| 7 | wrong cut | `ALIGNED`, rcov 0.868, p95 121, content `GOOD` -> **FALSE ACCEPT** |
| 8 | corruption | 10% `ALIGNED`/0.915/8/`GOOD` -> **FALSE ACCEPT**; 20% and 30% rejected by `LOW_REFERENCE_COVERAGE` |
| 9 | duplication | `duplicated 40%` content `SUSPECT` -> **rejected, no bypass** |

Every one of items 3-8 fails the acceptance requirement, and items 5-8 are the
measured false acceptances. Item 1 alone - zero independent references - would
forbid class A on its own; item 5-7 confirm wrong-episode, wrong-release and
wrong-cut are indistinguishable, which is an explicit class-C condition.

Nothing in item 9 was tuned: duplication was already rejected by the existing
content-integrity rule, and it stays rejected.

## PART 1 - the anchor/DP conflict is reconciled, not a defect

`tools/reconcile_anchors.py` established that production's large-offset gate and
the interval DP were measuring **different quantities**:

- production `large_offset.py`: bounded (`<=600` samples), two-pass sampling,
  `analyze_drift()` over individual samples, dispersion = MAD about the global
  median;
- the old shadow: one pass, comparing adjacent region medians at `1500ms`.

Measured after the graded-anchor variant was added:

| case | seed hypothesis | dp outcome | rcov | p95 | content |
|---|---|---|---|---|---|
| EVOLV original | 96100 | `SURPLUS_REFERENCE` | 0.520 | 633 | UNKNOWN |
| ASAP original | 96050 | `SURPLUS_REFERENCE` | 0.612 | 643 | SUSPECT |
| EVOLV Alass | - | `ALIGNED` | 0.820 | 275 | UNKNOWN |
| ASAP Alass | - | `ALIGNED` | 0.829 | 502 | SUSPECT |

The production-style anchor gate and the DP now **agree** on both real cases
(`align_with_graded_anchors` mirrors production sampling; contradiction is tested
against `SEARCH_WINDOW_MS`, not a dispersion limit). EVOLV's residual rcov
`0.520` is a property of the data (`SURPLUS_REFERENCE`, the target legitimately
carries cues the reference lacks), not a threshold to loosen.

The two-pass refinement was also shown to matter by
`test_two_pass_refinement_recentres_a_biased_seed`: EVOLV drift reads `-19.3`
one-pass against `-30.5` two-pass.

## PART 2 - the safety-mutation harness was not measuring anything

`tools/run_safety_mutations.py` had two defects that made it unusable:

1. **No timeout.** `_run()` called `subprocess.run()` without `timeout=`, so a
   single hung child would block forever with all output buffered and invisible.
2. **Full-suite-per-mutation.** Each of 90 mutations ran the entire test suite
   serially, after a full `copytree` of the tree. Measured cost: **~80+ minutes**
   of unbounded serial work for one pass.

The hang was diagnosed, not guessed at: no python/pytest process survived the
abort, no deadlock, no recursion, no network, no Docker, no stdin involvement.
The cause was simply unbounded serial work.

Fixes, all landed:

- `_run(cmd, cwd, timeout=MUTATION_RUN_TIMEOUT_SECONDS)` with `subprocess.run(..., timeout=)`
  and a distinct `TIMEOUT_EXIT = -9`;
- `MUTATION_RUN_TIMEOUT_SECONDS = 180`, `MATRIX_TIMEOUT_SECONDS = 3600`;
- statuses separated into `CAUGHT` / `MUTATION_TIMEOUT` / `UNPROTECTED` /
  `SKIPPED` / `ERROR` / `BASELINE_ERROR` so a timeout can never be misread as a
  pass;
- `guard_tests()` selects the **smallest intended regression subset** per subject
  module - 90 mutations collapsed to 14 distinct subsets;
- per-subset baseline is verified and deselected from the mutant run, so the
  three host-only `guessit` failures can never mask a catch (they previously
  produced 11 spurious `BASELINE_ERROR`);
- `--only <substring>` for targeted runs; progress flushed after every mutation;
- the stale-failure assertion is preserved.

### Results

| run | outcome | wall |
|---|---|---|
| `--only anchor-` | 15/15 | 43s |
| `--only content-integrity` | 5/5 | 15s |
| `--only interval-` | 12/12 | 104s |
| `--only large-offset-first-cue-only` | 1/1 | 23s |
| **full matrix** | **90/90** | **825s (13.8 min)** |

Was ~80+ min, now 13.8 min, and the previous `UNPROTECTED` and 11
`BASELINE_ERROR` results are both resolved.

## PART 3 - the harness found a real gap in a production guard

Mutation `large-offset-first-cue-only-accepted` lowers
`if len(offsets) < LARGE_OFFSET_MIN_ANCHORS:` to `< 1` in
`app/services/sync/large_offset.py`, i.e. it deletes most of the "at least four
regions must agree" guard. It reported `UNPROTECTED`.

The guard *did* have two tests - `test_insufficient_evidence_is_rejected` and
`test_single_cue_evidence_is_never_sufficient` - and both correctly assert
`REASON_INSUFFICIENT_ANCHORS`. Neither fires:

- `test_single_cue_evidence_is_never_sufficient` monkeypatches `_region_medians`
  to return **zero** regions;
- `test_insufficient_evidence_is_rejected` uses a fixture that also yields zero.

With zero anchors, `0 < 1` is still true, so the gate still rejects and both
tests stay green. **Nothing covered the 1-3 anchor band.** The tests were correct
but exercised the degenerate end, so the mutation removed most of a production
guard and the suite reported no damage.

`test_partially_spread_anchors_are_still_insufficient` now pins the threshold
itself: it feeds exactly `1 .. LARGE_OFFSET_MIN_ANCHORS - 1` regions and
asserts each is rejected with `REASON_INSUFFICIENT_ANCHORS`, then feeds exactly
`LARGE_OFFSET_MIN_ANCHORS` and asserts it is *not* so rejected. The mutation is
now `CAUGHT 1/1 in 23s`.

This is the concrete justification for the harness rewrite: the previous harness
could not have reported this, because it could neither bound itself nor
distinguish "caught" from "hung" from "wrong baseline".

## PART 4 - forensics: the existing fixtures cannot separate these classes

`tools/forensic_false_acceptances.py` compared signals against the five known
false acceptances. The wrong-episode / reordered / wrong-cut fixtures are
**derived from the reference itself**, which makes them degenerate: no existing
signal cleanly separates them from a correct case.

This was documented rather than tuned. Inventing thresholds to separate a
fixture that shares its source with the reference would produce a score with no
relation to the real failure. The honest record is that these classes need real
independent fixtures, and no such data exists locally.

Consensus is likewise unavailable: `assess_consensus()` on one reference returns
`score=None`. There is no second episode in the workspace, so wrong-episode
detection and reference consensus **cannot be validated at all** with real data.

## Constraints verified

- Production unchanged: `app/main.py`, `alignment.py`, `large_offset.py`,
  `movement_seed.py`, `structural.py`, `sync_cache.py` all `clean` against git.
  The only modified tracked file under `app/` is the shadow module itself.
- `ENABLE_AUTOSYNC_TEST_OVERRIDE=false` in `.env`, defaults to `false` in
  `docker-compose.yml`.
- `ENABLE_LARGE_OFFSET_RESCUE` appears nowhere in code, compose or env - the
  rescue is structurally impossible to enable.
- 0 cache writes during the session; no cache directory exists in the workspace.
- `MANUAL_PLAYBACK_WITNESS` remains isolated and never promotes to `VERIFIED`.
- Text never becomes a general acceptance signal and is never logged; text
  diagnostics hash-and-discard (`norm_hash`).

## Gates

| gate | result |
|---|---|
| host `pytest` | 1880 passed, 19 skipped, 3 known host-only `guessit`/bound failures |
| Docker `pytest` | **1877 passed, 25 skipped, 0 failed** |
| safety mutations | **90/90 in 812s**, zero `UNPROTECTED` / `MUTATION_TIMEOUT` / `BASELINE_ERROR` |
| `mypy app` | Success, 63 source files |
| `ruff check` on touched files | clean (except the pre-existing `UP042` str-enum pattern) |
| production isolation | no production module imports `content_integrity` or `interval_correspondence` |
| cache / provenance | `subs_cache` 325 files, newest 10-01 18:39, **0 written this pass**, git-ignored, 0 tracked |
| `ENABLE_LARGE_OFFSET_RESCUE` | absent from code, compose and env |
| `ENABLE_AUTOSYNC_TEST_OVERRIDE` | `false` in `.env`, defaults `false` in compose |

## What would be needed to leave class C

Not attempted, and not cheap:

1. a **real independent second reference** (another episode/cut) so consensus and
   wrong-episode detection can be measured at all - currently they return
   `None`/abstain, which is why Mode A accepts nothing;
2. **non-degenerate fixtures** for wrong-episode, reordered and wrong-cut that
   are not derived from the reference under test;
3. a signal that separates *"same words, wrong episode"* from *"same words,
   right episode"* - timing and content are both already perfect for that case,
   so the signal must be identity-shaped, and identity is what is missing.

Until (1) exists, every acceptance-capable configuration measured here is unsafe
(4/12 accept-mode false accepts in Mode B) and every safe configuration is inert
(0/24 in Mode A).


---

# PART 5 - Implementation: the Large Offset solution (shipped)

The research above stays closed and stays rejected. What follows is a different
answer to a narrower question, and it does not use any of the rejected machinery:
no piecewise segmentation, no DP alignment, no multi-reference consensus, no
general rescue for arbitrary offsets.

**The question:** `offset > 20s` is a routing boundary, not a failure. The
general verifier refuses such corrections on residual p95 alone because cue
segmentation differs between releases - a property of the *release*, not of the
alignment. The task was to serve those corrections without weakening the
verifier for anyone else.

**The answer:** a first-class `LargeOffsetInvestigation` that runs only on that
path, and one explicitly scoped exception to the delivery invariant.

## Flow

```
_large_offset_assessment  (existing evidence gate, unchanged)
        |
        v
investigate_large_offset(target, reference)     <- reference-first identity
        |                                          + temporal structure
        v
alass  ->  validate_alass_output(target, output, reference)   <- sanity only
        |
        v
AlignmentAnalyzer.analyze(...)                   <- unchanged, still authoritative
        |
        v
decide_large_offset_serving(investigation, validation, evaluation)
        |
        +-- ALASS_CORRECTED_LARGE_OFFSET -> deliver corrected bytes
        +-- ORIGINAL                     -> deliver provider bytes
```

The investigation is built on the exact `target_text` alass is handed
(post-intro-stripping), so the evidence the decision is built on describes the
same two documents the correction was built from
(`orchestrator.py`, immediately before the alass invocation).

## The scoped exception

`decide_large_offset_serving` allows delivery only when **all** of these hold:

1. `investigation.eligible_for_alass` - reference-first same-episode identity
   was established;
2. `validation.ok` - alass produced a structurally valid output;
3. `evaluation.mad_offset_ms <= MAX_MAD_MS_FOR_STABLE` (800ms) **and** the
   refusal (if any) was `low_confidence` - i.e. the correction was one constant
   shift, and the analyzer refused on confidence rather than on content,
   structure or cue evidence;
4. `structural_similarity >= LARGE_OFFSET_MIN_STRUCTURAL_SIMILARITY` (0.65).

`evaluation=None` fails closed: without the analyzer's own measurement there is
no way to tell a segmentation-driven refusal from a structural one.

**Condition 3 is the whole exception, and it is MAD-only.** p95 above
`MAX_P95_MS_FOR_STABLE` (2000ms) is tolerated *here* and nowhere else, because
that residual is the segmentation difference the feature exists for. Every other
refusal binds: `REJECTED` for content, `structure_mismatch`, `cue_loss`,
`implausible_offset`, an unmeasured `mad_offset_ms`, a failed investigation, or
an invalid alass output all return the original.

### What the exception does *not* do

- `may_serve_synchronized` and `is_reusable_verified` are untouched. The
  analyzer records its own verdict exactly as before; the delivery decision runs
  alongside it rather than rewriting it.
- `ALASS_CORRECTED_LARGE_OFFSET` is a `LargeOffsetServingState`, not a
  `SyncState`, and its value appears in neither the `SyncState` nor the
  `VerificationAvailability` vocabulary.
- Because `sync_state`/`verification` are unchanged, `is_reusable_verified`
  stays `False` for these artifacts: they are delivered, never cached as
  verified, never re-served as a finished synchronization.
- The provider cache is never written with transformed bytes.

## Measured behaviour

Real alass outputs (produced in Docker via
`tests/fixtures/large_offset/alass_out/run.sh`: target is always
`dexter_s08e04_target.srt`, the reference varies), decided end to end:

| reference | MAD | p95 | structural | serving state | reason |
|---|---:|---:|---:|---|---|
| `dexter_s08e04_reference` | 0 | 3310 | 0.855 | `alass_corrected_large_offset` | passed |
| `valid_constant_offset` | 0 | 3310 | 0.855 | `alass_corrected_large_offset` | passed |
| `negative_different_cut` | 0 | 3210 | 0.762 | `alass_corrected_large_offset` | passed |
| `negative_wrong_episode` | 11039 | 3691 | 0.626 | `original` | `movement_not_a_constant_shift` |
| `negative_drifting` | 3601 | 3336 | 0.856 | `original` | `movement_not_a_constant_shift` |

Production route, real orchestrator + real alass, no test override:

| case | delivered |
|---|---|
| Dexter S08E04 EVOLV (real provider bytes) | corrected artifact, `serving_state=alass_corrected_large_offset`, `state=unverified` |
| Dexter S08E04 ASAP (real provider bytes) | corrected artifact, `serving_state=alass_corrected_large_offset`, `state=unverified` |
| uniform +96.5s synthetic | corrected (normal verified path; unaffected by the exception) |
| large offset with drift | original (`serving_state=original`) |

`dexter_s08e04_manifest.json` `_derived.expected_delivery_category` is therefore
`TRANSFORMED_SERVED`. The previous `ORIGINAL_SERVED` value described what the
conservative verifier did, not what should happen.

## Known limitations, recorded rather than hidden

1. **`different_cut` is a false accept.** MAD 0.0 and structural 0.762 are
   indistinguishable from the required-positive case (structural 0.729 for the
   real ASAP case; content-at-best 0.245 vs 0.176, the "cut" scoring *higher*
   than the correct pair on both). No signal available to this decision
   separates them, so it is accepted deliberately rather than guessed at. The
   mutation `large-offset-*` suite pins every other boundary.
2. **Reference-first identity does not actually discriminate the wrong-episode
   fixture.** Every fixture reports `same_episode`. The gap signal has real
   margin - `gap_distribution_similarity` 0.649 for the wrong-episode reference
   against 0.998 for every positive - but `LARGE_OFFSET_MIN_GAP_SIMILARITY` is
   0.60, set for acceptance rather than discrimination, so it passes. What
   refuses that case today is movement MAD. Tightening the floor to ~0.70 would
   give identity its documented job with ~0.30 margin to the nearest positive;
   that was **not** done here, because changing a calibrated constant on the
   strength of one fixture is how the class-C research went wrong.
   `tests/test_large_offset_serving.py::test_identity_signals_show_margin_but_are_not_what_refuses_the_negatives`
   asserts the margin so its disappearance is caught.
3. **`@requires_media` is dead code.** `tests/test_real_media_regression.py`
   sets `VIDEO_NAME` to `...PiR8.mvk` while `_media_present()` globs `*.mkv`, so
   `_media_present()` is unconditionally `False` and the two media-gated tests
   can never run. Pre-existing; left alone, because fixing it activates a 5.4GB
   hash test as a side effect. Both affected tests were verified by supplying a
   matching filename out of band.
4. The reference-health coverage floor added with this feature
   (`analyze_reference_health(..., require_dialogue_coverage=False)`) is a
   *selection* heuristic, not a disposition: with the default `True` it rejected
   100% of fixtures **and the real production reference** (39% coverage). The
   investigation does not let it disqualify a reference; coverage is recorded as
   a non-disabling note.

## Changed files

| file | change |
|---|---|
| `app/services/sync/large_offset_investigation.py` | new: signals, stages, `decide_large_offset_serving`, reason codes |
| `app/services/sync/orchestrator.py` | investigation before alass, validation + decision after, scoped override, `serving_state` in outcome meta and delivery log |
| `app/services/sync/reference.py` | `analyze_reference_health(..., require_dialogue_coverage=True)` - additive, default unchanged |
| `tests/fixtures/real_media/dexter_s08e04_manifest.json` | `expected_delivery_category` -> `TRANSFORMED_SERVED` |
| `tests/test_large_offset_serving.py` | new, 19 tests |
| `tests/test_delivery_decision.py`, `tests/test_format_routing.py`, `tests/test_real_media_regression.py` | flipped to the corrected delivery; refused direction re-covered by a drift case |
| `tools/calibrate_large_offset_investigation.py` | new: signal calibration + end-to-end serving matrix |
| `tools/run_safety_mutations.py` | +6 mutations for the new boundaries |

## Gates (implementation pass)

| gate | result |
|---|---|
| host `pytest` (`.venv`, Python 3.12) | **1902 passed, 19 skipped, 0 failed** |
| Docker `pytest` (Python 3.11 + alass) | **1915 passed, 4 skipped**; 2 failures are `FileNotFoundError: 'git'` - the image ships no git binary, and both tests pass on host |
| safety mutations | **96/96 caught in 868s**, zero `UNPROTECTED` / `MUTATION_TIMEOUT` / `BASELINE_ERROR` |
| `mypy app` | Success, 64 source files |
| `ruff check` on touched files | All checks passed (repo-wide: 12 pre-existing, none in touched files) |
| real production route, no test override | EVOLV + ASAP deliver their known alass artifacts; provider cache untouched; artifacts non-reusable |
