# Synchronization architecture: frozen

The algorithm is frozen as of this document. Nothing below is a proposal. It
is a record of what the system does, what it is good at, and — stated plainly
— where it is known to be wrong.

## Reference selection is target-bound

The one exception to "frozen", promoted because a production incident proved the
old shape wrong.

```
TARGET VIDEO -> reference -> candidate/reference comparison -> alass
```

The requested subtitle is **not** permitted to choose the reference. It decides
only whether a synchronization happens.

The previous resolver ranked candidates, downloaded the first, and asked a
validator that closed over the requested subtitle whether it passed a
first-dialogue gate; a failure was read as "this reference is wrong", and the
loop continued until some reference happened to fit. For target
`Dexter.s8e05.This.little.piggy.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv` with a
candidate first dialoguing at ~106.95s, that meant downloading six English
candidates at ~10.16s, rejecting all six with the same ~+97s delta, and
aborting. The ±20s gate was correct throughout; the *decision* was inverted.

Selection now runs in two steps that are not mixed:

- **Step A, target reference.** Scored by `MatchTier` and
  `hard_compatibility_filter` against the target only, ordered by
  `ReferenceHealth`/`ReferenceTrust`, deduplicated by release family, bounded
  by `REFERENCE_POOL_LIMIT`. Implemented in
  `app/services/sync/reference_v2.py::decide_target_bound_reference`, which
  takes no candidate argument at all.
- **Step B, candidate synchronization.** The preflight comparison, the ±20s
  cue-sanity gate, alass, `AlignmentAnalyzer`, verification. Unchanged.

The anti-circularity rule: if any target-accepted reference fails against the
candidate, only references in a **stronger or equal identity band**
(`tier_band`, reusing the existing HASH/EXACT/SOURCE_FAMILY/weak scale) may
still be selected. A weaker reference that merely fits the candidate is
suppressed and the request fails closed with `CANDIDATE_DIFFERENT_TIMELINE`.

Two deliberate limits, both measured rather than assumed:

- **Reference loss was observed and rejected.** Promoting the shadow's
  `health_class == "unusable"` to a hard production veto refused references the
  previous policy accepted. Health therefore orders equally-strong candidates
  and is reported, but does not by itself cancel a reference. A hard target
  mismatch still cancels one absolutely.
- **The tier floor compares bands, not tiers.** Two references differing only
  by an arbitrary release group inside `SOURCE_FAMILY` are metadata noise, not
  evidence; letting that difference refuse a sync would be a new bug.

`NO_TARGET_BOUND_REFERENCE` is a valid, expected outcome. The system fails
closed rather than anchoring to a reference the target does not support.

## Production trust boundary

These distinctions are the architecture. Everything else is machinery around
them.

```
No evidence          is not  negative evidence
Internal agreement   is not  video correctness
Alass exit 0         is not  verified synchronization
Metadata prediction  is not  verified synchronization
```

When the required evidence is unavailable, the system **abstains**. Abstention
is a real outcome with its own reason codes, not a failure to be papered over,
and it is never silently converted into either a pass or a rejection.

## SyncState semantics

| state | meaning | who may create it |
|---|---|---|
| `VERIFIED_SYNCED` | The subtitle was already aligned with the target and that was measured. | `set_verdict`, with `verification` of `VERIFIED` or `CACHED`. |
| `VERIFIED_RESYNCED` | It was misaligned, alass re-timed it, and the result was then measured. | `set_verdict`, with `verification` of `VERIFIED` or `CACHED`. |
| `PROBABLE_SYNC` | Suggestive, not measured. Also the ceiling a *predicted* claim can reach. | The guard, when `verification` is `PREDICTED`. |
| `UNVERIFIED` | Not enough evidence for any claim. | The default, and the guard's fallback. |
| `REJECTED` | Evidence says this should not be served as synchronized. | `set_verdict`, from a real measurement (cue loss, proven mismatch). |

**One authoritative mechanism.** `VERIFIED_*` requires `verification` in
`{VERIFIED, CACHED}`, and that is enforced structurally by a field validator
on `SubtitleEvaluation.sync_state` with `validate_assignment=True`. It is
therefore impossible to reach a verified state by:

- constructing the model with a bare string literal, or
- assigning `evaluation.sync_state = SyncState.VERIFIED_SYNCED` directly.

Both of those were possible before this guard and are pinned by tests. A
legitimate verdict sets `verification` first and the state second; the ordering
is load-bearing and documented at the call site.

## Evidence availability

| availability | meaning | can support `VERIFIED_*` |
|---|---|---|
| `VERIFIED` | Measured now, in this request. | yes |
| `CACHED` | Recalled for an identical video + subtitle + engine version. | yes |
| `PREDICTED` | Inferred from metadata, before alignment. | **no** |
| `UNKNOWN` | Nothing established. | **no** |

`PREDICTED → never VERIFIED` and `UNKNOWN → never VERIFIED` are invariants
enforced by the guard rather than by convention. A cached verdict is not a
weaker *kind* of evidence: it is a measurement performed earlier under an
identity the cache key pins exactly.

## Target fingerprint invariant

```
target_filename = the actual target video's name, OR None
```

Never a subtitle filename, a subtitle release name, an archive filename or a
provider release name. `None` is meaningful and must stay `None`: with no
release-level evidence the system is conservative rather than
confidently wrong. The boundary where this is established is annotated in
`app/main.py`.

## Production and experimental paths

**Production**, unchanged by anything in the experimental set:

```
content hard gates → candidate ranking → verdict cache
  → Top-N serve-time alass → existing verification → SyncState
```

**Experimental**, measurement only, never consulted by the path above:

```
search-time predictor (shadow)      reference selector v2 (shadow)
shadow pool + opt-in expansion      adaptive audio detector
video-derived timeline validation   audit trail
calibrate_sync / golden / real-media / cut-detection benchmarks
```

Experimental components observe production and never write back into it. The
shadow selectors compute an alternative decision and record it; the
`ResolvedReference` the synchronizer receives is the legacy one, and a
behavioral test with a mutation check proves it.

### Experimental switches

Four settings, all off or inert by default, and deliberately **not** abstracted
behind one flag: they answer different questions and collapsing them would make
it impossible to enable measurement without enabling cost.

| setting | default | effect |
|---|---|---|
| `SYNC_AUDIT_ENABLED` | `False` | Records decisions. Required for the shadow pool to fetch at all. |
| `REFERENCE_SHADOW_POOL_FETCH_LIMIT` | `0` | Extra downloads for the shadow pool. Ignored unless the audit is recording. |
| `REFERENCE_SHADOW_POOL_LIMIT` | `4` | Size of the shadow pool. Purely observational at the default. |
| `ADAPTIVE_AUDIO_ENABLED` | `False` | Per-file adaptive landmarks. Benchmark and shadow only. |

Production defaults do not change.

## Known strengths

- Hard content gates that reject a subtitle whose release cannot match.
- Fingerprint-aware matching where the request supplies a hash.
- Conservative alass verification with a strict timeout and bounded Top-N.
- Residual analysis with MAD, p95 and drift, plus change-point detection.
- Structural and cut analysis that treats provider cue re-splitting as
  legitimate rather than as mismatch.
- Reference health scoring and duplicate timing-grid detection.
- Consensus evidence across independent sources, used to *withhold* rather
  than to grant.
- Search-time prediction, cached separately from verified verdicts.
- Exact verdict caching keyed on video identity + subtitle identity + language
  + engine version.
- Video-derived timeline validation where media is available.
- Deterministic ordering: same inputs, same output, no dependence on
  async completion order, provider response order, or filesystem enumeration.
- Audit and offline calibration tooling with no production write path.

## Known limitations

These are not caveats to be minimised. They are the honest state.

1. **Stremio may supply no video fingerprint at all.** Then there is no
   release-level evidence and matching is conservative by necessity.
2. **The addon does not receive the video file** on the normal request path.
   `stream_url` is passed through as an opaque string and never fetched.
3. **Video-derived validation is therefore unavailable in normal production
   requests.** The layer that would catch a wrong cut cannot run where it
   matters most. Making it run requires solving media availability first, and
   that is a product decision, not an algorithm change.
4. **Reference/candidate verification can still be internally consistent for a
   wrong cut.** Two artifacts sharing one incorrect timing model produce a
   perfect residual, total consensus and a clean verdict. This is a closed
   loop and no amount of analysis inside it can see it.
5. **Piecewise-displacement wrong cuts are an open experimental problem.** The
   dominant remaining failure class in the synthetic benchmark is
   `WRONG_CUT_MISSED`, and the missed cases are exactly this shape:
   `segment A → shift A, segment B → shift B`. Not attempted in this phase.
6. **No real programme media has been measured.** Every number in this
   repository is synthetic, and synthetic evidence describes behaviour, not
   real-world accuracy.
7. **A benchmark result is not a claim of correctness.** The synthetic
   benchmark's most useful output has been findings that contradicted
   expectations, which is a reason to trust it as an instrument and not as a
   scoreboard.

## Open research item: piecewise displacement

The unresolved failure, stated precisely:

```
WRONG_CUT_MISSED
    video:    ──── shift A ────
    subtitle: ──── shift A ──── shift B ────
                           ^ a discontinuity no single shift explains
```

A global offset and a different cut both produce a best shift. What separates
them is whether that shift holds across the whole timeline, which is what
regional correlation measures. It currently catches spliced edits and misses
smaller piecewise displacements.

When real media becomes available, collect specifically:

- inserted scene
- deleted scene
- recap
- TV edit
- different intro/outro
- extended edition
- different credits structure

and first ask **whether regional correlation already solves them**. Only a
repeatable blind spot demonstrated on real material justifies new machinery.

## VAD: deferred

No measured case requires semantic or audio understanding. The failures that
exist are landmark-extraction and comparison failures, which is a different
problem with a different solution space.

```
VAD = DEFERRED
```

Adding it because it *might* help would be complexity traded for uncertainty,
and uncertainty is not reduced that way.

## The change gate from here on

No synchronization algorithm change may be made on synthetic evidence alone. A
future change requires all five:

1. A real-media failure category.
2. Reproducible evidence.
3. A concrete explanation of why the current system cannot resolve it.
4. A regression fixture reproducing the property.
5. A measured trade-off against false positives and false negatives.

Synthetic tests may reproduce and prevent regressions. They may not
independently justify a new production heuristic.
