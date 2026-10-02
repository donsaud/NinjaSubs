# Synchronization architecture: frozen

The algorithm is frozen as of this document. Nothing below is a proposal. It
is a record of what the system does, what it is good at, and — stated plainly
— where it is known to be wrong.

## Path classification

```
PRODUCTION        request path: matching, reference selection, alass,
                  verification, caching, serving
EXPERIMENTAL      adaptive audio activity, shadow reference selection v2
                  (observation only; never substituted into a response)
BENCHMARK ONLY    golden corpus, coverage matrix, generated-media timeline
                  witness, real-media runner, safety mutation matrix
SHADOW ONLY       audit records; written on the request path, read by nobody
```

Nothing under EXPERIMENTAL, BENCHMARK ONLY or SHADOW ONLY can change what a user
receives. The video-derived timeline witness in particular is an offline
measurement tool: production receives no target video, so it never runs on the
request path.

## Cache identity and lifecycle

**Target identity is constructed in one place.** `video_fingerprint_from_meta`
hashes `imdb_id, season, episode, target_filename, video_hash, video_size`.
Because it hashes whatever it is given, two callers that build the dict
differently derive different identities for the same target. This was not
theoretical: the pre-download reuse lookup used a four-field dict while the
serve path wrote verdicts from the full request meta, so on any request
carrying a `videosize` -- every normal Stremio request -- the reuse path could
never find the verdict it was written to reuse. Both paths now share
`_target_fingerprint_meta`. `tests/test_target_identity_audit.py` pins them
against each other and asserts every field participates in the hash.

**Fingerprint strength is labelled, never overstated.**

| `fingerprint_source` | meaning |
|---|---|
| `video_hash` | a real content hash of the video: true identity |
| `stream_context` | digest of the stream URL: identifies a stream, not bytes |
| `target_filename` | a reported release name |
| `target_id` | the **subtitle** id; not video identity, and named as such |

The weak fallbacks are retained deliberately. Refusing hashless requests would
reject ordinary playback, which is the common case.

**`SyncCache` is process-local.** With `REDIS_URL` empty, `_local` is a
`cachetools.TTLCache` inside the process, so a restart is a guaranteed miss for
a byte-identical key. That is architecture, not a defect, and the
`[sync-cache]` diagnostic carries `store_layer` plus `process=` so a trace can
tell "same process, key drifted" from "new process, store empty".

**Payload reuse is target-bound.** `find_synced_for_target` requires the key's
own fingerprint segment to match. `find_synced_for` is deliberately not used
here: it matches on the subtitle id alone and will return an artifact
synchronized against a different video.

**Reference cache identity** is
`{imdb}_{season}_{episode}_{group}_v2_{digest}_{scope}_{source}_{kind}.srt`,
with two independent variation axes. `digest` covers media type, target
filename, video hash, size and languages. `scope` is the target subtitle's
cue-layout fingerprint, so a different target timing layout is a different
cache identity **by design**: a reference proven against one layout is not
reused for another. `stream_url` is excluded from the stem precisely because a
rotating signed token would otherwise defeat the cache on every request.

**Credentials** are supplied either by the environment or per-request through
the Stremio manifest/config, and `.env` is legitimately empty in the second
mode. Precedence is unchanged: `config or environment or ""`. Provenance is
recorded as `credential_source` where that choice is made, because after the
values are merged their origin is unrecoverable. Rotating a credential changes
the cache namespace; that isolation is intentional and `auth_digest` is
retained for it.

**Fail-closed.** A verdict that is not `VERIFIED_SYNCED` or
`VERIFIED_RESYNCED` never yields a served artifact, even when a matching
artifact exists for the same target and subtitle. A verified verdict whose
artifact is gone is not a payload: it falls through to the normal download.

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

## Release gates

One command runs every offline layer without network or provider credentials:

```bash
python tools/run_sync_benchmarks.py                # pytest + coverage + golden
python tools/run_sync_benchmarks.py --mutations    # adds the safety mutation matrix
python tools/run_sync_benchmarks.py --json-out report.json
```

It fails the gate on a regression, on a new false-verified golden case, on a
target-binding or negative-serving regression, and on an UNPROTECTED safety
mutation. It does **not** fail merely because a difficult case abstains
correctly, because a provider is unavailable offline, or because ground truth
is unknown. Those are distinct states and are reported as such.

`tools/run_safety_mutations.py` removes each safety guard in turn in a
temporary copy of the tree and requires the suite to fail. A guard whose
removal leaves the suite green is reported as UNPROTECTED, which is a release
blocker. Currently 6/6 mandatory mutations are caught: target-fingerprint
binding, negative-verdict serving, stale engine version, divergent target
identity, the strict timestamp grammar, and reference-cache target binding.

## Duplication audit (reliability, not style)

Recorded rather than refactored. Only duplication with *proven* divergence risk
and full test coverage on both paths is a candidate for consolidation; the rest
is documented so a future reader knows it was considered.

| Area | Sites | Risk | Decision |
|---|---|---|---|
| SRT timestamp parsing | `parse_srt_cues._to_ms`, `sync_service._parse_timestamp_ms`, `query._cue_timestamp_ms` | **real** -- this exact divergence shipped a bug where the sync path counted 584 cues and the verifier saw 0 | **kept separate, invariant tested.** `test_parse_srt_cues_agrees_with_the_sync_path_parser` asserts the two agree on both grammars. |
| ASS / cleaners / SAMI conversion | `ass_converter`, `cleaners`, `sami_converter` | low: different formats and call paths | left alone |
| Credential precedence | `config_parser.parse_user_config` is canonical; `subdl`, `subsource`, `opensubtitles`, `aggregator` each repeat `(api_key or settings.X or "")` | **real but currently consistent** -- the expressions are identical, so no divergence is demonstrated | left alone; consolidating would touch six provider files with no proven defect |
| Credential override detection | `main.py` `prefs.X if prefs.X != settings.X else ""` | none: different intent | not a duplicate, deliberately distinct |
| Fingerprint construction | `_normalize_fingerprint`, `video_fingerprint_from_meta`, `_target_fingerprint_meta`, `_fingerprint_source` | **was real** -- two callers supplied different field subsets, which made cache reuse unreachable in production | **normalised** through `_target_fingerprint_meta`; pinned by `tests/test_target_identity_audit.py` |
| Verification gate | one site: `_reusable_verified_fallback_sync` | none | single source of truth, mutation-proven |

## Measured coverage, and what it does not show

`python tools/benchmark_coverage_matrix.py` reports the corpus axes that exist
and, more usefully, the ones that do not, measured from the corpus labels and
the fixture bytes rather than from case names. Currently **19 required axis
values have no coverage**, including every FPS family, 480p/576p/2160p, BDRip,
DVD, movies, season packs, piecewise drift, and `MM:SS,mmm`.

That last one matters: the `MM:SS,mmm` parser is unit-tested and the mutation
suite proves the test bites, but **no golden case exercises it**. The corpus
predates that fix. It is a fixture gap, not a code gap.

## What the evidence does and does not support

Every accuracy number in this repository is synthetic. The golden corpus
contains 20 cases, the real-media runner reports NOT AVAILABLE, and the
ffmpeg-dependent layers are unavailable in CI. The honest claim is therefore
behavioural -- the system accepts correct synchronizations, rejects dangerous
ones, and abstains when ambiguous -- measured on constructed inputs.

It is **not** a claim of real-world accuracy across a catalogue. `false_verified`
is tracked separately from aggregate accuracy because a false VERIFIED is far
more dangerous than a correct UNVERIFIED. The independent video witness blocks
every false-verified case the golden corpus produces, but that blocking is an
offline measurement and does not exist on the request path, where production
receives no target video.

The four false-verified cases are classified, not treated as an algorithm
defect: two are reference circularity (providers agree and are jointly wrong,
which no threshold can fix without independent evidence) and two are
different-cut / different-edit fixtures. The threshold sweep in the evaluator
shows that refusing everything would drive `false_verified` to zero at the cost
of all coverage, which is why the evaluator presents a Pareto view and refuses
to recommend a threshold.

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

## Large Offset Evidence Gate

The 20s first-dialogue window is unchanged and still the normal fast path. What
changed is what happens when a reference misses it.

### The real-media case

`Dexter.s8e04.Scar.tissue.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv` opens its first
dialogue at 107.5s; a compatible BluRay reference opens at 10.7s. That ~96.6s
displacement was read as "different cut" and refused before alass ever ran.
Manual inspection confirmed the dialogue and recap content is the same episode
and the target simply sits later in time. Measured on the real pair: 560 target
cues against 562 reference cues, and five independent regions across the episode
recovering +96.56s / +96.12s / +95.72s / +96.26s / +97.28s — a constant
displacement, not a mismatch.

### What the gate does

A reference beyond the 20s window becomes a **Large Offset Candidate** and has
to earn an alass attempt from `app/services/sync/large_offset.py`. Required
evidence:

| signal | requirement |
| --- | --- |
| identity | supported by the existing selector (precondition, never proof) |
| anchors | ≥ 4 of 5 equal-duration regions agree, each cue pairing within 2500ms |
| dispersion | MAD across regional medians ≤ 1200ms |
| drift | \|slope\| ≤ 120 ms/min, reused from `analyze_drift` |
| structure | cosine similarity of cue-event profiles, target vs reference shifted by the estimated offset, ≥ 0.55 |
| ceiling | `LARGE_OFFSET_MAX_SECONDS = 180` |

`LARGE_OFFSET_MAX_SECONDS` is a starting point, not a proven value.
`tools/run_sync_benchmarks.py` reports observed offsets so it can be retuned
against evidence. All signals read *timings only* — no text, no language
assumption, no ffmpeg, no audio fingerprinting — so the path is equally valid
for an Arabic target against an English reference.

### What it deliberately does not do

The gate grants permission to attempt alignment. It never marks anything
verified, and it has no field that could. The verdict remains the post-alass
analyzer's.

There is a measured limit worth stating plainly: **subtitle timing alone cannot
reliably tell a same-episode constant shift from a different episode.** Measured
on the real S08E04 target against a real S08E05 reference, the 30s-bin density
cosine is 0.853 for the *wrong* episode versus 0.842 for the correct one; gap
sequence correlation is 0.805 versus 0.870. Dialogue rhythm is statistically
generic across episodes. The gate therefore narrows — it refuses over-ceiling
offsets, offsets with no dialogue seed, offsets with too few corroborating
regions, and offsets whose regions visibly disagree — but a wrong-episode
reference that looks structurally plausible may still receive an alass trial.
The information that settles it is in the audio, which is what alass aligns, so
the accept/reject decision stays post-alass where the residual, p95, MAD, drift
and coverage gates all still apply.

That is why the magnitude ceiling is widened only for a reference the gate has
vetted (`max_plausible_offset_ms`, default still `MAX_PLAUSIBLE_OFFSET_MS`
= 20s) and why the coarse offset is handed to the verifier
(`movement_offset_ms`, default 0) so a 96s correction can be paired and measured
at all — widening the *pairing radius* instead would make greedy
nearest-neighbour pair each cue with its temporally adjacent neighbour and
manufacture a wildly varying "movement".

