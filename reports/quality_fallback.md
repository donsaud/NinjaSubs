# Quality-aware Large-Offset reference fallback

Date: 2026-10-17
Scope: original Python NinjaSubs repository. No deployment, restart, commit or push. Production `ninjasubs` on port 7000 was not touched.

## Root cause

Mad Men S01E01 (`tt0804503:1:1`, HMAX WEB-DL WADU video) selected the cached
`Mad.Men.S01.720p.WEB-DL.AAC2.0.H264-BTN.srt` reference for both the DBO and SA89
Arabic subtitles. Both corrections had:

- weak target binding (`source_family_only`, WADU video vs BTN reference),
- `sync_state=unverified`, `verification=unknown`,
- verifier rejection `low_confidence`,
- residual p95 3957 ms (DBO) and 2546 ms (SA89),
- structural similarity 0.760 / 0.871,
- Large-Offset serving state `alass_corrected_large_offset`.

The orchestrator returned immediately after the first `serve_synchronized=True`
from the Large-Offset override. The planned pool was never advanced, so the
engine never checked whether another eligible reference could produce a better
result. Playback confirmed both delivered corrections were wrong.

The same path must preserve the confirmed Into the Wild (`binding=strong`,
p95 2511 ms) and Dexter (`binding=strong`, p95 4735 ms) corrections. Binding
strength, not residual p95 alone, separates the confirmed-good cases from the
Mad Men failures.

## Decision policy

A pure policy module, `app/services/sync/quality_fallback.py`, answers one
question: may a guarded corrected result be served immediately?

1. Only `alass_corrected_large_offset` is in scope. Everything else keeps its
   existing serving decision.
2. A result the normal serving policy already accepts is not deferred. This
   preserves normally verified results, including weak-binding results that
   earned independent verification.
3. A strong/exact binding guarded result is not deferred. This preserves Into
   the Wild and Dexter.
4. A weak/unknown/conflict binding guarded result is **provisional**. The loop
   may try additional eligible references. If none is better, the original
   subtitle is served instead of the known low-confidence correction.

The policy never verifies, promotes, transforms, or caches a success. The
existing verifier and Target Binding remain authoritative.

## Implementation

- `quality_fallback.py` returns `QualityFallbackAssessment(defer, reason,
  evidence)`, carrying binding, p95, MAD, structural similarity, verifier
  state and rejection reason for logs.
- In the ALASS path, the orchestrator computes `serve_without_override` and the
  quality assessment before writing caches.
- A deferred provisional:
  - does not get the Large-Offset serving override,
  - is not promoted to Sticky Reference,
  - is not written to the cross-subtitle content-addressed tier,
  - is not recorded as a successful result,
  - sets `rejected_candidates_seen=True` so the request is not negatively
    cached as "no reference".
- The loop continues to the next planned/prepared reference. Existing
  content dedup prevents re-evaluating identical reference bytes.
- Bounded fallback:
  - `QUALITY_FALLBACK_MAX_ATTEMPTS=2` extra ALASS candidates,
  - `QUALITY_FALLBACK_MAX_SECONDS=8.0` wall clock from the first provisional,
  - existing `ALASS_CANDIDATE_LIMIT`, `REFERENCE_POOL_LIMIT`, prefetch and
    request budgets still apply.
- A later strong-binding or normally verified candidate serves normally. If
  none does, the original subtitle is served.
- Telemetry: `quality_fallback_triggered`, `quality_fallback_attempts`,
  `quality_fallback_served`, `quality_fallback_exhausted`,
  `quality_fallback_first_p95_ms`, `quality_fallback_first_binding`.

## Regression evidence

### Real Mad Men measured evidence

Reproduced on the cached target/reference bytes with real ALASS:

| Case | Reference | Binding | p95 | MAD | Structural | Verifier |
|---|---|---|---:|---:|---:|---|
| DBO | BTN | weak | 4045 ms | 635 ms | 0.760 | unverified / low_confidence |
| SA89 | BTN | weak | 2684 ms | 616 ms | 0.870 | unverified / low_confidence |
| DBO | SA89 BluRay | unknown | 4684 ms | 0 | 0.724 | unverified / low_confidence |
| SA89 | SA89 BluRay | unknown | 0 ms | 0 | 1.0 | verified but binding unknown (not servable) |

### Mad Men before/after benchmark (3 runs, real ALASS)

`tools/benchmark_mad_men_quality.py`, two candidates (BTN then SA89 reference):

| Case | Before p50 | After p50 | Downloads | ALASS | Before served | After served | Quality fallback |
|---|---:|---:|---:|---:|---|---|---|
| DBO | 2,694.0 ms | 5,970.2 ms | 1 -> 2 | 1 -> 2 | wrong `c4e6333c2a7b41a5` | **original `3ebf6971f3305f61`** | triggered, exhausted |
| SA89 | 2,043.5 ms | 3,107.5 ms | 1 -> 2 | 1 -> 1 | wrong `0b80bfad2be9a7d3` | **original `b5d8948f7e0886af`** | triggered, exhausted |

The before hashes exactly match the previously delivered Mad Men artifacts.
After the change, both known-bad corrections are withheld and the original is
served. No weak content artifact was written (`content_artifacts=0` for both).

### Preserved confirmed cases

- **Into the Wild**: strong binding, p95 2511 ms. Post-change first request:
  9.84 s, 2 downloads, 1 ALASS, served `3aa5134cfa9a1f9e`; second identical
  subtitle 27.2 ms / 0 downloads / 0 ALASS via the content cache. Quality
  fallback not triggered.
- **Dexter S08E01**: measured strong release-group binding (`d3g`), p95 4735 ms,
  MAD 0, structural 0.745; policy allows the guarded correction.
- **Penny Dreadful S01E01**: verified Two-Point path is untouched; existing
  tests pass.
- **Compatible same-release subtitles**: Metadata Gate still returns
  `fast_original`; no sync, no fallback.
- **Cross-subtitle reuse**: 17 tests still pass; strong-binding content tier
  remains reusable.

## Tests

New `tests/test_quality_fallback.py` - **11 tests**:

- Mad Men DBO weak/low-confidence is deferred.
- Mad Men SA89 weak/low-confidence is deferred.
- Unknown/conflicting binding is deferred.
- Into the Wild strong binding is still served.
- Dexter strong binding (p95 4735) is still served.
- Normally verified weak binding is not deferred.
- Non-Large-Offset states are untouched.
- Fallback budget defaults are bounded.

Focused suite (quality, cross-subtitle, sticky, large-offset, Two-Point,
Dexter/Penny, prefetch): **175 passed, 9 skipped**.

Full suite: **11 failed, 4,871 passed, 40 skipped, 6 errors**. The 11 failures
and 6 errors are the exact same pre-existing node IDs as the previous clean
comparison; zero new failures. The +11 passes are the new quality tests.

`ruff` and `mypy` clean on modified files.

## Risks and trade-offs

- Weak-binding guarded corrections that might occasionally have been correct
  now fall back to the original. This is intentional: a known low-confidence
  correction was demonstrably wrong in the Mad Men case.
- The fallback can add up to 2 extra ALASS runs / 8 seconds. Accuracy is
  prioritized over latency, and the existing request budget still bounds it.
- Weak provisional results are not cached as successes, so a later request may
  re-evaluate the same reference against different subtitle content. Within a
  request, identical reference bytes are deduplicated and the attempt/time
  budget is enforced.
- Strong-binding guarded successes remain unchanged; this is the explicit
  guard for Into the Wild and Dexter.
- No deployment was performed.

## Files

- `app/config.py`
- `app/services/sync/orchestrator.py`
- `app/services/sync/quality_fallback.py` (new)
- `app/services/sync/telemetry.py`
- `tests/test_quality_fallback.py` (new)
- `tools/benchmark_mad_men_quality.py` (new)

## Approval requested

Please review the policy, implementation, regression evidence and benchmarks.
On approval, the changes can be deployed with the existing staged rollback
procedure; no production container was modified in this task.
