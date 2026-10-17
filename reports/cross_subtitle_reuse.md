# Cross-subtitle Sticky Reference and guarded artifact reuse

Date: 2026-10-17
Scope: original Python NinjaSubs repository. The reviewed changes were deployed to `ninjasubs` on port 7000; no other container was touched, and nothing was committed or pushed.

## Root cause

The production Into the Wild sequence used two subtitle IDs with byte-identical provider bytes
(`d8efd4b030f64a1d`, then `92788b7594b7f497`). The second request repeated discovery, two downloads
and ALASS despite the first request having already delivered `3aa5134cfa9a1f9e`.

Four independent causes:

1. **Legacy promotion gate.** `_promote_sticky_reference` returned immediately whenever Unified
   authority was OFF. Production is Unified OFF, so a successfully delivered guarded Large-Offset
   result was never written to the sticky tier.
2. **Movie member 0 vs None.** Movie proofs record `member_season=0` / `member_episode=0`; movie
   requests carry `None`. `get_video_scoped` and `get_sticky_reference` compared them strictly, so old
   movie entries never hit. This is why the production second request re-downloaded references.
3. **Per-subtitle payload key.** The ordinary resolution key includes the subtitle ID. A different ID
   with identical bytes is a different key, so the corrected payload was not found.
4. **Unknown verdicts are not cached.** `SyncCache.set_verdict` refuses a verdict whose
   `verification` is `unknown`. The guarded Large-Offset result is exactly `unverified/unknown`, so
   there was no content-addressed verdict entry for a second subtitle to reuse.

## Implementation

### Movie member normalization (`app/services/sync/cache.py`)
- New `_member_matches(stored, requested, movie=...)`: movie `None` and legacy `0` describe the same
  member; series keep exact comparison (season 0 specials are not season None).
- `set_video_scoped` / `set_sticky_reference` normalize movie member values to `None` on write.
- `get_video_scoped` / `get_sticky_reference` accept legacy `0` for movies.
- Sticky get/set accept an explicit `methodology`; legacy pointers use
  `("legacy", "large-offset-v1")` and Unified pointers keep `("unified", "timing-v3")`.

### Legacy guarded Sticky promotion (`app/services/sync/orchestrator.py`)
- `_promote_sticky_reference` has two separate regimes:
  - Unified: only `verified*` evaluations, unchanged.
  - Legacy: only a delivered `alass_corrected_large_offset` result with `UNVERIFIED` state and
    `exact`/`strong` binding.
- A plain UNVERIFIED/REJECTED candidate still never promotes. The sticky pointer is a first-attempt
  reference hint; every later request still runs the authoritative verifier.
- The ALASS serve path passes `serving_state` to promotion.

### Cross-subtitle guarded artifact tier
- The corrected artifact is also stored under a content-addressed key in the existing `SyncCache`:
  `final_sub:...:<video fingerprint>:content-<lang>-<size>-<auth>-<strict>:large_offset_content_v1:<original content hash>`.
  It excludes the subtitle ID and binds video, byte-identical original content and language.
- Provenance stored beside it: `serving_state=alass_corrected_large_offset`, guard generation,
  `sync_state=unverified`, `verification=unknown`, `target_binding`, full `reference_sha`,
  `artifact_sha256`, `original_content_hash`.
- On a new request, after decoding the original but before verdict lookup/discovery, the content key
  is loaded and all guard fields are re-validated. A hit is served as `UNVERIFIED/UNKNOWN` with
  strategy `large_offset_content_cache`.
- A same-subtitle guarded payload hit also publishes the content tier, so a later different ID can
  reuse it even if this process never re-ran ALASS.
- Invalid/missing/tampered content entries fall through to the normal ReferencePolicy and Progressive
  Prefetch pipeline. They are not treated as negative-cache rejections.
- Different original bytes, video release, language, credentials or strict-mode context get different
  keys and are independently verified.

### Metadata Gate and serving-policy guards
- Before a content hit is served, the current request runs `decide_metadata_gate`. Content reuse is
  allowed only when the gate says `INVESTIGATE` (or the request is explicitly forced). `FAST_ORIGINAL`
  and `IDENTITY_CONFLICT` keep their normal serve-original behavior; the cache cannot bypass the gate.
- The content tier is legacy-only and is excluded under Unified authority, matching the Large-Offset
  serving fence.
- Every replay re-checks the guard generation, unverified state, strong/exact binding, full reference
  digest and exact artifact digest.

## Safety review requested before deployment

### 1. Content key isolation
- Key binds `imdb_id`, movie/series season-episode identity, video hash or normalized target filename,
  video size, language, original content hash, credential digest and strict-mode context.
- It excludes only the subtitle ID, which is the intended cross-subtitle dimension for byte-identical
  content.
- Metadata Gate is checked at read time; `FAST_ORIGINAL`/`IDENTITY_CONFLICT` suppress the content hit.
- Large-Offset authorization requires the recorded corrected state, guard version, UNVERIFIED/UNKNOWN
  verdict, strong/exact binding, full reference SHA and exact artifact SHA.
- Different video, size, language, content hash, credentials or strict context cannot share an entry.

### 2. Legacy sticky is a hint, never evidence
- Legacy promotion requires a delivered `alass_corrected_large_offset` result, `UNVERIFIED` state and
  strong/exact binding.
- The pointer records `verification_method=legacy`, `verification_version=large-offset-v1` and
  evidence `serving_state=alass_corrected_large_offset`, `sync_state=unverified`,
  `verification=unknown`.
- It is never converted into a `SubtitleEvaluation` or a VERIFIED claim. The reference is yielded as a
  first attempt and the authoritative verifier still decides.
- Under Unified authority, `get_sticky_reference` is called with the Unified methodology and cannot
  read the legacy pointer. Tests pin this isolation.
- A plain unverified result without the corrected serving state never promotes.

### 3. Series episode matching
- `_member_matches` uses movie semantics only when both requested season and episode are `None`.
- Series comparisons remain exact: season 0 specials do not match season `None`; episode 0 does not
  match a missing episode.
- Writes normalize to `None` only for the movie side (`query.season is None` / `query.episode is None`);
  series values, including 0, are preserved.
- Tests cover movie `0`/`None`, series `0` vs `None`, and numeric-string normalization.

## Benchmark (5 runs, production settings)

`tools/benchmark_cross_subtitle_reuse.py` runs the real orchestrator, verifier and ALASS with mocked
providers carrying the exact Into the Wild reference bytes. Production settings
(`SYNC_REQUIRE_EXACT_MATCH=false`, `ALASS_CANDIDATE_LIMIT=8`, `SYNC_TOTAL_REQUEST_BUDGET=14`), 5 runs.
Calls:

1. First subtitle `d8efd4b030f64a1d`, normal discovery/download/ALASS.
2. Second subtitle `92788b7594b7f497`, byte-identical original content.
3. Third subtitle, different original content, for sticky/independent-validation coverage.

| Call | Before p50 | After p50 | Before downloads | After downloads | Before ALASS | After ALASS | Strategy after |
|---|---:|---:|---:|---:|---:|---:|---|
| First (cold) | 9,466.5 ms | 9,782.5 ms | 2 | 2 | 1 | 1 | `alass` |
| Second (identical bytes) | 9,630.0 ms | **27.2 ms** | 2 | **0** | 1 | **0** | `large_offset_content_cache` |
| Third (different bytes) | 9,686.5 ms | **5,992.9 ms** | 2 | **0** | 1 | 1 | `alass` |

Total discovery calls across the three requests: **3 before -> 1 after**. All three calls served the
same confirmed artifact prefix `3aa5134cfa9a1f9e`; the third call still executed ALASS because its
original content differs, proving independent validation is preserved.

The second call's 27.2 ms is total benchmark latency (internal ~16 ms), with 0 provider downloads,
0 ALASS executions and a cache hit.

## Regression tests

New `tests/test_cross_subtitle_reuse.py` - **17 tests**:
- Content key ignores subtitle ID but binds video, size, language, credentials and original bytes.
- Cross-subtitle reuse of a guarded artifact (0 planning, 0 ALASS).
- Same-subtitle payload hit publishes the content tier.
- Unknown verdict is not stored, but the content artifact is reusable.
- Metadata Gate suppression (`IDENTITY_CONFLICT`) prevents content reuse.
- Different original content does not reuse.
- Missing/tampered artifact, missing authorization marker, weak binding all fall back to planning.
- Video and language isolation.
- Legacy movie `member_season=0` video-cache lookup.
- Movie normalization does not weaken series exact matching.
- Legacy guarded sticky promotion, movie normalization, cross-ID lookup, Unified-mode isolation,
  different-video isolation.
- Legacy unverified result without corrected serving never promotes.
- First-attempt sticky precedes discovery for a new subtitle ID.
- Real Into the Wild two-ID test (skips only when the private fixture is absent).

Existing sticky/video-cache/large-offset/Two-Point/Dexter/Penny-focused suite:
**212 passed, 9 skipped**.

Full suite: **11 failed, 4,860 passed, 40 skipped, 6 errors**. The 11 failures + 6 errors are the exact
same pre-existing node IDs as the previous clean comparison; zero new failures. The extra 17 passes
are the new tests.

`ruff` and `mypy` are clean on the modified files.

## Production deployment (2026-10-17)

| Item | Value |
|---|---|
| New image | `sha256:7b9f455d626555beab06937d4e0fd202958eee2f5f91d6228b2d667edc162e17` |
| New container | `ninjasubs` - `6a9980de8168ee4f200180be1992f0f196ae4ad12927c8414b230c5021a25626` |
| Previous image | `sha256:c8c728cf8741435bc2b982eadb54563b797e135d8018485f2844fe61abff1af5` |
| Previous image rollback tag | `ninjasubs-ninjasubs:pre-crosssub-20261017` |
| Rollback container | `ninjasubs-pre-crosssub-rollback` (staged stopped, same env/mounts/port/restart) |
| Port | `0.0.0.0:7000->7000/tcp`, `[::]:7000->7000/tcp` |
| Restart policy | `unless-stopped` |
| Mounts | `subs_cache:/app/cache` rw; `/test-fixtures` ro |
| Network | `ninjasubs_default` |

Only `ninjasubs` was rebuilt/recreated. All other running container IDs were unchanged.

Effective settings after deployment: `ENABLE_SUBTITLE_SYNC=true`,
`SYNC_REQUIRE_EXACT_MATCH=false`, `ALASS_CANDIDATE_LIMIT=8`, `ALASS_MAX_CONCURRENT_SYNCS=1`,
`SYNC_TOTAL_REQUEST_BUDGET=14.0`, prefetch enabled (2/2), preflight enabled, Unified authority OFF,
`REDIS_URL` empty. Provider credentials were unchanged and not exposed.

### Real production two-subtitle measurements

Server-wide provider keys are empty; the exact cached English FSiHD reference was used. The cached
reference sidecar score was temporarily raised so the first request selected the confirmed English
reference instead of the higher-scored Arabic one; the sidecar was restored immediately afterwards.
No code or acceptance threshold was changed. The deployed code then promoted a legitimate legacy,
unverified sticky pointer to the English artifact.

| Measurement | ID1 `d8efd4b030f64a1d` cold | ID2 `92788b7594b7f497` warm |
|---|---:|---:|
| Total HTTP delivery | **6.793 s** | **0.130 s** |
| Internal synchronization | **6.526 s** | **0.001154 s** |
| Reference downloads | **0** (video-cache hit) | **0** |
| ALASS executions | **1** | **0** |
| Cache state | payload miss; video reference hit; sticky promoted | guarded content cache hit |
| Strategy | `alass` (`#video_cache`) | `large_offset_content_cache` |
| Delivered SHA-256 prefix | **`3aa5134cfa9a1f9e`** | **`3aa5134cfa9a1f9e`** |
| Verdict | `unverified` / `unknown` | `unverified` / `unknown` |
| Serving state | `alass_corrected_large_offset` | `alass_corrected_large_offset` |
| Promoted to VERIFIED | no | no |

The ID1 cold request had zero provider downloads because the exact English reference was already in
the video-scoped cache (now readable for movies after the member normalization). It still executed
affine, pre-shift, ALASS and verification. The ID2 request found the content-addressed guarded
artifact and performed no discovery or ALASS.

The temporary sticky pointer records:
`verification_method=legacy`, `verification_version=large-offset-v1`,
`sync_state=unverified`, `verification=unknown`,
`serving_state=alass_corrected_large_offset`. It is a hint only.

### Smoke checks

- `/health`: healthy.
- `/manifest.json`: HTTP 200.
- Subtitle listing: HTTP 200 (`{"subtitles":[]}` because server-wide provider keys are empty; per-user
  Nuvio keys are unchanged in the user config URL).
- Compatible same-release subtitle: HTTP 200 in 0.127 s, Metadata Gate `fast_original`, 0 ALASS,
  0 downloads, original bytes preserved.
- No `[ERROR]`, traceback, HTTP 429 or circuit-breaker event after deployment.

### Rollback

```powershell
docker stop ninjasubs
docker rm ninjasubs
docker rename ninjasubs-pre-crosssub-rollback ninjasubs
docker start ninjasubs
```

This restores the previous validated optimization image and its exact environment, mounts, port and
restart policy.

## Risks and safeguards

- The content artifact is only reused for the exact video fingerprint, size, language, original
  content hash, credential context and strict-mode context; different content is independently
  verified.
- The Metadata Gate is evaluated before content reuse and can suppress it.
- Replay never promotes `UNVERIFIED` to `VERIFIED`; it still reports `unverified/unknown` and
  `alass_corrected_large_offset`.
- Legacy sticky pointers are not read under Unified authority.
- Missing/invalid content entries fall back to the existing pipeline, not to an original-as-cache-hit.
- The content tier uses the existing `SyncCache`; no competing cache system was introduced.
- Content reuse is process-local because production has no Redis. After a restart, identical content
  may re-run ALASS once before the content tier is repopulated; the disk sticky/video reference still
  avoids provider discovery.
- Concurrent different subtitle IDs with identical bytes may both execute once before the winner
  publishes the content tier; subsequent requests hit it. Same-subtitle coalescing is unchanged.

## Files changed

- `app/services/sync/cache.py`
- `app/services/sync/orchestrator.py`
- `tests/test_cross_subtitle_reuse.py` (new)
- `tools/benchmark_cross_subtitle_reuse.py` (new)

## Status

Deployed, healthy, and smoke-tested. Awaiting Nuvio playback confirmation for the two Into the Wild
subtitle IDs and the Dexter/Penny regression cases.
