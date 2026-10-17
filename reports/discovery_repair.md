# Ranking / Reference discovery repair

Date: 2026-10-17
Scope: original Python NinjaSubs repository. No production container was restarted, rebuilt, or modified; no Rust, credentials, commits, or pushes were touched.

## 1. What changed

### P0 â€” Tier 1 survives Tier 2 failure/timeout

Files:
- `app/services/sync/external_strategy.py`
  - `_gather_candidates`
  - `_search_provider`
  - `_search_tiers`
  - `_search_subdl` / `_search_subsource` / `_search_opensubtitles`

Behavior:
- Each provider owns one deadline (`self.timeout`), passed explicitly into `_search_tiers`.
- Tier 1 is bounded by the remaining provider budget.
- Tier 2 is bounded by the remaining budget minus a small return margin.
- A Tier 2 timeout, `TimeoutError`, provider failure, or expired budget now returns the completed Tier 1 list instead of propagating and discarding it.
- Legacy/test search overrides that do not accept `deadline` are still bounded by an outer hard guard (`timeout + 0.25s`). The real provider path is owned by `_search_tiers`, so a completed Tier 1 cannot be discarded by the outer guard.
- Request cancellation and provider-fan-out semantics are preserved; no global request budget was raised.

### P1 â€” no unnecessary enrichment when an exact candidate is present

File:
- `app/services/sync/external_strategy.py`
  - `_exact_release_identity`
  - `_needs_season_enrichment`

Behavior:
- Tier 2 is skipped when Tier 1 already has a discovery-eligible, source-exact, same-release-group candidate for the requested season.
- A single-episode candidate must also prove the requested episode; a wrong-episode same-group release cannot suppress enrichment.
- This is a discovery shortcut only. Unresolved packs still require download, member selection, `reference_proof`, readiness/eligibility, Target Binding, and verification.
- The old ready source-compatible episode shortcut is unchanged.

### P2 â€” safe Display-to-Reference candidate hints

Files:
- `app/services/sync/discovery_hints.py` (new)
- `app/main.py` â€” register Display listing
- `app/services/sync/external_strategy.py::_gather_candidates` â€” merge hints

Behavior:
- Display stores a short-lived, bounded in-process snapshot of discovered provider candidates.
- The key is the target video identity (IMDb, media type, season/episode, target filename, hash, size) plus a digest of the effective provider credentials. No raw keys are stored in cache keys or logs.
- Reference may reuse those snapshots only as **candidate hints** for the same video and authorization context.
- Every hint still goes through the normal download, extraction, independence checks, `reference_proof`, post-extraction re-scoring, Target Binding, Timeline Affine/ALASS, and verifier. No hint is trusted automatically.
- The cache is memory-only, 10-minute TTL, bounded at 256 videos.
- Cross-video and cross-credential tests confirm isolation.

### Quality fallback

The existing undeployed quality fallback remains integrated:
- weak/unknown/conflict-binding Large-Offset corrections are provisional;
- a failed or deferred attempt can continue to another eligible candidate within the existing bounds;
- strong/exact results remain deliverable;
- no `UNVERIFIED` result is promoted and no serving threshold was changed.

## 2. Mad Men before/after

### Before (production, pre-repair)

The exact WADU candidate was absent from the reference plan because SubSource Tier 1 was discarded on the season-catalog timeout. The weak BTN result was served by the Large-Offset override:

| Target | Served artifact | Outcome |
|---|---|---|
| DBO | `c4e6333c2a7b41a5` | known-bad weak BTN correction |
| SA89 | `0b80bfad2be9a7d3` | known-bad weak BTN correction |

### After (repair benchmark)

Real cached Mad Men target/reference bytes, real orchestrator, real verifier, real Timeline Affine/ALASS paths. Provider discovery is deterministic and mocked; no provider network was used.

| Mode | Target | Reference selected | Binding | Verification | Strategy | Served SHA | Downloads / ALASS | Latency |
|---|---|---|---|---|---:|---|---:|---:|---:|
| `wadu_exact` | DBO | WADU exact pack | strong | verified | timeline_affine | `88936133387288e5` | 1 / 0 | 872.6 ms |
| `wadu_exact` | SA89 | WADU exact pack | strong | verified | timeline_affine | `8646d2a323963bc2` | 1 / 0 | 568.5 ms |
| `wadu_hint` | DBO | WADU from Display hint | strong | verified | timeline_affine | `88936133387288e5` | 1 / 0 | 785.2 ms |
| `wadu_hint` | SA89 | WADU from Display hint | strong | verified | timeline_affine | `8646d2a323963bc2` | 1 / 0 | 562.0 ms |
| `wadu_cached_btn` | DBO | cached BTN deferred, then WADU hint | strong (WADU) | verified | timeline_affine | `88936133387288e5` | 1 / 1 | 3294.5 ms |
| `wadu_cached_btn` | SA89 | cached BTN deferred, then WADU hint | strong (WADU) | verified | timeline_affine | `8646d2a323963bc2` | 1 / 1 | 2457.7 ms |
| `tier1_timeout` | DBO | BTN Tier 1 preserved, no better candidate | weak (deferred) | unknown | original | `3ebf6971f3305f61` | 1 / 1 | 3085.1 ms |
| `tier1_timeout` | SA89 | BTN Tier 1 preserved, no better candidate | weak (deferred) | unknown | original | `b5d8948f7e0886af` | 1 / 1 | 2266.4 ms |

Interpretation:
- `wadu_exact`: P1 skipped the season catalog; WADU ranked first and was served with a verified affine correction.
- `wadu_hint`: provider search failed entirely; P2 recovered WADU from the Display candidate hint. WADU still went through download/proof/binding/verification.
- `wadu_cached_btn`: the weak cached BTN reference was offered first and deferred by the quality fallback (`binding=weak`, p95 3957/2546 ms). The WADU hint was then selected and served verified. This directly proves a weak cached reference cannot suppress a more compatible independent candidate.
- `tier1_timeout`: P0 preserved the completed BTN Tier 1 (`search_calls` included the Tier 2 attempt where budget allowed). The quality fallback refused the known-bad weak correction and served the original. No wrong artifact was delivered.
- The exact bad BTN SHAs (`c4e633...`, `0b80bf...`) were not served when WADU was available.
- Quality-fallback telemetry still reports `quality_served=false` when the stronger WADU result is served through Timeline Affine; this is the known P6 reporting gap, not a serving decision.

## 3. Regression preservation

- **Into the Wild**
  - cold: 10.43 s, 2 downloads, 1 ALASS, served `3aa5134cfa9a1f9e`
  - warm: 32.1 ms, 0 downloads, 0 ALASS, served `3aa5134cfa9a1f9e`
- **Cross-subtitle reuse**
  - first subtitle: 10.45 s, 2 downloads, 1 ALASS, `3aa5134cfa9a1f9e`
  - second identical-content subtitle: 27.2 ms, 0 downloads, 0 ALASS, `3aa5134cfa9a1f9e`
  - different content: 6.10 s, 0 downloads, 1 ALASS, `3aa5134cfa9a1f9e`
- **Dexter / large-offset / season-pack / video-scoped cache**
  - `tests/test_incident_dexter_s08e05.py`
  - `tests/test_cache_consistency_dexter.py`
  - `tests/test_large_offset_serving.py`
  - `tests/test_cross_subtitle_reuse.py`
  - `tests/test_season_pack.py`
  - `tests/test_video_scoped_reference_cache.py`
  - 105 passed.

## 4. Focused tests added

`tests/test_reference_discovery_repair.py`:

- exact WADU pack skips unnecessary enrichment;
- wrong-episode same-group candidate does **not** skip enrichment;
- Tier 1 survives a Tier 2 timeout;
- provider deadline cannot discard a completed Tier 1;
- WADU outranks BTN in both shared policies;
- exact pack failure continues to a Tier 1 alternative within the same plan;
- hint cache is video- and authorization-scoped;
- Display hints recover candidates when the provider search fails, while returning copies rather than shared mutable state.

Benchmarks:
- `tools/benchmark_mad_men_discovery_repair.py`
- existing `tools/benchmark_mad_men_quality.py`, `tools/benchmark_into_the_wild.py`, `tools/benchmark_cross_subtitle_reuse.py`

## 5. Full regression result

Command: `python -m pytest -q --tb=no`

Result:

- **11 failed, 4,879 passed, 40 skipped, 6 errors**
- Baseline before these changes: 11 failed, 4,871 passed, 40 skipped, 6 errors.
- The eight new tests pass; no new failure node ID was introduced.
- The 11 failures and 6 errors are the same pre-existing parser-fingerprint/Penny/subtitle-safety/replay issues recorded before this work.

`ruff` and `mypy` are clean on the changed files.

## 6. Files changed

- `app/services/sync/external_strategy.py` â€” P0/P1, hint merge, deadline plumbing.
- `app/services/sync/discovery_hints.py` â€” new narrow Display-to-Reference hint cache.
- `app/main.py` â€” register Display discovery snapshots.
- `tests/test_reference_discovery_repair.py` â€” 7 focused regressions.
- `tools/benchmark_mad_men_discovery_repair.py` â€” real-byte Mad Men repair benchmark.

Previously undeployed quality-fallback work remains unchanged and integrated:
- `app/services/sync/quality_fallback.py`
- `app/services/sync/orchestrator.py`
- `app/services/sync/telemetry.py`
- `app/config.py`
- `tests/test_quality_fallback.py`
- `tools/benchmark_mad_men_quality.py`

## 7. Deferred follow-ups

Not changed in this phase:

- P3 source-family compatibility unification (scoring vs Target Binding/Metadata Gate).
- P4 Display/Reference tied-score language ordering.
- P5 OpenSubtitles `imdb_id_mismatch` investigation.
- P6 quality-fallback telemetry for affine/Two-Point/PAL/hybrid successes.
- Optional zero-download reuse of a Display-downloaded artifact. The current P2 deliberately re-downloads hinted candidates so unvalidated Display bytes never become trusted reference bytes.

No deployment, restart, commit, or push was performed. Production remains on its existing image.
