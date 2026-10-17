# Ranking / Display and Reference consistency audit

Date: 2026-10-17
Scope: original Python NinjaSubs repository. Read-only investigation. No production container was restarted, rebuilt, or modified; no Rust or unrelated service was touched; nothing was committed or pushed.

## 1. Executive summary

Display and Reference already share the same weighted release engine:

`release_scoring_integration.matches()` -> `release_scoring.engine.evaluate()` -> `policies.rank()`

They differ in eligibility and language/HI policy, not in the base compatibility score. For the Mad Men WADU candidate, both policies rank it **above** the BTN reference if it is in the pool:

| Candidate | Score | Percentage | Source priority | Display eligible | Reference discovery eligible |
|---|---:|---:|---:|---|---|
| WADU exact release (`Mad.Men.S01.REPACK.2160p.HMAX.WEB-DL...WADU.srt`) | **1409** | **97.92%** | 0 | yes | yes |
| BTN Web-DL (`Mad.Men.S01.720p.WEB-DL...BTN.srt`) | 1385 | 96.25% | 0 | no (English not requested) | yes |
| SA89 BluRay reference | 640 | 44.48% | 2 | yes | yes |

These are reconstructed with the shared engine from the cached release names/metadata. Production's BTN entry scored 1025 with the same source priority because its target profile carried slightly different comparable facts; the important invariant is the relative WADU > BTN ordering, which holds because WADU adds release-group, resolution and codec matches.

The WADU candidate was **not rejected by ranking or ReferencePolicy**. It was lost during provider discovery.

Confirmed root cause:

1. SubSource tier-1 episode search returned 21 candidates.
2. Because no candidate was *ready* (the exact WADU release is a season pack, so `season_pack_unresolved`), `_needs_season_enrichment()` triggered a tier-2 season catalog search.
3. The provider search has a single 5 s outer timeout. Tier-1 plus the season catalog exceeded it; `TimeoutError` cancelled the whole SubSource coroutine.
4. `_gather_candidates()` caught the exception and returned `[]` for SubSource, discarding tier-1 as well.
5. The resulting reference plan contained 99 candidates: SubDL 35 + OpenSubtitles 64. SubSource's 21 were absent. The WADU candidate was therefore never planned or downloaded.
6. The first remaining SubDL candidate (BTN Web-DL) was weak-binding and low-confidence; the Large-Offset override served it immediately, without trying the other planned candidates.

The undeployed quality fallback from the previous task fixes step 6, but **cannot fix step 4**. With the WADU candidate restored to the pool, the quality fallback defers BTN and selects WADU; a controlled run served a verified `timeline_affine` result instead of the wrong BTN correction.

## 2. Current pipelines

### Display / Ranking

```
provider fan-out (requested languages)
  -> SubtitleRelease objects
  -> release_scoring_integration.record()
       canonical release parsing + legacy identity enrichment
  -> target profile from video facts
  -> release_scoring.engine.matches() / evaluate()
       source family, title/year, season/episode, group, resolution, codec, edition
  -> DisplayPolicy.eligible()
       discovery_eligible + requested language + HI policy
  -> DisplayPolicy.key()
       semantic_key + requested-language order + HI + stable identity
  -> display_order()
       set score/match_percentage/match_tier, return ordered releases
  -> listing
```

### Reference selection / synchronization

```
provider fan-out (eng + ara)
  -> tier 1 episode query
  -> _needs_season_enrichment()
  -> tier 2 season catalog (if tier 1 has no READY source-compatible episode)
  -> combined pool, URL dedup
  -> release_scoring_integration.discovery_pairs()
       same record/evaluate engine
  -> ReferencePolicy(discovery=True).eligible()
       discovery_eligible + eng/ara
  -> ReferencePolicy.key()
       semantic_key + English-first + stable identity
  -> _select_planned_attempts()
       pool limit, provider/quota/breaker guards, source dedup
  -> lazy download -> decode/extract member
  -> reference_proof() / ReferencePolicy() ready rescore
  -> video-scoped cache / season-pack retention
  -> orchestrator loop:
       cue sanity -> Target Binding -> affine -> Two-Point -> ALASS
       -> verifier -> Large-Offset investigation -> quality fallback -> serve
```

The failure point is marked between **tier 1** and **combined pool**: tier 2 failure discards tier 1.

## 3. Shared engine vs policy differences

Both paths call the same `matches()`/`evaluate()` and the same `semantic_key()`:

- verified MovieHash first,
- `source_priority`,
- `ranking_score`,
- `score_without_hash`.

Differences are confined to:

| Concern | DisplayPolicy | ReferencePolicy |
|---|---|---|
| Eligibility | `discovery_eligible` + requested languages + HI | `discovery_eligible` (planning) or `ready_reference_eligible` (proof) + `eng`/`ara` |
| Tie-break after semantic key | requested-language order, HI, stable identity | English-first, stable identity |
| Purpose | user-visible subtitles | timing references |

For WADU vs BTN, the scores are not tied (1409 vs 1385), so the language tie-break does not affect the order. WADU is first in both policies.

### Confirmed policy inconsistencies

1. **Language tie-break divergence** (`policies.py`): when two shared candidates have identical `semantic_key`, Display can put the requested Arabic candidate first while Reference puts English first. This is purpose-driven but should be documented/tested; it is not the Mad Men cause.
2. **Source-family divergence**:
   - `release_scoring/engine.py` and `release_parsing/rules.py` treat `web_dl`/`webrip` as the same `web` family, `bluray`/`remux`/`bdrip` as `disc_hd`, `hdtv`/`sdtv` as `broadcast`.
   - `release_knowledge.compatible_source()` treats only exact source or BluRay/UHD-BluRay as compatible.
   - `target_binding.classify_target_binding()` and `metadata_gate` use `compatible_source()`.
   - Consequence: a candidate can rank with `source` MATCH in Display/Reference but fail Target Binding or be gated as a source conflict. Example: WEB-DL target vs WEBRip reference. This is a real architectural inconsistency, though not the Mad Men case (both WADU and BTN are WEB-DL).
   - `subtitle_matcher.SOURCE_FAMILIES` is a third, legacy map (e.g., BDRip is `DISC_LO` there), used by older helper paths.
3. **Parser boundary**: ranking uses canonical `release_parsing` + `_name_identity`; binding uses `release_metadata.parse_identity`/`matching.parse_release_identity`. They agree for WADU/BTN but are separate implementations. A single canonical identity should feed both.

## 4. Mad Men reconstruction

Timeline from production logs (2026-10-17):

| Time | Event |
|---|---|
| 10:15:26 | Display search for WADU video, S1E1, Arabic |
| 10:15:27-28 | SubSource returns 109 items, 11 final; SubDL 10; OpenSubtitles 4; listing returns 16 ranked subtitles |
| 10:15:31-32 | Display downloads `subtitle/10184771/download` from SubSource — the exact WADU season-pack subtitle (`ba4ceee32aab31d3`) |
| 10:15:32 | `evaluate sub=ba4ceee...` -> Metadata Gate `fast_original / same_release_lineage` |
| 10:16:10 | DBO sync starts; SubDL episode query 21, no ready Web-DL -> season catalog 35 |
| 10:16:12-14 | SubSource tier-1 episode query returns 21 candidates |
| 10:16:14.369 | Provider logs `Subsource returned 21 subtitles` |
| 10:16:15.35-15.59 | SubSource season-catalog pages 1-2 |
| **10:16:15.797** | **`[reference] subsource search failed: TimeoutError`** |
| 10:16:15.886 | Reference plan: `ranked_releases=15`, `unranked_releases=84` = **99 total** = SubDL 35 + OpenSubtitles 64; **no SubSource** |
| 10:16:15.916 | Plan starts with SubDL BTN Web-DL |
| 10:16:17.16 | BTN pack extracts S1E1; binding weak (`source_family_only`) |
| 10:16:18-19 | ALASS, p95 3957 ms, low confidence; Large-Offset serving allowed |
| 10:16:19.542 | BTN correction `c4e6333c2a7b41a5` served immediately |
| 10:16:19-22 | Second DBO request uses video-scoped BTN; same weak correction served |
| 10:16:43-45 | SA89 uses video-scoped BTN; p95 2546 ms; correction `0b80bfad2be9a7d3` served |
| 10:38 | SA89 repeat uses video-scoped BTN; same wrong correction served |

### Candidate-set evidence

The Display request and the Reference tier-1 request used the same SubSource season/episode query. Display downloaded WADU successfully. The Reference plan total proves SubSource contributed zero candidates: 99 = SubDL 35 + OS 64.

The 5 s timeout starts at the reference search start (~10:16:10.7), not at tier 2. `_DEFAULT_TIMEOUT` is 5.0 s while `UPSTREAM_TIMEOUT` is 6.0 s. Tier-1 completed at 14.369; tier-2 was still running at 15.59; the outer `asyncio.wait_for` cancelled the provider at ~15.7.

### Was WADU in tier 1 or only tier 2?

The logs do not print candidate names per tier, so this cannot be proven from the captured output. It is a **confirmed code defect either way**:

- If WADU was in tier 1, `_search_tiers` discarded it when tier 2 failed.
- If WADU was only in tier 2, the outer timeout cancelled the provider before it could return the pack.

Both are the same partial-result defect. Display's successful download of the WADU release in the same season/episode scope proves it was discoverable and provider-available.

## 5. Confirmed bugs and evidence

### B1 (critical) — tier-1 candidates are discarded when tier-2 enrichment times out

Files/functions:
- `app/services/sync/external_strategy.py::_search_tiers` (lines ~1776-1795)
- `app/services/sync/external_strategy.py::_gather_candidates` (lines ~946-963)
- `_DEFAULT_TIMEOUT = 5.0` (line 76)

Evidence:
- SubSource tier 1 returned 21 (`Subsource returned 21 subtitles`, 10:16:14.369).
- Tier 2 season catalog was requested.
- `subsource search failed: TimeoutError` at 10:16:15.797.
- Plan total 99 without SubSource.
- Display had just downloaded the exact WADU candidate from the same provider/scope.

This is the primary Mad Men root cause.

### B2 (critical, production behavior) — first weak guarded result was served immediately

File/function:
- `app/services/sync/orchestrator.py::_execute`, Large-Offset serving block.

Evidence:
- BTN weak/low-confidence result returned `serve_synchronized=True` via the Large-Offset override.
- The loop returned at candidate 1; remaining 5 planned candidates were never reached.
- The undeployed quality-fallback work fixes this by deferring weak/unknown/conflict-binding guarded results and continuing. It does not recover WADU if discovery already lost it.

### B3 (high) — Display-discovered candidates and downloaded artifacts are not reusable by Reference

Files:
- `app/main.py::_fetch_subtitles_handler` / display path -> `display_order`
- `app/cache.py` -> selected subtitle bytes/metadata only
- `app/services/sync/external_strategy.py::_download_candidate` -> reference cache/season-pack retention
- `app/services/sync/season_pack.py::retain_season_pack` (only called from reference download)

Evidence:
- WADU target downloaded at 10:15:32 and stored in `subs_cache/ba4ceee...srt` + `_meta`.
- Reference search at 10:16:10 did not read that candidate/artifact; it re-searched providers and lost SubSource to timeout.
- Display does not call `retain_season_pack`; the WADU pack was not indexed for reference reuse.
- The candidate metadata for `ba4ceee` even carried a different cached `target_filename` (SA89) from a prior request; a bridge must validate video identity, not trust cached metadata blindly.

A Display candidate may be offered as a **hint**, but it must still be downloaded/extracted, checked for independent content, rescored through the shared engine, Target-Bound, and verified. It must never be trusted automatically.

### B4 (high) — season-pack enrichment ignores unresolved exact packs

File/function:
- `app/services/sync/external_strategy.py::_needs_season_enrichment` (lines ~1754-1774)

Evidence:
- Synthetic WADU exact pack: `discovery_eligible=True`, `ready_reference_eligible=False`, readiness `season_pack_unresolved`.
- `_needs_season_enrichment([WADU], query)` returns `True` even with the exact release present, because readiness requires extraction.
- This triggers the tier-2 call that times out.

The intent (find a ready episode reference) is reasonable, but it must not make tier-1 candidates disposable.

### B5 (medium) — source-family compatibility is implemented three ways

Files:
- `app/services/release_scoring/engine.py` (`SOURCE_FAMILIES`)
- `app/services/release_parsing/rules.py` (`SOURCE_FAMILIES`)
- `app/services/subtitle_matcher.py` (`SOURCE_FAMILIES`, legacy)
- `app/services/sync/release_knowledge.py::compatible_source`
- `app/services/sync/target_binding.py` (line ~78)
- `app/services/sync/metadata_gate.py` (lines ~154,167)

Evidence/code:
- Scoring: `web_dl`/`webrip` -> `web`; `bluray`/`remux`/`bdrip` -> `disc_hd`; `hdtv`/`sdtv` -> `broadcast`.
- Binding: only `a == b` or both in `{bluray, uhd_bluray}`.
- A candidate can therefore be a source-family MATCH for ranking but fail binding. Not the Mad Men root cause; a real consistency defect elsewhere.
- Legacy `subtitle_matcher` map differs again (`BDRip`/`BRRip` -> `DISC_LO`).

### B6 (low/medium) — language tie-break differs between Display and Reference

File: `app/services/release_scoring/policies.py`

- Display key: requested-language order, then HI.
- Reference key: English first.
- Only affects exact `semantic_key` ties. WADU vs BTN is not a tie, so this did not cause Mad Men.

### B7 (medium, observed) — OpenSubtitles candidates are marked `imdb_id_mismatch`

Evidence:
- Both Into the Wild and Mad Men reference plans list OpenSubtitles candidates with `rejections=["imdb_id_mismatch"]` despite an IMDb-scoped search.
- This removes OS from the eligible reference pool.
- Root cause is not confirmed; likely provider feature/parent IMDb evidence normalization. Separate investigation needed; not the Mad Men cause.

### B8 (low) — quality-fallback success telemetry misses non-ALASS serve paths

File/function:
- `app/services/sync/orchestrator.py`, `quality_fallback_served` is set in the ALASS serve block only.
- If a stronger candidate is served by `_try_timeline_affine`, Two-Point, PAL or hybrid, `quality_fallback_triggered=true` but `quality_fallback_served=false`.

Evidence:
- Controlled run: cached weak BTN was deferred, WADU then served via `timeline_affine` (`889361...` / `8646d2...`); decision correct, telemetry flag false.
- Decision correctness is unaffected; reporting only.

### B9 (low) — docstring contradicts enrichment behavior

`ExternalExactStrategy` says series queries are enriched “only when the episode query returned nothing”, but `_needs_season_enrichment` enriches when there is no **ready** source-compatible episode. This is documentation, not a functional cause.

## 6. Sticky / video cache / prefetch / quality fallback interaction

- The video-scoped BTN reference is offered before planning. In production it was weak and served immediately, so planning never ran on later requests.
- The undeployed quality fallback now defers weak/unknown/conflict-binding guarded results. That lets the loop resume and plan, but only candidates that survived discovery can be tried.
- A cached weak reference cannot suppress a stronger WADU candidate **once WADU is in the plan**: controlled run with BTN pre-seeded in the video cache deferred BTN and selected WADU via `timeline_affine`.
- Progressive prefetch starts after planning; it does not fix provider discovery loss.
- Quality fallback does not promote `UNVERIFIED`; it preserves strong-binding Into the Wild/Dexter and safe-falls back to original for Mad Men when no stronger candidate is available.

## 7. Season-pack audit

- DisplayPolicy uses `discovery_eligible`, so unresolved packs are visible; WADU was visible and downloadable.
- ReferencePolicy(discovery=True) also admits unresolved packs. `reference_resolvable` requires source and season but not a resolved episode, so a pack can be planned.
- `_download_candidate` extracts the selected member, then `reference_proof` re-scores it through the same engine with `member=...`; `ReferencePolicy()` ready eligibility must pass before the video-scoped artifact is stored.
- `retain_season_pack` runs only on reference downloads, not Display downloads. The WADU Display artifact was therefore not retained for reference reuse.
- The unresolved WADU pack would be planned if tier-1 survived; extraction/proof/rescore would decide readiness. No ranking fix is needed.

## 8. Prioritized minimal-change repair plan

### P0 — preserve tier-1 results when tier-2 enrichment fails
- In `_search_tiers`, give tier 2 its own bounded timeout/try-except; return tier 1 on timeout/exception.
- In `_gather_candidates`, do not let a provider-level timeout discard a completed tier 1; expose tier1/tier2 counts and failure reason.
- Add a regression test with a fake provider: tier 1 returns candidates, tier 2 sleeps/raises; assert tier 1 reaches the plan.
- This is the smallest fix that restores the exact WADU candidate.

### P1 — avoid unnecessary enrichment when an exact/high-confidence pack is already present
- `_needs_season_enrichment` currently ignores unresolved exact packs. Consider treating a `discovery_eligible` candidate with source MATCH and exact release identity (same group/edition) as sufficient for discovery, so tier 2 is not required.
- Keep extraction/proof mandatory; this only avoids the extra provider search.

### P2 — bridge Display discovery to Reference safely (larger change)
- Cache a short-TTL per-video candidate listing from Display (keyed by imdb/season/episode/languages/credentials/provider) and let Reference planning use it as candidate hints.
- Or register a Display-downloaded exact-release target artifact as a **candidate hint** / video-scoped reference only after extracting the episode and proving independence; never auto-trust it.
- Any bridge must re-run extraction, `reference_proof`, Target Binding, and the verifier.
- This would have recovered WADU even if SubSource timed out.

### P3 — unify source-family compatibility
- Use one canonical source-family map/function for scoring, binding, and metadata gate; or explicitly document the stricter binding policy and test the differences.
- Add tests for WEB-DL vs WEBRip, BluRay vs REMUX/BDRip, HDTV vs WEB.

### P4 — align or document language tie-break
- Keep `semantic_key` identical; decide whether Reference's English-first tie-break is a purpose rule or should match Display's requested-language preference. Add a test for tied scores.

### P5 — investigate OpenSubtitles `imdb_id_mismatch`
- Separate from Mad Men; verify provider parent/series IMDb normalization before changing eligibility.

### P6 — quality-fallback telemetry for affine/Two-Point/PAL/hybrid serves
- Set `quality_fallback_served` when a deferred weak provisional is displaced by any stronger verified serving path.
- No decision change.

## 9. Regression tests to add and acceptance criteria

### Mandatory deterministic scenarios

1. **Tier-1 survives tier-2 timeout**
   - Fake provider returns two tier-1 releases; tier-2 raises `TimeoutError`.
   - Assert `_search_tiers` returns both tier-1 releases; `_gather_candidates` includes them; plan count includes them.
   - Acceptance: test fails on current code and passes after P0.

2. **Exact WADU pack is planned**
   - Build WADU release metadata from the real name/provider.
   - Assert `discovery_eligible=true`, `reference_resolvable=true`, plan includes it.
   - Assert Display and Reference both rank WADU above BTN (1409 vs 1385).

3. **WADU extraction / rescore / proof**
   - Use a deterministic ZIP with the WADU S1E1 member.
   - Assert member selection, `reference_proof` ready, video-scoped artifact stored, unresolved pack not cached.

4. **Weak cached BTN cannot suppress WADU**
   - Pre-seed video-scoped BTN; plan WADU + BTN.
   - Assert BTN is deferred as weak/low-confidence; WADU is attempted; final served artifact is the WADU-derived verified result, not `c4e633...`/`0b80bf...`.
   - Controlled evidence already exists: with BTN cached first and WADU in the pool, the current quality fallback selected WADU and served `timeline_affine` (`889361...` / `8646d2...`).

5. **Mad Men real bytes**
   - DBO/SA89 targets, BTN reference, WADU reference; assert:
     - before P0: WADU absent from pool; weak BTN correction served (current production behavior);
     - after P0: WADU in pool; quality fallback does not accept BTN; WADU is selected/verified; no wrong weak correction served.
   - Expected delivery: verified/strong result (or safe original if WADU extraction fails), never the known-bad BTN correction.

6. **Shared ordering for shared eligible candidates**
   - Use the same candidate list under Display and Reference; assert identical relative order for candidates eligible in both, ignoring language/HI-only policy differences.
   - WADU first; SA89 lower; BTN Display-ineligible but Reference second.

7. **Provider failure fallback**
   - One provider times out; another returns candidates; assert candidates from the healthy provider still plan/download and the request does not end with the weak first candidate.
   - Assert a failed download/extraction advances to the next eligible candidate.

8. **Preserved regressions**
   - Into the Wild strong binding still delivered (first and cross-subtitle warm).
   - Dexter strong-binding Large-Offset still delivered.
   - Penny verified Two-Point still delivered.
   - Compatible same-release subtitles still bypass synchronization.
   - Cross-subtitle cache/sticky tests unchanged.

### Measurable acceptance criteria

- A reference plan for Mad Men WADU includes the exact WADU candidate when SubSource tier 1 returns it.
- SubSource timeout no longer reduces the plan by the whole tier-1 batch; tier-1 count is preserved in telemetry and planning.
- WADU outranks BTN in both Display and Reference for the shared eligible set.
- DBO/SA89 never serve the known-bad `c4e633...`/`0b80bf...` artifacts when WADU is available; final result is verified/strong or safe original.
- Into the Wild/Dexter/Penny and cross-subtitle warm behavior are unchanged.
- No increase in false accepts; no `UNVERIFIED` promotion.
- Timeout case latency does not include a full extra 5 s tier-2 wait when tier 1 already has usable candidates.

## 10. Evidence artifacts

Reproduction and evidence files used during this audit:

- Production log slice: `mad_slice.log` (SubSource tier-1 21, timeout, plan 99, BTN weak serve).
- Shared ranking evaluation: `mad_rank_audit.py` (WADU 1409 > BTN 1385; both policies).
- Mad Men quality benchmark: `tools/benchmark_mad_men_quality.py` (before wrong BTN SHAs, after original).
- Controlled WADU reference benchmark: WADU selected and verified (`timeline_affine` 889361.../8646d2...).
- Cached WADU target metadata: `subs_cache/_meta/ba4ceee32aab31d3.json`; target artifact `subs_cache/ba4ceee32aab31d3.srt`.

## 11. Recommendation

Do not implement a new ranking system. The shared engine and ordering already rank WADU correctly.

Approve the minimal P0 repair first: preserve tier-1 candidates when tier-2 enrichment times out. Add P1 only if measurements show the extra enrichment is still unnecessary when an exact pack is present. P2 (Display-to-Reference bridge) is the next architectural improvement if provider timeouts remain a risk. Source-family unification (P3) and language tie-break documentation (P4) should follow as separate consistency work.

No production changes were made in this phase.
