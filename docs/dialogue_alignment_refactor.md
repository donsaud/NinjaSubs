# Dialogue-alignment authority and retained Season Packs

This change is general. No title, episode, release group, 96-second offset, or
provider receives a special acceptance exception. Existing activation scopes and
the kill switch remain intact. No commit, push, image build, deployment, credential
change, or production-cache deletion was performed.

## 1. Previous acceptance conditions

The existing Unified implementation already used text-established dialogue events
rather than cue numbers or nearest timestamps. It nevertheless required two
events in **each** opening/middle/ending third, a median under 750 ms separately
in each third, and overall median/P90/max residuals under 750/2000/3000 ms.
Three consecutive residuals over 2000 ms rejected a candidate. Independent groups
were deduplicated; five were required; the matching endpoints were held out.

Other production conditions remained outside that verdict:

* `_passes_cue_sanity` and the main reference loop admitted large offsets only
  after the large-offset evidence gate, with its 180-second ceiling.
* `SubtitleSyncService.sync` applied runtime and cue-count compatibility gates
  before ALASS and structural/span checks after it.
* Primary-regime proposals required eight groups, a majority over contradictory
  regimes, one quarter of the episode span, and a restricted scale interval.
* `_select_evaluation` could downgrade verified timing using a reference trust
  label, and `_target_binding` required source/release compatibility beyond known
  media identity.
* Legacy large-offset and constant-shift serving decisions were already fenced
  off under Unified authority. They did not need another acceptance override.

Identity, self-reference, content preservation, timestamp representability, and
approved HTTP payload integrity remain mandatory requirements.

## 2. Removed or consolidated vetoes

| File / function | Result |
|---|---|
| `app/services/sync/unified_verification.py::verify_candidate` | Regional quotas and regional median/max policy reasons are diagnostic only. One timing decision measures independently matched dialogue. |
| `app/services/sync/orchestrator.py::_passes_cue_sanity`, `_execute` reference loop | Under the existing Unified scope, absolute opening offsets no longer decide admission to timing proposals. Legacy large-offset serving policy does not run on this path. |
| `app/services/sync/orchestrator.py::_try_two_point_primary` | A representable primary model is proposed to the same verifier without a separate support/majority/span/scale serving bar. |
| `app/services/sync/orchestrator.py::_select_evaluation` | Reference trust/ranking labels are retained as provenance; they do not downgrade a measured alignment verdict. |
| `app/services/sync/orchestrator.py::_target_binding` | Known title and exact season/episode, or title/year for a movie, establish media identity under Unified. Ambiguity, wrong title/year/episode and absent identity still fail closed. Different source/group/cut labels do not independently invalidate measured dialogue. |
| `app/services/sync_service.py::SubtitleSyncService.sync` | ALASS is a proposal producer. Runtime/cue-population and post-process structural/span policies no longer decide synchronization. Empty/malformed input, process failures and unresolved concatenated season references still fail. |
| `app/services/sync/candidate_reconstruction.py` | Exact timing transfer restores pristine text; uniquely partitioned split output can be rejoined. Unrecoverable merged/dropped/rewritten output cannot acquire approval. |

The old `_timeline_rejection` and `_validate_synced_output` functions remain for
offline diagnostic compatibility and their existing unit tests; production
`sync` no longer uses them as synchronization acceptance gates.

`timing-v3` segments the new methodology in the verifier, payload/verdict cache,
listing, sticky references, and delivery logs. Existing v2 bytes are not deleted
or reinterpreted as v3 verdicts. Legacy fallback lookup excludes both versions.

## 3. The authoritative dialogue process

The shared correspondence matcher normalizes text for matching, recognizes
bounded exact joins across split/merged reference cues, excludes repeated or
ambiguous placements and generic utterances without context, deduplicates shared
cues, and enforces chronological correspondence. It does not select matches
using the proposed residuals. It withholds matching endpoints from measurement.

The verifier maps these events onto the pristine candidate's timings. It requires
five independent events and sustained evidence spanning one minute, or half the
subtitle span for short material. This prevents a handful of adjacent utterances
from substantiating a timing claim. It does not require events in fixed thirds.

The existing calibrated temporal tolerances remain: median 750 ms, P90 2000 ms,
maximum 3000 ms, and no three-event interior error cluster over 2000 ms. These are
residuals between corresponding dialogue, not movement from the original.
Constant positive/negative offsets, linear drift, and nonlinear proposals use
this same measurement. A valid shift to timestamp zero is allowed when actual
reference correspondence confirms it; an unexplained/mistimed collapse is not.

Cross-language cues are never matched by assumed translation or by similar
timestamp grids. Without trustworthy direct correspondence, the result remains
UNVERIFIED. No new translation service or timing-only shortcut was introduced.

## 4. Different-cut recaps and measurement scope

Absent/unmatched dialogue creates no timing error. Coverage in thirds stays in
telemetry for inspection, but does not authorize or veto synchronization.

Consecutive edge correspondence errors above 3000 ms may define an unsupported
head/tail when the remaining supported evidence constitutes at least 80% of the
matched groups. Interior errors are never trimmed. Larger contradictory sets
reject, and fewer than five supported groups or a narrow evidence span abstain.
The 80% rule conservatively limits partial support; it is not a release/algorithm
exception or a fixed regional quota.

The verdict and evaluation explicitly record the original-target supported
start/end, unmeasured cue count, conflicting edge-group count, and
`global_alignment_claim=False`. A VERIFIED result establishes alignment of the
corresponding dialogue measured; it does not claim that an unmeasured or
contradictory recap is synchronized. Original cues remain in the output.

## 5. Real Dexter outcomes

The new byte fixtures were copied from the existing local cache without changing
that cache. E04 uses target `d010faffc7142fdb` (781 cues); E05 uses target
`5220e5bed50af11f` (885 cues). Their retained d3g episode references have 562 and
584 cues respectively. Different cue counts do not determine acceptance.

| Case | Previous refusal | Independently measured result |
|---|---|---|
| E04 AFG / d3g | `region_uncovered:opening`, `region_uncovered:ending`; counts 1/4/1 | VERIFIED for six groups: median 180 ms, P90 288.5 ms, maximum 373 ms. Supported original-target interval 1,025,790–2,198,220 ms. |
| E05 ASAP / d3g | Opening/middle regional medians 941/822 ms | VERIFIED for eighteen groups: overall median 745 ms, P90 994.3 ms, maximum 1091 ms. Supported original-target interval 700,016–2,726,775 ms. |

`test_real_dexter_dialogue_verdict` checks these measurements, original text/order,
and rejection of a genuinely wrong 50-second proposal.
`test_real_e04_e05_cached_d3g_pack_delivers_without_provider_work` executes the
orchestrator on those real targets/references: approved transformed bytes, no
provider planning/download, and no ALASS invocation.

An additional executable assertion used the exact E05 reference whose SHA is
`b008f568a78c1d5f3ed17acd6a2b65f531a15405fa635bf6cd75ed23f5a1e902`
from the incident log. It reproduced the same eighteen-group residual statistics
and verified content preservation.

E02 is represented by its real AFG target and retained d3g reference. That pair
classifies as nonlinear and its original timestamps do not verify. The existing
`test_s08e02_real_bytes_serve_bulk` additionally exercises the available cached
independent reference that supports a primary-regime correction, preserving all
894 cues and passing final HTTP integrity. A guessed global 96.390-second
correction is rejected by the companion regression. No universal E02 reference
or whole-episode timing claim is made.

## 6. Actual Season Pack failure

The live container contained eight retained pack indexes and parsed the requested
video lineage as BluRay/d3g, but lookup returned no member. The real pack member
files have CRLF line endings. Their **raw bytes** match `content_sha`; reading
them with universal-newline `read_text()` changes CRLF to LF and makes the
recomputed checksum differ. Lookup therefore skipped intact members as corrupt,
then provider discovery/download repeated. This is a generic serialization bug,
not an episode/hash/group-key workaround.

The corrected retention writes UTF-8 bytes and lookup/promotion reads bytes
before decoding. This both avoids Windows newline translation and preserves
reference content/provenance. `test_crlf_members_survive_retention_and_checksum`
reproduces the relevant input and validates lookup in a fresh interpreter.

Two lifecycle deficiencies were also corrected. Retention previously associated
every downloaded alternative with the request's lineage before knowing whether
it won verification. Lookup also depended on provider candidate-planning
interfaces. Packs are now promoted from verified winners, and sticky/pack tiers
run once at the start of reference iteration, before provider-specific tiers.

## 7. Pack reuse and persistence proof

New acquisitions retain the complete original ZIP plus exact episode members and
checksums. The winning source digest and exact seed member identify which pack
becomes proven. Lookup requires the same IMDb/season, known source/group, and
matching available cut/edition identity; episode and target-subtitle ID are not
lineage components. Target language is not a reuse veto. Each new target still
needs independent timing and content verification.

Existing indexes without a winner marker can reuse their checksum-bound Unified
sticky seed. The source digest, series/season/group stem, exact seed episode,
artifact hash, member hash, and corresponding sanitized reference content must
agree. This compatibility recovery is read-only. It does not bless every old
download or fabricate a missing original archive.

Read-only lookup against the existing runtime cache now finds E02, E04 and E05
from the winning d3g source `518b87f28a5b9ffe`. No pack or credential was changed
to obtain those hits.

`test_one_acquisition_winner_promotion_and_e01_through_e05_reuse` proves one
acquisition, correct distinct episode members, fresh orchestrators for E02–E05,
two subtitle IDs per video, zero subsequent provider work, and independent
verification. Absent/corrupt members and different lineage fall back safely.
`test_pack_first_does_not_require_provider_planning_interface` proves lookup is
not conditional on a provider planner. `test_legacy_verified_seed_recovers_pack_read_only`
checks migration and rejects unrelated source proof.

The existing Compose bind `./subs_cache:/app/cache` and reference path
`/app/cache/references` preserve the state through normal container recreation.
Persistence was tested with a fresh Python process; actual Docker recreation was
not performed because rebuild/deployment was not authorized. Existing legacy
packs whose original ZIP was never retained remain usable by their verified
members; only new acquisitions can retain the original raw ZIP.

## 8. Content integrity and false acceptance

Verification now requires exact parsed original cue text, including styling and
whitespace, rather than letting normalization conceal an output edit. Matching
normalization remains separate. ALASS timings are transferred onto the original
content before verification, restoring credits and sanitization exclusions.
Split output is usable only when its entire text sequence uniquely partitions
back onto pristine cues. Merged output without recoverable original boundaries,
missing/invented words, corruption, new cue disorder, invalid durations, and
changed unmeasurable content fail closed.

Regression coverage includes positive/negative 96-second offsets, other negative
offsets, different-cut edges, missing opening, linear drift, nonlinear timing,
actual mistiming, ambiguity/coincidence, unsafe reconstruction, wrong episode,
self-reference, and SRT/VTT HTTP integrity. Existing content/identity safety
assertions were retained or strengthened; ALASS-specific harnesses isolate that
boundary when a now-valid primary proposal would otherwise preempt it.

## 9. Validation

Final focused run: **578 passed, 2 skipped in 54.94 seconds**.
The final zero-timestamp change also passed a focused **84-test** run against
the policy, verifier, and reconstruction suites (13.30 seconds).

Pytest output was shown directly with progress, never piped to a tail filter.
Lint and `git diff --check` were also run. Only existing dependency deprecation
warnings were observed in completed full runs.

## Takeover completion (2026-10-13, Cline session)

- Verified same working tree (`main`, `afa8b4c`) with Codex's 26 modified +
  18 untracked files intact. No code was changed during takeover.
- Previous full-suite run status: **could not be recovered** — no prior full-suite
  log existed and no pytest process was running, so a new full run was required.
- Focused re-validation (batched, `.venv`, `-p no:cacheprovider`):
  policy 28 passed; integrity/reconstruction 26 passed;
  unified verification + authority 63 passed;
  unified delivery + hybrid predelivery 82 passed, 1 skipped;
  large-offset bulk 11 passed; whiplash + penny 35 passed;
  season pack 14 passed; two-point primary + sticky + boundary intake 44 passed.
- Full suite (`-p no:cacheprovider -v --tb=short`, PID 1060):
  **4710 passed, 31 skipped, 0 failed in 287.22s (0:04:47)**.
  Full log: `full-suite-dialogue-alignment-final.log`.
- `git diff --check` clean (only pre-existing CRLF notices); `ruff check` on the
  six core implementation files passes.
- No commit, push, deployment, rebuild, credential, config, or cache action taken.
  Diff remains uncommitted, ready for controlled LOCAL deployment review.

## 10. Files and deployment readiness

Production work is concentrated in `unified_verification.py`, `orchestrator.py`,
`season_pack.py`, `candidate_reconstruction.py`, `query.py`, `hybrid_router.py`,
and `sync_service.py`. The methodology update also touches `main.py`,
`sync/cache.py`, `sync_cache.py`, and `delivery.py`. Tests were updated for the
intentionally removed policies, with a new `test_dialogue_alignment_policy.py`
and narrow fixture allowlisting in `.gitignore`.

All pre-existing modified/untracked work was preserved. The diff remains
uncommitted. Existing environment, provider credentials, Docker configuration,
images, activation gates and runtime cache were preserved. Readiness for a
controlled local deployment depends on the final complete test result, and
deployment still requires the user's explicit approval.
