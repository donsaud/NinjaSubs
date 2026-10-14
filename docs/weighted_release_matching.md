# Clean-room weighted release matching

## Research before implementation

Inspected current upstream Bazarr `custom_libs/subliminal_patch/score.py`,
`subtitle.py`, `core.py`, `providers/opensubtitlescom.py`, and
`bazarr/subtitles/manual.py` on 2026-10-10.
Sources: https://github.com/morpheus65535/bazarr/tree/master/custom_libs/subliminal_patch
and https://github.com/morpheus65535/bazarr/blob/master/bazarr/subtitles/manual.py .
The installed Bazarr container could not be inspected: Docker pipe access was denied.
Upstream master is a behavioral reference, not proof of installed library behavior.
No GPL implementation is vendored, imported, or translated. Numeric weight facts
and externally observable results are the research input to independent code.

## Observed behavior

Patched defaults differ from older unpatched Subliminal. Episode weights:
series 160, year 90, season 30, episode 30, source 25, group 20,
audio/resolution/video/HI/service 1 each; hash 359. Movie weights:
title 60, year 40, source 30, edition 30, group 15,
audio/resolution/video/HI/service 1 each; hash 179.
Bazarr uses additive matching evidence, configurable download minimums, and
separate raw and hash-free scores. The manual-search UI normalizes against
360 for episodes and 180 for movies. Its automatic TV path checks series and
episode identity independently of score. Providers supply matches and pack
metadata; selected archive-member handling is outside score calculation.
Source families are merged; a source mismatch can remove group credit.
Hash handling includes provider-specific validation and identity equivalences.
Some optional absent fields and an absent original-series year can count as
matches upstream. These absence heuristics are deliberately not reproduced.

## NinjaSubs semantics

Profiles contain immutable normalized facts, all contradictory evidence, and
independent provenance. Reuse only the existing pure parsing, normalization,
and provider-record deduplication primitives; never reuse V2 policy keys or
matching decisions. Independent new comparison, identity, scoring, and policy
layers consume those profiles. Technical extensions include remux, HDR/DV, and
bit depth as explainable zero-weight facts until validated weights exist.

States are MATCH, MISMATCH, UNKNOWN, CONFLICT. Source-family matches retain
specific subtypes and explain the family relationship. Unknowns earn no points
and are not mismatches. Repeated origins never create additional score.
Only facts comparable to target VIDEO facts earn credit.

Weights above are numeric research facts; HI is a separate late policy
preference, giving full release-score maxima 359/179. The actual normalization
denominator is the sum of weights available from target facts, shared by all
candidates for that request. Missing target facts cannot reward a candidate;
missing candidate facts comparable to known target facts earn zero.
Streaming/disc families use
the service point only when both sides explicitly declare comparable evidence.
Movie edition credit requires explicit comparable cut/version/variant evidence;
absence on both sides remains unknown. Mutually exclusive narrative versions
are hard identity conflicts; presentation/publisher labels are only soft facts.

Wrong known season/episode, conflicting media type, verified independent IMDb
mismatch, and unresolved title contradictions hard reject. Verified equal IMDb
can establish title/year identity without overwriting raw facts; it never
excuses wrong season/episode. Unverified hash strings earn no proof or rejection.
Only strict observed OpenSubtitles hash-query/literal-True/valid-size evidence
can establish MovieHash. Content SHA never establishes video hash.
NinjaSubs adds the independently verified hash weight (359/179) to metadata
points rather than Bazarr's hash-replacement branch: this preserves explanations
and makes verified hash strong through weighted evidence, not a field tuple.
Displayed compatibility saturates at 100%; hash-free percentage remains separate.

Display filters configured languages and HI exclusions. Reference independently
permits English and Arabic, never derives release identity from selected subtitle,
and distinguishes discovery from ready references. Ready references additionally
need independently resolved content identity, known source, exact episode when
TV, validated archive lifecycle, and at least 70% release compatibility.
70% is the retained matching-policy threshold, not a Bazarr or timing-verifier
threshold. Production cutover does not change it.
Display minimum is zero: uncertainty may be shown but contradictions cannot win.

Ranking first preserves verified MovieHash authority, then prioritizes the actual
VIDEO source: exact source subtype, compatible source family, known cross-source,
then unresolved/conflicting source evidence. Unknown video source imposes no
source preference. Within each class, descending additive score and hash-free
score precede late language/HI preferences and deterministic provider ID/canonical
metadata identity. DisplayPolicy and ReferencePolicy use the same semantic key.
Weights, percentages, readiness and synchronization verification are unchanged;
a source-matching season pack still needs actual independently validated member
extraction before it can be used. The contract is versioned as weighted-release-1.2,
which also separates cached display listings from the former score-only order.
Equal semantic keys are evidence-equivalent; deterministic ties are execution
order only. Provider availability and quotas never alter semantic scores.

Unresolved packs can remain discoverable; missing requested episode earns no
fictional episode points. Only an actual production-selected, independently
validated member can become ready. A 94% Bazarr score cannot be asserted from
pack names alone when year/episode evidence is absent.

## Authoritative production integration

`weighted-release-1.2` is now the only production release compatibility authority.
The core weights and 70% ReferencePolicy minimum are unchanged. Compatibility is
release evidence, never synchronization probability or timing verification.

Display: aggregator -> subtitle_matcher.rank_subtitles compatibility entrypoint
-> release_scoring_integration.display_order -> DisplayPolicy over the shared
weighted MatchResult. main's final refresh uses the same policy and video facts;
sync state/cache labels cannot inflate the score. Display cache keys include the
weighted contract so an old ranked listing cannot bypass the cutover.

Reference: external_strategy._select_candidates_ranked -> ReferencePolicy
(discovery) -> bounded operational quota/rejection filtering -> plan_candidates
-> lazy download in exactly that order -> actual selected archive member evidence
-> ReferencePolicy readiness -> unchanged synchronization, target binding, and
verifier rules. English and Arabic acquisition remains independent of display
language. Unresolved packs are discovery-only; unknown sources cannot become
ready. Extracted bare episode names use independent pack season plus actual
member episode evidence, never a copied request episode. Raw packs without a
selected/resolved member remain unusable.

The reusable parser, canonical evidence, provenance and provider-record dedup
utilities moved to `app/services/release_parsing/`; obsolete V2 policy/matching,
shadow, simulator, fixtures and docs were removed. Neither display nor reference
ranking requires shadow switches. Old release-matching switches were removed
from settings; .env was not edited. Historical scoring/ranking APIs delegate to
the weighted engine and retain no alternative score formulas. Historical tier
labels exist only for diagnostics/quota compatibility and retained target-binding
vetoes; they do not order release candidates. Timing-health/anti-circularity
checks remain separate safety checks, and their selection ordering consumes the
weighted key. No release score can authorize transformed delivery.

## Controlled next live test (not executed by this task)

Build/recreate manually from the reviewed working tree, without editing .env:

```powershell
Set-Location 'C:\Users\Saud\.gemini\antigravity\scratch\NinjaSubs'
docker compose build ninjasubs
if ($LASTEXITCODE -ne 0) { throw 'Build failed.' }
$since = (Get-Date).ToUniversalTime().ToString('o')
docker compose up -d --no-deps --no-build --force-recreate ninjasubs
if ($LASTEXITCODE -ne 0) { throw 'Container startup failed.' }
docker logs --timestamps --since $since --follow ninjasubs 2>&1 |
    Select-String '\[weighted-reference\]|\[reference\] candidate reached|winning_strategy|serving_state|served_is_transformed'
```

Play the previously observed Dexter S08E01 BluRay/PiR8 video. Exact BluRay
source precedes REMUX/BDRip compatible-family evidence, followed by WEB-DL/HDTV.
Within the same source class, the existing comparable metadata points decide.
Inspect safe candidate IDs and actual reference attempts. A cached verified
result may skip fresh planning: report that honestly rather than flushing caches.
Check transformed serving through existing verifier/serving telemetry separately.
No switches need enabling or disabling: production weighted ranking is always on.
No Docker build/restart/recreation/deployment was performed by this code task.

## Verification limitations

Installed Bazarr defaults and exact manual percentages are not asserted. The
implementation is clean-room and independently tested, with explicit differences
for missing evidence, denominator and strict hash proof. REAL captures and their
metadata/classifications were not rewritten. Log-derived/synthetic fixture
comparisons remain metadata-only proof. The code is production-authoritative;
the rebuilt container's next actual live attempt order remains to be observed.


## Source-aware numeric points (weighted-release-1.2)

Display and reference telemetry expose the same source-priority ranking points
and normalized percentage as the policy key. Let M be the existing target-only
metadata maximum and B=M+1. For known video source, source bonus is 3B for exact
subtype, 2B for compatible family, B for known cross-source and zero for unresolved
source. Unknown/conflicting VIDEO source earns no source bonus. Ranking points
are metadata points without hash + source bonus + 4B for strictly verified
MovieHash. The hash-free maximum is 4B-1 for known video source, otherwise M;
percentages saturate at 100. Each source bonus is explainable from existing source
evidence, not inferred identity. All candidates in a request share M and B.

This guarantees numeric points agree with the existing source-first order,
including incomplete season-pack metadata. Existing metadata contribution weights,
raw metadata percentages, 70% readiness checks and independently proven member
identity remain unchanged. Ranking percentage is relative release compatibility,
not subtitle timing confidence or readiness proof. Reference telemetry includes
metadata_score, metadata_percentage and source_priority_points for audit; retained
reference-cache provenance keeps its original raw evidence basis.

The screenshot's Bazarr percentages are not copied: NinjaSubs has independent
provider evidence and never awards missing season/episode/year matches. RAW REAL
evidence and fixture classifications are not rewritten.
