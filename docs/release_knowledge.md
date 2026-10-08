# Release metadata and knowledge foundation

`matching.parse_release_identity()` remains the canonical release identity entry
point. Its implementation lives in `release_metadata.py`; its immutable model,
three-state fields, comparisons and precompiled vocabulary live in
`release_knowledge.py`. No second target-binding classifier was introduced.

The flow is filename parsing â†’ semantic knowledge â†’ existing target binding â†’
reference ordering â†’ alignment â†’ independent verification. Metadata can choose
which reference to try; it cannot establish subtitle alignment or authorize a
verified response by itself. Existing strict MovieHash verification remains the
strongest exact authority. Providers and alignment algorithms are unchanged.

## Local vocabulary and attribution

`app/services/sync/release_knowledge.json` contains the independently authored
initial vocabulary. Each rule has an ID, category, canonical value, bounded
pattern, confidence, timing relevance, conflict policy and source attribution.
There is no runtime network lookup, imported TRaSH data, scraped rule database,
or movie-specific rule. Patterns compile once at import; filenames never become
executable code or regex patterns.

Supported categories and initial vocabulary:

| Category | Canonical values / examples |
| --- | --- |
| Identity | title, release year, season, episode |
| Cut | directors_cut, extended, theatrical, unrated, uncut, final_cut, international_cut, alternate_cut, alternate_version |
| Label / collection | criterion_collection, arrow, bfi, shout_factory |
| Presentation | imax, open_matte, fullscreen, widescreen |
| Master | remastered, restored; optional restoration_year |
| Packaging | special_edition, collectors_edition, anniversary_edition |
| Source | bluray, uhd_bluray, webdl, webrip, hdtv, dvd |
| Service | amazon (AMZN), netflix (NF), disney_plus (DSNP / Disney+), apple_tv_plus (ATVP / AppleTV+), hbo_max (HMAX), hulu, peacock (PCOK), itunes |
| Technical | release group, resolution, codec, bits, remux, HDR/HDR10/HDR10+/Dolby Vision, audio flags |
| Optional timing hints | explicitly marked FPS, duration; these do not establish a cut match |

The JSON is the authoritative complete alias list. Examples include Criterion /
Criterion Collection, Director's / Directors / Directorâ€™s Cut, Extended Cut /
Edition, Collector's / Collectors / Collectorâ€™s Edition, Remaster / Remastered,
Restoration / Restored, dotted/hyphenated WEB-DL and WEBRip. Bare CC, Special,
International and DC are deliberately not strong edition aliases.

GuessIt was already a dependency and is now pinned to `4.4.0` for reproducible
behavior. Its installed distribution declares `LGPL-3.0-or-later`, with the LGPL
v3 license in its distribution licenses directory. NinjaSubs calls its public
API; no GuessIt source or rule data is copied. No TRaSH material was used or
vendored, so there is no dependency on its licensing or remote availability.

## Context, uncertainty and provenance

Names undergo NFKC, Unicode apostrophe and punctuation/separator normalization.
Provider release names remain untouched on their original objects. Release year
or episode delimiters take precedence over technical words appearing in titles.
A leading numeric title such as 1917 or 2001 does not become its own release
year. Edition rules run only in the metadata region after a delimiter, never
in the title. Scene release-group suffixes are excluded from vocabulary scans.
Potentially confusing episode-title vocabulary before technical markers uses
GuessIt enrichment; recognized episode titles are excluded from tag scans.
Unstructured names do not supply cut tags merely because GuessIt recognizes a
word. Names with no safe delimiter can remain UNKNOWN.

`state(field)` returns KNOWN, UNKNOWN or AMBIGUOUS. `compare_field()` returns
MATCH, UNKNOWN or CONFLICT; two missing values are UNKNOWN, never evidence of
a match. Incompatible explicit cuts and identity differences are contradictions;
missing labels or different packaging cannot invent a cut conflict. Multiple
exclusive cuts, conflicting sources/services and conflicting provider hints
are AMBIGUOUS and cannot establish strict target binding. BLU-ray and explicitly
UHD Blu-ray remain compatible disc source families, without claiming identical
cuts. Technical quality alone never establishes a cut or collection identity.

Evidence records identify the parser/provider/knowledge source, rule ID and
confidence. Optional provider hints accept only bounded title/year/season/episode/
source fields; disagreement is preserved as ambiguity, not overwritten. Existing
providers currently supply release strings, so hints are an extension API, not
new provider behavior. Raw provider configuration is neither cached nor logged
by this layer. New candidate diagnostics contain only fixed semantic tuples,
canonical tags and source families, bounded to five candidates.

The existing EXACT / STRONG / WEAK / UNKNOWN / CONFLICT target-binding policy
consumes this metadata. A generic PublicHD reference for a Criterion target stays
UNKNOWN even if alignment scores are excellent. A title/source-compatible
Criterion reference is STRONG. Matching FraMeSToR disc references can still be
STRONG without MovieHash despite resolution/codec differences. Independent
alignment verification remains required in both cases.

## Reference ordering and caches

The complete search-result pool is ranked before the bounded plan is selected.
Lexicographic keys put hash hints first, then identity/conflicts, high-signal
cut/label/presentation evidence, master/restoration, packaging, source/service,
group and low-impact technical matches. Technical points cannot outweigh a cut
conflict. Conflicted candidates remain possible fallback hints rather than being
made new hard provider search constraints. Season/episode eligibility and
operational tie-breakers remain; quota/download eligibility is still checked
before consuming each plan slot. Downloads remain lazy and sequential.

The immutable parser LRU is capped at 2048 entries; inputs and normalized names
are capped at 2048 characters. Its identity includes schema version, parser
version, knowledge version/content digest and pinned GuessIt version. Knowledge
patterns compile once. Conventional year/episode plus technical releases use
the local fast path; ambiguous/incomplete names call cached GuessIt enrichment.
No new unbounded request tasks or remote metadata work were added.

Sync verdict engine is now **6**, payload/resolution namespace
**v9-release-context-v2**. This is an intentional semantic invalidation:
title words previously mistaken for edition tags must not retain old verified
trust. No existing cache or REAL evidence file is deleted or rewritten. Capture
V2 subtitle parser fingerprints are separate and unchanged. Future semantic
rule/parser changes require a parser/knowledge version review and a sync verdict/
namespace review whenever they can change target-binding or ordering decisions.

## Extension and validation

Add an independently sourced rule to the JSON with all provenance/policy fields;
keep aliases specific and regexes bounded. Add positive, title-context, missing,
ambiguous and conflicting examples to `tests/test_release_metadata.py`. Increment
the knowledge version, review cache semantics, and run metadata, binding,
planning, lazy/quota, MovieHash, sync-cache and full-suite regressions. Do not
weaken verification thresholds to accommodate a new alias.

Run the offline benchmark with `python -m tools.release_metadata_benchmark`.
It measures mixed releases at 10/50/100 candidates, cold and warm parsing/ranking,
GuessIt calls and LRU behavior; it neither downloads nor runs synchronization.
Wall-clock values are observations, not flaky CI thresholds.

Measured locally against HEAD's previous parser/ranking on the same mixed names:

| Count | Old cold / warm parse ms | New cold / warm parse ms | Old cold / warm ranking ms | New cold / warm ranking ms | Old / new cold GuessIt calls |
| --- | --- | --- | --- | --- | --- |
| 10 | 264.687 / 0.546 | 188.326 / 0.021 | 131.257 / 0.095 | 58.521 / 0.327 | 10 / 1 |
| 50 | 414.996 / 2.652 | 44.260 / 0.067 | 444.175 / 0.379 | 41.471 / 1.271 | 50 / 8 |
| 100 | 832.295 / 5.261 | 77.303 / 0.128 | 892.739 / 0.703 | 82.021 / 2.668 | 100 / 16 |

The first small batch includes GuessIt initialization. Warm calls use no GuessIt;
cold ranking invokes it only for the same ambiguous names. Semantic comparisons
increase warm ranking overhead, while cold ranking improves substantially.
An earlier all-conventional baseline measured 1069.106 ms cold and 5.510 ms warm
for 100 parses, motivating the fast path. Results vary by machine and process.

Remaining limits: filenames can omit or mislabel editions; localized/ambiguous
titles can remain UNKNOWN; multi-episode ranges and additional vocabulary need
explicit future rules/tests. No release metadata system proves identical cuts
without independent alignment evidence. Boundary remains shadow-only and its
promotion evidence gate is unchanged.


## User-visible Compatibility Score

The existing percentage and primary subtitle-list order now use one local engine:
`app/services/visible_compatibility.py::compatibility_score`. `rank_subtitles`
applies identity/episode/anime eligibility and existing deduplication, then scores
cached ReleaseMetadata. After scoped cache verdict enrichment,
`refresh_visible_compatibility` reuses the parsed objects and the same engine.
`presentation_key` orders the displayed integer descending; the badge formatter
only formats it. No second verification ranking can override that percentage.
The legacy additive execution score remains for existing reference execution
consumers, separate from the visible score; it is not a visible sorting key.

This is an ordinal Compatibility Score, not a calibrated sync probability.
The non-overlapping bands are EXACT 100, STRONG 85–99, WEAK 65–84,
UNKNOWN 25–64, CONFLICT 0–24. The old system normalized additive points by 135,
assigned 100 for same-group matches, and sorted on language/tier/sync state before
percentage. The bands replace that unsafe accumulation: exact proof is isolated;
strong metadata has limited verification headroom; partial positive evidence
cannot catch strong identity; missing evidence stays above proven contradiction.
The band boundaries are policy caps, not empirical probability estimates.

100 requires literal true pre-existing exact MovieHash provenance. Filename,
group, excellent alignment, or a transformed-output badge cannot manufacture it.
Current visible candidates have no embedded extraction provenance. Target binding
is authoritative: conflicting exclusive cuts score 10 (identity conflicts 0),
regardless of group/resolution/codec. Missing fields add no penalty. Differing
technical encodes add no contradiction penalty. Unsupported master differences
remain UNKNOWN rather than inventing new conflict rules.

Within a band, shared cut/label/presentation/master evidence has hierarchical
priority, followed by compatible source/group and identity. Technical matches have
at most two points (one in STRONG); resolution/codec/HDR/audio cannot compensate
for cuts. Existing REPACK/PROPER alignment contributes one point. Trusted scoped
verified/cached target-bound EXACT/STRONG evidence adds at most three points
within the existing band. Prediction, unverified transforms, failure/fallback,
and the Large Offset delivery exception add nothing. Alignment confidence and
identity confidence remain distinct.

Equal displayed scores use configured language order (normalized language codes),
HI preference, existing micro-rip/single-episode/technical quality signals,
release name, provider and download identity. Provider is only deterministic tie identity, never compatibility
credit. HI exclusion remains eligibility; HI preference and language preference
break ties rather than overriding a better score. Enabled-provider/language
filtering is retained upstream. Existing explicit cut-conflict eligibility filtering is preserved; diagnostic
requests that retain mismatches receive the low CONFLICT score.
All candidates are ranked before serialization;
there is no separate visible-result truncation in the current handler.

Aggregation keys include `visible-compatibility-v1` and the ReleaseMetadata
semantic version to prevent old scores from being reused. Parser version 2 adds
cached GuessIt edition context only for structured yearless filenames such as
`Movie.Directors.Cut.2160p...`; whole titles such as `Final.Cut` remain protected.
Because this affects binding input, verdict engine 6 and payload/resolution
namespace v9 prevent older interpretation-dependent results retaining trust.
Parser v2 also retains GuessIt group suffix enrichment (including normalized
underscore groups), resolves anime title context, and prevents a guessed release
year from being interpreted as a season. Multi-episode visible projections are
explicitly ambiguous rather than falsely claiming one exact episode.
Raw provider caches and subtitle parser/Capture V2 fingerprints are unchanged.
No cache or REAL artifact is deleted or rewritten.

Calibration examples are in `tests/test_visible_compatibility.py`: Criterion
96/94 > generic PublicHD 37 > WEB 29; Whiplash/FraMeSToR disc 90 without hash;
same-group Director's/Theatrical conflict 10. These are illustrative policy
indices and may evolve only with versioned evidence-backed semantics.
Future signals must belong to eligibility, compatibility, preference/quality,
or verified evidence explicitly; add band/monotonicity/trust tests and review
cache versions. Never add a provider-name or download-time bonus to compatibility.
Run the existing offline benchmark to measure visible cold/warm ranking, cached
refresh, final sorting and GuessIt invocation counts at 10/50/100 candidates.
