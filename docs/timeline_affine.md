# Timeline Affine synchronization

The affine candidate runs on pristine target text for each resolved, full-length
reference before the legacy cue-sanity/ALASS execution path. A successful affine
verification returns a measured `verified_resynced` artifact. Rejection returns
control to existing strategies. MovieHash lookup and identity proof are unchanged.

`timeline_affine.py` contains the reusable timeline, landmarks, candidate search,
verification and timestamp-only transformation. It uses existing cue parsing,
dialogue/metadata classification and reliable endpoint-run selection. The legacy
alignment analyzer and PAL thresholds are unchanged.

## Temporal evidence

- Sort and union usable dialogue intervals; duplicates and overlaps have no
  additional weight. Close gaps up to 250 ms to tolerate short segmentation gaps.
- Extract silence boundaries of gaps at least two seconds long, retaining large
  gaps in each of eight windows before filling the remaining landmark budget.
- Include reliable run endpoints and filtered outer dialogue onsets as hypotheses;
  isolated final dialogue can be a candidate even after a long silent scene.
  Never treat their
  positional correspondence as proof. Maximum 40 landmarks per subtitle.
- Pair landmarks by type, approximate normalized position (within 12% of the
  timeline), and compatible gap duration. At most three reference alternatives
  per target landmark.
- Generate at most 384 affine hypotheses from widely separated candidate pairs,
  plus constant shifts. Reuse existing slope limits 0.95–1.05; intercept magnitude
  is limited to five minutes. PAL proximity is a classification signal only.
- Rank using symmetric duration coverage against occupancy padded by a fixed
  350 ms boundary tolerance plus bounded robust onset precision. A median onset
  residual breaks occupancy plateaus; outer onset corroboration contributes at
  most 0.01 score weight. Every candidate still needs full middle verification.
  Verify at most eight leading candidates. Refine a
  precision fit only when at least six onset landmarks agree within 10 ms and
  cover at least 70% of the timeline; duration-sensitive gap ends are excluded.

## Independent verification

Every accepted model must have:

- at least two minutes of timeline and 16 independent occupancy intervals;
- symmetric occupancy coverage at least 0.80;
- improvement over the original of at least 0.06;
- occupancy agreement at least 0.70 in every one of eight windows;
- at least six landmark inliers within 1000 ms, distributed through five windows;
- no window preferring a residual shift greater than 1500 ms with at least 0.04
  extra occupancy agreement. Diagnostic residual shifts search ±8 seconds in
  500 ms steps; they never repair the candidate or create a piecewise output.

The fixed boundary tolerance tolerates display duration and small segmentation
differences. Symmetric true-duration denominators prevent truncated or sparse
reference coverage from looking complete. Distributed windows and residual shift
checks prevent good endpoints from hiding a broken middle. Ambiguous, featureless
or sparse timelines fail closed. These are new timeline criteria, not relaxed
legacy MAD/p95 gates.

Transformation replaces only timestamp spans on timestamp lines. Cue IDs, bodies,
line endings, layout, positioning tokens and order survive in the decoded target
string. Negative/invalid timing is rejected. Before serving, the artifact is
parsed again, text and non-timestamp content must match exactly, every timestamp
must equal the fitted affine model, and actual occupancy is measured again.

## Integration and cache identity

The work executes in `asyncio.to_thread` so CPU analysis does not run directly in
the async request loop. Existing reference candidate discovery, content SHA-256
deduplication, ALASS retry/budget handling and original-subtitle fallback remain.
Discovery admits a reference with plausible normalized duration to verification
even when its first dialogue lies outside the legacy 20-second execution window;
this is a ranking/admission hint and grants no trust or identity.

Affine artifacts carry `timeline_affine/timeline-affine-v2`, original reference
kind, reference SHA-256, slope and intercept. The resolution namespace is now
`v6-timeline-affine-v2`, and the sync verdict engine is version 3. Old rejection
and payload decisions cannot suppress the new strategy. Provider subtitle and
reference caches are not globally invalidated. Verified payload and verdict
fast paths reuse subsequent requests.

One structured outcome line records cue counts, occupancy run counts, landmarks,
scored/tested hypotheses, coefficients, inliers/outliers, before/after scores,
window agreement, local residual offsets, classification, acceptance and reasons.
Delivery logs record reference digest/provenance and measured serving state.
No subtitle dialogue or credentials are added to these logs.

## Validation status and limitations

Synthetic tests cover constant shift, generic drift, both PAL directions, mixed
segmentation, duplicates/overlaps, short/large gaps, metadata, duration noise,
outlier landmarks, wrong references, missing scenes, mid-film discontinuities,
damaged middle with intact endpoints, text integrity and cache reuse. Eleven new
mutations test winner selection, improvement, cue count independence, distributed
evidence, metadata, text, verdict delivery and MovieHash provenance/proof.

All three real Whiplash files are preserved under `tests/fixtures/whiplash`.
The required real regression independently recovers slope 1.042755149812734 and
intercept -8621.594569288398 ms. All 825 starts match Subtitle Edit exactly;
614 ends match exactly and all ends are within 1 ms. Target text and ordering
are unchanged. The oracle is opened only after generation.

Live Stremio playback against the rebuilt container also succeeded using a
different SubDL FraMeSToR UHD HYBRID REMUX Arabic reference (926 cues). Timeline
Affine recovered slope 1.0427472775355806 and intercept +2688.544928596355 ms,
improved the score from 0.4356 to 0.9579, and passed all eight windows with zero
local offsets. Delivery was verified and synchronized with transformed output;
ALASS did not run, and the user confirmed synchronization during playback.

Eight-window diagnosis is bounded temporal evidence, not proof of exact-file
identity or arbitrary edit detection. Brief localized edits and indistinguishable
occupancy patterns remain limitations. SRT-compatible complete references only;
partial references and insufficient evidence defer to existing paths. Piecewise
sync is not implemented, but can reuse the timeline and landmarks.
