# Boundary-aware verification — shadow only

`boundary_verifier.verify_boundary` evaluates three immutable texts: the pristine
request target, a reference, and a fixed transformed artifact. It never renders
timestamps or supplies an authorization. Its `BoundaryResult` is separate from
`SubtitleEvaluation` and cannot promote a production verdict or cache entry.

## Evidence

The verifier retains original file order, checks positive timings and start/end
ordering, then uses the existing usable-dialogue predicate to exclude metadata.
Target/output must retain the same usable-cue population in order. This is a
fixed-output verifier, not a matcher for independently resegmented targets.
Reference segmentation may be 1:1, 1:N or N:1; no common dialogue language or
greedy one-to-one reference pairing is required.

There must be exactly one meaningful offset transition, with at least 20 usable
cues and 60 seconds of dialogue span per segment. Both start and end movements
must agree with their segment median within 350ms, including the maximum error,
so a small third regime cannot disappear beneath p95. The two-constant model
must improve aggregate absolute error by at least 90% over one constant.

This conservative first shadow version supports upward edition jumps only.
Any downward jump, invalid chronology or cue crossing the boundary is refused.
It does not model arbitrary edits, removed dialogue, or K-segment cuts.

Independent reference evidence includes:

- A consecutive pair of reference dialogue runs around the proposed boundary.
  Its endpoint errors total at most 2000ms, and the observed gap difference
  agrees with the offset jump within 1000ms.
- Occupancy union with the existing 250ms run join and 350ms padding. Each side
  requires at least three runs, strong coverage in both directions (90%), and
  four distributed windows scoring at least 0.80.
- Distributed run starts: nearest reference-run-start p95 at most 1000ms.
  This measurement permits reuse and is supporting evidence, never the sole
  proof. Starts and gaps complement occupancy where cue ends differ by splitting.
- No consecutive poorly supported dialogue region spanning more than 5000ms.
- A holdout over alternating whole dialogue runs. Offsets are fitted only on
  training runs, and independent reference gap evidence resolves the training
  boundary ambiguity. At least eight training and eight withheld cues on each
  side must support the model; all withheld movements remain within 350ms and
  their forward reference occupancy coverage remains at least 90%.

These are isolated shadow gates. The legacy 2000ms p95 threshold, Timeline
Affine, ALASS, PAL and Large Offset rules are unchanged. Reference timing is
evidence conditional on the existing identity/reference selection; these tests
do not establish video or episode identity from occupancy alone.

## Integration and lifecycle

The orchestrator keeps an immutable pristine target snapshot before intro
stripping or pre-shifting. Immediately after its normal post-ALASS analyzer
returns, it schedules a narrowly eligible shadow check for a useful ALASS output
whose production verdict is piecewise or low confidence. Verified production
results skip this work; global Timeline Affine returns before this integration
point. The standalone API also supports `global_verified=True` to refuse
unnecessary escalation explicitly.

The orchestrator does not await the shadow job or consume its result. Each
owned job uses a private single-worker executor, separate from the shared ALASS
executor and its semaphores. At most one job per orchestrator is outstanding;
busy candidates are skipped rather than queued. Input limits (500000 characters
per text, 4000 cues) and cooperative 250ms computation checkpoints bound work.
A checkpoint can refuse an otherwise valid case on a slow machine; it cannot
change serving. Worker threads do not inherit mutable request telemetry context.

Cancellation signals a thread event checked at computation checkpoints. The
async owner shields and drains the actual worker even after repeated task
cancellation, and retains its slot until the worker exits. Its private executor
is then closed/joined. Thread work is not forcibly interruptible between
checkpoints, so input bounds and cheap searches still matter. Shutdown closes
admission, cancels/drains jobs, and is idempotent. Task completion callbacks
consume unexpected exceptions as well as releasing retained references.
Scheduling without a running event loop does not create an unawaited coroutine.

Logs contain only fixed reason/classification labels, numerical evidence and
`artifact=raw_transformed`. No subtitle bodies, URLs, credentials, configuration,
or exception messages are logged. Scheduling/evaluation failures are isolated
from delivery. Background CPU work may still contend for resources; synchronous
blocking and unbounded queues are avoided, not claimed impossible.

## Dexter and final-artifact distinction

The timing-only fixture preserves the exact known Desktop capture timestamps
and their existing dialogue flags, with input filenames and full file hashes.
It contains no real dialogue, credentials or local machine paths.

Dexter supports a division between target cues 39/40, within the silent interval
187250–193890ms. The dialogue offsets are -96961ms and +9661ms, with a 106622ms
jump and 58ms reference-gap discrepancy. Support is 38/739 usable dialogue cues,
four/124 target runs, with zero movement residuals. Alternating-run holdout
contains 15/352 withheld cues, also with zero movement residuals.

Legacy verification remains `p95=4489ms`, structural similarity 0.72,
`cut=piecewise`, `sync_state=unverified`, `verification=unknown`. The new result
is only a shadow `EDGE_DISCONTINUITY` acceptance.

Response cleaning shortens the first metadata cue's end from 10590ms to 10199ms,
clamping its overlap with cue 2. It changes no dialogue timing. Integration
evaluates the raw transformed artifact, not the later HTTP response body.
The standalone verifier accepts an explicit `artifact=final_served` label;
both versions are covered by fixture tests. Response processing is unchanged.

Whiplash remains positively verified by global Timeline Affine, subtype
`PAL_LIKE_AFFINE`, and does not invoke piecewise complexity. The simplest
positively supported timing model takes precedence.

## Before any production promotion

Keep this observational until multiple editions and negative media cases have
been checked, including wrong episodes, dense unrelated dialogue, removed or
missing dialogue, variable boundaries and transformed-content integrity. Review
CPU contention and skip rates under concurrent load. An accepted shadow result
does not verify episode identity or prove every isolated cue correct against
video. Production acceptance, cache authorization and delivery policy require
a separately reviewed milestone.
