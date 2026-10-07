# Selected live synchronization case capture

This diagnostic is OFF by default. It does not authorize synchronization,
modify response bytes, change provider discovery or cache reads/writes, execute
additional ALASS runs, or promote boundary shadow results. There is no public
route, request token, or user-config switch that enables it. No Dexter-specific
export hook remains in the baseline; this is a general replacement workflow.

## Explicit server-side selection

Set `NINJASUBS_CASE_CAPTURE_SELECTION` in the server process environment to a JSON
array. Every selector requires all five fields: case_id, imdb_id, season,
episode, and the **full SHA-256 of exact sync-ingress target bytes**. Movies use
null season/episode. Auto-sync must be requested. Unselected requests do no
capture filesystem work. No `.env` edit is needed or performed by this feature.

Example selectors for original files in the supplied season archive:

```json
[
  {
    "case_id": "dexter-s08e02-asap-live",
    "imdb_id": "tt0773262",
    "season": 8,
    "episode": 2,
    "target_sha256": "9085f245803b096809d2f7079c557d4d09211feb14022d8f96cdb3a0af4cb916"
  },
  {
    "case_id": "dexter-s08e06-evolve-live",
    "imdb_id": "tt0773262",
    "season": 8,
    "episode": 6,
    "target_sha256": "a88d2662cc933f861aafe97dc800d1d99bebd29d1d7dac435a724a0e22ccff96"
  }
]
```

These identify candidate bytes; they do not establish reference identity,
video compatibility, expected outcome, or provenance=real. Extraction,
transcoding or provider updates may change ingress bytes: inspect safe target
hashes first, then explicitly select the actual bytes. Do not replace the full
SHA condition with a prefix or wildcard. Converted ASS captures the converted
sync-ingress SRT; native ASS pass-through is not selected for sync capture.

Default root: `/tmp/ninjasubs_case_capture`. Optional server environment variable
`NINJASUBS_CASE_CAPTURE_ROOT` chooses a trusted private capture directory. This
is separate from production cache. The existing Compose file does not forward
these new variables: an operator must explicitly supply them to a running
process/image in a separately authorized deployment. Setting a host shell
variable alone does not enable the existing container. No deployment is part
of this milestone. Disable/remove the selector after controlled collection.

## Artifacts and identities

Each request gets `<case_id>/<random_run_id>/` so simultaneous requests never
overwrite one another. Private directory/file modes are used where supported.
Writes use same-directory temporary files and atomic replacement; the whole
capture is not a database transaction. A failure can leave a partial capture.

- `pristine_target.srt`: exact sync-ingress bytes before intro stripping or
  pre-shift. No repair, redaction or sorting is applied.
- `attempt-NN/selected_reference.srt`: actual resolved reference text, UTF-8.
- `attempt-NN/raw_transformed.srt`: exact ALASS output-file bytes when that
  execution was observed; otherwise the exact UTF-8 transform returned by the
  affine/PAL/orchestrator path, explicitly distinguished in metadata.
- `attempt-NN/verifier_transformed.srt`: present if universal-newline decoding
  changed the raw ALASS representation. Its own SHA binds verifier/shadow and
  orchestrator-output observations; raw SHA is never substituted for it.
- `attempt-NN/alignment_input.srt`: actual ALASS input after service sanitation,
  if different from the pristine target. `alass_reference_input.srt` records
  the sanitized reference supplied to ALASS. These are observational copies.
- `final_served.srt`: exact constructed SRT response body after cleaning. VTT
  or ASS responses retain their actual extension and bytes instead.
- `metadata.json`: safe identities, full content hashes, release labels,
  provider name when directly observed (otherwise null), reference kind,
  strategy, numerical/enum verifier evidence, affine attempt diagnostics,
  actual scheduled/completed/skipped boundary-shadow observation and timestamp.

Each transformed attempt is retained even when rejected by the service or
orchestrator. Root `selected_reference.srt` and `raw_transformed.srt` aliases are saved only
for an attempt matching the actual response input. `selected_attempt` records
the orchestrator winner; `response_selected_attempt` identifies the attempt
actually used by response construction. Explicit winner observations distinguish
references producing identical output. Without that observation, ambiguous
outputs do not imply a winning reference; inspect all matching attempts. Original fallback can have rejected raw artifacts only
under attempt directories. Do not confuse those with the actual final body.

`response_input_sha256` binds the bytes entering the response pipeline before
cleaning; `final_served_sha256` binds its body afterward. A detached job may
finish after original fallback. `orchestrator_completed_after_response` marks
this, and its completion cannot overwrite response input/final body evidence.
`artifact_sha256` maps relative artifact filenames to hashes of their exact
bytes, including newline-distinct ALASS representations and root aliases. Verify
this manifest after writes finish and again after copying. Construction is not
proof of network receipt or correct playback.
`final_artifact_verified=false` explicitly avoids granting new trust.

The existing single-flight/cache behavior is unchanged. A joining request or
cache hit can capture only ingress/final bytes because it does not independently
resolve a reference or execute a transform. Missing observations stay missing;
capture never forces a cache miss or repeats work to fill them. Background
shadow results can update metadata later; wait for completion before copying.

## Privacy, failure and bounds

No full request/config object is serialized. Credentials, provider/config URLs,
tokens, paths and free-text verifier reasons are not copied into metadata.
Known unsafe URLs/credentials in subtitle content cause export refusal, not
redaction or delivery changes. This conservative check can refuse captures
with promotional URLs. It detects known patterns, not every possible secret;
review artifacts manually before import. Capture code emits no log messages.

Each artifact is limited to 2 MB; each request retains at most 16 transform
attempts, reference identities and provider identities. One lazily started
process-wide writer thread handles filesystem operations through a queue of
at most 64 pending writes. Requests never wait for that queue or disk writes.
Backpressure or write failure abandons the affected capture, preserving delivery.
There are no additional provider calls or per-request background tasks. Existing
owned boundary workers retain their original lifecycle. Normal shutdown closes
producers before allowing up to five seconds for the capture writer to finish;
a blocked writer cannot cause replacement threads to accumulate. Per-file atomic
replacement prevents truncated committed files, but process termination or slow
storage can leave an incomplete capture or temporary file. Export is best effort,
not crash-durable. Manage disk retention separately.

ALASS output is read once: selected capture reads its raw bytes and reproduces
the existing universal-newline text decoding; disabled capture retains the
original text read path. No ALASS inputs, arguments or synchronization decisions
change. Copy captures before recreating the container: its default `/tmp` root
is ephemeral. Disable selection after controlled collection.

## REAL admission and evidence gate

Play S08E02 on a BluRay stream with the ASAP subtitle, and S08E06 with the
EVOLVE subtitle. Capture the exact reference actually chosen. Independently
check dialogue before/after any suspected edit and throughout playback. Record
the real video/release pairing and generator observation, with no URLs/tokens.
The previous S08E02 triplet remains UNKNOWN until documented reproduction.
Compare new raw SHA to the old unknown output separately from newline-normalized
verifier SHA; neither a mismatch nor similar timings establishes provenance.

Use `tools/import_sync_case.py` only after independent expectations are set.
Do not bypass its strict parser, privacy, population/order or replay checks.
The old S08E02 pristine target currently fails that parser in five malformed
blocks. Capturing it faithfully does not make it admissible; do not silently
repair it and label the modified bytes a pristine live capture.

No production promotion is allowed until there are at least two independent
REAL edge positives and two meaningful REAL rejection cases, FP=0, no
unnecessary escalation, Dexter preserved and Whiplash PAL-like affine preserved.
This feature collects evidence only and does not change that gate.


## Forward-only Capture V2

New captures have `schema_version=2`. `pristine_target.srt` continues to mean
exact sync-ingress bytes, never a canonical replacement. Existing V1 files are
not rewritten or automatically upgraded. Each completed ALASS boundary-shadow
attempt can additionally record:

- `boundary_target_input.srt`, `boundary_reference_input.srt` and
  `boundary_transformed_input.srt`: exact strings passed to that shadow worker,
  UTF-8 encoded, separate from ALASS sanitation and raw producer newline bytes.
- `boundary_parser_projection.json`: file-order start/end/usable-dialogue flags
  for all parsed cues, production parser implementation hash, original target
  hash and each exact verifier input hash. It contains no dialogue.

The projection uses the production parser per block and the same dialogue
eligibility predicate as the boundary verifier. Empty/untimed blocks are
omitted exactly as production does; ambiguous timing blocks, invalid timing or
unsafe order refuse diagnostic projection. Its construction, decoder proof,
source hashing and file writes run on the existing bounded writer, after the
shadow result. It neither changes the shadow worker inputs/results nor the
request budget. Failure means incomplete evidence; delivery continues.

`parser_language` is a bounded language identifier used to regenerate the
production decoder result from original ingress. Decoder output must equal the
captured shadow target. Selected reference must equal shadow reference; raw
ALASS output under universal-newline decoding must equal shadow transformed
input. The importer regenerates every projection and refuses parser drift.
Parser fingerprint normalizes source line endings only, allowing equivalent
Windows/container source files. Other implementation changes invalidate import
until explicitly reviewed; no automatic migration is allowed.

Use `tools/capture_v2.py` for new, complete V2 ALASS captures only. Import requires
explicit independent REAL origin and expected-outcome confirmation, exact
hashes, completed shadow result, non-positive legacy verification, projection
regeneration, and direct/projected/live shadow verdict equality. It copies
original files unchanged, retains raw/final separately and atomically publishes
only validated evidence. The normal strict SRT importer remains unchanged.
Neutral projection text is used only to reproduce boundary structural semantics;
legacy/global verification is never recomputed on fabricated dialogue. Recorded
legacy state is evidence, not new authorization. Final-body hash validation is
not final-body synchronization verification.

An explicit future import, with independently established expected rejection:

```sh
python tools/capture_v2.py <capture-directory> --attempt attempt-01   --case-id <portable-case-id> --real-live-confirmed   --expected-boundary-accepted false --independent-outcome-confirmed   --meaningful-negative-confirmed
```

For an independently labeled positive use expected `true` and omit the negative
flag. No capture is counted as REAL solely because its metadata says live.
Use the evidence gate policy in `docs/boundary_evidence_gate.md`. Selector
activation is still one exact case at a time; no wildcard capture or cache
invalidation is introduced. Existing cache hits can lack fresh worker inputs.
