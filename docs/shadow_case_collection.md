# Real shadow case collection and import

This workflow admits captured artifacts into an **offline** corpus. It never
exports live data itself, executes ALASS, fetches providers, changes a verifier,
authorizes a cache entry, or changes a Stremio response. Keep live shadow as-is.
Passing an import means its declared offline behavior was reproduced, not that
the subtitle was verified against video or approved for production serving.

## Collect the right artifacts

Save a controlled local capture of:

1. The pristine target before intro stripping or pre-shifting.
2. The actual selected reference, not a similarly named replacement.
3. The raw transformed output produced for that same request.
4. Optionally the exact final-served subtitle after response processing.

Raw transformed is preferred for verification. Response cleaning can change
metadata cue ends or formatting; do not replace raw with final-served silently.
Use existing authorized diagnostic files/captures. Log metrics alone do not
reconstruct the subtitle artifacts. This milestone adds no debug-export hooks;
if exact artifacts are unavailable, do not invent a triplet or claim a real case.
Keep originals outside the corpus until validation finishes.

Never capture provider URLs, signed links, request configurations, tokens, JWTs,
or credentials in metadata. Do not store user-home or absolute input paths.
The importer does not retain the supplied local source filenames. Provenance
labels should describe editions, such as `HDTV EVOLVE` and `BluRay DEMAND`.
Optional metadata JSON accepts only `target_release_family`,
`reference_release_family`, `media_type` (`movie`/`series`), `season`, `episode`.
No arbitrary request configuration, notes or extra fields are accepted.

## One repeatable import command

Run from the repository root (direct script and `python -m tools.import_sync_case`
are both supported). Use local relative paths, for example in PowerShell:

```powershell
python tools/import_sync_case.py `
  --case-id dexter-s08e04-hdtv-vs-bluray `
  --target incoming/target.srt `
  --reference incoming/reference.srt `
  --transformed incoming/transformed.srt `
  --served incoming/final_served.srt `
  --expected-model EDGE_DISCONTINUITY `
  --expected-boundary-accepted true `
  --expected-parent-model-failure true
```

The example is for a newly collected triplet, not another copy of the existing
Dexter seed. A Whiplash-like case uses `--expected-model GLOBAL_AFFINE`,
`--expected-parent-model PAL_LIKE_AFFINE`, `--expected-boundary-accepted false`.
An ordinary constant shift uses `CONSTANT_OFFSET`. Available rejection labels:
`DIFFERENT_CUT_REJECT`, `WRONG_REFERENCE_REJECT`, `MULTI_DISCONTINUITY_REJECT`,
`INSUFFICIENT_EVIDENCE_REJECT`; all require explicit `false` boundary acceptance.
These are corpus labels only, not new production enums.

Optional flags include `--expected-boundary-run true|false`,
`--expected-parent-model-failure true|false`, `--expected-reason REASON_CODE`,
`--metadata incoming/labels.json`. Boundary run defaults to false for ordinary
global/constant models and true for edge/rejection models. Edge acceptance must
be true; every other category must be false. Expectations are supplied before
evaluation, never inferred afterwards to turn a failure into PASS.

## Admission and privacy gates

Input files must be UTF-8 (BOM allowed), nonempty, bounded by the existing shadow
input limits, parseable in every block, with non-negative valid start/end stamps
and monotonic start/end order. Overlap is permitted if order is preserved. The
parser does not sort broken chronology into validity. Formatting-only Unicode
trailer blocks are harmless and ignored for validation; original bytes remain
unchanged. Non-UTF-8 files must be converted locally and that representation
recorded honestly; the importer does not silently guess encodings.

Target/output dialogue populations and ordered text must agree after whitespace
and HTML-tag normalization using the existing usable-dialogue predicate.
Metadata-only cue differences are allowed. Other dropped/reordered/changed
dialogue is refused, including in optional served output. This conservative
quality gate does not attempt fuzzy content identity or episode recognition.

Identical target/reference bytes are refused unless
`--allow-identical-target-reference` is explicitly supplied and recorded. Known
credential patterns, URLs, local paths and NULs in artifacts are refused.
Metadata additionally uses a field/character allowlist and refuses long opaque
tokens. Credential scanning removes Unicode formatting characters and uses
NFKC for inspection only; the stored subtitle is never redacted or rewritten.
These checks catch known patterns, not every conceivable opaque secret. Review
captures manually before collection; never relabel private data as safe to get
past admission. A rejected artifact must not be silently edited and represented
as the pristine capture. Keep non-speech privacy sanitization provenance clear
if an independently reviewed surrogate is needed in a later task.

Invalid inputs, missing expectations or replay mismatch exit nonzero and do not
register a case. Errors use safe fixed reasons and do not print input paths,
subtitle text or private exception details. A failure reports the violated
expectation fields and actual shadow reason. Successful import prints case ID
and PASS. No output is treated as production authorization.

## Storage, SHA identity and transactions

Case IDs are explicit lowercase slug IDs (hyphens/underscores allowed), bounded
and safe on Windows. Case directories are deterministic:

```text
tests/replay/cases/<case_id>/
  target.srt          # only if not already stored/shared
  reference.srt
  transformed.srt
  final_served.srt    # optional; may share another artifact
  case.json
  diagnostic.json
```

SHA-256 is computed over exact input bytes; cue counts and reuse flags are
recorded in `artifact_metadata`. Byte-identical artifacts under `tests/fixtures`
or `tests/replay/cases` are reused by relative fixture path. Duplicate roles in
one import also share the first artifact. No silent proliferation of copies.
Different encodings/newline bytes deliberately have different raw hashes.
Replay's existing `input_sha256` is a decoded-text hash, separately labeled;
it must not be confused with the raw artifact identity hash.

`case.json` is deterministic safe metadata plus the manifest entry. The corpus
manifest is updated by appending that entry only after immediate replay passes.
`diagnostic.json` contains hashes/counts, production verdict, guarded/ungated
shadow evidence, classification/boundary/offsets/support/coverage/residuals/gap/
holdout and runtime observations; it contains no dialogue or absolute paths.
Runtime is observational and variable; evidence remains deterministic.

Imports use a corpus lock and refuse a competing importer. Artifacts stage in
an owned temporary directory; the manifest replacement is atomic and failure
rolls back only the new case. Existing cases and unrelated files are never
overwritten. Manifest edits detected during evaluation cause refusal. Existing
IDs/directories are refused: repeated commands cannot silently alter a case.
If a process crashes, inspect incomplete owned artifacts and the lock manually;
the tool never removes somebody else's lock or silently cleans crash remnants.
Ordinary exceptions are rolled back, but this is not a crash-durable database
transaction across directory and manifest filesystem operations.

## Provenance and summary

Each case declares `provenance=real|derived|synthetic` and `source_case_id`.
Real capture cases use null source IDs. Derived cases require an existing source
case and `--provenance derived --source-case-id SOURCE_ID`. Synthetic cases are
explicitly synthetic; do not declare generated examples real. Real is an honest
human provenance assertion, not something timings alone can prove.

The current corpus has **2 REAL seeds, 11 DERIVED controls, 0 SYNTHETIC cases**.
Dexter's real seed is represented by a timing-only surrogate of its captures;
Whiplash uses real SRTs. These are not thirteen independent real productions.
The tests import into temporary roots; they add no new claimed live captures.

```powershell
python tools/sync_replay.py --corpus-summary
```

Summary rows show ID, provenance, category, expected/actual boundary acceptance,
classification, accepted, reason, duration and PASS/FAIL. Expectations include
parent model/failure, boundary-run/required decision and classification. Any
mismatch fails the command/test. `--output REPORT.json` optionally writes full
evidence; choose an output outside the repository. `--evidence-only` omits
variable runtime observations. REAL, DERIVED and SYNTHETIC confusion counts are
reported separately; the compatibility combined aggregate is not a real-case
headline. Counts concern boundary acceptance, not whether every subtitle is
synchronized. Whiplash is a real boundary-negative, positive affine seed.

The importer records the production analyzer with its default context and flags
indicating an already supplied transformed artifact. It neither executes ALASS
nor infers reference trust from a filename. Existing seed measurement context
is preserved in the manifest; imported results may differ from that context.
The replay simpler-model diagnostic remains separate from live scheduling.

## Collect next

Prioritize independently labeled real BluRay/HDTV intro differences, inserted/
removed recaps, credits differences, mid-episode edits, wrong-edition reference,
multiple discontinuities, ordinary global affine and constant-offset cases.
Include genuine negatives and trustworthy video/playback checks. Do not change
thresholds to make captures pass. The outcome remains **READY FOR REAL CASE
COLLECTION**, not readiness for production positive verification promotion.
