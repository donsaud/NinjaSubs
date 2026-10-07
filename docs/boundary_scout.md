# Offline boundary candidate scouting

`tools/boundary_scout.py` reads existing provider cache sidecars and pristine SRT
payloads, capture artifacts, and explicitly hash-bound real replay inputs. It
never searches providers, downloads subtitles, accesses synchronization caches,
imports cases, or updates evidence manifests. Dexter S08 and IMDb tt0758758 are
excluded. Missing metadata, conflicting episode identities, season packs,
unknown release families, duplicate hashes, timing-only fixtures, derived cases,
and generated outputs cannot become inputs.

Run with the repository Python environment:

```powershell
python tools/boundary_scout.py --capture-root <existing-capture-directory> --output-dir <external-report-directory>
```

Use `--alass <executable>` if the production executable is not available on PATH.
Reports must be outside the repository. No changes to `.env` are needed. Execution
is sequential; the production wrapper owns subprocess timeouts and removes its
temporary reference/target/output files. The scout uses private temporary report
storage and retains only JSON and Markdown reports. Production logging is
suppressed in this standalone process to avoid exposing dialogue or subprocess
stderr. Report metadata is allowlisted, rejects URLs/paths/credential-like labels,
and omits embedded episode titles. Errors expose exception types only.

Optional `--payload-manifest` accepts a local JSON object containing `payloads`.
Each entry must contain `provenance: real`, `role: target` or `reference`, a portable
relative `path`, original byte `sha256`, `imdb_id`, `media_type`, `season` and
`episode` for TV, `provider`, and `release_name`; `sub_id` and `lang` are optional.
Only provider-origin pristine payloads belong here; asserting `real` is not a way
to authorize generated or manually edited inputs. Absolute paths and traversal
are refused. This manifest is a scouting input, never a formal evidence manifest.

The pipeline imports production byte decoding, cue parsing/projection, Timeline
Affine, cue-sanity and Large Offset gates/investigation, intro-credit stripping,
alignment consistency and anchor pre-shift. Disc reference release metadata
supplies the proposed stream family; no stream availability or MovieHash identity
is asserted. Accepted affine results stop as `GLOBAL_SOLVED`; already aligned
results also stop. Otherwise the existing `SubtitleSyncService.sync` performs its
normal cleaning, eligibility checks, argument construction (`reference, target,
output`), execution, and output validation. Its existing capture hook is observed
through a request-local **memory-only** receiver; no capture directory or REAL
metadata is created. Raw producer bytes are hashed before the production
universal-newline representation is passed to the independent legacy and boundary
verifiers. Reference trust is not fabricated. Output rejected by the service
cannot be ranked as an EDGE candidate, even if observational boundary evidence
is accepted. Original files are never rewritten; production preprocessing affects
only decoded working strings.

The JSON contains original source hashes, parser counts, affine diagnostics,
actual subprocess launch status, Large Offset diagnostics when applicable,
legacy verdict, boundary classification/reasons, support, residual/reference
p95, coverage, windows, holdout results, gap disagreement, and runtime. Boundary
`gap_disagreement_ms` is the requested gap-error measurement; segment fields are
stored under `boundary.segments`. Ranking prioritizes accepted EDGE results,
legacy failures, two regimes, unchanged cue population, holdout, bilateral
support, coverage, and low residuals. Failed/unavailable subprocesses and global
solutions are excluded from the top-candidate table.

**Every output is DERIVED_SCOUTING_ONLY.** An accepted EDGE result requires an
independent real Stremio request with Capture V2 before any REAL evidence claim.
The tool neither makes a production promotion decision nor performs that import.
