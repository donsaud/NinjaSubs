# Offline Boundary Shadow Corpus

Run from the repository root with `python -m tools.sync_replay --output REPORT.json`.
Choose an output outside the repository; generated reports are not fixtures or source.
The command exits nonzero on expectation mismatches or false positives. It never
calls providers, executes ALASS, accesses a sync cache, or invokes the serving path.
No application module imports the harness. Production source is unchanged.

For validated local artifact import, provenance contracts and collection guidance,
see [Real shadow case collection](shadow_case_collection.md). The current corpus
explicitly labels two real seeds and eleven derived controls; summary reporting
separates REAL/DERIVED/SYNTHETIC counts. Run
`python tools/sync_replay.py --corpus-summary` to inspect them without saving a report.

## Inputs and evidence

`tests/replay/corpus.json` contains portable repository-relative fixture paths,
release-family metadata, expected model, shadow outcome/reason, and whether the
simpler-model check should allow a boundary diagnostic. A fixture is either an
SRT or an existing timing-only JSON with a named `role`. Absolute paths, traversal
and symlink escapes are refused. No provider URLs or credentials belong here.

Dexter reuses the production capture timing surrogate, with synthetic dialogue
and frozen usable-dialogue flags. Whiplash reuses target/reference/expected SRTs.
Whiplash expected is a fixed transformed artifact for verification, never an
input to affine fitting. Optional `served` is evaluated separately and labeled
`final_served`; the replay does not recreate HTTP cleanup or deliver bytes.

Each case runs the existing `AlignmentAnalyzer` over target/reference/output,
recording its full measured production verdict without changing it. Recorded
measurement context is explicit in the manifest: ALASS applied/successful,
reference trust, and Dexter's existing 180000ms Large Offset measurement ceiling.
This is measurement of an already captured output, not executing ALASS, identity
validation, the complete orchestrator, or a new production authorization.

The pure existing Timeline Affine fit uses target/reference only. A successful
fit supports the fixed output only when every start/end agrees within 350ms and
cue populations match. That replay comparison is conservative and observational;
it is not a production verifier gate. A reference fit cannot hide a different
fixed artifact. Positive production or supported affine evidence suppresses
boundary escalation. Whiplash's legacy analyzer can remain unverified while the
separate existing affine verifier is positively verified; neither is overwritten.

Both guarded and ungated boundary evidence are recorded. Ungated checks ensure
normal constant/affine/segmentation controls cannot claim a boundary even without
the simpler-model guard. `boundary_should_run` describes this replay's model
selection diagnostic, not live scheduler eligibility or an ALASS launch decision.

Evidence includes offsets, segments/support, boundary interval, residual p95/MAD,
reference run-start residuals, occupancy/coverage/windows, gap discrepancy,
holdout, reasons, production verdict, affine classification, and decoded input
SHA-256 values. Reports contain no subtitle bodies or local input paths.

Evidence and aggregate counts are deterministic; repeated replay tests compare
them exactly. Wall-clock durations live in a separate `observations` object and
are intentionally not deterministic. Add `--evidence-only` for deterministic
JSON without durations. The existing cooperative 250ms boundary
budget remains enabled. Slow hardware may report `budget_exceeded`; replay does
not fake the clock or relax the budget. This refusal is a mismatch, not success.

## Corpus and extension

There are two known real seeds, not thirteen independently observed productions.
The remaining eleven cases are declared deterministic timing variants:
wrong reference, sparse population, tiny segment, unsupported reference boundary,
three regimes, non-monotonic output, crossing cue, removed reference dialogue,
normal constant offset, global affine, and segmentation-only same edition.
Negative fixtures must fail; their recorded reason is checked, not inferred from
their names. A removed reference interval can fail a coverage gate before the
unsupported-interval gate; it is not a claim of unique internal gate coverage.

To add another real case, add portable fixture files and one manifest entry with
target/reference/transformed and optional served. No case-specific Python is
needed. Optional deterministic `operations` support `slice`, `drop`, `shift`,
`scale`, `end`, `swap`, and `split`. Indices are zero-based complete cue indices;
`stop` is exclusive except `swap` (second index). Mutations parse once and render
without sorting afterwards; deliberately broken chronology remains visible.
Derived cases are controls, not new real playback evidence. Use fresh IDs,
provenance notes, realistic expectations and recorded measurement context.

Aggregate positives/negatives are for boundary acceptance, not all synchronized
subtitles. Whiplash is a negative boundary control and a positive affine seed.
False positives and unnecessary escalations must remain zero. More real intro,
recap, credits, mid-episode edit, wrong episode and multiple-cut cases are still
needed. Passing this corpus supports further shadow data collection only; it
does not establish readiness for production positive verification.
