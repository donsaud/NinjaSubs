# Isolated real-subtitle intake

Run from the repository's Python environment, selecting a directory outside the
repository and outside cache, capture, and formal replay directories:

```powershell
python tools/boundary_scout_intake.py --container ninjasubs --output-dir <isolated-output-directory>
```

The tool first checks the existing container's configured production ALASS with
`--version`. Docker failure or a nonzero version check exits 2 before acquisition;
unavailability is never counted as an ordinary skipped ALASS pair. It copies only
Python source into a unique `/tmp/ninjasubs-boundary-scout-*/snapshot` directory,
executes in a separate process using the container's existing dependencies,
ALASS, and configured provider environment, and removes that temporary workspace
after a successful copy. On execution/copy failure any acquired corpus remains
under that unique temporary workspace for recovery; only the snapshot is removed.
There is no image build, service restart, deployment, `/app` edit, `.env` copy or
change, synchronization-cache access, or evidence-corpus import. Logs are
disabled in the isolated process; credentials and provider URLs are never
persisted or printed. Provider auth/circuit state stays in this separate process.

The explicit queue contains 24 ordinary episode identities across six series.
It excludes Dexter S08, tt0758758, specials, and the globally solved S02E02
scouting case. Exact requested episode tokens must agree with release metadata
and any selected ZIP member tokens. Titles are never used as identity fallbacks.
Extended/director/unrated/special-edition and ambiguous multi-episode labels are
refused. Providers without configured API keys are omitted. Each exact IMDb and
episode is searched through the existing SubDL, SubSource, and OpenSubtitles
providers for Arabic targets and English disc references. All queued identities
are considered; the tool does not stop on the first EDGE result. Each identity
retains at most three distinct targets and three references; requests/downloads
are sequential and paced. The queue can exceed 20 identities and 50 cross-release
pairs when safe provider data permits; neither number is fabricated or forced.

`intake/<source-sha>/source.srt` contains exact downloaded SRT bytes or exact ZIP
member bytes. `download.bin` retains the complete original download. Intake never
transcodes, normalizes, rewrites, or retimes these files. Hash directories are
published from private staging directories; existing source directories cannot
be overwritten. Nonempty previous intake requires a new output directory.
SHA duplicates are deduplicated, and conflicting media claims for the same SHA
remove that SHA from the usable manifest. Invalid/failed downloads do not publish
corpus entries. Archive and member sizes are bounded; filesystem member paths
are never extracted. RAR/7z/gzip and non-SRT inputs are refused rather than
silently losing source-byte provenance.

`intake-manifest.json` records provider and listing/archive ID, exact release,
language, exact IMDb/episode query, selected member label, source/download SHA,
timestamp, and relative source path. URL/token/config fields are excluded. The
manifest describes provider-origin **inputs**, not REAL verification evidence.

The existing scout receives only this manifest with an empty inventory root,
so production caches and the formal replay corpus are not even enumerated.
Production parser/projection, Timeline Affine, ALASS eligibility/execution,
legacy verification, and boundary verification keep their existing order.
Affine successes remain GLOBAL_SOLVED. Reports include acquisition and scan
counts, boundary rejection histograms, and the existing ranked top ten. Near
misses retain their rejection classification. Every result remains
DERIVED_SCOUTING_ONLY and requires independent live Stremio plus Capture V2
confirmation. No scouting result can satisfy or modify the evidence gate.

Results are `intake-manifest.json`, `boundary-scout-expanded.json`, and
`boundary-scout-expanded.md`. A blocked invocation writes explicitly marked
NOT_COMPLETED diagnostics only if those output files do not already exist;
its zeros describe no completed acquisition/scan, not verified negatives.
