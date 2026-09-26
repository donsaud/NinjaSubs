# Autosync audit — 2026-09-26

## Findings and fixes

| Finding | Correction |
| --- | --- |
| The OpenSubtitles serving route skipped autosync. | Apply the same preference and sync flow to cached, freshly downloaded, and fallback payloads. |
| Shared listing metadata could overwrite the current playback filename/hash. | Request playback context takes precedence; current user credentials also apply to cached raw subtitles. |
| Listing dropped the title/year and stream URL needed by reference providers. | Retain title/year and propagate supplied stream URLs through serving URLs. |
| Reference files were keyed only by title, episode, and release group. | Versioned identities include filename, video hash, size, stream URL, media type, and language. Old group-only entries are misses. |
| A shared release group bypassed checks for different cuts/sources. | Reject explicit source conflicts during reference selection and check cuts/sources before skipping alignment. |
| An edition cache write deleted the metadata of retained exact references. | Preserve their metadata; use unique temporary files and atomic replacement. |
| Concurrent cache writes could pair one payload with another verdict. | Bind verdict metadata to the payload checksum; partial or inconsistent entries are misses. |
| Strict requests could consume references/results selected under relaxed policy. | Bind result/flight identity to policy and reject relaxed reference provenance in strict mode. |
| Every retry resolved a reference before checking for a completed alignment. | Check the request-bound result cache first; provider outages do not invalidate a warm result. |
| Failed alignments repeatedly launched alass. | Briefly cache alignment failures as well as reference misses; isolate failures by current credentials/context. |
| Cancelling a player request unregistered a still-running shared job. | The job unregisters itself on completion, so later requests join it. Detached failures are consumed. |
| Cancelling external resolution left provider tasks running. | Cancel and drain child tasks in all exit paths; drain the orchestrator before closing application resources. |
| Several independent alignments could exhaust container resources. | Default to one concurrent alass job via `ALASS_MAX_CONCURRENT_SYNCS`; cancellation holds the slot until its worker exits. |
| The edition shift limit checked only the first three cues. | Apply it across the full cue population. |
| Autosync violated the preference to retain native ASS, serving SRT with an ASS content type. | Preserve native ASS/SSA without alignment when conversion is disabled. |
| A fallback response's 24-hour HTTP cache hid later background alignment. | Enabled autosync responses use `private, no-cache` so a later request can obtain the completed result. |
| Setup requirements and knobs were missing from the example environment/documentation. | Document enablement, prerequisites, limits, and concurrency settings. |

## Verification

`tests/test_autosync_audit.py` exercises the cache, orchestrator, real serving
handlers, cancellation, concurrency, and alignment guardrails. Its initial
12 checks failed before fixes; additional cases cover the subsequent findings.

Final results: **796 passed, 1 skipped** in the complete suite. The skipped
optional real-engine test was run separately with the downloaded binary;
all **23 audit checks passed**, including real alass alignment.

Run the complete suite:

```sh
python -m pytest -q --tb=short
```

The optional real-engine test runs when alass is available on `PATH`, or with:

```sh
NINJASUBS_TEST_ALASS=/path/to/alass python -m pytest tests/test_autosync_audit.py -q
```

Verified against the official [alass 2.0.0 release](https://github.com/kaegi/alass/releases/tag/v2.0.0):
a synthetic English reference and an Arabic target offset by three seconds
aligned across all ten cues with less than 20 milliseconds of residual error,
preserving every Arabic dialogue line. Live provider credentials and a real
Stremio player were not used for this audit.

## Remaining limits

- Arabic targets only; this pipeline currently uses English subtitle references,
  with no direct audio alignment.
- Embedded extraction needs a stream URL supplied by the client. Standard
  filename/hash-only requests use the other reference tiers.
- Matching release metadata and plausible timestamps cannot prove alignment
  quality for arbitrary real content. Selection remains conservative; missing
  or unsuitable references fall back to the original.
- References currently need more than 5 KiB and alignment needs at least five
  valid cues, so very short clips can fall back unchanged.
- The inline budget covers the sync phase. Upstream target download time is
  additional. A background result applies to a subsequent subtitle request;
  it cannot replace a subtitle already loaded into a player.
- Concurrent alass work is limited per process. Multiple server workers do not
  share a global queue or distributed lock.
- Persistent reference cache expiry is checked on lookup. It has no global
  disk quota or scheduled sweep for entries that are never requested again.
- Native ASS/SSA timing alignment with styling preserved remains unsupported
  by this wrapper; enabling conversion allows alignment through SRT.
