# The 562-cue to 0-cue transition: root cause

A real request for `ab461b8781aafd89` logged a contradictory pair of numbers for
one subtitle:

```
pre-alass   target_cue_count=562
post-alass  target_parsed_cue_count=0  ->  "target has 0 cues"
```

## Answer

| field | value |
|---|---|
| `original_format` | **ASS/SSA** (`[Script Info]` header) |
| `original_extension` | `.srt` — **mislabeled** |
| `pre_alass_parser` | SRT parser, on **already-converted** bytes (562 cues) |
| `alass_output_format` | **SRT**, 562 cues — Alass ran and succeeded |
| `post_alass_parser` | SRT parser, on **raw ASS** (0 cues) |
| `root_cause` | **format-routing defect**: the ASS guard only covers `convert_ass=False` |

Both cue counts are correct, and Alass did succeed. The defect is that Alass and
the verifier were fed **two different strings** derived from the same file.

## Evidence, byte for byte

The cached artifact is 68,310 bytes, starts `b'[Script Info]\r\n'`, has no BOM,
no nulls, and decodes as UTF-8. It is unmistakably ASS. `is_ass_subtitle` returns
`True`. Stored under an `.srt` name.

Feeding that raw ASS text to the SRT parser yields **0 cues**, over **50,721
characters** — matching the log's `target_raw_chars=50721 target_parsed_cue_count=0`
exactly.

Alass itself never saw the raw ASS. `SubtitleSyncService.sanitize_subtitle()`
(`app/services/sync_service.py:313`) detects ASS and runs
`convert_ass_to_srt(..., apply_rtl=False)` before handing anything to Alass, so
Alass received a well-formed SRT and returned a 562-cue SRT. Alass's own output
was never the problem.

Feeding the raw ASS text *directly* to Alass does fail outright:

```
expected SubRip index line, found '[Script Info]'
caused by: invalid digit found in string
```

That confirms Alass is SRT-only — but it is not the path production took, so it
is not the explanation for the log.

## The two strings

| stage | string | cues |
|---|---|---|
| `sanitize_subtitle` → Alass | ASS converted to SRT | 562 |
| Alass output | SRT | 562 |
| verifier (`AlignmentAnalyzer._as_cues`) | **raw** ASS decoded by `decode_subtitle_bytes` | 0 |

The pre-alass number came from the converted string; the post-alass number came
from the raw string. Nothing was corrupted in transit — two code paths simply
disagreed about what the payload was.

## The defect

`app/main.py:1864` (before the fix)

```python
if not convert_ass and is_ass_subtitle(payload):
    return payload          # preserve native ASS
return await _maybe_sync_subtitle(payload, ...)
```

The guard returns early only when the user has **disabled** ASS conversion. With
conversion **enabled** — the default — the raw ASS payload falls through to
`_maybe_sync_subtitle`. `SyncOrchestrator.evaluate_and_sync`
(`app/services/sync/orchestrator.py:258`) then decodes the raw bytes for its own
target cues, so the verifier receives ASS that
`AlignmentAnalyzer._as_cues()` parses with an SRT-only parser.

ASS→SRT conversion happened later still, in `_build_subtitle_response`
(`app/main.py:2022`), which runs *after* the sync attempt. So the ordering is
inverted:

```
provider bytes -> [SYNC: alass gets sanitized SRT, verifier gets raw ASS] -> [ASS->SRT] -> response
```

It should be:

```
provider bytes -> [ASS detection] -> [conversion OR native preservation] -> [SYNC]
```

## Why this is a real production defect, not an ASS edge case

1. **It is reachable from the default configuration.** `convert_ass_to_srt`
   defaults to on. The guard protects the non-default path.
2. **It fails safe, but wastefully and misleadingly.** The verifier refuses the
   0-cue input, so no bad subtitle is ever served and nothing is corrupted. But
   the verifier judges a string the user never receives, so its verdict is
   meaningless, and a correct, syncable ASS release is rejected because of its
   container format.
3. **A correctly routed identical file verifies.** Converting this same artifact
   first and syncing it gives 562/562 cues, residual p95 **0 ms**, and
   `VERIFIED_RESYNCED`. The content is fine; only the routing is wrong.

## Correct behaviour

| input | expected path |
|---|---|
| ASS content, `.ass` name | detect → sync (as SRT) **or** preserve natively, per preference |
| ASS content, `.srt` name | detect by **content** → same as above |
| SRT content | unchanged: detect → sync |

Detection must be by content, never by extension — which `is_ass_subtitle`
already does correctly. The only change needed is the *ordering*: the sync stage
must not receive a payload its parser cannot read.

## Scope of the fix

Route by content, so the sync stage only ever receives a format it can parse.
Detection stays content-based (`is_ass_subtitle`), never extension-based.

Both ASS preference paths are preserved:

| preference | path |
|---|---|
| `convert_ass=True` (default) | convert to SRT **before** sync, then run the normal SRT pipeline |
| `convert_ass=False` | return the native ASS byte-identically, bypassing sync entirely |

The native path must stay a bypass rather than a convert-then-sync: `{\pos}` and
styling have to survive for the user who asked to keep them.

No threshold, no acceptance rule, and no verifier decision changes.
