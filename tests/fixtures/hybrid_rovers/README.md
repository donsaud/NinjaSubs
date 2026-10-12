# Hybrid reachability fixtures — Dexter S08E01 (ROVERS reference)

Real artifacts for the regression tests reproducing the live 921-cue
request in `tests/test_hybrid_predelivery.py`. Do NOT edit these files:
tests assert exact bytes, cue counts, and analyzer verdicts against them.

## Files

- `dexter-s08e01-rovers-634.srt`
  Byte copy of the cached video-scoped reference
  `subs_cache/references/tt0773262_8_1_PiR8_v2_e3de2a312c092e32.video.subdl.edition.21c5fbb4a5ded29c.srt`
  (`Dexter.S08.1080p.BluRay.x264-ROVERS(CompleteSeason).srt`, subdl).
  SHA256 `4ca0a36b98d296581f56f923d37ad185b669503f4c359c1bce0eafb173207c42`,
  64310 bytes, 634 cues. This file is NOT claimed to be the user-confirmed
  iggig upload; it is the reference the live request actually used
  (634 cues), whatever its provenance.
- `dexter-msd-rovers-alass-synced.srt`
  Raw stdout-file bytes of official `kaegi/alass` v2.0.0
  (`alass <ref> <tgt> <out> --split-penalty 7.0`, series default), run
  offline against the mSD target
  (`tests/fixtures/two_point_sync/dexter/Dexter.S08E01.480p.HDTV.x264-mSD.srt`)
  and the reference above. Deterministic across runs.
  SHA256 `ef56eefb5c075ad0b09a494fd7fb0a2795859989ac4232379a1c591ba6c37a8c`,
  921 cues. IMPORTANT: preserve raw bytes (CRLF). A Python text-mode
  read/write round-trip rewrites line endings and changes the hash without
  changing any timestamp; tests read these bytes and decode only.

## Regeneration (offline, needs the alass binary)

```powershell
$tmp = Join-Path ([IO.Path]::GetTempPath()) "alassregen"
New-Item -ItemType Directory -Path $tmp -Force | Out-Null
Copy-Item "subs_cache/references/tt0773262_8_1_PiR8_v2_e3de2a312c092e32.video.subdl.edition.21c5fbb4a5ded29c.srt" "$tmp/ref.srt"
Copy-Item "tests/fixtures/two_point_sync/dexter/Dexter.S08E01.480p.HDTV.x264-mSD.srt" "$tmp/tgt.srt"
& <alass-cli.exe> "$tmp/ref.srt" "$tmp/tgt.srt" "$tmp/out.srt" --split-penalty 7.0
# expect SHA256 ef56eefb... and 921 cues
```

## Why committed

`test_hybrid_predelivery.py` pins the legacy chain (UNVERIFIED p95≈4735 +
large-offset serve grant) and the V2-before-delivery correction on these
exact bytes. Per the task, a mocked ALASS result must never stand in for
real-world evidence; these are real ALASS bytes. Follows the existing
`tests/fixtures/large_offset/` committed-srt precedent.
