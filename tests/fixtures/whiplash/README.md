# Whiplash regression fixtures

`target.srt` was recovered from the existing local subtitle cache
(`subs_cache/4b1b2c0fd16e8a02.srt`). Its 825 cues and first/last start
timestamps, 142640 and 5610800 milliseconds, match the reported original.
SHA-256: `4dd84bcaeda8f6da65364bfcad127aafe8fd89fa51409d823a2a5e4c63350cf3`.

Additional real files supplied by the user:

- `reference.srt`: `Whiplash.2014.1080p.REMUX.BluRay.AVC.DTS.HD.MA.5.1-HDA.srt`
- `expected.srt`: `Whiplash (2014) WEBRip.iTunes- synced by Subtitle Edit.ar.srt`

All three files are required regression fixtures. The test independently
generates output using only `target.srt` and `reference.srt`, then opens
`expected.srt`. It requires 825 exact start matches and end error at most 1 ms.
The oracle is never an input to synchronization or verification.
