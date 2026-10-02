# Manual Stremio playback validation

The API-side delivery path for Dexter S08E04 is verified. What is **not** verified
is what a player actually displays, because a container has no Stremio client, no
provider credentials, and no rendered pixels.

This page is the procedure for closing that on a machine that has Stremio.

**Read this distinction before you start.** It is the whole point of the test:

| | who decides it | how you check it |
|---|---|---|
| **What the addon served** | the API. Already verified. | `sha256` of the HTTP payload, compared to the expected hash below |
| **What the viewer sees on screen** | you, watching the episode | the checklist below |

Those two can differ. If the addon serves the provider's original bytes and you see
an unsynchronised opening, that is the two facts agreeing, not a bug.

## Target under test

| | |
|---|---|
| Series | Dexter, Season 8, Episode 4 |
| IMDb | `tt0773262` |
| Video release | `Dexter.S08E04.Scar.Tissue.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv` |
| | 1920×1080, 23.976 fps, 3006 s |
| Video sha256 | `c929b5f64845b1554de2580c4f91e44a6e70ab4eeca71449c4df97c404197e0f` |
| Video size | `5414925805` bytes |
| Subtitle language | Arabic (`ar`) |

## Prerequisites

1. Stremio installed and able to play the episode.
2. The real media available to Stremio at the release above.
3. NinjaSubs running and reachable.
4. **At least one provider credential configured.** Without one the addon
   advertises zero catalogs and Stremio lists no subtitles at all.

```bash
# .env
ENABLE_SUBTITLE_SYNC=true
SUBDL_KEY=<your key>          # or SUBSOURCE_KEY / OPENSUBTITLES_*

docker compose up -d --force-recreate
```

In Stremio: Settings → Add-ons → Install from URL → `http://localhost:7000/manifest.json`.
Then enable **AutoSync Subtitles** in the addon's configuration.

## Run the automated checks

```bash
python tools/validate_stremio_playback.py
```

It verifies the addon is healthy, that credentials and catalogs are present, prints
the expected artifact hashes from the committed manifest, prints the cached
provider hashes, and then prints the playback checklist. It does not drive the GUI.
Exit code `2` means the addon is not ready yet and playback cannot be attempted.

---

## Which subtitle cases to test

Test **A** and **B** in full. Test **C** only if your test environment can actually
select it.

### A. EVOLV — `74a34c232d159f8b`

| | |
|---|---|
| Expected decision | `ORIGINAL` |
| Expected served sha256 | `e82586e4cbacf6d394c7d34edafdc15747cebd7364ed5dcaba13b8a10788580c` |
| Alass artifact, **not** served | `2c95ea92cfb9d83431b16d7e3f82301ae0d4d3686b5c5523f8cd651589edad9b` |
| Verifier state | `UNVERIFIED` — residual p95 `3310 ms` against a `2000 ms` limit |
| Why the original is served | the verifier does not accept the Alass result |

### B. ASAP — `96c0442cc4656685`

| | |
|---|---|
| Expected decision | `ORIGINAL` |
| Expected served sha256 | `30246e82e06d3ee2a0e14ebbffe4d29e0559f52762afff10f8214b117d564ac2` |
| Alass artifact, **not** served | `63556bdb9a409cbc0b94d2ac2cfa20fca236847c42059dc59d20f4cb98c5d930` |
| Verifier state | `UNVERIFIED` — residual p95 `4750 ms` against a `2000 ms` limit |
| Why the original is served | the verifier does not accept the Alass result |

**Neither A nor B is `VERIFIED`.** Do not record either as verified, whatever you
see on screen. The production verifier is unchanged and reports `UNVERIFIED` for
both, and a visual pass is engineering evidence, not a verification state.

### C. Known-safe synthetic `+96.5 s` transformed case

| | |
|---|---|
| Expected decision | `TRANSFORMED` |
| Expected served sha256 | `9d9863c963947cb2dc26636ad6fa58224e39768ac39e048c5701fe0f7189aa5e` |
| Verifier state | `VERIFIED` — the positive control |
| Availability | only if selectable from your test environment |

This is the positive control that proves the pipeline *can* deliver a corrected
subtitle. If it is not selectable in your environment, record
`NOT SELECTABLE IN THIS ENVIRONMENT` and skip it — do not manufacture a result.

---

## Playback checklist

Run this **once per case** (A, B, and C if available).

For each case, do not record anything yet. Fill the table in after you have watched.

```
1. Open Dexter · Season 8 · Episode 4.
2. Confirm the video release is the PiR8 1080p BluRay listed above.
3. Select the Arabic subtitle for this case.
4. Start playback FROM THE BEGINNING.
```

### 4a. Position checks

| position | time | content | record |
|---|---|---|---|
| 0% | 00:00 | opening scene / credits | ______________ |
| 10% | 05:00 | first opening dialogue | ______________ |
| 25% | 12:30 | first quarter | ______________ |
| 50% | 25:00 | midpoint | ______________ |
| 75% | 37:30 | third quarter | ______________ |
| 90% | 45:00 | late episode | ______________ |
| 100% | 50:00 | ending | ______________ |

At each position record `PASS` / `PARTIAL` / `FAIL` / `UNDETERMINED`, plus what you
saw: *in sync*, *early by ~N s*, *late by ~N s*, *no subtitle shown*, or *text
visible with no speech*.

### 4b. Scene checks

- [ ] **Opening scene** — the provider subtitles carry an opening section displaced
      by roughly 96 seconds. It is *not* a uniform shift across the episode, so the
      opening may look different from the body. Record what you see at 0% and 10%.
- [ ] **Dialogue-heavy scene** — a stretch of continuous speech somewhere in the
      middle. Record whether text tracks the speech line by line.
- [ ] **Seek backward** ~5 minutes, then forward. Does the subtitle follow?
- [ ] **Seek forward** ~5 minutes. Does the subtitle follow?
- [ ] **Pause** 30 seconds, then resume. Does timing survive the pause?
- [ ] **Restart** playback from the beginning. Same as the first load?
- [ ] **Repeat after cache warm-up** — play again after a few minutes. Any change
      from the first pass?

### 4c. Per-case result sheet

Copy one of these per case. Leave the result blank until you have watched.

```
Case:            A (EVOLV)  /  B (ASAP)  /  C (synthetic +96.5s)
Release listed:  ____________________________________
Fallback notice:  yes / no
0%:   result ______________   saw ____________________
10%:  result ______________   saw ____________________
25%:  result ______________   saw ____________________
50%:  result ______________   saw ____________________
75%:  result ______________   saw ____________________
90%:  result ______________   saw ____________________
100%: result ______________   saw ____________________
Opening scene:    result ______________   saw ____________________
Dialogue-heavy:   result ______________   saw ____________________
Seek backward:    result ______________   saw ____________________
Seek forward:     result ______________   saw ____________________
Pause / resume:   result ______________   saw ____________________
Restart:          result ______________   saw ____________________
Cache warm-up:    result ______________   saw ____________________

What the addon served: ORIGINAL / TRANSFORMED / unknown
Served sha256:     ____________________________________
Matches expected? yes / no / could not check

OVERALL: PASS / PARTIAL / FAIL / UNDETERMINED
Notes:  ____________________________________________
```

Allowed results for each row and for OVERALL: `PASS`, `PARTIAL`, `FAIL`,
`UNDETERMINED`. Use `UNDETERMINED` rather than guessing.

---

## Confirming what the addon served

To turn "what the addon served" from an expectation into a measurement, capture
the actual payload. The addon log line is the direct way — one INFO line per
request:

```
[sync] delivery sub=tt0773262:8:4 decision=ORIGINAL state=unverified \
  verification=unknown original_sha256=e82586e4cbacf6d3 \
  transformed_sha256=2c95ea92cfb9d834 served_sha256=e82586e4cbacf6d3 \
  served_is_transformed=False bytes=60213
```

- `decision=ORIGINAL` → `served_sha256` must equal `original_sha256`
- `decision=TRANSFORMED` → `served_sha256` must equal `transformed_sha256`

Compare the `served_sha256` prefix (first 16 hex characters) against the expected
full hash in the case table above. If it does not match, the server served something
other than what this document expects — record that as a finding, not a visual
result.

---

## What this does and does not establish

A `PASS` here is engineering evidence that the delivered subtitle is watchable in
sync on the real player. It is **not** a verification state.

The production verifier is unchanged and continues to report `UNVERIFIED` for both
real cases, which is correct until the measurement problems documented in
[sync_ground_truth.md](sync_ground_truth.md) and
[residual_pairing_candidates.md](residual_pairing_candidates.md) are resolved. Do
not mark EVOLV or ASAP `VERIFIED` on the basis of this test.

Until it is run and reported, the rendered-player leg of the validation remains
**UNDETERMINED** regardless of what the logs show.
