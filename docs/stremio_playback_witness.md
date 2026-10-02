# STREMIO_PLAYBACK_WITNESS — manual playback witness

**Development/test only.** This page describes an isolated, opt-in service that
delivers one already-validated Alass output to Stremio, byte for byte.

## What question this answers

> "Can Stremio display the already-known-good Alass output correctly when that
> exact output is delivered directly to it?"

The production verifier currently **rejects** the Alass artifact for the real
Dexter S08E04 EVOLV and ASAP subtitles and serves the provider's original
instead. That is deliberate and this work changes nothing about it.

The open question is whether the rejected artifact would in fact have played
correctly. Until someone watches it in a real player, we cannot tell whether the
conservative fallback is protecting users from a bad subtitle, or whether it is
discarding a good one. This witness removes the production pipeline entirely so
the player is the only variable left.

## What the witness is not

It is not a second production path, and it shares no state with the real one:

| | production service | witness service |
|---|---|---|
| port | `7000` | `7100` |
| provider search | yes | **no** |
| alass | yes | **no** |
| sync / verifier | yes | **no** |
| provider disk cache | `./subs_cache` | **none mounted** |
| SyncCache / negative cache | yes | **none** |
| provider API keys | from your `.env` | **empty** |
| produces subtitles for | any media | **only Dexter S08E04** |

It reads a file and returns its bytes. There is no code path by which a witness
request can consult, mutate, or be decided by any production component.

## The fixtures

Both are pre-validated Alass outputs. **Do not regenerate them, and do not edit
their timestamps.** They are used exactly as they are.

| case | file | sha256 | bytes |
|---|---|---|---|
| EVOLV | `02_EVOLV_alass_output.srt` | `2c95ea92cfb9d83431b16d7e3f82301ae0d4d3686b5c5523f8cd651589edad9b` | 57,587 |
| ASAP | `04_ASAP_alass_output.srt` | `63556bdb9a409cbc0b94d2ac2cfa20fca236847c42059dc59d20f4cb98c5d930` | 62,406 |

Expected location: wherever you keep them. There is no default — set
`WITNESS_FIXTURE_DIR` in `.env` to the `autosync-diagnostics` folder holding
those two files.

These files stay on your machine. They are not in the repository, and the test
suite generates its own throwaway fixtures so nothing depends on the real media.

---

## 1. Enable the witness mode

Add **one line** to your `.env` — the folder holding the two `.srt` files. No
Python editing is required.

```env
WITNESS_FIXTURE_DIR=C:\Media\autosync-diagnostics
```

On Windows, keep the backslashes as shown. Compose takes this value verbatim
because the mount uses the long `type: bind` / `source:` / `target:` form rather
than the short `src:dst:ro` form — a Windows drive letter contains a colon, which
the short form has to disambiguate from its own separators.

> **This line is required, and there is deliberately no working default.**
> Docker *silently creates* a missing bind-mount source as an empty directory, so
> leaving it unset gives you a service that starts, reports itself healthy, and
> serves nothing. The preflight below exists to catch exactly that.

## 2. Start it

```bash
docker compose --profile witness up -d --build witness
```

This starts **only** the witness on port `7100`. It leaves the production service
on `7000` untouched, and plain `docker compose up` still never starts the
witness.

## 3. Verify before touching Stremio

Run the preflight. It checks the host folder, the live container's mount, both
fixture digests, and the bytes actually served over HTTP:

```bash
python tools/check_witness_mount.py
```

It must end with `OK  the witness will serve both fixtures byte-for-byte.`
Exit code `0` means proceed; `1` means do not proceed; `2` means the service is
not running.

<details>
<summary>Manual equivalent, if you prefer to check by hand</summary>

```bash
docker inspect ninjasubs-witness --format "{{json .Mounts}}"
docker exec ninjasubs-witness sh -c "ls -lah /witness"
docker exec ninjasubs-witness sha256sum /witness/02_EVOLV_alass_output.srt
curl -s http://localhost:7100/stremio-playback-witness/fixtures.json
```

`fixtures.json` must report `"mount_usable": true`, and both entries
`"available": true` with `"matches_validated_digest": true`.

To confirm byte identity yourself:

```bash
curl -s -o evolv.srt http://localhost:7100/stremio-playback-witness/subtitle/EVOLV/2c95ea92cfb9d83431b16d7e3f82301ae0d4d3686b5c5523f8cd651589edad9b.srt
certutil -hashfile evolv.srt SHA256
```

The printed digest must equal
`2c95ea92cfb9d83431b16d7e3f82301ae0d4d3686b5c5523f8cd651589edad9b`. Do the same
for ASAP using `ASAP` and its digest.

</details>

### If the preflight fails

| message | cause | fix |
|---|---|---|
| `WITNESS_FIXTURE_DIR is not set` | Compose fell back to `./autosync-diagnostics` | set the variable in `.env` |
| `directory exists but is EMPTY` | the host folder was not mounted | check the path in `.env` points at the real folder |
| `does not exist` | wrong path, or the service predates your `.env` edit | fix the path, then recreate |
| `matches_validated_digest: false` | the file is not the validated artifact | **stop.** Do not substitute or regenerate it |

After editing `.env` always recreate, or the old mount persists:

```bash
docker compose --profile witness up -d --force-recreate witness
```

## 4. Install the addon in Stremio

Stremio → Settings → Add-ons → Install from URL:

```
http://localhost:7100/stremio-playback-witness/manifest.json
```

The add-on is named **NinjaSubs Playback Witness (DEV ONLY)**.

> Install this **in addition to** the normal NinjaSubs add-on, not instead of it.
> You need both: the witness for the test, the normal one for the control.

> ### Check you used port 7100
>
> The production add-on has a long-standing catch-all route that answers
> `…/manifest.json` with the **normal NinjaSubs manifest**. So if you mistype the
> port and install `http://localhost:7000/stremio-playback-witness/manifest.json`,
> Stremio will happily install the *production* add-on and you will see no `TEST`
> subtitles — which looks like a working install but is not the witness.
>
> Confirm by name: the add-on must read **NinjaSubs Playback Witness (DEV ONLY)**.
> If it reads just **NinjaSubs**, you are on the wrong port.
>
> To check directly:
>
> ```bash
> curl -s http://localhost:7100/stremio-playback-witness/manifest.json | findstr /i "witness"
> ```
>
> Nothing matching means the witness is not running; start it again.

## 5. Which episode and which subtitle

- Open **Dexter → Season 8 → Episode 4**.
- Make sure the video is
  `Dexter.S08E04.Scar.Tissue.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv`.
- Open the subtitle picker. You should see exactly two entries, both labelled
  `TEST`:

```
ara | TEST - EVOLV ALASS WITNESS (known-good, do not edit)
ara | TEST - ASAP  ALASS WITNESS (known-good, do not edit)
```

**Confirming you picked the right one.** The label contains `TEST` and `WITNESS`.
A normal NinjaSubs entry looks different — it carries a provider name and a match
score. If you cannot see a `TEST - ... WITNESS` entry, Stremio is talking to the
wrong add-on.

---

## The protocol

Run once per witness subtitle (EVOLV, then ASAP). Record
`PASS` / `PARTIAL` / `FAIL` / `UNDETERMINED` per row. **Do not write `VERIFIED`
anywhere** — this is a playback observation, not a verification state.

```
Case:   A (EVOLV witness)  /  B (ASAP witness)
Label shown: ______________________________________
Started from 00:00:  yes / no

Position checks
  0%   (00:00)  opening scene / credits   result ______  saw ______________
  10%  (05:00)  first opening dialogue    result ______  saw ______________
  25%  (12:30)  first quarter             result ______  saw ______________
  50%  (25:00)  midpoint                  result ______  saw ______________
  75%  (37:30)  third quarter             result ______  saw ______________
  90%  (45:00)  late episode              result ______  saw ______________
  100% (50:00)  ending                    result ______  saw ______________

Scene and transport checks
  opening discontinuity     result ______  saw ______________
  dialogue-heavy scene      result ______  saw ______________
  seek backward ~5 min      result ______  saw ______________
  seek forward ~5 min       result ______  saw ______________
  pause 30 s, then resume   result ______  saw ______________
  restart from beginning    result ______  saw ______________
  replay after warm-up      result ______  saw ______________

OVERALL: PASS / PARTIAL / FAIL / UNDETERMINED
Notes: ____________________________________________
```

**What to look for.** The provider subtitles carry an opening section displaced by
roughly 96 seconds. That displacement is *not* a uniform shift across the episode,
so the opening can legitimately look different from the body — note both. The
Alass output was validated against the real video and should be in sync
throughout.

**What "what the addon served" means here.** For the witness it is fixed and known:
these exact bytes, verified by digest before you start. You do not need to
determine it. What you are reporting is only what appeared on screen.

---

## Control test — do this afterwards

Replay the same episode with the **normal NinjaSubs** add-on, using the ordinary
EVOLV candidate from the production add-on.

| | witness | production control |
|---|---|---|
| subtitle | `TEST - EVOLV ALASS WITNESS` | normal EVOLV candidate |
| served bytes | `2c95ea92…` (Alass output) | `e82586e4…` (provider original) |
| production decision | n/a — bypasses the pipeline | `ORIGINAL` — verifier rejected the Alass result |
| expected on screen | in sync throughout | original provider timing, ~96 s displaced opening |

Recording both is what makes the result interpretable. The control confirms the
player, the file, and the episode are all fine, so any difference is attributable
to the served bytes rather than to the test setup.

**Do not change either path to make them match.** A difference here is the finding,
not a defect to be smoothed over.

---

## If the witness plays correctly (`PASS`)

The conclusion is narrow and should be recorded exactly as stated:

> Known-good Alass output renders correctly in Stremio, while the production
> verifier currently rejects it and replaces it with the original.

That isolates the remaining question to NinjaSubs' **acceptance/delivery policy**
— not to Stremio's rendering, not to the subtitle format, and not to the Alass
output itself.

**Do not change the verifier on the strength of this test.** One witness narrows
where to look; it does not measure a general case. A change to the acceptance
policy needs its own evidence and its own review, and p95 is not the only thing
separating these artifacts from acceptance.

## If the witness does *not* play correctly

Do **not** change production sync logic. Work through this list in order, because
the first question is whether Stremio ever received the right bytes:

1. Did `/fixtures.json` report `matches_validated_digest: true`?
2. Did your own `curl` + `certutil` hash match the expected digest?
3. Do the response headers show `x-witness-sha256` equal to that digest?
4. Is `Content-Type` `text/plain; charset=utf-8`?
5. Does the file still contain its UTF-8 BOM, and are its line endings still
   CRLF? Stremio may handle those differently from your curl.
6. Is Stremio perhaps still showing a cached body from an earlier attempt? The URL
   is digest-versioned for exactly this reason — re-check the URL in the picker.
7. Only then consider client-side timing behaviour or player-side parsing.

The question to answer first is narrow: **did Stremio receive byte-identical
fixture content?** Everything else follows from that. If the preflight passed but
playback still looks wrong, repeat step 1 of the list above before suspecting the
player — the preflight already proves Stremio is being offered the right bytes.

---

## Turning it off

```bash
docker compose --profile witness down
```

Removing `ENABLE_STREMIO_PLAYBACK_WITNESS` / `WITNESS_FIXTURE_DIR` from `.env`
disables it permanently. With the flag absent the routes are **not mounted at
all** — the production add-on exposes no witness endpoint.

Removal from the codebase is a two-step revert: delete `app/playback_witness.py`
and remove the three lines in `app/main.py` that reference it. Nothing else in the
application imports it.
