# Target media capabilities: what can actually be derived from the video

This is the investigation §3 requires, performed before any validator was
written. It records what the project has, what the container provides, and
which signals are realistically available. The conclusions shaped everything
else in this phase, including one that changes the phase's premise.

## Finding 1: production has no target video (decisive)

**The application never receives, downloads or opens the target video.**

Evidence:

- `app/main.py:578` reads `stream_url` from the Stremio request and passes it
  through as an opaque string. Grepping every use of `stream_url` in `app/`
  finds only pass-through into params dicts and logs. Nothing fetches it.
- `app/services/sync/query.py` `ReferenceQuery` carries `target_filename`
  (a *name*), `stream_url`, `video_hash`, `video_size`. It carries no video
  path and no video bytes.
- `app/services/sync_service.py` writes exactly two temporary files per sync,
  both `suffix=".srt"`. `alass` is invoked as
  `alass <REFERENCE.srt> <TARGET.srt> <OUTPUT.srt>`. No media is ever decoded.
- There is no `video_path`, `.mkv` open, or video download anywhere in `app/`.

The only media tooling that exists is for *subtitle* extraction:
`app/extractor.py` shells out for archive/subtitle extraction, and the
Dockerfile installs `ffmpeg` alongside `libarchive-tools` for that.

**Consequence.** A video-derived timeline profile cannot be computed on the
production request path, because the input does not exist there. Any design
that assumes otherwise would have to download a multi-gigabyte stream per
request, which is not a validator, it is a different product.

This phase therefore builds the layer where video genuinely is available —
the real-media benchmark harness — and makes the production path report
`VIDEO_PROFILE_UNAVAILABLE`, which is *absence of evidence*, never negative
evidence. See "Honest reporting" below.

## Finding 2: the container has the tools

Verified by executing in the running container:

| binary | path |
|---|---|
| ffmpeg | `/usr/bin/ffmpeg` |
| ffprobe | `/usr/bin/ffprobe` |
| alass | `/usr/local/bin/alass` |

Relevant filters confirmed present in the image:

- `silencedetect` (A->A) — audio activity boundaries
- `scdet` (V->V) — video scene change
- `blackdetect` (V->V) — black intervals
- encoder `pcm_s16le`

Container Python is 3.11.16. The project venv is 3.12; the two are kept in
step by the existing workflow and nothing in this phase depends on the gap.

## Finding 3: no existing ffprobe wrapper exists

There is no `ffprobe` call anywhere in `app/`. A new wrapper is therefore new
code, not a reuse of an existing abstraction. It must fail soft: a missing
binary is a legitimate outcome on a developer machine (this host has none of
the three binaries), not an error worth crashing a sync over.

## Candidate signals

| signal | availability | cost | reliability | dependency | verdict |
|---|---|---|---|---|---|
| container duration | ffprobe, instant | negligible (header only) | high for duration, but a different cut can share a duration | none new | **use**, as support only |
| stream counts / codecs | ffprobe, instant | negligible | high | none new | **use**, cheap identity |
| audio activity (silencedetect) | ffmpeg, decodes audio only | seconds for a feature, ~1-3 min for a full episode | high: music/still frames and recaps differ structurally between cuts | ffmpeg already present | **use**, primary signal |
| scene boundaries (`scdet`) | ffmpeg, decodes video | most expensive; scales with video length | medium: encoder-dependent, and a re-encode changes boundaries | ffmpeg already present | **use**, low weight |
| keyframe / GOP layout | ffprobe `-skip_frame nokey` | low | **low**: not stable across encoders | none | reject, per §5.D |
| full-resolution frame decode | ffmpeg | prohibitive | n/a | ffmpeg | reject |
| speech recognition / VAD / embeddings | not installed | seconds-to-minutes, model download | n/a | **large new dependency** | **reject**, per §2 and §28 |

The smallest set that can plausibly detect a cut difference is **duration +
stream counts + coarse audio activity + coarse scene boundaries**, with audio
activity carrying the weight. Keyframes are rejected explicitly: the phase
specification warned they are not stable across encoders, and nothing here
depends on them.

## Cost model

The profile is extracted **once per target video** and cached. Per-candidate
validation then compares a cached profile against a subtitle's timeline, which
is arithmetic over landmark lists, not decoding. The intended shape:

```
Video -> one cached profile
Candidate 1 -> cheap comparison
Candidate 2 -> cheap comparison
Candidate 3 -> cheap comparison
```

Extraction is bounded by explicit sampling limits and a timeout, and it is
never invoked on the production path, so the per-request cost of this phase is
exactly zero.

## Honest reporting

The distinction §19 demands is enforced in code, not just documented:

- `VIDEO_PROFILE_UNAVAILABLE` — no video, or the binary is missing.
- `VIDEO_EVIDENCE_INSUFFICIENT` — a profile exists but too few landmarks to
  compare.
- `VIDEO_TIMELINE_MISMATCH` — landmarks actively disagree.

Only the third is negative evidence. The first two must never be allowed to
withhold a verification, and never to create one. A validator that cannot see
must abstain, not guess.

## What this means for the phase

The wrong-cut false positive the golden benchmark found is real, and it is not
a residual, consensus, cache or structural-analysis bug. It is the logical
consequence of only ever comparing two subtitle artifacts to each other.

The layer built here is the first evidence source outside that closed loop.
But it can only be *proven* where video exists, which today means the
real-media benchmark harness. Until someone points that harness at real
material and the numbers are in, the honest status of the wrong-cut gap is:

> **Detectable in principle by an independent video timeline signal; not yet
> measured on real media; and unreachable on the production path, which has no
> video.**

That is a smaller claim than the phase asked for, and it is the one the
evidence supports.
