# Real-media benchmark cases

The synthetic dataset proves the validator's *logic* against ground truth it
controls. It cannot prove the *extractor* works on real programmes. This
document covers the out-of-band workflow for real material.

## Why the media is not in the repository

Real episodes and films are copyrighted. The repository holds only the schema,
this documentation, the synthetic fixtures, and optionally a **redacted**
manifest whose paths and hashes point at media that lives on your own disk.
`tests/test_video_timeline.py` enforces that nothing but `.srt`, `.vtt` and
`.json` exists under the fixtures tree.

## The decisive constraint

The production request path **never receives the video**. `stream_url` is
passed through as an opaque string and never fetched; `alass` only ever
receives two `.srt` files. See `docs/video_timeline_capabilities.md`.

So this benchmark is the only place the video validator can run today. That is
a property of the architecture, not an oversight, and it means the validator's
production value is currently **unreachable** no matter how well it performs
here. Any decision to wire it in would first have to solve media availability.

## Manifest format

Out of band, anywhere on your machine:

```json
{
  "dataset_version": "real-v1",
  "schema_version": 1,
  "engine_version": 1,
  "protocol": "docs/golden_sync_annotation.md",
  "cases": [
    {
      "case_id": "real_dexter_s08e05_case_001",
      "category": "different_cut",
      "release_class": "bluray",
      "video":     {"path": "/media/tv/Dexter.S08E05.mkv", "sha256": "...", "duration_ms": 3598000},
      "reference": {"path": "/media/subs/ref.srt",        "sha256": "..."},
      "candidate": {"path": "/media/subs/cand.srt",       "sha256": "..."},
      "ground_truth": {
        "content": "same_release",
        "cut": "different_cut",
        "sync": "resyncable",
        "final": "incorrect",
        "review_status": "two_reviewers_agreed",
        "reviewer_count": 2,
        "review_notes": "Reference is the extended TV cut; back half carries recap material absent from the BluRay.",
        "established_by": "human_reviewed"
      }
    }
  ]
}
```

`RealMediaManifest.available_cases()` returns only the cases whose media is
actually present. Absent media is the normal local state and is reported, not
treated as an error.

## Categories to annotate

**Positive** — same release already synchronized; constant offset; FPS drift; a
valid resynchronization.

**Negative** — wrong episode; different cut; different release; plausible
metadata but wrong media; alass exit 0 with a wrong alignment; reference and
candidate internally consistent but wrong for the target.

**Edge** — sparse dialogue; credits-heavy subtitle; intro/recap differences;
extended edition; TV edit; WEB-DL versus BluRay; different FPS.

The last group matters most. Every case so far in this project has been a
splice or a global offset, and a validator tested only against those will look
better than it is.

## Ground truth must stay independent

Never label `final` because the current verifier agrees. Establish it by exact
file identity, an independently verified subtitle/video pair, or human review.
For reviewed cases record `review_status`, `reviewer_count` and
`review_notes`; never store subtitle text.

Where a reviewer watched the video and confirmed the timing, use
`established_by: human_reviewed`. Where it rests on file identity, use
`objective`. Conclusions from the two are not interchangeable, and the manifest
keeps them apart for exactly that reason.

## Running it

The profile cache is keyed by video identity and profile version, so extract
once and reuse:

```python
from app.services.sync.golden import load_real_media_manifest
from app.services.sync.video_timeline import (
    VideoProfileCache, extract_video_profile, validate_timeline_against_reference,
)
from app.services.subtitle_matcher import parse_srt_cues

cache = VideoProfileCache("/var/cache/ninjasubs/video_profiles")
for case in load_real_media_manifest("C:/bench/real.json").available_cases():
    profile = cache.get(case.video.sha256 or case.video.path)
    if profile is None:
        profile = extract_video_profile(case.video.path)
        if profile:
            cache.set(case.video.sha256 or case.video.path, profile)
    cues = parse_srt_cues(open(case.candidate.path, encoding="utf-8").read())
    evidence = validate_timeline_against_reference(profile, [c[0] for c in cues])
    print(case.case_id, case.ground_truth.cut.value, evidence.verdict.value,
          evidence.reason.value)
```

## Cost

Measured in the project's own container on a 288-second file:

| stage | cost | frequency |
|---|---|---|
| profile extraction (ffprobe + silencedetect) | ~2.8-3.3 s | once per target video, cached |
| per-candidate comparison | pure arithmetic over landmarks | once per candidate |

Extraction decodes **audio only** for landmarks; the scene-boundary pass
decodes video at 160px wide. Scene detection returned zero landmarks on the
synthetic test clip (a single solid colour), which is the correct answer for
that input and the reason audio carries the weight.

## What would justify wiring this into production

Not this phase. It would need, at minimum:

1. A real-media sample large enough for the intervals to mean something.
2. Evidence that it closes wrong cuts **without** materially increasing
   missed valid resynchronizations. The counterfactual block in
   `--shadow-report` reports exactly that pair of numbers.
3. A solution to media availability, since production has no video.

Until all three hold, the honest status is: detectable in principle, measured
on synthetic and generated media, and unreachable on the request path.
