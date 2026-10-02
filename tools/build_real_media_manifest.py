"""Build the real-media case manifest for the Dexter S08E04 fixture.

Uses the project's existing ``RealMediaManifest`` schema from
``app.services.sync.golden`` -- no new format is invented. Paths are resolved
from an environment variable so nothing transient is baked in, and the media
root is only ever read.

The manifest records identity (hashes) rather than content, so it can live in
the repository without embedding the 5.4 GB fixture.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, ".")

from app.services.sync.golden import (  # noqa: E402
    DATASET_VERSION,
    GroundTruthContent,
    GroundTruthCut,
    GroundTruthFinal,
    GroundTruthSource,
    GroundTruthSync,
    RealMediaCase,
    RealMediaFile,
    RealMediaManifest,
    RealMediaTruth,
)

# Where the 5.4 GB source video lives. Never committed; supply your own path.
# Unset falls back to ./test-media, which the media-dependent tests skip against.
MEDIA_ROOT = pathlib.Path(
    os.environ.get("NINJASUBS_REAL_MEDIA_ROOT", "test-media")
)
CACHE = pathlib.Path(os.environ.get("NINJASUBS_CACHE_DIR", "subs_cache"))
REFS = CACHE / "references"
VIDEO = MEDIA_ROOT / "Dexter.s8e04.Scar.tissue.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv"
REFERENCE = REFS / \
    "tt0773262_8_4_PiR8_v2_f80bfa730f1a046f_t6349da3c0192e384_subsource_edition.srt"

# Alass outputs recorded by the real-media investigation, keyed by subtitle id.
ALASS_SHA = {
    "74a34c232d159f8b": "2c95ea92cfb9d83431b16d7e3f82301ae0d4d3686b5c5523f8cd651589edad9b",
    "96c0442cc4656685": "63556bdb9a409cbc0b94d2ac2cfa20fca236847c42059dc59d20f4cb98c5d930",
}
# Offline PGS witness agreement measured against the real video's own English
# subtitle track. Regression/evaluation only; never a production signal.
PGS_AGREEMENT = {
    "dexter_s08e04_evol": 0.970,
    "dexter_s08e04_asap": 0.999,
}
PGS_ORIGINAL_AGREEMENT = {
    "dexter_s08e04_evol": 0.670,
    "dexter_s08e04_asap": 0.713,
}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def video_fingerprint() -> str:
    """Same identity shape the pipeline records: size + content digest."""
    return f"size:{VIDEO.stat().st_size}:sha256:{sha256_file(VIDEO)}"


def main() -> int:
    if not VIDEO.is_file():
        print(f"video fixture not found: {VIDEO}", file=sys.stderr)
        print("set NINJASUBS_REAL_MEDIA_ROOT to the fixture directory", file=sys.stderr)
        return 2

    print("hashing the 5.4 GB fixture once (read-only)...", file=sys.stderr)
    fingerprint = video_fingerprint()
    duration_ms = 0
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nk=1:nw=1", str(VIDEO)],
            capture_output=True, text=True, timeout=300, check=True,
        )
        duration_ms = int(float(out.stdout.strip()) * 1000)
    except Exception as exc:  # pragma: no cover - optional enrichment
        print(f"  (ffprobe unavailable: {exc})", file=sys.stderr)

    ref_sha = sha256_file(REFERENCE)
    # Store the media by FILENAME, never an absolute path: the manifest is
    # committed and must stay portable across machines. Resolution happens at
    # read time against NINJASUBS_REAL_MEDIA_ROOT.
    video = RealMediaFile(path=VIDEO.name, sha256=fingerprint,
                          duration_ms=duration_ms)

    cases = []
    for case_id, sub_id, _label, notes in (
        ("dexter_s08e04_evol", "74a34c232d159f8b", "EVOLV",
         "Real S08E04 Arabic subtitle; ~96s opening discrepancy, not a uniform "
         "global offset. Alass output manually confirmed synchronized against "
         "the real PiR8 video throughout."),
        ("dexter_s08e04_asap", "96c0442cc4656685", "ASAP",
         "Real S08E04 Arabic subtitle; same opening discrepancy shape as EVOLV. "
         "Alass output manually confirmed synchronized against the real PiR8 "
         "video throughout."),
    ):
        target = CACHE / f"{sub_id}.srt"
        if not target.is_file():
            print(f"  missing provider subtitle {target}", file=sys.stderr)
            continue
        truth = RealMediaTruth(
            cut=GroundTruthCut.SAME_CUT,
            sync=GroundTruthSync.RESYNCABLE,
            final=GroundTruthFinal.CORRECT,
            content=GroundTruthContent.SAME_CONTENT_DIFFERENT_RELEASE,
            review_status="manual+objective",
            reviewer_count=1,
            review_notes=(
                "Alass output inspected against the real PiR8 video across all "
                f"four quarters. Offline PGS witness: output "
                f"{PGS_AGREEMENT[case_id]:.1%} vs original "
                f"{PGS_ORIGINAL_AGREEMENT[case_id]:.1%} of cues within 1.5s."
            ),
            established_by=GroundTruthSource.OBJECTIVE,
        )
        cases.append(RealMediaCase(
            case_id=case_id,
            category="large_offset_piecewise",
            release_class="bluray",
            video=video,
            reference=RealMediaFile(path=REFERENCE.name, sha256=ref_sha,
                                    duration_ms=duration_ms),
            candidate=RealMediaFile(path=target.name, sha256=sha256_file(target),
                                    duration_ms=duration_ms),
            ground_truth=truth,
            notes=notes,
        ))

    manifest = RealMediaManifest(
        dataset_version=DATASET_VERSION,
        description=(
            "Dexter S08E04 real-media regression corpus. Alass artifact hashes "
            "and offline PGS witness scores are recorded here so results are "
            "comparable without the media being committed."
        ),
        protocol="docs/golden_sync_annotation.md",
        cases=cases,
    )

    out = pathlib.Path("tests/fixtures/real_media/dexter_s08e04_manifest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(manifest.model_dump_json())
    # Attach the derived identity the corpus must stay pinned to.
    payload["_derived"] = {
        "video_fingerprint": fingerprint,
        "video_size": VIDEO.stat().st_size,
        "reference_sha256": ref_sha,
        "alass_artifact_sha256": ALASS_SHA,
        "pgs_witness_agreement": PGS_AGREEMENT,
        "pgs_witness_agreement_original": PGS_ORIGINAL_AGREEMENT,
        "pgs_tolerance_s": 1.5,
        "expected_delivery_category": "ORIGINAL_SERVED",
        "notes": (
            "expected_delivery_category reflects the CURRENT conservative "
            "verifier: residual p95 rejects both cases, so the original "
            "provider subtitle is served. This is a regression expectation, "
            "not a claim that the alass artifact is inferior."
        ),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}", file=sys.stderr)
    print(json.dumps(payload["_derived"], indent=2))
    return 0


raise SystemExit(main())
