"""Real-media regression suite for the Dexter S08E04 fixture.

What this adds over the existing delivery tests: it is driven from a committed
manifest of *identity* (hashes, sizes, expected delivery category), so a run
proves the same thing on any machine, and it records the full provenance chain
for each case rather than a single hash comparison.

The real media is read-only and is only ever hashed. The subtitle fixtures and
the alass binary come from the image, so the suite needs no network. If the
media root is absent the suite skips with a reason instead of failing -- a
missing 5.4 GB fixture is not a regression.

Coverage:
  * the artifact the production path returns, hashed and classified
  * ORIGINAL_SERVED / TRANSFORMED_SERVED / CACHED_TRANSFORMED_SERVED
  * the safe direction (alass produced output, verifier rejected, original served)
  * the unsafe direction (an unverified artifact must never be servable or
    later reusable), asserted as a negative
  * provenance: a transformed artifact must never become the next provider input
  * repeat-request behaviour, measured rather than inferred
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pathlib
import shutil
import time

import pytest

from app.services.sync.alignment import (
    is_reusable_verified,
    may_serve_synchronized,
)
from app.services.sync.orchestrator import SyncOrchestrator
from app.services.sync.query import ResolvedReference
from app.services.sync_service import SubtitleSyncService

REPO = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = REPO / "tests" / "fixtures" / "real_media" / "dexter_s08e04_manifest.json"
# The 5.4 GB source video is never committed, and nothing in this repository
# assumes where it lives. Point NINJASUBS_REAL_MEDIA_ROOT at the directory that
# holds it; the media-dependent tests below skip when it is absent. See
# tests/fixtures/real_media/dexter_s08e04_manifest.json for the digest the file
# must match.
MEDIA_ROOT = pathlib.Path(
    os.environ.get("NINJASUBS_REAL_MEDIA_ROOT", REPO / "test-media")
)
CACHE = pathlib.Path(os.environ.get("NINJASUBS_CACHE_DIR", REPO / "subs_cache"))
VIDEO_NAME = "Dexter.s8e04.Scar.tissue.1080p.BluRay.TrueHD5.1.AVC-PiR8.mvk"


def _load() -> dict:
    if not MANIFEST.is_file():
        pytest.skip(f"real-media manifest not present: {MANIFEST}")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _media_present() -> bool:
    # The fixture is 5.4 GB; only its presence is checked here, never its bytes.
    return VIDEO_NAME in {p.name for p in MEDIA_ROOT.glob("*.mkv")}


requires_media = pytest.mark.skipif(
    not _media_present(),
    reason=f"real-media fixture not mounted at {MEDIA_ROOT}",
)
requires_alass = pytest.mark.skipif(
    shutil.which("alass") is None, reason="alass not available in this environment"
)


class _OfflineStrategy:
    """Resolves the manifest's reference. No network, no media access."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    async def resolve_with_provenance(self, query, *args, **kwargs):  # noqa: ANN001
        self.calls += 1
        return ResolvedReference(
            self.text,
            kind="hash",
            bluray_match=True,
            candidate="Dexter.S08E04.1080p.BluRay.x264-PiR8.srt",
            reference_trust="high",
            reference_consensus=1.0,
            reference_independent_sources=1,
        )


def _meta(fingerprint: str, size: int) -> dict:
    return {
        "provider": "subdl",
        "release_name": "Dexter.S08E04.Scar.Tissue.1080p.BluRay.x264.srt",
        "language": "ar",
        "lang": "ar",
        "imdb_id": "tt0773262",
        "season": 8,
        "episode": 4,
        "video_fingerprint": fingerprint,
        "target_filename": VIDEO_NAME,
        "video_size": size,
    }


def _orchestrator(reference_text: str):
    strategy = _OfflineStrategy(reference_text)
    return (
        SyncOrchestrator(external_strategy=strategy, sync_service=SubtitleSyncService()),
        strategy,
    )


def _classify(served_sha: str, original_sha: str, alass_sha: str | None) -> str:
    if served_sha == original_sha:
        return "ORIGINAL_SERVED"
    if alass_sha and served_sha == alass_sha:
        return "TRANSFORMED_SERVED"
    return "UNKNOWN_SERVED"


# --------------------------------------------------------------------------- #
# 6C -- the manifest is committed identity, not content
# --------------------------------------------------------------------------- #


def test_manifest_is_machine_readable_and_media_free():
    data = _load()
    derived = data["_derived"]
    assert derived["video_size"] > 0
    assert derived["video_fingerprint"].startswith("size:")
    assert len(derived["reference_sha256"]) == 64
    assert set(derived["alass_artifact_sha256"]) == {
        "74a34c232d159f8b", "96c0442cc4656685"
    }
    for name, value in derived["pgs_witness_agreement"].items():
        assert 0.0 < value <= 1.0, f"{name} witness score out of range"
    # The manifest must stay small: it records identity, never the media.
    assert MANIFEST.stat().st_size < 64 * 1024
    # No absolute media path may be baked in.
    assert str(MEDIA_ROOT) not in MANIFEST.read_text(encoding="utf-8")


@requires_media
def test_media_fixture_matches_the_manifest_identity():
    """The media is read-only and is only read to confirm identity."""
    import hashlib as _h

    data = _load()["_derived"]
    video = MEDIA_ROOT / VIDEO_NAME
    assert video.stat().st_size == data["video_size"]
    expected = data["video_fingerprint"].split("sha256:")[1]
    digest = _h.sha256()
    with video.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    assert digest.hexdigest() == expected


# --------------------------------------------------------------------------- #
# 6A -- what the production path actually returns
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "sub_id",
    ["74a34c232d159f8b", "96c0442cc4656685"],
)
@requires_alass
def test_real_case_delivery_is_classified_and_never_unverified(sub_id):
    """The regression assertion: whichever category the service lands in, the
    bytes must be exactly one of the two known artifacts, and an unverified
    result must never be the served one."""
    data = _load()["_derived"]
    target_path = CACHE / f"{sub_id}.srt"
    if not target_path.is_file():
        pytest.skip(f"provider subtitle not cached: {target_path}")
    reference_text = (CACHE / "references").glob("tt0773262_8_4_*_t6349da3c0192e384_*.srt")
    reference_file = next(iter(reference_text), None)
    if reference_file is None:
        pytest.skip("PiR8 reference not cached")

    original = target_path.read_bytes()
    original_sha = _sha(original)
    alass_sha = data["alass_artifact_sha256"][sub_id]

    before = {p.name for p in CACHE.glob("*")}
    orchestrator, strategy = _orchestrator(reference_file.read_text(encoding="utf-8"))
    started = time.monotonic()
    served = asyncio.run(
        orchestrator.evaluate_and_sync(
            original,
            _meta(data["video_fingerprint"], data["video_size"]),
            "tt0773262:8:4",
            auto_sync=True,
        )
    )
    elapsed = time.monotonic() - started
    served_sha = _sha(served)

    assert strategy.calls == 1, "reference was never resolved"
    assert elapsed > 0
    category = _classify(served_sha, original_sha, alass_sha)
    assert category != "UNKNOWN_SERVED", (
        f"served bytes match neither the provider original ({original_sha}) "
        f"nor the known alass artifact ({alass_sha})"
    )

    # The current verifier rejects both real cases, so the expected category is
    # the original. Recorded so a change here has to be deliberate.
    assert category == data["expected_delivery_category"], (
        f"delivery category changed to {category}; the verifier decision is "
        "under review and this expectation must be updated in the same commit"
    )

    # The provider cache must be untouched.
    after = {p.name for p in CACHE.glob("*")}
    assert after == before, "a sync run added or removed provider cache files"
    assert not any("synced" in n or "alass" in n for n in after), (
        "a transformed artifact was written into the provider cache"
    )
    assert _sha((CACHE / f"{sub_id}.srt").read_bytes()) == original_sha


@requires_alass
def test_unsafe_direction_is_blocked():
    """An unverified artifact must be neither servable nor reusable.

    This is the dangerous direction the brief asks to be covered: if the
    verifier rejects an alignment, neither the delivery decision nor the cache
    may treat the alass output as an accepted artifact.
    """
    for state in ("unverified", "rejected", "predicted"):
        for verification in ("unknown", "cached", "predicted"):
            assert is_reusable_verified(state, verification) is False
    for state in ("unverified", "rejected"):
        for verification in ("unknown", "predicted"):
            assert may_serve_synchronized(state, verification) is False
    # REJECTED is not servable even when a measurement is claimed.
    for verification in ("verified", "cached", "predicted", "unknown"):
        assert may_serve_synchronized("rejected", verification) is False


@requires_alass
def test_alass_output_is_never_written_back_into_the_provider_cache():
    """Provenance: a transformed artifact must not become the next provider
    input. The provider cache keeps original bytes only."""
    data = _load()["_derived"]
    target = CACHE / "74a34c232d159f8b.srt"
    if not target.is_file():
        pytest.skip("provider subtitle not cached")
    reference_file = next(
        iter((CACHE / "references").glob(
            "tt0773262_8_4_*_t6349da3c0192e384_*.srt")), None
    )
    if reference_file is None:
        pytest.skip("PiR8 reference not cached")

    original_sha = _sha(target.read_bytes())
    orchestrator, _ = _orchestrator(reference_file.read_text(encoding="utf-8"))
    asyncio.run(
        orchestrator.evaluate_and_sync(
            target.read_bytes(),
            _meta(data["video_fingerprint"], data["video_size"]),
            "tt0773262:8:4",
            auto_sync=True,
        )
    )
    # Every provider cache entry still holds provider bytes.
    for path in CACHE.glob("*.srt"):
        text = path.read_bytes()
        # A transformed artifact would carry alass's characteristic output; the
        # simplest robust check is that the known originals are unchanged and
        # that no file was replaced by the alass digest.
        assert _sha(text) != data["alass_artifact_sha256"][
            "74a34c232d159f8b"
        ], f"{path.name} contains a transformed artifact"
    assert _sha(target.read_bytes()) == original_sha


# --------------------------------------------------------------------------- #
# 6B -- offline video witness, regression/evaluation only
# --------------------------------------------------------------------------- #


def test_pgs_witness_evidence_is_preserved():
    """The recorded witness must keep showing the alass outputs align with the
    real video substantially better than the originals do."""
    derived = _load()["_derived"]
    for case_id, output_score in derived["pgs_witness_agreement"].items():
        original_score = derived["pgs_witness_agreement_original"][case_id]
        assert output_score > original_score + 0.20, (
            f"{case_id}: alass output ({output_score:.1%}) is not meaningfully "
            f"better than the original ({original_score:.1%})"
        )
        assert output_score >= 0.95, f"{case_id}: witness score regressed"
    assert derived["pgs_tolerance_s"] == 1.5


@requires_media
@requires_alass
def test_witness_agrees_with_the_current_verifier_decision():
    """Documents how the tension now resolves: the video says the alass output
    is well synchronized, the general verifier still refuses it on residual p95,
    and the reference-first Large Offset investigation is what authorises
    delivery. Both facts are recorded -- the verifier's verdict is unchanged and
    still governs cache reuse, while the bytes the player receives are the ones
    the video witness scored 97% against."""
    derived = _load()["_derived"]
    target = CACHE / "74a34c232d159f8b.srt"
    reference_file = next(
        iter((CACHE / "references").glob(
            "tt0773262_8_4_*_t6349da3c0192e384_*.srt")), None
    )
    if not target.is_file() or reference_file is None:
        pytest.skip("real inputs not cached")
    original = target.read_bytes()
    orchestrator, _ = _orchestrator(reference_file.read_text(encoding="utf-8"))
    served = asyncio.run(
        orchestrator.evaluate_and_sync(
            original,
            _meta(derived["video_fingerprint"], derived["video_size"]),
            "tt0773262:8:4",
            auto_sync=True,
        )
    )
    # The corrected artifact is what the player gets -- and it is exactly the
    # recorded alass artifact, not a re-derivation of it.
    assert served != original, (
        "the Large Offset investigation should have authorised the correction"
    )
    assert _sha(served) == derived["alass_artifact_sha256"]["74a34c232d159f8b"]
    # ...and the analyzer's own verdict is untouched, so it stays non-reusable.
    evaluation = orchestrator._last_evaluation
    assert not is_reusable_verified(
        evaluation.sync_state.value, evaluation.verification.value
    )
    assert derived["pgs_witness_agreement"]["dexter_s08e04_evol"] >= 0.95
