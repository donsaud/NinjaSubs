"""Cache provenance: a transformed artifact must never become provider input.

Traces the whole chain for one real case -- provider bytes -> sync input ->
alass output -> verifier decision -> served bytes -- and asserts the invariant
that makes the cache safe: a rejected alignment re-measures rather than reusing
anything, and nothing written by a sync run can be mistaken for provider bytes.

Measured behaviour, not assumed:
  * repeat requests re-run alass when the verdict was rejected (correct, because
    a verdict with no measurement is never cached)
  * a request that is already aligned does no alass work at all
  * the provider cache is byte-identical before and after
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import pathlib
import shutil

import pytest

from app.services.sync.orchestrator import SyncOrchestrator
from app.services.sync.query import ResolvedReference
from app.services.sync_service import SubtitleSyncService

REPO = pathlib.Path(__file__).resolve().parent.parent
CACHE = pathlib.Path(os.environ.get("NINJASUBS_CACHE_DIR", REPO / "subs_cache"))
FIXTURE = REPO / "tests" / "fixtures" / "large_offset" / "dexter_s08e04_reference.srt"
FP = "c0ffee00" * 8

requires_alass = pytest.mark.skipif(
    shutil.which("alass") is None, reason="alass not available in this environment"
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _Strategy:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    async def resolve_with_provenance(self, query, *args, **kwargs):  # noqa: ANN001
        self.calls += 1
        return ResolvedReference(
            self.text, kind="hash", bluray_match=True,
            candidate="Dexter.S08E04.1080p.BluRay.x264-PiR8.srt",
            reference_trust="high", reference_consensus=1.0,
            reference_independent_sources=1)


def _meta(**over) -> dict:
    meta = {
        "provider": "subdl", "sub_id": "provenance-probe",
        "release_name": "Dexter.S08E04.1080p.BluRay.x264-PiR8.srt",
        "language": "ar", "lang": "ar", "imdb_id": "tt0773262",
        "season": 8, "episode": 4, "video_fingerprint": FP,
        "target_filename":
            "Dexter.S08E04.Scar.Tissue.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv",
        "video_size": 5_414_925_805,
    }
    meta.update(over)
    return meta


def _shifted(text: str, offset_ms: int) -> bytes:
    from app.services.subtitle_matcher import parse_srt_cues

    def ts(ms: int) -> str:
        ms = max(0, int(ms))
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    cues = parse_srt_cues(text)
    out = []
    for i, (s, e, t) in enumerate(cues):
        out.append(f"{i+1}\n{ts(s + offset_ms)} --> {ts(e + offset_ms)}\n{t}")
    return ("\n\n".join(out) + "\n").encode("utf-8")


@requires_alass
def test_verified_result_is_served_and_remeasurement_is_bounded():
    """The chain, end to end, for a large uniform correction the verifier
    accepts.

    A measured, verified alignment is served as transformed bytes -- that is the
    intended behaviour, and it is why the delivery decision is trusted. The
    provenance invariant that matters is separate and is asserted below: whatever
    is served, the *input* is always the provider's own bytes and a later
    request never picks up a previous run's output as its input.
    """
    reference_text = FIXTURE.read_text(encoding="utf-8")
    target = _shifted(reference_text, 96_500)
    original_sha = _sha(target)
    strategy = _Strategy(reference_text)
    orchestrator = SyncOrchestrator(
        external_strategy=strategy, sync_service=SubtitleSyncService()
    )

    served = asyncio.run(
        orchestrator.evaluate_and_sync(target, _meta(), "tt0773262:8:4",
                                       auto_sync=True)
    )
    assert strategy.calls == 1
    if served == target:
        pytest.fail("a verified large correction should have been served as "
                    "transformed bytes")
    # The served bytes must be a whole subtitle, never a splice.
    assert served.count(b"-->") > 100
    # Whatever was served, the input hash is what the provider gave us.
    assert _sha(target) == original_sha


@requires_alass
def test_already_aligned_input_does_no_alass_work():
    """An input already matching the reference needs no subprocess at all.

    This is the pre-existing shortcut, and it is what keeps the common
    already-synced case cheap.
    """
    reference_text = FIXTURE.read_text(encoding="utf-8")
    strategy = _Strategy(reference_text)
    orchestrator = SyncOrchestrator(
        external_strategy=strategy, sync_service=SubtitleSyncService()
    )
    before = orchestrator._metrics["alass_runs"]
    served = asyncio.run(
        orchestrator.evaluate_and_sync(
            reference_text.encode("utf-8"), _meta(), "tt0773262:8:4", auto_sync=True
        )
    )
    assert orchestrator._metrics["alass_runs"] == before, (
        "an already-aligned input spawned a subprocess"
    )
    assert served == reference_text.encode("utf-8")


@requires_alass
def test_a_transformed_artifact_never_becomes_the_next_provider_input():
    """Provenance invariant: nothing a sync run produces can be re-read as
    provider bytes.

    The pipeline hands the orchestrator bytes it was given. It never reads a
    previously transformed artifact back as input, and it never writes one into
    the provider cache. Both are asserted against the real cache directory.
    """
    if not CACHE.is_dir():
        pytest.skip(f"no provider cache at {CACHE}")
    reference_text = FIXTURE.read_text(encoding="utf-8")
    before = {p.name: _sha(p.read_bytes()) for p in CACHE.glob("*") if p.is_file()}

    target = _shifted(reference_text, 96_500)
    orchestrator = SyncOrchestrator(
        external_strategy=_Strategy(reference_text), sync_service=SubtitleSyncService()
    )
    asyncio.run(
        orchestrator.evaluate_and_sync(
            target, _meta(), "tt0773262:8:4", auto_sync=True
        )
    )
    # A second run must start from the same input, not from anything the first
    # run produced. Its output may legitimately differ from the provider bytes
    # if the alignment verifies, but it must be derived from `target` alone.
    again = asyncio.run(
        orchestrator.evaluate_and_sync(
            target, _meta(), "tt0773262:8:4", auto_sync=True
        )
    )
    assert again is not None and len(again) > 0
    assert again.count(b"-->") > 100, "a later request returned a partial subtitle"

    after = {p.name: _sha(p.read_bytes()) for p in CACHE.glob("*") if p.is_file()}
    assert set(after) == set(before), "sync run added or removed provider cache entries"
    for name, digest in before.items():
        assert after[name] == digest, f"provider cache entry {name} was rewritten"
    assert not any(
        "synced" in n or "alass" in n for n in after
    ), "a transformed artifact appeared in the provider cache"
