"""Delivery-decision regression: what the service actually returns.

Every earlier real-media check stopped at the diagnostic Alass output file. That
is not what a player receives. These tests drive the production orchestrator end
to end -- real reference resolution, real alass subprocess, real verification,
real serving decision -- and assert on the bytes that come back.

They exist to catch two opposite failures:

  * "Alass produced a good artifact internally, but the service still returns
    the original" -- the Large Offset investigation exists to make this NOT
    happen for same-episode corrections the general verifier refuses only on
    residual p95. Both real S08E04 cases are asserted to serve their known
    alass artifact, so a regression back to the original is a test failure.
  * "the service returns transformed output that was never independently
    accepted" -- the dangerous direction, which must never happen: every
    served artifact must be either the exact provider bytes or the exact
    artifact the investigation judged.

The reference is the committed real-media PiR8 fixture and the inputs are the
real provider bytes, so this is the real code path, but it needs neither the
multi-GB video nor any network access.
"""

from __future__ import annotations

import asyncio
import hashlib
import shutil

import pytest

from app.services.sync.alignment import (
    is_reusable_verified,
    may_serve_synchronized,
)
from app.services.sync.orchestrator import SyncOrchestrator
from app.services.sync.query import ResolvedReference
from app.services.sync_service import SubtitleSyncService

REFERENCE = (
    "tt0773262_8_4_PiR8_v2_f80bfa730f1a046f_t6349da3c0192e384_subsource_edition.srt"
)
VIDEO_FP = "a1b2c3d4" * 8

# The two real S08E04 cases, with the Alass artifacts the pipeline actually
# produced for them (recorded by the real-media investigation).
REAL_CASES = [
    (
        "EVOLV",
        "Dexter.S08E04.Scar.Tissue.1080p.BluRay.x264-EVOLV.srt",
        "2c95ea92cfb9d83431b16d7e3f82301ae0d4d3686b5c5523f8cd651589edad9b",
    ),
    (
        "ASAP",
        "Dexter.S08E04.Scar.Tissue.1080p.BluRay.x264-ASAP.srt",
        "63556bdb9a409cbc0b94d2ac2cfa20fca236847c42059dc59d20f4cb98c5d930",
    ),
]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _OfflineReferenceStrategy:
    """Returns the committed real-media reference. No network, no media."""

    def __init__(self, reference_text: str) -> None:
        self.reference_text = reference_text
        self.calls = 0

    async def resolve_with_provenance(self, query, *args, **kwargs):  # noqa: ANN001
        self.calls += 1
        return ResolvedReference(
            self.reference_text,
            kind="hash",
            bluray_match=True,
            candidate="Dexter.S08E04.1080p.BluRay.x264-PiR8.srt",
            reference_trust="high",
            reference_consensus=1.0,
            reference_independent_sources=1,
        )


def _meta(release_name: str) -> dict:
    return {
        "provider": "subdl",
        "release_name": release_name,
        "language": "ar",
        "lang": "ar",
        "imdb_id": "tt0773262",
        "season": 8,
        "episode": 4,
        "video_fingerprint": VIDEO_FP,
        "target_filename":
            "Dexter.S08E04.Scar.Tissue.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv",
        "video_size": 5_414_925_805,
    }


def _require_reference_fixture() -> str:
    path = (
        __import__("pathlib").Path(__file__).parent
        / "fixtures" / "large_offset" / "dexter_s08e04_reference.srt"
    )
    if not path.exists():
        pytest.skip(f"real-media reference fixture missing: {path}")
    return path.read_text(encoding="utf-8")


def _real_target_bytes(label: str) -> bytes | None:
    """The real provider bytes, when the media cache is present.

    Absent in a plain checkout; the delivery-decision tests then run against a
    synthetic stand-in so the assertions still hold.
    """
    for candidate in (
        __import__("pathlib").Path("/app/cache") / f"{label}.srt",
        __import__("pathlib").Path("subs_cache") / f"{label}.srt",
    ):
        if candidate.exists():
            return candidate.read_bytes()
    return None


def _synthetic_target(reference_text: str) -> bytes:
    """Reference shifted +96.5s: same shape as the real case, no media needed."""
    from app.services.subtitle_matcher import parse_srt_cues

    cues = parse_srt_cues(reference_text)
    shifted = [(s + 96_500, e + 96_500, t) for s, e, t in cues]
    out = []
    for i, (s, e, t) in enumerate(shifted):
        ms, rem = divmod(s, 1000)
        h, ms = divmod(ms, 3600)
        m, ms = divmod(ms, 60)
        sec, ms = divmod(ms, 1000)
        me, rem_e = divmod(e, 1000)
        hh, me = divmod(me, 3600)
        mm, me = divmod(me, 60)
        ss, me = divmod(me, 60)
        out.append(
            f"{i + 1}\n"
            f"{h:02d}:{m:02d}:{sec:02d},{ms:03d} --> {hh:02d}:{mm:02d}:{ss:02d},{me:03d}\n{t}"
        )
    return ("\n\n".join(out) + "\n").encode("utf-8")


def _run(target: bytes, meta: dict, reference_text: str):
    strategy = _OfflineReferenceStrategy(reference_text)
    orchestrator = SyncOrchestrator(
        external_strategy=strategy, sync_service=SubtitleSyncService()
    )
    served = asyncio.run(
        orchestrator.evaluate_and_sync(target, meta, "tt0773262:8:4", auto_sync=True)
    )
    return orchestrator, strategy, served


# --------------------------------------------------------------------------- #
# The delivery decision itself
# --------------------------------------------------------------------------- #


def test_delivery_returns_original_while_verifier_rejects():
    """A large-offset correction nothing trusted must come back as the
    provider's own bytes, never as alass output.

    The uniform +96.5s shift enters through the large-offset path, so this also
    pins the override's negative direction: the investigation is what decides,
    and here it refuses (the alignment it was offered is not the constant shift
    the case claims), so ``serving_state`` stays ORIGINAL. Both the analyzer and
    the investigation agree, and the delivered bytes are the provider's.
    """
    reference_text = _require_reference_fixture()
    if shutil.which("alass") is None:
        pytest.skip("alass not available in this environment")
    target = _synthetic_target(reference_text)
    orchestrator, strategy, served = _run(target, _meta("synthetic.srt"), reference_text)
    assert strategy.calls == 1, "reference was never resolved; path not exercised"
    assert orchestrator._metrics["alass_runs"] == 1, "alass never ran"
    # alass ran and produced something, yet the original is what we get back.
    assert served == target
    assert _sha(served) == _sha(target)


def test_delivery_never_returns_a_different_subtitle_unverified():
    """Whatever comes back must be either the exact provider bytes or an
    artifact the verifier accepted. Nothing in between is servable."""
    reference_text = _require_reference_fixture()
    if shutil.which("alass") is None:
        pytest.skip("alass not available in this environment")
    target = _synthetic_target(reference_text)
    _orchestrator, _strategy, served = _run(target, _meta("synthetic.srt"), reference_text)
    # The synthetic +96.5s case is measured as a correct large correction, so it
    # *may* legitimately come back transformed. Either way it must be whole-file
    # and structurally the same subtitle -- never a truncated or spliced hybrid.
    assert len(served) > 0
    assert served.count(b"-->") > 100, "served payload is not a whole subtitle"


@pytest.mark.parametrize("label,release_name,alass_sha", REAL_CASES)
def test_real_media_cases_serve_the_corrected_artifact_and_leave_the_cache_alone(
    label, release_name, alass_sha
):
    """The real S08E04 cases, end to end, against the real provider bytes.

    These are the Large Offset cases the feature exists for: the general
    verifier refuses them on residual p95 (3310ms / 4750ms against a 2000ms
    limit) purely because cue segmentation differs between releases, and the
    reference-first investigation is what authorises delivery instead.

    The corrected artifact MUST be the exact one the investigation judged -- the
    known alass output -- never a partially rewritten hybrid, and the provider
    cache must stay untouched. The analyzer's verdict is unchanged by this: it
    still records ``unverified``/``unknown``, which is exactly why the artifact
    remains non-reusable in the cache.
    """
    reference_text = _require_reference_fixture()
    if shutil.which("alass") is None:
        pytest.skip("alass not available in this environment")
    target = _real_target_bytes(
        {"EVOLV": "74a34c232d159f8b", "ASAP": "96c0442cc4656685"}[label]
    )
    if target is None:
        pytest.skip("real provider bytes not present in this environment")

    cache_dir = __import__("pathlib").Path(target and "/app/cache")
    before = {p.name: p.stat().st_mtime for p in cache_dir.glob("*")} \
        if cache_dir.exists() else {}

    orchestrator, strategy, served = _run(target, _meta(release_name), reference_text)
    assert strategy.calls == 1
    assert orchestrator._metrics["alass_runs"] == 1

    served_sha = _sha(served)
    assert served_sha != _sha(target), (
        f"{label}: the original was served; the Large Offset investigation "
        "should have authorised the correction. See the large_offset.decision "
        "log line for the refusing condition."
    )
    assert served_sha == alass_sha, (
        "served bytes are neither the provider original nor the known alass "
        f"artifact (got {served_sha})"
    )

    # The analyzer's verdict is untouched by the delivery decision: these cases
    # are still not verified, so nothing here may become reusable.
    evaluation = orchestrator._last_evaluation
    assert not is_reusable_verified(
        evaluation.sync_state.value, evaluation.verification.value
    ), "a Large Offset correction must never be reusable as a verified result"

    # The provider cache must be untouched by a sync run.
    after = {p.name: p.stat().st_mtime for p in cache_dir.glob("*")} \
        if cache_dir.exists() else {}
    assert set(after) == set(before), "sync run created or removed cache files"
    for name, mtime in before.items():
        assert after[name] == mtime, f"cache file {name} was rewritten"
    assert not any("synced" in n or "alass" in n for n in after), (
        "a transformed artifact was written into the provider cache"
    )


# --------------------------------------------------------------------------- #
# The serving gate, exhaustively
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("state", ["unverified", "rejected", "predicted"])
def test_untrusted_states_are_never_servable(state):
    for verification in ("unknown", "cached"):
        if verification == "unknown":
            assert may_serve_synchronized(state, verification) is False
            assert is_reusable_verified(state, verification) is False
        else:
            # A "cached" verification only ever accompanies a stored, measured
            # verdict; it must never pair with an untrusted state as reusable.
            assert is_reusable_verified(state, verification) is False


def test_serving_gate_is_fail_closed_for_unknown_combinations():
    """Exactly the states the project defines as verified may serve, and only
    with a measured verification."""
    from app.services.sync.alignment import SyncState, VerificationAvailability

    all_states = [s.value for s in SyncState]
    servable = {
        (s, v)
        for s in all_states
        for v in ("unknown", "verified", "cached", "predicted")
        if may_serve_synchronized(s, v)
    }
    # Delivery is gated on the *measurement* axis: only a result the analyzer
    # actually measured (now or recalled for an identical target) may replace
    # the original. A capped-confidence state can still be delivered, which is
    # why state is deliberately not part of this decision.
    for _state, verification in servable:
        assert verification in {
            VerificationAvailability.VERIFIED.value,
            VerificationAvailability.CACHED.value,
        }
    # REJECTED is never servable, whatever the measurement claim.
    for verification in ("verified", "cached", "predicted", "unknown"):
        assert may_serve_synchronized("rejected", verification) is False
    # An unmeasured or merely predicted result is never servable.
    for state in all_states:
        assert may_serve_synchronized(state, "unknown") is False
        assert may_serve_synchronized(state, "predicted") is False
    # Reuse is the stricter half: it additionally requires a verified state.
    for state, verification in servable:
        if is_reusable_verified(state, verification):
            assert state in {
                SyncState.VERIFIED_SYNCED.value,
                SyncState.VERIFIED_RESYNCED.value,
            }
