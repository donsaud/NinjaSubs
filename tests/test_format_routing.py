"""Regression tests for the zero-cue / format-routing defect.

A real request for ``ab461b8781aafd89`` logged ``target_cue_count=562`` before
alass and ``target_parsed_cue_count=0`` after it. Both numbers were correct about
*different byte strings*:

  alass received   40,687 chars -> 562 cues   (sanitize_subtitle converted it)
  verifier received 50,721 chars -> 0 cues    (raw ASS, never converted)

The file is ASS/SSA content stored under an ``.srt`` name. The ASS guard in
``main._sync_subtitle_for_response`` only applied when the user had *disabled*
ASS conversion, so with the default (conversion enabled) the raw payload fell
through to an SRT-only pipeline.

These tests pin the corrected routing and, where the real artifact is present,
reproduce the incident exactly. See ``docs/zero_cue_root_cause.md``.
"""

from __future__ import annotations

import asyncio
import hashlib
import pathlib
import shutil

import pytest

import app.main as main
from app.extractor import is_ass_subtitle
from app.services.subtitle_matcher import parse_srt_cues
from app.services.sync_service import SubtitleSyncService, sanitize_subtitle
from app.utils.ass_converter import convert_ass_to_srt_bytes

REPO = pathlib.Path(__file__).resolve().parent.parent
CACHE = pathlib.Path(__import__("os").environ.get(
    "NINJASUBS_CACHE_DIR", REPO / "subs_cache"
))
INCIDENT_ID = "ab461b8781aafd89"
INCIDENT_PATH = CACHE / f"{INCIDENT_ID}.srt"
SRT_FIXTURE = REPO / "tests" / "fixtures" / "large_offset" / "dexter_s08e04_reference.srt"

requires_alass = pytest.mark.skipif(
    shutil.which("alass") is None, reason="alass not available in this environment"
)

ASS_V4_PLUS = (
    "[Script Info]\n"
    "ScriptType: v4.00+\n"
    "PlayResX: 1920\nPlayResY: 1080\n\n"
    "[V4+ Styles]\n"
    "Format: Name, Fontname, Fontsize, PrimaryColour\n"
    "Style: Default,Arial,60,&H00FFFFFF\n\n"
    "[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    "Dialogue: 0,0:00:10.00,0:00:14.00,Default,,0,0,0,,{\\pos(960,900)}"
    + "مرحبا"
    + "\n"
).encode()

# A well-formed SSA v4 document: the Format line must declare every field the
# Dialogue line supplies, which is what a real SSA release looks like.
SSA_V4 = (
    "[Script Info]\n"
    "ScriptType: v4.00\n"
    "PlayResX: 1920\nPlayResY: 1080\n\n"
    "[V4 Styles]\n"
    "Format: Name, Fontname, Fontsize, PrimaryColour, Bold\n"
    "Style: Default,Arial,60,&H00FFFFFF,0\n\n"
    "[Events]\n"
    "Format: Marked, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    "Dialogue: Marked=0,0:00:10.00,0:00:14.00,Default,,0,0,0,,مرحبا\n"
).encode()


def _srt_text() -> str:
    return SRT_FIXTURE.read_text(encoding="utf-8")


def _evolv() -> pathlib.Path:
    """The real EVOLV artifact for this episode.

    Alass corrects it, the general verifier does not trust the residual p95,
    and the reference-first Large Offset investigation is what authorises
    delivery. This is the observed ALASS_CORRECTED_LARGE_OFFSET case.
    """
    path = CACHE / "74a34c232d159f8b.srt"
    if not path.exists():
        pytest.skip("EVOLV artifact not cached")
    return path


def _run_sync(payload: bytes, *, convert_ass: bool) -> tuple[bytes, list[list]]:
    """Run the real response-side sync entry point, capturing what it forwards."""
    seen: list[list] = []
    original = main._maybe_sync_subtitle

    async def spy(sub_bytes: bytes, *args, **kwargs):  # noqa: ANN001
        seen.append(parse_srt_cues(sub_bytes.decode("utf-8", errors="replace")))
        return sub_bytes

    main._maybe_sync_subtitle = spy  # type: ignore[assignment]
    try:
        out = asyncio.run(
            main._sync_subtitle_for_response(
                payload,
                {"provider": "subdl", "sub_id": "probe", "language": "ar", "lang": "ar"},
                None,
                "tt0773262:8:4",
                None,
                auto_sync=True,
                convert_ass=convert_ass,
            )
        )
    finally:
        main._maybe_sync_subtitle = original
    return out, seen


# --------------------------------------------------------------------------- #
# The exact incident
# --------------------------------------------------------------------------- #


def test_incident_artifact_is_ass_under_an_srt_name():
    if not INCIDENT_PATH.is_file():
        pytest.skip(f"incident artifact not cached: {INCIDENT_PATH}")
    raw = INCIDENT_PATH.read_bytes()
    assert raw.lstrip()[:13].lower() == b"[script info]"
    assert is_ass_subtitle(raw) is True
    # The SRT parser cannot read it, which is the whole defect.
    assert len(parse_srt_cues(raw.decode("utf-8", errors="replace"))) == 0


def test_incident_artifact_splits_into_two_different_strings():
    """The two log lines described different byte strings. Pin that."""
    if not INCIDENT_PATH.is_file():
        pytest.skip(f"incident artifact not cached: {INCIDENT_PATH}")
    from app.extractor import decode_subtitle_bytes

    raw = INCIDENT_PATH.read_bytes()
    target_text = decode_subtitle_bytes(raw, lang="ar")
    sanitized = sanitize_subtitle(target_text)
    assert len(parse_srt_cues(target_text)) == 0, "raw ASS should not parse as SRT"
    assert len(parse_srt_cues(sanitized)) > 0, "sanitized form must parse"
    assert len(target_text) != len(sanitized)


def test_incident_routes_by_content_in_both_directions():
    if not INCIDENT_PATH.is_file():
        pytest.skip(f"incident artifact not cached: {INCIDENT_PATH}")
    raw = INCIDENT_PATH.read_bytes()

    # conversion enabled -> the sync stage must receive parseable cues
    _out, seen = _run_sync(raw, convert_ass=True)
    assert seen, "sync was not entered"
    assert len(seen[0]) > 0, (
        "sync received unparseable input: the 0-cue defect is back"
    )

    # conversion disabled -> native ASS preserved, sync never entered
    out, seen = _run_sync(raw, convert_ass=False)
    assert not seen, "native ASS must not enter the sync path"
    assert out == raw, "native ASS must be returned byte-identical"
    assert is_ass_subtitle(out) is True


# --------------------------------------------------------------------------- #
# A-H: the matrix the task asks for
# --------------------------------------------------------------------------- #


def test_A_normal_srt_reaches_sync_unchanged():
    _out, seen = _run_sync(_srt_text().encode("utf-8"), convert_ass=True)
    assert seen, "SRT must reach the sync stage"
    assert len(seen[0]) == len(parse_srt_cues(_srt_text()))


def test_B_ass_content_with_ass_extension():
    _out, seen = _run_sync(ASS_V4_PLUS, convert_ass=True)
    assert seen and len(seen[0]) > 0, "ASS must be converted before sync"


def test_C_ass_content_mislabelled_as_srt():
    """Extension is irrelevant; detection is by content."""
    assert ASS_V4_PLUS.endswith(b"\n")
    _out, seen = _run_sync(ASS_V4_PLUS, convert_ass=True)
    assert seen and len(seen[0]) > 0
    out, seen_off = _run_sync(ASS_V4_PLUS, convert_ass=False)
    assert not seen_off and out == ASS_V4_PLUS


def test_D_ssa_content_is_handled_like_ass():
    assert is_ass_subtitle(SSA_V4) is True
    _out, seen = _run_sync(SSA_V4, convert_ass=True)
    assert seen and len(seen[0]) > 0
    out, seen_off = _run_sync(SSA_V4, convert_ass=False)
    assert not seen_off and out == SSA_V4


@requires_alass
def test_E_srt_syncs_normally_with_alass():
    """An ordinary SRT must still sync, end to end."""
    reference_file = next(
        (CACHE / "references").glob("tt0773262_8_4_*_t6349da3c0192e384_*.srt"), None
    )
    if reference_file is None:
        pytest.skip("PiR8 reference not cached")
    service = SubtitleSyncService()
    result = service.sync(_srt_text(), reference_file.read_text(encoding="utf-8"))
    assert result, "SRT sync produced no output"
    assert len(parse_srt_cues(result)) > 0


def test_F_ass_with_sync_enabled_is_converted_not_preserved():
    out, seen = _run_sync(ASS_V4_PLUS, convert_ass=True)
    assert seen and len(seen[0]) > 0
    assert out != ASS_V4_PLUS, "an opted-in conversion must change the payload"
    assert b"[Script Info]" not in out


def test_G_ass_with_sync_disabled_is_preserved():
    out, seen = _run_sync(ASS_V4_PLUS, convert_ass=False)
    assert not seen, "sync must not run when native ASS is preserved"
    assert out == ASS_V4_PLUS


def test_H_converted_ass_is_deterministic_and_drops_positioning():
    a = convert_ass_to_srt_bytes(ASS_V4_PLUS, apply_rtl=False)
    b = convert_ass_to_srt_bytes(ASS_V4_PLUS, apply_rtl=False)
    assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()
    assert b"{\\pos" not in a
    assert len(parse_srt_cues(a.decode("utf-8", errors="replace"))) > 0


def test_native_ass_preservation_does_not_lose_positioning():
    """The whole point of preserving native ASS is that styling survives."""
    out, seen = _run_sync(ASS_V4_PLUS, convert_ass=False)
    assert b"{\\pos(960,900)}" in out
    assert b"[V4+ Styles]" in out
    assert b"\\fs" not in out.split(b"[Events]")[0].replace(b"Fontsize", b"")


def test_ordinary_srt_is_never_mistaken_for_ass():
    assert is_ass_subtitle(_srt_text().encode("utf-8")) is False
    _out, seen = _run_sync(_srt_text().encode("utf-8"), convert_ass=True)
    assert seen and len(seen[0]) == len(parse_srt_cues(_srt_text()))


@requires_alass
def test_conversion_does_not_contaminate_the_provider_cache():
    if not CACHE.is_dir():
        pytest.skip(f"no cache directory at {CACHE}")
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in CACHE.glob("*") if p.is_file()}
    if not INCIDENT_PATH.is_file():
        pytest.skip("incident artifact not cached")
    original_sha = hashlib.sha256(INCIDENT_PATH.read_bytes()).hexdigest()
    _run_sync(INCIDENT_PATH.read_bytes(), convert_ass=True)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in CACHE.glob("*") if p.is_file()}
    assert set(after) == set(before)
    for name, digest in before.items():
        assert after[name] == digest, f"cache entry {name} was rewritten"
    assert hashlib.sha256(INCIDENT_PATH.read_bytes()).hexdigest() == original_sha
    assert not any("synced" in n or "alass" in n for n in after)


# --------------------------------------------------------------------------- #
# Served-byte observability
# --------------------------------------------------------------------------- #


def test_delivery_log_names_both_hashes_and_the_decision():
    """The delivery line must identify which bytes were served, using digests
    only -- never subtitle text."""
    import logging

    from app.services.subtitle_matcher import parse_srt_cues as _p

    def ts(ms):
        ms = max(0, int(ms))
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    reference_file = next(
        (CACHE / "references").glob("tt0773262_8_4_*_t6349da3c0192e384_*.srt"), None
    )
    if reference_file is None:
        pytest.skip("PiR8 reference not cached")
    import asyncio

    from app.services.sync.orchestrator import SyncOrchestrator
    from app.services.sync.query import ResolvedReference
    from app.services.sync_service import SubtitleSyncService

    class _S:
        def __init__(self, t):
            self.t = t

        async def resolve_with_provenance(self, q, *a, **k):
            return ResolvedReference(
                self.t, kind="hash", bluray_match=True, candidate="x.srt",
                reference_trust="high", reference_consensus=1.0,
                reference_independent_sources=1,
            )

    def _meta(**o):
        m = {"provider": "subdl", "sub_id": "x", "release_name": "x.srt",
             "language": "ar", "lang": "ar", "imdb_id": "tt0773262", "season": 8,
             "episode": 4, "video_fingerprint": "a1" * 32,
             "target_filename": "v.mkv", "video_size": 1}
        m.update(o)
        return m

    if shutil.which("alass") is None:
        pytest.skip("alass not available; the delivery line is only reached on a real run")
    reference_text = reference_file.read_text(encoding="utf-8")
    cues = _p(reference_text)
    shifted = [(s + 96_500, e + 96_500, t) for s, e, t in cues]
    body = "\n\n".join(
        f"{i+1}\n{ts(s)} --> {ts(e)}\n{t}" for i, (s, e, t) in enumerate(shifted)
    ).encode("utf-8") + b"\n"

    def _drifted(cues, shift_ms=96_500, drift_ms=6_000):
        """A large-offset target whose correction is NOT a constant shift."""
        n = len(cues)
        parts = []
        for i, (s, e, t) in enumerate(cues):
            extra = int(drift_ms * (i / max(1, n - 1)))
            parts.append(
                f"{i+1}\n{ts(s + shift_ms + extra)} --> {ts(e + shift_ms + extra)}\n{t}"
            )
        return ("\n\n".join(parts) + "\n").encode("utf-8")

    captured: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    orch_logger = logging.getLogger("app.services.sync.orchestrator")
    handler = _Capture()
    previous_level = orch_logger.level
    orch_logger.setLevel(logging.INFO)
    orch_logger.addHandler(handler)
    try:
        # 1. A correction everything trusted: the transformed bytes go out.
        orch = SyncOrchestrator(
            external_strategy=_S(reference_text),
            sync_service=SubtitleSyncService(),
        )
        served = asyncio.run(
            orch.evaluate_and_sync(body, _meta(), "tt0773262:8:4", auto_sync=True)
        )
        lines = [m for m in captured if "[sync] delivery " in m]

        # 2. The Large Offset direction: the real EVOLV case, which the general
        #    verifier refuses on residual p95 but the reference-first
        #    investigation authorises. The corrected artifact is served, and the
        #    line must say so explicitly rather than only "TRANSFORMED".
        captured.clear()
        orch2 = SyncOrchestrator(
            external_strategy=_S(reference_text),
            sync_service=SubtitleSyncService(),
        )
        corrected = asyncio.run(
            orch2.evaluate_and_sync(
                _evolv().read_bytes(), _meta(), "tt0773262:8:4", auto_sync=True
            )
        )
        lines2 = [m for m in captured if "[sync] delivery " in m]

        # 3. The refused direction: alass still runs on a large-offset candidate,
        #    the investigation refuses it, and the provider bytes are served.
        captured.clear()
        orch3 = SyncOrchestrator(
            external_strategy=_S(reference_text),
            sync_service=SubtitleSyncService(),
        )
        drifted = _drifted(cues)
        original = asyncio.run(
            orch3.evaluate_and_sync(drifted, _meta(), "tt0773262:8:4", auto_sync=True)
        )
        lines3 = [m for m in captured if "[sync] delivery " in m]
    finally:
        orch_logger.removeHandler(handler)
        orch_logger.setLevel(previous_level)

    assert lines, "no delivery log line was emitted"
    line = lines[-1]
    # The safe direction: a verified correction serves the transformed bytes.
    assert "decision=TRANSFORMED" in line
    assert "served_is_transformed=True" in line
    assert "served_sha256=" in line
    assert "original_sha256=" in line
    assert "transformed_sha256=" in line
    served_digest = line.split("served_sha256=")[1].split()[0]
    assert served_digest == hashlib.sha256(served).hexdigest()[:16]
    # Never the subtitle text itself.
    assert "Ù…Ø±Ø­Ø¨Ø§" not in line
    assert "Dialogue:" not in line

    # The Large Offset direction: served, and named as such. It is served UNDER
    # the analyzer's own unchanged verdict, so the line must still read
    # unverified -- a "verified" claim here would mean the state was rewritten
    # to make delivery pass, which is the failure this feature must not cause.
    assert lines2, "no delivery line for the corrected case"
    line2 = lines2[-1]
    assert "decision=TRANSFORMED" in line2
    assert "served_is_transformed=True" in line2
    assert "serving_state=alass_corrected_large_offset" in line2
    assert "state=unverified" in line2, (
        "a Large Offset correction must be delivered under the analyzer's own "
        f"verdict, never a rewritten one: {line2}"
    )
    digest2 = line2.split("served_sha256=")[1].split()[0]
    assert digest2 == hashlib.sha256(corrected).hexdigest()[:16]

    # The refused direction: alass ran, the investigation refused, original out.
    assert lines3, "no delivery line for the refused case"
    line3 = lines3[-1]
    assert "served_is_transformed=False" in line3 or "decision=ORIGINAL" in line3
    assert "serving_state=original" in line3
    digest3 = line3.split("served_sha256=")[1].split()[0]
    assert digest3 == hashlib.sha256(original).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Full routing matrix
# --------------------------------------------------------------------------- #
#
# These assert the invariant the incident violated, over every combination of
# input format, conversion preference and AutoSync preference:
#
#   raw ASS/SSA must never enter the SRT-only sync path.
#
# Each mutation in tools/run_safety_mutations.py that targets the router must
# turn at least one of these red.


def _router_orchestrator(monkeypatch, reference_text="1\n00:00:00,000 --> 00:00:01,000\nx\n"):
    """Install an orchestrator whose spy records what the sync path receives."""
    from app.services.sync.orchestrator import SyncOrchestrator
    from app.services.sync.query import ResolvedReference

    class _Stub:
        async def resolve_with_provenance(self, q, *a, **k):
            return ResolvedReference(
                reference_text, kind="hash", bluray_match=True, candidate="stub.srt",
                reference_trust="high", reference_consensus=1.0,
                reference_independent_sources=1,
            )

    seen: list[dict] = []

    async def spy(self, sub_bytes, meta, target_id, auto_sync=False):
        from app.extractor import decode_subtitle_bytes

        text = decode_subtitle_bytes(sub_bytes, lang="ar")
        seen.append(
            {
                "is_ass": is_ass_subtitle(sub_bytes),
                "verifier_cues": len(parse_srt_cues(text)),
                "bytes": len(sub_bytes),
            }
        )
        return sub_bytes

    monkeypatch.setattr(main, "_get_sync_orchestrator",
                        lambda: SyncOrchestrator(external_strategy=_Stub()))
    monkeypatch.setattr(main.SyncOrchestrator, "evaluate_and_sync", spy)
    return seen


def _route(payload, *, convert_ass, auto_sync):
    return asyncio.run(
        main._sync_subtitle_for_response(
            payload, {"lang": "ar"}, {}, "tt0000000:1:1", None, auto_sync, convert_ass
        )
    )


def test_router_never_feeds_ass_to_the_srt_only_sync_path(monkeypatch):
    """ASS and SSA, with and without conversion, on every AutoSync setting."""
    for label, payload in (
        ("ASS", ASS_V4_PLUS), ("SSA", SSA_V4)
    ):
        for convert_ass in (True, False):
            for auto_sync in (True, False):
                seen = _router_orchestrator(monkeypatch)
                out = _route(payload, convert_ass=convert_ass, auto_sync=auto_sync)
                ctx = f"{label} convert={convert_ass} auto={auto_sync}"
                assert not (seen and seen[0]["is_ass"]), (
                    f"raw {label} entered the SRT-only sync path ({ctx})"
                )
                if seen:
                    assert seen[0]["verifier_cues"] > 0, (
                        f"verifier received 0 cues on a path it was meant to "
                        f"parse ({ctx})"
                    )
                # Preference decides the container, nothing else does.
                assert out_is_ass(out) is (not convert_ass), (
                    f"wrong container for {ctx}"
                )


def out_is_ass(payload: bytes) -> bool:
    return is_ass_subtitle(payload)


def test_router_converts_before_sync_not_after(monkeypatch):
    """With conversion enabled, the sync path must already hold SRT.

    This is the ordering assertion. If conversion moved back after the sync
    attempt -- back into the response builder -- the spy would see ASS here.
    """
    seen = _router_orchestrator(monkeypatch)
    out = _route(ASS_V4_PLUS, convert_ass=True, auto_sync=True)
    assert seen, "sync was never reached, so ordering was not exercised"
    assert not seen[0]["is_ass"], "sync received raw ASS; conversion ran too late"
    assert seen[0]["verifier_cues"] > 0
    assert not is_ass_subtitle(out), "converted payload served as ASS"


def test_router_preserves_native_ass_byte_identically(monkeypatch):
    """With conversion disabled, the provider's exact bytes must come back.

    Conversion is not permitted as a side effect: {\\pos} and styling belong to
    the user who asked to keep them.
    """
    _router_orchestrator(monkeypatch)
    for label, payload in (("ASS", ASS_V4_PLUS), ("SSA", SSA_V4)):
        out = _route(payload, convert_ass=False, auto_sync=True)
        assert out == payload, f"native {label} was not preserved byte-identically"
        assert is_ass_subtitle(out), f"native {label} lost its identity"


def test_router_detects_ass_by_content_not_extension(monkeypatch):
    """ASS stored under an .srt name must be routed exactly like a .ass file.

    The router never sees a filename, which is the point: detection is by
    content, so a mislabeled extension cannot change the outcome.
    """
    by_extension = _route(ASS_V4_PLUS, convert_ass=True, auto_sync=True)
    _router_orchestrator(monkeypatch)
    mislabeled = _route(ASS_V4_PLUS, convert_ass=True, auto_sync=True)
    assert by_extension == mislabeled
    assert not is_ass_subtitle(mislabeled)


@pytest.mark.skipif(not INCIDENT_PATH.exists(), reason="incident artifact not cached")
def test_real_incident_artifact_reaches_the_verifier_with_all_cues(monkeypatch):
    """The exact historical contradiction, inverted.

    ab461b8781aafd89 is 68,310 bytes of ASS/SSA under an .srt name. Raw, the
    SRT parser reads 0 cues over 50,721 characters -- which is precisely the
    `target_parsed_cue_count=0` in the original log. Through the corrected
    router the same file must reach the verifier with its 562 cues, so the
    562-versus-0 contradiction cannot reappear.
    """
    raw = INCIDENT_PATH.read_bytes()

    # The pre-fix failure, reproduced explicitly.
    assert is_ass_subtitle(raw) is True
    assert len(raw) == 68_310
    raw_text = raw.decode("utf-8")
    assert len(raw_text) == 50_721
    assert len(parse_srt_cues(raw_text)) == 0, (
        "the incident artifact no longer reproduces 0 raw cues; the regression "
        "below may be testing a stale premise"
    )

    # Through the corrected router the same bytes yield cues for the verifier.
    seen = _router_orchestrator(monkeypatch)
    out = _route(raw, convert_ass=True, auto_sync=True)
    assert seen, "sync was never reached"
    assert not seen[0]["is_ass"], "raw ASS reached the sync path again"
    assert seen[0]["verifier_cues"] == 562, (
        f"expected 562 cues at the verifier, got {seen[0]['verifier_cues']}"
    )
    assert len(parse_srt_cues(out.decode("utf-8"))) == 562

    # And preserving it stays byte-identical.
    _router_orchestrator(monkeypatch)
    assert _route(raw, convert_ass=False, auto_sync=True) == raw


def test_no_unnecessary_conversion_for_srt_input(monkeypatch):
    """An SRT must pass through untouched -- the router must not churn it."""
    seen = _router_orchestrator(monkeypatch)
    srt = SRT_FIXTURE.read_bytes()
    out = _route(srt, convert_ass=True, auto_sync=True)
    assert seen and not seen[0]["is_ass"]
    assert out == srt, "SRT was modified by the format router"
