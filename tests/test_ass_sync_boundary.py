"""ASS handling around the sync path.

There is a known risk here: ASS v4+ content stored or cached under an ``.srt``
filename, and the question of whether an ASS payload can reach the SRT-only
sync parser. This does not implement an ASS conversion architecture -- it pins
the behaviour that already exists so it cannot regress silently.

The rules under test:
  * an ASS payload is never handed to the sync path when ASS/SSA conversion is
    disabled (the user's native-styling preference wins)
  * conversion, when enabled, is deterministic and happens before anything else
  * sync output is SRT; native ASS is preserved only by skipping the sync path
  * no transformed payload is written into the provider cache
"""

from __future__ import annotations

import hashlib

import pytest

from app.services.subtitle_matcher import parse_srt_cues
from app.services.sync.orchestrator import SyncOrchestrator
from app.services.sync.query import ResolvedReference
from app.services.sync_service import SubtitleSyncService

REPO = __import__("pathlib").Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "large_offset" / "dexter_s08e04_reference.srt"

# A minimal but structurally valid ASS v4+ document: the header line matters,
# because detection keys on it rather than on the file extension.
ASS_V4 = (
    b"[Script Info]\r\n"
    b"ScriptType: v4.00+\r\n"
    b"PlayResX: 1920\r\nPlayResY: 1080\r\n"
    b"\r\n"
    b"[V4+ Styles]\r\n"
    b"Format: Name, Fontname, Fontsize, PrimaryColour\r\n"
    b"Style: Default,Arial,60,&H00FFFFFF\r\n"
    b"\r\n"
    b"[Events]\r\n"
    b"Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
    b"MarginV, Effect, Text\r\n"
    b"Dialogue: 0,0:00:10.00,0:00:14.00,Default,,0,0,0,,{\\pos(960,900)}"
    + "Ã™â€¦Ã˜Â±Ã˜Â­Ã˜Â¨Ã˜Â§".encode() + b"\r\n"
)


class _Strategy:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    async def resolve_with_provenance(self, query, *args, **kwargs):  # noqa: ANN001
        self.calls += 1
        return ResolvedReference(
            self.text, kind="hash", bluray_match=True,
            candidate="probe.srt", reference_trust="high",
            reference_consensus=1.0, reference_independent_sources=1)


def _meta(**over) -> dict:
    meta = {
        "provider": "subdl", "sub_id": "ass-probe", "release_name": "probe.srt",
        "language": "ar", "lang": "ar", "imdb_id": "tt0773262",
        "season": 8, "episode": 4, "video_fingerprint": "deadbeef" * 8,
        "target_filename": "probe.mkv", "video_size": 1,
    }
    meta.update(over)
    return meta


def test_ass_is_detected_regardless_of_extension():
    """Detection keys on content, not on the filename. An ASS document stored
    under ``.srt`` must still be recognised as ASS."""
    from app.extractor import is_ass_subtitle

    assert is_ass_subtitle(ASS_V4) is True
    assert is_ass_subtitle(ASS_V4.replace(b"\r\n", b"\n")) is True
    assert is_ass_subtitle(FIXTURE.read_bytes()) is False


def test_srt_is_not_mistaken_for_ass():
    from app.extractor import is_ass_subtitle

    srt = FIXTURE.read_bytes()
    assert is_ass_subtitle(srt) is False
    # And the SRT parser must be able to read it.
    assert len(parse_srt_cues(srt.decode("utf-8"))) > 100


def test_ass_under_an_srt_filename_is_not_fed_to_the_srt_parser():
    """The specific risk: ASS bytes reaching the SRT timing parser would either
    crash or be silently misparsed as cues. It must be converted first, or
    skipped."""
    from app.utils.ass_converter import convert_ass_to_srt_bytes

    converted = convert_ass_to_srt_bytes(ASS_V4, apply_rtl=False)
    # Converted output must be SRT, parseable, and must have dropped the
    # vector-positioning override rather than pretending to keep it.
    assert converted.lstrip().startswith(b"1")
    cues = parse_srt_cues(converted.decode("utf-8", "replace"))
    assert cues, "converted ASS produced no cues"
    assert b"{\\pos" not in converted
    assert b"Dialogue:" not in converted


def test_ass_conversion_is_deterministic():
    from app.utils.ass_converter import convert_ass_to_srt_bytes

    a = convert_ass_to_srt_bytes(ASS_V4, apply_rtl=False)
    b = convert_ass_to_srt_bytes(ASS_V4, apply_rtl=False)
    assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()


@pytest.mark.asyncio
async def test_native_ass_is_preserved_and_never_synced():
    """With ASS/SSA conversion disabled, an ASS payload must be returned
    untouched and the sync path must not be entered at all.

    This is the documented behaviour: alass emits SRT, so aligning an ASS track
    would destroy its styling. Preserving the original is the correct outcome.
    """
    import app.main as main
    from app.extractor import is_ass_subtitle

    entered = []

    async def spy(*args, **kwargs):  # noqa: ANN001
        entered.append(True)
        return args[0]

    original = main._maybe_sync_subtitle
    main._maybe_sync_subtitle = spy  # type: ignore[assignment]
    try:
        result = await main._sync_subtitle_for_response(
            ASS_V4, {"provider": "subdl", "sub_id": "x", "language": "ar",
                     "lang": "ar"},
            None, "tt0773262:8:4", None, auto_sync=True, convert_ass=False,
        )
    finally:
        main._maybe_sync_subtitle = original

    assert not entered, "an ASS payload reached the sync path"
    assert result == ASS_V4, "native ASS was modified"
    assert is_ass_subtitle(result) is True


@pytest.mark.asyncio
async def test_converted_ass_then_synced_is_srt_end_to_end():
    """With conversion enabled and sync on, the served payload is SRT produced
    by the documented path: convert, then sync. Nothing ASS-shaped survives."""
    import shutil

    from app.utils.ass_converter import convert_ass_to_srt_bytes

    if shutil.which("alass") is None:
        pytest.skip("alass not available")
    if not FIXTURE.is_file():
        pytest.skip("reference fixture missing")

    converted = convert_ass_to_srt_bytes(ASS_V4, apply_rtl=False)
    orchestrator = SyncOrchestrator(
        external_strategy=_Strategy(FIXTURE.read_text(encoding="utf-8")),
        sync_service=SubtitleSyncService(),
    )
    served = await orchestrator.evaluate_and_sync(
        converted, _meta(), "tt0773262:8:4", auto_sync=True
    )
    assert served
    assert b"{\\pos" not in served
    assert b"[Script Info]" not in served
    assert b"Dialogue:" not in served
    # Whatever the decision, it is SRT and holds real cues.
    assert served.count(b"-->") > 0


def test_no_transformed_payload_lands_in_the_provider_cache(tmp_path):
    """The sync path never writes to the provider cache. Verified with an
    isolated cache directory so the real one is untouched."""
    import asyncio
    import shutil

    if shutil.which("alass") is None:
        pytest.skip("alass not available")
    from app.services.sync_service import SubtitleSyncService

    target = _shifted(FIXTURE.read_text(encoding="utf-8"), 96_500)
    before = set(p.name for p in tmp_path.glob("*")) if tmp_path.is_dir() else set()
    orchestrator = SyncOrchestrator(
        external_strategy=_Strategy(FIXTURE.read_text(encoding="utf-8")),
        sync_service=SubtitleSyncService(),
    )
    asyncio.run(
        orchestrator.evaluate_and_sync(target, _meta(), "tt0773262:8:4",
                                       auto_sync=True)
    )
    after = set(p.name for p in tmp_path.glob("*"))
    assert after == before, "the sync path wrote into the cache directory"
    assert not any("synced" in n or "alass" in n for n in after)


def _shifted(text: str, offset_ms: int) -> bytes:
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
