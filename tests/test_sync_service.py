"""Tests for the alass-backed subtitle synchronization service."""

import os
import subprocess
from types import SimpleNamespace

import pytest

from app.services.sync_service import SubtitleSyncService, strip_sdh


def _srt(cue_text: str = "Hello", count: int = 6, start_seconds: int = 1) -> str:
    """Build a valid multi-cue SRT (>= 5 cues so the pre-alass validator passes)."""
    blocks = []
    for index in range(count):
        start = start_seconds + index * 2
        blocks.append(
            f"{index + 1}\n00:00:{start:02d},000 --> 00:00:{start + 1:02d},000\n{cue_text}"
        )
    return "\n\n".join(blocks) + "\n"


SRT = _srt("Hello")


def test_strip_sdh_removes_accessibility_indicators():
    text = "1\n00:00:01,000 --> 00:00:02,000\n[MUSIC PLAYING] John (whispering) ♪ la la ♪\n"
    out = strip_sdh(text)
    assert "MUSIC PLAYING" not in out
    assert "whispering" not in out
    assert "la la" not in out
    assert "John" in out


def test_sync_returns_none_for_empty_inputs():
    service = SubtitleSyncService()
    assert service.sync("", SRT) is None
    assert service.sync(SRT, "") is None


def test_sync_returns_output_on_success(monkeypatch):
    service = SubtitleSyncService()

    def fake_run(command, capture_output=True, timeout=None):
        out_path = command[3]
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(_shifted_output(3, text="Synced"))
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = service.sync(SRT, SRT)
    assert out is not None and "Synced" in out


def test_sync_uses_split_penalty_and_cleans_temp_files(monkeypatch):
    service = SubtitleSyncService()
    captured = {}

    def fake_run(command, capture_output=True, timeout=None):
        captured["command"] = command
        captured["temp_paths"] = list(command[1:4])
        with open(command[3], "w", encoding="utf-8") as handle:
            handle.write("1\n00:00:00,000 --> 00:00:01,000\nx\n")
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    service.sync(SRT, SRT)

    assert "--split-penalty" in captured["command"]
    assert "7.0" in captured["command"]
    # All temp files removed in finally.
    assert all(not os.path.exists(p) for p in captured["temp_paths"])


def test_alass_command_passes_reference_then_target(monkeypatch):
    """alass syntax is `alass <reference> <incorrect> <output>` (kaegi/alass).

    argv[1] must be the trusted reference file and argv[2] the target to fix;
    the output is written as `<target>.synced.srt`.
    """
    service = SubtitleSyncService()
    captured = {}

    def fake_run(command, capture_output=True, timeout=None):
        captured["command"] = command
        captured["reference"] = open(command[1], encoding="utf-8").read()
        captured["target"] = open(command[2], encoding="utf-8").read()
        with open(command[3], "w", encoding="utf-8") as handle:
            handle.write(command[2])  # content irrelevant; path is what we assert
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    service.sync(_srt("TARGET_MARKER"), _srt("REFERENCE_MARKER"))

    command = captured["command"]
    # argv[1] reference, argv[2] target, argv[3] = <target>.synced.srt
    assert "REFERENCE_MARKER" in captured["reference"]
    assert "TARGET_MARKER" not in captured["reference"]
    assert "TARGET_MARKER" in captured["target"]
    assert "REFERENCE_MARKER" not in captured["target"]
    assert command[3] == command[2] + ".synced.srt"
    # Sanity: argv is exactly [alass, reference, target, output, "--split-penalty", "7.0"].
    assert command[0] == service.alass_path
    assert command[4:] == ["--split-penalty", "7.0"]


def test_sync_returns_none_on_nonzero_exit_and_timeout(monkeypatch):
    service = SubtitleSyncService()

    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stderr=b"boom")
    )
    assert service.sync(SRT, SRT) is None

    def boom(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="alass", timeout=2.0)

    monkeypatch.setattr(subprocess, "run", boom)
    assert service.sync(SRT, SRT) is None


@pytest.mark.asyncio
async def test_sync_async_wraps_sync(monkeypatch):
    service = SubtitleSyncService()

    def fake_run(command, capture_output=True, timeout=None):
        with open(command[3], "w", encoding="utf-8") as handle:
            handle.write(_shifted_output(5, text="async"))
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = await service.sync_async(SRT, SRT, decision_kind="hash")
    assert out is not None and "async" in out


@pytest.mark.asyncio
async def test_maybe_sync_requires_server_and_user_flags(monkeypatch):
    """Auto-sync runs only when the server flag AND the user preference are on."""
    from app import main

    data = b"1\n00:00:01,000 --> 00:00:02,000\nx\n"

    monkeypatch.setattr(main.settings, "ENABLE_SUBTITLE_SYNC", False)
    assert await main._maybe_sync_subtitle(data, {"lang": "ara"}, "t", True) == data

    monkeypatch.setattr(main.settings, "ENABLE_SUBTITLE_SYNC", True)
    assert await main._maybe_sync_subtitle(data, {"lang": "ara"}, "t", False) == data

    # Both flags on but no HTTP client -> resolver unavailable -> unchanged (no crash).
    monkeypatch.setattr(main, "_http_client", None)
    assert await main._maybe_sync_subtitle(data, {"lang": "ara"}, "t", True) == data


def test_zip_extracted_reference_feeds_alass(monkeypatch):
    """A ZIP reference (as returned by SubDL) is extracted, then synced by alass."""
    import io
    import zipfile

    from app.services.reference_resolver import DualReferenceResolver

    ref_srt = _srt("ref", count=6)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("movie.en.srt", ref_srt)
    zip_bytes = buf.getvalue()

    assert zipfile.is_zipfile(io.BytesIO(zip_bytes))
    decoded = DualReferenceResolver()._decode_payload(zip_bytes, "movie.en.srt")
    assert decoded is not None and b"ref" in decoded

    service = SubtitleSyncService()

    def fake_run(command, capture_output=True, timeout=None):
        with open(command[3], "w", encoding="utf-8") as handle:
            handle.write(_shifted_output(3, text="Synced from zip"))
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    synced = service.sync(SRT, decoded.decode("utf-8"))
    assert synced is not None and "Synced from zip" in synced


def test_effective_timeout_scales_with_payload_size():
    """alass budget scales up for longs films, never below the configured floor."""
    svc = SubtitleSyncService(timeout=1.0)
    assert svc._effective_timeout("x" * 1_000, "y" * 1_000) == 4.0
    assert svc._effective_timeout("x" * 60_000, "y" * 1_000) == 6.0
    assert svc._effective_timeout("x" * 150_000, "y" * 1_000) == 30.0
    # Configured floor always wins when higher.
    assert SubtitleSyncService(timeout=12.0)._effective_timeout("x" * 1_000, "y" * 1_000) == 12.0


def test_sync_passes_effective_timeout_to_subprocess(monkeypatch):
    svc = SubtitleSyncService(timeout=1.0)
    captured: dict = {}

    def fake_run(command, capture_output=True, timeout=None):
        captured["timeout"] = timeout
        with open(command[3], "w", encoding="utf-8") as handle:
            handle.write(_shifted_output(5, text="Synced"))
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    long_target = _srt("word " * 25_000)
    svc.sync(long_target, SRT)
    assert captured["timeout"] == 30.0


def test_default_timeout_comes_from_settings():
    from app.config import settings

    assert SubtitleSyncService().timeout == float(settings.ALASS_TIMEOUT_SECONDS)


def test_sanitize_subtitle_normalises_formats():
    from app.services.sync_service import sanitize_subtitle

    # BOM + CRLF + trailing whitespace normalised.
    out = sanitize_subtitle("\ufeff1\r\n00:00:01,000 --> 00:00:02,000\r\nHi\r\n")
    assert "\r" not in out and "\ufeff" not in out
    assert out.endswith("Hi\n")

    # ASS/SSA converted to SRT.
    ass = (
        "[Script Info]\n[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,Hello\n"
    )
    ass_out = sanitize_subtitle(ass)
    assert "-->" in ass_out and "Dialogue:" not in ass_out

    # WebVTT converted to SRT timestamps.
    vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHi\n"
    vtt_out = sanitize_subtitle(vtt)
    assert "00:00:01,000 --> 00:00:02,000" in vtt_out


def test_sync_rejects_malformed_target_without_running_alass(monkeypatch):
    calls = {"n": 0}

    def fake_run(*args, **kwargs):  # pragma: no cover - must not be called
        calls["n"] += 1
        return SimpleNamespace(returncode=0, stderr=b"", stdout=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert SubtitleSyncService().sync("this is not a subtitle", SRT) is None
    assert calls["n"] == 0


def test_sync_logs_stderr_and_stdout_on_failure(monkeypatch, caplog):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stderr=b"boom reason", stdout=b"out reason"),
    )
    with caplog.at_level("WARNING"):
        assert SubtitleSyncService().sync(SRT, SRT) is None
    assert "with code 1" in caplog.text
    assert "boom reason" in caplog.text
    assert "out reason" in caplog.text


def test_sanitize_strips_inline_ass_tags_and_drops_empty_cues():
    from app.services.sync_service import sanitize_subtitle

    srt = (
        "1\n00:00:01,000 --> 00:00:02,000\n"
        "{\\fs36\\fad(300,1500)\\c&HEDE829&Comic Sans Ms}\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nReal line\n\n"
        "3\n00:00:05,000 --> 00:00:06,000\n"
    )
    out = sanitize_subtitle(srt)
    assert "{", "}" not in out
    assert "fs36" not in out
    assert "Real line" in out
    # Tag-only cue (1) and empty cue (3) dropped; numbering contiguous.
    assert out == "1\n00:00:03,000 --> 00:00:04,000\nReal line\n"


def test_sync_passes_reference_first_and_target_second(monkeypatch, caplog):
    """alass argv order must be: <reference> <target> <output>."""
    reference = _srt("REFERENCE_MARKER", start_seconds=5)
    target = _srt("TARGET_MARKER", start_seconds=5)
    captured: dict = {}

    def fake_run(command, capture_output=True, timeout=None):
        with open(command[1], encoding="utf-8") as handle:
            captured["ref"] = handle.read()
        with open(command[2], encoding="utf-8") as handle:
            captured["target"] = handle.read()
        with open(command[3], "w", encoding="utf-8") as handle:
            handle.write(_shifted_output(7, text="SYNCED"))
        return SimpleNamespace(returncode=0, stderr=b"", stdout=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with caplog.at_level("INFO"):
        out = SubtitleSyncService().sync(target, reference)

    assert "REFERENCE_MARKER" in captured["ref"]
    assert "TARGET_MARKER" in captured["target"]
    assert out is not None and "SYNCED" in out
    # First cue moved +2.00s and is logged.
    assert "applied shift" in caplog.text
    assert "+2.00s" in caplog.text


def test_sanitize_subtitle_converts_ass_payload_to_srt():
    from app.services.sync_service import sanitize_subtitle

    ass = (
        "[Script Info]\r\n"
        "; Script generated by Aegisub\r\n"
        "ScriptType: v4.00+\r\n\r\n"
        "[V4+ Styles]\r\n"
        "Format: Name, Fontname\r\n"
        "Style: Default,Arial\r\n\r\n"
        "[Events]\r\n"
        "Format: Layer, Start, End, Style, Text\r\n"
        "Dialogue: 0,0:00:06.92,0:00:09.00,Default,\u0644\u0627 \u064a\u062a\u062d\u0631\u0643 \u0623\u062d\u062f!\r\n"
        "Dialogue: 0,0:00:09.24,0:00:10.28,Default,\u0645\u0627\u0630\u0627 \u064a\u062c\u0631\u064a\n"
    )
    out = sanitize_subtitle(ass)
    assert "-->" in out
    assert "Dialogue:" not in out
    assert "\u0644\u0627 \u064a\u062a\u062d\u0631\u0643" in out


def test_sanitize_strips_trailing_position_tokens_from_timespan():
    """`00:03:03,508 --> 00:03:04,592 X1:0` must become a clean SubRip timespan."""
    from app.services.sync_service import sanitize_subtitle

    srt = (
        "1\n00:03:03,508 --> 00:03:04,592 X1:0\nHello there\n\n"
        "2\n00:03:05,000 --> 00:03:06,000 Y1:0 line:0 position:50%\nSecond line\n"
    )
    out = sanitize_subtitle(srt)
    assert "00:03:03,508 --> 00:03:04,592" in out
    assert "00:03:05,000 --> 00:03:06,000" in out
    assert "X1:0" not in out and "Y1:0" not in out
    assert "position:50%" not in out and "line:0" not in out


def test_sanitize_normalises_dotted_and_short_fractions():
    from app.services.sync_service import sanitize_subtitle

    srt = "1\n0:00:01.5 --> 0:00:02.25\nHi\n"
    out = sanitize_subtitle(srt)
    assert "00:00:01,500 --> 00:00:02,250" in out


def test_validator_rejects_too_few_cues_without_running_alass(monkeypatch, caplog):
    calls = {"n": 0}

    def fake_run(*args, **kwargs):  # pragma: no cover - must not run
        calls["n"] += 1
        return SimpleNamespace(returncode=0, stderr=b"", stdout=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    single_cue = "1\n00:00:01,000 --> 00:00:02,000\nOnly one\n"
    with caplog.at_level("WARNING"):
        assert SubtitleSyncService().sync(SRT, single_cue) is None
    assert calls["n"] == 0
    assert "[SRT_VALIDATION_ERROR]" in caplog.text


def test_validator_requires_start_before_end(monkeypatch):
    cue = "1\n00:00:02,000 --> 00:00:01,000\nBackwards\n"
    invalid = "\n\n".join(cue for _ in range(6))
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("alass must not run")),
    )
    assert SubtitleSyncService().sync(invalid, SRT) is None


def _shifted_output(start_seconds=70, count=6, text="shifted"):
    """Fake alass output: ``count`` 2s cues from ``start_seconds`` (retention-safe)."""
    blocks = []
    for i in range(count):
        start = start_seconds + i * 2
        blocks.append(f"{i + 1}\n{_fmt_ts(start)} --> {_fmt_ts(start + 2)}\n{text}")
    return "\n\n".join(blocks) + "\n"


def _mock_alass_output(monkeypatch, text):
    import subprocess

    def fake_run(command, capture_output=True, timeout=None):
        with open(command[3], "w", encoding="utf-8") as handle:
            handle.write(text)
        return SimpleNamespace(returncode=0, stderr=b"", stdout=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_sync_trusts_alass_output(monkeypatch):
    """A zero-exit alass output is served regardless of the applied shift."""
    _mock_alass_output(monkeypatch, _shifted_output())  # first cue moved by minutes
    out = SubtitleSyncService().sync(SRT, SRT, decision_kind="edition")
    assert out is not None and "shifted" in out

    _mock_alass_output(monkeypatch, _shifted_output(3))
    out = SubtitleSyncService().sync(SRT, SRT, decision_kind="edition")
    assert out is not None and "shifted" in out


def _fmt_ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    hours, rem = divmod(ms, 3600000)
    minutes, rem = divmod(rem, 60000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _span_srt(
    count: int, end_seconds: float, text: str = "line", offset_seconds: float = 0.0
) -> str:
    blocks = []
    for i in range(count):
        start = end_seconds * i / count + offset_seconds
        end = end_seconds * (i + 1) / count + offset_seconds
        blocks.append(f"{i + 1}\n{_fmt_ts(start)} --> {_fmt_ts(end)}\n{text} {i + 1}")
    return "\n\n".join(blocks) + "\n"


def _offset_srt(srt_text: str, offset_seconds: float) -> str:
    """Shift every cue of an SRT by a constant offset (uniform-shift fixtures)."""
    from app.services.sync_service import _TIMESPAN_LINE_REGEX, _parse_timestamp_ms

    offset_ms = int(round(offset_seconds * 1000))

    def _shift(match):
        start = _parse_timestamp_ms(match.group("start")) + offset_ms
        end = _parse_timestamp_ms(match.group("end")) + offset_ms
        return f"{_fmt_ts(start / 1000.0)} --> {_fmt_ts(end / 1000.0)}"

    return _TIMESPAN_LINE_REGEX.sub(_shift, srt_text)


def test_duration_gate_aborts_gross_mismatch(monkeypatch, caplog):
    """50min target vs 40min reference: alass must never run."""
    def forbidden_run(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("alass must not spawn on duration mismatch")

    import subprocess

    monkeypatch.setattr(subprocess, "run", forbidden_run)
    with caplog.at_level("WARNING"):
        out = SubtitleSyncService().sync(_span_srt(6, 3000.0), _span_srt(6, 2400.0))
    assert out is None
    assert "duration mismatch detected (target: 3000.0s, ref: 2400.0s)" in caplog.text


def test_duration_gate_allows_small_gaps(monkeypatch):
    """20s gap on a 10min short is within the 120s series floor: syncs."""
    target = _span_srt(6, 600.0, text="shifted")
    _mock_alass_output(monkeypatch, _offset_srt(target, 2.0))
    out = SubtitleSyncService().sync(target, _span_srt(6, 620.0))
    assert out is not None and "shifted" in out


def test_duration_gate_aborts_short_runtime_mismatch(monkeypatch, caplog):
    """5min gap on a 10-15min short exceeds the series tolerance: aborts."""
    import subprocess

    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no alass"))
    )
    with caplog.at_level("WARNING"):
        out = SubtitleSyncService().sync(_span_srt(6, 600.0), _span_srt(6, 900.0))
    assert out is None
    assert "duration mismatch detected (target: 600.0s, ref: 900.0s)" in caplog.text


def test_duration_gate_film_band_tolerance(monkeypatch, caplog):
    """Feature films tolerate 5min gaps but abort beyond max(300s, 8%)."""
    import subprocess

    target = _span_srt(6, 3950.0, text="shifted")
    _mock_alass_output(monkeypatch, _offset_srt(target, 2.0))
    out = SubtitleSyncService().sync(target, _span_srt(6, 3700.0))
    assert out is not None and "shifted" in out

    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no alass"))
    )
    with caplog.at_level("WARNING"):
        out = SubtitleSyncService().sync(_span_srt(6, 4100.0), _span_srt(6, 3700.0))
    assert out is None
    assert "duration mismatch detected" in caplog.text


def test_duration_ignores_trailing_credit_cues(monkeypatch):
    """98th percentile: outro cards minutes past dialogue must not abort sync."""
    from app.services.sync_service import _percentile_cue_end_ms

    bulk = _span_srt(100, 2900.0)
    credits = (
        "101\n00:54:10,000 --> 00:54:20,000\nترجمة: فلان\n\n"
        "102\n00:55:00,000 --> 00:55:10,000\nتعديل: فلان\n\n"
    )
    target = bulk + credits
    assert _percentile_cue_end_ms(target) / 1000.0 < 2950.0

    _mock_alass_output(monkeypatch, _offset_srt(target, 2.0))
    out = SubtitleSyncService().sync(target, _span_srt(100, 2950.0))
    assert out is not None
    assert out.count("-->") == target.count("-->")


def test_duration_gate_allows_compatible_runtimes(monkeypatch):
    """20s gap on a 50min episode passes both bounds and syncs."""
    target = _span_srt(6, 3000.0, text="shifted")
    _mock_alass_output(monkeypatch, _offset_srt(target, 2.0))
    out = SubtitleSyncService().sync(target, _span_srt(6, 3020.0))
    assert out is not None and "shifted" in out


def test_duration_gate_boundary_is_inclusive(monkeypatch):
    """Exactly 45.0s gap under the series tolerance proceeds (strict `>` threshold)."""
    target = _span_srt(6, 3000.0, text="shifted")
    _mock_alass_output(monkeypatch, _offset_srt(target, 2.0))
    out = SubtitleSyncService().sync(target, _span_srt(6, 3045.0))
    assert out is not None


def test_sanitize_converts_sami_payload_to_srt():
    from app.services.sync_service import sanitize_subtitle

    smi = (
        "<SAMI><HEAD><TITLE>t</TITLE></HEAD><BODY>\n"
        "<SYNC Start=1000><P Class=ENCC>Hello there\n"
        "<SYNC Start=3500><P Class=ENCC>General Kenobi\n"
        "</BODY></SAMI>"
    )
    out = sanitize_subtitle(smi)
    assert "00:00:01,000 --> 00:00:03,500" in out
    assert "Hello there" in out and "SYNC" not in out


def test_strip_sdh_never_spans_across_cues():
    """Lone markers and unclosed brackets must not eat neighbouring cues."""
    from app.services.sync_service import (
        _percentile_cue_end_ms,
        _valid_cue_count,
        sanitize_subtitle,
        strip_sdh,
    )

    body = _span_srt(60, 3000.0)
    poisoned = body.replace("line 2\n", "\u266a\n", 1).replace("line 3\n", "(looking away\n", 1)
    stripped = strip_sdh(poisoned)
    assert _valid_cue_count(stripped) == 60
    out = sanitize_subtitle(stripped)
    assert _valid_cue_count(out) == 60
    assert abs(_percentile_cue_end_ms(out) / 1000.0 - 2950.0) < 120.0


def test_strip_sdh_distant_music_markers_do_not_delete_between():
    """Two lone notes 48 cues apart must not delete everything between them."""
    from app.services.sync_service import _valid_cue_count, strip_sdh

    body = _span_srt(60, 3000.0)
    poisoned = body.replace("line 2\n", "\u266a\n", 1).replace("line 50\n", "\u266a\n", 1)
    assert _valid_cue_count(strip_sdh(poisoned)) == 60


def test_inline_ass_tag_does_not_span_lines():
    """A stray unclosed brace must not eat the next cue while seeking `}`."""
    from app.services.sync_service import _valid_cue_count, sanitize_subtitle

    srt = (
        "1\n00:00:01,000 --> 00:00:02,000\n{\\an8 stray\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nReal {\\i1}line\n"
    )
    out = sanitize_subtitle(srt)
    assert _valid_cue_count(out) == 2
    assert "Real" in out and "line" in out
    assert "00:00:03,000 --> 00:00:04,000" in out


def test_sync_trusts_large_shift(monkeypatch):
    """Even a >120s shift is served on a zero-exit alass run."""
    _mock_alass_output(monkeypatch, _shifted_output(150))
    out = SubtitleSyncService().sync(
        SRT, SRT, decision_kind="edition", is_series=True, source_confirmed=True
    )
    assert out is not None and "shifted" in out


def test_backward_jumps_detects_multi_episode_resets():
    from app.services.sync_service import _backward_jumps, _cue_spans

    single = _span_srt(30, 300.0)
    assert _backward_jumps(_cue_spans(single)) == 0

    # Three episodes concatenated: each restarts near zero -> >= 2 resets.
    pack = _span_srt(10, 300.0) + _span_srt(10, 300.0) + _span_srt(10, 300.0)
    assert _backward_jumps(_cue_spans(pack)) >= 2


def test_timeline_rejection_flags_season_pack_structure():
    from app.services.sync_service import _timeline_rejection

    target = _span_srt(30, 300.0)
    pack = _span_srt(10, 300.0) + _span_srt(10, 300.0) + _span_srt(10, 300.0)
    reason = _timeline_rejection(target, pack, decision_kind="edition", relaxed=True)
    assert reason is not None
    assert "season pack" in reason


def test_timeline_rejection_flags_cue_count_outlier():
    from app.services.sync_service import _timeline_rejection

    target = _span_srt(20, 300.0)
    ref = _span_srt(200, 300.0)  # same span, 10x the cue population
    reason = _timeline_rejection(target, ref, decision_kind="edition", relaxed=False)
    assert reason is not None
    assert "cue-count mismatch" in reason


def test_timeline_rejection_relaxed_tightens_runtime_band():
    from app.services.sync_service import _timeline_rejection

    target = _span_srt(6, 3000.0)
    ref = _span_srt(6, 3200.0)  # 200s gap: inside series 8%, outside relaxed 5%
    assert _timeline_rejection(target, ref, decision_kind="edition", relaxed=False) is None
    assert _timeline_rejection(target, ref, decision_kind="edition", relaxed=True) is not None


def test_timeline_rejection_allows_matching_cut():
    from app.services.sync_service import _timeline_rejection

    target = _span_srt(30, 3000.0)
    ref = _span_srt(30, 3020.0)
    assert _timeline_rejection(target, ref, decision_kind="edition", relaxed=True) is None
    # Exact kinds are exempt from the structural checks.
    pack = _span_srt(10, 300.0) + _span_srt(10, 300.0) + _span_srt(10, 300.0)
    assert _timeline_rejection(_span_srt(30, 300.0), pack, decision_kind="team", relaxed=False) is None


def test_timeline_rejection_source_confirmed_widens_band():
    """A bilaterally confirmed retail-disc pair tolerates a wider runtime gap."""
    from app.services.sync_service import _timeline_rejection

    target = _span_srt(30, 4232.9)
    ref = _span_srt(30, 4740.9)
    # Series 8% band rejects the ~508s gap.
    assert (
        _timeline_rejection(target, ref, decision_kind="edition", relaxed=False) is not None
    )
    # Same-source (BluRay/REMUX) master widens to 15% and passes.
    assert (
        _timeline_rejection(
            target, ref, decision_kind="edition", relaxed=False, source_confirmed=True
        )
        is None
    )


def test_sync_trusts_uniform_offset(monkeypatch):
    """A constant edition offset (small or large) is applied as alass produced it."""
    target = _span_srt(60, 6000.0)
    _mock_alass_output(monkeypatch, _offset_srt(target, 10.3))
    out = SubtitleSyncService().sync(target, target, decision_kind="edition")
    assert out is not None
    assert out != target


def test_sync_trusts_split_shift(monkeypatch):
    """A split/drifting alass alignment is still served verbatim."""
    target = _span_srt(12, 6000.0)
    blocks = target.strip().split("\n\n")
    synced = "\n\n".join(
        [_offset_srt(block, 2.0) for block in blocks[:6]]
        + [_offset_srt(block, 9.7) for block in blocks[6:]]
    ) + "\n"
    _mock_alass_output(monkeypatch, synced)
    out = SubtitleSyncService().sync(target, target, decision_kind="edition")
    assert out is not None and out.strip() == synced.strip()


def test_sync_logs_first_cue_before_alass(monkeypatch, caplog):
    """Both cue metrics are logged, each named for what it measures.

    The old label was a single ambiguous "first cue", which is how one
    incident produced an apparently contradictory pair of numbers for a single
    subtitle: a first *parsed* cue and a first *dialogue* cue are different
    measurements, and neither is wrong.
    """
    _mock_alass_output(monkeypatch, _shifted_output(5, text="SYNCED"))
    with caplog.at_level("INFO"):
        SubtitleSyncService().sync(SRT, SRT, decision_kind="edition")
    assert "target_first_parsed_cue_start_ms" in caplog.text
    assert "target_first_dialogue_start_ms" in caplog.text
    assert "target_cue_count" in caplog.text
    assert "reference_first_parsed_cue_start_ms" in caplog.text
    # The ambiguous label is gone, so a reader cannot mistake one metric for
    # the other.
    assert "first cue before alass" not in caplog.text


def test_parsed_and_dialogue_first_cue_are_distinct_metrics():
    """A non-speech head cue makes the two differ; that is the point."""
    from app.services.sync_service import _cue_starts_ms, _first_dialogue_start_ms

    head_credit = (
        "1\n00:00:02,000 --> 00:00:04,000\nsubtitle by someone\n\n"
        "2\n00:00:10,550 --> 00:00:12,000\nfirst real line\n\n"
        "3\n00:00:20,000 --> 00:00:21,000\nsecond real line\n"
    )
    assert _cue_starts_ms(head_credit, limit=1) == [2000]
    assert _first_dialogue_start_ms(head_credit) == 10_550
    # Both are available and they are not interchangeable.
    assert _cue_starts_ms(head_credit, limit=1)[0] != _first_dialogue_start_ms(head_credit)


def test_post_alass_artifact_does_not_match_the_pre_alass_figure(monkeypatch, tmp_path):
    """The cached artifact is the OUTPUT, so its timings carry the applied shift.

    This is the second half of the incident explanation: comparing a pre-alass
    log figure against a post-alass cached file compares two stages, not two
    answers.
    """
    _mock_alass_output(monkeypatch, _shifted_output(5, text="SYNCED"))
    produced = SubtitleSyncService().sync(SRT, SRT, decision_kind="edition")
    from app.services.sync_service import _cue_starts_ms

    pre = _cue_starts_ms(SRT, limit=1)
    post = _cue_starts_ms(produced if isinstance(produced, str) else produced.decode("utf-8"), limit=1)
    assert pre and post
    # The shift is visible, which is why the two must not share one label.
    assert pre != post


def test_timeline_rejection_partial_reference_skips_duration():
    """A sampled-prefix reference (first 15 min) must not be rejected on runtime."""
    from app.services.sync_service import _timeline_rejection

    target = _span_srt(30, 9000.0)  # ~150 min target
    ref = _span_srt(30, 900.0)  # ~15 min sampled prefix
    assert (
        _timeline_rejection(target, ref, decision_kind="embedded", relaxed=False) is not None
    )
    assert (
        _timeline_rejection(
            target, ref, decision_kind="embedded", relaxed=False, reference_partial=True
        )
        is None
    )
