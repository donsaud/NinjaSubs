"""Unit tests for in-memory ZIP unpacking, Zip Slip defense, and Arabic transcoding."""

import io
import zipfile

import pytest

from app.extractor import (
    SubtitleExtractionError,
    decode_subtitle_bytes,
    extract_srt_from_zip,
    is_zip_slip_attempt,
    transcode_to_utf8,
)


def create_mock_zip(files_dict: dict) -> bytes:
    """Helper to create an in-memory ZIP archive from a dict of {filename: bytes_or_str}."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files_dict.items():
            if isinstance(content, str):
                content = content.encode("utf-8")
            zf.writestr(name, content)
    return buf.getvalue()


def test_extract_single_valid_srt():
    """Verify clean extraction of a single .srt file in archive."""
    srt_content = "1\n00:00:01,000 --> 00:00:03,000\nمرحبا بك في ستريميو\n"
    zip_bytes = create_mock_zip({"movie.srt": srt_content})

    extracted = extract_srt_from_zip(zip_bytes)
    assert extracted.decode("utf-8") == srt_content


def test_zip_slip_protection_detected():
    """Verify is_zip_slip_attempt flags traversal attacks."""
    assert is_zip_slip_attempt("../../evil.srt") is True
    assert is_zip_slip_attempt("..\\..\\evil.srt") is True
    assert is_zip_slip_attempt("/etc/passwd.srt") is True
    assert is_zip_slip_attempt("C:\\Windows\\evil.srt") is True
    assert is_zip_slip_attempt("normal_folder/subtitle.srt") is False
    assert is_zip_slip_attempt("subtitle.srt") is False


def test_extract_ignores_zip_slip_and_extracts_safe_srt():
    """Verify archive containing traversal attacks ignores evil entries and extracts safe srt."""
    safe_content = "1\n00:00:01,000 --> 00:00:02,000\nSafe Subtitle\n"
    evil_content = "Evil Traversal Subtitle"

    zip_bytes = create_mock_zip(
        {
            "../../etc/evil.srt": evil_content,
            "subtitles/safe_release.srt": safe_content,
        }
    )

    extracted = extract_srt_from_zip(zip_bytes)
    assert extracted.decode("utf-8") == safe_content


def test_extract_ignores_directories_and_non_srt_files():
    """Verify extractor ignores directory entries, nfo, and non-SRT files."""
    valid_srt = "1\n00:00:01,000 --> 00:00:04,000\nترجمة عربية معتمدة\n"

    zip_bytes = create_mock_zip(
        {
            "subtitles/": b"",  # directory
            "subtitles/movie.nfo": "NFO text info",
            "subtitles/movie.txt": "Read me text",
            "subtitles/movie.sub": "MicroDVD sub format",
            "__MACOSX/._movie.srt": b"metadata",
            "subtitles/Arabic_Release.srt": valid_srt,
        }
    )

    extracted = extract_srt_from_zip(zip_bytes)
    assert extracted.decode("utf-8") == valid_srt


def test_extract_raises_when_no_valid_srt():
    """Verify SubtitleExtractionError is raised when no .srt is present."""
    zip_bytes = create_mock_zip(
        {
            "info.nfo": "Information only",
            "image.jpg": b"\xff\xd8\xff",
        }
    )

    with pytest.raises(SubtitleExtractionError, match="No valid subtitle files"):
        extract_srt_from_zip(zip_bytes)


def test_arabic_cp1256_transcoding():
    """Verify legacy Windows-1256 Arabic subtitle transcoding to clean UTF-8."""
    arabic_text = "1\n00:00:01,000 --> 00:00:05,000\nأهلاً وسهلاً بكم في هذا الفيلم الرائع\n"
    cp1256_bytes = arabic_text.encode("cp1256")

    # Raw bytes are NOT valid UTF-8
    with pytest.raises(UnicodeDecodeError):
        cp1256_bytes.decode("utf-8")

    # Transcoder converts to valid UTF-8
    transcoded = transcode_to_utf8(cp1256_bytes)
    decoded_utf8 = transcoded.decode("utf-8")
    assert decoded_utf8 == arabic_text


def test_multi_srt_episode_selection():
    """Verify correct episode is selected from multi-subtitle season pack."""
    ep1_content = "Episode 1 Subtitles"
    ep2_content = "Episode 2 Subtitles"
    ep3_content = "Episode 3 Subtitles"

    zip_bytes = create_mock_zip(
        {
            "House.of.the.Dragon.S02E01.1080p.srt": ep1_content,
            "House.of.the.Dragon.S02E02.1080p.srt": ep2_content,
            "House.of.the.Dragon.S02E03.1080p.srt": ep3_content,
        }
    )

    extracted = extract_srt_from_zip(
        zip_bytes,
        target_filename="House.of.the.Dragon.S02E02.1080p.WEB-DL.srt",
        season=2,
        episode=2,
    )
    assert extracted.decode("utf-8") == ep2_content


def test_extract_ass_preserves_format_without_conversion():
    """Verify ASS/SSA subtitles are preserved in their native format without conversion to SRT."""
    from app.extractor import is_ass_subtitle

    ass_content = (
        "[Script Info]\n"
        "Title: Anime Arabic Subtitles\n"
        "ScriptType: v4.00+\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour\n"
        "Style: Default,Arial,20,&H00FFFFFF\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:01.00,0:00:04.00,Default,,0,0,0,,{\\pos(100,200)}ترجمة عربية بأنماط وألوان\n"
    )

    zip_bytes = create_mock_zip(
        {
            "[MST-luckysubs] [02].ass": ass_content,
        }
    )

    extracted = extract_srt_from_zip(zip_bytes)
    assert is_ass_subtitle(extracted) is True
    decoded = extracted.decode("utf-8")
    assert "[Script Info]" in decoded
    assert "[V4+ Styles]" in decoded
    assert "{\\pos(100,200)}" in decoded
    assert "Dialogue:" in decoded
    assert "ترجمة عربية بأنماط وألوان" in decoded

    # Transcode preserves ASS directly
    transcoded = transcode_to_utf8(ass_content.encode("utf-8"))
    assert is_ass_subtitle(transcoded) is True
    assert "[V4+ Styles]" in transcoded.decode("utf-8")


def test_arabic_cp1256_with_stray_byte_keeps_arabic_readable():
    """A legacy cp1256 file with an undecodable byte must not become latin mojibake."""
    arabic_text = "1\n00:00:01,000 --> 00:00:05,000\nأهلاً وسهلاً بكم\n"
    raw = bytearray(arabic_text.encode("cp1256"))
    raw[10] = 0x81  # byte outside the cp1256 repertoire
    out = decode_subtitle_bytes(bytes(raw), lang="ara")
    assert "أهلاً وسهلاً بكم" in out
    assert "Ã" not in out and "ã" not in out
    # Strict clean UTF-8 output with no trace of the bad byte.
    out.encode("utf-8")
    assert "\x81" not in out


def test_arabic_broken_utf8_is_repaired_not_mojibake():
    """Mostly-valid UTF-8 with one corrupt byte keeps its Arabic text (single �)."""
    arabic_text = "1\n00:00:01,000 --> 00:00:05,000\nمرحبا بك في الحلقة\n"
    raw = bytearray(arabic_text.encode("utf-8"))
    raw[30] = 0xFF
    out = decode_subtitle_bytes(bytes(raw), lang="ara")
    assert out.count("�") == 1
    assert "مرحبا بك في الحلقة"[:6] in out
    out.encode("utf-8")


def test_arabic_utf16_bom_is_decoded():
    """UTF-16LE payloads with a BOM decode instead of becoming latin garbage."""
    arabic_text = "1\n00:00:01,000 --> 00:00:05,000\nمرحبا\n"
    out = decode_subtitle_bytes(arabic_text.encode("utf-16"), lang="ara")
    assert "مرحبا" in out


def test_unknown_lang_keeps_legacy_single_byte_order():
    """Without an Arabic language hint the historic cp1256-first order is kept."""
    arabic_text = "1\n00:00:01,000 --> 00:00:05,000\nأهلاً\n"
    assert decode_subtitle_bytes(arabic_text.encode("cp1256")) == arabic_text
    assert transcode_to_utf8(arabic_text.encode("cp1256")).decode("utf-8") == arabic_text


def test_extract_cp1256_zip_member_with_arabic_lang():
    """ZIP members in legacy encodings decode to readable Arabic with lang hint."""
    arabic_text = "1\n00:00:01,000 --> 00:00:05,000\nترجمة عربية\n"
    zip_bytes = create_mock_zip({"movie.srt": arabic_text.encode("cp1256")})
    assert extract_srt_from_zip(zip_bytes, lang="ara").decode("utf-8") == arabic_text


def test_fix_encoding_with_arabic_lang_repairs_broken_utf8():
    """Serve-time encoding fix honours the Arabic decoding order."""
    from app.utils.cleaners import fix_subtitle_encoding_bytes

    arabic_text = "1\n00:00:01,000 --> 00:00:05,000\nمرحبا بك\n"
    raw = bytearray(arabic_text.encode("utf-8"))
    raw[30] = 0xFF
    out = fix_subtitle_encoding_bytes(bytes(raw), lang="ara").decode("utf-8")
    assert out.count("�") == 1
    assert "مرحبا" in out


def test_detect_archive_kind():
    from app.extractor import detect_archive_kind

    assert detect_archive_kind(b"PK\x03\x04rest") == "zip"
    assert detect_archive_kind(b"Rar!\x1a\x07\x00rest") == "rar"
    assert detect_archive_kind(b"Rar!\x1a\x07\x01\x00rest") == "rar"
    assert detect_archive_kind(b"7z\xbc\xaf\x27\x1crest") == "7z"
    assert detect_archive_kind(b"\x00" * 257 + b"ustar\x00rest") == "tar"
    assert detect_archive_kind(b"\x1f\x8brest") == "gzip"
    assert detect_archive_kind(b"1\n00:00:01,000 --> 00:00:02,000\nhi\n") is None
    assert detect_archive_kind(b"") is None


def test_looks_like_subtitle():
    from app.extractor import looks_like_subtitle

    assert looks_like_subtitle(b"1\n00:00:01,000 --> 00:00:02,000\nhi\n") is True
    assert looks_like_subtitle(b"WEBVTT\n\n00:00.000 --> 00:01.000\nhi") is True
    assert looks_like_subtitle(b"[Script Info]\nDialogue: 0,0:00:01.00,0:00:02.00,Default\n") is True
    assert looks_like_subtitle(b"<html><body>upstream error</body></html>") is False
    assert looks_like_subtitle(b"Rar!\x1a\x07\x00binarygarbage") is False
    assert looks_like_subtitle(b"Subtitle notice text only") is False
    assert looks_like_subtitle(b"") is False


def test_extract_subtitle_from_archive_rar(monkeypatch):
    from app.extractor import extract_subtitle_from_archive

    srt = "1\n00:00:01,000 --> 00:00:02,000\nhello rar\n"
    monkeypatch.setattr(
        "app.extractor._extract_nonzip_members",
        lambda raw, kind: [("fl-itw-xvid-cd2.srt", srt.encode("utf-8"))],
    )
    out = extract_subtitle_from_archive(b"Rar!\x1a\x07\x00fake")
    assert "hello rar" in out.decode("utf-8")


def test_extract_subtitle_from_archive_rar_without_subtitles(monkeypatch):
    from app.extractor import extract_subtitle_from_archive

    monkeypatch.setattr("app.extractor._extract_nonzip_members", lambda raw, kind: [])
    with pytest.raises(SubtitleExtractionError, match="No valid subtitle files"):
        extract_subtitle_from_archive(b"Rar!\x1a\x07\x00fake")


def test_extract_subtitle_from_archive_unknown_format_raises():
    from app.extractor import extract_subtitle_from_archive

    with pytest.raises(SubtitleExtractionError, match="Unrecognised archive format"):
        extract_subtitle_from_archive(b"not-an-archive")
