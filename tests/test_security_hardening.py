"""Regression tests for request-path security and robustness hardening.

Each test pins one concrete defect: a reflected XSS on the configure page, a
500 on malformed tokens, unbounded archive inflation, archive symlinks, HTML
cached as a subtitle, credentials in the access log, torn metadata writes, and
a state-changing GET.
"""

import base64
import io
import json
import os
import shutil
import tarfile
import time
import zipfile

import pytest

from app.extractor import (
    SubtitleExtractionError,
    decode_subtitle_bytes,
    extract_srt_from_zip,
    extract_subtitle_from_archive,
    looks_like_subtitle,
)
from app.utils.config_parser import parse_user_config
from app.utils.request_diagnostics import redact_path

SRT = b"1\n00:00:01,000 --> 00:00:02,000\nHello\n\n"


def _token(payload: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")


# --------------------------------------------------------------------------- #
# Configure page XSS
# --------------------------------------------------------------------------- #


def test_configure_page_escapes_prefilled_credentials(client):
    payload = '"><img src=x onerror=alert(document.domain)>'
    token = _token({"subdl_key": payload, "languages": ["ara"]})

    resp = client.get(f"/{token}/configure")

    assert resp.status_code == 200
    assert payload not in resp.text
    assert "&quot;&gt;&lt;img src=x onerror=alert(document.domain)&gt;" in resp.text


# --------------------------------------------------------------------------- #
# Malformed token fields
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", [123, ["k"], {"k": "v"}, True])
def test_non_string_credential_is_ignored_not_crashing(bad):
    prefs = parse_user_config(_token({"subdl_key": bad, "subsource_key": bad}))
    assert isinstance(prefs.subdl_key, str)
    assert isinstance(prefs.subsource_key, str)


def test_configure_page_survives_non_string_credential(client):
    resp = client.get(f"/{_token({'opensubtitles_key': 42})}/configure")
    assert resp.status_code == 200


@pytest.mark.parametrize("value,expected", [("false", False), ("0", False), ("true", True), (True, True)])
def test_exclude_hi_string_values_parse_as_booleans(value, expected):
    assert parse_user_config(_token({"exclude_hi": value})).exclude_hi is expected


# --------------------------------------------------------------------------- #
# Archive size limits (zip bomb)
# --------------------------------------------------------------------------- #


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_oversized_zip_member_is_never_inflated(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "MAX_SUBTITLE_BYTES", 1024, raising=False)
    bomb = _zip({"bomb.srt": SRT + b"0" * 50_000})
    assert len(bomb) < 2048  # compresses to almost nothing

    with pytest.raises(SubtitleExtractionError):
        extract_srt_from_zip(bomb)


def test_zip_member_with_lying_size_header_is_rejected(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "MAX_SUBTITLE_BYTES", 1024, raising=False)
    archive = bytearray(_zip({"a.srt": SRT + b"0" * 50_000}))
    # Rewrite the declared uncompressed size (central directory) to 10 bytes.
    cd = archive.rfind(b"PK\x01\x02")
    archive[cd + 24 : cd + 28] = (10).to_bytes(4, "little")

    with pytest.raises(SubtitleExtractionError):
        extract_srt_from_zip(bytes(archive))


def test_oversized_archive_download_is_refused(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "MAX_ARCHIVE_BYTES", 100, raising=False)
    with pytest.raises(SubtitleExtractionError):
        extract_subtitle_from_archive(_zip({"a.srt": SRT * 20}))


def test_corrupt_zip_member_raises_extraction_error_not_zlib_error():
    archive = bytearray(_zip({"a.srt": SRT * 50}))
    # Corrupt the deflate stream just past the local header.
    start = archive.find(b"a.srt") + len("a.srt")
    for i in range(start, start + 20):
        archive[i] ^= 0xFF

    with pytest.raises(SubtitleExtractionError):
        extract_srt_from_zip(bytes(archive))


def test_corrupt_reference_zip_is_one_unusable_candidate():
    from app.services.sync.decode import decode_payload

    archive = bytearray(_zip({"a.srt": SRT * 500}))
    start = archive.find(b"a.srt") + len("a.srt")
    for i in range(start, start + 20):
        archive[i] ^= 0xFF

    assert decode_payload(bytes(archive), "Some.Release", min_bytes=10) is None


# --------------------------------------------------------------------------- #
# Symlinks inside non-ZIP archives
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(shutil.which("bsdtar") is None, reason="bsdtar not installed")
def test_tar_symlink_member_is_not_followed(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_bytes(SRT + b"HOST-SECRET\n")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        link = tarfile.TarInfo("evil.srt")
        link.type = tarfile.SYMTYPE
        link.linkname = str(secret)
        tf.addfile(link)
        real = tarfile.TarInfo("real.srt")
        real.size = len(SRT)
        tf.addfile(real, io.BytesIO(SRT))

    out = extract_subtitle_from_archive(buf.getvalue())

    assert b"HOST-SECRET" not in out
    assert out == SRT


# --------------------------------------------------------------------------- #
# Subtitle validity check
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "payload,ok",
    [
        (SRT, True),
        (b"[Script Info]\nDialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,Hi\n", True),
        (b"{1}{50}Hello|World\n{60}{90}Bye\n", True),
        ("1\r\n00:00:01,000 --> 00:00:02,000\r\nhi\r\n".encode("utf-16"), True),
        (b"<SAMI><BODY><SYNC Start=1000><P Class=ENCC>Hi</BODY></SAMI>", True),
        # An HTML comment contains "-->" but is not a subtitle.
        (b"<html><!-- cf-ray --><body>Access denied</body></html>", False),
        (b'{"message":"You have downloaded your allowed 20 subtitles"}', False),
    ],
)
def test_looks_like_subtitle(payload, ok):
    assert looks_like_subtitle(payload) is ok


def test_opensubtitles_guard_accepts_ass_downloads():
    from app.providers.opensubtitles import looks_like_subtitle_payload

    ass = b"[Script Info]\nScriptType: v4.00+\n\n[Events]\nDialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,Hi\n"
    assert looks_like_subtitle_payload(ass) is True


# --------------------------------------------------------------------------- #
# Legacy decoding by language
# --------------------------------------------------------------------------- #


def test_latin_script_legacy_subtitle_is_not_decoded_as_arabic():
    text = "Mañana ¿Qué? Straße schön árbol"
    assert decode_subtitle_bytes(text.encode("cp1252"), lang="spa") == text


@pytest.mark.parametrize("lang", ["ara", "ar", None])
def test_arabic_legacy_subtitle_still_decodes_as_cp1256(lang):
    text = "مرحبا بالعالم"
    assert decode_subtitle_bytes(text.encode("cp1256"), lang=lang) == text


# --------------------------------------------------------------------------- #
# Access-log redaction
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("suffix", ["/", "/favicon.ico", "/manifest.jsn", ""])
def test_token_is_redacted_without_a_route_marker(suffix):
    token = _token({"subdl_key": "SUPERSECRETKEY", "languages": ["ara", "eng"]})

    redacted = redact_path(f"/{token}{suffix}")

    assert token not in redacted
    assert token[:20] not in redacted
    assert redacted.startswith("/{cfg}")


def test_plain_paths_are_not_redacted():
    assert redact_path("/favicon.ico") == "/favicon.ico"


# --------------------------------------------------------------------------- #
# Metadata cache
# --------------------------------------------------------------------------- #


def test_metadata_write_is_atomic(test_cache_manager, monkeypatch):
    test_cache_manager.store_metadata("abc", {"release_name": "old"})

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    test_cache_manager.store_metadata("abc", {"release_name": "new"})

    # The failed write left the previous file intact and no temp debris behind.
    assert test_cache_manager.get_metadata("abc")["release_name"] == "old"
    assert [p.name for p in test_cache_manager.meta_dir.iterdir()] == ["abc.json"]


def test_stale_orphan_metadata_is_pruned(test_cache_manager):
    test_cache_manager.store_metadata("orphan", {"release_name": "x"})
    test_cache_manager.store_metadata("kept", {"release_name": "y"})
    test_cache_manager.get_subtitle_path("kept").write_bytes(SRT)
    old = time.time() - 30 * 24 * 3600
    for sid in ("orphan", "kept"):
        os.utime(test_cache_manager.get_meta_path(sid), (old, old))

    test_cache_manager._last_meta_prune = 0.0
    test_cache_manager.store_metadata("fresh", {"release_name": "z"})

    assert test_cache_manager.get_metadata("orphan") is None
    assert test_cache_manager.get_metadata("kept") is not None
    assert test_cache_manager.get_metadata("fresh") is not None


# --------------------------------------------------------------------------- #
# State-changing GET
# --------------------------------------------------------------------------- #


def test_cache_clear_rejects_get(client):
    assert client.get("/cache/clear").status_code in (404, 405)
    assert client.post("/cache/clear").status_code == 200

