"""Carry authorization with its immutable bytes through final HTTP rendering."""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass

from app.utils import cleaners, rtl
from two_point_sync.parser import parse_srt

logger = logging.getLogger(__name__)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class DeliveryAuthorization:
    request_id: str
    approved_sha256: str | None
    reference_sha256: str | None
    binding: str
    authoritative_verdict: str
    may_serve: bool
    original: bytes | None


class DeliveryPayload(bytes):
    authorization: DeliveryAuthorization

    def __new__(cls, payload: bytes, authorization: DeliveryAuthorization):
        result = super().__new__(cls, payload)
        result.authorization = authorization
        return result


@dataclass(frozen=True)
class RenderPreferences:
    options: cleaners.CleanOptions
    rtl_fix: bool
    ad_removal: bool
    keep_credits: bool
    strip_hi: bool
    eastern_numerals: bool
    strip_diacritics: bool


def permitted_render(payload: bytes, prefs: RenderPreferences) -> bytes:
    """Apply only the existing, explicitly selected formatting operations.

    Independent of main's rendering pipeline so unexpected changes in that
    pipeline cannot redefine the allowed dialogue/timing contract.
    """
    if prefs.options.fix_encoding:
        payload = cleaners.fix_subtitle_encoding_bytes(payload)
    if prefs.strip_hi:
        payload = cleaners.strip_hi_artifacts_bytes(payload)
    if prefs.ad_removal:
        payload = cleaners.strip_advertisements_bytes(payload, prefs.keep_credits)
    if prefs.strip_diacritics:
        payload = cleaners.strip_arabic_diacritics_bytes(payload)
    payload = cleaners.clean_subtitle_bytes(payload, prefs.options)
    if prefs.rtl_fix:
        payload = rtl.fix_rtl_punctuation_bytes(payload)
    if prefs.eastern_numerals:
        payload = cleaners.convert_eastern_arabic_numerals_bytes(payload)
    return payload


def _cues(payload: bytes, response_format: str):
    text = payload.decode("utf-8-sig")
    if response_format == "vtt":
        if not text.startswith("WEBVTT\n") and not text.startswith("WEBVTT\r\n"):
            raise ValueError("missing WebVTT header")
        text = text.split("\n", 1)[1].lstrip()
        text = "\n".join(
            re.sub(r"(\d{2}:\d{2}:\d{2})\.(\d{3})(?=\s|$)", r"\1,\2", line)
            if " --> " in line else line for line in text.splitlines())
    elif response_format != "srt":
        raise ValueError("unsupported authorized response format")
    cues = parse_srt(text)
    if not cues:
        raise ValueError("empty response")
    return [(cue.start_ms, cue.end_ms, cue.text) for cue in cues]


def final_body_preserves_authorization(
    approved: bytes, body: bytes, response_format: str, prefs: RenderPreferences,
) -> bool:
    try:
        original = _cues(approved, "srt")
        expected = _cues(permitted_render(approved, prefs), "srt")
        # Preference-authorized removals may reindex cues, but no retained cue
        # may be reordered or retimed, including by a timing-cleaning utility.
        remaining = iter(original)
        for start, end, _ in expected:
            if not any((old_start, old_end) == (start, end)
                       for old_start, old_end, _ in remaining):
                return False
        return _cues(body, response_format) == expected
    except Exception:
        return False


def original_response_body(original: bytes, response_format: str, prefs: RenderPreferences) -> bytes:
    """Safe original fallback; use the permitted utilities, not a failed renderer."""
    payload = permitted_render(original, prefs)
    if response_format == "vtt":
        text = payload.decode("utf-8-sig")
        lines = text.replace("\r\n", "\n").splitlines()
        lines = [re.sub(r"(\d{2}:\d{2}:\d{2}),(\d{3})", r"\1.\2", line)
                 if " --> " in line else line for line in lines]
        return ("WEBVTT\n\n" + "\n".join(lines)).encode("utf-8")
    return payload


def log_final_delivery(
    auth: DeliveryAuthorization, body: bytes, response_format: str, authorized: bool,
    integrity: bool, decision: str,
) -> None:
    logger.info(
        "[unified-delivery] req=%s engine=8 method=unified-v1 authoritative_verdict=%s "
        "reference_sha256=%s binding=%s may_serve=%s approved_candidate_sha256=%s "
        "http_body_sha256=%s response_format=%s decision=%s final_integrity=%s",
        auth.request_id, auth.authoritative_verdict, auth.reference_sha256 or "none",
        auth.binding, authorized, auth.approved_sha256 or "none", sha256(body),
        response_format, decision, integrity,
    )
