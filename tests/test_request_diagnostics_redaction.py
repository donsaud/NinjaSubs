"""The access log must never contain a credential-bearing configuration.

A configured NinjaSubs URL puts the user's provider credentials in its first
path segment, base64-encoded. uvicorn logs the raw request target, so without
interception every configured request writes an API key, username and password
into the container log in reversible base64 -- which is how credentials ended up
in a transcript earlier.

These tests pin the redaction and the sanitized diagnostics.
"""

from __future__ import annotations

import logging

from app.utils.config_parser import encode_user_config
from app.utils.request_diagnostics import (
    ConfigRedactingFilter,
    describe_configuration,
    looks_like_config_segment,
    redact_path,
)

SECRET_KEY = "SECRET-API-KEY-VALUE"
SECRET_USER = "SECRET-USERNAME"
SECRET_PASS = "SECRET-PASSWORD"


def _token(**overrides) -> str:
    base = {
        "subdl_key": "subdl-value",
        "subsource_key": "subsource-value",
        "opensubtitles_key": SECRET_KEY,
        "opensubtitles_username": SECRET_USER,
        "opensubtitles_password": SECRET_PASS,
    }
    base.update(overrides)
    return encode_user_config(
        base["subdl_key"],
        base["subsource_key"],
        opensubtitles_key=base["opensubtitles_key"],
        opensubtitles_username=base["opensubtitles_username"],
        opensubtitles_password=base["opensubtitles_password"],
        **overrides,
    )


# --- the token really does carry the secrets (so redaction is not vacuous) ---


def test_the_generated_token_really_contains_the_credentials():
    import base64
    import json

    token = _token()
    padded = token + "=" * ((4 - len(token) % 4) % 4)
    data = json.loads(base64.urlsafe_b64decode(padded).decode())
    assert data["opensubtitles_key"] == SECRET_KEY
    assert data["opensubtitles_username"] == SECRET_USER
    assert data["opensubtitles_password"] == SECRET_PASS


def test_a_configuration_segment_is_recognised():
    assert looks_like_config_segment(_token()) is True
    assert looks_like_config_segment("manifest.json") is False
    assert looks_like_config_segment("") is False
    assert looks_like_config_segment("a" * 60) is False  # long but not base64 JSON


# --- redaction ---


def test_a_configured_path_is_redacted():
    token = _token()
    redacted = redact_path(f"/{token}/subtitles/movie/tt1.json")
    assert redacted == "/{cfg}/subtitles/movie/tt1.json"
    assert token not in redacted


def test_an_unconfigured_path_is_left_alone():
    assert redact_path("/manifest.json") == "/manifest.json"
    assert redact_path("/subtitles/movie/tt2582802.json") == "/subtitles/movie/tt2582802.json"
    assert redact_path("/") == "/"
    assert redact_path("") == ""


def test_the_query_string_is_also_redacted():
    """Query-string configuration carries the same values as a path prefix."""
    token = _token()
    redacted = redact_path(f"/{token}/subtitles/movie/tt1.json?opensubtitles_key={SECRET_KEY}")
    assert SECRET_KEY not in redacted
    assert "opensubtitles_key=" not in redacted


def test_the_access_log_filter_redacts_the_recorded_target():
    token = _token()
    record = logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("172.20.0.1:1234", "GET", f"/{token}/manifest.json", "1.1", 200),
        exc_info=None,
    )
    assert ConfigRedactingFilter().filter(record) is True
    rendered = record.getMessage()
    assert token not in rendered
    assert SECRET_KEY not in rendered
    assert "/{cfg}/manifest.json" in rendered
    assert "172.20.0.1:1234" in rendered  # non-secret context is preserved


def test_the_filter_is_installed_on_the_access_logger():
    from app.utils.request_diagnostics import install_access_log_redaction

    install_access_log_redaction()
    access = logging.getLogger("uvicorn.access")
    assert any(isinstance(f, ConfigRedactingFilter) for f in access.filters)


# --- sanitized diagnostics ---


def test_configuration_summary_reports_presence_only():
    summary = describe_configuration(_token())
    assert summary["config_present"] is True
    assert summary["config_decoded"] is True
    assert summary["opensubtitles_key_present"] is True
    assert summary["opensubtitles_username_present"] is True
    assert summary["opensubtitles_password_present"] is True
    # No value, in any field, may equal or contain a secret.
    for value in summary.values():
        assert not (isinstance(value, str) and SECRET_KEY in value)
        assert not (isinstance(value, str) and SECRET_PASS in value)
    assert SECRET_KEY not in repr(summary)


def test_configuration_summary_for_an_absent_token():
    summary = describe_configuration("")
    assert summary == {"config_present": False, "config_decoded": False}


def test_configuration_summary_for_an_undecodable_token():
    summary = describe_configuration("z" * 80)
    assert summary["config_present"] is True
    assert summary["config_decoded"] is False


def test_diagnostics_lines_carry_no_secret(caplog):
    """The exact log line the middleware emits must not leak.

    Scoped to the application's own loggers. ``httpx`` logs the full URL of every
    request it makes, and Starlette's TestClient routes the incoming call through
    httpx, so a whole-log scan would flag httpx's own line. That is a test-harness
    path only: in production the app's httpx traffic is outbound to upstream
    providers and never contains the incoming configuration URL.
    """
    from fastapi.testclient import TestClient

    from app.main import app

    token = _token()
    with caplog.at_level(logging.DEBUG):
        with TestClient(app) as client:
            client.get(f"/{token}/manifest.json")

    app_records = [r for r in caplog.records if r.name.startswith("uvicorn")]
    blob = "\n".join(r.getMessage() for r in app_records)
    assert blob, "expected the diagnostics middleware to log something"
    assert SECRET_KEY not in blob
    assert SECRET_USER not in blob
    assert SECRET_PASS not in blob
    assert token not in blob
    # But it must still say enough to diagnose with.
    assert "config_present=1" in blob
    assert "config_decoded=True" in blob
    assert "os_key_present=True" in blob


def test_an_unconfigured_request_is_labelled_as_such(caplog):
    from fastapi.testclient import TestClient

    from app.main import app

    with caplog.at_level(logging.INFO):
        with TestClient(app) as client:
            client.get("/manifest.json")
    assert "route=unconfigured" in caplog.text
    assert "config_present=0" in caplog.text
