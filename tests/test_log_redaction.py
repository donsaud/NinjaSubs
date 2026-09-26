"""Tests for API-key/authorization redaction in log output."""

import logging

from app.utils.log_redaction import install_log_redaction, redact_secrets


def test_redact_query_api_key_parameters():
    assert redact_secrets("GET https://dl.subdl.com/x.zip?api_key=SECRET123&x=1") == (
        "GET https://dl.subdl.com/x.zip?api_key=[REDACTED]&x=1"
    )
    assert "SECRET123" not in redact_secrets("apiKey=SECRET123")
    assert "SECRET123" not in redact_secrets("subsource_api_key=SECRET123")
    assert "SECRET123" not in redact_secrets("opensubtitles_api_key=SECRET123")


def test_redact_authorization_and_x_api_key_headers():
    out = redact_secrets("Headers: {'Authorization': 'Bearer SECRET123', 'X-API-Key': 'SECRET123'}")
    assert "SECRET123" not in out
    assert "Authorization" in out and "X-API-Key" in out
    assert "[REDACTED]" in out


def test_redact_bearer_token_standalone():
    out = redact_secrets("using Bearer abcDEF123.xyz-456 now")
    assert "abcDEF123" not in out
    assert "Bearer [REDACTED]" in out


def test_install_log_redaction_scrubs_emitted_records(caplog):
    install_log_redaction()
    logger = logging.getLogger("test.redaction")
    with caplog.at_level("INFO"):
        logger.info("GET https://api.subdl.com/v1/subtitles?api_key=TOPSECRET123&type=tv")
    assert "TOPSECRET123" not in caplog.text
    assert "api_key=[REDACTED]" in caplog.text
