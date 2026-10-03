"""Log redaction must cover account passwords, not just API keys.

uvicorn's access log echoes the raw request target, so any credential a client
puts in a query string reaches the container log. The configure page posts the
password in a body to avoid that; this filter is the backstop for the URLs the
page does not control -- a hand-crafted one, a bookmark, a third-party client.
"""

from __future__ import annotations

import logging

# Importing the settings module is what installs the global LogRecord factory.
# Done explicitly so these tests do not depend on some other test having imported
# it first.
import app.config  # noqa: F401
from app.utils.log_redaction import REDACTED, redact_secrets


def test_an_api_key_is_redacted():
    assert "KEY123" not in redact_secrets("?api_key=KEY123")


def test_a_password_in_a_query_string_is_redacted():
    out = redact_secrets("GET /api/verify/opensubtitles?username=me&password=hunter2 HTTP/1.1")

    assert "hunter2" not in out
    assert f"password={REDACTED}" in out


def test_a_full_access_log_line_is_scrubbed():
    line = (
        '172.20.0.1:52856 - "GET /api/verify/opensubtitles'
        '?api_key=abc&username=me&password=hunter2 HTTP/1.1" 200 OK'
    )
    out = redact_secrets(line)

    assert "hunter2" not in out
    assert "abc&" not in out
    assert "username=me" in out  # the username is not a secret and stays legible


def test_password_aliases_are_covered():
    for param in ("password", "passwd", "pwd", "opensubtitles_password"):
        out = redact_secrets(f"?{param}=secretvalue")
        assert "secretvalue" not in out, param


def test_prose_about_passwords_is_not_mangled():
    """The pattern is anchored to ``name=``, so normal text survives.

    Worth pinning: a filter that matched the bare word would turn diagnostics
    into noise and train people to ignore the redaction entirely.
    """
    out = redact_secrets("password=abc the user has no password set")

    assert f"password={REDACTED}" in out
    assert "the user has no password set" in out
    assert redact_secrets("the user has no password") == "the user has no password"


def test_the_factory_scrubs_a_password_emitted_by_any_logger(caplog):
    with caplog.at_level(logging.INFO):
        logging.getLogger("uvicorn.access").info(
            'GET /sub/x.srt?password=hunter2 HTTP/1.1'
        )

    assert "hunter2" not in caplog.text


def test_redaction_is_idempotent():
    once = redact_secrets("?password=hunter2")
    assert redact_secrets(once) == once


def test_empty_input_is_safe():
    assert redact_secrets("") == ""
    assert redact_secrets(None) is None  # type: ignore[arg-type]
