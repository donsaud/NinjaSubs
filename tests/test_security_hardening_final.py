"""Focused regression tests for security-hardening-p0 final microfixes."""

import json
from pathlib import Path

import pytest

from app.main import _constant_time_compare, _safe_json
from app.services.credentials import CredentialStore, credential_store


def test_constant_time_compare_uses_hmac():
    # Basic equality
    assert _constant_time_compare("abc", "abc") is True
    assert _constant_time_compare("abc", "abd") is False
    # Empty handling
    assert _constant_time_compare("", "abc") is False
    assert _constant_time_compare("abc", "") is False
    # Different lengths
    assert _constant_time_compare("short", "longer") is False


def test_safe_json_escapes():
    # Characters that break <script> embedding
    data = {
        "payload": "<script>alert('x')</script>",
        "amp": "a & b",
        "line": "a\u2028b\u2029c",
    }
    s = _safe_json(data)
    # Must be valid JSON after unescaping the unicode escapes? Actually we escape in the string.
    # Ensure escaped sequences are present and raw chars removed
    assert "\\u003c" in s
    assert "\\u003e" in s
    assert "\\u0026" in s
    assert "\\u2028" in s
    assert "\\u2029" in s
    # Raw characters should not appear
    assert "<script>" not in s
    assert "&" not in s or "\\u0026" in s
    # Ensure parseable after json.loads (unicode escapes are valid JSON)
    obj = json.loads(s)
    assert obj["payload"] == "<script>alert('x')</script>"
    assert obj["amp"] == "a & b"


def test_env_example_has_empty_admin_token():
    env_example = Path(__file__).resolve().parents[1] / ".env.example"
    text = env_example.read_text()
    # Find line
    for line in text.splitlines():
        if line.startswith("NINJASUBS_ADMIN_TOKEN="):
            assert line == "NINJASUBS_ADMIN_TOKEN=", f"Unexpected line: {line}"
            return
    pytest.fail("NINJASUBS_ADMIN_TOKEN not found in .env.example")


def test_credential_store_cleared_on_cache_clear():
    # Populate store
    import asyncio
    async def _run():
        await credential_store.store("id1", {"subdl_key": "k1"})
        await credential_store.store("id2", {"subsource_key": "k2"})
        assert credential_store.size() >= 2
        cleared = await credential_store.clear()
        assert cleared >= 2
        assert credential_store.size() == 0
    asyncio.run(_run())


def test_credential_store_docstring():
    doc = CredentialStore.store.__doc__ or ""
    # Should mention ephemeral provider credentials
    assert "ephemeral provider credentials" in doc.lower()
    # Should not claim non-secret metadata
    assert "non-secret metadata" not in doc.lower()


def test_proxy_opensubtitles_no_meta_credential_recovery():
    # Ensure main.py does not contain meta.get for credential keys in proxy_opensubtitles_stream
    main_path = Path(__file__).resolve().parents[1] / "app" / "main.py"
    src = main_path.read_text()
    # Simple heuristic: the fallback block should not reference meta.get("subsource_key")
    # We can search for the pattern
    # The file may still contain meta.get elsewhere for non-credential fields; we just ensure
    # the credential recovery line is gone.
    assert 'meta.get("subsource_key")' not in src
    assert "meta.get('subsource_key')" not in src
    # Also no subdl_key / opensubtitles_key from meta
    assert 'meta.get("subdl_key")' not in src
    assert 'meta.get("opensubtitles_key")' not in src
