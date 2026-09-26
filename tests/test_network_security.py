"""Tests for the SSRF guard that protects server-side stream probing."""

import pytest

from app.utils.network_security import is_safe_public_url


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/video.mkv",
        "http://127.0.0.1:8080/video.mkv",
        "http://10.0.0.1/video.mkv",
        "http://172.16.5.5/video.mkv",
        "http://192.168.1.50/video.mkv",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/video.mkv",
        "file:///etc/passwd",
        "gopher://127.0.0.1:70/x",
        "ftp://example.com/video.mkv",
        "",
        None,
        "not a url",
    ],
)
def test_is_safe_public_url_rejects_unsafe(url):
    ok, reason = is_safe_public_url(url)
    assert ok is False
    assert reason


@pytest.mark.parametrize(
    "url",
    [
        "http://8.8.8.8/video.mkv",
        "https://8.8.8.8/video.mkv",
        "https://1.1.1.1/stream",
    ],
)
def test_is_safe_public_url_allows_public_literals(url):
    ok, reason = is_safe_public_url(url)
    assert ok is True, reason
    assert reason == "ok"


@pytest.mark.parametrize(
    "url",
    [
        "http://192.168.8.115:4000/movie.mkv",
        "http://10.0.0.5/movie.mkv",
        "http://172.16.5.5/movie.mkv",
        "http://127.0.0.1/movie.mkv",
    ],
)
def test_allow_private_permits_rfc1918_and_loopback(url):
    # Strict default still blocks them...
    assert is_safe_public_url(url)[0] is False
    # ...but the self-hosted flag permits RFC1918 + loopback hosts.
    ok, reason = is_safe_public_url(url, allow_private=True)
    assert ok is True, reason


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",
        "http://169.254.1.1/movie.mkv",
        "file:///etc/passwd",
    ],
)
def test_allow_private_still_blocks_metadata_and_bad_schemes(url):
    assert is_safe_public_url(url, allow_private=True)[0] is False
