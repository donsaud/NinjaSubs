"""SSRF guard for server-side media probing.

The embedded-subtitle strategy runs ``ffprobe``/``ffmpeg`` against a
caller-supplied ``stream_url``. Without validation that is an SSRF primitive:
an attacker can point the addon at loopback, RFC1918, link-local, or cloud
metadata endpoints (and ``file://`` lets ffmpeg read local files). This module
fails closed — anything that is not a plain http(s) URL resolving exclusively
to public addresses is rejected.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_ALLOWED_SCHEMES = frozenset({"http", "https"})

# Cloud instance-metadata endpoints are link-local, but pin the well-known AWS
# address explicitly so the guard stays correct if address semantics change.
_BLOCKED_ADDRESSES = frozenset({ipaddress.ip_address("169.254.169.254")})


def _is_blocked_address(address: str, *, allow_private: bool = False) -> bool:
    """True when an IP is a disallowed destination for server-side probing.

    Cloud-metadata and link-local addresses are *always* blocked. When
    ``allow_private`` is set (self-hosted / LAN setups, e.g. an AIOStreams
    proxy on ``192.168.x.x``), RFC1918/ULA private and loopback addresses are
    permitted; multicast/reserved/unspecified remain blocked.
    """
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return True
    if ip in _BLOCKED_ADDRESSES:
        return True
    # Link-local (incl. cloud metadata) is never a legitimate stream host.
    if ip.is_link_local:
        return True
    if allow_private and (ip.is_private or ip.is_loopback):
        return False
    # ``is_global`` is False for private/loopback/link-local/multicast/reserved/
    # unspecified ranges, which covers every disallowed destination.
    return not ip.is_global


def is_safe_public_url(url: str | None, *, allow_private: bool = False) -> tuple[bool, str]:
    """Validate a caller-supplied stream URL for server-side ffprobe/ffmpeg.

    Returns ``(ok, reason)``. Rejects non-http(s) schemes (e.g. ``file://``,
    ``gopher://``), unresolvable hosts, and any host that resolves to a
    non-public address. ``allow_private=True`` additionally permits RFC1918/
    loopback destinations for self-hosted LAN media servers. Cloud-metadata and
    link-local addresses stay blocked either way.
    """
    candidate = (url or "").strip()
    if not candidate:
        return False, "empty url"

    try:
        parsed = urlparse(candidate)
    except ValueError:
        return False, "unparseable url"

    scheme = (parsed.scheme or "").lower()
    if scheme not in _ALLOWED_SCHEMES:
        return False, f"scheme {scheme or '<none>'!r} not allowed"

    host = parsed.hostname
    if not host:
        return False, "missing host"

    try:
        infos = socket.getaddrinfo(host, None)
    except OSError as exc:
        return False, f"dns resolution failed: {exc}"

    addresses = {str(info[4][0]) for info in infos if info[4]}
    if not addresses:
        return False, "no resolved addresses"

    for address in addresses:
        if _is_blocked_address(address, allow_private=allow_private):
            return False, f"blocked address {address}"
    return True, "ok"
