"""Network utility functions for local LAN IP detection and base URL resolution."""

import os
import socket
import urllib.parse

from starlette.requests import Request

from app.config import settings


def get_local_lan_ip() -> str:
    """
    Resolve the host machine's primary local LAN IP address.
    Uses routing table lookup via UDP socket towards a dummy IP without sending packets.
    Supports environment variable overrides via HOST_IP, LAN_IP, or ADDON_BASE_URL.
    """
    # 1. Direct environment variable overrides
    env_ip = (os.getenv("HOST_IP") or os.getenv("LAN_IP") or "").strip()
    if env_ip:
        return env_ip

    addon_base = (os.getenv("ADDON_BASE_URL") or os.getenv("BASE_URL") or "").strip()
    if addon_base:
        try:
            parsed = urllib.parse.urlparse(addon_base)
            if parsed.hostname and parsed.hostname not in ("localhost", "127.0.0.1"):
                return parsed.hostname
        except Exception:
            pass

    # 2. Settings object overrides (if not localhost)
    settings_ip = (
        getattr(settings, "HOST_IP", "") or getattr(settings, "LAN_IP", "") or ""
    ).strip()
    if settings_ip and settings_ip not in ("localhost", "127.0.0.1"):
        return settings_ip

    settings_base = (
        getattr(settings, "ADDON_BASE_URL", "") or getattr(settings, "BASE_URL", "") or ""
    ).strip()
    if settings_base:
        try:
            parsed = urllib.parse.urlparse(settings_base)
            if parsed.hostname and parsed.hostname not in ("localhost", "127.0.0.1"):
                return parsed.hostname
        except Exception:
            pass

    # 2. UDP routing table lookup towards dummy external IP
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connects to an external dummy IP to determine the default interface's outgoing IP
        s.connect(("10.254.254.254", 1))
        ip = s.getsockname()[0]
    except Exception:
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        except Exception:
            ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def get_base_url(request: Request | None = None) -> str:
    """
    Determine externally reachable base URL.
    Prefers explicit settings.BASE_URL if configured.
    Otherwise dynamically utilizes the incoming request base_url so that
    whichever host/IP the client connects to (127.0.0.1, LAN IP, or reverse proxy),
    the returned links and logo match that exact reachable host.
    If request is None, falls back to LAN IP (or 127.0.0.1).
    """
    if settings.BASE_URL and settings.BASE_URL.strip():
        return settings.BASE_URL.rstrip("/")

    if request:
        return str(request.base_url).rstrip("/")

    lan_ip = get_local_lan_ip()
    port = settings.PORT or 7000
    if lan_ip and lan_ip not in ("127.0.0.1", "localhost"):
        return f"http://{lan_ip}:{port}"
    return f"http://127.0.0.1:{port}"
