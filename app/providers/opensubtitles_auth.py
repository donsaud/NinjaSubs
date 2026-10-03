"""OpenSubtitles.com user authentication: exchange credentials for a JWT.

Why this exists
---------------
OpenSubtitles.com v1 has two independent auth mechanisms:

* ``Api-Key`` -- identifies the *application*, required on every request.
* ``Authorization: <JWT>`` -- identifies the *user*, from ``POST /api/v1/login``.

Search does not need the second one. The API documentation is explicit: "There
is no limit for SearchSubtitles. Only one limit is applied, and it is about
Downloading subtitles." So ``GET /subtitles?moviehash=...`` already works with an
API key alone, and hash matching was never blocked on credentials.

What the JWT *does* buy is the download half, which is where this add-on was
quietly broken: the documentation lists exactly two endpoints that require user
authentication, and ``/download`` is one of them. This provider called
``POST /download`` with only an ``Api-Key``, so subtitle downloads ran
unauthenticated and were subject to the anonymous quota (5/day) rather than the
account's. Logging in fixes that, and a VIP account additionally returns a
different ``base_url`` that subsequent requests must use.

Why the cache is not optional
-----------------------------
``/login`` is rate-limited hard: 1 request/second, 10/minute, 30/hour, because
clients otherwise retry bad credentials in a loop. Logging in per subtitle
request would exhaust that in seconds and get the account throttled. Tokens last
24 hours, so this caches them in-process and refreshes well before expiry.

Single-flight matters for the same reason: a burst of concurrent subtitle
requests would otherwise each perform their own login, because at cold start no
token exists yet. One login serves all of them.

Credential hygiene
------------------
Credentials never appear in a log line, an exception message, or a cache key.
The cache is keyed by a SHA-256 digest, so the map can be inspected or dumped
without exposing anything, and a changed password simply produces a different
digest rather than mutating a shared entry.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger("uvicorn.error")

#: OpenSubtitles tokens are valid 24h. Refresh with margin so a token cannot
#: expire between the check and the request it was checked for.
TOKEN_TTL_SECONDS = 20 * 3600.0

#: Cap on cached entries. One add-on instance serves a handful of users at most,
#: but a long-lived process should not grow this map without bound if credential
#: sets churn (for example a user editing their password repeatedly).
MAX_CACHED_TOKENS = 32

#: Lock entries are pruned alongside tokens; a lock per credential digest would
#: otherwise outlive the token it was created for.
_MAX_TRACKED_LOCKS = MAX_CACHED_TOKENS * 2

#: How long a failed login suppresses further attempts for the same credentials.
#: OpenSubtitles allows only 30 logins per hour per consumer and specifically
#: throttles repeat attempts with wrong credentials, so an uncached failure would
#: let every concurrent subtitle request spend another slot against that budget
#: and lock the account out of authenticated downloads entirely.
FAILED_LOGIN_BACKOFF_SECONDS = 15 * 60.0


@dataclass(frozen=True)
class OpenSubtitlesToken:
    """A user JWT plus the host subsequent requests must be sent to."""

    token: str
    #: Host from the login response. OpenSubtitles returns e.g.
    #: ``api.opensubtitles.com`` or ``vip-api.opensubtitles.com``; requests must
    #: continue on whichever it returns.
    base_url: str
    expires_at: float

    @property
    def expired(self) -> bool:
        return time.monotonic() >= self.expires_at


def credential_digest(api_key: str, username: str, password: str) -> str:
    """Stable, non-reversible key for a credential set.

    SHA-256 over a salted concatenation. Used only to look entries up, never to
    authenticate anything, so a plain digest is the right tool -- and it means
    the cache cannot leak a password even if it is serialised for debugging.
    """
    material = f"{api_key}\x00{username}\x00{password}".encode()
    return hashlib.sha256(material).hexdigest()


class OpenSubtitlesTokenCache:
    """Process-wide, single-flight cache of user JWTs."""

    def __init__(self) -> None:
        self._tokens: dict[str, OpenSubtitlesToken] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        # digest -> monotonic time until which login is not retried.
        self._failed_until: dict[str, float] = {}

    def _lock_for(self, digest: str) -> asyncio.Lock:
        lock = self._locks.get(digest)
        if lock is None:
            if len(self._locks) >= _MAX_TRACKED_LOCKS:
                # Drop the oldest tracked lock. The worst case is that a
                # concurrent login for the evicted digest runs again, which the
                # rate limit tolerates; leaking locks does not.
                self._locks.pop(next(iter(self._locks)), None)
            lock = asyncio.Lock()
            self._locks[digest] = lock
        return lock

    def peek(self, api_key: str, username: str, password: str) -> OpenSubtitlesToken | None:
        """Return a live token without performing any I/O. For tests and diagnostics."""
        entry = self._tokens.get(credential_digest(api_key, username, password))
        if entry is None or entry.expired:
            return None
        return entry

    def invalidate(self, api_key: str, username: str, password: str) -> None:
        """Drop the cached token, forcing the next call to log in again.

        The negative-cache entry is cleared too: this is the "the token was
        rejected, re-authenticate now" path and must not be suppressed by a
        backoff left over from an earlier failure.
        """
        digest = credential_digest(api_key, username, password)
        self._tokens.pop(digest, None)
        self._failed_until.pop(digest, None)

    async def get_token(
        self,
        client: httpx.AsyncClient,
        *,
        api_key: str,
        username: str,
        password: str,
        base_url: str,
        user_agent: str,
    ) -> OpenSubtitlesToken | None:
        """Return a valid JWT for these credentials, logging in only if needed.

        ``None`` means authentication is unavailable or failed -- either no
        credentials were supplied, or OpenSubtitles rejected them. Callers treat
        that as "carry on unauthenticated", because search still works without a
        token and an account problem must not break subtitle lookup entirely.
        """
        username = (username or "").strip()
        password = password or ""
        if not username or not password or not api_key:
            return None

        digest = credential_digest(api_key, username, password)
        cached = self._tokens.get(digest)
        if cached is not None and not cached.expired:
            return cached

        # A recent failure is remembered so a bad password costs one request
        # rather than one per subtitle request.
        if time.monotonic() < self._failed_until.get(digest, 0.0):
            logger.warning(
                "[OpenSubtitles] Skipping login for user %s; a previous attempt "
                "failed recently",
                _mask(username),
            )
            return None

        # Single-flight: concurrent callers for the same credentials wait for
        # one login rather than each spending a slot in a 30/hour budget.
        async with self._lock_for(digest):
            cached = self._tokens.get(digest)
            if cached is not None and not cached.expired:
                return cached
            if time.monotonic() < self._failed_until.get(digest, 0.0):
                return None
            fresh = await self._login(
                client,
                api_key=api_key,
                username=username,
                password=password,
                base_url=base_url,
                user_agent=user_agent,
            )
            if fresh is None:
                if len(self._failed_until) >= MAX_CACHED_TOKENS:
                    self._failed_until.pop(next(iter(self._failed_until)), None)
                self._failed_until[digest] = (
                    time.monotonic() + FAILED_LOGIN_BACKOFF_SECONDS
                )
                return None
            self._failed_until.pop(digest, None)
            if len(self._tokens) >= MAX_CACHED_TOKENS:
                self._tokens.pop(next(iter(self._tokens)), None)
            self._tokens[digest] = fresh
            return fresh

    async def _login(
        self,
        client: httpx.AsyncClient,
        *,
        api_key: str,
        username: str,
        password: str,
        base_url: str,
        user_agent: str,
    ) -> OpenSubtitlesToken | None:
        url = f"{base_url.rstrip('/')}/login"
        try:
            resp = await client.post(
                url,
                headers={
                    "Api-Key": api_key,
                    "User-Agent": user_agent,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json={"username": username, "password": password},
            )
        except Exception as exc:
            # Never interpolate the exception into a log line without checking:
            # httpx errors can embed the request URL, and this one has no secret
            # in it, but the rule is applied unconditionally so it stays true if
            # the URL ever changes shape.
            logger.warning(
                "[OpenSubtitles] Login request failed for user %s: %s",
                _mask(username),
                type(exc).__name__,
            )
            return None

        if resp.status_code != 200:
            # 401 must stop further attempts with these credentials rather than
            # retrying: OpenSubtitles counts wrong-credential logins against the
            # 30/hour limit.
            logger.warning(
                "[OpenSubtitles] Login rejected for user %s (HTTP %s); "
                "continuing unauthenticated",
                _mask(username),
                resp.status_code,
            )
            return None

        try:
            payload = resp.json()
        except Exception:
            logger.warning("[OpenSubtitles] Login returned a non-JSON body")
            return None

        token = str(payload.get("token") or "").strip()
        if not token:
            logger.warning("[OpenSubtitles] Login response contained no token")
            return None

        resolved_base = _normalise_base_url(str(payload.get("base_url") or ""), base_url)
        if resolved_base == base_url and str(payload.get("base_url") or "").strip():
            logger.warning(
                "[OpenSubtitles] Login returned an unusable base_url; keeping the default"
            )

        logger.info(
            "[OpenSubtitles] Authenticated user %s; token cached for %.0fh (host=%s)",
            _mask(username),
            TOKEN_TTL_SECONDS / 3600.0,
            resolved_base,
        )
        return OpenSubtitlesToken(
            token=token,
            base_url=resolved_base,
            expires_at=time.monotonic() + TOKEN_TTL_SECONDS,
        )


def _normalise_base_url(raw: str, default: str) -> str:
    """Coerce the login response's ``base_url`` into a usable API root.

    OpenSubtitles returns a bare host -- ``api.opensubtitles.com`` or
    ``vip-api.opensubtitles.com`` -- not a URL. Callers build requests as
    ``f"{base_url}/subtitles"``, so taking the value verbatim would produce
    ``api.opensubtitles.com/subtitles``: no scheme, and the ``/api/v1`` prefix
    missing. Both are silently wrong rather than loudly broken, which is why
    this is normalised rather than trusted.

    Anything unrecognised falls back to ``default``; a malformed host must never
    become a request target.
    """
    value = (raw or "").strip().rstrip("/")
    if not value:
        return default

    if value.lower().startswith(("http://", "https://")):
        head, _, tail = value.partition("://")
        host, _, path = tail.partition("/")
        if not host or any(ch.isspace() for ch in value):
            return default
        # Already carries a path: leave it alone.
        if path:
            return value
        return f"{value}/api/v1"

    # Bare host, e.g. "vip-api.opensubtitles.com".
    if any(ch.isspace() for ch in value) or "/" in value or "." not in value:
        return default
    return f"https://{value}/api/v1"


def _mask(username: str) -> str:
    """Enough of a username to correlate logs, not enough to identify an account.

    Never log a password, in any form, including length or first characters.
    """
    value = (username or "").strip()
    if not value:
        return "<empty>"
    if len(value) <= 2:
        return "*" * len(value)
    return f"{value[0]}{'*' * (len(value) - 2)}{value[-1]}"


#: Shared by the provider and the request handlers.
OPENSUBTITLES_TOKENS = OpenSubtitlesTokenCache()
