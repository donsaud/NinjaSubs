"""STREMIO_PLAYBACK_WITNESS -- development/test-only isolated delivery path.

Purpose
-------
Answer exactly one question:

    "Can Stremio display an already-known-good Alass output correctly when that
     exact output is delivered directly to it?"

The verifier currently *rejects* the Alass artifact for the real Dexter S08E04
EVOLV and ASAP subtitles and serves the provider's original instead. That
behaviour is deliberate and unchanged by this module. The open question is
whether the rejected artifact would in fact have played correctly -- which is a
question about Stremio's rendering, not about Alass, the format, or our policy.

To answer it, the known-good artifact is delivered to Stremio directly, with no
production pipeline in the path at all.

Isolation guarantees
--------------------
This module deliberately does not import, call, or touch:

    * SubDL / SubSource / OpenSubtitles / Yifysubtitles / Subtitlecat
    * the subtitle aggregator, ranking, or scoring
    * ``SyncOrchestrator``, ``SubtitleSyncService``, or alass
    * ``SyncVerifier`` / ``AlignmentAnalyzer`` (any verifier state)
    * the provider disk cache, ``SyncCache``, or the negative cache

It reads a file from disk and returns its bytes. That is the whole
implementation. There is no path by which a witness request can consult, mutate,
or be decided by any production component.

Byte fidelity
-------------
The response body is exactly the file's bytes: no text transformation, no
newline rewriting, no BOM handling, no encoding change, no cue re-indexing.
The serving URL embeds the fixture's SHA-256, so the URL changes when the
fixture changes and a Stremio/proxy HTTP cache cannot silently serve a stale
copy. The digest is re-verified on every request; a URL that advertises one
digest and a file that hashes to another is refused rather than served.

Safety
------
* Disabled by default. ``ENABLE_STREMIO_PLAYBACK_WITNESS`` must be explicitly
  set to true, and the router is not even mounted unless it is.
* Subtitle *contents* are never logged. Only name, size, and digest.
* No credential is read, echoed, or logged.
* The fixture lives on the host filesystem and is never committed. This module
  has no build-time or runtime dependency on the real Dexter video or subtitles.

Removal is a one-line revert: delete this file and the two-line mount in
``app/main.py``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pathlib
from dataclasses import dataclass

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from app.models import SubtitleItem, SubtitlesResponse
from app.utils.network import get_base_url

logger = logging.getLogger(__name__)

#: Marks every artifact this module produces, so a witness response can never
#: be mistaken for a production subtitle if one ever leaks into a real session.
WITNESS_MARKER = "STREMIO_PLAYBACK_WITNESS"

#: Mount point. Deliberately far from any production subtitle route so the two
#: can never collide in a URL, a log, or a browser history entry.
MOUNT = "/stremio-playback-witness"

#: The one media item this witness serves. Anything else returns an empty list
#: without touching a provider, so the path cannot be used to search.
TARGET_IMDB = "tt0773262"
TARGET_SEASON = 8
TARGET_EPISODE = 4


def witness_enabled() -> bool:
    """True only when the witness is explicitly switched on.

    Off by default. Requires an explicit true value in the environment or in
    settings; anything absent, false, or unparseable means off.
    """
    raw = os.getenv("ENABLE_STREMIO_PLAYBACK_WITNESS", "").strip().lower()
    if raw in ("true", "1", "yes", "on"):
        return True
    if raw not in ("", "false", "0", "no", "off"):
        # An unrecognised value must not silently enable a test-only path.
        logger.warning("[witness] ignoring unrecognised ENABLE_STREMIO_PLAYBACK_WITNESS value")
        return False
    try:
        from app.config import settings

        return bool(getattr(settings, "ENABLE_STREMIO_PLAYBACK_WITNESS", False))
    except Exception:  # noqa: BLE001
        return False


def _fixture_dir() -> pathlib.Path:
    """Directory holding the pre-validated Alass outputs.

    Resolved from ``NINJASUBS_WITNESS_FIXTURE_DIR``, falling back to the
    ``autosync-diagnostics`` folder under the real-media root so it resolves the
    same way the rest of the project's test tooling does. Nothing is bundled: if
    the directory is absent the witness simply reports that it has no fixtures.
    """
    explicit = os.getenv("NINJASUBS_WITNESS_FIXTURE_DIR", "").strip()
    if explicit:
        return pathlib.Path(explicit)
    media_root = os.getenv("NINJASUBS_REAL_MEDIA_ROOT", "").strip()
    if media_root:
        return pathlib.Path(media_root) / "autosync-diagnostics"
    return pathlib.Path("autosync-diagnostics")


@dataclass(frozen=True)
class WitnessFixture:
    """One pre-validated Alass output, exposed as a Stremio subtitle."""

    key: str
    #: Stremio-visible label. Must be unmistakable in a subtitle picker.
    label: str
    filename: str
    #: Digest recorded when this artifact was validated against the real video.
    #: Used only to report drift; the URL digest is the authority per-request.
    expected_sha256: str


#: The two known-good artifacts. Each has its own label, URL, and digest, and
#: they are never combined into one response.
WITNESS_FIXTURES: tuple[WitnessFixture, ...] = (
    WitnessFixture(
        key="EVOLV",
        label="TEST - EVOLV ALASS WITNESS (known-good, do not edit)",
        filename="02_EVOLV_alass_output.srt",
        expected_sha256=(
            "2c95ea92cfb9d83431b16d7e3f82301ae0d4d3686b5c5523f8cd651589edad9b"
        ),
    ),
    WitnessFixture(
        key="ASAP",
        label="TEST - ASAP ALASS WITNESS (known-good, do not edit)",
        filename="04_ASAP_alass_output.srt",
        expected_sha256=(
            "63556bdb9a409cbc0b94d2ac2cfa20fca236847c42059dc59d20f4cb98c5d930"
        ),
    ),
)


def _fixture_by_key(key: str) -> WitnessFixture:
    for fx in WITNESS_FIXTURES:
        if fx.key.lower() == key.lower():
            return fx
    raise HTTPException(status_code=404, detail="unknown witness fixture")


# --------------------------------------------------------------------------- #
# Stremio request parsing
# --------------------------------------------------------------------------- #
# Stremio asks for subtitles as
#
#   /subtitles/{type}/{id}[/{extra}].json
#
# e.g. /subtitles/series/tt0773262%3A8%3A4/videoSize%3D...%26filename%3D....json
#
# The ASGI server percent-decodes path parameters before they reach a handler, so
# ``id`` arrives as "tt0773262:8:4" and ``extra`` as "videoSize=...&filename=...".
# Both are decoded defensively here as well, because a client that double-encodes
# is not malformed input to be trusted blindly.

#: Only these media types can ever match the witness target.
_ALLOWED_TYPES = frozenset({"series", "movie"})

#: Cap on the ``extra`` blob. It is read for logging only and never used to
#: select a file, so there is no reason to accept an unbounded string.
_MAX_EXTRA_LEN = 2048

#: Values of ``extra`` that are echoed into logs. Anything else is dropped, so a
#: caller cannot inject a secret or a control sequence through the URL.
_SAFE_EXTRA_KEYS = ("videoSize", "filename", "videoHash", "videoFilename")


def _decode(value: str) -> str:
    """Percent-decode, tolerating a malformed or already-decoded value.

    ``unquote`` leaves invalid escapes alone rather than raising, and the result
    is only ever compared against an allowlist, so a garbled value simply fails
    to match rather than being trusted.
    """
    from urllib.parse import unquote

    return unquote(value or "")


def _parse_stremio_target(media_type: str, raw_id: str) -> tuple[str, int, int] | None:
    """Strictly validate a Stremio ``{type}/{id}`` pair against the allowlist.

    Returns ``(imdb, season, episode)`` only for the one supported target, else
    None. Nothing here is derived from a path, so a traversal string cannot
    reach the filesystem: the comparison is against a fixed tuple.
    """
    media_type = _decode(media_type).strip().lower()
    if media_type not in _ALLOWED_TYPES:
        return None
    stem = _decode(raw_id).split(".")[0].strip()
    parts = stem.split(":")
    if len(parts) != 3:
        return None
    imdb_id, raw_season, raw_episode = parts
    # An IMDb id is 'tt' plus digits. Rejecting anything else here means a value
    # like '../../etc/passwd' can never be treated as an identifier.
    if not (imdb_id.startswith("tt") and imdb_id[2:].isdigit()):
        return None
    try:
        season, episode = int(raw_season), int(raw_episode)
    except ValueError:
        return None
    if (imdb_id, season, episode) != (TARGET_IMDB, TARGET_SEASON, TARGET_EPISODE):
        return None
    return imdb_id, season, episode


def _safe_extra_metadata(raw_extra: str) -> dict[str, str]:
    """Extract loggable metadata from Stremio's ``extra`` blob.

    Parsed for observability only. Deliberately inert:

    * no value is ever used to build a filesystem path, so a traversal string is
      just an inert string;
    * only allowlisted keys are kept, so a caller cannot smuggle a credential or
      a log control sequence through the URL;
    * a value is accepted only if it looks like a plain name, and is truncated;
    * order is not assumed -- Stremio sends ``videoSize`` and ``filename`` in
      either order.
    """
    if not raw_extra:
        return {}
    decoded = _decode(raw_extra)
    if len(decoded) > _MAX_EXTRA_LEN:
        logger.warning(
            "[witness] ignoring oversized extra argument (%d bytes, limit %d)",
            len(decoded),
            _MAX_EXTRA_LEN,
        )
        return {}
    out: dict[str, str] = {}
    for pair in decoded.split("&"):
        if not pair or "=" not in pair:
            continue
        key, _, value = pair.partition("=")
        key = key.strip()
        if key not in _SAFE_EXTRA_KEYS:
            continue
        # A name, not a path: reject anything with a separator or a parent ref.
        if "/" in value or "\\" in value or ".." in value:
            logger.warning("[witness] ignoring unsafe value for extra key %r", key)
            continue
        out[key] = value[:200]
    return out


class FixtureMountProblem(RuntimeError):
    """Why a fixture could not be served, in terms the operator can act on.

    A witness that answers "not available" without saying whether the host
    directory is missing, empty, or simply the wrong folder is indistinguishable
    from a broken service, and the usual cause -- a bind mount pointing at a path
    that does not exist on the host, which Docker silently creates as an empty
    directory -- is invisible from inside the container. So the reason is carried
    out to the caller and reported verbatim.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def _diagnose_fixture_dir() -> str | None:
    """Return None when the mount looks usable, else a human-readable reason."""
    root = _fixture_dir()
    try:
        if not root.exists():
            return (
                f"the configured fixture directory does not exist inside the "
                f"container: {root}"
            )
        if not root.is_dir():
            return f"the configured fixture path is not a directory: {root}"
        if not any(root.iterdir()):
            return (
                f"the fixture directory exists but is EMPTY: {root}. The host "
                f"folder was not mounted. Set WITNESS_FIXTURE_DIR in .env to the "
                f"folder containing 02_EVOLV_alass_output.srt and "
                f"04_ASAP_alass_output.srt, then recreate the witness service."
            )
    except OSError as exc:  # pragma: no cover - unreadable mount
        return f"the fixture directory could not be read: {root} ({exc})"
    return None


def _read_exact(fixture: WitnessFixture) -> tuple[bytes, str]:
    """Read the fixture's bytes and its digest, refusing anything inconsistent.

    Returns the exact bytes plus the digest they actually hash to. Raises
    FixtureMountProblem if the mount is unusable, and HTTP 409 if the digest
    contradicts what the caller asked for, because a URL promising one artifact
    must never serve another.
    """
    problem = _diagnose_fixture_dir()
    if problem:
        raise FixtureMountProblem("mount_unusable", problem)
    path = _fixture_dir() / fixture.filename
    if not path.is_file():
        raise FixtureMountProblem(
            "fixture_missing",
            f"{fixture.filename} is not present in {_fixture_dir()}. Expected the "
            f"pre-validated Alass output; do not substitute or regenerate it.",
        )
    data = path.read_bytes()
    return data, hashlib.sha256(data).hexdigest()


def _no_store_headers() -> dict[str, str]:
    return {
        # Belt. The versioned URL is the braces; this is the braces-and-braces.
        "Cache-Control": "no-store, max-age=0, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
        "X-NinjaSubs-Witness": WITNESS_MARKER,
    }


def build_witness_router() -> APIRouter:
    """Build the isolated router. Mounted only when the witness is enabled."""
    router = APIRouter(prefix=MOUNT, include_in_schema=False)

    def _require_enabled() -> None:
        # Checked per request as well as at mount time, so a settings change
        # cannot leave a live witness endpoint behind.
        if not witness_enabled():
            raise HTTPException(status_code=404, detail="not found")

    @router.get("/manifest.json")
    async def witness_manifest(request: Request) -> dict:
        _require_enabled()
        base = get_base_url(request).rstrip("/")
        return {
            "id": "org.ninjasubs.playback-witness",
            "version": "1.0.0",
            "name": "NinjaSubs Playback Witness (DEV ONLY)",
            "description": (
                "DEVELOPMENT/TEST ONLY. Serves one pre-validated Alass output "
                "byte-for-byte, with no provider search, no alass, and no "
                "verifier. It cannot produce a subtitle for anything other "
                f"than Dexter S{TARGET_SEASON:02d}E{TARGET_EPISODE:02d} "
                f"({TARGET_IMDB})."
            ),
            "logo": f"{base}/static/icon.png",
            "icon": f"{base}/static/icon.png",
            "resources": ["subtitles"],
            "types": ["series"],
            "idPrefixes": ["tt"],
            "catalogs": [],
            "behaviorHints": {"configurable": False, "configurationRequired": False},
        }

    def _build_subtitles_response(base: str) -> SubtitlesResponse:
        """The witness candidate list. No provider, ranking, sync, or verifier."""
        items: list[SubtitleItem] = []
        for fx in WITNESS_FIXTURES:
            try:
                _, digest = _read_exact(fx)
            except FixtureMountProblem:
                # A missing fixture is skipped, not fatal, so one absent file
                # does not hide the other witness.
                continue
            # Versioned by digest: a changed fixture yields a different URL, so
            # neither Stremio nor a proxy can serve a stale body for it.
            url = f"{base}{MOUNT}/subtitle/{fx.key}/{digest}.srt"
            items.append(
                SubtitleItem(
                    id=f"{WITNESS_MARKER}:{fx.key}:{digest[:16]}",
                    url=url,
                    # 'ara | label' is the form the player shows.
                    lang=f"ara | {fx.label}",
                    title=fx.label,
                    format="srt",
                )
            )
        return SubtitlesResponse(subtitles=items)

    # The route Stremio actually calls. This must be registered, and matched,
    # before the production catch-all `/{config}/subtitles/{type}/{id}/{extra}` --
    # without it the witness mount prefix is swallowed as a user config string
    # and the request is answered by the production aggregator, which returns
    # zero subtitles in the witness container because it holds no provider keys.
    # The witness router is included before the production routes are declared,
    # so first-registration-wins ordering puts these ahead of it.
    @router.api_route(
        "/subtitles/{media_type}/{media_id}/{extra:path}.json",
        methods=["GET", "HEAD", "OPTIONS"],
        response_model=SubtitlesResponse,
    )
    @router.api_route(
        "/subtitles/{media_type}/{media_id}/{extra:path}",
        methods=["GET", "HEAD", "OPTIONS"],
        response_model=SubtitlesResponse,
    )
    async def witness_subtitles_stremio(
        media_type: str, media_id: str, request: Request, extra: str = ""
    ) -> SubtitlesResponse:
        """Real Stremio shape: ``/subtitles/{type}/{id}/{extra}.json``.

        ``extra`` is parsed for logging only and is never used to choose a file.
        """
        _require_enabled()
        target = _parse_stremio_target(media_type, media_id)
        if target is None:
            return SubtitlesResponse(subtitles=[])
        metadata = _safe_extra_metadata(extra)
        logger.info(
            "[witness] subtitle request type=%s id=%s s=%d e=%d extra_keys=%s",
            media_type,
            target[0],
            target[1],
            target[2],
            sorted(metadata) or "none",
        )
        return _build_subtitles_response(get_base_url(request).rstrip("/"))

    @router.api_route(
        "/subtitles/{media_type}/{media_id}.json",
        methods=["GET", "HEAD", "OPTIONS"],
        response_model=SubtitlesResponse,
    )
    @router.api_route(
        "/subtitles/{media_type}/{media_id}",
        methods=["GET", "HEAD", "OPTIONS"],
        response_model=SubtitlesResponse,
    )
    async def witness_subtitles_stremio_no_extra(
        media_type: str, media_id: str, request: Request
    ) -> SubtitlesResponse:
        """Stremio's shape when it sends no ``extra`` blob."""
        _require_enabled()
        target = _parse_stremio_target(media_type, media_id)
        if target is None:
            return SubtitlesResponse(subtitles=[])
        return _build_subtitles_response(get_base_url(request).rstrip("/"))

    @router.api_route(
        "/subtitles/{raw_id}.json",
        methods=["GET", "HEAD", "OPTIONS"],
        response_model=SubtitlesResponse,
    )
    async def witness_subtitles_legacy(
        raw_id: str, request: Request
    ) -> SubtitlesResponse:
        """The simple ``/subtitles/tt0773262:8:4.json`` form, kept for scripts.

        Kept deliberately narrower than the Stremio shape: it accepts a bare
        ``tt...:s:e`` id only, so it cannot be reached with a traversal string.
        """
        _require_enabled()
        target = _parse_stremio_target("series", raw_id)
        if target is None:
            return SubtitlesResponse(subtitles=[])
        return _build_subtitles_response(get_base_url(request).rstrip("/"))

    @router.api_route("/subtitle/{key}/{digest}.srt", methods=["GET", "HEAD"])
    async def witness_subtitle(key: str, digest: str) -> Response:
        """Return the fixture's exact bytes. No transformation of any kind."""
        _require_enabled()
        fixture = _fixture_by_key(key)
        try:
            data, actual = _read_exact(fixture)
        except FixtureMountProblem as problem:
            # Reported verbatim: an unusable mount is an operator problem, and
            # hiding the reason behind a bare 503 is what made the original
            # empty-directory mount so hard to diagnose.
            logger.error("[witness] cannot serve %s: %s", fixture.key, problem.detail)
            raise HTTPException(status_code=503, detail=problem.detail) from problem
        if actual.lower() != digest.lower():
            # The URL advertises one artifact and the file is another. Serving it
            # would defeat the entire point of hashing the URL.
            logger.warning(
                "[witness] digest mismatch for %s: url=%s actual=%s",
                fixture.key,
                digest[:16],
                actual[:16],
            )
            raise HTTPException(status_code=409, detail="witness fixture digest mismatch")
        # Metadata only: name, size, digest. Never subtitle contents.
        logger.info(
            "[witness] served %s bytes=%d sha256=%s",
            fixture.key,
            len(data),
            actual[:16],
        )
        return Response(
            content=data,
            media_type="text/plain; charset=utf-8",
            headers={
                **_no_store_headers(),
                # Length and digest, so the client can confirm byte identity.
                "X-Witness-Fixture": fixture.key,
                "X-Witness-Sha256": actual,
                "X-Witness-Bytes": str(len(data)),
            },
        )

    @router.get("/fixtures.json")
    async def witness_fixtures() -> dict:
        """Safe diagnostics: is the mount usable, and does it still match?

        Names, sizes, and digests only. No contents, no credentials.

        The ``mount_problem`` field is the important one: Docker creates a
        missing bind-mount source as an empty directory, so a mistyped
        WITNESS_FIXTURE_DIR produces an enabled service that serves nothing. This
        states that outright instead of leaving it to be inferred.
        """
        _require_enabled()
        mount_problem = _diagnose_fixture_dir()
        entries = []
        for fx in WITNESS_FIXTURES:
            try:
                data, digest = _read_exact(fx)
            except FixtureMountProblem as problem:
                entries.append(
                    {
                        "key": fx.key,
                        "label": fx.label,
                        "available": False,
                        "reason": problem.reason,
                        "detail": problem.detail,
                    }
                )
                continue
            entries.append(
                {
                    "key": fx.key,
                    "label": fx.label,
                    "filename": fx.filename,
                    "available": True,
                    "bytes": len(data),
                    "sha256": digest,
                    "matches_validated_digest": digest == fx.expected_sha256,
                }
            )
        return {
            "marker": WITNESS_MARKER,
            "development_only": True,
            "fixture_dir": str(_fixture_dir()),
            "mount_usable": mount_problem is None,
            "mount_problem": mount_problem,
            "reminder": (
                None if mount_problem is None else
                "Set WITNESS_FIXTURE_DIR in .env to the host folder holding the "
                "pre-validated Alass outputs, then "
                "`docker compose --profile witness up -d --force-recreate witness`."
            ),
            "target": {
                "imdb_id": TARGET_IMDB,
                "season": TARGET_SEASON,
                "episode": TARGET_EPISODE,
            },
            "fixtures": entries,
        }

    return router
