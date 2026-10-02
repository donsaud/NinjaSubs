"""ENABLE_AUTOSYNC_TEST_OVERRIDE -- development/test only, OFF by default.

Purpose
-------
Answer one question:

    "Does the full production response path preserve known-good Alass timing?"

The verifier currently rejects the Alass output for the real Dexter S08E04 EVOLV
and ASAP subtitles and serves the provider's original instead. Manual playback
has established that the *known-good* Alass artifact for both cases is in fact
synchronised, and that the isolated witness endpoint delivers it correctly. So
rendering is not the problem. What remains untested is whether the ordinary
production serve-time pipeline -- ad removal, credits handling, diacritics,
eastern-Arabic numerals, RTL punctuation, encoding repair -- preserves that
timing on the way out.

This module makes that testable by substituting the already-validated artifact
for one request, so it travels the normal production response path. It is a
delivery experiment, **not** an acceptance change: no verdict, threshold, cache
entry, or verification state is altered, and nothing is written to any cache.

Safety
------
* **Off by default.** ``ENABLE_AUTOSYNC_TEST_OVERRIDE`` must be explicitly set to
  a true value. An absent, false, or unrecognised value leaves it off.
* **Allowlisted.** Only the two exact provider subtitle ids listed in
  ``KNOWN_CASES`` are affected. There is no wildcard and no way to name a file.
* **No caller-supplied path.** The filename is looked up from a constant. A
  request cannot influence which file is read, so this cannot be turned into an
  arbitrary file read.
* **Digest-pinned.** The fixture is served only if it still hashes to the value
  recorded when it was validated against the real video. A substituted or
  regenerated file is refused rather than served.
* **Cache-free.** Nothing here reads or writes the provider cache, ``SyncCache``,
  the verdict store, or the negative cache, so an override cannot pollute
  production state or create a reusable verified artifact.
* **DEV/TEST ONLY.** There is no runtime switch for a normal user, no per-request
  query parameter, and no configuration path that enables it by accident.

Removal is a one-line revert: delete this file and the hook in ``app/main.py``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pathlib
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: Marks every log line this module emits, so a forced delivery is always
#: attributable in the logs and can never be mistaken for a normal one.
OVERRIDE_MARKER = "AUTOSYNC_TEST_OVERRIDE"

ENVVAR = "ENABLE_AUTOSYNC_TEST_OVERRIDE"
FIXTURE_ENV = "NINJASUBS_TEST_OVERRIDE_FIXTURE_DIR"

_TRUE = ("true", "1", "yes", "on")
#: Explicit off values. ``docker-compose`` ships ``false`` as the default for
#: this variable, so these are the *normal* production state: they are off by
#: design, not surprising, and must not be logged as an unrecognised value on
#: every request. Anything else stays off too, but is worth a warning.
_FALSE = ("", "false", "0", "no", "off")


@dataclass(frozen=True)
class KnownCase:
    """One real subtitle whose Alass output was independently validated.

    ``sub_id`` is the provider subtitle id the production route already uses, so
    matching it needs no new plumbing and no client input.
    """

    sub_id: str
    label: str
    filename: str
    #: SHA-256 of the artifact as validated against the real Dexter S08E04 video.
    expected_sha256: str


#: The complete allowlist. Two entries, both real and both previously measured.
#: Nothing here is user-extensible at runtime.
KNOWN_CASES: tuple[KnownCase, ...] = (
    KnownCase(
        sub_id="74a34c232d159f8b",
        label="EVOLV",
        filename="02_EVOLV_alass_output.srt",
        expected_sha256=(
            "2c95ea92cfb9d83431b16d7e3f82301ae0d4d3686b5c5523f8cd651589edad9b"
        ),
    ),
    KnownCase(
        sub_id="96c0442cc4656685",
        label="ASAP",
        filename="04_ASAP_alass_output.srt",
        expected_sha256=(
            "63556bdb9a409cbc0b94d2ac2cfa20fca236847c42059dc59d20f4cb98c5d930"
        ),
    ),
)


def override_enabled() -> bool:
    """True only when the override is explicitly switched on.

    Off unless the environment says otherwise. An unrecognised value is treated
    as off rather than assumed on, so a typo cannot enable a test path.
    """
    raw = os.getenv(ENVVAR, "").strip().lower()
    if raw in _TRUE:
        return True
    if raw not in _FALSE:
        logger.warning(
            "[%s] ignoring unrecognised %s value; staying off", OVERRIDE_MARKER, ENVVAR
        )
    return False


def _fixture_dir() -> pathlib.Path:
    """Directory holding the pre-validated Alass outputs.

    Resolved the same way the project's other real-media tooling resolves it, so
    one setting works for both. Nothing is bundled: with the directory absent the
    override simply declines to act.
    """
    explicit = os.getenv(FIXTURE_ENV, "").strip()
    if explicit:
        return pathlib.Path(explicit)
    media_root = os.getenv("NINJASUBS_REAL_MEDIA_ROOT", "").strip()
    if media_root:
        return pathlib.Path(media_root) / "autosync-diagnostics"
    return pathlib.Path("autosync-diagnostics")


def case_for_sub_id(sub_id: str) -> KnownCase | None:
    """Exact allowlist lookup. No prefix, wildcard, or normalisation."""
    if not sub_id:
        return None
    for case in KNOWN_CASES:
        if case.sub_id == sub_id:
            return case
    return None


def forced_transformed_bytes(sub_id: str) -> bytes | None:
    """Return the validated artifact for an allowlisted subtitle, or None.

    None means "do not override", and is returned for every one of these:

      * the override is not enabled;
      * the subtitle id is not in the allowlist;
      * the fixture file is absent;
      * the fixture no longer matches its recorded digest.

    The digest check is what makes this safe to leave in a codebase: if someone
    regenerates or substitutes the file, the override refuses rather than
    delivering an artifact that was never validated against the real video.
    """
    if not override_enabled():
        return None
    case = case_for_sub_id(sub_id)
    if case is None:
        return None
    path = _fixture_dir() / case.filename
    if not path.is_file():
        logger.warning(
            "[%s] fixture absent for %s: %s (not overriding)",
            OVERRIDE_MARKER, case.label, path,
        )
        return None
    try:
        data = path.read_bytes()
    except OSError as exc:  # pragma: no cover - unreadable fixture
        logger.warning("[%s] cannot read %s: %s", OVERRIDE_MARKER, case.label, exc)
        return None
    actual = hashlib.sha256(data).hexdigest()
    if actual != case.expected_sha256:
        # Refuse loudly. Serving this would present an unvalidated artifact as
        # the known-good one, which is the exact confusion this module must
        # never create.
        logger.error(
            "[%s] REFUSING %s: fixture digest %s is not the validated %s",
            OVERRIDE_MARKER, case.label, actual[:16], case.expected_sha256[:16],
        )
        return None
    # Metadata only -- name, size, digest. Never subtitle contents.
    logger.warning(
        "[%s] DEV/TEST override active: sub_id=%s case=%s bytes=%d sha256=%s",
        OVERRIDE_MARKER, case.sub_id, case.label, len(data), actual[:16],
    )
    return data
