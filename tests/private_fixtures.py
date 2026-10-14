"""Optional private subtitle fixtures for real-byte regression coverage.

Real subtitle payloads (``.srt``/``.vtt``/``.ass``/``.ssa``) are
private/local-only and are never committed to Git. Tests that measure
genuine release bytes call :func:`require_subtitle` (or
:func:`read_subtitle_text`): when the file is present the test runs
unchanged; when absent the test skips with an informative reason instead
of failing.

To enable the real-byte suites locally, place the real subtitle bytes at
the exact relative paths the tests reference (e.g.
``tests/fixtures/penny/penny-s01e01-hdtv-ar.srt``). Those paths are
gitignored, so private fixtures can never be committed by accident.
Synthetic-data suites are unaffected and always run.
"""

from __future__ import annotations

import pathlib

import pytest


def require_subtitle(path: str | pathlib.Path) -> pathlib.Path:
    """Return ``path`` if the private fixture exists, else skip the test."""
    candidate = pathlib.Path(path)
    if not candidate.is_file():
        pytest.skip(
            f"private subtitle fixture absent: {candidate.name} "
            "(place real bytes at this path to enable; never commit them)"
        )
    return candidate


def read_subtitle_text(path: str | pathlib.Path, encoding: str = "utf-8-sig") -> str:
    """Read a private subtitle fixture, skipping when absent."""
    return require_subtitle(path).read_bytes().decode(encoding)


def require_many(paths) -> list[pathlib.Path]:
    """Require several private fixtures at once, else skip once."""
    missing = [pathlib.Path(path).name for path in paths
               if not pathlib.Path(path).is_file()]
    if missing:
        pytest.skip(
            f"private subtitle fixtures absent: {', '.join(sorted(missing))} "
            "(place real bytes at these paths to enable; never commit them)"
        )
    return [pathlib.Path(path) for path in paths]
