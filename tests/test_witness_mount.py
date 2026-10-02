"""Tests for the witness fixture *mount*, not the delivery path itself.

A missing bind mount is silent: Docker creates a missing source directory rather
than failing, so a misconfigured WITNESS_FIXTURE_DIR yields a running service
that serves nothing. That is what actually happened, and it was invisible from
inside the container. These tests pin the diagnosis so it cannot regress into a
bare "not available" again.

The compose mount itself is asserted by parsing docker-compose.yml, so the
wiring is checked without needing a running container.
"""

from __future__ import annotations

import hashlib
import pathlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.playback_witness as witness
from app.playback_witness import MOUNT, build_witness_router, witness_enabled

REPO = pathlib.Path(__file__).resolve().parent.parent
COMPOSE = REPO / "docker-compose.yml"
ENVVAR = "ENABLE_STREMIO_PLAYBACK_WITNESS"
FIXTURE_ENV = "NINJASUBS_WITNESS_FIXTURE_DIR"

GOOD_A = b"1\n00:00:01,000 --> 00:00:04,000\nalpha\n\n"
GOOD_B = b"1\n00:00:02,000 --> 00:00:05,000\nbeta\n\n"


def _client(tmp_path, monkeypatch, *, write=True) -> TestClient:
    """A witness app whose fixture directory is built to order."""
    fx = tmp_path / "fx"
    fx.mkdir(exist_ok=True)
    if write:
        (fx / witness.WITNESS_FIXTURES[0].filename).write_bytes(GOOD_A)
        (fx / witness.WITNESS_FIXTURES[1].filename).write_bytes(GOOD_B)
    monkeypatch.setenv(ENVVAR, "true")
    monkeypatch.setenv(FIXTURE_ENV, str(fx))
    app = FastAPI()
    app.include_router(build_witness_router())
    return TestClient(app)


# ------------------------------------------------------- missing the mount -- #


def test_missing_fixture_directory_is_reported_not_silent(tmp_path, monkeypatch):
    """A host folder that was never mounted must say so explicitly."""
    # Point at a path that does not exist at all -- a genuinely absent mount.
    monkeypatch.setenv(ENVVAR, "true")
    monkeypatch.setenv(FIXTURE_ENV, str(tmp_path / "never-created"))
    c = TestClient(FastAPI())
    c.app.include_router(build_witness_router())
    body = c.get(f"{MOUNT}/fixtures.json").json()
    assert body["mount_usable"] is False
    assert body["mount_problem"], "an unusable mount must carry a reason"
    assert "does not exist" in body["mount_problem"]
    for entry in body["fixtures"]:
        assert entry["available"] is False
        assert entry["reason"] == "mount_unusable"


def test_empty_fixture_directory_names_the_cause(tmp_path, monkeypatch):
    """The exact production failure: Docker mounted an empty directory.

    This is the case that was previously indistinguishable from a broken
    service, so the message has to name both the emptiness and the remedy.
    """
    c = _client(tmp_path, monkeypatch, write=False)
    body = c.get(f"{MOUNT}/fixtures.json").json()
    assert body["mount_usable"] is False
    problem = body["mount_problem"]
    assert "EMPTY" in problem
    assert "WITNESS_FIXTURE_DIR" in problem, "the remedy must be named"
    assert body["reminder"] and "force-recreate" in body["reminder"]


def test_serving_from_an_empty_mount_reports_the_reason(tmp_path, monkeypatch):
    """A 503 must carry the diagnosis, not a generic message."""
    c = _client(tmp_path, monkeypatch, write=False)
    digest = "0" * 64
    r = c.get(f"{MOUNT}/subtitle/EVOLV/{digest}.srt")
    assert r.status_code == 503
    assert "EMPTY" in r.json()["detail"] or "does not exist" in r.json()["detail"]


def test_wrong_directory_yields_no_subtitles_rather_than_wrong_ones(tmp_path, monkeypatch):
    """A folder that exists but holds no fixtures must offer nothing."""
    other = tmp_path / "wrong"
    other.mkdir()
    (other / "unrelated.txt").write_text("not a fixture", encoding="utf-8")
    monkeypatch.setenv(ENVVAR, "true")
    monkeypatch.setenv(FIXTURE_ENV, str(other))
    c = TestClient(FastAPI())
    c.app.include_router(build_witness_router())
    r = c.get(f"{MOUNT}/subtitles/tt0773262:8:4.json")
    assert r.json()["subtitles"] == []


def test_partial_mount_reports_only_the_present_fixture(tmp_path, monkeypatch):
    """One missing file must not be reported as a mount failure."""
    fx = tmp_path / "fx"
    fx.mkdir()
    (fx / witness.WITNESS_FIXTURES[0].filename).write_bytes(GOOD_A)
    monkeypatch.setenv(ENVVAR, "true")
    monkeypatch.setenv(FIXTURE_ENV, str(fx))
    c = TestClient(FastAPI())
    c.app.include_router(build_witness_router())
    body = c.get(f"{MOUNT}/fixtures.json").json()
    assert body["mount_usable"] is True, "a non-empty directory is a usable mount"
    states = {e["key"]: e["available"] for e in body["fixtures"]}
    assert states["EVOLV"] is True
    assert states["ASAP"] is False
    # And the one that is present is still offered.
    subs = c.get(f"{MOUNT}/subtitles/tt0773262:8:4.json").json()["subtitles"]
    assert [s["id"].split(":")[1] for s in subs] == ["EVOLV"]


# ------------------------------------------------------------- digest drift -- #


def test_digest_mismatch_is_reported_and_not_served(tmp_path, monkeypatch):
    """A fixture whose bytes changed must be flagged, not silently served."""
    c = _client(tmp_path, monkeypatch)
    # Replace a fixture with different content of the same name.
    fx = tmp_path / "fx"
    (fx / witness.WITNESS_FIXTURES[0].filename).write_bytes(b"1\n00:00:00,000 --> 00:00:01,000\nregenerated\n\n")
    body = c.get(f"{MOUNT}/fixtures.json").json()
    entry = next(e for e in body["fixtures"] if e["key"] == "EVOLV")
    assert entry["available"] is True
    assert entry["matches_validated_digest"] is False, (
        "a changed fixture must be reported as not matching the validated digest"
    )


def test_url_with_a_stale_digest_is_refused_after_a_change(tmp_path, monkeypatch):
    """The pre-change URL must stop working once the fixture changes."""
    c = _client(tmp_path, monkeypatch)
    fx = tmp_path / "fx"
    original = GOOD_A
    old_digest = hashlib.sha256(original).hexdigest()
    assert c.get(f"{MOUNT}/subtitle/EVOLV/{old_digest}.srt").status_code == 200
    (fx / witness.WITNESS_FIXTURES[0].filename).write_bytes(b"1\n00:00:00,000 --> 00:00:01,000\nregenerated\n\n")
    r = c.get(f"{MOUNT}/subtitle/EVOLV/{old_digest}.srt")
    assert r.status_code == 409, "a stale digest URL must not serve changed bytes"


# ------------------------------------------------------- compose mount wiring -- #


def _require(path: pathlib.Path):
    """Skip deployment assertions when run outside a checkout.

    ``docker-compose.yml``, ``.env.example`` and ``.gitignore`` are host-side
    deployment files: the runtime image carries only app/, docs/, tests/, tools/
    and requirements.txt. Asserting on their contents from inside a container
    would be testing the wrong thing, so these checks run where those files
    actually exist.
    """
    if not path.is_file():
        pytest.skip(f"{path.name} not present in this environment")
    return path


def _compose_witness_block() -> str:
    text = _require(COMPOSE).read_text(encoding="utf-8")
    start = text.index("\n  witness:")
    # The witness block runs to the end of the services mapping.
    tail = text[start:]
    end = tail.index("\nnetworks:")
    return tail[:end]


def test_compose_mounts_the_fixture_dir_read_only():
    """The witness mount must exist, be read-only, and target /witness."""
    block = _compose_witness_block()
    assert "type: bind" in block, "use long syntax; a Windows path contains a colon"
    assert "target: /witness" in block
    assert "read_only: true" in block
    assert "${WITNESS_FIXTURE_DIR" in block, "the host path must be configurable"


def test_compose_does_not_use_the_ambiguous_short_volume_syntax():
    """`C:\\path:/witness:ro` must not appear: the drive colon is ambiguous."""
    block = _compose_witness_block()
    volumes = block.split("volumes:")[1]
    short_form = [
        ln for ln in volumes.splitlines()
        if ln.strip().startswith("- ") and ":ro" in ln
    ]
    assert not short_form, f"short-form volume syntax found: {short_form}"


def test_witness_profile_is_opt_in():
    """A default `docker compose up` must never start the witness."""
    block = _compose_witness_block()
    assert "profiles:" in block and '"witness"' in block


def _production_config_lines() -> list[str]:
    """Config (non-comment) lines belonging to the production service.

    Comments are excluded deliberately: the witness service is documented in a
    comment block that sits above its definition and therefore falls inside the
    production slice. That prose legitimately mentions WITNESS_FIXTURE_DIR, and
    what matters is whether the *configuration* references it, not whether a
    human-readable note does.
    """
    text = _require(COMPOSE).read_text(encoding="utf-8")
    prod_block = text[text.index("\n  ninjasubs:"):text.index("\n  witness:")]
    return [
        ln for ln in prod_block.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


def test_production_service_never_receives_the_witness_mount():
    """The witness volume must not leak into the production service config."""
    config = _production_config_lines()
    joined = "\n".join(config)
    assert "/witness" not in joined
    assert "WITNESS_FIXTURE_DIR" not in joined
    assert "ENABLE_STREMIO_PLAYBACK_WITNESS" not in joined


def test_production_service_does_not_mount_the_fixture_directory():
    """Belt and braces: the fixture folder must not appear in production config."""
    joined = "\n".join(_production_config_lines())
    for name in (f.filename for f in witness.WITNESS_FIXTURES):
        assert name not in joined
    # Production keeps its own cache mount and nothing else.
    assert "./subs_cache:/app/cache" in joined


# --------------------------------------------------------------- packaging -- #


def test_witness_is_still_disabled_by_default(monkeypatch):
    """Repairing the mount must not have enabled anything anywhere."""
    # The application default is off.
    monkeypatch.delenv(ENVVAR, raising=False)
    monkeypatch.setattr(
        "app.config.settings.ENABLE_STREMIO_PLAYBACK_WITNESS", False, raising=False
    )
    assert witness_enabled() is False
    # Only the witness service turns it on, and only inside the opt-in profile.
    assert "ENABLE_STREMIO_PLAYBACK_WITNESS" in _compose_witness_block()
    assert "ENABLE_STREMIO_PLAYBACK_WITNESS" not in "\n".join(
        _production_config_lines()
    )


def test_env_example_documents_the_witness_variable():
    """A host operator must be able to discover the setting without reading code."""
    text = _require(REPO / ".env.example").read_text(encoding="utf-8")
    assert "WITNESS_FIXTURE_DIR" in text
    assert "check_witness_mount" in text, "the preflight tool must be advertised"


def test_auto_created_placeholder_is_gitignored():
    """Docker may create ./autosync-diagnostics; it must never be committed."""
    ignore = _require(REPO / ".gitignore").read_text(encoding="utf-8")
    assert "autosync-diagnostics/" in ignore


def test_no_fixture_or_media_file_is_tracked():
    if not pathlib.Path(".git").exists():
        pytest.skip("not a git checkout")
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=False
    ).stdout.split()
    for f in witness.WITNESS_FIXTURES:
        assert f.filename not in tracked
    assert not [t for t in tracked if t.lower().endswith((".mkv", ".mp4"))]


def test_preflight_tool_exists_and_is_importable():
    """The documented preflight command must actually be present."""
    path = _require(REPO / "tools" / "check_witness_mount.py")
    assert path.is_file()
    import subprocess

    proc = subprocess.run(
        ["python", str(path), "--help"], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0
    assert "WITNESS_FIXTURE_DIR" in proc.stdout
