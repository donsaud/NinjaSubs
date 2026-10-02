"""Tests for the STREMIO_PLAYBACK_WITNESS dev-only delivery path.

The witness exists to answer one question: can Stremio display an
already-known-good Alass output when that exact output is handed to it directly?
To keep that answer trustworthy, the path must be provably inert with respect to
the production pipeline. These tests pin both halves of that claim:

  * the bytes are delivered exactly, and the URL is bound to them by digest;
  * nothing in production is involved, reachable, or modified.

Fixtures are generated in a temp directory rather than read from the real
Dexter media, so the suite has no dependency on multi-GB local media and the
repository never has to contain it.
"""

from __future__ import annotations

import hashlib
import pathlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.playback_witness as witness
from app.playback_witness import (
    MOUNT,
    TARGET_EPISODE,
    TARGET_IMDB,
    TARGET_SEASON,
    WITNESS_FIXTURES,
    build_witness_router,
    witness_enabled,
)

ENVVAR = "ENABLE_STREMIO_PLAYBACK_WITNESS"
FIXTURE_ENV = "NINJASUBS_WITNESS_FIXTURE_DIR"

# A small, well-formed SRT. Not a real artifact: these tests verify transport
# fidelity, not subtitle content, and must not depend on the Dexter media.
SAMPLE_A = (
    b"1\n00:00:01,000 --> 00:00:04,000\nfirst line\n\n"
    b"2\n00:00:05,500 --> 00:00:09,000\nsecond line\n\n"
)
# Deliberately exercises things a transport must not normalise: CRLF endings,
# a UTF-8 BOM, trailing whitespace, and a non-ASCII payload.
SAMPLE_B = (
    "﻿1\r\n00:00:02,000 --> 00:00:06,000\r\n"
    "مرحبا بالعالم  \r\n\r\n"
    "2\r\n00:00:07,000 --> 00:00:11,000\r\n"
    "{\\pos(960,900)}styled\ttext\r\n"
).encode()


@pytest.fixture
def witness_env(tmp_path, monkeypatch):
    """An enabled witness backed by a temp fixture directory."""
    fx = tmp_path / "fx"
    fx.mkdir()
    (fx / WITNESS_FIXTURES[0].filename).write_bytes(SAMPLE_A)
    (fx / WITNESS_FIXTURES[1].filename).write_bytes(SAMPLE_B)
    monkeypatch.setenv(ENVVAR, "true")
    monkeypatch.setenv(FIXTURE_ENV, str(fx))
    return fx


@pytest.fixture
def client(witness_env):
    """A minimal app carrying only the witness router.

    Deliberately not app.main: a test must not be able to reach a production
    route from here even by accident.
    """
    app = FastAPI()
    app.include_router(build_witness_router())
    return TestClient(app), witness_env


def _sub_url(client, key):
    r = client.get(f"/{MOUNT.lstrip('/')}/subtitles/{TARGET_IMDB}:"
                   f"{TARGET_SEASON}:{TARGET_EPISODE}.json")
    assert r.status_code == 200
    for item in r.json()["subtitles"]:
        if key in item["id"]:
            return item["url"]
    raise AssertionError(f"no witness subtitle for {key}")


# --------------------------------------------------------------- gating ---- #


def test_witness_is_disabled_by_default(monkeypatch):
    """Absent any configuration, the witness must be off."""
    monkeypatch.delenv(ENVVAR, raising=False)
    monkeypatch.setattr(
        "app.config.settings.ENABLE_STREMIO_PLAYBACK_WITNESS", False, raising=False
    )
    assert witness_enabled() is False


@pytest.mark.parametrize("raw", ["", "false", "0", "no", "off", "maybe", "TRUEISH"])
def test_witness_stays_off_unless_explicitly_enabled(raw, monkeypatch):
    """Only an explicit true enables it. Anything else, including nonsense, is off."""
    monkeypatch.setenv(ENVVAR, raw)
    monkeypatch.setattr(
        "app.config.settings.ENABLE_STREMIO_PLAYBACK_WITNESS", False, raising=False
    )
    if raw.strip().lower() in ("true", "1", "yes", "on"):
        return
    assert witness_enabled() is False, f"{raw!r} must not enable the witness"


@pytest.mark.parametrize("raw", ["true", "1", "yes", "on", "TRUE", " On "])
def test_witness_enables_on_explicit_true(raw, monkeypatch):
    monkeypatch.setenv(ENVVAR, raw)
    assert witness_enabled() is True


def test_routes_are_not_mounted_on_the_production_app_by_default():
    """The production app must expose no witness route at all when disabled.

    A 404-guarded route would be weaker than an absent one: a mount that does
    not exist cannot be reached by a scanner or an OpenAPI crawl.
    """
    import app.main as main_app

    paths = set()
    for route in main_app.app.routes:
        nested = getattr(route, "routes", None)
        if nested:
            paths.update(getattr(r, "path", "") for r in nested)
        paths.add(getattr(route, "path", ""))
    assert not [p for p in paths if MOUNT in p], (
        f"witness routes present on the production app while disabled: "
        f"{[p for p in paths if MOUNT in p]}"
    )


def test_endpoints_are_refused_when_disabled(tmp_path, monkeypatch):
    """Even a directly-built router 404s when the flag is off."""
    monkeypatch.delenv(ENVVAR, raising=False)
    monkeypatch.setenv(FIXTURE_ENV, str(tmp_path))
    monkeypatch.setattr(
        "app.config.settings.ENABLE_STREMIO_PLAYBACK_WITNESS", False, raising=False
    )
    c = TestClient(FastAPI())
    c.app.include_router(build_witness_router())
    for path in (
        f"{MOUNT}/manifest.json",
        f"{MOUNT}/fixtures.json",
        f"{MOUNT}/subtitles/{TARGET_IMDB}:{TARGET_SEASON}:{TARGET_EPISODE}.json",
        f"{MOUNT}/subtitle/EVOLV/{'0' * 64}.srt",
    ):
        assert c.get(path).status_code == 404, path


# --------------------------------------------------------- byte fidelity --- #


def test_subtitle_bytes_are_identical_to_the_source_file(client):
    """The response body must be the file's bytes, nothing else."""
    c, fx = client
    for fixture, expected in zip(WITNESS_FIXTURES, (SAMPLE_A, SAMPLE_B), strict=True):
        url = _sub_url(c, fixture.key)
        r = c.get(url.replace("http://testserver", ""))
        assert r.status_code == 200
        assert r.content == expected, f"{fixture.key} bytes were altered in transit"
        assert r.content == (fx / fixture.filename).read_bytes()


def test_served_digest_matches_the_source_digest(client):
    c, fx = client
    for fixture in WITNESS_FIXTURES:
        url = _sub_url(c, fixture.key)
        r = c.get(url.replace("http://testserver", ""))
        source = hashlib.sha256((fx / fixture.filename).read_bytes()).hexdigest()
        assert hashlib.sha256(r.content).hexdigest() == source
        assert r.headers["x-witness-sha256"] == source
        assert int(r.headers["x-witness-bytes"]) == len(r.content)


def test_url_is_versioned_by_the_fixture_digest(client):
    """The URL must embed the digest, so a changed fixture cannot be cache-hit."""
    c, fx = client
    for fixture in WITNESS_FIXTURES:
        url = _sub_url(c, fixture.key)
        digest = hashlib.sha256((fx / fixture.filename).read_bytes()).hexdigest()
        assert digest in url, f"{fixture.key} URL is not versioned by digest: {url}"


def test_url_changes_when_the_fixture_changes(client):
    """Replacing the fixture must change the advertised URL."""
    c, fx = client
    before = _sub_url(c, WITNESS_FIXTURES[0].key)
    (fx / WITNESS_FIXTURES[0].filename).write_bytes(SAMPLE_A + b"\n3\n00:00:12,000 --> 00:00:13,000\nthird\n")
    after = _sub_url(c, WITNESS_FIXTURES[0].key)
    assert before != after, "a changed fixture must yield a different URL"


def test_url_claiming_the_wrong_digest_is_refused(client):
    """A URL that advertises one digest must never serve different bytes."""
    c, _fx = client
    r = c.get(f"{MOUNT}/subtitle/EVOLV/{'a' * 64}.srt")
    assert r.status_code == 409
    r = c.get(f"{MOUNT}/subtitle/EVOLV/deadbeef.srt")
    assert r.status_code == 409


def test_unknown_fixture_key_is_not_found(client):
    c, _fx = client
    assert c.get(f"{MOUNT}/subtitle/NOPE/{'0' * 64}.srt").status_code == 404


def test_no_transformation_is_applied(client):
    """BOM, CRLF, tabs, RTL text and position tags must all survive verbatim."""
    c, _fx = client
    url = _sub_url(c, WITNESS_FIXTURES[1].key)
    body = c.get(url.replace("http://testserver", "")).content
    assert body.startswith(b"\xef\xbb\xbf"), "BOM was stripped"
    assert b"\r\n" in body, "CRLF endings were rewritten"
    assert b"{\\pos(960,900)}" in body, "ASS position tag was stripped"
    assert "مرحبا بالعالم".encode() in body, "text was normalised"
    assert b"second line  " not in body or b"  \r\n" in body, "trailing space changed"


def test_response_headers_prevent_stale_caching(client):
    c, _fx = client
    url = _sub_url(c, WITNESS_FIXTURES[0].key)
    r = c.get(url.replace("http://testserver", ""))
    assert "no-store" in r.headers["cache-control"]
    assert r.headers["content-type"] == "text/plain; charset=utf-8"
    assert r.headers["x-ninjasubs-witness"] == witness.WITNESS_MARKER


# ------------------------------------------------------------- isolation --- #


def test_module_does_not_import_any_production_pipeline_component():
    """The witness must be a leaf: no provider, sync, verifier, or cache imports.

    Checked against the parsed import statements rather than the raw text,
    because the module's docstring deliberately *names* every component it
    excludes -- a substring search would match that documentation and report a
    violation that does not exist.
    """
    import ast

    banned = {
        "SubDLProvider", "SubSourceProvider", "OpenSubtitlesProvider",
        "YifysubtitlesProvider", "SubtitlecatProvider",
        "SyncOrchestrator", "SubtitleSyncService", "AlignmentAnalyzer",
        "SyncCache", "cache_manager", "LRUCacheManager", "aggregate_subtitles",
        "order_candidates", "SyncPredictor", "convert_ass_to_srt",
        "convert_ass_to_srt_bytes", "clean_subtitle_bytes", "strip_kashida",
        "reference_resolver", "ReferenceResolver",
    }
    tree = ast.parse(pathlib.Path(witness.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    app_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.update(a.asname or a.name for a in node.names)
            module = node.module or ""
            if module.startswith("app"):
                app_modules.add(module)
    # Nothing in the banned set may be imported.
    overlap = imported & banned
    assert not overlap, f"witness imports production components: {sorted(overlap)}"

    # And of the project's own modules, only these three may be referenced. The
    # router additionally reads settings lazily from app.config inside the gate
    # function, which is why it appears here.
    allowed_modules = {"app.models", "app.utils.network", "app.config"}
    assert app_modules <= allowed_modules, (
        f"witness reaches into unexpected project modules: "
        f"{sorted(app_modules - allowed_modules)}"
    )


def test_serving_a_witness_writes_nothing_to_the_provider_cache(client, tmp_path, monkeypatch):
    """A witness request must not read or write the provider cache."""
    import app.cache as cache_mod

    cache_dir = tmp_path / "provider_cache"
    cache_dir.mkdir()
    (cache_dir / "sentinel.srt").write_bytes(b"UNTOUCHED")
    monkeypatch.setattr(
        cache_mod.cache_manager, "cache_dir", str(cache_dir), raising=False
    )
    c, _fx = client
    before = sorted(p.name for p in cache_dir.iterdir())
    before_mtime = (cache_dir / "sentinel.srt").stat().st_mtime_ns

    url = _sub_url(c, WITNESS_FIXTURES[0].key)
    assert c.get(url.replace("http://testserver", "")).status_code == 200

    after = sorted(p.name for p in cache_dir.iterdir())
    assert before == after
    assert (cache_dir / "sentinel.srt").stat().st_mtime_ns == before_mtime
    assert (cache_dir / "sentinel.srt").read_bytes() == b"UNTOUCHED"


def test_witness_does_not_consult_the_sync_cache(client, monkeypatch):
    """No verdict lookup may occur on a witness request."""
    from app.services.sync_cache import SyncCache

    touched: list[str] = []

    original = SyncCache.get_verdict
    original_alias = SyncCache.get_verdict_by_ref

    def _trip(*a, **k):
        touched.append("get_verdict")
        raise AssertionError("witness consulted the sync verdict cache")

    def _trip_alias(*a, **k):
        touched.append("get_verdict_by_ref")
        raise AssertionError("witness consulted the sync verdict alias index")

    monkeypatch.setattr(SyncCache, "get_verdict", _trip)
    monkeypatch.setattr(SyncCache, "get_verdict_by_ref", _trip_alias)
    c, _fx = client
    url = _sub_url(c, WITNESS_FIXTURES[0].key)
    assert c.get(url.replace("http://testserver", "")).status_code == 200
    assert not touched
    monkeypatch.setattr(SyncCache, "get_verdict", original)
    monkeypatch.setattr(SyncCache, "get_verdict_by_ref", original_alias)


def test_witness_request_does_not_spawn_alass(client, monkeypatch):
    """Alass must not run for a witness request."""
    import shutil as _shutil

    def _boom(*a, **k):
        raise AssertionError("witness invoked alass")

    monkeypatch.setattr(_shutil, "which", lambda *a, **k: None)
    import app.services.sync_service as svc

    monkeypatch.setattr(svc.SubtitleSyncService, "sync", _boom, raising=False)

    c, _fx = client
    url = _sub_url(c, WITNESS_FIXTURES[0].key)
    assert c.get(url.replace("http://testserver", "")).status_code == 200


# -------------------------------------------------------------- targeting -- #


def test_subtitle_list_serves_only_the_one_target(client):
    c, _fx = client
    base = f"{MOUNT}/subtitles"
    r = c.get(f"{base}/{TARGET_IMDB}:{TARGET_SEASON}:{TARGET_EPISODE}.json")
    assert len(r.json()["subtitles"]) == 2
    for other in (
        "tt0111161:1:1",
        "tt0773262:1:1",
        "tt0773262:8:5",
        f"{TARGET_IMDB}",
        "kitsu:1234:1",
        "not-an-id",
    ):
        rr = c.get(f"{base}/{other}.json")
        assert rr.status_code == 200
        assert rr.json()["subtitles"] == [], f"{other} must yield nothing"


def test_each_fixture_has_its_own_url_hash_and_label(client):
    c, fx = client
    r = c.get(f"{MOUNT}/subtitles/{TARGET_IMDB}:"
              f"{TARGET_SEASON}:{TARGET_EPISODE}.json")
    items = r.json()["subtitles"]
    assert len({i["url"] for i in items}) == 2, "URLs must differ per fixture"
    assert len({i["lang"] for i in items}) == 2, "labels must differ per fixture"
    digests = {hashlib.sha256((fx / f.filename).read_bytes()).hexdigest()
               for f in WITNESS_FIXTURES}
    assert all(any(d in i["url"] for d in digests) for i in items)


def test_labels_are_visibly_test_only(client):
    """A witness must be unmistakable in a subtitle picker."""
    c, _fx = client
    r = c.get(f"{MOUNT}/subtitles/{TARGET_IMDB}:"
              f"{TARGET_SEASON}:{TARGET_EPISODE}.json")
    for item in r.json()["subtitles"]:
        assert "TEST" in item["lang"].upper()
        assert "WITNESS" in item["lang"].upper()
        assert "STREMIO_PLAYBACK_WITNESS" in item["id"]


def test_manifest_is_a_minimal_subtitle_addon(client):
    c, _fx = client
    body = c.get(f"{MOUNT}/manifest.json").json()
    assert body["resources"] == ["subtitles"]
    assert body["idPrefixes"] == ["tt"]
    assert body["catalogs"] == []
    assert "DEV ONLY" in body["name"]


# ---------------------------------------------------------------- secrets -- #


def test_witness_logs_expose_no_contents_and_no_credentials(client, caplog, monkeypatch):
    """Logs carry name, size and digest only -- never text or a key."""
    monkeypatch.setenv("SUBDL_API_KEY", "SENTINEL_KEY_MUST_NOT_BE_LOGGED")
    monkeypatch.setenv("SUBSOURCE_API_KEY", "SENTINEL_KEY_MUST_NOT_BE_LOGGED")
    c, _fx = client
    with caplog.at_level("INFO", logger="app.playback_witness"):
        url = _sub_url(c, WITNESS_FIXTURES[0].key)
        c.get(url.replace("http://testserver", ""))
        c.get(f"{MOUNT}/fixtures.json")
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "SENTINEL_KEY_MUST_NOT_BE_LOGGED" not in logged
    # None of the subtitle's own text may appear.
    for line in SAMPLE_A.decode().splitlines():
        if line and not line.isdigit() and "-->" not in line:
            assert line not in logged
    assert "00:00:01" not in logged, "a timestamp leaked into a log line"


def test_digest_mismatch_logs_a_digest_not_contents(client, caplog):
    c, _fx = client
    with caplog.at_level("WARNING", logger="app.playback_witness"):
        c.get(f"{MOUNT}/subtitle/EVOLV/{'b' * 64}.srt")
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "digest mismatch" in logged
    assert "first line" not in logged


# ------------------------------------------------------------- deployment -- #


def test_no_media_is_required_for_the_suite_to_pass(tmp_path, monkeypatch):
    """With no fixture directory at all, the endpoint degrades, never crashes."""
    monkeypatch.setenv(ENVVAR, "true")
    monkeypatch.setenv(FIXTURE_ENV, str(tmp_path / "does-not-exist"))
    c = TestClient(FastAPI())
    c.app.include_router(build_witness_router())
    assert c.get(f"{MOUNT}/manifest.json").status_code == 200
    assert c.get(f"{MOUNT}/fixtures.json").status_code == 200
    r = c.get(f"{MOUNT}/subtitles/{TARGET_IMDB}:"
              f"{TARGET_SEASON}:{TARGET_EPISODE}.json")
    assert r.status_code == 200 and r.json()["subtitles"] == []
    assert c.get(f"{MOUNT}/subtitle/EVOLV/{'0' * 64}.srt").status_code == 503


def test_witness_mount_is_far_from_production_subtitle_routes():
    """The two URL spaces must be disjoint, so a cached body cannot cross over."""
    assert MOUNT.startswith("/stremio-playback-witness")
    assert "/sub/" not in MOUNT
    for fixture in WITNESS_FIXTURES:
        assert "stremio-playback-witness" in f"{MOUNT}/subtitle/{fixture.key}/x.srt"


def test_production_port_serves_no_witness_functionality(monkeypatch):
    """Typing the witness path on the production port must not reach a witness.

    The production app has a long-standing catch-all at ``/{config}/manifest``,
    which happens to match ``/stremio-playback-witness/manifest`` and returns the
    *production* manifest. That is harmless -- no witness capability is exposed
    -- but it means a mistyped port yields the normal NinjaSubs addon rather than
    a 404, which would look like a working install. Pinned here so the behaviour
    cannot change silently, and called out in the witness documentation.
    """
    monkeypatch.delenv(ENVVAR, raising=False)
    monkeypatch.setattr(
        "app.config.settings.ENABLE_STREMIO_PLAYBACK_WITNESS", False, raising=False
    )
    from fastapi.testclient import TestClient

    import app.main as main_app

    c = TestClient(main_app.app)
    # The manifest-shaped path answers, but with production content.
    r = c.get(f"{MOUNT}/manifest.json")
    body = r.text
    assert witness.WITNESS_MARKER not in body, "witness content leaked to the production port"
    assert "Playback Witness" not in body
    # Every other witness path is genuinely absent.
    for path in (
        f"{MOUNT}/fixtures.json",
        f"{MOUNT}/subtitle/EVOLV/{'0' * 64}.srt",
        f"{MOUNT}/subtitles/{TARGET_IMDB}:{TARGET_SEASON}:{TARGET_EPISODE}.json",
    ):
        assert c.get(path).status_code == 404, f"{path} must not exist on production"


def test_repo_does_not_track_the_real_fixtures():
    """The fixtures live on the host; the repository must never contain them."""
    import subprocess

    if not pathlib.Path(".git").exists():
        pytest.skip("not a git checkout; nothing to assert about tracked files")
    tracked = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=False
    ).stdout.split()
    for fixture in WITNESS_FIXTURES:
        assert fixture.filename not in tracked
    for f in tracked:
        assert f.lower().endswith((".mkv", ".mp4")) is False
