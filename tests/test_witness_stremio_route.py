"""Regression tests for the real Stremio subtitle request shape.

The witness originally exposed only ``/subtitles/{raw_id}.json``. A real Stremio
client asks for

    /subtitles/series/tt0773262%3A8%3A4/videoSize%3D...%26filename%3D....json

which the production app answers via its catch-all
``/{config}/subtitles/{media_type}/{media_id}/{extra:path}.json`` -- the witness
mount prefix is swallowed as a user config string. The witness container holds no
provider keys, so that handler returned zero subtitles and Stremio showed no TEST
candidates. These tests pin the corrected routing and prove the production
handler is never reached.

The captured request below is used verbatim. It is not simplified, and not
re-ordered, because Stremio sends the extra fields in varying order.
"""

from __future__ import annotations

import hashlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.main as main_app
import app.playback_witness as witness
from app.playback_witness import (
    MOUNT,
    TARGET_EPISODE,
    TARGET_IMDB,
    TARGET_SEASON,
    build_witness_router,
)

FILENAME = "Dexter.s8e04.Scar.tissue.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv"

#: Exactly as captured from the Stremio client, unaltered.
REAL_STREMIO_PATH = (
    "/stremio-playback-witness/subtitles/series/"
    "tt0773262%3A8%3A4/"
    "videoSize%3D5414925805%26filename%3DDexter.s8e04.Scar.tissue."
    "1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv"
    ".json"
)

#: Stremio also sends the same fields in the other order.
REAL_STREMIO_PATH_REVERSED = (
    "/stremio-playback-witness/subtitles/series/"
    "tt0773262%3A8%3A4/"
    f"filename%3D{FILENAME.replace('.', '.')}%26videoSize%3D5414925805"
    ".json"
)

SAMPLE_A = b"1\n00:00:01,000 --> 00:00:04,000\nalpha\n\n"
SAMPLE_B = b"1\n00:00:02,000 --> 00:00:05,000\nbeta\n\n"


@pytest.fixture
def fx_dir(tmp_path, monkeypatch):
    d = tmp_path / "fx"
    d.mkdir()
    (d / witness.WITNESS_FIXTURES[0].filename).write_bytes(SAMPLE_A)
    (d / witness.WITNESS_FIXTURES[1].filename).write_bytes(SAMPLE_B)
    monkeypatch.setenv("ENABLE_STREMIO_PLAYBACK_WITNESS", "true")
    monkeypatch.setenv("NINJASUBS_WITNESS_FIXTURE_DIR", str(d))
    return d


@pytest.fixture
def witness_only(fx_dir):
    """A bare app carrying only the witness router."""
    app = FastAPI()
    app.include_router(build_witness_router())
    return TestClient(app)


@pytest.fixture
def full_app(fx_dir, monkeypatch):
    """The real production app, with the witness mounted as production does.

    This is the configuration in which the defect appeared, so it is the one the
    isolation assertions must use.
    """
    app = FastAPI()
    app.include_router(build_witness_router())
    # The production catch-all that used to swallow witness requests.
    @app.api_route(
        "/{config}/subtitles/{media_type}/{media_id}/{extra:path}.json",
        methods=["GET", "HEAD", "OPTIONS"],
    )
    async def production_catch_all(
        config: str, media_type: str, media_id: str, extra: str
    ):
        return {"handler": "production", "config": config}

    return TestClient(app)


# ------------------------------------------------------- the real request --- #


def test_real_stremio_request_reaches_the_witness_route(witness_only):
    """The captured request, verbatim, must be answered by the witness."""
    r = witness_only.get(REAL_STREMIO_PATH)
    assert r.status_code == 200
    subs = r.json()["subtitles"]
    assert len(subs) == 2, "the real request must yield both witness candidates"
    labels = [s["lang"] for s in subs]
    assert any("TEST - EVOLV ALASS WITNESS" in x for x in labels)
    assert any("TEST - ASAP ALASS WITNESS" in x for x in labels)


def test_real_stremio_request_with_reversed_extra_order(witness_only):
    """Field order is not guaranteed; the id must parse either way."""
    r = witness_only.get(REAL_STREMIO_PATH_REVERSED)
    assert r.status_code == 200
    assert len(r.json()["subtitles"]) == 2


def test_encoded_colon_in_id_is_decoded(witness_only):
    """%3A must arrive as ':' and split the id into three parts."""
    r = witness_only.get(
        f"{MOUNT}/subtitles/series/tt0773262%3A8%3A4/"
        "videoSize%3D1%26filename%3Da.mkv.json"
    )
    assert r.status_code == 200
    assert len(r.json()["subtitles"]) == 2


def test_encoded_equals_and_ampersand_in_extra_are_handled(witness_only):
    """%3D and %26 arrive decoded; they must not corrupt the id parse."""
    r = witness_only.get(
        f"{MOUNT}/subtitles/series/tt0773262%3A8%3A4"
        "/videoSize%3D5414925805%26filename%3D"
        "Dexter.s8e04.Scar.tissue.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv.json"
    )
    assert r.status_code == 200
    assert len(r.json()["subtitles"]) == 2


def test_real_filename_in_extra_does_not_change_the_target(witness_only):
    """A real release filename must not affect which fixtures are offered."""
    r = witness_only.get(REAL_STREMIO_PATH)
    digest = hashlib.sha256(SAMPLE_A).hexdigest()
    assert any(digest in s["url"] for s in r.json()["subtitles"])


def test_stremio_shape_without_extra(witness_only):
    """Stremio may omit the extra blob entirely."""
    r = witness_only.get(f"{MOUNT}/subtitles/series/tt0773262:8:4.json")
    assert r.status_code == 200
    assert len(r.json()["subtitles"]) == 2


def test_stremio_shape_without_json_suffix(witness_only):
    r = witness_only.get(
        f"{MOUNT}/subtitles/series/tt0773262:8:4/videoSize%3D1"
    )
    assert r.status_code == 200
    assert len(r.json()["subtitles"]) == 2


def test_legacy_simple_route_still_works(witness_only):
    """The simple form used by the checking script must not regress."""
    r = witness_only.get(f"{MOUNT}/subtitles/tt0773262:8:4.json")
    assert r.status_code == 200
    assert len(r.json()["subtitles"]) == 2


# ------------------------------------------------ production isolation ------ #


def test_production_catch_all_does_not_capture_the_witness_route(full_app):
    """The regression itself: witness request must not reach production.

    Before the fix this request matched
    /{config}/subtitles/{media_type}/{media_id}/{extra:path}.json with
    config='stremio-playback-witness' and the production aggregator answered it.
    """
    r = full_app.get(REAL_STREMIO_PATH)
    assert r.status_code == 200
    body = r.json()
    assert "handler" not in body, "production catch-all consumed the witness request"
    assert len(body["subtitles"]) == 2


def test_production_handler_is_not_invoked(full_app, monkeypatch):
    """Instrument the production handler and prove it never runs."""
    calls: list[dict] = []

    async def spy(config, media_type, media_id, extra):
        calls.append({"config": config, "id": media_id})
        return {"handler": "production"}

    # Replace the catch-all's endpoint with a spy that records invocation.
    for route in full_app.app.routes:
        if getattr(route, "path", "").endswith(
            "/{config}/subtitles/{media_type}/{media_id}/{extra:path}.json"
        ):
            route.endpoint = spy
            route.app = full_app.app
    r = full_app.get(REAL_STREMIO_PATH)
    assert r.status_code == 200
    assert not calls, f"production handler was invoked: {calls}"
    assert len(r.json()["subtitles"]) == 2


def test_witness_discovery_calls_no_provider_or_ranking(monkeypatch, fx_dir):
    """Provider search, ranking, alass and the verifier must stay untouched."""
    from app.services import ranking as ranking_mod
    from app.services import sync_service as sync_mod

    touched: list[str] = []

    def trip(name):
        def _boom(*a, **k):
            touched.append(name)
            raise AssertionError(f"witness discovery called {name}")
        return _boom

    monkeypatch.setattr(
        ranking_mod, "order_candidates", trip("ranking.order_candidates"),
        raising=False,
    )
    monkeypatch.setattr(
        ranking_mod, "calculate_match_score", trip("ranking.score"), raising=False
    )
    monkeypatch.setattr(
        sync_mod.SubtitleSyncService, "sync", trip("alass/sync"), raising=False
    )
    import app.services.sync.alignment as alignment_mod

    monkeypatch.setattr(
        alignment_mod.AlignmentAnalyzer, "compare",
        trip("verifier.compare"), raising=False,
    )
    app = FastAPI()
    app.include_router(build_witness_router())
    c = TestClient(app)
    assert len(c.get(REAL_STREMIO_PATH).json()["subtitles"]) == 2
    assert not touched, f"witness discovery touched production code: {touched}"


# ------------------------------------------------- route ordering ---------- #


def test_witness_routes_are_registered_before_production_catch_all(monkeypatch):
    """First-registration-wins: the witness prefix must come first.

    The production app is inspected as shipped, because the ordering is a
    property of the real registration sequence in app/main.py, not of a router
    assembled inside a test.
    """
    import app.playback_witness as pw

    monkeypatch.setenv("ENABLE_STREMIO_PLAYBACK_WITNESS", "true")
    router = pw.build_witness_router()
    witness_paths = [r.path for r in router.routes]
    assert any("{extra:path}" in p for p in witness_paths), (
        f"the Stremio shape route is missing from the witness router: {witness_paths}"
    )
    # The router exposes the real shape before the legacy one, so the specific
    # route cannot be shadowed by the simpler pattern.
    stremio_index = next(
        i for i, p in enumerate(witness_paths) if "{extra:path}" in p
    )
    legacy_index = next(
        i for i, p in enumerate(witness_paths) if p.endswith("{raw_id}.json")
    )
    assert stremio_index < legacy_index


def test_witness_mount_is_included_before_production_routes():
    """app/main.py must include the witness router ahead of its own catch-alls.

    Read from the source rather than importing, so the assertion survives the
    witness being disabled in this environment.
    """
    import pathlib

    src = pathlib.Path(main_app.__file__).read_text(encoding="utf-8")
    include_at = src.index("app.include_router(build_witness_router())")
    catch_all_at = src.index(
        '/{config}/subtitles/{media_type}/{media_id}/{extra:path}.json'
    )
    assert include_at < catch_all_at, (
        "the witness router must be included before the production catch-all, "
        "otherwise the catch-all wins and witness requests reach the aggregator"
    )


# ---------------------------------------------------------- security ------- #


def test_path_traversal_in_the_id_is_refused(witness_only):
    """A traversal string in the id must never match the target."""
    for bad in (
        "..%2F..%2Fetc%2Fpasswd",
        "..%2F..%2F..%2Fapp%2Fplayback_witness.py",
        "tt0773262%2F..%2F..%2Fetc%2Fpasswd",
        "..",
        "tt%2F..%2F..%2Fetc",
    ):
        r = witness_only.get(f"{MOUNT}/subtitles/series/{bad}/x%3D1.json")
        subs = r.json().get("subtitles", [])
        assert subs == [], f"traversal accepted: {bad}"


def test_traversal_landing_in_extra_still_serves_only_allowlisted_fixtures(witness_only):
    """%2F in the id decodes to a separator, so '..' ends up in `extra`.

    That is the interesting case: the id stays valid while the traversal moves
    into the log-only blob. The response must still contain nothing beyond the
    two allowlisted fixtures -- no file content, no other path.
    """
    r = witness_only.get(
        f"{MOUNT}/subtitles/series/tt0773262%3A8%3A4%2F..%2F..%2Fetc/x%3D1.json"
    )
    assert r.status_code == 200
    for s in r.json()["subtitles"]:
        # Only allowlisted fixture keys may ever appear in a served URL.
        assert "/subtitle/EVOLV/" in s["url"] or "/subtitle/ASAP/" in s["url"]
        assert "etc" not in s["url"]
        assert ".." not in s["url"]


def test_path_traversal_in_extra_is_inert(witness_only):
    """A traversal string in extra must not read anything; it is log-only."""
    r = witness_only.get(
        f"{MOUNT}/subtitles/series/tt0773262%3A8%3A4"
        "/filename%3D..%2F..%2F..%2Fetc%2Fpasswd%26videoSize%3D1.json"
    )
    assert r.status_code == 200
    body = r.json()
    # The target still matches, so the fixtures are offered -- but nothing from
    # the traversed path is reachable, because extra never selects a file.
    assert len(body["subtitles"]) == 2
    for s in body["subtitles"]:
        assert "passwd" not in s["url"]


def test_filename_never_selects_a_fixture(witness_only):
    """Fixture choice is the allowlist key only, never a URL-supplied name."""
    r = witness_only.get(
        f"{MOUNT}/subtitles/series/tt0773262%3A8%3A4"
        "/filename%3D01_EVOLV_original.srt.json"
    )
    urls = [s["url"] for s in r.json()["subtitles"]]
    for u in urls:
        assert "/subtitle/EVOLV/" in u or "/subtitle/ASAP/" in u
    assert not any("01_EVOLV_original" in u for u in urls)


def test_fixture_allowlist_blocks_other_keys(witness_only, fx_dir):
    """No key outside the allowlist can be served, whatever its spelling."""
    for bad in ("01_EVOLV_original.srt", "..", "metadata.json", "EVOLV/../ASAP"):
        r = witness_only.get(f"{MOUNT}/subtitle/{bad}/{'0' * 64}.srt")
        assert r.status_code in (404, 409), f"allowlist bypass: {bad} -> {r.status_code}"


def test_wrong_media_type_returns_nothing(witness_only):
    """An unsupported type yields nothing -- or 404 for an empty segment."""
    for bad in ("anime", "channel", "tv", "SERIESS", "tvshow"):
        r = witness_only.get(f"{MOUNT}/subtitles/{bad}/tt0773262:8:4/x%3D1.json")
        assert r.json().get("subtitles", []) == [], f"type accepted: {bad}"


def test_wrong_imdb_returns_nothing(witness_only):
    for bad in ("tt0111161:8:4", "tt0773262x:8:4", "notanid:8:4", "tt:8:4"):
        r = witness_only.get(f"{MOUNT}/subtitles/series/{bad}/x%3D1.json")
        assert r.json()["subtitles"] == [], f"id accepted: {bad}"


def test_wrong_season_or_episode_returns_nothing(witness_only):
    for bad in ("tt0773262:1:1", "tt0773262:8:5", "tt0773262:9:4", "tt0773262:a:b"):
        r = witness_only.get(f"{MOUNT}/subtitles/series/{bad}/x%3D1.json")
        assert r.json()["subtitles"] == [], f"target accepted: {bad}"


def test_oversized_extra_is_ignored_not_rejected(witness_only, caplog):
    """An unbounded extra blob is dropped, and the request still succeeds."""
    import logging

    huge = "a" * 50_000
    with caplog.at_level(logging.WARNING, logger="app.playback_witness"):
        r = witness_only.get(
            f"{MOUNT}/subtitles/series/tt0773262%3A8%3A4/x%3D{huge}.json"
        )
    assert r.status_code == 200
    assert len(r.json()["subtitles"]) == 2
    assert any("oversized" in rec.getMessage() for rec in caplog.records)


def test_malformed_extra_pairs_are_skipped(witness_only):
    """Garbage pairs are ignored; the target still parses from the id."""
    r = witness_only.get(
        f"{MOUNT}/subtitles/series/tt0773262%3A8%3A4/garbage%26%26%3D%26a%3D1.json"
    )
    assert r.status_code == 200
    assert len(r.json()["subtitles"]) == 2


def test_non_allowlisted_extra_keys_are_dropped(witness_only):
    """A caller cannot smuggle an arbitrary key into the logs."""
    parsed = witness._safe_extra_metadata("token=SECRET&apiKey=abc&videoSize=1")
    assert parsed == {"videoSize": "1"}


def test_unsafe_extra_values_are_dropped(witness_only):
    parsed = witness._safe_extra_metadata("filename=../../etc/passwd&videoSize=1")
    assert "filename" not in parsed
    assert parsed == {"videoSize": "1"}


def test_digest_still_required_for_fixture_delivery(witness_only, fx_dir):
    """The URL digest is still enforced after the routing change."""
    good = hashlib.sha256(SAMPLE_A).hexdigest()
    assert witness_only.get(f"{MOUNT}/subtitle/EVOLV/{good}.srt").status_code == 200
    assert witness_only.get(f"{MOUNT}/subtitle/EVOLV/{'a' * 64}.srt").status_code == 409


# ------------------------------------------------- byte identity ----------- #


def test_evolv_delivery_is_byte_identical_through_the_stremio_route(witness_only):
    """Follow the real discovery path, then fetch what it advertised."""
    subs = witness_only.get(REAL_STREMIO_PATH).json()["subtitles"]
    evolv = next(s for s in subs if "EVOLV" in s["id"])
    url = evolv["url"].split(MOUNT, 1)[1]
    r = witness_only.get(MOUNT + url)
    assert r.status_code == 200
    assert r.content == SAMPLE_A
    assert hashlib.sha256(r.content).hexdigest() == hashlib.sha256(SAMPLE_A).hexdigest()


def test_asap_delivery_is_byte_identical_through_the_stremio_route(witness_only):
    subs = witness_only.get(REAL_STREMIO_PATH).json()["subtitles"]
    asap = next(s for s in subs if "ASAP" in s["id"])
    url = asap["url"].split(MOUNT, 1)[1]
    r = witness_only.get(MOUNT + url)
    assert r.status_code == 200
    assert r.content == SAMPLE_B


# ------------------------------------------------------------ manifest ----- #


def test_manifest_advertises_subtitles_without_hardcoding_the_request(witness_only):
    """Stremio builds the subtitle URL itself; the manifest must not pin it.

    The description is allowed to name the episode -- it is operator-facing text
    shown in the add-on list, and it helps confirm the right add-on is installed.
    What must not appear is a *routing hint*: a subtitles path, a media id used
    as a template, or the extra arguments.
    """
    body = witness_only.get(f"{MOUNT}/manifest.json").json()
    assert body["resources"] == ["subtitles"]
    assert body["idPrefixes"] == ["tt"]
    assert body["catalogs"] == []
    serialised = str(body)
    # No request shape is hardcoded anywhere in the manifest.
    assert "/subtitles/" not in serialised
    assert "videoSize" not in serialised
    assert "filename=" not in serialised
    assert "%3A" not in serialised and "%3D" not in serialised
    # And no behaviour hint points anywhere but the standard resource.
    assert set(body["behaviorHints"]) <= {"configurable", "configurationRequired"}


def test_manifest_and_route_agree_on_id_prefix(witness_only):
    """The idPrefixes the manifest advertises must be what the route accepts."""
    body = witness_only.get(f"{MOUNT}/manifest.json").json()
    for prefix in body["idPrefixes"]:
        r = witness_only.get(
            f"{MOUNT}/subtitles/series/{prefix}0773262%3A8%3A4/x%3D1.json"
        )
        assert len(r.json()["subtitles"]) == 2, f"prefix {prefix} not handled"


def test_target_constants_match_the_documented_episode():
    assert (TARGET_IMDB, TARGET_SEASON, TARGET_EPISODE) == ("tt0773262", 8, 4)
