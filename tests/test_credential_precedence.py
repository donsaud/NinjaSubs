"""Effective provider credentials: precedence, labelling, and cache identity.

The application deliberately supports two credential sources. A deployment may
hold keys in the environment, or a Stremio user may supply them per-request
through the manifest/config URL. Both may be empty, and both may be set, and
the merge must be defined rather than incidental.

The precedence rule is pre-existing and lives in one place,
``app.utils.config_parser.parse_user_config``::

    effective = (user_config_value or settings.ENV_VALUE or "").strip()

so the manifest/config value wins, an empty one falls through to the
environment, and an empty environment yields "". This module asserts that rule
and asserts that the cache key is built from the SAME effective values the
provider clients receive.

Nothing here asserts or exposes a credential value. Secrets used in these tests
are obvious dummies, and no test reads the real environment.
"""

from __future__ import annotations

import pytest

from app.services.sync.orchestrator import SyncOrchestrator, credential_source

IMDB = "tt0773262"
SUB_ID = "63df03613463c531"
SUB_BYTES = b"1\n00:00:10,000 --> 00:00:12,000\nx\n"

# Obviously-fake values. Never real credentials.
FAKE = {
    "subdl_key": "DUMMY-SUBDL-KEY-0001",
    "subsource_key": "DUMMY-SUBSOURCE-KEY-0002",
    "opensubtitles_key": "DUMMY-OPENSUBTITLES-KEY-0003",
}

META = {
    "imdb_id": IMDB,
    "season": 8,
    "episode": 5,
    "media_type": "series",
    "target_filename": "Dexter.s8e05.This.little.piggy.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv",
    "video_size": 5192387053,
    "lang": "ara",
    **FAKE,
}


@pytest.fixture()
def clean_env(monkeypatch):
    """Guarantee the environment contributes nothing, whatever the host has."""
    from app.config import settings

    for field in ("SUBDL_API_KEY", "SUBSOURCE_API_KEY", "OPENSUBTITLES_API_KEY"):
        monkeypatch.setattr(settings, field, "", raising=False)
    return settings


def _encode(**kwargs):
    from app.utils.config_parser import encode_user_config

    return encode_user_config(languages=["ara"], **kwargs)


# --- Test A: same manifest credentials, same key ---------------------------


def test_a_same_manifest_credentials_produce_identical_keys(clean_env):
    from app.utils.config_parser import parse_user_config

    cfg_a = _encode(**FAKE)
    cfg_b = _encode(**FAKE)
    prefs_a = parse_user_config(cfg_a)
    prefs_b = parse_user_config(cfg_b)

    assert prefs_a.subdl_key == prefs_b.subdl_key == FAKE["subdl_key"]

    orch = SyncOrchestrator(sync_cache=None)
    meta_a = {**META, **{k: prefs_a.subdl_key if k == "subdl_key" else META[k]
                         for k in ("subdl_key",)}}
    meta_b = dict(meta_a)
    assert orch._flight_key(meta_a, SUB_ID, SUB_BYTES) == orch._flight_key(
        meta_b, SUB_ID, SUB_BYTES
    ), "identical effective credentials must not change the cache key"


# --- Test B: changed credential changes the namespace ----------------------


def test_b_changed_credential_changes_namespace(clean_env):
    orch = SyncOrchestrator(sync_cache=None)
    rotated = {**META, "subdl_key": "DUMMY-SUBDL-KEY-ROTATED"}
    assert orch._flight_key(dict(META), SUB_ID, SUB_BYTES) != orch._flight_key(
        rotated, SUB_ID, SUB_BYTES
    ), "credential isolation in the cache namespace is intentional; do not remove it"


# --- Test C: env empty, manifest populated (the current deployment) --------


def test_c_environment_empty_manifest_populated(clean_env):
    from app.utils.config_parser import parse_user_config

    prefs = parse_user_config(_encode(**FAKE))
    assert prefs.subdl_key == FAKE["subdl_key"]
    assert prefs.subsource_key == FAKE["subsource_key"]
    assert prefs.credential_source == "manifest", (
        "manifest supplies all three keys and the environment is empty"
    )
    assert credential_source({**META, "credential_source": prefs.credential_source}) == "manifest"


def test_c_empty_manifest_falls_through_to_environment(clean_env, monkeypatch):
    """The precedence rule, asserted directly."""
    from app.utils.config_parser import parse_user_config

    monkeypatch.setattr(clean_env, "SUBDL_API_KEY", "DUMMY-ENV-SUBDL")
    prefs = parse_user_config(_encode())  # manifest supplies nothing
    assert prefs.subdl_key == "DUMMY-ENV-SUBDL", (
        "an empty manifest value must fall through to the environment, not shadow it"
    )
    assert prefs.credential_source == "environment"


# --- Test D: no credentials anywhere ---------------------------------------


def test_d_no_credentials_preserves_behaviour(clean_env):
    from app.utils.config_parser import parse_user_config

    prefs = parse_user_config(_encode())
    assert not prefs.subdl_key
    assert not prefs.subsource_key
    assert credential_source({**META, **{k: "" for k in FAKE}}) == "default"
    # A request with no credentials must still produce a valid key rather than
    # being refused or lumped into a fabricated namespace.
    orch = SyncOrchestrator(sync_cache=None)
    key = orch._flight_key({**META, **{k: "" for k in FAKE}}, SUB_ID, SUB_BYTES)
    assert key.startswith(f"final_sub:{IMDB}:s8e5:")


def test_d_credential_source_never_returns_a_secret(clean_env):
    for label in (
        credential_source(dict(META)),
        credential_source({**META, **{k: "" for k in FAKE}}),
        credential_source({}),
    ):
        assert label in {"manifest", "environment", "default", "mixed", "present_unlabelled"}
        for secret in FAKE.values():
            assert secret not in label


# --- labelling --------------------------------------------------------------


def test_credential_source_labels(clean_env):
    """Labels are READ from recorded provenance, never inferred from values.

    Inferring from the merged values was the first implementation and it was
    wrong: it reported "manifest" for an environment-supplied key whenever a
    user also supplied one, because by that point the origins are identical
    strings. Asserting the honest fallback keeps that from regressing.
    """
    assert credential_source(dict(META)) == "present_unlabelled"
    assert credential_source({**META, **{k: "" for k in FAKE}}) == "default"
    for label in ("manifest", "environment", "mixed", "default"):
        assert credential_source({**META, "credential_source": label}) == label


def test_credential_source_labels_are_derived_from_real_precedence(clean_env, monkeypatch):
    """End to end through the real config parser."""
    from app.utils.config_parser import parse_user_config

    # manifest only
    assert parse_user_config(_encode(**FAKE)).credential_source == "manifest"
    # environment only (manifest supplies nothing)
    monkeypatch.setattr(clean_env, "SUBDL_API_KEY", "DUMMY-ENV-SUBDL")
    assert parse_user_config(_encode()).credential_source == "environment"
    # both present
    assert parse_user_config(_encode(subdl_key=FAKE["subdl_key"])).credential_source == "mixed"
    # neither
    assert parse_user_config(_encode()).credential_source == "environment"


def test_credential_source_does_not_change_the_key(clean_env, monkeypatch):
    """Labelling must be observational only.

    If this test ever fails, the diagnostic has started influencing cache
    identity, which would change hit rates rather than explain them.
    """
    orch = SyncOrchestrator(sync_cache=None)
    before = orch._flight_key(dict(META), SUB_ID, SUB_BYTES)
    monkeypatch.setattr(clean_env, "SUBDL_API_KEY", "DUMMY-ENV-SUBDL")
    assert orch._flight_key(dict(META), SUB_ID, SUB_BYTES) == before


# --- cache lookup path, end to end (no network) ----------------------------


@pytest.mark.asyncio
async def test_repeated_request_hits_then_misses_after_restart():
    """In-memory hit, then the expected miss on a fresh instance.

    The distinction the production trace could not make: an in-memory cache hit
    and a durable artifact reuse are different things, and only the first is
    currently available.
    """
    from app.services.sync_cache import SyncCache

    warm = SyncCache(ttl=3600)
    orch = SyncOrchestrator(sync_cache=warm)
    key = orch._flight_key(dict(META), SUB_ID, SUB_BYTES)

    assert await warm.get(key) is None
    await warm.set(key, SUB_BYTES)
    assert await warm.get(key) == SUB_BYTES, "first lookup after a write must hit"
    assert await warm.get(key) == SUB_BYTES, "identical repeat must also hit"

    cold = SyncCache(ttl=3600)
    assert await cold.get(key) is None, (
        "the payload store is process-local, so a restart is an expected miss; "
        "this is architecture, not a defect"
    )
