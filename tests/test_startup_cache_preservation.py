"""Container startup must not destroy persisted cache state.

Production finding: every container recreate deleted the entire on-disk metadata
directory through the /app/cache bind mount. Measured at 122 files lost on a
single recreate, and it is why cached artifacts kept vanishing between deploys.

The cause was a single call in the FastAPI lifespan hook:

    cache_manager.clear_metadata()   # unlink() on every _meta/*.json

The intent was to drop in-memory state on boot. The effect was to wipe
persisted state through a volume that exists precisely to be persistent.

These tests drive real application startup, because a source-level assertion
("does this string appear in main.py") proves nothing: the call can come back
under a different name, or via a helper. Only starting the app and looking at
the filesystem is evidence.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def meta_dir():
    """The live cache_manager's metadata directory (tests/cache/_meta)."""
    from app.cache import cache_manager

    meta_dir = cache_manager.meta_dir
    meta_dir.mkdir(parents=True, exist_ok=True)
    before = set(meta_dir.glob("*.json"))
    yield meta_dir
    for stale in set(meta_dir.glob("*.json")) - before:
        stale.unlink()


def _seed(meta_dir, name: str) -> None:
    (meta_dir / name).write_text(json.dumps({"imdb_id": "tt0773262", "lang": "ara"}),
                                 encoding="utf-8")


def test_startup_preserves_persisted_metadata(meta_dir):
    """Seeded metadata must survive an application startup."""
    _seed(meta_dir, "persisted_across_restart.json")

    from app.main import app

    with TestClient(app) as _client:  # entering the context runs lifespan
        pass

    assert (meta_dir / "persisted_across_restart.json").exists(), (
        "startup deleted persisted metadata from the cache mount; this is the "
        "regression that lost 122 files per container recreate"
    )


def test_startup_preserves_metadata_across_several_restarts(meta_dir):
    """Repeat starts must not erode state either."""
    _seed(meta_dir, "survives_many.json")

    from app.main import app

    for _ in range(3):
        with TestClient(app) as _client:
            pass

    assert (meta_dir / "survives_many.json").exists()


def test_startup_still_clears_in_memory_state(meta_dir):
    """The in-memory clears are the part that was actually wanted."""
    from app.services.cache import SUBTITLE_CACHE

    SUBTITLE_CACHE["sentinel"] = b"in-memory only"

    from app.main import app

    with TestClient(app) as _client:
        assert "sentinel" not in SUBTITLE_CACHE, (
            "in-memory cache must still be invalidated on startup"
        )


def test_ensure_dirs_still_runs(meta_dir):
    """A fresh volume legitimately has no directories; they must be created."""
    from app.main import app

    with TestClient(app) as _client:
        pass

    from app.cache import cache_manager

    assert cache_manager.cache_dir.is_dir()
    assert cache_manager.meta_dir.is_dir()


def test_clear_metadata_still_exists_as_an_explicit_operation(meta_dir):
    """The capability is retained; it is just no longer called on boot.

    Deleting the method outright would be a larger change than the fix needs,
    and it would break any caller that wants a deliberate wipe.
    """
    from app.cache import cache_manager

    _seed(meta_dir, "explicit_wipe.json")
    assert (meta_dir / "explicit_wipe.json").exists()
    cache_manager.clear_metadata()
    assert not (meta_dir / "explicit_wipe.json").exists()
