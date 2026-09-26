"""Tests for cache directory self-healing (metadata persistence)."""

import shutil

import pytest

from app.cache import LRUCacheManager


def _manager(tmp_path):
    return LRUCacheManager(cache_dir=str(tmp_path / "cache"), max_bytes=100000, max_files=10)


def test_store_metadata_recreates_missing_meta_dir(tmp_path):
    """A vanished ``_meta`` directory must be recreated before the write."""
    manager = _manager(tmp_path)
    shutil.rmtree(manager.meta_dir, ignore_errors=True)
    assert not manager.meta_dir.exists()

    manager.store_metadata("abc123", {"imdb_id": "tt1", "provider": "subdl"})

    assert (manager.meta_dir / "abc123.json").is_file()
    assert manager.get_metadata("abc123") == {"imdb_id": "tt1", "provider": "subdl"}


def test_store_metadata_recreates_removed_cache_root(tmp_path):
    """A missing cache root is recreated too (fresh/re-mounted volume)."""
    manager = _manager(tmp_path)
    shutil.rmtree(manager.cache_dir, ignore_errors=True)
    assert not manager.cache_dir.exists()

    manager.store_metadata("abc123", {"imdb_id": "tt1"})

    assert manager.get_metadata("abc123") == {"imdb_id": "tt1"}


def test_ensure_dirs_recreates_root_and_meta(tmp_path):
    manager = _manager(tmp_path)
    shutil.rmtree(manager.cache_dir, ignore_errors=True)

    assert manager.ensure_dirs() is True
    assert manager.cache_dir.is_dir()
    assert manager.meta_dir.is_dir()


@pytest.mark.asyncio
async def test_save_subtitle_recreates_removed_cache_root(tmp_path):
    manager = _manager(tmp_path)
    shutil.rmtree(manager.cache_dir, ignore_errors=True)

    saved = await manager.save_subtitle(
        "abc123", b"1\n00:00:01,000 --> 00:00:02,000\nhi\n"
    )

    assert saved is True
    assert (manager.cache_dir / "abc123.srt").is_file()
