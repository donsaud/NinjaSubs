"""Tests for the synced-subtitle cache (Redis with in-process TTL fallback)."""

import pytest

from app.services.sync_cache import SyncCache


def test_build_key_format():
    assert (
        SyncCache.build_key("tt0111161", "s1e2", "vhash1234", "abc123", "team")
        == "final_sub:tt0111161:s1e2:vhash1234:abc123:team"
    )


def test_build_key_binds_exact_payload():
    key = SyncCache.build_key("tt1", "s1e1", "vhash", "sub1", "edition", "deadbeef12345678")
    assert key == "final_sub:tt1:s1e1:vhash:sub1:edition:deadbeef12345678"
    # Fingerprint, decision, and content hash isolate media, editions, payloads.
    assert SyncCache.build_key("tt1", "s1e1", "other", "sub1", "team") != SyncCache.build_key(
        "tt1", "s1e1", "vhash", "sub1", "team"
    )
    assert SyncCache.build_key("tt1", "s1e1", "vhash", "sub1", "team") != SyncCache.build_key(
        "tt1", "s1e1", "vhash", "sub1", "edition"
    )


@pytest.mark.asyncio
async def test_local_ttl_roundtrip():
    cache = SyncCache()
    key = SyncCache.build_key("tt1", "movie", "vhash", "sub1", "team", "abc123")
    assert await cache.get(key) is None
    await cache.set(key, b"1\n00:00:01,000 --> 00:00:02,000\nx\n")
    assert (await cache.get(key)).startswith(b"1\n")


@pytest.mark.asyncio
async def test_connect_without_redis_url_is_noop():
    cache = SyncCache()
    await cache.connect()
    assert cache._redis is None  # noqa: SLF001 - intentional internal assertion
    await cache.close()


@pytest.mark.asyncio
async def test_meta_roundtrip():
    cache = SyncCache()
    key = SyncCache.build_key("tt1", "movie", "vhash", "sub1", "team", "abc123")
    assert await cache.get_meta(key) is None
    await cache.set_meta(key, {"status": "synced", "decision": "team"})
    assert await cache.get_meta(key) == {"status": "synced", "decision": "team"}


@pytest.mark.asyncio
async def test_meta_rejects_garbage():
    cache = SyncCache()
    key = SyncCache.build_key("tt1", "movie", "vhash", "sub1")
    cache._local_meta[SyncCache._meta_key(key)] = "not-json{{{"
    assert await cache.get_meta(key) is None


@pytest.mark.asyncio
async def test_negative_cache_roundtrip():
    cache = SyncCache()
    key = SyncCache.build_key("tt1", "s1e1", "vhash", "sub1")
    assert await cache.is_failed(key) is False
    await cache.mark_failed(key)
    assert await cache.is_failed(key) is True
    await cache.clear_failed(key)
    assert await cache.is_failed(key) is False


@pytest.mark.asyncio
async def test_negative_cache_isolates_keys():
    cache = SyncCache()
    a = SyncCache.build_key("tt1", "s1e1", "vhash", "sub1")
    b = SyncCache.build_key("tt1", "s1e1", "vhash", "sub2")
    await cache.mark_failed(a)
    assert await cache.is_failed(a) is True
    assert await cache.is_failed(b) is False
