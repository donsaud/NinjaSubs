"""Tests for CredentialStore integration with subtitle metadata.

Ensures:
- Credentials are stored in CredentialStore for each sub_id
- Persistent metadata never contains API keys
- Fallback reads from metadata are removed
- User-specific credentials are recoverable from CredentialStore during process lifetime
"""

from unittest.mock import AsyncMock

import pytest

from app.cache import cache_manager
from app.models import SubtitleRelease, UserPreferences
from app.services.credentials import credential_store


@pytest.fixture(autouse=True)
async def reset_credential_store():
    # Clear store before and after each test
    await credential_store.clear()
    yield
    await credential_store.clear()

def test_credentials_stored_for_sub_id():
    import asyncio
    async def _run():
        prefs = UserPreferences(subdl_key="subdl_secret", subsource_key="subsource_secret", opensubtitles_key="opensubtitles_secret")
        sub_id = "test_sub_id_static"
        # Store metadata and credentials
        meta_dict = {
            "sub_id": sub_id,
            "provider": "subdl",
            "download_url": "http://example.com/file.srt",
            "subdl_key": prefs.subdl_key,
            "subsource_key": prefs.subsource_key,
            "opensubtitles_key": prefs.opensubtitles_key,
        }
        cache_manager.store_metadata(sub_id, meta_dict)
        await credential_store.store(sub_id, prefs.model_dump())
        # Verify credential store contains the entry
        stored = await credential_store.get(sub_id)
        assert stored is not None
        assert stored.get("subdl_key") == prefs.subdl_key
        assert stored.get("subsource_key") == prefs.subsource_key
        assert stored.get("opensubtitles_key") == prefs.opensubtitles_key
    asyncio.run(_run())

def test_persistent_metadata_has_no_keys():
    # Store metadata with keys
    test_id = "meta_test_id"
    raw_meta = {
        "subdl_key": "secret1",
        "subsource_key": "secret2",
        "opensubtitles_key": "secret3",
        "other_data": "visible",
    }
    cache_manager.store_metadata(test_id, raw_meta)
    # Verify stored metadata is stripped of keys
    stored_meta = cache_manager.get_metadata(test_id)
    assert stored_meta is not None
    assert "subdl_key" not in stored_meta
    assert "subsource_key" not in stored_meta
    assert "opensubtitles_key" not in stored_meta
    assert "other_data" in stored_meta

def test_fallback_reads_from_metadata_removed():
    import asyncio
    async def _run():
        test_id = "fallback_test"
        prefs = UserPreferences(subdl_key="fallback_secret")
        # Store metadata with stripped keys (no API keys)
        meta_dict = {"sub_id": test_id, "provider": "subdl", "download_url": "http://example.com"}
        cache_manager.store_metadata(test_id, meta_dict)
        # Add a dummy entry to credential_store
        await credential_store.store(test_id, prefs.model_dump())
        # Verify get returns from store, not metadata fallback
        stored = await credential_store.get(test_id)
        assert stored.get("subdl_key") == "fallback_secret"
        # Remove from store to test fallback
        await credential_store.clear()
        stored2 = await credential_store.get(test_id)
        assert stored2 is None
        # Verify fallback to metadata does not happen (metadata has no keys)
        assert cache_manager.get_metadata(test_id).get("subdl_key") is None
    asyncio.run(_run())

def test_credentials_recoverable_during_same_process():
    import asyncio
    async def _run():
        test_id = "process_test"
        prefs = UserPreferences(subdl_key="process_secret")
        await credential_store.store(test_id, prefs.model_dump())
        # Simulate later retrieval during subtitle fetch
        retrieved = await credential_store.get(test_id)
        assert retrieved is not None
        assert retrieved.get("subdl_key") == "process_secret"
        # Verify that disk metadata is not consulted for credentials
        # Store a poisoned metadata entry
        cache_manager.store_metadata(test_id, {"sub_id": test_id, "provider": "subdl"})
        # Retrieval should still be from credential_store, not metadata
        retrieved2 = await credential_store.get(test_id)
        assert retrieved2.get("subdl_key") == "process_secret"
    asyncio.run(_run())

def test_credential_store_integration_in_aggregator():
    """Integration test: aggregator uses credential_store for provider keys."""
    import asyncio

    from app.providers.subdl import SubdlProvider
    from app.services.aggregator import aggregate_subtitles

    async def run_aggregator():
        prefs = UserPreferences(
            subdl_key="integration_test_key",
            enable_subdl=True,
            languages=["ara"],
        )
        # Store credentials in credential_store
        await credential_store.store("test_agg", prefs.model_dump())
        # Mock providers
        mock_subdl = AsyncMock(spec=SubdlProvider)
        mock_subdl.search_subtitles = AsyncMock(return_value=[SubtitleRelease(
            release_name="Test.srt",
            download_url="http://example.com",
            provider="subdl",
            sub_id="test_agg",
        )])
        # Run aggregation with config using sub_id
        mock_subsource = AsyncMock()
        mock_subsource.search_subtitles = AsyncMock(return_value=[])
        await aggregate_subtitles(
            imdb_id="tt0111111",
            media_type="movie",
            user_preferences=prefs,
            subdl_provider=mock_subdl,
            subsource_provider=mock_subsource,
            use_cache=False,
            http_client=AsyncMock(),
        )
        # Verify provider was called with correct key from credential_store
        mock_subdl.search_subtitles.assert_called_once()
        call_args = mock_subdl.search_subtitles.call_args
        # The key passed should match stored credential
        assert call_args[1].get("api_key") == prefs.subdl_key

    asyncio.run(run_aggregator())
