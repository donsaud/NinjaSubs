"""`subs_cache` must hold provider bytes, never a synchronization artifact.

Provenance defect proven in production flow (Whiplash 2014 2160p REMUX,
``d777e212d36310c7``):

* request 1 downloaded the provider subtitle, ran alass, and the analyzer
  returned ``unverified`` / ``unknown`` (median +266 ms, p95 4200 ms);
* the handler then wrote the *synchronized* bytes back over the provider entry
  (75326 B, content hash ``fc5c1b17...``);
* request 2 read those transformed bytes back as the source subtitle -- its
  content hash was ``fc5c1b17...`` instead of the provider's ``fffeeac6...`` --
  skipped alass because the input now "was already aligned" (+0.03 s), and
  re-reported ``probable_sync`` against the *same* reference (``9dd5b4d0...``).

That is a circular verification: the artifact being judged was the product of
the judgement, and the evidence was not independent. No provenance was recorded,
so transformed bytes were indistinguishable from provider bytes.

These tests pin the corrected contract: the disk subtitle cache is the
*provider* cache. Synchronized artifacts live in ``SyncCache``, which reuses
only verified-and-measured results.
"""

import pytest
from starlette.requests import Request

import app.main as main
from app.utils.config_parser import encode_user_config

SUB_ID = "d777e212d36310c7"


def _original() -> bytes:
    return (
        b"1\n00:00:01,000 --> 00:00:03,000\nORIGINAL PROVIDER CUE ONE\n\n"
        b"2\n00:00:04,000 --> 00:00:06,000\nORIGINAL PROVIDER CUE TWO\n"
    )


def _transformed() -> bytes:
    return (
        b"1\n00:00:21,000 --> 00:00:23,000\nORIGINAL PROVIDER CUE ONE\n\n"
        b"2\n00:00:24,000 --> 00:00:26,000\nORIGINAL PROVIDER CUE TWO\n"
    )


def _meta() -> dict:
    return {
        "sub_id": SUB_ID,
        "imdb_id": "tt2582802",
        "media_type": "movie",
        "lang": "ara",
        "release_name": "Whiplash 2014 1080p WEB-DL x264 AAC-JYK.srt",
        "target_filename": "Whiplash.2014.UHD.BluRay.2160p.REMUX-FraMeSToR",
        "video_size": "47843180000",
    }


def _request(config: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": f"/{config}/sub/{SUB_ID}.srt",
            "query_string": b"imdb=tt2582802&type=movie&filename=Whiplash.2014.REMUX-FraMeSToR",
            "headers": [],
        }
    )


@pytest.fixture
def provider_cache(tmp_path, monkeypatch):
    """Point the real disk cache at a temp dir holding the provider bytes."""
    import httpx

    root = tmp_path / "cache"
    (root / "_meta").mkdir(parents=True)
    monkeypatch.setattr(main.cache_manager, "cache_dir", root, raising=False)
    monkeypatch.setattr(main.cache_manager, "meta_dir", root / "_meta", raising=False)
    monkeypatch.setattr(main, "_http_client", httpx.AsyncClient(), raising=False)
    main.cache_manager.store_metadata(SUB_ID, _meta())
    return main.cache_manager


class TestSubsCacheProvenance:
    @pytest.mark.asyncio
    async def test_sync_output_is_not_written_over_provider_bytes(
        self, provider_cache, monkeypatch
    ):
        """The response is synchronized; the cache keeps the provider bytes."""

        original, transformed = _original(), _transformed()
        await provider_cache.save_subtitle(SUB_ID, original)

        async def fake_sync(*args, **kwargs):
            return transformed

        monkeypatch.setattr(main, "_sync_subtitle_for_response", fake_sync)
        config = encode_user_config(auto_sync=True, subdl_key="k")
        response = await main.serve_configured_subtitle(config, SUB_ID, _request(config))

        # The user still receives the synchronized subtitle.
        assert b"00:00:21,000" in response.body
        # ...but the cache still holds the provider bytes.
        cached = await provider_cache.get_subtitle(SUB_ID)
        assert cached == original, "synchronization artifact overwrote the provider cache"
        assert cached != transformed

    @pytest.mark.asyncio
    async def test_second_request_receives_original_bytes_not_the_artifact(
        self, provider_cache, monkeypatch
    ):
        """The decisive property: request 2 must re-verify from the true source.

        This is the circularity the defect created -- request 2 judging the
        output of request 1's alignment against the same reference.
        """

        original, transformed = _original(), _transformed()
        await provider_cache.save_subtitle(SUB_ID, original)

        seen: list[bytes] = []

        async def fake_sync(sub_bytes, *args, **kwargs):
            seen.append(sub_bytes)
            return transformed

        monkeypatch.setattr(main, "_sync_subtitle_for_response", fake_sync)
        config = encode_user_config(auto_sync=True, subdl_key="k")

        await main.serve_configured_subtitle(config, SUB_ID, _request(config))
        await main.serve_configured_subtitle(config, SUB_ID, _request(config))

        assert len(seen) == 2
        assert seen[0] == original
        assert seen[1] == original, (
            "second request re-analyzed the cached synchronization artifact "
            "instead of the original provider subtitle"
        )
        assert transformed not in seen

    @pytest.mark.asyncio
    async def test_cache_hit_path_never_writes_back(self, provider_cache, monkeypatch):
        """No provenance-destroying write happens on the cache-hit path."""

        original, transformed = _original(), _transformed()
        await provider_cache.save_subtitle(SUB_ID, original)

        writes: list[tuple] = []
        real_save = provider_cache.save_subtitle

        async def spy(sub_id, data):
            writes.append((sub_id, data))
            return await real_save(sub_id, data)

        async def fake_sync(*args, **kwargs):
            return transformed

        monkeypatch.setattr(provider_cache, "save_subtitle", spy)
        monkeypatch.setattr(main, "_sync_subtitle_for_response", fake_sync)
        config = encode_user_config(auto_sync=True, subdl_key="k")
        await main.serve_configured_subtitle(config, SUB_ID, _request(config))

        assert all(data != transformed for _, data in writes), (
            "a synchronization artifact was persisted to the provider cache"
        )

    @pytest.mark.asyncio
    async def test_unsynced_response_leaves_cache_untouched(self, provider_cache, monkeypatch):
        """When sync declines to change anything the cache is left alone too."""

        original = _original()
        await provider_cache.save_subtitle(SUB_ID, original)

        async def fake_sync(sub_bytes, *args, **kwargs):
            return sub_bytes

        monkeypatch.setattr(main, "_sync_subtitle_for_response", fake_sync)
        config = encode_user_config(auto_sync=True, subdl_key="k")
        await main.serve_configured_subtitle(config, SUB_ID, _request(config))

        assert await provider_cache.get_subtitle(SUB_ID) == original
