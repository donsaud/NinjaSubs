"""COMPATIBLE_RELEASE must not be attributed to the requested release.

Real Stremio-facing defect: candidate ``f3d00658998773ea`` is listed as
``[95%] [SubDL] The.Last.of.Us.S01E01.WEBRip.x264-ION10 (by ali talal)`` and its
stored ``release_name`` is the ION10 release, but the provider could not supply
it, ``COMPATIBLE_RELEASE`` selected a different release, and that release's bytes
were served. The listing therefore named release A while release B was delivered,
and the label was identical for every candidate that fell back the same way.

Fallback *selection* is intentional and unchanged. This module covers only the
disclosure of what was actually served:

* non-fallback listings are byte-for-byte unchanged;
* a fallback is disclosed in the label the user actually reads (both ``id`` and
  ``title`` of a Stremio subtitle entry are that label);
* the requested candidate identity -- ``sub_id``, ``release_name``, provider,
  URL -- is never rewritten;
* the note cannot leak between candidates, between episodes, or after a later
  successful request.
"""

import json
from unittest.mock import AsyncMock

import pytest

import app.main as main
from app.cache import cache_manager


@pytest.fixture
def store(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    (root / "_meta").mkdir(parents=True)
    monkeypatch.setattr(cache_manager, "cache_dir", root, raising=False)
    monkeypatch.setattr(cache_manager, "meta_dir", root / "_meta", raising=False)
    return cache_manager


REQUESTED = "The.Last.of.Us.S01E01.WEBRip.x264-ION10.srt"
SERVED = "The.Last.of.Us.S01.1080p.HMAX.WEB-DL.DDP5.1.x264-NTb.srt"


class TestFallbackDisclosure:
    def test_no_fallback_label_is_unchanged(self, store):
        """TEST A: the normal case must be byte-for-byte identical."""
        store.store_metadata("aaa1", {"sub_id": "aaa1", "release_name": REQUESTED})
        label = "[95%] [SubDL] The.Last.of.Us.S01E01.WEBRip.x264-ION10 (by ali talal)"
        assert main._with_fallback_disclosure(label, "aaa1") == label

    def test_missing_metadata_leaves_label_unchanged(self, store):
        label = "[95%] [SubDL] Something (by someone)"
        assert main._with_fallback_disclosure(label, "never-stored") == label

    def test_compatible_release_is_disclosed(self, store):
        """TEST B: a differing served release must be visible in the label."""
        store.store_metadata(
            "bbb2",
            {"sub_id": "bbb2", "release_name": REQUESTED, "served_release_name": SERVED},
        )
        label = "[95%] [SubDL] The.Last.of.Us.S01E01.WEBRip.x264-ION10 (by ali talal)"
        out = main._with_fallback_disclosure(label, "bbb2")

        assert out != label
        assert "fallback" in out.lower()
        # The requested release stays in front; the served one is disclosed.
        assert out.startswith(label)
        assert SERVED in out

    def test_same_release_is_not_disclosed(self, store):
        """A recorded-but-identical release must not add noise."""
        store.store_metadata(
            "ccc3", {"sub_id": "ccc3", "release_name": REQUESTED, "served_release_name": REQUESTED}
        )
        label = "[95%] [SubDL] Anything"
        assert main._with_fallback_disclosure(label, "ccc3") == label

    def test_disclosure_is_stable_across_repeats(self, store):
        """TEST C: reading twice must not mutate the label or the metadata."""
        store.store_metadata(
            "ddd4",
            {"sub_id": "ddd4", "release_name": REQUESTED, "served_release_name": SERVED},
        )
        label = "[95%] [SubDL] The.Last.of.Us.S01E01.WEBRip.x264-ION10 (by ali talal)"
        first = main._with_fallback_disclosure(label, "ddd4")
        second = main._with_fallback_disclosure(label, "ddd4")
        assert first == second

    def test_disclosure_does_not_leak_to_another_candidate(self, store):
        """TEST D/E: one candidate's fallback must not annotate another."""
        store.store_metadata(
            "eee5",
            {"sub_id": "eee5", "release_name": REQUESTED, "served_release_name": SERVED},
        )
        store.store_metadata("fff6", {"sub_id": "fff6", "release_name": SERVED})
        label = "[95%] [SubDL] Another.Candidate.srt"

        assert main._with_fallback_disclosure(label, "fff6") == label
        assert "fallback" in main._with_fallback_disclosure(label, "eee5").lower()

    def test_no_stale_disclosure_after_a_later_exact_request(self, store):
        """TEST F: a fresh successful request must clear the note."""
        sub_id = "ggg7"
        store.store_metadata(
            sub_id, {"sub_id": sub_id, "release_name": REQUESTED, "served_release_name": SERVED}
        )
        label = "[95%] [SubDL] The.Last.of.Us.S01E01.WEBRip.x264-ION10 (by ali talal)"
        assert "fallback" in main._with_fallback_disclosure(label, sub_id).lower()

        # The provider later serves the requested release itself.
        store.store_metadata(sub_id, {"sub_id": sub_id, "release_name": REQUESTED})
        assert main._with_fallback_disclosure(label, sub_id) == label

    def test_disclosure_survives_sanitization(self, store):
        """The note must not be stripped or mangled by credential sanitization."""
        store.store_metadata(
            "hhh8",
            {
                "sub_id": "hhh8",
                "release_name": REQUESTED,
                "served_release_name": SERVED,
                "subdl_key": "FAKE-SECRET-0001",
            },
        )
        stored = json.loads(store.get_meta_path("hhh8").read_text(encoding="utf-8"))
        assert stored["served_release_name"] == SERVED
        assert "FAKE-SECRET-0001" not in json.dumps(stored)
        label = "[95%] [SubDL] The.Last.of.Us.S01E01.WEBRip.x264-ION10 (by ali talal)"
        assert SERVED in main._with_fallback_disclosure(label, "hhh8")


class TestListingDisclosureEndToEnd:
    """Exercise the real listing seam, not the helper.

    Calling ``_with_fallback_disclosure`` directly would pass even if the
    listing stopped calling it, so these tests drive the actual subtitles
    endpoint and assert on the ``SubtitleItem`` the Stremio client receives.
    """

    @pytest.fixture(autouse=True)
    def _http_client(self, monkeypatch):
        import httpx

        monkeypatch.setattr(main, "_http_client", httpx.AsyncClient(), raising=False)

    @staticmethod
    def _release(name="Dune Part Two 2024 UHD Remux Arabic.srt"):
        from app.models import SubtitleRelease

        return SubtitleRelease(
            release_name=name,
            download_url="https://provider.test/m1.srt",
            provider="subdl",
            lang="ara",
        )

    @staticmethod
    def _sub_id(release) -> str:
        import hashlib

        key = f"{release.provider}:{release.release_name}:{release.download_url}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

    def _listing(self, client, store, served_release):
        from unittest.mock import AsyncMock, patch

        rel = self._release()
        sid = self._sub_id(rel)
        meta = {"sub_id": sid, "provider": "subdl", "release_name": rel.release_name,
                "download_url": rel.download_url}
        if served_release:
            meta["served_release_name"] = served_release
        # Do not clobber an existing record: the production path preserves
        # recorded provenance across search-time metadata rewrites, and a
        # lifecycle test depends on that.
        if store.get_metadata(sid) is None:
            store.store_metadata(sid, meta)

        with (
            patch("app.providers.subdl.SubdlProvider.search_subtitles",
                  new=AsyncMock(return_value=[rel])),
            patch("app.providers.subsource.SubsourceProvider.search_subtitles",
                  new=AsyncMock(return_value=[])),
            patch("app.providers.opensubtitles.OpenSubtitlesProvider.search_subtitles",
                  new=AsyncMock(return_value=[])),
            patch("app.providers.cinemeta.CinemetaClient.get_metadata",
                  new=AsyncMock(return_value=None)),
        ):
            resp = client.get(
                "/subtitles/movie/tt15239678/filename=Dune.Part.Two.2024.mkv.json?nocache=1"
            )
        assert resp.status_code == 200
        items = resp.json()["subtitles"]
        assert items, "listing produced no subtitles"
        return items[0]

    def test_full_fallback_lifecycle_through_the_listing_endpoint(self, client, store):
        """The whole chain, without touching a provider.

        listing for A -> fallback B selected -> provenance recorded -> listing
        discloses B -> cache reuse rewrites the record -> listing still
        discloses B, with A's identity and URL untouched.
        """

        rel = self._release()
        sid = self._sub_id(rel)
        url_before = None

        # 1. Initial listing, before any fallback: no disclosure.
        first = self._listing(client, store, None)
        url_before = first["url"]
        assert "fallback" not in first["title"].lower()

        # 2/3. A compatible fallback B is selected and recorded, exactly as the
        # download path does.
        stored = store.get_metadata(sid)
        stored["served_release_name"] = SERVED
        store.store_metadata(sid, stored)

        # 4. The listing now discloses B.
        second = self._listing(client, store, None)
        assert SERVED in second["title"]
        assert second["title"].startswith(first["title"])
        assert second["url"] == url_before, "the subtitle URL must not change"
        assert second["id"] == second["title"]

        # 5/6. A reuse request rewrites the same record the way the reuse
        # branch does.
        stored = store.get_metadata(sid)
        stored["provider"] = "subsource"
        stored["fallback_sync_cache_reuse"] = True
        store.store_metadata(sid, stored)

        # 7. The disclosure survives the reuse.
        third = self._listing(client, store, None)
        assert third["title"] == second["title"], "reuse lost or altered the disclosure"
        assert third["url"] == url_before
        assert store.get_metadata(sid)["release_name"] == rel.release_name, (
            "the requested candidate identity must survive"
        )

    def test_same_release_fallback_adds_no_disclosure(self, client, store):
        """C: requested == served must not produce a misleading note."""
        rel = self._release()
        sid = self._sub_id(rel)
        store.store_metadata(
            sid,
            {"sub_id": sid, "provider": "subdl", "release_name": rel.release_name,
             "download_url": rel.download_url,
             "served_release_name": rel.release_name},
        )
        item = self._listing(client, store, None)
        assert "fallback" not in item["title"].lower()

    def test_listing_discloses_the_served_fallback_release(self, client, store):
        served = "Dune Part Two 2024 1080p WEB-DL Arabic.srt"
        item = self._listing(client, store, served)

        assert "fallback" in item["title"].lower(), (
            "the listing still attributes another release's bytes to the "
            f"requested candidate: {item['title']!r}"
        )
        assert served in item["title"]
        # The id is the same label some clients render directly.
        assert item["id"] == item["title"]
        # Identity and URL are untouched.
        assert "/sub/" in item["url"]

    def test_listing_is_unchanged_without_a_fallback(self, client, store):
        item = self._listing(client, store, None)

        assert "fallback" not in item["title"].lower()
        assert "Dune Part Two 2024 UHD Remux Arabic" in item["title"]


class TestSyncCacheReuseDisclosure:
    """The reuse sibling path must carry the same disclosure.

    ``_reusable_verified_fallback_sync`` returns previously verified fallback
    bytes without downloading. Those bytes were verified for a specific
    release, which may not be the release the user asked for, so the disclosure
    recorded on the first request has to survive the reuse.
    """

    class _Rel:
        provider = "subsource"
        release_name = REQUESTED
        download_url = "https://subsource.test/a.srt"

    @pytest.fixture(autouse=True)
    def _env(self, monkeypatch):
        import httpx

        from app.config import settings

        monkeypatch.setattr(settings, "SUBSOURCE_API_KEY", "FAKE-SUBSOURCE-KEY", raising=False)
        monkeypatch.setattr(main, "_http_client", httpx.AsyncClient(), raising=False)

    async def _reuse(self, monkeypatch, meta):
        """Drive the caller's reuse branch with the download short-circuited."""
        from app.providers.subsource import SubsourceProvider

        rel = self._Rel()
        monkeypatch.setattr(
            SubsourceProvider, "search_subtitles", AsyncMock(return_value=[rel])
        )
        monkeypatch.setattr(
            main, "_reusable_verified_fallback_sync", AsyncMock(return_value=b"REUSED")
        )
        monkeypatch.setattr(
            main, "_fallback_candidate_ref", staticmethod(lambda *a, **k: "ref123")
        )
        outcome: dict = {}
        data = await main._fallback_download_subsource(
            imdb_id="tt1",
            media_type="series",
            season=1,
            episode=1,
            subsource_key="FAKE-SUBSOURCE-KEY",
            target_filename="Show.S01E01.1080p.BluRay.mkv",
            meta=meta,
            outcome=outcome,
        )
        return data, outcome

    @pytest.mark.asyncio
    async def test_reuse_carries_recorded_provenance(self, monkeypatch):
        """B: a compatible fallback stays disclosed when its bytes are reused."""
        data, outcome = await self._reuse(
            monkeypatch, {"release_name": REQUESTED, "served_release_name": SERVED}
        )
        assert data == b"REUSED"
        assert outcome["category"] == "SYNC_CACHE_REUSE"
        assert outcome["served_release_name"] == SERVED, (
            "reuse dropped the fallback disclosure"
        )

    @pytest.mark.asyncio
    async def test_reuse_without_provenance_invents_nothing(self, monkeypatch):
        """D: legacy metadata with no provenance must not fabricate a release."""
        data, outcome = await self._reuse(monkeypatch, {"release_name": REQUESTED})
        assert data == b"REUSED"
        assert outcome["served_release_name"] == ""

    @pytest.mark.asyncio
    async def test_label_stays_clean_when_reuse_has_no_provenance(self, monkeypatch, store):
        """D: no recorded release means no disclosure, not a wrong one."""
        store.store_metadata("reuse1", {"sub_id": "reuse1", "release_name": REQUESTED})
        label = "[95%] [SubDL] The.Last.of.Us.S01E01.WEBRip.x264-ION10 (by ali talal)"
        assert main._with_fallback_disclosure(label, "reuse1") == label

    @pytest.mark.asyncio
    async def test_disclosure_survives_a_reuse_style_metadata_rewrite(self, monkeypatch, store):
        """E/F: provenance survives the writes a reuse request performs."""
        sub_id = "reuse2"
        store.store_metadata(
            sub_id, {"sub_id": sub_id, "release_name": REQUESTED, "served_release_name": SERVED}
        )
        # A reuse request rewrites provider/provenance flags on the same record.
        stored = store.get_metadata(sub_id)
        stored["provider"] = "subsource"
        stored["fallback_sync_cache_reuse"] = True
        store.store_metadata(sub_id, stored)

        label = "[95%] [SubDL] The.Last.of.Us.S01E01.WEBRip.x264-ION10 (by ali talal)"
        out = main._with_fallback_disclosure(label, sub_id)
        assert SERVED in out
        assert store.get_metadata(sub_id)["served_release_name"] == SERVED

    def test_reuse_provenance_is_candidate_scoped(self, store):
        """E: candidate A's disclosure must never annotate candidate B."""
        store.store_metadata(
            "reuseA", {"sub_id": "reuseA", "release_name": REQUESTED, "served_release_name": SERVED}
        )
        store.store_metadata("reuseB", {"sub_id": "reuseB", "release_name": SERVED})
        label = "[95%] [SubDL] Other.Candidate.srt"
        assert "fallback" not in main._with_fallback_disclosure(label, "reuseB").lower()
        assert "fallback" in main._with_fallback_disclosure(label, "reuseA").lower()


class TestRequestedIdentityIsPreserved:
    def test_release_name_is_never_rewritten_by_the_fallback(self, store):
        """The requested identity stays intact: it drives sub_id and the URL."""
        meta = {"sub_id": "iii9", "provider": "subdl", "release_name": REQUESTED,
                "download_url": "https://dl.example.invalid/x.zip"}
        store.store_metadata("iii9", dict(meta))
        # Simulate the caller recording provenance.
        stored = store.get_metadata("iii9")
        stored["served_release_name"] = SERVED
        store.store_metadata("iii9", stored)

        after = store.get_metadata("iii9")
        assert after["release_name"] == REQUESTED, "requested identity must not be replaced"
        assert after["sub_id"] == "iii9"
        assert after["provider"] == "subdl"
        assert after["download_url"] == "https://dl.example.invalid/x.zip"
        assert after["served_release_name"] == SERVED

    def test_sub_id_derivation_is_unaffected(self, store):
        """sub_id is sha256(provider:release_name:download_url); disclosure adds no input."""
        import hashlib

        provider, release, url = "subdl", REQUESTED, "https://dl.example.invalid/x.zip"
        expected = hashlib.sha256(f"{provider}:{release}:{url}".encode()).hexdigest()[:16]
        store.store_metadata(
            expected,
            {"sub_id": expected, "provider": provider, "release_name": release,
             "download_url": url, "served_release_name": SERVED},
        )
        assert store.get_metadata(expected)["sub_id"] == expected
