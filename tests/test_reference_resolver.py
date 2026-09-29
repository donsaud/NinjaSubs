"""Tests for the cached reference resolver (SubDL, SubSource)."""

import asyncio

import pytest

from app.models import SubtitleRelease
from app.services.reference_resolver import (
    DualReferenceResolver,
    ReferenceDiskCache,
    ReferenceQuery,
    _title_from_filename,
)
from app.services.sync.external_strategy import (
    TIER_EXACT_GROUP,
    TIER_FALLBACK,
    TIER_SOURCE_EDITION,
    reference_tier,
)
from app.services.sync.query import ResolvedReference


@pytest.fixture(autouse=True)
def _isolated_reference_disk_cache(monkeypatch, tmp_path):
    """Keep resolver tests hermetic by redirecting the disk cache to tmp_path."""
    original_init = DualReferenceResolver.__init__

    def __init__(self, *args, cache=None, **kwargs):
        if cache is None:
            cache = ReferenceDiskCache(root=tmp_path / "references", ttl=3600.0)
        original_init(self, *args, cache=cache, **kwargs)

    monkeypatch.setattr(DualReferenceResolver, "__init__", __init__)


class _FakeProvider:
    def __init__(self, content: bytes | None, delay: float = 0.0, name: str = "fake"):
        self._content = content
        self._delay = delay
        self.name = name
        self.downloaded = 0

    async def search_subtitles(self, **kwargs):
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._content is None:
            return []
        return [
            SubtitleRelease(
                release_name=kwargs.get("target_filename") or "ref.en.srt",
                download_url=f"http://{self.name}/ref",
                provider=self.name,
                lang="eng",
            )
        ]

    async def download_archive(self, download_ref, api_key=None):
        self.downloaded += 1
        return self._content


def _query():
    # Fully-specified target (source + codec + res + group) so the strict
    # tree confirms the same-edition fake reference.
    return ReferenceQuery(imdb_id="tt1234567", target_filename="Movie.2024.1080p.WEB-DL.x264-GRP.srt")


BIG = ("1\n00:00:01,000 --> 00:00:02,000\n" + ("x" * 6000) + "\n").encode()


@pytest.mark.asyncio
async def test_resolve_returns_first_valid_reference():
    resolver = DualReferenceResolver(
        subdl_provider=_FakeProvider(BIG, name="subdl"),
        subsource_provider=_FakeProvider(None, name="subsource"),
    )
    content = await resolver.resolve(_query())
    assert content is not None
    assert len(content.encode("utf-8")) > 5120


@pytest.mark.asyncio
async def test_resolve_rejects_small_references():
    small = b"1\n00:00:01,000 --> 00:00:02,000\nshort\n"
    resolver = DualReferenceResolver(
        subdl_provider=_FakeProvider(small, name="subdl"),
        subsource_provider=_FakeProvider(small, name="subsource"),
    )
    assert await resolver.resolve(_query()) is None


@pytest.mark.asyncio
async def test_resolve_times_out_without_hanging():
    slow = _FakeProvider(BIG, delay=5.0, name="slow")
    resolver = DualReferenceResolver(
        subdl_provider=slow,
        subsource_provider=_FakeProvider(None, delay=5.0, name="slow2"),
        timeout=0.2,
    )
    start = asyncio.get_event_loop().time()
    content = await resolver.resolve(_query())
    elapsed = asyncio.get_event_loop().time() - start
    assert content is None
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_resolve_without_providers_returns_none():
    assert await DualReferenceResolver().resolve(_query()) is None


def test_reference_query_is_series():
    assert ReferenceQuery(imdb_id="tt1").is_series is False
    assert ReferenceQuery(imdb_id="tt1", media_type="series").is_series is True
    assert ReferenceQuery(imdb_id="tt1", season=1, episode=2).is_series is True


@pytest.mark.asyncio
async def test_opaque_filename_falls_back_to_episode_match():
    """Arbitrary debrid names (wVwm.mkv) must not block episode-level matching."""

    class _Capture:
        def __init__(self):
            self.calls: list = []

        async def search_subtitles(self, **kwargs):
            self.calls.append(kwargs)
            return []

        async def download_archive(self, url, api_key=None):  # pragma: no cover
            return None

    capture = _Capture()
    resolver = DualReferenceResolver(subdl_provider=capture, subsource_provider=None, timeout=0.5)
    await resolver.resolve(
        ReferenceQuery(imdb_id="tt0773262", target_filename="wVwm.mkv", season=8, episode=2)
    )
    first = capture.calls[0]
    assert first["target_filename"] is None
    assert first["imdb_id"] == "tt0773262"
    assert first["season"] == 8 and first["episode"] == 2


def _zip_with_srt(member: str, text: str) -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr(member, text)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_zip_reference_is_extracted_and_selected():
    """SubDL returns a ZIP; the resolver must extract the inner SRT."""
    srt = "1\n00:00:01,000 --> 00:00:02,000\n" + ("reference line\n" * 600) + "\n"
    zbytes = _zip_with_srt("movie.en.srt", srt)

    resolver = DualReferenceResolver(
        subdl_provider=_FakeProvider(zbytes, name="subdl"),
        subsource_provider=None,
    )
    content = await resolver.resolve(_query())
    assert content is not None
    assert "reference line" in content


def test_select_zip_member_prefers_requested_episode():
    """Season-pack ZIPs: pick the requested episode, ignoring 00/intro files."""
    from app.services.sync.matching import _member_episode_number

    resolver = DualReferenceResolver()
    members = [
        ("00 Dexter, SEASON 8 1080p x265 [23.976 FPS].srt", 296),
        ("Dexter S08E01 1080p.srt", 9000),
        ("Dexter S08E02 1080p.srt", 8000),
    ]
    # Episode 2 requested -> E02, not the 296-byte intro nor the largest file.
    assert resolver._select_zip_member(members, 8, 2) == "Dexter S08E02 1080p.srt"
    # No episode requested -> largest usable .srt.
    assert resolver._select_zip_member(members, 8, None) == "Dexter S08E01 1080p.srt"
    # Opaque names -> largest wins.
    assert resolver._select_zip_member([("a.srt", 6000), ("b.srt", 9000)], None, None) == "b.srt"
    # Bare episode numbers are recognised.
    assert resolver._select_zip_member(
        [("Dexter 2.srt", 7000), ("Dexter 1.srt", 8000)], 8, 2
    ) == "Dexter 2.srt"

    # Resolution/codec tokens are not mistaken for episode numbers.
    assert _member_episode_number("Show.1080p.WEB-DL.srt") is None
    assert _member_episode_number("Show.x265-GROUP.srt") is None
    assert _member_episode_number("Show.EP02.srt") == 2
    assert _member_episode_number("Show Episode 2.srt") == 2
    assert _member_episode_number("Show.S08E02.srt") == 2


@pytest.mark.asyncio
async def test_resolver_rejects_wrong_season_release():
    """A S01 candidate must never be used for a S08 request."""

    class _P:
        def __init__(self):
            self.downloaded: list[str] = []

        async def search_subtitles(self, **kwargs):
            return [
                SubtitleRelease(
                    release_name="Dexter.S01.2006.BluRay.x265-GRP.srt",
                    download_url="http://s01", provider="subdl", lang="eng",
                ),
                SubtitleRelease(
                    release_name="Dexter.S08.2006.BluRay.x265-GRP.srt",
                    download_url="http://s08", provider="subdl", lang="eng",
                ),
            ]

        async def download_archive(self, url, api_key=None):
            self.downloaded.append(url)
            return BIG

    provider = _P()
    resolver = DualReferenceResolver(subdl_provider=provider, subsource_provider=None, timeout=0.5)
    query = ReferenceQuery(imdb_id="tt0773262", media_type="series", season=8, episode=1)
    # The season-matched S08 pack is accepted (and sliced by member); S01 is not.
    content = await resolver.resolve(query)
    assert content is not None
    assert provider.downloaded == ["http://s08"]


def test_select_zip_member_rejects_wrong_season():
    resolver = DualReferenceResolver()
    members = [("Dexter S01E01.srt", 9000), ("Dexter S08E01.srt", 8000)]
    assert resolver._select_zip_member(members, 8, 1) == "Dexter S08E01.srt"
    # Only wrong-season members -> reject entirely (no cross-season fallback).
    assert resolver._select_zip_member([("Dexter S01E01.srt", 9000)], 8, 1) is None


@pytest.mark.asyncio
async def test_resolver_uses_top_matching_reference():
    """The first season/episode-matched English candidate is used as-is."""
    class _P:
        def __init__(self):
            self.downloaded: list[str] = []

        async def search_subtitles(self, **kwargs):
            return [
                SubtitleRelease(
                    release_name="Dexter.S08E01.1080p.BluRay.x264-GRP.srt",
                    download_url="http://bluray", provider="subdl", lang="eng",
                ),
                SubtitleRelease(
                    release_name="Dexter.S08E01.480p.HDTV.x264-mSD.srt",
                    download_url="http://hdtv", provider="subdl", lang="eng",
                ),
            ]

        async def download_archive(self, url, api_key=None):
            self.downloaded.append(url)
            return BIG

    provider = _P()
    resolver = DualReferenceResolver(subdl_provider=provider, subsource_provider=None, timeout=0.5)
    content = await resolver.resolve(
        ReferenceQuery(
            imdb_id="tt0773262",
            media_type="series",
            season=8,
            episode=1,
            target_filename="Dexter.S08E01.1080p.BluRay.x265-GRP.mkv",
        )
    )
    assert content is not None
    assert provider.downloaded == ["http://bluray"]


def test_season_number_parses_worded_seasons():
    from app.services.sync.matching import _season_number

    assert _season_number("Dexter Season One.srt") == 1
    assert _season_number("Dexter Season 8.srt") == 8
    assert _season_number("Dexter Season Eight.srt") == 8
    assert _season_number("The First Season.srt") == 1
    assert _season_number("Dexter.S08E01.srt") == 8
    assert _season_number("Dexter.8x01.srt") == 8
    assert _season_number("Dexter.S08.2006.BluRay.srt") == 8
    assert _season_number("Some.Movie.2024.1080p.srt") is None


@pytest.mark.asyncio
async def test_resolver_rejects_worded_wrong_season_candidate():
    """`Dexter Season One.srt` must be rejected for an S08 request."""
    captured: dict = {}

    class _P:
        async def search_subtitles(self, **kwargs):
            return [
                SubtitleRelease(
                    release_name="Dexter Season One.srt",
                    download_url="http://s1", provider="subsource", lang="eng",
                )
            ]

        async def download_archive(self, url, api_key=None):  # pragma: no cover
            captured["downloaded"] = url
            return BIG

    resolver = DualReferenceResolver(subsource_provider=_P(), subdl_provider=None, timeout=0.5)
    content = await resolver.resolve(
        ReferenceQuery(imdb_id="tt0773262", media_type="series", season=8, episode=1)
    )
    assert content is None
    assert "downloaded" not in captured


@pytest.mark.asyncio
async def test_fast_failure_does_not_cancel_slow_provider():
    """A 429/None from one provider must not cancel the slower usable provider."""
    slow = _FakeProvider(BIG, delay=0.2, name="subsource")
    fast_fail = _FakeProvider(None, delay=0.0, name="subdl")
    resolver = DualReferenceResolver(
        subdl_provider=fast_fail, subsource_provider=slow, timeout=1.5
    )
    content = await resolver.resolve(_query())
    assert content is not None
    assert "reference line" in content or len(content.encode()) > 5120


@pytest.mark.asyncio
async def test_resolver_logs_tiered_candidates(caplog):
    class _P:
        async def search_subtitles(self, **kwargs):
            return [
                SubtitleRelease(
                    release_name="Dexter.S08E01.480p.HDTV.x264-mSD.srt",
                    download_url="http://hdtv", provider="subdl", lang="eng",
                ),
                SubtitleRelease(
                    release_name="Dexter.S08E01.1080p.BluRay.x264-GRP.srt",
                    download_url="http://bluray", provider="subdl", lang="eng",
                ),
            ]

        async def download_archive(self, url, api_key=None):
            return BIG

    resolver = DualReferenceResolver(subdl_provider=_P(), subsource_provider=None, timeout=0.5)
    with caplog.at_level("INFO"):
        await resolver.resolve(
            ReferenceQuery(
                imdb_id="tt0773262", media_type="series", season=8, episode=1,
                target_filename="Dexter.S08E01.1080p.BluRay.x265-GRP.mkv",
            )
        )
    assert "candidate tier=" in caplog.text
    assert "selected reference" in caplog.text


@pytest.mark.asyncio
async def test_resolver_rejects_untagged_candidates_for_specific_season():
    """SubSource-style untagged names (dexter.107) must not be used for S08E01."""
    provider = _FakeProvider(BIG, name="subsource")

    async def _untagged(**kwargs):
        return [
            SubtitleRelease(
                release_name="dexter.107.hdtv-lol.srt",
                download_url="http://subsource/107",
                provider="subsource",
                lang="eng",
            ),
            SubtitleRelease(
                release_name="DexterFirst Season 2006.srt",
                download_url="http://subsource/516952",
                provider="subsource",
                lang="eng",
            ),
        ]

    provider.search_subtitles = _untagged  # type: ignore[method-assign]
    resolver = DualReferenceResolver(subsource_provider=provider, subdl_provider=None, timeout=0.5)
    content = await resolver.resolve(
        ReferenceQuery(imdb_id="tt0773262", media_type="series", season=8, episode=1)
    )
    assert content is None
    assert provider.downloaded == 0


@pytest.mark.asyncio
async def test_resolver_season_filter_keeps_tagged_candidate(monkeypatch):
    """A season-tagged S08 candidate is accepted; untagged peers are dropped."""

    class _P:
        def __init__(self):
            self.downloaded: list[str] = []

        async def search_subtitles(self, **kwargs):
            return [
                SubtitleRelease(
                    release_name="dexter.107.hdtv-lol.srt",
                    download_url="http://junk",
                    provider="subsource",
                    lang="eng",
                ),
                SubtitleRelease(
                    release_name="Dexter.S08E01.1080p.WEB-DL.srt",
                    download_url="http://good",
                    provider="subsource",
                    lang="eng",
                ),
            ]

        async def download_archive(self, url, api_key=None):
            self.downloaded.append(url)
            assert url == "http://good"
            return BIG

    provider = _P()
    resolver = DualReferenceResolver(subsource_provider=provider, subdl_provider=None, timeout=0.5)
    query = ReferenceQuery(imdb_id="tt0773262", media_type="series", season=8, episode=1)
    # The tagged S08E01 WEB-DL reference is accepted; untagged peers are dropped.
    assert await resolver.resolve(query) is not None
    assert provider.downloaded == ["http://good"]


def test_title_from_filename_derives_keywords():
    assert _title_from_filename("La Casa de Papel S01E01-E013....720P.srt") == "La Casa de Papel"
    assert _title_from_filename("Dexter.2006.S08E01.1080p.BluRay.x264.srt") == "Dexter"
    assert _title_from_filename("Money.Heist.S01E01.srt") == "Money Heist"
    assert _title_from_filename("Movie.2024.1080p.WEB-DL.srt") == "Movie"
    assert _title_from_filename("wVwm.mkv") is None
    assert _title_from_filename(None) is None


def _valid_reference_text() -> str:
    block = "1\n00:00:01,000 --> 00:00:02,000\nreference line\n"
    return block + ("reference line\n" * 600)


def test_reference_disk_cache_roundtrip_and_source_replacement(tmp_path):
    query = ReferenceQuery(imdb_id="tt0773262", media_type="series", season=8, episode=1)
    cache = ReferenceDiskCache(root=tmp_path / "disk", ttl=3600.0, min_bytes=100)

    first = _valid_reference_text()
    cache.set(query, "subdl", first)
    expected = tmp_path / "disk" / f"{query.cache_stem}_subdl_edition.srt"
    assert expected.exists()
    assert cache.get(query) == ResolvedReference(first, "edition", False)

    # Saving from another provider replaces the stale source file for the same episode.
    second = first.replace("reference line", "subsource line")
    cache.set(query, "subsource", second)
    assert list((tmp_path / "disk").glob("*.srt")) == [
        tmp_path / "disk" / f"{query.cache_stem}_subsource_edition.srt"
    ]
    assert cache.get(query) == ResolvedReference(second, "edition", False)


def test_reference_disk_cache_preserves_decision_kind(tmp_path):
    """A cached team verdict must survive recovery (no edition downgrade)."""
    query = ReferenceQuery(
        imdb_id="tt0758758",
        media_type="movie",
        target_filename="Into.The.Wild.2007.1080p.BluRay.x264-FSiHD",
    )
    cache = ReferenceDiskCache(root=tmp_path / "disk", ttl=3600.0, min_bytes=100)
    text = _valid_reference_text()

    cache.set(query, "subdl", text, kind="team")
    assert (tmp_path / "disk" / f"{query.cache_stem}_subdl_team.srt").exists()
    assert cache.get(query) == ResolvedReference(text, "team", False)

    # Legacy group-only entries cannot prove the playing video's identity.
    (tmp_path / "disk" / f"{query.cache_stem}_subdl_team.srt").unlink()
    legacy = tmp_path / "disk" / "tt0758758_movie_x_FSiHD_subdl.srt"
    legacy.write_text(text, encoding="utf-8")
    assert cache.get(query) is None
    legacy.unlink()

    # ... but an explicit edition verdict is never upgraded.
    explicit = tmp_path / "disk" / "tt0758758_movie_x_FSiHD_subdl_edition.srt"
    explicit.write_text(text, encoding="utf-8")
    assert cache.get(query) is None
    explicit.unlink()

    # ... and a group-less query still fails closed to "edition".
    unknown_query = ReferenceQuery(imdb_id="tt0758758", media_type="movie")
    assert unknown_query.effective_group == "unknown"
    unknown_legacy = tmp_path / "disk" / "tt0758758_movie_x_unknown_subdl.srt"
    unknown_legacy.write_text(text, encoding="utf-8")
    assert cache.get(unknown_query) is None
    unknown_legacy.unlink()

    # An edition save must never evict an exact-kind entry for the same stem.
    cache.set(query, "subsource", text, kind="team")
    cache.set(query, "subdl", text, kind="edition")
    names = sorted(p.name for p in (tmp_path / "disk").glob("*.srt"))
    assert names == [
        f"{query.cache_stem}_subdl_edition.srt",
        f"{query.cache_stem}_subsource_team.srt",
    ]
    assert cache.get(query).kind == "team"


def test_reference_disk_cache_rejects_invalid_and_expired_entries(tmp_path, caplog):
    query = ReferenceQuery(imdb_id="tt0773262", media_type="series", season=8, episode=1)
    cache = ReferenceDiskCache(root=tmp_path / "disk", ttl=3600.0, min_bytes=100)

    (tmp_path / "disk").mkdir(parents=True, exist_ok=True)
    (tmp_path / "disk" / f"{query.cache_stem}_subdl_edition.srt").write_text(
        "too short", encoding="utf-8"
    )
    (tmp_path / "disk" / f"{query.cache_stem}_subsource_edition.srt").write_text(
        "1\nno timestamps here\nsubtitle\n", encoding="utf-8"
    )
    expired = tmp_path / "disk" / f"{query.cache_stem}_subsource_edition.srt"
    expired.write_text(_valid_reference_text(), encoding="utf-8")
    expired_cache = ReferenceDiskCache(root=tmp_path / "disk", ttl=-1.0, min_bytes=100)
    with caplog.at_level("INFO"):
        assert expired_cache.get(query) is None
    assert not expired.exists()
    assert cache.get(query) is None


@pytest.mark.asyncio
async def test_resolve_returns_disk_cache_without_calling_providers(tmp_path, caplog):
    query = ReferenceQuery(imdb_id="tt0773262", media_type="series", season=8, episode=1)
    text = _valid_reference_text()
    cache = ReferenceDiskCache(root=tmp_path / "disk", ttl=3600.0, min_bytes=100)
    cache.set(query, "subsource", text)

    calls = {"search": 0, "download": 0}

    class _P:
        async def search_subtitles(self, **kwargs):
            calls["search"] += 1
            raise AssertionError("cached resolution must not query providers")

        async def download_archive(self, url, api_key=None):
            calls["download"] += 1
            raise AssertionError("cached resolution must not download providers")

    resolver = DualReferenceResolver(
        subdl_provider=_P(), subsource_provider=_P(), cache=cache
    )
    with caplog.at_level("INFO"):
        assert await resolver.resolve(query) == text
    assert calls == {"search": 0, "download": 0}
    assert "cache hit on disk" in caplog.text


@pytest.mark.asyncio
async def test_cached_reference_keeps_provenance(tmp_path):
    """Regression: a resolve keeps its kind + bluray flag on a disk cache hit."""
    query = ReferenceQuery(
        imdb_id="tt0758758",
        media_type="movie",
        target_filename="Into.The.Wild.2007.1080p.BluRay.x264-FSiHD",
    )
    cache = ReferenceDiskCache(root=tmp_path / "disk", ttl=3600.0, min_bytes=100)
    provider = _FakeProvider(BIG, name="subdl")

    async def _team_search(**kwargs):
        return _eng_release("Into.The.Wild.2007.1080p.BluRay.x264-FSiHD.ENG.srt")

    provider.search_subtitles = _team_search  # type: ignore[method-assign]

    async def _dead_search(**kwargs):
        raise AssertionError("cache hit must not query providers")

    async def _dead_download(url, api_key=None):
        raise AssertionError("cache hit must not download providers")

    first = DualReferenceResolver(subdl_provider=provider, cache=cache, timeout=0.5)
    resolved = await first.resolve_with_provenance(query)
    assert resolved.text is not None and resolved.kind == "edition"
    assert resolved.bluray_match is True

    provider.search_subtitles = _dead_search  # type: ignore[method-assign]
    provider.download_archive = _dead_download  # type: ignore[method-assign]
    second = DualReferenceResolver(subdl_provider=provider, cache=cache, timeout=0.5)
    resolved = await second.resolve_with_provenance(query)
    assert resolved.text is not None and resolved.kind == "edition"
    assert resolved.bluray_match is True


@pytest.mark.asyncio
async def test_resolve_saves_successful_reference_for_future_requests(tmp_path):
    query = ReferenceQuery(
        imdb_id="tt0773262", media_type="series", season=8, episode=1,
        target_filename="Dexter.S08E01.1080p.BluRay.x265-GRP.mkv",
    )
    cache = ReferenceDiskCache(root=tmp_path / "disk", ttl=3600.0, min_bytes=100)
    provider = _FakeProvider(BIG, name="subdl")
    release_name = "Dexter.S08E01.1080p.BluRay.x265-GRP.srt"
    async def _search(**kwargs):
        return _eng_release(release_name)

    provider.search_subtitles = _search  # type: ignore[method-assign]
    resolver = DualReferenceResolver(subdl_provider=provider, cache=cache, timeout=0.5)
    first = await resolver.resolve(query)
    assert first is not None
    assert (tmp_path / "disk" / f"{query.cache_stem}_subdl_edition.srt").exists()

    async def _fail_search(**kwargs):
        raise AssertionError("second resolution should use the disk cache")

    async def _fail_download(url, api_key=None):
        raise AssertionError("second resolution should use the disk cache")

    provider.search_subtitles = _fail_search  # type: ignore[method-assign]
    provider.download_archive = _fail_download  # type: ignore[method-assign]
    second = DualReferenceResolver(subdl_provider=provider, cache=cache, timeout=0.5)
    assert await second.resolve(query) == first
    assert provider.downloaded == 1


def _eng_release(release_name: str) -> list[SubtitleRelease]:
    return [
        SubtitleRelease(
            release_name=release_name,
            download_url="http://reference/ref",
            provider="subdl",
            lang="eng",
        )
    ]


def test_release_group_extraction():
    from app.services.sync.matching import _release_group as g

    assert g("Into.The.Wild.2007.1080p.BluRay.x264-FSiHD") == "FSiHD"
    assert g("Into.The.Wild.2007.1080p.BluRay.x264-FSiHD.ENG.srt") == "FSiHD"
    assert g("Dexter.S08E01.1080p.BluRay.x265-GRP.mkv") == "GRP"
    assert g("Dexter.S08E01.480p.HDTV.x264-mSD.srt") == "mSD"
    assert g("La.casa.de.papel.S01E01.WEBRip.x264-ION10.srt") == "ION10"
    assert g("Dexter (2006) - S08 (1080p BluRay x265 ImE)") == "ImE"
    assert g("DARK.S01.1080p.NF.WEBRip.DD5.1.x264-NTb.srt") == "NTb"
    # Source/codec/resolution markers are not groups.
    assert g("Movie.2024.1080p.WEB-DL.mkv") is None
    assert g("Movie.2024.1080p.WEB-DL.DD5.1.H.264-BS.srt") == "BS"
    assert g("Sopranos 1-11.srt") is None
    assert g("wVwm.mkv") is None
    assert g("The.Last.of.Us.S01.E1-2-3-4-5.srt") is None
    assert g("La Casa de Papel S01E01-E013....720P.srt") is None
    assert g(None) is None
    assert g("") is None


@pytest.mark.asyncio
async def test_resolver_prefers_exact_release_group():
    """End to end: the exact release group scores highest and wins."""

    class _P:
        def __init__(self):
            self.downloaded: list[str] = []

        async def search_subtitles(self, **kwargs):
            return [
                SubtitleRelease(
                    release_name="Into.the.Wild.2007.1080p.Bluray.x265.HEVC.10bit.AAC.5.1.Tigole.srt",
                    download_url="http://tigole",
                    provider="subdl",
                    lang="eng",
                ),
                SubtitleRelease(
                    release_name="Into.The.Wild.2007.1080p.BluRay.x264-FSiHD.ENG.srt",
                    download_url="http://fsihd",
                    provider="subdl",
                    lang="eng",
                ),
            ]

        async def download_archive(self, url, api_key=None):
            self.downloaded.append(url)
            return BIG

    provider = _P()
    resolver = DualReferenceResolver(subdl_provider=provider, subsource_provider=None, timeout=0.5)
    content = await resolver.resolve(
        ReferenceQuery(
            imdb_id="tt0758758",
            media_type="movie",
            target_filename="Into.The.Wild.2007.1080p.BluRay.x264-FSiHD",
        )
    )
    assert content is not None
    assert provider.downloaded == ["http://fsihd"]


def test_cache_stem_pins_stream_edition():
    base = {"imdb_id": "tt0758758", "media_type": "movie"}
    fsihd = ReferenceQuery(
        **base, target_filename="Into.The.Wild.2007.1080p.BluRay.x264-FSiHD"
    )
    yify = ReferenceQuery(
        **base, target_filename="Into.The.Wild.2007.720p.BluRay.x264-YIFY"
    )
    assert fsihd.cache_stem.startswith("tt0758758_movie_x_FSiHD_v2_")
    assert yify.cache_stem.startswith("tt0758758_movie_x_YIFY_v2_")
    assert fsihd.cache_stem != yify.cache_stem
    assert ReferenceQuery(imdb_id="tt1").cache_stem.startswith("tt1_movie_x_unknown_v2_")
    explicit = ReferenceQuery(
        imdb_id="tt1", release_group="PiR8", target_filename="Other-GRP.mkv"
    )
    assert explicit.cache_stem.startswith("tt1_movie_x_PiR8_v2_")


def test_cache_stem_scopes_distinct_subtitle_targets(tmp_path):
    """Two subtitle IDs for one video must not share a cached reference."""
    from app.services.sync.cache import ReferenceDiskCache

    video = {
        "imdb_id": "tt0773262",
        "media_type": "series",
        "season": 8,
        "episode": 5,
        "target_filename": "Dexter.s8e05.1080p.BluRay-PiR8.mkv",
    }
    target_a = ReferenceQuery(**video, target_sub_id="sub-a", target_cue_digest="abc123")
    target_b = ReferenceQuery(**video, target_sub_id="sub-b", target_cue_digest="def456")
    assert target_a.cache_stem != target_b.cache_stem

    cache = ReferenceDiskCache(root=tmp_path / "disk", ttl=3600.0, min_bytes=100)
    cache.set(target_a, "subdl", _valid_reference_text(), kind="edition", candidate="a")
    assert cache.get(target_a) is not None
    assert cache.get(target_b) is None


def test_cache_stem_shares_identical_target_cue_layouts(tmp_path):
    """Identical initial cue timings may reuse a validated reference."""
    from app.services.sync.cache import ReferenceDiskCache

    video = {
        "imdb_id": "tt0773262",
        "media_type": "series",
        "season": 8,
        "episode": 5,
        "target_filename": "Dexter.s8e05.1080p.BluRay-PiR8.mkv",
    }
    first_id = ReferenceQuery(**video, target_sub_id="sub-a", target_cue_digest="abc123")
    second_id = ReferenceQuery(**video, target_sub_id="sub-c", target_cue_digest="abc123")
    assert first_id.cache_stem == second_id.cache_stem

    cache = ReferenceDiskCache(root=tmp_path / "disk", ttl=3600.0, min_bytes=100)
    cache.set(first_id, "subdl", _valid_reference_text(), kind="edition", candidate="a")
    assert cache.get(second_id) is not None


def test_target_cue_digest_ignores_text_but_not_timing():
    """The layout fingerprint is timing-only and format-insensitive."""
    from app.services.sync.query import fingerprint_target_cues

    srt = (
        "1\n00:10:09,500 --> 00:10:11,000\nFirst line\n\n"
        "2\n00:10:12,000 --> 00:10:13,500\nSecond line\n"
    )
    same_timing = srt.replace("First line", "سطر أول").replace("Second line", "سطر ثان")
    vtt_timing = srt.replace(",", ".")
    shifted_timing = srt.replace("00:10:09,500", "00:01:46,000", 1)

    assert fingerprint_target_cues(srt) == fingerprint_target_cues(same_timing)
    assert fingerprint_target_cues(srt) == fingerprint_target_cues(vtt_timing)
    assert fingerprint_target_cues(srt) == fingerprint_target_cues(srt.encode())
    assert fingerprint_target_cues(srt) != fingerprint_target_cues(shifted_timing)
    assert fingerprint_target_cues(b"not a subtitle") is None


@pytest.mark.asyncio
async def test_different_editions_do_not_share_cached_reference(tmp_path):
    """FSiHD and YIFY releases of one movie must resolve/cache independently."""
    fsihd_query = ReferenceQuery(
        imdb_id="tt0758758",
        media_type="movie",
        target_filename="Into.The.Wild.2007.1080p.BluRay.x264-FSiHD",
    )
    yify_query = ReferenceQuery(
        imdb_id="tt0758758",
        media_type="movie",
        target_filename="Into.The.Wild.2007.720p.BluRay.x264-YIFY",
    )
    cache = ReferenceDiskCache(root=tmp_path / "disk", ttl=3600.0, min_bytes=100)

    fsihd_provider = _FakeProvider(BIG, name="subdl")
    resolver = DualReferenceResolver(
        subdl_provider=fsihd_provider, cache=cache, timeout=0.5
    )
    assert await resolver.resolve(fsihd_query) is not None
    assert (tmp_path / "disk" / f"{fsihd_query.cache_stem}_subdl_edition.srt").exists()

    # The YIFY edition must re-resolve instead of reusing the FSiHD timings.
    yify_provider = _FakeProvider(BIG, name="subdl")
    resolver = DualReferenceResolver(
        subdl_provider=yify_provider, cache=cache, timeout=0.5
    )
    assert await resolver.resolve(yify_query) is not None
    assert yify_provider.downloaded == 1
    assert (tmp_path / "disk" / f"{yify_query.cache_stem}_subdl_edition.srt").exists()


def test_edition_tags_extraction():
    from app.services.sync.matching import _edition_tags, is_retail_disc_source

    assert _edition_tags("Movie.EXTENDED.1080p.BluRay.x264-GRP") == frozenset({"extended"})
    assert _edition_tags("Movie.US.1080p.BluRay.x264-GRP") == frozenset({"us"})
    assert _edition_tags("Movie.1080p.BluRay.x264-GRP") == frozenset()
    # REMUX is an encoding method, not a cut marker.
    assert _edition_tags("Movie.2160p.BluRay.REMUX.HEVC-GRP") == frozenset()
    assert _edition_tags(None) == frozenset()
    # Expanded cut/edit markers that also change timing or framing.
    assert _edition_tags("Movie.IMAX.2160p.WEB-DL-GRP") == frozenset({"imax"})
    assert _edition_tags("Movie.Remastered.1080p.BluRay-GRP") == frozenset({"remastered"})
    assert _edition_tags("Movie.Criterion.1080p.BluRay-GRP") == frozenset({"criterion"})
    assert _edition_tags("Movie.Hybrid.1080p.BluRay-GRP") == frozenset({"hybrid"})
    # Multi-word markers match as phrases, never on the bare generic word.
    assert _edition_tags("Movie.Final.Cut.1080p-GRP") == frozenset({"final_cut"})
    assert _edition_tags("Movie.Open.Matte.1080p-GRP") == frozenset({"open_matte"})
    assert _edition_tags("Movie.Special.Edition.1080p-GRP") == frozenset({"special_edition"})
    assert _edition_tags("A.Special.Day.1080p-GRP") == frozenset()
    assert _edition_tags("Open.Season.1080p-GRP") == frozenset()

    # BluRay and REMUX are the same retail master: symmetric recap eligibility.
    assert is_retail_disc_source("bluray") is True
    assert is_retail_disc_source("remux") is True
    assert is_retail_disc_source("webdl") is False
    assert is_retail_disc_source(None) is False


def test_season_episode_provider_query_formats():
    """Every S/E notation providers emit must parse on both axes."""
    from app.services.sync.matching import _member_episode_number, _season_number

    cases = [
        ("Dexter.S08E01.1080p.srt", 8, 1),
        ("Dexter.8x01.720p.srt", 8, 1),
        ("Dexter Season 8 Episode 1.srt", 8, 1),
        ("dexter.s8e01.hdtv.srt", 8, 1),
        ("Show.1x02.WEB-DL.srt", 1, 2),
    ]
    for name, season, episode in cases:
        assert _season_number(name) == season, name
        assert _member_episode_number(name) == episode, name
    # Resolution-like and codec-like numbers must not parse as episodes.
    assert _member_episode_number("Movie.1920x1080.BluRay.srt") is None
    assert _member_episode_number("Show.2024.1080p.WEB-DL.srt") is None
    assert _member_episode_number("Show.x265-GROUP.srt") is None


@pytest.mark.asyncio
async def test_resolver_prefers_matching_group_and_source():
    """End to end: the group+source-matching candidate outscores a generic one."""

    class _P:
        def __init__(self):
            self.downloaded: list[str] = []

        async def search_subtitles(self, **kwargs):
            return [
                SubtitleRelease(
                    release_name="The.Great.Beauty.2013.720p.HDTV.x264-OTHER.srt",
                    download_url="http://other", provider="subdl", lang="eng",
                ),
                SubtitleRelease(
                    release_name="The.Great.Beauty.2013.CHD.1080p.BluRay.srt",
                    download_url="http://chd", provider="subdl", lang="eng",
                ),
            ]

        async def download_archive(self, url, api_key=None):
            self.downloaded.append(url)
            return BIG

    provider = _P()
    resolver = DualReferenceResolver(subdl_provider=provider, subsource_provider=None, timeout=0.5)
    content = await resolver.resolve(
        ReferenceQuery(
            imdb_id="tt2358896",
            media_type="movie",
            target_filename="The.Great.Beauty.2013.BluRay.1080p.DTS-HD.MA.5.1.X264-CHD",
        )
    )
    assert content is not None
    assert provider.downloaded == ["http://chd"]


@pytest.mark.asyncio
async def test_smi_zip_member_converted_to_reference():
    """Season-pack ZIPs with SAMI members still yield a usable reference."""
    import io
    import zipfile

    lines = "".join(f"<SYNC Start={1000 + i * 2000}><P Class=ENCC>line {i}\n" for i in range(200))
    smi = f"<SAMI><HEAD><TITLE>t</TITLE></HEAD><BODY>\n{lines}</BODY></SAMI>"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("show.s08e01.smi", smi)
    zbytes = buf.getvalue()

    class _P:
        async def search_subtitles(self, **kwargs):
            return [
                SubtitleRelease(
                    release_name="Show.S08E01.1080p.BluRay.x264-GRP.smi",
                    download_url="http://pack", provider="subdl", lang="eng",
                )
            ]

        async def download_archive(self, url, api_key=None):
            return zbytes

    resolver = DualReferenceResolver(subdl_provider=_P(), subsource_provider=None, timeout=0.5)
    content = await resolver.resolve(
        ReferenceQuery(
            imdb_id="tt1234567", media_type="series", season=8, episode=1,
            target_filename="Show.S08E01.1080p.BluRay.x265-GRP.mkv",
        )
    )
    assert content is not None
    assert "00:00:01,000 --> 00:00:03,000" in content
    assert "line 0" in content and "SYNC" not in content


@pytest.mark.asyncio
async def test_resolver_downloads_edition_matched_bluray():
    """End to end: PiR8 target resolves a same-source 720p/ALL BluRay edition."""

    class _P:
        def __init__(self):
            self.downloaded: list[str] = []

        async def search_subtitles(self, **kwargs):
            return [
                SubtitleRelease(
                    release_name="Dexter.S08.720p.BluRay.x264-DEMAND.srt",
                    download_url="http://demand", provider="subsource", lang="eng",
                ),
                SubtitleRelease(
                    release_name="Dexter.S08.ALL.BluRay.srt",
                    download_url="http://all", provider="subsource", lang="eng",
                ),
            ]

        async def download_archive(self, url, api_key=None):
            self.downloaded.append(url)
            return BIG

    provider = _P()
    resolver = DualReferenceResolver(subsource_provider=provider, subdl_provider=None, timeout=0.5)
    content = await resolver.resolve(
        ReferenceQuery(
            imdb_id="tt0773262", media_type="series", season=8, episode=1,
            target_filename="Dexter.s8e01.dir.fix.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv",
        )
    )
    assert content is not None
    assert provider.downloaded == ["http://demand"]


@pytest.mark.asyncio
async def test_resolver_scores_matching_source_highest():
    """End to end: the same-source-family candidate outscores a cross-source one."""

    class _P:
        def __init__(self):
            self.downloaded: list[str] = []

        async def search_subtitles(self, **kwargs):
            return [
                SubtitleRelease(
                    release_name="Dexter.S08.complete.720p.WEB-DL.H264-BS PublicHD.srt",
                    download_url="http://publichd", provider="subdl", lang="eng",
                ),
                SubtitleRelease(
                    release_name="Dexter.S08.720p.BluRay.x264-DEMAND.srt",
                    download_url="http://demand", provider="subdl", lang="eng",
                ),
            ]

        async def download_archive(self, url, api_key=None):
            self.downloaded.append(url)
            return BIG

    provider = _P()
    resolver = DualReferenceResolver(subdl_provider=provider, subsource_provider=None, timeout=0.5)
    content = await resolver.resolve(
        ReferenceQuery(
            imdb_id="tt0773262", media_type="series", season=8, episode=1,
            target_filename="Dexter.s8e01.A.Beautiful.Day.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv",
        )
    )
    assert content is not None
    assert provider.downloaded == ["http://demand"]


def test_member_episode_explicit_part_forms():
    from app.services.sync.matching import _member_episode_number as ep

    assert ep("Show.Part.02.srt") == 2
    assert ep("Show Part 2.srt") == 2
    assert ep("Show.Part02.srt") == 2
    assert ep("Show Episode 2.srt") == 2
    assert ep("Show.Ep.02.srt") == 2
    assert ep("Show.Ep02.srt") == 2


def test_select_zip_member_mixed_naming_picks_only_requested_episode():
    resolver = DualReferenceResolver()
    members = [
        ("Dexter.S08E01.720p.HDTV.srt", 9000),
        ("dexter.8x02.srt", 6000),
        ("Dexter.S08E03.srt", 8000),
    ]
    assert resolver._select_zip_member(members, 8, 2) == "dexter.8x02.srt"
    members = [
        ("Show.S01E03.srt", 9000),
        ("Show.S01.Part.02.srt", 7000),
    ]
    assert resolver._select_zip_member(members, 1, 2) == "Show.S01.Part.02.srt"


@pytest.mark.asyncio
async def test_tier1_candidate_used_without_season_query(tmp_path):
    """Any episode candidate is accepted immediately; no season fan-out."""

    class _TieredProvider:
        def __init__(self):
            self.searches: list = []
            self.downloaded: list[str] = []

        async def search_subtitles(self, **kwargs):
            self.searches.append(kwargs.get("episode"))
            if kwargs.get("episode") is None:
                return [
                    SubtitleRelease(
                        release_name="Dexter.S08.1080p.BluRay.x264-GRP.srt",
                        download_url="http://pack", provider="subdl", lang="eng",
                    )
                ]
            return [
                SubtitleRelease(
                    release_name="Dexter.S08E02.720p.HDTV.x264-EVOLVE.srt",
                    download_url="http://hdtv", provider="subdl", lang="eng",
                )
            ]

        async def download_archive(self, url, api_key=None):
            self.downloaded.append(url)
            return BIG

    provider = _TieredProvider()
    resolver = DualReferenceResolver(
        subdl_provider=provider,
        subsource_provider=None,
        cache=ReferenceDiskCache(root=tmp_path / "disk", ttl=3600.0, min_bytes=100),
        timeout=0.5,
    )
    content = await resolver.resolve(
        ReferenceQuery(
            imdb_id="tt0773262", media_type="series", season=8, episode=2,
            target_filename="Dexter.S08E02.1080p.BluRay.x264.mkv",
        )
    )
    assert content is not None
    assert provider.searches == [2]
    assert provider.downloaded == ["http://hdtv"]


@pytest.mark.asyncio
async def test_tier2_skipped_when_family_covered_or_unknown(tmp_path):
    """No second search for covered families, unknown targets, or movies."""

    class _CountingProvider:
        def __init__(self, release_name):
            self.release_name = release_name
            self.searches = 0

        async def search_subtitles(self, **kwargs):
            self.searches += 1
            return [
                SubtitleRelease(
                    release_name=self.release_name, download_url="http://x",
                    provider="subdl", lang="eng",
                )
            ]

        async def download_archive(self, url, api_key=None):
            return BIG

    def _resolver(release_name):
        provider = _CountingProvider(release_name)
        resolver = DualReferenceResolver(
            subdl_provider=provider,
            subsource_provider=None,
            cache=ReferenceDiskCache(root=tmp_path / "disk", ttl=3600.0, min_bytes=100),
            timeout=0.5,
        )
        return provider, resolver

    # Tier 1 already BluRay: single search. (Distinct IMDb per case so the
    # shared tmp cache cannot cross-hit between sub-cases.)
    provider, resolver = _releases_provider_helper(_CountingProvider, "Dexter.S08E02.1080p.BluRay.x264-GRP.srt", tmp_path)
    await resolver.resolve(
        ReferenceQuery(
            imdb_id="tt11", media_type="series", season=8, episode=2,
            target_filename="Dexter.S08E02.1080p.BluRay.x264.mkv",
        )
    )
    assert provider.searches == 1

    # Unknown target source: single search even with results present.
    provider, resolver = _releases_provider_helper(_CountingProvider, "Dexter.S08E02.720p.HDTV.x264-EVOLVE.srt", tmp_path)
    await resolver.resolve(
        ReferenceQuery(imdb_id="tt22", media_type="series", season=8, episode=2)
    )
    assert provider.searches == 1

    # Movies never trigger season enrichment.
    provider, resolver = _releases_provider_helper(_CountingProvider, "Movie.2024.720p.HDTV.srt", tmp_path)
    await resolver.resolve(
        ReferenceQuery(imdb_id="tt33", target_filename="Movie.2024.1080p.BluRay.x264-GRP.mkv")
    )
    assert provider.searches == 1


def _releases_provider_helper(cls, release_name, tmp_path):
    provider = cls(release_name)
    resolver = DualReferenceResolver(
        subdl_provider=provider,
        subsource_provider=None,
        cache=ReferenceDiskCache(root=tmp_path / "disk", ttl=3600.0, min_bytes=100),
        timeout=0.5,
    )
    return provider, resolver


@pytest.mark.asyncio
async def test_pack_extraction_cached_under_episode_key(tmp_path):
    """An E02 file sliced from a season pack is cached under the E02 stem."""
    import io
    import zipfile

    def _cue(text):
        return "1\n00:00:01,000 --> 00:00:02,000\n" + (text + "\n") * 600 + "\n"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("dexter.s08e01.smi", _cue("episode one"))
        archive.writestr("dexter.s08e02.smi", _cue("episode two"))
        archive.writestr("dexter.s08e03.smi", _cue("episode three"))
    zbytes = buf.getvalue()

    class _P:
        async def search_subtitles(self, **kwargs):
            return [
                SubtitleRelease(
                    release_name="Dexter.S08.BluRay.x264-GRP.smi",
                    download_url="http://pack", provider="subdl", lang="eng",
                )
            ]

        async def download_archive(self, url, api_key=None):
            return zbytes

    cache = ReferenceDiskCache(root=tmp_path / "disk", ttl=3600.0, min_bytes=100)
    resolver = DualReferenceResolver(subdl_provider=_P(), subsource_provider=None, cache=cache, timeout=0.5)
    content = await resolver.resolve(
        ReferenceQuery(
            imdb_id="tt0773262", media_type="series", season=8, episode=2,
            target_filename="Dexter.S08E02.1080p.BluRay.x264-GRP.mkv",
        )
    )
    assert content is not None
    assert "episode two" in content
    assert "episode one" not in content and "episode three" not in content
    assert len(list((tmp_path / "disk").glob("tt0773262_8_2_GRP_v2_*_subdl_edition.srt"))) == 1


def test_reference_disk_cache_restores_sidecar_verdict(tmp_path):
    """Kind, bluray flag, and candidate name must all survive a round trip."""
    query = ReferenceQuery(imdb_id="tt0758758", media_type="movie")
    cache = ReferenceDiskCache(root=tmp_path / "disk", ttl=3600.0, min_bytes=100)
    text = _valid_reference_text()
    cache.set(
        query, "subdl", text, kind="team",
        bluray_match=True, candidate="Into.The.Wild.2007.1080p.BluRay.x264-FSiHD.ENG.srt",
    )
    resolved = cache.get(query)
    assert resolved == ResolvedReference(
        text, "team", True, "Into.The.Wild.2007.1080p.BluRay.x264-FSiHD.ENG.srt",
    )


@pytest.mark.asyncio
async def test_tier2_relaxed_recovers_season_pack_for_uninformative_target(tmp_path, monkeypatch):
    """Relaxed + obfuscated target: an empty episode query falls back to the pack."""
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "SYNC_REQUIRE_EXACT_MATCH", False)

    class _PackProvider:
        def __init__(self):
            self.searches: list = []
            self.downloaded: list[str] = []

        async def search_subtitles(self, **kwargs):
            self.searches.append(kwargs.get("episode"))
            if kwargs.get("episode") is None:
                return [
                    SubtitleRelease(
                        release_name="Dexter.S08.1080p.BluRay.x264-GRP.srt",
                        download_url="http://pack", provider="subdl", lang="eng",
                    )
                ]
            return []

        async def download_archive(self, url, api_key=None):
            self.downloaded.append(url)
            return BIG

    provider = _PackProvider()
    resolver = DualReferenceResolver(
        subdl_provider=provider,
        subsource_provider=None,
        cache=ReferenceDiskCache(root=tmp_path / "disk", ttl=3600.0, min_bytes=100),
        timeout=0.5,
    )
    content = await resolver.resolve(
        ReferenceQuery(
            imdb_id="tt1632701", media_type="series", season=8, episode=1,
            target_filename="e4WcFo4Tz5J8PoFwiBfP880XsBuHk4dS.mkv",
        )
    )
    assert content is not None
    assert provider.searches == [1, None]
    assert provider.downloaded == ["http://pack"]


@pytest.mark.asyncio
async def test_tier2_enriches_empty_episode_query(tmp_path):
    """An empty episode query falls back to the season catalog."""

    class _CountingProvider:
        def __init__(self):
            self.searches: list = []

        async def search_subtitles(self, **kwargs):
            self.searches.append(kwargs.get("episode"))
            return []

        async def download_archive(self, url, api_key=None):
            return BIG

    provider = _CountingProvider()
    resolver = DualReferenceResolver(
        subdl_provider=provider,
        subsource_provider=None,
        cache=ReferenceDiskCache(root=tmp_path / "disk", ttl=3600.0, min_bytes=100),
        timeout=0.5,
    )
    await resolver.resolve(
        ReferenceQuery(
            imdb_id="tt33", media_type="series", season=8, episode=1,
            target_filename="e4WcFo4Tz5J8PoFwiBfP880XsBuHk4dS.mkv",
        )
    )
    assert provider.searches == [1, None]


def test_looks_like_season_pack_and_candidate_episode():
    from app.services.sync.matching import (
        candidate_episode_number,
        looks_like_season_pack,
    )

    assert looks_like_season_pack("Suits, SEASON 1, 1080p [23.976 FPS].srt") is True
    assert looks_like_season_pack("Suits.Season.1.1080p.BluRay.x265-DTG.srt") is True
    assert looks_like_season_pack("Suits.S01.Complete.720p.x264-MIXED.srt") is True
    assert looks_like_season_pack("Suits.S01-S09.1080p.WEB-DL.srt") is True
    assert looks_like_season_pack("Suits - First Season TV.srt") is True
    assert looks_like_season_pack("Suits.S01E01.Pilot.720p.WEB-DL.srt") is False
    assert looks_like_season_pack("Show.EP02.1080p.srt") is False
    assert looks_like_season_pack("Movie.2024.1080p.BluRay.x264-GRP.srt") is False

    assert candidate_episode_number("Suits.Season.1.1080p.DTG.srt") is None
    assert candidate_episode_number("Suits.S01E02.Pilot.srt") == 2
    assert candidate_episode_number("Into.the.Wild.2007.1080p.BluRay.DD5.1.x264-playHD-Rakuv.mkv") is None
    assert candidate_episode_number("Movie.2023.TrueHD.Atmos.7.1.4.mkv") is None


def test_reference_group_rank_prefers_retail_encodes():
    from app.services.sync.matching import reference_group_rank

    assert reference_group_rank("Into the Wild 2007 BluRay.1080p.DTS.x264-CHD.srt") == 0
    assert reference_group_rank("Into.The.Wild.2007.1080p BluRay.x264-FSiHD.ENG.srt") == 0
    assert reference_group_rank("Movie.2024.1080p.BluRay.x264-EbP.srt") == 0
    assert reference_group_rank("Movie.2024.1080p.BluRay.x264-HDC.srt") == 0
    assert reference_group_rank("Into.the.Wild.2007.1080p.Bluray.x265.HEVC.Tigole.srt") == 2
    assert reference_group_rank("Movie.2024.1080p.WEB-DL.x264.YIFY.srt") == 2
    assert reference_group_rank("Movie.2024.1080p.BluRay.x264-NTb.srt") == 1
    assert reference_group_rank("Movie.2024.1080p.BluRay.srt") == 1


# --------------------------------------------------------------------------- #
# Weighted multi-language reference scoring
# --------------------------------------------------------------------------- #
_MAD_MEN_TARGET = (
    "Mad.Men.S01E02.Ladies.Room.REPACK.2160p.HMAX.WEB-DL."
    "DDP5.1.DV.HDR.H.265-WADU.mkv"
)


def _cand(release_name, *, lang="eng", hi=False, hash_match=False):
    return SubtitleRelease(
        release_name=release_name,
        download_url=f"http://cdn/{release_name}",
        provider="subdl",
        lang=lang,
        hearing_impaired=hi,
        is_hash_match=hash_match,
    )


def _mad_men_query():
    return ReferenceQuery(
        imdb_id="tt1", media_type="series", season=1, episode=2,
        target_filename=_MAD_MEN_TARGET,
    )


def test_reference_tier_webdl_beats_generic_bluray():

    webdl = _cand("Mad.Men.S01E02.2160p.HMAX.WEB-DL.DDP5.1.H.265-BTN.srt")
    bluray = _cand("Mad.Men.S01E02.1080p.BluRay.23.976.FPS.x264-GRP.srt")
    # Target is a WEB-DL: the WEB-DL candidate shares the source medium while the
    # BluRay candidate does not, so it lands in a strictly better tier.
    assert reference_tier(_MAD_MEN_TARGET, webdl) == TIER_SOURCE_EDITION
    assert reference_tier(_MAD_MEN_TARGET, bluray) == TIER_FALLBACK
    assert reference_tier(_MAD_MEN_TARGET, webdl) < reference_tier(_MAD_MEN_TARGET, bluray)


def test_reference_tier_exact_group_beats_generic_english():

    wadu_es = _cand("Mad.Men.S01E02.2160p.HMAX.WEB-DL.DDP5.1.H.265-WADU.srt", lang="spa")
    generic_en = _cand("Mad.Men.S01E02.2160p.WEB-DL.x265.srt", lang="eng")
    # BTN appears in the target release name, so WADU's shared medium is
    # irrelevant: an exact group outranks language and property preference.
    assert reference_tier(_MAD_MEN_TARGET, wadu_es) == TIER_EXACT_GROUP
    assert reference_tier(_MAD_MEN_TARGET, generic_en) == TIER_SOURCE_EDITION
    assert reference_tier(_MAD_MEN_TARGET, wadu_es) < reference_tier(_MAD_MEN_TARGET, generic_en)


def test_reference_tier_is_language_agnostic():

    en = _cand("Mad.Men.S01E02.2160p.WEB-DL.x265.srt", lang="eng")
    fr = _cand("Mad.Men.S01E02.2160p.WEB-DL.x265.srt", lang="fra")
    # Tiers encode release identity only; language never influences them.
    assert reference_tier(_MAD_MEN_TARGET, en) == reference_tier(_MAD_MEN_TARGET, fr)


def test_select_reference_breaks_tier_ties_with_english_anchor():
    from app.services.sync.external_strategy import _select_reference

    en = _cand("Mad.Men.S01E02.2160p.WEB-DL.x265.srt", lang="eng")
    fr = _cand("Mad.Men.S01E02.2160p.WEB-DL.x265.srt", lang="fra")
    # Same tier, so the English reference anchor wins deterministically.
    assert _select_reference([fr, en], _mad_men_query()) is en


def test_select_reference_picks_higher_tier_candidate():
    from app.services.sync.external_strategy import _select_reference

    bluray = _cand("Mad.Men.S01E02.1080p.BluRay.23.976.FPS.x264-GRP.srt")
    webdl = _cand("Mad.Men.S01E02.2160p.HMAX.WEB-DL.H.265-BTN.srt")
    assert _select_reference([bluray, webdl], _mad_men_query()) is webdl


def test_select_reference_falls_back_deterministically():
    from app.services.sync.external_strategy import _select_reference

    # Both in the fallback tier with no shared metadata: the release name is the
    # final tie-breaker, so ordering is reproducible instead of input-dependent.
    first = _cand("Mad.Men.S01E02.release.one.srt", lang="por", hi=True)
    second = _cand("Mad.Men.S01E02.release.two.srt", lang="por", hi=True)
    assert _select_reference([first, second], _mad_men_query()) is first
    assert _select_reference([second, first], _mad_men_query()) is first


def test_select_reference_exact_group_language_beats_generic_english():
    from app.services.sync.external_strategy import _select_reference

    generic_en = _cand("Mad.Men.S01E02.1080p.BluRay.23.976.FPS.x264-GRP.srt", lang="eng")
    wadu_fr = _cand("Mad.Men.S01E02.2160p.HMAX.WEB-DL.H.265-WADU.srt", lang="fra")
    # Tier (exact group) outranks the English-language preference.
    assert _select_reference([generic_en, wadu_fr], _mad_men_query()) is wadu_fr


@pytest.mark.asyncio
async def test_cross_provider_tiering_beats_first_provider(tmp_path):
    """The first provider to answer must not win: tier the pool globally.

    Regression for Mad Men: subdl's 720p WEB-DL season pack must lose to an
    episode-specific streaming WEB-DL reference from another provider.
    """
    from app.services.sync.cache import ReferenceDiskCache
    from app.services.sync.external_strategy import ExternalExactStrategy

    class _Provider:
        def __init__(self, name, releases):
            self.name = name
            self._releases = releases
            self.downloaded: list[str] = []

        async def search_subtitles(self, **kwargs):
            return list(self._releases)

        async def download_archive(self, url, api_key=None):
            self.downloaded.append(url)
            return BIG

    subdl = _Provider(
        "subdl",
        [
            SubtitleRelease(
                release_name="Mad.Men.S01.720p.WEB-DL.AAC2.0.H264-BTN.srt",
                download_url="http://pack", provider="subdl", lang="eng",
            )
        ],
    )
    subsource = _Provider(
        "subsource",
        [
            SubtitleRelease(
                release_name="Mad.Men.S01E02.Ladies.Room.1080p.AMZN.WEB-DL.DDP5.1.H.264-SLiGNOME.srt",
                download_url="http://amzn", provider="subsource", lang="eng",
            )
        ],
    )
    strategy = ExternalExactStrategy(
        subdl_provider=subdl,
        subsource_provider=subsource,
        opensubtitles_provider=None,
        cache=ReferenceDiskCache(root=tmp_path / "refs", ttl=3600.0, min_bytes=100),
        timeout=1.0,
    )
    content = await strategy.resolve(_mad_men_query())
    assert content is not None
    assert subsource.downloaded == ["http://amzn"]
    assert subdl.downloaded == []


