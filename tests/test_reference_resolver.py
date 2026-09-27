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


def _releases(*names: str) -> list[SubtitleRelease]:
    return [
        SubtitleRelease(
            release_name=name, download_url=f"http://x/{i}",
            provider="subdl", lang="eng",
        )
        for i, name in enumerate(names)
    ]


def _tree_query(target: str | None, season: int | None = 8, episode: int | None = 1):
    return ReferenceQuery(
        imdb_id="tt0773262", media_type="series", season=season, episode=episode,
        target_filename=target,
    )


def test_tree_team_match_beats_generic_source():
    """Same group on a compatible source wins despite codec differences."""
    from app.services.sync.tree import decide

    target = "Dexter.S08E01.1080p.BluRay.x265-GRP.mkv"
    candidates = _releases(
        "Dexter.S08E01.480p.HDTV.x264-mSD.srt",
        "Dexter.S08E01.1080p.BluRay.x264-GRP.srt",
    )
    decision = decide(candidates, _tree_query(target), strict=True)
    assert decision.kind == "team"
    assert getattr(decision.release, "release_name", "") == (
        "Dexter.S08E01.1080p.BluRay.x264-GRP.srt"
    )


def test_tree_edition_match_requires_full_confirmation():
    """Source+codec+res+explicit S/E confirm an edition; unknowns abort in strict mode."""
    from app.services.sync.tree import decide

    target = "Dexter.S08E01.1080p.BluRay.x264.mkv"
    full = _releases("Dexter.S08E01.1080p.BluRay.x264-Other.srt")
    decision = decide(full, _tree_query(target), strict=True)
    assert decision.kind == "edition"

    # Unknown source on the candidate fails closed in strict mode.
    bare = _releases("Dexter.S08E01.srt")
    decision = decide(bare, _tree_query(target), strict=True)
    assert decision.kind == "abort"

    # ... but is accepted when strictness is relaxed.
    decision = decide(bare, _tree_query(target), strict=False)
    assert decision.kind == "edition"

    # A direct conflict (HDTV vs BluRay) aborts in both modes.
    hdtv = _releases("Dexter.S08E01.480p.HDTV.x264-mSD.srt")
    assert decide(hdtv, _tree_query(target), strict=True).kind == "abort"
    assert decide(hdtv, _tree_query(target), strict=False).kind == "abort"


def test_tree_aborts_without_target_edition_info():
    """No group/source/codec/res on the target means nothing can be confirmed."""
    from app.services.sync.tree import decide

    candidates = _releases("Dexter.S08E01.1080p.BluRay.x264-GRP.srt")
    decision = decide(candidates, _tree_query(None), strict=True)
    assert decision.kind == "abort"


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
async def test_resolver_logs_evaluated_candidates(caplog):
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
    assert "selected English reference" in caplog.text


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


def test_exact_release_group_beats_generic_source_match():
    """Into the Wild: the FSiHD reference must win the tree over Tigole."""
    from app.services.sync.tree import decide

    target = "Into.The.Wild.2007.1080p.BluRay.x264-FSiHD"
    candidates = _releases(
        "Into.the.Wild.2007.1080p.Bluray.x265.HEVC.10bit.AAC.5.1.Tigole.srt",
        "Into.The.Wild.2007.1080p.BluRay.x264-FSiHD.ENG.srt",
    )
    query = ReferenceQuery(imdb_id="tt0758758", media_type="movie", target_filename=target)
    decision = decide(candidates, query, strict=True)
    assert decision.kind == "team"
    assert getattr(decision.release, "release_name", "") == (
        "Into.The.Wild.2007.1080p.BluRay.x264-FSiHD.ENG.srt"
    )


@pytest.mark.asyncio
async def test_resolver_ignores_release_group():
    """End to end: release group is not a filter — the top candidate wins."""

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
    assert provider.downloaded == ["http://tigole"]


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


def test_tree_edition_tags_must_agree_in_strict_mode():
    """US vs plain and Extended vs plain abort strict; REMUX never counts as a cut."""
    from app.services.sync.tree import decide

    def query(target):
        return ReferenceQuery(imdb_id="tt1", media_type="movie", target_filename=target)

    us_target = "Movie.US.1080p.BluRay.x264"
    assert decide(
        _releases("Movie.1080p.BluRay.x264-Other.srt"), query(us_target), strict=True
    ).kind == "abort"
    assert decide(
        _releases("Movie.US.1080p.BluRay.x264-Other.srt"), query(us_target), strict=True
    ).kind == "edition"

    ext_target = "Movie.Extended.1080p.BluRay.x264"
    assert decide(
        _releases("Movie.1080p.BluRay.x264-Other.srt"), query(ext_target), strict=True
    ).kind == "abort"

    # "remux" is an encoding method, not a cut: a REMUX target accepts a
    # BluRay encode of the same master (source fuzziness already models this).
    remux_target = "Movie.2007.1080p.REMUX.AVC.DTS-HD.MA.5.1-AAA"
    assert decide(
        _releases("Movie.2007.1080p.BluRay.x264-BBB.srt"), query(remux_target), strict=True
    ).kind == "edition"

    # Relaxed mode ignores tag asymmetry (legacy availability).
    assert decide(
        _releases("Movie.1080p.BluRay.x264-Other.srt"), query(us_target), strict=False
    ).kind == "edition"


def test_tree_edition_prefers_uhd_remux_over_legacy_bluray():
    """Pulp Fiction: the 2160p REMUX wins over a listed-first legacy release."""
    from app.services.sync.tree import decide

    target = "Pulp.Fiction.1994.2160p.UHD.BluRay.HEVC.DTS-HD.MA.5.1-SPHD"
    candidates = _releases(
        "Pulp Fiction (Blu-ray).srt",
        "Pulp.Fiction.1994.2160p.BluRay.REMUX.HEVC.DTS-HD.MA.5.1-FGT.srt",
    )
    query = ReferenceQuery(imdb_id="tt0110912", media_type="movie", target_filename=target)
    decision = decide(candidates, query, strict=True)
    assert decision.kind == "edition"
    assert "FGT" in getattr(decision.release, "release_name", "")


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


def test_team_match_finds_non_trailing_group():
    """A group placed mid-filename (CHD.BluRay) still confirms the team."""
    from app.services.sync.tree import decide

    target = "The.Great.Beauty.2013.BluRay.1080p.DTS-HD.MA.5.1.X264-CHD"
    candidates = _releases(
        "The.Great.Beauty.2013.720p.HDTV.x264-OTHER.srt",
        "The.Great.Beauty.2013.CHD.1080p.BluRay.srt",
    )
    query = ReferenceQuery(imdb_id="tt2358896", media_type="movie", target_filename=target)
    decision = decide(candidates, query, strict=True)
    assert decision.kind == "team"
    assert "CHD.1080p" in getattr(decision.release, "release_name", "")


@pytest.mark.asyncio
async def test_resolver_uses_first_movie_candidate():
    """End to end: the top same-imdb candidate is used regardless of source."""

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
    assert provider.downloaded == ["http://other"]


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


def test_tree_edition_allows_same_source_resolution_cross():
    """720p BluRay shares the retail-disc master with 1080p BluRay."""
    from app.services.sync.tree import decide

    target = "Dexter.S08E01.1080p.BluRay.x264.mkv"

    def query():
        return ReferenceQuery(imdb_id="tt0773262", media_type="series", season=8, episode=1, target_filename=target)

    assert decide(_releases("Dexter.S08.720p.BluRay.x264-DEMAND.srt"), query(), strict=True).kind == "edition"
    assert decide(_releases("Dexter.S08.ALL.BluRay.srt"), query(), strict=True).kind == "edition"
    # Different sources never mix, even at the same resolution.
    assert decide(_releases("Dexter.S08E01.1080p.WEB-DL.x264-GRP.srt"), query(), strict=True).kind == "abort"
    assert decide(_releases("Dexter.S08E01.480p.HDTV.x264-mSD.srt"), query(), strict=True).kind == "abort"
    # Outside the 1080p/720p family the resolution must match exactly.
    assert decide(_releases("Dexter.S08E01.2160p.BluRay.x264-GRP.srt"), query(), strict=True).kind == "abort"


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


def test_same_source_outranks_webdl_despite_provider_order():
    """PiR8 BluRay target: a BluRay edition must beat a listed-first WEB-DL."""
    from app.services.sync.tree import decide

    target = "Dexter.s8e01.A.Beautiful.Day.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv"
    candidates = _releases(
        "Dexter.S08.complete.720p.WEB-DL.H264-BS PublicHD.srt",
        "Dexter.S08.720p.BluRay.x264-DEMAND.srt",
        "Dexter.S08.ALL.BluRay.srt",
    )
    query = ReferenceQuery(
        imdb_id="tt0773262", media_type="series", season=8, episode=1, target_filename=target
    )
    decision = decide(candidates, query, strict=True)
    assert decision.kind == "edition"
    assert "PublicHD" not in getattr(decision.release, "release_name", "")
    assert "BluRay" in getattr(decision.release, "release_name", "")


@pytest.mark.asyncio
async def test_resolver_accepts_cross_source_reference():
    """End to end: source type is not a filter — the top candidate is used."""

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
    assert provider.downloaded == ["http://publichd"]


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


def test_tree_team_vetoes_cut_conflicts():
    """Same group but conflicting cuts (Extended vs Theatrical) must not align."""
    from app.services.sync.tree import decide

    def query(target):
        return ReferenceQuery(imdb_id="tt1", media_type="movie", target_filename=target)

    ext_target = "Movie.Extended.1080p.BluRay.x264-GRP.mkv"
    theatrical = _releases("Movie.Theatrical.1080p.BluRay.x264-GRP.srt")
    # Strict: vetoed team cannot resurface; nothing else matches -> abort.
    assert decide(theatrical, query(ext_target), strict=True).kind == "abort"
    # Relaxed: explicit-vs-explicit cut conflicts veto everywhere too.
    assert decide(theatrical, query(ext_target), strict=False).kind == "abort"
    # Untagged same-group release stays wildcard-eligible when relaxed...
    assert (
        decide(
            _releases("Movie.1080p.BluRay.x264-GRP.srt"), query(ext_target), strict=False
        ).kind
        == "team"
    )
    # ... and matching cuts on both sides still confirm the team.
    assert (
        decide(
            _releases("Movie.Extended.1080p.BluRay.x264-GRP.srt"),
            query(ext_target),
            strict=True,
        ).kind
        == "team"
    )
    # Plain stream vs Extended team release: vetoed in strict mode.
    assert (
        decide(
            _releases("Movie.Extended.1080p.BluRay.x264-GRP.srt"),
            query("Movie.1080p.BluRay.x264.mkv"),
            strict=True,
        ).kind
        == "abort"
    )


def test_tree_relaxed_fallback_for_uninformative_target():
    """An obfuscated target aborts strict but is served best-match relaxed."""
    from app.services.sync.tree import decide

    obfuscated = "e4WcFo4Tz5J8PoFwiBfP880XsBuHk4dS.mkv"
    candidates = _releases(
        "Dexter.S08E01.1080p.WEB-DL.x264-NTb.srt",
        "Dexter.S08E01.2160p.WEB-DL.x265-GRP.srt",
        "Dexter.S08E02.1080p.WEB-DL.x264-NTb.srt",
        "Dexter.S08.1080p.BluRay.x264-GRP.srt",
    )
    # Strict: nothing to confirm against -> the exact production abort.
    strict_decision = decide(candidates, _tree_query(obfuscated), strict=True)
    assert strict_decision.kind == "abort"
    assert strict_decision.reason == "target carries no edition info to confirm against"

    # Relaxed: best same-episode candidate (UHD ranked first); wrong episode
    # is still dropped by the episode filter, not silently accepted.
    decision = decide(candidates, _tree_query(obfuscated), strict=False)
    assert decision.kind == "edition"
    assert decision.reason.startswith("relaxed fallback")
    assert "2160p" in decision.release.release_name
    assert "S08E02" not in decision.release.release_name

    # An untagged season pack is an acceptable relaxed fallback too.
    pack = _releases("Dexter.S08.1080p.BluRay.x264-GRP.srt")
    assert decide(pack, _tree_query(obfuscated), strict=True).kind == "abort"
    assert decide(pack, _tree_query(obfuscated), strict=False).kind == "edition"


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


def test_relaxed_fallback_prefers_explicit_episode_over_complete_pack():
    """A 'S01 complete' pack must never outrank the explicit E01 single."""
    from app.services.sync.tree import decide

    obfuscated = "e4WcFo4Tz5J8PoFwiBfP880XsBuHk4dS.mkv"
    pool = _releases(
        "Suits S01 complete (360p re-webrip).srt",
        "Suits.S01E01.1080p.WEB-DL.x264-NTb.srt",
    )
    decision = decide(pool, _tree_query(obfuscated, season=1, episode=1), strict=False)
    assert decision.kind == "edition"
    assert "S01E01" in decision.release.release_name
    assert "complete" not in decision.release.release_name


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


def test_tree_prefers_matching_source_single_over_unknown_pack():
    """BluRay stream: prefer a BluRay episode single over an unknown-source pack."""
    from app.services.sync.tree import decide

    query = _tree_query(
        "Suits.S01E01.Pilot.BluRay.1080p.10Bit.DtsHDMa5.1.HEVC-d3g.mkv",
        season=1,
        episode=1,
    )
    candidates = _releases(
        "Suits, SEASON 1, 1080p [23.976 FPS].srt",
        "Suits Season 1  720p mkv compression [mkvGOD].srt",
        "Suits.S01.Complete.720p.x264-MIXED.srt",
        "Suits - First Season TV.srt",
        "Suits Season 1 All 12 Episodes.srt",
        "Suits S01E01 720p BluRay DD5.1 x264-EbP.srt",
        "Suits.Season.1.1080p.BluRay.AAC5.1.x265-DTG.srt",
    )
    decision = decide(candidates, query, strict=False)
    assert decision.kind == "edition"
    assert "S01E01" in decision.release.release_name
    assert "BluRay" in decision.release.release_name
    assert "SEASON 1" not in decision.release.release_name


def test_tree_season_pack_usable_for_later_episode():
    """A 'Season 1' pack must survive the episode filter for S01E02 (sliceable)."""
    from app.services.sync.tree import decide

    query = _tree_query("Suits.S01E02.BluRay.1080p.x265-GRP.mkv", season=1, episode=2)
    candidates = _releases("Suits.Season.1.1080p.BluRay.AAC5.1.x265-DTG.srt")
    decision = decide(candidates, query, strict=False)
    assert decision.kind == "edition"
    assert "Season.1" in decision.release.release_name


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


def test_tree_prefers_retail_encode_over_micro_rip():
    """Among same-source editions, a scene retail encode beats a micro-rip."""
    from app.services.sync.tree import decide

    query = _tree_query(
        "Into.the.Wild.2007.1080p.BluRay.x264-GRP.mkv", season=None, episode=None
    )
    candidates = _releases(
        "Into.the.Wild.2007.1080p.Bluray.x265.HEVC.10bit.AAC.5.1.Tigole.srt",
        "Into the Wild 2007 BluRay.1080p.DTS.x264-CHD.srt",
    )
    decision = decide(candidates, query, strict=False)
    assert decision.kind == "edition"
    assert "CHD" in decision.release.release_name
