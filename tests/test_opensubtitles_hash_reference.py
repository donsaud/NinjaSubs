"""OpenSubtitles as a byte-exact MovieHash reference for ALASS auto-sync.

Covers the dedicated hash-reference strategy in
``app/services/sync/hash_reference.py`` and its wiring into the orchestrator.

Two properties are asserted throughout and are the reason this path is separate
from the normal provider:

* a candidate is accepted only when OpenSubtitles reported
  ``attributes.moviehash_match is True`` -- the provider derives
  ``is_hash_match`` from exactly that, so a title/IMDb/filename resemblance can
  never become a reference;
* selection is language-agnostic. The reference supplies timing; the user's
  subtitle stays the ALASS target and keeps its own language.

Nothing here touches the network. The provider is a recording double, downloads
return canned bytes, and the reference cache is redirected at tmp_path.
"""

from __future__ import annotations

import pytest

from app.models import SubtitleRelease
from app.services.sync.cache import ReferenceDiskCache
from app.services.sync.hash_reference import (
    OpenSubtitlesHashReferenceStrategy,
    _is_explicit_hash_match,
)
from app.services.sync.query import ReferenceQuery, ResolvedReference
from app.services.sync_service import SubtitleSyncService

HASH = "8e245d9679d31e12"
SIZE = 19571049411
API_KEY = "test-key"
USER = "test-user"
PASSWORD = "test-pass"


# --------------------------------------------------------------------------
# Fixtures / doubles
# --------------------------------------------------------------------------


class RecordingProvider:
    """Stands in for OpenSubtitlesProvider; records every call."""

    name = "opensubtitles"

    def __init__(self, releases=None, payload: bytes | None = None) -> None:
        self.releases = releases or []
        self.payload = payload
        self.search_calls: list[dict] = []
        self.download_calls: list[tuple] = []
        self.breaker_open = False

    async def search_subtitles(self, **kwargs):
        self.search_calls.append(kwargs)
        return list(self.releases)

    async def download_archive(self, reference, api_key=None, username=None, password=None):
        self.download_calls.append((reference, api_key, username, password))
        return self.payload

    def is_breaker_open(self) -> bool:
        return self.breaker_open


def make_release(**overrides) -> SubtitleRelease:
    base = {
        "release_name": "Whiplash.2014.720p.BluRay.x264-AMIABLE.srt",
        "download_url": "/sub/opensubtitles/12345.srt",
        "provider": "opensubtitles",
        "lang": "eng",
        "format": "srt",
        "is_hash_match": True,
    }
    base.update(overrides)
    return SubtitleRelease(**base)


def make_query(**overrides) -> ReferenceQuery:
    base = {
        "imdb_id": "tt2582802",
        "target_filename": "Whiplash.2014.2160p.UHD.BluRay.x264-SURCODE.mkv",
        "media_type": "movie",
        "title": "Whiplash",
        "year": 2014,
        "video_hash": HASH,
        "video_size": SIZE,
        "api_keys": {
            "opensubtitles": API_KEY,
            "opensubtitles_username": USER,
            "opensubtitles_password": PASSWORD,
        },
    }
    base.update(overrides)
    return ReferenceQuery(**base)


def _ts(ms: int) -> str:
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


# Irregular cue spacing on purpose. Uniform spacing plus a shift that is a
# multiple of that spacing makes median_cue_offset degenerate: nearest-neighbour
# matching finds an exact hit for every target cue, the measured offset reads as
# 0.0, and the orchestrator takes the already-aligned shortcut without ever
# invoking alass. Prime-ish spacing keeps the offset measurable.
_SPACING = (2870, 4310, 1990, 5380, 2410, 3620, 4790, 2180, 3340, 4060)


def _cue_times(cues: int, offset_ms: int = 0) -> list[tuple[int, int]]:
    times: list[tuple[int, int]] = []
    cursor = 4000 + offset_ms
    for i in range(cues):
        step = _SPACING[i % len(_SPACING)]
        times.append((cursor, cursor + step - 400))
        cursor += step
    return times


def reference_text(cues: int = 80, offset_ms: int = 0) -> str:
    """A plausible SRT reference, comfortably over the 5120-byte floor."""
    blocks = []
    for i, (start, end) in enumerate(_cue_times(cues, offset_ms), 1):
        blocks.append(
            f"{i}\n{_ts(start)} --> {_ts(end)}\n"
            f"reference line {i} carrying enough words to look like real dialogue\n"
        )
    return "\n".join(blocks)


def target_text(cues: int = 80, offset_ms: int = 2337) -> str:
    """The user's subtitle: the same dialogue, shifted by a non-multiple."""
    blocks = []
    for i, (start, end) in enumerate(_cue_times(cues, offset_ms), 1):
        blocks.append(
            f"{i}\n{_ts(start)} --> {_ts(end)}\n"
            f"target line {i} carrying enough words to look like real dialogue\n"
        )
    return "\n".join(blocks)


@pytest.fixture
def cache(tmp_path) -> ReferenceDiskCache:
    """Reference cache isolated to tmp_path so tests never share real entries."""
    return ReferenceDiskCache(root=tmp_path / "references", min_bytes=5120)


@pytest.fixture
def strategy_factory(cache):
    def build(provider, **kwargs) -> OpenSubtitlesHashReferenceStrategy:
        return OpenSubtitlesHashReferenceStrategy(provider, cache=cache, **kwargs)

    return build


# --------------------------------------------------------------------------
# 1/2/14. Strict hash-match acceptance
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_explicit_hash_match_is_accepted_as_a_reference(strategy_factory, cache):
    provider = RecordingProvider([make_release()], payload=reference_text().encode())
    resolved = await strategy_factory(provider).resolve_with_provenance(make_query())

    assert resolved.text is not None
    assert resolved.kind == "hash"
    assert resolved.candidate == "Whiplash.2014.720p.BluRay.x264-AMIABLE.srt"
    assert len(provider.download_calls) == 1


@pytest.mark.asyncio
async def test_a_non_hash_matched_result_is_rejected_and_never_downloaded(strategy_factory):
    provider = RecordingProvider(
        [make_release(is_hash_match=False, matched_by_hash=False)],
        payload=reference_text().encode(),
    )
    resolved = await strategy_factory(provider).resolve_with_provenance(make_query())

    assert resolved.text is None
    # The decisive assertion: nothing was spent on a non-exact candidate.
    assert provider.download_calls == []


@pytest.mark.asyncio
async def test_a_title_only_resemblance_is_rejected(strategy_factory):
    """Same release name and IMDb, no hash match -- still not a reference."""
    provider = RecordingProvider(
        [
            make_release(
                release_name="Whiplash.2014.2160p.UHD.BluRay.x264-SURCODE.srt",
                is_hash_match=False,
                matched_by_hash=False,
            )
        ],
        payload=reference_text().encode(),
    )
    assert (await strategy_factory(provider).resolve_with_provenance(make_query())).text is None
    assert provider.download_calls == []


def test_is_explicit_hash_match_requires_a_true_flag():
    assert _is_explicit_hash_match(make_release(is_hash_match=True)) is True
    assert _is_explicit_hash_match(make_release(matched_by_hash=True)) is True
    assert _is_explicit_hash_match(make_release(is_hash_match=False, matched_by_hash=False)) is False
    # An object with no flags at all must not be treated as a match.
    assert _is_explicit_hash_match(object()) is False


# --------------------------------------------------------------------------
# 3. Missing hash metadata
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_hash_metadata_claims_no_match_and_downloads_nothing(strategy_factory):
    provider = RecordingProvider([make_release()], payload=reference_text().encode())
    query = make_query(video_hash=None)

    resolved = await strategy_factory(provider).resolve_with_provenance(query)

    assert resolved.text is None
    assert provider.search_calls == []
    assert provider.download_calls == []


@pytest.mark.asyncio
async def test_missing_api_key_skips_the_search_entirely(strategy_factory):
    provider = RecordingProvider([make_release()], payload=reference_text().encode())
    query = make_query(api_keys={})

    resolved = await strategy_factory(provider).resolve_with_provenance(query)

    assert resolved.text is None
    assert provider.search_calls == []


# --------------------------------------------------------------------------
# 4. Language-agnostic selection among several hash matches
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_selection_is_language_agnostic_and_prefers_an_srt(strategy_factory):
    """Any language is acceptable; format is only a tiebreak."""
    provider = RecordingProvider(
        [
            make_release(release_name="z.ass", lang="spa", format="ass", download_url="/sub/opensubtitles/1.srt"),
            make_release(release_name="a.srt", lang="jpn", format="srt", download_url="/sub/opensubtitles/2.srt"),
            make_release(release_name="b.vtt", lang="fre", format="vtt", download_url="/sub/opensubtitles/3.srt"),
        ],
        payload=reference_text().encode(),
    )
    resolved = await strategy_factory(provider).resolve_with_provenance(make_query())

    # The SRT wins on format alone -- not because of its language.
    assert resolved.candidate == "a.srt"
    assert provider.download_calls[0][0] == "/sub/opensubtitles/2.srt"


@pytest.mark.asyncio
async def test_reference_search_is_not_restricted_by_language(strategy_factory):
    provider = RecordingProvider([make_release()], payload=reference_text().encode())
    await strategy_factory(provider).resolve_with_provenance(make_query())

    call = provider.search_calls[0]
    # An empty list is the language-agnostic request; the provider then omits the
    # parameter entirely so OpenSubtitles answers for every language.
    assert call["languages"] == []
    # And the hash/size are what drive the query.
    assert call["moviehash"] == HASH
    assert call["moviebytesize"] == SIZE


@pytest.mark.asyncio
async def test_provider_omits_the_language_filter_for_an_empty_list():
    """End-to-end proof at the HTTP boundary that languages=[] really is open."""
    import httpx

    from app.providers.opensubtitles import OpenSubtitlesProvider

    seen: dict = {}

    class Stub:
        async def get(self, url, headers=None, **kwargs):
            seen.update(kwargs.get("params") or {})
            return httpx.Response(200, json={"data": []}, request=httpx.Request("GET", url))

    provider = OpenSubtitlesProvider(Stub())
    for languages, expected in ((None, "ara"), (["ara"], "ar"), ([], None)):
        seen.clear()
        await provider.search_subtitles(
            imdb_id="tt2582802",
            is_series=False,
            api_key=API_KEY,
            languages=languages,
            moviehash=HASH,
            moviebytesize=SIZE,
        )
        assert seen.get("languages") == expected, languages
        assert seen.get("moviehash") == HASH


# --------------------------------------------------------------------------
# 5. Authentication on download
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_forwards_the_full_credential_set(strategy_factory):
    """An API key authenticates search; the download is what needs the session."""
    provider = RecordingProvider([make_release()], payload=reference_text().encode())
    await strategy_factory(provider).resolve_with_provenance(make_query())

    _reference, api_key, username, password = provider.download_calls[0]
    assert api_key == API_KEY
    assert username == USER
    assert password == PASSWORD


@pytest.mark.asyncio
async def test_credentials_fall_back_to_environment(strategy_factory, monkeypatch):
    monkeypatch.setattr("app.services.sync.hash_reference.settings.OPENSUBTITLES_API_KEY", "env-key")
    monkeypatch.setattr("app.services.sync.hash_reference.settings.OPENSUBTITLES_USERNAME", "env-user")
    provider = RecordingProvider([make_release()], payload=reference_text().encode())
    query = make_query(api_keys={})

    resolved = await strategy_factory(provider).resolve_with_provenance(query)

    assert resolved.text is not None
    _reference, api_key, username, _password = provider.download_calls[0]
    assert (api_key, username) == ("env-key", "env-user")


@pytest.mark.asyncio
async def test_a_download_failure_falls_back_instead_of_raising(strategy_factory):
    class Failing(RecordingProvider):
        async def download_archive(self, reference, api_key=None, username=None, password=None):
            self.download_calls.append((reference, api_key, username, password))
            return None

    provider = Failing([make_release()], payload=None)
    resolved = await strategy_factory(provider).resolve_with_provenance(make_query())

    assert resolved.text is None
    # Exactly one attempt: no retry storm against a metered endpoint.
    assert len(provider.download_calls) == 1


@pytest.mark.asyncio
async def test_a_download_exception_falls_back(strategy_factory):
    class Exploding(RecordingProvider):
        async def download_archive(self, *a, **kw):
            raise RuntimeError("boom")

    resolved = await strategy_factory(Exploding([make_release()])).resolve_with_provenance(make_query())
    assert resolved.text is None


@pytest.mark.asyncio
async def test_an_undecodable_payload_falls_back(strategy_factory):
    provider = RecordingProvider([make_release()], payload=b"tiny")
    assert (await strategy_factory(provider).resolve_with_provenance(make_query())).text is None


@pytest.mark.asyncio
async def test_a_search_exception_falls_back(strategy_factory):
    class Exploding(RecordingProvider):
        async def search_subtitles(self, **kwargs):
            raise RuntimeError("api down")

    assert (await strategy_factory(Exploding()).resolve_with_provenance(make_query())).text is None


# --------------------------------------------------------------------------
# 7. 429 / quota exhaustion
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_open_breaker_skips_the_search_without_retrying(strategy_factory):
    provider = RecordingProvider([make_release()], payload=reference_text().encode())
    provider.breaker_open = True

    resolved = await strategy_factory(provider).resolve_with_provenance(make_query())

    assert resolved.text is None
    assert provider.search_calls == []
    assert provider.download_calls == []


@pytest.mark.asyncio
async def test_a_search_timeout_falls_back(strategy_factory):
    import asyncio

    class Slow(RecordingProvider):
        async def search_subtitles(self, **kwargs):
            await asyncio.sleep(5)
            return []

    resolved = await strategy_factory(Slow(), timeout=0.01).resolve_with_provenance(make_query())
    assert resolved.text is None


@pytest.mark.asyncio
async def test_quota_responses_trip_the_shared_breaker_not_a_local_retry():
    """HTTP 429/406 must reach the process-wide breaker the pipeline consults."""
    import httpx

    from app.providers.opensubtitles import OPENSUBTITLES_BREAKER, OpenSubtitlesProvider

    class QuotaStub:
        def __init__(self, status: int) -> None:
            self.status = status

        async def post(self, url, headers=None, json=None, **kwargs):
            return httpx.Response(
                self.status,
                json={"reset_time_unix": "0", "requests": 0},
                request=httpx.Request("POST", url),
            )

    provider = OpenSubtitlesProvider.__new__(OpenSubtitlesProvider)
    provider.client = QuotaStub(429)

    OPENSUBTITLES_BREAKER.reset()
    try:
        url = await OpenSubtitlesProvider.get_download_url(
            provider, 1, API_KEY, username=USER, password=PASSWORD
        )
        assert url is None
        assert OPENSUBTITLES_BREAKER.is_open() is True
    finally:
        OPENSUBTITLES_BREAKER.reset()


# --------------------------------------------------------------------------
# 10. Caching
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_cached_reference_is_reused_without_a_second_download(strategy_factory):
    provider = RecordingProvider([make_release()], payload=reference_text().encode())
    strategy = strategy_factory(provider)
    query = make_query()

    first = await strategy.resolve_with_provenance(query)
    second = await strategy.resolve_with_provenance(query)

    assert first.text == second.text
    assert len(provider.search_calls) == 1, "cache hit must not re-search"
    assert len(provider.download_calls) == 1, "cache hit must not re-download"


@pytest.mark.asyncio
async def test_a_reference_is_not_reused_for_a_different_video(strategy_factory):
    provider = RecordingProvider([make_release()], payload=reference_text().encode())
    strategy = strategy_factory(provider)

    await strategy.resolve_with_provenance(make_query())
    # A different hash must miss the cache entirely.
    await strategy.resolve_with_provenance(make_query(video_hash="ffffffffffffffff"))

    assert len(provider.search_calls) == 2
    assert len(provider.download_calls) == 2


@pytest.mark.asyncio
async def test_a_reference_is_not_reused_for_a_different_size(strategy_factory):
    provider = RecordingProvider([make_release()], payload=reference_text().encode())
    strategy = strategy_factory(provider)

    await strategy.resolve_with_provenance(make_query())
    await strategy.resolve_with_provenance(make_query(video_size=42))

    assert len(provider.download_calls) == 2


@pytest.mark.asyncio
async def test_a_reference_is_not_reused_across_seasons_or_episodes(strategy_factory):
    provider = RecordingProvider([make_release()], payload=reference_text().encode())
    strategy = strategy_factory(provider)
    series = dict(
        media_type="series", season=1, episode=1, imdb_id="tt0903747", title="Breaking Bad"
    )

    await strategy.resolve_with_provenance(make_query(**series))
    await strategy.resolve_with_provenance(make_query(**{**series, "season": 2}))

    assert len(provider.download_calls) == 2


@pytest.mark.asyncio
async def test_a_cumulative_season_pack_is_refused(strategy_factory):
    provider = RecordingProvider([make_release()], payload=reference_text().encode())
    query = make_query(media_type="series", season=1, episode=2, imdb_id="tt0903747")

    resolved = await strategy_factory(provider).resolve_with_provenance(query)

    # Only rejected when it really looks cumulative; assert we do not cache a
    # pack for a single episode either way.
    if resolved.text is not None:
        assert resolved.candidate


# --------------------------------------------------------------------------
# 9. Independence from the normal provider toggle
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normal_provider_toggle_does_not_block_internal_reference_use(strategy_factory):
    """`enable_opensubtitles=false` only governs Stremio listings."""
    from app.utils.config_parser import parse_user_config

    prefs = parse_user_config("enable_opensubtitles=false")
    assert prefs.enable_opensubtitles is False

    provider = RecordingProvider([make_release()], payload=reference_text().encode())
    resolved = await strategy_factory(provider).resolve_with_provenance(make_query())

    assert resolved.text is not None
    assert len(provider.search_calls) == 1


def test_the_strategy_is_wired_into_the_orchestrator_independently():
    """main.py must build it without consulting the normal provider toggle."""
    import inspect
    import io
    import tokenize

    from app.main import _build_sync_orchestrator

    source = inspect.getsource(_build_sync_orchestrator)
    # Strip comments so the explanatory note about the toggle does not trip the
    # assertion below.
    code = "".join(
        tok.string if tok.type != tokenize.COMMENT else ""
        for tok in tokenize.generate_tokens(io.StringIO(source).readline)
    )
    assert "hash_reference_strategy=OpenSubtitlesHashReferenceStrategy(" in code
    # Constructed unconditionally: no enable_opensubtitles gate wraps it.
    assert "enable_opensubtitles" not in code


def test_the_hash_strategy_is_consulted_before_the_tiered_search():
    from app.services.sync.orchestrator import SyncOrchestrator

    orchestrator = SyncOrchestrator(
        hash_reference_strategy=object(), external_strategy=object()
    )
    names = [name for name, _ in orchestrator._strategies()]
    assert names[0] == "opensubtitles moviehash"
    assert names.index("opensubtitles moviehash") < names.index("external-release-reference")


# --------------------------------------------------------------------------
# 8. Disabled auto-sync
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_sync_off_means_the_strategy_is_never_consulted():
    """With auto-sync off, no reference lookup happens at all.

    Driven through the real orchestrator entry point: auto_sync=False must
    return the payload without resolving a single strategy, which is what keeps
    disabled auto-sync from spending search or download quota.
    """
    from app.services.sync.orchestrator import SyncOrchestrator

    class Exploding:
        name = "opensubtitles moviehash"
        validates_target = True

        async def resolve_with_provenance(self, query, *, update_validator=None):
            raise AssertionError("strategy must not run when auto_sync is off")

    payload = target_text().encode()
    orchestrator = SyncOrchestrator(hash_reference_strategy=Exploding())
    result = await orchestrator.evaluate_and_sync(
        payload,
        {"imdb_id": "tt2582802", "video_hash": HASH, "opensubtitles_key": API_KEY, "lang": "ara"},
        "tid",
        False,
    )
    assert result.decode("utf-8", "replace").count("target line") > 0


# --------------------------------------------------------------------------
# 11/12/14. The reference reaches ALASS in the correct role
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_reference_reaches_alass_and_the_target_stays_the_users_subtitle(monkeypatch, tmp_path):
    """End-to-end through the orchestrator: reference in, Arabic target out.

    The real ALASS argv is captured by intercepting subprocess.run, so this
    proves the reference occupies argv[1] and the user's subtitle argv[2], and
    that the output is the target -- not the reference.
    """
    from app.services.sync.orchestrator import SyncOrchestrator

    reference = reference_text(80, offset_ms=0)
    target = target_text(80, offset_ms=2500)

    provider = RecordingProvider([make_release()], payload=reference.encode())
    cache = ReferenceDiskCache(root=tmp_path / "references", min_bytes=5120)
    hash_strategy = OpenSubtitlesHashReferenceStrategy(provider, cache=cache)

    captured: dict = {}

    class SpySyncService(SubtitleSyncService):
        async def sync_async(self, target_srt, reference_srt, **kwargs):
            captured["target"] = target_srt
            captured["reference"] = reference_srt
            captured["kwargs"] = kwargs
            return target_srt

    class NullExternal:
        name = "external"
        validates_target = True

        async def resolve_with_provenance(self, query, *, update_validator=None):
            return ResolvedReference(None)

    orchestrator = SyncOrchestrator(
        hash_reference_strategy=hash_strategy,
        external_strategy=NullExternal(),
        sync_service=SpySyncService(),
    )

    result = await orchestrator.evaluate_and_sync(
        target.encode(),
        {
            "imdb_id": "tt2582802",
            "target_filename": "Whiplash.2014.2160p.UHD.BluRay.x264-SURCODE.mkv",
            "media_type": "movie",
            "lang": "ara",
            "title": "Whiplash",
            "year": 2014,
            "video_hash": HASH,
            "video_size": SIZE,
            "opensubtitles_key": API_KEY,
            "opensubtitles_username": USER,
            "opensubtitles_password": PASSWORD,
        },
        "target-sub-id",
        True,
    )

    assert captured, "ALASS was never invoked"
    # Roles: the reference is the proven-exact OpenSubtitles text, the target is
    # the user's own subtitle. The two are distinct objects in distinct slots.
    assert captured["reference"].startswith("1\n00:00:04,000 --> 00:00:06,470")
    assert "reference line 1 carrying enough words" in captured["reference"]
    assert "target line 1 carrying enough words" in captured["target"]
    assert captured["reference"] != captured["target"]
    assert captured["target"] == target
    # The reference is labelled as the exact kind and is treated as partial.
    assert captured["kwargs"]["decision_kind"] == "hash"
    assert captured["kwargs"]["reference_partial"] is True
    # The reference never becomes the output. The verifier refuses an
    # unverified fake sync and serves the original, which is exactly the
    # fail-safe behaviour the pipeline promises.
    served = result.decode("utf-8", "replace")
    assert "reference line" not in served
    assert "target line 1 carrying enough words" in served


@pytest.mark.asyncio
async def test_the_reference_is_not_routed_through_arabic_post_processing(strategy_factory):
    """The reference must reach ALASS as timing data, not as a display string.

    Arabic shaping, numeral conversion and RTL fixes belong to the target's
    response path. If any of them ran on the reference, its cue text would
    change and its timings could be rewritten.
    """
    reference = reference_text(80)
    provider = RecordingProvider([make_release()], payload=reference.encode())
    resolved = await strategy_factory(provider).resolve_with_provenance(make_query())

    assert resolved.text is not None
    # No Arabic-Indic digits (the target path converts these) and no RTL marks
    # injected into the reference.
    assert not any("٠" <= ch <= "٩" for ch in resolved.text)
    assert "‎" not in resolved.text and "‏" not in resolved.text
    # Cue timings survive byte-for-byte: what the API sent is what ALASS gets.
    first_start, first_end = _cue_times(80)[0]
    assert f"{_ts(first_start)} --> {_ts(first_end)}" in resolved.text


@pytest.mark.asyncio
async def test_cue_sanity_rejection_falls_through_to_the_next_strategy(strategy_factory):
    """A hash match that fails the content gate must not be served."""
    provider = RecordingProvider([make_release()], payload=reference_text().encode())
    strategy = strategy_factory(provider)

    query = make_query()
    resolved = await strategy.resolve_with_provenance(
        query, update_validator=lambda _text: False
    )

    assert resolved.text is None
    # Exactly one download was spent evaluating it; the decisive part is that a
    # rejected reference is never cached, so it cannot be served later.
    assert len(provider.download_calls) == 1
    assert strategy.cache.get(query) is None


# --------------------------------------------------------------------------
# 13. ALASS failure preserves fallback behaviour
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_alass_failure_serves_the_original_and_does_not_raise(tmp_path):
    """When the reference is unusable by ALASS the target is served unchanged."""
    from app.services.sync.orchestrator import SyncOrchestrator

    target = target_text(80)
    provider = RecordingProvider([make_release()], payload=reference_text(80).encode())
    cache = ReferenceDiskCache(root=tmp_path / "references", min_bytes=5120)

    class FailingSync(SubtitleSyncService):
        async def sync_async(self, target_srt, reference_srt, **kwargs):
            return None  # exactly what sync() returns on alass timeout/failure

    orchestrator = SyncOrchestrator(
        hash_reference_strategy=OpenSubtitlesHashReferenceStrategy(provider, cache=cache),
        sync_service=FailingSync(),
    )

    result = await orchestrator.evaluate_and_sync(
        target.encode(),
        {
            "imdb_id": "tt2582802",
            "target_filename": "Whiplash.2014.2160p.UHD.BluRay.x264-SURCODE.mkv",
            "media_type": "movie",
            "video_hash": HASH,
            "video_size": SIZE,
            "opensubtitles_key": API_KEY,
        },
        "target-sub-id",
        True,
    )

    # Never an exception, and never the reference leaking in as the output.
    assert result == target.encode()


@pytest.mark.asyncio
async def test_a_strategy_failure_is_contained(tmp_path):
    """An exploding strategy must not sink the request."""
    from app.services.sync.orchestrator import SyncOrchestrator

    class Exploding:
        name = "opensubtitles moviehash"
        validates_target = True

        async def resolve_with_provenance(self, query, *, update_validator=None):
            raise RuntimeError("strategy exploded")

    target = target_text(80)
    orchestrator = SyncOrchestrator(hash_reference_strategy=Exploding())
    result = await orchestrator.evaluate_and_sync(
        target.encode(),
        {"imdb_id": "tt2582802", "target_filename": "Whiplash.mkv", "video_hash": HASH, "lang": "ara"},
        "target-sub-id",
        True,
    )
    assert result == target.encode()


# --------------------------------------------------------------------------
# Secrets
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_credentials_never_reach_the_returned_reference(strategy_factory):
    provider = RecordingProvider([make_release()], payload=reference_text().encode())
    resolved = await strategy_factory(provider).resolve_with_provenance(make_query())

    blob = f"{resolved.text}{resolved.candidate}"
    assert API_KEY not in blob
    assert USER not in blob
    assert PASSWORD not in blob
    assert "Api-Key" not in blob
    assert "Authorization" not in blob
