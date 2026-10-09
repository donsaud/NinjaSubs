"""Deterministic trust regressions; mocked alignment is not REAL evidence."""

from types import SimpleNamespace

import pytest

from app.services.sync.alignment import SubtitleEvaluation, SyncState, VerificationAvailability
from app.services.sync.matching import parse_release_identity
from app.services.sync.orchestrator import SyncOrchestrator
from app.services.sync.query import ResolvedReference
from app.services.sync.target_binding import TargetBinding, classify_target_binding
from app.services.sync.timeline_affine import AffineResult
from app.services.sync_cache import SYNC_PAYLOAD_VERSION, SYNC_VERDICT_ENGINE_VERSION, SyncCache

TARGET = "The Great Beauty (2013) Criterion (1080p BluRay x265 10bit SARTRE)"
PUBLICHD = "The.Great.Beauty.2013.720p.BluRay.DTS.x264-PublicHD.srt"
CRITERION = PUBLICHD.replace("2013.", "2013.Criterion.Collection.")


def test_real_failure_release_metadata():
    target, reference = parse_release_identity(TARGET), parse_release_identity(PUBLICHD)
    assert target.edition_tags == ("criterion_collection",)
    assert (target.source_family, target.resolution, target.codec, target.bit_depth,
            target.release_group) == ("bluray", "1080p", "h265", 10, "SARTRE")
    assert (reference.source_family, reference.resolution, reference.codec,
            reference.release_group, reference.edition_tags) == (
                "bluray", "720p", "h264", "PublicHD", ())


@pytest.mark.parametrize("label, tag, field", [
    ("Criterion", "criterion_collection", "edition_tags"),
    ("Criterion Collection", "criterion_collection", "edition_tags"),
    ("Special Edition", "special_edition", "edition_tags"),
    ("Collector's Edition", "collectors_edition", "edition_tags"),
    ("Collectors Edition", "collectors_edition", "edition_tags"),
    ("Anniversary Edition", "anniversary_edition", "edition_tags"),
    ("Remastered", "remastered", "edition_tags"),
    ("Remaster", "remastered", "edition_tags"),
    ("Restored", "restored", "edition_tags"),
    ("Restoration", "restored", "edition_tags"),
    ("IMAX", "imax", "edition_tags"),
    ("Open Matte", "open_matte", "edition_tags"),
    ("Director's Cut", "directors_cut", "cut_tags"),
    ("Directors Cut", "directors_cut", "cut_tags"),
    ("Extended Cut", "extended", "cut_tags"),
    ("Extended Edition", "extended", "cut_tags"),
    ("Extended", "extended", "cut_tags"),
    ("Theatrical Cut", "theatrical", "cut_tags"),
    ("Unrated", "unrated", "cut_tags"),
    ("Uncut", "uncut", "cut_tags"),
    ("Final Cut", "final_cut", "cut_tags"),
    ("International Cut", "international_cut", "cut_tags"),
    ("International Version", "international_cut", "cut_tags"),
])
def test_word_aware_aliases(label, tag, field):
    metadata = parse_release_identity(f"Film.2020.{label}.1080p.BluRay.x264-GRP")
    assert tag in getattr(metadata, field)


@pytest.mark.parametrize("label", ["CC", "Special", "International", "open", "criterionish"])
def test_no_broad_aliases(label):
    parsed = parse_release_identity(f"Film.2020.{label}.1080p.BluRay.x264-GRP")
    assert parsed.edition_tags == parsed.cut_tags == ()


def test_supporting_metadata_is_not_cut_identity():
    parsed = parse_release_identity("Film.2020.UHD.BluRay.REMUX.2160p.HEVC.10bit.HDR10+.DV-GRP [1h46m57m]")
    assert parsed.source_family == "uhd_bluray" and parsed.remux
    assert parsed.resolution == "2160p" and parsed.bit_depth == 10
    assert "hdr10_plus" in parsed.hdr_flags and "dolby_vision" in parsed.hdr_flags
    assert parsed.duration_hint_seconds == 6417


@pytest.mark.parametrize("source, expected", [("BluRay", "bluray"), ("WEB-DL", "webdl"),
                                               ("WEBRip", "webrip"), ("HDTV", "hdtv"), ("DVD", "dvd")])
def test_source_families(source, expected):
    assert parse_release_identity(f"Film.2020.1080p.{source}.x264-GRP").source_family == expected


def test_great_beauty_binding():
    assert classify_target_binding(TARGET, PUBLICHD).state is TargetBinding.UNKNOWN
    assert classify_target_binding(TARGET, CRITERION).state is TargetBinding.STRONG


def test_explicit_cut_conflict():
    a = "Film.2020.Directors.Cut.1080p.BluRay.x265-GRP"
    b = "Film.2020.Theatrical.Cut.720p.BluRay.x264-GRP"
    assert classify_target_binding(a, b).state is TargetBinding.CONFLICT


def test_good_framestor_identity_ignores_encode_differences():
    a = "Whiplash.2014.1080p.BluRay.REMUX-FraMeSToR"
    b = "Whiplash.2014.2160p.UHD.BluRay.x265-FraMeSToR.srt"
    assert classify_target_binding(a, b).state is TargetBinding.STRONG
    assert classify_target_binding(a, b, exact_moviehash=True).state is TargetBinding.EXACT
    assert classify_target_binding(None, None, embedded=True).state is TargetBinding.EXACT


@pytest.mark.parametrize("a,b", [(None, None), ("Film", "Film"),
    ("Film.2020.BluRay.x265-GRP", "Film.2020.BluRay.x264-OTHER"),
    ("Film.2020.BluRay.x265-GRP", "Other.2020.BluRay.x265-GRP")])
def test_missing_or_weak_metadata_never_strong(a, b):
    assert not classify_target_binding(a, b).supports_verified


def text(tag):
    return "\n".join(f"{i+1}\n00:00:{i*2:02},000 --> 00:00:{i*2+1:02},000\n{tag} {i}\n"
                     for i in range(20))


@pytest.mark.asyncio
@pytest.mark.parametrize("has_strong", [True, False])
async def test_great_beauty_lazy_progression_and_fallback(monkeypatch, has_strong):
    from app.config import settings
    from app.services.sync import timeline_affine

    monkeypatch.setattr(settings, "ENABLE_SUBTITLE_SYNC", True)
    cache = SyncCache()
    seen = []
    releases = [PUBLICHD, CRITERION] if has_strong else [PUBLICHD, PUBLICHD.replace("PublicHD", "OTHER")]

    class External:
        cache = None

        async def plan_candidates(self, query):
            return [(release, self) for release in releases]

        async def download_planned_candidate(self, release, provider, query, validate, shared):
            seen.append(release)
            return ResolvedReference(text(release), kind="edition", candidate=release)

    def accepted(*args):
        return AffineResult(accepted=True, slope=0.9984844145727436,
                            intercept_ms=-33498.60620349459, transformed_text=text("transformed"),
                            evaluation=SubtitleEvaluation(sync_state=SyncState.VERIFIED_RESYNCED,
                                verification=VerificationAvailability.VERIFIED, coverage_score=0.9950))

    monkeypatch.setattr(timeline_affine, "attempt_timeline_affine", accepted)

    async def no_alass(*args, **kwargs):
        pytest.fail("accepted affine with insufficient binding must progress without ALASS")

    orch = SyncOrchestrator(external_strategy=External(), sync_cache=cache,
                            sync_service=SimpleNamespace(sync_async=no_alass))
    original = text("original").encode()
    meta = {"imdb_id": "tt2358891", "lang": "ara", "target_filename": TARGET,
            "video_size": 7626728974}
    output = await orch.evaluate_and_sync(original, meta, "8f76cbc1dcd2fd2e", auto_sync=True)
    assert seen == releases
    if has_strong:
        assert output == text("transformed").encode()
        assert orch._last_evaluation.target_binding == "strong"
        key = orch._flight_key(meta, "8f76cbc1dcd2fd2e", original)
        assert (await cache.get_meta(key))["target_binding"] == "strong"
        assert await orch.evaluate_and_sync(original, meta, "8f76cbc1dcd2fd2e", auto_sync=True) == output
        assert seen == releases
    else:
        assert output == original
        assert not cache._local and not cache._local_verdict


@pytest.mark.asyncio
async def test_cache_semantic_version_and_binding_safety():
    cache = SyncCache()
    key = cache.build_key("tt1", "movie", "video", "sub", "edition")
    assert SYNC_PAYLOAD_VERSION in key
    assert SYNC_VERDICT_ENGINE_VERSION == 7
    cache._local["final_sub:tt1:movie:video:sub:edition"] = b"old"
    assert await cache.find_synced_for_target("tt1", "movie", "video", "sub") is None
    alias = cache.build_verdict_alias_key("video", "sub")
    old = {"engine_version": 3, "sync_state": "verified_resynced",
           "verification": "verified", "target_binding": "strong"}
    import json
    cache._local_verdict[alias] = json.dumps(old)
    assert await cache.get_verdict_by_ref("video", "sub") is None
    for binding in ("weak", "unknown", "conflict", None):
        await cache.set_verdict("new", {**old, "target_binding": binding})
        assert await cache.get_verdict("new") is None
    await cache.set_verdict("new", {**old, "target_binding": "strong"}, alias_key=alias)
    assert (await cache.get_verdict_by_ref("video", "sub"))["verification"] == "cached"


def test_no_hash_cannot_claim_exact():
    assert SyncOrchestrator._target_binding(ResolvedReference("ref", kind="hash"), {}).state is not TargetBinding.EXACT


@pytest.mark.parametrize("name", ["Movie (2020) (Arbitrary Words)",
    "Movie (2020) (1080p BluRay x265 10bit)", "Movie (2020) (1080p BluRay x265 Special Edition)"])
def test_no_guessed_group_from_arbitrary_words(name):
    assert parse_release_identity(name).release_group is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["alass", "already_aligned"])
@pytest.mark.parametrize("has_strong", [False, True])
async def test_legacy_paths_cannot_bypass_target_binding(monkeypatch, mode, has_strong):
    from app.config import settings
    from app.services.sync import orchestrator as module
    from app.services.sync import timeline_affine

    monkeypatch.setattr(settings, "ENABLE_SUBTITLE_SYNC", True)
    monkeypatch.setattr(timeline_affine, "attempt_timeline_affine", lambda *a: AffineResult())
    consistency = SimpleNamespace(aligned=True, median_offset_s=0, p95_ms=0,
                                  sections_agreeing=5, sections_measured=5, coverage=1)
    monkeypatch.setattr(module, "measure_alignment_consistency",
                        lambda *a, **kw: consistency if mode == "already_aligned" else None)
    releases = [PUBLICHD, CRITERION] if has_strong else [PUBLICHD]
    seen = []

    class External:
        cache = None

        async def plan_candidates(self, query):
            return [(release, self) for release in releases]

        async def download_planned_candidate(self, release, provider, query, validate, shared):
            seen.append(release)
            return ResolvedReference(text("reference"), kind="edition", candidate=release)

    async def alass(*a, **kw):
        return text("transformed")

    cache = SyncCache()
    orch = SyncOrchestrator(external_strategy=External(), sync_cache=cache,
                            sync_service=SimpleNamespace(sync_async=alass))
    orch._analyzer = SimpleNamespace(analyze=lambda *a, **kw: SubtitleEvaluation(
        sync_state=SyncState.VERIFIED_RESYNCED if mode == "alass" else SyncState.VERIFIED_SYNCED,
        verification=VerificationAvailability.VERIFIED))
    original = text("original").encode()
    meta = {"imdb_id": "tt2358891", "lang": "ara", "target_filename": TARGET}
    output = await orch.evaluate_and_sync(original, meta, "target", auto_sync=True)
    assert seen == releases
    expected = text("transformed").encode() if mode == "alass" and has_strong else original
    assert output == expected
    for stored in cache._local_verdict.values():
        import json
        verdict = json.loads(stored)
        if verdict["sync_state"].startswith("verified"):
            assert verdict["target_binding"] == "strong"


def test_pal_promotion_and_large_offset_unverified_state():
    binding = classify_target_binding(TARGET, PUBLICHD)
    verified = SubtitleEvaluation(sync_state=SyncState.VERIFIED_RESYNCED,
                                  verification=VerificationAvailability.VERIFIED)
    assert not SyncOrchestrator._gate_target_binding(verified, binding)
    assert verified.sync_state is SyncState.UNVERIFIED
    # Existing unverified Large Offset decisions retain their measured verdict;
    # the separate delivery exception is covered by test_large_offset_serving.
    legacy = SubtitleEvaluation(sync_state=SyncState.UNVERIFIED,
                                verification=VerificationAvailability.UNKNOWN, p95_offset_ms=2500)
    assert not SyncOrchestrator._gate_target_binding(legacy, binding)
    assert legacy.sync_state is SyncState.UNVERIFIED and legacy.p95_offset_ms == 2500
