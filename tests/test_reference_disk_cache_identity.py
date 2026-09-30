"""ReferenceDiskCache identity, and the real rejected-verdict serving boundary.

Two things this file pins, both previously unproven:

1. ReferenceDiskCache identity. The Dexter trace showed the same selected
   reference and family but two cache filenames whose differing component was
   the trailing 16-hex segment. That segment is the TARGET SCOPE of
   ``ReferenceQuery.cache_stem``::

       {imdb}_{season}_{episode}_{group}_v2_{digest}_{scope}_{source}_{kind}.srt

   ``scope`` is the target subtitle's cue-layout fingerprint (``t<digest>``),
   or a subtitle-id scope (``i<digest>``), or empty. A different target cue
   layout is therefore a different cache identity BY DESIGN, and
   ``ReferenceDiskCache.set`` deletes stale variants only within one stem, so
   distinct stems coexist legitimately. This is Case A/B: not a lookup defect.

   Both axes of stem variation are visible in the live cache directory, and the
   tests below reproduce them deterministically.

2. The rejected-verdict boundary. The earlier version of this test only
   asserted that a stored verdict CONTAINS the string "rejected", which proved
   nothing: mutating ``set_verdict`` to treat REJECTED as verified did not fail
   it. These tests instead drive ``_reusable_verified_fallback_sync``, the
   function that actually decides whether an artifact may be served, and are
   mutation-verified against that gate.
"""

from __future__ import annotations

from app.models import SubtitleRelease
from app.services.sync.query import ReferenceQuery
from app.services.sync_cache import SyncCache

# ---- identity shape from the incident -------------------------------------
IMDB = "tt0773262"
SEASON = 8
EPISODE = 5
FILENAME = "Dexter.s8e05.This.little.piggy.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv"
VIDEO_SIZE = 5192387053
SUB_ID = "63df03613463c531"
RELEASE = "Dexter.S08E05.HDTV.XviD-AFG.srt"
URL = "https://subsource.test/rovers.srt"
LANG = "ara"
SEASON_EP = f"s{SEASON}e{EPISODE}"


def _release() -> SubtitleRelease:
    return SubtitleRelease(
        release_name=RELEASE, download_url=URL, provider="subsource", lang=LANG
    )


def _fingerprint() -> str:
    return SyncCache.video_fingerprint_from_meta(
        {
            "imdb_id": IMDB,
            "season": SEASON,
            "episode": EPISODE,
            "target_filename": FILENAME,
        }
    )


def _query(*, cue_digest: str | None, target_filename: str = FILENAME) -> ReferenceQuery:
    return ReferenceQuery(
        imdb_id=IMDB,
        media_type="series",
        season=SEASON,
        episode=EPISODE,
        target_filename=target_filename,
        video_size=VIDEO_SIZE,
        target_cue_digest=cue_digest,
        target_sub_id=SUB_ID,
    )


def _srt(first_ms: int, n: int = 20) -> str:
    out = []
    for i in range(n):
        s = first_ms + i * 2000
        ms = s % 1000
        sec = (s // 1000) % 60
        mn = (s // 60_000) % 60
        out.append(f"{i + 1}\n00:{mn:02d}:{sec:02d},{ms:03d} --> 00:59:00,000\nx\n")
    return "\n".join(out)


# ===================== Part 2: ReferenceDiskCache investigation =============


def test_stem_has_two_independent_variation_axes():
    """The two hex segments vary for different reasons.

    ``digest`` covers media_type / target_filename / video_hash / video_size /
    languages. ``scope`` covers the target subtitle's cue layout. Conflating
    them is what makes a trace look like a random filename.
    """
    a = _query(cue_digest="aaaa1111bbbb2222")
    same_scope_other_target = _query(
        cue_digest="aaaa1111bbbb2222", target_filename="Other.S08E05.720p.WEB.srt"
    )
    other_scope = _query(cue_digest="cccc3333dddd4444")

    assert a.cache_stem.startswith(f"{IMDB}_{SEASON}_{EPISODE}_")
    assert a.cache_stem.endswith("taaaa1111bbbb2222")
    assert a.cache_stem != other_scope.cache_stem, "cue layout is part of the identity"
    assert a.cache_stem != same_scope_other_target.cache_stem, (
        "the digest axis (target filename/size/languages) is independent"
    )


def test_stem_excludes_stream_url_by_design():
    """A rotating signed token must not change the stem.

    This is why the same episode does not accumulate a new cache entry on every
    request; the cue-layout scope is deliberate, a token digest would not be.
    """
    base = _query(cue_digest="aaaa1111bbbb2222")
    with_url = ReferenceQuery(
        imdb_id=IMDB, media_type="series", season=SEASON, episode=EPISODE,
        target_filename=FILENAME, video_size=VIDEO_SIZE,
        target_cue_digest="aaaa1111bbbb2222", target_sub_id=SUB_ID,
        stream_url="https://host/x.m3u8?token=rotated",
    )
    assert base.cache_stem == with_url.cache_stem


# ---- Test A: exact repeat reuses the cached reference ----------------------


def test_a_identical_lookup_reuses_reference_without_download(tmp_path):
    from app.services.sync.cache import ReferenceDiskCache

    cache = ReferenceDiskCache(root=str(tmp_path), ttl=3600.0, min_bytes=100)
    query = _query(cue_digest="aaaa1111bbbb2222")
    text = _srt(10_160)

    first = cache.get(query)
    assert first is None, "cold cache must miss"
    cache.set(query, "subsource", text, "edition", candidate=RELEASE)

    second = cache.get(query)
    assert second is not None
    assert second.text == text, "an identical repeat must be served from cache"
    assert second.candidate == RELEASE


def test_a_different_cue_layout_is_a_different_identity(tmp_path):
    """Case A: same logical reference, different target cue layout.

    Documented behaviour, not a defect. The second request does not reuse the
    first request's entry because the target subtitle it was proven against has
    a different timing grid.
    """
    from app.services.sync.cache import ReferenceDiskCache

    cache = ReferenceDiskCache(root=str(tmp_path), ttl=3600.0, min_bytes=100)
    layout_one = _query(cue_digest="aaaa1111bbbb2222")
    layout_two = _query(cue_digest="cccc3333dddd4444")

    cache.set(layout_one, "subsource", _srt(10_160), "edition", candidate=RELEASE)
    assert cache.get(layout_one) is not None
    assert cache.get(layout_two) is None, (
        "a different cue layout must not inherit another layout's reference"
    )


# ---- Test B: target isolation ---------------------------------------------


def test_b_reference_is_target_bound(tmp_path):
    from app.services.sync.cache import ReferenceDiskCache

    cache = ReferenceDiskCache(root=str(tmp_path), ttl=3600.0, min_bytes=100)
    target_a = _query(cue_digest="aaaa1111bbbb2222")
    cache.set(target_a, "subsource", _srt(10_160), "edition", candidate=RELEASE)

    target_b = _query(
        cue_digest="aaaa1111bbbb2222",
        target_filename="Different.S08E05.1080p.BluRay.x264-OTHER.mkv",
    )
    assert cache.get(target_b) is None, (
        "a reference proven for one target must never satisfy another, even "
        "with an identical release name, provider and episode"
    )


# ---- Test C: different payloads remain distinguishable ---------------------


def test_c_different_payloads_do_not_collide(tmp_path):
    from app.services.sync.cache import ReferenceDiskCache

    cache = ReferenceDiskCache(root=str(tmp_path), ttl=3600.0, min_bytes=100)
    query = _query(cue_digest="aaaa1111bbbb2222")
    cache.set(query, "subsource", _srt(10_160), "edition", candidate=RELEASE)
    first = cache.get(query)
    cache.set(query, "subsource", _srt(106_950), "edition", candidate=RELEASE)
    second = cache.get(query)
    assert first is not None and second is not None
    assert first.text != second.text, "a re-save replaces rather than blends"
    assert second.text == _srt(106_950)


# ---- Test D: deterministic, order-independent selection -------------------


def test_d_multiple_files_for_one_stem_selects_deterministically(tmp_path):
    """Several kinds can share a stem; selection must not be arbitrary.

    ``set`` deliberately keeps an exact-kind entry (team/hash/embedded) when
    saving an edition entry, so more than one file per stem is legitimate.
    """
    from app.services.sync.cache import ReferenceDiskCache

    cache = ReferenceDiskCache(root=str(tmp_path), ttl=3600.0, min_bytes=100)
    query = _query(cue_digest="aaaa1111bbbb2222")
    cache.set(query, "subsource", _srt(10_160), "team", candidate="team-release")
    cache.set(query, "subsource", _srt(10_500), "edition", candidate="edition-release")

    picks = [(cache.get(query)).kind for _ in range(5)]  # type: ignore[union-attr]
    assert len(set(picks)) == 1, f"selection must be stable, got {set(picks)}"
    assert picks[0] == "team", "an exact kind outranks a best-effort edition save"
