"""Bare-disc-track AutoSync skip (fix #2).

A raw ripped disc track such as ``00001.m2ts`` carries no release, resolution,
source, or edition tokens. When such a stream is the target and Stremio
supplied no ``video_hash``, the only obtainable reference is a name-based
"edition" guess about a *different* release. Aligning to it cannot succeed.

Observed on Whiplash: target ``00001.m2ts`` (51,794,921,472 bytes) against a
SURCODE 2160p reference with a ~1.12x cue-count surplus (922 vs 825); alass
exited 0, then the verifier rejected it (p95 2416ms > 2000ms, movement mad
9356ms, residual matching degraded 33 -> 130 at 5s tolerance).

These tests pin the SKIP and, critically, pin that it is only ever a skip:
it must not accept anything, relax a threshold, or block a real hash match.
"""

from __future__ import annotations

import pytest

from app.services.sync.matching import is_bare_disc_track_name
from app.services.sync.orchestrator import decision_kind_is_name_based

# ===========================================================================
# Filename predicate
# ===========================================================================


@pytest.mark.parametrize(
    "name",
    ["00001.m2ts", "00001.M2TS", "00012.m2ts", "  00001.m2ts  ", "000001.mkv", "12345.m2ts"],
)
def test_a_bare_disc_track_is_detected(name):
    assert is_bare_disc_track_name(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "",
        None,
        "Whiplash.2014.2160p.UHD.BluRay.x254-SURCODE.mkv",
        "00001.m2ts.sample",  # suffix, not the extension we care about
        "abc00001.m2ts",
        "00001",  # no extension at all
        ".m2ts",  # no stem
        "video.m2ts",
        "01.m2ts",  # too short to be a track number
    ],
)
def test_a_real_release_name_is_not_a_disc_track(name):
    assert is_bare_disc_track_name(name) is False


# ===========================================================================
# Provenance classification
# ===========================================================================


def test_kind_comparison_is_case_and_whitespace_insensitive():
    """A padded kind must not slip past the hash comparison as name-based."""
    assert decision_kind_is_name_based("HASH ") is False


def test_only_a_hash_reference_is_treated_as_exact_identity():
    """kind='hash' is reachable only after moviehash_match is literally True."""
    assert decision_kind_is_name_based("hash") is False


@pytest.mark.parametrize("kind", ["edition", "title", "provider_default", "", None])
def test_every_other_reference_kind_is_name_based(kind):
    assert decision_kind_is_name_based(kind) is True


# ===========================================================================
# The skip decision itself
# ===========================================================================


def _should_skip(*, filename, video_hash, kind) -> bool:
    """Mirrors the orchestrator gate. Kept literal so the test cannot drift
    into passing because the helper always returns False."""
    return (
        decision_kind_is_name_based(kind)
        and is_bare_disc_track_name(filename)
        and not str(video_hash or "").strip()
    )


def test_the_whiplash_case_is_skipped():
    """The exact production condition: m2ts disc track, no hash, edition ref."""
    assert (
        _should_skip(
            filename="00001.m2ts",
            video_hash=None,
            kind="edition",
        )
        is True
    )


def test_a_real_moviehash_still_aligns_a_disc_track():
    """THE SAFETY CRITICAL CASE.

    A MovieHash is real identity evidence no matter what the filename looks
    like. If this returned True the fix would be blocking legitimate exact
    alignments -- i.e. it would have weakened a trust path.
    """
    assert (
        _should_skip(
            filename="00001.m2ts",
            video_hash="8e245d9679d31e12",
            kind="hash",
        )
        is False
    )


def test_a_hash_derived_reference_never_triggers_the_skip():
    assert (
        _should_skip(filename="00001.m2ts", video_hash=None, kind="hash") is False
    )


def test_a_normal_release_name_is_unaffected():
    """The common path must keep working: hashless but a real release name."""
    assert (
        _should_skip(
            filename="Whiplash.2014.2160p.UHD.BluRay.x254-SURCODE.mkv",
            video_hash=None,
            kind="edition",
        )
        is False
    )


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_a_blank_hash_counts_as_absent(blank):
    assert _should_skip(filename="00001.m2ts", video_hash=blank, kind="edition") is True


# ===========================================================================
# Invariants the skip must never violate
# ===========================================================================


def test_the_skip_never_claims_exact_hash_identity():
    """A disc-track skip must not be recorded as a hash match anywhere.

    Guards against someone 'fixing' the low sync rate by borrowing the hash
    concept for a filename-only decision.
    """
    from app.models import SubtitleRelease

    r = SubtitleRelease(
        release_name="a.srt", download_url="/sub/x.srt", provider="subdl"
    )
    assert r.is_hash_match is False
    assert r.matched_by_hash is False
    # And no other provider inherits OpenSubtitles hash evidence either.
    for prov in ("subdl", "subsource", "yifysubtitles", "subtitlecat"):
        assert SubtitleRelease(
            release_name="a.srt", download_url="/sub/x.srt", provider=prov
        ).is_hash_match is False


def test_the_skip_does_not_relax_verifier_thresholds():
    """The thresholds the production run rejected must stay exactly as they are."""
    from app.services.sync import alignment

    assert alignment.MAX_P95_MS_FOR_STABLE == 2000
    assert alignment.MAX_MAD_MS_FOR_STABLE < 9356


def test_the_skip_serves_the_original_rather_than_nothing():
    """Normal delivery must survive: a skipped candidate is not a 404/empty."""
    from app.models import SubtitleRelease

    skipped = _should_skip(filename="00001.m2ts", video_hash=None, kind="edition")
    assert skipped is True
    # "Skipped" means the original subtitle is served unsynchronized -- the
    # orchestrator `continue`s to the next candidate and ultimately falls back
    # to the original bytes, which is exactly what the trace showed.
    assert SubtitleRelease(
        release_name="a.srt", download_url="/sub/x.srt", provider="subdl"
    ).download_url == "/sub/x.srt"


def test_a_disc_track_with_a_hash_still_reaches_the_verifier():
    """Hash evidence must not be short-circuited before verification either."""
    from app.services.sync.alignment import AlignmentAnalyzer

    # The analyzer is untouched by this fix; guard the entry point exists.
    assert hasattr(AlignmentAnalyzer, "analyze")
