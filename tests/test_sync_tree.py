"""Tests for the multi-tier edition fallback in the reference decision tree.

Relaxed rules let an external English reference be found even when the release
group differs:

* Tier 0 — exact source (``webdl`` -> ``webdl``), BluRay/REMUX family;
* Tier 1 — cross-format (``webdl`` stream anchored by a BluRay/REMUX reference,
  or a shared explicit ``23.976`` fps fingerprint);
* Tier 2 — relaxed fallback for an unknown source on one side.

DVD/CAM/screener references and mismatched episodes/seasons stay rejected.
"""

from types import SimpleNamespace

import pytest

from app.services.sync.query import ReferenceQuery
from app.services.sync.tree import decide


def _releases(*names: str) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(release_name=name, download_url=f"http://cdn/{i}.srt")
        for i, name in enumerate(names)
    ]


def _query(target: str, *, season: int | None = None, episode: int | None = None):
    media_type = "series" if season is not None else "movie"
    return ReferenceQuery(
        imdb_id="tt1",
        media_type=media_type,
        season=season,
        episode=episode,
        target_filename=target,
    )


def test_same_source_different_group_is_accepted():
    """Tier 0: a WEB-DL reference from another group confirms the edition."""
    target = "Movie.2024.1080p.WEB-DL.DDP5.1.H.264-WADU.mkv"
    decision = decide(
        _releases("Movie.2024.1080p.WEB-DL.DDP5.1.H.264-BTN.srt"),
        _query(target),
        strict=True,
    )
    assert decision.kind == "edition"
    assert "BTN" in decision.release.release_name


def test_same_source_codec_difference_is_accepted():
    """Tier 0 tolerates codec differences on the same source master."""
    target = "Movie.2024.1080p.WEB-DL.x264-WADU.mkv"
    decision = decide(
        _releases("Movie.2024.1080p.WEB-DL.x265-BTN.srt"),
        _query(target),
        strict=True,
    )
    assert decision.kind == "edition"


def test_webdl_target_accepts_bluray_reference():
    """Tier 1: a WEB-DL stream is anchored by a retail-disc reference."""
    target = "Show.S01E01.1080p.WEB-DL.x264-WADU.mkv"
    decision = decide(
        _releases("Show.S01E01.1080p.BluRay.x264-BTN.srt"),
        _query(target, season=1, episode=1),
        strict=True,
    )
    assert decision.kind == "edition"
    assert "BluRay" in decision.release.release_name


def test_webdl_target_accepts_remux_reference():
    target = "Movie.2024.2160p.WEB-DL.x265-WADU.mkv"
    decision = decide(
        _releases("Movie.2024.2160p.BluRay.REMUX.HEVC.DTS-HD.MA5.1-BTN.srt"),
        _query(target),
        strict=True,
    )
    assert decision.kind == "edition"


def test_webdl_target_accepts_23976_cross_format():
    """Tier 1: a shared explicit 23.976 fps fingerprint anchors the offset."""
    target = "Show.S01E01.1080p.WEB-DL.23.976.x264-WADU.mkv"
    decision = decide(
        _releases("Show.S01E01.1080p.HDTV.23.976.x264-BTN.srt"),
        _query(target),
        strict=True,
    )
    assert decision.kind == "edition"


def test_bluray_target_does_not_accept_webdl_reference():
    """The cross-format allowance is directional (WEB-DL target only)."""
    target = "Movie.2024.1080p.BluRay.x264-WADU.mkv"
    decision = decide(
        _releases("Movie.2024.1080p.WEB-DL.x264-BTN.srt"),
        _query(target),
        strict=True,
    )
    assert decision.kind == "abort"


@pytest.mark.parametrize(
    "name",
    [
        "Movie.2024.DVDRip.XviD-BTN.srt",
        "Movie.2024.HDCAM.x264-BTN.srt",
        "Movie.2024.SCREENER.x264-BTN.srt",
    ],
)
def test_dvd_cam_and_screener_are_rejected(name):
    target = "Movie.2024.1080p.WEB-DL.x264-WADU.mkv"
    assert decide(_releases(name), _query(target), strict=True).kind == "abort", name


def test_exact_source_outranks_cross_format():
    """A same-source candidate is preferred over a higher-tier cross-format one."""
    target = "Movie.2024.1080p.WEB-DL.x264-WADU.mkv"
    decision = decide(
        _releases(
            "Movie.2024.1080p.BluRay.x264-BTN.srt",  # tier 1
            "Movie.2024.1080p.WEB-DL.x264-OTHER.srt",  # tier 0
        ),
        _query(target),
        strict=True,
    )
    assert decision.kind == "edition"
    assert "OTHER" in decision.release.release_name


def test_wrong_episode_is_still_rejected_across_tiers():
    target = "Show.S01E01.1080p.WEB-DL.x264-WADU.mkv"
    decision = decide(
        _releases("Show.S01E02.1080p.BluRay.x264-BTN.srt"),
        _query(target, season=1, episode=1),
        strict=True,
    )
    assert decision.kind == "abort"
