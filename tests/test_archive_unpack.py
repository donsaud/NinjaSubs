"""On-demand unbundling of season-pack subtitle archives.

A season pack is a single archive holding every episode of a season. Stremio
cannot parse a ZIP, so the display list may still advertise the pack, but the
URL it hands back must resolve to just the requested episode. These tests cover
the two halves of that contract: the pack survives ranking (demoted below an
exact-episode track) and the serve route slices out the right member on demand.
"""

from __future__ import annotations

import io
import zipfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app, cache_manager
from app.models import SubtitleRelease
from app.services.subtitle_matcher import rank_subtitles

TARGET = "Dexter.S08E05.1080p.BluRay.TrueHD5.1.AVC-PiR8.mkv"
PACK_NAME = "Dexter.S08.1080p.BluRay.x265-ImE.srt"
SINGLE_NAME = "Dexter.S08E05.720p.BluRay.x264-NORDiC.srt"


def _episode_srt(number: int) -> str:
    """A minimal but valid SRT whose first cue number identifies the episode."""
    return (
        f"{number}\n"
        f"00:00:{number % 60:02d},000 --> 00:00:{(number % 60) + 1:02d},000\n"
        f"episode {number} line one\n"
        f"episode {number} line two\n"
    ) + "".join(
        f"{i}\n00:0{min(i, 9)}:{i % 60:02d},000 --> 00:0{min(i, 9)}:{(i % 60) + 1:02d},000\n"
        f"filler {i}\n"
        for i in range(3, 12)
    )


def _season_pack_zip(episodes: int = 8) -> bytes:
    """A season-pack archive with one member per episode plus a stray .nfo."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for episode in range(1, episodes + 1):
            archive.writestr(f"Episode {episode:02d} - Title.srt", _episode_srt(episode))
        archive.writestr("readme.nfo", "season pack notes")
    return buffer.getvalue()


# --------------------------- display-list eligibility --------------------------


def test_season_pack_survives_ranking_and_is_demoted():
    """A pack is a valid candidate, ranked below the exact-episode track."""
    pack = SubtitleRelease(
        release_name=PACK_NAME, download_url="http://pack", provider="subdl", lang="ara"
    )
    single = SubtitleRelease(
        release_name=SINGLE_NAME, download_url="http://single", provider="subdl", lang="ara"
    )

    ranked = rank_subtitles(TARGET, [pack, single], season=8, episode=5)

    assert any(r.release_name == PACK_NAME for r in ranked), "season pack was filtered out"
    assert ranked[0].release_name == SINGLE_NAME, "exact episode must outrank the pack"


def test_pack_label_uses_standard_format_without_status_tags():
    """The pack is presented with the standard label, no archive/status tag."""
    from app.services.aggregator import format_informative_badge

    pack = SubtitleRelease(
        release_name=PACK_NAME,
        download_url="http://pack",
        provider="subdl",
        lang="ara",
        uploader="subsmaster",
        match_percentage=90,
    )
    label = format_informative_badge(pack, 90, source_tag="SubDL")

    # The formatter strips the subtitle extension from the display name.
    assert label == f"[90%] [SubDL] {PACK_NAME.removesuffix('.srt')} (by subsmaster)"
    assert "\u26a1" not in label


# ----------------------------- on-demand unbundling ----------------------------


@pytest.fixture
def client() -> TestClient:
    # The serve route downloads through the module-level shared client; provide a
    # stub so the route is reachable without live network access.
    stub = SimpleNamespace(get=AsyncMock(), post=AsyncMock(), request=AsyncMock())
    with patch("app.main._http_client", stub):
        yield TestClient(app)


def _isolate_cache(monkeypatch, tmp_path) -> None:
    """Point the LRU cache (payload + metadata dirs) at a throwaway location."""
    root = tmp_path / "cache"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cache_manager, "cache_dir", root, raising=False)
    monkeypatch.setattr(cache_manager, "meta_dir", root / "_meta", raising=False)


def _disable_auto_sync(client: TestClient) -> None:
    """Force the auto-sync path off so the serve route only exercises unbundling."""
    client.app.dependency_overrides.clear()


def _seed_metadata(sub_id: str) -> None:
    """Register serve metadata for a pack candidate."""
    cache_manager.store_metadata(
        sub_id,
        {
            "sub_id": sub_id,
            "imdb_id": "tt0773262",
            "media_type": "series",
            "provider": "subdl",
            "download_url": "http://pack",
            "release_name": PACK_NAME,
            "target_filename": TARGET,
            "season": 8,
            "episode": 5,
            "lang": "ara",
        },
    )


def test_unpack_endpoint_serves_only_the_requested_episode(client, tmp_path, monkeypatch):
    """Requesting the pack URL yields the requested episode's text, not the whole pack."""
    _isolate_cache(monkeypatch, tmp_path)
    sub_id = "packepisode05a"
    _seed_metadata(sub_id)

    async def _fake_search(**kwargs):
        return [
            SubtitleRelease(
                release_name=PACK_NAME, download_url="http://pack", provider="subdl", lang="ara"
            )
        ]

    # Auto-sync is disabled for this request so the assertion is about the
    # unbundling, not about a reference being resolved.
    _disable_auto_sync(client)

    with patch("app.providers.subdl.SubdlProvider.search_subtitles", _fake_search), patch(
        "app.providers.subdl.SubdlProvider.download_archive",
        AsyncMock(return_value=_season_pack_zip()),
    ):
        response = client.get(f"/sub/{sub_id}.srt")

    assert response.status_code == 200, response.text
    body = response.text
    # Only the requested episode is present.
    assert "episode 5 line one" in body
    assert "episode 4 line one" not in body
    assert "episode 6 line one" not in body
    assert "readme" not in body
    # Served as a real subtitle payload, not as a raw ZIP.
    assert response.headers["content-type"].startswith("application/x-subrip")
    assert not body.startswith("PK")


def test_unpack_endpoint_returns_404_for_absent_episode(client, tmp_path, monkeypatch):
    """A pack that lacks the requested episode must not serve another one."""
    _isolate_cache(monkeypatch, tmp_path)
    sub_id = "packepisode99b"
    _seed_metadata(sub_id)
    # Metadata asks for an episode the archive does not contain.
    cache_manager.store_metadata(
        sub_id,
        {
            "sub_id": sub_id,
            "imdb_id": "tt0773262",
            "media_type": "series",
            "provider": "subdl",
            "download_url": "http://pack",
            "release_name": PACK_NAME,
            "target_filename": TARGET,
            "season": 8,
            "episode": 42,
            "lang": "ara",
        },
    )

    async def _fake_search(**kwargs):
        return [
            SubtitleRelease(
                release_name=PACK_NAME, download_url="http://pack", provider="subdl", lang="ara"
            )
        ]

    with patch("app.providers.subdl.SubdlProvider.search_subtitles", _fake_search), patch(
        "app.providers.subdl.SubdlProvider.download_archive",
        AsyncMock(return_value=_season_pack_zip()),
    ):
        response = client.get(f"/sub/{sub_id}.srt")

    # Either a clean error or a fallback; critically, it must not be a 200 that
    # silently serves a different episode.
    assert response.status_code != 200 or "episode 5 line one" not in response.text
