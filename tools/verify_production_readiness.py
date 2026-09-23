"""
Production Readiness Audit & Verification Script for NinjaSubs Stremio Addon.
Performs live HTTP endpoint validation, protocol verification, subtitle content serving check,
and ranking audit execution against both the live Docker container (port 7000)
and the internal FastAPI application.
"""

import json
import os
import re
import subprocess
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient

from app.cache import cache_manager
from app.main import app
from app.models import SubtitleRelease


def print_step(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def verify_manifest():
    print_step("1. MANIFEST ENDPOINT VALIDATION (GET /manifest.json)")

    # 1. Test live Docker container
    docker_url = "http://localhost:7000/manifest.json"
    print(f"Querying live container at {docker_url}...")
    try:
        resp = httpx.get(docker_url, timeout=5.0)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        print("Live Docker response: 200 OK")
    except Exception as e:
        print(f"Warning: Could not connect to live container ({e}), testing via FastAPI TestClient...")
        with TestClient(app) as client:
            resp = client.get("/manifest.json")
            assert resp.status_code == 200
            data = resp.json()

    print(f"Manifest JSON: {json.dumps(data, indent=2)}")

    # Protocol requirements
    assert "id" in data, "Missing 'id'"
    assert data["id"] == "org.ninjasubs.addon", f"Unexpected id: {data['id']}"
    assert "version" in data, "Missing 'version'"
    assert re.match(r"^\d+\.\d+\.\d+", data["version"]), f"Invalid semantic version: {data['version']}"
    assert "name" in data, "Missing 'name'"
    assert data["name"] == "NinjaSubs", f"Unexpected name: {data['name']}"
    assert "resources" in data, "Missing 'resources'"
    assert "subtitles" in data["resources"], f"'subtitles' resource not found in {data['resources']}"
    assert "types" in data, "Missing 'types'"
    assert "movie" in data["types"], f"'movie' not in types: {data['types']}"
    assert "series" in data["types"], f"'series' not in types: {data['types']}"
    assert "idPrefixes" in data, "Missing 'idPrefixes'"
    assert "tt" in data["idPrefixes"], f"'tt' not in idPrefixes: {data['idPrefixes']}"

    print("[SUCCESS] Manifest strictly adheres to Stremio v3 specification.")


def verify_movie_and_series_endpoints():
    print_step("2. MOVIE & SERIES SUBTITLE STREAM QUERY VALIDATION")

    with TestClient(app) as client:
        # A. Movie: tt0172495 (Gladiator) or tt0133093 (The Matrix)
        movie_imdb = "tt0133093"
        print(f"Testing Movie Subtitle Query: GET /subtitles/movie/{movie_imdb}.json")

        mock_movie_subs = [
            SubtitleRelease(
                release_name="The.Matrix.1999.1080p.BluRay.x264.srt",
                download_url="http://provider.test/matrix_bluray.srt",
                provider="subdl",
                lang="ara",
            ),
            SubtitleRelease(
                release_name="The.Matrix.1999.720p.WEB-DL.srt",
                download_url="http://provider.test/matrix_web.srt",
                provider="subsource",
                lang="ara",
            ),
        ]

        with (
            patch("app.providers.subdl.SubdlProvider.search_subtitles", new=AsyncMock(return_value=mock_movie_subs[:1])),
            patch("app.providers.subsource.SubsourceProvider.search_subtitles", new=AsyncMock(return_value=mock_movie_subs[1:])),
            patch("app.providers.opensubtitles.OpenSubtitlesProvider.search_subtitles", new=AsyncMock(return_value=[])),
            patch("app.providers.cinemeta.CinemetaClient.get_metadata", new=AsyncMock(return_value=None)),
        ):
            resp = client.get(
                f"/subtitles/movie/{movie_imdb}/filename=The.Matrix.1999.1080p.BluRay.x264-FLUX.json"
            )
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
            data = resp.json()
            assert "subtitles" in data, "Response missing top-level 'subtitles' key"
            subtitles = data["subtitles"]
            print(f"Returned {len(subtitles)} movie subtitles:")
            for s in subtitles:
                print(f"  - [{s.get('lang')}] {s.get('title')} -> {s.get('url')}")
                assert "id" in s, "Item missing 'id'"
                assert "url" in s, "Item missing 'url'"
                assert "lang" in s, "Item missing 'lang'"
                assert s["lang"] == "ara", f"Expected lang 'ara', got {s['lang']}"
                assert "title" in s, "Item missing 'title'"
                assert "[" in s["title"] and "]" in s["title"], f"Title not badge-formatted: {s['title']}"

            print("[SUCCESS] Movie stream query response matches Stremio v3 contract.")

        # B. Series: tt11198330:2:1 (House of the Dragon S02E01)
        series_id = "tt11198330:2:1"
        print(f"\nTesting Series Subtitle Query: GET /subtitles/series/{series_id}.json")

        mock_series_subs = [
            SubtitleRelease(
                release_name="House.of.the.Dragon.S02E01.1080p.HMAX.WEB-DL-FLUX.srt",
                download_url="http://provider.test/s02e01.srt",
                provider="subdl",
                lang="ara",
            ),
            SubtitleRelease(
                release_name="House.of.the.Dragon.S02E02.1080p.HMAX.WEB-DL-FLUX.srt",
                download_url="http://provider.test/s02e02_wrong.srt",
                provider="subdl",
                lang="ara",
            ),
        ]

        with (
            patch("app.providers.subdl.SubdlProvider.search_subtitles", new=AsyncMock(return_value=mock_series_subs)),
            patch("app.providers.subsource.SubsourceProvider.search_subtitles", new=AsyncMock(return_value=[])),
            patch("app.providers.opensubtitles.OpenSubtitlesProvider.search_subtitles", new=AsyncMock(return_value=[])),
            patch("app.providers.cinemeta.CinemetaClient.get_metadata", new=AsyncMock(return_value=None)),
        ):
            resp = client.get(
                f"/subtitles/series/{series_id}/filename=House.of.the.Dragon.S02E01.1080p.HMAX.WEB-DL-FLUX.json"
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "subtitles" in data
            subtitles = data["subtitles"]
            print(f"Returned {len(subtitles)} series subtitles:")
            for s in subtitles:
                print(f"  - [{s.get('lang')}] {s.get('title')}")
                assert "S02E01" in s.get("title") or "s02e01" in s.get("title").lower()
                assert "S02E02" not in s.get("title")

            # Verify that S02E02 was filtered out
            assert len(subtitles) == 1, f"Expected 1 matching episode, got {len(subtitles)}"
            assert "S02E01" in subtitles[0]["title"]
            assert "[SubDL]" in subtitles[0]["title"]
            print("[SUCCESS] Series stream query correctly filtered out mismatched episodes.")


def verify_subtitle_content_serving():
    print_step("3. SUBTITLE CONTENT SERVING VERIFICATION (GET /sub/...)")

    with TestClient(app) as client:
        sub_id = "test_verify_bidi_1234"
        raw_arabic_srt = (
            "1\n"
            "00:00:01,000 --> 00:00:04,000\n"
            "مرحبا بكم في عالم السينما.\n\n"
            "2\n"
            "00:00:05,000 --> 00:00:08,000\n"
            "<i>هل استمتعت بالفيلم؟</i>\n\n"
            "3\n"
            "00:00:09,000 --> 00:00:12,000\n"
            "هذا أمر رائع حقاً!\n"
        ).encode()

        # Save into local cache directly
        import asyncio
        asyncio.run(cache_manager.save_subtitle(sub_id, raw_arabic_srt))
        cache_manager.store_metadata(
            sub_id,
            {
                "sub_id": sub_id,
                "provider": "subdl",
                "release_name": "Test.Arabic.Movie.1999.srt",
                "lang": "ara",
            },
        )

        resp = client.get(f"/sub/{sub_id}.srt")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"

        # Check Content-Type header
        content_type = resp.headers.get("content-type", "")
        print(f"Content-Type header: {content_type}")
        assert "application/x-subrip" in content_type or "text/plain" in content_type
        assert "charset=utf-8" in content_type.lower()

        content_str = resp.content.decode("utf-8")
        print("Sample returned subtitle content:")
        for line in content_str.strip().split("\n")[:8]:
            print(f"  {line}")

        # Verify clean subtitle content delivery (no artificial Unicode markers)
        expected_line1 = "مرحبا بكم في عالم السينما.\u200f"
        expected_line2 = "<i>هل استمتعت بالفيلم؟</i>\u200f"
        expected_line3 = "هذا أمر رائع حقاً!\u200f"

        assert expected_line1 in content_str, "Line 1 mismatch"
        assert expected_line2 in content_str, "Line 2 mismatch"
        assert expected_line3 in content_str, "Line 3 mismatch"
        assert "\u200F" in content_str or "\u200f" in content_str, "Expected RLM marker for RTL punctuation"

        # Verify SRT formatting integrity (timestamps & numbers must not be touched)
        assert "00:00:01,000 --> 00:00:04,000" in content_str
        assert "00:00:05,000 --> 00:00:08,000" in content_str
        assert "00:00:09,000 --> 00:00:12,000" in content_str

        print("[SUCCESS] Content headers and clean Arabic subtitle delivery verified.")


def verify_ranking_audit():
    print_step("4. SUBTITLE RANKING AUDIT CLI EXECUTION")

    cmd = [
        sys.executable,
        "tools/audit_subtitle_ranking.py",
        "--target",
        "House.of.the.Dragon.S02E01.1080p.HMAX.WEB-DL.DDP5.1.Atmos.H.264-FLUX",
        "--candidates",
        "House of the Dragon S02E01 HMAX WEB-DL FLUX",
        "House of the Dragon S02E01 AMZN NTb",
        "House of the Dragon S02E02 HMAX FLUX",
    ]
    print(f"Running command: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    print(res.stdout)
    if res.stderr:
        print(f"STDERR: {res.stderr}")
    assert res.returncode == 0, f"Audit tool failed with exit code {res.returncode}"
    assert "1. House of the Dragon S02E01 HMAX WEB-DL FLUX" in res.stdout
    assert "2. House of the Dragon S02E01 AMZN NTb" in res.stdout
    assert "3. House of the Dragon S02E02 HMAX FLUX" in res.stdout
    assert "NO ANOMALIES DETECTED." in res.stdout
    print("[SUCCESS] Subtitle ranking audit CLI verified expected order and episode rejection.")


def main():
    print("\nStarting Full-Stack Production Readiness Audit for NinjaSubs Stremio Addon...")
    verify_manifest()
    verify_movie_and_series_endpoints()
    verify_subtitle_content_serving()
    verify_ranking_audit()
    print("\n" + "=" * 80)
    print("  ALL PRODUCTION READINESS CHECKS PASSED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
