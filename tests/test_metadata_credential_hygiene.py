"""Provider credentials must never be persisted in the subtitle cache.

Confirmed security finding: ``_meta/*.json`` was written with a raw
``json.dumps`` of the download metadata, which carries ``subdl_key``,
``subsource_key`` and ``opensubtitles_key`` verbatim, plus a ``download_url``
that embeds ``?api_key=...``. A real file on disk was inspected and contained
all three keys in plaintext.

The credentials were not merely stored, they were *reused*: the download path
read them back from cache and let them take precedence over the current
request's own config.

These tests use deterministic fake secrets only. No real credential appears
here or in any assertion.
"""

import json
import logging

import pytest

from app.cache import LRUCacheManager, sanitize_metadata

FAKE_SUBDL = "FAKE-SUBDL-SECRET-0001"
FAKE_SUBSOURCE = "sk_FAKE0000SECRET0002"
FAKE_OPENSUBTITLES = "FAKE0000OS0000000F"


@pytest.fixture
def cache(tmp_path):
    root = tmp_path / "cache"
    (root / "_meta").mkdir(parents=True)
    return LRUCacheManager(cache_dir=str(root))


def _meta_with_secrets(sub_id: str = "abc123") -> dict:
    return {
        "sub_id": sub_id,
        "provider": "subdl",
        "imdb_id": "tt1",
        "media_type": "movie",
        "lang": "ara",
        "release_name": "Movie.2020.1080p.WEB-DL-GRP.srt",
        "target_filename": "Movie.2020.1080p.BluRay-GRP.mkv",
        "video_size": "1234",
        "credential_source": "manifest",
        "subdl_key": FAKE_SUBDL,
        "subsource_key": FAKE_SUBSOURCE,
        "opensubtitles_key": FAKE_OPENSUBTITLES,
        "download_url": f"https://dl.example.invalid/sub/1.zip?api_key={FAKE_SUBDL}&id=7",
        "stream_url": f"https://stream.example.invalid/movie.mkv?token={FAKE_SUBSOURCE}",
    }


class TestNoCredentialsPersisted:
    def test_written_metadata_contains_no_raw_secrets(self, cache):
        cache.store_metadata("abc123", _meta_with_secrets())
        raw = cache.get_meta_path("abc123").read_text(encoding="utf-8")

        assert FAKE_SUBDL not in raw
        assert FAKE_SUBSOURCE not in raw
        assert FAKE_OPENSUBTITLES not in raw

    def test_secret_fields_are_absent_entirely(self, cache):
        cache.store_metadata("abc123", _meta_with_secrets())
        stored = json.loads(cache.get_meta_path("abc123").read_text(encoding="utf-8"))

        for field in ("subdl_key", "subsource_key", "opensubtitles_key"):
            assert field not in stored

    def test_bearer_token_is_not_persisted(self, cache):
        """A Subsource key is a bearer token (``sk_``) and must not survive."""
        cache.store_metadata("abc123", _meta_with_secrets())
        raw = cache.get_meta_path("abc123").read_text(encoding="utf-8")
        assert "sk_" not in raw

    def test_credential_bearing_url_params_are_stripped_but_url_survives(self, cache):
        cache.store_metadata("abc123", _meta_with_secrets())
        stored = json.loads(cache.get_meta_path("abc123").read_text(encoding="utf-8"))

        # The download still needs a usable URL, so only the secret param goes.
        assert stored["download_url"] == "https://dl.example.invalid/sub/1.zip?id=7"
        assert "api_key" not in stored["download_url"]
        assert stored["stream_url"] == "https://stream.example.invalid/movie.mkv"

    def test_non_sensitive_provenance_is_preserved(self, cache):
        cache.store_metadata("abc123", _meta_with_secrets())
        stored = json.loads(cache.get_meta_path("abc123").read_text(encoding="utf-8"))

        assert stored["provider"] == "subdl"
        assert stored["credential_source"] == "manifest"
        assert stored["release_name"] == "Movie.2020.1080p.WEB-DL-GRP.srt"
        assert stored["target_filename"] == "Movie.2020.1080p.BluRay-GRP.mkv"
        assert stored["imdb_id"] == "tt1"
        assert stored["lang"] == "ara"
        assert stored["video_size"] == "1234"

    def test_credential_presence_is_still_recorded(self, cache):
        """Diagnostics keep knowing a credential existed, without its value."""
        cache.store_metadata("abc123", _meta_with_secrets())
        stored = json.loads(cache.get_meta_path("abc123").read_text(encoding="utf-8"))

        assert stored["has_subdl_key"] is True
        assert stored["has_subsource_key"] is True
        assert stored["has_opensubtitles_key"] is True

    def test_metadata_without_credentials_is_left_untouched(self, cache):
        """No credential means nothing to record and nothing to rewrite."""
        meta = {
            "sub_id": "plain1",
            "imdb_id": "tt1",
            "provider": "subdl",
            "release_name": "Movie.srt",
        }
        cache.store_metadata("plain1", meta)
        stored = json.loads(cache.get_meta_path("plain1").read_text(encoding="utf-8"))
        assert stored == meta
        assert "has_subdl_key" not in stored

    def test_round_trip_preserves_cache_correctness(self, cache):
        cache.store_metadata("abc123", _meta_with_secrets())
        loaded = cache.get_metadata("abc123")

        assert loaded is not None
        assert loaded["sub_id"] == "abc123"
        assert loaded["imdb_id"] == "tt1"
        assert loaded["target_filename"] == "Movie.2020.1080p.BluRay-GRP.mkv"
        assert loaded["download_url"].startswith("https://dl.example.invalid/")


class TestLegacyMetadataIsNotReExposed:
    def test_legacy_file_with_secrets_is_sanitized_on_read(self, cache):
        """A pre-existing plaintext file must not hand the secret back."""
        path = cache.get_meta_path("legacy1")
        path.write_text(json.dumps(_meta_with_secrets("legacy1")), encoding="utf-8")

        loaded = cache.get_metadata("legacy1")

        assert loaded is not None
        assert "subdl_key" not in loaded
        assert loaded.get("subdl_key") is None
        assert FAKE_SUBDL not in json.dumps(loaded)
        assert FAKE_SUBSOURCE not in json.dumps(loaded)

    def test_legacy_file_is_rewritten_in_safe_form(self, cache):
        path = cache.get_meta_path("legacy1")
        path.write_text(json.dumps(_meta_with_secrets("legacy1")), encoding="utf-8")

        cache.get_metadata("legacy1")
        rewritten = path.read_text(encoding="utf-8")

        assert FAKE_SUBDL not in rewritten
        assert FAKE_SUBSOURCE not in rewritten
        assert "imdb_id" in rewritten, "useful metadata must survive the repair"

    def test_repair_never_touches_subtitle_payload(self, cache):
        payload = cache.get_subtitle_path("legacy1")
        payload.write_bytes(b"1\n00:00:01,000 --> 00:00:02,000\nKEEP ME\n")
        before = payload.read_bytes()

        cache.get_meta_path("legacy1").write_text(
            json.dumps(_meta_with_secrets("legacy1")), encoding="utf-8"
        )
        cache.get_metadata("legacy1")

        assert payload.read_bytes() == before

    def test_sanitized_read_does_not_log_the_secret(self, cache, caplog):
        cache.get_meta_path("legacy1").write_text(
            json.dumps(_meta_with_secrets("legacy1")), encoding="utf-8"
        )
        with caplog.at_level(logging.DEBUG):
            cache.get_metadata("legacy1")
        assert FAKE_SUBDL not in caplog.text
        assert FAKE_SUBSOURCE not in caplog.text


class TestSanitizerUnit:
    def test_reports_whether_it_changed_anything(self):
        _, changed = sanitize_metadata(_meta_with_secrets())
        assert changed is True
        _, changed = sanitize_metadata({"imdb_id": "tt1"})
        assert changed is False

    def test_does_not_mutate_the_caller_dict(self):
        original = _meta_with_secrets()
        sanitize_metadata(original)
        assert original["subdl_key"] == FAKE_SUBDL

    @pytest.mark.parametrize(
        "url",
        [
            "https://x.invalid/a?api_key=abc",
            "https://x.invalid/a?token=abc",
            "https://x.invalid/a?access_token=abc",
            "https://x.invalid/a?sig=abc",
            "https://x.invalid/a?id=1&key=abc&z=2",
        ],
    )
    def test_credential_params_are_removed(self, url):
        safe, _ = sanitize_metadata({"download_url": url})
        assert "abc" not in safe["download_url"]

    def test_non_credential_params_are_preserved(self):
        safe, _ = sanitize_metadata({"download_url": "https://x.invalid/a?id=1&name=b"})
        assert safe["download_url"] == "https://x.invalid/a?id=1&name=b"

    def test_url_without_query_is_untouched(self):
        safe, _ = sanitize_metadata({"download_url": "https://x.invalid/a.zip"})
        assert safe["download_url"] == "https://x.invalid/a.zip"

    def test_non_dict_input_is_passed_through(self):
        assert sanitize_metadata(None) == (None, False)
