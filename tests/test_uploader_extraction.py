"""Unit tests for robust uploader/author username extraction across providers."""

from app.utils.uploader import extract_uploader


def test_subsource_contributors_displayname():
    """SubSource exposes the name via contributors[].displayname (uploaderId is numeric)."""
    item = {"uploaderId": 4242, "contributors": [{"id": 4242, "displayname": "subsmaster"}]}
    assert extract_uploader(item) == "subsmaster"


def test_subsource_uploader_id_only_returns_empty():
    """A numeric uploaderId with no contributor name must not be shown as a username."""
    assert extract_uploader({"uploaderId": 4242, "contributors": []}) == ""


def test_subdl_author_and_uploader_strings():
    assert extract_uploader({"author": "SubsPlease"}) == "SubsPlease"
    assert extract_uploader({"uploader": "FLUX"}) == "FLUX"
    assert extract_uploader({"author": "12345"}) == ""


def test_opensubtitles_nested_uploader_name():
    assert extract_uploader({"uploader": {"name": "subsmaster", "uploader_id": 1}}) == "subsmaster"
    assert extract_uploader({"uploader": {"username": "joe"}}) == "joe"


def test_nested_user_object_and_list():
    assert extract_uploader({"user": {"displayname": "nested_user"}}) == "nested_user"
    assert extract_uploader({"contributors": [{"name": "first"}, {"name": "second"}]}) == "first"


def test_no_uploader_fields_returns_empty():
    assert extract_uploader({"releaseName": "Movie.2024.1080p", "language": "Arabic"}) == ""
    assert extract_uploader(None) == ""
    assert extract_uploader("not-a-dict") == ""
