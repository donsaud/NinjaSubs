"""Unit tests for user configuration parsing and encoding."""

from app.models import UserPreferences
from app.utils.config_parser import encode_user_config, parse_user_config


def test_encode_and_parse_defaults():
    """Test encoding with default options and parsing back."""
    encoded = encode_user_config("subdl_123", "subsource_456")
    prefs = parse_user_config(encoded)

    assert prefs.subdl_key == "subdl_123"
    assert prefs.subsource_key == "subsource_456"
    assert prefs.languages == ["ara"]
    assert prefs.exclude_hi is False
    assert prefs.nuvio_mode is False


def test_encode_and_parse_custom_preferences():
    """Test custom languages, exclude_hi, and nuvio_mode setting."""
    encoded = encode_user_config(
        subdl_key="custom_subdl",
        subsource_key="custom_subsource",
        languages=["ara", "eng"],
        exclude_hi=True,
        nuvio_mode=True,
    )
    prefs = parse_user_config(encoded)

    assert prefs.subdl_key == "custom_subdl"
    assert prefs.subsource_key == "custom_subsource"
    assert prefs.languages == ["ara", "eng"]
    assert prefs.exclude_hi is True
    assert prefs.nuvio_mode is True


def test_tuple_unpacking_backward_compatibility():
    """Verify UserPreferences supports tuple unpacking for legacy code."""
    encoded = encode_user_config("key_a", "key_b")
    subdl_key, subsource_key = parse_user_config(encoded)

    assert subdl_key == "key_a"
    assert subsource_key == "key_b"


def test_parse_legacy_json_schema():
    """Verify legacy base64 format with 'subdl' and 'subsource' keys still parses cleanly."""
    import base64
    import json

    legacy_payload = json.dumps({"subdl": "leg_dl", "subsource": "leg_source"}).encode("utf-8")
    b64 = base64.urlsafe_b64encode(legacy_payload).decode("utf-8").rstrip("=")

    prefs = parse_user_config(b64)
    assert prefs.subdl_key == "leg_dl"
    assert prefs.subsource_key == "leg_source"
    assert prefs.languages == ["ara"]
    assert prefs.exclude_hi is False


def test_parse_none_or_empty_falls_back_to_env():
    """Verify None or empty string returns UserPreferences populated from environment."""
    from app.config import settings

    prefs_none = parse_user_config(None)
    assert prefs_none.subdl_key == settings.SUBDL_API_KEY
    assert prefs_none.subsource_key == settings.SUBSOURCE_API_KEY
    assert prefs_none.languages == ["ara"]
    assert prefs_none.exclude_hi is False

    prefs_empty = parse_user_config("")
    assert prefs_empty.subdl_key == settings.SUBDL_API_KEY


def test_parse_corrupt_base64_falls_back_gracefully():
    """Verify corrupt base64 string doesn't raise, returns fallback preferences."""
    corrupt_inputs = [
        "!!!not_valid_base64$$$",
        "invalid_base64_symbols===???",
        "e30",
    ]
    for inp in corrupt_inputs:
        prefs = parse_user_config(inp)
        assert isinstance(prefs, UserPreferences)


def test_parse_invalid_json_payload():
    """Verify valid base64 but invalid JSON returns fallback preferences."""
    import base64

    not_json = base64.urlsafe_b64encode(b"hello world raw text").decode("utf-8")
    prefs = parse_user_config(not_json)
    assert isinstance(prefs, UserPreferences)

    array_json = base64.urlsafe_b64encode(b"['item1', 'item2']").decode("utf-8")
    prefs2 = parse_user_config(array_json)
    assert isinstance(prefs2, UserPreferences)


def test_available_languages_and_normalization():
    """Verify comprehensive language dataset, aliases, and Subdl/Subsource mappings."""
    from app.utils.language import (
        AVAILABLE_LANGUAGES,
        get_display_prefix,
        get_subdl_lang_code,
        get_subsource_lang_name,
        normalize_to_iso639_2,
    )

    # Verify key languages are in dataset
    codes = [entry["code"] for entry in AVAILABLE_LANGUAGES]
    for required in ("ara", "eng", "fra", "spa", "deu", "ita", "por", "tur", "fas", "ind"):
        assert required in codes

    # Aliases to ISO-639-2
    assert normalize_to_iso639_2("arabic") == "ara"
    assert normalize_to_iso639_2("ar") == "ara"
    assert normalize_to_iso639_2("farsi") == "fas"
    assert normalize_to_iso639_2("persian") == "fas"
    assert normalize_to_iso639_2("fa") == "fas"
    assert normalize_to_iso639_2("indonesian") == "ind"
    assert normalize_to_iso639_2("id") == "ind"
    assert normalize_to_iso639_2("fre") == "fra"

    # Subdl mappings (2-letter uppercase)
    assert get_subdl_lang_code("ara") == "AR"
    assert get_subdl_lang_code("eng") == "EN"
    assert get_subdl_lang_code("fra") == "FR"
    assert get_subdl_lang_code("fas") == "FA"
    assert get_subdl_lang_code("ind") == "ID"

    # Subsource mappings (English names)
    assert get_subsource_lang_name("ara") == "Arabic"
    assert get_subsource_lang_name("eng") == "English"
    assert get_subsource_lang_name("fra") == "French"
    assert get_subsource_lang_name("fas") == "Farsi"
    assert get_subsource_lang_name("ind") == "Indonesian"

    # Display prefix mappings
    assert get_display_prefix("ara") == "AR"
    assert get_display_prefix("eng") == "EN"
    assert get_display_prefix("fas") == "FA"


def test_legacy_types_payload_ignored_gracefully():
    """Verify legacy payloads containing types parse gracefully without error."""
    import base64
    import json

    # 1. Base64 payload with legacy types key
    legacy_payload = base64.urlsafe_b64encode(
        json.dumps(
            {"subdl_key": "k1", "subsource_key": "k2", "types": ["retail", "forced"]}
        ).encode("utf-8")
    ).decode("utf-8")
    prefs = parse_user_config(legacy_payload)
    assert prefs.subdl_key == "k1"
    assert prefs.subsource_key == "k2"

    # 2. Query string format with legacy types
    qs_config = "subdl=k1&types=retail,forced"
    prefs_qs = parse_user_config(qs_config)
    assert prefs_qs.subdl_key == "k1"


def test_opensubtitles_config_parsing_and_property():
    """Verify opensubtitles_key is properly encoded, parsed, and exposed via property."""
    from app.utils.config_parser import encode_user_config, parse_user_config

    # Base64 payload encoding & parsing
    encoded = encode_user_config(
        subdl_key="subdl_1",
        subsource_key="subsource_2",
        opensubtitles_key="os_key_3",
        languages=["ara"],
    )
    prefs = parse_user_config(encoded)
    assert prefs.subdl_key == "subdl_1"
    assert prefs.subsource_key == "subsource_2"
    assert prefs.opensubtitles_key == "os_key_3"
    assert prefs.opensubtitles_api_key == "os_key_3"

    # Query string format
    qs = "subdl=k1&subsource=k2&opensubtitles=os3"
    prefs_qs = parse_user_config(qs)
    assert prefs_qs.opensubtitles_key == "os3"

    # Pipe-separated format
    pipe = "k1|k2|ara|0|0|os_pipe"
    prefs_pipe = parse_user_config(pipe)
    assert prefs_pipe.opensubtitles_key == "os_pipe"


def test_opensubtitles_language_mapping():
    """Verify get_opensubtitles_lang_code maps ISO-639-2 codes to ISO-639-1."""
    from app.utils.language import get_opensubtitles_lang_code

    assert get_opensubtitles_lang_code("ara") == "ar"
    assert get_opensubtitles_lang_code("eng") == "en"
    assert get_opensubtitles_lang_code("fra") == "fr"
    assert get_opensubtitles_lang_code("spa") == "es"
    assert get_opensubtitles_lang_code("deu") == "de"
    assert get_opensubtitles_lang_code("fas") == "fa"
    assert get_opensubtitles_lang_code("ar") == "ar"
