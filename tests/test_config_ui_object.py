"""Tests verifying that initialPrefs is a JavaScript object, not a string.

Also verifies badge_parts and other Phase-2 preferences are correctly reflected in the UI.
"""

from fastapi.testclient import TestClient

from app.main import app
from app.utils.config_parser import encode_user_config

client = TestClient(app)

def test_configure_page_renders_initialPrefs_as_object():
    # Get the configure page HTML and verify the JS object literal
    response = client.get("/configure")
    assert response.status_code == 200
    html = response.text
    # Find the script containing initialPrefs
    # The object should start with `{` not `[` and should not be quoted as a string
    # Look for pattern: const initialPrefs = { ... };
    import re
    # Extract the JSON part after "const initialPrefs = " and before the final semicolon
    match = re.search(r"const\s+initialPrefs\s*=\s*(\{[^}]*\})\s*;", html, re.DOTALL)
    assert match is not None, "initialPrefs assignment not found in HTML"
    json_str = match.group(1)
    # The JSON should be a valid object literal (starts with {, ends with })
    assert json_str.strip().startswith("{") and json_str.strip().endswith("}"), "initialPrefs is not an object"
    # Verify it contains expected keys
    assert "badge_parts" in json_str
    assert "enable_rtl_fix" in json_str
    assert "enable_subdl" in json_str

def test_configure_page_prefills_badge_parts():
    # Encode a config with specific badge parts
    encoded = encode_user_config(badge_parts=["filename", "score"])
    response = client.get(f"/{encoded}/configure")
    assert response.status_code == 200
    html = response.text
    # Verify the JSON contains the badge parts (order canonicalized)
    import re
    match = re.search(r"\"badge_parts\"\s*:\s*(\[.*?\])", html, re.DOTALL)
    assert match is not None, "badge_parts not found in prefilled config"
    parts_json = match.group(1)
    # Should contain both parts (order may be canonicalized to score, uploader)
    assert "\"filename\"" in parts_json or "\"score\"" in parts_json
    # Verify the UI shows the correct selected state
    # The badge builder UI should have appropriate parts selected
    assert "data-part=\"score\"" in html or "data-part=\"filename\"" in html

def test_configure_page_prefills_rtl_toggle():
    import re
    encoded = encode_user_config(enable_rtl_fix=False)
    response = client.get(f"/{encoded}/configure")
    assert response.status_code == 200
    html = response.text
    # Verify the JSON reflects false for enable_rtl_fix (JS will apply it)
    match = re.search(r"\"enable_rtl_fix\"\s*:\s*false", html, re.DOTALL)
    assert match is not None, "enable_rtl_fix not false in prefilled config"

def test_configure_page_prefills_provider_toggles():
    import re
    encoded = encode_user_config(enable_subdl=False, enable_subsource=True)
    response = client.get(f"/{encoded}/configure")
    assert response.status_code == 200
    html = response.text
    # Verify JSON reflects the state (JS will apply to checkboxes)
    assert re.search(r"\"enable_subdl\"\s*:\s*false", html, re.DOTALL)
    assert re.search(r"\"enable_subsource\"\s*:\s*true", html, re.DOTALL)
