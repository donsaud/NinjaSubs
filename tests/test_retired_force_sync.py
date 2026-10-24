"""Retired forced-investigation preference cannot survive legacy settings."""

import base64
import inspect
import json
from urllib.parse import urlencode

import pytest

from app.main import _sync_meta_for_user
from app.models import UserPreferences
from app.services.sync.metadata_gate import MetadataGateAction, decide_metadata_gate
from app.utils.config_parser import encode_user_config, parse_user_config


@pytest.mark.parametrize("key", ["force_sync", "forceSync"])
@pytest.mark.parametrize("value", [True, "true", "1", "yes"])
@pytest.mark.parametrize("format_", ["json", "query"])
@pytest.mark.parametrize("auto_sync", [True, False])
def test_legacy_forced_investigation_is_ignored(key, value, format_, auto_sync):
    data = {"subdl_key": "test-key", key: value, "auto_sync": auto_sync}
    if format_ == "json":
        encoded = base64.urlsafe_b64encode(json.dumps(data).encode()).decode().rstrip("=")
    else:
        encoded = urlencode(data)
    prefs = parse_user_config(encoded)
    assert prefs.auto_sync is auto_sync
    assert prefs.subdl_key == "test-key"
    assert not getattr(prefs, "force_sync", False)
    assert "force_sync" not in prefs.model_dump()
    assert "forceSync" not in prefs.model_dump()
    # Re-saving the remaining settings cannot perpetuate the retired field.
    saved = encode_user_config(subdl_key=prefs.subdl_key, auto_sync=prefs.auto_sync)
    decoded = json.loads(base64.urlsafe_b64decode(saved + "=" * (-len(saved) % 4)))
    assert "force_sync" not in decoded and "forceSync" not in decoded
    assert "force_sync" not in inspect.signature(encode_user_config).parameters


def test_defaults_do_not_expose_a_forced_investigation_setting():
    assert UserPreferences().auto_sync is True
    assert not getattr(parse_user_config(None), "force_sync", False)
    assert "force_sync" not in UserPreferences.model_fields


@pytest.mark.parametrize("auto_sync", [True, False])
def test_stale_metadata_cannot_restore_force_sync_or_change_automatic_gate(auto_sync):
    prefs = parse_user_config(f"subdl=test-key&force_sync=true&auto_sync={auto_sync}")
    meta = _sync_meta_for_user(
        {"force_sync": True, "forceSync": True},
        {"force_sync": True, "forceSync": True}, prefs)
    assert "force_sync" not in meta and "forceSync" not in meta
    assert prefs.auto_sync is auto_sync
    for group, expected in [("d3g", MetadataGateAction.FAST_ORIGINAL),
                            ("CHD", MetadataGateAction.INVESTIGATE)]:
        decision = decide_metadata_gate(
            "Film.2020.1080p.BluRay-d3g.mkv", f"Film.2020.1080p.BluRay-{group}.srt",
            explicit_sync_request=bool(meta.get("force_sync")))
        assert decision.action is expected


def test_configure_has_auto_sync_without_retired_toggle_or_prefill(client):
    payload = {"subdl_key": "test-key", "auto_sync": False,
               "force_sync": True, "forceSync": True}
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    for url in ("/configure", f"/{encoded}/configure"):
        response = client.get(url)
        assert response.status_code == 200
        assert 'id="autoSync"' in response.text
        assert "Always Investigate Timing" not in response.text
        assert "force_sync" not in response.text
        assert "forceSync" not in response.text
    assert '"auto_sync": false' in response.text
