"""OpenSubtitles has one visible masked API-key field and no account UI."""

from tests.test_provider_credentials_ui import Elements


def test_opensubtitles_is_a_single_key_card(client):
    html = client.get("/configure").text
    fields = Elements(html).ids
    assert fields["opensubtitles-key"]["type"] == "password"
    assert "expanded-tall" not in fields["opensubtitlesKey"]["class"]
    assert "opensubtitles-username" not in fields
    assert "opensubtitles-password" not in fields
    assert "API Key + Account" not in html
    assert "API & account verified" not in html
    assert 'data-reveal="opensubtitles-key"' in html
