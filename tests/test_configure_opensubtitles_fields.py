"""The OpenSubtitles account fields must be *visible*, not merely present.

This exists because the fields shipped broken in a way that every
string-presence check passed. The markup was in the page and greppable, but the
inputs were wrapped in ``.provider-key`` -- and that class *alone* is the
collapsED style (``max-height: 0; opacity: 0``). Only ``.provider-key.expanded``
is visible, and the toggle script applies ``expanded`` to the card wrapper, not to
anything nested inside it. So the fields collapsed to zero height inside a card
that was visibly open.

The lesson these tests encode: assert the rendered geometry, not the text.
"""

from __future__ import annotations

import re

import pytest

#: Class that is invisible unless ``expanded`` is also present.
_COLLAPSIBLE = "provider-key"


def _card(html: str) -> str:
    """The OpenSubtitles key card, from its opening tag to its close."""
    start = html.index('id="opensubtitlesKey"')
    return html[start : html.index("</div>", html.index("opensubtitles-password", start)) + 6]


def test_the_two_inputs_exist(client):
    html = client.get("/configure").text

    assert 'id="opensubtitles-username"' in html
    assert 'id="opensubtitles-password"' in html


def test_the_inputs_are_inside_the_opensubtitles_card(client):
    """Directly inside the OpenSubtitles container, so they sit below the key."""
    card = _card(client.get("/configure").text)

    assert 'id="opensubtitles-username"' in card
    assert 'id="opensubtitles-password"' in card


def test_the_password_field_is_a_password_input(client):
    html = client.get("/configure").text

    assert re.search(r'type="password"[^>]*id="opensubtitles-password"', html)
    # Never pre-filled, and never echoed back into the page.
    assert re.search(r'id="opensubtitles-password"\s+value=""', html)


def test_neither_input_sits_inside_a_collapsed_container(client):
    """The regression.

    ``.provider-key`` on its own is invisible. Any wrapper between the card and
    the input that carries it without ``expanded`` hides the input, which is
    exactly what shipped.
    """
    html = client.get("/configure").text
    card = _card(html)

    # Every class attribute between the card root and the two inputs.
    classes = re.findall(r'class="([^"]*)"', card)
    for raw in classes:
        tokens = raw.split()
        if _COLLAPSIBLE in tokens:
            assert "expanded" in tokens or "expanded-tall" in tokens, (
                f"element inside the OpenSubtitles card carries {_COLLAPSIBLE!r} "
                f"without an expanded class, so it renders at zero height: {raw!r}"
            )


def test_the_inputs_do_not_use_the_action_wrap_class(client):
    """`.provider-key-wrap` reserves 118px of input padding for a link that
    these fields do not have, which would push their placeholder text off."""
    card = _card(client.get("/configure").text)

    username_block = card[card.index("opensubtitles-username") : card.index("opensubtitles-password")]
    assert "provider-key-wrap" not in username_block


def test_the_card_is_tall_enough_to_show_all_three_fields(client):
    """`.provider-key.expanded` is capped at 170px -- one field. Three inputs
    plus two note paragraphs would be clipped even when not collapsed."""
    html = client.get("/configure").text

    assert re.search(r'id="opensubtitlesKey"\s+class="[^"]*expanded-tall', html)
    assert ".provider-key.expanded.expanded-tall" in html


def test_the_tall_class_is_not_what_the_toggle_script_toggles(client):
    """The script must keep toggling only ``expanded``.

    If it ever toggled the whole class list, ``expanded-tall`` would be removed
    on collapse and never restored, silently re-clipping the card.
    """
    html = client.get("/configure").text

    assert re.search(r'classList\.toggle\("expanded",\s*toggle\.checked\)', html)


def test_the_single_field_cards_keep_the_original_height(client):
    """SubDL and SubSource have one field; they must not inherit the tall rule."""
    html = client.get("/configure").text

    for provider in ("subdlKey", "subsourceKey"):
        match = re.search(rf'id="{provider}"\s+class="([^"]*)"', html)
        assert match is not None, provider
        assert "expanded-tall" not in match.group(1)


@pytest.mark.parametrize("field", ["opensubtitles-username", "opensubtitles-password"])
def test_each_field_has_a_visible_label(client, field):
    """A placeholder is not a label: it vanishes as soon as the user types."""
    html = client.get("/configure").text

    assert f'for="{field}"' in html


def test_the_card_still_collapses_when_the_provider_is_disabled(client):
    """The fix must not make the fields permanently visible. OpenSubtitles is
    off by default, and the card is meant to stay collapsed until enabled."""
    html = client.get("/configure").text

    match = re.search(r'id="opensubtitlesKey"\s+class="([^"]*)"', html)
    assert match is not None
    # No bare `expanded` in the server-rendered markup; the script adds it when
    # the toggle is switched on.
    assert "expanded" not in match.group(1).split()
    assert "expanded-tall" in match.group(1).split()
