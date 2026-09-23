"""Translator-handle protection for subtitle ad stripping.

Handles directly following translator-attribution triggers (EN/AR) are
credits and must survive ``strip_advertisements`` with
``keep_translator_credits=True``; standalone promo handles are still removed.
"""

from app.utils.cleaners import strip_advertisements


def _srt_cue_body(*lines: str) -> str:
    return (
        "1\n"
        "00:00:01,000 --> 00:00:03,000\n" + "\n".join(lines) + "\n"
        "\n"
        "2\n"
        "00:05:00,000 --> 00:05:03,000\n"
        "Actual dialogue line\n"
    )


def test_arabic_tarjama_handle_preserved():
    out = strip_advertisements(
        _srt_cue_body("ترجمة: @D700mka"), keep_translator_credits=True
    )
    assert "@D700mka" in out
    assert "ترجمة" in out


def test_english_translated_by_handle_preserved():
    out = strip_advertisements(
        _srt_cue_body("Translated by: @D700mka"), keep_translator_credits=True
    )
    assert "@D700mka" in out
    assert "Translated by" in out


def test_arabic_timing_handle_preserved():
    out = strip_advertisements(
        _srt_cue_body("تعديل التوقيت: @D700mka"), keep_translator_credits=True
    )
    assert "@D700mka" in out
    assert "تعديل التوقيت" in out


def test_promo_handle_without_credit_context_stripped():
    out = strip_advertisements(
        _srt_cue_body("Follow @random_channel"), keep_translator_credits=True
    )
    assert "@random_channel" not in out


def test_mixed_credit_cue_keeps_handle_strips_link():
    out = strip_advertisements(
        _srt_cue_body("ترجمة: @D700mka www.adsite.com"),
        keep_translator_credits=True,
    )
    assert "@D700mka" in out
    assert "ترجمة" in out
    assert "www.adsite.com" not in out
    assert "Actual dialogue line" in out


def test_arabic_chained_handles_with_waw_preserved():
    out = strip_advertisements(
        _srt_cue_body("ترجمة: @D700mka و @SaudSub"),
        keep_translator_credits=True,
    )
    assert "@D700mka" in out
    assert "@SaudSub" in out


def test_english_chained_handles_with_and_preserved():
    out = strip_advertisements(
        _srt_cue_body("Translated by: @user1 and @user2"),
        keep_translator_credits=True,
    )
    assert "@user1" in out
    assert "@user2" in out


def test_chained_handles_with_ampersand_and_comma_preserved():
    out = strip_advertisements(
        _srt_cue_body("ترجمة: @user1, @user2 & @user3"),
        keep_translator_credits=True,
    )
    assert "@user1" in out
    assert "@user2" in out
    assert "@user3" in out


def test_promo_handle_after_credit_chain_still_stripped():
    out = strip_advertisements(
        _srt_cue_body("ترجمة: @D700mka و @SaudSub Follow @spammer"),
        keep_translator_credits=True,
    )
    assert "@D700mka" in out
    assert "@SaudSub" in out
    assert "@spammer" not in out
