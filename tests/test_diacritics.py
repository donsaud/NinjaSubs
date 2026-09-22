"""Tests for Arabic diacritics stripping (Shadda-preserving)."""

from app.main import _build_subtitle_response
from app.services.cache import build_cache_key
from app.utils.cleaners import (
    strip_arabic_diacritics,
    strip_arabic_diacritics_bytes,
    tashkeel_remove,
)
from app.utils.config_parser import encode_user_config, parse_user_config

# ذَهَبَ -> ذهب
DHAHABA_IN = "\u0630\u064e\u0647\u064e\u0628\u064e"
DHAHABA_OUT = "\u0630\u0647\u0628"
# عَلَّمَ -> علّم (Shadda U+0651 preserved)
ALLAMA_IN = "\u0639\u064e\u0644\u0651\u064e\u0645\u064e"
ALLAMA_OUT = "\u0639\u0644\u0651\u0645"


def _srt(text: str) -> str:
    return f"1\n00:00:01,000 --> 00:00:03,000\n{text}\n"


def test_strips_diacritics_from_arabic_text():
    assert strip_arabic_diacritics(DHAHABA_IN) == DHAHABA_OUT


def test_smart_strip_dhahaba_alwalad():
    # ذَهَبَ الوَلَدُ -> ذهب الولد
    assert (
        strip_arabic_diacritics(
            "\u0630\u064e\u0647\u064e\u0628\u064e \u0627\u0644\u0648\u064e\u0644\u064e\u062f\u064f"
        )
        == "\u0630\u0647\u0628 \u0627\u0644\u0648\u0644\u062f"
    )


def test_smart_preserves_shadda():
    # عَلَّمَ الرَّجُلُ -> علّم الرّجل (both Shaddas kept)
    out = strip_arabic_diacritics(
        "\u0639\u064e\u0644\u0651\u064e\u0645\u064e \u0627\u0644\u0631\u0651\u064e\u062c\u064f\u0644\u064f"
    )
    assert out == "\u0639\u0644\u0651\u0645 \u0627\u0644\u0631\u0651\u062c\u0644"
    assert out.count("\u0651") == 2


def test_smart_preserves_tanween_fatha():
    # شكراً جَزِيلاً وجداً -> شكراً جزيلاً وجداً
    assert (
        strip_arabic_diacritics(
            "\u0634\u0643\u0631\u0627\u064b "
            "\u062c\u064e\u0632\u0650\u064a\u0644\u0627\u064b "
            "\u0648\u062c\u062f\u0627\u064b"
        )
        == "\u0634\u0643\u0631\u0627\u064b "
        "\u062c\u0632\u064a\u0644\u0627\u064b "
        "\u0648\u062c\u062f\u0627\u064b"
    )


def test_smart_preserves_feminine_kasra():
    # أنتِ لكِ الأولويةُ -> أنتِ لكِ الأولوية
    out = strip_arabic_diacritics(
        "\u0623\u0646\u062a\u0650 \u0644\u0643\u0650 "
        "\u0627\u0644\u0623\u0648\u0644\u0648\u064a\u0629\u064f"
    )
    assert out == (
        "\u0623\u0646\u062a\u0650 \u0644\u0643\u0650 "
        "\u0627\u0644\u0623\u0648\u0644\u0648\u064a\u0629"
    )
    assert out.count("\u0650") == 2


def test_smart_preserves_feminine_pronoun_with_internal_vowels():
    # أَنْتِ عَلَّمْتِ الطَّالِبَ شُكْرًا جَزِيلًا
    out = strip_arabic_diacritics(
        "\u0623\u064e\u0646\u0652\u062a\u0650 "
        "\u0639\u064e\u0644\u0651\u064e\u0645\u0652\u062a\u0650 "
        "\u0627\u0644\u0637\u0651\u064e\u0627\u0644\u0650\u0628\u064e "
        "\u0634\u064f\u0643\u0652\u0631\u064b\u0627 "
        "\u062c\u064e\u0632\u0650\u064a\u0644\u064b\u0627"
    )
    # Kasra survives on both "أنتِ" and the feminine suffix Ta' "علّمتِ".
    assert out == (
        "\u0623\u0646\u062a\u0650 "
        "\u0639\u0644\u0651\u0645\u062a\u0650 "
        "\u0627\u0644\u0637\u0651\u0627\u0644\u0628 "
        "\u0634\u0643\u0631\u064b\u0627 "
        "\u062c\u0632\u064a\u0644\u064b\u0627"
    )


def test_preserves_feminine_suffix_ta():
    # عَلَّمْتِ -> علّمتِ (Kasra on the feminine suffix Ta' kept)
    assert (
        strip_arabic_diacritics(
            "\u0639\u064e\u0644\u0651\u064e\u0645\u0652\u062a\u0650"
        )
        == "\u0639\u0644\u0651\u0645\u062a\u0650"
    )
    for verb in ("\u0641\u064e\u0639\u064e\u0644\u0652\u062a\u0650",
                 "\u0642\u064f\u0644\u0652\u062a\u0650"):
        assert strip_arabic_diacritics(verb).endswith("\u062a\u0650")


def test_feminine_ta_and_pronoun_combined_sentence():
    # أَنْتِ عَلَّمْتِ الطَّالِبَ -> أنتِ علّمتِ الطّالب
    assert (
        strip_arabic_diacritics(
            "\u0623\u064e\u0646\u0652\u062a\u0650 "
            "\u0639\u064e\u0644\u0651\u064e\u0645\u0652\u062a\u0650 "
            "\u0627\u0644\u0637\u0651\u064e\u0627\u0644\u0650\u0628\u064e"
        )
        == "\u0623\u0646\u062a\u0650 \u0639\u0644\u0651\u0645\u062a\u0650 \u0627\u0644\u0637\u0651\u0627\u0644\u0628"
    )


def test_attached_kaf_pronoun_with_internal_vowels():
    # لَكِ مِنِّي جَزِيلُ الشُّكْرِ -> لكِ منّي جزيل الشّكر
    assert (
        strip_arabic_diacritics(
            "\u0644\u064e\u0643\u0650 "
            "\u0645\u0650\u0646\u0651\u0650\u064a "
            "\u062c\u064e\u0632\u0650\u064a\u0644\u064f "
            "\u0627\u0644\u0634\u0651\u064f\u0643\u0652\u0631\u0650"
        )
        == "\u0644\u0643\u0650 \u0645\u0646\u0651\u064a \u062c\u0632\u064a\u0644 \u0627\u0644\u0634\u0651\u0643\u0631"
    )


def test_smart_preserves_feminine_attachments():
    # منكِ عنكِ إليكِ عليكِ فيكِ معكِ
    text = (
        "\u0645\u0646\u0643\u0650 \u0639\u0646\u0643\u0650 \u0625\u0644\u064a\u0643\u0650 "
        "\u0639\u0644\u064a\u0643\u0650 \u0641\u064a\u0643\u0650 \u0645\u0639\u0643\u0650"
    )
    assert strip_arabic_diacritics(text) == text


def test_preserves_feminine_kaf_with_shadda_and_general_endings():
    # إِنَّكِ شَاهَدْتِ الفِلْمَ كَامِلًا -> إنّكِ شاهدتِ الفلم كاملًا
    out = strip_arabic_diacritics(
        "\u0625\u0650\u0646\u0651\u064e\u0643\u0650 "
        "\u0634\u064e\u0627\u0647\u064e\u062f\u0652\u062a\u0650 "
        "\u0627\u0644\u0641\u0650\u0644\u0652\u0645\u064e "
        "\u0643\u064e\u0627\u0645\u0650\u0644\u064b\u0627"
    )
    assert out == (
        "\u0625\u0646\u0651\u0643\u0650 "
        "\u0634\u0627\u0647\u062f\u062a\u0650 "
        "\u0627\u0644\u0641\u0644\u0645 "
        "\u0643\u0627\u0645\u0644\u064b\u0627"
    )
    # Both Kasras (Kaf and feminine Ta') survive; Shadda and Tanween kept.
    assert out.count("\u0650") == 2
    assert "\u0651" in out and "\u064b" in out

    # General trailing feminine Kaf across other words.
    for word, expected in (
        ("\u0643\u064e\u0623\u0646\u0651\u064e\u0643\u0650", "\u0643\u0623\u0646\u0651\u0643\u0650"),
        ("\u0643\u0650\u062a\u064e\u0627\u0628\u064f\u0643\u0650", "\u0643\u062a\u0627\u0628\u0643\u0650"),
        ("\u0631\u064e\u0623\u064e\u064a\u0652\u062a\u064f\u0643\u0650", "\u0631\u0623\u064a\u062a\u0643\u0650"),
    ):
        assert strip_arabic_diacritics(word) == expected


def test_shadda_strips_bundled_vowels_leaving_bare_shadda():
    # الشُّرْطَةَ -> الشّرطة (Damma/Sukun/Fatha stripped, bare Shadda kept)
    assert (
        tashkeel_remove(
            "\u0627\u0644\u0634\u0651\u064f\u0631\u0652\u0637\u064e\u0629\u064e"
        )
        == "\u0627\u0644\u0634\u0651\u0631\u0637\u0629"
    )


def test_full_sentence_tashkeel_removal():
    source = (
        "\u0623\u064e\u0646\u0652\u062a\u0650 "
        "\u0623\u064e\u062e\u0652\u0641\u064e\u064a\u0652\u062a\u0650 "
        "\u0633\u0650\u0644\u064e\u0627\u062d\u064e\u0643\u0650 "
        "\u0647\u064f\u0646\u064e\u0627\u060c "
        "\u0648\u064e\u062e\u064e\u062f\u064e\u0639\u0652\u062a\u0650 "
        "\u0627\u0644\u0634\u0651\u064f\u0631\u0652\u0637\u064e\u0629\u064e "
        "\u0639\u064e\u0645\u0652\u062f\u064b\u0627!"
    )
    expected = (
        "\u0623\u0646\u062a\u0650 "
        "\u0623\u062e\u0641\u064a\u062a\u0650 "
        "\u0633\u0644\u0627\u062d\u0643\u0650 "
        "\u0647\u0646\u0627\u060c "
        "\u0648\u062e\u062f\u0639\u062a\u0650 "
        "\u0627\u0644\u0634\u0651\u0631\u0637\u0629 "
        "\u0639\u0645\u062f\u064b\u0627!"
    )
    assert strip_arabic_diacritics(source) == expected


def test_all_tanween_types_are_preserved():
    # Dammatan (U+064C) and Kasratan (U+064D) must survive like Fathatan.
    assert "\u064c" in strip_arabic_diacritics(
        "\u0643\u0650\u062a\u064e\u0627\u0628\u064c"
    )
    assert "\u064d" in strip_arabic_diacritics(
        "\u0643\u0650\u062a\u064e\u0627\u0628\u064d"
    )


def test_non_arabic_and_punctuation_unaffected():
    text = "Hello, world! 100% (test) [ok] - dash."
    assert strip_arabic_diacritics(text) == text


def test_preserves_shadda():
    assert strip_arabic_diacritics(ALLAMA_IN) == ALLAMA_OUT
    assert "\u0651" in strip_arabic_diacritics(ALLAMA_IN)


def test_removes_sukun_superscript_and_small_zero():
    text = "\u0628\u0652\u062a\u0670\u0645\u06df"
    out = strip_arabic_diacritics(text)
    assert "\u0652" not in out
    assert "\u0670" not in out
    assert "\u06df" not in out


def test_non_arabic_text_is_untouched():
    content = "Hello world, 123. [Music] (sighs)"
    assert strip_arabic_diacritics(content) == content


def test_timestamps_and_cue_indices_are_untouched():
    content = _srt(DHAHABA_IN)
    out = strip_arabic_diacritics(content)
    assert out.startswith("1\n00:00:01,000 --> 00:00:03,000")
    assert DHAHABA_OUT in out


def test_tag_attributes_are_untouched():
    content = _srt(f'<font color="#00FF00">{DHAHABA_IN}</font>')
    out = strip_arabic_diacritics(content)
    assert 'color="#00FF00"' in out
    assert DHAHABA_OUT in out


def test_bytes_wrapper_and_idempotency():
    raw = _srt(DHAHABA_IN).encode("utf-8")
    out = strip_arabic_diacritics_bytes(raw).decode("utf-8")
    assert DHAHABA_OUT in out
    # Running twice is a no-op.
    assert strip_arabic_diacritics(out) == out


def test_config_roundtrip_and_cache_key():
    assert parse_user_config(encode_user_config()).strip_diacritics is False
    assert (
        parse_user_config(encode_user_config(strip_diacritics=True)).strip_diacritics
        is True
    )
    base = dict(media_type="movie", imdb_id="tt0111161")
    assert build_cache_key(**base, strip_diacritics=True) != build_cache_key(
        **base, strip_diacritics=False
    )


def test_pipeline_applies_only_when_enabled():
    src = _srt(DHAHABA_IN).encode("utf-8")
    on = _build_subtitle_response(src, "r", "srt", strip_diacritics=True).body.decode()
    off = _build_subtitle_response(src, "r", "srt", strip_diacritics=False).body.decode()
    assert DHAHABA_OUT in on
    assert "\u064e" not in on
    assert DHAHABA_IN in off
