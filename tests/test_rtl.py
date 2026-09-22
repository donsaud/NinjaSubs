"""Tests for RTL subtitle normalization:
- legacy "Reverse RTL start/end" (leading punctuation moved to the logical end)
- bracket inversion mirroring
- trailing neutral punctuation RLM fix
- ASS/SSA override tags and hard line breaks
"""

from app.extractor import transcode_to_utf8
from app.utils.rtl import RLM, contains_rtl, fix_rtl_punctuation

# ============================================================================
# 1. LEGACY "REVERSE RTL START/END"
# ============================================================================


def test_legacy_leading_period_moved_to_end():
    assert fix_rtl_punctuation(".والدي كان مزارعاً") == f"والدي كان مزارعاً.{RLM}"


def test_legacy_leading_exclamation_moved_to_end():
    assert fix_rtl_punctuation("! كلا، سأتولى هذا") == f"كلا، سأتولى هذا!{RLM}"


def test_legacy_leading_ellipsis_moved_to_end():
    assert fix_rtl_punctuation("... إطفاء كل") == f"إطفاء كل...{RLM}"


def test_legacy_leading_question_mark_moved_to_end():
    assert fix_rtl_punctuation("?هل هذا صحيح") == f"هل هذا صحيح?{RLM}"


def test_legacy_leading_arabic_comma_moved_to_end():
    assert fix_rtl_punctuation("،مرحبا") == f"مرحبا،{RLM}"


def test_legacy_fix_is_idempotent():
    once = fix_rtl_punctuation(".والدي كان مزارعاً")
    twice = fix_rtl_punctuation(once)
    assert once == twice
    assert once.count(RLM) == 1


def test_legacy_fix_inside_full_srt():
    srt = (
        "1\n"
        "00:00:01,000 --> 00:00:04,000\n"
        ".والدي كان مزارعاً\n"
    )
    fixed = fix_rtl_punctuation(srt)
    assert f"والدي كان مزارعاً.{RLM}" in fixed
    assert ".والدي" not in fixed


# ============================================================================
# 2. BRACKET / PARENTHESIS INVERSION
# ============================================================================


def test_inverted_paired_bracket_is_mirrored():
    assert fix_rtl_punctuation(")مرحبا(") == f"(مرحبا){RLM}"


def test_lone_leading_closing_bracket_moved_and_mirrored():
    assert fix_rtl_punctuation(")مرحبا") == f"مرحبا({RLM}"


def test_correct_opening_bracket_is_preserved():
    assert fix_rtl_punctuation("(مرحبا)") == f"(مرحبا){RLM}"


def test_same_direction_boundary_brackets_are_wrapped():
    """Legacy lines with the same bracket direction at both ends (e.g. ``(نص(``)."""
    assert fix_rtl_punctuation("(نص(") == f"(نص){RLM}"
    assert fix_rtl_punctuation(")نص)") == f"(نص){RLM}"
    assert fix_rtl_punctuation("[نص[") == f"[نص]{RLM}"


def test_inner_name_bracket_is_wrapped_not_whole_line():
    """A bracket glued to an inner name must wrap only that name, never the line."""
    expected = f"آسفة بشأن زوجتك يا سيّد (كوبر){RLM}"
    # Inner opening bracket with a stray sentence-leading bracket.
    assert fix_rtl_punctuation("(آسفة بشأن زوجتك يا سيّد (كوبر") == expected
    assert fix_rtl_punctuation(")آسفة بشأن زوجتك يا سيّد (كوبر") == expected
    # Lone inner opening and lone trailing closing (missing counterpart).
    assert fix_rtl_punctuation("آسفة بشأن زوجتك يا سيّد (كوبر") == expected
    assert fix_rtl_punctuation("آسفة بشأن زوجتك يا سيّد كوبر)") == expected


def test_inner_bracket_is_not_moved_to_outer_boundary():
    """Regression: the whole-sentence wrap must not be produced for a name mention."""
    fixed = fix_rtl_punctuation("(آسفة بشأن زوجتك يا سيّد (كوبر")
    assert not fixed.startswith("(آسفة")
    assert "(آسفة بشأن زوجتك يا سيّد كوبر)" not in fixed


def test_leading_stray_bracket_wraps_trailing_name():
    """A stray leading bracket must wrap the trailing name, never the first word."""
    expected = f"آسفة لما حصل لزوجتك، يا (كوبر).{RLM}"
    assert fix_rtl_punctuation(")آسفة لما حصل لزوجتك، يا كوبر.") == expected
    assert fix_rtl_punctuation(")آسفة لما حصل لزوجتك، يا (كوبر.") == expected
    # Same corruption but with the bracket glued after the first word.
    assert fix_rtl_punctuation("آسفة) لما حصل لزوجتك، يا كوبر.") == expected


def test_first_word_is_never_parenthesized():
    """Regression for the reported bug: the first word must not be wrapped."""
    for line in (
        ")آسفة لما حصل لزوجتك، يا كوبر.",
        ")آسفة لما حصل لزوجتك، يا (كوبر.",
        "آسفة) لما حصل لزوجتك، يا كوبر.",
    ):
        fixed = fix_rtl_punctuation(line)
        assert not fixed.startswith("(آسفة")
        assert "(آسفة)" not in fixed
        assert fixed.startswith("آسفة")


def test_leading_stray_bracket_without_period_wraps_trailing_name():
    assert (
        fix_rtl_punctuation(")آسفة لما حصل لزوجتك، يا كوبر")
        == f"آسفة لما حصل لزوجتك، يا (كوبر){RLM}"
    )


def test_well_formed_whole_sentence_wrap_is_preserved():
    assert fix_rtl_punctuation("(نص عربي)") == f"(نص عربي){RLM}"


def test_embedded_term_brackets_are_preserved():
    """An embedded parenthetical term must stay intact (no bracket jumping)."""
    assert fix_rtl_punctuation("يا سيد (كوبر)") == f"يا سيد (كوبر){RLM}"


def test_nested_balanced_brackets_are_preserved():
    assert fix_rtl_punctuation("(نص (توضيح))") == f"(نص (توضيح)){RLM}"


def test_valid_prefix_parenthetical_is_not_rewrapped():
    """A well-formed parenthetical followed by text must not be turned into a wrap."""
    assert fix_rtl_punctuation("(نص) وكذا") == "(نص) وكذا"


def test_bracket_fix_is_idempotent():
    once = fix_rtl_punctuation("(آسفة بشأن زوجتك يا سيّد (كوبر")
    twice = fix_rtl_punctuation(once)
    assert once == twice


def test_speaker_dash_with_brackets_is_isolated_at_start():
    line = "- (كوبر) قال لي"
    assert fix_rtl_punctuation(line) == f"{RLM}{line}"


# ============================================================================
# 3. SAFETY (SPEAKER DASH & NON-RTL)
# ============================================================================


def test_speaker_dash_is_isolated_at_visual_start():
    line = "- مرحبا بك يا صديقي"
    assert fix_rtl_punctuation(line) == f"{RLM}{line}"


def test_speaker_dash_with_trailing_period_gets_both_rlm_marks():
    line = "- مرحبا بك يا صديقي."
    assert fix_rtl_punctuation(line) == f"{RLM}{line}{RLM}"


def test_speaker_dash_isolation_is_idempotent():
    once = fix_rtl_punctuation("- مرحبا بك يا صديقي.")
    assert fix_rtl_punctuation(once) == once


def test_en_dash_and_em_dash_are_isolated():
    assert fix_rtl_punctuation("– مرحبا") == f"{RLM}– مرحبا"
    assert fix_rtl_punctuation("— مرحبا") == f"{RLM}— مرحبا"


def test_non_rtl_text_is_untouched():
    assert fix_rtl_punctuation("Hello world.") == "Hello world."
    assert fix_rtl_punctuation("...") == "..."
    assert fix_rtl_punctuation("") == ""


# ============================================================================
# 4. TRAILING NEUTRAL PUNCTUATION (RLM)
# ============================================================================


def test_line_without_trailing_punctuation_is_untouched():
    line = "مرحبا بك في ستريميو"
    assert fix_rtl_punctuation(line) == line


def test_trailing_period_gets_rlm():
    line = "تستخدم فيما يطلق عليه الرنين المغناطيسي."
    assert fix_rtl_punctuation(line) == f"{line}{RLM}"


def test_trailing_ellipsis_gets_rlm():
    assert fix_rtl_punctuation("إلى هذا بدلاً عني..").endswith(f"..{RLM}")


def test_arabic_question_mark_is_strong_rtl():
    """U+061F ARABIC QUESTION MARK is Bidi_Class=AL, so it already renders at the
    logical end and must not be altered."""
    import unicodedata

    assert unicodedata.bidirectional("\u061f") == "AL"
    line = "كيف حالك؟"
    assert fix_rtl_punctuation(line) == line


def test_crlf_and_trailing_whitespace_preserved():
    text = "سطر عربي.\r\nسطر ثان\r\n"
    assert fix_rtl_punctuation(text) == f"سطر عربي.{RLM}\r\nسطر ثان\r\n"


# ============================================================================
# 5. ASS / SSA HANDLING
# ============================================================================


def test_ass_hard_line_breaks_are_handled():
    line = "Dialogue: 0,0:00:01.00,0:00:04.00,Default,,0,0,0,,سطر أول.\\Nسطر ثان"
    fixed = fix_rtl_punctuation(line)
    assert f"سطر أول.{RLM}\\Nسطر ثان" in fixed


def test_ass_override_tags_preserved_and_legacy_punctuation_moved():
    line = "Dialogue: 0,0:00:01.00,0:00:04.00,Default,,0,0,0,,{\\i1}.مرحبا"
    fixed = fix_rtl_punctuation(line)
    assert fixed.endswith(f"{{\\i1}}مرحبا.{RLM}")


def test_ass_trailing_tags_keep_rlm_before_them():
    line = "Dialogue: 0,0:00:01.00,0:00:04.00,Default,,0,0,0,,مرحبا بك.{\\r}"
    fixed = fix_rtl_punctuation(line)
    assert fixed.endswith(f"مرحبا بك.{RLM}{{\\r}}")


def test_ass_override_tag_parentheses_are_not_treated_as_text():
    """Parentheses inside {\\pos(...)} must not be counted or rebalanced."""
    line = "Dialogue: 0,0:00:01.00,0:00:04.00,Default,,0,0,0,,{\\pos(100,200)}نص عربي"
    assert fix_rtl_punctuation(line) == line


def test_ass_override_tag_with_inverted_brackets():
    line = "Dialogue: 0,0:00:01.00,0:00:04.00,Default,,0,0,0,,{\\i1}(نص("
    fixed = fix_rtl_punctuation(line)
    assert fixed.endswith(f"{{\\i1}}(نص){RLM}")


# ============================================================================
# 6. LEADING QUOTE CLUSTERS
# ============================================================================


def test_leading_quote_cluster_pairs_inner_quote():
    """Reported case: ``."حول هراء "أبولو ...`` -> ``حول هراء "أبولو"...``."""
    assert (
        fix_rtl_punctuation('."حول هراء "أبولو ...')
        == f'حول هراء "أبولو"...{RLM}'
    )


def test_leading_quote_cluster_order_variants():
    assert (
        fix_rtl_punctuation('".حول هراء "أبولو ...')
        == f'حول هراء "أبولو"...{RLM}'
    )
    assert (
        fix_rtl_punctuation('."حول هراء "أبولو')
        == f'حول هراء "أبولو".{RLM}'
    )


def test_leading_quote_cluster_is_idempotent():
    once = fix_rtl_punctuation('."حول هراء "أبولو ...')
    assert fix_rtl_punctuation(once) == once


def test_balanced_quotes_are_preserved():
    assert fix_rtl_punctuation('"مرحبا"') == f'"مرحبا"{RLM}'


# ============================================================================
# 6b. QUOTED PHRASE EDGE PUNCTUATION
# ============================================================================


def test_quoted_arabic_leading_period_moves_after_closing_quote():
    """Reported case: ``".نطلق عليه صندوق السماء"`` used to flip the dot to the front."""
    assert (
        fix_rtl_punctuation('".نطلق عليه صندوق السماء"')
        == f'"نطلق عليه صندوق السماء".{RLM}'
    )


def test_quoted_arabic_trailing_period_inside_quote_moves_outside():
    assert (
        fix_rtl_punctuation('"نطلق عليه صندوق السماء."')
        == f'"نطلق عليه صندوق السماء".{RLM}'
    )


def test_quoted_arabic_guillemets_leading_period():
    assert fix_rtl_punctuation("«.نطلق عليه صندوق السماء»") == f"«نطلق عليه صندوق السماء».{RLM}"


def test_quoted_arabic_without_edge_punctuation_is_untouched():
    assert fix_rtl_punctuation('"نطلق عليه صندوق السماء"') == f'"نطلق عليه صندوق السماء"{RLM}'


def test_quoted_edge_punctuation_fix_is_idempotent():
    once = fix_rtl_punctuation('".نطلق عليه صندوق السماء"')
    assert fix_rtl_punctuation(once) == once


# ============================================================================
# 7. TRANSCODING INTEGRATION
# ============================================================================


def test_transcode_to_utf8_preserves_raw_text_for_cp1256():
    """Transcoding only fixes encoding; RTL normalization happens later (serve time)."""
    text = "1\n00:00:01,000 --> 00:00:05,000\n.أهلاً وسهلاً بكم\n"
    decoded = transcode_to_utf8(text.encode("cp1256")).decode("utf-8")
    assert decoded == text
    assert RLM not in decoded


def test_transcode_to_utf8_leaves_english_untouched():
    text = "1\n00:00:01,000 --> 00:00:05,000\nHello world.\n"
    assert transcode_to_utf8(text.encode("utf-8")).decode("utf-8") == text


def test_contains_rtl_detection():
    assert contains_rtl("مرحبا") is True
    assert contains_rtl("Hello") is False
    assert contains_rtl("") is False


# ============================================================================



# ============================================================================
# 8. PRE-REVERSED DIALOGUE DASHES
# ============================================================================


def test_pre_reversed_trailing_dash_moved_to_front():
    assert (
        fix_rtl_punctuation("هل استمتعتِ كثيراً؟ -")
        == f"{RLM}- هل استمتعتِ كثيراً؟"
    )


def test_pre_reversed_dot_dash_normalized():
    assert fix_rtl_punctuation(".أجل -") == f"{RLM}- أجل.{RLM}"
    assert fix_rtl_punctuation("- .أجل") == f"{RLM}- أجل.{RLM}"


def test_pre_reversed_multiline_cue():
    line1 = "هل استمتعتِ كثيراً؟ -"
    line2 = ".أجل -"
    source = line1 + chr(10) + line2
    expected = (
        f"{RLM}- هل استمتعتِ كثيراً؟"
        + chr(10)
        + f"{RLM}- أجل.{RLM}"
    )
    assert fix_rtl_punctuation(source) == expected


def test_standard_leading_dash_is_untouched():
    assert fix_rtl_punctuation("- مرحبا") == f"{RLM}- مرحبا"


def test_pre_reversed_fix_is_idempotent():
    once = fix_rtl_punctuation("هل استمتعتِ كثيراً؟ -")
    assert fix_rtl_punctuation(once) == once
