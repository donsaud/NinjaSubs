"""Evidenced legacy visual-order mirrors, without editing real subtitle artifacts."""

import re

import pytest

from app.main import _run_subtitle_optimization_pipeline
from app.utils.cleaners import CleanOptions
from app.utils.rtl import (
    RLM,
    fix_rtl_punctuation,
    fix_rtl_punctuation_bytes,
    normalize_legacy_dialogue_ellipsis,
    normalize_legacy_multiline_quotes,
    normalize_legacy_visual_order,
)

CORPUS = [
    ("(أنا لا أحلم (والت", "أنا لا أحلم (والت)"),
    ("(سمعته (كريس", "سمعته (كريس)"),
    ("(أجل، في الـ(كيز -", "أجل، في الـ(كيز) -"),
    ("(لا، بل (فورت لودردَيل -", "لا، بل (فورت لودردَيل) -"),
    ("(صحيح، (فورت لودردَيل -", "صحيح، (فورت لودردَيل) -"),
    ("كريس) هذا رائع)", "(كريس) هذا رائع"),
    ("دكستر)، ما هذا؟) -", "(دكستر)، ما هذا؟ -"),
    ("فنس)، هلّا حللتَ مكاني صباح اليوم) -", "(فنس)، هلّا حللتَ مكاني صباح اليوم -"),
    ("دكستر) محلّل لطخات دم)", "(دكستر) محلّل لطخات دم"),
    ("(ما يريب في (دكستر", "ما يريب في (دكستر)"),
    ("،لاغوِرتا) بريئة)", "(لاغوِرتا)، بريئة"),
    ("،روبرت بَيلي)، عمره 42 عامًا)", "(روبرت بَيلي)، عمره 42 عامًا"),
    ("...ماريا)، وفكّرتُ في أنّه)", "(ماريا)...، وفكّرتُ في أنّه"),
    ("…ماريا)، وفكّرتُ في أنّه)", "(ماريا)…، وفكّرتُ في أنّه"),
    ("...(عندما قُتل (مايك أندرسن", "عندما قُتل (مايك أندرسن)..."),
    ("…(عندما قُتل (مايك أندرسن", "عندما قُتل (مايك أندرسن)…"),
    ('"(سمّمتِ (دبرا" -', '"سمّمتِ (دبرا)" -'),
    ('"(لم أرد لها يومًا أن تقاسي هذا يا (دكس" -', '"لم أرد لها يومًا أن تقاسي هذا يا (دكس)" -'),
    ('"...(سابقًا في (دكستر"', '"سابقًا في (دكستر)..."'),
    ('"جَيمي)، قد رجعتُ)" -', '"(جَيمي)، قد رجعتُ" -'),
    ("<i>كريس) قاس على نفسه)", "<i>(كريس) قاس على نفسه"),
    ("<i> كريس) قد رتب لمكتب البريد)", "<i> (كريس) قد رتب لمكتب البريد"),
    ("(أن ينفصل (كريس </i>", "أن ينفصل (كريس) </i>"),
    ("<b>(مرحبا (أحمد</b>", "<b>مرحبا (أحمد)</b>"),
    ('<font color="#FF0000">دكستر) ما هذا؟)</font>', '<font color="#FF0000">(دكستر) ما هذا؟</font>'),
    ("<i><b>دكس)، كيف حالك؟)</b></i>", "<i><b>(دكس)، كيف حالك؟</b></i>"),
    ("<u>(قال (أحمد</u>", "<u>قال (أحمد)</u>"),
]


@pytest.mark.parametrize("original,expected", CORPUS)
def test_evidenced_legacy_normalization_and_full_pipeline(original, expected):
    normalized = normalize_legacy_visual_order(original)
    assert normalized == expected
    assert normalize_legacy_visual_order(normalized) == normalized
    fixed = fix_rtl_punctuation(original)
    assert fixed == fix_rtl_punctuation(expected)
    assert fix_rtl_punctuation(fixed) == fixed
    assert RLM + "(" not in fixed  # Failed parenthesis anchoring experiment stays removed.
    assert re.findall(r"<[^>]*>", fixed) == re.findall(r"<[^>]*>", original)


@pytest.mark.parametrize("marker", ["-", "–", "—"])
def test_trailing_marker_preserved_for_established_dialogue_stage(marker):
    raw = f"(أجل، في الـ(كيز {marker}"
    assert normalize_legacy_visual_order(raw) == f"أجل، في الـ(كيز) {marker}"
    assert fix_rtl_punctuation(raw) == f"{RLM}{marker} أجل، في الـ(كيز){RLM}"


@pytest.mark.parametrize("original,expected", [
    ("،روبرت)، عمره 42 عامًا)", "(روبرت)، عمره 42 عامًا"),
    ("...ماريا)...، جاءت)", "(ماريا)...، جاءت"),
    ("...ماريا)…، جاءت)", "(ماريا)…، جاءت"),
    ("…(قال (أحمد…", "قال (أحمد)…"),
    ("...(قال (أحمد...", "قال (أحمد)..."),
    ("،روبرت)...، عمره 42 عامًا)", "(روبرت)...، عمره 42 عامًا"),
    ("...ماريا)،… جاءت)", "(ماريا)،… جاءت"),
])
def test_mirrored_punctuation_consumed_when_destination_exists(original, expected):
    assert normalize_legacy_visual_order(original) == expected
    assert fix_rtl_punctuation(original) == fix_rtl_punctuation(expected)


@pytest.mark.parametrize("original", [
    "مرحبا (بك)", "(مرحبا بك)", "قال (أحمد) نعم", "هذا (الأول) وليس (الثاني)",
    "(12:30)", "(Netflix)", "COVID-19", "2025-2026", "123 - 456", "اسم-مركب", "A - B",
    "- أجل، في الـ(كيز)", "- لا، بل (فورت لودردَيل)", "(دكستر) محلّل لطخات دم",
    "ما يريب في (دكستر)", "(لاغوِرتا)، بريئة", "(ماريا)...، وفكّرتُ في أنّه",
    "(مرحبا بك -", "مرحبا (بك -", "(مرحبا", "مرحبا)", "(نص(", ")نص)",
    "((مرحبا", "مرحبا))", "((قال (أحمد", "أحمد) قال)))", "(قال ((أحمد",
    "(قال ((أحمد) النص",
    "....أحمد) قال)", "،،أحمد) قال)", "……أحمد) قال)", "!أحمد) قال)",
    '"(في سيّارة (دوكس', '،هانا مكّي)، طعنَت امرأة خمس مرّات)"', '(لا أقدر على ترك (دِب"',
    '"(جرت في (فورت لودردَيل', '(إن كانت (دِب) بصحبة (برغز"',
    '"(قال "أحمد" (عربي"', "<i><b>أحمد) قال)</i>", "<i>أحمد) قال)</b>",
    "<i class='x'>أحمد) قال)</i>", "أحمد)<i> قال)", "أحمد) </i>قال)",
    "(Hello (John", "John) Hello)", "(قال (أحمد--", "(قال (أحمد - -",
])
def test_modern_malformed_or_ambiguous_forms_are_not_guessed(original):
    assert normalize_legacy_visual_order(original) == original
    fixed = fix_rtl_punctuation(original)
    assert fixed.replace(RLM, "") == original
    assert fix_rtl_punctuation(fixed) == fixed


def test_full_subtitle_bytes_and_response_pipeline_preserve_cues_tags_and_timestamps():
    raw = ("1\r\n00:00:01,000 --> 00:00:03,000\r\n(أجل، في الـ(كيز -\r\n\r\n"
           "2\r\n00:00:03,500 --> 00:00:05,500\r\n(لا، بل (فورت لودردَيل -\r\n\r\n"
           "3\r\n00:00:06,000 --> 00:00:08,000\r\nفنس)، هلّا حللتَ مكاني صباح اليوم) -\r\n").encode()
    expected = raw.decode().replace("(أجل، في الـ(كيز -", "- أجل، في الـ(كيز)").replace(
        "(لا، بل (فورت لودردَيل -", "- لا، بل (فورت لودردَيل)").replace(
        "فنس)، هلّا حللتَ مكاني صباح اليوم) -", "- (فنس)، هلّا حللتَ مكاني صباح اليوم")
    fixed = fix_rtl_punctuation_bytes(raw)
    assert fixed.decode().replace(RLM, "") == expected
    assert fix_rtl_punctuation_bytes(fixed) == fixed
    options = CleanOptions(False, False, False, False, False, False, False)
    assert _run_subtitle_optimization_pipeline(raw, enable_rtl_fix=True,
                                              enable_ad_removal=False, options=options) == fixed
    assert _run_subtitle_optimization_pipeline(raw, enable_rtl_fix=False,
                                              enable_ad_removal=False, options=options) == raw


@pytest.mark.parametrize("wrap", [('"', '"'), ("«", "»"), ("“", "”"), ("‹", "›"), ("'", "'")])
def test_balanced_whole_line_quotes_inside_safe_markup(wrap):
    opening, closing = wrap
    raw = f'<font color="#FF0000"><i>{opening}(قال (أحمد{closing}</i></font> -'
    expected = f'<font color="#FF0000"><i>{opening}قال (أحمد){closing}</i></font> -'
    assert normalize_legacy_visual_order(raw) == expected
    assert fix_rtl_punctuation(raw) == fix_rtl_punctuation(expected)
    assert fix_rtl_punctuation(fix_rtl_punctuation(raw)) == fix_rtl_punctuation(raw)


def test_safe_flat_existing_parentheses_and_inline_overrides_are_preserved():
    raw = r"...(قال {\pos(100,200)}(أحمد) ثم (محمد"
    expected = r"قال {\pos(100,200)}(أحمد) ثم (محمد)..."
    assert normalize_legacy_visual_order(raw) == expected
    prefix = "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,"
    fixed = fix_rtl_punctuation(prefix + raw)
    assert fixed.startswith(prefix)
    assert r"{\pos(100,200)}" in fixed
    assert fix_rtl_punctuation(fixed) == fixed


MULTILINE_CORPUS = [
    ('،تواجه (دِب) هذه المحنة على طريقتها"\n"إنّما آمل أن تحادثني بين الفينة والأخرى',
     '"تواجه (دِب) هذه المحنة على طريقتها،\nإنّما آمل أن تحادثني بين الفينة والأخرى"'),
    ('أنتِ الشخص الوحيد"\n"الذي لا أضطرّ إلى إخفاء أيّ شيء عنه',
     '"أنتِ الشخص الوحيد\nالذي لا أضطرّ إلى إخفاء أيّ شيء عنه"'),
    ('،هانا مكّي)، طعنَت امرأة خمس مرّات)"\n"لا أفكّر إلّا في مجازاتها بما تستحقّ',
     '"(هانا مكّي)، طعنَت امرأة خمس مرّات\nلا أفكّر إلّا في مجازاتها بما تستحقّ"'),
    ('(لا أقدر على ترك (دِب"\n"تتحمّل وِزر جرمي',
     '"لا أقدر على ترك (دِب)\nتتحمّل وِزر جرمي"'),
    ('...البنزين، النار"\n"ستمحو أيّ أثر لوجودنا هنا',
     '"...البنزين، النار\nستمحو أيّ أثر لوجودنا هنا"'),
    ('...(معكم (جَيمي باتيستا"\n"اتركوا رسالةً من فضلكم',
     '"...معكم (جَيمي باتيستا)\nاتركوا رسالةً من فضلكم"'),
    ('صوفيا فوِنتس)، مكان عملها الحاليّ)"\n"...(هو (نادي يخوت مرفأ السَرْو',
     '"(صوفيا فوِنتس)، مكان عملها الحاليّ\n...هو (نادي يخوت مرفأ السَرْو)"'),
    ('،ثمّة أجزاء منّي لا أستطيع مشاطرتها ابني"\n"...(أمّا (زاك',
     '"ثمّة أجزاء منّي لا أستطيع مشاطرتها ابني،\n...أمّا (زاك)"'),
    ('،ثمّة جريمة أخرى عليّ ارتكابها"\n"ليست سهلة',
     '"ثمّة جريمة أخرى عليّ ارتكابها،\nليست سهلة"'),
    ('...(هكتُر استرادا)"\n"كنتُ في الثالثة حين قتلتَ أمّي',
     '"...(هكتُر استرادا)\nكنتُ في الثالثة حين قتلتَ أمّي"'),
    ('كلّ نفقاتها لثلاثة أسابيع"\n"(جرت في (فورت لودردَيل',
     '"كلّ نفقاتها لثلاثة أسابيع\nجرت في (فورت لودردَيل)"'),
    ('…(معكم (جَيمي باتيستا"\n"اتركوا رسالةً من فضلكم',
     '"…معكم (جَيمي باتيستا)\nاتركوا رسالةً من فضلكم"'),
    ('،مرحبا بكم،"\n"أهلا وسهلا', '"مرحبا بكم،\nأهلا وسهلا"'),
]


def _multiline_srt(body):
    return "1\n00:00:01,000 --> 00:00:03,000\n" + body + "\n\n"


@pytest.mark.parametrize("raw,expected", MULTILINE_CORPUS)
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_multiline_legacy_quotes_same_cue_and_full_pipeline(raw, expected, newline):
    source = _multiline_srt(raw).replace("\n", newline)
    wanted = _multiline_srt(expected).replace("\n", newline)
    normalized = normalize_legacy_multiline_quotes(source)
    assert normalized == wanted
    assert normalize_legacy_multiline_quotes(normalized) == normalized
    fixed = fix_rtl_punctuation(source)
    assert fixed.replace(RLM, "") == wanted
    assert fix_rtl_punctuation(fixed) == fixed
    assert fix_rtl_punctuation_bytes(source.encode()) == fixed.encode()
    assert len(fixed.splitlines()) == len(source.splitlines())
    options = CleanOptions(False, False, False, False, False, False, False)
    assert _run_subtitle_optimization_pipeline(source.encode(), enable_rtl_fix=True,
                                              enable_ad_removal=False, options=options) == fixed.encode()


@pytest.mark.parametrize("wrapper", ["<i>{}</i>", "<b><u>{}</u></b>",
                                     '<font color="#FF0000">{}</font>', "<i>{}", "{} </i>"])
def test_multiline_quote_markup_preserved(wrapper):
    raw, expected = MULTILINE_CORPUS[3]
    source = _multiline_srt("\n".join(wrapper.format(line) for line in raw.split("\n")))
    wanted = _multiline_srt("\n".join(wrapper.format(line) for line in expected.split("\n")))
    assert normalize_legacy_multiline_quotes(source) == wanted
    fixed = fix_rtl_punctuation(source)
    assert fixed.replace(RLM, "") == wanted
    assert fix_rtl_punctuation(fixed) == fixed
    assert re.findall(r"<[^>]*>", fixed) == re.findall(r"<[^>]*>", source)


@pytest.mark.parametrize("body", [
    '"مرحبا"', 'قال "مرحبا"', '"السطر الأول\nالسطر الثاني"',
    'مرحبا"\nسطر لا صلة له', 'مرحبا "بكم"\n"أهلا', 'Hello"\n"World',
    'مرحبا"\n"World', '<i class="x">مرحبا"</i>\n"أهلا',
    '<i>مرحبا</i>"\n"أهلا', '<i><b>مرحبا"</i>\n"أهلا',
])
def test_multiline_negative_candidates_not_normalized(body):
    source = _multiline_srt(body)
    assert normalize_legacy_multiline_quotes(source) == source


def test_multiline_does_not_infer_cues_or_cross_structural_boundaries():
    raw = 'مرحبا"\n"أهلا'
    for source in (raw, _multiline_srt('مرحبا"') + _multiline_srt('"أهلا'),
                   _multiline_srt('مرحبا"\n\n"أهلا'),
                   '1\nnot a timestamp\n' + raw):
        assert normalize_legacy_multiline_quotes(source) == source


def test_modern_multiline_quotes_and_ellipsis_remain_logical_in_full_pipeline():
    source = _multiline_srt('"...مرحبا (بك)\nأهلا وسهلا"')
    fixed = fix_rtl_punctuation(source)
    assert fixed.replace(RLM, "") == source
    assert fix_rtl_punctuation(fixed) == fixed


@pytest.mark.parametrize("first", [
    '<font color="#FF0000">مرحبا</font><font color="#0000FF">"</font>',
    '<i>مرحبا</i>"',
])
def test_quote_never_crosses_style_context(first):
    source = _multiline_srt(first + '\n"أهلا')
    assert normalize_legacy_multiline_quotes(source) == source


def test_ellipsis_keeps_style_boundary_and_whitespace():
    source = _multiline_srt('  ...<i>(قال (أحمد</i>"  \n  "أهلا وسهلا  ')
    wanted = _multiline_srt('  "...<i>قال (أحمد)</i>  \n  أهلا وسهلا"  ')
    assert normalize_legacy_multiline_quotes(source) == wanted
    fixed = fix_rtl_punctuation(source)
    assert fixed.replace(RLM, "") == wanted
    assert fix_rtl_punctuation(fixed) == fixed



def test_restored_terminal_quote_uses_existing_bidi_anchor():
    source = _multiline_srt('مرحبا بكم"\n"أهلا وسهلا')
    fixed = fix_rtl_punctuation(source)
    assert fixed.splitlines()[3] == 'أهلا وسهلا"' + RLM
    assert fix_rtl_punctuation(fixed) == fixed


def test_multiline_srt_structure_preserved_across_multiple_cues():
    first_raw, first_expected = MULTILINE_CORPUS[0]
    second_raw, second_expected = MULTILINE_CORPUS[6]
    source = ('\ufeff007\r\n00:01:02,003 --> 00:01:04,005\r\n' + first_raw.replace('\n', '\r\n')
              + '\r\n\r\n21\r\n00:02:02,003 --> 00:02:04,005\r\n'
              + second_raw.replace('\n', '\r\n') + '\r\n\r\n')
    expected = source.replace(first_raw.replace('\n', '\r\n'), first_expected.replace('\n', '\r\n'))
    expected = expected.replace(second_raw.replace('\n', '\r\n'), second_expected.replace('\n', '\r\n'))
    assert normalize_legacy_multiline_quotes(source) == expected
    fixed = fix_rtl_punctuation_bytes(source.encode())
    assert fixed.decode().replace(RLM, "") == expected
    assert fix_rtl_punctuation_bytes(fixed) == fixed


ELLIPSIS_DIALOGUE_CORPUS = [
    ("...حسنًا، اسمع -", "حسنًا، اسمع... -"),
    ("...نعم، ولكنّي -", "نعم، ولكنّي... -"),
    ("...في ظلّ هطول المطر -", "في ظلّ هطول المطر... -"),
    ("...آسف، لم -", "آسف، لم... -"),
    ("...لا أعرف وحسب، اسمع -", "لا أعرف وحسب، اسمع... -"),
    ("...(دكستر) -", "(دكستر)... -"),
    ("...(دِب) -", "(دِب)... -"),
    ("...(ما زال الأشرار يختفون، و(دوكس -", "ما زال الأشرار يختفون، و(دوكس)... -"),
    ("...دكستر)، ما هذا؟) -", "(دكستر)، ما هذا؟... -"),
    ("…(قال (أحمد -", "قال (أحمد)… -"),
    ("... مرحبا بكم -", "مرحبا بكم... -"),
    ("...مرحبا بكم… -", "مرحبا بكم… -"),
]


@pytest.mark.parametrize("raw,expected", ELLIPSIS_DIALOGUE_CORPUS)
@pytest.mark.parametrize("marker", ["-", "–", "—"])
def test_strict_dialogue_ellipsis_composes_with_parentheses_and_dash(raw, expected, marker):
    raw = raw[:-1] + marker
    expected = expected[:-1] + marker
    assert normalize_legacy_dialogue_ellipsis(raw) == expected
    assert normalize_legacy_dialogue_ellipsis(expected) == expected
    fixed = fix_rtl_punctuation(raw)
    assert fixed.replace(RLM, "") == marker + " " + expected[:-2]
    assert fix_rtl_punctuation(fixed) == fixed
    assert RLM + "(" not in fixed


@pytest.mark.parametrize("raw", [
    "...مرحبا", "…مرحبا", "...أريد أن أصعد لهم وأقول", "...ثلاثة",
    "...سأخرج الآن", "...لا أتكلم لأنه", "....مرحبا -", ".......مرحبا -",
    ".... حسنا دعنا", "....حسنا", ".......حَصلتُ على الربع الآن",
    "COVID-19", "2025-2026", "A - B", "مرحبا - بك", "مرحبا بك-",
    '" ...مرحبا - "', '...قال "مرحبا" -', "... -", "...مرحبا-", "……مرحبا -",
    "...مرحبا - بك -", "...Hello -", "...مرحبا (بك -", "...((مرحبا -",
    '<i class="x">...مرحبا -</i>', '<i>...مرحبا -</b>',
    '<i>...</i>مرحبا -', '<i>...مرحبا</i> بك -',
    '<font color="#FF0000">...</font><font color="#0000FF">مرحبا</font> -',
])
def test_strict_dialogue_ellipsis_declines_other_shapes(raw):
    assert normalize_legacy_dialogue_ellipsis(raw) == raw


@pytest.mark.parametrize("wrapper", ["<i>{}</i>", "<b><u>{}</u></b>",
                                     '<font color="#FF0000">{}</font>', "<i>{}"])
def test_strict_dialogue_ellipsis_preserves_markup(wrapper):
    raw = wrapper.format("...مرحبا بك") + " -"
    expected = wrapper.format("مرحبا بك...") + " -"
    assert normalize_legacy_dialogue_ellipsis(raw) == expected
    fixed = fix_rtl_punctuation(raw)
    assert re.findall(r"<[^>]*>", fixed) == re.findall(r"<[^>]*>", raw)
    assert "...مرحبا" not in fixed
    assert fix_rtl_punctuation(fixed) == fixed


def test_strict_ellipsis_srt_bytes_and_multiline_quote_composition():
    raw = '...مرحبا بكم -"\n"أهلا وسهلا'
    source = _multiline_srt(raw).replace("\n", "\r\n")
    expected = _multiline_srt('"- مرحبا بكم...\nأهلا وسهلا"').replace("\n", "\r\n")
    fixed = fix_rtl_punctuation_bytes(source.encode())
    assert fixed.decode().replace(RLM, "") == expected
    assert fix_rtl_punctuation_bytes(fixed) == fixed
    plain = _multiline_srt('...آسف، لم -')
    assert fix_rtl_punctuation(plain).replace(RLM, "") == _multiline_srt('- آسف، لم...')



def test_strict_dialogue_ellipsis_safe_orphan_suffix_with_marker_inside():
    raw = "...مرحبا بك - </i>"
    assert normalize_legacy_dialogue_ellipsis(raw) == "مرحبا بك... - </i>"
    assert fix_rtl_punctuation(fix_rtl_punctuation(raw)) == fix_rtl_punctuation(raw)



def test_strict_ellipsis_declines_inline_ass_style_changes():
    raw = r"...{\i1}مرحبا بك -"
    assert normalize_legacy_dialogue_ellipsis(raw) == raw
