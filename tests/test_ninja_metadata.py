"""Translator/uploader metadata must never become synchronization evidence.

Real subtitle files open with credit lines -- translator names, timing
credits, URLs, social handles, release-group tags -- that match verbatim
between two files and therefore look exactly like legitimate anchors to a
naive text matcher. A credit at a misleading timestamp is the sharpest form:
it proposes a wrong offset with full textual confidence.

These tests pin the two-gate defense:

1. The detector refuses metadata cues before indexing (they never match,
   never corroborate, never fit).
2. The fitters re-check anchor source text, so metadata cannot reach an
   offset cluster or a slope estimate even from a hand-built AnchorSet.

UNKNOWN stays usable throughout: only high-confidence metadata patterns are
refused, and corroboration plus the verifier still judge everything else.
"""

from __future__ import annotations

import pytest

from app.services.sync.ninja_points import (
    AnchorDetector,
    AnchorFitter,
    AnchorSet,
    SyncAnchor,
    _is_metadata_cue,
)


def _cue(start_ms: int, text: str, dur_ms: int = 2000) -> tuple:
    return (start_ms, start_ms + dur_ms, text)


def _srt_text(cues) -> str:
    def stamp(ms: int) -> str:
        h, r = divmod(max(0, ms), 3600000)
        m, r = divmod(r, 60000)
        s, ms2 = divmod(r, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms2:03d}"

    return "\n\n".join(
        f"{i}\n{stamp(s)} --> {stamp(e)}\n{t}"
        for i, (s, e, t) in enumerate(cues, start=1)
    ) + "\n"


# ===========================================================================
# Classifier: real-world credit shapes are metadata
# ===========================================================================


@pytest.mark.parametrize(
    "text",
    [
        "ترجمة فلان",
        "ترجمة وتعديل فلان",
        "تعديل التوقيت: XYZ",
        "إعداد وتنسيق: فلان",
        "ترجمة وتعديل: فلان",
        "Synced by XYZ",
        "Translated by XYZ",
        "Subtitle by XYZ",
        "Translated by XYZ www.example.com",
        "Telegram: @example",
        "Timing by subs4free",
        "Ripped by GOLD",
        "Encoded by Anonymous",
    ],
)
def test_credit_shapes_are_metadata(text):
    assert _is_metadata_cue(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "Run!",
        "Hello.",
        "Where are we going?",
        "فلان",  # bare name: ambiguous, must stay usable
        "Stand by me",  # "by" without attribution wording
        "The translation was bad",  # attribution word, no authorship signal
        "Thank you",
        "To be continued...",
        # Long lines are never metadata even with trigger words inside.
        "The translation by Habermas of this obscure text took him years of "
        "careful scholarly work and several revised editions",
    ],
)
def test_dialogue_and_ambiguous_text_stay_usable(text):
    assert _is_metadata_cue(text) is False


def test_empty_text_is_not_metadata():
    assert _is_metadata_cue("") is False
    assert _is_metadata_cue("   ") is False


def test_classification_is_deterministic():
    samples = ["ترجمة فلان", "Run!", "Synced by XYZ", "Hello there friend"]
    assert [_is_metadata_cue(s) for s in samples] == [_is_metadata_cue(s) for s in samples]


# ===========================================================================
# False-anchor scenario: shared credit must not dominate inference
# ===========================================================================


def _dialogue_pair(offset_ms: int, n: int = 6):
    """Shared dialogue lines plus one shared credit line on both sides."""
    target = [(_cue(1000, "ترجمة فلان"))]
    reference = [(_cue(1000, "ترجمة فلان"))]
    for i in range(n):
        target.append(_cue(60000 + i * 4000, f"dialogue line number {i} here"))
        reference.append(_cue(60000 + i * 4000 + offset_ms, f"dialogue line number {i} here"))
    return target, reference


def test_shared_credit_never_becomes_an_anchor():
    target, reference = _dialogue_pair(2000)
    anchors = AnchorDetector().detect(target, reference)
    assert len(anchors) == 6
    assert all("ترجمة" not in (a.source_text or "") for a in anchors)


def test_misleading_timestamp_credit_cannot_shift_the_fit():
    """Credit at +35s offset, dialogue at +2s: the fit must say +2s."""
    target = [(
        _cue(5000, "ترجمة فلان"),
        _cue(60000 + i * 4000, f"dialogue line number {i} here"),
    ) for i in range(6)]
    # Flatten: target cues then reference cues with the credit displaced.
    target_cues = [target[0][0]] + [t[1] for t in target]
    reference_cues = [_cue(40000, "ترجمة فلان")] + [
        _cue(60000 + i * 4000 + 2000, f"dialogue line number {i} here") for i in range(6)
    ]
    anchors = AnchorDetector().detect(target_cues, reference_cues)
    assert len(anchors) == 6
    model = AnchorFitter().fit_constant(anchors)
    assert model is not None
    assert model.offset_ms == pytest.approx(2000.0, abs=1.0)


def test_metadata_only_files_yield_no_anchors():
    target = [_cue(1000 + i * 3000, t) for i, t in enumerate(["ترجمة فلان", "Synced by XYZ"])]
    reference = [_cue(3000 + i * 3000, t) for i, t in enumerate(["ترجمة فلان", "Synced by XYZ"])]
    assert len(AnchorDetector().detect(target, reference)) == 0


def test_metadata_in_one_file_only_cannot_anchor():
    target = [_cue(1000, "ترجمة فلان"), _cue(60000, "dialogue line one here")]
    reference = [_cue(62000, "dialogue line one here")]
    anchors = AnchorDetector().detect(target, reference)
    # The credit has no counterpart; the dialogue line needs corroboration
    # which a single line cannot provide.
    assert len(anchors) == 0


# ===========================================================================
# Legitimate early dialogue stays eligible
# ===========================================================================


def test_legitimate_first_dialogue_cue_remains_eligible():
    target = [_cue(2000, "Run!"), _cue(8000, "Stop right there now")]
    reference = [_cue(4000, "Run!"), _cue(10000, "Stop right there now")]
    anchors = AnchorDetector().detect(target, reference)
    starts = sorted(a.subtitle_ms for a in anchors)
    assert starts and starts[0] == 2000


# ===========================================================================
# Fit-time recheck: hand-built sets cannot smuggle metadata into a fit
# ===========================================================================


def test_constant_fit_refuses_metadata_contaminated_set():
    contaminated = AnchorSet(
        anchors=(
            SyncAnchor(subtitle_ms=60000, reference_ms=62000, confidence=1.0,
                       support=2, source_text="dialogue line one here"),
            # Misleading credit anchor smuggled past the detector.
            SyncAnchor(subtitle_ms=5000, reference_ms=40000, confidence=1.0,
                       support=2, source_text="translated by xyz"),
        )
    )
    # Without the recheck the median of {+2s, +35s} would be +18.5s.
    assert AnchorFitter().fit_constant(contaminated) is None


def test_two_agreeing_credits_cannot_form_a_cluster():
    """The sharp case: TWO credits at nearly the same wrong offset would form
    a corroborated cluster of their own (+35s, +35.2s) and outvote a single
    dialogue anchor. The fit-time recheck must drop them before clustering,
    leaving one dialogue anchor -- below minimum -- instead of a wrong model.
    """
    contaminated = AnchorSet(
        anchors=(
            SyncAnchor(subtitle_ms=60000, reference_ms=62000, confidence=1.0,
                       support=3, source_text="dialogue line one here"),
            SyncAnchor(subtitle_ms=5000, reference_ms=40000, confidence=1.0,
                       support=3, source_text="translated by xyz"),
            SyncAnchor(subtitle_ms=9000, reference_ms=44200, confidence=1.0,
                       support=3, source_text="synced by xyz"),
        )
    )
    # Without the recheck the largest cluster is {+35s, +35.2s} and the model
    # would claim offset +35.1s. With it, only the dialogue anchor remains.
    assert AnchorFitter().fit_constant(contaminated) is None


def test_linear_fit_refuses_metadata_contaminated_set():
    honest = [
        SyncAnchor(subtitle_ms=float(i * 600000), reference_ms=float(i * 600000 + 2000),
                   confidence=1.0, support=4, source_text=f"dialogue passage {i}")
        for i in range(3)
    ]
    smuggled = SyncAnchor(subtitle_ms=900000.0, reference_ms=990000.0,
                           confidence=1.0, support=4, source_text="synced by xyz")
    # With the credit dropped, 3 honest anchors remain but span only... the
    # honest three span 1.2M ms on slope 1.0: fits fine. The assertion that
    # matters is that the credit never participates: remove one honest anchor
    # so that WITHOUT the recheck the slope would skew, and WITH it the set
    # still fits cleanly at slope 1.0.
    contaminated = AnchorSet(anchors=tuple(honest[:2] + [smuggled]))
    # Two honest anchors are below the linear minimum: refusal either way.
    assert AnchorFitter().fit_linear(contaminated) is None
    full = AnchorSet(anchors=tuple(honest + [smuggled]))
    model = AnchorFitter().fit_linear(full)
    assert model is not None
    assert model.slope == pytest.approx(1.0, abs=1e-9)
    assert model.intercept_ms == pytest.approx(2000.0, abs=1.0)


def test_long_markup_credit_cannot_shift_fit_despite_evading_classifier():
    """Real-world shape (Dexter d010faffc7142fdb cue 0): a 94-char markup-heavy
    credit block that the classifier does NOT catch (length cap, handle
    preceded by markup). Placed at divergent timing (+35s vs +2s dialogue), it
    must still not move the fit: the offset window and corroboration gates
    contain what the classifier missed. Defense in depth, pinned."""
    # Shape mirrors the real Dexter cue: markup-heavy, handle glued directly
    # to a tag close (so the handle pattern, which needs whitespace or
    # start-of-line before @, cannot see it), and over the length cap.
    credit = (
        "== Translated by ABCDEFGHIJKLMNOPQRST ==<font color=#00FFFF>@ghijklmnopqrs</font>=="
    )
    assert len(credit) > 80
    assert _is_metadata_cue(credit) is False  # documents the known classifier gap
    target = [(5000, 7000, credit)] + [
        (60000 + i * 4000, 62000 + i * 4000, "dialogue line number %d here" % i)
        for i in range(14)
    ]
    reference = [(40000, 42000, credit)] + [
        (60000 + i * 4000 + 2000, 62000 + i * 4000 + 2000, "dialogue line number %d here" % i)
        for i in range(14)
    ]
    anchors = AnchorDetector().detect(target, reference)
    assert not any("ranslated" in (a.source_text or "") for a in anchors)
    model = AnchorFitter().fit_constant(anchors)
    assert model is not None
    assert model.offset_ms == pytest.approx(2000.0, abs=250.0)


def test_detector_stats_report_metadata_refusals():
    stats: dict = {}
    target = [_cue(1000, "ترجمة فلان"), _cue(60000, "dialogue line one here"),
              _cue(64000, "dialogue line two here")]
    reference = [_cue(3000, "ترجمة فلان"), _cue(62000, "dialogue line one here"),
                 _cue(66000, "dialogue line two here")]
    AnchorDetector().detect(target, reference, stats=stats)
    assert stats.get("metadata_skipped") == 2
