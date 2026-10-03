"""The ``already aligned`` serving gate must rest on measured evidence.

Production history: three real releases of the same film were served verbatim
as "target already aligned (median offset +0.06s / +0.14s / +0.06s)" while
their residual p95 was 4401 / 4588 / 4401ms and playback was seconds out. The
median was computed by pairing every target cue with its *nearest* reference
cue, which in a densely cued subtitle collapses towards zero no matter how far
apart the timelines really are.

These tests pin the replacement: a verdict that needs real cue correspondence,
residual spread inside the verifier's own tolerance, and corroboration from
every section of the timeline.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from app.services.subtitle_matcher import measure_alignment_consistency
from app.services.sync.reference import is_dialogue_cue

_STEPS = itertools.cycle([2100, 3400, 1700, 4800, 2600, 3900, 2200, 4300])


def _ts(ms: int) -> str:
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def render(cues: list[tuple[int, int, str]]) -> str:
    return (
        "\n\n".join(
            f"{index}\n{_ts(start)} --> {_ts(end)}\n{text}"
            for index, (start, end, text) in enumerate(cues, 1)
        )
        + "\n"
    )


def dialogue_track(count: int = 380) -> list[tuple[int, int, str]]:
    """A reference track with irregular, human-like cue spacing."""
    cues: list[tuple[int, int, str]] = []
    position = 0
    for index in range(count):
        duration = next(_STEPS)
        cues.append((position, position + duration - 100, f"dialogue line {index} spoken here"))
        position += duration
    return cues


def measure(target: list[tuple[int, int, str]], reference: list[tuple[int, int, str]]):
    return measure_alignment_consistency(
        render(target), render(reference), is_dialogue=is_dialogue_cue
    )


def shift_after(cues: list[tuple[int, int, str]], cut_ms: int, offset_ms: int):
    return [
        (a + (0 if a < cut_ms else offset_ms), b + (0 if b < cut_ms else offset_ms), text)
        for a, b, text in cues
    ]


# --- the claim is still made when it is true ---------------------------------


def test_identical_tracks_are_reported_aligned():
    base = dialogue_track()
    result = measure(base, base)

    assert result is not None
    assert result.aligned is True
    assert result.reason == ""
    assert result.median_offset_s == pytest.approx(0.0, abs=0.01)
    assert result.p95_ms == pytest.approx(0.0, abs=1.0)
    assert result.sections_agreeing == result.sections_measured


@pytest.mark.parametrize("offset_ms", [-90, 120, 150])
def test_sub_threshold_jitter_is_still_aligned(offset_ms):
    """Real subtitles are never frame-exact; small drift must not cost an alass run."""
    base = dialogue_track()
    shifted = [(a + offset_ms, b + offset_ms, text) for a, b, text in base]

    result = measure(shifted, base)

    assert result is not None
    assert result.aligned is True


# --- the production failure ---------------------------------------------------


@pytest.mark.parametrize("offset_ms", [1400, 2000, 6000])
def test_global_desync_is_not_collapsed_to_zero(offset_ms):
    """The regression: a dense subtitle whose real offset used to read as ~0."""
    base = dialogue_track()
    shifted = [(a + offset_ms, b + offset_ms, text) for a, b, text in base]

    result = measure(shifted, base)

    assert result is not None
    assert result.aligned is False
    assert result.reason


def test_dense_evenly_spaced_tracks_do_not_hide_a_real_offset():
    """The exact production shape: metronomic cues, real displacement.

    A uniform 3s grid is the worst case for a nearest-neighbour median, because
    every cue has a close neighbour no matter how far the file has moved.
    """
    base = [(i * 3000, i * 3000 + 2600, f"line {i} of the film") for i in range(400)]
    shifted = [(a + 2000, b + 2000, text) for a, b, text in base]

    result = measure(shifted, base)

    assert result is not None
    assert result.aligned is False


def test_aligned_head_with_a_drifted_tail_is_rejected():
    """A subtitle that lines up for part of the film is not an aligned subtitle.

    Sections whose cues find no counterpart must count *against* the verdict.
    Skipping them let the half that happened to match carry the whole claim.
    """
    base = dialogue_track()
    drifted = shift_after(base, 600_000, 7500)

    result = measure(drifted, base)

    assert result is not None
    assert result.aligned is False
    # The drifted tail is caught even though the global median still reads zero.
    assert result.sections_agreeing < result.sections_measured
    assert result.reason


@pytest.mark.parametrize("offset_ms", [-3200, 4000, 7500])
def test_a_partially_displaced_tail_is_rejected(offset_ms):
    base = dialogue_track()
    result = measure(shift_after(base, 600_000, offset_ms), base)

    assert result is not None
    assert result.aligned is False


# --- the evidence itself ------------------------------------------------------


def test_non_dialogue_cues_are_not_evidence():
    """Credits, site tags and music cues must not corroborate an alignment.

    Those cues are near-identical between unrelated releases, so a subtitle made
    mostly of them can look perfectly aligned while its dialogue is seconds out.
    """
    base = dialogue_track()
    junk = [
        (0, 4000, "www.subdl.com"),
        (5000, 9000, "Downloaded from www.subdl.com"),
        (10_000, 16_000, "Subtitles by ..."),
        (20_000, 30_000, "[Music]"),
        (40_000, 46_000, "www.tvsubtitles.net"),
    ]
    # Same junk on both sides, and the dialogue pushed well out of place.
    shifted = junk + [(a + 5000, b + 5000, text) for a, b, text in base]

    result = measure(shifted, base + junk)

    assert result is not None
    assert result.aligned is False


def test_a_degenerate_handful_of_cues_yields_no_verdict():
    """Two cues can agree by coincidence; that is not evidence about a film."""
    base = dialogue_track(count=40)

    result = measure(base[:2], base)

    assert result is None


def test_samples_carry_the_first_two_matched_dialogue_cues():
    """The log has to show what the verdict was decided on, or it cannot be checked."""
    base = dialogue_track()
    result = measure(base, base)

    assert result is not None
    assert len(result.samples) == 2
    positions = [sample[0] for sample in result.samples]
    assert positions == sorted(positions)
    for _position_ms, target_text, reference_text in result.samples:
        assert target_text == reference_text
        assert len(target_text) <= 40


def test_samples_are_truncated():
    base = dialogue_track()
    verbose = [(a, b, f"a very long spoken line {'x' * 300} {a}") for a, b, _ in base]

    result = measure(verbose, verbose)

    assert result is not None
    for _position_ms, target_text, _reference_text in result.samples:
        assert len(target_text) == 40


# --- the real cached regressions ---------------------------------------------


_CACHE = Path(__file__).resolve().parents[1] / "subs_cache"
_REAL_REGRESSIONS = (
    "64fee37ea63cd03a",  # REMUX HDA, served as aligned at +0.06s, p95 4401ms
    "6e13c86162f529e2",  # YIFY 1080p, served as aligned at +0.14s, p95 4588ms
    "7b23fdbba5d35268",  # YIFY 720p, served as aligned at +0.06s, p95 4401ms
)


def _real_reference() -> str | None:
    references = sorted((_CACHE / "references").glob("*.srt"))
    return references[0].read_text(encoding="utf-8", errors="replace") if references else None


@pytest.mark.parametrize("target_id", _REAL_REGRESSIONS)
def test_cached_false_positives_are_not_served_as_aligned(target_id):
    """The three files that reached a player desynchronised by seconds."""
    reference = _real_reference()
    target_path = _CACHE / f"{target_id}.srt"
    if reference is None or not target_path.exists():
        pytest.skip("production subtitle cache is not present")

    result = measure_alignment_consistency(
        target_path.read_text(encoding="utf-8", errors="replace"),
        reference,
        is_dialogue=is_dialogue_cue,
    )

    assert result is not None
    assert result.aligned is False, f"{target_id} was served as already aligned"
    # The real displacement is seconds, not the ~0.06s the old median reported.
    assert abs(result.median_offset_s) > 0.2
