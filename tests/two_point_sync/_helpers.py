"""Shared synthetic fixtures for two_point_sync tests.

Builders only constructsubtitle inputs and apply known timing models to
*generate* fixtures. Detection, fitting and validation always run through the
production modules under test.
"""

from __future__ import annotations

from two_point_sync.parser import Cue, format_srt

SENTENCES = [
    "The lighthouse keeper arrives at dawn",
    "We missed the last train to Alexandria",
    "Her brother repairs old transistor radios",
    "The committee postponed the hearing again",
    "A sandstorm buried the northern highway",
    "He traded his watch for a paper map",
    "The orchestra tuned their instruments quietly",
    "She keeps a garden behind the bakery",
    "The ferry crossing was delayed by fog",
    "Old photographs faded in the sunlight",
    "The mechanic found the missing bolt",
    "They planted olive trees along the road",
    "The library closes at midnight on Fridays",
    "His letter never reached the capital",
    "The climbers waited out the storm",
    "She sold seashells near the harbour",
    "The trial resumed after the recess",
    "A falcon circled above the valley",
    "The baker woke before the roosters",
    "They crossed the border before sunrise",
    "The engine failed near the mountain pass",
    "She memorised every street in the district",
    "The verdict surprised the whole courtroom",
    "He carried the toolbox up the stairs",
    "Rain washed the dust from the rooftops",
    "The professor cancelled the morning lecture",
    "A merchant ship anchored in the bay",
    "The children played near the fountain",
    "She wrapped the parcel in brown paper",
    "The driver knew a shorter route",
    "Lanterns lit the narrow alleyway",
    "The archive stored maps of the delta",
]


def make_cue(number: int, start_ms: int, end_ms: int, text: str) -> Cue:
    return Cue(number=number, order=number - 1, start_ms=start_ms, end_ms=end_ms, text=text)


def build_dialogue(
    count: int,
    *,
    start_ms: int = 60_000,
    step_ms: int = 30_000,
    duration_ms: int = 2_000,
    first_number: int = 1,
    sentences: list[str] | None = None,
) -> list[Cue]:
    pool = sentences if sentences is not None else SENTENCES
    cues: list[Cue] = []
    for position in range(count):
        cues.append(
            make_cue(
                first_number + position,
                start_ms + position * step_ms,
                start_ms + position * step_ms + duration_ms,
                pool[position % len(pool)],
            )
        )
    return cues


def apply_affine(cues: list[Cue], scale: float, offset_ms: float) -> list[Cue]:
    """Generate a transformed copy (fixture generation only)."""
    converted: list[Cue] = []
    for cue in cues:
        converted.append(
            make_cue(
                cue.number,
                int(round(scale * cue.start_ms + offset_ms)),
                int(round(scale * cue.end_ms + offset_ms)),
                cue.text,
            )
        )
    return converted


def cues_to_srt_text(cues: list[Cue]) -> str:
    return format_srt(cues)
