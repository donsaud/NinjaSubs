"""False-positive benchmark for the synchronization classifier.

The objective is explicitly *not* to maximize how many subtitles are called
verified. It is to minimize how many INCORRECT ones are. This module builds a
labelled dataset of timing scenarios, runs the real classifier over it, and
reports the four numbers that matter:

    false_verified_count   - the metric to drive to zero
    false_rejected_count   - usable subtitles discarded
    true_verified_count    - correctly identified
    unverified_count       - honestly unclaimed

Thresholds are never tuned by eye; a change is only justified when it moves
these counts, and the assertions below encode that trade-off.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.services.sync.alignment import (
    AlignmentAnalyzer,
    SyncState,
)


def ts(ms: int) -> str:
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def build_srt(
    positions: list[float],
    *,
    duration_ms: int = 1_500,
    body: str = "I never thought I would find someone like you in my life tonight",
) -> str:
    return "\n\n".join(
        f"{i + 1}\n{ts(int(p))} --> {ts(int(p) + duration_ms)}\n{body}"
        for i, p in enumerate(positions)
    )


def even(count: int, step_ms: float = 2_000, start_ms: float = 0.0) -> list[float]:
    return [start_ms + i * step_ms for i in range(count)]


@dataclass
class Scenario:
    """One labelled case. ``expect_verified`` is the ground truth."""

    name: str
    target: str
    reference: str | None
    synced: str | None = None
    alass_applied: bool = False
    alass_successful: bool = False
    # Ground truth: could this subtitle legitimately be served as synchronized?
    expect_verified: bool = True
    expect_rejected: bool = False
    note: str = ""


@dataclass
class BenchmarkReport:
    total: int = 0
    false_verified: list[str] = field(default_factory=list)
    false_rejected: list[str] = field(default_factory=list)
    true_verified: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)
    details: dict[str, str] = field(default_factory=dict)

    @property
    def false_verified_count(self) -> int:
        return len(self.false_verified)

    @property
    def false_rejected_count(self) -> int:
        return len(self.false_rejected)

    @property
    def true_verified_count(self) -> int:
        return len(self.true_verified)

    @property
    def unverified_count(self) -> int:
        return len(self.unverified)

    @property
    def precision(self) -> float:
        claimed = self.true_verified_count + self.false_verified_count
        return self.true_verified_count / claimed if claimed else 1.0

    def summary(self) -> str:
        return (
            f"total={self.total} true_verified={self.true_verified_count} "
            f"false_verified={self.false_verified_count} "
            f"false_rejected={self.false_rejected_count} "
            f"unverified={self.unverified_count} precision={self.precision:.3f}"
        )


def build_dataset() -> list[Scenario]:
    """Representative timing scenarios, including the known-hard cases."""
    base = even(40)
    offset = even(40, start_ms=2_310)
    far = even(40, start_ms=97_000)
    drifting = [p + 1_200 * i for i, p in enumerate(even(40))]
    piecewise = [p + (12_000 if i >= 20 else 0.0) for i, p in enumerate(even(40))]
    sparse = [p for p in even(6)]

    return [
        # --- genuinely synchronized -----------------------------------------
        Scenario(
            "exact_already_aligned",
            build_srt(base),
            build_srt(base),
            expect_verified=True,
            note="same cut, same timings",
        ),
        Scenario(
            "constant_offset_corrected",
            build_srt(base),
            build_srt(offset),
            synced=build_srt(offset),
            alass_applied=True,
            alass_successful=True,
            expect_verified=True,
            note="stable 2.31s intro difference, re-timed",
        ),
        Scenario(
            "large_stable_offset_corrected",
            build_srt(base),
            build_srt(even(40, start_ms=14_000)),
            synced=build_srt(even(40, start_ms=14_000)),
            alass_applied=True,
            alass_successful=True,
            expect_verified=True,
            note="14s stable offset is unusual but re-timable",
        ),
        Scenario(
            "piecewise_recap_offset_corrected",
            build_srt(base),
            build_srt(piecewise),
            synced=build_srt(piecewise),
            alass_applied=True,
            alass_successful=True,
            expect_verified=False,
            note="recap shift mid-file: not a clean global re-time",
        ),
        # --- must never be claimed verified --------------------------------
        Scenario(
            "wrong_cut_97s",
            build_srt(base),
            build_srt(far),
            expect_verified=False,
            expect_rejected=True,
            note="different cut",
        ),
        Scenario(
            "wrong_episode_shape",
            build_srt(base),
            build_srt(even(20, step_ms=4_000, start_ms=97_000)),
            expect_verified=False,
            expect_rejected=True,
            note="wrong episode, different structure",
        ),
        Scenario(
            "alass_success_but_scrambled",
            build_srt(base),
            build_srt(offset),
            synced=build_srt([0, 41_000, 3_000, 58_000, 7_000, 22_000] * 7),
            alass_applied=True,
            alass_successful=True,
            expect_verified=False,
            note="exit 0, unusable output",
        ),
        Scenario(
            "alass_success_but_cue_loss",
            build_srt(base),
            build_srt(offset),
            synced=build_srt(offset[:5]),
            alass_applied=True,
            alass_successful=True,
            expect_verified=False,
            expect_rejected=True,
            note="exit 0, dropped 35 of 40 cues",
        ),
        Scenario(
            "progressive_drift_uncorrected",
            build_srt(base),
            build_srt(drifting),
            synced=build_srt(drifting),
            alass_applied=True,
            alass_successful=True,
            expect_verified=False,
            note="drift beyond tolerance is not a global shift",
        ),
        Scenario(
            "wrong_cut_same_shape_97s",
            build_srt(base),
            build_srt(far),
            expect_verified=False,
            expect_rejected=True,
            note="same shape, 97s out: unverifiable, must not be claimed",
        ),
        # --- honestly unverified -------------------------------------------
        Scenario(
            "no_reference",
            build_srt(base),
            None,
            expect_verified=False,
            note="nothing to compare against",
        ),
        Scenario(
            "sparse_dialogue",
            build_srt(sparse),
            build_srt(sparse),
            expect_verified=False,
            note="too few cues to measure",
        ),
        Scenario(
            "alass_failure",
            build_srt(base),
            build_srt(offset),
            alass_applied=True,
            alass_successful=False,
            expect_verified=False,
            note="no output produced",
        ),
    ]


def run_benchmark(
    scenarios: list[Scenario] | None = None,
    *,
    analyzer: AlignmentAnalyzer | None = None,
) -> BenchmarkReport:
    """Classify every scenario and bucket the outcome against ground truth."""
    analyzer = analyzer or AlignmentAnalyzer()
    report = BenchmarkReport(total=len(scenarios or build_dataset()))

    for scenario in scenarios or build_dataset():
        evaluation = analyzer.analyze(
            scenario.target,
            scenario.synced,
            scenario.reference,
            alass_applied=scenario.alass_applied,
            alass_successful=scenario.alass_successful,
        )
        state = evaluation.sync_state
        report.details[scenario.name] = f"{state.value} ({evaluation.explain()})"

        claimed_verified = state in (SyncState.VERIFIED_SYNCED, SyncState.VERIFIED_RESYNCED)
        rejected = state is SyncState.REJECTED

        if claimed_verified and not scenario.expect_verified:
            report.false_verified.append(scenario.name)
        elif claimed_verified:
            report.true_verified.append(scenario.name)
        elif rejected and not scenario.expect_rejected:
            report.false_rejected.append(scenario.name)
        else:
            report.unverified.append(scenario.name)

    return report


def assert_no_known_false_positives(report: BenchmarkReport) -> None:
    """The hard gate: nothing known-wrong may be claimed verified."""
    assert not report.false_verified, (
        f"false verified: {report.false_verified}\n" + "\n".join(
            f"  {name}: {report.details.get(name)}" for name in report.false_verified
        )
    )
