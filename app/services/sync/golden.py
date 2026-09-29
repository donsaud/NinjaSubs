"""Ground-truth schema for the golden synchronization dataset.

This module defines what "correct" means. It exists to be read by an evaluator
and by annotators, and it is deliberately inert:

* It imports nothing from the system under evaluation. No analyzer, no
  predictor, no trust model, no Alass wrapper. A label can therefore never be
  derived from the thing being measured.
* It computes no labels. Ground truth is authored in the manifest and loaded
  verbatim. ``load_manifest`` validates shape; it never derives a field the
  annotator did not write.

If a future change makes this module import the analyzer or the predictor, the
property tests fail. That is the point of the constraint.

The dimensions are intentionally separate. "Same content" is not "same
release", "same release" is not "same cut", "not synchronized" is not "cannot
be resynchronized", and a successful resync is not a correct one.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

DATASET_VERSION = "v1"
SCHEMA_VERSION = 1

#: Bumped whenever the analyzer or a verification threshold changes in a way
#: that would alter a benchmark result. Results from different engine versions
#: must not be silently combined.
GOLDEN_ENGINE_VERSION = 1


class GroundTruthContent(str, Enum):
    """Does the subtitle belong to the same work?"""

    SAME_RELEASE = "same_release"
    SAME_CONTENT_DIFFERENT_RELEASE = "same_content_different_release"
    DIFFERENT_CONTENT = "different_content"
    UNKNOWN = "unknown"


class GroundTruthCut(str, Enum):
    """Is it the same edit?"""

    SAME_CUT = "same_cut"
    DIFFERENT_CUT = "different_cut"
    UNKNOWN = "unknown"


class GroundTruthSync(str, Enum):
    """Was the subtitle already synchronized to its own source?"""

    ALREADY_SYNCED = "already_synced"
    RESYNCABLE = "resyncable"
    NOT_RESYNCABLE = "not_resyncable"
    UNKNOWN = "unknown"


class GroundTruthFinal(str, Enum):
    """After synchronization, is the resulting timing correct?"""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNKNOWN = "unknown"


class GroundTruthSource(str, Enum):
    """How the label was established."""

    #: Exact identity, or a known-good timing relationship derived from
    #: something other than the pipeline under test.
    OBJECTIVE = "objective"
    #: A person inspected the synchronization against the actual video.
    HUMAN_REVIEWED = "human_reviewed"
    UNKNOWN = "unknown"


class GoldenCaseError(ValueError):
    """Raised when a manifest is malformed. Never degrades silently."""


class AlignmentSimulation(str, Enum):
    """The alignment outcome the harness feeds the verifier.

    The analyzer's contract is to judge a subtitle that has already been
    re-timed. The harness therefore must supply a plausible alignment result,
    and it must supply one that is *declared* rather than improvised, or the
    benchmark would be measuring the harness's own imagination.

    These are not measurements of alass. They are the alignment outcomes the
    annotation says are possible for this case, and they exist so the
    verification layer can be scored against known-correct and known-incorrect
    alignments separately from the aligner.
    """

    #: A correct aligner landed on the true timing. Small declared jitter only.
    AS_TARGET = "as_target"
    #: A correct aligner landed on the true timing, but drift remains.
    AS_TARGET_WITH_DRIFT = "as_target_with_drift"
    #: Exit 0, plausible residuals, but part of the timeline is the wrong cut.
    PARTIAL_MISALIGN = "partial_misalign"
    #: Alass failed or produced nothing usable.
    NO_ALIGNMENT = "no_alignment"


class AlignmentExpectation(BaseModel):
    """A declared, auditable alignment outcome used as verifier input.

    Recorded in the manifest so a reader can see exactly what the verifier was
    shown, rather than trusting the harness.
    """

    mode: AlignmentSimulation = AlignmentSimulation.AS_TARGET
    #: Peak jitter added to an otherwise correct alignment, in ms.
    residual_ms: int = 0
    #: Per-minute drift left after a successful alignment.
    drift_per_minute_ms: float = 0.0
    #: For PARTIAL_MISALIGN: cue index where the wrong segment begins, and how
    #: far it is displaced. This is the shape alass produces on a different cut.
    displaced_from_index: int | None = None
    displaced_by_ms: int = 0
    #: Cues retained by the aligner, as a fraction of the candidate's cues.
    coverage: float = 1.0
    annotation_note: str | None = None


class CandidateTruth(BaseModel):
    """Independently established truth about one candidate subtitle."""

    provider: str | None = None
    release_name: str | None = None
    content: GroundTruthContent = GroundTruthContent.UNKNOWN
    cut: GroundTruthCut = GroundTruthCut.UNKNOWN
    original_sync: GroundTruthSync = GroundTruthSync.UNKNOWN
    #: Whether the target can be reached by re-aligning this subtitle.
    resyncable: bool | None = None
    final_alignment: GroundTruthFinal = GroundTruthFinal.UNKNOWN
    #: The alignment outcome handed to the verifier for this candidate.
    alignment: AlignmentExpectation = Field(default_factory=AlignmentExpectation)
    annotation_note: str | None = None

    @property
    def label(self) -> tuple[str, str, str, str, str]:
        """A hashable label tuple, for counting and grouping."""
        return (
            self.content.value,
            self.cut.value,
            self.original_sync.value,
            str(self.resyncable).lower(),
            self.final_alignment.value,
        )

    @property
    def is_known(self) -> bool:
        """False when any dimension is unknown, so it can be excluded."""
        return GroundTruthContent.UNKNOWN not in (self.content,) and (
            self.final_alignment is not GroundTruthFinal.UNKNOWN
        )


class CandidateFixture(BaseModel):
    """Where a candidate subtitle lives and what the system will be shown."""

    provider: str | None = None
    release_name: str | None = None
    path: str
    sha256: str | None = None
    language: str | None = None


class TargetFixture(BaseModel):
    """The video's true speech timing, expressed as a reference subtitle.

    For synthetic cases this is the timing model of the imagined video: its
    cue starts are the video's actual speech onsets, which is what the analyzer
    compares against. No media file is required or committed.
    """

    path: str
    filename: str | None = None
    sha256: str | None = None
    duration_ms: int | None = None
    #: False when the request would carry incomplete Stremio metadata. The
    #: evaluator must not fabricate the missing fields.
    fingerprint_complete: bool = True


class GoldenCase(BaseModel):
    """One labeled case."""

    case_id: str
    #: Free-form classification used for slicing results, e.g. "offset".
    failure_mode: str = "unspecified"
    release_class: str = "unknown"
    annotation_source: GroundTruthSource = GroundTruthSource.UNKNOWN
    target: TargetFixture
    candidates: list[CandidateFixture] = Field(default_factory=list)
    #: One entry per candidate, matched by provider+release_name when present.
    ground_truth: list[CandidateTruth] = Field(default_factory=list)
    #: Candidate keys that are objectively valid synchronization anchors. When
    #: more than one is listed, the case is a VALID_REFERENCE_SET and no single
    #: reference is forced to be "the" correct one.
    valid_reference_keys: list[str] = Field(default_factory=list)
    notes: str | None = None

    def truth_for(self, provider: str | None, release_name: str | None) -> CandidateTruth | None:
        for truth in self.ground_truth:
            if truth.release_name == release_name and truth.provider == provider:
                return truth
        return None

    @property
    def is_valid_reference_set(self) -> bool:
        return len(self.valid_reference_keys) > 1


class GoldenManifest(BaseModel):
    """A versioned dataset."""

    dataset_version: str = DATASET_VERSION
    schema_version: int = SCHEMA_VERSION
    #: The engine version these labels were authored against. Recorded so
    #: results from different engines are never silently pooled.
    engine_version: int = GOLDEN_ENGINE_VERSION
    description: str | None = None
    #: Protocol document describing how labels were assigned.
    protocol: str | None = None
    cases: list[GoldenCase] = Field(default_factory=list)

    def case(self, case_id: str) -> GoldenCase:
        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise GoldenCaseError(f"unknown case_id: {case_id!r}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: str | Path) -> GoldenManifest:
    """Load and validate a dataset manifest.

    Fails loudly on anything malformed. A dataset that loads partially would
    quietly bias every metric derived from it.
    """
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise GoldenCaseError(f"manifest not found: {manifest_path}")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GoldenCaseError(f"manifest is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise GoldenCaseError("manifest root must be an object")
    try:
        manifest = GoldenManifest.model_validate(raw)
    except ValidationError as exc:
        raise GoldenCaseError(f"manifest failed validation: {exc}") from exc

    if not manifest.cases:
        raise GoldenCaseError("manifest contains no cases")

    seen: set[str] = set()
    for case in manifest.cases:
        if not case.case_id:
            raise GoldenCaseError("every case needs a case_id")
        if case.case_id in seen:
            raise GoldenCaseError(f"duplicate case_id: {case.case_id!r}")
        seen.add(case.case_id)
        if not case.candidates:
            raise GoldenCaseError(f"case {case.case_id!r} has no candidates")
        if not case.ground_truth:
            raise GoldenCaseError(
                f"case {case.case_id!r} has candidates but no ground truth; "
                "every candidate needs an independently authored label"
            )
        declared = {(t.provider, t.release_name) for t in case.ground_truth}
        for candidate in case.candidates:
            if (candidate.provider, candidate.release_name) not in declared:
                raise GoldenCaseError(
                    f"case {case.case_id!r} candidate "
                    f"{(candidate.provider, candidate.release_name)!r} has no ground truth"
                )
    return manifest


def resolve_fixture(manifest_path: str | Path, relative: str) -> Path:
    """Resolve a fixture path relative to the manifest."""
    return (Path(manifest_path).parent / relative).resolve()
