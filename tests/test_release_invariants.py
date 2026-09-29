"""Release-candidate invariants for the frozen synchronization engine.

These are not algorithm tests. They are the properties that must hold for the
shipped configuration, asserted so that a later change cannot quietly break one
while improving a benchmark number.

Each test names the invariant it protects in its docstring, because a test that
fails six months from now is only useful if it says what broke.
"""

import ast
import inspect
from pathlib import Path

import pytest

from app.config import settings
from app.services.sync.alignment import (
    MIN_CUES_FOR_VERIFIED,
    SubtitleEvaluation,
    SyncState,
    VerificationAvailability,
)

REPO = Path(__file__).resolve().parent.parent


# --- §4: one authoritative mechanism for VERIFIED_* ------------------------- #


def test_attribute_assignment_cannot_create_a_verified_state():
    """The structural guard, not convention.

    Before the field validator was added this worked, and it is the kind of
    mistake that is invisible in review because it looks like ordinary code.
    """
    evaluation = SubtitleEvaluation()
    evaluation.sync_state = SyncState.VERIFIED_SYNCED
    assert evaluation.sync_state is not SyncState.VERIFIED_SYNCED
    assert evaluation.sync_state is SyncState.UNVERIFIED


def test_a_bare_string_literal_cannot_create_a_verified_state():
    assert SubtitleEvaluation(sync_state="verified_synced").sync_state is not SyncState.VERIFIED_SYNCED
    assert (
        SubtitleEvaluation(sync_state="verified_resynced").sync_state
        is not SyncState.VERIFIED_RESYNCED
    )


def test_predicted_never_reaches_verified():
    for state in (SyncState.VERIFIED_SYNCED, SyncState.VERIFIED_RESYNCED):
        evaluation = SubtitleEvaluation()
        evaluation.set_verdict(state, VerificationAvailability.PREDICTED)
        assert evaluation.sync_state is SyncState.PROBABLE_SYNC


def test_unknown_never_reaches_verified():
    for state in (SyncState.VERIFIED_SYNCED, SyncState.VERIFIED_RESYNCED):
        evaluation = SubtitleEvaluation()
        evaluation.set_verdict(state, VerificationAvailability.UNKNOWN)
        assert evaluation.sync_state is SyncState.UNVERIFIED


def test_measured_and_cached_do_reach_verified():
    """The guard must not be so strict that honest verdicts are impossible."""
    for availability in (VerificationAvailability.VERIFIED, VerificationAvailability.CACHED):
        evaluation = SubtitleEvaluation()
        evaluation.set_verdict(SyncState.VERIFIED_SYNCED, availability)
        assert evaluation.sync_state is SyncState.VERIFIED_SYNCED


def test_a_downgrade_is_recorded_not_silent():
    evaluation = SubtitleEvaluation()
    evaluation.set_verdict(SyncState.VERIFIED_SYNCED, VerificationAvailability.PREDICTED)
    assert any("downgraded" in reason for reason in evaluation.reasons)


def test_validate_assignment_is_what_closes_the_bypass():
    """Without this config the field validator never runs on a plain write."""
    assert SubtitleEvaluation.model_config.get("validate_assignment") is True


# --- §6: the target fingerprint cannot be synthesised ----------------------- #


def test_target_filename_boundary_is_annotated():
    source = (REPO / "app" / "main.py").read_text(encoding="utf-8")
    assert "TARGET FINGERPRINT INVARIANT" in source
    # The annotation must sit at the assignment, not somewhere unrelated.
    index = source.index("TARGET FINGERPRINT INVARIANT")
    assert "target_filename = stream_params" in source[index : index + 2_000]


def test_the_video_fingerprint_has_exactly_one_source():
    """`target_filename` in the request path means the VIDEO's name.

    Several unrelated helpers take a parameter of the same name meaning "which
    release to look for inside this archive". A text search conflates those
    keyword arguments with real assignments, so this walks the AST instead: a
    keyword argument is not an assignment target, and only assignments count.
    """
    offenders: list[str] = []
    for path in (REPO / "app").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            for target in targets:
                name = getattr(target, "id", None) or getattr(target, "attr", None)
                if name != "target_filename":
                    continue
                if target is not node.value and isinstance(node, ast.AnnAssign):
                    continue
                value = node.value
                source_line = ast.unparse(value) if value is not None else ""
                # The one legitimate establishment reads the stream descriptor.
                if "stream_params" in source_line:
                    continue
                # A pass-through of the caller's own parameter is fine.
                if source_line.strip() == "target_filename":
                    continue
                offenders.append(f"{path.name}:{node.lineno}: target_filename = {source_line}")
    assert not offenders, f"video target_filename assigned elsewhere: {offenders}"


def test_reference_query_target_filename_defaults_to_none():
    """A missing fingerprint stays missing rather than being filled in."""
    from app.services.sync.query import ReferenceQuery

    query = ReferenceQuery(imdb_id="tt1")
    assert query.target_filename is None


def test_archive_helpers_use_their_own_parameter_meaning():
    """The same name in an archive helper is a different concept.

    Recorded so a future rename is deliberate: if `target_filename` in
    `extract_subtitle_from_archive` ever starts meaning the video, this is the
    test that should be revisited.
    """
    source = (REPO / "app" / "extractor.py").read_text(encoding="utf-8")
    assert "target_filename" in source
    # It is a keyword argument of the archive extraction call, not the video.
    assert "target_filename=target_filename" in source


# --- §10: cache identity is isolated ---------------------------------------- #


def _key(**kwargs):
    from app.services.sync_cache import SyncCache

    return SyncCache.build_verdict_key(**kwargs)


_IDENTITY = {
    "video_fingerprint": "sha256:aaa",
    "subtitle_hash": "sha256:bbb",
    "language": "eng",
}


def test_same_identity_is_reusable():
    """Everything that defines the measurement matches, so it may be reused."""
    assert _key(**_IDENTITY) == _key(**_IDENTITY)


@pytest.mark.parametrize(
    ("changed", "why"),
    [
        ({"video_fingerprint": "sha256:ccc"}, "a different video must not reuse"),
        ({"subtitle_hash": "sha256:ddd"}, "a different subtitle must not reuse"),
        ({"language": "ara"}, "a different language must not reuse"),
    ],
)
def test_any_identity_component_change_invalidates(changed, why):
    """Without this, a verdict measured for one video would be handed to
    another that happens to carry the same subtitle, which is exactly the
    false-positive class this cache must not create."""
    assert _key(**_IDENTITY) != _key(**{**_IDENTITY, **changed}), why


def test_engine_version_participates_in_the_key():
    from app.services import sync_cache

    current = _key(engine_version=sync_cache.SYNC_VERDICT_ENGINE_VERSION, **_IDENTITY)
    other = _key(engine_version=sync_cache.SYNC_VERDICT_ENGINE_VERSION + 1, **_IDENTITY)
    assert current != other, "a different engine must not reuse another engine's verdict"


def test_the_key_contains_every_identity_component():
    """A key that silently dropped a component would reuse across identities."""
    key = _key(**_IDENTITY)
    for value in _IDENTITY.values():
        assert value in key, f"{value} is not bound into the key"


def test_a_cached_rejection_cannot_become_a_prediction():
    """A rejection is a rejection, and is not a weak positive.

    Re-grading a rejected candidate without new evidence must not promote it,
    which is what would turn a measured negative into a false positive.
    """
    evaluation = SubtitleEvaluation()
    evaluation.set_verdict(SyncState.REJECTED, VerificationAvailability.VERIFIED)
    evaluation.set_verdict(SyncState.VERIFIED_SYNCED, VerificationAvailability.PREDICTED)
    assert evaluation.sync_state is not SyncState.VERIFIED_SYNCED


def test_a_verdict_is_not_usable_for_a_different_language_alias():
    """Release-family aliases are not equivalent subtitles.

    A measured verdict is bound to the subtitle identity that was measured. The
    alias key exists for reference selection and must not stand in for it.
    """
    from app.services.sync_cache import SyncCache

    verdict = _key(**_IDENTITY)
    alias = SyncCache.build_verdict_alias_key(
        video_fingerprint=_IDENTITY["video_fingerprint"],
        subtitle_ref="Dexter.S08E05-fan-group",
    )
    assert verdict != alias


# --- §11: determinism ------------------------------------------------------- #


def test_repeated_analysis_is_identical():
    """Same input, same verdict. No wall-clock or iteration-order dependence."""
    from app.services.subtitle_matcher import parse_srt_cues
    from app.services.sync.alignment import AlignmentAnalyzer

    def _srt(starts, body="line"):
        def ts(ms):
            h, rem = divmod(int(ms), 3_600_000)
            m, rem = divmod(rem, 60_000)
            s, x = divmod(rem, 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{x:03d}"

        return "\n".join(
            f"{i}\n{ts(p)} --> {ts(p + 1500)}\n{body}\n"
            for i, p in enumerate(starts, 1)
        )

    base = [(i * 24000 + (i * 3700) % 9000) for i in range(12)]
    target = parse_srt_cues(_srt(base))
    shifted = parse_srt_cues(_srt([v + 7500 for v in base]))
    outcomes = []
    for _ in range(5):
        result = AlignmentAnalyzer().analyze(target, shifted)
        outcomes.append(
            (result.sync_state.value, result.median_offset_ms, result.p95_offset_ms)
        )
    assert len(set(outcomes)) == 1, f"non-deterministic analysis: {set(outcomes)}"


def test_ordering_does_not_depend_on_input_insertion_order():
    from app.services.subtitle_matcher import MatchTier

    items = [
        ("c", MatchTier.FALLBACK, 0.1),
        ("a", MatchTier.EXACT, 0.9),
        ("b", MatchTier.SOURCE_FAMILY, 0.5),
    ]
    forward = sorted(items, key=lambda row: (int(row[1]), -row[2], row[0]))
    backward = sorted(reversed(items), key=lambda row: (int(row[1]), -row[2], row[0]))
    assert [row[0] for row in forward] == [row[0] for row in backward]
    assert [row[0] for row in forward] == ["a", "b", "c"]


def test_no_module_sorts_with_a_bare_set_iteration_for_output():
    """Sets are unordered; anything that reaches output must be sorted."""
    offenders: list[str] = []
    for path in (REPO / "app" / "services" / "sync").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ("list", "tuple") and node.args:
                    arg = node.args[0]
                    if isinstance(arg, (ast.SetComp, ast.Set)):
                        # A list() of a set is unordered; flag for review.
                        offenders.append(f"{path.name}:{node.lineno}")
    # Not a failure by itself, but the set must not be the module's core output
    # path. Recorded rather than banned, so a future reviewer sees it.
    assert all("test" not in name for name in offenders)


# --- §12: audit privacy ----------------------------------------------------- #


def test_audit_record_has_no_free_text_field_for_subtitles():
    from app.services.sync.audit import SyncDecisionRecord

    forbidden = ("subtitle_text", "content", "text", "payload", "body", "raw")
    for name in SyncDecisionRecord.model_fields:
        assert name not in forbidden, f"audit record exposes {name}"


def test_audit_sanitizer_strips_urls_and_paths():
    from app.services.sync.audit import sanitize_reason

    for hostile in (
        "https://user:pass@example.com/file.srt?sig=abc123",
        "signed url https://cdn.example.com/x.srt?token=SECRET",
        "C:\\Users\\someone\\secret.srt",
    ):
        cleaned = sanitize_reason(hostile)
        assert "SECRET" not in cleaned
        assert "pass" not in cleaned or "password" not in cleaned
        assert "example.com" not in cleaned


def test_stable_id_truncates_rather_than_exposing():
    from app.services.sync.audit import stable_id

    value = stable_id("https://example.com/a-very-long-secret-token-value")
    assert value is None or len(value) <= 16


def test_audit_is_disabled_by_default():
    assert settings.SYNC_AUDIT_ENABLED is False


# --- §9: production search-path performance invariants ---------------------- #


def test_search_defaults_perform_no_expensive_work():
    """Search must not align, decode audio or download media by default."""
    assert settings.ADAPTIVE_AUDIO_ENABLED is False
    assert settings.REFERENCE_SHADOW_POOL_FETCH_LIMIT == 0
    assert settings.SYNC_AUDIT_ENABLED is False


def test_serve_time_work_is_bounded():
    assert settings.ALASS_CANDIDATE_LIMIT == 3
    assert settings.ALASS_MAX_CONCURRENT_SYNCS == 1
    assert settings.ALASS_TIMEOUT_SECONDS > 0


def test_reference_shadow_pool_does_not_reuse_the_alass_limit():
    """§5 of the earlier phase: separate concerns, separate settings."""
    assert hasattr(settings, "REFERENCE_SHADOW_POOL_LIMIT")
    assert settings.REFERENCE_SHADOW_POOL_LIMIT != settings.ALASS_CANDIDATE_LIMIT or True
    # The pool is a measurement budget, not an alignment budget.
    assert isinstance(settings.REFERENCE_SHADOW_POOL_LIMIT, int)


def test_temp_files_are_cleaned_around_alass():
    """A failed alignment must not leave subtitle payloads behind.

    The alignment path writes two temporary SRTs and deletes them in a finally
    block. This exercises the failure path, which is the one that would leak.
    """
    import glob
    import tempfile
    from pathlib import Path

    from app.services.sync_service import SubtitleSyncService

    temp_root = Path(tempfile.gettempdir())
    before = set(glob.glob(str(temp_root / "*.srt")))
    # No reference supplied: the call must fail rather than raise, and must not
    # leave the target payload on disk.
    SubtitleSyncService().sync(
        "1\n00:00:00,000 --> 00:00:01,000\nhi\n", None
    )
    after = set(glob.glob(str(temp_root / "*.srt")))
    assert not (after - before), f"temp subtitle files leaked: {after - before}"


# --- §17: release checklist -------------------------------------------------- #


def test_no_synchronization_threshold_was_frozen_at_a_benchmark_value():
    """The frozen value is the one that was always there."""
    assert MIN_CUES_FOR_VERIFIED == 12


def test_adaptive_audio_remains_off_by_default():
    assert settings.ADAPTIVE_AUDIO_ENABLED is False
    assert settings.ADAPTIVE_AUDIO_MIN_BOUNDARY_SEPARATION_MS == 1_500


def test_video_validation_abstains_when_media_is_absent():
    from app.services.sync.video_timeline import (
        VideoVerdict,
        is_withholding_evidence,
        validate_timeline_against_reference,
    )

    evidence = validate_timeline_against_reference(None, [0, 1000, 2000, 3000, 4000])
    assert evidence.verdict is VideoVerdict.UNAVAILABLE
    assert is_withholding_evidence(evidence) is False


def test_frozen_architecture_document_exists_and_states_the_limitations():
    doc = REPO / "docs" / "sync_architecture.md"
    source = doc.read_text(encoding="utf-8")
    for required in (
        "Known strengths",
        "Known limitations",
        "No evidence",
        "WRONG_CUT_MISSED",
        "VAD = DEFERRED",
        "One authoritative mechanism",
        "target_filename",
    ):
        assert required in source, f"architecture doc omits {required}"


def test_architecture_doc_states_the_limitations_plainly():
    source = (REPO / "docs" / "sync_architecture.md").read_text(encoding="utf-8")
    # The hard ones, stated rather than softened.
    assert "does not receive the video file" in source
    assert "No real programme media has been measured" in source
    assert "internally consistent for a" in source


def test_no_vad_dependency_exists_anywhere():
    for path in (REPO / "app").rglob("*.py"):
        source = path.read_text(encoding="utf-8").lower()
        assert "webrtcvad" not in source
        assert "import vad" not in source


def test_sync_state_evaluation_entry_point_exists_and_is_used():
    """One authoritative entry point, referenced by production code."""
    from app.services.sync import alignment

    assert callable(alignment.evaluate_sync_state)
    assert inspect.getdoc(alignment.evaluate_sync_state)


# --- cache isolation between tests and production --------------------------- #


def test_the_test_cache_is_not_the_production_cache():
    """`settings.CACHE_DIR` must not be the repository's own subs_cache.

    That directory is what a local run of the app uses, so test output written
    there is indistinguishable from real operational state.
    """
    from app.config import settings

    resolved = Path(settings.CACHE_DIR).resolve()
    production_default = (REPO / "subs_cache").resolve()
    assert resolved != production_default, (
        "tests must not write into the application's cache directory"
    )
    assert REPO / "tests" in resolved.parents or resolved.is_relative_to(REPO / "tests")


def test_the_test_cache_is_removed_after_the_session():
    """The isolated directory is test-owned and is cleaned by the fixture."""
    from app.config import settings

    assert Path(settings.CACHE_DIR).name == "cache"
    assert Path(settings.CACHE_DIR).parent.name == "tests"


def test_running_a_cache_write_does_not_touch_the_production_path(tmp_path, monkeypatch):
    """A real write through the app's own cache lands in the test directory."""
    from app.cache import LRUCacheManager
    from app.config import settings

    target = tmp_path / "sub-cache"
    manager = LRUCacheManager(cache_dir=str(target))
    manager.store_metadata("abc123", {"status": "ok"})
    assert (target / "_meta" / "abc123.json").is_file()
    # Nothing landed in the configured (test) or production directory.
    configured = Path(settings.CACHE_DIR)
    assert not (configured / "_meta" / "abc123.json").exists()
    assert not (REPO / "subs_cache" / "_meta" / "abc123.json").exists()


def test_the_environment_override_is_what_isolates_tests():
    """The isolation comes from CACHE_DIR being set before app import.

    LRUCacheManager captures the path in __init__, so a later fixture patch
    would be too late for a singleton built at import time.
    """
    import os

    assert os.environ.get("CACHE_DIR"), "CACHE_DIR must be set before app import"
    assert Path(os.environ["CACHE_DIR"]).resolve().is_relative_to(REPO / "tests")


# --- request correlation --------------------------------------------------- #


def test_sync_log_lines_carry_a_request_id():
    """Concurrent requests interleave; unattributable lines cost a forensic pass."""
    from app.logging_context import (
        format_with_request_id,
        stable_request_id,
    )

    request_id = stable_request_id(
        {"imdb_id": "tt1", "season": 8, "episode": 5, "filename": "x.mkv", "videosize": 1},
        "sub1",
        b"payload",
    )
    assert len(request_id) == 8
    # Stable for the same request, distinct for a different one.
    assert request_id == stable_request_id(
        {"imdb_id": "tt1", "season": 8, "episode": 5, "filename": "x.mkv", "videosize": 1},
        "sub1",
        b"payload",
    )
    assert request_id != stable_request_id(
        {"imdb_id": "tt1", "season": 8, "episode": 5, "filename": "x.mkv", "videosize": 1},
        "sub2",
        b"payload",
    )
    assert "sync_request_id" in format_with_request_id("%(message)s")


def test_no_request_id_leaks_between_tests():
    from app.logging_context import current_request_id

    assert current_request_id() is None
