"""Parser semantic fingerprints; snapshots and REAL evidence are read-only."""
from __future__ import annotations

import ast
import json

import pytest

from app.services.sync import parser_fingerprint as fp
from app.services.sync import parser_projection as pp
from tools import capture_v2 as v2
from tools.sync_replay import MANIFEST, ROOT

LEGACY = "afa56b70291934e1f90549e138f12cd677c1941c19c5d8c125e7a1479c96291f"


def changed_source(module, symbol, *, remove=False):
    """Change one AST binding in memory, never a repository/evidence file."""
    def reader(name):
        source = fp._read_source(name)
        if name != module:
            return source
        tree = ast.parse(source)
        found = False
        for index, node in enumerate(tree.body):
            if isinstance(node, ast.FunctionDef) and node.name == symbol:
                found = True
                if remove:
                    del tree.body[index]
                else:
                    node.body = [ast.Return(value=ast.Constant(value=None))]
                break
            if isinstance(node, ast.Assign | ast.AnnAssign):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(t, ast.Name) and t.id == symbol for t in targets):
                    found = True
                    node.value = ast.Constant(value=123456)
                    break
        assert found, symbol
        return ast.unparse(ast.fix_missing_locations(tree))
    return reader


@pytest.mark.parametrize("symbol", ["rank_subtitles", "deduplicate_subtitles"])
def test_unrelated_ranking_changes_preserve_current_and_legacy_digest(symbol):
    digest = fp.semantic_sha(changed_source("app.services.subtitle_matcher", symbol))
    assert digest == pp.implementation_sha()
    assert fp.matches_fingerprint(digest, fp.FINGERPRINT_SCHEMA, digest)
    assert fp.matches_fingerprint(LEGACY, 1, digest)


@pytest.mark.parametrize("module,symbol", [
    ("app.services.subtitle_matcher", "parse_srt_cues"),
    ("app.services.subtitle_matcher", "_strip_markup_for_speech"),
    ("app.services.subtitle_matcher", "_NON_SPEECH_PATTERNS"),
    ("app.services.subtitle_matcher", "_DIALOGUE_MIN_LETTERS"),
    ("app.services.sync.ninja_points", "_is_metadata_cue"),
    ("app.services.sync.ninja_points", "_METADATA_MAX_CHARS"),
    ("app.services.sync.ninja_points", "render_srt"),
    ("app.extractor", "decode_subtitle_bytes"),
    ("app.extractor", "_UTF8_REPAIR_TOLERANCE"),
    ("app.services.sync.parser_projection", "project"),
    ("app.services.sync.parser_projection", "render_projection"),
    ("app.services.sync.parser_projection", "VERSION"),
    ("app.services.sync.boundary_verifier", "_ordered"),
])
def test_actual_parser_dependency_changes_reject_new_and_legacy(module, symbol):
    old = pp.implementation_sha()
    changed = fp.semantic_sha(changed_source(module, symbol))
    assert changed != old
    assert not fp.matches_fingerprint(old, fp.FINGERPRINT_SCHEMA, changed)
    assert not fp.matches_fingerprint(LEGACY, 1, changed)


def test_missing_parser_dependency_fails_closed():
    with pytest.raises(ValueError, match="missing parser dependency"):
        fp.semantic_sha(changed_source("app.services.subtitle_matcher", "parse_srt_cues", remove=True))


@pytest.mark.parametrize("stored,schema", [("0" * 64, 1), ("0" * 64, 2),
    (LEGACY, 3), (LEGACY, True), (LEGACY, "1"), (LEGACY, None), (None, 1)])
def test_unknown_or_mismatched_fingerprints_refused(stored, schema):
    assert not fp.matches_fingerprint(stored, schema, pp.implementation_sha())


def test_comments_whitespace_and_line_endings_do_not_change_digest():
    def reader(module):
        return "# unrelated comment\n\n" + fp._read_source(module).replace("\n", "\r\n")
    assert fp.semantic_sha(reader) == pp.implementation_sha()


def test_existing_real_captures_replay_without_rewriting_any_evidence(monkeypatch):
    from tests.private_fixtures import require_many

    cases = [c for c in json.loads(MANIFEST.read_bytes())["cases"] if "capture_v2" in c]
    assert len(cases) == 2
    require_many([ROOT / c["capture_v2"]["path"] / "pristine_target.srt" for c in cases])
    before = {p: p.read_bytes() for p in (ROOT / "tests/replay").rglob("*") if p.is_file()}
    for case in cases:
        spec = case["capture_v2"]
        folder = ROOT / spec["path"]
        projection = json.loads((folder / spec["attempt"] / "boundary_parser_projection.json").read_bytes())
        assert "parser_fingerprint_schema" not in projection
        assert projection["parser_implementation_sha256"] == LEGACY
        result = v2.replay_v2(case, ROOT)
        assert result["evidence"]["shadow"]["accepted"] == case["expected_shadow_outcome"]
    assert before == {p: p.read_bytes() for p in before}
    changed = fp.semantic_sha(changed_source("app.services.subtitle_matcher", "parse_srt_cues"))
    monkeypatch.setattr(v2, "implementation_sha", lambda: changed)
    for case in cases:
        with pytest.raises(ValueError, match="parser drift"):
            v2.check_capture(ROOT / case["capture_v2"]["path"], case["capture_v2"]["attempt"])
    assert before == {p: p.read_bytes() for p in before}


def test_python_311_and_312_ast_shapes_have_the_same_digest(monkeypatch):
    expected = pp.implementation_sha()
    original_parse = ast.parse

    def python_311_shape(source):
        tree = original_parse(source)
        for node in ast.walk(tree):
            if getattr(node, "type_params", None) == []:
                delattr(node, "type_params")
        return tree

    monkeypatch.setattr(ast, "parse", python_311_shape)
    assert pp.implementation_sha() == expected
    assert fp.matches_fingerprint(LEGACY, 1, expected)


def test_nonempty_generic_parameters_are_not_ignored():
    # Add the newer AST field directly, so this test also runs on Python 3.11.
    normal = ast.parse("def parser(): return 1").body[0]
    generic = ast.parse("def parser(): return 1").body[0]
    generic._fields = (*generic._fields, "type_params") if "type_params" not in generic._fields else generic._fields
    generic.type_params = [ast.Name(id="T", ctx=ast.Load())]
    assert fp._normalized_ast(normal) != fp._normalized_ast(generic)
