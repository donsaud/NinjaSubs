"""Versioned transitive parser AST fingerprints; diagnostic only."""
from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

FINGERPRINT_SCHEMA = 2
ROOTS = (
    ("app.extractor", "decode_subtitle_bytes"),
    ("app.services.sync.parser_projection", "project"),
    ("app.services.sync.parser_projection", "render_projection"),
    ("app.services.sync.parser_projection", "canonical_bytes"),
    ("app.services.sync.boundary_verifier", "_ordered"),
)
# Audited baseline 4852daa (unchanged at 1c3f53d). The legacy four-file digest
# is paired with its exact parser closure, including projection/rendering and
# metadata helpers absent from the old fingerprint. Never infer compatibility
# from a capture or from successful replay alone.
LEGACY_PARSER_DIGESTS = {
    "afa56b70291934e1f90549e138f12cd677c1941c19c5d8c125e7a1479c96291f": "9e4a8192148b51785b71a9b16a70f619b0ffceb70b530162dc2364964bab0bff",
}


def _read_source(module: str) -> str:
    root = Path(__file__).resolve().parents[3]
    return root.joinpath(*module.split(".")).with_suffix(".py").read_text(encoding="utf-8")



def _normalized_ast(node: ast.AST) -> str:
    # Python 3.12 adds empty type_params fields absent in the Docker 3.11 AST.
    # Only omit the empty field: actual generic parameters remain fingerprinted.
    for child in ast.walk(node):
        if getattr(child, "type_params", None) == []:
            delattr(child, "type_params")
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def semantic_sha(read_source: Callable[[str], str] = _read_source) -> str:
    """Hash reachable functions, constants and imports as normalized ASTs.

    Follows transitive and function-local application symbol imports. Unknown
    application-module/relative imports fail closed. The injectable reader
    permits snapshot tests without changing files or depending on Git at runtime.
    """
    modules: dict[str, dict[str, ast.AST]] = {}
    closure: dict[str, str] = {}

    def bindings(module: str) -> dict[str, ast.AST]:
        if module not in modules:
            found: dict[str, ast.AST] = {}
            for node in ast.parse(read_source(module)).body:
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                    found[node.name] = node
                elif isinstance(node, ast.Assign | ast.AnnAssign):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        for name in ast.walk(target):
                            if isinstance(name, ast.Name):
                                found[name.id] = node
                elif isinstance(node, ast.Import | ast.ImportFrom):
                    for alias in node.names:
                        found[alias.asname or alias.name.split(".")[0]] = node
            modules[module] = found
        return modules[module]

    def visit(module: str, name: str, node: ast.AST | None = None) -> None:
        key = f"{module}:{name}"
        if key in closure:
            return
        if node is None:
            node = bindings(module).get(name)
        if node is None:
            raise ValueError(f"missing parser dependency: {key}")
        closure[key] = _normalized_ast(node)
        if isinstance(node, ast.ImportFrom):
            if node.level:
                raise ValueError("relative parser import requires explicit resolution")
            for alias in node.names:
                if (alias.asname or alias.name) == name and (node.module or "").startswith("app."):
                    visit(node.module or "", alias.name)
            return
        if isinstance(node, ast.Import):
            if any(alias.name.startswith("app.") for alias in node.names):
                raise ValueError("application module parser import requires explicit symbol import")
            return
        local_imports: dict[str, ast.AST] = {}
        for child in ast.walk(node):
            if isinstance(child, ast.Import | ast.ImportFrom):
                for alias in child.names:
                    local_imports[alias.asname or alias.name.split(".")[0]] = child
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                if child.id in local_imports:
                    visit(module, child.id, local_imports[child.id])
                elif child.id in bindings(module):
                    visit(module, child.id)

    for module, name in ROOTS:
        visit(module, name)
    payload = json.dumps({"schema": FINGERPRINT_SCHEMA, "dependencies": closure},
                         sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def matches_fingerprint(stored: object, schema: object, current: str) -> bool:
    """Accept only exact current digests or the pinned audited legacy pair."""
    if not isinstance(stored, str) or type(schema) is not int:
        return False
    if schema == FINGERPRINT_SCHEMA:
        return stored == current
    if schema == 1:
        return stored in LEGACY_PARSER_DIGESTS and LEGACY_PARSER_DIGESTS[stored] == current
    return False
