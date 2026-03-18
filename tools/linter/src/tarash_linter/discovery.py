"""AST-based provider discovery for tarash-gateway."""

from __future__ import annotations

import ast
from pathlib import Path

from tarash_linter.models import ProviderInfo

# Files in providers/ to skip
_EXCLUDED_FILES = {"__init__.py", "field_mappers.py"}

# Gateway package path relative to project root
_GATEWAY_PROVIDERS_REL = Path(
    "packages/tarash-gateway/src/tarash/tarash_gateway/providers"
)


def _extract_methods(class_node: ast.ClassDef) -> frozenset[str]:
    """Extract all method names (sync and async) from a class body."""
    methods: set[str] = set()
    for node in class_node.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.add(node.name)
    return frozenset(methods)


def _find_handler_classes(source: str) -> list[tuple[str, int, frozenset[str]]]:
    """Parse source and find all classes ending with ProviderHandler.

    Returns list of (class_name, line_number, methods).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    results = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name.endswith("ProviderHandler"):
            methods = _extract_methods(node)
            results.append((node.name, node.lineno, methods))
    return results


def discover_providers(project_root: Path) -> list[ProviderInfo]:
    """Discover all provider handler classes in tarash-gateway.

    Args:
        project_root: Root of the tarash monorepo.

    Returns:
        List of ProviderInfo for each discovered handler.
    """
    providers_dir = project_root / _GATEWAY_PROVIDERS_REL
    if not providers_dir.is_dir():
        return []

    results: list[ProviderInfo] = []

    for py_file in sorted(providers_dir.glob("*.py")):
        if py_file.name in _EXCLUDED_FILES:
            continue

        source = py_file.read_text(encoding="utf-8")
        handler_classes = _find_handler_classes(source)

        for class_name, class_line, methods in handler_classes:
            provider_name = py_file.stem
            rel_path = str(py_file.relative_to(project_root))
            results.append(
                ProviderInfo(
                    name=provider_name,
                    file=rel_path,
                    class_name=class_name,
                    class_line=class_line,
                    methods=methods,
                )
            )

    return results
