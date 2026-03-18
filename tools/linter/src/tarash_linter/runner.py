"""Lint runner — builds context and orchestrates rules."""

from __future__ import annotations

import ast
from pathlib import Path

from tarash_linter.discovery import discover_providers
from tarash_linter.models import LintConfig, Violation
from tarash_linter.rules import RULES, RuleContext, load_all_rules

# Auto-discover and register all rule modules
load_all_rules()

_GATEWAY_SRC_REL = Path("packages/tarash-gateway/src/tarash/tarash_gateway")
_GATEWAY_TESTS_REL = Path("packages/tarash-gateway/tests")


def parse_registry_mapping(source: str) -> dict[str, str]:
    """Parse registry.py source and extract provider -> handler class mapping.

    Looks for patterns like:
        if provider == "fal":
            ... FalProviderHandler() ...
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    mapping: dict[str, str] = {}

    for node in ast.walk(tree):
        # Match: if/elif provider == "name"
        if not isinstance(node, ast.If):
            continue

        # Process this if and all elif branches
        branches: list[ast.If] = [node]
        # Collect elif chain
        current = node
        while (
            current.orelse
            and len(current.orelse) == 1
            and isinstance(current.orelse[0], ast.If)
        ):
            current = current.orelse[0]
            branches.append(current)

        for branch in branches:
            test = branch.test
            # Match: provider == "name"
            if not isinstance(test, ast.Compare):
                continue
            if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
                continue
            if len(test.comparators) != 1:
                continue
            comparator = test.comparators[0]
            if not isinstance(comparator, ast.Constant) or not isinstance(
                comparator.value, str
            ):
                continue

            provider_name = comparator.value

            # Find handler class name in branch body by looking for Class() calls
            for stmt in ast.walk(ast.Module(body=branch.body, type_ignores=[])):
                if isinstance(stmt, ast.Call) and isinstance(stmt.func, ast.Name):
                    class_name = stmt.func.id
                    if class_name.endswith("ProviderHandler"):
                        mapping[provider_name] = class_name
                        break

    return mapping


def scan_test_files(test_root: Path, provider_names: list[str]) -> dict[str, list[str]]:
    """Scan a test directory for files matching test_{provider}*.py.

    Args:
        test_root: e.g., packages/tarash-gateway/tests/unit
        provider_names: list of provider names to search for

    Returns:
        Mapping of provider_name -> list of matching test file paths (as strings).
    """
    result: dict[str, list[str]] = {}
    if not test_root.is_dir():
        return result

    all_test_files = list(test_root.rglob("test_*.py"))
    for name in provider_names:
        prefix = f"test_{name}"
        matching = [
            str(f)
            for f in all_test_files
            if (f.name == f"{prefix}.py" or f.name.startswith(f"{prefix}_"))
        ]
        if matching:
            result[name] = sorted(matching)

    return result


def scan_test_functions(
    test_files: dict[str, list[str]],
) -> dict[str, set[str]]:
    """AST-parse test files and extract test function names per provider.

    Args:
        test_files: provider_name -> list of file paths

    Returns:
        provider_name -> set of test function names
    """
    result: dict[str, set[str]] = {}
    for provider_name, files in test_files.items():
        functions: set[str] = set()
        for filepath in files:
            try:
                source = Path(filepath).read_text(encoding="utf-8")
                tree = ast.parse(source)
            except (SyntaxError, OSError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("test_"):
                        functions.add(node.name)
        if functions:
            result[provider_name] = functions
    return result


def build_context(project_root: Path, provider_names: list[str]) -> RuleContext:
    """Build the shared RuleContext by parsing gateway files."""
    gw_src = project_root / _GATEWAY_SRC_REL
    gw_tests = project_root / _GATEWAY_TESTS_REL

    # Parse registry.py
    registry_file = gw_src / "registry.py"
    registry_source = (
        registry_file.read_text(encoding="utf-8") if registry_file.is_file() else ""
    )
    registry_mapping = parse_registry_mapping(registry_source)

    # Scan unit test files
    unit_test_root = gw_tests / "unit"
    unit_test_files = scan_test_files(unit_test_root, provider_names)
    unit_test_functions = scan_test_functions(unit_test_files)

    # Scan e2e test files
    e2e_test_root = gw_tests / "e2e"
    e2e_test_files = scan_test_files(e2e_test_root, provider_names)

    return RuleContext(
        project_root=str(project_root),
        registry_mapping=registry_mapping,
        unit_test_files=unit_test_files,
        unit_test_functions=unit_test_functions,
        e2e_test_files=e2e_test_files,
    )


def run_lint(
    project_root: Path,
    config: LintConfig,
    provider_filter: str | None = None,
) -> list[Violation]:
    """Run all selected rules against discovered providers.

    Args:
        project_root: Root of the tarash monorepo.
        config: Lint configuration (select, ignore, exclude-providers).
        provider_filter: If set, only lint this single provider.

    Returns:
        List of all violations found.
    """
    providers = discover_providers(project_root)

    # Filter by exclude list
    providers = [p for p in providers if p.name not in config.exclude_providers]

    # Filter by --provider flag
    if provider_filter:
        providers = [p for p in providers if p.name == provider_filter]

    if not providers:
        return []

    provider_names = [p.name for p in providers]
    context = build_context(project_root, provider_names)

    violations: list[Violation] = []
    for provider in providers:
        for rule in RULES:
            if not config.is_rule_selected(rule.code):
                continue
            violations.extend(rule.check(provider, context))

    # Sort by file, line, code for deterministic output
    violations.sort(key=lambda v: (v.file, v.line, v.code))
    return violations
