"""CLI entry point for tarash-lint."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

from tarash_linter.models import LintConfig, Violation
from tarash_linter.runner import run_lint


def load_config(project_root: Path) -> LintConfig:
    """Load lint config from pyproject.toml [tool.tarash-lint] section."""
    pyproject = project_root / "pyproject.toml"
    if not pyproject.is_file():
        return LintConfig()

    with open(pyproject, "rb") as f:
        data = tomllib.load(f)

    section = data.get("tool", {}).get("tarash-lint", {})
    if not section:
        return LintConfig()

    return LintConfig(
        select=section.get("select", []),
        ignore=section.get("ignore", []),
        exclude_providers=section.get("exclude-providers", []),
    )


def format_violations_text(violations: list[Violation]) -> str:
    """Format violations as standard linter text output."""
    if not violations:
        return ""
    return "\n".join(v.format_text() for v in violations)


def format_violations_json(violations: list[Violation]) -> str:
    """Format violations as JSON array."""
    return json.dumps([v.to_dict() for v in violations], indent=2)


def _find_project_root() -> Path:
    """Auto-detect project root by walking up to find pyproject.toml with workspace config."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        pyproject = parent / "pyproject.toml"
        if pyproject.is_file():
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
            if (
                "tool" in data
                and "uv" in data["tool"]
                and "workspace" in data["tool"]["uv"]
            ):
                return parent
    return current


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="tarash-lint",
        description="Structural linter for tarash-gateway providers",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Root of the tarash monorepo (auto-detected if omitted)",
    )
    parser.add_argument(
        "--select",
        type=str,
        default=None,
        help="Comma-separated rule codes/prefixes to enable (e.g., TRH1,TRH3)",
    )
    parser.add_argument(
        "--ignore",
        type=str,
        default=None,
        help="Comma-separated rule codes/prefixes to ignore (e.g., TRH301)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Lint only this provider (e.g., fal)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        dest="output_format",
        help="Output format (default: text)",
    )

    args = parser.parse_args(argv)

    # Resolve project root
    project_root = args.project_root or _find_project_root()
    project_root = project_root.resolve()

    # Load config from pyproject.toml, then override with CLI args
    config = load_config(project_root)
    if args.select:
        config = LintConfig(
            select=[s.strip() for s in args.select.split(",")],
            ignore=config.ignore,
            exclude_providers=config.exclude_providers,
        )
    if args.ignore:
        cli_ignore = [s.strip() for s in args.ignore.split(",")]
        config = LintConfig(
            select=config.select,
            ignore=list(set(config.ignore + cli_ignore)),
            exclude_providers=config.exclude_providers,
        )

    # Run linter
    try:
        violations = run_lint(project_root, config, provider_filter=args.provider)
    except Exception as e:
        print(f"tarash-lint: error: {e}", file=sys.stderr)
        sys.exit(2)

    # Format and print output
    if args.output_format == "json":
        output = format_violations_json(violations)
    else:
        output = format_violations_text(violations)

    if output:
        print(output)

    # Exit code
    sys.exit(1 if violations else 0)
