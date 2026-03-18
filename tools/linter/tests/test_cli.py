"""Tests for the CLI interface."""

import json
from pathlib import Path

import pytest

from tarash_linter.cli import (
    load_config,
    format_violations_text,
    format_violations_json,
    main,
)
from tarash_linter.models import LintConfig, Violation


# --- Helpers ---


def _make_workspace_root(tmp_path: Path) -> Path:
    """Create a minimal monorepo root with pyproject.toml workspace config."""
    (tmp_path / "pyproject.toml").write_text(
        '[tool.uv.workspace]\nmembers = ["packages/*"]\n'
    )
    providers_dir = (
        tmp_path
        / "packages"
        / "tarash-gateway"
        / "src"
        / "tarash"
        / "tarash_gateway"
        / "providers"
    )
    providers_dir.mkdir(parents=True)
    (providers_dir / "__init__.py").write_text("")
    (providers_dir / "field_mappers.py").write_text("")
    return tmp_path


def test_load_config_from_pyproject(tmp_path: Path):
    """Load config from [tool.tarash-lint] in pyproject.toml."""
    (tmp_path / "pyproject.toml").write_text(
        '[tool.tarash-lint]\nselect = ["TRH1"]\nignore = ["TRH101"]\nexclude-providers = ["azure_openai"]\n'
    )
    config = load_config(tmp_path)
    assert config.select == ["TRH1"]
    assert config.ignore == ["TRH101"]
    assert config.exclude_providers == ["azure_openai"]


def test_load_config_missing_pyproject(tmp_path: Path):
    """Missing pyproject.toml returns default config."""
    config = load_config(tmp_path)
    assert config == LintConfig()


def test_load_config_no_tarash_lint_section(tmp_path: Path):
    """pyproject.toml without [tool.tarash-lint] returns default config."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'foo'\n")
    config = load_config(tmp_path)
    assert config == LintConfig()


def test_format_violations_text():
    """Text format matches file:line:col: CODE message."""
    violations = [
        Violation(
            code="TRH101",
            file="providers/fal.py",
            line=100,
            col=0,
            message="missing '_get_client'",
        ),
        Violation(
            code="TRH201",
            file="providers/fal.py",
            line=100,
            col=0,
            message="not registered",
        ),
    ]
    output = format_violations_text(violations)
    lines = output.strip().split("\n")
    assert len(lines) == 2
    assert lines[0] == "providers/fal.py:100:0: TRH101 missing '_get_client'"


def test_format_violations_json():
    """JSON format produces a valid JSON array."""
    violations = [
        Violation(
            code="TRH101", file="providers/fal.py", line=100, col=0, message="test"
        ),
    ]
    output = format_violations_json(violations)
    data = json.loads(output)
    assert len(data) == 1
    assert data[0]["code"] == "TRH101"


def test_format_empty_violations_text():
    """Empty violations produce empty output."""
    assert format_violations_text([]) == ""


def test_format_empty_violations_json():
    """Empty violations produce empty JSON array."""
    assert format_violations_json([]) == "[]"


# --- main() integration tests ---


def test_main_select_override(tmp_path: Path, capsys):
    """CLI --select overrides the select list from pyproject.toml."""
    root = _make_workspace_root(tmp_path)
    # Config file says TRH4, but CLI will say TRH1 — only TRH1 rules should run.
    (root / "pyproject.toml").write_text(
        '[tool.uv.workspace]\nmembers = ["packages/*"]\n\n[tool.tarash-lint]\nselect = ["TRH4"]\n'
    )

    with pytest.raises(SystemExit) as exc_info:
        main(["--project-root", str(root), "--select", "TRH1"])

    # exit 0 (no violations) or 1 (violations) — both mean the CLI ran correctly.
    # The key assertion is that TRH4 rules were NOT used (no TRH4 codes in output).
    captured = capsys.readouterr()
    assert exc_info.value.code in (0, 1)
    assert "TRH4" not in captured.out


def test_main_ignore_merging(tmp_path: Path, capsys):
    """CLI --ignore merges with config ignore list."""
    root = _make_workspace_root(tmp_path)
    # Config already ignores TRH201; CLI adds TRH101.
    (root / "pyproject.toml").write_text(
        '[tool.uv.workspace]\nmembers = ["packages/*"]\n\n[tool.tarash-lint]\nignore = ["TRH201"]\n'
    )

    with pytest.raises(SystemExit) as exc_info:
        main(["--project-root", str(root), "--ignore", "TRH101"])

    assert exc_info.value.code in (0, 1)
    captured = capsys.readouterr()
    # Neither TRH101 nor TRH201 should appear in output (both are ignored).
    assert "TRH101" not in captured.out
    assert "TRH201" not in captured.out


def test_main_format_json(tmp_path: Path, capsys):
    """--format json outputs valid JSON regardless of violations."""
    root = _make_workspace_root(tmp_path)
    # Minimal workspace pyproject without tarash-lint section.
    (root / "pyproject.toml").write_text(
        '[tool.uv.workspace]\nmembers = ["packages/*"]\n'
    )

    with pytest.raises(SystemExit) as exc_info:
        main(["--project-root", str(root), "--format", "json"])

    assert exc_info.value.code in (0, 1)
    captured = capsys.readouterr()
    # Output must be valid JSON (array), or empty (no violations -> no print).
    stdout = captured.out.strip()
    if stdout:
        data = json.loads(stdout)
        assert isinstance(data, list)
