"""Tests for the CLI interface."""

import json
from pathlib import Path

from tarash_linter.cli import (
    load_config,
    format_violations_text,
    format_violations_json,
)
from tarash_linter.models import LintConfig, Violation


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
