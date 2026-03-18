"""Integration test: run tarash-lint against the real tarash-gateway codebase.

This test verifies that the linter produces only known violations on the current
codebase (with azure_openai excluded and TRH106 ignored), proving no false positives.
"""

from pathlib import Path

import pytest

from tarash_linter.models import LintConfig
from tarash_linter.runner import run_lint


def _find_repo_root() -> Path:
    """Walk up from this file to find the monorepo root."""
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "packages" / "tarash-gateway").is_dir():
            return parent
    pytest.skip("Could not find tarash monorepo root")
    return Path()  # unreachable, for type checker


# Known violations in the current codebase that are legitimate findings,
# not false positives. These represent real coverage gaps.
KNOWN_VIOLATIONS = {
    # stability has no error handling tests (only field mapper and success tests)
    ("TRH403", "stability"),
}


def test_no_false_positives_on_real_codebase():
    """The linter should produce only known violations on the existing codebase."""
    root = _find_repo_root()
    config = LintConfig(exclude_providers=["azure_openai"], ignore=["TRH106"])
    violations = run_lint(root, config)

    unexpected = [
        v for v in violations if (v.code, Path(v.file).stem) not in KNOWN_VIOLATIONS
    ]
    if unexpected:
        report = "\n".join(v.format_text() for v in unexpected)
        pytest.fail(f"Unexpected violations ({len(unexpected)}):\n{report}")
