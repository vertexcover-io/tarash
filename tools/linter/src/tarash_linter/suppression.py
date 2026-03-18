"""Inline suppression via # noqa comments."""

from __future__ import annotations

import re
from pathlib import Path

# Matches: # noqa   or   # noqa: TRH101, TRH201
_NOQA_RE = re.compile(
    r"#\s*noqa\b(?:\s*:\s*(?P<codes>[A-Z0-9,\s]+))?",
    re.IGNORECASE,
)


def parse_noqa_comments(lines: list[str]) -> dict[int, set[str] | None]:
    """Parse source lines for # noqa comments.

    Args:
        lines: Source file lines (1-indexed line numbers in result).

    Returns:
        Mapping of line_number -> set of suppressed codes, or None for blanket suppression.
        Only lines with # noqa comments appear in the dict.
    """
    result: dict[int, set[str] | None] = {}
    for i, line in enumerate(lines, start=1):
        match = _NOQA_RE.search(line)
        if match:
            codes_str = match.group("codes")
            if codes_str:
                codes = {c.strip() for c in codes_str.split(",") if c.strip()}
                result[i] = codes
            else:
                result[i] = None  # blanket suppression
    return result


def parse_noqa_comments_for_file(filepath: Path) -> dict[int, set[str] | None]:
    """Parse a source file for # noqa comments.

    Args:
        filepath: Path to the source file.

    Returns:
        Same as parse_noqa_comments.
    """
    try:
        lines = filepath.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError:
        return {}
    return parse_noqa_comments(lines)
