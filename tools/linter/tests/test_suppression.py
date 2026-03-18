"""Tests for # noqa inline suppression."""

from tarash_linter.suppression import parse_noqa_comments


def test_bare_noqa_suppresses_all():
    """# noqa with no codes suppresses all rules on that line."""
    lines = ["x = 1  # noqa\n", "y = 2\n"]
    result = parse_noqa_comments(lines)
    assert result[1] is None  # None = suppress all
    assert 2 not in result


def test_noqa_with_single_code():
    """# noqa: TRH101 suppresses only TRH101."""
    lines = ["x = 1  # noqa: TRH101\n"]
    result = parse_noqa_comments(lines)
    assert result[1] == {"TRH101"}


def test_noqa_with_multiple_codes():
    """# noqa: TRH101, TRH201 suppresses both."""
    lines = ["x = 1  # noqa: TRH101, TRH201\n"]
    result = parse_noqa_comments(lines)
    assert result[1] == {"TRH101", "TRH201"}


def test_noqa_case_insensitive():
    """# NOQA should work the same as # noqa."""
    lines = ["x = 1  # NOQA: TRH101\n"]
    result = parse_noqa_comments(lines)
    assert result[1] == {"TRH101"}


def test_noqa_with_whitespace():
    """Whitespace around codes should be handled."""
    lines = ["x = 1  # noqa:  TRH101 , TRH201 \n"]
    result = parse_noqa_comments(lines)
    assert result[1] == {"TRH101", "TRH201"}


def test_no_noqa_returns_empty():
    """Lines without # noqa produce an empty dict."""
    lines = ["x = 1\n", "y = 2\n"]
    result = parse_noqa_comments(lines)
    assert result == {}


def test_noqa_on_multiple_lines():
    """Multiple lines can have noqa comments."""
    lines = [
        "a = 1  # noqa: TRH101\n",
        "b = 2\n",
        "c = 3  # noqa\n",
    ]
    result = parse_noqa_comments(lines)
    assert result[1] == {"TRH101"}
    assert 2 not in result
    assert result[3] is None
