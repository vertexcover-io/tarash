"""Tests for the logging module."""

import logging

import pytest

from tarash.tarash_silence_remover.logging import (
    _redact_context,
    _redact_value,
    log_debug,
    log_error,
    log_info,
    log_warning,
)


@pytest.fixture
def caplog_at_debug(caplog):
    """Set caplog to capture DEBUG level."""
    with caplog.at_level(logging.DEBUG, logger="tarash.tarash_silence_remover"):
        yield caplog


def test_log_info_without_context(caplog_at_debug):
    """log_info emits message without context suffix."""
    log_info("hello")
    assert "hello" in caplog_at_debug.text
    assert "Context" not in caplog_at_debug.text


def test_log_info_with_context(caplog_at_debug):
    """log_info includes context dict in output."""
    log_info("step done", context={"count": 5})
    assert "step done | Context:" in caplog_at_debug.text
    assert "'count': 5" in caplog_at_debug.text


def test_log_debug_with_context(caplog_at_debug):
    """log_debug emits at DEBUG level with context."""
    log_debug("detail", context={"key": "val"})
    assert "detail | Context:" in caplog_at_debug.text


def test_log_warning_without_context(caplog_at_debug):
    """log_warning emits at WARNING level."""
    log_warning("watch out")
    assert "watch out" in caplog_at_debug.text


def test_log_warning_with_context(caplog_at_debug):
    """log_warning includes context."""
    log_warning("watch out", context={"reason": "low memory"})
    assert "watch out | Context:" in caplog_at_debug.text


def test_log_error_without_context(caplog_at_debug):
    """log_error emits at ERROR level."""
    log_error("boom")
    assert "boom" in caplog_at_debug.text


def test_log_error_with_context(caplog_at_debug):
    """log_error includes context."""
    log_error("boom", context={"code": 500})
    assert "boom | Context:" in caplog_at_debug.text


def test_log_info_custom_logger_name(caplog):
    """log_info uses custom logger_name."""
    name = "tarash.tarash_silence_remover.api"
    with caplog.at_level(logging.INFO, logger=name):
        log_info("custom", logger_name=name)
    assert "custom" in caplog.text


def test_log_info_default_logger_name(caplog):
    """Default logger name is tarash.tarash_silence_remover."""
    with caplog.at_level(logging.INFO, logger="tarash.tarash_silence_remover"):
        log_info("default logger")
    assert "default logger" in caplog.text


# --- _redact_context ---


def test_redact_context_sensitive_fields():
    """Sensitive field values are replaced with ***REDACTED***."""
    ctx = {"api_key": "sk-secret", "user": "alice"}
    result = _redact_context(ctx)
    assert result["api_key"] == "***REDACTED***"
    assert result["user"] == "alice"


def test_redact_context_case_insensitive():
    """Sensitive field matching is case-insensitive."""
    ctx = {"Authorization": "Bearer xyz", "name": "bob"}
    result = _redact_context(ctx)
    assert result["Authorization"] == "***REDACTED***"
    assert result["name"] == "bob"


def test_redact_context_none_returns_empty():
    """None context returns empty dict."""
    assert _redact_context(None) == {}


def test_redact_context_empty_returns_empty():
    """Empty context returns empty dict."""
    assert _redact_context({}) == {}


def test_redact_context_partial_match():
    """Fields containing sensitive substrings are redacted."""
    ctx = {"my_access_token": "abc123"}
    result = _redact_context(ctx)
    assert result["my_access_token"] == "***REDACTED***"


# --- _redact_value ---


def test_redact_value_bytes():
    """Bytes are replaced with length description."""
    assert _redact_value(b"hello") == "<bytes: length=5>"


def test_redact_value_long_string():
    """Strings over 100 chars are truncated."""
    long_str = "a" * 200
    result = _redact_value(long_str)
    assert result.startswith("a" * 50)
    assert "..." in result
    assert result.endswith("a" * 50)


def test_redact_value_short_string():
    """Short strings are returned as-is."""
    assert _redact_value("short") == "short"


def test_redact_value_none():
    """None passes through."""
    assert _redact_value(None) is None


def test_redact_value_numbers():
    """Numbers pass through."""
    assert _redact_value(42) == 42
    assert _redact_value(3.14) == 3.14


def test_redact_value_bool():
    """Booleans pass through."""
    assert _redact_value(True) is True


def test_redact_value_nested_dict():
    """Dicts are recursively processed."""
    result = _redact_value({"inner": b"data"})
    assert result == {"inner": "<bytes: length=4>"}


def test_redact_value_list():
    """Lists are recursively processed."""
    result = _redact_value([b"a", "ok"])
    assert result == ["<bytes: length=1>", "ok"]


def test_redact_value_tuple():
    """Tuples are recursively processed."""
    result = _redact_value((b"a", 1))
    assert result == ("<bytes: length=1>", 1)


def test_redact_value_pydantic_model():
    """Pydantic models are converted to dict first."""
    from pydantic import BaseModel

    class Dummy(BaseModel):
        name: str = "test"
        data: bytes = b"secret"

    result = _redact_value(Dummy())
    assert isinstance(result, dict)
    assert result["name"] == "test"
    assert result["data"] == "<bytes: length=6>"


# --- redact parameter on log functions ---


def test_log_info_redact_flag(caplog_at_debug):
    """redact=True causes sensitive context fields to be redacted."""
    log_info("msg", context={"api_key": "secret", "count": 1}, redact=True)
    assert "***REDACTED***" in caplog_at_debug.text
    assert "secret" not in caplog_at_debug.text
    assert "'count': 1" in caplog_at_debug.text
