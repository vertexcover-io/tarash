"""Tests for the logging module."""

import logging

import pytest

from tarash.tarash_audio_mixer.logging import (
    _redact_context,
    _redact_value,
    log_error,
    log_info,
)


@pytest.fixture
def caplog_at_debug(caplog):
    """Set caplog to capture DEBUG level."""
    with caplog.at_level(logging.DEBUG, logger="tarash.tarash_audio_mixer"):
        yield caplog


def test_log_info_with_context(caplog_at_debug):
    """log_info includes context dict in output."""
    log_info("step done", context={"count": 5})
    assert "step done | Context:" in caplog_at_debug.text
    assert "'count': 5" in caplog_at_debug.text


def test_log_info_without_context(caplog_at_debug):
    """log_info emits message without context suffix."""
    log_info("hello")
    assert "hello" in caplog_at_debug.text
    assert "Context" not in caplog_at_debug.text


def test_log_error_with_exc_info(caplog_at_debug):
    """log_error with exc_info=True includes traceback."""
    try:
        raise ValueError("boom")
    except ValueError:
        log_error("caught error", exc_info=True)
    assert "caught error" in caplog_at_debug.text
    assert "ValueError" in caplog_at_debug.text


def test_redact_sensitive_fields():
    """Sensitive field values are replaced with ***REDACTED***."""
    ctx = {
        "api_key": "sk-secret",
        "password": "p@ss",
        "token": "tok123",
        "user": "alice",
    }
    result = _redact_context(ctx)
    assert result["api_key"] == "***REDACTED***"
    assert result["password"] == "***REDACTED***"
    assert result["token"] == "***REDACTED***"
    assert result["user"] == "alice"


def test_redact_bytes_shows_length():
    """Bytes are replaced with length description."""
    assert _redact_value(b"hello") == "<bytes: length=5>"


def test_redact_long_strings_truncated():
    """Strings over 100 chars are truncated."""
    long_str = "a" * 200
    result = _redact_value(long_str)
    assert result.startswith("a" * 50)
    assert "..." in result
    assert result.endswith("a" * 50)
