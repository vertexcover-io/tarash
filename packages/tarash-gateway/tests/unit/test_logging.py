"""Unit tests for logging utilities."""

import logging
from unittest.mock import patch, Mock

import pytest

from tarash.tarash_gateway.logging import (
    log_debug,
    log_info,
    log_warning,
    log_error,
    _redact_context,
    _redact_value,
    _get_logger,
    ProviderLogger,
)


class TestRedactValue:
    """Tests for _redact_value function."""

    def test_redact_none_value(self):
        """Test that None values pass through unchanged."""
        result = _redact_value(None)
        assert result is None

    def test_redact_int_value(self):
        """Test that integers pass through unchanged."""
        result = _redact_value(42)
        assert result == 42

    def test_redact_float_value(self):
        """Test that floats pass through unchanged."""
        result = _redact_value(3.14)
        assert result == 3.14

    def test_redact_bool_value(self):
        """Test that booleans pass through unchanged."""
        assert _redact_value(True) is True
        assert _redact_value(False) is False

    def test_redact_bytes_value(self):
        """Test that bytes are replaced with length representation."""
        test_bytes = b"secret data"
        result = _redact_value(test_bytes)
        assert result == f"<bytes: length={len(test_bytes)}>"

    def test_redact_short_string(self):
        """Test that short strings pass through unchanged."""
        short_string = "This is a short string"
        result = _redact_value(short_string)
        assert result == short_string

    def test_redact_long_string(self):
        """Test that long strings are truncated."""
        long_string = "a" * 200  # String longer than 100 chars
        result = _redact_value(long_string)

        # Should truncate: first 50 + "..." + last 50
        expected = f"{long_string[:50]}...{long_string[-50:]}"
        assert result == expected

    def test_redact_dict_value(self):
        """Test that dicts are recursively processed."""
        test_dict = {
            "name": "test",
            "count": 42,
            "data": b"bytes",
        }
        result = _redact_value(test_dict)

        assert result["name"] == "test"
        assert result["count"] == 42
        assert result["data"] == "<bytes: length=5>"

    def test_redact_list_value(self):
        """Test that lists are recursively processed."""
        test_list = ["string", 42, b"bytes", None]
        result = _redact_value(test_list)

        assert result[0] == "string"
        assert result[1] == 42
        assert result[2] == "<bytes: length=5>"
        assert result[3] is None

    def test_redact_tuple_value(self):
        """Test that tuples are recursively processed and returned as tuples."""
        test_tuple = ("string", 42, b"bytes")
        result = _redact_value(test_tuple)

        assert isinstance(result, tuple)
        assert result[0] == "string"
        assert result[1] == 42
        assert result[2] == "<bytes: length=5>"

    def test_redact_nested_dict(self):
        """Test that nested dicts are recursively processed."""
        nested_dict = {
            "outer": {
                "inner": {
                    "data": b"secret",
                    "count": 10,
                }
            }
        }
        result = _redact_value(nested_dict)

        assert result["outer"]["inner"]["data"] == "<bytes: length=6>"
        assert result["outer"]["inner"]["count"] == 10

    def test_redact_pydantic_model(self):
        """Test that pydantic models are converted to dicts."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str
            value: int

        model = TestModel(name="test", value=42)
        result = _redact_value(model)

        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["value"] == 42


class TestRedactContext:
    """Tests for _redact_context function."""

    def test_redact_none_context(self):
        """Test that None context returns empty dict."""
        result = _redact_context(None)
        assert result == {}

    def test_redact_empty_context(self):
        """Test that empty dict returns empty dict."""
        result = _redact_context({})
        assert result == {}

    def test_redact_sensitive_api_key(self):
        """Test that api_key is redacted."""
        context = {"api_key": "sk-1234567890"}
        result = _redact_context(context)
        assert result["api_key"] == "***REDACTED***"

    def test_redact_sensitive_password(self):
        """Test that password is redacted."""
        context = {"password": "secret123"}
        result = _redact_context(context)
        assert result["password"] == "***REDACTED***"

    def test_redact_sensitive_token(self):
        """Test that token is redacted."""
        context = {"token": "bearer-token-123"}
        result = _redact_context(context)
        assert result["token"] == "***REDACTED***"

    def test_redact_sensitive_authorization(self):
        """Test that authorization is redacted."""
        context = {"authorization": "Bearer xyz"}
        result = _redact_context(context)
        assert result["authorization"] == "***REDACTED***"

    def test_redact_mixed_context(self):
        """Test context with both sensitive and non-sensitive fields."""
        context = {
            "api_key": "sk-secret",
            "provider": "openai",
            "model": "gpt-4",
            "password": "secret",
        }
        result = _redact_context(context)

        assert result["api_key"] == "***REDACTED***"
        assert result["password"] == "***REDACTED***"
        assert result["provider"] == "openai"
        assert result["model"] == "gpt-4"

    def test_redact_bytes_in_context(self):
        """Test that bytes values are redacted in context."""
        context = {
            "image_data": b"fake image bytes",
            "name": "test",
        }
        result = _redact_context(context)

        assert result["image_data"] == "<bytes: length=16>"
        assert result["name"] == "test"

    def test_redact_nested_sensitive_fields(self):
        """Test that nested structures are processed."""
        context = {
            "request": {
                "api_key": "secret",
                "data": b"bytes",
            }
        }
        result = _redact_context(context)

        # Sensitive field name check is at top level only
        # But nested values should be redacted
        assert result["request"]["data"] == "<bytes: length=5>"

    def test_redact_case_insensitive(self):
        """Test that sensitive field detection is case-insensitive."""
        context = {
            "API_KEY": "secret",
            "Password": "secret",
            "ACCESS_TOKEN": "secret",
        }
        result = _redact_context(context)

        assert result["API_KEY"] == "***REDACTED***"
        assert result["Password"] == "***REDACTED***"
        assert result["ACCESS_TOKEN"] == "***REDACTED***"


class TestGetLogger:
    """Tests for _get_logger function."""

    def test_get_logger_returns_logger(self):
        """Test that _get_logger returns a Logger instance."""
        logger = _get_logger("test.logger")
        assert isinstance(logger, logging.Logger)

    def test_get_logger_with_name(self):
        """Test that logger has the correct name."""
        logger = _get_logger("tarash.tarash_gateway.test")
        assert logger.name == "tarash.tarash_gateway.test"


class TestLogDebug:
    """Tests for log_debug function."""

    def test_log_debug_simple_message(self):
        """Test logging a simple debug message without context."""
        with patch("logging.Logger.debug") as mock_debug:
            log_debug("Test debug message")
            mock_debug.assert_called_once_with("Test debug message")

    def test_log_debug_with_context(self):
        """Test logging debug message with context."""
        with patch("logging.Logger.debug") as mock_debug:
            context = {"key": "value"}
            log_debug("Debug with context", context=context)
            mock_debug.assert_called_once_with(
                "Debug with context | Context: {'key': 'value'}"
            )

    def test_log_debug_with_redaction(self):
        """Test logging debug message with redaction enabled."""
        with patch("logging.Logger.debug") as mock_debug:
            context = {"api_key": "secret", "model": "gpt-4"}
            log_debug("Debug with redaction", context=context, redact=True)

            call_args = mock_debug.call_args[0][0]
            assert "***REDACTED***" in call_args
            assert "gpt-4" in call_args

    def test_log_debug_custom_logger_name(self):
        """Test logging with custom logger name."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            log_debug("Test", logger_name="custom.logger")

            mock_get_logger.assert_called_once_with("custom.logger")
            mock_logger.debug.assert_called_once()


class TestLogInfo:
    """Tests for log_info function."""

    def test_log_info_simple_message(self):
        """Test logging a simple info message without context."""
        with patch("logging.Logger.info") as mock_info:
            log_info("Test info message")
            mock_info.assert_called_once_with("Test info message")

    def test_log_info_with_context(self):
        """Test logging info message with context."""
        with patch("logging.Logger.info") as mock_info:
            context = {"status": "success"}
            log_info("Info with context", context=context)
            mock_info.assert_called_once_with(
                "Info with context | Context: {'status': 'success'}"
            )

    def test_log_info_with_redaction(self):
        """Test logging info message with redaction enabled."""
        with patch("logging.Logger.info") as mock_info:
            context = {"password": "secret", "user": "alice"}
            log_info("Info with redaction", context=context, redact=True)

            call_args = mock_info.call_args[0][0]
            assert "***REDACTED***" in call_args
            assert "alice" in call_args


class TestLogWarning:
    """Tests for log_warning function."""

    def test_log_warning_simple_message(self):
        """Test logging a simple warning message without context."""
        with patch("logging.Logger.warning") as mock_warning:
            log_warning("Test warning message")
            mock_warning.assert_called_once_with("Test warning message")

    def test_log_warning_with_context(self):
        """Test logging warning message with context."""
        with patch("logging.Logger.warning") as mock_warning:
            context = {"retry_count": 3}
            log_warning("Warning with context", context=context)
            mock_warning.assert_called_once_with(
                "Warning with context | Context: {'retry_count': 3}"
            )

    def test_log_warning_with_redaction(self):
        """Test logging warning message with redaction enabled."""
        with patch("logging.Logger.warning") as mock_warning:
            context = {"token": "secret-token", "attempt": 2}
            log_warning("Warning with redaction", context=context, redact=True)

            call_args = mock_warning.call_args[0][0]
            assert "***REDACTED***" in call_args
            assert "2" in str(call_args)


class TestLogError:
    """Tests for log_error function."""

    def test_log_error_simple_message(self):
        """Test logging a simple error message without context."""
        with patch("logging.Logger.error") as mock_error:
            log_error("Test error message")
            mock_error.assert_called_once_with("Test error message", exc_info=False)

    def test_log_error_with_context(self):
        """Test logging error message with context."""
        with patch("logging.Logger.error") as mock_error:
            context = {"error_code": 500}
            log_error("Error with context", context=context)
            mock_error.assert_called_once_with(
                "Error with context | Context: {'error_code': 500}",
                exc_info=False,
            )

    def test_log_error_with_redaction(self):
        """Test logging error message with redaction enabled."""
        with patch("logging.Logger.error") as mock_error:
            context = {"api_key": "secret", "status": "failed"}
            log_error("Error with redaction", context=context, redact=True)

            call_args = mock_error.call_args[0][0]
            assert "***REDACTED***" in call_args
            assert "failed" in call_args

    def test_log_error_with_exc_info(self):
        """Test logging error message with exception info."""
        with patch("logging.Logger.error") as mock_error:
            log_error("Error with exc_info", exc_info=True)
            mock_error.assert_called_once_with("Error with exc_info", exc_info=True)

    def test_log_error_with_context_and_exc_info(self):
        """Test logging error message with both context and exc_info."""
        with patch("logging.Logger.error") as mock_error:
            context = {"operation": "download"}
            log_error("Error with both", context=context, exc_info=True)

            call_args = mock_error.call_args
            assert (
                "Error with both | Context: {'operation': 'download'}" in call_args[0]
            )
            assert call_args[1]["exc_info"] is True


# ==================== ProviderLogger Tests ====================


@pytest.fixture
def provider_logger():
    """Create a basic ProviderLogger for testing."""
    return ProviderLogger(
        provider="test-provider",
        model="test-model",
        logger_name="test.logger",
    )


@pytest.fixture
def provider_logger_with_request_id():
    """Create a ProviderLogger with request_id for testing."""
    return ProviderLogger(
        provider="test-provider",
        model="test-model",
        logger_name="test.logger",
        request_id="req-123",
    )


def test_provider_logger_initialization(provider_logger):
    """ProviderLogger initializes with provider, model, and logger_name."""
    assert provider_logger.provider == "test-provider"
    assert provider_logger.model == "test-model"
    assert provider_logger.logger_name == "test.logger"
    assert provider_logger.request_id is None


def test_provider_logger_initialization_with_request_id(
    provider_logger_with_request_id,
):
    """ProviderLogger can be initialized with request_id."""
    assert provider_logger_with_request_id.request_id == "req-123"


def test_provider_logger_base_context_without_request_id(provider_logger):
    """Base context includes provider and model but not request_id when None."""
    context = provider_logger._build_context()
    assert context == {
        "provider": "test-provider",
        "model": "test-model",
    }
    assert "request_id" not in context


def test_provider_logger_base_context_with_request_id(provider_logger_with_request_id):
    """Base context includes request_id when provided."""
    context = provider_logger_with_request_id._build_context()
    assert context == {
        "provider": "test-provider",
        "model": "test-model",
        "request_id": "req-123",
    }


def test_provider_logger_build_context_merges_extra(provider_logger):
    """Extra fields are merged with base context."""
    extra = {"status": "completed", "duration": 5.2}
    context = provider_logger._build_context(extra)

    assert context == {
        "provider": "test-provider",
        "model": "test-model",
        "status": "completed",
        "duration": 5.2,
    }


def test_provider_logger_build_context_extra_overrides_base(provider_logger):
    """Extra fields can override base context fields."""
    extra = {"provider": "overridden-provider"}
    context = provider_logger._build_context(extra)

    assert context["provider"] == "overridden-provider"
    assert context["model"] == "test-model"


def test_provider_logger_with_request_id_returns_new_instance(provider_logger):
    """with_request_id returns a new ProviderLogger instance."""
    new_logger = provider_logger.with_request_id("req-456")

    assert new_logger is not provider_logger
    assert new_logger.request_id == "req-456"
    assert new_logger.provider == provider_logger.provider
    assert new_logger.model == provider_logger.model
    assert new_logger.logger_name == provider_logger.logger_name


def test_provider_logger_with_request_id_original_unchanged(provider_logger):
    """with_request_id does not modify the original logger."""
    provider_logger.with_request_id("req-456")

    assert provider_logger.request_id is None


def test_provider_logger_debug_calls_log_debug(provider_logger):
    """debug() calls log_debug with merged context."""
    with patch("tarash.tarash_gateway.logging.log_debug") as mock_log_debug:
        provider_logger.debug("Test message", {"extra_key": "value"})

        mock_log_debug.assert_called_once()
        call_kwargs = mock_log_debug.call_args[1]
        assert call_kwargs["context"]["provider"] == "test-provider"
        assert call_kwargs["context"]["model"] == "test-model"
        assert call_kwargs["context"]["extra_key"] == "value"
        assert call_kwargs["logger_name"] == "test.logger"
        assert call_kwargs["redact"] is False


def test_provider_logger_debug_with_redact(provider_logger):
    """debug() passes redact parameter to log_debug."""
    with patch("tarash.tarash_gateway.logging.log_debug") as mock_log_debug:
        provider_logger.debug("Test message", {"api_key": "secret"}, redact=True)

        call_kwargs = mock_log_debug.call_args[1]
        assert call_kwargs["redact"] is True


def test_provider_logger_info_calls_log_info(provider_logger):
    """info() calls log_info with merged context."""
    with patch("tarash.tarash_gateway.logging.log_info") as mock_log_info:
        provider_logger.info("Test message", {"status": "ok"})

        mock_log_info.assert_called_once()
        call_kwargs = mock_log_info.call_args[1]
        assert call_kwargs["context"]["provider"] == "test-provider"
        assert call_kwargs["context"]["status"] == "ok"


def test_provider_logger_warning_calls_log_warning(provider_logger):
    """warning() calls log_warning with merged context."""
    with patch("tarash.tarash_gateway.logging.log_warning") as mock_log_warning:
        provider_logger.warning("Test warning")

        mock_log_warning.assert_called_once()
        call_kwargs = mock_log_warning.call_args[1]
        assert call_kwargs["context"]["provider"] == "test-provider"


def test_provider_logger_error_calls_log_error(provider_logger):
    """error() calls log_error with merged context."""
    with patch("tarash.tarash_gateway.logging.log_error") as mock_log_error:
        provider_logger.error("Test error", {"error_code": 500})

        mock_log_error.assert_called_once()
        call_kwargs = mock_log_error.call_args[1]
        assert call_kwargs["context"]["provider"] == "test-provider"
        assert call_kwargs["context"]["error_code"] == 500
        assert call_kwargs["exc_info"] is False


def test_provider_logger_error_with_exc_info(provider_logger):
    """error() passes exc_info parameter to log_error."""
    with patch("tarash.tarash_gateway.logging.log_error") as mock_log_error:
        provider_logger.error("Test error", exc_info=True)

        call_kwargs = mock_log_error.call_args[1]
        assert call_kwargs["exc_info"] is True


def test_provider_logger_debug_without_extra(provider_logger):
    """debug() works without extra parameter."""
    with patch("tarash.tarash_gateway.logging.log_debug") as mock_log_debug:
        provider_logger.debug("Simple message")

        mock_log_debug.assert_called_once()
        call_kwargs = mock_log_debug.call_args[1]
        assert call_kwargs["context"] == {
            "provider": "test-provider",
            "model": "test-model",
        }


def test_provider_logger_includes_request_id_in_logs(provider_logger_with_request_id):
    """Logs include request_id when set."""
    with patch("tarash.tarash_gateway.logging.log_info") as mock_log_info:
        provider_logger_with_request_id.info("Request complete")

        call_kwargs = mock_log_info.call_args[1]
        assert call_kwargs["context"]["request_id"] == "req-123"


def test_provider_logger_chained_with_request_id(provider_logger):
    """ProviderLogger can be chained with with_request_id for request lifecycle."""
    with patch("tarash.tarash_gateway.logging.log_debug") as mock_log_debug:
        # Simulate request lifecycle
        provider_logger.debug("Starting request")

        # After getting request_id
        logger_with_id = provider_logger.with_request_id("req-789")
        logger_with_id.debug("Request submitted")

        # First call should not have request_id
        first_call_context = mock_log_debug.call_args_list[0][1]["context"]
        assert "request_id" not in first_call_context

        # Second call should have request_id
        second_call_context = mock_log_debug.call_args_list[1][1]["context"]
        assert second_call_context["request_id"] == "req-789"
