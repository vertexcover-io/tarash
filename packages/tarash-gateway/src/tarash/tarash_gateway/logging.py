"""Logging utilities for tarash-gateway."""

import logging
from dataclasses import dataclass, field
from typing import Any, cast

# Sensitive fields that should be sanitized in logs
_SENSITIVE_FIELDS = {
    "api_key",
    "password",
    "token",
    "secret",
    "authorization",
    "auth",
    "credentials",
    "access_token",
    "refresh_token",
}


def _redact_value(value: Any) -> Any:
    """Redact a value by replacing sensitive data with safe representations.

    Handles:
    - Bytes: Shows length instead of content
    - Dicts: Recursively processes each value
    - Lists/Tuples: Recursively processes each item
    - Pydantic models: Converts to dict first
    - Long strings: Truncates with ellipsis
    """
    # Handle None, numbers, booleans
    if value is None or isinstance(value, (int, float, bool)):
        return value

    # Handle bytes - show length instead of content
    if isinstance(value, bytes):
        return f"<bytes: length={len(value)}>"

    # Handle Pydantic models - convert to dict
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif hasattr(value, "dict"):
        value = value.dict()

    # Handle dicts - recursively process each value
    if isinstance(value, dict):
        dict_value = cast(dict[str, object], value)
        return {k: _redact_value(v) for k, v in dict_value.items()}

    # Handle lists - recursively process each item
    if isinstance(value, list):
        list_value = cast(list[object], value)
        return [_redact_value(item) for item in list_value]

    # Handle tuples - recursively process each item
    if isinstance(value, tuple):
        tuple_value = cast(tuple[object, ...], value)
        return tuple(_redact_value(item) for item in tuple_value)

    # Handle long strings - truncate with ellipsis
    if isinstance(value, str) and len(value) > 100:
        return f"{value[:50]}...{value[-50:]}" if len(value) > 100 else value

    # Return as-is for other types (str, etc.)
    return value


def _redact_context(context: dict[str, Any] | None) -> dict[str, Any]:
    """Redact sensitive fields in context dictionary.

    Checks field names against known sensitive patterns and redacts values.
    Also recursively processes all values to handle bytes, large strings, etc.
    """
    if not context:
        return {}

    redacted: dict[str, Any] = {}
    for key, value in context.items():
        key_lower = key.lower()
        # Check if any sensitive field is in the key name
        is_sensitive = any(sensitive in key_lower for sensitive in _SENSITIVE_FIELDS)

        if is_sensitive:
            # Fully redact sensitive fields
            redacted[key] = "***REDACTED***"
        else:
            # Process value to handle bytes, long strings, nested structures
            redacted[key] = _redact_value(value)

    return redacted


def _get_logger(logger_name: str) -> logging.Logger:
    """Get or create a logger with the given name."""
    return logging.getLogger(logger_name)


def log_debug(
    message: str,
    context: dict[str, Any] | None = None,
    logger_name: str = "tarash.tarash_gateway",
    redact: bool = False,
) -> None:
    """
    Log a debug message with optional context.

    Args:
        message: The log message
        context: Optional dictionary of context data
        logger_name: Name of the logger to use
        redact: If True, redact sensitive fields in context
    """
    logger = _get_logger(logger_name)

    if context:
        if redact:
            context = _redact_context(context)
        logger.debug(f"{message} | Context: {context}")
    else:
        logger.debug(message)


def log_info(
    message: str,
    context: dict[str, Any] | None = None,
    logger_name: str = "tarash.tarash_gateway",
    redact: bool = False,
) -> None:
    """
    Log an info message with optional context.

    Args:
        message: The log message
        context: Optional dictionary of context data
        logger_name: Name of the logger to use
        redact: If True, redact sensitive fields in context
    """
    logger = _get_logger(logger_name)

    if context:
        if redact:
            context = _redact_context(context)
        logger.info(f"{message} | Context: {context}")
    else:
        logger.info(message)


def log_warning(
    message: str,
    context: dict[str, Any] | None = None,
    logger_name: str = "tarash.tarash_gateway",
    redact: bool = False,
) -> None:
    """
    Log a warning message with optional context.

    Args:
        message: The log message
        context: Optional dictionary of context data
        logger_name: Name of the logger to use
        redact: If True, redact sensitive fields in context
    """
    logger = _get_logger(logger_name)

    if context:
        if redact:
            context = _redact_context(context)
        logger.warning(f"{message} | Context: {context}")
    else:
        logger.warning(message)


def log_error(
    message: str,
    context: dict[str, Any] | None = None,
    logger_name: str = "tarash.tarash_gateway",
    redact: bool = False,
    exc_info: bool = False,
) -> None:
    """
    Log an error message with optional context and exception info.

    Args:
        message: The log message
        context: Optional dictionary of context data
        logger_name: Name of the logger to use
        redact: If True, redact sensitive fields in context
        exc_info: If True, include exception traceback information
    """
    logger = _get_logger(logger_name)

    if context:
        if redact:
            context = _redact_context(context)
        logger.error(f"{message} | Context: {context}", exc_info=exc_info)
    else:
        logger.error(message, exc_info=exc_info)


@dataclass
class ProviderLogger:
    """Logger with pre-bound provider context.

    Reduces boilerplate by automatically including provider, model, and request_id
    in all log messages. Use `with_request_id()` to create a new logger with
    request context bound.

    Example:
        logger = ProviderLogger("fal", "minimax", "tarash.providers.fal")
        logger.debug("Creating client", {"base_url": "default"})

        # After getting request_id:
        logger = logger.with_request_id(request_id)
        logger.info("Request submitted")  # Includes request_id automatically
    """

    provider: str
    model: str
    logger_name: str
    request_id: str | None = None
    _base_context: dict[str, object] = field(
        default_factory=lambda: cast(dict[str, object], {}), repr=False
    )

    def __post_init__(self) -> None:
        """Build base context from provider/model/request_id."""
        self._base_context = {
            "provider": self.provider,
            "model": self.model,
        }
        if self.request_id is not None:
            self._base_context["request_id"] = self.request_id

    def _build_context(
        self, extra: dict[str, object] | None = None
    ) -> dict[str, object]:
        """Merge base context with extra fields."""
        if extra is None:
            return dict(self._base_context)
        return {**self._base_context, **extra}

    def with_request_id(self, request_id: str) -> "ProviderLogger":
        """Return a new logger with request_id bound to context."""
        return ProviderLogger(
            provider=self.provider,
            model=self.model,
            logger_name=self.logger_name,
            request_id=request_id,
        )

    def debug(
        self,
        message: str,
        extra: dict[str, object] | None = None,
        redact: bool = False,
    ) -> None:
        """Log a debug message with provider context."""
        log_debug(
            message,
            context=cast(dict[str, Any], self._build_context(extra)),
            logger_name=self.logger_name,
            redact=redact,
        )

    def info(
        self,
        message: str,
        extra: dict[str, object] | None = None,
        redact: bool = False,
    ) -> None:
        """Log an info message with provider context."""
        log_info(
            message,
            context=cast(dict[str, Any], self._build_context(extra)),
            logger_name=self.logger_name,
            redact=redact,
        )

    def warning(
        self,
        message: str,
        extra: dict[str, object] | None = None,
        redact: bool = False,
    ) -> None:
        """Log a warning message with provider context."""
        log_warning(
            message,
            context=cast(dict[str, Any], self._build_context(extra)),
            logger_name=self.logger_name,
            redact=redact,
        )

    def error(
        self,
        message: str,
        extra: dict[str, object] | None = None,
        redact: bool = False,
        exc_info: bool = False,
    ) -> None:
        """Log an error message with provider context."""
        log_error(
            message,
            context=cast(dict[str, Any], self._build_context(extra)),
            logger_name=self.logger_name,
            redact=redact,
            exc_info=exc_info,
        )
