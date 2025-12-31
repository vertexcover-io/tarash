"""Logging utilities for tarash-gateway."""

import logging
from typing import Any

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


def _sanitize_value(value: Any) -> Any:
    """Sanitize a value by replacing sensitive data with redacted markers."""
    if isinstance(value, dict):
        return {k: _sanitize_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    elif isinstance(value, str) and len(value) > 20:
        # For long strings (likely API keys or tokens), redact
        return f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "***"
    return value


def _sanitize_context(context: dict[str, Any] | None) -> dict[str, Any]:
    """Sanitize context dictionary by redacting sensitive fields."""
    if not context:
        return {}

    sanitized = {}
    for key, value in context.items():
        key_lower = key.lower()
        # Check if any sensitive field is in the key name
        is_sensitive = any(sensitive in key_lower for sensitive in _SENSITIVE_FIELDS)

        if is_sensitive:
            if isinstance(value, str):
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, dict):
                sanitized[key] = _sanitize_value(value)
            else:
                sanitized[key] = "***REDACTED***"
        else:
            sanitized[key] = _sanitize_value(value)

    return sanitized


def _get_logger(logger_name: str) -> logging.Logger:
    """Get or create a logger with the given name."""
    return logging.getLogger(logger_name)


def log_debug(
    message: str,
    context: dict[str, Any] | None = None,
    logger_name: str = "tarash.tarash_gateway",
    sanitize: bool = False,
) -> None:
    """
    Log a debug message with optional context.

    Args:
        message: The log message
        context: Optional dictionary of context data
        logger_name: Name of the logger to use
        sanitize: If True, sanitize sensitive fields in context
    """
    logger = _get_logger(logger_name)

    if context:
        if sanitize:
            context = _sanitize_context(context)
        logger.debug(f"{message} | Context: {context}")
    else:
        logger.debug(message)


def log_info(
    message: str,
    context: dict[str, Any] | None = None,
    logger_name: str = "tarash.tarash_gateway",
    sanitize: bool = False,
) -> None:
    """
    Log an info message with optional context.

    Args:
        message: The log message
        context: Optional dictionary of context data
        logger_name: Name of the logger to use
        sanitize: If True, sanitize sensitive fields in context
    """
    logger = _get_logger(logger_name)

    if context:
        if sanitize:
            context = _sanitize_context(context)
        logger.info(f"{message} | Context: {context}")
    else:
        logger.info(message)


def log_warning(
    message: str,
    context: dict[str, Any] | None = None,
    logger_name: str = "tarash.tarash_gateway",
    sanitize: bool = False,
) -> None:
    """
    Log a warning message with optional context.

    Args:
        message: The log message
        context: Optional dictionary of context data
        logger_name: Name of the logger to use
        sanitize: If True, sanitize sensitive fields in context
    """
    logger = _get_logger(logger_name)

    if context:
        if sanitize:
            context = _sanitize_context(context)
        logger.warning(f"{message} | Context: {context}")
    else:
        logger.warning(message)


def log_error(
    message: str,
    context: dict[str, Any] | None = None,
    logger_name: str = "tarash.tarash_gateway",
    sanitize: bool = False,
    exc_info: bool = False,
) -> None:
    """
    Log an error message with optional context and exception info.

    Args:
        message: The log message
        context: Optional dictionary of context data
        logger_name: Name of the logger to use
        sanitize: If True, sanitize sensitive fields in context
        exc_info: If True, include exception traceback information
    """
    logger = _get_logger(logger_name)

    if context:
        if sanitize:
            context = _sanitize_context(context)
        logger.error(f"{message} | Context: {context}", exc_info=exc_info)
    else:
        logger.error(message, exc_info=exc_info)
