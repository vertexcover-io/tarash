"""Exceptions for video generation."""

import functools
import inspect
import traceback
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from pydantic import ValidationError as PydanticValidationError

from tarash.tarash_gateway.logging import log_error

if TYPE_CHECKING:
    from tarash.tarash_gateway.video.models import (
        VideoGenerationConfig,
        VideoGenerationRequest,
        VideoGenerationResponse,
    )

F = TypeVar("F", bound=Callable[..., Any])


class TarashException(Exception):
    """Base exception for all Tarash video generation errors."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        model: str | None = None,
        request_id: str | None = None,
        raw_response: dict[str, Any] | None = None,
    ):
        self.message = message
        self.provider = provider
        self.model = model
        self.request_id = request_id
        self.raw_response = raw_response
        super().__init__(message)


class ValidationError(TarashException):
    """Input validation failed (client error, 400-level)."""

    pass


class ContentModerationError(TarashException):
    """Content violates provider's content policy."""

    pass


class HTTPError(TarashException):
    """HTTP-level error from provider API."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        model: str | None = None,
        request_id: str | None = None,
        raw_response: dict[str, Any] | None = None,
        status_code: int | None = None,
    ):
        super().__init__(message, provider, model, request_id, raw_response)
        self.status_code = status_code


class GenerationFailedError(TarashException):
    """Video generation failed on provider side (includes timeouts, cancellations)."""

    pass


class HTTPConnectionError(TarashException):
    """HTTP connection error (network failure, DNS resolution, etc.)."""

    pass


class TimeoutError(TarashException):
    """Request timed out."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        model: str | None = None,
        request_id: str | None = None,
        raw_response: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ):
        super().__init__(message, provider, model, request_id, raw_response)
        self.timeout_seconds = timeout_seconds


def is_retryable_error(error: Exception) -> bool:
    """Determine if an error should trigger a fallback retry.

    Retryable errors (should try fallback):
    - GenerationFailedError: Video generation failed on provider side
    - TimeoutError: Request timed out
    - HTTPConnectionError: Network/connection failure
    - HTTPError with codes: 429 (rate limit), 500, 502, 503, 504 (server errors)

    Non-retryable errors (should NOT try fallback):
    - ValidationError: Input validation failed (client error)
    - ContentModerationError: Content policy violation
    - HTTPError with codes: 400, 401, 403, 404 (client errors)
    - Other unknown exceptions

    Args:
        error: Exception to classify

    Returns:
        True if error should trigger fallback, False otherwise
    """
    # Retryable: Generation failures, timeouts, connection errors
    if isinstance(error, (GenerationFailedError, TimeoutError, HTTPConnectionError)):
        return True

    # Retryable: HTTP errors with specific status codes
    if isinstance(error, HTTPError):
        if error.status_code in (429, 500, 502, 503, 504):
            return True
        return False

    # Non-retryable: Validation and content moderation errors
    if isinstance(error, (ValidationError, ContentModerationError)):
        return False

    # Non-retryable: Unknown errors
    return False


def handle_video_generation_errors(func: Callable) -> Callable:
    """Decorator to handle only truly unhandled exceptions.

    - ValidationError, ContentModerationError, HTTPError, GenerationFailedError: Let propagate
    - PydanticValidationError: Let propagate (Pydantic's validation errors)
    - TarashException: Let propagate (our base exception)
    - Unknown exceptions: Wrap in TarashException
    """
    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(
            self: Any,
            config: "VideoGenerationConfig",
            request: "VideoGenerationRequest",
            *args: Any,
            **kwargs: Any,
        ) -> "VideoGenerationResponse":
            try:
                return await func(self, config, request, *args, **kwargs)
            except (
                PydanticValidationError,
                TarashException,
            ):
                # Let all Tarash exceptions and Pydantic validation errors propagate
                raise
            except Exception as ex:
                # Only wrap truly unknown exceptions
                log_error(
                    f"Unknown error while generating video: {ex}",
                    context={
                        "provider": config.provider,
                        "model": config.model,
                    },
                    logger_name="tarash.tarash_gateway.video.exceptions",
                    exc_info=True,
                )
                raise TarashException(
                    f"Unknown error while generating video: {ex}",
                    provider=config.provider,
                    model=config.model,
                    raw_response={
                        "error": str(ex),
                        "error_type": type(ex).__name__,
                        "traceback": traceback.format_exc(),
                    },
                ) from ex

        return async_wrapper
    else:

        @functools.wraps(func)
        def sync_wrapper(
            self: Any,
            config: "VideoGenerationConfig",
            request: "VideoGenerationRequest",
            *args: Any,
            **kwargs: Any,
        ) -> "VideoGenerationResponse":
            try:
                return func(self, config, request, *args, **kwargs)
            except (
                PydanticValidationError,
                TarashException,
            ):
                # Let all Tarash exceptions and Pydantic validation errors propagate
                raise
            except Exception as ex:
                # Only wrap truly unknown exceptions
                log_error(
                    f"Unknown error while generating video: {ex}",
                    context={
                        "provider": config.provider,
                        "model": config.model,
                    },
                    logger_name="tarash.tarash_gateway.video.exceptions",
                    exc_info=True,
                )
                raise TarashException(
                    f"Unknown error while generating video: {ex}",
                    provider=config.provider,
                    model=config.model,
                    raw_response={
                        "error": str(ex),
                        "error_type": type(ex).__name__,
                        "traceback": traceback.format_exc(),
                    },
                ) from ex

        return sync_wrapper
