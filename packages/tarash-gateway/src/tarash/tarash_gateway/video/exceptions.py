"""Exceptions for video generation."""

import functools
import inspect
import traceback
from typing import TYPE_CHECKING, Any, Callable

from pydantic import ValidationError as PydanticValidationError

from tarash.tarash_gateway.logging import log_error

if TYPE_CHECKING:
    from tarash.tarash_gateway.video.models import (
        VideoGenerationConfig,
        VideoGenerationRequest,
    )


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
            self,
            config: "VideoGenerationConfig",
            request: "VideoGenerationRequest",
            *args,
            **kwargs,
        ):
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
            self,
            config: "VideoGenerationConfig",
            request: "VideoGenerationRequest",
            *args,
            **kwargs,
        ):
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
