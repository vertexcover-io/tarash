"""Exceptions for video generation."""

import functools
import inspect
import traceback
from typing import TYPE_CHECKING, Any, Callable

from pydantic import ValidationError as PydanticValidationError

if TYPE_CHECKING:
    from tarash.tarash_gateway.video.models import (
        VideoGenerationConfig,
        VideoGenerationRequest,
    )


class VideoGenerationError(Exception):
    """Base exception for video generation errors."""

    def __init__(
        self,
        message: str,
        provider: str,
        raw_response: dict[str, Any] | None = None,
        request_id: str | None = None,
        model: str | None = None,
    ):
        self.message = message
        self.provider = provider
        self.raw_response = raw_response
        self.request_id = request_id
        self.model = model
        super().__init__(message)


class ProviderAPIError(VideoGenerationError):
    """Provider API returned an error."""

    pass


class ValidationError(VideoGenerationError):
    """Request validation failed."""

    pass


def handle_video_generation_errors(func: Callable) -> Callable:
    """Decorator to handle only truly unhandled exceptions.

    - ValidationError (both custom and Pydantic): Let propagate (don't wrap)
    - VideoGenerationError: Re-raise as-is (ensuring model is set)
    - Unknown exceptions: Wrap in VideoGenerationError
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
                ValidationError,
                PydanticValidationError,
                ProviderAPIError,
                VideoGenerationError,
            ):
                # Let validation and provider API errors propagate
                raise
            except Exception as ex:
                # Only wrap truly unknown exceptions
                raise VideoGenerationError(
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
                ValidationError,
                PydanticValidationError,
                ProviderAPIError,
                VideoGenerationError,
            ):
                # Let validation and provider API errors propagate
                raise
            except Exception as ex:
                # Only wrap truly unknown exceptions
                raise VideoGenerationError(
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
