import functools
import inspect
import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar, cast

from pydantic import ValidationError as PydanticValidationError

from tarash.tarash_gateway.logging import log_error

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from tarash.tarash_gateway.models import (
        VideoGenerationConfig,
        VideoGenerationRequest,
        VideoGenerationResponse,
        ProviderHandler,
        ProgressCallback,
    )

# Type alias needed at runtime for class attributes
AnyDict = dict[str, Any]

F = TypeVar(
    "F",
    bound=Callable[..., "VideoGenerationResponse | Awaitable[VideoGenerationResponse]"],
)


class TarashException(Exception):
    """Base class for all exceptions raised by the Tarash SDK.

    Carries structured context (provider, model, request ID, raw provider
    response) to make debugging straightforward. Catch this class to handle
    any SDK error, or catch subclasses for more granular handling.

    Attributes:
        message: Human-readable error description.
        provider: Provider identifier (e.g. ``"fal"``, ``"runway"``).
        model: Model name at the time of the error.
        request_id: Provider-assigned or Tarash-assigned request ID.
        raw_response: Unmodified provider response payload, if available.
    """

    message: str
    provider: str | None
    model: str | None
    request_id: str | None
    raw_response: AnyDict | None

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        model: str | None = None,
        request_id: str | None = None,
        raw_response: AnyDict | None = None,
    ):
        """Initialise a TarashException with structured context.

        Args:
            message: Human-readable description of the error.
            provider: Provider identifier (e.g. ``"fal"``).
            model: Model name used when the error occurred.
            request_id: Provider or Tarash request ID for tracing.
            raw_response: Raw provider response payload for debugging.
        """
        self.message = message
        self.provider = provider
        self.model = model
        self.request_id = request_id
        self.raw_response = raw_response
        super().__init__(message)


class ValidationError(TarashException):
    """Raised when request parameters fail validation before or during generation.

    Typically corresponds to a 400-level response from the provider. This error
    is **not retryable** — the same request will fail on every provider.
    """

    pass


class ContentModerationError(TarashException):
    """Raised when the prompt or input violates a provider's content policy.

    Typically corresponds to a 403 response. This error is **not retryable**.
    """

    pass


class HTTPError(TarashException):
    """Raised on an unexpected HTTP error response from the provider API.

    Retryable for server-side codes (429, 500, 502, 503, 504).
    Not retryable for client-side codes (400, 401, 403, 404).

    Attributes:
        status_code: HTTP status code returned by the provider, if available.
    """

    status_code: int | None

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        model: str | None = None,
        request_id: str | None = None,
        raw_response: dict[str, object] | None = None,
        status_code: int | None = None,
    ):
        """Initialise an HTTPError with an optional status code.

        Args:
            message: Human-readable description of the HTTP error.
            provider: Provider identifier.
            model: Model name used when the error occurred.
            request_id: Request ID for tracing.
            raw_response: Raw provider response payload.
            status_code: HTTP status code (e.g. 429, 500).
        """
        super().__init__(message, provider, model, request_id, raw_response)
        self.status_code = status_code


class GenerationFailedError(TarashException):
    """Raised when the provider reports that generation failed.

    Covers provider-side failures including internal errors, content
    timeouts within the provider's pipeline, and explicit cancellations.
    This error **is retryable** — a fallback provider may succeed.
    """

    pass


class HTTPConnectionError(TarashException):
    """Raised on a network-level failure before the provider responds.

    Covers DNS resolution failures, connection refused, and other transport
    errors. This error **is retryable** — a fallback provider may be reachable.
    """

    pass


class TimeoutError(TarashException):
    """Raised when a request exceeds the configured ``timeout`` seconds.

    This error **is retryable** — a fallback provider may respond faster.

    Attributes:
        timeout_seconds: The timeout value that was exceeded, in seconds.
    """

    timeout_seconds: float | None

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        model: str | None = None,
        request_id: str | None = None,
        raw_response: dict[str, object] | None = None,
        timeout_seconds: float | None = None,
    ):
        """Initialise a TimeoutError with the timeout duration.

        Args:
            message: Human-readable description.
            provider: Provider identifier.
            model: Model name used when the timeout occurred.
            request_id: Request ID for tracing.
            raw_response: Raw provider response payload, if any.
            timeout_seconds: The timeout threshold that was exceeded.
        """
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


def handle_video_generation_errors(func: F) -> F:
    """Decorator that wraps unhandled exceptions in ``TarashException``.

    Apply to provider ``generate_video`` and ``generate_video_async`` methods.
    Works with both sync and async functions automatically.

    Behaviour:
    - ``TarashException`` subclasses — propagate unchanged.
    - ``PydanticValidationError`` — propagate unchanged.
    - Any other exception — wrapped in ``TarashException`` with full traceback
      captured in ``raw_response`` and logged at ERROR level.

    Example:
        ```python
        class MyProvider:
            @handle_video_generation_errors
            async def generate_video_async(self, config, request, on_progress=None):
                ...

            @handle_video_generation_errors
            def generate_video(self, config, request, on_progress=None):
                ...
        ```
    """
    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(
            self: "ProviderHandler",
            config: "VideoGenerationConfig",
            request: "VideoGenerationRequest",
            on_progress: "ProgressCallback | None" = None,
        ) -> "VideoGenerationResponse":
            try:
                return await func(self, config, request, on_progress)  # pyright: ignore[reportAny]
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
                    logger_name="tarash.tarash_gateway.exceptions",
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

        return cast(F, async_wrapper)
    else:

        @functools.wraps(func)
        def sync_wrapper(
            self: "ProviderHandler",
            config: "VideoGenerationConfig",
            request: "VideoGenerationRequest",
            on_progress: "ProgressCallback | None" = None,
        ) -> "VideoGenerationResponse":
            try:
                return func(self, config, request, on_progress)  # pyright: ignore[reportReturnType]
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
                    logger_name="tarash.tarash_gateway.exceptions",
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

        return cast(F, sync_wrapper)
