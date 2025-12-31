"""Public API for video generation."""

# from typing import AsyncIterator

from tarash.tarash_gateway.logging import log_debug, log_error, log_info
from tarash.tarash_gateway.video.exceptions import ValidationError
from tarash.tarash_gateway.video.models import (
    ProgressCallback,
    ProviderHandler,
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
)
from tarash.tarash_gateway.video.providers import (
    OpenAIProviderHandler,
    FalProviderHandler,
    Veo3ProviderHandler,
    ReplicateProviderHandler,
)

# Replicate imports are conditional due to pydantic v1 compatibility issues with Python 3.14+
# ==================== Provider Registry ====================

# Singleton instances of handlers (stateless)
_HANDLER_INSTANCES: dict[str, ProviderHandler] = {}


def _get_handler(provider: str) -> ProviderHandler:
    """Get or create handler instance for provider."""
    if provider not in _HANDLER_INSTANCES:
        log_debug(
            "Selected provider",
            context={"provider": provider},
            logger_name="tarash.tarash_gateway.video.api",
        )
        if provider == "fal":
            _HANDLER_INSTANCES[provider] = FalProviderHandler()
        elif provider == "veo3":
            _HANDLER_INSTANCES[provider] = Veo3ProviderHandler()
        elif provider == "replicate":
            _HANDLER_INSTANCES[provider] = ReplicateProviderHandler()
        elif provider == "openai":
            _HANDLER_INSTANCES[provider] = OpenAIProviderHandler()
        else:
            log_error(
                "Unsupported provider",
                context={"provider": provider},
                logger_name="tarash.tarash_gateway.video.api",
            )
            raise ValidationError(
                f"Unsupported provider: {provider}",
                provider=provider,
            )

    return _HANDLER_INSTANCES[provider]


# ==================== Public API ====================


async def generate_video_async(
    config: VideoGenerationConfig,
    request: VideoGenerationRequest,
    on_progress: ProgressCallback | None = None,
) -> VideoGenerationResponse:
    """
    Generate video asynchronously with progress callback (sync or async).

    Args:
        config: Provider configuration
        request: Video generation request
        on_progress: Optional callback (sync or async) for progress updates

    Returns:
        Final VideoGenerationResponse when complete

    Raises:
        TarashException: If generation fails
    """
    log_info(
        "Video generation request received (async)",
        context={
            "config": config,
            "request": request,
        },
        logger_name="tarash.tarash_gateway.video.api",
        redact=True,
    )

    # Get handler for provider
    handler = _get_handler(config.provider)

    # Generate using handler with async callback support
    # Errors are logged by the handle_video_generation_errors decorator
    response = await handler.generate_video_async(
        config, request, on_progress=on_progress
    )
    return response


def generate_video(
    config: VideoGenerationConfig,
    request: VideoGenerationRequest,
    on_progress: ProgressCallback | None = None,
) -> VideoGenerationResponse:
    """
    Generate video synchronously (blocking) with progress callback.

    Args:
        config: Provider configuration
        request: Video generation request
        on_progress: Optional callback (sync or async) for progress updates

    Returns:
        Final VideoGenerationResponse when complete

    Raises:
        TarashException: If generation fails
    """
    log_info(
        "Video generation request received (sync)",
        context={
            "config": config,
            "request": request,
        },
        redact=True,
        logger_name="tarash.tarash_gateway.video.api",
    )

    # Get handler for provider
    handler = _get_handler(config.provider)

    # Generate using handler with callback support
    # Errors are logged by the handle_video_generation_errors decorator
    response = handler.generate_video(config, request, on_progress=on_progress)
    return response
