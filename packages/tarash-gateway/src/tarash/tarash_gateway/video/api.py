"""Public API for video generation."""

# from typing import AsyncIterator

from tarash.tarash_gateway.video.exceptions import VideoGenerationError
from tarash.tarash_gateway.video.models import (
    ProgressCallback,
    ProviderHandler,
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
)
from tarash.tarash_gateway.video.providers import OpenAIProviderHandler
from tarash.tarash_gateway.video.providers.fal import FalProviderHandler
from tarash.tarash_gateway.video.providers.veo3 import Veo3ProviderHandler

# Replicate imports are conditional due to pydantic v1 compatibility issues with Python 3.14+
try:
    from tarash.tarash_gateway.video.providers.replicate import ReplicateProviderHandler

    _REPLICATE_AVAILABLE = True
except Exception:
    ReplicateProviderHandler = None  # type: ignore
    _REPLICATE_AVAILABLE = False

# ==================== Provider Registry ====================

# Singleton instances of handlers (stateless)
_HANDLER_INSTANCES: dict[str, ProviderHandler] = {}


def _get_handler(provider: str) -> ProviderHandler:
    """Get or create handler instance for provider."""
    if provider not in _HANDLER_INSTANCES:
        if provider == "fal":
            _HANDLER_INSTANCES[provider] = FalProviderHandler()
        elif provider == "veo3":
            _HANDLER_INSTANCES[provider] = Veo3ProviderHandler()
        elif provider == "replicate":
            if not _REPLICATE_AVAILABLE or ReplicateProviderHandler is None:
                raise VideoGenerationError(
                    "Replicate provider is not available. This may be due to "
                    "pydantic v1 incompatibility with Python 3.14+. "
                    "Install with: pip install tarash-gateway[replicate]",
                    provider=provider,
                )
            _HANDLER_INSTANCES[provider] = ReplicateProviderHandler()
        elif provider == "openai":
            _HANDLER_INSTANCES[provider] = OpenAIProviderHandler()
        else:
            raise VideoGenerationError(
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
        VideoGenerationError: If generation fails
    """
    # Get handler for provider
    handler = _get_handler(config.provider)

    # Generate using handler with async callback support
    return await handler.generate_video_async(config, request, on_progress=on_progress)


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
        VideoGenerationError: If generation fails
    """
    # Get handler for provider
    handler = _get_handler(config.provider)

    # Generate using handler with callback support
    return handler.generate_video(config, request, on_progress=on_progress)
