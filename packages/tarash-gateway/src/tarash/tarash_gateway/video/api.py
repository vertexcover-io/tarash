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
    RunwayProviderHandler,
)
from tarash.tarash_gateway.video.providers.field_mappers import FieldMapper
from tarash.tarash_gateway.video.providers.fal import FAL_MODEL_REGISTRY
from tarash.tarash_gateway.video.providers.replicate import REPLICATE_MODEL_REGISTRY

# Replicate imports are conditional due to pydantic v1 compatibility issues with Python 3.14+
# ==================== Provider Registry ====================

# Singleton instances of handlers (stateless)
_HANDLER_INSTANCES: dict[str, ProviderHandler] = {}

# Field mapper registries for each provider (hardcoded built-in providers)
_FIELD_MAPPER_REGISTRIES: dict[str, dict[str, dict[str, FieldMapper]]] = {
    "fal": FAL_MODEL_REGISTRY,
    "replicate": REPLICATE_MODEL_REGISTRY,
}


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
        elif provider == "runway":
            _HANDLER_INSTANCES[provider] = RunwayProviderHandler()
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


def register_provider(
    provider: str,
    handler: ProviderHandler,
) -> None:
    """Register a custom provider handler.

    This allows extending the video generation API with custom providers
    without modifying the core library code.

    Args:
        provider: Provider name (e.g., "custom-provider")
        handler: Instance of ProviderHandler implementing the provider logic

    Examples:
        >>> class MyCustomHandler(ProviderHandler):
        ...     async def generate_video_async(self, config, request, on_progress=None):
        ...         # Custom implementation
        ...         pass
        ...     def generate_video(self, config, request, on_progress=None):
        ...         # Custom implementation
        ...         pass
        >>> register_provider("my-provider", MyCustomHandler())
    """
    if provider in _HANDLER_INSTANCES:
        log_info(
            f"Overwriting existing provider handler: {provider}",
            context={"provider": provider},
            logger_name="tarash.tarash_gateway.video.api",
        )

    _HANDLER_INSTANCES[provider] = handler
    log_debug(
        "Registered custom provider handler",
        context={"provider": provider},
        logger_name="tarash.tarash_gateway.video.api",
    )


def register_provider_field_mapping(
    provider: str,
    model_mappings: dict[str, dict[str, FieldMapper]],
) -> None:
    """Register field mappings for a provider's models.

    This allows configuring how VideoGenerationRequest fields are mapped
    to provider-specific API formats for different models.

    Args:
        provider: Provider name (e.g., "fal", "replicate", "custom-provider")
        model_mappings: Dict mapping model names/prefixes to their field mappers

    Examples:
        >>> from tarash.tarash_gateway.video.providers.field_mappers import (
        ...     passthrough_field_mapper,
        ...     duration_field_mapper,
        ... )
        >>> register_provider_field_mapping("my-provider", {
        ...     "my-provider/model-1": {
        ...         "prompt": passthrough_field_mapper("prompt", required=True),
        ...         "duration": duration_field_mapper(field_type="int"),
        ...     },
        ...     "my-provider/model-2": {
        ...         "prompt": passthrough_field_mapper("prompt", required=True),
        ...     },
        ... })
    """
    _FIELD_MAPPER_REGISTRIES[provider] = model_mappings
    log_debug(
        "Registered field mappings for provider",
        context={
            "provider": provider,
            "num_models": len(model_mappings),
        },
        logger_name="tarash.tarash_gateway.video.api",
    )


def get_provider_field_mapping(
    provider: str,
) -> dict[str, dict[str, FieldMapper]] | None:
    """Get the field mapper registry for a provider.

    Args:
        provider: Provider name (e.g., "fal", "replicate")

    Returns:
        Dict mapping model names to field mappers, or None if not registered

    Examples:
        >>> registry = get_provider_field_mapping("fal")
        >>> if registry:
        ...     model_mappers = registry.get("fal-ai/minimax")
    """
    return _FIELD_MAPPER_REGISTRIES.get(provider)


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

    # INTERCEPTION: Check for mock mode
    if config.mock and config.mock.enabled:
        from tarash.tarash_gateway.video.mock import handle_mock_request_async

        return await handle_mock_request_async(config.mock, request, on_progress)

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
