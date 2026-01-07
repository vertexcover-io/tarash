"""Public API for video generation."""

from tarash.tarash_gateway.logging import log_debug, log_info
from tarash.tarash_gateway.models import (
    ProviderHandler,
    ProgressCallback,
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
)
from tarash.tarash_gateway.orchestrator import ExecutionOrchestrator
from tarash.tarash_gateway.providers.fal import FAL_MODEL_REGISTRY
from tarash.tarash_gateway.providers.field_mappers import FieldMapper
from tarash.tarash_gateway.providers.replicate import REPLICATE_MODEL_REGISTRY
from tarash.tarash_gateway.registry import register_provider as _register_provider

# ==================== Provider Registry ====================

# Singleton orchestrator instance
_ORCHESTRATOR = ExecutionOrchestrator()

# Field mapper registries for each provider (hardcoded built-in providers)
_FIELD_MAPPER_REGISTRIES: dict[str, dict[str, dict[str, FieldMapper]]] = {
    "fal": FAL_MODEL_REGISTRY,
    "replicate": REPLICATE_MODEL_REGISTRY,
}


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
    _register_provider(provider, handler)


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
        >>> from tarash.tarash_gateway.providers.field_mappers import (
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
        logger_name="tarash.tarash_gateway.api",
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

    Supports automatic fallback to alternative providers if configured.
    Supports mock mode for testing and development.

    Args:
        config: Provider configuration (may include fallback_configs and mock config)
        request: Video generation request
        on_progress: Optional callback (sync or async) for progress updates

    Returns:
        Final VideoGenerationResponse when complete (includes execution_metadata)

    Raises:
        TarashException: If generation fails on all providers in fallback chain
    """
    log_info(
        "Video generation request received (async)",
        context={
            "config": config,
            "request": request,
        },
        logger_name="tarash.tarash_gateway.api",
        redact=True,
    )

    # Delegate to orchestrator (handles mock, fallbacks, and execution)
    return await _ORCHESTRATOR.execute_async(config, request, on_progress=on_progress)


def generate_video(
    config: VideoGenerationConfig,
    request: VideoGenerationRequest,
    on_progress: ProgressCallback | None = None,
) -> VideoGenerationResponse:
    """
    Generate video synchronously (blocking) with progress callback.

    Supports automatic fallback to alternative providers if configured.
    Supports mock mode for testing and development.

    Args:
        config: Provider configuration (may include fallback_configs and mock config)
        request: Video generation request
        on_progress: Optional callback (sync or async) for progress updates

    Returns:
        Final VideoGenerationResponse when complete (includes execution_metadata)

    Raises:
        TarashException: If generation fails on all providers in fallback chain
    """
    log_info(
        "Video generation request received (sync)",
        context={
            "config": config,
            "request": request,
        },
        redact=True,
        logger_name="tarash.tarash_gateway.api",
    )

    # Delegate to orchestrator (handles mock, fallbacks, and execution)
    return _ORCHESTRATOR.execute_sync(config, request, on_progress=on_progress)
