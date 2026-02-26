"""Provider handler registry and handler resolution."""

from typing import cast

from tarash.tarash_gateway.logging import log_debug, log_error, log_info
from tarash.tarash_gateway.exceptions import ValidationError
from tarash.tarash_gateway.models import (
    ProviderHandler,
    VideoGenerationConfig,
)
from tarash.tarash_gateway.providers import (
    FalProviderHandler,
    GoogleProviderHandler,
    OpenAIProviderHandler,
    ReplicateProviderHandler,
    RunwayProviderHandler,
    StabilityProviderHandler,
    XaiProviderHandler,
)

# Singleton instances of handlers (stateless)
_HANDLER_INSTANCES: dict[str, ProviderHandler] = {}


def get_handler(config: VideoGenerationConfig) -> ProviderHandler:
    """Get or create handler instance for the given config.

    Args:
        config: Video generation configuration

    Returns:
        Provider handler instance (MockProviderHandler if mock enabled, else provider handler)

    Raises:
        ValidationError: If provider is not supported
    """
    # Check mock first
    if config.mock and config.mock.enabled:
        from tarash.tarash_gateway.mock import MockProviderHandler

        log_info(
            "Using mock handler",
            context={"mock_enabled": True},
            logger_name="tarash.tarash_gateway.registry",
        )
        return MockProviderHandler()

    provider = config.provider

    if provider not in _HANDLER_INSTANCES:
        log_debug(
            "Selected provider",
            context={"provider": provider},
            logger_name="tarash.tarash_gateway.registry",
        )
        if provider == "fal":
            _HANDLER_INSTANCES[provider] = cast(ProviderHandler, FalProviderHandler())
        elif provider == "replicate":
            _HANDLER_INSTANCES[provider] = cast(
                ProviderHandler, ReplicateProviderHandler()
            )
        elif provider == "openai":
            _HANDLER_INSTANCES[provider] = cast(
                ProviderHandler, OpenAIProviderHandler()
            )
        elif provider == "runway":
            _HANDLER_INSTANCES[provider] = cast(
                ProviderHandler, RunwayProviderHandler()
            )
        elif provider == "stability":
            _HANDLER_INSTANCES[provider] = cast(
                ProviderHandler, StabilityProviderHandler()
            )
        elif provider == "google":
            _HANDLER_INSTANCES[provider] = cast(
                ProviderHandler, GoogleProviderHandler()
            )
        elif provider == "xai":
            _HANDLER_INSTANCES[provider] = cast(ProviderHandler, XaiProviderHandler())
        else:
            log_error(
                "Unsupported provider",
                context={"provider": provider},
                logger_name="tarash.tarash_gateway.registry",
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
            logger_name="tarash.tarash_gateway.registry",
        )

    _HANDLER_INSTANCES[provider] = handler
    log_debug(
        "Registered custom provider handler",
        context={"provider": provider},
        logger_name="tarash.tarash_gateway.registry",
    )
