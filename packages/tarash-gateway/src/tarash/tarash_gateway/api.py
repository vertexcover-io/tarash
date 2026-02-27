"""Public API for video and image generation."""

from tarash.tarash_gateway.logging import log_debug, log_info
from tarash.tarash_gateway.models import (
    AudioGenerationConfig,
    ImageGenerationConfig,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageProgressCallback,
    ProviderHandler,
    ProgressCallback,
    STSProgressCallback,
    STSRequest,
    STSResponse,
    TTSProgressCallback,
    TTSRequest,
    TTSResponse,
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

    Extends the SDK with a new provider at runtime without modifying library
    code. The registered handler will be used for all subsequent calls that
    specify this provider name in their config.

    Args:
        provider: Unique provider identifier (e.g. ``"my-provider"``). Must
            match the ``provider`` field passed in [VideoGenerationConfig][].
        handler: Instantiated handler implementing the [ProviderHandler][]
            protocol.

    Example:
        ```python
        from tarash.tarash_gateway import register_provider
        from tarash.tarash_gateway.models import (
            ProviderHandler,
            VideoGenerationConfig,
            VideoGenerationRequest,
            VideoGenerationResponse,
        )

        class MyHandler:
            def generate_video(self, config, request, on_progress=None):
                ...
            async def generate_video_async(self, config, request, on_progress=None):
                ...

        register_provider("my-provider", MyHandler())

        config = VideoGenerationConfig(provider="my-provider", model="my-model", api_key="...")
        ```
    """
    _register_provider(provider, handler)


def register_provider_field_mapping(
    provider: str,
    model_mappings: dict[str, dict[str, FieldMapper]],
) -> None:
    """Register field mappings for a provider's models.

    Configures how [VideoGenerationRequest][] fields are translated to the
    provider's own API parameter names and formats. Supports prefix-based
    lookup so a single mapping can cover a family of model variants.

    Args:
        provider: Provider identifier (e.g. ``"fal"``, ``"my-provider"``).
        model_mappings: Mapping from model name (or prefix) to a dict of
            ``{api_param_name: FieldMapper}``. Prefix matching is supported —
            the longest registered prefix wins.

    Example:
        ```python
        from tarash.tarash_gateway import register_provider_field_mapping
        from tarash.tarash_gateway.providers.field_mappers import (
            duration_field_mapper,
            passthrough_field_mapper,
        )

        register_provider_field_mapping("my-provider", {
            "my-provider/model-v1": {
                "prompt": passthrough_field_mapper("prompt", required=True),
                "num_seconds": duration_field_mapper(field_type="int"),
            },
        })
        ```
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
    """Return the field mapper registry for a provider.

    Args:
        provider: Provider identifier (e.g. ``"fal"``, ``"replicate"``).

    Returns:
        A mapping of model name/prefix → field mapper dict, or ``None`` if
        no mapping has been registered for the provider.

    Example:
        ```python
        from tarash.tarash_gateway import get_provider_field_mapping

        registry = get_provider_field_mapping("fal")
        if registry:
            mappers = registry.get("fal-ai/minimax")
        ```
    """
    return _FIELD_MAPPER_REGISTRIES.get(provider)


# ==================== Public API ====================


async def generate_video_async(
    config: VideoGenerationConfig,
    request: VideoGenerationRequest,
    on_progress: ProgressCallback | None = None,
) -> VideoGenerationResponse:
    """Generate a video asynchronously using the configured provider.

    Runs the full orchestration pipeline: applies field mappings, submits the
    request to the provider, polls for completion, and returns a unified
    response. Automatically falls back to ``config.fallback_configs`` on
    retryable errors.

    Args:
        config: Provider configuration including API key, model, timeout, and
            optional fallback chain.
        request: Video generation parameters (prompt, duration, aspect ratio,
            etc.). Unknown fields are captured into ``extra_params``.
        on_progress: Optional callback invoked on each polling cycle with a
            [VideoGenerationUpdate][]. Accepts both sync and async callables.

    Returns:
        [VideoGenerationResponse][] with the video URL, status, and full
        ``execution_metadata`` including timing and fallback attempts.

    Raises:
        TarashException: If generation fails on all providers in the fallback
            chain.

    Example:
        ```python
        import asyncio
        from tarash.tarash_gateway import generate_video_async
        from tarash.tarash_gateway.models import (
            VideoGenerationConfig,
            VideoGenerationRequest,
            VideoGenerationUpdate,
        )

        async def main():
            config = VideoGenerationConfig(
                provider="fal",
                model="fal-ai/veo3",
                api_key="FAL_KEY",
            )
            request = VideoGenerationRequest(
                prompt="A cat playing piano, cinematic lighting",
                duration_seconds=4,
                aspect_ratio="16:9",
            )

            def on_progress(update: VideoGenerationUpdate) -> None:
                print(f"{update.status} — {update.progress_percent}%")

            response = await generate_video_async(config, request, on_progress)
            print(response.video)

        asyncio.run(main())
        ```
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
    """Generate a video synchronously using the configured provider.

    Blocking version of [generate_video_async][]. Runs the full orchestration
    pipeline and returns only when generation is complete. Automatically falls
    back to ``config.fallback_configs`` on retryable errors.

    Args:
        config: Provider configuration including API key, model, timeout, and
            optional fallback chain.
        request: Video generation parameters (prompt, duration, aspect ratio,
            etc.). Unknown fields are captured into ``extra_params``.
        on_progress: Optional callback invoked on each polling cycle with a
            [VideoGenerationUpdate][]. Accepts both sync and async callables.

    Returns:
        [VideoGenerationResponse][] with the video URL, status, and full
        ``execution_metadata`` including timing and fallback attempts.

    Raises:
        TarashException: If generation fails on all providers in the fallback
            chain.

    Example:
        ```python
        from tarash.tarash_gateway import generate_video
        from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

        config = VideoGenerationConfig(
            provider="fal",
            model="fal-ai/veo3",
            api_key="FAL_KEY",
        )
        request = VideoGenerationRequest(
            prompt="A cat playing piano, cinematic lighting",
            duration_seconds=4,
            aspect_ratio="16:9",
        )
        response = generate_video(config, request)
        print(response.video)
        ```
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


# ==================== Image Generation API ====================


async def generate_image_async(
    config: ImageGenerationConfig,
    request: ImageGenerationRequest,
    on_progress: ImageProgressCallback | None = None,
) -> ImageGenerationResponse:
    """Generate an image asynchronously using the configured provider.

    Runs the full orchestration pipeline for image generation and returns a
    unified response. Automatically falls back to ``config.fallback_configs``
    on retryable errors.

    Args:
        config: Provider configuration including API key, model, timeout, and
            optional fallback chain.
        request: Image generation parameters (prompt, size, quality, style,
            etc.). Unknown fields are captured into ``extra_params``.
        on_progress: Optional callback invoked with an [ImageGenerationUpdate][]
            during generation. Accepts both sync and async callables.

    Returns:
        [ImageGenerationResponse][] containing a list of generated images
        (base64 or URL), status, and ``execution_metadata``.

    Raises:
        TarashException: If generation fails on all providers in the fallback
            chain.
        NotImplementedError: If the configured provider does not support image
            generation.

    Example:
        ```python
        import asyncio
        from tarash.tarash_gateway import generate_image_async
        from tarash.tarash_gateway.models import ImageGenerationConfig, ImageGenerationRequest

        async def main():
            config = ImageGenerationConfig(provider="openai", api_key="OPENAI_KEY")
            request = ImageGenerationRequest(
                prompt="A futuristic cityscape at dusk, photorealistic",
                size="1024x1024",
                quality="hd",
            )
            response = await generate_image_async(config, request)
            print(response.images[0])

        asyncio.run(main())
        ```
    """
    log_info(
        "Image generation request received (async)",
        context={
            "config": config,
            "request": request,
        },
        logger_name="tarash.tarash_gateway.api",
        redact=True,
    )

    # Delegate to orchestrator
    return await _ORCHESTRATOR.execute_image_async(
        config, request, on_progress=on_progress
    )


def generate_image(
    config: ImageGenerationConfig,
    request: ImageGenerationRequest,
    on_progress: ImageProgressCallback | None = None,
) -> ImageGenerationResponse:
    """Generate an image synchronously using the configured provider.

    Blocking version of [generate_image_async][]. Runs the full orchestration
    pipeline and returns only when generation is complete. Automatically falls
    back to ``config.fallback_configs`` on retryable errors.

    Args:
        config: Provider configuration including API key, model, timeout, and
            optional fallback chain.
        request: Image generation parameters (prompt, size, quality, style,
            etc.). Unknown fields are captured into ``extra_params``.
        on_progress: Optional callback invoked with an [ImageGenerationUpdate][]
            during generation. Accepts both sync and async callables.

    Returns:
        [ImageGenerationResponse][] containing a list of generated images
        (base64 or URL), status, and ``execution_metadata``.

    Raises:
        TarashException: If generation fails on all providers in the fallback
            chain.
        NotImplementedError: If the configured provider does not support image
            generation.

    Example:
        ```python
        from tarash.tarash_gateway import generate_image
        from tarash.tarash_gateway.models import ImageGenerationConfig, ImageGenerationRequest

        config = ImageGenerationConfig(provider="openai", api_key="OPENAI_KEY")
        request = ImageGenerationRequest(
            prompt="A futuristic cityscape at dusk, photorealistic",
            size="1024x1024",
            quality="hd",
        )
        response = generate_image(config, request)
        print(response.images[0])
        ```
    """
    log_info(
        "Image generation request received (sync)",
        context={
            "config": config,
            "request": request,
        },
        redact=True,
        logger_name="tarash.tarash_gateway.api",
    )

    # Delegate to orchestrator
    return _ORCHESTRATOR.execute_image_sync(config, request, on_progress=on_progress)


# ==================== TTS Generation API ====================


async def generate_tts_async(
    config: AudioGenerationConfig,
    request: TTSRequest,
    on_progress: TTSProgressCallback | None = None,
) -> TTSResponse:
    """Generate speech from text asynchronously using the configured provider.

    Args:
        config: Provider configuration including API key, model, and timeout.
        request: TTS parameters (text, voice_id, output_format, etc.).
        on_progress: Optional callback invoked during generation.

    Returns:
        [TTSResponse][] with base64-encoded audio and metadata.

    Raises:
        TarashException: If generation fails on all providers in the fallback chain.
        NotImplementedError: If the configured provider does not support TTS.
    """
    log_info(
        "TTS generation request received (async)",
        context={
            "config": config,
            "request": request,
        },
        logger_name="tarash.tarash_gateway.api",
        redact=True,
    )

    return await _ORCHESTRATOR.execute_tts_async(
        config, request, on_progress=on_progress
    )


def generate_tts(
    config: AudioGenerationConfig,
    request: TTSRequest,
    on_progress: TTSProgressCallback | None = None,
) -> TTSResponse:
    """Generate speech from text synchronously using the configured provider.

    Args:
        config: Provider configuration including API key, model, and timeout.
        request: TTS parameters (text, voice_id, output_format, etc.).
        on_progress: Optional callback invoked during generation.

    Returns:
        [TTSResponse][] with base64-encoded audio and metadata.

    Raises:
        TarashException: If generation fails on all providers in the fallback chain.
        NotImplementedError: If the configured provider does not support TTS.
    """
    log_info(
        "TTS generation request received (sync)",
        context={
            "config": config,
            "request": request,
        },
        redact=True,
        logger_name="tarash.tarash_gateway.api",
    )

    return _ORCHESTRATOR.execute_tts_sync(config, request, on_progress=on_progress)


# ==================== STS Generation API ====================


async def generate_sts_async(
    config: AudioGenerationConfig,
    request: STSRequest,
    on_progress: STSProgressCallback | None = None,
) -> STSResponse:
    """Convert speech to speech asynchronously using the configured provider.

    Args:
        config: Provider configuration including API key, model, and timeout.
        request: STS parameters (audio, voice_id, output_format, etc.).
        on_progress: Optional callback invoked during generation.

    Returns:
        [STSResponse][] with base64-encoded audio and metadata.

    Raises:
        TarashException: If generation fails on all providers in the fallback chain.
        NotImplementedError: If the configured provider does not support STS.
    """
    log_info(
        "STS generation request received (async)",
        context={
            "config": config,
            "request": request,
        },
        logger_name="tarash.tarash_gateway.api",
        redact=True,
    )

    return await _ORCHESTRATOR.execute_sts_async(
        config, request, on_progress=on_progress
    )


def generate_sts(
    config: AudioGenerationConfig,
    request: STSRequest,
    on_progress: STSProgressCallback | None = None,
) -> STSResponse:
    """Convert speech to speech synchronously using the configured provider.

    Args:
        config: Provider configuration including API key, model, and timeout.
        request: STS parameters (audio, voice_id, output_format, etc.).
        on_progress: Optional callback invoked during generation.

    Returns:
        [STSResponse][] with base64-encoded audio and metadata.

    Raises:
        TarashException: If generation fails on all providers in the fallback chain.
        NotImplementedError: If the configured provider does not support STS.
    """
    log_info(
        "STS generation request received (sync)",
        context={
            "config": config,
            "request": request,
        },
        redact=True,
        logger_name="tarash.tarash_gateway.api",
    )

    return _ORCHESTRATOR.execute_sts_sync(config, request, on_progress=on_progress)
