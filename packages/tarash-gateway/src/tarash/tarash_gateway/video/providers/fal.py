"""Fal.ai provider handler."""

import asyncio
import time
import traceback
from typing import Any

from fal_client.client import FalClientHTTPError

from tarash.tarash_gateway.logging import log_debug, log_error, log_info
from tarash.tarash_gateway.video.exceptions import (
    ContentModerationError,
    GenerationFailedError,
    HTTPError,
    TarashException,
    ValidationError,
    handle_video_generation_errors,
)
from tarash.tarash_gateway.video.models import (
    ProgressCallback,
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoGenerationUpdate,
)
from tarash.tarash_gateway.video.providers.field_mappers import (
    FieldMapper,
    apply_field_mappers,
    duration_field_mapper,
    extra_params_field_mapper,
    image_list_field_mapper,
    passthrough_field_mapper,
    single_image_field_mapper,
    video_url_field_mapper,
)

try:
    import fal_client
    from fal_client import (
        AsyncClient,
        Status,
        SyncClient,
        Completed,
        Queued,
        InProgress,
    )
except ImportError:
    pass

# Logger name constant
_LOGGER_NAME = "tarash.tarash_gateway.video.providers.fal"

# ==================== Model Field Mappings ====================


# Minimax models field mappings
MINIMAX_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "duration": duration_field_mapper(
        field_type="str", allowed_values=["6s", "10s"], provider="fal", model="minimax"
    ),
    "image_url": single_image_field_mapper(),
    "prompt_optimizer": passthrough_field_mapper("enhance_prompt"),
}

# Kling Video v2.6 models field mappings
# Supports both image-to-video and motion-control variants
KLING_VIDEO_V26_FIELD_MAPPERS: dict[str, FieldMapper] = {
    # Required for image-to-video
    "prompt": passthrough_field_mapper("prompt", required=True),
    "image_url": single_image_field_mapper(required=True, image_type="reference"),
    # Optional for image-to-video
    "duration": duration_field_mapper(field_type="str", allowed_values=["5", "10"]),
    "generate_audio": passthrough_field_mapper("generate_audio"),
    "negative_prompt": passthrough_field_mapper("negative_prompt"),
    "tail_image_url": single_image_field_mapper(
        required=False, image_type="last_frame"
    ),
    "cfg_scale": extra_params_field_mapper("cfg_scale"),
    "voice_ids": extra_params_field_mapper("voice_ids"),
    # Motion control specific fields
    "video_url": video_url_field_mapper(),
    "character_orientation": extra_params_field_mapper("character_orientation"),
    "keep_original_sound": extra_params_field_mapper("keep_original_sound"),
}

# Veo 3 and Veo 3.1 models field mappings
# Supports text-to-video, image-to-video, and first-last-frame-to-video
# Both versions use the same API parameters
VEO3_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "aspect_ratio": passthrough_field_mapper("aspect_ratio"),
    "duration": duration_field_mapper(
        field_type="str",
        allowed_values=["4s", "6s", "8s"],
        provider="fal",
        model="veo3",
    ),
    "generate_audio": passthrough_field_mapper("generate_audio"),
    "resolution": passthrough_field_mapper("resolution"),
    "auto_fix": passthrough_field_mapper("auto_fix"),
    "seed": passthrough_field_mapper("seed"),
    "negative_prompt": passthrough_field_mapper("negative_prompt"),
    # Image-to-video support
    "image_url": single_image_field_mapper(required=False, image_type="reference"),
    # First-last-frame-to-video support
    "first_frame_url": single_image_field_mapper(
        required=False, image_type="first_frame"
    ),
    "last_frame_url": single_image_field_mapper(
        required=False, image_type="last_frame"
    ),
}

# Sora 2 models field mappings
# Supports both text-to-video and image-to-video variants
SORA2_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "aspect_ratio": passthrough_field_mapper("aspect_ratio"),
    "resolution": passthrough_field_mapper("resolution"),
    "duration": duration_field_mapper(
        field_type="int", allowed_values=[4, 8, 12], provider="fal", model="sora-2"
    ),
    "delete_video": passthrough_field_mapper("delete_video"),
    # Image-to-video support (optional - required only for image-to-video variant)
    "image_url": single_image_field_mapper(required=False, image_type="reference"),
}

# Generic fallback field mappings
GENERIC_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "duration": duration_field_mapper(field_type="int"),
    "resolution": passthrough_field_mapper("resolution"),
    "aspect_ratio": passthrough_field_mapper("aspect_ratio"),
    "image_urls": image_list_field_mapper(),
    "video_url": video_url_field_mapper(),
    "seed": passthrough_field_mapper("seed"),
    "negative_prompt": passthrough_field_mapper("negative_prompt"),
    "generate_audio": passthrough_field_mapper("generate_audio"),
}


# ==================== Model Registry ====================


# Registry maps model names (or prefixes) to their field mappers
FAL_MODEL_REGISTRY: dict[str, dict[str, FieldMapper]] = {
    # Minimax - all variants use same field mappings
    "fal-ai/minimax": MINIMAX_FIELD_MAPPERS,
    # Kling Video v2.6 - supports both image-to-video and motion-control
    "fal-ai/kling-video/v2.6": KLING_VIDEO_V26_FIELD_MAPPERS,
    # Veo 3.1 - prefix must be registered before veo3 for longest-match precedence
    "fal-ai/veo3.1": VEO3_FIELD_MAPPERS,
    # Veo 3 - supports text-to-video, image-to-video, and first-last-frame-to-video
    "fal-ai/veo3": VEO3_FIELD_MAPPERS,
    # Sora 2 - supports both text-to-video and image-to-video variants
    "fal-ai/sora-2": SORA2_FIELD_MAPPERS,
    # Future models...
    # "fal-ai/hunyuan-video": HUNYUAN_FIELD_MAPPERS,
}


def get_field_mappers(model_name: str) -> dict[str, FieldMapper]:
    """Get the field mappers for a given model.

    Lookup Strategy:
    1. Try exact match first
    2. If not found, try prefix matching - find registry keys that are prefixes
       of the model_name, and use the longest matching prefix
    3. If no match found, return generic field mappers as fallback

    Examples:
    --------
    >>> get_field_mappers("fal-ai/minimax")
    MINIMAX_FIELD_MAPPERS  # Exact match

    >>> get_field_mappers("fal-ai/minimax-video")
    MINIMAX_FIELD_MAPPERS  # Prefix match

    >>> get_field_mappers("fal-ai/minimax/hailuo-02-fast/image-to-video")
    MINIMAX_FIELD_MAPPERS  # Prefix match

    >>> get_field_mappers("fal-ai/veo3.1/fast")
    VEO31_FIELD_MAPPERS  # Prefix match

    >>> get_field_mappers("fal-ai/unknown-model")
    GENERIC_FIELD_MAPPERS  # Fallback for unknown models

    Args:
        model_name: Full model name (e.g., "fal-ai/minimax/hailuo-02-fast/image-to-video")

    Returns:
        Dict mapping API field names to FieldMapper objects
    """
    # Try exact match first
    if model_name in FAL_MODEL_REGISTRY:
        return FAL_MODEL_REGISTRY[model_name]

    # Try prefix matching - find all registry keys that are prefixes of model_name
    # Use the longest matching prefix
    matching_prefix = None
    for registry_key in FAL_MODEL_REGISTRY:
        if model_name.startswith(registry_key):
            # Found a prefix match - keep if it's longer than current match
            if matching_prefix is None or len(registry_key) > len(matching_prefix):
                matching_prefix = registry_key

    if matching_prefix:
        return FAL_MODEL_REGISTRY[matching_prefix]

    # No match found - use generic fallback
    return GENERIC_FIELD_MAPPERS


def parse_fal_status(request_id: str, status: Status) -> VideoGenerationUpdate:
    """Parse Fal status update into VideoGenerationUpdate."""
    if isinstance(status, Completed):
        return VideoGenerationUpdate(
            request_id=request_id,
            status="completed",
            update={
                "metrics": status.metrics,
                "logs": status.logs,
            },
        )
    elif isinstance(status, Queued):
        return VideoGenerationUpdate(
            request_id=request_id,
            status="queued",
            update={"position": status.position},
        )
    elif isinstance(status, InProgress):
        return VideoGenerationUpdate(
            request_id=request_id,
            status="processing",
            update={"logs": status.logs},
        )
    else:
        raise ValueError(f"Unknown status: {status}")


# ==================== Provider Handler ====================


class FalProviderHandler:
    """Handler for Fal.ai provider."""

    def __init__(self):
        """Initialize handler (stateless, no config stored)."""

        try:
            import fal_client  # noqa: F401
        except ImportError:
            raise ImportError(
                "fal-client is required for Fal provider. "
                "Install with: pip install tarash-gateway[fal]"
            )

        self._sync_client_cache: dict[str, Any] = {}
        # Note: AsyncClient is NOT cached to avoid "Event Loop closed" errors
        # Each async request creates a new client to ensure proper cleanup

    def _get_client(
        self, config: VideoGenerationConfig, client_type: str
    ) -> AsyncClient | SyncClient:
        """
        Get or create Fal client for the given config.

        Args:
            config: Provider configuration
            client_type: Type of client to return ("sync" or "async")

        Returns:
            fal_client.SyncClient or fal_client.AsyncClient instance

        Note:
            AsyncClient instances are created fresh for each request to avoid
            "Event Loop closed" errors that occur when cached clients outlive
            the event loop they were created in.
        """

        # Use API key + base_url as cache key
        cache_key = f"{config.api_key}:{config.base_url or 'default'}"

        if client_type == "async":
            # Don't cache AsyncClient - create new instance for each request
            # This prevents "Event Loop closed" errors
            log_debug(
                "Creating new async Fal client",
                context={
                    "provider": config.provider,
                    "model": config.model,
                    "base_url": config.base_url or "default",
                },
                logger_name=_LOGGER_NAME,
            )
            return fal_client.AsyncClient(
                key=config.api_key,
                default_timeout=config.timeout,
            )
        else:  # sync
            if cache_key not in self._sync_client_cache:
                log_debug(
                    "Creating new sync Fal client",
                    context={
                        "provider": config.provider,
                        "model": config.model,
                        "base_url": config.base_url or "default",
                    },
                    logger_name=_LOGGER_NAME,
                )
                self._sync_client_cache[cache_key] = fal_client.SyncClient(
                    key=config.api_key,
                    default_timeout=config.timeout,
                )
            return self._sync_client_cache[cache_key]

    def _convert_request(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
    ) -> dict[str, Any]:
        """Convert VideoGenerationRequest to model-specific format.

        Process:
        1. Get model-specific field mappers from registry (with prefix matching)
        2. Apply field mappers to convert request to API format
        3. Merge with extra_params (allows manual overrides)

        Args:
            config: Provider configuration
            request: Generic video generation request

        Returns:
            Model-specific validated request dictionary

        Raises:
            ValueError: If validation fails during field mapping
        """
        # Get model-specific field mappers (with prefix matching)
        field_mappers = get_field_mappers(config.model)

        # Apply field mappers to convert request
        api_payload = apply_field_mappers(field_mappers, request)

        # Merge with extra_params (allows manual overrides)
        api_payload.update(request.extra_params)

        log_info(
            "Mapped request to provider format",
            context={
                "provider": config.provider,
                "model": config.model,
                "converted_request": api_payload,
            },
            logger_name=_LOGGER_NAME,
            redact=True,
        )

        return api_payload

    def _convert_response(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        request_id: str,
        provider_response: Any,
    ) -> VideoGenerationResponse:
        """
        Convert Fal response to VideoGenerationResponse.

        Args:
            config: Provider configuration
            request: Original video generation request
            request_id: Our request ID
            provider_response: Raw Fal response

        Returns:
            Normalized VideoGenerationResponse
        """
        # Extract video URL from Fal response
        # Fal returns: {"video": {"url": "..."}, ...} or {"video_url": "..."}
        video_url = None
        audio_url = None

        if isinstance(provider_response, dict):
            # Try different response formats
            if "video" in provider_response and isinstance(
                provider_response["video"], dict
            ):
                video_url = provider_response["video"].get("url")
            elif "video_url" in provider_response:
                video_url = provider_response["video_url"]

            if "audio" in provider_response and isinstance(
                provider_response["audio"], dict
            ):
                audio_url = provider_response["audio"].get("url")
            elif "audio_url" in provider_response:
                audio_url = provider_response["audio_url"]

        if not video_url:
            raise GenerationFailedError(
                f"No video URL found in Fal response: {provider_response}",
                provider=config.provider,
                model=config.model,
                raw_response=provider_response
                if isinstance(provider_response, dict)
                else {},
            )

        return VideoGenerationResponse(
            request_id=request_id,
            video=video_url,
            audio_url=audio_url,
            duration=provider_response.get("duration")
            if isinstance(provider_response, dict)
            else None,
            resolution=provider_response.get("resolution")
            if isinstance(provider_response, dict)
            else None,
            aspect_ratio=provider_response.get("aspect_ratio")
            if isinstance(provider_response, dict)
            else None,
            status="completed",
            raw_response=provider_response
            if isinstance(provider_response, dict)
            else {"data": str(provider_response)},
            provider_metadata={},
        )

    def _handle_error(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        request_id: str,
        ex: Exception,
    ) -> TarashException:
        """Handle errors during video generation."""
        if isinstance(ex, TarashException):
            return ex

        elif isinstance(ex, FalClientHTTPError):
            log_error(
                "Fal API HTTP error",
                context={
                    "provider": config.provider,
                    "model": config.model,
                    "request_id": request_id,
                    "status_code": ex.status_code,
                    "error_message": ex.message,
                },
                logger_name=_LOGGER_NAME,
            )
            # Map HTTP status codes to appropriate exception types
            raw_response = {
                "status_code": ex.status_code,
                "response_headers": ex.response_headers,
                "response": ex.response.content,
                "traceback": traceback.format_exc(),
            }

            if ex.status_code == 400:
                # Bad request - validation error
                return ValidationError(
                    f"Invalid request: {ex.message}",
                    provider=config.provider,
                    model=config.model,
                    request_id=request_id,
                    raw_response=raw_response,
                )
            elif ex.status_code == 403:
                # Content moderation / policy violation
                return ContentModerationError(
                    f"Content policy violation: {ex.message}",
                    provider=config.provider,
                    model=config.model,
                    request_id=request_id,
                    raw_response=raw_response,
                )
            else:
                # Other HTTP errors (401, 429, 500, 503, etc.)
                return HTTPError(
                    f"HTTP {ex.status_code}: {ex.message}",
                    provider=config.provider,
                    model=config.model,
                    request_id=request_id,
                    raw_response=raw_response,
                    status_code=ex.status_code,
                )
        else:
            log_error(
                "Unknown error in Fal video generation",
                context={
                    "provider": config.provider,
                    "model": config.model,
                    "request_id": request_id,
                    "error_type": type(ex).__name__,
                },
                logger_name=_LOGGER_NAME,
                exc_info=True,
            )
            raise TarashException(
                f"Unknown Error while generating video: {ex}",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response={
                    "error": str(ex),
                    "traceback": traceback.format_exc(),
                },
            )

    def _process_event(
        self,
        config: VideoGenerationConfig,
        request_id: str,
        event: Status,
        start_time: float,
    ) -> VideoGenerationUpdate:
        """Process a single event and log it with elapsed time.

        Args:
            config: Provider configuration
            request_id: Request ID
            event: Fal status event
            start_time: Start time for elapsed time calculation

        Returns:
            VideoGenerationUpdate object
        """
        update = parse_fal_status(request_id, event)
        elapsed_time = time.time() - start_time

        log_info(
            "Progress status update",
            context={
                "provider": config.provider,
                "model": config.model,
                "request_id": request_id,
                "status": update.status,
                "progress_percent": update.progress_percent,
                "time_elapsed_seconds": round(elapsed_time, 2),
            },
            logger_name=_LOGGER_NAME,
        )

        return update

    async def _process_events_async(
        self,
        config: VideoGenerationConfig,
        request_id: str,
        handler: Any,
        on_progress: ProgressCallback | None = None,
    ) -> Any:
        """Process events asynchronously and return final result.

        Args:
            config: Provider configuration
            request_id: Request ID
            handler: Fal async handler
            on_progress: Optional async progress callback

        Returns:
            Final result from handler.get()
        """
        start_time = time.time()

        async for event in handler.iter_events(
            with_logs=True, interval=config.poll_interval
        ):
            update = self._process_event(config, request_id, event, start_time)

            if on_progress:
                result = on_progress(update)
                if asyncio.iscoroutine(result):
                    await result

        return await handler.get()

    def _process_events_sync(
        self,
        config: VideoGenerationConfig,
        request_id: str,
        handler: Any,
        on_progress: Any | None = None,
    ) -> Any:
        """Process events synchronously and return final result.

        Args:
            config: Provider configuration
            request_id: Request ID
            handler: Fal sync handler
            on_progress: Optional progress callback

        Returns:
            Final result from handler.get()
        """
        start_time = time.time()

        for event in handler.iter_events(with_logs=True, interval=config.poll_interval):
            update = self._process_event(config, request_id, event, start_time)

            if on_progress:
                on_progress(update)

        return handler.get()

    @handle_video_generation_errors
    async def generate_video_async(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """
        Generate video asynchronously via Fal with async progress callback.

        Args:
            config: Provider configuration
            request: Video generation request
            on_progress: Optional async callback for progress updates

        Returns:
            Final VideoGenerationResponse when complete
        """
        client = self._get_client(config, "async")
        # Build Fal input (let validation errors propagate)
        fal_input = self._convert_request(config, request)

        log_debug(
            "Starting API call",
            context={
                "provider": config.provider,
                "model": config.model,
            },
            logger_name=_LOGGER_NAME,
        )

        # Submit to Fal using async API
        handler = await client.submit(
            config.model,
            arguments=fal_input,
        )

        request_id = handler.request_id

        log_debug(
            "Request submitted",
            context={
                "provider": config.provider,
                "model": config.model,
                "request_id": request_id,
            },
            logger_name=_LOGGER_NAME,
        )

        try:
            result = await self._process_events_async(
                config, request_id, handler, on_progress
            )

            log_debug(
                "Request complete",
                context={
                    "provider": config.provider,
                    "model": config.model,
                    "request_id": request_id,
                    "response": result,
                },
                logger_name=_LOGGER_NAME,
                redact=True,
            )

            # Parse response
            fal_result = (
                result
                if isinstance(result, dict)
                else (result.data if hasattr(result, "data") else {})
            )
            response = self._convert_response(config, request, request_id, fal_result)

            log_info(
                "Final generated response",
                context={
                    "provider": config.provider,
                    "model": config.model,
                    "request_id": request_id,
                    "response": response,
                },
                logger_name=_LOGGER_NAME,
                redact=True,
            )

            return response

        except Exception as ex:
            raise self._handle_error(config, request, request_id, ex)

    @handle_video_generation_errors
    def generate_video(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: Any | None = None,
    ) -> VideoGenerationResponse:
        """
        Generate video synchronously (blocking).

        Args:
            config: Provider configuration
            request: Video generation request
            on_progress: Optional callback for progress updates

        Returns:
            Final VideoGenerationResponse
        """
        client = self._get_client(config, "sync")

        # Build Fal input (let validation errors propagate)
        fal_input = self._convert_request(config, request)

        log_debug(
            "Starting API call",
            context={
                "provider": config.provider,
                "model": config.model,
            },
            logger_name=_LOGGER_NAME,
        )

        # Submit to Fal
        handler = client.submit(
            config.model,
            arguments=fal_input,
        )
        request_id = handler.request_id

        log_debug(
            "Request submitted",
            context={
                "provider": config.provider,
                "model": config.model,
                "request_id": request_id,
            },
            logger_name=_LOGGER_NAME,
        )

        try:
            result = self._process_events_sync(config, request_id, handler, on_progress)

            log_debug(
                "Request complete",
                context={
                    "provider": config.provider,
                    "model": config.model,
                    "request_id": request_id,
                    "response": result,
                },
                logger_name=_LOGGER_NAME,
                redact=True,
            )

            # Parse response
            fal_result = (
                result
                if isinstance(result, dict)
                else (result.data if hasattr(result, "data") else {})
            )
            response = self._convert_response(config, request, request_id, fal_result)

            log_info(
                "Final generated response",
                context={
                    "provider": config.provider,
                    "model": config.model,
                    "request_id": request_id,
                    "response": response,
                },
                logger_name=_LOGGER_NAME,
                redact=True,
            )

            return response

        except Exception as ex:
            raise self._handle_error(config, request, request_id, ex)
