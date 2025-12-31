"""Fal.ai provider handler."""

import asyncio
import traceback
from typing import Any

from fal_client.client import FalClientHTTPError

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

    >>> get_field_mappers("fal-ai/veo3.1")
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
        self._async_client_cache: dict[str, Any] = {}

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
        """

        # Use API key + base_url as cache key
        cache_key = f"{config.api_key}:{config.base_url or 'default'}"

        if client_type == "async":
            if cache_key not in self._async_client_cache:
                self._async_client_cache[cache_key] = fal_client.AsyncClient(
                    key=config.api_key,
                    default_timeout=config.timeout,
                )
            return self._async_client_cache[cache_key]
        else:  # sync
            if cache_key not in self._sync_client_cache:
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
            raise TarashException(
                f"Unknown error while generating video: {ex}",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response={
                    "error": str(ex),
                    "traceback": traceback.format_exc(),
                },
            )

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

        # Submit to Fal using async API
        handler = await client.submit(
            config.model,
            arguments=fal_input,
        )

        request_id = handler.request_id

        try:
            async for event in handler.iter_events(with_logs=True):
                if on_progress:
                    result = on_progress(parse_fal_status(request_id, event))
                    if asyncio.iscoroutine(result):
                        await result

            result = await handler.get()

            # Parse response
            fal_result = (
                result
                if isinstance(result, dict)
                else (result.data if hasattr(result, "data") else {})
            )
            response = self._convert_response(config, request, request_id, fal_result)

            return response

        except Exception as ex:
            raise self._handle_error(config, request, request_id, ex) from ex

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

        # Submit to Fal
        handler = client.submit(
            config.model,
            arguments=fal_input,
        )
        request_id = handler.request_id

        try:
            for event in handler.iter_events(with_logs=True):
                if on_progress:
                    on_progress(parse_fal_status(request_id, event))

            result = handler.get()

            # Parse response
            fal_result = (
                result
                if isinstance(result, dict)
                else (result.data if hasattr(result, "data") else {})
            )
            response = self._convert_response(config, request, request_id, fal_result)

            return response

        except Exception as ex:
            raise self._handle_error(config, request, request_id, ex) from ex
