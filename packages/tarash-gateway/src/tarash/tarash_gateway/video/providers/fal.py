"""Fal.ai provider handler."""

import asyncio
import time
import traceback
from typing import Any

from fal_client.client import FalClientHTTPError

from tarash.tarash_gateway.logging import log_debug, log_error, log_info
from tarash.tarash_gateway.video.exceptions import (
    GenerationFailedError,
    HTTPConnectionError,
    HTTPError,
    TarashException,
    TimeoutError,
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
    get_field_mappers_from_registry,
    image_list_field_mapper,
    passthrough_field_mapper,
    single_image_field_mapper,
    video_url_field_mapper,
)

try:
    import fal_client
    import httpx
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

# Provider name constant
_PROVIDER_NAME = "fal"

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


# Kling Video O1 - Unified mapper for all variants
# Supports: image-to-video, reference-to-video, video-to-video/edit
# Reference: https://fal.ai/models/fal-ai/kling-video/o1/*
#
# Usage for elements (reference-to-video and video-to-video/edit):
#   Pass elements directly via extra_params in the exact format expected by Fal:
#   extra_params={
#       "elements": [
#           {
#               "frontal_image_url": "url",
#               "reference_image_urls": ["url1", "url2"]  # Optional, 1-4 additional angles
#           }
#       ]
#   }
KLING_O1_FIELD_MAPPERS: dict[str, FieldMapper] = {
    # Core required field (all variants)
    "prompt": passthrough_field_mapper("prompt", required=True),
    # Image-to-video: start/end frame support
    "start_image_url": single_image_field_mapper(
        required=False, image_type="first_frame"
    ),
    "end_image_url": single_image_field_mapper(required=False, image_type="last_frame"),
    # Reference-to-video and video-to-video/edit: elements support (passed via extra_params)
    "elements": extra_params_field_mapper("elements"),
    "image_urls": image_list_field_mapper(image_type="reference"),
    # Optional parameters (variant-specific)
    "duration": duration_field_mapper(
        field_type="str",
        allowed_values=["5", "10"],
        provider="fal",
        model="kling-o1",
        add_suffix=False,  # o1 uses "5", "10" without "s" suffix
    ),
    "aspect_ratio": passthrough_field_mapper("aspect_ratio"),
    # Video-to-video/edit specific
    "video_url": video_url_field_mapper(required=False),
    "keep_audio": passthrough_field_mapper("keep_audio"),
}

# Veo 3 and Veo 3.1 models field mappings
# Supports text-to-video, image-to-video, first-last-frame-to-video, and video-to-video (extend-video)
#
# Notes on extend-video:
# - The fal-ai/veo3.1/fast/extend-video endpoint has stricter constraints:
#   - Only supports 7s duration (vs 4s/6s/8s for other variants)
#   - Only supports 720p resolution
#   - Requires both prompt and video_url
# - These API-level constraints are enforced by Fal's API, not by field mappers
# - Use extra_params for extend-video specific values like aspect_ratio: "auto"
VEO3_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "aspect_ratio": passthrough_field_mapper("aspect_ratio"),
    "duration": duration_field_mapper(
        field_type="str",
        allowed_values=["4s", "6s", "8s", "7s"],  # Added 7s for extend-video
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
    # Video-to-video support (extend-video)
    "video_url": video_url_field_mapper(),
}

# Sora 2 models field mappings
# Supports text-to-video, image-to-video, and video-to-video/remix variants
#
# Notes on video-to-video/remix:
# - The fal-ai/sora-2/video-to-video/remix endpoint remixes Sora-generated videos
# - Requires video_id (not video_url) - can only remix Sora 2 generated videos
# - Pass video_id via extra_params: extra_params={"video_id": "video_123"}
# - Does not use aspect_ratio, resolution, duration (inherited from original video)
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
    # Video-to-video/remix support (via extra_params)
    "video_id": extra_params_field_mapper("video_id"),
}

# Wan v2.6 and v2.5 - Unified mapper for all video generation endpoints
# Supports: text-to-video, image-to-video, reference-to-video
# Works with both v2.6 (wan/v2.6/*) and v2.5 (fal-ai/wan-25-preview/*)
WAN_VIDEO_GENERATION_MAPPERS: dict[str, FieldMapper] = {
    # Core parameters (all variants)
    "prompt": passthrough_field_mapper("prompt", required=True),
    "aspect_ratio": passthrough_field_mapper("aspect_ratio"),
    "resolution": passthrough_field_mapper("resolution"),
    "duration": duration_field_mapper(
        field_type="str", add_suffix=False
    ),  # Wan doesn't support "s" suffix
    "negative_prompt": passthrough_field_mapper("negative_prompt"),
    "seed": passthrough_field_mapper("seed"),
    # Image/Video inputs (optional - used by I2V and R2V variants)
    "image_url": single_image_field_mapper(required=False, image_type="reference"),
    "video_urls": extra_params_field_mapper(
        "video_urls"
    ),  # For R2V with @Video1, @Video2, @Video3
    # Wan-specific features
    "audio_url": extra_params_field_mapper("audio_url"),
    "enable_prompt_expansion": passthrough_field_mapper("enhance_prompt"),
    "multi_shots": extra_params_field_mapper("multi_shots"),
    "enable_safety_checker": extra_params_field_mapper("enable_safety_checker"),
}

# Wan v2.2-14b Animate/Move - Video+Image to Video with motion control
WAN_ANIMATE_MAPPERS: dict[str, FieldMapper] = {
    # Required inputs
    "video_url": video_url_field_mapper(required=True),
    "image_url": single_image_field_mapper(required=True, image_type="reference"),
    # Generation parameters
    "resolution": passthrough_field_mapper("resolution"),
    "seed": passthrough_field_mapper("seed"),
    # Motion control parameters (via extra_params)
    "guidance_scale": extra_params_field_mapper("guidance_scale"),
    "num_inference_steps": extra_params_field_mapper("num_inference_steps"),
    "shift": extra_params_field_mapper("shift"),
    # Quality/output parameters
    "video_quality": extra_params_field_mapper("video_quality"),
    "video_write_mode": extra_params_field_mapper("video_write_mode"),
    "use_turbo": extra_params_field_mapper("use_turbo"),
    "return_frames_zip": extra_params_field_mapper("return_frames_zip"),
    # Safety
    "enable_safety_checker": extra_params_field_mapper("enable_safety_checker"),
    "enable_output_safety_checker": extra_params_field_mapper(
        "enable_output_safety_checker"
    ),
}

# ByteDance Seedance - Unified mapper for all versions and variants
# Supports v1 (text-to-video, image-to-video, reference-to-video) and v1.5 (text-to-video)
# Works with all ByteDance Seedance models regardless of version
BYTEDANCE_SEEDANCE_FIELD_MAPPERS: dict[str, FieldMapper] = {
    # Core required
    "prompt": passthrough_field_mapper("prompt", required=True),
    # Core optional (all variants)
    "aspect_ratio": passthrough_field_mapper("aspect_ratio"),
    "resolution": passthrough_field_mapper("resolution"),
    "duration": duration_field_mapper(
        field_type="str",
        allowed_values=["2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
        provider="fal",
        model="bytedance-seedance",
        add_suffix=False,  # ByteDance uses "2", "3", etc. without "s" suffix
    ),
    "camera_fixed": passthrough_field_mapper("camera_fixed"),
    "seed": passthrough_field_mapper("seed"),
    "generate_audio": passthrough_field_mapper(
        "generate_audio"
    ),  # v1.5 only, ignored by v1
    "enable_safety_checker": extra_params_field_mapper("enable_safety_checker"),
    # Image-to-video support (v1/pro/image-to-video)
    # Using strict=False to allow reference-to-video with multiple reference images
    # When there's 1 reference image, both image_url and reference_image_urls work
    # When there are multiple, image_url returns None and reference_image_urls gets the list
    "image_url": single_image_field_mapper(
        required=False, image_type="reference", strict=False
    ),
    "end_image_url": single_image_field_mapper(required=False, image_type="last_frame"),
    # Reference-to-video support (v1/lite/reference-to-video)
    "reference_image_urls": image_list_field_mapper(image_type="reference"),
}

# Pixverse (v5 and v5.5) - Unified mapper for all variants
# Supports text-to-video, image-to-video, transition, effects, and swap
# All variants share common fields with variant-specific optional fields
PIXVERSE_FIELD_MAPPERS: dict[str, FieldMapper] = {
    # Core fields (text-to-video, image-to-video, transition)
    "prompt": passthrough_field_mapper(
        "prompt", required=False
    ),  # Required for most, but not swap/effects
    "aspect_ratio": passthrough_field_mapper("aspect_ratio"),
    "resolution": passthrough_field_mapper("resolution"),
    "duration": duration_field_mapper(
        field_type="str",
        allowed_values=["5", "8", "10"],
        provider="fal",
        model="pixverse-v5.5",
        add_suffix=False,  # Pixverse uses "5", "8", "10" without "s" suffix
    ),
    "style": passthrough_field_mapper("style"),
    "thinking_type": passthrough_field_mapper("thinking_type"),
    "seed": passthrough_field_mapper("seed"),
    "negative_prompt": passthrough_field_mapper("negative_prompt"),
    # Audio generation (text-to-video, image-to-video, transition)
    "generate_audio_switch": passthrough_field_mapper("generate_audio_switch"),
    # Multi-clip generation (text-to-video, image-to-video)
    "generate_multi_clip_switch": passthrough_field_mapper(
        "generate_multi_clip_switch"
    ),
    # Image-to-video / transition / effects / swap support
    "image_url": single_image_field_mapper(required=False, image_type="reference"),
    # Transition support (first and end frames)
    "first_image_url": single_image_field_mapper(
        required=False, image_type="first_frame"
    ),
    "end_image_url": single_image_field_mapper(required=False, image_type="last_frame"),
    # Effects variant
    "effect": extra_params_field_mapper("effect"),
    # Swap variant
    "video_url": video_url_field_mapper(required=False),
    "mode": extra_params_field_mapper("mode"),
    "keyframe_id": extra_params_field_mapper("keyframe_id"),
    "original_sound_switch": extra_params_field_mapper("original_sound_switch"),
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
    # Kling Video O1 - All variants (image-to-video, reference-to-video, video-to-video/edit) use unified mapper
    "fal-ai/kling-video/o1": KLING_O1_FIELD_MAPPERS,
    # Kling Video v2.6 - supports both image-to-video and motion-control
    "fal-ai/kling-video/v2.6": KLING_VIDEO_V26_FIELD_MAPPERS,
    # Veo 3.1 - prefix must be registered before veo3 for longest-match precedence
    # Supports all variants: text-to-video, image-to-video, first-last-frame-to-video, extend-video
    "fal-ai/veo3.1": VEO3_FIELD_MAPPERS,
    # Veo 3 - supports text-to-video, image-to-video, first-last-frame-to-video, and extend-video
    "fal-ai/veo3": VEO3_FIELD_MAPPERS,
    # Sora 2 - supports both text-to-video and image-to-video variants
    "fal-ai/sora-2": SORA2_FIELD_MAPPERS,
    # Wan v2.6 - All variants (text-to-video, image-to-video, reference-to-video) use unified mapper
    "wan/v2.6/": WAN_VIDEO_GENERATION_MAPPERS,
    # Wan v2.5 - Uses same unified mapper as v2.6
    "fal-ai/wan-25-preview/": WAN_VIDEO_GENERATION_MAPPERS,
    # Wan v2.2-14b Animate
    "fal-ai/wan/v2.2-14b/animate/": WAN_ANIMATE_MAPPERS,
    # ByteDance Seedance - Unified mapper for all versions (v1, v1.5) and variants (text-to-video, image-to-video, reference-to-video)
    "fal-ai/bytedance/seedance": BYTEDANCE_SEEDANCE_FIELD_MAPPERS,
    # Pixverse - All variants (text-to-video, image-to-video, transition, effects, swap) use unified mapper
    # Supports both v5 and v5.5 with same field mappings
    "fal-ai/pixverse/v5.5": PIXVERSE_FIELD_MAPPERS,  # v5.5 variants
    "fal-ai/pixverse/v5": PIXVERSE_FIELD_MAPPERS,  # v5 variants (same API)
    "fal-ai/pixverse/swap": PIXVERSE_FIELD_MAPPERS,  # Swap variant (works for both v5 and v5.5)
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
    return get_field_mappers_from_registry(
        model_name, FAL_MODEL_REGISTRY, GENERIC_FIELD_MAPPERS
    )


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

        # httpx timeout errors
        if isinstance(ex, httpx.TimeoutException):
            return TimeoutError(
                f"Request timed out: {str(ex)}",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response={"error": str(ex)},
                timeout_seconds=config.timeout,
            )

        # httpx connection errors
        if isinstance(ex, (httpx.ConnectError, httpx.NetworkError)):
            return HTTPConnectionError(
                f"Connection error: {str(ex)}",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response={"error": str(ex)},
            )

        # Fal HTTP errors
        if isinstance(ex, FalClientHTTPError):
            raw_response = {
                "status_code": ex.status_code,
                "response_headers": ex.response_headers,
                "response": ex.response.content,
            }

            # Validation errors (400, 422)
            if ex.status_code in (400, 422):
                return ValidationError(
                    ex.message,
                    provider=config.provider,
                    model=config.model,
                    request_id=request_id,
                    raw_response=raw_response,
                )

            # All other HTTP errors (401, 403, 429, 500, 503, etc.)
            return HTTPError(
                f"HTTP {ex.status_code}: {ex.message}",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response=raw_response,
                status_code=ex.status_code,
            )

        # Unknown errors
        log_error(
            f"Fal unknown error: {str(ex)}",
            context={
                "provider": config.provider,
                "model": config.model,
                "request_id": request_id,
                "error_type": type(ex).__name__,
            },
            logger_name=_LOGGER_NAME,
            exc_info=True,
        )
        return GenerationFailedError(
            f"Error while generating video: {str(ex)}",
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
