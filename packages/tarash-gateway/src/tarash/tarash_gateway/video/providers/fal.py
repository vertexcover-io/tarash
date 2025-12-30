"""Fal.ai provider handler."""

import asyncio
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Literal

from fal_client.client import FalClientHTTPError

from tarash.tarash_gateway.video.exceptions import (
    ProviderAPIError,
    VideoGenerationError,
    handle_video_generation_errors,
)
from tarash.tarash_gateway.video.models import (
    ImageType,
    ProgressCallback,
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoGenerationUpdate,
)
from tarash.tarash_gateway.video.utils import convert_to_data_url

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


# ==================== Field Mapping Framework ====================


@dataclass
class FieldMapper:
    """Maps a VideoGenerationRequest field to an API field with conversion.

    Attributes:
        source_field: Field name in VideoGenerationRequest
        converter: Function that takes (request, field_value) and returns converted value
        required: Whether this field is required (default: False)
    """

    source_field: str
    converter: Callable[[VideoGenerationRequest, Any], Any]
    required: bool = False


def apply_field_mappers(
    field_mappers: dict[str, FieldMapper],
    request: VideoGenerationRequest,
) -> dict[str, Any]:
    """Apply field mappers to convert VideoGenerationRequest to API format.

    Args:
        field_mappers: Dict mapping API field names to FieldMapper objects
        request: The video generation request

    Returns:
        Dict with API field names and converted values (None values excluded)

    Raises:
        ValueError: If a required field is None or missing
    """
    result = {}

    for api_field_name, mapper in field_mappers.items():
        # Get the source value from the request
        source_value = getattr(request, mapper.source_field, None)

        # Check if required field is missing
        if mapper.required and source_value is None:
            raise ValueError(
                f"Required field '{mapper.source_field}' is missing or None"
            )

        # Apply converter
        converted_value = mapper.converter(request, source_value)

        # Check if required field converter returned None
        if mapper.required and converted_value is None:
            raise ValueError(
                f"Required field '{api_field_name}' cannot be None after conversion"
            )

        # Only include non-None values and non-empty collections
        if (
            converted_value is not None
            and converted_value != []
            and converted_value != {}
        ):
            result[api_field_name] = converted_value

    return result


# ==================== Utility Functions for FieldMappers ====================


def duration_field_mapper(
    field_type: Literal["str", "int"],
    allowed_values: list[str] | list[int] | None = None,
) -> FieldMapper:
    """Create a FieldMapper for duration field.

    Args:
        field_type: Output type - "str" for "5s" format, "int" for integer seconds
        allowed_values: Optional list of allowed values for validation

    Returns:
        FieldMapper for duration_seconds -> duration

    Examples:
        >>> # String format with validation
        >>> duration_field_mapper("str", ["6s", "10s"])

        >>> # Integer format without validation
        >>> duration_field_mapper("int")
    """

    def converter(
        request: VideoGenerationRequest, value: int | None
    ) -> str | int | None:
        if value is None:
            return None

        # Validate allowed values
        if allowed_values:
            if field_type == "str":
                # For string type, convert first then validate
                str_value = f"{value}s"
                if str_value not in allowed_values:
                    # Extract seconds for better error messages
                    allowed_seconds = [
                        int(v.rstrip("s")) for v in allowed_values if isinstance(v, str)
                    ]
                    raise ValueError(
                        f"Minimax only supports {' or '.join(map(str, allowed_seconds))} second durations, got {value}"
                    )
                return str_value
            else:  # int
                if value not in allowed_values:
                    raise ValueError(
                        f"Duration must be one of {allowed_values}, got {value}"
                    )
                return value

        # No validation, just convert
        if field_type == "str":
            return f"{value}s"
        else:
            return value

    return FieldMapper(source_field="duration_seconds", converter=converter)


def single_image_field_mapper(
    required: bool = False,
    image_type: Literal[
        "reference", "first_frame", "last_frame", "asset", "style"
    ] = "reference",
) -> FieldMapper:
    """Create a FieldMapper for extracting single image from image_list.

    Validates that only 1 image of the specified type is provided and converts it to a URL string.

    Args:
        required: Whether this field is required (default: False)
        image_type: Type of image to extract (default: "reference")

    Returns:
        FieldMapper for image_list -> image_url (single string)
    """

    def converter(
        request: VideoGenerationRequest, value: list[ImageType] | None
    ) -> str | None:
        if not value or len(value) == 0:
            return None

        # Filter images by the specified type
        # For "reference", also accept "first_frame" for backward compatibility
        if image_type == "reference":
            filtered_images = [
                img for img in value if img["type"] in ("reference", "first_frame")
            ]
        else:
            filtered_images = [img for img in value if img["type"] == image_type]

        if len(filtered_images) == 0:
            return None

        if len(filtered_images) > 1:
            raise ValueError(
                f"Only 1 {image_type} image allowed, got {len(filtered_images)} images",
            )

        # Extract the image
        target_image = filtered_images[0]
        if isinstance(target_image, dict) and "image" in target_image:
            media = target_image["image"]
            if isinstance(media, dict) and "content" in media:
                return convert_to_data_url(media)
            return str(media)
        elif isinstance(target_image, str):
            return target_image

        return None

    return FieldMapper(
        source_field="image_list", converter=converter, required=required
    )


def image_list_field_mapper() -> FieldMapper:
    """Create a FieldMapper for converting image_list to list of URLs.

    Returns:
        FieldMapper for image_list -> image_urls (list of strings)
    """

    def converter(request: VideoGenerationRequest, value: list | None) -> list[str]:
        if not value:
            return []

        urls = []
        for item in value:
            if isinstance(item, dict) and "image" in item:
                media = item["image"]
                if isinstance(media, dict) and "content" in media:
                    urls.append(convert_to_data_url(media))
                else:
                    urls.append(str(media))
            elif isinstance(item, str):
                urls.append(item)

        return urls

    return FieldMapper(source_field="image_list", converter=converter)


def passthrough_field_mapper(source_field: str, required: bool = False) -> FieldMapper:
    """Create a FieldMapper that passes through the value unchanged.

    Args:
        source_field: Field name in VideoGenerationRequest
        required: Whether this field is required (default: False)

    Returns:
        FieldMapper that returns the value as-is
    """
    return FieldMapper(
        source_field=source_field,
        converter=lambda req, val: val,
        required=required,
    )


# ==================== Model Field Mappings ====================


# Minimax models field mappings
MINIMAX_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "duration": duration_field_mapper(field_type="str", allowed_values=["6s", "10s"]),
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
    "cfg_scale": FieldMapper(
        source_field="extra_params",
        converter=lambda req, val: val.get("cfg_scale")
        if isinstance(val, dict)
        else None,
        required=False,
    ),
    "voice_ids": FieldMapper(
        source_field="extra_params",
        converter=lambda req, val: val.get("voice_ids")
        if isinstance(val, dict)
        else None,
        required=False,
    ),
    # Motion control specific fields
    "video_url": FieldMapper(
        source_field="video",
        converter=lambda req, val: convert_to_data_url(val)
        if isinstance(val, dict) and "content" in val
        else str(val)
        if val
        else None,
        required=False,
    ),
    "character_orientation": FieldMapper(
        source_field="extra_params",
        converter=lambda req, val: val.get("character_orientation")
        if isinstance(val, dict)
        else None,
        required=False,
    ),
    "keep_original_sound": FieldMapper(
        source_field="extra_params",
        converter=lambda req, val: val.get("keep_original_sound")
        if isinstance(val, dict)
        else None,
        required=False,
    ),
}

# Generic fallback field mappings
GENERIC_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "duration": duration_field_mapper(field_type="int"),
    "resolution": passthrough_field_mapper("resolution"),
    "aspect_ratio": passthrough_field_mapper("aspect_ratio"),
    "image_urls": image_list_field_mapper(),
    "video_url": FieldMapper(
        source_field="video",
        converter=lambda req, val: convert_to_data_url(val)
        if isinstance(val, dict) and "content" in val
        else str(val)
        if val
        else None,
        required=False,
    ),
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
            raise ProviderAPIError(
                f"No video URL found in Fal response: {provider_response}",
                provider=config.provider,
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
    ) -> VideoGenerationResponse:
        """Handle errors during video generation."""
        if isinstance(ex, VideoGenerationError):
            return ex

        elif isinstance(ex, FalClientHTTPError):
            return VideoGenerationError(
                f"Request Failed with status code {ex.status_code}: {ex.message}",
                provider=config.provider,
                raw_response={
                    "status_code": ex.status_code,
                    "response_headers": ex.response_headers,
                    "response": ex.response.content,
                    "traceback": traceback.format_exc(),
                },
            )
        else:
            raise VideoGenerationError(
                f"Unknown Error while generating video: {ex}",
                provider=config.provider,
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
            raise self._handle_error(config, request, request_id, ex)
