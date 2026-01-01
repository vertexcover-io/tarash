"""Common field mapping framework for video generation providers.

This module provides a declarative way to map VideoGenerationRequest fields
to provider-specific API formats using FieldMapper objects.
"""

from dataclasses import dataclass
from typing import Any, Callable, Literal

from tarash.tarash_gateway.video.models import (
    ImageType,
    VideoGenerationRequest,
)
from tarash.tarash_gateway.video.utils import convert_to_data_url, validate_duration


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
    provider: str = "unknown",
    model: str | None = None,
) -> FieldMapper:
    """Create a FieldMapper for duration field.

    Args:
        field_type: Output type - "str" for "5s" format, "int" for integer seconds
        allowed_values: Optional list of allowed values for validation
        provider: Provider name for error messages (default: "unknown")
        model: Optional model name for error messages

    Returns:
        FieldMapper for duration_seconds -> duration

    Examples:
        >>> # String format with validation
        >>> duration_field_mapper("str", ["6s", "10s"], "fal", "minimax")

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
                # For string type, extract integer values from string format
                allowed_seconds = [
                    int(v.rstrip("s")) for v in allowed_values if isinstance(v, str)
                ]
                # Use shared validation function
                validate_duration(value, allowed_seconds, provider, model)
                return f"{value}s"
            else:  # int
                # Validate with shared function
                validate_duration(value, allowed_values, provider, model)
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


def extra_params_field_mapper(param_name: str, required: bool = False) -> FieldMapper:
    """Create a FieldMapper for extracting a value from extra_params.

    Args:
        param_name: Key name to extract from extra_params dict
        required: Whether this field is required (default: False)

    Returns:
        FieldMapper for extra_params -> value
    """
    return FieldMapper(
        source_field="extra_params",
        converter=lambda req, val: val.get(param_name)
        if isinstance(val, dict)
        else None,
        required=required,
    )


def video_url_field_mapper(required: bool = False) -> FieldMapper:
    """Create a FieldMapper for converting video field to URL.

    Handles both URL strings and MediaContent dicts with bytes.

    Args:
        required: Whether this field is required (default: False)

    Returns:
        FieldMapper for video -> video_url
    """
    return FieldMapper(
        source_field="video",
        converter=lambda req, val: (
            convert_to_data_url(val)
            if isinstance(val, dict) and "content" in val
            else str(val)
            if val
            else None
        ),
        required=required,
    )


# ==================== Model Registry Utilities ====================


def get_field_mappers_from_registry(
    model_name: str,
    registry: dict[str, dict[str, FieldMapper]],
    fallback_mappers: dict[str, FieldMapper],
) -> dict[str, FieldMapper]:
    """Get field mappers for a model using registry lookup with prefix matching.

    Lookup Strategy:
    1. Try exact match first
    2. If not found, try prefix matching - find registry keys that are prefixes
       of the model_name, and use the longest matching prefix
    3. If no match found, return fallback field mappers

    Args:
        model_name: Full model name (e.g., "fal-ai/minimax/hailuo-02-fast/image-to-video")
        registry: Registry mapping model names/prefixes to field mappers
        fallback_mappers: Default field mappers to use if no match found

    Returns:
        Dict mapping API field names to FieldMapper objects

    Examples:
        >>> registry = {
        ...     "fal-ai/minimax": MINIMAX_MAPPERS,
        ...     "fal-ai/veo3.1": VEO31_MAPPERS,
        ...     "fal-ai/veo3": VEO3_MAPPERS,
        ... }
        >>> get_field_mappers_from_registry("fal-ai/minimax", registry, GENERIC)
        MINIMAX_MAPPERS  # Exact match
        >>> get_field_mappers_from_registry("fal-ai/minimax/hailuo-02-fast", registry, GENERIC)
        MINIMAX_MAPPERS  # Prefix match
        >>> get_field_mappers_from_registry("fal-ai/veo3.1/fast", registry, GENERIC)
        VEO31_MAPPERS  # Prefix match (longest)
        >>> get_field_mappers_from_registry("fal-ai/unknown-model", registry, GENERIC)
        GENERIC  # Fallback for unknown models
    """
    # Try exact match first
    if model_name in registry:
        return registry[model_name]

    # Try prefix matching - find all registry keys that are prefixes of model_name
    # Use the longest matching prefix
    matching_prefix = None
    for registry_key in registry:
        if model_name.startswith(registry_key):
            # Found a prefix match - keep if it's longer than current match
            if matching_prefix is None or len(registry_key) > len(matching_prefix):
                matching_prefix = registry_key

    if matching_prefix:
        return registry[matching_prefix]

    # No match found - use fallback
    return fallback_mappers
