"""Common field mapping framework for video generation providers.

This module provides a declarative way to map VideoGenerationRequest fields
to provider-specific API formats using FieldMapper objects.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Literal, TypedDict, cast

from tarash.tarash_gateway.video.models import (
    MediaContent,
    VideoGenerationRequest,
)
from tarash.tarash_gateway.video.utils import convert_to_data_url, validate_duration


# Type alias for image list items
class ImageListItem(TypedDict, total=False):
    """Type for items in the image_list field."""

    type: str
    image: str | MediaContent


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
    converter: Callable[[VideoGenerationRequest, object], object]
    required: bool = False


def apply_field_mappers(
    field_mappers: dict[str, FieldMapper],
    request: VideoGenerationRequest,
) -> dict[str, object]:
    """Apply field mappers to convert VideoGenerationRequest to API format.

    Args:
        field_mappers: Dict mapping API field names to FieldMapper objects
        request: The video generation request

    Returns:
        Dict with API field names and converted values (None values excluded)

    Raises:
        ValueError: If a required field is None or missing
    """
    result: dict[str, object] = {}

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
    add_suffix: bool = True,
) -> FieldMapper:
    """Create a FieldMapper for duration field.

    Args:
        field_type: Output type - "str" for "5s" format, "int" for integer seconds
        allowed_values: Optional list of allowed values for validation
        provider: Provider name for error messages (default: "unknown")
        model: Optional model name for error messages
        add_suffix: Whether to add "s" suffix for string type (default: True)

    Returns:
        FieldMapper for duration_seconds -> duration

    Examples:
        >>> # String format with validation and suffix
        >>> duration_field_mapper("str", ["6s", "10s"], "fal", "minimax")

        >>> # String format without suffix (ByteDance)
        >>> duration_field_mapper("str", ["4", "5", "6"], "fal", "bytedance", add_suffix=False)

        >>> # Integer format without validation
        >>> duration_field_mapper("int")
    """

    def converter(_request: VideoGenerationRequest, value: object) -> str | int | None:
        if value is None:
            return None

        # Ensure value is an int for duration
        duration_value = int(value) if isinstance(value, (int, float, str)) else 0

        # Validate allowed values
        if allowed_values:
            if field_type == "str":
                # For string type, extract integer values from string format
                allowed_seconds = [
                    int(v.rstrip("s")) for v in allowed_values if isinstance(v, str)
                ]
                # Use shared validation function
                _ = validate_duration(duration_value, allowed_seconds, provider, model)
                return f"{duration_value}s" if add_suffix else str(duration_value)
            else:  # int
                # For int type, allowed_values should be list[int]
                # Cast is safe because caller should provide list[int] when field_type is "int"
                _ = validate_duration(
                    duration_value, cast(Sequence[int], allowed_values), provider, model
                )
                return duration_value

        # No validation, just convert
        if field_type == "str":
            return f"{duration_value}s" if add_suffix else str(duration_value)
        else:
            return duration_value

    return FieldMapper(source_field="duration_seconds", converter=converter)


def single_image_field_mapper(
    required: bool = False,
    image_type: Literal[
        "reference", "first_frame", "last_frame", "asset", "style"
    ] = "reference",
    strict: bool = True,
) -> FieldMapper:
    """Create a FieldMapper for extracting single image from image_list.

    Validates that only 1 image of the specified type is provided and converts it to a URL string.

    Args:
        required: Whether this field is required (default: False)
        image_type: Type of image to extract (default: "reference")
        strict: If True, raises error when multiple images of type are found.
                If False, returns None when multiple images are found (default: True)

    Returns:
        FieldMapper for image_list -> image_url (single string)
    """

    def converter(_request: VideoGenerationRequest, value: object) -> str | None:
        if not value:
            return None

        # Cast to list of ImageListItem for type safety
        image_list = cast(list[ImageListItem], value)
        if len(image_list) == 0:
            return None

        # Filter images by the specified type
        # For "reference", also accept "first_frame" for backward compatibility
        if image_type == "reference":
            filtered_images: list[ImageListItem] = [
                img
                for img in image_list
                if img.get("type") in ("reference", "first_frame")
            ]
        else:
            filtered_images = [
                img for img in image_list if img.get("type") == image_type
            ]

        if len(filtered_images) == 0:
            return None

        if len(filtered_images) > 1:
            if strict:
                raise ValueError(
                    f"Only 1 {image_type} image allowed, got {len(filtered_images)} images",
                )
            else:
                # Multiple images found, but not in strict mode - return None
                # This allows reference_image_urls to handle them
                return None

        # Extract the image
        target_image = filtered_images[0]
        if "image" in target_image:
            media = target_image["image"]
            # Check if media is MediaContent (dict with 'content' and 'content_type')
            if (
                isinstance(media, dict)
                and "content" in media
                and "content_type" in media
            ):
                # Type narrowing: we know media is a dict with required keys
                media_content: MediaContent = {
                    "content": media["content"],
                    "content_type": media["content_type"],
                }
                return convert_to_data_url(media_content)
            return str(media)
        return str(target_image)

    return FieldMapper(
        source_field="image_list", converter=converter, required=required
    )


def image_list_field_mapper(
    image_type: Literal["reference", "first_frame", "last_frame", "asset", "style"]
    | None = None,
) -> FieldMapper:
    """Create a FieldMapper for converting image_list to list of URLs.

    Args:
        image_type: Optional filter to only include images of this type.
                    If None, includes all images.

    Returns:
        FieldMapper for image_list -> image_urls (list of strings)
    """

    def converter(_request: VideoGenerationRequest, value: object) -> list[str]:
        if not value:
            return []

        # Cast to list of ImageListItem for type safety
        image_list = cast(list[ImageListItem], value)

        # Filter by image type if specified
        filtered_items: list[ImageListItem]
        if image_type is not None:
            filtered_items = [
                item for item in image_list if item.get("type") == image_type
            ]
        else:
            filtered_items = image_list

        urls: list[str] = []
        for item in filtered_items:
            if "image" in item:
                media = item["image"]
                # Check if media is MediaContent (dict with 'content' and 'content_type')
                if (
                    isinstance(media, dict)
                    and "content" in media
                    and "content_type" in media
                ):
                    # Type narrowing: we know media is a dict with required keys
                    media_content: MediaContent = {
                        "content": media["content"],
                        "content_type": media["content_type"],
                    }
                    urls.append(convert_to_data_url(media_content))
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

    def converter(_request: VideoGenerationRequest, val: object) -> object:
        if isinstance(val, dict):
            extra_params = cast(dict[str, object], val)
            return extra_params.get(param_name)
        return None

    return FieldMapper(
        source_field="extra_params",
        converter=converter,
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

    def converter(_request: VideoGenerationRequest, val: object) -> str | None:
        # Check if val is MediaContent (dict with 'content' and 'content_type')
        if isinstance(val, dict) and "content" in val and "content_type" in val:
            # Type narrowing: we know val is a dict with required keys
            media_content: MediaContent = {
                "content": val["content"],
                "content_type": val["content_type"],
            }
            return convert_to_data_url(media_content)
        elif isinstance(val, str):
            # val is a URL string
            return val
        elif val is not None:
            # val is some other type - convert to string
            return f"{val}"
        return None

    return FieldMapper(
        source_field="video",
        converter=converter,
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
