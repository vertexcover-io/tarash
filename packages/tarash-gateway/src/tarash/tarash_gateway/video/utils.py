"""Utility functions for video generation."""

import base64
from typing import Any, Type
from urllib.parse import urlparse

import httpx
from pydantic import TypeAdapter
from typing_extensions import TypedDict

from tarash.tarash_gateway.video.exceptions import ProviderAPIError, ValidationError


def validate_model_params(
    *,
    schema: Type[TypedDict],
    data: dict[str, Any],
    provider: str,
    model: str,
) -> dict[str, Any]:
    """
    Validate data against a TypedDict adapter and return validated dict.

    Uses Pydantic's TypeAdapter for TypedDict validation.
    Ensures __pydantic_config__ is set to forbid extra fields if not already set.

    Args:
        schema: TypedDict class to validate against
        data: Dictionary of data to validate
        provider: Provider name for error messages
        model: Optional model name for error messages

    Returns:
        Validated dictionary with excluded None values

    Raises:
        ValidationError: If validation fails
    """
    try:
        # Use Pydantic's TypeAdapter for TypedDict validation
        adapter = TypeAdapter(schema)
        validated = adapter.validate_python(data)

        # Remove None values from the validated dict
        return {k: v for k, v in validated.items() if v is not None}
    except Exception as e:
        raise ValidationError(
            f"Invalid model_params for {model}: {e}",
            provider=provider,
        ) from e


def convert_to_data_url(media_content: dict[str, Any]) -> str:
    """
    Convert MediaContent dict to base64 data URL.

    Args:
        media_content: Dictionary with 'content' (bytes) and 'content_type' (str) keys

    Returns:
        Base64 data URL string in format: data:{content_type};base64,{base64_encoded_content}
    """
    img_base64 = base64.b64encode(media_content["content"]).decode("utf-8")
    return f"data:{media_content['content_type']};base64,{img_base64}"


def download_media_from_url(url: str, provider: str = "unknown") -> tuple[bytes, str]:
    """
    Download media (image/video) from URL and return bytes with content type.

    Args:
        url: URL to download from
        provider: Provider name for error messages

    Returns:
        Tuple of (content_bytes, content_type)

    Raises:
        ProviderAPIError: If download fails
    """
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url)
            response.raise_for_status()

            content_type = response.headers.get(
                "content-type", "application/octet-stream"
            )
            return response.content, content_type
    except Exception as e:
        raise ProviderAPIError(
            f"Failed to download media from URL: {e}",
            provider=provider,
            raw_response={"url": url, "error": str(e)},
        ) from e


def get_filename_from_url(url: str) -> str:
    """
    Extract filename from URL path.

    Args:
        url: URL to extract filename from

    Returns:
        Filename or 'media' if extraction fails
    """
    try:
        parsed = urlparse(url)
        path = parsed.path
        if path and "/" in path:
            filename = path.split("/")[-1]
            if filename:
                return filename
    except Exception:
        pass
    return "media"
