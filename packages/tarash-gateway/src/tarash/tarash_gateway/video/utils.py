"""Utility functions for video generation."""

import base64
from typing import cast
from urllib.parse import urlparse

import httpx
from pydantic import TypeAdapter

from tarash.tarash_gateway.logging import log_debug, log_error
from tarash.tarash_gateway.video.exceptions import (
    HTTPError,
    TarashException,
    ValidationError,
)
from tarash.tarash_gateway.video.models import MediaContent


def validate_model_params(
    *,
    schema: type[object],  # TypedDict type
    data: dict[str, object],
    provider: str,
    model: str,
) -> dict[str, object]:
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
        validated = cast(dict[str, object], adapter.validate_python(data))

        # Remove None values from the validated dict
        return {k: v for k, v in validated.items() if v is not None}
    except Exception as e:
        raise ValidationError(
            f"Invalid model_params for {model}: {e}",
            provider=provider,
        ) from e


def convert_to_data_url(media_content: MediaContent) -> str:
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
    log_debug(
        "Downloading media from URL",
        context={"provider": provider, "url": url},
        logger_name="tarash.tarash_gateway.video.utils",
    )
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url)
            response = response.raise_for_status()

            content_type: str = cast(
                str, response.headers.get("content-type", "application/octet-stream")
            )
            content_size = len(response.content)
            log_debug(
                "Media downloaded successfully",
                context={
                    "provider": provider,
                    "url": url,
                    "content_type": content_type,
                    "content_size_bytes": content_size,
                },
                logger_name="tarash.tarash_gateway.video.utils",
            )
            return response.content, content_type
    except httpx.HTTPStatusError as e:
        log_error(
            f"Failed to download media from URL - HTTP error with status code {e.response.status_code}: {e}",
            context={
                "provider": provider,
                "url": url,
                "status_code": e.response.status_code,
            },
            logger_name="tarash.tarash_gateway.video.utils",
        )
        raise HTTPError(
            f"Failed to download media from URL: {e}",
            provider=provider,
            raw_response={"url": url, "response": e.response.content},
            status_code=e.response.status_code,
        ) from e
    except Exception as e:
        log_error(
            "Failed to download media from URL",
            context={"provider": provider, "url": url},
            logger_name="tarash.tarash_gateway.video.utils",
            exc_info=True,
        )
        raise TarashException(
            f"Failed to download media from URL: {e}",
            provider=provider,
            raw_response={"url": url, "exception": str(e)},
        ) from e


async def download_media_from_url_async(
    url: str, provider: str = "unknown"
) -> tuple[bytes, str]:
    """
    Download media (image/video) from URL asynchronously and return bytes with content type.

    Args:
        url: URL to download from
        provider: Provider name for error messages

    Returns:
        Tuple of (content_bytes, content_type)

    Raises:
        HTTPError: If download fails with HTTP error
        TarashException: If download fails for other reasons
    """
    log_debug(
        "Downloading media from URL (async)",
        context={"provider": provider, "url": url},
        logger_name="tarash.tarash_gateway.video.utils",
    )
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response = response.raise_for_status()

            content_type: str = cast(
                str, response.headers.get("content-type", "application/octet-stream")
            )
            content_size = len(response.content)
            log_debug(
                "Media downloaded successfully (async)",
                context={
                    "provider": provider,
                    "url": url,
                    "content_type": content_type,
                    "content_size_bytes": content_size,
                },
                logger_name="tarash.tarash_gateway.video.utils",
            )
            return response.content, content_type
    except httpx.HTTPStatusError as e:
        log_error(
            f"Failed to download media from URL - HTTP error with status code {e.response.status_code}: {e}",
            context={
                "provider": provider,
                "url": url,
                "status_code": e.response.status_code,
            },
            logger_name="tarash.tarash_gateway.video.utils",
        )
        raise HTTPError(
            f"Failed to download media from URL: {e}",
            provider=provider,
            raw_response={"url": url, "response": e.response.content},
            status_code=e.response.status_code,
        ) from e
    except Exception as e:
        log_error(
            "Failed to download media from URL (async)",
            context={"provider": provider, "url": url},
            logger_name="tarash.tarash_gateway.video.utils",
            exc_info=True,
        )
        raise TarashException(
            f"Failed to download media from URL: {e}",
            provider=provider,
            raw_response={"url": url, "exception": str(e)},
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


def validate_duration(
    duration_seconds: int | None,
    allowed_values: list[int],
    provider: str,
    model: str | None = None,
) -> int | None:
    """
    Validate duration against allowed values.

    Args:
        duration_seconds: Duration in seconds to validate
        allowed_values: List of allowed duration values
        provider: Provider name for error messages
        model: Optional model name for error messages

    Returns:
        The validated duration value or None if input is None

    Raises:
        ValidationError: If duration is not in allowed values

    Example:
        >>> validate_duration(5, [4, 8, 12], "openai", "sora-2")
        ValidationError: Invalid duration for openai (sora-2): 5 seconds. Allowed values: 4, 8, 12
    """
    if duration_seconds is None:
        return None

    if duration_seconds not in allowed_values:
        model_info = f" ({model})" if model else ""
        raise ValidationError(
            f"Invalid duration for {provider}{model_info}: {duration_seconds} seconds. Allowed values: {', '.join(map(str, allowed_values))}",
            provider=provider,
        )

    return duration_seconds
