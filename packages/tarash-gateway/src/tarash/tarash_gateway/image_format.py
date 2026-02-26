"""Image format detection, validation, and conversion utilities.

Provides a unified way to detect what format an image is in (URL, base64, or raw bytes)
and convert between formats when a provider requires a specific input type.
"""

from enum import Enum

from tarash.tarash_gateway.exceptions import ValidationError
from tarash.tarash_gateway.models import MediaContent, MediaType
from tarash.tarash_gateway.utils import convert_to_data_url, download_media_from_url


class ImageInputFormat(str, Enum):
    """Supported image input formats for provider APIs."""

    URL = "url"  # http/https URL string
    BASE64 = "base64"  # data:image/...;base64,... string (or raw base64 string)
    BYTES = "bytes"  # MediaContent dict {content, content_type}


def detect_image_format(media: MediaType) -> ImageInputFormat:
    """Detect the format of a media value.

    Args:
        media: A MediaType value (Base64 str, HttpUrl, or MediaContent dict)

    Returns:
        The detected ImageInputFormat
    """
    # MediaContent dict with content + content_type → BYTES
    if isinstance(media, dict) and "content" in media and "content_type" in media:
        return ImageInputFormat.BYTES

    # String types
    if isinstance(media, str):
        if media.startswith(("http://", "https://")):
            return ImageInputFormat.URL
        if media.startswith("data:"):
            return ImageInputFormat.BASE64
        # Raw base64 string without data: prefix
        return ImageInputFormat.BASE64

    # Pydantic HttpUrl objects
    url_str = str(media)
    if url_str.startswith(("http://", "https://")):
        return ImageInputFormat.URL

    return ImageInputFormat.BASE64


def _base64_to_bytes(media: str) -> MediaContent:
    """Convert a base64 data URL string to MediaContent bytes.

    Args:
        media: A data URL string like "data:image/png;base64,..."

    Returns:
        MediaContent dict with decoded bytes and content type
    """
    import base64

    if media.startswith("data:"):
        # Parse data:image/png;base64,<data>
        header, _, b64_data = media.partition(",")
        # header = "data:image/png;base64"
        content_type = header.split(":")[1].split(";")[0]
    else:
        # Raw base64 without prefix — assume image/png
        b64_data = media
        content_type = "image/png"

    content = base64.b64decode(b64_data)
    return MediaContent(content=content, content_type=content_type)


def ensure_image_format(
    media: MediaType,
    accepted_formats: list[ImageInputFormat],
    provider: str = "unknown",
) -> MediaType:
    """Validate and convert an image to one of the accepted formats.

    If the current format is already accepted, returns the media as-is.
    Otherwise, applies the appropriate conversion. Raises ValidationError
    for impossible conversions (e.g., base64/bytes → URL).

    Args:
        media: The image media value
        accepted_formats: List of formats the provider accepts
        provider: Provider name for error messages

    Returns:
        The media value in an accepted format

    Raises:
        ValidationError: If conversion is impossible
    """
    current = detect_image_format(media)

    if current in accepted_formats:
        return media

    # Try conversions in order of preference
    # BYTES → BASE64
    if (
        current == ImageInputFormat.BYTES
        and ImageInputFormat.BASE64 in accepted_formats
    ):
        media_content = MediaContent(
            content=media["content"],  # type: ignore[index]
            content_type=media["content_type"],  # type: ignore[index]
        )
        return convert_to_data_url(media_content)

    # BASE64 → BYTES
    if (
        current == ImageInputFormat.BASE64
        and ImageInputFormat.BYTES in accepted_formats
    ):
        return _base64_to_bytes(str(media))

    # URL → BASE64 (download then convert)
    if current == ImageInputFormat.URL and ImageInputFormat.BASE64 in accepted_formats:
        url_str = str(media)
        content_bytes, content_type = download_media_from_url(
            url_str, provider=provider
        )
        media_content = MediaContent(content=content_bytes, content_type=content_type)
        return convert_to_data_url(media_content)

    # URL → BYTES (download)
    if current == ImageInputFormat.URL and ImageInputFormat.BYTES in accepted_formats:
        url_str = str(media)
        content_bytes, content_type = download_media_from_url(
            url_str, provider=provider
        )
        return MediaContent(content=content_bytes, content_type=content_type)

    # Impossible conversions: BASE64/BYTES → URL
    if ImageInputFormat.URL in accepted_formats and current in (
        ImageInputFormat.BASE64,
        ImageInputFormat.BYTES,
    ):
        raise ValidationError(
            f"Cannot convert {current.value} image to URL. "
            f"Provider '{provider}' only accepts URL images. "
            f"Please provide an image URL instead.",
            provider=provider,
        )

    # Catch-all for unexpected cases
    accepted_str = ", ".join(f.value for f in accepted_formats)
    raise ValidationError(
        f"Cannot convert {current.value} image to any accepted format ({accepted_str}) "
        f"for provider '{provider}'.",
        provider=provider,
    )
