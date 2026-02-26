"""Tests for image format detection, validation, and conversion."""

from unittest.mock import patch

import pytest

from tarash.tarash_gateway.exceptions import ValidationError
from tarash.tarash_gateway.image_format import (
    ImageInputFormat,
    _base64_to_bytes,
    detect_image_format,
    ensure_image_format,
)
from tarash.tarash_gateway.models import MediaContent


# ==================== detect_image_format ====================


def test_detect_format_http_url():
    """HTTP URL string is detected as URL format."""
    assert detect_image_format("http://example.com/image.png") == ImageInputFormat.URL


def test_detect_format_https_url():
    """HTTPS URL string is detected as URL format."""
    assert detect_image_format("https://example.com/image.png") == ImageInputFormat.URL


def test_detect_format_base64_data_url():
    """Data URL with base64 content is detected as BASE64 format."""
    data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
    assert detect_image_format(data_url) == ImageInputFormat.BASE64


def test_detect_format_raw_base64():
    """Raw base64 string (no data: prefix) is detected as BASE64 format."""
    assert detect_image_format("iVBORw0KGgoAAAANSUhEUg==") == ImageInputFormat.BASE64


def test_detect_format_bytes():
    """MediaContent dict is detected as BYTES format."""
    media = MediaContent(content=b"\x89PNG\r\n", content_type="image/png")
    assert detect_image_format(media) == ImageInputFormat.BYTES


def test_detect_format_pydantic_httpurl():
    """Pydantic HttpUrl object is detected as URL format."""
    from pydantic import HttpUrl

    url = HttpUrl("https://example.com/image.jpg")
    assert detect_image_format(url) == ImageInputFormat.URL


# ==================== _base64_to_bytes ====================


def test_base64_to_bytes_data_url():
    """Data URL is decoded to MediaContent bytes."""
    import base64

    original = b"test image data"
    b64 = base64.b64encode(original).decode()
    data_url = f"data:image/jpeg;base64,{b64}"

    result = _base64_to_bytes(data_url)
    assert result["content"] == original
    assert result["content_type"] == "image/jpeg"


def test_base64_to_bytes_raw_base64():
    """Raw base64 string defaults to image/png content type."""
    import base64

    original = b"test image data"
    b64 = base64.b64encode(original).decode()

    result = _base64_to_bytes(b64)
    assert result["content"] == original
    assert result["content_type"] == "image/png"


# ==================== ensure_image_format ====================


def test_ensure_passthrough_when_already_accepted():
    """Media is returned as-is when already in an accepted format."""
    url = "https://example.com/image.png"
    result = ensure_image_format(url, [ImageInputFormat.URL])
    assert result == url


def test_ensure_bytes_to_base64():
    """BYTES media is converted to base64 data URL."""
    media = MediaContent(content=b"\x89PNG\r\n", content_type="image/png")
    result = ensure_image_format(media, [ImageInputFormat.BASE64])
    assert isinstance(result, str)
    assert result.startswith("data:image/png;base64,")


def test_ensure_base64_to_bytes():
    """BASE64 data URL is converted to MediaContent bytes."""
    import base64

    original = b"test image data"
    b64 = base64.b64encode(original).decode()
    data_url = f"data:image/jpeg;base64,{b64}"

    result = ensure_image_format(data_url, [ImageInputFormat.BYTES])
    assert isinstance(result, dict)
    assert result["content"] == original
    assert result["content_type"] == "image/jpeg"


def test_ensure_url_to_base64():
    """URL is downloaded and converted to base64 data URL."""
    with patch(
        "tarash.tarash_gateway.image_format.download_media_from_url"
    ) as mock_download:
        mock_download.return_value = (b"fake image bytes", "image/png")

        result = ensure_image_format(
            "https://example.com/image.png",
            [ImageInputFormat.BASE64],
            provider="test",
        )

        mock_download.assert_called_once_with(
            "https://example.com/image.png", provider="test"
        )
        assert isinstance(result, str)
        assert result.startswith("data:image/png;base64,")


def test_ensure_url_to_bytes():
    """URL is downloaded and returned as MediaContent bytes."""
    with patch(
        "tarash.tarash_gateway.image_format.download_media_from_url"
    ) as mock_download:
        mock_download.return_value = (b"fake image bytes", "image/png")

        result = ensure_image_format(
            "https://example.com/image.png",
            [ImageInputFormat.BYTES],
            provider="test",
        )

        mock_download.assert_called_once_with(
            "https://example.com/image.png", provider="test"
        )
        assert isinstance(result, dict)
        assert result["content"] == b"fake image bytes"
        assert result["content_type"] == "image/png"


def test_ensure_base64_to_url_raises_validation_error():
    """Converting BASE64 to URL is impossible and raises ValidationError."""
    data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
    with pytest.raises(ValidationError, match="Cannot convert base64 image to URL"):
        ensure_image_format(data_url, [ImageInputFormat.URL], provider="luma")


def test_ensure_bytes_to_url_raises_validation_error():
    """Converting BYTES to URL is impossible and raises ValidationError."""
    media = MediaContent(content=b"\x89PNG\r\n", content_type="image/png")
    with pytest.raises(ValidationError, match="Cannot convert bytes image to URL"):
        ensure_image_format(media, [ImageInputFormat.URL], provider="luma")


def test_ensure_validation_error_includes_provider():
    """ValidationError includes the provider name in the message."""
    data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
    with pytest.raises(ValidationError) as exc_info:
        ensure_image_format(data_url, [ImageInputFormat.URL], provider="luma")

    assert exc_info.value.provider == "luma"
    assert "luma" in str(exc_info.value.message)


def test_ensure_prefers_first_accepted_conversion():
    """When input is not in accepted formats, converts to first viable format."""
    media = MediaContent(content=b"\x89PNG\r\n", content_type="image/png")
    # Only BASE64 is accepted; BYTES→BASE64 conversion should be used
    result = ensure_image_format(media, [ImageInputFormat.BASE64])
    assert isinstance(result, str)
    assert result.startswith("data:image/png;base64,")


# ==================== Field mapper integration ====================


def test_single_image_field_mapper_rejects_base64_for_url_only_model():
    """single_image_field_mapper with accepted_formats=[URL] rejects base64."""
    from tarash.tarash_gateway.providers.field_mappers import single_image_field_mapper

    mapper = single_image_field_mapper(
        accepted_formats=[ImageInputFormat.URL],
        provider="luma",
    )

    image_list = [
        {"type": "reference", "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="}
    ]

    with pytest.raises(ValidationError, match="Cannot convert base64 image to URL"):
        mapper.converter(None, image_list)


def test_single_image_field_mapper_accepts_url_for_url_only_model():
    """single_image_field_mapper with accepted_formats=[URL] accepts URLs."""
    from tarash.tarash_gateway.providers.field_mappers import single_image_field_mapper

    mapper = single_image_field_mapper(
        accepted_formats=[ImageInputFormat.URL],
        provider="luma",
    )

    image_list = [{"type": "reference", "image": "https://example.com/image.png"}]
    result = mapper.converter(None, image_list)
    assert result == "https://example.com/image.png"


def test_image_list_field_mapper_converts_bytes_to_base64():
    """image_list_field_mapper with accepted_formats=[URL, BASE64] converts bytes."""
    from tarash.tarash_gateway.providers.field_mappers import image_list_field_mapper

    mapper = image_list_field_mapper(
        accepted_formats=[ImageInputFormat.URL, ImageInputFormat.BASE64],
        provider="fal",
    )

    image_list = [
        {
            "type": "reference",
            "image": MediaContent(content=b"\x89PNG\r\n", content_type="image/png"),
        }
    ]

    result = mapper.converter(None, image_list)
    assert len(result) == 1
    assert result[0].startswith("data:image/png;base64,")
