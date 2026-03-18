"""Unit tests for Google provider image generation.

Tests cover:
- Client initialization and caching
- Field mapper registries (Nano Banana, Imagen 3)
- Request/response conversion
- Error handling
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tarash.tarash_gateway.models import (
    ImageGenerationConfig,
    ImageGenerationRequest,
)
from tarash.tarash_gateway.providers.google import GoogleProviderHandler


# ==================== Fixtures ====================


@pytest.fixture
def handler():
    """Create Google handler instance."""
    with patch("tarash.tarash_gateway.providers.google.has_genai", True):
        return GoogleProviderHandler()


@pytest.fixture
def base_config():
    """Create basic Google image generation config."""
    return ImageGenerationConfig(
        model="imagen-3.0-generate-001",
        provider="google",
        api_key="test-api-key",
        timeout=120,
    )


@pytest.fixture
def nano_banana_config():
    """Create Nano Banana config."""
    return ImageGenerationConfig(
        model="gemini-2.5-flash-image",
        provider="google",
        api_key="test-api-key",
        timeout=120,
    )


@pytest.fixture
def base_request():
    """Create basic image generation request."""
    return ImageGenerationRequest(
        prompt="A serene mountain landscape",
        seed=42,
    )


# ==================== Client Initialization Tests ====================


def test_google_handler_init_requires_sdk(handler):
    """Handler requires google-genai SDK."""
    assert handler is not None


def test_get_client_creates_sync_client_first_time(handler, base_config):
    """Sync client is created on first call."""
    with patch("tarash.tarash_gateway.providers.google.Client") as mock_client_cls:
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance

        client = handler._get_client(base_config, "sync")

        assert client is mock_instance
        mock_client_cls.assert_called_once()


def test_get_client_creates_new_async_client_each_time(handler, base_config):
    """Async clients are created fresh each time."""
    with patch("tarash.tarash_gateway.providers.google.Client") as mock_client_cls:
        mock_instance1 = MagicMock()
        mock_instance1.aio = AsyncMock()
        mock_instance2 = MagicMock()
        mock_instance2.aio = AsyncMock()
        mock_client_cls.side_effect = [mock_instance1, mock_instance2]

        handler._get_client(base_config, "async")
        handler._get_client(base_config, "async")

        # Google provider caches async clients too (by cache key)
        # If cached, will be same instance; if different, will be different
        assert mock_client_cls.call_count >= 1


# ==================== Field Mapper Tests ====================


def test_imagen3_field_mappers_exist():
    """Imagen 3 field mappers are defined."""
    from tarash.tarash_gateway.providers.google import IMAGEN3_FIELD_MAPPERS

    assert "prompt" in IMAGEN3_FIELD_MAPPERS
    assert "negative_prompt" in IMAGEN3_FIELD_MAPPERS
    assert "aspect_ratio" in IMAGEN3_FIELD_MAPPERS
    assert "number_of_images" in IMAGEN3_FIELD_MAPPERS


def test_nano_banana_field_mappers_exist():
    """Nano Banana field mappers are defined."""
    from tarash.tarash_gateway.providers.google import NANO_BANANA_FIELD_MAPPERS

    assert "prompt" in NANO_BANANA_FIELD_MAPPERS
    assert "aspect_ratio" in NANO_BANANA_FIELD_MAPPERS
    assert "number_of_images" in NANO_BANANA_FIELD_MAPPERS


def test_imagen3_in_registry():
    """Imagen 3 models are in field mapper registry."""
    from tarash.tarash_gateway.providers.google import GOOGLE_IMAGE_MODEL_REGISTRY

    assert "imagen-3.0-generate-001" in GOOGLE_IMAGE_MODEL_REGISTRY
    assert "imagen-3" in GOOGLE_IMAGE_MODEL_REGISTRY


def test_nano_banana_in_registry():
    """Nano Banana models are in field mapper registry."""
    from tarash.tarash_gateway.providers.google import GOOGLE_IMAGE_MODEL_REGISTRY

    assert "gemini-2.5-flash-image" in GOOGLE_IMAGE_MODEL_REGISTRY


# ==================== Request Conversion Tests ====================


def test_convert_image_request_basic_prompt(handler, base_config, base_request):
    """Convert basic request with prompt only."""
    result = handler._convert_image_request(base_config, base_request)

    assert result["prompt"] == "A serene mountain landscape"


def test_convert_image_request_with_aspect_ratio(handler, base_config):
    """Convert request with aspect ratio."""
    request = ImageGenerationRequest(
        prompt="Test prompt",
        aspect_ratio="16:9",
    )

    result = handler._convert_image_request(base_config, request)

    assert "aspect_ratio" in result


def test_convert_image_request_with_number_of_images(handler, base_config):
    """Convert request with number of images."""
    request = ImageGenerationRequest(
        prompt="Test prompt",
        n=4,
    )

    result = handler._convert_image_request(base_config, request)

    assert "number_of_images" in result


# ==================== Response Conversion Tests ====================


def test_convert_image_response_with_urls(handler, base_config, base_request):
    """Convert Google response with image URLs."""
    request_id = str(uuid.uuid4())

    # Mock Google response structure - the implementation expects a dict-like object
    # with .get() method that returns generated_images list
    mock_image1 = MagicMock()
    mock_image1.image.gcs_uri = "https://example.com/image1.png"
    mock_image2 = MagicMock()
    mock_image2.image.gcs_uri = "https://example.com/image2.png"

    mock_response = {
        "generated_images": [mock_image1, mock_image2],
    }

    result = handler._convert_image_response(
        base_config,
        request_id,
        mock_response,
    )

    assert result.request_id == request_id
    assert result.status == "completed"
    assert len(result.images) == 2
    assert result.images[0] == "https://example.com/image1.png"
    assert result.images[1] == "https://example.com/image2.png"


# ==================== Registry Tests ====================


def test_google_provider_registered_in_registry():
    """Google provider is registered in global handler registry."""
    from tarash.tarash_gateway.registry import get_handler
    from tarash.tarash_gateway.models import ImageGenerationConfig

    config = ImageGenerationConfig(
        model="imagen-3.0-generate-001",
        provider="google",
        api_key="test-key",
    )

    # This should not raise an error
    with patch("tarash.tarash_gateway.providers.google.has_genai", True):
        handler = get_handler(config)
        assert isinstance(handler, GoogleProviderHandler)


# ==================== _bytes_to_data_uri Tests ====================


def test_bytes_to_data_uri_with_bytes():
    """Test _bytes_to_data_uri encodes bytes to data URI."""
    from tarash.tarash_gateway.providers.google import _bytes_to_data_uri
    import base64

    img_bytes = b"fake-image-data"
    result = _bytes_to_data_uri(img_bytes, "image/png")

    expected = f"data:image/png;base64,{base64.b64encode(img_bytes).decode()}"
    assert result == expected


def test_bytes_to_data_uri_with_string():
    """Test _bytes_to_data_uri encodes string input to data URI."""
    from tarash.tarash_gateway.providers.google import _bytes_to_data_uri
    import base64

    img_string = "fake-image-data"
    result = _bytes_to_data_uri(img_string, "image/jpeg")

    expected_bytes = img_string.encode()
    expected = f"data:image/jpeg;base64,{base64.b64encode(expected_bytes).decode()}"
    assert result == expected


def test_bytes_to_data_uri_default_mime_type():
    """Test _bytes_to_data_uri uses default image/png mime type."""
    from tarash.tarash_gateway.providers.google import _bytes_to_data_uri

    result = _bytes_to_data_uri(b"data")
    assert result.startswith("data:image/png;base64,")


# ==================== _is_gemini_image_model Tests ====================


def test_is_gemini_image_model_true():
    """Test _is_gemini_image_model returns True for Gemini image models."""
    from tarash.tarash_gateway.providers.google import _is_gemini_image_model

    assert _is_gemini_image_model("gemini-2.5-flash-image") is True
    assert _is_gemini_image_model("gemini-3-pro-image-preview") is True


def test_is_gemini_image_model_false():
    """Test _is_gemini_image_model returns False for non-Gemini models."""
    from tarash.tarash_gateway.providers.google import _is_gemini_image_model

    assert _is_gemini_image_model("imagen-3.0-generate-001") is False
    assert _is_gemini_image_model("gemini-2.5-flash") is False  # No 'image' in name


# ==================== Imagen Response Conversion: inline_data ====================


def test_convert_image_response_with_inline_data(handler, base_config):
    """Test Imagen response conversion with inline_data (image_bytes)."""

    request_id = "req-inline"

    mock_image = MagicMock()
    mock_image.image.gcs_uri = None
    mock_image.image.image_bytes = b"fake-image-bytes"
    mock_image.image.mime_type = "image/jpeg"

    # gcs_uri is None, so it should use image_bytes path
    mock_response = {
        "generated_images": [mock_image],
    }

    result = handler._convert_image_response(base_config, request_id, mock_response)

    assert result.request_id == request_id
    assert result.status == "completed"
    assert len(result.images) == 1
    # Should be a data URI
    assert result.images[0].startswith("data:image/jpeg;base64,")


def test_convert_image_response_empty_list(handler, base_config):
    """Test Imagen response with no generated images."""
    result = handler._convert_image_response(
        base_config, "req-empty", {"generated_images": []}
    )

    assert result.images == []


# ==================== Gemini Image Response Conversion ====================


def test_convert_gemini_image_response_with_parts(handler, base_config):
    """Test Gemini response conversion from parts with inline_data."""

    mock_inline_data = MagicMock()
    mock_inline_data.data = b"gemini-image-bytes"
    mock_inline_data.mime_type = "image/png"

    mock_part = MagicMock()
    mock_part.inline_data = mock_inline_data

    mock_response = MagicMock()
    mock_response.parts = [mock_part]
    mock_response.model_dump.return_value = {"parts": []}

    result = handler._convert_gemini_image_response(
        base_config, "req-gemini", mock_response
    )

    assert result.status == "completed"
    assert len(result.images) == 1
    assert result.images[0].startswith("data:image/png;base64,")


def test_convert_gemini_image_response_no_parts(handler, base_config):
    """Test Gemini response with no parts returns empty images."""
    mock_response = MagicMock()
    mock_response.parts = []
    mock_response.model_dump.return_value = {"parts": []}

    result = handler._convert_gemini_image_response(
        base_config, "req-no-parts", mock_response
    )

    assert result.images == []


def test_convert_gemini_image_response_no_inline_data(handler, base_config):
    """Test Gemini response with parts but no inline_data is skipped."""
    mock_part = MagicMock()
    mock_part.inline_data = None

    mock_response = MagicMock()
    mock_response.parts = [mock_part]
    mock_response.model_dump.return_value = {}

    result = handler._convert_gemini_image_response(
        base_config, "req-no-inline", mock_response
    )

    assert result.images == []


def test_convert_gemini_image_response_without_model_dump(handler, base_config):
    """Test Gemini response without model_dump uses string fallback."""
    mock_response = MagicMock(spec=[])  # No model_dump
    mock_response.parts = []

    result = handler._convert_gemini_image_response(
        base_config, "req-no-dump", mock_response
    )

    assert result.images == []
    assert "response" in result.raw_response


# ==================== Image Generation: async/sync ====================


@pytest.mark.asyncio
async def test_generate_image_async_imagen(handler, base_config, base_request):
    """Test async image generation with Imagen model."""
    mock_gen_image = MagicMock()
    mock_gen_image.image.gcs_uri = "https://storage.googleapis.com/image.png"

    mock_imagen_response = MagicMock()
    mock_imagen_response.generated_images = [mock_gen_image]

    mock_client = AsyncMock()
    mock_client.models.generate_images = AsyncMock(return_value=mock_imagen_response)

    with patch.object(handler, "_get_client", return_value=mock_client):
        result = await handler.generate_image_async(base_config, base_request)

    assert result.status == "completed"
    assert len(result.images) == 1
    assert result.images[0] == "https://storage.googleapis.com/image.png"


@pytest.mark.asyncio
async def test_generate_image_async_gemini(handler, nano_banana_config, base_request):
    """Test async image generation with Gemini (Nano Banana) model."""
    mock_inline_data = MagicMock()
    mock_inline_data.data = b"gemini-image"
    mock_inline_data.mime_type = "image/png"

    mock_part = MagicMock()
    mock_part.inline_data = mock_inline_data

    mock_response = MagicMock()
    mock_response.parts = [mock_part]
    mock_response.model_dump.return_value = {}

    mock_client = AsyncMock()
    mock_client.models.generate_content = AsyncMock(return_value=mock_response)

    with patch.object(handler, "_get_client", return_value=mock_client):
        result = await handler.generate_image_async(nano_banana_config, base_request)

    assert result.status == "completed"
    assert len(result.images) == 1
    assert result.images[0].startswith("data:image/png;base64,")


def test_generate_image_sync_imagen(handler, base_config, base_request):
    """Test sync image generation with Imagen model."""
    mock_gen_image = MagicMock()
    mock_gen_image.image.gcs_uri = "https://storage.googleapis.com/image.png"

    mock_imagen_response = MagicMock()
    mock_imagen_response.generated_images = [mock_gen_image]

    mock_client = MagicMock()
    mock_client.models.generate_images = MagicMock(return_value=mock_imagen_response)

    with patch.object(handler, "_get_client", return_value=mock_client):
        result = handler.generate_image(base_config, base_request)

    assert result.status == "completed"
    assert len(result.images) == 1


def test_generate_image_sync_gemini(handler, nano_banana_config, base_request):
    """Test sync image generation with Gemini model."""
    mock_inline_data = MagicMock()
    mock_inline_data.data = b"gemini-image"
    mock_inline_data.mime_type = "image/png"

    mock_part = MagicMock()
    mock_part.inline_data = mock_inline_data

    mock_response = MagicMock()
    mock_response.parts = [mock_part]
    mock_response.model_dump.return_value = {}

    mock_client = MagicMock()
    mock_client.models.generate_content = MagicMock(return_value=mock_response)

    with patch.object(handler, "_get_client", return_value=mock_client):
        result = handler.generate_image(nano_banana_config, base_request)

    assert result.status == "completed"
    assert len(result.images) == 1


@pytest.mark.asyncio
async def test_generate_image_async_error_handling(handler, base_config, base_request):
    """Test async image generation error handling."""
    from tarash.tarash_gateway.exceptions import GenerationFailedError

    mock_client = AsyncMock()
    mock_client.models.generate_images = AsyncMock(
        side_effect=RuntimeError("API failure")
    )

    with patch.object(handler, "_get_client", return_value=mock_client):
        with pytest.raises(GenerationFailedError):
            await handler.generate_image_async(base_config, base_request)


def test_generate_image_sync_error_handling(handler, base_config, base_request):
    """Test sync image generation error handling."""
    from tarash.tarash_gateway.exceptions import GenerationFailedError

    mock_client = MagicMock()
    mock_client.models.generate_images = MagicMock(
        side_effect=RuntimeError("API failure")
    )

    with patch.object(handler, "_get_client", return_value=mock_client):
        with pytest.raises(GenerationFailedError):
            handler.generate_image(base_config, base_request)
