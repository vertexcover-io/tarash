"""Unit tests for Google provider image generation.

Tests cover:
- Client initialization and caching
- Field mapper registries (Nano Banana, Imagen 3)
- Request/response conversion
- Error handling
"""

import uuid
from unittest.mock import MagicMock, patch

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
        model="gemini-2.5-flash-image-001",
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
def test_get_client_creates_sync_client_first_time(handler, base_config):
    """Sync client is created on first call."""

    with patch("tarash.tarash_gateway.providers.google.Client") as mock_client_cls:
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance

        client = handler._get_client(base_config, "sync")

        assert client is mock_instance
        mock_client_cls.assert_called_once()


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
        base_request,
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
