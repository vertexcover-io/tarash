"""Unit tests for OpenAI provider handler (image generation)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tarash.tarash_gateway.models import (
    ImageGenerationConfig,
    ImageGenerationRequest,
)
from tarash.tarash_gateway.providers.openai import OpenAIProviderHandler


# ==================== Fixtures ====================


@pytest.fixture
def handler():
    """Create handler instance for testing."""
    return OpenAIProviderHandler()


@pytest.fixture
def base_config():
    """Create test config."""
    return ImageGenerationConfig(
        model="dall-e-3",
        provider="openai",
        api_key="test-api-key",
        timeout=120,
        max_poll_attempts=60,
        poll_interval=2,
    )


@pytest.fixture
def base_request():
    """Create test request."""
    return ImageGenerationRequest(
        prompt="A serene mountain landscape",
    )


# ==================== Client Caching Tests ====================


def test_handler_initialization(handler):
    """Test handler initialization creates empty caches."""
    assert handler._sync_client_cache == {}
    assert handler._async_client_cache == {}


def test_get_client_creates_and_caches_sync_client(handler, base_config):
    """Test that sync clients are cached."""
    handler._sync_client_cache.clear()

    with patch("tarash.tarash_gateway.providers.openai.OpenAI") as mock_client_class:
        mock_instance = MagicMock()
        mock_client_class.return_value = mock_instance

        client1 = handler._get_client(base_config, "sync")
        client2 = handler._get_client(base_config, "sync")

        # Should be same instance (cached)
        assert client1 is client2
        # Should only create once
        assert mock_client_class.call_count == 1


def test_get_client_creates_and_caches_async_client(handler, base_config):
    """Test that async clients are cached (unlike Fal, OpenAI clients are safe to cache)."""
    handler._async_client_cache.clear()

    with patch(
        "tarash.tarash_gateway.providers.openai.AsyncOpenAI"
    ) as mock_client_class:
        mock_instance = AsyncMock()
        mock_client_class.return_value = mock_instance

        client1 = handler._get_client(base_config, "async")
        client2 = handler._get_client(base_config, "async")

        # Should be same instance (cached)
        assert client1 is client2
        # Should only create once
        assert mock_client_class.call_count == 1


def test_get_client_different_api_keys_use_different_cache(handler, base_config):
    """Test that different API keys create separate cache entries."""
    handler._sync_client_cache.clear()

    config2 = base_config.model_copy(update={"api_key": "different-key"})

    with patch("tarash.tarash_gateway.providers.openai.OpenAI") as mock_client_class:
        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_client_class.side_effect = [mock_instance1, mock_instance2]

        client1 = handler._get_client(base_config, "sync")
        client2 = handler._get_client(config2, "sync")

        # Should be different instances (different cache keys)
        assert client1 is not client2
        assert mock_client_class.call_count == 2


def test_get_client_uses_api_key_from_config(handler, base_config):
    """Test that client is created with API key from config."""
    handler._sync_client_cache.clear()

    with patch("tarash.tarash_gateway.providers.openai.OpenAI") as mock_client_class:
        handler._get_client(base_config, "sync")

        # Verify client was created with correct parameters
        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs["api_key"] == "test-api-key"
        assert call_kwargs["timeout"] == 120


# ==================== Field Mapper Tests ====================


def test_get_gpt_image_15_field_mappers():
    """Test that GPT-Image-1.5 field mappers exist."""
    from tarash.tarash_gateway.providers.openai import get_openai_image_field_mappers

    mappers = get_openai_image_field_mappers("gpt-image-1.5")

    # Check required fields
    assert "prompt" in mappers
    assert mappers["prompt"].required is True

    # Check optional fields
    assert "size" in mappers
    assert "quality" in mappers
    assert "n" in mappers
    assert "output_format" in mappers
    assert "background" in mappers


def test_gpt_image_15_field_mappers_apply_correctly():
    """Test that GPT-Image-1.5 field mappers convert request correctly.

    Note: output_format and background are not standard fields in ImageGenerationRequest,
    so they won't be in the field mapper output. They should be passed via extra_params
    and merged in _convert_image_request().
    """
    from tarash.tarash_gateway.providers.openai import get_openai_image_field_mappers
    from tarash.tarash_gateway.providers.field_mappers import apply_field_mappers

    mappers = get_openai_image_field_mappers("gpt-image-1.5")

    request = ImageGenerationRequest(
        prompt="A futuristic cityscape",
        size="1024x1024",
        quality="high",
        n=2,
    )

    result = apply_field_mappers(mappers, request)

    assert result["prompt"] == "A futuristic cityscape"
    assert result["size"] == "1024x1024"
    assert result["quality"] == "high"
    assert result["n"] == 2
    # output_format and background are not in field mapper output
    # They should be passed via extra_params and merged in _convert_image_request()


def test_gpt_image_15_with_extra_params(handler):
    """Test that GPT-Image-1.5 extra_params are merged correctly in _convert_image_request()."""
    config = ImageGenerationConfig(
        model="gpt-image-1.5",
        provider="openai",
        api_key="test-api-key",
        timeout=120,
    )

    request = ImageGenerationRequest(
        prompt="A futuristic cityscape",
        size="1024x1024",
        quality="high",
        n=2,
        extra_params={
            "output_format": "png",
            "background": "transparent",
        },
    )

    result = handler._convert_image_request(config, request)

    # Standard fields from mappers
    assert result["prompt"] == "A futuristic cityscape"
    assert result["size"] == "1024x1024"
    assert result["quality"] == "high"
    assert result["n"] == 2
    assert result["model"] == "gpt-image-1.5"

    # Extra params merged after field mapping
    assert result["output_format"] == "png"
    assert result["background"] == "transparent"


def test_get_dalle3_field_mappers():
    """Test that DALL-E 3 field mappers exist."""
    from tarash.tarash_gateway.providers.openai import get_openai_image_field_mappers

    mappers = get_openai_image_field_mappers("dall-e-3")

    # Check required fields
    assert "prompt" in mappers
    assert mappers["prompt"].required is True

    # Check optional fields
    assert "size" in mappers
    assert "quality" in mappers
    assert "style" in mappers
    assert "n" in mappers


def test_dalle3_field_mappers_apply_correctly():
    """Test that DALL-E 3 field mappers convert request correctly."""
    from tarash.tarash_gateway.providers.openai import get_openai_image_field_mappers
    from tarash.tarash_gateway.providers.field_mappers import apply_field_mappers

    mappers = get_openai_image_field_mappers("dall-e-3")

    request = ImageGenerationRequest(
        prompt="A serene lake at sunset",
        size="1024x1792",
        quality="hd",
        style="vivid",
        n=1,
    )

    result = apply_field_mappers(mappers, request)

    assert result["prompt"] == "A serene lake at sunset"
    assert result["size"] == "1024x1792"
    assert result["quality"] == "hd"
    assert result["style"] == "vivid"
    assert result["n"] == 1


def test_get_dalle2_field_mappers():
    """Test that DALL-E 2 field mappers exist."""
    from tarash.tarash_gateway.providers.openai import get_openai_image_field_mappers

    mappers = get_openai_image_field_mappers("dall-e-2")

    # Check required fields
    assert "prompt" in mappers
    assert mappers["prompt"].required is True

    # Check optional fields
    assert "size" in mappers
    assert "n" in mappers
    # DALL-E 2 doesn't have quality or style
    assert "quality" not in mappers
    assert "style" not in mappers


def test_dalle2_field_mappers_apply_correctly():
    """Test that DALL-E 2 field mappers convert request correctly."""
    from tarash.tarash_gateway.providers.openai import get_openai_image_field_mappers
    from tarash.tarash_gateway.providers.field_mappers import apply_field_mappers

    mappers = get_openai_image_field_mappers("dall-e-2")

    request = ImageGenerationRequest(
        prompt="A cute cat wearing a hat",
        size="512x512",
        n=4,
    )

    result = apply_field_mappers(mappers, request)

    assert result["prompt"] == "A cute cat wearing a hat"
    assert result["size"] == "512x512"
    assert result["n"] == 4


def test_openai_image_model_registry_has_all_models():
    """Test that the registry contains all OpenAI image models."""
    from tarash.tarash_gateway.providers.openai import OPENAI_IMAGE_MODEL_REGISTRY

    assert "gpt-image-1.5" in OPENAI_IMAGE_MODEL_REGISTRY
    assert "dall-e-3" in OPENAI_IMAGE_MODEL_REGISTRY
    assert "dall-e-2" in OPENAI_IMAGE_MODEL_REGISTRY


def test_get_openai_image_field_mappers_fallback_for_unknown_model():
    """Test that unknown models fall back to generic mappers."""
    from tarash.tarash_gateway.providers.openai import get_openai_image_field_mappers

    mappers = get_openai_image_field_mappers("unknown-model")

    # Should have basic fields from GENERIC_IMAGE_FIELD_MAPPERS
    assert "prompt" in mappers
    assert "size" in mappers
    assert "quality" in mappers
    assert "n" in mappers


# ==================== Image Generation Tests ====================


@pytest.mark.asyncio
async def test_generate_image_async_with_mocked_client(
    handler, base_config, base_request
):
    """Test async image generation with mocked OpenAI client."""
    from tarash.tarash_gateway.models import ImageGenerationResponse

    # Mock the async client
    with patch(
        "tarash.tarash_gateway.providers.openai.AsyncOpenAI"
    ) as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Mock the images.generate response
        mock_response = MagicMock()
        mock_image = MagicMock()
        mock_image.url = "https://example.com/image1.png"
        mock_image.revised_prompt = "A beautiful serene mountain landscape"
        mock_response.data = [mock_image]
        mock_response.model_dump.return_value = {
            "data": [{"url": "https://example.com/image1.png"}]
        }

        mock_client.images.generate = AsyncMock(return_value=mock_response)

        # Generate image
        response = await handler.generate_image_async(base_config, base_request)

        # Validate response
        assert isinstance(response, ImageGenerationResponse)
        assert response.status == "completed"
        assert len(response.images) == 1
        assert response.images[0] == "https://example.com/image1.png"
        assert response.request_id is not None
        assert response.revised_prompt == "A beautiful serene mountain landscape"

        # Verify client was called correctly
        mock_client.images.generate.assert_called_once()
        call_kwargs = mock_client.images.generate.call_args.kwargs
        assert call_kwargs["model"] == "dall-e-3"
        assert call_kwargs["prompt"] == "A serene mountain landscape"


def test_generate_image_sync_with_mocked_client(handler, base_config, base_request):
    """Test sync image generation with mocked OpenAI client."""
    from tarash.tarash_gateway.models import ImageGenerationResponse

    # Clear cache to ensure fresh client
    handler._sync_client_cache.clear()

    # Mock the sync client
    with patch("tarash.tarash_gateway.providers.openai.OpenAI") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock the images.generate response
        mock_response = MagicMock()
        mock_image = MagicMock()
        mock_image.url = "https://example.com/image1.png"
        mock_image.revised_prompt = None  # Not all responses have revised prompts
        mock_response.data = [mock_image]
        mock_response.model_dump.return_value = {
            "data": [{"url": "https://example.com/image1.png"}]
        }

        mock_client.images.generate.return_value = mock_response

        # Generate image
        response = handler.generate_image(base_config, base_request)

        # Validate response
        assert isinstance(response, ImageGenerationResponse)
        assert response.status == "completed"
        assert len(response.images) == 1
        assert response.images[0] == "https://example.com/image1.png"
        assert response.request_id is not None
        assert response.revised_prompt is None

        # Verify client was called correctly
        mock_client.images.generate.assert_called_once()
        call_kwargs = mock_client.images.generate.call_args.kwargs
        assert call_kwargs["model"] == "dall-e-3"
        assert call_kwargs["prompt"] == "A serene mountain landscape"


@pytest.mark.asyncio
async def test_generate_image_async_multiple_images(handler, base_config):
    """Test async generation of multiple images with DALL-E 2."""
    from tarash.tarash_gateway.models import ImageGenerationResponse

    config = base_config.model_copy(update={"model": "dall-e-2"})
    request = ImageGenerationRequest(
        prompt="A cute cat",
        size="512x512",
        n=3,
    )

    with patch(
        "tarash.tarash_gateway.providers.openai.AsyncOpenAI"
    ) as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Mock response with multiple images
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(url=f"https://example.com/image{i}.png", revised_prompt=None)
            for i in range(1, 4)
        ]
        mock_response.model_dump.return_value = {"data": []}

        mock_client.images.generate = AsyncMock(return_value=mock_response)

        # Generate images
        response = await handler.generate_image_async(config, request)

        # Validate response
        assert isinstance(response, ImageGenerationResponse)
        assert response.status == "completed"
        assert len(response.images) == 3
        assert all(img.startswith("https://example.com/") for img in response.images)


def test_convert_image_request_includes_model(handler, base_config, base_request):
    """Test that _convert_image_request includes model in output."""
    result = handler._convert_image_request(base_config, base_request)

    assert result["model"] == "dall-e-3"
    assert result["prompt"] == "A serene mountain landscape"


def test_convert_image_request_merges_extra_params(handler, base_config):
    """Test that extra_params are merged into the request."""
    request = ImageGenerationRequest(
        prompt="Test",
        extra_params={
            "custom_field": "custom_value",
            "another_field": 42,
        },
    )

    result = handler._convert_image_request(base_config, request)

    assert result["custom_field"] == "custom_value"
    assert result["another_field"] == 42


# ==================== Registry Integration Tests ====================


def test_openai_provider_registered_in_registry(base_config):
    """Test that OpenAI provider can be looked up in registry."""
    from tarash.tarash_gateway.registry import get_handler

    handler = get_handler(base_config)

    # Should return OpenAIProviderHandler
    from tarash.tarash_gateway.providers.openai import OpenAIProviderHandler

    assert isinstance(handler, OpenAIProviderHandler)
