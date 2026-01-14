"""End-to-end tests for OpenAI image generation.

These tests make actual API calls to the OpenAI service.
Requires OPENAI_API_KEY environment variable to be set.

Run with: pytest tests/e2e/test_openai_image.py -v --e2e
"""

import os

import pytest

from tarash.tarash_gateway.models import (
    ImageGenerationConfig,
    ImageGenerationRequest,
    ImageGenerationResponse,
)
from tarash.tarash_gateway.providers.openai import OpenAIProviderHandler

# ==================== Fixtures ====================


@pytest.fixture(scope="module")
def openai_api_key():
    """Get OpenAI API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    return api_key


@pytest.fixture
def handler():
    """Create an OpenAIProviderHandler instance."""
    return OpenAIProviderHandler()


# ==================== E2E Tests ====================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_dalle3_text_to_image_async(openai_api_key, handler):
    """Test DALL-E 3 text-to-image generation (async).

    This tests:
    - Basic text-to-image generation
    - DALL-E 3 model
    - HD quality
    - Vivid style
    """
    config = ImageGenerationConfig(
        model="dall-e-3",
        provider="openai",
        api_key=openai_api_key,
        timeout=120,
    )

    request = ImageGenerationRequest(
        prompt="A serene lake at sunset with mountains in the background",
        size="1024x1024",
        quality="hd",
        style="vivid",
        n=1,
    )

    print(f"\nGenerating image with model: {config.model}")
    response = await handler.generate_image_async(config, request)

    # Validate response structure
    assert isinstance(response, ImageGenerationResponse)
    assert response.request_id is not None
    assert response.status == "completed"
    assert isinstance(response.raw_response, dict)

    # Validate images
    assert len(response.images) == 1
    assert isinstance(response.images[0], str)
    assert response.images[0].startswith("http")

    # Validate revised prompt (DALL-E 3 may revise prompts)
    if response.revised_prompt:
        assert isinstance(response.revised_prompt, str)

    print("Image generated successfully")
    print(f"  Request ID: {response.request_id}")
    print(f"  Image URL: {response.images[0]}")
    if response.revised_prompt:
        print(f"  Revised prompt: {response.revised_prompt}")


@pytest.mark.e2e
def test_dalle3_text_to_image_sync(openai_api_key, handler):
    """Test DALL-E 3 text-to-image generation (sync).

    This tests:
    - Sync execution
    - Natural style
    - Portrait aspect ratio
    """
    config = ImageGenerationConfig(
        model="dall-e-3",
        provider="openai",
        api_key=openai_api_key,
        timeout=120,
    )

    request = ImageGenerationRequest(
        prompt="A futuristic city with flying cars",
        size="1024x1792",
        quality="standard",
        style="natural",
        n=1,
    )

    print(f"\nGenerating image with model: {config.model}")
    response = handler.generate_image(config, request)

    # Validate response
    assert isinstance(response, ImageGenerationResponse)
    assert response.request_id is not None
    assert response.status == "completed"
    assert len(response.images) == 1
    assert response.images[0].startswith("http")

    print("Image generated successfully")
    print(f"  Request ID: {response.request_id}")
    print(f"  Image URL: {response.images[0]}")


@pytest.mark.e2e
def test_dalle2_text_to_image_multiple(openai_api_key, handler):
    """Test DALL-E 2 multiple image generation.

    This tests:
    - DALL-E 2 model
    - Generating multiple images (n=2)
    - Smaller image size
    """
    config = ImageGenerationConfig(
        model="dall-e-2",
        provider="openai",
        api_key=openai_api_key,
        timeout=120,
    )

    request = ImageGenerationRequest(
        prompt="A cute cat wearing a hat",
        size="512x512",
        n=2,
    )

    print(f"\nGenerating images with model: {config.model}")
    response = handler.generate_image(config, request)

    # Validate response
    assert isinstance(response, ImageGenerationResponse)
    assert response.request_id is not None
    assert response.status == "completed"
    assert len(response.images) == 2
    assert all(img.startswith("http") for img in response.images)

    print("Images generated successfully")
    print(f"  Request ID: {response.request_id}")
    print("  Image URLs:")
    for i, url in enumerate(response.images, 1):
        print(f"    {i}. {url}")
