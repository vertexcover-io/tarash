"""End-to-end tests for Google AI image generation.

These tests make actual API calls to the Google AI service.
Requires Vertex AI credentials (GOOGLE_CLOUD_PROJECT, GOOGLE_APPLICATION_CREDENTIALS).

Run with: pytest tests/e2e/test_google_image.py -v --e2e
"""

import os

import pytest

from tarash.tarash_gateway import api
from tarash.tarash_gateway.models import (
    ImageGenerationConfig,
    ImageGenerationRequest,
    ImageGenerationResponse,
)

# ==================== Fixtures ====================


@pytest.fixture(scope="module")
def vertex_ai_provider_config():
    """Build Vertex AI provider_config from environment variables.

    Requires environment variables:
    - GOOGLE_CLOUD_PROJECT
    - GOOGLE_CLOUD_LOCATION (optional, defaults to us-central1)
    - GOOGLE_APPLICATION_CREDENTIALS (optional, for service account auth)
    """
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project:
        pytest.skip("GOOGLE_CLOUD_PROJECT not set for Vertex AI")

    # Build provider_config from environment variables
    provider_config: dict[str, str] = {
        "project": project,
        "location": os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
    }

    # Optionally add credentials_path if set
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path:
        provider_config["credentials_path"] = credentials_path

    return provider_config


@pytest.fixture
def vertex_ai_image_config_factory(vertex_ai_provider_config):
    """Factory fixture to create ImageGenerationConfig for Vertex AI.

    Usage:
        config = vertex_ai_image_config_factory("imagen-3.0-generate-001")
        config = vertex_ai_image_config_factory(
            model="gemini-2.5-flash-image",
            timeout=240
        )
    """

    def _create_config(
        model: str = "imagen-3.0-generate-001",
        timeout: int = 120,
    ) -> ImageGenerationConfig:
        return ImageGenerationConfig(
            model=model,
            provider="google",
            api_key=None,  # Use Vertex AI via provider_config
            timeout=timeout,
            provider_config=vertex_ai_provider_config,
        )

    return _create_config


# ==================== E2E Tests ====================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_imagen3_text_to_image(vertex_ai_image_config_factory):
    """
    Test Imagen 3 text-to-image generation.

    This tests:
    - Basic text-to-image generation with Imagen 3
    - Async API
    - Response structure validation
    """
    config = vertex_ai_image_config_factory("imagen-3.0-generate-001")

    request = ImageGenerationRequest(
        prompt="A serene mountain landscape at sunset with snow-capped peaks",
        aspect_ratio="16:9",
    )

    # Generate image using API
    response = await api.generate_image_async(config, request)

    # Validate response
    assert isinstance(response, ImageGenerationResponse)
    assert response.request_id is not None
    assert response.images is not None
    assert len(response.images) > 0
    assert response.status == "completed"

    # Images should be URLs or base64 data URIs
    for image_url in response.images:
        assert isinstance(image_url, str)
        assert image_url.startswith(("http", "data:")), (
            f"Expected URL or data URI, got: {image_url[:50]}..."
        )

    print("✓ Generated Imagen 3 image successfully")
    print(f"  Request ID: {response.request_id}")
    print(f"  Images: {len(response.images)}")
    print(f"  First image URL: {response.images[0][:80]}...")


@pytest.mark.e2e
def test_imagen3_sync_generation(vertex_ai_image_config_factory):
    """
    Test Imagen 3 synchronous image generation.

    This tests:
    - Sync API for image generation
    - Basic prompt
    """
    config = vertex_ai_image_config_factory("imagen-3.0-generate-001")

    request = ImageGenerationRequest(
        prompt="A futuristic city with flying cars and neon lights",
    )

    # Generate image using sync API
    response = api.generate_image(config, request)

    # Validate response
    assert isinstance(response, ImageGenerationResponse)
    assert response.request_id is not None
    assert response.images is not None
    assert len(response.images) > 0
    assert response.status == "completed"

    print(f"✓ Generated Imagen 3 image (sync): {response.request_id}")
    print(f"  Image URL: {response.images[0][:80]}...")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_imagen3_with_multiple_images(vertex_ai_image_config_factory):
    """
    Test Imagen 3 generating multiple images.

    This tests:
    - number_of_images parameter (via n)
    - Multiple image generation
    """
    config = vertex_ai_image_config_factory("imagen-3.0-generate-001")

    request = ImageGenerationRequest(
        prompt="A cute robot with friendly eyes",
        n=2,  # Generate 2 images
    )

    # Generate images using API
    response = await api.generate_image_async(config, request)

    # Validate response
    assert isinstance(response, ImageGenerationResponse)
    assert response.request_id is not None
    assert response.status == "completed"
    assert response.images is not None
    assert len(response.images) == 2, f"Expected 2 images, got {len(response.images)}"

    print(f"✓ Generated Imagen 3 images (multiple): {response.request_id}")
    print(f"  Number of images: {len(response.images)}")
    for i, url in enumerate(response.images):
        print(f"  Image {i + 1}: {url[:80]}...")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_nano_banana_text_to_image(vertex_ai_image_config_factory):
    """
    Test Gemini 2.5 Flash Image (Nano Banana) text-to-image generation.

    This tests:
    - Nano Banana model (fast, efficient)
    - Basic text-to-image generation
    """
    config = vertex_ai_image_config_factory("gemini-2.5-flash-image")

    request = ImageGenerationRequest(
        prompt="A cartoon-style illustration of a happy banana character",
    )

    # Generate image using API
    response = await api.generate_image_async(config, request)

    # Validate response
    assert isinstance(response, ImageGenerationResponse)
    assert response.request_id is not None
    assert response.images is not None
    assert len(response.images) > 0
    assert response.status == "completed"

    print("✓ Generated Nano Banana image successfully")
    print(f"  Request ID: {response.request_id}")
    print(f"  Images: {len(response.images)}")
    print(f"  First image URL: {response.images[0][:80]}...")
