"""End-to-end tests for Google AI image generation.

These tests make actual API calls to the Google AI service.
Requires GOOGLE_API_KEY environment variable to be set.

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
def google_api_key():
    """Get Google API key from environment."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY environment variable not set")
    return api_key


# ==================== E2E Tests ====================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_imagen3_text_to_image(google_api_key):
    """
    Test Imagen 3 text-to-image generation.

    This tests:
    - Basic text-to-image generation with Imagen 3
    - Async API
    - Response structure validation
    """
    config = ImageGenerationConfig(
        model="imagen-3.0-generate-001",
        provider="google",
        api_key=google_api_key,
        timeout=120,
    )

    request = ImageGenerationRequest(
        prompt="A serene mountain landscape at sunset with snow-capped peaks",
    )

    # Generate image using API
    response = await api.generate_image_async(config, request)

    # Validate response
    assert isinstance(response, ImageGenerationResponse)
    assert response.request_id is not None
    assert response.images is not None
    assert len(response.images) > 0
    assert response.status == "completed"

    # Images should be URLs
    for image_url in response.images:
        assert isinstance(image_url, str)
        assert image_url.startswith("http"), f"Expected HTTP URL, got: {image_url}"

    print("✓ Generated Imagen 3 image successfully")
    print(f"  Request ID: {response.request_id}")
    print(f"  Images: {len(response.images)}")
    print(f"  First image URL: {response.images[0][:80]}...")


@pytest.mark.e2e
def test_imagen3_sync_generation(google_api_key):
    """
    Test Imagen 3 synchronous image generation.

    This tests:
    - Sync API for image generation
    - Basic prompt
    """
    config = ImageGenerationConfig(
        model="imagen-3.0-generate-001",
        provider="google",
        api_key=google_api_key,
        timeout=120,
    )

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
async def test_imagen3_with_aspect_ratio(google_api_key):
    """
    Test Imagen 3 with custom aspect ratio.

    This tests:
    - aspect_ratio parameter
    - 16:9 landscape format
    """
    config = ImageGenerationConfig(
        model="imagen-3.0-generate-001",
        provider="google",
        api_key=google_api_key,
        timeout=120,
    )

    request = ImageGenerationRequest(
        prompt="A peaceful beach scene with palm trees and ocean waves",
        aspect_ratio="16:9",
    )

    # Generate image using API
    response = await api.generate_image_async(config, request)

    # Validate response
    assert isinstance(response, ImageGenerationResponse)
    assert response.request_id is not None
    assert response.status == "completed"
    assert response.images is not None
    assert len(response.images) > 0

    print(f"✓ Generated Imagen 3 image (16:9): {response.request_id}")
    print(f"  Image URL: {response.images[0][:80]}...")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_imagen3_with_negative_prompt(google_api_key):
    """
    Test Imagen 3 with negative prompt.

    This tests:
    - negative_prompt parameter
    - Content filtering
    """
    config = ImageGenerationConfig(
        model="imagen-3.0-generate-001",
        provider="google",
        api_key=google_api_key,
        timeout=120,
    )

    request = ImageGenerationRequest(
        prompt="A beautiful garden with colorful flowers",
        negative_prompt="blur, low quality, watermark",
    )

    # Generate image using API
    response = await api.generate_image_async(config, request)

    # Validate response
    assert isinstance(response, ImageGenerationResponse)
    assert response.request_id is not None
    assert response.status == "completed"
    assert response.images is not None
    assert len(response.images) > 0

    print(f"✓ Generated Imagen 3 image with negative prompt: {response.request_id}")
    print(f"  Image URL: {response.images[0][:80]}...")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_imagen3_with_multiple_images(google_api_key):
    """
    Test Imagen 3 generating multiple images.

    This tests:
    - number_of_images parameter (via n)
    - Multiple image generation
    """
    config = ImageGenerationConfig(
        model="imagen-3.0-generate-001",
        provider="google",
        api_key=google_api_key,
        timeout=120,
    )

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
async def test_nano_banana_text_to_image(google_api_key):
    """
    Test Gemini 2.5 Flash Image (Nano Banana) text-to-image generation.

    This tests:
    - Nano Banana model (fast, efficient)
    - Basic text-to-image generation
    """
    config = ImageGenerationConfig(
        model="gemini-2.5-flash-image-001",
        provider="google",
        api_key=google_api_key,
        timeout=120,
    )

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


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_execution_metadata_included(google_api_key):
    """
    Test that execution metadata is included in response.

    This tests:
    - ExecutionMetadata is attached to response
    - Contains correct attempt information
    """
    config = ImageGenerationConfig(
        model="imagen-3.0-generate-001",
        provider="google",
        api_key=google_api_key,
        timeout=120,
    )

    request = ImageGenerationRequest(
        prompt="A simple red apple on a white background",
    )

    response = await api.generate_image_async(config, request)

    # Validate execution metadata
    assert response.execution_metadata is not None
    assert response.execution_metadata.total_attempts == 1
    assert response.execution_metadata.successful_attempt == 1
    assert response.execution_metadata.fallback_triggered is False
    assert response.execution_metadata.configs_in_chain == 1
    assert len(response.execution_metadata.attempts) == 1

    # Validate attempt details
    attempt = response.execution_metadata.attempts[0]
    assert attempt.provider == "google"
    assert attempt.model == "imagen-3.0-generate-001"
    assert attempt.status == "success"
    assert attempt.error_type is None

    print("✓ Execution metadata correctly included")
    print(f"  Total attempts: {response.execution_metadata.total_attempts}")
    print(f"  Successful attempt: {response.execution_metadata.successful_attempt}")
    print(f"  Total elapsed: {response.execution_metadata.total_elapsed_seconds:.2f}s")
