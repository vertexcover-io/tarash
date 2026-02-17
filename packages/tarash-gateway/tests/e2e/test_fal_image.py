"""End-to-end tests for Fal image generation.

These tests make actual API calls to the Fal.ai service.
Requires FAL_KEY environment variable to be set.

Run with: pytest tests/e2e/test_fal_image.py -v --e2e
"""

import os

import pytest

from tarash.tarash_gateway import api
from tarash.tarash_gateway.models import (
    ImageGenerationConfig,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageGenerationUpdate,
)

# ==================== Fixtures ====================


@pytest.fixture(scope="module")
def fal_api_key():
    """Get Fal API key from environment."""
    api_key = os.getenv("FAL_KEY")
    if not api_key:
        pytest.skip("FAL_KEY environment variable not set")
    return api_key


# ==================== E2E Tests ====================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_flux_schnell_text_to_image(fal_api_key):
    """
    Test FLUX Schnell text-to-image model (fastest).

    This tests:
    - Basic text-to-image generation
    - Progress tracking
    - FLUX Schnell model (1-4 steps, very fast)
    """
    progress_updates = []

    async def progress_callback(update: ImageGenerationUpdate):
        progress_updates.append(update)
        print(f"  Progress: {update.status}")

    config = ImageGenerationConfig(
        model="fal-ai/flux/schnell",
        provider="fal",
        api_key=fal_api_key,
        timeout=120,
        max_poll_attempts=60,
        poll_interval=2,
    )

    request = ImageGenerationRequest(
        prompt="A serene mountain landscape at sunset with a lake reflection, photorealistic",
        seed=42,
    )

    # Generate image using API
    response = await api.generate_image_async(
        config, request, on_progress=progress_callback
    )

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

    # Validate progress tracking
    assert len(progress_updates) > 0, "Should receive at least one progress update"
    statuses = [update.status for update in progress_updates]
    assert "completed" in statuses, "Should receive completed status"

    print("✓ Generated FLUX Schnell image successfully")
    print(f"  Request ID: {response.request_id}")
    print(f"  Images: {len(response.images)}")
    print(f"  First image URL: {response.images[0][:80]}...")
    print(f"  Progress updates: {len(progress_updates)}")


@pytest.mark.e2e
def test_flux_schnell_sync_generation(fal_api_key):
    """
    Test FLUX Schnell synchronous image generation.

    This tests:
    - Sync API for image generation
    - Basic prompt with seed
    """
    config = ImageGenerationConfig(
        model="fal-ai/flux/schnell",
        provider="fal",
        api_key=fal_api_key,
        timeout=120,
        max_poll_attempts=60,
        poll_interval=2,
    )

    request = ImageGenerationRequest(
        prompt="A cute orange cat sitting on a windowsill, digital art style",
        seed=12345,
    )

    # Generate image using sync API
    response = api.generate_image(config, request)

    # Validate response
    assert isinstance(response, ImageGenerationResponse)
    assert response.request_id is not None
    assert response.images is not None
    assert len(response.images) > 0
    assert response.status == "completed"

    print(f"✓ Generated FLUX Schnell image (sync): {response.request_id}")
    print(f"  Image URL: {response.images[0][:80]}...")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_flux_dev_with_multiple_images(fal_api_key):
    """
    Test FLUX Dev model with multiple image generation.

    This tests:
    - FLUX Dev model (higher quality, 20-50 steps)
    - Multiple images (n parameter)
    - Custom image size
    """
    config = ImageGenerationConfig(
        model="fal-ai/flux/dev",
        provider="fal",
        api_key=fal_api_key,
        timeout=180,  # Longer timeout for dev model
        max_poll_attempts=90,
        poll_interval=2,
    )

    request = ImageGenerationRequest(
        prompt="A futuristic cityscape with flying cars and neon lights, cyberpunk style",
        n=2,  # Generate 2 images
        size="landscape_4_3",  # Fal-specific size format
        seed=42,
    )

    # Generate images using API
    response = await api.generate_image_async(config, request)

    # Validate response
    assert isinstance(response, ImageGenerationResponse)
    assert response.request_id is not None
    assert response.status == "completed"
    assert response.images is not None
    assert len(response.images) == 2, f"Expected 2 images, got {len(response.images)}"

    print(f"✓ Generated FLUX Dev images: {response.request_id}")
    print(f"  Number of images: {len(response.images)}")
    for i, url in enumerate(response.images):
        print(f"  Image {i + 1}: {url[:80]}...")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_execution_metadata_included(fal_api_key):
    """
    Test that execution metadata is included in response.

    This tests:
    - ExecutionMetadata is attached to response
    - Contains correct attempt information
    """
    config = ImageGenerationConfig(
        model="fal-ai/flux/schnell",
        provider="fal",
        api_key=fal_api_key,
        timeout=120,
        max_poll_attempts=60,
        poll_interval=2,
    )

    request = ImageGenerationRequest(
        prompt="A simple red apple on a white background",
        seed=999,
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
    assert attempt.provider == "fal"
    assert attempt.model == "fal-ai/flux/schnell"
    assert attempt.status == "success"
    assert attempt.error_type is None

    print("✓ Execution metadata correctly included")
    print(f"  Total attempts: {response.execution_metadata.total_attempts}")
    print(f"  Successful attempt: {response.execution_metadata.successful_attempt}")
    print(f"  Total elapsed: {response.execution_metadata.total_elapsed_seconds:.2f}s")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_flux2_text_to_image(fal_api_key):
    """
    Test FLUX.2 text-to-image model.

    This tests:
    - FLUX.2 model with guidance_scale
    - num_inference_steps parameter
    - Higher quality generation
    """
    config = ImageGenerationConfig(
        model="fal-ai/flux-2",
        provider="fal",
        api_key=fal_api_key,
        timeout=180,  # Longer timeout for Pro model
        max_poll_attempts=90,
        poll_interval=2,
    )

    request = ImageGenerationRequest(
        prompt="A photorealistic portrait of a woman in cyberpunk style, neon lighting",
        seed=42,
        extra_params={
            "guidance_scale": 7.5,
            "num_inference_steps": 30,
        },
    )

    response = await api.generate_image_async(config, request)

    # Validate response
    assert isinstance(response, ImageGenerationResponse)
    assert response.request_id is not None
    assert response.images is not None
    assert len(response.images) > 0
    assert response.status == "completed"

    print(f"✓ Generated FLUX.2 image: {response.request_id}")
    print(f"  Image URL: {response.images[0][:80]}...")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_flux_pro_ultra_with_aspect_ratio(fal_api_key):
    """
    Test Flux 1.1 Pro Ultra with extended aspect ratio support.

    This tests:
    - Flux 1.1 Pro Ultra model
    - Extended aspect ratios (21:9)
    - Ultra high quality generation
    """
    config = ImageGenerationConfig(
        model="fal-ai/flux-pro/v1.1-ultra",
        provider="fal",
        api_key=fal_api_key,
        timeout=240,  # Longer timeout for Ultra model
        max_poll_attempts=120,
        poll_interval=2,
    )

    request = ImageGenerationRequest(
        prompt="Epic cinematic landscape with mountains and valleys, ultra wide view",
        seed=99,
        aspect_ratio="21:9",  # Ultra-wide aspect ratio
        extra_params={
            "safety_tolerance": 3,
            "output_format": "png",
        },
    )

    response = await api.generate_image_async(config, request)

    # Validate response
    assert isinstance(response, ImageGenerationResponse)
    assert response.request_id is not None
    assert response.images is not None
    assert len(response.images) > 0
    assert response.status == "completed"

    print(f"✓ Generated Flux Pro Ultra image (21:9): {response.request_id}")
    print(f"  Image URL: {response.images[0][:80]}...")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_zimage_turbo_fast_generation(fal_api_key):
    """
    Test Z-Image-Turbo for fast distilled generation.

    This tests:
    - Z-Image-Turbo model (optimized for speed)
    - Fast generation with 8 inference steps
    - negative_prompt support
    """
    config = ImageGenerationConfig(
        model="fal-ai/z-image/turbo",
        provider="fal",
        api_key=fal_api_key,
        timeout=90,  # Shorter timeout for turbo model
        max_poll_attempts=45,
        poll_interval=2,
    )

    request = ImageGenerationRequest(
        prompt="A colorful abstract art piece with geometric patterns",
        seed=777,
        negative_prompt="blur, low quality, noise",
        extra_params={
            "num_inference_steps": 8,  # Fast distilled generation
            "enable_safety_checker": True,
        },
    )

    response = await api.generate_image_async(config, request)

    # Validate response
    assert isinstance(response, ImageGenerationResponse)
    assert response.request_id is not None
    assert response.images is not None
    assert len(response.images) > 0
    assert response.status == "completed"

    print(f"✓ Generated Z-Image-Turbo image (fast): {response.request_id}")
    print(f"  Image URL: {response.images[0][:80]}...")
