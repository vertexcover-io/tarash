"""End-to-end tests for Replicate image generation.

These tests make actual API calls to the Replicate service.
Requires REPLICATE_API_TOKEN environment variable to be set.

Run with: pytest tests/e2e/test_replicate_image.py -v --e2e
"""

import os

import pytest

from tarash.tarash_gateway.models import (
    ImageGenerationConfig,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageGenerationUpdate,
)
from tarash.tarash_gateway.providers.replicate import ReplicateProviderHandler

# ==================== Fixtures ====================


@pytest.fixture(scope="module")
def replicate_api_key():
    """Get Replicate API key from environment."""
    api_key = os.getenv("REPLICATE_API_TOKEN")
    if not api_key:
        pytest.skip("REPLICATE_API_TOKEN environment variable not set")
    return api_key


@pytest.fixture
def handler():
    """Create ReplicateProviderHandler instance."""
    return ReplicateProviderHandler()


# ==================== E2E Tests ====================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_flux_schnell_text_to_image(replicate_api_key, handler):
    """
    Test FLUX Schnell text-to-image model (fastest).

    This tests:
    - Basic text-to-image generation
    - Progress tracking
    - FLUX Schnell model (fast inference)
    """
    progress_updates = []

    async def progress_callback(update: ImageGenerationUpdate):
        progress_updates.append(update)
        print(f"  Progress: {update.status}")

    config = ImageGenerationConfig(
        model="black-forest-labs/flux-schnell",
        provider="replicate",
        api_key=replicate_api_key,
        timeout=120,
        max_poll_attempts=60,
        poll_interval=2,
    )

    request = ImageGenerationRequest(
        prompt="A serene mountain landscape at sunset with a lake reflection, photorealistic",
        aspect_ratio="16:9",
        seed=42,
    )

    # Generate image
    response = await handler.generate_image_async(
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
def test_sd35_large_turbo_generation(replicate_api_key, handler):
    """
    Test SD 3.5 Large Turbo synchronous image generation.

    This tests:
    - Synchronous generation (blocking)
    - SD 3.5 Large Turbo model
    - Basic configuration parameters
    """
    config = ImageGenerationConfig(
        model="stability-ai/stable-diffusion-3.5-large-turbo",
        provider="replicate",
        api_key=replicate_api_key,
        timeout=120,
        max_poll_attempts=60,
        poll_interval=2,
    )

    request = ImageGenerationRequest(
        prompt="A futuristic cityscape at night with neon lights, cyberpunk style",
        aspect_ratio="16:9",
        seed=123,
        extra_params={
            "num_inference_steps": 4,  # Turbo model uses fewer steps
            "output_format": "webp",
        },
    )

    # Generate image synchronously
    response = handler.generate_image(config, request)

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

    print("✓ Generated SD 3.5 Large Turbo image successfully")
    print(f"  Request ID: {response.request_id}")
    print(f"  Images: {len(response.images)}")
    print(f"  First image URL: {response.images[0][:80]}...")
