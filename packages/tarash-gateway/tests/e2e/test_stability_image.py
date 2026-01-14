"""End-to-end tests for Stability AI image generation.

These tests make actual API calls to the Stability AI service.
Requires STABILITY_API_KEY environment variable to be set.

Run with: pytest tests/e2e/test_stability_image.py -v --e2e
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
def stability_api_key():
    """Get Stability API key from environment."""
    api_key = os.getenv("STABILITY_API_KEY")
    if not api_key:
        pytest.skip("STABILITY_API_KEY environment variable not set")
    return api_key


# ==================== E2E Tests ====================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_sd35_large_turbo_generation(stability_api_key):
    """
    Test SD 3.5 Large Turbo text-to-image model (faster variant).

    This tests:
    - Basic text-to-image generation
    - SD 3.5 Large Turbo model (faster than base)
    - Data URL response format
    """
    config = ImageGenerationConfig(
        model="sd3.5-large-turbo",
        provider="stability",
        api_key=stability_api_key,
        timeout=120,
    )

    request = ImageGenerationRequest(
        prompt="A serene mountain landscape at sunset with a crystal clear lake, photorealistic",
        aspect_ratio="16:9",
        seed=42,
    )

    # Generate image using API
    response = await api.generate_image_async(config, request)

    # Validate response
    assert isinstance(response, ImageGenerationResponse)
    assert response.request_id is not None
    assert response.images is not None
    assert len(response.images) > 0
    assert response.status == "completed"

    # Images should be data URLs (base64 encoded)
    for image_url in response.images:
        assert isinstance(image_url, str)
        assert image_url.startswith("data:image/"), (
            f"Expected data URL, got: {image_url[:50]}..."
        )

    print("✓ Generated SD 3.5 Large Turbo image successfully")
    print(f"  Request ID: {response.request_id}")
    print(f"  Images: {len(response.images)}")
    print(f"  Content type: {response.content_type}")
    print(f"  Image data length: {len(response.images[0])} chars")


@pytest.mark.e2e
def test_stable_image_core_sync_generation(stability_api_key):
    """
    Test Stable Image Core synchronous image generation.

    This tests:
    - Sync API for image generation
    - Stable Image Core model (automatic quality optimization)
    - Negative prompts
    """
    config = ImageGenerationConfig(
        model="stable-image-core",
        provider="stability",
        api_key=stability_api_key,
        timeout=120,
    )

    request = ImageGenerationRequest(
        prompt="A cute orange tabby cat sitting on a windowsill with sunlight streaming through, digital art style",
        negative_prompt="blurry, low quality, distorted, ugly",
        aspect_ratio="1:1",
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

    # Verify data URL format
    assert response.images[0].startswith("data:image/")

    print(f"✓ Generated Stable Image Core image (sync): {response.request_id}")
    print(f"  Content type: {response.content_type}")
    print(f"  Image data length: {len(response.images[0])} chars")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_sd35_large_with_extra_params(stability_api_key):
    """
    Test SD 3.5 Large with extra parameters (cfg_scale, steps, output_format).

    This tests:
    - SD 3.5 Large base model
    - Extra parameters via extra_params
    - Custom cfg_scale and steps
    - JPEG output format
    """
    config = ImageGenerationConfig(
        model="sd3.5-large",
        provider="stability",
        api_key=stability_api_key,
        timeout=120,
    )

    request = ImageGenerationRequest(
        prompt="A futuristic cityscape with flying cars and neon lights, cyberpunk style",
        aspect_ratio="16:9",
        negative_prompt="realistic, photo, blur",
        seed=999,
        extra_params={
            "cfg_scale": 7.0,
            "steps": 40,
            "output_format": "jpeg",
        },
    )

    # Generate image using API
    response = await api.generate_image_async(config, request)

    # Validate response
    assert isinstance(response, ImageGenerationResponse)
    assert response.request_id is not None
    assert response.status == "completed"
    assert response.images is not None
    assert len(response.images) > 0

    # Verify JPEG content type
    assert response.content_type == "image/jpeg"
    assert response.images[0].startswith("data:image/jpeg;base64,")

    print(f"✓ Generated SD 3.5 Large image with extra params: {response.request_id}")
    print(f"  Content type: {response.content_type}")
    print("  Output format: JPEG")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_stable_image_ultra_with_seed(stability_api_key):
    """
    Test Stable Image Ultra with seed for reproducibility.

    This tests:
    - Stable Image Ultra model
    - Seed parameter for reproducible results
    - Portrait aspect ratio
    """
    config = ImageGenerationConfig(
        model="stable-image-ultra",
        provider="stability",
        api_key=stability_api_key,
        timeout=120,
    )

    request = ImageGenerationRequest(
        prompt="A portrait of a wise old wizard with a long white beard, fantasy art",
        aspect_ratio="2:3",
        seed=7777,
    )

    # Generate image using API
    response = await api.generate_image_async(config, request)

    # Validate response
    assert isinstance(response, ImageGenerationResponse)
    assert response.request_id is not None
    assert response.status == "completed"
    assert response.images is not None
    assert len(response.images) > 0

    # Verify data URL format
    assert response.images[0].startswith("data:image/")

    print(f"✓ Generated Stable Image Ultra image: {response.request_id}")
    print("  Aspect ratio: 2:3 (portrait)")
    print("  Seed: 7777")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_execution_metadata_included(stability_api_key):
    """
    Test that execution metadata is included in response.

    This tests:
    - ExecutionMetadata is attached to response
    - Contains correct attempt information
    """
    config = ImageGenerationConfig(
        model="sd3.5-large-turbo",
        provider="stability",
        api_key=stability_api_key,
        timeout=120,
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
    assert attempt.provider == "stability"
    assert attempt.model == "sd3.5-large-turbo"
    assert attempt.status == "success"
    assert attempt.error_type is None

    print("✓ Execution metadata correctly included")
    print(f"  Total attempts: {response.execution_metadata.total_attempts}")
    print(f"  Successful attempt: {response.execution_metadata.successful_attempt}")
    print(f"  Total elapsed: {response.execution_metadata.total_elapsed_seconds:.2f}s")
