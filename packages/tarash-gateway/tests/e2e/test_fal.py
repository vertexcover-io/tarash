"""End-to-end tests for Fal provider.

These tests make actual API calls to the Fal.ai service.
Requires FAL_KEY environment variable to be set.

Run with: pytest tests/e2e/test_fal.py -v -m e2e
Skip with: pytest tests/e2e/test_fal.py -v -m "not e2e"
"""

import os

import pytest

from tarash.tarash_gateway.video import api
from tarash.tarash_gateway.video.models import (
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoGenerationUpdate,
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
async def test_comprehensive_async_video_generation(fal_api_key):
    """
    Comprehensive async test combining:
    - Basic video generation
    - Progress tracking
    - Custom parameters (seed, negative_prompt)
    - Different image types
    """
    # Track progress updates
    progress_updates = []

    async def progress_callback(update: VideoGenerationUpdate):
        progress_updates.append(update)
        print(f"  Progress: {update.status}")

    # Test with various parameters

    fal_config = VideoGenerationConfig(
        model="fal-ai/veo3.1/fast",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="A serene lake at sunset with mountains in the background, cinematic quality",
        duration_seconds=4,
        aspect_ratio="16:9",
        resolution="720p",
        seed=42,
        negative_prompt="blur, low quality",
        generate_audio=True,
        model_params={
            "auto_fix": True,
        },
    )

    # Generate video using API
    response = await api.generate_video_async(
        fal_config, request, on_progress=progress_callback
    )

    # Validate response
    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"

    # Validate raw_response structure
    assert isinstance(response.raw_response, dict)
    assert "video" in response.raw_response, "raw_response should contain 'video' field"

    # Video should be a URL string (Fal returns URLs)
    assert isinstance(response.video, str), "Video should be a string"
    assert response.video.startswith("http"), (
        f"Expected HTTP URL, got: {response.video}"
    )
    video_type = "URL"
    video_info = response.video
    # Validate progress tracking
    assert len(progress_updates) > 0, "Should receive at least one progress update"
    statuses = [update.status for update in progress_updates]
    assert "completed" in statuses, "Should receive completed status"

    # Log details
    print("✓ Generated video successfully")
    print(f"  Request ID: {response.request_id}")
    print(f"  Video type: {video_type}")
    print(f"  Video info: {video_info}")
    print(f"  Progress updates: {len(progress_updates)}")
    print(f"  Statuses: {statuses}")
    print(
        f"  Duration: {response.duration}s" if response.duration else "  Duration: N/A"
    )
    print(
        f"  Resolution: {response.resolution}"
        if response.resolution
        else "  Resolution: N/A"
    )


@pytest.mark.e2e
def test_sync_video_generation_with_images(fal_api_key):
    """
    Sync test combining:
    - Basic sync generation
    - Reference images
    - Different aspect ratios
    """
    fal_config = VideoGenerationConfig(
        model="fal-ai/veo3.1/fast/image-to-video",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="A calm ocean wave rolling onto a sandy beach, inspired by the reference style",
        duration_seconds=5,
        aspect_ratio="9:16",
        model_params={
            "image_url": "https://storage.googleapis.com/falserverless/example_inputs/veo31_i2v_input.jpg",
        },
    )

    # Generate video using API (sync)
    response = api.generate_video(fal_config, request)

    # Validate response
    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"

    print(f"✓ Generated video with reference image: {response.request_id}")
