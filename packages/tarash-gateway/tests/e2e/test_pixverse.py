"""End-to-end tests for Pixverse provider via Fal.

These tests make actual API calls to the Fal.ai Pixverse service.
Requires FAL_KEY environment variable to be set.

Run with: uv run pytest tests/e2e/test_pixverse.py -v -m e2e
Skip with: uv run pytest tests/e2e/test_pixverse.py -v -m "not e2e"
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
async def test_pixverse_text_to_video(fal_api_key):
    """Test Pixverse v5.5 text-to-video generation."""
    # Track progress updates
    progress_updates = []

    async def progress_callback(update: VideoGenerationUpdate):
        progress_updates.append(update)
        print(f"  Progress: {update.status}")

    config = VideoGenerationConfig(
        model="fal-ai/pixverse/v5.5/text-to-video",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="A serene lake at sunset with mountains in the background",
        duration_seconds=5,
        aspect_ratio="16:9",
        resolution="720p",
        seed=42,
    )

    # Generate video
    response = await api.generate_video_async(
        config, request, on_progress=progress_callback
    )

    # Validate response
    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"

    # Validate raw_response structure
    assert isinstance(response.raw_response, dict)
    assert "video" in response.raw_response

    # Video should be a URL
    assert isinstance(response.video, str)
    assert response.video.startswith("http")

    # Validate progress tracking
    assert len(progress_updates) > 0
    statuses = [update.status for update in progress_updates]
    assert "completed" in statuses

    print(f"\n  ✓ Generated video URL: {response.video}")
    print(f"  ✓ Request ID: {response.request_id}")
    print(f"  ✓ Progress updates: {len(progress_updates)}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_pixverse_image_to_video(fal_api_key):
    """Test Pixverse v5.5 image-to-video generation."""
    progress_updates = []

    async def progress_callback(update: VideoGenerationUpdate):
        progress_updates.append(update)
        print(f"  Progress: {update.status}")

    config = VideoGenerationConfig(
        model="fal-ai/pixverse/v5.5/image-to-video",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    # Using a test image URL (public image)
    request = VideoGenerationRequest(
        prompt="A woman warrior walking with her wolf companion through the forest",
        image_list=[
            {
                "image": "https://v3.fal.media/files/zebra/qL93Je8ezvzQgDOEzTjKF_KhGKZTEebZcDw6T5rwQPK_output.png",
                "type": "reference",
            }
        ],
        duration_seconds=5,
        resolution="720p",
        extra_params={
            "style": "3d_animation",
        },
    )

    # Generate video
    response = await api.generate_video_async(
        config, request, on_progress=progress_callback
    )

    # Validate response
    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"

    # Video should be a URL
    assert isinstance(response.video, str)
    assert response.video.startswith("http")

    # Validate progress tracking
    assert len(progress_updates) > 0

    print(f"\n  ✓ Generated video URL: {response.video}")
    print(f"  ✓ Request ID: {response.request_id}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_pixverse_v5_text_to_video(fal_api_key):
    """Test Pixverse v5 text-to-video generation (should work same as v5.5)."""
    config = VideoGenerationConfig(
        model="fal-ai/pixverse/v5/text-to-video",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
    )

    request = VideoGenerationRequest(
        prompt="A peaceful garden with blooming flowers and butterflies",
        duration_seconds=5,
        aspect_ratio="16:9",
    )

    # Generate video
    response = await api.generate_video_async(config, request)

    # Validate response
    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"

    # Video should be a URL
    assert isinstance(response.video, str)
    assert response.video.startswith("http")

    print(f"\n  ✓ Pixverse v5 generated video URL: {response.video}")
