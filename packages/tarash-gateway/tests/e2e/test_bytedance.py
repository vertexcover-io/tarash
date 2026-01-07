"""End-to-end tests for ByteDance Seedance provider via Fal.

These tests make actual API calls to the Fal.ai ByteDance Seedance service.
Requires FAL_KEY environment variable to be set.

Run with: uv run pytest tests/e2e/test_bytedance.py -v -m e2e
Skip with: uv run pytest tests/e2e/test_bytedance.py -v -m "not e2e"
"""

import os

import pytest

from tarash.tarash_gateway import api
from tarash.tarash_gateway.models import (
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
async def test_bytedance_seedance_text_to_video(fal_api_key):
    """Test ByteDance Seedance v1.5 Pro text-to-video generation."""
    # Track progress updates
    progress_updates = []

    async def progress_callback(update: VideoGenerationUpdate):
        progress_updates.append(update)
        print(f"  Progress: {update.status}")

    config = VideoGenerationConfig(
        model="fal-ai/bytedance/seedance/v1.5/pro/text-to-video",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="A martial artist practicing in a serene dojo at sunrise",
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
async def test_bytedance_seedance_image_to_video(fal_api_key):
    """Test ByteDance Seedance v1 pro image-to-video generation."""
    progress_updates = []

    async def progress_callback(update: VideoGenerationUpdate):
        progress_updates.append(update)
        print(f"  Progress: {update.status}")

    config = VideoGenerationConfig(
        model="fal-ai/bytedance/seedance/v1/pro/image-to-video",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    # Using a test image URL (public image)
    request = VideoGenerationRequest(
        prompt="A warrior walking through an ancient forest",
        image_list=[
            {
                "image": "https://v3.fal.media/files/zebra/qL93Je8ezvzQgDOEzTjKF_KhGKZTEebZcDw6T5rwQPK_output.png",
                "type": "reference",
            }
        ],
        duration_seconds=5,
        aspect_ratio="16:9",
        resolution="720p",
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
async def test_bytedance_seedance_reference_to_video(fal_api_key):
    """Test ByteDance Seedance v1 lite reference-to-video generation."""
    progress_updates = []

    async def progress_callback(update: VideoGenerationUpdate):
        progress_updates.append(update)
        print(f"  Progress: {update.status}")

    config = VideoGenerationConfig(
        model="fal-ai/bytedance/seedance/v1/lite/reference-to-video",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    # Using test reference images
    request = VideoGenerationRequest(
        prompt="The characters interact and move through the scene",
        image_list=[
            {
                "image": "https://v3.fal.media/files/zebra/qL93Je8ezvzQgDOEzTjKF_KhGKZTEebZcDw6T5rwQPK_output.png",
                "type": "reference",
            },
            {
                "image": "https://v3.fal.media/files/zebra/qL93Je8ezvzQgDOEzTjKF_KhGKZTEebZcDw6T5rwQPK_output.png",
                "type": "reference",
            },
        ],
        duration_seconds=5,
        resolution="720p",
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
