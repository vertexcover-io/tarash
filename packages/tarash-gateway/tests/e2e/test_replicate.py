"""End-to-end tests for Replicate provider.

These tests make actual API calls to the Replicate service.
Requires REPLICATE_API_TOKEN environment variable to be set.

Run with: pytest tests/e2e/test_replicate.py -v -m e2e
Skip with: pytest tests/e2e/test_replicate.py -v -m "not e2e"
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
def replicate_api_key():
    """Get Replicate API key from environment."""
    api_key = os.getenv("REPLICATE_API_TOKEN")
    if not api_key:
        pytest.skip("REPLICATE_API_TOKEN environment variable not set")
    return api_key


@pytest.fixture(scope="module")
def kling_config(replicate_api_key):
    """Create Kling model configuration."""
    return VideoGenerationConfig(
        model="kwaivgi/kling-v2.1",
        provider="replicate",
        api_key=replicate_api_key,
        timeout=600,
        poll_interval=5,
        max_poll_attempts=120,
    )


@pytest.fixture(scope="module")
def luma_config(replicate_api_key):
    """Create Luma Dream Machine configuration."""
    return VideoGenerationConfig(
        model="luma/dream-machine",
        provider="replicate",
        api_key=replicate_api_key,
        timeout=600,
        poll_interval=5,
        max_poll_attempts=120,
    )


# ==================== E2E Tests ====================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_kling_text_to_video_async(kling_config):
    """Test Kling text-to-video generation with async API."""
    request = VideoGenerationRequest(
        prompt="A beautiful sunset over the ocean with gentle waves",
        duration_seconds=5,
        aspect_ratio="16:9",
    )

    updates_received = []

    async def progress_callback(update: VideoGenerationUpdate):
        updates_received.append(update)
        print(f"Status: {update.status}, Update: {update.update}")

    response = await api.generate_video_async(
        kling_config,
        request,
        on_progress=progress_callback,
    )

    # Verify response
    assert isinstance(response, VideoGenerationResponse)
    assert response.status == "completed"
    assert response.video is not None
    assert str(response.video).startswith("http")
    assert response.request_id is not None

    # Verify we received progress updates
    assert len(updates_received) > 0
    print(f"Received {len(updates_received)} progress updates")
    print(f"Video URL: {response.video}")


@pytest.mark.e2e
def test_kling_text_to_video_sync(kling_config):
    """Test Kling text-to-video generation with sync API."""
    request = VideoGenerationRequest(
        prompt="A cat playing with a ball of yarn",
        duration_seconds=5,
    )

    updates_received = []

    def progress_callback(update: VideoGenerationUpdate):
        updates_received.append(update)
        print(f"Status: {update.status}")

    response = api.generate_video(
        kling_config,
        request,
        on_progress=progress_callback,
    )

    # Verify response
    assert isinstance(response, VideoGenerationResponse)
    assert response.status == "completed"
    assert response.video is not None
    print(f"Video URL: {response.video}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_kling_image_to_video_async(kling_config):
    """Test Kling image-to-video generation with async API."""
    # Using a public image URL for testing
    test_image_url = (
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800"
    )

    request = VideoGenerationRequest(
        prompt="The mountains come alive with movement, clouds drifting slowly",
        duration_seconds=5,
        image_list=[{"image": test_image_url, "type": "reference"}],
    )

    response = await api.generate_video_async(kling_config, request)

    assert isinstance(response, VideoGenerationResponse)
    assert response.status == "completed"
    assert response.video is not None
    print(f"Video URL: {response.video}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_replicate_without_progress_callback(kling_config):
    """Test Replicate generation without progress callback (simpler flow)."""
    request = VideoGenerationRequest(
        prompt="A gentle river flowing through a forest",
        duration_seconds=5,
    )

    # No progress callback - uses async_run directly
    response = await api.generate_video_async(kling_config, request)

    assert isinstance(response, VideoGenerationResponse)
    assert response.status == "completed"
    assert response.video is not None
    print(f"Video URL: {response.video}")


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
async def test_kling_10_second_video(kling_config):
    """Test Kling 10-second video generation (longer, marked as slow)."""
    request = VideoGenerationRequest(
        prompt="A time-lapse of flowers blooming in a garden",
        duration_seconds=10,
    )

    response = await api.generate_video_async(kling_config, request)

    assert isinstance(response, VideoGenerationResponse)
    assert response.status == "completed"
    print(f"Video URL: {response.video}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_luma_text_to_video_async(luma_config):
    """Test Luma Dream Machine text-to-video generation."""
    request = VideoGenerationRequest(
        prompt="A serene lake at dawn with mist rising from the water",
        aspect_ratio="16:9",
    )

    updates_received = []

    async def progress_callback(update: VideoGenerationUpdate):
        updates_received.append(update)
        print(f"Status: {update.status}")

    response = await api.generate_video_async(
        luma_config,
        request,
        on_progress=progress_callback,
    )

    assert isinstance(response, VideoGenerationResponse)
    assert response.status == "completed"
    assert response.video is not None
    print(f"Video URL: {response.video}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_replicate_with_extra_params(kling_config):
    """Test Replicate generation with extra params passed through."""
    request = VideoGenerationRequest(
        prompt="A cyberpunk city at night with neon lights",
        duration_seconds=5,
        extra_params={
            "cfg_scale": 0.7,
        },
    )

    response = await api.generate_video_async(kling_config, request)

    assert isinstance(response, VideoGenerationResponse)
    assert response.status == "completed"
    print(f"Video URL: {response.video}")


@pytest.mark.e2e
def test_replicate_direct_handler_usage(replicate_api_key):
    """Test using ReplicateProviderHandler directly."""
    from tarash.tarash_gateway.video.providers.replicate import ReplicateProviderHandler

    handler = ReplicateProviderHandler()

    config = VideoGenerationConfig(
        model="kwaivgi/kling-v2.1",
        provider="replicate",
        api_key=replicate_api_key,
        poll_interval=5,
        max_poll_attempts=60,
    )

    request = VideoGenerationRequest(
        prompt="A peaceful meadow with butterflies",
        duration_seconds=5,
    )

    response = handler.generate_video(config, request)

    assert isinstance(response, VideoGenerationResponse)
    assert response.status == "completed"
    print(f"Video URL: {response.video}")
