"""End-to-end tests for Replicate provider.

These tests make actual API calls to the Replicate service.
Requires REPLICATE_API_KEY environment variable to be set.

Run with: pytest tests/e2e/test_replicate.py -v -m e2e
Skip with: pytest tests/e2e/test_replicate.py -v -m "not e2e"
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
def replicate_api_key():
    """Get Replicate API key from environment."""
    api_key = os.getenv("REPLICATE_API_KEY")
    if not api_key:
        pytest.skip("REPLICATE_API_KEY environment variable not set")
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
async def test_minimax_image_to_video(replicate_api_key):
    """
    Test Minimax image-to-video model.

    This tests:
    - Minimax-specific model (minimax/video-01)
    - Field mapping: image_list -> image_url
    - Prefix matching in registry
    - Image-to-video generation with duration
    """
    minimax_config = VideoGenerationConfig(
        model="minimax/video-01",
        provider="replicate",
        api_key=replicate_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="A person walking through a bustling city street",
        duration="6",
        image_list=[
            {
                "image": "https://fal.media/files/elephant/8kkhB12hEZI2kkbU8pZPA_test.jpeg",
                "type": "reference",
            }
        ],
    )

    # Generate video using API (async)
    response = await api.generate_video_async(minimax_config, request)

    # Validate response
    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"

    # Video should be a URL
    assert isinstance(response.video, str), "Video should be a string"
    assert response.video.startswith("http"), (
        f"Expected HTTP URL, got: {response.video}"
    )

    print(f"✓ Generated Minimax video: {response.request_id}")
    print(f"  Video URL: {response.video}")
    print(f"  Model: {minimax_config.model}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_veo31_first_last_frame_to_video(replicate_api_key):
    """
    Test Google Veo 3.1 first-last-frame-to-video.

    This tests:
    - Google Veo 3.1 model (google/veo-3.1)
    - Field mapping: image_list with type="first_frame" and "last_frame"
    - Validates both frames are required and correctly mapped
    - All veo3.1 parameters (aspect_ratio, resolution, duration, generate_audio)
    """
    veo31_config = VideoGenerationConfig(
        model="google/veo-3.1",
        provider="replicate",
        api_key=replicate_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt='A woman looks into the camera, breathes in, then exclaims energetically, "have you guys checked out Veo3.1 First-Last-Frame-to-Video? It\'s incredible!"',
        duration_seconds=8,
        aspect_ratio="16:9",
        resolution="720p",
        image_list=[
            {
                "image": "https://storage.googleapis.com/falserverless/example_inputs/veo31-flf2v-input-1.jpeg",
                "type": "first_frame",
            },
            {
                "image": "https://storage.googleapis.com/falserverless/example_inputs/veo31-flf2v-input-2.jpeg",
                "type": "last_frame",
            },
        ],
        generate_audio=True,
    )

    # Track progress updates
    progress_updates = []

    async def progress_callback(update: VideoGenerationUpdate):
        progress_updates.append(update)
        print(f"  Progress: {update.status}")

    # Generate video using API (async)
    response = await api.generate_video_async(
        veo31_config, request, on_progress=progress_callback
    )

    # Validate response
    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"

    # Video should be a URL
    assert isinstance(response.video, str), "Video should be a string"
    assert response.video.startswith("http"), (
        f"Expected HTTP URL, got: {response.video}"
    )

    # Validate progress tracking
    assert len(progress_updates) > 0, "Should receive at least one progress update"
    statuses = [update.status for update in progress_updates]
    assert "completed" in statuses, "Should receive completed status"

    print(f"✓ Generated veo3.1 first-last-frame-to-video: {response.request_id}")
    print(f"  Video URL: {response.video}")
    print(f"  Model: {veo31_config.model}")
    print("  Duration: 8s")
    print("  Aspect Ratio: 16:9")
    print("  Resolution: 720p")
    print("  Generate Audio: True")
    print(f"  Progress updates: {len(progress_updates)}")
