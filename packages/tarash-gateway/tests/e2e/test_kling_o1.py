"""End-to-end tests for Kling O1 models via Fal.

These tests make actual API calls to the Fal.ai Kling O1 service.
Requires FAL_KEY environment variable to be set.

Run with: uv run pytest tests/e2e/test_kling_o1.py -v -m e2e
Skip with: uv run pytest tests/e2e/test_kling_o1.py -v -m "not e2e"
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
async def test_kling_o1_image_to_video_pro(fal_api_key):
    """
    Test Kling O1 Pro image-to-video (first-last-frame).

    Model: fal-ai/kling-video/o1/pro/image-to-video
    Tests: Image-to-video with start and end frames
    """
    progress_updates = []

    async def progress_callback(update: VideoGenerationUpdate):
        progress_updates.append(update)
        print(f"  Progress: {update.status}")

    config = VideoGenerationConfig(
        model="fal-ai/kling-video/o1/image-to-video",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    # Using public test images
    request = VideoGenerationRequest(
        prompt="Create a magical timelapse transition from @Image1 to @Image2. The snow melts rapidly to reveal green grass, and the tree branches burst into bloom with pink flowers.",
        image_list=[
            {
                "image": "https://v3b.fal.media/files/b/rabbit/NaslJIC7F2WodS6DFZRRJ.png",
                "type": "first_frame",
            },
            {
                "image": "https://v3b.fal.media/files/b/tiger/BwHi22qoQnqaTNMMhe533.png",
                "type": "last_frame",
            },
        ],
        duration_seconds=5,
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
async def test_kling_o1_reference_to_video_standard(fal_api_key):
    """
    Test Kling O1 Standard reference-to-video.

    Model: fal-ai/kling-video/o1/standard/reference-to-video
    Tests: Multi-element composition with reference images
    """
    progress_updates = []

    async def progress_callback(update: VideoGenerationUpdate):
        progress_updates.append(update)
        print(f"  Progress: {update.status}")

    config = VideoGenerationConfig(
        model="fal-ai/kling-video/o1/standard/reference-to-video",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    # Using public test images
    request = VideoGenerationRequest(
        prompt="Take @Element1 as the character. Show the character walking through a magical forest with the visual style of @Image1. Camera follows smoothly behind.",
        image_list=[
            # Style reference
            {
                "image": "https://v3b.fal.media/files/b/koala/v9COzzH23FGBYdGLgbK3u.png",
                "type": "reference",
            },
        ],
        duration_seconds=5,
        aspect_ratio="16:9",
        extra_params={
            "elements": [
                {
                    "frontal_image_url": "https://v3b.fal.media/files/b/panda/MQp-ghIqshvMZROKh9lW3.png",
                    "reference_image_urls": [
                        "https://v3b.fal.media/files/b/kangaroo/YMpmQkYt9xugpOTQyZW0O.png"
                    ],
                }
            ]
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
    print(f"  ✓ Progress updates: {len(progress_updates)}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_kling_o1_video_to_video_edit_standard(fal_api_key):
    """
    Test Kling O1 Standard video-to-video/edit.

    Model: fal-ai/kling-video/o1/standard/video-to-video/edit
    Tests: Video editing with element replacement
    """
    progress_updates = []

    async def progress_callback(update: VideoGenerationUpdate):
        progress_updates.append(update)
        print(f"  Progress: {update.status}")

    config = VideoGenerationConfig(
        model="fal-ai/kling-video/o1/standard/video-to-video/edit",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    # Using public test video and images
    request = VideoGenerationRequest(
        prompt="Replace the main character with @Element1 while maintaining the same movements and camera angles. Apply the visual style from @Image1.",
        video="https://v3b.fal.media/files/b/rabbit/ku8_Wdpf-oTbGRq4lB5DU_output.mp4",
        image_list=[
            # Style reference
            {
                "image": "https://v3b.fal.media/files/b/lion/MKvhFko5_wYnfORYacNII_AgPt8v25Wt4oyKhjnhVK5.png",
                "type": "reference",
            },
        ],
        extra_params={
            "elements": [
                {
                    "frontal_image_url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=512&h=512&fit=crop",
                    "reference_image_urls": [
                        "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=512&h=512&fit=crop"
                    ],
                }
            ]
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
    print(f"  ✓ Progress updates: {len(progress_updates)}")


@pytest.mark.e2e
def test_kling_o1_sync_generation(fal_api_key):
    """Test Kling O1 video-to-video/reference with synchronous API."""
    config = VideoGenerationConfig(
        model="fal-ai/kling-video/o1/standard/video-to-video/reference",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    updates_received = []

    def progress_callback(update: VideoGenerationUpdate):
        updates_received.append(update)
        print(f"  Status: {update.status}")

    request = VideoGenerationRequest(
        prompt="Based on @Video1, generate the next shot. keep the style of the video",
        video="https://v3b.fal.media/files/b/rabbit/ku8_Wdpf-oTbGRq4lB5DU_output.mp4",
        extra_params={
            "elements": [
                {
                    "frontal_image_url": "https://v3b.fal.media/files/b/panda/MQp-ghIqshvMZROKh9lW3.png",
                    "reference_image_urls": [
                        "https://v3b.fal.media/files/b/kangaroo/YMpmQkYt9xugpOTQyZW0O.png",
                        "https://v3b.fal.media/files/b/zebra/d6ywajNyJ6bnpa_xBue-K.png",
                    ],
                }
            ]
        },
    )

    # Synchronous generation
    response = api.generate_video(config, request, on_progress=progress_callback)

    assert isinstance(response, VideoGenerationResponse)
    assert response.status == "completed"
    assert response.video is not None
    assert len(updates_received) > 0

    print(f"\n  ✓ Sync generated video URL: {response.video}")
    print(f"  ✓ Received {len(updates_received)} progress updates")
