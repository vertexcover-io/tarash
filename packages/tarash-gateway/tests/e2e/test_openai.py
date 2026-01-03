"""End-to-end tests for OpenAI provider (Sora).

These tests make actual API calls to the OpenAI Sora service.
Requires OPENAI_API_KEY environment variable to be set.

Run with: pytest tests/e2e/test_openai.py -v -m e2e
Skip with: pytest tests/e2e/test_openai.py -v -m "not e2e"
"""

import os
from pathlib import Path

import pytest

from tarash.tarash_gateway.video import api
from tarash.tarash_gateway.video.models import (
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
)

# ==================== Fixtures ====================


@pytest.fixture(scope="module")
def openai_api_key():
    """Get OpenAI API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    return api_key


@pytest.fixture(scope="module")
def openai_config(openai_api_key):
    """Create OpenAI configuration for Sora 2."""
    return VideoGenerationConfig(
        model="sora-2",
        provider="openai",
        api_key=openai_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )


@pytest.fixture(scope="module")
def shoe_image_path():
    """Get path to the shoe test image."""
    # Path relative to test file
    test_dir = Path(__file__).parent.parent
    image_path = test_dir / "media" / "shoe-image.jpg"

    if not image_path.exists():
        pytest.skip(f"Test image not found at {image_path}")

    return str(image_path)


# ==================== E2E Tests ====================


@pytest.mark.e2e
def test_sync_video_generation_with_image(openai_config, shoe_image_path):
    """
    Sync test for OpenAI Sora image-to-video:
    - Basic sync generation
    - Reference image (shoe-image.jpg)
    - 9:16 aspect ratio (portrait)
    - 4 seconds duration
    - sora-2 model (not pro)
    """
    # Read the image file
    with open(shoe_image_path, "rb") as f:
        image_content = f.read()

    request = VideoGenerationRequest(
        prompt="A sleek athletic shoe rotating slowly, showcasing its design from all angles",
        duration_seconds=4,
        aspect_ratio="9:16",
        image_list=[
            {
                "image": {
                    "content": image_content,
                    "content_type": "image/jpeg",
                },
                "type": "reference",
            }
        ],
    )

    # Generate video using API (sync)
    response = api.generate_video(openai_config, request)

    # Validate response
    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"

    # Validate raw_response structure
    assert isinstance(response.raw_response, dict)
    assert response.raw_response is not None, "raw_response should not be None"

    # Video should be a dict with content (OpenAI returns bytes)
    assert isinstance(response.video, dict), "Video should be a dict with content"
    assert "content" in response.video, "Video dict should have 'content' field"
    assert "content_type" in response.video, (
        "Video dict should have 'content_type' field"
    )

    video_size_mb = len(response.video["content"]) / (1024 * 1024)

    # Log details
    print("✓ Generated video with reference image successfully")
    print(f"  Request ID: {response.request_id}")
    print("  Video type: bytes")
    print(f"  Video size: {video_size_mb:.2f} MB")
    print(f"  Content type: {response.video['content_type']}")
    print(f"  Model: {openai_config.model}")
    print(
        f"  Duration: {response.duration}s" if response.duration else "  Duration: N/A"
    )
    print(
        f"  Resolution: {response.resolution}"
        if response.resolution
        else "  Resolution: N/A"
    )


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_sora2_remix_async(openai_config):
    """
    Async test for OpenAI Sora 2 video remix:
    - Uses client.videos.remix() endpoint
    - Requires video_id from a previous Sora 2 generation
    - Tests async generation with progress tracking
    - Remixes existing video with new prompt

    Note: This test requires a valid video_id from a previous generation.
    If you don't have one, first run test_sync_video_generation_with_image
    and use the returned video ID.
    """
    # You'll need to replace this with an actual video_id from a previous generation
    # For testing purposes, we'll skip if VIDEO_ID env var is not set
    video_id = os.getenv("SORA_VIDEO_ID")
    if not video_id:
        pytest.skip(
            "SORA_VIDEO_ID environment variable not set. "
            "Run a generation test first to get a video_id, then set: "
            "export SORA_VIDEO_ID=<your_video_id>"
        )

    # Track progress updates
    progress_updates = []

    async def progress_callback(update):
        progress_updates.append(update)
        print(f"  Progress: {update.status}")

    request = VideoGenerationRequest(
        prompt="Shift the color palette to teal, sand, and rust, with a warm backlight",
        extra_params={"video_id": video_id},
    )

    # Generate remixed video using API (async)
    response = await api.generate_video_async(
        openai_config, request, on_progress=progress_callback
    )

    # Validate response
    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"

    # Video should be a dict with content (OpenAI returns bytes)
    assert isinstance(response.video, dict), "Video should be a dict with content"
    assert "content" in response.video, "Video dict should have 'content' field"
    assert "content_type" in response.video

    # Validate progress tracking
    assert len(progress_updates) > 0, "Should receive at least one progress update"
    statuses = [update.status for update in progress_updates]
    assert "completed" in statuses, "Should receive completed status"

    video_size_mb = len(response.video["content"]) / (1024 * 1024)

    print(f"✓ Remixed video successfully: {response.request_id}")
    print(f"  Original video ID: {video_id}")
    print(f"  New video ID: {response.request_id}")
    print("  Video type: bytes")
    print(f"  Video size: {video_size_mb:.2f} MB")
    print(f"  Content type: {response.video['content_type']}")
    print(f"  Model: {openai_config.model}")
    print(f"  Progress updates: {len(progress_updates)}")
    print(f"  Statuses: {statuses}")
