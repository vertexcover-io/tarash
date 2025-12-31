"""End-to-end tests for OpenAI provider.

These tests make actual API calls to the OpenAI Sora service.
Requires OPENAI_API_KEY environment variable to be set.

Run with: pytest tests/e2e/test_openai.py -v --e2e
Skip with: pytest tests/e2e/test_openai.py -v
"""

import os
from pathlib import Path

import pytest

from tarash.tarash_gateway.video import api
from tarash.tarash_gateway.video.models import (
    ImageType,
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
def shoe_image() -> ImageType:
    """Load shoe image from test media directory and return as ImageType."""
    # Get the path to the shoe image
    test_dir = Path(__file__).parent.parent
    image_path = test_dir / "media" / "shoe-image.png"

    if not image_path.exists():
        pytest.skip(f"Shoe image not found at {image_path}")

    # Read the image as bytes
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Return as ImageType with MediaContent
    return {
        "image": {
            "content": image_bytes,
            "content_type": "image/png",
        },
        "type": "reference",
    }


# ==================== E2E Tests ====================


@pytest.mark.e2e
def test_sync_image_to_video_with_sora2(openai_api_key, shoe_image):
    """
    Test synchronous image-to-video generation with OpenAI Sora 2.

    This tests:
    - Sora 2 model
    - Image-to-video with local image file
    - Sync generation
    - Valid duration for Sora 2 (4, 8, or 12 seconds)
    - Aspect ratio conversion
    """
    openai_config = VideoGenerationConfig(
        model="sora-2",
        provider="openai",
        api_key=openai_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="A stylish athletic shoe rotating slowly on a white background, product photography, studio lighting",
        duration_seconds=4,  # Valid for Sora 2: 4, 8, or 12
        aspect_ratio="16:9",
        image_list=[shoe_image],
    )

    # Generate video using API (sync)
    response = api.generate_video(openai_config, request)

    # Validate response
    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"

    # Video should be MediaContent (dict with bytes)
    assert isinstance(response.video, dict), "Video should be a dict (MediaContent)"
    assert "content" in response.video, "Video should have 'content' field"
    assert "content_type" in response.video, "Video should have 'content_type' field"
    assert isinstance(response.video["content"], bytes), "Video content should be bytes"
    assert response.video["content_type"] == "video/mp4", (
        "Video content type should be video/mp4"
    )
    assert len(response.video["content"]) > 0, "Video content should not be empty"

    # Validate duration
    if response.duration:
        assert response.duration == 4.0, f"Expected 4 seconds, got {response.duration}"

    print(f"✓ Generated Sora 2 video with shoe image: {response.request_id}")
    print(f"  Video size: {len(response.video['content'])} bytes")
    print(f"  Content type: {response.video['content_type']}")
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
async def test_async_text_to_video_with_sora2(openai_api_key):
    """
    Test asynchronous text-to-video generation with OpenAI Sora 2.

    This tests:
    - Sora 2 model
    - Text-to-video (no image input)
    - Async generation
    - 4 second duration
    - Portrait aspect ratio (9:16)
    """
    openai_config = VideoGenerationConfig(
        model="sora-2",
        provider="openai",
        api_key=openai_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="A serene waterfall cascading down mossy rocks in a lush forest, cinematic lighting, high quality",
        duration_seconds=4,  # Valid for Sora 2: 4, 8, or 12
        aspect_ratio="9:16",  # Portrait format
    )

    # Generate video using API (async)
    response = await api.generate_video_async(openai_config, request)

    # Validate response
    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"

    # Video should be MediaContent (dict with bytes)
    assert isinstance(response.video, dict), "Video should be a dict (MediaContent)"
    assert "content" in response.video, "Video should have 'content' field"
    assert "content_type" in response.video, "Video should have 'content_type' field"
    assert isinstance(response.video["content"], bytes), "Video content should be bytes"
    assert response.video["content_type"] == "video/mp4", (
        "Video content type should be video/mp4"
    )
    assert len(response.video["content"]) > 0, "Video content should not be empty"

    # Validate duration
    if response.duration:
        assert response.duration == 4.0, f"Expected 4 seconds, got {response.duration}"

    # Validate resolution (9:16 aspect ratio should map to 720x1280)
    if response.resolution:
        assert response.resolution == "720x1280", (
            f"Expected 720x1280 for 9:16 aspect ratio, got {response.resolution}"
        )

    print(f"✓ Generated Sora 2 text-to-video (async): {response.request_id}")
    print(f"  Video size: {len(response.video['content'])} bytes")
    print(f"  Content type: {response.video['content_type']}")
    print(
        f"  Duration: {response.duration}s" if response.duration else "  Duration: N/A"
    )
    print(
        f"  Resolution: {response.resolution}"
        if response.resolution
        else "  Resolution: N/A"
    )
