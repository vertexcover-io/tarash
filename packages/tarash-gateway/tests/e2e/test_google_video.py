"""End-to-end tests for Google provider video generation using google-genai.

These tests make actual API calls to Google's Veo 3 video generation service.
Requires GOOGLE_API_KEY environment variable to be set.
For Vertex AI, set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION.

Run with: pytest tests/e2e/test_google_video.py -v -m e2e
Skip with: pytest tests/e2e/test_google_video.py -v -m "not e2e"
"""

import os
from pathlib import Path
from urllib.request import urlopen

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
def google_api_key():
    """Get Google API key from environment."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY environment variable not set")
    return api_key


@pytest.fixture(scope="module")
def google_video_config(google_api_key):
    """Create Google video generation configuration."""
    return VideoGenerationConfig(
        model="veo-3.1-generate-preview",
        provider="google",
        api_key=google_api_key,
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
@pytest.mark.asyncio
async def test_comprehensive_async_video_generation(google_video_config):
    """
    Comprehensive async test combining:
    - Basic video generation
    - Progress tracking
    - Custom parameters (seed, negative_prompt, person_generation, enhance_prompt)
    - Different image types
    """
    # Track progress updates
    progress_updates = []

    async def progress_callback(update: VideoGenerationUpdate):
        progress_updates.append(update)
        print(f"  Progress: {update.status}")

    # Test with various parameters
    request = VideoGenerationRequest(
        prompt="A serene lake at sunset with mountains in the background, cinematic quality",
        duration_seconds=5,
        aspect_ratio="16:9",
        negative_prompt="blur, low quality, distorted",
    )

    # Generate video using API
    response = await api.generate_video_async(
        google_video_config, request, on_progress=progress_callback
    )

    # Validate response
    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"

    # Validate raw_response structure
    assert isinstance(response.raw_response, dict)
    assert response.raw_response is not None, "raw_response should not be None"

    # Video should be a URL string (http or gs://) or dict with content
    if isinstance(response.video, str):
        assert response.video.startswith("http") or response.video.startswith(
            "gs://"
        ), f"Expected HTTP or GCS URL, got: {response.video}"
        video_type = "URL"
        video_info = response.video
    else:
        # Dict with binary content
        assert "content" in response.video, "Video dict should have 'content' field"
        assert "content_type" in response.video, (
            "Video dict should have 'content_type' field"
        )
        video_type = "bytes"
        video_info = f"{len(response.video['content'])} bytes, type: {response.video['content_type']}"

    # Validate progress tracking
    assert len(progress_updates) > 0, "Should receive at least one progress update"
    statuses = [update.status for update in progress_updates]

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
def test_sync_video_generation_with_all_image_types(google_video_config):
    """
    Sync test for Google Veo 3 interpolation mode:
    - Basic sync generation
    - Interpolation mode: first_frame + last_frame
    - 16:9 aspect ratio (supports both 16:9 and 9:16 for interpolation)
    - 8 second duration (required for interpolation)
    - Images passed as bytes with content_type

    Note: Cannot combine interpolation (first/last frames) with reference images (asset/style).
    These are mutually exclusive modes in Veo 3.1.
    """
    sample_image_urls = [
        "https://picsum.photos/1280/720?random=1",  # 16:9 aspect ratio
        "https://picsum.photos/1280/720?random=2",  # 16:9 aspect ratio
    ]

    # Download images and convert to bytes
    image_data = []
    for url in sample_image_urls:
        with urlopen(url) as response:
            image_bytes = response.read()
            image_data.append(image_bytes)

    print("Image data downloaded successfully")

    request = VideoGenerationRequest(
        prompt="Smooth transition from a peaceful morning scene to a vibrant sunset landscape",
        duration_seconds=8,  # Must be 8 when using interpolation
        aspect_ratio="16:9",
        image_list=[
            # Interpolation mode: first_frame + last_frame
            {
                "image": {
                    "content": image_data[0],
                    "content_type": "image/jpeg",
                },
                "type": "first_frame",
            },
            {
                "image": {
                    "content": image_data[1],
                    "content_type": "image/jpeg",
                },
                "type": "last_frame",
            },
            # Note: Cannot add asset/style reference images when using interpolation mode
        ],
    )

    # Generate video using API (sync)
    response = api.generate_video(google_video_config, request)

    # Validate response
    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"

    # Validate raw_response structure
    assert isinstance(response.raw_response, dict)
    assert response.raw_response is not None, "raw_response should not be None"

    # Video should be a URL string (http or gs://) or dict with content
    if isinstance(response.video, str):
        assert response.video.startswith("http") or response.video.startswith(
            "gs://"
        ), f"Expected HTTP or GCS URL, got: {response.video}"
        video_type = "URL"
        video_info = response.video
    else:
        # Dict with binary content
        assert "content" in response.video, "Video dict should have 'content' field"
        assert "content_type" in response.video, (
            "Video dict should have 'content_type' field"
        )
        video_type = "bytes"
        video_info = f"{len(response.video['content'])} bytes, type: {response.video['content_type']}"

    # Log details
    print("✓ Generated video with multiple image types successfully")
    print(f"  Request ID: {response.request_id}")
    print(f"  Video type: {video_type}")
    print(f"  Video info: {video_info}")
    print("  Image types used: first_frame, last_frame, asset")
    print(
        f"  Duration: {response.duration}s" if response.duration else "  Duration: N/A"
    )
    print(
        f"  Resolution: {response.resolution}"
        if response.resolution
        else "  Resolution: N/A"
    )


@pytest.mark.e2e
def test_sync_video_generation_with_local_image(google_video_config, shoe_image_path):
    """
    Sync test for Google Veo 3 reference images mode:
    - Basic sync generation
    - Reference images mode: 3 asset images from local file (shoe-image.jpg used 3 times)
    - 16:9 aspect ratio (required for reference images)
    - 8 seconds duration (required for reference images)
    - Binary image content

    Note: Reference images mode preserves the subject's appearance across the video.
    """
    # Read the image file
    with open(shoe_image_path, "rb") as f:
        image_content = f.read()

    request = VideoGenerationRequest(
        prompt="A sleek athletic shoe rotating slowly in a studio setting, showcasing its design from all angles",
        duration_seconds=8,  # Must be 8 when using reference images
        aspect_ratio="16:9",  # Must be 16:9 when using reference images
        image_list=[
            # Reference images mode: up to 3 asset images
            {
                "image": {
                    "content": image_content,
                    "content_type": "image/jpeg",
                },
                "type": "asset",
            },
            {
                "image": {
                    "content": image_content,
                    "content_type": "image/jpeg",
                },
                "type": "asset",
            },
            {
                "image": {
                    "content": image_content,
                    "content_type": "image/jpeg",
                },
                "type": "asset",
            },
        ],
    )

    # Generate video using API (sync)
    response = api.generate_video(google_video_config, request)

    # Validate response
    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"

    # Validate raw_response structure
    assert isinstance(response.raw_response, dict)
    assert response.raw_response is not None, "raw_response should not be None"

    # Video should be a URL string (http or gs://) or dict with content
    if isinstance(response.video, str):
        assert response.video.startswith("http") or response.video.startswith(
            "gs://"
        ), f"Expected HTTP or GCS URL, got: {response.video}"
        video_type = "URL"
        video_info = response.video
    else:
        # Dict with binary content
        assert "content" in response.video, "Video dict should have 'content' field"
        assert "content_type" in response.video, (
            "Video dict should have 'content_type' field"
        )
        video_type = "bytes"
        video_size_mb = len(response.video["content"]) / (1024 * 1024)
        video_info = f"{video_size_mb:.2f} MB, type: {response.video['content_type']}"

    # Log details
    print("✓ Generated video with local reference image successfully")
    print(f"  Request ID: {response.request_id}")
    print(f"  Video type: {video_type}")
    print(f"  Video info: {video_info}")
    print(f"  Model: {google_video_config.model}")
    print(
        f"  Duration: {response.duration}s" if response.duration else "  Duration: N/A"
    )
    print(
        f"  Resolution: {response.resolution}"
        if response.resolution
        else "  Resolution: N/A"
    )
