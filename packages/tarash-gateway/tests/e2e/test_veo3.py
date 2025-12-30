"""End-to-end tests for Veo3 provider using google-genai.

These tests make actual API calls to Google's Veo3 service.
Requires GOOGLE_API_KEY environment variable to be set.
For Vertex AI, set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION.

Run with: pytest tests/e2e/test_veo3.py -v -m e2e
Skip with: pytest tests/e2e/test_veo3.py -v -m "not e2e"
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
def google_api_key():
    """Get Google API key from environment."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY environment variable not set")
    return api_key


@pytest.fixture(scope="module")
def veo3_config(google_api_key):
    """Create Veo3 configuration."""
    return VideoGenerationConfig(
        model="veo-3.0-flash-001",
        provider="veo3",
        api_key=google_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )


# ==================== E2E Tests ====================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_comprehensive_async_video_generation(veo3_config):
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
        seed=42,
        negative_prompt="blur, low quality, distorted",
        generate_audio=True,
        model_params={
            "enhance_prompt": True,
            "person_generation": "allow_adult",
        },
    )

    # Generate video using API
    response = await api.generate_video_async(
        veo3_config, request, on_progress=progress_callback
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
def test_sync_video_generation_with_all_image_types(veo3_config):
    """
    Sync test combining:
    - Basic sync generation
    - All image types: first_frame (only one of reference/first_frame allowed), last_frame, asset, style
    - Different aspect ratios
    """
    sample_image_urls = [
        "https://picsum.photos/512/512?random=1",
        "https://picsum.photos/512/512?random=2",
        "https://picsum.photos/512/512?random=3",
        "https://picsum.photos/512/512?random=4",
    ]

    request = VideoGenerationRequest(
        prompt="Zoom out from this scene to reveal a vast landscape with flying birds",
        duration_seconds=5,
        aspect_ratio="1:1",
        image_list=[
            # Only ONE of reference or first_frame is allowed
            {"image": sample_image_urls[0], "type": "first_frame"},
            # But can combine with last_frame, asset, style
            {"image": sample_image_urls[1], "type": "last_frame"},
            {"image": sample_image_urls[2], "type": "asset"},
            {"image": sample_image_urls[3], "type": "style"},
        ],
    )

    # Generate video using API (sync)
    response = api.generate_video(veo3_config, request)

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
    print("✓ Generated video with all image types successfully")
    print(f"  Request ID: {response.request_id}")
    print(f"  Video type: {video_type}")
    print(f"  Video info: {video_info}")
    print("  Image types used: first_frame, last_frame, asset, style")
    print(
        f"  Duration: {response.duration}s" if response.duration else "  Duration: N/A"
    )
    print(
        f"  Resolution: {response.resolution}"
        if response.resolution
        else "  Resolution: N/A"
    )
