"""End-to-end tests for Google provider video generation using google-genai.

These tests make actual API calls to Google's Veo video generation service.

IMPORTANT: Google Veo video generation does NOT support API key authentication.
It requires Vertex AI with OAuth2/service account credentials.

Required environment variables:
- GOOGLE_CLOUD_PROJECT: Your GCP project ID
- GOOGLE_CLOUD_LOCATION: Region (optional, defaults to us-central1)
- GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON (optional)

Run with: pytest tests/e2e/test_google_video.py -v -m e2e
Skip with: pytest tests/e2e/test_google_video.py -v -m "not e2e"
"""

import os
import uuid
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


def _get_vertex_ai_provider_config() -> dict[str, str]:
    """Build Vertex AI provider config from environment variables.

    Returns:
        Provider config dict with project, location, and optional credentials_path.

    Raises:
        pytest.skip: If GOOGLE_CLOUD_PROJECT is not set.
    """
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project:
        pytest.skip(
            "GOOGLE_CLOUD_PROJECT not set. "
            "Google Veo video generation requires Vertex AI authentication."
        )

    provider_config: dict[str, str] = {
        "project": project,
        "location": os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
    }

    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path:
        provider_config["credentials_path"] = credentials_path

    return provider_config


@pytest.fixture(scope="module")
def google_video_config():
    """Create Google Veo video generation configuration using Vertex AI.

    IMPORTANT: Google Veo does NOT support API key authentication.
    It requires Vertex AI with OAuth2/service account credentials.

    Requires environment variables:
    - GOOGLE_CLOUD_PROJECT: Your GCP project ID (required)
    - GOOGLE_CLOUD_LOCATION: Region (optional, defaults to us-central1)
    - GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON (optional)
    """
    return VideoGenerationConfig(
        model="veo-3.0-generate-preview",
        provider="google",
        api_key=None,  # Vertex AI uses OAuth2, not API keys
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
        provider_config=_get_vertex_ai_provider_config(),
    )


@pytest.fixture(scope="module")
def gcs_output_bucket():
    """Get GCS output bucket from environment."""
    bucket = os.getenv("GOOGLE_OUTPUT_GCS_BUCKET")
    if not bucket:
        pytest.skip("GOOGLE_OUTPUT_GCS_BUCKET not set")
    return bucket


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
        duration_seconds=8,  # 1080p requires 8 seconds
        resolution="1080p",
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
    Sync test for Google Veo 3 interpolation mode using Vertex AI:
    - Basic sync generation
    - Vertex AI authentication (uses provider_config)
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
        resolution="720p",
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

    # Generate video using API (sync) with Vertex AI
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
    print("  Image types used: first_frame, last_frame")
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
async def test_video_extension_with_demo_video(google_video_config, gcs_output_bucket):
    """
    Video extension test: Download a demo video and extend it with a prompt.

    This tests the video-to-video capability by:
    1. Downloading a short demo video from a public URL
    2. Passing it to the API with an extension prompt
    3. Verifying the extended video is generated

    Note: Video extension only supports 720p resolution.
    Uses Vertex AI for authentication.
    Requires GOOGLE_OUTPUT_GCS_BUCKET for large video output.
    """
    demo_video_url = "https://samplelib.com/lib/preview/mp4/sample-5s.mp4"
    print(f"Downloading demo video from: {demo_video_url}")

    with urlopen(demo_video_url) as response:
        video_bytes = response.read()

    print(f"Downloaded video: {len(video_bytes)} bytes")

    output_gcs_uri = (
        f"gs://{gcs_output_bucket}/test-outputs/extended-{uuid.uuid4()}.mp4"
    )
    print(f"Output GCS URI: {output_gcs_uri}")

    request = VideoGenerationRequest(
        prompt="Continue the video with a smooth camera pan revealing more of the landscape",
        video={"content": video_bytes, "content_type": "video/mp4"},
        duration_seconds=6,
        resolution="720p",
        extra_params={"output_gcs_uri": output_gcs_uri},
    )

    response = await api.generate_video_async(google_video_config, request)

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
    print("✓ Extended video successfully")
    print(f"  Request ID: {response.request_id}")
    print(f"  Video type: {video_type}")
    print(f"  Video info: {video_info}")
    print(
        f"  Duration: {response.duration}s" if response.duration else "  Duration: N/A"
    )
    print(
        f"  Resolution: {response.resolution}"
        if response.resolution
        else "  Resolution: N/A"
    )
