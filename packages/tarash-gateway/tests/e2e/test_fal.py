"""End-to-end tests for Fal provider.

These tests make actual API calls to the Fal.ai service.
Requires FAL_KEY environment variable to be set.

Run with: pytest tests/e2e/test_fal.py -v -m e2e
Skip with: pytest tests/e2e/test_fal.py -v -m "not e2e"
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
async def test_comprehensive_async_video_generation(fal_api_key):
    """
    Comprehensive async test combining:
    - Basic video generation
    - Progress tracking
    - Custom parameters (seed, negative_prompt)
    - Different image types
    """
    # Track progress updates
    progress_updates = []

    async def progress_callback(update: VideoGenerationUpdate):
        progress_updates.append(update)
        print(f"  Progress: {update.status}")

    # Test with various parameters
    config = VideoGenerationConfig(
        model="fal-ai/veo3.1/fast",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    req = VideoGenerationRequest(
        prompt="A serene lake at sunset with mountains in the background, cinematic quality",
        duration_seconds=4,
        aspect_ratio="16:9",
        resolution="720p",
        seed=42,
        negative_prompt="blur, low quality",
        generate_audio=True,
        auto_fix=True,
    )

    # Generate video using API
    response = await api.generate_video_async(
        config, req, on_progress=progress_callback
    )

    # Validate response
    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"

    # Validate raw_response structure
    assert isinstance(response.raw_response, dict)
    assert "video" in response.raw_response, "raw_response should contain 'video' field"

    # Video should be a URL string (Fal returns URLs)
    assert isinstance(response.video, str), "Video should be a string"
    assert response.video.startswith("http"), (
        f"Expected HTTP URL, got: {response.video}"
    )
    video_type = "URL"
    video_info = response.video
    # Validate progress tracking
    assert len(progress_updates) > 0, "Should receive at least one progress update"
    statuses = [update.status for update in progress_updates]
    assert "completed" in statuses, "Should receive completed status"

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
def test_sync_video_generation_with_images(fal_api_key):
    """
    Sync test combining:
    - Basic sync generation
    - Reference images
    - Different aspect ratios
    """
    fal_config = VideoGenerationConfig(
        model="fal-ai/veo3.1/fast/image-to-video",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="A calm ocean wave rolling onto a sandy beach, inspired by the reference style",
        duration_seconds=5,
        aspect_ratio="9:16",
        image_url="https://storage.googleapis.com/falserverless/example_inputs/veo31_i2v_input.jpg",
    )

    # Generate video using API (sync)
    response = api.generate_video(fal_config, request)

    # Validate response
    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"

    print(f"✓ Generated video with reference image: {response.request_id}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_minimax_image_to_video(fal_api_key):
    """
    Test Minimax image-to-video model.

    This tests:
    - Minimax-specific model (fal-ai/minimax/hailuo-02-fast/image-to-video)
    - Field mapping: image_list -> image_url
    - Prefix matching in registry
    - Only prompt and image_url fields
    """
    minimax_config = VideoGenerationConfig(
        model="fal-ai/minimax/hailuo-02-fast/image-to-video",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="A person walking through a bustling city street",
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
async def test_kling_motion_control(fal_api_key):
    """
    Test Kling motion control model with image and video inputs.

    This tests:
    - Kling motion control model (fal-ai/kling-video/v2.6/standard/motion-control)
    - Image-to-video with motion guidance from reference video
    - Character orientation set to "video"
    - Keep original sound enabled
    - Field mapping: image_list -> image_url, video -> video_url
    """
    kling_config = VideoGenerationConfig(
        model="fal-ai/kling-video/v2.6/standard/motion-control",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="An african american woman dancing",
        image_list=[
            {
                "image": "https://v3b.fal.media/files/b/0a875302/8NaxQrQxDNHppHtqcchMm.png",
                "type": "reference",
            }
        ],
        video="https://v3b.fal.media/files/b/0a8752bc/2xrNS217ngQ3wzXqA7LXr_output.mp4",
        character_orientation="video",
        keep_original_sound=True,
    )

    # Generate video using API (async)
    response = await api.generate_video_async(kling_config, request)

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

    print(f"✓ Generated Kling motion-controlled video: {response.request_id}")
    print(f"  Video URL: {response.video}")
    print(f"  Model: {kling_config.model}")
    print("  Character orientation: video")
    print("  Keep original sound: True")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_veo31_image_to_video_with_image_list(fal_api_key):
    """
    Test veo3.1 image-to-video using image_list parameter.

    This tests:
    - Veo 3.1 image-to-video model (fal-ai/veo3.1/fast/image-to-video)
    - Field mapping: image_list with type="reference" -> image_url
    - All veo3.1 parameters (aspect_ratio, resolution, generate_audio, auto_fix)
    - Prefix matching in registry
    """
    veo31_config = VideoGenerationConfig(
        model="fal-ai/veo3.1/fast/image-to-video",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="A serene mountain landscape with clouds slowly moving across the sky",
        duration_seconds=6,
        aspect_ratio="16:9",
        resolution="720p",
        image_list=[
            {
                "image": "https://storage.googleapis.com/falserverless/example_inputs/veo31_i2v_input.jpg",
                "type": "reference",
            }
        ],
        generate_audio=True,
        auto_fix=True,
    )

    # Generate video using API (async)
    response = await api.generate_video_async(veo31_config, request)

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

    print(f"✓ Generated veo3.1 image-to-video: {response.request_id}")
    print(f"  Video URL: {response.video}")
    print(f"  Model: {veo31_config.model}")
    print("  Duration: 6s")
    print("  Aspect Ratio: 16:9")
    print("  Resolution: 720p")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_veo31_first_last_frame_to_video(fal_api_key):
    """
    Test veo3.1 first-last-frame-to-video.

    This tests:
    - Veo 3.1 first-last-frame model (fal-ai/veo3.1/fast/first-last-frame-to-video)
    - Field mapping: image_list with type="first_frame" and "last_frame"
    - Validates both frames are required and correctly mapped
    - All veo3.1 parameters
    """
    veo31_config = VideoGenerationConfig(
        model="fal-ai/veo3.1/fast/first-last-frame-to-video",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt='A woman looks into the camera, breathes in, then exclaims energetically, "have you guys checked out Veo3.1 First-Last-Frame-to-Video on Fal? It\'s incredible!"',
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
        auto_fix=False,
    )

    # Generate video using API (async)
    response = await api.generate_video_async(veo31_config, request)

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

    print(f"✓ Generated veo3.1 first-last-frame-to-video: {response.request_id}")
    print(f"  Video URL: {response.video}")
    print(f"  Model: {veo31_config.model}")
    print("  Duration: 8s")
    print("  Aspect Ratio: 16:9")
    print("  Resolution: 720p")
    print("  Generate Audio: True")
