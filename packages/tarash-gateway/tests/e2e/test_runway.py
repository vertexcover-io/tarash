"""End-to-end tests for Runway ML provider.

These tests make actual API calls to the Runway ML service.
Requires RUNWAYML_API_SECRET environment variable to be set.

Run with: uv run pytest tests/e2e/test_runway.py -v -m e2e
Skip with: uv run pytest tests/e2e/test_runway.py -v -m "not e2e"
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
def runway_api_key():
    """Get Runway API key from environment."""
    api_key = os.getenv("RUNWAYML_API_SECRET")
    if not api_key:
        pytest.skip("RUNWAYML_API_SECRET environment variable not set")
    return api_key


# ==================== E2E Tests ====================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_text_to_video_veo31_async(runway_api_key):
    """
    Test Runway text-to-video with Veo 3.1.

    This tests:
    - Text-to-video generation
    - Veo 3.1 model
    - Async operation
    - Progress tracking
    - Duration validation (4, 6, or 8 seconds)
    - Audio generation
    """
    # Track progress updates
    progress_updates = []

    async def progress_callback(update: VideoGenerationUpdate):
        progress_updates.append(update)
        print(f"  Progress: {update.status}")

    config = VideoGenerationConfig(
        model="veo3.1",
        provider="runway",
        api_key=runway_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="A cute bunny hopping in a meadow at sunset, cinematic quality",
        duration_seconds=4,
        aspect_ratio="16:9",
        generate_audio=True,
        seed=42,
    )

    # Generate video using API
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

    # Video should be a URL string
    assert isinstance(response.video, str), "Video should be a string"
    assert response.video.startswith("http"), (
        f"Expected HTTP URL, got: {response.video}"
    )

    # Validate progress tracking
    assert len(progress_updates) > 0, "Should receive at least one progress update"
    statuses = [update.status for update in progress_updates]
    assert "completed" in statuses, "Should receive completed status"

    # Log details
    print("✓ Generated text-to-video successfully")
    print(f"  Request ID: {response.request_id}")
    print(f"  Video URL: {response.video}")
    print(f"  Progress updates: {len(progress_updates)}")
    print(f"  Statuses: {statuses}")
    print("  Model: veo3.1")
    print("  Duration: 4s")
    print("  Aspect Ratio: 16:9")
    print("  Audio: Enabled")


@pytest.mark.e2e
def test_text_to_video_veo31_fast_sync(runway_api_key):
    """
    Test Runway text-to-video with Veo 3.1 Fast (sync).

    This tests:
    - Text-to-video generation
    - Veo 3.1 Fast model
    - Sync operation
    - Different aspect ratios
    - Different durations (6 seconds)
    """
    config = VideoGenerationConfig(
        model="veo3.1_fast",
        provider="runway",
        api_key=runway_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="A serene mountain landscape with clouds slowly moving",
        duration_seconds=6,
        aspect_ratio="9:16",  # Portrait mode
        generate_audio=False,
    )

    # Generate video using API (sync)
    response = api.generate_video(config, request)

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

    print("✓ Generated text-to-video (sync) successfully")
    print(f"  Request ID: {response.request_id}")
    print(f"  Video URL: {response.video}")
    print("  Model: veo3.1_fast")
    print("  Duration: 6s")
    print("  Aspect Ratio: 9:16 (portrait)")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_image_to_video_gen4_turbo(runway_api_key):
    """
    Test Runway image-to-video with Gen-4 Turbo.

    This tests:
    - Image-to-video generation
    - Gen-4 Turbo model
    - Image URL input
    - Duration range (2-10 seconds for image-to-video)
    - Custom aspect ratios
    """
    config = VideoGenerationConfig(
        model="gen4_turbo",
        provider="runway",
        api_key=runway_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="The dragon warrior comes to life",
        image_list=[
            {
                "image": "https://storage.googleapis.com/falserverless/model_tests/wan/dragon-warrior.jpg",
                "type": "reference",
            }
        ],
        duration_seconds=5,
        aspect_ratio="16:9",
        seed=12345,
    )

    # Generate video using API
    response = await api.generate_video_async(config, request)

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

    print("✓ Generated image-to-video successfully")
    print(f"  Request ID: {response.request_id}")
    print(f"  Video URL: {response.video}")
    print("  Model: gen4_turbo")
    print("  Duration: 5s")
    print("  Aspect Ratio: 16:9")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_image_to_video_with_veo31(runway_api_key):
    """
    Test Runway image-to-video with Veo 3.1 (supports both text and image input).

    This tests:
    - Image-to-video with Veo 3.1
    - VEO model flexibility (text OR image-to-video)
    - Different aspect ratios
    - Optional prompt text
    """
    config = VideoGenerationConfig(
        model="veo3.1",
        provider="runway",
        api_key=runway_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="Camera slowly zooms out revealing a beautiful landscape",
        image_list=[
            {
                "image": "https://storage.googleapis.com/falserverless/example_inputs/veo31_i2v_input.jpg",
                "type": "first_frame",  # Also accepts first_frame
            }
        ],
        duration_seconds=8,
        aspect_ratio="16:9",  # veo3.1 i2v only supports: 16:9, 9:16, 16:9-wide, 9:16-wide
        seed=999,
    )

    # Generate video using API
    response = await api.generate_video_async(config, request)

    # Validate response
    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"

    print("✓ Generated image-to-video with Veo 3.1 successfully")
    print(f"  Request ID: {response.request_id}")
    print(f"  Video URL: {response.video}")
    print("  Model: veo3.1")
    print("  Duration: 8s")
    print("  Aspect Ratio: 16:9 (landscape)")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_video_to_video_gen4_aleph(runway_api_key):
    """
    Test Runway video-to-video with Gen-4 Aleph.

    This tests:
    - Video-to-video generation
    - Gen-4 Aleph model
    - Video URL input
    - Optional image references
    - Prompt transformation
    """
    config = VideoGenerationConfig(
        model="gen4_aleph",
        provider="runway",
        api_key=runway_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="Add colorful artistic elements to the video",
        video="https://v3b.fal.media/files/b/rabbit/ku8_Wdpf-oTbGRq4lB5DU_output.mp4",
        aspect_ratio="16:9",
        seed=777,
    )

    # Generate video using API
    response = await api.generate_video_async(config, request)

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

    print("✓ Generated video-to-video successfully")
    print(f"  Request ID: {response.request_id}")
    print(f"  Video URL: {response.video}")
    print("  Model: gen4_aleph")
    print("  Aspect Ratio: 16:9")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_video_to_video_with_image_references(runway_api_key):
    """
    Test Runway video-to-video with image style references.

    This tests:
    - Video-to-video with style references
    - Image references parameter
    - Multiple inputs (video + images)
    """
    config = VideoGenerationConfig(
        model="gen4_aleph",
        provider="runway",
        api_key=runway_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="Apply the art style from the reference image to the video",
        video="https://v3b.fal.media/files/b/rabbit/ku8_Wdpf-oTbGRq4lB5DU_output.mp4",
        image_list=[
            {
                "image": "https://storage.googleapis.com/falserverless/example_inputs/veo31_i2v_input.jpg",
                "type": "reference",
            }
        ],
        aspect_ratio="4:3",
    )

    # Generate video using API
    response = await api.generate_video_async(config, request)

    # Validate response
    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"

    print("✓ Generated video-to-video with image references successfully")
    print(f"  Request ID: {response.request_id}")
    print(f"  Video URL: {response.video}")
    print("  Model: gen4_aleph")
    print("  With style reference image")
    print("  Aspect Ratio: 4:3")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_content_moderation_parameter(runway_api_key):
    """
    Test Runway with content moderation settings.

    This tests:
    - Content moderation parameter
    - Extra params usage
    - Custom Runway-specific parameters
    """
    config = VideoGenerationConfig(
        model="veo3.1_fast",
        provider="runway",
        api_key=runway_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="A famous celebrity walking down a red carpet at a movie premiere",
        duration_seconds=4,
        aspect_ratio="16:9",
        extra_params={
            "content_moderation": {
                "public_figure_threshold": "low"  # Less strict about public figures
            }
        },
    )

    # Generate video using API
    response = await api.generate_video_async(config, request)

    # Validate response
    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"

    print("✓ Generated video with content moderation settings")
    print(f"  Request ID: {response.request_id}")
    print(f"  Video URL: {response.video}")
    print("  Content moderation: public_figure_threshold=low")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_various_aspect_ratios(runway_api_key):
    """
    Test Runway with various aspect ratios.

    This tests:
    - Aspect ratio conversion
    - Different ratio formats
    - Text-to-video with custom ratios
    """
    config = VideoGenerationConfig(
        model="veo3.1",  # veo3.1 supports duration 4,6,8; veo3 only supports 8
        provider="runway",
        api_key=runway_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    # Test 9:16 portrait (text-to-video supports: 16:9, 9:16, 16:9-wide, 9:16-wide)
    request = VideoGenerationRequest(
        prompt="A cinematic wide-angle shot of a desert highway at sunset",
        duration_seconds=4,
        aspect_ratio="9:16",
    )

    # Generate video using API
    response = await api.generate_video_async(config, request)

    # Validate response
    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"

    print("✓ Generated video with portrait aspect ratio")
    print(f"  Request ID: {response.request_id}")
    print(f"  Video URL: {response.video}")
    print("  Aspect Ratio: 9:16 (portrait)")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_gen4_turbo_image_to_video(runway_api_key):
    """
    Test Runway Gen-4 Turbo model for image-to-video.

    This tests:
    - Gen-4 Turbo model
    - Image-to-video capability
    - Alternative model selection

    Note: gen3a_turbo only supports 768:1280, 1280:768 ratios which aren't in
    our standard ratio maps. Using gen4_turbo which supports more ratios.
    """
    config = VideoGenerationConfig(
        model="gen4_turbo",
        provider="runway",
        api_key=runway_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="The dragon warrior comes to life and looks around",
        image_list=[
            {
                "image": "https://storage.googleapis.com/falserverless/model_tests/wan/dragon-warrior.jpg",
                "type": "reference",
            }
        ],
        duration_seconds=10,  # Max duration for image-to-video
        aspect_ratio="9:16",  # Portrait mode
    )

    # Generate video using API
    response = await api.generate_video_async(config, request)

    # Validate response
    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"

    print("✓ Generated video with Gen-4 Turbo")
    print(f"  Request ID: {response.request_id}")
    print(f"  Video URL: {response.video}")
    print("  Model: gen4_turbo")
    print("  Duration: 10s (max for image-to-video)")
    print("  Aspect Ratio: 9:16 (portrait)")
