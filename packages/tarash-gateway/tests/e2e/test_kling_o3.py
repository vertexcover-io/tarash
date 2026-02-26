"""End-to-end tests for Kling O3 models via Fal.

These tests make actual API calls to the Fal.ai Kling O3 service.
Requires FAL_KEY environment variable to be set.

Run with: uv run pytest tests/e2e/test_kling_o3.py -v -m e2e
Skip with: uv run pytest tests/e2e/test_kling_o3.py -v -m "not e2e"
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
async def test_kling_o3_text_to_video_with_audio(fal_api_key):
    """
    Test Kling O3 text-to-video with audio generation.

    This tests:
    - Text-to-video generation (core T2V functionality)
    - Duration mapping (string format "3"-"15" without suffix)
    - Aspect ratio parameter
    - Audio generation (generate_audio=True)
    - Progress tracking via callback
    """
    progress_updates = []

    async def progress_callback(update: VideoGenerationUpdate):
        progress_updates.append(update)
        print(f"  Progress: {update.status}")

    config = VideoGenerationConfig(
        model="fal-ai/kling-video/o3/pro/text-to-video",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="A majestic eagle soaring over snow-capped mountains at golden hour, cinematic quality",
        duration_seconds=5,
        aspect_ratio="16:9",
        generate_audio=True,
    )

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

    print(f"\n  Generated video URL: {response.video}")
    print(f"  Request ID: {response.request_id}")
    print(f"  Progress updates: {len(progress_updates)}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_kling_o3_image_to_video_with_end_frame(fal_api_key):
    """
    Test Kling O3 image-to-video with start and end frame control.

    This tests:
    - Image-to-video generation
    - Field mapping: image_list with type="reference" -> image_url
    - Field mapping: image_list with type="last_frame" -> end_image_url
    - Duration in O3 range (3-15 seconds)
    - Progress tracking
    """
    progress_updates = []

    async def progress_callback(update: VideoGenerationUpdate):
        progress_updates.append(update)
        print(f"  Progress: {update.status}")

    config = VideoGenerationConfig(
        model="fal-ai/kling-video/o3/pro/image-to-video",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="Smooth transition from winter to spring, snow melts and flowers bloom",
        image_list=[
            {
                "image": "https://v3b.fal.media/files/b/rabbit/NaslJIC7F2WodS6DFZRRJ.png",
                "type": "reference",
            },
            {
                "image": "https://v3b.fal.media/files/b/tiger/BwHi22qoQnqaTNMMhe533.png",
                "type": "last_frame",
            },
        ],
        duration_seconds=5,
        aspect_ratio="16:9",
    )

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
    statuses = [update.status for update in progress_updates]
    assert "completed" in statuses

    print(f"\n  Generated video URL: {response.video}")
    print(f"  Request ID: {response.request_id}")
    print(f"  Progress updates: {len(progress_updates)}")


@pytest.mark.e2e
def test_kling_o3_reference_to_video_with_elements(fal_api_key):
    """
    Test Kling O3 reference-to-video with character elements (sync mode).

    This tests:
    - Reference-to-video generation
    - Elements support via extra_params (character/object definitions)
    - image_urls for style references (image_list_field_mapper)
    - Synchronous generation path
    - Progress callback in sync mode
    """
    progress_updates = []

    def progress_callback(update: VideoGenerationUpdate):
        progress_updates.append(update)
        print(f"  Progress: {update.status}")

    config = VideoGenerationConfig(
        model="fal-ai/kling-video/o3/standard/reference-to-video",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="Show @Element1 walking confidently through a futuristic cityscape with neon lights",
        image_list=[
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

    # Synchronous generation
    response = api.generate_video(config, request, on_progress=progress_callback)

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

    print(f"\n  Sync generated video URL: {response.video}")
    print(f"  Request ID: {response.request_id}")
    print(f"  Progress updates: {len(progress_updates)}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_kling_o3_multi_shot_text_to_video(fal_api_key):
    """
    Test Kling O3 multi-shot text-to-video (O3-unique feature).

    This tests:
    - Multi-shot storyboarding with multi_prompt as a list of shot objects
    - shot_type="customize" (the only permitted value)
    - Each shot object has prompt and duration fields
    - Longer duration for multi-shot (8 seconds total)
    - Progress tracking
    """
    progress_updates = []

    async def progress_callback(update: VideoGenerationUpdate):
        progress_updates.append(update)
        print(f"  Progress: {update.status}")

    config = VideoGenerationConfig(
        model="fal-ai/kling-video/o3/pro/text-to-video",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="",
        duration_seconds=8,
        aspect_ratio="16:9",
        extra_params={
            "multi_prompt": [
                {
                    "prompt": "A rocket launches from a launchpad with flames and smoke",
                    "duration": "4",
                },
                {
                    "prompt": "The rocket ascends through clouds into the blue sky, leaving a trail behind",
                    "duration": "4",
                },
            ],
            "shot_type": "customize",
        },
    )

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

    print(f"\n  Generated video URL: {response.video}")
    print(f"  Request ID: {response.request_id}")
    print(f"  Progress updates: {len(progress_updates)}")
    print("  Multi-shot: 2 shots via multi_prompt list")
