"""End-to-end tests for Kling V3 models via Fal.

These tests make actual API calls to the Fal.ai Kling V3 service.
Requires FAL_KEY environment variable to be set.

Run with: uv run pytest tests/e2e/test_kling_v3.py -v -m e2e
Skip with: uv run pytest tests/e2e/test_kling_v3.py -v -m "not e2e"
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
async def test_kling_v3_pro_text_to_video_multi_shot(fal_api_key):
    """
    Test Kling V3 Pro text-to-video with multi-shot and audio.

    Model: fal-ai/kling-video/v3/pro/text-to-video
    Tests: multi_prompt, generate_audio, extended duration (10s)
    """
    progress_updates = []

    async def progress_callback(update: VideoGenerationUpdate):
        progress_updates.append(update)
        print(f"  Progress: {update.status}")

    config = VideoGenerationConfig(
        model="fal-ai/kling-video/v3/pro/text-to-video",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    # When using multi_prompt, pass empty prompt (excluded by field mapper with warning)
    request = VideoGenerationRequest(
        prompt="",
        duration_seconds=10,
        aspect_ratio="16:9",
        generate_audio=True,
        negative_prompt="blur, distorted, low quality",
        extra_params={
            "multi_prompt": [
                {
                    "prompt": "Scene 1: A majestic eagle soars over snow-capped mountains at sunrise",
                    "duration": "5",
                },
                {
                    "prompt": "Scene 2: The eagle dives toward a crystal-clear lake below",
                    "duration": "5",
                },
            ],
        },
    )

    response = await api.generate_video_async(
        config, request, on_progress=progress_callback
    )

    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"
    assert isinstance(response.raw_response, dict)
    assert "video" in response.raw_response
    assert isinstance(response.video, str)
    assert response.video.startswith("http")
    assert len(progress_updates) > 0
    statuses = [update.status for update in progress_updates]
    assert "completed" in statuses

    print(f"\n  ✓ Generated video URL: {response.video}")
    print(f"  ✓ Request ID: {response.request_id}")
    print(f"  ✓ Progress updates: {len(progress_updates)}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_kling_v3_pro_image_to_video_with_elements(fal_api_key):
    """
    Test Kling V3 Pro image-to-video with elements.

    Model: fal-ai/kling-video/v3/pro/image-to-video
    Tests: start/end frame, elements, aspect_ratio, cfg_scale
    """
    progress_updates = []

    async def progress_callback(update: VideoGenerationUpdate):
        progress_updates.append(update)
        print(f"  Progress: {update.status}")

    config = VideoGenerationConfig(
        model="fal-ai/kling-video/v3/pro/image-to-video",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    # @Element1 references the first element in elements array
    # start_image_url/end_image_url are separate from @Image syntax
    request = VideoGenerationRequest(
        prompt="@Element1 walks gracefully through the scene with smooth motion",
        duration_seconds=8,
        aspect_ratio="16:9",
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
        extra_params={
            "cfg_scale": 0.6,
            "elements": [
                {
                    "frontal_image_url": "https://v3b.fal.media/files/b/panda/MQp-ghIqshvMZROKh9lW3.png",
                    "reference_image_urls": [
                        "https://v3b.fal.media/files/b/kangaroo/YMpmQkYt9xugpOTQyZW0O.png"
                    ],
                }
            ],
        },
    )

    response = await api.generate_video_async(
        config, request, on_progress=progress_callback
    )

    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"
    assert isinstance(response.video, str)
    assert response.video.startswith("http")
    assert len(progress_updates) > 0

    print(f"\n  ✓ Generated video URL: {response.video}")
    print(f"  ✓ Request ID: {response.request_id}")
    print(f"  ✓ Progress updates: {len(progress_updates)}")


@pytest.mark.e2e
def test_kling_v3_standard_sync_generation(fal_api_key):
    """
    Test Kling V3 Standard text-to-video with synchronous API.

    Model: fal-ai/kling-video/v3/standard/text-to-video
    Tests: Standard tier, sync API path, shot_type
    """
    progress_updates = []

    def progress_callback(update: VideoGenerationUpdate):
        progress_updates.append(update)
        print(f"  Progress: {update.status}")

    config = VideoGenerationConfig(
        model="fal-ai/kling-video/v3/standard/text-to-video",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="A peaceful zen garden with cherry blossoms gently falling, soft breeze moving the leaves",
        duration_seconds=5,
        aspect_ratio="16:9",
        extra_params={
            "shot_type": "customize",
        },
    )

    response = api.generate_video(config, request, on_progress=progress_callback)

    assert isinstance(response, VideoGenerationResponse)
    assert response.request_id is not None
    assert response.video is not None
    assert response.status == "completed"
    assert isinstance(response.video, str)
    assert response.video.startswith("http")
    assert len(progress_updates) > 0

    print(f"\n  ✓ Sync generated video URL: {response.video}")
    print(f"  ✓ Request ID: {response.request_id}")
    print(f"  ✓ Progress updates: {len(progress_updates)}")
