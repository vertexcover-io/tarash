"""E2E test for Alibaba HappyHorse via Fal provider.

This test is gated by FAL_KEY environment variable and will be skipped
if it's not set to keep CI stable.
"""

import os

import pytest

from tarash.tarash_gateway import api
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest, VideoGenerationResponse


@pytest.fixture(scope="module")
def fal_api_key():
    key = os.getenv("FAL_KEY")
    if not key:
        pytest.skip("FAL_KEY environment variable not set")
    return key


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_happy_horse_image_to_video_e2e_gated(fal_api_key):
    """Minimal image-to-video generation on HappyHorse, skipped if no key."""
    config = VideoGenerationConfig(
        model="alibaba/happy-horse/image-to-video",
        provider="fal",
        api_key=fal_api_key,
        timeout=600,
        max_poll_attempts=120,
        poll_interval=5,
    )

    request = VideoGenerationRequest(
        prompt="A majestic horse galloping across a field at sunset",
        duration_seconds=5,
        image_list=[
            {
                "image": "https://storage.googleapis.com/falserverless/example_inputs/veo31_i2v_input.jpg",
                "type": "reference",
            }
        ],
    )

    response = await api.generate_video_async(config, request)

    assert isinstance(response, VideoGenerationResponse)
    assert response.status == "completed"
    assert isinstance(response.video, str) and response.video.startswith("http")
