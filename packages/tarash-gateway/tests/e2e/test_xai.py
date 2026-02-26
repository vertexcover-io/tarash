"""End-to-end tests for xAI provider.

These tests make actual API calls to the xAI service.
Requires environment variables:
  - XAI_API_KEY: xAI API key

Run with: uv run pytest tests/e2e/test_xai.py -v -m e2e --e2e
"""

import os

import pytest

from tarash.tarash_gateway import api
from tarash.tarash_gateway.models import (
    ImageGenerationConfig,
    ImageGenerationRequest,
    ImageGenerationResponse,
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoGenerationUpdate,
)


@pytest.fixture(scope="module")
def xai_api_key():
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        pytest.skip("XAI_API_KEY environment variable not set")
    return api_key


@pytest.fixture(scope="module")
def video_config(xai_api_key):
    return VideoGenerationConfig(
        model="grok-imagine-video",
        provider="xai",
        api_key=xai_api_key,
        timeout=1800,
        max_poll_attempts=360,
        poll_interval=5,
    )


@pytest.fixture(scope="module")
def image_config(xai_api_key):
    return ImageGenerationConfig(
        model="grok-imagine-image",
        provider="xai",
        api_key=xai_api_key,
        timeout=120,
        max_poll_attempts=30,
        poll_interval=3,
    )


# ---- Video E2E ----


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_text_to_video_async_with_progress(video_config):
    """Async text-to-video — validates progress tracking and response structure."""
    progress_updates: list[VideoGenerationUpdate] = []

    async def on_progress(update: VideoGenerationUpdate) -> None:
        progress_updates.append(update)

    request = VideoGenerationRequest(
        prompt="A serene mountain lake at sunrise with mist rising from the water",
        aspect_ratio="16:9",
        duration_seconds=5,
        resolution="720p",
    )

    response = await api.generate_video_async(
        video_config, request, on_progress=on_progress
    )

    assert isinstance(response, VideoGenerationResponse)
    assert response.status == "completed"
    assert response.request_id is not None
    assert isinstance(response.video, str)
    assert response.video.startswith("http")
    assert response.content_type == "video/mp4"
    assert isinstance(response.raw_response, dict)
    assert len(progress_updates) > 0


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_image_to_video_async(video_config):
    """Async image-to-video with a public image URL."""
    test_image_url = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/"
        "PNG_transparency_demonstration_1.png/"
        "280px-PNG_transparency_demonstration_1.png"
    )

    request = VideoGenerationRequest(
        prompt="The object slowly rotates",
        aspect_ratio="1:1",
        duration_seconds=5,
        image_list=[{"type": "reference", "image": test_image_url}],
    )

    response = await api.generate_video_async(video_config, request)

    assert isinstance(response, VideoGenerationResponse)
    assert response.status == "completed"
    assert isinstance(response.video, str)
    assert response.video.startswith("http")


# ---- Image E2E ----


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_text_to_image_async(image_config):
    """Async text-to-image with grok-imagine-image."""
    request = ImageGenerationRequest(
        prompt="A photorealistic sunset over the ocean with vibrant colors",
        aspect_ratio="16:9",
        extra_params={"resolution": "1k"},
    )
    response = await api.generate_image_async(image_config, request)

    assert isinstance(response, ImageGenerationResponse)
    assert response.status == "completed"
    assert len(response.images) > 0
    assert isinstance(response.images[0], str)
    assert response.images[0].startswith("http")
    assert response.content_type == "image/png"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_image_editing_with_reference_async(image_config):
    """Async image editing with a reference image."""
    ref_image_url = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/"
        "PNG_transparency_demonstration_1.png/"
        "280px-PNG_transparency_demonstration_1.png"
    )
    request = ImageGenerationRequest(
        prompt="Add warm golden lighting to this image",
        image_list=[{"type": "reference", "image": ref_image_url}],
        aspect_ratio="1:1",
    )
    response = await api.generate_image_async(image_config, request)

    assert isinstance(response, ImageGenerationResponse)
    assert response.status == "completed"
    assert len(response.images) > 0
    assert response.images[0].startswith("http")
