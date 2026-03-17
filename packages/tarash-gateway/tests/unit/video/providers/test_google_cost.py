"""Tests for Google provider cost resolution (REQ-028)."""

from unittest.mock import MagicMock

import pytest

from tarash.tarash_gateway.models import (
    ImageGenerationConfig,
    ImageGenerationRequest,
    VideoGenerationConfig,
)
from tarash.tarash_gateway.pricing import PRICING_TABLE
from tarash.tarash_gateway.providers.google import GoogleProviderHandler


# ==================== Fixtures ====================


@pytest.fixture
def handler():
    """Create Google handler instance."""
    return GoogleProviderHandler()


@pytest.fixture
def video_config():
    """Config for Veo 3 video."""
    return VideoGenerationConfig(
        model="veo-3.0-generate-preview",
        provider="google",
        api_key="test-key",
    )


@pytest.fixture
def image_config():
    """Config for Imagen image generation."""
    return ImageGenerationConfig(
        model="imagen-3.0-generate-001",
        provider="google",
        api_key="test-key",
    )


@pytest.fixture
def image_request():
    """Create basic image request."""
    return ImageGenerationRequest(prompt="A cat")


# ==================== Video _convert_response cost tests ====================


def test_video_convert_response_includes_cost(handler, video_config):
    """REQ-028: Google Veo video response includes cost using output duration."""
    operation = MagicMock()
    operation.done = True
    operation.error = None
    operation.model_dump.return_value = {}

    video_obj = MagicMock()
    video_obj.uri = "https://example.com/video.mp4"
    video_obj.mime_type = "video/mp4"

    generated_video = MagicMock()
    generated_video.video = video_obj

    operation.response.generated_videos = [generated_video]

    response = handler._convert_response(video_config, "req-123", operation)

    assert response.cost is not None
    entry = PRICING_TABLE[("google", "veo-3.0-generate-preview")]
    assert response.cost.raw_unit == entry.unit


def test_video_unknown_model_no_cost(handler):
    """EDGE-001: Unknown Google video model returns cost=None."""
    config = VideoGenerationConfig(
        model="veo-99-unknown",
        provider="google",
        api_key="test-key",
    )
    operation = MagicMock()
    operation.done = True
    operation.error = None
    operation.model_dump.return_value = {}

    video_obj = MagicMock()
    video_obj.uri = "https://example.com/video.mp4"
    video_obj.mime_type = "video/mp4"

    generated_video = MagicMock()
    generated_video.video = video_obj
    operation.response.generated_videos = [generated_video]

    response = handler._convert_response(config, "req-456", operation)

    assert response.cost is None


# ==================== Image _convert_image_response cost tests ====================


def test_image_convert_response_includes_cost(handler, image_config, image_request):
    """REQ-028: Google Imagen image response includes cost with quantity=1."""
    gen_img = MagicMock()
    gen_img.image.gcs_uri = "gs://bucket/img.png"

    genai_response = {"generated_images": [gen_img]}

    response = handler._convert_image_response(image_config, "req-123", genai_response)

    assert response.cost is not None
    entry = PRICING_TABLE[("google", "imagen-3.0-generate-001")]
    assert response.cost.raw_unit == entry.unit
    assert response.cost.raw_amount == 1.0
    assert response.cost.amount_usd == pytest.approx(entry.usd_per_unit)


def test_gemini_image_includes_cost(handler):
    """REQ-028: Gemini image response includes cost."""
    config = ImageGenerationConfig(
        model="gemini-2.5-flash-image",
        provider="google",
        api_key="test-key",
    )

    gen_img = MagicMock()
    gen_img.image.gcs_uri = "gs://bucket/img.png"

    genai_response = {"generated_images": [gen_img]}

    response = handler._convert_image_response(config, "req-456", genai_response)

    assert response.cost is not None
    assert response.cost.raw_unit == "images"


def test_image_unknown_model_no_cost(handler, image_request):
    """EDGE-001: Unknown Google image model returns cost=None."""
    config = ImageGenerationConfig(
        model="unknown-image-model",
        provider="google",
        api_key="test-key",
    )
    gen_img = MagicMock()
    gen_img.image.gcs_uri = "gs://bucket/img.png"

    genai_response = {"generated_images": [gen_img]}

    response = handler._convert_image_response(config, "req-789", genai_response)

    assert response.cost is None
