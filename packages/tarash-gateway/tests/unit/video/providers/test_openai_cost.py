"""Tests for OpenAI provider cost resolution (REQ-027, REQ-030)."""

from unittest.mock import MagicMock

import pytest

from tarash.tarash_gateway.models import (
    ImageGenerationConfig,
    ImageGenerationRequest,
    VideoGenerationConfig,
    VideoGenerationRequest,
)
from tarash.tarash_gateway.pricing import PRICING_TABLE
from tarash.tarash_gateway.providers.openai import OpenAIProviderHandler


# ==================== Fixtures ====================


@pytest.fixture
def handler():
    """Create OpenAI handler instance."""
    return OpenAIProviderHandler()


@pytest.fixture
def video_config():
    """Config for Sora video."""
    return VideoGenerationConfig(
        model="sora",
        provider="openai",
        api_key="test-key",
    )


@pytest.fixture
def image_config():
    """Config for GPT Image generation."""
    return ImageGenerationConfig(
        model="gpt-image-1",
        provider="openai",
        api_key="test-key",
    )


@pytest.fixture
def base_request():
    """Create basic video request."""
    return VideoGenerationRequest(prompt="A sunset over mountains")


@pytest.fixture
def image_request():
    """Create basic image request."""
    return ImageGenerationRequest(prompt="A cat")


# ==================== Video _convert_response cost tests ====================


def test_video_convert_response_includes_cost(handler, video_config, base_request):
    """REQ-027: OpenAI video response includes cost."""
    video_mock = MagicMock()
    video_mock.status = "completed"
    video_mock.id = "video-123"
    video_mock.seconds = 8
    video_mock.size = "1280x720"
    video_mock.model_dump.return_value = {"id": "video-123", "status": "completed"}

    provider_response = {
        "video": video_mock,
        "content": b"fake-video-bytes",
        "content_type": "video/mp4",
    }

    response = handler._convert_response(
        video_config, base_request, "req-123", provider_response
    )

    assert response.cost is not None
    entry = PRICING_TABLE[("openai", "sora")]
    # Sora uses per-second pricing with output duration
    assert response.cost.raw_unit == entry.unit


def test_video_unknown_model_no_cost(handler, base_request):
    """EDGE-001: Unknown OpenAI video model returns cost=None."""
    config = VideoGenerationConfig(
        model="sora-unknown",
        provider="openai",
        api_key="test-key",
    )
    video_mock = MagicMock()
    video_mock.status = "completed"
    video_mock.id = "video-456"
    video_mock.seconds = 4
    video_mock.size = "1280x720"
    video_mock.model_dump.return_value = {"id": "video-456"}

    provider_response = {
        "video": video_mock,
        "content": b"fake-video-bytes",
        "content_type": "video/mp4",
    }

    response = handler._convert_response(
        config, base_request, "req-456", provider_response
    )

    assert response.cost is None


# ==================== Image _convert_image_response cost tests ====================


def test_image_convert_response_includes_cost(handler, image_config, image_request):
    """REQ-027: OpenAI image response includes cost with quantity=1."""
    provider_response = MagicMock()
    provider_response.data = [
        MagicMock(url="https://example.com/img.png", revised_prompt=None)
    ]
    provider_response.model_dump.return_value = {}

    response = handler._convert_image_response(
        image_config, image_request, "req-123", provider_response
    )

    assert response.cost is not None
    entry = PRICING_TABLE[("openai", "gpt-image-1")]
    assert response.cost.raw_unit == entry.unit
    assert response.cost.raw_amount == 1.0
    assert response.cost.amount_usd == pytest.approx(entry.usd_per_unit)


def test_image_dalle3_includes_cost(handler, image_request):
    """REQ-027: DALL-E 3 image response includes cost."""
    config = ImageGenerationConfig(
        model="dall-e-3",
        provider="openai",
        api_key="test-key",
    )
    provider_response = MagicMock()
    provider_response.data = [
        MagicMock(url="https://example.com/img.png", revised_prompt="revised")
    ]
    provider_response.model_dump.return_value = {}

    response = handler._convert_image_response(
        config, image_request, "req-456", provider_response
    )

    assert response.cost is not None
    assert response.cost.raw_unit == "images"


def test_image_unknown_model_no_cost(handler, image_request):
    """EDGE-001: Unknown OpenAI image model returns cost=None."""
    config = ImageGenerationConfig(
        model="unknown-image-model",
        provider="openai",
        api_key="test-key",
    )
    provider_response = MagicMock()
    img_mock = MagicMock()
    img_mock.url = "https://example.com/img.png"
    img_mock.revised_prompt = None
    provider_response.data = [img_mock]
    provider_response.model_dump.return_value = {}

    response = handler._convert_image_response(
        config, image_request, "req-789", provider_response
    )

    assert response.cost is None
