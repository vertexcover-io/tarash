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


def _make_image_response(usage=None):
    """Helper to create a mock OpenAI image response."""
    provider_response = MagicMock()
    provider_response.data = [
        MagicMock(url="https://example.com/img.png", revised_prompt=None)
    ]
    provider_response.model_dump.return_value = {}
    if usage is not None:
        provider_response.usage = usage
    else:
        provider_response.usage = None
    return provider_response


def _make_usage(
    input_tokens=100,
    output_tokens=4000,
    total_tokens=4100,
    text_input_tokens=50,
    image_input_tokens=50,
    cached_tokens=0,
    image_output_tokens=None,
    text_output_tokens=None,
):
    """Helper to create a mock OpenAI usage object with token details."""
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.total_tokens = total_tokens

    input_details = MagicMock()
    input_details.text_tokens = text_input_tokens
    input_details.image_tokens = image_input_tokens
    input_details.cached_tokens = cached_tokens
    usage.input_tokens_details = input_details

    if image_output_tokens is not None or text_output_tokens is not None:
        output_details = MagicMock()
        output_details.image_tokens = image_output_tokens or 0
        output_details.text_tokens = text_output_tokens or 0
        usage.output_tokens_details = output_details
    else:
        usage.output_tokens_details = None

    return usage


# ---- Token-based cost for gpt-image-1 ----


def test_image_gpt_image_1_token_based_cost(handler, image_config, image_request):
    """gpt-image-1 with usage data returns token-based cost, not flat rate."""
    usage = _make_usage(
        input_tokens=100,
        output_tokens=4000,
        total_tokens=4100,
        text_input_tokens=80,
        image_input_tokens=20,
        cached_tokens=0,
    )
    provider_response = _make_image_response(usage)

    response = handler._convert_image_response(
        image_config, image_request, "req-123", provider_response
    )

    assert response.cost is not None
    assert response.cost.raw_unit == "tokens"
    assert response.cost.raw_amount == 4100.0

    # Expected: text_input(80 * $5/1M) + image_input(20 * $10/1M) + output(4000 * $40/1M)
    expected = (
        (80 * 5.0 / 1_000_000) + (20 * 10.0 / 1_000_000) + (4000 * 40.0 / 1_000_000)
    )
    assert response.cost.amount_usd == pytest.approx(expected)


def test_image_gpt_image_1_with_cached_tokens(handler, image_config, image_request):
    """gpt-image-1 with cached input tokens uses lower rate."""
    usage = _make_usage(
        input_tokens=100,
        output_tokens=4000,
        total_tokens=4100,
        text_input_tokens=80,
        image_input_tokens=20,
        cached_tokens=30,
    )
    provider_response = _make_image_response(usage)

    response = handler._convert_image_response(
        image_config, image_request, "req-cached", provider_response
    )

    assert response.cost is not None
    assert response.cost.raw_unit == "tokens"
    # Cached tokens reduce cost — 30 cached from text (80 total text)
    # uncached_text=50, cached_text=30, uncached_image=20, cached_image=0
    expected = (
        50 * 5.0 / 1_000_000  # uncached text
        + 30 * 1.25 / 1_000_000  # cached text
        + 20 * 10.0 / 1_000_000  # uncached image
        + 4000 * 40.0 / 1_000_000  # output
    )
    assert response.cost.amount_usd == pytest.approx(expected)


def test_image_gpt_image_15_with_text_output(handler, image_request):
    """gpt-image-1.5 charges different rates for text and image output."""
    config = ImageGenerationConfig(
        model="gpt-image-1.5",
        provider="openai",
        api_key="test-key",
    )
    usage = _make_usage(
        input_tokens=200,
        output_tokens=5000,
        total_tokens=5200,
        text_input_tokens=150,
        image_input_tokens=50,
        cached_tokens=0,
        image_output_tokens=4800,
        text_output_tokens=200,
    )
    provider_response = _make_image_response(usage)

    response = handler._convert_image_response(
        config, image_request, "req-15", provider_response
    )

    assert response.cost is not None
    assert response.cost.raw_unit == "tokens"
    expected = (
        150 * 5.0 / 1_000_000  # text input
        + 50 * 8.0 / 1_000_000  # image input
        + 4800 * 32.0 / 1_000_000  # image output
        + 200 * 10.0 / 1_000_000  # text output
    )
    assert response.cost.amount_usd == pytest.approx(expected)


def test_image_gpt_image_1_mini_cost(handler, image_request):
    """gpt-image-1-mini uses its own lower rates."""
    config = ImageGenerationConfig(
        model="gpt-image-1-mini",
        provider="openai",
        api_key="test-key",
    )
    usage = _make_usage(
        input_tokens=100,
        output_tokens=3000,
        total_tokens=3100,
        text_input_tokens=60,
        image_input_tokens=40,
        cached_tokens=0,
    )
    provider_response = _make_image_response(usage)

    response = handler._convert_image_response(
        config, image_request, "req-mini", provider_response
    )

    assert response.cost is not None
    expected = (
        60 * 2.0 / 1_000_000  # text input
        + 40 * 2.5 / 1_000_000  # image input
        + 3000 * 8.0 / 1_000_000  # output
    )
    assert response.cost.amount_usd == pytest.approx(expected)


def test_image_gpt_image_1_no_usage_falls_back_to_pricing_table(
    handler, image_config, image_request
):
    """gpt-image-1 without usage data falls back to pricing table flat rate."""
    provider_response = _make_image_response(usage=None)

    response = handler._convert_image_response(
        image_config, image_request, "req-no-usage", provider_response
    )

    assert response.cost is not None
    assert response.cost.raw_unit == "images"
    assert response.cost.raw_amount == 1.0
    assert response.cost.amount_usd == pytest.approx(0.042)


# ---- DALL-E (flat rate, no token usage) ----


def test_image_dalle3_includes_cost(handler, image_request):
    """DALL-E 3 uses flat pricing table rate (no token usage from API)."""
    config = ImageGenerationConfig(
        model="dall-e-3",
        provider="openai",
        api_key="test-key",
    )
    provider_response = _make_image_response(usage=None)
    provider_response.data[0].revised_prompt = "revised"

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
    provider_response = _make_image_response(usage=None)

    response = handler._convert_image_response(
        config, image_request, "req-789", provider_response
    )

    assert response.cost is None
