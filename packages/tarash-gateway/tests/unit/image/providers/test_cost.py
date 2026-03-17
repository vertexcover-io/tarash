"""Tests for image provider cost resolution (REQ-025, REQ-026, REQ-029, REQ-030)."""

from unittest.mock import MagicMock

import pytest

from tarash.tarash_gateway.models import (
    ImageGenerationConfig,
    ImageGenerationRequest,
)
from tarash.tarash_gateway.pricing import PRICING_TABLE


# ==================== Stability (REQ-025) ====================


@pytest.fixture
def stability_handler():
    """Create Stability handler instance."""
    from tarash.tarash_gateway.providers.stability import StabilityProviderHandler

    return StabilityProviderHandler()


@pytest.fixture
def stability_config():
    """Config for Stability sd3.5-large."""
    return ImageGenerationConfig(
        model="sd3.5-large",
        provider="stability",
        api_key="test-key",
    )


@pytest.fixture
def base_image_request():
    """Create basic image request."""
    return ImageGenerationRequest(prompt="A cat")


def test_stability_image_cost_per_image(
    stability_handler, stability_config, base_image_request
):
    """REQ-025: Stability image uses quantity=1 per image."""
    response = stability_handler._convert_image_response(
        stability_config,
        base_image_request,
        "req-123",
        b"fake-image-bytes",
        "image/png",
    )

    assert response.cost is not None
    entry = PRICING_TABLE[("stability", "sd3.5-large")]
    assert response.cost.raw_unit == entry.unit
    assert response.cost.raw_amount == 1.0
    assert response.cost.amount_usd == pytest.approx(entry.usd_per_unit)


def test_stability_medium_model_cost(stability_handler, base_image_request):
    """REQ-025: Stability sd3.5-medium resolves correct cost."""
    config = ImageGenerationConfig(
        model="sd3.5-medium",
        provider="stability",
        api_key="test-key",
    )
    response = stability_handler._convert_image_response(
        config,
        base_image_request,
        "req-456",
        b"fake-image-bytes",
        "image/png",
    )

    assert response.cost is not None
    entry = PRICING_TABLE[("stability", "sd3.5-medium")]
    assert response.cost.amount_usd == pytest.approx(entry.usd_per_unit)


def test_stability_unknown_model_no_cost(stability_handler, base_image_request):
    """EDGE-001: Unknown Stability model returns cost=None."""
    config = ImageGenerationConfig(
        model="unknown-model",
        provider="stability",
        api_key="test-key",
    )
    response = stability_handler._convert_image_response(
        config,
        base_image_request,
        "req-789",
        b"fake-image-bytes",
        "image/png",
    )

    assert response.cost is None


# ==================== xAI (REQ-026) ====================


@pytest.fixture
def xai_handler():
    """Create xAI handler instance."""
    from tarash.tarash_gateway.providers.xai import XaiProviderHandler

    return XaiProviderHandler()


@pytest.fixture
def xai_image_config():
    """Config for xAI image generation."""
    return ImageGenerationConfig(
        model="grok-imagine-image",
        provider="xai",
        api_key="test-key",
    )


def test_xai_image_cost_per_image(xai_handler, xai_image_config):
    """REQ-026: xAI image uses quantity=1."""
    xai_response = MagicMock()
    xai_response.respect_moderation = True
    xai_response.url = "https://example.com/img.png"
    xai_response.model = "grok-imagine-image"

    response = xai_handler._convert_image_response(
        xai_image_config, "req-123", xai_response
    )

    assert response.cost is not None
    entry = PRICING_TABLE[("xai", "grok-imagine-image")]
    assert response.cost.raw_unit == entry.unit
    assert response.cost.raw_amount == 1.0
    assert response.cost.amount_usd == pytest.approx(entry.usd_per_unit)


def test_xai_video_cost_uses_duration(xai_handler):
    """REQ-026: xAI video uses output duration as quantity."""
    from tarash.tarash_gateway.models import (
        VideoGenerationConfig,
        VideoGenerationRequest,
    )

    config = VideoGenerationConfig(
        model="grok-imagine-video",
        provider="xai",
        api_key="test-key",
    )
    request = VideoGenerationRequest(prompt="A sunset")

    xai_response = MagicMock()
    xai_response.respect_moderation = True
    xai_response.url = "https://example.com/video.mp4"
    xai_response.duration = 10.0
    xai_response.model = "grok-imagine-video"

    response = xai_handler._convert_video_response(
        config, request, "req-456", xai_response
    )

    assert response.cost is not None
    entry = PRICING_TABLE[("xai", "grok-imagine-video")]
    assert response.cost.raw_unit == entry.unit
    assert response.cost.raw_amount == 10.0
    assert response.cost.amount_usd == pytest.approx(entry.usd_per_unit * 10.0)


def test_xai_video_no_duration_uses_fallback(xai_handler):
    """REQ-026: xAI video without duration uses quantity=1.0."""
    from tarash.tarash_gateway.models import (
        VideoGenerationConfig,
        VideoGenerationRequest,
    )

    config = VideoGenerationConfig(
        model="grok-imagine-video",
        provider="xai",
        api_key="test-key",
    )
    request = VideoGenerationRequest(prompt="A sunset")

    xai_response = MagicMock()
    xai_response.respect_moderation = True
    xai_response.url = "https://example.com/video.mp4"
    xai_response.duration = None
    xai_response.model = "grok-imagine-video"

    response = xai_handler._convert_video_response(
        config, request, "req-789", xai_response
    )

    assert response.cost is not None
    # Falls back to quantity=1.0
    assert response.cost.raw_amount == 1.0


# ==================== Replicate (REQ-029, EDGE-001) ====================


@pytest.fixture
def replicate_handler():
    """Create Replicate handler instance."""
    from tarash.tarash_gateway.providers.replicate import ReplicateProviderHandler

    return ReplicateProviderHandler()


def test_replicate_video_cost_is_none(replicate_handler):
    """REQ-029: Replicate video responses have cost=None (no pricing entries)."""
    from tarash.tarash_gateway.models import (
        VideoGenerationConfig,
        VideoGenerationRequest,
    )

    config = VideoGenerationConfig(
        model="some-replicate/model",
        provider="replicate",
        api_key="test-key",
    )
    request = VideoGenerationRequest(prompt="A cat")

    response = replicate_handler._convert_response(
        config, request, "pred-123", "https://example.com/video.mp4"
    )

    assert response.cost is None


def test_replicate_image_cost_is_none(replicate_handler):
    """REQ-029, EDGE-001: Replicate image responses have cost=None."""
    config = ImageGenerationConfig(
        model="some-replicate/image-model",
        provider="replicate",
        api_key="test-key",
    )
    request = ImageGenerationRequest(prompt="A cat")

    response = replicate_handler._convert_image_response(
        config, request, "pred-456", "https://example.com/img.png"
    )

    assert response.cost is None


# ==================== Azure OpenAI (REQ-030) ====================


def test_azure_openai_image_cost_matches_openai():
    """REQ-030: Azure OpenAI cost matches OpenAI behavior — no usage means cost=None."""
    from tarash.tarash_gateway.providers.azure_openai import AzureOpenAIProviderHandler

    handler = AzureOpenAIProviderHandler()

    config = ImageGenerationConfig(
        model="gpt-image-1",
        provider="openai",
        api_key="test-key",
    )

    provider_response = MagicMock()
    provider_response.data = [
        MagicMock(url="https://example.com/img.png", revised_prompt=None)
    ]
    provider_response.model_dump.return_value = {}
    provider_response.usage = None  # No token usage → cost is None

    # Azure OpenAI inherits from OpenAI, so same _convert_image_response
    response = handler._convert_image_response(
        config, ImageGenerationRequest(prompt="A cat"), "req-123", provider_response
    )

    # Without token usage data, cost should be None (no fallback)
    assert response.cost is None
