"""Tests for XaiProviderHandler image generation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tarash.tarash_gateway.exceptions import (
    ContentModerationError,
    GenerationFailedError,
    ValidationError,
)
from tarash.tarash_gateway.models import ImageGenerationConfig, ImageGenerationRequest
from tarash.tarash_gateway.providers.xai import XaiProviderHandler


@pytest.fixture
def handler():
    with (
        patch("tarash.tarash_gateway.providers.xai.has_xai_sdk", True),
        patch("tarash.tarash_gateway.providers.xai.Client", MagicMock()),
        patch("tarash.tarash_gateway.providers.xai.AsyncClient", MagicMock()),
    ):
        yield XaiProviderHandler()


@pytest.fixture
def image_config():
    return ImageGenerationConfig(
        model="grok-imagine-image",
        provider="xai",
        api_key="test-xai-key",
        timeout=120,
        poll_interval=0,
        max_poll_attempts=3,
    )


@pytest.fixture
def basic_image_request():
    return ImageGenerationRequest(
        prompt="A vibrant sunset over the ocean",
        aspect_ratio="16:9",
    )


def _make_xai_image_response(url=None, image=None, respect_moderation=True):
    resp = MagicMock()
    resp.url = url
    resp.image = image
    resp.respect_moderation = respect_moderation
    resp.model = "grok-imagine-image"
    return resp


# ---- request conversion tests ----


def test_convert_image_request_basic(handler, image_config, basic_image_request):
    params = handler._convert_image_request(image_config, basic_image_request)
    assert params["prompt"] == "A vibrant sunset over the ocean"
    assert params["model"] == "grok-imagine-image"
    assert params["aspect_ratio"] == "16:9"


def test_convert_image_request_no_optional_fields(handler, image_config):
    request = ImageGenerationRequest(prompt="Simple image")
    params = handler._convert_image_request(image_config, request)
    assert "aspect_ratio" not in params
    assert "resolution" not in params
    assert "image_url" not in params
    assert "image_urls" not in params


def test_convert_image_request_with_valid_resolution(handler, image_config):
    request = ImageGenerationRequest(prompt="test", extra_params={"resolution": "2k"})
    params = handler._convert_image_request(image_config, request)
    assert params["resolution"] == "2k"


def test_convert_image_request_invalid_resolution_raises(handler, image_config):
    request = ImageGenerationRequest(prompt="test", extra_params={"resolution": "720p"})
    with pytest.raises(ValidationError, match="resolution"):
        handler._convert_image_request(image_config, request)


def test_convert_image_request_single_image_becomes_image_url(handler, image_config):
    request = ImageGenerationRequest(
        prompt="Edit this",
        image_list=[{"type": "reference", "image": "https://example.com/img.jpg"}],
    )
    params = handler._convert_image_request(image_config, request)
    assert params["image_url"] == "https://example.com/img.jpg"
    assert "image_urls" not in params


def test_convert_image_request_multiple_images_becomes_image_urls(
    handler, image_config
):
    request = ImageGenerationRequest(
        prompt="Combine these",
        image_list=[
            {"type": "reference", "image": "https://example.com/img1.jpg"},
            {"type": "reference", "image": "https://example.com/img2.jpg"},
        ],
    )
    params = handler._convert_image_request(image_config, request)
    assert "image_url" not in params
    assert params["image_urls"] == [
        "https://example.com/img1.jpg",
        "https://example.com/img2.jpg",
    ]


# ---- response conversion tests ----


def test_convert_image_response_success_url(handler, image_config):
    xai_resp = _make_xai_image_response(url="https://images.x.ai/img.png")
    response = handler._convert_image_response(image_config, "req-img-1", xai_resp)
    assert response.request_id == "req-img-1"
    assert response.images == ["https://images.x.ai/img.png"]
    assert response.status == "completed"
    assert response.content_type == "image/png"


def test_convert_image_response_moderation_failed_raises(handler, image_config):
    xai_resp = _make_xai_image_response(
        url="https://images.x.ai/img.png", respect_moderation=False
    )
    with pytest.raises(ContentModerationError):
        handler._convert_image_response(image_config, "req-img-1", xai_resp)


def test_convert_image_response_missing_url_raises(handler, image_config):
    xai_resp = _make_xai_image_response(url=None, image=None)
    with pytest.raises(GenerationFailedError, match="No image"):
        handler._convert_image_response(image_config, "req-img-1", xai_resp)


# ---- integration tests ----


@pytest.mark.asyncio
async def test_generate_image_async_success(handler, image_config, basic_image_request):
    mock_client = AsyncMock()
    xai_resp = _make_xai_image_response(url="https://images.x.ai/img.png")
    mock_client.image.sample = AsyncMock(return_value=xai_resp)

    with patch(
        "tarash.tarash_gateway.providers.xai.AsyncClient", return_value=mock_client
    ):
        response = await handler.generate_image_async(image_config, basic_image_request)

    assert response.status == "completed"
    assert response.images == ["https://images.x.ai/img.png"]
    assert response.content_type == "image/png"


def test_generate_image_sync_success(handler, image_config, basic_image_request):
    mock_client = MagicMock()
    xai_resp = _make_xai_image_response(url="https://images.x.ai/img.png")
    mock_client.image.sample.return_value = xai_resp

    with patch("tarash.tarash_gateway.providers.xai.Client", return_value=mock_client):
        response = handler.generate_image(image_config, basic_image_request)

    assert response.status == "completed"
    assert response.images == ["https://images.x.ai/img.png"]


@pytest.mark.asyncio
async def test_generate_image_async_moderation_raises(
    handler, image_config, basic_image_request
):
    mock_client = AsyncMock()
    xai_resp = _make_xai_image_response(
        url="https://images.x.ai/img.png", respect_moderation=False
    )
    mock_client.image.sample = AsyncMock(return_value=xai_resp)

    with patch(
        "tarash.tarash_gateway.providers.xai.AsyncClient", return_value=mock_client
    ):
        with pytest.raises(ContentModerationError):
            await handler.generate_image_async(image_config, basic_image_request)
