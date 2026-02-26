"""Tests for XaiProviderHandler."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tarash.tarash_gateway.exceptions import (
    ContentModerationError,
    GenerationFailedError,
    HTTPConnectionError,
    HTTPError,
    TimeoutError,
    ValidationError,
)
from tarash.tarash_gateway.models import (
    VideoGenerationConfig,
    VideoGenerationRequest,
)
from tarash.tarash_gateway.providers.xai import (
    XaiProviderHandler,
    parse_xai_video_status,
)


@pytest.fixture
def handler():
    """Create XaiProviderHandler with xai_sdk mocked out."""
    with (
        patch("tarash.tarash_gateway.providers.xai.has_xai_sdk", True),
        patch("tarash.tarash_gateway.providers.xai.Client", MagicMock()),
        patch("tarash.tarash_gateway.providers.xai.AsyncClient", MagicMock()),
    ):
        h = XaiProviderHandler()
        yield h


def test_handler_raises_import_error_when_sdk_missing():
    """XaiProviderHandler raises ImportError if xai-sdk is not installed."""
    with patch("tarash.tarash_gateway.providers.xai.has_xai_sdk", False):
        with pytest.raises(ImportError, match="xai-sdk"):
            XaiProviderHandler()


# ---- base_config fixture ----
@pytest.fixture
def base_config():
    return VideoGenerationConfig(
        model="grok-imagine-video",
        provider="xai",
        api_key="test-xai-key",
        timeout=600,
        poll_interval=0,
        max_poll_attempts=3,
    )


# ---- basic_request fixture ----
@pytest.fixture
def basic_request():
    return VideoGenerationRequest(
        prompt="A cat playing piano",
        aspect_ratio="16:9",
    )


# ---- helper ----
def _make_xai_video_response(
    url=None, respect_moderation=True, model="grok-imagine-video"
):
    resp = MagicMock()
    resp.url = url
    resp.respect_moderation = respect_moderation
    resp.model = model
    resp.duration = 5
    return resp


def _make_xai_poll_response(status_name: str, url: str | None = None):
    resp = MagicMock()
    status = MagicMock()
    status.name = status_name  # "DONE", "PENDING", "EXPIRED"
    resp.status = status
    resp.url = url
    resp.respect_moderation = True
    resp.duration = 5
    resp.model = "grok-imagine-video"
    resp.request_id = "xai-req-abc"
    return resp


# ---- status parsing tests ----
def test_parse_xai_video_status_pending_maps_to_processing():
    update = parse_xai_video_status("req-1", "pending")
    assert update.status == "processing"
    assert update.request_id == "req-1"
    assert update.progress_percent is None


def test_parse_xai_video_status_done_maps_to_completed():
    update = parse_xai_video_status("req-1", "done")
    assert update.status == "completed"


def test_parse_xai_video_status_expired_maps_to_failed():
    update = parse_xai_video_status("req-1", "expired")
    assert update.status == "failed"


def test_parse_xai_video_status_unknown_defaults_to_processing():
    update = parse_xai_video_status("req-1", "unknown_state")
    assert update.status == "processing"


# ---- video request conversion tests ----
def test_convert_video_request_basic(handler, base_config, basic_request):
    params = handler._convert_video_request(base_config, basic_request)
    assert params["prompt"] == "A cat playing piano"
    assert params["model"] == "grok-imagine-video"
    assert params["aspect_ratio"] == "16:9"


def test_convert_video_request_no_optional_fields(handler, base_config):
    request = VideoGenerationRequest(prompt="Simple test")
    params = handler._convert_video_request(base_config, request)
    assert "duration" not in params
    assert "resolution" not in params
    assert "aspect_ratio" not in params
    assert "image_url" not in params
    assert "video_url" not in params


def test_convert_video_request_with_valid_duration(handler, base_config):
    request = VideoGenerationRequest(prompt="test", duration_seconds=10)
    params = handler._convert_video_request(base_config, request)
    assert params["duration"] == 10


def test_convert_video_request_invalid_duration_raises(handler, base_config):
    request = VideoGenerationRequest(prompt="test", duration_seconds=20)
    with pytest.raises(ValidationError, match="duration"):
        handler._convert_video_request(base_config, request)


def test_convert_video_request_with_valid_resolution(handler, base_config):
    request = VideoGenerationRequest(prompt="test", resolution="720p")
    params = handler._convert_video_request(base_config, request)
    assert params["resolution"] == "720p"


def test_convert_video_request_invalid_resolution_raises(handler, base_config):
    request = VideoGenerationRequest(prompt="test", resolution="1080p")
    with pytest.raises(ValidationError, match="resolution"):
        handler._convert_video_request(base_config, request)


def test_convert_video_request_with_image_url(handler, base_config):
    request = VideoGenerationRequest(
        prompt="Animate this",
        image_list=[{"type": "reference", "image": "https://example.com/img.jpg"}],
    )
    params = handler._convert_video_request(base_config, request)
    assert params["image_url"] == "https://example.com/img.jpg"


def test_convert_video_request_with_video_url(handler, base_config):
    request = VideoGenerationRequest(
        prompt="Edit this",
        video="https://example.com/source.mp4",
    )
    params = handler._convert_video_request(base_config, request)
    assert params["video_url"] == "https://example.com/source.mp4"


# ---- video response conversion tests ----
def test_convert_video_response_success(handler, base_config, basic_request):
    xai_resp = _make_xai_video_response(url="https://vidgen.x.ai/video.mp4")
    response = handler._convert_video_response(
        base_config, basic_request, "req-abc", xai_resp
    )
    assert response.request_id == "req-abc"
    assert response.video == "https://vidgen.x.ai/video.mp4"
    assert response.status == "completed"
    assert response.content_type == "video/mp4"


def test_convert_video_response_missing_url_raises(handler, base_config, basic_request):
    xai_resp = _make_xai_video_response(url=None)
    with pytest.raises(GenerationFailedError, match="No video URL"):
        handler._convert_video_response(base_config, basic_request, "req-abc", xai_resp)


def test_convert_video_response_moderation_failed_raises(
    handler, base_config, basic_request
):
    xai_resp = _make_xai_video_response(
        url="https://vidgen.x.ai/video.mp4", respect_moderation=False
    )
    with pytest.raises(ContentModerationError):
        handler._convert_video_response(base_config, basic_request, "req-abc", xai_resp)


# ---- grpc mock helpers ----
def _make_grpc_error(status_code, details_msg: str):
    """Create a fake grpc.RpcError with .code() and .details() methods."""
    import grpc

    class FakeRpcError(grpc.RpcError):
        def code(self):
            return status_code

        def details(self):
            return details_msg

    return FakeRpcError(details_msg)


# ---- error handling tests ----
def test_handle_error_tarash_exception_passthrough(handler, base_config):
    ex = ValidationError("already mapped", provider="xai")
    result = handler._handle_error(base_config, "req-1", ex)
    assert result is ex


def test_handle_error_deadline_exceeded_maps_to_timeout(handler, base_config):
    import grpc

    ex = _make_grpc_error(grpc.StatusCode.DEADLINE_EXCEEDED, "deadline exceeded")
    result = handler._handle_error(base_config, "req-1", ex)
    assert isinstance(result, TimeoutError)


def test_handle_error_unavailable_maps_to_connection_error(handler, base_config):
    import grpc

    ex = _make_grpc_error(grpc.StatusCode.UNAVAILABLE, "service unavailable")
    result = handler._handle_error(base_config, "req-1", ex)
    assert isinstance(result, HTTPConnectionError)


def test_handle_error_unauthenticated_maps_to_http_error_401(handler, base_config):
    import grpc

    ex = _make_grpc_error(grpc.StatusCode.UNAUTHENTICATED, "invalid API key")
    result = handler._handle_error(base_config, "req-1", ex)
    assert isinstance(result, HTTPError)
    assert result.status_code == 401


def test_handle_error_permission_denied_maps_to_content_moderation(
    handler, base_config
):
    import grpc

    ex = _make_grpc_error(grpc.StatusCode.PERMISSION_DENIED, "content policy violation")
    result = handler._handle_error(base_config, "req-1", ex)
    assert isinstance(result, ContentModerationError)


def test_handle_error_invalid_argument_maps_to_validation_error(handler, base_config):
    import grpc

    ex = _make_grpc_error(grpc.StatusCode.INVALID_ARGUMENT, "invalid prompt")
    result = handler._handle_error(base_config, "req-1", ex)
    assert isinstance(result, ValidationError)


def test_handle_error_generic_exception_returns_generation_failed(handler, base_config):
    ex = RuntimeError("unexpected crash")
    result = handler._handle_error(base_config, "req-1", ex)
    assert isinstance(result, GenerationFailedError)
    assert "Error while generating" in str(result)


# ---- video generation integration tests ----
@pytest.mark.asyncio
async def test_generate_video_async_success(handler, base_config, basic_request):
    mock_client = AsyncMock()
    pending_resp = _make_xai_poll_response("PENDING")
    done_resp = _make_xai_poll_response("DONE", url="https://vidgen.x.ai/video.mp4")

    mock_client.video.start = AsyncMock(return_value=pending_resp)
    mock_client.video.get = AsyncMock(side_effect=[pending_resp, done_resp])

    with patch(
        "tarash.tarash_gateway.providers.xai.AsyncClient", return_value=mock_client
    ):
        response = await handler.generate_video_async(base_config, basic_request)

    assert response.status == "completed"
    assert response.video == "https://vidgen.x.ai/video.mp4"


@pytest.mark.asyncio
async def test_generate_video_async_fires_progress_callbacks(
    handler, base_config, basic_request
):
    mock_client = AsyncMock()
    pending_resp = _make_xai_poll_response("PENDING")
    done_resp = _make_xai_poll_response("DONE", url="https://vidgen.x.ai/video.mp4")

    mock_client.video.start = AsyncMock(return_value=pending_resp)
    mock_client.video.get = AsyncMock(return_value=done_resp)

    updates = []

    def on_progress(update):
        updates.append(update)

    with patch(
        "tarash.tarash_gateway.providers.xai.AsyncClient", return_value=mock_client
    ):
        await handler.generate_video_async(
            base_config, basic_request, on_progress=on_progress
        )

    assert len(updates) >= 1


@pytest.mark.asyncio
async def test_generate_video_async_expired_raises_generation_failed(
    handler, base_config, basic_request
):
    mock_client = AsyncMock()
    pending_resp = _make_xai_poll_response("PENDING")
    expired_resp = _make_xai_poll_response("EXPIRED")

    mock_client.video.start = AsyncMock(return_value=pending_resp)
    mock_client.video.get = AsyncMock(return_value=expired_resp)

    with patch(
        "tarash.tarash_gateway.providers.xai.AsyncClient", return_value=mock_client
    ):
        with pytest.raises(GenerationFailedError, match="expired"):
            await handler.generate_video_async(base_config, basic_request)


@pytest.mark.asyncio
async def test_generate_video_async_timeout(handler, base_config, basic_request):
    mock_client = AsyncMock()
    pending_resp = _make_xai_poll_response("PENDING")

    mock_client.video.start = AsyncMock(return_value=pending_resp)
    mock_client.video.get = AsyncMock(return_value=pending_resp)

    with patch(
        "tarash.tarash_gateway.providers.xai.AsyncClient", return_value=mock_client
    ):
        with pytest.raises(TimeoutError, match="timed out"):
            await handler.generate_video_async(base_config, basic_request)


def test_generate_video_sync_success(handler, base_config, basic_request):
    mock_client = MagicMock()
    done_resp = _make_xai_poll_response("DONE", url="https://vidgen.x.ai/video.mp4")

    mock_client.video.start.return_value = done_resp
    mock_client.video.get.return_value = done_resp

    with patch("tarash.tarash_gateway.providers.xai.Client", return_value=mock_client):
        response = handler.generate_video(base_config, basic_request)

    assert response.status == "completed"
    assert response.video == "https://vidgen.x.ai/video.mp4"


# ---- registration tests ----
def test_xai_handler_exported_from_providers():
    from tarash.tarash_gateway.providers import XaiProviderHandler  # noqa: F401


def test_xai_registered_in_registry():
    from tarash.tarash_gateway.registry import _HANDLER_INSTANCES, get_handler
    from tarash.tarash_gateway.providers.xai import XaiProviderHandler

    _HANDLER_INSTANCES.pop("xai", None)

    config = VideoGenerationConfig(
        model="grok-imagine-video",
        provider="xai",
        api_key="test-key",
    )

    with (
        patch("tarash.tarash_gateway.providers.xai.has_xai_sdk", True),
        patch("tarash.tarash_gateway.providers.xai.Client", MagicMock()),
        patch("tarash.tarash_gateway.providers.xai.AsyncClient", MagicMock()),
    ):
        handler_instance = get_handler(config)

    assert isinstance(handler_instance, XaiProviderHandler)
