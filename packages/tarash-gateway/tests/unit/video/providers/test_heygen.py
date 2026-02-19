"""Tests for HeyGenProviderHandler."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tarash.tarash_gateway.exceptions import (
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
from tarash.tarash_gateway.providers.heygen import (
    HeyGenProviderHandler,
    parse_heygen_status,
)


# ==================== Fixtures ====================


@pytest.fixture
def handler():
    """Create a HeyGenProviderHandler instance."""
    return HeyGenProviderHandler()


@pytest.fixture
def base_config():
    """Create a base VideoGenerationConfig with HeyGen provider."""
    return VideoGenerationConfig(
        model="heygen/avatar-v2",
        provider="heygen",
        api_key="test-api-key",
        timeout=600,
        poll_interval=0,
        max_poll_attempts=3,
        provider_config={
            "avatar_id": "default-avatar",
            "voice_id": "default-voice",
        },
    )


@pytest.fixture
def basic_request():
    """Create a basic video generation request."""
    return VideoGenerationRequest(
        prompt="Hello, world! This is a test script.",
        aspect_ratio="16:9",
    )


@pytest.fixture
def mock_sync_client(handler):
    """Patch httpx.Client and provide mock sync client."""
    mock = MagicMock()
    with patch(
        "tarash.tarash_gateway.providers.heygen.httpx.Client", return_value=mock
    ):
        handler._sync_client_cache.clear()
        yield mock


@pytest.fixture
def mock_async_client(handler):
    """Patch httpx.AsyncClient and provide mock async client."""
    mock = AsyncMock()
    with patch(
        "tarash.tarash_gateway.providers.heygen.httpx.AsyncClient", return_value=mock
    ):
        handler._async_client_cache.clear()
        yield mock


# ==================== Client Caching Tests ====================


def test_get_client_caches_sync_client(handler, base_config, mock_sync_client):
    """Same api_key returns the same sync client instance."""
    client1 = handler._get_client(base_config, "sync")
    client2 = handler._get_client(base_config, "sync")

    assert client1 is client2
    assert client1 is mock_sync_client


def test_get_client_caches_async_client(handler, base_config, mock_async_client):
    """Same api_key returns the same async client instance."""
    client1 = handler._get_client(base_config, "async")
    client2 = handler._get_client(base_config, "async")

    assert client1 is client2
    assert client1 is mock_async_client


def test_get_client_different_api_keys_separate_cache(handler):
    """Different api_key values create separate cache entries."""
    config1 = VideoGenerationConfig(
        model="heygen/avatar-v2",
        provider="heygen",
        api_key="key-one",
        provider_config={},
    )
    config2 = VideoGenerationConfig(
        model="heygen/avatar-v2",
        provider="heygen",
        api_key="key-two",
        provider_config={},
    )

    with patch("tarash.tarash_gateway.providers.heygen.httpx.Client") as mock_cls:
        mock_cls.side_effect = [MagicMock(), MagicMock()]
        handler._sync_client_cache.clear()

        client1 = handler._get_client(config1, "sync")
        client2 = handler._get_client(config2, "sync")

        assert client1 is not client2


# ==================== Request Conversion Tests ====================


def test_convert_request_uses_provider_config_defaults(
    handler, base_config, basic_request
):
    """avatar_id/voice_id come from provider_config; background and voice use defaults."""
    payload = handler._convert_request(base_config, basic_request)

    vi = payload["video_inputs"][0]
    assert vi["character"]["avatar_id"] == "default-avatar"
    assert vi["voice"]["voice_id"] == "default-voice"
    assert vi["voice"]["input_text"] == basic_request.prompt
    assert vi["background"]["type"] == "color"
    assert vi["background"]["value"] == "#FFFFFF"
    assert vi["voice"]["speed"] == 1.0
    assert vi["voice"]["pitch"] == 0


def test_convert_request_extra_params_override_provider_config(handler, base_config):
    """extra_params avatar_id/voice_id override provider_config defaults."""
    request = VideoGenerationRequest(
        prompt="Override test",
        aspect_ratio="16:9",
        extra_params={
            "avatar_id": "override-avatar",
            "voice_id": "override-voice",
        },
    )
    payload = handler._convert_request(base_config, request)

    vi = payload["video_inputs"][0]
    assert vi["character"]["avatar_id"] == "override-avatar"
    assert vi["voice"]["voice_id"] == "override-voice"


def test_convert_request_default_aspect_ratio(handler, base_config):
    """No aspect_ratio defaults to 16:9 (1920×1080)."""
    request = VideoGenerationRequest(prompt="Test")
    payload = handler._convert_request(base_config, request)
    assert payload["dimension"] == {"width": 1920, "height": 1080}


def test_convert_request_missing_avatar_id_raises_validation_error(
    handler, basic_request
):
    """Missing avatar_id raises ValidationError."""
    config = VideoGenerationConfig(
        model="heygen/avatar-v2",
        provider="heygen",
        api_key="key",
        provider_config={"voice_id": "some-voice"},
    )
    with pytest.raises(ValidationError, match="avatar_id is required"):
        handler._convert_request(config, basic_request)


def test_convert_request_missing_voice_id_raises_validation_error(
    handler, basic_request
):
    """Missing voice_id raises ValidationError."""
    config = VideoGenerationConfig(
        model="heygen/avatar-v2",
        provider="heygen",
        api_key="key",
        provider_config={"avatar_id": "some-avatar"},
    )
    with pytest.raises(ValidationError, match="voice_id is required"):
        handler._convert_request(config, basic_request)


def test_convert_request_all_optional_params(handler, base_config):
    """All optional HeyGen params are correctly mapped into the payload."""
    request = VideoGenerationRequest(
        prompt="All params test",
        aspect_ratio="1:1",
        extra_params={
            "avatar_style": "circle",
            "background_type": "color",
            "background_value": "#000000",
            "voice_speed": 1.2,
            "voice_pitch": 10,
            "voice_emotion": "Excited",
            "caption": True,
            "title": "My Video",
            "matting": True,
        },
    )
    payload = handler._convert_request(base_config, request)

    vi = payload["video_inputs"][0]
    assert vi["character"]["avatar_style"] == "circle"
    assert vi["character"]["matting"] is True
    assert vi["background"]["type"] == "color"
    assert vi["background"]["value"] == "#000000"
    assert vi["voice"]["speed"] == 1.2
    assert vi["voice"]["pitch"] == 10
    assert vi["voice"]["emotion"] == "Excited"
    assert payload["caption"] is True
    assert payload["title"] == "My Video"


def test_convert_request_ignores_duration_and_media(handler, base_config):
    """duration_seconds, image_list, and video are silently ignored."""
    request = VideoGenerationRequest(
        prompt="Test",
        aspect_ratio="16:9",
        duration_seconds=10,
        image_list=[{"type": "reference", "image": "https://example.com/img.jpg"}],
    )
    # Should not raise
    payload = handler._convert_request(base_config, request)
    assert "duration" not in payload
    assert "image_list" not in payload


# ==================== Response Conversion Tests ====================


def test_convert_response_success(handler, base_config, basic_request):
    """Successful status data is converted to VideoGenerationResponse."""
    status_data = {
        "id": "vid-123",
        "status": "completed",
        "video_url": "https://cdn.heygen.com/video.mp4",
        "thumbnail_url": "https://cdn.heygen.com/thumb.jpg",
        "duration": 12.5,
    }
    resp = handler._convert_response(base_config, basic_request, "vid-123", status_data)

    assert resp.request_id == "vid-123"
    assert resp.video == "https://cdn.heygen.com/video.mp4"
    assert resp.status == "completed"
    assert resp.duration == 12.5
    assert resp.content_type == "video/mp4"
    assert resp.provider_metadata["thumbnail_url"] == "https://cdn.heygen.com/thumb.jpg"


def test_convert_response_failed_raises(handler, base_config, basic_request):
    """Failed status raises GenerationFailedError."""
    status_data = {
        "id": "vid-123",
        "status": "failed",
        "error": {"message": "Insufficient credits"},
    }
    with pytest.raises(GenerationFailedError, match="Insufficient credits"):
        handler._convert_response(base_config, basic_request, "vid-123", status_data)


def test_convert_response_no_url_raises(handler, base_config, basic_request):
    """Completed status without video_url raises GenerationFailedError."""
    status_data = {"id": "vid-123", "status": "completed"}
    with pytest.raises(GenerationFailedError, match="No video URL"):
        handler._convert_response(base_config, basic_request, "vid-123", status_data)


# ==================== Status Parsing Tests ====================


@pytest.mark.parametrize(
    "heygen_status,expected",
    [
        ("pending", "queued"),
        ("waiting", "queued"),
        ("processing", "processing"),
        ("completed", "completed"),
        ("failed", "failed"),
    ],
)
def test_parse_heygen_status_maps_all_statuses(heygen_status, expected):
    """All HeyGen statuses map to the correct internal StatusType."""
    update = parse_heygen_status("vid-1", {"status": heygen_status})
    assert update.status == expected
    assert update.request_id == "vid-1"
    assert update.progress_percent is None


def test_parse_heygen_status_unknown_status_defaults_to_processing():
    """Unknown status strings default to 'processing'."""
    update = parse_heygen_status("vid-1", {"status": "unknown_status"})
    assert update.status == "processing"


# ==================== Error Handling Tests ====================


def test_handle_error_400_returns_validation_error(handler, base_config, basic_request):
    """HTTP 400 errors map to ValidationError."""
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.json.return_value = {"error": "bad request"}
    ex = httpx.HTTPStatusError(
        "400 Bad Request", request=MagicMock(), response=mock_response
    )

    result = handler._handle_error(base_config, basic_request, "vid-1", ex)
    assert isinstance(result, ValidationError)
    assert result.raw_response["status_code"] == 400  # type: ignore[index]


def test_handle_error_401_returns_http_error(handler, base_config, basic_request):
    """HTTP 401 errors map to HTTPError."""
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.json.return_value = {"error": "unauthorized"}
    ex = httpx.HTTPStatusError(
        "401 Unauthorized", request=MagicMock(), response=mock_response
    )

    result = handler._handle_error(base_config, basic_request, "vid-1", ex)
    assert isinstance(result, HTTPError)
    assert result.status_code == 401


def test_handle_error_connection_error_returns_http_connection_error(
    handler, base_config, basic_request
):
    """ConnectError maps to HTTPConnectionError."""
    ex = httpx.ConnectError("Connection refused")

    result = handler._handle_error(base_config, basic_request, "vid-1", ex)
    assert isinstance(result, HTTPConnectionError)


def test_handle_error_timeout_returns_timeout_error(
    handler, base_config, basic_request
):
    """TimeoutException maps to TimeoutError."""
    ex = httpx.TimeoutException("Request timed out")

    result = handler._handle_error(base_config, basic_request, "vid-1", ex)
    assert isinstance(result, TimeoutError)


def test_handle_error_tarash_exception_passthrough(handler, base_config, basic_request):
    """TarashException subclasses are returned as-is."""
    ex = ValidationError("already wrapped", provider="heygen")

    result = handler._handle_error(base_config, basic_request, "vid-1", ex)
    assert result is ex


def test_handle_error_generic_exception_returns_generation_failed(
    handler, base_config, basic_request
):
    """Unknown exceptions map to GenerationFailedError."""
    ex = RuntimeError("unexpected")

    result = handler._handle_error(base_config, basic_request, "vid-1", ex)
    assert isinstance(result, GenerationFailedError)
    assert "Error while generating video" in str(result)


# ==================== Integration Tests ====================


@pytest.mark.asyncio
async def test_generate_video_async_success(
    handler, base_config, basic_request, mock_async_client
):
    """Successful async generation returns VideoGenerationResponse."""
    create_resp = MagicMock()
    create_resp.json.return_value = {"error": None, "data": {"video_id": "vid-abc"}}
    create_resp.raise_for_status = MagicMock()

    status_resp = MagicMock()
    status_resp.json.return_value = {
        "data": {
            "status": "completed",
            "video_url": "https://cdn.heygen.com/video.mp4",
            "duration": 12.5,
        }
    }
    status_resp.raise_for_status = MagicMock()

    mock_async_client.post = AsyncMock(return_value=create_resp)
    mock_async_client.get = AsyncMock(return_value=status_resp)

    response = await handler.generate_video_async(base_config, basic_request)

    assert response.request_id == "vid-abc"
    assert response.video == "https://cdn.heygen.com/video.mp4"
    assert response.status == "completed"
    assert response.duration == 12.5
    mock_async_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_generate_video_async_polls_until_complete(
    handler, base_config, basic_request, mock_async_client
):
    """Polling continues until terminal status is reached."""
    create_resp = MagicMock()
    create_resp.json.return_value = {"error": None, "data": {"video_id": "vid-abc"}}
    create_resp.raise_for_status = MagicMock()

    processing_resp = MagicMock()
    processing_resp.json.return_value = {"data": {"status": "processing"}}
    processing_resp.raise_for_status = MagicMock()

    completed_resp = MagicMock()
    completed_resp.json.return_value = {
        "data": {
            "status": "completed",
            "video_url": "https://cdn.heygen.com/video.mp4",
        }
    }
    completed_resp.raise_for_status = MagicMock()

    mock_async_client.post = AsyncMock(return_value=create_resp)
    mock_async_client.get = AsyncMock(side_effect=[processing_resp, completed_resp])

    response = await handler.generate_video_async(base_config, basic_request)

    assert response.status == "completed"
    assert mock_async_client.get.call_count == 2


@pytest.mark.asyncio
async def test_generate_video_async_raises_on_failed_status(
    handler, base_config, basic_request, mock_async_client
):
    """Failed HeyGen status raises GenerationFailedError."""
    create_resp = MagicMock()
    create_resp.json.return_value = {"error": None, "data": {"video_id": "vid-abc"}}
    create_resp.raise_for_status = MagicMock()

    failed_resp = MagicMock()
    failed_resp.json.return_value = {
        "data": {
            "status": "failed",
            "error": {"message": "Insufficient credits"},
        }
    }
    failed_resp.raise_for_status = MagicMock()

    mock_async_client.post = AsyncMock(return_value=create_resp)
    mock_async_client.get = AsyncMock(return_value=failed_resp)

    with pytest.raises(GenerationFailedError, match="Insufficient credits"):
        await handler.generate_video_async(base_config, basic_request)


@pytest.mark.asyncio
async def test_generate_video_async_raises_timeout_on_max_attempts(
    handler, base_config, basic_request, mock_async_client
):
    """TimeoutError is raised after max_poll_attempts without terminal status."""
    create_resp = MagicMock()
    create_resp.json.return_value = {"error": None, "data": {"video_id": "vid-abc"}}
    create_resp.raise_for_status = MagicMock()

    processing_resp = MagicMock()
    processing_resp.json.return_value = {"data": {"status": "processing"}}
    processing_resp.raise_for_status = MagicMock()

    mock_async_client.post = AsyncMock(return_value=create_resp)
    mock_async_client.get = AsyncMock(return_value=processing_resp)

    with pytest.raises(TimeoutError, match="timed out"):
        await handler.generate_video_async(base_config, basic_request)


@pytest.mark.asyncio
async def test_generate_video_async_sends_progress_updates(
    handler, base_config, basic_request, mock_async_client
):
    """Progress callbacks are invoked during polling."""
    create_resp = MagicMock()
    create_resp.json.return_value = {"error": None, "data": {"video_id": "vid-abc"}}
    create_resp.raise_for_status = MagicMock()

    status_resp = MagicMock()
    status_resp.json.return_value = {
        "data": {"status": "completed", "video_url": "https://cdn.heygen.com/video.mp4"}
    }
    status_resp.raise_for_status = MagicMock()

    mock_async_client.post = AsyncMock(return_value=create_resp)
    mock_async_client.get = AsyncMock(return_value=status_resp)

    updates = []

    def on_progress(update):
        updates.append(update)

    await handler.generate_video_async(
        base_config, basic_request, on_progress=on_progress
    )

    # Initial queued update + at least one poll update
    assert len(updates) >= 2
    statuses = [u.status for u in updates]
    assert "queued" in statuses
    assert "completed" in statuses


def test_generate_video_sync_success(
    handler, base_config, basic_request, mock_sync_client
):
    """Successful sync generation returns VideoGenerationResponse."""
    create_resp = MagicMock()
    create_resp.json.return_value = {"error": None, "data": {"video_id": "vid-xyz"}}
    create_resp.raise_for_status = MagicMock()

    status_resp = MagicMock()
    status_resp.json.return_value = {
        "data": {
            "status": "completed",
            "video_url": "https://cdn.heygen.com/video2.mp4",
        }
    }
    status_resp.raise_for_status = MagicMock()

    mock_sync_client.post = MagicMock(return_value=create_resp)
    mock_sync_client.get = MagicMock(return_value=status_resp)

    response = handler.generate_video(base_config, basic_request)

    assert response.request_id == "vid-xyz"
    assert response.video == "https://cdn.heygen.com/video2.mp4"
    assert response.status == "completed"
    mock_sync_client.post.assert_called_once()


def test_generate_video_sync_sends_progress_updates(
    handler, base_config, basic_request, mock_sync_client
):
    """Sync progress callbacks are invoked during polling."""
    create_resp = MagicMock()
    create_resp.json.return_value = {"error": None, "data": {"video_id": "vid-xyz"}}
    create_resp.raise_for_status = MagicMock()

    status_resp = MagicMock()
    status_resp.json.return_value = {
        "data": {"status": "completed", "video_url": "https://cdn.heygen.com/video.mp4"}
    }
    status_resp.raise_for_status = MagicMock()

    mock_sync_client.post = MagicMock(return_value=create_resp)
    mock_sync_client.get = MagicMock(return_value=status_resp)

    updates = []
    handler.generate_video(
        base_config, basic_request, on_progress=lambda u: updates.append(u)
    )

    # Initial queued update + at least one poll update
    assert len(updates) >= 2
    statuses = [u.status for u in updates]
    assert "queued" in statuses
