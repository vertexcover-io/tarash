"""Tests for FalProviderHandler."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fal_client.client import FalClientHTTPError
# from pydantic import ValidationError as Pydantic ValidationError  # Not used with FieldMapper approach

from tarash.tarash_gateway.video.exceptions import (
    ProviderAPIError,
    ValidationError,
    VideoGenerationError,
    handle_video_generation_errors,
)
from tarash.tarash_gateway.video.models import (
    VideoGenerationConfig,
    VideoGenerationRequest,
)
from tarash.tarash_gateway.video.providers.fal import (
    FalProviderHandler,
    parse_fal_status,
)


# ==================== Fixtures ====================


@pytest.fixture
def mock_sync_client():
    """Patch fal_client.SyncClient and provide mock."""
    mock = MagicMock()
    with patch(
        "tarash.tarash_gateway.video.providers.fal.fal_client.SyncClient",
        return_value=mock,
    ):
        yield mock


@pytest.fixture
def mock_async_client():
    """Patch fal_client.AsyncClient and provide mock."""
    mock = AsyncMock()
    with patch(
        "tarash.tarash_gateway.video.providers.fal.fal_client.AsyncClient",
        return_value=mock,
    ):
        yield mock


@pytest.fixture
def handler():
    """Create a FalProviderHandler instance."""
    return FalProviderHandler()


@pytest.fixture
def base_config():
    """Create a base VideoGenerationConfig."""
    return VideoGenerationConfig(
        model="fal-ai/veo3.1",
        provider="fal",
        api_key="test-api-key",
        timeout=600,
    )


@pytest.fixture
def base_request():
    """Create a base VideoGenerationRequest."""
    return VideoGenerationRequest(prompt="Test prompt")


# ==================== Initialization Tests ====================


def test_init_creates_empty_caches(handler):
    """Test that handler initializes with empty client caches."""
    assert handler._sync_client_cache == {}
    assert handler._async_client_cache == {}


# ==================== Client Management Tests ====================


def test_get_client_creates_and_caches_sync_client(
    handler, base_config, mock_sync_client
):
    """Test sync client creation and caching."""
    # Clear cache first
    handler._sync_client_cache.clear()

    client1 = handler._get_client(base_config, "sync")
    client2 = handler._get_client(base_config, "sync")

    assert client1 is client2  # Same instance (cached)
    assert client1 is mock_sync_client


def test_get_client_creates_and_caches_async_client(
    handler, base_config, mock_async_client
):
    """Test async client creation and caching."""
    # Clear cache first
    handler._async_client_cache.clear()

    client1 = handler._get_client(base_config, "async")
    client2 = handler._get_client(base_config, "async")

    assert client1 is client2  # Same instance (cached)
    assert client1 is mock_async_client


@pytest.mark.parametrize(
    "api_key,base_url",
    [
        ("key1", None),
        ("key2", None),
        ("key1", "https://api1.example.com"),
        ("key1", "https://api2.example.com"),
    ],
)
def test_get_client_creates_different_clients_for_different_configs(
    handler, api_key, base_url
):
    """Test different clients for different API keys and base_urls."""
    # Clear cache first
    handler._sync_client_cache.clear()

    mock_client1 = MagicMock()
    mock_client2 = MagicMock()

    config1 = VideoGenerationConfig(
        model="fal-ai/veo3.1",
        provider="fal",
        api_key=api_key,
        base_url=base_url,
        timeout=600,
    )
    config2 = VideoGenerationConfig(
        model="fal-ai/veo3.1",
        provider="fal",
        api_key="different-key",
        base_url="https://different.example.com",
        timeout=600,
    )

    with patch(
        "tarash.tarash_gateway.video.providers.fal.fal_client.SyncClient",
        side_effect=[mock_client1, mock_client2],
    ):
        client1 = handler._get_client(config1, "sync")
        client2 = handler._get_client(config2, "sync")

        assert client1 is not client2  # Different instances


# ==================== Request Conversion Tests ====================


def test_convert_request_with_minimal_fields(handler, base_config):
    """Test conversion with only prompt."""
    request = VideoGenerationRequest(prompt="A test video")
    result = handler._convert_request(base_config, request)

    assert result == {"prompt": "A test video"}


def test_convert_request_with_all_optional_fields(handler, base_config):
    """Test conversion with all optional fields and validated model_params."""
    request = VideoGenerationRequest(
        prompt="A test video",
        duration_seconds=5,
        resolution="1080p",
        aspect_ratio="16:9",
        image_list=[{"image": "https://example.com/image.jpg", "type": "reference"}],
        video="https://example.com/video.mp4",
        extra_params={"seed": 42, "generate_audio": True},
    )

    result = handler._convert_request(base_config, request)

    assert result["prompt"] == "A test video"
    assert result["duration"] == 5
    assert result["resolution"] == "1080p"
    assert result["aspect_ratio"] == "16:9"
    assert result["image_urls"] == ["https://example.com/image.jpg"]
    assert result["video_url"] == "https://example.com/video.mp4"
    assert result["seed"] == 42
    assert result["generate_audio"] is True
    # None values should not be included
    assert "negative_prompt" not in result


def test_convert_request_propagates_validation_errors(handler):
    """Test that validation errors propagate from Pydantic models."""
    # Use a model that has specific validation (Minimax with duration limit)
    config = VideoGenerationConfig(
        model="fal-ai/minimax-video",
        provider="fal",
        api_key="test-key",
    )
    request = VideoGenerationRequest(
        prompt="test",
        duration_seconds=15,  # Invalid for Minimax (only supports 6 or 10 seconds)
    )

    with pytest.raises(
        ValueError, match="Minimax only supports 6 or 10 second durations"
    ):
        handler._convert_request(config, request)


# ==================== Response Conversion Tests ====================


def test_convert_response_with_complete_dict_response(
    handler, base_config, base_request
):
    """Test conversion with complete dict response including all fields."""
    provider_response = {
        "video": {"url": "https://example.com/video.mp4"},
        "audio": {"url": "https://example.com/audio.mp3"},
        "duration": 5.5,
        "resolution": "1080p",
        "aspect_ratio": "16:9",
    }

    result = handler._convert_response(
        base_config, base_request, "req-123", provider_response
    )

    assert result.request_id == "req-123"
    assert result.video == "https://example.com/video.mp4"
    assert result.audio_url == "https://example.com/audio.mp3"
    assert result.duration == 5.5
    assert result.resolution == "1080p"
    assert result.aspect_ratio == "16:9"
    assert result.status == "completed"
    assert result.raw_response == provider_response
    assert result.provider_metadata == {}


def test_convert_response_with_missing_video_url_raises_error(
    handler, base_config, base_request
):
    """Test missing video URL raises ProviderAPIError with proper error message and raw_response."""
    provider_response = {"some_field": "value"}

    with pytest.raises(ProviderAPIError, match="No video URL found") as exc_info:
        handler._convert_response(
            base_config, base_request, "req-789", provider_response
        )

    assert exc_info.value.provider == "fal"
    assert exc_info.value.raw_response == provider_response

    # Test with non-dict response (empty dict after extraction)
    with pytest.raises(ProviderAPIError, match="No video URL found"):
        handler._convert_response(base_config, base_request, "req-789", {})


# ==================== Error Handling Tests ====================


def test_handle_error_with_video_generation_error(handler, base_config, base_request):
    """Test VideoGenerationError is returned as-is."""
    error = VideoGenerationError("Test error", provider="fal", model="test-model")
    result = handler._handle_error(base_config, base_request, "req-1", error)

    assert result is error


def test_handle_error_with_fal_client_http_error(handler, base_config, base_request):
    """Test FalClientHTTPError is converted to VideoGenerationError."""
    mock_response = MagicMock()
    mock_response.content = b"Error response"

    http_error = FalClientHTTPError(
        status_code=500,
        message="Internal Server Error",
        response_headers={"Content-Type": "application/json"},
        response=mock_response,
    )

    result = handler._handle_error(base_config, base_request, "req-2", http_error)

    assert isinstance(result, VideoGenerationError)
    assert "status code 500" in result.message
    assert result.provider == "fal"
    assert result.raw_response["status_code"] == 500
    assert result.raw_response["response_headers"] == {
        "Content-Type": "application/json"
    }


def test_handle_error_with_unknown_exception(handler, base_config, base_request):
    """Test unknown exception is converted to VideoGenerationError."""
    unknown_error = ValueError("Something went wrong")

    with pytest.raises(VideoGenerationError, match="Unknown Error"):
        handler._handle_error(base_config, base_request, "req-3", unknown_error)


# ==================== Status Parsing Tests ====================


def test_parse_fal_status_completed():
    """Test parsing Completed status with metrics and logs."""

    # Create mock classes that will pass isinstance checks
    class MockCompleted:
        def __init__(self):
            self.metrics = {"duration": 5.0}
            self.logs = ["Log entry 1", "Log entry 2"]

    mock_status = MockCompleted()

    # Patch the Completed import in the fal module
    with patch("tarash.tarash_gateway.video.providers.fal.Completed", MockCompleted):
        result = parse_fal_status("req-1", mock_status)

    assert result.request_id == "req-1"
    assert result.status == "completed"
    assert result.update["metrics"] == {"duration": 5.0}
    assert result.update["logs"] == ["Log entry 1", "Log entry 2"]


def test_parse_fal_status_queued():
    """Test parsing Queued status with position."""

    class MockQueued:
        def __init__(self):
            self.position = 3

    mock_status = MockQueued()

    with patch("tarash.tarash_gateway.video.providers.fal.Queued", MockQueued):
        result = parse_fal_status("req-2", mock_status)

    assert result.request_id == "req-2"
    assert result.status == "queued"
    assert result.update["position"] == 3


def test_parse_fal_status_in_progress():
    """Test parsing InProgress status with logs."""

    class MockInProgress:
        def __init__(self):
            self.logs = ["Processing...", "Almost done"]

    mock_status = MockInProgress()

    with patch("tarash.tarash_gateway.video.providers.fal.InProgress", MockInProgress):
        result = parse_fal_status("req-3", mock_status)

    assert result.request_id == "req-3"
    assert result.status == "processing"
    assert result.update == {"logs": ["Processing...", "Almost done"]}


def test_parse_fal_status_unknown_raises_error():
    """Test parsing unknown status raises ValueError."""
    unknown_status = MagicMock()

    with pytest.raises(ValueError, match="Unknown status"):
        parse_fal_status("req-4", unknown_status)


# ==================== Async Video Generation Tests ====================


@pytest.mark.asyncio
async def test_generate_video_async_success_with_progress_callbacks(
    handler, base_config, base_request
):
    """Test successful async generation with sync and async progress callbacks."""

    # Create mock status classes
    class MockQueued:
        def __init__(self):
            self.position = 1

    class MockInProgress:
        def __init__(self):
            self.logs = ["Processing"]

    class MockCompleted:
        def __init__(self):
            self.metrics = {}
            self.logs = []

    # Setup mock handler
    mock_handler = AsyncMock()
    mock_handler.request_id = "fal-req-123"

    # Mock iter_events to yield status updates
    async def mock_iter_events(with_logs):
        yield MockQueued()
        yield MockInProgress()
        yield MockCompleted()

    mock_handler.iter_events = mock_iter_events
    mock_handler.get = AsyncMock(
        return_value={"video": {"url": "https://example.com/video.mp4"}}
    )

    # Setup mock async client
    mock_async_client = AsyncMock()
    mock_async_client.submit = AsyncMock(return_value=mock_handler)

    # Clear cache and patch
    handler._async_client_cache.clear()
    with (
        patch(
            "tarash.tarash_gateway.video.providers.fal.fal_client.AsyncClient",
            return_value=mock_async_client,
        ),
        patch("tarash.tarash_gateway.video.providers.fal.Queued", MockQueued),
        patch("tarash.tarash_gateway.video.providers.fal.InProgress", MockInProgress),
        patch("tarash.tarash_gateway.video.providers.fal.Completed", MockCompleted),
    ):
        # Test with sync callback
        progress_calls = []

        def sync_callback(update):
            progress_calls.append(update)

        result = await handler.generate_video_async(
            base_config, base_request, on_progress=sync_callback
        )

        assert result.request_id == "fal-req-123"
        assert result.video == "https://example.com/video.mp4"
        assert len(progress_calls) == 3  # queued, processing, completed

        # Test with async callback
        async_progress_calls = []

        async def async_callback(update):
            async_progress_calls.append(update)

        await handler.generate_video_async(
            base_config, base_request, on_progress=async_callback
        )

        assert len(async_progress_calls) == 3


@pytest.mark.asyncio
async def test_generate_video_async_propagates_known_errors(
    handler, base_config, base_request, mock_async_client
):
    """Test that ValidationError and ProviderAPIError propagate without wrapping."""
    # Test Pydantic ValidationError propagation from model validation
    # Use Minimax model with duration that exceeds limits
    minimax_config = VideoGenerationConfig(
        model="fal-ai/minimax-video",
        provider="fal",
        api_key="test-key",
    )
    request_invalid = VideoGenerationRequest(
        prompt="test",
        duration_seconds=15,  # Invalid for Minimax (only supports 6 or 10 seconds)
    )

    with pytest.raises(
        VideoGenerationError, match="Minimax only supports 6 or 10 second durations"
    ):
        await handler.generate_video_async(minimax_config, request_invalid)

    mock_handler = AsyncMock()
    mock_handler.request_id = "req-1"

    # iter_events needs to be an async generator
    async def empty_iter_events(with_logs):
        return
        yield  # Make it a generator

    mock_handler.iter_events = empty_iter_events
    mock_handler.get = AsyncMock(return_value={})  # No video URL
    mock_async_client.submit = AsyncMock(return_value=mock_handler)

    handler._async_client_cache.clear()
    with pytest.raises(ProviderAPIError):
        await handler.generate_video_async(base_config, base_request)


@pytest.mark.asyncio
async def test_generate_video_async_handles_fal_http_error(
    handler, base_config, base_request, mock_async_client
):
    """Test FalClientHTTPError is handled by _handle_error."""
    mock_response = MagicMock()
    mock_response.content = b"Error"

    http_error = FalClientHTTPError(
        status_code=429,
        message="Rate limited",
        response_headers={},
        response=mock_response,
    )

    mock_async_client.submit = AsyncMock(side_effect=http_error)

    handler._async_client_cache.clear()
    with pytest.raises(VideoGenerationError, match="Unknown error"):
        await handler.generate_video_async(base_config, base_request)


@pytest.mark.asyncio
async def test_generate_video_async_wraps_unknown_exceptions(
    handler, base_config, base_request, mock_async_client
):
    """Test unknown exceptions are wrapped by decorator."""
    mock_async_client.submit = AsyncMock(side_effect=RuntimeError("Unexpected error"))

    handler._async_client_cache.clear()
    with pytest.raises(VideoGenerationError, match="Unknown error"):
        await handler.generate_video_async(base_config, base_request)


# ==================== Sync Video Generation Tests ====================


def test_generate_video_success_with_progress_callback(
    handler, base_config, base_request, mock_sync_client
):
    """Test successful sync generation with progress callback."""

    # Create mock status classes
    class MockQueued:
        def __init__(self):
            self.position = 1

    class MockCompleted:
        def __init__(self):
            self.metrics = {}
            self.logs = []

    mock_handler = MagicMock()
    mock_handler.request_id = "fal-req-456"

    def mock_iter_events(with_logs):
        yield MockQueued()
        yield MockCompleted()

    mock_handler.iter_events = mock_iter_events
    mock_handler.get.return_value = {"video": {"url": "https://example.com/video.mp4"}}

    progress_calls = []

    def progress_callback(update):
        progress_calls.append(update)

    # Clear cache and patch
    handler._sync_client_cache.clear()
    mock_sync_client.submit.return_value = mock_handler
    with (
        patch("tarash.tarash_gateway.video.providers.fal.Queued", MockQueued),
        patch("tarash.tarash_gateway.video.providers.fal.Completed", MockCompleted),
    ):
        result = handler.generate_video(
            base_config, base_request, on_progress=progress_callback
        )

    assert result.request_id == "fal-req-456"
    assert result.video == "https://example.com/video.mp4"
    assert len(progress_calls) == 2


def test_generate_video_handles_exceptions(
    handler, base_config, base_request, mock_sync_client
):
    """Test exception handling in sync generation."""
    mock_response = MagicMock()
    mock_response.content = b"Error"

    http_error = FalClientHTTPError(
        status_code=500,
        message="Server error",
        response_headers={},
        response=mock_response,
    )

    mock_sync_client.submit.side_effect = http_error

    handler._sync_client_cache.clear()
    with pytest.raises(VideoGenerationError, match="Unknown error"):
        handler.generate_video(base_config, base_request)


# ==================== Error Decorator Tests ====================


@pytest.mark.asyncio
async def test_handle_video_generation_errors_async_propagates_known_errors():
    """Test decorator propagates ValidationError, ProviderAPIError, VideoGenerationError."""

    @handle_video_generation_errors
    async def async_func(self, config, request):
        if request.prompt == "validation":
            raise ValidationError("Invalid", provider="fal")
        elif request.prompt == "provider":
            raise ProviderAPIError("API error", provider="fal")
        elif request.prompt == "video":
            raise VideoGenerationError("Gen error", provider="fal")
        else:
            raise RuntimeError("Unknown")

    config = VideoGenerationConfig(model="test-model", provider="fal", api_key="key")

    # Test ValidationError propagates
    with pytest.raises(ValidationError):
        await async_func(None, config, VideoGenerationRequest(prompt="validation"))

    # Test ProviderAPIError propagates
    with pytest.raises(ProviderAPIError):
        await async_func(None, config, VideoGenerationRequest(prompt="provider"))

    # Test VideoGenerationError propagates
    with pytest.raises(VideoGenerationError):
        await async_func(None, config, VideoGenerationRequest(prompt="video"))

    # Test unknown exception is wrapped
    with pytest.raises(VideoGenerationError, match="Unknown error") as exc_info:
        await async_func(None, config, VideoGenerationRequest(prompt="unknown"))

    assert exc_info.value.provider == "fal"
    assert exc_info.value.model == "test-model"
    assert "error" in exc_info.value.raw_response


def test_handle_video_generation_errors_sync_propagates_known_errors():
    """Test decorator propagates known errors for sync functions."""

    @handle_video_generation_errors
    def sync_func(self, config, request):
        if request.prompt == "validation":
            raise ValidationError("Invalid", provider="fal")
        elif request.prompt == "unknown":
            raise RuntimeError("Unknown")

    config = VideoGenerationConfig(model="test-model", provider="fal", api_key="key")

    # Test ValidationError propagates
    with pytest.raises(ValidationError):
        sync_func(None, config, VideoGenerationRequest(prompt="validation"))

    # Test unknown exception is wrapped
    with pytest.raises(VideoGenerationError, match="Unknown error"):
        sync_func(None, config, VideoGenerationRequest(prompt="unknown"))
