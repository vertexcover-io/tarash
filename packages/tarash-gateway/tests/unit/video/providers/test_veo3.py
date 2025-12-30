"""Tests for Veo3ProviderHandler."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
from tarash.tarash_gateway.video.providers.veo3 import (
    Veo3ProviderHandler,
    parse_veo3_operation,
)


# ==================== Fixtures ====================


@pytest.fixture
def mock_sync_client():
    """Patch Client and provide mock."""
    mock = MagicMock()
    with patch("tarash.tarash_gateway.video.providers.veo3.Client", return_value=mock):
        yield mock


@pytest.fixture
def mock_async_client():
    """Patch Client for async and provide mock.aio."""
    mock = MagicMock()
    mock.aio = AsyncMock()
    with patch("tarash.tarash_gateway.video.providers.veo3.Client", return_value=mock):
        yield mock.aio


@pytest.fixture
def handler():
    """Create a Veo3ProviderHandler instance."""
    return Veo3ProviderHandler()


@pytest.fixture
def base_config():
    """Create a base VideoGenerationConfig."""
    return VideoGenerationConfig(
        model="veo-3.0-flash-001",
        provider="veo3",
        api_key="test-api-key",
        timeout=600,
        poll_interval=1,
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
        ("key1", "https://us-central1-aiplatform.googleapis.com"),
        ("key1", "https://europe-west1-aiplatform.googleapis.com"),
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
        model="veo-3.0-flash-001",
        provider="veo3",
        api_key=api_key,
        base_url=base_url,
        timeout=600,
    )
    config2 = VideoGenerationConfig(
        model="veo-3.0-flash-001",
        provider="veo3",
        api_key="different-key",
        base_url="https://different.example.com",
        timeout=600,
    )

    with patch(
        "tarash.tarash_gateway.video.providers.veo3.Client",
        side_effect=[mock_client1, mock_client2],
    ):
        client1 = handler._get_client(config1, "sync")
        client2 = handler._get_client(config2, "sync")

        assert client1 is not client2  # Different instances


# ==================== Parameter Validation Tests ====================


def test_validate_params_with_empty_model_params(handler, base_config):
    """Test validation with empty model_params."""
    request_empty = VideoGenerationRequest(prompt="test", extra_params={})

    assert handler._validate_params(base_config, request_empty) == {}


def test_validate_params_with_valid_veo3_params(handler, base_config):
    """Test validation with valid Veo3VideoParams."""
    request = VideoGenerationRequest(
        prompt="test",
        extra_params={
            "person_generation": "allow_all",
        },
    )

    result = handler._validate_params(base_config, request)

    assert result == {
        "person_generation": "allow_all",
    }


def test_validate_params_with_invalid_person_generation(handler, base_config):
    """Test validation rejects invalid person_generation value."""
    request = VideoGenerationRequest(
        prompt="test",
        extra_params={
            "person_generation": "invalid_value",
        },
    )

    with pytest.raises(ValidationError):
        handler._validate_params(base_config, request)


# ==================== Request Conversion Tests ====================


def test_convert_request_with_minimal_fields(handler, base_config):
    """Test conversion with only prompt."""
    request = VideoGenerationRequest(prompt="A test video")
    result = handler._convert_request(base_config, request)

    assert result["prompt"] == "A test video"
    assert result["image"] is None
    assert result["video"] is None
    assert "config" in result


def test_convert_request_with_all_optional_fields(handler, base_config):
    """Test conversion with all optional fields and validated model_params."""
    request = VideoGenerationRequest(
        prompt="A test video",
        duration_seconds=5,
        aspect_ratio="16:9",
        number_of_videos=2,
        generate_audio=True,
        seed=42,
        negative_prompt="bad quality",
        enhance_prompt=True,
        image_list=[{"image": "https://example.com/image.jpg", "type": "reference"}],
        video="https://example.com/video.mp4",
    )

    result = handler._convert_request(base_config, request)

    assert result["prompt"] == "A test video"
    assert result["image"] == {"gcs_uri": "https://example.com/image.jpg"}
    assert result["video"] == {"uri": "https://example.com/video.mp4"}

    # Check config object
    config = result["config"]
    assert config.duration_seconds == 5
    assert config.aspect_ratio == "16:9"
    assert config.number_of_videos == 2
    assert config.generate_audio is True
    assert config.seed == 42
    assert config.negative_prompt == "bad quality"
    assert config.enhance_prompt is True


def test_convert_request_with_different_image_types(handler, base_config):
    """Test conversion with different image types (first_frame, last_frame, asset, style)."""
    request = VideoGenerationRequest(
        prompt="test",
        image_list=[
            {"image": "https://example.com/first.jpg", "type": "first_frame"},
            {"image": "https://example.com/last.jpg", "type": "last_frame"},
            {"image": "https://example.com/asset.jpg", "type": "asset"},
            {"image": "https://example.com/style.jpg", "type": "style"},
        ],
    )

    result = handler._convert_request(base_config, request)

    # first_frame should be in top-level 'image'
    assert result["image"] == {"gcs_uri": "https://example.com/first.jpg"}

    # last_frame and reference_images should be in config
    config = result["config"]
    assert config.last_frame.gcs_uri == "https://example.com/last.jpg"
    # asset and style should be in reference_images
    assert len(config.reference_images) == 2


def test_convert_request_with_image_bytes(handler, base_config):
    """Test conversion with image bytes instead of URLs."""
    request = VideoGenerationRequest(
        prompt="test",
        image_list=[
            {
                "image": {"content": b"fake-image-bytes", "content_type": "image/jpeg"},
                "type": "reference",
            }
        ],
    )

    result = handler._convert_request(base_config, request)

    assert result["image"] == {
        "image_bytes": b"fake-image-bytes",
        "mime_type": "image/jpeg",
    }
    assert result["prompt"] == "test"


def test_convert_request_propagates_validation_errors(handler, base_config):
    """Test that validation errors propagate."""
    request = VideoGenerationRequest(
        prompt="test",
        extra_params={"person_generation": "invalid"},
    )

    with pytest.raises(ValidationError):
        handler._convert_request(base_config, request)


# ==================== Response Conversion Tests ====================


def test_convert_response_with_complete_operation(handler, base_config, base_request):
    """Test conversion with complete operation including video URI."""
    # Create mock operation
    mock_video = MagicMock()
    mock_video.uri = "https://example.com/video.mp4"
    mock_video.video_bytes = None
    mock_video.mime_type = "video/mp4"

    mock_generated_video = MagicMock()
    mock_generated_video.video = mock_video

    mock_response = MagicMock()
    mock_response.generated_videos = [mock_generated_video]

    mock_operation = MagicMock()
    mock_operation.name = "operations/123"
    mock_operation.done = True
    mock_operation.error = None
    mock_operation.response = mock_response
    mock_operation.model_dump.return_value = {"name": "operations/123"}

    result = handler._convert_response(
        base_config, base_request, "req-123", mock_operation
    )

    assert result.request_id == "req-123"
    assert result.video == "https://example.com/video.mp4"
    assert result.content_type == "video/mp4"
    assert result.status == "completed"
    assert result.raw_response == {"name": "operations/123"}


def test_convert_response_with_video_bytes(handler, base_config, base_request):
    """Test conversion with video bytes instead of URI."""
    mock_video = MagicMock()
    mock_video.uri = None
    mock_video.video_bytes = b"fake-video-bytes"
    mock_video.mime_type = "video/mp4"

    mock_generated_video = MagicMock()
    mock_generated_video.video = mock_video

    mock_response = MagicMock()
    mock_response.generated_videos = [mock_generated_video]

    mock_operation = MagicMock()
    mock_operation.name = "operations/456"
    mock_operation.done = True
    mock_operation.error = None
    mock_operation.response = mock_response
    mock_operation.model_dump.return_value = {"name": "operations/456"}

    result = handler._convert_response(
        base_config, base_request, "req-456", mock_operation
    )

    assert result.request_id == "req-456"
    assert result.video == {"content": b"fake-video-bytes", "content_type": "video/mp4"}


def test_convert_response_with_incomplete_operation_raises_error(
    handler, base_config, base_request
):
    """Test incomplete operation raises ProviderAPIError."""
    mock_operation = MagicMock()
    mock_operation.done = False
    mock_operation.model_dump.return_value = {"done": False}

    with pytest.raises(ProviderAPIError, match="Operation is not completed"):
        handler._convert_response(base_config, base_request, "req-789", mock_operation)


def test_convert_response_with_operation_error_raises_error(
    handler, base_config, base_request
):
    """Test operation with error raises VideoGenerationError."""
    mock_operation = MagicMock()
    mock_operation.done = True
    mock_operation.error = "Something went wrong"
    mock_operation.model_dump.return_value = {"error": "Something went wrong"}

    with pytest.raises(VideoGenerationError, match="Video generation failed"):
        handler._convert_response(base_config, base_request, "req-999", mock_operation)


def test_convert_response_with_no_videos_raises_error(
    handler, base_config, base_request
):
    """Test operation with no generated videos raises ProviderAPIError."""
    mock_response = MagicMock()
    mock_response.generated_videos = []

    mock_operation = MagicMock()
    mock_operation.done = True
    mock_operation.error = None
    mock_operation.response = mock_response

    with pytest.raises(ProviderAPIError, match="No generated videos"):
        handler._convert_response(base_config, base_request, "req-000", mock_operation)


# ==================== Error Handling Tests ====================


def test_handle_error_with_video_generation_error(handler, base_config, base_request):
    """Test VideoGenerationError is returned as-is."""
    error = VideoGenerationError("Test error", provider="veo3", model="test-model")
    result = handler._handle_error(base_config, base_request, "req-1", error)

    assert result is error


def test_handle_error_with_unknown_exception(handler, base_config, base_request):
    """Test unknown exception is converted to VideoGenerationError."""
    unknown_error = ValueError("Something went wrong")

    result = handler._handle_error(base_config, base_request, "req-3", unknown_error)

    assert isinstance(result, VideoGenerationError)
    assert "Error while generating video" in result.message
    assert result.provider == "veo3"
    assert result.raw_response["error_type"] == "ValueError"


# ==================== Status Parsing Tests ====================


def test_parse_veo3_operation_queued():
    """Test parsing operation with QUEUED state."""
    mock_operation = MagicMock()
    mock_operation.name = "operations/123"
    mock_operation.metadata = {"state": "QUEUED"}

    result = parse_veo3_operation(mock_operation)

    assert result.request_id == "operations/123"
    assert result.status == "queued"
    assert result.update["metadata"] == {"state": "QUEUED"}


def test_parse_veo3_operation_processing():
    """Test parsing operation with PROCESSING state."""
    mock_operation = MagicMock()
    mock_operation.name = "operations/456"
    mock_operation.metadata = {"state": "PROCESSING"}

    result = parse_veo3_operation(mock_operation)

    assert result.request_id == "operations/456"
    assert result.status == "processing"


def test_parse_veo3_operation_with_no_metadata():
    """Test parsing operation with no metadata defaults to processing."""
    mock_operation = MagicMock()
    mock_operation.name = "operations/789"
    mock_operation.metadata = None

    result = parse_veo3_operation(mock_operation)

    assert result.request_id == "operations/789"
    assert result.status == "processing"
    # metadata is None in the operation, will be returned as-is
    assert result.update["metadata"] is None


# ==================== Async Video Generation Tests ====================


@pytest.mark.asyncio
async def test_generate_video_async_success_with_progress_callbacks(
    handler, base_config, base_request
):
    """Test successful async generation with sync and async progress callbacks."""
    # Create mock operation
    mock_operation = MagicMock()
    mock_operation.name = "operations/async-123"
    mock_operation.done = False
    mock_operation.metadata = {"state": "PROCESSING"}

    # Create completed operation
    mock_video = MagicMock()
    mock_video.uri = "https://example.com/async-video.mp4"
    mock_video.video_bytes = None
    mock_video.mime_type = "video/mp4"

    mock_generated_video = MagicMock()
    mock_generated_video.video = mock_video

    mock_response = MagicMock()
    mock_response.generated_videos = [mock_generated_video]

    mock_completed_operation = MagicMock()
    mock_completed_operation.name = "operations/async-123"
    mock_completed_operation.done = True
    mock_completed_operation.error = None
    mock_completed_operation.response = mock_response
    mock_completed_operation.model_dump.return_value = {"name": "operations/async-123"}

    # Setup mock async client
    mock_async_client = AsyncMock()
    mock_async_client.models.generate_videos = AsyncMock(return_value=mock_operation)
    mock_async_client.operations.get = AsyncMock(return_value=mock_completed_operation)

    # Clear cache and patch
    handler._async_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.video.providers.veo3.Client"
    ) as mock_client_class:
        mock_instance = MagicMock()
        mock_instance.aio = mock_async_client
        mock_client_class.return_value = mock_instance

        # Test with sync callback
        progress_calls = []

        def sync_callback(update):
            progress_calls.append(update)

        result = await handler.generate_video_async(
            base_config, base_request, on_progress=sync_callback
        )

        assert result.request_id == "operations/async-123"
        assert result.video == "https://example.com/async-video.mp4"
        assert len(progress_calls) >= 1  # At least one progress update

        # Test with async callback
        async_progress_calls = []

        async def async_callback(update):
            async_progress_calls.append(update)

        # Reset operation state
        handler._async_client_cache.clear()
        mock_operation.done = False

        with patch(
            "tarash.tarash_gateway.video.providers.veo3.Client"
        ) as mock_client_class:
            mock_instance = MagicMock()
            mock_instance.aio = mock_async_client
            mock_client_class.return_value = mock_instance

            await handler.generate_video_async(
                base_config, base_request, on_progress=async_callback
            )

        assert len(async_progress_calls) >= 1


@pytest.mark.asyncio
async def test_generate_video_async_propagates_known_errors(
    handler, base_config, base_request
):
    """Test that ValidationError and ProviderAPIError propagate without wrapping."""
    # Test ValidationError propagation
    request_invalid = VideoGenerationRequest(
        prompt="test", extra_params={"person_generation": "invalid"}
    )

    with pytest.raises(ValidationError):
        await handler.generate_video_async(base_config, request_invalid)


@pytest.mark.asyncio
async def test_generate_video_async_handles_timeout(handler, base_config, base_request):
    """Test async generation handles timeout after max poll attempts."""
    mock_operation = MagicMock()
    mock_operation.name = "operations/timeout"
    mock_operation.done = False
    mock_operation.metadata = {"state": "PROCESSING"}

    mock_async_client = AsyncMock()
    mock_async_client.models.generate_videos = AsyncMock(return_value=mock_operation)
    mock_async_client.operations.get = AsyncMock(return_value=mock_operation)

    # Set low max_poll_attempts for faster test
    timeout_config = VideoGenerationConfig(
        model="veo-3.0-flash-001",
        provider="veo3",
        api_key="test-api-key",
        max_poll_attempts=2,
        poll_interval=1,
    )

    handler._async_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.video.providers.veo3.Client"
    ) as mock_client_class:
        mock_instance = MagicMock()
        mock_instance.aio = mock_async_client
        mock_client_class.return_value = mock_instance

        with pytest.raises(VideoGenerationError, match="timed out"):
            await handler.generate_video_async(timeout_config, base_request)


@pytest.mark.asyncio
async def test_generate_video_async_wraps_unknown_exceptions(
    handler, base_config, base_request
):
    """Test unknown exceptions are wrapped by decorator."""
    mock_async_client = AsyncMock()
    mock_async_client.models.generate_videos = AsyncMock(
        side_effect=RuntimeError("Unexpected error")
    )

    handler._async_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.video.providers.veo3.Client"
    ) as mock_client_class:
        mock_instance = MagicMock()
        mock_instance.aio = mock_async_client
        mock_client_class.return_value = mock_instance

        with pytest.raises(VideoGenerationError, match="Unknown error"):
            await handler.generate_video_async(base_config, base_request)


# ==================== Sync Video Generation Tests ====================


def test_generate_video_success_with_progress_callback(
    handler, base_config, base_request
):
    """Test successful sync generation with progress callback."""
    # Create mock operation
    mock_operation = MagicMock()
    mock_operation.name = "operations/sync-456"
    mock_operation.done = False
    mock_operation.metadata = {"state": "QUEUED"}

    # Create completed operation
    mock_video = MagicMock()
    mock_video.uri = "https://example.com/sync-video.mp4"
    mock_video.video_bytes = None
    mock_video.mime_type = "video/mp4"

    mock_generated_video = MagicMock()
    mock_generated_video.video = mock_video

    mock_response = MagicMock()
    mock_response.generated_videos = [mock_generated_video]

    mock_completed_operation = MagicMock()
    mock_completed_operation.name = "operations/sync-456"
    mock_completed_operation.done = True
    mock_completed_operation.error = None
    mock_completed_operation.response = mock_response
    mock_completed_operation.model_dump.return_value = {"name": "operations/sync-456"}

    mock_sync_client = MagicMock()
    mock_sync_client.models.generate_videos.return_value = mock_operation
    mock_sync_client.operations.get.return_value = mock_completed_operation

    progress_calls = []

    def progress_callback(update):
        progress_calls.append(update)

    # Clear cache and patch
    handler._sync_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.video.providers.veo3.Client",
        return_value=mock_sync_client,
    ):
        result = handler.generate_video(
            base_config, base_request, on_progress=progress_callback
        )

    assert result.request_id == "operations/sync-456"
    assert result.video == "https://example.com/sync-video.mp4"
    assert len(progress_calls) >= 1


def test_generate_video_handles_exceptions(handler, base_config, base_request):
    """Test exception handling in sync generation."""
    mock_sync_client = MagicMock()
    mock_sync_client.models.generate_videos.side_effect = RuntimeError("Server error")

    handler._sync_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.video.providers.veo3.Client",
        return_value=mock_sync_client,
    ):
        with pytest.raises(VideoGenerationError, match="Unknown error"):
            handler.generate_video(base_config, base_request)


def test_generate_video_handles_timeout(handler, base_config, base_request):
    """Test sync generation handles timeout after max poll attempts."""
    mock_operation = MagicMock()
    mock_operation.name = "operations/timeout-sync"
    mock_operation.done = False

    mock_sync_client = MagicMock()
    mock_sync_client.models.generate_videos.return_value = mock_operation
    mock_sync_client.operations.get.return_value = mock_operation

    timeout_config = VideoGenerationConfig(
        model="veo-3.0-flash-001",
        provider="veo3",
        api_key="test-api-key",
        max_poll_attempts=2,
        poll_interval=1,
    )

    handler._sync_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.video.providers.veo3.Client",
        return_value=mock_sync_client,
    ):
        with pytest.raises(VideoGenerationError, match="timed out"):
            handler.generate_video(timeout_config, base_request)


# ==================== Error Decorator Tests ====================


@pytest.mark.asyncio
async def test_handle_video_generation_errors_async_propagates_known_errors():
    """Test decorator propagates ValidationError, ProviderAPIError, VideoGenerationError."""

    @handle_video_generation_errors
    async def async_func(self, config, request):
        if request.prompt == "validation":
            raise ValidationError("Invalid", provider="veo3")
        elif request.prompt == "provider":
            raise ProviderAPIError("API error", provider="veo3")
        elif request.prompt == "video":
            raise VideoGenerationError("Gen error", provider="veo3")
        else:
            raise RuntimeError("Unknown")

    config = VideoGenerationConfig(model="test-model", provider="veo3", api_key="key")

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

    assert exc_info.value.provider == "veo3"
    assert exc_info.value.model == "test-model"
    assert "error" in exc_info.value.raw_response


def test_handle_video_generation_errors_sync_propagates_known_errors():
    """Test decorator propagates known errors for sync functions."""

    @handle_video_generation_errors
    def sync_func(self, config, request):
        if request.prompt == "validation":
            raise ValidationError("Invalid", provider="veo3")
        elif request.prompt == "unknown":
            raise RuntimeError("Unknown")

    config = VideoGenerationConfig(model="test-model", provider="veo3", api_key="key")

    # Test ValidationError propagates
    with pytest.raises(ValidationError):
        sync_func(None, config, VideoGenerationRequest(prompt="validation"))

    # Test unknown exception is wrapped
    with pytest.raises(VideoGenerationError, match="Unknown error"):
        sync_func(None, config, VideoGenerationRequest(prompt="unknown"))
