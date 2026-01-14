"""Tests for GoogleProviderHandler video generation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tarash.tarash_gateway.exceptions import (
    GenerationFailedError,
    HTTPConnectionError,
    HTTPError,
    TarashException,
    TimeoutError,
    ValidationError,
    handle_video_generation_errors,
)
from tarash.tarash_gateway.models import (
    VideoGenerationConfig,
    VideoGenerationRequest,
)
from tarash.tarash_gateway.providers.google import (
    GoogleProviderHandler,
    parse_veo3_operation,
)


# ==================== Fixtures ====================


@pytest.fixture
def mock_sync_client():
    """Patch Client and provide mock."""
    mock = MagicMock()
    with patch("tarash.tarash_gateway.providers.google.Client", return_value=mock):
        yield mock


@pytest.fixture
def mock_async_client():
    """Patch Client for async and provide mock.aio."""
    mock = MagicMock()
    mock.aio = AsyncMock()
    with patch("tarash.tarash_gateway.providers.google.Client", return_value=mock):
        yield mock.aio


@pytest.fixture
def handler():
    """Create a GoogleProviderHandler instance."""
    with patch("tarash.tarash_gateway.providers.google.has_genai", True):
        return GoogleProviderHandler()


@pytest.fixture
def base_config():
    """Create a base VideoGenerationConfig."""
    return VideoGenerationConfig(
        model="veo-3.0-flash-001",
        provider="google",
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
        provider="google",
        api_key=api_key,
        base_url=base_url,
        timeout=600,
    )
    config2 = VideoGenerationConfig(
        model="veo-3.0-flash-001",
        provider="google",
        api_key="different-key",
        base_url="https://different.example.com",
        timeout=600,
    )

    with patch(
        "tarash.tarash_gateway.providers.google.Client",
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


def test_validate_params_with_valid_google_params(handler, base_config):
    """Test validation with valid Veo3VideoParams (Google video)."""
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


def test_convert_request_with_interpolation_mode(handler, base_config):
    """Test conversion with interpolation mode (first_frame + last_frame)."""
    request = VideoGenerationRequest(
        prompt="test",
        image_list=[
            {"image": "https://example.com/first.jpg", "type": "first_frame"},
            {"image": "https://example.com/last.jpg", "type": "last_frame"},
        ],
    )

    result = handler._convert_request(base_config, request)

    # first_frame should be in top-level 'image'
    assert result["image"] == {"gcs_uri": "https://example.com/first.jpg"}

    # last_frame should be in config
    config = result["config"]
    assert config.last_frame.gcs_uri == "https://example.com/last.jpg"
    # No reference images in interpolation mode
    assert config.reference_images is None


def test_convert_request_with_reference_images_mode(handler, base_config):
    """Test conversion with reference images mode (asset + style)."""
    request = VideoGenerationRequest(
        prompt="test",
        image_list=[
            {"image": "https://example.com/asset.jpg", "type": "asset"},
            {"image": "https://example.com/style.jpg", "type": "style"},
        ],
    )

    result = handler._convert_request(base_config, request)

    # No first_frame in reference images mode
    assert result["image"] is None

    # asset and style should be in reference_images
    config = result["config"]
    assert len(config.reference_images) == 2
    # No last_frame in reference images mode
    assert config.last_frame is None


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


def test_convert_request_rejects_last_frame_without_first_frame(handler, base_config):
    """Test that last_frame requires first_frame."""
    request = VideoGenerationRequest(
        prompt="test",
        image_list=[
            {
                "image": {"content": b"fake-bytes", "content_type": "image/jpeg"},
                "type": "last_frame",
            }
        ],
    )

    with pytest.raises(ValidationError) as exc_info:
        handler._convert_request(base_config, request)

    assert "first_frame" in str(exc_info.value).lower()
    assert "interpolation" in str(exc_info.value).lower()


def test_convert_request_rejects_first_frame_with_reference_images(
    handler, base_config
):
    """Test that first_frame and reference images are mutually exclusive."""
    request = VideoGenerationRequest(
        prompt="test",
        image_list=[
            {
                "image": {"content": b"fake-bytes-1", "content_type": "image/jpeg"},
                "type": "first_frame",
            },
            {
                "image": {"content": b"fake-bytes-2", "content_type": "image/jpeg"},
                "type": "asset",
            },
        ],
    )

    with pytest.raises(ValidationError) as exc_info:
        handler._convert_request(base_config, request)

    assert "reference images" in str(exc_info.value).lower()
    assert "first_frame" in str(exc_info.value).lower()


def test_convert_request_rejects_more_than_3_reference_images(handler, base_config):
    """Test that maximum 3 reference images are allowed."""
    request = VideoGenerationRequest(
        prompt="test",
        image_list=[
            {
                "image": {"content": b"fake-bytes-1", "content_type": "image/jpeg"},
                "type": "asset",
            },
            {
                "image": {"content": b"fake-bytes-2", "content_type": "image/jpeg"},
                "type": "asset",
            },
            {
                "image": {"content": b"fake-bytes-3", "content_type": "image/jpeg"},
                "type": "style",
            },
            {
                "image": {"content": b"fake-bytes-4", "content_type": "image/jpeg"},
                "type": "asset",
            },
        ],
    )

    with pytest.raises(ValidationError) as exc_info:
        handler._convert_request(base_config, request)

    assert "3" in str(exc_info.value)
    assert "reference images" in str(exc_info.value).lower()


def test_convert_request_accepts_first_frame_with_last_frame(handler, base_config):
    """Test that first_frame + last_frame is valid (interpolation mode)."""
    request = VideoGenerationRequest(
        prompt="test",
        image_list=[
            {
                "image": {"content": b"fake-bytes-1", "content_type": "image/jpeg"},
                "type": "first_frame",
            },
            {
                "image": {"content": b"fake-bytes-2", "content_type": "image/jpeg"},
                "type": "last_frame",
            },
        ],
    )

    # Should not raise
    result = handler._convert_request(base_config, request)
    assert result["image"]["image_bytes"] == b"fake-bytes-1"
    assert result["config"].last_frame.image_bytes == b"fake-bytes-2"


def test_convert_request_accepts_up_to_3_reference_images(handler, base_config):
    """Test that up to 3 reference images are allowed."""
    request = VideoGenerationRequest(
        prompt="test",
        image_list=[
            {
                "image": {"content": b"fake-bytes-1", "content_type": "image/jpeg"},
                "type": "asset",
            },
            {
                "image": {"content": b"fake-bytes-2", "content_type": "image/jpeg"},
                "type": "style",
            },
            {
                "image": {"content": b"fake-bytes-3", "content_type": "image/jpeg"},
                "type": "asset",
            },
        ],
    )

    # Should not raise
    result = handler._convert_request(base_config, request)
    assert len(result["config"].reference_images) == 3


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
    """Test incomplete operation raises GenerationFailedError."""
    mock_operation = MagicMock()
    mock_operation.done = False
    mock_operation.model_dump.return_value = {"done": False}

    with pytest.raises(GenerationFailedError, match="Operation is not completed"):
        handler._convert_response(base_config, base_request, "req-789", mock_operation)


def test_convert_response_with_operation_error_raises_error(
    handler, base_config, base_request
):
    """Test operation with error raises TarashException."""
    mock_operation = MagicMock()
    mock_operation.done = True
    mock_operation.error = "Something went wrong"
    mock_operation.model_dump.return_value = {"error": "Something went wrong"}

    with pytest.raises(TarashException, match="Video generation failed"):
        handler._convert_response(base_config, base_request, "req-999", mock_operation)


def test_convert_response_with_no_videos_raises_error(
    handler, base_config, base_request
):
    """Test operation with no generated videos raises GenerationFailedError."""
    mock_response = MagicMock()
    mock_response.generated_videos = []

    mock_operation = MagicMock()
    mock_operation.done = True
    mock_operation.error = None
    mock_operation.response = mock_response

    with pytest.raises(GenerationFailedError, match="No generated videos"):
        handler._convert_response(base_config, base_request, "req-000", mock_operation)


# ==================== Error Handling Tests ====================


def test_handle_error_with_video_generation_error(handler, base_config, base_request):
    """Test TarashException is returned as-is."""
    error = TarashException("Test error", provider="google", model="test-model")
    result = handler._handle_error(base_config, base_request, "req-1", error)

    assert result is error


def test_handle_error_with_unknown_exception(handler, base_config, base_request):
    """Test unknown exception is converted to TarashException."""
    unknown_error = ValueError("Something went wrong")

    result = handler._handle_error(base_config, base_request, "req-3", unknown_error)

    assert isinstance(result, GenerationFailedError)
    assert "Error while generating video" in result.message
    assert result.provider == "google"
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
    with patch("tarash.tarash_gateway.providers.google.Client") as mock_client_class:
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
            "tarash.tarash_gateway.providers.google.Client"
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
    """Test that ValidationError and GenerationFailedError propagate without wrapping."""
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
        provider="google",
        api_key="test-api-key",
        max_poll_attempts=2,
        poll_interval=1,
    )

    handler._async_client_cache.clear()
    with patch("tarash.tarash_gateway.providers.google.Client") as mock_client_class:
        mock_instance = MagicMock()
        mock_instance.aio = mock_async_client
        mock_client_class.return_value = mock_instance

        with pytest.raises(TarashException, match="timed out"):
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
    with patch("tarash.tarash_gateway.providers.google.Client") as mock_client_class:
        mock_instance = MagicMock()
        mock_instance.aio = mock_async_client
        mock_client_class.return_value = mock_instance

        with pytest.raises(GenerationFailedError, match="Error while generating video"):
            await handler.generate_video_async(base_config, base_request)


@pytest.mark.asyncio
async def test_generate_video_async_handles_400_client_error(
    handler, base_config, base_request
):
    """Test async generation handles Google API 400 error and converts to ValidationError."""

    # Create a mock ClientError class
    class MockClientError(Exception):
        def __init__(self, code, message, status):
            super().__init__(message)
            self.code = code
            self.message = message
            self.details = {"code": code, "message": message, "status": status}
            self.status = status

    mock_error = MockClientError(
        400,
        "dont_allow for personGeneration is currently not supported.",
        "INVALID_ARGUMENT",
    )

    mock_async_client = AsyncMock()
    mock_async_client.models.generate_videos = AsyncMock(side_effect=mock_error)

    handler._async_client_cache.clear()
    with patch("tarash.tarash_gateway.providers.google.Client") as mock_client_class:
        mock_instance = MagicMock()
        mock_instance.aio = mock_async_client
        mock_client_class.return_value = mock_instance

        with patch(
            "tarash.tarash_gateway.providers.google.ClientError", MockClientError
        ):
            with pytest.raises(ValidationError) as exc_info:
                await handler.generate_video_async(base_config, base_request)

            assert "dont_allow for personGeneration is currently not supported" in str(
                exc_info.value
            )
            assert exc_info.value.provider == "google"
            assert exc_info.value.raw_response["status_code"] == 400
            assert exc_info.value.raw_response["error_type"] == "INVALID_ARGUMENT"


@pytest.mark.asyncio
async def test_generate_video_async_handles_500_client_error(
    handler, base_config, base_request
):
    """Test async generation handles Google API 500 error and converts to HTTPError."""

    # Create a mock ClientError class
    class MockClientError(Exception):
        def __init__(self, code, message, status):
            super().__init__(message)
            self.code = code
            self.message = message
            self.details = {"code": code, "message": message, "status": status}
            self.status = status

    mock_error = MockClientError(500, "Internal server error", "INTERNAL_ERROR")

    mock_async_client = AsyncMock()
    mock_async_client.models.generate_videos = AsyncMock(side_effect=mock_error)

    handler._async_client_cache.clear()
    with patch("tarash.tarash_gateway.providers.google.Client") as mock_client_class:
        mock_instance = MagicMock()
        mock_instance.aio = mock_async_client
        mock_client_class.return_value = mock_instance

        with patch(
            "tarash.tarash_gateway.providers.google.ClientError", MockClientError
        ):
            with pytest.raises(HTTPError) as exc_info:
                await handler.generate_video_async(base_config, base_request)

            assert "Internal server error" in str(exc_info.value)
            assert exc_info.value.provider == "google"
            assert exc_info.value.raw_response["status_code"] == 500
            assert exc_info.value.raw_response["error_type"] == "INTERNAL_ERROR"


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
        "tarash.tarash_gateway.providers.google.Client",
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
        "tarash.tarash_gateway.providers.google.Client",
        return_value=mock_sync_client,
    ):
        with pytest.raises(GenerationFailedError, match="Error while generating video"):
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
        provider="google",
        api_key="test-api-key",
        max_poll_attempts=2,
        poll_interval=1,
    )

    handler._sync_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.providers.google.Client",
        return_value=mock_sync_client,
    ):
        with pytest.raises(TarashException, match="timed out"):
            handler.generate_video(timeout_config, base_request)


def test_generate_video_handles_400_client_error(handler, base_config, base_request):
    """Test sync generation handles Google API 400 error and converts to ValidationError."""

    # Create a mock ClientError class
    class MockClientError(Exception):
        def __init__(self, code, message, status):
            super().__init__(message)
            self.code = code
            self.message = message
            self.details = {"code": code, "message": message, "status": status}
            self.status = status

    mock_error = MockClientError(
        400,
        "dont_allow for personGeneration is currently not supported.",
        "INVALID_ARGUMENT",
    )

    mock_sync_client = MagicMock()
    mock_sync_client.models.generate_videos.side_effect = mock_error

    handler._sync_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.providers.google.Client",
        return_value=mock_sync_client,
    ):
        with patch(
            "tarash.tarash_gateway.providers.google.ClientError", MockClientError
        ):
            with pytest.raises(ValidationError) as exc_info:
                handler.generate_video(base_config, base_request)

            assert "dont_allow for personGeneration is currently not supported" in str(
                exc_info.value
            )
            assert exc_info.value.provider == "google"
            assert exc_info.value.raw_response["status_code"] == 400
            assert exc_info.value.raw_response["error_type"] == "INVALID_ARGUMENT"


def test_generate_video_handles_500_client_error(handler, base_config, base_request):
    """Test sync generation handles Google API 500 error and converts to HTTPError."""

    # Create a mock ClientError class
    class MockClientError(Exception):
        def __init__(self, code, message, status):
            super().__init__(message)
            self.code = code
            self.message = message
            self.details = {"code": code, "message": message, "status": status}
            self.status = status

    mock_error = MockClientError(500, "Internal server error", "INTERNAL_ERROR")

    mock_sync_client = MagicMock()
    mock_sync_client.models.generate_videos.side_effect = mock_error

    handler._sync_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.providers.google.Client",
        return_value=mock_sync_client,
    ):
        with patch(
            "tarash.tarash_gateway.providers.google.ClientError", MockClientError
        ):
            with pytest.raises(HTTPError) as exc_info:
                handler.generate_video(base_config, base_request)

            assert "Internal server error" in str(exc_info.value)
            assert exc_info.value.provider == "google"
            assert exc_info.value.raw_response["status_code"] == 500
            assert exc_info.value.raw_response["error_type"] == "INTERNAL_ERROR"


# ==================== Error Decorator Tests ====================


@pytest.mark.asyncio
async def test_handle_video_generation_errors_async_propagates_known_errors():
    """Test decorator propagates ValidationError, GenerationFailedError, TarashException."""

    @handle_video_generation_errors
    async def async_func(self, config, request, on_progress=None):
        if request.prompt == "validation":
            raise ValidationError("Invalid", provider="google")
        elif request.prompt == "provider":
            raise GenerationFailedError("API error", provider="google")
        elif request.prompt == "video":
            raise TarashException("Gen error", provider="google")
        else:
            raise RuntimeError("Unknown")

    config = VideoGenerationConfig(model="test-model", provider="google", api_key="key")

    # Test ValidationError propagates
    with pytest.raises(ValidationError):
        await async_func(None, config, VideoGenerationRequest(prompt="validation"))

    # Test GenerationFailedError propagates
    with pytest.raises(GenerationFailedError):
        await async_func(None, config, VideoGenerationRequest(prompt="provider"))

    # Test TarashException propagates
    with pytest.raises(TarashException):
        await async_func(None, config, VideoGenerationRequest(prompt="video"))

    # Test unknown exception is wrapped
    with pytest.raises(TarashException, match="Unknown error") as exc_info:
        await async_func(None, config, VideoGenerationRequest(prompt="unknown"))

    assert exc_info.value.provider == "google"
    assert exc_info.value.model == "test-model"
    assert "error" in exc_info.value.raw_response


def test_handle_video_generation_errors_sync_propagates_known_errors():
    """Test decorator propagates known errors for sync functions."""

    @handle_video_generation_errors
    def sync_func(self, config, request, on_progress=None):
        if request.prompt == "validation":
            raise ValidationError("Invalid", provider="google")
        elif request.prompt == "unknown":
            raise RuntimeError("Unknown")

    config = VideoGenerationConfig(model="test-model", provider="google", api_key="key")

    # Test ValidationError propagates
    with pytest.raises(ValidationError):
        sync_func(None, config, VideoGenerationRequest(prompt="validation"))

    # Test unknown exception is wrapped
    with pytest.raises(TarashException, match="Unknown error"):
        sync_func(None, config, VideoGenerationRequest(prompt="unknown"))


# ==================== Additional Coverage Tests ====================


def test_convert_to_image_with_unknown_format():
    """Test _convert_to_image with unknown format returns empty dict."""
    from tarash.tarash_gateway.providers.google import _convert_to_image

    result = _convert_to_image({"invalid": "format"})
    assert result == {}


def test_convert_to_video_with_unknown_format():
    """Test _convert_to_video with unknown format returns empty dict."""
    from tarash.tarash_gateway.providers.google import _convert_to_video

    result = _convert_to_video({"invalid": "format"})
    assert result == {}


def test_convert_request_with_style_only(handler, base_config):
    """Test _convert_request with only style image (lines 358-370)."""
    request = VideoGenerationRequest(
        prompt="test",
        image_list=[
            {"image": "https://example.com/style.jpg", "type": "style"},
        ],
    )

    result = handler._convert_request(base_config, request)

    # Should have reference_images with STYLE type
    config = result["config"]
    assert config.reference_images is not None
    assert len(config.reference_images) == 1
    # reference_type is an enum attribute, access it directly
    assert config.reference_images[0].reference_type.value == "STYLE"


def test_convert_image_response_with_gcs_uri(handler):
    """Test convert_image_response with gcs_uri field."""
    from tarash.tarash_gateway.models import (
        ImageGenerationConfig,
        ImageGenerationRequest,
    )

    config = ImageGenerationConfig(
        model="imagen-3.0-generate-001",
        provider="google",
        api_key="test-key",
    )
    request = ImageGenerationRequest(prompt="test")

    # Mock generated image with gcs_uri
    mock_gen_img = MagicMock()
    mock_gen_img.image = MagicMock()
    mock_gen_img.image.gcs_uri = "https://storage.googleapis.com/bucket/image.png"

    genai_response = {"generated_images": [mock_gen_img]}

    response = handler._convert_image_response(
        config, request, "req-123", genai_response
    )

    assert response.request_id == "req-123"
    assert len(response.images) == 1
    assert response.images[0] == "https://storage.googleapis.com/bucket/image.png"


def test_get_client_with_vertex_url_parsing():
    """Test _get_client parses location from Vertex AI base_url."""
    from tarash.tarash_gateway.providers.google import GoogleProviderHandler

    # Need a base_url that contains "vertex" to trigger the Vertex AI path
    # and "-aiplatform" to parse the location
    # Using "https://{location}-aiplatform.googleapis.com" format with vertex in a different part
    config = VideoGenerationConfig(
        model="veo-3.0-generate-001",
        provider="google",
        api_key="test-key",
        base_url="https://us-central1-aiplatform.googleapis.com/vertex",
    )

    mock_client = MagicMock()
    mock_client.aio = AsyncMock()

    # Patch has_genai BEFORE importing Client, and patch Client at the right location
    with patch("tarash.tarash_gateway.providers.google.has_genai", True):
        handler = GoogleProviderHandler()
        handler._async_client_cache.clear()

        with patch(
            "tarash.tarash_gateway.providers.google.Client", return_value=mock_client
        ) as mock_client_cls:
            handler._get_client(config, "async")
            # Verify Client was called with vertexai=True and location="us-central1"
            assert mock_client_cls.call_count == 1
            # Check that Client was called with the right arguments
            mock_client_cls.assert_called_once()
            call_kwargs = mock_client_cls.call_args.kwargs
            assert call_kwargs.get("vertexai") is True
            assert call_kwargs.get("location") == "us-central1"


def test_handle_error_with_aiohttp_timeout():
    """Test _handle_error with aiohttp timeout error."""
    from tarash.tarash_gateway.providers.google import has_aiohttp

    # Only test if aiohttp is available
    if not has_aiohttp:
        pytest.skip("aiohttp not available")

    # Import actual aiohttp to get real exception class
    try:
        from aiohttp import ClientTimeout
    except ImportError:
        pytest.skip("aiohttp ClientTimeout not available")

    config = VideoGenerationConfig(
        model="veo-3.0-generate-001", provider="google", api_key="test-key"
    )

    with patch("tarash.tarash_gateway.providers.google.has_genai", True):
        handler = GoogleProviderHandler()
        result = handler._handle_error(
            config,
            VideoGenerationRequest(),
            "req-1",
            ClientTimeout("Request timed out"),
        )

    assert isinstance(result, TimeoutError)


def test_handle_error_with_aiohttp_connection_error():
    """Test _handle_error with aiohttp connection error."""
    from tarash.tarash_gateway.providers.google import has_aiohttp

    # Only test if aiohttp is available
    if not has_aiohttp:
        pytest.skip("aiohttp not available")

    # Import actual aiohttp to get real exception class
    try:
        from aiohttp import ClientConnectorError
    except ImportError:
        pytest.skip("aiohttp ClientConnectorError not available")

    config = VideoGenerationConfig(
        model="veo-3.0-generate-001", provider="google", api_key="test-key"
    )

    with patch("tarash.tarash_gateway.providers.google.has_genai", True):
        handler = GoogleProviderHandler()
        # Create a real ClientConnectorError with a real connection error
        try:
            raise ClientConnectorError(None, None)
        except ClientConnectorError as ex:
            result = handler._handle_error(
                config, VideoGenerationRequest(), "req-1", ex
            )

    assert isinstance(result, HTTPConnectionError)


def test_convert_response_with_no_response_raises_error(
    handler, base_config, base_request
):
    """Test operation with None response raises GenerationFailedError (line 428)."""
    mock_operation = MagicMock()
    mock_operation.done = True
    mock_operation.error = None
    mock_operation.response = None  # No response
    mock_operation.model_dump.return_value = {"done": True}

    with pytest.raises(
        GenerationFailedError, match="No response in completed operation"
    ):
        handler._convert_response(base_config, base_request, "req-428", mock_operation)


def test_convert_response_with_none_video_obj_raises_error(
    handler, base_config, base_request
):
    """Test operation with None video_obj raises GenerationFailedError (line 444)."""
    mock_response = MagicMock()
    mock_generated_video = MagicMock()
    mock_generated_video.video = None  # No video object
    mock_response.generated_videos = [mock_generated_video]

    mock_operation = MagicMock()
    mock_operation.done = True
    mock_operation.error = None
    mock_operation.response = mock_response
    mock_operation.model_dump.return_value = {"done": True}

    with pytest.raises(GenerationFailedError, match="Video object is None"):
        handler._convert_response(base_config, base_request, "req-444", mock_operation)


def test_convert_response_with_no_video_bytes_or_mime_raises_error(
    handler, base_config, base_request
):
    """Test operation with no URI and no video_bytes/mime_type raises error (line 459)."""
    mock_video = MagicMock()
    mock_video.uri = None  # No URI
    mock_video.video_bytes = None  # No video bytes
    mock_video.mime_type = None  # No mime type

    mock_generated_video = MagicMock()
    mock_generated_video.video = mock_video

    mock_response = MagicMock()
    mock_response.generated_videos = [mock_generated_video]

    mock_operation = MagicMock()
    mock_operation.done = True
    mock_operation.error = None
    mock_operation.response = mock_response
    mock_operation.model_dump.return_value = {"done": True}

    with pytest.raises(GenerationFailedError, match="no URI and no video_bytes"):
        handler._convert_response(base_config, base_request, "req-459", mock_operation)


def test_convert_response_with_string_video_bytes(handler, base_config, base_request):
    """Test operation with string video_bytes gets encoded (line 466)."""
    mock_video = MagicMock()
    mock_video.uri = None
    mock_video.video_bytes = "fake-video-bytes-as-string"  # String instead of bytes
    mock_video.mime_type = "video/mp4"

    mock_generated_video = MagicMock()
    mock_generated_video.video = mock_video

    mock_response = MagicMock()
    mock_response.generated_videos = [mock_generated_video]

    mock_operation = MagicMock()
    mock_operation.done = True
    mock_operation.error = None
    mock_operation.response = mock_response
    mock_operation.model_dump.return_value = {"done": True}

    result = handler._convert_response(
        base_config, base_request, "req-466", mock_operation
    )

    assert result.video["content"] == b"fake-video-bytes-as-string"


def test_convert_request_with_multiple_last_frames_raises_error(handler, base_config):
    """Test more than 1 last_frame image raises ValidationError (line 313)."""
    request = VideoGenerationRequest(
        prompt="test",
        image_list=[
            {"image": "https://example.com/first.jpg", "type": "first_frame"},
            {"image": "https://example.com/last1.jpg", "type": "last_frame"},
            {"image": "https://example.com/last2.jpg", "type": "last_frame"},
        ],
    )

    with pytest.raises(ValidationError, match="only supports 1 last_frame"):
        handler._convert_request(base_config, request)


def test_convert_request_with_multiple_first_frames_raises_error(handler, base_config):
    """Test more than 1 first_frame image raises ValidationError (line 306)."""
    request = VideoGenerationRequest(
        prompt="test",
        image_list=[
            {"image": "https://example.com/first1.jpg", "type": "first_frame"},
            {"image": "https://example.com/first2.jpg", "type": "first_frame"},
        ],
    )

    with pytest.raises(ValidationError, match="only supports 1 reference/first_frame"):
        handler._convert_request(base_config, request)


def test_handle_error_with_httpx_timeout(handler, base_config):
    """Test _handle_error with httpx timeout error (line 495)."""
    import httpx

    config = VideoGenerationConfig(
        model="veo-3.0-generate-001", provider="google", api_key="test-key"
    )

    error = httpx.TimeoutException("Request timed out")

    with patch("tarash.tarash_gateway.providers.google.has_genai", True):
        handler = GoogleProviderHandler()
        result = handler._handle_error(
            config, VideoGenerationRequest(prompt="test"), "req-1", error
        )

    assert isinstance(result, TimeoutError)


def test_handle_error_with_httpx_connect_error(handler, base_config):
    """Test _handle_error with httpx connection error (line 506)."""
    import httpx

    config = VideoGenerationConfig(
        model="veo-3.0-generate-001", provider="google", api_key="test-key"
    )

    error = httpx.ConnectError("Connection failed")

    with patch("tarash.tarash_gateway.providers.google.has_genai", True):
        handler = GoogleProviderHandler()
        result = handler._handle_error(
            config, VideoGenerationRequest(prompt="test"), "req-1", error
        )

    assert isinstance(result, HTTPConnectionError)


def test_convert_image_response_with_image_but_no_gcs_uri(handler):
    """Test convert_image_response when image has no gcs_uri (lines 658-660)."""
    from tarash.tarash_gateway.models import (
        ImageGenerationConfig,
        ImageGenerationRequest,
    )

    config = ImageGenerationConfig(
        model="imagen-3.0-generate-001",
        provider="google",
        api_key="test-key",
    )
    request = ImageGenerationRequest(prompt="test")

    # Mock generated image with image attribute but no gcs_uri
    mock_gen_img = MagicMock()
    mock_gen_img.image = MagicMock()
    mock_gen_img.image.gcs_uri = None  # No gcs_uri

    genai_response = {"generated_images": [mock_gen_img]}

    response = handler._convert_image_response(
        config, request, "req-123", genai_response
    )

    assert response.request_id == "req-123"
    # Should have empty images list since gcs_uri is None
    assert len(response.images) == 0


def test_convert_image_response_with_no_image_attribute(handler):
    """Test convert_image_response when generated_image has no image attribute."""
    from tarash.tarash_gateway.models import (
        ImageGenerationConfig,
        ImageGenerationRequest,
    )

    config = ImageGenerationConfig(
        model="imagen-3.0-generate-001",
        provider="google",
        api_key="test-key",
    )
    request = ImageGenerationRequest(prompt="test")

    # Mock generated image without image attribute
    mock_gen_img = MagicMock()
    # Remove the image attribute entirely
    del mock_gen_img.image

    genai_response = {"generated_images": [mock_gen_img]}

    response = handler._convert_image_response(
        config, request, "req-123", genai_response
    )

    assert response.request_id == "req-123"
    # Should have empty images list
    assert len(response.images) == 0
