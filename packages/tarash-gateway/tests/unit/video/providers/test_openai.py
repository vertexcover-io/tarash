"""Tests for OpenAIProviderHandler."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tarash.tarash_gateway.exceptions import (
    GenerationFailedError,
    TimeoutError,
    ValidationError,
)
from tarash.tarash_gateway.models import (
    VideoGenerationConfig,
    VideoGenerationRequest,
)
from tarash.tarash_gateway.providers.openai import (
    OpenAIProviderHandler,
    parse_openai_video_status,
)


# ==================== Fixtures ====================


@pytest.fixture
def mock_sync_client():
    """Patch OpenAI and provide mock."""
    mock = MagicMock()
    with patch("tarash.tarash_gateway.providers.openai.OpenAI", return_value=mock):
        yield mock


@pytest.fixture
def mock_async_client():
    """Patch AsyncOpenAI and provide mock."""
    mock = AsyncMock()
    with patch("tarash.tarash_gateway.providers.openai.AsyncOpenAI", return_value=mock):
        yield mock


@pytest.fixture
def handler():
    """Create an OpenAIProviderHandler instance."""
    return OpenAIProviderHandler()


@pytest.fixture
def base_config():
    """Create a base VideoGenerationConfig."""
    return VideoGenerationConfig(
        model="sora-2",
        provider="openai",
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
    handler._sync_client_cache.clear()

    client1 = handler._get_client(base_config, "sync")
    client2 = handler._get_client(base_config, "sync")

    assert client1 is client2  # Same instance (cached)
    assert client1 is mock_sync_client


def test_get_client_creates_and_caches_async_client(
    handler, base_config, mock_async_client
):
    """Test async client creation and caching."""
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
        ("key1", "https://api1.openai.com/v1"),
        ("key1", "https://api2.openai.com/v1"),
    ],
)
def test_get_client_creates_different_clients_for_different_configs(
    handler, api_key, base_url
):
    """Test different clients for different API keys and base_urls."""
    handler._sync_client_cache.clear()

    mock_client1 = MagicMock()
    mock_client2 = MagicMock()

    config1 = VideoGenerationConfig(
        model="sora-2",
        provider="openai",
        api_key=api_key,
        base_url=base_url,
        timeout=600,
    )
    config2 = VideoGenerationConfig(
        model="sora-2",
        provider="openai",
        api_key="different-key",
        base_url="https://different.example.com",
        timeout=600,
    )

    with patch(
        "tarash.tarash_gateway.providers.openai.OpenAI",
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


def test_validate_params_with_valid_params(handler, base_config):
    """Test validation with valid OpenAIVideoParams."""
    request = VideoGenerationRequest(
        prompt="test",
        extra_params={},
    )

    result = handler._validate_params(base_config, request)
    assert result == {}


# ==================== Request Conversion Tests ====================


def test_convert_request_with_minimal_fields(handler, base_config):
    """Test conversion with only prompt."""
    request = VideoGenerationRequest(prompt="A test video")
    result = handler._convert_request(base_config, request)

    assert result["model"] == "sora-2"
    assert result["prompt"] == "A test video"


def test_convert_request_with_duration(handler, base_config):
    """Test conversion with duration."""
    request = VideoGenerationRequest(prompt="A test video", duration_seconds=8)
    result = handler._convert_request(base_config, request)

    assert result["seconds"] == "8"


def test_convert_request_with_valid_sora2_durations(handler, base_config):
    """Test that Sora 2 accepts valid durations: 4, 8, 12 seconds."""
    for duration in [4, 8, 12]:
        request = VideoGenerationRequest(
            prompt="A test video", duration_seconds=duration
        )
        result = handler._convert_request(base_config, request)
        assert result["seconds"] == str(duration)


def test_convert_request_with_valid_sora2_pro_durations(handler):
    """Test that Sora 2 Pro accepts valid durations: 10, 15, 25 seconds."""
    config = VideoGenerationConfig(
        provider="openai",
        model="sora-2-pro",
        api_key="test-key",
    )

    for duration in [10, 15, 25]:
        request = VideoGenerationRequest(
            prompt="A test video", duration_seconds=duration
        )
        result = handler._convert_request(config, request)
        assert result["seconds"] == str(duration)


def test_convert_request_with_invalid_sora2_duration(handler, base_config):
    """Test that Sora 2 rejects invalid durations."""
    request = VideoGenerationRequest(prompt="A test video", duration_seconds=5)

    with pytest.raises(ValidationError) as exc_info:
        handler._convert_request(base_config, request)

    assert "Invalid duration" in str(exc_info.value)
    assert "5 seconds" in str(exc_info.value)
    assert "4, 8, 12" in str(exc_info.value)


def test_convert_request_with_invalid_sora2_pro_duration(handler):
    """Test that Sora 2 Pro rejects invalid durations."""
    config = VideoGenerationConfig(
        provider="openai",
        model="sora-2-pro",
        api_key="test-key",
    )

    request = VideoGenerationRequest(prompt="A test video", duration_seconds=8)

    with pytest.raises(ValidationError) as exc_info:
        handler._convert_request(config, request)

    assert "Invalid duration" in str(exc_info.value)
    assert "8 seconds" in str(exc_info.value)
    assert "10, 15, 25" in str(exc_info.value)


def test_convert_request_with_aspect_ratio(handler, base_config):
    """Test conversion with aspect ratio to size mapping."""
    request = VideoGenerationRequest(prompt="A test video", aspect_ratio="16:9")
    result = handler._convert_request(base_config, request)

    assert result["size"] == "1280x720"

    # Test portrait
    request = VideoGenerationRequest(prompt="A test video", aspect_ratio="9:16")
    result = handler._convert_request(base_config, request)
    assert result["size"] == "720x1280"

    # Test square
    request = VideoGenerationRequest(prompt="A test video", aspect_ratio="1:1")
    result = handler._convert_request(base_config, request)
    assert result["size"] == "1024x1024"


def test_convert_request_with_all_optional_fields(handler, base_config):
    """Test conversion with all optional fields."""
    request = VideoGenerationRequest(
        prompt="A test video",
        duration_seconds=8,
        aspect_ratio="16:9",
    )

    result = handler._convert_request(base_config, request)

    assert result["model"] == "sora-2"
    assert result["prompt"] == "A test video"
    assert result["seconds"] == "8"
    assert result["size"] == "1280x720"


# ==================== Response Conversion Tests ====================


def test_convert_response_with_completed_video(handler, base_config, base_request):
    """Test conversion with completed video including URL."""
    mock_video = MagicMock()
    mock_video.id = "video-123"
    mock_video.status = "completed"
    mock_video.url = "https://example.com/video.mp4"
    mock_video.seconds = "8"
    mock_video.size = "1280x720"
    mock_video.model_dump.return_value = {
        "id": "video-123",
        "status": "completed",
        "url": "https://example.com/video.mp4",
    }

    # Create OpenAIVideoResponse dict with content
    response_data = {
        "video": mock_video,
        "content": b"fake video content",
        "content_type": "video/mp4",
    }

    result = handler._convert_response(
        base_config, base_request, "video-123", response_data
    )

    assert result.request_id == "video-123"
    assert result.video == {
        "content": b"fake video content",
        "content_type": "video/mp4",
    }
    assert result.content_type == "video/mp4"
    assert result.duration == 8.0
    assert result.resolution == "1280x720"
    assert result.status == "completed"


def test_convert_response_with_failed_video(handler, base_config, base_request):
    """Test conversion with failed video raises GenerationFailedError."""
    mock_error = MagicMock()
    mock_error.message = "Content policy violation"

    mock_video = MagicMock()
    mock_video.id = "video-456"
    mock_video.status = "failed"
    mock_video.error = mock_error
    mock_video.model_dump.return_value = {"id": "video-456", "status": "failed"}

    # Create OpenAIVideoResponse dict (content not needed for failed status)
    response_data = {
        "video": mock_video,
        "content": b"",
        "content_type": "video/mp4",
    }

    with pytest.raises(GenerationFailedError, match="Content policy violation"):
        handler._convert_response(base_config, base_request, "video-456", response_data)


def test_convert_response_with_incomplete_video_raises_error(
    handler, base_config, base_request
):
    """Test incomplete video raises GenerationFailedError."""
    mock_video = MagicMock()
    mock_video.id = "video-789"
    mock_video.status = "in_progress"
    mock_video.model_dump.return_value = {"id": "video-789", "status": "in_progress"}

    # Create OpenAIVideoResponse dict with empty content (incomplete video)
    response_data = {
        "video": mock_video,
        "content": b"",
        "content_type": "video/mp4",
    }

    with pytest.raises(GenerationFailedError, match="No video content found"):
        handler._convert_response(base_config, base_request, "video-789", response_data)


def test_convert_response_with_no_url_raises_error(handler, base_config, base_request):
    """Test missing video content raises GenerationFailedError."""
    mock_video = MagicMock()
    mock_video.id = "video-000"
    mock_video.status = "completed"
    mock_video.url = None
    mock_video.model_dump.return_value = {"id": "video-000", "status": "completed"}

    # Create OpenAIVideoResponse dict with no content
    response_data = {
        "video": mock_video,
        "content": None,
        "content_type": "video/mp4",
    }

    with pytest.raises(GenerationFailedError, match="No video content found"):
        handler._convert_response(base_config, base_request, "video-000", response_data)


# ==================== Error Handling Tests ====================


def test_handle_error_with_video_generation_error(handler, base_config, base_request):
    """Test GenerationFailedError is returned as-is."""
    error = GenerationFailedError("Test error", provider="openai", model="sora-2")
    result = handler._handle_error(base_config, base_request, "req-1", error)

    assert result is error


def test_handle_error_with_unknown_exception(handler, base_config, base_request):
    """Test unknown exception is converted to GenerationFailedError."""
    unknown_error = ValueError("Something went wrong")

    result = handler._handle_error(base_config, base_request, "req-3", unknown_error)

    assert isinstance(result, GenerationFailedError)
    assert "Error while generating video" in result.message
    assert result.provider == "openai"
    assert result.raw_response["error_type"] == "ValueError"


# ==================== Status Parsing Tests ====================


def test_parse_openai_video_status_queued():
    """Test parsing video with queued status."""
    mock_video = MagicMock()
    mock_video.id = "video-123"
    mock_video.status = "queued"
    mock_video.progress = 0
    mock_video.model = "sora-2"
    mock_video.size = "1280x720"
    mock_video.seconds = "8"

    result = parse_openai_video_status(mock_video)

    assert result.request_id == "video-123"
    assert result.status == "queued"
    assert result.progress_percent == 0


def test_parse_openai_video_status_in_progress():
    """Test parsing video with in_progress status."""
    mock_video = MagicMock()
    mock_video.id = "video-456"
    mock_video.status = "in_progress"
    mock_video.progress = 50
    mock_video.model = "sora-2"
    mock_video.size = "1280x720"
    mock_video.seconds = "8"

    result = parse_openai_video_status(mock_video)

    assert result.request_id == "video-456"
    assert result.status == "processing"
    assert result.progress_percent == 50


def test_parse_openai_video_status_completed():
    """Test parsing video with completed status."""
    mock_video = MagicMock()
    mock_video.id = "video-789"
    mock_video.status = "completed"
    mock_video.progress = 100
    mock_video.model = "sora-2"
    mock_video.size = "1280x720"
    mock_video.seconds = "8"

    result = parse_openai_video_status(mock_video)

    assert result.request_id == "video-789"
    assert result.status == "completed"
    assert result.progress_percent == 100


def test_parse_openai_video_status_unknown_defaults_to_processing():
    """Test parsing video with unknown status defaults to processing."""
    mock_video = MagicMock()
    mock_video.id = "video-000"
    mock_video.status = "unknown_status"
    mock_video.progress = None
    mock_video.model = "sora-2"
    mock_video.size = None
    mock_video.seconds = None

    result = parse_openai_video_status(mock_video)

    assert result.request_id == "video-000"
    assert result.status == "processing"
    assert result.progress_percent is None


# ==================== Async Video Generation Tests ====================


@pytest.mark.asyncio
async def test_generate_video_async_success_with_progress_callbacks(
    handler, base_config, base_request
):
    """Test successful async generation with sync and async progress callbacks."""
    # Create mock video objects
    mock_video_processing = MagicMock()
    mock_video_processing.id = "video-async-123"
    mock_video_processing.status = "in_progress"
    mock_video_processing.progress = 50
    mock_video_processing.model = "sora-2"
    mock_video_processing.size = "1280x720"
    mock_video_processing.seconds = "8"

    mock_video_completed = MagicMock()
    mock_video_completed.id = "video-async-123"
    mock_video_completed.status = "completed"
    mock_video_completed.url = "https://example.com/async-video.mp4"
    mock_video_completed.progress = 100
    mock_video_completed.model = "sora-2"
    mock_video_completed.size = "1280x720"
    mock_video_completed.seconds = "8"
    mock_video_completed.model_dump.return_value = {
        "id": "video-async-123",
        "status": "completed",
    }

    # Mock download response
    mock_download_response = AsyncMock()
    mock_download_response.response.aread = AsyncMock(
        return_value=b"fake video content"
    )

    # Setup mock async client
    mock_async_client = AsyncMock()
    mock_async_client.videos.create = AsyncMock(return_value=mock_video_processing)
    mock_async_client.videos.retrieve = AsyncMock(return_value=mock_video_completed)
    mock_async_client.videos.download_content = AsyncMock(
        return_value=mock_download_response
    )

    # Clear cache and patch
    handler._async_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.providers.openai.AsyncOpenAI",
        return_value=mock_async_client,
    ):
        # Test with sync callback
        progress_calls = []

        def sync_callback(update):
            progress_calls.append(update)

        result = await handler.generate_video_async(
            base_config, base_request, on_progress=sync_callback
        )

        mock_async_client.videos.create.assert_called_once()
        call_kwargs = mock_async_client.videos.create.call_args.kwargs
        assert call_kwargs["model"] == "sora-2"
        assert call_kwargs["prompt"] == "Test prompt"

        assert result.request_id == "video-async-123"
        assert result.video == {
            "content": b"fake video content",
            "content_type": "video/mp4",
        }
        assert len(progress_calls) >= 1

        mock_async_client.videos.create.reset_mock()

        # Test with async callback
        async_progress_calls = []

        async def async_callback(update):
            async_progress_calls.append(update)

        handler._async_client_cache.clear()
        mock_video_processing.status = "in_progress"

        with patch(
            "tarash.tarash_gateway.providers.openai.AsyncOpenAI",
            return_value=mock_async_client,
        ):
            await handler.generate_video_async(
                base_config, base_request, on_progress=async_callback
            )

        assert len(async_progress_calls) >= 1


@pytest.mark.asyncio
async def test_generate_video_async_handles_timeout(handler, base_config, base_request):
    """Test async generation handles timeout after max poll attempts."""
    mock_video = MagicMock()
    mock_video.id = "video-timeout"
    mock_video.status = "in_progress"
    mock_video.progress = 50
    mock_video.model = "sora-2"
    mock_video.size = "1280x720"
    mock_video.seconds = "8"

    mock_async_client = AsyncMock()
    mock_async_client.videos.create = AsyncMock(return_value=mock_video)
    mock_async_client.videos.retrieve = AsyncMock(return_value=mock_video)

    timeout_config = VideoGenerationConfig(
        model="sora-2",
        provider="openai",
        api_key="test-api-key",
        max_poll_attempts=2,
        poll_interval=1,
    )

    handler._async_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.providers.openai.AsyncOpenAI",
        return_value=mock_async_client,
    ):
        with pytest.raises(TimeoutError, match="timed out"):
            await handler.generate_video_async(timeout_config, base_request)


@pytest.mark.asyncio
async def test_generate_video_async_wraps_unknown_exceptions(
    handler, base_config, base_request
):
    """Test unknown exceptions are wrapped by decorator."""
    mock_async_client = AsyncMock()
    mock_async_client.videos.create = AsyncMock(
        side_effect=RuntimeError("Unexpected error")
    )

    handler._async_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.providers.openai.AsyncOpenAI",
        return_value=mock_async_client,
    ):
        with pytest.raises(GenerationFailedError, match="Error while generating video"):
            await handler.generate_video_async(base_config, base_request)


# ==================== Sync Video Generation Tests ====================


def test_generate_video_success_with_progress_callback(
    handler, base_config, base_request
):
    """Test successful sync generation with progress callback."""
    mock_video_queued = MagicMock()
    mock_video_queued.id = "video-sync-456"
    mock_video_queued.status = "queued"
    mock_video_queued.progress = 0
    mock_video_queued.model = "sora-2"
    mock_video_queued.size = "1280x720"
    mock_video_queued.seconds = "8"

    mock_video_completed = MagicMock()
    mock_video_completed.id = "video-sync-456"
    mock_video_completed.status = "completed"
    mock_video_completed.url = "https://example.com/sync-video.mp4"
    mock_video_completed.progress = 100
    mock_video_completed.model = "sora-2"
    mock_video_completed.size = "1280x720"
    mock_video_completed.seconds = "8"
    mock_video_completed.model_dump.return_value = {
        "id": "video-sync-456",
        "status": "completed",
    }

    # Mock download response
    mock_download_response = MagicMock()
    mock_download_response.read.return_value = b"fake video content"

    mock_sync_client = MagicMock()
    mock_sync_client.videos.create.return_value = mock_video_queued
    mock_sync_client.videos.retrieve.return_value = mock_video_completed
    mock_sync_client.videos.download_content.return_value = mock_download_response

    progress_calls = []

    def progress_callback(update):
        progress_calls.append(update)

    handler._sync_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.providers.openai.OpenAI",
        return_value=mock_sync_client,
    ):
        result = handler.generate_video(
            base_config, base_request, on_progress=progress_callback
        )

    mock_sync_client.videos.create.assert_called_once()
    call_kwargs = mock_sync_client.videos.create.call_args.kwargs
    assert call_kwargs["model"] == "sora-2"
    assert call_kwargs["prompt"] == "Test prompt"

    assert result.request_id == "video-sync-456"
    assert result.video == {
        "content": b"fake video content",
        "content_type": "video/mp4",
    }
    assert len(progress_calls) >= 1


def test_generate_video_handles_exceptions(handler, base_config, base_request):
    """Test exception handling in sync generation."""
    mock_sync_client = MagicMock()
    mock_sync_client.videos.create.side_effect = RuntimeError("Server error")

    handler._sync_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.providers.openai.OpenAI",
        return_value=mock_sync_client,
    ):
        with pytest.raises(GenerationFailedError, match="Error while generating video"):
            handler.generate_video(base_config, base_request)


def test_generate_video_handles_timeout(handler, base_config, base_request):
    """Test sync generation handles timeout after max poll attempts."""
    mock_video = MagicMock()
    mock_video.id = "video-timeout-sync"
    mock_video.status = "in_progress"
    mock_video.progress = 50
    mock_video.model = "sora-2"
    mock_video.size = "1280x720"
    mock_video.seconds = "8"

    mock_sync_client = MagicMock()
    mock_sync_client.videos.create.return_value = mock_video
    mock_sync_client.videos.retrieve.return_value = mock_video

    timeout_config = VideoGenerationConfig(
        model="sora-2",
        provider="openai",
        api_key="test-api-key",
        max_poll_attempts=2,
        poll_interval=1,
    )

    handler._sync_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.providers.openai.OpenAI",
        return_value=mock_sync_client,
    ):
        with pytest.raises(TimeoutError, match="timed out"):
            handler.generate_video(timeout_config, base_request)


# ==================== Image Field Mapper Tests ====================


def test_get_openai_image_field_mappers_gpt_image_15():
    """Test field mapper lookup for GPT-Image-1.5 model."""
    from tarash.tarash_gateway.providers.openai import get_openai_image_field_mappers

    mappers = get_openai_image_field_mappers("gpt-image-1.5")

    assert "prompt" in mappers
    assert "size" in mappers
    assert "quality" in mappers
    assert "n" in mappers
    assert mappers["prompt"].required is True


def test_get_openai_image_field_mappers_dalle3():
    """Test field mapper lookup for DALL-E 3."""
    from tarash.tarash_gateway.providers.openai import get_openai_image_field_mappers

    mappers = get_openai_image_field_mappers("dall-e-3")

    assert "prompt" in mappers
    assert "size" in mappers
    assert "quality" in mappers
    assert "style" in mappers
    assert "n" in mappers


def test_get_openai_image_field_mappers_dalle2():
    """Test field mapper lookup for DALL-E 2."""
    from tarash.tarash_gateway.providers.openai import get_openai_image_field_mappers

    mappers = get_openai_image_field_mappers("dall-e-2")

    assert "prompt" in mappers
    assert "size" in mappers
    assert "n" in mappers


def test_get_openai_image_field_mappers_unknown_model_uses_fallback():
    """Test that unknown models fall back to generic mappers."""
    from tarash.tarash_gateway.providers.openai import get_openai_image_field_mappers

    mappers = get_openai_image_field_mappers("unknown-model")

    assert "prompt" in mappers
    assert mappers["prompt"].required is True


def test_gpt_image_15_field_mappers_with_valid_request():
    """Test GPT-Image-1.5 field mappers with valid request."""
    from tarash.tarash_gateway.models import ImageGenerationRequest
    from tarash.tarash_gateway.providers.openai import get_openai_image_field_mappers
    from tarash.tarash_gateway.providers.field_mappers import apply_field_mappers

    mappers = get_openai_image_field_mappers("gpt-image-1.5")
    request = ImageGenerationRequest(
        prompt="A serene landscape",
        size="1024x1024",
        quality="high",
        n=2,
    )

    result = apply_field_mappers(mappers, request)

    assert result["prompt"] == "A serene landscape"
    assert result["size"] == "1024x1024"
    assert result["quality"] == "high"
    assert result["n"] == 2


def test_gpt_image_15_field_mappers_validates_size():
    """Test GPT-Image-1.5 validates size parameter."""
    from tarash.tarash_gateway.models import ImageGenerationRequest
    from tarash.tarash_gateway.providers.openai import get_openai_image_field_mappers
    from tarash.tarash_gateway.providers.field_mappers import apply_field_mappers

    mappers = get_openai_image_field_mappers("gpt-image-1.5")
    request = ImageGenerationRequest(
        prompt="Test",
        size="invalid_size",
    )

    with pytest.raises(ValueError, match="Invalid size"):
        apply_field_mappers(mappers, request)


def test_dalle3_field_mappers_with_valid_request():
    """Test DALL-E 3 field mappers with valid request."""
    from tarash.tarash_gateway.models import ImageGenerationRequest
    from tarash.tarash_gateway.providers.openai import get_openai_image_field_mappers
    from tarash.tarash_gateway.providers.field_mappers import apply_field_mappers

    mappers = get_openai_image_field_mappers("dall-e-3")
    request = ImageGenerationRequest(
        prompt="A surreal painting",
        size="1024x1792",
        quality="hd",
        style="vivid",
        n=1,
    )

    result = apply_field_mappers(mappers, request)

    assert result["prompt"] == "A surreal painting"
    assert result["size"] == "1024x1792"
    assert result["quality"] == "hd"
    assert result["style"] == "vivid"
    assert result["n"] == 1


def test_dalle3_field_mappers_validates_n_only_one():
    """Test DALL-E 3 only allows n=1."""
    from tarash.tarash_gateway.models import ImageGenerationRequest
    from tarash.tarash_gateway.providers.openai import get_openai_image_field_mappers
    from tarash.tarash_gateway.providers.field_mappers import apply_field_mappers

    mappers = get_openai_image_field_mappers("dall-e-3")
    request = ImageGenerationRequest(
        prompt="Test",
        n=2,
    )

    with pytest.raises(ValueError, match="n must be between 1 and 1"):
        apply_field_mappers(mappers, request)


def test_dalle2_field_mappers_with_valid_request():
    """Test DALL-E 2 field mappers with valid request."""
    from tarash.tarash_gateway.models import ImageGenerationRequest
    from tarash.tarash_gateway.providers.openai import get_openai_image_field_mappers
    from tarash.tarash_gateway.providers.field_mappers import apply_field_mappers

    mappers = get_openai_image_field_mappers("dall-e-2")
    request = ImageGenerationRequest(
        prompt="A digital art piece",
        size="512x512",
        n=4,
    )

    result = apply_field_mappers(mappers, request)

    assert result["prompt"] == "A digital art piece"
    assert result["size"] == "512x512"
    assert result["n"] == 4


def test_dalle2_field_mappers_allows_up_to_10_images():
    """Test DALL-E 2 allows up to 10 images."""
    from tarash.tarash_gateway.models import ImageGenerationRequest
    from tarash.tarash_gateway.providers.openai import get_openai_image_field_mappers
    from tarash.tarash_gateway.providers.field_mappers import apply_field_mappers

    mappers = get_openai_image_field_mappers("dall-e-2")
    request = ImageGenerationRequest(
        prompt="Test",
        n=10,
    )

    result = apply_field_mappers(mappers, request)
    assert result["n"] == 10


def test_dalle2_field_mappers_rejects_more_than_10_images():
    """Test DALL-E 2 rejects more than 10 images."""
    from tarash.tarash_gateway.models import ImageGenerationRequest
    from tarash.tarash_gateway.providers.openai import get_openai_image_field_mappers
    from tarash.tarash_gateway.providers.field_mappers import apply_field_mappers

    mappers = get_openai_image_field_mappers("dall-e-2")
    request = ImageGenerationRequest(
        prompt="Test",
        n=11,
    )

    with pytest.raises(ValueError, match="n must be between 1 and 10"):
        apply_field_mappers(mappers, request)


# ==================== Image Generation Tests ====================


@pytest.mark.asyncio
async def test_generate_image_async_dalle3_success(handler):
    """Test async image generation with DALL-E 3."""
    from tarash.tarash_gateway.models import (
        ImageGenerationConfig,
        ImageGenerationRequest,
    )

    config = ImageGenerationConfig(
        model="dall-e-3",
        provider="openai",
        api_key="test-api-key",
        timeout=120,
    )

    request = ImageGenerationRequest(
        prompt="A serene landscape",
        size="1024x1024",
        quality="hd",
        style="vivid",
        n=1,
    )

    mock_image = MagicMock()
    mock_image.url = "https://example.com/image.png"
    mock_image.revised_prompt = "A beautiful serene landscape at sunset"

    mock_response = MagicMock()
    mock_response.data = [mock_image]
    mock_response.model_dump.return_value = {
        "data": [{"url": "https://example.com/image.png"}]
    }

    mock_async_client = AsyncMock()
    mock_async_client.images.generate = AsyncMock(return_value=mock_response)

    handler._async_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.providers.openai.AsyncOpenAI",
        return_value=mock_async_client,
    ):
        response = await handler.generate_image_async(config, request)

    assert response.request_id is not None
    assert len(response.images) == 1
    assert response.images[0] == "https://example.com/image.png"
    assert response.revised_prompt == "A beautiful serene landscape at sunset"
    assert response.status == "completed"


def test_generate_image_sync_gpt_image_15_success(handler):
    """Test sync image generation with GPT-Image-1.5."""
    from tarash.tarash_gateway.models import (
        ImageGenerationConfig,
        ImageGenerationRequest,
    )

    config = ImageGenerationConfig(
        model="gpt-image-1.5",
        provider="openai",
        api_key="test-api-key",
        timeout=120,
    )

    request = ImageGenerationRequest(
        prompt="A futuristic city",
        size="1024x1792",
        quality="high",
        n=2,
    )

    mock_image1 = MagicMock()
    mock_image1.url = "https://example.com/image1.png"
    mock_image1.revised_prompt = None

    mock_image2 = MagicMock()
    mock_image2.url = "https://example.com/image2.png"
    mock_image2.revised_prompt = None

    mock_response = MagicMock()
    mock_response.data = [mock_image1, mock_image2]
    mock_response.model_dump.return_value = {
        "data": [
            {"url": "https://example.com/image1.png"},
            {"url": "https://example.com/image2.png"},
        ]
    }

    mock_sync_client = MagicMock()
    mock_sync_client.images.generate.return_value = mock_response

    handler._sync_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.providers.openai.OpenAI",
        return_value=mock_sync_client,
    ):
        response = handler.generate_image(config, request)

    assert response.request_id is not None
    assert len(response.images) == 2
    assert response.images[0] == "https://example.com/image1.png"
    assert response.images[1] == "https://example.com/image2.png"
    assert response.status == "completed"
