"""Tests for OpenAIProviderHandler."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tarash.tarash_gateway.video.exceptions import (
    ValidationError,
    VideoGenerationError,
)
from tarash.tarash_gateway.video.models import (
    VideoGenerationConfig,
    VideoGenerationRequest,
)
from tarash.tarash_gateway.video.providers.openai import (
    OpenAIProviderHandler,
    parse_openai_video_status,
)


# ==================== Fixtures ====================


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
        poll_interval=1,  # Fast polling for tests
        max_poll_attempts=3,
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


# ==================== Request Conversion Tests (Internal - Complex Logic) ====================
# Keep minimal tests for complex parameter mapping logic


def test_convert_request_with_invalid_duration(handler, base_config):
    """Test that invalid durations are rejected."""
    request = VideoGenerationRequest(prompt="A test video", duration_seconds=5)

    with pytest.raises(ValidationError) as exc_info:
        handler._convert_request(base_config, request)

    assert "Invalid duration" in str(exc_info.value)
    assert "5 seconds" in str(exc_info.value)
    assert "4, 8, 12" in str(exc_info.value)


def test_convert_request_with_invalid_aspect_ratio(handler, base_config):
    """Test that invalid aspect ratios are rejected."""
    request = VideoGenerationRequest(prompt="A test video", aspect_ratio="21:9")

    with pytest.raises(ValidationError) as exc_info:
        handler._convert_request(base_config, request)

    assert "Invalid aspect ratio" in str(exc_info.value)
    assert "21:9" in str(exc_info.value)


def test_convert_request_with_multiple_images_raises_error(handler, base_config):
    """Test that multiple reference images are rejected."""
    request = VideoGenerationRequest(
        prompt="A test video",
        image_list=[
            {"image": "https://example.com/img1.jpg", "type": "reference"},
            {"image": "https://example.com/img2.jpg", "type": "reference"},
        ],
    )

    with pytest.raises(ValidationError) as exc_info:
        handler._convert_request(base_config, request)

    assert "only supports 1 reference image" in str(exc_info.value)
    assert "got 2" in str(exc_info.value)


# ==================== Status Parsing Tests (Public Utility Function) ====================


def test_parse_openai_video_status_queued():
    """Test parsing video with queued status."""
    mock_video = MagicMock()
    mock_video.id = "video-123"
    mock_video.status = "queued"
    mock_video.progress = 0

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

    result = parse_openai_video_status(mock_video)

    assert result.request_id == "video-789"
    assert result.status == "completed"
    assert result.progress_percent == 100


# ==================== Async Video Generation Tests ====================


@pytest.mark.asyncio
async def test_generate_video_async_basic_success(handler, base_config, base_request):
    """Test successful async video generation with minimal parameters."""
    mock_video_completed = MagicMock()
    mock_video_completed.id = "video-123"
    mock_video_completed.status = "completed"
    mock_video_completed.seconds = "8"
    mock_video_completed.size = "1280x720"
    mock_video_completed.model_dump.return_value = {
        "id": "video-123",
        "status": "completed",
    }

    video_content = b"fake video content"
    mock_download_response = AsyncMock()
    mock_download_response.response.aread = AsyncMock(return_value=video_content)
    mock_download_response.encoding = "video/mp4"

    mock_async_client = AsyncMock()
    mock_async_client.videos.create = AsyncMock(return_value=mock_video_completed)
    mock_async_client.videos.retrieve = AsyncMock(return_value=mock_video_completed)
    mock_async_client.videos.download_content = AsyncMock(
        return_value=mock_download_response
    )

    handler._async_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.video.providers.openai.AsyncOpenAI",
        return_value=mock_async_client,
    ):
        result = await handler.generate_video_async(base_config, base_request)

        assert result.request_id == "video-123"
        assert isinstance(result.video, dict)
        assert result.video["content"] == video_content
        assert result.video["content_type"] == "video/mp4"
        assert result.content_type == "video/mp4"
        assert result.status == "completed"


@pytest.mark.asyncio
async def test_generate_video_async_with_duration(handler, base_config):
    """Test async generation with valid duration parameter."""
    request = VideoGenerationRequest(prompt="Test video", duration_seconds=8)

    mock_video_completed = MagicMock()
    mock_video_completed.id = "video-456"
    mock_video_completed.status = "completed"
    mock_video_completed.seconds = "8"
    mock_video_completed.size = "1280x720"
    mock_video_completed.model_dump.return_value = {
        "id": "video-456",
        "status": "completed",
    }

    video_content = b"fake video content"
    mock_download_response = AsyncMock()
    mock_download_response.response.aread = AsyncMock(return_value=video_content)

    mock_async_client = AsyncMock()
    mock_async_client.videos.create = AsyncMock(return_value=mock_video_completed)
    mock_async_client.videos.download_content = AsyncMock(
        return_value=mock_download_response
    )

    handler._async_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.video.providers.openai.AsyncOpenAI",
        return_value=mock_async_client,
    ):
        result = await handler.generate_video_async(base_config, request)

        assert result.duration == 8.0
        # Verify the API was called with correct duration (as string)
        call_args = mock_async_client.videos.create.call_args[1]
        assert call_args["seconds"] == "8"


@pytest.mark.asyncio
async def test_generate_video_async_with_aspect_ratio(handler, base_config):
    """Test async generation with aspect ratio parameter."""
    request = VideoGenerationRequest(prompt="Test video", aspect_ratio="16:9")

    mock_video_completed = MagicMock()
    mock_video_completed.id = "video-789"
    mock_video_completed.status = "completed"
    mock_video_completed.size = "1280x720"
    mock_video_completed.model_dump.return_value = {
        "id": "video-789",
        "status": "completed",
    }

    video_content = b"fake video content"
    mock_download_response = AsyncMock()
    mock_download_response.response.aread = AsyncMock(return_value=video_content)

    mock_async_client = AsyncMock()
    mock_async_client.videos.create = AsyncMock(return_value=mock_video_completed)
    mock_async_client.videos.download_content = AsyncMock(
        return_value=mock_download_response
    )

    handler._async_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.video.providers.openai.AsyncOpenAI",
        return_value=mock_async_client,
    ):
        result = await handler.generate_video_async(base_config, request)

        assert result.resolution == "1280x720"
        # Verify the API was called with converted size
        call_args = mock_async_client.videos.create.call_args[1]
        assert call_args["size"] == "1280x720"


@pytest.mark.asyncio
async def test_generate_video_async_with_image(handler, base_config):
    """Test async image-to-video generation."""
    request = VideoGenerationRequest(
        prompt="Animate this image",
        image_list=[
            {"image": "https://example.com/image.jpg", "type": "reference"},
        ],
    )

    mock_video_completed = MagicMock()
    mock_video_completed.id = "video-img"
    mock_video_completed.status = "completed"
    mock_video_completed.seconds = 4
    mock_video_completed.size = "1280x720"
    mock_video_completed.model_dump.return_value = {
        "id": "video-img",
        "status": "completed",
    }

    video_content = b"fake video content"
    mock_download_response = AsyncMock()
    mock_download_response.response.aread = AsyncMock(return_value=video_content)

    mock_async_client = AsyncMock()
    mock_async_client.videos.create = AsyncMock(return_value=mock_video_completed)
    mock_async_client.videos.download_content = AsyncMock(
        return_value=mock_download_response
    )

    handler._async_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.video.providers.openai.AsyncOpenAI",
        return_value=mock_async_client,
    ):
        with patch(
            "tarash.tarash_gateway.video.providers.openai.download_media_from_url",
            return_value=(b"fake image data", "image/jpeg"),
        ):
            result = await handler.generate_video_async(base_config, request)

            assert result.request_id == "video-img"
            # Verify input_reference was passed
            call_args = mock_async_client.videos.create.call_args[1]
            assert "input_reference" in call_args


@pytest.mark.asyncio
async def test_generate_video_async_with_progress_callbacks(
    handler, base_config, base_request
):
    """Test async generation with progress callbacks (sync and async)."""
    mock_video_processing = MagicMock()
    mock_video_processing.id = "video-progress"
    mock_video_processing.status = "in_progress"
    mock_video_processing.progress = 50

    mock_video_completed = MagicMock()
    mock_video_completed.id = "video-progress"
    mock_video_completed.status = "completed"
    mock_video_completed.seconds = 4
    mock_video_completed.size = "1280x720"
    mock_video_completed.model_dump.return_value = {
        "id": "video-progress",
        "status": "completed",
    }

    video_content = b"fake video content"
    mock_download_response = AsyncMock()
    mock_download_response.response.aread = AsyncMock(return_value=video_content)

    mock_async_client = AsyncMock()
    mock_async_client.videos.create = AsyncMock(return_value=mock_video_processing)
    mock_async_client.videos.retrieve = AsyncMock(return_value=mock_video_completed)
    mock_async_client.videos.download_content = AsyncMock(
        return_value=mock_download_response
    )

    # Test with sync callback
    progress_calls = []

    def sync_callback(update):
        progress_calls.append(update)

    handler._async_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.video.providers.openai.AsyncOpenAI",
        return_value=mock_async_client,
    ):
        result = await handler.generate_video_async(
            base_config, base_request, on_progress=sync_callback
        )

        assert result.request_id == "video-progress"
        assert len(progress_calls) >= 1
        assert any(call.status == "processing" for call in progress_calls)

    # Test with async callback
    async_progress_calls = []

    async def async_callback(update):
        async_progress_calls.append(update)

    handler._async_client_cache.clear()
    mock_video_processing.status = "in_progress"
    with patch(
        "tarash.tarash_gateway.video.providers.openai.AsyncOpenAI",
        return_value=mock_async_client,
    ):
        await handler.generate_video_async(
            base_config, base_request, on_progress=async_callback
        )

        assert len(async_progress_calls) >= 1


@pytest.mark.asyncio
async def test_generate_video_async_handles_failed_status(
    handler, base_config, base_request
):
    """Test async generation handles failed video status."""
    mock_error = MagicMock()
    mock_error.message = "Content policy violation"

    mock_video_failed = MagicMock()
    mock_video_failed.id = "video-failed"
    mock_video_failed.status = "failed"
    mock_video_failed.error = mock_error
    mock_video_failed.model_dump.return_value = {
        "id": "video-failed",
        "status": "failed",
    }

    mock_async_client = AsyncMock()
    mock_async_client.videos.create = AsyncMock(return_value=mock_video_failed)

    handler._async_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.video.providers.openai.AsyncOpenAI",
        return_value=mock_async_client,
    ):
        with pytest.raises(VideoGenerationError, match="Content policy violation"):
            await handler.generate_video_async(base_config, base_request)


@pytest.mark.asyncio
async def test_generate_video_async_handles_timeout(handler, base_config, base_request):
    """Test async generation handles timeout after max poll attempts."""
    mock_video = MagicMock()
    mock_video.id = "video-timeout"
    mock_video.status = "in_progress"
    mock_video.progress = 50

    mock_async_client = AsyncMock()
    mock_async_client.videos.create = AsyncMock(return_value=mock_video)
    mock_async_client.videos.retrieve = AsyncMock(return_value=mock_video)

    handler._async_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.video.providers.openai.AsyncOpenAI",
        return_value=mock_async_client,
    ):
        with pytest.raises(VideoGenerationError, match="timed out"):
            await handler.generate_video_async(base_config, base_request)


@pytest.mark.asyncio
async def test_generate_video_async_handles_bad_request_error(handler, base_config):
    """Test async generation converts BadRequestError (400) to ValidationError."""
    try:
        from openai import BadRequestError
    except ImportError:
        pytest.skip("openai package not available")

    # Use a valid duration but have the API return BadRequestError for other reasons
    request = VideoGenerationRequest(prompt="Test video", duration_seconds=8)

    bad_request_error = BadRequestError(
        "Invalid prompt content",
        response=MagicMock(status_code=400),
        body={"error": {"message": "Invalid prompt content"}},
    )

    mock_async_client = AsyncMock()
    mock_async_client.videos.create = AsyncMock(side_effect=bad_request_error)

    handler._async_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.video.providers.openai.AsyncOpenAI",
        return_value=mock_async_client,
    ):
        with pytest.raises(ValidationError) as exc_info:
            await handler.generate_video_async(base_config, request)

        assert "Invalid request parameters" in str(exc_info.value)
        assert exc_info.value.provider == "openai"


@pytest.mark.asyncio
async def test_generate_video_async_handles_download_failure(
    handler, base_config, base_request
):
    """Test async generation handles failure when downloading video content."""
    mock_video_completed = MagicMock()
    mock_video_completed.id = "video-download-fail"
    mock_video_completed.status = "completed"
    mock_video_completed.model_dump.return_value = {
        "id": "video-download-fail",
        "status": "completed",
    }

    mock_async_client = AsyncMock()
    mock_async_client.videos.create = AsyncMock(return_value=mock_video_completed)
    mock_async_client.videos.download_content = AsyncMock(
        side_effect=RuntimeError("Download failed")
    )

    handler._async_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.video.providers.openai.AsyncOpenAI",
        return_value=mock_async_client,
    ):
        with pytest.raises(VideoGenerationError, match="Download failed"):
            await handler.generate_video_async(base_config, base_request)


@pytest.mark.asyncio
async def test_generate_video_async_handles_network_error(
    handler, base_config, base_request
):
    """Test async generation handles network errors."""
    mock_async_client = AsyncMock()
    mock_async_client.videos.create = AsyncMock(
        side_effect=RuntimeError("Network connection failed")
    )

    handler._async_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.video.providers.openai.AsyncOpenAI",
        return_value=mock_async_client,
    ):
        with pytest.raises(VideoGenerationError, match="Network connection failed"):
            await handler.generate_video_async(base_config, base_request)


# ==================== Sync Video Generation Tests ====================


def test_generate_video_basic_success(handler, base_config, base_request):
    """Test successful sync video generation with minimal parameters."""
    mock_video_completed = MagicMock()
    mock_video_completed.id = "video-sync-123"
    mock_video_completed.status = "completed"
    mock_video_completed.seconds = "8"
    mock_video_completed.size = "1280x720"
    mock_video_completed.model_dump.return_value = {
        "id": "video-sync-123",
        "status": "completed",
    }

    video_content = b"fake video content"
    mock_download_response = MagicMock()
    mock_download_response.read.return_value = video_content
    mock_download_response.encoding = "video/mp4"

    mock_sync_client = MagicMock()
    mock_sync_client.videos.create.return_value = mock_video_completed
    mock_sync_client.videos.retrieve.return_value = mock_video_completed
    mock_sync_client.videos.download_content.return_value = mock_download_response

    handler._sync_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.video.providers.openai.OpenAI",
        return_value=mock_sync_client,
    ):
        result = handler.generate_video(base_config, base_request)

        assert result.request_id == "video-sync-123"
        assert isinstance(result.video, dict)
        assert result.video["content"] == video_content
        assert result.video["content_type"] == "video/mp4"
        assert result.status == "completed"


def test_generate_video_with_duration(handler, base_config):
    """Test sync generation with valid duration parameter."""
    request = VideoGenerationRequest(prompt="Test video", duration_seconds=12)

    mock_video_completed = MagicMock()
    mock_video_completed.id = "video-sync-duration"
    mock_video_completed.status = "completed"
    mock_video_completed.seconds = 12
    mock_video_completed.size = "1280x720"
    mock_video_completed.model_dump.return_value = {
        "id": "video-sync-duration",
        "status": "completed",
    }

    video_content = b"fake video content"
    mock_download_response = MagicMock()
    mock_download_response.read.return_value = video_content

    mock_sync_client = MagicMock()
    mock_sync_client.videos.create.return_value = mock_video_completed
    mock_sync_client.videos.download_content.return_value = mock_download_response

    handler._sync_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.video.providers.openai.OpenAI",
        return_value=mock_sync_client,
    ):
        result = handler.generate_video(base_config, request)

        assert result.duration == 12.0
        # Verify the API was called with correct duration (as string)
        call_args = mock_sync_client.videos.create.call_args[1]
        assert call_args["seconds"] == "12"


def test_generate_video_with_aspect_ratio(handler, base_config):
    """Test sync generation with aspect ratio parameter."""
    request = VideoGenerationRequest(prompt="Test video", aspect_ratio="9:16")

    mock_video_completed = MagicMock()
    mock_video_completed.id = "video-sync-aspect"
    mock_video_completed.status = "completed"
    mock_video_completed.size = "720x1280"
    mock_video_completed.model_dump.return_value = {
        "id": "video-sync-aspect",
        "status": "completed",
    }

    video_content = b"fake video content"
    mock_download_response = MagicMock()
    mock_download_response.read.return_value = video_content

    mock_sync_client = MagicMock()
    mock_sync_client.videos.create.return_value = mock_video_completed
    mock_sync_client.videos.download_content.return_value = mock_download_response

    handler._sync_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.video.providers.openai.OpenAI",
        return_value=mock_sync_client,
    ):
        result = handler.generate_video(base_config, request)

        assert result.resolution == "720x1280"
        # Verify the API was called with converted size
        call_args = mock_sync_client.videos.create.call_args[1]
        assert call_args["size"] == "720x1280"


def test_generate_video_with_progress_callback(handler, base_config, base_request):
    """Test sync generation with progress callback."""
    mock_video_processing = MagicMock()
    mock_video_processing.id = "video-sync-progress"
    mock_video_processing.status = "queued"
    mock_video_processing.progress = 0

    mock_video_completed = MagicMock()
    mock_video_completed.id = "video-sync-progress"
    mock_video_completed.status = "completed"
    mock_video_completed.seconds = 4
    mock_video_completed.size = "1280x720"
    mock_video_completed.model_dump.return_value = {
        "id": "video-sync-progress",
        "status": "completed",
    }

    video_content = b"fake video content"
    mock_download_response = MagicMock()
    mock_download_response.read.return_value = video_content

    mock_sync_client = MagicMock()
    mock_sync_client.videos.create.return_value = mock_video_processing
    mock_sync_client.videos.retrieve.return_value = mock_video_completed
    mock_sync_client.videos.download_content.return_value = mock_download_response

    progress_calls = []

    def progress_callback(update):
        progress_calls.append(update)

    handler._sync_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.video.providers.openai.OpenAI",
        return_value=mock_sync_client,
    ):
        result = handler.generate_video(
            base_config, base_request, on_progress=progress_callback
        )

        assert result.request_id == "video-sync-progress"
        assert len(progress_calls) >= 1
        assert any(call.status == "queued" for call in progress_calls)


def test_generate_video_handles_failed_status(handler, base_config, base_request):
    """Test sync generation handles failed video status."""
    mock_error = MagicMock()
    mock_error.message = "Inappropriate content detected"

    mock_video_failed = MagicMock()
    mock_video_failed.id = "video-sync-failed"
    mock_video_failed.status = "failed"
    mock_video_failed.error = mock_error
    mock_video_failed.model_dump.return_value = {
        "id": "video-sync-failed",
        "status": "failed",
    }

    mock_sync_client = MagicMock()
    mock_sync_client.videos.create.return_value = mock_video_failed

    handler._sync_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.video.providers.openai.OpenAI",
        return_value=mock_sync_client,
    ):
        with pytest.raises(
            VideoGenerationError, match="Inappropriate content detected"
        ):
            handler.generate_video(base_config, base_request)


def test_generate_video_handles_timeout(handler, base_config, base_request):
    """Test sync generation handles timeout after max poll attempts."""
    mock_video = MagicMock()
    mock_video.id = "video-sync-timeout"
    mock_video.status = "in_progress"
    mock_video.progress = 50

    mock_sync_client = MagicMock()
    mock_sync_client.videos.create.return_value = mock_video
    mock_sync_client.videos.retrieve.return_value = mock_video

    handler._sync_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.video.providers.openai.OpenAI",
        return_value=mock_sync_client,
    ):
        with pytest.raises(VideoGenerationError, match="timed out"):
            handler.generate_video(base_config, base_request)


def test_generate_video_handles_download_failure(handler, base_config, base_request):
    """Test sync generation handles failure when downloading video content."""
    mock_video_completed = MagicMock()
    mock_video_completed.id = "video-sync-download-fail"
    mock_video_completed.status = "completed"
    mock_video_completed.model_dump.return_value = {
        "id": "video-sync-download-fail",
        "status": "completed",
    }

    mock_sync_client = MagicMock()
    mock_sync_client.videos.create.return_value = mock_video_completed
    mock_sync_client.videos.download_content.side_effect = RuntimeError(
        "Download service unavailable"
    )

    handler._sync_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.video.providers.openai.OpenAI",
        return_value=mock_sync_client,
    ):
        with pytest.raises(VideoGenerationError, match="Download service unavailable"):
            handler.generate_video(base_config, base_request)


def test_generate_video_handles_invalid_duration(handler, base_config):
    """Test sync generation rejects invalid duration."""
    request = VideoGenerationRequest(prompt="Test video", duration_seconds=15)

    handler._sync_client_cache.clear()
    with patch("tarash.tarash_gateway.video.providers.openai.OpenAI"):
        with pytest.raises(ValidationError) as exc_info:
            handler.generate_video(base_config, request)

        assert "Invalid duration" in str(exc_info.value)
        assert "15 seconds" in str(exc_info.value)


def test_generate_video_handles_invalid_aspect_ratio(handler, base_config):
    """Test sync generation rejects invalid aspect ratio."""
    request = VideoGenerationRequest(prompt="Test video", aspect_ratio="4:3")

    handler._sync_client_cache.clear()
    with patch("tarash.tarash_gateway.video.providers.openai.OpenAI"):
        with pytest.raises(ValidationError) as exc_info:
            handler.generate_video(base_config, request)

        assert "Invalid aspect ratio" in str(exc_info.value)
        assert "4:3" in str(exc_info.value)


def test_generate_video_handles_network_error(handler, base_config, base_request):
    """Test sync generation handles network errors."""
    mock_sync_client = MagicMock()
    mock_sync_client.videos.create.side_effect = RuntimeError("Connection refused")

    handler._sync_client_cache.clear()
    with patch(
        "tarash.tarash_gateway.video.providers.openai.OpenAI",
        return_value=mock_sync_client,
    ):
        with pytest.raises(VideoGenerationError, match="Connection refused"):
            handler.generate_video(base_config, base_request)
