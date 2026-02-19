"""Tests for AzureOpenAIProviderHandler."""

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
from tarash.tarash_gateway.providers.azure_openai import (
    AzureOpenAIProviderHandler,
)
from tarash.tarash_gateway.providers.openai import parse_openai_video_status

# Azure uses the same status parsing as OpenAI
parse_azure_video_status = parse_openai_video_status


# ==================== Fixtures ====================


@pytest.fixture
def handler():
    """Create an AzureOpenAIProviderHandler instance."""
    return AzureOpenAIProviderHandler()


@pytest.fixture
def base_config():
    """Create a base VideoGenerationConfig for Azure."""
    return VideoGenerationConfig(
        model="sora-deployment",  # Azure uses deployment name
        provider="azure_openai",
        api_key="test-api-key",
        base_url="https://my-resource.openai.azure.com/",
        timeout=600,
        poll_interval=1,
        max_poll_attempts=3,
    )


@pytest.fixture
def base_request():
    """Create a base VideoGenerationRequest."""
    return VideoGenerationRequest(prompt="Test prompt")


# ==================== Initialization Tests ====================
# ==================== Azure Config Parsing Tests ====================
# These are Azure-specific and need to be tested


def test_parse_azure_config_basic(handler, base_config):
    """Test parsing basic Azure configuration."""
    result = handler._parse_azure_config(base_config)

    assert result["azure_endpoint"] == "https://my-resource.openai.azure.com"
    assert result["api_version"] == "2024-05-01-preview"  # Default version
    assert result["api_key"] == "test-api-key"
    assert result["timeout"] == 600


def test_parse_azure_config_with_api_version_in_url(handler):
    """Test parsing Azure configuration with api-version in URL."""
    config = VideoGenerationConfig(
        model="sora-deployment",
        provider="azure_openai",
        api_key="test-api-key",
        base_url="https://my-resource.openai.azure.com/?api-version=2025-01-01",
        timeout=600,
    )

    result = handler._parse_azure_config(config)

    assert result["azure_endpoint"] == "https://my-resource.openai.azure.com"
    assert result["api_version"] == "2025-01-01"


def test_parse_azure_config_missing_base_url_raises_error(handler):
    """Test that missing base_url raises ValidationError."""
    config = VideoGenerationConfig(
        model="sora-deployment",
        provider="azure_openai",
        api_key="test-api-key",
        base_url=None,
        timeout=600,
    )

    with pytest.raises(ValidationError, match="base_url to be set"):
        handler._parse_azure_config(config)


# ==================== Status Parsing Tests (Public Utility Function) ====================


def test_parse_azure_video_status_queued():
    """Test parsing video with queued status."""
    mock_video = MagicMock()
    mock_video.id = "video-123"
    mock_video.status = "queued"
    mock_video.progress = 0

    result = parse_azure_video_status(mock_video)

    assert result.request_id == "video-123"
    assert result.status == "queued"
    assert result.progress_percent == 0


def test_parse_azure_video_status_in_progress():
    """Test parsing video with in_progress status."""
    mock_video = MagicMock()
    mock_video.id = "video-456"
    mock_video.status = "in_progress"
    mock_video.progress = 50

    result = parse_azure_video_status(mock_video)

    assert result.request_id == "video-456"
    assert result.status == "processing"
    assert result.progress_percent == 50


def test_parse_azure_video_status_completed():
    """Test parsing video with completed status."""
    mock_video = MagicMock()
    mock_video.id = "video-789"
    mock_video.status = "completed"
    mock_video.progress = 100

    result = parse_azure_video_status(mock_video)

    assert result.request_id == "video-789"
    assert result.status == "completed"
    assert result.progress_percent == 100


# ==================== Async Video Generation Tests ====================


@pytest.mark.asyncio
async def test_generate_video_async_basic_success(handler, base_config, base_request):
    """Test successful async video generation with minimal parameters."""
    mock_video_completed = MagicMock()
    mock_video_completed.id = "video-async-123"
    mock_video_completed.status = "completed"
    mock_video_completed.seconds = 4
    mock_video_completed.size = "1280x720"
    mock_video_completed.model_dump.return_value = {
        "id": "video-async-123",
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

    with patch(
        "tarash.tarash_gateway.providers.azure_openai.AsyncAzureOpenAI",
        return_value=mock_async_client,
    ):
        result = await handler.generate_video_async(base_config, base_request)

        assert result.request_id == "video-async-123"
        assert isinstance(result.video, dict)
        assert result.video["content"] == video_content
        assert result.video["content_type"] == "video/mp4"
        assert result.status == "completed"


@pytest.mark.asyncio
async def test_generate_video_async_with_duration(handler, base_config):
    """Test async generation with valid duration parameter."""
    request = VideoGenerationRequest(prompt="Test video", duration_seconds=8)

    mock_video_completed = MagicMock()
    mock_video_completed.id = "video-duration"
    mock_video_completed.status = "completed"
    mock_video_completed.seconds = 8
    mock_video_completed.size = "1280x720"
    mock_video_completed.model_dump.return_value = {
        "id": "video-duration",
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

    with patch(
        "tarash.tarash_gateway.providers.azure_openai.AsyncAzureOpenAI",
        return_value=mock_async_client,
    ):
        result = await handler.generate_video_async(base_config, request)

        assert result.duration == 8.0
        # Verify the API was called with correct duration (as integer)
        call_args = mock_async_client.videos.create.call_args[1]
        assert call_args["seconds"] == 8


@pytest.mark.asyncio
async def test_generate_video_async_with_progress_callbacks(
    handler, base_config, base_request
):
    """Test async generation with progress callbacks."""
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

    progress_calls = []

    def sync_callback(update):
        progress_calls.append(update)

    with patch(
        "tarash.tarash_gateway.providers.azure_openai.AsyncAzureOpenAI",
        return_value=mock_async_client,
    ):
        result = await handler.generate_video_async(
            base_config, base_request, on_progress=sync_callback
        )

        assert result.request_id == "video-progress"
        assert len(progress_calls) >= 1


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

    with patch(
        "tarash.tarash_gateway.providers.azure_openai.AsyncAzureOpenAI",
        return_value=mock_async_client,
    ):
        with pytest.raises(TimeoutError, match="timed out"):
            await handler.generate_video_async(base_config, base_request)


@pytest.mark.asyncio
async def test_generate_video_async_handles_network_error(
    handler, base_config, base_request
):
    """Test async generation handles network errors."""
    mock_async_client = AsyncMock()
    mock_async_client.videos.create = AsyncMock(
        side_effect=RuntimeError("Connection failed")
    )

    with patch(
        "tarash.tarash_gateway.providers.azure_openai.AsyncAzureOpenAI",
        return_value=mock_async_client,
    ):
        with pytest.raises(GenerationFailedError, match="Connection failed"):
            await handler.generate_video_async(base_config, base_request)


# ==================== Sync Video Generation Tests ====================


def test_generate_video_basic_success(handler, base_config, base_request):
    """Test successful sync video generation with minimal parameters."""
    mock_video_completed = MagicMock()
    mock_video_completed.id = "video-sync-123"
    mock_video_completed.status = "completed"
    mock_video_completed.seconds = 4
    mock_video_completed.size = "1280x720"
    mock_video_completed.model_dump.return_value = {
        "id": "video-sync-123",
        "status": "completed",
    }

    video_content = b"fake video content"
    mock_download_response = MagicMock()
    mock_download_response.read.return_value = video_content

    mock_sync_client = MagicMock()
    mock_sync_client.videos.create.return_value = mock_video_completed
    mock_sync_client.videos.download_content.return_value = mock_download_response

    with patch(
        "tarash.tarash_gateway.providers.azure_openai.AzureOpenAI",
        return_value=mock_sync_client,
    ):
        result = handler.generate_video(base_config, base_request)

        assert result.request_id == "video-sync-123"
        assert isinstance(result.video, dict)
        assert result.video["content"] == video_content
        assert result.status == "completed"


def test_generate_video_with_progress_callback(handler, base_config, base_request):
    """Test sync generation with progress callback."""
    mock_video_queued = MagicMock()
    mock_video_queued.id = "video-sync-progress"
    mock_video_queued.status = "queued"
    mock_video_queued.progress = 0

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
    mock_sync_client.videos.create.return_value = mock_video_queued
    mock_sync_client.videos.retrieve.return_value = mock_video_completed
    mock_sync_client.videos.download_content.return_value = mock_download_response

    progress_calls = []

    def progress_callback(update):
        progress_calls.append(update)

    with patch(
        "tarash.tarash_gateway.providers.azure_openai.AzureOpenAI",
        return_value=mock_sync_client,
    ):
        result = handler.generate_video(
            base_config, base_request, on_progress=progress_callback
        )

    assert result.request_id == "video-sync-progress"
    assert len(progress_calls) >= 1


def test_generate_video_handles_timeout(handler, base_config, base_request):
    """Test sync generation handles timeout after max poll attempts."""
    mock_video = MagicMock()
    mock_video.id = "video-timeout"
    mock_video.status = "in_progress"
    mock_video.progress = 50

    mock_sync_client = MagicMock()
    mock_sync_client.videos.create.return_value = mock_video
    mock_sync_client.videos.retrieve.return_value = mock_video

    with patch(
        "tarash.tarash_gateway.providers.azure_openai.AzureOpenAI",
        return_value=mock_sync_client,
    ):
        with pytest.raises(TimeoutError, match="timed out"):
            handler.generate_video(base_config, base_request)


def test_generate_video_handles_network_error(handler, base_config, base_request):
    """Test sync generation handles network errors."""
    mock_sync_client = MagicMock()
    mock_sync_client.videos.create.side_effect = RuntimeError("Server error")

    with patch(
        "tarash.tarash_gateway.providers.azure_openai.AzureOpenAI",
        return_value=mock_sync_client,
    ):
        with pytest.raises(GenerationFailedError, match="Server error"):
            handler.generate_video(base_config, base_request)
