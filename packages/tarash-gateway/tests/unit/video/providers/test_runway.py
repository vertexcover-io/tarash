"""Tests for RunwayProviderHandler."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from runwayml import BadRequestError
from tarash.tarash_gateway.exceptions import (
    GenerationFailedError,
    TimeoutError,
    ValidationError,
)
from tarash.tarash_gateway.models import (
    VideoGenerationConfig,
    VideoGenerationRequest,
)
from tarash.tarash_gateway.providers.runway import (
    RunwayProviderHandler,
    _get_endpoint_from_model,
    parse_runway_task_status,
)


# ==================== Fixtures ====================


@pytest.fixture
def mock_sync_client():
    """Patch RunwayML and provide mock."""
    mock = MagicMock()
    with patch("tarash.tarash_gateway.providers.runway.RunwayML", return_value=mock):
        yield mock


@pytest.fixture
def mock_async_client():
    """Patch AsyncRunwayML and provide mock."""
    mock = AsyncMock()
    with patch(
        "tarash.tarash_gateway.providers.runway.AsyncRunwayML", return_value=mock
    ):
        yield mock


@pytest.fixture
def handler():
    """Create a RunwayProviderHandler instance."""
    return RunwayProviderHandler()


@pytest.fixture
def base_config():
    """Create a base VideoGenerationConfig."""
    return VideoGenerationConfig(
        model="veo3.1",
        provider="runway",
        api_key="test-api-key",
        timeout=600,
        poll_interval=1,
        max_poll_attempts=3,
    )


@pytest.fixture
def text_to_video_request():
    """Create a text-to-video request."""
    return VideoGenerationRequest(
        prompt="A bunny hopping in a meadow",
        aspect_ratio="16:9",
        duration_seconds=4,
    )


@pytest.fixture
def image_to_video_request():
    """Create an image-to-video request."""
    return VideoGenerationRequest(
        prompt="A cute bunny hopping",
        image_list=[
            {
                "type": "reference",
                "image": "https://example.com/bunny.jpg",
            }
        ],
        aspect_ratio="16:9",
        duration_seconds=5,
    )


@pytest.fixture
def video_to_video_request():
    """Create a video-to-video request."""
    return VideoGenerationRequest(
        prompt="Add easter elements to the video",
        video="https://example.com/video.mp4",
        aspect_ratio="16:9",
    )


# ==================== Endpoint Routing Tests ====================


def test_get_endpoint_text_to_video():
    """Test routing to text-to-video endpoint."""
    assert _get_endpoint_from_model("veo3.1", False, False) == "text_to_video"
    assert _get_endpoint_from_model("veo3", False, False) == "text_to_video"
    assert _get_endpoint_from_model("veo3.1_fast", False, False) == "text_to_video"


def test_get_endpoint_image_to_video():
    """Test routing to image-to-video endpoint."""
    assert _get_endpoint_from_model("veo3.1", True, False) == "image_to_video"
    assert _get_endpoint_from_model("gen4_turbo", True, False) == "image_to_video"
    assert _get_endpoint_from_model("gen3a_turbo", True, False) == "image_to_video"


def test_get_endpoint_video_to_video():
    """Test routing to video-to-video endpoint."""
    assert _get_endpoint_from_model("gen4_aleph", False, True) == "video_to_video"


def test_get_endpoint_validation_errors():
    """Test endpoint routing validation errors."""
    # gen4_aleph requires video
    with pytest.raises(ValidationError, match="requires video input"):
        _get_endpoint_from_model("gen4_aleph", False, False)

    # gen4_turbo requires image
    with pytest.raises(ValidationError, match="requires image input"):
        _get_endpoint_from_model("gen4_turbo", False, False)

    # veo models don't support video-to-video
    with pytest.raises(ValidationError, match="does not support video input"):
        _get_endpoint_from_model("veo3.1", False, True)


# ==================== Parameter Conversion Tests ====================


def test_convert_request_text_to_video(handler, base_config, text_to_video_request):
    """Test text-to-video request conversion."""
    endpoint, params = handler._convert_request(base_config, text_to_video_request)

    assert endpoint == "text_to_video"
    assert params["model"] == "veo3.1"
    assert params["prompt_text"] == "A bunny hopping in a meadow"
    assert params["ratio"] == "1280:720"  # Converted from 16:9
    assert params["duration"] == 4


def test_convert_request_image_to_video(handler, base_config, image_to_video_request):
    """Test image-to-video request conversion."""
    config = VideoGenerationConfig(
        model="gen4_turbo",
        provider="runway",
        api_key="test-api-key",
        timeout=600,
    )
    endpoint, params = handler._convert_request(config, image_to_video_request)

    assert endpoint == "image_to_video"
    assert params["model"] == "gen4_turbo"
    assert params["prompt_image"] == "https://example.com/bunny.jpg"
    assert params["prompt_text"] == "A cute bunny hopping"
    assert params["ratio"] == "1280:720"
    assert params["duration"] == 5


def test_convert_request_video_to_video(handler, video_to_video_request):
    """Test video-to-video request conversion."""
    config = VideoGenerationConfig(
        model="gen4_aleph",
        provider="runway",
        api_key="test-api-key",
        timeout=600,
    )
    endpoint, params = handler._convert_request(config, video_to_video_request)

    assert endpoint == "video_to_video"
    assert params["model"] == "gen4_aleph"
    assert params["video_uri"] == "https://example.com/video.mp4"
    assert params["prompt_text"] == "Add easter elements to the video"
    assert params["ratio"] == "1280:720"


def test_convert_request_aspect_ratio_validation(handler, base_config):
    """Test aspect ratio validation for different endpoints."""
    # Invalid text-to-video ratio
    request = VideoGenerationRequest(
        prompt="Test",
        aspect_ratio="4:3",  # Not supported for text-to-video
    )
    with pytest.raises(ValidationError, match="Invalid aspect ratio"):
        handler._convert_request(base_config, request)


def test_convert_request_duration_validation(handler, base_config):
    """Test duration validation."""
    # Text-to-video: must be 4, 6, or 8
    request = VideoGenerationRequest(
        prompt="Test",
        duration_seconds=5,  # Not allowed
    )
    with pytest.raises(ValidationError, match="Invalid duration"):
        handler._convert_request(base_config, request)


def test_convert_request_image_to_video_duration_range(handler):
    """Test image-to-video duration range (2-10 seconds)."""
    config = VideoGenerationConfig(
        model="gen4_turbo",
        provider="runway",
        api_key="test-api-key",
        timeout=600,
    )
    request = VideoGenerationRequest(
        prompt="Test",
        image_list=[
            {
                "type": "reference",
                "image": "https://example.com/image.jpg",
            }
        ],
        duration_seconds=15,  # Out of range
    )
    with pytest.raises(ValidationError, match="Duration must be"):
        handler._convert_request(config, request)


def test_convert_request_content_moderation(
    handler, base_config, text_to_video_request
):
    """Test content moderation parameter."""
    text_to_video_request.extra_params = {
        "content_moderation": {"public_figure_threshold": "low"}
    }

    endpoint, params = handler._convert_request(base_config, text_to_video_request)

    assert "content_moderation" in params
    assert params["content_moderation"]["public_figure_threshold"] == "low"


# ==================== Response Conversion Tests ====================


def test_convert_response_success(handler, base_config, text_to_video_request):
    """Test successful response conversion."""
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.status = "SUCCEEDED"
    mock_task.output = ["https://example.com/video.mp4"]

    response = handler._convert_response(
        base_config, text_to_video_request, "test-task-id", mock_task
    )

    assert response.request_id == "test-task-id"
    assert response.video == "https://example.com/video.mp4"
    assert response.status == "completed"
    assert response.content_type == "video/mp4"


def test_convert_response_failed_task(handler, base_config, text_to_video_request):
    """Test response conversion with failed task."""
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.status = "FAILED"
    mock_task.error = "Generation failed"

    with pytest.raises(GenerationFailedError, match="Generation failed"):
        handler._convert_response(
            base_config, text_to_video_request, "test-task-id", mock_task
        )


def test_convert_response_no_output(handler, base_config, text_to_video_request):
    """Test response conversion with no output."""
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.status = "SUCCEEDED"
    mock_task.output = None

    with pytest.raises(GenerationFailedError, match="No video URL found"):
        handler._convert_response(
            base_config, text_to_video_request, "test-task-id", mock_task
        )


# ==================== Status Parsing Tests ====================


def test_parse_runway_task_status():
    """Test parsing Runway task status."""
    mock_task = MagicMock()
    mock_task.id = "test-id"
    mock_task.status = "RUNNING"

    update = parse_runway_task_status(mock_task)

    assert update.request_id == "test-id"
    assert update.status == "processing"
    assert update.progress_percent is None


@pytest.mark.parametrize(
    "runway_status,expected_status",
    [
        ("PENDING", "queued"),
        ("THROTTLED", "queued"),
        ("RUNNING", "processing"),
        ("SUCCEEDED", "completed"),
        ("FAILED", "failed"),
        ("CANCELLED", "failed"),
    ],
)
def test_parse_runway_task_status_mapping(runway_status, expected_status):
    """Test status mapping from Runway to normalized format."""
    mock_task = MagicMock()
    mock_task.id = "test-id"
    mock_task.status = runway_status

    update = parse_runway_task_status(mock_task)

    assert update.status == expected_status


# ==================== Integration Tests ====================


@pytest.mark.asyncio
async def test_generate_video_async_text_to_video(
    handler, base_config, text_to_video_request, mock_async_client
):
    """Test async text-to-video generation."""
    # Mock task response
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.status = "SUCCEEDED"
    mock_task.output = ["https://example.com/video.mp4"]

    # Mock API calls
    mock_async_client.text_to_video.create = AsyncMock(return_value=mock_task)
    mock_async_client.tasks.retrieve = AsyncMock(return_value=mock_task)

    response = await handler.generate_video_async(base_config, text_to_video_request)

    assert response.request_id == "test-task-id"
    assert response.video == "https://example.com/video.mp4"
    assert response.status == "completed"


@pytest.mark.asyncio
async def test_generate_video_async_timeout(
    handler, base_config, text_to_video_request, mock_async_client
):
    """Test async generation timeout."""
    # Mock task that never completes
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.status = "RUNNING"

    mock_async_client.text_to_video.create = AsyncMock(return_value=mock_task)
    mock_async_client.tasks.retrieve = AsyncMock(return_value=mock_task)

    # Should timeout after max_poll_attempts (3)
    with pytest.raises(TimeoutError, match="timed out"):
        await handler.generate_video_async(base_config, text_to_video_request)


def test_generate_video_sync_text_to_video(
    handler, base_config, text_to_video_request, mock_sync_client
):
    """Test sync text-to-video generation."""
    # Mock task response
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.status = "SUCCEEDED"
    mock_task.output = ["https://example.com/video.mp4"]

    # Mock API calls
    mock_sync_client.text_to_video.create = MagicMock(return_value=mock_task)
    mock_sync_client.tasks.retrieve = MagicMock(return_value=mock_task)

    response = handler.generate_video(base_config, text_to_video_request)

    assert response.request_id == "test-task-id"
    assert response.video == "https://example.com/video.mp4"
    assert response.status == "completed"


# ==================== Error Handling Tests ====================


def test_handle_error_validation_error(handler, base_config, text_to_video_request):
    # Create a mock BadRequestError with proper attributes
    class MockBadRequestError(BadRequestError):
        def __init__(self, message):
            self.message = message
            self.status_code = 400
            self.body = {"error": message}

    error = MockBadRequestError("Invalid parameters")
    result = handler._handle_error(base_config, text_to_video_request, "test-id", error)

    assert isinstance(result, ValidationError)
    assert "Invalid parameters" in str(result)


def test_handle_error_generic_error(handler, base_config, text_to_video_request):
    """Test handling of generic errors."""
    error = Exception("Something went wrong")
    result = handler._handle_error(base_config, text_to_video_request, "test-id", error)

    assert isinstance(result, GenerationFailedError)
    assert "Error while generating video" in str(result)
