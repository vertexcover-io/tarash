"""Tests for RunwayProviderHandler."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from runwayml import BadRequestError
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
from tarash.tarash_gateway.providers.runway import (
    RunwayProviderHandler,
    _convert_media_to_file,
    _extract_video_url,
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


def test_convert_request_seed_not_for_text_to_video(
    handler, base_config, text_to_video_request
):
    """Test seed is NOT passed for text_to_video endpoint."""
    text_to_video_request.seed = 42

    endpoint, params = handler._convert_request(base_config, text_to_video_request)

    # seed should NOT be passed for text_to_video
    assert endpoint == "text_to_video"
    assert "seed" not in params


def test_convert_request_seed_for_image_to_video(handler):
    """Test seed IS passed for image_to_video endpoint."""
    config = VideoGenerationConfig(
        model="gen4_turbo",
        provider="runway",
        api_key="test-api-key",
        timeout=600,
    )
    request = VideoGenerationRequest(
        prompt="A bunny hopping",
        image_list=[{"type": "reference", "image": "https://example.com/image.jpg"}],
        duration_seconds=5,
        aspect_ratio="16:9",
        seed=42,
    )

    endpoint, params = handler._convert_request(config, request)

    assert endpoint == "image_to_video"
    assert "seed" in params
    assert params["seed"] == 42


def test_convert_request_content_moderation_not_for_text_to_video(
    handler, base_config, text_to_video_request
):
    """Test content_moderation is NOT passed for text_to_video endpoint."""
    text_to_video_request.extra_params = {
        "content_moderation": {"public_figure_threshold": "low"}
    }

    endpoint, params = handler._convert_request(base_config, text_to_video_request)

    # content_moderation should NOT be passed for text_to_video
    assert endpoint == "text_to_video"
    assert "content_moderation" not in params


def test_convert_request_content_moderation_for_image_to_video(handler):
    """Test content_moderation IS passed for image_to_video endpoint."""
    config = VideoGenerationConfig(
        model="gen4_turbo",
        provider="runway",
        api_key="test-api-key",
        timeout=600,
    )
    request = VideoGenerationRequest(
        prompt="A bunny hopping",
        image_list=[{"type": "reference", "image": "https://example.com/image.jpg"}],
        duration_seconds=5,
        aspect_ratio="16:9",
        extra_params={"content_moderation": {"public_figure_threshold": "low"}},
    )

    endpoint, params = handler._convert_request(config, request)

    assert endpoint == "image_to_video"
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

    mock_async_client.text_to_video.create.assert_called_once()
    call_kwargs = mock_async_client.text_to_video.create.call_args.kwargs
    assert call_kwargs["prompt_text"] == "A bunny hopping in a meadow"
    assert call_kwargs["ratio"] == "1280:720"
    assert call_kwargs["duration"] == 4

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

    mock_sync_client.text_to_video.create.assert_called_once()
    call_kwargs = mock_sync_client.text_to_video.create.call_args.kwargs
    assert call_kwargs["prompt_text"] == "A bunny hopping in a meadow"
    assert call_kwargs["ratio"] == "1280:720"
    assert call_kwargs["duration"] == 4

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


def test_handle_error_api_timeout(handler, base_config, text_to_video_request):
    """Test APITimeoutError maps to TimeoutError."""
    from runwayml import APITimeoutError

    ex = APITimeoutError.__new__(APITimeoutError)
    ex.args = ("Request timed out",)

    result = handler._handle_error(base_config, text_to_video_request, "test-id", ex)

    assert isinstance(result, TimeoutError)
    assert "timed out" in result.message
    assert result.provider == "runway"
    assert result.timeout_seconds == 600


def test_handle_error_api_connection_error(handler, base_config, text_to_video_request):
    """Test APIConnectionError maps to HTTPConnectionError."""
    from runwayml import APIConnectionError

    ex = APIConnectionError.__new__(APIConnectionError)
    ex.args = ("Connection refused",)

    result = handler._handle_error(base_config, text_to_video_request, "test-id", ex)

    assert isinstance(result, HTTPConnectionError)
    assert "Connection error" in result.message


def test_handle_error_api_status_error_generic(
    handler, base_config, text_to_video_request
):
    """Test generic APIStatusError (e.g. 500) maps to HTTPError."""
    from runwayml import APIStatusError

    ex = APIStatusError.__new__(APIStatusError)
    ex.status_code = 500
    ex.message = "Internal server error"
    ex.body = {"error": "Internal server error"}
    ex.args = ("Internal server error",)

    result = handler._handle_error(base_config, text_to_video_request, "test-id", ex)

    assert isinstance(result, HTTPError)
    assert result.status_code == 500


# ==================== Helper Function Tests ====================


def test_extract_video_url_from_string():
    """Test _extract_video_url with string output."""
    assert (
        _extract_video_url("https://example.com/video.mp4")
        == "https://example.com/video.mp4"
    )


def test_extract_video_url_from_list():
    """Test _extract_video_url with list output."""
    assert (
        _extract_video_url(["https://example.com/video.mp4"])
        == "https://example.com/video.mp4"
    )


def test_extract_video_url_from_empty_list():
    """Test _extract_video_url with empty list returns None."""
    assert _extract_video_url([]) is None


def test_extract_video_url_from_dict():
    """Test _extract_video_url with dict output."""
    assert (
        _extract_video_url({"url": "https://example.com/video.mp4"})
        == "https://example.com/video.mp4"
    )


def test_extract_video_url_from_dict_no_url():
    """Test _extract_video_url with dict without url key returns None."""
    assert _extract_video_url({"other": "data"}) is None


def test_extract_video_url_from_unknown_type():
    """Test _extract_video_url with unknown type returns None."""
    assert _extract_video_url(12345) is None


def test_convert_media_to_file_string():
    """Test _convert_media_to_file with string URL."""
    result = _convert_media_to_file("https://example.com/image.jpg", "image")
    assert result == "https://example.com/image.jpg"


def test_convert_media_to_file_dict_bytes():
    """Test _convert_media_to_file with MediaContent dict."""
    import io

    media = {"content": b"fake-image-data", "content_type": "image/jpeg"}
    result = _convert_media_to_file(media, "prompt_image")

    assert isinstance(result, io.BytesIO)
    assert result.name == "prompt_image.jpeg"
    assert result.read() == b"fake-image-data"


def test_convert_media_to_file_other_type():
    """Test _convert_media_to_file with HttpUrl-like type falls through to str()."""
    from unittest.mock import MagicMock

    mock_url = MagicMock()
    mock_url.__str__ = lambda self: "https://example.com/fallback.jpg"

    result = _convert_media_to_file(mock_url, "image")
    assert result == "https://example.com/fallback.jpg"


# ==================== Unknown Model Tests ====================


def test_get_endpoint_unknown_model_rejects_video():
    """Test unknown model rejects video input."""
    with pytest.raises(ValidationError, match="does not support video input"):
        _get_endpoint_from_model("unknown-model-v1", False, True)


def test_get_endpoint_unknown_model_image_to_video():
    """Test unknown model with image input routes to image_to_video."""
    assert _get_endpoint_from_model("unknown-model-v1", True, False) == "image_to_video"


def test_get_endpoint_unknown_model_text_to_video():
    """Test unknown model without inputs routes to text_to_video."""
    assert _get_endpoint_from_model("unknown-model-v1", False, False) == "text_to_video"


# ==================== Request Conversion: audio flag, seed, content_moderation ====================


def test_convert_request_text_to_video_with_audio(handler, base_config):
    """Test audio flag is passed for text-to-video endpoint."""
    request = VideoGenerationRequest(prompt="Test", generate_audio=True)
    endpoint, params = handler._convert_request(base_config, request)

    assert endpoint == "text_to_video"
    assert params["audio"] is True


def test_convert_request_image_to_video_no_reference_images_error(handler):
    """Test image-to-video endpoint raises when no reference image found."""
    config = VideoGenerationConfig(
        model="gen4_turbo",
        provider="runway",
        api_key="test-api-key",
        timeout=600,
    )
    request = VideoGenerationRequest(
        prompt="Test",
        image_list=[{"type": "last_frame", "image": "https://example.com/img.jpg"}],
    )
    with pytest.raises(ValidationError, match="No reference image found"):
        handler._convert_request(config, request)


def test_convert_request_v2v_with_reference_images(handler):
    """Test v2v endpoint passes reference images."""
    config = VideoGenerationConfig(
        model="gen4_aleph",
        provider="runway",
        api_key="test-api-key",
        timeout=600,
    )
    request = VideoGenerationRequest(
        prompt="Add elements",
        video="https://example.com/video.mp4",
        image_list=[
            {"type": "reference", "image": "https://example.com/ref.jpg"},
        ],
        aspect_ratio="16:9",
    )
    endpoint, params = handler._convert_request(config, request)

    assert endpoint == "video_to_video"
    assert params["references"] == [
        {"type": "image", "uri": "https://example.com/ref.jpg"}
    ]


def test_convert_request_v2v_seed_and_content_moderation(handler):
    """Test v2v endpoint passes seed and content_moderation."""
    config = VideoGenerationConfig(
        model="gen4_aleph",
        provider="runway",
        api_key="test-api-key",
        timeout=600,
    )
    request = VideoGenerationRequest(
        prompt="Test",
        video="https://example.com/video.mp4",
        seed=99,
        extra_params={"content_moderation": {"threshold": "low"}},
    )
    endpoint, params = handler._convert_request(config, request)

    assert endpoint == "video_to_video"
    assert params["seed"] == 99
    assert params["content_moderation"] == {"threshold": "low"}


# ==================== Image Generation: NotImplementedError ====================


@pytest.mark.asyncio
async def test_generate_image_async_not_implemented(handler):
    """Test async image generation raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="does not support image generation"):
        await handler.generate_image_async(None, None)


def test_generate_image_sync_not_implemented(handler):
    """Test sync image generation raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="does not support image generation"):
        handler.generate_image(None, None)
